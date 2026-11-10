#!/usr/bin/env python3
"""Contracts for independent outcome verification and bounded process accounting."""
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

spec = importlib.util.spec_from_file_location('measure_web_efficiency', pathlib.Path(__file__).with_name('measure-web-efficiency.py'))
measurement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measurement)


class MemoryQualificationContract(unittest.TestCase):
    def test_vps_incident_counts_do_not_blame_verified_custody_or_qualify_partial_pss(self):
        # Actual VPS shapes:4/50 and18/209 incomplete, versus53 complete samples.
        for samples, incomplete in ((50, 4), (209, 18), (53, 0)):
            with self.subTest(samples=samples, incomplete=incomplete):
                peak = 472257536
                measured = {
                    'process_custody': {'supported': True, 'all_reaped': True,
                                        'runtime_image_observed': True},
                    'process_family_memory': {
                        'supported': True, 'sample_count': samples,
                        'incomplete_sample_count': incomplete,
                        'peak_sampled_pss_bytes': peak, 'peak_sampled_private_bytes': 410910720},
                }
                measurement.qualify_cli_memory(measured)
                memory = measured['process_family_memory']
                qualified = measurement.measurement_qualification(measured, True, True)
                self.assertEqual(memory['complete_for_execution_surface'], incomplete == 0)
                self.assertEqual(memory['qualification_errors'],
                                 ['incomplete_memory_samples'] if incomplete else [])
                self.assertEqual(memory['peak_sampled_pss_bytes'], None if incomplete else peak)
                self.assertEqual(measured['observed_process_family_memory']['peak_sampled_pss_bytes'], peak)
                self.assertTrue(qualified['process_wall_time_verified'])
                self.assertEqual(qualified['sampled_process_family_memory_verified'], incomplete == 0)
                self.assertNotIn('not verified', memory['scope'])
                rejected = measurement.measurement_qualification(measured, False, True)
                self.assertFalse(rejected['process_wall_time_verified'])
                self.assertFalse(rejected['sampled_process_family_memory_verified'])


class HostEnvelopeContract(unittest.TestCase):
    def fixture(self, directory):
        proc_self = directory / 'proc-self'
        proc_self.mkdir()
        mount = directory / 'mounted hierarchy'
        leaf = mount / 'parent/leaf'
        leaf.mkdir(parents=True)
        (proc_self / 'cgroup').write_text('0::/tenant/parent/leaf\n')
        escaped = str(mount).replace('\\', '\\134').replace(' ', '\\040')
        (proc_self / 'mountinfo').write_text(
            f'24 23 0:28 /tenant {escaped} rw - cgroup2 cgroup rw\n')
        for group, cpu, memory, swap in (
                (leaf, 'max 100000', 'max', 'max'),
                (leaf.parent, '50000 100000', '1024', '40'),
                (mount, '200000 100000', '4096', '400')):
            for name, value in (('cpu.max', cpu), ('memory.max', memory), ('memory.swap.max', swap)):
                (group / name).write_text(value)
        return proc_self, mount, leaf

    def test_visible_ancestor_ceilings_cannot_be_relaxed_by_leaf(self):
        with tempfile.TemporaryDirectory() as directory:
            proc_self, mount, leaf = self.fixture(pathlib.Path(directory))
            observed = measurement.cgroup_v2_envelope(proc_self)
            self.assertEqual(observed['errors'], [])
            self.assertEqual(observed['leaf']['cpu.max'], 'max 100000')
            self.assertEqual(
                observed['cpu_quota_upper_bound_ratio'], [1, 2],
                'leaf-only CPU observation loses the tighter ancestor ceiling')
            self.assertEqual(observed['memory_max_upper_bound_bytes'], 1024)
            self.assertEqual(observed['swap_max_upper_bound_bytes'], 40)
            self.assertEqual(
                [row['path'] for row in observed['visible_ancestors']],
                [str(leaf), str(leaf.parent), str(mount)])

    def test_invalid_or_unmapped_metadata_is_explicitly_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            proc_self, mount, leaf = self.fixture(pathlib.Path(directory))
            (leaf.parent / 'cpu.max').write_text('10000 0')
            observed = measurement.cgroup_v2_envelope(proc_self)
            self.assertTrue(observed['errors'], 'invalid CPU metadata must remain visible')
            (proc_self / 'cgroup').write_text('0::/different-namespace\n')
            observed = measurement.cgroup_v2_envelope(proc_self)
            self.assertIsNone(observed['leaf'])
            self.assertIsNone(observed['cpu_quota_upper_bound'])
            self.assertTrue(observed['errors'], 'unmapped cgroup must not be called unlimited')

    @unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('taskset'),
                         'requires actual Linux CPU affinity')
    def test_real_single_cpu_affinity_bounds_the_observed_capacity(self):
        cpu = min(os.sched_getaffinity(0))
        code = (
            "import importlib.util,json,pathlib;"
            f"s=importlib.util.spec_from_file_location('meter',{str(pathlib.Path(measurement.__file__))!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "print(json.dumps(m.host_envelope()))")
        child = subprocess.run(
            ['taskset', '-c', str(cpu), sys.executable, '-c', code],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(child.returncode, 0, child.stderr)
        observed = json.loads(child.stdout)
        self.assertEqual(observed['cpu_affinity'], [cpu])
        self.assertLessEqual(observed['sustained_cpu_capacity_upper_bound'], 1)
        self.assertFalse(observed['limits_modified_by_harness'])


class OutcomeContract(unittest.TestCase):
    def setUp(self):
        self.fixture = measurement.Fixture({'rounds': 1, 'tasks': 1, 'concurrency': 1, 'nodes': 3})
        self.thread = threading.Thread(target=self.fixture.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.fixture.server_port}'

    def tearDown(self):
        self.fixture.shutdown()
        self.fixture.server_close()
        self.thread.join(timeout=5)

    def post(self, path, body):
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=body, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status
        except urllib.error.HTTPError as error:
            with error:
                return error.code

    def completed_receipt(self):
        row = {'round': 0, 'duration_ms': 50, 'succeeded': 1, 'attempted': 1,
               'tasks': [{'round': 0, 'worker': 0, 'task': 0, 'ok': True, 'error': None, 'duration_ms': 50}]}
        self.assertEqual(self.post('/round', row), 200)
        self.assertEqual(self.post('/result', {'schema': 'greppy.web-efficiency.workload.v1', 'rounds': [row]}), 200)

    def submit(self):
        return self.post('/submit', urllib.parse.urlencode({'q': self.fixture.token(0, 0, 0)}).encode())

    def test_claimed_success_without_post_effect_never_counts(self):
        self.completed_receipt()
        audit = self.fixture.audit()
        self.assertFalse(audit['all_tasks_verified'])
        self.assertEqual(audit['independently_verified_completed_tasks'], 0)

    def test_actual_utf8_post_effect_and_complete_receipt_qualify(self):
        self.assertEqual(self.submit(), 200)
        self.completed_receipt()
        audit = self.fixture.audit()
        self.assertTrue(audit['all_tasks_verified'], audit)
        self.assertEqual(audit['independently_verified_completed_tasks'], 1)
        self.assertEqual(audit['verified_task_durations_ms'], [50])

    def test_duplicate_side_effect_disqualifies_run(self):
        self.assertEqual(self.submit(), 200)
        self.assertEqual(self.submit(), 400)
        self.completed_receipt()
        self.assertFalse(self.fixture.audit()['all_tasks_verified'])

    def test_malformed_round_does_not_become_success_or_crash_audit(self):
        self.assertEqual(self.post('/result', {'schema': 'greppy.web-efficiency.workload.v1', 'rounds': [None]}), 200)
        audit = self.fixture.audit()
        self.assertFalse(audit['all_tasks_verified'])
        self.assertEqual(audit['independently_verified_completed_tasks'], 0)


class NativeEntryContract(unittest.TestCase):
    def test_entry_parses_with_only_the_native_playwright_import(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = pathlib.Path(directory) / 'entry.mjs'
            provenance = measurement.prepare_native_entry(entry)
            second = pathlib.Path(directory) / 'second.mjs'
            measurement.prepare_native_entry(second)
            self.assertEqual(entry.read_bytes(), second.read_bytes())

            self.assertEqual(provenance['sha256'], measurement.digest(entry))
            self.assertEqual(provenance['adapter_sha256'], measurement.digest(pathlib.Path(provenance['adapter_path'])))
            self.assertEqual(provenance['workload_sha256'], measurement.digest(pathlib.Path(provenance['workload_path'])))
            result = subprocess.run(['node', '--experimental-vm-modules',
                                     str(measurement.ROOT / 'scripts/check-efficiency-native-entry.mjs'),
                                     str(entry), str(measurement.ROOT / 'runtime/js/playwright.mjs')],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            checked = json.loads(result.stdout)
            self.assertEqual(checked['dependencies'], ['playwright'])
            self.assertEqual(checked['launch_policy'], provenance['native_launch_policy']['version'])
            self.assertEqual(checked['wrapper_sha256'], measurement.digest(measurement.ROOT / 'runtime/js/playwright.mjs'))
            self.assertIn(pathlib.Path(provenance['workload_path']).read_bytes(), entry.read_bytes())

    def test_native_argv_uses_hashed_entry_and_forged_success_without_post_still_fails(self):
        # A real child probes argv and posts fabricated receipts. It never executes a
        # browser; exit0 and forged ok must still fail the independent HTTP oracle.
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            probe = directory / 'receipt-probe'
            probe.write_text(f'#!{sys.executable}\n' + '''import argparse, hashlib, json, pathlib, urllib.request
p = argparse.ArgumentParser()
p.add_argument('--script', type=pathlib.Path, required=True)
p.add_argument('--fixture-url', required=True)
a = p.parse_args()
print(json.dumps({'entry': str(a.script), 'sha256': hashlib.sha256(a.script.read_bytes()).hexdigest()}))
row = {'round': 0, 'duration_ms': 1, 'succeeded': 1, 'attempted': 1,
       'tasks': [{'round': 0, 'worker': 0, 'task': 0, 'ok': True, 'error': None, 'duration_ms': 1}]}
for path, value in [('round', row), ('result', {'schema': 'greppy.web-efficiency.workload.v1', 'rounds': [row]})]:
    request = urllib.request.Request(a.fixture_url + path, data=json.dumps(value).encode(), method='POST')
    with urllib.request.urlopen(request, timeout=3) as response:
        assert response.status == 200
''')
            probe.chmod(0o700)
            output = directory / 'native.json'
            result = subprocess.run([sys.executable, str(pathlib.Path(measurement.__file__)),
                                     '--engine', 'native', '--runtime', str(probe), '--output', str(output),
                                     '--rounds', '1', '--tasks', '1', '--concurrency', '1', '--nodes', '3', '--timeout', '3'],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1, result.stderr)
            receipt = json.loads(output.read_text())
            self.assertFalse(receipt['quality_verified'])
            self.assertEqual(receipt['metrics']['exit_code'], 0, json.dumps(receipt['metrics'], ensure_ascii=False))
            self.assertEqual(receipt['audit']['independently_verified_completed_tasks'], 0)
            self.assertEqual(receipt['server_evidence']['observed_effects'], 0)
            self.assertEqual(receipt['server_evidence']['expected_effects'], 1)
            self.assertEqual(receipt['quality_error'], 'task has no independent server-side POST effect')
            invoked = json.loads(receipt['metrics']['stdout_tail'])
            entry = receipt['native_entry_provenance']
            self.assertEqual(invoked, {'entry': entry['path'], 'sha256': entry['sha256']})
            self.assertEqual(entry['sha256'], measurement.digest(pathlib.Path(entry['path'])))
            self.assertNotEqual(entry['path'], str(measurement.ROOT / 'runtime/fixtures/efficiency-native.mjs'))
            self.assertEqual(receipt['workload_sha256'], entry['workload_sha256'])


class CliBoundaryContract(unittest.TestCase):
    def test_hashed_cli_entry_keeps_independent_oracle_and_runtime_identity(self):
        # These adversarial child processes do not implement or start a browser.
        # They verify that CLI exit0/claimed success cannot qualify missing work,
        # and that even genuine POST effects cannot hide changed runtime bytes.
        state_parent = pathlib.Path(os.environ.get('GREPPY_TEST_SHORT_ROOT', tempfile.gettempdir()))
        if not state_parent.is_absolute() or len(os.fsencode(state_parent)) > 20:
            self.skipTest('set GREPPY_TEST_SHORT_ROOT to an absolute existing parent <=20 bytes')
        for changed_runtime in (False, True):
            with self.subTest(changed_runtime=changed_runtime), tempfile.TemporaryDirectory() as directory:
                directory = pathlib.Path(directory)
                runtime = directory / 'runtime'
                runtime.write_bytes(b'original runtime identity')
                probe = directory / 'cli-boundary-probe'
                probe.write_text(f'#!{sys.executable}\n' + """
import hashlib, json, os, pathlib, sys, urllib.parse, urllib.request
if sys.argv[1:] == ['web', 'runtime', 'stop', '--json']:
    # This adversarial probe creates no runtime; its owned cleanup is a no-op.
    raise SystemExit(0)
assert sys.argv[1:10] == ['web', 'do', '--json', 'session', 'new', '--profile', 'project', '::', 'run']
assert sys.argv[10:13] == ['--mode', 'standalone', '--script-file']
entry = pathlib.Path(sys.argv[13])
assert sys.argv[14:16] == ['--timeout', '3000']
state = pathlib.Path(os.environ['GREPPY_WEB_RUNTIME_DIR'])
assert pathlib.Path.cwd().samefile(state / 'workspace')
assert os.environ['GREPPY_RUNTIME_DIR'] == str(state)
assert 'GREPPY_WEB_SESSION' not in os.environ and 'GREPPY_RUN_ID' not in os.environ
runtime = pathlib.Path(os.environ['GREPPY_WEB_RUNTIME'])
url = json.loads(entry.read_text().splitlines()[0].removeprefix('const fixtureUrl = ').removesuffix(';'))
def post(path, body):
    request = urllib.request.Request(url + path, data=body, method='POST')
    with urllib.request.urlopen(request, timeout=3) as response:
        assert response.status == 200
""" + ("""
with urllib.request.urlopen(url, timeout=3) as response:
    import html, re
    config = json.loads(html.unescape(re.search(r'<pre id="workload-config">(.*?)</pre>', response.read().decode()).group(1)))
token = config['nonce'] + ':r0:w0:t0:é&+'
post('submit', urllib.parse.urlencode({'q': token}).encode())
runtime.write_bytes(b'changed runtime identity')
""" if changed_runtime else "") + """
row = {'round': 0, 'duration_ms': 1, 'succeeded': 1, 'attempted': 1,
       'tasks': [{'round': 0, 'worker': 0, 'task': 0, 'ok': True, 'error': None, 'duration_ms': 1}]}
post('round', json.dumps(row).encode())
post('result', json.dumps({'schema': 'greppy.web-efficiency.workload.v1', 'rounds': [row]}).encode())
print(json.dumps({'entry_sha256': hashlib.sha256(entry.read_bytes()).hexdigest()}))
""")
                probe.chmod(0o700)
                output = directory / 'cli.json'
                environment = dict(os.environ, GREPPY_WEB_SESSION='foreign-session', GREPPY_RUN_ID='foreign-run')
                result = subprocess.run(
                    [sys.executable, str(pathlib.Path(measurement.__file__)),
                     '--engine', 'cli', '--cli', str(probe), '--runtime', str(runtime),
                     '--state-parent', str(state_parent), '--output', str(output),
                     '--rounds', '1', '--tasks', '1', '--concurrency', '1', '--nodes', '3', '--timeout', '3'],
                    env=environment, capture_output=True, text=True, timeout=10)
                self.assertTrue(output.is_file(), result.stderr)
                receipt = json.loads(output.read_text())
                state = pathlib.Path(receipt['cli_provenance']['private_state_directory'])
                try:
                    self.assertEqual(
                        result.returncode, 1,
                        'changed runtime must fail even with verified POST effects' if changed_runtime else result.stderr)
                    self.assertEqual(receipt['metrics']['exit_code'], 0, receipt['metrics'])
                    self.assertEqual(receipt['metrics']['cleanup_exit_code'], 0)
                    self.assertFalse(
                        receipt['quality_verified'],
                        'changed runtime must fail even with verified POST effects' if changed_runtime
                        else 'forged success must fail without an independent POST effect')
                    expected = ('CLI runtime executable changed during measurement' if changed_runtime else
                                'task has no independent server-side POST effect')
                    self.assertEqual(receipt['quality_error'], expected)
                    self.assertEqual(receipt['audit']['independently_verified_completed_tasks'], int(changed_runtime))
                    self.assertEqual(receipt['cli_provenance']['sha256'], measurement.digest(probe))
                    memory = receipt['metrics']['process_family_memory']
                    self.assertIs(memory.get('complete_for_execution_surface'), False,
                                  'unregistered CLI runtime owner cannot establish whole-worker memory')
                    self.assertIsNone(memory['peak_sampled_pss_bytes'])
                    self.assertIsNone(memory['peak_sampled_private_bytes'])
                    self.assertIn('observed_process_family_memory', receipt['metrics'])
                    self.assertEqual(json.loads(receipt['metrics']['stdout_tail'])['entry_sha256'],
                                     receipt['native_entry_provenance']['sha256'])
                    checked = subprocess.run(
                        ['node', '--experimental-vm-modules',
                         str(measurement.ROOT / 'scripts/check-efficiency-native-entry.mjs'),
                         receipt['native_entry_provenance']['path'],
                         str(measurement.ROOT / 'runtime/js/playwright.mjs')],
                        capture_output=True, text=True, timeout=15)
                    self.assertEqual(checked.returncode, 0, checked.stderr)
                finally:
                    # The source of this completed probe starts no descendants.
                    shutil.rmtree(state)


class ProcessContract(unittest.TestCase):
    def wait_for_bounded_detached_helper(self, receipt):
        deadline = time.monotonic() + 12
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not receipt.exists():
            return
        record = json.loads(receipt.read_text())
        path = pathlib.Path('/proc') / str(record['pid']) / 'stat'
        while time.monotonic() < deadline:
            try:
                fields = path.read_text().rsplit(')', 1)[1].split()
            except FileNotFoundError:
                return
            if int(fields[19]) != record['starttime_ticks'] or fields[0] == 'Z':
                return
            time.sleep(0.02)
        self.fail('bounded owned helper did not reach its self-exit deadline')

    def detached_source(self, ignore_term=False):
        return r"""
import json, os, pathlib, signal, sys, time
if os.fork():
    os._exit(0)
os.setsid()
if IGNORE_TERM:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
private = bytearray(24 * 1024 * 1024)
for index in range(0, len(private), 4096):
    private[index] = 1
fields = pathlib.Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
record = {'pid': os.getpid(), 'starttime_ticks': int(fields[19])}
receipt = pathlib.Path(sys.argv[1])
stage = receipt.with_suffix('.stage')
stage.write_text(json.dumps(record))
os.replace(stage, receipt)
wall = time.monotonic()
cpu = time.process_time()
if IGNORE_TERM:
    while time.monotonic() - wall < 9:
        time.sleep(0.05)
else:
    while time.process_time() - cpu < 0.2 and time.monotonic() - wall < 2:
        sum(range(1000))
    time.sleep(0.65)
record['cpu_seconds'] = time.process_time() - cpu
stage = receipt.with_suffix('.stage')
stage.write_text(json.dumps(record))
os.replace(stage, receipt)
os._exit(0)
""".replace('IGNORE_TERM', repr(ignore_term))

    @unittest.skipUnless(sys.platform.startswith('linux'), 'actual Linux subreaper custody')
    def test_detached_family_cost_and_exit_are_not_lost_after_cli_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            receipt = directory / 'owned-detached.json'
            try:
                runtime = pathlib.Path(sys.executable).stat()
                measured = measurement.execute(
                    [sys.executable, '-c', self.detached_source(), str(receipt)],
                    directory, 5, runtime_identity=(runtime.st_dev, runtime.st_ino))
                self.assertGreaterEqual(
                    measured['process_cold_total_ms'], 550,
                    'detached child lifetime was omitted after original CLI exit')
                family = measured['process_custody']
                self.assertTrue(family['supported'], family)
                self.assertTrue(family['all_reaped'], family)
                self.assertTrue(family['runtime_image_observed'], family)
                self.assertFalse(measured['timed_out'])
                observed = measured['process_family_memory']
                self.assertGreaterEqual(
                    observed['peak_sampled_private_bytes'] or 0, 16 * 1024 * 1024,
                    'detached child private mappings were omitted')
                record = json.loads(receipt.read_text())
                self.assertGreaterEqual(
                    measured['user_cpu_seconds'] + measured['system_cpu_seconds'],
                    record['cpu_seconds'] * 0.8,
                    'adopted child wait4 CPU was omitted')
                identity = measurement._memory_module._stat(record['pid'])
                self.assertTrue(identity is None or identity[1] != record['starttime_ticks'],
                                'owned detached identity survived a completed measurement')
            finally:
                self.wait_for_bounded_detached_helper(receipt)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'actual Linux orphan cleanup')
    def test_timeout_reaps_detached_term_ignoring_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            receipt = directory / 'owned-timeout.json'
            try:
                measured = measurement.execute(
                    [sys.executable, '-c', self.detached_source(ignore_term=True), str(receipt)],
                    directory, 0.3)
                self.assertTrue(measured['timed_out'], 'detached timeout was called successful')
                self.assertTrue(measured['process_custody']['all_reaped'])
                self.assertLess(measured['process_cold_total_ms'], 8000,
                                'TERM-ignoring detached child escaped forced-cleanup deadline')
                record = json.loads(receipt.read_text())
                identity = measurement._memory_module._stat(record['pid'])
                self.assertTrue(identity is None or identity[1] != record['starttime_ticks'],
                                'timed-out detached identity survived cleanup')
            finally:
                self.wait_for_bounded_detached_helper(receipt)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'actual Linux existing-child isolation')
    def test_custody_refuses_to_adopt_a_preexisting_child(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)'])
        try:
            custody = measurement._memory_module.ProcessFamilyCustody()
            try:
                self.assertFalse(custody.supported,
                                 'custody claimed a child that predated the measured workload')
                self.assertIn('pre-existing children', custody.reason)
            finally:
                custody.close()
            self.assertIsNone(child.poll(), 'custody stopped an unrelated pre-existing child')
        finally:
            child.terminate()
            child.wait(timeout=5)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux process-family PSS integration')
    def test_execute_receipt_accounts_for_a_real_parent_and_child(self):
        child_source = '''import sys, time
private = bytearray(18 * 1024 * 1024)
for i in range(0, len(private), 4096): private[i] = 1
print('ready', flush=True)
time.sleep(0.9)
'''
        parent_source = '''import json, os, subprocess, sys
private = bytearray(18 * 1024 * 1024)
for i in range(0, len(private), 4096): private[i] = 1
child = subprocess.Popen([sys.executable, '-c', sys.argv[1]], stdout=subprocess.PIPE)
try:
    assert child.stdout.readline() == b'ready\\n'
    print(json.dumps({'parent': os.getpid(), 'child': child.pid}), flush=True)
    assert child.wait(timeout=5) == 0
finally:
    if child.poll() is None:
        child.kill()
        child.wait(timeout=5)
'''
        with tempfile.TemporaryDirectory() as directory:
            result = measurement.execute([sys.executable, '-c', parent_source, child_source], pathlib.Path(directory), 8)
            memory = result['process_family_memory']
            samples_path = pathlib.Path(memory['samples_path'])
            samples = [json.loads(row) for row in samples_path.read_text().splitlines()]
            self.assertEqual(memory['samples_sha256'], measurement.digest(samples_path))
        self.assertEqual(result['exit_code'], 0, result)
        pids = set(json.loads(result['stdout_tail']).values())
        self.assertTrue(any({p['pid'] for p in s['processes']} == pids and (s['private_bytes'] or 0) >= 36 * 1024 * 1024 for s in samples), f'owned parent/child mappings were not sampled: {samples}')
        self.assertGreaterEqual(memory['peak_sampled_private_bytes'], 36 * 1024 * 1024)
        self.assertGreater(memory['monitor_thread_cpu_seconds'], 0)

    def test_failed_process_retains_exit_cost_and_bounded_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            result = measurement.execute([sys.executable, '-c', "print('failure evidence'); raise SystemExit(7)"], pathlib.Path(directory), 3)
        self.assertEqual(result['exit_code'], 7)
        self.assertFalse(result['timed_out'])
        self.assertIn('failure evidence', result['stdout_tail'])
        self.assertGreater(result['process_cold_total_ms'], 0)
        self.assertGreater(result['max_individual_process_rss_bytes'], 0)

    def test_hung_owned_process_is_terminated_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            result = measurement.execute([sys.executable, '-c', 'import time; time.sleep(30)'], pathlib.Path(directory), .1)
        self.assertTrue(result['timed_out'])
        self.assertNotEqual(result['exit_code'], 0)
        self.assertLess(result['process_cold_total_ms'], 3000)


if __name__ == '__main__':
    unittest.main()
