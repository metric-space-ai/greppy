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
import unittest
import urllib.error
import urllib.parse
import urllib.request

spec = importlib.util.spec_from_file_location('measure_web_efficiency', pathlib.Path(__file__).with_name('measure-web-efficiency.py'))
measurement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measurement)


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
            self.assertEqual(receipt['metrics']['exit_code'], 0)
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
assert sys.argv[1:10] == ['web', 'do', '--json', 'session', 'new', '--profile', 'project', '::', 'run']
assert sys.argv[10:13] == ['--mode', 'standalone', '--script-file']
entry = pathlib.Path(sys.argv[13])
assert sys.argv[14:16] == ['--timeout', '3000']
state = pathlib.Path(os.environ['GREPPY_WEB_RUNTIME_DIR'])
assert pathlib.Path.cwd() == state / 'workspace'
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
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertEqual(receipt['metrics']['exit_code'], 0, receipt['metrics'])
                    self.assertFalse(receipt['quality_verified'])
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
        self.assertTrue(any({p['pid'] for p in s['processes']} == pids and (s['private_bytes'] or 0) >= 36 * 1024 * 1024 for s in samples), samples)
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
