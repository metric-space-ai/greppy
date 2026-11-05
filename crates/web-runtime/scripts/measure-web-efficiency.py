#!/usr/bin/env python3
"""Bounded, opt-in comparison of one identical complete browser workload.

Requires existing binaries/packages. Never installs, builds, changes host limits,
or claims that a standalone runtime measurement covers an LLM agent or CLI daemon.
"""
import argparse
from collections import Counter
import hashlib
import html
import http.server
import json
import os
import pathlib
import platform
import resource
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parents[1]


def digest(path):
    with pathlib.Path(path).open('rb') as source:
        hashed = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b''):
            hashed.update(block)
        return hashed.hexdigest()


def source_identity():
    def git(*args):
        result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    return {'harness_commit': git('rev-parse', 'HEAD'),
            'harness_source_status': git('status', '--porcelain', '--', 'scripts', 'runtime/fixtures/efficiency-workload.mjs',
                                         'runtime/fixtures/efficiency-native.mjs')}


def host_envelope():
    result = {'platform': platform.platform(), 'logical_cpus': os.cpu_count(),
              'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
              'limits_modified_by_harness': False, 'cgroup_v2': None}
    try:
        entries = pathlib.Path('/proc/self/cgroup').read_text().splitlines()
        relative = next(row[3:] for row in entries if row.startswith('0::'))
        if '..' not in pathlib.PurePosixPath(relative).parts:
            group = pathlib.Path('/sys/fs/cgroup') / relative.lstrip('/')
            result['cgroup_v2'] = {name: (group / name).read_text().strip()
                                   for name in ('cpu.max', 'memory.max', 'memory.swap.max')}
    except (OSError, StopIteration):
        pass
    return result


class Fixture(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, config):
        self.config = dict(config, nonce=secrets.token_hex(12))
        self.lock = threading.Lock()
        self.effects = set()
        self.violations = []
        self.receipts = []
        self.round_events = []
        self.started_at = None
        self.request_body_bytes = 0
        self.response_body_bytes = 0
        self.requests = 0
        self.expected = {self.token(r, w, t)
                         for r in range(config['rounds'])
                         for w in range(config['concurrency'])
                         for t in range(config['tasks'])}
        super().__init__(('127.0.0.1', 0), Handler)

    def token(self, r, w, t):
        return f"{self.config['nonce']}:r{r}:w{w}:t{t}:é&+"

    def verify(self):
        with self.lock:
            if len(self.receipts) != 1:
                return False, 'exactly one completed workload receipt is required'
            receipt = self.receipts[0]
            if receipt.get('schema') != 'greppy.web-efficiency.workload.v1':
                return False, 'invalid workload schema'
            rounds = receipt.get('rounds')
            if not isinstance(rounds, list) or len(rounds) != self.config['rounds']:
                return False, 'missing rounds'
            if [event[1] for event in self.round_events] != rounds:
                return False, 'round completion receipts disagree with final workload'
            seen = set()
            for r, row in enumerate(rounds):
                if not isinstance(row, dict):
                    return False, 'invalid round'
                tasks = row.get('tasks', [])
                if (not isinstance(tasks, list) or row.get('round') != r
                        or len(tasks) != self.config['concurrency'] * self.config['tasks']):
                    return False, 'missing or duplicated tasks'
                for task in tasks:
                    if not isinstance(task, dict):
                        return False, 'invalid task'
                    identity = (task.get('round'), task.get('worker'), task.get('task'))
                    if (identity[0] != r or type(identity[1]) is not int or type(identity[2]) is not int
                            or not 0 <= identity[1] < self.config['concurrency']
                            or not 0 <= identity[2] < self.config['tasks'] or identity in seen):
                        return False, 'invalid task identity'
                    seen.add(identity)
                    if task.get('ok') is not True or task.get('error') is not None:
                        return False, 'workload reported an unsuccessful task'
                    if type(task.get('duration_ms')) is not int or task['duration_ms'] < 0:
                        return False, 'invalid task timing'
                    if self.token(*identity) not in self.effects:
                        return False, 'task has no independent server-side POST effect'
                if row.get('succeeded') != len(tasks) or row.get('attempted') != len(tasks):
                    return False, 'success count disagrees with complete tasks'
            if self.violations or self.effects != self.expected:
                return False, 'missing, invalid or duplicate POST effects'
            return True, None

    def audit(self):
        valid, reason = self.verify()
        with self.lock:
            rounds = self.receipts[0].get('rounds', []) if len(self.receipts) == 1 else []
            if not isinstance(rounds, list):
                rounds = []
            reported = [task for row in rounds if isinstance(row, dict) and isinstance(row.get('tasks'), list)
                        for task in row['tasks'] if isinstance(task, dict)]
            identities = []
            for task in reported:
                identity = tuple(task.get(key) for key in ('round', 'worker', 'task'))
                if all(type(value) is int for value in identity):
                    identities.append(identity)
            counts = Counter(identities)
            verified = []
            for task in reported:
                identity = tuple(task.get(key) for key in ('round', 'worker', 'task'))
                if not all(type(value) is int for value in identity):
                    continue
                effect_verified = self.token(*identity) in self.effects
                if (counts[identity] == 1 and task.get('ok') is True and task.get('error') is None
                        and type(task.get('duration_ms')) is int and task['duration_ms'] >= 0 and effect_verified):
                    verified.append(task)
            return {'all_tasks_verified': valid, 'error': reason, 'scheduled_tasks': len(self.expected),
                    'reported_attempts': len(reported) if self.receipts else None,
                    'independently_verified_completed_tasks': len(verified),
                    'verified_task_durations_ms': [task.get('duration_ms') for task in verified],
                    'reported_failed_attempts': [task for task in reported if task.get('ok') is not True]}


class Handler(http.server.BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *_):
        pass

    def reply(self, status, body='', kind='text/html; charset=utf-8'):
        body = body.encode('utf-8')
        with self.server.lock:
            self.server.requests += 1
            self.server.response_body_bytes += len(body)
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == '/':
            config = html.escape(json.dumps(self.server.config, ensure_ascii=False))
            return self.reply(200, f'<!doctype html><title>Efficiency start</title>'
                              f'<pre id="workload-config">{config}</pre><a id="article" href="/article">Read article</a>')
        if path == '/article':
            nodes = self.server.config['nodes']
            paragraphs = ''.join(f'<p>payload-{i}</p>' for i in range(nodes))
            return self.reply(200, f'<!doctype html><title>Efficiency article</title><main>{paragraphs}'
                              f'<p>end:{nodes}</p></main><form method="post" action="/submit">'
                              '<label for="query">Query</label><input id="query" name="q">'
                              '<button id="submit" type="submit">Submit</button></form>')
        return self.reply(404, 'unknown fixture route')

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 1024 * 1024:
                return self.reply(413, 'invalid body length')
            body = self.rfile.read(length)
            with self.server.lock:
                self.server.request_body_bytes += len(body)
            path = urllib.parse.urlsplit(self.path).path
            if path == '/submit':
                token = urllib.parse.parse_qs(body.decode('utf-8')).get('q', [None])[0]
                with self.server.lock:
                    invalid = token not in self.server.expected or token in self.server.effects
                    if invalid:
                        self.server.violations.append('invalid or duplicate submission')
                    else:
                        self.server.effects.add(token)
                if invalid:
                    return self.reply(400, 'invalid or duplicate submission')
                return self.reply(200, f'<!doctype html><title>Efficiency receipt</title><p id="receipt">{html.escape(token)}</p>')
            if path == '/result':
                receipt = json.loads(body)
                if not isinstance(receipt, dict):
                    return self.reply(400, 'invalid receipt')
                with self.server.lock:
                    self.server.receipts.append(receipt)
                return self.reply(200, '{}', 'application/json')
            if path == '/round':
                receipt = json.loads(body)
                if not isinstance(receipt, dict):
                    return self.reply(400, 'invalid round receipt')
                with self.server.lock:
                    self.server.round_events.append((time.monotonic(), receipt))
                return self.reply(200, '{}', 'application/json')
            return self.reply(404, 'unknown fixture route')
        except (ValueError, UnicodeError):
            return self.reply(400, 'invalid POST')


def log_tail(path):
    with path.open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 8192))
        return stream.read(8192).decode('utf-8', 'replace')


def execute(command, directory, timeout, started=None):
    started = time.monotonic() if started is None else started
    terminated_at = None
    with (directory / 'stdout.log').open('wb') as stdout, (directory / 'stderr.log').open('wb') as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while True:
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    child.returncode = os.waitstatus_to_exitcode(status)
                    break
                elapsed = time.monotonic() - started
                if elapsed >= timeout and terminated_at is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated_at = time.monotonic()
                elif terminated_at is not None and time.monotonic() - terminated_at >= 5:
                    os.killpg(child.pid, signal.SIGKILL)
                time.sleep(0.02)
        except BaseException:
            os.killpg(child.pid, signal.SIGKILL)
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
            raise
    peak = usage.ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)
    return {'exit_code': child.returncode, 'timed_out': terminated_at is not None,
            'process_cold_total_ms': (time.monotonic() - started) * 1000,
            'user_cpu_seconds': usage.ru_utime, 'system_cpu_seconds': usage.ru_stime,
            'max_individual_process_rss_bytes': peak,
            'cpu_scope': 'wait4 child plus waited descendants; excludes fixture server and harness',
            'rss_scope': 'OS wait4 maximum individual process RSS; not concurrent process-tree aggregate',
            'stdout_tail': log_tail(directory / 'stdout.log'),
            'stderr_tail': log_tail(directory / 'stderr.log')}


def prepare_native_entry(path):
    """Inline the exact shared source at the native adapter's sole file import."""
    adapter_path = ROOT / 'runtime/fixtures/efficiency-native.mjs'
    workload_path = ROOT / 'runtime/fixtures/efficiency-workload.mjs'
    adapter = adapter_path.read_bytes()
    workload = workload_path.read_bytes()
    import_line = b"import { runEfficiency } from './efficiency-workload.mjs';"
    if adapter.count(import_line) != 1:
        raise ValueError('native adapter must contain exactly one shared-workload import')
    source = adapter.replace(import_line, workload, 1)
    with path.open('xb') as entry:
        entry.write(source)
    return {'path': str(path), 'sha256': hashlib.sha256(source).hexdigest(),
            'composition': 'inline-native-workload-v1',
            'adapter_path': str(adapter_path), 'adapter_sha256': hashlib.sha256(adapter).hexdigest(),
            'workload_path': str(workload_path), 'workload_sha256': hashlib.sha256(workload).hexdigest(),
            'generator_sha256': digest(pathlib.Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('native', 'chromium'), required=True)
    parser.add_argument('--runtime', type=pathlib.Path)
    parser.add_argument('--node', default='node')
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--tasks', type=int, default=2, help='independent tasks per worker per round; no retries')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=100)
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    if (not args.output.is_absolute() or not 1 <= args.rounds <= 100 or not 1 <= args.tasks <= 100
            or not 1 <= args.concurrency <= 16 or not 1 <= args.nodes <= 100000
            or args.rounds * args.tasks * args.concurrency > 1000 or not 1 <= args.timeout <= 600):
        parser.error('absolute output path and bounded workload required (at most1000 total tasks)')
    if args.output.exists():
        parser.error('output already exists; preserve existing measurement')
    reference = ROOT / 'scripts/efficiency-reference.mjs'
    executable = args.runtime.resolve() if args.runtime else pathlib.Path(shutil.which(args.node) or args.node)
    if args.engine == 'native' and not args.runtime:
        parser.error('--runtime must name the existing exact runtime binary')
    if not executable.is_file():
        parser.error(f'executable does not exist: {executable}')
    executable_before = executable.stat()
    provenance = source_identity()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    native_entry = None
    if args.engine == 'native':
        try:
            native_entry = prepare_native_entry(args.output.with_name(args.output.name + '.entry.mjs'))
        except (FileExistsError, ValueError) as error:
            parser.error(str(error))
    config = {key: getattr(args, key) for key in ('rounds', 'tasks', 'concurrency', 'nodes')}
    fixture = Fixture(config)
    thread = threading.Thread(target=fixture.serve_forever, name='owned-efficiency-fixture', daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{fixture.server_port}/'
    command = ([str(executable), '--script', native_entry['path'], '--fixture-url', url] if native_entry
               else [str(executable), str(reference), url])
    try:
        with tempfile.TemporaryDirectory(prefix='web-efficiency-', dir=args.output.parent) as temporary:
            fixture.started_at = time.monotonic()
            measured = execute(command, pathlib.Path(temporary), args.timeout, fixture.started_at)
        audit = fixture.audit()
        valid = audit['all_tasks_verified'] and measured['exit_code'] == 0 and not measured['timed_out']
        reason = audit['error']
        if reason is None and not valid:
            reason = 'process timed out' if measured['timed_out'] else 'engine process failed'
        executable_after = executable.stat()
        if (executable_before.st_dev, executable_before.st_ino, executable_before.st_size, executable_before.st_mtime_ns) != (
                executable_after.st_dev, executable_after.st_ino, executable_after.st_size, executable_after.st_mtime_ns):
            valid, reason = False, 'engine executable changed during measurement'
        if native_entry:
            try:
                entry_unchanged = digest(pathlib.Path(native_entry['path'])) == native_entry['sha256']
            except OSError:
                entry_unchanged = False
            if not entry_unchanged:
                valid, reason = False, 'executed native entry changed during measurement'
        measured['verified_completed_tasks_per_second_including_all_process_time'] = (
            audit['independently_verified_completed_tasks'] * 1000 / measured['process_cold_total_ms'])
        boundaries = [fixture.started_at] + [event[0] for event in fixture.round_events]
        measured['server_monotonic_round_intervals_ms'] = [
            (end - start) * 1000 for start, end in zip(boundaries, boundaries[1:])]
        measured['round_interval_scope'] = ('first: process start through first complete-round receipt; '
                                            'later: between complete-round receipts, same tasks, browser/context reused; '
                                            'includes receipt transport; unsuccessful rounds remain visible')
        receipt = {'schema': 'greppy.web-efficiency.measurement.v1', 'engine': args.engine,
                   'quality_verified': valid, 'quality_error': reason,
                   'configuration': config, 'host_envelope': host_envelope(),
                   'executable_sha256': digest(executable),
                   'source_provenance': provenance,
                   'workload_sha256': (native_entry['workload_sha256'] if native_entry
                                       else digest(ROOT / 'runtime/fixtures/efficiency-workload.mjs')),
                   'execution_surface': 'standalone runtime/Playwright; excludes LLM and CLI daemon session limits',
                   'cold_scope': 'new process/browser; OS caches are not cleared or claimed cold',
                   'metrics': measured, 'audit': audit,
                   'workload': fixture.receipts[0] if len(fixture.receipts) == 1 else None,
                   'server_evidence': {'expected_effects': len(fixture.expected), 'observed_effects': len(fixture.effects),
                                       'violations': fixture.violations, 'http_requests': fixture.requests,
                                       'request_body_bytes': fixture.request_body_bytes,
                                       'response_body_bytes': fixture.response_body_bytes,
                                       'network_scope': 'fixture HTTP bodies only; excludes headers, TCP/TLS and IPC'}}
        if native_entry:
            receipt['native_entry_provenance'] = native_entry
        if args.engine == 'chromium':
            try:
                metadata = json.loads(measured['stdout_tail'].splitlines()[0])['engine_metadata']
                metadata['chromium_executable_sha256'] = digest(metadata['chromium_executable'])
                receipt['reference_provenance'] = metadata
            except (ValueError, KeyError, IndexError, OSError):
                receipt['quality_verified'] = valid = False
                receipt['quality_error'] = reason = 'missing actual reference version/executable provenance'
        with args.output.open('x') as output:
            json.dump(receipt, output, indent=2)
            output.write('\n')
        print(json.dumps({'output': str(args.output), 'quality_verified': valid, 'quality_error': reason}))
        return 0 if valid else 1
    finally:
        fixture.shutdown()
        fixture.server_close()
        thread.join(timeout=5)


if __name__ == '__main__':
    raise SystemExit(main())
