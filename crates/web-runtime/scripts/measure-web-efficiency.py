#!/usr/bin/env python3
"""Bounded, opt-in comparison of one identical complete browser workload.

Requires existing binaries/packages. Never installs, builds, changes host limits,
or claims that a scripted web measurement covers LLM/model-provider work.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
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
_memory_spec = importlib.util.spec_from_file_location(
    'greppy_process_memory', ROOT.parents[1] / 'bench/process_memory.py')
_memory_module = importlib.util.module_from_spec(_memory_spec)
_memory_spec.loader.exec_module(_memory_module)
ProcessTreeMemorySampler = _memory_module.ProcessTreeMemorySampler


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
                                         'runtime/fixtures/efficiency-native.mjs', ':(top)bench/process_memory.py')}


def cgroup_v2_envelope(proc_self=pathlib.Path('/proc/self')):
    """Observe visible ancestor ceilings, never reserved or guaranteed capacity."""
    from fractions import Fraction

    result = {'leaf': None, 'visible_ancestors': [], 'errors': [],
              'cpu_quota_upper_bound': None, 'cpu_quota_upper_bound_ratio': None,
              'memory_max_upper_bound_bytes': None, 'swap_max_upper_bound_bytes': None,
              'scope': 'visible cgroup-v2 ancestor ceilings; hidden namespace ancestors and competing workloads are unknown'}

    def bounded_text(path, maximum=1024 * 1024):
        with path.open(encoding='utf-8') as source:
            value = source.read(maximum + 1)
        if len(value) > maximum:
            raise ValueError('resource metadata exceeds observation bound')
        return value.strip()

    def mount_field(value):
        for escaped, literal in (('\\040', ' '), ('\\011', '\t'), ('\\012', '\n'), ('\\134', '\\')):
            value = value.replace(escaped, literal)
        return value

    try:
        rows = bounded_text(proc_self / 'cgroup').splitlines()
        member = next(row[3:] for row in rows if row.startswith('0::'))
        membership = pathlib.PurePosixPath(member)
        if not membership.is_absolute() or '..' in membership.parts:
            raise ValueError('invalid cgroup-v2 membership')
        candidates = []
        for row in bounded_text(proc_self / 'mountinfo').splitlines():
            before, separator, after = row.partition(' - ')
            if not separator or not after.split() or after.split()[0] != 'cgroup2':
                continue
            fields = before.split()
            if len(fields) < 6:
                raise ValueError('invalid cgroup-v2 mount metadata')
            root = pathlib.PurePosixPath(mount_field(fields[3]))
            mount = pathlib.Path(mount_field(fields[4]))
            if not root.is_absolute() or '..' in root.parts or not mount.is_absolute() or '..' in mount.parts:
                raise ValueError('invalid cgroup-v2 mount path')
            if membership.is_relative_to(root):
                candidates.append((root, mount))
        if not candidates:
            raise ValueError('cgroup-v2 membership cannot be mapped to a visible mount')
        root, mount = max(candidates, key=lambda item: len(item[0].parts))
        group = mount / membership.relative_to(root)
        result.update(membership=member, mount_root=str(root), mount_point=str(mount))
        quotas, memory, swap = [], [], []
        for _ in range(256):
            values = {'path': str(group)}
            for name in ('cpu.max', 'memory.max', 'memory.swap.max'):
                try:
                    values[name] = bounded_text(group / name, 4096)
                except FileNotFoundError:
                    # The hierarchy root, or a disabled controller, has no file.
                    values[name] = None
                except (OSError, UnicodeError, ValueError) as error:
                    values[name] = None
                    result['errors'].append(f'{group / name}: {error}')
            if result['leaf'] is None:
                result['leaf'] = {name: values[name] for name in ('cpu.max', 'memory.max', 'memory.swap.max')}
            result['visible_ancestors'].append(values)
            try:
                if values['cpu.max'] is not None:
                    quota, period = values['cpu.max'].split()
                    if int(period) <= 0 or (quota != 'max' and int(quota) <= 0):
                        raise ValueError('nonpositive CPU quota or period')
                    if quota != 'max':
                        quotas.append(Fraction(int(quota), int(period)))
                for name, target in (('memory.max', memory), ('memory.swap.max', swap)):
                    value = values[name]
                    if value is not None and value != 'max':
                        if int(value) < 0:
                            raise ValueError('negative memory ceiling')
                        target.append(int(value))
            except ValueError as error:
                result['errors'].append(f'{group}: {error}')
            if group == mount:
                break
            group = group.parent
        else:
            result['errors'].append('visible cgroup ancestry exceeds 256 levels')
        if quotas:
            quota = min(quotas)
            result['cpu_quota_upper_bound'] = float(quota)
            result['cpu_quota_upper_bound_ratio'] = [quota.numerator, quota.denominator]
        result['memory_max_upper_bound_bytes'] = min(memory) if memory else None
        result['swap_max_upper_bound_bytes'] = min(swap) if swap else None
    except StopIteration:
        result['errors'].append('no cgroup-v2 membership')
    except (OSError, UnicodeError, ValueError) as error:
        result['errors'].append(str(error))
    return result


def host_envelope():
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    result = {'platform': platform.platform(), 'logical_cpus': os.cpu_count(),
              'cpu_affinity': affinity, 'limits_modified_by_harness': False, 'cgroup_v2': None}
    cgroup = cgroup_v2_envelope()
    result['cgroup_v2'] = cgroup['leaf']
    result['cgroup_v2_observation'] = cgroup
    bounds = [value for value in (len(affinity) if affinity else os.cpu_count(),
                                  cgroup['cpu_quota_upper_bound']) if value is not None]
    result['sustained_cpu_capacity_upper_bound'] = min(bounds) if bounds else None
    result['capacity_scope'] = 'receipt-time affinity/logical-CPU and visible quota rates averaged over their periods; no reserved capacity, hidden-ancestor or interval-stability claim'
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


def execute(command, directory, timeout, started=None, memory_log_path=None, *, env=None, cwd=None):
    started = time.monotonic() if started is None else started
    terminated_at = None
    memory_log_path = memory_log_path or directory / 'process-memory.jsonl'
    sampling_wall_ns = sampling_cpu_ns = 0
    # Both engines use this same bounded monitor. Stream raw receipts instead
    # of retaining a sample history in the harness heap.
    with (directory / 'stdout.log').open('wb') as stdout, (directory / 'stderr.log').open('wb') as stderr, memory_log_path.open('x') as memory_log:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True,
                                 env=env, cwd=cwd)
        try:
            memory = ProcessTreeMemorySampler(child.pid)
            next_sample = time.monotonic()
            while True:
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    child.returncode = os.waitstatus_to_exitcode(status)
                    break
                now = time.monotonic()
                if memory.supported and now >= next_sample:
                    sample_started = time.monotonic_ns()
                    cpu_started = time.thread_time_ns()
                    sample = memory.sample()
                    memory_log.write(json.dumps(sample, separators=(',', ':')) + '\n')
                    sampling_cpu_ns += time.thread_time_ns() - cpu_started
                    sampling_wall_ns += time.monotonic_ns() - sample_started
                    # Never catch up with a burst of scans after a slow sample.
                    next_sample = time.monotonic() + 0.25
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
            'process_family_memory': {
                'supported': memory.supported,
                'sample_count': memory.samples, 'incomplete_sample_count': memory.incomplete_samples,
                'peak_sampled_pss_bytes': memory.peak_pss_bytes,
                'peak_sampled_private_bytes': memory.peak_private_bytes,
                'minimum_sampling_gap_ms': 250,
                'sampling_wall_ms': sampling_wall_ns / 1_000_000,
                'monitor_thread_cpu_seconds': sampling_cpu_ns / 1_000_000_000,
                'samples_path': str(memory_log_path), 'samples_sha256': digest(memory_log_path),
                'sampler_sha256': digest(pathlib.Path(_memory_module.__file__)),
                'scope': 'observed Linux child process-family mappings; sampled peaks are lower bounds; excludes kernel/GPU/unmapped cache, fixture and harness; no inference about LLM/CLI-daemon memory',
                'timing_scope': 'process time includes concurrent sampling overhead; monitor CPU reported separately'},
            'stdout_tail': log_tail(directory / 'stdout.log'),
            'stderr_tail': log_tail(directory / 'stderr.log')}


def prepare_native_entry(path, fixture_url=None):
    """Inline the exact shared source at the native adapter's sole file import."""
    adapter_path = ROOT / 'runtime/fixtures/efficiency-native.mjs'
    workload_path = ROOT / 'runtime/fixtures/efficiency-workload.mjs'
    adapter = adapter_path.read_bytes()
    workload = workload_path.read_bytes()
    import_line = b"import { runEfficiency } from './efficiency-workload.mjs';"
    if adapter.count(import_line) != 1:
        raise ValueError('native adapter must contain exactly one shared-workload import')
    source = adapter.replace(import_line, workload, 1)
    if fixture_url is not None:
        # The public CLI run verb has no runtime fixture flag. Bind only the
        # generated loopback URL in the same hashed module as the workload.
        source = b'const fixtureUrl = ' + json.dumps(fixture_url).encode('utf-8') + b';\n' + source
    with path.open('xb') as entry:
        entry.write(source)
    return {'path': str(path), 'sha256': hashlib.sha256(source).hexdigest(),
            'composition': 'inline-native-workload-v1',
            'fixture_binding': 'module-local literal' if fixture_url is not None else 'runtime fixture flag',
            'adapter_path': str(adapter_path), 'adapter_sha256': hashlib.sha256(adapter).hexdigest(),
            'workload_path': str(workload_path), 'workload_sha256': hashlib.sha256(workload).hexdigest(),
            'generator_sha256': digest(pathlib.Path(__file__)),
            'native_launch_policy': {'version': 'native-fixed-headless-v1',
                                     'requested': {'headless': True},
                                     'effective': 'fixed native headless execution through launch() without options',
                                     'other_options': 'rejected, not dropped',
                                     'viewport': 'runtime default, no override',
                                     'contexts': 'unchanged newContext() per worker'}}


def cli_execution(cli, runtime, entry, state, timeout):
    """Use public CLI composition; never attach to a user's existing owner."""
    workspace = state / 'workspace'
    workspace.mkdir()
    temporary = state / 'tmp'
    temporary.mkdir()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith('GREPPY_')}
    environment.update({
        'GREPPY_WEB_RUNTIME': str(runtime),
        'GREPPY_WEB_RUNTIME_DIR': str(state),
        'GREPPY_RUNTIME_DIR': str(state),
        'GREPPY_STORE_DIR': str(state / 'store'),
        'GREPPY_WEB_AGENT': state.name,
        'TMPDIR': str(temporary),
    })
    command = [str(cli), 'web', 'do', '--json',
               'session', 'new', '--profile', 'project', '::',
               'run', '--mode', 'standalone', '--script-file', str(entry),
               '--timeout', str(int(timeout * 1000))]
    return command, environment, workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('native', 'chromium', 'cli'), required=True)
    parser.add_argument('--runtime', type=pathlib.Path)
    parser.add_argument('--cli', type=pathlib.Path)
    parser.add_argument('--state-parent', type=pathlib.Path,
                        help='CLI only: existing absolute private-state parent of at most20 bytes')
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
    executable = (args.cli.resolve() if args.engine == 'cli' and args.cli else
                  args.runtime.resolve() if args.runtime else pathlib.Path(shutil.which(args.node) or args.node))
    if args.engine in ('native', 'cli') and not args.runtime:
        parser.error('--runtime must name the existing exact runtime binary')
    if args.engine == 'cli' and (not args.cli or not args.state_parent
                                or not args.state_parent.is_absolute()
                                or not args.state_parent.is_dir()
                                or len(os.fsencode(args.state_parent)) + 12 > 32):
        parser.error('CLI mode requires --cli and an existing absolute --state-parent of at most20 bytes')
    if args.engine != 'cli' and (args.cli or args.state_parent):
        parser.error('--cli and --state-parent apply only to --engine cli')
    if not executable.is_file():
        parser.error(f'executable does not exist: {executable}')
    executable_before = executable.stat()
    runtime = args.runtime.resolve() if args.engine == 'cli' else None
    if runtime is not None and not runtime.is_file():
        parser.error(f'runtime executable does not exist: {runtime}')
    runtime_before = runtime.stat() if runtime is not None else None
    runtime_sha256 = digest(runtime) if runtime is not None else None
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
    url = f'http://127.0.0.1:{fixture.server_port}/'
    state = None
    if args.engine == 'cli':
        try:
            native_entry = prepare_native_entry(args.output.with_name(args.output.name + '.entry.mjs'), url)
            state = pathlib.Path(tempfile.mkdtemp(prefix='gw-', dir=args.state_parent))
            if len(os.fsencode(state)) > 32:
                raise ValueError('generated CLI private state exceeds32 bytes; use a shorter --state-parent')
        except (FileExistsError, ValueError) as error:
            fixture.server_close()
            parser.error(str(error))
    thread = threading.Thread(target=fixture.serve_forever, name='owned-efficiency-fixture', daemon=True)
    thread.start()
    command = ([str(executable), '--script', native_entry['path'], '--fixture-url', url] if native_entry
               else [str(executable), str(reference), url])
    try:
        with tempfile.TemporaryDirectory(prefix='web-efficiency-', dir=args.output.parent) as temporary:
            environment = cwd = None
            if state is not None:
                command, environment, cwd = cli_execution(executable, runtime, native_entry['path'], state, args.timeout)
            fixture.started_at = time.monotonic()
            measured = execute(command, pathlib.Path(temporary), args.timeout, fixture.started_at,
                               args.output.with_name(args.output.name + '.memory.jsonl'), env=environment, cwd=cwd)
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
        if runtime is not None:
            runtime_after = runtime.stat()
            if ((runtime_before.st_dev, runtime_before.st_ino, runtime_before.st_size, runtime_before.st_mtime_ns)
                    != (runtime_after.st_dev, runtime_after.st_ino, runtime_after.st_size, runtime_after.st_mtime_ns)
                    or digest(runtime) != runtime_sha256):
                valid, reason = False, 'CLI runtime executable changed during measurement'
            measured['process_family_memory']['scope'] = (
                'observed Linux CLI and runtime descendant mappings; sampled peaks are lower bounds; '
                'detached descendants are retained only if observed before reparenting; '
                'excludes kernel/GPU/unmapped cache, fixture, harness and LLM/model providers')
            observed = measured['process_family_memory']
            measured['observed_process_family_memory'] = observed
            measured['process_family_memory'] = {
                **observed,
                'peak_sampled_pss_bytes': None,
                'peak_sampled_private_bytes': None,
                'complete_for_execution_surface': False,
                'scope': 'unknown whole CLI/runtime memory: detached owner PID/birth was not registered from a supported owner receipt; observed samples are separate',
            }
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
                   'execution_surface': ('public CLI web do/session/run plus runtime; excludes LLM/model-provider work'
                                         if state is not None else
                                         'standalone runtime/Playwright; excludes LLM and CLI daemon session limits'),
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
        if state is not None:
            receipt['cli_provenance'] = {
                'executable': str(executable), 'sha256': receipt['executable_sha256'],
                'runtime_executable': str(runtime), 'runtime_sha256': runtime_sha256,
                'private_state_directory': str(state),
                'state_cleanup': 'preserved for ownership/lifecycle inspection; not automatically deleted',
                'lifecycle_scope': 'normal standalone CLI shutdown; no proof that every descendant exited',
                'session_profile': 'project; explicit loopback fixture, normal session limits',
                'entry_url_binding': 'module-local generated loopback URL; no production fixture env',
            }
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
