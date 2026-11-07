#!/usr/bin/env python3
"""Real timeout/signal cleanup contracts; no browser, install or fake runtime."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
METER = ROOT / 'crates/web-runtime/scripts/measure-web-efficiency.py'

class ObservationInterrupted(RuntimeError):
    pass

def exchange(state, message):
    row = json.loads((state / 'ready.json').read_text())
    with socket.create_connection(('127.0.0.1', row['port']), timeout=1) as client:
        client.sendall((row['nonce'] + ' ' + message + '\n').encode())
        return client.makefile('rb').readline(64).rstrip(b'\n')

def worker(state):
    signal.alarm(10)
    nonce = secrets.token_hex(12)
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen()
        server.settimeout(1)
        temporary = state / 'ready.tmp'
        temporary.write_text(json.dumps(
            {'pid': os.getpid(), 'port': server.getsockname()[1], 'nonce': nonce}))
        temporary.rename(state / 'ready.json')
        while True:
            try:
                client, _ = server.accept()
            except socket.timeout:
                continue
            with client:
                message = client.makefile('rb').readline(128).decode().strip()
                if message == nonce + ' stop':
                    client.sendall(b'bye\n')
                    return
                if message == nonce + ' ping':
                    client.sendall(b'ready\n')

def target(state, hold=True):
    signal.alarm(10)
    (state / 'target.json').write_text(json.dumps({'pid': os.getpid(), 'pgid': os.getpgrp()}))
    subprocess.Popen([sys.executable, '-B', __file__, 'worker', str(state)],
                     start_new_session=True)
    deadline = time.monotonic() + 3
    while not (state / 'ready.json').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('detached fixture never became ready')
        time.sleep(0.01)
    if hold:
        time.sleep(30)

def stop(state):
    if ((state / 'hold-first-cleanup').exists()
            and not (state / 'cleanup-started').exists()):
        signal.alarm(5)
        (state / 'cleanup-started').write_text('first control is live\n')
        while not (state / 'release-cleanup').exists():
            time.sleep(0.02)
    try:
        exchange(state, 'stop')
    except (OSError, FileNotFoundError):
        pass
    (state / 'cleanup-ran').write_text('explicit stop executed\n')

def responding(state):
    try:
        return exchange(state, 'ping') == b'ready'
    except (OSError, FileNotFoundError):
        return False

def check_case(output, case):
    spec = importlib.util.spec_from_file_location('meter_cleanup_target', METER)
    meter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(meter)
    output.mkdir()
    state = output / 'state'
    state.mkdir(mode=0o700)
    logs = output / 'logs'
    logs.mkdir()
    if case == 'interrupt-cleanup':
        (state / 'hold-first-cleanup').touch()
    cancelled = threading.Event()
    delivered = threading.Event()
    guardian = None
    previous = None
    measured = None
    interrupted = False

    def interrupt_when_ready():
        deadline = time.monotonic() + 3
        while not cancelled.is_set() and time.monotonic() < deadline:
            if (responding(state) and (case != 'interrupt-cleanup'
                                      or (state / 'cleanup-started').exists())):
                delivered.set()
                os.kill(os.getpid(), signal.SIGUSR1)
                return
            cancelled.wait(0.01)

    def interrupted_observation(_signal, _frame):
        raise ObservationInterrupted('real SIGUSR1 interrupted observation')

    if case != 'timeout':
        previous = signal.signal(signal.SIGUSR1, interrupted_observation)
        guardian = threading.Thread(target=interrupt_when_ready, name='owned-meter-interrupt')
        guardian.start()
    try:
        try:
            measured = meter.execute(
                [sys.executable, '-B', __file__,
                 'target-returns' if case == 'interrupt-cleanup' else 'target', str(state)],
                logs, 1 if case == 'timeout' else 4,
                cleanup_command=[sys.executable, '-B', __file__, 'stop', str(state)])
        except ObservationInterrupted:
            interrupted = True
        alive = responding(state)
        result = {'case': case, 'explicit_stop_executed': (state / 'cleanup-ran').exists(),
                  'detached_server_still_responds': alive,
                  'real_interrupt_delivered': delivered.is_set(), 'original_interrupt_preserved': interrupted,
                  'timed_out': measured['timed_out'] if measured else None,
                  'cleanup_exit_code': measured['cleanup_exit_code'] if measured else None}
        (output / 'contract.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
        if case == 'timeout':
            if (not measured or not measured['timed_out'] or measured['cleanup_exit_code'] != 0
                    or not result['explicit_stop_executed'] or alive):
                raise RuntimeError('meter.timeout_explicit_cleanup: timed-out owner was not stopped')
        elif not (delivered.is_set() and interrupted and result['explicit_stop_executed'] and not alive):
            raise RuntimeError('meter.interrupted_explicit_cleanup: interrupted owner was not stopped')
    finally:
        cancelled.set()
        if guardian is not None:
            guardian.join(timeout=4)
            if guardian.is_alive():
                raise RuntimeError('owned signal guardian exceeded its deadline')
            signal.signal(signal.SIGUSR1, previous)
        # Recovery belongs to this fixture only. The server also self-exits after10s.
        stop(state)
        deadline = time.monotonic() + 2
        while responding(state) and time.monotonic() < deadline:
            time.sleep(0.02)
        if responding(state):
            raise RuntimeError('owned fixture failed bounded shutdown')

def run_case(output, case, expected=None):
    result = subprocess.run([sys.executable, '-B', '-X',
                             'pycache_prefix=' + str(output / 'python-cache'),
                             __file__, '--case', case,
                             '--output', str(output)],
                            capture_output=True, text=True, timeout=15)
    print(result.stdout, end='')
    if expected is None:
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr, end='')
            raise RuntimeError('real cleanup contract failed')
    else:
        print(result.stderr, end='')
        if result.returncode != 1 or expected not in result.stderr:
            raise RuntimeError('cleanup source counter did not fail at its named contract')
        print('mutation: explicit meter cleanup rejected ' + expected)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=('timeout', 'interrupt', 'interrupt-cleanup'))
    parser.add_argument('--mutations', action='store_true')
    args = parser.parse_args()
    if os.name != 'posix' or not args.output.is_absolute() or args.output.exists():
        parser.error('POSIX and a new absolute output path are required')
    if args.case:
        check_case(args.output, args.case)
        return
    args.output.mkdir()
    for case in ('timeout', 'interrupt', 'interrupt-cleanup'):
        run_case(args.output / ('positive-' + case), case)
    if not args.mutations:
        return
    original = METER.read_text()
    faults = [
        ('timeout', 'if child.returncode is not None and cleanup_command is not None and cleanup is None:',
         'if child.returncode is not None and cleanup_command is not None and cleanup is None and terminated_at is None:',
         'meter.timeout_explicit_cleanup'),
        ('timeout', 'or now - cleanup_started_at >= 3):', 'or now - cleanup_started_at >= 0):',
         'meter.timeout_explicit_cleanup'),
        ('interrupt', '                    failure_cleanup(cleanup_command, directory, env, cwd, resource_group)',
         '                    pass  # source counter omits interrupted owner stop',
         'meter.interrupted_explicit_cleanup'),
        ('interrupt-cleanup', 'and (cleanup is None or cleanup.returncode != 0)):',
         'and cleanup is None):', 'meter.interrupted_explicit_cleanup'),
    ]
    try:
        for index, (case, before, after, marker) in enumerate(faults):
            if original.count(before) != 1:
                raise RuntimeError('cleanup source counter target is not unique')
            METER.write_text(original.replace(before, after, 1))
            try:
                run_case(args.output / ('fault-' + str(index)), case, marker)
            finally:
                METER.write_text(original)
            run_case(args.output / ('restore-' + str(index)), case)
    finally:
        METER.write_text(original)

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'target-returns':
        target(Path(sys.argv[2]), hold=False)
    elif len(sys.argv) == 3 and sys.argv[1] in ('worker', 'target', 'stop'):
        {'worker': worker, 'target': target, 'stop': stop}[sys.argv[1]](Path(sys.argv[2]))
    else:
        main()
