#!/usr/bin/env python3
"""Real Linux ownership contract for the meter's opt-in stderr diagnostic."""
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'diagnostic_meter', ROOT / 'crates/web-runtime/scripts/measure-web-efficiency.py')
meter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meter)


def check():
    if not sys.platform.startswith('linux'):
        raise RuntimeError('the owned-proc-fd diagnostic contract requires Linux')
    with tempfile.TemporaryDirectory(prefix='greppy-diagnostic-', dir=os.environ.get('RUNNER_TEMP')) as temporary:
        root = Path(temporary)
        state = root / 'state'
        state.mkdir(mode=0o700)
        body = b'padding\n' * 10000 + b'owned diagnostic\nnonce=' + b'a' * 40 + b'\n'
        # Both child logs are real regular files. One is deliberately outside
        # this test's designated state; neither contains user credentials.
        for inside in (True, False):
            path = (state if inside else root) / 'web-startup-contract.log'
            with path.open('xb') as errors:
                child = subprocess.Popen(
                    [sys.executable, '-B', '-c',
                     'import os,signal,sys; signal.alarm(10); os.unlink(sys.argv[1]); print("ready",flush=True); sys.stdin.readline()',
                     str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
                try:
                    # The parent wrote deterministic bytes; the child holds the
                    # now-unlinked stderr file without sharing a read offset.
                    errors.write(body)
                    errors.flush()
                    assert child.stdout.readline() == b'ready\n'
                    identity = meter._memory_module._stat(child.pid)
                    image = Path('/proc') / str(child.pid) / 'exe'
                    info = image.stat()
                    seen = {'pid': child.pid, 'starttime_ticks': identity[1]}
                    expected_image = (info.st_dev, info.st_ino)
                    result = meter.capture_owned_runtime_stderr(seen, expected_image, state)
                    if not inside:
                        assert result['status'] == 'unavailable', 'diagnostic_foreign_state'
                        continue
                    assert result['status'] == 'captured', result
                    assert result['offset_bytes'] == len(body) - 65536, 'diagnostic_tail_bound'
                    assert result['tail_sha256'] == hashlib.sha256(body[-65536:]).hexdigest()
                    assert 'owned diagnostic' in result['stderr_redacted_tail']
                    assert '<opaque>' in result['stderr_redacted_tail']
                    assert 'a' * 40 not in result['stderr_redacted_tail'], 'diagnostic_opaque_value'
                    wrong_birth = dict(seen, starttime_ticks=seen['starttime_ticks'] + 1)
                    assert meter.capture_owned_runtime_stderr(wrong_birth, expected_image, state)['status'] == 'unavailable', 'diagnostic_foreign_birth'
                    assert meter.capture_owned_runtime_stderr(seen, (info.st_dev, info.st_ino + 1), state)['status'] == 'unavailable', 'diagnostic_foreign_image'
                finally:
                    try:
                        child.communicate(input=b'stop\n', timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.communicate(timeout=5)
                        raise
                    assert child.returncode == 0, 'diagnostic_owned_child_reap'
    print('OWNED_RUNTIME_DIAGNOSTIC_OK')


if __name__ == '__main__':
    check()
