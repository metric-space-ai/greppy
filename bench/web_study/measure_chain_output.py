"""Deterministic output-size check, not a browser or agent performance benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
argv = [str(a.binary.resolve()), 'web', 'do', 'script', 'list', '::', 'script', 'list']
sha = hashlib.sha256(a.binary.read_bytes()).hexdigest()
results = []
with tempfile.TemporaryDirectory(prefix='chain-size-', dir=os.environ['TMPDIR']) as directory:
    for compact in (False, True):
        env = os.environ.copy()
        env.pop('GREPPY_WEB_VIEW', None)
        env['GREPPY_WEB_CHAIN_VIEW'] = 'compact' if compact else ''
        r = subprocess.run(argv, cwd=directory, env=env, capture_output=True)
        results.append({
            'compact': compact, 'exit_code': r.returncode,
            'stdout': r.stdout.decode('utf-8', errors='replace'),
            'stderr': r.stderr.decode('utf-8', errors='replace'),
            'stdout_bytes': len(r.stdout), 'stderr_bytes': len(r.stderr),
            'payloads_preserved': r.stdout.count(b'"scripts":[]') == 2,
        })
record = {
    'schema': 'greppy.chain-output-size.v1', 'argv': argv, 'binary_sha256': sha,
    'binary_unchanged': sha == hashlib.sha256(a.binary.read_bytes()).hexdigest(),
    'results': results, 'model_tokens': None, 'acceptance': False,
    'limitation': 'Debug CI-assets CLI; script-list only; no browser, model or speed measurement.',
}
if all(r['exit_code'] == 0 and r['payloads_preserved'] for r in results):
    record['stdout_bytes_change_percent'] = (results[1]['stdout_bytes'] / results[0]['stdout_bytes'] - 1) * 100
with a.output.open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record))
