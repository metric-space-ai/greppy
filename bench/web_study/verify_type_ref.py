"""Recheck the reported immediate observe -> type @ref regression, not a benchmark."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--cli', type=Path, required=True)
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--url', required=True)
p.add_argument('--workspace', type=Path, required=True)
p.add_argument('--report', type=Path, required=True)
a = p.parse_args()
a.workspace.mkdir(parents=True, exist_ok=False)
if a.report.exists(): raise FileExistsError('Do not overwrite a previous result')
env = os.environ.copy()
for name in ('GREPPY_WEB_SESSION', 'GREPPY_WEB_TAB', 'GREPPY_WEB_AGENT', 'GREPPY_WEB_VIEW'):
    env.pop(name, None)
env['GREPPY_RUN_ID'] = 'ref-check-' + uuid.uuid4().hex
env['GREPPY_WEB_RUNTIME'] = str(a.runtime)
env['GREPPY_WEB_RUNTIME_DIR'] = str(a.workspace / 'runtime')
record = {'purpose': 'excluded debug correctness reproduction', 'url': a.url, 'cwd': str(a.workspace),
          'environment': {k: env[k] for k in ('GREPPY_RUN_ID', 'GREPPY_WEB_RUNTIME', 'GREPPY_WEB_RUNTIME_DIR')},
          'cli': str(a.cli), 'runtime': str(a.runtime), 'records': [], 'passed': False}


def step(*args):
    argv = [str(a.cli), *args]
    try:
        r = subprocess.run(argv, env=env, cwd=a.workspace, capture_output=True, text=True, timeout=150)
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else (value or '')
        record['records'].append({'argv': argv, 'exit': None, 'timeout_seconds': 150,
                                  'stdout': decoded(error.stdout), 'stderr': decoded(error.stderr)})
        a.report.write_text(json.dumps(record, indent=2))
        raise
    record['records'].append({'argv': argv, 'exit': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr})
    a.report.write_text(json.dumps(record, indent=2))
    if r.returncode: raise RuntimeError('CLI operation failed; see recorded output')
    value = json.loads(r.stdout)
    return value.get('result', value)


try:
    created = step('web', 'session', 'create', '--profile', 'project', '--json')
    sid = created['session_id']
    step('web', 'open', a.url, '--session', sid, '--json')
    before = step('web', 'observe', '--session', sid, '--json')
    inputs = [n for n in before['actionables'] if n.get('tag') == 'input']
    if len(inputs) != 1: raise RuntimeError('Fixture does not have exactly one input')
    reference = inputs[0]['ref']
    step('web', 'type', reference, '10115', '--session', sid, '--json')
    after = step('web', 'observe', '--session', sid, '--json')
    inputs_after = [n for n in after['actionables'] if n.get('tag') == 'input']
    record['passed'] = len(inputs_after) == 1 and inputs_after[0].get('value', inputs_after[0].get('text')) == '10115'
    record['typed_reference'] = reference
except Exception as error:
    record['harness_or_operation_error'] = str(error)
finally:
    a.report.write_text(json.dumps(record, indent=2))
print(json.dumps({'report': str(a.report), 'passed': record['passed'], 'operations': len(record['records']), 'error': record.get('harness_or_operation_error')}))
raise SystemExit(0 if record['passed'] else 1)
