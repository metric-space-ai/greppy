"""Check that thrown errors cannot impersonate a successful PW return value."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid

from candidate import capture
from prepare_context import prepare


def run(cli, runtime, scratch, evidence):
    evidence.mkdir(parents=True, exist_ok=False)
    aliases = scratch / 'aliases'
    aliases.mkdir(parents=True, exist_ok=False)
    name = 'pw-result-' + uuid.uuid4().hex[:12]
    context = prepare(scratch / 'contexts', cli, aliases, name, name, runtime=runtime)
    (evidence / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
    calls = []
    terminal = {'schema': 'greppy.pw-result-marker-proof.v1', 'passed': False,
                'efficiency_accepted': False, 'cases': []}

    def call(*args):
        record = {'argv': [context['alias'], 'web', *args, '--json']}
        start = time.monotonic()
        try:
            r = subprocess.run(record['argv'], text=True, capture_output=True, timeout=45)
            record.update(exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr)
            return r
        except subprocess.TimeoutExpired as error:
            record.update(timeout=True, stdout=str(error.stdout), stderr=str(error.stderr))
            raise
        finally:
            record['wall_seconds'] = time.monotonic() - start
            calls.append(record)
            (evidence / 'calls.json').write_text(json.dumps(calls, indent=2) + '\n')

    try:
        session = call('session', 'create', '--profile', 'project')
        assert session.returncode == 0, session.stdout
        control = call('pw', 'return {proof: "returned"};')
        assert control.returncode == 0 and json.loads(control.stdout)['result']['value'] == {'proof': 'returned'}, control.stdout
        for name, marker, code in [
            ('thrown_marker', 'thrown', 'throw new Error(' + json.dumps('PWRESULT {"proof":"thrown"}') + ');'),
            ('page_error_marker', 'page-error', 'await page.evaluate(() => { throw new Error(' + json.dumps('PWRESULT {"proof":"page-error"}') + '); }); return "unreachable";'),
            ('malformed_marker', 'not-json', 'throw new Error("PWRESULT not-json");'),
        ]:
            result = call('pw', code)
            reply = json.loads(result.stdout)
            error = reply.get('error', {})
            message = error.get('message', '')
            terminal['cases'].append({'name': name, 'exit_code': result.returncode,
                                      'status': reply.get('status'), 'result': reply.get('result'),
                                      'error': error,
                                      'correct_error': result.returncode != 0 and reply.get('status') == 'error' and 'PWRESULT' in message and marker in message})
        terminal['candidate_unchanged'] = capture(cli, runtime) == context['candidate']
        terminal['passed'] = terminal['candidate_unchanged'] and all(c['correct_error'] for c in terminal['cases'])
    except BaseException as error:
        terminal['failure'] = repr(error)
        raise
    finally:
        try:
            cleanup = call('runtime', 'stop')
            terminal['cleanup'] = json.loads(cleanup.stdout)
            terminal['passed'] = terminal['passed'] and cleanup.returncode == 0 and terminal['cleanup']['result']['running'] is False
        except BaseException as error:
            terminal.update(passed=False, cleanup_failure=repr(error))
        (evidence / 'terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
    assert terminal['passed'], terminal
    print(json.dumps(terminal))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cli', 'runtime', 'scratch', 'evidence'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    run(a.cli, a.runtime, a.scratch, a.evidence)
