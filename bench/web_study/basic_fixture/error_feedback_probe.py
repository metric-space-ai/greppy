"""Native stale-action feedback and safe recovery; never an agent token trial."""
import argparse
import hashlib
from http.server import HTTPServer
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid
from candidate import capture
from prepare_context import prepare
import server as fixture


def run(cli, runtime, scratch, evidence, require_current_guidance=False):
    evidence.mkdir(parents=True, exist_ok=False)
    aliases = scratch / 'aliases'
    aliases.mkdir(parents=True)
    identity = 'error-feedback-' + uuid.uuid4().hex[:12]
    context = prepare(scratch / 'contexts', cli, aliases, identity, identity,
                      runtime=runtime, view='compact', chain_view='compact')
    (evidence / 'context.json').write_text(json.dumps(context, indent=2))
    fixture.set_dir(scratch / 'fixture-state')
    run_id = uuid.uuid4().hex[:12]
    fixture.write_state(fixture.new_state(run_id, 'basic-development-table-1', 'table'))
    sources = [Path(__file__), Path(fixture.__file__), Path(fixture.table_case.__file__),
               *sorted((fixture.ROOT / 'static').glob('*'))]
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()}
    before = hashes()
    http = HTTPServer(('127.0.0.1', 0), fixture.Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    calls = []
    terminal = dict(schema='greppy.error-feedback-probe.v1', passed=False, run_id=run_id,
                    scope='Deterministic native error/formatter/recovery preflight; bytes are not tokens',
                    require_current_guidance=require_current_guidance)

    def call(*args, human=False, code=0):
        argv = [context['alias'], 'web', *args] + ([] if human else ['--json'])
        record = dict(argv=argv)
        started = time.monotonic()
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=60)
            record.update(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except subprocess.TimeoutExpired as error:
            record.update(timeout=True, stdout=str(error.stdout), stderr=str(error.stderr))
            raise
        finally:
            record['wall_seconds'] = time.monotonic() - started
            calls.append(record)
            (evidence / 'calls.json').write_text(json.dumps(calls, indent=2))
        assert p.returncode == code, record
        if human:
            return p.stdout
        reply = json.loads(p.stdout)
        assert reply['status'] == ('ok' if code == 0 else 'error'), reply
        return reply

    try:
        call('open', f'http://127.0.0.1:{http.server_port}/?run_id={run_id}')
        observed = call('observe')['result']
        old_ref = next(a['ref'] for a in observed['actionables'] if a['name'] == 'Reserve Ember')
        call('select', 'css=#inventory-region', 'EU')
        call('check', 'css=#inventory-capacity')
        call('select', 'css=#inventory-order', 'ascending')
        call('wait', 'css=#inventory-table tbody tr:nth-child(2) button[aria-label="Reserve Cedar"]',
             '--native', '--timeout', '5000')
        state_before_refusal = fixture.load(run_id)
        error = call('click', old_ref, code=34)
        assert error['error']['code'] == 'STALE_REF', error
        snapshot = error['result']['page_state']
        assert snapshot['status'] == 'available', error
        fresh_ref = next(a['ref'] for a in snapshot['snapshot']['actionables'] if a['name'] == 'Reserve Ember')
        assert fresh_ref != old_ref
        human = call('click', old_ref, human=True, code=34)
        fresh = re.search(r'^\s*"(@[0-9]+)" "button"[^\n]*name="Reserve Ember"', human, re.M)
        feedback = dict(stdout_bytes=len(human.encode()),
                        one_content_block=human.splitlines().count('UNTRUSTED_PAGE_CONTENT') == 1,
                        readable_current_ref=fresh is not None,
                        raw_snapshot_duplicated='"snapshot":' in human,
                        redundant_false_choices=bool(re.search(r'"(?:label_truncated|value_truncated|choices_truncated)"\s*:\s*false', human)),
                        current_guidance='supplied page_state' in human and 'next: "run greppy web observe' not in human,
                        choices_visible=all(x in human for x in ('ascending', 'descending', 'Low to high')))
        terminal['feedback'] = feedback
        terminal['rejected_actions_changed_nothing'] = fixture.load(run_id) == state_before_refusal
        assert terminal['rejected_actions_changed_nothing']
        # Recover using a runtime-provided ref even if human presentation fails,
        # so the business oracle is independent of the feedback-quality gate.
        call('click', fresh.group(1) if fresh else fresh_ref)
        call('fill', 'css=#reservation-quantity', '3')
        call('click', 'css=#reservation-form button[type=submit]')
        call('wait', 'css=#reservation-dialog[open]', '--absent', '--native', '--timeout', '5000')
        call('reload')
        call('assert', 'css=#reservation-status p')
        state = fixture.load(run_id)
        terminal['oracle'] = fixture.verify(state)
        (evidence / 'verified-state.json').write_text(json.dumps(state, indent=2))
        terminal['candidate_unchanged'] = capture(cli, runtime) == context['candidate']
        terminal['fixture_unchanged'] = hashes() == before
        assert terminal['oracle']['ok'] and terminal['candidate_unchanged'] and terminal['fixture_unchanged']
        assert feedback['one_content_block'] and feedback['readable_current_ref'] and feedback['choices_visible']
        assert not feedback['raw_snapshot_duplicated'] and not feedback['redundant_false_choices']
        assert not require_current_guidance or feedback['current_guidance']
        terminal['passed'] = True
    except BaseException as error:
        terminal['failure'] = repr(error)
    finally:
        try:
            terminal['runtime_stopped'] = call('runtime', 'stop')['result']['running'] is False
        except BaseException as error:
            terminal['cleanup_failure'] = repr(error)
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)
        terminal['http_stopped'] = not thread.is_alive()
        terminal['passed'] = terminal['passed'] and terminal.get('runtime_stopped', False) and terminal['http_stopped']
        (evidence / 'terminal.json').write_text(json.dumps(terminal, indent=2))
    print(json.dumps(terminal))
    return 0 if terminal['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cli', 'runtime', 'scratch', 'evidence'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--require-current-guidance', action='store_true')
    args = parser.parse_args()
    raise SystemExit(run(args.cli, args.runtime, args.scratch, args.evidence, args.require_current_guidance))
