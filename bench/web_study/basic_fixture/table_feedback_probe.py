"""Native table-flow preflight; deterministic reproduction, never an agent trial."""
import argparse
import hashlib
from http.server import HTTPServer
import json
from pathlib import Path
import subprocess
import re
import threading
import time
import uuid
from candidate import capture
from prepare_context import prepare
import server as fixture


def run(cli, runtime, scratch, evidence, delivery='separate', require_compact_feedback=False):
    if require_compact_feedback and delivery != 'chain':
        raise ValueError('compact feedback gate requires human chain delivery')
    evidence.mkdir(parents=True, exist_ok=False)
    aliases = scratch / 'aliases'
    aliases.mkdir(parents=True)
    identity = 'table-feedback-' + uuid.uuid4().hex[:12]
    context = prepare(scratch / 'contexts', cli, aliases, identity, identity, runtime=runtime,
                      view='compact', chain_view='compact')
    (evidence / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
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
    terminal = dict(schema='greppy.table-feedback-probe.v1', passed=False, run_id=run_id, delivery=delivery,
                    scope='Deterministic native functional preflight; no Luna or provider-token comparison',
                    fixture_before=before)
    def call(*args):
        argv = [context['alias'], 'web', *args] if args[0] == 'do' else [context['alias'], 'web', *args, '--json']
        record = dict(argv=argv)
        started = time.monotonic()
        try:
            p = subprocess.run(argv, text=True, capture_output=True, timeout=60)
            record.update(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except subprocess.TimeoutExpired as error:
            record.update(timeout=True, stdout=str(error.stdout), stderr=str(error.stderr))
            raise
        finally:
            record['wall_seconds'] = time.monotonic() - started
            calls.append(record)
            (evidence / 'calls.json').write_text(json.dumps(calls, indent=2) + '\n')
        if args[0] == 'do':
            assert p.returncode == 0 and p.stdout.rstrip().endswith('chain: 2/2 steps executed, 0 failed'), (p.returncode, p.stdout)
            terminal['human_chain_terminal_verified'] = True
            text = p.stdout
            terminal['feedback'] = {
                'stdout_bytes': len(text.encode()),
                'page_content_blocks': text.splitlines().count('UNTRUSTED_PAGE_CONTENT'),
                'obsolete_confirmation_exposed': 'No reservations yet.' in text or 'name="Confirm reservation"' in text,
                'redundant_false_choice_flags': len(re.findall(r'"(?:label_truncated|value_truncated|choices_truncated)"\s*:\s*false', text)),
                'current_reservation_visible': 'Reserved 3 × Ember.' in text,
                'choice_values_visible': all(value in text for value in ('ascending', 'descending', 'none', 'All regions', 'Low to high', 'High to low')),
                'scope': 'Frozen fixture human-output checks and actual bytes; never provider tokens',
            }
            f = terminal['feedback']
            terminal['compact_feedback_passed'] = (f['page_content_blocks'] == 1 and
                not f['obsolete_confirmation_exposed'] and f['redundant_false_choice_flags'] == 0 and
                f['current_reservation_visible'] and f['choice_values_visible'])
            return None
        reply = json.loads(p.stdout)
        assert p.returncode == 0 and reply.get('status') == 'ok', (p.returncode, reply)
        return reply
    try:
        call('open', f'http://127.0.0.1:{http.server_port}/?run_id={run_id}')
        call('select', 'css=#inventory-region', 'EU')
        call('check', 'css=#inventory-capacity')
        call('select', 'css=#inventory-order', 'ascending')
        call('click', 'css=#inventory-table tbody tr:first-child button')
        call('fill', 'css=#reservation-quantity', '3')
        if delivery == 'chain':
            call('do', 'click', 'css=#reservation-form button[type=submit]', '::',
                 'wait', 'css=#reservation-dialog[open]', '--absent', '--native', '--timeout', '5000')
        else:
            call('click', 'css=#reservation-form button[type=submit]')
            # Capture truth independently, without driving or retrying the UI.
            immediate = fixture.load(run_id)
            (evidence / 'immediate-state.json').write_text(json.dumps(immediate, indent=2) + '\n')
            terminal['immediate_oracle'] = fixture.verify(immediate)
            held = call('wait', 'css=#reservation-dialog[open]', '--absent', '--native', '--timeout', '5000')
            assert held['result']['held'] is True, held
        call('assert', 'css=#reservation-status p')
        call('reload')
        call('observe')
        call('assert', 'css=#reservation-status p')
        state = fixture.load(run_id)
        (evidence / 'verified-state.json').write_text(json.dumps(state, indent=2) + '\n')
        terminal['oracle'] = fixture.verify(state)
        terminal['candidate_unchanged'] = capture(cli, runtime) == context['candidate']
        terminal['fixture_unchanged'] = hashes() == before
        terminal['confirm_calls'] = sum('css=#reservation-form button[type=submit]' in c['argv'] for c in calls)
        assert terminal['oracle']['ok'] and terminal['confirm_calls'] == 1, terminal
        assert terminal['candidate_unchanged'] and terminal['fixture_unchanged'], terminal
        if require_compact_feedback:
            assert terminal.get('compact_feedback_passed') is True, 'functional flow passed but compact feedback did not'
        terminal['passed'] = True
    except BaseException as error:
        terminal['failure'] = repr(error)
    finally:
        try:
            stopped = call('runtime', 'stop')
            terminal['runtime_stopped'] = stopped['result']['running'] is False
        except BaseException as error:
            terminal['cleanup_failure'] = repr(error)
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)
        terminal['http_stopped'] = not thread.is_alive()
        terminal['passed'] = terminal['passed'] and terminal.get('runtime_stopped', False) and terminal['http_stopped']
        (evidence / 'terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
    print(json.dumps({k: v for k, v in terminal.items() if k != 'fixture_before'}))
    return 0 if terminal['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cli', 'runtime', 'scratch', 'evidence'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--delivery', choices=('separate', 'chain'), default='separate')
    parser.add_argument('--require-compact-feedback', action='store_true')
    a = parser.parse_args()
    raise SystemExit(run(a.cli, a.runtime, a.scratch, a.evidence, a.delivery, a.require_compact_feedback))
