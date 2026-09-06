"""Paired technical outcome-condition probe; never an agent/token benchmark.

Five alternating pairs, fresh context/data per run, identical frozen fixture.
Sort endpoint: first click on a ref from the filter receipt, without another read.
Confirmation endpoint: saved-result and dialog state in the submit receipt.
No failed action is retried. Failures remain results.
"""
import argparse
import hashlib
from http.server import HTTPServer
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from candidate import capture
from prepare_context import prepare


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def run(a):
    a.evidence.mkdir(parents=True, exist_ok=False)
    a.scratch.mkdir(parents=True, exist_ok=False)
    aliases = a.scratch / 'aliases'
    aliases.mkdir()
    sys.path.insert(0, str(a.fixture))
    spec = importlib.util.spec_from_file_location('frozen_fixture', a.fixture / 'server.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    fixture.set_dir(a.scratch / 'states')
    sources = [a.fixture / 'server.py', a.fixture / 'table_case.py',
               *sorted((a.fixture / 'static').glob('*'))]
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    source_before = hashes()
    binary_before = capture(a.cli, a.runtime)
    conditions = ({'weak': 'text=3 matching items',
                   'strong': 'css=#price-heading[aria-sort=ascending]'} if a.stage == 'sort'
                  else {'weak': 'text={target}', 'strong': 'css=#reservation-status p'})
    plan = dict(schema='greppy.table-expectation-probe.v1', repeats=a.repeats, stage=a.stage,
                seed='basic-development-table-1', fixture=source_before,
                candidate=binary_before, provider_tokens=None, agent_comparison=False,
                conditions=conditions,
                endpoint=('first click of returned cheapest-item reference; no intervening UI read'
                          if a.stage == 'sort' else
                          'confirmation receipt exposes saved result and closed dialog; one submit only'),
                recovery='none; failed first click is retained, not repaired',
                order=[dict(pair=p, condition=c) for p in range(1, a.repeats + 1)
                       for c in (('weak', 'strong') if p % 2 else ('strong', 'weak'))])
    write(a.evidence / 'plan.json', plan)
    http = HTTPServer(('127.0.0.1', 0), fixture.Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    trials = []
    terminal = dict(passed=False, trials=trials, provider_tokens=None, agent_comparison=False)
    try:
        for entry in plan['order']:
            name = f"{entry['pair']:02d}-{entry['condition']}"
            folder = a.evidence / name
            folder.mkdir()
            identity = 'expect-' + uuid.uuid4().hex[:12]
            ctx = prepare(a.scratch / 'contexts', a.cli, aliases, identity, identity,
                          runtime=a.runtime, view='compact')
            write(folder / 'context.json', ctx)
            run_id = uuid.uuid4().hex[:12]
            initial = fixture.new_state(run_id, plan['seed'], 'table')
            fixture.write_state(initial)
            write(folder / 'initial-state.json', initial)
            target = min((i for i in initial['facts']['inventory']
                          if i['region'] == 'EU' and i['available'] >= 3),
                         key=lambda i: i['unit_cents'])
            records = []
            trial = dict(**entry, run_id=run_id, target=target['name'], harness_ok=False)
            trials.append(trial)

            def call(*args):
                argv = ([ctx['alias'], 'web', 'do', '--json', *args[1:]] if args[0] == 'do'
                        else [ctx['alias'], 'web', *args, '--json'])
                record = dict(argv=argv)
                started = time.monotonic()
                try:
                    p = subprocess.run(argv, capture_output=True, text=True, timeout=60)
                    record.update(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr,
                                  stdout_bytes=len(p.stdout.encode()))
                    return p.returncode, json.loads(p.stdout)
                except BaseException as error:
                    record['harness_error'] = repr(error)
                    raise
                finally:
                    record['seconds'] = time.monotonic() - started
                    records.append(record)
                    write(folder / 'calls.json', records)

            try:
                code, opened = call('open', f'http://127.0.0.1:{http.server_port}/?run_id={run_id}')
                assert code == 0, opened
                code, filtered = call('do', '--native',
                    'select', 'css=#inventory-region', 'EU', '::',
                    'check', 'css=#inventory-capacity', '::',
                    'select', 'css=#inventory-order', 'ascending', '--expect',
                    (plan['conditions'][entry['condition']] if a.stage == 'sort' else
                     'css=#price-heading[aria-sort=ascending]'), '--expect-timeout', '3000')
                trial['filter_exit'] = code
                if code == 0:
                    snap = filtered['result']['page_state']['snapshot']
                    matches = [n for n in snap['actionables'] if n.get('name') == 'Reserve ' + target['name']]
                    assert len(matches) == 1, matches
                    ref = matches[0]['ref']
                    # This is the immediate next browser call. No diagnostic read first.
                    code, clicked = call('click', ref)
                    trial.update(first_click_exit=code, first_click_ok=code == 0,
                                 returned_ref=ref, stale_ref='STALE_REF' in json.dumps(clicked),
                                 filter_snapshot_text=snap['text'],
                                 expectation=filtered['result']['steps'][-1].get('expectation'))
                    if code == 0:
                        scope = clicked['result']['page_state']['snapshot']['working_scope']
                        trial['correct_modal'] = scope.get('kind') == 'modal' and scope.get('name') == 'Reserve ' + target['name']
                    else:
                        trial['correct_modal'] = False
                if a.stage == 'confirmation':
                    assert trial.get('correct_modal'), 'confirmation setup did not reach correct modal'
                    condition = plan['conditions'][entry['condition']].format(target=target['name'])
                    code, saved = call('do', '--native', 'fill', 'css=#reservation-quantity', '3',
                        '::', 'click', 'css=#reservation-form button[type=submit]',
                        '--expect', condition, '--expect-timeout', '3000')
                    trial['confirmation_exit'] = code
                    trial['confirmation_expectation'] = saved['result']['steps'][-1].get('expectation')
                    final_snap = saved['result']['page_state']['snapshot']
                    trial['confirmation_snapshot_text'] = final_snap['text']
                    trial['old_modal_returned'] = final_snap['working_scope'].get('kind') == 'modal'
                    trial['saved_result_returned'] = ('Reserved 3 × ' + target['name'] + '.') in final_snap['text']
                    immediate = fixture.load(run_id)
                    write(folder / 'immediate-server-state.json', immediate)
                    trial['immediate_oracle'] = fixture.verify(immediate)
                    # One read-only reload for persistence verification, never resubmit.
                    code, reloaded = call('reload')
                    assert code == 0, reloaded
                    trial['reloaded_result_visible'] = ('Reserved 3 × ' + target['name'] + '.') in reloaded['result']['page_state']['snapshot']['text']
                state = fixture.load(run_id)
                write(folder / 'server-state.json', state)
                trial['server_filters_applied'] = (state['values']['region'] == 'EU' and
                    state['values']['capacity_only'] is True and state['values']['price_order'] == 'ascending')
                trial['server_mutations'] = [e['action'] for e in state['events']]
                if a.stage == 'confirmation':
                    trial['final_oracle'] = fixture.verify(state)
                trial['harness_ok'] = True
            except BaseException as error:
                trial['harness_failure'] = repr(error)
            finally:
                try:
                    code, stopped = call('runtime', 'stop')
                    trial['runtime_stopped'] = code == 0 and stopped['result']['running'] is False
                except BaseException as error:
                    trial['cleanup_failure'] = repr(error)
                write(folder / 'result.json', trial)
                write(a.evidence / 'progress.json', trials)
                print(json.dumps({k:v for k,v in trial.items() if k not in (
                    'filter_snapshot_text', 'expectation', 'confirmation_snapshot_text',
                    'confirmation_expectation')}), flush=True)
            if not trial['harness_ok'] or not trial.get('runtime_stopped'):
                raise RuntimeError('harness or cleanup failure; remaining trials not started')
        terminal['candidate_unchanged'] = binary_before == capture(a.cli, a.runtime)
        terminal['fixture_unchanged'] = source_before == hashes()
        terminal['passed'] = terminal['candidate_unchanged'] and terminal['fixture_unchanged']
        endpoints = ('first_click_ok', 'stale_ref', 'correct_modal', 'server_filters_applied')
        if a.stage == 'confirmation':
            endpoints += ('old_modal_returned', 'saved_result_returned', 'reloaded_result_visible')
        terminal['counts'] = {c: {k: sum(bool(t.get(k)) for t in trials if t['condition'] == c)
                                  for k in endpoints}
                              for c in ('weak', 'strong')}
    except BaseException as error:
        terminal['failure'] = repr(error)
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)
        terminal['http_stopped'] = not thread.is_alive()
        terminal['passed'] = terminal['passed'] and terminal['http_stopped']
        write(a.evidence / 'terminal.json', terminal)
    print(json.dumps({k:v for k,v in terminal.items() if k != 'trials'}))
    return 0 if terminal['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cli', 'runtime', 'fixture', 'scratch', 'evidence'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--stage', choices=('sort', 'confirmation'), default='sort')
    args = p.parse_args()
    if args.repeats < 1:
        p.error('--repeats must be positive')
    raise SystemExit(run(args))
