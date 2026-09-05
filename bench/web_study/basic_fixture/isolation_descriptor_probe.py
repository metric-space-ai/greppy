"""Native owner-isolation and descriptor DOM probe; no efficiency claim.

Uses unchanged candidate binaries. The new descriptor is evaluated explicitly;
this does not prove its integration into a newly compiled web.inspect.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid
from candidate import capture
from prepare_context import prepare


class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        marker = 'Owner A' if self.path == '/a' else 'Owner B'
        body = ('''<!doctype html><title>''' + marker + '''</title><h1>''' + marker + '''</h1>
<label for="order">Price order</label><select id="order">
<option value="">Unsorted</option><option value="ascending">Low to high</option>
<optgroup label="Unavailable" disabled><option value="descending">High to low</option></optgroup>
</select>''').encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *_args):
        pass


def run(cli, runtime, scratch, evidence):
    evidence.mkdir(parents=True, exist_ok=False)
    aliases = scratch / 'aliases'
    aliases.mkdir(parents=True)
    group = 'isolation-' + uuid.uuid4().hex[:12]
    contexts = [prepare(scratch / 'contexts', cli, aliases, 'probe-' + side,
                        group, runtime=runtime) for side in ('a', 'b')]
    (evidence / 'contexts.json').write_text(json.dumps(contexts, indent=2) + '\n')
    source = Path(__file__).resolve().parents[3] / 'crates/web-client/src'
    helper = (source / 'select-choices.js').read_text()
    descriptor = (source / 'describe-node.js').read_text()
    expression = '(function(){\n' + helper + '\nreturn (' + descriptor + ')(document.querySelector("select"), false);})()'
    (evidence / 'descriptor.js').write_text(expression)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Page)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    records, checks = [], []
    terminal = dict(schema='greppy.owner-descriptor-probe.v1', passed=False,
                    efficiency_acceptance=False, compiled_inspect_integration=False,
                    descriptor_sha256=hashlib.sha256(expression.encode()).hexdigest(), checks=checks)

    def call(context, *args):
        argv = [context['alias'], 'web', *args, '--json']
        started = time.monotonic()
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=90)
            record = dict(argv=argv, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except subprocess.TimeoutExpired as error:
            record = dict(argv=argv, timeout=True, stdout=str(error.stdout), stderr=str(error.stderr))
            raise
        finally:
            record['wall_seconds'] = time.monotonic() - started
            records.append(record)
            (evidence / 'calls.json').write_text(json.dumps(records, indent=2) + '\n')
        return p.returncode, json.loads(p.stdout)

    def ok(context, *args):
        code, reply = call(context, *args)
        assert code == 0 and reply['status'] == 'ok', (code, reply)
        return reply

    def check(name, condition):
        assert condition, name
        checks.append(name)

    try:
        check('distinct runtime owners', contexts[0]['runtime_id'] != contexts[1]['runtime_id'])
        sessions = []
        for context, side in zip(contexts, ('a', 'b')):
            ok(context, 'open', f'http://127.0.0.1:{server.server_port}/{side}')
            listed = ok(context, 'session', 'list')
            ids = set(re.findall(r'wrs_[a-zA-Z0-9]+', json.dumps(listed['result'])))
            check('one session in owner ' + side, len(ids) == 1)
            sessions.append(next(iter(ids)))
            initial = ok(context, "observe")
            check("implicit own context before foreign access " + side,
                  initial["result"]["url"] == f"http://127.0.0.1:{server.server_port}/{side}")
        check('distinct native sessions', sessions[0] != sessions[1])
        for own, foreign in ((0, 1), (1, 0)):
            code, refused = call(contexts[own], 'observe', '--session', sessions[foreign])
            check('foreign session refused by owner ' + str(own), code != 0 and
                  refused.get('status') == 'error' and
                  refused.get('error', {}).get('code', '').lower() in
                  ('session_not_found', 'session_not_owned', 'session_unknown', 'not_found'))
        implicit_preserved = []
        for context, side, session in zip(contexts, ('a', 'b'), sessions):
            code, implicit = call(context, 'observe')
            implicit_preserved.append(code == 0 and implicit.get('status') == 'ok')
            page = ok(context, 'observe', '--session', session)
            check('explicit own page survives foreign access ' + side,
                  page['result']['url'] == f'http://127.0.0.1:{server.server_port}/{side}')
        terminal['implicit_context_preserved'] = implicit_preserved
        result = ok(contexts[0], 'js', expression, '--session', sessions[0])['result']['value']

        choices = result['select_choices']
        check('native DOM option mapping', [(c['value'], c['label'], c['disabled']) for c in choices['choices']] ==
              [('', 'Unsorted', False), ('ascending', 'Low to high', False), ('descending', 'High to low', True)])
        check('native DOM descriptor schema and state', choices['schema'] == 'greppy.web.select-choices.v1' and
              choices['choices_total'] == 3 and choices['choices_truncated'] is False and result['value'] == '')
        check('candidate bytes unchanged', capture(cli, runtime) == contexts[0]['candidate'])
        check('implicit current sessions preserved', all(implicit_preserved))
        terminal['passed'] = True
    except BaseException as error:
        terminal['failure'] = repr(error)
        raise
    finally:
        terminal['cleanup'] = []
        for context in contexts:
            try:
                reply = ok(context, 'runtime', 'stop')
                terminal['cleanup'].append(dict(owner=context['runtime_id'], response=reply))
            except BaseException as error:
                terminal['cleanup'].append(dict(owner=context['runtime_id'], failure=repr(error)))
                terminal['passed'] = False
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        terminal['http_thread_stopped'] = not server_thread.is_alive()
        terminal['passed'] = terminal['passed'] and terminal['http_thread_stopped']
        (evidence / 'terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
    assert terminal['passed'], terminal
    print(json.dumps(dict(passed=True, checks=checks, evidence=str(evidence))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cli', 'runtime', 'scratch', 'evidence'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    run(a.cli, a.runtime, a.scratch, a.evidence)


if __name__ == '__main__':
    main()
