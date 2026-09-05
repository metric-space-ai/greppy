"""Native CLI regression proof, not an agent efficiency measurement.

Requires explicit executable paths. No missing-runtime skip, no test-data reset,
no retries of failed browser commands. Each call is retained before assertions.
"""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import time
import uuid

from candidate import capture
from prepare_context import prepare


PAGE = b'''<!doctype html><html><body>
<label for="quantity">Quantity</label>
<input id="quantity" type="number" value="1" disabled data-proof="original">
<button id="replace" onclick="const old = document.getElementById('quantity');
const next = old.cloneNode(true); next.value = '3'; next.disabled = false;
next.setAttribute('data-proof', 'replacement'); old.replaceWith(next)">Replace field</button>
</body></html>'''


class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)

    def log_message(self, *_args):
        pass


def run_probe(cli: Path, runtime: Path, scratch: Path, evidence: Path):
    evidence.mkdir(parents=True, exist_ok=False)
    alias_dir = scratch / 'aliases'
    alias_dir.mkdir(parents=True, exist_ok=True)
    identity = 'inspect-' + uuid.uuid4().hex[:12]
    context = prepare(scratch / 'contexts', cli, alias_dir, identity, identity,
                      runtime=runtime)
    (evidence / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    records = []
    checks = []

    def call(*args, expected_exit=0):
        started = time.monotonic()
        argv = [context['alias'], 'web', *args, '--json']
        try:
            result = subprocess.run(argv, text=True, capture_output=True, timeout=90)
            record = dict(argv=argv, exit_code=result.returncode,
                          stdout=result.stdout, stderr=result.stderr,
                          wall_seconds=time.monotonic() - started)
        except subprocess.TimeoutExpired as error:
            record = dict(argv=argv, timeout=True,
                          stdout=str(error.stdout), stderr=str(error.stderr),
                          wall_seconds=time.monotonic() - started)
            records.append(record)
            (evidence / 'calls.json').write_text(json.dumps(records, indent=2) + '\n')
            raise
        records.append(record)
        (evidence / 'calls.json').write_text(json.dumps(records, indent=2) + '\n')
        assert result.returncode == expected_exit, record
        response = json.loads(result.stdout)
        assert response['status'] == ('ok' if expected_exit == 0 else 'error'), response
        return response

    def check(name, condition):
        assert condition, name
        checks.append(name)

    terminal = dict(schema='greppy.web-inspect-proof.v1', passed=False,
                    efficiency_acceptance=False, checks=checks)
    try:
        call('open', f'http://127.0.0.1:{server.server_port}/')
        observed = call('observe')['result']
        matches = [node for node in observed['actionables'] if node.get('name') == 'Quantity']
        assert len(matches) == 1, observed
        ref = matches[0]['ref']
        tabs = call('tab', 'list')['result']['tabs']
        assert len(tabs) == 1, tabs
        original_tab = tabs[0]['tab']
        first = call('inspect', ref, '--tab', original_tab, '--attrs', '--html')
        check('native operation, no evaluate fallback', first['operation'] == 'web.inspect')
        value = first['result']['value']
        check('disabled original node read', value['node']['disabled'] is True and
              value['node']['value'] == '1' and value['node']['attrs']['data-proof'] == 'original')
        check('HTML and node shape retained', value['count'] == 1 and 'quantity' in value['html'])
        call('tab', 'new')
        tabs = call('tab', 'list')['result']['tabs']
        other_tabs = [item['tab'] for item in tabs if item['tab'] != original_tab]
        assert len(other_tabs) == 1, tabs
        again = call('inspect', ref, '--tab', original_tab)['result']['value']['node']
        check('explicit original tab survives active-tab switch', again['value'] == '1')
        query = call('inspect', 'css=#quantity', '--tab', original_tab)['result']['value']['node']
        check('query path also respects explicit tab', query['value'] == '1')
        present = call('assert', 'css=#quantity', '--tab', original_tab)
        check('assert uses selected original tab', present['result']['held'] is True)
        missing = call('assert', 'css=#quantity', '--tab', other_tabs[0], expected_exit=18)
        check('assert on other tab does not prove presence', missing['result']['held'] is False)
        waited = call('wait', 'css=#quantity', '--tab', original_tab, '--timeout', '1000')
        check('wait uses selected original tab', waited['result']['held'] is True)
        absent = call('wait', 'css=#quantity', '--absent', '--tab', original_tab,
                      '--timeout', '150', expected_exit=13)
        check('present field cannot satisfy absent wait', absent['result']['held'] is False)
        wrong = call('inspect', ref, '--tab', other_tabs[0], expected_exit=34)
        check('wrong-tab identity refused', wrong['error']['code'] == 'STALE_REF')
        call('click', 'css=#replace', '--tab', original_tab)
        stale = call('inspect', ref, '--tab', original_tab, expected_exit=34)
        check('replaced node refused without rebinding', stale['error']['code'] == 'STALE_REF')
        replacement = call('inspect', 'css=#quantity', '--tab', original_tab,
                           '--attrs')['result']['value']['node']
        check('replacement has independent current state', replacement['value'] == '3' and
              replacement['disabled'] is False and replacement['attrs']['data-proof'] == 'replacement')
        after = capture(cli, runtime)
        check('candidate executable bytes unchanged', after == context['candidate'])
        terminal['passed'] = True
    except BaseException as error:
        terminal['failure'] = repr(error)
        raise
    finally:
        server.shutdown()
        server.server_close()
        # Only this proof's unique runtime is stopped. Cleanup cannot turn a
        # failed test into a pass or erase its original call/timeout evidence.
        try:
            call('runtime', 'stop')
        except BaseException as error:
            terminal['cleanup_failure'] = repr(error)
            terminal['passed'] = False
        (evidence / 'terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
    assert terminal['passed'], terminal
    print(json.dumps(terminal))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cli', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    a = p.parse_args()
    run_probe(a.cli, a.runtime, a.scratch, a.evidence)


if __name__ == '__main__':
    main()
