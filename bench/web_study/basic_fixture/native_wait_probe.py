"""Native CLI integration probe, separate from agent efficiency measurements."""
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

PAGE = b'''<!doctype html><title>Native wait probe</title>
<button id="start" onclick="setTimeout(function(){ const n=document.createElement('p'); n.id='late'; n.textContent='Ready'; document.body.append(n); },1000)">Load later</button>
<button id="clear" onclick="document.getElementById('late').remove()">Clear result</button>
<label for="value">Quantity</label><input id="value" value="1">
<button id="replace" onclick="const n=document.getElementById('value'); n.replaceWith(n.cloneNode(true))">Replace field</button>
<label for="order">Order</label><select id="order"><option value="">Unsorted</option><option value="ascending">Low to high</option></select>
<button id="navigate" onclick="setTimeout(function(){ location.href='/landed'; },2000)">Navigate later</button>
'''


class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"<!doctype html><title>Landed</title><h1 id=landed>Navigation completed</h1>" if self.path == "/landed" else PAGE
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
    identity = 'native-wait-' + uuid.uuid4().hex[:12]
    context = prepare(scratch / 'contexts', cli, aliases, identity, identity, runtime=runtime)
    (evidence / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Page)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    calls, checks = [], []
    terminal = dict(schema='greppy.native-wait-cli-proof.v1', passed=False,
                    efficiency_acceptance=False, checks=checks)

    def call(*args, expected=0):
        argv = [context['alias'], 'web', *args, '--json']
        started = time.monotonic()
        record = dict(argv=argv)
        try:
            result = subprocess.run(argv, text=True, capture_output=True, timeout=60)
            record.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired as error:
            record.update(timeout=True, stdout=str(error.stdout), stderr=str(error.stderr))
            raise
        finally:
            record['wall_seconds'] = time.monotonic() - started
            calls.append(record)
            (evidence / 'calls.json').write_text(json.dumps(calls, indent=2) + '\n')
        assert result.returncode == expected, record
        reply = json.loads(result.stdout)
        assert reply['status'] == ('ok' if expected == 0 else 'error'), reply
        return reply

    def check(name, condition):
        assert condition, name
        checks.append(name)

    try:
        call('open', f'http://127.0.0.1:{server.server_port}/')
        tab = call('tab', 'list')['result']['tabs'][0]['tab']
        initial = call('observe')['result']
        field = [node for node in initial['actionables'] if node.get('name') == 'Quantity']
        assert len(field) == 1, initial
        original_ref = field[0]['ref']
        choices = [node for node in initial['actionables'] if node.get('name') == 'Order']
        check('initial Observe has usable option values', len(choices) == 1 and
              choices[0]['select_choices']['choices'][1]['value'] == 'ascending')
        inspected = call('inspect', 'css=#order')['result']['value']['node']
        check('compiled native Inspect shares choices', inspected['select_choices']['choices'][1]['value'] == 'ascending')
        timeout = call('wait', 'css=#late', '--native', '--timeout', '80', expected=13)
        check('native timeout keeps typed error and false verdict', timeout['error']['code'] == 'TIMEOUT' and
              timeout['result']['held'] is False and timeout['result']['wait_backend'] == 'native_v1')
        call('assert', 'css=#start', '--tab', tab)
        check('timeout preserves original page', True)
        call('click', 'css=#start', '--tab', tab)
        ready = call('wait', 'css=#late', '--native', '--timeout', '5000', '--tab', tab)
        check('native wait confirms delayed DOM state', ready['result']['held'] is True and
              ready['result']['wait_backend'] == 'native_v1' and ready['operation'] == 'web.wait')
        call('assert', 'css=#late', '--tab', tab)
        call('tab', 'new')
        tabs = call('tab', 'list')['result']['tabs']
        other = [item['tab'] for item in tabs if item['tab'] != tab]
        assert len(other) == 1, tabs
        inactive = call('wait', 'css=#late', '--native', '--tab', tab, '--timeout', '1000')
        check('native wait honors inactive explicit tab', inactive['result']['held'] is True)
        missing = call('wait', 'css=#late', '--native', '--tab', other[0], '--timeout', '80', expected=13)
        check('other tab cannot satisfy original condition', missing['result']['held'] is False)
        call('click', 'css=#clear', '--tab', tab)
        absent = call('wait', 'css=#late', '--absent', '--native', '--tab', tab, '--timeout', '1000')
        check('native absence requires a valid false condition', absent['result']['held'] is True)
        call('click', 'css=#replace', '--tab', tab)
        stale = call('wait', original_ref, '--absent', '--native', '--tab', tab, '--timeout', '1000', expected=34)
        check('stale reference cannot prove absence', stale['error']['code'] == 'STALE_REF')
        call('assert', 'css=#value', '--tab', tab)
        check('stale-ref failure preserves usable page', True)
        call('click', 'css=#navigate', '--tab', tab)
        landed_url = f'http://127.0.0.1:{server.server_port}/landed'
        navigated = call('wait', '--url', landed_url, '--native', '--tab', tab, '--timeout', '5000')
        check('URL wait survives a later document navigation', navigated['result']['held'] is True)
        call('assert', 'css=#landed', '--tab', tab)
        check('candidate bytes unchanged', capture(cli, runtime) == context['candidate'])
        terminal['passed'] = True
    except BaseException as error:
        terminal['failure'] = repr(error)
        raise
    finally:
        try:
            cleanup = call('runtime', 'stop')
            terminal['cleanup'] = cleanup
            terminal['passed'] = terminal['passed'] and cleanup['result']['running'] is False
        except BaseException as error:
            terminal.update(passed=False, cleanup_failure=repr(error))
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        terminal['http_thread_stopped'] = not thread.is_alive()
        terminal['passed'] = terminal['passed'] and terminal['http_thread_stopped']
        (evidence / 'terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
    assert terminal['passed'], terminal
    print(json.dumps(dict(passed=True, checks=checks, evidence=str(evidence))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cli', 'runtime', 'scratch', 'evidence'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    run(a.cli, a.runtime, a.scratch, a.evidence)
