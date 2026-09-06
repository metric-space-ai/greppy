"""Native scope byte/correctness probe, explicitly not agent-token acceptance."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
import uuid
from candidate import capture
from prepare_context import prepare

NOISE = ''.join(f'<section><h2>OTHER_APP_{i}</h2><button>Unrelated action {i}</button><p>Unrelated restored application content {i}</p></section>' for i in range(60))
PAGE = ('<!doctype html><html><head><title>Scope proof</title></head><body>' + NOISE + '''<section id="work"><h2>Own worksheet</h2><label for="cell">Cell A2</label><input id="cell" value="ORIGINAL"><button id="save" onclick="document.querySelector('#status').textContent='SAVED_ONCE';this.dataset.count=String(Number(this.dataset.count||0)+1)">Save own worksheet</button><p id="status">UNSAVED</p></section></body></html>''').encode()

class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)
    def log_message(self, *_args):
        pass

def run(cli, runtime, scratch, evidence):
    evidence.mkdir(parents=True, exist_ok=False)
    aliases = scratch / 'aliases'
    aliases.mkdir(parents=True, exist_ok=True)
    identity = 'scope-' + uuid.uuid4().hex[:12]
    context = prepare(scratch / 'contexts', cli, aliases, identity, identity, runtime=runtime)
    (evidence / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Page)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    calls, checks, pairs = [], [], []
    terminal = {'schema': 'greppy.scoped-observe-probe.v1', 'passed': False,
                'agent_token_acceptance': False, 'provider_tokens': None,
                'measurement': 'UTF-8 stdout bytes, not tokens; prepared native probe',
                'checks': checks, 'pairs': pairs}
    def call(*args, json_output=True, allow_error=False):
        argv = [context['alias'], 'web', *args] + (['--json'] if json_output else [])
        started = time.monotonic()
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=90)
            record = dict(argv=argv, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr,
                          bytes=len(p.stdout.encode()), seconds=time.monotonic()-started)
        except subprocess.TimeoutExpired as error:
            calls.append(dict(argv=argv, timeout=True, stdout=str(error.stdout), stderr=str(error.stderr)))
            (evidence / 'calls.json').write_text(json.dumps(calls, indent=2) + '\n')
            raise
        calls.append(record)
        (evidence / 'calls.json').write_text(json.dumps(calls, indent=2) + '\n')
        if not allow_error:
            assert p.returncode == 0, record
        return json.loads(p.stdout) if json_output else record
    def check(name, condition):
        assert condition, name
        checks.append(name)
    try:
        call('open', f'http://127.0.0.1:{server.server_port}/')
        whole = call('observe')['result']
        full_refs = {n['name']: n['ref'] for n in whole['actionables']}
        scoped = call('observe', 'css=#work')['result']
        check('scope excludes other applications', 'OTHER_APP_' not in json.dumps(scoped))
        check('scope includes current form and status', all(s in json.dumps(scoped) for s in ['ORIGINAL','UNSAVED','Cell A2']))
        check('explicit scope evidence', scoped['observation_scope']['query'] == 'css=#work' and scoped['observation_scope']['roots_returned'] == 1)
        refs = {n['name']: n['ref'] for n in scoped['actionables']}
        check('same nodes keep references across scopes', refs['Cell A2'] == full_refs['Cell A2'] and refs['Save own worksheet'] == full_refs['Save own worksheet'])
        missing = call('observe', 'css=#absent', allow_error=True)
        check('no match is explicit without broad fallback', calls[-1]['exit_code'] != 0 and missing['error']['code'] == 'NO_MATCH' and 'OTHER_APP_' not in json.dumps(missing))
        call('fill', refs['Cell A2'], 'EDITED')
        current = call('inspect', refs['Cell A2'])['result']['value']['node']
        check('prior reference still addresses current field', current['value'] == 'EDITED')
        call('click', refs['Save own worksheet'])
        saved = call('inspect', 'css=#save', '--attrs')['result']['value']['node']
        check('exactly one save event', saved['attrs']['data-count'] == '1')
        for i in range(5):
            outputs = {}
            for label in (('full', 'scoped') if i % 2 == 0 else ('scoped', 'full')):
                args = ('observe',) if label == 'full' else ('observe', 'css=#work')
                outputs[label] = call(*args, json_output=False)
            check(f'pair {i+1} scoped human output preserves relevant state', all(s in outputs['scoped']['stdout'] for s in ['EDITED','SAVED_ONCE']))
            check(f'pair {i+1} excludes unrelated text', 'OTHER_APP_' not in outputs['scoped']['stdout'])
            pairs.append({'pair': i+1, 'full_bytes': outputs['full']['bytes'], 'scoped_bytes': outputs['scoped']['bytes'],
                          'ratio': outputs['scoped']['bytes']/outputs['full']['bytes'],
                          'full_seconds': outputs['full']['seconds'], 'scoped_seconds': outputs['scoped']['seconds']})
        check('every scoped response smaller', all(p['ratio'] < 1 for p in pairs))
        check('candidate unchanged', capture(cli, runtime) == context['candidate'])
        terminal['median_byte_reduction_percent'] = 100*(1-statistics.median(p['ratio'] for p in pairs))
        terminal['passed'] = True
    except BaseException as error:
        terminal['failure'] = repr(error)
        raise
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        terminal['http_thread_stopped'] = not thread.is_alive()
        try:
            stopped = call('runtime', 'stop')
            terminal['runtime_stop'] = stopped
        except BaseException as error:
            terminal['cleanup_failure'] = repr(error)
            terminal['passed'] = False
        (evidence / 'terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
    assert terminal['passed'], terminal
    print(json.dumps(terminal))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cli','runtime','scratch','evidence'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    run(a.cli, a.runtime, a.scratch, a.evidence)
