"""Capture real Web producer/consumer contracts; never an agent-token benchmark."""
from __future__ import annotations
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import time
import uuid
from candidate import capture
from prepare_context import prepare

PAGE = b'''<!doctype html><html><head><title>Composition contract</title></head><body>
<main id="work"><h1>Available delivery options</h1>
<a class="offer" href="#economy" data-price="7">Economy delivery</a>
<a class="offer" href="#express" data-price="19">Express delivery</a>
<a class="offer" href="#premium" data-price="29">Premium delivery</a>
<a class="offer" href="#hidden" data-price="1" hidden>Hidden delivery</a>
<label for="note">Delivery note</label><input id="note" value="ORIGINAL">
<button id="save" onclick="document.querySelector('#state').textContent='SAVED';this.dataset.count=String(Number(this.dataset.count||0)+1)">Save delivery</button>
<p id="state">UNSAVED</p></main></body></html>'''

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
    identity = 'compose-' + uuid.uuid4().hex[:12]
    context = prepare(scratch / 'contexts', cli, aliases, identity, identity, runtime=runtime)
    (evidence / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Page)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    calls = []
    terminal = {'schema': 'greppy.web-composition-probe.v1', 'completed': False,
                'agent_token_acceptance': False, 'provider_tokens': None,
                'fixture_sha256': hashlib.sha256(PAGE).hexdigest(),
                'candidate_is_main_union': False,
                'candidate_scope': 'historically tested scope pair; composition development only'}
    def call(label, *args, stdin=None, required=True):
        argv = [context['alias'], 'web', *args]
        started = time.monotonic()
        try:
            p = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=60)
            record = dict(label=label, argv=argv, stdin=stdin, exit_code=p.returncode,
                          stdout=p.stdout, stderr=p.stderr, stdout_bytes=len(p.stdout.encode()),
                          seconds=time.monotonic()-started)
        except subprocess.TimeoutExpired as error:
            record = dict(label=label, argv=argv, stdin=stdin, timeout=True,
                          stdout=repr(error.stdout), stderr=repr(error.stderr))
            calls.append(record)
            (evidence / 'calls.json').write_text(json.dumps(calls, indent=2) + '\n')
            raise
        calls.append(record)
        (evidence / 'calls.json').write_text(json.dumps(calls, indent=2) + '\n')
        print('PROBE ' + json.dumps({'label': label, 'exit_code': p.returncode,
                                    'bytes': record['stdout_bytes']}), flush=True)
        if required and p.returncode:
            raise RuntimeError(f'{label} failed with exit {p.returncode}; see calls.json')
        return record
    try:
        call('open', 'open', f'http://127.0.0.1:{server.server_port}/', '--json')
        for label, args in [
            ('observe', ['observe', 'css=#work', '--json']),
            ('find', ['find', 'css=a.offer', '--json']),
            ('extract', ['extract', 'css=a.offer', '--fields', 'text,href,attr:data-price', '--json']),
        ]:
            producer = call(label, *args)
            # This intentionally checks composition without a bespoke JSON flattener.
            # A no-match is a captured contract result, not automatically a product bug.
            call(label + '-direct-match', 'match', 'text~/Economy/',
                 stdin=producer['stdout'], required=False)
        control = '\n'.join(json.dumps(r) for r in [
            {'text': 'Economy delivery', 'price': 7, 'visible': True},
            {'text': 'Express delivery', 'price': 19, 'visible': True},
            {'text': 'Hidden delivery', 'price': 1, 'visible': False},
        ]) + '\n'
        call('jsonl-control', 'match', 'text~/delivery/ visible=true price<10', stdin=control)
        terminal['candidate_unchanged'] = capture(cli, runtime) == context['candidate']
        if not terminal['candidate_unchanged']:
            raise RuntimeError('candidate bytes changed during probe')
        terminal['completed'] = True
    except BaseException as error:
        terminal['failure'] = repr(error)
        raise
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        terminal['http_thread_stopped'] = not thread.is_alive()
        try:
            stop = call('stop-owned-runtime', 'runtime', 'stop', '--json')
            terminal['runtime_stop'] = stop
        except BaseException as error:
            terminal['cleanup_failure'] = repr(error)
            terminal['completed'] = False
        (evidence / 'terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
    if not terminal['completed']:
        raise RuntimeError('probe incomplete; inspect terminal.json')
    print('PROBE terminal ' + json.dumps(terminal))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cli', 'runtime', 'scratch', 'evidence'):
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    run(a.cli, a.runtime, a.scratch, a.evidence)
