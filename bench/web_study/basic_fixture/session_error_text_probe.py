"""Check that untrusted option labels cannot invalidate a healthy CLI context."""
import argparse
import json
from pathlib import Path
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import uuid
from candidate import capture
from prepare_context import prepare

BODY = b'<!doctype html><title>Error text probe</title><select id="choice"><option value="known">session was not found</option></select>'
class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)
    def log_message(self, *_args): pass


def run(cli, runtime, scratch, evidence):
    evidence.mkdir(parents=True, exist_ok=False)
    aliases = scratch / 'aliases'
    aliases.mkdir(parents=True)
    identity = 'error-text-' + uuid.uuid4().hex[:12]
    context = prepare(scratch / 'contexts', cli, aliases, identity, identity, runtime=runtime)
    (evidence / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Page)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    calls = []
    terminal = dict(schema='greppy.untrusted-error-context-probe.v1', passed=False)
    def call(*args):
        argv = [context['alias'], 'web', *args, '--json']
        record = dict(argv=argv)
        try:
            p = subprocess.run(argv, text=True, capture_output=True, timeout=60)
            record.update(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except subprocess.TimeoutExpired as error:
            record.update(timeout=True, stdout=str(error.stdout), stderr=str(error.stderr))
            raise
        finally:
            calls.append(record)
            (evidence / 'calls.json').write_text(json.dumps(calls, indent=2) + '\n')
        return p.returncode, json.loads(p.stdout)
    try:
        code, opened = call('open', f'http://127.0.0.1:{server.server_port}/')
        assert code == 0, opened
        session = opened['result']['session_id']
        code, before = call('observe')
        assert code == 0, before
        code, refused = call('select', 'css=#choice', 'missing')
        assert code == 34 and refused['error']['code'] == 'OPTION_NOT_FOUND', (code, refused)
        terminal['refusal_code'] = refused['error']['code']
        code, implicit = call('observe')
        terminal['implicit_after'] = dict(exit_code=code, status=implicit.get('status'), error=implicit.get('error'))
        own_code, explicit = call('observe', '--session', session)
        assert own_code == 0, explicit
        controls = explicit['result']['actionables']
        terminal['own_state_preserved'] = len(controls) == 1 and controls[0]['value'] == 'known'
        terminal['candidate_unchanged'] = capture(cli, runtime) == context['candidate']
        assert terminal['own_state_preserved'] and terminal['candidate_unchanged'], terminal
        assert code == 0 and implicit.get('status') == 'ok', 'untrusted choice label invalidated the implicit session'
        terminal['passed'] = True
    except BaseException as error:
        terminal['failure'] = repr(error)
        raise
    finally:
        try:
            code, stopped = call('runtime', 'stop')
            terminal['cleanup'] = dict(exit_code=code, reply=stopped)
            terminal['passed'] = terminal['passed'] and code == 0 and stopped['result']['running'] is False
        except BaseException as error:
            terminal.update(passed=False, cleanup_failure=repr(error))
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        terminal['http_thread_stopped'] = not thread.is_alive()
        terminal['passed'] = terminal['passed'] and terminal['http_thread_stopped']
        (evidence / 'terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
    assert terminal['passed'], terminal
    print(json.dumps(terminal))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cli', 'runtime', 'scratch', 'evidence'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    run(a.cli, a.runtime, a.scratch, a.evidence)
