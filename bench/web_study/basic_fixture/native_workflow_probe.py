"""Real CLI/runtime workflow and modal-archive gate; no agent/token claim."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from candidate import capture
from prepare_context import prepare

PAGE = b'''<!doctype html><title>Workflow CLI gate</title>
<h1>BACKGROUND_SENTINEL</h1>
<button id="launch" onclick="document.getElementById('editor').showModal()">Edit quantity</button>
<button id="background">Background operation</button>
<dialog id="editor" aria-label="Edit quantity">
<label for="quantity">Quantity</label><input id="quantity" value="1">
<button id="save" onclick="const q=document.getElementById('quantity').value; fetch('/commit?quantity='+encodeURIComponent(q)).then(r=>r.text()).then(v=>{document.getElementById('saved').textContent='Saved '+v;document.getElementById('editor').close()})">Save</button>
</dialog><p id="saved">Not saved</p>
'''


def run(cli, runtime, scratch, evidence):
    evidence.mkdir(parents=True, exist_ok=False)
    aliases = scratch / 'aliases'
    aliases.mkdir(parents=True, exist_ok=True)
    identity = 'workflow-' + uuid.uuid4().hex[:12]
    context = prepare(scratch / 'contexts', cli, aliases, identity, identity,
                      runtime=runtime, view='compact')
    (evidence / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
    commits = []
    lock = threading.Lock()

    class Page(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == '/commit':
                quantity = parse_qs(url.query).get('quantity', [''])[0]
                time.sleep(.15)
                with lock:
                    commits.append(quantity)
                body = quantity.encode()
                kind = 'text/plain'
            else:
                body, kind = PAGE, 'text/html'
            self.send_response(200)
            self.send_header('Content-Type', kind + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Page)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    calls, checks = [], []
    terminal = dict(schema='greppy.native-workflow-cli-proof.v1', passed=False,
                    efficiency_acceptance=False, token_telemetry=None, checks=checks)

    def call(*args, human=False, expected=0):
        flags = [] if human else ["--json"]
        argv = ([context["alias"], "web", "do", *flags, *args[1:]] if args[0] == "do"
                else [context["alias"], "web", *args, *flags])
        started = time.monotonic()
        record = dict(argv=argv)
        try:
            response = subprocess.run(argv, text=True, capture_output=True, timeout=60)
            record.update(exit_code=response.returncode, stdout=response.stdout,
                          stderr=response.stderr, stdout_bytes=len(response.stdout.encode()))
        except subprocess.TimeoutExpired as error:
            record.update(timeout=True, stdout=str(error.stdout), stderr=str(error.stderr))
            raise
        finally:
            record['wall_seconds'] = time.monotonic() - started
            calls.append(record)
            (evidence / 'calls.json').write_text(json.dumps(calls, indent=2) + '\n')
        assert response.returncode == expected, record
        if human:
            return response.stdout
        return json.loads(response.stdout)

    def check(name, condition):
        assert condition, name
        checks.append(name)

    try:
        call('open', f'http://127.0.0.1:{server.server_port}/')
        opened = call('click', 'css=#launch', '--expect', 'css=dialog[open]')
        check('single action expectation uses native workflow',
              opened['operation'] == 'web.workflow' and opened['status'] == 'ok')
        state = opened['result']['page_state']['snapshot']
        check('native modal scope is present', state['working_scope']['kind'] == 'modal')
        human = call('observe', human=True)
        check('focused view retains dialog controls', 'Quantity' in human and 'Save' in human)
        check('focused view omits background text', 'BACKGROUND_SENTINEL' not in human)
        continuation = next((line[line.index('greppy web result next '):]
                             for line in human.splitlines() if 'greppy web result next ' in line), None)
        check('focused view offers archived full state', continuation is not None)
        archived = call(*shlex.split(continuation)[2:], human=True)
        check('archive recovers omitted background', 'BACKGROUND_SENTINEL' in archived)
        saved = call('do', '--native', 'fill', 'css=#quantity', '3', '::',
                     'click', 'css=#save', '--expect', 'text=Saved 3', '--expect-timeout', '3000')
        result = saved['result']
        check('native chain completes both steps', saved['status'] == 'ok' and result['completed_steps'] == 2)
        check('final expectation explicitly held', result['steps'][1]['expectation']['result']['held'] is True)
        check('final state already shows delayed effect', 'Saved 3' in result['page_state']['snapshot']['text'])
        check('no intermediate snapshots', all('page_state' not in step.get('action', {}).get('receipt', {}) for step in result['steps']))
        with lock:
            check('server independently received exactly one save with value 3', commits == ['3'])
        failed = call('do', '--native', 'click', 'css=#launch', '::',
                      'wait', 'css=#never', '--timeout', '80', '::', 'click', 'css=#save', expected=34)
        detail = failed['result']
        check('timeout preserves partial effects and stops later save',
              failed['status'] == 'error' and detail['failed_step'] == 2 and
              detail['completed_steps'] == 1 and detail['rolled_back'] is False)
        with lock:
            check('server saw no duplicate save after timeout', commits == ['3'])
        formatted = call('do', '--native', 'fill', 'css=#quantity', '4', human=True)
        check('human workflow summary and ordered step are present',
              'workflow:' in formatted and 'step 1: action="web.fill" status="ok"' in formatted)
        step_lines = [line for line in formatted.splitlines() if line.startswith('step ')]
        check('human steps omit only redundant protocol identity', step_lines and
              all(not any(key in line for key in ('session_id', 'tab_id', 'untrusted_content_boundary'))
                  for line in step_lines))
        command = next(line[line.index('greppy web result next '):]
                       for line in formatted.splitlines() if 'greppy web result next ' in line)
        archive = call(*shlex.split(command)[2:])
        check('human workflow archive fits this probe page', archive['next_cursor'] is None)
        original = json.loads(archive['content'])
        restored = original['result']
        check('human workflow archive preserves original receipt and current form state',
              restored['steps'][0]['action']['receipt']['session_id'] == restored['session_id'] and
              any(node.get('name') == 'Quantity' and node.get('value') == '4'
                  for node in restored['page_state']['snapshot']['actionables']))
        failed_human = call('do', '--native', 'wait', 'css=#never', '--timeout', '80', human=True, expected=34)
        check('human failure preserves timeout and partial execution counts',
              'FAILED' in failed_human and 'TIMEOUT' in failed_human and
              '"completed_steps":0' in failed_human and '"rolled_back":false' in failed_human)
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
        terminal['server_commits'] = commits
        (evidence / 'terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
    assert terminal['passed'], terminal
    print(json.dumps(dict(passed=True, checks=checks, evidence=str(evidence))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cli', 'runtime', 'scratch', 'evidence'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.cli, args.runtime, args.scratch, args.evidence)
