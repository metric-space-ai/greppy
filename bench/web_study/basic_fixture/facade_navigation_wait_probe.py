"""Probe the existing native Playwright waiter across full-document navigation."""
import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import time
import uuid

from candidate import capture
from native_wait_probe import Page
from prepare_context import prepare


def run(cli, runtime, scratch, evidence):
    evidence.mkdir(parents=True, exist_ok=False)
    aliases = scratch / 'aliases'
    aliases.mkdir(parents=True, exist_ok=False)
    name = 'facade-wait-' + uuid.uuid4().hex[:12]
    context = prepare(scratch / 'contexts', cli, aliases, name, name, runtime=runtime)
    (evidence / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Page)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    calls = []
    terminal = {'schema': 'greppy.facade-navigation-wait-proof.v1', 'passed': False,
                'native_cli_adapter_accepted': False, 'efficiency_accepted': False}

    def call(*args):
        record = {'argv': [context['alias'], 'web', *args, '--json']}
        start = time.monotonic()
        try:
            r = subprocess.run(record['argv'], text=True, capture_output=True, timeout=45)
            record.update(exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr)
            return r
        except subprocess.TimeoutExpired as error:
            record.update(timeout=True, stdout=str(error.stdout), stderr=str(error.stderr))
            raise
        finally:
            record['wall_seconds'] = time.monotonic() - start
            calls.append(record)
            (evidence / 'calls.json').write_text(json.dumps(calls, indent=2) + '\n')

    try:
        opened = call('open', f'http://127.0.0.1:{server.server_port}/')
        assert opened.returncode == 0, opened.stdout
        script = 'const initial_url = await page.url();\nawait page.goto(' + json.dumps(f'http://127.0.0.1:{server.server_port}/') + ');\n' + '''
await page.click('#start');
await page.waitForFunction(() => Boolean(document.querySelector('#late')), undefined, { timeout: 5000 });
const delayed = await page.locator('#late').count();
await page.click('#navigate');
let navigation_wait_error = null;
try {
  await page.waitForFunction(() => location.pathname === '/landed', undefined, { timeout: 5000 });
} catch (error) {
  navigation_wait_error = String(error);
}
const url = await page.url();
const location_url = await page.evaluate(() => location.href);
const title = await page.title();
const landed = await page.locator('#landed').count();
return { initial_url, delayed, navigation_wait_error, url, location_url, title, landed };
'''
        result = call('pw', script)
        terminal['facade_exit_code'] = result.returncode
        assert result.returncode == 0, result.stdout
        value = json.loads(result.stdout)['result']['value']
        terminal['observed'] = value
        terminal['candidate_unchanged'] = capture(cli, runtime) == context['candidate']
        terminal['passed'] = (value['delayed'] == 1 and value['navigation_wait_error'] is None and
                              value['landed'] == 1 and value['title'] == 'Landed' and
                              value['url'] == f'http://127.0.0.1:{server.server_port}/landed' and
                              value['location_url'] == value['url'] and
                              terminal['candidate_unchanged'])
    except BaseException as error:
        terminal['failure'] = repr(error)
        raise
    finally:
        try:
            cleanup = call('runtime', 'stop')
            terminal['cleanup'] = json.loads(cleanup.stdout)
            terminal['passed'] = terminal['passed'] and cleanup.returncode == 0 and terminal['cleanup']['result']['running'] is False
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
