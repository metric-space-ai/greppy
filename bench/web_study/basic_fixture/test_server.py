import json, os, subprocess, sys, tempfile, threading, unittest
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import server

class BasicFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.run_dir = Path(self.tmp.name)
        self.env = {**os.environ, 'GREPPY_BASIC_FIXTURE_RUN_DIR': self.tmp.name}
        server.set_dir(self.tmp.name)

    def tearDown(self): self.tmp.cleanup()
    def cli(self, *args): return subprocess.run([sys.executable, str(HERE / 'server.py'), *args, '--run-dir', self.tmp.name], env=self.env, text=True, capture_output=True)
    def create(self, case, seed='pair'): return self.cli('create-run', '--case', case, '--seed', seed).stdout.strip()
    def state(self, rid): return json.loads((self.run_dir / f'{rid}.json').read_text())
    def action(self, rid, action, value=None, origin=None):
        s = self.state(rid); payload = {'value': value};
        if origin is not None: payload['origin'] = origin
        server.mutate(s, action, payload)

    def test_cases_are_isolated_and_paired_facts_match(self):
        a, b = self.create('text'), self.create('text'); c = self.create('address', 'other')
        self.assertNotEqual(a, b); self.assertEqual(self.state(a)['facts'], self.state(b)['facts']); self.assertNotEqual(self.state(a)['facts'], self.state(c)['facts'])

    def test_exact_oracles_and_negative_states(self):
        rid = self.create('text'); self.action(rid, 'set_note', 'wrong'); self.assertNotEqual(self.cli('verify-run', rid).returncode, 0); self.action(rid, 'set_note', 'Ready for review'); self.assertEqual(self.cli('verify-run', rid).returncode, 0)
        rid = self.create('checkbox'); self.assertRaises(ValueError, self.action, rid, 'set_quantity', 3); self.action(rid, 'set_enabled', True); self.action(rid, 'set_quantity', 3); self.assertEqual(self.cli('verify-run', rid).returncode, 0)
        rid = self.create('address'); self.action(rid, 'set_country', 'France'); self.assertRaises(ValueError, self.action, rid, 'set_city', 'Berlin'); self.assertNotEqual(self.cli('verify-run', rid).returncode, 0); self.action(rid, 'set_country', 'Germany'); self.action(rid, 'set_city', 'Berlin'); self.action(rid, 'set_postcode', '10115'); self.assertEqual(self.cli('verify-run', rid).returncode, 0)
        rid = self.create('dialog'); self.action(rid, 'save', '', 'outside'); self.assertNotEqual(self.cli('verify-run', rid).returncode, 0); self.action(rid, 'save', '', 'task4-dialog'); self.assertEqual(self.cli('verify-run', rid).returncode, 0)

    def test_run_id_and_input_validation(self):
        self.assertNotEqual(self.cli('verify-run', '../x').returncode, 0)
        rid = self.create('checkbox'); self.assertRaises(ValueError, self.action, rid, 'set_enabled', 'true'); self.assertRaises(ValueError, self.action, rid, 'unknown', True); self.assertRaises(ValueError, self.action, rid, 'set_quantity', 1)

    def test_server_launched_through_symlink_serves_assets(self):
        alias = self.run_dir / 'server-alias.py'
        alias.symlink_to((HERE / 'server.py').resolve())
        child = subprocess.Popen([sys.executable, str(alias), 'serve', '--port', '0', '--run-dir', str(self.run_dir)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            base = child.stdout.readline().strip().rstrip('/')
            self.assertTrue(base.startswith('http://127.0.0.1:'))
            self.assertEqual(urlopen(base + '/static/app.js', timeout=5).status, 200)
            self.assertEqual(urlopen(base + '/static/styles.css', timeout=5).status, 200)
        finally:
            child.terminate()
            child.communicate(timeout=5)

    def test_http_assets_actions_and_port_zero(self):

        rid = self.create('text'); httpd = server.HTTPServer(('127.0.0.1', 0), server.Handler); thread = threading.Thread(target=httpd.serve_forever); thread.start(); base = f'http://127.0.0.1:{httpd.server_port}'
        try:
            self.assertEqual(urlopen(base + '/?run_id=' + rid).status, 200); self.assertEqual(urlopen(base + '/static/app.js').status, 200)
            with self.assertRaises(HTTPError): urlopen(base + '/static/missing.js')
            req = Request(base + '/api/action', data=json.dumps({'run_id': rid, 'action': 'set_note', 'payload': {'value': 'Ready for review'}}).encode(), headers={'Content-Type': 'application/json'})
            self.assertEqual(urlopen(req).status, 200); self.assertEqual(self.state(rid)['values']['note'], 'Ready for review')
        finally: httpd.shutdown(); thread.join()

if __name__ == '__main__': unittest.main()
