#!/usr/bin/env python3
"""Contracts for independent outcome verification and bounded process accounting."""
import importlib.util
import json
import pathlib
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request

spec = importlib.util.spec_from_file_location('measure_web_efficiency', pathlib.Path(__file__).with_name('measure-web-efficiency.py'))
measurement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measurement)


class OutcomeContract(unittest.TestCase):
    def setUp(self):
        self.fixture = measurement.Fixture({'rounds': 1, 'tasks': 1, 'concurrency': 1, 'nodes': 3})
        self.thread = threading.Thread(target=self.fixture.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.fixture.server_port}'

    def tearDown(self):
        self.fixture.shutdown()
        self.fixture.server_close()
        self.thread.join(timeout=5)

    def post(self, path, body):
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=body, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    def completed_receipt(self):
        row = {'round': 0, 'duration_ms': 50, 'succeeded': 1, 'attempted': 1,
               'tasks': [{'round': 0, 'worker': 0, 'task': 0, 'ok': True, 'error': None, 'duration_ms': 50}]}
        self.assertEqual(self.post('/round', row), 200)
        self.assertEqual(self.post('/result', {'schema': 'greppy.web-efficiency.workload.v1', 'rounds': [row]}), 200)

    def submit(self):
        return self.post('/submit', urllib.parse.urlencode({'q': self.fixture.token(0, 0, 0)}).encode())

    def test_claimed_success_without_post_effect_never_counts(self):
        self.completed_receipt()
        audit = self.fixture.audit()
        self.assertFalse(audit['all_tasks_verified'])
        self.assertEqual(audit['independently_verified_completed_tasks'], 0)

    def test_actual_utf8_post_effect_and_complete_receipt_qualify(self):
        self.assertEqual(self.submit(), 200)
        self.completed_receipt()
        audit = self.fixture.audit()
        self.assertTrue(audit['all_tasks_verified'], audit)
        self.assertEqual(audit['independently_verified_completed_tasks'], 1)
        self.assertEqual(audit['verified_task_durations_ms'], [50])

    def test_duplicate_side_effect_disqualifies_run(self):
        self.assertEqual(self.submit(), 200)
        self.assertEqual(self.submit(), 400)
        self.completed_receipt()
        self.assertFalse(self.fixture.audit()['all_tasks_verified'])

    def test_malformed_round_does_not_become_success_or_crash_audit(self):
        self.assertEqual(self.post('/result', {'schema': 'greppy.web-efficiency.workload.v1', 'rounds': [None]}), 200)
        audit = self.fixture.audit()
        self.assertFalse(audit['all_tasks_verified'])
        self.assertEqual(audit['independently_verified_completed_tasks'], 0)


class ProcessContract(unittest.TestCase):
    def test_failed_process_retains_exit_cost_and_bounded_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            result = measurement.execute([sys.executable, '-c', "print('failure evidence'); raise SystemExit(7)"], pathlib.Path(directory), 3)
        self.assertEqual(result['exit_code'], 7)
        self.assertFalse(result['timed_out'])
        self.assertIn('failure evidence', result['stdout_tail'])
        self.assertGreater(result['process_cold_total_ms'], 0)
        self.assertGreater(result['max_individual_process_rss_bytes'], 0)

    def test_hung_owned_process_is_terminated_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            result = measurement.execute([sys.executable, '-c', 'import time; time.sleep(30)'], pathlib.Path(directory), .1)
        self.assertTrue(result['timed_out'])
        self.assertNotEqual(result['exit_code'], 0)
        self.assertLess(result['process_cold_total_ms'], 3000)


if __name__ == '__main__':
    unittest.main()
