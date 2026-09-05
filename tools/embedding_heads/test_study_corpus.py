import copy
import json
from pathlib import Path
import tempfile
import unittest

from contracts import canonical
from study_corpus import index_trial, sha


class StudyCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trace = self.root / 'trace.jsonl'
        self.meta = self.root / 'metadata.json'
        self.manifest = self.root / 'manifest.json'
        self.trial = self.root / 'trial.json'
        self.envelopes = [
            {'schema': 'greppy.web-runtime.v1', 'operation': 'web.click',
             'status': 'ok', 'request_id': 'wrq_a', 'result': {'session_id': 'session-a'}},
            {'schema': 'greppy.web-runtime.v1', 'operation': 'web.observe',
             'status': 'ok', 'request_id': 'wrq_b',
             'result': {'session_id': 'session-a', 'actionables': [{'ref': '@1', 'label': 'PUBLIC_PAGE_TEXT'}]}}]

    def fixture(self, envelopes=None):
        lines = [canonical({'type': 'reasoning', 'private': 'PRIVATE_REASONING_SENTINEL'}) + '\n']
        tools = []
        for n, envelope in enumerate(envelopes or self.envelopes):
            call = f'call-{n}'
            request = {'type': 'custom_tool_call', 'call_id': call, 'name': 'exec',
                       'input': 'PRIVATE_COMMAND_SENTINEL'}
            output = [{'type': 'input_text', 'text': canonical(envelope)}]
            response = {'type': 'custom_tool_call_output', 'call_id': call, 'output': output}
            for payload in (request, response):
                source_line = 10 + len(lines)
                lines.append(canonical({'type': 'response_item', 'payload': payload}) + '\n')
                tool = {'call_id': call, 'source_line': source_line}
                if payload is request:
                    tool.update(kind='request', name='exec', arguments=request['input'])
                else:
                    tool.update(kind='response', result=output)
                tools.append(tool)
        raw = ''.join(lines).encode()
        self.trace.write_bytes(raw)
        bounds = {'first': 10, 'last': 9 + len(lines), 'record_count': len(lines)}
        byte_bounds = {'start': 100, 'end_exclusive': 100 + len(raw)}
        metadata = {'schema_version': 'codex-trace-export.v1', 'turn_id': 'turn',
                    'source': '/private/original.jsonl', 'line_boundaries': bounds,
                    'byte_boundaries': byte_bounds, 'tool_calls': tools}
        manifest = {'schema_version': 'codex-trace-manifest.v1', 'turn_id': 'turn',
                    'source': metadata['source'], 'export': str(self.trace), 'metadata': str(self.meta),
                    'line_boundaries': bounds, 'byte_boundaries': byte_bounds,
                    'sha256': sha(raw), 'byte_length': len(raw)}
        trial = {'schema': 'greppy.web-study.basic.v1', 'turn_id': 'turn',
                 'oracle': {'ok': False, 'private_details': 'ORACLE_SENTINEL'},
                 'artifacts': {'trace': str(self.trace), 'metadata': str(self.meta), 'manifest': str(self.manifest)}}
        for path, value in ((self.meta, metadata), (self.manifest, manifest), (self.trial, trial)):
            path.write_text(canonical(value))
        return metadata

    def run_index(self):
        return index_trial(self.trial, family='test-family')

    def test_only_pointers_and_same_session_receipts_exported(self):
        self.fixture()
        result = self.run_index()
        serialized = canonical(result)
        for hidden in ('PRIVATE_REASONING_SENTINEL', 'PRIVATE_COMMAND_SENTINEL', 'PUBLIC_PAGE_TEXT', 'ORACLE_SENTINEL'):
            self.assertNotIn(hidden, serialized)
        self.assertFalse(result['oracle']['ok'])  # Failed episodes stay failed.
        self.assertFalse(result['final_eligible'])
        self.assertEqual(result['observations'][0]['last_action']['operation'], 'web.click')
        self.assertEqual(result['events'][0]['source_line'], 12)
        self.assertEqual(result['events'][0]['export_line'], 3)

    def test_unknown_session_never_inherits_last_action(self):
        envelopes = copy.deepcopy(self.envelopes)
        del envelopes[1]['result']['session_id']
        self.fixture(envelopes)
        self.assertIsNone(self.run_index()['observations'][0]['last_action'])

    def test_different_session_never_inherits_last_action(self):
        envelopes = copy.deepcopy(self.envelopes)
        envelopes[1]['result']['session_id'] = 'session-b'
        self.fixture(envelopes)
        self.assertIsNone(self.run_index()['observations'][0]['last_action'])

    def test_corrupted_trace_rejected(self):
        self.fixture()
        self.trace.write_bytes(self.trace.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_index()

    def test_forged_metadata_output_rejected(self):
        metadata = self.fixture()
        metadata['tool_calls'][1]['result'] = []
        self.meta.write_text(canonical(metadata))
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.run_index()

    def test_original_line_not_local_line(self):
        metadata = self.fixture()
        metadata['tool_calls'][0]['source_line'] = 2
        self.meta.write_text(canonical(metadata))
        with self.assertRaisesRegex(ValueError, 'source line'):
            self.run_index()

    def test_no_nested_page_envelope_extraction(self):
        envelopes = copy.deepcopy(self.envelopes)
        envelopes[1]['schema'] = 'unrelated'
        envelopes[1]['page_text'] = canonical(self.envelopes[1])
        self.fixture(envelopes)
        self.assertEqual(self.run_index()['observations'], [])

    def test_explicit_adapter_pair_retains_action_without_inventing_verb(self):
        wrapper = {'schema': 'greppy.web-study.action-observe.v1', 'action_exit_code': 0,
                   'action': {'ok': True, 'session_id': 'session-a'},
                   'observation_exit_code': 0, 'observation': self.envelopes[1],
                   'task_success': 'not_evaluated', 'subprocess_count': 2}
        self.fixture([wrapper])
        result = self.run_index()
        action = result['observations'][0]['last_action']
        self.assertEqual(result['observations'][0]['action_context_status'], 'explicit_adapter_pair')
        self.assertIsNone(action['operation'])
        self.assertEqual(action['task_success'], 'not_evaluated')
        self.assertEqual(result['events'][0]['adapter']['decoded_json_pointer'], '/observation')

    def test_adapter_exit_status_contradiction_rejected(self):
        wrapper = {'schema': 'greppy.web-study.action-observe.v1', 'action_exit_code': 0,
                   'action': {'ok': True}, 'observation_exit_code': 34,
                   'observation': self.envelopes[1], 'task_success': 'not_evaluated', 'subprocess_count': 2}
        self.fixture([wrapper])
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            self.run_index()

    def test_adapter_cannot_claim_task_success(self):
        wrapper = {'schema': 'greppy.web-study.action-observe.v1', 'action_exit_code': 0,
                   'action': {}, 'observation_exit_code': 0, 'observation': self.envelopes[1],
                   'task_success': True, 'subprocess_count': 2}
        self.fixture([wrapper])
        with self.assertRaisesRegex(ValueError, 'invalid explicit'):
            self.run_index()

    def test_result_only_observation_has_no_fabricated_envelope(self):
        result = {'actionables': [], 'headings': [], 'links': [], 'ref_count': 0,
                  'refs_truncated': False, 'text': '', 'title': '', 'url': 'http://fixture',
                  'untrusted_content_boundary': 'UNTRUSTED_PAGE_CONTENT'}
        self.fixture([self.envelopes[0], result])
        indexed = self.run_index()
        self.assertEqual(indexed['observations'][0]['format'], 'observation_result_only')
        self.assertIsNone(indexed['observations'][0]['last_action'])
        self.assertIsNone(indexed['events'][1]['operation'])
        self.assertIsNone(indexed['events'][1]['status'])

    def test_unscoped_protocol_error_is_retained_without_guessing(self):
        error = {'schema': 'greppy.web-runtime.v1', 'status': 'error',
                 'request_id': 'wrq_error', 'error': {'code': 'protocol_violation'}}
        self.fixture([self.envelopes[0], error, self.envelopes[1]])
        result = self.run_index()
        self.assertEqual(len(result['events']), 3)
        self.assertIsNone(result['events'][1]['operation'])
        self.assertIsNone(result['observations'][0]['last_action'])

    def test_no_invented_missing_checked(self):
        self.fixture()
        result = self.run_index()
        self.assertNotIn('checked', canonical(result))
        self.assertEqual(result['observations'][0]['snapshot_sha256'], result['events'][1]['envelope_sha256'])


if __name__ == '__main__':
    unittest.main()
