import copy
import tempfile
import unittest
from pathlib import Path
from compare_native_inputs import compare
from contracts import canonical
from native_input_probe import file_sha


class BackendComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.left, self.right = [Path(self.temp.name)/x for x in ('cpu','cuda')]
        self.row = {'schema':'greppy.heads.native-feature.v1', 'production_eligible':False,
                    'input':{'id':'target-1','head':'log_classifier'}, 'backend':'cpu',
                    'binary_sha256':'b'*64, 'tokenizer_sha256':'t'*64, 'model_sha256':'m'*64,
                    'vector':[1.0]+[0.0]*767}
        self.report = {'schema':'greppy.heads.native-input-probe.v1', 'production_eligible':False,
                       'binary_sha256':'b'*64, 'tokenizer_sha256':'t'*64, 'model_sha256':'m'*64,
                       'input_sha256':'i'*64, 'probe_sha256':'p'*64}
        self.write(self.left)
        self.write(self.right)

    def write(self, folder, row=None, report=None):
        folder.mkdir(exist_ok=True)
        row = copy.deepcopy(self.row if row is None else row)
        report = copy.deepcopy(self.report if report is None else report)
        row['backend'] = folder.name
        (folder/'native.jsonl').write_text(canonical(row)+'\n')
        report['native'] = {'mode':folder.name, 'artifact_sha256':file_sha(folder/'native.jsonl')}
        (folder/'report.json').write_text(canonical(report))

    def test_reports_drift_without_granting_release(self):
        row = copy.deepcopy(self.row)
        row['vector'][:2] = [0.8,0.6]
        self.write(self.right, row)
        result = compare(self.left, self.right)
        self.assertAlmostEqual(result['maximum_absolute_vector_difference'], 0.6)
        self.assertAlmostEqual(result['minimum_cosine_similarity'], 0.8)
        self.assertFalse(result['production_eligible'])
        self.assertTrue(result['input_identity_equal'])

    def test_tampered_artifact_is_rejected(self):
        with (self.right/'native.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            compare(self.left, self.right)

    def test_different_input_or_binary_cannot_be_compared(self):
        row = copy.deepcopy(self.row); row['input']['id'] = 'different'
        self.write(self.right, row)
        with self.assertRaisesRegex(ValueError, 'prepared input'):
            compare(self.left, self.right)
        report = copy.deepcopy(self.report); report['binary_sha256'] = 'x'*64
        self.write(self.right, report=report)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            compare(self.left, self.right)

    def test_zero_vectors_and_release_claims_rejected(self):
        row = copy.deepcopy(self.row); row['vector'] = [0.0]*768
        self.write(self.right, row)
        with self.assertRaisesRegex(ValueError, 'zero native'):
            compare(self.left, self.right)
        row = copy.deepcopy(self.row); row['production_eligible'] = True
        self.write(self.right, row)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            compare(self.left, self.right)


    def test_python_log_targets_match_native_lf_boundaries(self):
        from corpus import source_spans, verify_spans
        text = 'progress\rupdate\vfield\fform\x85next\u2028line\u2029para\r\n\nlast\r'
        rows = source_spans(text, 'physical-lines')
        self.assertEqual([r['text'] for r in rows],
                         ['progress\rupdate\vfield\fform\x85next\u2028line\u2029para\r\n', '\n', 'last\r'])
        verify_spans(text, rows)
        self.assertEqual([r['text'] for r in source_spans('a\n', 'terminated')], ['a\n'])


if __name__ == '__main__':
    unittest.main()
