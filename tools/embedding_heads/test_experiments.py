import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from contracts import canonical, digest
from experiments import bound_file, file_sha, load_bundle, preference_pairs, require_gpu3
from synthetic_experiment_fixture import create_fixture


class ExperimentContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = create_fixture(Path(self.tmp.name) / 'bundle')
        self.manifest = json.loads(self.path.read_text())

    def save(self):
        self.path.write_text(canonical(self.manifest))

    def mutate_rows(self, head, split, mutation):
        item = self.manifest['heads'][head][split]['rows']
        path = self.path.parent / item['path']
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutation(rows)
        path.write_text(''.join(canonical(row) + '\n' for row in rows))
        item['sha256'] = file_sha(path)
        self.save()

    def test_valid_bundle_is_explicitly_synthetic(self):
        manifest, heads = load_bundle(self.path)
        self.assertEqual(manifest['role'], 'synthetic_pipeline_test')
        self.assertEqual(heads['log_classifier']['train'][0].shape, (32, 768))

    def test_corrupted_weights_and_escape_are_rejected(self):
        item = self.manifest['heads']['log_classifier']['train']['vectors']
        with (self.path.parent / item['path']).open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            load_bundle(self.path)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            bound_file(self.path.parent, {'path': '../secret', 'sha256': 'a' * 64})

    def test_final_partition_rejected_before_opening_its_files(self):
        self.manifest['heads']['log_ranker']['final'] = {'rows': {'path': 'must-not-open'}}
        self.save()
        with self.assertRaisesRegex(ValueError, 'only train/development'):
            load_bundle(self.path)

    def test_related_source_cannot_cross_split(self):
        self.mutate_rows('web_ranker', 'development',
                         lambda rows: rows[0].update(group_key='train-output-0'))
        with self.assertRaisesRegex(ValueError, 'cross splits'):
            load_bundle(self.path)

    def test_identical_input_cannot_cross_split(self):
        self.mutate_rows('web_ranker', 'development',
                         lambda rows: rows[0].update(input_sha256=digest(['log_classifier', 'train', 0])))
        with self.assertRaisesRegex(ValueError, 'cross splits'):
            load_bundle(self.path)

    def test_missing_task_or_stale_contract_is_rejected(self):
        contract = self.manifest['input_contracts']['web_ranker']
        contract['value']['task_conditioned'] = False
        contract['sha256'] = digest(contract['value'])
        self.save()
        with self.assertRaisesRegex(ValueError, 'task-conditioned'):
            load_bundle(self.path)

    def test_nonfinite_feature_is_rejected_even_with_matching_hash(self):
        item = self.manifest['heads']['log_classifier']['train']['vectors']
        path = self.path.parent / item['path']
        values = np.load(path)
        values[0, 0] = np.nan
        np.save(path, values)
        item['sha256'] = file_sha(path)
        self.save()
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            load_bundle(self.path)

    def test_unadmitted_rows_are_rejected(self):
        self.mutate_rows('log_classifier', 'train', lambda rows: rows[0].update(admission='held'))
        with self.assertRaisesRegex(ValueError, 'unadmitted'):
            load_bundle(self.path)

    def test_comparison_scope_cannot_mix_goals(self):
        self.mutate_rows('web_ranker', 'train', lambda rows: rows[0].update(task_sha256=digest('other task')))
        with self.assertRaisesRegex(ValueError, 'mixes sources or tasks'):
            load_bundle(self.path)

    def test_missing_class_is_rejected(self):
        self.mutate_rows('log_classifier', 'train',
                         lambda rows: [row.update(label=0) for row in rows if row['label'] == 1])
        with self.assertRaisesRegex(ValueError, 'all four classes'):
            load_bundle(self.path)

    def test_pair_sampling_is_bounded_reproducible_and_in_scope(self):
        _, data = load_bundle(self.path)
        rows = data['web_ranker']['train'][1]
        first = preference_pairs(rows, 17, 19)
        self.assertEqual(first.shape, (8 * 19, 2))
        np.testing.assert_array_equal(first, preference_pairs(rows, 17, 19))
        for hi, lo in first:
            self.assertEqual(rows[hi]['comparison_id'], rows[lo]['comparison_id'])
            self.assertGreater(rows[hi]['label'], rows[lo]['label'])
        self.assertEqual(preference_pairs([{'comparison_id': 'one', 'label': 0}], 1).shape, (0, 2))

    def test_training_refuses_other_hosts_before_cuda_use(self):
        with patch('socket.gethostname', return_value='not-gpu3'):
            with self.assertRaisesRegex(RuntimeError, 'GPU3'):
                require_gpu3('cuda:0')


if __name__ == '__main__':
    unittest.main()
