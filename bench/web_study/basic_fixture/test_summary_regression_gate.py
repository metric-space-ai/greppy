"""Individual regressions stay visible without becoming a per-run veto."""
import unittest
from summarize_trials import regression_audit, aggregate_token_changes, development_token_gate


def pair(repeat, input_change, output_change, passed=True):
    return dict(case='table', repeat=repeat, both_passed=passed,
                input_tokens_change_percent=input_change,
                output_tokens_change_percent=output_change)


class RegressionGateTests(unittest.TestCase):
    def test_input_regression_remains_visible_with_favorable_median(self):
        result = regression_audit([pair(1, 10, -20), pair(2, -60, -70), pair(3, -50, -40)])
        self.assertFalse(result['every_pair_uses_fewer_input_and_output'])
        self.assertEqual(result['regressions'], [dict(case='table', repeat=1,
                         metric='input_tokens', change_percent=10)])

    def test_missing_or_equal_usage_cannot_prove_strict_improvement(self):
        result = regression_audit([pair(1, None, -20), pair(2, -10, 0)])
        self.assertFalse(result['every_pair_uses_fewer_input_and_output'])
        self.assertEqual(len(result['not_strictly_improved']), 2)

    def test_failed_task_and_empty_block_cannot_pass(self):
        self.assertFalse(regression_audit([pair(1, -10, -20, False)])['every_pair_uses_fewer_input_and_output'])
        self.assertFalse(regression_audit([])['every_pair_uses_fewer_input_and_output'])

    def test_successful_strictly_lower_pairs_pass(self):
        result = regression_audit([pair(1, -10, -20), pair(2, -30, -40)])
        self.assertTrue(result['every_pair_uses_fewer_input_and_output'])
        self.assertEqual(result['regressions'], [])

class AggregateGateTests(unittest.TestCase):
    def records(self, last):
        rows = []
        for i in range(10):
            for arm, count in [('A', 100), ('C', last if i == 9 else 80)]:
                rows.append({'arm': arm, 'tokens': {'input_tokens': count, 'output_tokens': count}})
        return rows

    def test_nine_savings_and_one_modest_regression_pass(self):
        changes = aggregate_token_changes(self.records(110))
        self.assertAlmostEqual(changes['input_tokens'], -17)
        self.assertEqual(development_token_gate(True, changes), 'passes this development block only')

    def test_large_outlier_still_counts(self):
        self.assertEqual(development_token_gate(True, aggregate_token_changes(self.records(1000))), 'failed_or_unproven')

    def test_correctness_or_integrity_failure_cannot_pass(self):
        self.assertEqual(development_token_gate(False, aggregate_token_changes(self.records(110))), 'failed_or_unproven')

    def test_missing_usage_cannot_pass(self):
        rows = self.records(110)
        rows[-1]['tokens']['output_tokens'] = None
        self.assertEqual(development_token_gate(True, aggregate_token_changes(rows)), 'failed_or_unproven')


if __name__ == '__main__':
    unittest.main()
