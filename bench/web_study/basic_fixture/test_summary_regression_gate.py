"""A favorable median must not hide an individual input/output regression."""
import unittest
from summarize_trials import regression_audit


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


if __name__ == '__main__':
    unittest.main()
