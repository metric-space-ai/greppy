import unittest
from records_summary import compare

class PairedCosts(unittest.TestCase):
    def test_one_regression_does_not_cancel_nine_savings(self):
        result=compare([(100,80)]*9+[(100,110)])
        self.assertEqual(result['lower_cost_pairs'],9)
        self.assertAlmostEqual(result['aggregate_change_percent'],-17)
        self.assertAlmostEqual(result['median_paired_change_percent'],-20)
        self.assertAlmostEqual(result['paired_changes_percent'][-1],10)
    def test_large_outlier_remains_in_aggregate(self):
        result=compare([(100,80)]*9+[(100,1000)])
        self.assertEqual(result['lower_cost_pairs'],9)
        self.assertGreater(result['aggregate_change_percent'],0)
    def test_incomplete_telemetry_is_not_zero(self):
        self.assertFalse(compare([(100,None)])['available'])
    def test_bootstrap_is_reproducible(self):
        pairs=[(100,80),(100,110),(200,100)]
        self.assertEqual(compare(pairs),compare(pairs))

if __name__=='__main__': unittest.main()
