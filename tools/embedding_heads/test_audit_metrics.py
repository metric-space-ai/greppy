import copy
import unittest
from audit_metrics import summarize


def historic():
    return {'strata': dict(zip(('TP', 'FP', 'FN', 'TN'), [
        {'population': n, 'judged': j, 'audited_errors': e}
        for n, j, e in ((910,100,100),(1310,400,110),(119,119,119),(1661,0,0))
    ]))}


class CoverageTests(unittest.TestCase):
    def test_historic_audit_cannot_estimate_full_recall(self):
        out = summarize(historic())
        self.assertIsNone(out['error_recall_point_estimate'])
        self.assertAlmostEqual(out['error_precision_point_estimate'], 1270.25/2220)
        self.assertEqual(out['missing_strata'], ['TN'])
        self.assertEqual(out['unaudited_tn_sensitivity']['tn_errors_to_fall_below_90_percent'], 23)
        self.assertEqual(out['release_gate'], 'not_evaluated')

    def test_previously_missed_errors_enter_recall_denominator(self):
        data = historic()
        data['strata']['TN'].update(judged=100, audited_errors=2)
        out = summarize(data)
        self.assertTrue(out['all_strata_covered'])
        self.assertFalse(out['fully_audited'])
        self.assertAlmostEqual(out['error_recall_point_estimate'], 1270.25/(1270.25+119+33.22))
        self.assertLess(out['error_recall_point_estimate'], .9)

    def test_census_crosses_original_label_strata(self):
        data = {'strata': dict(zip(('TP','FP','FN','TN'), [
            {'population':n,'judged':n,'audited_errors':e} for n,e in ((10,8),(10,3),(5,2),(5,1))
        ]))}
        out = summarize(data)
        self.assertTrue(out['fully_audited'])
        self.assertEqual(out['error_precision_point_estimate'], 11/20)
        self.assertEqual(out['error_recall_point_estimate'], 11/14)

    def test_empty_populations_do_not_fabricate_metrics(self):
        data = {'strata': {s: {'population':0,'judged':0,'audited_errors':0} for s in ('TP','FP','FN','TN')}}
        out = summarize(data)
        self.assertIsNone(out['error_precision_point_estimate'])
        self.assertIsNone(out['error_recall_point_estimate'])
        self.assertTrue(out['fully_audited'])

    def test_missing_positive_stratum_also_blocks_precision(self):
        data = historic()
        data['strata']['TP'].update(judged=0,audited_errors=0)
        out = summarize(data)
        self.assertIsNone(out['error_precision_point_estimate'])
        self.assertIsNone(out['error_recall_point_estimate'])

    def test_invalid_counts_and_omitted_strata_are_rejected(self):
        for key,value in [('population',-1),('judged',True),('judged',911),('audited_errors',101),('judged',1.0)]:
            data = copy.deepcopy(historic())
            data['strata']['TP'][key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                summarize(data)
        data = historic()
        del data['strata']['TN']
        with self.assertRaises(ValueError):
            summarize(data)


if __name__ == '__main__':
    unittest.main()
