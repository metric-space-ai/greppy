import copy
import unittest

from audit_sampling import plan_audit, verify_plan
from contracts import digest


def source(index, *, domain='log', family='build', stage='broad', split='train'):
    return {'id': f'{domain}-{index}', 'sha256': digest([domain, index]), 'domain': domain,
            'family': family, 'length_bin': 'long', 'stage': stage, 'split': split,
            'examples': {f'{domain}-{index}-a': digest([domain, index, 'a']),
                         f'{domain}-{index}-b': digest([domain, index, 'b'])}}


class AuditSamplingTests(unittest.TestCase):
    def test_ten_percent_complete_sources_per_domain(self):
        sources = [source(i, domain=domain) for domain in ('log', 'web') for i in range(100)]
        plan = plan_audit(sources, seed=17)
        self.assertEqual(len(plan['random_strata']), 2)
        for stratum in plan['random_strata']:
            self.assertEqual(len(stratum['selected_sources']), 10)
            self.assertEqual(stratum['inclusion_probability'], .1)
        self.assertEqual(sum(x['grok_required'] for x in plan['assignments']), 40)
        for item in sources:
            selected = [x for x in plan['assignments'] if x['source_id'] == item['id']]
            self.assertEqual(selected[0]['random_cohort'], selected[1]['random_cohort'])
        verify_plan(plan, sources)

    def test_targeted_selection_does_not_change_random_cohort(self):
        sources = [source(i) for i in range(50)]
        base = plan_audit(sources, seed=43)
        unselected = next(x['example_id'] for x in base['assignments'] if not x['random_cohort'])
        extended = plan_audit(sources, seed=43, problem_examples=[unselected])
        self.assertEqual(base['random_selection_sha256'], extended['random_selection_sha256'])
        target = next(x for x in extended['assignments'] if x['example_id'] == unselected)
        self.assertTrue(target['grok_required'])
        self.assertTrue(target['targeted_cohort'])
        self.assertFalse(target['random_cohort'])
        verify_plan(extended, sources)

    def test_pilot_and_final_are_all_reviewed(self):
        sources = [source(1, stage='pilot', split='development'), source(2, stage='final', split='final')]
        plan = plan_audit(sources, seed=17)
        self.assertTrue(all(x['grok_required'] for x in plan['assignments']))
        self.assertFalse(any(x['random_cohort'] for x in plan['assignments']))
        self.assertEqual(plan['random_strata'], [])

    def test_sampling_order_independent_and_rare_strata_rounded_up(self):
        sources = [source(1, family='network'), source(2, family='runtime')]
        self.assertEqual(plan_audit(sources, seed=17), plan_audit(sources[::-1], seed=17))
        self.assertTrue(all(x['grok_required'] for x in plan_audit(sources, seed=17)['assignments']))

    def test_labels_are_not_allowed_in_sampling_population(self):
        item = source(1); item['predicted_class'] = 'error'
        with self.assertRaisesRegex(ValueError, 'declared source fields'):
            plan_audit([item], seed=17)

    def test_final_split_cannot_evade_full_review(self):
        with self.assertRaisesRegex(ValueError, 'final audit stage'):
            plan_audit([source(1, split='final')], seed=17)

    def test_duplicate_content_and_unknown_problem_rejected(self):
        a, b = source(1), source(2); b['sha256'] = a['sha256']
        with self.assertRaisesRegex(ValueError, 'duplicate source content'):
            plan_audit([a, b], seed=17)
        with self.assertRaisesRegex(ValueError, 'unknown examples'):
            plan_audit([a], seed=17, problem_examples=['missing'])

    def test_population_or_plan_tampering_rejected(self):
        sources = [source(i) for i in range(10)]
        plan = plan_audit(sources, seed=17)
        changed = copy.deepcopy(plan)
        changed['assignments'][0]['grok_required'] = not changed['assignments'][0]['grok_required']
        with self.assertRaisesRegex(ValueError, 'differs'):
            verify_plan(changed, sources)
        sources[0]['examples']['log-0-a'] = digest('changed teacher input')
        with self.assertRaisesRegex(ValueError, 'differs'):
            verify_plan(plan, sources)


if __name__ == '__main__':
    unittest.main()
