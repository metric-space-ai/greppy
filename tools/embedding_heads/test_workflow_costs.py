import copy
import unittest
import json

from workflow_costs import compare_runs, metadata_costs


def run(pair, arm, input_tokens=100, output_tokens=10, response_bytes=1000, calls=2):
    return {'pair_id': str(pair), 'run_id': f'{pair}-{arm}', 'arm': arm,
            'agent_configuration': {'model': 'test', 'effort': 'medium'}, 'success': True,
            'metrics': {'provider_input_tokens': input_tokens, 'provider_output_tokens': output_tokens,
                        'tool_result_json_bytes': response_bytes, 'tool_calls': calls,
                        'end_to_end_seconds': None}}


def compare(runs, kind='development_tool_comparison'):
    return compare_runs(runs, baseline='A', candidate='C', comparison_kind=kind)


class WorkflowCostsTests(unittest.TestCase):
    def test_smaller_results_do_not_hide_followup_cost(self):
        report = compare([run(1, 'A'), run(1, 'C', input_tokens=150, output_tokens=20, response_bytes=100, calls=3)])
        pair = report['pairs'][0]
        self.assertTrue(pair['smaller_results_more_calls'])
        self.assertTrue(pair['smaller_results_more_provider_input'])
        self.assertTrue(pair['smaller_results_more_provider_output'])
        self.assertFalse(report['production_eligible'])
        self.assertEqual(report['aggregate']['provider_input_tokens']['median_paired_change_percent'], 50)

    def test_paired_percentages_are_not_ratio_of_arm_medians(self):
        records = []
        for i, (a, c) in enumerate([(1, 2), (100, 90), (200, 201)]):
            records += [run(i, 'A', input_tokens=a), run(i, 'C', input_tokens=c)]
        self.assertEqual(compare(records)['aggregate']['provider_input_tokens']['median_paired_change_percent'], .5)

    def test_unknown_timing_is_not_zero(self):
        metric = compare([run(1, 'A'), run(1, 'C')])['aggregate']['end_to_end_seconds']
        self.assertEqual(metric['missing_pairs'], 1)
        self.assertEqual(metric['pairs_with_values'], 0)
        self.assertIsNone(metric['median_paired_difference'])

    def test_zero_baseline_reports_absolute_cost_without_fake_percentage(self):
        metric = compare([run(1, 'A', input_tokens=0), run(1, 'C', input_tokens=5)])['aggregate']['provider_input_tokens']
        self.assertEqual(metric['median_paired_difference'], 5)
        self.assertIsNone(metric['median_paired_change_percent'])
        self.assertEqual(metric['pairs_with_percentages'], 0)

    def test_failed_run_retained_and_incomplete_pair_rejected(self):
        a, c = run(1, 'A'), run(1, 'C'); c['success'] = False
        self.assertTrue(compare([a, c])['pairs'][0]['additional_task_failure'])
        with self.assertRaisesRegex(ValueError, 'incomplete pair'):
            compare([a])
        with self.assertRaisesRegex(ValueError, 'duplicate run'):
            compare([a, c, copy.deepcopy(c)])

    def test_model_configuration_mismatch_rejected(self):
        a, c = run(1, 'A'), run(1, 'C'); c['agent_configuration']['effort'] = 'high'
        with self.assertRaisesRegex(ValueError, 'configurations differ'):
            compare([a, c])

    def test_development_tool_comparison_cannot_masquerade_as_head_ablation(self):
        a, c = run(1, 'A'), run(1, 'C')
        with self.assertRaisesRegex(ValueError, 'same release'):
            compare([a, c], 'heads_on_off')
        a.update(release_sha256='a' * 64, heads_enabled=False, backend='cpu')
        c.update(release_sha256='a' * 64, heads_enabled=True, backend='cpu')
        self.assertFalse(compare([a, c], 'heads_on_off')['production_eligible'])
        c['backend'] = 'cuda'
        with self.assertRaisesRegex(ValueError, 'same explicit backend'):
            compare([a, c], 'heads_on_off')

    def test_invalid_counters_rejected(self):
        for bad in (float('nan'), float('inf'), -1, True):
            a, c = run(1, 'A'), run(1, 'C'); c['metrics']['tool_calls'] = bad
            with self.assertRaisesRegex(ValueError, 'invalid workflow cost'):
                compare([a, c])


    def test_metadata_counters_recomputed_and_missing_usage_rejected(self):
        result = [{'type': 'text', 'text': 'state: checked=true; quantity disabled'}]
        measured = len(json.dumps(result, ensure_ascii=False, separators=(',', ':')).encode())
        metadata = {'tool_calls': [{'kind': 'request', 'call_id': 'call'},
                                   {'kind': 'response', 'call_id': 'call', 'result': result,
                                    'result_json_bytes': measured}],
                    'token_usage_status': {'usage_missing': False}, 'token_usage_conflicts': [],
                    'cumulative_turn_token_usage': {'value': {'input_tokens': 100, 'output_tokens': 20}}}
        self.assertEqual(metadata_costs(metadata)['tool_result_json_bytes'], measured)
        wrong = copy.deepcopy(metadata); wrong['tool_calls'][1]['result_json_bytes'] -= 1
        with self.assertRaisesRegex(ValueError, 'actual JSON'):
            metadata_costs(wrong)
        wrong = copy.deepcopy(metadata); wrong['token_usage_status']['usage_missing'] = True
        with self.assertRaisesRegex(ValueError, 'provider usage'):
            metadata_costs(wrong)
        wrong = copy.deepcopy(metadata); wrong['tool_calls'].pop()
        with self.assertRaisesRegex(ValueError, 'tool-call coverage'):
            metadata_costs(wrong)


if __name__ == '__main__':
    unittest.main()
