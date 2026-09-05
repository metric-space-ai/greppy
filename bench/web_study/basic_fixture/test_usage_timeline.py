import unittest
from usage_timeline import timeline


def usage(total, last=None):
    values = lambda x: dict(zip(('input_tokens', 'output_tokens', 'cached_input_tokens'), x))
    return {'type': 'event_msg', 'payload': {'type': 'token_count', 'info': {
        'total_token_usage': values(total), 'last_token_usage': values(last or total)}}}


def call(identity):
    return {'type': 'response_item', 'payload': {'type': 'function_call', 'call_id': identity, 'name': 'exec'}}


class UsageTimelineTests(unittest.TestCase):
    def test_duplicates_do_not_double_count_or_consume_pending_calls(self):
        first = usage((100, 10, 80))
        result = timeline([call('a'), first, call('b'), first, usage((220, 22, 180), (120, 12, 100))])
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['responses']), 2)
        self.assertEqual(result['responses'][1]['calls'][0]['call_id'], 'b')
        self.assertEqual(sum(r['tokens']['input_tokens'] for r in result['responses']), 220)

    def test_multiple_calls_are_grouped_without_dividing_usage(self):
        result = timeline([call('a'), call('b'), usage((100, 10, 80))])
        self.assertEqual(len(result['responses'][0]['calls']), 2)
        self.assertEqual(result['responses'][0]['tokens']['output_tokens'], 10)

    def test_missing_or_noninteger_counters_fail_closed(self):
        for record in [usage((100, 10)), usage((100, True, 80)), {'type': 'event_msg', 'payload': {'type': 'token_count'}}]:
            self.assertFalse(timeline([record])['complete'])
        self.assertFalse(timeline([call('a')])['complete'])

    def test_prior_turn_baseline_reset_and_skipped_response_are_unattributable(self):
        for records in [[usage((100, 10, 80), (20, 2, 10))],
                        [usage((100, 10, 80)), usage((50, 5, 40))],
                        [usage((100, 10, 80)), usage((300, 30, 200), (100, 10, 80))]]:
            result = timeline(records)
            self.assertFalse(result['complete'])
            self.assertIsNone(result['responses'][-1]['tokens'])

    def test_impossible_cache_count_is_not_accepted(self):
        self.assertFalse(timeline([usage((100, 10, 101))])['complete'])

    def test_reasoning_and_page_content_cannot_supply_counters(self):
        fake = usage((999, 99, 900))
        records = [{'type': 'response_item', 'payload': {'type': 'reasoning', 'content': fake}},
                   {'type': 'response_item', 'payload': {'type': 'function_call_output', 'output': fake}},
                   usage((100, 10, 80))]
        result = timeline(records)
        self.assertTrue(result['complete'])
        self.assertEqual(result['total']['input_tokens'], 100)
        self.assertEqual(result['responses'][0]['calls'], [])


if __name__ == '__main__':
    unittest.main()
