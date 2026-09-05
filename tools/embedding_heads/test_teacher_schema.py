import unittest

from contracts import response_schema


class TeacherSchemaRegressionTests(unittest.TestCase):
    def examples(self, domain):
        return [{'id': 'example-a', 'domain': domain,
                 'records': [{'id': 'target-a', 'text': 'observed target'}],
                 'context': [{'id': 'context-a', 'text': 'observed context'}]}]

    def item(self, examples):
        return response_schema(examples)['properties']['annotations']['items']['properties']

    def test_web_only_schema_prevents_observed_m3_severity_failure(self):
        fields = self.item(self.examples('web'))
        self.assertEqual(fields['severity']['enum'], [None])
        self.assertIn('ZERO-BASED', fields['relevance']['description'])

    def test_schema_prevents_observed_empty_grok_evidence(self):
        fields = self.item(self.examples('web'))
        self.assertEqual(fields['evidence_ids']['minItems'], 1)
        self.assertEqual(set(fields['evidence_ids']['items']['enum']), {'target-a', 'context-a'})

    def test_targets_and_examples_are_bound_separately_from_context(self):
        fields = self.item(self.examples('log'))
        self.assertEqual(fields['example_id']['enum'], ['example-a'])
        self.assertEqual(fields['record_id']['enum'], ['target-a'])
        self.assertNotIn(None, fields['severity']['enum'])
        self.assertEqual(set(fields['severity']['enum']), {'error', 'warning', 'progress', 'text'})


if __name__ == '__main__':
    unittest.main()
