"""Reject misleading successful/complete claims in composition evidence."""
import unittest
from composition_replay import recover_rows

class CompositionEvidenceTests(unittest.TestCase):
    def envelope(self, result, status='ok'):
        return {'schema': 'greppy.web-runtime.v1', 'status': status, 'result': result}

    def test_error_is_not_empty_selection(self):
        with self.assertRaisesRegex(ValueError, 'did not succeed'):
            recover_rows('find', self.envelope({'value': {'nodes': [], 'count': 0}}, 'error'))

    def test_unknown_schema_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unrecognized'):
            recover_rows('find', {'schema': 'future', 'status': 'ok'})

    def test_truncation_not_counted_as_complete(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            recover_rows('observe', self.envelope({'actionables': [], 'refs_truncated': True}))

    def test_scope_truncation_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, 'incomplete scope'):
            recover_rows('observe', self.envelope({'actionables': [],
                'observation_scope': {'roots_truncated': True}}))

    def test_count_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'count differs'):
            recover_rows('extract', self.envelope({'value': {'count': 2, 'rows': [{'text': 'one'}]}}))

    def test_wrong_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, 'shape differs'):
            recover_rows('find', self.envelope({'value': {'rows': []}}))

    def test_no_visibility_or_ref_is_invented(self):
        row = {'text': 'Hidden delivery', 'attr:data-price': '1'}
        recovered = recover_rows('extract', self.envelope({'value': {'count': 1, 'rows': [row]}}))
        self.assertEqual(recovered, [row])
        self.assertNotIn('visible', recovered[0])
        self.assertNotIn('ref', recovered[0])
        self.assertEqual(recovered[0]['attr:data-price'], '1')

    def test_confirmed_empty_is_valid(self):
        self.assertEqual(recover_rows('find', self.envelope({'value': {'nodes': [], 'count': 0}})), [])

if __name__ == '__main__':
    unittest.main()
