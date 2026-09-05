import tempfile
from pathlib import Path
import unittest

from audit_log_archive import classify_payload, inspect_member


class ArchiveAuditTests(unittest.TestCase):
    def test_api_failure_is_not_a_build_capture(self):
        raw = b'{"message":"API rate limit exceeded","status":"403"}'
        self.assertEqual(classify_payload(raw, len(raw)), 'retrieval_error_json')
        self.assertEqual(classify_payload(b'', 0), 'empty')
        self.assertEqual(classify_payload(b' <!DOCTYPE html>broken', 21), 'html_response_requires_review')
        self.assertEqual(classify_payload(b'error: cannot compile\n', 22), 'text_capture_requires_review')

    def test_incomplete_json_prefix_does_not_invent_whole_payload(self):
        raw = b'{"message":"bad","status":"403"}'
        self.assertEqual(classify_payload(raw, len(raw) + 10), 'text_capture_requires_review')

    def test_hash_line_count_utf8_and_no_automatic_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'log').write_bytes('ä\r\nb\nlast'.encode())
            row = inspect_member(root, 'log')
            self.assertEqual(row['lines'], 3)
            self.assertTrue(row['valid_utf8'])
            self.assertFalse(row['privacy_admitted'])
            self.assertFalse(row['complete_build_output_verified'])
            self.assertFalse(row['final_test_eligible'])
            (root / 'log').write_bytes(b'bad\xff\n')
            self.assertFalse(inspect_member(root, 'log')['valid_utf8'])

    def test_paths_cannot_escape_or_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                inspect_member(root, '../outside')
            (root / 'real').write_text('a')
            (root / 'link').symlink_to(root / 'real')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                inspect_member(root, 'link')


if __name__ == '__main__':
    unittest.main()
