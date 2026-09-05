import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from corpus import target_examples
from queue_store import QueueStore
from teacher_queue import work_one
from test_corpus_admission import source


class DiagnosticTests(unittest.TestCase):
    def test_schema_failure_records_usage_and_safe_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            store=QueueStore(Path(temp)/'queue.sqlite')
            example=target_examples(source())[0]
            store.register_source('s1','train','g')
            store.enqueue('minimax','MiniMax-M3',[example])
            captured=io.StringIO()
            with patch('teacher_queue.minimax',return_value=({'annotations':[]},{'input_tokens':42})):
                with contextlib.redirect_stdout(captured):
                    self.assertTrue(work_one(store,'minimax'))
            self.assertIn('incomplete annotation coverage',captured.getvalue())
            with store.connect() as db:
                detail=db.execute("SELECT detail FROM events WHERE event='schema'").fetchone()[0]
            self.assertIn('input_tokens',detail)
            self.assertIn('42',detail)
            self.assertNotIn('source();',detail)


if __name__=='__main__': unittest.main()
