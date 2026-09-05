import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from corpus import target_examples
from queue_store import QueueStore
from teachers import bound_prompt, ProviderFailure
from test_corpus_admission import source


class BindingTests(unittest.TestCase):
    def setup_store(self,temp):
        store=QueueStore(Path(temp)/'queue.sqlite')
        example=target_examples(source())[0]
        store.register_source('s1','train','g')
        jid=store.enqueue('minimax','MiniMax-M3',[example])
        return store,example,jid

    def test_restarted_job_uses_bound_prompt_after_code_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            store,_,_=self.setup_store(temp)
            job=store.claim('minimax')
            with patch('contracts.prompt_for',return_value='different prompt'):
                self.assertEqual(bound_prompt(job),job['prompt'])
            corrupt=copy.deepcopy(job); corrupt['prompt']+='tampering'
            with self.assertRaises(ProviderFailure): bound_prompt(corrupt)

    def test_schema_failures_are_not_automatically_repeated(self):
        with tempfile.TemporaryDirectory() as temp:
            store,_,_=self.setup_store(temp)
            job=store.claim('minimax',now=10**12)
            store.fail(job,'schema',now=10**12)
            self.assertIsNone(store.claim('minimax',now=10**12+100))
            with store.connect() as db:
                row=db.execute('SELECT status,retry_count FROM jobs').fetchone()
            self.assertEqual((row['status'],row['retry_count']),('failed',0))

    def test_exact_key_recovers_binding_without_replaying_completed_job(self):
        with tempfile.TemporaryDirectory() as temp:
            store,example,jid=self.setup_store(temp)
            # Simulate metadata from the previous cache-key format; result bytes
            # remain immutable. Recovery is allowed only for an exact current key.
            with store.connect() as db:
                db.execute("UPDATE jobs SET prompt='',output_schema='{}',status='done',result='unchanged' WHERE id=?",(jid,))
            self.assertEqual(store.enqueue('minimax','MiniMax-M3',[example]),jid)
            self.assertIsNone(store.claim('minimax'))
            with store.connect() as db:
                row=db.execute('SELECT result,prompt FROM jobs').fetchone()
            self.assertEqual(row['result'],'unchanged')
            self.assertTrue(row['prompt'])


if __name__=='__main__': unittest.main()
