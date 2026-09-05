import copy
import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from contracts import SCHEMA, sanitized_example, validate_annotations, prompt_for
from queue_store import QueueStore
from teachers import visible_response_text, ProviderFailure


def example(domain='log'):
    return {'schema':SCHEMA,'id':'e1','source_id':'source1','family':'fixture','domain':domain,
            'split':'train','group_key':'group1','privacy_review':'synthetic','task':'Diagnose the operation.',
            'records':[{'id':'r1','text':'error: compile failed','protected':True},
                       {'id':'r2','text':'  42 | raise ValueError()','protected':False}],
            'context':[{'id':'c1','text':'build returned exit code 1'}]}


def labels(domain='log'):
    return {'annotations':[{'example_id':'e1','record_id':rid,'severity':label if domain=='log' else None,
            'relevance':relevance,'evidence_ids':[rid],'reason':'Observed source record.','ambiguous':False}
            for rid,label,relevance in [('r1','error',3),('r2','text',2)]]}


class ContractTests(unittest.TestCase):
    def test_complete_log_and_web_annotations(self):
        for domain in ('log','web'):
            validate_annotations(labels(domain),[example(domain)])

    def test_invalid_or_invented_annotations_rejected(self):
        mutations = [lambda x:x['annotations'].pop(),
                     lambda x:x['annotations'].append(copy.deepcopy(x['annotations'][0])),
                     lambda x:x['annotations'][0].update(evidence_ids=['invented']),
                     lambda x:x['annotations'][0].update(relevance=True),
                     lambda x:x['annotations'][0].update(relevance=4),
                     lambda x:x['annotations'][0].update(ambiguous='false'),
                     lambda x:x['annotations'][0].update(record_id=[]),
                     lambda x:x['annotations'][0].update(evidence_ids=['r1','r1']),
                     lambda x:x.update(thinking='untrusted'),
                     lambda x:x['annotations'][0].update(reason='')]
        for mutate in mutations:
            result=labels(); mutate(result)
            with self.subTest(result=result), self.assertRaises(ValueError):
                validate_annotations(result,[example()])
        with self.assertRaises(ValueError): validate_annotations(labels(),[example('web')])

    def test_private_fields_are_not_sent_and_common_secrets_redacted(self):
        x=example(); x['reasoning']='DO NOT SEND'; x['teacher_labels']=labels(); x['headers']={'Authorization':'secret'}
        x['records'][0]['text']='Authorization: Bearer abcdefghijklmnop password=hunter22 email me@example.org https://alice:secret@host/path'
        safe=sanitized_example(x); text=json.dumps(safe)
        for secret in ('DO NOT SEND','teacher_labels','hunter22','abcdefghijklmnop','me@example.org','alice:secret'):
            self.assertNotIn(secret,text)
        self.assertGreater(safe['redaction_count'],0)
        self.assertIn('<SECRET>',text)

    def test_only_visible_output_is_consumed(self):
        result={'output':[{'type':'reasoning','summary':[{'text':'PRIVATE'}]},
                           {'type':'message','role':'assistant','content':[{'type':'output_text','text':'{"annotations":[]}'}]}]}
        self.assertEqual(visible_response_text(result),'{"annotations":[]}')
        with self.assertRaises(ProviderFailure): visible_response_text({'output':[{'type':'reasoning','text':'PRIVATE'}]})

    def test_blind_prompt_does_not_contain_previous_labels(self):
        x=example(); x['m3_judgment']='M3_SECRET_JUDGMENT'; x['model_probabilities']=[.9,.1,0,0]
        self.assertNotIn('M3_SECRET_JUDGMENT',prompt_for([sanitized_example(x)]))


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.path=Path(self.temp.name)/'queue.sqlite'
        self.store=QueueStore(self.path); self.x=example()
        self.store.register_source('source1','train','group1')

    def tearDown(self): self.temp.cleanup()

    def enqueue(self, **kwargs): return self.store.enqueue('minimax','MiniMax-M3',[self.x],now=0,**kwargs)

    def test_done_is_idempotent_across_restarts(self):
        key=self.enqueue(); job=self.store.claim('minimax',now=1)
        self.store.complete(job,labels(),now=2)
        store=QueueStore(self.path)
        self.assertEqual(store.enqueue('minimax','MiniMax-M3',[self.x],now=3),key)
        self.assertIsNone(store.claim('minimax',now=4))
        self.assertEqual(store.status()['jobs'][0]['status'],'done')

    def test_changed_rubric_or_model_invalidates_cache(self):
        key=self.enqueue()
        self.assertNotEqual(key,self.enqueue(rubric='v2'))
        self.assertNotEqual(key,self.store.enqueue('minimax','other-model',[self.x],now=0))

    def test_invalid_result_never_committed(self):
        self.enqueue(); job=self.store.claim('minimax',now=1)
        with self.assertRaises(ValueError): self.store.complete(job,{'annotations':[]},now=2)
        self.assertEqual(self.store.status()['jobs'][0]['status'],'running')

    def test_quota_pauses_provider_without_consuming_retry_budget(self):
        self.enqueue(); job=self.store.claim('minimax',now=1)
        self.store.fail(job,'quota',retry_after=100,now=2)
        self.assertIsNone(self.store.claim('minimax',now=101))
        resumed=self.store.claim('minimax',now=103)
        self.assertIsNotNone(resumed); self.assertEqual(resumed['retry_count'],0)

    def test_auth_requires_explicit_resume(self):
        self.enqueue(); job=self.store.claim('minimax',now=1); self.store.fail(job,'auth',now=2)
        self.assertIsNone(self.store.claim('minimax',now=100000))
        self.store.resume_provider('minimax')
        self.assertIsNotNone(self.store.claim('minimax',now=100001))

    def test_transient_retries_are_bounded_to_three(self):
        self.enqueue()
        for i in range(4):
            job=self.store.claim('minimax',now=100*i+1); self.assertIsNotNone(job)
            self.store.fail(job,'transient',now=100*i+2)
        self.assertIsNone(self.store.claim('minimax',now=1000))
        self.assertEqual(self.store.status()['jobs'][0]['status'],'failed')

    def test_expired_worker_is_uncertain_and_cannot_complete(self):
        self.enqueue(); job=self.store.claim('minimax',lease_seconds=5,now=1)
        self.assertIsNone(self.store.claim('minimax',now=7))
        self.assertEqual(self.store.status()['jobs'][0]['status'],'uncertain')
        with self.assertRaises(ValueError): self.store.complete(job,labels(),now=8)

    def test_atomic_claim_has_one_owner(self):
        self.enqueue()
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs=list(pool.map(lambda _:QueueStore(self.path).claim('minimax',now=1),range(4)))
        self.assertEqual(sum(job is not None for job in jobs),1)

    def test_source_and_template_groups_cannot_cross_splits(self):
        with self.assertRaises(ValueError): self.store.register_source('source1','final','group1')
        with self.assertRaises(ValueError): self.store.register_source('source2','final','group1')
        self.store.register_source('source2','final','group2')

    def test_unreviewed_inputs_are_not_sent(self):
        del self.x['privacy_review']
        with self.assertRaises(ValueError): self.enqueue()


if __name__=='__main__': unittest.main()
