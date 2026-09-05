import copy
import tempfile
import unittest
from pathlib import Path
from admission import audit_queue
from contracts import SCHEMA, digest, sanitized_example
from corpus import grouped_splits, source_spans, target_examples, template_hash, verify_spans
from queue_store import QueueStore


def source(sid='s1', relation='repo-a', text='error: failure\n  source();\n'):
    return {'id':sid, 'text':text, 'text_hash':digest(text), 'template_hash':template_hash(text),
            'relation_keys':[relation], 'family':'build', 'task':'Find the cause of the build failure',
            'split':'train', 'group_key':'g', 'capture_complete':True, 'privacy_review':'synthetic'}


class CorpusTests(unittest.TestCase):
    def test_byte_exact_unicode_crlf_blank_and_no_final_newline(self):
        text = 'ä\r\n\nlast'
        records = source_spans(text, 's')
        verify_spans(text, records)
        self.assertEqual([(r['byte_start'],r['byte_end']) for r in records], [(0,4),(4,5),(5,9)])
        bad = copy.deepcopy(records); bad[1]['text'] = ' '
        with self.assertRaises(ValueError): verify_spans(text,bad)

    def test_all_lines_are_targets_once_context_does_not_inherit(self):
        examples = target_examples(source(),max_targets=1)
        self.assertEqual(len(examples),2)
        self.assertEqual(examples[0]['context'][0]['id'],examples[1]['records'][0]['id'])
        self.assertNotEqual(examples[0]['records'][0]['id'],examples[1]['records'][0]['id'])
        self.assertNotIn('severity',examples[1]['records'][0])

    def test_oversized_context_is_held_not_truncated(self):
        with self.assertRaisesRegex(ValueError,'without truncation'):
            target_examples(source(text='x'*60000),max_chars=40000)

    def test_unknown_capture_not_admitted(self):
        s=source(); s['capture_complete']=None
        with self.assertRaises(ValueError): target_examples(s)

    def test_transitive_relations_and_templates(self):
        a=source('a','repo-a','run 123'); b=source('b','repo-b','run 456')
        c=source('c','repo-b','different')
        splits=grouped_splits([a,b,c],seed='v1')
        self.assertEqual(len({r['group_key'] for r in splits.values()}),1)
        self.assertEqual(splits,grouped_splits([c,a,b],seed='v1'))

    def test_frozen_split_bridge_is_rejected(self):
        a=source('a','repo-a','one'); b=source('b','repo-b','two')
        old={'a':{'split':'train'},'b':{'split':'development'}}
        c=source('c','repo-a','two')
        with self.assertRaisesRegex(ValueError,'cross'):
            grouped_splits([a,b,c],seed='v1',frozen=old)

    def test_existing_split_survives_growth(self):
        a=source('a'); old=grouped_splits([a],seed='v1')
        b=source('b')
        new=grouped_splits([a,b],seed='v2',frozen=old)
        self.assertEqual(new['a']['split'],old['a']['split'])
        self.assertEqual(new['a']['group_key'],old['a']['group_key'])

    def test_old_or_unknown_sources_cannot_be_final(self):
        a=source(); a['requested_split']='final'
        with self.assertRaises(ValueError): grouped_splits([a],seed='v1')
        a['previously_exposed']=False
        self.assertEqual(grouped_splits([a],seed='v1')['s1']['split'],'final')


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store=QueueStore(Path(self.tmp.name)/'queue.sqlite')
        self.example=target_examples(source())[0]
        self.store.register_source('s1','train','g')

    def done(self,provider,*,conflict=False,model=None):
        self.store.enqueue(provider,model or ('MiniMax-M3' if provider=='minimax' else 'grok-4.6'),[self.example])
        job=self.store.claim(provider)
        rows=[{'example_id':self.example['id'],'record_id':r['id'],'severity':'error' if i==0 else 'text',
               'relevance':3 if i==0 else 2,'evidence_ids':[r['id']],'reason':'Observed source record','ambiguous':False}
              for i,r in enumerate(self.example['records'])]
        if conflict: rows[0]['severity']='text'
        self.store.complete(job,{'annotations':rows})

    def receipt(self):
        return {self.example['id']:{'example_sha256':digest(sanitized_example(self.example)),
                'evidence_artifact_sha256':'a'*64,'status':'pass','complete_capture':True,'privacy_admitted':True}}

    def test_agreement_without_oracle_is_held(self):
        self.done('minimax'); self.done('grok')
        result=audit_queue(self.store,{})
        self.assertEqual(result['counts'],{'review_complete':0,'held':1})
        self.assertIn('independent_evidence_missing',result['examples'][0]['reasons'])

    def test_disagreement_cannot_be_overridden_by_success_receipt(self):
        self.done('minimax'); self.done('grok',conflict=True)
        result=audit_queue(self.store,self.receipt())
        self.assertEqual(result['counts']['held'],1)
        self.assertEqual(result['examples'][0]['conflicts'][0]['fields'],['severity'])

    def test_missing_blind_judge_and_stale_receipt_are_held(self):
        self.done('minimax')
        receipt=self.receipt(); receipt[self.example['id']]['example_sha256']='b'*64
        reasons=audit_queue(self.store,receipt)['examples'][0]['reasons']
        self.assertIn('grok_missing',reasons)
        self.assertIn('independent_evidence_input_mismatch',reasons)

    def test_review_completion_requires_both_teachers_and_matching_evidence(self):
        self.done('minimax'); self.done('grok')
        self.assertEqual(audit_queue(self.store,self.receipt())['counts']['review_complete'],1)

    def test_multiple_teacher_versions_not_silently_selected(self):
        self.done('minimax'); self.done('grok'); self.done('grok',model='other')
        self.assertIn('grok_multiple_versions',audit_queue(self.store,self.receipt())['examples'][0]['reasons'])


if __name__=='__main__': unittest.main()
