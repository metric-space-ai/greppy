import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contracts import digest, sanitized_example, teacher_configuration
from corpus import target_examples
from queue_store import QueueStore
from teachers import ProviderFailure, minimax, grok
from test_corpus_admission import source
from web_records import observation_examples


class WebRecordsTests(unittest.TestCase):
    def envelope(self):
        return {'schema':'greppy.web-runtime.v1','operation':'web.observe','status':'ok',
                'result':{'actionables':[{'ref':'@1','disabled':False,'text':'on'},
                                         {'ref':'@2','checked':False,'disabled':True}],
                          'unknown_extension':{'future_state':None},'refs_truncated':True,
                          'continuation':{'cursor':'next'}}}

    def examples(self,envelope,**changes):
        args={'source_id':'episode1','goal':'Enable the product and set quantity to 3',
              'goal_version':1,'last_action':'check @1','group_key':'fixture-family',
              'privacy_review':'synthetic'}
        args.update(changes)
        return observation_examples(envelope,**args)

    def test_unknown_state_and_continuation_preserved_without_mutating_source(self):
        envelope=self.envelope(); before=copy.deepcopy(envelope)
        examples=self.examples(envelope)
        rows=examples[0]['records']
        first=json.loads(rows[0]['text'])
        self.assertNotIn('checked',first)
        self.assertFalse(json.loads(rows[1]['text'])['checked'])
        self.assertTrue(all(r['protected'] for r in rows))
        self.assertTrue(any(r['json_pointer']=='/result/continuation' for r in rows))
        self.assertEqual(envelope,before)

    def test_goal_versions_separate_scores_but_keep_source_records(self):
        first=self.examples(self.envelope())[0]
        second=self.examples(self.envelope(),goal_version=2)[0]
        self.assertNotEqual(first['id'],second['id'])
        self.assertEqual(first['records'],second['records'])
        self.assertNotEqual(digest(sanitized_example(first)),digest(sanitized_example(second)))

    def test_no_goal_no_ranker_and_missing_action_not_inferred(self):
        self.assertEqual(self.examples(self.envelope(),goal=None),[])
        with self.assertRaises(ValueError): self.examples(self.envelope(),last_action='')


class ConfigurationTests(unittest.TestCase):
    def test_configuration_and_full_prompt_change_job_key(self):
        with tempfile.TemporaryDirectory() as temp:
            store=QueueStore(Path(temp)/'queue.sqlite')
            example=target_examples(source())[0]
            store.register_source('s1','train','g')
            original=store.enqueue('minimax','MiniMax-M3',[example])
            self.assertEqual(original,store.enqueue('minimax','MiniMax-M3',[example]))
            changed={**teacher_configuration('minimax'),'max_output_tokens':8192}
            with patch('queue_store.teacher_configuration',return_value=changed):
                self.assertNotEqual(original,store.enqueue('minimax','MiniMax-M3',[example]))
            with patch('contracts.response_schema',return_value={'description':'new schema revision'}):
                self.assertNotEqual(original,store.enqueue('minimax','MiniMax-M3',[example]))
            job=store.claim('minimax')
            self.assertIn('max_output_tokens',job['configuration'])

    def test_unbound_legacy_configuration_cannot_dispatch(self):
        for provider in (minimax,grok):
            with self.assertRaises(ProviderFailure) as raised: provider({'configuration':{}})
            self.assertEqual(raised.exception.kind,'permanent')


if __name__=='__main__': unittest.main()
