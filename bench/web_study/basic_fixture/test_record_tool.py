import json
import unittest
from record_tool import collect
from records_fixture import initial, oracle


def envelope(result, status='ok'):
    return json.dumps({'schema':'greppy.web-runtime.v1','status':status,'request_id':'request','result':result})

class RecordToolTests(unittest.TestCase):
    def call(self, replies):
        remaining = list(replies)
        def invoke(argv, stdin):
            self.last_argv, self.last_stdin = argv, stdin
            return remaining.pop(0)
        return invoke

    def observation(self):
        return envelope({'ref_count':1,'refs_truncated':False,'actionables':[{'ref':'@1','name':'Offer','role':'button'}]})

    def test_error_not_filtered_into_empty_success(self):
        result, code = collect('/delegate','css=#work',[],'visible=true',self.call([(0,envelope({},'error'),'')]))
        self.assertEqual(code,1)
        self.assertEqual(result['error']['code'],'UPSTREAM_RESULT')
        self.assertEqual(result['records'],[])

    def test_stale_ref_failure_preserves_upstream_error(self):
        result, code = collect('/delegate','css=#work',[],'',self.call([
            (0,self.observation(),''),(1,'STALE_REF','not resolved')]))
        self.assertEqual(code,1)
        self.assertEqual(result['error']['source']['stdout'],'STALE_REF')

    def test_visibility_is_not_inferred(self):
        result, code = collect('/delegate','css=#work',[],'',self.call([
            (0,self.observation(),''),(0,envelope({'value':{'node':{'attrs':{}}}}),'')]))
        self.assertEqual(code,1)
        self.assertEqual(result['error']['code'],'SHAPE')

    def test_missing_price_is_null(self):
        result, code = collect('/delegate','css=#work',['data-price'],'',self.call([
            (0,self.observation(),''),(0,envelope({'session_id':'s','value':{'node':{'visible':True,'attrs':{}}}}),'')]))
        self.assertEqual(code,0)
        self.assertIsNone(result['records'][0]['data-price'])
        self.assertEqual(result['records'][0]['ref'],'@1')

    def test_sensitive_attribute_not_requested(self):
        result, code = collect('/delegate','css=#work',['value'],'',self.call([]))
        self.assertEqual(code,1)
        self.assertEqual(result['processing']['native_commands'],0)

    def test_truncation_not_claimed_complete(self):
        result, code = collect('/delegate','css=#work',[],'',self.call([
            (0,envelope({'refs_truncated':True,'actionables':[]}),'')]))
        self.assertEqual(code,1)
        self.assertEqual(result['error']['code'],'INCOMPLETE')

    def test_session_switch_rejected(self):
        observation=envelope({'ref_count':2,'actionables':[{'ref':'@1'},{'ref':'@2'}]})
        detail=lambda s: envelope({'session_id':s,'value':{'node':{'visible':True,'attrs':{}}}})
        result,code=collect('/delegate','css=#work',[],'',self.call([
            (0,observation,''),(0,detail('one'),''),(0,detail('two'),'')]))
        self.assertEqual(code,1)
        self.assertEqual(result['error']['code'],'CONTEXT')

    def test_oracle_rejects_hidden_cheapest_and_duplicate_save(self):
        state=initial('test','test')
        state['booking']={'offer':'hidden','note':'Leave with reception','accepted':True}
        state['events']=[{},{}]
        state['reloads_after_save']=1
        checks=oracle(state)
        self.assertFalse(checks['correct_offer'])
        self.assertFalse(checks['one_save'])
        self.assertTrue(checks['note'])

if __name__=='__main__': unittest.main()
