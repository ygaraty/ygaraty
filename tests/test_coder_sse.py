import importlib.util,json,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('bench',Path(__file__).parents[1]/'tools/benchmark_coder_backends.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class SSETests(unittest.TestCase):
 def test_content_usage_and_termination(self):
  lines=['data: '+json.dumps(x) for x in [{'choices':[{'delta':{'content':'Hi'},'finish_reason':None}]},{'choices':[{'delta':{'content':' there'},'finish_reason':'stop'}]},{'choices':[],'usage':{'completion_tokens':2}}]]+['data: [DONE]']
  r=m.consume_sse(lines,lambda:1.0);self.assertEqual(r['text'],'Hi there');self.assertTrue(r['done']);self.assertEqual(r['usage']['completion_tokens'],2);self.assertEqual(r['chunks'],2)
 def test_tool_fragments(self):
  calls=[{'index':0,'id':'call_1','function':{'name':'lookup_ticket','arguments':'{"ticket_'}},{'index':0,'function':{'arguments':'id":"KAA-42"}'}}]
  r=m.consume_sse(['data: '+json.dumps({'choices':[{'delta':{'tool_calls':[c]},'finish_reason':None}]}) for c in calls]+['data: [DONE]'])
  self.assertEqual(json.loads(r['tool_calls'][0]['function']['arguments']),{'ticket_id':'KAA-42'})
 def test_missing_termination_is_not_success(self):self.assertFalse(m.consume_sse([])['done'])
if __name__=='__main__':unittest.main()
