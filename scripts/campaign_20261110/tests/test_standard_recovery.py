import unittest
from scripts.campaign_20261110.standard_recovery import combine, CAMPAIGN

class RecoveryLedgerTests(unittest.TestCase):
 def setUp(self):
  self.key='disembedding_07'
  self.baseline={'campaign_id':CAMPAIGN,'observed_at':'2026-11-11T03:55:50+00:00','classes':[{'class_id':self.key,'task':'disembedding','accepted':0}]}
  self.current={'campaign_id':CAMPAIGN,'observed_at':'2026-11-11T04:00:00+00:00','classes':[{'class_id':self.key,'task':'disembedding','split':'eval_standard','accepted':0}]}
 def test_worker_pass_cannot_invent_acceptance(self):
  doc={'campaign_id':CAMPAIGN,'classes':[{'class_id':self.key,'state':'accepted','acceptance':{'accepted_standard':10}}]}
  d=combine(self.baseline,self.current,{},[doc]);self.assertEqual(d['newly_covered'],[])
  self.assertEqual(d['classes'][0]['accepted_standard'],0);self.assertEqual(d['classes'][0]['state'],'failed')
 def test_first_coverage_requires_authoritative_count_and_retains_time(self):
  self.current['classes'][0]['accepted']=1
  d=combine(self.baseline,self.current,{},[]);self.assertEqual(d['newly_covered'],[self.key])
  self.current['observed_at']='2026-11-11T05:00:00+00:00'
  e=combine(self.baseline,self.current,{},[],d)
  self.assertEqual(e['classes'][0]['first_accepted_observed_at'],'2026-11-11T04:00:00+00:00')
 def test_missing_count_is_unavailable_not_zero(self):
  self.current['classes']=[]
  self.assertIsNone(combine(self.baseline,self.current,{},[])['classes'][0]['accepted_standard'])
 def test_foreign_worker_evidence_is_ignored(self):
  d=combine(self.baseline,self.current,{},[{'campaign_id':'older','classes':[{'class_id':self.key,'blocker':'wrong'}]}])
  self.assertNotEqual(d['classes'][0]['blocker'],'wrong')
 def test_promised_artifacts_remain_parked(self):
  k='inter_object_relation_satisfaction-touching_03'
  self.baseline['classes'][0]['class_id']=k;self.current['classes'][0]['class_id']=k
  d=combine(self.baseline,self.current,{},[{'campaign_id':CAMPAIGN,'classes':[{'class_id':k,'state':'running'}]}])
  self.assertEqual(d['classes'][0]['state'],'parked')

if __name__=='__main__':unittest.main()
