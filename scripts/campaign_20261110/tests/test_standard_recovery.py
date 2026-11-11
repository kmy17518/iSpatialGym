import unittest
import json
from pathlib import Path
import tempfile
from scripts.campaign_20261110.standard_recovery import combine, CAMPAIGN, received_artifacts
from scripts.campaign_20261110.contracts import file_digest

class RecoveryLedgerTests(unittest.TestCase):
 def setUp(self):
  self.key='disembedding_07'
  self.baseline={'campaign_id':CAMPAIGN,'observed_at':'2026-11-11T03:55:50+00:00','classes':[{'class_id':self.key,'task':'disembedding','accepted':0}]}
  self.current={'campaign_id':CAMPAIGN,'observed_at':'2026-11-11T04:00:00+00:00','classes':[{'class_id':self.key,'task':'disembedding','split':'eval_standard','accepted':0}]}
 def test_worker_pass_cannot_invent_acceptance(self):
  doc={'campaign_id':CAMPAIGN,'classes':[{'class_id':self.key,'state':'accepted','acceptance':{'accepted_standard':10}}]}
  d=combine(self.baseline,self.current,{},[doc]);self.assertEqual(d['newly_covered'],[])
  self.assertEqual(d['classes'][0]['accepted_standard'],0);self.assertEqual(d['classes'][0]['state'],'failed')
 def test_reference_diagnosis_is_preserved_without_claiming_acceptance(self):
  diagnosis={'reference':{'demo_passed':True,'sha256':'verified-reference'},
    'first_failure_stage':'acceptance/orchestration','hypothesis':'missing distinct sibling',
    'next_discriminating_experiment':'Run own strict sibling gate.'}
  doc={'campaign_id':CAMPAIGN,'classes':[{'class_id':self.key,'state':'awaiting_acceptance','diagnosis':diagnosis}]}
  d=combine(self.baseline,self.current,{},[doc]);row=d['classes'][0]
  self.assertEqual(row['diagnosis'],diagnosis)
  self.assertEqual(row['accepted_standard'],0);self.assertEqual(d['newly_covered'],[])
  self.assertEqual(row['state'],'awaiting_acceptance')
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
 def test_received_artifact_lifts_only_its_park_without_inventing_acceptance(self):
  k='inter_object_relation_satisfaction-under_large_scale_03'
  self.baseline['classes'][0]['class_id']=k;self.current['classes'][0]['class_id']=k
  d=combine(self.baseline,self.current,{},[],artifact_updates={k:{'owner':'/root/templates_assets'}})
  self.assertEqual(d['classes'][0]['state'],'unattempted')
  self.assertEqual(d['classes'][0]['owner'],'/root/templates_assets')
  self.assertEqual(d['classes'][0]['accepted_standard'],0)
  self.assertEqual(d['newly_covered'],[])
 def test_missing_artifact_update_does_not_clear_explicit_hold(self):
  k='inter_object_relation_satisfaction-under_large_scale_01'
  self.baseline['classes'][0]['class_id']=k;self.current['classes'][0]['class_id']=k
  d=combine(self.baseline,self.current,{},[],artifact_updates={'inter_object_relation_satisfaction-under_large_scale_03':{}})
  self.assertEqual(d['classes'][0]['state'],'parked')
 def test_new_hold_preserves_historical_count_but_excludes_validation(self):
  k='inter_object_relation_satisfaction-nextto_04'
  self.baseline['classes'][0]['class_id']=k;self.current['classes'][0].update(class_id=k,accepted=1)
  row=combine(self.baseline,self.current,{},[])['classes'][0]
  self.assertEqual(row['accepted_standard'],1)
  self.assertEqual(row['state'],'parked')
  self.assertIs(row['canonical_validation_eligible'],False)
 def test_artifact_audit_requires_unchanged_ready_source_proof(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);target=root/'infra/standard_recovery';target.mkdir(parents=True)
   key='inter_object_relation_satisfaction-nextto_04';proof=target/'audit.json'
   audit={'campaign_id':CAMPAIGN,'classes':[{'class_id':key,'all_ready_and_exact_assets':True,'exact_source_template':True,'errors':[],'reservations':'passed'}]}
   proof.write_text(json.dumps(audit))
   update={'campaign_id':CAMPAIGN,'artifacts':[{'class_id':key,'owner_instruction':'upstream supplied','source_commit':'2a33c696',
      'audit':{'path':str(proof),'sha256':file_digest(proof)}}]}
   (target/'owner_artifact_updates.json').write_text(json.dumps(update))
   self.assertIn(key,received_artifacts(root))
   (root/'controllers').mkdir()
   (root/'controllers/training_production_gate.json').write_text(json.dumps({'campaign_id':CAMPAIGN,'parked_classes':[key]}))
   self.assertNotIn(key,received_artifacts(root))
   (root/'controllers/training_production_gate.json').unlink()
   audit['classes'][0]['all_ready_and_exact_assets']=False;proof.write_text(json.dumps(audit))
   with self.assertRaisesRegex(ValueError,'audit changed'):received_artifacts(root)

if __name__=='__main__':unittest.main()
