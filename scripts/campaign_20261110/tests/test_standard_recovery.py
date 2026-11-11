import unittest
from unittest.mock import patch
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
 def test_newest_worker_diagnosis_wins_in_either_file_order(self):
  old={'campaign_id':CAMPAIGN,'observed_at':'2026-11-11T10:00:00Z','owner':'/root',
   'classes':[{'class_id':self.key,'state':'running','blocker':'old','owner':'/root'}]}
  new={'campaign_id':CAMPAIGN,'observed_at':'2026-11-11T11:00:00+00:00','owner':'/root/dataset_contract',
   'classes':[{'class_id':self.key,'state':'failed','blocker':'actual replay failure','owner':'/root/dataset_contract'}]}
  for docs in [[old,new],[new,old]]:
   row=combine(self.baseline,self.current,{},docs)['classes'][0]
   self.assertEqual(row['blocker'],'actual replay failure');self.assertEqual(row['state'],'failed')
   self.assertEqual(row['owner'],'/root/dataset_contract');self.assertEqual(row['worker_observed_at'],new['observed_at'])
   self.assertEqual(row['accepted_standard'],0)
 def test_undated_worker_cannot_replace_dated_observation(self):
  docs=[{'campaign_id':CAMPAIGN,'observed_at':'2026-11-11T11:00:00Z','classes':[{'class_id':self.key,'blocker':'fresh'}]},
   {'campaign_id':CAMPAIGN,'classes':[{'class_id':self.key,'blocker':'undated'}]}]
  self.assertEqual(combine(self.baseline,self.current,{},docs)['classes'][0]['blocker'],'fresh')
 def test_delivered_heel_tight_has_compute_owner_and_requires_fresh_validation(self):
  k='inter_object_relation_satisfaction-under_large_scale_01'
  self.baseline['classes'][0]['class_id']=k;self.current['classes'][0]['class_id']=k
  row=combine(self.baseline,self.current,{},[])['classes'][0]
  self.assertEqual(row['owner'],'/root/compute');self.assertEqual(row['state'],'unattempted')
  self.assertEqual(row['accepted_standard'],0)
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
 @patch('scripts.campaign_20261110.standard_recovery.PARKED_CLASSES', {'inter_object_relation_satisfaction-under_large_scale_01'})
 def test_promised_artifacts_remain_parked(self):
  k='inter_object_relation_satisfaction-under_large_scale_01'
  self.baseline['classes'][0]['class_id']=k;self.current['classes'][0]['class_id']=k
  d=combine(self.baseline,self.current,{},[{'campaign_id':CAMPAIGN,'classes':[{'class_id':k,'state':'running'}]}])
  self.assertEqual(d['classes'][0]['state'],'parked')
 def test_owner_released_classes_have_engineering_owners_without_automatic_acceptance(self):
  for k,owner in [('touching_03','/root'),('under_large_scale_02','/root/compute'),('under_large_scale_03','/root/templates_assets'),('nextto_04','/root/templates_assets')]:
   k='inter_object_relation_satisfaction-'+k
   self.baseline['classes'][0]['class_id']=k;self.current['classes'][0]['class_id']=k
   row=combine(self.baseline,self.current,{},[])['classes'][0]
   self.assertEqual(row['state'],'unattempted');self.assertEqual(row['owner'],owner)
   self.assertEqual(row['accepted_standard'],0)
 def test_received_artifact_lifts_only_its_park_without_inventing_acceptance(self):
  k='inter_object_relation_satisfaction-under_large_scale_03'
  self.baseline['classes'][0]['class_id']=k;self.current['classes'][0]['class_id']=k
  d=combine(self.baseline,self.current,{},[],artifact_updates={k:{'owner':'/root/templates_assets'}})
  self.assertEqual(d['classes'][0]['state'],'unattempted')
  self.assertEqual(d['classes'][0]['owner'],'/root/templates_assets')
  self.assertEqual(d['classes'][0]['accepted_standard'],0)
  self.assertEqual(d['newly_covered'],[])
 @patch('scripts.campaign_20261110.standard_recovery.PARKED_CLASSES', {'inter_object_relation_satisfaction-under_large_scale_01'})
 def test_missing_artifact_update_does_not_clear_explicit_hold(self):
  k='inter_object_relation_satisfaction-under_large_scale_01'
  self.baseline['classes'][0]['class_id']=k;self.current['classes'][0]['class_id']=k
  d=combine(self.baseline,self.current,{},[],artifact_updates={'inter_object_relation_satisfaction-under_large_scale_03':{}})
  self.assertEqual(d['classes'][0]['state'],'parked')
 @patch('scripts.campaign_20261110.standard_recovery.PARKED_CLASSES', {'inter_object_relation_satisfaction-under_large_scale_01'})
 def test_new_hold_preserves_historical_count_but_excludes_validation(self):
  k='inter_object_relation_satisfaction-under_large_scale_01'
  self.baseline['classes'][0]['class_id']=k;self.current['classes'][0].update(class_id=k,accepted=1)
  row=combine(self.baseline,self.current,{},[])['classes'][0]
  self.assertEqual(row['accepted_standard'],1)
  self.assertEqual(row['state'],'parked')
  self.assertIs(row['canonical_validation_eligible'],False)
 def test_current_eligibility_overrides_history_and_reports_review(self):
  self.current['classes'][0]['accepted']=3
  projection={'schema':'isg-current-eligibility-v1','campaign_id':CAMPAIGN,'observed_at':'2026-11-11T06:00:00+00:00',
   'review_inventory_complete':False,'classes':[{'class_id':self.key,'split':'eval_standard','historical':3,'current_eligible':0,'under_review':3}]}
  d=combine(self.baseline,self.current,{},[],eligibility=projection)
  self.assertEqual(d['covered_classes'],0);self.assertEqual(d['zero_classes'],1)
  self.assertEqual(d['classes'][0]['historical_accepted_standard'],3)
  self.assertEqual(d['classes'][0]['accepted_standard'],0)
  self.assertEqual(d['acceptance_observed_at'],projection['observed_at'])
  self.assertEqual(d['newly_covered'],[])
 def test_newly_zero_class_enters_ledger_even_if_absent_from_original_baseline(self):
  self.current['classes'].append({'class_id':'new_zero','task':'other','split':'eval_standard','accepted':2})
  projection={'schema':'isg-current-eligibility-v1','campaign_id':CAMPAIGN,'observed_at':'now','classes':[]}
  d=combine(self.baseline,self.current,{},[],eligibility=projection)
  self.assertEqual(d['baseline_zero_classes'],1);self.assertEqual(d['zero_classes'],2)
  self.assertEqual(len(d['classes']),2)
  row=next(r for r in d['classes'] if r['class_id']=='new_zero')
  self.assertIsNone(row['baseline_accepted']);self.assertEqual(row['historical_accepted_standard'],2)
 def test_foreign_eligibility_projection_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'invalid current eligibility'):
   combine(self.baseline,self.current,{},[],eligibility={'campaign_id':'old'})
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
