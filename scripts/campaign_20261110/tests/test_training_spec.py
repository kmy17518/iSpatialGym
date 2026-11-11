from scripts.campaign_20261110.training_spec import executed_audit
from scripts.campaign_20261110.contracts import write_json_atomic


def test_empty_checklist_evidence_is_unknown_not_failed_or_verified(tmp_path):
    value=executed_audit(tmp_path,tmp_path,[])
    assert value['counts']=={'untested':135}
    assert not value['production_authorization']
    assert all(row['status']=='untested' and not row['evidence'] for row in value['requirements'])
    assert all(row['owners'] and row['next_action'] and row['exit_criteria'] for row in value['requirements'])


def test_canonical_partial_scope_does_not_verify_training_paths(tmp_path):
    write_json_atomic(tmp_path/'infra/standard_recovery/canonical_validation.json',
        {'campaign_id':'campaign_20261110','observed_at':'2026-11-11T00:00:00Z','validated':97,'total':100})
    value=executed_audit(tmp_path,tmp_path,[{'path':'original','sha256':'x'}])
    rows={row['id']:row for row in value['requirements']}
    assert rows['G15']['status']=='partial'
    assert rows['G15']['evidence'][0]['facts']['validated']==97
    assert rows['G16']['status']=='untested' and not rows['G16']['evidence']
    assert rows['G15']['chain']['verification']['complete_requirement'] is False
    assert rows['G15']['source_requirements']==[{'path':'original','sha256':'x'}]


def test_explicit_untested_family_observation_is_not_partial_success(tmp_path):
    write_json_atomic(tmp_path/'training_pipeline_canaries/pipeline_status.json',
        {'campaign_id':'campaign_20261110','family_paths':[{'task':'configuration_matching-attachment',
        'exact_configurations_with_demo_physics_export':0}]})
    value=executed_audit(tmp_path,tmp_path,[])
    rows=[r for r in value['requirements'] if r.get('tasks')==['configuration_matching-attachment']]
    assert len(rows)==5
    assert all(r['status']=='untested' and r['scope_status']=='unknown_no_executed_evidence' for r in rows)


def test_large_source_records_stay_in_source_not_requirement_payload(tmp_path):
    from scripts.campaign_20261110.training_spec import compact_facts,compact_family
    huge={'status':'partial','records':[{'blob':'x'*10000}]*1000,'summary':{'required':13712},'artifact_requests':[{}]*1000}
    assert compact_facts(huge)=={'status':'partial','summary':{'required':13712}}
    family={'task':'disembedding','aba_restoration':{'physics':{'status':'failed','attempts':[huge]*100}},'exact_validation':{'status':'partial','evidence':[huge]*100}}
    import json
    assert len(json.dumps(compact_family(family)))<1000
