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


def test_pool_mechanics_and_failed_restore_stay_narrow_partial_evidence(tmp_path):
    write_json_atomic(tmp_path/'training_preparation/pool_pair_r1/status.json',{
        'campaign_id':'campaign_20261110','observed_at':'2026-11-11T16:00:00Z','status':'repair_prepared',
        'required_model_count':2,'attempts':[{'job_id':'1','stage':'native_pool_switch','mechanics_passed':True,
        'independent_load_render_passed':False,'observed_stages':3,'failure':{'message':'teardown failed'}}],
        'unverified':['TRO_integration','physics_action_replay'],'next_action':'Run corrected restore.'})
    row=next(r for r in executed_audit(tmp_path,tmp_path,[])['requirements']if r['id']=='G16')
    assert row['status']=='partial'and not row['chain']['verification']['complete_requirement']
    facts=row['evidence'][0]['facts']
    assert facts['attempts'][0]['failure']['message']=='teardown failed'
    assert facts['unverified']==['TRO_integration','physics_action_replay']
    assert row['next_action']=='Run corrected restore.'


def test_nested_pool_results_keep_verified_stages_and_old_failure_without_native_payload(tmp_path):
    import json
    evidence={'path':'/campaign/switch/result.json','sha256':'a'*64}
    write_json_atomic(tmp_path/'training_preparation/pool_pair_r1/status.json',{
        'campaign_id':'campaign_20261110','status':'scoped_mechanics_verified','required_model_count':2,
        'attempts':[{'job_id':'old','failure':{'message':'pose differs'},'observed_stages':0},
            {'job_id':'new','stage':'switch','evidence':evidence,'result':{'mechanics_passed':True,
                'stages':[{'large_capture':'x'*10000}]*3}},
            {'job_id':'new','stage':'restore','result':{'independent_load_render_passed':True,'stages':[{}, {}, {}]}}],
        'verification':{'switch':evidence,'stages_each':3,'clean_exit':True},
        'process_accounting':{'job_id':'new','record':['new','COMPLETED','0:0','1160'],'query_returncode':0},
        'unverified':['TRO_integration','oracle_demo','physics_action_replay']})
    row=next(r for r in executed_audit(tmp_path,tmp_path,[])['requirements']if r['id']=='G16')
    facts=row['evidence'][0]['facts']
    assert row['status']=='partial'and not row['chain']['verification']['complete_requirement']
    assert facts['attempts'][0]['failure']['message']=='pose differs'
    assert facts['attempts'][1]['mechanics_passed'] is True
    assert facts['attempts'][2]['independent_load_render_passed'] is True
    assert [r['observed_stages']for r in facts['attempts']]==[0,3,3]
    assert facts['attempts'][1]['evidence']==facts['verification']['switch']==evidence
    assert facts['verification']['clean_exit'] is True
    assert facts['process_accounting']['record'][1:3]==['COMPLETED','0:0']
    assert len(json.dumps(facts))<2000
