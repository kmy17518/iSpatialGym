import json
from pathlib import Path
from scripts.campaign_20261110.infra_vast.cost_provenance import scope, rental_intent, retain


def job(tmp_path):
    p=tmp_path/'job';p.mkdir()
    (p/'job.env').write_text('ISGE_CODE=/code/trunk_abc\nALIAS_KEY=test_01\nFAMILY=test\nAWS_SECRET_ACCESS_KEY=never-record-this\n')
    return {'id':'123','job_dir':str(p),'gpu':'RTX 4090'}


def test_exact_queue_and_no_credentials(tmp_path):
    j=job(tmp_path);p=Path(j['job_dir'])
    config=p/'config.json';config.write_text(json.dumps({'campaign_id':'campaign_20261110','split':'train','template':{'task':'test','class_id':'test_01'}}))
    queue=p/'queue.json';queue.write_text(json.dumps({'campaign_id':'campaign_20261110','group_ids':['g2','g1'],'schema':'queue'}))
    with (p/'job.env').open('a') as f:f.write(f'ISPATIALGYM_CAMPAIGN_CONFIG={config}\nISPATIALGYM_CAMPAIGN_GOAL_QUEUE={queue}\n')
    value=scope(j)
    assert value['group_ids']==['g1','g2'] and value['split']=='train'
    assert 'never-record-this' not in json.dumps(value)
    intent=rental_intent([j],'RTX 4090')
    assert intent['exclusive_cost_assignment'] is False and len(intent['candidate_jobs'])==1


def test_replay_exact_source_group(tmp_path):
    j=job(tmp_path);p=Path(j['job_dir']);tro=p/'tro.json'
    tro.write_text(json.dumps({'ispatialgym':{'campaign':{'goal_group_id':'original'}}}))
    (p/'replay_inputs.json').write_text(json.dumps({'source_tro':str(tro),'source_job':'/source','snapshot':'/original','files':{str(tro):'pinned'}}))
    value=scope(j)
    assert value['group_ids']==['original'] and value['source_tro_sha256']=='pinned'
    assert value['source_job_dir']=='/source' and value['runtime_snapshot']=='/original'


def test_prerequisite_not_an_accepted_group(tmp_path):
    j=job(tmp_path);p=Path(j['job_dir'])
    (p/'native_inputs.json').write_text(json.dumps({'original_entry':{'class_id':'native_01'},'source_manifest':'/manifest','files':{'/manifest':'hash'},'entry_index':2,'fresh_start_index':500,'snapshot':'/native'}))
    value=scope(j)
    assert value['group_binding']=='prerequisite_candidate_only' and value['group_ids']==[]
    assert value['source_manifest_sha256']=='hash' and value['source_entry_index']==2


def test_missing_evidence_is_explicit(tmp_path):
    value=scope({'id':'1','job_dir':str(tmp_path/'absent')})
    assert value['status']=='incomplete' and value['errors']
    assert rental_intent([],'RTX 4090')['purpose']=='unassigned_shared_campaign_capacity'


def test_immutable_transition_idempotent_with_lost_bounds(tmp_path):
    payload={'end_kind':'worker_lost','last_observed_unix':12,'exact_runtime_end_unavailable':True}
    a=retain(tmp_path,'assignment_end','job_try',payload)
    b=retain(tmp_path,'assignment_end','job_try',{**payload,'last_observed_unix':15})
    assert a==b
    record=json.loads(Path(a['path']).read_text())
    assert record['last_observed_unix']==12 and record['exact_runtime_end_unavailable']


def test_direct_goal_plan_preserves_exact_group_and_input_hash(tmp_path):
    import hashlib
    j = job(tmp_path); p = Path(j['job_dir'])
    config = p / 'config.json'
    config.write_text(json.dumps({'campaign_id': 'campaign_20261110', 'split': 'eval_standard',
        'template': {'task': 'test', 'class_id': 'test_01'}, 'validation_key': {'template_sha256': 'a'*64}}))
    plan = p / 'plan.json'
    plan.write_text(json.dumps({'campaign_id': 'campaign_20261110', 'group_id': 'direct_standard_group',
        'task': 'test', 'split': 'eval_standard', 'contract_template_sha256': 'a'*64, 'schema': 'goal-plan'}))
    with (p / 'job.env').open('a') as f:
        f.write(f'ISPATIALGYM_CAMPAIGN_CONFIG={config}\nISPATIALGYM_CAMPAIGN_GOAL_PLAN={plan}\n')
    value = scope(j)
    assert value['status'] == 'observed'
    assert value['group_ids'] == ['direct_standard_group']
    assert value['group_binding'] == 'exact_goal_plan_group'
    assert {'path': str(plan), 'sha256': hashlib.sha256(plan.read_bytes()).hexdigest()} in value['evidence']
    # A conflicting identity must never retain an apparently usable group join.
    body = json.loads(plan.read_text()); body['contract_template_sha256'] = 'b'*64
    plan.write_text(json.dumps(body))
    rejected = scope(j)
    assert rejected['status'] == 'incomplete' and rejected['group_ids'] == []
    assert 'template differs' in rejected['errors'][0]
