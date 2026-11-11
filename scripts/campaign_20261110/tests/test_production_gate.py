import json
from pathlib import Path
from scripts.campaign_20261110.production_gate import check_dispatch, check_job, check_environment
from scripts.campaign_20261110.training_eligibility import classify_record, eligible_training_records
from scripts.campaign_20261110.contracts import digest


def setup(tmp_path, split):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'template': {'split':split,'task':'object_scaling','class_id':'apple'}}))
    env = {'ISPATIALGYM_CAMPAIGN_CONFIG':str(config), 'START':'50000','N':'3'}
    cmd = ['python','submit_fast.py','submit','apple','3','--base-index','50000','--extra-env','ISPATIALGYM_CAMPAIGN_CONFIG='+str(config)]
    job = tmp_path/'job'; job.mkdir()
    (job/'job.env').write_text('\n'.join(k+'='+v for k,v in env.items()))
    return env,cmd,job


def test_missing_policy_blocks_standalone_and_unknown_but_keeps_eval(tmp_path):
    env,cmd,job=setup(tmp_path,'train')
    assert not check_dispatch(tmp_path,cmd)['allowed']
    assert not check_job(tmp_path,job)['allowed']
    assert not check_environment(tmp_path,{})['allowed']
    raw=json.loads((tmp_path/'config.json').read_text());raw['template']['split']='eval_standard'
    (tmp_path/'config.json').write_text(json.dumps(raw))
    assert check_dispatch(tmp_path,cmd)['allowed']
    assert check_job(tmp_path,job)['allowed']


def test_replay_and_prerequisites_allowed_without_relabeling_collection(tmp_path):
    assert not check_job(tmp_path,tmp_path/'missing',kind='physics_replay')['allowed']
    assert not check_job(tmp_path,tmp_path/'missing',kind='native_dependency')['allowed']
    assert not check_job(tmp_path,tmp_path/'missing',kind='validation')['allowed']


def test_open_flag_and_validation_label_cannot_bypass(tmp_path):
    env,cmd,job=setup(tmp_path,'train')
    p=tmp_path/'controllers';p.mkdir()
    (p/'training_production_gate.json').write_text(json.dumps({'schema':'isg-training-production-gate-v1','campaign_id':'campaign_20261110','mode':'open','standalone_training_allowed':True}))
    assert not check_dispatch(tmp_path,cmd)['allowed']
    raw=json.loads((tmp_path/'config.json').read_text());raw['purpose']='validation'
    (tmp_path/'config.json').write_text(json.dumps(raw))
    assert not check_job(tmp_path,job)['allowed']


def member(i,split='train',source='eval_standard'):
    return {'instance_id':i,'group_id':'g','task':'object_scaling','class_id':'apple','split':split,'sampling_split':source}


def test_siblings_singleton_and_overflow_distinguished_without_mutation():
    t=member('t');e=member('e','eval_standard');g={'accepted':{'t':'train','e':'eval_standard'},'members':[t,e]}
    original=digest(g)
    assert classify_record(t,g)['category']=='eval_sibling'
    g1={'accepted':{'t':'train'},'members':[t]}
    assert classify_record(t,g1)['category']=='eval_singleton'
    t2=member('t2');g2={'accepted':{'t':'train','t2':'train'},'members':[t,t2]}
    assert classify_record(t,g2)['status']=='excluded'
    assert classify_record(member('x',source='train'),None)['category']=='standalone_training'
    assert classify_record(t,None)['status']=='unverified'
    assert digest(g)==original


def test_flagged_rows_do_not_consume_replacement_capacity():
    import sqlite3
    db=sqlite3.connect(':memory:');db.executescript('CREATE TABLE members(split,task,record);CREATE TABLE groups(id,record);')
    good=member('g');bad=member('b',source='train')
    group={'accepted':{'g':'train'},'members':[good]}
    db.execute('INSERT INTO groups VALUES (?,?)',('g',json.dumps(group)))
    for row in (good,bad):db.execute('INSERT INTO members VALUES (?,?,?)',('train','object_scaling',json.dumps(row)))
    assert eligible_training_records(db,'object_scaling')==[good]
    assert db.execute('SELECT COUNT(*) FROM members').fetchone()[0]==2


def test_camera_overflow_not_a_goal_group_singleton():
    row=member('c');row['task']='camera_relocalization-standard'
    assert classify_record(row,{'accepted':{'c':'train'},'members':[row]})['status']=='excluded'


def test_eval_label_cannot_bypass_exhausted_goal_quota(tmp_path):
    import sqlite3
    env,cmd,job=setup(tmp_path,'eval_standard')
    env['ISPATIALGYM_CAMPAIGN_GOAL_PLAN']='owned-goal'
    p=tmp_path/'dataset';p.mkdir()
    with sqlite3.connect(p/'allocations.sqlite') as db:
        db.execute('CREATE TABLE members(split,task,class_id)')
        db.executemany('INSERT INTO members VALUES (?,?,?)',[('eval_standard','object_scaling','apple')]*10)
    assert not check_environment(tmp_path,env)['allowed']


def test_exact_canary_authorization_is_bounded_and_expires(tmp_path,monkeypatch):
    monkeypatch.setattr('scripts.campaign_20261110.production_gate.canonical_training_ready',lambda root:False)
    from datetime import datetime,timezone,timedelta
    from scripts.campaign_20261110.contracts import file_digest
    env,cmd,job=setup(tmp_path,'train')
    proof=tmp_path/'authorization.json'
    value={'purpose':'training_pipeline_canary','config_sha256':file_digest(tmp_path/'config.json'),'start':50000,'count':3}
    evidence=tmp_path/'heldout_source.json';evidence.write_text('{}')
    work=tmp_path/'heldout_work.json'
    related={'owner':'heldout_worker','task':'object_scaling','class_id':'apple','state':'engineering_blocked',
             'blocker':'Exact model map missing','next_action':'Build the exact ready-model map',
             'evidence':[{'path':str(evidence),'sha256':file_digest(evidence)}]}
    work.write_text(json.dumps({'campaign_id':'campaign_20261110','training_config_sha256':value['config_sha256'],
        'training_class_id':'apple','task':'object_scaling','observed_at':datetime.now(timezone.utc).isoformat(),
        'model':related,'category_room':related}))
    value['heldout_parallel_work']={'path':str(work),'sha256':file_digest(work)}
    proof.write_text(json.dumps(value))
    auth={**value,'max_instances':3,'expires_at':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
          'authorization_path':str(proof),'authorization_sha256':file_digest(proof)}
    p=tmp_path/'controllers';p.mkdir()
    gate=p/'training_production_gate.json'
    policy={'schema':'isg-training-production-gate-v1','campaign_id':'campaign_20261110','mode':'blocked','allowed_canaries':[auth]}
    gate.write_text(json.dumps(policy))
    assert check_dispatch(tmp_path,cmd)['category']=='validation_canary'
    assert check_job(tmp_path,job)['allowed']
    changed=list(cmd);changed[4]='4';assert not check_dispatch(tmp_path,changed)['allowed']
    auth['expires_at']='2000-01-01T00:00:00+00:00';gate.write_text(json.dumps(policy))
    assert not check_dispatch(tmp_path,cmd)['allowed']


def test_heldout_parallel_work_requires_both_scopes_and_exact_identity(tmp_path):
    from datetime import datetime,timezone
    import pytest
    from scripts.campaign_20261110.production_gate import heldout_parallel_ready
    from scripts.campaign_20261110.contracts import file_digest
    cfg={'template':{'task':'object_scaling','class_id':'apple'}}
    evidence=tmp_path/'state.json';evidence.write_text('{}')
    row={'owner':'named_worker','task':'object_scaling','class_id':'apple','state':'running','active_jobs':['exact_job'],
         'evidence':[{'path':str(evidence),'sha256':file_digest(evidence)}]}
    value={'campaign_id':'campaign_20261110','training_config_sha256':'config_hash','training_class_id':'apple',
           'task':'object_scaling','observed_at':datetime.now(timezone.utc).isoformat(),'model':row,'category_room':row}
    path=tmp_path/'parallel.json';path.write_text(json.dumps(value))
    auth={'heldout_parallel_work':{'path':str(path),'sha256':file_digest(path)}}
    assert heldout_parallel_ready(tmp_path,cfg,'config_hash',auth)==auth['heldout_parallel_work']
    with pytest.raises(ValueError,match='exact training config'):heldout_parallel_ready(tmp_path,cfg,'changed',auth)
    value['category_room']={**row,'state':'engineering_blocked','blocker':'missing map'};path.write_text(json.dumps(value))
    auth['heldout_parallel_work']['sha256']=file_digest(path)
    with pytest.raises(ValueError,match='actionable'):heldout_parallel_ready(tmp_path,cfg,'config_hash',auth)


def test_projection_unverified_overrides_mechanically_proven_sibling_quota(tmp_path):
    import sqlite3
    db=sqlite3.connect(':memory:');db.executescript('CREATE TABLE members(split,task,record);CREATE TABLE groups(id,record);')
    row=member('g');group={'accepted':{'g':'train'},'members':[row]}
    db.execute('INSERT INTO groups VALUES (?,?)',('g',json.dumps(group)))
    db.execute('INSERT INTO members VALUES (?,?,?)',('train','object_scaling',json.dumps(row)))
    p=tmp_path/'accounting';p.mkdir()
    (p/'training_eligibility.json').write_text(json.dumps({'campaign_id':'campaign_20261110','status':'complete',
        'records':{'g':{'status':'unverified','accepted_record_sha256':digest(row)}}}))
    assert eligible_training_records(db,root=tmp_path)==[]


def test_projection_reaudits_native_demo_and_replay_without_rewriting_acceptance(tmp_path):
    import shutil
    from scripts.campaign_20261110.tests.test_group_lane import candidate, claim
    from scripts.campaign_20261110.training_eligibility import build_projection
    f,review=candidate(tmp_path,0)
    result=claim(tmp_path,f,review)
    source=tmp_path/'dataset/quotas.sqlite3';target=tmp_path/'dataset/allocations.sqlite'
    shutil.copyfile(source,target)
    before=target.read_bytes()
    projection=build_projection(tmp_path)
    assert projection['historically_accepted']==projection['eligible_accepted']==1
    assert projection['unverified']==projection['excluded']==0
    catalog=tmp_path/'manifests/dashboard_standard_catalog.json';catalog.parent.mkdir(exist_ok=True)
    catalog.write_text(json.dumps({'schema':'isg-dashboard-standard-catalog-v1',
        'templates':[{'class_id':'unattempted_class','task':'object_scaling'}]}))
    full=build_projection(tmp_path)
    assert full['coverage']['by_class']['unattempted_class']['eligible_instances']==0
    assert full['coverage']['by_class']['unattempted_class']['supports']['distinct']==0
    assert full['coverage']['standard_catalog']['classes']==1
    Path(review['members'][1]['report']).write_text('{}')
    after=build_projection(tmp_path)
    assert after['eligible_accepted']==0 and after['unverified']==1
    assert target.read_bytes()==before


def test_replay_checks_true_source_scope_but_not_exhausted_eval_quota(tmp_path):
    env,cmd,job=setup(tmp_path,'train')
    replay=tmp_path/'replay';replay.mkdir()
    (replay/'replay_inputs.json').write_text(json.dumps({'source_job':str(job)}))
    assert not check_job(tmp_path,replay,kind='physics_replay')['allowed']
    config=tmp_path/'config.json';value=json.loads(config.read_text());value['template']['split']='eval_standard';config.write_text(json.dumps(value))
    assert check_job(tmp_path,replay,kind='physics_replay')['allowed']
    value['template']['class_id']='inter_object_relation_satisfaction-under_large_scale_01';config.write_text(json.dumps(value))
    assert not check_job(tmp_path,replay,kind='physics_replay')['allowed']


def test_native_dependency_cannot_resume_owner_parked_class(tmp_path):
    job=tmp_path/'native';job.mkdir();p=job/'native_inputs.json'
    value={'campaign_root':str(tmp_path),'original_entry':{'class_id':'inter_object_relation_satisfaction-under_large_scale_01'}}
    p.write_text(json.dumps(value));assert not check_job(tmp_path,job,kind='native_dependency')['allowed']
    value['original_entry']['class_id']='inter_object_relation_satisfaction-inside_large_scale_01'
    p.write_text(json.dumps(value));assert check_job(tmp_path,job,kind='native_dependency')['allowed']


def test_available_owner_updates_resume_validation_without_releasing_training(tmp_path):
    from scripts.campaign_20261110.production_gate import PARKED_CLASSES
    assert PARKED_CLASSES == {'inter_object_relation_satisfaction-under_large_scale_01'}
    env,cmd,job=setup(tmp_path,'eval_standard')
    native=tmp_path/'native';native.mkdir()
    replay=tmp_path/'replay';replay.mkdir()
    (replay/'replay_inputs.json').write_text(json.dumps({'source_job':str(job)}))
    config=tmp_path/'config.json'
    value=json.loads(config.read_text())
    for suffix in ('touching_03','under_large_scale_02','under_large_scale_03','nextto_04'):
        cls='inter_object_relation_satisfaction-'+suffix
        value['template'].update(class_id=cls,split='eval_standard')
        config.write_text(json.dumps(value))
        (native/'native_inputs.json').write_text(json.dumps({'campaign_root':str(tmp_path),'original_entry':{'class_id':cls}}))
        assert check_dispatch(tmp_path,cmd)['allowed']
        assert check_job(tmp_path,replay,kind='physics_replay')['allowed']
        assert check_job(tmp_path,native,kind='native_dependency')['allowed']
        value['template']['split']='train';config.write_text(json.dumps(value))
        assert not check_dispatch(tmp_path,cmd)['allowed']
        assert not check_job(tmp_path,replay,kind='physics_replay')['allowed']


def test_deferred_successful_members_cannot_be_reclassified_as_singleton():
    row=member('t')
    group={'accepted':{'t':'train'},'members':[row],'deferred':['successful_but_over_quota']}
    assert classify_record(row,group)['status']=='excluded'


def test_bulk_eval_requires_exact_successful_group_proof_and_bounded_group_remains_allowed(tmp_path):
    env,cmd,job=setup(tmp_path,'eval_standard')
    raw=json.loads((tmp_path/'config.json').read_text());raw['validation_key']={'code':'new-runtime'}
    (tmp_path/'config.json').write_text(json.dumps(raw))
    assert check_dispatch(tmp_path,cmd)['allowed']
    cmd[4]='18'
    assert not check_dispatch(tmp_path,cmd)['allowed']


def test_exact_group_proof_requires_real_sibling_and_preserves_runtime_identity(tmp_path):
    import shutil
    from scripts.campaign_20261110.tests.test_group_lane import candidate,claim
    from scripts.campaign_20261110.training_eligibility import standard_group_proof
    f,review=candidate(tmp_path,0);claim(tmp_path,f,review)
    shutil.copyfile(tmp_path/'dataset/quotas.sqlite3',tmp_path/'dataset/allocations.sqlite')
    assert standard_group_proof(tmp_path,f.template.class_id,review['validation_key'])['group_id']==review['group_id']
    assert standard_group_proof(tmp_path,f.template.class_id,{**review['validation_key'],'code_sha256':'a'*64}) is None


def test_native_coverage_separates_original_removal_and_actual_augmentation():
    from scripts.campaign_20261110.training_eligibility import coverage_facts, coverage_summary
    template={'objects':[{'instance':'table','category':'table','model':'t','role':'support','semantic_kind':'object'},
                         {'instance':'ball','category':'ball','model':'b','role':'target','semantic_kind':'object'}]}
    row={'room':'house/room','validation_key':{'template_sha256':digest(template)}}
    raw={'scene':{'model':'house','scene_variant':{'removed_objects':['chair']}},
         'object_bindings':{'table':{'category':'table','model':'t','scene_object_name':'table_1'},
                            'ball':{'category':'ball','model':'b','scene_object_name':'ball_1'}},
         'campaign':{'perturbations':{'scales':{'table':{'default':[1,1,1]},'ball':{'default':[1,1,1]}},'removed_objects':[],'lighting':{}},
                     'runtime':{'observed_scales':{'table':[1,1,1],'ball':[1.2,1.2,1.2]},'augmentation':{'moved_objects':{},'lighting':{}}}}}
    fact=coverage_facts(row,raw,template);coverage=coverage_summary([fact])
    assert coverage['supports']['distinct']==1 and coverage['models']['distinct']==2
    assert coverage['measures']['instances_with_original_scene_removal']==1
    assert coverage['measures']['instances_with_verified_augmentation_removal']==0
    assert coverage['measures']['instances_with_nondefault_scale']==1
    assert coverage['measures']['instances_with_lighting_augmentation']==0
    assert coverage['coverage_evidence_complete']
    unknown=coverage_summary([coverage_facts(row,raw)])
    assert unknown['missing_evidence']['supports']==1 and not unknown['coverage_evidence_complete']
    import pytest
    with pytest.raises(ValueError,match='contract differs'):
        coverage_facts(row,raw,{'objects':[]})


def test_old_canary_without_parallel_evidence_cannot_bypass_bulk_hold(tmp_path):
    from datetime import datetime,timezone,timedelta
    from scripts.campaign_20261110.contracts import file_digest
    env,cmd,job=setup(tmp_path,'train');p=tmp_path/'controllers';p.mkdir()
    auth=tmp_path/'authorization.json';value={'purpose':'training_pipeline_canary','config_sha256':file_digest(tmp_path/'config.json'),'start':50000,'count':3}
    auth.write_text(json.dumps(value))
    entry={**value,'max_instances':3,'expires_at':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),'authorization_path':str(auth),'authorization_sha256':file_digest(auth)}
    (p/'training_production_gate.json').write_text(json.dumps({'schema':'isg-training-production-gate-v1','campaign_id':'campaign_20261110','mode':'open','standalone_training_allowed':True,'allowed_canaries':[entry]}))
    result=check_job(tmp_path,job)
    assert not result['allowed'] and 'heldout_parallel_work' in result['reason']
    report=tmp_path/'infra/standard_recovery/canonical_validation.json';report.parent.mkdir(parents=True)
    report.write_text(json.dumps({'schema':'isg-canonical-template-validation-v1','total':100,'validated':99,'classes':[]}))
    assert not check_dispatch(tmp_path,cmd)['allowed']
    report.write_text(json.dumps({'schema':'isg-canonical-template-validation-v1','total':100,'validated':100,'classes':[]}))
    assert not check_dispatch(tmp_path,cmd)['allowed']
