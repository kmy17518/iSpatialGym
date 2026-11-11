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
    assert check_job(tmp_path,tmp_path/'missing',kind='physics_replay')['allowed']
    assert check_job(tmp_path,tmp_path/'missing',kind='native_dependency')['allowed']
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


def test_exact_canary_authorization_is_bounded_and_expires(tmp_path):
    from datetime import datetime,timezone,timedelta
    from scripts.campaign_20261110.contracts import file_digest
    env,cmd,job=setup(tmp_path,'train')
    proof=tmp_path/'authorization.json'
    value={'purpose':'training_pipeline_canary','config_sha256':file_digest(tmp_path/'config.json'),'start':50000,'count':3}
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
