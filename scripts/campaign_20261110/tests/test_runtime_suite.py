import json
import pytest
from scripts.campaign_20261110 import runtime_suite as s, production_gate as g
from scripts.campaign_20261110.contracts import file_digest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path':str(path), 'sha256':file_digest(path)}


def fixture(root):
    base = root/'runtime_validation/run'
    native = root/'code/frozen'; native.mkdir(parents=True)
    (native/'.snapshot_commit').write_text('abc\n')
    runtime = {'path':str(native),'snapshot_sha256':file_digest(native/'.snapshot_commit'),'commit':'abc'}
    release = {'schema':'isg-current-runtime-release-v1','campaign_id':'campaign_20261110',
               'run_id':'run','run_started_at':1,'required_classes':100,'frozen_runtime':runtime}
    rr = put(base/'release.json', release)
    pointer = {'schema':'isg-current-runtime-suite-pointer-v1','release':rr,
        'manifest':put(base/'manifest.json', {'classes':[{'class_id':'class'+str(i)} for i in range(100)]}),
        'index_reservation':put(base/'index_reservation.json', {'proof':{'start':1000,'reserved_count':300}})}
    put(root/'runtime_validation/current.json',pointer)
    cfg={'purpose':'validation','template':{'split':'eval_standard','task':'object_scaling','class_id':'class0'},
         'validation_key':{'code_sha256':runtime['snapshot_sha256']}}
    cr=put(base/'config.json',cfg); goal=put(base/'goal.json',{'group_id':'suite0'})
    row={'run_id':'run','run_started_at':1,'release':rr,'frozen_runtime':runtime,'campaign_config':cr,
         'class_id':'class0','fresh_start':1000,'fresh_count':3,'group_id':'suite0','goal_plan':goal,'camera_exempt':False}
    rf=put(base/'rows/class0.json',row)
    env={'ISPATIALGYM_CAMPAIGN_CONFIG':cr['path'],'ISPATIALGYM_CAMPAIGN_GOAL_PLAN':goal['path'],
         'ISPATIALGYM_RUNTIME_SUITE_RUN':'run','ISPATIALGYM_RUNTIME_SUITE_ROW':rf['path'],
         'ISPATIALGYM_RUNTIME_SUITE_ROW_SHA256':rf['sha256'],'ISGE_CODE':str(native)}
    return base, env, cfg


def test_full_quota_suite_allowed_only_exact_scope(tmp_path,monkeypatch):
    base,env,cfg=fixture(tmp_path)
    monkeypatch.setattr(g,'evaluation_exhausted',lambda *args:True)
    result=g.check_environment(tmp_path,env,start=1000,count=3)
    assert result['allowed'] and result['category']=='runtime_suite_validation'
    assert result['runtime_suite']['production_authorization'] is False
    for key,value in [('ISGE_CODE',str(tmp_path/'code/other')),('ISPATIALGYM_RUNTIME_SUITE_RUN','other')]:
        assert not g.check_environment(tmp_path,{**env,key:value},start=1000,count=3)['allowed']
    assert not g.check_environment(tmp_path,env,start=1001,count=3)['allowed']
    assert not g.check_environment(tmp_path,env,start=1000,count=4)['allowed']
    assert g.check_environment(tmp_path,env,start=1000,count=3,kind='source_validation')['allowed']
    assert not g.check_environment(tmp_path,{k:v for k,v in env.items() if not k.startswith('ISPATIALGYM_RUNTIME_SUITE_')},start=1000,count=3)['allowed']


def test_changed_bytes_and_train_purpose_denied(tmp_path):
    base,env,cfg=fixture(tmp_path)
    cfg['template']['split']='train';put(base/'config.json',cfg)
    assert not g.check_environment(tmp_path,env,start=1000,count=3)['allowed']
    cfg['template']['split']='eval_standard';cfg['purpose']='training_production';put(base/'config.json',cfg)
    assert not g.check_environment(tmp_path,env,start=1000,count=3)['allowed']


def test_wrong_class_reserved_slot_and_missing_current100_fail_closed(tmp_path):
    base,env,cfg=fixture(tmp_path)
    path=base/'rows/class0.json';row=json.loads(path.read_text());row['fresh_start']=1003
    env['ISPATIALGYM_RUNTIME_SUITE_ROW_SHA256']=put(path,row)['sha256']
    assert not g.check_environment(tmp_path,env,start=1003,count=3)['allowed']
    assert not s.current_training_ready(tmp_path)


def test_dispatch_runtime_and_native_job_same_binding(tmp_path):
    base,env,cfg=fixture(tmp_path)
    command=['python','submit_fast.py','submit','class0','3','--base-index','1000','--code',env['ISGE_CODE']]
    for name,value in env.items():
        if name!='ISGE_CODE':command+=['--extra-env',name+'='+value]
    assert g.check_dispatch(tmp_path,command)['allowed']
    command[command.index('--code')+1]=str(tmp_path/'code/other')
    assert not g.check_dispatch(tmp_path,command)['allowed']
    job=tmp_path/'out/job';job.mkdir(parents=True)
    (job/'job.env').write_text('\n'.join(k+'='+v for k,v in {**env,'START':'1000','N':'3'}.items()))
    assert g.check_job(tmp_path,job)['allowed']
