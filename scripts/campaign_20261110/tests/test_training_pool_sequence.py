import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.campaign_20261110 import training_pool_sequence as sequence


def ready_sequence(tmp_path,monkeypatch):
    request=tmp_path/'requests.json';request.write_text(json.dumps({'sequence':['A','B','A'],'records':[{}, {}, {}]}))
    prepared=[]
    for i,model in enumerate(('a','b','a')):
        config=tmp_path/f'{i}.json';config.write_text(json.dumps({'template':{'objects':[{'role':'target','category':'bottle','model':model,'default_scale':[1,1,1]}]}}))
        prepared.append(({'start':100+i,'native_runtime':'runtime','config_path':str(config)},
            {'dependency_release':'both'}, {'ISPATIALGYM_CAMPAIGN_CONFIG':str(config),'ISPATIALGYM_CAMPAIGN_MODEL_POOL_MANIFEST':'pool'},None,None,None))
    monkeypatch.setattr(sequence,'prepare',lambda root,path,i:prepared[i])
    return request,prepared


def test_sequence_requires_distinct_indices_one_runtime_and_one_dependency_identity(tmp_path,monkeypatch):
    request,prepared=ready_sequence(tmp_path,monkeypatch)
    sequence.sequence_inputs(tmp_path,request,sequence.file_digest(request))
    for index,field,value in [(0,'start',101),(0,'native_runtime','different')]:
        old=prepared[index][0][field];prepared[index][0][field]=value
        with pytest.raises(ValueError):sequence.sequence_inputs(tmp_path,request,sequence.file_digest(request))
        prepared[index][0][field]=old
    prepared[1][1]['dependency_release']='different'
    with pytest.raises(ValueError,match='shared exact dependency'):sequence.sequence_inputs(tmp_path,request,sequence.file_digest(request))


def test_sequence_rejects_environment_and_requested_target_mismatch(tmp_path,monkeypatch):
    request,prepared=ready_sequence(tmp_path,monkeypatch)
    prepared[2][2]['OTHER_NATIVE_FLAG']='changed'
    with pytest.raises(ValueError,match='environments'):sequence.sequence_inputs(tmp_path,request,sequence.file_digest(request))
    prepared[2][2].pop('OTHER_NATIVE_FLAG');cfg=Path(prepared[2][0]['config_path'])
    value=json.loads(cfg.read_text());value['template']['objects'][0]['model']='c';cfg.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='target identity'):sequence.sequence_inputs(tmp_path,request,sequence.file_digest(request))


def stage(tmp_path):
    active={'category':'bottle','model':'a','visible':True,'planner_included':True,'collision_enabled':[True]}
    inactive={**active,'model':'b','visible':False,'planner_included':False,'collision_enabled':[False]}
    event={'event':'exported','active':{'a':'target'},'native_observation':{'pool_physics_rebuilds':1,'objects':{'a':active,'b':inactive}}}
    report=tmp_path/'pool.json';report.write_text('{}')
    mp=SimpleNamespace(members={'a':object(),'b':object()},active={},events=[event,{'event':'released'}],output=report)
    pool=SimpleNamespace(n_loads=1,world=SimpleNamespace(model_pool=mp,spawned={}),kit=object(),model_pool_key='pinned')
    scene={'metadata':{'ispatialgym':{'model_pool':{'active_names':['a'],'manifest_sha256':'pinned'}}},'objects_info':{'init_info':{'a':{},'floor':{}}}}
    (tmp_path/'template-partial_rooms.json').write_text(json.dumps(scene))
    return pool,scene


def test_stage_demands_same_world_and_active_only_export(tmp_path):
    pool,scene=stage(tmp_path);target={'category':'bottle','model':'a'}
    identity,proof=sequence.verify_stage(pool,None,0,target,tmp_path)
    assert proof['ordinary_cleanup_complete'] and proof['inactive_export_omissions']==1
    pool.world.model_pool.members['a']=object()
    with pytest.raises(ValueError,match='identity changed'):sequence.verify_stage(pool,identity,0,target,tmp_path)
    scene['objects_info']['init_info']['b']={};(tmp_path/'template-partial_rooms.json').write_text(json.dumps(scene))
    with pytest.raises(ValueError,match='leaked'):sequence.verify_stage(pool,None,0,target,tmp_path)


@pytest.mark.parametrize('field,value',[('visible',True),('planner_included',True),('collision_enabled',[True])])
def test_stage_rejects_inactive_model_participation(tmp_path,field,value):
    pool,_=stage(tmp_path);pool.world.model_pool.events[0]['native_observation']['objects']['b'][field]=value
    with pytest.raises(ValueError,match='inactive model'):sequence.verify_stage(pool,None,0,{'category':'bottle','model':'a'},tmp_path)
