from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from scripts.campaign_20261110.contracts import ContractError,file_digest
from ispatialgym.oracle.campaign_batch import record_scope,compare_saved_scenes,restore_parking,SCHEMA


def scene():
    names=('chair','table','target','robot_r1')
    return {'objects_info':{'init_info':{n:{'args':{'category':'chair' if n=='chair' else n,'model':n,'fixed_base':True}}for n in names}},
            'state':{'registry':{'object_registry':{n:{'root_link':{'pos':[0.,0.,0.],'ori':[0.,0.,0.,1.],'lin_vel':[0.,0.,0.]},'joint':[0.]}for n in names}}},
            'metadata':{'ispatialgym':{'moved_objects':{},'removed_objects':[]}}}


def records(tmp_path):
    p=tmp_path/'stable.json';p.write_text(json.dumps(scene()));ref={'path':str(p),'sha256':file_digest(p)}
    row={'instance_id':'a','task':'test','class_id':'test','room':'scene/room','variant':'variant','sampling_split':'train',
         'template':{'template_id':'a','scene':'scene','objects':[{'model':'same','scale':[1,1,1]}],
                     'scene_setup':{'training_height_dependency':{'stable_scene':ref},'inst_to_name':{'target':'target'},'training_scale_plan_sha256':'a'}},
         'validation_key':{'template_sha256':'a','code_sha256':'runtime','data_sha256':'data','physics_sha256':'physics'},
         'perturbations':{'scales':{'target':[1,1,1]},'removed_objects':['chair'],
                         'lighting':{'schema':'isg-lighting-v1','intensity_factor':.8,'skybox_color_factors':[1,1,1]}},
         'artifacts':{'tro':{'path':'a','sha256':'a'}}}
    b=deepcopy(row);b['instance_id']='b';b['template']['template_id']='b';b['validation_key']['template_sha256']='b'
    b['perturbations']['removed_objects']=[];b['perturbations']['lighting']['intensity_factor']=1.2
    return [row,b]


def test_explicit_changed_light_and_fixed_actor_scope(tmp_path):
    rows=records(tmp_path);proof=record_scope(rows)
    assert proof['parking_objects']==['chair'] and proof['slots']==1
    assert rows[0]['template']['scene_setup']['training_scale_plan_sha256']=='a'


@pytest.mark.parametrize('mutation',[
    lambda r:r[1]['template']['objects'][0].update(model='other'),
    lambda r:r[1]['template'].update(scene='other'),
    lambda r:r[1].update(room='other'),
    lambda r:r[1]['validation_key'].update(code_sha256='other'),
    lambda r:r[1]['perturbations']['scales'].update(target=[2,2,2]),
    lambda r:r[1]['perturbations'].update(lighting=deepcopy(r[0]['perturbations']['lighting'])),
    lambda r:r[1]['perturbations'].update(removed_objects=['chair']),
    lambda r:r[0]['perturbations'].update(removed_objects=['target']),
])
def test_scope_rejects_non_applicable_changes(tmp_path,mutation):
    rows=records(tmp_path);mutation(rows)
    with pytest.raises(ContractError):record_scope(rows)


def test_parking_requires_identical_fixed_actor_type(tmp_path):
    rows=records(tmp_path);ref=rows[0]['template']['scene_setup']['training_height_dependency']['stable_scene']
    p=tmp_path/'stable.json';value=json.loads(p.read_text());value['objects_info']['init_info']['chair']['args']['fixed_base']=False
    p.write_text(json.dumps(value));ref['sha256']=file_digest(p);rows[1]['template']['scene_setup']['training_height_dependency']['stable_scene']=deepcopy(ref)
    with pytest.raises(ContractError,match='fixed-base'):record_scope(rows)


def test_scene_compare_only_omits_declared_parking_state():
    a=scene();b=deepcopy(a);b['state']['registry']['object_registry']['chair']['root_link']['pos']=[-200,0,0]
    assert compare_saved_scenes(a,b,parking_objects=['chair'],task_objects={'target'})['maximum_numeric_difference']==0
    b['objects_info']['init_info']['chair']['args']['model']='different'
    with pytest.raises(ContractError,match='identity'):compare_saved_scenes(a,b,parking_objects=['chair'],task_objects={'target'})
    b=deepcopy(a);b['state']['registry']['object_registry']['table']['root_link']['pos']=[.006,0,0]
    with pytest.raises(ContractError,match='static scene'):compare_saved_scenes(a,b,parking_objects=['chair'],task_objects={'target'})
    b=deepcopy(a);b['state']['registry']['object_registry']['chair']['joint']=[.1]
    with pytest.raises(ContractError,match='static scene'):compare_saved_scenes(a,b,parking_objects=['chair'],task_objects={'target'})


def test_native_parking_restore_is_hash_bound_and_adds_no_step(tmp_path):
    tro=tmp_path/'tro.json';tro.write_text('{}');saved=tmp_path/'scene.json';saved.write_text('{}');state={'root_link':{'pos':[1,2,3]}}
    calls=[];obj=SimpleNamespace(fixed_base=True,load_state=lambda value,serialized:calls.append((value,serialized)))
    loader=SimpleNamespace(tro_path=str(tro),env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda *args:obj)))
    contract={'schema':SCHEMA,'members_by_tro':{str(tro):{'tro':{'path':str(tro),'sha256':file_digest(tro)},'scene':{'path':str(saved),'sha256':file_digest(saved)},'parking_states':{'chair':state}}}}
    assert restore_parking(loader,contract)['physics_steps_added']==0
    assert calls==[(state,False)] and calls[0][0] is not state
    tro.write_text('{"changed":true}')
    with pytest.raises(ContractError,match='input changed'):restore_parking(loader,contract)


def test_multislot_different_lighting_still_rejected(tmp_path):
    from ispatialgym.multienv.campaign_rendering import selected_records
    rows=records(tmp_path)
    for i,r in enumerate(rows):r['artifacts']['demo']={'path':str(tmp_path/str(i)/'demo.hdf5')};r['render_compatibility_sha256']=str(i)
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n'for r in rows));dirs=[str(tmp_path/str(i))for i in range(2)]
    with pytest.raises(ContractError,match='configuration'):selected_records(manifest,dirs)
    with pytest.raises(ContractError,match='one renderer slot'):selected_records(manifest,dirs,sequential_augmentation=True,slots=2)
    assert len(selected_records(manifest,dirs,sequential_augmentation=True,slots=1))==2


def test_constructor_count_preserves_results_and_restores_after_failure():
    from scripts.campaign_20261110.training_batch_probe import world_construction_counter
    calls=[];result=object()
    def construct(*a,**kw):
        calls.append((a,kw))
        if kw.get('fail'):raise RuntimeError('original')
        return result
    og=SimpleNamespace(Environment=construct)
    with pytest.raises(RuntimeError,match='original'):
        with world_construction_counter(og) as proof:
            assert og.Environment('exact',n=2) is result
            og.Environment(fail=True)
    assert proof['attempted']==2 and proof['completed']==1 and og.Environment is construct
    assert calls==[(('exact',),{'n':2}),((),{'fail':True})]


@pytest.mark.parametrize('name,exception',[('A_FAILURE_A','ControlledEpisodeFailure'),('A_INTERRUPT_A','ControlledEpisodeInterrupt')])
def test_failed_and_interrupted_controls_retain_native_call(name,exception):
    import scripts.campaign_20261110.training_batch_probe as probe
    cls=getattr(probe,exception);calls=[];env=SimpleNamespace(step=lambda *a,**kw:calls.append((a,kw)))
    original=env.step;proof={};cleanup=probe.abort_after_native_control_step(env,proof,cls)
    with pytest.raises(cls):env.step('native',unchanged=True)
    cleanup()
    assert env.step is original and calls==[(('native',),{'unchanged':True})]
    assert proof['exception_type']==exception and proof['physics_steps_added']==0
    assert probe.episode_order({'diagnostic_control':name,'diagnostic_only':True,'production_released':False},'physics_action_open_loop')==[0,0]
