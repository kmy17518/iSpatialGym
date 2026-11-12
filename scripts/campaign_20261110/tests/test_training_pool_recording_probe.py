import copy
from types import SimpleNamespace as N
import pytest
import torch
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_pool_recording_probe import (
    parent_request, active_document, recorded_config, verify_serialized, serialized_object_names)


def metadata(active):
    return {'task': {'task': {}, 'inst_to_name': {'target': active}, 'robot_poses': {'robot': [active]}},
            'ispatialgym': {'model_pool': {'manifest_sha256': 'exact', 'batch_id': 'batch',
                'active_names': [active], 'inactive_omitted': 1, 'active_properties': {active: {'model': active}}}}}


def test_native_metadata_wrapper_is_explicitly_verified_before_current_source_projection():
    a, b = metadata('A'), metadata('B')
    native = {'metadata': {'task': {**a, 'inst_to_name': {'target': 'B'}, 'robot_poses': {'robot': ['currentB']}}},
              'objects_info': {'init_info': {'A': {}, 'B': {}, 'background': {}}},
              'state': {'registry': {'object_registry': {'A': {}, 'B': {}, 'background': {}}}}}
    source = {'metadata': b}
    args = [source, [a,b], {'A','B'}, 'B', {'model':'B'}, {'target':'B'}, {'robot':['currentB']}]
    result = active_document(native, *args)
    assert set(result['objects_info']['init_info']) == {'B','background'}
    assert result['metadata']['task']['robot_poses'] == {'robot':['currentB']}
    assert result['metadata']['ispatialgym']['model_pool']['active_names'] == ['B']
    assert native['metadata']['task']['ispatialgym']['model_pool']['active_names'] == ['A']
    unknown = copy.deepcopy(native); unknown['metadata']['task']['unknown'] = 'must retain or reject'
    with pytest.raises(ContractError, match='known source'): active_document(unknown, *args)
    inactive_ref = copy.deepcopy(native); inactive_ref['objects_info']['unknown_reference'] = 'A'
    with pytest.raises(ValueError, match='inactive object'): active_document(inactive_ref, *args)
    wrong_properties = list(args); wrong_properties[4] = {'model':'wrong'}
    with pytest.raises(ContractError): active_document(native, *wrong_properties)


def test_recording_config_updates_current_tro_spawn_and_nominal_camera_preserving_native_defaults():
    config = {'env': {'action_frequency':30}, 'scene': {'scene_file':'B.json'},
              'task': {'tro_path':'B-tro.json'},
              'robots':[{'position':[2,3,4], 'sensor_config':{'VisionSensor':{'sensor_kwargs':{'image_width':128,'image_height':128}}}}]}
    defaults = {'env':{'device':None,'num_envs':7}, 'render':{'viewer_width':1280},
                'task':{'tro_path':'A-tro.json'}, 'robots':[{'position':[0,0,0]}]}
    out = recorded_config(config, 'active_B.json', 'B-tro.json', 1080, defaults)
    assert out['env'] == {'device':None,'num_envs':1,'action_frequency':30}
    assert out['task']['tro_path']=='B-tro.json' and out['robots'][0]['position']==[2,3,4]
    assert out['robots'][0]['sensor_config']['VisionSensor']['sensor_kwargs']['image_height']==1080
    assert defaults['task']['tro_path']=='A-tro.json' and config['scene']['scene_file']=='B.json'
    with pytest.raises(ContractError): recorded_config(config, 'active_B.json', 'A-tro.json', None)


@pytest.mark.parametrize('simulator_layout',[False,True])
def test_native_reader_checks_every_id_consumed_and_keeps_initial_background(simulator_layout):
    state={'registry':{'object_registry':{'A':{},'background':{}}}}
    if simulator_layout:state={0:state}
    owner=N(deserialize=lambda flat:(state,len(flat)))
    flat=torch.zeros(4)
    result=verify_serialized(owner,flat,{'A','B'},'A',{'A','B','background'},True)
    assert result['object_names']==['A','background']
    owner.deserialize=lambda flat:({'registry':{'object_registry':{'A':{}}}},len(flat))
    assert verify_serialized(owner,flat,{'A','B'},'A',{'A','B','background'},False)['object_names']==['A']
    with pytest.raises(ContractError,match='omitted'):verify_serialized(owner,flat,{'A','B'},'A',{'A','B','background'},True)
    owner.deserialize=lambda flat:({'registry':{'object_registry':{'B':{}}}},len(flat))
    with pytest.raises(ContractError,match='inactive'):verify_serialized(owner,flat,{'A','B'},'A',{'A','B','background'},False)
    owner.deserialize=lambda flat:({'registry':{'object_registry':{'A':{}}}},len(flat)-1)
    with pytest.raises(ContractError,match='trailing'):verify_serialized(owner,flat,{'A','B'},'A',{'A','B','background'},False)


def test_multiple_scenes_and_recovery_combination_are_rejected():
    with pytest.raises(ContractError,match='exactly one'):
        serialized_object_names({0:{'object_registry':{}},1:{'object_registry':{}}})
    request=dict(pool_active_recording_v1=True,pool_action_replay_v1=True,diagnostic_control='ABA',
                 load_only=False,production_released=False,benchmark_credit=0)
    assert 'pool_active_recording_v1' not in parent_request(request)
    with pytest.raises(ContractError):parent_request({**request,'pool_recovery_v1':True})


def test_installed_recording_scope_keeps_full_internal_snapshots_and_refreshes_export(tmp_path, monkeypatch):
    import json
    from scripts.campaign_20261110.contracts import file_digest
    from scripts.campaign_20261110.training_pool_recording_probe import install
    from ispatialgym.sampling import model_pool
    known = [metadata('A'),metadata('B')]; members=[]
    for active, meta in zip(('A','B'),known):
        scene_path=tmp_path/(active+'.json');scene_path.write_text(json.dumps({'metadata':meta}))
        tro_path=tmp_path/(active+'-tro.json');tro_path.write_text('{}')
        members.append({'active':active,'tro':{'path':str(tro_path),'sha256':file_digest(tro_path)},
                        'scene':{'path':str(scene_path),'sha256':file_digest(scene_path)}})
    initialized={'A':{},'B':{}}; objects={name:N(name=name)for name in ('A','B','background')}
    class Registry:
        def __init__(self):self._dump_filter=lambda obj:True
        def set_dump_filter(self,callback):self._dump_filter=callback
        def get_dict(self,key):assert key=='name';return objects
        def __call__(self,key,name):return objects[name]
    registry=Registry();scene=N(object_registry=registry)
    states=[]
    class Owner:
        def dump_state(self,serialized):
            assert serialized
            states.append({'registry':{'object_registry':{n:{}for n,obj in objects.items()if registry._dump_filter(obj)}}})
            return torch.tensor([len(states)-1.0])
        def deserialize(self,flat):return states[int(flat[0])],len(flat)
    owner=Owner();env=N(scene=scene)
    source_config={'env':{'action_frequency':30},'scene':{'scene_file':'B.json'},'task':{'tro_path':members[1]['tro']['path']},'robots':[{}]}
    class Loader:
        def __init__(self):
            self.env=env;self.task=N(object_scope={'target':objects['B']});self.tro_path=members[1]['tro']['path']
            self.tro=N(robot_poses={'robot':['currentB']});self.external_camera=False;self.scene_crop=None
            self.template_dir='template';self.tmeta={};self.activity='test';self.did=1;self.robot_name='robot';self.image_size=128
        def restore(self):return 'native restored'
    class Recorder:
        def __init__(self):self.env=env;self.cfg=N(hdf5=True);self.nominal_image_size=None
        def start(self,obs0,info0=None,state0=None):self.initial=state0;self._snapshot_scene()
        def _snapshot_scene(self):
            self.scene_file=json.dumps({'metadata':{'task':{**known[0],'inst_to_name':{'target':'B'},'robot_poses':{'robot':['currentB']}}},
                'objects_info':{'init_info':{n:{}for n in objects}},'state':states[int(owner.dump_state(True)[0])]})
            self.env_config={'env':{'device':None},'task':{'tro_path':'oldA'},'robots':[{}]}
    class Executor:
        def __init__(self,recorder):self.env=env;self.recorder=recorder;self.state_owner=owner
        def _dump_state(self,full):
            old=registry._dump_filter
            if full:registry.set_dump_filter(lambda obj:True)
            try:return owner.dump_state(True)
            finally:registry.set_dump_filter(old)
    monkeypatch.setattr(model_pool,'native_properties',lambda actor:{'model':actor.name})
    runner=N(InstanceLoader=Loader,Recorder=Recorder,Executor=Executor)
    current={'active':'B','tro':members[1]['tro'],'ordinal':1}
    flush=install(runner,{'initialized_actors':initialized,'members':members},current,tmp_path,lambda *a,**k:source_config)
    loader=Loader();assert loader.restore()=='native restored'
    recorder=Recorder();executor=Executor(recorder)
    full=executor._dump_state(True);assert serialized_object_names(owner.deserialize(full)[0])==set(objects)
    recorder.start({},state0=full)
    assert serialized_object_names(owner.deserialize(recorder.initial)[0])=={'B','background'}
    assert recorder.env_config['task']['tro_path']==members[1]['tro']['path']
    assert set(recorder.scene_file['objects_info']['init_info'])=={'B','background'}
    changed=lambda obj:obj.name=='B';registry.set_dump_filter(changed)
    step=executor._dump_state(False);assert serialized_object_names(owner.deserialize(step)[0])=={'B'}
    assert registry._dump_filter is changed
    checkpoint=executor._dump_state(True);assert serialized_object_names(owner.deserialize(checkpoint)[0])==set(objects)
    assert registry._dump_filter is changed
    flush();audit=json.loads((tmp_path/'pool_recording_audit.json').read_text())
    assert [r['initial']for r in audit['observations']]==[True,False]
