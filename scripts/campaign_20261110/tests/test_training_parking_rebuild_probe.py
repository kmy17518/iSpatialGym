from types import SimpleNamespace as N
import pytest
import torch
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_parking_rebuild_probe import author_fixed_parking


def fixture():
    calls = []
    class Actor:
        name='chair';fixed_base=True;kinematic_only=True;articulated=False;n_joints=0
        def __init__(self): self.pose=(torch.tensor([-249.84158325195312,-217.40103149414062,-200.1498565673828]),torch.tensor([0.,0.,0.,1.]))
        def get_position_orientation(self): return self.pose
        def set_position_orientation(self, position, orientation):
            calls.append((position,orientation));self.pose=(torch.tensor(position),torch.tensor(orientation))
    actor=Actor();scene=N(object_registry=lambda key,name:actor)
    state={'chair':{'root_link':{'pos':[6.264684200286865,15.34182071685791,.4798029363155365],'ori':[0.,0.,0.,1.]}}}
    return actor,scene,N(is_stopped=lambda:True),state,calls


def test_native_stopped_setter_receives_exact_source_components():
    actor,scene,sim,state,calls=fixture();result=author_fixed_parking(scene,sim,state)
    assert calls==[(state['chair']['root_link']['pos'],state['chair']['root_link']['ori'])]
    assert result[0]['source_root']==state['chair']['root_link'] and result[0]['after_stopped_authoring'][0]==state['chair']['root_link']['pos']


@pytest.mark.parametrize('field,value',[('fixed_base',False),('kinematic_only',False),('articulated',True),('n_joints',1)])
def test_nonfixed_or_articulated_actor_is_rejected_before_any_write(field,value):
    actor,scene,sim,state,calls=fixture();setattr(actor,field,value)
    with pytest.raises(ContractError,match='dynamic or articulated'):author_fixed_parking(scene,sim,state)
    assert calls==[]


def test_running_simulation_is_rejected_before_write():
    actor,scene,sim,state,calls=fixture();sim.is_stopped=lambda:False
    with pytest.raises(ContractError,match='stopped'):author_fixed_parking(scene,sim,state)
    assert calls==[]


def test_all_source_poses_are_validated_before_any_actor_is_changed():
    actor,scene,sim,state,calls=fixture();state['second']={'root_link':{'pos':[0.,0.,float('nan')],'ori':[0.,0.,0.,1.]}}
    with pytest.raises(ContractError,match='source pose'):author_fixed_parking(scene,sim,state)
    assert calls==[]


def test_installer_preserves_cold_callback_and_uses_only_existing_warm_lifecycle(tmp_path,monkeypatch):
    import json
    from ispatialgym.oracle import campaign_restore as CR
    from ispatialgym.oracle.campaign_batch import SCHEMA
    from scripts.campaign_20261110.contracts import file_digest
    from scripts.campaign_20261110.training_parking_rebuild_probe import install
    actor,scene,sim,states,writes=fixture();trace=[]
    def restart(meta,sim,*,execution=None,while_stopped=None):
        trace.append('stop')
        if while_stopped:while_stopped()
        trace.append('play')
        return {'observed_physics_step_calls':[],'physics_steps_requested':0}
    monkeypatch.setattr(CR,'restart_reused_world',restart)
    def attach(loader,shared):
        CR.restart_reused_world({},sim,while_stopped=lambda:trace.append('robot_anchor'))
        trace.append('ordinary_scene_and_parking_restore')
        return 'same_world'
    runner=N(InstanceLoader=N(attach=attach));source=tmp_path/'source.json';source.write_text('{}')
    reference={'path':str(source),'sha256':file_digest(source)}
    shared={'env':N(scene=scene),'sequential_augmentation':{'schema':SCHEMA,'parking_objects':['chair'],
        'protected_objects':['target'],'members_by_tro':{str(source):{'tro':reference,'scene':reference,'parking_states':states}}}}
    install(runner,{'parking_before_play_v1':True},tmp_path)
    CR.restart_reused_world({},sim,while_stopped=lambda:trace.append('cold_original'))
    assert trace==['stop','cold_original','play'] and not writes
    trace.clear()
    assert runner.InstanceLoader.attach(N(tro_path=str(source)),shared)=='same_world'
    assert trace==['stop','robot_anchor','play','ordinary_scene_and_parking_restore'] and len(writes)==1
    audit=json.loads((tmp_path/'parking_rebuild_audit.json').read_text())
    assert len(audit['events'])==1 and audit['events'][0]['status']=='completed'
