import copy
from types import SimpleNamespace as NS

import pytest

from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_pool_adapter import (
    merge_member_state, restart_without_steps, pool_restore_document, active_export)
from scripts.campaign_20261110.training_pool_world import scene_contract, isolated_member_scene
from scripts.campaign_20261110.tests.test_training_pool_world import scenes


def test_member_state_uses_incoming_actors_and_keeps_initialized_robot_articulation():
    baseline={'registry':{'object_registry':{'robot':{'root_link':{'pos':[0]*3,'ori':[0,0,0,1]},
                                                    'joint_positions':[1,2]},
                                           'A':{'root_link':{'pos':[9]*3}}}}}
    member={'state':{'registry':{'object_registry':{'A':{'root_link':{'pos':[1,2,3]}}}}}}
    saved=copy.deepcopy(baseline)
    actual=merge_member_state(baseline,member,'robot',{'position':[4,5,6],'orientation':[0,0,0,1]},
                              lambda p,q:(list(p),list(q)))
    assert baseline==saved
    assert actual['registry']['object_registry']['A']==member['state']['registry']['object_registry']['A']
    assert actual['registry']['object_registry']['robot']=={'root_link':{'pos':[4,5,6],'ori':[0,0,0,1]},'joint_positions':[1,2]}


def test_native_restart_changes_anchor_in_stopped_state_and_preserves_callbacks():
    events=[]
    class Sim:
        playing=True
        def is_playing(self):return self.playing
        def is_stopped(self):return not self.playing
        def stop(self):self.playing=False;events.append('stop')
        def play(self):self.playing=True;events.append('play')
        def step_physics(self):events.append('unexpected_step')
        def get_physics_dt(self):return 1/120
        def get_rendering_dt(self):return 1/30
        def get_sim_step_dt(self):return 1/30
    sim=Sim();sim._sim_context=NS(step=lambda:events.append('unexpected_context_step'))
    native=sim._sim_context.step
    def anchor():assert sim.is_stopped();events.append('anchor')
    result=restart_without_steps(sim,anchor)
    assert events==['stop','anchor','play'] and result['physics_steps_requested']==0
    assert 'step_physics'not in vars(sim) and sim._sim_context.step is native
    with pytest.raises(ContractError,match='stepped physics'):
        restart_without_steps(sim,lambda:sim.step_physics())


def test_export_removes_inactive_actor_and_preserves_active_model_scale_state():
    documents=scenes();contract=scene_contract(documents,['campaign_pool_A','campaign_pool_B','campaign_pool_A'],{'sha256':'pool'})
    union,isolation=isolated_member_scene(contract,documents[1],'campaign_pool_B')
    restore=pool_restore_document(union,isolation);saved=copy.deepcopy(restore)
    exported=active_export(restore,list(isolation),'campaign_pool_B')
    assert restore==saved
    assert 'campaign_pool_A'not in exported['objects_info']['init_info']
    assert 'campaign_pool_A'not in exported['state']['registry']['object_registry']
    assert exported['metadata']['ispatialgym']['model_pool']['active_properties']=={
        'campaign_pool_B':isolation['campaign_pool_B']['properties']}
    restore['metadata']['unresolved_reference']='campaign_pool_A'
    with pytest.raises(ValueError,match='inactive object remains referenced'):
        active_export(restore,list(isolation),'campaign_pool_B')


def test_stopped_authoring_checks_whole_inventory_before_mutation():
    from scripts.campaign_20261110.training_pool_adapter import author_isolation_stopped
    events=[]
    class Path:
        def __init__(self, value):self.value=value
        def MakeRelativePath(self, base):return self.value
        def __str__(self):return self.value
    class Prim:
        def GetPath(self):return Path('base/collision')
    class Attribute:
        def __init__(self, label):self.label=label
        def Set(self, value):events.append((self.label,value))
    collision=NS(GetPrim=lambda:Prim(),GetCollisionEnabledAttr=lambda:Attribute('collision'))
    obj=NS(category='bottle',model='exact',scale=NS(tolist=lambda:[1.,1.,1.]),n_joints=0,
           links={'base':NS(prim=Prim(),_collision_apis=[collision])},prim_path='/actor',visible=True,
           set_position_orientation=lambda **kw:events.append(('pose',kw)))
    scene=NS(object_registry=lambda kind,name:obj if name=='actor'else None)
    row={'active':False,'properties':{'category':'bottle','model':'exact','scale':[1.,1.,1.],
         'visible':False,'links':{'base':{'gravity_disabled':True,'collisions':{'base/collision':False}}}},
         'state':{'root_link':{'pos':[40,40,5],'ori':[0,0,0,1],'lin_vel':[0,0,0],'ang_vel':[0,0,0]}}}
    sim=NS(is_stopped=lambda:True)
    gravity=lambda prim:NS(GetDisableGravityAttr=lambda:Attribute('gravity'))
    with pytest.raises(ContractError,match='initialized geometry'):
        author_isolation_stopped(scene,sim,{'actor':row,'absent':row},gravity_api=gravity)
    assert events==[]
    result=author_isolation_stopped(scene,sim,{'actor':row},gravity_api=gravity)
    assert events==[('gravity',True),('collision',False),('pose',{'position':[40,40,5],'orientation':[0,0,0,1],'frame':'scene'})]
    assert obj.visible is False and result['native_velocity_writes']==0 and result['physics_steps_requested']==0
    sim.is_stopped=lambda:False
    with pytest.raises(ContractError,match='stopped physics'):
        author_isolation_stopped(scene,sim,{'actor':row},gravity_api=gravity)
