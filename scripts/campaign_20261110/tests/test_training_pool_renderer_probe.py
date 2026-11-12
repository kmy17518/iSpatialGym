import copy
from types import SimpleNamespace as N
import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_pool_renderer_probe import (
    selected_pool_records, project_active_loaded, source_state_expected, prohibit_native_mutation)
from scripts.campaign_20261110.training_restoration import same_loaded_state


def records():
    return [{'artifacts':{'demo':{'path':'/a/demo.hdf5'}},'render_compatibility_sha256':'real_A'},
            {'artifacts':{'demo':{'path':'/b/demo.hdf5'}},'render_compatibility_sha256':'real_B'}]


def test_exact_fixed_pool_sequence_retains_each_real_record_compatibility():
    rows=records();before=copy.deepcopy(rows)
    out=selected_pool_records(rows,['/a','/b','/a'],['/a','/b','/a'])
    assert out['/a']['render_compatibility_sha256']=='real_A' and out['/b']['render_compatibility_sha256']=='real_B'
    assert rows==before


@pytest.mark.parametrize('wanted',[['/a','/b'],['/b','/a','/b'],['/a','/c','/a']])
def test_other_sequences_never_bypass_native_compatibility(wanted):
    with pytest.raises(ContractError):selected_pool_records(records(),wanted,['/a','/b','/a'])


def test_active_projection_removes_only_declared_inactive_and_keeps_background():
    source={'objects':{'A':1,'B':2,'background':3},'lights':{},'skybox':{}}
    out=project_active_loaded(source,{'pool_names':['A','B']},'A')
    assert out['objects']=={'A':1,'background':3} and source['objects']['B']==2
    with pytest.raises(ContractError):project_active_loaded(out,{'pool_names':['A','B']},'A')


def observation():
    return {'objects':{'A':{'category':'bottle','model':'fixed','fixed_base':False,'kinematic_only':False,
        'root_prim_type':'RigidDynamicPrim','scale':[1,1,1],'position':[1,2,3],'orientation':[0,0,0,1],
        'joint_positions':[],'joint_velocities':[],'linear_velocity':[0,0,0],'angular_velocity':[0,0,0]}},
        'lights':{},'skybox':{'intensity':1,'color':[1,1,1]}}


def test_native_frame_readback_uses_existing_strict_tolerances_against_source_values():
    actual=observation();state={'registry':{'object_registry':{'A':{'root_link':{'pos':[1,2,3],
        'ori':[0,0,0,1],'lin_vel':[0,0,0],'ang_vel':[0,0,0]},'joint_pos':[],'joint_vel':[]}}}}
    expected,observed=source_state_expected(actual,state);same_loaded_state(expected,observed)
    state['registry']['object_registry']['A']['root_link']['pos']=[1.1,2,3]
    expected,observed=source_state_expected(actual,state)
    with pytest.raises(ContractError,match='pose'):same_loaded_state(expected,observed)
    assert actual==observation()


@pytest.mark.parametrize('name',['step_physics','stop','play','step'])
def test_no_physics_or_lifecycle_step_can_be_hidden_and_methods_restore_on_failure(name):
    sim=N(step_physics=lambda:None,stop=lambda:None,play=lambda:None,_sim_context=N(step=lambda:None));events=[]
    target=sim._sim_context if name=='step'else sim;prior=getattr(target,name)
    with pytest.raises(ContractError):
        with prohibit_native_mutation(sim,events):getattr(target,name)()
    assert events==[name] and getattr(target,name)is prior
