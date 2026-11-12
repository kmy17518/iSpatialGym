import copy
from types import SimpleNamespace as N
import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_pool_renderer_probe import (
    selected_pool_records, project_active_loaded, source_state_expected, prohibit_native_mutation, record_comparison,
    capture_scene_input, declared_delta, checked_episode_results, DELTA)
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


def test_first_strict_native_mismatch_retains_exact_cold_and_actual_evidence():
    expected=observation();actual=copy.deepcopy(expected);actual['objects']['A']['position']=[10,2,3];saved=[]
    with pytest.raises(ContractError):record_comparison(expected,actual,same_loaded_state,saved.append)
    assert len(saved)==1 and saved[0]['strict_match']is False and 'error'in saved[0]
    assert saved[0]['cold_native_expected']==expected and saved[0]['native_observed']==actual


def test_native_in_place_tensor_conversion_cannot_change_or_break_reviewed_input_hash():
    import torch
    from scripts.campaign_20261110.contracts import digest
    private={'state':{'root':{'position':[1.,2.,3.]}}}
    saved,sha=capture_scene_input(private)
    private['state']['root']['position']=torch.tensor(private['state']['root']['position'])
    assert saved=={'state':{'root':{'position':[1.,2.,3.]}}} and digest(saved)==sha


def cleanup_rows(fault='failure'):
    return [{'tag':'REPLAY_EPISODE_ABORTED','demo':'A','kept':False,'cleanup':{
        'exception_type':{'failure':'ControlledRenderFailure','interruption':'ControlledRenderInterrupt'}[fault],
        'boundary':'after_native_writer_action','frames_observed':2,'native_writer_abort_called':True,
        'trajectory_buffer_empty':True,'dataset_pending_frames':False,'writer_aborted':True,
        'physics_steps_added':0,'render_ticks_added':0,'expected_controlled_failure':True}},
        {'tag':'REPLAY_DEMO','demo':'B','kept':True},{'tag':'REPLAY_DEMO','demo':'A','kept':True}]


@pytest.mark.parametrize('fault',['failure','interruption'])
def test_cleanup_preserves_failed_prefix_and_requires_both_followthrough_members(fault):
    completed,aborted=checked_episode_results(cleanup_rows(fault),['A','B','A'],fault)
    assert [r['demo']for r in completed]==['B','A'] and len(aborted)==1
    assert declared_delta({})==DELTA
    assert declared_delta({'renderer_cleanup':fault})['native_writer_cleanup']['fault']==fault


@pytest.mark.parametrize('mutation',['kept','pending','extra_render','wrong_exception','short_prefix','missing_recovery'])
def test_cleanup_cannot_promote_abort_or_skip_native_buffer_and_recovery_checks(mutation):
    rows=cleanup_rows();proof=rows[0]['cleanup']
    if mutation=='kept':rows[0]['kept']=True
    elif mutation=='pending':proof['dataset_pending_frames']=True
    elif mutation=='extra_render':proof['render_ticks_added']=1
    elif mutation=='wrong_exception':proof['exception_type']='ValueError'
    elif mutation=='short_prefix':proof['frames_observed']=1
    elif mutation=='missing_recovery':rows.pop()
    with pytest.raises(ContractError):checked_episode_results(rows,['A','B','A'],'failure')


def test_ordinary_control_rejects_an_undeclared_abort():
    with pytest.raises(ContractError):checked_episode_results(cleanup_rows(),['A','B','A'],None)
    with pytest.raises(ContractError):declared_delta({'renderer_cleanup':'pretend_success'})
