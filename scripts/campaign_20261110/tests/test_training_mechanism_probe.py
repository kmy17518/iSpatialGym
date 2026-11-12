import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_mechanism_probe import scope, ORDERS, DELTA
from scripts.campaign_20261110.training_mechanism_probe import REBUILD_DELTA


def request(control):
    return {'schema':'isg-retained-mechanism-replay-request-v1','campaign_id':'campaign_20261110',
        'control':control,'physical_delta':dict(DELTA),'benchmark_credit':0,'production_released':False,
        'max_native_episodes':len(ORDERS[control]),'timeout_seconds':1800}


@pytest.mark.parametrize('control',list(ORDERS))
def test_exact_retained_source_controls_are_finite_and_never_new_draws(control):
    assert scope(request(control)) == ORDERS[control] and set(scope(request(control))) == {0}


@pytest.mark.parametrize('field,value',[('production_released',True),('benchmark_credit',1),
    ('timeout_seconds',36000),('max_native_episodes',3),('control','ABA'),
    ('physical_delta',{**DELTA,'additional_physics_steps':1})])
def test_control_cannot_expand_episodes_release_scope_or_change_physics(field,value):
    with pytest.raises(ContractError):scope({**request('AA'),field:value})


def test_declared_reconstruction_counts_cold_and_reused_episodes_without_input_changes():
    q={**request('AA'),'physics_reconstruction':'late_incoming_v1','physical_delta':REBUILD_DELTA}
    assert scope(q)==[0,0]
    assert REBUILD_DELTA['additional_physics_steps']==0 and not REBUILD_DELTA['reference_state_copied']


@pytest.mark.parametrize('change',[{'physics_reconstruction':'unknown'},
    {'physics_reconstruction':'late_incoming_v1'}, {'physical_delta':REBUILD_DELTA},
    {'physics_reconstruction':'late_incoming_v1','physical_delta':{**REBUILD_DELTA,'additional_stop_play_pairs':0}}])
def test_reconstruction_cannot_be_unreported_or_use_a_different_native_route(change):
    with pytest.raises(ContractError):scope({**request('AA'),**change})


def test_shared_lifecycle_preserves_existing_step_settings_and_exposes_clock_reset():
    from types import SimpleNamespace as N
    from ispatialgym.oracle.campaign_restore import restart_loaded_physics
    calls=[];sim=N(playing=True,current_time=3.,current_time_step_index=60)
    sim.is_playing=lambda:sim.playing;sim.is_stopped=lambda:not sim.playing
    sim.get_physics_dt=lambda:1/120;sim.get_rendering_dt=lambda:1/30;sim.get_sim_step_dt=lambda:1/30
    sim.step_physics=lambda:calls.append('step');sim._sim_context=N(step=lambda:calls.append('context_step'))
    def stop():sim.playing=False;calls.append('stop')
    def play():sim.playing=True;sim.current_time=0.;sim.current_time_step_index=0;calls.append('play')
    sim.stop=stop;sim.play=play
    before=sim.step_physics
    result=restart_loaded_physics(sim,while_stopped=lambda:calls.append('reconstruct'))
    assert calls==['stop','reconstruct','play'] and sim.step_physics is before
    assert result['observed_physics_step_calls']==[] and result['physics_steps_requested']==0
    assert result['time_step_settings_before']==result['time_step_settings_after']
    assert result['clock_before']['current_time']==3. and result['clock_after']['current_time']==0.


def test_legacy_lifecycle_does_not_opt_unrelated_sources_into_reconstruction():
    from ispatialgym.oracle.campaign_restore import restart_reused_world
    assert restart_reused_world({},None) is None
    with pytest.raises(ContractError):
        restart_reused_world({'scene_setup':{'restoration_options':{'reuse_native_lifecycle_v1':True}}},None)
