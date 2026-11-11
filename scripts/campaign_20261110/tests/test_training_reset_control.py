import pytest
from scripts.campaign_20261110.training_batch_probe import episode_order


def test_default_remains_distinct_a_b_a():
    assert episode_order({},'physics_action_open_loop')==[0,1,0]
    assert episode_order({},'kinematic_observation_rendering')==[0,1,0]


def test_repeated_a_requires_explicit_nonrelease_diagnostic():
    request={'diagnostic_control':'A_A','diagnostic_only':True,'production_released':False}
    assert episode_order(request,'physics_action_open_loop')==[0,0]
    for changed in ({'diagnostic_only':False},{'production_released':True},{'diagnostic_control':'A_B'}):
        with pytest.raises(ValueError):episode_order({**request,**changed},'physics_action_open_loop')
    with pytest.raises(ValueError):episode_order(request,'kinematic_observation_rendering')


def test_next_step_observation_uses_exactly_one_existing_call_and_restores_method():
    from types import SimpleNamespace
    from scripts.campaign_20261110.training_batch_probe import observe_next_native_step
    calls=[]
    def step(value):calls.append(value);return 'native-result'
    sim=SimpleNamespace(step_physics=step);result={}
    observe_next_native_step(sim,lambda:{'native_steps':len(calls)},result)
    assert calls==[]
    assert sim.step_physics(17)=='native-result'
    assert calls==[17] and sim.step_physics is step
    assert result['before']=={'native_steps':0} and result['after']=={'native_steps':1}
    assert result['physics_steps_added']==0


def test_next_step_native_error_propagates_and_restores_method():
    from types import SimpleNamespace
    from scripts.campaign_20261110.training_batch_probe import observe_next_native_step
    def step():raise RuntimeError('actual native failure')
    sim=SimpleNamespace(step_physics=step);result={}
    observe_next_native_step(sim,lambda:{},result)
    with pytest.raises(RuntimeError,match='native failure'):sim.step_physics()
    assert sim.step_physics is step and result=={}
