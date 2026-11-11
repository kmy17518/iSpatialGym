from types import SimpleNamespace
import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_batch_probe import (
    ControlledEpisodeAbort, abort_after_native_control_step, episode_order,
    observe_next_native_control_step)


def test_abort_executes_exact_action_then_restores_both_observers():
    calls=[]
    class Environment:
        def step(self, action, *, option):
            calls.append((action, option))
            return 'native-result'
    env=Environment(); observed={}; aborted={}
    observe_cleanup=observe_next_native_control_step(env, lambda:len(calls), observed)
    abort_cleanup=abort_after_native_control_step(env, aborted)
    with pytest.raises(ControlledEpisodeAbort):env.step([1, 2], option='unchanged')
    abort_cleanup();observe_cleanup()
    assert calls==[([1, 2], 'unchanged')]
    assert observed['before']==0 and observed['after']==1
    assert aborted['native_control_calls_completed']==1
    assert 'step' not in vars(env)
    assert env.step('next', option='same')=='native-result'


def test_unused_abort_cleanup_restores_existing_instance_method():
    native=lambda action:action
    env=SimpleNamespace(step=native)
    cleanup=abort_after_native_control_step(env,{})
    cleanup();cleanup()
    assert env.step is native


def test_real_native_failure_is_not_mislabeled_as_controlled_abort():
    def fail(action):raise ValueError('real native failure')
    env=SimpleNamespace(step=fail);proof={}
    cleanup=abort_after_native_control_step(env,proof)
    with pytest.raises(ValueError,match='real native'):env.step('action')
    cleanup()
    assert proof=={} and env.step is fail


@pytest.mark.parametrize('control,order',[('FRESH_A',[0]),('A_A',[0,0]),('A_ABORT_A',[0,0])])
def test_controls_are_explicit_and_cannot_be_render_release(control,order):
    value={'diagnostic_control':control,'diagnostic_only':True,'production_released':False}
    assert episode_order(value,'physics_action_open_loop')==order
    with pytest.raises(ContractError):episode_order(value,'kinematic_observation_rendering')
    value['diagnostic_only']=False
    with pytest.raises(ContractError):episode_order(value,'physics_action_open_loop')
