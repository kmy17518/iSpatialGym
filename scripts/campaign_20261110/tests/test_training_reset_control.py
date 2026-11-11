import pytest
from scripts.campaign_20261110.training_batch_probe import episode_order


def test_reset_step_observation_brackets_only_the_existing_native_call():
    from types import SimpleNamespace
    from scripts.campaign_20261110.training_batch_probe import observe_reset_boundary
    calls=[]
    def step(token):calls.append(token);return 'native'
    sim=SimpleNamespace(step_physics=step)
    scene=SimpleNamespace(reset=lambda:sim.step_physics('unchanged'))
    original=scene.reset;completed=[]
    result,before=observe_reset_boundary(scene,sim,lambda:scene.reset(),lambda:len(calls),completed)
    assert result=='native' and before==0 and completed==[1]
    assert calls==['unchanged'] and scene.reset is original and sim.step_physics is step


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


def test_control_step_observes_existing_action_and_not_each_physics_substep():
    from scripts.campaign_20261110.training_batch_probe import observe_next_native_control_step
    class Env:
        physics_steps=0
        calls=[]
        def step(self, action, *, n_render_iterations):
            self.calls.append((action,n_render_iterations))
            self.physics_steps+=4
            return 'unchanged native result'
    env=Env();result={};action=object()
    cleanup=observe_next_native_control_step(env,lambda:{'steps':env.physics_steps},result)
    assert env.physics_steps==0 and not result
    assert env.step(action,n_render_iterations=2)=='unchanged native result'
    assert env.calls==[(action,2)] and 'step' not in vars(env)
    assert result['before']=={'steps':0} and result['after']=={'steps':4}
    assert result['method']=='env.step' and result['physics_steps_added']==0
    env.step(action,n_render_iterations=3);cleanup()
    assert result['after']=={'steps':4} and env.physics_steps==8


@pytest.mark.parametrize('failure', ['observation','native','unused'])
def test_control_step_cleanup_preserves_original_override_on_all_exit_paths(failure):
    from types import SimpleNamespace
    from scripts.campaign_20261110.training_batch_probe import observe_next_native_control_step
    calls=[]
    def step(*args,**kwargs):
        calls.append((args,kwargs))
        if failure=='native':raise RuntimeError('native failure')
    def observe():
        if failure=='observation':raise RuntimeError('observation failure')
        return {}
    env=SimpleNamespace(step=step);result={}
    cleanup=observe_next_native_control_step(env,observe,result)
    if failure!='unused':
        with pytest.raises(RuntimeError,match=failure+' failure'):env.step('action')
    cleanup();cleanup()
    assert env.step is step and result=={}
    assert len(calls)==(1 if failure=='native' else 0)
