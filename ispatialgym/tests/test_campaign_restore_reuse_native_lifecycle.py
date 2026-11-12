import importlib.util
from pathlib import Path
import pytest
from scripts.campaign_20261110.contracts import ContractError
from ispatialgym.oracle import campaign_restore as m

class Sim:
 def __init__(self):self.playing=True;self.current_time=4.;self.current_time_step_index=480;self.dt=1/120;self.calls=[];self.extra_step=False;self.change_dt=False;self._sim_context=Context(self)
 def is_playing(self):return self.playing
 def is_stopped(self):return not self.playing
 def get_physics_dt(self):return self.dt
 def get_rendering_dt(self):return 1/30
 def get_sim_step_dt(self):return 1/30
 def step_physics(self):self.calls.append('step')
 def stop(self):self.calls.append('stop');self.playing=False;self.current_time=0.;self.current_time_step_index=0
 def play(self):
  self.calls.append('play');self.playing=True
  if self.extra_step:self._sim_context.step()
  if self.change_dt:self.dt=.1
class Context:
 def __init__(self,sim):self.sim=sim
 def step(self):self.sim.calls.append('context_step')
def meta():return {'campaign_id':'campaign_20261110','split':'train','class_id':'disembedding_07','task':'disembedding',
 'runtime':{'purpose':'training_pipeline_canary'},'scene_setup':{'restoration_options':{'pristine_mass_before_reset_v1':True,'reuse_native_lifecycle_v1':True}}}
def test_default_noop_and_exact_native_lifecycle():
 sim=Sim();assert m.restart_reused_world({},sim)is None and not sim.calls
 v=m.restart_reused_world(meta(),sim);assert sim.calls==['stop','play']and v['observed_physics_step_calls']==[]
 assert v['clock_before']['current_time']==4 and v['clock_after']['current_time']==0
 assert 'step_physics'not in vars(sim)and'step'not in vars(sim._sim_context)
def test_native_added_step_cannot_silently_pass():
 sim=Sim();sim.extra_step=True
 with pytest.raises(ContractError,match='stepping call'):m.restart_reused_world(meta(),sim)
 assert 'step_physics'not in vars(sim)and'step'not in vars(sim._sim_context)
def test_settings_or_purpose_change_rejected():
 sim=Sim();sim.change_dt=True
 with pytest.raises(ContractError,match='time-step settings'):m.restart_reused_world(meta(),sim)
 v=meta();v['runtime']['purpose']='training_production';sim=Sim()
 with pytest.raises(ContractError,match='bounded source'):m.restart_reused_world(v,sim)
 assert not sim.calls


def test_production_execution_binds_exact_metadata_before_mutation():
    import copy
    from scripts.campaign_20261110.contracts import digest
    value = meta()
    value['runtime']['purpose'] = 'training_production'
    execution = {'schema': 'isg-training-restoration-execution-v1', 'authorization_checked': True,
                 'category': 'validation_canary', 'metadata_sha256': digest(value)}
    sim = Sim()
    assert m.restart_reused_world(value, sim, execution=execution)['physics_steps_requested'] == 0
    changed = copy.deepcopy(value); changed['scene_setup']['extra'] = True
    sim = Sim()
    with pytest.raises(ContractError, match='bounded source'):
        m.restart_reused_world(changed, sim, execution=execution)
    assert sim.calls == []


def test_failed_play_restores_observation_wrappers():
    sim = Sim()
    def failed_play():
        raise RuntimeError('native play failed')
    sim.play = failed_play
    with pytest.raises(RuntimeError, match='native play failed'):
        m.restart_reused_world(meta(), sim)
    assert 'step_physics' not in vars(sim)
    assert 'step' not in vars(sim._sim_context)
