from types import SimpleNamespace as N
import copy
import pytest
import torch
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_late_rebuild_probe import reconstruct, own_full_state


def native_fixture(changed_mass=False, extra_step=False):
    calls=[];sim=N(stage=object(),stopped=False,step_physics=lambda:None,_sim_context=N(step=lambda:None));sim.is_stopped=lambda:sim.stopped
    class View:
        mass=2.
        def is_physics_handle_valid(self):return True
        def get_masses(self):return [self.mass]
        def get_inertias(self):return [[1.,0.,0.,0.,2.,0.,0.,0.,3.]]
        def get_coms(self):return [[0.,0.,0.,0.,0.,0.,1.]]
    view=View()
    class Obj:
        name='body';links={'base':N(_rigid_prim_view=view)}
        def get_position_orientation(self):return torch.tensor([1.,2.,3.]),torch.tensor([0.,0.,0.,1.])
        def set_position_orientation(self,**kwargs):assert sim.stopped;calls.append('author_root')
    obj=Obj()
    class Registry:
        _dump_filter=staticmethod(lambda obj:False)
        def set_dump_filter(self,value):self._dump_filter=value
    registry=Registry()
    state={'registry':{'object_registry':{'body':{'root_link':{'pos':[1,2,3]}}}}}
    class Scene:
        objects=[obj];object_registry=registry
        def dump_state(self,serialized):assert not serialized and registry._dump_filter(obj);return state
        def load_state(self,saved,serialized):assert not serialized and saved==state;calls.append('restore_full_state')
    scene=Scene();loader=N(env=N(scene=scene),tro=N(meta=N(raw={'campaign':{'exact':'source'}})),_restoration_execution={'verified':True})
    def restart(meta,simulation,execution,while_stopped):
        assert meta=={'exact':'source'} and execution=={'verified':True};calls.append('stop');sim.stopped=True;while_stopped();sim.stopped=False;calls.append('play')
        return {'observed_physics_step_calls':['unexpected']if extra_step else[],'physics_steps_requested':0}
    def release():calls.append('release')
    def force():
        calls.append('force_load')
        if changed_mass:view.mass=3.
    return loader,sim,restart,N(release_physics_objects=release,force_load_physics_from_usd=force),N(get_attached_stage=lambda:0),calls


def test_rebuild_uses_own_full_state_and_preserves_static_properties_without_new_world():
    loader,sim,restart,sdk,route,calls=native_fixture();prior=loader.env.scene.object_registry._dump_filter
    result=reconstruct(loader,sim,restart,sdk,route)
    assert calls==['stop','author_root','release','force_load','play','restore_full_state']
    assert result['static_body_properties_bit_identical'] and result['sdk']['explicit_sdk_reconstructions']==1
    assert result['before_root_poses']==result['after_root_poses']
    assert loader.env.scene.object_registry._dump_filter is prior


@pytest.mark.parametrize('changed_mass,extra_step',[(True,False),(False,True)])
def test_rebuild_fails_closed_on_static_property_change_or_added_step(changed_mass,extra_step):
    loader,sim,restart,sdk,route,_=native_fixture(changed_mass,extra_step)
    with pytest.raises(ContractError):reconstruct(loader,sim,restart,sdk,route)


def test_full_capture_never_leaves_recording_filter_changed_on_error():
    loader,*_=native_fixture();scene=loader.env.scene;prior=scene.object_registry._dump_filter
    def fail(**kwargs):raise ValueError('capture failed')
    scene.dump_state=fail
    with pytest.raises(ValueError):own_full_state(scene)
    assert scene.object_registry._dump_filter is prior


def test_manual_attachment_route_is_rejected_before_mutation():
    loader,sim,restart,sdk,route,calls=native_fixture();route.get_attached_stage=lambda:1
    with pytest.raises(ContractError,match='timeline'):reconstruct(loader,sim,restart,sdk,route)
    assert calls==['stop']


def test_post_play_state_restore_cannot_hide_an_added_step_and_keeps_failure_audit():
    loader,sim,restart,sdk,route,_=native_fixture();event={}
    original=sim._sim_context.step
    def load(saved,serialized):sim._sim_context.step()
    loader.env.scene.load_state=load
    with pytest.raises(ContractError,match='additional physics step'):
        reconstruct(loader,sim,restart,sdk,route,event)
    assert event['whole_boundary_physics_step_calls']==['step']
    assert event['sdk']['explicit_sdk_reconstructions']==1
    assert sim._sim_context.step is original
