from types import SimpleNamespace as N
import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_pool_consumers import planner_observation, recorded_registry_scope, excluding_inactive

def objects():
    return {n:N(name=n,links={'root':N(collision_meshes={'mesh':N(prim_path='/'+n+'/mesh')})})for n in['A','B']}

def planner(names,mask):
    return N(checker=N(data=N(meshes=N(names=[names],enable=[mask],count=[len(names)]))))

def test_inactive_cached_disabled_mesh_is_excluded_and_active_enabled_is_required():
    out=planner_observation(planner(['/A/mesh','/B/mesh'],[1,0]),objects(),'A')
    assert out['objects']['B']['colliders']['/B/mesh']=={'allocated':True,'enabled':False}
    assert out['enabled_mesh_count']==1 and not out['trajectory_collision_clearance_verified']
    with pytest.raises(ContractError,match='participation'):
        planner_observation(planner(['/A/mesh','/B/mesh'],[1,1]),objects(),'A')
    with pytest.raises(ContractError,match='participation'):
        planner_observation(planner(['/A/mesh','/B/mesh'],[0,0]),objects(),'A')

def test_absent_inactive_mesh_is_valid_but_duplicate_native_names_are_not():
    assert planner_observation(planner(['/A/mesh'],[1]),objects(),'A')['native_enable_mask_read']
    with pytest.raises(ContractError,match='ambiguous'):
        planner_observation(planner(['/A/mesh','/A/mesh'],[1,0]),objects(),'A')

def test_existing_room_filter_is_preserved_even_without_bounds():
    keep=excluding_inactive(lambda o:o.name!='outside',{'A','B'},{'A'})
    assert keep(N(name='A')) and keep(N(name='table')) and not keep(N(name='B')) and not keep(N(name='outside'))
    assert not excluding_inactive(None,{'A','B'},{'A'})(N(name='B'))

class Registry:
    def __init__(self):
        self.queried=[]
        def changed(obj):self.queried.append(obj.name);return obj.name=='A'
        self._dump_filter=changed
    def set_dump_filter(self,callback):self._dump_filter=callback
    def dump(self):return [n for n in ['A','B','sleeping_table']if self._dump_filter(N(name=n))]

def test_initial_recording_keeps_sleeping_background_without_advancing_changed_filter():
    registry=Registry();prior=registry._dump_filter;scene=N(object_registry=registry)
    with recorded_registry_scope(scene,{'A','B'},{'A'},initial_state=True):
        assert registry.dump()==['A','sleeping_table'] and registry.queried==[]
    assert registry._dump_filter is prior
    assert registry.dump()==['A']

def test_step_recording_composes_prior_filter_and_restores_it_after_failure():
    registry=Registry();prior=registry._dump_filter;scene=N(object_registry=registry)
    with pytest.raises(RuntimeError):
        with recorded_registry_scope(scene,{'A','B'},{'A'},initial_state=False):
            assert registry.dump()==['A'] and registry.queried==['A','sleeping_table']
            raise RuntimeError('recorder failed')
    assert registry._dump_filter is prior
