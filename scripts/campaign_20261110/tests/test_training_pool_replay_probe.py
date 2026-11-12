from types import SimpleNamespace as N
import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_pool_replay_probe import load_request, install_planner


def test_action_scope_is_explicit_and_parent_validation_only_projects_load_scope():
    request = {'pool_action_replay_v1': True, 'pool_preserve_native_anchor_v1': True,
               'load_only': False, 'diagnostic_control': 'ABA', 'production_released': False,
               'benchmark_credit': 0, 'source_sequence': {'sha256': 'exact'}}
    result = load_request(request)
    assert result['load_only'] and 'pool_action_replay_v1' not in result
    assert request['load_only'] is False and result['source_sequence'] == request['source_sequence']
    with pytest.raises(ContractError, match='scope'):
        load_request({**request, 'production_released': True})


def test_filter_switches_members_and_audit_reads_first_full_sync_once():
    objects = {n: N(name=n, links={'root': N(collision_meshes={'mesh': N(prim_path='/'+n+'/mesh')})})
               for n in ['A', 'B']}
    class Backend:
        def __init__(self):
            self.calls = 0; self._attached = {}
            self.robot = N(scene=N(object_registry=lambda key, name: objects[name]))
            self.checker = N(data=N(meshes=N(names=[['/A/mesh','/B/mesh']], enable=[[1,0]], count=[2])))
        def update_world(self): self.calls += 1; return 'synced'
    runner = N(make_obstacle_filter=lambda *a, **k: lambda obj: obj.name != 'outside', R1ProCuRobo=Backend)
    current = {'active': 'A', 'tro': 'A.json', 'ordinal': 0}; observations = []
    install_planner(runner, {'A','B'}, current, observations.append); backend = Backend()
    assert runner.make_obstacle_filter()(objects['A']) and not runner.make_obstacle_filter()(objects['B'])
    assert not runner.make_obstacle_filter()(N(name='outside'))
    assert backend.update_world() == 'synced'
    backend._attached['right'] = 'A'; backend.update_world()
    assert backend.calls == 2 and len(observations) == 1
    current.clear(); current.update(active='B', tro='B.json', ordinal=1)
    backend._attached.clear(); backend.checker.data.meshes.enable = [[0,1]]
    backend.update_world()
    assert len(observations) == 2 and observations[1]['objects']['A']['active'] is False
    assert runner.make_obstacle_filter()(objects['B']) and not runner.make_obstacle_filter()(objects['A'])
