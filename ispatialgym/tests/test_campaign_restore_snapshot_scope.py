import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.campaign_20261110.contracts import ContractError

from ispatialgym.oracle import campaign_restore as m



def meta():
    return {'campaign_id': 'campaign_20261110', 'split': 'train', 'class_id': 'disembedding_07',
            'task': 'disembedding', 'runtime': {'purpose': 'training_pipeline_canary'},
            'scene_setup': {'restoration_options': {'pristine_mass_before_reset_v1': True,
                                                   'complete_scene_snapshot_v1': True}}}


class Registry:
    def __init__(self):
        self.objects = ['cork', 'source_tray', 'destination_tray', 'table']
        self.calls = []
        self.filter_calls = []
        def changed(obj):
            self.filter_calls.append(obj)
            return obj == 'cork'
        self._dump_filter = changed

    def set_dump_filter(self, callback):
        self.calls.append(callback)
        self._dump_filter = callback

    def snapshot(self):
        return [obj for obj in self.objects if self._dump_filter(obj)]


def test_full_native_snapshots_without_advancing_recording_filter():
    registry = Registry(); prior = registry._dump_filter; report = {}
    with m.complete_scene_snapshot_scope(meta(), SimpleNamespace(object_registry=registry), report):
        assert registry.snapshot() == registry.objects
        assert registry.snapshot() == registry.objects
        assert registry.filter_calls == []
    assert registry._dump_filter is prior
    assert registry.snapshot() == ['cork']
    assert report['complete_scene_snapshot']['prior_filter_restored'] is True
    assert report['complete_scene_snapshot']['physics_steps_added'] == 0


def test_abort_restores_exact_callback_and_next_restore_is_complete():
    registry = Registry(); prior = registry._dump_filter; scene = SimpleNamespace(object_registry=registry)
    with pytest.raises(RuntimeError, match='native restore failed'):
        with m.complete_scene_snapshot_scope(meta(), scene, {}):
            assert registry.snapshot() == registry.objects
            raise RuntimeError('native restore failed')
    assert registry._dump_filter is prior
    with m.complete_scene_snapshot_scope(meta(), scene, {}):
        assert registry.snapshot() == registry.objects
    assert registry._dump_filter is prior


def test_disabled_scope_does_not_read_or_mutate_scene():
    report = {}
    with m.complete_scene_snapshot_scope({}, object(), report):
        pass
    assert report == {}


def test_wrong_scope_or_missing_registry_fails_before_mutation():
    registry = Registry(); bad = meta(); bad['runtime']['purpose'] = 'training_production'
    with pytest.raises(ContractError, match='exact authorized source'):
        with m.complete_scene_snapshot_scope(bad, SimpleNamespace(object_registry=registry), {}):
            pass
    assert registry.calls == []
    with pytest.raises(ContractError, match='registry filter'):
        with m.complete_scene_snapshot_scope(meta(), object(), {}):
            pass


def test_nested_scopes_restore_their_own_previous_filter():
    registry = Registry(); prior = registry._dump_filter; scene = SimpleNamespace(object_registry=registry)
    with m.complete_scene_snapshot_scope(meta(), scene, {}):
        outer = registry._dump_filter
        with m.complete_scene_snapshot_scope(meta(), scene, {}):
            assert registry.snapshot() == registry.objects
        assert registry._dump_filter is outer
    assert registry._dump_filter is prior


def test_production_execution_binds_exact_metadata_before_mutation():
    import copy
    from scripts.campaign_20261110.contracts import digest
    value = meta()
    value['runtime']['purpose'] = 'training_production'
    execution = {'schema': 'isg-training-restoration-execution-v1', 'authorization_checked': True,
                 'category': 'validation_canary', 'metadata_sha256': digest(value)}
    registry = Registry(); scene = SimpleNamespace(object_registry=registry)
    prior = registry._dump_filter
    with m.complete_scene_snapshot_scope(value, scene, {}, execution=execution):
        assert registry.snapshot() == registry.objects
    assert registry._dump_filter is prior
    changed = copy.deepcopy(value); changed['scene_setup']['extra'] = True
    registry.calls.clear()
    with pytest.raises(ContractError, match='exact authorized source'):
        with m.complete_scene_snapshot_scope(changed, scene, {}, execution=execution):
            pytest.fail('changed metadata accepted')
    assert registry.calls == []
    execution['authorization_checked'] = False
    with pytest.raises(ContractError, match='exact authorized source'):
        with m.complete_scene_snapshot_scope(value, scene, {}, execution=execution):
            pytest.fail('unchecked authorization accepted')
    assert registry.calls == []


def test_loader_holds_snapshot_scope_around_complete_restore_and_keeps_warm_restart_early():
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).parents[1] / 'oracle/run_isg.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'InstanceLoader')
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    scope = next(n for n in ast.walk(methods['restore']) if isinstance(n, ast.With))
    assert scope.items[0].context_expr.func.id == 'complete_scene_snapshot_scope'
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == '_restore_complete_snapshot' for n in ast.walk(scope))
    attach = ast.unparse(methods['attach'])
    assert attach.index('reset_pair_records(') < attach.index('restart_reused_world(') < attach.index('scene.load_state(')
    assert 'restart_reused_world' not in ast.unparse(methods['build_env'])


def test_conflicting_mass_options_fail_before_filter_mutation():
    value = meta(); value['scene_setup']['restoration_options']['serialized_mass_before_reset_v1'] = True
    registry = Registry()
    with pytest.raises(ContractError, match='exact bounded'):
        with m.complete_scene_snapshot_scope(value, SimpleNamespace(object_registry=registry), {}):
            pytest.fail('conflicting loader semantics accepted')
    assert registry.calls == []
