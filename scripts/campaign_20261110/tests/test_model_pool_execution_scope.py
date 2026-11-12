"""CPU authorization/identity regressions; no claim of native physics coverage."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from ispatialgym.sampling import model_pool as pool
from scripts.campaign_20261110 import production_gate


@pytest.fixture
def selected(tmp_path, monkeypatch):
    asset = tmp_path/'asset.usd'; asset.write_text('exact asset bytes')
    row = dict(category='bottle', model='a', dataset='assets', scale=[1., 1., 1.], copies=1,
               ready=True, readiness_provenance={'measured': True}, consumers=['selected_A'],
               asset_path=str(asset), asset_sha256=pool.file_reference(asset)['sha256'])
    selection = tmp_path/'selection.json'
    world = {'scene': 'scene', 'rooms': ['room'], 'fixed_bindings': {'support': 'table'}}
    selection.write_text(json.dumps({'schema': 'isg-model-pool-selection-v1', 'campaign_id': 'campaign_20261110',
                                    'world_scope': world, 'selected_entries': [row]}))
    manifest = {'schema': pool.SCHEMA, 'campaign_id': 'campaign_20261110', 'campaign_root': str(tmp_path),
                'mode': pool.SELECTED_MODE, 'production_authorization': False, 'benchmark_credit': 0,
                'production_released': False, 'selected_scope': 'selected_A', 'batch_id': 'batch_A',
                'selection_proof': pool.file_reference(selection), 'max_objects': 2, 'entries': [row], **world}
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(manifest))
    marker = tmp_path/'.snapshot_commit'; marker.write_text('exact native code')
    runtime = pool.file_reference(marker); monkeypatch.setattr(pool, 'native_runtime_reference', lambda: pool.file_reference(marker))
    config = {'campaign_id': 'campaign_20261110', 'split': 'train', 'purpose': 'training_production',
              'template': {'split': 'train'}, 'validation_key': {'code_sha256': runtime['sha256']}}
    config_path = tmp_path/'config.json'; config_path.write_text(json.dumps(config))
    env = {'ISPATIALGYM_CAMPAIGN_CONFIG': str(config_path), pool.ENV: str(path)}
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG', str(config_path))
    auth = {'purpose': 'training_pipeline_canary', 'validation_scope': 'exact_production_configuration',
            'production_released': False, 'benchmark_credit': 0, 'start': 100, 'count': 1, 'max_instances': 1,
            'config_sha256': pool.file_reference(config_path)['sha256'], 'native_runtime_sha256': runtime['sha256'],
            'pool_manifest': pool.file_reference(path)}
    auth_path = tmp_path/'authorization.json'; auth_path.write_text(json.dumps(auth))
    registered = {'authorization_path': str(auth_path), 'authorization_sha256': pool.file_reference(auth_path)['sha256'],
                  'start': 100, 'count': 1}
    gate = Mock(return_value={'allowed': True, 'category': 'validation_canary',
                            'config_sha256': auth['config_sha256'], 'authorization': registered, 'gate_sha256': 'policy'})
    monkeypatch.setattr(production_gate, 'check_environment', gate)
    policy = Mock(); monkeypatch.setattr(pool, 'check_policy', policy)
    return SimpleNamespace(path=path, manifest=manifest, config=config, config_path=config_path,
                           authorization=auth, auth_path=auth_path, env=env, gate=gate, policy=policy, marker=marker)


def read(s, context=None):
    return pool.read_manifest(s.path, scene='scene', rooms=['room'], fixed_bindings={'support': 'table'},
                              execution_context=context)


def authorize(s):
    return pool.authorize_execution(s.path, environment=s.env, start=100, count=1)


def test_structural_audit_is_nonrecursive_but_cannot_activate(selected, tmp_path):
    value, ref = read(selected)
    selected.gate.assert_not_called(); selected.policy.assert_called_once()
    with pytest.raises(ValueError, match='checked execution context'):
        pool.ModelPool(object(), value, ref, tmp_path/'journal.json')


def test_exact_bounded_authorization_binds_native_config_range_and_pool(selected):
    context = authorize(selected); value, ref = read(selected, context)
    selected.gate.assert_called_once_with(selected.path.parent, selected.env, kind='collection', start=100, count=1)
    assert context['manifest'] == ref and context['configuration'] == pool.file_reference(selected.config_path)
    pool.validate_execution_context(value, ref, context)
    context['start'] = 101
    with pytest.raises(ValueError, match='range differs'):
        pool.validate_execution_context(value, ref, context)


@pytest.mark.parametrize('field,value', [('pool_manifest', {'path': '/other', 'sha256': 'wrong'}),
    ('native_runtime_sha256', 'wrong'), ('config_sha256', 'wrong'), ('start', 101), ('count', 2),
    ('validation_scope', 'generic'), ('production_released', True), ('benchmark_credit', 1), ('max_instances', 25)])
def test_unbound_or_expanded_canary_cannot_activate(selected, field, value):
    selected.authorization[field] = value; selected.auth_path.write_text(json.dumps(selected.authorization))
    selected.gate.return_value['authorization']['authorization_sha256'] = pool.file_reference(selected.auth_path)['sha256']
    with pytest.raises(ValueError): authorize(selected)


@pytest.mark.parametrize('category,allowed', [('standalone_training', False), ('eval_model', True),
                                            ('released_training', False)])
def test_other_gate_decisions_do_not_authorize_selected_execution(selected, category, allowed):
    selected.gate.return_value.update(category=category, allowed=allowed)
    with pytest.raises(ValueError): authorize(selected)


def test_released_path_requires_current_gate_and_exact_applicable_pool(selected):
    selected.gate.return_value.update(category='released_training', production_release={
        'production_authorization': True, 'applicable_pool': {'pool_manifest': pool.file_reference(selected.path)}})
    context = authorize(selected); read(selected, context)
    selected.gate.return_value['production_release']['applicable_pool']['pool_manifest']['sha256'] = 'different'
    with pytest.raises(ValueError, match='exact release'): authorize(selected)


@pytest.mark.parametrize('changed', ['config', 'runtime', 'manifest'])
def test_post_gate_mutation_is_rejected_before_preload(selected, changed):
    context = authorize(selected); value, ref = read(selected)
    path = {'config': selected.config_path, 'runtime': selected.marker, 'manifest': selected.path}[changed]
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='changed'):
        pool.validate_execution_context(value, ref, context)


@pytest.mark.parametrize('field,value', [('variant', ['one', 'two']), ('scene', None), ('templates', 2),
                                       ('instances_per_template', None), ('instances_per_template', 0), ('start_index', None)])
def test_native_entry_requires_actual_explicit_single_range(selected, field, value):
    args = SimpleNamespace(variant=['one'], scene='scene', templates=1, instances_per_template=1, start_index=100)
    setattr(args, field, value)
    with pytest.raises(ValueError): pool.execution_from_args(selected.path, args, selected.env)
    selected.gate.assert_not_called()


def test_manifest_asset_and_reservation_checks_are_still_mandatory(selected):
    context = authorize(selected)
    Path = type(selected.path)
    Path(selected.manifest['entries'][0]['asset_path']).write_text('changed')
    with pytest.raises(ValueError, match='asset changed'): read(selected, context)


def test_legacy_bounded_mode_has_no_new_gate_or_execution_requirement(selected):
    selected.manifest['mode'] = 'bounded_validation'; selected.path.write_text(json.dumps(selected.manifest))
    value, ref = read(selected)
    assert pool.authorize_execution(selected.path, environment={}, start=None, count=None) is None
    pool.validate_execution_context(value, ref, None); selected.gate.assert_not_called()


def test_same_world_aba_renews_exact_config_and_range_before_reused_activation(selected, tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    from ispatialgym.sampling import sample_isg as sampler
    # Two selected physical members, one preload, three distinct authorizations.
    second = copy.deepcopy(selected.manifest['entries'][0]); second['model'] = 'b'
    selected.manifest['entries'].append(second)
    selection_path = Path(selected.manifest['selection_proof']['path'])
    selection = json.loads(selection_path.read_text()); selection['selected_entries'] = selected.manifest['entries']
    selection_path.write_text(json.dumps(selection)); selected.manifest['selection_proof'] = pool.file_reference(selection_path)
    selected.path.write_text(json.dumps(selected.manifest)); selected.authorization['pool_manifest'] = pool.file_reference(selected.path)
    configs = {}
    for name in ('A', 'B'):
        cfg = copy.deepcopy(selected.config); cfg['template']['selected_target'] = name
        path = tmp_path/(name+'.json'); path.write_text(json.dumps(cfg)); configs[name] = path
    class World:
        def __init__(self): self.spawned = {}; self.preloads = 0; self.cameras = True
        def preload_pool_objects(self, specs):
            self.preloads += 1; return {s['name']: object() for s in specs}
        def reset_pool_object(self, obj): pass
        def park_pool_object(self, obj): pass
        def observe_model_pool(self, members, active): return {'active': sorted(active)}
    world = World(); checked = []
    def decision(root, env, *, kind, start, count):
        selected.authorization.update(config_sha256=pool.file_reference(env['ISPATIALGYM_CAMPAIGN_CONFIG'])['sha256'], start=start, count=count)
        selected.auth_path.write_text(json.dumps(selected.authorization))
        checked.append((env['ISPATIALGYM_CAMPAIGN_CONFIG'], start, count))
        return {'allowed': True, 'category': 'validation_canary', 'config_sha256': selected.authorization['config_sha256'],
                'authorization': {'authorization_path': str(selected.auth_path),
                                  'authorization_sha256': pool.file_reference(selected.auth_path)['sha256'],
                                  'start': start, 'count': count}, 'gate_sha256': 'current'}
    selected.gate.side_effect = decision
    monkeypatch.setenv(pool.ENV, str(selected.path)); monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG', str(configs['A']))
    initial = pool.authorize_execution(selected.path, environment=dict(__import__('os').environ), start=99, count=1)
    value, ref = read(selected, initial)
    world.model_pool = pool.ModelPool(world, value, ref, tmp_path/'execution.json', execution_context=initial)
    native = sampler.WorldPool(str(tmp_path/'native')); native.world = world; native.kit = object(); native.n_loads = 1
    native.key = ('scene', ('room',), ()); native.model_pool_key = ref['sha256']; native.crop_key = ('full',)
    native.model_pool_execution_context = initial
    stable = tmp_path/'stable.json'; stable.write_text('{}')
    monkeypatch.setitem(sys.modules, 'omnigibson', SimpleNamespace(shutdown=lambda: pytest.fail('borrowed world shutdown')))
    monkeypatch.setitem(sys.modules, 'ispatialgym.sampling.robot_utils', SimpleNamespace(RobotKit=object))
    monkeypatch.setitem(sys.modules, 'ispatialgym.sampling.sim_utils', SimpleNamespace(SimWorld=object, stable_scene_path=lambda _: str(stable)))
    monkeypatch.setitem(sys.modules, 'ispatialgym.sampling.scene_crop', SimpleNamespace(request_key=lambda *_: ('full',), covers=lambda *_: True))
    monkeypatch.setattr(sampler, 'configure_logging', lambda _: None)
    monkeypatch.setattr(sampler, 'world_key', lambda *_: native.key)
    monkeypatch.setattr(sampler, 'world_cameras_required', lambda _: True)
    monkeypatch.setattr(sampler, 'scene_crop_request', lambda *_: ('full', {}, None))
    problem = SimpleNamespace(subtask='task', variant='variant', bindings=lambda _: {'support': 'table'})
    acquired = []
    def run(args, variant, actual, report, *_):
        assert actual is native
        same_world, _ = actual.get(problem, 'scene'); assert same_world is world
        context = world.model_pool.execution_context
        assert context['configuration'] == pool.file_reference(configs[variant])
        assert context['start'] == args.start_index
        acquired.append(world.model_pool.acquire('target', 'bottle', variant.lower(), 'assets', [1., 1., 1.]))
        world.model_pool.release_template()
        report['problems'][variant] = {'template': {'n_written': 1}}
    monkeypatch.setattr(sampler, 'run_variant', run)
    for i, name in enumerate(('A', 'B', 'A')):
        monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG', str(configs[name]))
        code, _, returned = sampler.main(['--variant', name, '--scene', 'scene', '--templates', '1',
            '--instances-per-template', '1', '--start-index', str(100+i), '--customs', str(tmp_path/'none'),
            '--out', str(tmp_path/('output'+str(i)))], shutdown=False, existing_pool=native)
        assert code == 0 and returned is native and native.world is world
    assert world.preloads == 1 and acquired[0] is acquired[2] and acquired[0] is not acquired[1]
    assert checked[-3:] == [(str(configs[name]), 100+i, 1) for i, name in enumerate(('A', 'B', 'A'))]
    assert [e['execution_context']['start'] for e in world.model_pool.events if e['event'] == 'acquired'] == [100, 101, 102]
