import copy
from dataclasses import asdict
import json
import sys
from pathlib import Path
from types import SimpleNamespace as N

import pytest
import torch

from ispatialgym.sampling.selected_context import construction_plan, construct_and_place, merge_native_actor
from scripts.campaign_20261110.contracts import ObjectRef, Template, digest, file_digest
from scripts.campaign_20261110.tests import test_prepared_articulation as prepared
from scripts.campaign_20261110.training_construct_selected_context import inputs, join_constructed

case = prepared.case
document = prepared.document


@pytest.fixture
def selected(case):
    asset = case.tmp / 'dataset/objects/door/model/usd/model.encrypted.usd'
    asset.parent.mkdir(parents=True); asset.write_bytes(b'exact selected asset')
    sha = file_digest(asset)
    numeric = document(case.tmp / 'numeric.json', {'status': 'native_defaults_and_geometry_recorded',
        'objects': [{'category': 'door', 'model': 'model', 'dataset': 'behavior-1k-assets',
                     'asset_path': str(asset), 'asset_sha256': sha, 'ready': True, 'default_scale': [1., 1., 1.]}]})
    choice = json.loads(Path(case.options['selection_ref']['path']).read_text())['rows'][0]
    choice['rooms'] = ['garden_0']; choice['roles'][0].update(
        asset_sha256=sha, original_default_scale=[1., 1., 1.], applied_scale=[1.2, 1.2, 1.2],
        numeric_preparation_evidence=numeric)
    selection = document(case.tmp / 'context_selection.json', {'rows': [choice]})
    row = copy.deepcopy(case.row); row.update(asset_sha256=sha, applied_scale=[1.2, 1.2, 1.2], scale_row_sha256=digest(choice))
    metadata = document(case.tmp / 'context_handles.json', {
        'schema': 'isg-exact-articulation-handle-binding-v1', 'source': selection, 'rows': [row]})
    source = {'objects_info': {'init_info': {
        'prototype': {'args': {'category': 'door', 'model': 'original', 'fixed_base': True, 'in_rooms': ['garden_0']}},
        'floor': {'args': {'category': 'floors', 'model': 'native_floor', 'in_rooms': ['garden_0']}}}},
        'state': {'registry': {'object_registry': {
            'prototype': {'root_link': {'pos': [3., 0., 1.]}}, 'floor': {'root_link': {'pos': [0., 0., -.1]}}}}}}
    scene = document(case.tmp / 'house_single_floor_stable.json', source)
    options = dict(metadata_ref=metadata, selection_ref=selection, configuration_id='config_A', variant='alias_A',
        scene_ref=scene, numeric_ref=numeric, prototype_name='prototype', floor_name='floor', room='garden_0',
        yaw_radians=[0., 1.57], max_candidates=8)
    return N(case=case, source=source, options=options, asset=asset, row=row, plan=construction_plan(**options))


def test_constructor_identity_retains_original_defaults_and_initializes_applied_scale(selected):
    plan = selected.plan
    assert plan['original_default_scale'] == [1., 1., 1.]
    assert plan['applied_scale'] == [1.2, 1.2, 1.2] and plan['fixed_base'] is True
    assert plan['model'] == 'model' and selected.source['objects_info']['init_info']['prototype']['args']['model'] == 'original'
    assert plan == construction_plan(**selected.options)
    assert not plan['motion_readiness_verified'] and not plan['production_released']


@pytest.mark.parametrize('change', ['asset', 'floor', 'prototype', 'room', 'scene', 'bound', 'yaw'])
def test_wrong_construction_inputs_fail_before_creating_a_world(selected, change):
    options = dict(selected.options)
    if change == 'asset': selected.asset.write_bytes(b'changed asset')
    if change == 'floor': options['floor_name'] = 'prototype'
    if change == 'prototype': options['prototype_name'] = 'floor'
    if change == 'room': options['room'] = 'sauna_0'
    if change == 'scene': options['scene_ref'] = document(selected.case.tmp / 'other_stable.json', selected.source)
    if change == 'bound': options['max_candidates'] = 257
    if change == 'yaw': options['yaw_radians'] = [float('nan')]
    with pytest.raises(ValueError): construction_plan(**options)


def exported(selected):
    plan = selected.plan; name = plan['actor_name']
    live = copy.deepcopy(selected.source)
    live['objects_info']['init_info']['prototype']['args']['model'] = 'do_not_export_mutation'
    live['objects_info']['init_info'][name] = {'args': {'category': plan['category'], 'model': plan['model'],
        'scale': plan['applied_scale'], 'fixed_base': plan['fixed_base'], 'dataset_name': plan['dataset']}}
    live['state']['registry']['object_registry'][name] = {'root_link': {'pos': [1., 2., 3.]}}
    metadata = {'name': name, 'model_key': 'door/model', 'original_default_scale': plan['original_default_scale'],
                'applied_scale': plan['applied_scale']}
    return merge_native_actor(selected.source, live, plan, metadata)


def test_private_export_preserves_source_actors_and_separates_applied_from_default_scale(selected):
    source = copy.deepcopy(selected.source); result = exported(selected)
    assert selected.source == source
    for name, value in source['objects_info']['init_info'].items():
        assert result['objects_info']['init_info'][name] == value
        assert result['state']['registry']['object_registry'][name] == source['state']['registry']['object_registry'][name]
    meta = result['metadata']['ispatialgym']['campaign_added_context_objects'][0]
    assert meta['applied_scale'] == [1.2] * 3 and meta['original_default_scale'] == [1.] * 3


def test_constructed_context_flows_into_normal_generation_manifest(selected):
    case, plan = selected.case, selected.plan
    out = case.tmp / 'construction'; out.mkdir()
    scene_dir = out / 'stable_scenes'; scene_dir.mkdir()
    context = document(scene_dir / 'house_single_floor_stable.json', exported(selected))
    proof = document(out / 'placement.json', {'plan': plan, 'native_scene': context, 'original_scene_entries_unchanged': True})
    customs = document(case.tmp / 'generation_customs.json', {'alias_A': case.entry})
    target = ObjectRef('target', 'target', 'door', 'model', plan['asset_sha256'], joint='j_leaf', ready=True,
                       readiness_source='catalog')
    template = Template('source_A', 'configuration_matching-articulation_close_large_scale', 'articulation_A',
        'train', plan['scene'], 'garden_0', (target,), digest('definition'), {'task_rooms': plan['rooms']})
    scale_plan = {'scales': {'target': {'default': [1.] * 3, 'applied': [1.2] * 3, 'mode': 'uniform'}},
                  'removed_objects': [], 'lighting': {}}
    job = {'candidate_id': template.template_id, 'class_id': template.class_id, 'scene': template.scene,
        'alias': 'alias_A', 'production_released': False, 'training_scale_plan': scale_plan,
        'sample_argv': ['python', '-m', 'ispatialgym.sampling.sample_isg', '--customs', customs['path'],
                        '--scene', plan['scene'], '--out', 'old_sampling', '--no-write-back']}
    request = {'metadata': selected.options['metadata_ref'], 'selection': selected.options['selection_ref'],
               'configuration_id': 'config_A', 'source_scene': selected.options['scene_ref'], 'source_customs': customs}
    candidate, bound, manifest = join_constructed(request, template, job, plan, context, proof, out)
    assert bound['sample_argv'][:3] == job['sample_argv'][:3]
    assert bound['environment']['ISPATIALGYM_STABLE_SCENE_DIR'] == str(scene_dir)
    assert candidate.scene_setup['inst_to_name']['target'] == plan['actor_name']
    assert candidate.scene_setup['selected_context_generation']['source_scene'] == selected.options['scene_ref']
    assert Path(manifest['path']).is_file()


@pytest.mark.parametrize('hit_selected_floor, blocked', [(True, False), (False, False), (True, True)])
def test_native_factory_creates_exact_variant_once_and_placement_is_finite(selected, monkeypatch, hit_selected_floor, blocked):
    plan = selected.plan; calls = []
    class Actor:
        name = plan['actor_name']; category = 'door'; model = 'model'
        scale = torch.tensor(plan['applied_scale'])
        joints = {'joint_j_leaf': N(is_revolute=True, lower_limit=0., upper_limit=1.5)}
        links = {'root': N(prim_path='/selected')}
        def set_position_orientation(self, pos, ori): self.pos, self.ori = pos.clone(), ori.clone()
        def get_position_orientation(self): return self.pos.clone(), self.ori.clone()
        @property
        def aabb(self): return self.pos - torch.tensor([.2, .2, .5]), self.pos + torch.tensor([.2, .2, .5])
    actor = Actor()
    floor = N(name='floor', category='floors', in_rooms=['garden_0'], links={'root': N(prim_path='/floor')},
              aabb=(torch.tensor([-5., -5., -.2]), torch.tensor([5., 5., 0.])))
    seg = N(room_ins_name_to_ins_id={'garden_0': 1}, room_ins_map=torch.ones((9, 9)), map_resolution=1.,
            map_to_world=lambda cells: cells.float() - 4)
    world = N(scene_model=plan['scene'], rooms=plan['rooms'], stable_path=plan['source_scene']['path'],
              model_pool=None, spawned={}, scene=N(seg_map=seg, objects=[floor, actor]),
              robot=N(links={'root': N(prim_path='/robot')}), obj=lambda name: floor)
    if blocked:
        world.scene.objects.append(N(name='blocking_fixture',
            aabb=(torch.tensor([-10., -10., -1.]), torch.tensor([10., 10., 10.]))))
    def spawn(name, category, model, **kwargs):
        calls.append(('spawn', name, category, model, kwargs)); return actor
    world.spawn = spawn; world.activate = lambda obj: calls.append(('activate', obj.name))
    def ray(starts, ends, **kwargs):
        calls.append(('ray',)); return [{'hit': True, 'rigidBody': '/floor' if hit_selected_floor else '/table',
                                        'position': [starts[0][0], starts[0][1], 0.]}]
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.sampling_utils', N(raytest_batch=ray))
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.asset_utils', N(get_dataset_path=lambda dataset: str(selected.asset.parents[4])))
    monkeypatch.setitem(sys.modules, 'ispatialgym.sampling.knowledge', N(is_ready=lambda model: True))
    monkeypatch.setitem(sys.modules, 'ispatialgym.sampling.sim_utils', N(_collision_piece_aabbs=lambda obj: [obj.aabb]))
    if hit_selected_floor and not blocked:
        actual, metadata, evidence = construct_and_place(world, plan)
        assert actual is actor and evidence['candidate_count'] == 1
        assert metadata['applied_scale'] == plan['applied_scale'] and calls[-1][0] == 'activate'
    else:
        with pytest.raises(ValueError, match='declared candidate bound') as error:
            construct_and_place(world, plan)
        assert len(error.value.placement_attempts) == plan['max_candidates']
        assert len([c for c in calls if c[0] == 'ray']) == plan['max_candidates']
        if blocked:
            assert all(row['blockers'] == ['blocking_fixture'] for row in error.value.placement_attempts)
    spawns = [c for c in calls if c[0] == 'spawn']
    assert len(spawns) == 1 and spawns[0][2:4] == ('door', 'model')
    assert spawns[0][4] == {'dataset': 'behavior-1k-assets', 'scale': [1.2] * 3, 'fixed_base': True}


def test_context_footprint_consumes_applied_scale_without_relabeling_native_default():
    from ispatialgym.sampling.scene_variants import _campaign_support_footprints, blocked_footprints
    obj = N(category='bottom_cabinet', model='native', scale=[1.2, 1.2, 1.2],
            aabb=(torch.tensor([-1., -1., 0.]), torch.tensor([1., 1., 2.])))
    scene = N(object_registry=lambda kind, name: obj)
    metadata = {'name': 'selected', 'model_key': 'bottom_cabinet/native',
        'original_default_scale': [1., 1., 1.], 'applied_scale': [1.2, 1.2, 1.2],
        'blocked_footprints': blocked_footprints([([ -1., -1., 0.], [1., 1., 2.])])}
    assert _campaign_support_footprints(scene, metadata) == metadata['blocked_footprints']
    legacy = {**metadata, 'default_scale': metadata['applied_scale']}; legacy.pop('applied_scale')
    assert _campaign_support_footprints(scene, legacy) == metadata['blocked_footprints']
    metadata['applied_scale'] = [1., 1., 1.]
    with pytest.raises(ValueError, match='initialized scale mismatch'):
        _campaign_support_footprints(scene, metadata)


@pytest.mark.parametrize('reservation', ['clear', 'model', 'room', 'changed_pointer', 'bad_scale', 'changed_customs'])
def test_constructor_admission_checks_active_reservations_before_native_work(selected, reservation):
    from scripts.campaign_20261110.reservation_policy import POLICY_KEYS, policy_digest
    case, plan = selected.case, selected.plan
    root = case.tmp / 'campaign'; (root / 'manifests').mkdir(parents=True)
    policy = {key: [] for key in POLICY_KEYS}
    if reservation == 'model':
        policy['model_holdout_models'] = policy['train_exclude_models'] = ['door/model']
    if reservation == 'room':
        policy['category_holdout_rooms'] = [{'scene': plan['scene'], 'room': plan['room']}]
    reservations = document(root / 'reservations.json', policy)
    hashes = document(root / 'hashes.json', {'schema': 'campaign-reserved-asset-hashes-v1',
        'reservation_sha256': reservations['sha256'], 'reserved_model_keys': policy['train_exclude_models'],
        'ready_assets': {}, 'pending_unready_model_keys': policy['train_exclude_models'], 'reserved_asset_hashes': []})
    pointer = document(root / 'manifests/active_reservations.json', {'schema': 'isg-active-reservations-v1',
        'reservations': reservations, 'asset_hashes': hashes, 'policy_sha256': policy_digest(policy)})
    target = ObjectRef('target', 'target', 'door', 'model', plan['asset_sha256'], joint='j_leaf',
                       ready=True, readiness_source='native_catalog')
    template = Template('candidate', 'configuration_matching-articulation_close_large_scale', 'articulation_A',
        'train', plan['scene'], plan['room'], (target,), digest('task'), {'task_rooms': plan['rooms']})
    templates = document(root / 'templates.json', {'templates': [asdict(template)]})
    customs = document(root / 'customs.json', {'alias_A': case.entry})
    jobs = document(root / 'jobs.json', {'jobs': [{'candidate_id': 'candidate', 'class_id': template.class_id,
        'scene': template.scene, 'alias': 'alias_A', 'production_released': False,
        'sample_argv': ['python', '-m', 'ispatialgym.sampling.sample_isg', '--customs', customs['path'],
                        '--scene', template.scene, '--out', 'future_sampling'],
        'training_scale_plan': {'scales': {'target': {'default': [1., 1., 1.],
            'applied': [1.5, 1.5, 1.5] if reservation == 'bad_scale' else plan['applied_scale'], 'mode': 'uniform'}}}}]})
    options = selected.options
    request = document(root / 'request.json', {'schema': 'isg-selected-context-generation-request-v1',
        'campaign_root': str(root), 'reservation_policy': pointer, 'production_released': False,
        'source_jobs': jobs, 'source_templates': templates, 'source_customs': customs, 'candidate_id': 'candidate',
        'metadata': options['metadata_ref'], 'selection': options['selection_ref'], 'configuration_id': 'config_A',
        'source_scene': options['scene_ref'], 'numeric_geometry': options['numeric_ref'],
        'prototype_name': 'prototype', 'floor_name': 'floor', 'placement_room': 'garden_0',
        'yaw_radians': options['yaw_radians'], 'max_candidates': options['max_candidates']})
    if reservation == 'changed_pointer':
        Path(pointer['path']).write_text('{}')
    if reservation == 'changed_customs':
        Path(customs['path']).write_text('{}')
    if reservation == 'clear':
        assert inputs(request)[3] == plan
    else:
        with pytest.raises(ValueError, match='held-out|reservation pointer changed|outside|input changed'):
            inputs(request)


@pytest.mark.parametrize('rooms', [['garden_0'], ['garden_0', 'sauna_0']])
def test_native_generation_accepts_only_the_complete_candidate_room_set(rooms):
    from scripts.campaign_20261110.training_native_canary import verify_native_rooms
    job = {'scene': 'house_single_floor', 'room': 'garden_0'}
    config = {'template': {'scene': job['scene'], 'room': job['room'], 'scene_setup': {'task_rooms': rooms}}}
    problem = N(rooms=lambda scene: list(reversed(rooms)))
    assert set(verify_native_rooms(problem, job, config)) == set(rooms)
    for wrong in ([], rooms + ['garden_0'], rooms + ['unselected_room'], ['sauna_0']):
        with pytest.raises(ValueError, match='room binding'):
            verify_native_rooms(N(rooms=lambda scene: wrong), job, config)
    if len(rooms) == 1:
        config['template'].pop('scene_setup')
        assert verify_native_rooms(problem, job, config) == rooms
