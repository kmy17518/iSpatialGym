import hashlib
import json
from dataclasses import asdict, replace

import pytest

from scripts.campaign_20261110.bind_fresh_candidates import bind_one, load_contract, finished_geometry_failure
from scripts.campaign_20261110.binding_pools import BindingPool, PoolCandidate, bind_variant, write_pool
from scripts.campaign_20261110.contracts import ObjectRef, Template


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def fixture(tmp_path):
    code = tmp_path / 'code'
    usd = code / 'Behavior-1K/datasets/behavior-1k-assets/objects/apple/omzprq/usd/omzprq.encrypted.usd'
    usd.parent.mkdir(parents=True); usd.write_bytes(b'test exact USD bytes')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    write(code / 'Behavior-1K/datasets/2026-challenge-task-instances/scenes/Rs_int/json/Rs_int_stable.json', {'objects_info': {'init_info': {}}})
    directory = tmp_path / 'fresh'
    bddl = directory / 'activity_definitions/object_scaling/problem1.bddl'
    bddl.parent.mkdir(parents=True); bddl.write_text('(define (problem test))')
    init = {'category': 'apple', 'model': 'omzprq', 'scale': [1., 1., 1.], 'dataset_name': 'behavior-1k-assets'}
    full = write(directory / 'template.json', {'objects_info': {'init_info': {'target_apple': {'args': init}}}})
    meta = {'n_written': 1, 'template_sha256': sha(full), 'bddl_sha256': sha(bddl), 'scene': 'Rs_int', 'task_rooms': ['living_room_0'],
            'variant': 'object_scaling-apple', 'template_id': 'runtime_apple',
            'bindings': [{'name': 'apple.n.01_1', 'role': 'target', 'scene_object_name': 'target_apple'}]}
    path = write(directory / 'template_meta.json', meta)
    write(directory / 'instances/fresh-tro_state.json', {})
    obj = ObjectRef(instance='apple.n.01_1', role='target', category='apple', model='omzprq', asset_sha256=sha(usd),
                    ready=True, readiness_source='test provider evidence')
    template = Template(template_id='candidate_apple', task='object_scaling', class_id='object_scaling-apple', split='train',
                        scene='Rs_int', room='living_room_0', objects=(obj,), definition_sha256='a'*64,
                        scene_setup={'task_rooms': ['living_room_0'], 'runtime_bddl_sha256': sha(bddl)})
    source = write(tmp_path / 'source_jobs.json', {})
    job = {'alias': 'object_scaling-apple', 'source_manifest': str(source)}
    policy = {key: [] for key in ('train_exclude_models', 'category_holdout_rooms', 'category_holdout_categories', 'category_holdout_models', 'model_holdout_models')}
    hashes = {'schema': 'campaign-reserved-asset-hashes-v1', 'reserved_model_keys': [], 'ready_assets': {}, 'pending_unready_model_keys': [], 'reserved_asset_hashes': []}
    pointer = {'policy_sha256': 'b'*64, 'asset_hashes': {'sha256': 'c'*64}}
    return template, job, path, code, policy, hashes, pointer


def test_exact_candidate_binds_and_keeps_release_pending(tmp_path):
    args = fixture(tmp_path)
    result = bind_one(*args)
    assert result.scene_setup['fresh_demo'] == result.scene_setup['fresh_replay'] == 'pending'
    assert result.scene_setup['candidate_fingerprint'] == args[0].fingerprint
    assert result.fingerprint != args[0].fingerprint


@pytest.mark.parametrize('mutation', ['scale', 'asset', 'role_count', 'reserved'])
def test_changed_physical_identity_or_split_is_rejected(tmp_path, mutation):
    args = fixture(tmp_path)
    template, job, metadata, code, policy, hashes, pointer = args
    if mutation == 'asset':
        next(code.rglob('*.encrypted.usd')).write_bytes(b'different USD')
    elif mutation == 'reserved':
        policy['train_exclude_models'] = ['apple/omzprq']
        hashes['reserved_model_keys'] = ['apple/omzprq']
        hashes['ready_assets'] = {'apple/omzprq': {'ready': True, 'asset_sha256': template.objects[0].asset_sha256}}
        hashes['reserved_asset_hashes'] = [template.objects[0].asset_sha256]
    else:
        meta = json.loads(metadata.read_text())
        if mutation == 'role_count':
            meta['bindings'] = []
        else:
            full = metadata.parent / 'template.json'; data = json.loads(full.read_text())
            data['objects_info']['init_info']['target_apple']['args']['scale'] = [2., 2., 2.]
            write(full, data); meta['template_sha256'] = hashlib.sha256(full.read_bytes()).hexdigest()
        write(metadata, meta)
    with pytest.raises(ValueError):
        bind_one(*args)


def test_candidate_fingerprint_cannot_be_relabelled(tmp_path):
    template = fixture(tmp_path)[0]
    row = {**asdict(template), 'fingerprint': template.fingerprint}
    row['class_id'] = 'object_scaling-bottle_of_pop'
    with pytest.raises(ValueError):
        load_contract(row)


def test_finished_native_failure_is_distinct_from_pending(tmp_path):
    template, job, *_ = fixture(tmp_path)
    runs = tmp_path / 'runs'
    job = {**job, 'sample_argv': ['python', '--out', '${CAMPAIGN_OUT}/validation/apple']}
    assert finished_geometry_failure(template, job, runs) is None
    path = runs / 'validation/apple/summary.json'
    report = {'args': {'variant': job['alias'], 'scene': template.scene}, 'n_written': 0,
        'problems': {job['alias']: {'native': {'rejections': {'missing_grasps:floating': 1}}}}}
    write(path, report)
    assert finished_geometry_failure(template, job, runs) is None
    write(path, {**report, 'finished': '2026-11-10T10:00:00'})
    result = finished_geometry_failure(template, job, runs)
    assert result['rejections'] == {'missing_grasps:floating': 1}
    assert result['summary_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    write(path, {**report, 'finished': '2026-11-10T10:00:00', 'n_written': 1})
    assert finished_geometry_failure(template, job, runs) is None


def test_geometry_summary_cannot_escape_or_change_identity(tmp_path):
    template, job, *_ = fixture(tmp_path)
    runs = tmp_path / 'runs'
    job = {**job, 'sample_argv': ['python', '--out', '${CAMPAIGN_OUT}/../other']}
    with pytest.raises(ValueError, match='outside'):
        finished_geometry_failure(template, job, runs)
    job['sample_argv'][-1] = '${CAMPAIGN_OUT}/validation/apple'
    write(runs / 'validation/apple/summary.json', {'args': {'variant': 'other', 'scene': template.scene}, 'finished': 'done'})
    with pytest.raises(ValueError, match='alias'):
        finished_geometry_failure(template, job, runs)


def test_fresh_binding_keeps_immutable_pool_and_rejects_renamed_active_binding(tmp_path):
    template, job, metadata, code, policy, hashes, pointer = fixture(tmp_path)
    template = replace(template, task='disembedding', class_id='disembedding_02')
    target = template.objects[0]
    inactive = replace(target, instance='choice', role='distractor', category='pear', model='test', asset_sha256='f'*64)
    pool = BindingPool('pool02', template.class_id, template.split, template.scene, template.room, (),
        (PoolCandidate('target', target), PoolCandidate('inactive', inactive)),
        {'target': (1, 1), 'distractor': (0, 1)}, template.definition_sha256)
    path = write_pool(tmp_path / 'pool.json', pool)
    template = bind_variant(template, path, {target.instance: 'target'})
    bound = bind_one(template, job, metadata, code, policy, hashes, pointer)
    assert bound.scene_setup['binding_pool'] == template.scene_setup['binding_pool']
    meta = json.loads(metadata.read_text()); meta['bindings'][0]['name'] = 'apple.n.01_2'
    write(metadata, meta)
    with pytest.raises(ValueError, match='active bindings'):
        bind_one(template, job, metadata, code, policy, hashes, pointer)


def bddl_only_relocation_fixture(tmp_path):
    import shutil
    args = fixture(tmp_path)
    metadata = args[2]
    customs = write(tmp_path / 'customs.json', {'object_scaling-apple': {}})
    meta = json.loads(metadata.read_text())
    meta.update(customs=str(customs), activity_definition='activity_definitions/object_scaling/problem1.bddl')
    write(metadata, meta)
    original = tmp_path / 'original'
    shutil.copytree(metadata.parent, original)
    relocated = metadata.parent / 'relocated_definition/problem1.bddl'
    relocated.parent.mkdir()
    shutil.copyfile(original / meta['activity_definition'], relocated)
    meta['activity_definition'] = str(relocated.relative_to(metadata.parent))
    write(metadata, meta)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    proof = {'schema': 'isg-native-source-relocation-v1', 'changed_fields': ['activity_definition'],
             'original_metadata': str(original / 'template_meta.json'),
             'original_metadata_sha256': sha(original / 'template_meta.json'),
             'metadata_sha256': sha(metadata), 'original_customs': str(customs), 'customs': str(customs),
             'customs_sha256': sha(customs), 'activity_definition': meta['activity_definition'],
             'native_bddl_sha256': meta['bddl_sha256'],
             'copied_files': [{'relative_path': str(p.relative_to(original)), 'sha256': sha(p)}
                              for p in original.rglob('*') if p.is_file()]}
    path = write(metadata.parent / 'native_source_relocation.json', proof)
    return args, path


def test_exact_bddl_only_relocation_keeps_unchanged_receipt(tmp_path):
    args, proof = bddl_only_relocation_fixture(tmp_path)
    before = proof.read_bytes()
    result = bind_one(*args)
    assert result.scene_setup['fresh_demo'] == 'pending'
    assert proof.read_bytes() == before


@pytest.mark.parametrize('mutation', ['customs_locator', 'definition_bytes', 'unlisted_metadata'])
def test_bddl_only_relocation_rejects_unlisted_or_changed_sources(tmp_path, mutation):
    args, path = bddl_only_relocation_fixture(tmp_path)
    proof = json.loads(path.read_text())
    metadata = args[2]
    meta = json.loads(metadata.read_text())
    if mutation == 'customs_locator':
        replacement = write(tmp_path / 'other_customs.json', {'object_scaling-apple': {}})
        proof['customs'] = meta['customs'] = str(replacement)
    elif mutation == 'definition_bytes':
        (metadata.parent / proof['activity_definition']).write_text('changed physical definition')
    else:
        meta['scene'] = 'different_scene'
    write(metadata, meta)
    proof['metadata_sha256'] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    write(path, proof)
    with pytest.raises(ValueError):
        bind_one(*args)
