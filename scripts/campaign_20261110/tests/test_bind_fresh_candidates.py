import hashlib
import json
from dataclasses import asdict

import pytest

from scripts.campaign_20261110.bind_fresh_candidates import bind_one, load_contract
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
