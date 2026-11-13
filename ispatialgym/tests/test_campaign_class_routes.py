import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ispatialgym.sampling import campaign_class_routes as routes
from scripts.campaign_20261110.contracts import ContractError, digest


@pytest.fixture(params=list(routes.recipes.PROFILES))
def routed(request):
    value = json.loads((Path(__file__).parent / 'fixtures/class_routes' / (request.param + '.json')).read_text())
    config, raw = value['config'], value['plan']
    plan = SimpleNamespace(**raw, sha256=digest(raw))
    candidates = [SimpleNamespace(base_pose2d=[float(i), 0., 0.], focus=[float(i), 0., 1.],
                                  translate=.2, tilt=-.3) for i in range(10)]
    trace = []
    assert sorted(routes.rank(config, plan, list(range(10)), candidates, trace=trace)) == list(range(10))
    record = routes.record(config, plan, trace)
    meta = {'runtime': {'class_route': record}, 'validation_key': config['validation_key'],
            'perturbations': config['perturbations'], 'scene_setup': config['template']['scene_setup'],
            'class_id': config['template']['class_id'], 'split': config['split'], 'goal_group_id': raw['group_id']}
    return config, plan, meta


def test_source_round_trip_without_ambient_config_and_matching_policy(routed, monkeypatch):
    config, plan, meta = routed
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG', '/this/must/not/be/read')
    meta = json.loads(json.dumps(meta))
    physical = {o['instance']: SimpleNamespace(category=o['category'], model=o['model'])
                for o in config['template']['objects']}
    source = routes.verify_recorded(meta, phase='oracle', physical=physical)
    replay = routes.verify_recorded(meta, phase='replay', physical=physical)
    assert source['source_record_sha256'] == replay['source_record_sha256']
    assert source['trace'][0]['action'] == 'oracle_native_unchanged'
    assert replay['trace'][0]['action'] == 'replay_recorded_route_verified'
    assert routes.execution_policy(config)['max_attempts'] in (15, 32)


@pytest.mark.parametrize('part', ['configuration', 'plan', 'trace', 'profile'])
def test_mutated_source_is_rejected(routed, part):
    _, _, meta = routed
    record = meta['runtime']['class_route']
    if part == 'trace': record[part].clear()
    else: record[part]['tampered'] = True
    with pytest.raises(ContractError, match='record changed'):
        routes.verify_recorded(meta)


def test_wrong_live_model_or_group_cannot_reuse_source(routed):
    config, _, meta = routed
    physical = {o['instance']: SimpleNamespace(category=o['category'], model='wrong')
                for o in config['template']['objects']}
    with pytest.raises(ContractError, match='model binding'):
        routes.verify_recorded(meta, physical=physical)
    meta['goal_group_id'] += '_other'
    with pytest.raises(ContractError, match='goal group'):
        routes.verify_recorded(meta)


def test_decoded_inputs_are_verified_not_only_declared_fingerprints(routed):
    config, _, _ = routed
    changed = copy.deepcopy(config)
    changed['template']['objects'][0]['model'] = 'another-model'
    with pytest.raises(ContractError, match='template fingerprint'):
        routes.resolve(changed)
    changed = copy.deepcopy(config)
    name = next(iter(changed['perturbations']['scales']))
    changed['perturbations']['scales'][name]['applied'][0] *= .9
    with pytest.raises(ContractError, match='scale'):
        routes.resolve(changed)


def test_required_branch_cannot_be_metadata_only(routed):
    config, plan, _ = routed
    with pytest.raises(ContractError, match='did not execute'):
        routes.record(config, plan, [])


def test_unaffected_and_legacy_sources_retain_existing_path():
    config = {'template': {'class_id': 'configuration_matching-attachment_04'}}
    assert routes.resolve(config) is None
    assert routes.rank(config, None, [0], [], trace=[]) is None
    assert routes.record(config, None, []) is None
    assert routes.execution_policy(config) is None
    assert routes.verify_recorded({'class_id': 'configuration_matching-attachment_03'}) is None
