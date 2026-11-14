"""train_r4 (2026-11-14): a non-canonical TRAIN sub-template of attachment_03 / under_small_scale_03 binds the retained recipe by
class + task + data/physics fingerprints; its goal plan must belong to its own template. Eval stays exact-fingerprint only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ispatialgym.sampling import campaign_class_recipes as R
from ispatialgym.sampling import campaign_class_routes as routes
from scripts.campaign_20261110.contracts import ObjectRef, Template, digest


def variant(class_id, split='train'):
    """The canonical fixture config moved to another room (a different template fingerprint), split ``split``."""
    value = json.loads((Path(__file__).parent / 'fixtures/class_routes' / (class_id + '.json')).read_text())
    config, raw = copy.deepcopy(value['config']), copy.deepcopy(value['plan'])
    config['split'] = split
    config['template']['room'] = config['template']['room'] + '_train_variant'
    data = dict(config['template']); data['objects'] = tuple(ObjectRef(**o) for o in data['objects'])
    config['validation_key']['template_sha256'] = Template(**data).fingerprint
    raw['contract_template_sha256'] = config['validation_key']['template_sha256']
    return config, raw


def candidates():
    return [SimpleNamespace(base_pose2d=[float(i), 0., 0.], focus=[float(i), 0., 1.], translate=.2, tilt=-.3) for i in range(10)]


@pytest.mark.parametrize('class_id', list(R.PROFILES))
def test_train_variant_binds_recipe_and_round_trips(class_id):
    config, raw = variant(class_id)
    profile = R.profile_for_config(config)
    base = R.PROFILES[class_id]
    assert profile['route_id'] == base['route_id'] and profile['historical_max_draws'] == base['historical_max_draws']
    assert profile['train_variant'] == {'binding': R.TRAIN_VARIANT_BINDING, 'split': 'train',
                                        'template_sha256': config['validation_key']['template_sha256'],
                                        'scale_sha256': config['validation_key']['scale_sha256'],
                                        'rule': profile['train_variant']['rule']}
    assert {k: v for k, v in profile.items() if k != 'train_variant'} == json.loads(json.dumps(base))
    plan = SimpleNamespace(**raw, sha256=digest(raw))
    trace = []
    order = routes.rank(config, plan, list(range(10)), candidates(), trace=trace)
    from ispatialgym.sampling.campaign_tabletop_starts import prioritize_written_stances
    assert order == prioritize_written_stances(list(range(10)), candidates(), [raw['anchor_manipulation_stance']], explore=4, exploit=2)
    assert trace[0]['train_variant_binding'] == R.TRAIN_VARIANT_BINDING
    record = routes.record(config, plan, trace)
    meta = {'runtime': {'class_route': record}, 'validation_key': config['validation_key'],
            'perturbations': config['perturbations'], 'scene_setup': config['template']['scene_setup'],
            'class_id': config['template']['class_id'], 'split': config['split'], 'goal_group_id': raw['group_id']}
    assert routes.verify_recorded(json.loads(json.dumps(meta)), phase='replay')['route_id'] == base['route_id']
    assert routes.execution_policy(config)['max_attempts'] == base['historical_max_draws']


@pytest.mark.parametrize('class_id', list(R.PROFILES))
def test_canonical_profile_unchanged(class_id):
    value = json.loads((Path(__file__).parent / 'fixtures/class_routes' / (class_id + '.json')).read_text())
    for split in ('eval_standard', 'train'):
        config = dict(value['config'], split=split)
        assert R.profile_for_config(config) == json.loads(json.dumps(R.PROFILES[class_id]))


@pytest.mark.parametrize('class_id', list(R.PROFILES))
def test_eval_variant_and_foreign_physics_still_fail_closed(class_id):
    config, raw = variant(class_id, split='eval_standard')
    with pytest.raises(ValueError):
        R.profile_for_config(config)
    for field in ('data_sha256', 'physics_sha256'):
        config, raw = variant(class_id)
        config['validation_key'][field] = 'b' * 64
        with pytest.raises(ValueError):
            R.profile_for_config(config)
    config, raw = variant(class_id)
    config['template']['task'] = 'other_task'
    with pytest.raises(ValueError):
        R.profile_for_config(config)


@pytest.mark.parametrize('class_id', list(R.PROFILES))
def test_train_variant_plan_must_belong_to_its_own_template(class_id):
    config, raw = variant(class_id)
    raw['contract_template_sha256'] = R.PROFILES[class_id]['template_sha256']   # the canonical template's plan
    plan = SimpleNamespace(**raw, sha256=digest(raw))
    with pytest.raises(ValueError, match='exact reviewed'):
        R.validate_recipe_inputs(R.PROFILES[class_id]['route_id'], config, plan, phase='sampling', trace=[])
