"""train_r6 (2026-11-15): the TRAIN clone of an exact canonical eval template keeps its audited recovery scopes.

OnlineGroups gives each group configuration the config split, so a canonical eval template authored as a train source
(train_gen author_canonical) runs its groups with template.split == 'train'. The fixtures are those real group
configurations (train_online_r1, one per class, the failing r4 sources). Each scope still refuses any other template.
"""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ispatialgym.sampling import campaign_runtime as runtime
from ispatialgym.sampling.campaign_held_goals import GoalPlanError
from ispatialgym.sampling.campaign_nextto_contact_proposal import checked_hint
from ispatialgym.sampling.campaign_tray_goals import reference_start_proposal
from scripts.campaign_20261110.contracts import ContractError, ObjectRef, Perturbations, ScaleState, Template

DATA = json.loads((Path(__file__).parent / 'data/train_r6_canonical/group_configs.json').read_text())

# option -> class whose canonical template carries it (contact_recovery scopes changed in train_r6)
OPTIONS = {
    'cap_capture_margin': 'configuration_matching-attachment_02',
    'cap_shallow_start_proposal': 'configuration_matching-attachment_02',
    'axial_magazine_contact_inward_1cm': 'alignment-axial_04',
    'axial_magazine_radial_escape': 'alignment-axial_04',
    'inside_goal_grasp_proposal': 'inter_object_relation_satisfaction-inside_01',
    'cereal_brace_gap_1mm': 'inter_object_relation_satisfaction-touching_02',
}


def load(class_id):
    config = DATA[class_id]
    data = dict(config['template'])
    data['objects'] = tuple(ObjectRef(**o) for o in data['objects'])
    template = Template(**data)
    perturbations = Perturbations({k: ScaleState(**v) for k, v in config['perturbations']['scales'].items()},
                                  tuple(config['perturbations'].get('removed_objects', [])),
                                  config['perturbations'].get('lighting', {}))
    return config, template, perturbations


def active(monkeypatch, template, perturbations, split='train'):
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG', '/nonexistent/r6_test_config.json')
    monkeypatch.setattr(runtime, 'configuration', lambda: ({'split': split}, template, perturbations, None))


def scaled(perturbations, factor):
    name = next(iter(perturbations.scales))
    s = perturbations.scales[name]
    return Perturbations({**perturbations.scales, name: ScaleState(s.default, tuple(v * factor for v in s.applied), 'uniform')})


def changed_model(template, role='target'):
    return replace(template, objects=tuple(replace(o, model='zzzzzz') if o.role == role else o for o in template.objects))


@pytest.mark.parametrize('class_id', sorted(runtime.CANONICAL_TRAIN_CLONES))
def test_fixture_is_the_failing_train_group_and_table_matches_it(class_id):
    config, template, perturbations = load(class_id)
    assert config['split'] == template.split == 'train'
    assert config['validation_key']['template_sha256'] == template.fingerprint
    assert config['validation_key']['scale_sha256'] == perturbations.scale_fingerprint
    assert runtime.CANONICAL_TRAIN_CLONES[class_id] == (replace(template, split='eval_standard').fingerprint,
                                                         perturbations.scale_fingerprint)
    assert runtime.canonical_scope(template, perturbations)
    assert runtime.canonical_scope(replace(template, split='eval_standard'), perturbations)


@pytest.mark.parametrize('class_id', sorted(runtime.CANONICAL_TRAIN_CLONES))
def test_scope_refuses_every_other_template(class_id, monkeypatch):
    _, template, perturbations = load(class_id)
    assert not runtime.canonical_scope(replace(template, room=template.room + '_x'), perturbations)
    assert not runtime.canonical_scope(replace(template, scene_setup={**template.scene_setup, 'extra': 1}), perturbations)
    assert not runtime.canonical_scope(changed_model(template, 'support'), perturbations)
    assert not runtime.canonical_scope(template, scaled(perturbations, 1.1))
    for split in ('eval_model', 'eval_category'):
        assert not runtime.canonical_scope(replace(template, split=split), perturbations)
    other = next(c for c in runtime.CANONICAL_TRAIN_CLONES if c != class_id)
    assert not runtime.canonical_scope(replace(template, class_id=other), perturbations)
    # no perturbations given and no active configuration: fail closed
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    assert not runtime.canonical_scope(template)


@pytest.mark.parametrize('option', sorted(OPTIONS))
def test_contact_recovery_accepts_train_clone(option, monkeypatch):
    _, original, perturbations = load(OPTIONS[option])
    assert original.scene_setup['recovery_options'].get(option) is True   # carried by the canonical template itself
    active(monkeypatch, original, perturbations)
    assert runtime.contact_recovery(option) is True
    active(monkeypatch, replace(original, split='eval_standard'), perturbations, 'eval_standard')
    assert runtime.contact_recovery(option) is True


@pytest.mark.parametrize('option', sorted(OPTIONS))
def test_contact_recovery_still_refuses_changed_train_template(option, monkeypatch):
    _, template, perturbations = load(OPTIONS[option])
    for bad, pert in ((changed_model(template), perturbations), (template, scaled(perturbations, 1.2)),
                      (replace(template, room=template.room + '_x'), perturbations)):
        active(monkeypatch, bad, pert)
        with pytest.raises(ContractError):
            runtime.contact_recovery(option)


@pytest.mark.parametrize('class_id', ['disembedding_01', 'disembedding_10'])
def test_reference_tray_proposal_train_clone(class_id, monkeypatch):
    _, template, perturbations = load(class_id)
    assert template.scene_setup.get('reference_start_proposal') is not None
    active(monkeypatch, template, perturbations)
    assert reference_start_proposal(template) is not None
    for bad, pert in ((changed_model(template), perturbations), (template, scaled(perturbations, 1.2))):
        active(monkeypatch, bad, pert)
        with pytest.raises(GoalPlanError, match='wrong scope'):
            reference_start_proposal(bad)


def test_nextto_contact_proposal_train_clone(monkeypatch):
    _, template, perturbations = load('inter_object_relation_satisfaction-nextto_large_scale_01')
    assert template.scene_setup.get('nextto_contact_source_proposal') is not None
    active(monkeypatch, template, perturbations)
    assert checked_hint(template) is not None
    for bad, pert in ((changed_model(template, 'partner'), perturbations), (template, scaled(perturbations, 1.2))):
        active(monkeypatch, bad, pert)
        with pytest.raises(GoalPlanError, match='wrong class/task/split'):
            checked_hint(bad)
