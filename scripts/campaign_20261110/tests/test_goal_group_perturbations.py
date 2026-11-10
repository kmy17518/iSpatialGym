from dataclasses import asdict, replace
import copy

import pytest

from scripts.campaign_20261110 import contracts as c, goal_groups as g


def fixture():
    obj = c.ObjectRef("target", "target", "apple", "ready_model", c.digest("usd"), ready=True, readiness_source="catalog")
    template = c.Template("template", "object_scaling", "apple", "train", "house_single_floor", "living_room_0",
                          (obj,), c.digest("definition"))
    perturb = c.Perturbations({"target": c.ScaleState((1, 1, 1), (.965, .965, .965), "uniform")},
                             ("chair_unused_0",), {"schema": "isg-lighting-v1", "intensity_factor": .9, "skybox_color_factors": [1, 1, 1]})
    block = {"campaign_id": "new", "task": template.task, "class_id": template.class_id, "split": "train",
             "perturbations": asdict(perturb), "runtime": {"observed_scales": {"target": [.965, .965, .965]}},
             "validation_key": {"scale_sha256": perturb.scale_fingerprint}}
    return template, perturb, {"tro": {"ispatialgym": {"campaign": block}}}


def test_train_group_retains_exact_verified_scale_scene_and_lighting():
    template, perturb, anchor = fixture()
    assert g.anchor_perturbations(anchor, template, "new") == perturb
    with pytest.raises(c.ContractError, match="no perturbations"):
        g.anchor_perturbations(anchor, replace(template, split="eval_standard"), "new")
    default = g.anchor_perturbations({"tro": {"ispatialgym": {}}}, template, "new")
    assert not default.perturbed and default.scales["target"].applied == (1, 1, 1)


def test_scale_carryover_requires_matching_live_measurement_and_key():
    template, _, anchor = fixture()
    for field, value, message in (("runtime", {"observed_scales": {"target": [1, 1, 1]}}, "live verification"),
                                  ("validation_key", {"scale_sha256": c.digest("wrong")}, "scale key"),
                                  ("class_id", "other", "another campaign/task/class")):
        changed = copy.deepcopy(anchor)
        changed["tro"]["ispatialgym"]["campaign"][field] = value
        with pytest.raises(c.ContractError, match=message):
            g.anchor_perturbations(changed, template, "new")
    wrong_default = replace(template, objects=(replace(template.objects[0], default_scale=(2, 2, 2)),))
    with pytest.raises(c.ContractError, match="default scale"):
        g.anchor_perturbations(anchor, wrong_default, "new")


def test_scale_changes_rebind_only_the_native_choice_suffix(tmp_path):
    from ispatialgym.sampling.spec import ModelChoice
    from ispatialgym.sampling.v2.problem import TemplateChoice

    template, perturb, anchor = fixture()
    source = TemplateChoice(template.scene, {'target': ModelChoice('apple', 'ready_model')}, rooms=[template.room])
    current = copy.deepcopy(source)
    current.models['target'].scale = [.965]*3
    g.write_json_atomic(tmp_path / 'template_meta.json', {'choice': source.to_json()})
    template = replace(template, scene_setup={'runtime_template_id': source.template_id, 'source_template': str(tmp_path)})
    anchor['template_meta'] = {'template_id': current.template_id, 'choice': current.to_json()}
    verified = g.anchor_perturbations(anchor, template, 'new')
    bound = g.bind_scaled_runtime_identity(anchor, template, verified)
    assert bound.scene_setup['runtime_template_id'] == current.template_id
    assert bound.fingerprint != template.fingerprint
    assert bound.scene_setup['scaled_runtime_identity']['scale_sha256'] == perturb.scale_fingerprint
    assert g.bind_scaled_runtime_identity(anchor, bound, verified) == bound
    # Neither a different room nor an extra grasp-key change is explained by scaling.
    for field, value in [('rooms', ['another_room']), ('support_tag', 'another_support')]:
        changed = copy.deepcopy(anchor)
        changed['template_meta']['choice'][field] = value
        choice = TemplateChoice.from_json(changed['template_meta']['choice'])
        changed['template_meta']['template_id'] = choice.template_id
        with pytest.raises(c.ContractError, match='beyond verified'):
            g.bind_scaled_runtime_identity(changed, template, verified)
    changed = copy.deepcopy(anchor)
    changed['template_meta']['choice']['models']['target']['grasp_key'] = 'different'
    with pytest.raises(c.ContractError, match='library root'):
        g.bind_scaled_runtime_identity(changed, template, verified)
    default = c.Perturbations({'target': c.ScaleState((1, 1, 1), (1, 1, 1), 'fixed')})
    with pytest.raises(c.ContractError, match='without verified training scale'):
        g.bind_scaled_runtime_identity(anchor, template, default)


def test_scaled_alias_change_requires_native_model_scale_and_positive_library(tmp_path):
    from pathlib import Path
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling import grasp_library as G

    source = {'category': 'apple', 'model': 'ready_model', 'grasp_key': 'apple_default'}
    current = {**source, 'grasp_key': 'apple_scaled'}
    state = c.ScaleState((1, 1, 1), (.965, .965, .965))
    paths = []
    for row, scale in ((source, state.default), (current, state.applied)):
        path = Path(S.model_dir('apple', 'ready_model', scale, str(tmp_path))) / 'library.json'
        data = {'schema': S.LIBRARY, 'category': 'apple', 'model': 'ready_model', 'dataset': 'behavior-1k-assets',
                'legacy_key': row['grasp_key'], 'scale': list(scale), 'constraint': 'tabletop',
                'rests': {'r0': {'schema': S.REST, 'rest_id': 'r0', 'quat_xyzw': [0, 0, 0, 1], 'h_root': .03, 'stable': True}},
                'grasps': [{'name': 'top_0', 'rest_id': 'r0', 'T_object_eef': {'position': [0, 0, .05], 'orientation_xyzw': [1, 0, 0, 0]}}]}
        g.write_json_atomic(path, data)
        paths.append(path)
    anchor = {'grasp_library_root': str(tmp_path)}
    proof = g.verify_scaled_grasp_route(anchor, source, current, state)
    assert proof['scaled']['sha256'] == c.file_digest(paths[1])
    with pytest.raises(S.IncompatibleRecord, match='scale'):
        g.verify_scaled_grasp_route(anchor, source, current, c.ScaleState((1, 1, 1), (.9, .9, .9)))
    data = g.read_json(paths[1])
    data['grasps'] = []
    g.write_json_atomic(paths[1], data)
    G._v2_index.cache_clear()
    with pytest.raises(c.ContractError, match='no positive native grasps'):
        g.verify_scaled_grasp_route(anchor, source, current, state)
