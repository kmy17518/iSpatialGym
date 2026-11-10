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
