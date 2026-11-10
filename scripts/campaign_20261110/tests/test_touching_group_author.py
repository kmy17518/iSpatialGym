from dataclasses import replace
import time

import pytest

from scripts.campaign_20261110 import contracts as c, goal_groups as g


def test_touching_author_pins_fresh_contact_and_pre_push_place_and_audits_source(tmp_path):
    from scripts.campaign_20261110.tests.test_goal_groups import GoalGroupControllerTests

    job, ledger, _, template, code = GoalGroupControllerTests().fixture(tmp_path)
    anchor = g.successful_anchor(job, campaign_root=tmp_path, ledger=ledger, campaign_started_at=time.time()-10)
    target = template.objects[0]
    partner = c.ObjectRef("box.n.01_1", "partner", "box", "b", c.digest("partner"), ready=True, readiness_source="ready fixture")
    template = replace(template, task="inter_object_relation_satisfaction-touching_small_scale", objects=(target, partner))
    anchor["template_meta"]["bindings"] = [{"name": target.instance, "role": "target"}, {"name": partner.instance, "role": "partner"}]
    bindings = anchor["tro"]["ispatialgym"]["object_bindings"]
    bindings[target.instance]["scene_object_name"] = "apple_a_0"
    bindings[partner.instance] = {"category": partner.category, "model": partner.model}
    anchor["planned_start_poses"][partner.instance] = [[1, 2.2, 3], [0, 0, 0, 1]]
    anchor["summary"]["goal_check"] = {"settled_poses": {"apple_a_0": [[1, 2, 3], [0, 0, 0, 1]]}}
    source = anchor["sources"]["sidecar"]
    c.write_json_atomic(source["path"], {"grasps": [{"object": target.instance,
        "place": {"position": [1.1, 1.9, 3], "orientation": [0, 0, 0, 1]},
        "eef_in_object": {"position": [.1, 0, 0], "orientation": [0, 0, 0, 1]}}]})
    source["sha256"] = c.file_digest(source["path"])
    kwargs = dict(anchor=anchor, template=template, campaign_id="new", code_sha256=c.file_digest(code / ".snapshot_commit"),
                  physics_sha256=c.digest("physics"), data_sha256=c.digest("maps"), magnitude_range=[0, 1])
    g.prepare_group(**kwargs, group_id="touch", output=tmp_path / "groups/touch")
    plan = g.read_json(tmp_path / "groups/touch/goal_plan.json")
    assert plan["goal_pose"]["position"] == [1, 2, 3]
    assert plan["place_pose"]["position"] == pytest.approx([1, 1.9, 3])
    c.write_json_atomic(source["path"], {"grasps": []})
    with pytest.raises(c.ContractError, match="sidecar changed"):
        g.prepare_group(**kwargs, group_id="changed", output=tmp_path / "groups/changed")
