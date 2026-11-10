from dataclasses import replace
import time

from scripts.campaign_20261110 import contracts as c, goal_groups as g


def test_attachment_author_pins_partner_and_mating_goal_without_copying_anchor(tmp_path):
    from scripts.campaign_20261110.tests.test_goal_groups import GoalGroupControllerTests

    job, ledger, _, template, code = GoalGroupControllerTests().fixture(tmp_path)
    anchor = g.successful_anchor(job, campaign_root=tmp_path, ledger=ledger, campaign_started_at=time.time()-10)
    partner = c.ObjectRef("bottle.n.01_1", "partner", "bottle", "b", c.digest("partner"),
                          ready=True, readiness_source="ready fixture")
    template = replace(template, task="configuration_matching-attachment", objects=template.objects+(partner,))
    anchor["tro"]["ispatialgym"]["object_bindings"][partner.instance] = {"category": partner.category, "model": partner.model}
    anchor["planned_start_poses"][partner.instance] = [[1, 2, 2.8], [0, 0, 0, 1]]
    output = tmp_path / "groups/attachment"
    g.prepare_group(anchor=anchor, template=template, campaign_id="new", group_id="attachment", output=output,
                    code_sha256=c.file_digest(code / ".snapshot_commit"), physics_sha256=c.digest("physics"),
                    data_sha256=c.digest("maps"), magnitude_range=[0, 1])
    plan = g.read_json(output / "goal_plan.json")
    assert plan["schema"] == "isg-attachment-goal-plan-v1"
    assert plan["partner_pose"]["position"] == [1, 2, 2.8]
    assert plan["goal_pose"]["position"] == [1, 2, 3]
    assert plan["target_name"] == template.objects[0].instance
    assert plan["partner_name"] == partner.instance
    assert "base_pose2d" not in plan and "magnitude_range" not in plan
    assert not plan["source_instance_is_member"]
    assert not list(output.glob("*-tro_state.json"))
