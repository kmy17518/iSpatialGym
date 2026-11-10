from dataclasses import replace
import time

from scripts.campaign_20261110 import contracts as c, goal_groups as g


def test_block_author_keeps_slot_order_and_explicit_ready_model_identity(tmp_path):
    from scripts.campaign_20261110.tests.test_goal_groups import GoalGroupControllerTests

    job, ledger, _, template, code = GoalGroupControllerTests().fixture(tmp_path)
    anchor = g.successful_anchor(job, campaign_root=tmp_path, ledger=ledger, campaign_started_at=time.time()-10)
    targets = tuple(replace(template.objects[0], instance=f"block{i}") for i in range(4))
    board = c.ObjectRef("board", "partner", "tray", "board", c.digest("board"),
                       ready=True, readiness_source="ready fixture", semantic_kind="board")
    template = replace(template, task="configuration_matching-block_design", objects=targets+(board,))
    anchor["tro"]["ispatialgym"]["object_bindings"] = {obj.instance: {"category": obj.category, "model": obj.model} for obj in template.objects}
    anchor["tro"]["ispatialgym"]["groups"] = {"blocks": {"members": [obj.instance for obj in targets],
        "target_slots": {f"slot_{i}": {"position": [i, 0, 1], "orientation": [0, 0, 0, 1]} for i in range(4)}}}
    anchor["planned_start_poses"] = {obj.instance: [[i, 2, 1], [0, 0, 0, 1]] for i, obj in enumerate(template.objects)}
    order = [obj.instance for obj in reversed(targets)]
    anchor["summary"].update(pattern="rgbw", pattern_faces=["red", "green", "blue", "white"], placement_order=order,
        up_faces=dict(zip(order, ["+x", "-x", "+y", "-y"])), flip_directions=dict(zip(order, ["right", "right", None, None])))
    output = tmp_path / "groups/blocks"
    g.prepare_group(anchor=anchor, template=template, campaign_id="new", group_id="blocks", output=output,
                    code_sha256=c.file_digest(code / ".snapshot_commit"), physics_sha256=c.digest("physics"),
                    data_sha256=c.digest("maps"), magnitude_range=[1, 4])
    plan = g.read_json(output / "goal_plan.json")
    assert plan["schema"] == "isg-block-goal-plan-v1"
    assert plan["goal_slots"][2]["position"] == [2, 0, 1]
    assert plan["board_pose"]["position"] == [4, 2, 1]
    assert plan["initial_up_faces"] == ["+x", "-x", "+y", "-y"]
    assert "base_pose2d" not in plan and not plan["source_instance_is_member"]
    # A renamed different model cannot enter this four-identical-block lane.
    template = replace(template, objects=(replace(targets[0], model="other"),)+targets[1:]+(board,))
    anchor["tro"]["ispatialgym"]["object_bindings"][targets[0].instance]["model"] = "other"
    import pytest
    with pytest.raises(c.ContractError, match="identical models"):
        g.prepare_group(anchor=anchor, template=template, campaign_id="new", group_id="bad", output=tmp_path / "bad",
                        code_sha256=c.digest("code"), physics_sha256=c.digest("physics"), data_sha256=c.digest("maps"), magnitude_range=[1, 4])
