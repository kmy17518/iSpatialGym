import math
import time
from dataclasses import replace

import pytest

from scripts.campaign_20261110 import held_goal_batches as b, goal_groups as g, contracts as c
from scripts.campaign_20261110.tests import test_goal_groups as controller_tests


def anchor_fixture(tmp_path):
    job, ledger, _, template, code = controller_tests.GoalGroupControllerTests().fixture(tmp_path)
    anchor = g.successful_anchor(job,campaign_root=tmp_path,ledger=ledger,campaign_started_at=time.time()-10)
    anchor["summary"].update(goal={"goal_depth_m":.4},trunk=[.5,0])
    anchor["template_meta"].update(hold={"camera_in_base":{"orientation":[0,0,0,1]},"trunk":[.5,0]},
                                    template_facts={"depth_windows":{"g":{"window":[.2,.6]}}})
    return anchor, template, code


def test_depth_proposals_precede_start_draws_and_stay_in_native_windows(tmp_path):
    anchor, _, _ = anchor_fixture(tmp_path)
    proposals=b.depth_goal_proposals(anchor,6,magnitude_range=[.05,.1],seed=7)
    depths=[p["depth_m"] for p in proposals]
    assert all(.205 <= value <= .595 for value in depths)
    assert all(abs(a-d)>=.005 for i,a in enumerate(depths) for d in depths[i+1:])
    assert all(abs(value-.4)>=.005 for value in depths)
    yaw=anchor["summary"]["base_pose2d"][2]
    assert math.isfinite(yaw)
    for p in proposals:
        assert p["goal_pose"]["position"][2] == pytest.approx(3-(p["depth_m"]-.4))
        assert p["goal_pose"]["orientation"] == [0,0,0,1]
    assert b.depth_goal_proposals(anchor,6,magnitude_range=[.05,.1],seed=7) == proposals


def test_batch_has_independent_group_files_and_one_immutable_config(tmp_path):
    anchor,template,code=anchor_fixture(tmp_path)
    result=b.prepare_batch(anchor=anchor,template=template,campaign_root=tmp_path,batch_id="batch",count=2,
        magnitude_range=[.05,.1],seed=7,campaign_id="new",code_sha256=c.file_digest(code/".snapshot_commit"),
        physics_sha256=c.digest("physics"),data_sha256=c.digest("data"))
    assert result["candidate_instances"] == 6
    assert result["group_ids"] == ["batch_0000","batch_0001"]
    for group in result["group_ids"]:
        assert (tmp_path/"groups"/group/"goal_proposal.json").exists()
        assert (tmp_path/"groups"/group/"campaign_config.json").exists()
    command=b.submission_command(result["queue"],snapshot=code,campaign_root=tmp_path,v2_root=tmp_path/"maps")
    assert command[command.index("submit")+2] == "6"
    assert not any(value.startswith("ISPATIALGYM_CAMPAIGN_GOAL_PLAN=") for value in command)
    assert any(value.startswith(b.ENV+"=") for value in command)


def test_rotation_goal_region_is_native_normalized_and_precedes_fresh_starts(tmp_path):
    anchor,_,_=anchor_fixture(tmp_path)
    anchor["summary"]["goal"] = {"axis_world":[0,0,1],"angle_deg":60}
    anchor["template_meta"]["init_orientation"] = {"goal_angle_range_deg":[30,120]}
    proposals=b.rotation_goal_proposals(anchor,20,seed=5)
    assert len(proposals)==20 and proposals==b.rotation_goal_proposals(anchor,20,seed=5)
    for row in proposals:
        assert 30 <= row["goal_angle_deg"] <= 120
        assert sum(v*v for v in row["goal_pose"]["orientation"]) == pytest.approx(1)
        reconstructed=b.quaternion_product(b.axis_angle(row["axis_world"],row["goal_angle_deg"]),row["reference_orientation"])
        assert reconstructed==pytest.approx(row["goal_pose"]["orientation"])


def test_mental_batch_uses_same_validated_plan_consumer_and_separate_fresh_groups(tmp_path):
    anchor,template,code=anchor_fixture(tmp_path)
    template=replace(template,task="mental_rotation",class_id="mental_rotation-mug")
    anchor["summary"]["goal"] = {"axis_world":[0,0,1],"angle_deg":60}
    anchor["template_meta"]["init_orientation"] = {"goal_angle_range_deg":[30,120]}
    result=b.prepare_batch(anchor=anchor,template=template,campaign_root=tmp_path,batch_id="mental",count=2,
        magnitude_range=[30,45],seed=7,campaign_id="new",code_sha256=c.file_digest(code/".snapshot_commit"),
        physics_sha256=c.digest("physics"),data_sha256=c.digest("data"))
    state=b.read_queue(result["queue"],result["config"])
    assert len(state["entries"])==2
    plans=[g.read_json(row["path"]) for row in state["entries"]]
    assert all(row["task"]=="mental_rotation" and row["magnitude_range"]==[30,45] for row in plans)
    assert plans[0]["goal_pose"] != plans[1]["goal_pose"]


def test_base_transform_preserves_relative_goal_and_world_quaternion():
    proposal={"goal_pose":{"position":[2,3,1.2],"orientation":[0,0,0,1]}}
    moved=b.move_goal_with_base(proposal,[1,3,0],[10,20,math.pi/2])
    assert moved["goal_pose"]["position"]==pytest.approx([10,21,1.2])
    assert moved["goal_pose"]["orientation"]==pytest.approx([0,0,math.sqrt(.5),math.sqrt(.5)])
    assert moved["base_pose2d"]==[10,20,math.pi/2]


def test_map_regions_produce_distinct_room_bases_and_reject_forged_cells(tmp_path):
    import cv2
    import numpy as np
    from scripts.campaign_20261110.base_regions import room_base_region
    anchor,_,_=anchor_fixture(tmp_path)
    scene=anchor["template_meta"]["scene"]
    room=anchor["tro"]["ispatialgym"]["scene"]["task_rooms"][0]
    category=room.rsplit("_",1)[0]
    assets=tmp_path/"data/behavior-1k-assets"
    layout=assets/"scenes"/scene/"layout";layout.mkdir(parents=True)
    (assets/"metadata").mkdir();(assets/"metadata/room_categories.txt").write_text(category+"\n")
    for name,value in [("floor_insseg_0.png",1),("floor_semseg_0.png",1),("floor_trav_0.png",255)]:
        cv2.imwrite(str(layout/name),np.full((200,200),value,dtype=np.uint8))
    region=room_base_region(tmp_path/"data",scene,room)
    proposals=b.depth_goal_proposals(anchor,6,magnitude_range=[.05,.1],seed=7)
    varied=b.apply_base_region(proposals,anchor,region,seed=5)
    assert len({tuple(row["base_pose2d"][:2]) for row in varied})==6
    assert all(row["base_pose2d"][:2] in region["free_xy"] for row in varied)
    region["free_xy"][0]=[999,999]
    with pytest.raises(c.ContractError,match="cells differ"):
        b.apply_base_region(proposals,anchor,region,seed=5)
