import math
import time

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
