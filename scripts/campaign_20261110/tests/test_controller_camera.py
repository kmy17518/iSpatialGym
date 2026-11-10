from types import SimpleNamespace

import pytest

from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.controller_camera import (
    allocation_for, authority, production_command, remaining_capacity, refresh_production,
)
from scripts.campaign_20261110 import controller_camera as controller
from scripts.campaign_20261110.contracts import write_json_atomic


def test_heldout_camera_overflow_cannot_turn_into_training():
    template = SimpleNamespace(split="eval_model", class_id="chair", scene="scene", room="room")
    accepted = {str(i): {"split": "eval_model", "class_id": "chair", "bucket": bucket}
                for i, bucket in enumerate(["3"] * 3 + ["5"] * 3 + ["10"] * 2)}
    assert remaining_capacity(template, accepted, {"scene/room": 100},
                              {"eval_model": {"3": 3, "5": 3, "10": 3}}) == 1
    accepted["last"] = {"split": "eval_model", "class_id": "chair", "bucket": "10"}
    assert remaining_capacity(template, accepted, {"scene/room": 100},
                              {"eval_model": {"3": 3, "5": 3, "10": 3}}) == 0


def test_training_capacity_accounts_for_other_classes_in_same_room():
    template = SimpleNamespace(split="train", class_id="chair", scene="scene", room="room")
    accepted = {"a": {"split": "train", "class_id": "hamper", "room": "scene/room", "bucket": "3"}}
    assert remaining_capacity(template, accepted, {"scene/room": 1}, {}) == 0
    accepted.update({str(i): {"split": "train", "class_id": "chair", "room": "other/room", "bucket": "3"}
                     for i in range(1000)})
    assert remaining_capacity(template, accepted, {"scene/room": 50}, {}) == 0


def test_camera_production_retains_exact_runtime_and_stationary_goal(tmp_path):
    cmd = ["python", "submit_fast.py", "submit", "camera_relocalization-obstructed-chair_1", "1",
           "--code", "/frozen/code", "--base-index", "10000000", "--tag", "validation",
           "--sample-limit-s", "2700", "--time-s", "5400", "--customs", "template",
           "--extra-env", "KEEP_SCENE=--bddl-root /owned/private/bddl",
           "--extra-env", "ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL=1",
           "--extra-env", "TEMPLATE=/owned/fresh/template",
           "--extra-env", "ISPATIALGYM_CAMPAIGN_CONFIG=/old/config.json",
           "--extra-env", "EVAL_INDEX=/old/index.json",
           "--extra-env", "ISGF_SAMPLING_CACHE=/old/cache",
           "--extra-env", "ISGF_COLLECT_EXTRA=--no-goal-override"]
    actual = production_command(cmd, directory=tmp_path, index=50000000, count=20)
    assert actual[4] == "20" and actual[actual.index("--code") + 1] == "/frozen/code"
    assert "KEEP_SCENE=--bddl-root /owned/private/bddl" in actual
    assert "ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL=1" in actual
    assert "TEMPLATE=/owned/fresh/template" in actual
    assert "ISPATIALGYM_CAMPAIGN_CONFIG=" + str(tmp_path / "campaign_config.json") in actual
    assert cmd[4] == "1"
    with pytest.raises(ContractError, match="preserve"):
        production_command(cmd[:-2], directory=tmp_path, index=50000000, count=20)


def test_each_camera_split_has_one_authority_and_allocations_are_unambiguous(tmp_path):
    t = SimpleNamespace(task="camera_relocalization-standard", split="eval_category")
    assert authority(tmp_path, t) == tmp_path / "dataset/camera_category_quota.json"
    t.task = "camera_relocalization-obstructed"
    assert authority(tmp_path, t) == tmp_path / "dataset/camera_obstructed_quota.json"
    with pytest.raises(ContractError, match="durable allocation"):
        allocation_for([{"jobid": "1"}, {"jobid": "1"}], "1")


def test_empty_camera_job_reaches_terminal_without_claiming_nonexistent_members(tmp_path, monkeypatch):
    def unexpected(*args):
        pytest.fail("empty reports must not enter release auditor")
    monkeypatch.setattr(controller,"review_and_claim",unexpected)
    record={"allocation":{"job_dir":str(tmp_path)},"status":"waiting"}
    assert refresh_production(record,None,None,None,tmp_path,tmp_path,closed=False) is None
    assert record["status"] == "waiting" and record["has_successful_reports"] is False
    write_json_atomic(tmp_path/"demo/failed/meta.json",{"success":False})
    assert refresh_production(record,None,None,None,tmp_path,tmp_path,closed=True) is None
    assert record["status"] == "complete" and "last_claim" not in record


def test_nfs_lock_failure_keeps_successful_members_but_requires_new_runtime_gate(tmp_path, monkeypatch):
    claimed=[]
    def claim(*args):
        claimed.append(True)
        return {"accepted":{"success":"train"}},{}
    monkeypatch.setattr(controller,"review_and_claim",claim)
    write_json_atomic(tmp_path/"demo/success/meta.json",{"success":True})
    (tmp_path/"sample.log").write_text("Traceback:\n fcntl.flock(stream.fileno(), fcntl.LOCK_EX)\nOSError: [Errno 5] Input/output error\n")
    record={"allocation":{"job_dir":str(tmp_path)},"status":"waiting"}
    failure=refresh_production(record,None,None,None,tmp_path,tmp_path,closed=True)
    assert failure["reason"] == "unsupported_nfs_failure_journal_lock"
    assert record["last_claim"]["accepted"] == {"success":"train"} and len(claimed) == 1
    assert refresh_production(record,None,None,None,tmp_path,tmp_path,closed=True) == failure
    assert len(claimed) == 1
