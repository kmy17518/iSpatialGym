from dataclasses import replace
from pathlib import Path

import pytest

from scripts.campaign_20261110 import contracts as c, group_lane as lane
from scripts.campaign_20261110.acceptance import read
from scripts.campaign_20261110.tests.test_acceptance import fixture, audit


def reservations():
    return {key: [] for key in ("model_holdout_models", "category_holdout_models", "category_holdout_categories",
                               "category_holdout_rooms", "train_exclude_models", "pending_custom_hard_reservations_not_ready")}


def candidate(root, number, magnitude=.075, goal=None, split="eval_standard"):
    f = fixture(root, prefix=f"g{number}_", split=split)
    for path in (f.job / "inst").rglob("*-tro_state.json"):
        tro = read(path)
        group = tro["ispatialgym"]["campaign"]["runtime"]["goal_group"]
        group["actual_magnitude"] = magnitude
        group["requested_goal"] = [0, 1, number if goal is None else goal]
        c.write_json_atomic(path, tro)
        iid = tro["ispatialgym"]["instance_id"]
        sidecar = path.with_name(iid+".oracle.json")
        c.write_json_atomic(sidecar, {"tro_sha256": c.file_digest(path)})
    gate = read(f.gate)
    gate["tro_sha256"] = c.file_digest(gate["instance"])
    c.write_json_atomic(f.gate, gate)
    return f, audit(f)


def claim(root, f, review, reserved=None):
    reserved = reservations() if reserved is None else reserved
    hashes = {"schema": "campaign-reserved-asset-hashes-v1", "reserved_model_keys": reserved["train_exclude_models"],
              "ready_assets": {}, "reserved_asset_hashes": [], "pending_unready_model_keys": reserved["train_exclude_models"]}
    return lane.claim_group(review, f.template, reserved, reservation_hashes=hashes,
                            root=root, started_at=f.since, state_path=root / "dataset/quotas.sqlite3")


def test_strict_eval_bucket_caps_with_train_siblings_and_idempotent_export(tmp_path):
    for index in range(4):
        f, review = candidate(tmp_path, index)
        result = claim(tmp_path, f, review)
        assert list(result["accepted"].values()).count("eval_standard") == (1 if index < 3 else 0)
        assert list(result["accepted"].values()).count("train") >= 1
        assert claim(tmp_path, f, review) == result
        exported = tmp_path / "dataset/groups" / (review["group_id"]+".json")
        assert read(exported) == result
    assert read(tmp_path / "dataset/group_counts.json")["counts"] == {"eval_standard": 3, "train": 5}


def test_duplicate_canonical_goal_cannot_bypass_group_cap(tmp_path):
    f, review = candidate(tmp_path, 0, goal=7)
    claim(tmp_path, f, review)
    f2, review2 = candidate(tmp_path, 1, goal=7)
    with pytest.raises(c.ContractError, match="canonical goal"):
        claim(tmp_path, f2, review2)
    assert not (tmp_path / "dataset/groups/g1_group.json").exists()


def test_changed_artifacts_and_reserved_models_fail_before_claim(tmp_path):
    f, review = candidate(tmp_path, 0)
    holdout = {**reservations(), "train_exclude_models": ["apple/abc"]}
    with pytest.raises(c.ContractError, match="held-out model"):
        claim(tmp_path, f, review, holdout)
    Path(review["members"][0]["demo"]).write_bytes(b"modified")
    with pytest.raises(c.ContractError, match="demo content"):
        claim(tmp_path, f, review)
    assert not (tmp_path / "dataset/quotas.sqlite3").exists()


def test_train_buckets_soft_eval_strict_and_boundary_rules(tmp_path):
    f, review = candidate(tmp_path, 0, magnitude=.31, split="train")
    assert set(claim(tmp_path, f, review)["accepted"].values()) == {"train"}
    with pytest.raises(c.ContractError, match="outside task buckets"):
        lane.bucket_index(replace(f.template, split="eval_standard"), {"actual_magnitude": .31}, strict=True)
    assert lane.bucket_index(f.template, {"actual_magnitude": .10}, strict=True) == 0
    tray = replace(f.template, task="disembedding")
    assert lane.bucket_index(tray, {"actual_crowdedness": .30}, strict=True) == 1
    joint = replace(f.template, task="configuration_matching-articulation_open_small_scale")
    assert lane.bucket_index(joint, {"actual_open_fraction": .5}, strict=True) == 0


def test_heldout_allocations_never_fall_back_to_training():
    rows = [{"instance_id": str(i), "bucket": 0} for i in range(3)]
    counts = lane.Counter({0: 3, 1: 3, 2: 2})
    q = lane.QUOTAS["object_scaling"]
    assert lane._allocate(rows, split="eval_model", q=q, eval_counts=counts, train_count=0) == {}
    rows[0]["bucket"] = 2
    assert lane._allocate(rows, split="eval_model", q=q, eval_counts=counts, train_count=0) == {"0": "eval_model"}
    assert lane._allocate(rows, split="eval_standard", q=q, eval_counts=lane.Counter(), train_count=q.train_instances) == {}


def test_quaternion_sign_does_not_create_another_canonical_goal():
    assert lane._canonical_goal({"orientation": [0, 0, 0, 1]}) == lane._canonical_goal({"orientation": [0, 0, 0, -1]})
    assert lane._canonical_goal([[1, 2, 3], [0, 0, 0, 1]]) == lane._canonical_goal([[1, 2, 3], [0, 0, 0, -1]])
