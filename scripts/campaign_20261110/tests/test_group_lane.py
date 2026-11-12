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


def room_plan(first_area=1000):
    return {"schema":"isg-room-plan-v1","rooms_per_task":10,"reserved_rooms":[],"tasks":{
        "object_scaling":[{"scene":"house_single_floor","room":"room","traversable_m2":first_area}]+
        [{"scene":"house_single_floor","room":f"other{i}","traversable_m2":1111} for i in range(9)]}}


def claim(root, f, review, reserved=None, plan=None):
    reserved = reservations() if reserved is None else reserved
    hashes = {"schema": "campaign-reserved-asset-hashes-v1", "reserved_model_keys": reserved["train_exclude_models"],
              "ready_assets": {}, "reserved_asset_hashes": [], "pending_unready_model_keys": reserved["train_exclude_models"]}
    return lane.claim_group(review, f.template, reserved, reservation_hashes=hashes,
                            root=root, started_at=f.since, state_path=root / "dataset/quotas.sqlite3",
                            training_room_plan=room_plan() if plan is None else plan)


def regression(root, f, review):
    reserved=reservations()
    hashes={'schema':'campaign-reserved-asset-hashes-v1','reserved_model_keys':[],
            'ready_assets':{},'reserved_asset_hashes':[],'pending_unready_model_keys':[]}
    return lane.claim_group(review,f.template,reserved,reservation_hashes=hashes,root=root,
        started_at=f.since,state_path=root/'dataset/quotas.sqlite3',training_room_plan=room_plan(),validation_only=True)


def test_full_strict_regression_never_creates_or_changes_quota_database(tmp_path):
    f,review=candidate(tmp_path,0)
    result=regression(tmp_path,f,review)
    assert result['validation_passed'] and result['quota_consumed']==0 and result['accepted']=={}
    assert not (tmp_path/'dataset').exists()
    claim(tmp_path,f,review)
    path=tmp_path/'dataset/quotas.sqlite3';before=path.read_bytes()
    f2,review2=candidate(tmp_path,1)
    assert regression(tmp_path,f2,review2)['validation_passed']
    assert path.read_bytes()==before
    Path(review2['members'][0]['demo']).write_bytes(b'changed')
    with pytest.raises(c.ContractError,match='demo content'):
        regression(tmp_path,f2,review2)
    assert path.read_bytes()==before


def test_regression_rejects_singleton_and_invalid_bucket(tmp_path):
    f=fixture(tmp_path,starts=(1,))
    with pytest.raises(c.ContractError):regression(tmp_path,f,audit(f))
    f2,review=candidate(tmp_path,1,magnitude=.31)
    with pytest.raises(c.ContractError,match='outside task buckets'):regression(tmp_path,f2,review)


def test_suite_marked_source_cannot_enter_legacy_single_gate_claim(tmp_path):
    f,review=candidate(tmp_path,0)
    (f.job/'job.env').write_text('ISPATIALGYM_RUNTIME_SUITE_RUN=current100\n')
    with pytest.raises(c.ContractError,match='dedicated actual-member'):
        claim(tmp_path,f,review)
    assert not (tmp_path/'dataset').exists()


def test_strict_eval_bucket_caps_with_train_siblings_and_idempotent_export(tmp_path):
    for index in range(4):
        f, review = candidate(tmp_path, index)
        result = claim(tmp_path, f, review)
        assert list(result["accepted"].values()).count("eval_standard") == (1 if index < 3 else 0)
        assert list(result["accepted"].values()).count("train") == (1 if index < 3 else 0)
        assert claim(tmp_path, f, review) == result
        exported = tmp_path / "dataset/groups" / (review["group_id"]+".json")
        if index < 3: assert read(exported) == result
        else: assert not exported.exists()
    assert read(tmp_path / "dataset/group_counts.json")["counts"] == {"eval_standard": 3, "train": 3}


def test_repeated_canonical_goal_can_form_another_fresh_group_under_owner_policy(tmp_path):
    f, review = candidate(tmp_path, 0, goal=7)
    claim(tmp_path, f, review)
    f2, review2 = candidate(tmp_path, 1, goal=7)
    second=claim(tmp_path, f2, review2)
    assert sorted(second["accepted"].values()) == ["eval_standard","train"]
    assert second["goal_reuse_policy"]["version"] == 2
    assert (tmp_path / "dataset/groups/g1_group.json").exists()


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
    assert claim(tmp_path, f, review)["accepted"] == {}
    with pytest.raises(c.ContractError, match="outside task buckets"):
        lane.bucket_index(replace(f.template, split="eval_standard"), {"actual_magnitude": .31}, strict=True)
    assert lane.bucket_index(f.template, {"actual_magnitude": .10}, strict=True) == 0
    tray = replace(f.template, task="disembedding")
    assert lane.bucket_index(tray, {"actual_crowdedness": .30}, strict=True) == 1
    joint = replace(f.template, task="configuration_matching-articulation_open_small_scale")
    assert lane.bucket_index(joint, {"actual_open_fraction": .5}, strict=True) == 0
    block = replace(f.template, task="configuration_matching-block_design")
    assert lane.bucket_index(block, {"actual_flips": 0}, strict=False) is None
    with pytest.raises(c.ContractError, match="flip count"):
        lane.bucket_index(block, {"actual_flips": 0}, strict=True)


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


def test_area_weighted_room_cap_counts_siblings_and_freezes_policy(tmp_path):
    plan=room_plan(first_area=1)
    policy=lane.training_room_policy(plan,"object_scaling")
    assert policy["caps"]["house_single_floor/room"]==1
    assert len(policy["caps"])==10 and sum(policy["caps"].values())==10000
    f,review=candidate(tmp_path,0)
    assert sorted(claim(tmp_path,f,review,plan=plan)["accepted"].values())==["eval_standard","train"]
    f2,review2=candidate(tmp_path,1)
    assert claim(tmp_path,f2,review2,plan=plan)["accepted"]=={}
    with pytest.raises(c.ContractError,match="room allocation policy changed"):
        claim(tmp_path,f2,review2,plan=room_plan())


def test_unverified_historical_rows_leave_replacement_capacity_open(tmp_path):
    f,review=candidate(tmp_path,0)
    path=tmp_path/"dataset/quotas.sqlite3";path.parent.mkdir()
    db=lane._db(path)
    with db:
        for i in range(999):
            record={"room":"house_single_floor/other0"}
            db.execute("INSERT INTO members VALUES (?,?,?,?,?,?,?,?)",(f"old{i}","old","train","object_scaling","apple",0,"old",c.canonical_json(record)))
    db.close()
    assert sorted(claim(tmp_path,f,review)["accepted"].values())==["eval_standard","train"]
    f2,review2=candidate(tmp_path,1)
    assert sorted(claim(tmp_path,f2,review2)["accepted"].values())==["eval_standard","train"]


def test_room_cap_migration_reads_verified_legacy_tro_without_changing_it(tmp_path):
    tro=tmp_path/"old-tro.json"
    c.write_json_atomic(tro,{"ispatialgym":{"scene":{"model":"house_single_floor","task_rooms":["living_room_0"]}}})
    record={"tro":str(tro),"tro_sha256":c.file_digest(tro)}
    assert lane.accepted_room(record)=="house_single_floor/living_room_0"
    before=tro.read_bytes()
    assert lane.accepted_room(record)=="house_single_floor/living_room_0" and tro.read_bytes()==before
    tro.write_text("{}")
    with pytest.raises(c.ContractError,match="accepted TRO changed"):
        lane.accepted_room(record)
