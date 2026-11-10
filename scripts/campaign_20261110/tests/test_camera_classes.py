from scripts.campaign_20261110.camera_classes import allocate_members
import pytest

from scripts.campaign_20261110 import camera_classes, contracts as c
from scripts.campaign_20261110.tests.test_camera_lane import fixture, audit


def members(prefix, split, counts=(5, 5, 5)):
    return [{"instance_id": f"{prefix}_{bucket}_{i}", "bucket": bucket, "geodesic_m": distance,
             "room": "scene/room", "sampling_split": split, "perturbed": False, "artifacts": {"tro": f"{prefix}_{bucket}_{i}"}}
            for bucket, distance, count in zip(("3", "5", "10"), (2, 4, 8), counts) for i in range(count)]


def test_class_specific_eval_and_soft_training_buckets():
    accepted = {}
    for cls in ("hamper", "chair"):
        rows = members(cls, "eval_standard", (5, 5, 5))
        added, deferred = allocate_members(accepted, rows, class_id=cls, training_room_caps={"scene/room": 100})
        assert list(added.values()).count("eval_standard") == 10
        assert list(added.values()).count("train") == 5 and not deferred
        assert allocate_members(accepted, rows, class_id=cls, training_room_caps={"scene/room": 100}) == ({}, [])


def test_model_overflow_never_becomes_training_and_category_quota_is_task_wide():
    accepted = {}
    for cls in ("hamper", "chair"):
        added, deferred = allocate_members(accepted, members(cls, "eval_model"), class_id=cls, training_room_caps={"scene/room": 100})
        assert list(added.values()) == ["eval_model"] * 9 and len(deferred) == 6
    added, deferred = allocate_members(accepted, members("cat1", "eval_category"), class_id="footstool", training_room_caps={"scene/room": 100})
    assert list(added.values()) == ["eval_category"] * 9 and len(deferred) == 6
    added, deferred = allocate_members(accepted, members("cat2", "eval_category"), class_id="second_category", training_room_caps={"scene/room": 100})
    assert not added and len(deferred) == 15
    assert not any(row["split"] == "train" for row in accepted.values())


def test_standard_camera_category_keeps_its_distinct_three_four_three_quota():
    accepted={}
    added,deferred=allocate_members(accepted,members("standardcat","eval_category"),class_id="camera_relocalization-standard",
        training_room_caps={"scene/room":1000},eval_quotas={"eval_category":{"3":3,"5":4,"10":3}})
    assert list(added.values())==["eval_category"]*10 and len(deferred)==5
    assert [sum(row["bucket"]==b for row in accepted.values()) for b in ("3","5","10")]==[3,4,3]


def test_training_honors_room_and_class_caps_and_accepts_only_held_in():
    accepted = {}
    added, deferred = allocate_members(accepted, members("train", "train", (1002, 0, 0)), class_id="hamper", training_room_caps={"scene/room": 2000})
    assert len(added) == 1000 and len(deferred) == 2
    added, deferred = allocate_members(accepted, members("chair", "train", (4, 0, 0)), class_id="chair", training_room_caps={"scene/room": 1002})
    assert len(added) == 2 and len(deferred) == 2
    rows = members("perturbed", "eval_standard", (1, 0, 0))
    rows[0]["perturbed"] = True
    added, _ = allocate_members(accepted, rows, class_id="new", training_room_caps={"scene/room": 2000})
    assert list(added.values()) == ["train"]


def test_obstructed_audit_and_atomic_claim_bind_poses_and_global_reservations(tmp_path):
    f = fixture(tmp_path, obstructed=True)
    review = audit(f)
    reservations = {"model_holdout_models": [], "category_holdout_models": [], "category_holdout_categories": [],
        "category_holdout_rooms": [], "train_exclude_models": [], "pending_custom_hard_reservations_not_ready": []}
    hashes = {"schema": "campaign-reserved-asset-hashes-v1", "reserved_model_keys": [], "ready_assets": {},
              "pending_unready_model_keys": [], "reserved_asset_hashes": []}
    caps = {"house_single_floor/room": 900, **{f"s/r{i}": 900 for i in range(9)}}
    classes = [f.template.class_id]+[camera_classes.TASK+f"-other_{i}" for i in range(8)]
    kwargs = dict(root=f.root, started_at=f.since, standard_classes=classes, training_room_caps=caps,
                  reservations=reservations, reservation_hashes=hashes)
    path = tmp_path/"dataset/camera_obstructed.json"
    result = camera_classes.claim_obstructed(review, path, **kwargs)
    assert result["counts"]["splits"] == {"eval_standard": 10, "train": 2}
    assert camera_classes.claim_obstructed(review, path, **kwargs)["accepted"] == {}
    target = f.inst/"camera_01-tro_state.json"
    tro = camera_classes.read(target)
    tro["ispatialgym"]["pose_targets"]["hamper.n.02_1"]["position"][0] += .1
    c.write_json_atomic(target, tro)
    with pytest.raises(c.ContractError, match="goal differs from its initial"):
        camera_classes.claim_obstructed(review, path, **kwargs)


def test_blocked_and_free_distance_must_both_match_strict_bucket(tmp_path):
    f = fixture(tmp_path, obstructed=True)
    summary = f.inst/"camera_01.summary.json"
    c.write_json_atomic(summary, {"geodesic_m": 2, "geodesic_blocked_m": 4, "path_bucket_m": 5})
    with pytest.raises(c.ContractError, match="changed the strict camera bucket"):
        audit(f)
