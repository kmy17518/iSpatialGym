"""Atomic camera-obstructed allocation across classes and held-out splits."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import tempfile

from scripts.campaign_20261110.camera_lane import audit_camera, path_bucket
from scripts.campaign_20261110.contracts import ObjectRef, Template, digest, require, write_json_atomic
from scripts.campaign_20261110.acceptance import read
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.storage import atomic_directory_lock, durable_atomic_copy

TASK = "camera_relocalization-obstructed"
EVAL_QUOTAS = {"eval_standard": {"3": 3, "5": 4, "10": 3},
               "eval_model": {"3": 3, "5": 3, "10": 3},
               "eval_category": {"3": 3, "5": 3, "10": 3}}


def allocate_members(accepted, members, *, class_id, training_room_caps):
    """Pure quota decision: held-out overflow can never become training."""
    eval_counts = Counter((row["split"], "*" if row["split"] == "eval_category" else row["class_id"], row["bucket"])
                          for row in accepted.values() if row["split"] != "train")
    room_counts = Counter(row["room"] for row in accepted.values() if row["split"] == "train")
    class_counts = Counter(row["class_id"] for row in accepted.values() if row["split"] == "train")
    added, deferred = {}, []
    for member in members:
        iid, bucket, room = member["instance_id"], member["bucket"], member["room"]
        require(path_bucket(member["geodesic_m"]) == bucket, "invalid camera distance bucket")
        if iid in accepted:
            require(accepted[iid]["artifacts"] == member["artifacts"] and accepted[iid]["class_id"] == class_id,
                    "accepted camera identity changed")
            continue
        source = member["sampling_split"]
        slot = (source, "*" if source == "eval_category" else class_id, bucket)
        if source in EVAL_QUOTAS and not member["perturbed"] and eval_counts[slot] < EVAL_QUOTAS[source][bucket]:
            split = source
            eval_counts[slot] += 1
        elif (source in ("eval_standard", "train") and room in training_room_caps
              and room_counts[room] < training_room_caps[room] and class_counts[class_id] < 1000):
            split = "train"
            room_counts[room] += 1
            class_counts[class_id] += 1
        else:
            deferred.append(iid)
            continue
        accepted[iid] = {**member, "class_id": class_id, "split": split}
        added[iid] = split
    return added, deferred


def claim_obstructed(review, state_path, *, root, started_at, standard_classes,
                     training_room_caps, reservations, reservation_hashes):
    root, state_path = Path(root).resolve(), Path(state_path).resolve()
    require(state_path.is_relative_to(root), "camera state outside campaign")
    classes = sorted(set(standard_classes))
    require(len(classes) == 9 and all(name.startswith(TASK + "-") for name in classes), "freeze the nine camera obstacle classes")
    require(len(training_room_caps) == 10 and sum(training_room_caps.values()) == 9000
            and all(isinstance(n, int) and n > 0 for n in training_room_caps.values()), "obstructed training requires its ten-room 9000-instance plan")
    raw = dict(review["contract"])
    raw["objects"] = tuple(ObjectRef(**row) for row in raw["objects"])
    template = Template(**raw)
    require(template.task == review["task"] == TASK, "wrong camera task")
    require(template.split == "eval_category" or template.class_id in classes, "camera class outside frozen standard/model inventory")
    check_reservations(template, reservations, reservation_hashes)
    fresh = audit_camera(review["source_job"], review["gate"]["path"], root=root, started_at=started_at, template=template)
    require(digest(fresh) == digest(review), "camera release evidence changed after review")
    reservation_policy = {key: reservations[key] for key in ("model_holdout_models", "category_holdout_models",
        "category_holdout_categories", "category_holdout_rooms", "train_exclude_models", "pending_custom_hard_reservations_not_ready")}
    policy = {"standard_classes": classes, "eval_quotas": EVAL_QUOTAS, "training_room_caps": training_room_caps,
              "training_per_class": 1000, "reservations_sha256": digest(reservation_policy)}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_directory_lock(state_path.with_suffix(state_path.suffix + ".lockdir")):
        state = read(state_path) if state_path.exists() else {
            "schema": "isg-camera-obstructed-quota-state-v1", "campaign_id": review["campaign_id"],
            "task": TASK, "policy": policy, "accepted": {}}
        require(state["campaign_id"] == review["campaign_id"] and state["task"] == TASK, "camera state belongs to another campaign")
        require(state["policy"] == policy, "camera quota policy changed")
        added, deferred = allocate_members(state["accepted"], review["members"], class_id=template.class_id,
                                           training_room_caps=training_room_caps)
        for iid in added:
            state["accepted"][iid].update({"validation_key": review["validation_key"], "gate": review["gate"],
                "evidence": review["evidence"], "release_review_sha256": digest(review), "contract": asdict(template)})
        rows = state["accepted"].values()
        state["counts"] = {"splits": dict(Counter(row["split"] for row in rows)),
            "train_classes": dict(Counter(row["class_id"] for row in rows if row["split"] == "train")),
            "train_rooms": dict(Counter(row["room"] for row in rows if row["split"] == "train"))}
        with tempfile.TemporaryDirectory(prefix="isg-camera-state-", dir="/tmp") as tmp:
            local = Path(tmp)/"state.json"
            write_json_atomic(local, state)
            durable_atomic_copy(local, state_path)
    return {"accepted": added, "deferred": deferred, "counts": state["counts"], "state": str(state_path)}
