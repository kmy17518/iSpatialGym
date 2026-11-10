#!/usr/bin/env python3
"""Audit fresh camera evidence and atomically allocate strict evaluation quotas."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.campaign_20261110.acceptance import artifact, read, report_passed
from scripts.campaign_20261110.contracts import (
    Perturbations, ScaleState, ValidationEvidence, ValidationKey, digest, file_digest,
    require, require_validation, validate_campaign_block, write_json_atomic,
)
from scripts.campaign_20261110.goal_groups import load_template_contract

TASK = "camera_relocalization-standard"
EVAL_QUOTAS = {"3": 3, "5": 4, "10": 3}


def path_bucket(distance):
    require(isinstance(distance, (int, float)) and not isinstance(distance, bool)
            and math.isfinite(distance) and 0 < distance <= 10, "invalid camera geodesic distance")
    return next(str(hi) for hi in (3, 5, 10) if distance <= hi)


def room_caps(plan, total=1000):
    """Largest remainder preserves the exact total while respecting area weights."""
    rows = plan["tasks"][TASK]
    require(plan["rooms_per_task"] == 10 and len(rows) == 10, "camera training requires ten rooms")
    weights = {f"{row['scene']}/{row['room']}": float(row["traversable_m2"]) for row in rows}
    require(len(weights) == 10 and all(math.isfinite(v) and v > 0 for v in weights.values()), "invalid room areas")
    require(isinstance(total, int) and total >= 10, "invalid training total")
    scaled = {key: value / sum(weights.values()) * total for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in scaled.items()}
    for key in sorted(weights, key=lambda k: (-(scaled[k] - result[k]), k))[:total - sum(result.values())]:
        result[key] += 1
    require(all(v > 0 for v in result.values()), "every training room needs a nonzero allocation")
    return result


def _block(tro, template, started_at):
    meta = tro["ispatialgym"]
    block = meta.get("campaign")
    require(block is not None, "unannotated baseline cannot release camera production")
    validate_campaign_block(block, object_bindings=meta["object_bindings"])
    require(block["task"] == TASK and block["class_id"] == template.class_id, "not the requested camera class")
    require(block["template_sha256"] == template.fingerprint, "camera ready contract differs from instance")
    require(block["created_at"] >= started_at, "historical camera instance")
    require(block.get("goal_group_id") is None, "camera lane does not allocate manipulation groups")
    require(block["split"] in ("eval_standard", "train"), "held-out instance cannot enter camera standard/train lane")
    return meta, block, ValidationKey(**block["validation_key"])


def _report(path, mode, iid, snapshot, root, since):
    path = artifact(path, root, since)
    report = read(path)
    report_passed(report, mode)
    require(report["instance_id"] == iid, "report/TRO identity mismatch")
    require(Path(report["provenance"]["isg_root"]).resolve() == snapshot, "report used another code snapshot")
    created = datetime.datetime.fromisoformat(report["started_utc"].replace("Z", "+00:00")).timestamp()
    require(created >= since, "historical camera report")
    return path, report, created


def audit_camera(job, gate_path, *, root, started_at, template):
    """The gate may come from a previous fresh job with the identical validation key."""
    root, job = Path(root).resolve(), Path(job).resolve()
    require(template.task == TASK, "camera lane requires the standard camera task")
    require(job.is_relative_to(root / "out"), "camera job outside campaign output")
    gate_path = artifact(gate_path, root, started_at)
    gate = read(gate_path)
    require(gate.get("success") is True and gate.get("replay_mode") == "physics_action_open_loop", "no successful action replay gate")
    source_job = Path(gate["source_job"]).resolve()
    require(source_job.is_relative_to(root / "out"), "gate source outside campaign output")
    source_tro = artifact(gate["instance"], root, started_at)
    source_demo = artifact(gate["demo"], root, started_at)
    require(source_tro.is_relative_to(source_job / "inst") and source_demo.is_relative_to(source_job / "demo"), "gate source outside job")
    require(gate.get("tro_sha256") == file_digest(source_tro), "gate TRO changed or lacks digest")
    require(gate.get("demo_sha256") == file_digest(source_demo), "gate demo changed or lacks digest")
    meta, campaign, key = _block(read(source_tro), template, started_at)
    snapshot = Path(gate["snapshot"]).resolve()
    require(snapshot.is_relative_to(root / "code"), "unowned code snapshot")
    identity = snapshot / ".snapshot_commit"
    if not identity.is_file():
        identity = snapshot / ".snapshot_isg"
    require(identity.is_file() and file_digest(identity) == key.code_sha256, "gate code differs from validation key")
    iid = meta["instance_id"]
    evidence = []
    for path, mode, kind in ((source_demo.parent / "meta.json", "oracle", "demo"),
                             (gate_path.parent / "replay" / iid / "meta.json", "replay", "open_loop_replay")):
        path, _, created = _report(path, mode, iid, snapshot, root, started_at)
        evidence.append(ValidationEvidence(key, kind, True, True, created, str(path), file_digest(path), file_digest(source_demo), iid, True))
    require_validation(key, evidence, campaign_started_at=started_at)
    members = []
    for report_path in sorted((job / "demo").glob("*/meta.json")):
        report = read(report_path)
        if report.get("success") is not True:
            continue
        iid = report["instance_id"]
        report_path, _, _ = _report(report_path, "oracle", iid, snapshot, root, started_at)
        paths = list((job / "inst").rglob(iid + "-tro_state.json"))
        require(len(paths) == 1, "camera member TRO missing or duplicated")
        tro_path = artifact(paths[0], root, started_at)
        meta, block, member_key = _block(read(tro_path), template, started_at)
        require(member_key == key, "camera job mixes validation keys")
        require(block["campaign_id"] == campaign["campaign_id"], "camera job mixes campaigns")
        require(meta["instance_id"] == iid, "camera member identity mismatch")
        sidecar = artifact(tro_path.with_name(iid + ".oracle.json"), root, started_at)
        side = read(sidecar)
        require(side["tro_sha256"] == file_digest(tro_path), "camera sidecar/TRO mismatch")
        summary_path = artifact(tro_path.with_name(iid + ".summary.json"), root, started_at)
        summary = read(summary_path)
        # Some writer versions wrap the sampler summary in a metadata record.
        summary = summary.get("summary", summary)
        bucket = path_bucket(summary["geodesic_m"])
        require(float(summary["path_bucket_m"]) == float(bucket) == float(side["path_bucket_m"]), "camera bucket disagrees with geodesic/sidecar")
        p = block["perturbations"]
        perturbations = Perturbations({n: ScaleState(**v) for n, v in p["scales"].items()}, tuple(p["removed_objects"]), p["lighting"])
        demo = artifact(report_path.parent / "demo.hdf5", root, started_at)
        paths_by_kind = {"tro": tro_path, "sidecar": sidecar, "summary": summary_path, "demo": demo, "report": report_path}
        members.append({"instance_id": iid, "sampling_split": block["split"], "bucket": bucket,
                        "geodesic_m": summary["geodesic_m"], "room": f"{template.scene}/{template.room}",
                        "perturbed": perturbations.perturbed,
                        "artifacts": {name: {"path": str(path), "sha256": file_digest(path)} for name, path in paths_by_kind.items()}})
    require(members, "no successfully demonstrated camera members")
    return {"schema": "isg-camera-release-review-v1", "campaign_id": campaign["campaign_id"], "task": TASK,
            "class_id": template.class_id, "validation_key": asdict(key), "template_validation_passed": True,
            "gate": {"path": str(gate_path), "sha256": file_digest(gate_path)}, "evidence": [asdict(e) for e in evidence],
            "members": members, "accepted": False}


def claim_camera(review, state_path, *, root, started_at, training_room_caps):
    """Claim each fresh ID once. Training buckets are deliberately unconstrained."""
    root, state_path = Path(root).resolve(), Path(state_path).resolve()
    require(state_path.is_relative_to(root), "quota state must stay inside campaign root")
    require(review.get("schema") == "isg-camera-release-review-v1" and review.get("template_validation_passed") is True, "camera review not validated")
    require(len(training_room_caps) == 10 and sum(training_room_caps.values()) == 1000, "training requires the ten-room 1000-instance plan")
    key = ValidationKey(**review["validation_key"])
    evidence = [ValidationEvidence(**{**row, "key": ValidationKey(**row["key"])}) for row in review["evidence"]]
    require_validation(key, evidence, campaign_started_at=started_at)
    for row in [review["gate"]] + [item for member in review["members"] for item in member["artifacts"].values()]:
        path = artifact(row["path"], root, started_at)
        require(file_digest(path) == row["sha256"], "camera artifact changed after review")
    for row in evidence:
        require(file_digest(artifact(row.report_path, root, started_at)) == row.report_sha256, "camera evidence changed after review")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(state_path.suffix + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read(state_path) if state_path.exists() else {
            "schema": "isg-camera-quota-state-v1", "campaign_id": review["campaign_id"], "task": TASK,
            "eval_quotas": EVAL_QUOTAS, "training_room_caps": training_room_caps, "accepted": {}}
        require(state["campaign_id"] == review["campaign_id"] and state["task"] == TASK, "quota state belongs to another campaign/task")
        require(state["eval_quotas"] == EVAL_QUOTAS and state["training_room_caps"] == training_room_caps, "quota policy changed")
        accepted = state["accepted"]
        eval_counts = Counter(row["bucket"] for row in accepted.values() if row["split"] == "eval_standard")
        train_counts = Counter(row["room"] for row in accepted.values() if row["split"] == "train")
        added, deferred = {}, []
        for member in review["members"]:
            iid, bucket, room = member["instance_id"], member["bucket"], member["room"]
            require(bucket in EVAL_QUOTAS and path_bucket(member["geodesic_m"]) == bucket, "invalid reviewed bucket")
            if iid in accepted:
                require(accepted[iid]["artifacts"] == member["artifacts"], "accepted camera ID changed")
                continue
            if member["sampling_split"] == "eval_standard" and not member["perturbed"] and eval_counts[bucket] < EVAL_QUOTAS[bucket]:
                split = "eval_standard"
                eval_counts[bucket] += 1
            elif room in training_room_caps and train_counts[room] < training_room_caps[room]:
                split = "train"
                train_counts[room] += 1
            else:
                deferred.append(iid)
                continue
            accepted[iid] = {**member, "split": split, "validation_key": asdict(key), "gate": review["gate"],
                             "evidence": review["evidence"], "release_review_sha256": digest(review)}
            added[iid] = split
        state["counts"] = {"eval_standard": dict(eval_counts), "train_rooms": dict(train_counts), "train_total": sum(train_counts.values())}
        write_json_atomic(state_path, state)
    return {"accepted": added, "deferred": deferred, "counts": state["counts"], "state": str(state_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("job", "gate", "root", "contracts", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--started-at", required=True, type=float)
    parser.add_argument("--claim-state")
    parser.add_argument("--room-plan")
    args = parser.parse_args()
    review = audit_camera(args.job, args.gate, root=args.root, started_at=args.started_at,
                          template=load_template_contract(args.contracts, TASK))
    write_json_atomic(args.output, review)
    if args.claim_state:
        require(args.room_plan is not None, "claim requires the ten-room plan")
        result = claim_camera(review, args.claim_state, root=args.root, started_at=args.started_at,
                              training_room_caps=room_caps(read(args.room_plan)))
        print(result)
    else:
        print({"validated_members": len(review["members"]), "accepted": False})


if __name__ == "__main__":
    main()
