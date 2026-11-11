#!/usr/bin/env python3
"""Audit fresh camera evidence and atomically allocate strict evaluation quotas."""
from __future__ import annotations

import argparse
import datetime
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.campaign_20261110.acceptance import artifact, read, report_passed, replay_report_path
from scripts.campaign_20261110.contracts import (
    ObjectRef, Template, Perturbations, ScaleState, ValidationEvidence, ValidationKey, digest, file_digest,
    require, require_validation, validate_campaign_block, write_json_atomic,
)
from scripts.campaign_20261110.goal_groups import load_template_contract
from scripts.campaign_20261110.storage import atomic_directory_lock
from scripts.campaign_20261110.reservation_policy import active_policy

TASK = "camera_relocalization-standard"
CAMERA_TASKS = (TASK, "camera_relocalization-obstructed")
EVAL_QUOTAS = {"3": 3, "5": 4, "10": 3}


def path_bucket(distance):
    require(isinstance(distance, (int, float)) and not isinstance(distance, bool)
            and math.isfinite(distance) and 0 < distance <= 10, "invalid camera geodesic distance")
    return next(str(hi) for hi in (3, 5, 10) if distance <= hi)


def room_caps(plan, total=1000, *, task=TASK):
    """Largest remainder preserves the exact total while respecting area weights."""
    require(task in CAMERA_TASKS, "invalid camera task")
    rows = plan["tasks"][task]
    require(plan["rooms_per_task"] == 10 and len(rows) == 10, "camera training requires ten rooms")
    weights = {f"{row['scene']}/{row['room']}": float(row["traversable_m2"]) for row in rows}
    reserved = {f"{row['scene']}/{row['room']}" for row in plan.get("reserved_rooms", [])}
    require(not set(weights) & reserved, "camera training plan includes a held-out room")
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
    require(block["task"] == template.task and block["class_id"] == template.class_id, "not the requested camera class")
    require(block["template_sha256"] == template.fingerprint, "camera ready contract differs from instance")
    require(block["created_at"] >= started_at, "historical camera instance")
    require(block.get("goal_group_id") is None, "camera lane does not allocate manipulation groups")
    require(block["split"] == template.split or (template.split == "eval_standard" and block["split"] == "train"), "camera split differs from ready contract")
    if template.task == "camera_relocalization-obstructed":
        proof = template.scene_setup.get("runtime_bddl_adaptation", {})
        require(proof.get("schema") == "isg-stationary-obstacle-definition-v1", "missing stationary-obstacle definition proof")
        require(meta.get("bddl_sha256") == template.scene_setup.get("runtime_bddl_sha256"), "obstacle runtime definition differs from contract")
        require(proof.get("measure") == [["pose_reached", "?agent.n.01_1"]], "obstacle measure must stay camera-only")
        initial = block.get("runtime", {}).get("planned_start_poses", {})
        targets = meta.get("pose_targets", {})
        for name in proof["obstacle_bindings"]:
            require(name in initial and name in targets, "stationary obstacle lacks initial/goal pose")
            target = targets[name]
            require(target.get("frame", "world") in (None, "world") and target.get("position_metric") == target.get("orientation_metric") == "full", "stationary obstacle must gate its world pose")
            require(target.get("position") == initial[name][0] and target.get("orientation") == initial[name][1], "stationary obstacle goal differs from its initial pose")
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
    require(template.task in CAMERA_TASKS, "camera lane requires a camera task")
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
                             (replay_report_path(gate,gate_path,iid,root=root,started_at=started_at), "replay", "open_loop_replay")):
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
        free_bucket = path_bucket(summary["geodesic_m"])
        distance = summary["geodesic_blocked_m"] if template.task == "camera_relocalization-obstructed" else summary["geodesic_m"]
        bucket = path_bucket(distance)
        require(bucket == free_bucket, "obstructed path changed the strict camera bucket")
        require(float(summary["path_bucket_m"]) == float(bucket) == float(side["path_bucket_m"]), "camera bucket disagrees with geodesic/sidecar")
        p = block["perturbations"]
        perturbations = Perturbations({n: ScaleState(**v) for n, v in p["scales"].items()}, tuple(p["removed_objects"]), p["lighting"])
        demo = artifact(report_path.parent / "demo.hdf5", root, started_at)
        paths_by_kind = {"tro": tro_path, "sidecar": sidecar, "summary": summary_path, "demo": demo, "report": report_path}
        members.append({"instance_id": iid, "sampling_split": block["split"], "bucket": bucket,
                        "geodesic_m": distance, "room": f"{template.scene}/{template.room}",
                        "perturbed": perturbations.perturbed,
                        "artifacts": {name: {"path": str(path), "sha256": file_digest(path)} for name, path in paths_by_kind.items()}})
    require(members, "no successfully demonstrated camera members")
    return {"schema": "isg-camera-release-review-v1", "campaign_id": campaign["campaign_id"], "task": template.task,
            "class_id": template.class_id, "source_job": str(job), "contract": asdict(template),
            "validation_key": asdict(key), "template_validation_passed": True,
            "gate": {"path": str(gate_path), "sha256": file_digest(gate_path)}, "evidence": [asdict(e) for e in evidence],
            "members": members, "accepted": False}


def claim_camera(review, state_path, *, root, started_at, training_room_caps):
    """Claim each fresh ID once. Training buckets are deliberately unconstrained."""
    root, state_path = Path(root).resolve(), Path(state_path).resolve()
    require(state_path.is_relative_to(root), "quota state must stay inside campaign root")
    require(review.get("schema") == "isg-camera-release-review-v1" and review.get("template_validation_passed") is True, "camera review not validated")
    require(review["task"] == TASK and all(row["sampling_split"] in ("eval_standard", "train") for row in review["members"]), "use obstructed/heldout quota allocator for this camera review")
    require(len(training_room_caps) == 10 and sum(training_room_caps.values()) == 1000, "training requires the ten-room 1000-instance plan")
    raw_template = dict(review["contract"])
    raw_template["objects"] = tuple(ObjectRef(**value) for value in raw_template["objects"])
    fresh = audit_camera(review["source_job"], review["gate"]["path"], root=root, started_at=started_at, template=Template(**raw_template))
    require(digest(fresh) == digest(review), "camera release evidence changed after review")
    key = ValidationKey(**review["validation_key"])
    evidence = [ValidationEvidence(**{**row, "key": ValidationKey(**row["key"])}) for row in review["evidence"]]
    require_validation(key, evidence, campaign_started_at=started_at)
    for row in [review["gate"]] + [item for member in review["members"] for item in member["artifacts"].values()]:
        path = artifact(row["path"], root, started_at)
        require(file_digest(path) == row["sha256"], "camera artifact changed after review")
    for row in evidence:
        require(file_digest(artifact(row.report_path, root, started_at)) == row.report_sha256, "camera evidence changed after review")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_directory_lock(state_path.with_suffix(state_path.suffix + ".lockdir")):
        active=active_policy(root)
        if active is not None:
            from scripts.campaign_20261110.group_lane import check_reservations
            check_reservations(Template(**raw_template),active[1],active[2])
        state = read(state_path) if state_path.exists() else {
            "schema": "isg-camera-quota-state-v1", "campaign_id": review["campaign_id"], "task": TASK,
            "eval_quotas": EVAL_QUOTAS, "training_room_caps": training_room_caps, "accepted": {}}
        require(state["campaign_id"] == review["campaign_id"] and state["task"] == TASK, "quota state belongs to another campaign/task")
        if active is not None:
            require(state.get("reservation_policy_sha256",active[0]["policy_sha256"])==active[0]["policy_sha256"], "camera quota state uses a superseded reservation policy")
            state["reservation_policy_sha256"]=active[0]["policy_sha256"]
        require(state["eval_quotas"] == EVAL_QUOTAS and state["training_room_caps"] == training_room_caps, "quota policy changed")
        accepted = state["accepted"]
        eval_counts = Counter(row["bucket"] for row in accepted.values() if row["split"] == "eval_standard")
        # Cameras have no goal-group sibling exception. Historical rows remain
        # stored, but cannot consume compliant replacement capacity.
        train_counts = Counter()
        added, deferred = {}, []
        for member in review["members"]:
            iid, bucket, room = member["instance_id"], member["bucket"], member["room"]
            require(bucket in EVAL_QUOTAS and path_bucket(member["geodesic_m"]) == bucket, "invalid reviewed bucket")
            if iid in accepted:
                require(accepted[iid]["artifacts"] == member["artifacts"], "accepted camera ID changed")
                require(digest(accepted[iid].get("contract",review["contract"])) == digest(review["contract"]), "accepted camera contract changed")
                # Older quota rows predate the standalone export. Backfill only
                # from the freshly re-audited, identical physical artifacts.
                accepted[iid]["contract"] = review["contract"]
                accepted[iid]["source_job"] = review["source_job"]
                continue
            if member["sampling_split"] == "eval_standard" and not member["perturbed"] and eval_counts[bucket] < EVAL_QUOTAS[bucket]:
                split = "eval_standard"
                eval_counts[bucket] += 1
            else:
                deferred.append(iid)
                continue
            accepted[iid] = {**member, "split": split, "validation_key": asdict(key), "gate": review["gate"],
                             "evidence": review["evidence"], "release_review_sha256": digest(review),
                             "contract":review["contract"],"source_job":review["source_job"]}
            added[iid] = split
        state["counts"] = {"eval_standard": dict(eval_counts), "train_rooms": dict(train_counts), "train_total": sum(train_counts.values())}
        write_json_atomic(state_path, state)
    return {"accepted": added, "deferred": deferred, "counts": state["counts"], "state": str(state_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("job", "gate", "root", "contracts", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--started-at", required=True, type=float)
    parser.add_argument("--class-id", default=TASK)
    parser.add_argument("--claim-state")
    parser.add_argument("--room-plan")
    args = parser.parse_args()
    review = audit_camera(args.job, args.gate, root=args.root, started_at=args.started_at,
                          template=load_template_contract(args.contracts, args.class_id))
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
