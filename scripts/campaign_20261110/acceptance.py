#!/usr/bin/env python3
"""Audit fresh demo/replay evidence and propose a goal group's split allocation.

This writes a reviewable release record. It neither mutates source instances nor
counts an instance as accepted before a separate quota allocator claims it.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.campaign_20261110.contracts import (
    GroupMember, Perturbations, ScaleState, ValidationEvidence, ValidationKey,
    allocate_goal_group, digest, file_digest, require, require_validation,
    validate_campaign_block, write_json_atomic,
)


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path, root, since):
    path = Path(path).resolve()
    require(path.is_relative_to(root), f"artifact outside campaign: {path}")
    require(path.is_file() and path.stat().st_mtime >= since, f"artifact absent or predates campaign: {path}")
    return path


def report_passed(report, mode):
    require(report.get("mode") == mode and report.get("status") == "success" and report.get("success") is True,
            f"{mode} did not succeed")
    qa = report.get("qa") or {}
    require(qa.get("enabled") is True and not qa.get("hard_triggered"), f"{mode} QA missing or failed")
    require(not any(v.get("triggered") for v in qa.get("metrics", {}).values() if v.get("mode") == "hard"),
            f"{mode} has a hard QA violation")
    require(report.get("goal_override") is None, "campaign goal was overwritten by the oracle")
    restore = report.get("loader", {}).get("campaign_restoration", {})
    require(restore.get("scale_verified") is True, f"{mode} has no verified campaign restoration")


def replay_report_path(gate, gate_path, instance_id, *, root, started_at):
    """Resolve the exact successful attempt without scanning unrelated reports."""
    gate_path=Path(gate_path).resolve();root=Path(root).resolve()
    def checked(value):
        path=artifact(value,root,started_at)
        require(path.is_relative_to(gate_path.parent),'replay report outside this gate attempt tree')
        report=read(path)
        require(report.get('instance_id')==instance_id,'replay report instance differs from gate')
        require(report.get('mode')=='replay','gate report is not a replay')
        return path
    declared=gate.get('replay_meta_paths')
    require(declared is None or (isinstance(declared,list) and declared and all(isinstance(p,str) for p in declared)),
            'invalid explicit replay report list')
    identity=gate.get('scene_identity') or {}
    if identity.get('replay_report') is not None:
        require(identity.get('verified_replay') is True,'scene replay report was not verified')
        path=checked(identity['replay_report'])
        require(identity.get('replay_report_sha256')==file_digest(path),'scene replay report hash changed')
        if declared is not None:
            require(sum(Path(p).resolve()==path for p in declared)==1,'scene replay report differs from declared attempt')
        return path
    if declared is not None:
        require(len({str(Path(p).resolve()) for p in declared})==len(declared),'duplicate replay reports declared')
        matches=[]
        for value in declared:
            path=artifact(value,root,started_at)
            require(path.is_relative_to(gate_path.parent),'replay report outside this gate attempt tree')
            if read(path).get('instance_id')==instance_id:matches.append(checked(path))
        require(len(matches)==1,'gate must identify exactly one matching replay report')
        return matches[0]
    return checked(gate_path.parent/'replay'/instance_id/'meta.json')


def audit_job(job, gate_path, *, root, started_at, group_closed=False, group_id=None):
    root, job = Path(root).resolve(), Path(job).resolve()
    require(job.is_relative_to(root / "out"), "job outside campaign output")
    gate_path = artifact(gate_path, root, started_at)
    gate = read(gate_path)
    require(gate.get("success") is True and gate.get("replay_mode") == "physics_action_open_loop", "no successful action replay gate")
    source_job = Path(gate["source_job"]).resolve()
    require(source_job.is_relative_to(root / "out"), "replay gate source outside campaign output")
    require(source_job == job or group_id is not None, "a reused template gate requires an explicit current group id")
    source_tro = artifact(gate["instance"], root, started_at)
    source_demo = artifact(gate["demo"], root, started_at)
    require(source_tro.is_relative_to(source_job / "inst") and source_demo.is_relative_to(source_job / "demo"), "gate source outside job")
    require(gate.get("demo_sha256") == file_digest(source_demo), "gate demo content changed or has no digest")
    require(gate.get("tro_sha256") == file_digest(source_tro), "gate TRO content changed or has no digest")
    raw = read(source_tro)
    campaign = raw["ispatialgym"].get("campaign")
    require(campaign is not None, "baseline canary cannot release a campaign template")
    validate_campaign_block(campaign, object_bindings=raw["ispatialgym"]["object_bindings"])
    group_id = group_id or campaign.get("goal_group_id")
    require(isinstance(group_id, str) and bool(group_id), "missing goal group identity")
    group_dir = (root / "groups" / group_id).resolve()
    require(group_dir.is_relative_to(root / "groups"), "invalid group path")
    require(not (group_dir / "release_block.json").exists(), "goal group is blocked from release; inspect release_block.json")
    key = ValidationKey(**campaign["validation_key"])
    snapshot = Path(gate["snapshot"]).resolve()
    require(snapshot.is_relative_to(root / "code"), "gate does not use an owned campaign snapshot")
    identity = snapshot / ".snapshot_commit"
    if not identity.is_file():
        identity = snapshot / ".snapshot_isg"
    require(file_digest(identity) == key.code_sha256, "gate code differs from campaign validation key")
    iid = raw["ispatialgym"]["instance_id"]
    demo_meta_path = artifact(source_demo.parent / "meta.json", root, started_at)
    replay_meta_path = replay_report_path(gate,gate_path,iid,root=root,started_at=started_at)
    evidence = []
    for path, mode, kind in ((demo_meta_path, "oracle", "demo"), (replay_meta_path, "replay", "open_loop_replay")):
        report = read(path)
        report_passed(report, mode)
        require(report["instance_id"] == iid, "report instance differs from gate")
        require(Path(report["provenance"]["isg_root"]).resolve() == snapshot, "report used another code snapshot")
        created = datetime.datetime.fromisoformat(report["started_utc"].replace("Z", "+00:00")).timestamp()
        evidence.append(ValidationEvidence(key, kind, True, True, created, str(path), file_digest(path),
                                           file_digest(source_demo), iid, True))
    require_validation(key, evidence, campaign_started_at=started_at)
    members, artifacts, duplicates, seen_starts = [], [], [], set()
    for path in sorted((job / "demo").glob("*/meta.json")):
        report = read(path)
        if report.get("success") is not True:
            continue
        report_passed(report, "oracle")
        artifact(path, root, started_at)
        require(Path(report["provenance"]["isg_root"]).resolve() == snapshot, "member used another code snapshot")
        require(datetime.datetime.fromisoformat(report["started_utc"].replace("Z", "+00:00")).timestamp() >= started_at,
                "member report predates campaign")
        member_id = report["instance_id"]
        paths = list((job / "inst").rglob(member_id + "-tro_state.json"))
        require(len(paths) == 1, "member TRO missing or duplicated")
        tro_path = artifact(paths[0], root, started_at)
        tro = read(tro_path)
        meta = tro["ispatialgym"]
        block = meta["campaign"]
        validate_campaign_block(block, object_bindings=meta["object_bindings"])
        if block.get("goal_group_id") != group_id:
            continue  # A queued world may collect multiple groups with this same key.
        require(ValidationKey(**block["validation_key"]) == key, "job mixes template validation keys")
        require(block["split"] == campaign["split"] and block["class_id"] == campaign["class_id"]
                and block["task"] == campaign["task"], "job mixes split/class/task identities")
        require(block["campaign_id"] == campaign["campaign_id"], "job mixes campaign identities")
        require(meta["instance_id"] == member_id, "member report/TRO identity mismatch")
        sidecar_path = artifact(tro_path.with_name(member_id + ".oracle.json"), root, started_at)
        require(read(sidecar_path)["tro_sha256"] == file_digest(tro_path), "member sidecar/TRO mismatch")
        demo_path = artifact(path.parent / "demo.hdf5", root, started_at)
        group = block.get("runtime", {}).get("goal_group", {})
        goal = group.get("requested_goal")
        start = group.get("requested_start")
        require(goal and start, "group member lacks intended goal/start provenance")
        p = block["perturbations"]
        perturbations = Perturbations({n: ScaleState(**v) for n, v in p["scales"].items()}, tuple(p["removed_objects"]), p["lighting"])
        record = {"instance_id": member_id, "tro": str(tro_path), "tro_sha256": file_digest(tro_path),
                          "demo": str(demo_path), "demo_sha256": file_digest(demo_path), "report": str(path),
                          "report_sha256": file_digest(path), "sidecar": str(sidecar_path), "sidecar_sha256": file_digest(sidecar_path)}
        start_key = (digest(goal), digest(start), perturbations.fingerprint)
        if start_key in seen_starts:
            duplicates.append({**record, "reason": "duplicate_start_grasp_configuration"})
            continue
        seen_starts.add(start_key)
        members.append(GroupMember(member_id, block["campaign_id"], key.template_sha256, digest(goal), digest(start),
                                   perturbations, block["created_at"], True))
        artifacts.append(record)
    require(members, "no successfully demonstrated goal-group members")
    allocation = None
    if group_closed:
        if campaign["split"] in ("train", "eval_standard"):
            allocation = allocate_goal_group(members, campaign_id=campaign["campaign_id"], campaign_started_at=started_at,
                                             standard_eval_requested=campaign["split"] == "eval_standard", max_members=3)
        else:
            # Held-out models/categories never enter training, even as singleton fallbacks.
            allocate_goal_group(members, campaign_id=campaign["campaign_id"], campaign_started_at=started_at,
                                 standard_eval_requested=False, max_members=3)
            allocation = {m.instance_id: campaign["split"] for m in members}
    result = {"schema": "isg-release-review-v1", "campaign_id": campaign["campaign_id"], "class_id": campaign["class_id"],
            "task": campaign["task"], "group_id": group_id, "validation_key": asdict(key),
            "template_validation_passed": True, "evidence": [asdict(e) for e in evidence], "members": artifacts,
            "gate": {"path": str(gate_path), "sha256": file_digest(gate_path)}, "source_job": str(job),
            "group_closed": group_closed, "proposed_allocation": allocation, "accepted": False,
            "pending": "quota claim and final dataset export" if allocation else "group completion"}
    if duplicates:
        result["excluded_duplicate_members"] = duplicates
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True)
    p.add_argument("--gate", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--started-at", type=float, required=True)
    p.add_argument("--group-closed", action="store_true")
    p.add_argument("--group-id")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = audit_job(a.job, a.gate, root=a.root, started_at=a.started_at, group_closed=a.group_closed, group_id=a.group_id)
    write_json_atomic(a.output, result)
    print(json.dumps({k: result[k] for k in ("class_id", "group_id", "template_validation_passed", "proposed_allocation", "accepted")}))


if __name__ == "__main__":
    main()
