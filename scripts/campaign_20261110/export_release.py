"""Export only committed campaign allocations and plan immutable rendering shards.

This is a read-only consumer of quota state. Physics action replay evidence is
audited separately from the kinematic replay used to produce observation data.
Source TROs keep their sampling split; the manifest is authoritative for the
final split. A release is visible only after the entire audit has passed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time

from scripts.campaign_20261110.acceptance import artifact, read, report_passed
from scripts.campaign_20261110.contracts import (
    ObjectRef, Template, ValidationEvidence, ValidationKey, audit_splits,
    canonical_json, digest, file_digest, require, require_validation,
    task_quotas, validate_campaign_block, write_json_atomic,
)
from scripts.campaign_20261110.group_lane import check_reservations, physical_template
from scripts.campaign_20261110.storage import atomic_directory_lock

CAMERA_STATES = ("camera_quota.json", "camera_obstructed_quota.json", "camera_category_quota.json")


def snapshot_allocations(root, database=None):
    """Capture a coherent, locked snapshot; never open shared storage in SQLite."""
    root = Path(root).resolve()
    database = Path(database or root / "dataset/allocations.sqlite").resolve()
    require(database.is_relative_to(root / "dataset"), "quota database outside campaign dataset")
    sources = [(database, database.with_suffix(".lockdir"))]
    sources += [(root / "dataset" / name, root / "dataset" / (name + ".lockdir")) for name in CAMERA_STATES]
    rows, groups, inputs = [], {}, []
    with ExitStack() as locks, tempfile.TemporaryDirectory(prefix="isg-release-snapshot-", dir="/tmp") as temporary:
        for _, lock in sorted(sources):
            locks.enter_context(atomic_directory_lock(lock))
        for source, _ in sources:
            if not source.exists():
                inputs.append({"path": str(source), "present": False})
                continue
            local = Path(temporary) / source.name
            shutil.copyfile(source, local)
            inputs.append({"path": str(source), "present": True, "sha256": file_digest(local)})
            if source == database:
                with closing(sqlite3.connect(f"file:{local}?mode=ro", uri=True)) as db:
                    require(db.execute("PRAGMA quick_check").fetchall() == [("ok",)], "quota database corrupt")
                    groups = {gid: json.loads(record) for gid, record in db.execute("SELECT id,record FROM groups")}
                    for iid, gid, split, record in db.execute("SELECT id,group_id,split,record FROM members ORDER BY id"):
                        member = json.loads(record)
                        require(member["instance_id"] == iid and member["group_id"] == gid and member["split"] == split,
                                "database columns disagree with accepted record")
                        require(gid in groups and groups[gid]["accepted"].get(iid) == split,
                                "member has no committed group assignment")
                        require(member in groups[gid]["members"], "committed group/member records disagree")
                        rows.append({**member, "allocation_source": str(source)})
            else:
                state = read(local)
                require(state.get("schema") in ("isg-camera-quota-state-v1", "isg-camera-class-quota-state-v1"),
                        "unknown camera quota schema")
                for iid, member in state["accepted"].items():
                    require(member["instance_id"] == iid, "camera state identity mismatch")
                    rows.append({**member, "task": state["task"], "allocation_source": str(source)})
    return rows, groups, inputs


def template_from(raw):
    return Template(**{**raw, "objects": tuple(ObjectRef(**obj) for obj in raw["objects"])})


def contract_index(root):
    result = {}
    for path in sorted((Path(root) / "groups").glob("*/campaign_config.json")):
        template = template_from(read(path)["template"])
        result[template.fingerprint] = template
    return result


class ArtifactAudit:
    def __init__(self, root, started_at):
        self.root, self.started_at = Path(root).resolve(), started_at
        self.checked = {}

    def check(self, record):
        path = artifact(record["path"], self.root, self.started_at)
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        if self.checked.get(str(path)) != (record["sha256"], signature):
            require(file_digest(path) == record["sha256"], f"accepted artifact changed: {path}")
            after = path.stat()
            require(signature == (after.st_size, after.st_mtime_ns, after.st_ino), "artifact changed during audit")
            self.checked[str(path)] = (record["sha256"], signature)
        return path


def normalize_record(row, *, root, started_at, templates, reservations, reservation_hashes, auditor):
    artifacts = row.get("artifacts") or {kind: {"path": row[kind], "sha256": row[kind + "_sha256"]}
                                        for kind in ("tro", "sidecar", "demo", "report")}
    require({"tro", "sidecar", "demo", "report"} <= set(artifacts), "incomplete accepted artifact record")
    for item in artifacts.values():
        auditor.check(item)
    raw = read(artifacts["tro"]["path"])["ispatialgym"]
    block = raw["campaign"]
    validate_campaign_block(block, object_bindings=raw["object_bindings"])
    require(raw["instance_id"] == row["instance_id"], "accepted TRO identity mismatch")
    require(block["created_at"] >= started_at, "accepted instance predates campaign")
    require(block["split"] == row["sampling_split"] and block["validation_key"] == row["validation_key"],
            "accepted record differs from campaign provenance")
    require(row["split"] == row["sampling_split"] or (row["split"] == "train" and row["sampling_split"] == "eval_standard"),
            "held-out instance leaked into another split")
    template = template_from(row["contract"]) if row.get("contract") else templates.get(block["template_sha256"])
    require(template is not None and template.fingerprint == block["template_sha256"], "missing exact ready template contract")
    require(template.task == block["task"] == row["task"] and template.class_id == block["class_id"], "accepted task/class mismatch")
    check_reservations(template, reservations, reservation_hashes)
    require(read(artifacts["sidecar"]["path"])["tro_sha256"] == artifacts["tro"]["sha256"], "sidecar/TRO hash mismatch")
    report = read(artifacts["report"]["path"])
    report_passed(report, "oracle")
    require(report["instance_id"] == row["instance_id"], "accepted demo report identity mismatch")
    gate = read(auditor.check(row["gate"]))
    require(gate.get("success") is True and gate.get("replay_mode") == "physics_action_open_loop", "missing physics action replay gate")
    auditor.check({"path": gate["instance"], "sha256": gate["tro_sha256"]})
    auditor.check({"path": gate["demo"], "sha256": gate["demo_sha256"]})
    key = ValidationKey(**row["validation_key"])
    snapshot = Path(gate["snapshot"]).resolve()
    require(snapshot.is_relative_to(Path(root).resolve() / "code"), "replay snapshot outside campaign")
    identity = snapshot / ".snapshot_commit"
    if not identity.exists():
        identity = snapshot / ".snapshot_isg"
    auditor.check({"path": str(identity), "sha256": key.code_sha256})
    require(Path(report["provenance"]["isg_root"]).resolve() == snapshot, "demo used a different snapshot")
    evidence = []
    for item in row["evidence"]:
        e = ValidationEvidence(**{**item, "key": ValidationKey(**item["key"])})
        evidence.append(e)
        rep = read(auditor.check({"path": e.report_path, "sha256": e.report_sha256}))
        report_passed(rep, "oracle" if e.kind == "demo" else "replay")
    require_validation(key, evidence, campaign_started_at=started_at)
    report_variant = report.get("variant") or template.class_id
    runtime = block.get("runtime", {})
    # A shared skybox is global. Absolute lighting, actual scales, removals and
    # the exact code/data/physics key all participate in compatibility.
    render_config = {"physical_template": physical_template(template), "validation_key": row["validation_key"],
                     "variant": report_variant, "scene_mode": "recorded", "scene_crop": report.get("scene_crop"),
                     "perturbations": block["perturbations"], "augmentation": runtime.get("augmentation", {})}
    result = {"instance_id": row["instance_id"], "campaign_id": block["campaign_id"], "split": row["split"],
              "sampling_split": row["sampling_split"], "task": template.task, "class_id": template.class_id,
              "template": asdict(template), "physical_template_sha256": physical_template(template),
              "group_id": row.get("group_id"), "bucket": row.get("bucket"),
              "room": f"{template.scene}/{template.room}", "variant": report_variant,
              "artifacts": artifacts, "validation_key": row["validation_key"], "gate": row["gate"],
              "evidence": row["evidence"], "allocation_source": row["allocation_source"],
              "perturbations": block["perturbations"], "runtime_augmentation": runtime.get("augmentation", {}),
              "render_compatibility_sha256": digest(render_config)}
    return result, template


def audit_assignments(rows, groups, templates):
    require(rows, "no accepted instances to export")
    require(len({row["instance_id"] for row in rows}) == len(rows), "duplicate instance across authoritative allocators")
    require(len({row["campaign_id"] for row in rows}) == 1, "mixed campaigns in export")
    grouped = defaultdict(list)
    for row in rows:
        if row["group_id"]:
            grouped[row["group_id"]].append(row)
        else:
            require(row["task"].startswith("camera_relocalization-"), "non-camera instance lacks a goal group")
    for gid, members in grouped.items():
        require(gid in groups and len(members) <= 3, "missing or oversized accepted goal group")
        require(groups[gid]["accepted"] == {m["instance_id"]: m["split"] for m in members}, "incomplete accepted group export")
        if any(m["split"] == "eval_standard" for m in members):
            require(sum(m["split"] == "eval_standard" for m in members) == 1 and any(m["split"] == "train" for m in members),
                    "standard evaluation lost its successful training sibling")
        require(len({canonical_json(m["perturbations"]) for m in members}) == 1, "group perturbations differ")
    # The same physical contract can be rebound to a new snapshot; use its hash
    # as identity for the independent cross-role split-leakage audit.
    errors = audit_splits(replace(t, template_id=t.fingerprint) for t in templates)
    require(not errors, "global split leakage: " + "; ".join(errors))
    quotas = {q.task: q for q in task_quotas()}
    counts = Counter((m["split"], m["task"], m["task"] if m["split"] == "eval_category" else m["class_id"]) for m in rows)
    for (split, task, _), count in counts.items():
        q = quotas[task]
        cap = {"eval_standard": 10, "eval_model": q.model_instances_per_class,
               "eval_category": q.category_instances, "train": q.train_instances // q.standard_classes}[split]
        require(count <= cap, "accepted class/task exceeds quota")
    for task, q in quotas.items():
        require(sum(m["split"] == "train" and m["task"] == task for m in rows) <= q.train_instances, "training task exceeds quota")
    pinned = defaultdict(set)
    buckets = Counter()
    for row in rows:
        if row["split"] != "train":
            scope = row["task"] if row["split"] == "eval_category" else row["class_id"]
            pinned[row["split"], row["task"], scope].add(row["physical_template_sha256"])
            q = quotas[row["task"]]
            limits = getattr(q, {"eval_standard": "standard_buckets", "eval_model": "model_buckets",
                                 "eval_category": "category_buckets"}[row["split"]])
            if limits:
                bucket = row["bucket"]
                if row["task"].startswith("camera_relocalization-"):
                    require(bucket in ("3", "5", "10"), "unknown camera evaluation bucket")
                    bucket = ("3", "5", "10").index(bucket)
                require(isinstance(bucket, int) and 0 <= bucket < len(limits), "unknown evaluation bucket")
                key = (row["split"], row["task"], scope, bucket)
                buckets[key] += 1
                require(buckets[key] <= limits[bucket], "accepted evaluation bucket exceeds quota")
    require(all(len(values) == 1 for values in pinned.values()), "evaluation class changed physical template")


def publish(root, tag, rows, groups, inputs, *, shard_size=32):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag) is not None, "unsafe release tag")
    require(isinstance(shard_size, int) and shard_size > 0, "invalid shard size")
    releases = Path(root).resolve() / "dataset/releases"
    releases.mkdir(parents=True, exist_ok=True)
    destination = releases / tag
    require(not destination.exists(), "release already exists; use a new immutable tag")
    temporary = Path(tempfile.mkdtemp(prefix="." + tag + ".", dir=releases))
    try:
        ordered = sorted(rows, key=lambda row: row["instance_id"])
        (temporary / "instances.jsonl").write_text("".join(canonical_json(row) + "\n" for row in ordered))
        by_key = defaultdict(list)
        for row in ordered:
            by_key[row["render_compatibility_sha256"]].append(row)
        shards = []
        for key, values in sorted(by_key.items()):
            for offset in range(0, len(values), shard_size):
                batch = values[offset:offset + shard_size]
                sid = f"{key[:20]}_{offset // shard_size:04d}"
                listing = temporary / "shards" / (sid + ".txt")
                listing.parent.mkdir(exist_ok=True)
                listing.write_text("".join(str(Path(row["artifacts"]["demo"]["path"]).parent) + "\n" for row in batch))
                shards.append({"id": sid, "compatibility_sha256": key, "variant": batch[0]["variant"],
                               "instances": [m["instance_id"] for m in batch], "list": str(destination / "shards" / listing.name),
                               "list_sha256": file_digest(listing), "scene_mode": "recorded", "slots": 1,
                               "purpose": "kinematic_observation_rendering"})
        write_json_atomic(temporary / "shards.json", {"schema": "isg-render-shards-v1", "shards": shards})
        write_json_atomic(temporary / "groups.json", groups)
        summary = {"schema": "isg-dataset-release-v1", "campaign_id": ordered[0]["campaign_id"], "created_at": time.time(),
                   "source_states": inputs, "instances": len(rows), "counts": dict(Counter(row["split"] for row in rows)),
                   "tasks": {task: dict(Counter(row["split"] for row in rows if row["task"] == task)) for task in sorted({r["task"] for r in rows})},
                   "physical_templates": len({r["physical_template_sha256"] for r in rows}), "render_shards": len(shards),
                   "files": {name: file_digest(temporary / name) for name in ("instances.jsonl", "shards.json", "groups.json")},
                   "rendered": False, "split_authority": "instances.jsonl split; source TRO sampling_split remains unchanged"}
        write_json_atomic(temporary / "release.json", summary)
        os.rename(temporary, destination)
        return destination, summary
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "tag", "reservations", "reservation-hashes"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--started-at", type=float, required=True)
    p.add_argument("--shard-size", type=int, default=32)
    a = p.parse_args()
    source, groups, inputs = snapshot_allocations(a.root)
    contracts = contract_index(a.root)
    reservations, hashes = read(a.reservations), read(a.reservation_hashes)
    inputs += [{"path": str(Path(path).resolve()), "sha256": file_digest(path)} for path in (a.reservations, a.reservation_hashes)]
    auditor = ArtifactAudit(a.root, a.started_at)
    rows, templates = [], []
    for row in source:
        normalized, template = normalize_record(row, root=a.root, started_at=a.started_at, templates=contracts,
            reservations=reservations, reservation_hashes=hashes, auditor=auditor)
        rows.append(normalized)
        templates.append(template)
    audit_assignments(rows, groups, templates)
    destination, summary = publish(a.root, a.tag, rows, groups, inputs, shard_size=a.shard_size)
    print(json.dumps({"release": str(destination), "instances": summary["instances"], "counts": summary["counts"], "shards": summary["render_shards"]}))


if __name__ == "__main__":
    main()
