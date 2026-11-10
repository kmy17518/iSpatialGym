#!/usr/bin/env python3
"""Claim audited goal groups and publish immutable final split assignments.

TROs and demos retain their verified collection hashes. Dataset consumers use
the exported assignment's `split`; `sampling_split` records the collection lane.
SQLite runs locally under an atomic directory lock, then publishes by replacement.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from scripts.campaign_20261110.acceptance import audit_job, read
from scripts.campaign_20261110.contracts import (
    Perturbations, ScaleState, canonical_json, digest, file_digest, require,
    task_quotas, write_json_atomic,
)
from scripts.campaign_20261110.goal_groups import load_template_contract
from scripts.campaign_20261110.storage import atomic_directory_lock, local_sqlite_transaction
from scripts.campaign_20261110.reservation_policy import require_active_policy
from scripts.campaign_20261110.goal_reuse_policy import POLICY as GOAL_REUSE_POLICY, require_database_policy
from scripts.campaign_20261110.binding_pools import (
    audit_objects, canonical_pool_perturbations, conceptual_identity, load_pool_for_template,
)

QUOTAS = {q.task: q for q in task_quotas()}
ROOM_WEIGHTED_TASKS = {"object_scaling", "mental_rotation"}


def training_room_policy(plan, task):
    """Freeze ten actual room areas and integer caps by largest remainder."""
    require(task in ROOM_WEIGHTED_TASKS, "room policy requires a held-object task")
    rows = plan["tasks"][task]
    require(plan.get("schema") == "isg-room-plan-v1" and plan["rooms_per_task"] == len(rows) == 10,
            "held training requires the frozen ten-room plan")
    weights = {f"{row['scene']}/{row['room']}": float(row["traversable_m2"]) for row in rows}
    reserved = {f"{row['scene']}/{row['room']}" for row in plan.get("reserved_rooms", [])}
    require(len(weights) == 10 and all(math.isfinite(v) and v > 0 for v in weights.values()), "invalid room traversal areas")
    require(not set(weights) & reserved, "held training plan includes a reserved room")
    total = QUOTAS[task].train_instances
    real = {room: total*area/sum(weights.values()) for room,area in weights.items()}
    caps = {room: math.floor(value) for room,value in real.items()}
    for room in sorted(weights,key=lambda key: (-(real[key]-caps[key]),key))[:total-sum(caps.values())]:
        caps[room] += 1
    require(all(n > 0 for n in caps.values()), "training room received no capacity")
    return {"method":"traversable_area_largest_remainder", "rooms_per_task":10,
            "traversable_m2":weights,"caps":caps,"total":total}


def accepted_room(record):
    """Read old accepted records without rewriting their verified TRO/export."""
    if record.get("room"):
        return record["room"]
    require(file_digest(record["tro"]) == record["tro_sha256"], "previously accepted TRO changed")
    scene = read(record["tro"])["ispatialgym"].get("scene", {})
    rooms = scene.get("task_rooms", [])
    require(scene.get("model") and len(rooms) == 1, "legacy training record lacks an unambiguous room")
    return f"{scene['model']}/{rooms[0]}"


def physical_template(template):
    """Task template identity excludes code paths and provenance timestamps."""
    value = {"task": template.task, "scene": template.scene, "room": template.room,
                   "rooms": sorted(set(template.scene_setup.get("task_rooms", ())) | {template.room}),
                   "objects": [{key: value for key, value in asdict(obj).items()
                                if key not in ("readiness_source", "ready")}
                               for obj in sorted(template.objects, key=lambda obj: obj.instance)],
                   "bindings": template.scene_setup.get("inst_to_name", {})}
    if template.scene_setup.get("camera_context") is not None:
        from scripts.campaign_20261110.camera_context import load_camera_context
        refs = load_camera_context(template)
        value["camera_context"] = {"scene_sha256": template.scene_setup["camera_context"]["scene"]["sha256"],
            "objects": [{key:item for key,item in asdict(obj).items() if key not in ("readiness_source","ready")}
                        for obj in sorted(refs,key=lambda obj:obj.instance)]}
    return digest(value)


def conceptual_template(template):
    """Count a maximum pool once; never use this identity for physics gating."""
    return conceptual_identity(template) or physical_template(template)


def check_reservations(template, reservations, hashes):
    pool = load_pool_for_template(template)
    if pool is not None:
        # Inactive preloaded candidates must also respect every exclusion.
        check_reservations(pool.audit_template(), reservations, hashes)
    require(hashes.get("schema") == "campaign-reserved-asset-hashes-v1", "missing reserved asset-byte audit")
    require(set(hashes["reserved_model_keys"]) == set(reservations["train_exclude_models"]), "reserved asset audit has different model identities")
    ready = hashes["ready_assets"]
    require(set(ready) | set(hashes["pending_unready_model_keys"]) == set(hashes["reserved_model_keys"]), "reserved asset audit is incomplete")
    require(all(value.get("ready") is True for value in ready.values()), "reserved ready-asset audit contains an unready asset")
    require(set(hashes["reserved_asset_hashes"]) == {value["asset_sha256"] for value in ready.values()}, "reserved byte hashes are incomplete")
    rooms = {(template.scene, room) for room in set(template.scene_setup.get("task_rooms", ())) | {template.room}}
    held_rooms = {(row["scene"], row["room"]) for row in reservations["category_holdout_rooms"]}
    pending = reservations.get("pending_custom_hard_reservations_not_ready", [])
    model_pool = set(reservations["model_holdout_models"]) | {
        row["category"]+"/"+row["model"] for row in pending if row["split"] == "eval_model"}
    category_pool = set(reservations["category_holdout_models"]) | {
        row["category"]+"/"+row["model"] for row in pending if row["split"] == "eval_category"}
    categories = set(reservations["category_holdout_categories"])
    require(bool(rooms & held_rooms) if template.split == "eval_category" else not rooms & held_rooms,
            "template violates held-out room reservation")
    for obj in audit_objects(template):
        require(obj.ready and obj.readiness_source, "unready model in release")
        if obj.semantic_kind in ("wall", "floor"):
            continue
        if template.split in ("train", "eval_standard"):
            require(obj.model_key not in reservations["train_exclude_models"], "held-out model in held-in release")
            require(obj.asset_sha256 not in hashes["reserved_asset_hashes"], "held-out asset bytes in held-in release")
            require(obj.category not in categories or obj.semantic_kind in ("tray", "board"), "held-out category in held-in release")
        elif template.split == "eval_model":
            require(obj.model_key in model_pool and obj.model_key not in category_pool, "unreserved model in model holdout")
            require(obj.category not in categories or obj.semantic_kind in ("tray", "board"), "category holdout in model holdout")
        else:
            require(obj.model_key in category_pool, "unreserved model in category holdout")
            require(obj.category in categories or obj.semantic_kind in ("tray", "board"), "held-in category in category holdout")
        if template.split in ("eval_model", "eval_category"):
            require(obj.model_key in ready and ready[obj.model_key]["asset_sha256"] == obj.asset_sha256,
                    "held-out model bytes differ from its ready reservation audit")


def bucket_index(template, group, *, strict):
    task = template.task
    if task == "object_scaling":
        value, edges = group.get("actual_magnitude"), (.05, .10, .15, .30)
    elif task == "mental_rotation":
        value, edges = group.get("actual_magnitude"), (30, 45, 75, 120)
    elif task == "disembedding":
        value, edges = group.get("actual_crowdedness"), (0, .30, .45, 1)
    elif task.startswith("configuration_matching-articulation_"):
        value, edges = group.get("actual_open_fraction"), (0, .5, 1)
    elif task.startswith("alignment-"):
        value = group.get("actual_magnitude_deg")
        edges = template.scene_setup.get("bucket_policy", {}).get("edges", ())
        require(len(edges) == 4, "alignment needs three pinned native-range buckets")
    elif task == "configuration_matching-block_design":
        value = group.get("actual_flips")
        if not strict and isinstance(value, int) and not isinstance(value, bool) and value == 0:
            return None
        require(isinstance(value, int) and 1 <= value <= 4, "block group lacks verified flip count")
        return value - 1
    else:
        require(not QUOTAS[task].standard_buckets, "missing strict bucket adapter")
        return 0
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), "missing actual bucket metric")
    require(all(math.isfinite(x) for x in edges) and all(a < b for a, b in zip(edges, edges[1:])), "invalid bucket edges")
    if not edges[0] <= value <= edges[-1]:
        require(not strict, "eval metric lies outside task buckets")
        return None
    # Crowdedness explicitly assigns .30/.45 to the upper bucket; other tasks
    # assign a shared boundary to the lower bucket, including exactly50% open.
    return min(sum(value >= edge if task == "disembedding" else value > edge for edge in edges[1:-1]), len(edges)-2)


def _canonical_goal(value):
    if isinstance(value, dict):
        return {key: _canonical_quat(item) if key == "orientation" else _canonical_goal(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and all(isinstance(item, (tuple, list)) for item in value) and len(value[0]) == 3 and len(value[1]) == 4:
            return [_canonical_goal(value[0]), _canonical_quat(value[1])]
        return [_canonical_goal(item) for item in value]
    return round(value, 6) if isinstance(value, float) else value


def _canonical_quat(value):
    require(len(value) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in value), "invalid goal quaternion")
    norm = math.sqrt(sum(v*v for v in value))
    require(norm > 0, "zero goal quaternion")
    sign = next((1 if v > 0 else -1 for v in reversed(value) if abs(v) > 1e-12), 1)
    return [round(sign*v/norm, 6) for v in value]


def _db(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS groups (id TEXT PRIMARY KEY, goal TEXT NOT NULL, review TEXT NOT NULL, record TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS members (id TEXT PRIMARY KEY, group_id TEXT NOT NULL, split TEXT NOT NULL,
            task TEXT NOT NULL, class_id TEXT NOT NULL, bucket INTEGER, template TEXT NOT NULL, record TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS quotas ON members (split, task, class_id, bucket);
        CREATE TABLE IF NOT EXISTS assets (split TEXT NOT NULL, model TEXT NOT NULL, sha TEXT NOT NULL,
            category TEXT NOT NULL, kind TEXT NOT NULL, PRIMARY KEY(split,model,sha,kind));
    """)
    try:
        require_database_policy(db)
    except BaseException:
        db.close()
        raise
    return db


def _quota(q, split):
    if split == "eval_standard":
        return 10, q.standard_buckets
    if split == "eval_model":
        return q.model_instances_per_class, q.model_buckets
    return q.category_instances, q.category_buckets


def _allocate(rows, *, split, q, eval_counts, train_count, train_capacity=None):
    """Whole standard groups preserve a train sibling; train buckets stay soft."""
    total, buckets = _quota(q, split) if split != "train" else (0, ())
    def space(row):
        return sum(eval_counts.values()) < total and (not buckets or eval_counts[row["bucket"]] < buckets[row["bucket"]])
    result = {}
    if split in ("eval_model", "eval_category"):
        for row in rows:
            if space(row):
                result[row["instance_id"]] = split
                eval_counts[row["bucket"]] += 1
        return result
    selected = next((row for row in rows if space(row)), None) if split == "eval_standard" and len(rows) >= 2 else None
    available = q.train_instances-train_count
    if train_capacity is not None:
        available = min(available,train_capacity)
    if available <= 0:
        return {}  # Never publish an eval member without its training sibling.
    result = {selected["instance_id"]:"eval_standard"} if selected is not None else {}
    for row in rows:
        if row is selected:
            continue
        if available <= 0:
            break
        result[row["instance_id"]] = "train"
        available -= 1
    return result


def claim_group(review, template, reservations, *, reservation_hashes, root, started_at, state_path, training_room_plan=None):
    root, state_path = Path(root).resolve(), Path(state_path).resolve()
    require(state_path.is_relative_to(root), "quota database outside campaign")
    require(review.get("group_closed") is True, "cannot allocate an open goal group")
    # Re-read the actual reports, source hashes and gate rather than trusting a
    # previously generated review file or a controller's claimed success.
    fresh = audit_job(review["source_job"], review["gate"]["path"], root=root, started_at=started_at,
                      group_closed=True, group_id=review["group_id"])
    require(digest(fresh) == digest(review), "release evidence changed after review")
    require(review["validation_key"]["template_sha256"] == template.fingerprint, "release contract differs from validated template")
    pool = load_pool_for_template(template, root=root)
    check_reservations(template, reservations, reservation_hashes)
    conceptual = conceptual_template(template)
    room = f"{template.scene}/{template.room}"
    room_policy = None
    if template.task in ROOM_WEIGHTED_TASKS and template.split in ("eval_standard", "train"):
        if training_room_plan is None:
            path = root/"manifests/train_room_plan_20261110.json"
            require(path.is_file(), "held training release requires the frozen ten-room plan")
            training_room_plan = read(path)
        room_policy = training_room_policy(training_room_plan,template.task)
        require(room in room_policy["caps"], "held training room outside frozen ten-room plan")
    gid = review["group_id"]
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", gid), "unsafe group id")
    rows, goals = [], []
    for member in review["members"]:
        raw = read(member["tro"])["ispatialgym"]
        block = raw["campaign"]
        require(block["split"] == template.split, "release split differs from contract")
        group = block["runtime"]["goal_group"]
        p = block["perturbations"]
        perturb = Perturbations({n: ScaleState(**v) for n, v in p["scales"].items()}, tuple(p["removed_objects"]), p["lighting"])
        perturb.validate(template.split, p["scales"])
        bucket = bucket_index(template, group, strict=template.split != "train")
        goals.append(digest({"template": conceptual if pool else physical_template(template),
                             "goal": _canonical_goal(group["requested_goal"]),
                             "base": _canonical_goal(group.get("base_pose2d")),
                             "perturbations": canonical_pool_perturbations(template, perturb) if pool else asdict(perturb)}))
        rows.append({**member, "bucket": bucket, "sampling_split": template.split})
    require(len(set(goals)) == 1, "goal group changes goal/base/template/perturbations")
    policy = {key: reservations[key] for key in ("model_holdout_models", "category_holdout_models", "category_holdout_categories",
              "category_holdout_rooms", "train_exclude_models", "pending_custom_hard_reservations_not_ready")}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    def transact(local_path):
        db = _db(local_path)
        try:
            with db:
                for key, value in (("campaign", review["campaign_id"]), ("reservations", digest(policy))):
                    previous = db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
                    require(previous is None or previous[0] == value, "campaign or holdout reservations changed after release")
                    db.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)", (key, value))
                if room_policy is not None:
                    name = "training_rooms:"+template.task
                    previous = db.execute("SELECT value FROM metadata WHERE key=?",(name,)).fetchone()
                    require(previous is None or previous[0] == canonical_json(room_policy), "training room allocation policy changed")
                    db.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)",(name,canonical_json(room_policy)))
                previous = db.execute("SELECT review,record FROM groups WHERE id=?", (gid,)).fetchone()
                if previous:
                    require(previous[0] == digest(review), "accepted group changed")
                    return json.loads(previous[1])
                split, task = template.split, template.task
                scope = task if split == "eval_category" else template.class_id
                counts = Counter(dict(db.execute("SELECT bucket,COUNT(*) FROM members WHERE split=? AND task=? AND class_id=? GROUP BY bucket",
                                                  (split, task, scope))))
                train_count = db.execute("SELECT COUNT(*) FROM members WHERE split='train' AND task=?", (task,)).fetchone()[0]
                class_count = db.execute("SELECT COUNT(*) FROM members WHERE split='train' AND task=? AND class_id=?",
                                         (task,template.class_id)).fetchone()[0]
                class_cap = QUOTAS[task].train_instances//QUOTAS[task].standard_classes
                capacity = class_cap-class_count
                if room_policy is not None:
                    records = [json.loads(row[0]) for row in db.execute("SELECT record FROM members WHERE split='train' AND task=?",(task,))]
                    room_counts = Counter(accepted_room(row) for row in records)
                    require(set(room_counts) <= set(room_policy["caps"]), "accepted training data includes a room outside frozen plan")
                    require(all(room_counts[key] <= cap for key,cap in room_policy["caps"].items()), "accepted training data already exceeds room cap")
                    capacity = min(capacity,room_policy["caps"][room]-room_counts[room])
                allocation = _allocate(rows, split=split, q=QUOTAS[task], eval_counts=counts, train_count=train_count,train_capacity=capacity)
                if not allocation:
                    return {"accepted": {}, "deferred": [row["instance_id"] for row in rows], "reason": "quota full"}
                physical = physical_template(template)
                for obj in audit_objects(template, root=root):
                    if obj.semantic_kind in ("wall", "floor"):
                        continue
                    for old_split, model, sha, category, kind in db.execute(
                            "SELECT split,model,sha,category,kind FROM assets WHERE model=? OR sha=? OR category=?",
                            (obj.model_key, obj.asset_sha256, obj.category)):
                        if old_split == split or {old_split, split} <= {"train", "eval_standard"}:
                            continue
                        require(model != obj.model_key and sha != obj.asset_sha256, "asset model/hash leaked across held-out splits")
                        if "eval_category" in (old_split, split):
                            require(category != obj.category or obj.semantic_kind in ("tray", "board") or kind in ("tray", "board"),
                                    "asset category leaked across held-out splits")
                    db.execute("INSERT OR IGNORE INTO assets VALUES (?,?,?,?,?)",
                               (split, obj.model_key, obj.asset_sha256, obj.category, obj.semantic_kind))
                for row in rows:
                    iid = row["instance_id"]
                    require(db.execute("SELECT id FROM members WHERE id=?", (iid,)).fetchone() is None, "instance already allocated")
                    if iid not in allocation:
                        continue
                    final = allocation[iid]
                    if final.startswith("eval_"):
                        templates = {v[0] for v in db.execute("SELECT DISTINCT template FROM members WHERE split=? AND task=? AND class_id=?",
                                                            (final, task, scope))}
                        require(not templates or templates == {conceptual}, "evaluation class changed its pinned conceptual template")
                    record = {**row, "split": final, "group_id": gid, "task": task, "class_id": template.class_id,
                              "room":room,"validation_key": review["validation_key"], "gate": review["gate"], "evidence": review["evidence"],
                              "physical_template_sha256": physical, "conceptual_template_sha256": conceptual,
                              "binding_pool": template.scene_setup.get("binding_pool")}
                    db.execute("INSERT INTO members VALUES (?,?,?,?,?,?,?,?)",
                               (iid, gid, final, task, scope, row["bucket"], conceptual, canonical_json(record)))
                record = {"schema": "isg-group-assignment-v1", "campaign_id": review["campaign_id"], "group_id": gid,
                          "accepted": allocation, "deferred": [row["instance_id"] for row in rows if row["instance_id"] not in allocation],
                          "canonical_goal_sha256": goals[0], "release_review_sha256": digest(review),
                          "conceptual_template_sha256": conceptual, "physical_template_sha256": physical,
                          "pool_sha256": pool.fingerprint if pool else None,
                          "goal_reuse_policy": GOAL_REUSE_POLICY,
                          "training_policy": {"per_class_cap":class_cap,"room_policy":room_policy},
                          "members": [json.loads(value[0]) for value in db.execute("SELECT record FROM members WHERE group_id=? ORDER BY id", (gid,))]}
                destination = root / "dataset/groups" / (gid + ".json")
                if destination.exists():
                    require(read(destination) == record, "orphan group export has different content")
                else:
                    write_json_atomic(destination, record)
                db.execute("INSERT INTO groups VALUES (?,?,?,?)", (gid, goals[0], digest(review), canonical_json(record)))
            return record
        finally:
            db.close()

    with atomic_directory_lock(state_path.with_suffix(".lockdir")):
        require_active_policy(root,reservations)
        with local_sqlite_transaction(state_path) as local_path:
            record = transact(local_path)
            db = _db(local_path)
            try:
                counts = {split: count for split, count in db.execute("SELECT split,COUNT(*) FROM members GROUP BY split")}
                summary = {"campaign_id": review["campaign_id"], "counts": counts,
                           "goal_reuse_policy": GOAL_REUSE_POLICY,
                           "groups": db.execute("SELECT COUNT(*) FROM groups").fetchone()[0], "database": str(state_path),
                           "conceptual_templates": db.execute("SELECT COUNT(DISTINCT template) FROM members").fetchone()[0],
                           "physically_validated_variants": len({json.loads(record).get("physical_template_sha256", identity)
                               for identity,record in db.execute("SELECT template,record FROM members")}),
                           "training_room_policies": {key.split(":",1)[1]:json.loads(value) for key,value in db.execute(
                               "SELECT key,value FROM metadata WHERE key LIKE 'training_rooms:%'")}}
            finally:
                db.close()
        # Counts cannot advertise a transaction until its database is durable.
        write_json_atomic(root / "dataset/group_counts.json", summary)
        return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("review", "contracts", "reservations", "reservation-hashes", "root", "state"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--started-at", type=float, required=True)
    a = p.parse_args()
    review = read(a.review)
    hashes = read(a.reservation_hashes)
    require(hashes["reservation_sha256"] == file_digest(a.reservations), "reservation file changed after asset audit")
    result = claim_group(review, load_template_contract(a.contracts, review["class_id"]), read(a.reservations),
                         reservation_hashes=hashes, root=a.root, started_at=a.started_at, state_path=a.state)
    print(json.dumps({key: result[key] for key in ("accepted", "deferred")}))


if __name__ == "__main__":
    main()
