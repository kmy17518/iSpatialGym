"""Continue audited pair, surface, tray and joint groups through eval and training.

The held controller supplies exact-key reconciliation, replay auditing, fresh
index reservation, atomic state, bounded concurrency and never-started migration.
This adapter changes only task quota selection and fresh group authoring.
"""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import shutil
import sqlite3
import tempfile

from scripts.campaign_20261110 import controller_held as held
from scripts.campaign_20261110.contracts import file_digest, require, task_quotas, write_json_atomic
from scripts.campaign_20261110.controller_families import read, reserve_indices

QUOTAS = {q.task: q for q in task_quotas()}


def supports(task):
    return (task in ("alignment-world_vertical", "alignment-axial", "disembedding")
            or task.startswith("configuration_matching-articulation_")
            or task.startswith("inter_object_relation_satisfaction-")) and task in QUOTAS


def quotas(template, split=None):
    q, split = QUOTAS[template.task], split or template.split
    if split == "train":
        return q.standard_buckets or (10,)
    values = {"eval_standard": q.standard_buckets, "eval_model": q.model_buckets,
              "eval_category": q.category_buckets}[split]
    total = {"eval_standard": 10, "eval_model": q.model_instances_per_class,
             "eval_category": q.category_instances}[split]
    require(total is not None and total > 0, "task quota remains unresolved")
    return values or (total,)


def ranges(template):
    task = template.task
    if task.startswith("alignment-"):
        edges = template.scene_setup.get("bucket_policy", {}).get("edges", ())
        require(len(edges) == 4 and all(a < b for a, b in zip(edges, edges[1:])), "missing native alignment bucket policy")
    elif task == "disembedding":
        edges = (0, .30, .45, 1)
    elif task.startswith("configuration_matching-articulation_"):
        edges = (0, .5, 1)
    else:
        edges = (0, 1)
    return tuple(zip(edges, edges[1:]))


def read_counts(root, template):
    result = {"evaluation": Counter(), "train_buckets": Counter(), "class_remaining": 1000,
              "task_remaining": QUOTAS[template.task].train_instances, "evaluation_sibling_reserve": 10}
    source = root / "dataset/allocations.sqlite"
    if not source.exists():
        return result
    with tempfile.TemporaryDirectory(prefix="isg-group-counts-", dir="/tmp") as temporary:
        local = Path(temporary) / "state.sqlite"
        shutil.copyfile(source, local)
        with sqlite3.connect(f"file:{local}?mode=ro", uri=True) as db:
            scope = template.task if template.split == "eval_category" else template.class_id
            result["evaluation"] = Counter(dict(db.execute(
                "SELECT bucket,COUNT(*) FROM members WHERE split=? AND task=? AND class_id=? GROUP BY bucket",
                (template.split, template.task, scope))))
            result["train_buckets"] = Counter(dict(db.execute(
                "SELECT bucket,COUNT(*) FROM members WHERE split='train' AND task=? AND class_id=? GROUP BY bucket",
                (template.task, template.class_id))))
            result["class_remaining"] -= sum(result["train_buckets"].values())
            result["task_remaining"] -= db.execute("SELECT COUNT(*) FROM members WHERE split='train' AND task=?", (template.task,)).fetchone()[0]
            standard = db.execute("SELECT COUNT(*) FROM members WHERE split='eval_standard' AND task=? AND class_id=?",
                                  (template.task, template.class_id)).fetchone()[0]
            result["evaluation_sibling_reserve"] = max(0, 10 - standard)
    return result


def pending(root):
    result = []
    for path in sorted((root / "controllers").glob("*/state.json")):
        for state in read(path).get("seeds", {}).values():
            for record in state.get("queues", []):
                if record.get("status") == "complete":
                    continue
                ids = [gid for gid in record["prepared"]["group_ids"] if gid not in record.get("groups", {})]
                if not ids:
                    continue
                template = read(root / "groups" / ids[0] / "campaign_config.json")["template"]
                if not supports(template["task"]):
                    continue
                require(record.get("decision") is not None, "generic pending queue lacks its quota decision")
                result.append({"task": template["task"], "class_id": template["class_id"], "phase": record["phase"],
                    "bucket": record["decision"]["bucket"], "groups": len(ids), "state": str(path)})
    return result


def reserve_pending_counts(counts, template, records):
    result = {**counts, "evaluation": Counter(counts["evaluation"]), "train_buckets": Counter(counts["train_buckets"])}
    for row in records:
        if row["task"] != template.task:
            continue
        n, phase = 3 * row["groups"], row["phase"]
        same = row["class_id"] == template.class_id
        if phase == template.split and (same or phase == "eval_category"):
            result["evaluation"][row["bucket"]] += row["groups"] if phase == "eval_standard" else n
        if phase in ("train", "eval_standard"):
            result["task_remaining"] = max(0, result["task_remaining"] - n)
            if same:
                result["class_remaining"] = max(0, result["class_remaining"] - n)
                result["train_buckets"][row["bucket"]] += n
    return result


def next_work(template, counts, *, max_groups=6, training_validated=False, mode="eval_then_train"):
    require(mode in ("eval_then_train", "train_only"), "unknown group controller phase policy")
    split = template.split
    if split != "train" and mode != "train_only":
        deficits = [max(0, n - counts["evaluation"].get(i, 0)) for i, n in enumerate(quotas(template))]
        if any(deficits):
            bucket = max(range(len(deficits)), key=lambda i: (deficits[i], -i))
            groups = min(max_groups, deficits[bucket] if split == "eval_standard" else (deficits[bucket] + 2) // 3)
            if split == "eval_standard":
                groups = min(groups, counts["class_remaining"], counts["task_remaining"])
            return {"phase": split, "bucket": bucket, "groups": groups, "needs_new_gate": False} if groups else None
    if split not in ("train", "eval_standard"):
        return None
    reserve = counts.get("evaluation_sibling_reserve", 0)
    spare = min(max(0, counts["class_remaining"] - reserve), counts["task_remaining"])
    if spare <= 0:
        return None
    # Another controller can still be collecting standard evaluation. Every
    # outstanding eval needs a fresh training sibling in this class. Pending
    # groups already reserve their full three candidates, conservatively.
    available_groups = spare // 3 if reserve else (spare + 2) // 3
    if not available_groups:
        return None
    weights = quotas(template, "train")
    bucket = min(range(len(weights)), key=lambda i: (counts["train_buckets"].get(i, 0) / weights[i], i))
    return {"phase": "train", "bucket": bucket, "groups": min(max_groups, available_groups) if training_validated else 1,
            "needs_new_gate": not training_validated}


def prepare_queue(spec, state, decision, cfg, root, template, save):
    from scripts.campaign_20261110.goal_groups import successful_anchor, read_env
    from scripts.campaign_20261110.goal_batches import prepare_batch, submission_command
    from scripts.campaign_20261110.group_lane import check_reservations
    from scripts.campaign_20261110.reservation_policy import active_policy
    phase = decision["phase"]
    if phase == "train" and template.split != "train":
        template = replace(template, split="train", template_id=template.template_id + "__train")
    policy = active_policy(root)
    require(policy is not None, "generic queue requires active reservations")
    check_reservations(template, policy[1], policy[2])
    anchor = successful_anchor(spec["anchor_job"], campaign_root=root, ledger=spec["anchor_ledger"], campaign_started_at=cfg["started_at"])
    snapshot = Path(spec["snapshot"])
    number = len(state["queues"]) + 1
    name = "group_auto_" + spec["id"] + "_" + str(number).zfill(4)
    seed_config = read(spec["seed_config"])
    key = seed_config["validation_key"]
    require(file_digest(snapshot / ".snapshot_commit") == key["code_sha256"], "generic source code identity changed")
    prepared = prepare_batch(anchor=anchor, template=template, campaign_root=root, batch_id=name,
        count=decision["groups"], magnitude_range=ranges(template)[decision["bucket"]],
        seed=cfg["seed"] + number * 1009 + spec["seed_offset"], campaign_id=cfg["campaign_id"],
        code_sha256=key["code_sha256"], physics_sha256=key["physics_sha256"], data_sha256=key["data_sha256"],
        budget_seconds=spec.get("group_budget_seconds", 900))
    actual = read(prepared["config"])["validation_key"]
    if phase == seed_config["split"]:
        require(actual == key, "generic author changed an already validated source key")
    if phase in state.setdefault("phase_keys", {}):
        require(state["phase_keys"][phase] == actual, "generic phase changed its immutable key")
    state["phase_keys"][phase] = actual
    index = reserve_indices(root, name, prepared["candidate_instances"])
    original_env = read_env(Path(spec["anchor_job"]) / "job.env")
    names = ("KEEP_SCENE", "START_PER_ATTEMPT", "ISG_CUSTOMS", "ISPATIALGYM_CAMPAIGN_CUSTOMS_SHA256", "ISPATIALGYM_STABLE_SCENE_DIR")
    extras = [name + "=" + original_env[name] for name in names if original_env.get(name)]
    extras += ["ISPATIALGYM_MAP_OVERLAY_RELEASE=3"]
    command = submission_command(prepared["queue"], snapshot=snapshot, campaign_root=root, v2_root=spec["v2_root"],
        base_index=index, extra_env=extras)
    if original_env.get("START_PER_ATTEMPT") == "0":
        command += ["--customs", "template"]
    record = {"queue": prepared["queue"], "prepared": prepared, "directory": str(Path(prepared["queue"]).parent),
        "validation_key": actual, "phase": phase, "index": index, "count": prepared["candidate_instances"],
        "command": command, "status": "ready", "needs_new_gate": decision["needs_new_gate"], "decision": dict(decision)}
    if state.get("next_dispatch_backends"):
        record["dispatch_backends"] = state.pop("next_dispatch_backends")
    state["queues"].append(record)
    save()
    write_json_atomic(Path(record["directory"]) / "submission.json", record)
    return record


def main():
    import sys
    held.main(adapter=sys.modules[__name__])


if __name__ == "__main__":
    main()
