"""Reuse fresh validated goals across bounded groups in one native scene load."""
from pathlib import Path

from ispatialgym.sampling.campaign_goal_queue import read_queue, schema_for_task
from scripts.campaign_20261110 import goal_groups as g
from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from scripts.campaign_20261110.goal_reuse_policy import POLICY
from scripts.campaign_20261110.held_goal_batches import submission_command


def prepare_batch(*, anchor, template, campaign_root, batch_id, count, magnitude_range, seed,
                  campaign_id, code_sha256, physics_sha256, data_sha256, budget_seconds=900):
    require(batch_id and all(c.isalnum() or c in "_.-" for c in batch_id), "invalid goal batch ID")
    require(isinstance(count, int) and not isinstance(count, bool) and 1 <= count <= 6, "batch requires one to six groups")
    schema = schema_for_task(template.task)
    require(template.task not in ("object_scaling", "mental_rotation"), "held tasks use their native region batch author")
    root = Path(campaign_root).resolve()
    queue_dir = root / "queues" / batch_id
    require(not queue_dir.exists(), "queue already exists")
    entries, groups = [], []
    for ordinal in range(count):
        group_id = f"{batch_id}_{ordinal:04d}"
        directory = root / "groups" / group_id
        g.prepare_group(anchor=anchor, template=template, campaign_id=campaign_id, group_id=group_id, output=directory,
            code_sha256=code_sha256, physics_sha256=physics_sha256, data_sha256=data_sha256,
            magnitude_range=magnitude_range, budget_seconds=budget_seconds, max_members=3, reuse_anchor_goal=True)
        path = directory / "goal_plan.json"
        entries.append({"path": str(path), "sha256": file_digest(path)})
        groups.append(group_id)
        write_json_atomic(directory / "goal_proposal.json", {"method": "fresh_anchor_goal_before_distinct_starts",
            "seed": seed, "ordinal": ordinal, "anchor_sources": anchor["sources"], "goal_reuse_policy": POLICY,
            "source_instance_is_member": False, "native_checks_unchanged": True})
    config = root / "groups" / groups[0] / "campaign_config.json"
    require(all(file_digest(root / "groups" / gid / "campaign_config.json") == file_digest(config) for gid in groups),
            "queued groups differ in exact template/scale/scene/light identity")
    queue_dir.mkdir(parents=True, exist_ok=False)
    path = queue_dir / "queue.json"
    write_json_atomic(path, {"schema": schema, "campaign_id": campaign_id, "config_path": str(config),
        "config_sha256": file_digest(config), "entries": entries, "group_ids": groups, "seed": seed,
        "goal_reuse_policy": POLICY})
    read_queue(path, config)
    return {"queue": str(path), "config": str(config), "group_ids": groups, "candidate_instances": 3 * count}


__all__ = ["prepare_batch", "submission_command"]
