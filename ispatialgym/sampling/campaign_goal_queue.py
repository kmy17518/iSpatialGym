"""Immutable goal queues: amortize one scene load across audited goal groups."""
from __future__ import annotations

import json
import copy
import os
import time
from pathlib import Path

from ispatialgym.sampling.campaign_held_goals import ENV as PLAN_ENV, validate_goal_plan
from scripts.campaign_20261110.contracts import digest, file_digest, require
from scripts.campaign_20261110.goal_reuse_policy import POLICY as GOAL_REUSE_POLICY

ENV = "ISPATIALGYM_CAMPAIGN_GOAL_QUEUE"
SCHEMA = "isg-held-goal-queue-v1"
PAIR_SCHEMA = "isg-pair-goal-queue-v1"
SURFACE_SCHEMA = "isg-surface-goal-queue-v1"
TRAY_SCHEMA = "isg-tray-goal-queue-v1"
JOINT_SCHEMA = "isg-joint-goal-queue-v1"
RESET_FIELDS = ("_campaign_goal_started_at", "_campaign_goal_plan_sha", "_campaign_group_written", "_campaign_latest_proposal",
                "_campaign_group_started_at", "_campaign_pair_plan_sha", "_campaign_pair_active", "_campaign_native_goal",
                "_campaign_surface_plan_sha", "_campaign_surface_active", "_campaign_tray_plan_sha", "_campaign_tray_active",
                "_campaign_destination", "_campaign_joint_plan_sha", "_campaign_joint_active",
                "_campaign_joint_written_starts", "_campaign_stable_probed_sha")


def canonical_goal(plan):
    # Quaternion sign does not distinguish two physical goals.
    goal = plan["goal_pose"]
    q = [float(v) for v in goal["orientation"]]
    first = next((v for v in q if abs(v) > 1e-12), 1.)
    if first < 0:
        q = [-v for v in q]
    return {"position": [0.0 if float(v) == 0 else float(v) for v in goal["position"]],
            "orientation": [0.0 if v == 0 else v for v in q]}


def canonical_pair_goal(plan):
    return {"target": canonical_goal(plan), "partner": canonical_goal({"goal_pose": plan["partner_pose"]})}


def schema_for_task(task):
    if task in ("object_scaling", "mental_rotation"):
        return SCHEMA
    if task in ("alignment-world_vertical", "alignment-axial"):
        return SURFACE_SCHEMA
    if task == "disembedding":
        return TRAY_SCHEMA
    if task.startswith("configuration_matching-articulation_"):
        return JOINT_SCHEMA
    if task.startswith("inter_object_relation_satisfaction-"):
        from ispatialgym.sampling.campaign_pair_goals import RELATIONS
        require(task in {f"inter_object_relation_satisfaction-{relation}_{scale}_scale"
                for scale in ("small", "large") for relation in RELATIONS}, "unsupported queued pair task")
        return PAIR_SCHEMA
    raise ValueError("unsupported queued goal task: " + task)


def canonical_plan_goal(plan, schema):
    if schema == PAIR_SCHEMA:
        return canonical_pair_goal(plan)
    if schema == TRAY_SCHEMA:
        return {"target": canonical_goal(plan), "destination": canonical_goal({"goal_pose": plan["destination_pose"]})}
    if schema == JOINT_SCHEMA:
        return {"root": canonical_goal(plan), "joints": plan["goal_joints"]}
    return canonical_goal(plan)


def read_queue(path, config_path):
    path, config_path = Path(path).resolve(), Path(config_path).resolve()
    raw, config = json.loads(path.read_text()), json.loads(config_path.read_text())
    schema = raw.get("schema")
    require(schema in (SCHEMA, PAIR_SCHEMA, SURFACE_SCHEMA, TRAY_SCHEMA, JOINT_SCHEMA), "invalid goal queue schema")
    require(raw.get("config_sha256") == file_digest(config_path), "goal queue config changed")
    require(raw.get("campaign_id") == config.get("campaign_id"), "goal queue campaign differs from config")
    template = config["template"]
    require(schema == schema_for_task(template["task"]), "queue schema differs from task")
    require(raw.get("entries"), "empty held-goal queue")
    require(raw.get("goal_reuse_policy",GOAL_REUSE_POLICY) == GOAL_REUSE_POLICY, "unknown queued goal-reuse policy")
    ids, entries = set(), []
    for row in raw["entries"]:
        plan_path = Path(row["path"]).resolve()
        require(plan_path.is_file() and file_digest(plan_path) == row["sha256"], "queued goal plan changed")
        plan = json.loads(plan_path.read_text())
        stamp = time.time()
        identity = dict(template_id=template["scene_setup"]["runtime_template_id"],
                        scene=template["scene"], rooms=[template["room"]], now=stamp, activated_at=stamp)
        if schema == SCHEMA:
            validate_goal_plan(plan, task=template["task"], **identity)
        elif schema == PAIR_SCHEMA:
            from ispatialgym.sampling.campaign_pair_goals import validate_pair_plan
            relation, scale, _ = template["task"].split("-", 1)[1].split("_")
            validate_pair_plan(plan, relation=relation, scale=scale, **identity)
        elif schema == SURFACE_SCHEMA:
            from ispatialgym.sampling.campaign_surface_goals import validate_surface_plan
            validate_surface_plan(plan, **identity)
        elif schema == TRAY_SCHEMA:
            from ispatialgym.sampling.campaign_tray_goals import validate_tray_plan
            validate_tray_plan(plan, **identity)
        else:
            from ispatialgym.sampling.campaign_joint_goals import validate_joint_plan
            targets = [obj for obj in template["objects"] if obj["role"] == "target"]
            require(len(targets) == 1 and targets[0].get("joint"), "joint queue requires one declared target joint")
            validate_joint_plan(plan, target=targets[0]["instance"], joint=targets[0]["joint"], **identity)
        require(plan["task"] == template["task"], "queued plan task differs from config")
        require(plan["campaign_id"] == config["campaign_id"] and plan["split"] == config["split"], "queue plan campaign/split mismatch")
        require(plan.get("contract_template_sha256") == config["validation_key"]["template_sha256"], "queue plan template differs from config")
        require(plan["max_members"] <= 3, "queued groups permit at most three members")
        require(plan["group_id"] not in ids, "duplicate queued goal-group ID")
        goal = digest(canonical_plan_goal(plan, schema))
        ids.add(plan["group_id"])
        entries.append({**row, "path": str(plan_path), "group_id": plan["group_id"], "goal_sha256": goal, "max_members": plan["max_members"]})
    return {"path": str(path), "sha256": file_digest(path), "schema": schema, "task": template["task"], "config_path": str(config_path),
            "config_sha256": file_digest(config_path), "entries": entries, "index": 0,"goal_reuse_policy":GOAL_REUSE_POLICY}


def _event(state, event, **details):
    record = {"schema": "isg-goal-queue-event-v1", "created_at": time.time(), "pid": os.getpid(),
              "queue_sha256": state["sha256"], "index": state["index"], "event": event,
              "goal_reuse_policy_version":GOAL_REUSE_POLICY["version"], **details}
    path = Path(state["path"]).with_suffix(".events.jsonl")
    # One sampler owns one queue submission. Append preserves closed groups if
    # the worker exits while preparing its next goal.
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _activate(sampler, state):
    require(file_digest(state["path"]) == state["sha256"], "goal queue changed while running")
    require(file_digest(state["config_path"]) == state["config_sha256"], "queue scene/scale/light config changed while running")
    row = state["entries"][state["index"]]
    require(file_digest(row["path"]) == row["sha256"], "queued plan changed before activation")
    for name in RESET_FIELDS:
        if hasattr(sampler, name):
            delattr(sampler, name)
    if "original_goal_cells" in state:
        sampler.place["cells"] = copy.deepcopy(state["original_goal_cells"])
    if "original_joint_pairs" in state:
        sampler._pairs = copy.deepcopy(state["original_joint_pairs"])
    state["published"] = 0
    os.environ[PLAN_ENV] = row["path"]
    if state["schema"] == JOINT_SCHEMA and state["task"].endswith("_small_scale"):
        # The next native pair draw filters this full, physically probed cache
        # using the newly activated plan before drawing any start or grasp.
        sampler._pair_drawn = False
    _event(state, "activated", group_id=row["group_id"], goal_sha256=row["goal_sha256"], plan_sha256=row["sha256"])


def activate_initial(sampler):
    """Called before the core's group-deadline hook. Disabled is a true no-op."""
    if not os.environ.get(ENV):
        return False
    if hasattr(sampler, "_campaign_goal_queue"):
        return True
    config_path = os.environ.get("ISPATIALGYM_CAMPAIGN_CONFIG")
    require(config_path, "goal queue requires campaign config")
    state = read_queue(os.environ[ENV], config_path)
    require(not os.environ.get(PLAN_ENV) or Path(os.environ[PLAN_ENV]).resolve() == Path(state["entries"][0]["path"]),
            "single goal plan conflicts with queue's first plan")
    sampler._campaign_goal_queue = state
    # Large placement groups pin their native candidate cell array. Restore the
    # original template array before activating a different goal in that scene.
    if isinstance(getattr(sampler, "place", None), dict) and "cells" in sampler.place:
        state["original_goal_cells"] = copy.deepcopy(sampler.place["cells"])
    if state["schema"] == JOINT_SCHEMA and state["task"].endswith("_small_scale"):
        require(bool(getattr(sampler, "_pairs", None)), "small-articulation queue requires native probed angle pairs")
        state["original_joint_pairs"] = copy.deepcopy(sampler._pairs)
    _activate(sampler, state)
    return True


def advance(sampler, reason):
    if not os.environ.get(ENV):
        return False
    state = getattr(sampler, "_campaign_goal_queue", None)
    require(state is not None, "goal queue was never activated")
    require(reason in ("campaign_group_complete", "campaign_group_timeout"), "invalid goal queue transition")
    row = state["entries"][state["index"]]
    _event(state, "closed", group_id=row["group_id"], reason=reason, written=state.get("published", 0), write_attempts=getattr(sampler, "_campaign_group_written", 0))
    if state["index"] + 1 == len(state["entries"]):
        state["exhausted"] = True
        return False
    state["index"] += 1
    _activate(sampler, state)
    return True


def after_written(sampler):
    """Close full groups after publication, so transitions do not consume attempts."""
    if not os.environ.get(ENV):
        return False
    state = sampler._campaign_goal_queue
    row = state["entries"][state["index"]]
    state["published"] = state.get("published", 0) + 1
    _event(state, "published", group_id=row["group_id"], written=state["published"], write_attempts=getattr(sampler, "_campaign_group_written", 0))
    if getattr(sampler, "_campaign_group_written", 0) >= row["max_members"]:
        return not advance(sampler, "campaign_group_complete")
    return False


def finalize(sampler, reason="job_end"):
    state = getattr(sampler, "_campaign_goal_queue", None)
    if state is None or state.get("exhausted"):
        return
    row = state["entries"][state["index"]]
    _event(state, "closed", group_id=row["group_id"], reason=reason,
           written=state.get("published", 0), write_attempts=getattr(sampler, "_campaign_group_written", 0), unstarted_groups=len(state["entries"])-state["index"]-1)
    state["exhausted"] = True
