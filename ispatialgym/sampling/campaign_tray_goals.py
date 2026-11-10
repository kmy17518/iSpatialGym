"""Immutable disembedding goals, selected before any member's start is drawn."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass

from ispatialgym.sampling.campaign_held_goals import ENV, GoalGroupTimeout, GoalPlanError, _require, _vector

SCHEMA = "isg-tray-goal-plan-v1"


def pose(value, name):
    _require(isinstance(value, dict), f"missing {name}")
    p, q = _vector(value.get("position", ()), 3, name), _vector(value.get("orientation", ()), 4, name)
    _require(abs(sum(x * x for x in q) - 1) < 1e-5, f"{name}: quaternion must be normalized")
    return p, q


def pose_error(a, b):
    """Position metres and quaternion geodesic radians (sign invariant)."""
    p, q = a
    pp, qq = b
    distance = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(p, pp)))
    qn = math.sqrt(sum(float(x) ** 2 for x in q))
    qqn = math.sqrt(sum(float(x) ** 2 for x in qq))
    if qn <= 0 or qqn <= 0:
        raise GoalPlanError("zero quaternion")
    dot = abs(sum(float(x) * float(y) for x, y in zip(q, qq))) / (qn * qqn)
    return distance, 2 * math.acos(min(1.0, dot))


def polygon_area(points):
    """Shoelace area of one convex footprint; no dependency on a simulator."""
    points = list(points)
    return abs(sum(float(a[0]) * float(b[1]) - float(b[0]) * float(a[1])
                   for a, b in zip(points, points[1:] + points[:1]))) / 2


@dataclass(frozen=True)
class TrayGoalPlan:
    campaign_id: str
    group_id: str
    split: str
    template_id: str
    contract_template_sha256: str
    target_name: str
    source_name: str
    destination_name: str
    goal: tuple
    source: tuple
    destination: tuple
    crowdedness_range: tuple
    deadline_at: float
    max_members: int
    sha256: str

    def summary(self):
        as_pose = lambda x: {"position": list(x[0]), "orientation": list(x[1])}
        return {"campaign_id": self.campaign_id, "group_id": self.group_id, "split": self.split,
                "runtime_template_id": self.template_id, "contract_template_sha256": self.contract_template_sha256,
                "requested_goal": {"target": as_pose(self.goal), "destination": as_pose(self.destination)},
                "target_name": self.target_name, "destination_name": self.destination_name,
                "plan_sha256": self.sha256, "crowdedness_range": list(self.crowdedness_range),
                "bucket_strict": self.split != "train", "deadline_at": self.deadline_at,
                "max_members": self.max_members, "phase": "planned_before_start"}

    def accepts_coverage(self, value):
        lo, hi = self.crowdedness_range
        # The specified buckets are [0,.3), [.3,.45), [.45,1].
        return self.split == "train" or (lo <= value and (value < hi or (hi == 1 and value <= hi)))


def validate_tray_plan(raw, *, template_id, scene, rooms, activated_at, now=None):
    now = time.time() if now is None else now
    _require(isinstance(raw, dict) and raw.get("schema") == SCHEMA and raw.get("task") == "disembedding",
             "invalid tray goal plan schema/task")
    _require(raw.get("template_id") == template_id and raw.get("scene") == scene and raw.get("room") in rooms,
             "tray plan template/scene/room mismatch")
    for key in ("campaign_id", "group_id", "target_name", "source_name", "destination_name"):
        _require(isinstance(raw.get(key), str) and bool(raw[key]), f"missing {key}")
    _require(len({raw[k] for k in ("target_name", "source_name", "destination_name")}) == 3, "tray/target names must differ")
    _require(raw.get("split") in ("train", "eval_standard", "eval_model", "eval_category"), "invalid split")
    sha = raw.get("contract_template_sha256")
    _require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "invalid template hash")
    goal, source, destination = (pose(raw.get(k), k) for k in ("goal_pose", "source_pose", "destination_pose"))
    lo, hi = _vector(raw.get("crowdedness_range", ()), 2, "crowdedness range")
    _require(0 <= lo < hi <= 1, "invalid crowdedness range")
    created, budget = float(raw.get("created_at", 0)), float(raw.get("budget_seconds", 0))
    _require(all(math.isfinite(x) for x in (created, budget, activated_at, now))
             and 0 < created <= activated_at <= now and budget > 0, "invalid time budget")
    if now >= activated_at + budget:
        raise GoalGroupTimeout("tray goal-group time budget expired")
    count = raw.get("max_members", 3)
    _require(isinstance(count, int) and not isinstance(count, bool) and 2 <= count <= 5, "invalid group size")
    fingerprint = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return TrayGoalPlan(raw["campaign_id"], raw["group_id"], raw["split"], template_id, sha, raw["target_name"],
                        raw["source_name"], raw["destination_name"], goal, source, destination, (lo, hi),
                        activated_at + budget, count, fingerprint)


def load_tray_plan(*, template_id, scene, rooms, activated_at, expected_sha256=None):
    path = os.environ.get(ENV)
    if not path:
        return None
    with open(path) as stream:
        plan = validate_tray_plan(json.load(stream), template_id=template_id, scene=scene, rooms=rooms, activated_at=activated_at)
    _require(expected_sha256 is None or plan.sha256 == expected_sha256, "tray plan changed while sampling")
    return plan
