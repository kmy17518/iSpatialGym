"""Opt-in fixed surface-goal plans for wall and axial alignment.

The goal is selected before starts, while initial robot stances and contact
programs may vary. No plan changes a legacy oracle or a physical acceptance rule.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from ispatialgym.sampling.campaign_bucket_policy import POLICY as BUCKET_POLICY
from dataclasses import dataclass

from ispatialgym.sampling.campaign_held_goals import ENV, GoalGroupTimeout, GoalPlanError, _require, _vector


SCHEMA = "isg-surface-goal-plan-v1"


@dataclass(frozen=True)
class SurfaceGoalPlan:
    campaign_id: str
    group_id: str
    split: str
    task: str
    template_id: str
    contract_template_sha256: str
    position: tuple[float, ...]
    orientation: tuple[float, ...]
    magnitude_range_deg: tuple[float, ...]
    deadline_at: float
    max_members: int
    sha256: str
    anchor_base_pose2d: tuple | None = None

    def summary(self):
        return {"campaign_id": self.campaign_id, "group_id": self.group_id, "split": self.split,
                "runtime_template_id": self.template_id, "contract_template_sha256": self.contract_template_sha256,
                "requested_goal": {"position": list(self.position), "orientation": list(self.orientation)},
                "plan_sha256": self.sha256, "magnitude_range_deg": list(self.magnitude_range_deg),
                "bucket_strict": False, "bucket_policy": BUCKET_POLICY, "deadline_at": self.deadline_at, "max_members": self.max_members,
                "phase": "planned_before_start"}


def validate_surface_plan(raw, *, template_id, scene, rooms, activated_at, now=None):
    now = time.time() if now is None else now
    _require(isinstance(raw, dict) and raw.get("schema") == SCHEMA, "invalid surface goal schema")
    _require(raw.get("task") in ("alignment-world_vertical", "alignment-axial"), "unsupported surface goal task")
    _require(raw.get("template_id") == template_id and raw.get("scene") == scene and raw.get("room") in rooms,
             "surface goal template/scene/room mismatch")
    for name in ("campaign_id", "group_id"):
        _require(isinstance(raw.get(name), str) and bool(raw[name]), f"missing {name}")
    _require(raw.get("split") in ("train", "eval_standard", "eval_model", "eval_category"), "invalid surface goal split")
    sha = raw.get("contract_template_sha256")
    _require(isinstance(sha, str) and len(sha) == 64 and all(x in "0123456789abcdef" for x in sha), "invalid contract template hash")
    goal = raw.get("goal_pose") or {}
    position = _vector(goal.get("position", ()), 3, "surface goal position")
    orientation = _vector(goal.get("orientation", ()), 4, "surface goal quaternion")
    _require(abs(sum(x * x for x in orientation) - 1) <= 1e-5, "surface goal quaternion must be normalized")
    lo, hi = _vector(raw.get("magnitude_range_deg", ()), 2, "surface deviation range")
    _require(0 < lo < hi <= 90, "invalid requested deviation range")
    created, budget = float(raw.get("created_at", 0)), float(raw.get("budget_seconds", 0))
    _require(all(math.isfinite(x) for x in (created, budget, activated_at, now))
             and 0 < created <= activated_at <= now and budget > 0, "invalid surface goal time budget")
    deadline = activated_at + budget
    if now >= deadline:
        raise GoalGroupTimeout("surface goal-group time budget expired")
    count = raw.get("max_members", 3)
    _require(isinstance(count, int) and not isinstance(count, bool) and 2 <= count <= 5, "invalid group size")
    from ispatialgym.sampling.campaign_tabletop_starts import anchor_base
    anchor = anchor_base(raw)
    fingerprint = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return SurfaceGoalPlan(raw["campaign_id"], raw["group_id"], raw["split"], raw["task"], template_id, sha, position,
                           orientation, (lo, hi), deadline, count, fingerprint, anchor)


def load_surface_plan(*, template_id, scene, rooms, activated_at, expected_sha256=None):
    path = os.environ.get(ENV)
    if not path:
        return None
    with open(path) as stream:
        plan = validate_surface_plan(json.load(stream), template_id=template_id, scene=scene, rooms=rooms,
                                     activated_at=activated_at)
    _require(expected_sha256 is None or expected_sha256 == plan.sha256, "surface goal plan changed while sampling")
    return plan


def deviation_range(plan: SurfaceGoalPlan, allowed_rad, *, fallback=False):
    """Use the native physical range independently of descriptive bucket labels."""
    lo, hi = (float(x) for x in allowed_rad)
    _require(math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi, "invalid physical deviation range")
    return lo, hi
