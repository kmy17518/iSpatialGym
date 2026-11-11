"""Pure validation and draws for opt-in held-object goal-first campaign groups.

The sampler consumes one immutable plan per worker via
ISPATIALGYM_CAMPAIGN_GOAL_PLAN. A planner selects the goal and base before any
member starts are drawn. A fresh baseline anchor may supply a candidate plan,
but that anchor is not a member of the final group.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCHEMA = "isg-held-goal-plan-v1"
ENV = "ISPATIALGYM_CAMPAIGN_GOAL_PLAN"


class GoalPlanError(ValueError):
    pass


class GoalGroupTimeout(GoalPlanError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GoalPlanError(message)


def _vector(values: Sequence[float], n: int, label: str) -> tuple[float, ...]:
    _require(isinstance(values, (list, tuple)) and len(values) == n, f"{label} needs {n} numbers")
    try:
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise GoalPlanError(f"{label} must be numeric") from exc
    _require(all(math.isfinite(v) for v in out), f"{label} must be finite")
    return out


@dataclass(frozen=True)
class HeldGoalPlan:
    campaign_id: str
    group_id: str
    split: str
    task: str
    template_id: str
    scene: str
    room: str
    position: tuple[float, ...]
    orientation: tuple[float, ...]
    base_pose2d: tuple[float, ...]
    magnitude_range: tuple[float, ...]
    created_at: float
    deadline_at: float
    max_members: int
    contract_template_sha256: str | None
    sha256: str

    def summary(self) -> dict[str, Any]:
        return {"group_id": self.group_id, "campaign_id": self.campaign_id, "plan_sha256": self.sha256,
                "runtime_template_id": self.template_id, "contract_template_sha256": self.contract_template_sha256,
                "split": self.split, "bucket_strict": self.split != "train",
                "requested_goal": {"position": list(self.position), "orientation": list(self.orientation)},
                "base_pose2d": list(self.base_pose2d), "magnitude_range": list(self.magnitude_range),
                "max_members": self.max_members, "deadline_at": self.deadline_at, "phase": "planned_before_start"}


def validate_goal_plan(raw: Mapping[str, Any], *, task: str, template_id: str, scene: str,
                       rooms: Sequence[str], now: float | None = None,
                       activated_at: float | None = None) -> HeldGoalPlan:
    _require(isinstance(raw, Mapping) and raw.get("schema") == SCHEMA, "invalid held-goal plan schema")
    _require(task in ("object_scaling", "mental_rotation") and raw.get("task") == task, "held-goal plan task mismatch")
    _require(raw.get("template_id") == template_id and raw.get("scene") == scene, "held-goal plan template/scene mismatch")
    _require(raw.get("room") in rooms, "held-goal plan room is outside this template")
    for k in ("campaign_id", "group_id"):
        _require(isinstance(raw.get(k), str) and bool(raw[k]), f"missing {k}")
    split = raw.get("split")
    _require(split in ("train", "eval_standard", "eval_model", "eval_category"), "missing or invalid plan split")
    goal = raw.get("goal_pose") or {}
    position = _vector(goal.get("position", ()), 3, "goal position")
    orientation = _vector(goal.get("orientation", ()), 4, "goal quaternion xyzw")
    _require(abs(sum(x * x for x in orientation) - 1.0) <= 1e-5, "goal quaternion must be normalized")
    base = _vector(raw.get("base_pose2d", ()), 3, "base_pose2d")
    bounds = _vector(raw.get("magnitude_range", ()), 2, "magnitude_range")
    limits = (0.05, 0.30) if task == "object_scaling" else (30.0, 120.0)
    _require(limits[0] <= bounds[0] < bounds[1] <= limits[1], "magnitude range outside campaign task bounds")
    now = time.time() if now is None else float(now)
    created = float(raw.get("created_at", 0))
    if "deadline_at" in raw:
        deadline = float(raw["deadline_at"])
    else:
        budget = float(raw.get("budget_seconds", 0))
        _require(math.isfinite(budget) and budget > 0, "invalid group time budget")
        activation = now if activated_at is None else float(activated_at)
        _require(math.isfinite(activation) and created <= activation <= now, "invalid group activation time")
        deadline = activation + budget
    _require(math.isfinite(created) and math.isfinite(deadline) and 0 < created < deadline, "invalid group time budget")
    _require(created <= now, "goal group has not started")
    if now >= deadline:
        raise GoalGroupTimeout("goal-group time budget expired")
    count = raw.get("max_members", 3)
    _require(isinstance(count, int) and not isinstance(count, bool) and 2 <= count <= 5, "invalid goal-group member limit")
    if task == "object_scaling":
        _require(count <= 3, "object scaling permits at most three group members")
    contract_hash = raw.get("contract_template_sha256")
    if contract_hash is not None:
        _require(isinstance(contract_hash, str) and len(contract_hash) == 64
                 and all(x in "0123456789abcdef" for x in contract_hash), "invalid contract template hash")
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return HeldGoalPlan(raw["campaign_id"], raw["group_id"], split, task, template_id, scene, raw["room"], position,
                        orientation, base, bounds, created, deadline, count, contract_hash, hashlib.sha256(encoded).hexdigest())


def load_goal_plan(*, task: str, template_id: str, scene: str, rooms: Sequence[str],
                   expected_sha256: str | None = None, activated_at: float | None = None) -> HeldGoalPlan | None:
    path = os.environ.get(ENV)
    if not path:
        return None
    with open(path) as stream:
        plan = validate_goal_plan(json.load(stream), task=task, template_id=template_id, scene=scene, rooms=rooms,
                                  activated_at=activated_at)
    _require(expected_sha256 is None or plan.sha256 == expected_sha256, "goal plan changed while worker was running")
    return plan


def campaign_grasp_candidates(order, successful_uses, initial_limit=4):
    """Try diversity candidates first, then retain previously feasible grasps.

    Failed IK draws do not advance grasp-use counts. Without this fallback,
    one feasible grasp can fall permanently outside the first four after the
    first member of a fixed-goal group succeeds. Every returned grasp still
    undergoes the same native start, path, collision, and physical checks.
    """
    names = list(order)
    _require(type(initial_limit) is int and initial_limit > 0, "invalid grasp candidate limit")
    _require(len(names) == len(set(names)), "duplicate grasp candidates")
    return names[:initial_limit] + [name for name in names[initial_limit:]
                                   if int(successful_uses.get(name, 0)) > 0]


def draw_start_depth(rng, goal_depth: float, window: Sequence[float], magnitude_range: Sequence[float], *,
                     min_change: float = 0.05, direction: str | None = None,
                     start_band: Sequence[float] | None = None) -> float:
    """Uniform draw over all admissible start intervals conditioned on one goal.

    The RNG needs only uniform(); both numpy Generator and random.Random work.
    Interval-length weighting avoids bias toward the smaller direction interval.
    """
    lo, hi = _vector(window, 2, "depth window")
    mlo, mhi = _vector(magnitude_range, 2, "depth change range")
    _require(math.isfinite(goal_depth) and lo <= goal_depth <= hi, "goal is outside reachable depth window")
    _require(lo < hi and 0 < mlo < mhi and math.isfinite(min_change), "invalid depth draw bounds")
    mlo = max(mlo, min_change)
    _require(mlo < mhi, "minimum depth change empties the requested bucket")
    _require(direction in (None, "toward", "away"), "unknown depth direction")
    s_lo, s_hi = {"toward": (lo + min_change, hi), "away": (lo, hi - min_change), None: (lo, hi)}[direction]
    if start_band is not None:
        a, b = _vector(start_band, 2, "start depth band")
        _require(0 <= a <= b <= 1, "start band must contain fractions in [0,1]")
        s_lo, s_hi = s_lo + a * (s_hi - s_lo), s_lo + b * (s_hi - s_lo)
    intervals = []
    if direction in (None, "away"):
        intervals.append((max(s_lo, goal_depth - mhi), min(s_hi, goal_depth - mlo)))
    if direction in (None, "toward"):
        intervals.append((max(s_lo, goal_depth + mlo), min(s_hi, goal_depth + mhi)))
    intervals = [(a, b) for a, b in intervals if b > a]
    _require(bool(intervals), "no start depth satisfies the fixed goal and requested bucket")
    total = sum(b - a for a, b in intervals)
    draw = float(rng.uniform(0.0, total))
    for a, b in intervals:
        if draw <= b - a:
            return a + draw
        draw -= b - a
    return intervals[-1][1]
