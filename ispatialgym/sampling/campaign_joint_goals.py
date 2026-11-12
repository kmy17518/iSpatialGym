"""Pure contracts for fixed articulation goals and initial-to-goal joint changes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass

from ispatialgym.sampling.campaign_held_goals import ENV, GoalGroupTimeout, GoalPlanError, _require, _vector

__all__ = ["ENV", "GoalGroupTimeout", "GoalPlanError", "JointGoalPlan", "SCHEMA", "load_joint_plan", "validate_joint_plan"]

SCHEMA = "isg-joint-goal-plan-v1"


def unique_probed_angles(values):
    """Keep the exact angles that passed physics; rounding can cross a strict bucket."""
    result = [float(value) for value in values]
    _require(all(math.isfinite(value) for value in result), "nonfinite probed joint angle")
    return sorted(set(result))


def start_key(start):
    return hashlib.sha256(json.dumps(start, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def start_was_written(sampler, start):
    return start_key(start) in getattr(sampler, "_campaign_joint_written_starts", set())


def remember_written_start(sampler, start):
    if not hasattr(sampler, "_campaign_joint_written_starts"):
        sampler._campaign_joint_written_starts = set()
    sampler._campaign_joint_written_starts.add(start_key(start))


@dataclass(frozen=True)
class JointGoalPlan:
    campaign_id: str
    group_id: str
    split: str
    template_id: str
    contract_template_sha256: str
    target: str
    joint: str
    pose: tuple
    joints: dict
    fraction_range: tuple
    deadline_at: float
    max_members: int
    sha256: str
    anchor_base_pose2d: tuple | None = None
    anchor_manipulation_stance: dict | None = None

    def summary(self):
        return {"campaign_id": self.campaign_id, "group_id": self.group_id, "split": self.split,
                "runtime_template_id": self.template_id, "contract_template_sha256": self.contract_template_sha256,
                "plan_sha256": self.sha256, "requested_goal": {"pose": self.pose, "joints": self.joints},
                "joint": self.joint, "fraction_range": self.fraction_range, "bucket_strict": self.split != "train",
                "deadline_at": self.deadline_at, "max_members": self.max_members, "phase": "planned_before_start"}

    def initial(self, *, direction, lower, upper, stable, rng):
        goal = self.joints[self.joint]
        span = upper-lower
        _require(span > 0 and lower-1e-6 <= goal <= upper+1e-6, "fixed articulation goal outside joint limits")
        def in_bucket(value):
            fraction = (value-lower)/span
            lo, hi = self.fraction_range
            return lo <= fraction <= hi and (lo == 0 or fraction > lo)
        if direction == "open":
            _require(goal > lower+1e-6, "opening goal is closed")
            _require(self.split == "train" or in_bucket(goal), "fixed opening goal outside strict eval bucket")
            return float(lower), float(goal)
        _require(direction == "close" and abs(goal-lower) < 1e-6, "closing goal must be the closed joint limit")
        candidates = [float(v) for v in stable if v > goal+1e-6 and (self.split == "train" or in_bucket(v))]
        _require(bool(candidates), "no stable initial angle in the requested articulation bucket")
        return candidates[int(rng.integers(len(candidates)))], float(goal)

    def verify_joint_change(self, names, initial, goal, tolerances=None):
        _require(set(names) == set(self.joints), "articulation goal joint set differs from object")
        for name, q0, q1 in zip(names, initial, goal):
            _require(abs(float(q1)-self.joints[name]) < 1e-6, f"articulation goal changed: {name}")
            if name != self.joint:
                tolerance = (tolerances or {}).get(name, 1e-6)
                _require(abs(float(q0)-float(q1)) <= tolerance, f"another joint changes initial-to-goal: {name}")

    def closing_probe_values(self, *, direction, lower, upper, stable, count=7):
        """Fill missing closing-start grid points within the strict eval bucket.

        Every returned value still needs the native passive-stability probe and
        all later contact/reach/path checks. Training retains its soft bucket.
        """
        if direction != "close" or self.split == "train":
            return []
        _require(upper > lower and isinstance(count, int) and 1 <= count <= 31, "invalid stable probe range/count")
        lo, hi = self.fraction_range
        # A legacy low-angle grid often includes the fully-open endpoint. That
        # single endpoint must not suppress every interior high-bucket probe.
        # Deduplication skips a probe, never substitutes or admits an angle.
        values = [lower + (lo+(hi-lo)*k/(count+1))*(upper-lower) for k in range(1, count+2)]
        return [value for value in values if not any(abs(value-v) <= 1e-6 for v in stable)]

    def eligible_pairs(self, pairs, *, direction, lower, upper):
        """Keep already physically probed angle pairs with this exact goal."""
        _require(upper > lower and direction in ("open", "close"), "invalid articulation limits/direction")
        lo, hi = self.fraction_range
        goal = self.joints[self.joint]
        result = []
        for q0, q1 in pairs:
            if abs(q1-goal) > 1e-6:
                continue
            if not (lower-1e-6 <= q0 <= upper+1e-6 and lower-1e-6 <= q1 <= upper+1e-6):
                continue
            if not (q1 > q0+1e-6 if direction == "open" else q0 > q1+1e-6):
                continue
            fraction = ((q1 if direction == "open" else q0)-lower)/(upper-lower)
            if self.split == "train" or (lo <= fraction <= hi and (lo == 0 or fraction > lo)):
                # Keep the exact native probe identity. Callers select records
                # by this pair; replacing q1 with a close float from the TRO
                # would discard a valid record during the subsequent draw.
                result.append((float(q0), float(q1)))
        _require(bool(result), "no native angle pair satisfies the fixed goal and articulation bucket")
        return result


def validate_joint_plan(raw, *, template_id, scene, rooms, target, joint, activated_at, now=None):
    now = time.time() if now is None else now
    _require(isinstance(raw, dict) and raw.get("schema") == SCHEMA, "invalid articulation goal plan schema")
    _require(raw.get("template_id") == template_id and raw.get("scene") == scene and raw.get("room") in rooms,
             "articulation goal template/scene/room mismatch")
    _require(raw.get("target") == target and raw.get("joint") == joint, "articulation goal role/joint mismatch")
    for name in ("campaign_id", "group_id"):
        _require(isinstance(raw.get(name), str) and bool(raw[name]), f"missing {name}")
    _require(raw.get("split") in ("train", "eval_standard", "eval_model", "eval_category"), "invalid articulation split")
    sha = raw.get("contract_template_sha256")
    _require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "invalid articulation template hash")
    pos = _vector(raw["goal_pose"]["position"], 3, "articulation root position")
    quat = _vector(raw["goal_pose"]["orientation"], 4, "articulation root quaternion")
    _require(abs(sum(v*v for v in quat)-1) < 1e-5, "articulation root quaternion not normalized")
    joints = raw.get("goal_joints")
    _require(isinstance(joints, dict) and joint in joints and all(isinstance(n,str) and n and isinstance(v,(int,float))
             and math.isfinite(v) for n,v in joints.items()), "invalid articulation goal joints")
    bounds = _vector(raw.get("fraction_range", []), 2, "articulation bucket")
    _require(0 <= bounds[0] < bounds[1] <= 1, "invalid articulation bucket")
    created, budget = float(raw.get("created_at", 0)), float(raw.get("budget_seconds", 0))
    _require(all(math.isfinite(v) for v in (created,budget,activated_at,now)) and 0 < created <= activated_at <= now
             and budget > 0, "invalid articulation group budget")
    deadline = activated_at + budget
    if now >= deadline:
        raise GoalGroupTimeout("articulation goal-group time budget expired")
    count = raw.get("max_members", 3)
    _require(isinstance(count,int) and not isinstance(count,bool) and 2 <= count <= 3, "invalid articulation group size")
    from ispatialgym.sampling.campaign_tabletop_starts import anchor_base, anchor_manipulation
    base, manipulation = anchor_base(raw), anchor_manipulation(raw)
    fingerprint = hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return JointGoalPlan(raw["campaign_id"],raw["group_id"],raw["split"],template_id,sha,target,joint,(pos,quat),
                         {n:float(v) for n,v in joints.items()},bounds,deadline,count,fingerprint,
                         base, manipulation)


def carton_source_hint_enabled(plan, class_id):
    """Only the measured carton fixed-root recovery may use these proposals."""
    return (plan is not None and class_id == 'configuration_matching-articulation_open_small_scale_02'
            and plan.anchor_base_pose2d is not None and plan.anchor_manipulation_stance is not None)


def order_carton_stances(plan, class_id, order, candidates):
    if not carton_source_hint_enabled(plan, class_id):
        return order
    from ispatialgym.sampling.campaign_tabletop_starts import prioritize_written_stances
    return prioritize_written_stances(order, candidates, [plan.anchor_manipulation_stance])


def load_joint_plan(*, expected_sha256=None, **kwargs):
    path = os.environ.get(ENV)
    if not path:
        return None
    with open(path) as stream:
        plan = validate_joint_plan(json.load(stream), **kwargs)
    _require(expected_sha256 is None or expected_sha256 == plan.sha256, "articulation plan changed while sampling")
    return plan
