"""Opt-in fixed pair goals for the small-scale pick/place relations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass

from ispatialgym.sampling.campaign_held_goals import ENV, GoalGroupTimeout, GoalPlanError, _require
from ispatialgym.sampling.campaign_tray_goals import pose, pose_error

SCHEMA = "isg-pair-goal-plan-v1"
RELATIONS = ("ontop", "under", "inside", "nextto")


@dataclass(frozen=True)
class PairGoalPlan:
    campaign_id: str
    group_id: str
    split: str
    template_id: str
    contract_template_sha256: str
    relation: str
    target_name: str
    partner_name: str
    release: tuple
    goal: tuple
    partner: tuple
    deadline_at: float
    max_members: int
    sha256: str
    scale: str = "small"
    goal_cell: tuple | None = None

    @property
    def orientation_required(self):
        return self.scale == "large" or self.relation != "inside"

    def summary(self):
        as_pose = lambda x: {"position": list(x[0]), "orientation": list(x[1])}
        return {"campaign_id": self.campaign_id, "group_id": self.group_id, "split": self.split,
                "runtime_template_id": self.template_id, "contract_template_sha256": self.contract_template_sha256,
                "requested_goal": {"target": as_pose(self.goal), "partner": as_pose(self.partner)},
                "requested_release": as_pose(self.release), "relation": self.relation,
                "target_name": self.target_name, "partner_name": self.partner_name,
                "plan_sha256": self.sha256, "deadline_at": self.deadline_at, "max_members": self.max_members,
                "phase": "planned_before_start", "target_orientation_required": self.orientation_required, "scale": self.scale}

    def verify_settled(self, target, partner, *, position_tol=.03, orientation_tol=.20, partner_position_tol=.005):
        dp, dq = pose_error(partner, self.partner)
        if dp > partner_position_tol or dq > orientation_tol:
            raise GoalPlanError(f"partner drifted {dp:.6f} m / {dq:.6f} rad")
        dp, dq = pose_error(target, self.goal)
        if dp > position_tol or (self.orientation_required and dq > orientation_tol):
            raise GoalPlanError(f"settled pair target drifted {dp:.6f} m / {dq:.6f} rad")


def vary_nextto_gap(goal, release, partner, *, gap_max, observed_gap, base_yaw, fraction):
    """Move along the pair's lateral direction within the middle 80% gap range."""
    _require(all(math.isfinite(float(x)) for x in (gap_max, observed_gap, base_yaw, fraction))
             and gap_max > 0 and 0 <= observed_gap < gap_max and 0 <= fraction <= 1, "invalid nextto gap proposal")
    tangent = (-math.sin(base_yaw), math.cos(base_yaw))
    lateral = sum((goal[0][i] - partner[0][i]) * tangent[i] for i in range(2))
    _require(abs(lateral) > 1e-4, "nextto pair has no lateral side")
    desired = (.1 + .8 * fraction) * gap_max
    shift = (desired - observed_gap) * (1 if lateral > 0 else -1)
    def shifted(value):
        return ([float(value[0][i]) + (tangent[i] * shift if i < 2 else 0) for i in range(3)], list(value[1]))
    return shifted(goal), shifted(release), desired


def validate_pair_plan(raw, *, template_id, scene, rooms, relation, activated_at, now=None, scale="small"):
    now = time.time() if now is None else now
    _require(isinstance(raw, dict) and raw.get("schema") == SCHEMA, "invalid pair goal schema")
    _require(scale in ("small", "large") and relation in (RELATIONS if scale == "small" else ("ontop", "inside"))
             and raw.get("task") == f"inter_object_relation_satisfaction-{relation}_{scale}_scale", "unsupported pair goal task")
    goal_cell = raw.get("goal_cell")
    if scale == "large":
        _require(isinstance(goal_cell, (list, tuple)) and len(goal_cell) == 2
                 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in goal_cell), "large pair goal requires fixed cell")
        goal_cell = tuple(float(v) for v in goal_cell)
    _require(raw.get("template_id") == template_id and raw.get("scene") == scene and raw.get("room") in rooms,
             "pair goal template/scene/room mismatch")
    for name in ("campaign_id", "group_id", "target_name", "partner_name"):
        _require(isinstance(raw.get(name), str) and bool(raw[name]), f"missing {name}")
    _require(raw["target_name"] != raw["partner_name"], "pair bindings must differ")
    _require(raw.get("split") in ("train", "eval_standard", "eval_model", "eval_category"), "invalid pair split")
    sha = raw.get("contract_template_sha256")
    _require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "invalid contract hash")
    release, goal, partner = (pose(raw.get(k), k) for k in ("release_pose", "goal_pose", "partner_pose"))
    created, budget = float(raw.get("created_at", 0)), float(raw.get("budget_seconds", 0))
    _require(all(math.isfinite(x) for x in (created, budget, activated_at, now))
             and 0 < created <= activated_at <= now and budget > 0, "invalid time budget")
    if now >= activated_at + budget:
        raise GoalGroupTimeout("pair goal-group time budget expired")
    count = raw.get("max_members", 3)
    _require(isinstance(count, int) and not isinstance(count, bool) and 2 <= count <= 5, "invalid group size")
    fingerprint = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return PairGoalPlan(raw["campaign_id"], raw["group_id"], raw["split"], template_id, sha, relation,
                        raw["target_name"], raw["partner_name"], release, goal, partner, activated_at + budget, count, fingerprint, scale, goal_cell)


def load_pair_plan(*, template_id, scene, rooms, relation, activated_at, expected_sha256=None, scale="small"):
    path = os.environ.get(ENV)
    if not path:
        return None
    with open(path) as stream:
        plan = validate_pair_plan(json.load(stream), template_id=template_id, scene=scene, rooms=rooms,
                                  relation=relation, activated_at=activated_at, scale=scale)
    _require(expected_sha256 is None or plan.sha256 == expected_sha256, "pair goal plan changed while sampling")
    return plan
