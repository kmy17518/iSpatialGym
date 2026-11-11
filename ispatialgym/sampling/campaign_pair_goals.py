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
RELATIONS = ("ontop", "under", "inside", "nextto", "touching")


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
    place: tuple | None = None
    base_push: dict | None = None
    placement_mode: str | None = None
    release_policy: str = "fixed"
    anchor_base_pose2d: tuple | None = None
    floor_push: dict | None = None
    anchor_manipulation_stance: dict | None = None

    @property
    def orientation_required(self):
        return self.scale == "large" or self.relation != "inside"

    def summary(self):
        as_pose = lambda x: {"position": list(x[0]), "orientation": list(x[1])}
        result = {"campaign_id": self.campaign_id, "group_id": self.group_id, "split": self.split,
                "runtime_template_id": self.template_id, "contract_template_sha256": self.contract_template_sha256,
                "requested_goal": {"target": as_pose(self.goal), "partner": as_pose(self.partner)},
                "requested_release": as_pose(self.release), "relation": self.relation,
                "target_name": self.target_name, "partner_name": self.partner_name,
                "plan_sha256": self.sha256, "deadline_at": self.deadline_at, "max_members": self.max_members,
                "phase": "planned_before_start", "target_orientation_required": self.orientation_required, "scale": self.scale}
        result["release_policy"] = self.release_policy
        if self.place is not None:
            result["requested_place"] = as_pose(self.place)
        if self.base_push is not None:
            result["base_push"] = self.base_push
        if self.placement_mode is not None:
            result["placement_mode"] = self.placement_mode
        if self.floor_push is not None:
            result["floor_push"] = self.floor_push
        return result

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


def object_pose_from_grasp(place, eef_in_object):
    """Recover the actual pre-push object place from the fresh oracle sidecar."""
    pp, pq = pose(place, "gripper place")
    gp, gq = pose(eef_in_object, "gripper in object")
    def multiply(a, b):
        x, y, z, w = a
        X, Y, Z, W = b
        return (w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z)
    oq = multiply(pq, (-gq[0], -gq[1], -gq[2], gq[3]))
    offset = multiply(multiply(oq, (*gp, 0.0)), (-oq[0], -oq[1], -oq[2], oq[3]))
    return tuple(pp[i]-offset[i] for i in range(3)), oq


def vary_base_push_gap(goal, release, *, inward, gap_max, observed_gap, fraction):
    """Choose a large next-to goal on the audited partner side before drawing starts."""
    _require(len(inward) == 2 and all(math.isfinite(float(v)) for v in inward)
             and abs(math.hypot(*inward)-1) < 1e-4, "invalid partner side normal")
    _require(all(math.isfinite(float(v)) for v in (gap_max, observed_gap, fraction))
             and gap_max > 0 and 0 <= observed_gap < gap_max and 0 <= fraction <= 1, "invalid base-push gap proposal")
    desired = (.1 + .8 * fraction) * gap_max
    shift = observed_gap - desired
    def shifted(value):
        return ([float(value[0][i]) + (inward[i]*shift if i < 2 else 0) for i in range(3)], list(value[1]))
    return shifted(goal), shifted(release), desired


def validate_pair_plan(raw, *, template_id, scene, rooms, relation, activated_at, now=None, scale="small"):
    now = time.time() if now is None else now
    _require(isinstance(raw, dict) and raw.get("schema") == SCHEMA, "invalid pair goal schema")
    _require(scale in ("small", "large") and relation in RELATIONS
             and raw.get("task") == f"inter_object_relation_satisfaction-{relation}_{scale}_scale", "unsupported pair goal task")
    base_push = raw.get("base_push")
    placement_mode = raw.get("placement_mode")
    release_policy = raw.get("release_policy", "fixed")
    _require(release_policy == "fixed" or (release_policy == "native_grasp_height" and scale == "small" and relation == "inside"),
             "unsupported pair release policy")
    if placement_mode is not None:
        _require(scale == "large" and relation in ("touching", "under") and placement_mode in ("surface", "drop_push") and base_push is None
                 and (relation != "under" or placement_mode == "drop_push"),
                 "unsupported pair placement mode")
    _require(not (scale == "large" and relation == "under") or placement_mode == "drop_push",
             "large under requires a native drop-push plan")
    if scale == "large" and relation in ("nextto", "touching") and placement_mode is None:
        _require(isinstance(base_push, dict) and isinstance(base_push.get("side"), str)
                 and base_push.get("face") in ("+x", "-x", "+y", "-y"), "large relation requires an audited base push")
        for key, length in (("direction", 3), ("side_inward", 2)):
            v = base_push.get(key)
            _require(isinstance(v, (list, tuple)) and len(v) == length
                     and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
                     and abs(math.sqrt(sum(x*x for x in v))-1) < 1e-4, f"invalid base push {key}")
        _require(abs(base_push["direction"][2]) < 1e-6, "base push must be horizontal")
        _require(all(isinstance(base_push.get(k), (int, float)) and math.isfinite(base_push[k])
                     for k in ("goal_gap_m", "nextto_bound_m")) and base_push["nextto_bound_m"] > 0,
                 "base push lacks measured gap bounds")
        if relation == "nextto":
            _require(.1-1e-6 <= base_push["goal_gap_m"]/base_push["nextto_bound_m"] <= .9+1e-6,
                     "base push goal gap must be in middle eighty percent")
    else:
        _require(base_push is None, "base push descriptor on a non-push task")
    goal_cell = raw.get("goal_cell")
    if scale == "large" and base_push is None:
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
    place = pose(raw.get("place_pose"), "pre-push place") if (relation == "touching" and scale == "small") or placement_mode == "drop_push" else None
    if place is not None:
        _require(pose_error(place, goal)[0] > .001, "touching place must precede its contact goal")
    if placement_mode == "drop_push":
        _require(math.dist(goal_cell, release[0][:2]) <= .001, "drop-push goal cell differs from its contact endpoint")
        _require(pose_error((release[0], place[1]), release)[1] <= .05, "drop-push changes orientation after release")
    floor_push = raw.get("floor_push")
    if scale == "large" and relation == "under":
        _require(isinstance(floor_push, dict) and isinstance(floor_push.get("edge"), str)
                 and bool(floor_push["edge"]), "large under requires the audited native push edge")
        direction = floor_push.get("direction")
        _require(isinstance(direction, (list, tuple)) and len(direction) == 2
                 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in direction)
                 and abs(math.hypot(*direction)-1) < 1e-4, "invalid floor push direction")
        delta = [release[0][i]-place[0][i] for i in range(2)]
        _require(sum(delta[i]*direction[i] for i in range(2)) > .01
                 and abs(delta[0]*direction[1]-delta[1]*direction[0]) <= .001,
                 "floor push drop and goal do not share the recorded forward lane")
    else:
        _require(floor_push is None, "floor push descriptor belongs only to large under")
    created, budget = float(raw.get("created_at", 0)), float(raw.get("budget_seconds", 0))
    _require(all(math.isfinite(x) for x in (created, budget, activated_at, now))
             and 0 < created <= activated_at <= now and budget > 0, "invalid time budget")
    if now >= activated_at + budget:
        raise GoalGroupTimeout("pair goal-group time budget expired")
    count = raw.get("max_members", 3)
    _require(isinstance(count, int) and not isinstance(count, bool) and 2 <= count <= 5, "invalid group size")
    fingerprint = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    from ispatialgym.sampling.campaign_tabletop_starts import anchor_base, anchor_manipulation
    hint = anchor_manipulation(raw)
    # This is only an audited proposal hint. The sampler separately scopes its
    # use to measured class recoveries; native feasibility remains mandatory.
    _require(hint is None or (scale == 'small' and relation in ('inside', 'nextto', 'touching', 'ontop', 'under')),
             'source manipulation proposal requires a supported small pair relation')
    return PairGoalPlan(raw["campaign_id"], raw["group_id"], raw["split"], template_id, sha, relation,
                        raw["target_name"], raw["partner_name"], release, goal, partner, activated_at + budget, count, fingerprint, scale, goal_cell, place, base_push, placement_mode, release_policy, anchor_base(raw), floor_push, hint)


def load_pair_plan(*, template_id, scene, rooms, relation, activated_at, expected_sha256=None, scale="small"):
    path = os.environ.get(ENV)
    if not path:
        return None
    with open(path) as stream:
        plan = validate_pair_plan(json.load(stream), template_id=template_id, scene=scene, rooms=rooms,
                                  relation=relation, activated_at=activated_at, scale=scale)
    _require(expected_sha256 is None or plan.sha256 == expected_sha256, "pair goal plan changed while sampling")
    return plan
