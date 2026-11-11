"""Campaign attachment goals fixed before the target's start and grasp draw."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass

from ispatialgym.sampling.campaign_held_goals import ENV, GoalGroupTimeout, GoalPlanError, _require
from ispatialgym.sampling.campaign_tray_goals import pose, pose_error

SCHEMA = "isg-attachment-goal-plan-v1"


@dataclass(frozen=True)
class AttachmentGoalPlan:
    campaign_id: str
    group_id: str
    split: str
    template_id: str
    contract_template_sha256: str
    target_name: str
    partner_name: str
    goal: tuple
    partner: tuple
    deadline_at: float
    max_members: int
    sha256: str
    anchor_base_pose2d: tuple | None = None

    def summary(self):
        def as_pose(value):
            return {"position": list(value[0]), "orientation": list(value[1])}
        return {"campaign_id": self.campaign_id, "group_id": self.group_id, "split": self.split,
                "runtime_template_id": self.template_id, "contract_template_sha256": self.contract_template_sha256,
                "requested_goal": {"target": as_pose(self.goal), "partner": as_pose(self.partner)},
                "target_name": self.target_name, "partner_name": self.partner_name,
                "plan_sha256": self.sha256, "deadline_at": self.deadline_at,
                "max_members": self.max_members, "phase": "planned_before_start"}

    def verify(self, target, partner, *, position_tol, orientation_tol, partner_position_tol=.005):
        dp, dq = pose_error(partner, self.partner)
        if dp > partner_position_tol or dq > orientation_tol:
            raise GoalPlanError(f"attachment partner drifted {dp:.6f} m / {dq:.6f} rad")
        dp, dq = pose_error(target, self.goal)
        if dp > position_tol or dq > orientation_tol:
            raise GoalPlanError(f"attachment target drifted {dp:.6f} m / {dq:.6f} rad")


def validate_attachment_plan(raw, *, template_id, scene, rooms, activated_at, now=None):
    now = time.time() if now is None else now
    _require(isinstance(raw, dict) and raw.get("schema") == SCHEMA, "invalid attachment goal schema")
    _require(raw.get("task") == "configuration_matching-attachment", "unsupported attachment goal task")
    _require(raw.get("template_id") == template_id and raw.get("scene") == scene and raw.get("room") in rooms,
             "attachment goal template/scene/room mismatch")
    for name in ("campaign_id", "group_id", "target_name", "partner_name"):
        _require(isinstance(raw.get(name), str) and bool(raw[name]), f"missing {name}")
    _require(raw["target_name"] != raw["partner_name"], "attachment bindings must differ")
    _require(raw.get("split") in ("train", "eval_standard", "eval_model", "eval_category"), "invalid attachment split")
    sha = raw.get("contract_template_sha256")
    _require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "invalid contract hash")
    goal, partner = (pose(raw.get(key), key) for key in ("goal_pose", "partner_pose"))
    created, budget = float(raw.get("created_at", 0)), float(raw.get("budget_seconds", 0))
    _require(all(math.isfinite(x) for x in (created, budget, activated_at, now))
             and 0 < created <= activated_at <= now and budget > 0, "invalid attachment time budget")
    if now >= activated_at + budget:
        raise GoalGroupTimeout("attachment goal-group time budget expired")
    count = raw.get("max_members", 3)
    _require(isinstance(count, int) and not isinstance(count, bool) and 2 <= count <= 3, "invalid attachment group size")
    fingerprint = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    from ispatialgym.sampling.campaign_tabletop_starts import anchor_base
    return AttachmentGoalPlan(raw["campaign_id"], raw["group_id"], raw["split"], template_id, sha,
                              raw["target_name"], raw["partner_name"], goal, partner, activated_at + budget, count, fingerprint, anchor_base(raw))


def load_attachment_plan(*, template_id, scene, rooms, activated_at, expected_sha256=None):
    path = os.environ.get(ENV)
    if not path:
        return None
    with open(path) as stream:
        plan = validate_attachment_plan(json.load(stream), template_id=template_id, scene=scene, rooms=rooms,
                                        activated_at=activated_at)
    _require(expected_sha256 is None or expected_sha256 == plan.sha256, "attachment goal changed while sampling")
    return plan
