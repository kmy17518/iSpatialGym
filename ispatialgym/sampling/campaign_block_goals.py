"""Fixed four-pocket block goals with fresh scatter positions and grasp draws."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass

from ispatialgym.sampling.campaign_held_goals import ENV, GoalGroupTimeout, GoalPlanError, _require
from ispatialgym.sampling.campaign_tray_goals import pose, pose_error

SCHEMA = "isg-block-goal-plan-v1"


@dataclass(frozen=True)
class BlockGoalPlan:
    campaign_id: str
    group_id: str
    split: str
    template_id: str
    contract_template_sha256: str
    target_names: tuple
    board_name: str
    board: tuple
    slots: tuple
    pattern_id: str
    pattern: tuple
    initial_orientations: tuple
    initial_up_faces: tuple
    flip_directions: tuple
    deadline_at: float
    max_members: int
    sha256: str

    def summary(self):
        def as_pose(value):
            return {"position": list(value[0]), "orientation": list(value[1])}
        return {"campaign_id": self.campaign_id, "group_id": self.group_id, "split": self.split,
                "runtime_template_id": self.template_id, "contract_template_sha256": self.contract_template_sha256,
                "requested_goal": {"slots": [as_pose(value) for value in self.slots], "board": as_pose(self.board)},
                "pattern_id": self.pattern_id, "pattern": list(self.pattern), "board_name": self.board_name,
                "plan_sha256": self.sha256, "deadline_at": self.deadline_at, "max_members": self.max_members,
                "phase": "planned_before_start"}

    def verify_slot(self, actual, index):
        dp, dq = pose_error(actual, self.slots[index])
        if dp > .005 or dq > .05:
            raise GoalPlanError(f"block slot {index} differs from shared goal by {dp:.6f} m / {dq:.6f} rad")

    def verify_board(self, actual):
        dp, dq = pose_error(actual, self.board)
        if dp > .005 or dq > .05:
            raise GoalPlanError(f"board differs from shared goal by {dp:.6f} m / {dq:.6f} rad")


def validate_block_plan(raw, *, template_id, scene, rooms, activated_at, now=None):
    now = time.time() if now is None else now
    _require(isinstance(raw, dict) and raw.get("schema") == SCHEMA, "invalid block goal schema")
    _require(raw.get("task") == "configuration_matching-block_design", "unsupported block task")
    _require(raw.get("template_id") == template_id and raw.get("scene") == scene and raw.get("room") in rooms,
             "block template/scene/room mismatch")
    for key in ("campaign_id", "group_id", "board_name", "pattern_id"):
        _require(isinstance(raw.get(key), str) and bool(raw[key]), f"missing {key}")
    _require(raw.get("split") in ("train", "eval_standard", "eval_model", "eval_category"), "invalid block split")
    sha = raw.get("contract_template_sha256")
    _require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "invalid contract hash")
    names = raw.get("target_names")
    _require(isinstance(names, list) and len(names) == 4 and all(isinstance(v, str) and v for v in names)
             and len(set(names + [raw["board_name"]])) == 5, "block plan needs four distinct block names and one board")
    slots = raw.get("goal_slots")
    _require(isinstance(slots, list) and len(slots) == 4, "block plan requires four goal slots")
    slots = tuple(pose(value, "block goal slot") for value in slots)
    board = pose(raw.get("board_pose"), "board pose")
    pattern, up, directions, initial = (raw.get(key) for key in ("pattern", "initial_up_faces", "flip_directions", "initial_orientations"))
    _require(all(isinstance(v, list) and len(v) == 4 for v in (pattern, up, directions, initial)), "block profile must have four entries")
    _require(all(isinstance(v, str) and v for v in pattern), "invalid block pattern")
    faces = {"+x", "-x", "+y", "-y", "+z", "-z"}
    _require(all(v in faces for v in up) and len(set(up)) == 4, "block initial top faces must be distinct")
    _require(all(v in (None, "left", "right", "near", "far") for v in directions), "invalid flip direction")
    initial = tuple(pose({"position": [0, 0, 0], "orientation": value}, "initial orientation")[1] for value in initial)
    created, budget = float(raw.get("created_at", 0)), float(raw.get("budget_seconds", 0))
    _require(all(math.isfinite(x) for x in (created, budget, activated_at, now))
             and 0 < created <= activated_at <= now and budget > 0, "invalid block time budget")
    if now >= activated_at + budget:
        raise GoalGroupTimeout("block goal-group time budget expired")
    count = raw.get("max_members", 3)
    _require(isinstance(count, int) and not isinstance(count, bool) and 2 <= count <= 3, "invalid block group size")
    fingerprint = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return BlockGoalPlan(raw["campaign_id"], raw["group_id"], raw["split"], template_id, sha, tuple(names), raw["board_name"],
                         board, slots, raw["pattern_id"], tuple(pattern), initial, tuple(up), tuple(directions),
                         activated_at+budget, count, fingerprint)


def load_block_plan(*, template_id, scene, rooms, activated_at, expected_sha256=None):
    path = os.environ.get(ENV)
    if not path:
        return None
    with open(path) as stream:
        plan = validate_block_plan(json.load(stream), template_id=template_id, scene=scene, rooms=rooms, activated_at=activated_at)
    _require(expected_sha256 is None or expected_sha256 == plan.sha256, "block goal changed while sampling")
    return plan
