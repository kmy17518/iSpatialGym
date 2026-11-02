"""Tool surface of the L3 agent (RoboProbe Inspect EEF: ``move_eef`` with named absolute targets, ``give_up``; plus R1Pro's
``move_base`` and ``move_torso``) and the translation of a tool call into a motion request. There is no success tool:
the environment ends a successful episode."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional

from ispatialgym.harness.config import HarnessConfig
from ispatialgym.harness.core import pose
from ispatialgym.harness.core.approver import parse_eef_targets
from ispatialgym.harness.core.types import Decision, MotionRequest
from ispatialgym.harness.llm.base import ToolCall, ToolSpec

MOTION_TOOLS = ("move_eef", "move_base", "move_torso")
_NOTE = {"type": "string", "description": "What you observe right now and why you chose this motion, in one or two plain sentences."}
_LEGACY_KEYS = ("arm", "position", "delta_position", "orientation_rpy_deg", "delta_rpy_deg")


class ToolArgError(ValueError):
    """A malformed call; the message goes back to the model as the tool result and counts as a repair."""


def move_eef_description(cfg: HarnessConfig) -> str:
    negate = bool((cfg.perturb or {}).get("negate_xyz"))
    return (
        "Move to absolute base-frame grasp-point targets. Name only the dimensions you intend to change; every unnamed dimension holds "
        "its observed value, so lowering one hand 2 cm is a single entry. Positions are metres in the robot base frame (x forward, y left, "
        "z up, origin on the floor at the centre of the base). Orientation is three angles in degrees measured from a level grasp pointing "
        "straight ahead, not absolute angles: at pitch_deg=roll_deg=yaw_deg=0 the outward tool axis points at base +x and the jaws close "
        "along base y. pitch_deg turns about base +y (0 is level, +90 is pointing straight down), roll_deg about base +x, which is the jaw "
        "rotation of a level grasp, and yaw_deg about base +z (positive turns the tool to the left; for a straight-down grasp it is the "
        "jaw rotation). Naming dimensions of both arms moves them simultaneously along time-aligned paths. The motion plays at a fixed "
        f"safe speed, so how long it takes follows from how far it goes; a move needing over {cfg.limits.max_duration_s:g}s is refused "
        "and has to be split. A gripper value named alongside a pose moves the jaw only after the arm has arrived, so naming both in one "
        f"call is a safe way to grasp at a pose. Per-dimension bounds: {pose.bounds_text(negate)}."
    )


def tool_specs(cfg: HarnessConfig) -> List[ToolSpec]:
    lim = cfg.limits
    specs = [
        ToolSpec("move_eef", move_eef_description(cfg), {
            "type": "object",
            "properties": {
                "targets": {"type": "object", "additionalProperties": {"type": "number"},
                            "description": "Map of dimension name to value. Valid names: " + ", ".join(pose.labels())},
                "note": _NOTE,
            },
            "required": ["targets", "note"],
        }),
        ToolSpec("move_base",
                 "Drive the wheeled base relative to its current pose: forward_m forward, left_m to the left, turn_deg about the vertical axis "
                 f"(positive turns left). At most {lim.base_max_m:g} m and {lim.base_max_turn_deg:g} deg per call. The base does not avoid "
                 "obstacles. The arms and the torso keep their joints, so the grasp points ride along with the base.",
                 {"type": "object", "properties": {"forward_m": {"type": "number"}, "left_m": {"type": "number"}, "turn_deg": {"type": "number"},
                                                   "note": _NOTE}, "required": ["note"]}),
        ToolSpec("move_torso",
                 "Set the torso: translate from 0 (upright, tallest) to 2 (lowest), and tilt_deg, an upper-body pitch offset of at most "
                 f"{lim.torso_tilt_max_deg:g} deg either way. The head camera and both arms move with the torso.",
                 {"type": "object", "properties": {"translate": {"type": "number", "minimum": 0, "maximum": 2}, "tilt_deg": {"type": "number"},
                                                   "note": _NOTE}, "required": ["note"]}),
    ]
    if cfg.allow_done:
        specs.append(ToolSpec("done", "End the episode because the task is complete from your point of view.",
                              {"type": "object", "properties": {"summary": {"type": "string"}, "hindsight": {"type": "string"}}, "required": ["summary"]}))
    if not cfg.disable_give_up:
        specs.append(ToolSpec("give_up", "End the episode because the task cannot be completed.",
                              {"type": "object", "properties": {"reason": {"type": "string"}, "hindsight": {"type": "string"}},
                               "required": ["reason", "hindsight"]}))
    return specs


def merge_parallel_calls(calls: List[ToolCall]) -> List[ToolCall]:
    """Fold several parallel move_eef calls naming disjoint dimensions into the one call per turn."""
    if len(calls) < 2:
        return calls
    merged: Dict[str, Any] = {}
    notes: List[str] = []
    for c in calls:
        targets = (c.arguments or {}).get("targets") if c.name == "move_eef" else None
        if not isinstance(targets, Mapping):
            return calls
        for label, value in targets.items():
            if str(label) in merged:
                return calls
            merged[str(label)] = value
        note = (c.arguments or {}).get("note")
        if isinstance(note, str) and note.strip():
            notes.append(note.strip())
    args = {"targets": merged, "note": " | ".join(notes)}
    return [ToolCall(id=calls[0].id, name="move_eef", arguments=args, raw_arguments=json.dumps(args))]


def _num(v: Any, name: str, default: Optional[float] = 0.0) -> Optional[float]:
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ToolArgError(f"{name} must be a finite number, got {v!r}")
    return float(v)


def to_decision(call: ToolCall, cfg: HarnessConfig) -> Decision:
    """A valid motion call -> its request (model frame; :mod:`core.perturb` maps it to what is executed)."""
    args = call.arguments or {}
    note = str(args.get("note") or "").strip()
    if call.name == "move_eef":
        legacy = [k for k in _LEGACY_KEYS if k in args]
        if legacy:
            raise ToolArgError(f"move_eef does not take {legacy[0]!r}. Name base-frame dimensions instead, for example "
                               '{"targets": {"left_z": 0.90, "left_yaw_deg": -90}}.')
        by_arm, error = parse_eef_targets(args.get("targets"), bool((cfg.perturb or {}).get("negate_xyz")))
        if error:
            raise ToolArgError(error)
        return Decision("motion", MotionRequest("move_eef", {"targets": {str(k): float(v) for k, v in args["targets"].items()}}, note))
    if call.name == "move_base":
        params = {k: _num(args.get(k), k) for k in ("forward_m", "left_m", "turn_deg")}
        return Decision("motion", MotionRequest("move_base", params, note))
    if call.name == "move_torso":
        params = {"translate": _num(args.get("translate"), "translate", None), "tilt_deg": _num(args.get("tilt_deg"), "tilt_deg", None)}
        if params["translate"] is None and params["tilt_deg"] is None:
            raise ToolArgError("move_torso needs translate and / or tilt_deg")
        return Decision("motion", MotionRequest("move_torso", {k: v for k, v in params.items() if v is not None}, note))
    raise ToolArgError(f"unknown tool {call.name!r}")
