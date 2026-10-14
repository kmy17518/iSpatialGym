"""Bounds of the motion layer: request checks before anything moves (a violation is a correctable, repairable tool
result) and the last check on every 23-D action (non-finite values end the episode)."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ispatialgym.harness.config import MotionLimits
from ispatialgym.harness.core import pose
from ispatialgym.harness.core.types import MotionRejected


class SafetyAbort(RuntimeError):
    """An action the harness must not send (NaN / inf)."""


def parse_eef_targets(raw: Any, negate_xyz: bool = False) -> Tuple[Optional[Dict[str, Dict[str, float]]], Optional[str]]:
    """``move_eef.targets`` -> ``{arm: {axis: value}}``, or the repair message of the first bad entry."""
    if not isinstance(raw, Mapping) or not raw:
        return None, "move_eef.targets must be a non-empty object of dimension name to value"
    valid = pose.labels()
    by_arm: Dict[str, Dict[str, float]] = {}
    for raw_label, raw_value in raw.items():
        label = str(raw_label)
        if label not in valid:
            return None, f"unknown dimension {label!r}; valid names: {', '.join(valid)}"
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)) or not math.isfinite(float(raw_value)):
            return None, f"value for {label!r} must be a finite number, got {raw_value!r}"
        value = float(raw_value)
        lo, hi = pose.bounds(label, negate_xyz)
        if not lo <= value <= hi:
            return None, f"target for {label!r} is outside [{lo:.4g}, {hi:.4g}]"
        arm, axis = pose.split_label(label)
        by_arm.setdefault(arm, {})[axis] = value
    return by_arm, None


class TargetApprover:
    def __init__(self, limits: MotionLimits):
        self.limits = limits

    def check_base(self, forward_m: float, left_m: float, turn_deg: float) -> None:
        lim = self.limits
        vals = (float(forward_m), float(left_m), float(turn_deg))
        if not all(math.isfinite(v) for v in vals):
            raise MotionRejected("move_base arguments must be finite numbers")
        d = math.hypot(vals[0], vals[1])
        if d > lim.base_max_m:
            raise MotionRejected(f"the base move is {d:.2f} m; one move_base call may drive at most {lim.base_max_m:.2f} m - split it")
        if abs(vals[2]) > lim.base_max_turn_deg:
            raise MotionRejected(f"the turn is {vals[2]:.0f} deg; one move_base call may turn at most {lim.base_max_turn_deg:.0f} deg - split it")
        if d < 1e-4 and abs(vals[2]) < 1e-3:
            raise MotionRejected("move_base with zero motion")

    def check_torso(self, translate, tilt_deg) -> None:
        if translate is None and tilt_deg is None:
            raise MotionRejected("move_torso needs translate and / or tilt_deg")
        if translate is not None and not (0.0 <= float(translate) <= 2.0):
            raise MotionRejected("translate must be within 0 (upright) .. 2 (lowest)")
        if tilt_deg is not None and abs(float(tilt_deg)) > self.limits.torso_tilt_max_deg:
            raise MotionRejected(f"|tilt_deg| must be at most {self.limits.torso_tilt_max_deg:.0f}")


def check_action(action) -> None:
    a = np.asarray(action.detach().cpu().numpy() if hasattr(action, "detach") else action, dtype=float)
    if not np.all(np.isfinite(a)):
        raise SafetyAbort(f"non-finite action {a.tolist()}")
