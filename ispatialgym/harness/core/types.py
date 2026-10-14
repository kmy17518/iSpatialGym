"""Data types passed between the run_isg adapter, the motion layer and the planners (simulator-free)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ARMS = ("left", "right")


class CapabilityFailure(Exception):
    """Model-side end of the episode (repairs exhausted, several tool calls, refusal): scored as a failure."""


class InfrastructureFailure(BaseException):
    """Provider or harness failure (missing key, outage, retries exhausted, a harness bug): the episode is not scored.

    A ``BaseException`` so that no ``except Exception`` on the way folds an outage into the success rate; run_isg
    records the episode as ``status: error``."""

    def __init__(self, message: str, *, retryable: bool = False, retry_after_s: Optional[float] = None, status: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after_s = retry_after_s
        self.status = status


@dataclass
class ArmState:
    grasp: Dict[str, float]  # x, y, z, pitch_deg, roll_deg, yaw_deg of the grasp point in the base frame (core.pose)
    gripper_opening: float  # 0 = closed .. 1 = fully open (finger joints)
    joints: Tuple[float, ...] = ()  # arm joint positions [rad]

    def to_dict(self) -> Dict[str, Any]:
        return {"grasp": {k: round(float(v), 4) for k, v in self.grasp.items()}, "gripper_opening": round(self.gripper_opening, 3),
                "joints": [round(float(v), 4) for v in self.joints]}


@dataclass
class RobotState:
    """Robot-only state (proprioception and odometry); never object or scene state."""

    step: int
    arms: Dict[str, ArmState]
    base_odometry: Tuple[float, float, float]  # x, y [m], yaw [deg] relative to the episode start, start frame
    torso_translate: float  # JoyLo scalar: 0 upright, 1 bent downward, 2 lowest
    torso_tilt_deg: float  # upper-body pitch offset

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "arms": {a: s.to_dict() for a, s in self.arms.items()},
                "base_odometry": {"x_m": round(self.base_odometry[0], 3), "y_m": round(self.base_odometry[1], 3),
                                  "yaw_deg": round(self.base_odometry[2], 1)},
                "torso": {"translate": round(self.torso_translate, 3), "tilt_deg": round(self.torso_tilt_deg, 1)}}


@dataclass
class Observation:
    step: int  # control step
    images: Dict[str, np.ndarray]  # camera name -> HxWx3 uint8 (head, left_wrist, right_wrist)
    state: RobotState
    steps_remaining: Optional[int] = None  # control steps left before the episode ends


@dataclass
class MotionRequest:
    """A planner's request to the motion layer; ``kind`` in move_eef | move_base | move_torso | hold."""

    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass
class MotionPlan:
    """What an accepted request will play: the tool result is written from it before anything moves."""

    kind: str
    steps: int  # control steps (waypoints) the motion plays
    dt: float
    gripper_steps: int = 0  # the last ``gripper_steps`` move the jaws, once the arms have arrived
    approximate: bool = False  # closed-loop base / torso motions: ``steps`` is the planned reference, the tracker may take longer
    reports: Optional[str] = None  # what the next observation's arrival check measures ("grasp point", "base", "torso")

    @property
    def seconds(self) -> float:
        return self.steps * self.dt

    def accepted_text(self) -> str:
        about = "about " if self.approximate else ""
        text = f"Accepted: playing {about}{self.steps} waypoints ({self.seconds:.1f}s)."
        if self.gripper_steps:
            text += f" The last {self.gripper_steps} move the gripper, once the arm has arrived."
        if self.reports:
            text += f" The next observation reports how close the {self.reports} landed."
        return text


@dataclass
class MotionOutcome:
    ok: bool
    message: str  # summary for the trace and the code-agent API
    steps: int = 0
    details: Dict[str, Any] = field(default_factory=dict)  # "arrival": what the next observation reports


class MotionRejected(Exception):
    """A request refused before anything moves; the message goes back to the planner as the tool result.
    ``repairable``: a malformed or out-of-bounds request (counts toward the repair limit); otherwise an unreachable
    pose, which the planner may simply replace. ``detail``: why, for the trace only."""

    def __init__(self, message: str, repairable: bool = True, detail: Optional[str] = None):
        super().__init__(message)
        self.repairable = repairable
        self.detail = detail


@dataclass
class Decision:
    """A planner's next step: a motion, or the end of its episode (give_up / done / budget / capability failure)."""

    kind: str  # "motion" | "stop"
    request: Optional[MotionRequest] = None
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


#: planner stops that end the episode as a failure whatever the scene looks like (RoboDojo marks them failed)
PLANNER_STOPS = ("give_up", "done", "call_budget", "capability_failure")


def arrival_lines(details: Optional[Dict[str, Any]]) -> List[str]:
    """The arrival check of the previous motion, as the next observation prints it."""
    arrival = (details or {}).get("arrival") or {}
    kind = arrival.get("kind")
    if kind == "move_eef":
        parts = [f"{arm} {v['pos_mm']:.0f} mm and {v['rot_deg']:.1f} deg away" for arm, v in sorted((arrival.get("arms") or {}).items())]
        return ["Arrival check for the previous move_eef target: " + "; ".join(parts) + "."] if parts else []
    if kind == "move_base":
        return [f"Arrival check for the previous move_base target: {arrival['pos_mm']:.0f} mm and {arrival['yaw_deg']:.1f} deg away."]
    if kind == "move_torso":
        parts = []
        if arrival.get("translate") is not None:
            parts.append(f"translate {arrival['translate']:.2f}")
        if arrival.get("tilt_deg") is not None:
            parts.append(f"tilt {arrival['tilt_deg']:.1f} deg")
        return ["Arrival check for the previous move_torso target: " + " and ".join(parts) + " away."] if parts else []
    return []
