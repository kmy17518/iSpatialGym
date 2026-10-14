"""Harness configuration: one JSON file per episode (``ISG_HARNESS_CONFIG``), written by ``harness.run``."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

ENV_VAR = "ISG_HARNESS_CONFIG"


@dataclass
class MotionLimits:
    """Bounds of the non-learned motion layer (``core.motion``)."""

    max_duration_s: float = 10.0  # one move_eef call plays at most this long (RoboProbe's playout cap)
    arm_velocity_scale: float = 0.25  # per-step joint budget = this x the joint's velocity limit x dt (the oracle's arm planning scale)
    gripper_wait_steps: int = 20  # the jaw phase waits at most this long for the fingers to stop or a grasp to attach (attaching takes 9)
    gripper_partial_steps: int = 12  # a partial opening holds its command this long
    base_max_m: float = 1.5  # one move_base call: largest translation
    base_max_turn_deg: float = 180.0  # ... and rotation
    torso_tilt_max_deg: float = 30.0  # |tilt offset| allowed by move_torso


@dataclass
class HarnessConfig:
    planner: str = "l3_tools"
    model: str = "openai/gpt-6-astra"  # provider/model: openai/..., anthropic/..., compat/..., scripted:<file.json>
    model_options: Dict[str, Any] = field(default_factory=dict)  # effort, max_output_tokens, base_url, timeout_s, ...
    max_calls: int = 170  # LLM calls per episode, every response counted (RoboProbe Inspect EEF: 170)
    max_steps: int = 18000  # control steps per episode (30 Hz), reported to the model as env steps remaining
    max_repairs: int = 3  # malformed calls on one observation before the episode ends as a capability failure
    image_horizon: Optional[int] = 2  # camera images kept in the newest N observation turns (None: all)
    image_size: Optional[int] = None  # longest side of every image sent to the model (None: the camera's own resolution)
    jpeg_quality: int = 95
    cameras: List[str] = field(default_factory=lambda: ["head", "left_wrist", "right_wrist"])
    goal_image: Optional[str] = None  # PNG of the goal (head camera at the settled goal state)
    alias: Optional[str] = None  # the instance's task alias (instruction templates, recipe)
    instruction: Optional[str] = None  # explicit task statement (overrides the template)
    instruction_level: str = "alias"  # "family" (no object names) | "alias" (object categories from the alias)
    use_recipe: bool = True  # append the family's TASK RECIPE to the Goal turn
    stop_on_success: bool = True  # end the stream at the task's official success (the environment ends it, not the model)
    allow_done: bool = False  # probe only: offer a `done` tool (ends the episode as a failure, like give_up)
    disable_give_up: bool = False  # probe only: no give_up, the model acts until a budget ends the episode
    limits: MotionLimits = field(default_factory=MotionLimits)
    perturb: Dict[str, Any] = field(default_factory=dict)  # core.perturb.Perturbations fields
    trace_dir: Optional[str] = None  # default: <run_isg out dir>/harness
    seed: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HarnessConfig":
        d = dict(d or {})
        limits = MotionLimits(**(d.pop("limits", None) or {}))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown harness config keys: {unknown}")
        return cls(limits=limits, **d)

    @classmethod
    def from_file(cls, path: str) -> "HarnessConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_env(cls) -> "HarnessConfig":
        path = os.environ.get(ENV_VAR)
        return cls.from_file(path) if path else cls()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
