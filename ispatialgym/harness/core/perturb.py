"""RoboProbe's interaction-time probes, unannounced to the planner: flipped / blacked-out cameras and a mirrored position
frame on what the planner sees and commands, a bias on where its gripper targets are executed. Ground truth (success,
recordings) is never perturbed."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

from ispatialgym.harness.core.types import MotionRequest, RobotState


@dataclass
class Perturbations:
    flip_vision_ud: bool = False  # images upside down
    flip_vision_lr: bool = False  # images mirrored left-right
    mask_cameras: List[str] = field(default_factory=list)  # cameras sent as a black image, labelled MASKED (e.g. ["head"])
    negate_xyz: bool = False  # the model's position frame is mirrored: reported and commanded x, y, z are negated together
    jitter_dx: float = 0.0  # fixed base-frame offset added to every executed grasp-point target (m)
    jitter_dy: float = 0.0
    jitter_dz: float = 0.0
    jitter_sphere_r: float = 0.0  # plus an offset of this length in a random direction ...
    jitter_sphere_mean_r: float = 0.0  # ... or of exponential length with this mean (wins over jitter_sphere_r)
    jitter_sphere_per_move: bool = False  # redraw the random offset for every move (default: once per episode)
    seed: Optional[int] = 0

    def __post_init__(self):
        for k in ("jitter_sphere_r", "jitter_sphere_mean_r"):
            if getattr(self, k) < 0:
                raise ValueError(f"{k} must be >= 0")
        self.rng = np.random.default_rng(self.seed)
        self.sphere = np.zeros(3)
        self._draw_sphere()

    @classmethod
    def from_dict(cls, d) -> "Perturbations":
        return cls(**(d or {}))

    @property
    def active(self) -> bool:
        return bool(self.flip_vision_ud or self.flip_vision_lr or self.mask_cameras or self.negate_xyz or self.jitter_dx or self.jitter_dy
                    or self.jitter_dz or self.jitter_sphere_r or self.jitter_sphere_mean_r)

    # -- what the planner sees ------------------------------------------------------------------------------------
    def is_masked(self, name: str) -> bool:
        return name in self.mask_cameras

    def view(self, name: str, img: np.ndarray) -> np.ndarray:
        if self.flip_vision_ud:
            img = img[::-1]
        if self.flip_vision_lr:
            img = img[:, ::-1]
        if self.is_masked(name):
            return np.zeros_like(img)
        return np.ascontiguousarray(img)

    def observed(self, state: RobotState) -> RobotState:
        """The state as the planner reads it (positions negated in a mirrored frame)."""
        if not self.negate_xyz:
            return state
        arms = {arm: replace(s, grasp={k: (-v if k in ("x", "y", "z") else v) for k, v in s.grasp.items()}) for arm, s in state.arms.items()}
        x, y, yaw = state.base_odometry
        return replace(state, arms=arms, base_odometry=(-x, -y, yaw))

    # -- what the robot executes ----------------------------------------------------------------------------------
    def _draw_sphere(self) -> None:
        if self.jitter_sphere_mean_r > 0:
            radius = float(self.rng.exponential(self.jitter_sphere_mean_r))
        elif self.jitter_sphere_r > 0:
            radius = float(self.jitter_sphere_r)
        else:
            self.sphere = np.zeros(3)
            return
        v = self.rng.normal(size=3)
        n = float(np.linalg.norm(v))
        v = np.array([1.0, 0.0, 0.0]) if not np.isfinite(n) or n == 0.0 else v / n
        self.sphere = v * radius

    def jitter(self) -> Tuple[float, float, float]:
        return (self.jitter_dx + float(self.sphere[0]), self.jitter_dy + float(self.sphere[1]), self.jitter_dz + float(self.sphere[2]))

    def request(self, req: MotionRequest) -> MotionRequest:
        """The request as executed: model-frame positions mapped back to the base frame, the gripper bias attached."""
        if req.kind == "move_eef":
            if self.jitter_sphere_per_move:
                self._draw_sphere()
            j = self.jitter()
            if not (self.negate_xyz or any(j)):
                return req
            r = copy.deepcopy(req)
            if self.negate_xyz:
                r.params["targets"] = {k: (-float(v) if k.split("_", 1)[1] in ("x", "y", "z") else v) for k, v in r.params["targets"].items()}
            if any(j):
                r.params["jitter_m"] = list(j)
            return r
        if req.kind == "move_base" and self.negate_xyz:
            r = copy.deepcopy(req)
            r.params["forward_m"] = -float(r.params.get("forward_m") or 0.0)
            r.params["left_m"] = -float(r.params.get("left_m") or 0.0)
            return r
        return req
