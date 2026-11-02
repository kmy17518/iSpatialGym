"""What the planner sees: one observation turn (the goal image, the state named as the tools take it, env steps remaining,
the arrival check of the previous motion, labelled camera images) and the pinned goal image of the first turn."""

from __future__ import annotations

import base64
import io
from typing import List, Optional, Sequence

import numpy as np

from ispatialgym.harness.config import HarnessConfig
from ispatialgym.harness.core import pose
from ispatialgym.harness.core.perturb import Perturbations
from ispatialgym.harness.core.types import Observation, RobotState
from ispatialgym.harness.llm.base import Content, ImagePart, TextPart

GOAL_LABEL = "goal image (the head camera once the task is done)"
GRASP_HEADER = "Base-frame grasp-point state, named as move_eef takes them (metres; degrees from the level forward reference):"
BASE_HEADER = "Base pose relative to where the episode started, for context (metres; degrees; move_base moves relative to the current pose):"
TORSO_HEADER = "Torso, named as move_torso takes it:"
JOINT_HEADER = "Arm joint angles, radians, for context; not commandable:"


def encode_rgb(rgb: np.ndarray, size: Optional[int], quality: int = 95) -> bytes:
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8))
    if size:
        w, h = img.size
        s = float(size) / max(w, h)
        if s < 1.0:
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def state_lines(state: RobotState) -> List[str]:
    """The state block, every commandable dimension under the name its tool takes, in the model's frame."""
    lines = [GRASP_HEADER]
    for label in pose.labels():
        arm, axis = pose.split_label(label)
        s = state.arms.get(arm)
        if s is None:
            lines.append(f"{label}=unavailable")
        elif axis == "gripper":
            lines.append(f"{label}={s.gripper_opening:.2f}")
        elif axis in pose.ANGLE_AXES:
            lines.append(f"{label}={round(s.grasp[axis], 1) + 0.0:.1f}")
        else:
            lines.append(f"{label}={s.grasp[axis]:.4f}")
    x, y, yaw = state.base_odometry
    lines += [BASE_HEADER, f"base_x={x:.4f}", f"base_y={y:.4f}", f"base_yaw_deg={yaw:.1f}"]
    lines += [TORSO_HEADER, f"torso_translate={state.torso_translate:.2f}", f"torso_tilt_deg={state.torso_tilt_deg:.1f}"]
    lines.append(JOINT_HEADER)
    for arm in pose.ARMS:
        s = state.arms.get(arm)
        for i, q in enumerate(s.joints if s is not None else ()):
            lines.append(f"{arm}_arm_joint{i + 1}={q:.4f}")
    return lines


class ObservationEncoder:
    def __init__(self, cfg: HarnessConfig, perturb: Optional[Perturbations] = None):
        self.cfg = cfg
        self.perturb = perturb or Perturbations()

    def image_part(self, label: str, rgb: np.ndarray, pinned: bool = False) -> ImagePart:
        jpeg = encode_rgb(rgb, self.cfg.image_size, self.cfg.jpeg_quality)
        return ImagePart(label=label, data_b64=base64.b64encode(jpeg).decode("ascii"), pinned=pinned)

    def goal_part(self, rgb: np.ndarray, pinned: bool = True) -> ImagePart:
        """The first turn's copy is pinned; an observation's copy ages out with the image horizon like the camera images."""
        return self.image_part(GOAL_LABEL, rgb, pinned=pinned)

    def camera_parts(self, obs: Observation, turn: int) -> List[ImagePart]:
        parts = []
        for name in self.cfg.cameras:
            img = obs.images.get(name)
            if img is None:
                continue
            label = f"camera '{name}' (step {turn})" + (", MASKED" if self.perturb.is_masked(name) else "")
            parts.append(self.image_part(label, self.perturb.view(name, img)))
        return parts

    def observation_content(self, obs: Observation, turn: int, extra_lines: Sequence[str] = (), prefix: str = "",
                            goal: Optional[ImagePart] = None) -> Content:
        """``goal`` (the goal image) opens the turn where RoboProbe's ``Instruction:`` line stands."""
        lines = ([prefix] if prefix else []) + state_lines(self.perturb.observed(obs.state))
        if obs.steps_remaining is not None:
            lines.append(f"Env steps remaining before the episode ends: {obs.steps_remaining}")
        lines += list(extra_lines)
        return ([goal] if goal is not None else []) + [TextPart("\n".join(lines))] + list(self.camera_parts(obs, turn))
