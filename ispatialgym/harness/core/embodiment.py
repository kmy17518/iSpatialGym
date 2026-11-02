"""R1Pro as the harness sees it (needs the simulator): robot-only state in the base frame, the camera frames of an
observation, and conversions between the base frame and the world. Reads go through ``oracle.motions.Motions``'s
robot accessors; nothing here reads object or scene state."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.harness.core import pose
from ispatialgym.harness.core.types import ARMS, ArmState, RobotState
from ispatialgym.robot import conventions as C

Pose = Tuple[th.Tensor, th.Tensor]

#: harness camera name -> R1Pro camera link (sensor key ``<robot>:<link>:Camera:0``)
CAMERA_LINKS = {"head": "zed_link", "left_wrist": "left_realsense_link", "right_wrist": "right_realsense_link"}


def _np_rgb(x: Any) -> np.ndarray:
    a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a[..., :3])


class R1ProEmbodiment:
    def __init__(self, executor):
        self.executor = executor
        self.motions = executor.motions
        self.robot = self.motions.robot
        self.backend = self.motions.backend
        self.dt = float(self.motions.dt)
        self.start_pose = self.base_pose_world()

    # -- frames ---------------------------------------------------------------------------------------------------
    def base_pose_world(self) -> Pose:
        pos, quat = self.robot.get_position_orientation()
        return pos.float().clone(), quat.float().clone()

    def to_base(self, pose_world: Pose) -> Pose:
        return C.relative(pose_world, self.base_pose_world())

    def to_world(self, pose_base: Pose) -> Pose:
        return C.compose(self.base_pose_world(), pose_base)

    # -- state ----------------------------------------------------------------------------------------------------
    def q_full(self) -> th.Tensor:
        return self.motions.q_full()

    def eef_base(self, arm: str) -> Pose:
        return self.to_base(self.motions.eef_pose(arm))

    def grasp_values(self, arm: str) -> Dict[str, float]:
        """The arm's six Cartesian dimensions (grasp point, angles) in the base frame, as move_eef names them."""
        p, q = self.eef_base(arm)
        return pose.pose_to_values(p.double().cpu().numpy(), q.double().cpu().numpy())

    def arm_joints(self, arm: str) -> Tuple[float, ...]:
        q = self.robot.get_joint_positions()[self.robot.arm_control_idx[arm]]
        return tuple(float(v) for v in q)

    def gripper_opening(self, arm: str) -> float:
        idx = self.robot.gripper_control_idx[arm]
        q = float(self.robot.get_joint_positions()[idx].float().mean())
        lo, hi = self.motions._gripper_limits(arm)
        return float(min(1.0, max(0.0, (q - float(lo)) / max(float(hi) - float(lo), 1e-6))))

    def arm_step_limits(self, arm: str, scale: float) -> np.ndarray:
        """Per-control-step budget of each arm joint: its velocity limit x ``scale`` x dt (rad)."""
        names = list(self.robot.joints.keys())
        idx = [int(i) for i in self.robot.arm_control_idx[arm]]
        vel = np.array([float(self.robot.joints[names[i]].max_velocity) for i in idx])
        return np.maximum(vel * float(scale) * self.dt, 1e-4)

    def torso(self) -> Tuple[float, float]:
        from ispatialgym.oracle.trunk import TrunkState

        ts = TrunkState.from_q(self.robot.get_joint_positions()[self.robot.trunk_control_idx].float())
        return float(ts.translate), float(ts.tilt)

    def odometry(self) -> Tuple[float, float, float]:
        """Base pose relative to the episode start, in the start frame: x, y [m], yaw [deg]."""
        p, q = C.relative(self.base_pose_world(), self.start_pose)
        return float(p[0]), float(p[1]), math.degrees(C.yaw_of(q))

    def state(self, step: int) -> RobotState:
        arms = {arm: ArmState(grasp=self.grasp_values(arm), gripper_opening=self.gripper_opening(arm), joints=self.arm_joints(arm)) for arm in ARMS}
        translate, tilt = self.torso()
        return RobotState(step=step, arms=arms, base_odometry=self.odometry(), torso_translate=translate, torso_tilt_deg=math.degrees(tilt))

    def link_position_base(self, link: str) -> Optional[Tuple[float, float, float]]:
        prim = getattr(self.robot, "links", {}).get(link)
        if prim is None:
            return None
        p, _ = self.to_base(tuple(t.float() for t in prim.get_position_orientation()))
        return float(p[0]), float(p[1]), float(p[2])

    def mounting(self) -> Dict[str, Tuple[float, float, float]]:
        """Shoulders and head camera in the base frame, as the episode starts (robot-only geometry)."""
        out = {}
        for name, link in (("left shoulder", "left_arm_link1"), ("right shoulder", "right_arm_link1"), ("head camera", CAMERA_LINKS["head"])):
            p = self.link_position_base(link)
            if p is not None:
                out[name] = p
        return out

    def images(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        robot_obs = obs.get(self.robot.name)
        if not isinstance(robot_obs, dict):
            robot_obs = next(v for v in obs.values() if isinstance(v, dict) and "proprio" in v)
        out = {}
        for name, link in CAMERA_LINKS.items():
            suffix = f"{link}:Camera:0"
            key = next((k for k in robot_obs if k.endswith(suffix)), None)
            if key is not None and isinstance(robot_obs[key], dict) and "rgb" in robot_obs[key]:
                out[name] = _np_rgb(robot_obs[key]["rgb"])
        return out
