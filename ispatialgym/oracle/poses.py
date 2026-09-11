"""Small pose utilities shared by the oracle stack.

Conventions: positions are ``(3,)`` float tensors, quaternions are ``(4,)`` **xyzw** (OmniGibson
convention). cuRobo uses wxyz; conversion happens only inside :mod:`curobo_planner`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch as th

import omnigibson.utils.transform_utils as T

Pose = Tuple[th.Tensor, th.Tensor]  # (pos xyz, quat xyzw)


def as_tensor(x, dtype=th.float32) -> th.Tensor:
    if isinstance(x, th.Tensor):
        return x.detach().to(dtype=dtype, device="cpu")
    return th.as_tensor(x, dtype=dtype)


def identity_pose() -> Pose:
    return th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])


def pose_to_mat(pose: Pose) -> th.Tensor:
    return T.pose2mat((as_tensor(pose[0]), as_tensor(pose[1])))


def mat_to_pose(mat: th.Tensor) -> Pose:
    pos, quat = T.mat2pose(mat)
    return pos, quat


def compose(a: Pose, b: Pose) -> Pose:
    """Return a * b (apply b in a's frame): ``T_a @ T_b``."""
    return mat_to_pose(pose_to_mat(a) @ pose_to_mat(b))


def invert(a: Pose) -> Pose:
    return mat_to_pose(T.pose_inv(pose_to_mat(a)))


def relative(target: Pose, frame: Pose) -> Pose:
    """Pose of ``target`` expressed in ``frame`` (both given in the same parent frame)."""
    return mat_to_pose(T.pose_inv(pose_to_mat(frame)) @ pose_to_mat(target))


def transform_points(pose: Pose, pts: th.Tensor) -> th.Tensor:
    mat = pose_to_mat(pose)
    return pts @ mat[:3, :3].T + mat[:3, 3]


def quat_xyzw_to_wxyz(q: th.Tensor) -> th.Tensor:
    return q[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q: th.Tensor) -> th.Tensor:
    return q[..., [1, 2, 3, 0]]


def quat_angle(q1: th.Tensor, q2: th.Tensor) -> float:
    """Geodesic angle (rad) between two xyzw quaternions."""
    d = float(th.abs(th.dot(as_tensor(q1) / th.linalg.norm(as_tensor(q1)), as_tensor(q2) / th.linalg.norm(as_tensor(q2)))))
    return 2.0 * math.acos(min(1.0, d))


def slerp(q1: th.Tensor, q2: th.Tensor, t: float) -> th.Tensor:
    q1 = as_tensor(q1) / th.linalg.norm(as_tensor(q1))
    q2 = as_tensor(q2) / th.linalg.norm(as_tensor(q2))
    d = float(th.dot(q1, q2))
    if d < 0.0:
        q2 = -q2
        d = -d
    if d > 0.9995:
        out = q1 + t * (q2 - q1)
        return out / th.linalg.norm(out)
    theta0 = math.acos(d)
    theta = theta0 * t
    s2 = math.sin(theta) / math.sin(theta0)
    s1 = math.cos(theta) - d * s2
    return s1 * q1 + s2 * q2


def quat_with_z_axis(z_axis: th.Tensor, reference_quat: th.Tensor) -> th.Tensor:
    """Orientation whose z-axis equals ``z_axis`` (world) and whose x-axis is as close as possible to the
    x-axis of ``reference_quat`` (used for approach directions while keeping the current wrist yaw)."""
    z = as_tensor(z_axis)
    z = z / th.linalg.norm(z)
    x_ref = T.quat2mat(as_tensor(reference_quat))[:, 0]
    x = x_ref - th.dot(x_ref, z) * z
    if th.linalg.norm(x) < 1e-6:  # reference x parallel to z: fall back to reference y
        y_ref = T.quat2mat(as_tensor(reference_quat))[:, 1]
        x = th.cross(y_ref, z, dim=0)
    x = x / th.linalg.norm(x)
    y = th.cross(z, x, dim=0)
    return T.mat2quat(th.stack([x, y, z], dim=1))


def top_down_quat(reference_quat: th.Tensor) -> th.Tensor:
    """Gripper approach axis pointing straight down, wrist yaw taken from ``reference_quat``."""
    return quat_with_z_axis(th.tensor([0.0, 0.0, -1.0]), reference_quat)


def yaw_of(quat_xyzw: th.Tensor) -> float:
    return float(T.quat2euler(as_tensor(quat_xyzw))[2])


def quat_from_yaw(yaw: float) -> th.Tensor:
    return T.euler2quat(th.tensor([0.0, 0.0, float(yaw)]))


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def pose2d_to_pose(x: float, y: float, yaw: float, z: float = 0.0) -> Pose:
    return th.tensor([x, y, z], dtype=th.float32), quat_from_yaw(yaw)


def pose_to_pose2d(pose: Pose) -> Tuple[float, float, float]:
    p = as_tensor(pose[0])
    return float(p[0]), float(p[1]), yaw_of(pose[1])


@dataclass
class FramedPose:
    """A pose together with the frame it is expressed in.

    ``frame`` is one of ``"world"``, ``"base"`` (robot base_link), ``"eef:<arm>"``, ``"obj:<name>"``.
    Resolution to world happens in :class:`ispatialgym.oracle.skills.Skills` which has access to
    the robot and the scene.
    """

    pos: th.Tensor
    quat: th.Tensor
    frame: str = "world"

    @staticmethod
    def create(pos, quat=None, frame: str = "world") -> "FramedPose":
        pos = as_tensor(pos)
        quat = th.tensor([0.0, 0.0, 0.0, 1.0]) if quat is None else as_tensor(quat)
        return FramedPose(pos=pos, quat=quat, frame=frame)

    @property
    def pose(self) -> Pose:
        return self.pos, self.quat
