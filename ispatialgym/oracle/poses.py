"""Small pose utilities shared by the oracle stack.

Conventions: positions are ``(3,)`` float tensors, quaternions are ``(4,)`` **xyzw** (OmniGibson
convention). cuRobo uses wxyz; conversion happens only inside :mod:`curobo_planner`.

Everything here moved to :mod:`ispatialgym.robot.conventions` (TODO F5); this module re-exports the v1 names.
"""

from __future__ import annotations

__all__ = [
    "FramedPose", "Pose", "as_tensor", "compose", "identity_pose", "invert", "mat_to_pose", "pose2d_to_pose",
    "pose_to_mat", "pose_to_pose2d", "quat_angle", "quat_from_yaw", "quat_with_z_axis", "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz", "relative", "slerp", "top_down_quat", "transform_points", "wrap_angle", "yaw_of",
]

from ispatialgym.robot.conventions import (  # noqa: E402,F401  (moved to robot/conventions, F5)
    FramedPose,
    Pose,
    as_tensor,
    compose,
    identity_pose,
    invert,
    mat_to_pose,
    pose2d_to_pose,
    pose_to_mat,
    pose_to_pose2d,
    quat_angle,
    quat_from_yaw,
    quat_with_z_axis,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    relative,
    slerp,
    top_down_quat,
    transform_points,
    wrap_angle,
    yaw_of,
)
