"""Conventions shared by the sampler and the oracle (oracle doc §3.5, TODO F5).

Every function that turns sidecar facts into poses lives here. The sampler validates the *outputs* of these functions
(IK / collision), the oracle tracks the *same outputs*; neither package may re-implement one. Everything is pure:
tensors in, tensors out, no simulator, no cuRobo. The only file access is the grasp library (``grasp_pose``).

Frames (oracle doc §3.1)
------------------------
* ``world``      OmniGibson world frame, z up, floor at z = 0 for the scenes in use.
* ``base``       robot ``base_link``: x ahead, y = robot's left, z up; its origin stands ``NOMINAL_BASE_Z`` above the
                 floor. A *pose2d* ``(x, y, yaw)`` is the planar world pose of ``base_link``.
* ``eef:<arm>``  ``<arm>_eef_link``: **a = approach = eef +z** (points from the wrist toward the fingertips), **c =
                 closing = eef +y** (the pads close along ±y; the axis is undirected, ``c`` and ``-c`` describe the
                 same pinch up to a 180° wrist turn), eef +x = c × a. The fingertips are ``FINGERTIP_LENGTH`` ahead
                 of the origin along a; the pad centre is ``CONTACT_CENTER_Z`` (negative: behind the origin) along a.
* ``obj:<name>`` the object's root (``base_link``) frame; grasp-library entries are eef poses in this frame.
* camera         the USD camera prim ``<robot>:zed_link:Camera:0``: looks along **-z**, +y up (``CameraModel``).

Positions are ``(3,)`` float tensors, quaternions ``(4,)`` **xyzw**, a ``Pose`` is the tuple ``(pos, quat)``;
``FramedPose`` adds the frame tag. Unless a docstring says otherwise a function returns its result in the frame its
pose arguments were given in (everything is frame-agnostic rigid geometry).

Waypoint rule (spec §3.9)
-------------------------
Every waypoint generator returns the ``n`` poses at fractions ``k/n``, ``k = 1..n`` of the motion: the start pose is
**not** repeated, the last entry **is** the goal. ``endpoints=True`` prepends the start (full path, ``n + 1`` poses).
So "3 intermediate waypoints" = ``line_waypoints(p0, p1, 4)`` (three screening poses + the exact-IK goal), the push
arc "δ/2 and the goal" = ``arc_waypoints(..., n=2)``, "5 samples" of a trunk sweep = ``trunk_sweep(a, b, 5)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch as th

from ispatialgym.oracle import r1pro_model as M

Pose = Tuple[th.Tensor, th.Tensor]  # (pos xyz, quat xyzw)
Pose2D = Tuple[float, float, float]  # (x, y, yaw) of base_link in the world

__all__ = [
    # types / frames
    "Pose", "Pose2D", "FramedPose", "as_tensor", "as_pose", "identity_pose", "IDENTITY_QUAT",
    # quaternion / pose primitives
    "unit", "qmul", "qconj", "qinv", "quat_to_mat", "mat_to_quat", "quat_rotate", "quat_from_axis_angle",
    "quat_from_euler", "euler_of", "quat_from_yaw", "yaw_of", "wrap_angle", "quat_angle", "slerp", "quat_with_z_axis",
    "top_down_quat", "quat_xyzw_to_wxyz", "quat_wxyz_to_xyzw", "pose_to_mat", "mat_to_pose", "compose", "invert",
    "relative", "transform_points", "pose2d_to_pose", "pose_to_pose2d",
    # eef frame
    "approach_axis", "closing_axis", "eef_quat", "pad_center", "fingertip",
    # gripper constants
    "FINGER_OPEN", "FINGER_CLOSED", "CONTACT_CENTER_Z", "FINGERTIP_LENGTH", "PAD_Z_RANGE", "GRIPPER_OPEN_M",
    "GRIPPER_MAX_OPENING_M", "HANDLE_EEF_OFFSET", "PRE_DIST", "pad_clearance",
    # grasp library -> eef
    "grasp_entry", "grasp_pose", "grasp_pose_from_transform", "object_in_eef", "rest_quat",
    # pre / contact poses
    "pre_pose", "handle_eef", "contact_eef", "fingertip_eef", "default_closing_axis",
    # waypoints
    "line_waypoints", "slerp_waypoints", "orbit_waypoints", "arc_waypoints", "HandleWaypoint", "handle_waypoints",
    "flip_waypoints",
    # stances / transitions / base follow
    "TransitionPoses", "transition_poses", "transition_standoffs", "trunk_sweep", "eef_base_transform",
    "base_follow_poses", "base_follow_residual",
    # trunk (JoyLo)
    "TORSO_JOINT_LIMITS", "translate_to_torso_q", "torso_q_to_translate_tilt", "torso_q",
    # head camera / torso FK
    "CAMERA_IN_ZED", "camera_offset_in_zed", "torso_fk", "camera_pose", "shoulder_position",
    # configurations
    "NOMINAL_BASE_Z", "JOINT_INDEX", "q_with_base_pose", "base_pose2d_of_q",
    # blocks
    "BLOCK_EDGE", "CubeGrasp", "cube_grasp", "cube_pick_tilt",
]


# ==================================================================================================================
# 1. Types, frames and pose primitives
# ==================================================================================================================


def as_tensor(x, dtype=th.float32) -> th.Tensor:
    """CPU tensor of ``dtype`` from a tensor / list / tuple (detached)."""
    if isinstance(x, th.Tensor):
        return x.detach().to(dtype=dtype, device="cpu")
    return th.as_tensor(x, dtype=dtype)


IDENTITY_QUAT = th.tensor([0.0, 0.0, 0.0, 1.0])


def identity_pose() -> Pose:
    return th.zeros(3), IDENTITY_QUAT.clone()


@dataclass
class FramedPose:
    """A pose together with the frame it is expressed in.

    ``frame`` is ``"world"``, ``"base"`` (robot ``base_link``), ``"eef:<arm>"`` or ``"obj:<name>"`` (oracle doc §3.1).
    Resolution to the world frame is the caller's business (it needs the simulator or the sidecar); every function in
    this module treats a ``FramedPose`` like the plain ``(pos, quat)`` it wraps and returns the frame unchanged.
    """

    pos: th.Tensor
    quat: th.Tensor
    frame: str = "world"

    @staticmethod
    def create(pos, quat=None, frame: str = "world") -> "FramedPose":
        pos = as_tensor(pos)
        quat = IDENTITY_QUAT.clone() if quat is None else as_tensor(quat)
        return FramedPose(pos=pos, quat=quat, frame=frame)

    @property
    def pose(self) -> Pose:
        return self.pos, self.quat


PoseLike = Union[FramedPose, Pose, Sequence]


def as_pose(pose: PoseLike) -> Pose:
    """``(pos (3,), quat (4,))`` float tensors from a ``Pose`` tuple / list or a :class:`FramedPose`."""
    if isinstance(pose, FramedPose):
        return as_tensor(pose.pos), as_tensor(pose.quat)
    pos, quat = pose
    return as_tensor(pos), as_tensor(quat)


def _like(template: PoseLike, pos: th.Tensor, quat: th.Tensor):
    """Return ``(pos, quat)`` wrapped like ``template`` (a FramedPose keeps its frame)."""
    if isinstance(template, FramedPose):
        return FramedPose(pos=pos, quat=quat, frame=template.frame)
    return pos, quat


def unit(v) -> th.Tensor:
    """Normalised copy of a vector; raises on a (near-)zero vector."""
    v = as_tensor(v)
    n = float(th.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("cannot normalise a zero vector")
    return v / n


def qmul(a, b) -> th.Tensor:
    """Hamilton product ``a ⊗ b`` of xyzw quaternions: ``R(qmul(a, b)) = R(a) @ R(b)`` (apply ``b`` first)."""
    a, b = as_tensor(a), as_tensor(b)
    x1, y1, z1, w1 = a[0], a[1], a[2], a[3]
    x0, y0, z0, w0 = b[0], b[1], b[2], b[3]
    return th.stack(
        [
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ]
    )


def qconj(q) -> th.Tensor:
    q = as_tensor(q)
    return th.cat([-q[:3], q[3:]])


def qinv(q) -> th.Tensor:
    q = as_tensor(q)
    return qconj(q) / th.dot(q, q)


def quat_to_mat(q) -> th.Tensor:
    """(3, 3) rotation matrix of a xyzw quaternion (normalised first)."""
    q = as_tensor(q, th.float64)
    q = q / th.linalg.norm(q)
    x, y, z, w = q.tolist()
    R = th.tensor(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=th.float64,
    )
    return R.to(th.float32)


def mat_to_quat(R) -> th.Tensor:
    """xyzw quaternion of a (3, 3) rotation matrix (same branch / sign convention as OmniGibson's ``mat2quat``)."""
    m = as_tensor(R, th.float64)
    m00, m01, m02 = float(m[0, 0]), float(m[0, 1]), float(m[0, 2])
    m10, m11, m12 = float(m[1, 0]), float(m[1, 1]), float(m[1, 2])
    m20, m21, m22 = float(m[2, 0]), float(m[2, 1]), float(m[2, 2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        sq = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * sq, (m21 - m12) / sq, (m02 - m20) / sq, (m10 - m01) / sq
    elif m00 > m11 and m00 > m22:
        sq = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (m21 - m12) / sq, 0.25 * sq, (m01 + m10) / sq, (m02 + m20) / sq
    elif m11 > m22:
        sq = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (m02 - m20) / sq, (m01 + m10) / sq, 0.25 * sq, (m12 + m21) / sq
    else:
        sq = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (m10 - m01) / sq, (m02 + m20) / sq, (m12 + m21) / sq, 0.25 * sq
    q = th.tensor([x, y, z, w], dtype=th.float64)
    return (q / th.linalg.norm(q)).to(th.float32)


def quat_rotate(q, v) -> th.Tensor:
    """``R(q) @ v`` for one ``(3,)`` vector or a ``(N, 3)`` batch."""
    R = quat_to_mat(q)
    v = as_tensor(v)
    return v @ R.T if v.ndim == 2 else R @ v


def quat_from_axis_angle(axis, angle: float) -> th.Tensor:
    """Right-hand rotation by ``angle`` (rad) about ``axis`` (normalised here)."""
    a = unit(axis).to(th.float64)
    h = 0.5 * float(angle)
    q = th.cat([a * math.sin(h), th.tensor([math.cos(h)], dtype=th.float64)])
    return q.to(th.float32)


def quat_from_euler(roll: float, pitch: float, yaw: float) -> th.Tensor:
    """xyzw quaternion of fixed-axis (URDF / OmniGibson) roll-pitch-yaw: ``R = R_z(yaw) R_y(pitch) R_x(roll)``."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    return th.tensor(
        [
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
            cy * cr * cp + sy * sr * sp,
        ],
        dtype=th.float32,
    )


def euler_of(q) -> Tuple[float, float, float]:
    """``(roll, pitch, yaw)`` in (-π, π] with ``R = R_z(yaw) R_y(pitch) R_x(roll)`` (OmniGibson ``quat2euler``)."""
    x, y, z, w = as_tensor(q, th.float64).tolist()
    roll = math.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
    return wrap_angle(roll), wrap_angle(pitch), wrap_angle(yaw)


def quat_from_yaw(yaw: float) -> th.Tensor:
    """Rotation about world z by ``yaw`` (rad)."""
    return th.tensor([0.0, 0.0, math.sin(0.5 * float(yaw)), math.cos(0.5 * float(yaw))], dtype=th.float32)


def yaw_of(quat) -> float:
    """Yaw (rad, in (-π, π]) of a xyzw quaternion."""
    return euler_of(quat)[2]


def wrap_angle(a: float) -> float:
    """Wrap to (-π, π]."""
    a = (float(a) + math.pi) % (2.0 * math.pi) - math.pi
    return a if a != -math.pi else math.pi


def quat_angle(q1, q2) -> float:
    """Geodesic angle (rad) between two xyzw quaternions (sign-invariant)."""
    a, b = as_tensor(q1, th.float64), as_tensor(q2, th.float64)
    d = abs(float(th.dot(a / th.linalg.norm(a), b / th.linalg.norm(b))))
    return 2.0 * math.acos(min(1.0, d))


def slerp(q1, q2, t: float) -> th.Tensor:
    """Shortest-path spherical interpolation; ``t = 0`` → ``q1``, ``t = 1`` → ``q2`` (up to sign)."""
    a = as_tensor(q1, th.float64)
    b = as_tensor(q2, th.float64)
    a, b = a / th.linalg.norm(a), b / th.linalg.norm(b)
    d = float(th.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        out = a + float(t) * (b - a)
        return (out / th.linalg.norm(out)).to(th.float32)
    theta0 = math.acos(d)
    theta = theta0 * float(t)
    s2 = math.sin(theta) / math.sin(theta0)
    s1 = math.cos(theta) - d * s2
    return (s1 * a + s2 * b).to(th.float32)


def quat_with_z_axis(z_axis, reference_quat) -> th.Tensor:
    """Orientation whose z axis is ``z_axis`` and whose x axis is as close as possible to that of ``reference_quat``
    (an approach direction keeping the current wrist yaw; falls back to the reference y axis when x ∥ z)."""
    z = unit(z_axis)
    Rr = quat_to_mat(reference_quat)
    x = Rr[:, 0] - th.dot(Rr[:, 0], z) * z
    if float(th.linalg.norm(x)) < 1e-6:
        x = th.cross(Rr[:, 1], z, dim=0)
    x = unit(x)
    y = th.cross(z, x, dim=0)
    return mat_to_quat(th.stack([x, y, z], dim=1))


def top_down_quat(reference_quat) -> th.Tensor:
    """Approach axis straight down (a = -ẑ), wrist yaw taken from ``reference_quat``."""
    return quat_with_z_axis(th.tensor([0.0, 0.0, -1.0]), reference_quat)


def quat_xyzw_to_wxyz(q) -> th.Tensor:
    return as_tensor(q)[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q) -> th.Tensor:
    return as_tensor(q)[..., [1, 2, 3, 0]]


def pose_to_mat(pose: PoseLike) -> th.Tensor:
    """(4, 4) homogeneous matrix of a pose."""
    pos, quat = as_pose(pose)
    mat = th.eye(4)
    mat[:3, :3] = quat_to_mat(quat)
    mat[:3, 3] = pos
    return mat


def mat_to_pose(mat) -> Pose:
    mat = as_tensor(mat)
    return mat[:3, 3].clone(), mat_to_quat(mat[:3, :3])


def compose(a: PoseLike, b: PoseLike) -> Pose:
    """``T_a @ T_b``: pose ``b`` given in ``a``'s frame, expressed in ``a``'s parent frame (e.g. ``compose(obj_pose,
    eef_in_obj)`` = the eef in the world)."""
    pa, qa = as_pose(a)
    pb, qb = as_pose(b)
    return pa + quat_rotate(qa, pb), qmul(qa, qb)


def invert(a: PoseLike) -> Pose:
    pa, qa = as_pose(a)
    qi = qinv(qa)
    return -quat_rotate(qi, pa), qi


def relative(target: PoseLike, frame: PoseLike) -> Pose:
    """Pose of ``target`` expressed in ``frame`` (both given in the same parent frame): ``inv(T_frame) @ T_target``."""
    return compose(invert(frame), target)


def transform_points(pose: PoseLike, pts) -> th.Tensor:
    """Points ``(N, 3)`` given in the pose's frame, expressed in its parent frame."""
    pos, quat = as_pose(pose)
    return as_tensor(pts).reshape(-1, 3) @ quat_to_mat(quat).T + pos


def pose2d_to_pose(x: float, y: float, yaw: float, z: float = 0.0) -> Pose:
    return th.tensor([float(x), float(y), float(z)], dtype=th.float32), quat_from_yaw(yaw)


def pose_to_pose2d(pose: PoseLike) -> Pose2D:
    pos, quat = as_pose(pose)
    return float(pos[0]), float(pos[1]), yaw_of(quat)


# ==================================================================================================================
# 2. End-effector frame and gripper constants
# ==================================================================================================================

#: finger joint value of an open / closed gripper (both fingers, both arms).
FINGER_OPEN = 0.05
FINGER_CLOSED = 0.0
#: pad centre along the eef approach axis (m): the pads span ``PAD_Z_RANGE`` in the eef frame, so their centre lies
#: 1.19 cm *behind* the eef origin (toward the wrist). ``contact_eef`` puts this point on a push / brace contact.
#: (R1Pro right gripper, measured for the grasp library: ``r1pro_right_gripper_eef.json`` ``contact_center_z``.)
CONTACT_CENTER_Z = -0.01186
#: pad extent along the eef z axis (m), fingertip side positive.
PAD_Z_RANGE = (-0.03811, 0.01189)
#: distance from the eef origin to the fingertips along the approach axis (m) - OmniGibson ``eef_to_fingertip_lengths``
#: of the R1Pro fingers (0.01880 / 0.01874).
FINGERTIP_LENGTH = 0.0188
#: nominal open width used by the spec's clearance rules (spec §4.6: ``x = (0.10 - pinch width)/2 + 0.01``).
GRIPPER_OPEN_M = 0.10
#: measured maximum opening between the finger pads (m).
GRIPPER_MAX_OPENING_M = 0.0969
#: handle grasps: eef origin = contact - 0.012·a (spec §3.6 / §5.3; the annotated handle contact is the deepest pinch
#: point, 12 mm from the door face, so the fingertip-side end of the pads (+0.012 along a) lands on it).
HANDLE_EEF_OFFSET = 0.012
#: pre-grasp / pre-contact / pre-place distance back along -a (spec §3.6).
PRE_DIST = 0.10


def pad_clearance(pinch_width: float) -> float:
    """``(GRIPPER_OPEN_M - pinch width) / 2``: free room per pad when the open gripper closes on ``pinch_width``
    (spec §4.6 distractor spacing; icl ``pad_clearance_m``)."""
    return 0.5 * (GRIPPER_OPEN_M - float(pinch_width))


def approach_axis(quat) -> th.Tensor:
    """a = eef +z (unit, in the frame of ``quat``): from the wrist toward the fingertips."""
    return quat_to_mat(quat)[:, 2]


def closing_axis(quat) -> th.Tensor:
    """c = eef +y (unit): the pads close along ±c."""
    return quat_to_mat(quat)[:, 1]


def eef_quat(approach, closing) -> th.Tensor:
    """eef orientation from the approach direction ``a`` (→ eef +z) and the closing axis ``c`` (→ eef +y after
    projecting it perpendicular to ``a``); eef +x = c × a. ``a`` and ``c`` must not be parallel."""
    z = unit(approach)
    c = as_tensor(closing)
    y = c - th.dot(c, z) * z
    if float(th.linalg.norm(y)) < 1e-6:
        raise ValueError("closing axis is parallel to the approach")
    y = unit(y)
    x = th.cross(y, z, dim=0)
    return mat_to_quat(th.stack([x, y, z], dim=1))


def pad_center(pose: PoseLike) -> th.Tensor:
    """World point between the pad centres for an eef pose: ``origin + CONTACT_CENTER_Z·a``."""
    pos, quat = as_pose(pose)
    return pos + CONTACT_CENTER_Z * approach_axis(quat)


def fingertip(pose: PoseLike) -> th.Tensor:
    """Fingertip plane centre for an eef pose: ``origin + FINGERTIP_LENGTH·a``."""
    pos, quat = as_pose(pose)
    return pos + FINGERTIP_LENGTH * approach_axis(quat)


# ==================================================================================================================
# 3. Grasp library -> eef poses
# ==================================================================================================================


@lru_cache(maxsize=4096)
def grasp_entry(grasp_key: str, grasp_name: str):
    """The :class:`ispatialgym.sampling.grasp_library.Grasp` named ``grasp_name`` of library key ``grasp_key``
    (validated grasps first, then the unfiltered list, so a row that was later flagged still resolves)."""
    from ispatialgym.sampling import grasp_library as GL

    for clean in (True, False):
        try:
            grasps = GL.load_grasps(grasp_key, require_clean=clean)
        except ValueError:
            continue
        for g in grasps:
            if g.name == grasp_name:
                return g
    raise KeyError(f"grasp {grasp_name!r} not found in library entry {grasp_key!r}")


def rest_quat(grasp_key: str) -> th.Tensor:
    """Orientation the library validated the grasps of ``grasp_key`` in (spec §3.2 rest orientation; identity for the
    canonical +z-up pose)."""
    from ispatialgym.sampling import grasp_library as GL

    rq = GL.rest_quat_xyzw(grasp_key)
    return IDENTITY_QUAT.clone() if rq is None else as_tensor(rq)


def grasp_pose_from_transform(obj_pose: PoseLike, eef_in_object: PoseLike) -> Pose:
    """eef pose = ``obj_pose ∘ T_obj→eef``; frame = that of ``obj_pose``. ``eef_in_object`` is the library entry
    (``Grasp.pose``) or the sidecar's ``eef_in_object``; the inverse is the TRO's ``object_pose_in_eef_frame``."""
    return compose(obj_pose, eef_in_object)


def grasp_pose(obj_pose: PoseLike, grasp_key: str, grasp_name: str, yaw: Optional[float] = None) -> Pose:
    """eef pose (frame of ``obj_pose``) of grasp ``(grasp_key, grasp_name)`` on an object at ``obj_pose``: the spec's
    ``object pose ∘ T_obj→eef`` with ``T_obj→eef`` = the library entry.

    ``obj_pose`` is the object's root-link pose. ``yaw`` selects which orientation the object is taken at:

    * ``None`` - ``obj_pose``'s own orientation (the oracle's settled object pose; the goal pose for ``Place``).
    * a float (rad) - the sampler's draw of spec §4.6: the object stands in the library's rest orientation turned
      by ``yaw`` about world z, ``R_z(yaw) ∘ rest_quat(grasp_key)``; ``obj_pose``'s orientation is **replaced**, its
      position kept. Passing the recorded ``yaw`` together with the settled pose therefore never applies the yaw
      twice - it yields the eef for the nominal (un-drifted) object orientation at the settled position.
    """
    pos, quat = as_pose(obj_pose)
    if yaw is not None:
        quat = qmul(quat_from_yaw(yaw), rest_quat(grasp_key))
    return grasp_pose_from_transform((pos, quat), grasp_entry(grasp_key, grasp_name).pose)


def object_in_eef(grasp_key: str, grasp_name: str) -> Pose:
    """``T_eef→obj`` of a library grasp (the TRO ``grasps[].object_pose_in_eef_frame``): inverse of the entry."""
    return invert(grasp_entry(grasp_key, grasp_name).pose)


# ==================================================================================================================
# 4. Pre-poses and contact poses (spec §3.6)
# ==================================================================================================================


def pre_pose(pose: PoseLike, dist: float = PRE_DIST):
    """Pre-grasp / pre-contact / pre-place: ``pose`` moved ``dist`` back along **-a** (a = eef +z), same orientation.
    Returns the same kind as given (a FramedPose keeps its frame)."""
    pos, quat = as_pose(pose)
    return _like(pose, pos - float(dist) * approach_axis(quat), quat)


def handle_eef(contact, approach=None, closing=None) -> Pose:
    """Handle grasp eef pose (spec §3.6 / §5.3): approach ``a`` along ``approach`` (into the handle), pads closing
    along ``closing`` (the annotation's ``pinch_axis``), **origin = contact - HANDLE_EEF_OFFSET·a** (0.012 m).

    ``contact`` is the annotated pinch point (icl ``require/handles.csv`` / ``artic/handles``): the deepest point the
    pads may reach, 12 mm off the door face for bars, so the fingertip-side end of the pads (``PAD_Z_RANGE[1]``) sits
    on it and the fingertips 7 mm beyond. All three arguments in one frame; the result is in that frame. The closing
    axis is never defaulted (a bar pinched across the wrong axis is a silent failure). One recorded waypoint may be
    passed instead of the three vectors: a sidecar ``articulation.waypoints[k]`` dict (``contact`` / ``approach`` /
    ``pinch``), a :class:`HandleWaypoint`, or a ``(contact, approach, pinch)`` sequence.
    """
    if approach is None and closing is None:
        if isinstance(contact, HandleWaypoint):
            contact, approach, closing = contact.contact, contact.approach, contact.pinch
        elif isinstance(contact, dict):
            contact, approach, closing = contact["contact"], contact["approach"], contact["pinch"]
        else:
            contact, approach, closing = contact
    if approach is None or closing is None:
        raise TypeError("handle_eef needs contact, approach and closing (or one recorded waypoint)")
    q = eef_quat(approach, closing)
    return as_tensor(contact) - HANDLE_EEF_OFFSET * approach_axis(q), q


def default_closing_axis(approach) -> th.Tensor:
    """Closing axis used when a contact rule does not fix one: horizontal and perpendicular to ``a`` (pads vertical),
    i.e. eef +x = world up projected ⊥ a; for a vertical approach the pads close along world y (eef +x = world x)."""
    a = unit(approach)
    up = th.tensor([0.0, 0.0, 1.0]) if abs(float(a[2])) < 0.9 else th.tensor([1.0, 0.0, 0.0])
    x = unit(up - th.dot(up, a) * a)
    return th.cross(a, x, dim=0)


_default_closing = default_closing_axis  # early callers used the private name


def contact_eef(point, approach, closing=None) -> Pose:
    """Push / brace contact eef pose (spec §3.6): **pad centre on ``point``**, approach ``a`` along ``approach`` (into
    the surface: a = -outward normal), pads closing along ``closing`` (spec §5.4 / §5.5 give it per contact kind, e.g.
    ``c = +y_base`` or ``c = n`` (wall normal); default ``default_closing_axis``: horizontal ⊥ a, pads vertical).
    **origin = point - CONTACT_CENTER_Z·a** (the origin is 1.19 cm *beyond* the point, the fingertips 1.88 cm beyond
    the origin)."""
    a = unit(approach)
    q = eef_quat(a, default_closing_axis(a) if closing is None else closing)
    return as_tensor(point) - CONTACT_CENTER_Z * a, q


def fingertip_eef(point, approach, closing=None, clearance: float = 0.005) -> Pose:
    """Stop / brace pose whose **fingertip plane** is ``clearance`` before ``point`` along ``a`` (spec §5.5 brace
    candidates: "fingertip face 5 mm below / above / outside B"): origin = point - (FINGERTIP_LENGTH + clearance)·a."""
    a = unit(approach)
    q = eef_quat(a, default_closing_axis(a) if closing is None else closing)
    return as_tensor(point) - (FINGERTIP_LENGTH + float(clearance)) * a, q


# ==================================================================================================================
# 5. Waypoints (spec §3.9)
# ==================================================================================================================


def _fractions(n: int, endpoints: bool) -> List[float]:
    if int(n) < 1:
        raise ValueError("n must be >= 1")
    fr = [k / float(n) for k in range(1, int(n) + 1)]
    return ([0.0] + fr) if endpoints else fr


def _is_pose(x) -> bool:
    if isinstance(x, FramedPose):
        return True
    if isinstance(x, th.Tensor):
        return False
    return isinstance(x, (tuple, list)) and len(x) == 2 and not isinstance(x[0], (int, float))


def line_waypoints(p0, p1, n: int = 3, endpoints: bool = False):
    """Straight-line waypoints from ``p0`` to ``p1`` at fractions ``k/n`` (the start excluded, the goal last;
    ``endpoints=True`` prepends the start). Positions ``(3,)`` in → ``(n, 3)`` tensor; poses in → list of poses with
    the orientation SLERPed alongside. Used for grasp approaches, lifts, descents, retreats, the held-object camera
    ray (spec §6.3: ``n = 4`` = 3 intermediate + goal) and ``arm_along`` targets."""
    fr = _fractions(n, endpoints)
    if _is_pose(p0):
        (a, qa), (b, qb) = as_pose(p0), as_pose(p1)
        return [_like(p0, a + (b - a) * s, slerp(qa, qb, s)) for s in fr]
    a, b = as_tensor(p0), as_tensor(p1)
    return th.stack([a + (b - a) * s for s in fr])


def slerp_waypoints(q0, q1, n: int = 3, endpoints: bool = False) -> th.Tensor:
    """Orientation waypoints ``(n, 4)`` on the shortest SLERP path from ``q0`` to ``q1`` at fractions ``k/n`` (start
    excluded, goal last; ``endpoints=True`` prepends ``q0``) - the in-hand rotation of mental rotation (spec §6.3)."""
    return th.stack([slerp(q0, q1, s) for s in _fractions(n, endpoints)])


def orbit_waypoints(pose: PoseLike, pivot, delta_quat, n: int = 3, frame: str = "world", endpoints: bool = False) -> List[Pose]:
    """Rigidly rotate ``pose`` about the point ``pivot`` by ``delta_quat`` through ``n`` SLERP steps (fractions
    ``k/n``; the last is the goal). ``frame="world"``: the rotation is applied in the world (``R_k R_0``);
    ``frame="eef"``: about the eef's own axes (``R_0 R_k``). Every waypoint keeps ``|p_k - pivot|`` and the rigid
    attachment pivot↔eef, so with ``pivot`` = the held object's centre the object turns in place (Reorient: mental
    rotation, in-hand yaw). Positions and orientations are returned in the frame ``pose`` was given in."""
    p0, q0 = as_pose(pose)
    piv = as_tensor(pivot)
    dq = as_tensor(delta_quat)
    q1 = qmul(dq, q0) if frame == "world" else qmul(q0, dq)
    if frame not in ("world", "eef"):
        raise ValueError(f"frame must be 'world' or 'eef', got {frame!r}")
    R0t = quat_to_mat(q0).T
    r = p0 - piv
    out = []
    for s in _fractions(n, endpoints):
        qk = slerp(q0, q1, s)
        Rk = quat_to_mat(qk) @ R0t  # incremental rotation from the start, in the parent frame
        out.append(_like(pose, piv + Rk @ r, qk))
    return out


def arc_waypoints(contact, centre, axis, delta: float, n: int = 2):
    """Push-arc waypoints (spec §5.5 / §6.10, §3.9 "δ/2 and the goal"): the contact eef pose rotated rigidly about the
    axis (``centre``, ``axis``) by ``k·delta/n``. A rotation of the pushed body about its hinge / COM carries the
    material contact point *and* the tangent approach with it, so ``a`` stays tangent to the circle and ``c`` (∥ the
    axis for the wall tasks) stays put. ``contact`` = a pose → list of ``n`` poses; a bare point ``(3,)`` → ``(n, 3)``
    positions. ``delta`` in rad, right-hand about ``axis``; ``n = 2`` gives ``[δ/2, goal]``."""
    if _is_pose(contact):
        return orbit_waypoints(contact, centre, quat_from_axis_angle(axis, delta), n=n, frame="world")
    p = as_tensor(contact)
    c = as_tensor(centre)
    return th.stack([c + quat_rotate(quat_from_axis_angle(axis, float(delta) * s), p - c) for s in _fractions(n, False)])


@dataclass
class HandleWaypoint:
    """h(q) at one joint value (spec §5.3): the moved contact point and directions plus the eef poses derived from
    them. Frame = the object frame, or the world when ``handle_waypoints`` got ``obj_pose``."""

    q: float
    contact: th.Tensor  # (3,)
    approach: th.Tensor  # unit, into the handle
    pinch: th.Tensor  # unit closing axis
    eef: Pose  # handle_eef(contact, approach, pinch)
    pre_grasp: Pose  # pre_pose(eef)

    def to_sidecar(self) -> Dict[str, List[float]]:
        return {"contact": self.contact.tolist(), "approach": self.approach.tolist(), "pinch": self.pinch.tolist()}


def _joint_kind(joint_info: Dict[str, Any]) -> str:
    kind = str(joint_info.get("type", joint_info.get("joint_type", ""))).lower()
    if "prismatic" in kind or "slider" in kind:
        return "prismatic"
    if "revolute" in kind or "hinge" in kind or "continuous" in kind:
        return "revolute"
    raise ValueError(f"joint type {kind!r} is neither revolute nor prismatic")


def handle_waypoints(handle: Dict[str, Any], joint_info: Dict[str, Any], q_values: Sequence[float], obj_pose: Optional[PoseLike] = None,
                     pre_dist: float = PRE_DIST) -> List[HandleWaypoint]:
    """The handle trajectory **h(q)** of spec §5.3 at the joint values ``q_values`` (e.g. ``q_init`` and the
    ``q_k = q_init + k (q_goal - q_init)``, ``k ∈ {0.25, 0.5, 0.75, 1}``).

    ``handle`` = the annotation of the **closed** configuration in the object frame (scaled to the instance by the
    caller): ``contact`` (3,), ``pinch_axis`` (3,), ``approach_dir`` (or ``approach_dir_obj``) (3,). ``joint_info``:
    ``type`` (``RevoluteJoint`` / ``PrismaticJoint``), ``axis_dir_obj`` (3,), ``hinge_point_obj`` (3,; revolute) and
    the closed value ``closed_q`` (default ``lower``, default 0). Revolute: contact, pinch and approach are rotated
    about (``hinge_point_obj``, ``axis_dir_obj``) by ``q - closed_q`` (right-hand rule, the joint prim's own sign);
    prismatic: the contact is translated by ``(q - closed_q)·axis_dir_obj``, directions unchanged. Per waypoint the
    eef is ``handle_eef(contact(q), approach(q), pinch(q))`` and the pre-grasp ``pre_pose(eef, pre_dist)``. With
    ``obj_pose`` (world pose of the object root) everything is returned in the world frame.
    """
    contact0 = as_tensor(handle["contact"])
    pinch0 = unit(handle["pinch_axis"])
    approach0 = unit(handle.get("approach_dir", handle.get("approach_dir_obj")))
    kind = _joint_kind(joint_info)
    axis = unit(joint_info["axis_dir_obj"])
    q_closed = float(joint_info.get("closed_q", joint_info.get("lower", 0.0)))
    hinge = as_tensor(joint_info.get("hinge_point_obj", [0.0, 0.0, 0.0]))
    op, oq = as_pose(obj_pose) if obj_pose is not None else identity_pose()
    out: List[HandleWaypoint] = []
    for q in q_values:
        d = float(q) - q_closed
        if kind == "revolute":
            R = quat_to_mat(quat_from_axis_angle(axis, d))
            contact = hinge + R @ (contact0 - hinge)
            approach, pinch = R @ approach0, R @ pinch0
        else:
            contact = contact0 + d * axis
            approach, pinch = approach0.clone(), pinch0.clone()
        cw = op + quat_rotate(oq, contact)
        aw, pw = quat_rotate(oq, approach), quat_rotate(oq, pinch)
        eef = handle_eef(cw, aw, pw)
        out.append(HandleWaypoint(q=float(q), contact=cw, approach=aw, pinch=pw, eef=eef, pre_grasp=pre_pose(eef, pre_dist)))
    return out


def flip_waypoints(eef_pose: PoseLike, pivot, yaw: float, flip_axis, flip_sign: int = 1, *, n_yaw: int = 2, n_flip: int = 3,
                   flip_deg: float = 90.0) -> List[Pose]:
    """In-hand reorientation of a held block (spec §6.5, Reorient): first the smallest yaw ``yaw`` (rad, about world z
    through ``pivot`` = the block centre; skipped when 0), then **one flip of ``flip_sign·flip_deg``** (right-hand)
    about the horizontal table axis ``flip_axis`` through the pivot - the axis the pinch is parallel to, so the pads
    stay flat. Both stages are ``orbit_waypoints`` of the eef about the pivot (the block turns in place); the list is
    the yaw stage (``n_yaw`` poses) followed by the flip stage (``n_flip`` poses, the last one is the reoriented eef
    pose from which ``Place`` proceeds to the slot). Argument order follows the sidecar block: ``yaw_deg`` (as rad),
    ``flip_axis``, ``flip_sign``."""
    out: List[Pose] = []
    cur = as_pose(eef_pose)
    if abs(float(yaw)) > 1e-9:
        out.extend(orbit_waypoints(cur, pivot, quat_from_yaw(yaw), n=n_yaw))
        cur = out[-1]
    a = unit(flip_axis)
    if abs(float(a[2])) > math.sin(math.radians(15.0)):
        raise ValueError("flip_axis must be a horizontal table axis")
    out.extend(orbit_waypoints(cur, pivot, quat_from_axis_angle(a, math.radians(float(flip_sign) * float(flip_deg))), n=n_flip))
    return out


# ==================================================================================================================
# 6. Stances, transitions (spec §3.8) and base follow (spec §5.3)
# ==================================================================================================================


def _stance_fields(stance) -> Tuple[Pose2D, float, float, Optional[float]]:
    """``(base_pose2d, translate, tilt, stand_off)`` from a sidecar ``stances.<name>`` dict or any object with those
    attributes (``sampling.v2.stances.Stance``)."""
    if isinstance(stance, dict):
        get = stance.get
    else:
        get = lambda k, d=None: getattr(stance, k, d)  # noqa: E731
    bp = get("base_pose2d")
    if bp is None:
        raise ValueError("stance has no base_pose2d")
    vals = bp.reshape(-1).tolist() if isinstance(bp, th.Tensor) else [float(v) for v in bp]  # keep double precision
    x, y, yaw = (float(v) for v in vals[:3])
    so = get("stand_off")
    return (x, y, yaw), float(get("translate", M.DEFAULT_TRUNK_TRANSLATE)), float(get("tilt", 0.0)), (None if so is None else float(so))


@dataclass
class TransitionPoses:
    """The three legs of a stance transition (spec §3.8, ``Transition`` action): drive back from ``start`` to
    ``drive_back`` (the arriving stance backed off to stand-off ``stand_off`` = s_T), move the trunk from
    ``trunk_from`` to ``trunk``, drive in to ``drive_in`` (the arriving stance)."""

    start: Pose2D
    drive_back: Pose2D
    trunk_from: Tuple[float, float]
    trunk: Tuple[float, float]
    drive_in: Pose2D
    stand_off: float

    @property
    def order(self) -> Tuple[str, str, str]:
        return ("drive-back", "trunk", "drive-in")

    def __iter__(self):
        """Unpacks as the three legs: ``drive_back, trunk, drive_in = transition_poses(...)``."""
        yield self.drive_back
        yield self.trunk
        yield self.drive_in


def transition_poses(stance_from, stance_to, s_T: float) -> TransitionPoses:
    """Poses of the recorded transition between two consecutive stances (spec §3.8).

    The trunk change happens at the **transition stand-off** ``s_T`` on the arriving stance's own axis: the
    ``drive_back`` pose is ``stance_to`` translated ``s_T - stance_to.stand_off`` backwards along its forward direction
    (-x of the base, i.e. away from the edge / face along +n), keeping the yaw; the trunk pair is the arriving
    stance's ``(translate, tilt)``; ``drive_in`` is ``stance_to`` itself. ``stance_from`` supplies the start pose
    and the trunk the sweep starts from. Stances are sidecar dicts or objects with ``base_pose2d``, ``translate``,
    ``tilt``, ``stand_off``. The swept trunk samples the sampler checks at ``drive_back`` are ``trunk_sweep``.
    """
    p_from, t0, k0, _ = _stance_fields(stance_from)
    (x, y, yaw), t1, k1, s = _stance_fields(stance_to)
    if s is None:
        raise ValueError("the arriving stance needs a stand_off")
    back = float(s_T) - s
    if back < -1e-6:
        raise ValueError(f"transition stand-off {s_T} is smaller than the stance stand-off {s}")
    drive_back = (x - back * math.cos(yaw), y - back * math.sin(yaw), yaw)
    return TransitionPoses(start=p_from, drive_back=drive_back, trunk_from=(t0, k0), trunk=(t1, k1), drive_in=(x, y, yaw), stand_off=float(s_T))


def departure_transition_poses(stance_from, s_T: float, trunk_to: Tuple[float, float]) -> TransitionPoses:
    """Poses of a **departure** transition (owner 2026‑09‑19 21:56, IORS large-scale carry: "make torso straight, then
    navigate"; the mirror image of :func:`transition_poses`). The trunk change happens on the *departing* stance's own
    axis: ``drive_back`` is ``stance_from`` translated ``s_T - stance_from.stand_off`` backwards along its forward
    direction (away from the edge / the floor target, i.e. along the way the robot arrived), the trunk pair is the
    **travel trunk** ``trunk_to`` (not a stance's), and there is no drive-in: ``drive_in == drive_back`` — the recorded
    path that follows starts there. ``start`` is the departing stance itself; the swept samples are ``trunk_sweep``."""
    (x, y, yaw), t0, k0, s = _stance_fields(stance_from)
    s = 0.0 if s is None else float(s)
    back = float(s_T) - s
    if back < -1e-6:
        raise ValueError(f"departure transition stand-off {s_T} is smaller than the stance stand-off {s}")
    drive_back = (x - back * math.cos(yaw), y - back * math.sin(yaw), yaw)
    t1, k1 = float(trunk_to[0]), float(trunk_to[1])
    return TransitionPoses(start=(x, y, yaw), drive_back=drive_back, trunk_from=(t0, k0), trunk=(t1, k1), drive_in=drive_back, stand_off=float(s_T))


def transition_standoffs(stand_off: float, step: float = 0.1, n: int = 6) -> List[float]:
    """The candidate transition stand-offs of spec §3.8, smallest first: ``{s, s + 0.1, …, s + 0.5}``."""
    return [round(float(stand_off) + k * step, 6) for k in range(int(n))]


def trunk_sweep(trunk_from: Tuple[float, float], trunk_to: Tuple[float, float], n: int = 5, endpoints: bool = False) -> List[Tuple[float, float]]:
    """``n`` (translate, tilt) samples of a straight trunk motion at fractions ``k/n`` (start excluded, target last;
    spec §3.8 "5 samples between the two trunk values"). The motion is interpolated **jointly** - both scalars reach
    their targets together - which is what ``trunk_to`` must execute for the sweep to be the motion that was checked."""
    (t0, k0), (t1, k1) = (float(trunk_from[0]), float(trunk_from[1])), (float(trunk_to[0]), float(trunk_to[1]))
    return [(t0 + (t1 - t0) * s, k0 + (k1 - k0) * s) for s in _fractions(n, endpoints)]


def eef_base_transform(eef_pose: PoseLike, base_pose: PoseLike) -> Pose:
    """``T_eef→base``: the base_link pose expressed in the eef frame (both poses in the world) - constant while the
    arm is locked at its IK solution q* (spec §5.3)."""
    return relative(base_pose, eef_pose)


def base_follow_poses(h_poses: Sequence[PoseLike], T_eef_base: PoseLike) -> List[Pose2D]:
    """Base-follow targets ``b_k = h(q_k) ∘ T_eef→base(q*)`` (spec §5.3): with the handle grasped and the arm locked,
    the base_link pose that keeps the eef on the handle at every recorded handle pose ``h(q_k)`` (world eef poses,
    e.g. ``[w.eef for w in handle_waypoints(...)]``), projected to the planar ``(x, y, yaw)`` the holonomic base can
    realise. Valid for vertical-axis revolute and horizontal prismatic joints (``follow_mode: base``); check
    ``base_follow_residual`` before trusting the projection."""
    return [pose_to_pose2d(compose(h, T_eef_base)) for h in h_poses]


def base_follow_residual(h_poses: Sequence[PoseLike], T_eef_base: PoseLike, base_z: float = None) -> float:
    """Largest non-planar part of the base-follow poses: ``max_k |z_k - base_z| + |roll_k| + |pitch_k|`` (m + rad).
    ``base_z`` defaults to the z of the first pose, so the residual measures the *change* of height along the way."""
    worst = 0.0
    z_ref = base_z
    for h in h_poses:
        pos, quat = compose(h, T_eef_base)
        roll, pitch, _ = euler_of(quat)
        if z_ref is None:
            z_ref = float(pos[2])
        worst = max(worst, abs(float(pos[2]) - float(z_ref)) + abs(roll) + abs(pitch))
    return worst


# ==================================================================================================================
# 7. JoyLo trunk keyframes (moved from oracle/trunk.py)
# ==================================================================================================================

_UPRIGHT = th.tensor(M.TORSO_UPRIGHT, dtype=th.float32)
_DOWNWARD = th.tensor(M.TORSO_DOWNWARD, dtype=th.float32)
_GROUND = th.tensor(M.TORSO_GROUND, dtype=th.float32)

#: URDF limits of ``torso_joint1..4`` (lower, upper), for clamping the tilt without a simulator.
TORSO_JOINT_LIMITS = (
    th.tensor([-1.1345, -2.7925, -1.8326, -3.0543], dtype=th.float32),
    th.tensor([1.8326, 2.5307, 1.5708, 3.0543], dtype=th.float32),
)


def translate_to_torso_q(translate: float) -> th.Tensor:
    """Torso joint positions ``(4,)`` on the JoyLo keyframe curve for ``translate`` in [0, 2] (tilt 0): piecewise
    linear ``upright (0) → downward (1) → ground (2)`` (``r1pro_model.TORSO_*``); clamped to the range."""
    lo, hi = M.TRUNK_TRANSLATE_RANGE
    t = min(max(float(translate), lo), hi)
    if t <= 1.0:
        return (1.0 - t) * _UPRIGHT + t * _DOWNWARD
    s = t - 1.0
    return (1.0 - s) * _DOWNWARD + s * _GROUND


def torso_q_to_translate_tilt(q) -> Tuple[float, float]:
    """Inverse of the keyframe curve: ``translate`` from ``torso_joint1`` (monotonic along the curve), ``tilt`` = the
    residual of ``torso_joint3`` with respect to the curve value."""
    q = as_tensor(q)
    j1 = float(q[0])
    if j1 > float(_DOWNWARD[0]):
        translate = 1.0 + (j1 - float(_DOWNWARD[0])) / (float(_GROUND[0]) - float(_DOWNWARD[0]))
    else:
        translate = (j1 - float(_UPRIGHT[0])) / (float(_DOWNWARD[0]) - float(_UPRIGHT[0]))
    translate = min(max(translate, M.TRUNK_TRANSLATE_RANGE[0]), M.TRUNK_TRANSLATE_RANGE[1])
    tilt = float(q[M.TRUNK_TILT_JOINT_INDEX] - translate_to_torso_q(translate)[M.TRUNK_TILT_JOINT_INDEX])
    return translate, tilt


def torso_q(translate: float, tilt: float, joint_limits: Optional[Tuple[th.Tensor, th.Tensor]] = None) -> th.Tensor:
    """Torso joint positions ``(4,)`` for the JoyLo pair: the keyframe curve at ``translate`` plus ``tilt`` added to
    ``torso_joint3`` (index ``TRUNK_TILT_JOINT_INDEX``), clamped to ``joint_limits`` when given (e.g.
    ``TORSO_JOINT_LIMITS``)."""
    q = translate_to_torso_q(translate).clone()
    j = M.TRUNK_TILT_JOINT_INDEX
    val = float(q[j]) + float(tilt)
    if joint_limits is not None:
        val = min(max(val, float(joint_limits[0][j])), float(joint_limits[1][j]))
    q[j] = val
    return q


# ==================================================================================================================
# 8. Head camera and torso forward kinematics (URDF r1pro.urdf / USD r1pro.usda)
# ==================================================================================================================

#: The torso chain of ``r1pro.urdf`` from ``base_link``: (joint origin xyz, joint axis) - all origins have rpy 0.
_TORSO_CHAIN = (
    ((-0.079032, 0.0, 0.34265), (0.0, 1.0, 0.0)),  # torso_joint1 -> torso_link1
    ((0.0, 0.0, 0.4), (0.0, 1.0, 0.0)),  # torso_joint2 -> torso_link2
    ((0.0, 0.0001005, 0.3), (0.0, -1.0, 0.0)),  # torso_joint3 -> torso_link3
    ((0.0, -0.00010015, 0.09962), (0.0, 0.0, 1.0)),  # torso_joint4 -> torso_link4
)
#: fixed ``zed_joint``: torso_link4 -> zed_link (xyz, rpy).
_ZED_IN_TORSO4 = ((0.065511, 0.060003, 0.47598), (-1.9199, 0.0, -1.5708))
#: fixed ``<arm>_arm_base_joint`` + ``<arm>_arm_joint1`` origin: torso_link4 -> ``<arm>_arm_link1`` (the shoulder).
_SHOULDER_IN_TORSO4 = {"left": (-0.00048618, 0.097234 + 0.0735, 0.30302), "right": (-0.00048706, -0.097236 - 0.0735, 0.30302)}
#: head camera sensor prim ``zed_link/Camera`` in ``zed_link`` (USD: translate (0.06, 0, 0.01), orient wxyz (0, -1, 0, 0)
#: = 180° about x). Camera convention: looks along its -z, +y up.
CAMERA_IN_ZED: Pose = (th.tensor([0.06, 0.0, 0.01]), th.tensor([1.0, 0.0, 0.0, 0.0]))
#: base_link height above the floor when R1Pro stands on its wheels (measured on the flat floor plane).
NOMINAL_BASE_Z = 0.0053


def camera_offset_in_zed(robot=None, sensor_key: Optional[str] = None) -> Pose:
    """Pose of the head camera sensor frame in ``zed_link``. Without a robot: the USD constant ``CAMERA_IN_ZED``.
    With an OmniGibson robot (duck-typed: ``robot.sensors[key].get_position_orientation()`` and
    ``robot.links["zed_link"]``): measured from the simulator, as v1 did - use it to verify the constant."""
    if robot is None:
        return CAMERA_IN_ZED[0].clone(), CAMERA_IN_ZED[1].clone()
    key = sensor_key or next(k for k in robot.sensors if k.endswith(M.HEAD_CAMERA_SENSOR_SUFFIX))
    cp, cq = robot.sensors[key].get_position_orientation()
    zp, zq = robot.links[M.HEAD_CAMERA_LINK].get_position_orientation()
    return relative((cp.float(), cq.float()), (zp.float(), zq.float()))


def _hmat(xyz, quat) -> th.Tensor:
    mat = th.eye(4, dtype=th.float64)
    mat[:3, :3] = quat_to_mat(quat).to(th.float64)
    mat[:3, 3] = th.tensor(xyz, dtype=th.float64)
    return mat


def torso_fk(torso_q_) -> Dict[str, Pose]:
    """Poses in the **base_link frame** of the torso chain for the 4 torso joint values (OmniGibson order
    ``torso_joint1..4``): ``torso_link1..4``, ``zed_link``, ``camera`` (the sensor frame, ``CAMERA_IN_ZED``),
    ``left_shoulder`` / ``right_shoulder`` (``<arm>_arm_link1`` origins). Pure URDF kinematics; the arms are not
    included (their FK is cuRobo's)."""
    q = as_tensor(torso_q_, th.float64).reshape(-1)
    if q.numel() != 4:
        raise ValueError("torso_fk expects the 4 torso joint values")
    out: Dict[str, Pose] = {}
    mat = th.eye(4, dtype=th.float64)
    for i, ((xyz, axis), qi) in enumerate(zip(_TORSO_CHAIN, q.tolist())):
        mat = mat @ _hmat(xyz, quat_from_axis_angle(axis, qi))
        out[f"torso_link{i + 1}"] = mat_to_pose(mat.to(th.float32))
    zed = mat @ _hmat(_ZED_IN_TORSO4[0], quat_from_euler(*_ZED_IN_TORSO4[1]))
    out["zed_link"] = mat_to_pose(zed.to(th.float32))
    out["camera"] = mat_to_pose((zed @ _hmat(CAMERA_IN_ZED[0].tolist(), CAMERA_IN_ZED[1])).to(th.float32))
    for arm, xyz in _SHOULDER_IN_TORSO4.items():
        out[f"{arm}_shoulder"] = mat_to_pose((mat @ _hmat(xyz, IDENTITY_QUAT)).to(th.float32))
    return out


def _base_pose_from_virtual_joints(q28: th.Tensor) -> Pose:
    """World pose of base_link encoded by the six virtual base joints (root frame at the world origin)."""
    j = JOINT_INDEX
    pos = th.stack([q28[j["base_footprint_x_joint"]], q28[j["base_footprint_y_joint"]], q28[j["base_footprint_z_joint"]]])
    rx, ry, rz = float(q28[j["base_footprint_rx_joint"]]), float(q28[j["base_footprint_ry_joint"]]), float(q28[j["base_footprint_rz_joint"]])
    # chain x -> y -> z -> rx -> ry -> rz: rotation = R_x(rx) R_y(ry) R_z(rz)
    quat = qmul(qmul(quat_from_axis_angle([1.0, 0.0, 0.0], rx), quat_from_axis_angle([0.0, 1.0, 0.0], ry)), quat_from_yaw(rz))
    return pos.to(th.float32), quat


def _base_pose_arg(base_pose) -> Pose:
    if isinstance(base_pose, FramedPose):
        return as_pose(base_pose)
    if isinstance(base_pose, (tuple, list)) and len(base_pose) == 3 and isinstance(base_pose[0], (int, float)):
        return pose2d_to_pose(base_pose[0], base_pose[1], base_pose[2], z=NOMINAL_BASE_Z)
    if isinstance(base_pose, th.Tensor) and base_pose.numel() == 3:
        x, y, yaw = base_pose.reshape(-1).tolist()
        return pose2d_to_pose(x, y, yaw, z=NOMINAL_BASE_Z)
    return as_pose(base_pose)


def camera_pose(q, base_pose=None, cam_in_zed: Optional[PoseLike] = None) -> Pose:
    """World pose of the head camera sensor frame for a robot configuration.

    ``q``: the 28-D OmniGibson joint vector **or** the 4 torso joint values (``torso_q(translate, tilt)``).
    ``base_pose``: world pose of ``base_link`` - a ``Pose``, a ``FramedPose`` or a planar ``(x, y, yaw)`` (then
    ``z = NOMINAL_BASE_Z``, floor at z = 0). ``None`` with a 28-D ``q`` reads the base from the virtual base joints
    (root frame = world, as ``q_with_base_pose`` builds them); ``None`` with a 4-D ``q`` returns the pose in the base
    frame. ``cam_in_zed`` defaults to ``CAMERA_IN_ZED``. Used by the sampler's view solve (spec §4.3 / §5.2), the
    camera-relocalization goal (``SetTorso``) and the gripper-in-FOV monitor.
    """
    qt = as_tensor(q).reshape(-1)
    if qt.numel() == 28:
        torso = qt[[JOINT_INDEX[n] for n in M.TORSO_JOINTS]]
        base = _base_pose_from_virtual_joints(qt) if base_pose is None else _base_pose_arg(base_pose)
    elif qt.numel() == 4:
        torso = qt
        base = identity_pose() if base_pose is None else _base_pose_arg(base_pose)
    else:
        raise ValueError("camera_pose expects a 28-D joint vector or the 4 torso joint values")
    zed = torso_fk(torso)["zed_link"]
    cam = compose(zed, CAMERA_IN_ZED if cam_in_zed is None else cam_in_zed)
    return compose(base, cam)


def shoulder_position(translate: float, tilt: float = 0.0, arm: str = "right") -> th.Tensor:
    """``<arm>_arm_link1`` origin in the base frame for a JoyLo trunk pair (the "shoulder (x, z)" column of oracle doc
    §5.2; add ``NOMINAL_BASE_Z`` for the height above the floor)."""
    return torso_fk(torso_q(translate, tilt))[f"{arm}_shoulder"][0]


# ==================================================================================================================
# 9. Configurations: virtual base joints (spec §3.3 stance checks; track_path's collision monitor)
# ==================================================================================================================

#: index of every joint in the 28-D OmniGibson vector (== the cuRobo backend's ``_jidx``).
JOINT_INDEX: Dict[str, int] = M.joint_index_map(M.ALL_JOINTS)


def q_with_base_pose(q, x: float, y: float, yaw: float, root_pose2d: Pose2D = (0.0, 0.0, 0.0), base_z: Optional[float] = None,
                     root_z: float = 0.0, joint_index: Optional[Dict[str, int]] = None) -> th.Tensor:
    """Copy of the 28-D configuration ``q`` whose **virtual base joints** put ``base_link`` at world ``(x, y, yaw)``.

    OmniGibson keeps the articulation root (``base_footprint_x``) where the robot was spawned and the six virtual
    joints hold the offset root → base_link, expressed in the **root frame**; ``root_pose2d`` is the root's world
    ``(x, y, yaw)`` (``backend.root_pose()``; the default identity makes the joints the world pose itself). Sets
    ``base_footprint_{x, y, rz}_joint`` (yaw wrapped). With ``base_z`` (world height of base_link, e.g.
    ``floor_z + NOMINAL_BASE_Z``) it also sets ``base_footprint_z_joint = base_z - root_z`` and zeroes roll / pitch so
    the hypothetical robot stands upright at the nominal height whatever ``q`` carried (``RobotKit.q_at_base_pose``);
    otherwise z / roll / pitch are left as in ``q`` (``Skills.q_with_base_pose``).
    """
    j = JOINT_INDEX if joint_index is None else joint_index
    rx, ry, ryaw = (float(v) for v in root_pose2d)
    c, s = math.cos(ryaw), math.sin(ryaw)
    dx, dy = float(x) - rx, float(y) - ry
    qk = as_tensor(q).clone()
    qk[j["base_footprint_x_joint"]] = c * dx + s * dy
    qk[j["base_footprint_y_joint"]] = -s * dx + c * dy
    qk[j["base_footprint_rz_joint"]] = wrap_angle(float(yaw) - ryaw)
    if base_z is not None:
        qk[j["base_footprint_z_joint"]] = float(base_z) - float(root_z)
        qk[j["base_footprint_rx_joint"]] = 0.0
        qk[j["base_footprint_ry_joint"]] = 0.0
    return qk


def base_pose2d_of_q(q, root_pose2d: Pose2D = (0.0, 0.0, 0.0), joint_index: Optional[Dict[str, int]] = None) -> Pose2D:
    """Inverse of :func:`q_with_base_pose`: the world ``(x, y, yaw)`` of base_link encoded in ``q``."""
    j = JOINT_INDEX if joint_index is None else joint_index
    qt = as_tensor(q)
    rx, ry, ryaw = (float(v) for v in root_pose2d)
    c, s = math.cos(ryaw), math.sin(ryaw)
    bx, by = float(qt[j["base_footprint_x_joint"]]), float(qt[j["base_footprint_y_joint"]])
    return rx + c * bx - s * by, ry + s * bx + c * by, wrap_angle(float(qt[j["base_footprint_rz_joint"]]) + ryaw)


# ==================================================================================================================
# 10. Block design: cube grasps (spec §6.5)
# ==================================================================================================================

#: edge of the block-design cube ``toy_dice/isgblk`` (m); pitch of the goal grid = edge + 1 mm.
BLOCK_EDGE = 0.025
_FACE_AXIS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


@dataclass
class CubeGrasp:
    """One cube grasp (spec §6.5): ``eef`` = the eef pose (world), ``approach`` / ``closing`` its axes, ``pinch_point``
    = the material point of the block between the pad centres (world) and ``pinch_point_block`` the same in the
    block frame, ``eef_in_block`` = the rigid in-hand transform ``T_block→eef`` (``compose(block_pose', eef_in_block)``
    is the eef pose after the block moved - the place pose at the goal block pose), ``tilt_deg`` the tilt about the
    closing axis, ``yaw`` the (snapped) wrist yaw, ``face_pair`` the pinched face pair."""

    eef: Pose
    approach: th.Tensor
    closing: th.Tensor
    pinch_point: th.Tensor
    pinch_point_block: th.Tensor
    eef_in_block: Pose
    tilt_deg: float
    yaw: float
    face_pair: str

    @property
    def downward(self) -> bool:
        """Whether the approach points down (the only feasible pick approaches)."""
        return float(self.approach[2]) < 0.0


def cube_pick_tilt(tilt_place_deg: float, flip_sign: int) -> float:
    """Pick tilt paired with a place tilt for a block that flips by ``flip_sign·90°`` about the pinch axis
    (``d_pick = R_a(∓90°)·d_place``): ``tilt_place - flip_sign·90``. Exactly one sign gives a downward pick approach
    for a place tilt in (0°, 90°); ``cube_grasp(...).downward`` tells which."""
    return float(tilt_place_deg) - 90.0 * float(flip_sign)


def cube_grasp(block_pose: PoseLike, face_pair: str, tilt_pick_deg: float, yaw: float, edge: float = BLOCK_EDGE,
               tip_clearance: float = 0.003) -> CubeGrasp:
    """Pinch grasp of a cube (spec §6.5 cube grasp set: pinch across one face pair × approach through a perpendicular
    face × tilt about the pinch axis).

    * ``block_pose``: world pose of the block (centre, orientation). ``face_pair`` ∈ {"x", "y", "z"}: the block-frame
      axis the pads close along; it must be horizontal in the world (the flip axis of spec §6.5 is a horizontal
      table axis and the pinch is parallel to it).
    * ``yaw`` (rad): wrist yaw of the tilt-0 top-down pinch = world yaw of the eef +x axis. Only two values are
      admissible for a given face pair (eef +x ⊥ the pinch axis); ``yaw`` is **snapped to the nearer one**, which
      fixes the sign of the closing axis ``c`` and therefore the sign convention of the tilt.
    * ``tilt_pick_deg``: right-hand rotation of the approach about ``c`` from straight down (``a = R_c(tilt)·(-ẑ)``,
      positive toward eef +x); the pads stay flat on the face pair for any tilt (spec §6.5: {0, ±30, ±45, ±60}; the
      pick of a flipping block uses ``cube_pick_tilt``).
    * Pinch depth: the pad centre lies on the approach line through the block centre, pulled back toward the
      gripper by the smallest offset ``d ≥ 0`` that keeps the **fingertips ``tip_clearance`` above the block's
      bottom plane** (the support) - for a 2.5 cm cube at 45° that is the top corner region of the face; for large
      blocks ``d = 0`` (pad centre at the block centre). The eef origin then follows ``contact_eef``.

    The in-hand transform is rigid, so the place eef pose of the same grasp is ``compose(goal_block_pose,
    result.eef_in_block)``; flipping the block by ``+90°`` about ``c`` turns a ``θ - 90°`` pick approach into the
    ``+θ`` place approach and only one flip sign yields a downward pick (``CubeGrasp.downward``).
    """
    p_b, q_b = as_pose(block_pose)
    if face_pair not in _FACE_AXIS:
        raise ValueError(f"face_pair must be 'x', 'y' or 'z', got {face_pair!r}")
    c_b = th.tensor(_FACE_AXIS[face_pair], dtype=th.float32)
    c_w = quat_rotate(q_b, c_b)
    if abs(float(c_w[2])) > math.sin(math.radians(15.0)):
        raise ValueError(f"face pair {face_pair!r} is not horizontal in the world (its axis has z = {float(c_w[2]):.3f})")
    # admissible wrist yaws: eef +x ⊥ c_w (horizontal), two of them 180° apart; snap the requested yaw to the nearer
    yaw_c = math.atan2(float(c_w[1]), float(c_w[0]))
    cands = [wrap_angle(yaw_c + 0.5 * math.pi), wrap_angle(yaw_c - 0.5 * math.pi)]
    yaw_s = min(cands, key=lambda a: abs(wrap_angle(a - float(yaw))))
    x0 = th.tensor([math.cos(yaw_s), math.sin(yaw_s), 0.0])
    z0 = th.tensor([0.0, 0.0, -1.0])
    y0 = th.cross(z0, x0, dim=0)  # = ±c_w
    q0 = mat_to_quat(th.stack([x0, y0, z0], dim=1))
    q_eef = qmul(q0, quat_from_axis_angle([0.0, 1.0, 0.0], math.radians(float(tilt_pick_deg))))  # tilt about the eef y axis
    a = approach_axis(q_eef)
    c = closing_axis(q_eef)
    # pinch depth: fingertips (pad centre + L·a) at least tip_clearance above the bottom plane
    L = FINGERTIP_LENGTH - CONTACT_CENTER_Z
    az = float(a[2])
    d = 0.0
    if az < -1e-6:
        d = max(0.0, L - (0.5 * float(edge) - float(tip_clearance)) / (-az))
    pinch = p_b - d * a
    eef = contact_eef(pinch, a, c)
    return CubeGrasp(
        eef=eef,
        approach=a,
        closing=c,
        pinch_point=pinch,
        pinch_point_block=quat_rotate(qinv(q_b), pinch - p_b),
        eef_in_block=relative(eef, (p_b, q_b)),
        tilt_deg=float(tilt_pick_deg),
        yaw=yaw_s,
        face_pair=face_pair,
    )
