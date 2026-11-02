"""Base-frame Cartesian dimensions the L3 agent names one at a time (RoboProbe's Inspect EEF surface on R1Pro).

Positions are the grasp point, the centre of the finger pads, in the robot base frame (x forward, y left, z up, origin on
the floor at the centre of the base). Orientation is three angles in degrees measured from :data:`REFERENCE`, a level
grasp pointing straight ahead with the jaws closing along the base y axis: extrinsic rotations about the base axes applied
to the reference, pitch about +y first, then roll about +x, then yaw about +z. The triple is singular at roll = +-90 deg,
the jaws closing vertically, which R1Pro rarely uses: over the oracle demos of the counted eval instances 1-4 % of the
gripper orientations come within 15 deg of it (and 1-7 % within 15 deg of a +-180 deg wrap), against 28 % within 15 deg of
the singularity of a raw base-frame roll/pitch/yaw of the eef link, which sits at level side grasps. Both arms use the one
reference: on both, the closing axis points to base +y in 84-97 % of the forward approaches.

Quaternions are xyzw (OmniGibson). The ``_deg`` suffix is part of each angle's name so the unit cannot be misread.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np

from ispatialgym.robot.conventions import CONTACT_CENTER_Z, FINGERTIP_LENGTH, GRIPPER_MAX_OPENING_M, PAD_Z_RANGE

ARMS: Tuple[str, ...] = ("left", "right")
POSITION_AXES: Tuple[str, ...] = ("x", "y", "z")
ANGLE_AXES: Tuple[str, ...] = ("pitch_deg", "roll_deg", "yaw_deg")

#: Holding the middle angle (roll) in [-90, 90] covers every orientation exactly once.
ANGLE_BOUNDS: Dict[str, Tuple[float, float]] = {"pitch_deg": (-180.0, 180.0), "roll_deg": (-90.0, 90.0), "yaw_deg": (-180.0, 180.0)}
GRIPPER_BOUNDS = (0.0, 1.0)

#: Every grasp point of the counted eval demos (x 0.09..0.97 m, left y -0.12..0.61, right y -0.73..0.09, z 0.05..1.72)
#: lies at least 0.25 m inside these boxes, so they only reject gross magnitude errors.
POSITION_BOUNDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "left": {"x": (-0.3, 1.3), "y": (-0.5, 1.0), "z": (-0.2, 2.0)},
    "right": {"x": (-0.3, 1.3), "y": (-1.0, 0.5), "z": (-0.2, 2.0)},
}

#: Grasp point in the eef link frame: the pad contact centre on the approach axis (eef +z, wrist -> fingertips).
GRASP_POINT_IN_EEF = np.array([0.0, 0.0, CONTACT_CENTER_Z])
PAD_LENGTH_M = PAD_Z_RANGE[1] - PAD_Z_RANGE[0]
FINGERTIPS_BEYOND_GRASP_M = FINGERTIP_LENGTH - CONTACT_CENTER_Z
MAX_JAW_OPENING_M = GRIPPER_MAX_OPENING_M

#: Base-from-eef rotation of the zero orientation: approach (eef +z) along base +x, closing axis (eef +y) along base +y.
REFERENCE = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

_GIMBAL_EPS = 1e-7


def arm_labels(arm: str) -> Tuple[str, ...]:
    return (*(f"{arm}_{a}" for a in POSITION_AXES), *(f"{arm}_{a}" for a in ANGLE_AXES), f"{arm}_gripper")


def labels() -> Tuple[str, ...]:
    """All 14 dimensions, left arm first."""
    return tuple(label for arm in ARMS for label in arm_labels(arm))


def split_label(label: str) -> Tuple[str, str]:
    arm, _, axis = label.partition("_")
    return arm, axis


def bounds(label: str, negate_xyz: bool = False) -> Tuple[float, float]:
    """Advertised ``[low, high]`` of one dimension (mirrored for a negated position frame)."""
    arm, axis = split_label(label)
    if axis == "gripper":
        return GRIPPER_BOUNDS
    if axis in ANGLE_BOUNDS:
        return ANGLE_BOUNDS[axis]
    lo, hi = POSITION_BOUNDS[arm][axis]
    return (-hi, -lo) if negate_xyz else (lo, hi)


def bounds_text(negate_xyz: bool = False) -> str:
    return ", ".join(f"{label}: [{lo:.4g}, {hi:.4g}]" for label, (lo, hi) in ((lb, bounds(lb, negate_xyz)) for lb in labels()))


# -- rotations ------------------------------------------------------------------------------------------------------

def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError("a quaternion needs four finite numbers with a non-zero norm")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def angles_to_matrix(pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """Base-from-eef rotation of one angle triple."""
    rel = _rz(math.radians(yaw_deg)) @ _rx(math.radians(roll_deg)) @ _ry(math.radians(pitch_deg))
    return rel @ REFERENCE


def matrix_to_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """``(pitch_deg, roll_deg, yaw_deg)`` of a base-from-eef rotation."""
    m = np.asarray(R, dtype=float) @ REFERENCE.T
    roll = math.asin(float(np.clip(m[2, 1], -1.0, 1.0)))
    if abs(math.cos(roll)) > _GIMBAL_EPS:
        pitch = math.atan2(-float(m[2, 0]), float(m[2, 2]))
        yaw = math.atan2(-float(m[0, 1]), float(m[1, 1]))
    else:
        # roll at +-90 couples pitch and yaw; pin yaw and put the whole remaining rotation on pitch
        yaw = 0.0
        pitch = math.atan2(float(m[0, 2]), float(m[0, 0]))
    # adding zero folds -0.0 into 0.0 so the reference renders as plain zeros
    return math.degrees(pitch) + 0.0, math.degrees(roll) + 0.0, math.degrees(yaw) + 0.0


def slerp(q0: Sequence[float], q1: Sequence[float], t: float) -> np.ndarray:
    a, b = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    th0 = math.acos(d)
    return (math.sin((1.0 - t) * th0) * a + math.sin(t * th0) * b) / math.sin(th0)


def orientation_error_deg(q1: Sequence[float], q2: Sequence[float]) -> float:
    a, b = np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)
    d = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


# -- eef pose <-> named values --------------------------------------------------------------------------------------

def pose_to_values(pos: Sequence[float], quat: Sequence[float]) -> Dict[str, float]:
    """Eef link pose (base frame) -> grasp-point position and angles. With :func:`values_to_pose` the only place the
    grasp-point offset is applied, so observation, targets and arrival check all speak one frame."""
    R = quat_to_matrix(quat)
    p = np.asarray(pos, dtype=float).reshape(3) + R @ GRASP_POINT_IN_EEF
    pitch, roll, yaw = matrix_to_angles(R)
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "pitch_deg": pitch, "roll_deg": roll, "yaw_deg": yaw}


def values_to_pose(values: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Named values -> eef link pose (base frame). The offset comes off along the *target* orientation's approach axis,
    so a rotation keeps the grasp point where it is and the wrist swings around it."""
    R = angles_to_matrix(values["pitch_deg"], values["roll_deg"], values["yaw_deg"])
    grasp = np.array([values["x"], values["y"], values["z"]], dtype=float)
    return grasp - R @ GRASP_POINT_IN_EEF, matrix_to_quat(R)
