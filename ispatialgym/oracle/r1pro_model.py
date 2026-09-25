"""Static model of the R1Pro robot used by the oracle stack.

This module has no OmniGibson / cuRobo runtime dependency so it can be imported and tested
without launching Isaac Sim. It provides:

* joint-group definitions (base, torso, arms, fingers) in OmniGibson naming,
* the JoyLo trunk keyframes,
* the *embodiments* (active joint sets) we plan with,
* generation of a derived URDF with the holonomic virtual base chain, the ``*_eef_link``
  frames and ``attached_object_*`` frames (cuRobo v2 is URDF-only),
* generation of a cuRobo v2 robot dictionary for a given embodiment (locked joints).
"""

from __future__ import annotations

import copy
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import yaml

# --------------------------------------------------------------------------------------
# Joint / link names (OmniGibson naming)
# --------------------------------------------------------------------------------------

ARMS = ("left", "right")

BASE_VIRTUAL_JOINTS = [
    "base_footprint_x_joint",
    "base_footprint_y_joint",
    "base_footprint_z_joint",
    "base_footprint_rx_joint",
    "base_footprint_ry_joint",
    "base_footprint_rz_joint",
]
#: The three planar DoFs actually driven by the holonomic base controller.
BASE_JOINTS = ["base_footprint_x_joint", "base_footprint_y_joint", "base_footprint_rz_joint"]
#: Virtual DoFs that are always locked (they only exist so PhysX can float the base).
BASE_ALWAYS_LOCKED_JOINTS = ["base_footprint_z_joint", "base_footprint_rx_joint", "base_footprint_ry_joint"]

TORSO_JOINTS = [f"torso_joint{i}" for i in range(1, 5)]
ARM_JOINTS: Dict[str, List[str]] = {arm: [f"{arm}_arm_joint{i}" for i in range(1, 8)] for arm in ARMS}
FINGER_JOINTS: Dict[str, List[str]] = {arm: [f"{arm}_gripper_finger_joint{i}" for i in (1, 2)] for arm in ARMS}

#: All 28 joints in the order OmniGibson exposes them (breadth-first, arms interleaved).
ALL_JOINTS: List[str] = (
    BASE_VIRTUAL_JOINTS
    + TORSO_JOINTS
    + [j for pair in zip(ARM_JOINTS["left"], ARM_JOINTS["right"]) for j in pair]
    + FINGER_JOINTS["left"]
    + FINGER_JOINTS["right"]
)
assert len(ALL_JOINTS) == 28

ROOT_LINK = "base_footprint_x"  # articulation root in OmniGibson; cuRobo kinematics base link
BASE_LINK = "base_link"  # base footprint link (what ``robot.get_position_orientation`` reports)
TORSO_TOP_LINK = "torso_link4"
HEAD_CAMERA_LINK = "zed_link"
EEF_LINK: Dict[str, str] = {arm: f"{arm}_eef_link" for arm in ARMS}
GRIPPER_LINK: Dict[str, str] = {arm: f"{arm}_gripper_link" for arm in ARMS}
FINGER_LINKS: Dict[str, List[str]] = {arm: [f"{arm}_gripper_finger_link{i}" for i in (1, 2)] for arm in ARMS}
ATTACHED_OBJECT_LINK: Dict[str, str] = {arm: f"attached_object_{arm}_eef_link" for arm in ARMS}
#: Number of collision-sphere slots reserved for a grasped object on each arm.
ATTACHED_OBJECT_NUM_SPHERES = 32

#: Pose of ``{arm}_eef_link`` in ``{arm}_gripper_link`` (from the OmniGibson USD: 6 cm along -z, 180 deg about y).
EEF_IN_GRIPPER_XYZ = (0.0, 0.0, -0.06)
EEF_IN_GRIPPER_RPY = (0.0, math.pi, 0.0)

#: Head camera sensor key suffix (``<robot_name>:zed_link:Camera:0``).
HEAD_CAMERA_SENSOR_SUFFIX = f"{HEAD_CAMERA_LINK}:Camera:0"

#: The BEHAVIOR-1K evaluator's chassis mass for r1 / r1pro (``omnigibson/eval/evaluator.py: EVAL_BASE_LINK_MASS``,
#: "heavier base so contact with the (heavy) world does not shove the robot around during eval"); the URDF's
#: ``base_link`` weighs 47.63 kg. The holonomic base stands on three bounding-sphere wheels with its z / rx / ry virtual
#: joints undriven and gravity disabled on every link above it, so the chassis mass alone resists the wrench of a loaded
#: arm: at 47.6 kg the dishwasher pull tipped it 0.8 deg / 5 mm and it snapped back at the release (0921-050314).
#: ``ISG_BASE_MASS_KG`` overrides (0 = leave the URDF's mass).
EVAL_BASE_LINK_MASS_KG = 250.0


def apply_eval_base_mass(robot, mass_kg: Optional[float] = None) -> Dict[str, float]:
    """Set the robot's base footprint link mass the way the BEHAVIOR-1K evaluator does (a mass write needs the simulator
    stopped: ``og.sim.stop()`` / ``play()`` around it, then ``update_handles``). ``mass_kg`` defaults to
    ``ISG_BASE_MASS_KG`` or :data:`EVAL_BASE_LINK_MASS_KG`; ``<= 0`` leaves the mass alone. Returns
    ``{"before": kg, "after": kg}``."""
    import omnigibson as og

    if mass_kg is None:
        mass_kg = float(os.environ.get("ISG_BASE_MASS_KG", EVAL_BASE_LINK_MASS_KG))
    link = robot.base_footprint_link if hasattr(robot, "base_footprint_link") else robot.links[BASE_LINK]
    before = float(link.mass)
    if mass_kg <= 0.0 or abs(before - mass_kg) < 1e-6:
        return {"before": before, "after": before}
    og.sim.stop()
    link.mass = float(mass_kg)
    og.sim.play()
    if hasattr(og.sim, "update_handles"):
        og.sim.update_handles()
    return {"before": before, "after": float(link.mass)}

# --------------------------------------------------------------------------------------
# JoyLo trunk keyframes (joylo/gello/utils/og_teleop_cfg.py, r1pro entry)
# --------------------------------------------------------------------------------------

TORSO_UPRIGHT = (0.45, -0.4, 0.0, 0.0)
TORSO_DOWNWARD = (1.6, -2.5, -0.94, 0.0)
TORSO_GROUND = (1.735, -2.57, -2.1, 0.0)
TRUNK_TRANSLATE_RANGE = (0.0, 2.0)
DEFAULT_TRUNK_TRANSLATE = 0.5
#: Index (within the 4 torso joints) that JoyLo uses for the tilt offset.
TRUNK_TILT_JOINT_INDEX = 2

# --------------------------------------------------------------------------------------
# JoyLo / BEHAVIOR-challenge action layout (23-D)
# --------------------------------------------------------------------------------------

CONTROLLER_ORDER = ["base", "trunk", "arm_left", "gripper_left", "arm_right", "gripper_right"]
CONTROLLER_DIMS = {"base": 3, "trunk": 4, "arm_left": 7, "gripper_left": 1, "arm_right": 7, "gripper_right": 1}
ACTION_DIM = sum(CONTROLLER_DIMS.values())
#: Base velocity command limits in the base frame after de-normalisation: (vx, vy) m/s, wz rad/s.
BASE_VEL_LIMITS = (0.75, 0.75, 1.0)

#: JoyLo / BEHAVIOR-challenge controller configuration (OmniGibson/omnigibson/eval/r1pro.yaml).
JOYLO_CONTROLLER_CONFIG = {
    "base": {
        "name": "HolonomicBaseJointController",
        "motor_type": "velocity",
        "vel_kp": 150,
        "command_input_limits": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        "command_output_limits": [[-BASE_VEL_LIMITS[0], -BASE_VEL_LIMITS[1], -BASE_VEL_LIMITS[2]], list(BASE_VEL_LIMITS)],
        "use_impedances": False,
    },
    "trunk": {
        "name": "JointController",
        "motor_type": "position",
        "pos_kp": 150,
        "command_input_limits": None,
        "command_output_limits": None,
        "use_impedances": False,
        "use_delta_commands": False,
    },
    "arm_left": {
        "name": "JointController",
        "motor_type": "position",
        "pos_kp": 150,
        "command_input_limits": None,
        "command_output_limits": None,
        "use_impedances": False,
        "use_delta_commands": False,
    },
    "arm_right": {
        "name": "JointController",
        "motor_type": "position",
        "pos_kp": 150,
        "command_input_limits": None,
        "command_output_limits": None,
        "use_impedances": False,
        "use_delta_commands": False,
    },
    "gripper_left": {
        "name": "MultiFingerGripperController",
        "mode": "smooth",
        "command_input_limits": "default",
        "command_output_limits": "default",
    },
    "gripper_right": {
        "name": "MultiFingerGripperController",
        "mode": "smooth",
        "command_input_limits": "default",
        "command_output_limits": "default",
    },
}

#: JoyLo default reset joint positions (trunk at DEFAULT_TRUNK_TRANSLATE, arms zero, grippers open).
JOYLO_RESET_JOINT_POS = [0.0] * 6 + [1.025, -1.45, -0.47, 0.0] + [0.0] * 14 + [0.05] * 4

#: Natural "ready" arm configuration (elbows bent, hands ~0.5 m in front of the shoulders, grippers
#: pointing forward-down). Found by IK with a large joint-limit margin; used before lowering the
#: torso and as the rest pose between manipulations. Left arm is the mirror image.
ARM_READY_Q: Dict[str, List[float]] = {
    "right": [-0.073, -0.326, 0.951, -1.639, -0.632, 0.137, -0.926],
    "left": [-0.073, 0.326, -0.951, -1.639, 0.632, 0.137, 0.926],
}


def joylo_robot_config(name: str = "robot", **overrides) -> dict:
    """OmniGibson ``robots`` entry for an R1Pro driven in the JoyLo action space."""
    cfg = {
        "type": "R1Pro",
        "name": name,
        "obs_modalities": ["rgb", "proprio"],
        "action_normalize": False,
        "self_collisions": True,
        "grasping_mode": "assisted",
        "reset_joint_pos": list(JOYLO_RESET_JOINT_POS),
        "controller_config": copy.deepcopy(JOYLO_CONTROLLER_CONFIG),
    }
    cfg.update(overrides)
    return cfg

# --------------------------------------------------------------------------------------
# Embodiments (active joint groups for planning)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Embodiment:
    """A set of joints cuRobo is allowed to move, plus the frames it computes poses for."""

    name: str
    active_joints: List[str]
    tool_frames: List[str]
    #: Per-joint velocity scale applied to URDF velocity limits (natural-looking demo speeds).
    velocity_scale: float = 1.0
    max_acceleration: float = 10.0
    max_jerk: float = 500.0

    @property
    def locked_joints(self) -> List[str]:
        return [j for j in ALL_JOINTS if j not in self.active_joints]

    def has_frame(self, name: str) -> bool:
        return name in self.tool_frames


def _emb(name, active, frames, **kw):
    return Embodiment(name=name, active_joints=list(active), tool_frames=list(frames), **kw)


EMBODIMENTS: Dict[str, Embodiment] = {
    # Planar base only; whole body (at its current configuration) is checked for collisions.
    "base": _emb("base", BASE_JOINTS, [BASE_LINK], velocity_scale=0.6, max_acceleration=1.0, max_jerk=10.0),
    # Single arm, torso and base locked.
    "left_arm": _emb("left_arm", ARM_JOINTS["left"], [EEF_LINK["left"]], velocity_scale=0.25, max_acceleration=4.0, max_jerk=200.0),
    "right_arm": _emb("right_arm", ARM_JOINTS["right"], [EEF_LINK["right"]], velocity_scale=0.25, max_acceleration=4.0, max_jerk=200.0),
    # Both arms (bimanual reach); tool frames are both eefs.
    "both_arms": _emb(
        "both_arms",
        ARM_JOINTS["left"] + ARM_JOINTS["right"],
        [EEF_LINK["left"], EEF_LINK["right"]],
        velocity_scale=0.25,
        max_acceleration=4.0,
        max_jerk=200.0,
    ),
    # Arm + torso for hard reaches (opt-in; see design doc on why the torso is normally locked).
    "left_arm_torso": _emb("left_arm_torso", TORSO_JOINTS + ARM_JOINTS["left"], [EEF_LINK["left"]], velocity_scale=0.25, max_acceleration=3.0, max_jerk=150.0),
    "right_arm_torso": _emb("right_arm_torso", TORSO_JOINTS + ARM_JOINTS["right"], [EEF_LINK["right"]], velocity_scale=0.25, max_acceleration=3.0, max_jerk=150.0),
    # Everything (IK-only queries such as "where should base+torso be so that the camera / hand reaches X").
    "whole_body": _emb(
        "whole_body",
        BASE_JOINTS + TORSO_JOINTS + ARM_JOINTS["left"] + ARM_JOINTS["right"],
        [EEF_LINK["left"], EEF_LINK["right"], HEAD_CAMERA_LINK, BASE_LINK, TORSO_TOP_LINK],
        velocity_scale=0.5,
        max_acceleration=3.0,
        max_jerk=150.0,
    ),
    # Base + torso (camera relocalization style viewpoint IK).
    "base_torso": _emb(
        "base_torso",
        BASE_JOINTS + TORSO_JOINTS,
        [HEAD_CAMERA_LINK, BASE_LINK, TORSO_TOP_LINK],
        velocity_scale=0.5,
        max_acceleration=1.0,
        max_jerk=10.0,
    ),
}

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------


def robot_assets_dir() -> str:
    """Directory of the OmniGibson R1Pro assets (``.../omnigibson-robot-assets/models/r1pro``)."""
    try:
        from omnigibson.utils.asset_utils import get_dataset_path  # lazy: needs omnigibson

        return os.path.join(get_dataset_path("omnigibson-robot-assets"), "models", "r1pro")
    except Exception:  # pragma: no cover - fallback for tooling without omnigibson
        from ispatialgym import REPO_ROOT

        return os.path.join(REPO_ROOT, "Behavior-1K", "datasets", "omnigibson-robot-assets", "models", "r1pro")


def source_urdf_path() -> str:
    return os.path.join(robot_assets_dir(), "urdf", "r1pro.urdf")


def source_curobo_yaml_path() -> str:
    """The shipped v1 cuRobo description; we reuse its collision spheres / self-collision data."""
    return os.path.join(robot_assets_dir(), "curobo", "r1pro_description_curobo_default.yaml")


_FINGER_HULLS: Dict[str, Optional[List[tuple]]] = {}


def _rpy_matrix(r: float, p: float, y: float):
    import numpy as np

    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def finger_hulls_eef(arm: str = "right") -> Optional[List[tuple]]:
    """Convex-hull vertices of the two finger meshes (``FINGER_LINKS``) in the eef frame at finger joint 0, with each
    finger's slide axis in the eef frame: ``[(points (N, 3) ndarray, axis (3,) ndarray), ...]``. The finger joint value
    ``q`` moves the finger by ``q * axis`` (URDF prismatic joints ``{arm}_gripper_finger_joint1/2``, opening ~ 2 q).
    Parsed once from the shipped URDF + visual meshes (``urdf/meshes/<finger link>.obj``); None when unavailable
    (tooling without the assets). Used by the sampler's thin-part rule: the finger envelope - 3.8 cm across the closing
    plane, so a 30 deg tilt hangs a tip corner up to 1.2 cm below the fingertip axis point - must clear the support."""
    if arm in _FINGER_HULLS:
        return _FINGER_HULLS[arm]
    out: Optional[List[tuple]] = None
    try:
        import numpy as np
        from scipy.spatial import ConvexHull

        root = ET.parse(source_urdf_path()).getroot()
        joints = {j.find("child").get("link"): j for j in root.iter("joint") if j.find("child") is not None}
        R_ge = _rpy_matrix(*EEF_IN_GRIPPER_RPY)  # gripper -> eef rotation; p_eef = R_ge^T (p_gripper - t)
        t_ge = np.asarray(EEF_IN_GRIPPER_XYZ, dtype=float)
        out = []
        for link in FINGER_LINKS[arm]:
            j = joints[link]
            assert j.find("parent").get("link") == GRIPPER_LINK[arm], f"{link} is not a child of {GRIPPER_LINK[arm]}"
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") if o is not None and o.get("xyz") else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()]
            R_gf = _rpy_matrix(*rpy)
            axis_g = R_gf @ np.array([float(v) for v in j.find("axis").get("xyz").split()])
            lk = [l for l in root.iter("link") if l.get("name") == link][0]
            vis = lk.find("visual")
            mesh = vis.find("geometry").find("mesh")
            path = os.path.join(robot_assets_dir(), "urdf", mesh.get("filename"))
            scale = np.array([float(v) for v in (mesh.get("scale") or "1 1 1").split()])
            pts = np.array([[float(v) for v in line.split()[1:4]] for line in open(path) if line.startswith("v ")]) * scale
            vo = vis.find("origin")
            if vo is not None:
                pts = pts @ _rpy_matrix(*[float(v) for v in (vo.get("rpy") or "0 0 0").split()]).T + np.array([float(v) for v in (vo.get("xyz") or "0 0 0").split()])
            pts_g = pts @ R_gf.T + xyz  # finger frame -> gripper frame (joint at q = 0)
            pts_e = (pts_g - t_ge) @ R_ge  # gripper -> eef (R_ge is symmetric for rpy (0, pi, 0); written as R^T via @ R)
            hull = pts_e[ConvexHull(pts_e).vertices]
            out.append((hull.astype(float), (R_ge.T @ axis_g).astype(float)))
    except Exception:  # noqa: BLE001 - assets missing: the caller falls back to the fingertip constants
        out = None
    _FINGER_HULLS[arm] = out
    return out


def finger_points_eef(finger_q: float, arm: str = "right"):
    """The finger hull vertices in the eef frame at finger joint value ``finger_q`` (both fingers stacked, (N, 3) ndarray),
    None without the assets."""
    import numpy as np

    hulls = finger_hulls_eef(arm)
    if not hulls:
        return None
    return np.vstack([pts + float(finger_q) * axis for pts, axis in hulls])


def cache_dir() -> str:
    d = os.environ.get("ISPATIALGYM_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "ispatialgym"))
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------------------
# Derived URDF
# --------------------------------------------------------------------------------------


def _add_link(root: ET.Element, name: str) -> ET.Element:
    link = ET.SubElement(root, "link", name=name)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="0.001")
    ET.SubElement(inertial, "inertia", ixx="1e-6", ixy="0", ixz="0", iyy="1e-6", iyz="0", izz="1e-6")
    return link


def _add_joint(
    root: ET.Element,
    name: str,
    jtype: str,
    parent: str,
    child: str,
    xyz=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
    axis: Optional[Sequence[float]] = None,
    limits: Optional[Sequence[float]] = None,
) -> ET.Element:
    joint = ET.SubElement(root, "joint", name=name, type=jtype)
    ET.SubElement(joint, "origin", xyz=" ".join(f"{v:.9g}" for v in xyz), rpy=" ".join(f"{v:.9g}" for v in rpy))
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    if axis is not None:
        ET.SubElement(joint, "axis", xyz=" ".join(f"{v:g}" for v in axis))
    if limits is not None:
        lower, upper, velocity, effort = limits
        ET.SubElement(joint, "limit", lower=f"{lower:.9g}", upper=f"{upper:.9g}", velocity=f"{velocity:g}", effort=f"{effort:g}")
    return joint


def generate_holonomic_urdf(
    src_urdf: Optional[str] = None,
    dst_urdf: Optional[str] = None,
    base_xy_limit: float = 10.0,
    base_yaw_limit: float = 2.0 * math.pi,
    base_lin_vel: float = BASE_VEL_LIMITS[0],
    base_ang_vel: float = BASE_VEL_LIMITS[2],
) -> str:
    """Write the derived R1Pro URDF used by cuRobo and return its path.

    Additions relative to the shipped ``r1pro.urdf``:

    * the six virtual base joints/links, in the exact chain OmniGibson creates in the USD
      (``base_footprint_x -> y -> z -> rx -> ry -> rz -> base_link``), so joint values are
      identical to the simulator's virtual joint positions,
    * ``left_eef_link`` / ``right_eef_link`` fixed frames,
    * ``attached_object_{left,right}_eef_link`` fixed frames for grasped-object spheres,
    * absolute mesh paths (meshes are not loaded by cuRobo, but tools can visualise).
    """
    src_urdf = src_urdf or source_urdf_path()
    dst_urdf = dst_urdf or os.path.join(cache_dir(), "r1pro", "r1pro_holonomic.urdf")
    os.makedirs(os.path.dirname(dst_urdf), exist_ok=True)

    tree = ET.parse(src_urdf)
    root = tree.getroot()
    src_dir = os.path.dirname(os.path.abspath(src_urdf))

    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn and not os.path.isabs(fn):
            mesh.set("filename", os.path.join(src_dir, fn))

    # Mirror OmniGibson's articulation: wheel / steering joints are fixed in the USD (the holonomic
    # base is modelled by the virtual joints), so fix every movable joint OmniGibson does not expose.
    known = set(ALL_JOINTS)
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed" and joint.get("name") not in known:
            joint.set("type", "fixed")
            for tag in ("axis", "limit", "dynamics", "mimic", "calibration", "safety_controller"):
                for el in joint.findall(tag):
                    joint.remove(el)

    # Virtual base chain (matches omnigibson/examples/robots/import_custom_robot.py ordering).
    chain = [
        ("base_footprint_x", "base_footprint_x_joint", "prismatic", (1, 0, 0), (-base_xy_limit, base_xy_limit, base_lin_vel, 1000.0)),
        ("base_footprint_y", "base_footprint_y_joint", "prismatic", (0, 1, 0), (-base_xy_limit, base_xy_limit, base_lin_vel, 1000.0)),
        ("base_footprint_z", "base_footprint_z_joint", "prismatic", (0, 0, 1), (-1.0, 1.0, 0.1, 1000.0)),
        ("base_footprint_rx", "base_footprint_rx_joint", "revolute", (1, 0, 0), (-math.pi, math.pi, 0.1, 1000.0)),
        ("base_footprint_ry", "base_footprint_ry_joint", "revolute", (0, 1, 0), (-math.pi, math.pi, 0.1, 1000.0)),
        ("base_footprint_rz", "base_footprint_rz_joint", "revolute", (0, 0, 1), (-base_yaw_limit, base_yaw_limit, base_ang_vel, 1000.0)),
    ]
    for link_name, _, _, _, _ in chain:
        _add_link(root, link_name)
    for i, (link_name, joint_name, jtype, axis, limits) in enumerate(chain):
        child = chain[i + 1][0] if i + 1 < len(chain) else BASE_LINK
        _add_joint(root, joint_name, jtype, link_name, child, axis=axis, limits=limits)

    for arm in ARMS:
        _add_link(root, EEF_LINK[arm])
        _add_joint(root, f"{arm}_eef_link_joint", "fixed", GRIPPER_LINK[arm], EEF_LINK[arm], xyz=EEF_IN_GRIPPER_XYZ, rpy=EEF_IN_GRIPPER_RPY)
        _add_link(root, ATTACHED_OBJECT_LINK[arm])
        _add_joint(root, f"{ATTACHED_OBJECT_LINK[arm]}_joint", "fixed", EEF_LINK[arm], ATTACHED_OBJECT_LINK[arm])

    ET.indent(tree, space="  ")
    tree.write(dst_urdf, xml_declaration=True, encoding="utf-8")
    return dst_urdf


# --------------------------------------------------------------------------------------
# cuRobo v2 robot dictionary
# --------------------------------------------------------------------------------------

#: (link_a, link_b) templates (``{arm}`` placeholder) added to the v1 self-collision ignore list.
#:
#: * ``arm_link1 <-> arm_link3`` overlap by 3 mm in the straight-arm pose (cuRobo v2 has no tolerance slack).
#: * A grasped object (spheres on ``attached_object_<arm>_eef_link``) sits between the pads, i.e. inside the palm's
#:   and the wrist's collision spheres by construction: the shipped yaml exempts it against the finger links only,
#:   so every grasp that puts the object deep in the gripper self-collided once attached and every attached check
#:   (lift, carry, place) failed (owner decision A7, 2026-09-16). As cuRobo's stock configs do (franka:
#:   ``attached_object`` vs ``panda_hand`` / fingers / ``panda_link7``), the object is exempt against the gripper
#:   body, the wrist camera and the two wrist links; against everything else (the other arm, the torso, the base,
#:   the world) it is still checked.
#: * ``torso_link4 <-> arm_link2``: the shoulder-mount sphere of the trunk (r 8 cm at the arm joint) against the upper
#:   arm's first sphere (r 5.5 cm, 6.5 cm below the joint) sits at the margin in the arms-down rest posture (3.7 mm
#:   clearance) and inside it as soon as the upper arm turns toward the body: the wall-alignment brace left the right
#:   upper arm 5.4 mm inside, the sampler's push pre witness for the left arm exactly on the boundary, and every
#:   plan failed "start / goal configuration in collision" (license plate 0916-163025 / 173228). A grandparent pair
#:   (torso_link4 - arm_link1 - arm_link2), exempt as cuRobo's stock configs exempt them; the chest sphere
#:   (``torso_link4`` r 15 cm) and the base still check against the upper arm.
EXTRA_SELF_COLLISION_IGNORE_PAIRS = (
    # W1 (owner 2026-09-20 22:15; pod probe x0l0zrorcpakz1 2026-09-21): along the JoyLo keyframe curve the padded spheres of
    # base_link and torso_link2 overlap from translate 1.0 on (2 mm at 1.0, 20 mm at 1.5, 37 mm at the GROUND keyframe 2.0)
    # while the links' collision MESHES never touch (clearance 5.6 cm at 1.0, 3.5 cm at 1.5, 1.5 cm at 2.0; the simulator
    # reports no contact) - torso_link2's pose relative to the base is a function of translate alone, so the pair is exempt
    # and the "ground half" of the curve (translate 1 -> 2, shoulders 8 cm lower, floor reach 0.40-0.85 m) becomes usable
    ("base_link", "torso_link2"),
    ("{arm}_arm_link1", "{arm}_arm_link3"),
    ("torso_link4", "{arm}_arm_link2"),
    ("attached_object_{arm}_eef_link", "{arm}_gripper_link"),
    ("attached_object_{arm}_eef_link", "{arm}_realsense_link"),
    ("attached_object_{arm}_eef_link", "{arm}_arm_link7"),
    ("attached_object_{arm}_eef_link", "{arm}_arm_link6"),
    ("attached_object_{arm}_eef_link", "{arm}_gripper_finger_link1"),
    ("attached_object_{arm}_eef_link", "{arm}_gripper_finger_link2"),
)

_V1_KEYS_TO_DROP = {
    "usd_flip_joint_limits",
    "usd_flip_joints",
    "usd_robot_root",
    "use_usd_kinematics",
    "external_asset_path",
    "link_names",
    "ee_link",
    "extra_links",  # attached_object_* links are real links in the derived URDF
    "mesh_link_names",
    "use_global_cumul",
}


def load_v1_kinematics_yaml(path: Optional[str] = None) -> dict:
    with open(path or source_curobo_yaml_path()) as f:
        return yaml.safe_load(f)["robot_cfg"]["kinematics"]


def build_curobo_robot_dict(
    embodiment: Embodiment,
    urdf_path: str,
    lock_values: Dict[str, float],
    default_joint_position: Dict[str, float],
    v1_kinematics: Optional[dict] = None,
    collision_sphere_buffer: float = 0.002,
    load_collision: bool = True,
) -> dict:
    """Build a cuRobo v2 ``robot_cfg`` dictionary for ``embodiment``.

    Args:
        embodiment: joint group to keep active; every other joint of :data:`ALL_JOINTS` is locked.
        urdf_path: derived URDF from :func:`generate_holonomic_urdf`.
        lock_values: joint value for every locked joint (typically the current sim state; the
            planner refreshes these at plan time anyway).
        default_joint_position: default (retract) value for every joint in :data:`ALL_JOINTS`.
        v1_kinematics: parsed ``robot_cfg.kinematics`` of the shipped v1 yaml (spheres etc.).
    """
    v1 = copy.deepcopy(v1_kinematics if v1_kinematics is not None else load_v1_kinematics_yaml())
    for k in _V1_KEYS_TO_DROP:
        v1.pop(k, None)

    kin = dict(v1)
    kin["format_version"] = 2.0
    kin["urdf_path"] = os.path.abspath(urdf_path)
    kin["asset_root_path"] = os.path.dirname(os.path.abspath(urdf_path))
    kin["base_link"] = ROOT_LINK
    kin["tool_frames"] = list(embodiment.tool_frames)
    kin["collision_sphere_buffer"] = collision_sphere_buffer
    kin["extra_collision_spheres"] = {ATTACHED_OBJECT_LINK[a]: ATTACHED_OBJECT_NUM_SPHERES for a in ARMS}
    kin["grasp_contact_link_names"] = [
        *[GRIPPER_LINK[a] for a in ARMS],
        *[l for a in ARMS for l in FINGER_LINKS[a]],
        *[ATTACHED_OBJECT_LINK[a] for a in ARMS],
    ]
    # Sphere pairs that overlap by a few mm in the arm's nominal (straight) pose and therefore must
    # be ignored for self-collision (not in the shipped v1 list; cuRobo v2 has no tolerance slack).
    ignore = {k: list(v) for k, v in (kin.get("self_collision_ignore") or {}).items()}
    for arm in ARMS:
        for a, b in EXTRA_SELF_COLLISION_IGNORE_PAIRS:
            a, b = a.format(arm=arm), b.format(arm=arm)
            if b not in ignore.setdefault(a, []) and a not in ignore.get(b, []):
                ignore[a].append(b)
    kin["self_collision_ignore"] = ignore
    if not load_collision:
        kin["collision_link_names"] = []
        kin["collision_spheres"] = None
        kin["extra_collision_spheres"] = None
        kin["self_collision_ignore"] = None
        kin["self_collision_buffer"] = None

    n = len(ALL_JOINTS)
    kin["cspace"] = {
        "joint_names": list(ALL_JOINTS),
        "default_joint_position": [float(default_joint_position[j]) for j in ALL_JOINTS],
        "cspace_distance_weight": [1.0] * n,
        "null_space_weight": [1.0] * n,
        "max_acceleration": float(embodiment.max_acceleration),
        "max_jerk": float(embodiment.max_jerk),
        "velocity_scale": float(embodiment.velocity_scale),
    }
    kin["lock_joints"] = {j: float(lock_values[j]) for j in embodiment.locked_joints}
    return {"robot_cfg": {"kinematics": kin}}


def dump_curobo_yaml(robot_dict: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(robot_dict, f, default_flow_style=None, sort_keys=False)
    return path


# --------------------------------------------------------------------------------------
# Small helpers used across the oracle stack
# --------------------------------------------------------------------------------------


def joint_index_map(joint_names: Iterable[str]) -> Dict[str, int]:
    return {n: i for i, n in enumerate(joint_names)}


def arm_of_eef_link(link_name: str) -> str:
    for arm, l in EEF_LINK.items():
        if l == link_name:
            return arm
    raise KeyError(link_name)
