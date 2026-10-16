"""R1Pro in the JoyLo / BEHAVIOR-challenge action space.

The 23-D action, in ``CONTROLLER_ORDER``::

    base (3)       vx, vy [m/s] and wz [rad/s] in the base frame, normalized to [-1, 1] by BASE_VEL_LIMITS
    trunk (4)      absolute torso joint positions
    arm_left (7)   absolute joint positions        gripper_left (1)    +1 open ... -1 close
    arm_right (7)  absolute joint positions        gripper_right (1)
"""

from __future__ import annotations

import copy

ARMS = ("left", "right")
HEAD_CAMERA_LINK = "zed_link"
HEAD_CAMERA_SENSOR_SUFFIX = f"{HEAD_CAMERA_LINK}:Camera:0"
BASE_LINK = "base_link"

CONTROLLER_ORDER = ("base", "trunk", "arm_left", "gripper_left", "arm_right", "gripper_right")
CONTROLLER_DIMS = {"base": 3, "trunk": 4, "arm_left": 7, "gripper_left": 1, "arm_right": 7, "gripper_right": 1}
ACTION_DIM = sum(CONTROLLER_DIMS.values())
BASE_VEL_LIMITS = (0.75, 0.75, 1.0)

#: the BEHAVIOR-challenge controller configuration (OmniGibson/omnigibson/eval/r1pro.yaml)
_JOINT_POSITION = {
    "name": "JointController",
    "motor_type": "position",
    "pos_kp": 150,
    "command_input_limits": None,
    "command_output_limits": None,
    "use_impedances": False,
    "use_delta_commands": False,
}
_GRIPPER = {
    "name": "MultiFingerGripperController",
    "mode": "smooth",
    "command_input_limits": "default",
    "command_output_limits": "default",
}
CONTROLLER_CONFIG = {
    "base": {
        "name": "HolonomicBaseJointController",
        "motor_type": "velocity",
        "vel_kp": 150,
        "command_input_limits": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        "command_output_limits": [[-v for v in BASE_VEL_LIMITS], list(BASE_VEL_LIMITS)],
        "use_impedances": False,
    },
    "trunk": dict(_JOINT_POSITION),
    "arm_left": dict(_JOINT_POSITION),
    "arm_right": dict(_JOINT_POSITION),
    "gripper_left": dict(_GRIPPER),
    "gripper_right": dict(_GRIPPER),
}

#: virtual base joints at zero, the trunk at JoyLo's default translate, arms at zero, grippers open
RESET_JOINT_POS = [0.0] * 6 + [1.025, -1.45, -0.47, 0.0] + [0.0] * 14 + [0.05] * 4

#: the BEHAVIOR-1K evaluator's chassis mass (omnigibson/eval/evaluator.py: EVAL_BASE_LINK_MASS)
EVAL_BASE_LINK_MASS_KG = 250.0


def robot_config(name: str = "robot", **overrides) -> dict:
    """OmniGibson ``robots`` entry of an R1Pro in the JoyLo action space."""
    cfg = {
        "type": "R1Pro",
        "name": name,
        "obs_modalities": ["rgb", "proprio"],
        "action_normalize": False,
        "self_collisions": True,
        "grasping_mode": "assisted",
        "reset_joint_pos": list(RESET_JOINT_POS),
        "controller_config": copy.deepcopy(CONTROLLER_CONFIG),
    }
    cfg.update(overrides)
    return cfg


def apply_eval_base_mass(robot, mass_kg: float | None = EVAL_BASE_LINK_MASS_KG) -> dict[str, float]:
    """Set the base link's mass as the BEHAVIOR-1K evaluator does (the simulator is stopped around the write)."""
    import omnigibson as og

    link = robot.base_footprint_link if hasattr(robot, "base_footprint_link") else robot.links[BASE_LINK]
    before = float(link.mass)
    if not mass_kg or abs(before - mass_kg) < 1e-6:
        return {"before": before, "after": before}
    og.sim.stop()
    link.mass = float(mass_kg)
    og.sim.play()
    og.sim.update_handles()
    return {"before": before, "after": float(link.mass)}
