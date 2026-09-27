"""Assembly of 23-D JoyLo actions for R1Pro.

The JoyLo / BEHAVIOR-challenge action is (see :mod:`r1pro_model`)::

    base (3)      vx, vy [m/s] and wz [rad/s] in the base_link frame, normalised to [-1, 1]
    trunk (4)     absolute torso joint positions
    arm_left (7)  absolute joint positions          gripper_left (1)   +1 open ... -1 close
    arm_right (7) absolute joint positions          gripper_right (1)

Because arms and trunk are position controlled, *every* action must carry a target for them or
they drift toward zero. :class:`ActionState` therefore keeps hold targets for all position
controlled groups and fills the whole vector on every call; skills only modify the parts they move.
"""

from __future__ import annotations

from typing import Dict, List

import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.poses import as_tensor

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0


class ActionState:
    """Hold targets + assembly of the full JoyLo action for one R1Pro."""

    def __init__(self, robot):
        self.robot = robot
        self.idx: Dict[str, th.Tensor] = {k: as_tensor(v, th.long) for k, v in robot.controller_action_idx.items()}
        assert set(self.idx) == set(M.CONTROLLER_ORDER), f"unexpected controllers {list(self.idx)}"
        self.action_dim = int(robot.action_dim)
        self.base_vel_limits = th.tensor(M.BASE_VEL_LIMITS, dtype=th.float32)
        self.trunk_target: th.Tensor = th.zeros(4)
        self.arm_target: Dict[str, th.Tensor] = {}
        self.gripper_cmd: Dict[str, float] = {arm: GRIPPER_OPEN for arm in M.ARMS}
        self.base_vel = th.zeros(3)  # m/s, m/s, rad/s (base frame)
        self.hold_current()
        for arm in M.ARMS:
            q = robot.get_joint_positions()[robot.gripper_control_idx[arm]]
            lo, hi = robot.joint_lower_limits[robot.gripper_control_idx[arm]], robot.joint_upper_limits[robot.gripper_control_idx[arm]]
            frac = float(((q - lo) / (hi - lo)).mean())
            self.gripper_cmd[arm] = GRIPPER_OPEN if frac > 0.5 else GRIPPER_CLOSE

    # ------------------------------------------------------------------ targets

    def hold_current(self, arms: bool = True, trunk: bool = True):
        """Snap hold targets to the current joint positions."""
        q = self.robot.get_joint_positions()
        if trunk:
            self.trunk_target = q[self.robot.trunk_control_idx].clone()
        if arms:
            for arm in M.ARMS:
                self.arm_target[arm] = q[self.robot.arm_control_idx[arm]].clone()

    def set_from_q(self, q_full: th.Tensor, arms=("left", "right"), trunk: bool = True):
        """Set hold targets from a full 28-D joint vector (e.g. a planned waypoint)."""
        q_full = as_tensor(q_full)
        if trunk:
            self.trunk_target = q_full[self.robot.trunk_control_idx].clone()
        for arm in arms:
            self.arm_target[arm] = q_full[self.robot.arm_control_idx[arm]].clone()

    def set_base_velocity(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0):
        self.base_vel = th.tensor([vx, vy, wz], dtype=th.float32)

    def stop_base(self):
        self.base_vel = th.zeros(3)

    def set_gripper(self, arm: str, close: bool):
        self.gripper_cmd[arm] = GRIPPER_CLOSE if close else GRIPPER_OPEN

    # ------------------------------------------------------------------ assembly

    def action(self) -> th.Tensor:
        a = th.zeros(self.action_dim, dtype=th.float32)
        a[self.idx["base"]] = th.clamp(self.base_vel / self.base_vel_limits, -1.0, 1.0)
        a[self.idx["trunk"]] = self.trunk_target
        for arm in M.ARMS:
            a[self.idx[f"arm_{arm}"]] = self.arm_target[arm]
            a[self.idx[f"gripper_{arm}"]] = self.gripper_cmd[arm]
        return a

    # ------------------------------------------------------------------ diagnostics

    def arm_error(self, arm: str) -> float:
        q = self.robot.get_joint_positions()[self.robot.arm_control_idx[arm]]
        return float(th.max(th.abs(q - self.arm_target[arm])))

    def trunk_error(self) -> float:
        q = self.robot.get_joint_positions()[self.robot.trunk_control_idx]
        return float(th.max(th.abs(q - self.trunk_target)))


def apply_recorded_targets(hold, robot, targets: Dict[str, float]) -> List[str]:
    """A TRO's ``robot_state.controller_targets`` (joint name -> drive target) into ``hold``'s targets of the
    position-controlled groups the record covers completely (``trunk``, ``arm_left``, ``arm_right``); returns those
    groups. A held payload keeps the arm a steady offset below its target, so the recorded joints are an equilibrium only
    under the recorded targets - holding them at their own values lets the payload sag once more."""
    names = list(robot.joints.keys())
    groups = {"trunk": robot.trunk_control_idx, **{f"arm_{arm}": robot.arm_control_idx[arm] for arm in M.ARMS}}
    done = []
    for group, idx in groups.items():
        joints = [names[int(i)] for i in as_tensor(idx, th.long).tolist()]
        if not all(j in targets for j in joints):
            continue
        vals = th.tensor([float(targets[j]) for j in joints], dtype=th.float32)
        if group == "trunk":
            hold.trunk_target = vals
        else:
            hold.arm_target[group[len("arm_"):]] = vals
        done.append(group)
    return done
