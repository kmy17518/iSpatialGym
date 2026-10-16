"""Hold targets of R1Pro's position-controlled groups, assembled into full 23-D actions.

Arms and trunk are position controlled, so every action must carry a target for them or they drift toward zero.
:class:`ActionState` keeps a target for every group and fills the whole vector on each call.
"""

from __future__ import annotations

import torch as th

from ispatialgym.robot import r1pro

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0


class ActionState:
    """Hold targets and the assembled action of one R1Pro."""

    def __init__(self, robot):
        self.robot = robot
        self.idx: dict[str, th.Tensor] = {
            k: th.as_tensor(v, dtype=th.long, device="cpu") for k, v in robot.controller_action_idx.items()
        }
        if set(self.idx) != set(r1pro.CONTROLLER_ORDER):
            raise ValueError(f"unexpected controllers {list(self.idx)}")
        self.action_dim = int(robot.action_dim)
        self.base_vel_limits = th.tensor(r1pro.BASE_VEL_LIMITS, dtype=th.float32)
        self.base_vel = th.zeros(3)
        self.trunk_target = th.zeros(4)
        self.arm_target: dict[str, th.Tensor] = {}
        self.gripper_cmd: dict[str, float] = {}
        self.hold_current()
        q = robot.get_joint_positions()
        for arm in r1pro.ARMS:
            idx = robot.gripper_control_idx[arm]
            lo, hi = robot.joint_lower_limits[idx], robot.joint_upper_limits[idx]
            self.gripper_cmd[arm] = GRIPPER_OPEN if float(((q[idx] - lo) / (hi - lo)).mean()) > 0.5 else GRIPPER_CLOSE

    def hold_current(self) -> None:
        """Set the arm and trunk targets to the current joint positions."""
        q = self.robot.get_joint_positions()
        self.trunk_target = q[self.robot.trunk_control_idx].clone()
        for arm in r1pro.ARMS:
            self.arm_target[arm] = q[self.robot.arm_control_idx[arm]].clone()

    def action(self) -> th.Tensor:
        a = th.zeros(self.action_dim, dtype=th.float32)
        a[self.idx["base"]] = th.clamp(self.base_vel / self.base_vel_limits, -1.0, 1.0)
        a[self.idx["trunk"]] = self.trunk_target
        for arm in r1pro.ARMS:
            a[self.idx[f"arm_{arm}"]] = self.arm_target[arm]
            a[self.idx[f"gripper_{arm}"]] = self.gripper_cmd[arm]
        return a


def apply_recorded_targets(hold: ActionState, robot, targets: dict[str, float]) -> list[str]:
    """Copy a TRO's ``robot_state.controller_targets`` (joint name -> drive target) into ``hold`` for every group the
    record covers completely; returns those groups.

    A held payload keeps an arm a steady offset below its target, so the recorded joints are at equilibrium only under
    the recorded targets.
    """
    names = list(robot.joints.keys())
    groups = {"trunk": robot.trunk_control_idx, **{f"arm_{arm}": robot.arm_control_idx[arm] for arm in r1pro.ARMS}}
    done = []
    for group, idx in groups.items():
        joints = [names[int(i)] for i in th.as_tensor(idx, dtype=th.long, device="cpu").tolist()]
        if not all(j in targets for j in joints):
            continue
        values = th.tensor([float(targets[j]) for j in joints], dtype=th.float32)
        if group == "trunk":
            hold.trunk_target = values
        else:
            hold.arm_target[group[len("arm_") :]] = values
        done.append(group)
    return done
