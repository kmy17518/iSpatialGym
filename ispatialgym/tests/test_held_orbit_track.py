"""``IKChecker.track_chain`` as the mental-rotation orbit check uses it (owner 2026-10-13, mental_rotation-walker: "the robot seems to
stutter"): ``refine_jumps`` tracks a chain failing on a joint jump once more at half spacing, as the oracle's ``arm_along`` does, and
``min_limit_margin`` rejects a chain whose tracked configurations come closer to an arm joint limit than the margin (the oracle's
differential IK froze right_arm_joint6 8 mrad from its limit and swung the other joints). Both off by default: today's tracking."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2.ik import IKChecker
from ispatialgym.tests.test_v2_oracle import FakeRobot

J6_LIMIT = 1.047198


def ramp(j6_end):
    """A plan whose path turns right_arm_joint6 linearly from the start to ``j6_end`` over the waypoints."""

    def plan(robot, q_start, n):
        j6 = int(robot.arm_control_idx["right"][5])
        path = q_start.unsqueeze(0).repeat(n + 1, 1).clone()
        path[:, j6] = th.linspace(float(q_start[j6]), j6_end, n + 1)
        return SimpleNamespace(success=True, status="ok", q_traj=path, diagnostics={})

    return plan


def jump(robot, q_start, n):
    return SimpleNamespace(success=False, status="joint jump 0.24 rad at waypoint 3", q_traj=None, diagnostics={})


class _Backend:
    def __init__(self, robot, plans):
        self.robot, self.plans, self.calls = robot, list(plans), []

    def solve_ik_path(self, emb, waypoints, q_start=None, **kw):
        self.calls.append({"n": len(waypoints), **kw})
        return self.plans.pop(0)(self.robot, q_start, len(waypoints))


class _Kit:
    def __init__(self, plans):
        self.robot = FakeRobot()
        j6 = self.robot.arm_control_idx["right"][5]
        self.robot.joint_lower_limits[j6], self.robot.joint_upper_limits[j6] = -J6_LIMIT, J6_LIMIT  # the R1Pro's wrist pitch
        self.backend = _Backend(self.robot, plans)
        self.stats = {}

    def _ensure_synced(self):
        pass

    def collision_free(self, q, margin=0.0):
        q = th.as_tensor(q)
        return th.ones(q.shape[0] if q.dim() == 2 else 1, dtype=th.bool)


def _chain():
    q = C.quat_from_axis_angle(th.tensor([0.0, 0.0, 1.0]), 0.2)
    return [(th.tensor([0.5, -0.2, 1.1]), q), (th.tensor([0.5, -0.2, 1.13]), q)]  # 3 cm: 3 dense poses at the oracle's 1 cm


def _q_from(kit, j6=-1.0):
    q = kit.robot.q.clone()
    q[kit.robot.arm_control_idx["right"][5]] = j6
    return q


@pytest.mark.parametrize("j6_end, margin, ok", [(-1.04, 0.02, False), (-1.04, None, True), (-1.02, 0.02, True)])
def test_min_limit_margin_rejects_a_chain_tracked_against_a_joint_limit(j6_end, margin, ok):
    kit = _Kit([ramp(j6_end)])
    ikc = IKChecker(kit)
    q_end, why = ikc.track_chain(_q_from(kit), _chain(), "right", min_limit_margin=margin)
    if ok:
        assert why == "ok" and float(q_end[kit.robot.arm_control_idx["right"][5]]) == pytest.approx(j6_end)
    else:
        assert q_end is None and why.startswith("joint_limit: chain tracking (3 dense poses): right_arm_joint6 7.2 mrad from its limit at dense pose 3")
    assert [c["n"] for c in kit.backend.calls] == [3]


def test_a_start_inside_the_margin_is_not_judged_by_the_chain():
    # the start is the caller's to judge (the mental-rotation draw skips such a start): the chain checks the configurations after it
    kit = _Kit([ramp(-1.035)])
    q_end, why = IKChecker(kit).track_chain(_q_from(kit, j6=-1.035), _chain(), "right", min_limit_margin=0.02)
    assert q_end is None and "dense pose 1" in why  # the chain stays where the start stood: 12.2 mrad from the limit
    kit = _Kit([ramp(-0.95)])
    assert IKChecker(kit).track_chain(_q_from(kit, j6=-1.035), _chain(), "right", min_limit_margin=0.02)[1] == "ok"  # it leaves at once


def test_refine_jumps_tracks_once_more_at_half_spacing_as_arm_along_does():
    kit = _Kit([jump, ramp(-0.95)])
    q_end, why = IKChecker(kit).track_chain(_q_from(kit), _chain(), "right", refine_jumps=True, min_limit_margin=0.02)
    assert why == "ok" and q_end is not None
    assert [c["n"] for c in kit.backend.calls] == [3, 6] and kit.backend.calls[1]["max_joint_step"] == IKChecker.APPROACH_MAX_STEP
    # off (today's tracking): the jump fails the chain
    kit = _Kit([jump, ramp(-0.95)])
    q_end, why = IKChecker(kit).track_chain(_q_from(kit), _chain(), "right")
    assert q_end is None and why == "approach_untrackable: chain tracking (3 dense poses): joint jump 0.24 rad at waypoint 3"
    assert [c["n"] for c in kit.backend.calls] == [3]
    # a jump at half spacing too: the first tracking's reason stands
    kit = _Kit([jump, jump])
    q_end, why = IKChecker(kit).track_chain(_q_from(kit), _chain(), "right", refine_jumps=True)
    assert q_end is None and "joint jump 0.24 rad" in why and [c["n"] for c in kit.backend.calls] == [3, 6]
