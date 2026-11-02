"""train-fast RECORD_CHANGED_ONLY (2026-11-01): the per-step recording filter keeps BEHAVIOR's rule (awake, initialized) and leaves out an
awake object whose pose and joints have not changed since its last recorded state (``Executor._changed_filter``); stubs."""

from types import SimpleNamespace

import torch as th

from ispatialgym.oracle.executor import Executor


class _Obj:
    def __init__(self, name, pos, active=True, joints=None):
        self.name, self.pos, self.is_active, self.initialized, self.joints = name, list(pos), active, True, joints
        self.n_joints = 0 if joints is None else len(joints)

    def get_position_orientation(self):
        return th.tensor(self.pos), th.tensor([0.0, 0.0, 0.0, 1.0])

    def get_joint_positions(self):
        return th.tensor(self.joints)


def test_unchanged_awake_objects_are_recorded_once_then_left_out():
    flt = Executor._changed_filter(SimpleNamespace(RECORD_CHANGE_TOL=1e-5))
    bush, cup, door = _Obj("bush", [5, 5, 0]), _Obj("cup", [1, 1, 0.8]), _Obj("door", [2, 0, 0], joints=[0.0])
    asleep = _Obj("sofa", [3, 3, 0], active=False)
    assert [flt(o) for o in (bush, cup, door, asleep)] == [True, True, True, False]  # the first state: every awake object
    cup.pos[0] += 0.01
    door.joints[0] += 0.2
    assert [flt(o) for o in (bush, cup, door, asleep)] == [False, True, True, False]
    bush.pos[2] += 2e-6  # sub-tolerance chatter: still left out
    assert flt(bush) is False
    for _ in range(10):
        bush.pos[2] += 2e-6  # a slow creep: recorded once it adds up past the tolerance (against the last RECORDED state)
    assert flt(bush) is True and flt(bush) is False
