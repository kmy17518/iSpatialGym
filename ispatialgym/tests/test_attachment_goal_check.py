"""Attachment.after_goal_check re-solves the attach at the settled partner pose from the stance the attach was recorded at
(owner 2026-10-11 18:30): with a second base pose it solved from the manipulation stance and rejected every such attempt."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace as NS

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.tasks import attachment as AT


class _Attachment(AT.Attachment):
    tname = "cap.n.01_1"
    pname = "bottle.n.01_1"


def _sampler(monkeypatch, attach_stance, reachable_from):
    s = object.__new__(_Attachment)
    manip, second = NS(name="manipulation"), NS(name="attach_2")
    grasp = NS(eef_in_object=(th.zeros(3), C.IDENTITY_QUAT), extras={"attach_stance": attach_stance} if attach_stance else {},
               waypoints=[], pre_place=None, place=None)
    s.t_role, s.p_role = "male", "female"
    s.sidecar = NS(grasp_of=lambda name: grasp, stance=lambda name: manip, stances={"manipulation": manip, "attach_2": second}, extra={})
    s._attached_pose = lambda f_pose: f_pose
    s._insertion_axis_world = lambda f_pose: th.tensor([0.0, 0.0, 1.0])
    s._fingers_in_partner = lambda *a: None
    s.g = lambda key: 0.03 if key == "pre_attach_back_m" else None
    s.gen = lambda key, default=None: default
    s.obj = lambda role: NS(name=role)
    s.choice = NS(models={"male": NS(scale=None)})
    s.kit = NS()
    s.counters = {}
    solved_from = []

    def exact(poses, stance, arm, **kw):
        solved_from.append(stance.name)
        ok = stance.name == reachable_from
        return NS(ok=ok, reason="" if ok else "no IK solution", q=th.zeros(3))

    s.ikc = NS(exact=exact)
    monkeypatch.setattr(AT.TT, "posed_in_hand", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(AT, "attached", lambda *a, **k: contextlib.nullcontext())
    draft = NS(object_poses={s.pname: (th.tensor([0.5, 0.0, 0.8]), C.IDENTITY_QUAT)})
    return s, draft, solved_from, grasp


def test_after_goal_check_solves_from_the_recorded_attach_stance(monkeypatch):
    s, draft, solved_from, grasp = _sampler(monkeypatch, "attach_2", reachable_from="attach_2")
    s.after_goal_check(draft)
    assert solved_from == ["attach_2"] and grasp.place is not None and grasp.pre_place is not None
    # without a second base pose: the manipulation stance
    s, draft, solved_from, _ = _sampler(monkeypatch, None, reachable_from="manipulation")
    s.after_goal_check(draft)
    assert solved_from == ["manipulation"]
    # an attach the recorded stance cannot reach is still rejected
    s, draft, solved_from, _ = _sampler(monkeypatch, "attach_2", reachable_from="manipulation")
    with pytest.raises(Reject):
        s.after_goal_check(draft)
    assert solved_from == ["attach_2"]
