"""Simulator-free test of the IORS large-scale A -> B leg's torso-then-navigate fallback (oracle-fix 2026-11-06, survey1 under_large#02:
'collision | navigate | right_realsense_link in contact with walls_mrghjk_0' in 3 / 3 demos): the path swept with B's arms records them
as ``paths[].arms_q`` (what the oracle's navigate keeps an empty arm at), as ``core.record_edge_travel`` does
(``pytest ispatialgym/tests/test_iors_large_leg_ab_arms.py``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as th

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")

from ispatialgym.sampling.sim_utils import Reject  # noqa: E402
from ispatialgym.sampling.v2 import ik as IK  # noqa: E402
from ispatialgym.sampling.v2.stances import PathRecord, Transition  # noqa: E402

R_READY = [-0.073, -0.326, 0.951, -1.639, -0.632, 0.137, -0.926]
L_UNTUCKED = [0.0, 1.57, 0.0, -1.57, 1.57, 0.0, 0.0]


class _Leg:
    """The attributes :meth:`IORSLarge.record_leg_a_b` reads: the navigate-then-torso path collides, the departure and the path at B's
    trunk are recorded."""

    record_leg_a_b = IL.IORSLarge.record_leg_a_b

    def __init__(self):
        self.gp = SimpleNamespace(variant="inter_object_relation_satisfaction-under_large_scale-briefcase-desk_mdhelw_0-floor_pitched")
        self.kit = None
        self.sidecar = SimpleNamespace(paths=[], transitions=[])
        self.path_calls = []

    def record_path(self, frm, to, **kw):
        self.path_calls.append(kw)
        if "arm_q" not in kw:
            raise Reject("path_collision", "local path init -> manipulation collides at swept pose 12/24 at trunk (0.5, 0.0)")
        p = PathRecord(frm.name, to.name, "local", [tuple(frm.base_pose2d), tuple(to.base_pose2d)], 0.758, "untucked", None, kw.get("trunk"))
        self.sidecar.paths.append(p)
        return p

    def record_transition(self, frm, to, **kw):
        raise AssertionError("not reached: the primary path collides first")

    def record_departure_transition(self, frm, to, travel_trunk, **kw):
        t = Transition(frm.name, to.name, 1.2066, trunk_from=(frm.translate, frm.tilt), trunk_to=tuple(travel_trunk), n_samples=8, at="departure")
        self.sidecar.transitions.append(t)
        return t

    def _drop_leg_records(self, to):
        self.sidecar.paths = [p for p in self.sidecar.paths if p.to != to]


def _stance(name, pose, translate, tilt):
    return SimpleNamespace(name=name, base_pose2d=pose, translate=translate, tilt=tilt, stand_off=1.2066, carried=None)


def test_torso_then_navigate_path_records_the_arms_it_was_swept_with(monkeypatch):
    monkeypatch.setattr(IK, "stance_arm_vectors", lambda kit, st, *a, **k: (th.tensor(R_READY), th.tensor(L_UNTUCKED), 0.05))
    monkeypatch.setattr(IL.NAV, "departure_pose", lambda st, s: tuple(st.base_pose2d))
    leg = _Leg()
    init = _stance("init", (18.5437, -11.7985, 2.7517), 0.5, 0.0)
    manip = _stance("manipulation", (17.9759, -11.2962, -3.3382), 0.95, -0.3)
    out = leg.record_leg_a_b(init, manip)
    assert out["order"] == "torso_then_navigate"
    (path,) = leg.sidecar.paths
    assert leg.path_calls[-1]["trunk"] == (0.95, -0.3)
    # the record carries B's arms: the oracle keeps the right arm at 'ready' (no re-pose to untucked), the program check sweeps with them
    assert path.arms_q is not None
    assert path.arms_q["right"] == pytest.approx(R_READY, abs=1e-5) and path.arms_q["left"] == pytest.approx(L_UNTUCKED, abs=1e-5)
    assert path.to_sidecar()["arms_q"]["right"] == pytest.approx(R_READY, abs=1e-5)
