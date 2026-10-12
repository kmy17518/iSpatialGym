"""IORS small carry height (owner 2026-10-12 item 4A) and the push stance's lift (item 7d): touching's carry clears only the table
("auto": unless the carried drive to the push stance takes the box over the partner), the other relations clear the partner's top as
before (``generation.carry_clears`` / ``carry_clearance_m``), the touching place gap knob (``touching_place_gap_m``), and a lift
without a configuration at the push stance rejected as ``lift_unreachable`` instead of being swept with the untucked hand
(``path_collision``). ``IORSSmall`` methods run on stand-ins; no simulator is launched."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.ik import IKResult
from ispatialgym.sampling.v2.stances import EdgeFrame, GraspRecord, Stance
from ispatialgym.sampling.v2.tasks import iors_small as IS

#: touching#02's boxes: the target box_of_cereal/iucshm 5.3 x 19.6 x 27.3 cm, the partner box_of_cereal/dkhgxn 7.8 x 15.5 x 24.2 cm
TARGET = (0.0534, 0.1958, 0.2727)
PARTNER = (0.0784, 0.1548, 0.2417)


def _box(dx, dy, h):
    return th.tensor([[sx * dx / 2, sy * dy / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, h)], dtype=th.float32)


def _world():
    return SimpleNamespace(footprint=lambda h, pose: G.convex_hull_2d(G.local_points_world(h, pose)[:, :2]), hull_at=G.local_points_world)


def _sampler(relation="touching", gen=None, **extra):
    """An ``IORSSmall`` without a simulator: the two touching#02 boxes standing upright, the generation of ``gen``."""
    s = IS.IORSSmall.__new__(IS.IORSSmall)
    s.relation, s.gp, s.spec = relation, SimpleNamespace(alias="a"), SimpleNamespace(generation=dict(gen or {}))
    s.hull = {"target": _box(*TARGET), "partner": _box(*PARTNER)}
    s.rest = {"target": C.IDENTITY_QUAT.clone(), "partner": C.IDENTITY_QUAT.clone()}
    s.pair, s.shelf_start, s.world = {"g_NT": 0.0554}, None, _world()
    for k, v in extra.items():
        setattr(s, k, v)
    return s


def _edge():
    """The robot faces +x at the edge x = 0.40; its left (the edge tangent, the partner's side) is +y."""
    return EdgeFrame("-x", th.tensor([0.40, 0.0]), th.tensor([1.0, 0.0]), 1.2)


def _stance(name="manipulation", lateral=0.0):
    st = Stance(name, (0.0, float(lateral), 0.0), 0.25, -0.5, 0.40, _edge(), lateral=float(lateral))
    st.arms = ["right"]
    return st


def _pose(x, y, z=0.81):
    return th.tensor([float(x), float(y), float(z)]), C.IDENTITY_QUAT.clone()


# ---------------------------------------------------------------------------------------------------------------- the rule and the heights


@pytest.mark.parametrize("relation", ["ontop", "under", "inside", "nextto"])
def test_the_other_relations_clear_the_partner_top_as_before(relation):
    s = _sampler(relation)
    assert s.carry_rule() == "partner"
    h = TT.height_of(s.hull["partner"], s.rest["partner"])
    old = max(0.03, h + 0.03)  # the expression the carry height had before 4A
    assert s.carry_height_over(True) == pytest.approx(old) and s.carry_height_over(True) == pytest.approx(PARTNER[2] + 0.03, abs=1e-5)
    s2 = _sampler(relation, gen={"carry_clearance_m": 0.05})
    assert s2.carry_height_over(True) == pytest.approx(PARTNER[2] + 0.05, abs=1e-5)


def test_touching_clears_only_the_table_by_default_and_the_keys_override_it():
    s = _sampler("touching")
    assert s.carry_rule() == "auto"
    assert s.carry_height_over(False) == pytest.approx(0.03)  # the box's bottom 3 cm above the table, not 0.27 m
    assert _sampler("touching", gen={"carry_clearance_m": 0.04}).carry_height_over(False) == pytest.approx(0.04)
    assert _sampler("touching", gen={"carry_clears": "table"}).carry_rule() == "table"
    assert _sampler("touching", gen={"carry_clears": "partner"}).carry_rule() == "partner"
    assert _sampler("nextto", gen={"carry_clears": "table"}).carry_rule() == "table"  # opt-in for another relation
    with pytest.raises(Reject) as e:
        _sampler("touching", gen={"carry_clears": "high"}).carry_rule()
    assert e.value.reason == "spec" and "carry_clears" in str(e.value)


def test_the_touching_place_gap_knob_moves_the_place_and_lengthens_the_push():
    s = _sampler("touching")
    assert s.place_gap() == pytest.approx(0.0454, abs=1e-6)  # g_NT - 1 cm, as before
    wide = _sampler("touching", gen={"touching_place_gap_m": 0.07})
    assert wide.place_gap() == pytest.approx(0.07)
    assert wide.push_distance() == pytest.approx(0.07 + 0.03 + 0.005)  # the push covers the longer gap + the margin
    assert _sampler("nextto", gen={"touching_place_gap_m": 0.07}).place_gap() == pytest.approx(0.0454, abs=1e-6)  # touching only


# ---------------------------------------------------------------------------------------------------------------- "auto": the carried drive


def test_the_drive_gap_is_the_start_gap_minus_the_base_shift():
    s = _sampler("touching")
    t_pose = _pose(0.70, 0.0)
    p_pose = _pose(0.70, TARGET[1] / 2 + 0.10 + PARTNER[1] / 2)  # start gap 10 cm along +y (the robot's left)
    manip = _stance()
    assert s.drive_partner_gap(t_pose, p_pose, manip, manip) == pytest.approx(0.10, abs=1e-4)
    assert s.drive_partner_gap(t_pose, p_pose, manip, _stance("push", 0.06)) == pytest.approx(0.04, abs=1e-4)
    assert s.drive_partner_gap(t_pose, p_pose, manip, _stance("push", 0.142)) == 0.0  # c2_k0 attempt 0's shift: into the partner
    assert s.drive_partner_gap(t_pose, p_pose, manip, _stance("push", -0.05)) == pytest.approx(0.10, abs=1e-4)  # away from it


def _grasped(z=0.95):
    return GraspRecord("box.n.01_1", "right", "k", "grasp_0", 0.0, grasp=_pose(0.70, 0.0, z), eef_in_object=_pose(0.0, 0.0, 0.14))


@pytest.mark.parametrize("shift, clears, height", [(0.04, "table", 0.03), (0.142, "partner", PARTNER[2] + 0.03)])
def test_auto_raises_the_carry_only_when_the_drive_passes_over_the_partner(shift, clears, height):
    s = _sampler("touching", bindings=SimpleNamespace(one=lambda role: SimpleNamespace(name="box.n.01_1")))
    d = Draft()
    d.object_poses["box.n.01_1"] = _pose(0.70, 0.0)
    g = _grasped()
    g.carry_height = s.carry_height_over(False)
    s.auto_carry_height(d, g, _stance(), _stance("push", shift), _pose(0.70, TARGET[1] / 2 + 0.10 + PARTNER[1] / 2))
    assert g.carry_height == pytest.approx(height, abs=1e-5)
    assert d.summary["carry_lift"]["rule"] == "auto" and d.summary["carry_lift"]["clears"] == clears


# ---------------------------------------------------------------------------------------------------------------- 7d: the push stance's lift


class _Obj:
    def __init__(self, pose):
        self.pose = pose

    def get_position_orientation(self):
        return self.pose

    def set_position_orientation(self, p, q):
        self.pose = (p, q)

    def keep_still(self):
        pass


def _push_sampler(monkeypatch, lift_ok: bool, shift: float, rule=None):
    """``IORSSmall.push_stance`` on a stand-in: the push class / place pass at the first lateral, the push stance ``shift`` along the edge,
    the lift IK answering ``lift_ok`` (the lift end it was asked for recorded), ``record_path`` recorded."""
    monkeypatch.setattr(TT, "whole_body_check_at", lambda *a, **k: [True])
    monkeypatch.setattr(IS, "attached", lambda *a, **k: contextlib.nullcontext())
    calls = {"lift_end": [], "paths": []}
    target, partner = _Obj(_pose(0.70, 0.0)), _Obj(_pose(0.70, TARGET[1] / 2 + 0.10 + PARTNER[1] / 2))

    def exact(poses, st, arm, **kw):
        calls["lift_end"].append(poses[0])
        return IKResult(True, th.arange(28, dtype=th.float32), "") if lift_ok else IKResult(False, None, "exact IK failed at pose 1 of 1")

    def record_path(frm, to, **kw):
        calls["paths"].append(kw)
        return SimpleNamespace(length_m=abs(shift))

    def push_stance_for(manip, t_place, lateral=None):
        st = _stance("push", shift)
        st.extras["lateral_shift_m"] = shift
        return st

    s = _sampler("touching", gen={} if rule is None else {"carry_clears": rule},
                 bindings=SimpleNamespace(one=lambda role: SimpleNamespace(name="box.n.01_1" if role == "target" else "box.n.01_2")),
                 objects={"box.n.01_1": target, "box.n.01_2": partner}, _brace_gate={"ok": False, "why": "test"},
                 _touching_place_pos=th.tensor([0.70, 0.04, 0.81]), sidecar=SimpleNamespace(add_stance=lambda st: None, set_goal=lambda *a, **k: None),
                 kit=SimpleNamespace(sync_world=lambda: None, robot=SimpleNamespace(arm_control_idx={"right": list(range(0, 7)), "left": list(range(7, 14))})),
                 ikc=SimpleNamespace(exact=exact), support_objects=lambda: [], record_path=record_path, brace_gate_at=lambda *a: {"ok": False},
                 push_stance_for=push_stance_for, push_chain_feasible=lambda *a, **k: "push_tip_h-",
                 place_infeasible_why=lambda *a: None)
    d = Draft()
    d.object_poses["box.n.01_1"] = target.pose
    d.goal_poses["box.n.01_1"] = _pose(0.70, 0.0)
    g = _grasped()
    g.carry_height = s.carry_height_over(s.carry_rule() == "partner")
    return s, d, g, calls


def test_a_lift_without_configuration_at_the_push_stance_is_lift_unreachable_not_path_collision(monkeypatch):
    s, d, g, calls = _push_sampler(monkeypatch, lift_ok=False, shift=0.04)
    with pytest.raises(Reject) as e:
        s.push_stance(d, _stance(), g)
    assert e.value.reason == "lift_unreachable" and "carried drive to the push stance" in str(e.value)
    assert calls["paths"] == []  # never swept with the untucked hand


def test_the_drive_is_swept_at_the_lift_configuration_of_the_auto_carry_height(monkeypatch):
    s, d, g, calls = _push_sampler(monkeypatch, lift_ok=True, shift=0.04)
    st = s.push_stance(d, _stance(), g)
    assert st is not None and st.carried == "box.n.01_1"
    assert float(calls["lift_end"][0][0][2]) == pytest.approx(0.95 + 0.03, abs=1e-5)  # the table clearance: the drive keeps 6 cm from the partner
    arm_q = calls["paths"][0]["arm_q"]
    assert th.equal(arm_q["right"], th.arange(0, 7, dtype=th.float32)) and th.equal(arm_q["left"], th.arange(7, 14, dtype=th.float32))
    # a shift beyond the start gap takes the box over the partner: the lift clears its top
    s, d, g, calls = _push_sampler(monkeypatch, lift_ok=True, shift=0.142)
    s.push_stance(d, _stance(), g)
    assert float(calls["lift_end"][0][0][2]) == pytest.approx(0.95 + PARTNER[2] + 0.03, abs=1e-5) and d.summary["carry_lift"]["clears"] == "partner"
    # carry_clears "table" keeps the table clearance whatever the drive (the path check then judges the box against the partner)
    s, d, g, calls = _push_sampler(monkeypatch, lift_ok=True, shift=0.142, rule="table")
    s.push_stance(d, _stance(), g)
    assert float(calls["lift_end"][0][0][2]) == pytest.approx(0.95 + 0.03, abs=1e-5) and "carry_lift" not in d.summary
