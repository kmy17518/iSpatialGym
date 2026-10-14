"""Simulator-free tests of the large-scale IORS touching mode (owner 2026-10-12 ~16:05, touching_large#03; spec §5.6 "Touching
(large-scale)"): the goal / drop geometry at the partner's face, the face probe's verdict, the mode adoption, the push chain's
per-yaw goal, the ``push_touching`` records, the oracle program and the push-stop facts
(``pytest ispatialgym/tests/test_iors_large_touching.py``)."""

from __future__ import annotations

import copy
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle import sidecar as SC
from ispatialgym.oracle.programs import expand, program_iors_large
from ispatialgym.oracle.sidecar import Sidecar, SidecarError
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2.stances import EdgeFrame, SidecarDraft, Stance

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")
TM = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large_touching")

SUBTASK = "inter_object_relation_satisfaction-touching_large_scale"
FIXTURE = os.path.join(os.path.dirname(__file__), "data", "sidecars", f"{SUBTASK}.json")
Q0 = (0.0, 0.0, 0.0, 1.0)


def _rect(w: float, d: float) -> th.Tensor:
    """Hull points of a w x d x 0.07 box about its root (a tray)."""
    return th.tensor([[sx * w / 2, sy * d / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, 0.07)], dtype=th.float32)


# ----------------------------------------------------------------------------------------------------------------
# geometry at the face
# ----------------------------------------------------------------------------------------------------------------


def test_goal_and_drop_cells_at_the_face():
    hull = _rect(0.29, 0.31)
    back, front = TM.extent_along(hull, (0.0, 1.0))
    assert back == pytest.approx(0.155) and front == pytest.approx(0.155)
    q45 = G.quat_from_yaw(math.pi / 4)
    _, front45 = TM.extent_along(G.local_points_world(hull, (th.zeros(3), q45))[:, :2], (0.0, 1.0))
    assert front45 == pytest.approx((0.145 + 0.155) / math.sqrt(2), abs=1e-4)  # a turned tray reaches farther ahead
    # the bookcase face at y = -0.781, the push toward +y: the goal root 2 mm + the front short of it, the drop half + 0.10 back
    face, d = (-4.053, -0.781), (0.0, 1.0)
    g = TM.touching_goal_xy(face, d, front, 0.002)
    assert g.tolist() == pytest.approx([-4.053, -0.781 - 0.157], abs=1e-6)
    c = TM.touching_drop_xy(face, d, 0.155, 0.10)
    assert c.tolist() == pytest.approx([-4.053, -0.781 - 0.255], abs=1e-6)
    assert float((g - c) @ TM.unit_xy(d)) == pytest.approx(0.098, abs=1e-6)  # the push: ~10 cm toward the face
    assert TM.TOUCHING_GAP_M <= TM.TOUCHING_GAP_MAX_M == 0.005


def test_face_from_hits():
    # (offset behind the AABB edge, on the partner, name)
    f, why = TM.face_from_hits([(0.004, True, None), (0.0, True, None), (None, False, None), (0.28, True, None)])
    assert f == pytest.approx(0.0) and why == ""  # the nearest partner surface is what the target meets first
    f, why = TM.face_from_hits([(None, False, None)] * 3)
    assert f is None and why.startswith("no_face")  # nothing at the target's height: it would pass under the partner
    f, why = TM.face_from_hits([(0.01, True, None), (-0.12, False, "chair_x_0")])
    assert f is None and why.startswith("face_blocked") and "chair_x_0" in why
    f, why = TM.face_from_hits([(0.01, True, None), (0.31, False, "wall_x_0")])
    assert f == pytest.approx(0.01)  # a hit behind the face does not block it
    f, why = TM.face_from_hits([(0.25, True, None)])
    assert f is None and why.startswith("face_recessed")  # a cavity the target would enter
    assert TM.face_from_hits([(0.25, True, None)], max_inset=0.30)[0] == pytest.approx(0.25)


def test_lanes_and_footprint_overlap():
    assert TM.centred_lanes(0.205) == [0.0, 0.1, -0.1, 0.2, -0.2]  # otwukr_1: 0.76 m face, 0.155 half extent, 2 cm end margin
    assert TM.centred_lanes(0.05) == [0.0] and TM.centred_lanes(-0.01) == []
    lo, hi = (0.0, 0.0), (1.0, 0.3)
    assert TM.square_hits_box((0.5, -0.2), 0.155, lo, hi) is False
    assert TM.square_hits_box((0.5, -0.15), 0.155, lo, hi) is True
    assert TM.square_hits_box((0.5, -0.158), 0.155, lo, hi, margin=0.005) is True


def test_push_touching_record():
    q = th.tensor(Q0)
    pre = (th.tensor([0.0, -1.2, 0.15]), q)
    rec = IL.PushUnderRecord("right", pre, (th.tensor([0.0, -1.2, 0.05]), q), (th.tensor([0.0, -1.15, 0.05]), q), (th.tensor([0.0, -1.1, 0.05]), q),
                             th.tensor([0.0, 1.0, 0.0]), 0.1, "storage_box.n.01_1", stance="push", pitch_deg=30.0, blade_standoff_m=0.01,
                             half_extent_m=0.15, drop_cell=[0.0, -1.03], goal_cell=[0.0, -0.94], partner="bookcase.n.01_1")
    t = TM.PushTouchingRecord.from_push_under(rec, goal_cell=[0.0, -0.935], relation="touching", face_point=[0.0, -0.78], gap_m=0.002)
    d = t.to_sidecar()
    assert d["kind"] == TM.PUSH_TOUCHING_KIND == "push_touching" and d["relation"] == "touching" and d["partner"] == "bookcase.n.01_1"
    assert d["goal_cell"] == [0.0, -0.935] and d["face_point"] == [0.0, -0.78] and d["drop_cell"] == [0.0, -1.03] and d["stance"] == "push"
    assert t.kind == "push" and t.arm == "right" and t.object == "storage_box.n.01_1" and t.distance_m == pytest.approx(0.1)
    assert rec.extras["goal_cell"] == [0.0, -0.94]  # the push-under record is not changed
    pc = SC.PushContact.from_dict(d, "contacts.push")
    assert pc.kind == "push_touching" and pc.extra["partner"] == "bookcase.n.01_1" and len(pc.waypoints) == 3


# ----------------------------------------------------------------------------------------------------------------
# the mode in IORSLarge
# ----------------------------------------------------------------------------------------------------------------


class FloorBound(Exception):
    pass


def _bare(relation: str = "touching"):
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.relation, s.template_facts, s.template_meta = relation, {}, {}
    return s


def test_touching_hook_adopts_the_mode(monkeypatch):
    def _floor_bound(self):
        raise FloorBound(self.relation)

    monkeypatch.setattr(IL.IORSLarge, "bind_floor_v2", _floor_bound)
    assert IL.IORSLarge._sample_touching(_bare()) is TM.TouchingMode
    s = _bare()
    with pytest.raises(FloorBound):
        s.bind_scene_roles()  # adopted, then IORSLarge's own binding (the ontop / under flow with the mode's overrides)
    assert s.relation_mode is TM.TouchingMode and isinstance(s, TM.TouchingMode) and isinstance(s, IL.IORSLarge)
    assert s.push_under is True  # the push-under drop + floor push, whatever generation.under_mode says
    assert type(_bare()) is IL.IORSLarge
    # the floor mesh leaves the checks for any init surface (IORSLarge keys it on the under relation / a floor init)
    s.ikc, s.obj, s.surface = SimpleNamespace(ignoring=lambda objs: ("ignoring", list(objs))), (lambda role: f"{role}_obj"), {"kind": "support"}
    assert s._floor_relaxed() == ("ignoring", ["floor_obj"])
    # generation.init_support_fallback: the floor init (not the partner's own top, under's fallback)
    s.gp, s.support_drops = _GP(init_support_fallback=True), {"coffee_table_osroux_0": "geodesic 4.10 m > 3.00 m"}
    assert s.init_support_fallback("support") == {"from": "support", "to": "floor", "reason": "coffee_table_osroux_0: geodesic 4.10 m > 3.00 m"}
    assert s.init_support_fallback("floor") is None and _mode().init_support_fallback("support") is None


class _GP:
    def __init__(self, **gen):
        self.variant, self.generation = f"{SUBTASK}-storage_box-bookcase-coffee_table", dict(gen)

    def gen(self, key, default=None):
        return self.generation.get(key, default)


def _mode(**gen):
    s = _bare()
    s._adopt_mode(TM.TouchingMode)
    s.gp = _GP(**gen)
    s.hull = _rect(0.29, 0.31)
    s.rest, s.floor_z, s.seg_arm = G.IDENTITY_QUAT.clone(), 0.0, "left"
    s.place = {"height": 0.07, "half": 0.155, "faces": {}, "chain_failed": set()}
    return s


#: bookcase_otwukr_1's -y AABB edge (house_double_floor_upper), the face 6 mm behind it (touching#03)
EDGE = EdgeFrame("-y", th.tensor([-4.053, -0.817]), th.tensor([0.0, 1.0]), 0.769)
FACE_OFFSET = 0.006


def _stance_c(s, so: float, lane: float = -0.2, trunk=(2.0, 0.0)) -> Stance:
    """A frontal touching stance C ``so`` m off the edge, the lane 0.25 m to the left arm's side, with the mode's extras."""
    lat_b = IL.frontal_lane_lateral(lane, 0.25, "left")
    xy = EDGE.base_xy(so, lat_b)
    st = Stance("push", (float(xy[0]), float(xy[1]), EDGE.yaw), trunk[0], trunk[1], so, EDGE, None, lat_b, None, ["left"])
    face = EDGE.mid + EDGE.tangent * lane + EDGE.inward * FACE_OFFSET
    drop = TM.touching_drop_xy(face, EDGE.inward, 0.155, 0.10)
    st.extras.update({"face_point": [float(v) for v in face], "push_dir": [0.0, 1.0], "drop_cell": [float(v) for v in drop]})
    return st


def test_goal_per_yaw_and_the_push_chain(monkeypatch):
    s = _mode()
    face, d2 = [-4.053, -0.781], [0.0, 1.0]
    g0 = s.touching_goal_at(face, d2, th.tensor(Q0))
    assert g0.tolist() == pytest.approx([-4.053, -0.781 - 0.155 - TM.TOUCHING_GAP_M], abs=1e-5)
    g45 = s.touching_goal_at(face, d2, G.quat_from_yaw(math.pi / 4))
    assert float(g0[1] - g45[1]) == pytest.approx((0.145 + 0.155) / math.sqrt(2) - 0.155, abs=1e-4)
    assert _mode(touching_goal_gap_m=0.02).touching_gap == TM.TOUCHING_GAP_MAX_M  # clamped to the owner's 0-5 mm
    seen = []

    def chain(self, place, quat, drop_xy, goal_xy, d2_):
        seen.append([float(v) for v in goal_xy])
        return {"distance_m": 0.1}

    monkeypatch.setattr(IL.IORSLarge, "_push_under_chain", chain)
    place = _stance_c(s, 0.90)
    face, drop = place.extras["face_point"], place.extras["drop_cell"]
    g0 = s.touching_goal_at(face, d2, th.tensor(Q0))
    ch = s._push_under_chain(place, th.tensor(Q0), drop, [9.0, 9.0], d2)
    assert seen[-1] == pytest.approx(g0.tolist(), abs=1e-6) and ch["goal_cell"] == pytest.approx(g0.tolist(), abs=1e-6)  # the nominal goal is replaced
    assert ch["blade_span_m"] == pytest.approx([0.472, 0.60], abs=0.002)  # the blade 0.47 -> 0.60 m ahead: inside the floor reach
    assert s._push_under_chain(place, th.tensor(Q0), g0 + th.tensor([0.0, 0.005]), [9.0, 9.0], d2) is None  # nothing left to push
    assert s._push_under_chain(SimpleNamespace(extras={}), th.tensor(Q0), drop, [9.0, 9.0], d2) == {"distance_m": 0.1}  # no face: as recorded
    assert seen[-1] == [9.0, 9.0]


def test_push_chain_skips_a_blade_off_the_floor_reach_and_a_chain_that_failed(monkeypatch):
    """touching#03 eval 2026-10-12: from the push-under's stand-offs (0.75 / 0.80 m) the tray's blade started 0.27-0.32 m ahead of
    the base and all 252-498 chains of each attempt failed their pre-pose; such a yaw is not solved, and a chain that failed is
    not solved again from the same stance at the same orientation (the next B candidate reaches the same C)."""
    s = _mode()
    calls = []

    def chain(self, place, quat, drop_xy, goal_xy, d2_):
        calls.append((place.stand_off, [round(float(v), 4) for v in G.t(quat).tolist()]))
        return None

    monkeypatch.setattr(IL.IORSLarge, "_push_under_chain", chain)
    q45, q0 = G.quat_from_yaw(math.pi / 4), th.tensor(Q0)
    for so in (0.75, 0.80):  # the eval's C: the corner-first yaw (the push-under's largest offset) and the aligned one
        st = _stance_c(s, so)
        for q in (q45, q0):
            span = s.blade_span_at(st, q, st.extras["drop_cell"], s.touching_goal_at(st.extras["face_point"], [0.0, 1.0], q), [0.0, 1.0])
            assert span[0] < TM.BLADE_AHEAD_M[0] - 0.02  # on the base rim
        assert s._push_under_chain(st, q45, st.extras["drop_cell"], [9.0, 9.0], [0.0, 1.0]) is None
    assert calls == []  # skipped without an IK
    st = _stance_c(s, 0.90)
    assert s._push_under_chain(st, q45, st.extras["drop_cell"], [9.0, 9.0], [0.0, 1.0]) is None and len(calls) == 1  # in reach: solved (and failed)
    assert s._push_under_chain(st, q45, st.extras["drop_cell"], [9.0, 9.0], [0.0, 1.0]) is None and len(calls) == 1  # failed before: not again
    assert s._push_under_chain(st.with_trunk(0.95, -1.2, "push"), q45, st.extras["drop_cell"], [9.0, 9.0], [0.0, 1.0]) is None and len(calls) == 2  # another trunk
    assert s._push_under_chain(st, q0, st.extras["drop_cell"], [9.0, 9.0], [0.0, 1.0]) is None and len(calls) == 3  # another orientation
    s.seg_arm = "right"
    assert s._push_under_chain(st, q45, st.extras["drop_cell"], [9.0, 9.0], [0.0, 1.0]) is None and len(calls) == 4  # another arm
    assert _mode(touching_blade_ahead_m=[0.25, 0.85]).blade_window == (0.25, 0.85)


def test_blade_span_and_the_frontal_stand_offs():
    # the chain's geometry ahead of a frontal base: start = drop - back - stand-off; goal = face - gap - front - back - stand-off + 3 cm
    assert TM.blade_span(0.65, 0.906, 0.155, 0.155, 0.023, 0.002, 0.03) == pytest.approx((0.472, 0.601))
    assert TM.span_violation((0.472, 0.601), (0.40, 0.85)) == 0.0 and TM.span_violation((0.315, 0.385), (0.40, 0.85)) == pytest.approx(0.085)
    assert TM.span_violation((0.60, 0.90), (0.40, 0.85)) == pytest.approx(0.05)
    inside = sorted([(0.415, 0.485), (0.472, 0.601), (0.30, 0.40)], key=lambda sp: TM.blade_rank(sp, (0.40, 0.85)))
    assert inside == [(0.472, 0.601), (0.415, 0.485), (0.30, 0.40)]  # in reach, middle nearest 0.55 first; off the reach last
    s = _mode()
    assert s.blade_standoff_m() == pytest.approx(0.018 + IL.PUSH_UNDER_CLEARANCE_M)  # the closed finger's upper section: the tray's back face reaches it
    prof = s.blade_profile()
    assert len(prof) == TM.PROFILE_YAWS and min(b for b, _ in prof) == pytest.approx(0.145, abs=1e-4)
    assert max(b for b, _ in prof) == pytest.approx(math.hypot(0.145, 0.155), abs=2e-3)  # corner-first
    kw = dict(face_offset=FACE_OFFSET, half=0.155, extra=0.10, profile=prof, standoff=s.blade_standoff_m(), gap=0.002, goal_extra=0.03,
              window=TM.BLADE_AHEAD_M, drop_max=TM.DROP_AHEAD_MAX_M)
    fits = [so for so in TM.STAND_OFFS_M if TM.stand_off_fits(so, **kw)]
    assert fits == [0.85, 0.9, 0.95, 1.0]  # the push-under's 0.5-0.8 m put every blade start on the base rim; 1.05+ drops > 0.80 m ahead
    # the C order: the nearest drop from which every yaw's blade start clears 0.41 m (stand-off 0.90 m)
    assert s.push_drop_ahead_pref() == pytest.approx(0.41 + math.hypot(0.145, 0.155) + 0.023, abs=2e-3)
    assert _mode(push_under_drop_ahead_pref_m=0.5).push_drop_ahead_pref() == 0.5


def test_touching_yaw_rank_puts_the_blade_in_reach_first():
    s = _mode()
    q0, q45 = th.tensor(Q0), G.quat_from_yaw(math.pi / 4)
    assert IL.IORSLarge.push_yaw_rank(s, None, q0, 0.21) == -0.21  # push-under: the largest contact offset first
    assert IL.IORSLarge.PUSH_YAW_ORDER == "largest-offset" and s.PUSH_YAW_ORDER == "blade-window"
    st = _stance_c(s, 0.90)
    assert s.push_yaw_rank(st, q0, 0.155) < s.push_yaw_rank(st, q45, 0.213) < 1.0  # both in reach, the aligned blade nearer the middle
    st = _stance_c(s, 0.85)
    assert s.push_yaw_rank(st, q0, 0.155) < 1.0 < s.push_yaw_rank(st, q45, 0.213)  # the corner-first blade starts on the base rim
    assert s.push_yaw_rank(Stance("push", (0.0, 0.0, 0.0), 2.0, 0.0), q45, 0.213) == -0.213  # no face point: the push-under rule
    assert IL.IORSLarge.push_drop_ahead_pref(_mode()) is None and IL.IORSLarge.push_drop_ahead_pref(_mode(push_under_drop_ahead_pref_m=0.3)) == 0.3


def test_touching_records_turn_the_push_under_records(monkeypatch):
    s = _mode()
    s.sidecar = SidecarDraft(s.gp.variant, "inter_object_relation_satisfaction")
    names = {"target": "storage_box.n.01_1", "partner": "bookcase.n.01_1", "floor": "floor.n.01_1"}
    s.bindings = SimpleNamespace(one=lambda r: SimpleNamespace(name=names[r]), has=lambda r: r in names)
    face, d2 = [-4.053, -0.781], [0.0, 1.0]
    place = Stance("placement", (-4.053, -1.5, math.pi / 2), 0.75, -0.9, 0.7, None, None, 0.0, None, ["right"])
    place.extras.update({"face_point": face, "push_dir": d2, "face_offset_m": 0.0, "gap_m": 0.002, "edge": "-y", "goal_cell": [-4.053, -0.938]})
    s.sidecar.add_stance(place)
    q = th.tensor(Q0)
    s.sidecar.add_contact(IL.PushUnderRecord("right", (th.tensor([-4.053, -1.3, 0.15]), q), (th.tensor([-4.053, -1.2, 0.05]), q), (th.tensor([-4.053, -1.15, 0.05]), q),
                                             (th.tensor([-4.053, -1.1, 0.05]), q), th.tensor([0.0, 1.0, 0.0]), 0.098, names["target"], stance="push",
                                             half_extent_m=0.155, blade_standoff_m=0.01, drop_cell=[-4.053, -1.036], goal_cell=[-4.053, -0.938], partner=names["partner"]))
    from ispatialgym.sampling.v2.stances import GraspRecord

    s.sidecar.add_grasp(GraspRecord(names["target"], "right", "storage_box__eicujj__t", "g0", 0.0, extras={"then": "push_under", "goal_cell": [-4.053, -0.938]}))
    d = Draft()
    d.summary.update({"goal_cell": [-4.053, -0.938], "goal_quat": list(G.quat_from_yaw(math.pi / 2).tolist()), "push_under": {"pitch_deg": 30.0}})
    s.touching_records(d)
    rec = s.sidecar.contacts["push"]
    assert isinstance(rec, TM.PushTouchingRecord) and rec.to_sidecar()["kind"] == "push_touching"
    g = th.tensor(d.summary["goal_cell"])
    # turned by 90 deg the tray's 0.29 side faces the bookcase: the goal root 0.145 + 2 mm short of the face
    assert g.tolist() == pytest.approx([-4.053, -0.781 - 0.145 - 0.002], abs=1e-4)
    assert rec.extras["goal_cell"] == pytest.approx(g.tolist(), abs=1e-6) and rec.extras["relation"] == "touching" and rec.extras["partner"] == names["partner"]
    assert rec.distance_m == pytest.approx(float(g[1]) + 1.036, abs=1e-4)
    assert s.sidecar.grasps[0].extras["then"] == "push_touching" and s.sidecar.grasps[0].extras["goal_cell"] == pytest.approx(g.tolist(), abs=1e-6)
    assert "push_under" not in d.summary and d.summary["push_touching"]["pitch_deg"] == 30.0 and d.summary["push_touching"]["edge"] == "-y"
    assert s.sidecar.language["goal_numbers"]["push_distance_cm"] == pytest.approx(100 * rec.distance_m)
    s.sidecar.contacts.pop("push")
    with pytest.raises(Reject):
        s.touching_records(Draft())


# ----------------------------------------------------------------------------------------------------------------
# the oracle program and the push-stop facts
# ----------------------------------------------------------------------------------------------------------------


def test_touching_large_program_chain():
    sc = Sidecar.load(FIXTURE)
    assert sc.problems() == [] and sc.contacts.push.kind == "push_touching"
    actions = expand(sc.variant, sc)
    assert [a.name for a in actions] == ["transition", "grasp", "lift", "navigate", "place", "retreat", "transition", "push", "retreat"]
    place, ret1, to_push, push, ret2 = actions[4:]
    assert place.kind == "drop" and place.on == "floor.n.01_1" and place.avoid == () and ret1.after is place
    assert to_push.stance_from.translate != to_push.stance_to.translate  # the deep push crouch entered with empty hands
    assert push.kind == "push_touching" and push.touches == ("floor.n.01_1", "bookcase.n.01_1") and push.avoid == () and push.follow == "bookcase.n.01_1"
    assert push.arms == ("right",) and push.objects == ("storage_box.n.01_1",) and ret2.after is push
    assert "along the floor until it touches" in push.annotate(sc)[0] and "let it drop" in place.annotate(sc)[0]
    with open(FIXTURE) as f:
        d = json.load(f)
    for mutate, word in ((lambda x: x["contacts"]["push"].update(kind="push_under"), "push_touching"), (lambda x: x["contacts"]["push"].pop("partner"), "partner"),
                         (lambda x: x["grasps"][0].pop("place_kind"), "place_kind"), (lambda x: x["contacts"]["push"].update(arm="left"), "arm")):
        d2 = copy.deepcopy(d)
        mutate(d2)
        assert any(word in p for p in Sidecar.from_dict(d2).problems()), word
    with pytest.raises(SidecarError, match="push_touching"):
        d3 = copy.deepcopy(d)
        d3["contacts"]["push"]["kind"] = "push"
        program_iors_large(Sidecar.from_dict(d3), relation="touching")


def test_push_stop_facts():
    obj = (th.tensor([0.0, -0.94, 0.04]), th.tensor(Q0))
    poses = {"bookcase.n.01_1": (th.tensor([0.0, -0.629, 0.531]), th.tensor(Q0))}
    state = SimpleNamespace(object_pose=lambda n: poses[n], init_pose=lambda n: (th.tensor([0.0, -0.631, 0.531]), th.tensor(Q0)))
    action = SimpleNamespace(contact=SimpleNamespace(object="storage_box.n.01_1"), follow="bookcase.n.01_1")
    rec = EX.push_stop_facts(action, state, {"push_stopped": [{"leg": "push", "steps": 40, "residual": 0.0}]}, obj[0], 0.001)
    assert rec["stop"] == "goal_reached" and rec["legs_stopped"][0]["steps"] == 40 and rec["residual_final"] == 0.001
    assert rec["object_position"] == pytest.approx([0.0, -0.94, 0.04]) and rec["follow"] == {"name": "bookcase.n.01_1", "moved_m": 0.002, "moved_z_m": 0.0}
    rec = EX.push_stop_facts(SimpleNamespace(contact=action.contact, follow=None), state, {}, obj[0], None)
    assert rec["stop"] == "leg_end" and "follow" not in rec and rec["residual_final"] is None
