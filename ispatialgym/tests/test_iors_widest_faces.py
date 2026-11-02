"""IORS small touching, widest faces (owner 2026-10-13, touching#02: "aligned means their widest faces point the same way; now they are
perpendicular"): ``generation.widest_faces`` keeps the target's and the partner's widest upright faces parallel at init and at the goal
(the target's init yaws, the partner's yaws, the goal yaw); without the key the draws are today's; ``tabletop.phase3_target`` keeps F over
other yaws than the band's 8 out of the band. ``IORSSmall`` methods run on stand-ins; no simulator is launched."""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.sampling.v2.tasks import iors_small as IS

#: touching#02's boxes (bbox_size, local x the thin axis for both): the target box_of_cereal/iucshm at x0.8, the partner box_of_cereal/dkhgxn
TARGET = (0.8 * 0.0534, 0.8 * 0.1958, 0.8 * 0.2727)
PARTNER = (0.0784, 0.1548, 0.2417)
CUSTOMS = os.path.join(os.path.dirname(__file__), "..", "assets", "task_specs", "isg_customs.json")


def _box(dx, dy, h):
    return th.tensor([[sx * dx / 2, sy * dy / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, h)], dtype=th.float32)


def _world():
    return SimpleNamespace(footprint=lambda h, pose: G.convex_hull_2d(G.local_points_world(h, pose)[:, :2]), hull_at=G.local_points_world)


def _sampler(relation="touching", gen=None, partner=PARTNER, **extra):
    s = IS.IORSSmall.__new__(IS.IORSSmall)
    s.relation, s.gp, s.spec = relation, SimpleNamespace(variant="a"), SimpleNamespace(generation=dict(gen or {}))
    s.hull = {"target": _box(*TARGET), "partner": _box(*partner)}
    s.rest = {"target": C.IDENTITY_QUAT.clone(), "partner": C.IDENTITY_QUAT.clone()}
    s.pair, s.shelf_start, s.world = {"g_NT": 0.0554}, None, _world()
    for k, v in extra.items():
        setattr(s, k, v)
    return s


def _stance(inward=(0.0, 1.0)):
    """c8_k0's frame by default: the robot faces +y at the bar's edge, the edge tangent (its left) points to -x (yaw 180 deg)."""
    edge = EdgeFrame("e", th.tensor([8.2, 0.0]), th.tensor([float(inward[0]), float(inward[1])]), 1.2)
    st = Stance("manipulation", (8.2, -0.75, math.atan2(inward[1], inward[0])), 0.25, -0.5, 0.40, edge)
    st.arms = ["right"]
    return st


def _quat(yaw_deg):
    return G.quat_from_yaw(math.radians(yaw_deg))


def _pose(yaw_deg, x=8.2, y=0.07):
    return th.tensor([x, y, 1.16]), _quat(yaw_deg)


def _between(s, role_a, qa, role_b, qb):
    d = (s.widest_face_normal_yaw(s.hull[role_a], qa) - s.widest_face_normal_yaw(s.hull[role_b], qb)) % math.pi
    return math.degrees(min(d, math.pi - d))


# ---------------------------------------------------------------------------------------------------------------- the faces


def test_the_removed_instance_ended_perpendicular():
    """touching#02 c8_k0: the target's goal yaw -0.93 deg, the partner's 87.9 deg - both edge-aligned, in different 90 deg turns: the
    target's widest face along the edge, the partner's facing the robot."""
    s, st = _sampler(), _stance()
    assert s.widest_face_axis(s.hull["target"], _quat(-0.93), st) == "tangent"
    assert s.widest_face_axis(s.hull["partner"], _quat(87.9), st) == "depth"
    assert _between(s, "target", _quat(-0.93), "partner", _quat(87.9)) == pytest.approx(88.83, abs=0.01)


def test_widest_face_normal_is_the_footprint_short_axis_whatever_the_local_frame():
    s = _sampler(partner=(PARTNER[1], PARTNER[0], PARTNER[2]))  # a partner whose thin axis is its local y
    for yaw in (0.0, 30.0, 90.0, 135.0):
        assert math.degrees(s.widest_face_normal_yaw(s.hull["target"], _quat(yaw))) == pytest.approx(yaw % 180.0, abs=1e-3)
        assert math.degrees(s.widest_face_normal_yaw(s.hull["partner"], _quat(yaw))) == pytest.approx((yaw + 90.0) % 180.0, abs=1e-3)


# ---------------------------------------------------------------------------------------------------------------- the key


def test_off_by_default_and_spec_errors():
    assert _sampler().widest_faces() is None
    for mode in IS.WIDEST_FACES:
        assert _sampler(gen={"widest_faces": mode}).widest_faces() == mode
    for s in (_sampler("nextto", gen={"widest_faces": "parallel"}), _sampler(gen={"widest_faces": "front"}),
              _sampler(gen={"widest_faces": "parallel", "edge_aligned": False}),
              _sampler(gen={"widest_faces": "parallel"}, shelf_start={"lean_side": 1.0, "lean_deg": 15.0})):
        with pytest.raises(Reject) as e:
            s.widest_faces()
        assert e.value.reason == "spec" and "widest_faces" in str(e.value)


@pytest.mark.parametrize("inward", [(0.0, 1.0), (1.0, 0.0), (math.cos(0.3), math.sin(0.3))])
def test_target_init_yaws_per_mode(inward):
    st = _stance(inward)
    tan = math.atan2(float(st.tangent[1]), float(st.tangent[0]))
    s = _sampler(gen={"widest_faces": "parallel"})
    yaws = s.widest_faces_target_yaws(st)
    assert len(yaws) == 4 and all(abs(s.edge_yaw_delta(s.hull["target"], _quat(math.degrees(y)), st)) < 1e-4 for y in yaws)
    axes = [s.widest_face_axis(s.hull["target"], _quat(math.degrees(y)), st) for y in yaws]
    assert sorted(axes) == ["depth", "depth", "tangent", "tangent"]
    for mode in ("tangent", "depth"):
        sm = _sampler(gen={"widest_faces": mode})
        ys = sm.widest_faces_target_yaws(st)
        assert len(ys) == 2 and abs(C.wrap_angle(ys[1] - ys[0])) == pytest.approx(math.pi, abs=1e-5)
        assert all(sm.widest_face_axis(sm.hull["target"], _quat(math.degrees(y)), st) == mode for y in ys)
        ref = tan if mode == "tangent" else tan + math.pi / 2
        assert all(_between(sm, "target", _quat(math.degrees(y)), "target", _quat(math.degrees(ref))) < 1e-3 for y in ys)


@pytest.mark.parametrize("partner", [PARTNER, (PARTNER[1], PARTNER[0], PARTNER[2])])
def test_partner_yaws_and_goal_parallel_to_the_target(partner):
    """Every target init yaw the key draws: the partner's two turns (180 deg apart) are parallel to it, and the goal yaw (the edge-aligned
    yaw nearest the init yaw) is the init yaw - parallel at init and at the goal, whichever local axis is each model's thin one."""
    st = _stance()
    s = _sampler(gen={"widest_faces": "parallel"}, partner=partner)
    for y in s.widest_faces_target_yaws(st):
        t_pose = _pose(math.degrees(y))
        p_yaws = s.widest_faces_partner_yaws(t_pose, st)
        assert len(p_yaws) == 2 and abs(C.wrap_angle(p_yaws[1] - p_yaws[0])) == pytest.approx(math.pi, abs=1e-5)
        for py in p_yaws:
            p_pose = _pose(math.degrees(py), x=8.0)
            facts = s.widest_faces_facts(t_pose, p_pose, st)
            assert facts["init_deg"] == pytest.approx(0.0, abs=0.01) and facts["goal_deg"] == pytest.approx(0.0, abs=0.01)
            assert facts["target_init_axis"] == facts["target_goal_axis"] == facts["partner_axis"] and facts["mode"] == "parallel"
        dot = abs(float(th.dot(G.t(s.target_goal_quat(t_pose, st)), G.t(t_pose[1]))))
        assert 2.0 * math.acos(min(1.0, dot)) < 1e-3  # no in-hand turn


def _place_partner_yaws(s, t_pose, st, monkeypatch):
    """The yaws ``place_partner``'s touching branch tries (every draw refused, so all 8 tries run)."""
    tried = []
    monkeypatch.setattr(IS, "left_contact_band_cells", lambda *a, **k: None)
    s.shelf_partner_pose, s.region_side, s.meta_spots, s.template_facts, s.bands = None, 1.0, None, {}, None
    s.draw_partner_pose = lambda rng, side, t, manip, cam, yaw: tried.append(yaw) or (None, "refused")
    with pytest.raises(Reject) as e:
        s.place_partner(np.random.default_rng(0), t_pose, st, None)
    assert e.value.reason == "partner_placement" and len(tried) == 8
    return tried


def test_place_partner_tries_the_parallel_turns_only_with_the_key(monkeypatch):
    st = _stance()
    t_pose = _pose(0.0)  # the target's widest face along the edge (normal along x = the tangent)
    s = _sampler(gen={"widest_faces": "parallel"})
    tried = _place_partner_yaws(s, t_pose, st, monkeypatch)
    assert tried == [s.widest_faces_partner_yaws(t_pose, st)[i % 2] for i in range(8)]
    assert all(_between(s, "target", t_pose[1], "partner", _quat(math.degrees(y))) < 1e-3 for y in tried)
    # without the key: the two edge-aligned footprints alternate (one of them perpendicular to the target's widest face, as in c8_k0)
    s0 = _sampler()
    tried0 = _place_partner_yaws(s0, t_pose, st, monkeypatch)
    yaw0 = s0.edge_yaw_delta(s0.hull["partner"], s0.rest["partner"], st)
    assert tried0 == [yaw0 + (i % 2) * math.pi / 2 for i in range(8)]
    assert sorted(round(_between(s0, "target", t_pose[1], "partner", _quat(math.degrees(y)))) for y in tried0[:2]) == [0, 90]


def test_phase3_two_tier_draws_the_widest_faces_yaws_without_jitter(monkeypatch):
    st = _stance()
    seen = []
    monkeypatch.setattr(TT, "phase3_target", lambda *a, **kw: seen.append(kw) or ("pose", "grasp"))
    for gen, expect in (({"widest_faces": "parallel"}, True), ({}, False)):
        s = _sampler(gen=gen, grasps_flush=["g"], grasps=["g"], top="top", arm_bands=None,
                     choice=SimpleNamespace(models={"target": SimpleNamespace(scale=None)}))
        s.support_objects = lambda: []
        assert s.phase3_two_tier(st, np.random.default_rng(0)) == ("pose", "grasp")
        kw = seen[-1]
        if expect:
            assert kw["yaws"] == pytest.approx(s.widest_faces_target_yaws(st)) and kw["jitter"] == 0.0
        else:
            assert "yaws" not in kw and "jitter" not in kw


# ---------------------------------------------------------------------------------------------------------------- the band store


def _phase3(monkeypatch, yaws=None):
    calls, stored = {}, []

    class IKC:
        def feasible_sets(self, cells, yaws_, grasps, pose_at, stance, arm, ignore, eef_of):
            calls["yaws"] = list(yaws_)
            return {0: [(y, grasps[0]) for y in yaws_]}

        def exact(self, poses, stance, arm, **kw):
            return SimpleNamespace(ok=True, reason="", q=None)

    class Top:
        z = 0.8

        def polygon_supported(self, fp, margin=0.0):
            return True

    monkeypatch.setattr(TT, "store_feasible_set", lambda *a, **k: stored.append(a))
    s = SimpleNamespace(gp=SimpleNamespace(gen=lambda k, d=None: False if k == "grasp_twins" else d), ikc=IKC(), bands=None, kit=SimpleNamespace(stats={}),
                        world=SimpleNamespace(bottom_offset=lambda hull, q: 0.01, footprint=lambda hull, pose: th.zeros(4, 2)),
                        bindings=SimpleNamespace(one=lambda role: SimpleNamespace(name="box__of__cereal.n.01_1")))
    manip = SimpleNamespace(focus=th.tensor([1.0, 2.0, 0.8]), translate=0.3, tilt=0.0, name="manipulation", working_arms=["right"])
    g = SimpleNamespace(name="grasp_0", key="box_of_cereal__iucshm__s080", pose=(th.zeros(3), C.IDENTITY_QUAT.clone()), finger_q=None)
    kw = {} if yaws is None else {"yaws": yaws, "jitter": 0.0}
    pose, rec = TT.phase3_target(s, manip, np.random.default_rng(0), [g], _box(*TARGET), C.IDENTITY_QUAT.clone(), [], top=Top(), support_z=0.8, **kw)
    return calls, stored, rec


def test_phase3_target_keeps_f_over_other_yaws_out_of_the_band(monkeypatch):
    yaws = [-0.002 + k * math.pi / 2 for k in range(4)]
    calls, stored, rec = _phase3(monkeypatch, yaws)
    assert calls["yaws"] == yaws and stored == []
    assert any(rec.yaw == pytest.approx(y, abs=1e-9) for y in yaws)  # no jitter
    monkeypatch.setenv("ISPATIALGYM_FAST_BAND_EVIDENCE", "0")  # the band store is the base path (BAND_EVIDENCE keeps counts instead)
    monkeypatch.setenv("ISPATIALGYM_FAST_BAND_REFINE", "0")  # (and so does BAND_REFINE)
    calls, stored, rec = _phase3(monkeypatch)  # the band's 8 yaws: F is stored as before
    assert calls["yaws"] == list(TT.YAWS) and len(stored) == 1


# ---------------------------------------------------------------------------------------------------------------- the customs


def test_touching02_variants_carry_the_key_and_every_use_is_valid():
    with open(CUSTOMS) as f:
        customs = json.load(f)
    t02 = [k for k in customs if k.startswith("inter_object_relation_satisfaction-touching_small_scale-box_of_cereal")]
    assert len(t02) == 12 and all(customs[k]["generation"].get("widest_faces") == "parallel" for k in t02)
    for k, e in customs.items():
        mode = (e.get("generation") or {}).get("widest_faces") if isinstance(e, dict) else None
        if mode is not None:
            assert mode in IS.WIDEST_FACES and e["iddl"] == "inter_object_relation_satisfaction/touching_small_scale.iddl", k
