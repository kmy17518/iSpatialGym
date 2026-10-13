"""Simulator-free tests of the push-under's pitched hand (owner 2026-10-12 ~20:55 item C, under_large #02 / #03: ``generation.push_under_hand``
= ``"pitched_far"``): the wrist turns' quaternions per pitch, the hand's lowest sphere and the fingertip stand-off, the palm test, the chain's
poses / pitch ladder / record, the precedence over ``push_under_palm_contact`` and stance C's farther defaults
(``pytest ispatialgym/tests/test_iors_large_push_pitched.py``)."""

from __future__ import annotations

import importlib.util
import math
import os
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle import sidecar as SC
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")

ALIAS = "inter_object_relation_satisfaction-under_large_scale-briefcase-desk_mdhelw_0-floor"
L = C.FINGERTIP_LENGTH
#: the hand's lowest sphere below the fingertips per turn (``tabletop.hand_spheres_eef``): a fingertip with the camera up, a knuckle with the
#: closing axis in the push plane and the camera's spheres reaching up (plane-), the camera hanging below (lateral- / plane)
DROPS = {60.0: {"lateral": 0.0016, "plane-": 0.0082, "lateral-": 0.0191, "plane": 0.0200},
         70.0: {"lateral": 0.0026, "plane-": 0.0250, "lateral-": 0.0382, "plane": 0.0391}}


def _box(dx: float, dy: float, h: float) -> th.Tensor:
    """Hull of a dx x dy x h box, the root at its bottom centre (8 corners, none between bottom and top)."""
    return th.tensor([[sx * dx / 2, sy * dy / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, h)], dtype=th.float32)


class _GP:
    def __init__(self, **gen):
        self.alias, self.generation = ALIAS, dict(gen)

    def gen(self, key, default=None):
        return self.generation.get(key, default)


class _IKC:
    """The exact IK as a recorder: every chain asked, the ground slab state and the ignored objects at the call; ``ok`` decides."""

    def __init__(self, ok=lambda chain: True):
        self.ok, self.calls, self.solutions, self._world = ok, [], {}, {"ground": True, "ignored": []}

    def ignoring(self, objs, ground=True, ground_meshes=True):
        ikc = self

        class _Ctx:
            def __enter__(self_):
                ikc._world = {"ground": ground, "ignored": [o.name for o in objs]}

            def __exit__(self_, *exc):
                ikc._world = {"ground": True, "ignored": []}
                return False

        return _Ctx()

    def exact(self, poses, stance, arm="right", cache=True, fingers=None, q_first=None, **kw):
        self.calls.append({"poses": list(poses), "fingers": fingers, **self._world})
        ok = bool(self.ok(poses))
        return SimpleNamespace(ok=ok, q=th.zeros(28) if ok else None, reason="" if ok else "exact IK failed at pose 1 of 4")

    def q_locked(self, stance, **kw):
        return th.zeros(28)

    @staticmethod
    def _pose_key(pose):
        return tuple(round(float(v), 3) for v in list(G.t(pose[0])) + list(G.t(pose[1])))


def _sampler(height: float = 0.27, com_z: float = 0.115, hull=None, **gen) -> "IL.IORSLarge":
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.relation, s.template_facts, s.template_meta = "under", {}, {}
    s.gp = _GP(**gen)
    s.hull = _box(0.33, 0.27, height) if hull is None else hull
    s.rest, s.floor_z, s.seg_arm = G.IDENTITY_QUAT.clone(), 0.0, "right"
    s.place = {"height": height, "half": 0.165}
    objs = {"floor": SimpleNamespace(name="floors_jluwuz_0"), "target": SimpleNamespace(name="briefcase.n.01_1")}
    s.obj = lambda role: objs[role]
    s._com_local = ("briefcase.n.01_1", th.tensor([0.0, 0.0, com_z]))
    s.ikc, s.kit = _IKC(), SimpleNamespace(floor_z=0.0)
    return s


def _stance() -> Stance:
    edge = EdgeFrame("-y", th.tensor([1.5, 0.0]), th.tensor([1.0, 0.0]), 1.2)
    return Stance("push", (0.1, 0.0, 0.0), 2.0, 0.0, 0.95, edge, None, 0.0, None, ["right"])


def _push(s):
    return s._push_under_chain(_stance(), G.IDENTITY_QUAT.clone(), (1.0, 0.0), (1.40, 0.0), (1.0, 0.0))


# ----------------------------------------------------------------------------------------------------------------
# the hand: turns, lowest sphere, stand-off
# ----------------------------------------------------------------------------------------------------------------


def test_pitched_hand_quaternions_per_pitch():
    d = th.tensor([0.0, 1.0, 0.0])
    up = th.tensor([0.0, 0.0, 1.0])
    left = th.tensor([-1.0, 0.0, 0.0])  # z x d: left of the push
    for pitch in (60.0, 65.0, 70.0):
        p = math.radians(pitch)
        a = th.tensor([0.0, math.sin(p), -math.cos(p)])  # down and toward the push: the fingertips ahead of and below the palm
        n = th.tensor([0.0, math.cos(p), math.sin(p)])  # across the finger axis in the push plane, toward the face and up
        turns = dict(TT.pitched_hand_quats(pitch, (0.0, 1.0)))
        assert list(turns) == ["lateral", "lateral-", "plane", "plane-"]
        closing = {"lateral": left, "lateral-": -left, "plane": n, "plane-": -n}
        for label, q in turns.items():
            assert th.allclose(C.approach_axis(q), a, atol=1e-6), (pitch, label)
            assert th.allclose(C.approach_axis(q), TT.pitched_approach(pitch, d), atol=1e-6)
            assert th.allclose(C.closing_axis(q), closing[label], atol=1e-6), (pitch, label)
        # eef +x = c x a: back and down for "lateral" - the wrist camera (eef -x side) on the hand's upper side
        assert th.allclose(C.quat_to_mat(turns["lateral"])[:, 0].float(), -n, atol=1e-6)
        assert float(C.quat_to_mat(turns["lateral"])[:, 0] @ up) < 0.0
        for name in ("lateral", "plane"):  # each turn and its twin: the same pose turned pi about the approach axis
            assert C.quat_angle(C.twin_pose((th.zeros(3), turns[name]))[1], turns[f"{name}-"]) < 1e-4


def test_pitched_hand_drop_per_turn():
    for pitch, drops in DROPS.items():
        for d in ((1.0, 0.0), (0.0, 1.0), (-0.6, 0.8)):  # the push's heading does not matter
            for label, q in TT.pitched_hand_quats(pitch, d):
                assert TT.pitched_hand_drop(q) == pytest.approx(drops[label], abs=1e-4), (pitch, d, label)
    # the camera-up turn hangs only its fingertip spheres below the tips: the closed fingers' spheres are part of the hand
    q = dict(TT.pitched_hand_quats(60.0, (1.0, 0.0)))["lateral"]
    R = C.quat_to_mat(q).float()
    tip_z = L * float(R[2, 2])
    lowest = min(TT.hand_spheres_eef(), key=lambda s: float(R[2] @ th.tensor(s[1])) - s[2])
    assert lowest[0] == "finger" and tip_z - (float(R[2] @ th.tensor(lowest[1])) - lowest[2]) == pytest.approx(DROPS[60.0]["lateral"], abs=1e-4)
    assert [p for p, _, _ in TT.hand_spheres_eef()].count("finger") == len(TT.FINGER_SPHERES_EEF) == 8


def test_finger_spheres_are_the_robot_model():
    from ispatialgym import REPO_ROOT

    roots = [os.path.join(REPO_ROOT, "Behavior-1K", "datasets")] + ([os.environ["OMNIGIBSON_DATA_PATH"]] if os.environ.get("OMNIGIBSON_DATA_PATH") else [])
    spec = importlib.util.find_spec("omnigibson")
    if spec is not None and spec.origin:
        roots.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(spec.origin))), "datasets"))
    root = next((os.path.join(r, "omnigibson-robot-assets", "models", "r1pro") for r in roots
                 if os.path.exists(os.path.join(r, "omnigibson-robot-assets", "models", "r1pro", "curobo", "r1pro_description_curobo_default.yaml"))), None)
    if root is None or not os.path.exists(os.path.join(root, "urdf", "r1pro.urdf")):
        pytest.skip("R1Pro cuRobo description / URDF not available")
    import yaml

    from ispatialgym.oracle.r1pro_model import _rpy_matrix

    with open(os.path.join(root, "curobo", "r1pro_description_curobo_default.yaml")) as f:
        spheres = yaml.safe_load(f)["robot_cfg"]["kinematics"]["collision_spheres"]
    joints = {j.find("child").get("link"): j for j in ET.parse(os.path.join(root, "urdf", "r1pro.urdf")).getroot().iter("joint") if j.find("child") is not None}
    R_ge, t_ge = _rpy_matrix(*M.EEF_IN_GRIPPER_RPY), np.asarray(M.EEF_IN_GRIPPER_XYZ)
    for arm in ("left", "right"):
        model = []
        for finger in M.FINGER_LINKS[arm]:  # the finger joints at 0: closed
            o = joints[finger].find("origin")
            Rf, tf = _rpy_matrix(*[float(v) for v in (o.get("rpy") or "0 0 0").split()]), np.array([float(v) for v in o.get("xyz").split()])
            model += [((Rf @ np.asarray(s["center"], dtype=float) + tf - t_ge) @ R_ge, float(s["radius"])) for s in spheres[finger]]
        assert len(model) == len(TT.FINGER_SPHERES_EEF), arm
        for c, r in TT.FINGER_SPHERES_EEF:  # within the constants' 0.5 mm rounding
            assert any(abs(r - rm) < 1e-9 and np.abs(np.asarray(c) - cm).max() <= 5e-4 for cm, rm in model), (arm, c, r)


def test_pitched_tip_offsets():
    for pitch in (60.0, 65.0, 70.0):
        p = math.radians(pitch)
        for label, q in TT.pitched_hand_quats(pitch, (1.0, 0.0)):
            h = 0.007 if label.startswith("lateral") else 0.0084  # the tips' half width across a lateral closing axis / half thickness along one in the plane
            so, rise, hh = TT.pitched_tip_offsets(pitch, q, 0.005)
            assert hh == h and so == pytest.approx(L * math.sin(p) + h * math.cos(p) + 0.005) and rise == pytest.approx(L * math.cos(p))
            # the leading edge (h across the finger axis toward the face) stands the clearance short of the contact point
            eef = th.tensor([-so, 0.0, 0.10 + rise])
            edge = C.fingertip((eef, q)) + h * th.tensor([math.cos(p), 0.0, math.sin(p)])
            assert edge.tolist() == pytest.approx([-0.005, 0.0, 0.10 + h * math.sin(p)], abs=1e-6)
    # the flat hand's stand-off is the 90-deg end of the same rule
    q90 = dict(TT.flat_hand_quats((1.0, 0.0)))["lateral"]
    assert TT.pitched_tip_offsets(90.0, q90, 0.005)[0] == pytest.approx(TT.flat_hand_standoff(0.005))


# ----------------------------------------------------------------------------------------------------------------
# the palm test
# ----------------------------------------------------------------------------------------------------------------


def test_palm_test_replaces_the_finger_section_rule():
    box = _box(0.33, 0.27, 0.27)  # the 27 cm briefcase pushed along +x: its back face 0.165 behind the root
    q = dict(TT.pitched_hand_quats(60.0, (1.0, 0.0)))["lateral"]
    so, rise, h = TT.pitched_tip_offsets(60.0, q, 0.005)
    eef = th.tensor([-0.165 - so, 0.0, 0.08 + rise])
    gap, part = IL.pitched_palm_gap(box, (1.0, 0.0), eef, q)
    # the finger-axis rule rejects the face at every blade pitch (it rises past the 7 cm finger section); the hand leaning away clears it
    assert TT.blade_standoff((0.0, 0.27), 0.08, 60.0, 0.005) is None
    assert gap > 0.005 and part == "finger"
    # every sphere is behind the face: the gap is the leading finger sphere's distance to the face plane
    R = C.quat_to_mat(q).float()
    fronts = [float((eef + R @ th.tensor(c))[0]) + r for _, c, r in TT.hand_spheres_eef()]
    assert gap == pytest.approx(-0.165 - max(fronts), abs=1e-5)
    # 3 cm into the face: a finger reaches it
    gap_in, part_in = IL.pitched_palm_gap(box, (1.0, 0.0), eef + th.tensor([0.03, 0.0, 0.0]), q)
    assert gap_in < 0.0 and part_in == "finger"
    # a face leaning back over the fingertips (the top 0.30 m behind the bottom): the knuckles / camera above the tips meet it
    lean = th.tensor([[-0.165, -0.13, 0.0], [-0.165, 0.13, 0.0], [0.165, -0.13, 0.0], [0.165, 0.13, 0.0],
                      [-0.465, -0.13, 0.27], [-0.465, 0.13, 0.27], [0.165, -0.13, 0.27], [0.165, 0.13, 0.27]])
    off = IL.band_back_offset(lean, (1.0, 0.0), 0.08 - h * math.sin(math.radians(60)), 0.08 + h * math.sin(math.radians(60)), section=True)
    gap_l, part_l = IL.pitched_palm_gap(lean, (1.0, 0.0), th.tensor([-off - so, 0.0, 0.08 + rise]), q)
    assert gap_l < 0.0 and part_l in ("knuckle", "wrist camera", "gripper body")
    # above the object nothing is met
    assert IL.pitched_palm_gap(_box(0.33, 0.27, 0.02), (1.0, 0.0), th.tensor([-0.10, 0.0, 0.30]), q) == (math.inf, "")


# ----------------------------------------------------------------------------------------------------------------
# the chain
# ----------------------------------------------------------------------------------------------------------------


def test_pitched_chain_stand_off_poses_and_record():
    s = _sampler(push_under_hand="pitched_far")
    ch = _push(s)
    assert ch is not None and len(s.ikc.calls) == 4  # pitch 60's four turns solved, the orientation rule keeps the generator's first
    for call in s.ikc.calls:
        assert call["ground"] is True and call["ignored"] == ["floors_jluwuz_0", "briefcase.n.01_1"] and call["fingers"] == C.FINGER_CLOSED
    p = math.radians(60.0)
    q = dict(TT.pitched_hand_quats(60.0, (1.0, 0.0)))["lateral"]
    so, rise, h = TT.pitched_tip_offsets(60.0, q, IL.PUSH_UNDER_CLEARANCE_M)
    tip = IL.PUSH_FAR_TIP_M
    assert ch["pitch_deg"] == 60.0 and ch["wrist_sign"] == "lateral" and ch["ground_ignored"] is False and "palm_contact" not in ch
    assert ch["blade_standoff_m"] == pytest.approx(so) and ch["fingertip_z_m"] == pytest.approx(tip) and ch["push_height_m"] == pytest.approx(tip + rise)
    assert ch["half_extent_m"] == pytest.approx(0.165, abs=1e-6)  # the box's back face, met between its corners (cross-sections)
    start, mid, goal, pre = ch["start"][0], ch["mid"][0], ch["goal"][0], ch["pre"][0]
    assert start.tolist() == pytest.approx([1.0 - 0.165 - so, 0.0, tip + rise], abs=1e-5)
    assert goal.tolist() == pytest.approx([1.40 - 0.165 - so + 0.03, 0.0, tip + rise], abs=1e-5)  # push_goal_extra_m past the goal cell
    assert mid.tolist() == pytest.approx(((start + goal) / 2).tolist(), abs=1e-6)
    # the pre-pose 3 cm back along the push and 6 cm up: the fingertips' edge 3.5 cm from the face, the wrist out of the base
    assert IL.PUSH_FAR_PRE_M == (0.03, 0.06) and pre.tolist() == pytest.approx((start + th.tensor([-0.03, 0.0, 0.06])).tolist(), abs=1e-5)
    assert th.allclose(ch["pre"][1], ch["start"][1]) and _sampler(push_under_hand="pitched_far", push_far_pre_m=[0.0, 0.05])._push_under_chain(
        _stance(), G.IDENTITY_QUAT.clone(), (1.0, 0.0), (1.40, 0.0), (1.0, 0.0))["pre"][0].tolist() == pytest.approx((start + th.tensor([0.0, 0.0, 0.05])).tolist(), abs=1e-5)
    edge = C.fingertip(ch["start"]) + h * th.tensor([math.cos(p), 0.0, math.sin(p)])
    assert edge.tolist() == pytest.approx([1.0 - 0.165 - IL.PUSH_UNDER_CLEARANCE_M, 0.0, tip + h * math.sin(p)], abs=1e-5)  # 5 mm short of the face
    rec_x = ch["record"]
    assert rec_x["push_hand"] == "pitched_far" and rec_x["palm_clear"] is True and rec_x["wrist_turn"] == "lateral"
    assert rec_x["contact_height_m"] == pytest.approx(tip + h * math.sin(p)) and rec_x["target_com_height_m"] == pytest.approx(0.115)
    assert rec_x["contact_height_m"] < rec_x["target_com_height_m"] and rec_x["palm_gap_m"] > 0.0
    assert rec_x["hand_low_m"] == pytest.approx(tip - DROPS[60.0]["lateral"], abs=1e-4)
    assert rec_x["fingertip_band_m"] == pytest.approx([tip - h * math.sin(p), tip + h * math.sin(p)])
    # the record as the C loop writes it: push_hand / pitch_deg / palm_clear and the heights reach the oracle's PushContact
    rec = IL.PushUnderRecord("right", ch["pre"], ch["start"], ch["mid"], ch["goal"], ch["direction"], ch["distance_m"], "briefcase.n.01_1", stance="push",
                             pitch_deg=ch["pitch_deg"], blade_standoff_m=ch["blade_standoff_m"], push_height_m=ch["push_height_m"], half_extent_m=ch["half_extent_m"],
                             fingertip_z_m=ch["fingertip_z_m"], ground_ignored=ch["ground_ignored"], drop_cell=[1.0, 0.0], goal_cell=[1.4, 0.0],
                             **IL.push_under_palm_flags(ch), **ch["record"])
    pc = SC.PushContact.from_dict(rec.to_sidecar(), "contacts.push")
    assert pc.kind == "push_under" and pc.mid is not None and pc.extra["push_hand"] == "pitched_far" and pc.extra["palm_clear"] is True
    assert pc.extra["pitch_deg"] == 60.0 and pc.extra["contact_height_m"] == pytest.approx(rec_x["contact_height_m"]) and pc.extra["target_com_height_m"] == pytest.approx(0.115)
    assert "palm_contact" not in pc.extra and "fingertip_z" not in pc.extra  # the oracle's top-down palm clearance has nothing to raise


def test_pitch_ladder_turn_order_and_the_height_rule():
    s = _sampler(push_under_hand="pitched_far")
    s.ikc = _IKC(ok=lambda chain: False)
    assert _push(s) is None
    pitches = [round(math.degrees(math.acos(-float(C.approach_axis(c["poses"][1][1])[2]))), 3) for c in s.ikc.calls]
    assert pitches == [60.0] * 4 + [65.0] * 4 + [70.0] * 4  # the ladder in its order, four turns each
    assert all(float(c["poses"][1][0][2]) == pytest.approx(IL.PUSH_FAR_TIP_M + L * math.cos(math.radians(p))) for c, p in zip(s.ikc.calls, pitches))
    labels = [label for _, _, label, _ in s.pitched_turns(60.0, (1.0, 0.0))]
    assert labels == ["lateral", "plane-", "lateral-", "plane"]  # the hand's lowest part nearest the tips first
    # pitch 60 fails, 65 solves
    s.ikc = _IKC(ok=lambda chain: abs(float(C.approach_axis(chain[1][1])[2]) + math.cos(math.radians(65.0))) < 1e-4)
    ch = _push(s)
    assert ch["pitch_deg"] == 65.0 and ch["record"]["push_hand"] == "pitched_far"
    # the key orders the ladder
    s = _sampler(push_under_hand="pitched_far", push_under_far_pitches_deg=[70, 60])
    s.ikc = _IKC(ok=lambda chain: False)
    _push(s)
    assert [round(math.degrees(math.acos(-float(C.approach_axis(c["poses"][1][1])[2])))) for c in s.ikc.calls] == [70] * 4 + [60] * 4
    # the height rule: the key's height, raised to the lowest sphere + the floor clearance, lowered below the centre of mass
    s = _sampler(push_under_hand="pitched_far", push_far_tip_m=0.0)
    for tip, drop, label, q in s.pitched_turns(70.0, (1.0, 0.0)):
        assert tip == pytest.approx(drop + IL.PUSH_FAR_FLOOR_CLEARANCE_M) and drop == pytest.approx(DROPS[70.0][label], abs=1e-4)
    for tip, drop, label, q in _sampler(push_under_hand="pitched_far").pitched_turns(60.0, (1.0, 0.0), com_h=0.06):
        assert tip == pytest.approx(0.06 - TT.flat_tip_half_height(q) * math.sin(math.radians(60.0)) - 1e-3)  # 8 cm lowered under the centre of mass
    turns70 = {label: tip for tip, _, label, _ in s.pitched_turns(70.0, (1.0, 0.0), com_h=0.04)}
    assert turns70["lateral"] is not None and turns70["plane"] is None and turns70["lateral-"] is None  # the camera below: no room under 4 cm
    # a low object (centre of mass 2.5 cm up): no turn below it, no IK
    s = _sampler(height=0.05, com_z=0.012, push_under_hand="pitched_far")
    assert _push(s) is None and s.ikc.calls == []


def test_face_reaching_the_hand_skips_the_turn():
    lean = th.tensor([[-0.165, -0.135, 0.0], [-0.165, 0.135, 0.0], [0.165, -0.135, 0.0], [0.165, 0.135, 0.0],
                      [-0.465, -0.135, 0.27], [-0.465, 0.135, 0.27], [0.165, -0.135, 0.27], [0.165, 0.135, 0.27]])
    s = _sampler(hull=lean, push_under_hand="pitched_far")
    assert _push(s) is None and s.ikc.calls == []  # every turn at every pitch meets the overhang above its tips


def test_blade_stays_the_default_and_pitched_takes_precedence_over_the_palm():
    s = _sampler(height=0.27, com_z=0.115)
    assert s.push_hand == "blade" and _push(s) is None and s.ikc.calls == []  # the face reaches the blade's palm at every pitch
    assert s.push_drop_ahead_pref() is None
    s.gp = _GP(push_under_hand="pitched_far", push_under_palm_contact=True)
    ch = _push(s)
    assert ch is not None and ch["record"]["push_hand"] == "pitched_far" and "palm_contact" not in ch and IL.push_under_palm_flags(ch) == {}
    assert all(abs(float(C.approach_axis(c["poses"][1][1])[2]) + 0.5) < 1e-5 for c in s.ikc.calls)  # pitch 60 only: no vertical palm blade
    # the yaw order's contact offset: the tips' band through the box's faces
    assert s.push_contact_offset(G.IDENTITY_QUAT.clone(), (1.0, 0.0)) == pytest.approx(0.165, abs=1e-6)
    assert s.push_contact_offset(G.IDENTITY_QUAT.clone(), (0.0, 1.0)) == pytest.approx(0.135, abs=1e-6)
    assert s.push_drop_ahead_pref() == IL.PUSH_FAR_DROP_AHEAD_PREF_M and _sampler(push_under_hand="pitched_far", push_under_drop_ahead_pref_m=0.6).push_drop_ahead_pref() == 0.6


# ----------------------------------------------------------------------------------------------------------------
# stance C farther out
# ----------------------------------------------------------------------------------------------------------------


def _candidates(monkeypatch, **gen):
    from ispatialgym.sampling.v2 import prompt0 as P0

    s = _sampler(**gen)
    edge = EdgeFrame("-y", th.tensor([0.0, 0.0]), th.tensor([0.0, 1.0]), 1.6)  # the desk's -y edge, inward +y
    objs = {"floor": s.obj("floor"), "target": s.obj("target"),
            "partner": SimpleNamespace(name="desk_mdhelw_0", aabb=(th.tensor([-0.8, 0.0, 0.0]), th.tensor([0.8, 0.8, 0.75])))}
    s.obj = lambda role: objs[role]
    s.place.update({"cells": th.tensor([[x / 100.0, 0.07] for x in range(-80, 81, 2)]), "edges": [edge]})
    s._free_floor_cell = lambda xy, why=None: True
    monkeypatch.setattr(P0, "base_ok", lambda self, xy: True)
    monkeypatch.setattr(P0, "body_free", lambda self, cands: th.ones(len(cands), dtype=th.bool))
    monkeypatch.setattr(P0, "camera_poses", lambda self, cands: [None] * len(cands))
    monkeypatch.setattr(P0, "in_frame", lambda self, cam, pts: True)
    monkeypatch.setattr(P0, "axis_angle", lambda self, cam, focus: 0.0)
    return [st for st in s.push_under_candidates(edge, (0.0, -1.0)) if st.extras.get("stance_kind") == "frontal"]


def test_stance_c_stands_farther_out_for_the_pitched_hand(monkeypatch):
    out = 0.165 + IL.PUSH_UNDER_DROP_EXTRA_M
    blade = _candidates(monkeypatch)
    assert {round(st.stand_off, 3) for st in blade} == {so for so in IL.PUSH_UNDER_STAND_OFFS if 0.25 <= so - out <= 0.60}  # today's: <= 0.80
    assert {(st.translate, st.tilt) for st in blade} == set(IL.FLOOR_TRUNKS)
    assert {abs(st.extras["lane_offset_m"]) for st in blade} == {0.25, 0.0}
    far = _candidates(monkeypatch, push_under_hand="pitched_far")
    assert {round(st.stand_off, 3) for st in far} == set(IL.PUSH_FAR_STAND_OFFS)  # 0.80-1.00: every drop cell within 0.80 m ahead
    assert {(st.translate, st.tilt) for st in far} == set(IL.PUSH_FAR_TRUNKS) == {(2.0, 0.0), (1.75, 0.0)}
    assert {abs(st.extras["lane_offset_m"]) for st in far} == set(IL.PUSH_FAR_LANE_OFFSETS) == {0.3, 0.25}  # no centred lane
    assert max(st.extras["drop_ahead_m"] for st in far) == pytest.approx(1.0 - out) and 1.0 - out <= IL.PUSH_FAR_DROP_AHEAD_M[1]
    # the keys override the mode's defaults: a farther list alone is cut at today's 0.60 m window unless push_under_drop_ahead_m widens it
    cut = _candidates(monkeypatch, push_under_stand_offs_m=[0.8, 0.9, 1.0])
    assert {round(st.stand_off, 3) for st in cut} == {0.8}
    keyed = _candidates(monkeypatch, push_under_stand_offs_m=[0.9, 1.0], push_under_drop_ahead_m=[0.25, 0.75], floor_trunks=[[2.0, 0.0]])
    assert {round(st.stand_off, 3) for st in keyed} == {0.9, 1.0} and {(st.translate, st.tilt) for st in keyed} == {(2.0, 0.0)}
    lanes = _candidates(monkeypatch, push_under_hand="pitched_far", frontal_lane_offset_m=[0.35])
    assert {abs(st.extras["lane_offset_m"]) for st in lanes} == {0.35}
