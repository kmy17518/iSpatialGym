"""Simulator-free tests of the push-under's pitched hand (owner 2026-10-12 ~20:55 item C, under_large #02 / #03: ``generation.push_under_hand``
= ``"pitched_far"``): the wrist turns' quaternions per pitch, the hand's lowest sphere and the fingertip stand-off, the palm test, the chain's
poses / pitch ladder / record, the precedence over ``push_under_palm_contact``, stance C's farther defaults and the goal check's re-solve
seeded from the sampled chain (``pytest ispatialgym/tests/test_iors_large_push_pitched.py``)."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import logging
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
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")

VARIANT = "inter_object_relation_satisfaction-under_large_scale-briefcase-desk_mdhelw_0-floor"
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
        self.variant, self.generation = VARIANT, dict(gen)

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


# ----------------------------------------------------------------------------------------------------------------
# the goal check's re-solve seeded from the sampled chain (owner 2026-10-13 ~01:05 item 1)
# ----------------------------------------------------------------------------------------------------------------

ARM_IDX = {"right": th.arange(14, 21), "left": th.arange(7, 14)}
TNAME = "briefcase.n.01_1"
FRESH_FAIL = "approach_untrackable: a configuration just after the start pose collides (3/3; self-collision (base_link x left_arm_link6))"


class _BranchIKC(_IKC):
    """The exact IK in two wrist branches, writing the IK memory as ``IKChecker.exact`` does (the first pose's configuration - ``q_first``
    when given - before the approach is tracked, the other poses' on success). ``ok(q_first)`` decides a push chain: by default a chain
    solved afresh fails (the default branch: under#02 pitched_c2_k0's left wrist into the base) and a chain leaving from a given
    configuration passes; the drop re-check (two poses) passes. Wrist twins by ``conventions.twin_arm_q``."""

    def __init__(self, ok=lambda q_first: q_first is not None, witness_ok=True):
        super().__init__()
        self.chain_ok, self.witness_ok, self.witness_calls = ok, witness_ok, []
        self.kit = SimpleNamespace(floor_z=0.0, robot=SimpleNamespace(arm_control_idx=ARM_IDX))

    @staticmethod
    def config(arm: str, k: int) -> th.Tensor:
        q = th.zeros(28)
        q[ARM_IDX[arm]] = th.tensor([0.1, 0.2, 0.3, 0.4, 1.5, 0.6, 0.7]) + 0.01 * k
        return q

    def exact(self, poses, stance, arm="right", ignore_for_last=(), cache=True, fingers=None, q_first=None, **kw):
        self.calls.append({"poses": list(poses), "fingers": fingers, "stance": stance.name, "arm": arm, "q_first": None if q_first is None else q_first.clone(),
                           "memory": dict(self.solutions), **self._world})
        if len(poses) == 2:
            return SimpleNamespace(ok=True, q=th.zeros(28), reason="")
        k = len(self.calls)
        self.solutions[(arm, self._pose_key(poses[0]))] = self.config(arm, k) if q_first is None else q_first.clone()
        ok = bool(self.chain_ok(q_first))
        if ok:
            for i, p in enumerate(poses[1:], 1):
                self.solutions[(arm, self._pose_key(p))] = self.config(arm, 10 * k + i)
        return SimpleNamespace(ok=ok, q=self.config(arm, k) if ok else None, reason="" if ok else (FRESH_FAIL if q_first is None else "approach_untrackable: from the seed"))

    def joint_limit_margin(self, q, arm):
        return 1.0, f"{arm}_arm_joint1"

    def twin_q(self, q, arm):
        t = C.twin_arm_q(q[ARM_IDX[arm]])
        if t is None:
            return None
        out = q.clone()
        out[ARM_IDX[arm]] = t
        return out

    def q_locked(self, stance, **kw):
        q = th.zeros(28)
        q[0] = 5.0  # a base joint: the stance's own, never the record's
        return q

    def collision_free_at(self, q, pose, arm, ignore=()):
        self.witness_calls.append({"q": q.clone(), "pose": pose, "arm": arm, **self._world})
        return self.witness_ok

    def eef_in_frame(self, pose, cam):
        return True


def _push_calls(calls):
    return [c for c in calls if len(c["poses"]) == 4]


def _sampled_chain():
    """The pitched chain of a left-arm push whose every turn fails in its own branch: the orientation rule takes a wrist-flipped one."""
    s = _sampler(push_under_hand="pitched_far")
    s.seg_arm, s.ikc = "left", _BranchIKC()
    s.kit = s.ikc.kit
    return s, _push(s)


def _record(ch, seed=True):
    """The push record as the C loop writes it."""
    return IL.PushUnderRecord("left", ch["pre"], ch["start"], ch["mid"], ch["goal"], ch["direction"], ch["distance_m"], TNAME, stance="push",
                              pitch_deg=ch["pitch_deg"], blade_standoff_m=ch["blade_standoff_m"], push_height_m=ch["push_height_m"],
                              half_extent_m=ch["half_extent_m"], fingertip_z_m=ch["fingertip_z_m"], ground_ignored=ch["ground_ignored"],
                              drop_cell=[1.0, 0.0], goal_cell=[1.4, 0.0], **IL.push_under_palm_flags(ch), **ch["record"], **(ch["seed"] if seed else {}))


def _goal_check(monkeypatch, rec, ikc):
    """``after_goal_check`` of the push-under attempt whose push record is ``rec`` (the goal settled on the goal cell, the drop re-check passes)."""
    s = _sampler(push_under_hand="pitched_far")
    s.seg_arm, s.ikc, s.kit = "left", ikc, ikc.kit
    s.bindings = SimpleNamespace(one=lambda role: SimpleNamespace(name=TNAME))
    stances = {"placement": dataclasses.replace(_stance(), name="placement", translate=0.95, tilt=-1.2), "push": _stance(),
               "manipulation": SimpleNamespace(name="manipulation")}
    eef = (th.tensor([1.0, 0.0, 0.3]), G.IDENTITY_QUAT.clone())
    grasp = SimpleNamespace(arm="left", grasp=eef, pre_place=eef, place=eef, eef_in_object=(th.zeros(3), G.IDENTITY_QUAT.clone()))
    s.sidecar = SimpleNamespace(grasp_of=lambda name: grasp, stance=lambda name: stances[name], stances=stances, contacts={"push": rec})
    s.camera_pose_at = lambda stance: None
    monkeypatch.setattr(IL, "attached", lambda *a, **k: contextlib.nullcontext())
    s.after_goal_check(SimpleNamespace(pose_targets={TNAME: {"position": [1.40, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}}))
    return s


def test_the_push_record_keeps_the_sampled_chain_configuration():
    s, ch = _sampled_chain()
    assert ch is not None and ch["orient"]["source"] == "map"  # every turn failed in its own branch: a wrist-flipped one was taken
    calls = _push_calls(s.ikc.calls)
    assert all(c["q_first"] is None for c in calls[:4]) and calls[-1]["q_first"] is not None
    idx, seed = ARM_IDX["left"], ch["seed"]
    assert seed["chain_branch"] == "map" and len(seed["chain_arm_q"]) == 4 and all(len(q) == 7 for q in seed["chain_arm_q"])
    # the pre-pose's configuration is the wrist-flipped start the chain was tracked from, the others the chain's solved poses
    assert seed["chain_arm_q"][0] == pytest.approx(calls[-1]["q_first"][idx].tolist(), abs=1e-5)
    for q, pose in zip(seed["chain_arm_q"][1:], (ch["start"], ch["mid"], ch["goal"])):
        assert q == pytest.approx(s.ikc.solutions[("left", s.ikc._pose_key(pose))][idx].tolist(), abs=1e-5)
    rec = _record(ch)
    assert rec.extras["chain_arm_q"] == seed["chain_arm_q"] and rec.extras["chain_branch"] == "map"
    pc = SC.PushContact.from_dict(rec.to_sidecar(), "contacts.push")
    assert pc.extra["chain_arm_q"] == seed["chain_arm_q"] and pc.extra["chain_branch"] == "map"
    # without every configuration (an orientation result without them, one missing) no seed is recorded
    assert IL.push_chain_seed(s.kit, SimpleNamespace(proposal=SimpleNamespace(chain=[None] * 4)), "left") == {}
    assert IL.push_chain_seed(s.kit, SimpleNamespace(proposal=SimpleNamespace(chain=[None] * 4), qs=[th.zeros(28)] * 3), "left") == {}


def test_the_goal_check_re_solve_leaves_from_the_sampled_configuration(monkeypatch, caplog):
    s, ch = _sampled_chain()
    rec = _record(ch)
    n0, nw0 = len(s.ikc.calls), len(s.ikc.witness_calls)
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        _goal_check(monkeypatch, rec, s.ikc)
    calls = _push_calls(s.ikc.calls[n0:])
    assert len(calls) == 1  # seeded: no unseeded solve
    c, q_first = calls[0], calls[0]["q_first"]
    assert q_first is not None and q_first[ARM_IDX["left"]].tolist() == pytest.approx(rec.extras["chain_arm_q"][0], abs=1e-5)
    assert float(q_first[0]) == 5.0  # the other joints from the stance's configuration
    assert c["stance"] == "push" and c["arm"] == "left" and c["fingers"] == C.FINGER_CLOSED
    assert c["ignored"] == ["floors_jluwuz_0", "briefcase.n.01_1"] and c["ground"] is True
    assert c["poses"][0] is rec.pre and th.allclose(c["poses"][1][0], ch["start"][0])
    # the configuration was verified as a witness of the pre-pose first, in the same world - the very test the sampler ran when it took the chain
    w = s.ikc.witness_calls[nw0:]
    assert len(w) == 1 and w[0]["pose"] is rec.pre and th.equal(w[0]["q"], q_first) and w[0]["ignored"] == ["floors_jluwuz_0", "briefcase.n.01_1"]
    assert nw0 == 1 and th.equal(s.ikc.witness_calls[0]["q"], q_first) and s.ikc.witness_calls[0]["pose"] is rec.pre
    assert th.allclose(rec.goal[0], c["poses"][3][0]) and rec.extras["goal_cell"] == pytest.approx([1.40, 0.0])
    msgs = [r.getMessage() for r in caplog.records if "push-under goal check re-solve" in r.getMessage()]
    assert len(msgs) == 1 and "push-under goal check re-solve: seeded ok (the sampled chain's wrist-flipped branch, left arm)" in msgs[0]


def test_the_goal_check_falls_back_to_the_unseeded_solve(monkeypatch, caplog):
    _, ch = _sampled_chain()

    def run(ikc, seed=True):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
            _goal_check(monkeypatch, _record(ch, seed=seed), ikc)
        msgs = [r.getMessage() for r in caplog.records if "push-under goal check re-solve" in r.getMessage()]
        assert len(msgs) == 1
        return _push_calls(ikc.calls), msgs[0]

    # the seeded solve fails, the unseeded passes - with the IK memory as it was before the seeded one
    ikc = _BranchIKC(ok=lambda q_first: q_first is None)
    ikc.solutions[("left", (0.0,))] = th.ones(28)
    calls, msg = run(ikc)
    assert [c["q_first"] is not None for c in calls] == [True, False]
    assert calls[1]["memory"].keys() == calls[0]["memory"].keys() == {("left", (0.0,))}
    assert "push-under goal check re-solve: seeded failed (approach_untrackable: from the seed) -> unseeded ok" in msg
    # both fail: the push is unreachable, the unseeded solve's reason first, the seeded one's after it
    ikc = _BranchIKC(ok=lambda q_first: False)
    with pytest.raises(Reject) as e:
        run(ikc)
    assert e.value.reason == "push_unreachable" and str(e.value) == (f"[push_unreachable] push-under chain to the settled goal fails the exact IK from stance C: "
                                                                      f"{FRESH_FAIL} (seeded from the sampled chain's pre-pose configuration: approach_untrackable: from the seed)")
    assert "push-under goal check re-solve: seeded failed (approach_untrackable: from the seed) -> unseeded failed" in caplog.text
    # the recorded configuration no longer a witness of the pre-pose (it collides here): not pinned, the unseeded solve decides
    ikc = _BranchIKC(ok=lambda q_first: q_first is None, witness_ok=False)
    calls, msg = run(ikc)
    assert [c["q_first"] for c in calls] == [None] and len(ikc.witness_calls) == 1
    assert "push-under goal check re-solve: seeded failed (the sampled pre-pose configuration is off the pre-pose or collides here) -> unseeded ok" in msg
    # a record without the configuration (written before it was kept): the unseeded solve alone
    ikc = _BranchIKC(ok=lambda q_first: q_first is None)
    calls, msg = run(ikc, seed=False)
    assert [c["q_first"] for c in calls] == [None] and ikc.witness_calls == []
    assert "push-under goal check re-solve: no seed (the record has no chain_arm_q) -> unseeded ok" in msg


# ----------------------------------------------------------------------------------------------------------------
# the pinned-start check when the chain is accepted (owner 2026-10-13 ~03:15 item 1)
# ----------------------------------------------------------------------------------------------------------------


class _PinIKC(_BranchIKC):
    """``_BranchIKC`` whose goal-check witness test fails for the first ``n_bad`` configurations asked (a wrist-flipped option mapped from a
    remembered solution of another stance C: its arm joints off the pre-pose on this stance) and whose seeded push chains fail on the calls
    numbered in ``seeded_fail`` (1-based among the seeded 4-pose calls)."""

    def __init__(self, n_bad=0, seeded_fail=()):
        super().__init__()
        self.n_bad, self.seeded_fail, self.n_seeded = n_bad, set(seeded_fail), 0

    def collision_free_at(self, q, pose, arm, ignore=()):
        super().collision_free_at(q, pose, arm, ignore)
        return len(self.witness_calls) > self.n_bad

    def exact(self, poses, stance, arm="right", ignore_for_last=(), cache=True, fingers=None, q_first=None, **kw):
        if len(poses) == 4 and q_first is not None:
            self.n_seeded += 1
            if self.n_seeded in self.seeded_fail:
                self.calls.append({"poses": list(poses), "fingers": fingers, "stance": stance.name, "arm": arm, "q_first": q_first.clone(),
                                   "memory": dict(self.solutions), **self._world})
                return SimpleNamespace(ok=False, q=None, reason="approach_untrackable: differential IK did not converge at waypoint 17/19")
        return super().exact(poses, stance, arm, ignore_for_last=ignore_for_last, cache=cache, fingers=fingers, q_first=q_first, **kw)


def _pinned(ikc, caplog, **gen):
    s = _sampler(push_under_hand="pitched_far", **gen)
    s.seg_arm, s.ikc, s.kit = "left", ikc, ikc.kit
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        ch = _push(s)
    return s, ch, [r.getMessage() for r in caplog.records if "push-under pinned-start check" in r.getMessage()]


def test_the_pinned_start_check_passes_over_an_option_off_the_pre_pose(caplog):
    s, ch, msgs = _pinned(_PinIKC(n_bad=1), caplog)
    # the first wrist-flipped option is no witness of its pre-pose on this stance: passed over, the next one is taken and recorded
    assert ch is not None and ch["orient"]["source"] == "map" and ch["orient"]["chosen"] == "pitch 60 plane-"
    w = s.ikc.witness_calls
    assert len(w) == 2 and all(c["ignored"] == ["floors_jluwuz_0", "briefcase.n.01_1"] and c["ground"] is True and c["arm"] == "left" for c in w)
    idx = ARM_IDX["left"]
    for c, call in zip(w, [c for c in _push_calls(s.ikc.calls) if c["q_first"] is not None]):
        # the configuration as the goal check rebuilds it: the stance's locked joints, the arm's recorded (rounded) joints, at the pre-pose
        assert float(c["q"][0]) == 5.0 and c["q"][idx].tolist() == pytest.approx(call["q_first"][idx].tolist(), abs=1e-5) and c["pose"] is call["poses"][0]
    assert ch["seed"]["chain_arm_q"][0] == pytest.approx(w[1]["q"][idx].tolist(), abs=1e-5) and w[1]["pose"] is ch["pre"]
    assert len(msgs) == 1 and "push-under pinned-start check: pitch 60 lateral [wrist-flipped, left arm] passed over - the recorded pre-pose configuration is off the pre-pose " \
                               "or collides on this stance (C d 0.95 λ +0.00 trunk (2.0, 0.0))" in msgs[0]
    bad = [o for o in ch["orient"]["options"] if o["label"] == "pitch 60 lateral" and o["source"] == "map" and not o["ok"]]
    assert bad and bad[-1]["why"].startswith("downstream: the recorded pre-pose configuration is off the pre-pose")


def test_the_pinned_start_check_rejects_the_chain_when_no_option_holds(caplog):
    s, ch, msgs = _pinned(_PinIKC(n_bad=10 ** 6), caplog)
    assert ch is None  # every option of every pitch passed over: the goal check's seed test would have failed each
    n_map = len([c for c in _push_calls(s.ikc.calls) if c["q_first"] is not None])
    assert n_map == len(s.ikc.witness_calls) == len(msgs) > 0
    assert "pinned-start check" in caplog.text and "downstream: the recorded pre-pose configuration is off the pre-pose" in caplog.text
    # generation.push_pinned_start_check "off": today's acceptance, the first wrist-flipped option recorded untested
    caplog.clear()
    s, ch, msgs = _pinned(_PinIKC(n_bad=10 ** 6), caplog, push_pinned_start_check="off")
    assert ch is not None and ch["orient"]["chosen"] == "pitch 60 lateral" and s.ikc.witness_calls == [] and msgs == []


def test_the_pinned_start_check_without_a_record_or_a_shift_key_adds_no_solve(caplog):
    s, ch, msgs = _pinned(_PinIKC(), caplog)
    assert ch is not None and msgs == [] and len(s.ikc.witness_calls) == 1
    assert len(_push_calls(s.ikc.calls)) == 5  # the four turns afresh + the wrist-flipped option's own solve: no settle re-solve without the key
    # a chosen option without every configuration recorded has nothing to pin: it passes untested (the goal check solves afresh then)
    assert s.push_pinned_check(_stance(), SimpleNamespace(proposal=SimpleNamespace(chain=[None] * 4), qs=None), "left", True) == (True, "")
    assert len(s.ikc.witness_calls) == 1


def test_the_pinned_start_check_on_the_blade_keeps_the_slab_off_below_its_tips(caplog):
    s = _sampler(height=0.05, com_z=0.02)
    s.seg_arm, s.ikc = "left", _PinIKC()
    s.kit = s.ikc.kit
    ch = _push(s)
    assert ch is not None and ch["ground_ignored"] is True and s.push_hand == "blade"
    assert len(s.ikc.witness_calls) == 1 and s.ikc.witness_calls[0]["ground"] is False  # the goal check's world: ground=not ground_ignored
    assert s.ikc.witness_calls[0]["ignored"] == ["floors_jluwuz_0", "briefcase.n.01_1"]


def test_the_settle_shift_key_re_solves_the_pinned_chain_to_a_farther_goal(caplog):
    # the first option's re-solve to the goal 4 mm farther fails (seeded call 2): passed over; the second option's passes (call 4)
    s, ch, msgs = _pinned(_PinIKC(seeded_fail=(2,)), caplog, push_pinned_settle_m=0.004)
    assert ch is not None and ch["orient"]["chosen"] == "pitch 60 plane-"
    seeded = [c for c in _push_calls(s.ikc.calls) if c["q_first"] is not None]
    assert len(seeded) == 4  # option 1: its solve + the shifted re-solve; option 2: the same
    for own, shifted, w in zip(seeded[0::2], seeded[1::2], s.ikc.witness_calls):
        assert shifted["poses"][0] is own["poses"][0] and shifted["poses"][1] is own["poses"][1]
        assert (shifted["poses"][3][0] - own["poses"][3][0]).tolist() == pytest.approx([0.004, 0.0, 0.0], abs=1e-6)  # along the push (+x)
        assert shifted["poses"][2][0].tolist() == pytest.approx(((own["poses"][1][0] + shifted["poses"][3][0]) / 2).tolist(), abs=1e-6)
        assert th.equal(shifted["q_first"], w["q"]) and shifted["fingers"] == C.FINGER_CLOSED and shifted["stance"] == "push"
        assert shifted["ignored"] == ["floors_jluwuz_0", "briefcase.n.01_1"] and shifted["ground"] is True
    # the IK memory is left as it was: the shifted goal's solution is not remembered
    assert ("left", s.ikc._pose_key(seeded[3]["poses"][3])) not in s.ikc.solutions
    assert len(msgs) == 1 and "pitch 60 lateral [wrist-flipped, left arm] passed over - the seeded re-solve fails with the push goal 4 mm farther: " \
                               "approach_untrackable: differential IK did not converge at waypoint 17/19" in msgs[0]
