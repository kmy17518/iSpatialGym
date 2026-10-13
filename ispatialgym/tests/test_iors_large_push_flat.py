"""Simulator-free tests of the push-under's flat hand (owner 2026-10-12 ~19:40, under_large #02 / #03: ``generation.push_under_hand`` =
``"flat"``): the wrist turns' quaternions, the fingertip stand-off and chain poses, the height rule (the hand's lowest collision sphere
above the floor, the tips below the centre of mass), the fingertip band, the record's flags and the precedence over
``push_under_palm_contact`` (``pytest ispatialgym/tests/test_iors_large_push_flat.py``)."""

from __future__ import annotations

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

ALIAS = "inter_object_relation_satisfaction-under_large_scale-briefcase-desk-floor"
#: the hand's lowest collision sphere below the eef origin per flat-hand turn (``PALM_SPHERES_EEF`` + the wrist camera's): the gripper
#: body with the camera up (lateral), the knuckles (vertical-), the camera hanging below (lateral- / vertical)
DROPS = {"lateral": 0.040, "vertical-": 0.057, "lateral-": 0.074, "vertical": 0.075}


def _box(dx: float, dy: float, h: float) -> th.Tensor:
    """Hull of a dx x dy x h box, the root at its bottom centre (a briefcase lying flat: 8 corners, none between bottom and top)."""
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


def _sampler(height: float = 0.12, com_z: float = 0.06, **gen) -> "IL.IORSLarge":
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.relation, s.template_facts, s.template_meta = "under", {}, {}
    s.gp = _GP(**gen)
    s.hull = _box(0.33, 0.27, height)
    s.rest, s.floor_z, s.seg_arm = G.IDENTITY_QUAT.clone(), 0.0, "right"
    s.place = {"height": height, "half": 0.165}
    objs = {"floor": SimpleNamespace(name="floors_jluwuz_0"), "target": SimpleNamespace(name="briefcase.n.01_1")}
    s.obj = lambda role: objs[role]
    s._com_local = ("briefcase.n.01_1", th.tensor([0.0, 0.0, com_z]))
    s.ikc, s.kit = _IKC(), SimpleNamespace(floor_z=0.0)
    return s


def _stance() -> Stance:
    edge = EdgeFrame("-y", th.tensor([1.5, 0.0]), th.tensor([1.0, 0.0]), 1.2)
    return Stance("push", (0.5, 0.0, 0.0), 2.0, 0.0, 1.0, edge, None, 0.0, None, ["right"])


def _push(s):
    return s._push_under_chain(_stance(), G.IDENTITY_QUAT.clone(), (1.0, 0.0), (1.40, 0.0), (1.0, 0.0))


# ----------------------------------------------------------------------------------------------------------------
# the wrist turns
# ----------------------------------------------------------------------------------------------------------------


def test_flat_hand_quaternions_per_wrist_turn():
    d = (0.0, 1.0)
    a90 = TT.pitched_approach(90.0, th.tensor(d))
    assert th.allclose(a90, th.tensor([0.0, 1.0, 0.0]), atol=1e-6)  # the pitched blade at 90 deg = the push direction
    with pytest.raises(ValueError):
        C.eef_quat(a90, th.tensor([0.0, 1.0, 0.0]))  # the blade's closing axis along the push is parallel to that approach
    turns = dict(TT.flat_hand_quats(d))
    assert list(turns) == ["lateral", "lateral-", "vertical", "vertical-"]
    closing = {"lateral": [-1.0, 0.0, 0.0], "lateral-": [1.0, 0.0, 0.0], "vertical": [0.0, 0.0, 1.0], "vertical-": [0.0, 0.0, -1.0]}
    for label, q in turns.items():
        assert th.allclose(C.approach_axis(q), a90, atol=1e-6), label  # fingers forward along the push
        assert th.allclose(C.closing_axis(q), th.tensor(closing[label]), atol=1e-6), label
    # eef +x = c x a: lateral puts it down (the wrist camera, on the eef's -x side, up); its twin puts the camera below
    assert th.allclose(th.cross(C.closing_axis(turns["lateral"]), a90, dim=0), th.tensor([0.0, 0.0, -1.0]), atol=1e-6)
    for name in ("lateral", "vertical"):  # each turn and its twin: the same pinch turned pi about the approach axis
        twin = C.twin_pose((th.zeros(3), turns[name]))[1]
        assert C.quat_angle(twin, turns[f"{name}-"]) < 1e-4


def test_flat_tip_band_per_wrist_turn():
    turns = dict(TT.flat_hand_quats((1.0, 0.0)))
    # the tips' vertical half size: across the closing axis (lateral closing) or along it (vertical closing), the finger profiles' tip rows
    assert TT.flat_tip_half_height(turns["lateral"]) == TT.flat_tip_half_height(turns["lateral-"]) == TT.FINGER_SIDE_PROFILE[0][1] == 0.007
    assert TT.flat_tip_half_height(turns["vertical"]) == TT.flat_tip_half_height(turns["vertical-"]) == TT.FINGER_PROFILE[0][1] == 0.0084


# ----------------------------------------------------------------------------------------------------------------
# the height rule
# ----------------------------------------------------------------------------------------------------------------


def test_hand_drop_below_per_wrist_turn():
    for d in ((1.0, 0.0), (0.0, 1.0), (-0.6, 0.8)):  # the push's heading does not matter
        for label, q in TT.flat_hand_quats(d):
            assert TT.hand_drop_below(q) == pytest.approx(DROPS[label], abs=1e-4), (d, label)
            assert TT.hand_drop_below(q) == pytest.approx(TT.palm_half_extent(q, th.tensor([0.0, 0.0, -1.0])))
    # a top-down blade: the whole hand stands above the eef origin (the knuckles 5.25 cm up)
    assert TT.hand_drop_below(C.eef_quat([0.0, 0.0, -1.0], [0.0, 1.0, 0.0])) == pytest.approx(-0.0525, abs=1e-4)


def _r1pro_assets():
    from ispatialgym import REPO_ROOT

    for root in (os.environ.get("OMNIGIBSON_DATA_PATH"), os.path.join(REPO_ROOT, "Behavior-1K", "datasets")):
        path = os.path.join(root or "", "omnigibson-robot-assets", "models", "r1pro")
        if root and os.path.exists(os.path.join(path, "curobo", "r1pro_description_curobo_default.yaml")) and os.path.exists(os.path.join(path, "urdf", "r1pro.urdf")):
            return path
    return None


def test_the_table_holds_the_hands_lowest_sphere():
    """The hand's lowest collision sphere for a horizontal approach is in ``PALM_SPHERES_EEF`` + the wrist camera's: the closed fingers'
    and ``arm_link7``'s spheres (rigid with the hand too) stay above it at every flat turn (cuRobo yaml + URDF, both arms)."""
    root = _r1pro_assets()
    if root is None:
        pytest.skip("R1Pro cuRobo description / URDF not available")
    import yaml

    from ispatialgym.oracle.r1pro_model import _rpy_matrix

    with open(os.path.join(root, "curobo", "r1pro_description_curobo_default.yaml")) as f:
        spheres = yaml.safe_load(f)["robot_cfg"]["kinematics"]["collision_spheres"]
    joints = {j.find("child").get("link"): j for j in ET.parse(os.path.join(root, "urdf", "r1pro.urdf")).getroot().iter("joint") if j.find("child") is not None}

    def origin(link):
        o = joints[link].find("origin")
        return _rpy_matrix(*[float(v) for v in (o.get("rpy") or "0 0 0").split()]), np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])

    R_ge, t_ge = _rpy_matrix(*M.EEF_IN_GRIPPER_RPY), np.asarray(M.EEF_IN_GRIPPER_XYZ)
    for arm in ("left", "right"):
        grip, link7 = f"{arm}_gripper_link", f"{arm}_arm_link7"
        R7, t7 = origin(grip)  # the gripper in link7 (fixed)
        parts = {link7: lambda c: R7.T @ (c - t7)}
        for finger in M.FINGER_LINKS[arm]:  # finger joints at 0: closed
            Rf, tf = origin(finger)
            parts[finger] = lambda c, Rf=Rf, tf=tf: Rf @ c + tf
        rigid = [((to_grip(np.asarray(s["center"], dtype=float)) - t_ge) @ R_ge, float(s["radius"])) for link, to_grip in parts.items() for s in spheres[link]]
        for label, q in TT.flat_hand_quats((1.0, 0.0)):
            R = C.quat_to_mat(q).numpy()
            lowest_other = max(r - float(R[2] @ c) for c, r in rigid)
            assert lowest_other < TT.hand_drop_below(q) - 0.002, (arm, label, lowest_other)


def test_flat_turn_heights():
    s = _sampler()
    turns = s.flat_turns((0.0, 1.0))
    assert [label for _, _, label, _ in turns] == ["lateral", "vertical-", "lateral-", "vertical"]  # lowest fingertips first
    for tip, below, label, _ in turns:
        assert below == pytest.approx(DROPS[label], abs=1e-4), label
        assert tip == pytest.approx(below + IL.PUSH_FLAT_FLOOR_CLEARANCE_M), label  # the lowest sphere 1 cm above the floor
    assert _sampler(push_flat_floor_clearance_m=0.02).flat_turns((0.0, 1.0))[0][0] == pytest.approx(DROPS["lateral"] + 0.02, abs=1e-4)
    assert IL.PUSH_FLAT_FLOOR_CLEARANCE_M == 0.01 and 0.05 <= turns[0][0] and turns[-1][0] <= 0.10  # every turn's fingertips 5-10 cm up


def test_band_back_offset_meets_a_face_between_the_vertices():
    box = _box(0.33, 0.27, 0.12)
    # the fingertips' band 4.3-5.7 cm: no corner of the box lies in it, the plain point selection finds nothing
    assert IL.band_back_offset(box, (1.0, 0.0), 0.043, 0.057) is None
    assert IL.band_back_offset(box, (1.0, 0.0), 0.043, 0.057, section=True) == pytest.approx(0.165, abs=1e-6)
    assert IL.band_back_offset(box, (0.0, 1.0), 0.043, 0.057, section=True) == pytest.approx(0.135, abs=1e-6)
    assert IL.band_back_offset(box, (1.0, 0.0), 0.13, 0.15, section=True) is None  # above the box: nothing to touch
    # a wedge leaning back (the back face 0.15 behind the root at the bottom, 0.20 at the top 0.10 up): its section at the band's top
    wedge = th.tensor([[-0.15, -0.1, 0.0], [-0.15, 0.1, 0.0], [0.15, -0.1, 0.0], [0.15, 0.1, 0.0], [-0.20, -0.1, 0.10], [-0.20, 0.1, 0.10],
                       [0.15, -0.1, 0.10], [0.15, 0.1, 0.10]])
    assert IL.band_back_offset(wedge, (1.0, 0.0), 0.04, 0.06, section=True) == pytest.approx(0.18, abs=1e-6)
    # the plain selection (the blade's band, default) is unchanged
    assert IL.band_back_offset(box, (1.0, 0.0), 0.05, 0.13) == pytest.approx(0.165)


# ----------------------------------------------------------------------------------------------------------------
# the chain
# ----------------------------------------------------------------------------------------------------------------


def test_flat_chain_stand_off_poses_and_record():
    s = _sampler(push_under_hand="flat")
    ch = _push(s)
    assert ch is not None and len(s.ikc.calls) == 1  # one turn below the briefcase's centre of mass (6 cm): lateral, the camera up
    call = s.ikc.calls[0]
    assert call["ground"] is True and call["ignored"] == ["floors_jluwuz_0", "briefcase.n.01_1"] and call["fingers"] == C.FINGER_CLOSED
    so = TT.flat_hand_standoff(IL.PUSH_UNDER_CLEARANCE_M)
    assert so == pytest.approx(C.FINGERTIP_LENGTH + 0.005) == pytest.approx(ch["blade_standoff_m"])
    tip = DROPS["lateral"] + IL.PUSH_FLAT_FLOOR_CLEARANCE_M
    start, mid, goal_p, pre = ch["start"][0], ch["mid"][0], ch["goal"][0], ch["pre"][0]
    assert start.tolist() == pytest.approx([1.0 - 0.165 - so, 0.0, tip], abs=1e-4)  # the eef origin the fingertip length + 5 mm behind the face
    assert float(C.fingertip(ch["start"])[0]) == pytest.approx(1.0 - 0.165 - 0.005, abs=1e-4)  # the tips 5 mm short of the back face
    assert goal_p.tolist() == pytest.approx([1.40 - 0.165 - so + 0.03, 0.0, tip], abs=1e-4)  # push_goal_extra_m past the goal cell
    assert mid.tolist() == pytest.approx(((start + goal_p) / 2).tolist(), abs=1e-6)
    assert pre.tolist() == pytest.approx((start - th.tensor([0.10, 0.0, 0.0])).tolist(), abs=1e-5)  # backed off horizontally at push height
    assert th.allclose(C.approach_axis(ch["start"][1]), th.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert ch["pitch_deg"] == 90.0 and ch["ground_ignored"] is False and ch["wrist_sign"] == "lateral" and "palm_contact" not in ch
    assert ch["half_extent_m"] == pytest.approx(0.165, abs=1e-6) and ch["push_height_m"] == ch["fingertip_z_m"] == pytest.approx(tip, abs=1e-4)
    rec_x = ch["record"]
    assert rec_x["push_hand"] == "flat" and rec_x["palm_clear"] is True and rec_x["wrist_turn"] == "lateral" and rec_x["com_height_m"] == pytest.approx(0.06)
    assert rec_x["fingertip_band_m"] == pytest.approx([tip - 0.007, tip + 0.007], abs=1e-4) and rec_x["palm_drop_m"] == pytest.approx(DROPS["lateral"], abs=1e-4)
    # the record as the C loop writes it: push_hand / palm_clear reach the oracle's PushContact (its palm clearance skips palm_clear)
    rec = IL.PushUnderRecord("right", ch["pre"], ch["start"], ch["mid"], ch["goal"], ch["direction"], ch["distance_m"], "briefcase.n.01_1", stance="push",
                             pitch_deg=ch["pitch_deg"], blade_standoff_m=ch["blade_standoff_m"], push_height_m=ch["push_height_m"], half_extent_m=ch["half_extent_m"],
                             fingertip_z_m=ch["fingertip_z_m"], ground_ignored=ch["ground_ignored"], drop_cell=[1.0, 0.0], goal_cell=[1.4, 0.0],
                             **IL.push_under_palm_flags(ch), **ch["record"])
    pc = SC.PushContact.from_dict(rec.to_sidecar(), "contacts.push")
    assert pc.kind == "push_under" and pc.extra["push_hand"] == "flat" and pc.extra["palm_clear"] is True and pc.extra["pitch_deg"] == 90.0
    assert "palm_contact" not in pc.extra and pc.extra["ground_ignored"] is False


def test_flat_chain_turn_order_and_the_centre_of_mass_rule():
    # a taller object (com 10 cm up): every turn is below it, the generator's order is the fingertips' height
    s = _sampler(height=0.28, com_z=0.10, push_under_hand="flat")
    s.ikc = _IKC(ok=lambda chain: False)
    assert _push(s) is None
    heights = [round(float(c["poses"][1][0][2]), 4) for c in s.ikc.calls]
    assert heights == pytest.approx([DROPS[k] + 0.01 for k in ("lateral", "vertical-", "lateral-", "vertical")], abs=1e-4)
    assert all(c["ground"] is True for c in s.ikc.calls)  # the floor stays in the IK world at every height
    # the second turn when the first fails
    s.ikc = _IKC(ok=lambda chain: float(chain[1][0][2]) > 0.06)
    ch = _push(s)
    assert ch["wrist_sign"] == "vertical-" and ch["fingertip_z_m"] == pytest.approx(DROPS["vertical-"] + 0.01, abs=1e-4)
    # no turn below the centre of mass (4.5 cm): no chain and no IK
    s = _sampler(com_z=0.045, push_under_hand="flat")
    assert _push(s) is None and s.ikc.calls == []
    assert s.com_height(G.IDENTITY_QUAT.clone()) == pytest.approx(0.045)


def test_blade_stays_the_default_and_flat_takes_precedence_over_the_palm():
    # under_large #02's evidence: a 13 cm briefcase's face reaches the blade's palm at every pitch - no chain, no IK; the flat hand pushes it
    s = _sampler(height=0.13, com_z=0.065)
    assert s.push_hand == "blade" and _push(s) is None and s.ikc.calls == []
    z_lo = s.push_height(30.0)[1]
    assert s.push_contact_offset(G.IDENTITY_QUAT.clone(), (1.0, 0.0)) == s.contact_offset_along(G.IDENTITY_QUAT.clone(), (1.0, 0.0), z_lo, z_lo + s.BLADE_SPAN_M)
    s.gp = _GP(push_under_hand="flat")
    ch = _push(s)
    assert ch is not None and ch["record"]["push_hand"] == "flat"
    assert s.push_contact_offset(G.IDENTITY_QUAT.clone(), (1.0, 0.0)) == pytest.approx(0.165, abs=1e-6)  # the tips' band, through the faces
    # both keys on (115 / 117 / 118): the flat hand, the palm blade not tried
    s.gp, s.ikc = _GP(push_under_hand="flat", push_under_palm_contact=True), _IKC()
    ch = _push(s)
    assert ch["pitch_deg"] == 90.0 and ch["record"]["push_hand"] == "flat" and "palm_contact" not in ch and IL.push_under_palm_flags(ch) == {}
    assert [round(float(C.approach_axis(c["poses"][1][1])[2]), 6) for c in s.ikc.calls] == [0.0]  # one horizontal approach, no vertical blade
    s.gp = _GP(push_under_hand="palm")
    with pytest.raises(IL.Reject):
        s.push_hand
