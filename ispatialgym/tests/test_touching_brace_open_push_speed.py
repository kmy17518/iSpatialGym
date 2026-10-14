"""IORS touching: the open-finger brace on a wide partner face and the push speed (owner 2026-10-14 ~02:04, after the touching#02
pair1013_c8_k8 replay: "left brace should be open gripper like touching 1 because it's wide (so make this situational key for the touching
tasks) and right gripper should push more gently(slowly) - slow push should be parameter of pushes - so that the cereal box doesn't wobble
that much"). ``generation.brace_open_when_wide``: the braced face's width from the partner hull's horizontal cut, the open-finger stop
required on a wide face (no closed-tip fallback, the draw rejected without one), today's candidates on a narrow face. ``generation.push_speed_m_s``:
recorded on the touching push, the oracle's pad speed for every leg of that push. c8_k8's partner (box_of_cereal/dkhgxn's 56 collision
points, pairgoal probe 17859694) at its recorded poses; the oracle on the fake worlds of ``test_v2_oracle`` / ``test_push_under_strokes``.
No simulator."""

from __future__ import annotations

import contextlib
import json
import math
import os
import types
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.stances import EdgeFrame, SidecarDraft, Stance

CUSTOMS = os.path.join(os.path.dirname(__file__), "..", "assets", "task_specs", "isg_customs.json")
BAR = "inter_object_relation_satisfaction-touching-box_of_cereal_s080-box_of_cereal-bar"

#: box_of_cereal/dkhgxn's collision boundary points (its own frame, scale 1; the sampler's local_hull), 0.1 mm
DKHGXN = th.tensor([
    (-0.0345, -0.0728, 0.1190), (-0.0344, -0.0731, -0.1205), (-0.0342, -0.0731, 0.1195), (0.0330, -0.0732, 0.1194),
    (0.0334, -0.0731, -0.1212), (-0.0344, 0.0767, -0.1212), (-0.0345, -0.0727, -0.1212), (-0.0344, 0.0767, 0.1195),
    (-0.0339, -0.0108, 0.1199), (-0.0347, 0.0440, -0.1210), (0.0329, 0.0761, 0.1194), (0.0184, 0.0757, -0.1215),
    (0.0184, -0.0722, -0.1215), (0.0259, -0.0019, -0.1215), (0.0334, 0.0767, -0.1212), (0.0335, 0.0767, 0.1189),
    (0.0335, -0.0727, 0.1189), (-0.0174, -0.0719, 0.1202), (-0.0174, 0.0757, 0.1202), (0.0235, -0.0436, 0.1199),
    (-0.0176, 0.0760, -0.1215), (-0.0176, -0.0722, -0.1215), (-0.0345, -0.0735, -0.0905), (0.0331, -0.0737, -0.0895),
    (0.0329, -0.0737, 0.0887), (-0.0392, 0.0076, -0.0599), (-0.0389, -0.0008, 0.0287), (-0.0355, 0.0767, 0.0910),
    (-0.0352, 0.0768, -0.0641), (-0.0382, -0.0167, -0.0305), (-0.0382, -0.0167, 0.0290), (0.0363, 0.0528, -0.0601),
    (0.0361, -0.0537, -0.0601), (0.0331, 0.0769, -0.0093), (0.0339, 0.0764, 0.0876), (0.0337, 0.0765, -0.0605),
    (0.0338, -0.0730, -0.0601), (-0.0116, -0.0733, 0.1192), (0.0106, -0.0746, 0.0887), (-0.0117, -0.0758, -0.0004),
    (-0.0117, -0.0750, -0.0895), (0.0106, -0.0750, -0.0894), (-0.0117, -0.0761, -0.0301), (0.0102, 0.0770, 0.1189),
    (-0.0116, 0.0779, 0.0880), (0.0107, 0.0779, 0.0880), (-0.0117, 0.0788, 0.0287), (0.0107, 0.0787, 0.0287),
    (0.0245, 0.0577, 0.1198), (-0.0349, 0.0562, 0.1190), (-0.0378, 0.0563, -0.0900), (-0.0395, 0.0562, -0.0602),
    (-0.0402, 0.0360, -0.0602), (0.0373, -0.0181, -0.0601), (0.0375, 0.0173, -0.0601), (0.0382, -0.0181, -0.0304),
], dtype=th.float32)


def _box(dx, dy, dz):
    return th.tensor([[sx * dx / 2, sy * dy / 2, sz * dz / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=th.float32)


#: box_of_cereal/iucshm at x0.8 (its collision box)
IUCSHM_S080 = _box(0.0427, 0.1567, 0.2182)
#: pair1013_c8_k8: the partner's init pose (tro), the target's goal (pose target), the push direction and the gap at the place
P_POSE = (th.tensor([6.010707855224609, -0.006311286240816116, 1.1717959642410278]),
          th.tensor([-6.771671905880794e-05, 7.710242789471522e-05, 0.9998342990875244, 0.018206704407930374]))
T_GOAL = (th.tensor([6.068144798278809, -0.009876023977994919, 1.1596795320510864]),
          th.tensor([-2.7509699066285975e-05, 0.0007938016788102686, -0.006294335704296827, 0.9999799728393555]))
DIRECTION = th.tensor([-0.9999967813491821, 0.0025431327521800995, 0.0])
T_PLACE = (T_GOAL[0] - DIRECTION * 0.0493, T_GOAL[1])
#: the B3 edge of bar_udatjt_0 as the push stance saw it: the robot faces +y, its left (the push direction) is -x
EDGE = EdgeFrame("B3", th.tensor([6.0, -0.30]), th.tensor([0.0025431327521800995, 0.9999967813491821]), 1.5)


def _iors():
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    return IS


def _hull_at(hull, pose):
    return C.transform_points((th.as_tensor(pose[0]).float(), th.as_tensor(pose[1]).float()), hull)


def _stance() -> Stance:
    st = Stance("push", (6.0199, -0.7451, math.pi / 2), 0.25, -0.27, 0.45, EDGE, lateral=0.0)
    st.arms = ["right", "left"]
    return st


def _sampler(gen=None, partner=None, **extra):
    """An ``IORSSmall`` stand-in with c8_k8's two hulls (``partner`` replaces dkhgxn's), the generation ``gen`` and the touching methods."""
    IS = _iors()
    g = dict(gen or {})
    fields = dict(shelf_start=None, relation="touching", hull={"target": IUCSHM_S080, "partner": DKHGXN if partner is None else partner},
                  world=SimpleNamespace(hull_at=_hull_at), gen=lambda key, default=None: g.get(key, default), tol=lambda key, default=None: default,
                  pair={"g_NT": 0.0584}, gp=SimpleNamespace(alias=BAR))
    fields.update(extra)
    s = SimpleNamespace(**fields)
    for name in ("touching_points", "touching_brace_candidates", "wide_stop_spread", "brace_tip_heights", "brace_open_rule", "_open_brace_candidates",
                 "brace_open_when_wide", "push_speed", "push_distance", "push_margin", "place_gap", "pair_goal_align", "touching_contacts"):
        setattr(s, name, types.MethodType(getattr(IS.IORSSmall, name), s))
    return s


def _brace_inputs(s, p_pose=P_POSE, t_pose=T_PLACE):
    st = _stance()
    p_init, b_point = s.touching_points(t_pose, p_pose, st)
    p_hull = _hull_at(s.hull["partner"], p_pose)
    face_z = (float(p_hull[:, 2].min()), float(p_hull[:, 2].max()))
    return st, p_init, b_point, p_hull, face_z


def _side_h():
    a = C.unit(-DIRECTION)
    return C.unit(th.cross(th.tensor([0.0, 0.0, 1.0]), -a, dim=0))


# ----------------------------------------------------------------------------------------------------------------
# the face's width: the hull's cut, not its vertices
# ----------------------------------------------------------------------------------------------------------------


def test_the_face_run_comes_from_the_hulls_cut_where_the_vertex_band_sees_nothing():
    IS = _iors()
    box = _box(0.0784, 0.1548, 0.2417)  # dkhgxn's extents as a plain 8-corner hull
    outward, side_h = th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 1.0, 0.0])
    cut = IS.hull_cut_points(box, 0.0)
    assert len(cut) > 0 and float(cut[:, 0].max()) == pytest.approx(0.0392, abs=1e-6) and float(cut[:, 1].min()) == pytest.approx(-0.0774, abs=1e-6)
    assert IS.face_run_at(box, 0.0, outward, side_h) == pytest.approx((-0.0774, 0.0774), abs=1e-6)
    assert IS.face_run_at(box, 0.13, outward, side_h) is None and len(IS.hull_cut_points(box, -0.2)) == 0  # above / below the hull
    # the wide stop's rule reads the vertices within 3 cm of the brace height: a box has none at mid height
    s = _sampler()
    assert s.wide_stop_spread(box, th.tensor([0.0392, 0.0, 0.0]), outward, side_h, 0.0) is None
    # a face turned against the push keeps a short run only (the cut's points within 1 cm of the outermost)
    q = C.quat_from_yaw(math.radians(30.0))
    turned = C.transform_points((th.zeros(3), q), box)
    lo, hi = IS.face_run_at(turned, 0.0, outward, side_h)
    assert hi - lo < 0.05


def test_c8_k8_had_no_wide_stop_and_the_key_requires_the_9cm_open_stop():
    """pair1013_c8_k8 braced dkhgxn with the closed tips (brace_tip_h-, orient reason reach): the wide stop's rule saw two hull points on its
    15.5 cm face within 3 cm of the brace height, 2.3 cm left / 7.2 cm right of the push line - no 5-9 cm spread fits. The cut measures the
    face at 0.151 m with the push line 2.6 mm off its centre: the 9 cm open stop."""
    s = _sampler({"brace_open_when_wide": True})
    st, p_init, b_point, p_hull, face_z = _brace_inputs(s)
    assert float(b_point[2]) == pytest.approx(1.1711, abs=2e-4)  # the recorded brace contact height (contact_z)
    outward, side_h = C.unit(DIRECTION), _side_h()
    s_push = float(p_init[:2] @ side_h[:2])
    assert s.wide_stop_spread(p_hull, b_point, outward, side_h, s_push, spreads=[0.09, 0.07, 0.05]) is None
    rule = s.brace_open_rule(b_point, face_z, -DIRECTION, p_hull, p_init)
    assert rule["wide"] and rule["required"] and rule["spread_m"] == 0.09 and rule["min_width_m"] == pytest.approx(0.07)
    assert rule["face_width_m"] == pytest.approx(0.151, abs=0.002) and abs(rule["push_line_from_face_centre_m"]) < 0.005
    assert rule["height_m"] == pytest.approx(1.1711, abs=2e-4)
    # without the key: no rule, and today's stream starts with the closed tips (the chosen class of c8_k8)
    s0 = _sampler()
    assert s0.brace_open_rule(b_point, face_z, -DIRECTION, p_hull, p_init) is None
    kinds0 = [k for _pre, _c, k, _e in s0.touching_brace_candidates(st, b_point, face_z, -DIRECTION, partner_hull_w=p_hull, push_point=p_init)]
    assert kinds0[:2] == ["brace_tip_h-", "brace_tip_h"] and not any(k.startswith("brace_wide") for k in kinds0)


def test_with_the_key_a_wide_face_is_braced_by_the_open_stop_only():
    s = _sampler({"brace_open_when_wide": True})
    st, p_init, b_point, p_hull, face_z = _brace_inputs(s)
    cands = list(s.touching_brace_candidates(st, b_point, face_z, -DIRECTION, partner_hull_w=p_hull, push_point=p_init))
    assert [k for _p, _c, k, _e in cands] == ["brace_wide_h-", "brace_wide_h", "brace_wide_h-_lo", "brace_wide_h_lo"]  # no closed tips, blades, pads
    side_h = _side_h()
    s_push = float(p_init[:2] @ side_h[:2])
    for pre, contact, kind, ex in cands:
        assert ex["fingers_opening_m"] == pytest.approx(0.0315) and ex["stop_width_m"] == 0.09 and ex["pose_class"] == kind
        assert ex["open_when_wide"]["required"] is True and ex["open_when_wide"]["spread_m"] == 0.09
        tip = C.fingertip((contact[0], contact[1]))
        assert float(tip[:2] @ side_h[:2]) == pytest.approx(s_push, abs=1e-4)  # the two open tips straddle the push line
        assert th.allclose(C.approach_axis(contact[1]), C.unit(-DIRECTION), atol=1e-5)  # horizontal, into the far face
        assert float(th.linalg.norm(pre[0] - contact[0])) == pytest.approx(0.05, abs=1e-5)
        # the fingertips 5 mm before the far face
        assert float((tip[:2] - b_point[:2]) @ C.unit(-DIRECTION)[:2]) == pytest.approx(-0.005, abs=1e-4)
    # the camera-up turn is today's wide-stop pose (fingertip_eef with -h), the other turn its wrist twin; the lower height's z
    (_, c0, _, e0), (_, c1, _, _), (_, c2, _, e2), _ = cands
    a = C.unit(-DIRECTION)
    want = C.fingertip_eef(th.tensor([float(b_point[0]), float(b_point[1]), e0["contact_z"]]) + side_h * e0["push_line_side_m"], a, -C.default_closing_axis(a), 0.005)
    assert th.allclose(c0[0], want[0], atol=1e-5) and abs(abs(float(th.dot(c0[1], want[1]))) - 1.0) < 1e-6
    assert th.allclose(c0[0], c1[0], atol=1e-6) and C.quat_angle(c0[1], c1[1]) == pytest.approx(math.pi, abs=1e-4)
    assert e2["contact_z"] == pytest.approx(face_z[0] + 0.075, abs=1e-4) and e0["contact_z"] > e2["contact_z"]


def test_a_narrow_face_keeps_todays_candidates():
    # the partner turned a quarter: its 7.8 cm face toward the push is still wide (>= 7 cm); a 6 cm box is narrow
    narrow = _box(0.06, 0.06, 0.2417)
    s = _sampler({"brace_open_when_wide": True}, partner=narrow)
    st, p_init, b_point, p_hull, face_z = _brace_inputs(s)
    rule = s.brace_open_rule(b_point, face_z, -DIRECTION, p_hull, p_init)
    assert rule["wide"] is False and rule["required"] is False and rule["face_width_m"] == pytest.approx(0.06, abs=1e-3)
    kinds = [k for _p, _c, k, _e in s.touching_brace_candidates(st, b_point, face_z, -DIRECTION, partner_hull_w=p_hull, push_point=p_init)]
    kinds0 = [k for _p, _c, k, _e in _sampler(partner=narrow).touching_brace_candidates(st, b_point, face_z, -DIRECTION, partner_hull_w=p_hull, push_point=p_init)]
    assert kinds == kinds0 and kinds[0] == "brace_tip_h-"
    # the partner turned a quarter: its 7.84 cm face is wide (>= 7 cm) and takes the 5 cm spread only with the push line within
    # 7.84 / 2 - 1 - 2.5 = 0.42 cm of its centre
    turned = _sampler({"brace_open_when_wide": True}, partner=_box(0.1548, 0.0784, 0.2417))
    st, p_init, b_point, p_hull, face_z = _brace_inputs(turned)
    r2 = turned.brace_open_rule(b_point, face_z, -DIRECTION, p_hull, p_init)
    off = r2["push_line_from_face_centre_m"]
    assert r2["required"] and r2["face_width_m"] == pytest.approx(0.0784, abs=1e-3) and abs(off) > 0.0042 and r2["spread_m"] is None
    side_h = _side_h()
    centred = (P_POSE[0] + th.tensor([float(side_h[0]), float(side_h[1]), 0.0]) * off, P_POSE[1])
    st, p_init, b_point, p_hull, face_z = _brace_inputs(turned, p_pose=centred)
    r3 = turned.brace_open_rule(b_point, face_z, -DIRECTION, p_hull, p_init)
    assert abs(r3["push_line_from_face_centre_m"]) < 1e-4 and r3["spread_m"] == 0.05


def test_a_wide_face_whose_push_line_is_near_its_end_has_no_open_stop():
    s = _sampler({"brace_open_when_wide": True})
    # the target 6 cm along the face: the push line 6.3 cm from the face's centre (its half width 7.6 cm)
    side_h = _side_h()
    t_off = (T_PLACE[0] + th.tensor([float(side_h[0]), float(side_h[1]), 0.0]) * 0.06, T_PLACE[1])
    st, p_init, b_point, p_hull, face_z = _brace_inputs(s, t_pose=t_off)
    rule = s.brace_open_rule(b_point, face_z, -DIRECTION, p_hull, p_init)
    assert rule["required"] and rule["spread_m"] is None and abs(rule["push_line_from_face_centre_m"]) > 0.05
    assert list(s.touching_brace_candidates(st, b_point, face_z, -DIRECTION, partner_hull_w=p_hull, push_point=p_init)) == []


def test_the_keys_are_validated():
    assert _sampler().brace_open_when_wide() is False and _sampler().push_speed() is None
    assert _sampler({"brace_open_when_wide": False}).brace_open_when_wide() is False
    assert _sampler({"brace_open_when_wide": True}).brace_open_when_wide() is True
    assert _sampler({"push_speed_m_s": 0.02}).push_speed() == pytest.approx(0.02)
    for bad in ({"brace_open_when_wide": "yes"}, {"brace_open_when_wide": 1}):
        with pytest.raises(Reject) as e:
            _sampler(bad).brace_open_when_wide()
        assert e.value.reason == "spec"
    with pytest.raises(Reject):
        _sampler({"brace_open_when_wide": True}, relation="nextto").brace_open_when_wide()
    with pytest.raises(Reject):
        _sampler({"brace_open_when_wide": True}, shelf_start={"lean_side": 1.0, "lean_deg": 10.0}).brace_open_when_wide()
    for v in (0.0, -0.02, "slow", float("nan")):
        with pytest.raises(Reject) as e:
            _sampler({"push_speed_m_s": v}).push_speed()
        assert e.value.reason == "spec"
    with pytest.raises(Reject):
        _sampler({"push_speed_m_s": 0.02}, relation="ontop").push_speed()


# ----------------------------------------------------------------------------------------------------------------
# touching_contacts: the record and the rejection
# ----------------------------------------------------------------------------------------------------------------


def _contacts_sampler(gen, brace_choice, gate=None):
    tname, pname = "box__of__cereal.n.01_1", "box__of__cereal.n.01_2"
    partner = SimpleNamespace(get_position_orientation=lambda: (P_POSE[0].clone(), P_POSE[1].clone()))

    def push_choice(st, t_pose_, p_init, t_face_z, direction, dist, ignore_push):
        init = C.fingertip_eef(p_init, direction, -C.default_closing_axis(direction), 0.005)
        chain = [C.pre_pose(init, 0.05), init, (init[0] + direction * dist, init[1])]
        return SimpleNamespace(proposal=SimpleNamespace(payload=("push_tip_h-", init, {"tip_front": True, "pre_dist_m": 0.05, "stop_clearance_m": 0.005}),
                                                        chain=chain)), None, []

    s = _sampler(gen, bindings=SimpleNamespace(one=lambda role: SimpleNamespace(name=tname if role == "target" else pname)), _touching_place_pos=T_PLACE[0],
                 obj=lambda role: partner if role == "partner" else SimpleNamespace(), support_objects=lambda: [], posed=lambda role, pose: contextlib.nullcontext(),
                 _brace_gate=gate or {"ok": True}, sidecar=SidecarDraft(BAR, "inter_object_relation_satisfaction"))
    s._touching_push_choice = push_choice
    s._touching_brace_choice = brace_choice
    from ispatialgym.sampling.tasks.base import Draft

    draft = Draft()
    draft.goal_poses[tname] = T_GOAL
    draft.pose_targets[tname] = {"position": [float(v) for v in T_GOAL[0]]}
    return s, draft


def test_touching_contacts_records_the_open_stop_and_the_push_speed_for_the_oracle():
    from ispatialgym.oracle import execute as EX
    from ispatialgym.oracle.sidecar import BraceContact, PushContact

    def brace_choice(st, push_chain, t_pose, p_pose, b_point, face_z, direction, dist, p_hull, p_init, ignore_push, ignore_brace, **kw):
        first = next(iter(s.touching_brace_candidates(st, b_point, face_z, -direction, partner_hull_w=p_hull, push_point=p_init)))
        return SimpleNamespace(proposal=SimpleNamespace(payload=first)), None, []

    s, draft = _contacts_sampler({"brace_open_when_wide": True, "push_speed_m_s": 0.02}, lambda *a, **k: brace_choice(*a, **k))
    st = _stance()
    s.touching_contacts(draft, st)
    d = s.sidecar.to_dict()["contacts"]
    push, brace = PushContact.from_dict(d["push"], "contacts.push"), BraceContact.from_dict(d["brace"], "contacts.brace")
    assert push.extra["push_speed_m_s"] == pytest.approx(0.02) and EX.push_speed_of(push) == pytest.approx(0.02)
    assert brace.kind == "brace_wide_h-" and brace.extra["fingers_opening_m"] == pytest.approx(0.0315) and brace.extra["stop_width_m"] == 0.09
    tf, br = draft.summary["touching"], draft.summary["brace"]
    assert tf["push_speed_m_s"] == pytest.approx(0.02) and tf["brace_open_when_wide"]["required"] is True
    assert br["present"] is True and br["kind"] == "brace_wide_h-" and br["open_when_wide"]["face_width_m"] == pytest.approx(0.151, abs=0.002)
    assert st.arms == ["right", "left"] and st.contacts == ["push", "brace"]
    # without the keys: no speed recorded, no rule (today's record)
    s0, draft0 = _contacts_sampler({}, lambda *a, **k: (None, None, ["brace_tip_h-: left (brace): exact IK failed"]))
    s0.touching_contacts(draft0, _stance())
    p0 = PushContact.from_dict(s0.sidecar.to_dict()["contacts"]["push"], "contacts.push")
    assert "push_speed_m_s" not in p0.extra and EX.push_speed_of(p0) is None
    assert "brace_open_when_wide" not in draft0.summary["touching"] and draft0.summary["brace"]["present"] is False


def test_touching_contacts_rejects_a_wide_face_without_the_open_stop():
    fails = ["brace_wide_h-: left (brace): exact IK failed at pose 2 of 2", "brace_wide_h: wrist camera within 5 mm of the world"]
    s, draft = _contacts_sampler({"brace_open_when_wide": True}, lambda *a, **k: (None, None, list(fails)))
    with pytest.raises(Reject) as e:
        s.touching_contacts(draft, _stance())
    assert e.value.reason == "brace_open_required"
    msg = str(e.value)
    assert "0.15" in msg and "requires the open-finger stop" in msg and "exact IK failed" in msg
    assert "push" not in s.sidecar.contacts  # nothing recorded
    # the joint band gate said no: the same rejection (no push-only touching on a wide face)
    s2, draft2 = _contacts_sampler({"brace_open_when_wide": True}, lambda *a, **k: pytest.fail("no brace search without the gate"),
                                   gate={"ok": False, "why": "brace OFF left band at push"})
    with pytest.raises(Reject) as e2:
        s2.touching_contacts(draft2, _stance())
    assert e2.value.reason == "brace_open_required" and "OFF left band" in str(e2.value)
    # a narrow face with the key: push-only as today
    s3, draft3 = _contacts_sampler({"brace_open_when_wide": True}, lambda *a, **k: (None, None, ["brace_tip_h-: x"]))
    s3.hull = {"target": IUCSHM_S080, "partner": _box(0.06, 0.06, 0.2417)}
    s3.touching_contacts(draft3, _stance())
    assert draft3.summary["brace"]["present"] is False and draft3.summary["touching"]["brace_open_when_wide"]["required"] is False


def test_customs_set_the_keys_on_touching02s_bar_alias_only():
    with open(CUSTOMS) as f:
        customs = json.load(f)
    gen = customs[BAR]["generation"]
    assert gen["brace_open_when_wide"] is True and gen["push_speed_m_s"] == pytest.approx(0.02)
    assert gen["widest_faces"] == "parallel" and gen["touching_place_gap_m"] == 0.05 and gen["pair_goal_align"]["faces"] == "widest"
    with_open = [a for a, v in customs.items() if isinstance(v, dict) and "brace_open_when_wide" in (v.get("generation") or {})]
    assert with_open == [BAR]
    with_speed = [a for a, v in customs.items() if isinstance(v, dict) and "push_speed_m_s" in (v.get("generation") or {})]
    assert set(with_speed) - {BAR} == {a for a in with_speed if "nextto_large_scale" in a}  # the large-scale base push's own knob


# ----------------------------------------------------------------------------------------------------------------
# the oracle: the recorded speed on every pad leg
# ----------------------------------------------------------------------------------------------------------------


def test_push_speed_of_reads_the_record():
    from ispatialgym.oracle import execute as EX

    for extra, want in (({"push_speed_m_s": 0.02}, 0.02), ({"push_speed_m_s": "0.03"}, 0.03), ({}, None), ({"push_speed_m_s": None}, None),
                        ({"push_speed_m_s": 0.0}, None), ({"push_speed_m_s": -0.1}, None), ({"push_speed_m_s": "slow"}, None)):
        got = EX.push_speed_of(SimpleNamespace(extra=extra))
        assert (got is None and want is None) or got == pytest.approx(want)
    assert EX.push_speed_of(SimpleNamespace()) is None
    assert EX._speed_kw({}) == {} and EX._speed_kw({"push_speed_m_s": 0.02}) == {"speed": 0.02}


def test_push_legs_run_at_the_recorded_speed():
    from ispatialgym.oracle import execute as EX

    seen = []

    class _Motions:
        def arm_along(self, arm, waypoints, loaded=False, partial=False, speed=None, **kw):
            seen.append(speed)
            return iter(())

    cfg = EX.ExecuteConfig()
    line, arc = SimpleNamespace(kind="line"), SimpleNamespace(kind="arc", angle=0.5)
    art = {"joint": "j_lid", "q_from": 0.8, "q_to": 0.2, "tolerance": 0.15}
    for geom, ctx in ((line, {}), (line, {"push_speed_m_s": 0.02}), (arc, {}), (arc, {"push_speed_m_s": 0.02}),
                      (arc, {"push_articulation": art}), (arc, {"push_articulation": art, "push_speed_m_s": 0.02})):
        list(EX._push_leg(_Motions(), SimpleNamespace(), "right", "box.n.01_1", [], geom, None, cfg, dict(ctx), "push"))
    assert seen == [cfg.push_speed_line, 0.02, cfg.push_speed_arc, 0.02, None, 0.02]
    assert cfg.push_speed_line is None  # a straight push without the record keeps the arm's Cartesian speed


def test_the_base_nudge_is_capped_at_the_recorded_speed():
    from ispatialgym.oracle import execute as EX
    from ispatialgym.oracle.base_tracker import BaseTrackerConfig

    seen = []

    class _Motions:
        def __init__(self):
            self.robot = SimpleNamespace(get_joint_positions=lambda: th.zeros(28), arm_control_idx={"right": list(range(7))})
            self.state = SimpleNamespace(arm_target={})
            self.cfg = SimpleNamespace(base_tracker=BaseTrackerConfig())

        def track_path(self, wps, tracker_cfg=None, **kw):
            seen.append(tuple(tracker_cfg.v_max))
            return iter(())

    m = _Motions()
    cfg = EX.ExecuteConfig()
    state = SimpleNamespace(base_pose2d=lambda: (0.0, 0.0, 0.0))
    for speed in (None, 0.02, 0.5):
        list(EX._base_nudge(m, state, "right", "box.n.01_1", [1.0, 0.0, 0.0], 0.01, cfg, "push", speed=speed))
    v = cfg.push_base_nudge_speed
    assert seen == [(v, v, 0.2), (0.02, 0.02, 0.2), (v, v, 0.2)]


def test_a_straight_push_with_a_recorded_speed_moves_the_pad_at_it(monkeypatch):
    """End to end (``execute_push`` on the push-under fake world): with ``push_speed_m_s`` every loaded pad leg - the push from the lowered
    pre-contact, the back-off, a rock-back round's return - is streamed at it and the pad never moves faster; without it the legs keep the
    arm's speed (0.10 m/s). The object reaches its goal either way."""
    from ispatialgym.oracle import execute as EX
    from ispatialgym.oracle.motions import Motions
    from ispatialgym.tests.test_push_under_strokes import push_action, run_push, sandal_world

    real = Motions.arm_along
    nudges = []
    real_nudge = EX._base_nudge

    def run(extra):
        legs = []

        def arm_along(self, arm, waypoints, loaded=False, **kw):
            rec = {"loaded": loaded, "speed": kw.get("speed"), "steps": []}
            legs.append(rec)
            prev = C.as_tensor(self.eef_pose(arm)[0]).float().clone()
            for a in real(self, arm, waypoints, loaded=loaded, **kw):
                yield a
                p = C.as_tensor(self.eef_pose(arm)[0]).float().clone()
                rec["steps"].append(float(th.linalg.norm(p - prev)))
                prev = p

        def base_nudge(*a, **k):
            nudges.append(k.get("speed"))
            return (yield from real_nudge(*a, **k))

        monkeypatch.setattr(Motions, "arm_along", arm_along)
        monkeypatch.setattr(EX, "_base_nudge", base_nudge)
        env, robot, sandal, goal, _ = sandal_world(yaw_deg=0.0, off_m=0.0)
        ctx, motions = run_push(env, robot, push_action(extra))
        return ctx, motions, legs, sandal, goal

    ctx, motions, legs, sandal, goal = run({"push_speed_m_s": 0.02})
    loaded = [leg for leg in legs if leg["loaded"]]
    assert ctx["push_speed_m_s"] == 0.02 and loaded and all(leg["speed"] == 0.02 for leg in loaded)
    fastest = max(max(leg["steps"] or [0.0]) for leg in loaded)
    assert fastest <= 0.02 * motions.dt * 1.55  # a segment of < 1.5 steps' travel is streamed in one step
    assert float(goal[0] - sandal.pos[0]) < 0.01
    assert all(sp == 0.02 for sp in nudges)
    n_slow = sum(len(leg["steps"]) for leg in loaded)
    nudges.clear()
    ctx0, motions0, legs0, sandal0, goal0 = run({})
    loaded0 = [leg for leg in legs0 if leg["loaded"]]
    assert "push_speed_m_s" not in ctx0 and all(leg["speed"] is None for leg in loaded0) and all(sp is None for sp in nudges)
    assert max(max(leg["steps"] or [0.0]) for leg in loaded0) > 0.02 * motions0.dt * 2.0  # the arm's 0.10 m/s
    assert float(goal0[0] - sandal0.pos[0]) < 0.01
    assert n_slow > 2 * sum(len(leg["steps"]) for leg in loaded0)


def test_an_arc_push_with_a_recorded_speed_runs_its_closed_loop_at_it():
    from typing import Any, Dict

    from ispatialgym.oracle import execute as EX
    from ispatialgym.oracle.executor import SimState
    from ispatialgym.oracle.motions import Motions, MotionsConfig
    from ispatialgym.tests.test_v2_oracle import _FkBackend, axial_world

    steps = {}
    for speed in (None, 0.02):
        sc, env, actions, board, brace, push, geom = axial_world(gain=1.0, tol=0.10)
        if speed is not None:
            push.contact.extra["push_speed_m_s"] = speed
        motions = Motions(env, _FkBackend(env.robot), MotionsConfig(settle_steps=5, gripper_steps=5, cartesian_step=0.05, cartesian_rot_step=0.3))
        state = SimState(env, env.task, env.robot)
        cfg = EX.ExecuteConfig(push_arc_tracking="closed_loop", push_backoff_m=0.02)
        ctx: Dict[str, Any] = {"cfg": cfg, "witnesses": {}, "hooks": {}}
        targets = push.targets(sc, state)
        for a in EX.run_action(push, motions, state, targets, ctx):
            env.step(a)
        cl = ctx["push_arc_closed_loop"]
        assert cl["speed_m_s"] == (cfg.push_speed_arc if speed is None else speed)
        assert C.quat_angle(env.task._tro.meta.pose_targets[push.contact.object].orientation, board.quat) < 0.10
        steps[speed] = cl["steps"]
    assert steps[0.02] > steps[None]  # 0.02 < push_speed_arc 0.035: more control steps for the same arc
