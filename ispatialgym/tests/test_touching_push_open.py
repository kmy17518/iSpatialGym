"""Touching small scale, ``generation.push_gripper_open`` (oracle-fix 2026-11-06, owner on touching#02's cereal box: "cereal box - does it
right now push with gripper open? if it doesn't can you try with gripper open at lower height and see if it improves yield?"): the push
candidates begin with the OPEN right gripper low on the face - ``push_open_v-`` (closing vertical, the tips one above the other, the lower one
2.35 cm above the support) and ``push_open_h-`` / ``push_open_h`` (closing horizontal, the tips side by side from 5 cm), lowest first - and
today's closed classes follow; the sampler checks them with the open fingers and records the opening, a brace that joins only today's class
keeps that class, the oracle holds the gripper at the opening through the push and its retreat, the program check replays with it, and
without the key everything is today's. ``IORSSmall`` methods run on stand-ins (the x0.8 cereal box iucshm, test_touching_brace_open_push_speed's
c8_k8 pair), the oracle on the fake worlds of ``test_v2_oracle`` / ``test_push_under_strokes``; no simulator is launched."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import types
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import generation_defaults as GD
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import orient as O
from ispatialgym.sampling.v2 import program_check as PC
from ispatialgym.sampling.v2.ik import IKResult
from ispatialgym.sampling.v2.stances import SidecarDraft
from ispatialgym.sampling.v2.tasks import iors_small as IS
from ispatialgym.tests import test_iors_small_lift as LIFT
from ispatialgym.tests import test_touching_brace_open_push_speed as BO

TOUCHING = "inter_object_relation_satisfaction-touching_small_scale"
CUSTOMS = os.path.join(os.path.dirname(__file__), "..", "assets", "task_specs", "isg_customs.json")
SIDECARS = os.path.join(os.path.dirname(__file__), "data", "sidecars")
#: box_of_cereal/iucshm at x0.8 standing at the place (0.70, 0.0) on a 0.81 m table, its 15.67 cm wide face across the push (+y, the edge
#: tangent of LIFT's stance): the push meets its -y face at y = -0.02135
H = 0.2182
CEREAL = LIFT._box(0.1567, 0.0427, H)
PLACE = (th.tensor([0.70, 0.0, 0.81]), C.IDENTITY_QUAT.clone())
D = th.tensor([0.0, 1.0, 0.0])
FACE_Y = -0.0427 / 2


def _sampler(gen=None, hull=None):
    s = LIFT._sampler("touching", gen={"push_gripper_open": True, **(gen or {})})
    s.hull = {"target": CEREAL if hull is None else hull, "partner": LIFT._box(*LIFT.PARTNER)}
    return s


def _open(s, hull=None):
    hull_w = G.local_points_world(s.hull["target"] if hull is None else hull, PLACE)
    face_z = (float(hull_w[:, 2].min()), float(hull_w[:, 2].max()))
    return list(s.open_push_candidates(LIFT._stance(), hull_w, face_z, D))


# ---------------------------------------------------------------------------------------------------------------- the key


def test_the_key_is_off_by_default_and_on_for_the_tabletop_touching_push_only():
    s = LIFT._sampler("touching")
    assert s.push_gripper_open_on() is False
    s.subtask = TOUCHING
    assert s.push_gripper_open_on() is False and "push_gripper_open" not in GD.defaults_for(TOUCHING)  # opt-in per alias
    assert _sampler().push_gripper_open_on() is True
    shelf = _sampler()
    shelf.shelf_start = {"lean_side": 1.0, "lean_deg": 10.0}
    assert shelf.push_gripper_open_on() is False  # the leaning book keeps today's push
    assert LIFT._sampler("nextto", gen={"push_gripper_open": True}).push_gripper_open_on() is False
    assert IS.push_fingers({}) == 0.0 and IS.push_fingers(None) == 0.0 and IS.push_fingers({"fingers_opening_m": 0.035}) == pytest.approx(0.035)


# ---------------------------------------------------------------------------------------------------------------- the candidates


def test_the_open_classes_rise_from_the_lowest_fingertip_with_the_opening_recorded():
    cands = _open(_sampler())
    hs = [ex["contact_height_m"] for _, _, ex in cands]
    assert hs == sorted(hs)
    assert {k for k, _, _ in cands} == {"push_open_v-", "push_open_h-", "push_open_h"}
    # the vertical class first: the eef axis at the lower knuckle's lowest (6.2 cm: 4.2 cm + r 1.5 cm + 5 mm), the tips 3.85 cm below / above it
    k0, (o, q), v0 = cands[0]
    assert k0 == "push_open_v-" and v0["eef_height_m"] == pytest.approx(0.062)
    assert v0["contact_height_m"] == pytest.approx(0.0235) and v0["tip_heights_m"] == pytest.approx([0.0235, 0.1005])
    assert v0["fingers_opening_m"] == pytest.approx(IS.PUSH_OPEN_OPENING_M) == pytest.approx(0.035) and v0["tip_spread_m"] == pytest.approx(0.077)
    assert v0["hand_clearance_m"] == pytest.approx(0.005, abs=1e-4)  # the knuckle; the open lower finger's base sphere 8.5 mm clear
    assert 0.062 - IS.open_fingers_drop_below(q, 0.035) == pytest.approx(0.0085, abs=1e-4)
    assert v0["push_gripper_open"] is True and v0["tip_front"] is True and v0["blade_half_thickness_m"] is None and "fingertip_z" not in v0
    # approach along the push, the closing axis down (the wrist camera, eef -y side, up), the fingertip plane 5 mm before the face
    assert th.allclose(C.approach_axis(q), D, atol=1e-5) and th.allclose(C.closing_axis(q), th.tensor([0.0, 0.0, -1.0]), atol=1e-5)
    ft = C.fingertip((o, q))
    assert [float(v) for v in ft] == pytest.approx([0.70, FACE_Y - 0.005, 0.81 + 0.062], abs=1e-4)
    assert v0["contact_point"] == pytest.approx([0.70, FACE_Y, 0.81 + 0.0235], abs=1e-4)
    # the eef axis in 1 cm steps up to half the target's height
    assert [ex["eef_height_m"] for k, _, ex in cands if k == "push_open_v-"] == pytest.approx([0.062, 0.072, 0.082, 0.092, 0.102])
    assert max(ex["eef_height_m"] for _, _, ex in cands) <= 0.5 * H + 1e-6
    # the horizontal class from 5 cm (the gripper body), both tips at the height; the camera-down turn only where its camera clears the support
    hz = [(k, ex) for k, _, ex in cands if k.startswith("push_open_h")]
    assert min(ex["contact_height_m"] for _, ex in hz) == pytest.approx(0.05)
    assert [k for k, ex in hz if ex["contact_height_m"] < 0.075] == ["push_open_h-"] * 3
    assert min(ex["contact_height_m"] for k, ex in hz if k == "push_open_h") == pytest.approx(0.08)
    assert all(ex["hand_clearance_m"] >= IS.PUSH_OPEN_SUPPORT_CLEARANCE_M - 1e-6 for _, _, ex in cands)
    k, (o, q), ex = next(c for c in cands if c[0] == "push_open_h-")
    assert ex["tip_heights_m"] == pytest.approx([0.05, 0.05]) and th.allclose(C.approach_axis(q), D, atol=1e-5) and abs(float(C.closing_axis(q)[2])) < 1e-5
    assert float(C.fingertip((o, q))[1]) == pytest.approx(FACE_Y - 0.005, abs=1e-4)


def test_the_opening_key_and_a_narrow_face():
    # a wider opening lowers the tip until the open lower finger itself reaches the support
    wide = _open(_sampler({"push_open_opening_m": 0.045}))[0][2]
    assert wide["eef_height_m"] == pytest.approx(0.0685, abs=1e-4) and wide["hand_clearance_m"] == pytest.approx(0.005, abs=1e-4)
    assert wide["contact_height_m"] == pytest.approx(0.0685 - 0.0485, abs=1e-4)
    narrow = _open(_sampler({"push_open_opening_m": 0.02}))[0][2]
    assert narrow["contact_height_m"] == pytest.approx(0.062 - 0.0235, abs=1e-4)
    # a 6 cm face takes the vertical tips on its centre line, not two tips 7.7 cm apart side by side
    kinds = {k for k, _, _ in _open(_sampler(hull=LIFT._box(0.06, 0.0427, H)))}
    assert kinds == {"push_open_v-"}


def test_the_choice_tries_the_open_classes_first_with_the_open_fingers(monkeypatch):
    def run(s):
        calls, seen = [], {}

        def exact(poses, st, arm, **kw):
            calls.append((len(poses), kw.get("fingers")))
            return IKResult(True, th.zeros(28), "", 1)

        s.ikc = SimpleNamespace(q_locked=lambda st: None, exact=exact, solutions={}, _pose_key=lambda p: None)

        def choose_first(ikc, arm, stream, **kw):
            cands = list(stream)
            seen["kinds"] = [c[0] for c in cands]
            for c in (cands[0], next(c for c in cands if c[0] == "push_tip_h-")):
                ok, _r, _qs, why = kw["evaluate"](O.Proposal(kw["label"](c), 0, c, kw["chain"](c)))
                assert ok, why
            seen["label"] = kw["label"](cands[0])
            return None, None, []

        monkeypatch.setattr(IS.O, "choose_first", choose_first)
        hull_w = G.local_points_world(s.hull["target"], PLACE)
        p_init = IS.push_face_point(hull_w, D)
        s._touching_push_choice(LIFT._stance(), PLACE, p_init, (0.81, 0.81 + H), D, 0.08, [])
        today = [k for k, _, _ in s.touching_push_candidates(LIFT._stance(), p_init, (0.81, 0.81 + H), D)]
        return seen, calls, today

    seen, calls, today = run(_sampler())
    n_open = len(_open(_sampler()))
    assert seen["kinds"][:n_open] == [k for k, _, _ in _open(_sampler())] and seen["kinds"][n_open:] == today
    assert seen["label"] == "push_open_v-@2.4cm"
    # the open class's chain and its retreat clear pose with the opening, today's class closed
    assert calls == [(3, pytest.approx(0.035)), (1, pytest.approx(0.035)), (3, 0.0), (1, 0.0)]
    # without the key: today's classes, closed
    off = _sampler()
    off.spec.generation.pop("push_gripper_open")
    seen0, calls0, today0 = run(off)
    assert seen0["kinds"] == today0 and calls0 == [(3, 0.0), (1, 0.0), (3, 0.0), (1, 0.0)]
    # open_classes=False (the brace fallback): today's classes alone
    s = _sampler()
    s.ikc = SimpleNamespace(q_locked=lambda st: None)
    got = {}

    def kinds_only(ikc, arm, stream, **kw):
        got["kinds"] = [c[0] for c in stream]
        return None, None, []

    monkeypatch.setattr(IS.O, "choose_first", kinds_only)
    hull_w = G.local_points_world(s.hull["target"], PLACE)
    s._touching_push_choice(LIFT._stance(), PLACE, IS.push_face_point(hull_w, D), (0.81, 0.81 + H), D, 0.08, [], open_classes=False)
    assert got["kinds"] == today


# ---------------------------------------------------------------------------------------------------------------- the record and the brace


def _contacts(gen, brace_ok):
    """``touching_contacts`` on c8_k8's stand-in (test_touching_brace_open_push_speed): the push choice gives the open class first (today's
    push_tip_h- with ``open_classes=False``), the brace choice passes when ``brace_ok(push_fingers_q)``; ``seen``: the push fingers each brace
    search ran with."""
    tname, pname = "box__of__cereal.n.01_1", "box__of__cereal.n.01_2"
    partner = SimpleNamespace(get_position_orientation=lambda: (BO.P_POSE[0].clone(), BO.P_POSE[1].clone()))
    seen = []
    s = BO._sampler(gen, bindings=SimpleNamespace(one=lambda role: SimpleNamespace(name=tname if role == "target" else pname)), _touching_place_pos=BO.T_PLACE[0],
                    obj=lambda role: partner if role == "partner" else SimpleNamespace(), support_objects=lambda: [], posed=lambda role, pose: contextlib.nullcontext(),
                    _brace_gate={"ok": True}, sidecar=SidecarDraft(BO.BAR, "inter_object_relation_satisfaction"))

    def push_choice(st, t_pose_, p_init, t_face_z, direction, dist, ignore_push, open_classes=True, **kw):
        if open_classes and s.push_gripper_open_on():
            init = C.fingertip_eef(p_init - th.tensor([0.0, 0.0, 0.07]), direction, th.tensor([0.0, 0.0, -1.0]), 0.005)
            kind, extras = "push_open_v-", {"tip_front": True, "pre_dist_m": 0.05, "stop_clearance_m": 0.005, "fingers_opening_m": 0.035, "push_gripper_open": True,
                                            "contact_height_m": 0.0235, "tip_heights_m": [0.0235, 0.1005], "eef_height_m": 0.062}
        else:
            init = C.fingertip_eef(p_init, direction, -C.default_closing_axis(direction), 0.005)
            kind, extras = "push_tip_h-", {"tip_front": True, "pre_dist_m": 0.05, "stop_clearance_m": 0.005}
        chain = [C.pre_pose(init, 0.05), init, (init[0] + direction * dist, init[1])]
        return SimpleNamespace(proposal=SimpleNamespace(payload=(kind, init, extras), chain=chain)), None, []

    def brace_choice(st, push_chain, t_pose, p_pose, b_point, face_z, direction, dist, p_hull, p_init, ignore_push, ignore_brace, push_fingers_q=0.0, **kw):
        seen.append(push_fingers_q)
        if not brace_ok(push_fingers_q):
            return None, None, [f"brace_wide_h-: left (brace): exact IK failed at pose 2 of 2 (push fingers {push_fingers_q})"]
        first = next(iter(s.touching_brace_candidates(st, b_point, face_z, -direction, partner_hull_w=p_hull, push_point=p_init)))
        return SimpleNamespace(proposal=SimpleNamespace(payload=first)), None, []

    s._touching_push_choice = push_choice
    s._touching_brace_choice = brace_choice
    from ispatialgym.sampling.tasks.base import Draft

    draft = Draft()
    draft.goal_poses[tname] = BO.T_GOAL
    draft.pose_targets[tname] = {"position": [float(v) for v in BO.T_GOAL[0]]}
    return s, draft, seen


BAR_GEN = {"brace_open_when_wide": True, "push_speed_m_s": 0.02, "push_gripper_open": True}


def test_touching_contacts_records_the_open_push_for_the_oracle_and_the_program_check():
    from ispatialgym.oracle.sidecar import PushContact

    s, draft, seen = _contacts(BAR_GEN, lambda q: True)
    s.touching_contacts(draft, BO._stance())
    d = s.sidecar.to_dict()["contacts"]
    assert d["push"]["pose_class"] == "push_open_v-" and d["push"]["fingers_opening_m"] == pytest.approx(0.035)
    push = PushContact.from_dict(d["push"], "contacts.push")
    assert push.extra["fingers_opening_m"] == pytest.approx(0.035) and push.extra["tip_heights_m"] == [0.0235, 0.1005] and push.extra["push_speed_m_s"] == 0.02
    assert d["brace"]["kind"] == "brace_wide_h-" and seen == [pytest.approx(0.035)]  # the two-arm solve with the push's open fingers
    tf = draft.summary["touching"]
    assert tf["push_kind"] == "push_open_v-" and tf["push_gripper_open"] == {"open": True, "fingers_opening_m": 0.035}


def test_a_brace_that_joins_only_todays_class_keeps_that_class():
    from ispatialgym.oracle.sidecar import PushContact

    s, draft, seen = _contacts(BAR_GEN, lambda q: q == 0.0)
    s.touching_contacts(draft, BO._stance())
    d = s.sidecar.to_dict()["contacts"]
    assert d["push"]["pose_class"] == "push_tip_h-" and "fingers_opening_m" not in d["push"] and d["brace"]["kind"] == "brace_wide_h-"
    assert "fingers_opening_m" not in PushContact.from_dict(d["push"], "contacts.push").extra
    assert seen == [pytest.approx(0.035), 0.0]
    note = draft.summary["touching"]["push_gripper_open"]
    assert note["open"] is False and note["open_push"] == "push_open_v-" and note["kept"] == "push_tip_h-"
    # no brace with either: the wide partner face requires it (brace_open_when_wide) - rejected as before
    s2, draft2, seen2 = _contacts(BAR_GEN, lambda q: False)
    with pytest.raises(Reject) as e:
        s2.touching_contacts(draft2, BO._stance())
    assert e.value.reason == "brace_open_required" and seen2 == [pytest.approx(0.035), 0.0]
    # without the brace requirement the open push is kept push-only
    s3, draft3, _ = _contacts({"push_gripper_open": True}, lambda q: False)
    s3.touching_contacts(draft3, BO._stance())
    assert s3.sidecar.to_dict()["contacts"]["push"]["pose_class"] == "push_open_v-" and draft3.summary["brace"]["present"] is False


def test_without_the_key_the_record_is_todays():
    s, draft, seen = _contacts({"brace_open_when_wide": True, "push_speed_m_s": 0.02}, lambda q: True)
    s.touching_contacts(draft, BO._stance())
    d = s.sidecar.to_dict()["contacts"]
    assert d["push"]["pose_class"] == "push_tip_h-" and "fingers_opening_m" not in d["push"] and seen == [0.0]
    assert "push_gripper_open" not in draft.summary["touching"]


def test_the_push_stance_search_falls_back_to_todays_class_for_the_brace():
    """``push_chain_feasible(with_brace=True)``: a brace that does not join the open push from a stance is tried with today's classes there
    (``"<closed>+<brace>"``) before the stance counts as push-only."""
    s, _draft, seen = _contacts(BAR_GEN, lambda q: q == 0.0)
    s.posed = lambda role, pose: contextlib.nullcontext()
    s.push_chain_feasible = types.MethodType(IS.IORSSmall.push_chain_feasible, s)
    st = BO._stance()
    assert s.push_chain_feasible(st, BO.T_PLACE, BO.P_POSE) == "push_open_v-" and s._push_chain_height == pytest.approx(0.0235)
    assert s.push_chain_feasible(st, BO.T_PLACE, BO.P_POSE, with_brace=True) == "push_tip_h-+brace_wide_h-"
    assert s._push_chain_height is None and seen == [pytest.approx(0.035), 0.0]
    s2, _d2, seen2 = _contacts(BAR_GEN, lambda q: True)
    s2.push_chain_feasible = types.MethodType(IS.IORSSmall.push_chain_feasible, s2)
    assert s2.push_chain_feasible(st, BO.T_PLACE, BO.P_POSE, with_brace=True) == "push_open_v-+brace_wide_h-" and seen2 == [pytest.approx(0.035)]
    s3, _d3, seen3 = _contacts({"brace_open_when_wide": True}, lambda q: q == 0.0)  # the key off: one search, closed
    s3.push_chain_feasible = types.MethodType(IS.IORSSmall.push_chain_feasible, s3)
    assert s3.push_chain_feasible(st, BO.T_PLACE, BO.P_POSE, with_brace=True) == "push_tip_h-+brace_wide_h-" and seen3 == [0.0]


# ---------------------------------------------------------------------------------------------------------------- the oracle


def _touching_actions(opening=0.035):
    from ispatialgym.oracle.programs import expand
    from ispatialgym.oracle.sidecar import Sidecar

    with open(os.path.join(SIDECARS, f"{TOUCHING}.json")) as f:
        d = json.load(f)
    if opening is not None:
        d = copy.deepcopy(d)
        d["contacts"]["push"].update({"pose_class": "push_open_v-", "fingers_opening_m": opening, "push_gripper_open": True})
    sc = Sidecar.from_dict(d)
    return sc, expand(sc.variant, sc)


def test_the_program_and_the_annotation_of_an_open_push():
    sc, actions = _touching_actions()
    names = [a.name for a in actions]
    push = actions[names.index("push")]
    retreat = actions[names.index("push") + 1]
    assert retreat.name == "retreat" and retreat.after is push and retreat.release is False  # the open hand leaves as it is
    from ispatialgym.oracle.actions import push_is_open

    assert push_is_open(push.contact)
    fine, numbers = push.annotate(sc)
    assert "with the open right" in fine and "with the open right" in numbers
    sc0, actions0 = _touching_actions(None)
    push0 = next(a for a in actions0 if a.name == "push")
    assert not push_is_open(push0.contact) and "with the closed right" in push0.annotate(sc0)[0]
    assert [a.name for a in actions0] == names  # the same program


def test_the_oracle_holds_the_gripper_open_through_the_push_and_its_retreat(monkeypatch):
    """``execute_push`` on the push-under fake world (the left hand along +x) with a generic push record that carries ``fingers_opening_m``:
    the gripper opens to the record (never closes) before the transit, stays there through the push legs, and the retreat after the push
    (``release`` False, as the touching program builds it) leaves it there."""
    from ispatialgym.oracle import execute as EX
    from ispatialgym.oracle.actions import Retreat
    from ispatialgym.oracle.executor import SimState
    from ispatialgym.oracle.motions import Motions
    from ispatialgym.tests.test_push_under_strokes import OBJ, push_action, run_push, sandal_world

    closes = []
    real = Motions.gripper

    def gripper(self, arm, close, **kw):
        closes.append((arm, close))
        return (yield from real(self, arm, close, **kw))

    monkeypatch.setattr(Motions, "gripper", gripper)
    env, robot, _sandal, _goal, _ = sandal_world(yaw_deg=0.0, off_m=0.0)
    push = push_action({"kind": "push", "fingers_opening_m": 0.035, "pose_class": "push_open_v-"})
    assert push.kind == "push"
    ctx, motions = run_push(env, robot, push)
    idx = robot.gripper_control_idx["left"]
    assert ctx["push_gripper"] == {"open": True, "fingers_opening_m": pytest.approx(0.035), "pose_class": "push_open_v-"}
    assert float(robot.q[idx].mean()) == pytest.approx(0.035, abs=1e-3) and motions.gripper_intent_closed["left"] is False
    state = SimState(env, env.task, robot)
    ret = Retreat(arm="left", object=OBJ, after=push)
    n = 0
    for a in EX.run_action(ret, motions, state, ret.targets(None, state), {}):
        env.step(a)
        n += 1
    assert n > 0 and float(robot.q[idx].mean()) == pytest.approx(0.035, abs=1e-3) and motions.gripper_intent_closed["left"] is False
    assert ("left", True) not in closes
    # without the opening in the record: closed as before
    closes.clear()
    env0, robot0, *_ = sandal_world(yaw_deg=0.0, off_m=0.0)
    ctx0, motions0 = run_push(env0, robot0, push_action({"kind": "push"}))
    assert "push_gripper" not in ctx0 and motions0.gripper_intent_closed["left"] is True and ("left", True) in closes


def test_the_program_check_replays_the_open_push_and_its_retreat_with_the_opening():
    sc, actions = _touching_actions()
    names = [a.name for a in actions]
    push, retreat = actions[names.index("push")], actions[names.index("push") + 1]
    brace = next(a for a in actions if a.name == "brace")
    assert PC.push_open_fingers(push, "push", None) == pytest.approx(0.035)
    assert PC.push_open_fingers(retreat, "retreat", "push") == pytest.approx(0.035)
    assert PC.push_open_fingers(brace, "brace", None) is None
    left_retreat = next(a for a in actions[names.index("push") + 2:] if a.name == "retreat")
    assert PC.push_open_fingers(left_retreat, "retreat", left_retreat.after.name) is None
    _sc0, actions0 = _touching_actions(None)
    push0 = next(a for a in actions0 if a.name == "push")
    assert PC.push_open_fingers(push0, "push", None) is None


def test_the_program_check_solves_the_push_chain_with_the_record_opening(monkeypatch):
    """``check_action_targets`` on the open push: every IK of its chain (the pre-contact witness and the push legs) runs with the record's
    opening - the fingers the sampler's chain check used and the oracle holds."""
    sc, actions = _touching_actions()
    push = next(a for a in actions if a.name == "push")
    seen = []

    class _IKC:
        def q_locked(self, stance, fingers=None, **kw):
            seen.append(("q_locked", fingers))
            return th.zeros(28)

        def reaches(self, q, pose, arm):
            return False

        def ignoring(self, objs, ground=True):
            return contextlib.nullcontext()

        def exact(self, poses, stance, arm, fingers=None, **kw):
            seen.append(("exact", fingers))
            return IKResult(True, None, "", len(poses))  # no witness to polish

        def waypoints_ok(self, chain, stance, arm, fingers=None, **kw):
            seen.append(("waypoints", fingers))
            return IKResult(True, None, "", len(chain))

    monkeypatch.setenv("ISG_FINGERTIP_FRAME_CHECK", "off")
    monkeypatch.setenv("ISG_PUSH_REPLAY", "off")
    monkeypatch.setenv("ISG_FLIPPED_PUSH_CHECK", "off")
    obj = SimpleNamespace(name=push.contact.object)
    sampler = SimpleNamespace(objects={push.contact.object: obj, "mug.n.04_1": SimpleNamespace(name="mug")}, support_objects=lambda: [], ikc=_IKC(),
                              kit=SimpleNamespace(robot=SimpleNamespace(arm_control_idx={"right": list(range(7)), "left": list(range(7, 14))})))
    PC.check_action_targets(sc, sampler, push, push.targets(sc, None), "manipulation", None)
    ik = [f for k, f in seen if k in ("exact", "waypoints")]
    assert ik and all(f == pytest.approx(0.035) for f in ik)
    # today's closed record: closed fingers
    sc0, actions0 = _touching_actions(None)
    push0 = next(a for a in actions0 if a.name == "push")
    seen.clear()
    PC.check_action_targets(sc0, sampler, push0, push0.targets(sc0, None), "manipulation", None)
    ik0 = [f for k, f in seen if k in ("exact", "waypoints")]
    assert ik0 and all(f == 0.0 for f in ik0)


# ---------------------------------------------------------------------------------------------------------------- customs


def test_customs_keep_the_key_off_on_the_12_cereal_box_touching_aliases_after_the_validation():
    """The 12 cereal-box aliases carry the key OFF since the round-2 validation (open: 9 / 16 demos, 8.1 attempts per instance; closed:
    16 / 16, 5.9) with both notes; no alias has it on."""
    with open(CUSTOMS) as f:
        customs = json.load(f)
    on = sorted(a for a, v in customs.items() if isinstance(v, dict) and (v.get("generation") or {}).get("push_gripper_open"))
    cereal = sorted(a for a in customs if "box_of_cereal" in a and "touching_small_scale" in a)
    assert len(cereal) == 12 and on == []
    for a in cereal:
        g = customs[a]["generation"]
        assert g["push_gripper_open"] is False and g["push_height_sweep"] is False  # today's lateral rule: the first lateral with push + brace
        note = g["_push_gripper_open_note"]
        assert note.startswith(f"oracle-fix 2026-11-06 (owner feedback on the {a} demo:") and "push_open_v-" in note
        assert "9 / 16 demos" in g["_push_gripper_open_off_note"]
    assert BO.BAR in cereal
