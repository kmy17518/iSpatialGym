"""The BRACE OPENING rule (owner decision 2026-11-07) and the brace-opening mismatch fixes (i7-braceopen) - no simulator.

* ``ISPATIALGYM_BRACE_FULL_OPEN`` ("0" default = the openings of before 2026-11-07, "1" = an open brace opens fully, narrower only
  when the braced face cannot take the full spread): ``tabletop.brace_opening``, ``articulation_small.wide_stop_layout`` (the
  push-close brace ``close_push_brace_opening_m`` and the carton-v4 wide stop ``v4_brace_opening_m``), ``block_design.pack_brace_opening``
  (the packing tail's wide brace).
* Default on, independent of the switch: every check of a wide brace uses the record's ``fingers_opening_m`` - the opening the oracle
  executes (``execute_brace``) - not closed fingers: ``tabletop.brace_second_arm(brace_fingers=...)`` (the two-arm solve), the pack's
  pushes with the brace locked, the program check's brace retreat and locked brace hand (``program_check.brace_open_fingers``,
  ``locked_seed``, ``_Tracker.locked_fingers``).
"""

import inspect
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.ik import IKResult

Q_ID = th.tensor([0.0, 0.0, 0.0, 1.0])
ARM_IDX = {"right": th.arange(0, 7), "left": th.arange(7, 14)}
GRIP_IDX = {"right": th.tensor([14, 15]), "left": th.tensor([16, 17])}


def _kit(**kw):
    return SimpleNamespace(robot=SimpleNamespace(arm_control_idx=ARM_IDX, gripper_control_idx=GRIP_IDX), **kw)


def _q_locked(stance=None, fingers=None, **kw):
    q = th.zeros(28)
    f = C.FINGER_OPEN if fingers is None else float(fingers)
    q[GRIP_IDX["right"]] = f
    q[GRIP_IDX["left"]] = f
    return q


@pytest.fixture
def switch(monkeypatch):
    def _set(on: bool):
        monkeypatch.setenv(TT.BRACE_FULL_OPEN_ENV, "1" if on else "0")
    _set(False)
    return _set


# ---------------------------------------------------------------------------------------------------------- the rule and its switch

def test_the_default_is_full_open(monkeypatch):
    """i7-contactbrace 2026-11-08: the owner's rule adopted after the A/B - unset = on; ``0`` = the openings of before."""
    monkeypatch.delenv(TT.BRACE_FULL_OPEN_ENV, raising=False)
    assert TT.BRACE_FULL_OPEN_DEFAULT == "1" and TT.brace_full_open() and TT.brace_opening(0.04) == 0.05
    assert TT.contact_map_knobs("touch_brace") == {"brace_full_open": True}


def test_brace_opening_per_switch(switch, monkeypatch):
    assert not TT.brace_full_open()  # switched off: today's openings
    assert TT.brace_opening(0.04) == 0.04 and TT.brace_opening(0.04, room=0.01) == 0.04
    monkeypatch.setenv(TT.BRACE_FULL_OPEN_ENV, "0")
    assert TT.brace_opening(0.04) == 0.04
    switch(True)
    assert TT.brace_full_open() and TT.BRACE_FULL_OPENING_M == C.FINGER_OPEN == 0.05
    assert TT.brace_opening(0.04) == 0.05  # fully open
    assert TT.brace_opening(0.04, room=0.2) == 0.05  # the face takes the full spread
    assert TT.brace_opening(0.04, room=0.03) == 0.03  # narrower only when the face cannot take it
    assert TT.brace_opening(0.04, room=-0.01) == 0.0


def test_brace_fingers_is_the_records_opening_else_closed():
    assert TT.brace_fingers({"fingers_opening_m": 0.04}) == 0.04
    assert TT.brace_fingers({"pose_class": "blade_down"}) == C.FINGER_CLOSED
    assert TT.brace_fingers(None) == C.FINGER_CLOSED


def test_with_gripper_sets_one_hand():
    q = _q_locked(fingers=0.0)
    q2 = TT.with_gripper(_kit(), q, "left", 0.05)
    assert q2 is not q and q2[GRIP_IDX["left"]].tolist() == pytest.approx([0.05, 0.05]) and q2[GRIP_IDX["right"]].tolist() == [0.0, 0.0]
    assert th.equal(TT.with_gripper(None, q, "left", None), q)  # None: an unchanged clone (no kit needed)


# ----------------------------------------------------------------------------- articulation_small: the push-close / v4 face stops

def _layout():
    from ispatialgym.sampling.v2.tasks.articulation_small import wide_stop_layout

    return wide_stop_layout


def _today(opening, inset, ends, near_i):
    """The inline computation of before 2026-11-07 (close_push_brace_* / v4_brace_*)."""
    a = ends[near_i] + (inset if near_i == 0 else -inset)
    return opening, min(max(a, ends[0] + 0.01), ends[1] - 0.01)


@pytest.mark.parametrize("ends", [[-0.12, 0.12], [0.31, 0.58], [-0.03, 0.03]])
@pytest.mark.parametrize("near_i", [0, 1])
@pytest.mark.parametrize("opening,inset", [(0.04, 0.04), (0.04, 0.05)])
def test_wide_stop_layout_switch_off_is_today(switch, ends, near_i, opening, inset):
    assert _layout()(opening, inset, ends, near_i) == _today(opening, inset, ends, near_i)


def test_wide_stop_layout_full_open_keeps_both_fingers_on_the_face(switch):
    switch(True)
    m = TT.BRACE_STOP_END_MARGIN_M
    # a carton face 24 cm long: fully open, the stop moved in so the near pad face is 1 cm inside the corner (inset 0.04 -> 0.06)
    for near_i, sign in ((0, 1.0), (1, -1.0)):
        q, a = _layout()(0.04, 0.04, [-0.12, 0.12], near_i)
        assert q == 0.05
        assert a == pytest.approx(sign * (-0.12 + 0.05 + m))
        assert a - q >= -0.12 + m - 1e-9 and a + q <= 0.12 - m + 1e-9
    # the v4 inset 0.05: moved to 0.06 as well
    assert _layout()(0.04, 0.05, [0.0, 0.3], 0) == (0.05, pytest.approx(0.06))
    # a larger configured inset is kept
    assert _layout()(0.04, 0.09, [0.0, 0.3], 0) == (0.05, pytest.approx(0.09))
    # a face 8 cm long cannot take the full spread: narrower, centred
    q, a = _layout()(0.04, 0.04, [0.0, 0.08], 0)
    assert q == pytest.approx(0.04 - m) and a == pytest.approx(0.04)
    # the stop moved onto the push line keeps the face fit with the wider opening (wide_stop_positions' own rule)
    from ispatialgym.sampling.v2.tasks.articulation_small import wide_stop_positions

    q, a = _layout()(0.04, 0.04, [-0.12, 0.12], 0)
    for s in wide_stop_positions(a, 0.05, q, [-0.12, 0.12]):
        assert -0.12 + 0.01 + q <= s <= 0.12 - 0.01 - q


def test_close_push_and_v4_stops_are_checked_at_their_opening():
    """The call sites: the push-close stop's two-arm solve passes the stop's opening for the brace hand (the fist closed), its lift-off and
    diagnostics use it; the v4 wide stop is solved / probed at the record's opening with the lever hand closed in the locked config."""
    from ispatialgym.sampling.v2.tasks import articulation_small as AS

    src = inspect.getsource(AS.ArticulationSmall._close_push_flow)
    stop = src[src.index("def stop_at"):src.index("for kind in kinds")]
    assert "brace_fingers=opening" in stop and "fingers=C.FINGER_CLOSED, ignore_while_locked" in stop  # fist closed, brace hand open
    assert stop.count("cache=False, fingers=opening)") == 3  # lift-off, the two diagnostics
    assert "FINGER_CLOSED" not in stop.replace("fingers=C.FINGER_CLOSED, ignore_while_locked", "")
    assert "wide_stop_layout(float(self.gen(\"close_push_brace_opening_m\", 0.04))" in src
    v4 = inspect.getsource(AS.ArticulationSmall._sample_with_v4)
    wide = v4[v4.index("wide_stop_layout(float(self.gen(\"v4_brace_opening_m\""):v4.index("def brace_in_pinch_solution")]
    assert "fingers=b_fingers, q_seed=seed_b" in wide and "q_seed=q_b, cache=False, fingers=b_fingers" in wide
    assert "TT.with_gripper(self.kit, r.q, lever_arm, C.FINGER_CLOSED)" in wide and "return rec, (q_b.clone()" in wide


# ----------------------------------------------------------------------------------------------------- the two-arm solve

class _BraceIKC:
    """exact() records the finger joints it was asked to solve with: ``fingers`` without a seed, the seed's otherwise (as IKChecker)."""

    def __init__(self):
        self.calls = []

    def exact(self, poses, stance, arm, ignore_for_last=(), q_seed=None, cache=True, fingers=None, approached=None, q_first=None):
        q0 = _q_locked(fingers=fingers) if q_seed is None else q_seed.clone()
        self.calls.append({"arm": arm, "grip": {a: q0[GRIP_IDX[a]].tolist() for a in ("right", "left")},
                           "q_first": None if q_first is None else q_first[GRIP_IDX["left"]].tolist()})
        q0[ARM_IDX[arm]] = 0.3
        return IKResult(True, q0, "", len(poses))

    def q_locked(self, stance, fingers=None, **kw):
        return _q_locked(fingers=fingers)

    @contextmanager
    def ignoring(self, objs):
        yield


def _two_arm(brace_fingers, brace_only=False, q_first=None):
    seen = []
    ikc = _BraceIKC()

    def free(q, margin=0.0):
        seen.append(q.clone())
        return th.tensor([True])

    s = SimpleNamespace(ikc=ikc, kit=_kit(collision_free=free))
    pose = (th.zeros(3), Q_ID)
    res = TT.brace_second_arm(s, None, "right", [pose, pose], "left", [pose, pose], fingers=C.FINGER_CLOSED, brace_fingers=brace_fingers,
                              brace_only_check=([], 0.003) if brace_only else None, q_first_brace=q_first)
    return res, ikc.calls, seen


def test_brace_second_arm_solves_the_brace_hand_at_its_opening():
    res, calls, seen = _two_arm(0.05, brace_only=True, q_first=th.full((28,), 0.1))
    assert res.ok
    work, brace = calls
    assert work["arm"] == "right" and work["grip"] == {"right": [0.0, 0.0], "left": [0.0, 0.0]}
    assert brace["arm"] == "left" and brace["grip"] == {"right": [0.0, 0.0], "left": pytest.approx([0.05, 0.05])}  # the fist closed, the stop open
    assert brace["q_first"] == pytest.approx([0.05, 0.05])
    combined, brace_only = seen
    assert combined[GRIP_IDX["left"]].tolist() == pytest.approx([0.05, 0.05]) and combined[GRIP_IDX["right"]].tolist() == [0.0, 0.0]
    assert brace_only[GRIP_IDX["left"]].tolist() == pytest.approx([0.05, 0.05])
    assert res.q[GRIP_IDX["left"]].tolist() == pytest.approx([0.05, 0.05])


def test_brace_second_arm_without_brace_fingers_is_unchanged():
    res, calls, seen = _two_arm(None, brace_only=True, q_first=th.full((28,), 0.1))
    assert res.ok and [c["grip"] for c in calls] == [{"right": [0.0, 0.0], "left": [0.0, 0.0]}] * 2
    assert all(q[GRIP_IDX["left"]].tolist() == [0.0, 0.0] for q in seen)


# ---------------------------------------------------------------------------------------------------------- block design pack brace

def _bd():
    from ispatialgym.sampling.v2.tasks import block_design as BD

    return BD


def _old_pack(spread):
    BD = _bd()
    return max(0.0, min(float(C.FINGER_OPEN), 0.5 * spread - 0.5 * BD.FINGER_HALF_THICKNESS)) if spread > 0 else 0.0


@pytest.mark.parametrize("spread", [0.0, 0.026, 0.035, 0.12, 0.3])
def test_pack_brace_opening_switch_off_is_today(switch, spread):
    BD = _bd()
    assert BD.pack_brace_opening(spread) == _old_pack(spread)
    assert BD.pack_brace_opening(spread, face_half_m=0.2) == _old_pack(spread)  # the face is not consulted with the switch off


def test_pack_brace_opening_full_open(switch):
    BD = _bd()
    switch(True)
    m, half_t = BD.PACK_BRACE_END_MARGIN_M, BD.FINGER_HALF_THICKNESS
    # two 2.5 cm dice 3.5 cm apart (the 10 mm place grid): the faces span +-3 cm - each finger's outer side 5 mm inside the outer edges
    assert BD.pack_brace_opening(0.035, face_half_m=0.03) == pytest.approx(0.03 - half_t - m)
    assert BD.pack_brace_opening(0.035, face_half_m=0.03) > _old_pack(0.035)
    # a long braced line takes the full spread
    assert BD.pack_brace_opening(0.12, face_half_m=0.0725) == 0.05
    # never narrower than the fingers on the cubes' centres
    assert BD.pack_brace_opening(0.035, face_half_m=0.02) == pytest.approx(_old_pack(0.035))
    # one braced cube is a closed blade, not an open brace
    assert BD.pack_brace_opening(0.0, face_half_m=0.0125) == 0.0
    assert BD.pack_brace_opening(0.035) == pytest.approx(_old_pack(0.035))  # no face known: today's


def _cube_hull(e=0.025):
    h = 0.5 * e
    return th.tensor([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])


def test_pack_face_half_is_the_braced_cubes_extent_along_the_closing_axis():
    BD = _bd()
    s = object.__new__(BD.BlockDesign)
    s.hull = _cube_hull()
    world = {"a": (th.tensor([0.5, 0.0175, 0.8]), Q_ID), "b": (th.tensor([0.5, -0.0175, 0.8]), Q_ID)}
    mid = th.tensor([0.5, 0.0, 0.8])
    # braced on their -x faces (brace direction +x): the closing axis runs along y, the faces span -0.03 .. 0.03
    assert s._pack_face_half(world, ["a", "b"], mid, th.tensor([1.0, 0.0, 0.0])) == pytest.approx(0.03)
    assert s._pack_face_half(world, ["a", "b"], mid + th.tensor([0.0, 0.005, 0.0]), th.tensor([-1.0, 0.0, 0.0])) == pytest.approx(0.025)
    s.hull = None
    assert s._pack_face_half(world, ["a", "b"], mid, th.tensor([1.0, 0.0, 0.0])) is None


def _pack_sampler(monkeypatch, captured):
    BD = _bd()
    s = object.__new__(BD.BlockDesign)
    s.spec = SimpleNamespace(generation={})
    s.gp = SimpleNamespace(variant="configuration_matching-block_design-block_4-breakfast_table")
    s.hull = _cube_hull()
    s.c_local = th.zeros(3)
    s.edge = 0.025
    s.objects = {"a": object(), "b": object(), "c": object()}
    ikc = _BraceIKC()
    s.ikc = ikc
    s.kit = _kit()
    s._blade_poses_at = lambda mid, normal, manip, prefix, modes=("pad", "side"): iter([((th.tensor([0.48, 0.0, 0.82]), Q_ID), th.tensor([0.0, 0.0, -1.0]),
                                                                                         "pack_brace_blade_side_0", 0.007)])

    def pushes(world, pushes_, push_dir, manip, support, push_arm, q_seed, braced_objs, b_name):
        captured["q_seed"] = q_seed.clone()
        return [{"arm": push_arm}], ""

    s._pack_pushes = pushes

    def choose_first(ikc_, arm, stream, *, key, label, chain, evaluate, q_start, camera_idx=None, stage="", downstream=None, arm_of=None):
        c = next(iter(stream))
        p = SimpleNamespace(label=label(c), payload=c, chain=chain(c), q_first=None)
        ok, res, _qs, why = evaluate(p)
        assert ok, why
        sc = SimpleNamespace(proposal=p, result=res)
        assert downstream(sc)[0]
        return sc, None, []

    monkeypatch.setattr(BD.O, "choose_first", choose_first)
    world = {"a": (th.tensor([0.5, 0.0175, 0.8]), Q_ID), "b": (th.tensor([0.5, -0.0175, 0.8]), Q_ID), "c": (th.tensor([0.55, 0.0175, 0.8]), Q_ID)}
    return s, ikc, world


@pytest.mark.parametrize("on", [False, True])
def test_pack_pushes_are_checked_with_the_brace_hand_at_its_opening(switch, monkeypatch, on):
    BD = _bd()
    switch(on)
    captured = {}
    s, ikc, world = _pack_sampler(monkeypatch, captured)
    stage, why = s._pack_stage("columns", world, ["a", "b"], [("c", 0.009)], th.tensor([-1.0, 0.0, 0.0]), th.tensor([1.0, 0.0, 0.0]),
                               Stance_("manipulation"), object())
    assert stage is not None, why
    q_open = BD.pack_brace_opening(0.035, 0.03 if on else None)
    assert stage["brace"]["fingers_opening_m"] == round(q_open, 4)
    if not on:
        assert stage["brace"]["fingers_opening_m"] == round(_old_pack(0.035), 4)  # the record is today's with the switch off
    else:
        assert q_open == pytest.approx(0.03 - BD.FINGER_HALF_THICKNESS - BD.PACK_BRACE_END_MARGIN_M)
    # the brace chain at its opening (as before), the pushes with the brace hand held there and the push hand closed (the fix)
    assert ikc.calls[0]["arm"] == "left" and ikc.calls[0]["grip"]["left"] == pytest.approx([q_open] * 2)
    q_seed = captured["q_seed"]
    assert q_seed[GRIP_IDX["left"]].tolist() == pytest.approx([q_open] * 2) and q_seed[GRIP_IDX["right"]].tolist() == [0.0, 0.0]
    assert q_seed[ARM_IDX["left"]].tolist() == pytest.approx([0.3] * 7)  # the brace arm at its contact


def Stance_(name):
    from ispatialgym.sampling.v2.stances import Stance

    return Stance(name, (0.0, 0.0, 0.0), 0.75, -0.3, stand_off=0.35)


# ----------------------------------------------------------------------------------------------------------------- program check

def _brace(arm="left", opening=None):
    extra = {"pose_class": "wide_stop", "fingers_opening_m": opening} if opening is not None else {"pose_class": "blade"}
    return SimpleNamespace(name="brace", arms=[arm], contact=SimpleNamespace(arm=arm, extra=extra))


def _push(arm="right", extra=None):
    return SimpleNamespace(name="push", arms=[arm], contact=SimpleNamespace(arm=arm, extra=dict(extra or {})))


def _retreat(after, arm=None):
    return SimpleNamespace(name="retreat", arms=[arm or after.arms[0]], after=after)


def test_brace_open_fingers_covers_the_brace_and_its_retreat():
    from ispatialgym.sampling.v2 import program_check as PC

    wide, closed = _brace(opening=0.04), _brace()
    assert PC.brace_open_fingers(wide, "brace", None) == 0.04
    assert PC.brace_open_fingers(_retreat(wide), "retreat", "brace") == 0.04  # the oracle lifts the open stop off as it is
    assert PC.brace_open_fingers(closed, "brace", None) is None
    assert PC.brace_open_fingers(_retreat(closed), "retreat", "brace") is None
    push = _push(extra={"fingers_opening_m": 0.035})
    assert PC.brace_open_fingers(push, "push", None) is None and PC.brace_open_fingers(_retreat(push), "retreat", "push") is None
    assert PC.push_open_fingers(push, "push", None) == 0.035  # an open push stays the push rule's


def test_tracker_holds_a_wide_brace_open_through_the_other_arms_actions():
    from ispatialgym.sampling.v2 import program_check as PC
    from ispatialgym.sampling.v2.stances import SidecarDraft

    tracker = PC._Tracker(SidecarDraft("configuration_matching-articulation_close_small_scale-carton-breakfast_table", "configuration_matching"))
    sampler = SimpleNamespace(kit=_kit())
    q = [0.0] * 28
    q[7:14] = [0.2] * 7
    wide = _brace(opening=0.04)
    tracker.after(wide, {"contact": {"q": q}}, sampler)
    push = _push()
    assert set(tracker.locked_arms(push)) == {"left"} and tracker.locked_fingers(push) == {"left": 0.04}
    assert tracker.locked_fingers(_retreat(push)) == {"left": 0.04}
    assert tracker.locked_fingers(_retreat(wide)) == {}  # the brace arm acting: not locked
    tracker.after(_retreat(wide), {}, sampler)
    assert tracker.locked_fingers(push) == {} and tracker.locked_arms(push) == {}
    tracker.after(_brace(), {"contact": {"q": q}}, sampler)  # a closed brace: locked, its fingers the acting arm's as before
    assert set(tracker.locked_arms(push)) == {"left"} and tracker.locked_fingers(push) == {}


def test_locked_seed_puts_the_locked_brace_hand_at_its_opening():
    from ispatialgym.sampling.v2 import program_check as PC

    sampler = SimpleNamespace(ikc=SimpleNamespace(q_locked=_q_locked), kit=_kit())
    locked = {"left": th.full((7,), 0.2)}
    q = PC.locked_seed(sampler, None, 0.0, locked, {"left": 0.04})
    assert q[ARM_IDX["left"]].tolist() == pytest.approx([0.2] * 7)
    assert q[GRIP_IDX["left"]].tolist() == pytest.approx([0.04, 0.04]) and q[GRIP_IDX["right"]].tolist() == [0.0, 0.0]
    # no wide brace locked: the acting arm's opening on both hands, as before
    q0 = PC.locked_seed(sampler, None, 0.0, locked, None)
    assert q0[GRIP_IDX["left"]].tolist() == [0.0, 0.0]
    assert th.equal(PC.locked_seed(sampler, None, 0.0, locked, {}), q0)
    # an entry for an arm that is not locked is ignored
    assert th.equal(PC.locked_seed(sampler, None, 0.0, locked, {"right": 0.05}), q0)


def test_check_action_targets_receives_the_trackers_locked_fingers():
    from ispatialgym.sampling.v2 import program_check as PC

    assert "locked_fingers" in inspect.signature(PC.check_action_targets).parameters
    run_src = inspect.getsource(PC.run)
    assert "locked_fingers=tracker.locked_fingers(action)" in run_src
    body = inspect.getsource(PC.check_action_targets)
    assert "q_seed = locked_seed(sampler, stance, fingers, locked, locked_fingers)" in body
    assert "wide = brace_open_fingers(action, name, after_name)" in body


# ---------------------------------------------------------------------------------------------------------- i7-contactbrace: one switch, one rule

def test_one_switch_definition(switch, monkeypatch):
    """i7-contactbrace 2026-11-08: ``ISPATIALGYM_BRACE_FULL_OPEN`` is defined once (tabletop); contact_poses (the touching wide stop) and the
    contact library's identity read the same switch with the same parsing."""
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    assert CPZ.BRACE_FULL_OPEN_ENV is TT.BRACE_FULL_OPEN_ENV and CPZ.brace_full_open_on is TT.brace_full_open
    assert CPZ.BRACE_FULL_OPEN_SPREAD_M == round(2 * TT.BRACE_FULL_OPENING_M + CPZ.CLOSED_TIP_SPREAD_M, 4) == 0.127
    for v, on in (("1", True), ("true", True), ("on", True), ("0", False), ("", False), ("no", False)):
        monkeypatch.setenv(TT.BRACE_FULL_OPEN_ENV, v)
        assert TT.brace_full_open() is on and CPZ.brace_full_open_on() is on
        assert CPZ.brace_wide_spreads(CPZ.PoseGen({}))[0] == (0.127 if on else 0.09)


@pytest.mark.parametrize("opening", [None, 0.0, -0.01, 0.0315, 0.04, 0.05, "x"])
def test_brace_fingers_is_the_oracles_brace_is_open(opening):
    """One rule of the brace checks: ``tabletop.brace_fingers`` = the opening ``execute_brace`` holds (``actions.brace_is_open``), and the
    touching brace's ``contact_poses.contact_fingers(.., "brace")`` is the same function."""
    from ispatialgym.oracle import actions as AC
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    extras = {} if opening is None else {"fingers_opening_m": opening}
    want = float(opening) if AC.brace_is_open(SimpleNamespace(extra=extras)) else C.FINGER_CLOSED
    assert TT.brace_fingers(extras) == want == CPZ.contact_fingers(extras, "brace")


def test_program_check_brace_opening_is_the_one_rule():
    from ispatialgym.sampling.v2 import program_check as PC

    assert PC.brace_open_fingers(_brace(opening=0.0), "brace", None) == C.FINGER_CLOSED  # a recorded 0 is the closed gripper (as before)
    assert PC.brace_open_fingers(_brace(opening="x"), "brace", None) == C.FINGER_CLOSED  # the oracle: not open -> closed
    assert PC.brace_open_fingers(_brace(opening=0.05), "brace", None) == 0.05


def test_touching_brace_library_identity_unchanged_with_the_switch_off(switch):
    """The touching_brace knobs of the libraries built 2026-11-07 (contact_libs_20261107e: ntbrtz / dkhgxn) resolve the same with the switch
    off - the library digests and the map keys built from them do not move; on, the identity changes."""
    from ispatialgym.grasping import contact_library as CL

    sub = "inter_object_relation_satisfaction-touching_small_scale"
    built = {"brace_wide_spreads_m": [0.09, 0.07, 0.05], "brace_full_open": False}
    got = CL.resolved_knobs("touching_brace", {}, sub)
    assert {k: got[k] for k in built} == built and got["brace_full_open"] is False
    got_w = CL.resolved_knobs("touching_brace", {"brace_open_when_wide": True}, sub)
    assert got_w["brace_open_when_wide"] is True and got_w["brace_wide_spreads_m"] == [0.09, 0.07, 0.05]
    switch(True)
    on = CL.resolved_knobs("touching_brace", {}, sub)
    assert on["brace_full_open"] is True and on["brace_wide_spreads_m"] == [0.127, 0.09, 0.07, 0.05]


def _map_files(root, entry, key, knobs):
    import json
    import os

    import numpy as np

    from ispatialgym.sampling import contact_maps as CMAP

    d = os.path.join(root, entry)
    os.makedirs(d, exist_ok=True)
    meta = {"schema": CMAP.SCHEMA, "kind": "touch_brace", "arm": "left", "object": {"entry": entry}, "rest_id": "r0",
            "height": {"value": 0.8126}, "pose_set": {"digest": key, "classes": []}, "library": {"families": ["touching_brace"]}}
    if knobs is not None:
        meta["library"]["knobs"] = {"touching_brace": knobs}
    base = os.path.join(d, f"cmap__touch_brace__box__m__h0.8126__{key}")
    with open(base + ".json", "w") as f:
        json.dump(meta, f)
    np.savez_compressed(base + ".npz", cells=np.zeros((1, 2)), yaws=np.zeros(1))


def test_touch_brace_maps_are_read_only_under_the_opening_rule_they_were_built_with(switch, tmp_path):
    """The contact libraries / maps use the opening the oracle executes: a touch_brace map records its library's knobs (not in its key);
    the sampler keeps only a map built under the run's ``ISPATIALGYM_BRACE_FULL_OPEN`` (a map without the record = built before it, off)."""
    from ispatialgym.sampling import contact_maps as CMAP

    root = str(tmp_path)
    _map_files(root, "old", "a" * 16, None)
    _map_files(root, "on", "b" * 16, {"brace_full_open": True, "brace_wide_spreads_m": [0.127, 0.09, 0.07, 0.05]})
    _map_files(root, "off", "c" * 16, {"brace_full_open": False})
    assert TT.contact_map_knobs("touch_brace") == {"brace_full_open": False} and TT.contact_map_knobs("touch_push") is None

    def found(entry):
        cm = CMAP.ContactMap.find(root, "touch_brace", entry, 0.8126, knobs_require=TT.contact_map_knobs("touch_brace"))
        return cm is not None

    assert (found("old"), found("off"), found("on")) == (True, True, False)
    switch(True)
    assert TT.contact_map_knobs("touch_brace") == {"brace_full_open": True}
    assert (found("old"), found("off"), found("on")) == (False, False, True)
    assert CMAP.ContactMap.find(root, "touch_brace", "old", 0.8126) is not None  # no requirement: as before
    assert CMAP.library_knob({"library": {"knobs": {"touching_brace": {"brace_full_open": True}}}}, "brace_full_open", False) is True
    assert CMAP.library_knob({"library": {}}, "brace_full_open", False) is False


def test_map_meta_records_the_library_knobs_outside_the_key():
    from ispatialgym.sampling import contact_maps as CMAP

    src = inspect.getsource(CMAP.compute_contact_map)
    assert '"knobs": {f: (v or {}).get("knobs")' in src
    meta = {"schema": CMAP.SCHEMA, "kind": "touch_brace", "arm": "left", "plane": "slab", "edge_distances": [0.45], "config": "c", "policy_key": "p",
            "robot": "r", "object": {"entry": "e"}, "height": {"value": 0.8}, "pose_set": {"digest": "d"}, "library": {"digest": "L"}}
    k0 = CMAP.map_key(meta)
    meta["library"]["knobs"] = {"touching_brace": {"brace_full_open": True}}
    assert CMAP.map_key(meta) == k0  # existing map keys unchanged


def test_brace_opening_counters():
    s = SimpleNamespace(kit=SimpleNamespace(stats={}))
    TT.count_brace_opening(s, {"fingers_opening_m": 0.05}, ok=False)
    TT.count_brace_opening(s, {"fingers_opening_m": 0.0315}, ok=True)
    TT.count_brace_opening(s, {"pose_class": "brace_tip_h-"})
    assert s.kit.stats == {"brace_opening.failed.0.0500": 1, "brace_opening.passed.0.0315": 1, "brace_opening.recorded.closed": 1}
