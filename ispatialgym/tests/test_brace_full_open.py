"""The brace-opening mismatch fixes (i7-braceopen 2026-11-07) - no simulator.

* Default on: every check of a wide brace uses the record's ``fingers_opening_m`` - the opening the oracle
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

def test_close_push_and_v4_stops_are_checked_at_their_opening():
    """The call sites: the push-close stop's two-arm solve passes the stop's opening for the brace hand (the fist closed), its lift-off and
    diagnostics use it; the v4 wide stop is solved / probed at the record's opening with the lever hand closed in the locked config."""
    from ispatialgym.sampling.v2.tasks import articulation_small as AS

    src = inspect.getsource(AS.ArticulationSmall._close_push_flow)
    stop = src[src.index("def stop_at"):src.index("for kind in kinds")]
    assert "brace_fingers=opening" in stop and "fingers=C.FINGER_CLOSED, ignore_while_locked" in stop  # fist closed, brace hand open
    assert stop.count("cache=False, fingers=opening)") == 3  # lift-off, the two diagnostics
    assert "FINGER_CLOSED" not in stop.replace("fingers=C.FINGER_CLOSED, ignore_while_locked", "")
    v4 = inspect.getsource(AS.ArticulationSmall._sample_with_v4)
    wide = v4[v4.index("opening = float(self.gen(\"v4_brace_opening_m\""):v4.index("def brace_in_pinch_solution")]
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


def _cube_hull(e=0.025):
    h = 0.5 * e
    return th.tensor([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])


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


def test_pack_pushes_are_checked_with_the_brace_hand_at_its_opening(monkeypatch):
    captured = {}
    s, ikc, world = _pack_sampler(monkeypatch, captured)
    stage, why = s._pack_stage("columns", world, ["a", "b"], [("c", 0.009)], th.tensor([-1.0, 0.0, 0.0]), th.tensor([1.0, 0.0, 0.0]),
                               Stance_("manipulation"), object())
    assert stage is not None, why
    q_open = _old_pack(0.035)
    assert stage["brace"]["fingers_opening_m"] == round(q_open, 4)  # the record is unchanged
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
