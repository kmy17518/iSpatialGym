"""Tests of the wrist-orientation rule (``sampling/v2/orient.py``, owner 2026-09-27): the order (joint-limit safe, then more
clearance when neither is safe, then easier to reach with the generator's order within the tolerance), the decision record,
the wrist-flipped options (from passing solutions and from a failed chain's witnesses), the re-ranking of a verified option,
the camera check, the downstream fallback, ``pairs`` and ``choose_first``. A real ``IKChecker`` on a stub kit with the
``r1pro.urdf`` right-arm limits; no simulator is launched.
"""

import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import orient as O

ARM_IDX = th.tensor([M.ALL_JOINTS.index(n) for n in M.ARM_JOINTS["right"]])
ARM_LO = [-4.4506, -3.1416, -2.356196, -2.0944, -2.356196, -1.047198, -1.5708]
ARM_HI = [1.3090, 0.1745, 2.356196, 0.3491, 2.356196, 1.047198, 1.5708]


def _q(arm_values):
    q = th.zeros(28)
    q[ARM_IDX] = th.tensor(arm_values, dtype=th.float32)
    return q


def _ikc(links_clear=None):
    from ispatialgym.sampling.v2.ik import IKChecker

    lo, hi = th.full((28,), -10.0), th.full((28,), 10.0)
    lo[ARM_IDX], hi[ARM_IDX] = th.tensor(ARM_LO), th.tensor(ARM_HI)
    kit = SimpleNamespace(robot=SimpleNamespace(arm_control_idx={"right": ARM_IDX}, joint_lower_limits=lo, joint_upper_limits=hi), stats={})
    if links_clear is not None:
        kit.links_clear = links_clear
    return IKChecker(kit)


def _contact(x=0.6):
    eef = (th.tensor([x, -0.2, 0.85]), C.eef_quat([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]))
    return [C.pre_pose(eef), eef]


def _twin_chain(chain):
    return [C.twin_pose(p) for p in chain]


class _Solver:
    """``evaluate`` for :func:`orient.choose`: ``fresh[label]`` = the chain's configurations (None = the IK fails), a chain started
    from ``q_first`` ends in ``flipped[label]`` (default: the chain's configurations as the flip predicts them); every solution is
    remembered in ``ikc.solutions`` like ``IKChecker.exact`` does. ``calls`` = ``(label, q_first)``."""

    def __init__(self, ikc, fresh, flipped=None, ok=None):
        self.ikc, self.fresh, self.flipped, self.ok, self.calls = ikc, fresh, dict(flipped or {}), ok, []

    def __call__(self, p):
        self.calls.append((p.label, p.q_first))
        if p.q_first is None:
            qs = self.fresh.get(p.label)
        else:
            qs = self.flipped.get(p.label) or [p.q_first] + [p.q_first] * (len(p.chain) - 1)
        if qs is None:
            return False, None, None, "exact IK failed at pose 1 of 2"
        for pose, q in zip(p.chain, qs):
            self.ikc.solutions[("right", self.ikc._pose_key(pose))] = q.clone()
        if self.ok is not None and not self.ok(qs[-1]):
            return False, qs[-1], None, "fails the caller's check"
        return True, qs[-1], None, ""


def _pair(chain=None):
    chain = chain or _contact()
    return [O.Proposal("row", 0, "row", chain), O.Proposal("twin", 1, "twin", _twin_chain(chain))]


def _scored(label, rank, margin, reach, predicted=False):
    s = O.Scored(O.Proposal(label, rank, None, []), True, margin=margin, reach=reach, predicted=predicted)
    return s


# ------------------------------------------------------------------------------------------------------------------
# the order and the reason
# ------------------------------------------------------------------------------------------------------------------


def test_prefer_is_safety_then_clearance_then_reach_then_the_generators_order():
    safe, tol = 0.05, O.REACH_TOL
    assert O.prefer(_scored("a", 1, safe, 3.0), _scored("b", 0, safe - 0.001, 0.0)) < 0, "safe before unsafe, whatever the reach"
    assert O.prefer(_scored("a", 1, 0.04, 3.0), _scored("b", 0, 0.02, 0.0)) < 0, "neither safe: more clearance"
    assert O.prefer(_scored("a", 1, 0.0405, 3.0), _scored("b", 0, 0.0400, 0.0)) > 0, "clearances within 1 mrad are equal: reach"
    assert O.prefer(_scored("a", 1, 0.9, 1.0), _scored("b", 0, 0.06, 1.0 + tol + 0.01)) < 0, "both safe: the easier reach, not more margin"
    assert O.prefer(_scored("a", 1, 0.9, 1.0), _scored("b", 0, 0.06, 1.0 + tol - 0.01)) > 0, "within the tolerance: the generator's order"
    assert O.prefer(_scored("a", 0, 0.9, 1.0, predicted=True), _scored("a", 0, 0.9, 1.0)) > 0, "a solved configuration before a predicted one"


def test_reason_names_what_separates_the_best_other_orientation():
    row, twin = _scored("row", 0, 0.9, 1.0), _scored("twin", 1, 0.9, 1.0)
    assert O.reason(row, []) == "only feasible"
    assert O.reason(row, [_scored("row", 0, 0.9, 3.0, predicted=True)]) == "only feasible", "the same orientation in another branch"
    assert O.reason(twin, [_scored("row", 0, 0.02, 0.0)]) == "safety"
    assert O.reason(_scored("twin", 1, 0.04, 1.0), [_scored("row", 0, 0.02, 0.0)]) == "clearance"
    assert O.reason(twin, [_scored("row", 0, 0.9, 1.0 + O.REACH_TOL + 0.1)]) == "reach"
    assert O.reason(row, [twin]) == "default"
    # the best other orientation decides (the list is in the order), not any worse one behind it
    assert O.reason(row, [_scored("twin", 1, 0.9, 1.1), _scored("twin", 1, 0.9, 3.0, predicted=True)]) == "default"


# ------------------------------------------------------------------------------------------------------------------
# choose
# ------------------------------------------------------------------------------------------------------------------


def test_choose_takes_the_twin_in_the_rows_branch_when_that_is_easier_to_reach():
    ikc = _ikc()
    props = _pair()
    q_row = _q([0.0, -1.0, 0.0, -1.5, -1.2, 0.0, 0.0])
    q_far = _q([0.9, -2.0, 1.5, -0.5, 0.3, 0.5, 1.0])  # the twin's own solve: another shoulder branch, far from the start, no flip
    ev = _Solver(ikc, {"row": [q_row, q_row], "twin": [q_far, q_far]})
    chosen, dec = O.choose(ikc, "right", props, ev, _q([0.0, -1.0, 0.0, -1.5, 1.6, 0.0, 0.0]), stage="test")
    flipped = ikc.twin_q(q_row, "right")
    assert chosen.proposal.label == "twin" and chosen.proposal.source == "map" and not chosen.predicted
    assert ev.calls == [("row", None), ("twin", None), ("twin", ev.calls[-1][1])] and th.allclose(ev.calls[-1][1], flipped)
    assert dec["stage"] == "test" and dec["chosen"] == "twin" and dec["source"] == "map" and dec["reason"] == "reach" and dec["against_default"] is True
    kinds = [(o["label"], o["source"], bool(o.get("predicted"))) for o in dec["options"]]
    assert kinds == [("row", "ik", False), ("twin", "ik", False), ("twin", "map", True), ("twin", "map", False)]
    assert chosen.reach == pytest.approx(math.pi - 2.8, abs=1e-3)
    st = ikc.kit.stats
    assert st["orient_choices"] == 1 and st["orient_reason.reach"] == 1 and st["orient_source.map"] == 1 and st["orient_against_default"] == 1


def test_choose_re_ranks_a_flipped_option_that_verifies_worse_than_predicted():
    ikc = _ikc()
    props = _pair()
    q_row = _q([0.0, -1.0, 0.0, -1.5, -1.2, 0.0, 0.0])
    q_tw = _q([0.0, -1.0, 0.0, -1.5, 0.4, 0.0, 0.0])  # no flipped counterpart (|q5| < 45 deg), 1.2 rad from the start
    q_bad = _q([0.0, -1.0, 0.0, -1.5, 2.33, 0.0, 0.0])  # where the flipped chain really ends: 26 mrad from joint 5's limit
    ev = _Solver(ikc, {"row": [q_row, q_row], "twin": [q_tw, q_tw]}, flipped={"twin": [ikc.twin_q(q_row, "right"), q_bad]})
    chosen, dec = O.choose(ikc, "right", props, ev, _q([0.0, -1.0, 0.0, -1.5, 1.6, 0.0, 0.0]))
    assert chosen.proposal.label == "twin" and chosen.proposal.source == "ik", "the verified flip went back behind the twin's own solve"
    verified = [o for o in dec["options"] if o["source"] == "map" and not o.get("predicted")]
    assert len(verified) == 1 and verified[0]["ok"] and verified[0]["margin_rad"] == pytest.approx(0.026, abs=1e-3)
    assert dec["reason"] == "reach", "the twin's own solve (1.2 rad) against the row (2.8 rad)"


def test_choose_neither_safe_takes_more_clearance():
    ikc = _ikc()
    props = _pair()
    q_row = _q([0.3, 0.15, 0.2, -1.2, 0.5, 0.2, 0.1])  # joint 2 25 mrad from its limit, |q5| < 45 deg: no flip
    q_tw = _q([0.3, 0.13, 0.2, -1.2, 0.3, -0.2, -0.1])  # joint 2 45 mrad from it, 2 rad farther from the start
    ev = _Solver(ikc, {"row": [q_row, q_row], "twin": [q_tw, q_tw]})
    chosen, dec = O.choose(ikc, "right", props, ev, _q([0.3, 0.15, 0.2, -1.2, 2.3, 0.2, 0.1]))
    assert chosen.proposal.label == "twin" and dec["reason"] == "clearance" and chosen.margin == pytest.approx(0.0445, abs=1e-3)


def test_choose_maps_the_witnesses_a_failed_chain_left():
    ikc = _ikc()
    props = _pair()
    q_pre = _q([0.3, -1.0, 0.2, -1.2, 2.20, 0.4, -0.5])
    ikc.solutions[("right", ikc._pose_key(props[0].chain[0]))] = q_pre.clone()  # the row's pre-grasp solved; its approach stalled

    def ev(p):
        if p.label == "row":
            return False, None, None, "ik_joint_limit: right_arm_joint5 within 5 mrad of its limit at the end of the tracked approach"
        if p.q_first is None:
            return False, None, None, "exact IK failed at pose 1 of 2"
        for pose in p.chain:
            ikc.solutions[("right", ikc._pose_key(pose))] = p.q_first.clone()
        return True, p.q_first, None, ""

    chosen, dec = O.choose(ikc, "right", props, ev, _q([0.0] * 7))
    assert chosen.proposal.label == "twin" and chosen.proposal.source == "map" and th.allclose(chosen.proposal.q_first, ikc.twin_q(q_pre, "right"))
    assert dec["reason"] == "only feasible"


def test_choose_without_flipped_options():
    ikc = _ikc()
    props = _pair()
    q_row = _q([0.0, -1.0, 0.0, -1.5, -1.2, 0.0, 0.0])
    ev = _Solver(ikc, {"row": [q_row, q_row], "twin": None})
    chosen, dec = O.choose(ikc, "right", props, ev, _q([0.0, -1.0, 0.0, -1.5, 1.6, 0.0, 0.0]), map_twins=False)
    assert chosen.proposal.label == "row" and all(o["source"] == "ik" for o in dec["options"]) and len(ev.calls) == 2


def test_choose_checks_the_wrist_camera_only_between_orientations():
    ikc = _ikc(links_clear=lambda q, links, margin: (float(q[ARM_IDX[4]]) > 0.0,))  # the camera clears the world with q5 > 0
    q_row = _q([0.0, -1.0, 0.0, -1.5, -1.2, 0.0, 0.0])
    q_tw = _q([0.0, -1.0, 0.0, -1.5, 0.4, 0.0, 0.0])
    ev = _Solver(ikc, {"row": [q_row, q_row], "twin": [q_tw, q_tw]})
    chosen, dec = O.choose(ikc, "right", _pair(), ev, q_row, map_twins=False)
    row = next(o for o in dec["options"] if o["label"] == "row")
    assert chosen.proposal.label == "twin" and not row["ok"] and row["why"] == "wrist camera within 5 mm of the world at chain pose 2/2"
    # a generator's single orientation (a camera-up-only fist) is its hard rule: evaluated, not re-checked, not counted
    ikc1 = _ikc(links_clear=lambda q, links, margin: (False,))
    one, dec1 = O.choose(ikc1, "right", _pair()[:1], _Solver(ikc1, {"row": [q_row, q_row]}), q_row)
    assert one.proposal.label == "row" and dec1["reason"] == "only feasible" and "orient_choices" not in ikc1.kit.stats
    # camera_idx picks the chain poses to check (a push: its contact, not the goal)
    ikc2 = _ikc(links_clear=lambda q, links, margin: (float(q[ARM_IDX[4]]) > 0.0,))
    q_mixed = [q_tw, q_row]  # the contact clear, the goal not
    ev2 = _Solver(ikc2, {"row": q_mixed, "twin": [q_tw, q_tw]})
    chosen2, _ = O.choose(ikc2, "right", _pair(), ev2, q_tw, camera_idx=[0], map_twins=False)
    assert chosen2.proposal.label == "row"


def test_choose_tries_the_next_option_when_the_downstream_check_fails():
    ikc = _ikc()
    q = _q([0.0, -1.0, 0.0, -1.5, -1.2, 0.0, 0.0])
    ev = _Solver(ikc, {"row": [q, q], "twin": [q, q]})
    seen = []

    def downstream(s):
        seen.append(s.proposal.label)
        return (s.proposal.label != "row"), "the push fails with the brace there"

    chosen, dec = O.choose(ikc, "right", _pair(), ev, q, downstream=downstream, map_twins=False)
    assert seen == ["row", "twin"] and chosen.proposal.label == "twin" and dec["reason"] == "only feasible"
    row = next(o for o in dec["options"] if o["label"] == "row")
    assert not row["ok"] and row["why"] == "downstream: the push fails with the brace there"
    # nothing passes downstream: no choice
    none, dec_none = O.choose(ikc, "right", _pair(), ev, q, downstream=lambda s: (False, "no"), map_twins=False)
    assert none is None and dec_none["chosen"] is None and dec_none["reason"] is None


# ------------------------------------------------------------------------------------------------------------------
# pairs, choose_first
# ------------------------------------------------------------------------------------------------------------------


def test_pairs_groups_consecutive_wrist_twins():
    a, b, c = _contact(0.5)[1], _contact(0.6)[1], _contact(0.7)[1]
    stream = [a, C.twin_pose(a), b, c, C.twin_pose(c), C.twin_pose(C.twin_pose(c))]
    groups = list(O.pairs(stream, key=lambda p: p))
    assert [len(g) for g in groups] == [2, 1, 2, 1]
    assert groups[1][0] is b and groups[3][0] is stream[5]
    assert O.is_twin(a, C.twin_pose(a)) and not O.is_twin(a, b) and not O.is_twin(a, a)


def test_choose_first_takes_the_first_group_with_a_passing_member():
    ikc = _ikc()
    x, y = _contact(0.5), _contact(0.7)
    cands = [("x", x), ("x-", _twin_chain(x)), ("y", y)]
    q = _q([0.0, -1.0, 0.0, -1.5, 0.3, 0.0, 0.0])

    def ev(p):
        if p.payload[0].startswith("x"):
            return False, None, None, f"{p.payload[0]} collides"
        for pose in p.chain:
            ikc.solutions[("right", ikc._pose_key(pose))] = q.clone()
        return True, q, None, ""

    chosen, dec, fails = O.choose_first(ikc, "right", cands, key=lambda c: c[1][1], label=lambda c: c[0], chain=lambda c: c[1], evaluate=ev, q_start=q)
    assert chosen.proposal.payload[0] == "y" and dec["chosen"] == "y" and fails == ["x: x collides", "x-: x- collides"]
    assert "orient_choices" in ikc.kit.stats and ikc.kit.stats["orient_choices"] == 1, "the pair was a choice (nothing chosen); y alone is not"
    assert O.choose_first(ikc, "right", cands[:2], key=lambda c: c[1][1], label=lambda c: c[0], chain=lambda c: c[1], evaluate=ev, q_start=q)[0] is None
