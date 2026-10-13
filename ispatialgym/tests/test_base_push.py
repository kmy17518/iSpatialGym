"""The oracle's ``BasePush`` (owner 2026-10-12, IORS nextto large-scale): a floor-standing object pushed along a straight floor line by
the base with the closed hand on it - the executor against stub motions (approach, drive, the NextTo stop, drift, the re-contact of a
lagging object) and the program check's tracker after the push (no simulator)."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest
import torch as th

from ispatialgym.oracle import actions as A
from ispatialgym.oracle import execute as EX
from ispatialgym.oracle.motions import as_world_pose
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import program_check as PC
from ispatialgym.sampling.v2.stances import SidecarDraft, Stance

CART = "serving_cart.n.01_1"
TABLE = "breakfast_table.n.01_1"


# ----------------------------------------------------------------------------------------------------------------
# oracle: executor with stub motions
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class _Tracker:
    v_max: Tuple[float, float, float] = (0.3, 0.3, 0.5)


class _World:
    """A cart on a line (x) pushed by a hand fixed relative to the base: the hand at base x + ``reach`` (plus the arm's own
    moves), the cart's back face ``half`` behind its centre; NextTo once the cart's centre passes ``nextto_x``; ``drift`` m sideways
    per base step while pushed; ``stick`` steps of the drive in which the cart does not follow the hand."""

    def __init__(self, cart_x=1.0, base_x=0.25, reach=0.55, half=0.2, nextto_x=2.5, drift=0.0, stick=0):
        self.cart = [cart_x, 0.0]
        self.base = [base_x, 0.0, 0.0]
        self.reach, self.half, self.nextto_x, self.drift, self.stick = reach, half, nextto_x, drift, stick
        self.arm = -0.1  # the hand's offset from base + reach (the pre-contact 10 cm back)

    def hand_x(self) -> float:
        return self.base[0] + self.reach + self.arm

    def contact(self) -> None:
        face = self.cart[0] - self.half
        if self.hand_x() > face:
            self.cart[0] += self.hand_x() - face


class _Motions:
    def __init__(self, w: _World):
        self.w = w
        self.robot = SimpleNamespace(arm_control_idx={"right": th.arange(14, 21), "left": th.arange(7, 14)}, get_joint_positions=lambda: th.zeros(28))
        self.state = SimpleNamespace(arm_target={}, trunk_error=lambda: 0.0)
        self.cfg = SimpleNamespace(base_tracker=_Tracker(), diff_ik_soft_pos_tol_free=0.02)
        self.calls: List[Any] = []
        self.backend = SimpleNamespace(update_world=lambda **kw: None)

    def eef_pose(self, arm):
        return th.tensor([self.w.hand_x(), 0.0, 0.45]), th.tensor([0.0, 0.0, 0.0, 1.0])

    def _deflection(self, arm):
        return None

    def base_pose2d(self):
        return tuple(self.w.base)

    def gripper(self, arm, close, **kw):
        self.calls.append(("gripper", close))
        yield th.zeros(23)

    def arm_to(self, arm, pose=None, **kw):
        self.calls.append(("arm_to", sorted(kw)))
        yield th.zeros(23)

    def arm_along(self, arm, waypoints, loaded=False, **kw):
        p = as_world_pose(waypoints[-1])[0]
        self.w.arm += float(p[0]) - self.w.hand_x()
        self.w.contact()
        self.calls.append(("arm_along", round(float(p[0]), 3)))
        yield th.zeros(23)

    def hold(self, steps=1):
        for _ in range(int(steps)):
            yield th.zeros(23)

    def settle(self, max_steps=None, vel_tol=None):
        yield th.zeros(23)

    def unload(self, arms, steps=None, trunk=True):
        if False:
            yield th.zeros(23)
        self.calls.append(("unload",))
        return {a: 0.0 for a in arms}

    def track_path(self, waypoints, tracker_cfg=None, **kw):
        self.calls.append(("track_path", tuple(round(v, 3) for v in waypoints[-1]), tracker_cfg.v_max))
        goal = waypoints[-1][0]
        step = 0.02
        while self.w.base[0] < goal - 1e-9:
            self.w.base[0] = min(goal, self.w.base[0] + step)
            if self.w.stick > 0:
                self.w.stick -= 1
                self.w.arm -= step  # the arm yields: the hand stays on the stuck cart
            else:
                self.w.contact()
                self.w.cart[1] += self.w.drift
            yield th.zeros(23)


class _State:
    def __init__(self, w: _World):
        self.w = w

    def object(self, name):
        return SimpleNamespace(name=name)

    def object_pose(self, name):
        return th.tensor([self.w.cart[0], self.w.cart[1], 0.43]), th.tensor([0.0, 0.0, 0.0, 1.0])

    def base_pose2d(self):
        return tuple(self.w.base)

    def nextto(self, a, b):
        assert (a, b) == (CART, TABLE)
        return self.w.cart[0] >= self.w.nextto_x


def _push(**extra) -> A.BasePush:
    q = list(C.eef_quat(th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 1.0, 0.0])).tolist())
    from ispatialgym.oracle.sidecar import Pose, PushContact

    c = PushContact(arm="right", object=CART, pre=Pose([0.70, 0.0, 0.45], q), init=Pose([0.80, 0.0, 0.45], q), goal=Pose([2.60, 0.0, 0.45], q),
                    direction=[1.0, 0.0, 0.0], distance_m=1.8, extra={"kind": "base_push", "base_start": [0.25, 0.0, 0.0], "base_end": [2.05, 0.0, 0.0], **extra})
    return A.BasePush(contact=c, base_yaw=0.0, partner=TABLE, end_stance="placement")


def _run(push, w):
    motions, state = _Motions(w), _State(w)
    ctx: Dict[str, Any] = {"cfg": EX.ExecuteConfig(), "hooks": {}}
    list(EX.execute_base_push(push, motions, state, push.targets(None, state), ctx))
    return ctx, motions, EX.check_postcondition(push, state, {}, ctx)


def test_base_push_stops_after_nextto_and_the_push_on():
    """Pre-contact, guarded approach to the face (the cart moves), the arm held, the base driven toward the recorded end; NextTo
    holds once the cart passes 2 m, the drive goes on ``nextto_push_on_m`` and stops; the arm unloaded; the record says so."""
    w = _World(nextto_x=2.0)
    ctx, motions, post = _run(_push(nextto_push_on_m=0.10), w)
    rep = ctx["base_push"]
    assert post is None and rep["result"] == "pushed" and rep["stop"].startswith("nextto") and rep["nextto_end"] is True
    assert motions.calls[0] == ("gripper", True) and motions.calls[1][0] == "arm_to" and rep["pre"] == "the object present"
    assert rep["approaches"][0]["how"] == "object moved" and rep["recontacts"] == 0
    first = rep["nextto_first_at_m"]
    assert first is not None and rep["pushed_m"] == pytest.approx(first + 0.10, abs=0.03) and rep["pushed_m"] < 1.8 - 0.2
    assert rep["drift_m"] == pytest.approx(0.0) and [c for c in motions.calls if c[0] == "track_path"][0][1] == (2.05, 0.0, 0.0)
    assert [c for c in motions.calls if c[0] == "track_path"][0][2] == (0.08, 0.08, 0.2)  # the recorded speed default


def test_base_push_runs_the_distance_without_nextto_and_stops_on_drift():
    w = _World(nextto_x=99.0)
    ctx, _, post = _run(_push(), w)
    rep = ctx["base_push"]
    assert post is None and rep["stop"] == "distance done" and rep["nextto_end"] is False and rep["base_travel_m"] == pytest.approx(1.8, abs=1e-6)
    w = _World(nextto_x=99.0, drift=0.004)
    ctx, _, _ = _run(_push(drift_tol_m=0.05), w)
    rep = ctx["base_push"]
    assert rep["stop"].startswith("drift") and abs(rep["drift_m"]) == pytest.approx(0.052, abs=0.005) and rep["pushed_m"] < 1.0


def test_base_push_recontacts_a_lagging_cart_and_gives_up_without_contact():
    """The cart sticks while the base drives: once it lags the hand by ``lag_m`` the drive stops, the arm is unloaded, the hand backs
    off 2 cm and approaches again; then the drive goes on. Without contact at all (the face out of reach) nothing is driven:
    ``no_motion``."""
    w = _World(nextto_x=99.0, stick=4)
    ctx, motions, post = _run(_push(lag_m=0.05), w)
    rep = ctx["base_push"]
    assert rep["recontacts"] == 1 and rep["lags"][0].startswith("lag") and len(rep["approaches"]) == 2 and post is None
    assert rep["stop"] == "distance done" and ("unload",) in motions.calls
    w = _World(cart_x=5.0, nextto_x=99.0)
    ctx, motions, post = _run(_push(), w)
    assert ctx["base_push"]["result"].startswith("no contact") and post == "no_motion" and not [c for c in motions.calls if c[0] == "track_path"]


def test_base_push_turn_limit_is_35_deg_unless_the_record_sets_it():
    """A single-hand push turns the cart ~20 deg by mid-push: the drive stops at 35 deg (0.61 rad) unless the record's contacts.push
    extra sets yaw_tol_rad."""
    assert EX.BASE_PUSH_DEFAULTS["yaw_tol_rad"] == pytest.approx(0.61)
    assert EX.base_push_knobs({"yaw_tol_rad": 0.35})["yaw_tol_rad"] == pytest.approx(0.35)
    assert EX.base_push_knobs({"drift_tol_m": 0.1})["yaw_tol_rad"] == pytest.approx(0.61)


def test_nextto_now_reads_the_state_or_gives_none():
    w = _World(cart_x=3.0)
    assert EX.nextto_now(_State(w), CART, TABLE) is True
    assert EX.nextto_now(SimpleNamespace(object=lambda n: SimpleNamespace(states={})), CART, TABLE) is None
    assert EX.push_frame_offset((0.0, 0.0), (1.0, 0.5), (0.0, 2.0)) == pytest.approx((0.5, -1.0))
    assert EX.base_push_knobs({"lag_m": 0.07, "drift_tol_m": True})["lag_m"] == 0.07 and EX.base_push_knobs(None) == EX.BASE_PUSH_DEFAULTS


# ----------------------------------------------------------------------------------------------------------------
# program check: the tracker after a base push
# ----------------------------------------------------------------------------------------------------------------


def test_program_check_tracker_moves_to_the_push_end_with_the_cart():
    sd = SidecarDraft("inter_object_relation_satisfaction-nextto_large_scale-serving_cart-breakfast_table-floor", "inter_object_relation_satisfaction")
    for name, xy in (("init", (0.0, 0.0)), ("manipulation", (1.0, 0.0)), ("placement", (2.8, 0.0))):
        sd.add_stance(Stance(name, (xy[0], xy[1], 0.0), 0.75, -0.3))

    class Obj:
        def __init__(self):
            self.p, self.q = th.tensor([1.75, 0.0, 0.43]), th.tensor([0.0, 0.0, 0.0, 1.0])

        def get_position_orientation(self):
            return self.p.clone(), self.q.clone()

        def set_position_orientation(self, p, q):
            self.p, self.q = th.as_tensor(p).clone(), th.as_tensor(q).clone()

        def keep_still(self):
            pass

    cart = Obj()
    sampler = SimpleNamespace(objects={CART: cart}, kit=SimpleNamespace(sync_world=lambda: None))
    tr = PC._Tracker(sd)
    tr.stance = "manipulation"
    push = _push()
    assert tr.before(push) == ("manipulation", None)
    tr.after(push, {}, sampler)
    assert tr.stance == "placement" and th.allclose(cart.p, th.tensor([3.55, 0.0, 0.43]))
    tr.restore_moved(sampler)
    assert th.allclose(cart.p, th.tensor([1.75, 0.0, 0.43]))


def test_base_push_action_contract():
    """The record's base line is required; the targets carry the pre-contact, the approach, the contact and the line; the hand ends where
    the state says it is, else at the recorded end; the executor table covers the action."""
    push = _push()
    t = push.targets(None, None)
    assert set(t) == set(push.TARGET_KEYS) and t["base_path"] == [[0.25, 0.0, 0.0], [2.05, 0.0, 0.0]] and len(t["approach"]) == A.N_WAYPOINTS
    assert t["approach"][-1].position == pytest.approx(t["contact"].position, abs=1e-6) and push.end_pose(None, None) == push.contact.goal
    here = (th.tensor([1.0, 0.0, 0.45]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    assert push.end_pose(None, SimpleNamespace(eef_pose=lambda arm: here)).position == pytest.approx([1.0, 0.0, 0.45])
    assert push.qa_exempt == frozenset({A.QA_FAILED_GRASP}) and A.BasePush in A.VOCABULARY and EX.coverage()["BasePush"]
    push.contact.extra.pop("base_end")
    with pytest.raises(Exception, match="base_end"):
        push.targets(None, None)
