"""The oracle's ``BasePush`` (owner 2026-10-12, IORS nextto large-scale): a floor-standing object pushed along a straight floor line by
the base with the closed hand on it - the executor against stub motions (approach, drive, the NextTo stop, drift, the re-contact of a
lagging object, the realign of a turning one) and the program check's tracker after the push (no simulator)."""

from __future__ import annotations

import logging
import math
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
    """A cart pushed along the floor by a hand fixed relative to the base: the hand at the base pose (x, y, yaw) moved ``reach`` plus the
    arm's own offset in the base frame; the cart's back face ``half`` behind its centre (the root; the face's outward normal its local
    -x); NextTo once the cart's centre passes ``nextto_x``; ``drift`` m sideways per base step while pushed; ``stick`` steps of the drive
    in which the cart does not follow the hand; ``spin`` rad the cart turns about the hand per base step while pushed."""

    def __init__(self, cart_x=1.0, base_x=0.25, reach=0.55, half=0.2, nextto_x=2.5, drift=0.0, stick=0, spin=0.0):
        self.cart = [cart_x, 0.0]
        self.yaw = 0.0
        self.base = [base_x, 0.0, 0.0]
        self.reach, self.half, self.nextto_x, self.drift, self.stick, self.spin = reach, half, nextto_x, drift, stick, spin
        self.arm = [-0.1, 0.0]  # the hand's offset from base + reach in the base frame (the pre-contact 10 cm back)

    def hand(self) -> Tuple[float, float]:
        bx, by, byaw = self.base
        ox, oy = self.reach + self.arm[0], self.arm[1]
        return bx + math.cos(byaw) * ox - math.sin(byaw) * oy, by + math.sin(byaw) * ox + math.cos(byaw) * oy

    def put_hand(self, x: float, y: float) -> None:
        bx, by, byaw = self.base
        dx, dy = x - bx, y - by
        self.arm = [math.cos(byaw) * dx + math.sin(byaw) * dy - self.reach, -math.sin(byaw) * dx + math.cos(byaw) * dy]

    def contact(self, heading=None, drive=False) -> None:
        """A hand past the face moves the cart until the face is back at the hand: along ``heading`` (the base's motion; the face's
        normal when None); on a drive the cart then turns ``spin`` about the hand."""
        hx, hy = self.hand()
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        depth = c * (hx - self.cart[0]) + s * (hy - self.cart[1]) + self.half
        if depth <= 0.0:
            return
        ux, uy = heading if heading is not None else (c, s)
        k = depth / max(c * ux + s * uy, 0.1)
        self.cart = [self.cart[0] + k * ux, self.cart[1] + k * uy]
        if drive and self.spin:
            cs, sn = math.cos(self.spin), math.sin(self.spin)
            rx, ry = self.cart[0] - hx, self.cart[1] - hy
            self.cart = [hx + cs * rx - sn * ry, hy + sn * rx + cs * ry]
            self.yaw += self.spin


#: the base's virtual joints in the stub's 28-D configuration (``execute._q_at_base``)
_JIDX = {"base_footprint_x_joint": 0, "base_footprint_y_joint": 1, "base_footprint_z_joint": 2, "base_footprint_rz_joint": 5}


class _Motions:
    def __init__(self, w: _World, blocked: bool = False):
        self.w = w
        self.robot = SimpleNamespace(arm_control_idx={"right": th.arange(14, 21), "left": th.arange(7, 14)}, get_joint_positions=lambda: th.zeros(28))
        self.state = SimpleNamespace(arm_target={}, trunk_error=lambda: 0.0)
        self.cfg = SimpleNamespace(base_tracker=_Tracker(), diff_ik_soft_pos_tol_free=0.02)
        self.calls: List[Any] = []
        self.checked: List[Tuple[th.Tensor, float]] = []  # the configurations and the margin of each collision check
        self.blocked = blocked
        self.backend = SimpleNamespace(update_world=lambda **kw: None, q_full=lambda: th.zeros(28), _jidx=_JIDX, world_to_root=lambda pose: pose,
                                       check_collision=self._check_collision)

    def _check_collision(self, qs, sync_world=False, margin=0.0, **kw):
        self.checked.append((th.as_tensor(qs).clone(), margin))
        return th.full((len(qs),), self.blocked)

    def eef_pose(self, arm):
        hx, hy = self.w.hand()
        return th.tensor([hx, hy, 0.45]), th.tensor([0.0, 0.0, 0.0, 1.0])

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
        self.w.put_hand(float(p[0]), float(p[1]))
        self.w.contact()
        self.calls.append(("arm_along", (round(float(p[0]), 3), round(float(p[1]), 3))))
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
        """The base in 2 cm steps along the straight line to the last waypoint, turning in proportion."""
        goal = [float(v) for v in waypoints[-1]]
        self.calls.append(("track_path", tuple(round(v, 3) for v in goal), tracker_cfg.v_max))
        x0, y0, yaw0 = self.w.base
        dx, dy, dyaw = goal[0] - x0, goal[1] - y0, C.wrap_angle(goal[2] - yaw0)
        length = math.hypot(dx, dy)
        heading = (dx / length, dy / length) if length > 1e-9 else None
        n = math.ceil(length / 0.02 - 1e-9) if heading else int(abs(dyaw) > 1e-9)
        for k in range(1, n + 1):
            f = min(1.0, k * 0.02 / length) if heading else 1.0
            hand = self.w.hand()
            self.w.base = [x0 + f * dx, y0 + f * dy, yaw0 + f * dyaw]
            if self.w.stick > 0:
                self.w.stick -= 1
                self.w.put_hand(*hand)  # the arm yields: the hand stays on the stuck cart
            else:
                self.w.contact(heading, drive=True)
                self.w.cart[1] += self.w.drift
            yield th.zeros(23)


class _State:
    def __init__(self, w: _World):
        self.w = w

    def object(self, name):
        return SimpleNamespace(name=name)

    def object_pose(self, name):
        return th.tensor([self.w.cart[0], self.w.cart[1], 0.43]), C.quat_from_yaw(self.w.yaw)

    def base_pose2d(self):
        return tuple(self.w.base)

    def nextto(self, a, b):
        assert (a, b) == (CART, TABLE)
        return self.w.cart[0] >= self.w.nextto_x


def _push(**extra) -> A.BasePush:
    q = list(C.eef_quat(th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 1.0, 0.0])).tolist())
    from ispatialgym.oracle.sidecar import Pose, PushContact

    c = PushContact(arm="right", object=CART, pre=Pose([0.70, 0.0, 0.45], q), init=Pose([0.80, 0.0, 0.45], q), goal=Pose([2.60, 0.0, 0.45], q),
                    direction=[1.0, 0.0, 0.0], distance_m=1.8, extra={"kind": "base_push", "base_start": [0.25, 0.0, 0.0], "base_end": [2.05, 0.0, 0.0],
                                                                      "contact_in_object": [-0.2, 0.0, 0.02], **extra})
    return A.BasePush(contact=c, base_yaw=0.0, partner=TABLE, end_stance="placement")


def _run(push, w, blocked=False):
    motions, state = _Motions(w, blocked=blocked), _State(w)
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


def test_realign_line_runs_from_the_footprint_centre_to_its_goal():
    """The cart (root = footprint centre, push face 0.2 m behind it) pushed 0.6 m along +x and turned 15 deg about the hand: the line runs
    from its centre to the centre's goal (2.8, 0) and enters the face where the hand goes, at the contact's height; no line once the face
    is 70 deg off it or the goal is reached / passed."""
    pose0 = (th.tensor([1.0, 0.0, 0.43]), C.quat_from_yaw(0.0))
    loc, d = [-0.2, 0.0, 0.02], [1.0, 0.0, 0.0]
    t = math.radians(15.0)
    centre = th.tensor([1.4, 0.0]) + 0.2 * th.tensor([math.cos(t), math.sin(t)])
    pose = (th.tensor([float(centre[0]), float(centre[1]), 0.43]), C.quat_from_yaw(t))
    g = EX.realign_line(pose0, pose, loc, d, 1.8)
    goal = th.tensor([2.8, 0.0])
    u = (goal - centre) / th.linalg.norm(goal - centre)
    assert th.allclose(g["centre"], centre, atol=1e-5) and th.allclose(g["goal"], goal, atol=1e-5) and th.allclose(g["direction"][:2], u, atol=1e-5)
    assert g["remaining_m"] == pytest.approx(float(th.linalg.norm(goal - centre)), abs=1e-5)
    pt = g["point"]
    assert math.cos(t) * float(pt[0] - centre[0]) + math.sin(t) * float(pt[1] - centre[1]) == pytest.approx(-0.2, abs=1e-5) and float(pt[2]) == pytest.approx(0.45, abs=1e-5)
    back = centre - pt[:2]
    assert abs(float(back[0] * u[1] - back[1] * u[0])) < 1e-5 and float(back @ u) > 0.0  # on the line, behind the centre
    assert g["line_rad"] == pytest.approx(math.atan2(float(u[1]), float(u[0])), abs=1e-6) and g["line_rad"] < 0.0 < g["offset_m"]
    assert EX.realign_line(pose0, (pose[0], C.quat_from_yaw(math.radians(70.0))), loc, d, 1.8) is None
    assert EX.realign_line(pose0, (th.tensor([2.81, 0.0, 0.43]), C.quat_from_yaw(0.0)), loc, d, 1.8) is None
    assert EX.realign_line(pose0, (th.tensor([3.2, 0.0, 0.43]), C.quat_from_yaw(0.1)), loc, d, 1.8) is None


def test_base_push_realigns_a_turning_cart_and_goes_on(caplog):
    """A cart that turns about the hand while pushed: each 15 deg turn since the push start / the last realign stops the drive; the hand
    is unloaded and backs off, the base move (collision-checked) puts the record's stand - base to pre-contact - onto the line from the
    cart's footprint centre to its goal where it enters the push face, the hand approaches again and the drive goes on along that line;
    after 3 realigns the yaw tolerance stops the drive. Each realign and the count are logged."""
    w = _World(nextto_x=99.0, spin=0.03)
    with caplog.at_level(logging.INFO, logger="ispatialgym.oracle.execute"):
        ctx, motions, post = _run(_push(yaw_tol_rad=0.35, drift_tol_m=1.0), w)
    rep = ctx["base_push"]
    assert post is None and rep["realigns"] == 3 and rep["recontacts"] == 0 and rep["stop"].startswith("drift") and "turned since realign 3" in rep["stop"]
    assert [a["leg"] for a in rep["approaches"]] == ["approach", "realign 1", "realign 2", "realign 3"] and {a["how"] for a in rep["approaches"]} == {"object moved"}
    drives = [c for c in motions.calls if c[0] == "track_path"]
    assert len(drives) == 7 and len(motions.checked) == 3
    goal = th.tensor([2.8, 0.0])
    for k, r in enumerate(rep["realign_moves"]):
        assert r["result"] == "pushing on the new line" and 14.9 < r["leg_turn_deg"] < 20.0 and r["turn_after_deg"] == pytest.approx(r["turn_deg"], abs=0.5)
        assert r["turn_deg"] == pytest.approx((k + 1) * r["leg_turn_deg"], abs=1.0)  # the turn is measured from the last realign
        x, y, yaw = r["object_pose2d"]
        centre, pt, pre = th.tensor([x, y]), th.tensor(r["point"][:2]), th.tensor(r["pre"][:2])
        u = (goal - centre) / th.linalg.norm(goal - centre)
        assert math.cos(yaw) * float(pt[0] - x) + math.sin(yaw) * float(pt[1] - y) == pytest.approx(-0.2, abs=1e-3)  # on the push face
        back = centre - pt
        assert abs(float(back[0] * u[1] - back[1] * u[0])) < 1e-3 and float(back @ u) > 0.0  # on the line from the centre to the goal
        assert th.allclose(pre, pt - (0.10 + C.FINGERTIP_LENGTH) * u, atol=1e-3)
        moves = [c for c in motions.calls if c[0] == "arm_along"]
        i = next(j for j, c in enumerate(moves) if th.allclose(th.tensor(c[1]), pre, atol=1.5e-3))
        assert th.allclose(th.tensor(moves[i + 1][1]) - pre, 0.02 * u, atol=2e-3)  # the guarded approach goes along the new line
        bx, by, byaw = r["base_move"]["to"]
        assert byaw == pytest.approx(math.radians(r["line_deg"]), abs=1e-3)
        rel = (math.cos(byaw) * float(pre[0] - bx) + math.sin(byaw) * float(pre[1] - by), -math.sin(byaw) * float(pre[0] - bx) + math.cos(byaw) * float(pre[1] - by))
        assert rel == pytest.approx((0.45, 0.0), abs=1e-3)  # the record's base_start -> pre
        q_last, margin = motions.checked[k]
        assert margin == EX.BASE_PUSH_DEFAULTS["realign_margin_m"] and [float(v) for v in q_last[-1][[0, 1, 5]]] == pytest.approx([bx, by, byaw], abs=1e-3)
        assert drives[1 + 2 * k][1] == pytest.approx((bx, by, byaw), abs=1e-3)
        assert drives[2 + 2 * k][1] == pytest.approx((bx + r["remaining_m"] * float(u[0]), by + r["remaining_m"] * float(u[1]), byaw), abs=2e-3)
    assert sum("realign 1/3:" in m or "realign 2/3:" in m or "realign 3/3:" in m for m in caplog.messages) == 3
    assert any(m.startswith("base_push(serving_cart.n.01_1): pushed") and m.endswith("0 re-contacts, 3 realigns") for m in caplog.messages)


def test_base_push_realign_needs_a_clear_base_move_and_follows_the_record():
    """The base move's collision check reports a hit: no move, the push ends at the turn (hand backed off). The record's knobs: one realign
    at 0.4 rad - past the 0.35 rad yaw tolerance, which waits for the realigns to be used; no realigns, or a record without
    ``contact_in_object``: the yaw tolerance stops the drive as before."""
    w = _World(nextto_x=99.0, spin=0.03)
    ctx, motions, post = _run(_push(yaw_tol_rad=0.35, drift_tol_m=1.0), w, blocked=True)
    rep = ctx["base_push"]
    assert post is None and rep["realigns"] == 1 and rep["stop"].startswith("turn:") and "realign 1 failed: the base move collides" in rep["stop"]
    assert rep["realign_moves"][0]["result"].startswith("failed") and len(motions.checked) == 1
    assert [c[1] for c in motions.calls if c[0] == "track_path"] == [(2.05, 0.0, 0.0)]
    hx, hy = w.hand()
    assert math.cos(w.yaw) * (hx - w.cart[0]) + math.sin(w.yaw) * (hy - w.cart[1]) + w.half < -0.09  # the hand stays backed off the face
    ctx, _, _ = _run(_push(yaw_tol_rad=0.35, drift_tol_m=1.0, realigns=1, realign_turn_rad=0.4), _World(nextto_x=99.0, spin=0.03))
    rep = ctx["base_push"]
    assert rep["realigns"] == 1 and 22.9 < rep["realign_moves"][0]["leg_turn_deg"] < 25.0 and "turned since realign 1" in rep["stop"]
    for push in (_push(yaw_tol_rad=0.35, drift_tol_m=1.0, realigns=0), _push(yaw_tol_rad=0.35, drift_tol_m=1.0)):
        if "realigns" not in push.contact.extra:
            push.contact.extra.pop("contact_in_object")
        w = _World(nextto_x=99.0, spin=0.03)
        ctx, motions, _ = _run(push, w)
        rep = ctx["base_push"]
        assert rep["realigns"] == 0 and "realign_moves" not in rep and not motions.checked and rep["turn_deg"] > 20.0
        assert rep["stop"].startswith("drift") and "deg turned (tolerance 1.00 m / 20 deg)" in rep["stop"]
    knobs = EX.base_push_knobs({"realigns": 1, "realign_turn_rad": 0.2, "realign_clear_m": 0.05})
    assert (knobs["realigns"], knobs["realign_turn_rad"], knobs["realign_clear_m"], knobs["realign_margin_m"]) == (1.0, 0.2, 0.05, 0.01)


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
