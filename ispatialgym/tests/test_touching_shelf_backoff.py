"""oracle-fix 2026-11-06 (owner feedback on the touching#03 demo: "instead of swinging arm backwards to reach grasp, I prefer robot moving
backwards then reaching through forward movement"): the shelf back-off (``generation.shelf_backoff_transit_m``) and the transit reach limit
(``generation.shelf_transit_max_reach_rad``) - the sampler records the backed-off transition with the pre-grasp arm, the core re-sweep keeps
it, the program check visits it pre-shaped, the oracle pre-shapes and carries the arm there inside the transition (CPU only)."""

from __future__ import annotations

import contextlib
import math
from types import SimpleNamespace

import pytest
import torch as th

IS = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_small")
from ispatialgym.oracle import execute as EX  # noqa: E402
from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from ispatialgym.sampling.sim_utils import Reject  # noqa: E402
from ispatialgym.sampling.v2 import core as CO  # noqa: E402
from ispatialgym.sampling.v2 import navigation as NAV  # noqa: E402
from ispatialgym.sampling.v2 import program_check as PCK  # noqa: E402
from ispatialgym.sampling.v2 import tabletop as TT  # noqa: E402
from ispatialgym.sampling.v2.ik import IKChecker, IKResult, stance_arm_vectors  # noqa: E402
from ispatialgym.sampling.v2.stances import GraspRecord, SidecarDraft, Stance, Transition  # noqa: E402

LEFT, RIGHT = th.arange(10, 17), th.arange(17, 24)
GRIP = {"left": th.tensor([24, 25]), "right": th.tensor([26, 27])}
UNTUCK = th.tensor([0.0, 1.57, 0.0, -1.57, 1.57, 0.0, 0.0])
#: t03tip5_kmuacd_s5 0000's recorded pre-grasp witness (the shoulder-flipped twin) and the near branch of the same pose
FLIPPED = th.tensor([-3.233, 2.629, 2.158, -1.108, 0.836, -0.962, 0.399])
NEAR = th.tensor([0.129, 0.199, -0.188, -1.107, 0.548, -0.665, -0.119])
PRESHAPE = 0.0265
YAW = -math.pi / 2  # the robot faces -y, the bookcase front at small y


class _Kit:
    """28-D configurations: base x, y at 0, 1 and yaw at 5, trunk (translate, tilt) at 6, 7, the left / right arm at 10-16 / 17-23, the fingers
    at 24-27. ``hit(q)``: the bookcase as a rule (default: the near branch's ramp passes over the bookcase top while the base stands closer
    than y 0.70 - left joint 2 between 0.3 and 1.4 midway from untucked 1.57 to 0.199). Every checked batch is kept (``checked``)."""

    def __init__(self, hit=None):
        self.robot = SimpleNamespace(arm_control_idx={"left": LEFT, "right": RIGHT}, gripper_control_idx=GRIP, trunk_control_idx=th.arange(6, 10),
                                     joint_lower_limits=th.full((28,), -3.3), joint_upper_limits=th.full((28,), 3.3))
        self.hit = hit if hit is not None else (lambda q: float(q[1]) < 0.70 and 0.3 < float(q[LEFT[1]]) < 1.4)
        self.checked = []
        self.stats = {}

    def arm_q(self, name, arm):
        if name == "ready":
            return th.tensor(M.ARM_READY_Q[arm], dtype=th.float32)
        return UNTUCK.clone() if name == "untucked" else th.zeros(7)

    def q_config(self, arms="untucked", translate=None, tilt=0.0, torso_zero=False, right_arm=None, left_arm=None, fingers=0.05):
        q = th.zeros(28)
        q[6], q[7] = float(0.5 if translate is None else translate), float(tilt)
        q[LEFT] = left_arm if left_arm is not None else self.arm_q(arms, "left")
        q[RIGHT] = right_arm if right_arm is not None else self.arm_q(arms, "right")
        q[24:28] = float(fingers)
        return q

    def q_at_base_pose(self, q, x, y, yaw):
        out = q.clone()
        out[0], out[1], out[5] = float(x), float(y), float(yaw)
        return out

    def trav_free(self, xy):
        return True

    def sync_world(self, *a, **k):
        pass

    def collision_free(self, qs, margin=0.0):
        qs = qs.unsqueeze(0) if qs.dim() == 1 else qs
        self.checked.append((qs.clone(), float(margin)))
        return th.tensor([not self.hit(q) for q in qs], dtype=th.bool)


class _IKC:
    """``exact`` returns ``solve(q_seed)`` (7 arm joints or None) on the locked configuration and remembers it for the first pose."""

    def __init__(self, kit, solve):
        self.kit, self.solve, self.solutions, self.calls = kit, solve, {}, []

    _pose_key = staticmethod(IKChecker._pose_key)

    def q_locked(self, st, arms=None, right_arm=None, left_arm=None, fingers=None):
        r, l, f = stance_arm_vectors(self.kit, st, fingers=fingers)
        return self.kit.q_at_base_pose(self.kit.q_config(translate=st.translate, tilt=st.tilt, right_arm=r, left_arm=l, fingers=f), *st.base_pose2d)

    def exact(self, poses, st, arm, q_seed=None, cache=True, fingers=None, **kw):
        self.calls.append("seeded" if q_seed is not None else "plain")
        q_arm = self.solve(q_seed)
        if q_arm is None:
            return IKResult(False, None, "exact IK failed at pose 1 of 1", len(poses))
        q = (q_seed if q_seed is not None else self.q_locked(st, fingers=fingers)).clone()
        q[self.kit.robot.arm_control_idx[arm]] = q_arm
        self.solutions[(arm, self._pose_key(poses[0]))] = q.clone()
        return IKResult(True, q, "", len(poses))

    def _collision_blame(self, *a, **k):
        return "blame"

    def ignoring(self, objs, **kw):
        return contextlib.nullcontext()

    def stance_collision_free(self, st, **kw):
        return True


class _GP:
    def __init__(self, **gen):
        self.variant, self.generation = "touching-shelf-test", dict(gen)

    def gen(self, key, default=None):
        return self.generation.get(key, default)


class _Small:
    """The IORSSmall methods of the shelf back-off on a stub world: init at stand-off 0.45 (trunk 0.321), the manipulation stance 0.1 m behind it
    at stand-off 0.55 (trunk 0.25), the arrival transition as ``record_stance_travel`` recorded it (s_T 0.55)."""

    for _n in ("shelf_transit_why", "shelf_backoff_m", "_shelf_arrival", "_shelf_backoff_floor", "_shelf_transit_start", "_shelf_backoff_arms", "_shelf_ramp_why",
               "_shelf_backoff_transit_why", "_shelf_backoff_transition", "record_shelf_backoff", "shelf_backoff_witness_why", "turn_recheck"):
        locals()[_n] = getattr(IS.IORSSmall, _n)
    departure_of = CO.SamplerV2.departure_of
    path_end_standoff = CO.SamplerV2.path_end_standoff

    def __init__(self, solve, hit=None, **gen):
        self.gp = _GP(**gen)
        self.gen = self.gp.gen
        self.kit = _Kit(hit)
        self.ikc = _IKC(self.kit, solve)
        self.shelf_start = {"lean_side": 1.0}
        self.choice = SimpleNamespace(models={"target": SimpleNamespace(scale=None)})
        self.sidecar = SidecarDraft("touching-shelf-test", "inter_object_relation_satisfaction")
        self.init = Stance("init", (0.0, 0.465, YAW), 0.32135636, 0.0, stand_off=0.45, arms="untucked")
        self.manip = Stance("manipulation", (0.0, 0.565, YAW), 0.25, 0.0, stand_off=0.55, arms=["left"], idle_posture="untucked")
        self.sidecar.stances.update({"init": self.init, "manipulation": self.manip})
        self.sidecar.transitions.append(Transition("init", "manipulation", 0.55, trunk_from=(0.32135636, 0.0), trunk_to=(0.25, 0.0)))
        q = th.tensor([0.5, 0.5, 0.5, 0.5])
        self.grasp = GraspRecord("hardback.n.01_1", "left", "hardback__acbrnv__bu", "grasp_0~r180", YAW, pre_grasp=(th.tensor([0.3, 0.043, 0.88]), q),
                                 grasp=(th.tensor([0.3, -0.057, 0.88]), q))


def test_backoff_records_the_transition_backed_off_with_the_arm_at_its_pre_grasp():
    """generation.shelf_backoff_transit_m 0.20: the transit ramp is checked at the drive-back pose (s_T 0.75, the init trunk the oracle moves the
    arms at) - clear there while at the stance it passes over the bookcase (the 0.10 m back-off still does) - and the arrival transition is
    recorded again at s_T = stand-off + 0.20 with arms_q (the left arm's pre-grasp joints, the right arm idle) and the pre-shaped fingers."""
    s = _Small(lambda q_seed: NEAR, shelf_backoff_transit_m=0.2, shelf_transit_max_reach_rad=1.6)
    assert s.shelf_transit_why(s.manip, s.grasp, "left", PRESHAPE) is None
    assert th.allclose(s.grasp.extras["_transit_q"], NEAR) and s.grasp.extras["shelf_transit"]["reach_rad"] == pytest.approx(1.371, abs=1e-3)
    assert s.grasp.extras["shelf_transit"]["s_T"] == pytest.approx(0.75)
    ramps = [qs for qs, m in s.kit.checked if len(qs) > 8 and float((qs[0, LEFT] - qs[-1, LEFT]).abs().max()) > 1.0]
    assert ramps and float(ramps[0][0, 1]) == pytest.approx(0.765) and float(ramps[0][0, 6]) == pytest.approx(0.32135636)  # drive-back pose, init trunk
    assert float(ramps[0][0, 24]) == pytest.approx(PRESHAPE) and th.allclose(ramps[0][0, LEFT], UNTUCK) and th.allclose(ramps[0][-1, LEFT], NEAR)
    tr = s.record_shelf_backoff(s.manip, s.grasp, "left", PRESHAPE)
    assert len(s.sidecar.transitions) == 1 and s.sidecar.transitions[0] is tr and tr.stand_off == pytest.approx(0.75)
    d = tr.to_sidecar()
    assert d["stand_off"] == pytest.approx(0.75) and d["arms_q"]["left"] == pytest.approx(NEAR.tolist(), abs=1e-5) and d["arms_q"]["right"] == pytest.approx(UNTUCK.tolist())
    assert abs(d["arms_q"]["left"][0]) < 1.0 and d["preshape_m"] == {"left": PRESHAPE} and d["backoff_m"] == 0.2 and d["backoff_arm"] == "left"
    assert "_backoff_tr" not in s.grasp.to_sidecar() and "_transit_q" not in s.grasp.to_sidecar() and "shelf_transit" in s.grasp.to_sidecar()
    # the drive-in was swept with the arm at its pre-grasp and the pre-shaped fingers, the drive-back from init with init's untucked arms
    drive_in = [qs for qs, m in s.kit.checked if float(qs[0, 6]) == pytest.approx(0.25) and th.allclose(qs[:, LEFT], NEAR.expand(len(qs), 7)) and len(qs) > 1]
    assert drive_in and all(float(qs[0, 24]) == pytest.approx(PRESHAPE) for qs in drive_in)
    back = [qs for qs, m in s.kit.checked if float(qs[0, 6]) == pytest.approx(0.32135636) and len(qs) > 1 and th.allclose(qs[:, LEFT], UNTUCK.expand(len(qs), 7))]
    assert back and float(back[0][0, 24]) == pytest.approx(0.05)
    assert s.kit.stats["shelf_backoff.recorded"] == 1
    # at the stance (no back-off) and backed off 0.10 m the near branch's ramp passes over the bookcase top: refused
    for gen in ({"shelf_transit_max_reach_rad": 1.6}, {"shelf_backoff_transit_m": 0.1}):
        s0 = _Small(lambda q_seed: NEAR, **gen)
        why = s0.shelf_transit_why(s0.manip, s0.grasp, "left", PRESHAPE)
        assert why is not None and why.startswith("transit check: the straight left-arm ramp") and "_transit_q" not in s0.grasp.extras


def test_backoff_transition_without_a_free_sweep_refuses_the_grasp():
    """The backed-off transition is part of the transit check: an arm at its pre-grasp that collides on the drive-in refuses the grasp, nothing
    is recorded; record_shelf_backoff without a prior check runs it itself and rejects."""
    hit = lambda q: float(q[6]) == pytest.approx(0.25) and th.allclose(q[LEFT], NEAR) and 0.6 < float(q[1]) < 0.7  # noqa: E731
    s = _Small(lambda q_seed: NEAR, hit=hit, shelf_backoff_transit_m=0.2)
    why = s.shelf_transit_why(s.manip, s.grasp, "left", PRESHAPE)
    assert why.startswith("shelf back-off: no s_T >= 0.75") and "_backoff_tr" not in s.grasp.extras
    with pytest.raises(Reject) as e:
        s.record_shelf_backoff(s.manip, s.grasp, "left", PRESHAPE)
    assert e.value.reason == "no_transition" and s.sidecar.transitions[0].stand_off == pytest.approx(0.55)


def test_reach_limit_rejects_the_flipped_branch_and_accepts_the_near_one():
    """generation.shelf_transit_max_reach_rad 1.6: the recorded flipped twin [-3.233, 2.629, ...] lies 3.23 rad from untucked - re-solved seeded at
    the transit's start posture (then ready) it is replaced by the near branch, or the grasp is refused when every re-solve lands there too;
    the near branch (1.37 rad) passes without a re-solve."""
    s = _Small(lambda q_seed: FLIPPED if q_seed is None else NEAR, shelf_backoff_transit_m=0.2, shelf_transit_max_reach_rad=1.6)
    assert s.shelf_transit_why(s.manip, s.grasp, "left", PRESHAPE) is None
    f = s.grasp.extras["shelf_transit"]
    assert th.allclose(s.grasp.extras["_transit_q"], NEAR) and f["first_reach_rad"] == pytest.approx(3.233, abs=1e-3) and f["reach_rad"] == pytest.approx(1.371, abs=1e-3)
    assert f["resolved_from"] == "the transit's start posture" and s.ikc.calls == ["plain", "seeded"] and s.kit.stats["shelf_transit.reach_resolved"] == 1
    assert th.allclose(s.ikc.solutions[("left", s.ikc._pose_key(s.grasp.pre_grasp))][LEFT], NEAR)  # the program check's witness follows
    s = _Small(lambda q_seed: FLIPPED, shelf_backoff_transit_m=0.2, shelf_transit_max_reach_rad=1.6)
    why = s.shelf_transit_why(s.manip, s.grasp, "left", PRESHAPE)
    assert why.startswith("transit check: the pre-grasp witness lies 3.23 rad") and s.ikc.calls == ["plain", "seeded", "seeded"]
    assert s.kit.stats["shelf_transit.reach_rejected"] == 1
    s = _Small(lambda q_seed: NEAR, shelf_backoff_transit_m=0.2, shelf_transit_max_reach_rad=1.6)
    assert s.shelf_transit_why(s.manip, s.grasp, "left", PRESHAPE) is None and s.ikc.calls == ["plain"] and "first_reach_rad" not in s.grasp.extras["shelf_transit"]
    # without the reach limit the flipped twin passes the back-off (its ramp swings up and away), as the recorded demo did
    s = _Small(lambda q_seed: FLIPPED, shelf_backoff_transit_m=0.2)
    assert s.shelf_transit_why(s.manip, s.grasp, "left", PRESHAPE) is None and th.allclose(s.grasp.extras["_transit_q"], FLIPPED)


def test_knobs_off_keep_the_transit_check_and_the_memo_key():
    """Both knobs off (every alias but touching#03): shelf_transit_why is the 2026-11-04 check (ramp from the stance's configuration) and the
    fixed-start memo key is unchanged."""
    s = _Small(lambda q_seed: NEAR)
    assert s.shelf_backoff_m() == 0.0
    why = s.shelf_transit_why(s.manip, s.grasp, "left", PRESHAPE)
    assert why == "transit check: the straight left-arm ramp from the stance to the pre-grasp witness penetrates (direct and via ready)"
    assert "_transit_q" not in s.grasp.extras
    s.shelf_start = None
    s.gp.generation["shelf_backoff_transit_m"] = 0.2
    assert s.shelf_backoff_m() == 0.0  # not a shelf layout
    k = SimpleNamespace(ikc=SimpleNamespace(q_locked=lambda st: th.zeros(3)), gen=_GP().gen, shelf_backoff_m=lambda: 0.0)
    assert IS.IORSSmall._fixed_start_memo_key(k, s.manip, "left", PRESHAPE, False)[-1] is False
    k.shelf_backoff_m = lambda: 0.2
    assert IS.IORSSmall._fixed_start_memo_key(k, s.manip, "left", PRESHAPE, False)[-3:] == ("backoff", 0.2, None)


def test_core_resweep_keeps_the_backoff_record_and_sweeps_its_arms():
    """check_stances_with_objects (the objects placed): a transition record carrying arms_q / preshape_m is re-swept with them - its drive-in
    with the arm at the pre-grasp - and the new record keeps them; a placed object in the way of that drive-in rejects (no_transition)."""

    class _Core:
        check_stances_with_objects = CO.SamplerV2.check_stances_with_objects
        departure_of = CO.SamplerV2.departure_of
        path_end_standoff = CO.SamplerV2.path_end_standoff
        object_joints_of = CO.SamplerV2.object_joints_of
        carried_attachment = CO.SamplerV2.carried_attachment
        carry_arm_q = CO.SamplerV2.carry_arm_q

        def __init__(self, hit=None):
            small = _Small(lambda q_seed: NEAR, hit=hit or (lambda q: False), shelf_backoff_transit_m=0.2)
            assert small.shelf_transit_why(small.manip, small.grasp, "left", PRESHAPE) is None
            small.record_shelf_backoff(small.manip, small.grasp, "left", PRESHAPE)
            self.sidecar, self.kit, self.ikc, self.gp, self.objects = small.sidecar, small.kit, small.ikc, small.gp, {}

        def stance_check_ignores(self, st):
            return []

    c = _Core()
    c.kit.checked.clear()
    c.check_stances_with_objects(None)
    tr = c.sidecar.transitions[0]
    assert tr.stand_off == pytest.approx(0.75) and tr.extras["arms_q"]["left"] == pytest.approx(NEAR.tolist(), abs=1e-5) and tr.extras["preshape_m"] == {"left": PRESHAPE}
    assert any(float(qs[0, 6]) == pytest.approx(0.25) and th.allclose(qs[:, LEFT], NEAR.expand(len(qs), 7)) and float(qs[0, 24]) == pytest.approx(PRESHAPE)
               for qs, m in c.kit.checked if len(qs) > 1)
    c = _Core()
    c.kit.hit = lambda q: th.allclose(q[LEFT], NEAR) and float(q[1]) < 0.62  # the placed book 5 cm before the stance
    with pytest.raises(Reject) as e:
        c.check_stances_with_objects(None)
    assert e.value.reason == "no_transition"
    assert NAV.recorded_arms_to({}) == (None, None) and NAV.recorded_arms_to({"arms_q": {"left": [1, 2]}, "preshape_m": {"left": 0.02, "right": 0.03}}) == (None, 0.03)


def test_program_check_visits_the_transition_pre_shaped_and_the_witness_must_follow_arms_q():
    """The program check's base-pose checks of a transition use its record's preshape_m; after the program check the pre-grasp witness of the
    backed-off arm must lie within SHELF_BACKOFF_WITNESS_GAP_RAD of arms_q (a witness re-solved into another branch would be swung to by
    the grasp); a held turn is not offered while the record carries the arm to this turn's pre-grasp."""
    act = SimpleNamespace(transition=SimpleNamespace(extra={"preshape_m": {"left": PRESHAPE}}))
    assert PCK._record_preshape(act) == PRESHAPE and PCK._record_preshape(SimpleNamespace(transition=SimpleNamespace(extra={}))) is None
    s = _Small(lambda q_seed: NEAR, shelf_backoff_transit_m=0.2)
    assert s.shelf_backoff_witness_why() is None  # no back-off record
    assert s.shelf_transit_why(s.manip, s.grasp, "left", PRESHAPE) is None
    s.record_shelf_backoff(s.manip, s.grasp, "left", PRESHAPE)
    assert s.shelf_backoff_witness_why().startswith("shelf back-off: no pre-grasp witness")

    def witness(q_arm):
        q = th.zeros(28)
        q[LEFT] = q_arm
        return [{"action": 1, "name": "grasp", "arm": "left", "targets": {"pre_grasp": {"q": q.tolist()}}}]

    s.sidecar.ik_witnesses = witness(NEAR + 0.05)  # the settled book a few mm off
    assert s.shelf_backoff_witness_why() is None
    s.sidecar.ik_witnesses = witness(FLIPPED)
    assert "3.36 rad from the joints the backed-off transition carries the arm to" in s.shelf_backoff_witness_why()
    with pytest.raises(TT.TurnNotOffered):
        s.turn_recheck(None, s.grasp)


class _Motions:
    """Records the motions in order; the gripper command as ``Motions.gripper_command`` (opening 0..0.05 m -> -1..1)."""

    def __init__(self):
        self.calls, self.transit_preshape = [], {}
        self.state = SimpleNamespace(gripper_cmd={"left": 1.0, "right": 1.0})
        self.gripper_intent_closed = {}

    def _step(self):
        yield th.zeros(1)

    def gripper_command(self, arm, q_m):
        return max(-1.0, min(1.0, 2.0 * float(q_m) / 0.05 - 1.0))

    def gripper_closed(self, arm):
        return bool(self.gripper_intent_closed.get(arm, False))

    def gripper_opening(self, arm, q_m, steps=None):
        self.calls.append(("gripper_opening", arm, round(float(q_m), 4)))
        self.state.gripper_cmd[arm], self.gripper_intent_closed[arm] = self.gripper_command(arm, q_m), False
        yield from self._step()

    def gripper(self, arm, close, **kw):
        self.calls.append(("gripper", arm, bool(close)))
        self.state.gripper_cmd[arm], self.gripper_intent_closed[arm] = (-1.0 if close else 1.0), bool(close)
        yield from self._step()

    def arm_to(self, arms, q=None, **kw):
        self.calls.append(("arm_to", tuple(arms), {a: [round(float(v), 3) for v in qq.tolist()] for a, qq in q.items()}))
        yield from self._step()

    def trunk_to(self, t, k):
        self.calls.append(("trunk_to", round(float(t), 3), round(float(k), 3)))
        yield from self._step()

    def track_path(self, wps, **kw):
        self.calls.append(("track_path", [[round(float(v), 3) for v in w] for w in wps]))
        yield from self._step()


def test_oracle_transition_preshapes_then_carries_the_arm_then_drives_in_with_it_frozen():
    """execute_transition with arms_q + preshape (the back-off record): drive back, the fingers pre-shaped, the arm to its pre-grasp joints (one
    arm_to, a straight ramp first in Motions), the trunk, then the drive-in - one straight base leg, no arm motion after the arm_to; the grasp
    that follows keeps the pre-shape (no full opening at the pre-grasp) while the command still holds it."""
    from ispatialgym.oracle.actions import Transition as TransitionAction, _preshape_of

    q = th.zeros(28)
    q[LEFT], q[RIGHT] = UNTUCK, UNTUCK
    robot = SimpleNamespace(get_joint_positions=lambda: q, arm_control_idx={"left": LEFT, "right": RIGHT})
    state = SimpleNamespace(robot=robot, held_object=lambda arm: None)
    mo = _Motions()
    targets = {"drive_back": [0.0, 0.765, YAW], "trunk": (0.25, 0.0), "drive_in": [0.0, 0.565, YAW],
               "arms_q": {"left": NEAR.tolist(), "right": UNTUCK.tolist()}, "preshape": {"left": PRESHAPE}}
    ctx = {}
    action = SimpleNamespace(stance_to=SimpleNamespace(extra={}, working_arms=["left"]))
    list(EX.execute_transition(action, mo, state, targets, ctx))
    names = [c[0] for c in mo.calls]
    assert names == ["track_path", "gripper_opening", "arm_to", "trunk_to", "track_path"], mo.calls
    assert mo.calls[1] == ("gripper_opening", "left", PRESHAPE) and mo.calls[2][1] == ("left",) and mo.calls[2][2]["left"] == pytest.approx(NEAR.tolist(), abs=1e-3)
    assert mo.calls[4] == ("track_path", [[0.0, 0.565, round(YAW, 3)]]) and ctx["preshape_set"] == {"left": PRESHAPE}
    assert EX.transit_preshape_held(mo, "left") == pytest.approx(PRESHAPE) and EX.transit_preshape_held(mo, "right") is None
    list(mo.gripper("left", close=False))
    assert EX.transit_preshape_held(mo, "left") is None  # opened since: the grasp opens as before
    # without preshape (every record before 2026-11-06) the transition is as before
    mo = _Motions()
    list(EX.execute_transition(action, mo, state, dict(targets, preshape=None), {}))
    assert [c[0] for c in mo.calls] == ["track_path", "arm_to", "trunk_to", "track_path"] and EX.transit_preshape_held(mo, "left") is None
    assert _preshape_of({"preshape_m": {"left": "x"}}) is None and _preshape_of({"preshape_m": {"left": 0.02}}) == {"left": 0.02} and _preshape_of({}) is None
    assert "preshape" in TransitionAction.TARGET_KEYS
