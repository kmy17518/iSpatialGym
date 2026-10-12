"""Block design: the re-pose after a between-block base move (owner 2026-10-12 item 2, ``generation.repose_after_base_move``) - no simulator.

The sampler (``BlockDesign._block_program`` / ``_repose_record`` / ``_tracked_from``), the program (``programs.block_repose`` ->
``Grasp.repose``), the program check (``program_check.check_repose`` and the pre-grasp witness of a re-posed grasp) and the oracle
(``execute.repose_arm``). Off (the default) every call is today's.
"""

import copy
import dataclasses
import json
import os
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.sidecar import Pose, Sidecar, SidecarError
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.ik import IKChecker, IKResult
from ispatialgym.sampling.v2.stances import Stance

DATA = os.path.join(os.path.dirname(__file__), "data", "sidecars")
ALIAS = "configuration_matching-block_design-block_4-breakfast_table_board"
POSTURE = [0.8, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0]


def _bd():
    from ispatialgym.sampling.v2.tasks import block_design as BD

    return BD


def _sampler(gen):
    BD = _bd()
    s = object.__new__(BD.BlockDesign)
    s.spec = SimpleNamespace(generation=dict(gen))
    s.gp = SimpleNamespace(alias=ALIAS)
    return s


def _stance(name, x):
    return Stance(name, (x, 0.0, 0.0), 0.75, -0.3, stand_off=0.45)


# ---------------------------------------------------------------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------------------------------------------------------------


def test_tracked_legs_keep_todays_call_unless_the_block_is_reposed():
    """``_tracked_from`` calls ``track_chain`` exactly as before (no finish bound) unless the block is re-posed; then with the program
    check's continuation acceptance: the oracle's soft tolerances, iteration budget and the 0.6 rad finish bound."""
    BD = _bd()
    s = _sampler({})
    calls = []
    s.ikc = SimpleNamespace(track_chain=lambda q, poses, arm, **kw: calls.append(kw) or (q, "ok"), FINISH_MAX_EXCURSION=IKChecker.FINISH_MAX_EXCURSION,
                            ORACLE_SOFT_POS_TOL=IKChecker.ORACLE_SOFT_POS_TOL, ORACLE_SOFT_ROT_TOL=IKChecker.ORACLE_SOFT_ROT_TOL,
                            ORACLE_DIFF_IK_MAX_ITERS=IKChecker.ORACLE_DIFF_IK_MAX_ITERS)
    q0, poses, support = th.zeros(28), [(th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0]))] * 2, object()
    today = {"ignore": [support], "ignore_end": [], "soft_stall_m": 0.015, "clear_links": BD.ARM_CLEAR_LINKS, "clear_margin": 0.012}
    BD.BlockDesign._tracked_from(s, q0, poses, [support])
    assert calls[-1] == today
    s._oracle_tracking = False
    BD.BlockDesign._tracked_from(s, q0, poses, [support])
    assert calls[-1] == today
    s._oracle_tracking = True
    BD.BlockDesign._tracked_from(s, q0, poses, [support])
    assert calls[-1] == {**today, "finish_max_excursion": 0.6, "soft_pos_tol": 0.01, "soft_rot_tol": 0.03, "max_iters": 100}
    BD.BlockDesign._tracked_from(s, q0, poses, [support], hard=True)  # a reorient: no finish at all, the same acceptance
    assert calls[-1]["soft_stall_m"] is None and calls[-1]["finish_max_excursion"] == 0.6 and calls[-1]["max_iters"] == 100


def _program_sampler(gen, fail=()):
    BD = _bd()
    s = _sampler(gen)
    seen, recorded = [], []

    def chain(rng, st, *a):
        seen.append((st.name, s._oracle_tracking))
        if st.name in fail:
            raise Reject("grasp_unreachable", f"{st.name}: no class")
        return {"instance": "toy_dice.n.01_2", "grasp_class": "x/t+30/y0", "pre_pick": {"position": [0.5, 0.0, 0.9], "orientation": [0.0, 0.0, 0.0, 1.0]}}

    def repose(st, rec, *a):
        recorded.append(st.name)
        if st.name == "unplannable":
            raise Reject("repose_unplannable", "block toy_dice.n.01_2: no joint-space plan")
        return {"arm": "right", "target": "pre_grasp", "q": [0.0] * 7, "pose": rec["pre_pick"], "planned_from": "stance posture"}

    s._block_chain, s._repose_record = chain, repose
    s._centre = lambda pose: pose
    return BD, s, seen, recorded


def _run_program(s, BD, cands, step, current):
    poses = {0: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), 1: (th.ones(3), th.tensor([0.0, 0.0, 0.0, 1.0]))}
    return BD.BlockDesign._block_program(s, None, cands, ["toy_dice.n.01_1", "toy_dice.n.01_2"], poses, dict(poses), {0: {"flips": False}, 1: {"flips": True}},
                                         [0, 1], step, th.tensor([1.0, 0.0]), th.tensor([0.0, 1.0]), 0.0, current=current)


def test_block_program_reposes_only_at_a_stance_reached_by_a_base_move():
    """Off: no record, today's tracking. On: a block from block 2 on at a stance other than where the previous block left the robot is
    re-posed (record + the program check's tracking while its chain is solved); the stance it stands at is not; a re-pose that cannot
    be planned fails that stance and the next one is tried."""
    cur, other, third = _stance("cur", 0.0), _stance("other", 0.1), _stance("third", 0.2)
    BD, s, seen, recorded = _program_sampler({}, fail=("cur",))
    rec, st = _run_program(s, BD, [cur, other], 1, cur)
    assert st is other and "repose" not in rec and seen == [("cur", False), ("other", False)] and recorded == []
    BD, s, seen, recorded = _program_sampler({"repose_after_base_move": True})
    rec, st = _run_program(s, BD, [cur, other], 1, cur)
    assert st is cur and "repose" not in rec and seen == [("cur", False)] and recorded == []
    rec, st = _run_program(s, BD, [other], 1, cur)
    assert st is other and rec["repose"]["target"] == "pre_grasp" and seen[-1] == ("other", True) and recorded == ["other"] and s._oracle_tracking is False
    rec, st = _run_program(s, BD, [other], 0, cur)  # the first block: its travel comes from init
    assert "repose" not in rec and seen[-1] == ("other", False)
    unpl = _stance("unplannable", 0.3)
    rec, st = _run_program(s, BD, [unpl, third], 1, cur)
    assert st is third and rec["repose"]["planned_from"] == "stance posture" and recorded[-2:] == ["unplannable", "third"]
    with pytest.raises(Reject) as e:
        _run_program(s, BD, [unpl], 1, cur)
    assert "[repose_unplannable]" in str(e.value)


class _Kit:
    """Arm joints 0..6; ``collision_free`` refuses the configurations whose first joint is in ``collides`` (per margin)."""

    def __init__(self, plan_ok=True, collides=()):
        self.robot = SimpleNamespace(arm_control_idx={"right": th.arange(7)})
        self.collides, self.plans, self.plan_ok = dict(collides), [], plan_ok
        self.backend = SimpleNamespace(plan_joint=self.plan_joint)

    def collision_free(self, q, margin=0.0):
        bad = self.collides.get(round(float(q[0]), 3))
        return th.tensor([not (bad is not None and margin >= bad)])

    def plan_joint(self, emb, q_goal, q_start=None, max_attempts=4, sync_world=True):
        self.plans.append((emb, [round(float(v), 3) for v in q_start[:7]], [round(float(v), 3) for v in q_goal[:7]]))
        return SimpleNamespace(success=self.plan_ok, q_traj=th.zeros(6, 28), status="ok" if self.plan_ok else "trajopt failed (fake)")


def _ikc():
    def q_locked(st, right_arm=None, **kw):
        q = th.zeros(28)
        q[:7] = th.tensor(POSTURE) if right_arm is None else th.as_tensor(right_arm, dtype=th.float32)
        return q

    return SimpleNamespace(q_locked=q_locked, reaches=lambda q, pose, arm: float(th.max(th.abs(q[:3] - th.as_tensor(pose[0])))) < 0.005)


def test_repose_record_is_planned_with_the_world_present():
    """The re-pose target is the chain's first pre-grasp configuration; it must reach ``pre_pick`` and be collision-free in the world
    before the pick (placed blocks at their slots, the others at init), and a joint-space plan must exist from where the previous block
    left the arm (on the new stance) - from the stance posture when that start collides there."""
    BD = _bd()
    s = _sampler({"repose_after_base_move": True})
    teleported = []
    s._teleport = lambda objects: teleported.append(dict(objects))
    q_pre = [0.5, 0.0, 0.9, 0.1, 0.2, 0.3, 0.4]
    prev = th.zeros(28)
    prev[:7] = th.tensor([0.3, 0.1, 0.6, 0.0, 0.0, 0.0, 1.0])
    rec = {"pre_pick": {"position": [0.5, 0.0, 0.9], "orientation": [0.0, 0.0, 0.0, 1.0]}, "flip_mode": "in_place_then_move"}
    names = ["b0", "b1", "b2"]
    poses = {i: (th.full((3,), float(i)), th.tensor([0.0, 0.0, 0.0, 1.0])) for i in range(3)}
    goals = {i: (th.full((3,), 10.0 + i), th.tensor([0.0, 0.0, 0.0, 1.0])) for i in range(3)}
    st = _stance("manipulation_4", 0.1)

    def record(kit, pre=q_pre, prev_end=prev):
        s.kit, s.ikc, s._chain_pre_q, s._prev_end_q = kit, _ikc(), pre, prev_end
        return BD.BlockDesign._repose_record(s, st, rec, names, poses, goals, [2, 0, 1], 1)

    kit = _Kit()
    out = record(kit)
    assert out == {"arm": "right", "target": "pre_grasp", "q": q_pre, "pose": rec["pre_pick"], "planned_from": "previous block's end"}
    assert kit.plans == [("right_arm", [0.3, 0.1, 0.6, 0.0, 0.0, 0.0, 1.0], q_pre)]
    assert set(teleported[-1]) == {"b0", "b1", "b2"} and float(teleported[-1]["b2"][0][0]) == 12.0 and float(teleported[-1]["b0"][0][0]) == 0.0
    kit = _Kit(collides={0.3: 0.01})  # the previous end hits the world at the new stance (1 cm): the stance posture is the start
    assert record(kit)["planned_from"] == "stance posture" and kit.plans[-1][1] == POSTURE
    assert record(_Kit(), prev_end=None)["planned_from"] == "stance posture"
    kit = _Kit(plan_ok=False)
    with pytest.raises(Reject) as e:
        record(kit)
    assert e.value.reason == "repose_unplannable" and len(kit.plans) == BD.REPOSE_PLAN_ROUNDS
    for bad in (dict(pre=None), dict(pre=[0.6, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0])):  # no configuration / one that misses pre_pick
        with pytest.raises(Reject) as e:
            record(_Kit(), **bad)
        assert e.value.reason == "repose_unplannable"
    with pytest.raises(Reject) as e:
        record(_Kit(collides={0.5: 0.0}))
    assert "collides" in str(e.value)


# ---------------------------------------------------------------------------------------------------------------------------------
# program
# ---------------------------------------------------------------------------------------------------------------------------------


def _raw():
    with open(os.path.join(DATA, "configuration_matching-block_design.json")) as f:
        return json.load(f)


def _two_stances(d):
    """The last block works from ``manipulation_2`` 0.3 m along the edge (recorded path + transition), as the sampler writes it."""
    last = sorted(d["blocks"], key=lambda x: x["order"])[-1]
    manip = d["stances"]["manipulation"]
    st2 = dict(manip)
    st2["base_pose2d"] = [manip["base_pose2d"][0], manip["base_pose2d"][1] + 0.3, manip["base_pose2d"][2]]
    st2["arms"] = ["right"]
    d["stances"]["manipulation_2"] = st2
    last["stance"] = "manipulation_2"
    d.setdefault("paths", []).append({"from": "manipulation", "to": "manipulation_2", "mode": "local", "waypoints": [list(manip["base_pose2d"]), list(st2["base_pose2d"])],
                                      "length_m": 0.3, "posture": "untucked", "carried": None})
    d.setdefault("transitions", []).append({"from": "manipulation", "to": "manipulation_2", "stand_off": float(manip.get("stand_off", 0.35)), "order": ["drive-back", "trunk", "drive-in"]})
    return last


REPOSE = {"arm": "right", "target": "pre_grasp", "q": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], "pose": {"position": [7.2, 3.0, 0.95], "orientation": [0.0, 1.0, 0.0, 0.0]},
          "from_stance": "manipulation", "planned_from": "previous block's end"}


def test_program_puts_the_repose_on_the_first_grasp_after_the_base_move():
    """``blocks[k].repose`` rides on the block's first grasp right after its travel (``Grasp.repose``); without it every grasp is today's.
    A record without a base move before the block, from another stance, for another arm or malformed is a sidecar error; a flip-in-place
    block's re-grasp carries none."""
    d = _raw()
    last = _two_stances(d)
    plain = expand(ALIAS, Sidecar.from_dict(copy.deepcopy(d)))
    last["repose"] = copy.deepcopy(REPOSE)
    acts = expand(ALIAS, Sidecar.from_dict(d))
    assert [a.summary() for a in acts] == [a.summary() for a in plain]
    assert all(a.repose is None for a in plain if a.name == "grasp")
    grasps = [(i, a) for i, a in enumerate(acts) if a.name == "grasp"]
    i_last, g_last = grasps[-1]
    assert g_last.object == last["instance"] and acts[i_last - 1].name == "transition" and acts[i_last - 1].transition.to == "manipulation_2"
    assert g_last.repose == {"arm": "right", "target": "pre_grasp", "q": REPOSE["q"], "pose": Pose.from_dict(REPOSE["pose"], "p"), "from_stance": "manipulation"}
    assert all(a.repose is None for _, a in grasps[:-1])
    bad = [
        (lambda b, f: b.__setitem__("repose", copy.deepcopy(REPOSE)), "no base move"),  # the anchor: no base move before it
        (lambda b, f: f.__setitem__("repose", {**REPOSE, "from_stance": "init"}), "from_stance"),
        (lambda b, f: f.__setitem__("repose", {**REPOSE, "arm": "left"}), "arm"),
        (lambda b, f: f.__setitem__("repose", {**REPOSE, "q": [0.0] * 6}), "q"),
        (lambda b, f: f.__setitem__("repose", {**REPOSE, "target": "stance_posture"}), "target"),
        (lambda b, f: f.__setitem__("repose", {k: v for k, v in REPOSE.items() if k != "pose"}), "pose"),
    ]
    for mutate, what in bad:
        d2 = _raw()
        last2 = _two_stances(d2)
        mutate(sorted(d2["blocks"], key=lambda x: x["order"])[0], last2)
        with pytest.raises(SidecarError) as e:
            expand(ALIAS, Sidecar.from_dict(d2))
        assert what in str(e.value), (what, str(e.value))
    # a flip-in-place block after the base move: the re-pose is the first grasp's, the re-grasp's record has none
    d3 = _raw()
    last3 = _two_stances(d3)
    last3["repose"] = copy.deepcopy(REPOSE)
    last3["table_flip"] = {"in_hand_flip": True, "place": {"position": [7.2, 3.0, 0.86], "orientation": [0.0, 0.0, 0.0, 1.0]},
                           "regrasp": {"face_pair": "x", "tilt_pick_deg": 30.0, "yaw_deg": 0.0, "carry_height": 0.12,
                                       "pick": {"position": [7.2, 3.0, 0.85], "orientation": [0.0, 1.0, 0.0, 0.0]}}}
    g3 = [a for a in expand(ALIAS, Sidecar.from_dict(d3)) if a.name == "grasp" and a.object == last3["instance"]]
    assert len(g3) == 2 and g3[0].repose is not None and g3[1].repose is None and "repose" not in g3[1].block.extra


# ---------------------------------------------------------------------------------------------------------------------------------
# program check
# ---------------------------------------------------------------------------------------------------------------------------------

_QUAT = th.tensor([0.0, 1.0, 0.0, 0.0])
PRE = (th.tensor([0.60, -0.20, 0.95]), _QUAT)
GRASP = (th.tensor([0.60, -0.20, 0.85]), _QUAT)


def _q_at(pos, base=POSTURE):
    """A configuration whose toy FK puts the eef at ``pos`` (the first three arm joints are the eef position)."""
    q = th.zeros(28)
    q[:7] = th.tensor(base)
    q[:3] = th.as_tensor(pos, dtype=th.float32)
    return q


class _CheckSampler:
    """The sampler surface ``check_action_targets`` uses for a cube grasp: a toy FK (arm joints 0..2 = eef position), the IK calls
    recorded, cuRobo's joint-space planner recorded."""

    def __init__(self, plan_ok=True):
        self.calls = []
        self.objects = {"toy_dice.n.01_2": SimpleNamespace(name="targets_toy_dice_isgblk_1")}
        self.gp = SimpleNamespace(gen=lambda k, d=None: d)
        table = SimpleNamespace(name="breakfast_table_rhjoby_0")
        self.support_objects = lambda: [table]
        sampler = self

        def reaches(q, pose, arm):
            return float(th.max(th.abs(q[:3] - th.as_tensor(pose[0])))) < 0.005

        def exact(poses, stance, arm, **kw):
            sampler.calls.append(("exact", [round(float(v), 3) for v in poses[0][0]]))
            return IKResult(True, _q_at(poses[-1][0], base=[0.0] * 7), "", len(poses))

        def waypoints_ok(chain, stance, arm, q_first=None, **kw):
            sampler.calls.append(("waypoints_ok", None if q_first is None else [round(float(v), 3) for v in q_first[:7]]))
            return IKResult(True, _q_at(chain[-1][0], base=[float(v) for v in q_first[:7]]) if q_first is not None else _q_at(chain[-1][0]), "", len(chain))

        def plan_joint(emb, q_goal, q_start=None, max_attempts=4, sync_world=True):
            sampler.calls.append(("plan_joint", emb, [round(float(v), 3) for v in q_start[:7]], [round(float(v), 3) for v in q_goal[:7]]))
            return SimpleNamespace(success=plan_ok, q_traj=th.zeros(5, 28) if plan_ok else None, status="ok" if plan_ok else "trajopt failed (fake)")

        self.ikc = SimpleNamespace(q_locked=lambda st, fingers=None, **kw: _q_at([0.3, 0.0, 0.6]), reaches=reaches, exact=exact, waypoints_ok=waypoints_ok,
                                   polish_witness=lambda q, pose, arm, ignore=(): (q.clone(), 0.0), joint_limit_margin=lambda q, arm: (1.0, "right_arm_joint1"),
                                   canonical_q=lambda q, st: q, ignoring=lambda objs, ground=True: nullcontext(), solutions={}, _pose_key=lambda p: tuple(),
                                   _collision_blame=lambda q, ign: "fake blame", WITNESS_POS_TOL=IKChecker.WITNESS_POS_TOL, WITNESS_ANG_TOL=IKChecker.WITNESS_ANG_TOL,
                                   WITNESS_RESIDUAL_MAX=IKChecker.WITNESS_RESIDUAL_MAX)
        self.kit = SimpleNamespace(robot=SimpleNamespace(arm_control_idx={"right": th.arange(7), "left": th.arange(7, 14)}), floor_z=0.0,
                                   collision_free=lambda q, margin=0.0: th.tensor([True]), backend=SimpleNamespace(plan_joint=plan_joint))


def _check(monkeypatch, repose, q_from, plan_ok=True):
    from ispatialgym.oracle import actions as A
    from ispatialgym.sampling.v2 import program_check as PC

    monkeypatch.setenv("ISG_FINGERTIP_FRAME_CHECK", "off")
    sampler = _CheckSampler(plan_ok=plan_ok)
    st = Stance("manipulation_4", (0.0, 0.0, 0.0), 0.75, -0.3, stand_off=0.5)
    sidecar = SimpleNamespace(stance=lambda name: st, grasp_of=lambda name: None, blocks=[], stances={"manipulation_4": st})
    action = SimpleNamespace(name="grasp", arms=("right",), object="toy_dice.n.01_2", kind="cube", grasp=None, repose=repose)
    targets = {"pre_grasp": PRE, "grasp": GRASP, "approach": A._line(A._conventions(), Pose.from_dict({"position": PRE[0].tolist(), "orientation": PRE[1].tolist()}, "p"),
                                                                       Pose.from_dict({"position": GRASP[0].tolist(), "orientation": GRASP[1].tolist()}, "g"))}
    n, wit = PC.check_action_targets(sidecar, sampler, action, targets, "manipulation_4", None, q_seed_prev=q_from, q_end_prev=q_from)
    return sampler.calls, wit


def test_program_check_tracks_the_reposed_grasp_from_the_recorded_target(monkeypatch):
    """A re-posed grasp: the joint-space re-pose is planned from where the travel left the arm to the recorded configuration, which is
    the pre-grasp witness (no IK re-solve of the pre-grasp) the approach leaves from. Without a re-pose the calls are today's: the
    pre-grasp solved by IK, no plan."""
    from ispatialgym.sampling.v2 import program_check as PC

    q7 = [0.60, -0.20, 0.95, 0.1, 0.2, 0.3, 0.4]
    rp = {"arm": "right", "target": "pre_grasp", "q": q7, "pose": Pose.from_dict({"position": PRE[0].tolist(), "orientation": PRE[1].tolist()}, "p"), "from_stance": "manipulation"}
    travel_end = _q_at([0.40, 0.10, 0.70])
    calls, wit = _check(monkeypatch, rp, travel_end)
    assert calls[0] == ("plan_joint", "right_arm", [0.4, 0.1, 0.7] + POSTURE[3:], [round(v, 3) for v in q7])
    assert [c[0] for c in calls] == ["plan_joint", "waypoints_ok"] and calls[1][1] == [round(v, 3) for v in q7]
    assert wit["pre_grasp"]["chain"].startswith("the recorded re-pose target") and [round(v, 3) for v in wit["pre_grasp"]["q"][:7]] == [round(v, 3) for v in q7]
    assert wit["__repose__"]["planned_from"] == "where the travel left the arm" and wit["__repose__"]["rounds"] == 1
    calls, wit = _check(monkeypatch, rp, None)  # the travel re-posed the arm to the stance posture
    assert calls[0][2] == [0.3, 0.0, 0.6] + POSTURE[3:] and wit["__repose__"]["planned_from"] == "the stance posture"
    calls, wit = _check(monkeypatch, None, travel_end)  # off: today's calls
    assert [c[0] for c in calls] == ["exact", "waypoints_ok"] and calls[0][1] == [0.6, -0.2, 0.95] and "__repose__" not in wit
    with pytest.raises(Reject) as e:
        _check(monkeypatch, {**rp, "pose": Pose.from_dict({"position": [0.62, -0.2, 0.95], "orientation": PRE[1].tolist()}, "p")}, travel_end)
    assert e.value.reason == "program_check_failed" and "not this grasp's pre-grasp" in str(e.value)
    with pytest.raises(Reject) as e:
        _check(monkeypatch, {**rp, "q": [0.0] * 7}, travel_end)
    assert "misses the pre-grasp" in str(e.value)
    calls = []
    with pytest.raises(Reject) as e:
        _check(monkeypatch, rp, travel_end, plan_ok=False)
    assert e.value.reason == "repose_unplannable" and f"in {PC.REPOSE_PLAN_ROUNDS} rounds" in str(e.value)


# ---------------------------------------------------------------------------------------------------------------------------------
# oracle
# ---------------------------------------------------------------------------------------------------------------------------------


def test_oracle_grasp_reposes_before_its_transit(tmp_path):
    """``Grasp.repose``: the arm's joint-space move to the recorded configuration runs before the grasp's transit (a plan that failed
    before the robot moved is planned again); a grasp without it runs as before."""
    from ispatialgym.oracle import execute as EX
    from ispatialgym.tests.test_v2_oracle import _place_goal, _Res, iors_world, make_executor, success_eval

    q7 = [0.5, 0.1, 0.7, 0.0, 0.0, 0.0, 1.0]

    def run(path, repose):
        sc, env = iors_world()
        env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
        ex, motions, backend, _ = make_executor(env, sc, path)
        acts = expand(sc.alias, sc)
        gi = next(i for i, a in enumerate(acts) if a.name == "grasp")
        if repose:
            acts[gi] = dataclasses.replace(acts[gi], repose={"arm": "right", "target": "pre_grasp", "q": q7, "from_stance": "init",
                                                             "pose": Pose.from_dict({"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}, "p")})
        seen, failed = [], []
        orig_arm_to, orig_plan = motions.arm_to, backend.plan_joint

        def arm_to(arms, pose=None, q=None, **kw):
            seen.append(("q", [round(float(v), 3) for v in q["right"]]) if isinstance(q, dict) and "right" in q else ("pose" if pose is not None else "q", None))
            return orig_arm_to(arms, pose=pose, q=q, **kw)

        def plan_joint(emb, q_goal, q_start=None, **kw):
            if repose and not failed and th.allclose(q_goal[env.robot.arm_control_idx["right"]], th.tensor(q7)):
                failed.append(True)
                return _Res(False, status="trajopt to the joint goal failed (fake)")
            return orig_plan(emb, q_goal, q_start=q_start, **kw)

        motions.arm_to, backend.plan_joint = arm_to, plan_joint
        result = ex.run_program(acts)
        return result, result["actions"][gi], seen

    result, grasp, seen = run(tmp_path / "a", True)
    assert result["success"], result["verdict"]
    assert grasp["notes"]["repose"] == {"target": "pre_grasp", "from_stance": "init", "rounds": 2}
    k = seen.index(("q", q7))
    assert seen[k + 1] == ("q", q7) and seen[k + 2][0] == "pose"  # the failed round, the re-pose, then the transit to the pre-grasp
    result, grasp, seen = run(tmp_path / "b", False)
    assert result["success"] and "repose" not in grasp["notes"] and ("q", q7) not in seen
    assert EX.repose_arm is not None
