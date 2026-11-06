"""Loads the hands bear (owner 2026-10-09 audit of the v3.0-reconciled demos): the drives stop pressing before a loaded contact is let go
of, a free-hanging object is released as the fingers open, a next-to / touching release keeps the opening fingers off the partner, and a
demo whose torso a load bent visibly is discarded (``robot_wobble``)."""

from __future__ import annotations

import math

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.executor import ExecutorConfig
from ispatialgym.oracle.qa import TorsoLoad
from ispatialgym.oracle.motions import as_world_pose
from ispatialgym.oracle.programs import expand
from ispatialgym.tests.test_v2_oracle import FakeRobot, _place_goal, iors_world, make_executor, quiet_qa, success_eval


def _motions():
    sc, env = iors_world()
    _ex, motions, _backend, _rec = make_executor(env, sc)
    return env, motions


def test_unload_bleeds_the_arm_and_the_torso_onto_the_measured_joints_and_is_silent_unloaded():
    env, motions = _motions()
    r = env.robot
    motions.state.hold_current()
    assert list(motions.unload(["right"])) == []  # nothing presses: no step, no target changed
    r.q[r.trunk_control_idx[0]] += 0.02  # a load bent the torso 1.1 deg past its target
    r.q[r.arm_control_idx["right"][3]] -= 0.03
    gen = motions.unload(["right"])
    steps = []
    try:
        while True:
            steps.append(next(gen))
    except StopIteration as stop:
        out = stop.value
    assert len(steps) == motions.cfg.unload_steps
    assert th.allclose(motions.state.trunk_target, r.q[r.trunk_control_idx])
    assert th.allclose(motions.state.arm_target["right"], r.q[r.arm_control_idx["right"]])
    assert out["trunk"] == pytest.approx(0.02, abs=1e-6) and out["right"] == pytest.approx(0.03, abs=1e-6)
    assert float(steps[0][r.controller_action_idx["base"]].abs().max()) == 0.0


def test_unwind_trunk_ignores_deflections_within_the_threshold():
    env, motions = _motions()
    r = env.robot
    motions.state.hold_current()
    t0 = motions.state.trunk_target.clone()
    r.q[r.trunk_control_idx[1]] += 0.0005
    assert motions.unwind_trunk(0.001) == 0.0 and th.equal(motions.state.trunk_target, t0)
    r.q[r.trunk_control_idx[1]] += 0.002
    assert motions.unwind_trunk(0.001) == pytest.approx(0.0025, abs=1e-6)
    assert th.allclose(motions.state.trunk_target, r.q[r.trunk_control_idx])


def test_until_stops_after_the_step_that_meets_the_condition_and_closes_the_motion():
    closed = []

    def motion():
        try:
            for i in range(10):
                yield i
        finally:
            closed.append(True)

    seen = []

    def stop():
        return len(seen) >= 3

    gen = EX._until(motion(), stop)
    try:
        while True:
            seen.append(next(gen))
    except StopIteration as e:
        stopped = e.value
    assert seen == [0, 1, 2] and stopped is True and closed == [True]
    gen2 = EX._until(iter(range(4)), lambda: False)
    out = []
    try:
        while True:
            out.append(next(gen2))
    except StopIteration as e:
        assert e.value is False
    assert out == [0, 1, 2, 3]


def test_gripper_opening_reaches_the_open_command_on_its_first_step():
    env, motions = _motions()
    r = env.robot
    idx = r.controller_action_idx["gripper_right"]
    r.q[r.gripper_control_idx["right"]] = 0.02
    first = next(motions.gripper("right", close=False))
    assert float(first[idx]) == pytest.approx(1.0)


def test_free_standing_follow_turns_the_hand_rigidly_about_the_start_hinge_and_slows_while_the_torso_bends():
    from types import SimpleNamespace as NS

    hinge, axis, r = th.tensor([1.0, 0.0, 1.0]), th.tensor([0.0, 1.0, 0.0]), 0.1
    q0 = th.tensor([0.0, 0.0, 0.0, 1.0])

    def on_circle(ang, rad=r):
        return (hinge + th.tensor([rad * math.cos(ang), 0.0, rad * math.sin(ang)]), q0)

    body = {"pos": th.tensor([1.0, 0.1, 0.8]), "calls": 0}

    class State:
        def object(self, name):
            return NS(fixed_base=False)

        def object_pose(self, name):
            body["calls"] += 1
            p = body["pos"] + (th.tensor([0.005, 0.0, 0.0]) if body["calls"] > 1 else th.zeros(3))  # the body slides 5 mm once the follow runs
            return p, q0

        def joint_position(self, name, joint):
            return 0.0

    seen, trunk = [], [0.0, math.radians(0.3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    class Mot:
        cfg = NS(cartesian_speed=0.1)
        state = NS(trunk_error=lambda: trunk[min(len(seen) - 1, len(trunk) - 1)])

        def eef_pose(self, arm):
            return on_circle(0.0, 0.105)  # the grasp sits 5 mm outside the recorded handle circle

        def arm_along(self, arm, wps, converge=True, speed=None, **kw):
            seen.append(([w[0].clone() for w in wps], speed))
            return iter(())

    art = NS(object="blender", joint="j_lid", extra={"hinge_world": hinge.tolist(), "axis_world": axis.tolist(),
                                                     "object_pose": {"position": body["pos"].tolist(), "orientation": q0.tolist()}})
    targets = {"handle_poses": [on_circle(0.31), on_circle(0.62)], "q_values": [0.31, 0.62]}
    note = {}
    cfg = EX.ExecuteConfig()
    list(EX._follow_handle_poses(NS(arm="right", articulation=art), Mot(), State(), targets, cfg, note))
    info = note["handle_arc"]
    assert info["waypoints"] == 21 and info["chunks"] == 6 and info["sense"] == -1.0  # 0.62 rad / 0.03 -> 21 waypoints, 4 per chunk
    pts = [p for chunk, _ in seen for p in chunk]
    assert all(abs(float(th.linalg.norm((p - hinge)[[0, 2]])) - 0.105) < 1e-4 for p in pts)  # the grasp's own circle: no chord, no snap
    assert th.allclose(pts[-1], on_circle(0.62, 0.105)[0], atol=1e-4)  # the body's slide does not move the hinge the hand turns about
    assert info["end_off_recorded_m"] == pytest.approx(0.005, abs=1e-4)
    assert seen[1][1] is None and seen[2][1] == pytest.approx(0.1 * cfg.follow_slow_factor) and seen[3][1] is None and info["slowed"] == 1
    for obj in (NS(fixed_base=True), NS()):  # a fixed-base articulation / no fixed_base attribute: the recorded poses as they are
        seen.clear()
        st = State()
        st.object = lambda name, o=obj: o
        list(EX._follow_handle_poses(NS(arm="right", articulation=art), Mot(), st, targets, cfg, {}))
        assert len(seen) == 1 and len(seen[0][0]) == 2


def test_partner_clear_shift_moves_the_place_away_from_the_partner_by_what_the_opening_misses():
    place, partner = th.tensor([1.0, 0.0, 0.8]), th.tensor([0.9, 0.0, 0.85])
    shift, need = EX.partner_clear_shift(0.012, th.tensor([-1.0, 0.0, 0.0]), place, partner, 0.03)
    assert need == pytest.approx(0.012) and th.allclose(shift, th.tensor([0.012, 0.0, 0.0]))
    oblique = th.tensor([-math.cos(math.radians(60)), -math.sin(math.radians(60)), 0.0])
    shift, need = EX.partner_clear_shift(0.012, oblique, place, partner, 0.03)
    assert need == pytest.approx(0.024, abs=1e-6) and th.allclose(shift, th.tensor([0.024, 0.0, 0.0]), atol=1e-6)
    shift, need = EX.partner_clear_shift(0.05, None, place, partner, 0.03)
    assert need == pytest.approx(0.05) and th.allclose(shift, th.tensor([0.03, 0.0, 0.0]))
    shift, _ = EX.partner_clear_shift(0.01, None, th.tensor([0.9, 0.0, 0.5]), partner, 0.03)
    assert shift is None


def test_shift_place_targets_moves_every_place_target_and_keeps_the_orientations():
    from ispatialgym.oracle.actions import Pose

    q = [0.0, 0.0, 0.0, 1.0]
    tg = {"pre_place": Pose(position=[1.0, 0.0, 1.0], orientation=q), "place": Pose(position=[1.0, 0.0, 0.9], orientation=q),
          "descent": [Pose(position=[1.0, 0.0, 0.95], orientation=q)], "retreat": [Pose(position=[1.0, 0.0, 1.0], orientation=q)],
          "carry": [Pose(position=[0.5, 0.0, 1.0], orientation=q)], "carry_via": [], "other": 7}
    out = EX.shift_place_targets(tg, th.tensor([0.01, -0.02, 0.0]))
    for k in ("pre_place", "place"):
        p, o = as_world_pose(out[k])
        p0, _ = as_world_pose(tg[k])
        assert th.allclose(p, p0 + th.tensor([0.01, -0.02, 0.0])) and th.allclose(o, th.tensor(q))
    for k in ("descent", "retreat", "carry"):
        assert th.allclose(as_world_pose(out[k][0])[0], as_world_pose(tg[k][0])[0] + th.tensor([0.01, -0.02, 0.0]))
    assert out["carry_via"] == [] and out["other"] == 7


def test_torso_load_records_the_peak_per_action_and_forgets_a_rolled_back_attempt():
    r = FakeRobot()
    tl = TorsoLoad(r)
    a = th.zeros(M.ACTION_DIM)
    a[r.controller_action_idx["trunk"]] = r.q[r.trunk_control_idx]
    tl.observe(1, a, "grasp")
    assert tl.report()["max_deg"] == 0.0 and tl.report()["at_step"] is None
    r.q[r.trunk_control_idx[0]] += math.radians(0.8)
    tl.observe(2, a, "grasp")
    r.q[r.trunk_control_idx[0]] += math.radians(3.0)
    tl.observe(3, a, "push_close")
    rep = tl.report()
    assert rep["max_deg"] == pytest.approx(3.8, abs=1e-3) and rep["at_step"] == 3 and rep["action"] == "push_close"
    assert rep["per_action_deg"]["grasp"] == pytest.approx(0.8, abs=1e-3)
    tl.trim(2)
    assert tl.report()["max_deg"] == pytest.approx(0.8, abs=1e-3) and tl.report()["action"] == "grasp"


@pytest.mark.parametrize("deg, hard", [(2.0, True), (0.5, False)])
def test_robot_wobble_fails_an_oracle_demo_whose_torso_a_load_bent_visibly(tmp_path, deg, hard, monkeypatch):
    quiet_qa(monkeypatch)
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path, qa=True)
    seen = {"n": 0}
    real = TorsoLoad.observe

    def observe(self, step, action, name):
        dev = real(self, step, action, name)
        if name == "place" and seen["n"] < 3:
            seen["n"] += 1
            self.steps[int(step)] = (deg, name)  # the place's contact bent the torso this far at three steps
            return deg
        return dev

    monkeypatch.setattr(TorsoLoad, "observe", observe)
    result = ex.run_program(expand(sc.variant, sc))
    wobble = result["qa"]["metrics"]["robot_wobble"]
    assert wobble["max_deg"] == pytest.approx(deg, abs=1e-6) and wobble["action"] == "place"
    if hard:
        assert not result["success"] and result["reason"] == "robot_wobble" and seen["n"] == 1  # stopped at the first bent step
    else:
        assert result["success"], result["verdict"]
        assert not wobble["triggered"] and wobble["steps_over_note"] == 3


def test_goal_occlusion_fails_an_oracle_demo_with_its_own_verdict(tmp_path, monkeypatch):
    from ispatialgym.oracle.executor import Executor

    quiet_qa(monkeypatch)
    monkeypatch.setenv("ISPATIALGYM_GOAL_OCCLUSION", "on")
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path, qa=True)
    rep = {"method": "live head-camera depth", "threshold": 0.5, "objects": {}, "max_occluded": 0.62, "worst": "target_apple_0", "occluded": True}
    monkeypatch.setattr(Executor, "goal_occlusion_report", lambda self: dict(rep))
    result = ex.run_program(expand(sc.variant, sc))
    assert not result["success"] and result["reason"] == "goal_occlusion" and result["geometric_success"]
    assert result["qa"]["metrics"]["goal_occlusion"]["report"]["worst"] == "target_apple_0" and result["qa"]["stopped_by"] == "goal_occlusion"


def test_is_nextto_place_reads_the_sidecar_subtask_not_the_relation_word():
    """oracle-fix 2026-11-03: the release-clearance cap applies to NEXT-TO subtasks (no push closes the gap after the release) and not to
    touching, whose place shares the relation word "next to"; an unknown sidecar keeps the touching behaviour."""
    from types import SimpleNamespace

    from ispatialgym.oracle.execute import ExecuteConfig, _is_nextto_place

    nextto = SimpleNamespace(variant="inter_object_relation_satisfaction-nextto_small_scale-marker-colored_pencil-breakfast_table")
    nextto_large = SimpleNamespace(variant="inter_object_relation_satisfaction-nextto_large_scale-serving_cart-coffee_table-floor")
    touching = SimpleNamespace(variant="inter_object_relation_satisfaction-touching_small_scale-spray_bottle-bottle_of_detergent-desk")
    assert _is_nextto_place({"sidecar": nextto}) and _is_nextto_place({"sidecar": nextto_large})
    assert not _is_nextto_place({"sidecar": touching}) and not _is_nextto_place({}) and not _is_nextto_place({"sidecar": SimpleNamespace(variant=None)})
    cfg = ExecuteConfig()
    assert 0 < cfg.place_partner_clear_nextto_max_m < cfg.place_partner_clear_max_m  # the cap stays under the nextto tolerance of 0.03


def test_reaim_place_sets_the_object_not_the_hand_on_the_recorded_place():
    """oracle-fix 2026-11-03 (bar soap placed_orientation): with the object turned in the hand since the grasp, the re-aimed place puts the
    OBJECT where the recorded place with the recorded grasp would; never below the recorded hand; nothing when the re-aimed pose is out of
    reach, when the in-hand pose is within the thresholds, or for a non-support place."""
    from types import SimpleNamespace

    from ispatialgym.oracle.actions import Place
    from ispatialgym.oracle.execute import ExecuteConfig, reaim_place_for_inhand_pose
    from ispatialgym.oracle.sidecar import Pose
    from ispatialgym.robot import conventions as C

    def yaw_q(a):
        return th.tensor([0.0, 0.0, math.sin(a / 2), math.cos(a / 2)])

    down = th.tensor([1.0, 0.0, 0.0, 0.0])  # a top-down hand
    eio_rec = (th.tensor([0.0, 0.0, 0.03]), down)
    eio_live = C.compose((th.zeros(3), yaw_q(0.3)), (th.tensor([0.004, 0.0, 0.03]), down))  # the soap turned 0.3 rad + slid 4 mm in the hand
    obj_pose = (th.tensor([0.5, 0.2, 0.9]), yaw_q(0.1))
    eef_pose = C.compose(obj_pose, eio_live)
    soap = SimpleNamespace(name="soap")
    state = SimpleNamespace(object=lambda n: soap, held_object=lambda arm: soap, eef_pose=lambda arm: eef_pose, object_pose=lambda n: obj_pose)
    grasp = SimpleNamespace(object="bar_soap.n.01_1", arm="left", extra={"eef_in_object": {"position": eio_rec[0].tolist(), "orientation": eio_rec[1].tolist()}})
    sidecar = SimpleNamespace(grasps=[grasp])
    reach = {"ok": True}
    motions = SimpleNamespace(backend=SimpleNamespace(solve_ik=lambda *a, **k: (th.zeros(28) if reach["ok"] else None), update_world=lambda **k: None),
                              robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(28)))

    def P(x, y, z):
        return Pose(position=[x, y, z], orientation=down.tolist())

    targets = {"pre_place": P(0.6, 0.0, 1.0), "place": P(0.6, 0.0, 0.92), "descent": [P(0.6, 0.0, 0.96)], "retreat": [P(0.6, 0.0, 1.0)], "carry": [P(0.4, 0.0, 1.1)]}
    action = Place(kind="support", arm="left", object="bar_soap.n.01_1", place=None, pre_place=None, on="soap_dish.n.01_1")
    cfg = ExecuteConfig()
    ctx = {}
    out = reaim_place_for_inhand_pose(action, motions, state, sidecar, targets, ctx, cfg)
    assert out is not None and ctx["place_inhand_reaim"]["applied"] and ctx["place_inhand_reaim"]["inhand_rot_rad"] == pytest.approx(0.3, abs=1e-3)
    for k in ("place", "pre_place"):
        want = C.compose(as_world_pose(targets[k]), C.invert(eio_rec))  # where the recorded hand + recorded grasp set the object
        got = C.compose(as_world_pose(out[k]), C.invert(eio_live))  # where the re-aimed hand + live grasp set it
        lift = ctx["place_inhand_reaim"]["lift_m"]
        assert th.allclose(got[0] - th.tensor([0.0, 0.0, lift]), want[0], atol=1e-5) and C.quat_angle(got[1], want[1]) < 1e-4
    assert as_world_pose(out["place"])[0][2] >= 0.92 - 1e-6  # never below the recorded hand
    assert out["carry"] == targets["carry"] and len(out["descent"]) == 1 and len(out["retreat"]) == 1
    reach["ok"] = False
    assert reaim_place_for_inhand_pose(action, motions, state, sidecar, targets, {}, cfg) is None
    reach["ok"] = True
    still = SimpleNamespace(object=lambda n: soap, held_object=lambda arm: soap, eef_pose=lambda arm: C.compose(obj_pose, eio_rec), object_pose=lambda n: obj_pose)
    ctx2 = {}
    assert reaim_place_for_inhand_pose(action, motions, still, sidecar, targets, ctx2, cfg) is None and "thresholds" in ctx2["place_inhand_reaim"]["reason"]
    drop = Place(kind="drop", arm="left", object="bar_soap.n.01_1", place=None, pre_place=None, on="soap_dish.n.01_1")
    assert reaim_place_for_inhand_pose(drop, motions, state, sidecar, targets, {}, cfg) is None
    assert ExecuteConfig().place_inhand_reaim is False  # opt-in (ISPATIALGYM_PLACE_INHAND_REAIM=1)


def test_push_recontact_rounds_move_the_contact_with_the_object(monkeypatch):
    """oracle-fix 2026-11-03 (A.2): a free object's arc push still beyond its goal tolerance is made again from where the object lies - the
    recorded contact moved rigidly with the object, an arc of the residual about the moved centre - until within the tolerance; a skill
    error ends the rounds; nothing happens within the tolerance."""
    from types import SimpleNamespace

    from ispatialgym.oracle import execute as EX
    from ispatialgym.oracle.motions import SkillError
    from ispatialgym.oracle.sidecar import Pose
    from ispatialgym.robot import conventions as C

    def yq(a):
        return th.tensor([0.0, 0.0, math.sin(a / 2), math.cos(a / 2)])

    goal_yaw = 0.69
    obj = {"pose": (th.tensor([0.0, 0.0, 0.7]), yq(0.0))}
    pose0 = (th.tensor([0.0, 0.0, 0.7]), yq(0.0))
    init = Pose(position=[0.15, 0.0, 0.71], orientation=[1.0, 0.0, 0.0, 0.0])
    wps = [init] + [Pose(position=[float(v) for v in p.tolist()], orientation=[float(v) for v in q.tolist()])
                    for p, q in (EX._rotate_pose_about(init, th.tensor([0.0, 0.0, 0.71]), f * goal_yaw) for f in (0.5, 1.0))]
    targets = {"pre": Pose(position=[0.15, 0.0, 0.79], orientation=[1.0, 0.0, 0.0, 0.0]), "waypoints": wps, "direction": [0.0, 1.0, 0.0]}
    geom = EX.push_geometry(wps, targets["direction"])
    calls = []

    def residual(g, state, o, p0, art):
        return C.wrap_angle(goal_yaw - C.yaw_of(obj["pose"][1])), "goal_orientation"

    def arc_loop(motions, state, arm, o, arc, g, pose_now, cfg, ctx, label):
        calls.append({"arc": arc, "geom": g, "pose_now": pose_now})
        p, q = obj["pose"]
        obj["pose"] = (p + th.tensor([0.0, 0.02, 0.0]), C.qmul(yq(0.8 * float(g.angle)), q))  # the object follows 80 % of the round's arc
        yield th.zeros(1)

    def gen(*a, **k):
        yield th.zeros(1)

    monkeypatch.setattr(EX, "push_residual", residual)
    monkeypatch.setattr(EX, "_push_arc_closed_loop", arc_loop)
    monkeypatch.setattr(EX, "reach_push_pre", gen)
    monkeypatch.setattr(EX, "_hold_until_still", gen)
    state = SimpleNamespace(object_pose=lambda o: obj["pose"], eef_pose=lambda arm: (th.tensor([0.1, 0.1, 0.71]), th.tensor([1.0, 0.0, 0.0, 0.0])),
                            pose_target=lambda o: SimpleNamespace(orientation_tolerance=0.05))
    base = {"pose": [0.0, -0.6, 1.5708], "moves": []}

    def track_path(goals, tracker_cfg=None):
        base["moves"].append(list(goals[0]))
        base["pose"] = list(goals[0])
        yield th.zeros(1)

    from ispatialgym.oracle.motions import MotionsConfig

    motions = SimpleNamespace(arm_along=gen, protecting=None, base_pose2d=lambda: tuple(base["pose"]), track_path=track_path, cfg=MotionsConfig(),
                              follow_phase="arm")
    monkeypatch.setattr(EX, "realign_base_clear", lambda m, a, b, margin: None)
    action = SimpleNamespace(name="push", arm="right", contact=SimpleNamespace(object="book", arm="right"))
    cfg = EX.ExecuteConfig()
    ctx = {}
    obj["pose"] = (th.tensor([0.03, 0.04, 0.7]), yq(0.32))  # the first push turned it 0.32 rad and slid it (door_lags)
    end = (th.tensor([0.1, 0.1, 0.71]), th.tensor([1.0, 0.0, 0.0, 0.0]))
    g = EX._push_recontact_rounds(action, motions, state, targets, geom, pose0, [], [], cfg, ctx, end)
    try:
        while True:
            next(g)
    except StopIteration:
        pass
    rec = ctx["push_recontact"]
    assert len(calls) == 2 and len(rec["rounds"]) == 2 and rec["stop"] == "within the goal tolerance"
    c0 = calls[0]
    T = C.compose(c0["pose_now"], C.invert(pose0))
    want_init = C.compose(T, as_world_pose(init))  # the recorded contact moved rigidly with the object
    assert th.allclose(as_world_pose(c0["arc"][0])[0], want_init[0], atol=1e-5)
    assert float(c0["geom"].angle) == pytest.approx(goal_yaw - 0.32, abs=2e-3)  # an arc of the residual
    assert th.allclose(c0["geom"].pivot[:2], th.tensor([0.03, 0.04]), atol=1e-3)  # about the object's centre now
    assert rec["rounds"][1]["residual_after_rad"] < rec["rounds"][0]["residual_before_rad"]
    # the base followed the object's planar shift (its pivot moved 5 cm in round 1): the stance carried, the yaw kept
    assert rec["rounds"][0]["base_move"]["blocked"] is None and base["moves"][0] == pytest.approx([0.03, -0.56, 1.5708], abs=1e-3)
    # within the tolerance: nothing
    obj["pose"] = (th.tensor([0.0, 0.0, 0.7]), yq(goal_yaw - 0.01))
    calls.clear()
    ctx2 = {}
    g = EX._push_recontact_rounds(action, motions, state, targets, geom, pose0, [], [], cfg, ctx2, end)
    try:
        while True:
            next(g)
    except StopIteration:
        pass
    assert calls == [] and ctx2["push_recontact"]["stop"] == "within the goal tolerance"
    # a round that never touched the object: the next one moves the contact 2.5 cm toward the pivot
    follow = {"frac": 0.0}

    def arc_loop_miss(motions, state, arm, o, arc, g, pose_now, cfg, ctx, label):
        calls.append({"arc": arc, "geom": g, "pose_now": pose_now})
        p, q = obj["pose"]
        obj["pose"] = (p, C.qmul(yq(follow["frac"] * float(g.angle)), q))
        follow["frac"] = 0.8
        yield th.zeros(1)

    monkeypatch.setattr(EX, "_push_arc_closed_loop", arc_loop_miss)
    obj["pose"] = (th.tensor([0.0, 0.0, 0.7]), yq(0.0))  # the first push never touched it
    calls.clear()
    ctx4 = {}
    g = EX._push_recontact_rounds(action, motions, state, targets, geom, pose0, [], [], cfg, ctx4, end)
    try:
        while True:
            next(g)
    except StopIteration:
        pass
    r4 = ctx4["push_recontact"]["rounds"]
    assert r4[0]["untouched_before"] and r4[0]["inward_m"] == pytest.approx(0.025) and r4[1]["inward_m"] == pytest.approx(0.05)
    r0 = float(th.linalg.norm(as_world_pose(calls[0]["arc"][0])[0][:2] - calls[0]["geom"].pivot[:2]))
    assert r0 == pytest.approx(0.15 - 0.025, abs=1e-3)  # the contact 2.5 cm further in than the recorded 0.15 m
    # a skill error ends the rounds
    obj["pose"] = (th.tensor([0.0, 0.0, 0.7]), yq(0.3))

    def boom(*a, **k):
        raise SkillError("plan_failed", "no plan to the moved pre-contact")
        yield

    monkeypatch.setattr(EX, "reach_push_pre", boom)
    ctx3 = {}
    g = EX._push_recontact_rounds(action, motions, state, targets, geom, pose0, [], [], cfg, ctx3, end)
    try:
        while True:
            next(g)
    except StopIteration:
        pass
    assert ctx3["push_recontact"]["stop"] == "skill error" and "plan_failed" in ctx3["push_recontact"]["rounds"][0]["error"]


def test_place_touch_stop_rule():
    """oracle-fix 2026-11-03 (bar soap on its dish): a support place onto a partner stops its descent at the first touch, or once the partner
    moved 1.5 mm; only with the switch / the variant's generation key, only for relation 'on'."""
    from types import SimpleNamespace

    from ispatialgym.oracle import execute as EX

    cfg = EX.ExecuteConfig()
    place = SimpleNamespace(kind="support", on="soap_dish.n.01_1", relation="on")
    assert not EX.place_touch_stop_wanted(place, {}, cfg)
    assert EX.place_touch_stop_wanted(place, {}, EX.ExecuteConfig(place_touch_stop=True))
    EX._VARIANT_GENERATION_CACHE["v_touch"] = {"place_touch_stop": True}
    assert EX.place_touch_stop_wanted(place, {"sidecar": SimpleNamespace(variant="v_touch")}, cfg)
    assert not EX.place_touch_stop_wanted(SimpleNamespace(kind="support", on="x", relation="next to"), {}, EX.ExecuteConfig(place_touch_stop=True))
    assert not EX.place_touch_stop_wanted(SimpleNamespace(kind="drop", on="x", relation="on"), {}, EX.ExecuteConfig(place_touch_stop=True))
    w = {"touch": False, "p": th.tensor([0.0, 0.0, 0.66])}
    state = SimpleNamespace(object_pose=lambda n: (w["p"], th.tensor([0.0, 0.0, 0.0, 1.0])), touching_pair=lambda a, b: w["touch"])
    rec = {}
    stop = EX.place_touch_stop_check(state, "soap", "dish", th.tensor([0.0, 0.0, 0.66]), cfg, rec)
    assert stop() is False
    w["touch"] = True
    assert stop() is True and "touches" in rec["why"]
    w["touch"], w["p"] = False, th.tensor([0.002, 0.0, 0.66])
    assert stop() is True and "moved 2.0 mm" in rec["why"]


def test_wobble_min_steps_gate():
    """oracle-fix 2026-11-03 (the dishwasher's single-step 1.003-1.093 deg spikes): with wobble_min_steps 1 (default) the first step over the
    limit fires robot_wobble, with 3 only a run of 3 consecutive steps does; every step over is recorded."""
    from types import SimpleNamespace

    from ispatialgym.oracle import qa as QA

    def run(min_steps, devs):
        q = object.__new__(QA.QA)
        q.t = QA.QAThresholds(wobble_min_steps=min_steps)
        q.wobble = QA._Spans(10)
        q.action_name = "base_follow"
        seq = iter(devs)
        q.torso = SimpleNamespace(observe=lambda step, a, name: next(seq))
        hard = []
        for k in range(len(devs)):
            q._wobble(k, None, hard)
        return hard, q.wobble.steps

    assert QA.QAThresholds().wobble_min_steps == 1
    hard, n = run(1, [0.5, 1.003, 0.9, 0.8])
    assert [h[0] for h in hard] == ["robot_wobble"] and n == 1
    hard, n = run(3, [0.5, 1.003, 0.9, 0.8])
    assert hard == [] and n == 1
    hard, n = run(3, [1.1, 1.2, 1.3, 1.2, 0.5])
    assert [h[0] for h in hard] == ["robot_wobble"] and hard[0][2]["steps"] == 3 and n == 4
