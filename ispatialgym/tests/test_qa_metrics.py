"""The ``run_isg --qa`` metrics (``ispatialgym.oracle.qa.QA``, owner 2026-10-14): BEHAVIOR's demo QA (contacts, pause, jerky
motion, ghost hand, gripper in view, failed grasp, head camera upright) plus the former oracle demo gates (object tilt, goal
occlusion, robot wobble, bad release), on the fake robot of ``test_v2_oracle``. In an oracle rollout the first hard metric is
the verdict that fails the demo; in a stream every metric is only logged."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle import qa as QA
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.executor import Executor, ExecutorConfig
from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.recorder import PHASE_ORACLE, PHASE_SETTLE
from ispatialgym.robot import conventions as C
from ispatialgym.tests.test_v2_oracle import (CAM_KEY, FakeContacts, FakeLink, FakeObject, FakeRobot, FakeScene, _place_goal, iors_world, make_executor,
                                              quiet_qa, success_eval)
from ispatialgym.tests.test_vec_stream import _State, make_world

NO_PAUSE = QA.QAThresholds(pause_steps=10**6)


class QARobot(FakeRobot):
    """The fake robot with BEHAVIOR's link groups (arm / gripper / finger links) and a ``torso_link4`` pitched by ``head_quat``."""

    def __init__(self):
        super().__init__()
        self.head_quat = C.IDENTITY_QUAT.clone()
        self.links["torso_link4"] = FakeLink(f"{self.prim_path}/torso_link4", lambda: (self.pos, self.head_quat))
        for a in M.ARMS:
            self.links[f"{a}_arm_link1"] = FakeLink(f"{self.prim_path}/{a}_arm_link1", lambda a=a: self.eef_pose(a))
        self.arm_links = {a: [self.links[f"{a}_arm_link1"]] for a in M.ARMS}
        self.gripper_links = {a: [self.links[M.GRIPPER_LINK[a]]] for a in M.ARMS}
        self.finger_links = {a: [self.links[n] for n in M.FINGER_LINKS[a]] for a in M.ARMS}


def hold(robot, grip=(-1.0, -1.0)) -> th.Tensor:
    """The action holding the arms and the trunk where they are, the grippers commanded ``grip`` (left, right)."""
    a = th.zeros(M.ACTION_DIM)
    idx = robot.controller_action_idx
    a[idx["trunk"]] = robot.q[robot.trunk_control_idx]
    for arm, g in zip(M.ARMS, grip):
        a[idx[f"arm_{arm}"]] = robot.q[robot.arm_control_idx[arm]]
        a[idx[f"gripper_{arm}"]] = g
    return a


def world(oracle=True, objects=(), thresholds=NO_PAUSE, **kw):
    """``(robot, qa, contacts)``: ``contacts.pairs`` is the step's ``RigidContactAPI`` pairs."""
    robot = QARobot()
    contacts = SimpleNamespace(pairs=set())
    qa = QA.QA(robot, oracle=oracle, scene=FakeScene({o.name: o for o in objects}, robot), thresholds=thresholds,
               contact_pairs=lambda: contacts.pairs, **kw)
    return robot, qa, contacts


GRASP = SimpleNamespace(name="grasp", arms=("right",))


def test_the_metric_table_mirrors_behavior_plus_the_demo_gates_and_the_oracle_rules():
    assert set(QA.HARD_METRICS) == {"robot_touching_objects", "robot_touching_itself", "pause", "jerky_motion", "ghost_hand", "object_tilt",
                                    "goal_occlusion", "robot_wobble", "bad_release"}
    assert set(QA.SOFT_METRICS) == {"held_object_drift", "stalled", "gripper_in_view", "failed_grasp", "head_camera_upright", "arm_hand_contacts"}
    assert {k for k, v in QA.QA_METRICS.items() if v.get("oracle_only")} == {"gripper_in_view", "failed_grasp"}
    assert {k for k, v in QA.QA_METRICS.items() if v.get("retried")} == set(QA.RULE_METRICS.values()) == {"held_object_drift", "stalled", "gripper_in_view", "failed_grasp"}
    assert set(QA.RULE_METRICS) <= set(QA.CATEGORIES) and {"held_object_drift", "stalled", "gripper_out_of_fov"} <= set(QA.STEP_CATEGORIES)
    assert set(QA.HARD_METRICS) <= set(QA.CATEGORIES) and set(QA.EPILOGUE_CATEGORIES) < set(QA.CATEGORIES)
    t = QA.QAThresholds()
    assert (t.pause_steps, t.pause_vel, t.vel_avg, t.vel_prop_over_09max, t.acc_avg, t.jerk_avg, t.ghost_rad) == (50, 0.001, 1.2, 0.005, 3.0, 150.0, 0.1)
    assert (t.head_link, t.head_tilt_rad, t.head_tilt_steps, t.object_tilt_pos_m, t.object_tilt_rad, t.wobble_deg) == ("torso_link4", 0.4, 30, 0.01, 0.04, 1.0)
    c = QA.QAConfig()
    assert (c.drift_pos, c.drift_rot, c.stall_steps, c.stall_vel, c.fov_margin, c.fov_min_steps) == (0.05, 0.5, 150, 0.01, 0.05, 5)


def test_contacts_split_by_link_group_and_the_first_hard_one_stops_the_oracle():
    table = FakeObject("table_0", [1.0, 0.0, 0.4], category="breakfast_table")
    cup = FakeObject("cup_0", [0.6, 0.0, 0.8], category="cup")
    floor = FakeObject("floors_0", [0.0, 0.0, 0.0], category="floors")
    robot, qa, c = world(objects=(table, cup, floor), task_objects=["cup_0"])
    P, finger = robot.prim_path, M.FINGER_LINKS["right"][0]
    qa.begin_action(GRASP, intended={"cup_0"})
    # the acting fingers on the intended object, the wheels on the floor: nothing
    c.pairs = {(f"{P}/{finger}", "/World/cup_0/base_link"), (f"{P}/base_link", "/World/floors_0/base_link")}
    assert qa.observe(1, hold(robot)) is None
    # an arm link on the table for two steps: one soft arm / hand contact run
    c.pairs = {(f"{P}/right_arm_link1", "/World/table_0/base_link")}
    assert qa.observe(2, hold(robot)) is None and qa.observe(3, hold(robot)) is None
    # the two fingers of one gripper meeting (the closed fist of a push / brace): not the robot touching itself
    l0, l1 = M.FINGER_LINKS["left"]
    c.pairs = {(f"{P}/{l0}", f"{P}/{l1}")}
    assert qa.observe(4, hold(robot)) is None
    # an arm link on the base: robot_touching_itself stops the oracle at once ...
    c.pairs = {(f"{P}/right_arm_link1", f"{P}/base_link")}
    v = qa.observe(5, hold(robot))
    assert v.category == "robot_touching_itself" and v.step == 5 and v.action == "grasp" and QA.is_hard_qa(v)
    # ... and a later hard metric is only logged
    c.pairs = {(f"{P}/base_link", "/World/table_0/base_link")}
    assert qa.observe(6, hold(robot)) is None
    rep = qa.report()
    m = rep["metrics"]
    assert m["arm_hand_contacts"]["runs"] == [{"link": "right_arm_link1", "object": "table_0", "task_object": False, "action": "grasp",
                                               "first_step": 2, "last_step": 3, "n": 2}]
    assert [r["links"] for r in m["robot_touching_itself"]["runs"]] == [["base_link", "right_arm_link1"]] and m["robot_touching_itself"]["first"]["step"] == 5
    assert m["robot_touching_objects"]["runs"] == [{"link": "base_link", "object": "table_0", "action": "grasp", "first_step": 6, "last_step": 6, "n": 1}]
    assert rep["hard_triggered"] == ["robot_touching_objects", "robot_touching_itself"] and rep["soft_triggered"] == ["arm_hand_contacts"]
    assert rep["stopped_by"] == "robot_touching_itself" and rep["mode"] == "oracle"


def test_a_stream_logs_every_metric_and_never_stops():
    cup = FakeObject("cup_0", [0.6, 0.0, 0.8], category="cup")
    robot, qa, c = world(oracle=False, objects=(cup,), task_objects=["cup_0"], camera=QA.CameraModel.from_hfov())
    P = robot.prim_path
    c.pairs = {(f"{P}/{M.FINGER_LINKS['right'][0]}", "/World/cup_0/base_link"), (f"{P}/base_link", "/World/cup_0/base_link")}
    bad = {"step": 1, "arm": "right", "object": "cup_0", "kind": "released_between_steps"}
    assert qa.observe(1, hold(robot), release_violations=[bad]) is None
    assert qa.finish(goal_occlusion=lambda: {"occluded": True, "worst": "cup_0", "max_occluded": 0.7, "threshold": 0.5}, step=1) is None
    rep = qa.report()
    assert rep["mode"] == "stream" and rep["stopped_by"] is None
    assert rep["hard_triggered"] == ["robot_touching_objects", "goal_occlusion", "bad_release"] and rep["soft_triggered"] == ["arm_hand_contacts"]
    assert rep["metrics"]["arm_hand_contacts"]["runs"][0]["task_object"] is True  # a stream has no intended contacts
    for name in ("gripper_in_view", "failed_grasp"):
        assert rep["metrics"][name] == {"measured": False, "why": "oracle only", "mode": "soft", "triggered": False}


def test_pause_fires_after_50_still_action_steps_and_skips_the_settle_window():
    robot, qa, _ = world(thresholds=QA.QAThresholds())
    a = hold(robot)
    assert all(qa.observe(i, a, acting=False) is None for i in range(1, 80))  # the settle window does not count
    verdicts = [qa.observe(i, a) for i in range(80, 131)]
    assert verdicts[:-1] == [None] * 50 and verdicts[-1].category == "pause" and verdicts[-1].details["first_step"] == 80
    qa.finish(step=130)
    assert qa.report()["metrics"]["pause"]["runs"] == [{"first_step": 80, "last_step": 130, "n": 51}]
    # a joint moving faster than 0.001 rad/s restarts the count
    robot, qa, _ = world(thresholds=QA.QAThresholds())
    for i in range(1, 101):
        if i == 50:
            robot.q[robot.arm_control_idx["right"][0]] += 0.001
        assert qa.observe(i, hold(robot)) is None
    assert qa.report()["metrics"]["pause"]["max_pause_steps"] == 50 and not qa.report()["metrics"]["pause"]["triggered"]


def test_ghost_hand_fires_when_a_gripper_opens_with_its_arm_off_the_command():
    robot, qa, _ = world()
    j = robot.controller_action_idx["arm_right"][0]

    def act(grip_right, off=0.0):
        a = hold(robot, grip=(-1.0, grip_right))
        a[j] += off
        return a

    assert qa.observe(1, act(-1.0)) is None
    assert qa.observe(2, act(-1.0, off=0.2)) is None  # off its command while the gripper stays closed: counted only
    assert qa.observe(3, act(1.0, off=0.05)) is None  # opens within 0.1 rad
    assert qa.observe(4, act(-1.0)) is None
    v = qa.observe(5, act(1.0, off=0.2))
    assert v.category == "ghost_hand" and v.details["arm"] == "right" and v.details["error_rad"] == pytest.approx(0.2, abs=1e-4)
    gh = qa.report()["metrics"]["ghost_hand"]
    assert gh["per_arm"]["right"] == {"n_steps_total": 2, "n_steps_while_ungrasping": 1} and gh["per_arm"]["left"]["n_steps_total"] == 0


@pytest.mark.parametrize("oracle", [True, False])
def test_jerky_motion_is_judged_over_the_action_steps_at_the_end(oracle):
    robot, qa, _ = world(oracle=oracle)
    j = robot.arm_control_idx["right"][0]
    for i in range(1, 21):  # slow and smooth: 0.001 rad a step
        robot.q[j] += 0.001
        assert qa.observe(i, hold(robot)) is None
    assert qa.finish(step=20) is None and qa.report()["metrics"]["jerky_motion"]["failed"] == []
    robot, qa, _ = world(oracle=oracle)
    for i in range(1, 21):  # a 0.1 rad shake every step
        robot.q[j] += 0.1 if i % 2 else -0.1
        assert qa.observe(i, hold(robot)) is None
    v = qa.finish(step=20)
    jm = qa.report()["metrics"]["jerky_motion"]
    assert jm["failed"] == ["vel_avg", "acc_avg", "jerk_avg"] and jm["values"]["vel_avg"] == pytest.approx(3.0, rel=1e-3) and jm["triggered"]
    assert (v.category == "jerky_motion") if oracle else (v is None)


def test_gripper_in_view_logs_the_out_of_view_steps_of_an_open_span_in_the_oracle():
    robot, qa, _ = world(camera=QA.CameraModel.from_hfov())
    out_q = th.tensor([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # the right eef far behind the robot: outside the frame
    in_q = th.tensor([0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0])  # in front of the robot: in frame
    r = robot.arm_control_idx["right"]
    qa.begin_action(GRASP)
    robot.q[r] = out_q
    assert qa.observe(1, hold(robot)) is None  # no span open yet: the transit to the pre-grasp is not watched
    qa.set_fov_enabled(True)  # the pre-grasp is reached
    robot.q[r] = in_q
    assert qa.observe(2, hold(robot)) is None
    robot.q[r] = out_q
    assert all(qa.observe(i, hold(robot)) is None for i in (3, 4, 5))
    assert qa.observe(6, hold(robot, grip=(-1.0, 1.0))) is None  # the gripper opens out of the frame
    qa.set_fov_enabled(False)  # released: the retreat may leave the frame
    assert qa.observe(7, hold(robot, grip=(-1.0, 1.0))) is None
    before = qa.report()["metrics"]["gripper_in_view"]
    assert qa.observe(8, hold(robot)) is None  # the gripper closes out of the frame with the span closed: not counted
    rep = qa.report()
    fov = rep["metrics"]["gripper_in_view"]
    assert fov == before and all(t["step"] != 8 for t in fov["grasp_changes_outside_view"])
    # the program over with a span still open (a demo ending with the object in hand): the settle window is outside the
    # oracle's FOV rule, its out-of-view steps and toggles do not count
    qa.begin_action(GRASP)
    qa.set_fov_enabled(True)
    qa.end_action()
    assert qa.fov_span["right"]
    assert qa.observe(9, hold(robot, grip=(-1.0, 1.0)), acting=False) is None and qa.observe(10, hold(robot), acting=False) is None
    assert qa.report()["metrics"]["gripper_in_view"] == before
    assert fov["measured"] and fov["out_of_view"]["runs"] == [{"arm": "right", "action": "grasp", "first_step": 3, "last_step": 6, "n": 4}]
    assert fov["grasp_changes_outside_view"] == [{"step": 6, "arm": "right", "command": "open", "action": "grasp"}]
    assert rep["soft_triggered"] == ["gripper_in_view"] and rep["hard_triggered"] == [] and rep["stopped_by"] is None


def test_the_oracle_rules_are_logged_soft_with_their_retry():
    robot, qa, _ = world(rule_limits={"stalled": {"steps": 150, "vel": 0.01}})
    qa.note_rule(QA.Verdict("failed_grasp", "the assisted grasp did not attach", action="grasp", action_index=1, step=7), retried=True,
                 arms=("right",), objects=("apple.n.01_1",))
    qa.note_rule(QA.Verdict("gripper_out_of_fov", "right fingertips out of the head camera frame for 5 steps", action="lift", action_index=2,
                            attempt=0, step=40, details={"arm": "right", "fov_out_steps": 5, "fov_first_entry_step": 3}), retried=True)
    qa.note_rule(QA.Verdict("stalled", "robot stationary for 150 steps inside place", action="place", action_index=3, attempt=1, step=300,
                            details={"steps": 150}), retried=False)
    qa.note_rule(QA.Verdict("collision", "left_gripper_finger_link1 in contact with bowl_0 for 2 steps", step=9))  # a rule no soft metric logs
    rep = qa.report()
    m = rep["metrics"]
    assert m["failed_grasp"]["events"] == [{"step": 7, "action": "grasp", "action_index": 1, "attempt": 0, "arm": "right", "object": "apple.n.01_1",
                                            "message": "the assisted grasp did not attach", "retried": True}]
    (fov,) = m["gripper_in_view"]["events"]
    assert (fov["arm"], fov["fov_out_steps"], fov["fov_first_entry_step"], fov["retried"]) == ("right", 5, 3, True)
    (stall,) = m["stalled"]["events"]
    assert (stall["attempt"], stall["steps"], stall["retried"]) == (1, 150, False) and m["stalled"]["limits"] == {"steps": 150, "vel": 0.01}
    assert m["held_object_drift"]["events"] == [] and all(m[k]["retried_by_oracle"] for k in QA.RULE_METRICS.values())
    assert rep["soft_triggered"] == ["stalled", "gripper_in_view", "failed_grasp"] and rep["hard_triggered"] == [] and rep["stopped_by"] is None


def test_a_streams_rule_verdicts_are_logged_as_runs():
    robot, qa, _ = world(oracle=False)
    for step in (10, 11, 12, 20):
        qa.note_rule(QA.Verdict("stalled", f"robot stationary for {step} steps inside stream", step=step, details={"steps": step}))
    qa.note_rule(QA.Verdict("held_object_drift", "cup_0 drifted 6.0 cm / 3 deg off the right hand while grasped", step=15,
                            details={"arm": "right", "object": "cup_0"}))
    m = qa.report()["metrics"]
    assert [(r["first_step"], r["last_step"], r["n"]) for r in m["stalled"]["runs"]] == [(10, 12, 3), (20, 20, 1)] and m["stalled"]["steps"] == 4
    assert m["held_object_drift"]["runs"][0]["object"] == "cup_0" and m["held_object_drift"]["triggered"]
    assert m["gripper_in_view"]["measured"] is False and m["failed_grasp"]["measured"] is False


def test_head_camera_upright_counts_the_pitched_steps_after_3_s_of_driving():
    robot, qa, _ = world()
    robot.head_quat = th.tensor([0.0, math.sin(0.25), 0.0, math.cos(0.25)])  # 0.5 rad of pitch
    for i in range(1, 131):
        robot.pos = robot.pos + th.tensor([0.2 / 30.0, 0.0, 0.0])  # driving at 0.2 m/s
        qa.observe(i, hold(robot))
    qa.finish(step=130)
    rep = qa.report()
    hc = rep["metrics"]["head_camera_upright"]
    assert hc["measured"] and hc["steps_tilted_while_navigating"] == 40 and hc["runs"] == [{"first_step": 91, "last_step": 130, "n": 40}]
    assert hc["max_abs_pitch_rad"] == pytest.approx(0.5, abs=1e-4) and rep["soft_triggered"] == ["head_camera_upright"]


def test_object_tilt_holds_an_operated_body_to_10_mm_and_0_04_rad():
    box = FakeObject("toolbox_0", [1.0, 0.0, 0.5], category="toolbox")
    robot, qa, _ = world(roots={"toolbox.n.01_1": box})
    assert qa.observe(1, hold(robot)) is None
    box.pos = box.pos + th.tensor([0.008, 0.0, 0.0])  # 8 mm: inside the band
    assert qa.observe(2, hold(robot)) is None
    box.quat = th.tensor([math.sin(0.025), 0.0, 0.0, math.cos(0.025)])  # tipped 0.05 rad
    v = qa.observe(3, hold(robot))
    assert v.category == "object_tilt" and v.details["object"] == "toolbox.n.01_1" and v.details["tilt_rad"] == pytest.approx(0.05, abs=1e-4)
    ot = qa.report()["metrics"]["object_tilt"]
    assert ot["band"] == {"pos_m": 0.01, "tilt_rad": 0.04} and ot["objects"]["toolbox.n.01_1"]["exceeded"]
    robot, qa, _ = world()
    assert qa.report()["metrics"]["object_tilt"]["measured"] is False  # nothing is opened or closed on a free-standing object


def test_robot_wobble_when_a_load_bends_the_torso_off_its_command():
    robot, qa, _ = world()
    a = hold(robot)
    assert qa.observe(1, a) is None
    robot.q[robot.trunk_control_idx[1]] += math.radians(0.8)
    assert qa.observe(2, a) is None
    robot.q[robot.trunk_control_idx[1]] += math.radians(0.7)
    v = qa.observe(3, a)
    assert v.category == "robot_wobble" and v.details["deg"] == pytest.approx(1.5, abs=1e-3)
    rw = qa.report()["metrics"]["robot_wobble"]
    assert rw["max_deg"] == pytest.approx(1.5, abs=1e-3) and rw["over_limit"]["runs"][0]["first_step"] == 3


def test_bad_release_and_goal_occlusion_are_hard():
    robot, qa, _ = world()
    bad = {"step": 4, "arm": "right", "object": "cup_0", "kind": "released_between_steps"}
    v = qa.observe(4, hold(robot), release_violations=[bad])
    assert v.category == "bad_release" and v.details["object"] == "cup_0"
    assert qa.report()["metrics"]["bad_release"]["violations"] == [{**bad, "action": None}]
    robot, qa, _ = world()
    for i in range(1, 6):
        qa.observe(i, hold(robot))
    go = {"occluded": True, "worst": "target_apple_0", "max_occluded": 0.62, "threshold": 0.5}
    v = qa.finish(goal_occlusion=lambda: dict(go), step=5)
    assert v.category == "goal_occlusion" and "62%" in v.message and qa.report()["metrics"]["goal_occlusion"]["report"] == go
    robot, qa, _ = world()

    def broken():
        raise RuntimeError("no depth")

    assert qa.finish(goal_occlusion=broken, step=0) is None and "no depth" in qa.report()["metrics"]["goal_occlusion"]["report"]["error"]


def test_restore_forgets_the_rolled_back_steps_and_a_step_that_is_not_live_never_stops():
    table = FakeObject("table_0", [1.0, 0.0, 0.4], category="breakfast_table")
    robot, qa, c = world(objects=(table,))
    P = robot.prim_path
    qa.begin_action(GRASP)
    assert all(qa.observe(i, hold(robot)) is None for i in (1, 2, 3))
    snap = qa.snapshot()
    c.pairs = {(f"{P}/right_arm_link1", "/World/table_0/base_link")}
    assert all(qa.observe(i, hold(robot)) is None for i in (4, 5))
    qa.restore(snap, 3)
    c.pairs = set()
    assert qa.observe(4, hold(robot)) is None
    rep = qa.report()
    assert rep["metrics"]["arm_hand_contacts"]["runs"] == [] and rep["soft_triggered"] == [] and "first" not in rep["metrics"]["arm_hand_contacts"]
    # the hold step after a restore is not live: its hard metric is logged, the next live one stops the oracle
    c.pairs = {(f"{P}/base_link", "/World/table_0/base_link")}
    assert qa.observe(5, hold(robot), live=False) is None
    v = qa.observe(6, hold(robot))
    assert v.category == "robot_touching_objects" and v.step == 6 and qa.report()["metrics"]["robot_touching_objects"]["first"]["step"] == 5


@pytest.mark.parametrize("oracle", [True, False])
def test_report_has_every_metric_with_its_mode(oracle):
    robot, qa, _ = world(oracle=oracle)
    qa.observe(1, hold(robot))
    qa.finish(step=1)
    rep = qa.report()
    assert set(rep["metrics"]) == set(QA.QA_METRICS) and all(rep["metrics"][k]["mode"] == QA.QA_METRICS[k]["mode"] for k in QA.QA_METRICS)
    assert rep["enabled"] and rep["mode"] == ("oracle" if oracle else "stream") and rep["steps_observed"] == 1
    assert rep["hard_triggered"] == rep["soft_triggered"] == [] and rep["stopped_by"] is None


def test_pause_skips_the_oracles_hold_and_settle_waits():
    robot, qa, _ = world(thresholds=QA.QAThresholds())
    a = hold(robot)
    for i in range(1, 121):  # 120 still steps, every 30th a hold / settle wait of the motion layer
        assert qa.observe(i, a, settling=i % 30 == 0) is None
    pause = qa.report()["metrics"]["pause"]
    assert pause["max_pause_steps"] == 29 and not pause["triggered"]


def test_motions_mark_their_hold_and_settle_waits():
    sc, env = iors_world()
    _ex, motions, _backend, _rec = make_executor(env, sc)
    next(motions.hold(2))
    assert motions.settling
    motions._emit()
    assert not motions.settling
    next(motions.settle(max_steps=3))
    assert motions.settling


def test_the_oracle_retries_a_stall_and_qa_logs_every_attempt(tmp_path, monkeypatch):
    """The stalled rule keeps its oracle behaviour (owner 2026-10-14): the action is retried and the demo fails once the retries
    are used up; QA logs every attempt as a soft metric. (The fake base drives without moving the base joints: its transition
    stands still.)"""
    quiet_qa(monkeypatch)
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, *_ = make_executor(env, sc, tmp_path, qa=True, qa_cfg=QA.QAConfig(stall_steps=20, fov_min_steps=100000),
                           exec_cfg=ExecutorConfig(settle_steps=50, retries=1))
    result = ex.run_program(expand(sc.variant, sc))
    assert not result["success"] and result["reason"] == "stalled"
    stall = result["qa"]["metrics"]["stalled"]
    assert [(e["action_index"], e["attempt"], e["retried"]) for e in stall["events"]] == [(0, 0, True), (0, 1, False)]
    assert stall["mode"] == "soft" and stall["limits"] == {"steps": 20, "vel": 0.01}
    assert result["qa"]["hard_triggered"] == [] and result["qa"]["stopped_by"] is None


def test_a_stream_logs_stalled_and_pause_but_not_over_a_replayed_settle_window():
    def run(phases):
        env, source, cfg = make_world("never_reached")
        mon = QA.Monitors(env, env.robot, QA.QAConfig(stall_steps=10, fov_min_steps=100000))
        mon._contact_api = FakeContacts(lambda s, q: [])
        ex = Executor(env, env.task, None, recorder=None, motions=None, monitors=mon, cfg=cfg, sim=env.sim, state=_State(), tracked_objects={}, cam_key=CAM_KEY)
        ex.last_obs, ex.last_info = env._obs(), env._info()
        ex.stream_phases = phases

        def stream():
            while True:
                a = source(ex.last_obs, ex.last_info)
                if a is None:
                    return
                yield a

        return ex.run_stream(stream())["qa"]["metrics"]

    m = run(None)  # 120 steps of driving with the joints still
    assert [(r["first_step"], r["last_step"]) for r in m["stalled"]["runs"]] == [(10, 120)] and m["pause"]["triggered"]
    m = run([PHASE_ORACLE] * 40 + [PHASE_SETTLE] * 80)  # the replayed demo's last 80 rows are its settle window
    assert [(r["first_step"], r["last_step"]) for r in m["stalled"]["runs"]] == [(10, 40)]
    assert m["pause"]["max_pause_steps"] == 39 and not m["pause"]["triggered"]


def test_a_held_object_drift_in_the_settle_window_fails_the_demo_and_is_logged(tmp_path, monkeypatch):
    """The settle window has no retry: a held-object-drift verdict there fails the demo, and QA logs it as not retried."""
    quiet_qa(monkeypatch)
    sc, env = iors_world()
    ex, motions, *_ = make_executor(env, sc, tmp_path, qa=True)
    ex._init_qa(oracle=True, actions=expand(sc.variant, sc))
    robot, apple = env.robot, env.objects["apple.n.01_1"]
    robot._ag_obj_in_hand["right"] = apple
    robot._ag_obj_constraints = {"left": None, "right": True}
    motions.state.gripper_cmd["right"] = -1.0  # closed: the fake's assisted grasp keeps the apple

    def slip(action):
        if env.steps == 5:
            apple.pos = apple.pos + th.tensor([0.2, 0.0, 0.0])

    env.after_step = slip
    result = ex.epilogue(None, 100)
    assert not result["success"] and result["reason"] == "held_object_drift"
    (ev,) = result["qa"]["metrics"]["held_object_drift"]["events"]
    assert (ev["action"], ev["retried"], ev["object"], ev["step"]) == ("epilogue", False, apple.name, 5)
    assert "held_object_drift" in result["qa"]["soft_triggered"]
