"""i7-contact 2026-11-07, step 2: the PUSH / BRACE contact libraries (``grasping/contact_library.py``: entries from the shared record builders in
the object frame, per object and stable rest) and their physics validation (``grasping/validate_contacts.py``: the entry as the ORACLE's action,
run through ``execute.run_action``, judged by the push / brace acceptance). The oracle runs on the fake world of ``test_v2_oracle``; the Isaac
Sim world (``KitContactWorld``) is not started. No simulator."""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_library as CL
from ispatialgym.grasping import schema as S
from ispatialgym.grasping import validate_contacts as VC
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import contact_poses as CPZ

TOUCHING = "inter_object_relation_satisfaction-touching_small_scale"
#: box_of_cereal/iucshm at x0.8 (the touching_02 target), its root at the bottom centre
CEREAL = (0.0427, 0.1567, 0.2182)


def _box_geometry(dx, dy, dz):
    """A box's geometry.json piece (root at the bottom centre) and its 8 corners."""
    v = [[sx * dx / 2, sy * dy / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, dz)]
    # 12 triangles of the box (vertex index = 4 sx + 2 sy + z)
    f = [[0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3], [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6]]
    return {"name": "base_link/box", "vertices": v, "faces": f}


def _model_dir(root, eid, dims, rest_q=(0.0, 0.0, 0.0, 1.0), h_root=0.0, stable=True, grasps=()):
    d = os.path.join(root, "models", eid)
    os.makedirs(d, exist_ok=True)
    cat, model, scale = eid.split("__")[0], eid.split("__")[1], [0.8, 0.8, 0.8]
    rest = {"schema": S.REST, "rest_id": "r0", "quat_xyzw": list(rest_q), "h_root": h_root, "stable": stable}
    S.save_json(os.path.join(d, "library.json"), {"schema": S.LIBRARY, "category": cat, "model": model, "scale": scale, "rests": {"r0": rest}, "grasps": list(grasps)})
    S.save_json(os.path.join(d, "geometry.json"), {"format": "isg-object-geometry/1", "category": cat, "model": model, "pieces": [_box_geometry(*dims)]})
    return d


@pytest.fixture
def glib(tmp_path):
    root = str(tmp_path / "glib")
    _model_dir(root, "box_of_cereal__iucshm__s0.8", CEREAL)
    _model_dir(root, "box_of_cereal__dkhgxn__s1", (0.077, 0.155, 0.242))
    _model_dir(root, "magazine__flat__s1", (0.20, 0.28, 0.012))
    _model_dir(root, "no_rest__x__s1", (0.1, 0.1, 0.1), stable=False)
    return root


def _job(eid, families, **kw):
    cat, model = eid.split("__")[:2]
    return {"id": eid, "category": cat, "model": model, "entry": eid, "families": families, "subtask": TOUCHING, **kw}


# ---------------------------------------------------------------------------------------------------------------- the builder


def test_a_touching_push_library_is_the_samplers_records_in_the_object_frame(glib):
    libs = CL.build_job(_job("box_of_cereal__iucshm__s0.8", ["touching_push"], generation={"push_height_sweep": True}), glib)
    lib = libs["push"]
    assert set(libs) == {"push"} and lib["schema"] == CL.LIB_SCHEMA and lib["object"]["rest_id"] == "r0" and lib["object"]["n_hull_points"] == 8
    fam = lib["families"]["touching_push"]
    assert fam["knobs"]["push_height_sweep"] is True and fam["knobs"]["push_max_height_fraction"] == 0.5 and fam["yaws_deg"][:2] == [0.0, 45.0]
    e0 = [e for e in lib["entries"] if e["yaw_deg"] == 0.0]
    assert e0 and all(e["kind"] == "push" and e["arm_role"] == "work" and e["family"] == "touching_push" for e in e0)
    # the same records as the sampler's builder on the same box at the same yaw (the rule frame: push along +y)
    from ispatialgym.tests import contact_poses_scenarios as CS
    from ispatialgym.tests import test_iors_small_lift as LIFT

    s = CS._push_sampler({"push_height_sweep": True}, LIFT._box(*CEREAL))
    s.hull["target"] = LIFT._box(*CEREAL)
    pose = (th.tensor([0.0, 0.0, 0.0]), C.IDENTITY_QUAT.clone())
    hull_w = LIFT._box(*CEREAL)
    face_z = (0.0, CEREAL[2])
    want = list(s.touching_push_candidates(LIFT._stance(), CPZ.push_face_point(hull_w, CS.D), face_z, CS.D, t_hull_w=hull_w))
    assert [e["class"] for e in e0] == [CL._label(k, x) for k, _p, x in want]
    for e, (k, p, x) in zip(e0, want):
        w = CPZ.entry_world(e, pose)
        assert th.allclose(w["contact"][0], p[0], atol=1e-5) and C.quat_angle(w["contact"][1], p[1]) < 1e-4
    tips = [e for e in e0 if e["class"].startswith("push_tip_h-@")]
    assert tips and all(e["pre_pose_rule"]["rule"] == "lowered_to_push_pre_dist_m" and e["pre"]["executed_m"] == pytest.approx(0.04) for e in tips)
    assert all(e["push"]["stroke_m"] == CL.TOUCH_PUSH_STROKE_M for e in e0) and lib["digest"] == CL._library_digest(lib)


def test_a_touching_brace_library_and_the_wide_stop(glib):
    libs = CL.build_job(_job("box_of_cereal__dkhgxn__s1", ["touching_brace"], generation={"brace_open_when_wide": True}), glib)
    lib = libs["brace"]
    e0 = [e for e in lib["entries"] if e["yaw_deg"] == 0.0]
    assert e0 and e0[0]["class"].startswith("brace_wide_h") and e0[0]["fingers"]["state"] == "open"  # the 15.5 cm face takes the open stop
    assert all(e["kind"] == "brace" and e["stop"] == CPZ.brace_stop_rule() and e["load"]["direction"] == [0.0, 1.0, 0.0] for e in lib["entries"])
    assert lib["families"]["touching_brace"]["knobs"]["brace_open_when_wide"] is True
    narrow = CL.build_job(_job("box_of_cereal__dkhgxn__s1", ["touching_brace"]), glib)["brace"]
    assert not any(e["class"].startswith("brace_wide_h-_lo") for e in narrow["entries"]) and narrow["digest"] != lib["digest"]


def test_an_axial_push_library_on_the_shared_axial_rules(glib):
    lib = CL.build_job(_job("magazine__flat__s1", ["axial_push"], params={"axial_push": {"yaws": 2, "turn_deg": 40.0}}), glib)["push"]
    e = lib["entries"]
    assert {x["yaw_deg"] for x in e} == {0.0, 180.0} and all(x["push"]["arc"] and x["pre_pose_rule"]["rule"] == "recorded" for x in e)
    assert {0.0, 30.0, 45.0} <= {x["pitch_deg"] for x in e} and any(x["class"].endswith("_side") for x in e)
    x = e[0]
    assert x["pre"]["executed_m"] == pytest.approx(C.PRE_DIST, abs=1e-5)  # an arc keeps the recorded pre (push_lowered_pre)
    # the eef at the axial push height: 0.0218 m above the support for a 1.2 cm target
    w = CPZ.entry_world(x, CL.object_frame([0, 0, 0, 1.0], 0.0, 0.0))
    assert float(w["contact"][0][2]) == pytest.approx(CPZ.axial_push_height(0.0, 0.012), abs=1e-5) == pytest.approx(0.0218, abs=1e-6)
    assert x["arc"]["turn_rad"] == pytest.approx(math.radians(40.0) * (1 if "+spin" in x["class"] else -1))


def test_missing_objects_and_families_say_what_is_missing(glib, tmp_path):
    with pytest.raises(LookupError, match="stable"):
        CL.build_job(_job("no_rest__x__s1", ["touching_push"]), glib)
    with pytest.raises(LookupError, match="v2 model directory"):
        CL.build_job(_job("teapot__mvrhya__s1", ["touching_brace"]), glib)
    with pytest.raises(NotImplementedError, match="block_design"):
        CL.build_job(_job("box_of_cereal__iucshm__s0.8", ["block_push"]), glib)
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([_job("box_of_cereal__iucshm__s0.8", ["touching_push"]), _job("teapot__mvrhya__s1", ["touching_brace"]),
                                _job("box_of_cereal__dkhgxn__s1", ["touching_brace"])]))
    root = str(tmp_path / "contacts")
    CL.build_cli(SimpleNamespace(jobs=str(jobs), glib=glib, root=root, only=None, shard=None))
    rows = {(r["kind"], r["id"]) for r in CL.summary(root)}
    assert rows == {("push", "box_of_cereal__iucshm__s0.8"), ("brace", "box_of_cereal__dkhgxn__s1")}
    log = [json.loads(line) for line in open(os.path.join(root, "logs", "build_0of1.jsonl"))]
    assert [r["ok"] for r in log] == [True, False, True] and "v2 model directory" in log[1]["error"]


# ---------------------------------------------------------------------------------------------------------------- validation: the oracle's action


def _tip_entry(glib):
    lib = CL.build_job(_job("box_of_cereal__iucshm__s0.8", ["touching_push"], generation={"push_height_sweep": True}), glib)["push"]
    return next(e for e in lib["entries"] if e["yaw_deg"] == 0.0 and e["class"].startswith("push_tip_h-@"))


def test_an_entry_becomes_the_oracles_push_and_the_oracle_lowers_its_pre(glib):
    from ispatialgym.oracle import execute as EX
    from ispatialgym.oracle.actions import Brace, Push

    e = _tip_entry(glib)
    T = CL.object_frame([0, 0, 0, 1.0], 0.0, 0.0, support_z=0.81)
    T[:2, 3] = [0.7, -0.1]
    act = VC.entry_action(e, T, "box.n.01_1", "right")
    assert isinstance(act, Push) and act.contact.mid is None and act.contact.distance_m == pytest.approx(0.08)
    init = np.array(act.contact.init.position)
    pre = np.array(act.contact.pre.position)
    assert np.linalg.norm(pre - init) == pytest.approx(0.05, abs=1e-5)  # the RECORDED pre: the oracle lowers it itself
    low = EX.push_lowered_pre(act.contact, act.contact.pre, act.contact.init, EX.ExecuteConfig())
    assert np.linalg.norm(np.array(low.position) - init) == pytest.approx(0.04, abs=1e-5)
    w = CPZ.entry_world(e, T)
    assert np.allclose(low.position, w["pre_executed"][0].numpy(), atol=1e-5)  # = the entry's executed pre
    assert np.allclose(np.array(act.contact.goal.position) - init, [0.0, 0.08, 0.0], atol=1e-5) and act.contact.extra["pose_class"] == "push_tip_h-"
    b = CL.build_job(_job("box_of_cereal__dkhgxn__s1", ["touching_brace"]), glib)["brace"]["entries"][0]
    bact = VC.entry_action(b, T, "partner.n.01_1", "left")
    assert isinstance(bact, Brace) and bact.contact.kind == b["class"] and np.allclose(bact.contact.direction, [0.0, -1.0, 0.0], atol=1e-5)


def _fake_push_world(e, T_obj, react):
    from ispatialgym.oracle.executor import SimState
    from ispatialgym.oracle.motions import Motions, MotionsConfig
    from ispatialgym.tests.test_v2_oracle import FakeBackend, FakeEnv, FakeObject, FakeRobot

    robot = FakeRobot(base=(0.0, 0.0, 0.0))
    from ispatialgym.grasping import schema as SS

    box = FakeObject("box.n.01_1", T_obj[:3, 3].tolist(), quat=SS.mat_to_quat(T_obj[:3, :3]).tolist(), category="box_of_cereal", extent=CEREAL)
    env = FakeEnv(robot, {"box.n.01_1": box})
    env.task._tro = None
    env.after_step = lambda _a: react(robot, box)
    motions = Motions(env, FakeBackend(robot), MotionsConfig(settle_steps=5, gripper_steps=5, cartesian_step=0.01, cartesian_rot_step=0.3))
    return env, robot, box, motions, SimState(env, env.task, robot)


def _carry(robot, box, face_y, slip=0.0, tip=0.0):
    """The fake physics: the box's -y face is pushed by the fingertips (it moves with them once they reach it), ``slip`` m sideways per
    contact step, ``tip`` rad of tilt about x per contact step."""
    p, q = robot.eef_pose("right")
    y_tip = float(p[1]) + C.FINGERTIP_LENGTH * float(C.approach_axis(q)[1])
    face = float(box.pos[1]) - CEREAL[1] / 2
    if y_tip > face:
        box.pos = box.pos + th.tensor([slip, y_tip - face, 0.0])
        if tip:
            box.quat = C.qmul(C.quat_from_axis_angle(th.tensor([1.0, 0.0, 0.0]), tip), th.as_tensor(box.quat).float()) if hasattr(C, "quat_from_axis_angle") else box.quat


def test_the_oracle_runs_a_push_entry_and_the_acceptance_judges_it(glib):
    e = dict(_tip_entry(glib), goal_target=None)  # the stroke rules (a push without a task goal)
    e["push"] = {k: v for k, v in e["push"].items() if k != "stop_gap_m"}
    T = CL.object_frame([0, 0, 0, 1.0], 0.0, 0.0, support_z=0.81)
    T[:2, 3] = [0.7, -0.1]
    pts = CL.hull_points(SimpleNamespace(union=SimpleNamespace(vertices=np.array(_box_geometry(*CEREAL)["vertices"]))))
    cfg = VC.ContactValidateConfig()
    for slip, want in ((0.0, "valid"), (0.004, "invalid")):
        env, robot, box, motions, state = _fake_push_world(e, T, lambda r, b, s=slip: _carry(r, b, 0.0, slip=s))
        act = VC.entry_action(e, T, "box.n.01_1", "right")
        track = VC.PeakTracker(e, T, pts)

        def step(a, env=env, robot=robot, box=box, track=track):
            env.step(a)
            track(VC._T((box.pos, th.as_tensor(box.quat).float())), VC._T(robot.eef_pose("right")))

        run = VC.run_oracle(act, motions, state, step, cfg.max_steps)
        assert run["completed"] and run["ctx"].get("push_pre_lowered", {}).get("dist_m") == 0.04  # the oracle lowered the recorded pre itself
        T1 = VC._T((box.pos, th.as_tensor(box.quat).float()))
        verdict = VC.judge_push(e, T, T1, VC._T(robot.eef_pose("right")), pts, cfg, dict(run, postcondition=None), tip_gap_m=track.gap)
        assert verdict["verdict"] == want, verdict
        if want == "valid":
            assert verdict["measures"]["moved_along_m"] >= verdict["measures"]["needed_m"] and verdict["measures"]["tip_gap_m"] <= 0.001
        else:
            assert "slip" in verdict["reasons"]


def test_a_touching_push_entry_carries_its_partner_and_goal_and_the_axial_its_turns(glib):
    """Run D (1): a touching push's target moves the GAP (the push distance less the task's 0.035 margin) and ends touching its partner -
    the entry carries the partner's place (``push.stop_gap_m``) and the task goal (``goal_target``: translate, tolerances 0.03 / 0.2); the
    axial push entries turn through the task's goal deviations 25 / 45 / 65 deg (``AXIAL_PUSH_TURNS_DEG``), each its own class, goal = turn."""
    e = _tip_entry(glib)
    assert e["push"]["stop_gap_m"] == pytest.approx(e["push"]["stroke_m"] - CL.TOUCH_PUSH_MARGIN_M) == pytest.approx(0.045)
    assert e["goal_target"] == {"kind": "translate", "along_push_m": e["push"]["stop_gap_m"], "position_tolerance": 0.03, "orientation_tolerance": 0.2,
                                "source": e["goal_target"]["source"]}
    lib = CL.build_job(_job("magazine__flat__s1", ["axial_push"], params={"axial_push": {"yaws": 1}}), glib)["push"]["entries"]
    by_turn = {}
    for x in lib:
        t = x["class"].rsplit("@t", 1)[1]
        by_turn.setdefault(t, []).append(x)
        assert x["goal_target"]["kind"] == "turn" and x["goal_target"]["turn_rad"] == x["arc"]["turn_rad"] and x["goal_target"]["orientation_tolerance"] == 0.1
        assert abs(x["arc"]["turn_rad"]) == pytest.approx(math.radians(float(t)), abs=1e-5)
    assert sorted(by_turn) == ["25", "45", "65"] and len({len(v) for v in by_turn.values()}) == 1


def test_a_push_with_a_task_goal_is_judged_by_the_tasks_goal_check():
    """Run D (1): calibrated on the replays - a push with a task goal passes when the TASK's check passes: the goal pose within its tolerances
    (touching 0.03 m / 0.2 rad full rotation: a topple / slip fails it; axial orientation 0.1, no position), a touching target touching its
    partner (the stop's face <= 3 mm + 1 mm slack: the touching predicate's geometric branch), the oracle clean. The stroke measures (a touching
    target moves the gap, not the stroke) stay diagnostics."""
    from ispatialgym.sampling import predicates as P

    cfg = VC.ContactValidateConfig()
    assert cfg.touching_gap_max_m == P.TOUCHING_GAP_MAX_M == 0.003
    pts = np.array(_box_geometry(*CEREAL)["vertices"])
    e = {"push": {"direction": [0.0, 1.0, 0.0], "stroke_m": 0.08, "stop_gap_m": 0.045}, "extras": {"stop_clearance_m": 0.005},
         "goal_target": {"kind": "translate", "along_push_m": 0.045, "position_tolerance": 0.03, "orientation_tolerance": 0.2}}
    T0 = np.eye(4)
    T0[:3, 3] = [0.7, -0.1, 0.81]

    def after(dy, dx=0.0, tilt=0.0, yaw=0.0):
        T = T0.copy()
        ct, st, cy, sy = math.cos(tilt), math.sin(tilt), math.cos(yaw), math.sin(yaw)
        T[:3, :3] = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]]) @ np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        T[:3, 3] = T0[:3, 3] + [dx, dy, 0.0]
        return T

    stop = VC.stop_pose(e, T0, pts, cfg)
    lead = T0[1, 3] + CEREAL[1] / 2
    assert stop["face_along_m"] == pytest.approx(lead + 0.045 + cfg.stop_slack_m)
    assert stop["centre"][1] == pytest.approx(stop["face_along_m"] + cfg.stop_size_m[0] / 2) and stop["centre"][0] == pytest.approx(0.7)
    assert stop["centre"][2] - cfg.stop_size_m[2] / 2 == pytest.approx(0.81 + 0.001)  # its bottom just above the support
    assert VC.stop_pose(dict(e, push={"direction": [0.0, 1.0, 0.0]}), T0, pts, cfg) is None
    run = {"completed": True, "postcondition": None}
    eef = np.eye(4)

    def judge(T1, **kw):
        return VC.judge_push(e, T0, T1, eef, pts, cfg, run, touch_gap_m=VC.touch_gap(stop, T1, pts), **kw)

    ok = judge(after(0.045))
    assert ok["verdict"] == "valid" and ok["measures"]["touch_gap_m"] == pytest.approx(0.001) and ok["measures"]["goal_position_err_m"] == pytest.approx(0.0)
    assert "short" in ok["measures"]["diagnostic_reasons"]  # the stroke rule would have called it short: the target moves the gap
    assert judge(after(0.035))["reasons"] == ["not_touching"]  # 1 cm short of the partner (within the 0.03 position tolerance)
    assert judge(after(0.08))["reasons"] == ["goal_position"]  # run D's world: no partner, the target ran the push margin on
    assert "goal_orientation" in judge(after(0.045, tilt=0.3))["reasons"]  # a topple
    assert judge(after(0.045, tilt=0.15))["verdict"] == "valid"  # within the task's 0.2 rad
    assert judge(after(0.045, yaw=0.25))["reasons"] == ["goal_orientation"]
    assert VC.judge_push(e, T0, after(0.045), eef, pts, cfg, run)["reasons"] == ["not_touching"]  # no stop measured: not shown touching
    assert judge(after(0.045), other_slab_N=3.0)["reasons"] == ["non_finger_contact"]
    # an axial push: orientation only (no position, no drift), the task's 0.1 rad
    a = {"push": {"direction": [0.0, 1.0, 0.0], "stroke_m": 0.1, "arc": True}, "arc": {"turn_rad": 0.9}, "extras": {},
         "goal_target": {"kind": "turn", "turn_rad": 0.95, "orientation_tolerance": 0.1}}
    g = VC.goal_pose(a, T0)
    assert g["position_tolerance"] is None and VC._quat_angle(g["T"][:3, :3], after(0.0, yaw=0.95)[:3, :3]) == pytest.approx(0.0, abs=1e-9)
    assert VC.judge_push(a, T0, after(0.0, dx=0.08, yaw=0.88), eef, pts, cfg, run)["verdict"] == "valid"  # 0.07 rad short, 8 cm drift: the task's check
    assert VC.judge_push(a, T0, after(0.0, yaw=0.80), eef, pts, cfg, run)["reasons"] == ["goal_orientation"]
    assert VC.judge_push(a, T0, after(0.0, yaw=-0.95), eef, pts, cfg, run)["reasons"] == ["goal_orientation"]  # the wrong sense
    pt = VC.goal_pose_target("box.n.01_1", g)
    assert pt.position is None and pt.orientation_tolerance == 0.1 and pt.orientation_metric == "full"
    pt = VC.goal_pose_target("box.n.01_1", VC.goal_pose(e, T0))
    assert np.allclose(pt.position, [0.7, -0.055, 0.81]) and pt.position_tolerance == 0.03 and pt.orientation_tolerance == 0.2


def test_the_oracle_stops_a_touching_push_at_its_goal_in_the_fake_world(glib):
    """Run D (2): touching_02's 'topple' was the WORLD - no partner and no goal: the oracle ran the push margin on (0.084 m) and the tall narrow
    target (4.3 cm deep, 21.8 cm tall) tipped. With the task's goal as the object's pose target the oracle's own closed loop stops the push
    when the target reaches it (``push_stopped``), the target touches the partner's place; without it the push runs the whole stroke (in
    Isaac Sim the partner stand-in stops the target)."""
    e = _tip_entry(glib)
    T = CL.object_frame([0, 0, 0, 1.0], 0.0, 0.0, support_z=0.81)
    T[:2, 3] = [0.7, -0.1]
    pts = CL.hull_points(SimpleNamespace(union=SimpleNamespace(vertices=np.array(_box_geometry(*CEREAL)["vertices"]))))
    cfg = VC.ContactValidateConfig()
    goal = VC.goal_pose(e, T)
    stop = VC.stop_pose(e, T, pts, cfg)
    out = {}
    for with_goal in (True, False):
        env, robot, box, motions, state = _fake_push_world(e, T, lambda r, b: _carry(r, b, 0.0))
        if with_goal:
            env.task._tro = SimpleNamespace(meta=SimpleNamespace(pose_targets={"box.n.01_1": VC.goal_pose_target("box.n.01_1", goal)}))
        act = VC.entry_action(e, T, "box.n.01_1", "right")
        run = VC.run_oracle(act, motions, state, env.step, cfg.max_steps)
        T1 = VC._T((box.pos, th.as_tensor(box.quat).float()))
        out[with_goal] = (run, VC.judge_push(e, T, T1, VC._T(robot.eef_pose("right")), pts, cfg, dict(run, postcondition=None), goal=goal,
                                             touch_gap_m=VC.touch_gap(stop, T1, pts)))
    run, v = out[True]
    assert run["ctx"].get("push_stopped") and v["verdict"] == "valid", v
    assert v["measures"]["goal_position_err_m"] < 0.01 and -0.01 < v["measures"]["touch_gap_m"] <= cfg.touching_gap_max_m + cfg.stop_slack_m
    run, v = out[False]  # the fake world has no partner physics: the target runs the stroke on, 3 cm through the partner's place
    assert not run["ctx"].get("push_stopped") and v["measures"]["moved_along_m"] > 0.07 and v["measures"]["touch_gap_m"] < -0.025


def test_the_replays_take_the_demos_goal_partner_and_variant():
    """Run D (1)-(2): a replay's goal is the demo's - a touching push stops at the partner's place (the record's push distance less the 0.035
    margin: touching_02 0.0493, touching_01 0.0217), the TRO's goal from the replay start kept beside it; an axial push's turn is the TRO's
    orientation target (axial_01 0.953 rad, axial_03 1.026: the record's arc 0.967 / 1.034); the variant reaches the oracle's ctx (the
    board game's push_arc_stall_ends_arc, which run D's axial_01 replay ran without)."""
    from ispatialgym.oracle import execute as EX

    want = {"touching_02": (0.0493, 0.0445), "touching_01": (0.0217, 0.0235)}
    for inst, (gap, tro) in want.items():
        sit = _replay(inst, "push")
        assert sit["pose_target"]["position_tolerance"] == 0.03 and "touching" in sit["variant"]
        e = VC.record_entry(sit)
        assert e["push"]["stop_gap_m"] == pytest.approx(gap, abs=2e-4) and e["goal_target"]["along_push_m"] == e["push"]["stop_gap_m"]
        assert e["goal_target"]["tro_along_push_m"] == pytest.approx(tro, abs=5e-4) and e["goal_target"]["orientation_tolerance"] == 0.2
    for inst, turn, arc in (("axial_01", 0.9527, 0.9666), ("axial_03", 1.026, 1.0344)):
        e = VC.record_entry(_replay(inst, "push"))
        g = e["goal_target"]
        assert g["kind"] == "turn" and g["turn_rad"] == pytest.approx(turn, abs=1e-3) and g["arc_turn_rad"] == pytest.approx(arc, abs=1e-3)
        assert g["orientation_tolerance"] == 0.1 and "stop_gap_m" not in e["push"]
    assert VC.record_goal(_replay("touching_02", "brace"), {}) is None
    sit = _replay("axial_01", "push")
    assert EX.variant_generation(VC.oracle_ctx(sit["variant"]))["push_arc_stall_ends_arc"] is True and VC.oracle_ctx(None) == {}


def test_the_push_and_brace_acceptance_rules():
    cfg = VC.ContactValidateConfig()
    e = {"push": {"direction": [0.0, 1.0, 0.0], "stroke_m": 0.08}, "extras": {"stop_clearance_m": 0.005}}
    pts = np.array(_box_geometry(*CEREAL)["vertices"])
    T0 = np.eye(4)

    def after(dy, dx=0.0, yaw=0.0, tilt=0.0):
        T = np.eye(4)
        cy, sy, ct, st = math.cos(yaw), math.sin(yaw), math.cos(tilt), math.sin(tilt)
        T[:3, :3] = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]]) @ np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        T[:3, 3] = [dx, dy, 0.0]
        return T

    eef = np.eye(4)
    eef[:3, :3] = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]]).T @ np.eye(3)  # any orientation; place the tips at the rear face below
    eef[:3, :3] = C.quat_to_mat(C.eef_quat(th.tensor([0.0, 1.0, 0.0]), th.tensor([1.0, 0.0, 0.0]))).numpy()

    def tips_at(T):
        E = eef.copy()
        E[:3, 3] = T[:3, 3] + np.array([0.0, -CEREAL[1] / 2 - C.FINGERTIP_LENGTH, 0.05])
        return E

    run = {"completed": True, "postcondition": None}
    ok = VC.judge_push(e, T0, after(0.075), tips_at(after(0.075)), pts, cfg, run)
    assert ok["verdict"] == "valid" and ok["measures"]["needed_m"] == pytest.approx(0.065)
    assert VC.judge_push(e, T0, after(0.05), tips_at(after(0.05)), pts, cfg, run)["reasons"] == ["short"]
    assert "topple" in VC.judge_push(e, T0, after(0.075, tilt=0.3), tips_at(after(0.075)), pts, cfg, run)["reasons"]
    assert "slip" in VC.judge_push(e, T0, after(0.075, yaw=0.3), tips_at(after(0.075)), pts, cfg, run)["reasons"]
    far = tips_at(after(0.075))
    far[1, 3] -= 0.05
    assert "contact_lost" in VC.judge_push(e, T0, after(0.075), far, pts, cfg, run)["reasons"]
    assert VC.judge_push(e, T0, after(0.075), tips_at(after(0.075)), pts, cfg, run, other_slab_N=3.0)["reasons"] == ["non_finger_contact"]
    assert VC.judge_push(e, T0, after(0.075), tips_at(after(0.075)), pts, cfg, {"completed": False, "postcondition": "incomplete"})["verdict"] == "invalid"
    load = np.array([0.0, 1.0, 0.0])
    held = VC.judge_brace({}, T0, after(0.006), np.eye(4), np.eye(4), load, cfg, run)
    assert held["verdict"] == "valid" and held["measures"]["moved_along_load_m"] == pytest.approx(0.006)
    assert VC.judge_brace({}, T0, after(0.03), np.eye(4), np.eye(4), load, cfg, run)["reasons"] == ["not_held"]
    moved = np.eye(4)
    moved[1, 3] = 0.02
    assert VC.judge_brace({}, T0, after(0.006), np.eye(4), moved, load, cfg, run)["reasons"] == ["hand_pushed"]
    assert VC.judge_brace({}, T0, after(0.0), np.eye(4), np.eye(4), load, cfg, {"completed": True, "postcondition": "not_reached"})["reasons"] == ["oracle:not_reached"]


def test_the_brace_pusher_drives_the_partner_into_the_brace():
    cfg = VC.ContactValidateConfig()
    pts = np.array(_box_geometry(0.077, 0.155, 0.242)["vertices"])
    T = np.eye(4)
    T[:3, 3] = [0.7, 0.1, 0.81]
    path = VC.pusher_path({"load": {"direction": [0.0, 1.0, 0.0]}}, T, pts, cfg, 1.0 / 30.0)
    face = 0.1 - 0.155 / 2
    half = cfg.brace_pusher_size_m[0] / 2
    assert path[0][1] + half == pytest.approx(face - cfg.brace_pusher_gap_m) and path[-1][1] + half == pytest.approx(face + cfg.brace_press_m)
    assert path[0][2] == pytest.approx(0.81 + 0.121) and len(path) - 1 == math.ceil((cfg.brace_pusher_gap_m + cfg.brace_press_m) / (cfg.brace_pusher_speed_m_s / 30.0))


def test_the_touching_partner_stand_in_is_kinematic(monkeypatch):
    """i7-contactfu (owner A2): OmniGibson honours kinematic_only only with a fixed base - the stand-in is built with both (the scene probe
    found it dynamic, 2.4 kg, before)."""
    import sys
    import types

    made = []

    class _Prim:
        def __init__(self, **kw):
            made.append(kw)

        def set_position_orientation(self, *a):
            pass

    mod = types.ModuleType("omnigibson.objects.primitive_object")
    mod.PrimitiveObject = _Prim
    monkeypatch.setitem(sys.modules, "omnigibson.objects.primitive_object", mod)
    w = VC.KitContactWorld.__new__(VC.KitContactWorld)
    w.cfg, w._stop = VC.ContactValidateConfig(), None
    w.tw = SimpleNamespace(th=th, scene=SimpleNamespace(add_object=lambda o: None), og=SimpleNamespace(sim=SimpleNamespace(step=lambda: None)))
    w._stop_obj()
    assert made and made[0]["fixed_base"] is True and made[0]["kinematic_only"] is True
    try:
        from omnigibson.utils.usd_utils import compute_kinematic_only
    except Exception:  # noqa: BLE001 - no OmniGibson here: the kwargs above are the contract
        return
    sc = th.tensor(VC.ContactValidateConfig().stop_size_m)
    assert compute_kinematic_only(made[0]["fixed_base"], sc, 0, 0, made[0]["kinematic_only"], False)
    assert not compute_kinematic_only(False, sc, 0, 0, True, False)  # the old construction was dynamic


def test_validate_library_writes_each_entrys_verdict_and_the_timing(glib):
    lib = CL.build_job(_job("box_of_cereal__iucshm__s0.8", ["touching_push"], params={"touching_push": {"yaws": 2}}), glib)["push"]
    calls = []

    class World:
        def settle(self, obj, geom, quat):
            calls.append(("settle", round(math.degrees(S.yaw_of_quat(quat)), 3)))
            return {"rest_id": "r0", "world_position": [0.0, 0.0, 0.81], "world_quat_xyzw": list(quat)}, "state"

        def plan_entry(self, obj, geom, entry, T, arm):
            return None if entry["class"].startswith("push_h") else {"q": th.zeros(28), "base": [0, 0, 0], "trunk": [0.5, 0.0]}

        def trial_push(self, obj, entry, plan, state, rest, arm):
            ok = "tip" in entry["class"]
            return {"verdict": "valid" if ok else "invalid", "reasons": [] if ok else ["short"]}

    cfg = VC.ContactValidateConfig(max_entries=len(lib["entries"]))
    tim = VC.validate_library(World(), lib, None, None, np.zeros((8, 3)), cfg, "right")
    assert calls == [("settle", 0.0), ("settle", 180.0)]  # one settled rest per yaw slot (screened entries are not settled for)
    v = {e["class"]: e["validation"] for e in lib["entries"] if e["yaw_deg"] == 0.0}
    screened = {k for k, x in v.items() if x["verdict"] == "screened"}
    assert all(v[k]["reasons"] == ["sampler_hand_overlap"] and v[k]["screen"] for k in screened)  # the sampler never records these: no trial
    assert all(e["validation"]["verdict"] == "screened" for e in lib["entries"] if (e.get("sampler_screen") or {}).get("hand_overlap"))
    assert all(x["verdict"] == "valid" for k, x in v.items() if "tip" in k and k not in screened)
    pads = [x for k, x in v.items() if k.startswith("push_h") and k not in screened]
    assert all(x["verdict"] == "invalid" and x["reasons"] == ["no_placement"] for x in pads)
    assert all(x["reasons"] == ["short"] for k, x in v.items() if k.startswith("push_blade") and k not in screened)
    assert any(x["verdict"] == "valid" for x in v.values())
    assert tim["kind"] == "push" and tim["rest_id"] == "r0" and tim["n_entries"] + tim.get("n_screened", 0) == len(lib["entries"]) and set(tim) >= {"settle_s", "plan_s", "trials_s", "total_s"}
    assert lib["validated"]["oracle_config"]["push_pre_dist_m"] == 0.04 and lib["validation_timing"] is tim


def test_an_axial_brace_library_is_the_axial_programs_stop_brace_at_the_goal(glib, tmp_path):
    """``axial_brace`` (the axial program's own ``_brace_candidates`` rule): per goal yaw, end and turn sense, the stop brace at the goal - the
    thin magazine (1.2 cm) gets the blade first; the entries are in the goal object frame and are the rule's own poses; the load is the
    arriving end's direction; a v2 rest that is not the upright identity one is refused."""
    lib = CL.build_job(_job("magazine__flat__s1", ["axial_brace"], params={"axial_brace": {"yaws": 2, "turn_deg": 40.0, "long_axis": [0.0, 1.0, 0.0]}}), glib)["brace"]
    e = lib["entries"]
    assert e and all(x["family"] == "axial_brace" and x["arm_role"] == "brace" and x["kind"] == "brace" and x["stop"] == CPZ.brace_stop_rule() for x in e)
    assert {x["yaw_deg"] for x in e} == {0.0, 180.0} and e[0]["class"].split("_", 1)[1].startswith("blade_0")  # thin: the blade first
    assert any("_h" in x["class"] or "_v" in x["class"] for x in e)  # the pads on the stop face too
    x = e[0]
    g = x["goal"]
    # the rule itself on the same inputs gives the entry's contact at the goal pose
    from ispatialgym.sampling.v2 import tabletop as TT

    pts = np.array(_box_geometry(0.20, 0.28, 0.012)["vertices"])
    hull = th.as_tensor(pts, dtype=th.float32)
    com_local = hull.mean(dim=0)
    q0 = C.quat_from_yaw(math.radians(g["init_yaw_deg"]))
    pose_init = (th.tensor([0.0, 0.0, 0.0]), q0)
    com = pose_init[0] + C.quat_rotate(q0, com_local)
    long_l = th.tensor([0.0, 1.0, 0.0])
    proj = (hull - com_local) @ long_l
    half = float(proj.max() if g["end_sign"] > 0 else -proj.min()) - CL.AXIAL_END_INSET_M
    end = com + C.quat_rotate(q0, long_l) * g["end_sign"] * half
    goal_pose, stop_pts, d_stop = CPZ.axial_stop_geometry(hull, com_local, pose_init, end, com, g["spin"], math.radians(g["yaw_goal_deg"]), TT.contact_height(0.0, 0.012))
    _pre, contact, kind, _x = next(iter(CPZ.axial_brace_candidates(CPZ.MAP_FRAME, hull, stop_pts, d_stop, None, pose_init, com, g["turn_rad"],
                                                                     z_push=CPZ.axial_push_height(0.0, 0.012), face_z=(0.0, 0.012), height=0.012)))
    w = CPZ.entry_world(x, goal_pose)
    assert kind == x["extras"]["kind"] and th.allclose(w["contact"][0], contact[0], atol=1e-5) and C.quat_angle(w["contact"][1], contact[1]) < 1e-4
    assert np.allclose(x["load"]["direction"], d_stop.numpy(), atol=1e-5)
    root = str(tmp_path / "g2")
    _model_dir(root, "magazine__lying__s1", (0.20, 0.28, 0.012), rest_q=(0.7071068, 0.0, 0.0, 0.7071068))
    with pytest.raises(ValueError, match="identity"):
        CL.build_job(_job("magazine__lying__s1", ["axial_brace"]), root)


def test_the_axial_brace_load_pushes_at_the_arriving_end():
    cfg = VC.ContactValidateConfig()
    pts = np.array(_box_geometry(0.20, 0.28, 0.012)["vertices"])
    T = np.eye(4)
    T[:3, 3] = [0.6, 0.0, 0.8126]
    e = {"load": {"direction": [1.0, 0.0, 0.0], "point_object": [0.0, 0.11, 0.006]}}
    path = VC.pusher_path(e, T, pts, cfg, 1.0 / 30.0)
    assert path[0][1] == pytest.approx(0.11) and path[-1][0] + cfg.brace_pusher_size_m[0] / 2 == pytest.approx(0.6 - 0.10 + cfg.brace_press_m)
    assert path[0][2] == pytest.approx(0.8126 + cfg.brace_pusher_size_m[2] / 2 + 0.001)  # the 5 cm box above the support, not in it


# ---------------------------------------------------------------------------------------------------------------- the first cluster run's fixes


def test_an_entry_that_cannot_run_fails_alone_and_the_model_goes_on(glib):
    """(b) an entry whose pre-contact the oracle cannot reach (``SkillError plan_failed``) is ``unreachable``, a chain whose contact / goal has no
    IK from the placement is ``unreachable`` without a trial, any other error ``invalid`` - the other entries are validated."""
    from ispatialgym.oracle.motions import SkillError

    lib = CL.build_job(_job("box_of_cereal__iucshm__s0.8", ["touching_push"], params={"touching_push": {"yaws": 1}}), glib)["push"]
    live = [e for e in lib["entries"] if not (e.get("sampler_screen") or {}).get("hand_overlap")][:4]
    lib["entries"] = live
    names = [e["class"] for e in live]

    class World:
        def settle(self, obj, geom, quat):
            return {"rest_id": "r0", "world_position": [0.0, 0.0, 0.81], "world_quat_xyzw": list(quat)}, "state"

        def plan_entry(self, obj, geom, entry, T, arm):
            assert T is not None and T[2, 3] == pytest.approx(0.81)  # the SETTLED rest pose, not the live object
            return {"q": th.zeros(28), "base": [0, 0, 0], "trunk": [0.5, 0.0], "chain_unreachable": ["goal"] if entry["class"] == names[1] else []}

        def trial_push(self, obj, entry, plan, state, rest, arm):
            if entry["class"] == names[0]:
                raise SkillError("plan_failed", "arm_to(right, pose): IK failed: goal pose unreachable")
            if entry["class"] == names[2]:
                raise RuntimeError("boom")
            return {"verdict": "valid", "reasons": []}

    tim = VC.validate_library(World(), lib, None, None, np.zeros((8, 3)), VC.ContactValidateConfig(), "right")
    v = [e["validation"] for e in lib["entries"]]
    assert v[0]["verdict"] == "unreachable" and v[0]["reasons"] == ["oracle:plan_failed", "unreachable"]
    assert v[1]["verdict"] == "unreachable" and v[1]["reasons"] == ["chain_unreachable"] and v[1]["trials"] == []
    assert v[2]["verdict"] == "invalid" and v[2]["reasons"] == ["error:RuntimeError"] and v[3]["verdict"] == "valid"
    assert tim["n_errors"] == 2 and tim["n_entries"] == 4


def test_the_validator_reads_the_model_dir_the_build_read(glib):
    """(c) the build records the model directory it read (``object.source``: a manifest's ``model_dir``); the validator's job uses it."""
    d = os.path.join(glib, "models", "box_of_cereal__dkhgxn__s1")
    job = {**_job("box_of_cereal__dkhgxn__s1", ["touching_brace"]), "entry": None, "model_dir": d}
    lib = CL.build_job(job, None)["brace"]
    assert lib["object"]["source"] == d
    vj = VC.library_job(lib, "box_of_cereal__dkhgxn__s1")
    assert vj["model_dir"] == d and CL.load_object("/nonexistent/glib", vj)[1] == "r0"


def test_the_trial_trace_names_the_mechanism():
    """(a) the per-trial debug record: the object's first motion BEFORE the hand reached the contact (flung by the arm / the start pose) vs
    after it, with the loads of that step and a sampled trace."""
    T0 = np.eye(4)
    Tc = np.eye(4)
    Tc[:3, 3] = [0.0, -0.1, 0.05]
    tr = VC.TrialTrace(T0, Tc, every=2)
    E = np.eye(4)
    E[:3, 3] = [0.0, -0.3, 0.3]
    flung = np.eye(4)
    flung[:3, 3] = [0.0, 0.02, 0.0]
    tr(T0, E, {"finger_obj": 0.0, "other_slab": 0.0, "other_link": None})
    tr(flung, E, {"finger_obj": 0.0, "other_slab": 1.5, "other_link": "right_arm_link6"})
    rec = tr.end(flung, E)
    assert rec["first_motion"]["step"] == 2 and rec["first_motion"]["before_contact"] and rec["first_motion"]["eef_to_contact_m"] > 0.2
    assert rec["first_motion"]["loads"]["other_link"] == "right_arm_link6" and rec["at_contact"] is None and len(rec["trace"]) == 1 and rec["steps"] == 2
    tr2 = VC.TrialTrace(T0, Tc)
    tr2(T0, Tc)
    pushed = np.eye(4)
    pushed[:3, 3] = [0.0, 0.01, 0.0]
    tr2(pushed, Tc)
    assert tr2.rec["at_contact"]["step"] == 1 and tr2.rec["first_motion"]["before_contact"] is False


def test_the_start_pose_avoids_a_hand_inside_the_object(glib):
    """(a) the robot is TELEPORTED to the trial's start: the executed pre-contact when the hand beyond the fingers clears the object, else the
    recorded one, else no trial; and the builder records the sampler's own hand screen (entries it never records are not validated)."""
    lib = CL.build_job(_job("box_of_cereal__iucshm__s0.8", ["touching_push"], params={"touching_push": {"yaws": 1}}), glib)["push"]
    pts = np.array(_box_geometry(*CEREAL)["vertices"])
    T = CL.object_frame([0, 0, 0, 1.0], 0.0, 0.0)
    tips = [e for e in lib["entries"] if e["class"].startswith("push_tip_h-@") and not e["sampler_screen"]["hand_overlap"]]
    assert tips and VC.start_pose_of(tips[0], pts, T) == "executed"
    assert all("hand_overlap" in e["sampler_screen"] and e["sampler_screen"]["against"] == "hull" for e in lib["entries"])
    fat = pts * np.array([1.0, 2.5, 1.0])  # the pushed face 2.5x farther out: the hand at both pre-contacts inside the object
    assert VC.start_pose_of(tips[0], fat, T) is None


def test_touching_checks_the_wide_brace_at_its_opening(monkeypatch):
    """The touching sampler's two-arm solve, its diagnostic and the brace retreat use the brace record's opening (a wide stop's
    ``fingers_opening_m``), as the oracle and the program check; ``ISPATIALGYM_TOUCH_BRACE_AT_OPENING=0`` = the checks of before."""
    from ispatialgym.sampling.v2.tasks import iors_small as IS
    from ispatialgym.tests import test_touching_brace_open_push_speed as BO

    calls = []

    class IKC:
        def exact_both_arms(self, work, other, st, ig_w, ig_o, fingers=None, q_first_other=None, **kw):
            calls.append(("both", fingers, kw.get("fingers_other", "absent")))
            return SimpleNamespace(ok=False, reason="no")

        def exact(self, poses, st, arm, ignore_for_last=(), cache=True, fingers=None, **kw):
            calls.append(("exact", arm, fingers))
            return SimpleNamespace(ok=True, reason="")

        def ignoring(self, objs):
            import contextlib

            return contextlib.nullcontext()

        def q_locked(self, st, **kw):
            return None

    def choose_first(ikc, arm, stream, key, label, chain, evaluate, q_start, stage):
        c = next(iter(stream))
        evaluate(SimpleNamespace(chain=chain(c), payload=c, q_first=None))
        return None, {"options": []}, []

    monkeypatch.setattr(IS.O, "choose_first", choose_first)
    s = BO._sampler(gen={"brace_open_when_wide": True})
    s.ikc, s.obj = IKC(), (lambda role: role)
    s.posed = lambda role, pose: __import__("contextlib").nullcontext()
    s.hand_overlap = lambda *a, **k: None
    s._touching_brace_choice = IS.IORSSmall._touching_brace_choice.__get__(s)
    st, p_init, b_point, p_hull, face_z = BO._brace_inputs(s)
    s._touching_brace_choice(st, [], BO.T_PLACE, BO.P_POSE, b_point, face_z, BO.DIRECTION, 0.05, p_hull, p_init, [], [], push_fingers_q=0.0)
    opening = next(iter(s.touching_brace_candidates(st, b_point, face_z, -BO.DIRECTION, partner_hull_w=p_hull, push_point=p_init)))[3]["fingers_opening_m"]
    assert opening == pytest.approx(0.0315)
    assert calls[0] == ("both", 0.0, pytest.approx(0.0315)) and calls[1] == ("exact", "left", pytest.approx(0.0315))  # the solve, the diagnostic
    calls.clear()
    monkeypatch.setenv("ISPATIALGYM_TOUCH_BRACE_AT_OPENING", "0")
    s._touching_brace_choice(st, [], BO.T_PLACE, BO.P_POSE, b_point, face_z, BO.DIRECTION, 0.05, p_hull, p_init, [], [], push_fingers_q=0.0)
    assert calls[0] == ("both", 0.0, "absent") and calls[1] == ("exact", "left", 0.0)  # before: the push's fingers, the diagnostic closed


def test_exact_both_arms_puts_the_other_arms_fingers_on_its_own_gripper():
    from ispatialgym.sampling.v2 import ik as IK

    ikc = IK.IKChecker.__new__(IK.IKChecker)
    robot = SimpleNamespace(arm_control_idx={"left": th.tensor([0, 1]), "right": th.tensor([2, 3])},
                            gripper_control_idx={"left": th.tensor([4, 5]), "right": th.tensor([6, 7])})
    ikc.kit = SimpleNamespace(robot=robot, collision_free=lambda q, margin=0.0: [True])
    seen = []

    def exact(poses, stance, arm, ignore=(), q_seed=None, cache=True, fingers=None, q_first=None, **kw):
        seen.append((arm, None if q_seed is None else q_seed.clone(), fingers))
        q = th.zeros(8) if q_seed is None else q_seed.clone()
        if q_seed is None:
            q[4:8] = float(fingers)  # both grippers at the working arm's fingers
        return SimpleNamespace(ok=True, q=q, reason="")

    ikc.exact = exact
    ikc.ignoring = lambda objs: __import__("contextlib").nullcontext()
    r = ikc.exact_both_arms(("right", [1]), ("left", [2]), None, fingers=0.0, fingers_other=0.0315)
    assert r.ok and seen[1][0] == "left" and seen[1][2] == pytest.approx(0.0315)
    assert th.allclose(seen[1][1][4:6], th.tensor([0.0315, 0.0315])) and th.allclose(seen[1][1][6:8], th.zeros(2))  # the brace hand open, the push hand closed
    seen.clear()
    ikc.exact_both_arms(("right", [1]), ("left", [2]), None, fingers=0.0)
    assert th.allclose(seen[1][1][4:8], th.zeros(4)) and seen[1][2] == 0.0  # as before


def test_the_validation_order_samples_every_yaw():
    es = [{"yaw_deg": y, "class": f"c{k}"} for y in (0.0, 45.0, 90.0) for k in range(3)]
    order = [(e["yaw_deg"], e["class"]) for e in VC.entry_order(es)]
    assert order[:3] == [(0.0, "c0"), (45.0, "c0"), (90.0, "c0")] and order[3] == (0.0, "c1") and len(order) == 9


# ---------------------------------------------------------------------------------------------------------------- the second cluster run's fixes

#: box_of_sanitary_napkins/humkqw's collision box (its v2 geometry.json, root at the box centre) and the classes the SAMPLER rejected before it
#: executed push_blade_lean20@3.5cm on touching_01 ($OFH/out/abv2_aim2/inter_object_relation_satisfaction-touching_01/g300068-300071, instances
#: 300068-300070: summary.touching.classes_failed, identical in the three)
HUMKQW = (0.069541, 0.095778, 0.109204)
RECORDED_FAILED = ["push_blade_0-@0.5cm: knuckle -15 mm from the hull (margin 5 mm)", "push_blade_0@0.5cm: knuckle -15 mm from the hull (margin 5 mm)",
                   "push_blade_0-@1.5cm: knuckle -15 mm from the hull (margin 5 mm)", "push_blade_0@1.5cm: knuckle -15 mm from the hull (margin 5 mm)",
                   "push_blade_lean20-@2.5cm: knuckle -3 mm from the hull (margin 5 mm)", "push_blade_lean20@2.5cm: knuckle -3 mm from the hull (margin 5 mm)"]


def test_the_librarys_hand_screen_is_the_samplers_on_a_recorded_touching_instance(tmp_path):
    """(1) the second run validated 48 / 48 napkins entries as ``screened``: not a frame bug - the round-robin's first entries per yaw were the
    lowest swept blades, which the SAMPLER rejects the same way (the recorded ``classes_failed``) before it executes push_blade_lean20@3.5cm;
    the library's screen reproduces those six verdicts message for message and passes the executed class. The validator now caps the
    entries the sampler would try (screened ones are marked, not counted)."""
    root = str(tmp_path / "glib")
    d = _model_dir(root, "box_of_sanitary_napkins__humkqw__s1", HUMKQW, h_root=HUMKQW[2] / 2)
    g = json.load(open(os.path.join(d, "geometry.json")))
    g["pieces"][0]["vertices"] = [[x, y, z - HUMKQW[2] / 2] for x, y, z in g["pieces"][0]["vertices"]]  # the root at the box centre, as humkqw
    S.save_json(os.path.join(d, "geometry.json"), g)
    lib = CL.build_job(_job("box_of_sanitary_napkins__humkqw__s1", ["touching_push"], params={"touching_push": {"yaws": 4}}), root)["push"]
    e0 = [e for e in lib["entries"] if e["yaw_deg"] == 0.0]
    got = [f"{e['class']}: {e['sampler_screen']['hand_overlap']}" for e in e0[:6]]
    assert got == RECORDED_FAILED
    first_ok = next(e for e in e0 if not e["sampler_screen"]["hand_overlap"])
    assert first_ok["class"] in ("push_blade_lean20-@3.5cm", "push_blade_lean20@3.5cm")  # the executed class (its '-' turn failed the IK there)
    # the validation order: the sampler's candidates only, every yaw, in the sampler's order within a yaw
    order = VC.entry_order(lib["entries"])
    assert all(not (e.get("sampler_screen") or {}).get("hand_overlap") for e in order[:8]) and order[0]["class"] == first_ok["class"]
    assert {e["yaw_deg"] for e in order[:4]} == {0.0, 90.0, 180.0, 270.0}


def test_the_base_placement_is_the_samplers_stance_relation():
    """(2) the trial's base: the contact ``ahead`` / ``lateral`` of the base as measured on the demos (the touching push 0.69-0.84 m ahead,
    0.06-0.20 m right; the brace on the left), turned into ``TabletopWorld.plan``'s object-relative offsets."""
    contact, obj = (0.70, -0.10), (0.70, -0.06)
    offs = VC.placement_offsets("touching_push", contact, obj, 0.25, 0.30)
    rel = VC.STANCE_RELATIONS["touching_push"]
    assert len(offs) == len(rel["ahead_m"]) * len(rel["lateral_m"])
    for (off0, off1), (ah, la) in zip(offs, [(a, b) for a in rel["ahead_m"] for b in rel["lateral_m"]]):
        bx, by = obj[0] - (0.25 + 0.30 + off0), obj[1] - off1  # TabletopWorld.plan's base at yaw 0
        assert (contact[0] - bx, contact[1] - by) == (pytest.approx(ah), pytest.approx(la))
    assert all(la < 0 for la in rel["lateral_m"]) and all(la >= 0 for la in VC.STANCE_RELATIONS["touching_brace"]["lateral_m"])
    assert set(VC.STANCE_RELATIONS) >= {"touching_push", "touching_brace", "axial_push", "axial_brace"}


def test_an_arc_push_is_judged_by_its_turn():
    """(2) the axial push turns the target about its centre of mass: the straight rules (moved along d, sideways slip, tip gap) do not apply."""
    cfg = VC.ContactValidateConfig()
    e = {"push": {"direction": [0.0, 1.0, 0.0], "stroke_m": 0.1, "arc": True}, "arc": {"turn_rad": 0.7}, "extras": {}}
    T0 = np.eye(4)

    def turned(a, dx=0.0, tilt=0.0):
        T = np.eye(4)
        c, s_, ct, st = math.cos(a), math.sin(a), math.cos(tilt), math.sin(tilt)
        T[:3, :3] = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]]) @ np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        T[0, 3] = dx
        return T

    run = {"completed": True, "postcondition": None}
    ok = VC.judge_push(e, T0, turned(0.62), np.eye(4), np.zeros((8, 3)), cfg, run)
    assert ok["verdict"] == "valid" and ok["measures"]["needed_rad"] == pytest.approx(0.5)
    assert VC.judge_push(e, T0, turned(0.3), np.eye(4), np.zeros((8, 3)), cfg, run)["reasons"] == ["short_turn"]
    assert VC.judge_push(e, T0, turned(-0.7), np.eye(4), np.zeros((8, 3)), cfg, run)["reasons"] == ["short_turn"]  # the wrong sense
    assert VC.judge_push(e, T0, turned(0.7, dx=0.08), np.eye(4), np.zeros((8, 3)), cfg, run)["reasons"] == ["drift"]
    assert "topple" in VC.judge_push(e, T0, turned(0.7, tilt=0.4), np.eye(4), np.zeros((8, 3)), cfg, run)["reasons"]


def test_the_verdict_table(tmp_path, glib):
    lib = CL.build_job(_job("box_of_cereal__iucshm__s0.8", ["touching_push"], params={"touching_push": {"yaws": 1}}, validate={"support_z": 1.0505}), glib)["push"]
    assert lib["validate"] == {"support_z": 1.0505}
    for i, e in enumerate(lib["entries"]):
        ok = i % 2 == 0
        e["validation"] = {"verdict": "valid" if ok else "invalid", "reasons": [] if ok else ["short"],
                           "trials": [{"verdict": "valid" if ok else "invalid", "measures": {"moved_along_m": 0.07 if ok else 0.03}}]}
    root = str(tmp_path / "c")
    CL.save_library(root, lib, "box_of_cereal__iucshm__s0.8")
    rows = {r["class"]: r for r in CL.verdict_table(root)}
    assert CL.class_family("push_tip_h-@5.0cm") == "push_tip_h-" and CL.class_family("+end-spin_blade_0@f2#") == "+end-spin_blade_0"
    assert CL.class_family("+end-spin_blade_0@t65") == "+end-spin_blade_0@t65" == CL.class_family("+end-spin_blade_0@t65@f1")
    r = next(iter(rows.values()))
    assert sum(r["verdicts"].values()) >= 1 and all(set(m) <= {"valid", "invalid"} for m in (x["measures"] for x in rows.values()))
    assert any("valid" in x["measures"] and x["measures"]["valid"]["moved_along_m"]["median"] == 0.07 for x in rows.values())


# ---------------------------------------------------------------------------------------------------------------- run C: replays of recorded demos

REPLAY = os.path.join(os.path.dirname(__file__), "data", "contact_replay")


def _replay(inst, kind):
    o = json.load(open(os.path.join(REPLAY, f"{inst}.oracle.json")))
    t = json.load(open(os.path.join(REPLAY, f"{inst}.tro_state.json")))
    return VC.replay_situation(o, kind, t)


def _recorded_model(root, eid):
    o = json.load(open(os.path.join(REPLAY, "objects.json")))[eid]
    d = os.path.join(root, "models", eid)
    os.makedirs(d, exist_ok=True)
    rest = {"schema": S.REST, "rest_id": "r0", "quat_xyzw": o["rest"]["quat_xyzw"], "h_root": o["rest"]["h_root"], "stable": True}
    S.save_json(os.path.join(d, "library.json"), {"schema": S.LIBRARY, "category": o["category"], "model": o["model"], "scale": o["scale"], "rests": {"r0": rest}, "grasps": []})
    S.save_json(os.path.join(d, "geometry.json"), {"format": "isg-object-geometry/1", "pieces": [{"name": "p", "vertices": o["vertices"], "faces": o["faces"]}]})
    return d, o["rest"]["quat_xyzw"]


#: (recorded demo, kind, the contacted object's v2 entry (geometry + rest copied in data/contact_replay/objects.json), family, the alias's knobs,
#:  the class the sampler executed): touching_02 (fix2 g300008: push_tip_h- + the wide brace), touching_01 (abv2_aim2 g300068: the leaning blade),
#:  axial_01 (fix2 g300006: the arc push)
CASES = [("touching_02", "push", "box_of_cereal__iucshm__s0.8", "touching_push", {"push_height_sweep": False, "push_gripper_open": False}, "push_tip_h-"),
         ("touching_02", "brace", "box_of_cereal__dkhgxn__s1", "touching_brace", {"brace_open_when_wide": True}, "brace_wide_h-"),
         ("touching_01", "push", "box_of_sanitary_napkins__humkqw__s1", "touching_push", {}, "push_blade_lean20@3.5cm"),
         ("axial_01", "push", "board_game__qgvsnb__s1", "axial_push", {}, None)]


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_the_library_reproduces_the_recorded_demo_records(case, tmp_path):
    """Run C's replay finding 1: the library is NOT what diverges - built at the recorded object's exact yaw in the record's stance frame, the
    entry of the executed class (axial: the best of the rule's classes) lands on the recorded contact within 1.5 mm / 1 deg and its pre-contact
    within 2 mm (the recorded objects' v2 geometry and rest; the stance frame from the sidecar)."""
    inst, kind, eid, fam, gen, cls = case
    sit = _replay(inst, kind)
    d, rest_q = _recorded_model(str(tmp_path), eid)
    R = sit["T_base_object"][:3, :3] @ S.quat_to_mat(np.asarray(rest_q, dtype=float)).T
    params = {"yaws_deg": [math.degrees(math.atan2(R[1, 0], R[0, 0]))], **({"long_axis": [0.0, 1.0, 0.0]} if fam == "axial_push" else {})}
    sub = "alignment-axial" if fam == "axial_push" else TOUCHING
    lib = CL.build_job({"id": eid, "model_dir": d, "families": [fam], "generation": gen, "subtask": sub, "params": {fam: params}}, None)[kind]
    cands = [e for e in lib["entries"] if cls is None or e["class"] == cls]
    best = min((VC.replay_entry_check(sit, e) for e in cands), key=lambda c: c["contact_err_m"])
    assert best["contact_err_m"] < 0.0015 and best["pre_err_m"] < 0.0025
    assert cls is None or best["contact_angle_deg"] < 1.0  # (axial: the class's twin turn shares the position: 0 or 180 deg)


def test_the_replayed_situations_are_the_samplers_stance_relation():
    """Run C's replay finding 2: in the record's stance frame the touching push runs along +y, its contact 0.73-0.84 m ahead and 0.10-0.13 m
    right, the brace 0.08 m left; the axial push 0.72 m ahead; the witness configuration of the pre-contact is the oracle's own (its base moved
    to the origin); the support heights are the real ones."""
    rel = VC.STANCE_RELATIONS
    for inst, kind, fam in (("touching_02", "push", "touching_push"), ("touching_02", "brace", "touching_brace"), ("touching_01", "push", "touching_push"),
                            ("axial_01", "push", "axial_push"), ("axial_03", "push", "axial_push")):
        sit = _replay(inst, kind)
        a, l_ = sit["relation"]["contact_ahead_m"], sit["relation"]["contact_lateral_m"]
        assert min(rel[fam]["ahead_m"]) - 0.12 <= a <= max(rel[fam]["ahead_m"]) + 0.25, (inst, a)
        assert min(rel[fam]["lateral_m"]) - 0.15 <= l_ <= max(rel[fam]["lateral_m"]) + 0.05, (inst, l_)
        assert sit["witness_q"] is not None and sit["witness_q"][0] == 0.0 and sit["witness_q"][1] == 0.0 and sit["witness_q"][5] == 0.0
        if fam != "axial_push":
            assert np.allclose(sit["record"]["direction"], [0.0, 1.0 if kind == "push" else -1.0, 0.0], atol=0.01)
    assert _replay("touching_02", "push")["support_z"] == pytest.approx(1.0505, abs=1e-3) and _replay("touching_01", "push")["support_z"] == pytest.approx(0.8126, abs=2e-3)
    assert _replay("touching_02", "push")["object_from"] == "place" and _replay("touching_02", "brace")["object_from"] == "program start"
    e = VC.record_entry(_replay("axial_01", "push"))
    assert e["push"]["arc"] and e["arc"]["turn_rad"] != 0.0 and e["pre_pose_rule"]["rule"] == "recorded"


def test_world_extras_and_the_axial_records_extras(glib):
    """Run C's replay findings 3-4: (a) the library's world-valued extras (fingertip_z, contact_z, contact_point) are in the rule frame
    (support z 0) - the oracle reads them as world values, so they move to the trial's object pose; (b) the axial push entries carry the
    sampler's record's extras (none): their blade fingertip_z made the oracle's palm lift raise every axial push ~0.8 m (plan_failed)."""
    lib = CL.build_job(_job("magazine__flat__s1", ["axial_push"], params={"axial_push": {"yaws": 1}}), glib)["push"]
    assert all(e["extras"] == {} and "fingertip_z_rule" in e["axial"] for e in lib["entries"])
    e = {"extras": {"fingertip_z": 0.005, "contact_z": 0.06, "contact_point": [0.0, -0.05, 0.005], "stop_clearance_m": 0.005},
         "T_rule_object": {"position": [0.0, 0.0, 0.1], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}}
    T = np.eye(4)
    T[:3, 3] = [0.7, -0.1, 0.9126]  # the object on a 0.8126 support
    w = VC.world_extras(e, T)
    assert w["fingertip_z"] == pytest.approx(0.8176) and w["contact_z"] == pytest.approx(0.8726) and w["contact_point"] == pytest.approx([0.7, -0.15, 0.8176])
    assert w["stop_clearance_m"] == 0.005 and VC.world_extras({"extras": {"fingertip_z": 1.0}}, T)["fingertip_z"] == 1.0  # no rule frame: as recorded


def test_the_validator_ik_runs_in_a_clean_world():
    """Run C's replay finding 5 (the cause of '1 valid, then 47 no_placement / chain_unreachable'): the oracle's planning syncs the cuRobo
    world WITH the slab and the object; the validator's own IK (placement, chain check) then saw them at the contact. Every plan_entry first
    resets it to the robot and the ground (the support and the contacted object ignored, as the sampler's exact IK at a contact)."""
    calls = []
    kw = VC.KitContactWorld.__new__(VC.KitContactWorld)
    kw._pusher, kw._stop = "pusher", "stop"
    kw.tw = SimpleNamespace(slab="slab", backend=SimpleNamespace(update_world=lambda ignore_objects=(): calls.append(list(ignore_objects))))
    kw.clean_world("obj")
    assert calls == [["slab", "obj", "pusher", "stop"]]


def test_an_axial_trial_runs_on_the_tasks_com_hinge():
    """Run E: the axial demos turn the target on the vertical COM hinge ``AlignmentAxial.after_settle`` installs (``hinge_sink_m`` below the
    top, the variant's resistance); without it the replays drifted 6.8 / 19.5 cm and the arc lagged. The validator installs the same hinge for
    the axial families and the alignment-axial replays, nothing for the others."""
    assert VC.axial_hinged({"family": "axial_push"}, None) and VC.axial_hinged({"family": "axial_brace"}, None)
    assert VC.axial_hinged({"family": "replay"}, "alignment-axial-board_game-breakfast_table")
    assert not VC.axial_hinged({"family": "touching_push"}, "inter_object_relation_satisfaction-touching_small_scale-box_of_cereal_s080-box_of_cereal-bar")
    board = VC.axial_hinge_spec("alignment-axial-board_game-breakfast_table", 0.5, (0.4, 0.3))
    assert board["resistance"] is None and board["sink_m"] == pytest.approx(0.001)  # the frictionless hinge
    paper = VC.axial_hinge_spec("alignment-axial-paperback_book-desk", 0.3, (0.2, 0.13))
    assert paper["resistance"] is not None and paper["resistance"]["friction"] > 0 and paper["resistance"]["damping"] > 0  # its customs resistance
    kw = VC.KitContactWorld.__new__(VC.KitContactWorld)
    kw.variant = None
    assert kw._hinge_on(None, {"family": "touching_brace"}, np.zeros((8, 3))) is None
    kw._hinge_off(None, None)  # no hinge: nothing to remove


def test_the_screened_diagnostic_validates_only_the_screened_entries(glib):
    """Run E owner question 2 (a DIAGNOSTIC, its own root): ``diagnose_screened`` runs the physics trial on the entries the sampler's hand_overlap
    screen removes - their verdicts carry ``diagnostic`` and the screen - and leaves the live entries' records alone; the screen is unchanged."""
    lib = CL.build_job(_job("box_of_cereal__iucshm__s0.8", ["touching_push"], params={"touching_push": {"yaws": 1}}), glib)["push"]
    scr = [e for e in lib["entries"] if (e.get("sampler_screen") or {}).get("hand_overlap")]
    assert scr and len(scr) < len(lib["entries"])

    class World:
        def settle(self, obj, geom, quat):
            return {"rest_id": "r0", "world_position": [0.0, 0.0, 0.81], "world_quat_xyzw": list(quat)}, "state"

        def plan_entry(self, obj, geom, entry, T, arm):
            return {"q": th.zeros(28), "base": [0, 0, 0], "trunk": [0.5, 0.0]}

        def trial_push(self, obj, entry, plan, state, rest, arm):
            return {"verdict": "valid", "reasons": []}

    VC.validate_library(World(), lib, None, None, np.zeros((8, 3)), VC.ContactValidateConfig(max_entries=len(lib["entries"]), diagnose_screened=True), "right")
    assert all(e["validation"]["verdict"] == "valid" and e["validation"]["diagnostic"] and e["validation"]["screen"] for e in scr)
    assert all("validation" not in e for e in lib["entries"] if e not in scr)


def test_the_validation_order_samples_every_axial_end_and_turn_sense():
    """i7-contact 2026-11-07: the axial libraries loop yaw -> end -> spin -> class, so the per-yaw round-robin validated the ``+end+spin``
    group only (48 = 6 per yaw x 8) and the maps held no other end or turn sense (axial brace present 0 / 11 on i7_contact: every written
    instance's stop face was the +end's, at the goal 180 deg from the stance - ik_pre). The order now round-robins the (yaw, end-spin) groups;
    a non-axial library keeps its order."""
    es = [{"yaw_deg": y, "class": f"{g}_blade_0@f{k}"} for y in (0.0, 45.0, 90.0) for g in ("+end+spin", "+end-spin", "-end+spin", "-end-spin")
          for k in range(3)]
    order = VC.entry_order(es)
    first = {(e["yaw_deg"], VC.axial_group(e)) for e in order[:12]}
    assert len(first) == 12 and all(e["class"].endswith("@f0") for e in order[:12])  # every (yaw, end, spin) once before any second class
    assert [VC.axial_group(e) for e in order[:3]] == ["+end+spin"] * 3 and [e["yaw_deg"] for e in order[:3]] == [0.0, 45.0, 90.0]
    assert VC.axial_group({"class": "push_tip_h-@5.0cm"}) == "" and VC.axial_group({"class": "-end+spin_30-@t45"}) == "-end+spin"
    touch = [{"yaw_deg": y, "class": f"c{k}"} for y in (0.0, 45.0) for k in range(2)]
    assert [(e["yaw_deg"], e["class"]) for e in VC.entry_order(touch)] == [(0.0, "c0"), (45.0, "c0"), (0.0, "c1"), (45.0, "c1")]


def test_a_rerun_keeps_the_earlier_verdicts_when_asked():
    """``skip_validated``: the entries an earlier run validated are neither re-run nor counted against ``max_entries`` (the new order's sample
    adds to the old one)."""
    es = [{"yaw_deg": 0.0, "class": f"{g}_blade_0@f0"} for g in ("+end+spin", "+end-spin", "-end+spin", "-end-spin")]
    es[0]["validation"] = {"verdict": "valid", "trials": ["kept"]}
    ran = []

    class World:
        def settle(self, obj, geom, q):
            return {}, None

        def plan_entry(self, obj, geom, e, T, arm):
            ran.append(e["class"])
            return None

    lib = {"kind": "brace", "object": {"rest": {"quat_xyzw": [0.0, 0.0, 0.0, 1.0]}}, "entries": es}
    tim = VC.validate_library(World(), lib, None, None, np.zeros((1, 3)), VC.ContactValidateConfig(max_entries=2, skip_validated=True), "left")
    assert ran == ["+end-spin_blade_0@f0", "-end+spin_blade_0@f0"] and tim["n_kept"] == 1 and es[0]["validation"]["trials"] == ["kept"]
    ran.clear()
    VC.validate_library(World(), lib, None, None, np.zeros((1, 3)), VC.ContactValidateConfig(max_entries=2), "left")
    assert ran == ["+end+spin_blade_0@f0", "+end-spin_blade_0@f0"]  # the default re-runs from the top


def test_the_hand_overlap_rank_switch_is_off_by_default_and_ranks_when_on(monkeypatch):
    """Owner item 3 (2026-11-07): the hand-overlap screen of the touching PUSH is a ranking hint only under the default-off switch - the
    sampler tries the clean candidates first and the overlapping ones after them (not rejected, the overlap named in the extras); the
    validator (``screen_as_rank``) validates the screened entries after the live ones, counted, instead of marking them screened."""
    from ispatialgym.sampling.v2 import orient as O
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    monkeypatch.delenv("ISPATIALGYM_HAND_OVERLAP_RANK", raising=False)
    assert not IS.hand_overlap_rank_on()
    pose = (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0]))
    cands = [("push_blade_0", pose, {}), ("push_tip_h", pose, {}), ("push_blade_lean8", pose, {})]
    over = {"push_blade_0": "knuckle -15 mm from the box (margin 5 mm)", "push_blade_lean8": "knuckle -3 mm from the box (margin 5 mm)"}
    seen = {}

    def choose_first(ikc, arm, cs, key, label, chain, evaluate, q_start, camera_idx, stage):
        cs = list(cs)
        seen["order"] = [c[0] for c in cs]
        seen["verdicts"] = [evaluate(SimpleNamespace(chain=chain(c), payload=c, q_first=None))[3] for c in cs]
        seen["hints"] = [c[2].get("hand_overlap_hint") for c in cs]
        return None, {}, []

    monkeypatch.setattr(O, "choose_first", choose_first)
    me = SimpleNamespace(shelf_start=None, push_height_sweep_on=lambda: False, push_gripper_open_on=lambda: False, pair_goal_align=lambda: None,
                         touching_push_candidates=lambda *a, **k: iter(cands),
                         ikc=SimpleNamespace(exact=lambda *a, **k: SimpleNamespace(ok=True, reason=""), q_locked=lambda st: None))
    # the overlap is keyed by the candidate (identity of its eef pose tensor)
    poses = {c[0]: (th.zeros(3) + i, pose[1]) for i, c in enumerate(cands)}
    cands[:] = [(k, poses[k], x) for k, _p, x in cands]
    me.push_hand_overlap = lambda init, goal, tp, d, dist, against="box": over.get(next(k for k, p in poses.items() if p[0] is init[0]))
    monkeypatch.setattr(IS.O, "chain_qs", lambda *a, **k: [])
    IS.IORSSmall._touching_push_choice(me, None, pose, th.zeros(3), (0.0, 0.1), th.tensor([0.0, 1.0, 0.0]), 0.05, [])
    assert seen["order"] == ["push_blade_0", "push_tip_h", "push_blade_lean8"] and seen["verdicts"][0].startswith("knuckle") and seen["verdicts"][1] == ""
    monkeypatch.setenv("ISPATIALGYM_HAND_OVERLAP_RANK", "1")
    IS.IORSSmall._touching_push_choice(me, None, pose, th.zeros(3), (0.0, 0.1), th.tensor([0.0, 1.0, 0.0]), 0.05, [])
    assert seen["order"] == ["push_tip_h", "push_blade_0", "push_blade_lean8"] and seen["verdicts"] == ["", "", ""]
    assert seen["hints"] == [None, over["push_blade_0"], over["push_blade_lean8"]]
    # the validator: screened entries validated after the live ones, counted
    es = [{"yaw_deg": 0.0, "class": "a", "sampler_screen": {"hand_overlap": "knuckle -15 mm"}}, {"yaw_deg": 0.0, "class": "b"}]
    ran = []

    class World:
        def settle(self, obj, geom, q):
            return {}, None

        def plan_entry(self, obj, geom, e, T, arm):
            ran.append(e["class"])
            return None

    lib = {"kind": "push", "object": {"rest": {"quat_xyzw": [0.0, 0.0, 0.0, 1.0]}}, "entries": es}
    VC.validate_library(World(), lib, None, None, np.zeros((1, 3)), VC.ContactValidateConfig(), "right")
    assert ran == ["b"] and es[0]["validation"]["verdict"] == "screened"
    ran.clear()
    VC.validate_library(World(), lib, None, None, np.zeros((1, 3)), VC.ContactValidateConfig(screen_as_rank=True), "right")
    assert ran == ["b", "a"] and es[0]["validation"]["verdict"] == "invalid"
