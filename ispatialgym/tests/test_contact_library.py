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
    e = _tip_entry(glib)
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
    assert calls == [("settle", 0.0), ("settle", 180.0)]  # one settled rest per yaw slot
    v = {e["class"]: e["validation"] for e in lib["entries"] if e["yaw_deg"] == 0.0}
    assert all(x["verdict"] == "valid" for k, x in v.items() if "tip" in k)
    pads = [x for k, x in v.items() if k.startswith("push_h")]
    assert pads and all(x["verdict"] == "invalid" and x["reasons"] == ["no_placement"] for x in pads)
    assert all(x["reasons"] == ["short"] for k, x in v.items() if k.startswith("push_blade"))
    assert tim["kind"] == "push" and tim["rest_id"] == "r0" and tim["n_entries"] == len(lib["entries"]) and set(tim) >= {"settle_s", "plan_s", "trials_s", "total_s"}
    assert lib["validated"]["oracle_config"]["push_pre_dist_m"] == 0.04 and lib["validation_timing"] is tim
