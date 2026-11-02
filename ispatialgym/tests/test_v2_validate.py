"""Simulator-free tests of the reload validation (``validate_instances``) and the v2 renders (``render_instances``):
layout detection (spec §1), the §3.5 tolerance table and state comparison, the report schema, the program-check hook
wiring (fake sampler, ``program_check.run`` monkeypatched), the sidecar checks, the loader-contract report, the goal
materialisation with fake entities, and the contact sheet / top-down overlay / montage compositions.

``pytest ispatialgym/tests/test_v2_validate.py``
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import types

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling import render_instances as R
from ispatialgym.sampling import validate_instances as V
from ispatialgym.sampling.v2 import program_check as PC
from ispatialgym.sampling.v2.stances import GraspRecord, PathRecord, Stance, Transition

DATA = os.path.join(os.path.dirname(__file__), "data", "sidecars")
IORS = "inter_object_relation_satisfaction-ontop_small_scale"
VARIANT = "inter_object_relation_satisfaction-ontop_small_scale-apple-plate-breakfast_table"


def _fixture(subtask: str) -> dict:
    with open(os.path.join(DATA, f"{subtask}.json")) as f:
        return json.load(f)


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f) if not isinstance(data, str) else f.write(data)
    return path


def _v2_template(root, variant=VARIANT, tid="house_single_floor__target-apple-a__partner-plate-b", n=2, sidecar=True, meta_extra=None):
    tdir = os.path.join(root, variant, tid)
    meta = {"variant": variant, "template_id": tid, "subtask": IORS, "scene": "house_single_floor", "sampler_version": "2.0.0-dev", "spec_hash": "abc",
            "problem": "inter_object_relation_satisfaction-20", "rooms": ["dining_room_0"],
            "bindings": [{"role": "target", "name": "apple.n.01_1"}, {"role": "partner", "name": "plate.n.04_1"},
                         {"role": "support", "name": "breakfast_table.n.01_1"}, {"role": "floor", "name": "floor.n.01_1"}]}
    meta.update(meta_extra or {})
    _write(os.path.join(tdir, "template_meta.json"), meta)
    _write(os.path.join(tdir, "template.json"), {"state": {}})
    for i in range(n):
        iid = f"{variant}__{tid}__{i:04d}"
        _write(os.path.join(tdir, "instances", f"{iid}-tro_state.json"), {"ispatialgym": {}})
        if sidecar:
            _write(os.path.join(tdir, "instances", f"{iid}.oracle.json"), {"schema": "isg-oracle-v1"})
        _write(os.path.join(tdir, "instances", f"{iid}.summary.json"), {"instance_id": iid})
    return tdir


def _v1_template(root, problem="inter_object_relation_satisfaction-0", tid="house_single_floor__apple-a"):
    tdir = os.path.join(root, problem, tid)
    _write(os.path.join(tdir, "template_meta.json"), {"problem": problem, "scene": "house_single_floor", "inst_to_name": {}})
    _write(os.path.join(tdir, "instances", f"{problem}__{tid}__0000-tro_state.json"), {"ispatialgym": {}})
    return tdir


# ----------------------------------------------------------------------------------------------------------------
# layout detection (spec §1)
# ----------------------------------------------------------------------------------------------------------------


def test_detect_layout_by_sidecar_and_meta(tmp_path):
    root = str(tmp_path)
    t_sidecar = _v2_template(root, n=1)
    t_meta = _v2_template(root, tid="house_single_floor__target-apple-c__partner-plate-d", n=1, sidecar=False)
    t_meta_keys = _v2_template(root, tid="house_single_floor__target-apple-e__partner-plate-f", n=1, sidecar=False, meta_extra={"sampler_version": None})
    t_v1 = _v1_template(root)
    assert V.detect_layout(t_sidecar) == "v2"
    assert V.detect_layout(t_meta) == "v2"  # sampler_version 2.x
    assert V.detect_layout(t_meta_keys) == "v2"  # subtask + spec_hash keys of the v2 core
    assert V.detect_layout(t_v1) == "v1"
    assert V.detect_layout(str(tmp_path / "nowhere")) == "v1"


def test_list_instances_and_files(tmp_path):
    tdir = _v2_template(str(tmp_path), n=2)
    insts = V.list_instances(tdir)
    assert [i.iid[-4:] for i in insts] == ["0000", "0001"]
    assert all(i.sidecar_path and i.sidecar_path.endswith(V.SIDECAR_SUFFIX) for i in insts)
    assert all(i.summary_path and i.summary_path.endswith(V.SUMMARY_SUFFIX) for i in insts)
    assert len(insts[0].paths) == 3
    os.remove(insts[1].sidecar_path)
    assert V.list_instances(tdir)[1].sidecar_path is None
    assert V.list_instances(str(tmp_path / "empty")) == []


def test_discover_templates_filters(tmp_path):
    root = str(tmp_path)
    a = _v2_template(root, n=1)
    b = _v2_template(root, variant="alignment-axial-keyboard-breakfast_table", tid="house_single_floor__target-keyboard-x", n=1)
    v1 = _v1_template(root)
    # bookkeeping directories are not templates
    os.makedirs(os.path.join(root, "renders"))
    os.makedirs(os.path.join(root, VARIANT, "activity_definitions", "inter_object_relation_satisfaction"))
    _write(os.path.join(root, "scene_variants", "x.json"), {})
    _write(os.path.join(root, "task_custom_lists.json"), {})
    found = V.discover_templates(root)
    assert found == sorted([os.path.abspath(a), os.path.abspath(b), os.path.abspath(v1)], key=lambda p: (os.path.dirname(p), p)) or set(found) == {os.path.abspath(a), os.path.abspath(b), os.path.abspath(v1)}
    assert V.discover_templates(root, variants=[VARIANT]) == [os.path.abspath(a)]
    assert V.discover_templates(root, templates=[os.path.basename(b)]) == [os.path.abspath(b)]
    # a template directory given directly is taken as is, even outside the root
    assert V.discover_templates(str(tmp_path / "nowhere"), templates=[a]) == [os.path.abspath(a)]
    assert V.discover_templates(str(tmp_path / "nowhere")) == []


def test_template_identity_and_support_instances(tmp_path):
    tdir = _v2_template(str(tmp_path), n=1)
    meta = V._read_json(os.path.join(tdir, "template_meta.json"))
    ident = V.template_identity(tdir, meta)
    assert ident["variant"] == VARIANT and ident["template_id"] == os.path.basename(tdir) and ident["subtask"] == IORS
    # variant / template id fall back to the directory names (v1 meta)
    ident1 = V.template_identity(tdir, {"problem": "x-0"})
    assert ident1["variant"] == VARIANT and ident1["template_id"] == os.path.basename(tdir)
    # the supports program_check ignores for the final grasp / contact pose = SamplerV2.support_objects roles
    assert V.support_instances(meta) == ["breakfast_table.n.01_1"]
    assert V.support_instances({"bindings": [{"role": "wall", "name": "wall.n.01_1"}, {"role": "tray_dest", "name": "tray.n.01_2"}, {"role": "target", "name": "x"}]}) == ["wall.n.01_1", "tray.n.01_2"]
    assert V.support_instances({}) == []


# ----------------------------------------------------------------------------------------------------------------
# §3.5 tolerance table and state comparison
# ----------------------------------------------------------------------------------------------------------------


def test_reload_tolerance_table():
    t = V.RELOAD_TOLERANCES
    assert t["robot_base_position_m"] == pytest.approx(0.002) and t["robot_base_orientation_rad"] == pytest.approx(0.02)
    assert t["object_position_m"] == pytest.approx(0.001) and t["object_orientation_rad"] == pytest.approx(0.005)
    assert t["joint_tolerance_fraction"] == pytest.approx(0.5)
    assert V.JOINT_TOLERANCE_FLOOR == {"revolute": 0.05, "prismatic": 0.01}


def test_joint_tolerance_half_of_target_or_floor():
    # plain dict targets: {inst: {joint: (position, tolerance)}}
    targets = {"cabinet.n.01_1": {"j_link_0": (1.2, 0.12)}}
    assert V.joint_tolerance(targets, "cabinet.n.01_1", "j_link_0") == pytest.approx(0.06)
    # a joint without a target uses half the §8 floor of its type
    assert V.joint_tolerance(targets, "cabinet.n.01_1", "j_link_1") == pytest.approx(0.025)
    assert V.joint_tolerance(targets, "drawer.n.01_1", "j_slide", "prismatic") == pytest.approx(0.005)
    assert V.joint_tolerance(None, "x", "y") == pytest.approx(0.025)
    # ArticulationTarget-like objects (``.joints``) and dict entries with a "tolerance" key
    art = types.SimpleNamespace(joints={"j": (0.3, 0.08)})
    assert V.joint_tolerance({"o": art}, "o", "j") == pytest.approx(0.04)
    assert V.joint_tolerance({"o": {"j": {"position": 0.3, "tolerance": 0.2}}}, "o", "j") == pytest.approx(0.1)


def _quat_z(yaw):
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


def test_stored_state_from_tro_pieces():
    object_states = {
        "apple.n.01_1": {"is_asleep": False, "root_link": {"pos": [1.0, 2.0, 0.9], "ori": [0, 0, 0, 1]}, "non_kin": {}},
        # BEHAVIOR's EntityPrim._dump_state: joint_pos in the object's joint order
        "cabinet.n.01_1": {"root_link": {"pos": [0, 0, 0], "ori": [0, 0, 0, 1]}, "joint_pos": [0.4, 0.1, 1.8], "joint_vel": [0.0, 0.0, 0.0]},
        "drawer.n.01_1": {"root_link": {"pos": [0, 0, 0], "ori": [0, 0, 0, 1]}, "joints": {"j_slide": {"pos": 0.05, "vel": 0.0}}},
        "weird": "not a dict",
    }
    robot_poses = {"robot": [{"position": [3.0, 4.0, 0.005], "orientation": _quat_z(1.0)}]}
    robot_state = {"joint_positions": {"torso_joint1": 1.0, "base_footprint_x_joint": 5.0, "right_arm_joint1": -1.3}}
    st = V.stored_state(object_states, robot_poses, robot_state, {"cabinet.n.01_1": ["j_door1", "j_door2", "j_door3"]})
    assert st["robot_base"] == ([3.0, 4.0, 0.005], _quat_z(1.0))
    assert st["robot_joints"] == {"torso_joint1": 1.0, "right_arm_joint1": -1.3}  # base virtual joints excluded
    assert set(st["objects"]) == {"apple.n.01_1", "cabinet.n.01_1", "drawer.n.01_1"}
    assert st["object_joints"] == {"cabinet.n.01_1": {"j_door1": 0.4, "j_door2": 0.1, "j_door3": 1.8}, "drawer.n.01_1": {"j_slide": 0.05}}
    # without the loaded object's joint names the values are indexed
    assert V.stored_state(object_states, None, None)["object_joints"]["cabinet.n.01_1"] == {"joint_0": 0.4, "joint_1": 0.1, "joint_2": 1.8}
    empty = V.stored_state({}, None, None)
    assert empty["robot_base"] is None and empty["robot_joints"] == {}


def test_evaluate_state_tolerances():
    stored = {"robot_base": ([0.0, 0.0, 0.0], _quat_z(0.0)), "robot_joints": {"a": 0.0, "b": 1.0},
              "objects": {"apple.n.01_1": ([1.0, 1.0, 0.8], _quat_z(0.0)), "plate.n.04_1": ([1.2, 1.0, 0.8], _quat_z(0.0))},
              "object_joints": {"cabinet.n.01_1": {"j0": 0.5, "j1": 0.0}}}
    # everything inside the tolerances
    cur = copy.deepcopy(stored)
    cur["robot_base"] = ([0.0015, 0.0, 0.0], _quat_z(0.01))
    cur["objects"]["apple.n.01_1"] = ([1.0008, 1.0, 0.8], _quat_z(0.004))
    cur["robot_joints"]["a"] = 0.02
    cur["object_joints"]["cabinet.n.01_1"]["j0"] = 0.55
    checks, failures = V.evaluate_state(stored, cur, {("cabinet.n.01_1", "j0"): 0.06})
    assert failures == []
    assert checks["robot_base"]["ok"] and checks["robot_base"]["pos_err_m"] == pytest.approx(0.0015)
    assert checks["robot_joints"]["ok"] and checks["robot_joints"]["worst_joint"] == "a"
    assert checks["objects"]["apple.n.01_1"]["ok"] and checks["object_joints"]["cabinet.n.01_1"]["j0"]["ok"]
    # each criterion fails on its own
    bad = copy.deepcopy(stored)
    bad["robot_base"] = ([0.0025, 0.0, 0.0], _quat_z(0.0))
    bad["objects"]["plate.n.04_1"] = ([1.2, 1.0, 0.8], _quat_z(0.006))
    bad["robot_joints"]["b"] = 1.03
    bad["object_joints"]["cabinet.n.01_1"]["j1"] = 0.03  # floor: half of 0.05 rad
    bad["object_joints"]["cabinet.n.01_1"]["j0"] = 0.55  # inside its own half tolerance
    del bad["objects"]["apple.n.01_1"]
    checks, failures = V.evaluate_state(stored, bad, {("cabinet.n.01_1", "j0"): 0.06})
    assert not checks["robot_base"]["ok"] and not checks["objects"]["plate.n.04_1"]["ok"] and checks["objects"]["apple.n.01_1"]["missing"]
    assert checks["robot_joints"]["failed"] == ["b"] and checks["object_joints"]["cabinet.n.01_1"]["j0"]["ok"]
    assert not checks["object_joints"]["cabinet.n.01_1"]["j1"]["ok"]
    kinds = " ".join(failures)
    assert "robot base moved" in kinds and "plate.n.04_1 moved" in kinds and "apple.n.01_1: not in the loaded scope" in kinds
    assert "robot joints off" in kinds and "cabinet.n.01_1.j1 off" in kinds and len(failures) == 5


def test_pose_deviation():
    dp, da = V.pose_deviation([0, 0, 0], _quat_z(0.0), [0.003, 0.004, 0.0], _quat_z(0.1))
    assert dp == pytest.approx(0.005) and da == pytest.approx(0.1, abs=1e-5)


def test_settle_drift():
    loaded = {"robot_base": ([0, 0, 0], _quat_z(0)), "objects": {"a": ([1, 1, 1], _quat_z(0)), "gone": ([0, 0, 0], _quat_z(0))}}
    settled = {"robot_base": ([0, 0, -0.001], _quat_z(0)), "objects": {"a": ([1, 1, 1], _quat_z(0.02))}}
    d = V.settle_drift(loaded, settled)
    assert d["robot_base"]["pos_m"] == pytest.approx(0.001) and d["objects"]["a"]["ang_rad"] == pytest.approx(0.02, abs=1e-5) and "gone" not in d["objects"]
    assert V.settle_drift({}, {}) == {"robot_base": None, "objects": {}}


# ----------------------------------------------------------------------------------------------------------------
# report schema
# ----------------------------------------------------------------------------------------------------------------


def test_report_schema_and_counts(tmp_path):
    tdir = _v2_template(str(tmp_path), n=3)
    meta = V._read_json(os.path.join(tdir, "template_meta.json"))
    rep = V.new_template_report(tdir, meta, "v2", 60)
    for key in ("schema", "layout", "variant", "template_id", "subtask", "scene", "settle_steps", "goal_settle_steps", "tolerances", "joint_tolerance_floor",
                "loader", "instances", "n_instances", "n_ok", "reload_failed", "n_error", "deleted", "error", "started", "finished", "time_s"):
        assert key in rep
    assert rep["schema"] == V.REPORT_SCHEMA == "isg-validation-v1" and rep["tolerances"] == V.RELOAD_TOLERANCES and rep["settle_steps"] == 60
    iids = [i.iid for i in V.list_instances(tdir)]
    ok = V.new_instance_record(iids[0])
    assert set(ok) >= {"iid", "ok", "status", "failures", "warnings", "checks", "contact_sheet", "time_s"}
    V.finish_instance_record(ok)
    bad = V.new_instance_record(iids[1])
    bad["failures"].append("robot base moved 3.0 mm")
    V.finish_instance_record(bad)
    err = V.new_instance_record(iids[2])
    err["status"] = "error"
    err["error"] = "RuntimeError('boom')"
    V.finish_instance_record(err)
    assert (ok["status"], bad["status"], err["status"]) == ("ok", "reload_failed", "error") and ok["ok"] and not bad["ok"] and not err["ok"]
    for r in (ok, bad, err):
        rep["instances"][r["iid"]] = r
    V.finalize_template_report(rep)
    assert (rep["n_instances"], rep["n_ok"], rep["reload_failed"], rep["n_error"]) == (3, 1, 1, 1)
    assert V.failed_instance_ids(rep) == [iids[1]]  # errors are not deleted
    line = V.summary_line(rep)
    assert line.startswith("VALIDATION_TEMPLATE ")
    body = json.loads(line[len("VALIDATION_TEMPLATE "):])
    assert body["variant"] == VARIANT and body["reload_failed"] == 1 and body["n_error"] == 1 and set(body["fails"]) == {iids[1], iids[2]}
    # the report is JSON-serialisable
    json.dumps(rep, default=str)


def test_merge_per_instance_reports(tmp_path):
    tdir = _v2_template(str(tmp_path), n=2)
    meta = V._read_json(os.path.join(tdir, "template_meta.json"))
    base = V.new_template_report(tdir, meta, "v2")
    iids = [i.iid for i in V.list_instances(tdir)]
    for i, iid in enumerate(iids):
        part = V.new_template_report(tdir, meta, "v2")
        rec = V.new_instance_record(iid)
        if i:
            rec["failures"].append("x")
        part["instances"][iid] = V.finish_instance_record(rec)
        part["loader"] = {"scene_load_s": 12.0}
        V.finalize_template_report(part)
        V.merge_template_reports(base, part)
    assert set(base["instances"]) == set(iids) and base["n_ok"] == 1 and base["reload_failed"] == 1 and base["loader"]["scene_load_s"] == 12.0


def test_delete_failed_instances(tmp_path):
    tdir = _v2_template(str(tmp_path), n=2)
    insts = V.list_instances(tdir)
    removed = V.delete_failed_instances(tdir, [insts[0].iid])
    assert len(removed) == 3 and not any(os.path.exists(p) for p in insts[0].paths)
    assert all(os.path.exists(p) for p in insts[1].paths)
    assert [i.iid for i in V.list_instances(tdir)] == [insts[1].iid]
    assert V.delete_failed_instances(tdir, ["does-not-exist"]) == []


# ----------------------------------------------------------------------------------------------------------------
# sidecar -> program check wiring
# ----------------------------------------------------------------------------------------------------------------


def test_draft_from_sidecar_rebuilds_typed_records():
    d = _fixture(IORS)
    draft = V.draft_from_sidecar(d)
    assert draft.to_dict() == d and draft.to_dict() is not d  # the written dict itself, copied
    assert set(draft.stances) == {"init", "manipulation"}
    st = draft.stance("manipulation")
    assert isinstance(st, Stance) and st.translate == pytest.approx(0.75) and st.tilt == pytest.approx(-0.42) and list(st.base_pose2d) == pytest.approx([7.2, 2.5545, 1.5708])
    assert st.arms == ["right"] and draft.stance("init").arms == "untucked"
    assert len(draft.transitions) == 1 and isinstance(draft.transitions[0], Transition) and draft.transitions[0].to == "manipulation"
    g = draft.grasp_of("apple.n.01_1")
    assert isinstance(g, GraspRecord) and g.arm == "right" and g.place_on == "plate.n.04_1" and g.carry_height == pytest.approx(0.14)
    assert g.eef_in_object is None and g.grasp[0].tolist() == pytest.approx([7.2, 3.12, 0.863])
    assert draft.grasp_of("nothing") is None and draft.language["objects"]["apple.n.01_1"]["display_name"] == "apple"
    with pytest.raises(KeyError):
        draft.stance("placement")


def test_draft_from_sidecar_paths_contacts_and_extras():
    cam = V.draft_from_sidecar(_fixture("camera_relocalization-standard"))
    assert len(cam.paths) == 1 and isinstance(cam.paths[0], PathRecord) and cam.paths[0].to == "goal" and cam.goal["translate"] == pytest.approx(0.8)
    axial = V.draft_from_sidecar(_fixture("alignment-axial"))
    assert set(axial.contacts) == {"push", "brace"} and axial.contacts["push"].kind == "push" and axial.contacts["brace"].brace_kind
    assert axial.contacts["push"].goal is not None and axial.contacts["brace"].contact is not None
    blocks = V.draft_from_sidecar(_fixture("configuration_matching-block_design"))
    assert len(blocks.blocks) >= 1 and blocks.to_dict()["blocks"] == _fixture("configuration_matching-block_design")["blocks"]
    art = V.draft_from_sidecar(_fixture("configuration_matching-articulation_open_large_scale"))
    assert art.articulation and "waypoints" in art.articulation
    # every fixture round-trips through the draft without loss
    for name in sorted(os.listdir(DATA)):
        d = _fixture(name[:-5])
        assert V.draft_from_sidecar(d).to_dict() == d


class _FakeKit:
    pass


def test_run_program_check_calls_program_check_run(monkeypatch):
    d = _fixture(IORS)
    calls = []

    def fake_run(variant, sidecar, sampler, subtask=None):
        calls.append((variant, sidecar, sampler, subtask))
        # the hook hands program_check exactly what it reads from the sampler (its docstring)
        assert sampler.ikc is ikc and sampler.kit is kit and sampler.subtask == IORS
        assert sampler.objects == objects and sampler.support_objects() == ["TABLE"]
        assert sidecar.to_dict() == d and sidecar.stance("init").translate == pytest.approx(0.5)
        return {"mode": "expand", "actions": [{"name": "transition", "stance": "manipulation", "carried": None, "n_poses": 3},
                                              {"name": "grasp", "stance": "manipulation", "carried": None, "n_poses": 2}]}

    monkeypatch.setattr(PC, "run", fake_run)
    kit, ikc = _FakeKit(), object()
    objects = {"apple.n.01_1": "APPLE", "plate.n.04_1": "PLATE", "breakfast_table.n.01_1": "TABLE"}
    sampler = V.ValidationSampler(kit, ikc, objects, [objects["breakfast_table.n.01_1"]], IORS)
    out = V.run_program_check(VARIANT, V.draft_from_sidecar(d), sampler, IORS)
    assert len(calls) == 1 and calls[0][0] == VARIANT and calls[0][3] == IORS
    assert out["ok"] and out["mode"] == "expand" and [a["name"] for a in out["actions"]] == ["transition", "grasp"] and out["failed_action"] is None
    assert out["time_s"] is not None


def test_run_program_check_reports_failing_action(monkeypatch):
    class FakeReject(Exception):
        def __init__(self, reason, message, **details):
            super().__init__(f"[{reason}] {message}")
            self.reason = reason
            self.details = details

    def failing(variant, sidecar, sampler, subtask=None):
        raise FakeReject("program_check_failed", "grasp: exact IK failed (stance manipulation, arm right)", action="grasp")

    monkeypatch.setattr(PC, "run", failing)
    sampler = V.ValidationSampler(_FakeKit(), object(), {}, [], IORS)
    out = V.run_program_check(VARIANT, V.draft_from_sidecar(_fixture(IORS)), sampler, IORS)
    assert not out["ok"] and out["failed_action"] == "grasp" and "exact IK failed" in out["reason"] and out["actions"] == []

    def crashing(variant, sidecar, sampler, subtask=None):
        raise RuntimeError("cuRobo exploded")

    monkeypatch.setattr(PC, "run", crashing)
    out = V.run_program_check(VARIANT, V.draft_from_sidecar(_fixture(IORS)), sampler, IORS)
    assert not out["ok"] and out["failed_action"] == "run" and "cuRobo exploded" in out["reason"]


def test_program_arms_from_sidecars():
    arms = V._program_arms([_fixture(IORS), _fixture("alignment-axial")])
    assert arms == ["left", "right"]
    assert V._program_arms([{"variant": "nonsense"}]) == ["right"]


def test_check_sidecar_validates_and_verifies_the_tro(tmp_path):
    d = _fixture(IORS)
    tro = tmp_path / "x-tro_state.json"
    tro.write_text('{"ispatialgym": {}}')
    d["tro_sha256"] = hashlib.sha256(tro.read_bytes()).hexdigest()
    path = _write(str(tmp_path / "x.oracle.json"), d)
    rec, raw = V.check_sidecar(path, str(tro))
    assert rec["ok"] and rec["loaded"] and rec["valid"] and rec["tro_sha256_ok"] and rec["problems"] == []
    assert rec["subtask"] == IORS and rec["variant"] == VARIANT and raw == d
    # sha mismatch
    tro.write_text('{"ispatialgym": {"changed": 1}}')
    rec, raw = V.check_sidecar(path, str(tro))
    assert not rec["ok"] and rec["valid"] and not rec["tro_sha256_ok"] and any("tro_sha256 mismatch" in p for p in rec["problems"]) and raw is not None
    # a sidecar the programs cannot expand (required block missing) is invalid
    bad = copy.deepcopy(d)
    del bad["grasps"]
    path2 = _write(str(tmp_path / "bad.oracle.json"), bad)
    rec, raw = V.check_sidecar(path2, str(tro))
    assert rec["loaded"] and not rec["valid"] and not rec["ok"] and any("grasps" in p for p in rec["problems"])
    # unreadable
    path3 = _write(str(tmp_path / "broken.oracle.json"), "{not json")
    rec, raw = V.check_sidecar(path3, str(tro))
    assert not rec["loaded"] and raw is None and rec["problems"]


def test_loader_contract_report():
    extras = {"constraints": [{"child": "hardback.n.01_1", "parent": "breakfast_table.n.01_1"}],
              "grasps": [{"arm": "right", "object": "cap.n.02_1"}, {"arm": "left", "object": "mug.n.04_1"}],
              "attachment_targets": [{"child": "a", "parent": "b"}]}
    loader_report = {"constraints": [{"child": "hardback.n.01_1", "prim_path": "/x/isg_hinge"}],
                     "grasps": [{"arm": "right", "object": "cap.n.02_1", "attached": True, "attached_after_reset": True, "object_in_eef_dev_mm": 0.4}],
                     "attachments": [{"child": "a", "parent": "b", "collision_filtered": True}],
                     "after_reset": {"hinges_recreated": 1, "grasps": [], "articulation_reset": ["cabinet.n.01_1"]}}
    rec, failures = V.loader_contract_report(extras, loader_report)
    assert rec["constraints"] == {"expected": 1, "restored": 1, "recreated_after_reset": 1, "ok": True}
    assert rec["attachments"]["ok"] and rec["articulation_reset"] == ["cabinet.n.01_1"]
    assert [g["ok"] for g in rec["grasps"]] == [True, False] and failures == ["grasp of mug.n.04_1 (left) was not restored"] and not rec["ok"]
    rec, failures = V.loader_contract_report({"constraints": extras["constraints"]}, {"constraints": [], "after_reset": {}})
    assert failures == ["loader restored 0 of 1 constraints"]
    rec, failures = V.loader_contract_report({"grasps": extras["grasps"][:1]}, {"grasps": [{"arm": "right", "object": "cap.n.02_1", "attached": True, "attached_after_reset": False}]})
    assert failures == ["grasp of cap.n.02_1 (right) not attached"]
    rec, failures = V.loader_contract_report({}, {})
    assert rec["ok"] and failures == []


# ----------------------------------------------------------------------------------------------------------------
# goal materialisation with fake entities
# ----------------------------------------------------------------------------------------------------------------


class FakeEntity:
    def __init__(self, pos, quat, joints=None):
        self.pos = th.tensor(pos, dtype=th.float32)
        self.quat = th.tensor(quat, dtype=th.float32)
        self.joints = joints or {}
        self.q = th.zeros(len(self.joints))
        self.still = 0
        self.woken = 0

    def get_position_orientation(self):
        return self.pos.clone(), self.quat.clone()

    def set_position_orientation(self, p, q):
        self.pos, self.quat = th.as_tensor(p, dtype=th.float32).clone(), th.as_tensor(q, dtype=th.float32).clone()

    def keep_still(self):
        self.still += 1

    def wake(self):
        self.woken += 1

    def get_joint_positions(self):
        return self.q.clone()

    def set_joint_positions(self, q):
        self.q = th.as_tensor(q, dtype=th.float32).clone()

    def set_joint_velocities(self, v):
        pass


class FakeSim:
    def __init__(self):
        self.physics_steps = 0
        self.full_steps = 0

    def step_physics(self):
        self.physics_steps += 1

    def step(self):
        self.full_steps += 1


def _target(frame="world", position=None, orientation=None, position_metric=None, orientation_metric=None):
    return types.SimpleNamespace(frame=frame, position=position, orientation=orientation, position_metric=position_metric, orientation_metric=orientation_metric)


def _task(satisfied=True):
    compiled = types.SimpleNamespace(binary_goal_head_idxs=[0], check_goal=lambda fn: (satisfied, {"satisfied": [0] if satisfied else [], "unsatisfied": [] if satisfied else [0]}))
    return types.SimpleNamespace(compiled_task=compiled, evaluate_predicate=lambda *a: True, group_entities={}, groups={},
                                 problem=types.SimpleNamespace(goal=[["ontop", "apple.n.01_1", "plate.n.04_1"]], init=[]))


def _tro(pose_targets, extras=None, groups=None, articulation_targets=None):
    meta = types.SimpleNamespace(pose_targets=pose_targets, extras=extras or {}, groups=groups or {}, articulation_targets=articulation_targets or {})
    return types.SimpleNamespace(meta=meta)


def test_materialise_goal_full_world_pose_and_skips():
    apple = FakeEntity([1.0, 1.0, 0.8], _quat_z(0.0))
    cap = FakeEntity([0.5, 0.5, 1.2], _quat_z(0.0))
    robot = FakeEntity([0, 0, 0], _quat_z(0.0))
    physical = {"apple.n.01_1": apple, "cap.n.02_1": cap, "agent.n.01_1": robot}
    targets = {"apple.n.01_1": _target(position=[1.3, 1.0, 0.85], orientation=_quat_z(0.5), position_metric="full", orientation_metric="full"),
               "cap.n.02_1": _target(frame="agent.n.01_1::robot_r1:zed_link:Camera:0", position=[0, 0, -0.35], position_metric="depth")}
    tro = _tro(targets, extras={"grasps": [{"arm": "right", "object": "cap.n.02_1"}]})
    og = types.SimpleNamespace(sim=FakeSim())
    out = V.materialise_goal(None, _task(True), tro, physical, robot, None, og, settle_steps=7)
    assert out["materialised"] and out["moved"] == ["apple.n.01_1"] and out["skipped"] == ["cap.n.02_1: held object"]
    assert apple.pos.tolist() == pytest.approx([1.3, 1.0, 0.85]) and cap.pos.tolist() == pytest.approx([0.5, 0.5, 1.2])
    assert out["binary_goal_after_teleport"] is True and out["unsatisfied"] == []
    assert og.sim.physics_steps == 7 and og.sim.full_steps == 1 and robot.still == 7 and apple.woken == 1
    # unsatisfied after the teleport is reported with the BDDL text
    out = V.materialise_goal(None, _task(False), tro, physical, robot, None, og, settle_steps=1)
    assert out["binary_goal_after_teleport"] is False and out["unsatisfied"] == ["['ontop', 'apple.n.01_1', 'plate.n.04_1']"]


def test_materialise_goal_position_only_target_keeps_orientation(monkeypatch):
    # IORS with orientation_metric None (customs override): the position is reached, the current orientation kept;
    # the robot goes to the sidecar goal stance (the stance the executor captures the goal image from)
    apple = FakeEntity([1.0, 1.0, 0.8], _quat_z(0.7))
    robot = FakeEntity([0, 0, 0], _quat_z(0.0))
    moved_robot = []
    monkeypatch.setattr(V, "_teleport_robot_to_goal", lambda r, block: moved_robot.append(block))
    tro = _tro({"apple.n.01_1": _target(position=[1.3, 1.0, 0.85], position_metric="full")})
    og = types.SimpleNamespace(sim=FakeSim())
    sidecar = {"goal": {"base_pose2d": [7.2, 2.55, 1.57], "translate": 0.75, "tilt": -0.42, "arms": "same"}}
    out = V.materialise_goal(None, _task(), tro, {"apple.n.01_1": apple}, robot, sidecar, og, settle_steps=1)
    assert out["materialised"] and apple.pos.tolist() == pytest.approx([1.3, 1.0, 0.85]) and apple.quat.tolist() == pytest.approx(_quat_z(0.7))
    assert moved_robot == [sidecar["goal"]] and out["robot_view"] == "sidecar goal stance"
    # without a goal block the robot stays where it is
    out = V.materialise_goal(None, _task(), tro, {"apple.n.01_1": apple}, robot, None, og, settle_steps=1)
    assert out["robot_view"] == "init stance" and len(moved_robot) == 1


def test_materialise_goal_nothing_to_move_and_attachments():
    robot = FakeEntity([0, 0, 0], _quat_z(0.0))
    ent = FakeEntity([1, 1, 1], _quat_z(0.0))
    og = types.SimpleNamespace(sim=FakeSim())
    tro = _tro({"x.n.01_1": _target(orientation=_quat_z(0.3), orientation_metric="full")})
    out = V.materialise_goal(None, _task(), tro, {"x.n.01_1": ent}, robot, {"goal": {"base_pose2d": [0, 0, 0], "translate": 0.5, "tilt": 0.0}}, og)
    assert not out["materialised"] and out["binary_goal_after_teleport"] is None and og.sim.physics_steps == 0
    assert out["skipped"] == ["x.n.01_1: frame world, position None, orientation full"] and out["robot_view"] == "init stance"
    tro = _tro({}, extras={"attachment_targets": [{"child": "a", "parent": "b"}]})
    out = V.materialise_goal(None, _task(), tro, {}, robot, None, og)
    assert not out["materialised"] and out["skipped"] == ["attachment needs AttachedTo.set_value"]
    # an agent camera target needs the sidecar goal block ...
    tro = _tro({"agent.n.01_1": _target(position=[0, 0, 1], orientation=_quat_z(0), position_metric="full", orientation_metric="full")})
    out = V.materialise_goal(None, _task(), tro, {"agent.n.01_1": robot}, robot, None, og)
    assert not out["materialised"] and out["skipped"] == ["agent.n.01_1: camera target without a sidecar goal block"]


def test_materialise_goal_camera_relocalization_moves_the_robot(monkeypatch):
    # ... and with it the robot is the thing that moves (camera relocalization)
    robot = FakeEntity([0, 0, 0], _quat_z(0.0))
    calls = []
    monkeypatch.setattr(V, "_teleport_robot_to_goal", lambda r, block: calls.append((r, block)))
    tro = _tro({"agent.n.01_1": _target(position=[0, 0, 1], orientation=_quat_z(0), position_metric="full", orientation_metric="full")})
    goal = {"base_pose2d": [15.6, 1.9, -1.2], "translate": 0.8, "tilt": -0.25, "arms": "untucked"}
    og = types.SimpleNamespace(sim=FakeSim())
    out = V.materialise_goal(None, _task(), tro, {"agent.n.01_1": robot}, robot, {"goal": goal}, og, settle_steps=3)
    assert out["materialised"] and out["moved"] == ["agent.n.01_1 (sidecar goal stance)"] and calls == [(robot, goal)] and og.sim.physics_steps == 3


def test_materialise_goal_rotates_hinged_object_about_its_anchor():
    # alignment: the orientation-only target of a hinged object is reached by rotating about the hinge anchor
    book = FakeEntity([2.0, 1.0, 0.9], _quat_z(0.0))
    robot = FakeEntity([0, 0, 0], _quat_z(0.0))
    anchor = [1.0, 1.0, 0.9]
    tro = _tro({"hardback.n.01_1": _target(orientation=_quat_z(math.pi / 2), orientation_metric="full")},
               extras={"constraints": [{"child": "hardback.n.01_1", "anchor_position": anchor, "axis": [0, 0, 1]}]})
    og = types.SimpleNamespace(sim=FakeSim())
    out = V.materialise_goal(None, _task(), tro, {"hardback.n.01_1": book}, robot, None, og, settle_steps=1)
    assert out["materialised"] and out["moved"] == ["hardback.n.01_1 (about the hinge)"]
    assert book.pos.tolist() == pytest.approx([1.0, 2.0, 0.9], abs=1e-5)  # (1, 0, 0) from the anchor rotated by 90 deg
    assert book.quat.tolist() == pytest.approx(_quat_z(math.pi / 2), abs=1e-6)


def test_materialise_goal_groups_and_articulation():
    a, b = FakeEntity([0, 0, 0], _quat_z(0)), FakeEntity([1, 0, 0], _quat_z(0))
    cab = FakeEntity([3, 3, 0], _quat_z(0), joints={"j_link_0": object(), "j_link_1": object()})
    robot = FakeEntity([0, 0, 0], _quat_z(0.0))
    slot = lambda x: types.SimpleNamespace(position=[x, 5.0, 0.8], orientation=_quat_z(0))  # noqa: E731
    group = types.SimpleNamespace(target_slots={"s0": slot(0.0), "s1": slot(0.1)}, assignment_mode="fixed", member_to_slot={"a.n.01_1": "s1", "a.n.01_2": "s0"})
    art = types.SimpleNamespace(joints={"j_link_1": (1.2, 0.1)})
    tro = _tro({}, groups={"targets_group.n.01_1": group}, articulation_targets={"cabinet.n.01_1": art})
    task = _task()
    task.groups = {"targets_group.n.01_1": ["a.n.01_1", "a.n.01_2"]}
    og = types.SimpleNamespace(sim=FakeSim())
    out = V.materialise_goal(None, task, tro, {"a.n.01_1": a, "a.n.01_2": b, "cabinet.n.01_1": cab}, robot, None, og, settle_steps=2)
    assert out["materialised"] and out["moved"] == ["a.n.01_1", "a.n.01_2", "cabinet.n.01_1 joints"]
    assert a.pos.tolist() == pytest.approx([0.1, 5.0, 0.8]) and b.pos.tolist() == pytest.approx([0.0, 5.0, 0.8])
    assert cab.q.tolist() == pytest.approx([0.0, 1.2])


# ----------------------------------------------------------------------------------------------------------------
# images: contact sheet, top-down overlay, montage
# ----------------------------------------------------------------------------------------------------------------


def test_compose_contact_sheet_shapes():
    init = np.full((1080, 1080, 4), 120, dtype=np.uint8)
    goal = th.rand(1080, 1080, 3)
    sheet = V.compose_contact_sheet(init, goal, "iid [ok]", tile=200)
    assert sheet.shape == (228, 400, 3) and sheet.dtype == np.uint8
    sheet = V.compose_contact_sheet(init, None, "iid [reload_failed] robot base moved", note="cap.n.02_1: held object", tile=100)
    assert sheet.shape == (128, 200, 3)
    # the grey "goal not materialised" panel is distinguishable from the init tile
    assert int(sheet[100, 150].mean()) != int(sheet[100, 50].mean())


def test_topdown_overlay_draws_stances_paths_objects():
    fm = np.zeros((200, 200), dtype=np.uint8)
    fm[60:140, 60:140] = 255  # free floor
    res = 0.1

    def to_map(x, y):  # OmniGibson convention: row from y, col from x, origin at the map centre
        return int(round(y / res + 100)), int(round(x / res + 100))

    d = _fixture(IORS)
    stances, paths = R.sidecar_overlay_facts(d)
    assert set(stances) == {"init", "manipulation"} and paths == []
    stances = {k: [v[0] - 7.2, v[1] - 2.4545, v[2]] for k, v in stances.items()}  # around the origin of the synthetic map
    robot = [-0.6, -0.4, 1.5708]  # off the init stance so that both arrows stay visible
    img = R.topdown_overlay(fm, to_map, stances, [[[0.0, 0.0, 0.0], [0.5, 0.5, 0.3]]], {"apple.n.01_1": [-0.6, 0.65]}, robot, res, window_m=1.5, scale=4, title="t")
    # window_m 1.5 -> 31 cells, grown to cover the anchors (+1 m): 0.5 * 1.1 m + 1 m = 1.55 m -> 33 cells
    assert img.shape == (33 * 4, 33 * 4, 3) and img.dtype == np.uint8
    # a path that leaves the default window widens the crop instead of being cut
    wide = R.topdown_overlay(fm, to_map, stances, [[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]], resolution_m=res, window_m=1.5, scale=1)
    assert wide.shape[1] >= 60
    flat = img.reshape(-1, 3)
    for colour in (R.STANCE_COLOURS["init"], R.STANCE_COLOURS["manipulation"], R.ROBOT_COLOUR, R.OBJECT_COLOUR, R.PATH_COLOUR):
        assert (flat == np.array(colour, dtype=np.uint8)).all(axis=1).any(), colour
    # world +y points up: the object 0.65 m ahead (+y) sits above the robot on the canvas
    ys, xs = np.nonzero((flat == np.array(R.OBJECT_COLOUR, dtype=np.uint8)).all(axis=1).reshape(img.shape[:2]))
    ry, rx = np.nonzero((flat == np.array(R.ROBOT_COLOUR, dtype=np.uint8)).all(axis=1).reshape(img.shape[:2]))
    assert ys.mean() < ry.mean()
    # the goal block becomes the "goal" stance; an empty sidecar gives nothing
    st, _ = R.sidecar_overlay_facts(_fixture("camera_relocalization-standard"))
    assert "goal" in st and R.sidecar_overlay_facts(None) == ({}, [])
    # a crop that falls off the map still renders
    img2 = R.topdown_overlay(fm, to_map, {"init": [-30.0, -30.0, 0.0]}, resolution_m=res, window_m=1.0, scale=2)
    assert img2.ndim == 3


def test_montage_grid():
    a = np.zeros((1080, 1080, 3), dtype=np.uint8)
    rows = [[a, None, a], [a, a, None], [None, None, None]]
    img = R.montage(rows, ["one", "two", "three"], tile=50)
    assert img.shape == (18 + 3 * 68, 150, 3)
    assert R.montage([], [], tile=50).shape == (18, 150, 3)


# ----------------------------------------------------------------------------------------------------------------
# CLI plumbing (no simulator): parser, worker argv, layout dispatch
# ----------------------------------------------------------------------------------------------------------------


def test_cli_parser_and_worker_argv(tmp_path):
    p = V.build_parser()
    args = p.parse_args(["--root", str(tmp_path), "--variant", VARIANT, "--template", "t1", "--delete-failed", "--no-render", "--fresh-process-per-instance"])
    assert args.root == str(tmp_path) and args.variant == [VARIANT] and args.template == ["t1"] and args.delete_failed and args.no_render and args.fresh_process_per_instance
    assert args.settle_steps == V.DEFAULT_SETTLE_STEPS == 60 and not args.worker
    argv = V._worker_argv(args, "/x/variant/t1", "iid-0")
    assert argv[:3] == [os.sys.executable, "-m", "ispatialgym.sampling.validate_instances"]
    assert "--worker" in argv and argv[argv.index("--template") + 1] == "/x/variant/t1" and argv[argv.index("--instance") + 1] == "iid-0" and "--no-render" in argv
    assert "--no-goal" not in argv and argv[argv.index("--root") + 1] == str(tmp_path)
    assert V._worker_report_path("/x/t", None).endswith("validation/report.json") and V._worker_report_path("/x/t", "i").endswith("validation/report-i.json")
    with pytest.raises(SystemExit):
        V.main([])  # neither --root nor --out


def test_main_without_v2_workers_writes_aggregate(tmp_path, monkeypatch):
    """The parent process never imports the simulator: the workers are subprocesses (mocked here)."""
    root = str(tmp_path)
    tdir = _v2_template(root, n=1)
    iid = V.list_instances(tdir)[0].iid

    def fake_worker(argv, timeout):
        # the worker writes <template>/validation/report.json; simulate one failed instance
        meta = V._read_json(os.path.join(tdir, "template_meta.json"))
        rep = V.new_template_report(tdir, meta, "v2")
        rec = V.new_instance_record(iid)
        rec["failures"].append("robot base moved 3.1 mm / 0.0010 rad (tolerance 2 mm / 0.02 rad)")
        rep["instances"][iid] = V.finish_instance_record(rec)
        V.finalize_template_report(rep)
        V._write_json(V._worker_report_path(tdir, None), rep)
        return 0

    monkeypatch.setattr(V, "_run_worker", fake_worker)
    report_path = str(tmp_path / "agg.json")
    V.main(["--root", root, "--delete-failed", "--report", report_path])
    agg = V._read_json(report_path)
    assert agg["reload_failed"] == 1 and agg["n_ok"] == 0 and agg["deleted"] == [iid]
    assert V.list_instances(tdir) == []  # the failed instance was deleted (TRO, sidecar, summary)
    per_template = V._read_json(os.path.join(tdir, "validation", "report.json"))
    assert per_template["deleted"] == [iid] and len(per_template["deleted_paths"]) == 3


def test_exact_checks_thresholds_and_sidecar_comparison():
    """``exact_checks`` (``--exact``): a reload that reproduces the TRO passes; a joint 2e-4 rad off right after the loader, a
    right hand drifting 2 mm in the settle, a sidecar arms_q entry 5e-4 rad from the TRO and a start depth off the loaded
    scene each fail on their own."""
    q_id = [0.0, 0.0, 0.0, 1.0]
    joints = {"torso_joint1": 1.025, **{f"right_arm_joint{k}": 0.1 * k for k in range(1, 8)}, **{f"left_arm_joint{k}": 0.0 for k in range(1, 8)}}
    cam = ([0.0, 0.0, 1.0], [0.0, -0.7071068, 0.0, 0.7071068])  # looks along world +x (its -z)
    obj = ([0.30, 0.0, 1.0], q_id)
    stored = {"robot_base": ([1.0, 2.0, 0.0], q_id), "robot_joints": dict(joints), "objects": {"can.n.01_1": obj}, "object_joints": {}}

    def snap(joint_off=0.0, hand_dx=0.0):
        j = dict(joints)
        j["right_arm_joint5"] += joint_off
        return {"robot_base": stored["robot_base"], "robot_joints": j, "objects": {"can.n.01_1": obj}, "object_joints": {},
                "right_eef": ([0.25 + hand_dx, 0.0, 1.0], q_id), "head_camera": cam, "zed_link": cam}

    sidecar = {"grasps": [{"object": "can.n.01_1", "grasp": {"position": [0.25, 0.0, 1.0], "orientation": q_id}}],
               "stances": {"init": {"base_pose2d": [1.0, 2.0, 0.0], "hold": {"start_depth_m": 0.30},
                                    "arms_q": {"right": [0.1 * k for k in range(1, 8)], "left": [0.0] * 7}}}}
    summary = {"goal": {"init_depth_m": 0.30}}
    ok = V.exact_checks(snap(), snap(), stored, sidecar, summary)
    assert ok["ok"], ok["failures"]
    assert ok["sidecar_vs_instance"]["start_depth_m"]["loaded"] == pytest.approx(0.30, abs=1e-6)
    assert not V.exact_checks(snap(joint_off=2e-4), snap(), stored, sidecar, summary)["ok"]
    assert not V.exact_checks(snap(), snap(hand_dx=0.002), stored, sidecar, summary)["ok"]
    bad = copy.deepcopy(sidecar)
    bad["stances"]["init"]["arms_q"]["right"][4] += 5e-4
    res = V.exact_checks(snap(), snap(), stored, bad, summary)
    assert not res["ok"] and res["sidecar_vs_instance"]["worst_arm_joint"] == "right_arm_joint5"
    far = copy.deepcopy(sidecar)
    far["stances"]["init"]["hold"]["start_depth_m"] = 0.31
    assert not V.exact_checks(snap(), snap(), stored, far, {"goal": {"init_depth_m": 0.31}})["ok"]
    assert V.quat_angle64(q_id, [0.0, 0.0, math.sin(0.0005), math.cos(0.0005)]) == pytest.approx(0.001, rel=1e-6)


def test_validation_sampler_carries_the_variant_generation():
    """The program check reads per-problem witness overrides through ``sampler.gp.gen`` (``'ValidationSampler' object has
    no attribute 'gp'`` failed every held-object reload); unknown variants fall back to the defaults."""
    s = V.ValidationSampler(None, None, {}, [], "object_scaling", {"witness_residual_max_m": 0.004})
    assert s.gp.gen("witness_residual_max_m", 0.002) == 0.004 and s.gp.gen("witness_residual_pre_m", 0.002) == 0.002
    assert s.grasp_ignore_objects() == [] and V.ValidationSampler(None, None, {}, [], "x").gp.gen("k", 1) == 1
    assert V.customs_generation("object_scaling-canned_food")["min_depth_change_m"] == 0.05
    assert V.customs_generation("no-such-variant") == {}
