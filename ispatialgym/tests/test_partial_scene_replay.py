"""2026-11-01 train-fast (PARTIAL_SCENE): a demo collected in the sampler's cropped world, replayed in the full variant scene - the
collector's world choice (``run_isg.scene_crop_of`` / ``build_env_config`` / ``template_key`` / ``_scene_leaves``), the recorder's hdf5
attrs (``recorder.scene_crop_attrs``) and the replayer's ``--scene`` decision and merged scene file
(``multienv.replay.merge_partial_scene_file``). Simulator-free."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from ispatialgym.multienv import replay as RP
from ispatialgym.oracle.recorder import scene_crop_attrs

CROP = {"mode": "rooms", "radius_m": None, "center_object": None, "n_loaded_json": 4, "n_dropped": 2, "dropped": ["E", "F"],
        "loaded_rooms": ["kitchen_0"], "loaded_scene_file": "template-partial_rooms.json"}


def _obj(name, x, cls="DatasetObject", module="omnigibson.objects.dataset_object", rooms=("kitchen_0",)):
    return {"class_module": module, "class_name": cls, "args": {"name": name, "category": "thing", "model": "m", "in_rooms": list(rooms)}}


def _state(x):
    return {"root_link": {"pos": [float(x), 0.0, 0.0], "ori": [0.0, 0.0, 0.0, 1.0]}, "joints": {}, "non_kin": {}}


def _scene(objs, robot=None, pos=(0.0, 0.0, 0.0), systems=None, tag="full"):
    init_info = {n: _obj(n, x) for n, x in objs.items()}
    reg = {n: _state(x) for n, x in objs.items()}
    if robot is not None:
        rname, rx, rcls, rmod = robot
        init_info[rname] = _obj(rname, rx, cls=rcls, module=rmod)
        reg[rname] = _state(rx)
    return {"versions": {"omnigibson": {"version": tag}}, "metadata": {"task": {"tag": tag}},
            "init_info": {"class_name": "InteractiveTraversableScene", "args": {"scene_model": "house", "tag": tag}},
            "objects_info": {"init_info": init_info},
            "state": {"pos": list(pos), "ori": [0.0, 0.0, 0.0, 1.0], "registry": {"object_registry": reg, "system_registry": dict(systems or {})}}}


def _full():
    # the template's robot is merge_scene_files' generic Robot entry
    return _scene({n: i for i, n in enumerate("ABCDEF")}, robot=("robot_r1", 0.0, "Robot", "omnigibson.robots.robot"), systems={"water": {"n": 0}})


def _recorded():
    return _scene({"A": 10.0, "B": 11.0}, robot=("robot_r1", 12.0, "R1Pro", "omnigibson.robots.r1pro"), systems={"dust": {"n": 1}}, tag="recorded")


def _x(scene_file, name):
    return scene_file["state"]["registry"]["object_registry"][name]["root_link"]["pos"][0]


# -------------------------------------------------------------------------------------------------------------- merged scene file


def test_merge_restricted_to_the_loaded_objects():
    full, rec = _full(), _recorded()
    full0, rec0 = copy.deepcopy(full), copy.deepcopy(rec)
    loaded = list("ABCDE") + ["robot_r1"]  # F lies outside the full scene's rooms
    m = RP.merge_partial_scene_file(full, rec, loaded)
    names = set(m["objects_info"]["init_info"])
    assert names == set("ABCDE") | {"robot_r1"}
    assert set(m["state"]["registry"]["object_registry"]) == names
    # A, B and the robot as recorded; C..E as in the template
    assert (_x(m, "A"), _x(m, "B"), _x(m, "robot_r1")) == (10.0, 11.0, 12.0)
    assert [_x(m, n) for n in "CDE"] == [2.0, 3.0, 4.0]
    assert m["objects_info"]["init_info"]["robot_r1"]["class_name"] == "R1Pro"
    assert m["metadata"] == rec["metadata"] and m["versions"] == rec["versions"]
    assert m["init_info"] == full["init_info"]  # the replay scene is the full one
    assert m["state"]["registry"]["system_registry"] == {"water": {"n": 0}, "dust": {"n": 1}}
    # inputs untouched, the result shares nothing with them (restore(update_initial_file=True) edits it)
    assert full == full0 and rec == rec0
    m["objects_info"]["init_info"]["C"]["args"]["name"] = "changed"
    m["state"]["registry"]["object_registry"]["A"]["root_link"]["pos"][0] = -1.0
    assert full == full0 and rec == rec0


def test_merge_without_loaded_names_is_the_env_base():
    m = RP.merge_partial_scene_file(_full(), _recorded())
    assert set(m["objects_info"]["init_info"]) == set("ABCDEF") | {"robot_r1"}
    assert m["objects_info"]["init_info"]["robot_r1"]["class_module"] == "omnigibson.robots.r1pro"  # the template's Robot entry dropped
    assert _x(m, "robot_r1") == 12.0 and _x(m, "F") == 5.0


def test_merge_fails_loudly_on_a_recorded_object_the_full_scene_lacks():
    with pytest.raises(ValueError, match=r"not in the full replay scene.*'B'"):
        RP.merge_partial_scene_file(_full(), _recorded(), ["A", "C", "D", "robot_r1"])


def test_merge_requires_one_scene_frame():
    rec = _recorded()
    rec["state"]["pos"] = [5.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="different scene poses"):
        RP.merge_partial_scene_file(_full(), rec, None)

    def to_frame(sf):  # Scene._scene_file_in_scene_frame's contract: None when saved at the origin
        if sf["state"]["pos"] == [0.0, 0.0, 0.0]:
            return None
        out = copy.deepcopy(sf)
        dx = out["state"]["pos"][0]
        for s in out["state"]["registry"]["object_registry"].values():
            s["root_link"]["pos"][0] -= dx
        out["state"]["pos"] = [0.0, 0.0, 0.0]
        return out

    m = RP.merge_partial_scene_file(_full(), rec, None, to_scene_frame=to_frame)
    assert m["state"]["pos"] == [0.0, 0.0, 0.0] and _x(m, "A") == 5.0 and _x(m, "C") == 2.0


#: the OmniGibson the train-fast jobs run (the test env's own may predate the recorded non-kin schema helpers)
JOB_DATA_WRAPPER = "/home/minyeongk/Desktop/projects/Behavior-1K-train-fast-6a6194600/OmniGibson/omnigibson/envs/data_wrapper.py"


def _job_data_wrapper_helpers():
    """``_add_recorded_non_kin_states_to_scene_file`` of the job's OmniGibson (pure python: its source, executed alone), else of the
    installed one; skipped when neither has it."""
    import ast

    for path in (os.environ.get("ISPATIALGYM_JOB_DATA_WRAPPER") or JOB_DATA_WRAPPER,):
        if os.path.exists(path):
            tree = ast.parse(open(path).read())
            fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_add_recorded_non_kin_states_to_scene_file"]
            if fn:
                ns: dict = {}
                exec(compile(ast.Module(body=fn, type_ignores=[]), path, "exec"), ns)
                return SimpleNamespace(_add_recorded_non_kin_states_to_scene_file=ns[fn[0].name])
    dw = pytest.importorskip("omnigibson.envs.data_wrapper")
    if not hasattr(dw, "_add_recorded_non_kin_states_to_scene_file"):
        pytest.skip("no _add_recorded_non_kin_states_to_scene_file in this OmniGibson")
    return dw


def test_merged_base_takes_the_recorded_non_kin_schema():
    """``env_config``'s ``_add_recorded_non_kin_states_to_scene_file`` on the merged base: the recorded objects get the recorded
    schema, the cropped-away ones none (they are never deserialized from a recorded row)."""
    data_wrapper = _job_data_wrapper_helpers()
    rec = _recorded()
    rec["state"]["registry"]["object_registry"]["A"]["non_kin"] = {"ToggledOn": {"value": False}, "Open": {"value": True}}
    m = RP.merge_partial_scene_file(_full(), rec)
    data_wrapper._add_recorded_non_kin_states_to_scene_file(scene_file=m, recorded_scene_files=[rec])
    ii = m["objects_info"]["init_info"]
    assert ii["A"]["args"]["recorded_non_kin_state_names"] == ["ToggledOn", "Open"]
    assert "recorded_non_kin_state_names" not in ii["C"]["args"]
    rec["objects_info"]["init_info"]["G"] = _obj("G", 0)
    rec["state"]["registry"]["object_registry"]["G"] = _state(0)
    with pytest.raises(ValueError, match="'G'"):
        data_wrapper._add_recorded_non_kin_states_to_scene_file(scene_file=RP.merge_partial_scene_file(_full(), _recorded()), recorded_scene_files=[rec])


# -------------------------------------------------------------------------------------------------------------- --scene decision


@pytest.mark.parametrize("requested,crop,expected", [
    ("auto", None, "recorded"),
    ("auto", {"mode": "full"}, "recorded"),
    ("auto", {"mode": "rooms"}, "full"),
    ("auto", {"mode": "region", "radius_m": 3.0}, "full"),
    ("recorded", {"mode": "rooms"}, "recorded"),
    ("full", None, "full"),
])
def test_scene_mode(requested, crop, expected):
    assert RP.scene_mode(requested, crop) == expected


def test_scene_mode_rejects_unknown():
    with pytest.raises(ValueError):
        RP.scene_mode("partial", None)


# -------------------------------------------------------------------------------------------------------------- recorder attrs


def test_scene_crop_attrs(tmp_path):
    assert scene_crop_attrs(None, None) == {}
    assert scene_crop_attrs({"mode": "full"}, str(tmp_path / "template.json"), ["a"]) == {}
    full = tmp_path / "template.json"
    full.write_text(json.dumps(_full()))
    a = scene_crop_attrs(CROP, str(full), ["kitchen_0", "living_room_0"])
    assert json.loads(a["isg_scene_crop"]) == CROP
    assert json.loads(a["isg_full_scene_file"]) == str(full)
    assert json.loads(a["isg_full_load_room_instances"]) == ["kitchen_0", "living_room_0"]
    assert json.loads(a["isg_full_scene_sha256"]) == hashlib.sha256(full.read_bytes()).hexdigest()
    b = scene_crop_attrs(CROP, str(tmp_path / "missing.json"), None)
    assert "isg_full_scene_sha256" not in b and json.loads(b["isg_full_load_room_instances"]) is None


def test_recorder_writes_the_crop_attrs_only_when_cropped(tmp_path):
    h5py = pytest.importorskip("h5py")
    from ispatialgym.tests.test_multienv import _run

    full = tmp_path / "template.json"
    full.write_text("{}")

    def crop(rec):
        rec.scene_crop, rec.full_scene_file, rec.full_load_room_instances = dict(CROP), str(full), ["kitchen_0", "living_room_0"]

    r1, _, p1 = _run(tmp_path / "crop", per_scene=False, async_frames=False, capture=crop)
    r0, _, p0 = _run(tmp_path / "plain", per_scene=False, async_frames=False)
    assert r1["success"] and r0["success"] and r1["n_steps"] == r0["n_steps"]
    with h5py.File(p1["hdf5"], "r") as f1, h5py.File(p0["hdf5"], "r") as f0:
        assert json.loads(f1["data"].attrs["isg_scene_crop"]) == CROP
        assert json.loads(f1["data"].attrs["isg_full_scene_file"]) == str(full)
        assert not any(k.startswith("isg_full") or k == "isg_scene_crop" for k in f0["data"].attrs)
        # the per-step states are the same format
        assert np.array_equal(f1["data/demo_0/state"][()], f0["data/demo_0/state"][()])


# -------------------------------------------------------------------------------------------------------------- the replayer's Demo


def _write_demo(d, attrs=None, tro_path=None):
    h5py = pytest.importorskip("h5py")
    os.makedirs(d, exist_ok=True)
    with h5py.File(os.path.join(d, "demo.hdf5"), "w") as f:
        data = f.create_group("data")
        data.attrs["config"] = json.dumps({"scene": {"load_room_instances": ["kitchen_0"]}, "task": {"tro_path": tro_path}})
        data.attrs["scene_file"] = json.dumps(_recorded())
        for k, v in (attrs or {}).items():
            data.attrs[k] = v
        g = data.create_group("demo_0")
        g["state"] = np.zeros((3, 4), dtype=np.float32)
        g["state_size"] = np.full(3, 4)
        g["action"] = np.zeros((3, 23), dtype=np.float32)
        for k in ("reward", "terminated", "truncated"):
            g[k] = np.zeros(3)
    return d


def test_demo_reads_the_crop_and_finds_the_full_template(tmp_path):
    tdir = tmp_path / "template"
    (tdir / "instances").mkdir(parents=True)
    (tdir / "template.json").write_text(json.dumps(_full()))
    tro = str(tdir / "instances" / "i0-tro_state.json")
    attrs = scene_crop_attrs(CROP, str(tdir / "template.json"), ["kitchen_0", "living_room_0"])
    d = RP.Demo(_write_demo(str(tmp_path / "demo_crop"), attrs, tro))
    assert RP.is_cropped(d.scene_crop) and RP.scene_mode("auto", d.scene_crop) == "full"
    assert d.full_rooms_known and d.full_rooms == ["kitchen_0", "living_room_0"]
    assert RP.full_scene_path(d) == str(tdir / "template.json")
    assert set(RP.load_full_scene(d)["objects_info"]["init_info"]) == set("ABCDEF") | {"robot_r1"}
    # moved elsewhere: the recorded absolute path is gone, the config's tro_path still names the template
    d.full_scene_file = "/nonexistent/template.json"
    assert RP.full_scene_path(d) == str(tdir / "template.json")
    # a demo collected in the full world: no crop attrs, auto = recorded (today)
    p = RP.Demo(_write_demo(str(tmp_path / "demo_plain"), None, tro))
    assert p.scene_crop is None and not p.full_rooms_known and RP.scene_mode("auto", p.scene_crop) == "recorded"
    q = RP.Demo(_write_demo(str(tmp_path / "demo_none"), None, None))
    with pytest.raises(SystemExit, match="no full template"):
        RP.full_scene_path(q)


# -------------------------------------------------------------------------------------------------------------- the collector


def test_collector_world_choice(tmp_path, monkeypatch):
    from ispatialgym.oracle import run_isg as R

    tdir = tmp_path / "tpl"
    (tdir / "activity_definitions").mkdir(parents=True)
    (tdir / "template.json").write_text(json.dumps(_full()))
    (tdir / "template-partial_rooms.json").write_text(json.dumps(_scene({"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0})))
    meta = {"problem": "act-0", "scene": "house", "template_id": "t", "load_room_instances": ["kitchen_0", "living_room_0"]}
    cropped = {**meta, "scene_crop": dict(CROP)}
    monkeypatch.delenv("ISPATIALGYM_FAST", raising=False)
    monkeypatch.delenv("ISPATIALGYM_FAST_PARTIAL_SCENE", raising=False)
    assert R.scene_crop_of(meta) is None and R.scene_crop_of({**meta, "scene_crop": {"mode": "full"}}) is None
    assert R.scene_crop_of(cropped) == CROP
    tro = SimpleNamespace(robot_poses={})
    cfg = R.build_env_config(str(tdir), cropped, tro, "x-tro_state.json", "act", 0, "robot_r1", 64, scene_crop=R.scene_crop_of(cropped))
    assert cfg["scene"]["scene_file"] == str(tdir / "template-partial_rooms.json") and cfg["scene"]["load_room_instances"] == ["kitchen_0"]
    cfg0 = R.build_env_config(str(tdir), cropped, tro, "x-tro_state.json", "act", 0, "robot_r1", 64)
    assert cfg0["scene"]["scene_file"] == str(tdir / "template.json") and cfg0["scene"]["load_room_instances"] == meta["load_room_instances"]
    assert R.loaded_scene_path(str(tdir), cropped).endswith("template-partial_rooms.json")
    # a cropped copy never shares a prebuilt world with an uncropped one; an uncropped key is unchanged by the crop code
    k_plain, k_crop = R.template_key(str(tdir), meta), R.template_key(str(tdir), cropped)
    assert k_plain != k_crop and k_crop != R.template_key(str(tdir), {**meta, "scene_crop": {**CROP, "dropped": ["F"]}})
    (tdir / "template_meta.json").write_text(json.dumps(cropped))
    (tdir / "instances").mkdir()
    tro_path = tdir / "instances" / "i0-tro_state.json"
    tro_path.write_text("{}")
    leaves = R._scene_leaves(str(tdir), str(tro_path))
    assert not any("/E/" in k or "/F/" in k for k in leaves) and any("/D/" in k for k in leaves)
    with pytest.raises(FileNotFoundError):
        R.build_env_config(str(tdir), cropped, tro, "x", "act", 0, "robot_r1", 64, scene_crop={**CROP, "loaded_scene_file": "nope.json"})
    # the switch off: the full template, as before
    monkeypatch.setenv("ISPATIALGYM_FAST_PARTIAL_SCENE", "0")
    assert R.scene_crop_of(cropped) is None and R.template_key(str(tdir), cropped) == k_plain
    assert R.loaded_scene_path(str(tdir)).endswith("template.json")
