"""Scene variants (``sampling.scene_variants``): the objects a variant takes out of the scene are parked outside it with a
fixed base and listed with their scene-file pose, in the variant and in ``template.json`` (owner 2026-10-11), or deleted
(``mode="remove"``, the reach-band worlds)."""

import copy
import hashlib
import json
import os

import pytest

from ispatialgym.sampling import scene_variants as SV


def _obj(name, category, fixed):
    return {"class_module": "omnigibson.objects.dataset_object", "class_name": "DatasetObject",
            "args": {"name": name, "category": category, "model": "abcdef", "fixed_base": fixed, "in_rooms": ["dining_room_0"]}}


def _root(pos, ori=(0.0, 0.0, 0.0, 1.0)):
    return {"root_link": {"pos": list(pos), "ori": list(ori), "lin_vel": [0.1, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.2]}}


def _scene(tmp_path):
    scene = {"metadata": {"task": {}},
             "objects_info": {"init_info": {"chair_0": _obj("chair_0", "straight_chair", False), "chair_1": _obj("chair_1", "straight_chair", False),
                                            "table_0": _obj("table_0", "breakfast_table", True), "floors_0": _obj("floors_0", "floors", True)}},
             "state": {"registry": {"object_registry": {"chair_0": _root([1.0, 2.0, 0.4], (0.0, 0.0, 0.6, 0.8)), "chair_1": _root([3.0, -4.0, 0.4]),
                                                        "table_0": _root([2.0, 0.0, 0.5]), "floors_0": _root([0.0, 0.0, -0.9])}}}}
    path = tmp_path / "house_stable.json"
    path.write_text(json.dumps(scene))
    return str(path), scene


def test_moved_objects_stay_in_the_file_parked_outside_the_scene(tmp_path):
    path, scene = _scene(tmp_path)
    vpath, v = SV.write_variant(path, ["chair_0", "chair_1", "not_in_the_scene"], str(tmp_path / "out"))
    assert json.load(open(vpath)) == v
    init, reg, meta = v["objects_info"]["init_info"], v["state"]["registry"]["object_registry"], v["metadata"]["ispatialgym"]
    assert meta["removed_objects"] == ["chair_0", "chair_1"] and meta["variant_mode"] == "move" and meta["source_scene_file"] == "house_stable.json"
    moved = meta["moved_objects"]
    assert moved["chair_0"] == {"pos": [1.0, 2.0, 0.4], "ori": [0.0, 0.0, 0.6, 0.8], "fixed_base": False, "moved_to": reg["chair_0"]["root_link"]["pos"]}
    roots = [s["root_link"]["pos"] for s in scene["state"]["registry"]["object_registry"].values()]
    for name in ("chair_0", "chair_1"):
        root = reg[name]["root_link"]
        assert init[name]["args"]["fixed_base"] is True and root["lin_vel"] == [0.0, 0.0, 0.0] and root["ang_vel"] == [0.0, 0.0, 0.0]
        assert root["ori"] == scene["state"]["registry"]["object_registry"][name]["root_link"]["ori"]
        assert all(root["pos"][k] <= min(r[k] for r in roots) - SV.PARK_MARGIN_M + 1e-9 for k in range(3))  # beyond every root of the scene
    assert reg["chair_0"]["root_link"]["pos"] != reg["chair_1"]["root_link"]["pos"]
    assert init["table_0"] == scene["objects_info"]["init_info"]["table_0"] and reg["table_0"] == scene["state"]["registry"]["object_registry"]["table_0"]


def test_remove_mode_deletes_the_objects_and_keeps_its_file_name(tmp_path):
    path, _ = _scene(tmp_path)
    rpath, r = SV.write_variant(path, ["chair_0"], str(tmp_path / "out"), mode="remove")
    assert "chair_0" not in r["objects_info"]["init_info"] and "chair_0" not in r["state"]["registry"]["object_registry"]
    assert r["metadata"]["ispatialgym"]["removed_objects"] == ["chair_0"] and "moved_objects" not in r["metadata"]["ispatialgym"]
    assert os.path.basename(rpath) == f"house__{hashlib.sha256(b'chair_0').hexdigest()[:10]}.json"  # the reach-band worlds' names as before
    mpath, _ = SV.write_variant(path, ["chair_0"], str(tmp_path / "out"))
    assert mpath != rpath
    with pytest.raises(ValueError):
        SV.write_variant(path, ["chair_0"], str(tmp_path / "out"), mode="hide")


def test_the_template_lists_the_moved_objects(tmp_path, monkeypatch):
    """``SimWorld.save_template`` copies the variant's ``moved_objects`` into ``template.json`` (OmniGibson's merge keeps
    the live scene's metadata only)."""
    from types import SimpleNamespace

    from ispatialgym.sampling import sim_utils as SU

    path, _ = _scene(tmp_path)
    _, variant = SV.write_variant(path, ["chair_0"], str(tmp_path / "out"))
    live = copy.deepcopy(variant)
    live["metadata"] = {"task": {"inst_to_name": {}}}

    def save(json_path):
        with open(json_path, "w") as f:
            json.dump(live, f)

    w = object.__new__(SU.SimWorld)
    w.render_hidden, w.stable_scene = [], variant
    w.scene = SimpleNamespace(write_task_metadata=lambda key, data: None, save=save)
    monkeypatch.setattr(SU, "merge_scene_files", lambda a, b, keep_robot_from="b": copy.deepcopy(b))
    _, full = w.save_template(str(tmp_path / "template"), {}, {})
    tpl = json.load(open(full))
    assert tpl["metadata"]["ispatialgym"]["moved_objects"] == variant["metadata"]["ispatialgym"]["moved_objects"]
    assert tpl["metadata"]["ispatialgym"]["source_scene_file"] == "house_stable.json" and tpl["metadata"]["task"] == {"inst_to_name": {}}
    assert tpl["objects_info"]["init_info"]["chair_0"]["args"]["fixed_base"] is True
