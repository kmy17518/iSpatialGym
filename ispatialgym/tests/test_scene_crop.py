"""train-fast SCENE_CROP (2026-11-01, ``ispatialgym.sampling.scene_crop``): the crop of a scene json, pure json (no simulator).

A synthetic scene: task room ``room_a`` (floor x, y in [-4, 4]) with the support table at the centre, the neighbouring ``room_b``
(today's load: adjacent), a far ``room_c`` (never loaded), walls / doors (loaded whatever the rooms). Boxes come from ``args._box``
(``box_fn``), grown by the margin the crop asks for (+ 1 m for an articulated object).
"""

from __future__ import annotations

import json
import os

import pytest

from ispatialgym.sampling import scene_crop as SC

ALWAYS = {"walls", "ceilings", "roof", "fence", "door", "sliding_door"}
STRUCTURE = {"walls", "floors", "ceilings", "roof"}
KW = dict(radius=2.5, obstacle_margin=1.0, slim_margin=1.0, region_margin=0.35, always=ALWAYS, structure=STRUCTURE)


def _box_fn(name, args, state, margin, articulated):
    if "_box" not in args:
        return None
    lo, hi = args["_box"]
    g = margin + (1.0 if articulated else 0.0)
    return [v - g for v in lo], [v + g for v in hi]


def _obj(cat, rooms, lo, hi, joint_pos=None, cls="DatasetObject"):
    args = {"category": cat, "in_rooms": rooms, "model": "m"}
    if lo is not None:
        args["_box"] = [list(lo), list(hi)]
    state = {"root_link": {"pos": [(a + b) / 2 for a, b in zip(lo or [0, 0, 0], hi or [0, 0, 0])], "ori": [0, 0, 0, 1]}}
    if joint_pos is not None:
        state["joint_pos"] = joint_pos
    return {"class_name": cls, "args": args}, state


def scene(fridge_joint=0.0):
    objs = {
        "floors_a": _obj("floors", ["room_a"], (-4, -4, -0.1), (4, 4, 0)),
        "floors_b": _obj("floors", ["room_b"], (4.2, -4, -0.1), (10, 4, 0)),
        "table": _obj("breakfast_table", ["room_a"], (-0.5, -1, 0), (0.5, 1, 0.8)),
        "chair_near": _obj("straight_chair", ["room_a"], (0.6, -0.3, 0), (1.1, 0.3, 0.9)),
        "sofa_far": _obj("sofa", ["room_a"], (3.6, -3.9, 0), (4.0, -3.0, 0.9)),
        "cabinet_far": _obj("bottom_cabinet", ["room_a"], (3.5, 3.5, 0), (3.9, 3.9, 1.0)),
        "vase": _obj("vase", ["room_a"], (3.6, 3.6, 1.0), (3.8, 3.8, 1.3)),  # bound (partner) and resting on cabinet_far
        "fridge": _obj("fridge", ["room_a"], (3.5, 0.0, 0), (3.9, 0.5, 1.8), joint_pos=[fridge_joint]),
        "mystery": _obj("thing", ["room_a"], None, None),
        "moved_chair": _obj("straight_chair", ["room_a"], (-100.3, -100.3, -100), (-99.7, -99.7, -99.1)),
        "wall_cross": _obj("walls", [], (-5, 2.9, 0), (5, 3.1, 3)),
        "wall_mid": _obj("walls", [], (3.8, -4, 0), (4.0, 4, 3)),
        "wall_far": _obj("walls", [], (20, -4, 0), (20.2, 4, 3)),
        "door_far": _obj("door", [], (30, 0, 0), (31, 0.1, 2)),
        "bed_b": _obj("bed", ["room_b"], (6, 0, 0), (8, 2, 0.6)),
        "desk_c": _obj("desk", ["room_c"], (50, 0, 0), (51, 1, 0.8)),
        "robot_r1": _obj("agent", [], (0, 0, 0), (0.5, 0.5, 1.5), cls="Robot"),
    }
    return {
        "metadata": {"ispatialgym": {"removed_objects": ["moved_chair"]}, "task": {}},
        "init_info": {"args": {"scene_model": "synthetic"}},
        "objects_info": {"init_info": {n: o[0] for n, o in objs.items()}},
        "state": {"pos": [0, 0, 0], "ori": [0, 0, 0, 1], "registry": {"object_registry": {n: o[1] for n, o in objs.items()}, "system_registry": {"water": {}}}},
    }


BINDINGS = {"support": ["table"], "partner": ["vase"], "floor": {"room": "room_a"}}
TODAY = ["room_a", "room_b"]
JSON_OBJECTS = 16  # robot excluded
TODAY_SET = {"floors_a", "floors_b", "table", "chair_near", "sofa_far", "cabinet_far", "vase", "fridge", "mystery", "moved_chair", "wall_cross",
             "wall_mid", "wall_far", "door_far", "bed_b"}


def crop(mode, sc=None, bindings=BINDINGS, **kw):
    return SC.compute_crop(sc or scene(), mode, ["room_a"], TODAY, bindings=bindings, box_fn=_box_fn, **{**KW, **kw})


def test_full_is_todays_loader_rule():
    p = crop("full")
    assert p.mode == "full" and not p.cropped
    assert set(p.kept) == TODAY_SET and p.dropped == []
    assert p.n_json == JSON_OBJECTS and p.n_today == len(TODAY_SET)
    assert p.meta() == {"mode": "full"}
    assert p.describe().startswith("SCENE_CROP full: loading 15 of 16 scene-json objects")


def test_rooms_keeps_what_can_meet_the_collision_world():
    p = crop("rooms")
    # the neighbour's floor meets the bounds: kept, and its room joins load_room_instances (only kept objects are in the file)
    assert p.mode == "rooms" and p.loaded_rooms == ["room_a", "room_b"]
    # the task floor extent (+1 m json margin) +1 m obstacle margin: [-6, 6]; the far wall / door, the parked chair leave
    assert set(p.kept) == {"floors_a", "floors_b", "table", "chair_near", "sofa_far", "cabinet_far", "vase", "fridge", "mystery", "wall_cross", "wall_mid"}
    # dropped = today's load minus the crop: the neighbour's bed (not a candidate: its room is not a task room), far structure, the parked chair
    assert p.dropped == sorted({"bed_b", "wall_far", "door_far", "moved_chair"})
    assert p.dropped_by_category == {"bed": 1, "walls": 1, "door": 1, "straight_chair": 1}
    assert "desk_c" not in p.kept and "desk_c" not in p.dropped  # never loaded at all


def test_region_restricts_to_the_support_radius():
    p = crop("region")
    assert p.mode == "region" and p.center_object == "table" and p.radius_m == 2.5
    assert p.region_xy == ([-3.0, -3.5], [3.0, 3.5])
    kept = set(p.kept)
    # near furniture, the wall crossing the region, the task floor, the support, the unknown box
    assert {"table", "chair_near", "wall_cross", "floors_a", "mystery"} <= kept
    # outside the radius: the sofa, the closed fridge, the wall at the room's far side
    assert not ({"sofa_far", "fridge", "wall_mid", "wall_far"} & kept)
    # the bound partner far away is kept, and what it rests on (resting closure)
    assert {"vase", "cabinet_far"} <= kept
    assert set(p.dropped) == TODAY_SET - kept


@pytest.mark.parametrize("mode", ["rooms", "region"])
def test_requested_parked_objects_survive_crop_and_keep_their_restore_state(mode):
    sc = scene()
    sc["objects_info"]["init_info"]["moved_chair"]["args"]["fixed_base"] = True
    sc["metadata"]["ispatialgym"]["moved_objects"] = {
        "moved_chair": {"pos": [1, 1, 0], "moved_to": [-100, -100, -99.55], "fixed_base": False}
    }
    assert "moved_chair" not in crop(mode, sc=sc).kept
    plan = crop(mode, sc=sc, extra_keep=["moved_chair"])
    loaded = SC.cropped_scene(sc, plan)
    assert SC.loaded_by_rule(loaded["objects_info"]["init_info"]["moved_chair"]["args"], plan.loaded_rooms, ALWAYS)
    assert loaded["objects_info"]["init_info"]["moved_chair"]["args"]["fixed_base"]
    assert loaded["state"]["registry"]["object_registry"]["moved_chair"] == sc["state"]["registry"]["object_registry"]["moved_chair"]
    assert loaded["metadata"]["ispatialgym"]["moved_objects"] == sc["metadata"]["ispatialgym"]["moved_objects"]


def test_region_grows_an_open_articulated_object():
    assert "fridge" not in crop("region").kept
    assert "fridge" in crop("region", sc=scene(fridge_joint=0.5)).kept  # an open door may swing into the region
    assert "fridge" in crop("rooms").kept


def test_region_resting_closure_follows_chains():
    sc = scene()
    init, reg = sc["objects_info"]["init_info"], sc["state"]["registry"]["object_registry"]
    # a stand under the far cabinet: cabinet (kept for the vase) rests on it
    init["cabinet_far"]["args"]["_box"] = [[3.5, 3.5, 0.3], [3.9, 3.9, 1.0]]
    init["stand"], reg["stand"] = _obj("shelf", ["room_a"], (3.4, 3.4, 0), (4.0, 4.0, 0.3))
    p = crop("region", sc=sc)
    assert {"vase", "cabinet_far", "stand"} <= set(p.kept)


def test_region_falls_back_to_rooms_without_a_single_support():
    for b, why in (({"floor": {"room": "room_a"}}, "no support"), ({"support": ["table", "chair_near"]}, "2 support candidates"),
                   ({"support": {"room": "room_a"}}, "no support")):
        p = crop("region", bindings=b)
        assert p.mode == "rooms" and p.requested_mode == "region" and why in p.note
        assert p.meta()["mode"] == "rooms" and p.meta()["requested_mode"] == "region"
        assert p.key() == ("rooms",) == SC.request_key("region", b)


def test_region_unions_the_declared_neighbours():
    b = {**BINDINGS, "support_surfaces": [{"name": "table", "neighbors": ["chair_near"]}]}
    p = crop("region", bindings=b)
    assert p.center_names == ["table", "chair_near"] and p.region_xy == ([-3.0, -3.5], [3.6, 3.5])
    assert p.key() == SC.request_key("region", b, 2.5)


def test_no_task_floor_is_full():
    sc = scene()
    del sc["objects_info"]["init_info"]["floors_a"]
    p = crop("region", sc=sc)
    assert p.mode == "full" and "no floor" in p.note and p.meta() == {"mode": "full"}


def test_meta_and_log_line():
    p = crop("region")
    m = p.meta()
    assert set(m) == {"mode", "radius_m", "center_object", "n_loaded_json", "n_dropped", "dropped", "loaded_rooms", "loaded_scene_file"}
    assert m["n_loaded_json"] == len(p.kept) and m["n_dropped"] == len(m["dropped"]) and m["dropped"] == sorted(m["dropped"])
    # floors_b is kept by the resting closure of the bound vase (its grown footprint reaches over the room boundary)
    assert m["loaded_rooms"] == ["room_a", "room_b"] and m["loaded_scene_file"] == "template-partial_rooms.json"
    line = p.describe()
    assert line.startswith(f"SCENE_CROP region: loading {len(p.kept)} of 16 scene-json objects (task rooms ['room_a'], radius 2.50 m around table)")
    assert f"dropped {len(p.dropped)}" in line and "walls" in line


def test_cropped_file(tmp_path):
    sc = scene()
    p = crop("region", sc=sc)
    path = SC.write_cropped_scene(sc, p, str(tmp_path), "synthetic", "/x/synthetic__abc.json")
    assert os.path.basename(path).startswith("synthetic__crop_") and p.path == path
    with open(path) as f:
        out = json.load(f)
    assert set(out["objects_info"]["init_info"]) == set(p.kept) | {"robot_r1"}
    assert set(out["state"]["registry"]["object_registry"]) == set(p.kept) | {"robot_r1"}
    assert out["metadata"] == sc["metadata"] and out["init_info"] == sc["init_info"]  # the loaded world's task metadata is unchanged
    assert out["state"]["registry"]["system_registry"] == {"water": {}}
    assert len(sc["objects_info"]["init_info"]) == 17  # the source is not modified
    # same inputs, same file; another crop, another file
    assert SC.write_cropped_scene(sc, crop("region", sc=sc), str(tmp_path), "synthetic", "/x/synthetic__abc.json") == path
    assert SC.write_cropped_scene(sc, crop("rooms", sc=sc), str(tmp_path), "synthetic", "/x/synthetic__abc.json") != path


def test_covers():
    region = crop("region").key()
    other = ("region", 2.5, "chair_near", ("chair_near",))
    assert SC.covers(("full",), region) and SC.covers(("rooms",), region) and SC.covers(region, region)
    assert not SC.covers(region, other) and not SC.covers(region, ("rooms",)) and not SC.covers(("rooms",), ("full",))
    assert not SC.covers(None, ("full",))


def test_switch(monkeypatch):
    for k in ("ISPATIALGYM_FAST", "ISPATIALGYM_FAST_SCENE_CROP"):
        monkeypatch.delenv(k, raising=False)
    assert SC.requested_mode("inter_object_relation_satisfaction-ontop_small_scale") == "region"
    assert SC.requested_mode("inter_object_relation_satisfaction-ontop_large_scale") == "rooms"
    assert SC.requested_mode("camera_relocalization-standard") == "full"
    assert SC.requested_mode("disembedding") == "region"
    assert SC.requested_mode("object_scaling") == "rooms"
    assert SC.requested_mode("no_such_family") == "full" and SC.requested_mode(None) == "full"
    monkeypatch.setenv("ISPATIALGYM_FAST_SCENE_CROP", "rooms")
    assert SC.requested_mode("alignment-axial") == "rooms"
    monkeypatch.setenv("ISPATIALGYM_FAST_SCENE_CROP", "0")
    assert SC.requested_mode("alignment-axial") == "full"
    monkeypatch.setenv("ISPATIALGYM_FAST_SCENE_CROP", "bogus")
    assert SC.requested_mode("alignment-axial") == "region"
    monkeypatch.setenv("ISPATIALGYM_FAST_SCENE_CROP", "region")
    monkeypatch.setenv("ISPATIALGYM_FAST", "0")
    assert SC.requested_mode("alignment-axial") == "full"


def test_every_registered_family_has_a_default():
    from ispatialgym.sampling.sample_isg import PIXEL_SUBTASK_PREFIXES

    assert set(SC.SCENE_CROP_DEFAULTS.values()) <= set(SC.MODES)
    for prefix in ("camera_relocalization", "alignment-world_vertical"):
        assert all(m == "full" for s, m in SC.SCENE_CROP_DEFAULTS.items() if s.startswith(prefix))
    assert all(any(s.startswith(p) for s in SC.SCENE_CROP_DEFAULTS) for p in PIXEL_SUBTASK_PREFIXES)


@pytest.mark.parametrize("bindings,expected", [(BINDINGS, ["table", "vase"]), ({"support": {"room": "a"}, "target": ["x", 3]}, ["x"])])
def test_bound_names(bindings, expected):
    assert SC.bound_names(bindings) == expected


def test_region_neighbour_floor_only_where_the_robot_can_stand():
    """2026-11-01: a neighbour room's floor is a candidate (OmniGibson loads floors only with their room). Out of the region it
    leaves and its room is not loaded; inside the region (a table next to the room boundary) it stays and its room loads."""
    far = crop("region", bindings={"support": ["table"], "floor": {"room": "room_a"}})
    assert "floors_b" in far.dropped and far.loaded_rooms == ["room_a"]
    sc = scene()
    sc["objects_info"]["init_info"]["floors_b"]["args"]["_box"] = [[3.2, -4, -0.1], [10, 4, 0]]  # 0.2 m past the region edge (x 3.0)
    near = crop("region", sc=sc, bindings={"support": ["table"], "floor": {"room": "room_a"}})
    assert "floors_b" in near.kept and near.loaded_rooms == ["room_a", "room_b"]
    assert "bed_b" not in near.kept  # the neighbour's furniture is still no candidate
