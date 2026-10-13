"""Simulator-free checks of the reach-band cache keying (owner decision 2026-09-16 item 9): the ``--placed`` spec grammar,
its canonical form, and ``ReachBands.find`` matching legacy / stance-relative placed-tray bands through it."""

import json
import os
import tempfile

import numpy as np
import pytest

from ispatialgym.sampling import reach_bands as RB

FOCUS = RB.PLACED_MODE_FOCUS


def test_placed_spec_forms_agree():
    canon = "tray/gsxbym*1.25@focus_centred_edge_aligned"
    assert RB.PlacedSpec.parse(canon).canonical() == canon
    assert RB.PlacedSpec.parse(("tray/gsxbym", 1.25, FOCUS)).canonical() == canon
    assert RB.PlacedSpec.parse(["tray/gsxbym", [1.25, 1.25, 1.25], FOCUS]).canonical() == canon
    assert RB.PlacedSpec.parse({"category": "tray", "model": "gsxbym", "scale": [1.25, 1.25, 1.25], "mode": FOCUS}).canonical() == canon
    assert RB.placed_spec("tray/gsxbym", 1.25) == canon
    assert RB.placed_spec("tray/gsxbym", [1.25, 1.25, 1.25]) == canon
    assert RB.placed_spec("tray/gsxbym", 1.25, RB.PLACED_MODE_EDGE) == "tray/gsxbym*1.25@edge"


def test_placed_spec_legacy_and_defaults():
    # the legacy fleet bands carry ``placed_specs: ["tray/gsxbym"]`` = the fixed edge placement at the canonical size
    p = RB.PlacedSpec.parse("tray/gsxbym")
    assert (p.category, p.model, p.scale, p.mode) == ("tray", "gsxbym", None, RB.PLACED_MODE_EDGE)
    assert p.canonical() == "tray/gsxbym@edge"
    assert RB.PlacedSpec.parse("tray/gsxbym*1").canonical() == "tray/gsxbym@edge"  # scale 1 = canonical size
    assert RB.PlacedSpec.parse("tray/gsxbym*1,1.5,2").canonical() == "tray/gsxbym*1,1.5,2@edge"
    assert RB.PlacedSpec.parse("tray/gsxbym@somebody_elses_mode").canonical() == "tray/gsxbym@somebody_elses_mode"  # kept verbatim
    with pytest.raises(ValueError):
        RB.PlacedSpec.parse("no_slash")
    assert RB.canonical_placed({"placed_specs": ["tray/gsxbym"]}) == ["tray/gsxbym@edge"]
    assert RB.canonical_placed({"placed": [{"category": "tray", "model": "gsxbym"}]}) == ["tray/gsxbym@edge"]  # metas before placed_specs
    assert RB.canonical_placed(["b/y", ("a/x", 2, FOCUS)]) == ["a/x*2@focus_centred_edge_aligned", "b/y@edge"]


def _write_band(d, name, meta_extra):
    meta = {"scene": "s", "support": "t", "object": "mug/ycbmug", "grasp_key": "mug__ycbmug__t", "generic_pinch": False, "lattice": 0.08, "mode": "tabletop",
            "trunks": [{"translate": 0.5, "tilt": 0.0, "collides": False}], "scale": None}
    meta.update(meta_extra)
    np.savez_compressed(os.path.join(d, name + ".npz"), **{RB.trunk_key(0.5, 0.0): np.zeros((1, 2), dtype=np.float32)})
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(meta, f)


def test_find_matches_placed_specs_through_canonical_form():
    with tempfile.TemporaryDirectory() as d:
        _write_band(d, "plain", {})
        _write_band(d, "legacy_tray", {"placed_specs": ["tray/gsxbym"], "placed": [{"category": "tray", "model": "gsxbym"}]})
        _write_band(d, "focus_tray", {"placed_specs": ["tray/gsxbym*1.25@focus_centred_edge_aligned"],
                                      "placed": [{"category": "tray", "model": "gsxbym", "scale": [1.25, 1.25, 1.25], "mode": FOCUS}]})
        find = lambda **kw: [os.path.basename(p)[:-4] for p in RB.ReachBands.find("s", "t", "mug/ycbmug", "mug__ycbmug__t", dirs=[d], **kw)]  # noqa: E731
        assert find() == ["plain"]
        assert find(placed=["tray/gsxbym"]) == ["legacy_tray"]
        assert find(placed=["tray/gsxbym@edge"]) == ["legacy_tray"]
        assert find(placed=[RB.placed_spec("tray/gsxbym", 1.25)]) == ["focus_tray"]
        assert find(placed=[("tray/gsxbym", 1.25, FOCUS)]) == ["focus_tray"]
        assert find(placed=[("tray/gsxbym", [1.25, 1.25, 1.25], FOCUS)]) == ["focus_tray"]
        assert find(placed=["tray/gsxbym@focus_centred_edge_aligned"]) == []  # canonical tray size: not the 1.25 band
        assert find(placed=["tray/gsxbym*1.25"]) == []  # 1.25 in the legacy edge mode: no such band
        with pytest.raises(FileNotFoundError):
            RB.ReachBands.load("s", "t", "mug/ycbmug", "mug__ycbmug__t", placed=["tray/hvlfig"], dirs=[d])
        rb = RB.ReachBands.load("s", "t", "mug/ycbmug", "mug__ycbmug__t", placed=[RB.placed_spec("tray/gsxbym", 1.25)], dirs=[d])
        assert rb.meta["placed"][0]["mode"] == FOCUS


def test_find_matches_support_joints():
    """``--support-joints`` (inside_large_scale#02, the fridge door open): a band computed with the support's joints set is found only
    when the same joints are asked for (within 0.02); without a request only the bands of the support as the scene file has it."""
    with tempfile.TemporaryDirectory() as d:
        _write_band(d, "closed", {})
        _write_band(d, "door_open", {"support_joints": {"j_left_door": 2.3}})
        find = lambda **kw: [os.path.basename(p)[:-4] for p in RB.ReachBands.find("s", "t", "mug/ycbmug", "mug__ycbmug__t", dirs=[d], **kw)]  # noqa: E731
        assert find() == ["closed"] and find(support_joints={}) == ["closed"]
        assert find(support_joints={"j_left_door": 2.3}) == ["door_open"] and find(support_joints={"j_left_door": 2.31}) == ["door_open"]
        assert find(support_joints={"j_left_door": 1.57}) == [] and find(support_joints={"j_right_door": 2.3}) == []
        with pytest.raises(FileNotFoundError, match="support_joints"):
            RB.ReachBands.load("s", "t", "mug/ycbmug", "mug__ycbmug__t", support_joints={"j_left_door": 1.0}, dirs=[d])
    args = RB.build_parser().parse_args(["--scene", "s", "--support", "fridge_petcxr_0", "--surface-z", "1.20", "--support-joints", '{"j_left_door": 2.3}'])
    assert json.loads(args.support_joints) == {"j_left_door": 2.3}
    assert RB.build_parser().parse_args(["--scene", "s", "--support", "t"]).support_joints is None


# ----------------------------------------------------------------------------------------------- --jobs worker (no simulator)

#: a stable-scene stub: the ``in_rooms`` of the supports decide the loaded rooms; the chair models are unknown to the
#: bddl bounding-box table, so ``chairs_near_tables`` removes nothing (the removal geometry is tested with the real data)
_STABLE = {"objects_info": {"init_info": {"table_a": {"args": {"category": "breakfast_table", "model": "zzzz", "in_rooms": ["dining_room_0"]}},
                                          "table_b": {"args": {"category": "desk", "model": "zzzz", "in_rooms": ["living_room_0"]}},
                                          "shelf_c": {"args": {"category": "shelf", "model": "zzzz", "in_rooms": ["dining_room_0"]}}}},
           "state": {"registry": {"object_registry": {}}}}


def _job(jid, *args):
    return {"id": jid, "args": list(args)}


def test_select_jobs_and_group_by_key():
    manifest = [_job("a", "--scene", "s", "--support", "table_a"), _job("b", "--scene", "s", "--support", "table_b"), _job("c", "--scene", "s", "--support", "shelf_c")]
    assert [j["id"] for j in RB.select_jobs(manifest, None)] == ["a", "b", "c"]
    assert [j["id"] for j in RB.select_jobs(manifest, ["c", "a"])] == ["c", "a"]
    with pytest.raises(KeyError):
        RB.select_jobs(manifest, ["a", "nope"])
    # groups keep the order of their first item and the given order inside a group
    items = [("a", 1), ("b", 2), ("c", 1), ("d", 3), ("e", 2)]
    assert [i[0] for i in RB.group_by_key(items, lambda it: it[1])] == ["a", "c", "b", "e", "d"]


def test_band_world_key_groups_supports_of_one_room():
    ap = RB.build_parser()
    key = lambda *a: RB.band_world_key(ap.parse_args(list(a)), stable=_STABLE)  # noqa: E731
    ka = key("--scene", "s", "--support", "table_a")
    kc = key("--scene", "s", "--support", "shelf_c", "--object", "mug/ycbmug")
    kb = key("--scene", "s", "--support", "table_b")
    assert ka == ("s", ("dining_room_0",), ())
    assert ka == kc and kb != ka  # same room, no chairs to remove -> one world; the other room -> another
    assert key("--scene", "s", "--support", "table_a", "--rooms", "living_room_0") == kb  # --rooms overrides the support's rooms
    assert key("--scene", "s", "--support", "floor", "--room", "dining_room_0") == ka  # floor mode: the room, no chair removal
    under = ap.parse_args(["--scene", "s", "--support", "floor", "--under", "table_b"])
    assert RB.band_world_key(under, stable=_STABLE) == kb
    assert under.room == "living_room_0"  # floor-under: the lattice floor's room is the partner's first room
    with pytest.raises(SystemExit):
        key("--scene", "s", "--support", "floor")
    with pytest.raises(SystemExit):
        key("--scene", "s", "--support", "not_there")
    assert RB.band_arms(ap.parse_args(["--scene", "s", "--support", "t", "--arm", "left"])) == ("left",)
    assert RB.band_arms(ap.parse_args(["--scene", "s", "--support", "t", "--brace"])) == ("right", "left")
    with pytest.raises(SystemExit):
        RB.band_arms(ap.parse_args(["--scene", "s", "--support", "t", "--brace", "--generic-contact"]))


def _obj(name, category, model, pos, rooms, joints=False, scale=None):
    init = {"class_module": "omnigibson.objects.dataset_object", "class_name": "DatasetObject",
            "args": {"name": name, "category": category, "model": model, "scale": scale or [1.0, 1.0, 1.0], "in_rooms": rooms}}
    state = {"root_link": {"pos": list(pos), "ori": [0.0, 0.0, 0.0, 1.0]}}
    if joints:
        state["joint_pos"] = [0.0]
    return init, state


def test_slim_drop_list_keeps_every_possible_obstacle(monkeypatch):
    """--slim-load: objects of the loaded neighbour rooms (and the always-loaded structures) are dropped only when their json box,
    grown by the safety margin, cannot meet the task rooms' floor box + the obstacle margin; task-room objects, unknown boxes,
    articulated objects near the boundary and supports of kept objects stay."""
    from ispatialgym.sampling import knowledge as K
    from ispatialgym.sampling import scene_variants as SV

    boxes = {"floor": [4.0, 4.0, 0.2], "cube": [0.5, 0.5, 0.5], "wall": [0.2, 4.0, 3.0], "door": [0.2, 1.0, 2.2], "plank": [3.0, 0.3, 0.05]}
    monkeypatch.setattr(K, "canonical_bbox", lambda model: list(boxes[model]) if model in boxes else None)
    monkeypatch.setattr(SV, "_metadata", lambda category, model: {"base_link_offset": [0.0, 0.0, 0.0]})
    objs = [
        _obj("floor_task", "floors", "floor", (0.0, 0.0, -0.1), ["task"]),
        _obj("floor_next", "floors", "floor", (4.0, 0.0, -0.1), ["next"]),  # touches the task floor: kept
        _obj("cube_task_far", "box", "cube", (30.0, 0.0, 0.25), ["task"]),  # task room: never dropped
        _obj("cube_near", "box", "cube", (3.5, 0.0, 0.25), ["next"]),  # within the margins: kept
        _obj("cube_far", "box", "cube", (9.0, 0.0, 0.25), ["next"]),  # 7 m from the floor box edge (2 m): dropped
        _obj("cube_unknown", "box", "nomodel", (9.0, 0.0, 0.25), ["next"]),  # unknown box: kept
        _obj("cube_other_room", "box", "cube", (9.0, 0.0, 0.25), ["elsewhere"]),  # not loaded at all: not listed
        _obj("wall_far", "walls", "wall", (12.0, 0.0, 1.5), []),  # always loaded structure, far: dropped
        _obj("wall_near", "walls", "wall", (2.5, 0.0, 1.5), []),  # always loaded structure, near: kept
        _obj("door_far", "door", "door", (12.0, 3.0, 1.1), []),  # always loaded door, far: dropped
        _obj("door_edge", "door", "door", (5.2, 0.0, 1.1), [], joints=True),  # articulated: +2.2 m extent reaches the bounds -> kept
        _obj("plank", "board", "plank", (4.6, 0.0, 0.6), ["next"]),  # kept (within margins) and rests on ...
        _obj("stool_far", "stool", "cube", (5.9, 0.0, 0.325), ["next"]),  # ... this: its top (0.575 + margin) meets the plank's bottom -> kept by the closure
    ]
    stable = {"objects_info": {"init_info": {i["args"]["name"]: i for i, _ in objs}}, "state": {"registry": {"object_registry": {i["args"]["name"]: s for i, s in objs}}}}
    dropped = RB._slim_drop_list(stable, ["task"], ["task", "next"], obstacle_margin=1.0, margin=0.5)
    assert dropped == ["cube_far", "door_far", "wall_far"]
    # a larger obstacle margin keeps more; nothing is ever dropped from the task rooms
    assert RB._slim_drop_list(stable, ["task"], ["task", "next"], obstacle_margin=6.0, margin=0.5) == ["door_far", "wall_far"]
    assert "cube_task_far" not in RB._slim_drop_list(stable, ["task"], ["task", "next"], obstacle_margin=0.0, margin=0.0)
