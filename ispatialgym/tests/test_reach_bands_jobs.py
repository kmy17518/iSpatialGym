"""Simulator-free checks of the reach-band job grouping helper ``scripts/reach_bands/make_jobs.py --group-by-world``: manifest
jobs sharing a world key (scene, loaded rooms, removed chairs) become one worker job (``reach_bands --jobs MANIFEST --ids ...``),
chunked, in priority order, with the member jobs untouched."""

import importlib.util
import json
import os
import tempfile

from ispatialgym import REPO_ROOT

_SPEC = importlib.util.spec_from_file_location("make_jobs", os.path.join(REPO_ROOT, "scripts", "reach_bands", "make_jobs.py"))
MJ = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MJ)

#: stable-scene stub (see test_reach_bands._STABLE): unknown chair models -> no chair removal, rooms from ``in_rooms``
_STABLE = {"objects_info": {"init_info": {"table_a": {"args": {"category": "breakfast_table", "model": "zzzz", "in_rooms": ["dining_room_0"]}},
                                          "desk_b": {"args": {"category": "desk", "model": "zzzz", "in_rooms": ["living_room_0"]}},
                                          "shelf_c": {"args": {"category": "shelf", "model": "zzzz", "in_rooms": ["dining_room_0"]}}}},
           "state": {"registry": {"object_registry": {}}}}


def _job(jid, weight, *args):
    return {"id": jid, "kind": "tabletop", "subtasks": ["s"], "picks": [jid], "slot": "representative", "scene": "s", "weight": weight, "args": list(args),
            "cmd": "x"}


def test_group_by_world(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "s", "json"))
        with open(os.path.join(d, "s", "json", "s_stable.json"), "w") as f:
            json.dump(_STABLE, f)
        monkeypatch.setattr(MJ, "STABLE", os.path.join(d, "{scene}", "json", "{scene}_stable.json"))
        jobs = [_job("j1", 2.0, "--scene", "s", "--support", "table_a", "--object", "mug/a", "--grasp-key", "k"),
                _job("j2", 3.0, "--scene", "s", "--support", "desk_b", "--object", "mug/a", "--grasp-key", "k"),
                _job("j3", 2.0, "--scene", "s", "--support", "shelf_c", "--object", "mug/b", "--grasp-key", "k"),
                _job("j4", 2.0, "--scene", "s", "--support", "table_a", "--object", "mug/c", "--generic-pinch"),
                _job("j5", 1.0, "--scene", "s", "--support", "floor", "--room", "dining_room_0", "--object", "mug/c", "--grasp-key", "k"),
                _job("j6", 4.0, "--scene", "s", "--support", "nowhere", "--object", "mug/c", "--grasp-key", "k")]  # unknown support: its own worker job
        groups = MJ.group_by_world(jobs, "manifest.json", 2, wave=7)
        by_ids = {tuple(g["ids"]): g for g in groups}
        # dining room: j1, j3, j4, j5 share the key -> chunks of 2 in manifest order; living room: j2 alone; j6 alone
        assert set(by_ids) == {("j1", "j3"), ("j4", "j5"), ("j2",), ("j6",)}
        g = by_ids[("j1", "j3")]
        assert g["args"] == ["--jobs", "manifest.json", "--ids", "j1", "j3"] and g["cmd"].endswith("--jobs manifest.json --ids j1 j3 --out $OUT")
        assert g["kind"] == "worker" and g["wave"] == 7 and g["weight"] == 2.0 and g["world_key"] == ["s", ["dining_room_0"], []]
        assert by_ids[("j4", "j5")]["weight"] == 2.0 and by_ids[("j2",)]["weight"] == 3.0
        assert [g["ids"] for g in groups][0] == ["j6"]  # heaviest first
        assert all("cmd" in j and j["cmd"] == "x" for j in jobs)  # the member jobs are untouched
