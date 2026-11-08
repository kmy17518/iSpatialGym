"""i7-goalmaps 2026-11-08 (owner decision, prompt_goalmaps_axialrest_overpath.md section 1): the goal surfaces of the IORS large predicates
in the height-anchor rule (``ispatialgym.sampling.height_anchors``). Simulator-free."""

import csv
import json
import os

from ispatialgym import REPO_ROOT
from ispatialgym.sampling import height_anchors as HA

ROLES = os.path.join(REPO_ROOT, "ispatialgym", "assets", "icl", "roles")
CAND = os.path.join(ROLES, "candidates.csv")
REPS = os.path.join(ROLES, "representatives.csv")
ONTOP = "inter_object_relation_satisfaction-ontop_large_scale"


def _inv(root, scene, kind, name, top, bottom, category="desk"):
    d = os.path.join(root, scene, kind)
    os.makedirs(d, exist_ok=True)
    json.dump({"category": category, "aabb_world": {"min": [0.0, 0.0, bottom], "max": [1.0, 1.0, top]}, "top": {"z": top}},
              open(os.path.join(d, f"{name}.json"), "w"))


def _table(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return str(path)


def _world(tmp_path):
    """Scene S: desk_a (a support row, top 0.75), bed_b (the ontop partner, inventory in partners/ only, top 0.50), bed_c (a partner of
    the representatives schema: the instance in ``name``, top 0.62); mug m1 the ontop target, cup m2 a target of another subtask."""
    inv = str(tmp_path / "inv")
    _inv(inv, "S", "supports", "desk_a", 0.75, 0.70)
    _inv(inv, "S", "partners", "bed_b", 0.50, -0.01, "bed")
    _inv(inv, "S", "partners", "bed_c", 0.62, -0.01, "bed")
    fields = ["subtask", "role", "category", "model", "scene", "rank", "top_z"]
    cand = _table(tmp_path / "candidates.csv", [
        {"subtask": ONTOP, "role": "target", "category": "mug", "model": "m1"},
        {"subtask": ONTOP, "role": "support", "category": "desk", "model": "desk_a", "scene": "S", "rank": "4"},
        {"subtask": ONTOP, "role": "partner", "category": "bed", "model": "bed_b", "scene": "S", "rank": "4"},
        {"subtask": "disembedding", "role": "target", "category": "cup", "model": "m2"},
        {"subtask": "disembedding", "role": "support", "category": "desk", "model": "desk_a", "scene": "S", "rank": "4"},
        # a partner of a predicate whose goal reads no tabletop map: never a surface
        {"subtask": "inter_object_relation_satisfaction-under_large_scale", "role": "partner", "category": "bed", "model": "bed_c", "scene": "S"},
    ], fields)
    reps = _table(tmp_path / "representatives.csv", [
        {"subtask": ONTOP, "pick": ONTOP + "#01", "role": "partner", "slot": "heldout", "category": "bed", "model": "ccc", "scene": "S", "name": "bed_c"},
    ], ["family", "subtask", "pick", "role", "slot", "category", "model", "scene", "name"])
    return inv, cand, reps


def test_partner_goal_surface_becomes_an_anchor_of_its_target(tmp_path):
    inv, cand, reps = _world(tmp_path)
    surfaces, rep = HA.build_surfaces([cand, reps], inventory=inv, customs=str(tmp_path / "none.json"))
    by = {s.id: s for s in surfaces}
    assert by["S/bed_b/top"].key == "0.5000" and by["S/bed_b/top"].goal_subtasks == [ONTOP] and by["S/bed_b/top"].subtasks == []
    assert by["S/bed_c/top"].key == "0.6200"  # the representatives schema: the instance from ``name``
    assert sorted(rep["goal_surfaces_new"]) == ["S/bed_b/top", "S/bed_c/top"] and not rep["goal_missing_inventory"]
    ids = {s.id for s in HA.model_surfaces([cand, reps], "mug", "m1", surfaces)}
    assert ids == {"S/desk_a/top", "S/bed_b/top", "S/bed_c/top"}
    assert {s.id for s in HA.model_surfaces(cand, "cup", "m2", surfaces)} == {"S/desk_a/top"}  # another subtask's target: not its goal
    A = HA.AnchorSet.from_surfaces(surfaces)
    a, k = A.lookup(0.50)
    assert k == "exact" and a["key"] == "0.5000" and a["surfaces"] == ["S/bed_b/top"]
    wl = HA.workload([cand, reps], [("mug", "m1")], surfaces)["mug/m1"]
    assert wl["table_heights"] == ["0.5000", "0.6200", "0.7500"] and sorted(wl["goal_surfaces"]) == ["S/bed_b/top", "S/bed_c/top"]
    # off: the former rule (supports only)
    off, _ = HA.build_surfaces([cand, reps], inventory=inv, customs=str(tmp_path / "none.json"), goal_surfaces=False)
    assert [s.id for s in off] == ["S/desk_a/top"]


def test_existing_anchors_and_keys_are_identical():
    """The real role tables: every support surface keeps its height, key, status and subtasks; every former anchor keeps its key / height
    and resolves exactly to itself; the new anchors hold goal surfaces only."""
    old, _ = HA.build_surfaces([CAND, REPS], goal_surfaces=False)
    new, rep = HA.build_surfaces([CAND, REPS])
    nb = {s.id: s for s in new}
    for s in old:
        t = nb[s.id]
        assert (t.height, t.key, t.precision, t.status, t.subtasks, t.floor_z) == (s.height, s.key, s.precision, s.status, s.subtasks, s.floor_z)
    assert rep["n_support_surfaces"] == len(old) and len(new) == len(old) + len(rep["goal_surfaces_new"])
    A0, A1 = HA.AnchorSet.from_surfaces(old), HA.AnchorSet.from_surfaces(new)
    k1 = {a["key"]: a for a in A1.anchors}
    for a in A0.anchors:
        b = k1[a["key"]]
        assert b["height"] == a["height"] and b["precision"] == a["precision"] and set(a["surfaces"]) <= set(b["surfaces"])
        hit, kind = A1.lookup(a["height"])
        assert kind == "exact" and hit["key"] == a["key"]
    added = [a for a in A1.anchors if a["key"] not in {x["key"] for x in A0.anchors}]
    assert added and all(nb[sid].goal_subtasks and not nb[sid].subtasks for a in added for sid in a["surfaces"])
    # the ontop_large_01 goal (countertop_fjkase_0) is also a support row: its surface (and key) is the support's
    fj = nb["house_single_floor/countertop_fjkase_0/top"]
    assert ONTOP in fj.goal_subtasks and fj.key == {s.id: s for s in old}[fj.id].key


def test_an_uncovered_height_stays_uncovered(tmp_path):
    """No height union: a query more than the window from every anchor is uncovered with the goal anchors as without them; a goal anchor
    covers only its own window."""
    inv, cand, reps = _world(tmp_path)
    off, _ = HA.build_surfaces([cand, reps], inventory=inv, customs=str(tmp_path / "none.json"), goal_surfaces=False)
    on, _ = HA.build_surfaces([cand, reps], inventory=inv, customs=str(tmp_path / "none.json"))
    A0, A1 = HA.AnchorSet.from_surfaces(off), HA.AnchorSet.from_surfaces(on)
    for h in (0.40, 0.56, 0.68, 0.7399):  # > 1 cm from 0.50 / 0.62 / 0.75
        assert A0.lookup(h) == (None, "uncovered") and A1.lookup(h) == (None, "uncovered")
    assert A0.lookup(0.505) == (None, "uncovered")
    a, k = A1.lookup(0.505)
    assert k == "approx" and a["key"] == "0.5000"
    # the campaign's live anchors (v2maps_int1, 22 measured_live) + the ontop_large_01 goal top 0.9158: 0.9158 was uncovered (11.7 mm below
    # 0.9275), resolves exactly once added; 0.9050 (> 1 cm from 0.8874 / 0.9158) stays uncovered
    live = [0.3471, 0.4230, 0.4877, 0.4923, 0.5145, 0.6677, 0.6789, 0.6791, 0.6854, 0.7351, 0.7548, 0.7671, 0.7853, 0.7913, 0.8126, 0.8186,
            0.8668, 0.8728, 0.8874, 0.9275, 0.9336, 1.0505]

    def aset(hs):
        return HA.AnchorSet.from_surfaces([HA.Surface("S", f"s{i}", "top", "t", h, 1e-4, "live", 0.0, "f", status="measured_live")
                                           for i, h in enumerate(hs)])
    B0, B1 = aset(live), aset(live + [0.9158])
    assert B0.lookup(0.9158) == (None, "uncovered")
    a, k = B1.lookup(0.9158)
    assert k == "exact" and a["key"] == "0.9158"
    assert B0.lookup(0.9050) == (None, "uncovered") and B1.lookup(0.9050) == (None, "uncovered")
    for h in live:  # every former live anchor still resolves to itself
        a, k = B1.lookup(h)
        assert k == "exact" and abs(a["height"] - h) < 1e-12
