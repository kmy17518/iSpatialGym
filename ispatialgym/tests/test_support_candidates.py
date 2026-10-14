"""Simulator-free tests of the per-candidate support selection (owner 2026-10-11, the 55 representative tasks): one support row of a
pick per job (``GroundedProblem.override_support``, ``sample_isg --scene S --support NAME --support-row ROW``), the binding of the
selected support by every v2 task module that binds one, and the candidate listing / submission order (``support_candidates``,
eval85 ``pick_candidates.py``) on a fresh grounding of the shipped sheet."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from ispatialgym.eval.names import subtask_name
from ispatialgym.sampling.v2.problem import GroundedProblem, GroundingError, TemplateChoice, iter_template_choices, subtask_of_variant, support_row_tag

VARIANT = "inter_object_relation_satisfaction-ontop_small_scale-apple-plate-countertop"
SUBTASK = subtask_name(VARIANT)  # the customs key
S = "house_single_floor"
SURFACES = [{"name": "countertop_kelker_0", "init": "ontop", "top_z": 0.89, "row": "p#01~T3"},
            {"name": "countertop_kelzer_0", "init": "ontop", "top_z": 0.89, "row": "p#01~T2"},
            {"name": "countertop_ikwqer_0", "init": "ontop", "top_z": 0.93, "row": "p#01~T1"}]


def _tabletop_entry():
    """A grounded multi-support block as ``ground_isg`` merges it: the last-grounded row first, a ``candidates.csv`` object without a
    surface entry, the union of the rooms and the floor of one candidate."""
    return {"iddl": "inter_object_relation_satisfaction/ontop_small_scale.iddl", "definition_id": 0,
            "roles": {"target": {"category": "apple", "sampleable": True, "representative": "a"},
                      "partner": {"category": "plate", "sampleable": True, "representative": "p"},
                      "support": {"category": "countertop", "sampleable": False}, "floor": {"sampleable": False}},
            "scenes": {S: {"rooms": ["kitchen_0", "utility_room_0"],
                           "bindings": {"support": ["countertop_kelker_0", "countertop_kelzer_0", "countertop_ikwqer_0", "countertop_extra_0"],
                                        "support_surfaces": copy.deepcopy(SURFACES), "floor": {"room": "kitchen_0"}},
                           "whitelist": {"target": {"apple": ["a"]}, "partner": {"plate": ["p"]}}, "blacklist": {}, "_note": "kept"},
                       "Rs_int": {"rooms": ["kitchen_0"], "bindings": {"support": ["countertop_tpuwys_0"], "floor": {"room": "kitchen_0"}},
                                  "whitelist": {"target": {"apple": ["a"]}, "partner": {"plate": ["p"]}}, "blacklist": {}}},
            "generation": {"templates_max": 5, "instances_per_template": 1}, "constraints": []}


def _wall_entry():
    regions = [{"id": "living_room_0/walls_a_0/r00", "wall": "walls_a_0", "room": "living_room_0"},
               {"id": "corridor_0/walls_b_0/r01", "wall": "walls_b_0", "room": "corridor_0"},
               {"id": "corridor_0/walls_b_0/r02", "wall": "walls_b_0", "room": "corridor_0"}]
    return {"iddl": "alignment/world_vertical.iddl", "definition_id": 0,
            "roles": {"target": {"category": "painting", "sampleable": True, "representative": "d"}, "support": {"category": "walls", "sampleable": False},
                      "floor": {"sampleable": False}},
            "scenes": {S: {"rooms": ["living_room_0", "corridor_0"],
                           "bindings": {"support": ["walls_a_0", "walls_b_0"], "support_regions": regions, "floor": {"room": "living_room_0"}},
                           "whitelist": {"target": {"painting": ["d"]}}, "blacklist": {}}},
            "generation": {}, "constraints": []}


def _gp(entry=None, variant=VARIANT, **kw) -> GroundedProblem:
    """The problem of ``entry`` under ``variant``; its subtask from the variant (the synthetic entries share one ``iddl``)."""
    return GroundedProblem(variant, entry if entry is not None else _tabletop_entry(), subtask=subtask_of_variant(variant), **kw)


# ----------------------------------------------------------------------------------------------------------------
# GroundedProblem.override_support / TemplateChoice
# ----------------------------------------------------------------------------------------------------------------


def test_support_row_tag():
    assert support_row_tag("disembedding#05~T3") == "T3"
    assert support_row_tag("living_room_0/walls_dhsptd_0/r00") == "r00"
    assert support_row_tag("p#01~T1~T2") == "T1_T2" and support_row_tag("x-y#03") == "03"


def test_override_support_narrows_the_block_and_keeps_the_source(tmp_path):
    customs = tmp_path / "customs.json"
    customs.write_text(json.dumps({SUBTASK: _tabletop_entry(), "_comment": "c"}))
    gp = _gp(json.loads(customs.read_text())[SUBTASK], customs_path=str(customs))
    full = next(iter_template_choices(gp, S, np.random.default_rng(0)))
    assert full.template_id == "house_single_floor__partner-plate-p__target-apple-a" and full.support_tag is None
    rec = gp.override_support(S, "countertop_ikwqer_0", rooms=["utility_room_0"], note="test")
    assert rec == {"scene": S, "support": "countertop_ikwqer_0", "row": "p#01~T1", "init": "ontop", "top_z": 0.93, "surface": None, "regions": None,
                   "rooms": ["utility_room_0"], "floor_room": "utility_room_0", "tag": "countertop_ikwqer_0-T1"}
    # the block of that support alone: first (only) name, its surface entry, its room and floor (the merged block's floor was kitchen_0)
    assert gp.bindings(S) == {"support": ["countertop_ikwqer_0"], "support_surfaces": [SURFACES[2]], "floor": {"room": "utility_room_0"}}
    assert gp.rooms(S) == ["utility_room_0"] and gp.support_tag(S) == "countertop_ikwqer_0-T1" and gp.room_tag(S) is None
    assert gp.entry["scenes"][S]["_note"] == "kept" and gp.entry["scenes"][S]["_candidate"]["note"] == "test"
    assert gp.entry["scenes"]["Rs_int"] == _tabletop_entry()["scenes"]["Rs_int"] and gp.support_tag("Rs_int") is None
    choice = next(iter_template_choices(gp, S, np.random.default_rng(0)))
    assert choice.template_id == "house_single_floor__support-countertop_ikwqer_0-T1__partner-plate-p__target-apple-a"
    assert choice.bindings["support"] == ["countertop_ikwqer_0"] and choice.rooms == ["utility_room_0"] and choice.support_tag == "countertop_ikwqer_0-T1"
    back = TemplateChoice.from_json(json.loads(json.dumps(choice.to_json())))
    assert back.template_id == choice.template_id and back.support_tag == choice.support_tag
    assert TemplateChoice.from_json({k: v for k, v in choice.to_json().items() if k != "support_tag"}).template_id == full.template_id
    # what the candidate run learns goes to the copy under --out; the source keeps the entry's own block
    gp.add_blacklist(S, "target", "apple", "a", "failed at the utility-room counter")
    gp.save_customs(str(tmp_path / "copy.json"))
    gp.save_customs()
    src = json.loads(customs.read_text())
    assert src["_comment"] == "c" and src[SUBTASK]["scenes"] == _tabletop_entry()["scenes"]
    cp = json.loads((tmp_path / "copy.json").read_text())[SUBTASK]["scenes"][S]
    assert cp["bindings"]["support"] == ["countertop_ikwqer_0"] and cp["blacklist"]["target"]["apple"]["a"].startswith("failed") and cp["_candidate"]["tag"]
    # another candidate re-narrows from the entry's own block; the merged block's floor room stays when it is the support's
    rec = gp.override_support(S, "countertop_kelker_0", row="p#01~T3", rooms=["kitchen_0"])
    assert gp.bindings(S) == {"support": ["countertop_kelker_0"], "support_surfaces": [SURFACES[0]], "floor": {"room": "kitchen_0"}}
    assert rec["tag"] == "countertop_kelker_0-T3" and gp.whitelist(S, "target") == {"apple": ["a"]}
    # a support of the block without a surface entry: no surface, the block's rooms and floor when its rooms are unknown
    rec = gp.override_support(S, "countertop_extra_0")
    assert gp.bindings(S) == {"support": ["countertop_extra_0"], "floor": {"room": "kitchen_0"}} and gp.rooms(S) == ["kitchen_0", "utility_room_0"]
    assert rec["row"] is None and rec["tag"] == "countertop_extra_0" and rec["init"] is None


def test_override_support_rows_walls_floor_partner_and_errors():
    # one support with two surface rows (two picks sharing the variant): the row is required
    e = _tabletop_entry()
    e["scenes"][S]["bindings"]["support_surfaces"].append({"name": "countertop_kelker_0", "init": "ontop", "top_z": 0.89, "row": "q#02~T1"})
    gp = _gp(e)
    with pytest.raises(GroundingError, match="name one"):
        gp.override_support(S, "countertop_kelker_0")
    assert gp.override_support(S, "countertop_kelker_0", row="q#02~T1")["tag"] == "countertop_kelker_0-T1"
    assert gp.bindings(S)["support_surfaces"] == [{"name": "countertop_kelker_0", "init": "ontop", "top_z": 0.89, "row": "q#02~T1"}]
    for bad in (dict(scene=S, support="countertop_kelker_0", row="p#01~T9"), dict(scene=S, support="countertop_nope_0"),
                dict(scene="Rs_int", support="countertop_kelker_0"), dict(scene="restaurant_diner", support="countertop_kelker_0")):
        with pytest.raises(GroundingError):
            _gp().override_support(**bad)
    held = {"iddl": "object_scaling/object_scaling.iddl", "roles": {"target": {"category": "apple", "sampleable": True}, "floor": {"sampleable": False}},
            "scenes": {S: {"rooms": ["living_room_0"], "bindings": {"floor": {"room": "living_room_0"}}}}}
    with pytest.raises(GroundingError, match="no support role"):
        _gp(held, variant="object_scaling-apple").override_support(S, "floor")
    # a wall: its regions (a region id as the row), the region's room whatever the object's rooms say
    walls = _gp(_wall_entry(), variant="alignment-world_vertical-painting-walls")
    rec = walls.override_support(S, "walls_b_0", rooms=["living_room_0", "corridor_0"])
    assert walls.bindings(S)["support"] == ["walls_b_0"] and [r["id"] for r in walls.bindings(S)["support_regions"]] == ["corridor_0/walls_b_0/r01", "corridor_0/walls_b_0/r02"]
    assert walls.rooms(S) == ["corridor_0"] and walls.bindings(S)["floor"] == {"room": "corridor_0"} and rec["tag"] == "walls_b_0" and rec["row"] is None
    rec = walls.override_support(S, "walls_b_0", row="corridor_0/walls_b_0/r02")
    assert [r["id"] for r in walls.bindings(S)["support_regions"]] == ["corridor_0/walls_b_0/r02"] and rec["tag"] == "walls_b_0-r02"
    assert walls.override_support(S, "walls_a_0")["tag"] == "walls_a_0-r00" and walls.rooms(S) == ["living_room_0"]
    # a floor-bound support (small-scale articulation of a tall target, §5.4): selected as "floor", the block unchanged
    fl = _tabletop_entry()
    fl["scenes"][S]["bindings"] = {"support": {"room": "kitchen_0"}, "floor": {"room": "kitchen_0"}}
    gp = _gp(fl)
    assert gp.override_support(S, "floor", rooms=["dining_room_0"])["tag"] == "floor"
    assert gp.bindings(S) == fl["scenes"][S]["bindings"] and gp.rooms(S) == fl["scenes"][S]["rooms"]
    for kw in (dict(support="countertop_kelker_0"), dict(support="floor", row="x")):
        with pytest.raises(GroundingError):
            _gp(fl).override_support(S, **kw)
    # a scene partner anchors the floor room (§5.6: large-scale IORS), the support's rooms are added
    lg = _tabletop_entry()
    lg["roles"]["partner"] = {"category": "bed", "sampleable": False}
    lg["scenes"][S]["bindings"] = {"partner": ["bed_rrcvaq_1"], "support": ["shelf_amyper_1", "desk_bhkhxo_0"], "floor": {"room": "childs_room_1"}}
    gp = _gp(lg, variant="inter_object_relation_satisfaction-under_large_scale-sandal-bed-shelf")
    rec = gp.override_support(S, "shelf_amyper_1", rooms=["corridor_0"])
    assert gp.bindings(S)["floor"] == {"room": "childs_room_1"} and gp.rooms(S) == ["childs_room_1", "corridor_0"] and rec["floor_room"] == "childs_room_1"
    assert gp.bindings(S)["partner"] == ["bed_rrcvaq_1"] and gp.bindings(S)["support"] == ["shelf_amyper_1"]


# ----------------------------------------------------------------------------------------------------------------
# sample_isg plumbing
# ----------------------------------------------------------------------------------------------------------------


def _args(**kw):
    d = {"seed": None, "seed_split": None, "seed_index": 0, "scene": None, "room": None, "scene_from_split": False, "support": None, "support_row": None}
    d.update(kw)
    return NS(**d)


def _stable(**rooms):
    return {"objects_info": {"init_info": {name: {"args": {"in_rooms": r}} for name, r in rooms.items()}}}


def test_sample_isg_candidate_plumbing(tmp_path):
    from ispatialgym.sampling import sample_isg as SI

    gp = _gp()
    assert SI.variant_candidate(_args(scene=S), gp) is None and SI.variant_candidate(NS(scene=S), gp) is None
    rec = SI.variant_candidate(_args(scene=S, support="countertop_kelzer_0", support_row="p#01~T2"), gp, _stable(countertop_kelzer_0=["kitchen_0"]))
    assert rec["source"] == "--support" and rec["rooms"] == ["kitchen_0"] and rec["tag"] == "countertop_kelzer_0-T2"
    assert "--support countertop_kelzer_0 --support-row p#01~T2" in gp.entry["scenes"][S]["_candidate"]["note"]
    # an object the stable scene does not have: the block's rooms stay
    gp2 = _gp()
    assert SI.variant_candidate(_args(scene=S, support="countertop_extra_0"), gp2, _stable())["rooms"] == ["kitchen_0", "utility_room_0"]
    assert SI.object_rooms(_stable(a=["x", "", "y"]), "a") == ["x", "y"] and SI.object_rooms({}, "a") == []
    for argv in (["--variant", VARIANT, "--support", "countertop_kelzer_0"], ["--variant", VARIANT, "--scene", S, "--support-row", "p#01~T2"],
                 ["--variant", VARIANT, "--scene", S, "--room", "kitchen_0", "--support", "countertop_kelzer_0"]):
        with pytest.raises(SystemExit):
            SI.main(argv)
    # templates already produced under --out count for their own candidate only
    choice = next(iter_template_choices(gp, S, np.random.default_rng(0)))
    d = tmp_path / gp.variant / choice.template_id
    d.mkdir(parents=True)
    (d / "template_meta.json").write_text(json.dumps({"scene": S, "rooms": gp.rooms(S), "choice": choice.to_json()}))
    assert SI._existing_templates(str(tmp_path), gp, S) == [{"partner": "plate/p", "target": "apple/a"}]
    other = _gp()
    other.override_support(S, "countertop_kelker_0", row="p#01~T3", rooms=["kitchen_0"])
    assert SI._existing_templates(str(tmp_path), other, S) == [] and SI._existing_templates(str(tmp_path), _gp(), S) == []


def test_candidate_world_key_on_the_real_scene():
    """house_single_floor's two desks in two child rooms: the merged block loads both rooms, removes the swivel chair at one desk and
    puts the floor in one room; each candidate loads its own room with its own floor and chairs."""
    from ispatialgym.sampling import sample_isg as SI

    try:
        stable = SI.stable_scene_json(S)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no stable scene json: {e!r}")
    entry = _tabletop_entry()
    entry["roles"]["support"]["category"] = "desk"
    entry["scenes"][S] = {"rooms": ["childs_room_0", "childs_room_1"], "bindings": {"support": ["desk_xhjsub_0", "desk_bhkhxo_0"], "floor": {"room": "childs_room_0"}},
                          "whitelist": {"target": {"apple": ["a"]}, "partner": {"plate": ["p"]}}, "blacklist": {}}
    full = SI.world_key(_gp(copy.deepcopy(entry)), S, stable)
    assert full[1] == ("childs_room_0", "childs_room_1")
    for desk in ("desk_bhkhxo_0", "desk_xhjsub_0"):
        gp = _gp(copy.deepcopy(entry))
        rec = SI.variant_candidate(_args(scene=S, support=desk), gp, stable)
        key = SI.world_key(gp, S, stable)
        assert key[1] == tuple(SI.object_rooms(stable, desk)) == (rec["floor_room"],) and gp.bindings(S)["floor"] == {"room": rec["floor_room"]}
        assert set(key[2]) <= set(full[2])


# ----------------------------------------------------------------------------------------------------------------
# every v2 task module that binds a support binds the selected one
# ----------------------------------------------------------------------------------------------------------------


def _choice(gp, scene=S):
    return TemplateChoice(scene, {}, {}, gp.bindings(scene), gp.rooms(scene), gp.room_tag(scene), gp.support_tag(scene))


def _stub(cls, gp, scene=S):
    s = object.__new__(cls)
    s.gp, s.scene, s.subtask, s.choice = gp, scene, gp.subtask, _choice(gp, scene)
    s.bound = {}
    s.objects = {"floor_obj": NS(name="floor_obj")}
    s.world = NS(obj=lambda name: NS(name=name))
    s.bind_floor_v2 = lambda role="floor": NS(name="floor_obj")
    s.bind_scene_object = lambda role, obj: s.bound.__setitem__(role, obj.name)
    s.template_facts = {}
    return s


@pytest.mark.parametrize("module,cls,variant", [
    ("iors_small", "IORSSmall", VARIANT),
    ("disembedding", "Disembedding", "disembedding-beaker-beaker_2_beer_glass_1-countertop"),
    ("block_design", "BlockDesign", "configuration_matching-block_design-block_4-countertop"),
    ("attachment", "Attachment", "configuration_matching-attachment-cork-wine_bottle-countertop"),
    ("articulation_small", "ArticulationSmall", "configuration_matching-articulation_open_small_scale-blender-countertop"),
    ("alignment_axial", "AlignmentAxial", "alignment-axial-board_game-countertop"),
])
def test_v2_modules_bind_the_selected_support(module, cls, variant):
    import importlib

    Cls = getattr(importlib.import_module(f"ispatialgym.sampling.v2.tasks.{module}"), cls)
    full = _stub(Cls, _gp(variant=variant))
    Cls.bind_scene_roles(full)
    assert full.bound["support"] == "countertop_kelker_0"  # the merged block: whatever row was grounded last
    for name, row in (("countertop_ikwqer_0", "p#01~T1"), ("countertop_kelzer_0", "p#01~T2")):
        gp = _gp(variant=variant)
        gp.override_support(S, name, row=row)
        s = _stub(Cls, gp)
        Cls.bind_scene_roles(s)
        assert s.bound["support"] == name


def test_articulation_small_floor_support_candidate():
    from ispatialgym.sampling.v2.tasks.articulation_small import ArticulationSmall

    e = _tabletop_entry()
    e["scenes"][S]["bindings"] = {"support": {"room": "kitchen_0"}, "floor": {"room": "kitchen_0"}}
    gp = _gp(e, variant="configuration_matching-articulation_open_small_scale-microwave-floor")
    gp.override_support(S, "floor")
    s = _stub(ArticulationSmall, gp)
    s.floor_support, s.floor_room = False, None
    ArticulationSmall.bind_scene_roles(s)
    assert s.floor_support and s.floor_room == "kitchen_0" and "support" not in s.bound


def test_alignment_wall_spots_only_on_the_selected_region():
    import torch as th

    from ispatialgym.sampling.v2.tasks.alignment_wall import AlignmentWall

    def spots(gp):
        s = _stub(AlignmentWall, gp)
        s.wall_role = "support"
        s.patch_extents = lambda: (0.4, 0.4)
        s.obj = lambda role: NS(aabb=(th.zeros(3), th.zeros(3)))
        s._in_room = lambda xy: True
        s._region_spots = lambda regions, w, h, z: [{"wall_name": r["wall"], "region": r["id"], "point": th.zeros(3), "normal": th.tensor([1.0, 0.0, 0.0])}
                                                    for r in regions]
        return [(sp["wall_name"], sp["region"]) for sp in AlignmentWall.find_spots(s)]

    variant = "alignment-world_vertical-painting-walls"
    assert {w for w, _ in spots(_gp(_wall_entry(), variant=variant))} == {"walls_a_0", "walls_b_0"}
    gp = _gp(_wall_entry(), variant=variant)
    gp.override_support(S, "walls_b_0", row="corridor_0/walls_b_0/r02")
    assert spots(gp) == [("walls_b_0", "corridor_0/walls_b_0/r02")]


def test_iors_large_init_surface_is_the_selected_support():
    from ispatialgym.sampling.v2.tasks.iors_large import IORSLarge

    variant = "inter_object_relation_satisfaction-ontop_large_scale-soap_dispenser-countertop-countertop"
    e = _tabletop_entry()
    e["roles"]["partner"] = {"category": "countertop", "sampleable": False}
    e["scenes"][S]["bindings"] = {"partner": ["countertop_fjkase_0"], "support": ["countertop_sjxber_0", "countertop_ikwqer_0"],
                                  "support_surfaces": [{"name": "countertop_sjxber_0", "init": "ontop", "top_z": 0.93, "row": "o#01~T2"},
                                                       {"name": "countertop_ikwqer_0", "init": "ontop", "top_z": 0.93, "row": "o#01~T1"}],
                                  "floor": {"room": "utility_room_0"}}

    def candidates(gp):
        assert gp.subtask == "inter_object_relation_satisfaction-ontop_large_scale"
        s = _stub(IORSLarge, gp)
        s.relation = "ontop"
        s.obj = lambda role: NS(name="countertop_fjkase_0" if role == "partner" else role)
        kind, objs = IORSLarge.surface_candidates(s)
        return kind, [o.name for o in objs]

    assert candidates(_gp(copy.deepcopy(e), variant=variant)) == ("support", ["countertop_sjxber_0", "countertop_ikwqer_0"])
    gp = _gp(copy.deepcopy(e), variant=variant)
    gp.override_support(S, "countertop_ikwqer_0", row="o#01~T1", rooms=["utility_room_0"])
    assert candidates(gp) == ("support", ["countertop_ikwqer_0"]) and gp.bindings(S)["support_surfaces"][0]["row"] == "o#01~T1"


# ----------------------------------------------------------------------------------------------------------------
# support_candidates on the shipped sheet
# ----------------------------------------------------------------------------------------------------------------

#: picks of the new sheet (2026-10-11) frozen for these tests, one per support kind: tabletop surfaces (three supports of one category in
#: one scene, two desks in two rooms), wall regions, no support (a scene target), small-scale articulation, init=inside and floor rows
#: (large-scale ontop) and a pick the grounder skips (nextto#04)
