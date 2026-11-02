"""Simulator-free tests of the alignment support rules (``support_rules``, owner 2026-10-13: alignment-axial supports are rectangular,
alignment-world_vertical walls are not transparent): the criteria on reference shapes, on the B4 top outlines and on the BEHAVIOR-1K
link tags of the seven scenes, and every place that applies them - ``ground_isg``, ``support_candidates``,
``GroundedProblem.override_support`` / ``apply_support_rules`` (``sample_isg``), ``multiply_isg`` and the two samplers."""

from __future__ import annotations

import copy
import json
import math
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

from ispatialgym.sampling import support_rules as SR
from ispatialgym.sampling.v2.problem import GroundedProblem, GroundingError, TemplateChoice

AXIAL, WALL = "alignment-axial", "alignment-world_vertical"
OFFICE = "office_cubicles_right"
needs_assets = pytest.mark.skipif(not os.path.isdir(os.path.join(SR.data_root(), "behavior-1k-assets", "objects", "walls")),
                                  reason="no BEHAVIOR-1K assets under $OMNIGIBSON_DATA_PATH")


def _box(w, h, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x * c - y * s, x * s + y * c) for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def _rounded(w, h, r, n=64):
    out = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0.0), (-w / 2 + r, h / 2 - r, 90.0), (-w / 2 + r, -h / 2 + r, 180.0), (w / 2 - r, -h / 2 + r, 270.0)):
        out += [(cx + r * math.cos(math.radians(a0 + 90 * t / n)), cy + r * math.sin(math.radians(a0 + 90 * t / n))) for t in range(n + 1)]
    return out


def test_rectangularity_of_reference_shapes():
    t = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    assert SR.rectangularity(_box(1.2, 0.6, yaw=0.5)) == pytest.approx(1.0)
    assert SR.rectangularity(np.stack([0.37 * np.cos(t), 0.37 * np.sin(t)], 1)) == pytest.approx(math.pi / 4, abs=2e-3)  # disc
    assert SR.rectangularity(np.stack([0.6 * np.cos(t), 0.3 * np.sin(t)], 1)) == pytest.approx(math.pi / 4, abs=2e-3)  # ellipse
    octagon = [(math.cos(a), math.sin(a)) for a in np.arange(8) * math.pi / 4 + math.pi / 8]
    assert SR.rectangularity(octagon) == pytest.approx(2 / (1 + math.sqrt(2)), abs=1e-6)
    assert SR.rectangularity(_rounded(1.2, 0.6, 0.30)) == pytest.approx(1 - (1 - math.pi / 4) * 0.6 / 1.2, abs=2e-3)  # stadium 2 : 1
    assert SR.rectangularity(_rounded(1.2, 0.6, 0.10)) == pytest.approx(1 - (4 - math.pi) * 0.01 / 0.72, abs=2e-3)  # rounded corners
    ell = [(0, 0), (1.4, 0), (1.4, 1.4), (0.6, 1.4), (0.6, 0.6), (0, 0.6)]  # an L-shaped corner desk, 0.6 m arms
    assert SR.rectangularity(ell) == pytest.approx(1 - 0.6 * 0.8 / 2 / 1.96, abs=1e-6)  # the hull cuts half the 0.6 x 0.8 m notch
    assert SR.rectangularity([(0, 0), (1, 0)]) == 0.0 and SR.rectangularity([(0, 0), (1, 0), (2, 0)]) == 0.0
    # the threshold: round, oval and L-shaped outlines below it, a top with 0.22 m corner radii on 1.2 x 0.6 m still above it
    assert max(math.pi / 4, 2 / (1 + math.sqrt(2)), SR.rectangularity(_rounded(1.2, 0.6, 0.30)), SR.rectangularity(ell)) < SR.MIN_TOP_RECTANGULARITY
    assert SR.rectangularity(_rounded(1.2, 0.6, 0.21)) > SR.MIN_TOP_RECTANGULARITY


def test_rule_of_and_names():
    assert SR.rule_of(AXIAL) == SR.RECTANGULAR_TOP and SR.rule_of(WALL) == SR.OPAQUE_WALL
    assert SR.rule_of("disembedding") is None and SR.rule_of(None) is None
    assert SR.split_object_name("walls_dmcumz_0") == ("walls", "dmcumz")
    assert SR.split_object_name("bottom_cabinet_no_top_gjrero_4") == ("bottom_cabinet_no_top", "gjrero")
    assert SR.split_object_name("walls") is None and SR.split_object_name("walls_x_y") is None and SR.split_object_name("") is None


def test_axial_rule_on_the_b4_tops():
    """The B4 tops of the seven scenes: the round lobby coffee table of alignment-axial#04 and the twelve-sided diner tables measure
    0.80, the oval Rs_int coffee table 0.90, the L-shaped cubicle desks 0.82; the rectangular axial candidates 0.966 and up."""
    r = SR.top_rectangularity
    assert r(OFFICE, "coffee_table_mtchfd_0") == pytest.approx(0.800, abs=0.005)
    assert r("restaurant_diner", "pedestal_table_nlfgwe_0") == pytest.approx(0.800, abs=0.005)
    assert r("Rs_int", "coffee_table_fqluyq_0") == pytest.approx(0.902, abs=0.005)
    assert r(OFFICE, "cubicle_hzalfx_0") == pytest.approx(0.819, abs=0.005)
    assert r("house_single_floor", "countertop_kelzer_0") == pytest.approx(0.966, abs=0.005)
    for scene, name in ((OFFICE, "coffee_table_mtchfd_0"), ("restaurant_diner", "pedestal_table_nlfgwe_0"), ("Rs_int", "coffee_table_fqluyq_0"),
                        (OFFICE, "cubicle_hzalfx_0"), (OFFICE, "desk_pfsjbc_0"), ("restaurant_diner", "bar_xxftww_0")):
        assert SR.exclusion_reason(AXIAL, scene, name).startswith("not_rectangular: "), name
        assert SR.exclusion_reason("disembedding", scene, name) is None
    for scene, name in (("house_single_floor", "breakfast_table_rhjoby_0"), (OFFICE, "desk_mdhelw_1"), ("house_single_floor", "countertop_kelzer_0"),
                        ("house_double_floor_lower", "coffee_table_koagbh_0"), ("house_double_floor_upper", "desk_iotfzl_0")):
        assert SR.exclusion_reason(AXIAL, scene, name) is None, name
    assert SR.exclusion_reason(AXIAL, "house_single_floor", "desk_nope_0").startswith("unchecked: ")
    assert SR.exclusion_reason(AXIAL, OFFICE, "floor") is None
    with pytest.raises(SR.SupportRuleError, match="not_rectangular"):
        SR.check_support(AXIAL, OFFICE, "coffee_table_mtchfd_0")


@needs_assets
def test_world_vertical_rule_on_the_link_tags():
    """Six of the 309 walls of the seven scenes carry a glass link: five glass partitions of office_cubicles_right (walls_dmcumz_0 of
    alignment-world_vertical#02 among them) and a glass panel of hotel_suite_large."""
    assert SR.glass_links("walls_dmcumz_0") == ["glass"] and SR.glass_links("walls_vkjkbk_0") == ["glass1", "glass2"]
    assert SR.glass_links("walls_gjtspo_0") == ["base_link"] and SR.glass_links("walls_wcngqp_0") == []
    assert SR.glass_links("walls_nope_0") is None and SR.glass_links("nope") is None
    for scene, name in ((OFFICE, "walls_dmcumz_0"), (OFFICE, "walls_iuufle_0"), (OFFICE, "walls_vaphll_0"), (OFFICE, "walls_vkjkbk_0"),
                        (OFFICE, "walls_woavqu_0"), ("hotel_suite_large", "walls_gjtspo_0")):
        assert SR.exclusion_reason(WALL, scene, name).startswith("transparent_wall: "), name
    # world_vertical#02's other candidates and the walls of the counted world-vertical instances
    for scene, name in (("house_double_floor_upper", "walls_ynhwyn_0"), (OFFICE, "walls_wcngqp_0"), ("house_single_floor", "walls_snssjy_0"),
                        ("house_single_floor", "walls_bpbucg_0"), ("house_single_floor", "walls_dhsptd_0"), ("restaurant_diner", "walls_ztyvde_0"),
                        (OFFICE, "walls_gkewle_0"), ("house_double_floor_lower", "walls_cvdpaq_0")):
        assert SR.exclusion_reason(WALL, scene, name) is None, name
    assert SR.exclusion_reason(WALL, OFFICE, "walls_nope_0").startswith("unchecked: ")
    assert SR.split_supports(WALL, OFFICE, ["walls_wcngqp_0", "walls_dmcumz_0"])[0] == ["walls_wcngqp_0"]


@needs_assets
def test_supports_of_the_counted_alignment_instances_are_admitted():
    """runs/s3_55tasks/SELECTED.tsv, counted = yes on 2026-10-13: axial#01 / #05 on breakfast_table_rhjoby_0, axial#03 on desk_mdhelw_1,
    world_vertical#01 / #03 / #04 / #05 on walls_dhsptd_0 / walls_ztyvde_0 / walls_gkewle_0 / walls_cvdpaq_0."""
    counted = [(AXIAL, "house_single_floor", "breakfast_table_rhjoby_0"), (AXIAL, OFFICE, "desk_mdhelw_1"),
               (WALL, "house_single_floor", "walls_dhsptd_0"), (WALL, "restaurant_diner", "walls_ztyvde_0"), (WALL, OFFICE, "walls_gkewle_0"),
               (WALL, "house_double_floor_lower", "walls_cvdpaq_0")]
    assert [SR.exclusion_reason(*c) for c in counted] == [None] * len(counted)


# ----------------------------------------------------------------------------------------------------------------
# GroundedProblem: override_support (sample_isg --support), apply_support_rules (sample_isg without --support)
# ----------------------------------------------------------------------------------------------------------------


def _axial_entry():
    surf = lambda name, row, z: {"name": name, "init": "ontop", "top_z": z, "row": row}  # noqa: E731
    block = lambda support, surfaces, room: {"rooms": [room], "bindings": {"support": support, "support_surfaces": surfaces, "floor": {"room": room}},  # noqa: E731
                                             "whitelist": {"target": {"magazine": ["tfovwy"]}}, "blacklist": {}}
    return {"iddl": "alignment/axial.iddl", "definition_id": 0,
            "roles": {"target": {"category": "magazine", "sampleable": True, "representative": "tfovwy"},
                      "support": {"category": "coffee_table", "sampleable": False}, "floor": {"sampleable": False}},
            "scenes": {OFFICE: block(["coffee_table_mtchfd_0", "desk_mdhelw_1"], [surf("coffee_table_mtchfd_0", "alignment-axial#04~T5", 0.45),
                                                                                  surf("desk_mdhelw_1", "x#01~T1", 0.79)], "lobby_0"),
                       "house_double_floor_upper": block(["coffee_table_osroux_0"], [surf("coffee_table_osroux_0", "alignment-axial#04~T3", 0.36)],
                                                         "living_room_0")},
            "generation": {}, "constraints": []}


def _wall_clock_entry(floor_room="lobby_0"):
    """The office block of the shipped alignment-world_vertical-wall_clock-walls (world_vertical#02's candidates 1 and 3)."""
    regions = [{"id": "lobby_0/walls_wcngqp_0/r02", "wall": "walls_wcngqp_0", "room": "lobby_0"},
               {"id": "meeting_room_0/walls_dmcumz_0/r00", "wall": "walls_dmcumz_0", "room": "meeting_room_0"}]
    return {"iddl": "alignment/world_vertical.iddl", "definition_id": 0,
            "roles": {"target": {"category": "wall_clock", "sampleable": True, "representative": "jerval"},
                      "support": {"category": "walls", "sampleable": False}, "floor": {"sampleable": False}},
            "scenes": {OFFICE: {"rooms": ["lobby_0", "meeting_room_0"],
                                "bindings": {"support": ["walls_wcngqp_0", "walls_dmcumz_0"], "support_regions": regions, "floor": {"room": floor_room}},
                                "whitelist": {"target": {"wall_clock": ["jerval"]}}, "blacklist": {}}},
            "generation": {}, "constraints": []}


def _gp(entry, subtask, **kw):
    return GroundedProblem(f"{subtask}-x-y", entry, subtask=subtask, **kw)


@needs_assets
def test_override_support_refuses_an_excluded_support():
    gp = _gp(_axial_entry(), AXIAL)
    with pytest.raises(GroundingError, match="excluded by the rectangular_top support rule .not_rectangular"):
        gp.override_support(OFFICE, "coffee_table_mtchfd_0", row="alignment-axial#04~T5")
    assert gp.override_support(OFFICE, "desk_mdhelw_1")["tag"] == "desk_mdhelw_1-T1"
    assert gp.override_support("house_double_floor_upper", "coffee_table_osroux_0")["support"] == "coffee_table_osroux_0"
    walls = _gp(_wall_clock_entry(), WALL)
    with pytest.raises(GroundingError, match="transparent_wall"):
        walls.override_support(OFFICE, "walls_dmcumz_0", row="meeting_room_0/walls_dmcumz_0/r00")
    assert walls.override_support(OFFICE, "walls_wcngqp_0")["tag"] == "walls_wcngqp_0-r02"
    # another subtask on the same block: no rule
    assert _gp(_axial_entry(), "disembedding").override_support(OFFICE, "coffee_table_mtchfd_0", row="alignment-axial#04~T5")["support"]


@needs_assets
def test_apply_support_rules_narrows_the_block(tmp_path):
    customs = tmp_path / "customs.json"
    variant = f"{WALL}-x-y"
    customs.write_text(json.dumps({variant: _wall_clock_entry()}))
    gp = GroundedProblem(variant, json.loads(customs.read_text())[variant], customs_path=str(customs), subtask=WALL)
    dropped = gp.apply_support_rules(OFFICE)
    assert list(dropped) == ["walls_dmcumz_0"] and dropped["walls_dmcumz_0"].startswith("transparent_wall: ")
    b = gp.bindings(OFFICE)
    assert b["support"] == ["walls_wcngqp_0"] and [r["id"] for r in b["support_regions"]] == ["lobby_0/walls_wcngqp_0/r02"]
    assert b["floor"] == {"room": "lobby_0"} and gp.rooms(OFFICE) == ["lobby_0", "meeting_room_0"]
    assert gp.apply_support_rules(OFFICE) == {}  # idempotent
    # the source customs file keeps the entry's own block, the copy gets the narrowed one
    gp.save_customs()
    gp.save_customs(str(tmp_path / "copy.json"))
    assert json.loads(customs.read_text())[variant]["scenes"] == _wall_clock_entry()["scenes"]
    assert json.loads((tmp_path / "copy.json").read_text())[variant]["scenes"][OFFICE]["bindings"]["support"] == ["walls_wcngqp_0"]
    # the floor of the dropped wall's room moves to the remaining regions' room
    moved = _gp(_wall_clock_entry(floor_room="meeting_room_0"), WALL)
    moved.apply_support_rules(OFFICE)
    assert moved.bindings(OFFICE)["floor"] == {"room": "lobby_0"} and moved.rooms(OFFICE) == ["lobby_0"]
    # an axial block: the round table and its surface entry leave
    ax = _gp(_axial_entry(), AXIAL)
    assert list(ax.apply_support_rules(OFFICE)) == ["coffee_table_mtchfd_0"] and ax.apply_support_rules("house_double_floor_upper") == {}
    assert ax.bindings(OFFICE)["support"] == ["desk_mdhelw_1"] and [e["name"] for e in ax.bindings(OFFICE)["support_surfaces"]] == ["desk_mdhelw_1"]
    # nothing left: an error; no rule, a floor-bound support or an unknown scene: untouched
    only = _wall_clock_entry()
    only["scenes"][OFFICE]["bindings"]["support"] = ["walls_dmcumz_0"]
    with pytest.raises(GroundingError, match="excludes every support"):
        _gp(only, WALL).apply_support_rules(OFFICE)
    other = _gp(_axial_entry(), "disembedding")
    assert other.apply_support_rules(OFFICE) == {} and other.bindings(OFFICE)["support"] == ["coffee_table_mtchfd_0", "desk_mdhelw_1"]
    fl = _axial_entry()
    fl["scenes"][OFFICE]["bindings"]["support"] = {"room": "lobby_0"}
    assert _gp(fl, AXIAL).apply_support_rules(OFFICE) == {} and _gp(_axial_entry(), AXIAL).apply_support_rules("Rs_int") == {}


# ----------------------------------------------------------------------------------------------------------------
# multiply_isg and the samplers
# ----------------------------------------------------------------------------------------------------------------


def _choice(gp, scene):
    return TemplateChoice(scene, {}, {}, gp.bindings(scene), gp.rooms(scene), gp.room_tag(scene), gp.support_tag(scene))


@needs_assets
def test_multiply_refuses_a_template_on_an_excluded_support():
    from ispatialgym.sampling import multiply_isg as MI

    gp = _gp(_axial_entry(), AXIAL)
    with pytest.raises(SystemExit, match="coffee_table_mtchfd_0"):
        MI.choice_for_template(gp, {"choice": _choice(gp, OFFICE).to_json()})
    gp.override_support(OFFICE, "desk_mdhelw_1")
    assert MI.choice_for_template(gp, {"choice": _choice(gp, OFFICE).to_json()}).bindings["support"] == ["desk_mdhelw_1"]


def _sampler(cls, gp, scene):
    s = object.__new__(cls)
    s.gp, s.scene, s.subtask, s.choice = gp, scene, gp.subtask, _choice(gp, scene)
    s.bound, s.template_facts = {}, {}
    s.world = NS(obj=lambda name: NS(name=name))
    s.bind_floor_v2 = lambda role="floor": NS(name="floor_obj")
    s.bind_scene_object = lambda role, obj: s.bound.__setitem__(role, obj.name)
    return s


@needs_assets
def test_samplers_refuse_excluded_supports():
    import torch as th

    from ispatialgym.sampling.v2.tasks.alignment_axial import AlignmentAxial
    from ispatialgym.sampling.v2.tasks.alignment_wall import AlignmentWall

    s = _sampler(AlignmentAxial, _gp(_axial_entry(), AXIAL), OFFICE)
    with pytest.raises(SR.SupportRuleError, match="coffee_table_mtchfd_0"):
        AlignmentAxial.bind_scene_roles(s)
    gp = _gp(_axial_entry(), AXIAL)
    gp.apply_support_rules(OFFICE)
    s = _sampler(AlignmentAxial, gp, OFFICE)
    AlignmentAxial.bind_scene_roles(s)
    assert s.bound["support"] == "desk_mdhelw_1"

    def spots(gp, unbound=False):
        s = _sampler(AlignmentWall, gp, OFFICE)
        if unbound:  # no wall names bound: the 24-ray search may hit any wall of the scene
            s.choice.bindings.pop("support")
            s.choice.bindings.pop("support_regions")
        s.wall_role = "support"
        s.patch_extents = lambda: (0.4, 0.4)
        s.obj = lambda role: NS(aabb=(th.zeros(3), th.zeros(3)))
        s._in_room = lambda xy: True
        hit = lambda wall, region: {"wall_name": wall, "region": region, "point": th.zeros(3), "normal": th.tensor([1.0, 0.0, 0.0])}  # noqa: E731
        s._region_spots = lambda regions, w, h, z: [hit(r["wall"], r["id"]) for r in regions]
        s._ray_spots = lambda w, h, z: [hit("walls_dmcumz_0", None), hit("walls_vkjkbk_0", None), hit("walls_rlchwi_0", None)]
        got = [(sp["wall_name"], sp["region"]) for sp in AlignmentWall.find_spots(s)]
        return got, s.template_facts["spot_search"]["excluded_walls"]

    got, excluded = spots(_gp(_wall_clock_entry(), WALL))
    assert got == [("walls_wcngqp_0", "lobby_0/walls_wcngqp_0/r02")] and list(excluded) == ["walls_dmcumz_0"]
    got, excluded = spots(_gp(_wall_clock_entry(), WALL), unbound=True)
    assert got == [("walls_rlchwi_0", None)] and sorted(excluded) == ["walls_dmcumz_0", "walls_vkjkbk_0"]


# ----------------------------------------------------------------------------------------------------------------
# ground_isg and support_candidates on the shipped sheet
# ----------------------------------------------------------------------------------------------------------------

#: the support rows of the shipped representative sheet the rules exclude (pick~T<triplet>: support)
EXCLUDED_ROWS = {"alignment-axial#03~T3": "cubicle_hzalfx_0", "alignment-axial#04~T4": "coffee_table_fqluyq_0",
                 "alignment-axial#04~T5": "coffee_table_mtchfd_0", "alignment-axial#05~T4": "pedestal_table_nlfgwe_0",
                 "alignment-world_vertical#02~T1": "walls_dmcumz_0"}


@needs_assets
def test_grounder_skips_excluded_rows_and_binds_no_excluded_support():
    from ispatialgym.sampling import ground_isg as GI
    from ispatialgym.sampling import iddl as I
    from ispatialgym.sampling.customs import Customs

    iddls, icl = I.load_all(), GI.ICL()
    for mode in ("representative", "all"):
        g = GI.Grounder(iddls, icl, Customs(), options=GI.Options(mode=mode, dry_run=True, representative_test=False, fallbacks=False))
        for st in (AXIAL, WALL):
            combos = g.combinations(st)
            assert combos, (mode, st)
            for comb in combos:
                for sc, objs in comb.roles["support"].scene_objects.items():
                    assert all(SR.exclusion_reason(st, sc, GI._support_object_name(o)) is None for o in objs), (mode, comb.label, sc)
            skipped = {lbl.rsplit("/", 1)[0]: why for lbl, why in g.report.skipped.get(st, [])}
            if mode == "representative":
                built = {c.pick for c in combos}
                for label, support in EXCLUDED_ROWS.items():
                    if label.startswith(st + "#"):
                        assert label not in built and skipped[label].startswith(f"{SR.SKIP_REASON}: {support}@"), label
                assert {"alignment-axial#04~T1", "alignment-axial#04~T2", "alignment-axial#04~T3"} <= built or st != AXIAL
                assert {f"alignment-world_vertical#02~T{i}" for i in range(2, 6)} <= built or st != WALL
            elif st == AXIAL:  # --all: the categories whose every candidate is excluded (diner pedestal tables, cubicles) are skipped
                assert any(why.startswith(f"{SR.SKIP_REASON}: pedestal_table_nlfgwe_") for why in skipped.values())
                assert any(why.startswith(f"{SR.SKIP_REASON}: cubicle_") for why in skipped.values())


@needs_assets
def test_pick_candidates_mark_the_excluded_supports():
    from ispatialgym.sampling import ground_isg as GI
    from ispatialgym.sampling import iddl as I
    from ispatialgym.sampling import support_candidates as SC
    from ispatialgym.sampling.v2.problem import DEFAULT_CUSTOMS

    with open(DEFAULT_CUSTOMS) as f:
        customs = json.load(f)
    picks = sorted({label.split("~")[0] for label in EXCLUDED_ROWS})
    cands = SC.pick_candidates(copy.deepcopy(customs), GI.ICL(), I.load_all(), picks=picks)
    for pick in picks:
        for c in cands[pick]:
            if c.label in EXCLUDED_ROWS:
                assert not c.ok and c.status.startswith(f"not_grounded: {SR.SKIP_REASON}: {EXCLUDED_ROWS[c.label]}@{c.scene}: "), (c.label, c.status)
            else:
                assert SR.SKIP_REASON not in c.status, (c.label, c.status)
