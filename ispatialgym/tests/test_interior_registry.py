"""integration 2026-11-08 (lead decision: a data / config choice, not a gate): the repo's interior registry carries the specs the consolidated
root's interior maps were built with, so no job depends on ISPATIALGYM_INTERIORS. The digests below are the ``interior.digest`` of the maps
in $OFH/v2maps_int1 (fridge: jar exmsmm, i7-interior place maps; sink: plate luhkiz; bookcase jaysra: the hardback lean entries)."""
from ispatialgym.sampling import interiors as IN

INT1_INTERIOR_DIGESTS = {
    "fridge_petcxr_0@0.7064/j_left_door2.3": "0b539d8a83ab",  # door per piece (i7-lean a4a0fa1e measure, v2maps_i7fridge4)
    "drop_in_sink_lkklqs_0@basin": "bdb7f8c6f92e",  # the open-top basin (v2maps_i7sink2)
    "bookcase_jaysra_0@0.9472": "9d713ac87564",  # inside_large_01's bay (i7-overhang)
}
#: i7-goalmaps 2026-11-08 (owner decision 1: goal-surface maps of the IORS large predicates): measured specs of goal surfaces not (yet) in
#: v2maps_int1 - the hall tree's alcove board of touching_large_02 (interiors.measure, slurm 18379979)
GOAL_INTERIOR_DIGESTS = {
    "hall_tree_upkvgr_0@0.5809": "d6947c534b88",
}


def test_registry_digests_equal_the_consolidated_roots_interior_maps(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_INTERIORS", raising=False)
    reg = IN.load_registry()
    for name, digest in {**INT1_INTERIOR_DIGESTS, **GOAL_INTERIOR_DIGESTS}.items():
        assert name in reg, name
        assert reg[name].digest() == digest, (name, reg[name].digest(), digest)


def test_the_sink_is_an_open_top_basin_and_the_fridge_door_is_boxed_per_piece():
    reg = IN.load_registry(IN.REGISTRY)
    sink = reg["drop_in_sink_lkklqs_0@basin"]
    assert sink.open_top and "top" in sink.open_sides
    fr = reg["fridge_petcxr_0@0.7064/j_left_door2.3"]
    assert len(fr.extra_boxes) > 1 and all(b[0].startswith("door_left_door_") for b in fr.extra_boxes)


#: i7-t03specs 2026-11-08 (owner item 4): touching_03's other two bookcases, measured with ``interiors measure`` ($OFH/t03specs_20261108,
#: trunk_ee781f49; otwukr_1's boards in the narrower bay, ``--lateral -0.19``) - the digests of the interior maps of the lean entry
#: hardback__acbrnv__s1__lean_left_20_c5c708 staged in $OFH/t03specs_20261108/v2maps_t03specs (kmuacd's: the lean root's)
T03_INTERIOR_DIGESTS = {
    "bookcase_kmuacd_0@0.8354": "b58d4e5c790e",  # row T1
    "bookcase_opryhp_0@0.3722": "b8577f909128",  # row T6
    "bookcase_otwukr_1@0.3888": "41a3fa72d143",  # row T7
    "bookcase_otwukr_1@0.7204": "71e59f596aa4",  # row T8
}


def test_every_touching_03_board_has_a_spec(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_INTERIORS", raising=False)
    reg = IN.load_registry()
    for name, digest in T03_INTERIOR_DIGESTS.items():
        assert reg[name].digest() == digest, (name, reg[name].digest(), digest)
    # the customs' support rows (touching#03 T1 / T6 / T7 / T8) at their live board heights resolve to their own spec
    scene = "house_double_floor_upper"
    for support, h, name in (("bookcase_kmuacd_0", 0.8354, "bookcase_kmuacd_0@0.8354"), ("bookcase_opryhp_0", 0.3722, "bookcase_opryhp_0@0.3722"),
                             ("bookcase_otwukr_1", 0.3888, "bookcase_otwukr_1@0.3888"), ("bookcase_otwukr_1", 0.7204, "bookcase_otwukr_1@0.7204")):
        assert IN.spec_for_support(scene, support, h, registry=reg).name == name
    otw = reg["bookcase_otwukr_1@0.3888"]
    assert otw.back_thickness_m is not None and otw.open_sides == ["front"] and 0.29 < otw.width_m < 0.32  # one bay, its back panel

def test_the_hall_tree_board_is_found_at_the_touching_goal_height(monkeypatch):
    """i7-goalmaps: the spec serves touching_large_02's lookup (house_double_floor_lower / hall_tree_upkvgr_0 at the live 0.5809 m); its
    raster has a back panel and two open edges (front +u, side -v)."""
    monkeypatch.delenv("ISPATIALGYM_INTERIORS", raising=False)
    sp = IN.spec_for_support("house_double_floor_lower", "hall_tree_upkvgr_0", 0.5809)
    assert sp is not None and sp.name == "hall_tree_upkvgr_0@0.5809" and sp.back_thickness_m is not None
    assert tuple(sp.edge_distances) == (0.1, 0.2, 0.3) and abs(sp.board_setback_m - 0.3047) < 1e-9


#: i7-goalmaps 2026-11-08 (owner decision 5c.2): the specs the front-side switch must leave unchanged - this registry's plus the ones
#: i7-ft2 (awvzkn) and i7-t03specs (opryhp / otwukr_1) measured on their branches (digests of those branches' registries)
# integration-2 (merge of i7-goalmaps after i7-t03specs): + the touching_03 bookcases now in this registry (T03_INTERIOR_DIGESTS)
# integration-2 (merge of i7-ft2): + its sink awvzkn (the digest i7-goalmaps checked on / off on i7-ft2's registry)
ALL_KNOWN_DIGESTS = {**INT1_INTERIOR_DIGESTS, **GOAL_INTERIOR_DIGESTS, "bookcase_kmuacd_0@0.8354": "b58d4e5c790e", **T03_INTERIOR_DIGESTS,
                     "drop_in_sink_awvzkn_0@basin": "4571693d44c4", "tabletop_sink_vekaaf_0@basin": "0f063c1c46a9"}


class _Raster:
    def __init__(self, z, edges):
        self.z, self.headroom, self.mask, self.open_edges = z, None, None, list(edges)


def test_the_front_side_switch_changes_no_digest(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_INTERIORS", raising=False)
    for on in ("0", "1"):
        monkeypatch.setenv(IN.FRONT_SIDE_ENV, on)
        reg = IN.load_registry()
        for name, digest in ALL_KNOWN_DIGESTS.items():
            assert reg[name].digest() == digest, (on, name, reg[name].digest(), digest)


def test_front_plus_side_is_its_own_case_only_with_the_switch(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_INTERIORS", raising=False)
    reg = IN.load_registry()
    ht = reg["hall_tree_upkvgr_0@0.5809"]
    monkeypatch.setenv(IN.FRONT_SIDE_ENV, "0")  # off: the rule before - two open edges = front + back
    assert IN.open_sides_case(["+u", "-v"]) == "front_back"
    assert "open sides" in IN.live_mismatch(ht, _Raster(0.5809, ["+u", "-v"]))
    monkeypatch.setenv(IN.FRONT_SIDE_ENV, "1")
    assert IN.open_sides_case(["+u", "-v"]) == "front_side" and IN.open_sides_case(["+u", "-u"]) == "front_back"
    assert IN.open_sides_case(["+u", "-u", "+v"]) == "front_back" and IN.open_sides_case(["+u"]) == "front"
    assert IN.live_mismatch(ht, _Raster(0.5809, ["+u", "-v"])) is None
    # the measured rasters of the other specs give the same verdict with the switch on and off
    for name, edges in (("bookcase_jaysra_0@0.9472", ["+u", "-u"]), ("drop_in_sink_lkklqs_0@basin", ["+u", "-u", "+v", "-v"]),
                        ("fridge_petcxr_0@0.7064/j_left_door2.3", ["+u"])):
        sp = reg[name]
        verdicts = []
        for on in ("0", "1"):
            monkeypatch.setenv(IN.FRONT_SIDE_ENV, on)
            verdicts.append(IN.live_mismatch(sp, _Raster(sp.height_m, edges), joints=dict(sp.joint_values) or None))
        assert verdicts[0] == verdicts[1] is None, (name, verdicts)


def test_the_front_side_switch_is_on_by_default(monkeypatch):
    """owner decision 5q.1: ISPATIALGYM_INTERIOR_FRONT_SIDE is the default; ``=0`` restores the rule before."""
    monkeypatch.delenv(IN.FRONT_SIDE_ENV, raising=False)
    assert IN.front_side_on() and IN.open_sides_case(["+u", "-v"]) == "front_side"
    ht = IN.load_registry()["hall_tree_upkvgr_0@0.5809"]
    assert IN.live_mismatch(ht, _Raster(0.5809, ["+u", "-v"])) is None
    for off in ("0", "off", "false"):
        monkeypatch.setenv(IN.FRONT_SIDE_ENV, off)
        assert not IN.front_side_on()


def test_every_registry_spec_keeps_its_digest_and_verdict_on_and_off(monkeypatch):
    """5q.1: EVERY spec of the registry (not only a listed subset) has a known digest, unchanged with the switch on / off / unset, and the
    live-raster verdict for its own open sides (one open edge for a back panel, an opposite pair for an open back) is the same either way;
    only the hall tree's adjacent pair (front + side) changes verdict."""
    monkeypatch.delenv("ISPATIALGYM_INTERIORS", raising=False)
    names = None
    for on in (None, "0", "1"):
        if on is None:
            monkeypatch.delenv(IN.FRONT_SIDE_ENV, raising=False)
        else:
            monkeypatch.setenv(IN.FRONT_SIDE_ENV, on)
        reg = IN.load_registry()
        names = set(reg) if names is None else names
        assert set(reg) == names
        for name, sp in reg.items():
            assert name in ALL_KNOWN_DIGESTS, f"a registry spec without a recorded digest: {name} ({sp.digest()})"
            assert sp.digest() == ALL_KNOWN_DIGESTS[name], (on, name, sp.digest())
    reg = IN.load_registry()
    for name, sp in reg.items():
        edges = ["+u"] if sp.back_thickness_m is not None else ["+u", "-u"]
        verdicts = []
        for on in ("0", "1"):
            monkeypatch.setenv(IN.FRONT_SIDE_ENV, on)
            verdicts.append(IN.live_mismatch(sp, _Raster(sp.height_m, edges), joints=dict(sp.joint_values) or None))
        assert verdicts[0] == verdicts[1] is None, (name, verdicts)

def test_the_plate_in_sink_variants_basin_is_registered():
    """i7-ft2 2026-11-08 (owner 7.2): drop_in_sink_awvzkn_0's basin (house_single_floor; the bar / countertop plate-in-sink variants), measured
    by interiors.measure (slurm 18377335): an open-top basin at 0.7316 m, the digest its interior maps are built with."""
    reg = IN.load_registry(IN.REGISTRY)
    sink = reg["drop_in_sink_awvzkn_0@basin"]
    assert sink.open_top and "top" in sink.open_sides and abs(sink.height_m - 0.7316) < 1e-6
    assert sink.digest() == "4571693d44c4"
    assert IN.spec_for_support("house_single_floor", "drop_in_sink_awvzkn_0", 0.7316) is not None


def test_the_vekaaf_plate_in_sink_basin_is_registered():
    """i7-sink 2026-11-09 (owner A3b): tabletop_sink_vekaaf_0's basin (house_single_floor utility_room_0; the plate-in-sink variant
    plate-tabletop_sink_vekaaf_0-countertop), measured by interiors.measure (slurm 18409919): an open-top basin at 0.9559 m, open front and
    back (the live raster's four open edges), the digest its interior maps are built with."""
    reg = IN.load_registry(IN.REGISTRY)
    sink = reg["tabletop_sink_vekaaf_0@basin"]
    assert sink.open_top and "top" in sink.open_sides and abs(sink.height_m - 0.9559) < 1e-6 and sink.back_thickness_m is None
    assert sink.digest() == "0f063c1c46a9"
    assert IN.spec_for_support("house_single_floor", "tabletop_sink_vekaaf_0", 0.9559) is not None
    assert IN.live_mismatch(sink, _Raster(0.9559, ["+u", "-u", "+v", "-v"])) is None
