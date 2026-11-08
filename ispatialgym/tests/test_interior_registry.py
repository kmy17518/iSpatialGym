"""integration 2026-11-08 (lead decision: a data / config choice, not a gate): the repo's interior registry carries the specs the consolidated
root's interior maps were built with, so no job depends on ISPATIALGYM_INTERIORS. The digests below are the ``interior.digest`` of the maps
in $OFH/v2maps_int1 (fridge: jar exmsmm, i7-interior place maps; sink: plate luhkiz; bookcase jaysra: the hardback lean entries)."""
from ispatialgym.sampling import interiors as IN

INT1_INTERIOR_DIGESTS = {
    "fridge_petcxr_0@0.7064/j_left_door2.3": "0b539d8a83ab",  # door per piece (i7-lean a4a0fa1e measure, v2maps_i7fridge4)
    "drop_in_sink_lkklqs_0@basin": "bdb7f8c6f92e",  # the open-top basin (v2maps_i7sink2)
    "bookcase_jaysra_0@0.9472": "9d713ac87564",  # inside_large_01's bay (i7-overhang)
}


def test_registry_digests_equal_the_consolidated_roots_interior_maps(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_INTERIORS", raising=False)
    reg = IN.load_registry()
    for name, digest in INT1_INTERIOR_DIGESTS.items():
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
