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


def test_the_hall_tree_board_is_found_at_the_touching_goal_height(monkeypatch):
    """i7-goalmaps: the spec serves touching_large_02's lookup (house_double_floor_lower / hall_tree_upkvgr_0 at the live 0.5809 m); its
    raster has a back panel and two open edges (front +u, side -v) - ``live_mismatch`` reads two open edges as an open back (owner question)."""
    monkeypatch.delenv("ISPATIALGYM_INTERIORS", raising=False)
    sp = IN.spec_for_support("house_double_floor_lower", "hall_tree_upkvgr_0", 0.5809)
    assert sp is not None and sp.name == "hall_tree_upkvgr_0@0.5809" and sp.back_thickness_m is not None
    assert tuple(sp.edge_distances) == (0.1, 0.2, 0.3) and abs(sp.board_setback_m - 0.3047) < 1e-9
