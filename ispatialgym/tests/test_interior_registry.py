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


def test_the_plate_in_sink_variants_basin_is_registered():
    """i7-ft2 2026-11-08 (owner 7.2): drop_in_sink_awvzkn_0's basin (house_single_floor; the bar / countertop plate-in-sink variants), measured
    by interiors.measure (slurm 18377335): an open-top basin at 0.7316 m, the digest its interior maps are built with."""
    reg = IN.load_registry(IN.REGISTRY)
    sink = reg["drop_in_sink_awvzkn_0@basin"]
    assert sink.open_top and "top" in sink.open_sides and abs(sink.height_m - 0.7316) < 1e-6
    assert sink.digest() == "4571693d44c4"
    assert IN.spec_for_support("house_single_floor", "drop_in_sink_awvzkn_0", 0.7316) is not None
