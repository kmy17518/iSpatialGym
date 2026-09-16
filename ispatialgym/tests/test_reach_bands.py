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
