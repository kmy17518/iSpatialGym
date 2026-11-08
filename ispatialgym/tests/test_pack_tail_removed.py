"""i7-contact3 (owner 3.7, 2026-11-08): the block design packing tail (the wide pack brace + its pushes) is removed; the block design uses
the board variant. Simulator-free: the oracle refuses a sidecar that still carries ``pack``, the customs supersede the pack-tail variants
by the board variant, and the sampler has no pack path left."""

from __future__ import annotations

import json
import os

import pytest

from ispatialgym import ASSETS_ROOT
from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.sidecar import Sidecar, SidecarError
from ispatialgym.tests.test_sidecar_programs import raw

BOARD = "configuration_matching-block_design-block_4-breakfast_table_board"
PACK_VARIANTS = ("configuration_matching-block_design-block_4-breakfast_table", "configuration_matching-block_design-block_4-coffee_table",
                 "configuration_matching-block_design-block_4-conference_table", "configuration_matching-block_design-block_4-desk")


def test_the_oracle_refuses_a_sidecar_with_a_pack_section():
    d = raw("configuration_matching-block_design")
    sc = Sidecar.from_dict(d)
    expand(sc.variant, sc)  # the board / grid program itself is untouched
    d["pack"] = {"stance": "manipulation", "stages": [{"brace": None, "pushes": [{"arm": "right"}]}]}
    sc = Sidecar.from_dict(d)
    with pytest.raises(SidecarError, match="packing tail"):
        expand(sc.variant, sc)


def test_the_pack_tail_variants_are_superseded_by_the_board_variant():
    with open(os.path.join(ASSETS_ROOT, "task_specs", "isg_customs.json")) as f:
        customs = json.load(f)
    for v in PACK_VARIANTS:
        e = customs[v]
        assert e["superseded_by"] == BOARD
        assert e["generation"]["block_pack_tail"] is True  # the marker the sampler refuses
        assert "place_gap_m" not in e["generation"] and "_pack_note" not in e["generation"]
    live = [k for k, e in customs.items() if "block_design" in k and isinstance(e, dict) and not e.get("superseded_by")]
    assert BOARD in live and not any((customs[k].get("generation") or {}).get("block_pack_tail") for k in live)


def test_the_sampler_has_no_pack_path():
    from ispatialgym.sampling.v2.stances import SidecarDraft
    from ispatialgym.sampling.v2.tasks import block_design as BD

    for name in ("_pack_tail", "_pack_stage", "_pack_pushes", "_pack_face_half", "_pack_posture_variants"):
        assert not hasattr(BD.BlockDesign, name)
    assert not hasattr(BD, "pack_brace_opening") and not hasattr(BD, "PACK_BRACE_END_MARGIN_M")
    assert "pack" not in SidecarDraft.__init__.__code__.co_names
