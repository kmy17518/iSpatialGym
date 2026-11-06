"""oracle-fix 2026-11-06 (survey1 disembedding_07 cellulose tape / disembedding_10 table fork, 0 / 4 demos each, symbolic_only): the
per-alias opt-ins ``generation.target_orientation_equivalences`` (the disembedding pose target carries the target's symmetry group) and
``generation.place_inhand_reaim`` (the oracle's support place re-aimed to the measured in-hand pose for that variant). CPU only."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

DIS = pytest.importorskip("ispatialgym.sampling.v2.tasks.disembedding")
from ispatialgym.oracle import execute as EX  # noqa: E402
from ispatialgym.sampling import customs as CU  # noqa: E402

TAPE = "disembedding-cellulose_tape-cork_1_coaster_1_cap_1-conference_table"
FORK = "disembedding-tablefork-tablefork_1_table_knife_1_tablespoon_1-breakfast_table"


class _GP:
    def __init__(self, tags, **gen):
        self.generation, self.tags = dict(gen), tags

    def gen(self, key, default=None):
        return self.generation.get(key, default)

    def orientation_equivalences(self, role, category, model):
        return CU.equivalence_group(self.tags) if self.tags and self.tags != ["none"] else None


class _S:
    target_equivalences = DIS.Disembedding.target_equivalences

    def __init__(self, tags, **gen):
        self.gp = _GP(tags, **gen)
        self.choice = SimpleNamespace(models={"target": SimpleNamespace(category="cellulose_tape", model="kavsnx")})

    def gen(self, key, default=None):
        return self.gp.gen(key, default)


def test_target_equivalences_off_keeps_the_identity_only_goal():
    S = _S
    assert S(["x2", "zinf"]).target_equivalences() is None  # today: no equivalences in the pose target


def test_target_equivalences_on_carry_the_tags_group():
    S = _S
    eq = S(["x2", "zinf"], target_orientation_equivalences=True).target_equivalences()
    assert eq is not None and len(eq) == 144 and eq[0] == [0.0, 0.0, 0.0, 1.0]  # zinf at 5 deg x the half turn about x
    assert S([], target_orientation_equivalences=True).target_equivalences() is None  # untagged / asymmetric (the fork): unchanged


def test_place_inhand_reaim_wanted_per_variant():
    cfg = EX.ExecuteConfig()
    assert cfg.place_inhand_reaim is False and not EX.place_inhand_reaim_wanted({}, cfg)
    assert EX.place_inhand_reaim_wanted({}, EX.ExecuteConfig(place_inhand_reaim=True))
    EX._VARIANT_GENERATION_CACHE["v_reaim"] = {"place_inhand_reaim": True}
    assert EX.place_inhand_reaim_wanted({"sidecar": SimpleNamespace(variant="v_reaim")}, cfg)
    EX._VARIANT_GENERATION_CACHE["v_plain"] = {}
    assert not EX.place_inhand_reaim_wanted({"sidecar": SimpleNamespace(variant="v_plain")}, cfg)


def test_customs_turn_the_knobs_on_for_the_two_aliases_only():
    with open(CU.DEFAULT_CUSTOMS_PATH) as f:
        data = json.load(f)
    assert data[FORK]["generation"]["place_inhand_reaim"] is True and "target_orientation_equivalences" not in data[FORK]["generation"]
    assert data[TAPE]["generation"]["place_inhand_reaim"] is True and data[TAPE]["generation"]["target_orientation_equivalences"] is True
    assert data[TAPE]["roles"]["target"]["orientation_equivalences"]["cellulose_tape/kavsnx"] == ["x2", "zinf"]
    on = [k for k, v in data.items() if isinstance(v, dict) and (v.get("generation") or {}).get("place_inhand_reaim")]
    assert sorted(on) == sorted([FORK, TAPE])
