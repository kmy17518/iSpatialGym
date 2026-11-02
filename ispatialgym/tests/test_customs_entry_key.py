"""``customs.entry_key``: a variant's customs entry in this repository's list (keyed by variant) and in the eval release's
(keyed by subtask name), and the lookups that go through it (sampler, validation, protection, tolerance overrides)."""

from __future__ import annotations

import json

from ispatialgym.eval import tolerance_profile as TP
from ispatialgym.eval.names import subtask_name
from ispatialgym.oracle import protection as PROT
from ispatialgym.sampling import customs as CU

VARIANT = "inter_object_relation_satisfaction-ontop_small_scale-apple-plate-countertop"
SIBLING = "inter_object_relation_satisfaction-ontop_small_scale-apple-plate-breakfast_table"  # the same subtask, another support
LEGACY = "inter_object_relation_satisfaction-touching-box_of_cereal-box_of_cereal-bar"  # written before the small-scale rename


def test_variant_first_then_subtask():
    subtask = subtask_name(VARIANT)
    assert subtask == subtask_name(SIBLING)
    assert CU.entry_key({VARIANT: {}, SIBLING: {}}, VARIANT) == VARIANT  # this repository's list: each variant its own entry
    assert CU.entry_key({subtask: {}}, VARIANT) == subtask == CU.entry_key({subtask: {}}, SIBLING)  # the release's list
    assert CU.entry_key({VARIANT: {}, subtask: {}}, VARIANT) == VARIANT
    assert CU.entry_key({}, VARIANT) is None and CU.entry_key({VARIANT: {}}, None) is None
    assert CU.entry_key({VARIANT: {}}, "not-a-variant-of-any-iddl") is None


def test_a_variant_written_before_the_rename_finds_its_renamed_entry():
    renamed = LEGACY.replace("-touching-", "-touching_small_scale-")
    assert CU.entry_key({renamed: {}}, LEGACY) == renamed


def test_the_shipped_list_resolves_every_variant_to_its_own_entry():
    with open(CU.DEFAULT_CUSTOMS_PATH) as f:
        data = json.load(f)
    variants = [k for k in data if not k.startswith("_")]
    assert len(variants) > 83  # every grounded variant, not the release's 83 subtasks
    assert all(CU.entry_key(data, v) == v for v in variants)


def test_lookups_read_either_list(tmp_path):
    subtask = subtask_name(VARIANT)
    entry = {"protection": {"exclude": ["plate.n.04_1"]}, "evaluation": {"tolerances": {"pose": {"position": 0.02}}}}
    for key in (VARIANT, subtask):
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps({key: entry}))
        assert PROT.variant_override(VARIANT, customs_path=str(path)) == {"exclude": ["plate.n.04_1"]}
        assert TP.overrides_for(VARIANT, customs_path=str(path))["pose"] == {"position": 0.02}
    assert PROT.variant_override(VARIANT, customs_path=str(tmp_path / "missing.json")) == {}
