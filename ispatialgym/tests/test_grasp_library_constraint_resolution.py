"""Specialized consumers must not silently use a standard selected-model hint."""
import json
from types import SimpleNamespace

import pytest

from ispatialgym.sampling import grasp_library as GL


def setup(monkeypatch, tmp_path, entries):
    monkeypatch.setattr(GL, "library_root", lambda: str(tmp_path))
    monkeypatch.setattr(GL, "v2_entry", lambda key: SimpleNamespace(data={"constraint": entries[key]}) if key in entries else None)


@pytest.mark.parametrize("kind", ["insert", "i", "fl", "tabletop_insert"])
def test_specialized_role_rejects_explicit_standard_and_unsuffixed_fallback(monkeypatch, tmp_path, kind):
    setup(monkeypatch, tmp_path, {"food__model__ts": "tabletop", "food__model": "tabletop"})
    with pytest.raises(FileNotFoundError, match="incompatible.*tabletop"):
        GL.resolve_key("food", "model", "food__model__ts", [kind])


def test_specialized_role_selects_compatible_conventional_entry(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path, {"food__model__ts": "tabletop", "food__model__i": "tabletop_insert"})
    assert GL.resolve_key("food", "model", "food__model__ts", ["insert"]) == "food__model__i"


def test_flower_uses_metadata_and_explicit_constraint_is_narrower(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path, {"flower__model__r50": "flower_stem_top", "flower__model__i": "tabletop_insert"})
    assert GL.resolve_key("flower", "model", "flower__model__r50", ["insert"]) == "flower__model__r50"
    assert GL.resolve_key("flower", "model", "flower__model__r50", ["flower_stem_top"]) == "flower__model__r50"
    with pytest.raises(FileNotFoundError):
        GL.resolve_key("flower", "model", "flower__model__i", ["flower_stem_top"])


def test_ordinary_explicit_library_and_legacy_multi_kind_call_unchanged(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path, {"pillow__model__level": "tabletop_level_pinch", "food__model__ts": "tabletop"})
    assert GL.resolve_key("pillow", "model", "pillow__model__level", ["tabletop"]) == "pillow__model__level"
    assert GL.resolve_key("food", "model", "food__model__ts", ["tabletop", "insert"]) == "food__model__ts"


def test_existing_legacy_result_cannot_bypass_v2_specialization(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path, {"food__model__ts": "tabletop"})
    (tmp_path / "results").mkdir()
    (tmp_path / "results/food__model__ts.json").write_text(json.dumps({"constraint": "tabletop_insert"}))
    with pytest.raises(FileNotFoundError):
        GL.resolve_key("food", "model", "food__model__ts", ["insert"])
