"""Train 2026-11-14: prepared per-configuration (scaled) floating libraries are reachable by their selected key, and a
training-scaled object is never served by the legacy scale-1 library."""
from types import SimpleNamespace

import pytest

from ispatialgym.sampling.v2.tasks import held_object as HO


def lib(category, model, constraint='floating'):
    return SimpleNamespace(data={'category': category, 'model': model, 'constraint': constraint}, scale=(1.0868,) * 3, path='/x/y/library.json')


def test_selected_key_accepted_for_same_floating_model(monkeypatch):
    monkeypatch.setattr(HO.GL, 'v2_entry', lambda k: lib('beeswax_candle', 'nhdnje') if k == 'campaign_selected__abc' else None)
    assert HO.floating_key('beeswax_candle', 'nhdnje', 'campaign_selected__abc') == 'campaign_selected__abc'


@pytest.mark.parametrize('entry', [None, lib('apple', 'netbsb'), lib('beeswax_candle', 'nhdnje', 'tabletop')])
def test_selected_key_refused_otherwise(monkeypatch, entry):
    monkeypatch.setattr(HO.GL, 'v2_entry', lambda k: entry)
    with pytest.raises(FileNotFoundError):
        HO.floating_key('beeswax_candle', 'nhdnje', 'campaign_selected__abc')


def test_legacy_library_refused_for_train_scaled_object(monkeypatch):
    monkeypatch.setattr(HO.GL, 'v2_entry', lambda k: None)
    monkeypatch.setattr(HO, '_train_scaled_model', lambda k: True)
    with pytest.raises(HO.V2ScaleMismatch):
        HO.check_v2_floating('beeswax_candle__nhdnje', (1.0868,) * 3)


def test_legacy_library_unchanged_without_train_scaling(monkeypatch):
    monkeypatch.setattr(HO.GL, 'v2_entry', lambda k: None)
    monkeypatch.setattr(HO, '_train_scaled_model', lambda k: False)
    HO.check_v2_floating('beeswax_candle__nhdnje', (1.0,) * 3)   # no exception: eval / unperturbed path unchanged
