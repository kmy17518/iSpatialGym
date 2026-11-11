from types import SimpleNamespace
import pytest
import torch
from ispatialgym.sampling.campaign_jar_start import VARIANT, connection_filter
from ispatialgym.sampling.v2 import navigation as nav


def test_disconnected_free_start_is_redrawn_before_expensive_native_checks(monkeypatch):
    # Two large free regions separated by a solid wall, like the measured
    # countertop map. North B and C must not get a south initial A.
    grid = torch.ones((9, 9), dtype=torch.bool)
    grid[4, :] = False
    sampler = SimpleNamespace(gp=SimpleNamespace(variant=VARIANT,
        gen=lambda key: {'connected_to_manipulation': True}), kit=object(), template_facts={})
    monkeypatch.setattr(nav, 'eroded_map', lambda kit: grid)
    monkeypatch.setattr(nav, '_to_map', lambda kit, xy: tuple(int(v) for v in xy))
    stance = lambda x, y: SimpleNamespace(base_pose2d=(x, y, 0))
    check = connection_filter(sampler, [stance(7, 6), stance(6, 6)])
    assert check(stance(1, 6)) is False
    assert check(stance(1, 6)) is False  # Trunk variations reuse the same cell.
    assert check(stance(6, 4)) is True
    assert sampler.template_facts['connected_free_start']['tested_cells'] == 2
    assert sampler.template_facts['connected_free_start']['disconnected_cells'] == 1
    assert sampler.template_facts['connected_free_start']['full_navigation_check_required']


def test_default_draw_does_not_read_a_map_or_require_native_candidates(monkeypatch):
    monkeypatch.setattr(nav, 'eroded_map', lambda kit: (_ for _ in ()).throw(AssertionError('unexpected map read')))
    for cfg in [None, False, True, {}, {'connected_to_manipulation': False}]:
        sampler = SimpleNamespace(gp=SimpleNamespace(gen=lambda key: cfg))
        assert connection_filter(sampler, None)(None) is True


def test_only_explicit_exact_variant_can_use_connection_screen():
    for variant, cfg, message in [('other', True, 'audited'), (VARIANT, 'true', 'boolean')]:
        sampler = SimpleNamespace(gp=SimpleNamespace(variant=variant,
            gen=lambda key: {'connected_to_manipulation': cfg}))
        with pytest.raises(ValueError, match=message):
            connection_filter(sampler, [object()])
