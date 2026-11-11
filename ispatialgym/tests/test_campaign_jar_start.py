from types import SimpleNamespace
import pytest
import torch
from ispatialgym.sampling.campaign_jar_start import VARIANT, connection_filter, rank_initial_edges, _rank_component_edges
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


def test_placement_component_order_keeps_disconnected_edges_as_fallback(monkeypatch):
    grid = torch.ones((9, 9), dtype=torch.bool)
    grid[4, :] = False
    monkeypatch.setattr(nav, 'eroded_map', lambda kit: grid)
    monkeypatch.setattr(nav, '_to_map', lambda kit, xy: tuple(int(v) for v in xy))
    s = SimpleNamespace(kit=object(), template_facts={})
    rows = [(SimpleNamespace(name=name), SimpleNamespace(base_pose2d=(x, y, 0)))
            for name, x, y in [('south', 1, 2), ('north_a', 7, 2), ('north_b', 7, 5)]]
    result = _rank_component_edges(s, iter(rows), [7, 6], nav)
    assert result == [rows[1], rows[2], rows[0]]
    assert all(any(pair is original for original in rows) for pair in result)
    assert s.template_facts['placement_component_edge_order']['all_original_candidates_retained']


def test_component_order_is_lazy_and_unchanged_by_default():
    def fail():
        raise AssertionError('default must not consume edge generator')
        yield
    candidates = fail()
    s = SimpleNamespace(gp=SimpleNamespace(gen=lambda key: {'connected_to_manipulation': True}))
    assert rank_initial_edges(s, candidates) is candidates


def test_component_hint_rejects_other_variant_or_unowned_proof():
    for variant, hint, message in [('other', {}, 'audited jar'),
                                 (VARIANT, {'source_path': '/tmp/proof.json', 'source_sha256': 'x'}, 'outside')]:
        s = SimpleNamespace(gp=SimpleNamespace(variant=variant, gen=lambda key: {'prefer_placement_component': hint}))
        with pytest.raises(ValueError, match=message):
            rank_initial_edges(s, [])


def test_component_hint_rejects_changed_diagnostic(monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(Path, 'read_bytes', lambda path: b'changed proof')
    hint = {'source_path': '/vision/u/minyeong/projects/isg_home/campaign_20261110/diagnostics/test/diagnosis.json',
            'source_sha256': '0' * 64}
    s = SimpleNamespace(gp=SimpleNamespace(variant=VARIANT, gen=lambda key: {'prefer_placement_component': hint}))
    with pytest.raises(ValueError, match='diagnostic changed'):
        rank_initial_edges(s, [])
