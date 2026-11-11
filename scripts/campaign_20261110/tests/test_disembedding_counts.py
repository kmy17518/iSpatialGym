import pytest

from scripts.campaign_20261110.materialize_disembedding_counts import count_tuples, solve_counts, existing_pool_for_counts


def profile(rows=None):
    return {'schema': 'isg-native-tray-coverage-profile-v1', 'fixed_layout': False, 'source_floor_area_m2': .1,
            'target': {'category': 'box_of_shampoo', 'model': 'target', 'area_m2': .01, 'scale': [1., 1., 1.]},
            'distractors': rows or [{'category': 'box_of_shampoo', 'model': 'distractor', 'area_m2': .01, 'scale': [1., 1., 1.]}]}


def test_exact_counts_cover_all_bins_without_scaling_or_packing_inference():
    result = solve_counts(profile(), max_distractors=8)
    assert [row['bucket'] for row in result] == [0, 1, 2]
    for row in result:
        assert row['predicted_native_coverage'] == pytest.approx((.01 + .01 * row['n_distractors']) / .1)
        low, high = row['crowdedness_range']
        assert low <= row['predicted_native_coverage'] < high


def test_one_selected_model_per_category_allows_exact_native_whitelist():
    rows = [{'category': 'box', 'model': 'a', 'area_m2': .008, 'scale': [1., 1., 1.]},
            {'category': 'box', 'model': 'b', 'area_m2': .02, 'scale': [1., 1., 1.]}]
    for row in solve_counts(profile(rows)):
        assert len(row['counts']) == 1


def test_duplicate_observed_copies_do_not_duplicate_maximum_pool_candidates():
    row = profile()['distractors'][0]
    assert solve_counts(profile([row, dict(row), dict(row)])) == solve_counts(profile([row]))


def test_fixed_layout_and_inconsistent_native_measurements_fail_closed():
    p = profile(); p['fixed_layout'] = True
    with pytest.raises(ValueError):
        solve_counts(p)
    p = profile(); p['distractors'].append({**p['distractors'][0], 'area_m2': .02})
    with pytest.raises(ValueError):
        solve_counts(p)


def test_area_budget_and_finite_maximum_are_enforced():
    assert all(row['n_distractors'] <= 3 and row['predicted_native_coverage'] <= .4 for row in solve_counts(profile(), 3, .4))
    tuples = list(count_tuples(4, 4))
    assert len(tuples) == len(set(tuples)) == 70
    assert all(sum(row) <= 4 for row in tuples)


def test_large_single_category_pool_is_bounded_without_cartesian_explosion():
    p = profile(); p['target']['area_m2'] = .001; p['distractors'][0]['area_m2'] = .001
    result = solve_counts(p, max_distractors=48)
    assert result[-1]['bucket'] == 2 and result[-1]['n_distractors'] > 24
    p['distractors'] = [{**p['distractors'][0], 'category': str(i)} for i in range(6)]
    with pytest.raises(ValueError, match='enumeration budget'):
        solve_counts(p, max_distractors=48)


def test_existing_pool_keeps_inactive_candidates_and_rejects_capacity_extension(tmp_path):
    from scripts.campaign_20261110.tests.test_binding_pools import variant, pool
    template = variant(tmp_path, count=3)
    maximum, choices, path = existing_pool_for_counts(template, 9, tmp_path)
    assert maximum.fingerprint == pool().fingerprint
    assert ('distractor', 'tissue', 't') in choices
    assert path.name == 'pool.json'
    with pytest.raises(ValueError, match='immutable maximum pool'):
        existing_pool_for_counts(template, 10, tmp_path)


def test_fewer_bodies_is_an_explicit_prediction_only_with_strict_bucket_margin():
    p = profile([
        {'category': category, 'model': model, 'area_m2': area, 'scale': [1., 1., 1.]}
        for category, model, area in [('cap', 'iqeyba', .004373847741942424),
            ('coaster', 'httqaj', .006213426806118583), ('cork', 'uyceta', .0009951372280284812)]])
    p['source_floor_area_m2'] = .0927
    p['target']['area_m2'] = .001965467577987906
    original = solve_counts(p)
    assert [row['n_distractors'] for row in original[1:]] == [12, 11]
    assert original == solve_counts(p, selection_policy='bin_centre')
    changed = solve_counts(p, selection_policy='fewest_bodies')
    assert [row['n_distractors'] for row in changed[1:]] == [5, 7]
    for row in changed:
        low, high = row['crowdedness_range']
        assert low + .02 <= row['predicted_native_coverage'] < high - .02
        assert row['selection_policy'] == 'fewest_bodies'
        assert sum(row['counts'].values()) == row['n_distractors']
        assert set(row['counts']).issubset({'cap/iqeyba', 'coaster/httqaj', 'cork/uyceta'})
    with pytest.raises(ValueError, match='unknown count selection'):
        solve_counts(p, selection_policy='relax_bucket')


def test_fewer_bodies_reports_missing_bucket_instead_of_crossing_margin():
    p = profile([{'category': 'box', 'model': 'a', 'area_m2': .0445, 'scale': [1., 1., 1.]}])
    p['target']['area_m2'] = .001
    p['source_floor_area_m2'] = .1
    assert solve_counts(p, max_distractors=1)[0]['bucket'] == 2
    assert solve_counts(p, max_distractors=1, selection_policy='fewest_bodies') == []
