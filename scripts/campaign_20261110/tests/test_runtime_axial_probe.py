import pytest
from scripts.campaign_20261110.runtime_axial_probe import expanded_scan, expanded_candidates, approach_distance


def test_only_candidate_limit_changes_preserving_exact_check_calls():
    source='if len(paired) + len(push_only) >= 6:\n    break\nreturn paired + push_only\n'
    assert expanded_scan(source,6)==source
    assert expanded_scan(source,24)==source.replace('>= 6','>= 24')
    with pytest.raises(ValueError):expanded_scan(source,25)
    with pytest.raises(ValueError):expanded_scan(source.replace('>= 6','>= 7'),24)
    with pytest.raises(ValueError):expanded_scan(source+source,24)


def test_checked_candidate_search_never_changes_noncampaign_path():
    source='scan_order = order[:64] if campaign_plan is not None else order[:6]\nscan = self._scan_candidates(rng, init, scan_order)\n'
    assert expanded_candidates(source,64)==source
    assert expanded_candidates(source,256)==source.replace('order[:64]','order[:256]')
    with pytest.raises(ValueError):expanded_candidates(source,257)
    with pytest.raises(ValueError):expanded_candidates(source.replace('order[:64]','order[:100]'),256)


def test_counterfactual_changes_only_pre_not_contact_arc_or_goal():
    source='arc = unchanged_arc()\nreturn [C.pre_pose(p_init), p_init, arc[0], arc[1]]\n'
    assert approach_distance(source,.10)==source
    assert approach_distance(source,.05)==source.replace('C.pre_pose(p_init)','C.pre_pose(p_init, dist=0.05)')
    with pytest.raises(ValueError):approach_distance(source,.01)
    with pytest.raises(ValueError):approach_distance(source.replace('arc[1]','arc[2]'),.05)
