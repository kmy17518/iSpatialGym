import pytest
from scripts.campaign_20261110.runtime_axial_probe import expanded_scan


def test_only_candidate_limit_changes_preserving_exact_check_calls():
    source='if len(paired) + len(push_only) >= 6:\n    break\nreturn paired + push_only\n'
    assert expanded_scan(source,6)==source
    assert expanded_scan(source,24)==source.replace('>= 6','>= 24')
    with pytest.raises(ValueError):expanded_scan(source,25)
    with pytest.raises(ValueError):expanded_scan(source.replace('>= 6','>= 7'),24)
    with pytest.raises(ValueError):expanded_scan(source+source,24)
