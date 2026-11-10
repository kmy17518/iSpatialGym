import pytest

from scripts.campaign_20261110.rebind_floating_aliases import verify_alias_only


def test_alias_extension_preserves_exact_physical_metadata():
    original = {'grasps': [{'pose': [1, 2, 3]}], 'scale': [1, 1, 1], 'legacy_keys': ['old']}
    verify_alias_only(original, {**original, 'legacy_keys': ['old', 'apple__model__f']}, 'apple__model__f')


@pytest.mark.parametrize('change', ['scale', 'grasps', 'remove_alias'])
def test_alias_repair_rejects_physical_or_existing_route_changes(change):
    original = {'grasps': [{'pose': [1, 2, 3]}], 'scale': [1, 1, 1], 'legacy_keys': ['old']}
    repaired = {**original, 'legacy_keys': ['old', 'apple__model__f']}
    if change == 'remove_alias':
        repaired['legacy_keys'] = ['apple__model__f']
    else:
        repaired[change] = []
    with pytest.raises(ValueError):
        verify_alias_only(original, repaired, 'apple__model__f')
