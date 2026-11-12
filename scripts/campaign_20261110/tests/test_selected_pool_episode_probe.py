from scripts.campaign_20261110.selected_pool_episode_probe import boundary,live_identity
from types import SimpleNamespace
import pytest


def test_native_incompatible_keys_force_reload():
    assert boundary(None,'A')=='load'
    assert boundary('A','B',True)=='reload'
    assert boundary('A','A',False)=='reload'
    assert boundary('A','A',True)=='reuse'


def fixture():
    target={'native_name':'campaign_pool_A','category':'bottle_of_pop','model':'pvabxf','scale':[1.,1.,1.]}
    support={'native_name':'floor','category':'floors','model':'uzsntg','scale':[1.,1.,1.]}
    expected={'target':target,'support':support,'scene':'house_single_floor','room':'living_room_0'}
    objects=[SimpleNamespace(name=s['native_name'],category=s['category'],model=s['model'],scale=s['scale'],in_rooms=['living_room_0']) for s in [target,support]]
    return SimpleNamespace(objects=objects),expected


def test_checks_actual_support_room_and_inactive_model_absence():
    scene,expected=fixture();assert live_identity(scene,expected)['task_room']=='living_room_0'
    scene.objects.append(SimpleNamespace(name='campaign_pool_B'))
    with pytest.raises(AssertionError,match='stale'):live_identity(scene,expected)
    scene.objects.pop();scene.objects[1].in_rooms=['kitchen_0']
    with pytest.raises(AssertionError,match='room'):live_identity(scene,expected)


def test_rejects_wrong_scale_or_model():
    scene,expected=fixture();scene.objects[0].scale=[.9,.9,.9]
    with pytest.raises(AssertionError,match='scale'):live_identity(scene,expected)
    scene,expected=fixture();scene.objects[0].model='hjrwqb'
    with pytest.raises(AssertionError,match='identity'):live_identity(scene,expected)
