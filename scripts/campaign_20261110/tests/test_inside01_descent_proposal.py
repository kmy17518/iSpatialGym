from pathlib import Path
from types import SimpleNamespace as NS
import json
import numpy as np
import pytest
from ispatialgym.sampling.campaign_inside_descent import swept_hand_box,enabled
from ispatialgym.grasping.hand_geometry import HandGeometry,LinkHull,load


def test_continuous_sweep_finds_collision_between_clear_endpoints():
    points=np.array([[x,y,z]for x in[-.005,.005]for y in[-.005,.005]for z in[-.005,.005]])
    hand=HandGeometry('right',[LinkHull('finger','finger',points,pieces=[points])],'usd_collision')
    box=np.array([[-.01,-.01,-.01],[.01,.01,.01]])
    for z in[-.06,.06]:assert not swept_hand_box(hand,[0,0,z],[0,0,0,1],box,rise=0)['overlap']
    assert swept_hand_box(hand,[0,0,-.06],[0,0,0,1],box,rise=.12)['overlap']
    assert not swept_hand_box(hand,[.1,0,-.06],[0,0,0,1],box,rise=.12)['overlap']


def test_measured_failed_descent_rejected_successful_reference_retained():
    evidence=json.loads(Path(__file__).with_name('inside01_descent_reference.json').read_text())
    hand=load('right');assert hand.source=='usd_collision'
    for row in evidence['rows']:
        result=swept_hand_box(hand,row['pre']['position'],row['pre']['orientation'],np.array(row['box']).reshape(2,3))
        assert result['overlap'] is (not row['success'])


def test_exact_campaign_class_scope(monkeypatch):
    from ispatialgym.sampling import campaign_runtime
    monkeypatch.setenv('ISPATIALGYM_INSIDE01_DESCENT_PROPOSAL','1')
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG',raising=False)
    assert not enabled()
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','exact.json')
    for cls,expected in [('inter_object_relation_satisfaction-inside_01',True),('inter_object_relation_satisfaction-inside_04',False)]:
        monkeypatch.setattr(campaign_runtime,'configuration',lambda:(None,NS(class_id=cls)))
        assert enabled() is expected
    monkeypatch.delenv('ISPATIALGYM_INSIDE01_DESCENT_PROPOSAL')
    assert not enabled()


def test_invalid_or_non_native_geometry_is_not_a_pass():
    hand=load('right')
    with pytest.raises(ValueError):swept_hand_box(hand,[0,0,0],[0,0,0,0],np.zeros((2,3)))
    with pytest.raises(ValueError):swept_hand_box(NS(source='urdf_visual'),[0,0,0],[0,0,0,1],np.zeros((2,3)))
