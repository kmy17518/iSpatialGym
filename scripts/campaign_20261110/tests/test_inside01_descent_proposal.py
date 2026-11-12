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


def test_blocked_approach_uses_remaining_bounded_start_draws(monkeypatch):
    from ispatialgym.tests.test_campaign_under04_start_gap import draw
    from ispatialgym.sampling.sim_utils import Reject
    seen=[]
    def allowed(arm,pre):
        seen.append((arm,pre))
        return len(seen)==4
    (_,rec),calls=draw(monkeypatch,grasp_pose_allowed=allowed)
    assert rec.arm=='right' and len(calls)==len(seen)==4
    rejected=[]
    with pytest.raises(Reject,match='caller_grasp_pose'):
        draw(monkeypatch,grasp_pose_allowed=lambda arm,pre:rejected.append(pre) or False)
    assert len(rejected)==5


def test_no_approach_predicate_preserves_original_random_draw(monkeypatch):
    import torch as th
    from ispatialgym.tests.test_campaign_under04_start_gap import draw
    (a,_),ac=draw(monkeypatch)
    (b,_),bc=draw(monkeypatch,grasp_pose_allowed=lambda arm,pre:True)
    assert len(ac)==len(bc)==1
    assert all(th.equal(x,y) for x,y in zip(a,b))


def test_draw_predicate_keeps_the_exact_native_descent_geometry():
    import torch as th
    from ispatialgym.sampling.campaign_inside_descent import pregrasp_predicate
    evidence=json.loads(Path(__file__).with_name('inside01_descent_reference.json').read_text())
    for row in evidence['rows']:
        pred=pregrasp_predicate(np.array(row['box']).reshape(2,3))
        pre=(th.tensor(row['pre']['position']),th.tensor(row['pre']['orientation']))
        assert pred('right',pre) is row['success']


def test_horizontal_open_hand_sweep_rejects_mid_approach_collision():
    points=np.array([[x,y,z]for x in[-.005,.005]for y in[-.005,.005]for z in[-.005,.005]])
    hand=HandGeometry('right',[LinkHull('finger','finger',points,pieces=[points])],'usd_collision')
    box=np.array([[-.01,-.01,-.01],[.01,.01,.01]])
    assert not swept_hand_box(hand,[-.06,0,0],[0,0,0,1],box,rise=0)['overlap']
    assert swept_hand_box(hand,[-.06,0,0],[0,0,0,1],box,displacement=[.12,0,0])['overlap']
    assert not swept_hand_box(hand,[-.06,.1,0],[0,0,0,1],box,displacement=[.12,0,0])['overlap']


def test_segment_predicate_keeps_original_bounded_draw_budget(monkeypatch):
    from ispatialgym.tests.test_campaign_under04_start_gap import draw
    from ispatialgym.sampling.sim_utils import Reject
    seen=[]
    def allowed(arm,pre,grasp):
        seen.append((arm,pre,grasp));return len(seen)==3
    (_,record),calls=draw(monkeypatch,grasp_segment_allowed=allowed)
    assert len(calls)==len(seen)==3 and record.arm=='right'
    with pytest.raises(Reject,match='caller_grasp_segment'):
        draw(monkeypatch,grasp_segment_allowed=lambda *args:False)


def test_native_pieces_preserve_clear_route_inside_overapproximated_box():
    from ispatialgym.sampling.campaign_inside_descent import swept_hand_pieces
    def cube(lo,hi):return np.array([[x,y,z]for x in[lo[0],hi[0]]for y in[lo[1],hi[1]]for z in[lo[2],hi[2]]])
    points=cube([-.005]*3,[.005]*3)
    hand=HandGeometry('right',[LinkHull('finger','finger',points,pieces=[points])],'usd_collision')
    sides=[cube([-.02,-.04,-.02],[.02,-.02,.02]),cube([-.02,.02,-.02],[.02,.04,.02])]
    bounds=np.stack([np.vstack(sides).min(0),np.vstack(sides).max(0)])
    assert swept_hand_box(hand,[-.06,0,0],[0,0,0,1],bounds,displacement=[.12,0,0])['overlap']
    assert not swept_hand_pieces(hand,[-.06,0,0],[0,0,0,1],[.12,0,0],sides)
    assert swept_hand_pieces(hand,[-.06,0,0],[0,0,0,1],[.12,0,0],sides+[cube([-.01]*3,[.01]*3)])
