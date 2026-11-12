import numpy as np
import pytest
from ispatialgym.oracle.campaign_axial_contact import bounded_lowering


def test_contact_height_moves_only_with_native_support_clearance():
    pts=np.array([[0,0,-.018],[.01,.01,.02]])
    rotations=[np.eye(3)]*3
    amount,before=bounded_lowering(pts,rotations,[.759,.759,.759],[.735])
    assert amount==pytest.approx(.002) and before==pytest.approx(.006)
    assert before-amount>=.003
    amount,before=bounded_lowering(pts,rotations,[.757,.759,.759],[.735])
    assert amount==pytest.approx(.001) and before-amount==pytest.approx(.003)
    assert bounded_lowering(pts,rotations,[.755,.759,.759],[.735])[0]==0


def test_whole_hand_geometry_and_every_orientation_bound_the_move():
    pts=np.array([[0,0,-.018],[.04,0,-.01]])
    angle=.5; rot=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    assert bounded_lowering(pts,[np.eye(3),rot],[.759,.759],[.735])[0]==0


@pytest.mark.parametrize('points,rotations,heights,tops',[([],[],[],[]),([[0,0,float('nan')]], [np.eye(3)],[.76],[.735]),([[0,0,-.018]],[np.eye(3)],[.76],[])])
def test_missing_or_nonfinite_geometry_never_moves(points,rotations,heights,tops):
    assert bounded_lowering(points,rotations,heights,tops)[0]==0


def test_height_policy_is_default_off_and_exact_magazine_only(monkeypatch):
    from types import SimpleNamespace as NS
    from ispatialgym.sampling import campaign_runtime as CR
    from scripts.campaign_20261110.contracts import ContractError
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG',raising=False)
    assert not CR.contact_recovery('axial_magazine_contact_height')
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','/unused')
    target=NS(role='target',category='magazine',model='tfovwy',default_scale=(1.,1.,1.))
    template=NS(class_id='alignment-axial_04',task='alignment-axial',split='eval_standard',objects=[target],scene_setup={'recovery_options':{'axial_magazine_contact_height':True}})
    monkeypatch.setattr(CR,'configuration',lambda:({},template,None,None))
    assert CR.contact_recovery('axial_magazine_contact_height')
    target.model='other'
    with pytest.raises(ContractError):CR.contact_recovery('axial_magazine_contact_height')
