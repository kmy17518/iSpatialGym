from types import SimpleNamespace as NS
import math
import pytest
import torch
from ispatialgym.oracle.actions import Place, Retreat
from ispatialgym.oracle.sidecar import Pose
from ispatialgym.sampling import campaign_runtime as CR
from scripts.campaign_20261110.contracts import ContractError


def action(**kw):
    place=Place(kind='attach',arm='right',object='lid',place=Pose([0.,0.,1.],[0.,0.,0.,1.]),
                pre_place=Pose([0.,0.,1.1],[0.,0.,0.,1.]),on='pot')
    return Retreat(arm='right',object='lid',after=place,distance=.05,**kw)


def test_vertical_clear_leg_preserves_recorded_start_distance_and_orientation():
    a=action(clear_direction=[0.,0.,1.]);recorded=a.after.end_pose(None,None)
    state=NS(eef_pose=lambda arm:Pose([.002,0.,1.1],[0.,0.,0.,1.]))
    points=a.targets(None,state)['retreat'];end=a.end_pose(None,state)
    assert end.position==pytest.approx([recorded.position[0],recorded.position[1],recorded.position[2]+.05])
    assert end.orientation==recorded.orientation and a.distance==.05
    assert points[-1]==end
    assert math.dist(end.position,recorded.position)==pytest.approx(.05)


@pytest.mark.parametrize('direction',[[0.,0.,0.],[0.,float('nan'),1.],[0.,1.], [0.,float('inf'),0.]])
def test_invalid_clear_direction_is_rejected(direction):
    with pytest.raises(ValueError):action(clear_direction=direction)


def test_single_direction_cannot_change_existing_two_leg_retreat():
    with pytest.raises(ValueError):action(clear_direction=[0.,0.,1.],lift_first_m=.05)
    a=action();start=a.after.end_pose(None,None);end=a.end_pose(None,None)
    assert math.dist(start.position,end.position)==pytest.approx(.05)
    assert a.clear_direction is None


def test_vertical_retreat_policy_is_exact_and_default_off(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG',raising=False)
    assert not CR.contact_recovery('lid_vertical_final_retreat')
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','bound.json')
    obj=NS(role='target',category='lid',model='pbmsji',default_scale=(.65,.65,.65))
    template=NS(class_id='configuration_matching-attachment_04',task='configuration_matching-attachment',split='eval_standard',
                objects=[obj],scene_setup={'recovery_options':{'lid_vertical_final_retreat':True}})
    monkeypatch.setattr(CR,'configuration',lambda:(None,template,None,None))
    assert CR.contact_recovery('lid_vertical_final_retreat')
    obj.default_scale=(1.,1.,1.)
    with pytest.raises(ContractError):CR.contact_recovery('lid_vertical_final_retreat')
    obj.default_scale=(.65,.65,.65);template.class_id='configuration_matching-attachment_02'
    with pytest.raises(ContractError):CR.contact_recovery('lid_vertical_final_retreat')
