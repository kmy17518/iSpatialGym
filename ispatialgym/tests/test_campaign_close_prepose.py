from types import SimpleNamespace as NS

import pytest
import torch as th

from ispatialgym.oracle import execute as E
from ispatialgym.sampling import campaign_runtime as CR
from scripts.campaign_20261110.contracts import ContractError


def bind(monkeypatch, enabled=True):
    template=NS(class_id='configuration_matching-articulation_close_large_scale_01',
                task='configuration_matching-articulation_close_large_scale',
                scene_setup={'recovery_options':{'close_door_prepose_margin':enabled}})
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','/unused')
    monkeypatch.setattr(CR,'configuration',lambda:(None,template,None,None))
    return template


@pytest.mark.parametrize('residual,expected',[(.00001,2),(.001999,2),(.0047487,-1),(float('nan'),-1)])
def test_prepose_margin_filters_only_checked_candidate_witness(monkeypatch,residual,expected):
    bind(monkeypatch);calls=[];q=th.zeros(3)
    def solve(*a,**kw):calls.append(kw);return q
    backend=NS(q_full=lambda:q,solve_ik=solve,
               fk=lambda links,q:{links[0]:(th.tensor([residual,0.,0.]),th.tensor([0.,0.,0.,1.]))})
    motions=NS(backend=backend);monkeypatch.setattr(E,'_q_at_base',lambda *a:q)
    pose=(th.zeros(3),th.tensor([0.,0.,0.,1.]))
    count,pre=E.pushable_arc_from(motions,'right',(0.,0.,0.),pose,pose,[pose,pose])
    assert count==expected
    assert len(calls)==(4 if expected==2 else 1)
    assert (pre is None)==(expected==-1)
    assert all(c['sync_world']is False for c in calls)


def test_unconfigured_scoring_does_not_request_extra_fk(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG',raising=False);q=th.zeros(3)
    backend=NS(q_full=lambda:q,solve_ik=lambda *a,**kw:q)
    monkeypatch.setattr(E,'_q_at_base',lambda *a:q)
    pose=(th.zeros(3),th.tensor([0.,0.,0.,1.]))
    assert E.pushable_arc_from(NS(backend=backend),'right',(0.,0.,0.),pose,pose,[pose])[0]==1


def test_option_rejects_other_class_and_nonboolean(monkeypatch):
    t=bind(monkeypatch);assert CR.contact_recovery('close_door_prepose_margin')
    t.class_id='configuration_matching-articulation_close_large_scale_02'
    with pytest.raises(ContractError):CR.contact_recovery('close_door_prepose_margin')
    t.scene_setup['recovery_options']['close_door_prepose_margin']=1
    with pytest.raises(ContractError):CR.contact_recovery('close_door_prepose_margin')
