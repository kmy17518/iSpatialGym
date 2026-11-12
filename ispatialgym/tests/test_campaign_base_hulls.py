from types import SimpleNamespace as NS
import numpy as np
import pytest
from ispatialgym.sampling.campaign_base_hull_screen import first_close,screen_record
from ispatialgym.sampling import campaign_runtime as CR
from scripts.campaign_20261110.contracts import ContractError


def box(lo,hi):return np.array([[x,y,z]for x in[lo[0],hi[0]]for y in[lo[1],hi[1]]for z in[lo[2],hi[2]]])


def test_route_detects_native_piece_between_clear_endpoints():
    base=box([-.1,-.1,0],[.1,.1,.2]);chair=box([.4,-.04,0],[.6,.04,.2])
    assert first_close([base],[(0,0,0),(1,0,0)],0,[chair])is None
    hit=first_close([base],[(0,0,0),(.5,0,0),(1,0,0)],0,[chair])
    assert hit['pose_index']==1 and hit['common_insphere_radius_m']>0
    assert first_close([base],[(0,.3,0),(.5,.3,0),(1,.3,0)],0,[chair])is None
    assert first_close([base],[(.5,0,0)],.3,[chair])is None


def test_near_native_tangency_is_screened_without_requiring_penetration():
    base=box([-.1,-.1,0],[.1,.1,.2]);chair=box([.101,-.04,0],[.2,.04,.2])
    assert first_close([base],[(0,0,0)],0,[chair],insphere_margin=0)is None
    hit=first_close([base],[(0,0,0)],0,[chair]);assert -.005<hit['common_insphere_radius_m']<0


def test_invalid_geometry_never_becomes_clear():
    with pytest.raises(ValueError):first_close([],[(0,0,0)],0,[])
    with pytest.raises(ValueError):first_close([np.zeros((3,3))],[(0,0,0)],0,[np.zeros((8,3))])


def test_native_screen_default_off_and_exact_class_only(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG',raising=False)
    assert screen_record(None,None)is None
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','bound.json')
    t=NS(class_id='inter_object_relation_satisfaction-nextto_large_scale_01',task='inter_object_relation_satisfaction-nextto_large_scale',scene_setup={'recovery_options':{'nextto_native_base_hulls':True}})
    monkeypatch.setattr(CR,'configuration',lambda:(None,t,None,None))
    assert CR.contact_recovery('nextto_native_base_hulls')
    t.class_id='inter_object_relation_satisfaction-nextto_large_scale_02'
    with pytest.raises(ContractError):CR.contact_recovery('nextto_native_base_hulls')
