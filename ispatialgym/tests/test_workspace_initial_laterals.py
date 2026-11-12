"""Initial workspace recovery retains the physical gates and is opt-in."""
from types import SimpleNamespace
import math

import pytest
import torch

from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance


def setup(monkeypatch, *, workspace_laterals=(-.18,), floor=True, collision_free=True):
    edge=EdgeFrame.from_edge('+u', [0,0], [-1,0], .97)
    xy=edge.base_xy(.45)
    initial=Stance('init',(float(xy[0]),float(xy[1]),edge.yaw),.5,0,.45,edge)
    calls=[]
    def free(xy):
        calls.append(('floor',round(float(xy[1]),4)))
        return floor
    def clear(st):
        calls.append(('collision',st.lateral))
        return collision_free
    sampler=SimpleNamespace(kit=SimpleNamespace(floor_z=0,trav_free=free),
                            ikc=SimpleNamespace(stance_collision_free=clear),camera_pose_at=lambda _:None)
    top=SimpleNamespace(name='heldout_table',z=.7588)
    band=SimpleNamespace(path=None)
    def workspace(s,b,st,t):
        if not any(abs(st.lateral-lat)<1e-6 for lat in workspace_laterals):
            return {'cells_world':torch.zeros(0,2)}
        return {'cells_world':torch.zeros(2,2),'d_ws':.3,'d_near':.2,'d_far':.4,
                'lam_min':.64,'lam_max':.72}
    fk=SimpleNamespace(phi_m=lambda _:1,x_at=lambda _:0,t_of_z=lambda _:0.5,z_at=lambda _:1,
                       hand_below=.2,pitch=.1)
    monkeypatch.setattr(TT,'camera_fk',lambda _:fk)
    monkeypatch.setattr(TT,'workspace',workspace)
    monkeypatch.setattr(TT,'image_border_lateral',lambda *a:(.8,.8))
    monkeypatch.setattr(TT,'init_trunk_delta',lambda *a:{'delta':.3,'delta_min':.2,'delta_max':.4,'delta_aim':.3})
    return sampler,initial,top,band,calls


def test_empty_midpoint_remains_rejected_without_opt_in(monkeypatch):
    s,st,top,band,calls=setup(monkeypatch)
    with pytest.raises(Reject,match='no_workspace'):
        TT.phase1b_init_trunk(s,st,.3,bands=band,top=top)
    assert calls==[]


def test_recovery_checks_floor_collision_and_retains_end_margin(monkeypatch):
    s,st,top,band,calls=setup(monkeypatch)
    result=TT.phase1b_init_trunk(s,st,.3,bands=band,top=top,
                                workspace_laterals=[math.nan,-.2,-.18])
    assert result.lateral==-.18 and result.extras['workspace_initial_lateral_m']==-.18
    assert result.extras['n_workspace_cells']==2
    assert calls==[('floor',-.18),('collision',-.18)]


@pytest.mark.parametrize('floor,collision_free,expected_checks',[(False,True,1),(True,False,2)])
def test_shift_cannot_bypass_physical_start_gates(monkeypatch,floor,collision_free,expected_checks):
    s,st,top,band,calls=setup(monkeypatch,floor=floor,collision_free=collision_free)
    with pytest.raises(Reject,match='no_workspace'):
        TT.phase1b_init_trunk(s,st,.3,bands=band,top=top,workspace_laterals=[-.18])
    assert len(calls)==expected_checks


def test_successful_midpoint_does_not_change_with_fallback_configured(monkeypatch):
    s,st,top,band,calls=setup(monkeypatch,workspace_laterals=[0,-.18])
    result=TT.phase1b_init_trunk(s,st,.3,bands=band,top=top,workspace_laterals=[-.18])
    assert result.lateral==0 and 'workspace_initial_lateral_m' not in result.extras
    assert calls==[]
