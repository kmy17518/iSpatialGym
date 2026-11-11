from types import SimpleNamespace
import math
import numpy as np
import pytest
from ispatialgym.sampling.campaign_held_goals import GoalPlanError
from ispatialgym.sampling.campaign_tabletop_starts import anchor_base, ordered_edges, edge_lateral, near_lateral


def test_anchor_is_optional_but_present_anchor_must_be_finite():
    assert anchor_base({}) is None
    assert anchor_base({'anchor_base_pose2d':[1,2,3]}) == (1.,2.,3.)
    for value in ([1,2], [1,2,float('nan')], [True,2,3]):
        with pytest.raises(GoalPlanError):anchor_base({'anchor_base_pose2d':value})


def test_goal_viewpoint_ranks_native_edges_without_discarding_any():
    def edge(x, yaw):
        return (SimpleNamespace(mid=(x,0),tangent=(0,1),length=2.,base_xy=lambda off,lat:(x+off,lat)),
                SimpleNamespace(stand_off=.5,base_pose2d=(x+.5,0,yaw)))
    right, distant = edge(0, math.pi), edge(2,0)
    rows=[distant,right]
    assert ordered_edges(rows,None) is rows
    assert ordered_edges(rows,(.5,.2,-math.pi)) == [right,distant]
    assert edge_lateral(right[0],(.5,.2,0)) == pytest.approx(.2)


def test_lateral_proposals_remain_inside_native_bounds_and_keep_start_variation():
    rng=np.random.default_rng(5)
    values=[near_lateral(rng,-.2,.2,.16) for _ in range(100)]
    assert min(values)>=.08 and max(values)<=.2 and len(set(values))==100
    assert -.2<=near_lateral(rng,-.2,.2,-4)<=-.12
    with pytest.raises(GoalPlanError):near_lateral(rng,.2,-.2,0)
