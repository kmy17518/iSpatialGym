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


def test_written_member_proposes_only_existing_checked_stances_and_keeps_exploration():
    from ispatialgym.sampling.campaign_tabletop_starts import prioritize_written_stances
    cands=[SimpleNamespace(base_pose2d=(float(i),0.,0.),translate=0.,tilt=0.,focus=None) for i in range(10)]
    order=list(range(10))
    assert prioritize_written_stances(order,cands,[]) is order
    prior={'base_pose2d':[8.,0.,0.],'translate':0.,'tilt':0.}
    ranked=prioritize_written_stances(order,cands,[prior])
    assert ranked[:4] == order[:4]
    assert ranked[4] == 8 and len(ranked[:6]) == 6
    assert sorted(ranked) == order and len(set(ranked)) == len(order)


def test_source_stance_requires_exact_summary_provenance_and_finite_pose():
    from ispatialgym.sampling.campaign_tabletop_starts import anchor_manipulation
    import copy
    raw = {'anchor_sources': {'summary': {'sha256': 'a'*64}},
           'anchor_manipulation_stance': {'base_pose2d':[9.09,3.30,3.14], 'focus':[8.18,3.46,.83],
                                          'translate':0., 'tilt':-.61, 'source_summary_sha256':'a'*64}}
    assert anchor_manipulation({}) is None
    assert anchor_manipulation(raw)['tilt'] == -.61
    for key, value in [('source_summary_sha256','b'*64), ('focus',[0.,float('nan'),0.]), ('translate',True)]:
        changed=copy.deepcopy(raw);changed['anchor_manipulation_stance'][key]=value
        with pytest.raises(GoalPlanError):anchor_manipulation(changed)


def test_source_stance_nextto_scope_does_not_enable_other_pair_classes():
    from ispatialgym.sampling.campaign_tabletop_starts import source_stance_hint_enabled
    assert source_stance_hint_enabled('nextto', 'small', 'inter_object_relation_satisfaction-nextto_02')
    assert source_stance_hint_enabled('nextto', 'small', 'inter_object_relation_satisfaction-nextto_01')
    assert source_stance_hint_enabled('touching', 'small', 'inter_object_relation_satisfaction-touching_01')
    assert source_stance_hint_enabled('inside', 'small', 'inter_object_relation_satisfaction-inside_01')
    for relation, scale, cls in [('nextto','large','inter_object_relation_satisfaction-nextto_02'),
            ('touching','small','inter_object_relation_satisfaction-touching_02'),
            ('nextto','small','inter_object_relation_satisfaction-nextto_04')]:
        assert not source_stance_hint_enabled(relation, scale, cls)


def test_measured_small_pair_scope_preserves_unmeasured_and_large_classes():
    from ispatialgym.sampling.campaign_tabletop_starts import source_stance_hint_enabled
    for relation, suffixes in [('under', ('01','03','04')), ('ontop', ('01','02','04'))]:
        for suffix in suffixes:
            cls='inter_object_relation_satisfaction-'+relation+'_small_scale_'+suffix
            assert source_stance_hint_enabled(relation, 'small', cls)
            assert not source_stance_hint_enabled(relation, 'large', cls)
            assert not source_stance_hint_enabled('touching', 'small', cls)
    assert source_stance_hint_enabled('nextto','small','inter_object_relation_satisfaction-nextto_03')
    for relation, suffix in [('under','02'),('ontop','03')]:
        cls='inter_object_relation_satisfaction-'+relation+'_small_scale_'+suffix
        assert not source_stance_hint_enabled(relation,'small',cls)


def test_under_source_torso_and_focus_ranking_retains_checked_pool():
    from types import SimpleNamespace
    from ispatialgym.sampling.campaign_tabletop_starts import prioritize_written_stances
    # Under04's initial and manipulation bases coincide, but preserving only
    # that base loses the successful .25/-0.5325 torso and placement focus.
    source={'base_pose2d':[-10.126945,-2.997623,.0000012],
            'focus':[-9.216946,-2.917622,.882931],'translate':.25,'tilt':-.532533}
    candidates=[SimpleNamespace(**{**source,'translate':.8,'tilt':-.1,
        'focus':[-9.4,-3.0,.88]}) for _ in range(8)]
    candidates.append(SimpleNamespace(**source))
    order=list(range(9));ranked=prioritize_written_stances(order,candidates,[source])
    assert ranked[:4]==order[:4] and ranked[4]==8
    assert sorted(ranked)==order


def test_nextto_reference_stance_is_tried_without_replacing_native_candidates():
    from types import SimpleNamespace
    from ispatialgym.sampling.campaign_tabletop_starts import prioritize_written_stances
    # Source1207 at0.45m stand-off reaches the fixed release; group1397's
    # frame-ranked shortlist repeatedly uses stances~0.15m farther away.
    source={'base_pose2d':[5.389566,3.647212,-.002552], 'focus':[6.219768,3.725094,.890702],
            'translate':0.,'tilt':-.610867}
    candidates=[SimpleNamespace(base_pose2d=[5.239566,3.647212,-.002552],focus=[6.35,3.73,.89],translate=.75,tilt=-.19) for _ in range(8)]
    candidates.append(SimpleNamespace(**source))
    order=list(range(9));ranked=prioritize_written_stances(order,candidates,[source])
    assert ranked[:4]==order[:4]
    assert 8 in ranked[:6]
    assert sorted(ranked)==order
    assert len(set(ranked))==len(order)
