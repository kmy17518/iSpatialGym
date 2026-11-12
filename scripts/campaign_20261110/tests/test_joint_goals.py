from dataclasses import replace
from types import SimpleNamespace

import pytest

from ispatialgym.sampling import campaign_joint_goals as j


def plan():
    return j.JointGoalPlan("fresh","g1","eval_standard","template","a"*64,"target","hinge",
                           ((0,0,0),(0,0,0,1)),{"hinge":0.,"other":0.3},(0.,0.5),100.,3,"b"*64)


def test_closed_goal_is_chosen_before_initial_angle_and_train_bucket_is_soft():
    p = plan()
    rng = SimpleNamespace(integers=lambda n: n-1)
    assert p.initial(direction="close",lower=0,upper=1,stable=[.3,.5,.8],rng=rng) == (.5,0.)
    assert replace(p,split="train").initial(direction="close",lower=0,upper=1,stable=[.8],rng=rng) == (.8,0.)
    with pytest.raises(j.GoalPlanError,match="no stable"):
        p.initial(direction="close",lower=0,upper=1,stable=[.8],rng=rng)


def test_opening_start_is_closed_and_requested_goal_remains_fixed():
    p = replace(plan(),joints={"hinge":.4,"other":.3})
    assert p.initial(direction="open",lower=0,upper=1,stable=[.2,.6],rng=None) == (0.,.4)
    with pytest.raises(j.GoalPlanError,match="bucket"):
        replace(p,joints={"hinge":.8,"other":.3}).initial(direction="open",lower=0,upper=1,stable=[],rng=None)


def test_one_joint_rule_checks_only_initial_to_goal():
    p = plan()
    p.verify_joint_change(["hinge","other"],[.4,.3],[0,.3])
    # No intermediate-state restriction: an oracle may temporarily press the other flap.
    with pytest.raises(j.GoalPlanError,match="another joint"):
        p.verify_joint_change(["hinge","other"],[.4,.2],[0,.3])
    with pytest.raises(j.GoalPlanError,match="goal changed"):
        p.verify_joint_change(["hinge","other"],[.4,.3],[.1,.3])


def test_native_small_angle_pairs_keep_goal_and_soften_only_training_bucket():
    p = plan()
    native = [(.3, 0), (.5, 0), (.8, 0)]
    assert p.eligible_pairs(native, direction="close", lower=0, upper=1) == native[:2]
    assert replace(p, split="train").eligible_pairs(native, direction="close", lower=0, upper=1) == native
    opening = replace(p, joints={"hinge": .4, "other": .3})
    assert opening.eligible_pairs([(0, .2), (0, .4), (0, .8)], direction="open", lower=0, upper=1) == [(0, .4)]
    with pytest.raises(j.GoalPlanError, match="no native angle"):
        opening.eligible_pairs([(0, .2), (0, .8)], direction="open", lower=0, upper=1)


def test_physics_probed_half_angle_is_not_rounded_out_of_strict_bucket():
    upper = 2.79252552986145  # Native hinged_jar/jnjtrl joint range.
    half = .5 * upper
    angles = j.unique_probed_angles([half + .000001, half, half])
    assert angles == [half, half + .000001]  # Only exact duplicates coalesce.
    p = plan()
    assert p.eligible_pairs([(q, 0.) for q in angles], direction="close", lower=0., upper=upper) == [(half, 0.)]
    assert replace(p, fraction_range=(.5, 1.)).eligible_pairs(
        [(q, 0.) for q in angles], direction="close", lower=0., upper=upper) == [(half + .000001, 0.)]
    # The historical rounded value really is outside bucket0; keep rejecting it.
    with pytest.raises(j.GoalPlanError, match="no native angle"):
        p.eligible_pairs([(round(half, 4), 0.)], direction="close", lower=0., upper=upper)
    with pytest.raises(j.GoalPlanError, match="nonfinite"):
        j.unique_probed_angles([float("nan")])


def test_opening_pair_retains_native_float_identity_for_record_selection():
    goal = .501782238483429
    native = (0., .5017822310328484)
    p = replace(plan(), joints={'hinge':goal,'other':.3}, fraction_range=(.5,1.))
    eligible = set(p.eligible_pairs([native], direction='open', lower=0., upper=.872664749622345))
    assert native in eligible
    assert (native[0],goal) not in eligible
    # Existing goal tolerance and strict native bucket remain unchanged.
    with pytest.raises(j.GoalPlanError):
        p.eligible_pairs([(0.,goal+.00001)], direction='open', lower=0., upper=.872664749622345)
    half=.5*.872664749622345
    with pytest.raises(j.GoalPlanError):
        replace(p,joints={'hinge':half+1e-8,'other':.3}).eligible_pairs(
            [(0.,half)], direction='open', lower=0., upper=.872664749622345)


def test_articulation_plan_rejects_wrong_identity_and_queue_time_does_not_consume_budget():
    raw={"schema":j.SCHEMA,"campaign_id":"fresh","group_id":"g","split":"train","template_id":"t",
         "contract_template_sha256":"a"*64,"scene":"scene","room":"room","target":"target","joint":"hinge",
         "goal_pose":{"position":[0,0,0],"orientation":[0,0,0,1]},"goal_joints":{"hinge":.5},
         "fraction_range":[0,.5],"created_at":1,"budget_seconds":30,"max_members":3}
    kwargs=dict(template_id="t",scene="scene",rooms=["room"],target="target",joint="hinge",activated_at=100,now=110)
    assert j.validate_joint_plan(raw,**kwargs).deadline_at == 130
    with pytest.raises(j.GoalPlanError,match="role/joint"):
        j.validate_joint_plan({**raw,"joint":"other"},**kwargs)
    with pytest.raises(j.GoalGroupTimeout):
        j.validate_joint_plan(raw,**{**kwargs,"now":131})
def test_only_written_joint_starts_block_duplicates_and_other_grasps_remain_eligible():
    from types import SimpleNamespace
    from ispatialgym.sampling.campaign_joint_goals import start_was_written, remember_written_start

    sampler = SimpleNamespace()
    first = {"joints": {"hinge": 0.0}, "grasp": {"arm": "right", "pinch_sign": 1, "bar_offset_m": 0.0}}
    assert not start_was_written(sampler, first)
    remember_written_start(sampler, first)
    assert start_was_written(sampler, first)
    assert not start_was_written(sampler, {**first, "grasp": {**first["grasp"], "pinch_sign": -1}})
    assert not start_was_written(sampler, {**first, "joints": {"hinge": .25}})



def test_missing_eval_bucket_proposes_values_for_native_physical_testing_only():
    p = replace(plan(), fraction_range=(.5, 1.))
    values = p.closing_probe_values(direction='close', lower=0, upper=2, stable=[.36, .42])
    assert len(values) == 8 and all(1 < v <= 2 for v in values) and values[-1] == 2
    assert p.closing_probe_values(direction='close', lower=0, upper=2, stable=[1.2]) == values
    assert replace(p, split='train').closing_probe_values(direction='close', lower=0, upper=2, stable=[.4]) == []
    assert p.closing_probe_values(direction='open', lower=0, upper=2, stable=[]) == []


@pytest.mark.parametrize('upper,legacy_range', [(1.908, (.30, .45)), (1.571, (.18, .25))])
def test_fully_open_legacy_endpoint_does_not_suppress_high_bucket_probes(upper, legacy_range):
    p = replace(plan(), fraction_range=(.5, 1.))
    stable = [upper * legacy_range[0], upper * legacy_range[1], upper]
    proposed = p.closing_probe_values(direction='close', lower=0, upper=upper, stable=stable)
    assert len(proposed) == 7 and all(.5 < q / upper < 1 for q in proposed)
    # Merely proposing new values does not make them eligible: only the
    # physically verified stable list is consumed by initial().
    rng = SimpleNamespace(integers=lambda n: 0)
    assert p.initial(direction='close', lower=0, upper=upper, stable=stable, rng=rng) == (upper, 0.)
    passed = stable + proposed
    assert p.closing_probe_values(direction='close', lower=0, upper=upper, stable=passed) == []


def test_closing_grid_deduplication_preserves_exact_stable_values_and_strict_boundaries():
    p = replace(plan(), fraction_range=(.5, 1.))
    near = 1.125 + 5e-7
    stable = [1., near, 2.]
    before = list(stable)
    values = p.closing_probe_values(direction='close', lower=0, upper=2, stable=stable)
    assert stable == before and len(values) == 6 and 1.125 not in values
    assert p.eligible_pairs([(1., 0.), (near, 0.)], direction='close', lower=0, upper=2) == [(near, 0.)]


def test_carton_viewpoint_hint_requires_audited_source_and_preserves_goal():
    raw={"schema":j.SCHEMA,"campaign_id":"fresh","group_id":"g","split":"eval_standard","template_id":"t",
         "contract_template_sha256":"a"*64,"scene":"scene","room":"room","target":"target","joint":"hinge",
         "goal_pose":{"position":[15.85,-8.22,.84],"orientation":[0,0,0,1]},"goal_joints":{"hinge":.738},
         "fraction_range":[0,.5],"created_at":1,"budget_seconds":30,"max_members":3,
         "anchor_base_pose2d":[15.87,-7.49,-1.57],"anchor_sources":{"summary":{"sha256":"c"*64}},
         "anchor_manipulation_stance":{"base_pose2d":[15.87,-7.49,-1.57],"focus":[15.87,-8.24,.97],
                                      "translate":.75,"tilt":0.,"source_summary_sha256":"c"*64}}
    kwargs=dict(template_id="t",scene="scene",rooms=["room"],target="target",joint="hinge",activated_at=100,now=110)
    p=j.validate_joint_plan(raw,**kwargs)
    assert p.pose==((15.85,-8.22,.84),(0.,0.,0.,1.)) and p.joints=={'hinge':.738}
    assert j.carton_source_hint_enabled(p,'configuration_matching-articulation_open_small_scale_02')
    assert not j.carton_source_hint_enabled(p,'configuration_matching-articulation_open_small_scale_01')
    assert not j.carton_source_hint_enabled(plan(),'configuration_matching-articulation_open_small_scale_02')
    with pytest.raises(j.GoalPlanError,match='summary provenance'):
        j.validate_joint_plan({**raw,'anchor_sources':{'summary':{'sha256':'d'*64}}},**kwargs)
    with pytest.raises(j.GoalPlanError,match='anchor base'):
        j.validate_joint_plan({**raw,'anchor_base_pose2d':[float('nan'),0,0]},**kwargs)


def test_carton_ranking_retains_exploration_full_native_pool_and_other_classes():
    hint={'base_pose2d':[0.,0.,0.],'focus':[0.,.75,1.],'translate':.75,'tilt':0.}
    p=replace(plan(),anchor_base_pose2d=(0.,0.,0.),anchor_manipulation_stance=hint)
    candidates=[SimpleNamespace(base_pose2d=(float(i),0.,0.),focus=(0.,.75,1.),translate=.75,tilt=0.) for i in range(8)]
    order=[7,6,5,4,3,2,1,0]
    ranked=j.order_carton_stances(p,'configuration_matching-articulation_open_small_scale_02',order,candidates)
    assert ranked[:4]==order[:4] and ranked[4:6]==[0,1]
    assert sorted(ranked)==sorted(order)
    assert j.order_carton_stances(p,'configuration_matching-articulation_close_small_scale_02',order,candidates)==order
