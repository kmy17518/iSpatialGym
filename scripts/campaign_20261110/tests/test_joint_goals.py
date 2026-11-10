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

