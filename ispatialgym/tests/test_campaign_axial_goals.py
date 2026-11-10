import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.campaign_surface_goals import SurfaceGoalPlan
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.tasks import alignment_axial as ax
from ispatialgym.tests.test_axial_settled_rest import _ax, _unit, SPOON_REST


def test_start_turn_preserves_fixed_goal_com_with_noncentral_tilted_root():
    goal = ([.6, -.2, .82], _unit(SPOON_REST).tolist())
    local_com = th.tensor([.04, -.02, .012])
    start = ax.initial_about_fixed_com(*goal, local_com, math.radians(37))
    wanted = th.tensor(goal[0])+C.quat_rotate(th.tensor(goal[1]), local_com)
    actual = start[0]+C.quat_rotate(start[1], local_com)
    assert th.allclose(actual, wanted, atol=1e-7)
    assert float(C.quat_angle(start[1], th.tensor(goal[1]))) == pytest.approx(math.radians(37), abs=1e-6)


def test_native_stop_geometry_reaches_fixed_pose_before_contact_checks():
    s = _ax(rest=SPOON_REST)
    goal_q = C.qmul(C.quat_from_yaw(.8), _unit(SPOON_REST))
    goal_p = th.tensor([.6, .1, .83])
    s._campaign_surface_active = SimpleNamespace(orientation=goal_q.tolist())
    start = ax.initial_about_fixed_com(goal_p.tolist(), goal_q.tolist(), s.com_local, -.5)
    com = goal_p+C.quat_rotate(goal_q, s.com_local)
    end = com+C.quat_rotate(start[1], th.tensor([.06, 0, 0]))
    result, _, _ = s._stop_geometry(start, end, com, 1, .8, .835)
    assert th.allclose(result[0], goal_p, atol=1e-7)
    assert th.allclose(result[1], goal_q, atol=1e-7)


def test_shared_goal_survives_native_residual_but_eval_bucket_stays_strict():
    plan = SurfaceGoalPlan("campaign", "g", "eval_standard", "alignment-axial", "template", "a"*64,
                           (.6, 0, .8), (0, 0, 0, 1), (20, 40), 100, 3, "b"*64)
    s = _ax()
    s._campaign_surface_active = plan
    s.campaign_group_remaining_s = lambda: 100
    s.ikc = SimpleNamespace(WITNESS_POS_TOL=.005, WITNESS_ANG_TOL=.05)
    s.tname = "target"
    s.tol = lambda key, default: default
    start_q = C.quat_from_yaw(math.radians(35))
    draft = SimpleNamespace(goal_poses={"target": (th.tensor([.601, 0, .8]), th.tensor([0., 0, 0, 1]))},
                            object_poses={"target": (th.tensor([.6, 0, .8]), start_q)},
                            pose_targets={"target": {}}, summary={"campaign_goal_group": plan.summary()})
    ax.AlignmentAxial.campaign_verify_goal(s, draft)
    assert draft.goal_poses["target"][0].tolist() == pytest.approx(plan.position)
    assert draft.summary["campaign_goal_group"]["actual_magnitude_deg"] == pytest.approx(35, abs=1e-4)
    draft.object_poses["target"] = (draft.object_poses["target"][0], C.quat_from_yaw(math.radians(50)))
    with pytest.raises(Reject, match="strict eval bucket"):
        ax.AlignmentAxial.campaign_verify_goal(s, draft)
    s._campaign_surface_active = replace(plan, split="train")
    ax.AlignmentAxial.campaign_verify_goal(s, draft)
    draft.goal_poses["target"] = (th.tensor([.61, 0, .8]), th.tensor([0., 0, 0, 1]))
    with pytest.raises(Reject, match="native axial endpoint"):
        ax.AlignmentAxial.campaign_verify_goal(s, draft)
