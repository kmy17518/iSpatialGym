"""Stationary obstacle goals must gate success without becoming measures."""

import pytest

from ispatialgym.activity.problem import parse_problem_string
from ispatialgym.sampling.campaign_obstacles import initial_pose_targets, with_stationary_goals


BDDL = """(define (problem camera_relocalization-1)
    (:domain ispatialgym)
    (:objects agent.n.01_1 - agent.n.01 hamper.n.01_1 - hamper.n.01 floor.n.01_1 - floor.n.01)
    (:init (ontop agent.n.01_1 floor.n.01_1) (ontop hamper.n.01_1 floor.n.01_1))
    (:goal (and (pose_reached ?agent.n.01_1) (pose_settled ?agent.n.01_1)))
    (:measure (pose_reached ?agent.n.01_1))
)"""


@pytest.mark.parametrize("legacy", [False, True])
def test_obstacle_gates_goal_but_never_enters_measure(legacy):
    source = BDDL.replace("(:measure (pose_reached ?agent.n.01_1))", "") if legacy else BDDL
    patched = with_stationary_goals(source, ["hamper.n.01_1"])
    problem = parse_problem_string(patched, "camera_relocalization", 1)
    assert ["pose_reached", "?hamper.n.01_1"] in problem.raw_goal
    assert problem.raw_measure == [["pose_reached", "?agent.n.01_1"]]
    assert with_stationary_goals(patched, ["hamper.n.01_1"]) == patched


def test_invalid_or_measured_obstacle_fails_closed():
    with pytest.raises(ValueError, match="declared"):
        with_stationary_goals(BDDL, ["missing.n.01_1"])
    source = BDDL.replace("(:measure (pose_reached ?agent.n.01_1))", "(:measure (pose_reached ?hamper.n.01_1))")
    with pytest.raises(ValueError, match="measure"):
        with_stationary_goals(source, ["hamper.n.01_1"])


def test_goal_target_copies_exact_initial_pose():
    pose = [[1, 2, 3], [0, 0, 0, 1]]
    targets = initial_pose_targets({"hamper": pose}, ["hamper"], position_tolerance=0.005, orientation_tolerance=0.03)
    assert targets["hamper"]["position"] == pose[0]
    assert targets["hamper"]["orientation"] == pose[1]
    pose[0][0] = 999
    assert targets["hamper"]["position"][0] == 1
