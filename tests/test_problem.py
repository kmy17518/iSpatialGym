import glob
import os

import pytest

from ispatialgym.activity.problem import ACTIVITY_DEFINITIONS_ROOT, ISGProblemError, load_problem, parse_problem_string

GROUP_PROBLEM = """
(define (problem configuration_matching-0)
    (:domain ispatialgym)
    (:variant configuration_matching-block_design-block_2-table)
    (:objects
        toy_dice.n.01_1 toy_dice.n.01_2 - toy_dice.n.01
        tray.n.01_1 - tray.n.01
        agent.n.01_1 - agent.n.01
    )
    (:groups
        toy_dice.n.01_1 toy_dice.n.01_2 - targets_group.n.01_1
    )
    (:init
        (ontop targets_group.n.01_1 tray.n.01_1)
    )
    (:goal
        (and
            (ontop ?targets_group.n.01_1 ?tray.n.01_1)
            (layout_reached ?targets_group.n.01_1)
            (not (grasped ?agent.n.01_1 ?targets_group.n.01_1))
            (pose_settled ?targets_group.n.01_1)
        )
    )
    (:measure
        (layout_reached ?targets_group.n.01_1)
    )
)
"""


def shipped_problems():
    paths = sorted(glob.glob(os.path.join(ACTIVITY_DEFINITIONS_ROOT, "*", "problem*.bddl")))
    return [
        (os.path.basename(os.path.dirname(p)), int(os.path.basename(p)[len("problem") : -len(".bddl")])) for p in paths
    ]


@pytest.mark.parametrize("activity,definition_id", shipped_problems())
def test_shipped_problems_compile(activity, definition_id):
    problem = load_problem(activity, definition_id)
    compiled = problem.compile()
    assert problem.name == f"{activity}-{definition_id}"
    assert problem.measure and compiled.binary_goal_head_idxs and compiled.pose_settled_head_idxs
    assert set(problem.measure_pose_reached_args) <= set(compiled.all_pose_reached_args)


def test_groups_expand_pointwise_and_keep_aggregates():
    problem = parse_problem_string(GROUP_PROBLEM, "configuration_matching", 0)
    members = ["toy_dice.n.01_1", "toy_dice.n.01_2"]
    assert problem.init == [["ontop", m, "tray.n.01_1"] for m in members]
    assert ["layout_reached", "?targets_group.n.01_1"] in problem.goal
    assert [["not", ["grasped", "?agent.n.01_1", f"?{m}"]] for m in members] == [
        g for g in problem.goal if g[0] == "not"
    ]
    assert problem.goal_pose_settled_args == members
    compiled = problem.compile()
    assert compiled.is_gating("layout_reached", "targets_group.n.01_1")
    assert compiled.in_measure("layout_reached", "targets_group.n.01_1")
    assert compiled.pose_settled_insts == members


@pytest.mark.parametrize(
    "old,new,match",
    [
        ("(:domain ispatialgym)", "(:domain behavior)", "domain"),
        ("configuration_matching-0)", "configuration_matching-3)", "does not match"),
        (
            "(layout_reached ?targets_group.n.01_1)\n    )\n)",
            "(ontop ?tray.n.01_1 ?tray.n.01_1)\n    )\n)",
            ":measure only accepts",
        ),
        ("(ontop targets_group.n.01_1 tray.n.01_1)", "(pose_reached tray.n.01_1)", "must not contain"),
    ],
)
def test_malformed_problems_raise(old, new, match):
    assert old in GROUP_PROBLEM
    with pytest.raises(ISGProblemError, match=match):
        parse_problem_string(GROUP_PROBLEM.replace(old, new), "configuration_matching", 0)
