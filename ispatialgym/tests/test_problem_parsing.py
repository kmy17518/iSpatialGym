"""Parsing, validation, group expansion and compilation of the shipped iSpatialGym problem files (no simulator)."""

import os
import tempfile

import pytest

from ispatialgym.activity import (
    ISGProblemError,
    collect_predicate_args,
    list_activities,
    list_problem_ids,
    load_problem,
)
from ispatialgym.activity.problem import parse_problem_string

# Expected structure per problem: (alias, binary-success predicates, pose_settled args, gating pose_reached args,
# gating layout_reached args, non-gating (:measure) pose_reached args). Binary predicates are the multiset of
# predicate tokens of the expanded binary goal.
EXPECTED = {
    ("camera_relocalization", 0): (None, ["pose_reached"], ["agent.n.01_1"], ["agent.n.01_1"], [], []),
    ("object_scaling", 0): (None, ["grasped", "pose_reached"], ["mug.n.04_1", "agent.n.01_1"], ["mug.n.04_1"], [], []),
    ("mental_rotation", 0): (None, ["grasped", "pose_reached"], ["mug.n.04_1", "agent.n.01_1"], ["mug.n.04_1"], [], []),
    ("disembedding", 0): (
        None,
        ["ontop", "not ontop", "not ontop", "not ontop", "not grasped"],
        ["mug.n.04_1", "agent.n.01_1"],
        [],
        [],
        ["mug.n.04_1"],
    ),
    ("configuration_matching", 0): (
        "block_design",
        ["ontop"] * 4 + ["layout_reached"] + ["not grasped"] * 4,
        ["toy_dice.n.01_1", "toy_dice.n.01_2", "toy_dice.n.01_3", "toy_dice.n.01_4", "agent.n.01_1"],
        [],
        ["blocks_group.n.01_1"],
        ["blocks_group.n.01_1"],
    ),
    ("configuration_matching", 1): (
        "articulation",
        ["open", "layout_reached", "not grasped"],
        ["cabinet.n.01_1", "agent.n.01_1"],
        [],
        ["cabinet.n.01_1"],
        [],
    ),
    ("configuration_matching", 2): (
        "articulation",
        ["not open", "layout_reached", "not grasped"],
        ["cabinet.n.01_1", "agent.n.01_1"],
        [],
        ["cabinet.n.01_1"],
        [],
    ),
    ("configuration_matching", 3): (
        "attachment",
        ["attached", "ontop", "layout_reached", "not grasped", "not grasped"],
        ["digital_camera.n.01_1", "camera_tripod.n.01_1", "agent.n.01_1"],
        [],
        ["attachment_group.n.01_1"],
        ["attachment_group.n.01_1"],
    ),
    ("world_alignment", 0): (
        "table_surface",
        ["ontop", "pose_reached", "not grasped"],
        ["fishing_rod.n.01_1", "agent.n.01_1"],
        ["fishing_rod.n.01_1"],
        [],
        [],
    ),
    ("world_alignment", 1): (
        "world_vertical",
        ["pose_reached", "not grasped"],
        ["painting.n.01_1", "agent.n.01_1"],
        ["painting.n.01_1"],
        [],
        [],
    ),
    ("world_alignment", 2): (
        "world_horizontal",
        ["pose_reached", "not grasped"],
        ["towel_rack.n.01_1", "agent.n.01_1"],
        ["towel_rack.n.01_1"],
        [],
        [],
    ),
}
for i, rel in enumerate(["ontop", "under", "inside", "nextto", "touching"]):
    EXPECTED[("inter_object_relation_satisfaction", i)] = (
        rel,
        [rel, "not grasped", "not grasped"],
        ["apple.n.01_1", "bowl.n.01_1", "agent.n.01_1"],
        [],
        [],
        ["apple.n.01_1", "bowl.n.01_1"],
    )


def _binary_tokens(problem):
    out = []
    for conj in problem.binary_goal:
        if conj[0] == "not":
            out.append(f"not {conj[1][0]}")
        else:
            out.append(conj[0])
    return out


def test_all_shipped_problems_are_listed():
    found = {(act, pid) for act in list_activities() for pid in list_problem_ids(act)}
    assert found == set(EXPECTED)


@pytest.mark.parametrize("activity,definition_id", sorted(EXPECTED))
def test_shipped_problem_structure(activity, definition_id):
    alias, binary, settled, pose_reached, layout_reached, measured = EXPECTED[(activity, definition_id)]
    problem = load_problem(activity, definition_id)
    compiled = problem.compile()
    assert problem.name == f"{activity}-{definition_id}"
    assert problem.alias == alias
    assert sorted(_binary_tokens(problem)) == sorted(binary)
    assert compiled.pose_settled_insts == settled
    assert compiled.pose_reached_args == pose_reached
    assert compiled.layout_reached_args == layout_reached
    assert compiled.measure_pose_reached_args == measured
    assert compiled.measure_layout_reached_args == []
    assert compiled.all_pose_reached_args == pose_reached + measured
    for arg in pose_reached:
        assert compiled.is_gating("pose_reached", arg)
    for arg in measured:
        assert not compiled.is_gating("pose_reached", arg)
    # measured atoms never appear in the compiled goal
    assert not any(a in measured for head in compiled.goal_conditions for a in collect_predicate_args(head, "pose_reached"))
    # Binary heads + pose_settled heads partition the goal
    assert sorted(compiled.binary_goal_head_idxs + compiled.pose_settled_head_idxs) == list(range(len(problem.goal)))
    assert len(compiled.pose_settled_head_idxs) == len(settled)
    # Exactly one ground option for these conjunctive goals
    assert len(compiled.ground_goal_state_options) == 1
    # No group symbol survives in a pointwise atom
    for conj in problem.goal:
        lit = conj[1] if conj[0] == "not" else conj
        if lit[0] not in ("pose_reached", "layout_reached"):
            assert not any(problem.is_group(a) for a in lit[1:]), conj


@pytest.mark.parametrize("activity,definition_id", sorted(EXPECTED))
def test_expanded_bddl_round_trips(activity, definition_id):
    problem = load_problem(activity, definition_id)
    again = parse_problem_string(problem.expanded_bddl(), activity, definition_id)
    assert again.goal == problem.goal
    assert again.init == problem.init
    assert again.groups == problem.groups
    assert again.alias == problem.alias
    assert again.measure == problem.measure


def test_expanded_bddl_is_stock_compatible():
    """The expanded text compiles with the stock bddl pipeline (unknown sections are ignored by parse_problem)."""
    from bddl.activity import Conditions, get_goal_conditions, get_object_scope

    problem = load_problem("configuration_matching", 0)
    conds = Conditions("configuration_matching", 0, "ispatialgym", predefined_problem=problem.expanded_bddl())
    scope = get_object_scope(conds) | set(problem.groups)
    goals = get_goal_conditions(conds, scope)
    assert len(goals) == len(problem.goal)


def test_compiled_goal_evaluates_with_callback():
    problem = load_problem("disembedding", 0)
    compiled = problem.compile()
    seen = []

    def evaluate(predicate_cls, *entities):
        seen.append((predicate_cls.__name__, entities))
        # target on the destination tray, distractors not, nothing grasped, everything settled
        if predicate_cls.__name__ == "OnTop":
            return entities == ("mug.n.04_1", "tray.n.01_2")
        if predicate_cls.__name__ == "Grasped":
            return False
        return True

    ok, status = compiled.check_goal(evaluate)
    assert ok and not status["unsatisfied"]
    assert ("OnTop", ("mug.n.04_2", "tray.n.01_2")) in seen


# ----------------------------------------------------------------------------------------------------------------
# Malformed problems
# ----------------------------------------------------------------------------------------------------------------

HEADER = "(define (problem t-0) (:domain ispatialgym)"
OBJECTS = "(:objects mug.n.04_1 mug.n.04_2 - mug.n.04 tray.n.01_1 - tray.n.01 floor.n.01_1 - floor.n.01 agent.n.01_1 - agent.n.01)"
GROUPS = "(:groups mug.n.04_1 mug.n.04_2 - g.n.01_1)"
INIT = "(:init (ontop agent.n.01_1 floor.n.01_1))"


def _parse(body):
    return parse_problem_string(body, "t", 0)


def test_group_expansion_rules():
    p = _parse(
        f"{HEADER} {OBJECTS} {GROUPS} (:init (ontop g.n.01_1 tray.n.01_1) (not (ontop g.n.01_1 tray.n.01_1)))"
        "(:goal (and (ontop ?g.n.01_1 ?tray.n.01_1) (not (grasped ?agent.n.01_1 ?g.n.01_1)) (pose_reached ?g.n.01_1)"
        " (layout_reached ?g.n.01_1) (or (ontop ?g.n.01_1 ?tray.n.01_1) (ontop ?mug.n.04_1 ?tray.n.01_1))"
        " (pose_settled ?g.n.01_1))))"
    )
    assert p.init == [
        ["ontop", "mug.n.04_1", "tray.n.01_1"],
        ["ontop", "mug.n.04_2", "tray.n.01_1"],
        ["not", ["ontop", "mug.n.04_1", "tray.n.01_1"]],
        ["not", ["ontop", "mug.n.04_2", "tray.n.01_1"]],
    ]
    assert p.goal == [
        ["ontop", "?mug.n.04_1", "?tray.n.01_1"],
        ["ontop", "?mug.n.04_2", "?tray.n.01_1"],
        ["not", ["grasped", "?agent.n.01_1", "?mug.n.04_1"]],
        ["not", ["grasped", "?agent.n.01_1", "?mug.n.04_2"]],
        ["pose_reached", "?g.n.01_1"],
        ["layout_reached", "?g.n.01_1"],
        [
            "or",
            ["and", ["ontop", "?mug.n.04_1", "?tray.n.01_1"], ["ontop", "?mug.n.04_2", "?tray.n.01_1"]],
            ["ontop", "?mug.n.04_1", "?tray.n.01_1"],
        ],
        ["pose_settled", "?mug.n.04_1"],
        ["pose_settled", "?mug.n.04_2"],
    ]
    compiled = p.compile()
    assert compiled.pose_reached_args == ["g.n.01_1"]
    assert compiled.pose_settled_insts == ["mug.n.04_1", "mug.n.04_2"]
    assert "g.n.01_1" in compiled.object_scope


GOAL = "(:goal (and (ontop ?mug.n.04_1 ?tray.n.01_1) (pose_reached ?mug.n.04_2)))"


def test_measure_section():
    p = _parse(f"{HEADER} {OBJECTS} {GROUPS} {INIT} {GOAL} (:measure (pose_reached ?mug.n.04_1) (layout_reached ?g.n.01_1)))")
    assert p.measure == [["pose_reached", "?mug.n.04_1"], ["layout_reached", "?g.n.01_1"]]
    compiled = p.compile()
    assert compiled.pose_reached_args == ["mug.n.04_2"]  # gating
    assert compiled.measure_pose_reached_args == ["mug.n.04_1"]  # reported only
    assert compiled.measure_layout_reached_args == ["g.n.01_1"]
    assert compiled.all_pose_reached_args == ["mug.n.04_2", "mug.n.04_1"]
    assert compiled.is_gating("pose_reached", "mug.n.04_2") and not compiled.is_gating("pose_reached", "mug.n.04_1")
    # the measured atoms are not part of the goal (only the gating one is evaluated by check_goal)
    seen = []
    compiled.check_goal(lambda cls, *ents: seen.append((cls.__name__, ents)) or True)
    assert ("PoseReached", ("mug.n.04_2",)) in seen and ("PoseReached", ("mug.n.04_1",)) not in seen
    assert "(:measure" in p.expanded_bddl()


@pytest.mark.parametrize(
    "body,match",
    [
        (f"{HEADER} {OBJECTS} {INIT} {GOAL} (:measure (ontop ?mug.n.04_1 ?tray.n.01_1)))", "only accepts"),
        (f"{HEADER} {OBJECTS} {INIT} {GOAL} (:measure (pose_reached ?mug.n.04_2)))", "already a gating"),
        (f"{HEADER} {OBJECTS} {INIT} {GOAL} (:measure (pose_reached ?mug.n.04_1) (pose_reached ?mug.n.04_1)))", "more than once"),
        (f"{HEADER} {OBJECTS} {INIT} {GOAL} (:measure (pose_reached ?bowl.n.01_1)))", "not a declared"),
        (f"{HEADER} {OBJECTS} {INIT} {GOAL} (:measure (pose_reached ?mug.n.04_1 ?tray.n.01_1)))", "exactly one"),
        (f"{HEADER} {OBJECTS} {INIT} (:measure (pose_reached ?mug.n.04_1)) {GOAL})", "after"),
    ],
)
def test_malformed_measure_sections_are_rejected(body, match):
    with pytest.raises(ISGProblemError, match=match):
        _parse(body)


@pytest.mark.parametrize(
    "body,match",
    [
        (f"(define (problem t-0) (:domain behavior-1k) {OBJECTS} {INIT} (:goal (ontop ?mug.n.04_1 ?tray.n.01_1)))", "domain"),
        (f"{HEADER} {OBJECTS} (:groups mug.n.04_1 mug.n.04_9 - g.n.01_1) {INIT} (:goal (ontop ?mug.n.04_1 ?tray.n.01_1)))", "not declared"),
        (f"{HEADER} {OBJECTS} (:groups mug.n.04_1 - g.n.01_1 mug.n.04_1 - h.n.01_1) {INIT} (:goal (ontop ?mug.n.04_1 ?tray.n.01_1)))", "belongs to both"),
        (f"{HEADER} {OBJECTS} (:groups mug.n.04_1 - g.n.01_1 g.n.01_1 - h.n.01_1) {INIT} (:goal (ontop ?mug.n.04_1 ?tray.n.01_1)))", "nest"),
        (f"{HEADER} {OBJECTS} (:groups mug.n.04_1 - tray.n.01_1) {INIT} (:goal (ontop ?mug.n.04_1 ?tray.n.01_1)))", "must not be declared"),
        (f"{HEADER} {OBJECTS} {GROUPS} (:init (pose_reached mug.n.04_1)) (:goal (ontop ?mug.n.04_1 ?tray.n.01_1)))", "must not contain"),
        (f"{HEADER} {OBJECTS} {GROUPS} {INIT} (:goal (and (forall (?m - mug.n.04) (ontop ?g.n.01_1 ?tray.n.01_1)))))", "quantifier"),
        (f"{HEADER} {OBJECTS} (:groups mug.n.04_1 - g.n.01_1 mug.n.04_2 - h.n.01_1) {INIT} (:goal (ontop ?g.n.01_1 ?h.n.01_1)))", "at most one group"),
        (f"{HEADER} {OBJECTS} {GROUPS} {INIT} (:goal (and (or (pose_settled ?mug.n.04_1) (ontop ?mug.n.04_1 ?tray.n.01_1)))))", "direct child"),
        (f"{HEADER} {OBJECTS} {INIT} (:goal (ontop ?mug.n.04_1 ?bowl.n.01_1)))", "not a declared"),
        (f"{HEADER} {OBJECTS} {INIT} (:goal (cooked ?mug.n.04_1)))", "unknown predicate"),
        (f"{HEADER} {OBJECTS} {INIT} (:goal (ontop ?mug.n.04_1)))", "argument"),
        (f"{HEADER} {OBJECTS} {INIT} (:goal (pose_settled ?mug.n.04_1)))", "at least one"),
        (f"{HEADER} {OBJECTS} (:alias x) {INIT} (:goal (ontop ?mug.n.04_1 ?tray.n.01_1)))", "directly follow"),
        (f"(define (problem t-1) (:domain ispatialgym) {OBJECTS} {INIT} (:goal (ontop ?mug.n.04_1 ?tray.n.01_1)))", "does not match"),
    ],
)
def test_malformed_problems_are_rejected(body, match):
    with pytest.raises(ISGProblemError, match=match):
        _parse(body)


def test_list_problem_ids_requires_contiguity():
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "act"))
        open(os.path.join(root, "act", "problem1.bddl"), "w").close()
        with pytest.raises(ISGProblemError, match="contiguous"):
            list_problem_ids("act", root=root)
