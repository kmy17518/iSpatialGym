"""Parsing, validation, group expansion and compilation of the shipped iSpatialGym problem files (no simulator);
the ``alignment`` activity with its retired name ``world_alignment``; the ``problems.json`` variant registry; and the
v2 problem shapes of ``docs/task_bddl_formats.md``."""

import json
import os
import tempfile

import pytest

from ispatialgym.activity import (
    VARIANT_RE,
    LEGACY_ACTIVITIES,
    ISGProblemError,
    activity_dir,
    variant_of,
    canonical_activity,
    collect_predicate_args,
    id_of_variant,
    list_activities,
    list_problem_ids,
    load_problem,
    load_problem_registry,
    problem_variants,
    problem_path,
    problem_registry_path,
    same_activity,
    validate_problem_registry,
)
from ispatialgym import ASSETS_ROOT
from ispatialgym.activity.problem import parse_problem_file, parse_problem_string

#: the problems EXPECTED describes (the v1 templates and the first grounded picks); the shipped tree holds the eval
#: release's problems only
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bddl")

# Expected structure per problem: (variant, binary-success predicates, pose_settled args, :goal pose_reached args,
# :goal layout_reached args, :measure pose_reached args, :measure layout_reached args) - an atom may be in both sections
# (2026-10-14). Binary predicates are the multiset of predicate tokens of the expanded binary goal.
EXPECTED = {
    ('camera_relocalization', 0): ('camera_relocalization-standard', ['pose_reached'], ['agent.n.01_1'], ['agent.n.01_1'], [], ['agent.n.01_1'], []),
    ('object_scaling', 0): ('object_scaling-canned_food', ['grasped', 'pose_reached'], ['canned_food.n.01_1', 'agent.n.01_1'], ['canned_food.n.01_1'], [], ['canned_food.n.01_1'], []),
    ('mental_rotation', 0): ('mental_rotation-classroom_mock_up', ['grasped', 'pose_reached'], ['mock-up.n.01_1', 'agent.n.01_1'], ['mock-up.n.01_1'], [], ['mock-up.n.01_1'], []),
    ('disembedding', 0): (None, ['ontop', 'not ontop', 'not ontop', 'not ontop', 'not grasped'], ['mug.n.04_1', 'agent.n.01_1'], [], [], ['mug.n.04_1'], []),
    ('configuration_matching', 0): ('block_design', ['ontop', 'ontop', 'ontop', 'ontop', 'layout_reached', 'not grasped', 'not grasped', 'not grasped', 'not grasped'], ['toy_dice.n.01_1', 'toy_dice.n.01_2', 'toy_dice.n.01_3', 'toy_dice.n.01_4', 'agent.n.01_1'], [], ['blocks_group.n.01_1'], ['blocks_group.n.01_1'], ['blocks_group.n.01_1']),
    ('configuration_matching', 1): ('articulation', ['open', 'not grasped'], ['cabinet.n.01_1', 'agent.n.01_1'], [], [], [], ['cabinet.n.01_1']),
    ('configuration_matching', 2): ('articulation', ['not open', 'not grasped'], ['cabinet.n.01_1', 'agent.n.01_1'], [], [], [], ['cabinet.n.01_1']),
    ('configuration_matching', 3): ('attachment', ['attached', 'ontop', 'layout_reached', 'not grasped', 'not grasped'], ['digital_camera.n.01_1', 'camera_tripod.n.01_1', 'agent.n.01_1'], [], ['attachment_group.n.01_1'], ['attachment_group.n.01_1'], ['attachment_group.n.01_1']),
    ('alignment', 0): ('table_surface', ['ontop', 'pose_reached', 'not grasped'], ['fishing_rod.n.01_1', 'agent.n.01_1'], ['fishing_rod.n.01_1'], [], ['fishing_rod.n.01_1'], []),
    ('alignment', 1): ('world_vertical', ['pose_reached', 'not grasped'], ['painting.n.01_1', 'agent.n.01_1'], ['painting.n.01_1'], [], ['painting.n.01_1'], []),
    ('inter_object_relation_satisfaction', 0): ('ontop', ['ontop', 'not grasped', 'not grasped'], ['apple.n.01_1', 'bowl.n.01_1', 'agent.n.01_1'], [], [], ['apple.n.01_1', 'bowl.n.01_1'], []),
    ('inter_object_relation_satisfaction', 1): ('under', ['under', 'not grasped', 'not grasped'], ['apple.n.01_1', 'bowl.n.01_1', 'agent.n.01_1'], [], [], ['apple.n.01_1', 'bowl.n.01_1'], []),
    ('inter_object_relation_satisfaction', 2): ('inside', ['inside', 'not grasped', 'not grasped'], ['apple.n.01_1', 'bowl.n.01_1', 'agent.n.01_1'], [], [], ['apple.n.01_1', 'bowl.n.01_1'], []),
    ('inter_object_relation_satisfaction', 3): ('nextto', ['nextto', 'not grasped', 'not grasped'], ['apple.n.01_1', 'bowl.n.01_1', 'agent.n.01_1'], [], [], ['apple.n.01_1', 'bowl.n.01_1'], []),
    ('inter_object_relation_satisfaction', 4): ('touching', ['touching', 'not grasped', 'not grasped'], ['apple.n.01_1', 'bowl.n.01_1', 'agent.n.01_1'], [], [], ['apple.n.01_1', 'bowl.n.01_1'], []),
}


def _binary_tokens(problem):
    out = []
    for conj in problem.binary_goal:
        if conj[0] == "not":
            out.append(f"not {conj[1][0]}")
        else:
            out.append(conj[0])
    return out


def _shipped_problems():
    return sorted((act, pid) for act in list_activities() for pid in list_problem_ids(act))


def test_all_shipped_problems_are_listed():
    """Every customs entry's problem (``isg_customs.json``: iddl family + definition id) is shipped - this repository also ships
    the v1 templates and the problems of earlier groundings next to them (the eval release ships its subtasks' only); the
    retired directory name is gone."""
    with open(os.path.join(ASSETS_ROOT, "task_specs", "isg_customs.json")) as f:
        customs = json.load(f)
    expected = {(e["iddl"].split("/")[0], int(e["definition_id"])) for k, e in customs.items() if not k.startswith("_")}
    assert expected <= set(_shipped_problems())
    assert "world_alignment" not in list_activities()
    assert "alignment" in list_activities()


@pytest.mark.parametrize("activity,definition_id", _shipped_problems())
def test_every_shipped_problem_compiles(activity, definition_id):
    problem = load_problem(activity, definition_id)
    assert problem.name == f"{activity}-{definition_id}"
    assert problem.variant is None or VARIANT_RE.match(problem.variant)
    compiled = problem.compile()
    assert compiled.binary_goal_head_idxs
    assert variant_of(activity, definition_id) == problem.variant
    # one variant = one BDDL in v2; the v1 articulation templates (open / close) still share ``articulation``
    if problem.variant is not None and list(problem_variants(activity).values()).count(problem.variant) == 1:
        assert id_of_variant(activity, problem.variant) == definition_id


@pytest.mark.parametrize("activity,definition_id", sorted(EXPECTED))
def test_shipped_problem_structure(activity, definition_id):
    variant, binary, settled, pose_reached, layout_reached, measure_pose, measure_layout = EXPECTED[(activity, definition_id)]
    problem = load_problem(activity, definition_id, root=FIXTURES)
    compiled = problem.compile()
    assert problem.name == f"{activity}-{definition_id}"
    assert problem.variant == variant
    assert sorted(_binary_tokens(problem)) == sorted(binary)
    assert compiled.pose_settled_insts == settled
    assert compiled.pose_reached_args == pose_reached
    assert compiled.layout_reached_args == layout_reached
    assert compiled.measure_pose_reached_args == measure_pose
    assert compiled.measure_layout_reached_args == measure_layout
    assert compiled.all_pose_reached_args == list(dict.fromkeys(pose_reached + measure_pose))
    for pred, goal_args, measure_args in (("pose_reached", pose_reached, measure_pose), ("layout_reached", layout_reached, measure_layout)):
        for arg in set(goal_args) | set(measure_args):
            assert compiled.is_gating(pred, arg) == (arg in goal_args) and compiled.in_measure(pred, arg) == (arg in measure_args)
    # an atom only in :measure never appears in the compiled goal
    only_measured = [a for a in measure_pose if a not in pose_reached]
    assert not any(a in only_measured for head in compiled.goal_conditions for a in collect_predicate_args(head, "pose_reached"))


@pytest.mark.parametrize("activity,definition_id", sorted(EXPECTED))
def test_expanded_bddl_round_trips(activity, definition_id):
    problem = load_problem(activity, definition_id, root=FIXTURES)
    again = parse_problem_string(problem.expanded_bddl(), activity, definition_id)
    assert again.goal == problem.goal
    assert again.init == problem.init
    assert again.groups == problem.groups
    assert again.variant == problem.variant
    assert again.measure == problem.measure


def test_expanded_bddl_is_stock_compatible():
    """The expanded text compiles with the stock bddl pipeline (unknown sections are ignored by parse_problem)."""
    from bddl.activity import Conditions, get_goal_conditions, get_object_scope

    problem = load_problem("configuration_matching", 0, root=FIXTURES)
    conds = Conditions("configuration_matching", 0, "ispatialgym", predefined_problem=problem.expanded_bddl())
    scope = get_object_scope(conds) | set(problem.groups)
    goals = get_goal_conditions(conds, scope)
    assert len(goals) == len(problem.goal)


def test_compiled_goal_evaluates_with_callback():
    problem = load_problem("disembedding", 0, root=FIXTURES)
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


def test_an_atom_may_be_in_goal_and_measure():
    """2026-10-14: a ``pose_reached`` / ``layout_reached`` atom may sit in :goal (gating) and in :measure (judged by geometric
    success) at once; it is evaluated once per target."""
    p = _parse(f"{HEADER} {OBJECTS} {GROUPS} {INIT} {GOAL} (:measure (pose_reached ?mug.n.04_2) (layout_reached ?g.n.01_1)))")
    compiled = p.compile()
    assert compiled.pose_reached_args == ["mug.n.04_2"] and compiled.measure_pose_reached_args == ["mug.n.04_2"]
    assert compiled.all_pose_reached_args == ["mug.n.04_2"]
    assert compiled.is_gating("pose_reached", "mug.n.04_2") and compiled.in_measure("pose_reached", "mug.n.04_2")
    assert not compiled.is_gating("layout_reached", "g.n.01_1") and compiled.in_measure("layout_reached", "g.n.01_1")
    seen = []
    compiled.check_goal(lambda cls, *ents: seen.append((cls.__name__, ents)) or True)
    assert ("PoseReached", ("mug.n.04_2",)) in seen and ("LayoutReached", ("g.n.01_1",)) not in seen


@pytest.mark.parametrize(
    "body,match",
    [
        (f"{HEADER} {OBJECTS} {INIT} {GOAL} (:measure (ontop ?mug.n.04_1 ?tray.n.01_1)))", "only accepts"),
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
        (f"{HEADER} {OBJECTS} (:variant x) {INIT} (:goal (ontop ?mug.n.04_1 ?tray.n.01_1)))", "directly follow"),
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
        with pytest.raises(ISGProblemError, match=r"contiguous.*found ids \[1\], missing \[0\]"):
            list_problem_ids("act", root=root)
        open(os.path.join(root, "act", "problem0.bddl"), "w").close()
        open(os.path.join(root, "act", "problem3.bddl"), "w").close()
        with pytest.raises(ISGProblemError, match=r"found ids \[0, 1, 3\], missing \[2\]"):
            list_problem_ids("act", root=root)
        with pytest.raises(FileNotFoundError, match="No activity directory"):
            list_problem_ids("other", root=root)


# ----------------------------------------------------------------------------------------------------------------
# Activity names: ``alignment`` and its retired name ``world_alignment``
# ----------------------------------------------------------------------------------------------------------------


def test_legacy_activity_mapping():
    assert LEGACY_ACTIVITIES == {"world_alignment": "alignment"}
    assert canonical_activity("world_alignment") == "alignment"
    assert canonical_activity("alignment") == "alignment"
    assert canonical_activity("disembedding") == "disembedding"
    assert same_activity("world_alignment", "alignment") and not same_activity("alignment", "disembedding")


def test_legacy_activity_name_loads_renamed_templates():
    assert problem_path("world_alignment", 0) == problem_path("alignment", 0)
    assert problem_path("alignment", 0).endswith(os.path.join("alignment", "problem0.bddl"))
    assert activity_dir("world_alignment").endswith(os.sep + "alignment")
    ids = list_problem_ids("alignment", FIXTURES)
    assert list_problem_ids("world_alignment", FIXTURES) == ids == [0, 1]  # the v1 templates
    problem = load_problem("world_alignment", 0, root=FIXTURES)
    assert problem.activity == "alignment" and problem.name == "alignment-0" and problem.canonical_name == "alignment-0"
    assert problem.variant == "table_surface"
    assert problem.path == problem_path("alignment", 0, FIXTURES)


def test_problem_name_accepts_canonical_and_retired_activity():
    body = "{} (:objects fishing_rod.n.01_1 - fishing_rod.n.01 floor.n.01_1 - floor.n.01 agent.n.01_1 - agent.n.01) {} (:goal (and (pose_reached ?fishing_rod.n.01_1))))"
    # v1 concrete problems (runlogs/runpod/samp) are named world_alignment-N and parsed under the v1 key ...
    legacy = parse_problem_string(body.format("(define (problem world_alignment-1) (:domain ispatialgym)", INIT), "world_alignment", 1)
    assert legacy.activity == "world_alignment" and legacy.canonical_activity == "alignment"
    assert legacy.name == "world_alignment-1" and legacy.canonical_name == "alignment-1"
    # ... or under the canonical activity; a canonical name is accepted under the retired activity too
    assert parse_problem_string(body.format("(define (problem world_alignment-1) (:domain ispatialgym)", INIT), "alignment", 1).activity == "alignment"
    assert parse_problem_string(body.format("(define (problem alignment-1) (:domain ispatialgym)", INIT), "world_alignment", 1).activity == "world_alignment"
    # the definition id may be inferred from the name
    inferred = parse_problem_string(body.format("(define (problem alignment-7) (:domain ispatialgym)", INIT), "alignment", None)
    assert inferred.definition_id == 7
    with pytest.raises(ISGProblemError, match="does not match 'alignment-1' / 'world_alignment-1'"):
        parse_problem_string(body.format("(define (problem disembedding-1) (:domain ispatialgym)", INIT), "alignment", 1)
    with pytest.raises(ISGProblemError, match="does not match"):
        parse_problem_string(body.format("(define (problem alignment-2) (:domain ispatialgym)", INIT), "alignment", 1)
    with pytest.raises(ISGProblemError, match="shape"):
        parse_problem_string(body.format("(define (problem alignment) (:domain ispatialgym)", INIT), "alignment", 1)


def test_legacy_root_directory_is_found_under_either_name():
    """A v1 template directory still says ``activity_definitions/world_alignment``; both names resolve to it and the
    problem keeps the directory's activity."""
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "world_alignment"))
        with open(problem_path("alignment", 0, FIXTURES), "r") as f:
            text = f.read().replace("(define (problem alignment-0)", "(define (problem world_alignment-0)")
        with open(os.path.join(root, "world_alignment", "problem0.bddl"), "w") as f:
            f.write(text)
        assert activity_dir("alignment", root) == os.path.join(root, "world_alignment")
        assert list_activities(root) == ["world_alignment"]
        assert list_problem_ids("alignment", root) == [0]
        for name in ("alignment", "world_alignment"):
            problem = load_problem(name, 0, root=root)
            assert problem.activity == "world_alignment" and problem.canonical_activity == "alignment"
            assert problem.variant == "table_surface"
        assert parse_problem_file(os.path.join(root, "world_alignment", "problem0.bddl")).name == "world_alignment-0"
        # the canonical directory wins once it exists
        os.makedirs(os.path.join(root, "alignment"))
        assert activity_dir("world_alignment", root) == os.path.join(root, "alignment")
        with pytest.raises(FileNotFoundError):
            load_problem("world_alignment", 0, root=root)


def test_tro_activity_check_accepts_retired_name():
    from ispatialgym.eval.targets import SCHEMA_VERSION, ISGTroMeta, TROValidationError
    from ispatialgym.eval.tro import validate_tro_against_problem

    compiled = load_problem("alignment", 0, root=FIXTURES).compile()

    def meta(activity, variant="table_surface"):
        return ISGTroMeta.from_dict(
            {
                "schema_version": SCHEMA_VERSION,
                "activity": activity,
                "activity_definition_id": 0,
                "task_variant": variant,
                "pose_targets": {"fishing_rod.n.01_1": {"orientation": [0, 0, 0, 1], "orientation_tolerance": 0.05}},
            }
        )

    validate_tro_against_problem(meta("alignment"), compiled, require_bindings=False)
    validate_tro_against_problem(meta("world_alignment"), compiled, require_bindings=False)  # v1 TROs
    with pytest.raises(TROValidationError, match="TRO is for 'disembedding-0'"):
        validate_tro_against_problem(meta("disembedding"), compiled, require_bindings=False)
    with pytest.raises(TROValidationError, match="task_variant"):
        validate_tro_against_problem(meta("world_alignment", variant="alignment-axial-fishing_rod-table"), compiled, require_bindings=False)


def test_v1_bddl_writer_and_sampler_dispatch_accept_both_names(monkeypatch):
    import sys
    import types

    from ispatialgym.sampling.bddl_writer import Bindings, build_problem
    from ispatialgym.sampling.tasks import get_sampler_class

    # the v1 sampler module imports OmniGibson; stub it so that only the dispatch is exercised here
    stub = types.ModuleType("ispatialgym.sampling.tasks.world_alignment")
    for cls_name in ("TableSurfaceSampler", "WorldVerticalSampler"):
        setattr(stub, cls_name, type(cls_name, (), {}))
    monkeypatch.setitem(sys.modules, "ispatialgym.sampling.tasks.world_alignment", stub)

    for activity in ("world_alignment", "alignment"):
        b = Bindings()
        b.add("target", "fishing_rod.n.01", category="fishing_rod", model="zhmrtc", dataset="behavior")
        b.add("table", "table.n.02", category="breakfast_table", scene_object_name="table_0")
        b.add("floor", "floor.n.01", scene_object_name="floors")
        problem = build_problem(activity, 0, b)
        # rendered under the name it was asked for: v1 output stays ``world_alignment-0``
        assert problem.name == f"{activity}-0" and problem.canonical_name == "alignment-0"
        assert problem.text.startswith(f"(define (problem {activity}-0)")
        assert problem.variant == "table_surface"
        assert get_sampler_class(f"{activity}-0").__name__ == "TableSurfaceSampler"
        assert get_sampler_class(f"{activity}-1").__name__ == "WorldVerticalSampler"
    with pytest.raises(ValueError, match="world_alignment has no definition 2"):  # world horizontal was retired
        build_problem("world_alignment", 2, b)
    with pytest.raises(KeyError):
        get_sampler_class("alignment-2")


# ----------------------------------------------------------------------------------------------------------------
# problems.json registry
# ----------------------------------------------------------------------------------------------------------------

V2_VARIANT = "inter_object_relation_satisfaction-ontop_small_scale-apple-plate-breakfast_table"
IORS_OBJECTS = (
    "(:objects apple.n.01_1 - apple.n.01 plate.n.04_1 - plate.n.04 breakfast_table.n.01_1 - breakfast_table.n.01"
    " floor.n.01_1 - floor.n.01 agent.n.01_1 - agent.n.01)"
)


def _iors_problem(activity, pid, variant, relation="ontop"):
    variant_section = f"(:variant {variant})" if variant else ""
    return (
        f"(define (problem {activity}-{pid}) (:domain ispatialgym) {variant_section} {IORS_OBJECTS}"
        " (:init (ontop apple.n.01_1 breakfast_table.n.01_1) (ontop plate.n.04_1 breakfast_table.n.01_1) (ontop agent.n.01_1 floor.n.01_1))"
        f" (:goal (and ({relation} ?apple.n.01_1 ?plate.n.04_1) (not (grasped ?agent.n.01_1 ?apple.n.01_1))"
        " (pose_settled ?apple.n.01_1) (pose_settled ?agent.n.01_1))) (:measure (pose_reached ?apple.n.01_1)))"
    )


def _write_activity(root, activity, variants, registry=None):
    """``variants``: variant per definition id (None = no variant); ``registry``: JSON value for problems.json."""
    directory = os.path.join(root, activity)
    os.makedirs(directory, exist_ok=True)
    for pid, variant in enumerate(variants):
        with open(os.path.join(directory, f"problem{pid}.bddl"), "w") as f:
            f.write(_iors_problem(activity, pid, variant))
    if registry is not None:
        with open(os.path.join(directory, "problems.json"), "w") as f:
            json.dump(registry, f)
    return directory


def test_problem_registry_lookup_and_fallback():
    with tempfile.TemporaryDirectory() as root:
        act = "inter_object_relation_satisfaction"
        second = act + "-nextto_small_scale-spatula-soup_ladle-breakfast_table"
        directory = _write_activity(root, act, [V2_VARIANT, second, None], registry={V2_VARIANT: 0, second: 1})
        assert problem_registry_path(act, root) == os.path.join(directory, "problems.json")
        assert load_problem_registry(directory) == {V2_VARIANT: 0, second: 1}
        assert load_problem_registry(os.path.join(directory, "problems.json")) == {V2_VARIANT: 0, second: 1}
        assert list_problem_ids(act, root) == [0, 1, 2]  # problems.json is not a problem file
        assert problem_variants(act, root) == {0: V2_VARIANT, 1: second, 2: None}
        assert id_of_variant(act, V2_VARIANT, root) == 0 and id_of_variant(act, second, root) == 1
        assert variant_of(act, 0, root) == V2_VARIANT and variant_of(act, 1, root) == second and variant_of(act, 2, root) is None
        assert validate_problem_registry(act, root) == {V2_VARIANT: 0, second: 1}
        problem = load_problem(act, id_of_variant(act, V2_VARIANT, root), root=root)
        assert problem.task_variant == V2_VARIANT
        with pytest.raises(ISGProblemError, match="known variants"):
            id_of_variant(act, act + "-inside_small_scale-apple-bowl-breakfast_table", root)
        with pytest.raises(FileNotFoundError, match="no problem 5"):
            variant_of(act, 5, root)

        # without a registry the variants come from the files (the shipped templates have none yet)
        os.remove(os.path.join(directory, "problems.json"))
        assert load_problem_registry(directory, missing_ok=True) == {}
        with pytest.raises(FileNotFoundError):
            load_problem_registry(directory)
        assert id_of_variant(act, second, root) == 1 and variant_of(act, 1, root) == second
        with pytest.raises(FileNotFoundError):
            validate_problem_registry(act, root)


def test_problem_registry_accepted_shapes():
    with tempfile.TemporaryDirectory() as root:
        act = "inter_object_relation_satisfaction"
        directory = _write_activity(root, act, [V2_VARIANT])
        path = os.path.join(directory, "problems.json")
        for shape in (
            {V2_VARIANT: 0},
            {"0": V2_VARIANT},
            [{"variant": V2_VARIANT, "id": 0}],
            [{"variant": V2_VARIANT, "definition_id": "0"}],
            {"activity": act, "problems": {V2_VARIANT: 0}},
            {"activity": "inter_object_relation_satisfaction", "problems": [{"variant": V2_VARIANT, "id": 0}]},
        ):
            with open(path, "w") as f:
                json.dump(shape, f)
            assert load_problem_registry(directory) == {V2_VARIANT: 0}, shape
            assert validate_problem_registry(act, root) == {V2_VARIANT: 0}


@pytest.mark.parametrize(
    "registry,match",
    [
        ({V2_VARIANT: 0, "b": 0}, "registered for both"),
        ({V2_VARIANT: -1}, "non-negative"),
        ({V2_VARIANT: 1.5}, "non-negative"),
        ({V2_VARIANT: True}, "non-negative"),
        ({"Bad-Variant": 0}, "invalid variant"),
        ([{"variant": V2_VARIANT}], "no 'id'"),
        (["x"], "must be an object"),
        (3, "must be a JSON object"),
        ({"activity": "disembedding", "problems": {V2_VARIANT: 0}}, "directory is"),
    ],
)
def test_malformed_problem_registries_are_rejected(registry, match):
    with tempfile.TemporaryDirectory() as root:
        directory = _write_activity(root, "inter_object_relation_satisfaction", [V2_VARIANT], registry=registry)
        with pytest.raises(ISGProblemError, match=match):
            load_problem_registry(directory)


def test_problem_registry_must_agree_with_the_files():
    with tempfile.TemporaryDirectory() as root:
        act = "inter_object_relation_satisfaction"
        other = act + "-nextto_small_scale-spatula-soup_ladle-breakfast_table"
        # stale registry: variant registered under an id whose file declares another variant
        _write_activity(root, act, [V2_VARIANT, other], registry={V2_VARIANT: 1, other: 0})
        with pytest.raises(ISGProblemError, match=r"registers .* -> 0 but .*problem0.bddl declares \(:variant"):
            variant_of(act, 0, root)
        with pytest.raises(ISGProblemError, match=r"registers .* -> 1 but .*problem1.bddl declares"):
            id_of_variant(act, V2_VARIANT, root)
        with pytest.raises(ISGProblemError, match="declares"):
            validate_problem_registry(act, root)
        # registered id without a file
        _write_activity(root, act, [V2_VARIANT, other], registry={V2_VARIANT: 0, other: 1, act + "-touching_small_scale-a-b-c": 2})
        with pytest.raises(ISGProblemError, match="problem2.bddl does not exist"):
            validate_problem_registry(act, root)
        with pytest.raises(ISGProblemError, match="does not exist"):
            variant_of(act, 2, root)
        # aliased file that the registry does not know
        _write_activity(root, act, [V2_VARIANT, other], registry={V2_VARIANT: 0})
        with pytest.raises(ISGProblemError, match="does not register it"):
            validate_problem_registry(act, root)
        assert id_of_variant(act, other, root) == 1  # ... but the file scan still finds it
        # a registry-less directory with two files sharing a variant cannot resolve it (one variant = one BDDL)
        _write_activity(root, act, [V2_VARIANT, V2_VARIANT])
        os.remove(os.path.join(root, act, "problems.json"))
        with pytest.raises(ISGProblemError, match="all declare"):
            id_of_variant(act, V2_VARIANT, root)


# ----------------------------------------------------------------------------------------------------------------
# Variant grammar (v2: <family>-<subtask>-<category slots>)
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant",
    [
        "table_surface",  # v1
        "block_design",
        V2_VARIANT,
        "camera_relocalization-standard",
        "camera_relocalization-obstructed-trash_can_1",
        "configuration_matching-block_design-block_4-table",
        "configuration_matching-articulation_open_large_scale-cabinet",
        "alignment-world_vertical-painting-wall",
        "disembedding-mug-bottle_of_beer_3_bottle_of_sake_1-table",
    ],
)
def test_variant_grammar_accepts_v1_and_v2_variants(variant):
    assert VARIANT_RE.match(variant)
    p = _parse(f"{HEADER} (:variant {variant}) {OBJECTS} {INIT} {GOAL})")
    assert p.variant == variant and p.task_variant == variant
    assert f"(:variant {variant})" in p.expanded_bddl()


@pytest.mark.parametrize("variant", ["-a", "a-", "a--b", "a.b", "a b", "1a", "a_-b-", "ä", "a/b", "a:b"])
def test_variant_grammar_rejects_malformed_variants(variant):
    assert not VARIANT_RE.match(variant)
    with pytest.raises(ISGProblemError, match="variant"):
        _parse(f"{HEADER} (:variant {variant}) {OBJECTS} {INIT} {GOAL})")


def test_variant_grammar_is_lowercase():
    # the regex is lowercase-only; BDDL's tokenizer lowercases the file anyway, so ``A-b`` parses as ``a-b``
    assert not VARIANT_RE.match("A-b") and not VARIANT_RE.match("a-B")
    assert _parse(f"{HEADER} (:variant Alignment-Axial-Fishing_Rod-Table) {OBJECTS} {INIT} {GOAL})").variant == "alignment-axial-fishing_rod-table"


def test_problem_files_written_before_the_rename():
    p = _parse(f"{HEADER} (:alias {V2_VARIANT}) {OBJECTS} {INIT} {GOAL})")  # legacy spelling
    assert p.variant == V2_VARIANT and f"(:variant {V2_VARIANT})" in p.expanded_bddl()
    with pytest.raises(ISGProblemError, match="Duplicate section"):
        _parse(f"{HEADER} (:alias a) (:variant a) {OBJECTS} {INIT} {GOAL})")  # legacy spelling
    with tempfile.TemporaryDirectory() as root:
        act = "inter_object_relation_satisfaction"
        directory = _write_activity(root, act, [V2_VARIANT], registry=[{"alias": V2_VARIANT, "id": 0}])  # legacy spelling
        with open(os.path.join(directory, "problem0.bddl"), "w") as f:
            f.write(_iors_problem(act, 0, V2_VARIANT).replace("(:variant ", "(:alias "))  # legacy spelling
        assert load_problem_registry(directory) == {V2_VARIANT: 0}
        assert problem_variants(act, root) == {0: V2_VARIANT} and validate_problem_registry(act, root) == {V2_VARIANT: 0}


# ----------------------------------------------------------------------------------------------------------------
# v2 problem shapes (docs/task_bddl_formats.md)
# ----------------------------------------------------------------------------------------------------------------

AGENT_ON_FLOOR = "(ontop agent.n.01_1 floor.n.01_1)"


def test_v2_obstructed_camera_relocalization_with_obstacles_group():
    p = parse_problem_string(
        "(define (problem camera_relocalization-1) (:domain ispatialgym)"
        " (:variant camera_relocalization-obstructed-trash_can_1)"
        " (:objects trash_can.n.01_1 - trash_can.n.01 floor.n.01_1 - floor.n.01 agent.n.01_1 - agent.n.01)"
        " (:groups trash_can.n.01_1 - obstacles_group.n.01_1)"
        f" (:init (ontop obstacles_group.n.01_1 floor.n.01_1) {AGENT_ON_FLOOR})"
        " (:goal (and (pose_reached ?agent.n.01_1) (pose_settled ?agent.n.01_1))))",
        "camera_relocalization",
        1,
    )
    assert p.groups == {"obstacles_group.n.01_1": ["trash_can.n.01_1"]}
    assert p.init == [["ontop", "trash_can.n.01_1", "floor.n.01_1"], ["ontop", "agent.n.01_1", "floor.n.01_1"]]
    compiled = p.compile()
    assert compiled.pose_reached_args == ["agent.n.01_1"] and compiled.pose_settled_insts == ["agent.n.01_1"]
    assert "obstacles_group.n.01_1" in compiled.object_scope
    ok, _ = compiled.check_initial_conditions(lambda cls, *ents: True)
    assert ok


def test_v2_under_small_scale_has_partner_first():
    """``(under ?partner ?target)``: the manipulated glass is set down on the coaster, so the *partner* ends under it."""
    p = parse_problem_string(
        "(define (problem inter_object_relation_satisfaction-4) (:domain ispatialgym)"
        " (:variant inter_object_relation_satisfaction-under_small_scale-water_glass-coaster-breakfast_table)"
        " (:objects water_glass.n.02_1 - water_glass.n.02 coaster.n.03_1 - coaster.n.03"
        " breakfast_table.n.01_1 - breakfast_table.n.01 floor.n.01_1 - floor.n.01 agent.n.01_1 - agent.n.01)"
        " (:init (ontop water_glass.n.02_1 breakfast_table.n.01_1) (ontop coaster.n.03_1 breakfast_table.n.01_1)"
        f" (not (under coaster.n.03_1 water_glass.n.02_1)) {AGENT_ON_FLOOR})"
        " (:goal (and (under ?coaster.n.03_1 ?water_glass.n.02_1)"
        " (not (grasped ?agent.n.01_1 ?water_glass.n.02_1)) (not (grasped ?agent.n.01_1 ?coaster.n.03_1))"
        " (pose_settled ?water_glass.n.02_1) (pose_settled ?coaster.n.03_1) (pose_settled ?agent.n.01_1)))"
        " (:measure (pose_reached ?water_glass.n.02_1) (pose_reached ?coaster.n.03_1)))",
        "inter_object_relation_satisfaction",
        4,
    )
    assert p.binary_goal[0] == ["under", "?coaster.n.03_1", "?water_glass.n.02_1"]
    assert ["not", ["under", "coaster.n.03_1", "water_glass.n.02_1"]] in p.init
    compiled = p.compile()
    assert compiled.measure_pose_reached_args == ["water_glass.n.02_1", "coaster.n.03_1"]
    seen = []
    compiled.check_goal(lambda cls, *ents: seen.append((cls.__name__, ents)) or True)
    assert ("Under", ("coaster.n.03_1", "water_glass.n.02_1")) in seen


def test_v2_under_large_scale_partner_is_the_support():
    """Fixed scene partner (bed) that is also the target's init support: declared once, no ``pose_settled`` /
    ``pose_reached`` for it, ``(under ?target ?partner)``."""
    p = parse_problem_string(
        "(define (problem inter_object_relation_satisfaction-5) (:domain ispatialgym)"
        " (:variant inter_object_relation_satisfaction-under_large_scale-basketball-bed-bed)"
        " (:objects basketball.n.02_1 - basketball.n.02 bed.n.01_1 - bed.n.01 floor.n.01_1 - floor.n.01 agent.n.01_1 - agent.n.01)"
        f" (:init (ontop basketball.n.02_1 bed.n.01_1) (not (under basketball.n.02_1 bed.n.01_1)) {AGENT_ON_FLOOR})"
        " (:goal (and (under ?basketball.n.02_1 ?bed.n.01_1) (not (grasped ?agent.n.01_1 ?basketball.n.02_1))"
        " (pose_settled ?basketball.n.02_1) (pose_settled ?agent.n.01_1)))"
        " (:measure (pose_reached ?basketball.n.02_1)))",
        "inter_object_relation_satisfaction",
        5,
    )
    assert p.instances == ["basketball.n.02_1", "bed.n.01_1", "floor.n.01_1", "agent.n.01_1"]
    compiled = p.compile()
    assert compiled.pose_settled_insts == ["basketball.n.02_1", "agent.n.01_1"]
    assert compiled.all_pose_reached_args == ["basketball.n.02_1"]
    # ontop large-scale: partner fixed, init support is another scene object
    q = parse_problem_string(
        "(define (problem inter_object_relation_satisfaction-6) (:domain ispatialgym)"
        " (:variant inter_object_relation_satisfaction-ontop_large_scale-pillow-bed-breakfast_table)"
        " (:objects pillow.n.01_1 - pillow.n.01 bed.n.01_1 - bed.n.01 breakfast_table.n.01_1 - breakfast_table.n.01"
        " floor.n.01_1 - floor.n.01 agent.n.01_1 - agent.n.01)"
        f" (:init (ontop pillow.n.01_1 breakfast_table.n.01_1) (not (ontop pillow.n.01_1 bed.n.01_1)) {AGENT_ON_FLOOR})"
        " (:goal (and (ontop ?pillow.n.01_1 ?bed.n.01_1) (not (grasped ?agent.n.01_1 ?pillow.n.01_1))"
        " (pose_settled ?pillow.n.01_1) (pose_settled ?agent.n.01_1)))"
        " (:measure (pose_reached ?pillow.n.01_1)))",
        "inter_object_relation_satisfaction",
        6,
    )
    assert q.compile().pose_settled_insts == ["pillow.n.01_1", "agent.n.01_1"]


def test_v2_alignment_wall_support_without_literal():
    p = parse_problem_string(
        "(define (problem alignment-3) (:domain ispatialgym) (:variant alignment-world_vertical-painting-wall)"
        " (:objects painting.n.01_1 - painting.n.01 wall.n.01_1 - wall.n.01 floor.n.01_1 - floor.n.01 agent.n.01_1 - agent.n.01)"
        f" (:init {AGENT_ON_FLOOR})"
        " (:goal (and (pose_reached ?painting.n.01_1) (not (grasped ?agent.n.01_1 ?painting.n.01_1))"
        " (pose_settled ?painting.n.01_1) (pose_settled ?agent.n.01_1))))",
        "world_alignment",  # retired spelling of the activity ...
        3,
    )
    assert p.canonical_activity == "alignment" and p.name == "world_alignment-3"
    assert "wall.n.01_1" in p.instances and not any("wall.n.01_1" in " ".join(map(str, lit)) for lit in p.init + p.goal)
    compiled = p.compile()
    assert compiled.pose_reached_args == ["painting.n.01_1"]
    assert "wall.n.01_1" in compiled.object_scope  # ... the hinge parent still gets a scope binding


@pytest.mark.parametrize("group", ["targets_group.n.01_1", "blocks_group.n.01_1"])
def test_v2_block_design_group_names(group):
    dice = " ".join(f"toy_dice.n.01_{i}" for i in range(1, 5))
    p = parse_problem_string(
        "(define (problem configuration_matching-0) (:domain ispatialgym)"
        " (:variant configuration_matching-block_design-block_4-table)"
        f" (:objects {dice} - toy_dice.n.01 table.n.02_1 - table.n.02 floor.n.01_1 - floor.n.01 agent.n.01_1 - agent.n.01)"
        f" (:groups {dice} - {group})"
        f" (:init (ontop {group} table.n.02_1) {AGENT_ON_FLOOR})"
        f" (:goal (and (ontop ?{group} ?table.n.02_1) (layout_reached ?{group}) (not (grasped ?agent.n.01_1 ?{group}))"
        f" (pose_settled ?{group}) (pose_settled ?agent.n.01_1)))"
        f" (:measure (pose_reached ?{group})))",
        "configuration_matching",
        0,
    )
    compiled = p.compile()
    assert compiled.layout_reached_args == [group] and compiled.measure_pose_reached_args == [group]
    assert compiled.pose_settled_insts == [f"toy_dice.n.01_{i}" for i in range(1, 5)] + ["agent.n.01_1"]
    assert sorted(_binary_tokens(p)) == sorted(["ontop"] * 4 + ["layout_reached"] + ["not grasped"] * 4)


@pytest.mark.parametrize("group", ["assembly_group.n.01_1", "attachment_group.n.01_1"])
def test_v2_attachment_group_names(group):
    p = parse_problem_string(
        "(define (problem configuration_matching-3) (:domain ispatialgym)"
        " (:variant configuration_matching-attachment-digital_camera-camera_tripod-table)"
        " (:objects digital_camera.n.01_1 - digital_camera.n.01 camera_tripod.n.01_1 - camera_tripod.n.01"
        " table.n.02_1 - table.n.02 floor.n.01_1 - floor.n.01 agent.n.01_1 - agent.n.01)"
        f" (:groups digital_camera.n.01_1 camera_tripod.n.01_1 - {group})"
        f" (:init (ontop {group} table.n.02_1) (not (attached digital_camera.n.01_1 camera_tripod.n.01_1)) {AGENT_ON_FLOOR})"
        " (:goal (and (attached ?digital_camera.n.01_1 ?camera_tripod.n.01_1) (ontop ?camera_tripod.n.01_1 ?table.n.02_1)"
        f" (layout_reached ?{group}) (not (grasped ?agent.n.01_1 ?{group})) (pose_settled ?{group}) (pose_settled ?agent.n.01_1)))"
        f" (:measure (pose_reached ?{group})))",
        "configuration_matching",
        3,
    )
    compiled = p.compile()
    assert compiled.layout_reached_args == [group] and compiled.measure_pose_reached_args == [group]
    assert compiled.pose_settled_insts == ["digital_camera.n.01_1", "camera_tripod.n.01_1", "agent.n.01_1"]
