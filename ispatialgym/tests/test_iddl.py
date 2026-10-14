"""The iddl parser (``sampling/iddl.py``) on the 23 shipped files and on malformed input (no simulator)."""

import os

import pytest

from ispatialgym.sampling import iddl as I

IDDLS = I.load_all()

#: subtask -> (explicit roles, :require forms, variant slots, composite groups)
EXPECTED = {
    "camera_relocalization-standard": ([], [], [], {}),
    "camera_relocalization-obstructed": (["obstacles"], [], ["obstacles"], {}),
    "object_scaling": (["target"], ["grasps:floating"], ["target"], {}),
    "mental_rotation": (["target"], ["grasps:floating"], ["target"], {}),
    "disembedding": (["target", "distractors", "support", "source", "destination"], ["grasps:tabletop"], ["target", "distractors", "support"], {}),
    "configuration_matching-block_design": (["targets", "support"], [], ["targets", "support"], {}),
    "configuration_matching-articulation_open_large_scale": (["target"], ["handle"], ["target"], {}),
    "configuration_matching-articulation_close_large_scale": (["target"], ["handle"], ["target"], {}),
    "configuration_matching-articulation_open_small_scale": (["target", "support"], ["handle"], ["target", "support"], {}),
    "configuration_matching-articulation_close_small_scale": (["target", "support"], ["handle"], ["target", "support"], {}),
    "configuration_matching-attachment": (["target", "partner", "support"], ["grasps:attachment"], ["target", "partner", "support"], {"assembly": ["target", "partner"]}),
    "alignment-axial": (["target", "support"], ["contacts"], ["target", "support"], {}),
    "alignment-world_vertical": (["target", "support"], ["contacts"], ["target", "support"], {}),
    "inter_object_relation_satisfaction-ontop_small_scale": (["target", "partner", "support"], ["grasps:tabletop"], ["target", "partner", "support"], {}),
    "inter_object_relation_satisfaction-under_small_scale": (["target", "partner", "support"], ["grasps:tabletop"], ["target", "partner", "support"], {}),
    "inter_object_relation_satisfaction-inside": (["target", "partner", "support"], ["grasps:insert"], ["target", "partner", "support"], {}),
    "inter_object_relation_satisfaction-nextto": (["target", "partner", "support"], ["grasps:tabletop"], ["target", "partner", "support"], {}),
    "inter_object_relation_satisfaction-touching": (["target", "partner", "support"], ["grasps:tabletop"], ["target", "partner", "support"], {}),
    "inter_object_relation_satisfaction-ontop_large_scale": (["target", "partner", "support"], ["grasps:tabletop"], ["target", "partner", "support"], {}),
    "inter_object_relation_satisfaction-under_large_scale": (["target", "partner", "support"], ["grasps:tabletop"], ["target", "partner", "support"], {}),
    "inter_object_relation_satisfaction-inside_large_scale": (["target", "partner", "support"], ["grasps:tabletop"], ["target", "partner", "support"], {}),
    "inter_object_relation_satisfaction-touching_large_scale": (["target", "partner", "support"], ["grasps:tabletop"], ["target", "partner", "support"], {}),
    "inter_object_relation_satisfaction-nextto_large_scale": (["target", "partner", "support"], [], ["target", "partner", "support"], {}),  # base push: no grasp
}


def test_all_23_files_load_keyed_by_subtask_id():
    assert len(IDDLS) == 23
    assert set(IDDLS) == set(EXPECTED)
    for subtask, iddl in IDDLS.items():
        assert iddl.subtask == subtask
        assert iddl.family in I.FAMILIES
        assert iddl.path and os.path.basename(os.path.dirname(iddl.path)) == iddl.family
        assert I.relative_path(iddl.path).startswith(iddl.family + "/")
        assert iddl.comment  # every file starts with a description


@pytest.mark.parametrize("subtask", sorted(EXPECTED))
def test_roles_requires_slots(subtask):
    roles, requires, slots, groups = EXPECTED[subtask]
    iddl = IDDLS[subtask]
    assert iddl.roles == roles
    assert [r.form for r in iddl.require] == requires
    assert iddl.variant_slots == slots
    assert iddl.groups == groups
    assert iddl.variant_template.startswith(subtask)
    assert "agent" in iddl.literal_roles() and "floor" in iddl.literal_roles()
    for r in iddl.require:
        assert r.role in roles
        assert r.missing_reason.startswith("missing_")
    # every variant slot is an explicit role and every literal role is explicit, implicit or a composite
    for s in slots:
        assert s in roles
    for r in iddl.literal_roles():
        assert r in roles or r in I.IMPLICIT_ROLES or r in groups


def test_predicate_table_matches_the_bddl_domain():
    from bddl.parsing import parse_domain

    _, _, _, _, predicates = parse_domain("ispatialgym")
    assert {p: len(args) for p, args in predicates.items()} == I.PREDICATES


def test_require_forms():
    r = I.Require("grasps", "target", "tabletop")
    assert r.form == "grasps:tabletop" and r.missing_reason == "missing_grasps:tabletop"
    assert r.customs_value() == {"grasps": "tabletop"}
    assert I.Require("contacts", "target").customs_value() == {"contacts": True}
    assert I.Require("handle", "target").missing_reason == "missing_handle"


def test_under_argument_order_per_subtask():
    small = IDDLS["inter_object_relation_satisfaction-under_small_scale"]
    large = IDDLS["inter_object_relation_satisfaction-under_large_scale"]
    assert ["under", "?partner", "?target"] in small.goal
    assert ["not", ["under", "partner", "target"]] in small.init
    assert ["under", "?target", "?partner"] in large.goal
    assert ["not", ["under", "target", "partner"]] in large.init


def test_pose_evaluated_roles_and_measure():
    assert IDDLS["mental_rotation"].pose_evaluated_roles() == ["target"]
    assert IDDLS["camera_relocalization-standard"].pose_evaluated_roles() == ["agent"]
    dis = IDDLS["disembedding"]
    assert dis.measure == [["pose_reached", "?target"]] and dis.pose_evaluated_roles() == ["target"]
    assert "pose_reached" in dis.predicates_on("target") and "ontop" in dis.predicates_on("distractors")
    att = IDDLS["configuration_matching-attachment"]
    assert att.pose_evaluated_roles() == ["assembly"] and att.is_composite("assembly")
    assert IDDLS["inter_object_relation_satisfaction-nextto"].pose_evaluated_roles() == ["target", "partner"]


def test_fill_variant():
    iddl = IDDLS["inter_object_relation_satisfaction-nextto"]
    assert iddl.fill_variant({"target": "spatula", "partner": "soup_ladle", "support": "breakfast_table"}) == \
        "inter_object_relation_satisfaction-nextto-spatula-soup_ladle-breakfast_table"
    with pytest.raises(I.IDDLError, match="slots without a value"):
        iddl.fill_variant({"target": "spatula"})
    assert IDDLS["camera_relocalization-standard"].fill_variant({}) == "camera_relocalization-standard"


@pytest.mark.parametrize("subtask", sorted(EXPECTED))
def test_round_trip(subtask):
    iddl = IDDLS[subtask]
    again = I.parse(iddl.to_text())
    assert again.subtask == iddl.subtask and again.family == iddl.family and again.variant_template == iddl.variant_template
    assert again.init == iddl.init and again.goal == iddl.goal and again.measure == iddl.measure
    assert again.require == iddl.require and again.groups == iddl.groups


GOOD = """
; a comment
(define (iddl alignment-axial)
    (:family alignment)
    (:variant alignment-axial-{target}-{support})
    (:require (contacts target))
    (:init (ontop target support) (ontop agent floor))
    (:goal (and (ontop ?target ?support) (pose_reached ?target) (not (grasped ?agent ?target)) (pose_settled ?target) (pose_settled ?agent)))
)
"""


def test_parse_minimal_file():
    iddl = I.parse(GOOD)
    assert iddl.roles == ["target", "support"] and iddl.comment == "a comment"
    assert iddl.require == [I.Require("contacts", "target")]


@pytest.mark.parametrize(
    "text,match",
    [
        (GOOD.replace("(:family alignment)", ""), "missing section :family"),
        (GOOD.replace("(:family alignment)", "(:family alignments)"), "unknown family"),
        (GOOD.replace("(:variant alignment-axial-{target}-{support})", "(:variant axial-{target})"), "must start with the subtask id"),
        (GOOD.replace("(:variant alignment-axial-{target}-{support})", "(:variant alignment-axial-{target}-{target})"), "repeats a slot"),
        (GOOD.replace("(:variant alignment-axial-{target}-{support})", "(:variant alignment-axial-{floor})"), "explicit non-composite role"),
        (GOOD.replace("(contacts target)", "(contacts target tabletop)"), "takes exactly one role"),
        (GOOD.replace("(contacts target)", "(grasps target sideways)"), "unknown grasps pipeline"),
        (GOOD.replace("(contacts target)", "(pushes target)"), "unknown :require form"),
        (GOOD.replace("(contacts target)", "(contacts partner)"), "not a role of the file"),
        (GOOD.replace("(ontop target support) (ontop agent floor)", "(ontop ?target ?support) (ontop agent floor)"), "no \\? prefix"),
        (GOOD.replace("(ontop ?target ?support)", "(ontop target ?support)"), "carry the \\? prefix"),
        (GOOD.replace("(ontop target support)", "(pose_reached target)"), ":init must not contain"),
        (GOOD.replace("(pose_settled ?target)", "(or (pose_settled ?target) (ontop ?target ?support))"), "direct child"),
        (GOOD.replace("(ontop ?target ?support)", "(cooked ?target)"), "unknown predicate"),
        (GOOD.replace("(ontop ?target ?support)", "(ontop ?target)"), "takes 2 argument"),
        (GOOD.replace("(not (grasped ?agent ?target))", "(not (grasped ?target ?agent))"), "first argument of 'grasped'"),
        (GOOD.replace("(:goal (and", "(:goal (or"), "exactly one \\(and"),
        (GOOD.replace("(:init", "(:init (ontop target support)) (:init"), "duplicate section"),
        (GOOD.replace("(:require (contacts target))", "(:options (push target))"), "unknown section"),
        (GOOD.replace("(pose_reached ?target)", "(forall (?x - cat) (ontop ?x ?support))"), "quantifiers"),
        (GOOD.rstrip()[:-1], "unexpected end of file"),
        (GOOD + ")", "unexpected '\\)'|trailing"),
    ],
)
def test_malformed_files_are_rejected(text, match):
    with pytest.raises(I.IDDLError, match=match):
        I.parse(text)


WITH_MEASURE = """
(define (iddl alignment-axial)
    (:family alignment)
    (:variant alignment-axial-{target}-{support})
    (:init (ontop target support) (ontop agent floor))
    (:goal (and (ontop ?target ?support) (layout_reached ?target) (pose_settled ?agent)))
    (:measure (pose_reached ?target))
)
"""


def test_measure_rules():
    iddl = I.parse(WITH_MEASURE)
    assert iddl.measure == [["pose_reached", "?target"]] and iddl.pose_evaluated_roles() == ["target"]
    with pytest.raises(I.IDDLError, match="only accepts"):
        I.parse(WITH_MEASURE.replace("(:measure (pose_reached ?target))", "(:measure (ontop ?target ?support))"))
    with pytest.raises(I.IDDLError, match="both a gating goal atom"):
        I.parse(WITH_MEASURE.replace("(layout_reached ?target)", "(pose_reached ?target)"))
    with pytest.raises(I.IDDLError, match="twice"):
        I.parse(WITH_MEASURE.replace("(:measure (pose_reached ?target))", "(:measure (pose_reached ?target) (pose_reached ?target))"))


def test_load_all_checks_the_layout(tmp_path):
    fam = tmp_path / "alignment"
    fam.mkdir()
    (fam / "axial.iddl").write_text(GOOD)
    assert list(I.load_all(str(tmp_path))) == ["alignment-axial"]
    (fam / "other.iddl").write_text(GOOD)  # declares alignment-axial under the wrong file name
    with pytest.raises(I.IDDLError, match="does not match the file layout"):
        I.load_all(str(tmp_path))
