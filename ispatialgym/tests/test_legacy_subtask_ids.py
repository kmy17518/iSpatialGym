"""Data written before the small-scale IORS tasks were renamed (2026-10-14: inside / nextto / touching ->
inside_small_scale / nextto_small_scale / touching_small_scale) reads under the current names: the eval85 runs keep their
BDDL copies, sidecars, template metadata and TROs (``activity.problem.LEGACY_SUBTASK_IDS``)."""

import json
import os
import tempfile

import pytest

from ispatialgym.activity.problem import (
    LEGACY_SUBTASK_IDS,
    canonical_variant,
    id_of_variant,
    load_problem_registry,
    parse_problem_string,
)
from ispatialgym.eval import goal_occlusion as GO
from ispatialgym.eval import names as N
from ispatialgym.eval.targets import SCHEMA_VERSION, ISGTroMeta
from ispatialgym.oracle.sidecar import Sidecar, from_legacy_keys, family_of_subtask, subtask_of_variant, variant_templates

IORS = "inter_object_relation_satisfaction"
SIDECARS = os.path.join(os.path.dirname(__file__), "data", "sidecars")
OLD_VARIANT = "inter_object_relation_satisfaction-touching-box_of_cereal_s080-box_of_cereal-bar"  # legacy spelling
NEW_VARIANT = f"{IORS}-touching_small_scale-box_of_cereal_s080-box_of_cereal-bar"


def test_canonical_variant():
    assert canonical_variant(OLD_VARIANT) == NEW_VARIANT and canonical_variant(NEW_VARIANT) == NEW_VARIANT
    for old, new in LEGACY_SUBTASK_IDS.items():
        assert canonical_variant(old) == new and canonical_variant(old + "-{target}-{partner}-{support}") == new + "-{target}-{partner}-{support}"
        assert new in variant_templates() and old not in variant_templates()
    for kept in (f"{IORS}-inside_large_scale-plate-drop_in_sink-countertop", f"{IORS}-ontop_small_scale-a-b-c", f"{IORS}-inside#02",
                 "inter_object_relation_satisfaction-touching_02/c1_k0", "alignment-axial-board_game-breakfast_table", None, 3):  # legacy spelling
        assert canonical_variant(kept) == kept


def test_names_of_an_old_variant():
    assert subtask_of_variant(OLD_VARIANT) == f"{IORS}-touching_small_scale"
    assert family_of_subtask("inter_object_relation_satisfaction-nextto") == IORS  # legacy spelling
    assert N.task_name(OLD_VARIANT) == N.task_name(NEW_VARIANT) == f"{IORS}_touching_small_scale"
    assert N.subtask_name(OLD_VARIANT) == N.subtask_name(NEW_VARIANT) == f"{IORS}_touching_small_scale_box_of_cereal_s080_box_of_cereal"


def _iors_problem(variant: str, did: int = 0) -> str:
    return f"""(define (problem {IORS}-{did})
    (:domain ispatialgym)
    (:variant {variant})
    (:objects
        apple.n.01_1 - apple.n.01
        bowl.n.01_1 - bowl.n.01
        breakfast_table.n.01_1 - breakfast_table.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    (:init (ontop apple.n.01_1 breakfast_table.n.01_1) (ontop bowl.n.01_1 breakfast_table.n.01_1) (ontop agent.n.01_1 floor.n.01_1))
    (:goal (and (nextto ?apple.n.01_1 ?bowl.n.01_1) (pose_settled ?apple.n.01_1)))
    (:measure (pose_reached ?apple.n.01_1))
)"""


def test_problem_copy_and_registry_written_before_the_rename():
    old = "inter_object_relation_satisfaction-nextto-apple-bowl-breakfast_table"  # legacy spelling
    new = f"{IORS}-nextto_small_scale-apple-bowl-breakfast_table"
    assert parse_problem_string(_iors_problem(old), IORS, 0).variant == new
    assert parse_problem_string(_iors_problem(old).replace("(:variant ", "(:alias "), IORS, 0).variant == new  # legacy spelling
    with tempfile.TemporaryDirectory() as root:
        directory = os.path.join(root, IORS)
        os.makedirs(directory)
        with open(os.path.join(directory, "problem0.bddl"), "w") as f:
            f.write(_iors_problem(old))
        with open(os.path.join(directory, "problems.json"), "w") as f:
            json.dump({"activity": IORS, "problems": [{"variant": old, "id": 0}]}, f)
        assert load_problem_registry(directory) == {new: 0}
        assert id_of_variant(IORS, old, root) == id_of_variant(IORS, new, root) == 0


@pytest.mark.parametrize("task", ["inside", "nextto", "touching"])
def test_sidecar_written_before_the_rename_loads(task):
    path = os.path.join(SIDECARS, f"{IORS}-{task}_small_scale.json")
    with open(path) as f:
        d = json.load(f)
    old = {k: (v.replace(f"{task}_small_scale", task, 1) if k in ("variant", "instance_id") else v) for k, v in d.items()}
    assert old["variant"].startswith(f"{IORS}-{task}-")
    assert Sidecar.from_dict(old) == Sidecar.load(path)
    assert from_legacy_keys({"alias": old["variant"], "subtask": f"{IORS}-{task}"}) == {"variant": d["variant"], "subtask": f"{IORS}-{task}_small_scale"}  # legacy spelling


def test_tro_written_before_the_rename():
    meta = ISGTroMeta.from_dict({"schema_version": SCHEMA_VERSION, "activity": IORS, "activity_definition_id": 0, "task_variant": OLD_VARIANT,
                                 "instance_id": OLD_VARIANT + "__house_single_floor__target-box_of_cereal-iucshm"})
    assert meta.task_variant == NEW_VARIANT and meta.instance_id.startswith(NEW_VARIANT + "__")


def test_contained_target_keeps_both_scales():
    for variant in ("inter_object_relation_satisfaction-inside-hard_candy-hingeless_jar-bar",  # legacy spelling
                    f"{IORS}-inside_small_scale-hard_candy-hingeless_jar-bar", f"{IORS}-inside_large_scale-plate-drop_in_sink-countertop",
                    f"{IORS}-inside_small_scale", f"{IORS}-inside_large_scale"):
        assert GO.contained_target(variant), variant
    for variant in (NEW_VARIANT, f"{IORS}-nextto_small_scale-a-b-c", f"{IORS}-ontop_large_scale-a-b-c"):
        assert not GO.contained_target(variant), variant
