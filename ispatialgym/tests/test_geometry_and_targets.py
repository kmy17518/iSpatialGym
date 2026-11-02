"""Pose math, member-to-slot assignment and TRO target validation (no simulator)."""

import math
import os

import pytest
import torch as th

from ispatialgym.activity import load_problem
from ispatialgym.eval.geometry import (
    IDENTITY_QUAT,
    assign_by_cost_fn,
    assign_by_slot_position,
    axis_angle_quat,
    compose_pose,
    invert_pose,
    orientation_error,
    quat_angle,
    quat_apply,
    quat_multiply,
    relation_error,
    relative_pose,
)
from ispatialgym.eval.targets import (
    SCHEMA_VERSION,
    GroupTarget,
    ISGTroMeta,
    PoseTarget,
    TROValidationError,
)
from ispatialgym.eval.tro import TROFile, validate_tro_against_problem

Z90 = axis_angle_quat([0, 0, 1], math.pi / 2)
X180 = axis_angle_quat([1, 0, 0], math.pi)
CUBE_SYMMETRIES = [axis_angle_quat([0, 0, 1], k * math.pi / 2) for k in range(1, 4)]  # a yaw-symmetric block
#: the v1 template problems these tests compile (the shipped tree holds the eval release's problems only)
BDDL_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bddl")


def _pose(pos, quat=IDENTITY_QUAT):
    return th.tensor(pos, dtype=th.float32), quat


# ----------------------------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------------------------


def test_quaternion_basics():
    assert quat_angle(IDENTITY_QUAT, Z90) == pytest.approx(math.pi / 2, abs=1e-6)
    assert quat_angle(Z90, -Z90) == pytest.approx(0.0, abs=1e-6)  # double cover
    v = quat_apply(Z90, th.tensor([1.0, 0.0, 0.0]))
    assert th.allclose(v, th.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    # quat_multiply(q1, q0): q0 first then q1 -> two quarter turns = half turn
    assert quat_angle(quat_multiply(Z90, Z90), axis_angle_quat([0, 0, 1], math.pi)) == pytest.approx(0.0, abs=1e-6)


def test_pose_composition_and_relative_pose():
    a = _pose([1.0, 0.0, 0.0], Z90)
    b = _pose([1.0, 1.0, 0.0], IDENTITY_QUAT)
    rel = relative_pose(a, b)  # b in a's frame: a is rotated +90 deg about z, so world +y is a's +x
    assert th.allclose(rel[0], th.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert quat_angle(rel[1], axis_angle_quat([0, 0, 1], -math.pi / 2)) == pytest.approx(0.0, abs=1e-6)
    back = compose_pose(a, rel)
    assert th.allclose(back[0], b[0], atol=1e-6) and quat_angle(back[1], b[1]) == pytest.approx(0.0, abs=1e-6)
    inv = compose_pose(a, invert_pose(a))
    assert th.allclose(inv[0], th.zeros(3), atol=1e-6) and quat_angle(inv[1], IDENTITY_QUAT) == pytest.approx(0.0, abs=1e-6)


def test_orientation_error_with_equivalences():
    goal = IDENTITY_QUAT
    current = Z90
    assert orientation_error(current, goal) == pytest.approx(math.pi / 2, abs=1e-6)
    assert orientation_error(current, goal, CUBE_SYMMETRIES) == pytest.approx(0.0, abs=1e-6)
    # a tilt is not a symmetry
    assert orientation_error(X180, goal, CUBE_SYMMETRIES) == pytest.approx(math.pi, abs=1e-6)


def test_relation_error_is_invariant_to_global_motion_and_symmetries():
    slot_a, slot_b = _pose([0.0, 0.0, 0.0]), _pose([0.1, 0.0, 0.0])
    goal_rel = relative_pose(slot_a, slot_b)
    # displaced and rotated as a whole
    motion = _pose([2.0, -1.0, 0.5], axis_angle_quat([0, 0, 1], 0.7))
    a, b = compose_pose(motion, slot_a), compose_pose(motion, slot_b)
    pos_err, ori_err = relation_error(a, b, goal_rel)
    assert pos_err == pytest.approx(0.0, abs=1e-5) and ori_err == pytest.approx(0.0, abs=1e-5)
    # rotate the anchor by a symmetry: relation only holds modulo the anchor's equivalences
    a_sym = (a[0], quat_multiply(a[1], Z90))
    pos_err, ori_err = relation_error(a_sym, b, goal_rel)
    assert pos_err > 0.05
    pos_err, ori_err = relation_error(a_sym, b, goal_rel, CUBE_SYMMETRIES, CUBE_SYMMETRIES)
    assert pos_err == pytest.approx(0.0, abs=1e-5) and ori_err == pytest.approx(0.0, abs=1e-5)
    # a genuinely wrong relative position stays wrong
    pos_err, _ = relation_error(a, (b[0] + th.tensor([0.0, 0.05, 0.0]), b[1]), goal_rel, CUBE_SYMMETRIES, CUBE_SYMMETRIES)
    assert pos_err == pytest.approx(0.05, abs=1e-5)


def test_assignment_by_slot_position_is_deterministic():
    slots = {"s0": th.tensor([0.0, 0.0, 0.0]), "s1": th.tensor([1.0, 0.0, 0.0]), "s2": th.tensor([2.0, 0.0, 0.0])}
    members = {"m_c": th.tensor([2.1, 0.0, 0.0]), "m_a": th.tensor([0.9, 0.0, 0.0]), "m_b": th.tensor([-0.1, 0.0, 0.0])}
    assert assign_by_slot_position(members, slots) == {"m_a": "s1", "m_b": "s0", "m_c": "s2"}
    # exact tie: all members at the same spot -> lexicographically first permutation
    tie = {m: th.zeros(3) for m in members}
    assert assign_by_slot_position(tie, slots) == {"m_a": "s0", "m_b": "s1", "m_c": "s2"}
    with pytest.raises(ValueError):
        assign_by_slot_position({"m": th.zeros(3)}, slots)


def test_assignment_by_cost_fn_tie_break():
    got = assign_by_cost_fn(["b", "a"], ["s1", "s0"], lambda perm: 0.0)
    assert got == {"a": "s0", "b": "s1"}


# ----------------------------------------------------------------------------------------------------------------
# targets
# ----------------------------------------------------------------------------------------------------------------


def test_pose_target_defaults_and_partial_metrics():
    full = PoseTarget.from_dict(
        "mug.n.04_1",
        {"position": [0, 0, 1], "orientation": [0, 0, 0, 1], "position_tolerance": 0.02, "orientation_tolerance": 0.1},
    )
    assert full.position_metric == "full" and full.orientation_metric == "full" and full.frame == "world"
    ori_only = PoseTarget.from_dict(
        "mug.n.04_1", {"orientation": [0, 0, 0, 1], "orientation_tolerance": 0.1, "position_metric": None}
    )
    assert ori_only.position is None and ori_only.position_metric is None and ori_only.orientation_metric == "full"
    depth = PoseTarget.from_dict(
        "mug.n.04_1",
        {
            "frame": "agent.n.01_1::robot_r1:zed_link:Camera:0",
            "position": [0.0, -0.05, -0.6],
            "position_metric": "depth",
            "position_tolerance": 0.02,
        },
    )
    assert depth.frame_instance == "agent.n.01_1" and depth.frame_link == "robot_r1:zed_link:Camera:0"
    kwargs = depth.pose_reached_kwargs("robot_r1::robot_r1:zed_link:Camera:0")
    assert kwargs["pos_axes"] == "depth" and kwargs["goal_ori"] is None and kwargs["pos_tolerance"] == 0.02


@pytest.mark.parametrize(
    "spec,match",
    [
        ({"position": [0, 0, 1]}, "position_tolerance"),
        ({"orientation": [0, 0, 0, 1], "orientation_tolerance": 0.1, "position_metric": "full"}, "position is required"),
        ({"position": [0, 0, 1], "position_metric": "depth", "position_tolerance": 0.1}, "camera frame"),
        ({"position": [0, 0, 1], "position_tolerance": 0.0}, "> 0"),
        ({"orientation": [0, 0, 0, 2], "orientation_tolerance": 0.1}, "unit quaternion"),
        ({"position_metric": None, "orientation_metric": None}, "at least one"),
        ({"position": [0, 0, 1], "position_tolerance": 0.1, "bogus": 1}, "unknown fields"),
        ({"position": [0, 0], "position_tolerance": 0.1}, "3 entries"),
    ],
)
def test_pose_target_rejects_bad_specs(spec, match):
    with pytest.raises(TROValidationError, match=match):
        PoseTarget.from_dict("mug.n.04_1", spec)


def _group_dict(consistent=True, mode="exchangeable"):
    slots = {
        "slot_0": {"position": [0.0, 0.0, 0.0], "orientation": [0, 0, 0, 1]},
        "slot_1": {"position": [0.1, 0.0, 0.0], "orientation": [0, 0, 0, 1]},
    }
    rel_pos = [0.1, 0.0, 0.0] if consistent else [0.2, 0.0, 0.0]
    d = {
        "members": ["toy_dice.n.01_1", "toy_dice.n.01_2"],
        "assignment": {"mode": mode},
        "orientation_equivalences": [],
        "target_slots": slots,
        "pose_tolerances": {"position": 0.02, "orientation": 0.1},
        "layout_relations": [{"from": "slot_0", "to": "slot_1", "position": rel_pos, "orientation": [0, 0, 0, 1]}],
        "layout_tolerances": {"position": 0.02, "orientation": 0.1},
    }
    if mode == "fixed":
        d["assignment"]["member_to_slot"] = {"toy_dice.n.01_1": "slot_0", "toy_dice.n.01_2": "slot_1"}
    return d


def test_group_target_validation():
    g = GroupTarget.from_dict("blocks_group.n.01_1", _group_dict())
    assert g.assignment_cost == "layout" and g.supports_pose_reached and g.supports_layout_reached
    with pytest.raises(TROValidationError, match="inconsistent with target_slots"):
        GroupTarget.from_dict("blocks_group.n.01_1", _group_dict(consistent=False))
    fixed = GroupTarget.from_dict("attachment_group.n.01_1", _group_dict(mode="fixed"))
    assert fixed.member_to_slot == {"toy_dice.n.01_1": "slot_0", "toy_dice.n.01_2": "slot_1"}
    bad = _group_dict(mode="fixed")
    bad["assignment"]["member_to_slot"]["toy_dice.n.01_2"] = "slot_0"
    with pytest.raises(TROValidationError, match="bijection"):
        GroupTarget.from_dict("g", bad)
    one_slot = _group_dict()
    del one_slot["target_slots"]["slot_1"]
    one_slot["layout_relations"] = []
    with pytest.raises(TROValidationError, match="one slot per member"):
        GroupTarget.from_dict("g", one_slot)


def test_group_target_pinned_members():
    d = _group_dict()
    d["assignment"]["pinned"] = {"toy_dice.n.01_2": "slot_1"}
    assert GroupTarget.from_dict("g", d).pinned == {"toy_dice.n.01_2": "slot_1"}
    assert GroupTarget.from_dict("g", _group_dict()).pinned == {}
    fixed = _group_dict(mode="fixed")
    fixed["assignment"]["pinned"] = {"toy_dice.n.01_2": "slot_1"}
    with pytest.raises(TROValidationError, match="for exchangeable groups"):
        GroupTarget.from_dict("g", fixed)
    for pinned, match in (({"mug.n.04_1": "slot_1"}, "non-members"), ({"toy_dice.n.01_2": "slot_9"}, "distinct target_slots"),
                          ({"toy_dice.n.01_1": "slot_1", "toy_dice.n.01_2": "slot_1"}, "distinct target_slots")):
        bad = _group_dict()
        bad["assignment"]["pinned"] = pinned
        with pytest.raises(TROValidationError, match=match):
            GroupTarget.from_dict("g", bad)


def _meta_dict(activity="world_alignment", definition_id=0, variant="table_surface", **overrides):
    d = {
        "schema_version": SCHEMA_VERSION,
        "activity": activity,
        "activity_definition_id": definition_id,
        "task_variant": variant,
        "instance_id": f"{activity}-000001",
        "scene": {"model": "Rs_int", "floor": 0},
        "object_bindings": {
            "fishing_rod.n.01_1": {"scene_object_name": "fishing_rod_0", "category": "fishing_rod", "model": "zhmrtc"},
            "table.n.02_1": {"scene_object_name": "table_0"},
            "floor.n.01_1": {"scene_object_name": "floors"},
        },
        "pose_targets": {
            "fishing_rod.n.01_1": {
                "orientation": [0, 0, 0, 1],
                "orientation_tolerance": 0.05,
                "position_metric": None,
            }
        },
    }
    d.update(overrides)
    return d


def test_tro_meta_and_problem_validation():
    compiled = load_problem("world_alignment", 0, root=BDDL_FIXTURES).compile()
    meta = ISGTroMeta.from_dict(_meta_dict())
    validate_tro_against_problem(meta, compiled)
    assert meta.measured_instances() == ["fishing_rod.n.01_1"]

    with pytest.raises(TROValidationError, match="schema_version"):
        ISGTroMeta.from_dict(_meta_dict(schema_version="isg-tro-v0"))
    with pytest.raises(TROValidationError, match="task_variant"):
        validate_tro_against_problem(ISGTroMeta.from_dict(_meta_dict(variant="world_vertical")), compiled)
    with pytest.raises(TROValidationError, match="no pose_targets"):
        validate_tro_against_problem(ISGTroMeta.from_dict(_meta_dict(pose_targets={})), compiled)
    with pytest.raises(TROValidationError, match="missing physical instances"):
        validate_tro_against_problem(ISGTroMeta.from_dict(_meta_dict(object_bindings={})), compiled)
    validate_tro_against_problem(ISGTroMeta.from_dict(_meta_dict(object_bindings={})), compiled, require_bindings=False)


def test_tro_file_splits_behavior_state_from_metadata():
    data = {"fishing_rod.n.01_1": {"root_link": {"pos": [0, 0, 0]}}, "robot_poses": {"robot": [{"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}]}, "ispatialgym": _meta_dict()}
    tro = TROFile.from_dict(data)
    assert set(tro.object_states) == {"fishing_rod.n.01_1"}
    assert tro.robot_poses["robot"][0]["position"] == [0, 0, 0]
    assert tro.meta.name == "world_alignment-0"
    with pytest.raises(TROValidationError, match="reserved"):
        TROFile.from_dict({"fishing_rod.n.01_1": {}})


def test_layout_targets_are_required_by_layout_reached_goals():
    compiled = load_problem("configuration_matching", 1, root=BDDL_FIXTURES).compile()
    base = _meta_dict(
        activity="configuration_matching",
        definition_id=1,
        variant="articulation",
        object_bindings={"cabinet.n.01_1": {"scene_object_name": "cabinet_0"}, "floor.n.01_1": {"scene_object_name": "floors"}},
        pose_targets={},
    )
    with pytest.raises(TROValidationError, match="articulation_targets"):
        validate_tro_against_problem(ISGTroMeta.from_dict(base), compiled)
    base["articulation_targets"] = {"cabinet.n.01_1": {"joints": {"j_link_1": {"position": 1.2, "tolerance": 0.05}}}}
    validate_tro_against_problem(ISGTroMeta.from_dict(base), compiled)
    # a root pose target for the (fixed-base) cabinet is neither gated nor measured by the problem -> rejected
    base["pose_targets"] = {"cabinet.n.01_1": {"orientation": [0, 0, 0, 1], "orientation_tolerance": 0.1}}
    with pytest.raises(TROValidationError, match=r"no \(pose_reached cabinet.n.01_1\) atom"):
        validate_tro_against_problem(ISGTroMeta.from_dict(base), compiled)

    compiled = load_problem("configuration_matching", 0, root=BDDL_FIXTURES).compile()
    block = _meta_dict(
        activity="configuration_matching",
        definition_id=0,
        variant="block_design",
        object_bindings={},
        pose_targets={},
        groups={"blocks_group.n.01_1": _group_dict()},
    )
    with pytest.raises(TROValidationError, match="members"):
        validate_tro_against_problem(ISGTroMeta.from_dict(block), compiled, require_bindings=False)


def _dice_group(with_pose_tolerances=True):
    members = [f"toy_dice.n.01_{i}" for i in range(1, 5)]
    slots = {f"slot_{i}": {"position": [0.1 * i, 0.0, 0.0], "orientation": [0, 0, 0, 1]} for i in range(4)}
    relations = [
        {"from": f"slot_{i}", "to": f"slot_{j}", "position": [0.1 * (j - i), 0.0, 0.0], "orientation": [0, 0, 0, 1]}
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    d = {
        "members": members,
        "target_slots": slots,
        "layout_relations": relations,
        "layout_tolerances": {"position": 0.02, "orientation": 0.1},
    }
    if with_pose_tolerances:
        d["pose_tolerances"] = {"position": 0.02, "orientation": 0.1}
    return d


def test_measure_atoms_require_and_license_targets():
    """(:measure ...) atoms need a TRO target just like goal atoms, and TRO targets need a goal or :measure atom."""
    # Disembedding: (pose_reached ?mug.n.04_1) lives in :measure -> a pose target is required ...
    compiled = load_problem("disembedding", 0, root=BDDL_FIXTURES).compile()
    meta = _meta_dict(activity="disembedding", definition_id=0, variant=None, object_bindings={}, pose_targets={})
    with pytest.raises(TROValidationError, match=r"\(pose_reached mug.n.04_1\) in :measure has no pose_targets"):
        validate_tro_against_problem(ISGTroMeta.from_dict(meta), compiled, require_bindings=False)
    meta["pose_targets"] = {"mug.n.04_1": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1], "position_tolerance": 0.02, "orientation_tolerance": 0.1}}
    validate_tro_against_problem(ISGTroMeta.from_dict(meta), compiled, require_bindings=False)
    # ... and a target for an object that no atom mentions is rejected
    meta["pose_targets"]["mug.n.04_2"] = dict(meta["pose_targets"]["mug.n.04_1"])
    with pytest.raises(TROValidationError, match=r"pose_targets\['mug.n.04_2'\] has no \(pose_reached mug.n.04_2\) atom"):
        validate_tro_against_problem(ISGTroMeta.from_dict(meta), compiled, require_bindings=False)

    # Block design: layout_reached gates, pose_reached is measured -> the group needs relations AND slots/tolerances
    compiled = load_problem("configuration_matching", 0, root=BDDL_FIXTURES).compile()
    block = _meta_dict(activity="configuration_matching", definition_id=0, variant="block_design", object_bindings={}, pose_targets={})
    block["groups"] = {"blocks_group.n.01_1": _dice_group(with_pose_tolerances=False)}
    with pytest.raises(TROValidationError, match=r"\(pose_reached blocks_group.n.01_1\) in :measure needs groups"):
        validate_tro_against_problem(ISGTroMeta.from_dict(block), compiled, require_bindings=False)
    block["groups"] = {"blocks_group.n.01_1": _dice_group()}
    validate_tro_against_problem(ISGTroMeta.from_dict(block), compiled, require_bindings=False)
    # a group target for a group that no atom mentions is rejected
    compiled_wa = load_problem("world_alignment", 0, root=BDDL_FIXTURES).compile()
    stray = _meta_dict(groups={"blocks_group.n.01_1": _dice_group()})
    with pytest.raises(TROValidationError, match="not a declared BDDL group"):
        validate_tro_against_problem(ISGTroMeta.from_dict(stray), compiled_wa)
