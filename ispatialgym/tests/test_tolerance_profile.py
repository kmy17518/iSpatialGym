"""The ``isg-v2`` evaluation tolerances (``ispatialgym.eval.tolerance_profile``): scale groups, the 10 % rule, the relation floors,
the scale factor, ``evaluation.tolerances`` overrides and object scaling's full camera-frame position."""

import json
from types import SimpleNamespace

import pytest

from ispatialgym.eval import tolerance_profile as TP

IORS = "inter_object_relation_satisfaction"


@pytest.mark.parametrize("variant,kind,group", [
    ("object_scaling-apple", "pose", "micro"),
    ("mental_rotation-mug", "pose", "micro"),
    ("alignment-axial-board_game-breakfast_table", "pose", "micro"),
    ("alignment-world_vertical-painting-walls", "pose", "micro"),
    ("configuration_matching-articulation_open_small_scale-blender-countertop", "joints", "micro"),
    ("configuration_matching-articulation_close_large_scale-door-floor", "joints", "micro"),
    ("configuration_matching-articulation_open_small_scale-blender-countertop", "pose", "small"),
    ("configuration_matching-block_design-block_4-breakfast_table", "group_layout", "micro"),
    ("configuration_matching-attachment-cap-bottle-countertop", "group_layout", "micro"),
    ("configuration_matching-block_design-block_4-breakfast_table", "group_pose", "small"),
    ("disembedding-beaker-beaker_1_soda_cup_1-desk", "pose", "small"),
    (f"{IORS}-touching_small_scale-a-b-bar", "pose", "small"),
    (f"{IORS}-ontop_small_scale-a-b-bar", "pose", "small"),
    (f"{IORS}-nextto_large_scale-serving_cart-coffee_table-floor", "pose", "large"),
    ("camera_relocalization-obstructed-hamper_1", "pose", "large"),
])
def test_scale_groups(variant, kind, group):
    assert TP.scale_group(variant, kind) == group


@pytest.mark.parametrize("variant,kind", [("camera_relocalization-standard", "joints"), ("configuration_matching-attachment-a-b-c", "pose"),
                                          ("disembedding-a-b-c", "group_pose"), ("nope", "pose"), ("alignment-axial-a-b", "fingers")])
def test_unknown_target_raises(variant, kind):
    with pytest.raises(TP.ToleranceProfileError):
        TP.scale_group(variant, kind)


def plan1(variant, target, scale=1.0, overrides=None):
    return TP.plan(variant, [{"name": "x", **target}], scale=scale, overrides=overrides)["x"]


def test_micro_pose_floors_and_ten_percent_of_the_change():
    v = "mental_rotation-mug"
    rec = plan1(v, {"kind": "pose", "components": ["position", "orientation"], "change": {"position": 0.05, "orientation": 1.0}})
    assert rec["tolerances"] == {"position": 0.01, "orientation": pytest.approx(0.1)} and rec["scale_group"] == "micro"
    rec = plan1(v, {"kind": "pose", "components": ["position", "orientation"], "change": {"position": 0.3, "orientation": 0.2}})
    assert rec["tolerances"] == {"position": pytest.approx(0.03), "orientation": 0.05}
    rec = plan1("alignment-axial-a-b", {"kind": "pose", "components": ["orientation"], "change": {"position": None, "orientation": 0.9}})
    assert rec["tolerances"] == {"position": None, "orientation": pytest.approx(0.09)}  # an orientation-only target


def test_joints_per_joint_and_relation_floors():
    v = "configuration_matching-articulation_open_small_scale-blender-countertop"
    rec = plan1(v, {"kind": "joints", "joints": {"lid": {"type": "revolute", "change": -1.5}, "slide": {"type": "prismatic", "change": 0.2},
                                                  "still": {"type": "revolute", "change": 0.0}}})
    assert rec["tolerances"] == {"lid": pytest.approx(0.15), "slide": pytest.approx(0.02), "still": 0.05}
    assert rec["change"] == {"lid": 1.5, "slide": 0.2, "still": 0.0}
    with pytest.raises(TP.ToleranceProfileError):
        plan1(v, {"kind": "joints", "joints": {"j": {"type": "spherical", "change": 0.1}}})
    rel = plan1("configuration_matching-block_design-b-c", {"kind": "group_layout", "components": ["position", "orientation"],
                                                            "change": {"position": 0.5, "orientation": 2.0}})
    assert rel["tolerances"] == {"position": 0.01, "orientation": 0.05}  # floors only, whatever the change


def test_small_and_large_fixed_values():
    small = plan1(f"{IORS}-inside_small_scale-a-b-c", {"kind": "pose", "components": ["position"], "change": {"position": 0.4}})
    assert small["tolerances"] == {"position": 0.03, "orientation": None}
    large = plan1("camera_relocalization-standard", {"kind": "pose", "components": ["position", "orientation"], "change": {"position": 2.0, "orientation": 1.0}})
    assert large["tolerances"] == {"position": 0.05, "orientation": 0.15}


def test_scale_multiplies_everything_overrides_included():
    v = "mental_rotation-mug"
    t = {"kind": "pose", "components": ["position", "orientation"], "change": {"position": 0.0, "orientation": 0.0}}
    assert plan1(v, t, scale=3)["tolerances"] == {"position": pytest.approx(0.03), "orientation": pytest.approx(0.15)}
    rec = plan1(v, t, scale=3, overrides={"pose": {"position": 0.02}})
    assert rec["tolerances"] == {"position": pytest.approx(0.06), "orientation": pytest.approx(0.15)}
    assert rec["override"] == {"position": True, "orientation": False}
    j = plan1("configuration_matching-articulation_open_large_scale-door-floor", {"kind": "joints", "joints": {"d": {"type": "revolute", "change": 1.0}}},
              scale=2, overrides={"joints": {"revolute": 0.2, "fraction": 0.5}})
    assert j["tolerances"] == {"d": pytest.approx(1.0)} and j["override"] == {"d": True}
    rel = plan1("configuration_matching-attachment-a-b-c", {"kind": "group_layout", "components": ["position", "orientation"], "change": {}},
                overrides={"layout": {"orientation": 0.1}})
    assert rel["tolerances"] == {"position": 0.01, "orientation": 0.1}
    assert TP.default_scale("policy") == 3.0 and TP.default_scale("oracle") == TP.default_scale("replay") == 1.0
    with pytest.raises(TP.ToleranceProfileError):
        plan1(v, t, scale=0)


@pytest.mark.parametrize("bad", [{"poses": {}}, {"pose": {"pos": 0.1}}, {"joints": {"revolute": -1}}, {"pose": 3}, {"layout": {"position": True}}])
def test_malformed_overrides_raise(bad):
    with pytest.raises(TP.ToleranceProfileError):
        TP.validate_overrides(bad)


def test_overrides_are_read_from_the_subtask_entry(tmp_path):
    path = tmp_path / "isg_customs.json"
    path.write_text(json.dumps({"_comment": "x", "mental_rotation_mug": {"iddl": "mental_rotation/mental_rotation.iddl", "evaluation": {"tolerances": {"pose": {"orientation": 0.08}}}},
                                "mental_rotation_crab": {"iddl": "mental_rotation/mental_rotation.iddl"}}))
    assert TP.overrides_for("mental_rotation-mug", str(path)) == {"pose": {"orientation": 0.08}}
    assert TP.overrides_for("mental_rotation-crab", str(path)) == {}
    assert TP.overrides_for("mental_rotation-mug", str(tmp_path / "missing.json")) == {}


class FakeTask:
    """``configure_targets`` / ``collect_errors`` of a loaded task: the errors are those of the targets as configured."""

    def __init__(self, meta, errors):
        self._tro, self.errors, self.configured = SimpleNamespace(meta=meta), errors, []

    def configure_targets(self, meta):
        self.configured.append({k: (t.position_metric, t.position_tolerance) for k, t in meta.pose_targets.items()})

    def collect_errors(self):
        return self.errors


def pose_target(position_metric="full", orientation_metric="full", pt=0.5, ot=0.5):
    return SimpleNamespace(position_metric=position_metric, orientation_metric=orientation_metric, position_tolerance=pt, orientation_tolerance=ot)


def test_apply_to_task_object_scaling_full_metric_and_meta_record(tmp_path):
    customs = tmp_path / "c.json"
    customs.write_text("{}")
    meta = SimpleNamespace(task_variant="object_scaling-apple", pose_targets={"apple.n.01_1": pose_target("depth", None, 0.2, None)}, groups={},
                           articulation_targets={})
    task = FakeTask(meta, {"pose_errors": {"apple.n.01_1": {"position_error": 0.12, "orientation_error": None}}, "layout_errors": {}})
    rec = TP.apply_to_task(task, "replay", customs_path=str(customs))
    t = meta.pose_targets["apple.n.01_1"]
    assert t.position_metric == "full" and t.position_tolerance == pytest.approx(0.012) and t.orientation_tolerance is None
    assert task.configured == [{"apple.n.01_1": ("full", 0.2)}, {"apple.n.01_1": ("full", pytest.approx(0.012))}]  # before measuring, then the profile's
    assert rec["name"] == "isg-v2" and rec["mode"] == "replay" and rec["scale"] == 1.0
    assert rec["targets"]["apple.n.01_1"] == {"kind": "pose", "instance": "apple.n.01_1", "scale_group": "micro", "change": {"position": 0.12, "orientation": None},
                                              "tolerances": {"position": pytest.approx(0.012), "orientation": None},
                                              "override": {"position": False, "orientation": False}, "note": {"position_metric": "full", "was": "depth"}}


def test_apply_to_task_articulation_and_policy_scale(tmp_path):
    from omnigibson.utils.constants import JointType

    customs = tmp_path / "c.json"
    customs.write_text("{}")
    art = SimpleNamespace(joints={"lid": (1.2, 0.3)})
    meta = SimpleNamespace(task_variant="configuration_matching-articulation_open_large_scale-door", pose_targets={}, groups={},
                           articulation_targets={"door.n.01_1": art})
    errs = {"pose_errors": {}, "layout_errors": {"door.n.01_1": {"joints": {"lid": {"error": 1.2, "joint_type": JointType.JOINT_REVOLUTE}}}}}
    rec = TP.apply_to_task(FakeTask(meta, errs), "policy", customs_path=str(customs))
    assert art.joints == {"lid": (1.2, pytest.approx(0.36))}  # max(0.05, 0.12) x 3
    assert rec["scale"] == 3.0 and rec["targets"]["door.n.01_1:joints"]["change"] == {"lid": 1.2}


def test_apply_to_task_articulation_with_a_root_pose_target_on_the_same_instance(tmp_path):
    """Articulation small-scale: the object has a root pose target and joint targets; both get their own tolerance."""
    from omnigibson.utils.constants import JointType

    customs = tmp_path / "c.json"
    customs.write_text("{}")
    art = SimpleNamespace(joints={"lid": (1.2, 0.3)})
    meta = SimpleNamespace(task_variant="configuration_matching-articulation_open_small_scale-blender-breakfast_table",
                           pose_targets={"blender.n.01_1": pose_target("full", "full", 0.5, 0.5)}, groups={}, articulation_targets={"blender.n.01_1": art})
    errs = {"pose_errors": {"blender.n.01_1": {"position_error": 0.001, "orientation_error": 0.002}},
            "layout_errors": {"blender.n.01_1": {"joints": {"lid": {"error": 1.2, "joint_type": JointType.JOINT_REVOLUTE}}}}}
    rec = TP.apply_to_task(FakeTask(meta, errs), "replay", customs_path=str(customs))
    root = meta.pose_targets["blender.n.01_1"]
    assert (root.position_tolerance, root.orientation_tolerance) == (0.03, 0.10)  # small group
    assert art.joints == {"lid": (1.2, pytest.approx(0.12))}  # micro: 10 % of 1.2 rad
    assert set(rec["targets"]) == {"blender.n.01_1", "blender.n.01_1:joints"}
    assert rec["targets"]["blender.n.01_1"]["instance"] == rec["targets"]["blender.n.01_1:joints"]["instance"] == "blender.n.01_1"


def test_target_tolerances_read_the_loaded_meta():
    meta = SimpleNamespace(pose_targets={"a": pose_target("full", None, 0.01, 0.2)}, groups={"g": SimpleNamespace(pose_tolerances=(0.03, 0.1), layout_tolerances=None)},
                           articulation_targets={"d": SimpleNamespace(joints={"j": (0.5, 0.07)})})
    assert TP.target_tolerances(meta, "a", "pose_reached") == {"position": 0.01, "orientation": None}
    assert TP.target_tolerances(meta, "g", "pose_reached") == {"position": 0.03, "orientation": 0.1}
    assert TP.target_tolerances(meta, "g", "layout_reached") is None
    assert TP.target_tolerances(meta, "d", "layout_reached") == {"joints": {"j": 0.07}}
    assert TP.target_tolerances(None, "a", "pose_reached") is None
