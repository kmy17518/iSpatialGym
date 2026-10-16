from types import SimpleNamespace

import pytest

from ispatialgym.eval import tolerance_profile as TP

IORS = "inter_object_relation_satisfaction"


@pytest.mark.parametrize(
    "variant,kind,group",
    [
        ("object_scaling-apple", "pose", "micro"),
        ("alignment-world_vertical-painting-walls", "pose", "micro"),
        ("configuration_matching-articulation_open_small_scale-blender-table", "joints", "micro"),
        ("configuration_matching-articulation_open_small_scale-blender-table", "pose", "small"),
        ("configuration_matching-block_design-block_4-table", "group_layout", "micro"),
        ("configuration_matching-block_design-block_4-table", "group_pose", "small"),
        ("disembedding-jigsaw_puzzle_piece-jigsaw_puzzle_piece_3-table", "pose", "small"),
        (f"{IORS}-ontop_small_scale-a-b-table", "pose", "small"),
        (f"{IORS}-nextto_large_scale-a-b-floor", "pose", "large"),
        ("camera_relocalization-obstructed-hamper_1", "pose", "large"),
    ],
)
def test_scale_groups(variant, kind, group):
    assert TP.scale_group(variant, kind) == group


def test_unknown_targets_raise():
    with pytest.raises(TP.ToleranceProfileError):
        TP.scale_group("camera_relocalization-standard", "joints")
    with pytest.raises(TP.ToleranceProfileError):
        TP.plan("mental_rotation-mug", [{"name": "x", "kind": "pose", "components": ["position"]}], scale=0)


def plan1(variant, target, scale=1.0):
    return TP.plan(variant, [{"name": "x", **target}], scale=scale)["x"]


def test_micro_poses_take_ten_percent_of_the_change_above_the_floors():
    big = plan1(
        "mental_rotation-mug",
        {"kind": "pose", "components": ["position", "orientation"], "change": {"position": 0.3, "orientation": 1.0}},
    )
    assert big["tolerances"] == {"position": pytest.approx(0.03), "orientation": pytest.approx(0.1)}
    small = plan1(
        "mental_rotation-mug", {"kind": "pose", "components": ["orientation"], "change": {"orientation": 0.1}}, scale=3
    )
    assert small["tolerances"] == {"position": None, "orientation": pytest.approx(0.15)}


def test_joints_relations_and_fixed_groups():
    joints = plan1(
        "configuration_matching-articulation_open_small_scale-blender-table",
        {
            "kind": "joints",
            "joints": {"lid": {"type": "revolute", "change": -1.5}, "still": {"type": "prismatic", "change": 0.0}},
        },
    )
    assert joints["tolerances"] == {"lid": pytest.approx(0.15), "still": 0.01}
    relation = plan1(
        "configuration_matching-block_design-b-c",
        {"kind": "group_layout", "components": ["position", "orientation"], "change": {"position": 1.0}},
    )
    assert relation["tolerances"] == {"position": 0.01, "orientation": 0.05}
    large = plan1(
        "camera_relocalization-obstructed-x", {"kind": "pose", "components": ["position", "orientation"], "change": {}}
    )
    assert large["tolerances"] == {"position": 0.05, "orientation": 0.15}


class FakeTask:
    def __init__(self, meta, errors):
        self.tro, self.errors, self.configured = SimpleNamespace(meta=meta), errors, []

    def configure_targets(self, meta):
        self.configured.append({k: (t.position_metric, t.position_tolerance) for k, t in meta.pose_targets.items()})

    def collect_errors(self):
        return self.errors


def test_apply_to_task_object_scaling_compares_the_full_position():
    target = SimpleNamespace(
        position_metric="depth", orientation_metric=None, position_tolerance=0.2, orientation_tolerance=None
    )
    meta = SimpleNamespace(
        task_variant="object_scaling-apple", pose_targets={"apple.n.01_1": target}, groups={}, articulation_targets={}
    )
    task = FakeTask(
        meta,
        {"pose_errors": {"apple.n.01_1": {"position_error": 0.12, "orientation_error": None}}, "layout_errors": {}},
    )
    record = TP.apply_to_task(task, scale=3.0)
    assert target.position_metric == "full" and target.position_tolerance == pytest.approx(0.036)
    assert task.configured == [{"apple.n.01_1": ("full", 0.2)}, {"apple.n.01_1": ("full", pytest.approx(0.036))}]
    assert record["name"] == "isg-v2" and record["targets"]["apple.n.01_1"]["note"] == {
        "position_metric": "full",
        "was": "depth",
    }
