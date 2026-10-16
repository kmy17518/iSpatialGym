import json

import pytest

from ispatialgym.eval import instances as INST

SCENE = "house_single_floor"


def make_dataset(root, subtasks):
    """A dataset with one instance per ``(task_family, task, subtask)``."""
    lines = []
    for family, task, subtask in subtasks:
        json_dir = root / "scenes" / SCENE / "json"
        prefix = f"{SCENE}_task_{subtask}"
        inst_dir = json_dir / f"{prefix}_instances"
        inst_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "template": f"scenes/{SCENE}/json/{prefix}_0_0_template.json",
            "template_meta": f"scenes/{SCENE}/json/{prefix}_0_0_template-meta.json",
            "tro_state": f"scenes/{SCENE}/json/{prefix}_instances/{prefix}_0_0_template-tro_state.json",
            "goal_rgb": f"scenes/{SCENE}/json/{prefix}_instances/{prefix}_0_0_template-goal_rgb.png",
        }
        for rel in files.values():
            (root / rel).write_text("{}")
        lines.append({"task_family": family, "task": task, "subtask": subtask, "instance": 0, "task_instance": 0, "scene": SCENE,
                      "problem": f"{family}-0", "max_steps": 600, "files": files})  # fmt: skip
    (root / "metadata").mkdir(exist_ok=True)
    (root / "metadata" / "instances.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    return root


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(
        tmp_path,
        [
            ("alignment", "alignment_axial", "alignment_axial_board_game"),
            ("alignment", "alignment_vertical", "alignment_vertical_painting"),
            ("mental_rotation", "mental_rotation", "mental_rotation_carving_knife"),
        ],
    )


def test_names_resolve_at_the_most_specific_level(dataset):
    (rec,) = INST.resolve("alignment_vertical_painting", [0], str(dataset))
    assert rec.task == "alignment_vertical" and rec.max_steps == 600
    assert [r.subtask for r in INST.resolve("alignment", [0, 1], str(dataset))] == [
        "alignment_axial_board_game",
        "alignment_vertical_painting",
    ]
    assert (
        INST.resolve("mental_rotation", [0], str(dataset))[0].instance_id
        == f"{SCENE}_task_mental_rotation_carving_knife_0_0_template"
    )
    with pytest.raises(IndexError):
        INST.resolve("alignment_axial", [1], str(dataset))
    with pytest.raises(KeyError):
        INST.resolve("object_scaling", [0], str(dataset))


def test_instance_files_map_to_their_template(dataset):
    (rec,) = INST.resolve("alignment_axial_board_game", [0], str(dataset))
    files = INST.template_files(rec.path("tro_state"))
    assert files["template"] == rec.path("template") and files["template_meta"] == rec.path("template_meta")
    assert files["root"] == str(dataset)
    assert INST.record_of(rec.path("tro_state"), files["root"]).subtask == rec.subtask
    with pytest.raises(ValueError):
        INST.template_files(str(dataset / "metadata" / "instances.jsonl"))


def test_step_budget():
    assert INST.step_budget(600) == {
        "max_steps": 600,
        "settle_steps": 45,
        "wallclock_s": 600.0,
        "instance_max_steps": 600,
    }
    assert INST.step_budget(None)["max_steps"] == INST.DEFAULT_MAX_STEPS
    assert INST.step_budget(600, max_steps=5, wallclock_s=2.5)["wallclock_s"] == 2.5
    assert INST.step_budget(600, max_steps=5)["settle_steps"] == 45  # the window only has to fit with settle eval on
    with pytest.raises(ValueError):
        INST.step_budget(600, max_steps=5, settle_eval=True)
