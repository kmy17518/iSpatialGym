"""ispatialgym.eval.instances and ispatialgym.eval.names: name / index resolution, the dataset layout's template files, the
episodes (demos) and run_isg's step budget, goal images, the export's problem edits (no simulator)."""
import importlib.util
import json
import os

import pytest

from ispatialgym.activity.problem import load_problem, parse_problem_string
from ispatialgym.eval import instances as INST
from ispatialgym.eval import names as N
from ispatialgym.eval.targets import GroupTarget
from ispatialgym.eval.tro import load_tro, validate_tro_against_problem

EXPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "eval_instances", "export_counted.py")


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(data if isinstance(data, str) else json.dumps(data))


def _dataset(root, entries):
    """entries: (task_family, task, subtask, scene); one instance each, files laid out as the exporter writes them."""
    lines, counts = [], {}
    for fam, task, sub, scene in entries:
        jd = os.path.join("scenes", scene, "json")
        prefix = f"{scene}_task_{sub}"
        files = {"template": os.path.join(jd, f"{prefix}_0_0_template.json"),
                 "template_meta": os.path.join(jd, f"{prefix}_0_0_template-meta.json"),
                 "tro_state": os.path.join(jd, f"{prefix}_instances", f"{prefix}_0_0_template-tro_state.json")}
        for kind in files:
            _write(os.path.join(root, files[kind]), {"problem": "alignment-2"} if kind == "template_meta" else {})
        lines.append({"task_family": fam, "task": task, "subtask": sub, "instance": 0, "task_instance": counts.get(task, 0), "scene": scene,
                      "files": files, "source": {"run": sub}})
        counts[task] = counts.get(task, 0) + 1
    _write(os.path.join(root, "metadata", "instances.jsonl"), "".join(json.dumps(x) + "\n" for x in lines))


ENTRIES = [("alignment", "alignment_axial", "alignment_axial_board_game", "house_single_floor"),
           ("alignment", "alignment_axial", "alignment_axial_keyboard", "office_cubicles_right"),
           ("alignment", "alignment_vertical", "alignment_vertical_painting", "Rs_int"),
           ("disembedding", "disembedding", "disembedding_toy_dice_toy_dice_2_toy_figure_1", "house_single_floor")]


def test_resolve_levels(tmp_path):
    root = str(tmp_path)
    _dataset(root, ENTRIES)
    assert [r.subtask for r in INST.resolve("alignment_axial", [0, 1], root)] == [ENTRIES[0][2], ENTRIES[1][2]]
    assert [r.subtask for r in INST.resolve("alignment", [2], root)] == [ENTRIES[2][2]]
    (rec,) = INST.resolve("alignment_axial_keyboard", [0], root)
    assert rec.task == "alignment_axial" and rec.task_instance == 1 and rec.path("tro_state").startswith(root)
    # a name that is both a task family and a task resolves to the same instances
    assert [r.subtask for r in INST.resolve("disembedding", [0], root)] == [ENTRIES[3][2]]
    with pytest.raises(IndexError):
        INST.resolve("alignment_axial", [2], root)
    with pytest.raises(KeyError):
        INST.resolve("alignment-axial", [0], root)


def test_template_files(tmp_path):
    root = str(tmp_path)
    _dataset(root, ENTRIES[:1])
    (rec,) = INST.resolve(ENTRIES[0][2], [0], root)
    tf = INST.template_files(rec.path("tro_state"))
    assert tf["template"] == rec.path("template") and tf["template_meta"] == rec.path("template_meta")
    assert tf["activity_definitions"] == INST.ACTIVITY_DEFINITIONS and tf["root"] == root  # the repository's problems
    # a sampler output (<template>/instances/<iid>-tro_state.json) is not the dataset layout
    other = os.path.join(root, "out", "variant", "tid", "instances", "x__0000-tro_state.json")
    _write(other, {})
    assert INST.template_files(other) is None


def _goal_files(rec):
    tro = rec.path("tro_state")
    rec.files.update({"oracle": tro.replace("-tro_state.json", ".oracle.json"), "goal_rgb": tro.replace("-tro_state.json", "-goal_rgb.png")})
    _write(tro, {"ispatialgym": {"goal_observation": None, "pose_targets": {}}})
    _write(rec.path("oracle"), {"schema": "x", "tro_sha256": "0" * 64})


def test_register_goal_image(tmp_path, monkeypatch):
    monkeypatch.setenv(INST.RECORDS_ENV, str(tmp_path / "records"))
    root = str(tmp_path / "ds")
    _dataset(root, ENTRIES[:1])
    (rec,) = INST.resolve(ENTRIES[0][2], [0], root)
    _goal_files(rec)
    tro = rec.path("tro_state")
    assert not INST.register_goal_image(rec)  # no goal image yet
    _write(rec.path("goal_rgb"), "png bytes")
    cam = {"sensor_key": "robot_r1:zed_link:Camera:0", "position": [1.0, 2.0, 1.5], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
           "intrinsics": {"fx": 459.0, "fy": 459.0, "cx": 540.0, "cy": 540.0, "width": 1080, "height": 1080}}
    source = {"replay": "rerun", "run_dir": "/runs/t/r", "replay_dir": "/replays/t/r/replay_log/d", "success": True}
    _write(INST.goal_record_path(rec), {"source": source, "sha256": "x", "camera": cam, "image_size": [1080, 1080]})
    assert INST.register_goal_image(rec)
    go = json.load(open(tro))["ispatialgym"]["goal_observation"]
    assert go["rgb_path"] == os.path.basename(rec.path("goal_rgb")) and go["rgb_sha256"] == INST.sha256_of(rec.path("goal_rgb"))
    assert go["sensor_key"] == cam["sensor_key"] and go["camera_pose_world"] == {"position": cam["position"], "orientation": cam["orientation_xyzw"]}
    assert go["resolution"] == [1080, 1080] and go["intrinsics"] == cam["intrinsics"]
    assert not {"source", "info_path", "n_renders"} & set(go)  # where the image came from stays in the internal record
    assert json.load(open(tro))["ispatialgym"]["pose_targets"] == {}
    assert json.load(open(rec.path("oracle")))["tro_sha256"] == INST.sha256_of(tro)


def _replay(replay_dir, size, instance_id, success=True, mode="replay"):
    """A run_isg replay dir: meta.json (as run_isg writes it) and final_rgb.png of size x size."""
    from PIL import Image

    os.makedirs(replay_dir, exist_ok=True)
    fx = size * 17.0 / 40.0
    _write(os.path.join(replay_dir, "meta.json"), {"instance_id": instance_id, "mode": mode, "success": success, "goal_observation": None,
                                                   "loader": {"head_camera_key": "robot_r1:zed_link:Camera:0"},
                                                   "qa": {"camera": {"fx": fx, "fy": fx, "cx": size / 2, "cy": size / 2, "width": size, "height": size}}})
    Image.new("RGB", (size, size), (size % 256, 10, 20)).save(os.path.join(replay_dir, "final_rgb.png"))


def test_set_goal(tmp_path, monkeypatch):
    monkeypatch.setenv(INST.RECORDS_ENV, str(tmp_path / "records"))
    import h5py

    root = str(tmp_path / "ds")
    _dataset(root, ENTRIES[:1])
    (rec,) = INST.resolve(ENTRIES[0][2], [0], root)
    _goal_files(rec)
    run_dir = str(tmp_path / "eval" / "runs" / "s3" / "pick_01" / "c1_k1")
    demo_id = "variant__scene__0000"
    demo = os.path.join(run_dir, "demo", demo_id, "demo.hdf5")
    os.makedirs(os.path.dirname(demo))
    poses = [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [8.7, 3.2, 1.3, 0.1, 0.2, 0.3, 0.927]]
    with h5py.File(demo, "w") as f:
        f.create_dataset("data/demo_0/isg/head_camera_pose", data=poses)
    rec.source = {"run_dir": run_dir, "demo_hdf5": demo}
    (_, rerun), (_, original) = INST.replay_dirs(rec)
    assert rerun == str(tmp_path / "eval" / "review_videos" / "replays" / "s3" / "pick_01" / "c1_k1" / "replay_log" / demo_id)
    assert original == os.path.join(run_dir, "replay_log", demo_id)

    with pytest.raises(INST.GoalImageError, match="no usable replay"):  # never the demo's own frame
        INST.set_goal(rec)
    _replay(original, 16, demo_id)
    _replay(rerun, 12, demo_id, success=False)
    info = INST.set_goal(rec)  # the re-run failed: the original replay
    assert info["source"]["replay"] == "original" and info["image_size"] == [16, 16]
    _replay(rerun, 12, demo_id)
    info = INST.set_goal(rec)  # the re-run first
    assert info["source"] == {"replay": "rerun", "run_dir": run_dir, "replay_dir": rerun, "success": True, "frame": os.path.join(rerun, "final_rgb.png")}
    assert info["image_size"] == [12, 12] and info["sha256"] == INST.sha256_of(rec.path("goal_rgb")) == INST.sha256_of(os.path.join(rerun, "final_rgb.png"))
    cam = info["camera"]
    assert cam["sensor_key"] == "robot_r1:zed_link:Camera:0" and cam["intrinsics"]["width"] == 12
    assert cam["position"] == pytest.approx(poses[-1][:3]) and cam["orientation_xyzw"] == pytest.approx(poses[-1][3:])
    assert json.load(open(INST.goal_record_path(rec))) == info

    # another image size: a run_isg --out dir of this instance (run_isg writes <out>/<instance id>/)
    out = str(tmp_path / "rereplay")
    _replay(os.path.join(out, INST.instance_id(rec)), 20, INST.instance_id(rec))
    info = INST.set_goal(rec, out)
    assert info["source"]["replay"] == "run" and info["source"]["run_dir"] == out and info["image_size"] == [20, 20]
    go = json.load(open(rec.path("tro_state")))["ispatialgym"]["goal_observation"]
    assert go["resolution"] == [20, 20] and go["intrinsics"]["fx"] == pytest.approx(8.5) and "source" not in go
    assert go["rgb_sha256"] == INST.sha256_of(rec.path("goal_rgb"))
    assert json.load(open(rec.path("oracle")))["tro_sha256"] == INST.sha256_of(rec.path("tro_state"))
    # a run of another instance or a run that is not a replay is refused
    _replay(str(tmp_path / "other"), 20, "another_instance")
    with pytest.raises(INST.GoalImageError, match="a run of 'another_instance'"):
        INST.set_goal(rec, str(tmp_path / "other"))
    _replay(str(tmp_path / "oracle_run"), 20, demo_id, mode="oracle")
    with pytest.raises(INST.GoalImageError, match="not a replay"):
        INST.set_goal(rec, str(tmp_path / "oracle_run"))


def test_names_leave_out_the_support():
    assert N.task_family("alignment-axial-board_game-breakfast_table") == "alignment"
    assert N.task_name("alignment-axial-board_game-breakfast_table") == "alignment_axial"
    assert N.subtask_name("alignment-axial-board_game-breakfast_table") == "alignment_axial_board_game"
    assert N.task_name("alignment-world_vertical-painting-walls") == "alignment_vertical"
    assert N.subtask_name("alignment-world_vertical-painting-walls") == "alignment_vertical_painting"
    assert N.subtask_name("camera_relocalization-standard") == "camera_relocalization_standard"
    assert N.subtask_name("camera_relocalization-obstructed-vacuum_1") == "camera_relocalization_obstructed_vacuum_1"
    # a group slot is one token; the partner stays, the support goes
    assert N.subtask_name("disembedding-toy_dice-toy_dice_2_toy_figure_1-breakfast_table") == "disembedding_toy_dice_toy_dice_2_toy_figure_1"
    assert N.subtask_name("inter_object_relation_satisfaction-ontop_large_scale-soap_dispenser-countertop-floor") \
        == "inter_object_relation_satisfaction_ontop_large_scale_soap_dispenser_countertop"
    assert N.subtask_name("configuration_matching-articulation_close_large_scale-door") == "configuration_matching_articulation_close_large_scale_door"
    # a token naming one object model or one scene object is reduced to its category; a category ending in a six-letter
    # word is not a model (pepper_shaker)
    assert N.subtask_name("inter_object_relation_satisfaction-ontop_small_scale-bar_soap_utgixp-soap_dish-shelf") \
        == "inter_object_relation_satisfaction_ontop_small_scale_bar_soap_soap_dish"
    assert N.subtask_name("inter_object_relation_satisfaction-inside_large_scale-plate-drop_in_sink_lkklqs_0-bar") \
        == "inter_object_relation_satisfaction_inside_large_scale_plate_drop_in_sink"
    assert N.object_category("pepper_shaker") == "pepper_shaker" and N.object_category("hamper_1") == "hamper_1"
    assert (N.task_family("mental_rotation-statue"), N.task_name("mental_rotation-statue"), N.subtask_name("mental_rotation-statue")) \
        == ("mental_rotation", "mental_rotation", "mental_rotation_statue")


def test_step_budget():
    assert INST.step_budget(806) == {"mode": "oracle", "max_steps": 1612, "settle_steps": 45, "wallclock_s": 1612.0, "demo_length": 806,
                                     "demo_settle_steps": None, "episode_cap": 1657}  # the oracle's epilogue reserved on top
    assert INST.step_budget(100)["max_steps"] == 300  # at least 300 steps
    assert INST.step_budget(None)["max_steps"] == 3000 and INST.step_budget(None)["episode_cap"] == 3045
    b = INST.step_budget(806, max_steps=500, settle_steps=10, wallclock_s=60.0)
    assert (b["max_steps"], b["settle_steps"], b["wallclock_s"], b["episode_cap"]) == (500, 10, 60.0, 510)
    # a replay streams its demo's settle rows (all by default): no epilogue, the episode is max_steps
    b = INST.step_budget(806, mode="replay", demo_settle_steps=322)
    assert (b["settle_steps"], b["episode_cap"], b["demo_settle_steps"]) == (322, 1612, 322)
    assert INST.step_budget(806, settle_steps=100, mode="replay", demo_settle_steps=322)["settle_steps"] == 100
    b = INST.step_budget(806, mode="policy")
    assert (b["settle_steps"], b["episode_cap"]) == (45, 1612)


@pytest.mark.parametrize("kw,match", [
    ({"settle_steps": 0}, "at least 1"),
    ({"mode": "replay", "demo_settle_steps": 322, "settle_steps": 323}, "exceeds the demo's 322 settle rows"),
    ({"mode": "replay"}, "settle rows are unknown"),
    ({"mode": "policy", "max_steps": 40}, "exceeds --max-steps 40"),
    ({"mode": "train"}, "mode must be"),
])
def test_step_budget_rejects_a_window_that_cannot_fit(kw, match):
    with pytest.raises(ValueError, match=match):
        INST.step_budget(806, **kw)


def test_episodes_demos_and_roots(tmp_path, monkeypatch):
    root = str(tmp_path / "data" / INST.DATASET_NAME)
    _dataset(root, ENTRIES[:2])
    demo = os.path.join("scenes", "house_single_floor", "json", "x_instances", "x_0_0_template" + INST.DEMO_SUFFIX)
    _write(os.path.join(root, "metadata", "episodes.jsonl"),
           json.dumps({"episode_index": 0, "subtask": ENTRIES[0][2], "instance": 0, "length": 806, "demo": demo}) + "\n")
    monkeypatch.delenv(INST.DEMOS_ENV, raising=False)
    first, second = INST.resolve("alignment_axial", [0, 1], root)
    assert first.episode["length"] == 806 and second.episode == {} and second.demo_path() is None
    assert first.demo_path() == os.path.join(str(tmp_path / "data" / INST.DEMOS_NAME), demo)  # the sibling demos dataset
    monkeypatch.setenv(INST.DEMOS_ENV, str(tmp_path / "elsewhere"))
    assert first.demo_path() == os.path.join(str(tmp_path / "elsewhere"), demo)
    # next to the challenge instances in OmniGibson's data folder
    monkeypatch.delenv(INST.ROOT_ENV, raising=False)
    monkeypatch.delenv(INST.DEMOS_ENV, raising=False)
    monkeypatch.setenv("OMNIGIBSON_DATA_PATH", str(tmp_path / "og"))
    assert INST.default_root() == str(tmp_path / "og" / INST.DATASET_NAME)
    assert INST.demos_root() == str(tmp_path / "og" / INST.DEMOS_NAME)


def _export_module():
    spec = importlib.util.spec_from_file_location("export_counted", EXPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BOARD_PROBLEM = """(define (problem configuration_matching-11)
    (:domain ispatialgym)
    (:variant configuration_matching-block_design-block_4-breakfast_table_board)
    (:objects toy_dice.n.01_1 toy_dice.n.01_2 - toy_dice.n.01 tray.n.01_1 - tray.n.01 breakfast_table.n.01_1 - breakfast_table.n.01
              floor.n.01_1 - floor.n.01 agent.n.01_1 - agent.n.01)
    (:groups {members} - targets_group.n.01_1)
    (:init {init} (ontop agent.n.01_1 floor.n.01_1))
    (:goal (and {ontop} (layout_reached ?targets_group.n.01_1) (not (grasped ?agent.n.01_1 ?targets_group.n.01_1))
                (pose_settled ?targets_group.n.01_1) {settled} (pose_settled ?agent.n.01_1)))
    (:measure (pose_reached ?targets_group.n.01_1) (layout_reached ?targets_group.n.01_1)))"""
#: the run's problem (blocks only in the group) and the release's (the board added, ontop per block)
RUN_PARTS = {"members": "toy_dice.n.01_1 toy_dice.n.01_2", "ontop": "(ontop ?targets_group.n.01_1 ?tray.n.01_1)", "settled": "(pose_settled ?tray.n.01_1)",
             "init": "(ontop targets_group.n.01_1 breakfast_table.n.01_1) (ontop tray.n.01_1 breakfast_table.n.01_1)"}
BOARD_PARTS = {"members": "toy_dice.n.01_1 toy_dice.n.01_2 tray.n.01_1", "ontop": "(ontop ?toy_dice.n.01_1 ?tray.n.01_1) (ontop ?toy_dice.n.01_2 ?tray.n.01_1)",
               "settled": "", "init": "(ontop targets_group.n.01_1 breakfast_table.n.01_1)"}


def _board_problem(**parts):
    return parse_problem_string(BOARD_PROBLEM.format(**{**BOARD_PARTS, **parts}), "configuration_matching", 11)


def test_export_accepts_members_added_to_a_group():
    E = _export_module()
    run = _board_problem(**RUN_PARTS)
    assert E.members_added(run, _board_problem()) == {"targets_group.n.01_1": ["tray.n.01_1"]}
    assert E.members_added(run, run) is None
    for parts in ({"init": "(ontop targets_group.n.01_1 breakfast_table.n.01_1) (ontop tray.n.01_1 floor.n.01_1)"},  # init changed
                  {"ontop": "(ontop ?toy_dice.n.01_1 ?tray.n.01_1)"},  # a goal atom of the run dropped
                  {"ontop": BOARD_PARTS["ontop"] + " (ontop ?toy_dice.n.01_1 ?toy_dice.n.01_2)"},  # an atom not on the added member
                  {"members": "toy_dice.n.01_1 tray.n.01_1"}):  # a run member dropped
        assert E.members_added(run, _board_problem(**parts)) is None, parts


def test_export_pins_added_members(tmp_path):
    h5py = pytest.importorskip("h5py")
    E = _export_module()
    demo = tmp_path / "demo.hdf5"
    with h5py.File(demo, "w") as f:  # the board at its start, then at the demo's end (turned 90 deg)
        f.create_dataset("data/demo_0/isg/object_pose/tray.n.01_1", data=[[9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0],
                                                                         [0.15, 0.1, 0.0, 0.0, 0.0, 0.70710678, 0.70710678]])
    group = {"members": ["toy_dice.n.01_1", "toy_dice.n.01_2"], "assignment": {"mode": "exchangeable", "cost": "layout"},
             "orientation_equivalences": [], "pose_tolerances": {"position": 0.01, "orientation": 0.15},
             "target_slots": {"slot_0": {"position": [0.0, 0.0, 0.0], "orientation": [0, 0, 0, 1]},
                              "slot_1": {"position": [0.1, 0.0, 0.0], "orientation": [0, 0, 0, 1]}},
             "layout_relations": [{"from": "slot_0", "to": "slot_1", "position": [0.1, 0.0, 0.0], "orientation": [0, 0, 0, 1]}],
             "layout_tolerances": {"position": 0.01, "orientation": 0.15}}
    block = {"groups": {"targets_group.n.01_1": group}}
    E.pin_added_members(block, {"targets_group.n.01_1": ["tray.n.01_1"]}, str(demo))
    assert group["members"] == ["toy_dice.n.01_1", "toy_dice.n.01_2"] and len(group["layout_relations"]) == 1  # the run's data stays
    g = GroupTarget.from_dict("targets_group.n.01_1", block["groups"]["targets_group.n.01_1"])  # the relations fit the slots
    assert g.members == ["toy_dice.n.01_1", "toy_dice.n.01_2", "tray.n.01_1"] and g.pinned == {"tray.n.01_1": "slot_tray.n.01_1"}
    assert g.target_slots["slot_tray.n.01_1"].position.tolist() == pytest.approx([0.15, 0.1, 0.0])
    assert {(r.from_slot, r.to_slot) for r in g.layout_relations} == {("slot_0", "slot_1"), ("slot_tray.n.01_1", "slot_0"), ("slot_tray.n.01_1", "slot_1")}


@pytest.mark.skipif(not os.path.exists(os.path.join(INST.default_root(), "metadata", "instances.jsonl")), reason="no exported dataset")
def test_exported_dataset_consistent():
    index = INST.load_index()
    assert index, "empty index"
    # published: no BDDL problems (the repository's) and no goal records (internal)
    assert not os.path.exists(os.path.join(INST.default_root(), "activity_definitions"))
    assert not [f for _d, _s, fs in os.walk(INST.default_root()) for f in fs if f.endswith("-goal.json")]
    names = [r.subtask for r in index]
    assert len(names) == len(set(names))
    problems = {}
    for rec in index:
        for kind in ("template", "template_meta", "tro_state", "oracle"):
            assert os.path.exists(rec.path(kind)), (rec.subtask, kind)
        tf = INST.template_files(rec.path("tro_state"))
        assert tf is not None and tf["template"] == rec.path("template") and tf["template_meta"] == rec.path("template_meta")
        assert "variant" not in rec.subtask
        # the sidecar is rejected when its tro_sha256 does not match the instance file
        assert json.load(open(rec.path("oracle")))["tro_sha256"] == INST.sha256_of(rec.path("tro_state")), rec.subtask
        # every instance has its goal image, registered in the instance file
        assert os.path.exists(rec.path("goal_rgb")) and os.path.exists(INST.goal_record_path(rec)), rec.subtask
        info = json.load(open(INST.goal_record_path(rec)))
        # a replay that re-pinned the measured targets qualified by its symbolic success (its geometric one failed on them)
        assert info["source"]["success"] is True or info["source"].get("measured_targets") is True, rec.subtask
        assert info["sha256"] == INST.sha256_of(rec.path("goal_rgb")), rec.subtask
        tro = json.load(open(rec.path("tro_state")))["ispatialgym"]
        go = tro.get("goal_observation")
        assert go and go["rgb_path"] == os.path.basename(rec.path("goal_rgb")) and go["rgb_sha256"] == info["sha256"], rec.subtask
        assert go["resolution"] == info["image_size"] and not {"source", "info_path"} & set(go), rec.subtask
        # the names come from the variant; the support is left out
        variant = rec.source["variant"]
        assert (rec.task_family, rec.task, rec.subtask) == (N.task_family(variant), N.task_name(variant), N.subtask_name(variant)), rec.subtask
        tmeta = json.load(open(rec.path("template_meta")))
        sidecar = json.load(open(rec.path("oracle")))
        assert tmeta["variant"] == tmeta["eval"]["variant"] == sidecar["variant"] == tro["task_variant"] == variant, rec.subtask
        # the problem: file name, its own name, its variant, the template metadata and the instance file agree
        activity, did = rec.extra["problem"].rsplit("-", 1)
        problem_file = os.path.join(INST.ACTIVITY_DEFINITIONS, activity, f"problem{did}.bddl")  # the repository's, as in BEHAVIOR
        text = open(problem_file).read()
        assert f"(define (problem {activity}-{did})" in text and f"(:variant {tmeta['bddl_variant']})" in text, rec.subtask
        assert tmeta["problem"] == rec.extra["problem"] and int(tro["activity_definition_id"]) == int(did)
        problems.setdefault(activity, set()).add(int(did))
        # the instance file fits its problem: a target per pose_reached / layout_reached atom, the groups' members
        validate_tro_against_problem(load_tro(rec.path("tro_state")).meta, load_problem(activity, int(did)).compile(generate_ground_options=False))
        # the demo replays use, and its length the step budget
        assert rec.episode.get("length", 0) > 0 and os.path.exists(rec.demo_path()), rec.subtask
        assert INST.sha256_of(rec.demo_path()) == rec.episode["demo_sha256"], rec.subtask
    # the problems of each activity are numbered from 0
    assert all(ids == set(range(len(ids))) for ids in problems.values()), problems
    with open(os.path.join(os.path.dirname(INST.__file__), "..", "assets", "task_specs", "isg_customs.json")) as f:
        assert {k for k in json.load(f) if not k.startswith("_")} == set(names)
