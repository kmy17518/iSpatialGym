"""The 2026-10-14 scoring change around a run: a replay's settle rows cut to ``--settle-steps``, a checkpoint rollback of the task's
success tracking, and the summary of old and new ``meta.json``."""

import json

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.tests.test_v2_oracle import _place_goal, iors_world, make_executor, success_eval


def _demo(path, phases):
    h5py = pytest.importorskip("h5py")
    acts = np.arange(len(phases) * M.ACTION_DIM, dtype=np.float32).reshape(len(phases), M.ACTION_DIM)
    with h5py.File(path, "w") as f:
        f.create_dataset("data/demo_0/action", data=acts)
        f.create_dataset("data/demo_0/isg/phase", data=np.asarray(phases, dtype=np.int64))
    return acts


def test_replay_streams_the_action_rows_and_the_first_settle_rows(tmp_path):
    from ispatialgym.oracle import run_isg as R

    path = str(tmp_path / "demo.hdf5")
    acts = _demo(path, [0] * 6 + [1] * 4)
    assert R.replay_counts(path) == (6, 4)
    rows = list(R.replay_rows(path, 0, 6, 2))
    assert len(rows) == 8 and np.array_equal(np.stack([r.numpy() for r in rows]), acts[:8])
    assert len(list(R.replay_rows(path, 0, 6, 4))) == 10  # all of them (the default)
    # rows recorded out of order are refused
    bad = str(tmp_path / "bad.hdf5")
    _demo(bad, [0, 0, 1, 0, 1])
    with pytest.raises(ValueError, match="action rows then the settle rows"):
        R.replay_counts(bad)
    # a LeRobot episode: the eval dataset's counts or an error
    lerobot = tmp_path / "lerobot"
    lerobot.mkdir()
    with pytest.raises(ValueError, match="no action / settle row counts"):
        R.replay_counts(str(lerobot))
    assert R.replay_counts(str(lerobot), 0, {"action_steps": 700, "settle_steps": 144, "length": 844}) == (700, 144)


def test_settle_steps_beyond_the_demos_settle_rows_is_an_error():
    from ispatialgym.eval import instances as INST

    with pytest.raises(ValueError, match="exceeds the demo's 4 settle rows"):
        INST.step_budget(10, settle_steps=5, mode="replay", demo_settle_steps=4)


def test_rollback_restores_the_success_tracking(tmp_path):
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, *_ = make_executor(env, sc, tmp_path)
    hold = th.zeros(M.ACTION_DIM)
    idx = env.robot.controller_action_idx
    hold[idx["trunk"]] = env.robot.q[env.robot.trunk_control_idx]
    for arm in M.ARMS:
        hold[idx[f"arm_{arm}"]] = env.robot.q[env.robot.arm_control_idx[arm]]
    for _ in range(3):
        ex.step(hold.clone())
    cp = ex.checkpoint(0)
    assert cp.task_tracking == {"metrics_state": env.task.tracker.state()}
    before = env.task.tracker.state()
    env.task.tracker.first["off"]["symbolic"] = 2  # what a rolled-back attempt left behind
    env.task.tracker.g_run = 7
    for _ in range(2):
        ex.step(hold.clone())
    ex.restore(cp, 0, 1)
    after = env.task.tracker.state()
    # the rolled-back attempt's tracking is gone (restore steps its one hold step on from the checkpoint's)
    assert after["first"] == before["first"] and after["g_run"] == before["g_run"] == 0
    assert after["last"][0] == env._current_step and env._current_step - before["last"][0] == ex.n - cp.step <= 1


def test_set_goal_measured_targets_repins_only_the_measure_only_targets(tmp_path):
    from ispatialgym.activity.problem import load_problem
    from ispatialgym.eval import instances as INST

    compiled = load_problem("inter_object_relation_satisfaction", 1).compile()  # ontop_small_scale kielbasa / platter
    measured = [i for i in compiled.measure_pose_reached_args if not compiled.is_gating("pose_reached", i)]
    assert len(measured) == 2
    target = {"link": None, "frame": "world", "position": [1.0, 2.0, 0.8], "orientation": [0.0, 0.0, 0.0, 1.0], "position_metric": "full",
              "orientation_metric": "full", "position_tolerance": 0.03, "orientation_tolerance": 0.2}
    tro = {"ispatialgym": {"activity": "inter_object_relation_satisfaction", "activity_definition_id": 1,
                           "pose_targets": {measured[0]: dict(target), measured[1]: {**target, "orientation": None, "orientation_metric": None}}}}
    (tmp_path / "x-tro_state.json").write_text(json.dumps(tro))
    rec = INST.InstanceRecord(task_family="f", task="t", subtask="s", instance=0, task_instance=0, scene="x", files={"tro_state": "x-tro_state.json"},
                              root=str(tmp_path))
    final = {measured[0]: {"frame": "world", "link": None, "position": [1.01, 2.0, 0.8], "orientation": [0.0, 0.0, 0.38, 0.92]},
             measured[1]: {"frame": "world", "link": None, "position": [1.0, 2.05, 0.8], "orientation": [0.0, 0.0, 0.0, 1.0]}}
    changed = INST.repin_measured_targets(rec, final)
    new = json.loads((tmp_path / "x-tro_state.json").read_text())["ispatialgym"]["pose_targets"]
    assert new[measured[0]]["position"] == [1.01, 2.0, 0.8] and new[measured[0]]["orientation"] == [0.0, 0.0, 0.38, 0.92]
    assert new[measured[1]]["position"] == [1.0, 2.05, 0.8] and new[measured[1]]["orientation"] is None  # not compared: stays None
    assert changed[measured[0]]["position_was"] == [1.0, 2.0, 0.8] and "orientation_was" not in changed[measured[1]]
    with pytest.raises(INST.GoalImageError, match="frame"):
        INST.repin_measured_targets(rec, {**final, measured[0]: {**final[measured[0]], "frame": "robot_r1::zed_link:Camera:0"}})
    with pytest.raises(INST.GoalImageError, match="give --run"):
        INST.set_goal(rec, None, measured_targets=True)


def _write_meta(root, name, meta):
    d = root / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(meta))


def test_summary_of_new_and_old_meta(tmp_path, capsys):
    from ispatialgym.harness.summarize import episode_rows, main

    new = {"mode": "replay", "status": "success", "success": True, "task_family": "alignment", "settle_eval": True, "symbolic_success": True,
           "geometric_success": True, "error_metric": {"position_error": None, "orientation_error": 0.02, "case": "pose"},
           "other_mode": {"settle_eval": False, "symbolic_success": True, "geometric_success": True}}
    _write_meta(tmp_path, "a", new)
    _write_meta(tmp_path, "b", {**new, "status": "oracle_failed", "success": False, "geometric_success": False,
                                "error_metric": {"position_error": None, "orientation_error": 0.4, "case": "pose"},
                                "other_mode": {"settle_eval": False, "symbolic_success": True, "geometric_success": False}})
    _write_meta(tmp_path, "c", {**new, "symbolic_success": False, "geometric_success": False, "success": False, "status": "oracle_failed",
                                "error_metric": {"position_error": None, "orientation_error": 1.0, "case": "pose"},
                                "other_mode": {"settle_eval": False, "symbolic_success": False, "geometric_success": False}})
    _write_meta(tmp_path, "old", {"mode": "replay", "status": "success", "success": True, "variant": "alignment-axial-x-y"})  # before the change
    _write_meta(tmp_path, "err", {"mode": "replay", "status": "error", "error": "boom", "task_family": "alignment"})
    (tmp_path / "noise").mkdir()
    (tmp_path / "noise" / "meta.json").write_text(json.dumps({"something": "else"}))  # not a run_isg meta.json
    rows = {r["episode"]: r for r in episode_rows(str(tmp_path))}
    assert set(rows) == {"a", "b", "c", "old", "err"}
    assert (rows["a"]["geometric"], rows["b"]["geometric"], rows["b"]["symbolic"], rows["b"]["other_symbolic"]) == (True, False, True, True)
    assert rows["old"]["old_meta"] and rows["old"]["geometric"] is None and rows["old"]["symbolic"] is True and rows["err"]["status"] == "unscored"
    main([str(tmp_path)])
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.strip().startswith("alignment"))
    # geometric over the episodes that have it (the old one has not), symbolic over all four, then the other mode's two
    assert line.split()[1:9] == ["1/3", "33.3%", "3/4", "75.0%", "1/3", "33.3%", "2/3", "66.7%"]
    # orientation error: mean over all (0.02, 0.4, 1.0) and over the symbolically successful ones (0.02, 0.4)
    assert "0.4733 / 0.2100" in line and "- / -" in line  # no position component
    assert "flagged: 1 meta.json" in out and "unscored: err" in out
