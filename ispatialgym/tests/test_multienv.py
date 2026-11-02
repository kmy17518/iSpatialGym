"""Simulator-free tests of the opt-in paths used by ispatialgym.multienv: per-scene checkpoints / recorded states
(``ExecutorConfig.per_scene_state``) and off-thread frames (``RecorderConfig.async_frames``). Each must leave the
recorded episode exactly as the default path records it."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle.executor import ExecutorConfig
from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.recorder import Recorder, RecorderConfig
from ispatialgym.tests.test_v2_oracle import _place_goal, iors_world, make_executor, success_eval


def _run(tmp_path, per_scene: bool, async_frames: bool, video: bool = False, capture=None, rcfg_kw=None, ecfg_kw=None):
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    env.robot.ag_fail_next = 1  # one failed grasp: a checkpoint restore and a rolled-back segment
    if per_scene:
        env.scene.dump_state = env.sim.dump_state
        env.scene.load_state = env.sim.load_state

        def forbidden(*a, **kw):
            raise AssertionError("per_scene_state: the simulator's state must not be used")

        env.sim.dump_state = forbidden
        env.sim.load_state = forbidden
    rcfg = RecorderConfig(hdf5=True, lerobot=False, views=("head",) if video else (), async_frames=async_frames, **(rcfg_kw or {}))
    ecfg = ExecutorConfig(settle_steps=50, per_scene_state=per_scene, **(ecfg_kw or {}))
    state0 = env.scene.dump_state(serialized=True) if per_scene else env.sim.dump_state(serialized=True)
    ex, motions, backend, rec = make_executor(env, sc, None, exec_cfg=ecfg)
    rec = Recorder(str(tmp_path), env, env.robot, "robot_r1:zed_link:Camera:0", dict(env.objects), fps=30, cfg=rcfg, instance_id=sc.instance_id)
    if capture is not None:
        capture(rec)
    rec.start(env.get_obs()[0], {}, state0=state0)
    ex.recorder = rec
    result = ex.run_program(expand(sc.variant, sc))
    paths = rec.close({"instance_id": sc.instance_id, "success": bool(result["success"])}, qa=result["monitors"]["stats"], success=bool(result["success"]))
    return result, rec, paths


def test_per_scene_state_records_the_same_episode(tmp_path):
    h5py = pytest.importorskip("h5py")
    r0, rec0, p0 = _run(tmp_path / "sim", per_scene=False, async_frames=False)
    r1, rec1, p1 = _run(tmp_path / "scene", per_scene=True, async_frames=False)
    assert r0["success"] and r1["success"] and r0["n_steps"] == r1["n_steps"]
    assert len(rec0.states) == len(rec1.states) and all(th.equal(a, b) for a, b in zip(rec0.states, rec1.states))
    with h5py.File(p0["hdf5"], "r") as f0, h5py.File(p1["hdf5"], "r") as f1:
        for key in ("action", "state", "state_size"):
            assert np.array_equal(f0[f"data/demo_0/{key}"][()], f1[f"data/demo_0/{key}"][()]), key
        assert list(f0["data/demo_0/rollbacks"].keys()) == list(f1["data/demo_0/rollbacks"].keys())


def test_async_frames_hand_the_sinks_the_same_frames(tmp_path):
    def capture_into(store):
        def hook(rec):
            orig_video = rec._write_view_frame

            def video(view, frame):
                store["video"].append(np.array(frame, copy=True))
                return orig_video(view, frame)

            rec._write_view_frame = video

            class FakeDataset:
                root = str(tmp_path / f"lerobot_{id(store)}")

                def add_frame(self, frame):
                    store["lerobot"].append({k: (v.clone() if isinstance(v, th.Tensor) else v) for k, v in frame.items()})

                def save_episode(self):
                    store["saved"] = True

                def finalize(self):
                    os.makedirs(self.root, exist_ok=True)

            orig_start = rec.start

            def start(*a, **kw):
                out = orig_start(*a, **kw)
                rec.dataset = FakeDataset()
                return out

            rec.start = start

        return hook

    sync, asyn = {"video": [], "lerobot": []}, {"video": [], "lerobot": []}
    r0, rec0, p0 = _run(tmp_path / "sync", per_scene=False, async_frames=False, video=True, capture=capture_into(sync))
    r1, rec1, p1 = _run(tmp_path / "async", per_scene=False, async_frames=True, video=True, capture=capture_into(asyn))
    assert r0["n_steps"] == r1["n_steps"] and r0["success"] == r1["success"]
    assert len(sync["video"]) == len(asyn["video"]) == r0["n_steps"]
    assert all(np.array_equal(a, b) for a, b in zip(sync["video"], asyn["video"]))
    assert len(sync["lerobot"]) == len(asyn["lerobot"]) > 0
    for a, b in zip(sync["lerobot"], asyn["lerobot"]):
        assert a.keys() == b.keys()
        for k in a:
            if isinstance(a[k], th.Tensor):
                assert th.equal(a[k], b[k]), k
            else:
                assert a[k] == b[k], k
    assert np.array_equal(rec0.last_rgb, rec1.last_rgb)
    assert p0["video"] and p1["video"] and os.path.getsize(p0["video"]) == os.path.getsize(p1["video"])


class _FakeLeRobotDataset:
    """What ShardWriter's LeRobot calls touch: add_frame (slow, as a writer waiting on its encoders), save_episode,
    clear_episode_buffer, finalize. ``events`` is the sequence of calls with the frames' values."""

    def __init__(self, fail_at=None):
        self.events, self.pending, self.num_episodes, self.fail_at, self.n_added = [], 0, 0, fail_at, 0
        self.meta = SimpleNamespace(video_keys=[])

    def add_frame(self, frame):
        import time

        time.sleep(0.002)
        if self.fail_at is not None and self.n_added == self.fail_at:
            raise RuntimeError("encoder crashed")
        self.n_added += 1
        self.pending += 1
        self.events.append(("frame", {k: (v.tolist() if isinstance(v, th.Tensor) else v) for k, v in frame.items()}))

    def has_pending_frames(self):
        return self.pending > 0

    def clear_episode_buffer(self):
        self.pending = 0
        self.events.append(("clear",))

    def save_episode(self):
        self.pending = 0
        self.num_episodes += 1
        self.events.append(("save",))

    def finalize(self):
        self.events.append(("finalize",))


def _shard_writer(async_flush, dataset):
    BL = pytest.importorskip("ispatialgym.oracle.behavior_lerobot")
    w = object.__new__(BL.ShardWriter)
    w.env = SimpleNamespace(flat=lambda obs: obs)
    w._process_obs = lambda obs, info: dict(obs)
    w._parse_step_data = lambda action, obs, reward, terminated, truncated, info: {
        "obs": dict(obs), "action": action, "reward": reward, "terminated": terminated, "truncated": truncated}
    w.dataset, w.task_name = dataset, "task"
    w.flush_every_n_steps, w.flush_every_n_traj, w.traj_count, w.step_count = 4, 1, 0, 0
    w.current_traj_history, w.current_episode_step_count, w.aborted = [], 0, False
    w._bg_flush = BL._BackgroundFlush() if async_flush else None
    return w


def _episode(w, n, start=0.0):
    w.begin({"observation.state": th.tensor([start])})
    for i in range(1, n + 1):
        w.step(th.tensor([float(i)]), {"observation.state": th.tensor([start + i])}, 0.0, False, i == n)


@pytest.mark.parametrize("n", [3, 4, 5, 23])
def test_async_shard_hands_the_writer_the_same_frames(n):
    out = {}
    for mode in (False, True):
        ds = _FakeLeRobotDataset()
        w = _shard_writer(mode, ds)
        _episode(w, n)
        idx0 = w.end(save=True)
        _episode(w, 7, start=100.0)  # a second episode in the same shard
        idx1 = w.end(save=True)
        w.close()
        out[mode] = (ds.events, idx0, idx1)
    assert out[False] == out[True]
    assert sum(e[0] == "frame" for e in out[True][0]) == n + 7 and out[True][1:] == (0, 1)


def test_async_shard_abort_drops_the_episode_and_the_next_one_is_intact():
    out = {}
    for mode in (False, True):
        ds = _FakeLeRobotDataset()
        w = _shard_writer(mode, ds)
        _episode(w, 13)
        w.abort()  # a rollback mid-episode
        assert not ds.has_pending_frames() and w.end(save=True) is None
        _episode(w, 9, start=50.0)
        idx = w.end(save=True)
        w.close()
        # the first episode's frames the writer got before the abort (with async_flush the not-yet-started segments are
        # dropped, possibly all of them) end at its clear
        clears = [i for i, e in enumerate(ds.events) if e[0] == "clear"]
        saved = ds.events[clears[-1] + 1:] if clears else ds.events
        out[mode] = (saved, idx, ds.num_episodes)
    assert out[False] == out[True] and out[True][1:] == (0, 1)


def test_async_shard_writer_error_reaches_the_stepping_thread():
    ds = _FakeLeRobotDataset(fail_at=2)
    w = _shard_writer(True, ds)
    with pytest.raises(RuntimeError, match="encoder crashed"):
        _episode(w, 30)
        w.end(save=True)
    w.abort()
    w.close()


def test_async_shard_error_in_the_last_segment_drops_the_episode_and_closes_the_shard():
    """The last segment (submitted by the last step) fails: end() reports it, as the recorder's close() expects, after
    dropping the episode and finalizing the shard (what the synchronous path ends with after its failed step)."""
    ds = _FakeLeRobotDataset(fail_at=5)  # the 6th frame: in the segment flushed by step 8, the last one
    w = _shard_writer(True, ds)
    _episode(w, 8)
    with pytest.raises(RuntimeError, match="encoder crashed"):
        w.end(save=True)
    assert ds.num_episodes == 0 and not ds.has_pending_frames() and ds.events[-1] == ("finalize",)
    assert w._bg_flush is None and w.dataset is None


def test_async_frames_without_frame_sinks_takes_the_synchronous_path(tmp_path):
    """``--record hdf5`` (production): no video or LeRobot sink, so no frame is encoded; last_rgb as without the flag."""
    r0, rec0, _ = _run(tmp_path / "sync", per_scene=False, async_frames=False)
    r1, rec1, _ = _run(tmp_path / "async", per_scene=False, async_frames=True)
    assert rec1._pipe is not None and rec1._step_frames({}) is None
    assert r0["n_steps"] == r1["n_steps"] and r0["success"] == r1["success"]
    assert np.array_equal(rec0.last_rgb, rec1.last_rgb)


# -- physics-only collection and the recorded BEHAVIOR proprioception ------------------------------------------------


def _stub_behavior_proprio(rec):
    """The fake robot has no BEHAVIOR proprioception: its joint positions stand in (= state[:28] of the fake's state)."""
    rec._behavior_proprio_now = lambda: rec.robot.q.detach().cpu().float().numpy().copy()


def _datasets(g, prefix=""):
    h5py = pytest.importorskip("h5py")
    out = {}
    for k, v in g.items():
        if isinstance(v, h5py.Group):
            out.update(_datasets(v, f"{prefix}{k}/"))
        else:
            out[prefix + k] = v[()]
    return out


def _bitwise(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def test_behavior_proprio_rows_pair_with_the_recorded_states(tmp_path):
    """Row i of isg/behavior_proprio is the proprioception of state[i] (before action[i]), across the committed episode
    and the kept rolled-back segment; without the flag the file is unchanged."""
    h5py = pytest.importorskip("h5py")
    r1, _, p1 = _run(tmp_path / "on", per_scene=False, async_frames=False, capture=_stub_behavior_proprio, rcfg_kw={"behavior_proprio": True})
    r0, _, p0 = _run(tmp_path / "off", per_scene=False, async_frames=False, rcfg_kw={"behavior_proprio": False})
    assert r1["success"] and r0["n_steps"] == r1["n_steps"]
    with h5py.File(p1["hdf5"], "r") as f1, h5py.File(p0["hdf5"], "r") as f0:
        demo = f1["data/demo_0"]
        T = int(demo.attrs["num_samples"])
        bp = demo["isg/behavior_proprio"][()]
        assert bp.shape == (T, 28) and bp.dtype == np.float32
        assert np.array_equal(bp, demo["state"][()][:, :28])
        assert len(demo["rollbacks"]) > 0
        for k, g in demo["rollbacks"].items():
            assert np.array_equal(g["isg/behavior_proprio"][()], g["state"][()][:, :28]), k
        d1, d0 = _datasets(demo), _datasets(f0["data/demo_0"])
        assert {k for k in d1 if k not in d0} == {k for k in d1 if k.endswith("isg/behavior_proprio")}
        for k in d0:
            assert _bitwise(d0[k], d1[k]), k


def test_behavior_proprio_rollback_to_a_mark_resumes_from_the_kept_step(tmp_path):
    """A rollback to a mark inside the pending segment (a deferred-commit group) resumes from the proprioception recorded
    after the last kept step; one to an empty segment from the committed state's."""
    h5py = pytest.importorskip("h5py")
    sc, env = iors_world()
    rec = Recorder(str(tmp_path), env, env.robot, "robot_r1:zed_link:Camera:0", dict(env.objects), fps=30,
                   cfg=RecorderConfig(hdf5=True, lerobot=False, behavior_proprio=True), instance_id=sc.instance_id)
    _stub_behavior_proprio(rec)
    rec.start(env.get_obs()[0], {}, state0=env.sim.dump_state(serialized=True))
    step = [0]

    def advance(dq):
        env.robot.q = env.robot.q + dq
        rec.record(step[0], th.zeros(23), env.get_obs()[0], {}, state=env.sim.dump_state(serialized=True))
        step[0] += 1

    for _ in range(3):
        advance(0.01)
    cp = env.sim.dump_state(serialized=True)  # the state after step 2 of the segment
    m = rec.mark()
    for _ in range(2):
        advance(0.02)
    env.sim.load_state(cp)
    assert rec.rollback(1, 0, {"category": "test"}, state_before=cp, mark=m) == 2
    advance(0.03)
    rec.commit()
    cp = env.sim.dump_state(serialized=True)
    advance(0.04)
    env.sim.load_state(cp)
    assert rec.rollback(2, 0, {"category": "test"}, state_before=cp, mark=rec.mark() - 1) == 1
    advance(0.05)
    paths = rec.close({"instance_id": sc.instance_id, "success": True}, success=True)
    with h5py.File(paths["hdf5"], "r") as f:
        demo = f["data/demo_0"]
        assert int(demo.attrs["num_samples"]) == 5
        assert np.array_equal(demo["isg/behavior_proprio"][()], demo["state"][()][:, :28])
        for k, g in demo["rollbacks"].items():
            assert np.array_equal(g["isg/behavior_proprio"][()], g["state"][()][:, :28]), k


def test_behavior_proprio_is_the_writers_r1pro_layout_whatever_the_robot_class():
    """ispatialgym's robot class is not named R1Pro: the layout is BEHAVIOR's R1Pro one (the shard writer's), as float32."""
    from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES

    from ispatialgym.oracle import physics_only as PO

    idx = PROPRIOCEPTION_INDICES["R1Pro"]
    n = max(s.stop for s in idx.values())
    ref = th.arange(n, dtype=th.float64) / 7.0

    class Robot:
        def _get_proprioception_dict(self):
            return {**{k: ref[s].clone() for k, s in reversed(list(idx.items()))}, "joint_qpos": th.zeros(28, dtype=th.float64)}

    p = PO.behavior_proprio(Robot())
    assert p.dtype == np.float32 and p.shape == (n,)
    assert _bitwise(p, ref.float().numpy())


def test_physics_only_recorder_refuses_frame_sinks_and_keeps_no_frame(tmp_path):
    sc, env = iors_world()
    for kw in ({"views": ("head",)}, {"lerobot": True}, {"behavior_shard": True}):
        cfg = RecorderConfig(**{"hdf5": True, "lerobot": False, "physics_only": True, **kw})
        with pytest.raises(ValueError, match="records demo.hdf5 only"):
            Recorder(str(tmp_path), env, env.robot, "robot_r1:zed_link:Camera:0", dict(env.objects), fps=30, cfg=cfg, instance_id=sc.instance_id)


def test_physics_only_executor_records_the_same_hdf5_without_restore_renders(tmp_path, monkeypatch):
    """The fake's physics does not depend on rendering, so the physics-only episode must record the same demo.hdf5;
    it runs inside physics_only.episode, a rollback renders nothing (transforms synced after the load instead) and no
    frame is kept."""
    h5py = pytest.importorskip("h5py")
    from ispatialgym.oracle import physics_only as PO

    calls = {"episode": 0, "sync": 0}

    class _Episode:
        def __init__(self, sim):
            calls["episode"] += 1
            self.sim = sim

        def __enter__(self):
            calls["renders_before"] = self.sim.renders

        def __exit__(self, *exc):
            calls["renders_in"] = self.sim.renders - calls["renders_before"]
            return False

    monkeypatch.setattr(PO, "episode", _Episode)
    monkeypatch.setattr(PO, "sync_transforms", lambda sim=None: calls.__setitem__("sync", calls["sync"] + 1))
    r0, rec0, p0 = _run(tmp_path / "rendered", per_scene=False, async_frames=False, capture=_stub_behavior_proprio,
                        rcfg_kw={"behavior_proprio": True})
    assert calls["episode"] == 0 and calls["sync"] == 0
    r1, rec1, p1 = _run(tmp_path / "physics_only", per_scene=False, async_frames=False, capture=_stub_behavior_proprio,
                        rcfg_kw={"physics_only": True, "behavior_proprio": True}, ecfg_kw={"physics_only": True})
    assert calls["episode"] == 1 and calls["renders_in"] == 0 and calls["sync"] == len(rec1.rollbacks) >= 1
    assert r0["success"] == r1["success"] and r0["n_steps"] == r1["n_steps"]
    assert rec0.last_rgb is not None and rec1.last_rgb is None
    assert [c["restore_renders"]["n"] for c in r1.get("checkpoint_reports", [])] in ([], [0] * len(r1.get("checkpoint_reports", [])))
    with h5py.File(p0["hdf5"], "r") as f0, h5py.File(p1["hdf5"], "r") as f1:
        d0, d1 = _datasets(f0["data/demo_0"]), _datasets(f1["data/demo_0"])
        assert d0.keys() == d1.keys() and "isg/behavior_proprio" in d0
        for k in d0:
            assert _bitwise(d0[k], d1[k]), k


def test_scene_file_saved_away_from_the_origin_loads_in_the_scene_frame():
    """Behavior-1K Scene._scene_file_in_scene_frame: a scene file saved while its scene sat at x = 63.185 m (env 1 of a
    vector environment) is loaded as the same file saved at the origin; a file saved at the origin loads unchanged."""
    import copy
    import json

    import omnigibson.utils.transform_utils as T
    from omnigibson.scenes.scene_base import Scene

    def file_at(scene_pos):
        return {"init_info": {}, "objects_info": {"init_info": {}}, "metadata": {"task": {}}, "state": {
            "pos": list(scene_pos), "ori": [0.0, 0.0, 0.0, 1.0],
            "registry": {
                "object_registry": {
                    "table": {"is_asleep": True, "root_link": {"pos": [scene_pos[0] + 1.25, 2.5, 0.75], "ori": [0.0, 0.0, 0.38268343, 0.92387953],
                                                                "lin_vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]},
                              "non_kin": {"AABB": {"aabb": [1.0, 2.0, 0.0, 2.0, 3.0, 1.0]}}},
                    "robot_r1": {"root_link": {"pos": [scene_pos[0] - 0.5, 0.0, 0.0], "ori": [0.0, 0.0, 0.0, 1.0],
                                               "lin_vel": [0.1, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.2]}, "joint_pos": [0.1, 0.2]},
                },
                "system_registry": {"water": {"n_particles": 0, "particle_velocities": []}},
            }}}

    assert Scene._scene_file_in_scene_frame(file_at((0.0, 0.0, 0.0))) is None
    away = file_at((63.185, 0.0, 0.0))
    before = copy.deepcopy(away)
    local = Scene._scene_file_in_scene_frame(away)
    assert away == before
    assert local["state"]["pos"] == [0.0, 0.0, 0.0] and local["state"]["ori"] == [0.0, 0.0, 0.0, 1.0]
    objs = local["state"]["registry"]["object_registry"]
    np.testing.assert_allclose(objs["table"]["root_link"]["pos"], [1.25, 2.5, 0.75], atol=1e-5)
    np.testing.assert_allclose(objs["robot_r1"]["root_link"]["pos"], [-0.5, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(objs["table"]["root_link"]["ori"], [0.0, 0.0, 0.38268343, 0.92387953], atol=1e-6)
    assert objs["robot_r1"]["root_link"]["lin_vel"] == [pytest.approx(0.1), 0.0, 0.0] and objs["robot_r1"]["joint_pos"] == [0.1, 0.2]
    assert objs["table"]["non_kin"] == before["state"]["registry"]["object_registry"]["table"]["non_kin"]
    # the loaded file is plain JSON (the scene prebuild writes it to disk); a later restore of the saved file rebases the
    # world-frame poses by the same inverse transform, so it lands on these positions bit for bit
    json.dumps(local)
    inv = T.invert_pose_transform(th.tensor([63.185, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    again = Scene._rebase_registry_state_poses(before["state"]["registry"], *inv)
    assert th.equal(th.as_tensor(objs["table"]["root_link"]["pos"], dtype=th.float32), again["object_registry"]["table"]["root_link"]["pos"])


def test_lazy_frames_keep_the_same_last_rgb(tmp_path):
    r0, rec0, _ = _run(tmp_path / "eager", per_scene=False, async_frames=False, rcfg_kw={"lazy_frames": False})
    r1, rec1, _ = _run(tmp_path / "lazy", per_scene=False, async_frames=False, rcfg_kw={"lazy_frames": True})
    assert r0["n_steps"] == r1["n_steps"] and isinstance(rec0.last_rgb, np.ndarray)
    assert np.array_equal(rec0.last_rgb, rec1.last_rgb)
