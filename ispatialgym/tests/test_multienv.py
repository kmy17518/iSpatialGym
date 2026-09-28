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


def _run(tmp_path, per_scene: bool, async_frames: bool, video: bool = False, capture=None):
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
    rcfg = RecorderConfig(hdf5=True, lerobot=False, video=video, async_frames=async_frames)
    ecfg = ExecutorConfig(settle_min_steps=50, settle_fraction=0.2, per_scene_state=per_scene)
    state0 = env.scene.dump_state(serialized=True) if per_scene else env.sim.dump_state(serialized=True)
    ex, motions, backend, rec = make_executor(env, sc, None, exec_cfg=ecfg)
    rec = Recorder(str(tmp_path), env, env.robot, "robot_r1:zed_link:Camera:0", dict(env.objects), fps=30, cfg=rcfg, instance_id=sc.instance_id)
    if capture is not None:
        capture(rec)
    rec.start(env.get_obs()[0], {}, state0=state0)
    ex.recorder = rec
    result = ex.run_program(expand(sc.alias, sc))
    paths = rec.close({"instance_id": sc.instance_id, "success": bool(result["success"])}, qa=result["qa"]["stats"], success=bool(result["success"]))
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
            orig_video = rec._write_video_frame

            def video(frame, external=False):
                store["video"].append(np.array(frame, copy=True))
                return orig_video(frame, external=external)

            rec._write_video_frame = video

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
