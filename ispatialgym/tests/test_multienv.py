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
