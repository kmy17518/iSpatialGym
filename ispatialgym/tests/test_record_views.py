"""``run_isg --record-view`` (owner 2026-10-14): the recorder writes one ``<view>_rgb.mp4`` per requested view - the head
camera, BEHAVIOR's external camera, the two wrist cameras, and BEHAVIOR's JoyLo replay tiling of the four (left wrist | head |
right wrist over the external camera) - each with the frames of the committed steps only."""

from __future__ import annotations

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.recorder import TILE_PX, VIEWS, Recorder, RecorderConfig, tiled_frame
from ispatialgym.tests.test_v2_oracle import CAM_KEY, FakeEnv, FakeRobot

imageio = pytest.importorskip("imageio.v2")

LEFT, RIGHT, EXTERNAL = "robot_r1:left_realsense_link:Camera:0", "robot_r1:right_realsense_link:Camera:0", "external_sensor0"
COLORS = {"left_wrist": (0, 200, 0), "head": (200, 0, 0), "right_wrist": (0, 0, 200), "external": (200, 200, 0)}


def _img(color, h=16, w=16) -> th.Tensor:
    return th.tensor(color, dtype=th.uint8).expand(h, w, 3).clone()


def test_tiled_frame_is_behaviors_joylo_replay_layout():
    T = TILE_PX
    f = tiled_frame(*(np.asarray(_img(COLORS[v])) for v in ("left_wrist", "head", "right_wrist")), np.asarray(_img(COLORS["external"], 32, 48)))
    assert f.shape == (3 * T, 3 * T, 3) and f.dtype == np.uint8
    assert [tuple(f[T // 2, x]) for x in (T // 2, 3 * T // 2, 5 * T // 2)] == [COLORS["left_wrist"], COLORS["head"], COLORS["right_wrist"]]
    assert tuple(f[2 * T, 3 * T // 2]) == COLORS["external"] and tuple(f[3 * T - 1, 0]) == COLORS["external"]
    g = tiled_frame(None, np.asarray(_img(COLORS["head"])), None, None)  # a missing view is black
    assert tuple(g[T // 2, T // 2]) == (0, 0, 0) and tuple(g[2 * T, T]) == (0, 0, 0) and tuple(g[T // 2, 3 * T // 2]) == COLORS["head"]


class ViewEnv(FakeEnv):
    """The fake world whose observation also has the wrist cameras (unless ``wrists=False``) and BEHAVIOR's external camera."""

    def __init__(self, robot, wrists=True):
        super().__init__(robot, {})
        self.wrists = wrists

    def _obs(self):
        obs = super()._obs()
        if self.wrists:
            obs[self.robot.name][LEFT] = {"rgb": _img(COLORS["left_wrist"])}
            obs[self.robot.name][RIGHT] = {"rgb": _img(COLORS["right_wrist"])}
        obs["external"] = {EXTERNAL: {"rgb": _img(COLORS["external"], 32, 48)}}
        return obs


def _hold(robot) -> th.Tensor:
    a = th.zeros(M.ACTION_DIM)
    idx = robot.controller_action_idx
    a[idx["trunk"]] = robot.q[robot.trunk_control_idx]
    for arm in M.ARMS:
        a[idx[f"arm_{arm}"]] = robot.q[robot.arm_control_idx[arm]]
    return a


def _record(tmp_path, views, wrists=True):
    """Five recorded steps: three committed, two rolled back."""
    robot = FakeRobot()
    env = ViewEnv(robot, wrists=wrists)
    rec = Recorder(str(tmp_path), env, robot, CAM_KEY, {}, fps=30, cfg=RecorderConfig(hdf5=False, lerobot=False, views=views, external_camera=EXTERNAL,
                                                                                        video_size=96, frame_buffer="raw"))
    rec.start(env._obs(), {})
    for i in range(5):
        a = _hold(robot)
        obs, _r, _t, _tr, info = env.step(a)
        rec.record(i + 1, a, obs, info)
        if i == 2:
            rec.commit()
    rec.rollback(4, 0)
    return rec.close({"instance_id": "views"})


def _frames(path):
    return [np.asarray(f) for f in imageio.mimread(path)]


def test_every_view_is_its_own_video_of_the_committed_steps(tmp_path):
    paths = _record(tmp_path, VIEWS)
    assert set(paths["views"]) == set(VIEWS) and "view_errors" not in paths and paths["video"] == paths["views"]["head"]
    shape = {"head": (96, 96), "left_wrist": (96, 96), "right_wrist": (96, 96), "tiled": (96, 96), "external": (64, 96)}  # --video-size: the width
    for view in VIEWS:
        assert paths["views"][view] == str(tmp_path / f"{view}_rgb.mp4")
        frames = _frames(paths["views"][view])
        assert len(frames) == 3 and frames[0].shape[:2] == shape[view], view
    near = lambda px, color: np.abs(px.astype(int) - np.array(color)).max() <= 30  # noqa: E731  (lossy video)
    assert near(_frames(paths["views"]["external"])[0][32, 48], COLORS["external"])
    tiled = _frames(paths["views"]["tiled"])[0]
    assert all(near(tiled[16, x], COLORS[v]) for x, v in ((16, "left_wrist"), (80, "right_wrist"))) and near(tiled[72, 48], COLORS["external"])


def test_a_missing_camera_leaves_its_view_out_and_its_tile_black(tmp_path):
    paths = _record(tmp_path, ("tiled", "left_wrist"), wrists=False)
    assert set(paths["views"]) == {"tiled"} and paths["view_errors"] == {"left_wrist": "no camera"} and paths["video"] is None
    tiled = _frames(paths["views"]["tiled"])[0]
    assert tiled[16, 16].max() <= 30 and np.abs(tiled[72, 48].astype(int) - np.array(COLORS["external"])).max() <= 30


def test_an_unknown_view_is_refused(tmp_path):
    robot = FakeRobot()
    with pytest.raises(ValueError, match="unknown record views"):
        Recorder(str(tmp_path), FakeEnv(robot, {}), robot, CAM_KEY, {}, cfg=RecorderConfig(hdf5=False, lerobot=False, views=("top",)))
