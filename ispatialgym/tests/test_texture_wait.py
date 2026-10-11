"""``oracle.texture_wait`` without the simulator: renders until every camera image stops sharpening (owner 2026-10-11 11:46)."""

import numpy as np

from ispatialgym.oracle import texture_wait as TW


def _image(sharp: float, seed: int = 0) -> np.ndarray:
    """A 256 x 320 rgb image whose Laplacian variance grows with ``sharp`` (a blurred checkerboard)."""
    y, x = np.mgrid[0:256, 0:320]
    board = (((x // 4) + (y // 4)) % 2).astype(np.float32)
    blur = np.full_like(board, board.mean())
    img = blur + float(sharp) * (board - blur)
    return np.repeat((img * 255.0)[..., None], 3, axis=-1).astype(np.uint8)


def test_sharpness_rises_with_detail_and_the_crop_is_the_bottom_centre():
    assert TW.sharpness(_image(0.0)) < 1e-6 < TW.sharpness(_image(0.2)) < TW.sharpness(_image(1.0))
    c = TW.bottom_centre(np.zeros((480, 640, 3)), 240)
    assert c.shape == (240, 240, 3)


def test_wait_renders_until_the_images_stop_sharpening():
    levels = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]  # mip levels arriving over the first renders, then full resolution
    state = {"n": 0}

    def render():
        state["n"] += 1

    def get_obs():
        s = levels[min(state["n"], len(levels) - 1)]
        return {"robot0": {"zed": {"rgb": _image(s)}, "proprio": np.zeros(3)}, "external": {"ext0": {"rgb": _image(s)}}}

    status, obs = TW.wait_for_textures(get_obs, render, "robot0", external="ext0", max_s=30)
    assert status["complete"] and status["renders"] == len(levels) - 1 + TW.STABLE_FRAMES
    assert set(status["sharpness_capture"]) == {"zed", "external:ext0"}
    assert status["sharpness_capture"]["zed"][0] > 10 * status["sharpness_start"]["zed"][0]
    assert TW.sharpness(obs["robot0"]["zed"]["rgb"]) == TW.sharpness(_image(1.0))


def test_wait_reports_a_timeout_as_incomplete():
    import time

    state = {"n": 0}

    def render():  # 50 ms renders: the image is still sharpening (5 % per render, full detail after ~90) when 0.5 s are up
        time.sleep(0.05)
        state["n"] += 1

    status, _ = TW.wait_for_textures(lambda: {"robot0": {"zed": {"rgb": _image(min(1.0, 0.01 * 1.05 ** state["n"]))}}}, render, "robot0", max_s=0.5)
    assert not status["complete"] and status["why"].startswith("timeout")
