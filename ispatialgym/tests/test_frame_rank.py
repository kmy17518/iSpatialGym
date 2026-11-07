"""oracle-fix 2026-11-07 (owner prompt E): the eef box's frame margin agrees with the frame rule."""

from types import SimpleNamespace

import numpy as np
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.v2 import ik as IKm


def test_eef_frame_margin_sign_matches_eef_in_frame():
    cm = G.CameraModel(640, 480, 400.0, 400.0, 320.0, 240.0)
    fake = SimpleNamespace(kit=SimpleNamespace(world=SimpleNamespace(camera_model=cm)))
    cam = (th.tensor([0.0, 0.0, 1.5]), th.tensor([0.0, 0.0, 0.0, 1.0]))  # looking down -z
    rng = np.random.default_rng(0)
    n_in = n_out = 0
    for _ in range(300):
        p = th.tensor([rng.uniform(-1.2, 1.2), rng.uniform(-1.0, 1.0), rng.uniform(0.2, 1.2)], dtype=th.float32)
        q = th.tensor(rng.normal(size=4), dtype=th.float32)
        q = q / q.norm()
        m = IKm.IKChecker.eef_frame_margin(fake, (p, q), cam)
        inside = IKm.IKChecker.eef_in_frame(fake, (p, q), cam)
        assert inside == (m >= 0.0)
        n_in += inside
        n_out += not inside
    assert n_in > 20 and n_out > 20
    centre = IKm.IKChecker.eef_frame_margin(fake, (th.tensor([0.0, 0.0, 0.8]), th.tensor([0.0, 0.0, 0.0, 1.0])), cam)
    edge = IKm.IKChecker.eef_frame_margin(fake, (th.tensor([0.5, 0.0, 0.8]), th.tensor([0.0, 0.0, 0.0, 1.0])), cam)
    assert centre > edge  # deeper inside at the image centre
