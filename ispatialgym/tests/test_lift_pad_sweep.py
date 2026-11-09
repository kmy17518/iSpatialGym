"""i7-yield2 (2026-11-08): the grasp support lift is refused when the library's pad sweep meets no face of the object at the lifted pose.

disembedding_10 (tablefork pqrtkc on the source tray): the executor's 5.8 mm lift put the pads' fingertip-side end above the fork's neck
(the pad-margin rule read the AABB top 0.84 - the raised handle end; the neck tops out at 0.830): top_9 closed on nothing (failed_grasp,
retried unlifted), top_2 closed on one edge and the closure stood still into a [pause] (int2_7 300019 / 300020).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.grasping import candidates as CA
from ispatialgym.oracle import execute as EX
from ispatialgym.tests.test_v21_phase_a import _grasp_setup


def test_pad_footprint_matches_the_library_sweep():
    assert list(EX.LIFT_SWEEP_PAD_Z) == pytest.approx(list(CA.PAD_Z))
    assert list(EX.LIFT_SWEEP_PAD_X) == pytest.approx(list(CA.PAD_X))
    assert EX.LIFT_SWEEP_OPEN_HALF == pytest.approx(CA.PAD_OPEN_HALF)


def _lift_with(monkeypatch, sweep, on=True):
    action, state, targets = _grasp_setup()
    monkeypatch.setattr(EX, "_finger_low_z", lambda pos, quat, arm: 0.794)
    monkeypatch.setattr(EX, "_approach_tracks", lambda motions, arm, q_hint, poses: True)
    calls = []

    def probe(obj, pos, quat):
        calls.append(float(th.as_tensor(pos)[2]))
        return sweep

    monkeypatch.setattr(EX, "_pad_sweep_hits", probe)
    cfg = EX.ExecuteConfig()
    cfg.grasp_support_pad_sweep = on
    ctx = {"next_place_kind": "support"}
    out = EX.lift_grasp_off_support(action, None, state, targets, ctx, cfg)
    return out, targets, ctx, calls


def test_lift_refused_when_the_lifted_pads_meet_nothing(monkeypatch):
    out, targets, ctx, calls = _lift_with(monkeypatch, (0, 24))
    rec = ctx["grasp_support_lift"]
    assert out is targets and not rec["applied"] and "pad sweep" in rec["reason"] and rec["pad_sweep"] == {"hits": 0, "rays": 24}
    need = 0.80 + EX.ExecuteConfig().grasp_support_clearance_m - 0.794
    assert calls == [pytest.approx(targets["grasp"].position[2] + need, abs=1e-6)]  # probed at the LIFTED grasp


def test_lift_kept_when_the_pads_meet_the_object_or_the_sweep_is_not_judged(monkeypatch):
    out, targets, ctx, _ = _lift_with(monkeypatch, (3, 24))
    assert out is not targets and ctx["grasp_support_lift"]["applied"] and ctx["grasp_support_lift"]["pad_sweep"]["hits"] == 3
    out, targets, ctx, _ = _lift_with(monkeypatch, None)
    assert out is not targets and ctx["grasp_support_lift"]["applied"] and "pad_sweep" not in ctx["grasp_support_lift"]


def test_switch_off_runs_no_sweep(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_LIFT_PAD_SWEEP", raising=False)
    assert EX.ExecuteConfig().grasp_support_pad_sweep is True  # default on (i7-yield2 A/B, 2026-11-08)
    monkeypatch.setenv("ISPATIALGYM_LIFT_PAD_SWEEP", "0")
    assert EX.ExecuteConfig().grasp_support_pad_sweep is False
    out, targets, ctx, calls = _lift_with(monkeypatch, (0, 24), on=False)
    assert out is not targets and ctx["grasp_support_lift"]["applied"] and calls == []


def test_pad_sweep_hits_counts_rays_through_the_object(monkeypatch):
    """A 5 mm-thick slab (z 0..0.005) under a top-down pinch: the pads' fingertip-side samples (+10 mm along the approach) cross it when the
    eef stands 12 mm above it, none does 6 mm higher."""
    obj = SimpleNamespace(prim_path="/World/fork")
    quat = th.tensor([1.0, 0.0, 0.0, 0.0])  # 180 deg about x: approach (eef +z) points down, closing along -y

    def raycast(starts, ends):
        out = []
        for s, e in zip(starts, ends):
            z = 0.5 * (s[2] + e[2])
            assert s[2] == pytest.approx(e[2]) and abs(s[1] - e[1]) == pytest.approx(2 * EX.LIFT_SWEEP_OPEN_HALF)
            out.append([{"hit": True, "rigidBody": "/World/fork/base_link"}] if 0.0 <= z <= 0.005 else [{"hit": True, "rigidBody": "/World/tray/base_link"}])
        return out

    monkeypatch.setattr(EX, "_raycast_all", raycast)
    assert EX._pad_sweep_hits(obj, th.tensor([0.0, 0.0, 0.012]), quat) == (6, 24)  # dz = +0.01 -> z 0.002, 3 dx x 2 sides
    assert EX._pad_sweep_hits(obj, th.tensor([0.0, 0.0, 0.018]), quat) == (0, 24)
    assert EX._pad_sweep_hits(SimpleNamespace(), th.zeros(3), quat) is None
