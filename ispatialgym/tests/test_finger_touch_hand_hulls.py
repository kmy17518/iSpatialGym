"""i7-ft2 2026-11-08 (owner decision 7.1): the FingerTouch geometry rule judges the hand's rigid links (palm / wrist camera / wrist) as the
support policy's hull fallback does (``contact_policy.hull_fallback``, i7-plateqtf2): the padded spheres are the screen, a NARROW sphere
failure passes on the exact hulls with 0.5 mm of real clearance, and a sphere pass also needs the hulls clear (P1 §3).
``ISPATIALGYM_FINGER_TOUCH_HAND_HULLS`` = 0 / shadow / 1 (the default since the A/B).

The toy kit is test_finger_touch_geometry's (one configuration 4 cm before the grasp, fully open: it collides with the counter); a fake
``HandChecker`` reports fixed hull clearances per role.
"""

from __future__ import annotations

import logging

import pytest
import torch as th

from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.tests.test_finger_touch_geometry import COUNTER, _checker, _FakeHands, _kit, _q_at

S04 = 0.04
def _palm_bottom(depth_m: float) -> float:
    """The palm's sphere bottom below the eef that puts its spheres ``depth_m`` into the counter 4 cm before the grasp."""
    z = float(_q_at(S04)[7:10][2])
    return -(z + depth_m)


class _Hands(_FakeHands):
    """The finger hull a fixed clearance (clear), the hand roles their own fixed clearances."""

    def __init__(self, palm: float, finger: float = 0.003):
        super().__init__(fixed=finger)
        self.palm = float(palm)

    def clearance(self, T, support, finger_q=(0.05, 0.05)):
        out = super().clearance(T, support, finger_q)
        for r in ("palm", "wrist_camera", "wrist"):
            out[r] = th.tensor([self.palm if r == "palm" else 0.05])
        return out


def _judge(monkeypatch, mode, palm_sphere_depth, palm_hull, arm_bottom=0.09):
    if mode is None:
        monkeypatch.delenv(IK.FINGER_TOUCH_HAND_HULLS_ENV, raising=False)
    else:
        monkeypatch.setenv(IK.FINGER_TOUCH_HAND_HULLS_ENV, mode)
    kit = _kit(palm_bottom=_palm_bottom(palm_sphere_depth))
    if arm_bottom != 0.09:
        kit.bottoms["right_arm_link6"] = arm_bottom
    ikc = _checker(kit, "geometry", _Hands(palm_hull))
    ok, _ = ikc._finger_touch_ok(_q_at(S04)[None], "right", [], [kit.obstacles[0]])
    return bool(ok[0]), kit.stats, ikc


def test_off_keeps_the_sphere_verdict_and_the_default_is_on(monkeypatch):
    ok, st, ikc = _judge(monkeypatch, "0", 0.001, 0.001)
    assert IK.finger_touch_hand_hulls_mode() == "off" and not ok
    assert not any(k.startswith("finger_touch.hand_hulls") for k in st)
    assert "hand_hulls" not in ikc.cache_scope["finger_touch"]
    ok, st, ikc = _judge(monkeypatch, None, 0.001, 0.001)  # unset: the default since the A/B (owner 7.1)
    assert IK.finger_touch_hand_hulls_mode() == "on" and ok and "hand_hulls" in ikc.cache_scope["finger_touch"]


def test_narrow_sphere_failure_passes_on_the_hull(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        ok, st, ikc = _judge(monkeypatch, "1", 0.001, 0.001)  # spheres 1 mm in (< the 2 mm buffer), hull 1.0 mm clear (>= 0.5)
    assert ok
    assert st["finger_touch.hand_hulls.fallback_passed"] == 1 and st["finger_touch.hand_hulls.verdict_to_pass"] == 1
    assert st["finger_touch.hand_hulls.decision_to_pass"] == 1
    assert ikc.cache_scope["finger_touch"][-3:] == ["hand_hulls", IK.HULL_FALLBACK_MIN_CLEARANCE_M, 0.0]
    msg = [r.getMessage() for r in caplog.records if "hand hulls (on)" in r.getMessage()]
    assert msg and "hand hull clearance 1.0 mm (needs 0.5 mm)" in msg[0] and COUNTER in msg[0], msg


def test_the_fallback_threshold_and_the_buffer(monkeypatch):
    ok, st, _ = _judge(monkeypatch, "1", 0.001, 0.0003)  # hull 0.3 mm < 0.5 mm
    assert not ok and st["finger_touch.hand_hulls.fallback_failed.hull_clearance"] == 1
    ok, st, _ = _judge(monkeypatch, "1", 0.003, 0.001)  # spheres 3 mm in: deeper than the buffer, the hull is not asked
    assert not ok and st["finger_touch.hand_hulls.fallback_failed.deeper_than_buffer"] == 1


def test_a_non_hand_link_keeps_the_rejection(monkeypatch):
    z = float(_q_at(S04)[7:10][2])
    ok, st, _ = _judge(monkeypatch, "1", 0.001, 0.001, arm_bottom=-(z + 0.0005))  # the forearm's spheres 0.5 mm into the counter too
    assert not ok and st["finger_touch.hand_hulls.fallback_failed.non_hand_link"] == 1


def test_a_sphere_pass_needs_the_spec_hull(monkeypatch):
    ok, st, _ = _judge(monkeypatch, "1", -0.004, -0.0005)  # spheres 4 mm clear, the palm hull 0.5 mm into the counter (P1 §3)
    assert not ok
    assert st["finger_touch.hand_hulls.spec_rejected"] == 1 and st["finger_touch.hand_hulls.verdict_to_fail"] == 1
    assert st["finger_touch.hand_hulls.decision_to_fail"] == 1
    ok, st, _ = _judge(monkeypatch, "1", -0.004, 0.0)  # hull exactly at the margin: passes
    assert ok and st["finger_touch.hand_hulls.spec_checked"] == 1 and "finger_touch.hand_hulls.spec_rejected" not in st


def test_shadow_counts_without_changing_the_verdict(monkeypatch):
    ok, st, ikc = _judge(monkeypatch, "shadow", 0.001, 0.001)
    assert not ok, "shadow: the sphere verdict stands"
    assert st["finger_touch.hand_hulls.verdict_to_pass"] == 1 and st["finger_touch.hand_hulls.decision_to_pass"] == 1
    assert "hand_hulls" not in ikc.cache_scope["finger_touch"]
    ok, st, _ = _judge(monkeypatch, "shadow", -0.004, -0.0005)
    assert ok and st["finger_touch.hand_hulls.verdict_to_fail"] == 1


@pytest.mark.parametrize("v,want", [("", "on"), ("0", "off"), ("off", "off"), ("1", "on"), ("on", "on"), ("shadow", "shadow")])
def test_mode_values(monkeypatch, v, want):
    monkeypatch.setenv(IK.FINGER_TOUCH_HAND_HULLS_ENV, v)
    assert IK.finger_touch_hand_hulls_mode() == want
