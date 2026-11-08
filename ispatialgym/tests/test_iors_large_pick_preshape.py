"""i7-interior2 2026-11-08 (owner 6.1-6.3): IORS large's pick measures the support policy at the grasp's pre-shape (as
tabletop.grasp_at_pose), the large overhang fallback / the stance-C map cap (default on: owner 6.1 / 6.2) and the straightened inside carry (default on: owner 2026-11-08b) switches.
Simulator-free."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch as th

from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.ik import IKResult


class _IKC:
    def __init__(self):
        self._policy_opening = None
        self.seen = []

    @contextmanager
    def policy_opening(self, opening_m):
        prev = self._policy_opening
        self._policy_opening = opening_m
        try:
            yield
        finally:
            self._policy_opening = prev

    def exact(self, poses, stance, arm, **kw):
        self.seen.append(self._policy_opening)
        return IKResult(False, None, "support_policy: finger_penetration on bar_egwapq_0", 1)


def _sampler():
    from ispatialgym.sampling.v2.tasks.iors_large import IORSLarge

    s = SimpleNamespace(ikc=_IKC(), kit=SimpleNamespace(stats={}), gp=SimpleNamespace(variant="v"),
                        choice=SimpleNamespace(models={"target": SimpleNamespace(scale=None)}))
    s.grasp_ignore_objects = lambda: []
    return s, IORSLarge


def test_the_segment_fallback_pick_is_policed_at_the_pre_shape():
    s, IL = _sampler()
    g = SimpleNamespace(name="lip_shallow_22", pose=(th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), meta={"width": 0.00574, "mode": "top"})
    assert IL._regrasp_at_b(s, g, (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), None, "right", 0.89, False, {}) is None
    want = TT.policy_opening_of(g)
    assert want is not None and want < 0.02 and s.ikc.seen == [want]  # the plate's lip pinch: ~1.3 cm per finger, not fully open (0.05)
    assert s.ikc._policy_opening is None  # restored after the pick
    assert s.kit.stats["pick_preshape.segment_fallback"] == 1


def test_a_grasp_without_a_width_keeps_the_configuration_opening(monkeypatch):
    s, IL = _sampler()
    g = SimpleNamespace(name="g", pose=(th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), meta={})
    IL._regrasp_at_b(s, g, (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), None, "right", 0.89, False, {})
    assert s.ikc.seen == [None] and s.kit.stats["pick_preshape.segment_fallback.q"] == 1
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY_OPENING", "q")  # the existing switch turns the pre-shape off everywhere, this path too
    s2, _ = _sampler()
    g2 = SimpleNamespace(name="g", pose=g.pose, meta={"width": 0.006})
    IL._regrasp_at_b(s2, g2, (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), None, "right", 0.89, False, {})
    assert s2.ikc.seen == [None]


def test_the_switch_defaults(monkeypatch):
    from ispatialgym.sampling.v2.tasks import iors_large as L
    from ispatialgym.sampling.v2.tasks import iors_large_inside as LI

    for env in (L.LARGE_OVERHANG_FALLBACK_ENV, L.INSIDE_CARRY_STRAIGHT_ENV, LI.INSIDE_MAP_CAP_ENV):
        monkeypatch.delenv(env, raising=False)
    # the owner decided 6.1 (overhang fallback), 6.2 (map cap) and 2026-11-08b item 1 (the straightened inside carry): default on
    assert L.large_overhang_fallback_on() and L.inside_carry_straight_on() and LI.inside_map_cap_on()
    for env in (L.LARGE_OVERHANG_FALLBACK_ENV, L.INSIDE_CARRY_STRAIGHT_ENV, LI.INSIDE_MAP_CAP_ENV):
        monkeypatch.setenv(env, "0")
    assert not L.large_overhang_fallback_on() and not L.inside_carry_straight_on() and not LI.inside_map_cap_on()
    for env in (L.LARGE_OVERHANG_FALLBACK_ENV, L.INSIDE_CARRY_STRAIGHT_ENV, LI.INSIDE_MAP_CAP_ENV):
        monkeypatch.setenv(env, "1")
    assert L.large_overhang_fallback_on() and L.inside_carry_straight_on() and LI.inside_map_cap_on()


def test_the_map_cap_keeps_every_base_pose():
    from ispatialgym.sampling.v2.tasks.iors_large_inside import best_per_base

    sts = [SimpleNamespace(base_pose2d=(0.0, float(b), 0.0), extras={"reach_margin_m": m}) for b in range(4) for m in (0.1, 0.3, 0.2, 0.05, 0.25)]
    kept = best_per_base(sts, 3)
    assert len(kept) == 12 and {st.base_pose2d for st in kept} == {st.base_pose2d for st in sts}
    assert all(st.extras["reach_margin_m"] >= 0.2 for st in kept)  # the most reach margin per base pose


def test_the_arm_side_switch_and_the_camera_blame_are_opt_in(monkeypatch):
    """i7-interior2 owner items 2 / 3: ISPATIALGYM_INSIDE_ARM_SIDE (the fridge's place joint-2 limit; default on since the owner's 5t) and
    ISPATIALGYM_CAMERA_BLAME (the sink pick's wrist-camera diagnostic, off unless set); the blame only reports (camera_clear's verdict is unchanged)."""
    from ispatialgym.sampling.v2 import orient as O
    from ispatialgym.sampling.v2.tasks import iors_large_inside as LI

    monkeypatch.delenv(LI.INSIDE_ARM_SIDE_ENV, raising=False)
    assert LI.inside_arm_side_on()  # default on since the owner's 5t (relation "inside" only)
    monkeypatch.setenv(LI.INSIDE_ARM_SIDE_ENV, "0")
    assert not LI.inside_arm_side_on()
    monkeypatch.setenv(LI.INSIDE_ARM_SIDE_ENV, "1")
    assert LI.inside_arm_side_on() and LI.INSIDE_ARM_SIDE_AIM_M > 0
    calls = []
    kit = SimpleNamespace(links_clear=lambda q, links, m: calls.append(m) or th.tensor([m <= 0.0]), stats={},
                          fk=lambda links, q: {links[0]: (th.tensor([0.5, 0.0, 0.9]), th.tensor([0.0, 0.0, 0.0, 1.0]))}, world=None)
    ikc = SimpleNamespace(kit=kit)
    monkeypatch.delenv(O.CAMERA_BLAME_ENV, raising=False)
    assert O.camera_clear(ikc, th.zeros(28), "right") is False and calls == [O.CAMERA_MARGIN]
    monkeypatch.setenv(O.CAMERA_BLAME_ENV, "1")
    calls.clear()
    assert O.camera_clear(ikc, th.zeros(28), "right") is False  # the verdict stays
    assert kit.stats.get("camera_blame.clear_at.+0mm") == 1 and kit.stats.get("camera_blame.none") == 1
