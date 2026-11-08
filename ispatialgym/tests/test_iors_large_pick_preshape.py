"""i7-interior2 2026-11-08 (owner 6.1-6.3): IORS large's pick measures the support policy at the grasp's pre-shape (as
tabletop.grasp_at_pose), the large overhang fallback / the stance-C map cap / the straightened inside carry switches (default off).
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


def test_the_new_switches_default_off(monkeypatch):
    from ispatialgym.sampling.v2.tasks import iors_large as L
    from ispatialgym.sampling.v2.tasks import iors_large_inside as LI

    for env in (L.LARGE_OVERHANG_FALLBACK_ENV, L.INSIDE_CARRY_STRAIGHT_ENV, LI.INSIDE_MAP_CAP_ENV):
        monkeypatch.delenv(env, raising=False)
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
