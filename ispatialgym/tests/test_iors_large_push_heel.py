"""Simulator-free tests of the push-under's heel-first yaw lock (owner 2026-10-13 ~02:55, under_large#01: ``generation.push_under_yaw`` =
``"heel_first"``): the footprint's heel (the end of its long axis farther from the root), the yaws the lock keeps for a sandal-like hull - the
heel toward the robot within one yaw step, the contact on the heel's half - the drawn (jittered) yaw, the rest orientation, the spec checks
and the log line; nothing changes without the key (``pytest ispatialgym/tests/test_iors_large_push_heel.py``)."""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.v2.stances import Stance

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")

VARIANT = "inter_object_relation_satisfaction-under_large_scale-sandal-bed-floor_iorsl"
HEEL, TOE = 0.128, 0.117
YAWS = [k * 2.0 * math.pi / 16 for k in range(16)]
PUSH = (1.0, 0.0)  # the push along +x: the robot behind the target at -x


def _sandal() -> th.Tensor:
    """A sandal lying flat, the root at its sole: the long axis along x, the heel 12.8 cm behind the root (-x) with its face up to 3.5 cm, the
    toe 11.7 cm ahead up to 4.5 cm, the straight sides 10 cm apart, the strap 2.7 cm behind the root up to 7.2 cm (the pod probe of
    2026-09-20 and the 0.128 / 0.117 m contact offsets of the 27ee9069 runs)."""
    sole = [(-HEEL, 0.0), (-0.122, 0.03), (-0.122, -0.03), (-0.09, 0.05), (-0.09, -0.05), (0.07, 0.05), (0.07, -0.05), (0.105, 0.03), (0.105, -0.03), (TOE, 0.0)]
    pts = [(x, y, 0.0) for x, y in sole]
    pts += [(-HEEL, 0.0, 0.035), (-0.122, 0.03, 0.035), (-0.122, -0.03, 0.035)]  # the heel's face
    pts += [(TOE, 0.0, 0.045), (0.105, 0.03, 0.045), (0.105, -0.03, 0.045)]  # the toe
    pts += [(-0.027, 0.045, 0.072), (-0.027, -0.045, 0.072), (0.03, 0.045, 0.072), (0.03, -0.045, 0.072)]  # the strap
    return th.tensor(pts, dtype=th.float32)


class _GP:
    def __init__(self, **gen):
        self.variant, self.generation = VARIANT, dict(gen)

    def gen(self, key, default=None):
        return self.generation.get(key, default)


def _sampler(hull=None, rest=None, **gen) -> "IL.IORSLarge":
    """The sandal at the v3.0 block's fingertips (push_tip_frac 0.3: 2.16 cm up), the blade hand."""
    gen.setdefault("push_tip_frac", 0.3)
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.relation, s.template_facts = "under", {}
    s.gp = _GP(**gen)
    s.hull = _sandal() if hull is None else hull
    s.rest = G.IDENTITY_QUAT.clone() if rest is None else rest
    s.floor_z = 0.0
    s.place = {"height": 0.072, "half": HEEL}
    objs = {"target": SimpleNamespace(name="sandal_chgcrm_0")}
    s.obj = lambda role: objs[role]
    s.kit = SimpleNamespace(stats={})
    return s


def _place() -> Stance:
    return Stance("push", (0.0, 0.0, 0.0), 2.0, 0.0, stand_off=0.80, lateral=0.05)


def _q(s, yaw: float):
    return G.qmul(G.quat_from_yaw(yaw), s.rest)


def _offs(s, yaws, d_yaw: float = 0.0):
    """The chain-first screen's contact offsets (-1 when nothing lies in the band)."""
    out = {}
    for y in yaws:
        o = s.push_contact_offset(_q(s, y + d_yaw), PUSH)
        out[y] = -1.0 if o is None else o
    return out


def _chain_first(s, yaws, d_yaw: float = 0.0):
    """The chain-first screen: ``(its two yaws, the offsets, the yaws the lock keeps)`` - the kept yaws, the largest contact offset first."""
    offs = _offs(s, yaws, d_yaw)
    kept = s.push_yaws_locked(yaws, d_yaw, PUSH, offs, _place())
    return sorted(kept, key=lambda y: s.push_yaw_rank(None, _q(s, y + d_yaw), offs[y]))[:2], offs, kept


# ----------------------------------------------------------------------------------------------------------------
# the heel of a footprint
# ----------------------------------------------------------------------------------------------------------------


def test_footprint_heel_is_the_long_axis_end_farther_from_the_root():
    h, heel, toe = IL.footprint_heel(_sandal())
    assert h.tolist() == pytest.approx([-1.0, 0.0], abs=1e-6) and heel == pytest.approx(HEEL, abs=1e-6) and toe == pytest.approx(TOE, abs=1e-6)
    # turned by any yaw, the heel turns with it
    for yaw in (0.5, 1.9, -2.6):
        pts = G.local_points_world(_sandal(), (th.zeros(3), G.quat_from_yaw(yaw)))
        h, heel, toe = IL.footprint_heel(pts)
        assert float(h @ th.tensor([-math.cos(yaw), -math.sin(yaw)])) > math.cos(math.radians(1.0)), yaw
        assert heel == pytest.approx(HEEL, abs=2e-3) and toe == pytest.approx(TOE, abs=2e-3)
    # a box 0.30 x 0.10 with its root 2 cm off centre: the long side is the axis (not the diagonal), the heel its farther end
    box = th.tensor([[x, y, z] for x in (-0.17, 0.13) for y in (-0.05, 0.05) for z in (0.0, 0.04)])
    h, heel, toe = IL.footprint_heel(box)
    assert h.tolist() == pytest.approx([-1.0, 0.0], abs=1e-6) and heel == pytest.approx(0.17) and toe == pytest.approx(0.13)


# ----------------------------------------------------------------------------------------------------------------
# the keys
# ----------------------------------------------------------------------------------------------------------------


def test_the_keys():
    assert _sampler().push_yaw_lock() is None and _sampler(push_under_yaw="none").push_yaw_lock() is None
    assert _sampler(push_under_yaw="heel_first").push_yaw_lock() == "heel_first" and IL.PUSH_UNDER_YAW_LOCKS == ("heel_first", "broadside")  # i7-floorpush r2 added the broadside lock
    with pytest.raises(IL.Reject) as e:
        _sampler(push_under_yaw="toe_first").push_yaw_lock()
    assert e.value.reason == "spec"
    # the tolerance: one step of the yaw grid unless set
    assert _sampler().push_yaw_tol_deg() == pytest.approx(22.5) and _sampler(n_yaws=8).push_yaw_tol_deg() == pytest.approx(45.0)
    assert _sampler(push_under_yaw_tol_deg=10.0).push_yaw_tol_deg() == pytest.approx(10.0)


def test_without_the_key_every_yaw_is_kept(caplog):
    s = _sampler()
    s.hull = None  # no heel is looked for: the hull is not read by the lock
    assert all(s.push_yaw_kept(_q(s, y), PUSH, 0.05) for y in YAWS)
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        assert s.push_yaws_locked(YAWS, 0.0, PUSH, {}, _place()) == YAWS
    assert "push-under yaw lock" not in caplog.text and s.kit.stats == {} and s.template_facts == {}


def test_the_default_order_takes_the_toe_when_no_heel_first_yaw_is_offered():
    # c1_k1 at 27ee9069: F_init at B held toe-first yaws only - the largest offset left is the toe's (0.117 m), and the chain pushed it
    s = _sampler()
    offs = _offs(s, YAWS)
    assert offs[0.0] == pytest.approx(HEEL) and offs[math.pi] == pytest.approx(TOE)  # tips 2.16 cm: the heel's face is in the band
    toe_side = [y for y in YAWS if abs(math.remainder(y - math.pi, 2.0 * math.pi)) < math.radians(50.0)]
    best, offs_t, kept = _chain_first(s, toe_side)
    assert best[0] == pytest.approx(math.pi) and offs_t[best[0]] == pytest.approx(TOE) and kept == toe_side
    # every yaw offered: the heel first
    assert _chain_first(s, YAWS)[0][0] == 0.0


# ----------------------------------------------------------------------------------------------------------------
# heel_first
# ----------------------------------------------------------------------------------------------------------------


def test_heel_first_keeps_the_larger_half_extent_end_only(caplog):
    s = _sampler(push_under_yaw="heel_first")
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        best, offs, kept = _chain_first(s, YAWS)
    # the heel toward the robot (yaw 0) and the grid yaws one step either side, every one met at the heel (> 0.12 m behind the root)
    assert kept == pytest.approx([0.0, YAWS[1], YAWS[15]]) and all(offs[y] > 0.12 for y in kept)
    assert best[0] == 0.0 and offs[best[0]] == pytest.approx(HEEL)
    assert not s.push_yaw_kept(_q(s, math.pi), PUSH) and offs[math.pi] == pytest.approx(TOE)  # toe first: never
    assert not any(s.push_yaw_kept(_q(s, y), PUSH) for y in (YAWS[4], YAWS[12]))  # across the push: never
    assert "push-under yaw lock heel_first: kept 3 of 16 yaws at C (d 0.80 λ +0.05, carry yaw offset +0.00): [0.0, 0.39, 5.89], offsets [0.128, " in caplog.text
    assert s.kit.stats == {"push_yaw_lock.removed": 1}
    tf = s.template_facts["push_under_yaw"]
    assert tf["lock"] == "heel_first" and tf["heel_m"] == pytest.approx(HEEL) and tf["toe_m"] == pytest.approx(TOE) and tf["tol_deg"] == 22.5
    assert tf["heel_dir_local"] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-4)
    # the B / C base-yaw change turns the carried yaws: the heel-first yaw between two grid yaws keeps those two (11.25 deg off each)
    d_yaw = -math.radians(11.25)
    kept = s.push_yaws_locked(YAWS, d_yaw, PUSH, _offs(s, YAWS, d_yaw), _place())
    assert kept == pytest.approx([0.0, YAWS[1]])


def test_heel_first_with_no_heel_first_yaw_keeps_none(caplog):
    # c1_k1's B: F_init's yaws all toe-ish - the lock leaves nothing to chain (the C is skipped; the B pre-screen ranks such a B last)
    s = _sampler(push_under_yaw="heel_first")
    toe_side = [y for y in YAWS if abs(math.remainder(y - math.pi, 2.0 * math.pi)) < math.radians(50.0)]
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        best, _, kept = _chain_first(s, toe_side)
    assert best == [] == kept and "push-under yaw lock heel_first: kept 0 of 5 yaws" in caplog.text


def test_heel_first_needs_the_contact_on_the_heel():
    # the fingertips 3.6 cm up (push_tip_frac 0.5, before the v3.0 block): above the heel's 3.5 cm face the blade meets the strap 2.7 cm behind
    # the root at the heel-first yaw - not on the heel's half, so not kept; the default order took the toe (0.117 m) there
    s = _sampler(push_under_yaw="heel_first", push_tip_frac=0.5)
    offs = _offs(s, YAWS)
    assert offs[0.0] == pytest.approx(0.027) and offs[math.pi] == pytest.approx(TOE)
    assert s.push_yaws_locked(YAWS, 0.0, PUSH, offs, _place()) == []
    assert _chain_first(_sampler(push_tip_frac=0.5), YAWS)[0][0] == pytest.approx(math.pi)
    # the contact passed by the caller is what is judged
    s = _sampler(push_under_yaw="heel_first")
    assert s.push_yaw_kept(_q(s, 0.0), PUSH, 0.07) and not s.push_yaw_kept(_q(s, 0.0), PUSH, 0.06) and not s.push_yaw_kept(_q(s, 0.0), PUSH, -1.0)


def test_the_drawn_yaw_is_judged_at_its_own_angle():
    # the B IK's retries jitter the init yaw by up to 22.5 deg: the yaw the chain is recorded at is checked again
    s = _sampler(push_under_yaw="heel_first")
    assert s.push_yaw_kept(_q(s, YAWS[1]), PUSH) and s.push_yaw_kept(_q(s, -0.3), PUSH)
    assert not s.push_yaw_kept(_q(s, YAWS[1] + 0.2), PUSH)  # 34 deg off
    assert _sampler(push_under_yaw="heel_first", push_under_yaw_tol_deg=40.0).push_yaw_kept(_q(s, YAWS[1] + 0.2), PUSH)
    assert not _sampler(push_under_yaw="heel_first", push_under_yaw_tol_deg=10.0).push_yaw_kept(_q(s, YAWS[1]), PUSH)
    # another push direction: the heel against it
    d = (math.cos(2.0), math.sin(2.0))
    assert s.push_yaw_kept(_q(s, 2.0), d) and not s.push_yaw_kept(_q(s, 2.0 + math.pi), d)


def test_the_heel_follows_the_rest_orientation():
    # a hull whose heel lies along its local +y, stood at rest turned -90 deg about z: the heel points +x at yaw 0, so a push along +x meets
    # it at yaw pi
    hull = G.local_points_world(_sandal(), (th.zeros(3), G.quat_from_yaw(-math.pi / 2)))  # the heel at local +y
    rest = G.quat_from_yaw(-math.pi / 2)
    s = _sampler(hull=hull, rest=rest, push_under_yaw="heel_first")
    assert s.push_yaw_kept(_q(s, math.pi), PUSH) and not s.push_yaw_kept(_q(s, 0.0), PUSH)
    assert s.template_facts["push_under_yaw"]["heel_dir_local"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-4)


def test_a_target_without_a_heel_is_a_spec_error():
    # the root at the box's centre: both ends of the long axis 0.15 m away
    box = th.tensor([[x, y, z] for x in (-0.15, 0.15) for y in (-0.05, 0.05) for z in (0.0, 0.07)])
    s = _sampler(hull=box, push_under_yaw="heel_first")
    with pytest.raises(IL.Reject) as e:
        s.push_yaw_kept(_q(s, 0.0), PUSH, 0.15)
    assert e.value.reason == "spec" and "no heel" in str(e.value)
    # 4 mm apart is still none (PUSH_HEEL_MIN_ASYM_M 5 mm), 6 mm is one
    s = _sampler(hull=box + th.tensor([0.002, 0.0, 0.0]), push_under_yaw="heel_first")
    with pytest.raises(IL.Reject):
        s.push_heel()
    h, heel, toe = _sampler(hull=box + th.tensor([0.003, 0.0, 0.0]), push_under_yaw="heel_first").push_heel()
    assert heel == pytest.approx(0.153) and toe == pytest.approx(0.147) and h.tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


def test_the_heel_is_memoised_per_target():
    s = _sampler(push_under_yaw="heel_first")
    first = s.push_heel()
    s.hull = None  # not read again for the same target
    assert s.push_heel() is first
    s.obj = lambda role: SimpleNamespace(name="sandal_chgcrm_1")
    s.hull = _sandal()
    assert s.push_heel() is not first


# ----------------------------------------------------------------------------------------------------------------
# i7-ul01reach: ISPATIALGYM_PUSH_HEEL_TIGHT (default off) - heel_first's tolerance and the init-yaw jitter narrowed
# ----------------------------------------------------------------------------------------------------------------


def test_heel_tight_off_by_default(monkeypatch):
    monkeypatch.delenv(IL.PUSH_HEEL_TIGHT_ENV, raising=False)
    s = _sampler(push_under_yaw="heel_first")
    assert IL.push_heel_tight_tol_deg() is None and s.push_heel_tight() is None
    assert s.push_yaw_tol_deg() == pytest.approx(22.5) and s.init_yaw_jitter() == pytest.approx(IL.TARGET_JITTER)
    assert s.push_yaw_kept(_q(s, YAWS[15]), PUSH)  # today: the grid's -22.5 deg is kept next to 0
    assert "push_heel_tight.dropped" not in s.kit.stats and "push_heel_tight.jitter_capped" not in s.kit.stats
    for v in ("0", "off", ""):
        monkeypatch.setenv(IL.PUSH_HEEL_TIGHT_ENV, v)
        assert IL.push_heel_tight_tol_deg() is None


def test_heel_tight_drops_the_off_grid_neighbours_and_caps_the_jitter(monkeypatch):
    monkeypatch.setenv(IL.PUSH_HEEL_TIGHT_ENV, "1")
    s = _sampler(push_under_yaw="heel_first")
    assert s.push_yaw_tol_deg() == pytest.approx(22.5)  # today's tolerance; the tight window is applied inside it
    assert s.push_yaw_kept(_q(s, 0.0), PUSH) and s.push_yaw_kept(_q(s, math.radians(-4.0)), PUSH)
    assert not s.push_yaw_kept(_q(s, YAWS[15]), PUSH) and not s.push_yaw_kept(_q(s, -0.12), PUSH)
    assert s.kit.stats["push_heel_tight.dropped"] == 2
    assert not s.push_yaw_kept(_q(s, math.pi), PUSH) and s.kit.stats["push_heel_tight.dropped"] == 2  # the toe: today drops it too
    assert s.init_yaw_jitter() == pytest.approx(math.radians(IL.PUSH_HEEL_TIGHT_TOL_DEG) * IL.PUSH_HEEL_TIGHT_JITTER_FRAC)
    assert s.kit.stats["push_heel_tight.jitter_capped"] == 1
    assert s.template_facts["push_under_yaw"]["tight_tol_deg"] == pytest.approx(IL.PUSH_HEEL_TIGHT_TOL_DEG)
    assert abs(s.template_facts["push_under_yaw"]["heel_angle_rest_deg"]) == pytest.approx(180.0, abs=1e-3)
    # the chain-first screen keeps the heel-back grid yaw only
    assert _chain_first(s, YAWS)[2] == [0.0]
    # a number is the window in degrees; inside the spec's tolerance
    monkeypatch.setenv(IL.PUSH_HEEL_TIGHT_ENV, "8")
    s8 = _sampler(push_under_yaw="heel_first")
    assert s8.push_heel_tight() == pytest.approx(8.0) and s8.push_yaw_kept(_q(s8, math.radians(-7.0)), PUSH)
    s3 = _sampler(push_under_yaw="heel_first", push_under_yaw_tol_deg=3.0)
    assert not s3.push_yaw_kept(_q(s3, math.radians(-5.0)), PUSH)
    monkeypatch.setenv(IL.PUSH_HEEL_TIGHT_ENV, "-2")
    with pytest.raises(ValueError):
        IL.push_heel_tight_tol_deg()


def test_heel_tight_acts_only_on_the_heel_first_lock(monkeypatch):
    monkeypatch.setenv(IL.PUSH_HEEL_TIGHT_ENV, "1")
    s = _sampler()
    assert s.push_heel_tight() is None and s.push_yaw_tol_deg() == pytest.approx(22.5) and s.init_yaw_jitter() == pytest.approx(IL.TARGET_JITTER)
    assert s.kit.stats == {}


def test_heel_tight_window_is_around_the_grid_yaw_nearest_heel_back(monkeypatch):
    # uh_ul01_on r1: the sandal chgcrm's heel stands 6 deg off heel-back at yaw 0 - a window around heel-back kept none of the 16 yaws
    monkeypatch.setenv(IL.PUSH_HEEL_TIGHT_ENV, "1")
    pts = G.local_points_world(_sandal(), (th.zeros(3), G.quat_from_yaw(math.radians(6.0))))
    s = _sampler(hull=pts, push_under_yaw="heel_first")
    assert s.push_yaw_kept(_q(s, 0.0), PUSH) and s.push_yaw_kept(_q(s, math.radians(4.0)), PUSH) and s.push_yaw_kept(_q(s, math.radians(-4.0)), PUSH)
    assert not s.push_yaw_kept(_q(s, YAWS[15]), PUSH) and not s.push_yaw_kept(_q(s, math.radians(-8.0)), PUSH)
    assert _chain_first(s, YAWS)[2] == [0.0]
    assert abs(abs(s.template_facts["push_under_yaw"]["heel_angle_rest_deg"]) - 174.0) < 0.6
