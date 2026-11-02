"""Simulator-free tests of the grasp reach margin at stance B (owner 2026-10-13 ~03:15 item 1, under_large #02 pitched copy 146:
``generation.grasp_reach_margin_m``): a grasp is kept only when its chain also solves that much deeper along the approach, its finger
assignments filtered in order, the segment fallback's regrasp included; nothing changes without the key
(``pytest ispatialgym/tests/test_iors_large_grasp_margin.py``)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2.stances import Stance

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")

VARIANT = "inter_object_relation_satisfaction-under_large_scale-briefcase-desk_mdhelw_0-floor_pitched"
FLOOR = SimpleNamespace(name="floors_jluwuz_0")
DOWN = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0]))  # top-down: the approach axis -z


class _GP:
    def __init__(self, **gen):
        self.variant, self.generation = VARIANT, dict(gen)

    def gen(self, key, default=None):
        return self.generation.get(key, default)


class _IKC:
    """The exact IK as a recorder, writing the IK memory on success; ``ok(poses)`` decides."""

    def __init__(self, ok=lambda poses: True):
        self.ok, self.calls, self.solutions = ok, [], {("left", (0.0,)): th.ones(28)}

    def exact(self, poses, stance, arm="right", ignore_for_last=(), scale_key=None, cache=True, **kw):
        self.calls.append({"poses": list(poses), "stance": stance, "arm": arm, "ignore_for_last": list(ignore_for_last), "scale_key": scale_key, "cache": cache})
        ok = bool(self.ok(poses))
        if ok:
            for p in poses:
                self.solutions[(arm, tuple(round(float(v), 3) for v in list(G.t(p[0])) + list(G.t(p[1]))))] = th.zeros(28)
        return IK.IKResult(ok, th.zeros(28) if ok else None, "" if ok else "exact IK failed at pose 2 of 2", len(poses))


def _sampler(ok=lambda poses: True, **gen) -> "IL.IORSLarge":
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.gp, s.ikc, s.kit = _GP(**gen), _IKC(ok), SimpleNamespace(stats={})
    s.grasp_ignore_objects = lambda: [FLOOR]
    s.choice = SimpleNamespace(models={"target": SimpleNamespace(scale=(1.0, 1.0, 1.0))})
    return s


def _stance() -> Stance:
    return Stance("manipulation", (0.0, 0.0, 0.0), 0.95, -0.3, 0.55, None, None, 0.0, None, ["left"])


def _grasp(x: float, name: str):
    eef = (th.tensor([x, 0.2, 0.30]), DOWN.clone())
    return SimpleNamespace(name=name), eef, IK.pre_pose(eef)


def test_the_key():
    assert _sampler().grasp_reach_margin() is None
    assert _sampler(grasp_reach_margin_m=0.0).grasp_reach_margin() is None and _sampler(grasp_reach_margin_m=-0.01).grasp_reach_margin() is None
    assert _sampler(grasp_reach_margin_m=0.015).grasp_reach_margin() == pytest.approx(0.015)


def test_without_the_key_nothing_is_solved():
    s = _sampler()
    g, eef, pre = _grasp(0.6, "grasp_0")
    opts = ([(g, eef, pre, False)], {"chosen": "row"})
    assert s._b_margin_options(opts, _stance(), "left", "B") == (opts, "")
    assert s.grasp_margin_fails(_stance(), "left", eef, pre) is None and s.ikc.calls == []


def test_the_grasp_solves_deeper_along_its_approach(caplog):
    s = _sampler(grasp_reach_margin_m=0.015)
    g, eef, pre = _grasp(0.6, "grasp_0")
    memory = dict(s.ikc.solutions)
    st = _stance()
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        out, why = s._b_margin_options(([(g, eef, pre, False)], None), st, "left", "B")
    assert out == ([(g, eef, pre, False)], None) and why == ""
    c = s.ikc.calls[0]
    assert len(s.ikc.calls) == 1 and c["stance"] is st and c["arm"] == "left" and c["cache"] is False
    assert c["ignore_for_last"] == [FLOOR] and c["scale_key"] == (1.0, 1.0, 1.0)  # the B solve's ignore set and scale
    # [pre, grasp] 1.5 cm deeper along the approach axis (down for a top-down grasp), the orientations kept
    assert c["poses"][0][0].tolist() == pytest.approx((pre[0] + th.tensor([0.0, 0.0, -0.015])).tolist(), abs=1e-6)
    assert c["poses"][1][0].tolist() == pytest.approx([0.6, 0.2, 0.285], abs=1e-6)
    assert all(th.allclose(p[1], DOWN) for p in c["poses"])
    assert s.ikc.solutions == memory  # the deeper solve leaves no witness behind
    assert s.kit.stats == {"grasp_reach_margin.checked": 1}
    assert "grasp reach margin [B] 0.015 m deeper along the approach (left arm, stance (0.95, -0.30)): ['grasp_0'] kept" in caplog.text


def test_the_finger_assignments_are_filtered_in_order(caplog):
    # the row's grasp lies 1 cm farther out: only its twin keeps the margin (the deeper eef beyond x 0.705 has no solution)
    s = _sampler(ok=lambda poses: float(poses[-1][0][0]) < 0.705, grasp_reach_margin_m=0.015)
    row, twin = _grasp(0.71, "grasp_1"), _grasp(0.70, "grasp_1~r180")
    facts = {"stage": "grasp B", "chosen": "row", "reason": "default"}
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        (kept, f), why = s._b_margin_options(([(*row, False), (*twin, True)], facts), _stance(), "left", "B")
    assert kept == [(*twin, True)] and why == "grasp_1: exact IK failed at pose 2 of 2"
    assert f["chosen"] == "twin" and f["reach_margin_flip"] == "the row fails the reach margin: grasp_1: exact IK failed at pose 2 of 2" and facts["chosen"] == "row"
    assert s.kit.stats == {"grasp_reach_margin.checked": 2, "grasp_reach_margin.rejected": 1}
    assert "['grasp_1~r180'] kept; grasp_1: exact IK failed at pose 2 of 2" in caplog.text
    # the first kept: the record unchanged; none kept: no option
    (kept, f), _ = s._b_margin_options(([(*twin, True), (*row, False)], facts), _stance(), "left", "B")
    assert kept == [(*twin, True)] and f is facts
    assert s._b_margin_options(([(*row, False)], facts), _stance(), "left", "B") == (None, "grasp_1: exact IK failed at pose 2 of 2")


def test_the_segment_fallback_regrasp_keeps_the_margin(caplog):
    s = _sampler(ok=lambda poses: float(poses[-1][0][2]) > 0.29, grasp_reach_margin_m=0.015)  # the grasp solves, 1.5 cm deeper it does not
    g = SimpleNamespace(name="grasp_0", pose=(th.tensor([0.0, 0.0, 0.10]), DOWN.clone()))
    t_pose = (th.tensor([0.6, 0.2, 0.20]), G.IDENTITY_QUAT.clone())
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        assert s._regrasp_at_b(g, t_pose, _stance(), "left", 0.0, False, {}) is None
    assert len(s.ikc.calls) == 2 and "segment fallback: the left arm's regrasp of grasp_0 at B fails (grasp reach margin: grasp_0: exact IK failed" in caplog.text
    eef = IK.grasp_pose(t_pose, g.pose)
    assert s.ikc.calls[1]["poses"][1][0].tolist() == pytest.approx((G.t(eef[0]) + th.tensor([0.0, 0.0, -0.015])).tolist(), abs=1e-6)
    # with the margin kept (or without the key) the regrasp's options are today's
    s = _sampler(grasp_reach_margin_m=0.015)
    (opts, facts) = s._regrasp_at_b(g, t_pose, _stance(), "left", 0.0, False, {})
    assert len(opts) == 1 and opts[0][0] is g and opts[0][3] is False and facts is None and len(s.ikc.calls) == 2
    s = _sampler()
    assert s._regrasp_at_b(g, t_pose, _stance(), "left", 0.0, False, {})[0][0][0] is g and len(s.ikc.calls) == 1
