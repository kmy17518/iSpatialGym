"""Attachment's template keeps only the Phase-2 survivors that leave the partner a cell an attempt can use (owner 2026-10-12, 7h): a
partner footprint on free cells of the top at the required gap from the focus, among the attach-band cells nearest the focus at the
survivor's stance or a second base pose; separate rejections for a Phase 2 that found nothing and for survivors without a partner
cell; the attempts' memory of rejected candidates survives the split cone's per-attempt position draw."""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.sim_utils import Reject, SupportTop
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.sampling.v2.tasks import attachment as AT

RES = 0.02
EDGE = EdgeFrame.from_edge("+x", th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]), 1.6)  # the top beyond x = 1.0, the robot facing +x
FOCUS = (1.40, 0.0)


def _box(hx, hy, h):
    return th.tensor([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (0.0, h)], dtype=th.float32)


def _top(occupied=lambda x, y: False):
    """The top x in [1.0, 2.0], y in [-0.8, 0.8] (2 cm cells); ``occupied(x, y)``: cells another object (a sink) takes."""
    xs = 1.0 + RES / 2 + RES * th.arange(50, dtype=th.float32)
    ys = -0.8 + RES / 2 + RES * th.arange(80, dtype=th.float32)
    occ = th.tensor([[bool(occupied(float(x), float(y))) for y in ys] for x in xs])
    return SupportTop("table_x_0", xs, ys, th.ones(50, 80, dtype=th.bool), occ, 0.8, 0.0, [], RES)


def _sampler(monkeypatch, top, rel_cells):
    """A 10 cm male and a 20 x 20 cm partner; the attach band = ``rel_cells`` (robot frame: x ahead, y to the left) at every trunk."""
    s = object.__new__(AT.Attachment)
    s.t_role, s.p_role, s.top, s.attach_bands = "target", "partner", top, object()
    s.hull = {"target": _box(0.05, 0.05, 0.03), "partner": _box(0.10, 0.10, 0.14)}
    s.rest = {"target": C.IDENTITY_QUAT.clone()}
    s.g = lambda key: {"partner_gap_min_m": 0.04, "partner_centre_distance_min_m": 0.08}[key]
    s.gp, s.template_facts = NS(alias="a"), {}
    rel = th.tensor(rel_cells, dtype=th.float32).reshape(-1, 2)

    def band_cells_world(bands, st):
        x, y, yaw = (float(v) for v in st.base_pose2d)
        assert abs(yaw) < 1e-9
        return rel + th.tensor([x, y]) if len(rel) else th.zeros(0, 2)

    monkeypatch.setattr(AT.TT, "band_cells_world", band_cells_world)
    return s


def _cand(lateral=0.0):
    c = Stance("manipulation", (0.55, 0.0, 0.0), 0.5, 0.0, 0.45, EDGE).shifted(lateral=lateral, name="manipulation")
    c.focus = th.tensor([FOCUS[0], FOCUS[1], 0.83])
    return c


LATTICE = [(x, y) for x in np.arange(0.48, 1.13, 0.08) for y in np.arange(-0.48, 0.17, 0.08)]  # the right arm's cells, 0.48-1.12 m ahead


def test_open_top_keeps_the_survivor(monkeypatch):
    s = _sampler(monkeypatch, _top(), LATTICE)
    kept, facts = s._partner_room([_cand()])
    assert len(kept) == 1 and facts["dropped"] == {"no_cells": 0, "support": 0, "gap": 0} and facts["male_radius_m"] == pytest.approx(0.0707, abs=1e-4)


def test_a_sink_around_the_focus_drops_the_survivor_at_every_stance(monkeypatch):
    s = _sampler(monkeypatch, _top(lambda x, y: (x - FOCUS[0]) ** 2 + (y - FOCUS[1]) ** 2 < 0.6 ** 2), LATTICE)
    kept, facts = s._partner_room([_cand()])
    assert kept == [] and facts["dropped"]["support"] == 1


def test_cells_only_next_to_the_focus_fail_the_gap(monkeypatch):
    near = [(FOCUS[0] - 0.55 + dx, FOCUS[1] + dy) for dx in (-0.08, 0.0, 0.08) for dy in (-0.08, 0.0, 0.08)]  # within 11 cm of the focus
    s = _sampler(monkeypatch, _top(), near)
    assert len(s._partner_room([_cand()])[0]) == 1  # a second base pose 0.3 m along the edge moves the cells away from the focus
    monkeypatch.setattr(AT.Attachment, "_second_base_stances", staticmethod(lambda manip: iter(())))
    kept, facts = s._partner_room([_cand()])
    assert kept == [] and facts["dropped"]["gap"] == 1
    s = _sampler(monkeypatch, _top(), [])
    assert s._partner_room([_cand()])[1]["dropped"]["no_cells"] == 1


def test_a_second_base_pose_with_room_keeps_the_survivor(monkeypatch):
    sink = _top(lambda x, y: y < 0.10)  # free only on the robot's left of y = 0.10
    s = _sampler(monkeypatch, sink, LATTICE)
    kept, _ = s._partner_room([_cand()])
    assert len(kept) == 1  # a stance shifted left along the edge brings band cells onto the free part
    monkeypatch.setattr(AT.Attachment, "_second_base_stances", staticmethod(lambda manip: iter(())))
    kept, facts = s._partner_room([_cand()])
    assert kept == [] and facts["dropped"]["support"] == 1  # from the grasp stance alone there is no room


def test_second_base_stances_are_the_ones_the_attempt_tries():
    c = _cand()
    out = list(AT.Attachment._second_base_stances(c))
    assert len(out) == len(AT.SECOND_BASE_LATERALS) * 3
    assert [(d, t) for d, t, _ in out[:3]] == [(0.10, (0.5, 0.0)), (0.10, (0.5, 0.0)), (0.10, (0.25, -0.3))]
    for dlat, trunk, st in out:
        assert st.lateral == pytest.approx(c.lateral + dlat) and (st.translate, st.tilt) == trunk and st.name == "attach"


def test_separate_rejections_for_no_phase2_pair_and_no_partner_cell(monkeypatch):
    s = object.__new__(AT.Attachment)
    s.template_facts = {}
    with pytest.raises(Reject) as e:
        s._with_partner_room([], EDGE)
    assert e.value.reason == "no_manipulation_stance" and "Phase 2 found no (stance, focus) pair" in str(e.value)
    monkeypatch.setattr(AT.Attachment, "_partner_room", lambda self, cands: ([], {"survivors": len(cands), "kept": 0, "dropped": {"support": 2}}))
    with pytest.raises(Reject) as e:
        s._with_partner_room([_cand(), _cand(0.1)], EDGE)
    assert e.value.reason == "no_partner_cell" and "Phase 2 left 2 survivors, none with a partner cell" in str(e.value)
    assert s.template_facts["partner_room"]["+x"]["dropped"] == {"support": 2}
    monkeypatch.setattr(AT.Attachment, "_partner_room", lambda self, cands: (cands[1:], {"kept": 1}))
    c = [_cand(), _cand(0.1)]
    assert s._with_partner_room(c, EDGE) == c[1:]


def test_the_reject_memory_survives_the_split_cone_position_draw():
    def cand(lateral, pos):
        c = _cand(lateral)
        c.extras.update({"_position": pos, "angle_class": 0, "init_view_class": 0})
        return c

    s = object.__new__(AT.Attachment)
    s.manip_candidates = [cand(0.0, (5, 0)), cand(0.1, (5, 0))]
    rejects = {AT.Attachment._candidate_key(s.manip_candidates[0]): 1}
    assert s._candidate_order(np.random.default_rng(0), rejects) == [1, 0]
    # another attempt drew the same position again: new objects, another order - the rejected stance still goes last
    s.manip_candidates = [cand(0.1, (5, 0)), cand(0.0, (5, 0))]
    assert s._candidate_order(np.random.default_rng(0), rejects) == [0, 1]
    # the split cone's per-attempt draw keeps the memory
    s.subtask, s.gen = AT.SUBTASK, (lambda key, default=None: {"phase2_split_cone": True}.get(key, default))
    s._candidate_rejects = dict(rejects)
    TT.split_cone_draw(s, np.random.default_rng(1))
    assert s._candidate_rejects == rejects and len(s.manip_candidates) == 2
