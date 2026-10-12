"""Block design: one stance for every block (owner 2026-10-12 item 3, ``generation.block_single_stance``) - no simulator.

``BlockDesign._flip_zone_miss`` (the flip-zone rule ``_rank_stances`` excludes a stance by), ``_single_stance_block`` (the
``second_stance_needed`` rejection), ``sample_instance`` (that rejection ends the attempt) and ``_regrasp_leg`` (no re-grasp from
another stance). Off (the default) every call is today's.
"""

from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.stances import Stance

ALIAS = "configuration_matching-block_design-block_4-breakfast_table_board"
Q_ID = th.tensor([0.0, 0.0, 0.0, 1.0])


def _bd():
    from ispatialgym.sampling.v2.tasks import block_design as BD

    return BD


def _sampler(gen):
    BD = _bd()
    s = object.__new__(BD.BlockDesign)
    s.spec = SimpleNamespace(generation=dict(gen))
    s.gp = SimpleNamespace(alias=ALIAS)
    s._centre = lambda pose: pose
    return BD, s


def _stance(name, x=0.0, y=0.0, stand_off=0.35):
    """A stance at (x, y) facing +x: base depth = world x - x, base lateral = world y - y."""
    return Stance(name, (x, y, 0.0), 0.75, -0.3, stand_off=stand_off)


def _pose(x, y):
    return th.tensor([x, y, 0.83]), Q_ID


def test_flip_zone_miss_names_why_and_is_the_rank_rule():
    """The flip zone of a stance: the cell right of -0.10 m lateral and >= ``flip_cell_min_edge_depth_m`` in from the edge;
    ``_rank_stances`` skips exactly the stances it misses for a flipping block (a picking block keeps the current stance first)."""
    BD, s = _sampler({})
    cur = _stance("cur")
    assert BD.BlockDesign._flip_zone_miss(s, cur, th.tensor([0.60, -0.20])) is None
    assert "base lateral -0.05 m, outside the flip zone (<= -0.10 m)" in BD.BlockDesign._flip_zone_miss(s, cur, th.tensor([0.60, -0.05]))
    assert "0.05 m in from the edge, outside the flip zone (>= 0.10 m)" in BD.BlockDesign._flip_zone_miss(s, cur, th.tensor([0.40, -0.20]))
    _, s2 = _sampler({"flip_cell_min_edge_depth_m": 0.04})
    assert BD.BlockDesign._flip_zone_miss(s2, cur, th.tensor([0.40, -0.20])) is None
    left = _stance("left", y=0.10)  # 0.10 m to the robot's left: the cell at -0.05 lies at -0.15 of its base
    pool = [cur, left]
    assert BD.BlockDesign._rank_stances(s, pool, cur, _pose(0.60, -0.05), _pose(0.55, 0.0), True) == [left]
    assert BD.BlockDesign._rank_stances(s, pool, cur, _pose(0.60, -0.05), _pose(0.55, 0.0), False)[0] is cur
    assert BD.BlockDesign._rank_stances(s, pool, cur, _pose(0.60, -0.20), _pose(0.55, 0.0), True)[0] is cur


def _single(gen, chain=None):
    BD, s = _sampler(gen)
    calls = []

    def block_program(rng, cands, *a, current=None):
        calls.append(([c.name for c in cands], current.name))
        if chain is not None:
            raise chain
        return {"instance": "toy_dice.n.01_2"}, cands[0]

    s._block_program = block_program
    return BD, s, calls


def _run_single(BD, s, flips, cell=(0.60, -0.20)):
    cur = _stance("manipulation")
    poses = {0: _pose(0.6, -0.3), 1: _pose(*cell)}
    return BD.BlockDesign._single_stance_block(s, None, cur, ["toy_dice.n.01_1", "toy_dice.n.01_2"], poses, dict(poses), {0: {"flips": False}, 1: {"flips": flips}},
                                               [0, 1], 1, th.tensor([0.0, -1.0]), th.tensor([-1.0, 0.0]), 0.0)


def test_single_stance_block_rejects_with_second_stance_needed():
    """A block after the first is solved from the first block's stance only; a flipping block whose cell lies outside its flip zone
    is rejected before any chain is planned, and a block without a chain there is rejected with the chain's reason - both as
    ``second_stance_needed`` naming the block."""
    BD, s, calls = _single({"block_single_stance": True})
    rec, st = _run_single(BD, s, flips=True)
    assert calls == [(["manipulation"], "manipulation")] and st.name == "manipulation"
    with pytest.raises(Reject) as e:
        _run_single(BD, s, flips=True, cell=(0.60, -0.05))
    msg = str(e.value)
    assert e.value.reason == "second_stance_needed" and "toy_dice.n.01_2 (flip, placement 2)" in msg and "outside the flip zone" in msg and len(calls) == 1
    rec, st = _run_single(BD, s, flips=False, cell=(0.60, -0.05))  # a pick has no flip zone
    assert calls[-1] == (["manipulation"], "manipulation")
    BD, s, calls = _single({"block_single_stance": True}, chain=Reject("place_unreachable", "block toy_dice.n.01_2 (flip): no stance of 1 passes its chain"))
    with pytest.raises(Reject) as e:
        _run_single(BD, s, flips=True)
    assert e.value.reason == "second_stance_needed" and "no chain from the first block's stance" in str(e.value) and "[place_unreachable]" in str(e.value)


def test_one_stance_tracks_every_block_as_the_program_check():
    """With one stance every block's chain - the first block's too, at the stance it stands at - is tracked with the program check's
    acceptance (``_oracle_tracking``) and records no re-pose (no base move); off, today's tracking."""
    for gen, expect in (({"block_single_stance": True}, True), ({"block_single_stance": True, "repose_after_base_move": True}, True), ({}, False)):
        BD, s = _sampler(gen)
        seen = []

        def chain(rng, st, *a):
            seen.append(s._oracle_tracking)
            return {"instance": "toy_dice.n.01_1", "grasp_class": "x/t+30/y0"}

        s._block_chain = chain
        cur = _stance("manipulation")
        poses = {0: _pose(0.6, -0.2)}
        for step in (0, 1):
            rec, st = BD.BlockDesign._block_program(s, None, [cur], ["toy_dice.n.01_1", "toy_dice.n.01_2"], {**poses, 1: _pose(0.6, -0.3)}, {**poses, 1: _pose(0.6, -0.3)},
                                                    {0: {"flips": False}, 1: {"flips": False}}, [0, 1], step, th.tensor([0.0, -1.0]), th.tensor([-1.0, 0.0]), 0.0, current=cur)
            assert "repose" not in rec and st is cur
        assert seen == [expect, expect] and s._oracle_tracking is False


def _instance_sampler(gen, reasons):
    BD, s = _sampler(gen)
    focus = th.tensor([0.6, -0.2, 0.83])
    cand = Stance("manipulation", (0.0, 0.0, 0.0), 0.5, -0.3, stand_off=0.35, focus=focus, extras={"focus_info": {"depth": 0.55, "cells": []}})
    s.init_stance = _stance("init")
    s.manip_candidates = [cand, cand.with_trunk(0.75, -0.3)]
    s._flips_expected = lambda: False
    s.sidecar = SimpleNamespace(reset_instance=lambda: None, add_stance=lambda st: None)
    s.world = SimpleNamespace(spawned={}, park=lambda o, i: None)
    s.kit = SimpleNamespace(sync_world=lambda: None)
    tried = []

    def sample_with(rng, init, manip, k):
        tried.append(k)
        raise Reject(reasons[len(tried) - 1], f"candidate {k}")

    s._sample_with = sample_with
    return BD, s, tried


def test_second_stance_needed_ends_the_attempt():
    """``second_stance_needed`` leaves ``sample_instance`` at once (no further manipulation candidate); any other rejection moves on
    to the next candidate as before."""
    import numpy as np

    BD, s, tried = _instance_sampler({"block_single_stance": True}, ["second_stance_needed", "layout"])
    with pytest.raises(Reject) as e:
        BD.BlockDesign.sample_instance(s, np.random.default_rng(0))
    assert e.value.reason == "second_stance_needed" and len(tried) == 1
    BD, s, tried = _instance_sampler({}, ["layout", "board_placement"])
    with pytest.raises(Reject) as e:
        BD.BlockDesign.sample_instance(s, np.random.default_rng(0))
    assert e.value.reason == "target_placement" and len(tried) == 2


def _regrasp_sampler(gen):
    BD, s = _sampler(gen)
    ranked = []
    s._teleport = lambda objects: None
    s._has_board = lambda: False
    s._class_attachable = lambda *a: False  # no re-grasp class at all: every stance's leg 2 fails
    s._pick_turns = lambda screened, st, support, block, d: iter(())
    s._carry_height = lambda: 0.12
    s.camera_pose_at = lambda st: None
    s.edge = 0.0254
    s.top = SimpleNamespace(z=0.81)
    s._proven_stances = [_stance("proven", y=0.05)]
    s._pool = [_stance("pool", y=0.10)]
    s._rank_stances = lambda pool, cur, a, b, flipping: ranked.append(cur.name) or list(pool)
    return BD, s, ranked


def test_regrasp_from_another_stance_is_not_tried_with_one_stance():
    """With one stance the flip-in-place re-grasp is tried from its leg-1 stance only (no proven / ranked stances); off, as before."""
    for gen, expect_ranked in (({"block_single_stance": True}, []), ({}, ["manipulation"])):
        BD, s, ranked = _regrasp_sampler(gen)
        diag = []
        manip = _stance("manipulation")
        leg2, st2 = BD.BlockDesign._regrasp_leg(s, manip, ["toy_dice.n.01_2"], 0, object(), object(), _pose(0.6, -0.2), _pose(0.55, 0.0), 0.0,
                                                th.tensor([0.0, -1.0]), th.tensor([-1.0, 0.0]), None, [], [], {"toy_dice.n.01_2": _pose(0.6, -0.2)}, [0.10], diag, "x/l70")
        assert leg2 is None and st2 is manip and ranked == expect_ranked
        assert any("not tried (generation.block_single_stance)" in d for d in diag) == bool(gen)
