"""i7-yield2 (2026-11-08): world_vertical_04 lost 49 of 51 attempts to [too_few_visible] at settle_validate (the init view of a bare wall:
floor, wall, target). ``ISPATIALGYM_WALL_VIS_SCREEN=1`` screens each draw (spot, delta, init stance) with the settle check's own rule at
the drawn init view and redraws within the attempt (default on since the A/B); =0, the draw is the one draw of before and nothing is rendered."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import prompt0 as P0
from ispatialgym.sampling.v2.tasks import alignment_wall as AW


class _Stop(Exception):
    pass


class _Wall(AW.AlignmentWall):
    target = SimpleNamespace(set_position_orientation=lambda *a: None, keep_still=lambda: None)
    tname = "target_x"


def _task(views):
    t = _Wall.__new__(_Wall)
    t.spots = [{"id": i, "weight": 1.0} for i in range(5)]
    t.kit = SimpleNamespace(sync_world=lambda: None, stats={})
    t.drawn = []
    t.views = list(views)
    t.deviation_range = lambda: (0.2, 0.6)
    t.contacts_for = lambda spot, delta: (t.drawn.append(spot["id"]) or {"pose_tilted": (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), "hull_tilted": th.zeros(4, 3)})
    t.face_of = lambda spot: None
    t.param = lambda key, default=None, v1_key=None: default
    t.visible_at = lambda init: t.views.pop(0)
    t.sidecar = SimpleNamespace(add_stance=lambda st: setattr(t, "added", st))

    def screen(cands, geo, stats=None):
        raise _Stop()  # the Phase-2 search the screened draw goes on to

    t.screen = screen
    t.step1 = {i: [] for i in range(5)}
    return t


@pytest.fixture
def init_draw(monkeypatch):
    monkeypatch.setattr(P0, "phase1_init_pose", lambda *a, **k: SimpleNamespace(name=None, base_pose2d=(0.0, 0.0, 0.0)))


def test_switch_parsing(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_WALL_VIS_SCREEN", raising=False)
    assert AW.wall_vis_screen_draws() == 40  # default on (i7-yield2 A/B, 2026-11-08)
    monkeypatch.setenv("ISPATIALGYM_WALL_VIS_SCREEN", "0")
    assert AW.wall_vis_screen_draws() == 0
    monkeypatch.setenv("ISPATIALGYM_WALL_VIS_SCREEN", "1")
    assert AW.wall_vis_screen_draws() == 40
    monkeypatch.setenv("ISPATIALGYM_WALL_VIS_SCREEN_DRAWS", "7")
    assert AW.wall_vis_screen_draws() == 7


def test_off_draws_once_and_renders_nothing(monkeypatch, init_draw):
    monkeypatch.setenv("ISPATIALGYM_WALL_VIS_SCREEN", "0")
    t = _task([])
    with pytest.raises(_Stop):
        t.sample_instance(np.random.default_rng(0))
    assert len(t.drawn) == 1 and t.views == [] and t.kit.stats == {} and t.added.name == "init"


def test_on_redraws_until_the_init_view_shows_enough_objects(monkeypatch, init_draw):
    monkeypatch.setenv("ISPATIALGYM_WALL_VIS_SCREEN", "1")
    few, many = ["floor", "wall", "target"], ["floor", "wall", "target", "door", "wall2"]
    t = _task([few, few, many])
    with pytest.raises(_Stop):
        t.sample_instance(np.random.default_rng(0))
    assert len(t.drawn) == 3 and t.kit.stats == {"vis_screen.redrawn": 2, "vis_screen.passed": 1, "vis_screen.draws_3": 1}
    # the first draw is the one draw of the switch off (same rng stream)
    off = _task([])
    monkeypatch.setenv("ISPATIALGYM_WALL_VIS_SCREEN", "0")
    with pytest.raises(_Stop):
        off.sample_instance(np.random.default_rng(0))
    assert t.drawn[0] == off.drawn[0]


def test_on_exhausted_rejects_too_few_visible(monkeypatch, init_draw):
    monkeypatch.setenv("ISPATIALGYM_WALL_VIS_SCREEN", "1")
    monkeypatch.setenv("ISPATIALGYM_WALL_VIS_SCREEN_DRAWS", "4")
    t = _task([["floor", "wall", "target"]] * 4)
    with pytest.raises(Reject) as e:
        t.sample_instance(np.random.default_rng(1))
    assert e.value.reason == "too_few_visible" and "vis screen: 4 draws" in str(e.value)
    assert len(t.drawn) == 4 and t.kit.stats == {"vis_screen.redrawn": 4, "vis_screen.exhausted": 1}


def test_visible_at_teleports_to_the_stance_and_back():
    t = _Wall.__new__(_Wall)
    moves = []
    q_now = th.arange(28, dtype=th.float32)
    t.robot = SimpleNamespace(name="robot_r1", get_joint_positions=lambda: q_now.clone())
    t.kit = SimpleNamespace(world_base_pose2d=lambda: (1.0, 2.0, 0.5), jidx={n: i for i, n in enumerate(M.BASE_VIRTUAL_JOINTS)})
    t.stance_q = lambda st: th.full((28,), 0.3)
    t.world = SimpleNamespace(teleport_robot=lambda base, q: moves.append((tuple(base), q.clone())),
                              visible_pixels=lambda exclude_structure=True: {"a": 500, "b": 100, "robot_r1": 9000, "c": 200})
    t.param = lambda key, default=None, v1_key=None: default
    vis = t.visible_at(SimpleNamespace(base_pose2d=(3.0, 4.0, 0.1)))
    assert vis == ["a", "c"]
    assert [m[0] for m in moves] == [(3.0, 4.0, 0.1), (1.0, 2.0, 0.5)]
    nb = len(M.BASE_VIRTUAL_JOINTS)
    assert float(moves[0][1][:nb].abs().max()) == 0.0 and float(moves[0][1][nb]) == pytest.approx(0.3)
    assert float(moves[1][1][:nb].abs().max()) == 0.0 and float(moves[1][1][nb]) == pytest.approx(float(q_now[nb]))
