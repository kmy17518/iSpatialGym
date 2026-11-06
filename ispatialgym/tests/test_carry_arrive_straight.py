"""oracle-fix 2026-11-06 (hardback inside_large#01, generation.carry_arrive_straight_m): a carried path that arrives straight - the grid path
ends ``arrive_straight_m`` behind the placement stance along its own axis, at its heading, and the last leg drives straight in, so the turn
to the stance's heading happens there and not at the stance (the book ~0.8 m ahead of the base swept into bookcase_jaysra_0 on the turn,
the base's wheels touched bed_rrcvaq_1). No simulator."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch as th

from ispatialgym.sampling.v2 import navigation as NAV


def _grid(monkeypatch, goals):
    monkeypatch.setenv("ISG_ATTACHED_CHECKS", "off")
    monkeypatch.setattr(NAV, "local_map", lambda kit, rooms=None, blocked=(): th.ones(60, 60, dtype=th.bool))
    monkeypatch.setattr(NAV, "_to_map", lambda kit, xy: (int(round(float(xy[1]) / 0.05)), int(round(float(xy[0]) / 0.05))))
    monkeypatch.setattr(NAV, "_to_world", lambda kit, ij: th.tensor([ij[1] * 0.05, ij[0] * 0.05]))
    monkeypatch.setattr(NAV, "nearest_free_cell", lambda free, ij, r=15: ij)
    monkeypatch.setattr(NAV, "same_region", lambda free, a, b: True)
    monkeypatch.setattr(NAV, "grid_shortest_path", lambda free, a, b: (goals.append(b), [a, (a[0], b[1]), b])[1])
    monkeypatch.setattr(NAV, "simplify_path", lambda cells, free: list(cells))
    monkeypatch.setattr(NAV, "path_collision_free", lambda kit, poses, trunk, posture="untucked", margin=0.01, q_arms=None, fingers=None: (True, -1))
    st = lambda name, x, y, yaw: SimpleNamespace(name=name, base_xy=th.tensor([x, y]), yaw=yaw, base_pose2d=(x, y, yaw), translate=0.0, tilt=0.0)  # noqa: E731
    # B facing +x at the desk, C facing -y at the bookcase (the hardback's layout)
    return st("pick", 2.0, 2.0, 0.0), st("placement", 1.0, 0.5, -math.pi / 2)


def test_the_path_ends_behind_the_stance_and_drives_straight_in(monkeypatch):
    goals = []
    b, c = _grid(monkeypatch, goals)
    arm_q = {"right": th.zeros(7), "left": th.zeros(7)}
    rec = NAV.plan_path(None, b, c, arrive_straight_m=0.35, arm_q=arm_q)
    pre, last = rec.waypoints[-2], rec.waypoints[-1]
    assert last == c.base_pose2d
    assert abs(pre[0] - 1.0) < 1e-6 and abs(pre[1] - 0.85) < 1e-6 and abs(pre[2] - (-math.pi / 2)) < 1e-9  # 0.35 m behind C, at C's heading
    assert goals[-1] == (17, 20)  # the grid path is planned to the point behind C (y 0.85, x 1.0 on the 5 cm grid)
    # the last leg is a straight drive along C's heading: no turn between the two last waypoints
    assert abs(math.atan2(last[1] - pre[1], last[0] - pre[0]) - c.yaw) < 1e-6


def test_without_the_key_the_path_ends_at_the_stance(monkeypatch):
    goals = []
    b, c = _grid(monkeypatch, goals)
    rec = NAV.plan_path(None, b, c, arm_q={"right": th.zeros(7), "left": th.zeros(7)})
    assert rec.waypoints[-1] == c.base_pose2d and goals[-1] == (10, 20)  # planned to C itself
    assert all(abs(w[1] - 0.85) > 1e-6 for w in rec.waypoints)  # no point behind C
