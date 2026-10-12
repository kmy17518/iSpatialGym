"""Simulator-free tests of the large-scale IORS relaxation keys (owner 2026-10-12 item 2, the large-scale eval picks; spec §5.6
"Relaxation keys"): every key unset keeps the rule as before (``pytest ispatialgym/tests/test_iors_large_relaxations.py``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.sampling import geometry as G

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")


class _GP:
    def __init__(self, alias: str = "inter_object_relation_satisfaction-ontop_large_scale-mouse-desk-floor", **gen):
        self.alias, self.generation = alias, dict(gen)

    def gen(self, key, default=None):
        return self.generation.get(key, default)


# ----------------------------------------------------------------------------------------------------------------
# 2a: partner partly in view (generation.partner_view_min_points)
# ----------------------------------------------------------------------------------------------------------------


def test_enough_in_view_counts_the_points_in_frame():
    inside = th.tensor([True, False, False])
    assert not IL.enough_in_view(inside)  # default: every point, as before
    assert IL.enough_in_view(th.tensor([True, True, True]))
    assert IL.enough_in_view(inside, 1) and not IL.enough_in_view(inside, 2)
    assert IL.enough_in_view(th.tensor([True, True, True]), 5) and not IL.enough_in_view(th.tensor([True, True, False]), 5)  # fewer points: all
    assert not IL.enough_in_view(th.tensor([False, False]), 1)
    assert "partner_view_min_points" in IL.RELAXATION_KEYS


class _ViewStub:
    """The attributes :meth:`IORSLarge.goal_region_visible` / :meth:`partner_points_in_view` read: a 90° pinhole camera at the origin
    looking down (-z), the region points given."""

    goal_region_visible = IL.IORSLarge.goal_region_visible
    partner_points_in_view = IL.IORSLarge.partner_points_in_view

    def __init__(self, points, **gen):
        self.gp = _GP(**gen)
        self.world = SimpleNamespace(camera_model=G.CameraModel(640, 480, 320.0, 320.0, 320.0, 240.0))
        self._points = th.tensor(points, dtype=th.float32)

    def camera_pose_at(self, stance):
        return th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])

    def goal_region_points(self, from_xy):
        return self._points


def test_goal_region_visible_with_part_of_the_partner_in_frame():
    st = SimpleNamespace(focus=th.tensor([0.0, 0.0, 0.0]), base_pose2d=(0.0, 0.0, 0.0))
    # 5 top-patch points in frame + 3 bottom-edge points outside (u > 640 at x = 2 m, 1 m below the camera)
    pts = [[0.0, 0.0, -1.0], [0.1, 0.1, -1.0], [0.1, -0.1, -1.0], [-0.1, 0.1, -1.0], [-0.1, -0.1, -1.0], [2.0, 0.0, -1.0], [2.0, 0.5, -1.0], [2.0, -0.5, -1.0]]
    assert not _ViewStub(pts).goal_region_visible(st)  # today: every point
    assert _ViewStub(pts, partner_view_min_points=1).goal_region_visible(st)
    assert _ViewStub(pts, partner_view_min_points=5).goal_region_visible(st) and not _ViewStub(pts, partner_view_min_points=6).goal_region_visible(st)
    # the 5 % margin is the one of P0.in_frame: a point 3 % inside the left border does not count
    edge = [[-1.0 + 0.03 * 2.0, 0.0, -1.0]]
    assert not _ViewStub(edge, partner_view_min_points=1).goal_region_visible(st)
    # nothing in frame (behind the camera) never passes
    assert not _ViewStub([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], partner_view_min_points=1).goal_region_visible(st)
    assert _ViewStub([[0.0, 0.0, -1.0]] * 3).goal_region_visible(st)
