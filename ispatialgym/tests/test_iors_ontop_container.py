"""Ontop small scale, ``generation.ontop_centre: "container"`` (oracle-fix 2026-11-06, owner on the ontop_small#02 steak demo: "want to
scale down steak so it can be grasped lying down & fits inside the floor of the frying pan (right now it's standing up)"): the goal centres
the target's footprint on the partner's floor (the lowest flat region seen from above, :class:`iors_small.ContainerFloor`) instead of the
partner's AABB centre - a pan's AABB holds its long handle -, and the pair check fits the footprint in the floor's disc. ``IORSSmall``
methods run on stand-ins (a pan with a 20 cm floor, a wall and a 21 cm handle); no simulator is launched."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.tasks import iors_small as IS

#: the pan's floor centre in its own frame (x +0.02 - the handle reaches to x -0.30, the AABB centre lies at x -0.084)
FLOOR_C, FLOOR_R, FLOOR_Z = (0.02, 0.0), 0.10, -0.007
#: the lying x0.6 steak's box (7.95 x 15.2 x 4.0 cm), its origin 1 cm off its footprint's centre along x
STEAK = (0.0795, 0.152, 0.040)


def _ring(r, z, n=32, c=FLOOR_C):
    return [[c[0] + r * math.cos(2 * math.pi * k / n), c[1] + r * math.sin(2 * math.pi * k / n), z] for k in range(n)]


def _pan_pieces():
    """Convex pieces as a frying pan's collision decomposition: the floor slab, 24 wall segments, the handle bar."""
    pieces = [th.tensor(_ring(FLOOR_R, -0.02) + _ring(FLOOR_R, FLOOR_Z), dtype=th.float32)]
    for k in range(24):
        a0, a1 = 2 * math.pi * k / 24, 2 * math.pi * (k + 1) / 24
        pieces.append(th.tensor([[FLOOR_C[0] + r * math.cos(a), FLOOR_C[1] + r * math.sin(a), z] for r in (FLOOR_R, 0.112) for a in (a0, a1)
                                 for z in (-0.02, 0.035)], dtype=th.float32))
    pieces.append(th.tensor([[x, y, z] for x in (-0.30, -0.09) for y in (-0.015, 0.015) for z in (0.0, 0.02)], dtype=th.float32))
    return pieces


def _box(dx, dy, h, off=(0.0, 0.0)):
    return th.tensor([[off[0] + sx * dx / 2, off[1] + sy * dy / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, h)], dtype=th.float32)


def _sampler(gen=None, target=None, target_rest=None):
    pieces = _pan_pieces()
    s = IS.IORSSmall.__new__(IS.IORSSmall)
    s.relation, s.gp, s.spec = "ontop", SimpleNamespace(variant="ontop-steak-frying_pan"), SimpleNamespace(generation=dict(gen or {}))
    s.hull = {"target": _box(*STEAK, off=(-0.01, 0.0)) if target is None else target, "partner": th.cat(pieces)}
    s.rest = {"target": C.IDENTITY_QUAT.clone() if target_rest is None else target_rest, "partner": C.IDENTITY_QUAT.clone()}
    s.shelf_start, s.template_facts, s.pair = None, {}, {}
    s.world = SimpleNamespace(hull_at=G.local_points_world, local_collision_pieces=lambda obj: pieces,
                              footprint=lambda h, pose: G.convex_hull_2d(G.local_points_world(h, pose)[:, :2]))
    s.obj = lambda role: SimpleNamespace(links={})
    s.sidecar = SimpleNamespace(stance=lambda name: None)
    s.target_goal_quat = lambda t_pose, st, p_pose=None: G.t(t_pose[1])
    s.release_surface = lambda xy, p_pose, top: None
    return s


#: the pan on the table, turned 90 deg: its handle points to world -y
P_POSE = (th.tensor([1.0, 0.5, 0.82]), G.quat_from_yaw(math.pi / 2))
T_POSE = (th.tensor([0.7, -0.2, 0.80]), C.IDENTITY_QUAT.clone())


def test_the_floor_is_the_lowest_flat_disc_not_the_handle_or_the_rim():
    f = IS.ContainerFloor.build(_pan_pieces())
    assert f is not None
    assert f.centre_xy.tolist() == pytest.approx(list(FLOOR_C), abs=0.003)
    assert f.z == pytest.approx(FLOOR_Z, abs=1e-4)
    assert f.radius == pytest.approx(FLOOR_R, abs=0.004)  # the wall's inner chords
    assert f.radius_at((-0.084, 0.0)) < 0.02  # the AABB centre lies at the floor's far edge


def test_the_goal_centres_the_footprint_on_the_floor_not_the_aabb():
    s = _sampler({"ontop_centre": "container"})
    floor = s.container_floor()
    assert floor["source"] == "floor" and floor["centre_root"].tolist() == pytest.approx([FLOOR_C[0], FLOOR_C[1], FLOOR_Z], abs=0.003)
    w = s.container_floor_at(P_POSE)
    assert w["xy"].tolist() == pytest.approx([1.0, 0.5 + FLOOR_C[0]], abs=0.003)  # the pan turned 90 deg: its +x is world +y
    goal = s.goal_pose(T_POSE, P_POSE)
    fp = G.local_points_world(s.hull["target"], goal)[:, :2]
    assert IS.min_area_rect(fp)[3].tolist() == pytest.approx(w["xy"].tolist(), abs=0.003)  # the steak's footprint, not its origin, centred
    # today's rule: the partner's world AABB centre, 10 cm toward the handle
    s0 = _sampler()
    goal0 = s0.goal_pose(T_POSE, P_POSE)
    p_w = G.local_points_world(s0.hull["partner"], P_POSE)
    aabb_c = (p_w[:, :2].max(dim=0).values + p_w[:, :2].min(dim=0).values) / 2
    assert goal0[0][:2].tolist() == pytest.approx(aabb_c.tolist(), abs=1e-5)
    assert float(th.linalg.norm(goal0[0][:2] - goal[0][:2])) > 0.09
    assert goal[0][2] == pytest.approx(goal0[0][2], abs=1e-6)  # the height rule is unchanged (no partner ray here: the hull top)


def test_the_pair_check_fits_the_footprint_in_the_floor_disc():
    out = _sampler({"ontop_centre": "container"}).pair_admissibility()
    assert out["tier"] == "floor_disc" and out["target_reach_m"] == pytest.approx(math.hypot(*STEAK[:2]) / 2, abs=1e-3)
    # the unscaled steak standing on its edge (25 x 6.3 cm footprint) passes the pan's box but leaves its floor
    standing = _box(0.063, 0.251, 0.129)
    assert _sampler(target=standing).pair_admissibility()["tier"] in ("any_yaw", "aligned", "centred")
    with pytest.raises(Reject) as e:
        _sampler({"ontop_centre": "container"}, target=standing).pair_admissibility()
    assert e.value.reason == "partner_too_small" and "floor disc" in str(e.value)
    with pytest.raises(Reject) as e:
        _sampler({"ontop_centre": "middle"}).pair_admissibility()
    assert e.value.reason == "spec"


def test_without_a_floor_the_fillable_volume_centres_and_the_box_fits():
    s = _sampler({"ontop_centre": "container"})
    s.world.local_collision_pieces = lambda obj: []  # nothing seen from above
    s._fillable_root = lambda obj: th.tensor([0.03, 0.0, 0.01])
    floor = s.container_floor()
    assert floor["source"] == "fillable" and floor["radius_m"] is None
    assert s.container_floor_at(P_POSE)["xy"].tolist() == pytest.approx([1.0, 0.53], abs=1e-5)
    assert s.pair_admissibility()["tier"] in ("any_yaw", "aligned", "centred")  # no disc: the box tiers
