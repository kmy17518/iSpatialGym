"""IORS small, under_small#04 (owner 2026-10-13 ~18:40: "align the spatula with the paper sheet, and move the sheet to a flat part of the
support"): ``generation.partner_aligned_to_target`` - the partner drawn with its long side along the target's long axis, the target's footprint
centred on the partner at the goal - and ``generation.partner_flat_top`` - the partner's footprint (grown by ``partner_flat_margin_m``) on the
part of the support's top seen from above as one flat face (:class:`iors_small.FlatTop`, rasterised from the visual meshes). On the synthetic
table of ``test_iors_start_region``. No simulator."""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.stances import Stance
from ispatialgym.sampling.v2.tasks import iors_small as IS
from ispatialgym.tests.test_iors_start_region import _box, _sampler

#: the 0.76 spatula's footprint (target_fit: 0.2689 x 0.0595) with its origin 1.65 cm toward one end, as hkvsep's toward its blade
SPATULA = _box(0.1345, 0.03, 0.021) + th.tensor([0.0165, 0.0, 0.0])
#: paper_sheet/uqbipw at its rest orientation: 0.21 (x) x 0.29 (y)
SHEET = _box(0.105, 0.145, 0.0005)


def _fp(hull, pose):
    return G.convex_hull_2d(G.local_points_world(hull, pose)[:, :2])


def _under(gen=None, tier="aligned"):
    """The synthetic table (top z 0.75, x 1.0-1.6, robot at x 0.55 facing +x) with the spatula and the sheet; D_p for the sheet's 0.29 m."""
    s = _sampler("under", L=3.0, gen=gen, pair={"tier": tier})
    s.hull = {"target": SPATULA, "partner": SHEET}
    s.init_stance.extras["margins"] = {"D_p": 0.32}
    s.meta_spots = None
    return s


def _manip(s, y=-0.35):
    c = Stance("manipulation", (0.55, 0.0, 0.0), 0.5, 0.0, 0.45, s.edge)
    c.focus = th.tensor([1.31, y, 0.78])
    m = c.with_trunk(c.translate, c.tilt, "manipulation")
    m.arms = ["right"]
    return m


# ---------------------------------------------------------------------------------------------------------------- alignment


def test_min_area_rect_and_the_aligned_partner_yaws():
    spat = _fp(SPATULA, (th.tensor([1.0, 2.0, 0.0]), G.quat_from_yaw(math.radians(-117.9))))
    ang, long, short, centre = IS.min_area_rect(spat)
    assert math.degrees(ang) == pytest.approx(62.1, abs=1e-3) and long == pytest.approx(0.269, abs=1e-5) and short == pytest.approx(0.06, abs=1e-5)
    off = th.tensor([0.0165 * math.cos(math.radians(-117.9)), 0.0165 * math.sin(math.radians(-117.9))])
    assert centre.tolist() == pytest.approx((th.tensor([1.0, 2.0]) + off).tolist(), abs=1e-5)  # the rectangle's centre, not the origin
    sheet = _fp(SHEET, (th.zeros(3), G.quat_from_yaw(0.0)))
    assert math.degrees(IS.min_area_rect(sheet)[0]) == pytest.approx(90.0, abs=1e-4)
    # c3_k1: the sheet at yaw 93.9 deg, the spatula at -117.9: the long axes 58.2 deg apart (31.8 deg off the sheet's short side)
    assert IS.long_axes_angle_deg(spat, _fp(SHEET, (th.zeros(3), G.quat_from_yaw(math.radians(93.88))))) == pytest.approx(58.22, abs=0.01)
    yaws = IS.target_aligned_partner_yaws(spat, sheet, across=False)
    assert [round(math.degrees(y), 3) for y in yaws] == [-27.9, 152.1]
    assert all(IS.long_axes_angle_deg(spat, _fp(SHEET, (th.zeros(3), G.quat_from_yaw(y)))) < 1e-3 for y in yaws)
    across = IS.target_aligned_partner_yaws(spat, sheet, across=True)
    assert across[:2] == yaws and [round(IS.long_axes_angle_deg(spat, _fp(SHEET, (th.zeros(3), G.quat_from_yaw(y)))), 3) for y in across[2:]] == [90.0, 90.0]


def test_place_partner_lays_the_sheet_along_the_spatula():
    """With the key every drawn sheet has its long side along the spatula's long axis (tier aligned: never across), at any target yaw; without
    it the yaws are random."""
    s = _under({"partner_aligned_to_target": True})
    manip = _manip(s)
    s.region_side = 1.0
    rng_t = np.random.default_rng(7)
    for seed in range(10):
        t_pose = (th.tensor([1.31, -0.35, 0.76]), G.quat_from_yaw(float(rng_t.uniform(-math.pi, math.pi))))
        p_pose = s.place_partner(np.random.default_rng(seed), t_pose, manip, None)
        assert IS.long_axes_angle_deg(_fp(SPATULA, t_pose), _fp(SHEET, p_pose)) < 0.01 and s.partner_draw["long_axes_deg"] < 0.01
        assert s.partner_draw["side"] == "left"
    # any_yaw: the across yaws are tried too; the yaw list repeats the shuffled aligned yaws up to PARTNER_YAW_DRAWS tries
    t_pose = (th.tensor([1.31, -0.35, 0.76]), G.quat_from_yaw(0.4))
    assert len(s.aligned_partner_yaws(t_pose, np.random.default_rng(0))) == IS.PARTNER_YAW_DRAWS
    wide = _under({"partner_aligned_to_target": True}, tier="any_yaw")
    angles = {round(IS.long_axes_angle_deg(_fp(SPATULA, t_pose), _fp(SHEET, (th.zeros(3), G.qmul(G.quat_from_yaw(y), wide.rest["partner"]))))) for y in
              wide.aligned_partner_yaws(t_pose, np.random.default_rng(0))}
    assert angles == {0, 90}
    plain = _under()
    plain.region_side = 1.0
    seen = [IS.long_axes_angle_deg(_fp(SPATULA, t_pose), _fp(SHEET, plain.place_partner(np.random.default_rng(k), t_pose, manip, None))) for k in range(10)]
    assert max(seen) > 10.0 and "long_axes_deg" not in plain.partner_draw


def test_goal_centres_the_spatula_footprint_on_the_sheet():
    """The goal keeps the target's init yaw (parallel to the sheet when the sheet was drawn along it); with the key the target's footprint, not
    its origin, is centred on the sheet's - equal margins at both ends of the aligned long axes."""
    yaw = 0.3
    t_pose = (th.tensor([1.31, -0.35, 0.761]), G.quat_from_yaw(yaw))
    p_pose = (th.tensor([1.30, -0.02, 0.7503]), G.quat_from_yaw(yaw - math.pi / 2))
    for on in (True, False):
        s = _under({"partner_aligned_to_target": True} if on else {})
        s.sidecar = types.SimpleNamespace(stance=lambda name, m=_manip(s): m)
        s.release_surface = lambda xy, pose, top: None
        goal = s.goal_pose(t_pose, p_pose)
        assert G.yaw_of(goal[1]) == pytest.approx(yaw, abs=1e-5) and float(goal[0][2]) == pytest.approx(0.7503 + 0.0005 + 0.003, abs=1e-5)  # 3 mm over the sheet
        t_c, p_c = IS.min_area_rect(_fp(SPATULA, goal))[3], IS.min_area_rect(_fp(SHEET, p_pose))[3]
        gap = float(th.linalg.norm(t_c - p_c))
        if on:
            assert gap < 1e-4  # 1.05 cm of sheet beyond each end of the 0.269 m spatula
        else:
            assert gap == pytest.approx(0.0165, abs=1e-4) and goal[0][:2].tolist() == pytest.approx(p_c.tolist(), abs=1e-4)  # the origin centred


def test_partner_aligned_to_target_is_a_spec_error_for_touching_and_meta_links():
    assert _sampler().partner_aligned_to_target() is False and _under({"partner_aligned_to_target": True}).partner_aligned_to_target() is True
    with pytest.raises(Reject) as r:
        _sampler("touching", gen={"partner_aligned_to_target": True}, pair={"g_NT": 0.05}).partner_aligned_to_target()
    assert r.value.reason == "spec"
    with pytest.raises(Reject) as r:
        _sampler(gen={"partner_aligned_to_target": True, "partner_on_meta_link": "heatsource"}).partner_aligned_to_target()
    assert r.value.reason == "spec"


# ---------------------------------------------------------------------------------------------------------------- flat top


def _slab(x0, x1, y0, y1, z):
    a, b, c, d = [x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]
    return [[a, b, c], [a, c, d]]


def _two_units(seam_y=0.0, gap=0.004, z=0.75, rim=True):
    """Two table units over x 1.0-1.6 (the synthetic top), y -1.5..1.5, a ``gap`` wide seam at ``seam_y`` with the frame 0.35 m below seen through
    it, and (``rim``) a 2 cm raised lip, 1 cm high, along the back edge."""
    tri = _slab(1.0, 1.6, -1.5, seam_y - gap / 2, z) + _slab(1.0, 1.6, seam_y + gap / 2, 1.5, z) + _slab(1.0, 1.6, -1.5, 1.5, z - 0.35)
    if rim:
        tri += _slab(1.58, 1.6, -1.5, 1.5, z + 0.01)
    return np.array(tri, dtype=float)


def _flat(tri, res=0.002, tol=0.003, z_hint=0.75):
    x0, y0 = 0.99, -1.51
    nx, ny = int(math.ceil(0.62 / res)) + 1, int(math.ceil(3.02 / res)) + 1
    zbuf, nbuf = IS.raster_from_above(tri, x0, y0, nx, ny, res)
    return IS.FlatTop.from_raster(x0, y0, res, zbuf, nbuf, z_hint, tol, 0.99)


def test_raster_from_above_sees_the_seam_and_the_rim():
    flat = _flat(_two_units())
    assert flat.z_ref == pytest.approx(0.75, abs=1e-6)
    ys = flat.y0 + (np.arange(flat.flat.shape[1]) + 0.5) * flat.res
    xs = flat.x0 + (np.arange(flat.flat.shape[0]) + 0.5) * flat.res
    i = int(np.argmin(np.abs(xs - 1.3)))
    seam = np.abs(ys) < 0.002
    assert seam.any() and not flat.flat[i, seam].any() and np.allclose(flat.z[i, seam], 0.40)  # the frame below is what is seen in the seam
    assert flat.flat[i, (np.abs(ys) > 0.003) & (np.abs(ys) < 1.49)].all()
    assert not flat.flat[(xs > 1.581) & (xs < 1.599)].any() and not flat.flat[xs > 1.601].any()  # the lip, and beyond the top
    # a sheet 2 cm from the seam keeps 2 cm (to the 2 mm cells' centres), one across it does not, one on the rim neither
    sheet = _fp(SHEET, (th.tensor([1.30, -0.165, 0.0]), G.quat_from_yaw(0.0)))  # y -0.31 .. -0.02
    assert flat.clearance(sheet, 0.05) == pytest.approx(0.019, abs=0.0015) and flat.heights(sheet) == pytest.approx((0.75, 0.75))
    assert flat.clearance(_fp(SHEET, (th.tensor([1.30, 0.05, 0.0]), G.quat_from_yaw(0.3))), 0.05) < 0.0
    assert flat.clearance(_fp(SHEET, (th.tensor([1.48, -0.6, 0.0]), G.quat_from_yaw(0.0))), 0.05) < 0.0  # x up to 1.585: on the lip
    assert flat.clearance(_fp(SHEET, (th.tensor([1.30, -0.6, 0.0]), G.quat_from_yaw(0.0))), 0.05) == pytest.approx(0.05)  # nothing within 5 cm
    # a groove 2 mm deep is not flat with the default 3 mm tolerance only when it is deeper than that
    groove = np.array(_slab(1.0, 1.6, -1.5, -0.002, 0.75) + _slab(1.0, 1.6, 0.002, 1.5, 0.75) + _slab(1.0, 1.6, -0.002, 0.002, 0.748), dtype=float)
    assert _flat(groove).flat[i, seam].all() and not _flat(groove, tol=0.001).flat[i, seam].any()


def test_partner_draw_keeps_the_sheet_off_the_seam(monkeypatch):
    """The flat mask removes the gaps whose sheet comes within 1 cm of the seam; every drawn sheet keeps more than that (the template built
    the raster from the support's visual meshes); without the key some draws cover the seam."""
    tri = _two_units(seam_y=-0.02)
    support = types.SimpleNamespace(name="table_x_0", aabb=(th.tensor([1.0, -1.5, 0.0]), th.tensor([1.6, 1.5, 0.76])),
                                    links={"base_link": types.SimpleNamespace(visual_meshes={"m": types.SimpleNamespace(
                                        points_in_world_frame=th.tensor(tri.reshape(-1, 3), dtype=th.float32), faces=th.arange(3 * len(tri)).reshape(-1, 3))})})
    monkeypatch.setattr(IS.IORSSmall, "support", property(lambda self: support))
    s = _under({"partner_aligned_to_target": True, "partner_flat_top": True})
    s.world.sync_render_transforms = lambda: None
    s.flat_top = s.build_flat_top()
    facts = s.template_facts["partner_flat_top"]
    assert facts["z_ref"] == pytest.approx(0.75) and facts["margin_m"] == 0.01 and facts["n_triangles"] == len(tri) - 2  # the frame is 35 cm down
    manip = _manip(s)
    s.region_side = 1.0
    t_pose = (th.tensor([1.31, -0.30, 0.76]), G.quat_from_yaw(math.pi / 2))  # the spatula along the edge; the sheet's gaps span the seam
    for seed in range(12):
        fp = _fp(SHEET, s.place_partner(np.random.default_rng(seed), t_pose, manip, None))
        # the seam's cells are centred at y -0.021 / -0.019 (2 mm raster): the sheet ends more than 1 cm from them
        assert float(fp[:, 1].min()) > -0.019 + 0.01 or float(fp[:, 1].max()) < -0.021 - 0.01
        assert s.partner_draw["flat_top"]["clearance_m"] > 0.01 and s.partner_draw["flat_top"]["seen_height_range_mm"] == pytest.approx([0.0, 0.0], abs=1e-3)
    plain = _under({"partner_aligned_to_target": True})
    plain.region_side = 1.0
    across = 0
    for seed in range(12):
        fp = _fp(SHEET, plain.place_partner(np.random.default_rng(seed), t_pose, manip, None))
        across += int(float(fp[:, 1].min()) < -0.02 < float(fp[:, 1].max()))
    assert across > 0 and "flat_top" not in plain.partner_draw
    # a top that is a seam everywhere the sheet may go: no gap passes, the attempt's partner placement is rejected
    s.flat_top.flat[:] = False
    with pytest.raises(Reject) as e:
        s.place_partner(np.random.default_rng(0), t_pose, manip, None)
    assert e.value.reason == "partner_placement" and "flat 0/" in str(e.value)
