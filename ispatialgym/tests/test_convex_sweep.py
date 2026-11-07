"""``convex_sweep.first_touch`` (owner 2026-11-07, configuration_matching-attachment_04 option B: the attach hover from the swept
first touch) on synthetic convex pieces: a lid over a pot whose loop handles lie outside the lid's radius (the first touch is the
rim, not the handles), a part offset sideways, parallel faces, a peg into a ring, crossed ridges (an edge-edge contact), random
hulls against an independent Minkowski-difference ray cast, degenerate pieces. The pot / lid / fingertip shapes are shared with
``test_attach_hover_swept.py``."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ispatialgym.oracle import convex_sweep as CS

UP = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------------------------------------------------- shapes


def _box(x0, x1, y0, y1, z0, z1):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=np.float64)


def _cyl(r, z0, z1, n=24, cx=0.0, cy=0.0):
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    ring = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)
    return np.concatenate([np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z1)]])


def _sector(r0, r1, a0, a1, z0, z1, n=3):
    a = np.linspace(a0, a1, n)
    pts = [[r * math.cos(t), r * math.sin(t), z] for r in (r0, r1) for t in a for z in (z0, z1)]
    return np.array(pts, dtype=np.float64)


def _rotz(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    return pts @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T


def _pieces(arrs):
    return [CS.convex_piece(a) for a in arrs]


#: the stockpot / lid of configuration_matching-attachment_04 in round numbers (world, the pot's base at z 0.85): the wall's rim top
#: 64.6 mm up, four loop handles 89.8 mm up at r 119-145 mm on the diagonals (their xy boxes reach inside the lid's box, the pieces
#: stay outside its 111.3 mm radius), the lid seated 12.3 mm into the rim (the attachment point below where it rests)
BASE_Z, RIM_TOP, HANDLE_TOP, LID_R = 0.85, 0.85 + 0.0646, 0.85 + 0.0898, 0.1113
SEAT_BOTTOM = RIM_TOP - 0.0123


def _pot():
    wall = [_sector(0.104, 0.116, k * math.pi / 8, (k + 1) * math.pi / 8, BASE_Z, RIM_TOP) for k in range(16)]
    bottom = [_cyl(0.104, BASE_Z, BASE_Z + 0.005)]
    handles = [_rotz(_box(0.119, 0.145, -0.012, 0.012, BASE_Z + 0.05, HANDLE_TOP), math.pi / 4 + k * math.pi / 2) for k in range(4)]
    return wall + bottom + handles


def _lid(dz=0.0, dx=0.0):
    disc = _cyl(LID_R, SEAT_BOTTOM, SEAT_BOTTOM + 0.008, n=32, cx=dx)
    knob = _cyl(0.012, SEAT_BOTTOM + 0.008, SEAT_BOTTOM + 0.028, n=12, cx=dx)
    return [disc + [0.0, 0.0, dz], knob + [0.0, 0.0, dz]]


def _fingers(dz=0.0):
    return [_box(0.012, 0.020, -0.008, 0.008, SEAT_BOTTOM + 0.012 + dz, SEAT_BOTTOM + 0.05 + dz),
            _box(-0.020, -0.012, -0.008, 0.008, SEAT_BOTTOM + 0.012 + dz, SEAT_BOTTOM + 0.05 + dz)]


def _mink_first_touch(A: np.ndarray, B: np.ndarray, u: np.ndarray):
    """An independent exact answer: the largest t with t u in conv(B - A) (a ray cast through the Minkowski difference's hull)."""
    from scipy.spatial import ConvexHull

    h = ConvexHull((B[None, :, :] - A[:, None, :]).reshape(-1, 3))
    n, off = h.equations[:, :3], h.equations[:, 3]
    k = n @ u
    pos, neg, par = k > 1e-12, k < -1e-12, np.abs(k) <= 1e-12
    if par.any() and (off[par] > 1e-12).any():
        return None
    hi, lo = (-off[pos] / k[pos]).min(), (-off[neg] / k[neg]).max()
    return hi if lo <= hi + 1e-9 else None


# ---------------------------------------------------------------------------------------------------------------------- the sweep


def test_lid_over_a_pot_first_touches_the_rim_not_the_handles_outside_its_radius():
    pot = _pieces(_pot())
    lid = _pieces(_lid())
    res = CS.first_touch(lid, pot, UP)
    assert res["t"] == pytest.approx(0.0123, abs=1e-9)  # the lid's bottom on the rim top
    assert res["pair"][0] == 0 and res["pair"][1] < 16  # the disc on a wall piece
    # the handles' xy boxes meet the lid's: the box rule's top would be theirs, 25.2 mm above the rim
    lo, hi = _lid()[0][:, :2].min(axis=0) - 0.01, _lid()[0][:, :2].max(axis=0) + 0.01
    under = [p for p in _pot() if (p[:, :2].min(axis=0) <= hi).all() and (p[:, :2].max(axis=0) >= lo).all()]
    assert max(float(p[:, 2].max()) for p in under) == pytest.approx(HANDLE_TOP)
    assert res["pairs_overlapping"] > res["pairs_tested"] >= 1  # the bound stopped the search
    # the lid already lifted 5 cm: the first touch is 5 cm lower along the axis
    assert CS.first_touch(_pieces(_lid(dz=0.05)), pot, UP)["t"] == pytest.approx(0.0123 - 0.05, abs=1e-9)


def test_a_part_offset_sideways_meets_only_what_lies_under_it():
    fixed = _pieces([_box(-0.02, 0.02, -0.02, 0.02, 0.0, 0.05), _box(0.05, 0.08, -0.02, 0.02, 0.0, 0.10)])
    part = lambda dx: _pieces([_box(-0.02 + dx, 0.02 + dx, -0.02, 0.02, 0.20, 0.22)])  # noqa: E731
    assert CS.first_touch(part(0.0), fixed, UP)["t"] == pytest.approx(0.05 - 0.20)  # over the low block only
    res = CS.first_touch(part(0.04), fixed, UP)  # 1 cm over the tall block, its edge on the low block's
    assert res["t"] == pytest.approx(0.10 - 0.20) and res["pair"] == (0, 1)
    res = CS.first_touch(part(0.15), fixed, UP)  # beside both: never meets
    assert res["t"] is None and res["pair"] is None and res["pairs_overlapping"] == 0


def test_parallel_faces_touch_at_the_gap_whatever_the_turn_about_the_axis():
    fixed = _pieces([_box(-0.05, 0.05, -0.05, 0.05, 0.0, 0.03)])
    for ang in (0.0, math.pi / 4, 0.3):
        part = _pieces([_rotz(_box(-0.03, 0.03, -0.03, 0.03, 0.07, 0.09), ang)])
        assert CS.first_touch(part, fixed, UP)["t"] == pytest.approx(0.03 - 0.07, abs=1e-12)
    # a face parallel to the axis (side by side, touching only along it): the touch spans the whole overlap, its top end counts
    part = _pieces([_box(0.05, 0.08, -0.01, 0.01, 0.02, 0.04)])
    assert CS.first_touch(part, fixed, UP)["t"] == pytest.approx(0.03 - 0.02, abs=1e-9)


def test_a_peg_narrower_than_the_opening_goes_in_to_the_bottom():
    neck = [_sector(0.0095, 0.014, k * math.pi / 8, (k + 1) * math.pi / 8, 0.0, 0.05) for k in range(16)] + [_cyl(0.0095, 0.0, 0.002, n=16)]
    peg = _pieces([_cyl(0.009, 0.06, 0.09, n=16)])
    assert CS.first_touch(peg, _pieces(neck), UP)["t"] == pytest.approx(0.002 - 0.06, abs=1e-9)
    fat = _pieces([_cyl(0.012, 0.06, 0.09, n=16)])  # wider than the opening: on the neck's top
    assert CS.first_touch(fat, _pieces(neck), UP)["t"] == pytest.approx(0.05 - 0.06, abs=1e-9)


def test_crossed_ridges_meet_edge_to_edge():
    def prism(d, down, z):
        side = np.cross(d, UP)
        return np.array([s * d + o for s in (-0.1, 0.1) for o in (np.zeros(3), side * 0.02 + [0, 0, 0.02 if down else -0.02],
                                                                    -side * 0.02 + [0, 0, 0.02 if down else -0.02])]) + [0.003, 0.004, z]

    A, B = prism(np.array([1.0, 0, 0]), True, 0.05), prism(np.array([0, 1.0, 0]), False, 0.0)
    a, b = CS.convex_piece(A), CS.convex_piece(B)
    assert CS.pair_first_touch(a, b, UP) == pytest.approx(-0.05, abs=1e-12)  # no vertex of either lies on a face of the other
    no_edges = CS.ConvexPiece(a.verts, a.normals, a.offsets, a.e0[:0], a.e1[:0], a.edge_faces[:0])
    assert CS.pair_first_touch(no_edges, b, UP) is None


def test_the_sweep_matches_a_minkowski_difference_ray_cast():
    rng = np.random.default_rng(7)
    for _ in range(60):
        A = rng.normal(size=(int(rng.integers(4, 30)), 3)) * rng.uniform(0.01, 0.08, 3) + rng.normal(size=3) * 0.04
        B = rng.normal(size=(int(rng.integers(4, 30)), 3)) * rng.uniform(0.01, 0.08, 3)
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        a, b = CS.convex_piece(A), CS.convex_piece(B)
        got, want = CS.pair_first_touch(a, b, u), _mink_first_touch(a.verts, b.verts, u)
        assert (got is None) == (want is None)
        if got is not None:
            assert got == pytest.approx(want, abs=1e-10)


def test_degenerate_pieces_are_skipped_or_joggled():
    assert CS.convex_piece(np.zeros((3, 3))) is None
    flat = CS.convex_piece(_box(-0.01, 0.01, -0.01, 0.01, 0.0, 0.0))  # a zero-thickness plate: joggled
    assert flat is not None
    res = CS.first_touch([None] + _pieces([_box(-0.01, 0.01, -0.01, 0.01, 0.05, 0.06)]), [flat, None], UP)
    assert res["t"] == pytest.approx(-0.05, abs=1e-6) and res["pair"] == (1, 0)
