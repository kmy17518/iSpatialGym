"""of-v2maps 2026-11-07 (owner prompt G): the support-first map lookup (tabletop.support_first_cells / support_first_keep)."""

import math
from types import SimpleNamespace

import numpy as np
import torch as th

from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.v2 import tabletop as TT


class _MapView:
    """A map view: base-frame lattice cells (x = 0.35 + k 0.08) at one trunk / stand-off."""

    def __init__(self, cells):
        self.meta = {"mode": "map_view", "lattice": 0.08, "stand_offs": [0.35]}
        self._cells = np.asarray(cells, dtype=float)

    def band(self, translate, tilt, stand_off=None):
        return self._cells

    to_base = RB.ReachBands.to_base
    to_world = RB.ReachBands.to_world


def _stance():
    return SimpleNamespace(translate=0.5, tilt=0.0, stand_off=0.35, base_pose2d=(0.0, 0.0, 0.0))


def test_the_narrow_strip_between_lattice_rows_reads_the_nearest_row():
    # nextto_04: the eroded 0.36 m bench leaves focus depths 0.174-0.186 m (base x 0.524-0.536 at stand-off 0.35); lattice rows 0.51 / 0.59
    band = _MapView([[0.51, y] for y in (-0.08, 0.0, 0.08)] + [[0.59, y] for y in (-0.08, 0.0, 0.08)])
    raster = th.tensor([[x, y] for x in (0.524, 0.53, 0.536) for y in np.arange(-0.10, 0.101, 0.02)], dtype=th.float32)
    cells, facts = TT.support_first_cells(band, _stance(), raster)
    assert facts["n_kept"] == len(raster) and len(cells) == len(raster)  # every strip cell lies within 0.04 of a feasible row point
    assert 0.0 < facts["dist_mean_m"] <= 0.04 and facts["agree_mean"] > 0
    # today's lookup: no lattice point inside the strip
    assert not any(0.524 <= float(c[0]) <= 0.536 for c in band._cells)


def test_no_feasible_lattice_point_within_tolerance_keeps_nothing_and_widening_finds_one():
    band = _MapView([[0.67, 0.0]])  # the only feasible point 0.13 m deeper than the strip
    raster = th.tensor([[0.53, 0.0], [0.60, 0.0]], dtype=th.float32)
    cells, facts = TT.support_first_cells(band, _stance(), raster)
    assert len(cells) == 0 and facts["n_kept"] == 0
    cells_w, _ = TT.support_first_cells(band, _stance(), raster, tol_frac=TT.SUPPORT_FIRST_WIDEN_FRAC)
    assert [round(float(c[0]), 2) for c in cells_w] == [0.60]


def test_support_first_keep_is_off_by_default_and_ignores_scene_bands(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_MAP_SUPPORT_FIRST", raising=False)
    monkeypatch.delenv("ISPATIALGYM_MAP_LATTICE_DIAG", raising=False)
    keep = [th.tensor([0.5, 0.0])]
    out, arms = TT.support_first_keep(SimpleNamespace(), _stance(), None, 0.17, _MapView([[0.51, 0.0]]), {}, keep, {})
    assert out is keep
    monkeypatch.setenv("ISPATIALGYM_MAP_SUPPORT_FIRST", "1")
    scene_band = SimpleNamespace(meta={"mode": "band"})
    out, arms = TT.support_first_keep(SimpleNamespace(), _stance(), None, 0.17, scene_band, {}, keep, {})
    assert out is keep  # a legacy scene band is untouched


def test_v2_rest_h_root_placement_is_opt_in(monkeypatch):
    # of-v2maps 2026-11-07 (the isg1020 spec: z = support + h_root, no 3 mm, no recomputed bottom offset)
    s = SimpleNamespace(world=SimpleNamespace(bottom_offset=lambda hull, q: 0.02), _v2_rest_h={(0.0, 0.0, 0.0, 1.0): 0.0185, (-0.0, -0.0, -0.0, -1.0): 0.0185})
    q = th.tensor([0.0, 0.0, 0.0, 1.0])
    monkeypatch.delenv("ISPATIALGYM_V2_REST_H_ROOT", raising=False)
    p, _ = TT.object_pose_on(s, th.zeros(4, 3), q, (0.1, 0.2), 0.3, 0.75)
    assert abs(float(p[2]) - (0.75 + 0.02 + TT.Z_OFFSET)) < 1e-6
    monkeypatch.setenv("ISPATIALGYM_V2_REST_H_ROOT", "1")
    p, _ = TT.object_pose_on(s, th.zeros(4, 3), q, (0.1, 0.2), 0.3, 0.75)
    assert abs(float(p[2]) - (0.75 + 0.0185)) < 1e-6
    p, _ = TT.object_pose_on(s, th.zeros(4, 3), th.tensor([0.0, 0.0, 0.7071, 0.7071]), (0.1, 0.2), 0.3, 0.75)  # an unregistered rest
    assert abs(float(p[2]) - (0.75 + 0.02 + TT.Z_OFFSET)) < 1e-6
