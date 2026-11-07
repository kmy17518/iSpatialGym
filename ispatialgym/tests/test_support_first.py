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


def test_v2_rest_h_root_placement_by_entry_and_rest(monkeypatch):
    # of-v2maps 2026-11-07 (the isg1020 spec: z = support + h_root, no 3 mm, no recomputed bottom offset); prompt I item 2: the default for v2
    # rests, resolved by the object's entry + rest_id - two objects at the identity rest keep their own h_root (the former quaternion
    # registry gave both the last one's), and a rolled rest keeps the bottom-offset rule
    from ispatialgym.sampling import grasp_library as GL

    rest = lambda h: SimpleNamespace(stable=True, h_root=h, quat_xyzw=[0.0, 0.0, 0.0, 1.0])
    libs = {"cup__k": SimpleNamespace(path="/v2/models/cup__s1/library.json", rests={"r0": rest(0.0185)}),
            "box__k": SimpleNamespace(path="/v2/models/box__s1/library.json", rests={"r0": rest(0.0610)})}
    monkeypatch.setattr(GL, "v2_entry", lambda key: libs.get(key))
    ht, hp = th.zeros(4, 3), th.ones(4, 3)
    s = SimpleNamespace(world=SimpleNamespace(bottom_offset=lambda hull, q: 0.02), hull={"target": ht, "partner": hp}, grasp_key="cup__k",
                        choice=SimpleNamespace(models={"partner": SimpleNamespace(grasp_key="box__k", category="box", model="m")}))
    q = th.tensor([0.0, 0.0, 0.0, 1.0])
    monkeypatch.delenv("ISPATIALGYM_V2_REST_H_ROOT", raising=False)
    p, _ = TT.object_pose_on(s, ht, q, (0.1, 0.2), 0.3, 0.75)
    assert abs(float(p[2]) - (0.75 + 0.0185)) < 1e-6  # the default
    p, _ = TT.object_pose_on(s, hp, -q, (0.1, 0.2), 0.3, 0.75)
    assert abs(float(p[2]) - (0.75 + 0.0610)) < 1e-6  # the partner's own entry (and -q is the same rest)
    assert TT.v2_rest_info(s, hp)["entry"] == "box__s1" and TT.v2_rest_info(s, hp)["rest_id"] == "r0"
    p, _ = TT.object_pose_on(s, ht, th.tensor([0.0, 0.0, 0.7071, 0.7071]), (0.1, 0.2), 0.3, 0.75)  # not the measured rest
    assert abs(float(p[2]) - (0.75 + 0.02 + TT.Z_OFFSET)) < 1e-6
    p, _ = TT.object_pose_on(s, th.full((4, 3), 2.0), q, (0.1, 0.2), 0.3, 0.75)  # a hull of no role (a distractor)
    assert abs(float(p[2]) - (0.75 + 0.02 + TT.Z_OFFSET)) < 1e-6
    monkeypatch.setenv("ISPATIALGYM_V2_REST_H_ROOT", "0")
    p, _ = TT.object_pose_on(s, ht, q, (0.1, 0.2), 0.3, 0.75)
    assert abs(float(p[2]) - (0.75 + 0.02 + TT.Z_OFFSET)) < 1e-6


def test_v2_only_blocks_every_legacy_band(monkeypatch):
    # of-v2maps 2026-11-07 (owner: no legacy band): ReachBands.load / BandComputer refuse under ISPATIALGYM_V2_ONLY=1
    import pytest

    from ispatialgym.sampling.sim_utils import Reject

    monkeypatch.setenv("ISPATIALGYM_V2_ONLY", "1")
    with pytest.raises(Reject) as e:
        RB.ReachBands.load("house_single_floor", "breakfast_table_rhjoby_0", "apple", "apple__x")
    assert e.value.reason == "legacy_band_blocked"
    with pytest.raises(Reject):
        RB.BandComputer(None, None)
    monkeypatch.delenv("ISPATIALGYM_V2_ONLY")
    with pytest.raises(FileNotFoundError):  # off: the former behaviour (a cache miss)
        RB.ReachBands.load("no_such_scene", "no_such_support", "apple", "apple__x", dirs=[])


def test_rest_z_applies_h_root_to_a_yawed_v2_rest_only(monkeypatch):
    # prompt I item 2: the partner placements (place_partner / partner_pose_at / pose_centred_on) take the full orientation R_z(yaw) ∘ rest
    import math

    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling import grasp_library as GL

    tilt = [0.0, 0.7071068, 0.0, 0.7071068]  # a side rest
    monkeypatch.setattr(GL, "v2_entry", lambda key: SimpleNamespace(path="/v2/models/m/library.json",
                                                                    rests={"r0": SimpleNamespace(stable=True, h_root=0.033, quat_xyzw=tilt)}))
    hp = th.ones(4, 3)
    s = SimpleNamespace(world=SimpleNamespace(bottom_offset=lambda hull, q: 0.05), hull={"partner": hp},
                        choice=SimpleNamespace(models={"partner": SimpleNamespace(grasp_key="m__k", category="m", model="m")}))
    monkeypatch.delenv("ISPATIALGYM_V2_REST_H_ROOT", raising=False)
    q = C.qmul(C.quat_from_yaw(1.3), th.tensor(tilt))
    assert abs(TT.rest_z(s, hp, q, 0.7) - 0.733) < 1e-6
    rolled = C.qmul(C.quat_from_axis_angle(th.tensor([1.0, 0.0, 0.0]), math.radians(10.0)), th.tensor(tilt))
    assert abs(TT.rest_z(s, hp, rolled, 0.7) - (0.7 + 0.05 + TT.Z_OFFSET)) < 1e-6  # a leaned / rolled rest: the bottom-offset rule


# ------------------------------------------------------------------------------------------------------------------
# i7-unblock 2026-11-07 (owner decision 21:40): support-first as a generic automatic fallback per support
# ------------------------------------------------------------------------------------------------------------------


class _Top:
    """A support top raster (2 cm, axis-aligned): ``nx`` x ``ny`` free cells from the origin; SupportTop's contains / local_axes."""

    def __init__(self, name, nx, ny, res=0.02):
        self.name, self.resolution, self.yaw = name, res, 0.0
        self.xs = th.arange(nx, dtype=th.float32) * res
        self.ys = th.arange(ny, dtype=th.float32) * res
        self.free = th.ones(nx, ny, dtype=th.bool)

    def cells(self, free_only=True):
        idx = th.nonzero(self.free)
        return th.stack([self.xs[idx[:, 0]], self.ys[idx[:, 1]]], dim=1)

    def local_axes(self):
        return th.tensor([1.0, 0.0]), th.tensor([0.0, 1.0])

    def contains(self, xy, margin=0.0):
        pts = [th.as_tensor(xy, dtype=th.float32)]
        if margin > 0:
            pts += [pts[0] + th.tensor([dx, dy]) for dx, dy in ((margin, 0), (-margin, 0), (0, margin), (0, -margin), (margin, margin),
                                                                  (-margin, -margin), (margin, -margin), (-margin, margin))]
        for p in pts:
            i, j = int(round(float(p[0]) / self.resolution)), int(round(float(p[1]) / self.resolution))
            if i < 0 or j < 0 or i >= len(self.xs) or j >= len(self.ys) or not bool(self.free[i, j]):
                return False
        return True


BENCH = ("bench_pfjnnj_0", 19, 58)  # nextto_04: the 0.36 x 1.14 m top, 1102 cells
TABLE = ("breakfast_table_rhjoby_0", 58, 145)  # a 1.16 x 2.90 m table top


def _sampler(top, gen=None):
    return SimpleNamespace(top=top, kit=SimpleNamespace(stats={}), gen=lambda k, d=None: (gen or {}).get(k, d))


def _run(monkeypatch, top, half_extent, keep_first, keep_sf=("sf",), gen=None, env=None):
    """phase2_manipulation_stance with a stub Phase 2: it calls support_first_keep once with ``keep_first`` (map-first) or reports the
    support-first cells; returns (output, the active flags of its calls, the kit counters)."""
    monkeypatch.delenv("ISPATIALGYM_MAP_SUPPORT_FIRST", raising=False)
    monkeypatch.delenv("ISPATIALGYM_MAP_LATTICE_DIAG", raising=False)
    if env is not None:
        monkeypatch.setenv("ISPATIALGYM_MAP_SUPPORT_FIRST", env)
    s = _sampler(_Top(*top), gen)
    view = _MapView([[0.51, 0.0]])
    calls = []

    def impl(s_, init, he, **kw):
        active = bool(s_.__dict__.get("_sf_active"))
        calls.append(active)
        if active:
            return list(keep_sf)
        keep, _ = TT.support_first_keep(s_, _stance(), s_.top, he + 0.02, view, {}, [th.tensor([0.1, 0.1])] * keep_first, {})
        return ["map"] if keep else []

    monkeypatch.setattr(TT, "_phase2_manipulation_stance", impl)
    out = TT.phase2_manipulation_stance(s, SimpleNamespace(edge=None), half_extent, bands=view)
    return out, calls, s.kit.stats, s


def test_the_narrow_bench_triggers_support_first_geometrically(monkeypatch):
    out, calls, stats, s = _run(monkeypatch, BENCH, 0.154, keep_first=0)
    assert calls == [True] and out == ["sf"] and stats == {"map_support_first.geometric": 1}
    geo = TT.support_first_geometric(s, s.top, 0.174, {"_": _MapView([[0.51, 0.0]])})
    assert geo["support"] == "bench_pfjnnj_0" and geo["usable_m"] == 0.02 and geo["lattice_m"] == 0.08 and geo["raster_cells"] == 40
    assert s._sf_active is None  # restored after the call


def test_a_normal_tabletop_keeps_map_first_unchanged(monkeypatch):
    out, calls, stats, _ = _run(monkeypatch, TABLE, 0.10, keep_first=3)
    assert calls == [False] and out == ["map"] and stats == {}
    assert TT.support_first_geometric(_sampler(_Top(*TABLE)), _Top(*TABLE), 0.12, {"_": _MapView([[0.51, 0.0]])}) is None


def test_zero_map_first_cells_fall_back_once_and_are_counted(monkeypatch):
    out, calls, stats, _ = _run(monkeypatch, TABLE, 0.10, keep_first=0)
    assert calls == [False, True] and out == ["sf"] and stats == {"map_support_first.fallback": 1}
    # map-first cells but no survivor: no fallback (the cells were there; the camera / reach rules rejected them)
    out, calls, stats, _ = _run(monkeypatch, TABLE, 0.10, keep_first=2, keep_sf=())
    assert calls == [False] and stats == {}


def test_the_override_forces_support_first_on_or_off(monkeypatch):
    out, calls, stats, _ = _run(monkeypatch, BENCH, 0.154, keep_first=0, gen={"map_support_first": False})
    assert calls == [False] and out == [] and stats == {}  # forced off: neither trigger
    out, calls, stats, _ = _run(monkeypatch, TABLE, 0.10, keep_first=3, gen={"map_support_first": True})
    assert calls == [True] and stats == {}
    out, calls, stats, _ = _run(monkeypatch, TABLE, 0.10, keep_first=3, env="1")
    assert calls == [True]
    out, calls, stats, _ = _run(monkeypatch, BENCH, 0.154, keep_first=0, gen={"map_support_first": True}, env="0")
    assert calls == [False]  # the job switch wins
    monkeypatch.delenv("ISPATIALGYM_MAP_SUPPORT_FIRST")
    assert TT.support_first_tag(_sampler(None), _MapView([])) == TT.SUPPORT_FIRST_AUTO_RULE
    assert TT.support_first_tag(_sampler(None, {"map_support_first": True}), _MapView([])) is True
    assert TT.support_first_tag(_sampler(None), SimpleNamespace(meta={"mode": "band"})) is False


def test_a_scene_band_is_never_read_support_first(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_MAP_SUPPORT_FIRST", raising=False)
    s = _sampler(_Top(*BENCH))
    calls = []
    monkeypatch.setattr(TT, "_phase2_manipulation_stance", lambda s_, i, he, **kw: calls.append(s_.__dict__.get("_sf_active")) or [])
    TT.phase2_manipulation_stance(s, SimpleNamespace(edge=None), 0.154, bands=SimpleNamespace(meta={"mode": "band"}))
    assert calls == [None] and s.kit.stats == {}
