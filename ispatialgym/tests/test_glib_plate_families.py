"""i7-plate 2026-11-07: the opt-in candidate families (lip_shallow, rim_straddle), the gate dump and the owner-decision switches, on a
synthetic plate built like plate luhkiz's collision (rim sectors as convex hulls of a sloped shell - each hull fills its chord - and an
open centre, optionally a foot disk). CPU only, no simulator, no Behavior-1K asset."""

import json
import math
import os

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from ispatialgym.grasping import candidates as CA  # noqa: E402
from ispatialgym.grasping import object_geometry as OG  # noqa: E402
from ispatialgym.grasping import pipeline as PL  # noqa: E402

R_IN, R_OUT, Z_IN, Z_OUT, THICK = 0.077, 0.120, -0.005, 0.015, 0.004


def _plate(n_sectors: int = 7, floor: bool = False) -> OG.ObjectGeometry:
    """Rim sectors: hulls of a conical shell (top rising from Z_IN at R_IN to Z_OUT at R_OUT, THICK below it); ``floor``: a foot ring
    whose hull is a disk under the centre (luhkiz's piece #0)."""
    pieces = []
    span = 2 * math.pi / n_sectors
    for k in range(n_sectors):
        pts = []
        for a in np.linspace(k * span, (k + 1) * span, 7):
            for r in np.linspace(R_IN, R_OUT, 6):
                zt = Z_IN + (Z_OUT - Z_IN) * (r - R_IN) / (R_OUT - R_IN)
                pts += [(r * math.cos(a), r * math.sin(a), zt), (r * math.cos(a), r * math.sin(a), zt - THICK)]
        pieces.append(trimesh.convex.convex_hull(np.array(pts)))
    if floor:
        ring = [(0.082 * math.cos(a), 0.082 * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, 24, endpoint=False) for z in (-0.011, -0.006)]
        pieces.append(trimesh.convex.convex_hull(np.array(ring)))
    return OG.ObjectGeometry(info={"meta_links": {}}, pieces=pieces, union=trimesh.util.concatenate(pieces))


def _cfg(**kw) -> CA.CandidateConfig:
    from dataclasses import asdict

    return CA.CandidateConfig(**{**asdict(CA.CandidateConfig()), **kw})


# ------------------------------------------------------------------------------------------------ switches and overrides


def test_families_off_and_owner_switches_default_on(monkeypatch):
    # i7-switches 2026-11-07: the two owner-decision switches are the default for new builds (adopted after the count)
    monkeypatch.delenv("ISG_GLIB_GATE_DUMP", raising=False)
    c = CA.CandidateConfig()
    assert tuple(c.families) == () and not c.gate_dump and c.blade_contact_at_retreat and c.retreat_shift_exact
    monkeypatch.setenv("ISG_GLIB_GATE_DUMP", "1")
    assert CA.CandidateConfig().gate_dump
    assert set(CA.FAMILIES) == {"lip_shallow", "rim_straddle"}


def test_cand_overrides_apply_per_manifest_entry(monkeypatch):
    monkeypatch.setenv("ISG_GLIB_CAND_OVERRIDES", '{"n_candidates": 24000, "cone_deg": 60.0}')
    monkeypatch.delenv("ISG_GLIB_DENSIFY", raising=False)
    plate = {"id": "plate__luhkiz__s1", "cand_overrides": {"families": ["lip_shallow", "rim_straddle"], "n_candidates": 48000}}
    other = {"id": "bowl__x__s1"}
    cp, _ = PL.candidate_overrides(plate)
    co, _ = PL.candidate_overrides(other)
    assert list(cp.families) == ["lip_shallow", "rim_straddle"] and cp.n_candidates == 48000 and cp.cone_deg == 60.0  # the entry wins
    assert tuple(co.families) == () and co.n_candidates == 24000 and co.cone_deg == 60.0
    c0, _ = PL.candidate_overrides()  # the old call (no job) still works
    assert c0.n_candidates == 24000
    with pytest.raises(ValueError):
        PL.candidate_overrides({"cand_overrides": {"familes": ["lip_shallow"]}})


# ------------------------------------------------------------------------------------------------ the plate geometry helpers


def test_plate_axis_and_rim_profile_on_a_sector_plate():
    g = _plate()
    cx, cy, r_max = CA.plate_axis(g.union)
    assert abs(cx) < 0.003 and abs(cy) < 0.003 and abs(r_max - R_OUT) < 1e-3
    mid = math.pi / 7  # the middle of sector 0: its hull fills the chord, so the inner face sits inside R_IN there
    pr = CA.rim_profile(g, np.array([mid, 0.0]), np.array([0.0, 0.0]))
    assert pr["free"].all() and np.isfinite(pr["r_in"]).all() and np.isfinite(pr["r_out"]).all()
    assert (pr["r_in"] < pr["r_out"]).all()
    assert pr["r_in"][0] < pr["r_in"][1] + 1e-9  # chord fill: the mid-sector inner face is further in than at the sector's edge
    assert pr["r_out"][1] < R_OUT  # at z 0 the outer face is the shell's underside, inside the lip
    # a foot disk under the centre: the axis point inside it is not free (no inner face there)
    gf = _plate(floor=True)
    pf = CA.rim_profile(gf, np.array([mid]), np.array([-0.008]))
    assert not pf["free"][0] and np.isnan(pf["r_in"][0])


def test_rim_straddle_poses_are_vertical_radial_and_straddle_the_rim():
    g = _plate()
    cfg = _cfg(straddle_n_azimuth=14, straddle_n_heights=6, straddle_tilts_deg=(0.0, 15.0, -15.0), straddle_pad_offsets_m=(0.0, 0.014, 0.05))
    T, s0 = CA.straddle_family_poses(np.zeros((0, 4, 4)), g, cfg)
    assert len(T) and len(T) == len(s0)
    assert set(np.round(s0, 4)) == {0.0, 0.014}  # 0.05 puts the contact past tip_max_m: skipped
    R = T[:, :3, :3]
    assert np.allclose(np.einsum("nij,nik->njk", R, R), np.eye(3), atol=1e-9) and np.allclose(np.linalg.det(R), 1.0)
    tl = CA.tilt_deg(T)
    assert set(np.round(tl, 3)) <= {0.0, 15.0}
    c = T[:, :3, 3] + T[:, :3, 2] * (CA.CONTACT_Z + s0)[:, None]
    rad = np.c_[c[:, :2] / np.linalg.norm(c[:, :2], axis=1, keepdims=True), np.zeros(len(c))]
    tan = np.cross([0.0, 0.0, 1.0], rad)
    y, z = T[:, :3, 1], T[:, :3, 2]
    assert np.abs(np.einsum("ij,ij->i", y, tan)).max() < 1e-9 and np.abs(np.einsum("ij,ij->i", z, tan)).max() < 1e-9  # radial plane
    vert = np.isclose(tl, 0.0)
    assert np.allclose(np.abs(np.einsum("ij,ij->i", y[vert], rad[vert])), 1.0)  # vertical approach: closing exactly radial
    # the contact centre sits midway between the rim's inner and outer faces at its height: one finger inside, one outside
    th_ = np.arctan2(c[vert, 1], c[vert, 0])
    pr = CA.rim_profile(g, th_, c[vert, 2])
    r_c = np.linalg.norm(c[vert, :2], axis=1)
    assert np.allclose(r_c, 0.5 * (pr["r_in"] + pr["r_out"]), atol=1e-6)
    assert (pr["r_in"] < r_c).all() and (r_c < pr["r_out"]).all()


def _lip_pose(theta: float, r: float, z: float, tilt_deg: float) -> np.ndarray:
    """An eef pose whose pad contact (CONTACT_Z along the approach) is at (r, theta, z), approach tilted ``tilt_deg`` from down toward
    the centre, closing axis tangential."""
    rad = np.array([math.cos(theta), math.sin(theta), 0.0])
    t = math.radians(tilt_deg)
    zax = -math.cos(t) * np.array([0, 0, 1.0]) - math.sin(t) * rad
    y = np.cross([0, 0, 1.0], rad)
    x = np.cross(y, zax)
    c = r * rad + np.array([0, 0, z])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, zax, c - zax * CA.CONTACT_Z
    return T


def test_lip_shallow_takes_the_outer_lip_in_its_tilt_window_and_raises_the_eef():
    g = _plate()
    T_all = np.stack([_lip_pose(0.3, R_OUT - 0.004, 0.013, 20.0),   # lip, inside the top mode's window: only the raised copies
                      _lip_pose(1.3, R_OUT - 0.004, 0.013, 55.0),   # lip, in the top / side gap: every offset
                      _lip_pose(2.3, R_OUT - 0.004, 0.013, 80.0),   # lip, beyond lip_tilt_deg: out
                      _lip_pose(3.3, 0.090, 0.005, 20.0)])          # not the lip: out
    cfg = _cfg(lip_tilt_deg=(0.0, 70.0), lip_pad_offsets_m=(0.0, 0.01, 0.02, 0.05))
    T, s0 = CA.lip_family_poses(T_all, g, cfg)
    assert sorted(np.round(s0, 4).tolist()) == [0.0, 0.01, 0.01, 0.02, 0.02]  # 0.05: past tip_max_m
    assert np.isclose(CA.tilt_deg(T[s0 == 0.0]), 55.0).all()
    lip = {0.3: T_all[0], 1.3: T_all[1]}
    for t, s in zip(T, s0):
        c = t[:3, 3] + t[:3, 2] * (CA.CONTACT_Z + s)  # the contact point is kept, the eef moved back by the offset
        src = lip[round(math.atan2(c[1], c[0]), 6)]
        assert np.allclose(c, src[:3, 3] + src[:3, 2] * CA.CONTACT_Z, atol=1e-12) and np.allclose(t[:3, :3], src[:3, :3])
    # without the top mode the offset-0 copy inside its window is proposed too
    T2, s2 = CA.lip_family_poses(T_all, g, _cfg(modes=("side",), lip_pad_offsets_m=(0.0,)))
    assert len(T2) == 2 and (s2 == 0.0).all()


# ------------------------------------------------------------------------------------------------ retreat bookkeeping and the dump


def test_object_retreats_shift_as_written_and_exact():
    # a hand clear of the object only once its eef moved >= 9 mm back along the approach (the 5 mm try fails, the 10 mm one clears)
    T0 = _lip_pose(0.0, 0.1, 0.01, 0.0)[None]
    z0 = float(T0[0, 2, 3])

    def clear_fn(T, q):
        return np.where(T[:, 2, 3] - z0 >= 0.009, 0.01, -0.01)

    for exact, want in ((False, 0.015), (True, 0.010)):
        T = T0.copy()
        shift = np.zeros(1)
        clear = CA._clear_grouped(clear_fn, T, np.array([0.02]))
        ok = clear >= 0.001
        real = CA._object_retreats(clear_fn, T, np.array([0.02]), shift, clear, ok, _cfg(retreat_shift_exact=exact))
        assert ok[0] and math.isclose(float(T[0, 2, 3] - z0), 0.010, abs_tol=1e-12)
        assert math.isclose(float(real[0]), 0.010, abs_tol=1e-12) and math.isclose(float(shift[0]), want, abs_tol=1e-12)


def test_gate_dump_counts_stats_and_caps():
    d = CA._GateDump(True, max_records=2)
    d.gate("lip_shallow", "sweep", 4, 1, np.array([0, 1, 3]), {"width_mm": np.array([np.inf, 5.0, 6.0, 7.0]), "why": np.array(["width_lost", "touch", "x", "touch"])},
           np.array(["width_lost", "touch", "touch"]))
    st = d.pools["lip_shallow"][0]
    assert st["dropped_by"] == {"touch": 2, "width_lost": 1} and len(st["dropped"]) == 2 and st["dropped"][0]["width_mm"] is None
    assert st["stats"]["width_mm"]["n_finite"] == 2
    off = CA._GateDump(False)
    off.gate("top", "pool", 1, 1)
    assert off.pools == {}


# ------------------------------------------------------------------------------------------------ end to end (stub sampler)


def test_generate_runs_the_families_through_the_shared_gates(tmp_path, monkeypatch):
    g = _plate()
    monkeypatch.setattr(OG, "load", lambda model_dir: g)
    T_stub = np.stack([_lip_pose(a, R_OUT - 0.004, 0.013, t) for a in np.linspace(0, 2 * math.pi, 9, endpoint=False) for t in (20.0, 55.0, 65.0)])

    class _Sampler:
        @staticmethod
        def sample_antipodal(*a, **k):
            return list(T_stub)

    monkeypatch.setattr(CA, "_isaac_sampler", lambda: _Sampler)
    job = {"category": "plate", "model": "synthetic", "scale": [1, 1, 1], "constraint": "tabletop", "rest_hint_wxyz": None, "id": "plate__synthetic__s1"}
    small = dict(straddle_n_azimuth=8, straddle_n_heights=4, straddle_tilts_deg=(0.0, 10.0), straddle_pad_offsets_m=(0.014, 0.028), topk=4)
    d0, d1 = tmp_path / "off", tmp_path / "on"
    d0.mkdir()
    d1.mkdir()
    off = CA.generate(job, str(d0), cfg=_cfg(**small), densify=())
    on = CA.generate(job, str(d1), cfg=_cfg(families=("lip_shallow", "rim_straddle"), gate_dump=True, **small), densify=())
    # the modes are untouched by the families
    strip = ("features", "geometry", "T_object_eef", "T_rest_eef", "name", "mode", "score", "width_m")
    assert [{k: c[k] for k in strip} for c in on["candidates"] if "family" not in c] == [{k: c[k] for k in strip} for c in off["candidates"]]
    assert on["log"]["top"] == off["log"]["top"] and on["log"]["side"] == off["log"]["side"]
    assert on["log"]["lip_shallow"]["n_pool"] > 0 and 0 < on["log"]["rim_straddle"]["n_pool"] <= 8 * 4 * 2 * 2
    for c in on["candidates"]:
        if "family" in c:
            assert c["mode"] == "top" and c["name"].startswith(c["family"] + "_") and "pad_offset_m" in c["features"]
    assert not os.path.exists(d0 / CA.GATE_DUMP_FILE)
    dump = json.load(open(d1 / CA.GATE_DUMP_FILE))
    assert dump["schema"] == CA.GATE_DUMP_SCHEMA and dump["families"] == ["lip_shallow", "rim_straddle"]
    pools = dump["passes"][-1]["pools"]
    assert {"top", "side", "lip_shallow", "rim_straddle"} <= set(pools)
    for stages in pools.values():
        assert stages[0]["gate"] == "pool"
        for st in stages:
            assert st["n_out"] <= st["n_in"]
        for a, b in zip(stages, stages[1:]):  # every gate starts from what the previous one kept
            assert b["n_in"] == a["n_out"]
    geo = [st for st in pools["rim_straddle"] if st["gate"] == "geometry"]
    if geo:
        assert all("blade_depth_at_real_contact_mm" in r and "reason" in r for r in geo[0]["records"])
