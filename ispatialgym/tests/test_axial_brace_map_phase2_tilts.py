"""i7-bracefine (owner A9 #24, 2026-11-09): the axial_brace contact-map builder adds a trunk column at every Phase 2 stance tilt
(``tabletop.tilt_scan`` per translate) behind ``ISPATIALGYM_AXIAL_BRACE_MAP_PHASE2_TILTS`` (axial_brace only; in the map identity only when
on), and the map reads such a column under its own recorded key. Simulator-free: the fake synthetic world of test_contact_maps."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import object_geometry as OG
from ispatialgym.sampling import contact_maps as CMAP
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling import reach_maps as RM
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.tests.test_axial_brace_map_fix import _margin_world
from ispatialgym.tests.test_contact_library import _model_dir
from ispatialgym.tests.test_contact_maps import _FakeHC, _lib

ENV = "ISPATIALGYM_AXIAL_BRACE_MAP_PHASE2_TILTS"


@pytest.fixture
def glib(tmp_path):
    root = str(tmp_path / "glib")
    _model_dir(root, "magazine__flat__s1", (0.20, 0.28, 0.012))
    return root


def test_the_switch_is_off_by_default_and_enters_the_key_only_when_on(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert not CMAP.phase2_tilts_on()
    off = CMAP.contact_config("axial_brace")
    assert off.tilt_columns is None and "tilt_columns" not in off.key()  # the maps built before keep their keys
    monkeypatch.setenv(ENV, "1")
    on = CMAP.contact_config("axial_brace")
    assert on.key()["tilt_columns"] == "phase2" and {k: v for k, v in on.key().items() if k != "tilt_columns"} == off.key()
    for kind in ("touch_push", "touch_brace", "push_blade", "floor_push"):  # axial_brace only
        assert CMAP.contact_config(kind).tilt_columns is None


def test_the_columns_are_the_grid_plus_phase2s_stance_tilts():
    h = 0.8126
    assert CMAP.trunk_columns(h) == [(float(a), float(b)) for a, b in RB.trunk_grid(h)]
    cols = CMAP.trunk_columns(h, "phase2")
    assert set(CMAP.trunk_columns(h)) <= set(cols)
    fk = TT.CameraFK(*TT.camera_table()[:4])
    for tr in RB.TRANSLATES:
        want = TT.tilt_scan(fk, float(tr))
        assert CMAP.phase2_stance_tilts(tr) == pytest.approx(want, abs=1e-9)
        have = [ti for t0, ti in cols if t0 == tr]
        for ti in want:  # every stance tilt reads a built column (within the 5 mrad snap)
            assert min(abs(ti - h_) for h_ in have) <= CMAP.TILT_SNAP_RAD
    assert len(cols) == len(set(cols)) and len({CMAP.trunk_key(*c) for c in cols}) == len(cols)
    assert CMAP.trunk_key(0.5, -0.3) == "t0.50_k-0.3" and CMAP.trunk_key(0.5, -0.4542) == "t0.50_k-0.4542"
    with pytest.raises(ValueError):
        CMAP.trunk_columns(h, "dense")


def _map(monkeypatch, glib, on):
    lib = _lib(glib, "magazine__flat__s1", "axial_brace", "brace", long_axis=[0, 1, 0])
    monkeypatch.setattr(CP, "HandChecker", _FakeHC)
    monkeypatch.setattr(CP, "sphere_support_ok", lambda sph, ln, sb, arm, policy, **k: (th.ones(sph.shape[0], dtype=th.bool), None))
    monkeypatch.setattr(RM, "_solve", lambda world, entry, T, q_lock, seed_active=None: (th.ones(T.shape[0], world.cfg.seeds, dtype=th.bool),
                                                                                         th.zeros(T.shape[0], world.cfg.seeds, 7)))
    monkeypatch.setattr(RM, "robot_digest", lambda: "fakerobot")
    monkeypatch.delenv("ISPATIALGYM_AXIAL_BRACE_MAP_JOINT_MARGIN", raising=False)
    if on:
        monkeypatch.setenv(ENV, "1")
    else:
        monkeypatch.delenv(ENV, raising=False)
    cfg = CMAP.contact_config("axial_brace", lattice_m=0.16, seeds=2, edge_distances=(0.45,))
    geom = OG.load(os.path.join(glib, "models", "magazine__flat__s1"))
    return CMAP.compute_contact_map(_margin_world(cfg), lib, "axial_brace", 0.75, {"key": "h0.75", "height": 0.75}, geom, cfg=cfg)


def test_the_built_columns_are_written_and_read_under_their_keys(monkeypatch, glib, tmp_path):
    meta_off, arr_off = _map(monkeypatch, glib, False)
    meta_on, arr_on = _map(monkeypatch, glib, True)
    k_off = {t["key"] for t in meta_off["trunks"]}
    k_on = {t["key"] for t in meta_on["trunks"]}
    assert k_off < k_on and all(f"s0.45_{k}" in arr_on for k in k_on)
    for k in k_off:  # the former columns as before
        assert (arr_on[f"s0.45_{k}"] == arr_off[f"s0.45_{k}"]).all()
    assert meta_on["config"]["tilt_columns"] == "phase2" and "tilt_columns" not in meta_off["config"]
    assert meta_on["key"] != meta_off["key"] and meta_on["pose_set"]["digest"] == meta_off["pose_set"]["digest"]
    cm = CMAP.ContactMap(CMAP.save(str(tmp_path / "maps"), meta_on, arr_on))
    fine = next(t for t in meta_on["trunks"] if t["key"] not in k_off and t.get("eligible"))
    assert (cm.status(0.45, fine["translate"], fine["tilt"]) == arr_on[f"s0.45_{fine['key']}"]).all()
    # a stance at that Phase 2 tilt reads the built column (nearest and intersection reads alike)
    xy = cm.cells[0].astype(float)
    out = cm.lookup(0.45, fine["translate"], fine["tilt"], xy, 0.0, tilt_read="intersection")
    assert out["trunk"] == [fine["translate"], fine["tilt"]] and out["tilt_read"]["how"] == "built"
    assert cm.lookup(0.45, fine["translate"], fine["tilt"], xy, 0.0)["trunk"] == out["trunk"]
    assert np.isfinite(meta_on["timing"]["total_s"])
