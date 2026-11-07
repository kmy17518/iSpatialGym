"""train-fast TILT_SNAP (owner 2026-11-01, inside_02): an empty F at a Phase 2 stance's continuous tilt -> the stance moves to its band's
grid tilt when the work-camera rules still pass there and F is screened there; otherwise the cell is kept only when both grid tilts
around the stance's tilt hold it (``tabletop._tilt_snap``; stubs, no simulator)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import Stance

FAR, NEAR = (0.96, 0.08), (0.60, 0.0)


class _Band:
    """Grid tilts -0.9 (cells up to 0.80 m) and -1.2 (up to 0.96 m) at translate 0.25; cells in the base frame."""

    def __init__(self):
        self.meta = {}
        self.cells = {(0.25, -0.9): np.array([NEAR]), (0.25, -1.2): np.array([NEAR, FAR])}

    def nearest_trunk(self, translate, tilt, stand_off=None):
        c = [k for k in self.cells if abs(k[0] - translate) < 1e-6]
        return min(c, key=lambda k: abs(k[1] - tilt)) if c else None

    def to_base(self, xy, base_pose2d):
        return np.asarray(xy, dtype=float).reshape(1, 2)

    def cell_index(self, translate, tilt, cell, stand_off=None):
        k = self.nearest_trunk(translate, tilt)
        hits = [i for i, c in enumerate(self.cells[k]) if np.allclose(c, cell, atol=1e-6)]
        return hits[0] if hits else None


def _sampler(camera_ok=True, reach_at=(-1.2,)):
    def feasible_sets(cells, yaws, grasps, pose_at, stance, arm, ignore, eef_of, cache=True):
        return {0: [(0.0, grasps[0])]} if any(abs(stance.tilt - t) < 1e-6 for t in reach_at) else {}

    s = SimpleNamespace(ikc=SimpleNamespace(stances_collision_free=lambda sts: th.ones(len(sts), dtype=th.bool), feasible_sets=feasible_sets),
                        world=SimpleNamespace(camera_model=SimpleNamespace(angle_off_axis=lambda cam, focus: 0.1)))
    s._phase2_camera_rule = {"max_angle": 0.5, "boxed": lambda st, cam, focus, info: camera_ok}
    return s


def _run(monkeypatch, cell, tilt, **kw):
    monkeypatch.setattr(TT, "stance_cameras", lambda s, sts: [None] * len(sts))
    st = Stance("manipulation", (0.0, 0.0, 0.0), 0.25, tilt, focus=th.tensor([cell[0], cell[1], 0.5]))
    manip = st.with_trunk(st.translate, st.tilt, "manipulation")
    st_of = {"right": st}
    F, facts = TT._tilt_snap(_sampler(**kw), manip, st_of, ["right"], {"right": _Band()}, th.tensor(cell), [0.0], ["g"], None, [], None)
    return F, facts, manip, st_of


def test_snaps_to_the_grid_tilt_when_the_camera_passes(monkeypatch):
    F, facts, manip, st_of = _run(monkeypatch, FAR, -1.07)
    assert facts["outcome"] == "adopted" and F["right"] == [(0.0, "g")]
    assert manip.tilt == pytest.approx(-1.2) and st_of["right"].tilt == pytest.approx(-1.2) and manip.extras["tilt_snap"]["from"][1] == pytest.approx(-1.07)


def test_camera_failing_falls_back_to_the_bracket(monkeypatch):
    F, facts, manip, _ = _run(monkeypatch, FAR, -1.07, camera_ok=False)
    assert F is None and facts["first"] == "camera" and facts["outcome"] == "out_of_bracket"  # held at -1.2 only, not at -0.9
    assert manip.tilt == pytest.approx(-1.07)  # unchanged
    F, facts, _, _ = _run(monkeypatch, NEAR, -1.07, camera_ok=False)
    assert F is None and facts["outcome"] == "in_bracket"  # both grid tilts hold it: an ordinary miss, kept


def test_empty_at_the_grid_tilt_too(monkeypatch):
    F, facts, _, _ = _run(monkeypatch, FAR, -1.07, reach_at=())
    assert F is None and facts["first"] == "empty_at_grid" and facts["outcome"] == "out_of_bracket"


def test_a_stance_on_a_grid_tilt_is_not_moved(monkeypatch):
    F, facts, manip, _ = _run(monkeypatch, FAR, -1.2)
    assert F is None and facts["first"] == "on_grid" and facts["outcome"] == "in_bracket" and manip.tilt == pytest.approx(-1.2)


class _MapView(_Band):
    """i7-unblock 2026-11-07: a reach-map view (multi-distance: meta stand_offs, mode map_view) - the FAR cell is held at the grid tilt
    -1.2 at stand-off 0.45 only."""

    def __init__(self):
        super().__init__()
        self.meta = {"mode": "map_view", "stand_offs": [0.35, 0.45, 0.55]}
        self.cells_so = {0.35: {(0.25, -0.9): np.array([NEAR]), (0.25, -1.2): np.array([NEAR])}, 0.45: dict(self.cells), 0.55: dict(self.cells)}
        self.cells = self.cells_so[0.45]
        self.asked = []

    def _maps(self, stand_off=None):
        return (self.cells_so[min(self.cells_so, key=lambda s: abs(s - stand_off))] if stand_off is not None else self.cells), {}, {}

    def nearest_trunk(self, translate, tilt, stand_off=None):
        self.asked.append(stand_off)
        c = [k for k in self._maps(stand_off)[0] if abs(k[0] - translate) < 1e-6]
        return min(c, key=lambda k: abs(k[1] - tilt)) if c else None

    def cell_index(self, translate, tilt, cell, stand_off=None):
        k = self.nearest_trunk(translate, tilt, stand_off)
        hits = [i for i, c in enumerate(self._maps(stand_off)[0][k]) if np.allclose(c, cell, atol=1e-6)]
        return hits[0] if hits else None


def test_a_map_view_snaps_at_the_stances_stand_off(monkeypatch):
    monkeypatch.setattr(TT, "stance_cameras", lambda s, sts: [None] * len(sts))
    for so, outcome in ((0.45, "adopted"), (0.35, "out_of_bracket")):
        st = Stance("manipulation", (0.0, 0.0, 0.0), 0.25, -1.07, focus=th.tensor([FAR[0], FAR[1], 0.5]), stand_off=so)
        manip = st.with_trunk(st.translate, st.tilt, "manipulation")
        view = _MapView()
        reach = (-1.2,) if so == 0.45 else ()
        F, facts = TT._tilt_snap(_sampler(reach_at=reach), manip, {"right": st}, ["right"], {"right": view}, th.tensor(FAR), [0.0], ["g"], None, [], None)
        assert facts["outcome"] == outcome and all(a == so for a in view.asked), (so, facts, view.asked)
    # a legacy multi-distance band (not a map view) keeps the former answer
    band = _Band()
    band.meta = {"stand_offs": [0.35, 0.45]}
    st = Stance("manipulation", (0.0, 0.0, 0.0), 0.25, -1.07, focus=th.tensor([FAR[0], FAR[1], 0.5]), stand_off=0.45)
    F, facts = TT._tilt_snap(_sampler(), st.with_trunk(0.25, -1.07, "manipulation"), {"right": st}, ["right"], {"right": band}, th.tensor(FAR), [0.0], ["g"], None, [], None)
    assert F is None and facts["outcome"] == "no_band"
