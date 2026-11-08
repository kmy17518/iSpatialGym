"""i7-tray 2026-11-08 (owner 1.1): the disembedding layout pad check (``ISPATIALGYM_LAYOUT_PAD_CHECK``) on the real embedded figure.

disembedding_01 went 4/4 -> 0/4 when the jigsaw entry was rebuilt: rows top_5 / top_10 of the rebuilt entry put the pre-shaped fingers
onto the figure's frame (isgefb) and inner part (isgefd) - the oracle's approach stopped on them 13-22 mm short - while the old rows
(top_2 / top_3 at the piece's far end) and the rebuilt top_8 / top_12 clear every part. The check reads the parts' convex collision
pieces at the layout's relative poses (assembly.json) and the hand's exported collision hulls."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym import REPO_ROOT
from ispatialgym.sampling.v2.tasks import disembedding as D

MODEL = os.path.join(REPO_ROOT, "ispatialgym", "assets", "objects", "embedded_figure", "model")


def _obj_vertices(path):
    return np.array([[float(v) for v in ln.split()[1:4]] for ln in open(path) if ln.startswith("v ")])


def _parts_in_target_frame():
    asm = json.load(open(os.path.join(MODEL, "assembly.json")))
    o_t = np.array(asm["parts"]["isgefa"]["origin_in_figure_m"][:3])
    parts = {}
    for name in ("isgefb", "isgefc", "isgefd"):
        off = np.array(asm["parts"][name]["origin_in_figure_m"][:3]) - o_t  # every part keeps the figure frame's orientation
        d = os.path.join(MODEL, name, "meshes", "collision")
        parts[name] = [D.HullObstacle.of_points(_obj_vertices(os.path.join(d, f)) + off, f"{name}/{f}") for f in sorted(os.listdir(d))]
    return parts


def _row(name, pos, quat_xyzw, width=0.012):
    return SimpleNamespace(name=name, pose=(th.tensor(pos), th.tensor(quat_xyzw)), meta={"width": width})


# T_object_eef of v2maps_int1's jigsaw entry (switches + pad sweep) and of the switches-off entry the 4/4 runs drew
ROWS = [
    _row("top_5", [0.002514, -0.030184, 0.018737], [0.9807853, -0.0, 0.1950903, -0.0]),
    _row("top_10", [0.010087, -0.030184, 0.017487], [-0.0, 0.9469301, 0.0, -0.3214395]),
    _row("top_8", [0.047583, -0.00027, 0.0193], [0.7905522, -0.6123946, -0.0, -0.0]),
    _row("old_top_3", [0.046048, 0.005676, 0.023921], [0.5799013, 0.7485928, 0.19685, -0.254113]),
]


@pytest.fixture(scope="module")
def parts():
    if not os.path.isdir(MODEL):
        pytest.skip("embedded figure model not in the tree")
    return _parts_in_target_frame()


def test_hull_obstacle_sdf_inside_outside():
    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    h = D.HullObstacle.of_points(cube)
    d = h.sdf(th.tensor([[0.5, 0.5, 0.5], [2.0, 0.5, 0.5], [0.5, 0.5, 0.9]]))
    assert d[0] == pytest.approx(-0.5, abs=1e-6) and d[1] == pytest.approx(1.0, abs=1e-6) and d[2] == pytest.approx(-0.1, abs=1e-6)


def test_switch_default_all(monkeypatch):
    # owner decision 2026-11-08 item 4: ``all`` is the default for fixed layouts; 0 restores the former pick
    monkeypatch.delenv(D.LAYOUT_PAD_SWITCH, raising=False)
    assert D.layout_pad_on() and D.layout_pad_mode() == "all"
    monkeypatch.setenv(D.LAYOUT_PAD_SWITCH, "0")
    assert not D.layout_pad_on()


def test_rows_on_the_frame_vs_clear_rows(parts):
    # measured offline from the demos (i7-tray clearance.py / jig_count.py): top_5 / top_10 -6 mm into the frame + isgefd, top_8 +1.3 mm,
    # old top_3 +0.5 mm (isgefd)
    cl = D.layout_pad_clearance(ROWS, parts)
    assert set(cl) == {r.name for r in ROWS}
    for n in ("top_8", "old_top_3"):
        assert cl[n][0] > -0.002, (n, cl[n])
    assert min(cl["top_5"][0], cl["top_10"][0]) < -0.002, cl


def test_mode_values(monkeypatch):
    for v, want in (("0", "off"), ("off", "off"), ("1", "all"), ("all", "all"), ("heavy", "heavy"), ("bogus", "all")):
        monkeypatch.setenv(D.LAYOUT_PAD_SWITCH, v)
        assert D.layout_pad_mode() == want


def test_per_part_clearance_separates_the_frame_from_the_light_inner_part(parts):
    per = D.layout_pad_clearances(ROWS, parts)
    # top_5 stands on the frame (heavy: 173 g vs the target's 29 g); top_8 / the old top_3 on none
    assert per["top_5"]["isgefb"] < -0.002
    assert all(d > -0.002 for d in per["top_8"].values()) and all(d > -0.002 for d in per["old_top_3"].values())
