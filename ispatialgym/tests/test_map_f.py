"""of-v2maps 2026-11-07 (prompt I item 4): F from the map's content (v2/map_f.py). Simulator-free."""

import json
import math
from types import SimpleNamespace

import numpy as np
import torch as th

from ispatialgym.sampling import grasp_library as GL
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling import reach_maps as RM
from ispatialgym.sampling.v2 import map_f as MF


def _map(tmp_path):
    cells = np.array([[0.51, 0.0], [0.59, 0.0], [0.51, 0.08]])
    yaws = np.array([k * math.pi / 4 for k in range(8)])
    grasps = [{"name": "top_0", "base": "top_0", "twin": False, "param": 0.0, "family_id": "f0"},
              {"name": "top_0@+0.300", "base": "top_0", "twin": False, "param": 0.3, "family_id": "f0"},
              {"name": "top_0~r180", "base": "top_0", "twin": True, "param": 0.0, "family_id": "f0"},
              {"name": "side_1", "base": "side_1", "twin": False, "param": None, "family_id": None}]
    st = np.full((3, 8, 4), RM.STATUS["ik_grasp"], dtype=np.uint8)
    st[0, 0, 0] = st[0, 0, 1] = st[0, 0, 2] = st[0, 2, 3] = RM.STATUS["feasible"]
    meta = {"schema": RM.SCHEMA, "kind": "tabletop", "key": "mk", "edge_distances": [0.45], "config": {"lattice_m": 0.08},
            "trunks": [{"translate": 0.25, "tilt": -1.0, "eligible": True}], "grasps": grasps, "arm": "right", "height": {"value": 0.8}}
    base = tmp_path / "map__a__b__h0.8000__mk"
    (tmp_path / (base.name + ".json")).write_text(json.dumps(meta))
    np.savez_compressed(str(base) + ".npz", cells=cells, yaws=yaws, **{"s0.45_t0.25_k-1.0": st})
    return str(base) + ".json"


def _view(path):
    return RB.ReachBands({"mode": "map_view", "map_path": path, "stand_off": 0.45, "overhang": 0.0}, {}, {}, None, {}, {}, {}, {})


def test_focus_cell_and_F_from_map(tmp_path, monkeypatch):
    MF._RM.clear()
    view = _view(_map(tmp_path))
    stance = SimpleNamespace(stand_off=0.45, translate=0.25, tilt=-0.95, base_pose2d=(1.0, 2.0, math.pi / 2))
    # world focus = base + R(yaw) @ (0.52, 0.01): the cell (0.51, 0.0)
    fx = [1.0 - 0.01, 2.0 + 0.52]
    mc = MF.focus_cell(view, stance, fx)
    assert mc is not None and mc.index == 0 and mc.trunk == (0.25, -1.0) and mc.edge_distance == 0.45
    assert MF.focus_cell(view, stance, [1.0, 2.0 + 0.9]) is None  # off the lattice
    assert MF.focus_cell(view, SimpleNamespace(**{**stance.__dict__, "translate": 0.3}), fx) is None  # no trunk at this translate
    fam = {"type": "rotational", "axis": [0, 0, 1.0], "center": [0, 0, 0]}
    monkeypatch.setattr(GL, "v2_entry", lambda key: SimpleNamespace(families=lambda: {"f0": fam}))
    g0 = GL.Grasp(key="k", name="top_0", position=th.tensor([0.05, 0.0, 0.1]), quat=th.tensor([1.0, 0, 0, 0]), finger_q=[], meta={"family_id": "f0", "width": 0.02})
    g1 = GL.Grasp(key="k", name="side_1", position=th.tensor([0.0, 0.0, 0.05]), quat=th.tensor([0, 0, 0, 1.0]), finger_q=[], meta={})
    F, W, facts = MF.F_from_map(mc, [g0, g1], np.random.default_rng(0))
    names = sorted(g.name.split("@")[0] for _, g in F)
    assert names == ["side_1", "top_0", "top_0"] and facts["members"] == 1  # the twin merged onto its row, the member its own grasp
    yaws = {round(y, 4) for y, g in F if g.name == "top_0"}
    assert yaws == {round(math.pi / 2, 4)}  # world yaw = the base yaw + the map yaw (0)
    member = next(g for _, g in F if "@" in g.name)
    assert 0.3 - 0.15 - 1e-9 <= member.meta["family_param"] <= 0.3 + 0.15 + 1e-9  # continuous within half the member spacing
    assert abs(W[(round(math.pi / 2, 4), "top_0")] - (1 / 2 + 1)) < 1e-9  # top_0 rows: 1/2 (the non-twin family) + 1 (the twin row)
    monkeypatch.setenv("ISPATIALGYM_MAP_F_FAMILY", "0")
    F2, _, f2 = MF.F_from_map(mc, [g0, g1], np.random.default_rng(0))
    assert f2["members"] == 0 and len(F2) == 2


def test_family_member_grasp_and_its_name(monkeypatch):
    fam = {"type": "rotational", "axis": [0, 0, 1.0], "center": [0, 0, 0]}
    g = GL.Grasp(key="k", name="rim_2", position=th.tensor([0.05, 0.0, 0.1]), quat=th.tensor([0, 0, 0, 1.0]), finger_q=[], meta={"family_id": "f", "width": 0.01})
    m = GL.family_member(g, fam, math.pi / 2)
    assert m.name == "rim_2@+1.5708" and np.allclose(m.position.numpy(), [0.0, 0.05, 0.1], atol=1e-6) and m.meta["width"] == 0.01
    assert m.meta["family_base"] == "rim_2" and g.name == "rim_2"  # the representative is untouched


def test_draw_order_is_the_permutation_without_weights():
    F = [(0.0, SimpleNamespace(name="a"), []), (1.0, SimpleNamespace(name="b"), [])]
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    assert list(MF.draw_order(a, F, None)) == list(b.permutation(2))
    w = {(0.0, "a"): 1e-6, (1.0, "b"): 1e6}
    assert list(MF.draw_order(np.random.default_rng(1), F, w))[0] == 1  # the heavy pair first
    assert sorted(MF.draw_order(np.random.default_rng(1), F, {(0.0, "a"): 1.0})) == [0, 1]  # a pair without weight: the plain permutation


def test_release_count_matches_the_place_pose_in_the_base_frame(tmp_path):
    from ispatialgym.grasping import schema as S

    MF._RM.clear()
    path = _map(tmp_path)
    m = json.load(open(path))
    T_oe = {"position": [0.0, 0.0, 0.1], "orientation_xyzw": [1.0, 0.0, 0.0, 0.0]}
    for g in m["grasps"]:
        g["T_object_eef"] = T_oe
    m["object"] = {"rests": {"r0": {"quat_xyzw": [0, 0, 0, 1.0], "h_root": 0.02}}}
    m["rest_id"] = "r0"
    open(path, "w").write(json.dumps(m))
    view = _view(path)
    stance = SimpleNamespace(stand_off=0.45, translate=0.25, tilt=-1.0, base_pose2d=(1.0, 2.0, 0.0))
    # the map entry (cell (0.51, 0), yaw 0, top_0): eef in the base frame at (0.51, 0, 0.8 + 0.02 + 0.1); in the world + base offset
    T = np.eye(4)
    T[:3, :3] = S.quat_to_mat([1.0, 0.0, 0.0, 0.0])
    T[:3, 3] = [1.0 + 0.51, 2.0, 0.0 + 0.92]
    assert MF.release_count(view, stance, T, "top_0", floor_z=0.0, place_support_z=0.8) >= 1
    assert MF.release_count(view, stance, T, "top_0", floor_z=0.0, place_support_z=0.9) is None  # not the map's height
    T2 = T.copy(); T2[0, 3] += 0.3
    assert MF.release_count(view, stance, T2, "top_0", floor_z=0.0, place_support_z=0.8) == 0  # no entry there ("no compatible entry")
