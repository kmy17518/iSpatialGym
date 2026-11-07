"""i7-generic 2026-11-07 (prompt I item 7, the generic kind): the GENERIC-PINCH reach map - its pose set (the legacy generic band's), its
identity (``reach_maps.generic_map_key`` / ``find_generic``), the band view's per-cell class bitmask (``map_sampling.as_bands``) as the
generic consumers read it, ``tabletop.v2_map_view`` routing and the CLI's pseudo-entries. Simulator-free."""

import json
import logging
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.sampling import map_sampling as MS
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling import reach_maps as RM

TRUNK = (0.25, -1.0)
TKEY = "t0.25_k-1.0"
CELLS = np.array([[0.51, 0.0], [0.59, 0.0], [0.51, 0.08]], dtype=np.float32)


def _status():
    """Per d_e (cells 3, 1 "yaw", 8 classes): cell 0 passes p00_y0 / p30_y0 / p60_y90, cell 1 only p45_y0, cell 2 nothing (at 0.45);
    at 0.35 cell 0 passes p00_y90 only; at 0.55 nothing."""
    n = len(RB.GENERIC_CLASSES)
    st = {d: np.full((3, 1, n), RM.STATUS["ik_pre"], dtype=np.uint8) for d in (0.35, 0.45, 0.55)}
    for name in ("p00_y0", "p30_y0", "p60_y90"):
        st[0.45][0, 0, RB.GENERIC_CLASSES.index(name)] = RM.STATUS["feasible"]
    st[0.45][1, 0, RB.GENERIC_CLASSES.index("p45_y0")] = RM.STATUS["feasible"]
    st[0.45][2, 0, :] = RM.STATUS["hand_policy"]
    st[0.35][0, 0, RB.GENERIC_CLASSES.index("p00_y90")] = RM.STATUS["feasible"]
    return st


def _generic_map(root, h, *, arm="right", pose_set=None, policy_key=None, robot=None, config=None, status=None):
    ps = pose_set or RM.generic_pose_set()
    meta = {"schema": RM.SCHEMA, "kind": RM.GENERIC_KIND, "arm": arm, "object": None, "object_key": RM.GENERIC_KIND, "rest_id": None,
            "height": {"value": h, "anchor_key": f"{h:.4f}"}, "edge_distances": [0.35, 0.45, 0.55],
            "config": config if config is not None else RM.MapConfig(twins=False).key(),
            "policy_key": policy_key or CP.DEFAULT.key(arm), "robot": robot or RM.robot_digest(), "pose_set": ps,
            "grasps": [{"name": n, "base": n, "twin": False} for n in ps["classes"]], "classes": list(ps["classes"]),
            "trunks": [{"translate": TRUNK[0], "tilt": TRUNK[1], "eligible": True, "key": TKEY, "posture": "untucked"},
                       {"translate": 0.5, "tilt": 0.0, "eligible": False, "key": "t0.50_k+0.0"}]}
    meta["key"] = RM.generic_map_key(meta)
    st = status or _status()
    arrays = {"cells": CELLS, "yaws": np.zeros(1, dtype=np.float32), **{f"s{d:.2f}_{TKEY}": a for d, a in st.items()}}
    return RM.save_generic(os.path.join(root, RM.GENERIC_DIR), meta, arrays)


def _anchors(path, keys=("0.7351", "0.8126")):
    path.write_text(json.dumps({"schema": "isg-height-anchors/1", "max_approx_m": 0.01,
                                "anchors": [{"key": k, "height": float(k), "precision": 1e-4, "surfaces": [f"s/{k}"]} for k in keys]}))
    return str(path)


def test_the_pose_set_is_the_legacy_generic_band_set():
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.robot_utils import RobotKit

    ps = RM.generic_pose_set()
    assert ps["classes"] == RB.GENERIC_CLASSES and ps["dz_m"] == 0.03 and ps["fingers_q"] == C.FINGER_OPEN and ps["pre_dist_m"] == RB.PRE_GRASP
    assert RM.generic_dz(0.025) == 0.03 and RM.generic_dz(0.05) == 0.03 and RM.generic_dz(0.10) == 0.06  # max(0.6 h, 0.03)
    R = RM.generic_rotations(ps)
    # the legacy band's candidates at a cell (map frame: base yaw 0, forward +x), tilt-major / yaw-minor = the class bit order
    legacy = RobotKit.grasp_candidates_at(RobotKit, th.tensor([0.6, 0.1, 0.83]), [0.0, math.pi / 2], RB.TILTS_GENERIC, th.tensor([1.0, 0.0]))
    assert len(legacy) == len(R) == 8
    rows = RM.generic_class_rows(ps, R)
    for k, ((pre, grasp), row) in enumerate(zip(legacy, rows)):
        Rl = RM.S.quat_to_mat(grasp[1].tolist())
        assert np.allclose(Rl, R[k], atol=1e-5), row["name"]
        a = R[k][:, 2]  # approach = eef +z: the pre-grasp 0.10 m back along it, as the map's TP
        assert np.allclose((grasp[0] - pre[0]).numpy(), RB.PRE_GRASP * a, atol=1e-5)
        tilt = math.degrees(math.acos(-a[2]))
        assert abs(tilt - row["tilt_deg"]) < 1e-3 and (row["tilt_deg"] == 0 or a[0] > 0)  # pitched toward the robot's forward
        assert row["T_cell_eef"]["position"] == [0.0, 0.0, 0.03]
    assert RM.generic_pose_set(fingers_q=0.0)["fingers_rule"] == "explicit"


def test_generic_map_key_and_find_enforce_the_identity(tmp_path):
    from ispatialgym.sampling.height_anchors import AnchorSet

    root = str(tmp_path)
    p = _generic_map(root, 0.8126)
    assert os.path.basename(p).startswith("map__generic__h0.8126__") and os.path.dirname(p) == os.path.join(root, RM.GENERIC_DIR)
    gdir = os.path.join(root, RM.GENERIC_DIR)
    ps = RM.generic_pose_set()
    pk, rb = CP.DEFAULT.key("right"), RM.robot_digest()
    kw = dict(pose_set=ps, policy_key=pk, robot=rb)
    rm = RM.find_generic(gdir, 0.8126, arm="right", **kw)
    assert rm is not None and rm.path == p and rm.lookup["how"] == "height"
    assert rm.status(0.45, *TRUNK).shape == (3, 1, 8) and len(rm.feasible_cells(0.45, *TRUNK)) == 2
    assert RM.find_generic(gdir, 0.8126, arm="left", **kw) is None  # another arm
    assert RM.find_generic(gdir, 0.8126, arm="right", pose_set=RM.generic_pose_set(dz_m=0.06), policy_key=pk, robot=rb) is None  # another dz
    assert RM.find_generic(gdir, 0.8126, arm="right", pose_set=RM.generic_pose_set(fingers_q=0.0), policy_key=pk, robot=rb) is None  # fingers
    assert RM.find_generic(gdir, 0.8126, arm="right", pose_set=ps, policy_key="isg-support-contact-v0-old", robot=rb) is None  # policy
    assert RM.find_generic(gdir, 0.8126, arm="right", pose_set=ps, policy_key=pk, robot="oldrobot") is None  # robot model
    assert RM.find_generic(gdir, 0.8126, arm="right", config_require={"preshape": "max"}, **kw) is None
    assert RM.find_generic(gdir, 0.8126, arm="right", config_require={"preshape": "grasp", "lattice_m": 0.08}, **kw) is not None
    assert RM.find_generic(gdir, 0.8131, arm="right", **kw) is None  # no anchors: the map's own height only
    A = AnchorSet(json.load(open(_anchors(tmp_path / "a.json")))["anchors"], 0.01)
    hit = RM.find_generic(gdir, 0.8131, arm="right", anchors=A, **kw)
    assert hit is not None and hit.lookup == {"how": "approx", "query_h": 0.8131, "map_h": 0.8126}
    assert RM.find_generic(gdir, 0.8126, arm="right", anchors=A, **kw).lookup["how"] == "exact"
    assert RM.find_generic(gdir, 0.7351, arm="right", anchors=A, **kw) is None  # covered anchor, no map there
    assert RM.find_generic(gdir, 0.95, arm="right", anchors=A, **kw) is None  # uncovered
    # the key covers the identity: a meta edited after the fact (another height under the old key) does not serve
    m = json.load(open(p))
    m["height"]["value"] = 0.7351
    q = p.replace("h0.8126", "h0.7351")
    json.dump(m, open(q, "w"))
    os.link(p[:-5] + ".npz", q[:-5] + ".npz")
    assert RM.find_generic(gdir, 0.7351, arm="right", **kw) is None
    # never served for a library entry, nor a tabletop map for the generic query
    lib = SimpleNamespace(identity_key=lambda: "obj-key")
    assert RM.ReachMap.find(gdir, lib, 0.8126, rest_id="r0", arm="right") is None
    tab = {"schema": RM.SCHEMA, "kind": "tabletop", "object_key": "obj-key", "rest_id": "r0", "arm": "right", "height": {"value": 0.8126}}
    (tmp_path / RM.GENERIC_DIR / "map__a__b__h0.8126__t.json").write_text(json.dumps(tab))
    assert RM.find_generic(gdir, 0.8126, arm="right", **kw).path == p
    assert RM._generic_meta_matches(p, "right", ps) and not RM._generic_meta_matches(p, "left", ps)
    assert not RM._generic_meta_matches(p, "right", RM.generic_pose_set(dz_m=0.06))


def test_band_view_of_a_generic_map_carries_the_class_bitmask(tmp_path):
    from ispatialgym.sampling.v2 import map_f as MF
    from ispatialgym.sampling.v2 import tabletop as TT

    view = MS.as_bands(RM.ReachMap(_generic_map(str(tmp_path), 0.8126)), overhang=0.0)
    m = view.meta
    assert m["mode"] == "map_view" and m["map_kind"] == "generic" and m["generic_pinch"] and m["grasp_names"] == []
    assert m["classes"] == RB.GENERIC_CLASSES and m["stand_offs"] == [0.35, 0.45, 0.55] and m["stand_off"] == 0.45
    bit = {n: 1 << k for k, n in enumerate(RB.GENERIC_CLASSES)}
    assert np.allclose(view.cells_so[0.45][TRUNK], CELLS[:2]) and view.classes_so[0.45][TRUNK].tolist() == [
        bit["p00_y0"] | bit["p30_y0"] | bit["p60_y90"], bit["p45_y0"]]
    assert view.classes_so[0.35][TRUNK].tolist() == [bit["p00_y90"]] and len(view.cells_so[0.55][TRUNK]) == 0
    assert view.classes[TRUNK].tolist() == view.classes_so[0.45][TRUNK].tolist()  # the reference stand-off
    assert sorted(view.pairs_so[0.45][TRUNK].tolist()) == sorted([[0, -1, RB.GENERIC_CLASSES.index(n)] for n in ("p00_y0", "p30_y0", "p60_y90")]
                                                                  + [[1, -1, RB.GENERIC_CLASSES.index("p45_y0")]])
    assert view.feasible(*TRUNK, CELLS[1], stand_off=0.45) == [("generic", RB.GENERIC_CLASSES.index("p45_y0"))]
    assert view.classes_at(*TRUNK, CELLS[0], stand_off=0.45) == ["p00_y0", "p30_y0", "p60_y90"]
    # the consumers, unchanged: cell_classes -> approach_pitch_order (alignment_axial's push pitches), joint_placement's "classes"
    names = TT.cell_classes(view, TRUNK[0], -0.95, CELLS[0], 0.45)
    assert names == ["p00_y0", "p30_y0", "p60_y90"] and TT.approach_pitch_order(names) == [0.0, 30.0, 60.0, 45.0]
    assert TT.cell_classes(view, *TRUNK, CELLS[0], 0.35) == ["p00_y90"]
    base = (1.0, 2.0, math.pi / 2)  # world = base + R(yaw) @ cell
    stance = SimpleNamespace(translate=TRUNK[0], tilt=TRUNK[1], stand_off=0.45, base_pose2d=base,
                             world_to_base=lambda xy: th.tensor(view.to_base(np.asarray(xy, dtype=float)[None, :2], base)[0]))
    work = view.to_world(CELLS[1:2], base)[0]
    jp = TT.joint_placement(view, None, stance, work, view.to_world(np.array([[0.6, 0.2]]), base)[0])
    assert jp["work_ok"] and jp["classes"] == ["p45_y0"] and TT.approach_pitch_order(jp["classes"]) == [45.0, 0.0, 30.0, 60.0]
    assert not MF.is_view(view)  # map-seeded F reads library grasps only


class _FakeWorld:
    """The SynthWorld surface compute_generic_map uses, on CPU: a configuration is (translate, tilt), the IK "solution" of a pose is the
    pose itself (flattened 4x4: the FK eef is the target), every sphere far above the slab, a trunk with tilt < -0.5 collides."""

    def __init__(self, cfg):
        self.cfg, self.dev = cfg, th.device("cpu")
        self.backend = SimpleNamespace(check_collision=lambda q, emb_name, margin: th.tensor([bool(q[0, 1] < -0.5)]),
                                       robot_spheres_world=lambda q: (th.tensor([[[0.0, 0.0, 2.0, 0.05]]]), ["torso_link4"]),
                                       position_tolerance=0.005, orientation_tolerance=0.05)

    def q_config(self, translate, tilt, posture="untucked", fingers=0.05):
        return th.tensor([translate, tilt, fingers])

    def shoulder(self, translate, tilt):
        return np.array([0.3, -0.2, 1.0]), 0.9

    def entry(self):
        return None

    def link_of_sphere(self):
        return ["torso_link4"]

    def full_q(self, entry, q_active, q_lock):
        return q_active

    def spheres(self, q_full):
        return th.tensor([0.0, 0.0, 2.0, 0.05]).repeat(q_full.shape[0], 1, 1)

    def eef(self, q_full):
        return q_full.reshape(-1, 4, 4)


def _fake_solve(world, entry, T, q_lock, seed_active=None):
    n, S_ = T.shape[0], world.cfg.seeds
    ok = (T[:, 0, 3] < 0.95)[:, None].repeat(1, S_)  # a pre-grasp / grasp beyond 0.95 m ahead "has no IK"
    return ok, T.reshape(n, 1, 16).repeat(1, S_, 1)


def test_compute_generic_map_statuses_on_a_fake_world(tmp_path, monkeypatch):
    from ispatialgym.sampling.v2 import tabletop as TT

    monkeypatch.setattr(RM, "_solve", _fake_solve)
    monkeypatch.setattr(RB, "trunk_grid", lambda h: [(0.25, -0.3), (0.5, -0.8)])
    cfg = RM.MapConfig(seeds=2, twins=False, lattice_m=0.16)
    meta, arrays = RM.compute_generic_map(_FakeWorld(cfg), 0.8126, {"key": "0.8126", "surfaces": ["house_single_floor/breakfast_table_rhjoby_0/top"]})
    assert meta["kind"] == "generic" and meta["key"] == RM.generic_map_key(meta) and meta["pose_set"] == RM.generic_pose_set()
    assert meta["object_key"] == "generic" and meta["height"]["anchor_key"] == "0.8126" and [g["name"] for g in meta["grasps"]] == RB.GENERIC_CLASSES
    cells = arrays["cells"]
    assert arrays["yaws"].tolist() == [0.0] and all(arrays[f"s{d:.2f}_t0.25_k-0.3"].shape == (len(cells), 1, 8) for d in (0.35, 0.45, 0.55))
    assert [t["eligible"] for t in meta["trunks"]] == [True, False]
    assert (arrays["s0.45_t0.50_k-0.8"] == RM.STATUS["trunk_ineligible"]).all()
    st = {d: arrays[f"s{d:.2f}_t0.25_k-0.3"][:, 0, :] for d in (0.35, 0.45, 0.55)}
    ci = lambda x, y: int(np.argmin(np.abs(cells - np.array([x, y])).sum(axis=1)))  # noqa: E731
    g = RB.GENERIC_CLASSES.index
    c = ci(0.67, -0.17)
    # the fully open fingers (the default opening) of the pitched wrist-yaw-90 classes reach below the slab top 3 cm under the eef: the
    # contact policy's 2 mm finger rule (p30 / p45) and the wrist camera (p60) reject them at every d_e; the other classes pass
    for d in (0.35, 0.45, 0.55):
        assert [int(st[d][c, g(n)]) for n in ("p30_y90", "p45_y90", "p60_y90")] == [RM.STATUS["hand_policy"]] * 3
        assert all(int(st[d][c, g(n)]) == RM.STATUS["feasible"] for n in ("p00_y0", "p00_y90", "p30_y0", "p45_y0", "p60_y0"))
    lo = ci(0.35, -0.17)  # the eef point 0.35 m ahead: over the slab at d_e 0.35 only
    assert st[0.35][lo, g("p00_y0")] == RM.STATUS["feasible"] and st[0.45][lo, g("p00_y0")] == RM.STATUS["off_support"]
    assert (st[0.45][ci(0.99, -0.17)] != RM.STATUS["feasible"]).all() and st[0.45][ci(0.99, -0.17), g("p00_y0")] == RM.STATUS["ik_pre"]
    assert (st[0.45][ci(1.15, 0.8)] == RM.STATUS["reach_bound"]).all()
    assert meta["trunks"][0]["class_counts"]["0.45"]["p30_y90"] == 0 and meta["trunks"][0]["class_counts"]["0.45"]["p00_y0"] > 0
    # the saved map reads back through find_generic + as_bands: the cell's bitmask names its classes for the consumers
    p = RM.save_generic(str(tmp_path / RM.GENERIC_DIR), meta, arrays)
    rm = RM.find_generic(str(tmp_path / RM.GENERIC_DIR), 0.8126, arm="right", pose_set=RM.generic_pose_set(), policy_key=CP.DEFAULT.key("right"),
                         robot=RM.robot_digest())
    assert rm is not None and rm.path == p
    view = MS.as_bands(rm)
    assert TT.cell_classes(view, 0.25, -0.3, cells[c], 0.45) == ["p00_y0", "p00_y90", "p30_y0", "p45_y0", "p60_y0"]


def _spec(**kw):
    from ispatialgym.sampling.v2 import tabletop as TT

    return TT.BandSpec(object="block/abc", generic_pinch=True, height=0.025, **kw)


def test_v2_map_view_routes_a_generic_pinch_to_its_map(monkeypatch):
    from ispatialgym.sampling.v2 import tabletop as TT

    calls = []
    monkeypatch.setattr(TT, "generic_map_bands", lambda s, spec, top, arm=None: calls.append(arm) or "GENERIC")
    monkeypatch.setattr(TT, "map_bands", lambda *a, **k: pytest.fail("a generic pinch is not a library map"))
    s = SimpleNamespace(world=SimpleNamespace(floor_z=0.0, scene_model="house"))
    top = SimpleNamespace(name="table_0", z=0.8126)
    assert TT.v2_map_view(s, _spec(), top) == "GENERIC" and TT.v2_map_view(s, _spec(), top, arm="left") == "GENERIC" and calls == [None, "left"]


def test_generic_map_bands_hit_miss_and_requests(tmp_path, monkeypatch, caplog):
    from ispatialgym.sampling.v2 import tabletop as TT

    _generic_map(str(tmp_path), 0.8126)
    monkeypatch.setenv("ISPATIALGYM_REACH_MAPS", str(tmp_path))
    monkeypatch.setenv("ISPATIALGYM_REACH_MAP_ANCHORS", _anchors(tmp_path / "anchors.json"))
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY", "x")  # recorded: restored (unset) after the test
    monkeypatch.delenv("ISPATIALGYM_SUPPORT_POLICY")
    monkeypatch.delenv("ISPATIALGYM_REACH_MAP_REQUIRE", raising=False)
    monkeypatch.delenv("ISPATIALGYM_GENERIC_MAP_FINGERS", raising=False)
    s = SimpleNamespace(world=SimpleNamespace(floor_z=0.0, scene_model="house_single_floor"), ikc=SimpleNamespace())
    top = SimpleNamespace(name="breakfast_table_rhjoby_0", z=0.8126)
    view = TT.v2_map_view(s, _spec(), top)
    assert view is not None and view.meta["map_kind"] == "generic" and view.meta["anchor_lookup"]["how"] == "exact"
    assert view.meta["classes"] == RB.GENERIC_CLASSES and s.ikc.support_plane_z == {"breakfast_table_rhjoby_0": 0.8126}
    assert os.environ.get("ISPATIALGYM_SUPPORT_POLICY") == "grasp"
    near = TT.generic_map_bands(s, _spec(), SimpleNamespace(name="t2", z=0.8131))  # within the 0.8126 anchor's range
    assert near is not None and near.meta["anchor_lookup"] == {"how": "approx", "query_h": 0.8131, "map_h": 0.8126}
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        assert TT.generic_map_bands(s, _spec(), SimpleNamespace(name="desk_iotfzl_0", z=0.7351)) is None  # no map at the anchor
        assert TT.generic_map_bands(s, _spec(), top, arm="left") is None  # no left-arm map
        assert TT.generic_map_bands(s, TT.BandSpec(object="tall/x", generic_pinch=True, height=0.10), top) is None  # dz 0.06: another pose set
        monkeypatch.setenv("ISPATIALGYM_GENERIC_MAP_FINGERS", "0")
        assert TT.generic_map_bands(s, _spec(), top) is None  # a closed-blade map was asked for
        monkeypatch.delenv("ISPATIALGYM_GENERIC_MAP_FINGERS")
        assert TT.generic_map_bands(s, _spec(z_support=0.8187), top) is None  # a raised plane needs an EXACT anchor
    reqs = [json.loads(r.getMessage()[len("MAP_REQUEST "):]) for r in caplog.records if r.getMessage().startswith("MAP_REQUEST")]
    assert [(r["kind"], r["h"], r["arm"], r["dz_m"], r["fingers_q"]) for r in reqs] == [
        ("generic", 0.7351, "right", 0.03, 0.05), ("generic", 0.8126, "left", 0.03, 0.05), ("generic", 0.8126, "right", 0.06, 0.05),
        ("generic", 0.8126, "right", 0.03, 0.0), ("generic", 0.8187, "right", 0.03, 0.05)]
    assert reqs[0]["support"] == "desk_iotfzl_0" and reqs[-1]["plane"] == "z_support" and reqs[0]["scene"] == "house_single_floor"
    monkeypatch.delenv("ISPATIALGYM_REACH_MAPS")
    assert TT.generic_map_bands(s, _spec(), top) is None  # no maps root


def test_cli_generic_pseudo_entries(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": "cup__abc", "category": "cup", "model": "abc"}, {"id": "_generic_a", "kind": "generic"},
                                    {"id": "_generic_b", "kind": "generic", "fingers_q": 0.0}]))
    anchors = _anchors(tmp_path / "anchors.json", keys=("0.6854", "0.7351", "0.7853", "0.8126"))
    by_id = tmp_path / "by_id.json"
    by_id.write_text(json.dumps({"_generic_a": ["0.8126", "0.7853", "0.6793"], "_generic_b": ["0.7351"]}))
    done = []

    def fake_compute(world, h, anc, cfg=None, policy=None, pose_set=None):
        done.append((h, cfg.arm, pose_set["fingers_q"]))
        meta = {"schema": RM.SCHEMA, "kind": RM.GENERIC_KIND, "arm": cfg.arm, "height": {"value": h}, "pose_set": pose_set, "trunks": [],
                "timing": {}, "grasps": [], "edge_distances": list(cfg.edge_distances), "config": cfg.key()}
        meta["key"] = RM.generic_map_key(meta)
        return meta, {"cells": CELLS, "yaws": np.zeros(1, dtype=np.float32)}

    monkeypatch.setattr(RM, "SynthWorld", lambda cfg: SimpleNamespace(cfg=cfg))
    monkeypatch.setattr(RM, "compute_generic_map", fake_compute)
    a = SimpleNamespace(jobs=str(manifest), only=None, shard=None, anchors=anchors, heights=["0.6854"], heights_by_id=str(by_id), lattice=0.08, seeds=8,
                        batch=2048, arm="right", out=str(tmp_path / "maps"), skip_existing=True)
    RM._compute_generic_cli(a)
    assert sorted(done) == [(0.7351, "right", 0.0), (0.7853, "right", 0.05), (0.8126, "right", 0.05)]  # the library entry is not a generic job
    files = sorted(os.listdir(tmp_path / "maps" / RM.GENERIC_DIR))
    assert len([f for f in files if f.endswith(".json")]) == 3 and all(f.startswith("map__generic__h") for f in files)
    out = capsys.readouterr().out
    assert out.count('"ok": true') == 3 and out.rstrip().endswith("GB_MAP_DONE")
    assert '"ok": false, "h": "0.6793"' in out and "not an anchor key" in out  # the anchor file lacks it: said, not skipped silently
    done.clear()
    RM._compute_generic_cli(a)  # --skip-existing: same arm / kind / pose set
    assert done == []
    RM._compute_generic_cli(SimpleNamespace(**{**a.__dict__, "arm": "left", "shard": "1/2"}))  # shard 1/2 = the second generic entry
    assert done == [(0.7351, "left", 0.0)]
    with pytest.raises(SystemExit):
        RM._compute_generic_cli(SimpleNamespace(**{**a.__dict__, "shard": "2/3"}))  # no generic entry in the shard: no world started
