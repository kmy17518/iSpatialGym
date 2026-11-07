"""of-v2maps 2026-11-07 (prompt I item 1, the isg1020 spec P2 §11): the v2 lookup and the exact-IK cache enforce what a v2 record was made for -
a deterministic entry per legacy key, its stable rest and that rest's grasps only, the shared contact policy, the robot model, and a cache key
that keeps policed and unpoliced witnesses apart. Simulator-free."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import schema as S
from ispatialgym.sampling import grasp_library as GL


def _entry(root, category, model, scale, *, legacy_key, rests, grasps, policy_key=None, dirname=None):
    d = S.model_dir(category, model, scale, root) if dirname is None else os.path.join(root, "models", dirname)
    os.makedirs(d, exist_ok=True)
    S.save_json(os.path.join(d, "library.json"), {
        "schema": S.LIBRARY, "category": category, "model": model, "scale": scale, "arm": "right", "legacy_key": legacy_key,
        "policy_key": policy_key if policy_key is not None else CP.DEFAULT.key("right"),
        "rests": {rid: {"schema": S.REST, "rest_id": rid, "quat_xyzw": q, "h_root": h, "stable": st} for rid, (q, h, st) in rests.items()},
        "grasps": [{"name": n, "rest_id": rid, "T_object_eef": {"position": [0, 0, 0.1], "orientation_xyzw": [1, 0, 0, 0]}, "width_m": 0.02}
                   for n, rid in grasps]})
    return os.path.join(d, "library.json")


@pytest.fixture
def v2root(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_GRASP_LIBRARY_V2", str(tmp_path))
    GL._v2_index.cache_clear()
    yield str(tmp_path)
    GL._v2_index.cache_clear()


def test_entry_choice_is_deterministic_and_prefers_a_stable_rest(v2root):
    # two entries share a legacy key (a retry copied into the same root): the one with a stable rest wins, whatever the glob order
    _entry(v2root, "knife", "aaa", [1, 1, 1], legacy_key="knife__aaa__st", rests={"r0": ([0, 0, 0, 1], 0.01, False)}, grasps=[("a", "r0")],
           dirname="a_unstable")
    good = _entry(v2root, "knife", "aaa", [1, 1, 1], legacy_key="knife__aaa__st", rests={"r0": ([0, 0, 0, 1], 0.01, True)}, grasps=[("b", "r0")],
                  dirname="z_stable")
    assert GL._v2_index(v2root)["knife__aaa__st"] == good


def test_rest_choice_and_grasps_of_that_rest_only(v2root):
    q1 = [0.7071068, 0, 0, 0.7071068]
    _entry(v2root, "knife", "bbb", [1, 1, 1], legacy_key="knife__bbb",
           rests={"r1": (q1, 0.02, True), "r0": ([0, 0, 0, 1], 0.01, True), "r2": ([0, 0, 1, 0], 0.03, False)},
           grasps=[("g0", "r0"), ("g1", "r1"), ("g1b", "r1"), ("g2", "r2")])
    lib = GL.v2_entry("knife__bbb")
    assert GL.v2_rest_id(lib) == "r0"  # stable rests only, then the id
    assert [g.name for g in GL.load_grasps("knife__bbb")] == ["g0"]  # was: every rest's grasps at r0's pose
    assert np.allclose(GL.rest_quat_xyzw("knife__bbb").numpy(), [0, 0, 0, 1])
    assert GL.v2_rest_id(SimpleNamespace(rests={"r0": SimpleNamespace(stable=False)})) is None


def test_an_entry_of_another_contact_policy_does_not_serve(v2root):
    _entry(v2root, "cup", "ccc", [1, 1, 1], legacy_key="cup__ccc", rests={"r0": ([0, 0, 0, 1], 0.05, True)}, grasps=[("g", "r0")],
           policy_key="isg-support-contact-v0-old")
    with pytest.raises(ValueError):
        GL.v2_entry("cup__ccc")


def _map(tmp_path, robot):
    from ispatialgym.sampling import reach_maps as RM

    base = tmp_path / f"map__cat__model__h0.8000__{robot}"
    meta = {"schema": RM.SCHEMA, "kind": "tabletop", "object_key": "obj-key", "rest_id": "r0", "arm": "right", "height": {"value": 0.8},
            "policy_key": "pk", "robot": robot}
    (tmp_path / (base.name + ".json")).write_text(json.dumps(meta))
    np.savez_compressed(str(base) + ".npz", cells=np.zeros((1, 2)), yaws=np.zeros(8))


def test_map_lookup_checks_policy_and_robot(tmp_path):
    from ispatialgym.sampling import reach_maps as RM

    _map(tmp_path, "oldrobot")
    lib = SimpleNamespace(identity_key=lambda: "obj-key")
    kw = dict(rest_id="r0", arm="right")
    assert RM.ReachMap.find(str(tmp_path), lib, 0.8, **kw) is not None  # unchecked: as before
    assert RM.ReachMap.find(str(tmp_path), lib, 0.8, policy_key="pk", robot="oldrobot", **kw) is not None
    assert RM.ReachMap.find(str(tmp_path), lib, 0.8, policy_key="other", **kw) is None
    assert RM.ReachMap.find(str(tmp_path), lib, 0.8, robot=RM.robot_digest(), **kw) is None  # a map of another robot model


def test_exact_cache_scope_carries_the_support_policy(monkeypatch):
    from ispatialgym.sampling.v2 import ik as IK

    chk = IK.IKChecker.__new__(IK.IKChecker)
    chk.cache_scope = {"scene": "s", "rooms": ["a"], "full_rooms": ["a", "b"]}
    monkeypatch.delenv("ISPATIALGYM_SUPPORT_POLICY", raising=False)
    off = chk.exact_cache_scope()
    assert "support_policy" not in off and "full_rooms" not in off  # legacy runs keep their key
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY", "grasp")
    on = chk.exact_cache_scope()
    assert on["support_policy"] == [CP.DEFAULT.key("right"), "grasp"] and {k: v for k, v in on.items() if k != "support_policy"} == off
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY", "1")
    assert chk.exact_cache_scope()["support_policy"][1] == "all"


def test_spa_signature_names_a_map_view():
    from ispatialgym.sampling.v2.tasks.iors_small import IORSSmall

    view = SimpleNamespace(path=None, meta={"map_key": "k123", "map_path": "/x/maps/e/map__a__b__h0.8__k123.json", "lattice": 0.08, "trunks": [1, 2]})
    sig = IORSSmall._spa_band_signature(view)
    assert sig["key"] == "k123" and sig["map_path"] == "map__a__b__h0.8__k123.json" and sig["n_trunks"] == 2
    legacy = SimpleNamespace(path="/b/band.npz", meta={"key": "band-key", "lattice": 0.08, "trunks": []})
    assert IORSSmall._spa_band_signature(legacy)["key"] == "band-key" and IORSSmall._spa_band_signature(legacy)["map_path"] is None


def test_map_lookup_config_require_and_the_legacy_default(tmp_path):
    from ispatialgym.sampling import reach_maps as RM

    _map(tmp_path, "r")  # written without config.preshape: the former batch rule
    lib = SimpleNamespace(identity_key=lambda: "obj-key")
    kw = dict(rest_id="r0", arm="right")
    assert RM.ReachMap.find(str(tmp_path), lib, 0.8, config_require={"preshape": "max"}, **kw) is not None
    assert RM.ReachMap.find(str(tmp_path), lib, 0.8, config_require={"preshape": "grasp"}, **kw) is None
    assert RM.MapConfig().preshape == "grasp" and RM.MapConfig().key()["preshape"] == "grasp"


def test_grasp_geometry_takes_one_opening_per_pose():
    import torch as th

    from ispatialgym.grasping import contact_policy as CP

    seen = []

    class _Checker:
        policy = SimpleNamespace(approach_step_m=0.05, sweep_steps=2)

        def check(self, T, support, q, floor=False):
            seen.append((round(q[0], 4), int(T.shape[0])))
            n = int(T.shape[0])
            return th.ones(n, dtype=th.bool), ["ok"] * n, {r: th.full((n,), 1.0) for r in CP.H.ROLES}

    T = th.eye(4).repeat(3, 1, 1)
    out = CP.check_grasp_geometry(_Checker(), T, None, [0.01, 0.01, 0.02], open_q=[0.02, 0.03, 0.02])
    assert bool(out["ok"].all())
    approach = [x for x in seen[:-6]]
    assert {q for q, _ in approach} == {0.02, 0.03} and all(n in (1, 2) for _, n in approach)  # grouped per opening, never the max for all


def test_the_approach_tail_is_policed_where_the_support_is_ignored(monkeypatch):
    from ispatialgym.sampling.v2 import ik as IK

    table = SimpleNamespace(name="table", aabb=([0, 0, 0], [1, 1, 0.8]))
    chk = IK.IKChecker.__new__(IK.IKChecker)
    chk.kit = SimpleNamespace(backend=SimpleNamespace(), stats={})
    chk.support_plane_z = {"table": 0.8}  # 2026-11-07: only registered supports are policed
    chk.approached_objects = lambda pose: []
    evals = []
    chk.support_policy_eval = lambda q, arm, sup, appr: evals.append(int(q)) or {"ok": int(q) != 9, "reason": "finger_penetration"}
    traj = list(range(11))  # configuration i = i
    args = (traj, 10, 0.1, "right", None, None, [8, 9], [1, 2], [])
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY", "grasp")
    why = chk._policy_tail(*args, ignore_end=[table], ignore_start=())
    assert "support policy on the approach tail" in why and evals == [8, 9] and chk.kit.stats["support_policy_tail_rejected"] == 1
    evals.clear()
    assert chk._policy_tail(*args, ignore_end=(), ignore_start=[table]) == "" and evals == [1, 2]  # backward tracking: the start's tail
    evals.clear()
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY_TAIL", "0")
    assert chk._policy_tail(*args, ignore_end=[table], ignore_start=()) == "" and evals == []
    monkeypatch.delenv("ISPATIALGYM_SUPPORT_POLICY_TAIL")
    chk.kit = SimpleNamespace(backend=SimpleNamespace(_attached={"right": {"obj": "cup"}}), stats={})
    assert chk._policy_tail(*args, ignore_end=[table], ignore_start=()) == ""  # scope "grasp": a holding arm is not policed
    monkeypatch.delenv("ISPATIALGYM_SUPPORT_POLICY")
    assert chk._policy_tail(*args, ignore_end=[table], ignore_start=()) == ""  # policy off: as before


def test_v2_only_refuses_legacy_grasps_and_campaign_links_serve(v2root, monkeypatch):
    _entry(v2root, "mug", "ppp", [1, 1, 1], legacy_key=None, rests={"r0": ([0, 0, 0, 1], 0.05, True)}, grasps=[("g", "r0")])
    p = os.path.join(v2root, "models")
    d = json.load(open(os.path.join(p, os.listdir(p)[0], "library.json")))
    d["legacy_keys"] = ["mug__ppp__t", "mug__ppp"]
    S.save_json(os.path.join(p, os.listdir(p)[0], "library.json"), d)
    GL._v2_index.cache_clear()
    idx = GL._v2_index(v2root)
    assert idx["mug__ppp__t"] == idx["mug__ppp"]  # one entry serves every linked key
    monkeypatch.setenv("ISPATIALGYM_V2_ONLY", "1")
    assert [g.name for g in GL.load_grasps("mug__ppp__t")] == ["g"]
    with pytest.raises(ValueError, match="v2 only"):
        GL.load_grasps("apple__omzprq")  # no v2 entry: the legacy library is not served


def test_v2_map_view_routes_by_kind_and_logs_requests(monkeypatch, caplog):
    import logging

    from ispatialgym.sampling.v2 import tabletop as TT

    calls = []
    monkeypatch.setattr(TT, "map_bands", lambda s, spec, top, arm=None, **kw: calls.append(kw) or "VIEW")
    monkeypatch.setattr(TT, "surface_band_kwargs", lambda s: {})
    s = SimpleNamespace(world=SimpleNamespace(floor_z=0.0, scene_model="house"))
    top = SimpleNamespace(name="table_0", z=0.8126)
    assert TT.v2_map_view(s, TT.BandSpec(object="a/b", grasp_key="k"), top) == "VIEW" and calls[-1] == {}
    tray = TT.BandSpec(object="a/b", grasp_key="k", placed=["tray/uekqey@focus_centred_edge_aligned"], z_support=0.8187)
    assert TT.v2_map_view(s, tray, top) == "VIEW" and calls[-1] == {"z_support": 0.8187, "exact_anchor": True, "plane": "tray_floor"}
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        assert TT.v2_map_view(s, TT.BandSpec(object="a/b", generic_pinch=True), top) is None
        assert TT.v2_map_view(s, TT.BandSpec(object="t/m", grasp_key="attach:lid__x", placed=["attach:lid/x"]), top) is None
    reqs = [r.getMessage() for r in caplog.records if r.getMessage().startswith("MAP_REQUEST")]
    assert len(reqs) == 2 and '"kind": "generic"' in reqs[0] and '"h": 0.8126' in reqs[0] and '"kind": "attach"' in reqs[1]


def test_left_contact_band_is_none_under_v2_only(monkeypatch):
    from ispatialgym.sampling.v2 import tabletop as TT

    monkeypatch.setenv("ISPATIALGYM_V2_ONLY", "1")
    s = SimpleNamespace(bands_left={})
    assert TT.load_left_contact_band(s, 0.12) is None and list(s.bands_left.values()) == [None]


def test_map_bands_miss_logs_the_map_to_build(v2root, tmp_path, monkeypatch, caplog):
    # i7-floor 2026-11-07: the config requirement reused the name of the MAP_REQUEST fields - a miss raised TypeError (``** None``) without
    # ISPATIALGYM_REACH_MAP_REQUIRE and logged the requirement instead of the map with it
    import logging

    from ispatialgym.sampling.v2 import tabletop as TT

    _entry(v2root, "apple", "xyz", [1, 1, 1], legacy_key="apple__xyz__t", rests={"r0": ([0, 0, 0, 1], 0.04, True)}, grasps=[("g0", "r0")])
    monkeypatch.setenv("ISPATIALGYM_REACH_MAPS", str(tmp_path / "maps"))
    for k in ("ISPATIALGYM_REACH_MAP_ANCHORS", "ISPATIALGYM_REACH_MAP_REQUIRE", "ISPATIALGYM_V2_ONLY"):
        monkeypatch.delenv(k, raising=False)
    s = SimpleNamespace(world=SimpleNamespace(floor_z=0.0, scene_model="house"), ikc=SimpleNamespace())
    top = SimpleNamespace(name="table_0", z=0.8126)
    spec = TT.BandSpec(object="apple/xyz", grasp_key="apple__xyz__t")
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        assert TT.map_bands(s, spec, top, "right") is None
        monkeypatch.setenv("ISPATIALGYM_REACH_MAP_REQUIRE", '{"preshape": "grasp"}')
        assert TT.map_bands(s, spec, top, "left") is None
    reqs = [json.loads(r.getMessage().split(" ", 1)[1]) for r in caplog.records if r.getMessage().startswith("MAP_REQUEST")]
    assert [(r["kind"], r["arm"], r["h"], r["support"], r["rest_id"]) for r in reqs] == [("tabletop", "right", 0.8126, "table_0", "r0"),
                                                                                         ("tabletop", "left", 0.8126, "table_0", "r0")]


def test_only_registered_supports_are_policed(monkeypatch):
    from ispatialgym.sampling.v2 import ik as IK

    table, wall = SimpleNamespace(name="table_0"), SimpleNamespace(name="walls")
    monkeypatch.delenv("ISPATIALGYM_SUPPORT_POLICY_REGISTERED", raising=False)
    assert IK.policed_supports([table, wall], None, planes={"table_0": 0.81}) == [table]  # a wall has no registered plane
    assert IK.policed_supports([table, wall], None) == [table, wall]  # no planes passed: as before
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY_REGISTERED", "0")
    assert IK.policed_supports([table, wall], None, planes={"table_0": 0.81}) == [table, wall]
