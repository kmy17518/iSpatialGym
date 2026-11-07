"""i7-lean 2026-11-07 (owner decision: the "lean" rest kind of the v2 grasp library, touching_03): the lean spec and its identity, the
lean settle's geometry and record, the rest kind in the schema (entry ids, compatibility, identity keys), the grasp library's lean / stable
separation (a lean rest never serves a stable lookup, nor the reverse; stable keys and choices unchanged), the contact policy's normal
obstacles, the candidates' board obstacle and the pipeline's job resolution. Simulator-free."""

import json
import math
import os

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import lean as LN
from ispatialgym.grasping import schema as S
from ispatialgym.sampling import grasp_library as GL

#: the legacy upright-shelf key's rest (runlogs/grasp_library_20260905/cands/hardback__acbrnv__bu.yaml): 15 deg toward the robot's left
HB_KEY = "hardback__acbrnv__bu"
HB_REST_WXYZ = [0.5609855267969309, 0.4304593345768794, 0.43045933457687935, -0.5609855267969309]
HB_BOX = (np.array([-0.11137, -0.08724, -0.0156]), np.array([0.11137, 0.08428, 0.0156]))  # the collision bbox (roles_all geometry.json)


def _box_vertices(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=float)


# ------------------------------------------------------------------------------------------------------------------ the spec


def test_lean_spec_identity_separates_angle_side_board_and_gap():
    a = LN.LeanSpec(11.0, "left")
    others = [LN.LeanSpec(20.0, "left"), LN.LeanSpec(11.0, "right"), LN.LeanSpec(11.0, "left", LN.Board(thickness_m=0.03)),
              LN.LeanSpec(11.0, "left", gap_m=0.004)]
    assert len({a.digest(), *[o.digest() for o in others]}) == 5 and len({a.tag(), *[o.tag() for o in others]}) == 5
    assert a.tag().startswith("lean_left_11_") and len(a.tag().split("_")[-1]) == 6 and a.rest_id() == a.tag()
    assert LN.LeanSpec(11, "left").digest() == a.digest()  # int / float spelling: the same identity
    j = a.to_json()
    assert j["kind"] == "lean" and j["deg"] == 11.0 and j["side"] == "left" and j["board"]["thickness_m"] == 0.02 and j["digest"] == a.digest()
    assert LN.LeanSpec.from_json(j) == a and LN.LeanSpec.from_json(json.loads(json.dumps(j))).tag() == a.tag()
    assert a.sign == 1.0 and LN.LeanSpec(20, "right").sign == -1.0
    for bad in ({"deg": 11, "side": "up"}, {"deg": 0, "side": "left"}, {"deg": 50, "side": "left"}):
        with pytest.raises(ValueError):
            LN.LeanSpec(**bad)


def test_lean_keys_split_and_never_parse_a_plain_key():
    k = LN.lean_key(HB_KEY, LN.LeanSpec(11, "left"))
    assert k == f"{HB_KEY}@{LN.LeanSpec(11, 'left').tag()}"
    assert LN.split_lean_key(k) == (HB_KEY, LN.LeanSpec(11, "left").tag()) and LN.is_lean_key(k)
    assert LN.split_lean_key(HB_KEY) == (HB_KEY, None) and not LN.is_lean_key(HB_KEY) and not LN.is_lean_key(None)
    assert LN.split_lean_key("attach:lid@x") == ("attach:lid@x", None)  # an '@' without a lean tag is no lean key


# ------------------------------------------------------------------------------------------------------------------ orientation math


def test_upright_and_lean_quat_round_trip_the_upright_shelf_key():
    q_key = LN.xyzw(HB_REST_WXYZ)
    lo = LN.lean_of(q_key)
    assert abs(lo["deg"] - 15.0) < 1e-6 and lo["side"] == 1.0 and abs(lo["toward_x_deg"]) < 1e-6
    q_up = LN.upright(q_key)
    up = LN.up_axis_local(q_key)
    assert np.allclose(S.quat_to_mat(q_up) @ up, [0, 0, 1.0], atol=1e-9)  # the same local axis, now vertical
    assert S.tilt_distance(LN.lean_quat(q_up, LN.LeanSpec(15.0, "left")), q_key) < 1e-6  # leaned back: the key's rest
    for deg, side in ((11.0, "left"), (20.0, "left"), (11.0, "right")):
        q = LN.lean_quat(q_up, LN.LeanSpec(deg, side))
        lo = LN.lean_of(q, up)
        assert abs(lo["deg"] - deg) < 1e-6 and lo["side"] == LN.SIDES[side] and abs(lo["toward_x_deg"]) < 1e-6
    # the sampler's re-lean convention (tabletop.releaned_quat: about rest-frame +x) gives the same orientation
    from ispatialgym.sampling.v2 import tabletop as TT

    r = TT.releaned_quat(th.tensor(q_key, dtype=th.float32), 1.0, 15.0, 1.0, 11.0).double().numpy()
    q11 = LN.lean_quat(q_up, LN.LeanSpec(11.0, "left"))
    assert min(np.abs(r - q11).max(), np.abs(r + q11).max()) < 1e-5


def test_initial_pose_sets_the_board_gap_beyond_the_extreme_point():
    V = _box_vertices(*HB_BOX)
    q_up = LN.upright(LN.xyzw(HB_REST_WXYZ))
    for side in ("left", "right"):
        spec = LN.LeanSpec(11.0, side)
        q0 = LN.lean_quat(q_up, spec)
        st = LN.initial_pose(V, q0, spec, slab_top_z=0.75, obj_xy=(0.1, -0.2), drop_m=0.003)
        W = V @ S.quat_to_mat(q0).T + np.asarray(st["position"])
        assert abs(W[:, 2].min() - 0.753) < 1e-9  # 3 mm above the slab
        ext = W[:, 1].max() if side == "left" else W[:, 1].min()
        assert abs((st["board_face_y"] - ext) * spec.sign - 0.002) < 1e-9  # the gap to the board's inner face
        assert abs(LN.board_face_y(st["board_centre"], st["board_dims"], spec.sign) - st["board_face_y"]) < 1e-12
        c, d = st["board_centre"], st["board_dims"]
        assert abs(c[2] - d[2] / 2 - 0.75) < 1e-12 and d == [0.40, 0.02, 0.40]  # standing on the slab
        assert c[0] - d[0] / 2 <= W[:, 0].min() and c[0] + d[0] / 2 >= W[:, 0].max()  # spanning the object's depth


def test_lean_record_and_its_acceptance():
    spec = LN.LeanSpec(11.0, "left")
    q_up = LN.upright(LN.xyzw(HB_REST_WXYZ))
    up = LN.up_axis_local(q_up)
    q = LN.lean_quat(q_up, LN.LeanSpec(11.6, "left"))  # the gap closed: a little more lean than requested
    p = np.array([0.0, 0.01, 0.86])
    rec = LN.lean_record(spec, q, p, [0.0, 0.07, 0.95], [0.4, 0.02, 0.4], up, board_load_N=0.9, weight_N=7.85)
    assert abs(rec["measured_deg"] - 11.6) < 1e-3 and rec["measured_side"] == "left" and rec["board_contact"]
    assert abs(rec["board_face_y_m"] - (0.06 - 0.01)) < 1e-9  # inner face 0.06 (centre 0.07 - half thickness), relative to the root
    assert rec["board_in_root"]["centre"] == [0.0, 0.06, 0.09] and rec["board_in_root"]["half"] == [0.2, 0.01, 0.2]
    assert rec["digest"] == spec.digest() and rec["tag"] == spec.tag()
    assert LN.lean_ok(rec) == (True, "ok")
    assert not LN.lean_ok(dict(rec, measured_deg=14.0))[0]  # leaned 3 deg further: another rest
    assert not LN.lean_ok(dict(rec, measured_side="right"))[0]
    assert not LN.lean_ok(LN.lean_record(spec, q, p, [0.0, 0.07, 0.95], [0.4, 0.02, 0.4], up, 0.05, 7.85))[0]  # the board carries nothing
    ob = LN.board_obstacle_in_rest(rec)
    assert ob.kind == "box" and ob.centre == (0.0, 0.06, 0.09) and ob.half == (0.2, 0.01, 0.2)
    assert float(ob.sdf(th.tensor([[0.0, 0.045, 0.09]]))[0]) > 0 and float(ob.sdf(th.tensor([[0.0, 0.061, 0.09]]))[0]) < 0


# ------------------------------------------------------------------------------------------------------------------ the schema


def _rest(rid, q, h, *, lean=None, stable=True):
    d = {"schema": S.REST, "rest_id": rid, "quat_xyzw": list(q), "h_root": h, "stable": stable}
    if lean is not None:
        d.update({"rest_kind": "lean", "lean": lean})
    return d


def _lean_block(spec, face=0.05):
    return {**spec.to_json(), "measured_deg": spec.deg, "measured_side": spec.side, "board_face_y_m": face,
            "board_in_root": {"centre": [0.0, face + 0.01, 0.09], "half": [0.2, 0.01, 0.2]}, "board_pose": {"centre": [0, 0, 0], "dims": [0.4, 0.02, 0.4]}}


def test_entry_ids_and_job_dirs_keep_stable_ones_and_separate_lean_ones(tmp_path):
    assert S.entry_id("hardback", "acbrnv", [1, 1, 1]) == "hardback__acbrnv__s1"  # unchanged
    spec = LN.LeanSpec(11, "left")
    assert S.entry_id("hardback", "acbrnv", [1, 1, 1], lean=spec.to_json()) == f"hardback__acbrnv__s1__{spec.tag()}"
    stable_job = {"category": "hardback", "model": "acbrnv", "scale": [1, 1, 1], "lean": spec.to_json()}  # no rest_kind: a stable job
    assert S.job_dir(stable_job, str(tmp_path)).endswith("models/hardback__acbrnv__s1")
    base = {"category": "hardback", "model": "acbrnv", "scale": [1.0, 1.0, 1.0], "dataset": "behavior-1k-assets"}
    j = LN.make_job(base, spec, LN.upright(LN.xyzw(HB_REST_WXYZ)), base_key=HB_KEY)
    assert j["id"] == f"hardback__acbrnv__s1__{spec.tag()}" and S.job_dir(j, str(tmp_path)).endswith("models/" + j["id"])
    assert j["rest_kind"] == "lean" and j["legacy_key"] is None and j["lean_keys"] == [HB_KEY] and LN.is_lean_job(j)
    j20 = LN.make_job(base, LN.LeanSpec(20, "left"), LN.upright(LN.xyzw(HB_REST_WXYZ)), base_key=HB_KEY)
    assert j20["id"] != j["id"]


def test_rest_kind_compatibility_never_crosses(tmp_path):
    spec = LN.LeanSpec(11, "left")
    lean_lib = S.ModelLibrary(path="x/library.json", data={"schema": S.LIBRARY, "category": "hardback", "model": "acbrnv", "scale": [1, 1, 1],
                                                          "rest_kind": "lean", "grasps": []},
                              rests={spec.tag(): S.Rest.from_record(_rest(spec.tag(), [0, 0, 0, 1], 0.11, lean=_lean_block(spec)))})
    assert lean_lib.rests[spec.tag()].kind == "lean" and lean_lib.rest_kind == "lean"
    with pytest.raises(S.IncompatibleRecord):
        S.check_compatible(lean_lib, rest_id=spec.tag())  # a stable consumer
    S.check_compatible(lean_lib, rest_id=spec.tag(), rest_kind="lean", lean=spec.to_json())
    with pytest.raises(S.IncompatibleRecord):
        S.check_compatible(lean_lib, rest_id=spec.tag(), rest_kind="lean", lean=LN.LeanSpec(20, "left").to_json())  # another lean
    stable_lib = S.ModelLibrary(path="y/library.json", data={"schema": S.LIBRARY, "category": "c", "model": "m", "grasps": []},
                                rests={"r0": S.Rest.from_record(_rest("r0", [0, 0, 0, 1], 0.02))})
    assert stable_lib.rests["r0"].kind == "stable" and stable_lib.rests["r0"].lean is None and stable_lib.rest_kind == "stable"
    S.check_compatible(stable_lib, rest_id="r0")
    with pytest.raises(S.IncompatibleRecord):
        S.check_compatible(stable_lib, rest_id="r0", rest_kind="lean")


def test_stable_identity_keys_do_not_move_and_lean_ones_carry_the_lean():
    data = {"schema": S.LIBRARY, "category": "c", "model": "m", "scale": [1, 1, 1], "dataset": "d", "policy_key": "pk",
            "grasps": [{"name": "g", "rest_id": "r0", "family_id": None, "T_object_eef": {"position": [0, 0, 0.1], "orientation_xyzw": [1, 0, 0, 0]},
                        "finger_q": [0.01, 0.01], "width_m": 0.02}], "families": {}}
    lib = S.ModelLibrary(path="p", data=data, rests={"r0": S.Rest.from_record(_rest("r0", [0, 0, 0, 1], 0.02))})
    old = {"schema": S.LIBRARY, "category": "c", "model": "m", "scale": [1.0, 1.0, 1.0], "dataset": "d", "policy_key": "pk",
           "rests": {"r0": {"quat_xyzw": [0.0, 0.0, 0.0, 1.0], "h_root": 0.02}},
           "grasps": S.digest([{k: g.get(k) for k in ("name", "rest_id", "family_id", "T_object_eef", "finger_q", "width_m")} for g in data["grasps"]]),
           "families": S.digest({})}
    assert lib.identity() == old and lib.identity_key() == S.digest(old, 16)  # the formula before the lean kind
    spec = LN.LeanSpec(11, "left")
    ll = S.ModelLibrary(path="p", data=data, rests={"r0": S.Rest.from_record(_rest("r0", [0, 0, 0, 1], 0.02, lean=_lean_block(spec)))})
    ident = ll.identity()["rests"]["r0"]
    assert ident["rest_kind"] == "lean" and ident["lean"] == spec.digest() and ident["board_face_y_m"] == 0.05
    assert ll.identity_key() != lib.identity_key()
    ll2 = S.ModelLibrary(path="p", data=data, rests={"r0": S.Rest.from_record(_rest("r0", [0, 0, 0, 1], 0.02, lean=_lean_block(LN.LeanSpec(20, "left"))))})
    assert ll2.identity_key() != ll.identity_key()


# ------------------------------------------------------------------------------------------------------------------ the grasp library


def _write_entry(root, *, legacy_key=None, lean=None, rests, grasps, lean_keys=None):
    eid = S.entry_id("hardback", "acbrnv", [1, 1, 1], lean=None if lean is None else lean.to_json())
    d = os.path.join(root, "models", eid)
    os.makedirs(d, exist_ok=True)
    data = {"schema": S.LIBRARY, "category": "hardback", "model": "acbrnv", "scale": [1.0, 1.0, 1.0], "arm": "right", "legacy_key": legacy_key,
            "policy_key": CP.DEFAULT.key("right"), "rests": rests,
            "grasps": [{"name": n, "rest_id": rid, "T_object_eef": {"position": [0, 0, z], "orientation_xyzw": [1, 0, 0, 0]}, "width_m": 0.031}
                       for n, rid, z in grasps]}
    if lean is not None:
        data.update({"rest_kind": "lean", "lean": lean.to_json(), "lean_keys": list(lean_keys or [])})
    S.save_json(os.path.join(d, "library.json"), data)
    return os.path.join(d, "library.json")


@pytest.fixture
def v2root(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_GRASP_LIBRARY_V2", str(tmp_path))
    for k in ("ISPATIALGYM_V2_ONLY",):
        monkeypatch.delenv(k, raising=False)
    GL._v2_index.cache_clear()
    GL._v2_lean_index.cache_clear()
    yield str(tmp_path)
    GL._v2_index.cache_clear()
    GL._v2_lean_index.cache_clear()


def _two_leans(root, with_stable=True):
    s11, s20 = LN.LeanSpec(11, "left"), LN.LeanSpec(20, "left")
    q11 = LN.lean_quat(LN.upright(LN.xyzw(HB_REST_WXYZ)), s11).tolist()
    q20 = LN.lean_quat(LN.upright(LN.xyzw(HB_REST_WXYZ)), s20).tolist()
    # a lean entry whose legacy_key ALSO names the key (defence in depth: lean entries never enter the stable index)
    p11 = _write_entry(root, lean=s11, legacy_key=HB_KEY, lean_keys=[HB_KEY], rests={s11.tag(): _rest(s11.tag(), q11, 0.11, lean=_lean_block(s11))},
                       grasps=[("side_0", s11.tag(), 0.01), ("side_1", s11.tag(), 0.02)])
    p20 = _write_entry(root, lean=s20, lean_keys=[HB_KEY], rests={s20.tag(): _rest(s20.tag(), q20, 0.105, lean=_lean_block(s20, 0.07))},
                       grasps=[("side_0", s20.tag(), 0.03)])
    ps = None
    if with_stable:
        ps = _write_entry(root, legacy_key=HB_KEY, rests={"r0": _rest("r0", [0, 0, 0, 1], 0.0156)}, grasps=[("top_0", "r0", 0.05)])
    return s11, s20, q11, q20, p11, p20, ps


def test_stable_lookups_are_unchanged_by_lean_entries(v2root):
    s11, s20, q11, q20, p11, p20, ps = _two_leans(v2root)
    assert GL._v2_index(v2root) == {HB_KEY: ps}  # the lean entries never enter the stable index
    lib = GL.v2_entry(HB_KEY)
    assert lib.path == ps and GL.v2_rest_id(lib) == "r0" and lib.rest_kind == "stable"
    assert [g.name for g in GL.load_grasps(HB_KEY)] == ["top_0"]
    assert np.allclose(GL.rest_quat_xyzw(HB_KEY).numpy(), [0, 0, 0, 1])
    # the lean keys: each served by its own lean entry and rest only
    k11, k20 = LN.lean_key(HB_KEY, s11), LN.lean_key(HB_KEY, s20)
    l11 = GL.v2_entry(k11)
    assert l11.path == p11 and l11.rest_kind == "lean" and GL.v2_rest_id(l11) is None  # a stable lookup of a lean entry: nothing
    assert GL.v2_rest_id(l11, lean=s11.tag()) == s11.tag() and GL.v2_rest_id(l11, lean=s20.tag()) is None
    assert GL.v2_rest_id(lib, lean=s11.tag()) is None  # a lean lookup of the stable entry: nothing
    assert [g.name for g in GL.load_grasps(k11)] == ["side_0", "side_1"] and [g.key for g in GL.load_grasps(k11)] == [k11, k11]
    assert [g.name for g in GL.load_grasps(k20)] == ["side_0"] and float(GL.load_grasps(k20)[0].position[2]) == pytest.approx(0.03)
    assert np.allclose(GL.rest_quat_xyzw(k11).numpy(), q11, atol=1e-6) and np.allclose(GL.rest_quat_xyzw(k20).numpy(), q20, atol=1e-6)
    assert [t for t, _ in GL.v2_lean_tags(HB_KEY)] == [s11.tag(), s20.tag()] and GL.v2_lean_tags("other__key") == []
    # an unknown lean (another board) is a miss, never the stable entry nor another lean
    k_other = LN.lean_key(HB_KEY, LN.LeanSpec(11, "left", LN.Board(thickness_m=0.03)))
    assert GL.v2_entry(k_other) is None
    with pytest.raises(ValueError):
        GL.load_grasps(k_other)
    with pytest.raises(ValueError):
        GL.rest_quat_xyzw(k_other)


def test_lean_entries_alone_never_serve_the_stable_key(v2root, monkeypatch):
    s11, *_ = _two_leans(v2root, with_stable=False)
    assert GL.v2_entry(HB_KEY) is None and GL._v2_index(v2root) == {}
    monkeypatch.setenv("ISPATIALGYM_V2_ONLY", "1")
    with pytest.raises(ValueError, match="v2 only"):
        GL.load_grasps(HB_KEY)  # v2only: no v2 entry for the stable key - the lean entries do not stand in
    assert [g.name for g in GL.load_grasps(LN.lean_key(HB_KEY, s11))] == ["side_0", "side_1"]


def test_the_oracle_resolves_a_lean_grasp_by_its_key(v2root):
    from ispatialgym.robot import conventions as C

    s11, *_ = _two_leans(v2root)
    k11 = LN.lean_key(HB_KEY, s11)
    g = C.grasp_entry(k11, "side_1")  # execute.grasp_pinch_width's width lookup (no oracle change: the key names the lean entry)
    assert g.name == "side_1" and g.meta["width"] == pytest.approx(0.031) and g.meta["rest_id"] == s11.tag()
    assert C.grasp_entry(HB_KEY, "top_0").name == "top_0"
    with pytest.raises(KeyError):
        C.grasp_entry(HB_KEY, "side_1")  # the stable key never sees the lean grasps
    assert np.allclose(C.rest_quat(k11).numpy(), GL.rest_quat_xyzw(k11).numpy())


# ------------------------------------------------------------------------------------------------------------------ the contact policy


@pytest.fixture(scope="module")
def checker():
    return CP.HandChecker("right", CP.DEFAULT, spacing=0.003)


def test_a_normal_obstacle_allows_no_finger_touch_the_policed_support_2mm(checker):
    P, R = checker.points((0.03, 0.03))
    roles = list(__import__("ispatialgym.grasping.hand_geometry", fromlist=["ROLES"]).ROLES)
    fi = roles.index("finger")
    tip = P[(R == fi) & (P[:, 2] > 0.0)]
    y_f = float(tip[:, 1].max())  # the finger's outer face near the tips (eef frame = world: identity pose)
    T = th.eye(4).unsqueeze(0)
    board = CP.Support.box([0.0, y_f - 0.001 + 0.01, 0.015], [0.05, 0.01, 0.015], name="board")  # 1 mm into the finger, tips' height only
    ok_s, reason_s, cl = checker.check(T, board, (0.03, 0.03))
    assert -0.002 < float(cl["finger"][0]) < 0.0 and all(float(cl[r][0]) > 0 for r in ("palm", "wrist_camera", "wrist"))
    assert bool(ok_s[0]) and reason_s[0] == "ok"  # as the policed support: within the fingers' 2 mm allowance
    ok_o, reason_o, _ = checker.check_obstacles(T, [board], (0.03, 0.03))
    assert not bool(ok_o[0]) and reason_o == ["obstacle_contact"]  # as a normal obstacle: no allowance
    far = CP.Support.box([0.0, y_f + 0.002 + 0.01, 0.015], [0.05, 0.01, 0.015])
    assert bool(checker.check_obstacles(T, [far], (0.03, 0.03))[0][0]) and bool(checker.check_obstacles(T, [], (0.03, 0.03))[0][0])
    plane = CP.Support.plane(-0.5)
    ok, reasons, cl2 = checker.check_with_obstacles(T, plane, [board], (0.03, 0.03))
    assert not bool(ok[0]) and reasons == ["obstacle_contact"]
    ok0, r0, c0 = checker.check(T, plane, (0.03, 0.03))
    ok1, r1, c1 = checker.check_with_obstacles(T, plane, (), (0.03, 0.03))
    assert bool(ok0[0]) == bool(ok1[0]) and r0 == r1 and float(c0["finger"][0]) == float(c1["finger"][0])


def test_check_grasp_geometry_with_and_without_obstacles(checker):
    T = th.eye(4).unsqueeze(0)
    T[0, 2, 3] = 0.5  # high above the policed plane (the wrist reaches 0.244 m behind the eef: 0.1 m more at the pre-grasp)
    plane = CP.Support.plane(0.0)
    base = CP.check_grasp_geometry(checker, T, plane, [0.016], open_q=0.03)
    same = CP.check_grasp_geometry(checker, T, plane, [0.016], open_q=0.03, obstacles=())
    assert bool(base["ok"][0]) and base["reason"] == same["reason"] and "obstacle_min" not in base and "obstacle_min" not in same
    assert float(base["finger_min"][0]) == float(same["finger_min"][0])
    # a board crossing the approach (the pre-grasp 10 cm back along -z): fails at the approach with obstacle_contact
    bar = CP.Support.box([0.0, 0.0, 0.45], [0.1, 0.1, 0.005], name="board_above")
    res = CP.check_grasp_geometry(checker, T, plane, [0.016], open_q=0.03, obstacles=[bar])
    assert not bool(res["ok"][0]) and res["reason"][0] == "obstacle_contact" and (res["stage"][0] == "pre_grasp" or res["stage"][0].startswith("approach"))
    assert float(res["obstacle_min"][0]) < 0.0
    clear = CP.check_grasp_geometry(checker, T, plane, [0.016], open_q=0.03, obstacles=[CP.Support.box([1.0, 0, 0.5], [0.05, 0.05, 0.05])])
    assert bool(clear["ok"][0]) and float(clear["obstacle_min"][0]) > 0.5


def test_spheres_clear_judges_every_link_the_fingers_included():
    links = ["right_gripper_finger_link1", "right_gripper_link", "torso_link1"]
    S_ = th.tensor([[[0, 0.005, 0.5, 0.01], [0, 0.3, 0.5, 0.04], [0, -0.3, 0.5, 0.1]]], dtype=th.float32)
    wall = CP.Support.box([0.0, 0.05, 0.5], [0.2, 0.04, 0.2], name="wall")  # inner face at y = 0.01
    ok, worst = CP.spheres_clear(S_, links, [wall])
    assert not bool(ok[0]) and worst[0] == "right_gripper_finger_link1"  # the finger sphere 5 mm off the face, r 10 mm
    okp, _ = CP.sphere_support_ok(S_, links, wall, "right")
    assert bool(okp[0])  # as the policed support the finger sphere is not judged (rule 4)
    assert bool(CP.spheres_clear(S_, links, [wall], skip_links=["right_gripper_finger_link1"])[0][0])
    assert bool(CP.spheres_clear(S_, links, [])[0][0])


# ------------------------------------------------------------------------------------------------------------------ pipeline + library


def test_resolve_job_reads_the_lean_settle_and_leaves_other_jobs(tmp_path):
    spec = LN.LeanSpec(11, "left")
    base = {"category": "hardback", "model": "acbrnv", "scale": [1, 1, 1]}
    j = LN.make_job(base, spec, LN.upright(LN.xyzw(HB_REST_WXYZ)), base_key=HB_KEY)
    d = str(tmp_path)
    assert LN.resolve_job(j, d) is j  # no record yet
    q = LN.lean_quat(LN.upright(LN.xyzw(HB_REST_WXYZ)), LN.LeanSpec(11.4, "left"))
    rest = _rest(spec.tag(), q.tolist(), 0.1103, lean=dict(_lean_block(spec), measured_deg=11.4))
    S.save_json(os.path.join(d, LN.SETTLE_FILE), {"schema": LN.SETTLE_SCHEMA, "rest": rest})
    r = LN.resolve_job(j, d)
    assert r["generation_rest_wxyz"] == LN.wxyz(q) and r["lean_rest"]["board_face_y_m"] == 0.05 and r["lean_rest"]["measured_deg"] == 11.4
    assert r["lean_rest"]["lean_digest"] == spec.digest() and r["id"] == j["id"]
    stable = {"category": "c", "model": "m", "scale": [1, 1, 1]}
    assert LN.resolve_job(stable, d) is stable


def test_candidates_of_a_lean_job_need_the_measured_rest():
    from ispatialgym.grasping import candidates as CA

    j = LN.make_job({"category": "hardback", "model": "acbrnv", "scale": [1, 1, 1]}, LN.LeanSpec(11, "left"), [0, 0, 0, 1.0], base_key=HB_KEY)
    with pytest.raises((ValueError, FileNotFoundError)):
        CA._generate(dict(j), "/nonexistent", CA.CandidateConfig(), CP.DEFAULT, __import__("ispatialgym.grasping.symmetry", fromlist=["x"]).SymmetryConfig(), "right")


def test_build_library_marks_a_lean_entry():
    from ispatialgym.grasping import validate_tabletop as VT

    spec = LN.LeanSpec(11, "left")
    j = LN.make_job({"category": "hardback", "model": "acbrnv", "scale": [1, 1, 1]}, spec, [0, 0, 0, 1.0], base_key=HB_KEY)
    cand = {"arm": "right", "pipeline": "a1v2", "candidates": [{"name": "side_0", "mode": "side", "T_object_eef": {"position": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1]},
                                                                  "width_m": 0.031, "preshape_q": 0.0255, "family_id": None}], "families": {}}
    trial = {"verdict": "valid", "candidate": "side_0", "rest_id": spec.tag(), "achieved": {"T_object_eef_at_attach": {"position": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1]}},
             "closure": {"fingers_final_m": [0.015, 0.015]}, "forces": {"finger_slab": {"max": 0.0}, "obj_slab_ratio": {"max": 1.0}}}
    lib = VT.build_library(j, cand, {spec.tag(): {"rest_kind": "lean"}}, [trial], CP.DEFAULT, {})
    assert lib["rest_kind"] == "lean" and lib["lean"]["tag"] == spec.tag() and lib["lean_keys"] == [HB_KEY] and lib["legacy_key"] is None
    assert lib["grasps"][0]["rest_id"] == spec.tag()
    st = VT.build_library({"category": "c", "model": "m", "scale": [1, 1, 1], "legacy_key": "c__m"}, cand, {"r0": {}}, [dict(trial, rest_id="r0")], CP.DEFAULT, {})
    assert "rest_kind" not in st and "lean" not in st and st["legacy_key"] == "c__m"  # a stable entry's record as before


def test_lean_manifest_from_the_legacy_upright_key(tmp_path):
    from ispatialgym.grasping import manifest as MF

    if not os.path.exists(os.path.join(MF.LEGACY_ROOT, "cands", f"{HB_KEY}.yaml")):
        pytest.skip("legacy library not in this checkout")
    out = str(tmp_path / "lean_jobs.json")
    LN.main(["manifest", "--legacy-key", HB_KEY, "--deg", "11", "20", "--side", "left", "--out", out])
    jobs = json.load(open(out))
    assert [j["lean"]["deg"] for j in jobs] == [11.0, 20.0] and {j["lean"]["side"] for j in jobs} == {"left"}
    assert all(j["id"] == f"hardback__acbrnv__s1__{j['lean']['tag']}" and j["lean_keys"] == [HB_KEY] and j["legacy_key"] is None for j in jobs)
    assert jobs[0]["lean_base"]["key_lean_deg"] == pytest.approx(15.0) and jobs[0]["lean_base"]["key_lean_side"] == "left"
    q0 = LN.lean_quat(LN.base_quat(jobs[0]), LN.spec_of(jobs[0]))
    lo = LN.lean_of(q0)
    assert abs(lo["deg"] - 11.0) < 1e-4 and lo["side"] == 1.0
    assert abs(math.degrees(math.acos(min(1.0, (S.quat_to_mat(LN.base_quat(jobs[0])) @ LN.up_axis_local(LN.xyzw(HB_REST_WXYZ)))[2])))) < 1e-4
