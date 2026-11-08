"""Simulator-free checks of the grasp library v2 records, the exact height anchors, the symmetry families and the reach-map views
(``ispatialgym.grasping`` / ``ispatialgym.sampling.{height_anchors,reach_maps,map_sampling}``)."""

import json
import math
import os

import numpy as np
import pytest

from ispatialgym.grasping import schema as S
from ispatialgym.sampling import height_anchors as HA


# ------------------------------------------------------------------------------------------------ rest placement


def _rest(quat=(0, 0, 0, 1.0), h=0.031, rid="r0", stable=True):
    return S.Rest.from_record({"schema": S.REST, "rest_id": rid, "quat_xyzw": list(quat), "h_root": h, "reference_yaw": 0.0, "stable": stable})


def test_placement_uses_h_root_directly_at_any_height_and_yaw():
    tilt = S.mat_to_quat(np.array([[1, 0, 0], [0, math.cos(0.3), -math.sin(0.3)], [0, math.sin(0.3), math.cos(0.3)]]))
    r = _rest(tilt, 0.0425)
    T_oe = S.pose_mat([0.01, 0.0, 0.05], S.mat_to_quat(np.diag([1.0, -1.0, -1.0])))
    for h in (0.4, 0.8126, 1.0505):
        for yaw in (0.0, 1.1, -2.5):
            T = S.place_object(r, yaw, (0.6, -0.1), h)
            assert np.allclose(T[:3, 3], [0.6, -0.1, h + 0.0425])  # no 3 mm, no bottom offset
            R_exp = S.quat_to_mat(S.quat_yaw(yaw)) @ S.quat_to_mat(tilt)
            assert np.allclose(T[:3, :3], R_exp, atol=1e-9)
            assert np.allclose(S.world_eef(r, yaw, (0.6, -0.1), h, T_oe), T @ T_oe)
    # a yaw only turns the object about the vertical: the tilt (yaw-free rest) is unchanged
    assert S.tilt_distance(S.mat_to_quat(S.place_object(r, 0.7, (0, 0), 0.5)[:3, :3]), tilt) < 1e-6


def test_incompatible_library_records_are_rejected(tmp_path):
    root = str(tmp_path)
    d = S.model_dir("teacup", "wopjex", None, root)
    lib = {"schema": S.LIBRARY, "category": "teacup", "model": "wopjex", "scale": [1, 1, 1], "policy_key": "isg-support-contact-v1-abc",
           "rests": {"r0": {"schema": S.REST, "rest_id": "r0", "quat_xyzw": [0, 0, 0, 1], "h_root": 0.048, "stable": True}},
           "grasps": [{"name": "top_0", "rest_id": "r0", "T_object_eef": {"position": [0, 0, 0.1], "orientation_xyzw": [1, 0, 0, 0]}}]}
    S.save_json(os.path.join(d, "library.json"), lib)
    assert S.load_model("teacup", "wopjex", None, root) is not None
    with pytest.raises(S.IncompatibleRecord):
        S.load_model("teacup", "wopjex", None, root, require={"policy_key": "isg-support-contact-v1-other"})
    with pytest.raises(S.IncompatibleRecord):
        S.load_model("teacup", "wopjex", None, root, require={"rest_id": "r1"})
    assert S.load_model("teacup", "wopjex", [0.8, 0.8, 0.8], root) is None  # another scale is another entry, never this one
    lib2 = dict(lib, rests={"r0": {**lib["rests"]["r0"], "stable": False}})
    S.save_json(os.path.join(d, "library.json"), lib2)
    with pytest.raises(S.IncompatibleRecord):
        S.load_model("teacup", "wopjex", None, root, require={"rest_id": "r0"})
    lib3 = dict(lib, schema="isg-grasp-library/1")
    S.save_json(os.path.join(d, "library.json"), lib3)
    with pytest.raises(S.IncompatibleRecord):
        S.load_model("teacup", "wopjex", None, root)


def test_legacy_rest_hints_are_not_rest_records():
    assert not S.legacy_has_rest({"rest_quat_wxyz": [1, 0, 0, 0], "rest_pose": "x_down"})
    with pytest.raises(S.IncompatibleRecord):
        S.Rest.from_record({"rest_quat_wxyz": [1, 0, 0, 0]})


def test_entry_ids_separate_scale_constraint_and_rest():
    a = S.entry_id("lid", "pbmsji", [0.65] * 3, "attachment_free")
    b = S.entry_id("lid", "pbmsji", None, "attachment_free")
    c = S.entry_id("sunflower", "dkajbh", None, "flower_stem_top", [0.906, -0.42, -0.046, -0.021])
    assert a == "lid__pbmsji__s0.65__attachment_free" and b == "lid__pbmsji__s1__attachment_free" and c.startswith("sunflower__dkajbh__s1__flower_stem_top__r")
    assert S.scale_tag([1, 1, 1.75]) == "s1x1x1.75"


# ------------------------------------------------------------------------------------------------ height anchors


def _surf(scene, name, h, prec=1e-4, surface="top", subtasks=("t",)):
    return HA.Surface(scene, name, surface, "table", h, prec, "test", 0.0, "scene_floor_0", subtasks=list(subtasks))


def test_distinct_heights_in_one_bin_split_and_identical_heights_share():
    ss = [_surf("a", "t1", 0.7351), _surf("a", "t2", 0.7362), _surf("b", "t3", 0.7351), _surf("c", "t4", 0.7600)]
    A = HA.AnchorSet.from_surfaces(ss)
    assert [a["key"] for a in A.anchors] == ["0.7351", "0.7362", "0.7600"]
    assert sorted(A.anchors[0]["surfaces"]) == ["a/t1/top", "b/t3/top"]
    assert A.bins()["0.72"] == ["0.7351", "0.7362"]  # both in the nominal 0.72-0.74 bin, still two anchors


def test_ranges_midpoints_ties_and_coverage():
    A = HA.AnchorSet.from_surfaces([_surf("a", "t1", 0.7351), _surf("a", "t2", 0.7362), _surf("a", "t3", 0.8126)])
    a, k = A.lookup(0.7351)
    assert k == "exact" and a["key"] == "0.7351"
    a, k = A.lookup((0.7351 + 0.7362) / 2)  # the midpoint of the first two: the lower anchor
    assert k == "approx" and a["key"] == "0.7351"
    a, k = A.lookup(0.7358)
    assert a["key"] == "0.7362"
    a, k = A.lookup(0.7700)  # 3.4 cm from 0.7362 and 4.3 cm from 0.8126: beyond the 1 cm bound
    assert a is None and k == "uncovered"
    a, k = A.lookup(0.8026)  # 1.0 cm below 0.8126: the clipped range's edge
    assert a is not None and a["key"] == "0.8126"
    for x in A.anchors:  # every range holds its own anchor and is at most 2 cm wide
        assert x["range"][0] <= x["height"] <= x["range"][1] and x["range"][1] - x["range"][0] <= 0.02 + 1e-12


def test_exact_lookup_prefers_the_finest_anchor_over_a_coarse_one():
    # of-v2maps 2026-11-06: $GBH/anchors.json holds an interior board authored at 0.81 (1e-2 m) and the breakfast table's measured top
    # 0.8126 (1e-4 m); 0.8126 is "exact" for both keys and the height-sorted scan returned 0.81 - every map lookup there missed
    A = HA.AnchorSet.from_surfaces([_surf("a", "board", 0.81, prec=1e-2, surface="interior"), _surf("a", "t1", 0.8126), _surf("a", "t2", 0.7853)])
    a, k = A.lookup(0.8126)
    assert k == "exact" and a["key"] == "0.8126"
    a, k = A.lookup(0.81)  # the coarse anchor still resolves its own height
    assert k == "exact" and a["key"] == "0.81"
    a, k = A.lookup(0.7853)
    assert k == "exact" and a["key"] == "0.7853"


def test_inventory_rows_dedup_floor_datum_and_interior_planes():
    surfaces, rep = HA.build_surfaces([os.path.join(os.path.dirname(HA.INVENTORY), "..", "ispatialgym", "assets", "icl", "roles", "candidates.csv")])
    tops = [s for s in surfaces if s.surface == "top"]
    assert len({s.id for s in tops}) == len(tops)  # one surface per support whatever the number of task rows
    assert all(len(s.rows) >= 1 for s in tops)
    assert not rep["missing_inventory"]
    inter = [s for s in surfaces if s.surface.startswith("interior@")]
    assert all(s.status == "unmeasured_interior" and s.precision == 1e-2 for s in inter)
    # i7-goalmaps 2026-11-08 (the floor datum fix): the 7 interior planes of candidates.csv were the countertop / shelf bindings (0.81 -
    # 1.22 m) seen against tops measured from a cabinet top; measured from the floor they lie within 3 cm of those tops and name them
    assert not inter
    bt = [s for s in tops if s.support == "breakfast_table_rhjoby_0"][0]
    assert bt.key == "0.8126" and bt.floor_z == 0.0
    assert HA.key(0.81260000001) == "0.8126" and HA.key(0.8125) != HA.key(0.8126)


# ------------------------------------------------------------------------------------------------ symmetry


def _cyl():
    import trimesh

    return trimesh.creation.cylinder(radius=0.04, height=0.1, sections=96)


def test_rotational_family_of_a_rim_pinch_and_dedup():
    from ispatialgym.grasping import symmetry as SY

    m = _cyl()
    idx = SY.SurfaceIndex(m, n=20000)
    # a side pinch of the cylinder wall: approach -x at the wall (+x side), closing along +-y... pinch across the wall thickness is not a
    # solid cylinder's, so pinch across the whole cylinder from the top instead: approach -z, closing y, pads at mid height
    T = np.eye(4)
    T[:3, :3] = np.diag([-1.0, 1.0, -1.0])  # x = -x, y = y, z = -z: approach down
    T[:3, 3] = [0.0, 0.0, -0.01186 * -1 + 0.0]

    def width(Ts):
        return np.full(len(Ts), 0.08)

    def clear(Ts):
        return np.ones(len(Ts))

    fam = SY.detect(T, m, idx, width, clear, rest_R=np.eye(4))
    assert fam["type"] == "rotational" and fam["interval"] == [-math.pi, math.pi] and fam["vertical_yaw_equivalent"]
    # members of the family fold into the representative, a stranger does not
    cands = [{"name": "a", "T": T}, {"name": "b", "T": S.family_member_T(fam, T, 0.7)}, {"name": "c", "T": SY.twin_T(T)}]
    T2 = T.copy()
    T2[:3, 3] += [0.0, 0.0, 0.03]
    cands.append({"name": "d", "T": T2 if False else S.pose_mat([0.2, 0, 0], [0, 0, 0, 1])})
    reps, fams = SY.dedup(cands, {"a": fam, "b": fam, "c": fam, "d": {"type": "none"}})
    assert [r["name"] for r in reps] == ["a", "d"]
    rel = {m_["name"]: m_["relation"] for m_ in fams[reps[0]["family_id"]]["members"]}
    assert rel["b"] == "member" and rel["c"] == "twin"
    assert fams[reps[0]["family_id"]]["physics"] == "representative_only"


def test_translational_family_of_a_bar():
    import trimesh

    from ispatialgym.grasping import symmetry as SY

    bar = trimesh.creation.box(extents=[0.30, 0.02, 0.02])
    idx = SY.SurfaceIndex(bar, n=20000)
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])  # approach down, closing across the bar (y), pad-width axis along the bar (x)
    T[:3, 3] = [0.0, 0.0, 0.0 - 0.01186]
    fam = SY.detect(T, bar, idx, lambda Ts: np.full(len(Ts), 0.02), lambda Ts: np.ones(len(Ts)), rest_R=np.eye(4))
    assert fam["type"] == "translational"
    lo, hi = fam["interval"]
    assert hi - lo > 0.2 and abs(abs(fam["axis"][0]) - 1.0) < 1e-6
    samples = S.family_samples(fam, 4)
    assert samples[0] == lo and samples[-1] == hi and len(samples) == 4


# ------------------------------------------------------------------------------------------------ reach-map views


class _Map:
    """A hand-built ReachMap stand-in (no files)."""

    def __init__(self):
        from ispatialgym.sampling import reach_maps as RM

        self.RM = RM
        cells = np.array([[0.5, 0.0], [0.58, 0.0], [0.5, 0.08]], dtype=np.float32)
        self.cells, self.yaws = cells, np.array([2 * math.pi * k / 8 for k in range(8)], dtype=np.float32)
        g = [{"name": "top_0", "base": "top_0", "twin": False, "family_id": "f0", "param": 0.0},
             {"name": "top_0@+0.5", "base": "top_0", "twin": False, "family_id": "f0", "param": 0.5},
             {"name": "top_0@+1.0", "base": "top_0", "twin": False, "family_id": "f0", "param": 1.0},
             {"name": "top_0@+1.5", "base": "top_0", "twin": False, "family_id": "f0", "param": 1.5},
             {"name": "side_0", "base": "side_0", "twin": False, "family_id": "f1", "param": 0.0}]
        st = np.full((3, 8, 5), RM.STATUS["ik_pre"], dtype=np.uint8)
        st[0, 0, :4] = RM.STATUS["feasible"]  # cell 0: the whole 4-sample family at yaw 0
        st[1, 0, 4] = RM.STATUS["feasible"]   # cell 1: the one-sample grasp at yaw 0
        self.meta = {"key": "k", "config": {"lattice_m": 0.08}, "arm": "right", "grasps": g, "edge_distances": [0.45], "height": {"value": 0.8},
                     "trunks": [{"translate": 0.5, "tilt": 0.0, "eligible": True, "posture": "untucked"}]}
        self.path = None
        self.arrays = {"s0.45_t0.50_k+0.0": st}

    status = lambda self, d, tr, ti: self.RM.ReachMap.status(self, d, tr, ti)  # noqa: E731
    cell_weights = lambda self, d, tr, ti: self.RM.ReachMap.cell_weights(self, d, tr, ti)  # noqa: E731


def test_cell_weights_normalise_family_discretisation():
    m = _Map()
    w = m.cell_weights(0.45, 0.5, 0.0)
    assert abs(w[0] - w[1]) < 1e-9 and w[2] == 0.0  # 4 samples of one family weigh what 1 sample of another does


def test_proposals_are_continuous_and_bounded():
    from ispatialgym.sampling import map_sampling as MS

    m = _Map()
    rng = np.random.default_rng(0)
    ps = MS.proposals(m, 0.45, 0.5, 0.0, 200, rng)
    assert {p.cell_index for p in ps} <= {0, 1}
    for p in ps:
        c = m.cells[p.cell_index]
        assert abs(p.cell_base[0] - c[0]) <= 0.04 + 1e-9 and abs(p.cell_base[1] - c[1]) <= 0.04 + 1e-9
        assert abs(p.yaw_base - m.yaws[p.yaw_index]) <= math.pi / 8 + 1e-9
    assert len({round(p.cell_base[0], 6) for p in ps}) > 50  # not the grid values
    w = MS.AdaptiveWidth()
    w.update(0, False)
    w.update(0, False)
    assert w.of(0) == 0.25


def test_band_view_maps_edge_distance_to_stand_off():
    from ispatialgym.sampling import map_sampling as MS

    m = _Map()
    rb = MS.as_bands(m, overhang=0.06)
    assert rb.meta["mode"] == "map_view" and rb.path is None and rb.meta["stand_offs"] == [0.39]
    cells = rb.band(0.5, 0.0, stand_off=0.39)
    assert cells.shape == (2, 2)
    pairs = rb.feasible(0.5, 0.0, [0.5, 0.0], stand_off=0.39)
    assert {n for _, n in pairs} == {"top_0", "top_0@+0.5", "top_0@+1.0", "top_0@+1.5"}


def test_grasp_set_keeps_the_tested_representative():
    from ispatialgym.sampling import reach_maps as RM

    T = S.pose_mat([0, 0, 0.1], S.mat_to_quat(np.diag([1.0, -1.0, -1.0])))
    fam = {"type": "rotational", "axis": [1, 0, 0], "center": [0, 0, 0], "interval": [-0.873, 1.396], "vertical_yaw_equivalent": False}
    lib = S.ModelLibrary(path="x", data={"families": {"f0": fam}, "grasps": [{"name": "top_0", "rest_id": "r0", "family_id": "f0", "width_m": 0.04,
                                                                              "T_object_eef": S.mat_pose(T)}]}, rests={"r0": _rest()})
    gs = RM.grasp_set(lib, "r0", RM.MapConfig(family_samples=4))
    names = [g["name"] for g in gs]
    assert "top_0" in names and "top_0~r180" in names and sum(1 for g in gs if not g["twin"]) == 5  # 0 + 4 samples
    assert [g for g in gs if g["name"] == "top_0"][0]["physics"] == "tested"


# ------------------------------------------------------------------------------------------------ the other arm of a map view (of-v2maps)


class _ViewBand:
    """A ReachBands stand-in: ``meta`` (mode / arm) and ``mirrored``."""

    def __init__(self, arm, mode="map_view", path=None):
        self.meta, self.path = {"mode": mode, "arm": arm, "map_path": path}, path

    @property
    def arm(self):
        return self.meta["arm"]

    def mirrored(self):
        if self.meta["mode"] == "map_view":
            raise ValueError("map views are not mirrored")
        return _ViewBand("left" if self.arm == "right" else "right", mode="tabletop", path=None)


def _arm_band_sampler():
    from types import SimpleNamespace

    from ispatialgym.sampling.v2 import tabletop as TT

    spec = TT.BandSpec(object="cat/model", grasp_key="cat__model", arm="right")
    return SimpleNamespace(band_spec=spec, top=SimpleNamespace(name="table_0", z=0.8126), world=SimpleNamespace(scene_model="scene", floor_z=0.0),
                           template_facts={})


def test_map_view_other_arm_takes_its_own_map(monkeypatch):
    from ispatialgym.sampling.v2 import tabletop as TT

    asked = []
    monkeypatch.setattr(TT, "map_bands", lambda s, spec, top, arm=None: asked.append(arm) or _ViewBand(arm, path=f"map_{arm}"))
    monkeypatch.setattr(TT, "band_facts", lambda b: {"arm": b.arm, "mode": b.meta["mode"]})
    s = _arm_band_sampler()
    own = TT.arm_band(s, "left", bands=_ViewBand("right", path="map_right"))
    assert asked == ["left"] and own.arm == "left" and own.meta["map_path"] == "map_left"
    assert s.template_facts["arm_bands"]["left"] == {"arm": "left", "mode": "map_view"}
    assert TT.arm_band(s, "left", bands=own) is own and asked == ["left"]  # an arm's own view is returned as is, no second lookup


def test_map_view_other_arm_without_a_map_keeps_the_legacy_path(monkeypatch):
    from ispatialgym.sampling import reach_bands as RB
    from ispatialgym.sampling.v2 import tabletop as TT

    monkeypatch.setattr(TT, "map_bands", lambda s, spec, top, arm=None: None)
    monkeypatch.setattr(TT, "band_facts", lambda b: {"arm": b.arm})
    loads = []

    def load(scene, support, obj, gkey, *, arm="right", **kw):
        loads.append(arm)
        if arm == "left":
            raise FileNotFoundError("no left band")
        return _ViewBand("right", mode="tabletop", path="legacy_right")

    monkeypatch.setattr(RB.ReachBands, "load", staticmethod(load))
    own = TT.arm_band(_arm_band_sampler(), "left", bands=_ViewBand("right", path="map_right"))
    assert loads == ["left", "right"] and own.arm == "left" and own.meta["mode"] == "tabletop"  # the right legacy band, mirrored

    loads.clear()
    monkeypatch.setattr(RB.ReachBands, "load", staticmethod(lambda *a, arm="right", **kw: loads.append(arm) or _ViewBand(arm, mode="tabletop", path=f"legacy_{arm}")))
    own = TT.arm_band(_arm_band_sampler(), "left", bands=_ViewBand("right", path="map_right"))
    assert loads == ["left"] and own.path == "legacy_left"  # the left arm's own legacy band when cached


def test_skip_existing_counts_only_the_runs_arm_and_kind(tmp_path):
    from ispatialgym.sampling import reach_maps as RM

    p = tmp_path / "map__cat__model__h0.8126__abc.json"
    p.write_text(json.dumps({"arm": "right", "kind": "tabletop"}))
    # i7-unblock 2026-11-07: a map of the former trunk eligibility rule (no config field) counts for a run of that rule only; a default
    # (per_d_e) run computes the map again
    assert RM._map_meta_matches(str(p), "right", "tabletop", trunk_eligibility=None)
    assert not RM._map_meta_matches(str(p), "right", "tabletop")
    q = tmp_path / "map__cat__model__h0.8126__def.json"
    q.write_text(json.dumps({"arm": "right", "kind": "tabletop", "config": {"trunk_eligibility": "per_d_e"}}))
    assert RM._map_meta_matches(str(q), "right", "tabletop") and not RM._map_meta_matches(str(q), "right", "tabletop", trunk_eligibility=None)
    assert not RM._map_meta_matches(str(q), "left", "tabletop")  # a left-arm run computes its own map
    assert not RM._map_meta_matches(str(q), "right", "brace")
    bad = tmp_path / "map__broken.json"
    bad.write_text("{")
    assert not RM._map_meta_matches(str(bad), "right", "tabletop")


def test_find_with_an_anchor_set_returns_the_map_at_the_anchor(tmp_path):
    # of-v2maps 2026-11-06 (v2smoke1): the anchor lookup's kind ("exact") overwrote find's map kind ("tabletop"), so with an anchor set every
    # map failed the kind test - "reach map miss" at an existing map's height for every alias
    from types import SimpleNamespace

    from ispatialgym.sampling import reach_maps as RM

    lib = SimpleNamespace(identity_key=lambda: "obj-key")
    base = tmp_path / "map__cat__model__h0.8126__k"
    meta = {"schema": RM.SCHEMA, "kind": "tabletop", "object_key": "obj-key", "rest_id": "r0", "arm": "left", "height": {"value": 0.8126}}
    (tmp_path / (base.name + ".json")).write_text(json.dumps(meta))
    np.savez_compressed(str(base) + ".npz", cells=np.zeros((1, 2)), yaws=np.zeros(8))
    A = HA.AnchorSet.from_surfaces([_surf("a", "board", 0.81, prec=1e-2, surface="interior"), _surf("a", "t1", 0.8126)])
    rm = RM.ReachMap.find(str(tmp_path), lib, 0.8126, rest_id="r0", arm="left", anchors=A)
    assert rm is not None and rm.path.endswith("h0.8126__k.json")
    assert RM.ReachMap.find(str(tmp_path), lib, 0.8131, rest_id="r0", arm="left", anchors=A) is None  # 0.8131 resolves to the 0.81 anchor: no map there
    assert RM.ReachMap.find(str(tmp_path), lib, 0.8126, rest_id="r0", arm="right", anchors=A) is None  # another arm's map is not this one
    assert RM.ReachMap.find(str(tmp_path), lib, 0.8126, rest_id="r0", arm="left", anchors=A, kind="brace") is None


def test_support_policy_scope_and_the_holding_arm(monkeypatch):
    # of-v2maps 2026-11-06: map_bands turns the policy on for the GRASP approach only (v2maps1: the policy, applied to every chain with an
    # ignore_for_last, rejected places / releases - touching_02 62 / 66 place_unreachable); "1" keeps the 2026-11-02 behaviour
    from types import SimpleNamespace

    from ispatialgym.sampling.v2 import ik as IK

    for v, on, scope in (("", False, "all"), ("0", False, "all"), ("1", True, "all"), ("grasp", True, "grasp")):
        monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY", v)
        assert IK.support_policy_on() is on and IK.support_policy_scope() == scope
    chk = IK.IKChecker.__new__(IK.IKChecker)
    chk.kit = SimpleNamespace(backend=SimpleNamespace(_attached={"left": {"obj": "book"}}))
    assert chk._holding("left") and not chk._holding("right")
    chk.kit = SimpleNamespace(backend=SimpleNamespace())  # a backend without attachments: never holding
    assert not chk._holding("left")


def test_policed_supports_and_the_measured_top_plane():
    # of-v2maps 2026-11-06 (v2scope1): a pushed box the chain approaches is not a support of the approach (touching_01 'finger_penetration
    # on target_box...'); the plane is the measured top map_bands registered, not the AABB top (inside_03 / nextto_03 / ontop_small_04
    # 'finger_penetration' on tables whose box rises above the working surface)
    from types import SimpleNamespace

    from ispatialgym.sampling.v2 import ik as IK

    table = SimpleNamespace(name="table_0", aabb=([0, 0, 0], [1, 1, 0.83]))
    box = SimpleNamespace(name="box_0", aabb=([0, 0, 0.81], [0.1, 0.1, 0.95]))
    assert IK.policed_supports([table], [box]) == [table]  # a grasp of the box: the table is policed
    assert IK.policed_supports([table, box], [box]) == [table]  # a push of the box: the box is not
    assert IK.policed_supports([box], [box]) == []  # a handle / contact chain on the object itself: nothing to police
    assert IK.policed_supports([table], None) == [table]
    assert IK.support_plane_z(table, {"table_0": 0.8126}) == 0.8126  # the measured top wins over the AABB's 0.83
    assert IK.support_plane_z(table, {}) == 0.83 and IK.support_plane_z(table, None) == 0.83
    assert IK.support_plane_z(SimpleNamespace(name="wall"), None) is None


def test_finger_link_clearance_per_hull():
    # of-v2maps 2026-11-06 (the support-policy diagnostic): each finger link's own minimum clearance to the plane
    from types import SimpleNamespace

    import torch as th

    from ispatialgym.grasping import contact_policy as CP
    from ispatialgym.sampling.v2 import ik as IK

    cube = np.array([[x, y, z] for x in (-0.005, 0.005) for y in (-0.005, 0.005) for z in (0.0, 0.01)], dtype=np.float64)
    l1 = SimpleNamespace(link="right_gripper_finger_link1", role="finger", pieces=[cube], points=cube, axis=np.array([0.0, 1.0, 0.0]))
    l2 = SimpleNamespace(link="right_gripper_finger_link2", role="finger", pieces=[cube - np.array([0.0, 0.0, 0.003])], points=cube, axis=np.array([0.0, -1.0, 0.0]))
    palm = SimpleNamespace(link="right_gripper_link", role="palm", pieces=[cube], points=cube, axis=None)
    hc = SimpleNamespace(geom=SimpleNamespace(links=[l1, l2, palm]), spacing=0.003)
    T = th.eye(4)
    T[2, 3] = 0.80  # the hull bottoms at z 0.800 and 0.797
    d = IK.finger_link_clearance(hc, T, CP.Support.plane(0.801), (0.02, 0.02))
    assert set(d) == {"right_gripper_finger_link1", "right_gripper_finger_link2"}
    assert abs(d["right_gripper_finger_link1"] - (-0.001)) < 1e-6 and abs(d["right_gripper_finger_link2"] - (-0.004)) < 1e-6


def test_policy_opening_is_the_oracle_preshape(monkeypatch):
    # of-v2maps 2026-11-06 (v2diag2): the support policy measures a library grasp at width / 2 + 10 mm (execute.execute_grasp's pre-shape)
    from types import SimpleNamespace

    from ispatialgym.sampling.v2 import ik as IKm
    from ispatialgym.sampling.v2 import tabletop as TT

    monkeypatch.delenv("ISPATIALGYM_SUPPORT_POLICY_OPENING", raising=False)
    assert abs(TT.policy_opening_of(SimpleNamespace(meta={"width": 0.01169})) - 0.015845) < 1e-6  # tablespoon top_6: preshape_q 0.01585
    assert TT.policy_opening_of(SimpleNamespace(meta={"width": 0.2})) == 0.05  # at most fully open
    assert TT.policy_opening_of(SimpleNamespace(meta={})) is None and TT.policy_opening_of(SimpleNamespace(meta=None)) is None
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY_OPENING", "q")
    assert TT.policy_opening_of(SimpleNamespace(meta={"width": 0.01169})) is None
    ikc = SimpleNamespace(_policy_opening=None)
    with IKm.IKChecker.policy_opening(ikc, 0.016):
        assert ikc._policy_opening == 0.016
        with IKm.IKChecker.policy_opening(ikc, None):
            assert ikc._policy_opening is None
        assert ikc._policy_opening == 0.016
    assert ikc._policy_opening is None


def test_record_policy_opening_resolves_the_recorded_grasp(monkeypatch):
    # of-v2maps 2026-11-07 (v2open1): the after-settle re-check of a GraspRecord polices it at the oracle's pre-shape too
    from types import SimpleNamespace

    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.v2 import ik as IKm
    from ispatialgym.sampling.v2 import tabletop as TT

    monkeypatch.delenv("ISPATIALGYM_SUPPORT_POLICY_OPENING", raising=False)
    monkeypatch.setattr(C, "grasp_entry", lambda key, name: SimpleNamespace(meta={"width": 0.01554}))  # carving knife top_3
    ikc = SimpleNamespace(_policy_opening=None)
    ikc.policy_opening = lambda m: IKm.IKChecker.policy_opening(ikc, m)
    s = SimpleNamespace(ikc=ikc)
    with TT.record_policy_opening(s, SimpleNamespace(grasp_key="k", grasp_name="top_3", extras={})):
        assert abs(ikc._policy_opening - 0.01777) < 1e-6
    with TT.record_policy_opening(s, SimpleNamespace(grasp_key="k", grasp_name="top_3", extras={"pinch_width_m": 0.02})):
        assert abs(ikc._policy_opening - 0.02) < 1e-6  # the sampler's own pad width wins, as in the oracle
    assert ikc._policy_opening is None
    with TT.record_policy_opening(SimpleNamespace(ikc=SimpleNamespace()), SimpleNamespace(grasp_key="k", grasp_name="g", extras={})):
        pass  # an IK checker without the switch (the tests' fakes): a no-op


def test_support_policy_fallback_switch_and_pad_overlap(monkeypatch):
    # of-v2maps 2026-11-06: the opt-in fallback's switch and the overlap of the fingers with the object along the approach
    import torch as th

    from ispatialgym.sampling.v2 import ik as IKm
    from ispatialgym.sampling.v2 import tabletop as TT

    for v, want in (("", ""), ("raise", "raise"), ("RAISE+relax", "raise+relax"), ("1", "")):
        monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY_FALLBACK", v)
        assert TT.support_policy_fallback_mode() == want
    a = th.tensor([0.0, 0.0, -1.0])  # approach straight down: the fingertips are the lowest points
    obj = th.tensor([[0.0, 0.0, 0.000], [0.0, 0.0, 0.008]])  # an 8 mm slab, top at z 0.008
    f1 = th.tensor([[0.01, 0.0, 0.002], [0.01, 0.0, 0.05]])  # finger 1 reaches 6 mm down the slab
    f2 = th.tensor([[-0.01, 0.0, 0.006], [-0.01, 0.0, 0.05]])  # finger 2 only 2 mm
    assert abs(IKm.pad_overlap_m({"l1": f1, "l2": f2}, obj, a) - 0.002) < 1e-6
    assert IKm.pad_overlap_m({"l1": f1 + th.tensor([0.0, 0.0, 0.01])}, obj, a) < 0  # raised past the slab: no overlap


def test_frame_diag_projects_the_eef_box_and_flags_the_border(monkeypatch, tmp_path):
    # of-v2maps 2026-11-06 (the plate's gripper_not_in_frame): the 8 box corners eef_in_frame tests, in / out of the 5 % border
    from types import SimpleNamespace

    import torch as th

    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling.v2 import ik as IKm
    from ispatialgym.sampling.v2 import tabletop as TT

    monkeypatch.setenv("ISPATIALGYM_FRAME_DIAG", str(tmp_path))
    cm = G.CameraModel(640, 480, 400.0, 400.0, 320.0, 240.0)
    cam = (th.tensor([0.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))  # looking along -z (camera convention of CameraModel.project)
    def fk(names, q):
        raise RuntimeError("no fk")
    s = SimpleNamespace(world=SimpleNamespace(camera_model=cm, cam=None), ikc=SimpleNamespace(exact=lambda *a, **k: IKm.IKResult(False, None, "x", 1),
                        q_locked=lambda st: th.zeros(28)), kit=SimpleNamespace(backend=SimpleNamespace(fk=fk)), gp=SimpleNamespace(variant="v"))
    st = SimpleNamespace(base_pose2d=(0.0, 0.0, 0.0), translate=0.0, tilt=0.0)
    centre = (th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    rec = TT.frame_diag(s, st, "right", centre, "place", cam_pose=cam)
    assert rec["box_in_margin"] == 8 and rec["max_past_border_px"] == 0.0 and not rec["ik_ok"] and len(rec["points"]) == 8
    side = (th.tensor([1.0, 0.0, -1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))  # 1 m to the side at 1 m depth: u ~ 720 px > 640
    rec = TT.frame_diag(s, st, "right", side, "place", cam_pose=cam)
    assert rec["box_in_image"] == 0 and rec["max_past_border_px"] > 100
