"""i7-artic 2026-11-07 (owner decision: handles stay with the ARTICULATION library): articulation prefix maps for articulation_open_small_01 /
02 and articulation_close_small_01 / 02 - the prefix kind of ``reach_maps`` (compute on a fake world, key, save, ``find_prefix``'s identity
checks, the handle-only band view ``prefix_bands``) and the consumer (``articulation_small._brace_band`` under v2 only:
``sampling.v2.artic_prefix``; a miss rejects missing_map, a hit is the handle-only view, off v2 only the legacy path). The grasp maps' keys
are unchanged. Simulator-free."""

import itertools
import json
import logging
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import artic_motions as AM
from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import schema as S
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling import reach_maps as RM
from ispatialgym.sampling.sim_utils import Reject

HANDLES = os.path.join(os.path.dirname(__file__), "..", "..", "runlogs", "grasp_library_20260905", "artic", "handles")
ALIASES = {"open_01": "configuration_matching-articulation_open_small_scale-blender-breakfast_table",
           "open_02": "configuration_matching-articulation_open_small_scale-carton-desk",
           "close_01": "configuration_matching-articulation_close_small_scale-blender-breakfast_table",
           "close_02": "configuration_matching-articulation_close_small_scale-carton-breakfast_table"}


# ------------------------------------------------------------------------------------------------------------------
# the grasp maps' keys did not move
# ------------------------------------------------------------------------------------------------------------------


def _tmeta(**kw):
    m = {"schema": RM.SCHEMA, "kind": "tabletop", "arm": "right", "object_key": "obj-key", "rest_id": "r0", "edge_distances": [0.35, 0.45, 0.55],
         # integration 2026-11-08: the configuration of 7fb655f6's maps - the former trunk rule (trunk_eligibility None, left out of the key);
         # per_d_e became the default of NEW builds (i7-unblock 58cd9939), so MapConfig() now keys a per_d_e map
         "config": RM.MapConfig(trunk_eligibility=None).key(), "policy_key": "pk", "robot": "rbt", "brace": None,
         "height": {"value": 0.8126, "anchor_key": "0.8126"}}
    m.update(kw)
    return m


def test_existing_map_keys_unchanged():
    # the values of 7fb655f6 (before the prefix kind): tabletop, floor, brace and generic maps keep their keys
    fl = _tmeta(kind="floor", plane="floor", height={"value": 0.0, "anchor_key": "floor"}, config=RM.floor_config().key(), edge_distances=[0.0])
    g = {"schema": RM.SCHEMA, "kind": "generic", "arm": "left", "height": {"value": 0.7853}, "pose_set": RM.generic_pose_set(),
         "config": RM.MapConfig(twins=False, trunk_eligibility=None).key(), "policy_key": "pk", "robot": "rbt"}
    assert RM.map_key(_tmeta()) == "f66a7ad86c8570a7"
    assert RM.map_key(fl) == "534645fb3aaad974"
    assert RM.map_key(_tmeta(kind="brace", brace={"x": 1})) == "7911e306d28b0358"
    assert RM.generic_map_key(g) == "59776590ec7742c5"


# ------------------------------------------------------------------------------------------------------------------
# the alias motions the consumer tests read (artic_motions.build on the annotations)
# ------------------------------------------------------------------------------------------------------------------


def _hull(doc):
    lo = [min(doc["base_link"]["bbox_min"][i], *[j["link_bbox_min"][i] for j in doc["joints"]]) for i in range(3)]
    hi = [max(doc["base_link"]["bbox_max"][i], *[j["link_bbox_max"][i] for j in doc["joints"]]) for i in range(3)]
    return th.tensor(list(itertools.product(*zip(lo, hi))), dtype=th.float32)


def _motion(tag):
    alias = ALIASES[tag]
    entry = AM.load_alias(alias)
    cat, model, joint, hkey = AM.target_model(entry)
    with open(os.path.join(HANDLES, f"{hkey}.json")) as f:
        doc = json.load(f)
    j = next(x for x in doc["joints"] if x["joint"] == joint)
    return AM.build(alias, entry, doc, _hull(doc), (j["lower"], j["upper"]))


# ------------------------------------------------------------------------------------------------------------------
# reach_maps: the prefix kind on a fake world
# ------------------------------------------------------------------------------------------------------------------


class _FakeWorld:
    """test_generic_maps' fake SynthWorld: the IK "solution" of a pose is the pose itself (flattened 4x4), spheres far above the slab."""

    def __init__(self, cfg):
        self.cfg, self.dev = cfg, th.device("cpu")
        self.backend = SimpleNamespace(check_collision=lambda q, emb_name, margin: th.tensor([bool(q[0, 1] < -0.5)]),
                                       robot_spheres_world=lambda q: (th.tensor([[[0.0, 0.0, 2.0, 0.05]]]), ["torso_link4"]))

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
    ok = (T[:, 0, 3] < 0.95)[:, None].repeat(1, S_)  # beyond 0.95 m ahead "has no IK"
    return ok, T.reshape(n, 1, 16).repeat(1, S_, 1)


REST = {"schema": S.REST, "rest_id": "r0", "quat_xyzw": [0.0, 0.0, 0.0, 1.0], "h_root": 0.1, "intended_quat_xyzw": [0.0, 0.0, 0.0, 1.0], "stable": True}


def _wp(q, x, z=0.3):
    # an eef pointing down (approach -z: the pre-grasp 10 cm ABOVE), at (x, 0, z) in the object frame
    return {"q": q, "eef_in_object": {"position": [x, 0.0, z], "orientation_xyzw": [1.0, 0.0, 0.0, 0.0]}}


def _alib():
    a = [_wp(0.0, 0.0), _wp(0.1, 0.02), _wp(0.2, 0.04), _wp(0.3, 0.30)]  # the last waypoint jumps 26 cm: a "branch jump" at step_max 0.2
    b = [_wp(1.0, 0.0), _wp(0.9, -0.01)]
    seg = lambda wps, i: {"id": f"s{i}", "variant": f"v{i}", "rest_id": "r0", "waypoints": wps, "fingers": {"pre": 0.0, "path": 0.0},  # noqa: E731
                          "validated_travel_rad": abs(wps[-1]["q"] - wps[0]["q"]), "arm_validated": "right"}
    return {"schema": RM.ARTIC_LIBRARY_V2, "category": "blender", "model": "eyedvd", "scale": [1, 1, 1], "alias": "al", "motion": "pull_open", "joint": "j_lid",
            "q_start": 0.0, "handle_sig": "sig0", "face_yaw_offset": math.pi, "arms_offered": ["right", "left"], "rests": {"r0": REST},
            "segments": [seg(a, 0), seg(b, 1)], "profile": {"brake_candidate": "brake:0.5"}}


def test_compute_prefix_map_on_a_fake_world(tmp_path, monkeypatch):
    monkeypatch.setattr(RM, "_solve", _fake_solve)
    monkeypatch.setattr(RB, "trunk_grid", lambda h: [(0.25, -0.3), (0.5, -0.8)])
    cfg = RM.MapConfig(seeds=2, twins=False, lattice_m=0.16)
    meta, arrays = RM.compute_prefix_map(_FakeWorld(cfg), _alib(), 0.8126, {"key": "0.8126"}, cfg=cfg, step_max_rad=0.2)
    assert meta["kind"] == RM.PREFIX_KIND and meta["key"] == RM.prefix_map_key(meta) and meta["yaws"] == [round(_w(math.pi), 4)]
    assert [t["eligible"] for t in meta["trunks"]] == [True, False] and meta["validated_travel_rad"] == pytest.approx(0.3)
    cells = arrays["cells"]
    ci = lambda x, y: int(np.argmin(np.abs(cells - np.array([x, y])).sum(axis=1)))  # noqa: E731
    a = arrays["s0.45_t0.25_k-0.3__seg0"][:, 0]
    b = arrays["s0.45_t0.25_k-0.3__seg1"][:, 0]
    assert (arrays["s0.45_t0.50_k-0.8__seg0"] == -1).all()  # the ineligible trunk
    # the object yawed pi: the waypoints run toward the robot (-x); the 26 cm jump exceeds step_max -> the contiguous prefix ends at 0.2
    assert a[ci(0.67, 0.0)] == pytest.approx(0.2) and b[ci(0.67, 0.0)] == pytest.approx(0.1)
    assert a[ci(1.15, 0.0)] == -1  # beyond the fake IK's reach: not even the first contact
    meta2, arrays2 = RM.compute_prefix_map(_FakeWorld(cfg), _alib(), 0.8126, {"key": "0.8126"}, cfg=cfg, step_max_rad=0.5)
    assert arrays2["s0.45_t0.25_k-0.3__seg0"][ci(0.83, 0.0), 0] == pytest.approx(0.3) and meta2["key"] != meta["key"]
    with pytest.raises(ValueError):
        RM.compute_prefix_map(_FakeWorld(cfg), dict(_alib(), schema="isg-artic-library/1"), 0.8126, {"key": "0.8126"}, cfg=cfg)
    # save / read back / never a grasp map
    p = RM.save_prefix(str(tmp_path), meta, arrays)
    pm = RM.PrefixMap(p)
    assert os.path.basename(p).startswith("prefix__blender__eyedvd__pull_open__right__h0.8126__") and pm.accepted(0.45, 0.25, -0.3, 0.15)[ci(0.67, 0.0)]
    assert not pm.accepted(0.45, 0.25, -0.3, 0.25)[ci(0.67, 0.0)]
    assert RM.ReachMap.find(str(tmp_path), SimpleNamespace(identity_key=lambda: meta["object_key"]), 0.8126, rest_id="r0") is None


def _w(a):
    return math.atan2(math.sin(a), math.cos(a))


# ------------------------------------------------------------------------------------------------------------------
# find_prefix + the handle-only band view
# ------------------------------------------------------------------------------------------------------------------


CELLS = np.array([[0.51, 0.0], [0.59, 0.0], [0.51, 0.08]], dtype=np.float32)
TRUNK = (0.25, -1.0)
TKEY = "t0.25_k-1.0"


def _pmeta(**kw):
    meta = {"schema": RM.SCHEMA, "kind": RM.PREFIX_KIND, "prefix_version": RM.PREFIX_VERSION, "arm": "right",
            "object": {"category": "carton", "model": "cdmmwy", "scale": [1.0, 1.0, 1.0], "rest": dict(REST)}, "object_key": "ok", "rest_id": "r0",
            "joint": "j_right_lid", "motion": "push_close", "q_start": 1.6493, "handle_sig": "sig", "face_yaw_offset": 0.0, "yaws": [0.0],
            "segments": [{"index": 0, "variant": "tip#0+"}, {"index": 1, "variant": "tip#0-"}], "segment_ids": ["a", "b"], "validated_travel_rad": 0.4,
            "min_open_rad": 0.0, "step_max_rad": 0.5, "arm_validated": ["right"], "height": {"value": 0.8126, "anchor_key": "0.8126"}, "edge_distances": [0.35, 0.45, 0.55],
            "config": RM.MapConfig(twins=False).key(), "policy_key": CP.DEFAULT.key("right"), "robot": RM.robot_digest(),
            "trunks": [{"translate": TRUNK[0], "tilt": TRUNK[1], "key": TKEY, "eligible": True, "posture": "untucked"},
                       {"translate": 0.5, "tilt": 0.0, "key": "t0.50_k+0.0", "eligible": False}]}
    for k, v in kw.items():
        if k == "object":
            meta["object"] = dict(meta["object"], **v)
        else:
            meta[k] = v
    meta["key"] = RM.prefix_map_key(meta)
    return meta


def _write_prefix(root, meta, prefix=None):
    arrays = {"cells": CELLS, "yaws": np.zeros(1, dtype=np.float32)}
    for d in meta["edge_distances"]:
        for t in meta["trunks"]:
            for si in range(len(meta["segments"])):
                arrays[f"s{d:.2f}_{t['key']}__seg{si}"] = np.full((3, 1), -1.0, dtype=np.float32)
    # at 0.45 / the eligible trunk: cell 0 reaches 0.3 rad on segment 0, cell 1 only the first contact (0) on segment 1, cell 2 nothing
    arrays[f"s0.45_{TKEY}__seg0"][0, 0] = 0.3
    arrays[f"s0.45_{TKEY}__seg1"][1, 0] = 0.0
    arrays[f"s0.35_{TKEY}__seg1"][2, 0] = 0.05
    if prefix:
        arrays.update(prefix)
    return RM.save_prefix(root, meta, arrays)


def _spec(**kw):
    sp = {"category": "carton", "model": "cdmmwy", "scale": [1.0, 1.0, 1.0], "rest_quat": [0.0, 0.0, 0.0, 1.0], "joint": "j_right_lid", "motion": "push_close",
          "q_start": 1.6493, "needed_travel": 0.0, "arm": "right", "face_yaw_offset": 0.0, "handle_sig": "sig", "height": 0.8126}
    sp.update(kw)
    return sp


def _anchors(path, keys=("0.7853", "0.8126")):
    from ispatialgym.sampling.height_anchors import AnchorSet

    an = {"max_approx_m": 0.01, "anchors": [{"key": k, "height": float(k), "precision": 1e-4, "surfaces": [f"s/{k}"]} for k in keys]}
    path.write_text(json.dumps(an))
    return AnchorSet(an["anchors"], 0.01)


def test_find_prefix_checks_its_spec(tmp_path):
    root = str(tmp_path / RM.PREFIX_DIR)
    p = _write_prefix(root, _pmeta())
    ok = dict(policy_key=CP.DEFAULT.key("right"), robot=RM.robot_digest(), config_require={"preshape": "grasp"})
    pm = RM.find_prefix(root, _spec(), **ok)
    assert pm is not None and pm.path == p and pm.lookup["how"] == "height"
    anchors = _anchors(tmp_path / "anchors.json")
    assert RM.find_prefix(root, _spec(height=0.8156), anchors=anchors, **ok).lookup["how"] == "approx"  # within the anchor's 1 cm
    assert RM.find_prefix(root, _spec(height=0.8156), **ok) is None  # no anchors: the height itself
    assert RM.find_prefix(root, _spec(q_start=1.6493 + 0.02), **ok) is not None  # within the 0.03 rad start tolerance
    assert RM.find_prefix(root, _spec(face_yaw_offset=2 * math.pi + 0.03), **ok) is not None  # wrapped
    for bad in ({"arm": "left"}, {"category": "blender"}, {"model": "x"}, {"scale": [1.0, 1.0, 2.0]}, {"rest_quat": [0.0, 0.0, 0.7071, 0.7071]},
                {"joint": "j_left_lid"}, {"motion": "pull_open"}, {"q_start": 1.70}, {"handle_sig": "other"}, {"face_yaw_offset": 0.2},
                {"needed_travel": 0.5}, {"height": 0.7853}, {"step_max_rad": 0.4}, {"min_open_rad": 0.2}):
        assert RM.find_prefix(root, _spec(**bad), **ok) is None, bad
    for bad in ({"policy_key": "other"}, {"robot": "other"}, {"config_require": {"preshape": "max"}}):
        assert RM.find_prefix(root, _spec(), **{**ok, **bad}) is None, bad
    # an unstable rest, a rest of another intended placement, a hand-edited identity, another prefix version: never served
    for i, kw in enumerate(({"object": {"rest": dict(REST, stable=False)}}, {"object": {"rest": dict(REST, intended_quat_xyzw=[0.0, 0.0, 1.0, 0.0])}},
                            {"prefix_version": "isg-artic-prefix/1"})):
        d = str(tmp_path / f"bad{i}")
        _write_prefix(d, _pmeta(**kw))
        assert RM.find_prefix(d, _spec(), **ok) is None, kw
    d = str(tmp_path / "edited")
    q = _write_prefix(d, _pmeta())
    m = json.load(open(q))
    m["handle_sig"] = "edited"
    json.dump(m, open(q, "w"))
    assert RM.find_prefix(d, _spec(handle_sig="edited"), **ok) is None
    # a grasp map in the directory is never a prefix map
    assert RM.find_prefix(str(tmp_path / "nothing"), _spec(), **ok) is None


def test_prefix_bands_is_the_handle_only_view(tmp_path):
    pm = RM.PrefixMap(_write_prefix(str(tmp_path), _pmeta()))
    v = RM.prefix_bands(pm, 0.0, yaw_index=0)
    assert v.meta["mode"] == "map_view" and v.meta["map_kind"] == RM.PREFIX_KIND and v.meta["handle_only"] and v.meta["stand_offs"] == [0.35, 0.45, 0.55]
    assert v.trunks() == [TRUNK] and v.meta["grasp_names"] == ["tip#0+", "tip#0-"]
    assert np.allclose(v.band(*TRUNK, stand_off=0.45), CELLS[[0, 1]]) and np.allclose(v.band(*TRUNK, stand_off=0.35), CELLS[[2]])
    assert len(v.band(*TRUNK, stand_off=0.55)) == 0
    v2 = RM.prefix_bands(pm, 0.25)  # a pull's band fraction: only the cell whose segment reaches 0.25 rad
    assert np.allclose(v2.band(*TRUNK, stand_off=0.45), CELLS[[0]]) and v2.meta["needed_travel_rad"] == 0.25
    assert v2.pairs_so[0.45][TRUNK].tolist() == [[0, 0, 0]]


# ------------------------------------------------------------------------------------------------------------------
# the consumer: articulation_small._brace_band under v2 only
# ------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def v2only(tmp_path, monkeypatch):
    maps = tmp_path / "maps"
    monkeypatch.setenv("ISPATIALGYM_V2_ONLY", "1")
    monkeypatch.setenv("ISPATIALGYM_REACH_MAPS", str(maps))
    for k in ("ISPATIALGYM_ARTIC_PREFIX_MAPS", "ISPATIALGYM_REACH_MAP_ANCHORS", "ISPATIALGYM_REACH_MAP_REQUIRE"):
        monkeypatch.delenv(k, raising=False)
    return maps


def _sampler(motion):
    """The attributes ``artic_prefix.prefix_spec`` reads, from a built alias motion (what the live sampler holds at _brace_band)."""
    v2 = motion["motion"] == "v2_ajar_pull"
    bases = motion["base_handles"]
    tgt = SimpleNamespace(scale=th.ones(3))
    s = SimpleNamespace(role="target", choice=SimpleNamespace(models={"target": SimpleNamespace(category=motion["category"], model=motion["model"])}),
                        obj=lambda role: tgt, joint_name=motion["joint"], handle_arm=motion["handle_arm"], facing_local=th.tensor(motion["facing_local"]),
                        jinfo=motion["jinfo"], handle=bases[0], handle_variants=bases if v2 else None, v2=v2, close_push=motion["motion"] == "push_close",
                        direction=motion["direction"], top=SimpleNamespace(z=0.8126, name="breakfast_table_rhjoby_0"),
                        world=SimpleNamespace(floor_z=0.0, scene_model="house_single_floor"), gp=SimpleNamespace(variant=motion["alias"]),
                        template_facts={}, handle_only_band=False)
    s._q_main = lambda: float(motion["q_start"])
    s._band_q_values = lambda: [float(motion["q_start"])]
    return s


def _requests(caplog):
    return [json.loads(r.getMessage().split(" ", 1)[1]) for r in caplog.records if r.getMessage().startswith("MAP_REQUEST")]


def test_consumer_miss_rejects_missing_map_and_a_hit_is_the_handle_only_view(v2only, caplog):
    from ispatialgym.sampling.v2 import artic_prefix as APX
    from ispatialgym.sampling.v2.tasks.articulation_small import ArticulationSmall as AS

    m = _motion("close_02")
    s = _sampler(m)
    spec = APX.prefix_spec(s)
    assert spec["handle_sig"] == m["handle_sig"] and spec["face_yaw_offset"] == pytest.approx(m["face_yaw_offset"])  # the sampler's own handles
    assert (spec["motion"], spec["needed_travel"], spec["height"]) == ("push_close", 0.0, pytest.approx(0.8126))
    edge = SimpleNamespace(name="e0")
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        with pytest.raises(Reject) as e:
            AS._brace_band(s, edge, None)  # read BEFORE the in-process BandComputer (which v2 only blocks): a miss is missing_map
    assert e.value.reason == "missing_map" and not s.handle_only_band
    req = _requests(caplog)
    assert req and req[0]["kind"] == RM.PREFIX_KIND and req[0]["motion"] == "push_close" and req[0]["handle_sig"] == m["handle_sig"]
    # the map of this motion in <ISPATIALGYM_REACH_MAPS>/_articulation serves; the view is handle-only
    _write_prefix(str(v2only / RM.PREFIX_DIR), _pmeta(handle_sig=m["handle_sig"], q_start=round(m["q_start"], 4), face_yaw_offset=round(m["face_yaw_offset"], 4)))
    v = AS._brace_band(s, edge, None)
    assert s.handle_only_band and v.meta["map_kind"] == RM.PREFIX_KIND and np.allclose(v.band(*TRUNK, stand_off=0.45), CELLS[[0, 1]])
    assert s.template_facts["artic_prefix_map"]["handle_sig"] == m["handle_sig"]
    # a pull whose band needs more travel than the map validated: a miss
    s2 = _sampler(m)
    s2._band_q_values = lambda: [float(m["q_start"]), float(m["q_start"]) - 0.6]
    with pytest.raises(Reject) as e2:
        AS._brace_band(s2, edge, None)
    assert e2.value.reason == "missing_map"


def test_consumer_off_v2_only_keeps_the_legacy_path(monkeypatch):
    from ispatialgym.sampling.v2.tasks.articulation_small import ArticulationSmall as AS

    monkeypatch.delenv("ISPATIALGYM_V2_ONLY", raising=False)
    s = _sampler(_motion("open_01"))
    with pytest.raises(AttributeError):  # the legacy body runs (this stand-in has no world / kit for it): the prefix map is not read
        AS._brace_band(s, SimpleNamespace(name="e0"), None)
    assert not s.handle_only_band and "artic_prefix_map" not in s.template_facts


# ------------------------------------------------------------------------------------------------------------------
# owner 2026-11-07: an arm's map only from segments validated with that arm
# ------------------------------------------------------------------------------------------------------------------


def test_a_left_map_from_right_arm_segments_is_impossible(tmp_path, monkeypatch):
    monkeypatch.setattr(RM, "_solve", _fake_solve)
    monkeypatch.setattr(RB, "trunk_grid", lambda h: [(0.25, -0.3)])
    left = RM.MapConfig(arm="left", seeds=2, twins=False, lattice_m=0.16)
    with pytest.raises(ValueError, match="no left-arm validated segment"):
        RM.compute_prefix_map(_FakeWorld(left), _alib(), 0.8126, {"key": "0.8126"}, cfg=left)  # both segments right-arm validated
    mixed = _alib()
    mixed["segments"][1] = dict(mixed["segments"][1], arm_validated="left", id="s1L")
    meta, _ = RM.compute_prefix_map(_FakeWorld(left), mixed, 0.8126, {"key": "0.8126"}, cfg=left)
    assert meta["arm"] == "left" and meta["arm_validated"] == ["left"] and meta["segment_ids"] == ["s1L"] and [s["variant"] for s in meta["segments"]] == ["v1"]
    right = RM.MapConfig(seeds=2, twins=False, lattice_m=0.16)
    meta_r, _ = RM.compute_prefix_map(_FakeWorld(right), mixed, 0.8126, {"key": "0.8126"}, cfg=right)
    assert meta_r["arm_validated"] == ["right"] and meta_r["segment_ids"] == ["s0"]
    # a map whose arm is not the arm its segments were validated with never serves, even with a self-consistent key
    ok = dict(policy_key=CP.DEFAULT.key("left"), robot=RM.robot_digest())
    d = str(tmp_path / "left_from_right")
    _write_prefix(d, _pmeta(arm="left", policy_key=CP.DEFAULT.key("left"), arm_validated=["right"]))
    assert RM.find_prefix(d, _spec(arm="left"), **ok) is None
    d2 = str(tmp_path / "left")
    _write_prefix(d2, _pmeta(arm="left", policy_key=CP.DEFAULT.key("left"), arm_validated=["left"]))
    assert RM.find_prefix(d2, _spec(arm="left"), **ok) is not None
    d3 = str(tmp_path / "old")
    _write_prefix(d3, _pmeta(arm_validated=None))
    assert RM.find_prefix(d3, _spec(), policy_key=CP.DEFAULT.key("right"), robot=RM.robot_digest()) is None
