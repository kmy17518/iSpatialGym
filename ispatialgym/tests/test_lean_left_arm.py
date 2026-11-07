"""i7-overhang 2026-11-07 (owner decision: a LEFT-arm validator for lean rests - hardback__bqvvvx__bu at 16.7 deg on the left board,
inside_large_01's goal side): the validating arm in the lean job, its entry id, the library and its identity, the mirrored validation offsets,
the pipeline's scope (lean only), the left-arm entry served by its lean key; and the inside mode's nearer-place cap. Simulator-free."""

import json
import os
from types import SimpleNamespace

import pytest

from ispatialgym.grasping import lean as LN
from ispatialgym.grasping import schema as S
from ispatialgym.grasping import validate_tabletop as VT
from ispatialgym.sampling import grasp_library as GL

BU = "hardback__bqvvvx__bu"
SPEC = LN.LeanSpec(16.7, "left")


def _base():
    return {"category": "hardback", "model": "bqvvvx", "scale": [1.0, 1.0, 1.0], "dataset": "behavior-1k-assets"}


def test_the_validating_arm_is_in_the_job_the_entry_id_and_the_identity(tmp_path):
    r = LN.make_job(_base(), SPEC, [0, 0, 0, 1.0], base_key=BU)
    l = LN.make_job(_base(), SPEC, [0, 0, 0, 1.0], base_key=BU, validate_arm="left")
    assert "validate_arm" not in r and r["id"] == f"hardback__bqvvvx__s1__{SPEC.tag()}"  # the right-arm entry as before
    assert l["validate_arm"] == "left" and l["id"] == f"hardback__bqvvvx__s1__{SPEC.tag()}__valleft" and LN.job_arm(l) == "left" and LN.job_arm(r) == "right"
    assert S.job_dir(l, str(tmp_path)).endswith(l["id"]) and S.job_dir(r, str(tmp_path)).endswith(r["id"])
    assert S.job_validate_arm(l) == "left" and S.job_validate_arm(r) is None and S.job_validate_arm({**_base(), "validate_arm": "left"}) is None  # lean only
    with pytest.raises(ValueError):
        LN.make_job(_base(), SPEC, [0, 0, 0, 1.0], validate_arm="up")
    # the library records the arm (build_library) and the identity carries it; a right-arm identity as before
    rest = {"schema": S.REST, "rest_id": SPEC.tag(), "quat_xyzw": [0, 0, 0, 1.0], "h_root": 0.115, "stable": True, "rest_kind": "lean", "lean": SPEC.to_json()}
    lib_l = VT.build_library(l, {"candidates": [], "arm": "left"}, {SPEC.tag(): rest}, [], VT.CP.DEFAULT, {}, validate_arm="left")
    lib_r = VT.build_library(r, {"candidates": [], "arm": "right"}, {SPEC.tag(): rest}, [], VT.CP.DEFAULT, {})
    assert lib_l["validate_arm"] == "left" and lib_l["arm"] == "left" and "validate_arm" not in lib_r
    ml = S.ModelLibrary("l", lib_l, {SPEC.tag(): S.Rest.from_record(rest)})
    mr = S.ModelLibrary("r", lib_r, {SPEC.tag(): S.Rest.from_record(rest)})
    assert ml.identity()["validate_arm"] == "left" and "validate_arm" not in mr.identity() and ml.identity_key() != mr.identity_key()


def test_the_left_arm_world_mirrors_the_object_and_refuses_other_jobs():
    offs = VT.ValidateConfig().base_offsets
    m = VT.mirrored_offsets(offs)
    assert all(a == x and b == -y for (a, b), (x, y) in zip(m, offs)) and all(y < 0 for _, y in offs)
    l = LN.make_job(_base(), SPEC, [0, 0, 0, 1.0], base_key=BU, validate_arm="left")
    VT.check_job_arm(l, "left")
    with pytest.raises(ValueError, match="left arm"):
        VT.check_job_arm(l, "right")
    with pytest.raises(ValueError, match="lean jobs only"):
        VT.check_job_arm(_base(), "left")  # a stable job never in the left-arm world
    from ispatialgym.grasping import pipeline as PL

    with pytest.raises(SystemExit, match="lean stage only"):
        PL.main(["validate", "--jobs", "/nonexistent.json", "--arm", "left"])


def test_the_lean_key_serves_the_left_arm_entry(tmp_path, monkeypatch):
    root = str(tmp_path / "glib")
    l = LN.make_job(_base(), SPEC, [0, 0, 0, 1.0], base_key=BU, validate_arm="left")
    rest = {"schema": S.REST, "rest_id": SPEC.tag(), "quat_xyzw": [0, 0, 0, 1.0], "h_root": 0.115, "stable": True, "rest_kind": "lean", "lean": SPEC.to_json()}
    lib = VT.build_library(l, {"candidates": [], "arm": "left"}, {SPEC.tag(): rest}, [], VT.CP.DEFAULT, {}, validate_arm="left")
    lib["grasps"] = [{"name": "side_0", "rest_id": SPEC.tag(), "family_id": None, "T_object_eef": {"position": [0, -0.07, 0.06], "orientation_xyzw": [0, 0, 0, 1.0]},
                      "width_m": 0.03, "preshape_q": 0.025, "finger_q": [0.015, 0.015]}]
    d = S.job_dir(l, root)
    os.makedirs(d)
    with open(os.path.join(d, "library.json"), "w") as f:
        json.dump(lib, f)
    monkeypatch.setenv("ISPATIALGYM_GRASP_LIBRARY_V2", root)
    for f in (GL._v2_index, GL._v2_lean_index, GL._v2_overhang_index):
        f.cache_clear()
    k = LN.lean_key(BU, SPEC)
    e = GL.v2_entry(k)
    assert e is not None and e.path.endswith("__valleft/library.json") and e.data["validate_arm"] == "left" and [g.name for g in GL.load_grasps(k)] == ["side_0"]
    assert GL.v2_entry(BU) is None  # never the stable key
    for f in (GL._v2_index, GL._v2_lean_index, GL._v2_overhang_index):
        f.cache_clear()


def test_inside_stance_c_cap_drops_the_far_stand_offs():
    """``generation.inside_stand_off_max_m`` (owner: keep the place positions nearer for inside_large_01): the stand-offs beyond it are not
    laid out, and ``placement_stats["stand_off_capped"]`` counts the lattice candidates removed."""
    import torch as th

    from ispatialgym.sampling.v2.stances import EdgeFrame
    from ispatialgym.sampling.v2.tasks import iors_large_inside as ILI

    seen = []

    class S_:
        _placement_candidates_at = ILI.InsideMode._placement_candidates_at

        def __init__(self, cap):
            gen = {"inside_stand_offs_m": [0.55, 0.65, 0.75, 0.85], "inside_stand_off_max_m": cap}
            self.gp = SimpleNamespace(variant="hb", gen=lambda k, d=None: gen.get(k, d))
            self.partner = SimpleNamespace(name="bookcase_jaysra_0")
            self.place = {"half": 0.1, "height": 0.2, "cells": th.zeros(0, 2), "z": 0.9472}
            self.template_facts = {}
            self.seg_arm = "right"

        def inside_band(self, arm):
            return None

    import ispatialgym.sampling.v2.prompt0 as P0

    orig = P0.base_ok
    P0.base_ok = lambda s, xy: seen.append(round(float(xy[0]), 2)) or False
    try:
        edge = EdgeFrame("+u", th.tensor([0.0, 0.0]), th.tensor([1.0, 0.0]), 0.4)
        for cap, want in ((None, 0), (0.70, 2)):
            s = S_(cap)
            seen.clear()
            assert s._placement_candidates_at(edge, th.zeros(2)) == []
            st = s.template_facts["placement_stats"]
            n_tr = len([1 for _ in ILI.RB.TRANSLATES for _ in ILI.RB.TILTS])
            assert st.get("stand_off_capped", 0) == want * n_tr  # one lateral per stand-off on the 0.4 m edge
            assert len(seen) == 4 - want
    finally:
        P0.base_ok = orig
