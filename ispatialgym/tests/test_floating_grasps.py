"""i7-floating 2026-11-07: floating grasps in the v2 library (``grasping.floating`` / ``validate_floating``) - the GraspGen re-targeting,
the floating gates on a synthetic bar, ``floating`` in the entry identity, the lookup of a floating entry (no rest, every grasp, never a
placement), the held sampler's guard (a tabletop entry never serves a floating key; scale enforced under v2 only), the validator's
held-motion legs and closure constants, and the pipeline dispatch. Simulator-free (the validator's sim code is not run)."""

import json
import math
import os

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from ispatialgym.grasping import candidates as CA  # noqa: E402
from ispatialgym.grasping import contact_policy as CP  # noqa: E402
from ispatialgym.grasping import floating as FL  # noqa: E402
from ispatialgym.grasping import manifest as MF  # noqa: E402
from ispatialgym.grasping import pipeline as PL  # noqa: E402
from ispatialgym.grasping import schema as S  # noqa: E402
from ispatialgym.grasping import validate_floating as VF  # noqa: E402
from ispatialgym.robot import conventions as C  # noqa: E402
from ispatialgym.sampling import grasp_library as GL  # noqa: E402

BAR = (0.04, 0.04, 0.10)


# ------------------------------------------------------------------------------------------------ re-targeting


def _eef(approach, closing, pad):
    z = np.asarray(approach, float) / np.linalg.norm(approach)
    y = np.asarray(closing, float) / np.linalg.norm(closing)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = np.cross(y, z), y, z
    T[:3, 3] = np.asarray(pad, float) - z * CA.CONTACT_Z  # the eef origin: the pad centre is CONTACT_Z along the approach
    return T


def test_panda_to_eef_puts_the_pad_centre_on_the_panda_tcp_and_closes_along_panda_x():
    Tp = np.eye(4)
    Tp[:3, :3] = trimesh.transformations.rotation_matrix(0.7, [0.3, -0.2, 0.9])[:3, :3]
    Tp[:3, 3] = [0.1, -0.05, 0.3]
    Te = FL.panda_to_eef(Tp[None])[0]
    tcp = Tp[:3, 3] + Tp[:3, 2] * FL.PANDA_DEPTH
    pad = Te[:3, 3] + Te[:3, 2] * CA.CONTACT_Z
    assert np.allclose(pad, tcp, atol=1e-9)
    assert np.allclose(Te[:3, 2], Tp[:3, 2])  # the approach is kept
    assert abs(abs(float(Te[:3, 1] @ Tp[:3, 0])) - 1.0) < 1e-9  # R1Pro closes along y = Panda's closing x
    assert np.allclose(FL.T_PANDA_TO_EEF[:3, 3], [0, 0, FL.PANDA_DEPTH - C.CONTACT_CENTER_Z])  # the legacy T_CONV


# ------------------------------------------------------------------------------------------------ identity and jobs


def test_floating_is_part_of_the_entry_id_and_never_a_tabletop_id():
    assert S.entry_id("apple", "omzprq", None, "floating") == "apple__omzprq__s1__floating"
    assert S.entry_id("apple", "omzprq", None, "floating") != S.entry_id("apple", "omzprq", None)
    assert S.entry_id("teaspoon", "eilfij", [1, 1, 2], "floating") == "teaspoon__eilfij__s1x1x2__floating"
    # existing ids unchanged
    assert S.entry_id("apple", "omzprq", None) == "apple__omzprq__s1"
    assert S.entry_id("lid", "tnwisp", None, "attachment_free") == "lid__tnwisp__s1__attachment_free"


def test_floating_jobs_from_legacy_floating_keys():
    j = FL.floating_job("teaspoon__eilfij__sf", {"category": "teaspoon", "model": "eilfij", "pipeline": "graspgen", "mode": "C", "constraint": "floating",
                                                  "scale": [1.0, 1.0, 2.0], "task": "mental_rotation"}, ["mental_rotation"])
    assert j["id"] == "teaspoon__eilfij__s1x1x2__floating" and j["legacy_key"] == "teaspoon__eilfij__sf" and j["constraint"] == "floating"
    assert j["floating_mode"] == "C" and j["held_tasks"] == ["mental_rotation"] and j["scale"] == [1.0, 1.0, 2.0]
    k = FL.floating_job("carving_knife__alekva", {"pipeline": "graspgen", "mode": "H", "constraint": "handle_half"})
    assert k["floating_mode"] == "H+handle" and k["id"] == "carving_knife__alekva__s1__floating"
    m = FL.floating_job("classroom_mock_up__isgsma__f", {"category": "classroom_mock_up", "model": "isgsma", "dataset": "ispatialgym-assets", "mode": "C"})
    assert m["dataset"] == "ispatialgym-assets" and m["id"] == "classroom_mock_up__isgsma__s1__floating"


def test_manifest_floating_job_reads_only_graspgen_keys(tmp_path):
    root = tmp_path / "legacy"
    (root / "results").mkdir(parents=True)
    (root / "results" / "apple__aaa.json").write_text(json.dumps({"pipeline": "graspgen", "mode": "C", "task": "object_scaling", "category": "apple", "model": "aaa"}))
    (root / "results" / "apple__aaa__t.json").write_text(json.dumps({"pipeline": "a1", "constraint": "tabletop"}))
    j = MF.floating_job_of("apple__aaa", str(root))
    assert j["held_tasks"] == ["object_scaling"] and j["id"] == "apple__aaa__s1__floating"
    assert MF.floating_job_of("apple__aaa__t", str(root)) is None


# ------------------------------------------------------------------------------------------------ candidates on a synthetic bar


def _bar_dir(tmp_path):
    d = tmp_path / "models" / "bar__xxx__s1__floating"
    d.mkdir(parents=True)
    box = trimesh.creation.box(extents=BAR)
    S.save_json(str(d / "geometry.json"), {"format": "isg-object-geometry/1", "category": "bar", "model": "xxx", "scale": [1, 1, 1], "meta_links": {},
                                            "pieces": [{"name": "base_link__c0", "link": "base_link", "vertices": box.vertices.tolist(), "faces": box.faces.tolist()}]})
    return str(d)


def _graspgen(d, T_eef, conf):
    Tp = np.einsum("nij,jk->nik", np.asarray(T_eef), np.linalg.inv(FL.T_PANDA_TO_EEF))
    np.savez(os.path.join(d, FL.GRASPGEN_FILE), T=Tp, conf=np.asarray(conf, float))
    S.save_json(os.path.join(d, FL.GRASPGEN_META), {"schema": "isg-graspgen-floating/1", "n": len(Tp)})


def test_floating_candidates_pass_the_hand_object_gates_and_no_support_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ISG_GLIB_GATE_DUMP", "1")
    d = _bar_dir(tmp_path)
    good = [_eef([0, 1, 0], [1, 0, 0], [0, 0, z]) for z in (-0.03, 0.0, 0.03)]  # side pinches across the 4 cm faces
    under = _eef([0, 0, 1], [1, 0, 0], [0, 0, -0.03])  # approach from BELOW (a support would forbid it; floating has none)
    inside = _eef([0, 0, 1], [1, 0, 0], [0, 0, 0.0])  # along the long axis to its centre: the palm inside the bar
    wide = _eef([1, 0, 0], [0, 0, 1], [0, 0, 0])  # closing along the 10 cm axis: wider than the aperture
    T = np.stack(good + [under, inside, wide])
    _graspgen(d, T, [0.9, 0.8, 0.7, 0.85, 0.95, 0.99])
    job = {"id": "bar__xxx__s1__floating", "category": "bar", "model": "xxx", "scale": [1, 1, 1], "constraint": "floating", "floating_mode": "C",
           "legacy_key": "bar__xxx"}
    out = FL.generate(job, d, cfg=FL.FloatingConfig(max_representatives=10))
    lg = out["log"]["graspgen"]
    assert lg["n_graspgen"] == 6 and lg["n_width_antipodal"] == 5  # the wide pinch: width_range
    assert lg["n_object_clear"] == 4  # the hand inside the bar
    assert out["constraint"] == "floating" and out["pipeline"] == FL.PIPELINE and out["families"] == {}
    assert len(out["candidates"]) == 4 and out["representatives"] == [c["name"] for c in out["candidates"]]
    for c in out["candidates"]:
        assert c["preshape_q"] == FL.FloatingConfig().open_q == CA.FINGER_OPEN  # the held sampler opens the fingers fully
        assert abs(c["width_m"] - 0.04) < 1e-3 and c["geometry"]["ok"] and c["geometry"]["support"] is None
    approaches = sorted(round(float(S.pose_of(c["T_object_eef"])[2, 2]), 3) for c in out["candidates"])
    assert approaches[-1] == 1.0  # the from-below pinch survived: no support gate
    dump = S.load_json(os.path.join(d, CA.GATE_DUMP_FILE))
    gates = [g["gate"] for g in dump["passes"][0]["pools"]["graspgen"]]
    assert gates[:6] == ["pool", "width_antipodal", "object_clear", "sweep", "lever", "geometry"] and "support_retreat" not in gates
    assert dump["constraint"] == "floating"


def test_floating_candidates_need_the_graspgen_stage(tmp_path):
    d = _bar_dir(tmp_path)
    with pytest.raises(FileNotFoundError):
        FL.generate({"category": "bar", "model": "xxx", "constraint": "floating"}, d)
    rec = PL._cand_one(({"id": "bar__xxx__s1__floating", "category": "bar", "model": "xxx", "scale": [1, 1, 1], "constraint": "floating"}, str(tmp_path), False))
    assert rec["ok"] is False and "graspgen" in rec["error"] and rec["constraint"] == "floating"


def test_mode_scores_follow_the_legacy_modes():
    conf = np.array([0.90, 0.95, 0.94])
    perp = np.array([0.001, 0.05, 0.002])
    ones = np.ones(3)
    s_h, _ = FL.mode_scores("H", conf, perp, 0.1, ones, ones, ones, FL.FloatingConfig())
    assert int(np.argmax(s_h)) == 2  # confidence-first; within 0.01 of the best confidence the CoM breaks the tie
    s_c, keep = FL.mode_scores("C", conf, perp, 0.1, ones, ones, ones, FL.FloatingConfig())
    assert keep.tolist() == [True, False, True] and int(np.argmax(s_c)) == 0  # the lever restriction, the confidence gate, then the CoM
    _, keep_all = FL.mode_scores("C", conf, np.full(3, 0.09), 0.1, ones, ones, ones, FL.FloatingConfig())
    assert keep_all.all()  # none within the mode's lever cap: the legacy rule keeps them all
    s_k, _ = FL.mode_scores("H+handle", conf, perp, 0.1, ones, ones, ones, FL.FloatingConfig(), th_=np.array([0.3, 0.9, 0.1]))
    assert int(np.argmax(s_k)) == 0  # the middle of the handle


# ------------------------------------------------------------------------------------------------ the lookup


@pytest.fixture
def v2root(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_GRASP_LIBRARY_V2", str(tmp_path))
    monkeypatch.delenv("ISPATIALGYM_V2_ONLY", raising=False)
    GL._v2_index.cache_clear()
    yield str(tmp_path)
    GL._v2_index.cache_clear()


def _lib(root, category, model, scale, constraint, legacy_key=None, legacy_keys=None, grasps=(("gg_0", None), ("gg_1", None)), rests=None):
    d = S.model_dir(category, model, scale, root, constraint)
    os.makedirs(d, exist_ok=True)
    S.save_json(os.path.join(d, "library.json"), {
        "schema": S.LIBRARY, "category": category, "model": model, "scale": list(S.canonical_scale(scale)), "arm": "right", "constraint": constraint,
        "legacy_key": legacy_key, "legacy_keys": legacy_keys, "policy_key": CP.DEFAULT.key("right"), "rests": rests or {},
        "grasps": [{"name": n, "rest_id": r, "T_object_eef": {"position": [0, 0, 0.1], "orientation_xyzw": [0, 0, 0, 1]}, "width_m": 0.03, "finger_q": [0.02, 0.02]}
                   for n, r in grasps]})
    return d


def test_a_floating_entry_serves_every_grasp_has_no_rest_and_never_a_placement(v2root):
    _lib(v2root, "apple", "omzprq", None, "floating", legacy_keys=["apple__omzprq"])
    lib = GL.v2_entry("apple__omzprq")
    assert lib is not None and lib.floating and lib.rests == {} and GL.v2_rest_id(lib) is None
    assert [g.name for g in GL.load_grasps("apple__omzprq")] == ["gg_0", "gg_1"]
    assert all(g.meta["rest_id"] is None for g in GL.load_grasps("apple__omzprq"))
    with pytest.raises(ValueError):
        GL.rest_quat_xyzw("apple__omzprq")  # a floating entry never serves a placement
    assert lib.identity()["constraint"] == "floating"
    with pytest.raises(S.IncompatibleRecord):
        S.check_compatible(lib, rest_id="r0")


def test_a_floating_entry_with_a_rest_reference_is_incompatible(v2root):
    _lib(v2root, "apple", "bad", None, "floating", legacy_key="apple__bad", grasps=(("gg_0", "r0"),))
    with pytest.raises(S.IncompatibleRecord):
        GL.v2_entry("apple__bad")


def test_tabletop_identity_and_lookup_unchanged(v2root):
    rest = {"r0": {"schema": S.REST, "rest_id": "r0", "quat_xyzw": [0, 0, 0, 1], "h_root": 0.04, "stable": True}}
    _lib(v2root, "apple", "omzprq", None, "tabletop", legacy_key="apple__omzprq__t", grasps=(("top_0", "r0"),), rests=rest)
    lib = GL.v2_entry("apple__omzprq__t")
    assert not lib.floating and "constraint" not in lib.identity() and GL.v2_rest_id(lib) == "r0"
    assert np.allclose(GL.rest_quat_xyzw("apple__omzprq__t").numpy(), [0, 0, 0, 1])
    # the floating entry of the same model is a separate directory and a separate key
    _lib(v2root, "apple", "omzprq", None, "floating", legacy_keys=["apple__omzprq"])
    GL._v2_index.cache_clear()
    assert GL.v2_entry("apple__omzprq").floating and not GL.v2_entry("apple__omzprq__t").floating


def test_held_sampler_refuses_a_non_floating_entry_and_enforces_scale(v2root, monkeypatch):
    from ispatialgym.sampling.v2.tasks import held_object as HO

    HO.check_v2_floating("nothing__here", None)  # no v2 entry: the legacy path, nothing to check
    rest = {"r0": {"schema": S.REST, "rest_id": "r0", "quat_xyzw": [0, 0, 0, 1], "h_root": 0.04, "stable": True}}
    _lib(v2root, "mug", "ppzttc", None, "tabletop", legacy_keys=["mug__ppzttc"], grasps=(("top_0", "r0"),), rests=rest)
    GL._v2_index.cache_clear()
    with pytest.raises(ValueError, match="not floating"):
        HO.check_v2_floating("mug__ppzttc", None)
    _lib(v2root, "teaspoon", "eilfij", [1, 1, 2], "floating", legacy_keys=["teaspoon__eilfij__sf"])
    GL._v2_index.cache_clear()
    HO.check_v2_floating("teaspoon__eilfij__sf", [1.0, 1.0, 2.0])
    HO.check_v2_floating("teaspoon__eilfij__sf", None)  # logged only (v2 without v2 only)
    monkeypatch.setenv("ISPATIALGYM_V2_ONLY", "1")
    with pytest.raises(HO.V2ScaleMismatch):
        HO.check_v2_floating("teaspoon__eilfij__sf", None)
    assert HO.floating_key("teaspoon", "eilfij", "teaspoon__eilfij__sf") == "teaspoon__eilfij__sf"


# ------------------------------------------------------------------------------------------------ the validator (pure parts)


def test_drift_limits_are_the_tasks_goal_tolerances_and_the_verdict_rule(monkeypatch):
    mr, os_small, os_big = VF.drift_limits("mental_rotation"), VF.drift_limits("object_scaling", 0.05), VF.drift_limits("object_scaling", -0.30)
    assert mr == {"pos_m": None, "rot_rad": 0.15}
    assert os_small == {"pos_m": 0.02, "rot_rad": None} and math.isclose(os_big["pos_m"], 0.06)
    # every collected successful demo (largest in-hand drift 0.24 mm / 4.4 mrad) passes
    assert VF.drift_ok((0.00024, 0.0044), mr) and VF.drift_ok((0.00024, 0.0044), os_small)
    assert not VF.drift_ok((0.0, 0.2), mr) and VF.drift_ok((0.05, 0.0), mr) and not VF.drift_ok((0.03, 0.0), os_small)
    ok, ag, unr = {"outcome": "ok"}, {"outcome": "ag_not_formed"}, {"outcome": "unreachable"}
    assert VF.verdict_of([ok, ag, unr]) == ("valid", [])
    assert VF.verdict_of([ok, {"outcome": "dropped_in_motion"}]) == ("invalid", ["dropped_in_motion"])
    assert VF.verdict_of([ok, {"outcome": "drift_beyond_task_tolerance"}])[0] == "invalid"
    assert VF.verdict_of([]) == ("invalid", ["no_drawn_motion"]) and VF.verdict_of([ag, ag]) == ("invalid", ["ag_not_formed"])
    assert VF.verdict_of([ag, unr]) == ("invalid", ["held_motion_unreachable"])
    a, b = VF.motion_rng("e", "gg_0", "mental_rotation").random(3), VF.motion_rng("e", "gg_0", "mental_rotation").random(3)
    assert np.allclose(a, b) and not np.allclose(a, VF.motion_rng("e", "gg_1", "mental_rotation").random(3))
    assert VF.held_tasks_of({"held_tasks": ["object_scaling"]}) == ["object_scaling"]
    assert VF.held_tasks_of({}) == ["mental_rotation", "object_scaling"]
    monkeypatch.setenv("ISG_GLIB_FLOAT_VAL_OVERRIDES", '{"motion_pairs": 5}')
    assert PL.floating_validate_overrides().motion_pairs == 5


def test_validator_closure_constants_are_the_held_samplers():
    from ispatialgym.sampling import generation_defaults as GD
    from ispatialgym.sampling.v2.tasks import held_object as HO

    cfg = VF.FloatingValidateConfig()
    assert cfg.attach_steps == HO.GRASP_ATTACH_STEPS and cfg.gravity_steps == HO.GRASP_GRAVITY_STEPS
    for task in VF.HELD_TASKS:
        g = GD.defaults_for(task)
        assert cfg.advance_m_per_step == g["grasp_finger_advance_m_per_step"] and cfg.press_m == g["grasp_press_m"]
    assert VF.hold_of("object_scaling").roll_deg == 90.0 and VF.hold_of("mental_rotation").roll_deg == 15.0
    # the init re-posing steps of establish_grasp (``for _ in range(8)``)
    import inspect

    assert "for _ in range(8):" in inspect.getsource(HO.HeldObjectV2.establish_grasp)


def test_floating_library_keeps_grasps_valid_at_every_hold():
    cand_doc = {"arm": "right", "pipeline": FL.PIPELINE, "floating_mode": "C",
                "candidates": [{"name": n, "T_object_eef": {"position": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1]}, "width_m": 0.03, "preshape_q": 0.05,
                                "floating_mode": "C"} for n in ("gg_0", "gg_1")]}

    def tr(n, task, ok):
        return {"candidate": n, "task": task, "verdict": "valid" if ok else "invalid", "closure": {"fingers_final_m": [0.02, 0.02]},
                "achieved": {"T_object_eef_at_attach": None}, "legs_executed": 3, "legs_unreachable": []}

    trials = [tr("gg_0", "mental_rotation", True), tr("gg_0", "object_scaling", True), tr("gg_1", "mental_rotation", True), tr("gg_1", "object_scaling", False)]
    lib = VF.build_library({"category": "a", "model": "b", "scale": None, "legacy_key": "a__b"}, cand_doc, trials, ["mental_rotation", "object_scaling"],
                           CP.DEFAULT, {})
    assert [g["name"] for g in lib["grasps"]] == ["gg_0"] and lib["constraint"] == "floating" and lib["rests"] == {}
    assert lib["grasps"][0]["rest_id"] is None and lib["legacy_key"] == "a__b"


def test_pipeline_skips_floating_jobs_in_the_rest_retry(tmp_path):
    from types import SimpleNamespace

    j = {"id": "apple__omzprq__s1__floating", "category": "apple", "model": "omzprq", "scale": [1, 1, 1], "constraint": "floating"}
    d = S.job_dir(j, str(tmp_path))
    os.makedirs(d)
    S.save_json(os.path.join(d, "rests.json"), {"rests": {"r0_other": {"intended": False, "stable": True, "quat_xyzw": [0, 0, 0, 1]}}})
    assert PL.rest_retry_jobs(SimpleNamespace(root=str(tmp_path), force=False), [j]) == []
