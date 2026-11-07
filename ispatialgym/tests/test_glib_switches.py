"""i7-switches 2026-11-07 (owner decision 1: the plate switches ``blade_contact_at_retreat`` / ``retreat_shift_exact`` adopted after a
count): the switch state is part of a v2 entry's identity - recorded in ``library.json`` (``build_library``), read back from
``candidates.json`` for an entry validated before the record, all off for an entry generated before the switches existed; it enters
``ModelLibrary.identity`` (the reach maps' ``object_key``) only when a switch is on, so every existing key stays; ``check_compatible`` /
``load_model`` can require a state; a map of the off entry never serves the on entry. And the two switches themselves: the exact
retreat can keep a pose the as-written bookkeeping drops, and the blade lookup sits at the recorded (exact) retreat. CPU only."""

import json
import math
import os

import numpy as np
import pytest

from ispatialgym.grasping import schema as S

#: ``identity_key`` of :func:`_lib_data` computed with the code BEFORE the switch state existed (of-v2maps 33d8ecaf)
GOLDEN_KEY = "cd78dd8e53bcfdbf"
#: every candidate switch off (i7-padsweep added the third)
OFF3 = {"blade_contact_at_retreat": False, "retreat_shift_exact": False, "sweep_from_preshape": False}


def _lib_data(**extra):
    rest = {"schema": S.REST, "rest_id": "r0", "quat_xyzw": [0.0, 0.0, 0.0, 1.0], "h_root": 0.0123, "reference_yaw": 0.0, "stable": True}
    d = {"schema": S.LIBRARY, "category": "plate", "model": "synthetic", "scale": [1.0, 1.0, 1.0], "dataset": "behavior-1k-assets", "policy_key": "pk0",
         "rests": {"r0": rest}, "families": {"f0": {"type": "none", "representative": "top_0"}},
         "grasps": [{"name": "top_0", "rest_id": "r0", "family_id": "f0", "T_object_eef": {"position": [0.0, 0.0, 0.1], "orientation_xyzw": [1.0, 0.0, 0.0, 0.0]},
                     "finger_q": [0.01, 0.01], "width_m": 0.02}]}
    d.update(extra)
    return d


def _lib(data, path="x"):
    return S.ModelLibrary(path=path, data=data, rests={k: S.Rest.from_record(v) for k, v in data["rests"].items()})


def _write_entry(root, data, cand_config=None):
    """An entry directory under ``root`` (library.json + optionally a candidates.json with ``config``) at its entry id."""
    d = S.model_dir(data["category"], data["model"], data["scale"], str(root))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "library.json"), "w") as f:
        json.dump(data, f)
    if cand_config is not None:
        with open(os.path.join(d, "candidates.json"), "w") as f:
            json.dump({"schema": S.CANDIDATES, "config": cand_config, "candidates": [], "representatives": []}, f)
    return d


# ------------------------------------------------------------------------------------------------ the identity


def test_switch_state_reads_off_and_keeps_every_existing_key(tmp_path):
    # i7-padsweep: a third switch (sweep_from_preshape) joined; the record carries it, the identity only when on (keys unchanged)
    assert S.CANDIDATE_SWITCHES == ("blade_contact_at_retreat", "retreat_shift_exact", "sweep_from_preshape")
    assert S.candidate_switches(None) == S.candidate_switches({}) == OFF3
    lib = _lib(_lib_data())
    assert lib.candidate_switches() == OFF3
    assert "candidate_switches" not in lib.identity() and lib.identity_key() == GOLDEN_KEY
    # an explicit all-off record (every build from now on), an old candidates.json config without the fields, one with them False
    assert _lib(_lib_data(candidate_switches={"blade_contact_at_retreat": False, "retreat_shift_exact": False})).identity_key() == GOLDEN_KEY
    for i, cfg in enumerate(({"n_candidates": 6000}, {"blade_contact_at_retreat": False, "retreat_shift_exact": False})):
        root = tmp_path / f"r{i}"
        _write_entry(root, _lib_data(), cfg)
        got = S.load_model("plate", "synthetic", [1, 1, 1], str(root))
        assert got.candidate_switches() == OFF3 and got.identity_key() == GOLDEN_KEY


def test_switch_state_on_changes_the_identity():
    keys = {}
    for sw in ({"blade_contact_at_retreat": True}, {"retreat_shift_exact": True}, {"blade_contact_at_retreat": True, "retreat_shift_exact": True}):
        lib = _lib(_lib_data(candidate_switches=sw))
        full = S.candidate_switches(sw)
        assert lib.candidate_switches() == full and lib.identity()["candidate_switches"] == {k: full[k] for k in S._IDENTITY_PAIR}
        keys[tuple(sorted(sw))] = lib.identity_key()
    assert len(set(keys.values())) == 3 and GOLDEN_KEY not in keys.values()  # the same grasps under each state: three distinct entries


def test_switch_state_from_the_candidates_config_when_library_json_has_no_record(tmp_path):
    # an entry generated with the switches (6e418172 opt-in) and validated before library.json recorded them
    _write_entry(tmp_path, _lib_data(), {"blade_contact_at_retreat": True, "retreat_shift_exact": True})
    lib = S.load_model("plate", "synthetic", [1, 1, 1], str(tmp_path))
    assert lib.candidate_switches() == dict(OFF3, blade_contact_at_retreat=True, retreat_shift_exact=True)
    assert lib.identity_key() != GOLDEN_KEY
    # the library.json record wins over the candidates config
    _write_entry(tmp_path / "rec", _lib_data(candidate_switches={"blade_contact_at_retreat": False, "retreat_shift_exact": False}),
                 {"blade_contact_at_retreat": True, "retreat_shift_exact": True})
    assert S.load_model("plate", "synthetic", [1, 1, 1], str(tmp_path / "rec")).identity_key() == GOLDEN_KEY


def test_a_reader_can_require_the_switch_state(tmp_path):
    off, on = _lib(_lib_data()), _lib(_lib_data(candidate_switches={"blade_contact_at_retreat": True, "retreat_shift_exact": True}))
    S.check_compatible(off, candidate_switches={"blade_contact_at_retreat": False})
    S.check_compatible(on, candidate_switches={"blade_contact_at_retreat": True, "retreat_shift_exact": True})
    with pytest.raises(S.IncompatibleRecord):
        S.check_compatible(off, candidate_switches={"retreat_shift_exact": True})
    with pytest.raises(S.IncompatibleRecord):
        S.check_compatible(on, candidate_switches={"blade_contact_at_retreat": False})
    with pytest.raises(S.IncompatibleRecord):
        S.check_compatible(off, candidate_switches={"no_such_switch": False})
    S.check_compatible(off)  # no requirement: as before
    _write_entry(tmp_path, _lib_data())
    assert S.load_model("plate", "synthetic", [1, 1, 1], str(tmp_path), require={"candidate_switches": {"blade_contact_at_retreat": False}}) is not None
    with pytest.raises(S.IncompatibleRecord):
        S.load_model("plate", "synthetic", [1, 1, 1], str(tmp_path), require={"candidate_switches": {"blade_contact_at_retreat": True}})


def test_build_library_records_the_switch_state():
    from ispatialgym.grasping import contact_policy as CP
    from ispatialgym.grasping import validate_tabletop as VT

    job = {"category": "plate", "model": "synthetic", "scale": [1, 1, 1], "constraint": "tabletop", "legacy_key": "plate__synthetic"}
    for cfg, want in ((None, (False, False)), ({"n_candidates": 6000}, (False, False)),
                      ({"blade_contact_at_retreat": True, "retreat_shift_exact": False}, (True, False)),
                      ({"blade_contact_at_retreat": True, "retreat_shift_exact": True}, (True, True))):
        cand_doc = {"candidates": [], "families": {}, "arm": "right", "pipeline": "a1v2", **({"config": cfg} if cfg is not None else {})}
        lib = VT.build_library(job, cand_doc, {}, [], CP.DEFAULT, None)
        assert lib["candidate_switches"] == dict(zip(S.CANDIDATE_SWITCHES, want + (False,))) and lib["legacy_key"] == "plate__synthetic"


def test_a_map_of_the_off_entry_never_serves_the_on_entry(tmp_path):
    from ispatialgym.sampling import reach_maps as RM

    off, on = _lib(_lib_data()), _lib(_lib_data(candidate_switches={"blade_contact_at_retreat": True, "retreat_shift_exact": True}))
    cfg = RM.MapConfig().key()
    meta = {"schema": RM.SCHEMA, "kind": "tabletop", "arm": "right", "object_key": off.identity_key(), "rest_id": "r0",
            "height": {"value": 0.8126, "anchor_key": "0.8126"}, "edge_distances": list(cfg["edge_distances"]), "config": cfg, "policy_key": "pk0",
            "robot": "rbt", "brace": None, "object": off.identity(), "grasps": [{"name": "top_0", "base": "top_0", "twin": False}], "trunks": []}
    meta["key"] = RM.map_key(meta)
    base = os.path.join(str(tmp_path), f"map__plate__synthetic__h0.8126__{meta['key']}")
    np.savez_compressed(base + ".npz", cells=np.zeros((1, 2), dtype=np.float32), yaws=np.zeros(1, dtype=np.float32))
    with open(base + ".json", "w") as f:
        json.dump(meta, f)
    assert RM.ReachMap.find(str(tmp_path), off, 0.8126, rest_id="r0") is not None
    assert RM.ReachMap.find(str(tmp_path), on, 0.8126, rest_id="r0") is None


def test_the_candidates_log_line_names_both_switches(tmp_path, monkeypatch):
    from ispatialgym.grasping import candidates as CA
    from ispatialgym.grasping import pipeline as PL

    def fake_generate(job, d, cfg=None, densify=()):
        return {"candidates": [], "families": {}, "representatives": [], "log": {}}

    monkeypatch.setattr(CA, "generate", fake_generate)
    job = {"id": "plate__synthetic__s1", "category": "plate", "model": "synthetic", "scale": [1, 1, 1], "constraint": "tabletop",
           "cand_overrides": {"retreat_shift_exact": True}}
    rec = PL._cand_one((job, str(tmp_path), True))
    assert rec["ok"] and rec["cand_config"]["retreat_shift_exact"] is True and "blade_contact_at_retreat" in rec["cand_config"]


# ------------------------------------------------------------------------------------------------ the switches


def _down(z: float) -> np.ndarray:
    """A top grasp: the eef at height z, approach (+z of the eef) straight down."""
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[2, 3] = z
    return T


def test_exact_retreat_keeps_a_pose_the_written_bookkeeping_drops():
    """A hand clear only after a >= 14 mm retreat: as written the failed 5 / 10 mm tries add 15 mm to the shift, so the 15 mm try would put
    the contact past ``tip_max_m`` and is never made (dropped); with the exact shift it is tried and clears (the count can go up)."""
    from ispatialgym.grasping import candidates as CA

    from dataclasses import asdict

    z0 = 0.10

    def clear_fn(T, q):
        return np.where(T[:, 2, 3] - z0 >= 0.014, 0.01, -0.01)

    got = {}
    for exact in (False, True):
        cfg = CA.CandidateConfig(**{**asdict(CA.CandidateConfig()), "retreat_shift_exact": exact})
        T = _down(z0)[None].copy()
        shift = np.zeros(1)
        clear = CA._clear_grouped(clear_fn, T, np.array([0.02]))
        ok = clear >= cfg.obj_margin_m
        real = CA._object_retreats(clear_fn, T, np.array([0.02]), shift, clear, ok, cfg)
        got[exact] = (bool(ok[0]), float(real[0]), float(shift[0]), float(T[0, 2, 3] - z0))
    assert got[False][0] is False and math.isclose(got[False][1], 0.0, abs_tol=1e-12) and math.isclose(got[False][2], 0.015, abs_tol=1e-12)
    assert got[True][0] is True and math.isclose(got[True][1], 0.015, abs_tol=1e-12) and math.isclose(got[True][2], 0.015, abs_tol=1e-12)
    assert math.isclose(got[True][3], 0.015, abs_tol=1e-12)


def _run_generate(tmp_path, monkeypatch, **cfg_kw):
    """``candidates.generate`` on a box with a stub sampler (top grasps over its top face); returns the output and, per call of
    ``finger_blades_clear``, the ``contact_z`` it was given (None: the default)."""
    trimesh = pytest.importorskip("trimesh")
    from dataclasses import asdict

    from ispatialgym.grasping import candidates as CA
    from ispatialgym.grasping import object_geometry as OG

    box = trimesh.creation.box(extents=(0.06, 0.03, 0.05))
    box.apply_translation((0.0, 0.0, 0.025))
    g = OG.ObjectGeometry(info={"meta_links": {}}, pieces=[box], union=box.copy())
    monkeypatch.setattr(OG, "load", lambda model_dir: g)
    T_stub = [_down(z) for z in (0.03, 0.02, 0.01, 0.0)]
    for T, dx in zip(T_stub, (-0.015, -0.005, 0.005, 0.015)):
        T[0, 3] = dx

    class _Sampler:
        @staticmethod
        def sample_antipodal(*a, **k):
            return list(T_stub)

    monkeypatch.setattr(CA, "_isaac_sampler", lambda: _Sampler)
    calls = []
    real_fbc = CA.finger_blades_clear

    def spy(geom, hc, Ts, pre_q, contact_q, **kw):
        calls.append(None if "contact_z" not in kw else np.asarray(kw["contact_z"]).reshape(-1).copy())
        return real_fbc(geom, hc, Ts, pre_q, contact_q, **kw)

    monkeypatch.setattr(CA, "finger_blades_clear", spy)
    job = {"category": "box", "model": "synthetic", "scale": [1, 1, 1], "constraint": "tabletop", "rest_hint_wxyz": None, "id": "box__synthetic__s1"}
    d = tmp_path / ("_".join(f"{k}{int(bool(v))}" for k, v in sorted(cfg_kw.items())) or "default")
    d.mkdir()
    cfg = CA.CandidateConfig(**{**asdict(CA.CandidateConfig()), "modes": ("top",), "topk": 4, "gate_dump": False, **cfg_kw})
    return CA.generate(job, str(d), cfg=cfg, densify=()), calls


def test_blade_lookup_sits_at_the_recorded_retreat_with_both_switches(tmp_path, monkeypatch):
    from ispatialgym.grasping import candidates as CA

    out_off, calls_off = _run_generate(tmp_path, monkeypatch, blade_contact_at_retreat=False, retreat_shift_exact=False)
    assert calls_off and all(c is None for c in calls_off)  # as written: the default CONTACT_Z lookup
    out_on, calls_on = _run_generate(tmp_path, monkeypatch, blade_contact_at_retreat=True, retreat_shift_exact=True)
    assert len(calls_on) == 1 and len(out_on["candidates"]) == len(calls_on[0])
    shifts = np.array([c["features"]["shift_m"] for c in out_on["candidates"]])
    assert (shifts > 0.005).any()  # a retreated grasp (the support retreat: the stub's lowest pose has its tips 19 mm into the table)
    assert np.allclose(calls_on[0], CA.CONTACT_Z + shifts, atol=1e-5)  # the pads' real contact = the recorded (exact) shift
    assert out_on["config"]["blade_contact_at_retreat"] is True and out_on["config"]["retreat_shift_exact"] is True


def test_new_builds_record_both_switches_on(monkeypatch):
    """i7-switches: after the count both switches are the default for new builds - a default config records them on (so a new entry's
    identity carries them), an explicit False reproduces the build before the adoption."""
    from dataclasses import asdict

    from ispatialgym.grasping import candidates as CA
    from ispatialgym.grasping import pipeline as PL

    monkeypatch.delenv("ISG_GLIB_CAND_OVERRIDES", raising=False)
    assert S.candidate_switches(asdict(CA.CandidateConfig())) == dict(OFF3, blade_contact_at_retreat=True, retreat_shift_exact=True, sweep_from_preshape=True)
    cfg, _ = PL.candidate_overrides({"cand_overrides": {"blade_contact_at_retreat": False, "retreat_shift_exact": False, "sweep_from_preshape": False}})
    assert S.candidate_switches(asdict(cfg)) == OFF3
