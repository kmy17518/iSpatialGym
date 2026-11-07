"""i7-ladder 2026-11-07: the grasp ladder's steps 3 (agent proposals) and 4 (legacy grasps) - given poses through the shared gates, the
proposal contract, the entry's source record, selection / markers / reports / strips, the step-4 list and its owner gate. CPU only, no
simulator, no Behavior-1K asset (synthetic geometry; every output under tmp_path)."""

import json
import math
import os
import sys

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from ispatialgym.grasping import candidates as CA  # noqa: E402
from ispatialgym.grasping import ladder as LD  # noqa: E402
from ispatialgym.grasping import object_geometry as OG  # noqa: E402
from ispatialgym.grasping import pipeline as PL  # noqa: E402
from ispatialgym.grasping import schema as S  # noqa: E402
from ispatialgym.grasping import validate_tabletop as VT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "grasp_ladder"))


def _box(ext=(0.03, 0.05, 0.04)) -> OG.ObjectGeometry:
    m = trimesh.creation.box(extents=ext)
    return OG.ObjectGeometry(info={"meta_links": {}, "mass": 0.1}, pieces=[m], union=trimesh.util.concatenate([m]))


def _cfg(**kw) -> CA.CandidateConfig:
    from dataclasses import asdict

    return CA.CandidateConfig(**{**asdict(CA.CandidateConfig()), **kw})


def _prop(**kw):
    p = {"from_marker": None, "contact_mm": [0.0, 0.0, 10.0], "closing_axis": [1.0, 0.0, 0.0], "approach": [0.0, 0.0, -1.0],
         "pad_offset_mm": [0.0], "expand": None, "rationale": "pinch the 30 mm width from above", "confidence": "high"}
    p.update(kw)
    return p


def _doc(props, i="box__x__s1", feasible=True):
    return {"contract": LD.CONTRACT, "id": i, "feasible": feasible, "proposals": props, "object_note": "synthetic"}


# ------------------------------------------------------------------------------------------------ defaults unchanged


def test_given_pose_fields_default_off_and_families_unchanged():
    c = CA.CandidateConfig()
    assert tuple(c.agent_proposals) == () and tuple(c.legacy_grasps) == () and tuple(c.families) == ()
    assert set(CA.FAMILIES) == {"lip_shallow", "rim_straddle"} and CA.POSE_FAMILIES == ("agent", "legacy")
    assert not CA.given_poses_only(c) and not CA.given_poses_only(_cfg(families=("agent", "lip_shallow"), modes=()))
    assert CA.given_poses_only(_cfg(families=("agent",), modes=())) and not CA.given_poses_only(_cfg(families=("agent",)))


# ------------------------------------------------------------------------------------------------ proposal -> poses


def test_proposal_frame_is_a_right_handed_eef_frame_and_rejects_degenerate_pinches():
    R = CA.proposal_frame([0, 0, -2.0], [1.0, 0.2, 0.3])
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and math.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[:, 2], [0, 0, -1]) and abs(R[:, 1] @ R[:, 2]) < 1e-12 and R[:, 1][0] > 0.9
    with pytest.raises(ValueError):
        CA.proposal_frame([0, 0, -1.0], [0.2, 0.0, -1.0])  # closing within 60 deg of the approach
    with pytest.raises(ValueError):
        CA.proposal_frame([0, 0, 0.0], [1.0, 0, 0])


def test_agent_poses_keep_the_contact_and_expand_as_documented():
    cfg = CA.CandidateConfig()
    p = _prop(pad_offset_mm=[0.0, 20.0, 40.0])  # 40 mm: past the fingertips (tip_max_m) -> dropped
    T, s0 = CA.agent_proposal_poses(p, cfg)
    n_ex = np.prod([len(v) for v in CA.AGENT_EXPAND_DEFAULT.values()])
    assert len(T) == n_ex * 2 and sorted(set(np.round(s0, 4))) == [0.0, 0.02]
    c = T[:, :3, 3] + T[:, :3, 2] * (CA.CONTACT_Z + s0)[:, None]
    slide = np.einsum("ij,ij->i", c - np.array([0, 0, 0.01]), T[:, :3, 0])
    assert np.allclose(c - T[:, :3, 0] * slide[:, None], [0, 0, 0.01], atol=1e-12)  # the contact moves only along the pads' width axis
    assert set(np.round(np.abs(slide) * 1000, 6)) == {0.0, 3.0}
    assert set(np.round(CA.tilt_deg(T), 6)) == {0.0, 10.0}  # tilts about the closing axis
    R = T[:, :3, :3]
    assert np.allclose(np.einsum("nij,nik->njk", R, R), np.eye(3), atol=1e-9)
    one = _prop(expand={"tilt_deg": [0.0], "yaw_deg": [0.0], "slide_mm": [0.0]}, pad_offset_mm=[5.0])
    T1, s1 = CA.agent_proposal_poses(one, cfg)
    assert len(T1) == 1 and math.isclose(s1[0], 0.005) and np.allclose(T1[0, :3, 3], [0, 0, 0.01 - (CA.CONTACT_Z + 0.005) * -1.0])
    big = _prop(expand={"tilt_deg": list(range(-40, 41, 5)), "yaw_deg": list(range(-40, 41, 5)), "slide_mm": [0, 1, 2]})
    assert len(CA.agent_proposal_poses(big, cfg)[0]) == CA.AGENT_EXPAND_MAX


def test_legacy_pose_is_the_eef_in_the_rest_frame():
    g = {"name": "grasp_0", "position": [0.1, 0.0, 0.02], "quat_xyzw": [-0.7071068, 0.7071068, 0.0, 0.0]}
    R = CA.rest_rotation([0.7071068, 0.7071068, 0.0, 0.0])
    T, s0 = CA.legacy_grasp_poses(g, R)
    assert T.shape == (1, 4, 4) and s0.tolist() == [0.0]
    assert np.allclose(T[0], R @ S.pose_mat(g["position"], g["quat_xyzw"]))
    T0 = CA.legacy_grasp_poses(g, np.eye(4))[0][0]
    assert np.allclose(T0[:3, 2], [0, 0, -1], atol=1e-6) and np.allclose(T0[:3, 1], [-1, 0, 0], atol=1e-6)  # the legacy yaml's convention


def test_pose_family_pools_split_top_and_side_and_record_errors():
    cfg = _cfg(agent_proposals=(_prop(), _prop(approach=[1.0, 0.0, -0.05], closing_axis=[0.0, 1.0, 0.0], pad_offset_mm=[0.0, 10.0]),
                                _prop(closing_axis=[0.0, 0.0, 1.0])))
    pools = CA.pose_family_pools("agent", cfg, np.eye(4))
    names = [p[0] for p in pools]
    assert names == ["agent_p0", "agent_p1_side", "agent_p2"]
    side = pools[1]
    assert side[1] == "side" and len(side[2]) and (side[3] == 0).all() and side[4]["dropped_side_offsets"] > 0
    assert pools[0][1] == "top" and pools[0][4] == {"source": "agent", "proposal": 0}
    assert len(pools[2][2]) == 0 and "error" in pools[2][4]
    lg = CA.pose_family_pools("legacy", _cfg(legacy_grasps=({"name": "g7", "position": [0, 0, 0.03], "quat_xyzw": [1.0, 0, 0, 0]},)), np.eye(4))
    assert [(p[0], p[1], len(p[2]), p[4]["source"], p[4]["legacy_grasp"]) for p in lg] == [("legacy_g7", "top", 1, "legacy", "g7")]


# ------------------------------------------------------------------------------------------------ end to end through the gates


def _stub_generate(tmp_path, monkeypatch, cfg, geom=None, name="run"):
    g = geom or _box()
    monkeypatch.setattr(OG, "load", lambda model_dir: g)

    def _no_sampler():
        raise AssertionError("given poses only: the antipodal sampler must not run")

    monkeypatch.setattr(CA, "_isaac_sampler", _no_sampler)
    d = tmp_path / name
    d.mkdir()
    job = {"category": "box", "model": "x", "scale": [1, 1, 1], "constraint": "tabletop", "rest_hint_wxyz": None, "id": "box__x__s1"}
    return CA.generate(job, str(d), cfg=cfg, densify=((4, 2),)), d


def test_generate_runs_proposals_through_the_same_gates(tmp_path, monkeypatch):
    # a 30 x 50 x 40 mm box on the table: pinch the 30 mm width from above (passes), pinch the 50 mm width from below the support
    # (approach up: the support gate), and a pinch in the air beside it (no width)
    props = (_prop(contact_mm=[0.0, 0.0, 10.0], pad_offset_mm=[0.0, 8.0]), _prop(contact_mm=[0.0, 0.0, -10.0], approach=[0, 0, 1.0], closing_axis=[0, 1.0, 0]),
             _prop(contact_mm=[80.0, 0.0, 10.0]))
    cfg = _cfg(modes=(), families=("agent",), agent_proposals=props, gate_dump=True, topk=4)
    out, d = _stub_generate(tmp_path, monkeypatch, cfg)
    log = out["log"]
    assert out["log"]["n_sampled"] == 0 and "densified_from" not in log  # no sampling, no densify pass
    assert {"agent_p0", "agent_p1_side", "agent_p2"} <= set(log) and "agent_p1" not in log and log["agent_p0"]["source"] == "agent"
    assert log["agent_p2"]["n_pool"] > 0 and log["agent_p2"]["n_width_antipodal"] == 0  # nothing between the fingers
    assert log["agent_p1_side"]["n_pool"] > 0 and log["agent_p1_side"].get("n_support", 0) == 0  # from below the support: the side lift
    good = [c for c in out["candidates"] if c["geometry"]["ok"]]
    assert good and all(c["source"] == "agent" and c["proposal"] == 0 and c["family"] == "agent_p0" for c in good)
    assert all(c["name"].startswith(c["family"] + "_") for c in out["candidates"])
    assert set(out["representatives"]) <= {c["name"] for c in good}
    dump = json.load(open(d / CA.GATE_DUMP_FILE))
    pools = dump["passes"][-1]["pools"]
    assert dump["final_pass"] == 0 and {"agent_p0", "agent_p1_side", "agent_p2"} <= set(pools)
    rec = [st for st in pools["agent_p2"] if st["gate"] == "width_antipodal"][0]
    assert rec["n_out"] == 0 and rec["dropped"] and len(rec["dropped"][0]["approach_k"]) == 3  # the axes ride along in the records
    # the report: per proposal pool the gate chain, the killing gate with its reasons
    rep = LD.entry_report(str(d))
    assert rep["pools"]["agent_p2"]["killed_by"]["gate"] == "width_antipodal" and rep["pools"]["agent_p0"]["geometry_ok"] == len(good)
    assert "agent_p0" in LD.summarize(rep)
    # the pipeline's candidates log line keeps the given-pose pools
    rec_line = {k: v for k, v in out["log"].items() if k in ("n_sampled",) or (isinstance(v, dict) and "source" in v)}
    assert set(rec_line) == {"n_sampled", "agent_p0", "agent_p1_side", "agent_p2"}


def test_default_generation_is_unchanged_by_the_new_code(tmp_path, monkeypatch):
    # the sampled path with no given poses: the same candidates whether or not proposals exist in another entry's config
    g = _box()
    monkeypatch.setattr(OG, "load", lambda model_dir: g)
    Tb = []
    for a in np.linspace(0, math.pi, 6, endpoint=False):
        R = CA.proposal_frame([0, 0, -1.0], [math.cos(a), math.sin(a), 0.0])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.array([0, 0, 0.012]) - R[:, 2] * CA.CONTACT_Z
        Tb.append(T)

    class _Sampler:
        @staticmethod
        def sample_antipodal(*a, **k):
            return list(Tb)

    monkeypatch.setattr(CA, "_isaac_sampler", lambda: _Sampler)
    job = {"category": "box", "model": "x", "scale": [1, 1, 1], "constraint": "tabletop", "rest_hint_wxyz": None, "id": "box__x__s1"}
    outs = []
    for name, cfg in (("plain", _cfg(topk=4)), ("with_agent", _cfg(topk=4, families=("agent",), agent_proposals=(_prop(),)))):
        d = tmp_path / name
        d.mkdir()
        outs.append(CA.generate(job, str(d), cfg=cfg, densify=()))
    strip = ("name", "mode", "score", "T_object_eef", "width_m", "preshape_q", "contact_q", "features", "geometry")
    plain = [{k: c[k] for k in strip} for c in outs[0]["candidates"]]
    assert plain and plain == [{k: c[k] for k in strip} for c in outs[1]["candidates"] if "source" not in c]
    assert all("source" not in c and "proposal" not in c for c in outs[0]["candidates"])
    assert outs[0]["log"]["top"] == outs[1]["log"]["top"] and "agent_p0" in outs[1]["log"]


# ------------------------------------------------------------------------------------------------ the contract


def test_check_proposals_accepts_the_contract_and_names_every_break():
    assert LD.check_proposals(_doc([_prop()]), "box__x__s1") == []
    assert LD.check_proposals(_doc([], feasible=False)) == []
    bad = LD.check_proposals(_doc([_prop(pad_offset_mm=[30.0]), _prop(closing_axis=[0, 0, 1.0]), _prop(confidence="sure"),
                                   _prop(from_marker="Z"), _prop(expand={"tilt_deg": [0, 90]}), {"contact_mm": [0, 0, 0]}]), "other")
    joined = " | ".join(bad)
    for frag in ("id 'box__x__s1' != 'other'", "proposal 0: pad_offset_mm", "proposal 1: closing axis", "proposal 2: confidence",
                 "proposal 3: from_marker", "proposal 4: expand.tilt_deg", "proposal 5: keys"):
        assert frag in joined, frag
    assert LD.check_proposals(_doc([])) == ["feasible but no proposal"]
    assert any("> 8" in e for e in LD.check_proposals(_doc([_prop()] * 9)))
    assert LD.check_proposals({"contract": "x"})[0].startswith("keys")


def test_contract_schema_matches_the_checker():
    sch = json.load(open(os.path.join(HERE, "..", "..", "scripts", "grasp_ladder", "contract.schema.json")))
    assert set(sch["required"]) == {"contract", "id", "feasible", "proposals", "object_note"}
    assert set(sch["properties"]["proposals"]["items"]["required"]) == set(_prop())
    assert sch["properties"]["contract"]["enum"] == [LD.CONTRACT]


def test_agent_job_keeps_the_entry_identity_and_overrides_parse(monkeypatch):
    monkeypatch.delenv("ISG_GLIB_CAND_OVERRIDES", raising=False)
    job = {"category": "cellulose_tape", "model": "kavsnx", "scale": [1.0, 1.0, 1.0], "constraint": "tabletop", "rest_hint_wxyz": None,
           "legacy_key": "cellulose_tape__kavsnx", "id": "cellulose_tape__kavsnx__s1", "cand_overrides": {"families": ["lip_shallow"]}, "retry_status": "ZERO"}
    j = LD.agent_job(job, _doc([_prop(), _prop(expand={"tilt_deg": [0.0], "yaw_deg": [0.0], "slide_mm": [0.0]})]), "/b/proposals/x.json")
    assert S.job_dir(j, "/r") == S.job_dir(job, "/r") and j["id"] == job["id"] and "retry_status" not in j
    assert j["ladder"]["step"] == 3 and j["ladder"]["source"] == "agent" and j["ladder"]["n_proposals"] == 2
    cfg, _ = PL.candidate_overrides(j)
    assert list(cfg.families) == ["agent"] and list(cfg.modes) == [] and len(cfg.agent_proposals) == 2 and cfg.gate_dump
    assert "expand" not in cfg.agent_proposals[0] and cfg.agent_proposals[1]["expand"]["tilt_deg"] == [0.0]
    lj = LD.legacy_job(job, [{"name": "grasp_0", "position": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1.0], "legacy_physics": True}], job["legacy_key"])
    lc, _ = PL.candidate_overrides(lj)
    assert list(lc.families) == ["legacy"] and lc.legacy_grasps[0] == {"name": "grasp_0", "position": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1.0]}
    assert lj["ladder"]["step"] == 4 and S.job_dir(lj, "/r") == S.job_dir(job, "/r")


# ------------------------------------------------------------------------------------------------ the library records the source


def _trial(name):
    return {"candidate": name, "verdict": "valid", "rest_id": "r0", "achieved": {"T_object_eef_at_attach": {"position": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1]}},
            "closure": {"fingers_final_m": [0.01, 0.01]}, "forces": {"finger_slab": {"max": 0.0}, "obj_slab_ratio": {"max": 0.0}}}


def test_build_library_records_agent_source_only_on_given_poses():
    import ispatialgym.grasping.contact_policy as CP

    base = {"T_object_eef": {"position": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1]}, "width_m": 0.01, "preshape_q": 0.015, "mode": "top"}
    cand_doc = {"candidates": [dict(base, name="top_0"), dict(base, name="agent_p1_0", family="agent_p1", source="agent", proposal=1)], "families": {}}
    job = {"category": "c", "model": "m", "scale": [1, 1, 1], "constraint": "tabletop", "legacy_key": "c__m"}
    lib = VT.build_library(job, cand_doc, {}, [_trial("top_0"), _trial("agent_p1_0")], CP.DEFAULT, {})
    g0, g1 = lib["grasps"]
    assert "source" not in g0 and "family" not in g0 and "proposal" not in g0
    assert g1["source"] == "agent" and g1["proposal"] == 1 and g1["family"] == "agent_p1"
    assert lib["source"] == ["agent"] and "ladder" not in lib
    plain = VT.build_library(job, {"candidates": [dict(base, name="top_0")], "families": {}}, {}, [_trial("top_0")], CP.DEFAULT, {})
    assert "source" not in plain and "ladder" not in plain  # the shared path writes the same keys as before
    lad = VT.build_library(dict(job, ladder={"step": 3}), cand_doc, {}, [_trial("agent_p1_0")], CP.DEFAULT, {})
    assert lad["ladder"] == {"step": 3}


# ------------------------------------------------------------------------------------------------ selection, markers, strips, step 4


def _entry(root, eid, n=None, geometry=True):
    d = os.path.join(root, "models", eid)
    os.makedirs(d, exist_ok=True)
    if geometry:
        S.save_json(os.path.join(d, "geometry.json"), {"pieces": []})
    if n is not None:
        S.save_json(os.path.join(d, "library.json"), {"grasps": [{"name": f"g{i}"} for i in range(n)]})
    return d


def test_select_zero_takes_built_entries_valid_nowhere(tmp_path):
    r1, r2 = str(tmp_path / "L2"), str(tmp_path / "L3")
    _entry(r1, "a", 0), _entry(r2, "a", 0)        # zero in both rounds -> selected, src = the later root
    _entry(r1, "b", 0), _entry(r2, "b", 2)        # valid in round 2 -> not selected
    _entry(r1, "c", None)                         # never validated -> not a step-3 entry
    _entry(r1, "e", 0)                            # zero, but excluded
    jobs = [{"id": i} for i in "abcde"]
    sel = LD.select_zero(jobs, [r1, r2], exclude=["e"])
    assert [s["job"]["id"] for s in sel] == ["a"] and sel[0]["src_dir"] == os.path.join(r2, "models", "a")
    assert sel[0]["status"] == {r1: 0, r2: 0}


def test_markers_from_dump_prefer_deeper_gates_and_spread():
    dump = {"passes": [{"pools": {"top": [
        {"gate": "pool", "n_in": 10, "n_out": 10},
        {"gate": "width_antipodal", "n_in": 10, "n_out": 4, "dropped": [{"contact_mm": [0, 40, 0], "why": "antipodal"}]},
        {"gate": "sweep", "n_in": 4, "n_out": 0, "dropped": [
            {"contact_mm": [10, 0, 5], "why": "touch", "touch_mm": 40.0, "limit_mm": 14.0, "approach_k": [0, 0, -1000], "closing_k": [1000, 0, 0], "width_mm": 13.0},
            {"contact_mm": [12, 0, 5], "why": "touch", "touch_mm": 41.0, "limit_mm": 14.0},
            {"contact_mm": [-30, 0, 5], "why": "touch", "touch_mm": 45.0, "limit_mm": 14.0}]}]}}]}
    m = LD.markers_from_dump(dump, None, k=3, sep_mm=15.0)
    assert [x["marker"] for x in m] == ["A", "B", "C"]
    assert [x["gate"] for x in m] == ["sweep", "sweep", "width_antipodal"]  # the 12 mm one is within 15 mm of the 10 mm one
    assert m[0]["contact_mm"] == [10, 0, 5] and m[0]["approach"] == [0, 0, -1.0] and m[0]["closing_axis"] == [1.0, 0, 0]
    assert LD.markers_from_dump(None, None) == []


def test_geometry_summary_and_strip(tmp_path):
    g = _box()
    gs = LD.geometry_summary(g, np.eye(4))
    assert gs["extents_mm"] == [30.0, 50.0, 40.0] and gs["support_z_mm"] == -20.0 and gs["n_pieces"] == 1 and gs["attachment_mates"] == []
    pytest.importorskip("matplotlib")
    p = LD.render_strip(g, [{"marker": "A", "pool": "top", "gate": "sweep", "why": "touch", "contact_mm": [0, 0, 10], "width_mm": 30.0,
                             "approach": [0, 0, -1.0], "closing_axis": [1.0, 0, 0], "measures": {"touch_mm": 40}}], "synthetic box", str(tmp_path / "s.png"))
    assert os.path.getsize(p) > 10000 and open(p, "rb").read(4) == b"\x89PNG"


def test_legacy_grasps_read_the_isaac_yaml(tmp_path):
    os.makedirs(tmp_path / "cands")
    os.makedirs(tmp_path / "results")
    (tmp_path / "cands" / "k__m.yaml").write_text("grasps:\n  grasp_0:\n    position: [0.1, 0.0, 0.02]\n    orientation:\n      w: 0.0\n"
                                                  "      xyz: [-0.7071, 0.7071, 0.0]\n  grasp_1:\n    position: [0.0, 0.0, 0.03]\n    orientation:\n"
                                                  "      w: 1.0\n      xyz: [0.0, 0.0, 0.0]\n")
    S.save_json(str(tmp_path / "results" / "k__m.json"), {"tested": [{"grasp": "grasp_0", "ag_attached": True, "held_after_gravity": True},
                                                                     {"grasp": "grasp_1", "ag_attached": True, "held_after_gravity": False}]})
    gs = LD.legacy_grasps("k__m", str(tmp_path))
    assert [g["name"] for g in gs] == ["grasp_0", "grasp_1"] and gs[0]["quat_xyzw"] == [-0.7071, 0.7071, 0.0, 0.0]
    assert [g["legacy_physics"] for g in gs] == [True, False] and LD.legacy_grasps("none", str(tmp_path)) == []


def test_step4_list_and_owner_gate(tmp_path, monkeypatch):
    import legacy_step as LS

    batch = tmp_path / "batch"
    batch.mkdir()
    jobs = [{"id": i, "legacy_key": k, "alias_key": f"alias_{i}"} for i, k in (("a", "k_a"), ("b", None), ("c", "k_c"), ("d", "k_d"))]
    S.save_json(str(batch / "entries.json"), [{"job": j, "src_dir": str(tmp_path), "status": {}} for j in jobs])
    S.save_json(str(batch / "report.json"), {"rows": [{"id": "a", "proposals": 2, "feasible": True, "n_valid": 0},
                                                      {"id": "b", "proposals": 0, "feasible": False},
                                                      {"id": "c", "proposals": 3, "feasible": True, "n_valid": 4}]})
    monkeypatch.setattr(LD, "legacy_grasps", lambda key, root=None: [{"name": "g0", "legacy_physics": True}] if key == "k_a" else [])
    rows = {r["id"]: r for r in LS.step4_rows(str(batch), legacy_root=str(tmp_path))}
    assert set(rows) == {"a", "b", "d"}  # c got grasps at step 3
    assert rows["a"]["after_step3"] == "0 physics-valid after step 3" and rows["a"]["n_legacy"] == 1 and rows["a"]["n_legacy_held"] == 1
    assert rows["b"]["after_step3"] == "proposer: infeasible" and rows["b"]["n_legacy"] == 0
    assert rows["d"]["after_step3"] == "step 3 not reported"
    with pytest.raises(SystemExit):
        LS.main(["submit", "--batch", str(batch), "--code", "/x", "--out", str(tmp_path / "out")])
    assert not os.path.exists(tmp_path / "out")
