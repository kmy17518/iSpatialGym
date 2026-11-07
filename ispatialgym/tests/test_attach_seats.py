"""i7-attach 2026-11-07: the library's attachment rule reads its mating direction from the oracle's seat axis (owner decision 2026-11-07,
cork ncxgpe). ``CandidateConfig.attach_axis`` "seat" (opt-in; default "meta_z" = as written) takes ``-(R_male_link @ loc)``, ``loc`` = the
oracle's seat axis in the female link frame, recorded per pair in the simulator (``scripts/grasp_bands/attach_seat_facts.py``:
``predicates.seat_axis`` on the loaded pair) into ``assets/grasping/attach_seats.json``. Checked here for EVERY attach object of the
campaign libraries: the rule's axis equals the oracle's seat axis, the table equals what ``predicates.seat_axis`` computes today, and
"meta_z" keeps the frames as written. CPU only, no simulator, no Behavior-1K asset."""

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from ispatialgym.grasping import candidates as CA  # noqa: E402
from ispatialgym.grasping import object_geometry as OG  # noqa: E402
from ispatialgym.grasping import pipeline as PL  # noqa: E402
from ispatialgym.grasping import schema as S  # noqa: E402

TABLE = S.load_json(CA.ATTACH_SEATS_FILE) or {}
PAIRS = TABLE.get("pairs", [])
needs_table = pytest.mark.skipif(not PAIRS, reason="assets/grasping/attach_seats.json missing")


def _meta_frames_as_written(info, R):
    """``candidates.attachment_frames`` before i7-attach (a4620a1e), verbatim."""
    out = []
    for link, ml in (info.get("meta_links") or {}).items():
        if not isinstance(ml, dict):
            continue
        for k, v in ml.get("attachment", {}).items():
            for e in (v.values() if isinstance(v, dict) else v):
                if isinstance(e, dict) and "position" in e:
                    q = e.get("orientation") or [0.0, 0.0, 0.0, 1.0]
                    Rm = S.quat_to_mat(q)
                    out.append((R[:3, :3] @ np.asarray(e["position"], dtype=float), R[:3, :3] @ (Rm @ np.array([0.0, 0.0, -1.0]))))
    if not out:
        lo, hi = np.array(info["coll_bbox_min"]), np.array(info["coll_bbox_max"])
        c = (lo + hi) / 2
        c[2] = lo[2]
        out.append((c, np.array([0.0, 0.0, -1.0])))
    return out


def _info(rec):
    m = rec["male"]
    return {"meta_links": json.loads(json.dumps(m.get("meta_links") or {})), "coll_bbox_min": m.get("coll_bbox_min", [-0.01] * 3),
            "coll_bbox_max": m.get("coll_bbox_max", [0.01] * 3)}


def _rot(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    R = np.eye(4)
    R[:3, :3] = S.quat_to_mat(q / np.linalg.norm(q))
    return R


# ------------------------------------------------------------------------------------------------ the switch


def test_attach_axis_defaults_to_meta_z_and_opts_in_per_entry(monkeypatch):
    monkeypatch.delenv("ISG_GLIB_CAND_OVERRIDES", raising=False)
    assert CA.CandidateConfig().attach_axis == "meta_z"
    cork = {"id": "cork__ncxgpe__s1__attachment_free", "cand_overrides": {"attach_axis": "seat"}}
    assert PL.candidate_overrides(cork)[0].attach_axis == "seat"
    assert PL.candidate_overrides({"id": "cap__xxqyrt__s1__attachment_free"})[0].attach_axis == "meta_z"


# ------------------------------------------------------------------------------------------------ every attach object of the libraries


@needs_table
def test_the_table_covers_every_attach_entry_of_the_libraries():
    """Every attach entry of the campaign libraries (roles_all_20261102: 33 jobs + 1 derived rest; v2maps_i7 / v2maps_20261106: the
    two lids) has a record, found by (category, model, scale) as the generator looks it up."""
    ids = TABLE["library_entries"]
    assert len(ids) == 34
    for eid in ids:
        cat, model, sc = eid.split("__")[:3]
        scale = [float(v) for v in sc[1:].split("x")] if "x" in sc else float(sc[1:])
        assert CA.attach_seat_record({"category": cat, "model": model, "scale": scale}) is not None, eid


@needs_table
def test_meta_z_keeps_the_frames_as_written_for_every_attach_entry():
    for rec in PAIRS:
        info = _info(rec)
        for R in (np.eye(4), _rot(len(rec["id"]))):
            old = _meta_frames_as_written(info, R)
            new = CA.attachment_frames(info, R)
            assert len(old) == len(new), rec["id"]
            for (p0, d0), (p1, d1) in zip(old, new):
                assert np.array_equal(p0, p1) and np.array_equal(d0, d1), rec["id"]  # bit for bit


def _oracle_dirs(rec):
    """The oracle's mating direction per matched female link, in the male root frame, from the recorded LIVE values: the partner at the
    identity, the male root placed so that its male link frame is the female link frame (``AttachedTo``'s alignment), the recorded world
    seat axis (``axis_root``) expressed in that male root frame, negated (the male moves into the seat)."""
    out = []
    links = rec["male"].get("links") or {}
    for prt in rec.get("partners") or []:
        for f in prt.get("female") or []:
            ml = links[f["child_link"]]
            R_ml = S.quat_to_mat(ml["orientation_root_xyzw"])
            R_fl = S.quat_to_mat(f["orientation_root_xyzw"])
            R_root = R_fl @ R_ml.T  # the seated male root in the partner frame
            d = -(R_root.T @ np.asarray(f["axis_root"], dtype=float))
            out.append((ml["meta_link_id"], d / np.linalg.norm(d), f"{prt['category']}/{prt['model']}:{f['parent_link']}"))
    return out


@needs_table
def test_the_rule_axis_equals_the_oracle_seat_axis_for_every_attach_object():
    n_seat = 0
    for rec in PAIRS:
        info = _info(rec)
        src = []
        mates = CA.attachment_frames(info, np.eye(4), seat=rec, sources=src)
        oracle = _oracle_dirs(rec)
        if not oracle:  # no male meta link or no partner with a matching female link: the rule as written (nothing to seat)
            assert all(s["source"] in ("meta_z", "bottom_centre") for s in src), rec["id"]
            old = _meta_frames_as_written(info, np.eye(4))
            assert all(np.array_equal(d0, d1) for (_, d0), (_, d1) in zip(old, mates)), rec["id"]
            continue
        dirs = [m for (_, m), s in zip(mates, src) if s["source"] == "seat"]  # unrounded (``sources`` rounds to 1e-6)
        for _, d, who in oracle:
            n_seat += 1
            assert any(float(np.dot(d, x)) > math.cos(math.radians(0.01)) for x in dirs), (rec["id"], who, d.tolist(), [x.tolist() for x in dirs])
        live = {ml["meta_link_id"]: ml for ml in (rec["male"].get("links") or {}).values()}
        for (p, m), s in zip(mates, src):  # every frame the rule uses is one of the oracle's (or a meta link without a partner female)
            if s["source"] == "seat":
                assert any(float(np.dot(m, d)) > math.cos(math.radians(0.01)) for _, d, _ in oracle), rec["id"]
                # the point: the male link where the simulator puts it (the applied scale included), not the metadata position
                assert np.allclose(p, live[s["meta_link"].split("/")[1]]["position_root"], atol=1e-9), rec["id"]
    assert n_seat > 0


@needs_table
def test_the_table_is_what_predicates_seat_axis_computes_today(monkeypatch):
    """Each recorded axis is reproduced by ``predicates.seat_axis`` itself on stubs of the recorded facts (the female link pose and the
    boundary-point centroid in the partner root frame, the partner at the identity): regenerate the table when the rule changes."""
    th = pytest.importorskip("torch")
    from omnigibson.object_states import AttachedTo

    from ispatialgym.sampling import predicates as PRED

    n = 0
    for rec in PAIRS:
        for prt in rec.get("partners") or []:
            for f in prt.get("female") or []:
                monkeypatch.setattr(PRED, "_SEAT_AXIS_LOCAL", {})
                cen = th.tensor([f["centroid_root"]], dtype=th.float32)
                monkeypatch.setattr(PRED, "_boundary_points", lambda obj, cen=cen: cen)
                link = SimpleNamespace(get_position_orientation=lambda f=f: (th.tensor(f["position_root"]), th.tensor(f["orientation_root_xyzw"])))
                partner = SimpleNamespace(name="p", states={AttachedTo: SimpleNamespace(links={f["parent_link"]: link})})
                a = PRED.seat_axis(SimpleNamespace(name="m"), partner, f["parent_link"])
                assert a.tolist() == pytest.approx(f["axis_root"], abs=2e-4), (rec["id"], prt["model"])
                n += 1
    assert n > 0


@needs_table
def test_the_cork_ncxgpe_seats_along_its_own_minus_z_and_the_frame_rule_points_sideways():
    rec = CA.attach_seat_record({"category": "cork", "model": "ncxgpe", "scale": [1, 1, 1]})
    info = _info(rec)
    (_, d_old), = CA.attachment_frames(info, np.eye(4))
    (_, d_new), = CA.attachment_frames(info, np.eye(4), seat=rec)
    assert abs(d_old[2]) < 1e-3 and abs(d_old[1]) > 0.999  # the meta link frame (0.5, -0.5, 0.5, 0.5): -z -> +y
    assert d_new[2] < -0.999  # the oracle: down the bottle neck, along the cork's -z


# ------------------------------------------------------------------------------------------------ generation on a synthetic cork


def _cork(r=0.0088, z0=-0.0183, z1=0.0400):
    """A cork-like cylinder (ncxgpe's 8.8 mm radius; taller, so a top pinch's fingertips stay above its mating face - the bottom - by
    more than the rule's 2 mm, and only the axis decides), its attachment meta link at the bottom in ncxgpe's frame."""
    piece = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=32)
    piece.apply_translation([0.0, 0.0, 0.5 * (z0 + z1)])
    q = [0.5, -0.5, 0.5, 0.5]
    info = {"meta_links": {"base_link": {"attachment": {"corkbottleM": [{"position": [0.0, 0.0, z0], "orientation": q}]}}}}
    return OG.ObjectGeometry(info=info, pieces=[piece], union=trimesh.util.concatenate([piece])), q


def _pinch(theta, z):
    """A top-down pinch with its pad contact at (0, 0, z), closing axis at azimuth ``theta``."""
    zax = np.array([0.0, 0.0, -1.0])
    y = np.array([math.cos(theta), math.sin(theta), 0.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = np.cross(y, zax), y, zax, np.array([0.0, 0.0, z]) - zax * CA.CONTACT_Z
    return T


def test_generate_on_a_cork_meta_z_rejects_every_pinch_and_seat_keeps_them(tmp_path, monkeypatch):
    g, q = _cork()
    monkeypatch.setattr(OG, "load", lambda model_dir: g)
    T_stub = np.stack([_pinch(a, z) for a in np.linspace(0, math.pi, 6, endpoint=False) for z in (0.026, 0.030, 0.034)])

    class _Sampler:
        @staticmethod
        def sample_antipodal(*a, **k):
            return list(T_stub)

    monkeypatch.setattr(CA, "_isaac_sampler", lambda: _Sampler)
    loc = (S.quat_to_mat(q).T @ np.array([0.0, 0.0, 1.0])).tolist()  # the seat axis +z of the cork when seated, in the female link frame
    seats = {"schema": "isg-attach-seats/1", "pairs": [{"id": "cork__synthetic__s1__attachment_free",
             "male": {"category": "cork", "model": "synthetic", "scale": [1, 1, 1], "links": {"m0": {"meta_link_id": "corkbottleM", "sub_id": "0"}}},
             "partners": [{"category": "wine_bottle", "model": "x", "female": [{"child_link": "m0", "parent_link": "f0", "loc": loc}]}]}]}
    f = tmp_path / "seats.json"
    f.write_text(json.dumps(seats))
    monkeypatch.setattr(CA, "ATTACH_SEATS_FILE", str(f))
    job = {"category": "cork", "model": "synthetic", "scale": [1, 1, 1], "constraint": "attachment_free", "rest_hint_wxyz": None,
           "id": "cork__synthetic__s1__attachment_free"}
    from dataclasses import asdict

    outs = {}
    for mode in ("meta_z", "seat"):
        d = tmp_path / mode
        d.mkdir()
        cfg = CA.CandidateConfig(**{**asdict(CA.CandidateConfig()), "attach_axis": mode, "gate_dump": True, "modes": ("top",)})
        outs[mode] = (CA.generate(job, str(d), cfg=cfg, densify=()), json.load(open(d / CA.GATE_DUMP_FILE)))
    (o0, d0), (o1, _) = outs["meta_z"], outs["seat"]
    n_in = o0["log"]["top"]["n_lever"]
    assert n_in > 0 and o1["log"]["top"]["n_lever"] == n_in  # the gates before the rule are the same
    assert o0["log"]["top"]["n_attachment_free"] == 0 and o1["log"]["top"]["n_attachment_free"] == n_in
    assert o0["log"]["attach_axis"]["mates"][0]["source"] == "meta_z" and o1["log"]["attach_axis"]["mates"][0]["source"] == "seat"
    g0 = [st for st in d0["passes"][-1]["pools"]["top"] if st["gate"] == "attachment_free"][0]
    assert g0["n_out"] == 0 and set(g0["dropped_by"]) <= {"side", "face_gap", "side+face_gap"} and "face_gap_mm" in g0["stats"]
    assert g0["records"][0]["mates"][0]["dir"][1] == pytest.approx(1.0, abs=1e-6)  # the frame rule's +y
    # the seat record of another model is never used
    assert CA.attach_seat_record({"category": "cork", "model": "other", "scale": 1}) is None


# ------------------------------------------------------------------------------------------------ the axis in the entry identity


def _attach_lib_data(**extra):
    rest = {"schema": S.REST, "rest_id": "r0", "quat_xyzw": [0.0, 0.0, 0.0, 1.0], "h_root": 0.0183, "reference_yaw": 0.0, "stable": True}
    d = {"schema": S.LIBRARY, "category": "cork", "model": "synthetic", "scale": [1.0, 1.0, 1.0], "dataset": "behavior-1k-assets",
         "constraint": "attachment_free", "policy_key": "pk0", "rests": {"r0": rest}, "families": {"f0": {"type": "none", "representative": "top_0"}},
         "grasps": [{"name": "top_0", "rest_id": "r0", "family_id": "f0", "T_object_eef": {"position": [0.0, 0.0, 0.1], "orientation_xyzw": [1.0, 0.0, 0.0, 0.0]},
                     "finger_q": [0.01, 0.01], "width_m": 0.0176}]}
    d.update(extra)
    return d


def _lib(data, path="x"):
    return S.ModelLibrary(path=path, data=data, rests={k: S.Rest.from_record(v) for k, v in data["rests"].items()})


def _write(root, data, cand_config=None):
    d = S.model_dir(data["category"], data["model"], data["scale"], str(root), constraint=data.get("constraint", "tabletop"))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "library.json"), "w") as f:
        json.dump(data, f)
    if cand_config is not None:
        with open(os.path.join(d, "candidates.json"), "w") as f:
            json.dump({"schema": S.CANDIDATES, "config": cand_config, "candidates": [], "representatives": []}, f)
    return d


def test_the_attach_axis_is_in_the_identity_only_off_the_legacy_axis(tmp_path):
    """Every existing key stays: an attach entry generated before i7-attach (no field: "meta_z"), an explicit "meta_z" record and any
    non-attach entry (the rule never runs; its config's attach_axis is ignored) keep the key computed without the axis. A "seat" entry -
    the record or, validated before the record, its candidates.json config - gets its own key; check_compatible can require an axis."""
    base = _lib(_attach_lib_data())
    key0 = base.identity_key()
    assert base.attach_axis() == "meta_z" and "attach_axis" not in base.identity()
    assert _lib(_attach_lib_data(attach_axis="meta_z")).identity_key() == key0
    seat = _lib(_attach_lib_data(attach_axis="seat"))
    assert seat.attach_axis() == "seat" and seat.identity()["attach_axis"] == "seat" and seat.identity_key() != key0
    tab = _attach_lib_data(constraint="tabletop", attach_axis="seat")
    assert _lib(tab).attach_axis() is None and "attach_axis" not in _lib(tab).identity()
    for i, (cfg, want) in enumerate(((None, "meta_z"), ({"n_candidates": 6000}, "meta_z"), ({"attach_axis": "seat"}, "seat"))):
        root = tmp_path / f"r{i}"
        _write(root, _attach_lib_data(), cfg)
        got = S.load_model("cork", "synthetic", [1, 1, 1], str(root), constraint="attachment_free")
        assert got.attach_axis() == want and (got.identity_key() == key0) == (want == "meta_z")
    S.check_compatible(seat, attach_axis="seat")
    with pytest.raises(S.IncompatibleRecord):
        S.check_compatible(base, attach_axis="seat")


def test_build_library_records_the_attach_axis_for_attach_entries_only():
    from ispatialgym.grasping import contact_policy as CP
    from ispatialgym.grasping import validate_tabletop as VT

    cand = {"pipeline": "a1v2", "constraint": "attachment_free", "config": {"attach_axis": "seat"}, "candidates": [], "representatives": [],
            "families": {}, "arm": "right"}
    job = {"category": "cork", "model": "synthetic", "scale": [1, 1, 1], "constraint": "attachment_free"}
    rec = VT.build_library(job, cand, {}, [], CP.DEFAULT, {})
    assert rec["attach_axis"] == "seat"
    rec = VT.build_library(job, dict(cand, config={}), {}, [], CP.DEFAULT, {})
    assert rec["attach_axis"] == "meta_z"
    rec = VT.build_library(dict(job, constraint="tabletop"), dict(cand, constraint="tabletop"), {}, [], CP.DEFAULT, {})
    assert "attach_axis" not in rec
