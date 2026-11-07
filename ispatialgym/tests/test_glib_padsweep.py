"""i7-padsweep 2026-11-07 (owner decision: the closing sweep starts at the oracle's pre-shape, a shared gate): ``CandidateConfig.
sweep_from_preshape`` - the oracle's pre-shape rule reused, the sweep from it on a tape-like ring (a wall pinch next to the core hole),
the old sweep unchanged when off, and the switch in the entry identity (only when on: every existing key, incl. the plate switches' pair,
unchanged). CPU only, synthetic geometry."""

import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from ispatialgym.grasping import candidates as CA  # noqa: E402
from ispatialgym.grasping import object_geometry as OG  # noqa: E402
from ispatialgym.grasping import schema as S  # noqa: E402

R_IN, R_OUT, H = 0.0175, 0.031, 0.022  # tape kavsnx: the 35 mm core hole, a 13.5 mm wall, 22 mm tall


def _ring(n=24) -> OG.ObjectGeometry:
    pieces = []
    span = 2 * math.pi / n
    for k in range(n):
        pts = [(r * math.cos(a), r * math.sin(a), z) for a in np.linspace(k * span, (k + 1) * span, 4) for r in (R_IN, R_OUT) for z in (-H / 2, H / 2)]
        pieces.append(trimesh.convex.convex_hull(np.array(pts)))
    return OG.ObjectGeometry(info={"meta_links": {}}, pieces=pieces, union=trimesh.util.concatenate(pieces))


def _wall_pinch(z=0.0, theta=0.0):
    """Approach straight down, closing radial, the pad contact centre mid-wall at height ``z``."""
    rad = np.array([math.cos(theta), math.sin(theta), 0.0])
    zax = np.array([0, 0, -1.0])
    R = np.stack([np.cross(rad, zax), rad, zax], axis=1)  # eef x = y cross z
    c = 0.5 * (R_IN + R_OUT) * rad + np.array([0, 0, z])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, c - R[:, 2] * CA.CONTACT_Z
    return T


def _planes(g):
    return [(pc.face_normals, np.einsum("ij,ij->i", pc.face_normals, pc.triangles[:, 0])) for pc in g.pieces]


def test_default_off_and_the_oracle_preshape_rule_is_reused():
    from dataclasses import asdict

    from ispatialgym.oracle.motions import MotionsConfig

    assert CA.CandidateConfig().sweep_from_preshape is False
    m = MotionsConfig().gripper_preshape_margin
    w = np.array([0.0, 0.0135, 0.04, 0.09])
    # execute.execute_grasp: opening = 0.5 * grasp_pinch_width + motions.cfg.gripper_preshape_margin; the validator caps it at fully open
    assert np.allclose(CA.oracle_preshape_q(w), np.minimum(0.05, 0.5 * w + m)) and CA.PRESHAPE_MARGIN == m
    # the sweep's start in its own open-pad convention: PAD_OPEN_HALF at the fully open fingers (the old start), less what the pre-shape closed
    assert math.isclose(float(CA.preshape_pad_half(np.array([0.09]))[0]), CA.PAD_OPEN_HALF)
    assert math.isclose(float(CA.preshape_pad_half(np.array([0.0135]))[0]), CA.PAD_OPEN_HALF - (0.05 - (0.00675 + m)))
    assert math.isclose(float(CA.preshape_pad_half(np.array([np.inf]))[0]), CA.PAD_OPEN_HALF)  # no width: fully open
    assert "sweep_from_preshape" in asdict(CA.CandidateConfig())


def test_the_sweep_from_the_preshape_passes_a_wall_pinch_next_to_the_core_hole():
    g = _ring()
    T = np.stack([_wall_pinch(0.0, a) for a in (0.0, 0.13, 1.0)])
    _, width, _, _ = CA.contact_features(g.union, T, CA.CONTACT_Z)
    assert np.allclose(width, R_OUT - R_IN, atol=0.002)
    limit = width / 2 + CA.CandidateConfig().sweep_margin_m
    old = CA.pad_sweep(g.union, T)
    assert (old > limit).all() and (old > 0.035).all()  # the fully open inner pad starts inside the opposite wall (tape: touch 41-48.5 mm)
    new = CA.pad_sweep(g.union, T, open_half=CA.preshape_pad_half(width), pieces=_planes(g))
    assert (new <= limit).all() and (new > width / 2 - 0.002).all()  # the pads meet the wall at the pinch
    # the new form at the fully open start reproduces the old sweep exactly
    assert np.array_equal(CA.pad_sweep(g.union, T, open_half=np.full(len(T), CA.PAD_OPEN_HALF)), old)


def test_a_pad_that_starts_inside_a_piece_meets_the_object_at_its_start():
    g = _ring()
    T = _wall_pinch(0.0)[None]
    h0 = np.array([0.045])  # the inner pad starts at r -20.75 mm: inside the far wall (17.5-31 mm)
    t = CA.pad_sweep(g.union, T, open_half=h0, pieces=_planes(g))
    assert math.isclose(float(t[0]), 0.045)
    t_no = CA.pad_sweep(g.union, T, open_half=h0)  # without the pieces the ray from inside sees only the far wall's exit face
    assert float(t_no[0]) < 0.045


def test_generate_with_the_switch_passes_the_ring_top_grasps(tmp_path, monkeypatch):
    g = _ring()
    monkeypatch.setattr(OG, "load", lambda model_dir: g)
    T_stub = np.stack([_wall_pinch(z, a) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False) for z in (0.0, 0.006)])

    class _Sampler:
        @staticmethod
        def sample_antipodal(*a, **k):
            return list(T_stub)

    monkeypatch.setattr(CA, "_isaac_sampler", lambda: _Sampler)
    from dataclasses import asdict

    job = {"category": "tape", "model": "ring", "scale": [1, 1, 1], "constraint": "tabletop", "rest_hint_wxyz": None, "id": "tape__ring__s1"}
    logs = {}
    for on in (False, True):
        d = tmp_path / str(on)
        d.mkdir()
        cfg = CA.CandidateConfig(**{**asdict(CA.CandidateConfig()), "sweep_from_preshape": on, "topk": 4, "cone_deg": 60.0})
        out = CA.generate(job, str(d), cfg=cfg, densify=())
        logs[on] = out["log"]["top"]
        assert out["config"]["sweep_from_preshape"] is on and S.candidate_switches(out["config"])["sweep_from_preshape"] is on
    assert logs[False].get("n_object_clear", 0) > 0 and logs[False]["n_sweep"] == 0  # as the tape: every top grasp dies at the sweep
    assert logs[True]["n_object_clear"] == logs[False]["n_object_clear"] and logs[True]["n_sweep"] > 0
    assert {k: v for k, v in logs[True].items() if k in ("n_tilt", "n_width_antipodal", "n_support")} == \
        {k: v for k, v in logs[False].items() if k in ("n_tilt", "n_width_antipodal", "n_support")}  # the gates before the sweep unchanged


def _lib(**extra):
    rest = {"schema": S.REST, "rest_id": "r0", "quat_xyzw": [0.0, 0.0, 0.0, 1.0], "h_root": 0.0123, "reference_yaw": 0.0, "stable": True}
    d = {"schema": S.LIBRARY, "category": "plate", "model": "synthetic", "scale": [1.0, 1.0, 1.0], "dataset": "behavior-1k-assets", "policy_key": "pk0",
         "rests": {"r0": rest}, "families": {"f0": {"type": "none", "representative": "top_0"}},
         "grasps": [{"name": "top_0", "rest_id": "r0", "family_id": "f0", "T_object_eef": {"position": [0.0, 0.0, 0.1], "orientation_xyzw": [1.0, 0.0, 0.0, 0.0]},
                     "finger_q": [0.01, 0.01], "width_m": 0.02}]}
    d.update(extra)
    return S.ModelLibrary(path="x", data=d, rests={k: S.Rest.from_record(v) for k, v in d["rests"].items()})


def test_the_switch_enters_the_identity_only_when_on():
    golden = "cd78dd8e53bcfdbf"  # test_glib_switches.GOLDEN_KEY: the key before any switch existed
    assert _lib().identity_key() == golden
    assert _lib(candidate_switches={"sweep_from_preshape": False}).identity_key() == golden
    # the plate switches' pair as 884279fd wrote it (an entry rebuilt with them: v2maps_i7sw) - no third key, so its key is unchanged
    pair = _lib(candidate_switches={"blade_contact_at_retreat": True, "retreat_shift_exact": True, "sweep_from_preshape": False})
    assert pair.identity()["candidate_switches"] == {"blade_contact_at_retreat": True, "retreat_shift_exact": True}
    assert pair.identity_key() == _lib(candidate_switches={"blade_contact_at_retreat": True, "retreat_shift_exact": True}).identity_key()
    blade = _lib(candidate_switches={"blade_contact_at_retreat": True})
    assert blade.identity()["candidate_switches"] == {"blade_contact_at_retreat": True, "retreat_shift_exact": False}
    on = _lib(candidate_switches={"blade_contact_at_retreat": True, "retreat_shift_exact": True, "sweep_from_preshape": True})
    assert on.identity()["candidate_switches"] == {"blade_contact_at_retreat": True, "retreat_shift_exact": True, "sweep_from_preshape": True}
    only = _lib(candidate_switches={"sweep_from_preshape": True})
    assert only.identity()["candidate_switches"] == {"sweep_from_preshape": True}
    keys = {x.identity_key() for x in (_lib(), pair, blade, on, only)}
    assert len(keys) == 5
    S.check_compatible(on, candidate_switches={"sweep_from_preshape": True})
    with pytest.raises(S.IncompatibleRecord):
        S.check_compatible(pair, candidate_switches={"sweep_from_preshape": True})
