"""i7-attachdef 2026-11-08 (owner decision 5.1, an approved gate change): ``CandidateConfig.outer_pinch_preshape`` - with outer_pinch, the
pinch the pads close on is judged at the oracle's pre-shape opening (``preshape_pad_half`` = ``oracle_preshape_q``: width / 2 + the
pre-shape margin), not from the fully open pads (+-44.45 mm)."""

from __future__ import annotations

import math

import numpy as np


def _cup(r_out=0.026, wall=0.003, h=0.011, top=0.0023, n=24):
    """The attachfix cap: a top disc + n convex wall segments, open side down."""
    import trimesh

    pieces = [trimesh.creation.cylinder(radius=r_out, height=top, sections=48)]
    pieces[0].apply_translation([0, 0, h - top / 2])
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        pts = [[r * math.cos(a), r * math.sin(a), z] for a in (a0, a1) for r in (r_out - wall, r_out) for z in (0.0, h - top)]
        pieces.append(trimesh.convex.convex_hull(np.array(pts)))
    return pieces, trimesh.util.concatenate(pieces)


def _planes(pieces):
    return [(p.face_normals, np.einsum("ij,ij->i", p.face_normals, p.triangles[:, 0])) for p in pieces]


def _pinch(c, closing, approach=(0, 0, 1.0)):
    from ispatialgym.grasping import candidates as CA

    R = CA.proposal_frame(np.array(approach, float), np.array(closing, float))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(c, float) - R[:, 2] * CA.CONTACT_Z
    return T[None]


def test_the_cap_keeps_its_diameter_pinch():
    from ispatialgym.grasping import candidates as CA

    pieces, u = _cup()
    T = _pinch([0, 0, 0.004], [1.0, 0, 0])  # pad centre in the cavity
    _, w_full, _, _ = CA.contact_features(u, T, CA.CONTACT_Z, outer_half=0.0889 / 2)
    _, w_pre, ant, _ = CA.contact_features(u, T, CA.CONTACT_Z, outer_half=CA.PreshapeOuter(pieces))
    assert abs(w_pre[0] - 0.052) < 0.0015 and abs(w_pre[0] - w_full[0]) < 1e-6 and ant[0] > 0.9


def test_a_wall_pinch_next_to_a_far_wall_comes_back():
    """A 44 mm glass: the pad centre in one 3 mm wall, closing radially. From the fully open pads (+-44.45 mm) the inward pad starts
    beyond the far wall (39.5-42.5 mm away) and the measure spans the glass; the pre-shaped pads (1.5 + 8.5 mm) meet only the wall."""
    from ispatialgym.grasping import candidates as CA

    pieces, u = _cup(r_out=0.022, wall=0.003, h=0.04)
    c = [0.0205 * math.cos(math.pi / 24), 0.0205 * math.sin(math.pi / 24), 0.02]  # mid-wall, mid-segment
    closing = [math.cos(math.pi / 24), math.sin(math.pi / 24), 0.0]
    T = _pinch(c, closing, approach=(0, 0, -1.0))
    _, w_full, _, _ = CA.contact_features(u, T, CA.CONTACT_Z, outer_half=0.0889 / 2)
    _, w_pre, ant, _ = CA.contact_features(u, T, CA.CONTACT_Z, outer_half=CA.PreshapeOuter(pieces))
    _, w_in, _, _ = CA.contact_features(u, T, CA.CONTACT_Z)
    assert w_full[0] > 0.02  # the fully-open measure spans the cavity (the thin pinch lost)
    assert abs(w_pre[0] - 0.003) < 0.0005 and abs(w_pre[0] - w_in[0]) < 1e-4 and ant[0] > 0.9  # the wall, as the pad-centre measure
    # the pre-shape the width implies is the oracle's (one definition)
    h = CA.preshape_pad_half(w_pre)[0]
    assert abs(h - (CA.oracle_preshape_q(w_pre)[0] - (CA.FINGER_OPEN - CA.PAD_OPEN_HALF))) < 1e-12


def test_a_convex_solid_is_unchanged_and_a_pinch_too_wide_for_the_preshape_is_dropped():
    import trimesh

    from ispatialgym.grasping import candidates as CA

    box = trimesh.creation.box(extents=[0.03, 0.05, 0.02])
    T = _pinch([0, 0, 0], [1.0, 0, 0])
    w0 = CA.contact_features(box, T, CA.CONTACT_Z)[1][0]
    w1 = CA.contact_features(box, T, CA.CONTACT_Z, outer_half=CA.PreshapeOuter([box]))[1][0]
    assert abs(w0 - 0.03) < 1e-6 and abs(w1 - w0) < 1e-6
    wide = trimesh.creation.box(extents=[0.12, 0.05, 0.02])  # wider than any pre-shape: no pinch holds
    assert not np.isfinite(CA.contact_features(wide, T, CA.CONTACT_Z, outer_half=CA.PreshapeOuter([wide]))[1][0])


def test_a_non_convex_single_piece_is_judged_by_containment():
    """hardback bqvvvx: one non-convex piece (a whole-book hull) - its face planes do not bound its inside; the pinch across the book is
    found by containment (before: no outer boundary, every candidate dropped)."""
    import trimesh

    from ispatialgym.grasping import candidates as CA

    from shapely.geometry import Polygon

    notch = trimesh.creation.extrude_polygon(Polygon(
        [(-0.03, -0.02), (0.05, -0.02), (0.05, 0.02), (0.03, 0.02), (0.03, 0.0), (-0.03, 0.0)]), 0.03)
    notch.apply_translation([0, 0, -0.015])
    assert not notch.is_convex and notch.is_watertight
    T = _pinch([0.0, -0.01, 0.0], [0, 0, 1.0], approach=(1.0, 0, 0))
    w = CA.contact_features(notch, T, CA.CONTACT_Z, outer_half=CA.PreshapeOuter([notch]))[1][0]
    assert abs(w - 0.03) < 1e-4


def test_the_switch_is_the_default_and_enters_the_identity_only_when_on():
    from dataclasses import asdict

    from ispatialgym.grasping import candidates as CA
    from ispatialgym.grasping import schema as S

    assert "outer_pinch_preshape" in S.CANDIDATE_SWITCHES
    assert S.candidate_switches({})["outer_pinch_preshape"] is False
    assert CA.CandidateConfig().outer_pinch_preshape is True  # owner 5.1: the default for new builds (its own commit)
    assert "outer_pinch_preshape" in asdict(CA.CandidateConfig())


def test_with_the_switch_the_first_hit_proposals_join_the_all_hit_ones(tmp_path, monkeypatch):
    """With every hit the Isaac sampler pairs each point with the FURTHEST hit (no thin pinch is proposed): the switch adds the first-hit
    proposals (same seed) and judges every proposal at the pre-shape. Without the switch one sampling, as before."""
    from dataclasses import asdict

    from ispatialgym.grasping import object_geometry as OG
    from ispatialgym.tests.test_glib_padsweep import _ring, _wall_pinch

    from ispatialgym.grasping import candidates as CA

    g = _ring()
    monkeypatch.setattr(OG, "load", lambda model_dir: g)
    calls = []
    T_stub = np.stack([_wall_pinch(z, a) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False) for z in (0.0, 0.006)])

    class _Sampler:
        @staticmethod
        def sample_antipodal(mesh, *a, **k):
            calls.append(bool(mesh.ray.all_hits))
            return list(T_stub)

    monkeypatch.setattr(CA, "_isaac_sampler", lambda: _Sampler)
    seen = []
    real_geo = CA.CP.check_grasp_geometry

    def _geo(*a, **k):  # the opening the policy check judges each candidate at
        seen.append(k.get("open_q"))
        return real_geo(*a, **k)

    monkeypatch.setattr(CA.CP, "check_grasp_geometry", _geo)
    job = {"category": "tape", "model": "ring", "scale": [1, 1, 1], "constraint": "tabletop", "rest_hint_wxyz": None, "id": "tape__ring__s1"}
    for pre, want in ((False, [True]), (True, [True, False])):
        calls.clear()
        seen.clear()
        d = tmp_path / str(pre)
        d.mkdir()
        cfg = CA.CandidateConfig(**{**asdict(CA.CandidateConfig()), "outer_pinch": True, "outer_pinch_preshape": pre, "topk": 4,
                                    "cone_deg": 60.0})
        out = CA.generate(job, str(d), cfg=cfg, densify=())
        assert calls == want and out["log"]["n_sampled"] == len(want) * len(T_stub)
        # with the switch one opening per candidate (its own pre-shape), without it the batch's widest (a scalar)
        assert all((np.ndim(q) == 1) if pre else isinstance(q, float) for q in seen)
        if pre:  # the ring's one-wall pinches pass the width gate only judged at the pre-shape (the fully-open measure spans the core)
            assert seen and out["log"]["top"].get("n_selected", 0) > 0
