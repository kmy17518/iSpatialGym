"""of-v2maps 2026-11-07: the settle-first survey's clustering (ispatialgym.grasping.settle_first)."""

import math

import numpy as np

from ispatialgym.grasping import schema as S
from ispatialgym.grasping import settle_first as SF


def _cylinder(r=0.03, length=0.20, n=48, wobble=0.0):
    """Vertices of a cylinder along the object x axis; ``wobble`` makes the cross-section not quite round (a coarse hull)."""
    pts = []
    for x in np.linspace(-length / 2, length / 2, 9):
        for k in range(n):
            t = 2 * math.pi * k / n
            rr = r * (1.0 + wobble * math.cos(6 * t))
            pts.append([x, rr * math.cos(t), rr * math.sin(t)])
    return np.array(pts)


def _q_axis(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.array([*(a * math.sin(ang / 2)), math.cos(ang / 2)])


def _outcome(q, h, stable=True, yaw0=0.0, drop=0.003):
    return {"quat_xyzw": list(q), "h_root": h, "stable": stable, "yaw0": yaw0, "drop_m": drop, "tilt_from_intended_rad": 0.0}


def test_yaw_free_removes_the_yaw_and_keeps_the_tilt():
    q = S.quat_mul(S.quat_yaw(1.1), _q_axis([1, 0, 0], 0.3))
    f = SF.yaw_free(q)
    assert abs(S.yaw_of_quat(f)) < 1e-6 and abs(S.tilt_distance(f, q)) < 1e-6


def test_rolls_of_a_near_round_cylinder_are_one_rest_and_standing_up_is_another():
    V = _cylinder(wobble=0.06)  # not quite round: exact symmetry would split it
    lying = [_outcome(_q_axis([1, 0, 0], a), 0.0300 + 0.0005 * k, yaw0=0.5 * k) for k, a in enumerate((0.0, 0.9, 2.1, 3.0))]
    up = [_outcome(_q_axis([0, 1, 0], -math.pi / 2), 0.100, drop=0.05), _outcome(_q_axis([0, 1, 0], -math.pi / 2), 0.1004, drop=0.02)]
    cl = SF.cluster_rests(lying + up, V, designated_xyzw=[0, 0, 0, 1])
    assert [c["n"] for c in cl] == [4, 2] and [c["rest_id"] for c in cl] == ["s0", "s1"]
    assert len(cl[0]["roll_merged"]) >= 1 and cl[0]["roll_merged"][0]["axis_index"] == 2  # merged by the roll about the long (x) axis
    assert cl[0]["stability"] == 1.0 and cl[0]["tilt_from_designated_rad"] < 0.5 and abs(cl[1]["h_root"] - 0.1002) < 1e-6
    assert cl[0]["method"].startswith(SF.METHOD)


def test_rolls_of_a_box_are_not_merged():
    V = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.03, 0.03) for z in (-0.01, 0.01)])  # a flat box: rolled 90 deg = another face
    a = _outcome(_q_axis([1, 0, 0], 0.0), 0.010)
    b = _outcome(_q_axis([1, 0, 0], math.pi / 2), 0.012)  # within h_tol, but the cross-section is far from round
    cl = SF.cluster_rests([a, b], V)
    assert len(cl) == 2


def test_jobs_keep_the_id_at_the_main_rest_and_derive_one_per_other_frequent_rest():
    job = {"id": "spoon__x__s1", "category": "spoon", "model": "x", "scale": [1, 1, 1], "constraint": "tabletop"}
    V = _cylinder()
    outs = [_outcome(_q_axis([1, 0, 0], 0.0), 0.03)] * 3 + [_outcome(_q_axis([0, 1, 0], -math.pi / 2), 0.10)] * 2 + \
           [_outcome(_q_axis([0, 1, 0], math.pi / 2), 0.10, stable=True)]
    cl = SF.cluster_rests(outs, V)
    jobs = SF.jobs_for_rests(job, cl)
    assert jobs[0]["id"] == job["id"] and jobs[0]["rest_pose"] == "settle_first" and jobs[0]["settle_first"]["rest_id"] == "s0"
    assert len(jobs) == 2 and jobs[1]["derived_from"] == job["id"] and jobs[1]["id"] != job["id"]  # the 1-outcome rest gets no job
    assert jobs[0].get("rest_hint_wxyz") is None and S.job_dir(jobs[0]).endswith(job["id"])  # the designated directory is kept
    w, x, y, z = jobs[0]["generation_rest_wxyz"]
    assert abs(S.tilt_distance([x, y, z, w], cl[0]["quat_xyzw"])) < 1e-5 and "members" not in jobs[0]["settle_first"]


def test_the_retry_overrides_parse_and_default(monkeypatch):
    # of-v2maps 2026-11-07: the escalation ladder's opt-in candidate / densify / validation overrides
    from ispatialgym.grasping import pipeline as P

    monkeypatch.delenv("ISG_GLIB_CAND_OVERRIDES", raising=False)
    monkeypatch.delenv("ISG_GLIB_DENSIFY", raising=False)
    monkeypatch.delenv("ISG_GLIB_VAL_OVERRIDES", raising=False)
    cfg, dz = P.candidate_overrides()
    assert cfg.n_candidates == 6000 and dz == ((4, 2), (8, 4)) and P.validate_overrides().max_trials == 10
    monkeypatch.setenv("ISG_GLIB_CAND_OVERRIDES", '{"n_candidates": 24000, "max_representatives": 20}')
    monkeypatch.setenv("ISG_GLIB_DENSIFY", "4x2,16x8")
    monkeypatch.setenv("ISG_GLIB_VAL_OVERRIDES", '{"max_trials": 30}')
    cfg, dz = P.candidate_overrides()
    assert cfg.n_candidates == 24000 and cfg.max_representatives == 20 and cfg.n_orientations == 12 and dz == ((4, 2), (16, 8))
    assert P.validate_overrides().max_trials == 30
