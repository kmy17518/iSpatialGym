"""i7-attachfix 2026-11-08 (owner direction 02:20): the attachment aliases' fixes.

* item 1 (cap xxqyrt): ``CandidateConfig.outer_pinch`` - every ray hit for the antipodal sampler, the pinch measured from outside - and
  ``attach_face_retreat`` - the top retreat clears the attachment rule's face gap;
* items 2b / 3 / 4 (one class: the object not where the plan thinks): ``run_isg.tro_pose_mismatches`` / ``detach_scene_attachments``;
* item 3: ``sampling.v2.witness_check`` (an independent FK of the recorded IK witnesses before an instance is written).
"""

from __future__ import annotations

import math
import os
import types

import numpy as np
import pytest

# ------------------------------------------------------------------------------------------------ item 1: a hollow cup, open side down


def _cup(r_out=0.026, wall=0.003, h=0.011, top=0.0023, n=24):
    """A cap as the export makes it: a top disc + n wall segments (convex pieces), open side down, z from 0 to h."""
    import trimesh

    pieces = [trimesh.creation.cylinder(radius=r_out, height=top, sections=48)]
    pieces[0].apply_translation([0, 0, h - top / 2])
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        pts = []
        for a in (a0, a1):
            for r in (r_out - wall, r_out):
                for z in (0.0, h - top):
                    pts.append([r * math.cos(a), r * math.sin(a), z])
        pieces.append(trimesh.convex.convex_hull(np.array(pts)))
    return pieces, trimesh.util.concatenate(pieces)


def _pinch(c, closing, approach=(0, 0, 1.0)):
    from ispatialgym.grasping import candidates as CA

    R = CA.proposal_frame(np.array(approach, float), np.array(closing, float))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(c, float) - R[:, 2] * CA.CONTACT_Z
    return T[None]


def test_all_ray_hits_returns_every_surface_along_the_ray():
    from ispatialgym.grasping import candidates as CA

    _, u = _cup()
    o = np.array([[0.04, 0.0, 0.004]])
    d = np.array([[-1.0, 0.0, 0.0]])
    loc, idx, _ = CA.all_ray_hits(u.ray, o, d)
    xs = sorted(round(float(x), 4) for x in loc[:, 0])
    # outer wall, inner wall, inner wall, outer wall (and the piece-cut faces it may graze): the far outer face is among them
    assert min(xs) < -0.025 and max(xs) > 0.025 and len(xs) >= 4
    # the proxy returns them only with all_hits (the embree default: the first one)
    px = CA._MeshProxy(u, all_hits=True)
    loc2, _, _ = px.ray.intersects_location(o, d, multiple_hits=True)
    assert len(loc2) >= 4


def test_outer_pinch_measures_the_diameter_the_pads_close_on():
    from ispatialgym.grasping import candidates as CA

    _, u = _cup()
    T = _pinch([0, 0, 0.004], [1.0, 0, 0])  # the pad centre in the cavity, closing across the diameter
    _, w_in, _, _ = CA.contact_features(u, T, CA.CONTACT_Z)
    _, w_out, ant, _ = CA.contact_features(u, T, CA.CONTACT_Z, outer_half=0.0889 / 2)
    assert w_in[0] < 0.047  # before: the INNER wall (cavity) faces, ~46 mm
    assert abs(w_out[0] - 0.052) < 0.0015 and ant[0] > 0.9  # the outer diameter
    # a convex solid: both measures agree
    import trimesh

    box = trimesh.creation.box(extents=[0.03, 0.05, 0.02])
    Tb = _pinch([0, 0, 0], [1.0, 0, 0])
    assert abs(CA.contact_features(box, Tb, CA.CONTACT_Z)[1][0] - CA.contact_features(box, Tb, CA.CONTACT_Z, outer_half=0.0445)[1][0]) < 1e-6


def test_attach_face_retreat_puts_the_fingers_inside_the_mating_face():
    from ispatialgym.grasping import candidates as CA
    from ispatialgym.grasping import contact_policy as CP

    _, u = _cup()
    hc = CP.HandChecker("right", CP.DEFAULT, spacing=0.004)
    m = np.array([0.0, 0.0, -1.0])  # mating downward: the face is the rim on the table (z = 0)
    face = float((u.vertices @ m).max())
    T = _pinch([0, 0, 0.008], [1.0, 0, 0], approach=(0, 0, -1.0))
    q = np.array([0.036])
    s = CA.attach_face_retreat_m(hc, T, q, [(m, face)], 0.001)
    assert s[0] > 0.0
    T2 = T.copy()
    T2[:, :3, 3] -= T2[:, :3, 2] * s[:, None]
    P, _ = hc.geom.points(("finger",), (0.036, 0.036))
    gap = face - float(((P @ T2[0, :3, :3].T + T2[0, :3, 3]) @ m).max())
    assert gap >= CA.ATTACH_FACE_GAP_M + 0.001 - 1e-9
    # already clear: no retreat
    assert CA.attach_face_retreat_m(hc, T2, q, [(m, face)], 0.001)[0] == pytest.approx(0.0, abs=1e-9)


def test_the_two_switches_are_opt_in_and_enter_the_identity_only_when_on():
    from dataclasses import asdict

    from ispatialgym.grasping import candidates as CA
    from ispatialgym.grasping import schema as S

    sw = S.candidate_switches(asdict(CA.CandidateConfig()))
    assert sw["outer_pinch"] is False and sw["attach_face_retreat"] is False
    assert "outer_pinch" in S.CANDIDATE_SWITCHES and "attach_face_retreat" in S.CANDIDATE_SWITCHES


# ------------------------------------------------------------------------------------------------ items 2b / 3 / 4: the world restore


def test_tro_pose_mismatches_flags_the_reused_world_cases():
    from ispatialgym.oracle import run_isg as RI

    tro = {"lid.n.02_1": ([6.1561, 0.0909, 1.0585], [0, 0, -0.138, 0.9904]), "teapot.n.01_1": ([6.3355, -0.0559, 1.0936], [0, 0, 0.1537, 0.9881])}
    # attachment_01 300115: the lid at the prebuild instance's pose (76.6 mm), the teapot where the TRO puts it
    live = {"lid.n.02_1": ([6.0797, 0.0970, 1.0585], [0, 0, -0.1141, 0.9935]), "teapot.n.01_1": ([6.3355, -0.0559, 1.0936], [0, 0, 0.1537, 0.9881])}
    bad = RI.tro_pose_mismatches(tro, live)
    assert [b["instance"] for b in bad] == ["lid.n.02_1"] and abs(bad[0]["pos_m"] - 0.0766) < 1e-3
    # a settle-sized difference passes; a pure yaw beyond the tolerance does not
    live_ok = {k: ([v[0][0] + 0.0004, v[0][1], v[0][2]], v[1]) for k, v in tro.items()}
    assert RI.tro_pose_mismatches(tro, live_ok) == []
    yaw = math.radians(5.0)
    live_rot = dict(live_ok, **{"lid.n.02_1": (tro["lid.n.02_1"][0], [0, 0, math.sin(yaw / 2), math.cos(yaw / 2)])})
    tro_rot = dict(tro, **{"lid.n.02_1": (tro["lid.n.02_1"][0], [0, 0, 0, 1.0])})
    assert [b["instance"] for b in RI.tro_pose_mismatches(tro_rot, live_rot)] == ["lid.n.02_1"]


def test_detach_scene_attachments_removes_every_joint_outside_the_state_load(monkeypatch):
    pytest.importorskip("omnigibson.object_states")
    from omnigibson.object_states import AttachedTo

    from ispatialgym.oracle import run_isg as RI

    calls = []

    class St:
        def __init__(self, parent):
            self.parent = parent

        def set_value(self, other, value):
            calls.append((other.name, value))
            self.parent = None

    teapot = types.SimpleNamespace(name="partner_teapot", states={AttachedTo: St(None)})
    lid = types.SimpleNamespace(name="target_lid", states={AttachedTo: St(teapot)})
    table = types.SimpleNamespace(name="table", states={})
    out = RI.detach_scene_attachments(types.SimpleNamespace(objects=[table, teapot, lid]))
    assert out == [{"child": "target_lid", "parent": "partner_teapot"}] and calls == [("partner_teapot", False)]
    assert lid.states[AttachedTo].parent is None
    assert RI.detach_scene_attachments(types.SimpleNamespace(objects=[table, teapot, lid])) == []


# ------------------------------------------------------------------------------------------------ item 3: the recorded IK witnesses

# attachment_01 instance 300115's pre_grasp witness (i7_attswept sidecar): FK reaches the pose within 0.01 mm
W115 = {"arm": "right", "q": [6.17337, -0.6455, 0.00532, 0.0, 0.0, 1.56825, 0.7375, -0.925, -0.50127, 0.0, 0.0, -2.26901, 1.57, 0.08656, 0.0,
                              1.9744, -1.57, -1.00442, 1.57, -0.98261, 0.0, 0.99169, 0.0, 1.00524, 0.05, 0.05, 0.05, 0.05],
        "pose": {"position": [6.16664, 0.07449, 1.18651], "orientation": [0.995241, -0.040577, -0.053413, -0.070678]}}


def _fk_or_skip():
    from ispatialgym.oracle import r1pro_model as M
    from ispatialgym.sampling.v2 import witness_check as WC

    if not os.path.exists(M.source_urdf_path()):
        pytest.skip("no R1Pro URDF in this environment")
    return WC, WC.default_fk()


def test_witness_fk_reaches_the_recorded_pose_and_a_moved_one_is_caught():
    WC, fk = _fk_or_skip()
    wit = [{"action": 3, "name": "grasp", "arm": "right", "targets": {"pre_grasp": {"q": W115["q"], "pose": W115["pose"]}}}]
    err = WC.witness_errors(wit, fk)
    assert len(err) == 1 and err[0]["pos_m"] < 1e-4 and err[0]["rot_rad"] < 1e-3
    assert WC.witness_violations(wit, fk) == []
    # the pose 77 mm off (what the oracle saw in 300115 - there the LID was off its TRO pose, the witness was fine)
    moved = {"position": [6.16664 - 0.0765, 0.07449 + 0.006, 1.18651], "orientation": W115["pose"]["orientation"]}
    wit2 = [{"action": 3, "name": "grasp", "arm": "right", "targets": {"pre_grasp": {"q": W115["q"], "pose": moved}}}]
    bad = WC.witness_violations(wit2, fk)
    assert len(bad) == 1 and abs(bad[0]["pos_m"] - 0.0767) < 2e-3
    # a q-only target (base_follow regrasp) is skipped
    assert WC.witness_errors([{"name": "x", "arm": "right", "targets": {"regrasp[0]": {"q": W115["q"]}}}], fk) == []


def test_reset_pair_records_unfilters_the_joined_pair_and_clears_the_pending(monkeypatch):
    """run_isg.Loader.attach now calls it before the state load (as the vectorized loader's cleanup): the un-filter of the previous
    episode's joined pair (a stop / play) must not run inside the next instance's restore."""
    from ispatialgym.sampling import predicates as PRED

    calls = []

    def fake_filter(child, parent, filtered):
        calls.append((child.name, parent.name, filtered))
        PRED._FILTERED.discard(PRED._pair_key(child, parent))
        return True

    monkeypatch.setattr(PRED, "set_pair_collision_filter", fake_filter)
    scene = types.SimpleNamespace(idx=0)
    lid = types.SimpleNamespace(name="target_lid", scene=scene)
    teapot = types.SimpleNamespace(name="partner_teapot", scene=scene)
    scene.object_registry = lambda attr, name: {"target_lid": lid, "partner_teapot": teapot}.get(name)
    key = PRED._pair_key(lid, teapot)
    monkeypatch.setattr(PRED, "_FILTERED", {key})
    monkeypatch.setattr(PRED, "_PENDING_UNFILTER", {key})
    monkeypatch.setattr(PRED, "_PENDING_FILTER", set())
    monkeypatch.setattr(PRED, "_RELEASED_TOUCHING", set())
    out = PRED.reset_pair_records(scene)
    assert calls == [("target_lid", "partner_teapot", False)] and out == {"unfiltered": 1, "pending_cleared": 1}
    assert not PRED._FILTERED and not PRED._PENDING_UNFILTER


def test_the_alias_knob_lift_is_off_for_a_v2_library_grasp(monkeypatch):
    """cap aliases (isg_customs grasp_height_extra_m 5 mm): a v2 grasp keeps its validated height (the floors still apply)."""
    pytest.importorskip("omnigibson")
    from ispatialgym.sampling import grasp_library as GL
    from ispatialgym.sampling.v2.tasks import attachment as AT

    monkeypatch.setattr(GL, "v2_entry", lambda key: object() if key == "cap__xxqyrt" else None)
    assert AT.v2_library_lift_off("cap__xxqyrt") is True
    assert AT.v2_library_lift_off("lid__legacy_only") is False and AT.v2_library_lift_off(None) is False

    def boom(key):
        raise FileNotFoundError(key)

    monkeypatch.setattr(GL, "v2_entry", boom)
    assert AT.v2_library_lift_off("cap__xxqyrt") is False
