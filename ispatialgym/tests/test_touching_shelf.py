"""The touching#03 shelf layout's push (owner 2026-10-11 11:46: "partner leaning on the right board, the LEFT arm grasps / places /
pushes rightward, no brace, push parameters mirrored, new branches only"): the push contact on the target's left face, the push
along -tangent by the left arm with the right hand's contact laterals and wrist-turn rules mirrored, no brace at any stance, the
goal nudge toward the partner, the left-arm grasp at the fixed start, the tabletop touching unchanged, and the oracle program for
the recorded left push. ``IORSSmall`` methods run on stand-ins; no simulator is launched."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import types
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2.stances import EdgeFrame, SidecarDraft, Stance

VARIANTS = os.path.join(os.path.dirname(__file__), "data", "sidecar_variants")
#: hardback/acbrnv (the touching#03 target and partner class): 22.3 x 17.2 x 3.1 cm
BOOK = th.tensor([0.22274, 0.17153, 0.03120]) / 2
#: the edge frame of the tests: the robot faces +x, its left (the edge tangent) is +y, so a lateral is a world y
TANGENT = th.tensor([0.0, 1.0])


def _iors():
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    return IS


def _book_hull() -> th.Tensor:
    return th.tensor([[sx * BOOK[0], sy * BOOK[1], sz * BOOK[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=th.float32)


def _edge() -> EdgeFrame:
    return EdgeFrame("-v", th.tensor([0.0, 0.0]), th.tensor([1.0, 0.0]), 0.8)


def _stance(name: str = "manipulation", arms=("left",)) -> Stance:
    st = Stance(name, (-0.45, 0.0, 0.0), 0.5, 0.0, 0.45, _edge(), lateral=0.0)
    st.arms = list(arms)
    return st


def _leaning(y: float, lean_side: float, board_z: float = 0.81):
    """A book standing on its long side on the board at lateral ``y``, spine toward the robot, leaning 10 deg toward ``lean_side``."""
    from ispatialgym.sampling.v2 import tabletop as TT

    q = TT.upright_lean_quat(_book_hull(), lean_side, 10.0)
    pts = C.transform_points((th.zeros(3), q), _book_hull())
    return th.tensor([0.10, float(y), board_z - float(pts[:, 2].min())]), q


def _hull_at(hull, pose):
    return C.transform_points((th.as_tensor(pose[0]).float(), th.as_tensor(pose[1]).float()), hull)


def _sampler(lean_side=1.0, **extra):
    """An ``IORSSmall`` stand-in: the shelf start (None = the tabletop layout), the two book hulls, a world whose hulls follow the
    poses and the generation defaults; the touching methods bound to it."""
    IS = _iors()
    fields = dict(shelf_start=None if lean_side is None else {"lean_side": float(lean_side), "lean_deg": 10.0}, relation="touching",
                  hull={"target": _book_hull(), "partner": _book_hull()}, world=SimpleNamespace(hull_at=_hull_at),
                  gen=lambda key, default=None: default, tol=lambda key, default=None: default, pair={"g_NT": 0.044},
                  gp=SimpleNamespace(variant="inter_object_relation_satisfaction-touching_small_scale-hardback-hardback-shelf"))
    fields.update(extra)
    s = SimpleNamespace(**fields)
    for name in ("shelf_push_sign", "shelf_push_arm", "shelf_push_direction", "touching_points", "push_stance_for", "brace_gate_at", "push_chain_feasible",
                 "_touching_push_choice", "touching_push_candidates", "_leaning_blade", "_raised_blade", "push_distance", "push_margin", "place_gap",
                 "push_hand_overlap", "hand_overlap", "pair_goal_align", "push_speed"):
        setattr(s, name, types.MethodType(getattr(IS.IORSSmall, name), s))
    return s


def _scene():
    """The place of the shelf layout: the target 14 cm left of the partner, both leaning right (the partner against the right board)."""
    t_pose = _leaning(0.10, -1.0)
    p_pose = _leaning(-0.04, -1.0)
    return t_pose, p_pose


# ----------------------------------------------------------------------------------------------------------------
# the layout: arm, direction, contact face, gap
# ----------------------------------------------------------------------------------------------------------------


def test_the_left_arm_pushes_to_the_robots_right_when_the_partner_leans_on_the_right_board():
    st = _stance()
    s = _sampler(1.0)
    assert s.shelf_push_sign() == -1.0 and s.shelf_push_arm() == "left"
    assert th.allclose(s.shelf_push_direction(st), th.tensor([0.0, -1.0, 0.0]))  # -tangent: the robot's right
    # the mirror image (a key leaning right): the right arm pushes to the left, as on the tabletop
    s2 = _sampler(-1.0)
    assert s2.shelf_push_arm() == "right" and th.allclose(s2.shelf_push_direction(st), th.tensor([0.0, 1.0, 0.0]))


def test_the_push_contact_is_on_the_targets_left_face_and_the_tabletop_points_are_unchanged():
    IS = _iors()
    st = _stance()
    t_pose, p_pose = _scene()
    t_w, p_w = _hull_at(_book_hull(), t_pose), _hull_at(_book_hull(), p_pose)
    work, far = _sampler(1.0).touching_points(t_pose, p_pose, st)
    mid_t = 0.5 * float(t_w[:, 2].min() + t_w[:, 2].max())
    # the target's left face (its largest lateral), mid height, on the line through its centre; the partner's far (right) face
    assert float(work[1]) == pytest.approx(float(t_w[:, 1].max()), abs=1e-5) and float(work[2]) == pytest.approx(mid_t, abs=1e-5)
    assert float(work[0]) == pytest.approx(float(t_w[:, 0].mean()), abs=1e-5)
    assert float(far[1]) == pytest.approx(float(p_w[:, 1].min()), abs=1e-5)
    # the tabletop layout keeps its points (the target's right face, the partner's left face), and the general face point along
    # +tangent is that right-face point
    work0, brace0 = _sampler(None).touching_points(t_pose, p_pose, st)
    assert float(work0[1]) == pytest.approx(float(t_w[:, 1].min()), abs=1e-5) and float(brace0[1]) == pytest.approx(float(p_w[:, 1].max()), abs=1e-5)
    assert th.allclose(IS.push_face_point(t_w, TANGENT), work0, atol=1e-6)
    assert th.allclose(IS.push_face_point(t_w, -TANGENT), work, atol=1e-6)


def test_the_gap_and_the_push_distance_are_measured_along_the_shelf_push():
    IS = _iors()
    t_pose, p_pose = _scene()
    t_w, p_w = _hull_at(_book_hull(), t_pose), _hull_at(_book_hull(), p_pose)
    gap = IS.gap_along(t_w, p_w, th.tensor([0.0, -1.0, 0.0]))
    assert gap == pytest.approx(float(t_w[:, 1].min()) - float(p_w[:, 1].max()), abs=1e-6) and gap > 0.0
    # along +tangent the partner lies behind the target: the tabletop measure would read an overlap
    assert IS.gap_along(t_w, p_w, th.tensor([0.0, 1.0, 0.0])) < 0.0
    s = _sampler(1.0)
    assert s.push_distance(gap) == pytest.approx(gap + 0.03 + 0.005)


def test_no_brace_at_any_stance_of_the_shelf_layout():
    t_pose, p_pose = _scene()
    gate = _sampler(1.0).brace_gate_at(_stance("push"), t_pose, p_pose)
    assert gate["ok"] is False and "no brace" in gate["why"] and gate["stance"] == "push"


# ----------------------------------------------------------------------------------------------------------------
# the push stance: the right hand's contact lateral mirrored
# ----------------------------------------------------------------------------------------------------------------


def test_the_push_stance_puts_the_left_face_left_of_centre_by_the_mirrored_lateral():
    t_pose = _leaning(0.25, -1.0)  # the left face 0.28 m left of the manipulation stance's centre line
    t_w = _hull_at(_book_hull(), t_pose)
    left_face = t_w[int(th.argmax(t_w[:, 1])), :2]
    right_face = t_w[int(th.argmin(t_w[:, 1])), :2]
    manip = _stance()
    st = _sampler(1.0).push_stance_for(manip, t_pose)
    assert st is not None and st.name == "push" and st.arms == ["left"]
    assert st.local(left_face)[1] == pytest.approx(0.12, abs=1e-4)  # default -0.12 for the right hand -> +0.12
    assert st.extras["push_contact_lateral_m"] == pytest.approx(0.12, abs=1e-4)
    assert abs(st.base_pose2d[2] - manip.base_pose2d[2]) < 1e-9 and (st.translate, st.tilt) == (manip.translate, manip.tilt)
    # an explicit lateral of the search is taken as given
    assert _sampler(1.0).push_stance_for(manip, t_pose, lateral=0.16).local(left_face)[1] == pytest.approx(0.16, abs=1e-4)
    # the tabletop layout: the right face 12 cm right of centre, the right arm
    st0 = _sampler(None).push_stance_for(_stance(arms=("right",)), t_pose)
    assert st0.arms == ["right"] and st0.local(right_face)[1] == pytest.approx(-0.12, abs=1e-4)


# ----------------------------------------------------------------------------------------------------------------
# the push classes: mirrored poses, the camera up / away from the book first for the left wrist
# ----------------------------------------------------------------------------------------------------------------


def _camera_offset(pose) -> th.Tensor:
    """Where the wrist camera's spheres (``HAND_SPHERES_EEF``, the same for both grippers) sit relative to the eef origin (world)."""
    IS = _iors()
    centres = th.tensor(IS.HAND_SPHERES_EEF["wrist camera"][0], dtype=th.float32)
    return C.transform_points((th.as_tensor(pose[0]).float(), th.as_tensor(pose[1]).float()), centres).mean(dim=0) - th.as_tensor(pose[0]).float()


def test_the_left_hand_push_classes_mirror_the_right_hand_ones_with_the_camera_up_or_away_first():
    """The left hand's push along -tangent on a box's left face is the right hand's along +tangent on the mirrored box's right face:
    the same classes in the same order, every eef origin and approach axis reflected in the robot's sagittal plane. The horizontal
    tip / pad pushes are the exact mirror images (axes reflected, the eef y flipped back: the codebase's mirror of a hand pose) and
    their first turn keeps the camera up; a vertical closing keeps its camera-up turn; the blades keep the rule of
    ``PUSH_WRIST_TURNS`` - the first turn holds the camera on the side away from the pushed object (its exact mirror would hang it
    over the object: the two wrist cameras sit at the same eef-frame position, they are not mirrored)."""
    st = _stance()
    s = _sampler(1.0)
    M = th.diag(th.tensor([1.0, -1.0, 1.0]))
    face_z = (0.81, 0.92)  # an 11 cm face: tips, pads, low / leaning / raised blades all apply
    fp_right = th.tensor([0.10, -0.06, 0.865])
    fp_left = M @ fp_right
    right = list(s.touching_push_candidates(st, fp_right, face_z, th.tensor([0.0, 1.0, 0.0])))
    left = list(s.touching_push_candidates(st, fp_left, face_z, th.tensor([0.0, -1.0, 0.0])))
    assert [k for k, _, _ in left] == [k for k, _, _ in right] and len(left) >= 8
    kinds = [k for k, _, _ in left]
    assert {"push_tip_h-", "push_tip_h", "push_tip_v-", "push_h", "push_v"} <= set(kinds) and any(k.startswith("push_blade") for k in kinds)
    for (kind, (o_l, q_l), _), (_, (o_r, q_r), _) in zip(left, right):
        R_l, R_r = C.quat_to_mat(q_l), C.quat_to_mat(q_r)
        assert th.allclose(o_l, M @ o_r, atol=1e-6), kind
        assert th.allclose(R_l[:, 2], M @ R_r[:, 2], atol=1e-5), kind
        if kind.startswith("push_tip_h") or kind == "push_h":
            assert th.allclose(R_l, M @ R_r @ M, atol=1e-5), f"{kind}: not the mirror image of the right hand's"
    pose_of = {k: p for k, p, _ in left}
    pose_of_r = {k: p for k, p, _ in right}
    # horizontal tips, first turn: the camera above the eef axis for either hand (the other turn hangs it below)
    for poses in (pose_of, pose_of_r):
        assert float(_camera_offset(poses["push_tip_h-"])[2]) > 0.04 and float(_camera_offset(poses["push_tip_h"])[2]) < -0.04
        assert float(_camera_offset(poses["push_tip_v-"])[2]) > 0.0  # the vertical closing: camera up only
    # blades, first turn: the camera on the outward side of the pushed face (+tangent for the left face, -tangent for the right
    # one), 6 cm farther out than in the other turn
    blades = [k for k in kinds if k.startswith("push_blade") and k.endswith("-")]
    assert len(blades) >= 3
    for k in blades:
        for poses, outward in ((pose_of, TANGENT), (pose_of_r, -TANGENT)):
            first, other = float(_camera_offset(poses[k])[:2] @ outward), float(_camera_offset(poses[k.rstrip("-")])[:2] @ outward)
            assert first > 0.02 and first > other + 0.05, (k, first, other)
    # the tip pushes start outside the left face: the fingertips 5 mm left of it, pointing right
    tip = C.fingertip(pose_of["push_tip_h-"])
    assert float(tip[1]) == pytest.approx(float(fp_left[1]) + 0.005, abs=1e-6)


def test_the_push_chain_is_solved_for_the_left_arm_along_minus_tangent_and_never_braced(monkeypatch):
    IS = _iors()
    st = _stance("push")
    t_pose, p_pose = _scene()
    calls = {}

    def choose_first(ikc, arm, stream, **kw):
        cands = list(stream)
        calls.update(arm=arm, cands=cands, stage=kw.get("stage"), chain=kw["chain"](cands[0]))
        return SimpleNamespace(proposal=SimpleNamespace(payload=cands[0], chain=calls["chain"])), None, []

    monkeypatch.setattr(IS.O, "choose_first", choose_first)
    s = _sampler(1.0, ikc=SimpleNamespace(q_locked=lambda stance: th.zeros(28)), support_objects=lambda: [], obj=lambda role: role,
                 posed=lambda role, pose: contextlib.nullcontext())
    assert s.push_chain_feasible(st, t_pose, p_pose) == "push_tip_h-"
    assert calls["arm"] == "left" and calls["stage"] == "touching push"
    d = th.tensor([0.0, -1.0, 0.0])
    t_w = _hull_at(_book_hull(), t_pose)
    for kind, (o, q), _ in calls["cands"]:
        if kind.startswith("push_tip") or kind in ("push_h", "push_v"):
            assert th.allclose(C.approach_axis(q), d, atol=1e-5), kind  # every horizontal class approaches along -tangent
        assert float(o[1]) > float(t_w[:, 1].max()) - 0.03, f"{kind}: the hand starts on the book's left"
    # the chain: the tip push's 5 cm pre-contact back along -tangent, then the measured gap along the push plus the margin
    pre, init, goal = calls["chain"]
    gap = IS.gap_along(t_w, _hull_at(_book_hull(), p_pose), d)
    assert th.allclose(init[0] - pre[0], d * 0.05, atol=1e-6) and th.allclose(goal[0] - init[0], d * (gap + 0.035), atol=1e-6)
    # with a brace asked for, the shelf layout has none to offer
    assert s.push_chain_feasible(st, t_pose, p_pose, with_brace=True) is None


def test_the_tabletop_push_chain_keeps_the_right_arm_along_plus_tangent(monkeypatch):
    IS = _iors()
    calls = {}

    def choose_first(ikc, arm, stream, **kw):
        cands = list(stream)
        calls.update(arm=arm, cands=cands)
        return None, None, []

    monkeypatch.setattr(IS.O, "choose_first", choose_first)
    s = _sampler(None, ikc=SimpleNamespace(q_locked=lambda stance: th.zeros(28)), support_objects=lambda: [], obj=lambda role: role,
                 posed=lambda role, pose: contextlib.nullcontext())
    t_pose, p_pose = _leaning(-0.10, 1.0), _leaning(0.04, 1.0)
    assert s.push_chain_feasible(_stance(arms=("right",)), t_pose, p_pose) is None
    assert calls["arm"] == "right"
    tip = next(p for k, p, _ in calls["cands"] if k == "push_tip_h-")
    assert th.allclose(C.approach_axis(tip[1]), th.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def test_the_shelf_layout_geometry_chain_on_a_compartment():
    """The whole mirrored chain on a 0.60 m compartment (side boards at x = -/+0.30, the robot faces +y so its left is -x) with the
    library rest of hardback__acbrnv__bu: the target leans against the left board, the partner against the right one, the goal is
    the target against the partner in the mirrored lean (goal side "left"), the place sits the place gap back toward the left
    board, the push stance puts the target's left face 12 cm left of centre, the push contact is that face and every push class
    approaches from the left along +x; the push spans the place gap plus the margin."""
    import math

    from ispatialgym.sampling.sim_utils import SupportTop
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.sampling.v2.stances import nearer_side_lateral
    from ispatialgym.tests.test_v2_tabletop import BU_LEAN_DEG, BU_REST_XYZW

    IS = _iors()
    hull = _book_hull()
    xs, ys = th.arange(-0.29, 0.30, 0.02), th.arange(0.01, 0.30, 0.02)
    top = SupportTop("shelf_x_2", xs, ys, th.ones(len(xs), len(ys), dtype=th.bool), th.zeros(len(xs), len(ys), dtype=th.bool), 0.81, 0.0)
    edge = EdgeFrame("-v", th.tensor([0.0, 0.01]), th.tensor([0.0, 1.0]), 0.58)
    support = SimpleNamespace(name="shelf_x_2", links={"base_link": SimpleNamespace(prim_path="/World/shelf_x_2/base_link")})

    def raycast(starts, ends):  # either side board, 0.30 m from the compartment's centre line
        return [{"hit": True, "rigidBody": "/World/shelf_x_2/base_link/collisions", "distance": 0.30}]

    world = SimpleNamespace(raycast=raycast, hull_at=_hull_at, bottom_offset=lambda h, q: -float(C.transform_points((th.zeros(3), q), h)[:, 2].min()))
    s = _sampler(1.0, world=world, edge=edge)
    s.shelf_start["lean_deg"] = BU_LEAN_DEG
    s.target_goal_quat = types.MethodType(IS.IORSSmall.target_goal_quat, s)
    s.place_pose_for = types.MethodType(IS.IORSSmall.place_pose_for, s)
    t_rest = th.tensor(BU_REST_XYZW)
    t_start, tf = TT.shelf_lean_pose(s, hull, t_rest, top, edge, support, 1.0)
    p_start, pf = TT.shelf_lean_pose(s, hull, TT.upright_lean_quat(hull, -1.0, BU_LEAN_DEG), top, edge, support, -1.0)
    t_w, p_w = _hull_at(hull, t_start), _hull_at(hull, p_start)
    assert tf["lean_side"] == "left" and pf["lean_side"] == "right"
    assert float(t_w[:, 0].min()) == pytest.approx(-0.30 + TT.SHELF_WALL_GAP_M, abs=1e-4) and float(p_w[:, 0].max()) == pytest.approx(0.30 - TT.SHELF_WALL_GAP_M, abs=1e-4)
    manip = Stance("manipulation", (0.0, -0.44, math.pi / 2), 0.5, 0.0, 0.45, edge, lateral=0.0)
    manip.arms = ["left"]
    d = s.shelf_push_direction(manip)
    assert th.allclose(d, th.tensor([1.0, 0.0, 0.0]), atol=1e-6), "the robot's right"
    # the goal: the target against the partner in the mirrored lean, on the partner side facing its start
    q_goal = s.target_goal_quat(t_start, manip)
    g_w0 = C.transform_points((th.zeros(3), q_goal), hull)
    lean = lambda pts: float(pts[pts[:, 2] > float(pts[:, 2].mean())][:, 0].mean() - pts[pts[:, 2] < float(pts[:, 2].mean())][:, 0].mean())  # noqa: E731
    assert lean(g_w0) > 0.02, "the goal leans to the robot's right, toward the partner"
    p_lat = (p_w[:, :2] - manip.base_xy) @ manip.tangent
    side, lat = nearer_side_lateral(manip.local(t_start[0])[1], (float(p_lat.min()), float(p_lat.max())),
                                    (float((g_w0[:, :2] @ manip.tangent).min()), float((g_w0[:, :2] @ manip.tangent).max())), 0.0)
    assert side == "left"
    goal = (th.tensor([*manip.world(manip.local(t_start[0])[0], lat).tolist(), float(t_start[0][2])]), q_goal)
    assert IS.gap_along(_hull_at(hull, goal), p_w, d) == pytest.approx(0.0, abs=1e-5)
    # the place: back from the goal against the push by the place gap (after_goal_check), the draw's place agrees
    place = (goal[0] - d * s.place_gap(), q_goal)
    drawn = s.place_pose_for(t_start, p_start, manip)
    assert th.allclose(drawn[0], place[0], atol=1e-5) and C.quat_angle(drawn[1], q_goal) < 1e-5
    gap = IS.gap_along(_hull_at(hull, place), p_w, d)
    assert gap == pytest.approx(s.place_gap(), abs=1e-5) and s.push_distance(gap) == pytest.approx(s.place_gap() + 0.035, abs=1e-5)
    # the push stance: the target's left face (its smallest x) 12 cm left of the robot's centre line, the left arm working
    st = s.push_stance_for(manip, place)
    pl_w = _hull_at(hull, place)
    assert st.arms == ["left"] and st.local(pl_w[int(th.argmin(pl_w[:, 0])), :2])[1] == pytest.approx(0.12, abs=1e-4)
    # the push contact on that face, mid height; every push class comes from the left of the book
    work, _ = s.touching_points(place, p_start, st)
    assert float(work[0]) == pytest.approx(float(pl_w[:, 0].min()), abs=1e-5)
    cands = list(s.touching_push_candidates(st, work, (float(pl_w[:, 2].min()), float(pl_w[:, 2].max())), d))
    tip = next(p for k, p, _ in cands if k == "push_tip_h-")
    assert th.allclose(C.approach_axis(tip[1]), d, atol=1e-5) and float(C.fingertip(tip)[0]) == pytest.approx(float(work[0]) - 0.005, abs=1e-6)
    assert all(float(o[0]) < float(work[0]) + 0.015 for _, (o, _q), _ in cands)


# ----------------------------------------------------------------------------------------------------------------
# the recorded push and the goal nudge
# ----------------------------------------------------------------------------------------------------------------


def test_touching_contacts_records_the_left_push_with_its_layout_and_no_brace():
    from ispatialgym.oracle.programs import TOUCHING_SHELF_LAYOUT
    from ispatialgym.oracle.sidecar import PushContact
    from ispatialgym.sampling.tasks.base import Draft

    IS = _iors()
    tname, pname = "hardback.n.01_1", "hardback.n.01_2"
    t_pose, p_pose = _scene()
    seen = {}

    def push_choice(st, t_pose_, p_init, t_face_z, direction, dist, ignore_push):
        seen.update(p_init=p_init, direction=direction, dist=dist)
        init = C.fingertip_eef(p_init, direction, -C.default_closing_axis(direction), 0.005)
        chain = [C.pre_pose(init, 0.05), init, (init[0] + direction * dist, init[1])]
        return SimpleNamespace(proposal=SimpleNamespace(payload=("push_tip_h-", init, {"tip_front": True, "pre_dist_m": 0.05}), chain=chain)), None, []

    partner = SimpleNamespace(get_position_orientation=lambda: (p_pose[0].clone(), p_pose[1].clone()))
    s = _sampler(1.0, bindings=SimpleNamespace(one=lambda role: SimpleNamespace(name=tname if role == "target" else pname)), _touching_place_pos=t_pose[0],
                 obj=lambda role: partner if role == "partner" else SimpleNamespace(), support_objects=lambda: [], posed=lambda role, pose: contextlib.nullcontext(),
                 _brace_gate={"ok": False, "why": "shelf (owner 2026-10-11): the side board holds the partner, no brace"},
                 sidecar=SidecarDraft("a", "inter_object_relation_satisfaction"))
    s._touching_push_choice = push_choice
    draft = Draft()
    draft.goal_poses[tname] = (t_pose[0] + th.tensor([0.0, -0.034, 0.0]), t_pose[1])
    draft.pose_targets[tname] = {"position": [float(v) for v in draft.goal_poses[tname][0]]}
    st = _stance("push")
    IS.IORSSmall.touching_contacts(s, draft, st)
    rec = s.sidecar.contacts["push"]
    d = th.tensor([0.0, -1.0, 0.0])
    gap = IS.gap_along(_hull_at(_book_hull(), t_pose), _hull_at(_book_hull(), p_pose), d)
    assert rec.arm == "left" and rec.object == tname and rec.stance == "push" and th.allclose(rec.direction, d)
    assert rec.distance_m == pytest.approx(gap + 0.035, abs=1e-6) and th.allclose(seen["direction"], d)
    assert rec.extras["layout"] == TOUCHING_SHELF_LAYOUT and rec.extras["pose_class"] == "push_tip_h-"
    assert "brace" not in s.sidecar.contacts and st.arms == ["left"] and st.contacts == ["push"]
    assert draft.summary["brace"]["present"] is False and "no brace" in draft.summary["brace"]["why"]
    tf = draft.summary["touching"]
    assert tf["layout"] == TOUCHING_SHELF_LAYOUT and tf["push_arm"] == "left" and tf["push_direction"] == [0.0, -1.0, 0.0]
    assert tf["gap_at_place_m"] == pytest.approx(gap, abs=1e-4)
    # the written record carries the layout to the oracle
    d_push = s.sidecar.to_dict()["contacts"]["push"]
    pc = PushContact.from_dict(d_push, "contacts.push")
    assert pc.arm == "left" and pc.extra["layout"] == TOUCHING_SHELF_LAYOUT and pc.direction == pytest.approx([0.0, -1.0, 0.0])


def test_the_goal_nudge_moves_the_target_toward_the_partner_on_the_right_board():
    from omnigibson.object_states import Touching

    IS = _iors()
    moves = []

    class Book:
        def __init__(self, touch_after):
            self.pos, self.quat, self.n = th.tensor([0.10, 0.10, 0.92]), th.tensor([0.0, 0.0, 0.0, 1.0]), touch_after
            self.states = {Touching: SimpleNamespace(get_value=lambda other: len(moves) >= self.n)}

        def get_position_orientation(self):
            return self.pos.clone(), self.quat.clone()

        def set_position_orientation(self, pos, quat):
            moves.append(pos - self.pos)
            self.pos = pos

        def keep_still(self):
            pass

    for lean_side, expect in ((1.0, -1.0), (None, 1.0)):
        moves.clear()
        t, p = Book(3), Book(10 ** 6)
        s = _sampler(lean_side, objects={"t": t, "p": p}, robot=None, sidecar=SimpleNamespace(stance=lambda name: _stance()),
                     world=SimpleNamespace(wake=lambda objs: None, step_physics=lambda n, still=(): None))
        IS.IORSSmall._nudge_until_touching(s, "t", "p")
        assert len(moves) == 3 and all(float(m[1]) == pytest.approx(expect * 0.001, abs=1e-6) and float(m[0]) == 0.0 for m in moves)


# ----------------------------------------------------------------------------------------------------------------
# the left arm's grasp at the fixed start
# ----------------------------------------------------------------------------------------------------------------


def test_the_left_arm_takes_the_library_pinch_at_the_fixed_start_in_the_turn_the_rule_prefers():
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.tests.test_left_arm import START_L, START_R, _grasp_row, _q, _two_arm_sampler

    g = _grasp_row()
    s = _two_arm_sampler({"right": None, "left": _q(right=START_R, left=START_L)}, twins=True)
    manip = _stance(arms=("left",))
    obj = (th.tensor([0.6, 0.0, 0.77]), C.IDENTITY_QUAT)
    rec = TT.grasp_at_pose(s, manip, obj, [g], [], 0.0, arm="left", object_name="hardback.n.01_1")
    assert rec.arm == "left" and rec.stance == "manipulation" and C.split_twin(rec.grasp_name)[0] == "grasp_4"
    assert all(call == ("left", "left") for call in s.calls), "every chain solved with the left arm at the stance it works at"
    tc = rec.extras["twin_choice"]
    assert tc["arm"] == "left" and {tc["row"]["name"], tc["twin"]["name"]} == {"grasp_4", "grasp_4~r180"} and "wrist_flip" in rec.extras
    assert TT.fallback_arms(rec) == [], "the shelf layout's arm has no fallback: the right arm was never offered"
    # neither turn reachable: grasp_unreachable
    s2 = _two_arm_sampler({"right": None, "left": None}, twins=True)
    with pytest.raises(Exception, match="grasp_unreachable|no library grasp"):
        TT.grasp_at_pose(s2, _stance(arms=("left",)), obj, [g], [], 0.0, arm="left", object_name="hardback.n.01_1")
    # without twins the library row alone, as before
    s3 = _two_arm_sampler({"right": None, "left": _q(right=START_R, left=START_L)}, twins=False)
    rec3 = TT.grasp_at_pose(s3, _stance(arms=("left",)), obj, [g], [], 0.0, arm="left", object_name="hardback.n.01_1")
    assert rec3.grasp_name == "grasp_4" and "twin_choice" not in rec3.extras


# ----------------------------------------------------------------------------------------------------------------
# the oracle program
# ----------------------------------------------------------------------------------------------------------------


def _shelf_record() -> dict:
    """The push-only touching fixture turned into the shelf layout: the left hand grasps, places and pushes the target to the
    robot's right (the fixture's robot faces +y, so its right is +x)."""
    from ispatialgym.oracle.programs import TOUCHING_SHELF_LAYOUT

    with open(os.path.join(VARIANTS, "inter_object_relation_satisfaction-touching_small_scale__push_only.json")) as f:
        d = json.load(f)
    d["grasps"][0]["arm"] = "left"
    d["stances"]["manipulation"]["arms"] = ["left"]
    push = d["contacts"]["push"]
    push.update({"arm": "left", "layout": TOUCHING_SHELF_LAYOUT, "direction": [1.0, 0.0, 0.0]})
    init = push["init"]["position"]
    push["pre"]["position"] = [init[0] - 0.1, init[1], init[2]]
    push["goal"]["position"] = [init[0] + push["distance_m"], init[1], init[2]]
    return d


def test_the_oracle_program_runs_the_recorded_left_push_of_the_shelf_layout():
    from ispatialgym.oracle.programs import expand
    from ispatialgym.oracle.sidecar import Sidecar

    d = _shelf_record()
    sc = Sidecar.from_dict(d)
    assert sc.problems() == [] and sc.validate() is sc
    actions = expand(sc.variant, sc)
    assert [a.name for a in actions] == ["transition", "grasp", "lift", "place", "retreat", "push", "retreat"]
    assert all(a.arms == ("left",) for a in actions[1:]), [(a.name, a.arms) for a in actions]
    push = actions[-2]
    assert push.contact.direction == pytest.approx([1.0, 0.0, 0.0]) and push.follow == d["grasps"][0]["place_on"]
    assert actions[-1].after is push and actions[-1].release is False
    fine, numbers = push.annotate(sc)
    assert "to the right" in fine and "left" in fine


def test_the_oracle_refuses_a_shelf_record_with_a_brace_or_another_pushing_hand():
    from ispatialgym.oracle.sidecar import Sidecar

    d = _shelf_record()
    other = copy.deepcopy(d)
    other["contacts"]["push"]["arm"] = "right"
    assert any("shelf layout" in p for p in Sidecar.from_dict(other).problems())
    braced = copy.deepcopy(d)
    braced["contacts"]["brace"] = {"arm": "right", "object": d["grasps"][0]["place_on"], "kind": "far_side", "pre": d["contacts"]["push"]["pre"],
                                   "contact": d["contacts"]["push"]["init"], "direction": [-1.0, 0.0, 0.0]}
    assert any("shelf layout" in p for p in Sidecar.from_dict(braced).problems())
    # without the layout the left push stays refused (spec §6.11: the right gripper pushes on the tabletop)
    plain = copy.deepcopy(d)
    del plain["contacts"]["push"]["layout"]
    assert any("right gripper pushes" in p for p in Sidecar.from_dict(plain).problems())


def test_shelf_pose_facts_keep_the_lean_sign():
    """The shelf pose facts name the side ('left' / 'right'); merged into the shelf start they must not replace the lean sign the
    push arm, the partner's lean and the goal side compute with (s3 touching#03 c6_k0: float('left') in shelf_push_sign)."""
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    for sign, name, arm in ((1.0, "left", "left"), (-1.0, "right", "right")):
        start = {"key": "hardback__x__u", "lean_side": sign, "lean_deg": 6.0}
        IS.merge_shelf_pose_facts(start, {"edge": "+u", "lean_side": name, "depth_m": 0.2})
        assert start["lean_side"] == sign and start["lean_side_name"] == name and start["edge"] == "+u"
        s = SimpleNamespace(shelf_start=start)
        assert IS.IORSSmall.shelf_push_sign(s) == -sign
        s.shelf_push_sign = lambda s=s: IS.IORSSmall.shelf_push_sign(s)
        assert IS.IORSSmall.shelf_push_arm(s) == arm


def test_the_side_board_ray_takes_only_a_face_of_the_supports_own_body():
    """tabletop.shelf_wall_lateral (s3 touching#03 c1 / c2 / c8): the ray's first hit must be the support's own body at least
    SHELF_RAY_MIN_HIT_M from the ray's start - a board face 0.30 m out gives +-0.30 m; a miss, a room wall behind the planks (c1 / c2)
    and a hit at distance 0 (c8: the edge midpoint inside a divider, the ray starts in it) give no board, and the leaning start is
    rejected (shelf_wall) instead of standing the book in the divider."""
    from ispatialgym.sampling.sim_utils import Reject, SupportTop
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.tests.test_v2_tabletop import BU_REST_XYZW

    edge = EdgeFrame("-v", th.tensor([0.0, 0.01]), th.tensor([0.0, 1.0]), 0.58)  # the robot faces +y: its left is -x
    support = SimpleNamespace(name="bookcase_x_0", links={"base_link": SimpleNamespace(prim_path="/World/bookcase_x_0/base_link")})
    board = "/World/bookcase_x_0/base_link/collisions/x__base_link__collision__3"
    hits = {"board": {"hit": True, "rigidBody": board, "distance": 0.30}, "miss": {"hit": False},
            "wall": {"hit": True, "rigidBody": "/World/scene_0/walls_abc_0/base_link", "distance": 0.12},
            "inside": {"hit": True, "rigidBody": board, "distance": 0.0}, "near": {"hit": True, "rigidBody": board, "distance": 0.0004}}
    rays = []

    def sampler(case):
        def raycast(starts, ends):
            rays.append((starts.clone(), ends.clone()))
            return [dict(hits[case])]
        return SimpleNamespace(world=SimpleNamespace(raycast=raycast, hull_at=_hull_at))

    for side in (1.0, -1.0):
        rays.clear()
        lat, fact = TT.shelf_wall_lateral(sampler("board"), edge, 0.12, 0.9, side, support)
        assert lat == pytest.approx(side * 0.30) and fact == {"hit": True, "body": board, "distance": 0.30}
        (start, end), = rays  # one horizontal ray from the edge midpoint 0.12 m in, outward along the side's tangent (-x = left)
        assert th.allclose(start, th.tensor([[0.0, 0.13, 0.9]])) and th.allclose(end, th.tensor([[-side * 0.59, 0.13, 0.9]]))
    for case in ("miss", "wall", "inside", "near"):
        lat, fact = TT.shelf_wall_lateral(sampler(case), edge, 0.12, 0.9, 1.0, support)
        assert lat is None, case
        assert ("starts inside" in fact.get("why", "")) == (case in ("inside", "near")), case
    xs, ys = th.arange(-0.29, 0.30, 0.02), th.arange(0.01, 0.30, 0.02)
    top = SupportTop("bookcase_x_0", xs, ys, th.ones(len(xs), len(ys), dtype=th.bool), th.zeros(len(xs), len(ys), dtype=th.bool), 0.81, 0.0)
    with pytest.raises(Reject) as e:
        TT.shelf_lean_pose(sampler("inside"), _book_hull(), th.tensor(BU_REST_XYZW), top, edge, support, 1.0)
    assert e.value.reason == "shelf_wall" and "starts inside" in str(e.value)
