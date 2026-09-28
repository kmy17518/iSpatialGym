"""The left arm as one more dimension of a contact's orientation choice (owner 2026-09-27, first-demo v2.2): the mirrored
reach band of the other arm (``ReachBands.mirrored`` / ``load_arm``), no simulator."""

import json
import math
import os
from types import SimpleNamespace
import tempfile

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.v2 import orient as O

R_IDX = th.tensor([M.ALL_JOINTS.index(n) for n in M.ARM_JOINTS["right"]])
L_IDX = th.tensor([M.ALL_JOINTS.index(n) for n in M.ARM_JOINTS["left"]])
R_LO = [-4.4506, -3.1416, -2.356196, -2.0944, -2.356196, -1.047198, -1.5708]
R_HI = [1.3090, 0.1745, 2.356196, 0.3491, 2.356196, 1.047198, 1.5708]
# the left arm's limits are the right arm's under the joint sign map
L_LO = [lo if s > 0 else -hi for s, lo, hi in zip(RB.MIRROR_JOINT_SIGNS, R_LO, R_HI)]
L_HI = [hi if s > 0 else -lo for s, lo, hi in zip(RB.MIRROR_JOINT_SIGNS, R_LO, R_HI)]


def _band_files(d, *, arm=None, mode="tabletop", cells=((0.48, -0.16), (0.56, -0.40), (0.64, 0.08)), pairs=((0, 1, 0), (0, 4, 1), (1, 0, 1), (2, -1, 3)),
                name="tabletop__s__t__apple-x__k", extra=None):
    meta = {"scene": "s", "support": "t", "object": "apple/x", "grasp_key": "apple__x__t", "generic_pinch": False, "lattice": 0.08, "mode": mode,
            "grasp_names": ["grasp_0", "grasp_1"], "stance": {"base_pose2d": [2.0, 1.0, math.pi / 2]}, "union_extent": [-0.40, 0.08],
            "trunks": [{"translate": 0.5, "tilt": 0.0, "collides": False, "posture": "ready", "shoulder_world": [1.8, 1.1, 1.3],
                        "idle_arm_q": {"left": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "right": [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]}},
                       {"translate": 0.95, "tilt": -0.9, "collides": True}],
            "feasible_sets": [{"trunk": [0.5, 0.0], "cell": [0.48, -0.16], "feasible": [[0.785, "grasp_0"], [3.142, "grasp_1"]]}], **(extra or {})}
    if arm is not None:
        meta["arm"] = arm
    np.savez_compressed(os.path.join(d, name + ".npz"), **{"t0.50_k+0.0": np.asarray(cells, dtype=np.float32),
                                                           "t0.50_k+0.0__pairs": np.asarray(pairs, dtype=np.int32).reshape(-1, 3)})
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(meta, f)
    return os.path.join(d, name + ".npz")


def test_mirror_joint_signs_map_the_ready_postures_onto_each_other():
    from ispatialgym.oracle import r1pro_model as M

    assert RB.mirror_arm_q(M.ARM_READY_Q["right"]) == pytest.approx(M.ARM_READY_Q["left"])
    assert RB.mirror_arm_q(M.ARM_READY_Q["left"]) == pytest.approx(M.ARM_READY_Q["right"])
    assert RB.other_arm("right") == "left" and RB.other_arm("left") == "right"


def test_mirrored_band_cells_yaws_extent_and_meta():
    with tempfile.TemporaryDirectory() as d:
        src = _band_files(d)
        rb = RB.ReachBands.load_arm("s", "t", "apple/x", "apple__x__t", arm="left", dirs=[d])
        assert rb.arm == "left" and rb.meta["mirrored_from"] == "right" and rb.meta["mirror_source"] == src and rb.path is None
        assert np.allclose(rb.band(0.5, 0.0), [[0.48, 0.16], [0.56, 0.40], [0.64, -0.08]])
        assert rb.union_extent() == pytest.approx((-0.08, 0.40)) and rb.meta["union_extent"] == pytest.approx([-0.08, 0.40])
        # yaw index k -> -k (mod 8), the generic marker (-1) kept, the grasp kept
        assert rb.pairs[(0.5, 0.0)].tolist() == [[0, 7, 0], [0, 4, 1], [1, 0, 1], [2, -1, 3]]
        assert sorted(rb.feasible(0.5, 0.0, (0.48, 0.16))) == sorted([(RB.YAWS[7], "grasp_0"), (RB.YAWS[4], "grasp_1")])
        assert rb.feasible(0.5, 0.0, (0.48, -0.16)) == [], "the source's cell is not in the mirror"
        t = rb.meta["trunks"][0]
        assert t["idle_arm_q"]["right"] == pytest.approx(RB.mirror_arm_q([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
        assert t["idle_arm_q"]["left"] == pytest.approx(RB.mirror_arm_q([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]))
        assert t["posture"] == "ready"
        # the shoulder moves to the other side of the band stance (base at (2, 1) facing +y: lateral = -dx)
        assert t["shoulder_world"] == pytest.approx([2.2, 1.1, 1.3])
        fs = rb.meta["feasible_sets"][0]
        assert fs["cell"] == pytest.approx([0.48, 0.16]) and fs["feasible"] == [[pytest.approx(2 * math.pi - 0.785, abs=1e-3), "grasp_0"],
                                                                                 [pytest.approx(2 * math.pi - 3.142, abs=1e-3), "grasp_1"]]
        # the posture record of the trunk and the source band itself are untouched
        again = RB.ReachBands.load("s", "t", "apple/x", "apple__x__t", dirs=[d])
        assert np.allclose(again.band(0.5, 0.0)[:, 1], [-0.16, -0.40, 0.08]) and again.arm == "right" and "mirrored_from" not in again.meta


def test_a_real_band_of_the_arm_wins_over_the_mirror():
    with tempfile.TemporaryDirectory() as d:
        _band_files(d)
        real = _band_files(d, arm="left", cells=((0.40, 0.24),), pairs=((0, 2, 0),), name="tabletop__s__t__apple-x__left")
        rb = RB.ReachBands.load_arm("s", "t", "apple/x", "apple__x__t", arm="left", dirs=[d])
        assert rb.path == real and "mirrored_from" not in rb.meta and np.allclose(rb.band(0.5, 0.0), [[0.40, 0.24]])
        # the right arm's own band is read as before
        assert RB.ReachBands.load_arm("s", "t", "apple/x", "apple__x__t", arm="right", dirs=[d]).arm == "right"


def test_floor_bands_mirror_and_other_kinds_never_do():
    with tempfile.TemporaryDirectory() as d:
        _band_files(d, mode="floor", name="floor__s__t__apple-x__k")
        assert RB.ReachBands.load_arm("s", "t", "apple/x", "apple__x__t", arm="left", dirs=[d]).meta["mirrored_from"] == "right"
    for mode in ("brace", "handle"):
        assert not RB.mirrorable({"mode": mode})
    assert not RB.mirrorable({"mode": "tabletop", "contact_set": "generic_contact"})
    with tempfile.TemporaryDirectory() as d:
        _band_files(d, arm="left", extra={"contact_set": "generic_contact", "height_class": 0.1}, name="contact__s__t__k")
        assert RB.ReachBands.load_arm("s", "t", arm="left", contact_set="generic_contact", height=0.1, dirs=[d]).arm == "left"
        with pytest.raises(FileNotFoundError):
            RB.ReachBands.load_arm("s", "t", arm="right", contact_set="generic_contact", height=0.1, dirs=[d])
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(FileNotFoundError):
            RB.ReachBands.load_arm("s", "t", "apple/x", "apple__x__t", arm="left", dirs=[d])


def test_a_mirrored_band_keeps_its_feasible_sets_in_memory():
    import torch as th

    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.sampling.v2.stances import Stance

    with tempfile.TemporaryDirectory() as d:
        src = _band_files(d)
        before = (open(src, "rb").read(), open(src[:-4] + ".json").read())
        rb = RB.ReachBands.load_arm("s", "t", "apple/x", "apple__x__t", arm="left", dirs=[d])
        st = Stance("manipulation", (2.0, 1.0, math.pi / 2), 0.5, 0.0)
        st.focus = th.tensor([2.0 - 0.16, 1.0 + 0.48, 0.8])  # base (0.48, +0.16)
        TT.store_feasible_set(rb, st, [(RB.YAWS[7], type("G", (), {"name": "grasp_0"})())])
        assert rb.meta["feasible_sets"][-1]["cell"] == pytest.approx([0.48, 0.16])
        assert (open(src, "rb").read(), open(src[:-4] + ".json").read()) == before
        assert sorted(os.listdir(d)) == sorted([os.path.basename(src), os.path.basename(src)[:-4] + ".json"]), "no band file was written"


# ------------------------------------------------------------------------------------------------------------------
# the chooser with arms
# ------------------------------------------------------------------------------------------------------------------


def _ikc2():
    """A real IKChecker on a stub kit that knows both arms with the r1pro.urdf limits (the left's under the sign map)."""
    from ispatialgym.sampling.v2.ik import IKChecker

    lo, hi = th.full((28,), -10.0), th.full((28,), 10.0)
    lo[R_IDX], hi[R_IDX] = th.tensor(R_LO), th.tensor(R_HI)
    lo[L_IDX], hi[L_IDX] = th.tensor(L_LO), th.tensor(L_HI)
    kit = SimpleNamespace(robot=SimpleNamespace(arm_control_idx={"right": R_IDX, "left": L_IDX}, joint_lower_limits=lo, joint_upper_limits=hi), stats={})
    return IKChecker(kit)


def _q(right=None, left=None):
    q = th.zeros(28)
    if right is not None:
        q[R_IDX] = th.tensor(right, dtype=th.float32)
    if left is not None:
        q[L_IDX] = th.tensor(left, dtype=th.float32)
    return q


def _chain(x=0.6, lat=-0.05):
    eef = (th.tensor([x, lat, 0.85]), C.eef_quat([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]))
    return [C.pre_pose(eef), eef]


# the untucked-like starting postures: the left one the right one under the sign map
START_R = [0.0, -1.0, 0.0, -1.5, 0.3, 0.0, 0.0]
START_L = RB.mirror_arm_q(START_R)


def _arm_props(chain=None):
    chain = chain or _chain()
    turns = lambda: [O.Proposal("row", 0, "row", chain), O.Proposal("twin", 1, "twin", [C.twin_pose(p) for p in chain])]  # noqa: E731
    return O.with_arms([("right", turns()), ("left", turns())])


class _ArmSolver:
    """``evaluate`` over both arms: ``answer[(arm, turn)]`` = the arm's chain configurations (None = its IK fails)."""

    def __init__(self, ikc, answer):
        self.ikc, self.answer, self.calls = ikc, answer, []

    def __call__(self, p):
        turn = p.label.split(":")[-1]
        self.calls.append((p.arm, turn, p.q_first is not None))
        qs = self.answer.get((p.arm, turn)) if p.q_first is None else [p.q_first] * len(p.chain)
        if qs is None:
            return False, None, None, "exact IK failed at pose 1 of 2"
        for pose, q in zip(p.chain, qs):
            self.ikc.solutions[(p.arm, self.ikc._pose_key(pose))] = q.clone()
        return True, qs[-1], None, ""


def _start():
    return {"right": _q(right=START_R), "left": _q(left=START_L)}


def test_with_arms_ranks_every_proposal_of_the_default_arm_first():
    props = _arm_props()
    assert [(p.label, p.rank, p.arm) for p in props] == [("right:row", 0, "right"), ("right:twin", 1, "right"), ("left:row", 2, "left"), ("left:twin", 3, "left")]
    one = O.with_arms([("right", [O.Proposal("row", 0, "row", _chain())])])
    assert one[0].label == "row" and one[0].arm == "right", "one arm: the generator's labels as before"


def test_the_right_arm_wins_within_the_reach_tolerance():
    ikc = _ikc2()
    q_r = _q(right=[0.0, -1.0, 0.0, -1.5, 0.3 + 1.0, 0.0, 0.0])  # 1.0 rad from its start
    q_l = _q(left=RB.mirror_arm_q([0.0, -1.0, 0.0, -1.5, 0.3 + 0.8, 0.0, 0.0]))  # 0.8 rad from its own start
    ev = _ArmSolver(ikc, {("right", "row"): [q_r, q_r], ("left", "row"): [q_l, q_l]})
    chosen, dec = O.choose(ikc, "right", _arm_props(), ev, _start(), map_twins=False, check_camera=False)
    assert chosen.proposal.arm == "right" and chosen.proposal.label == "right:row"
    ac = dec["arm_choice"]
    assert ac == {"default": "right", "offered": ["right", "left"], "chosen": "right", "reason": "default", "against_default": False, "viable": ["right", "left"]}
    assert dec["against_default"] is False and {o["arm"] for o in dec["options"]} == {"right", "left"}


def test_the_left_arm_wins_on_reach_beyond_the_tolerance():
    ikc = _ikc2()
    q_r = _q(right=[0.0, -1.0, 0.0, -1.5, 0.3 + 1.5, 0.0, 0.0])
    q_l = _q(left=RB.mirror_arm_q([0.0, -1.0, 0.0, -1.5, 0.3 + 0.5, 0.0, 0.0]))
    ev = _ArmSolver(ikc, {("right", "row"): [q_r, q_r], ("left", "row"): [q_l, q_l]})
    chosen, dec = O.choose(ikc, "right", _arm_props(), ev, _start(), map_twins=False, check_camera=False, stage="grasp test")
    assert chosen.proposal.arm == "left" and chosen.reach == pytest.approx(0.5, abs=1e-4)
    ac = dec["arm_choice"]
    assert ac["chosen"] == "left" and ac["reason"] == "reach" and ac["against_default"] is True and dec["against_default"] is True
    st = ikc.kit.stats
    assert st["orient_arm_choices"] == 1 and st["orient_arm.left"] == 1 and st["orient_arm_reason.reach"] == 1


def test_a_joint_limit_safe_arm_comes_before_an_unsafe_one_whatever_the_reach():
    ikc = _ikc2()
    q_r = _q(right=[0.0, 0.15, 0.0, -1.5, 0.3, 0.0, 0.0])  # joint 2 25 mrad from its upper limit, no reach at all
    q_l = _q(left=RB.mirror_arm_q([0.0, -1.0, 0.0, -1.5, 0.3 + 2.0, 0.0, 0.0]))  # safe, 2 rad of joint 5
    ev = _ArmSolver(ikc, {("right", "row"): [q_r, q_r], ("left", "row"): [q_l, q_l]})
    chosen, dec = O.choose(ikc, "right", _arm_props(), ev, _start(), map_twins=False, check_camera=False)
    assert chosen.proposal.arm == "left" and dec["arm_choice"]["reason"] == "safety"
    # neither arm safe: more clearance
    q_l2 = _q(left=RB.mirror_arm_q([0.0, 0.13, 0.0, -1.5, 0.3, 0.0, 0.0]))  # 45 mrad
    chosen2, dec2 = O.choose(ikc, "right", _arm_props(), _ArmSolver(ikc, {("right", "row"): [q_r, q_r], ("left", "row"): [q_l2, q_l2]}), _start(),
                             map_twins=False, check_camera=False)
    assert chosen2.proposal.arm == "left" and dec2["arm_choice"]["reason"] == "clearance"


def test_the_other_arm_when_the_default_arm_fails():
    ikc = _ikc2()
    q_l = _q(left=START_L)
    chosen, dec = O.choose(ikc, "right", _arm_props(), _ArmSolver(ikc, {("left", "row"): [q_l, q_l]}), _start(), map_twins=False, check_camera=False)
    assert chosen.proposal.arm == "left" and dec["arm_choice"]["reason"] == "only feasible" and dec["arm_choice"]["viable"] == ["left"]
    none, dec_none = O.choose(ikc, "right", _arm_props(), _ArmSolver(ikc, {}), _start(), map_twins=False, check_camera=False)
    assert none is None and dec_none["arm_choice"]["chosen"] is None and O.viable_arms(dec_none) == []


def test_the_segment_falls_back_to_the_other_arm_when_a_later_contact_fails():
    ikc = _ikc2()
    q_r = _q(right=START_R)
    q_l = _q(left=START_L)
    seen = []

    def place(s):  # the segment's later contact (the place): unreachable for the right arm
        seen.append(s.proposal.arm)
        return s.proposal.arm != "right", "place exact IK failed at pose 2 of 2"

    ev = _ArmSolver(ikc, {("right", "row"): [q_r, q_r], ("left", "row"): [q_l, q_l]})
    chosen, dec = O.choose(ikc, "right", _arm_props(), ev, _start(), map_twins=False, check_camera=False, downstream=place)
    assert seen == ["right", "left"] and chosen.proposal.arm == "left"
    right = next(o for o in dec["options"] if o["arm"] == "right" and o["label"] == "right:row")
    assert not right["ok"] and right["why"].startswith("downstream: place")
    assert dec["arm_choice"]["reason"] == "only feasible" and dec["arm_choice"]["against_default"] is True


def test_each_arm_is_scored_from_its_own_starting_posture():
    ikc = _ikc2()
    q_l = _q(left=START_L)
    # from one shared configuration the left arm would look 1.5 rad away (its joints at zero there); from its own start it is 0
    chosen, dec = O.choose(ikc, "right", _arm_props(), _ArmSolver(ikc, {("left", "row"): [q_l, q_l]}), _start(), map_twins=False, check_camera=False)
    assert chosen.reach == pytest.approx(0.0, abs=1e-6)
    assert O.reach(ikc, q_l, _q(right=START_R), "left") == pytest.approx(1.5, abs=1e-6)


# ------------------------------------------------------------------------------------------------------------------
# stance first: phase 2 with both arms' bands, ranked by whichever arm reaches better
# ------------------------------------------------------------------------------------------------------------------


def _phase2_stub(monkeypatch):
    from ispatialgym.sampling.v2 import tabletop as TT

    monkeypatch.setattr(TT, "camera_fk", lambda s=None, hand_below=None: SimpleNamespace(tilt_limit=lambda tr: 0.0))
    s = SimpleNamespace(kit=SimpleNamespace(trav_free=lambda xy: True, camera_poses_batch=lambda qs: [None] * len(qs), floor_z=0.0),
                        ikc=SimpleNamespace(stances_collision_free=lambda cands: th.ones(len(cands), dtype=th.bool)), stance_q=lambda c: th.zeros(28),
                        world=SimpleNamespace(camera_model=SimpleNamespace(angle_off_axis=lambda cam, focus: 0.0, all_inside=lambda cam, pts, margin: True)))
    s._arm_len = {"right": 0.80, "left": 0.80}  # the stretched arm, both arms (reach_margin's cache)
    top = SimpleNamespace(contains=lambda c, erosion: True, z=0.8)
    return TT, s, top


def _bands_pair(d):
    # the right band: a cell on the robot's right; its mirror is the left band; plus a cell far out on the right only
    _band_files(d, cells=((0.56, -0.24), (0.56, -0.56)), pairs=((0, 0, 0), (1, 0, 0)))
    rb = RB.ReachBands.load("s", "t", "apple/x", "apple__x__t", dirs=[d])
    return rb, RB.ReachBands.load_arm("s", "t", "apple/x", "apple__x__t", arm="left", dirs=[d])


def test_phase2_offers_the_foci_of_both_arms_and_tags_their_arms(monkeypatch):
    from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

    TT, s, top = _phase2_stub(monkeypatch)
    edge = EdgeFrame.from_edge("+x", th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]), 1.2)
    init = Stance("init", (0.55, 0.0, 0.0), 0.5, 0.0, 0.45, edge)
    with tempfile.TemporaryDirectory() as d:
        rb, lb = _bands_pair(d)
        kw = dict(target_height=0.06, stand_offs=(0.45,), translates=(0.5,), n_tilts=1, bands=rb, top=top, focus_z=0.83)
        only_right = TT.phase2_manipulation_stance(s, init, 0.03, **kw)
        both = TT.phase2_manipulation_stance(s, init, 0.03, arm_bands={"right": rb, "left": lb}, **kw)
    lat = lambda c: round(float(c.world_to_base(c.focus[:2])[1]), 2)  # noqa: E731
    assert sorted(lat(c) for c in only_right) == [-0.56, -0.24] and all("focus_arms" not in c.extras for c in only_right)
    assert {lat(c): c.extras["focus_arms"] for c in both} == {-0.56: ["right"], -0.24: ["right"], 0.24: ["left"], 0.56: ["left"]}


def test_phase2_reach_order_ranks_by_whichever_arm_reaches_better(monkeypatch):
    from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

    TT, s, top = _phase2_stub(monkeypatch)
    edge = EdgeFrame.from_edge("+x", th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]), 1.2)
    init = Stance("init", (0.55, 0.0, 0.0), 0.5, 0.0, 0.45, edge)
    with tempfile.TemporaryDirectory() as d:
        rb, lb = _bands_pair(d)
        kw = dict(target_height=0.06, stand_offs=(0.45,), translates=(0.5,), n_tilts=1, bands=rb, top=top, focus_z=0.83, order="reach")
        both = TT.phase2_manipulation_stance(s, init, 0.03, arm_bands={"right": rb, "left": lb}, **kw)
    lat = lambda c: round(float(c.world_to_base(c.focus[:2])[1]), 2)  # noqa: E731
    order = [lat(c) for c in both]
    # the near cells (|λ| 0.24) come first whichever side they are on: each is scored by the arm on its side
    assert set(order[:2]) == {-0.24, 0.24} and set(order[2:]) == {-0.56, 0.56}
    near_left = next(c for c in both if lat(c) == 0.24)
    m_left, arm = TT.best_reach_margin(s, near_left, near_left.focus, ["right", "left"])
    assert arm == "left" and m_left > TT.reach_margin(s, near_left, near_left.focus, "right") + 0.05
    # scored by the right arm alone (the order before) the same stance ranks behind the right arm's near cell
    near_right = next(c for c in both if lat(c) == -0.24)
    assert TT.reach_margin(s, near_left, near_left.focus, "right") < TT.reach_margin(s, near_right, near_right.focus, "right") - 0.05


# ------------------------------------------------------------------------------------------------------------------
# phase 3 with both arms and the segment's fallback
# ------------------------------------------------------------------------------------------------------------------


def _grasp_row(name="grasp_4"):
    from ispatialgym.sampling import grasp_library as GL

    return GL.Grasp("k", name, th.tensor([0.0, 0.0, 0.12]), C.eef_quat([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]), [0.05, 0.02])


def _two_arm_sampler(answer, *, mirrored_left=None, twins=False):
    """A sampler stand-in for ``phase3_target`` / ``pick_twin`` over both arms: a real IKChecker on a two-arm kit; ``q_locked`` gives
    each arm its starting posture (the working arm's); ``exact`` / ``feasible_sets`` answered per arm by ``answer[arm]`` (the chain's
    end configuration, None = fails); stances are always collision-free."""
    from ispatialgym.sampling.v2.ik import IKResult

    ikc = _ikc2()
    s = SimpleNamespace(kit=ikc.kit, ikc=ikc, calls=[], gen=lambda key, default=None: (twins if key == "grasp_twins" else default))
    s.__dict__["bands_by_arm"] = {}
    starts = {"right": _q(right=START_R, left=START_L), "left": _q(right=START_R, left=START_L)}
    ikc.q_locked = lambda stance, **kw: starts[(stance.working_arms or ["right"])[0]].clone()
    ikc.stances_collision_free = lambda sts: th.ones(len(sts), dtype=th.bool)

    def exact(poses, stance, arm="right", **kw):
        s.calls.append((arm, (stance.working_arms or ["?"])[0]))
        q = answer.get(arm)
        if q is None:
            return IKResult(False, None, "exact IK failed at pose 1 of 2", len(poses))
        for p in poses:
            ikc.solutions[(arm, ikc._pose_key(p))] = q.clone()
        return IKResult(True, q.clone(), "", len(poses))

    ikc.exact = exact
    s.world = SimpleNamespace(bottom_offset=lambda hull, quat: 0.02, footprint=lambda hull, pose: th.zeros(4, 2))
    return s


def _manip():
    from ispatialgym.sampling.v2.stances import Stance

    m = Stance("manipulation", (0.0, 0.0, 0.0), 0.5, 0.0, 0.45)
    m.focus, m.arms = th.tensor([0.6, 0.0, 0.8]), ["right"]
    return m


def test_phase3_takes_the_arm_the_rule_prefers_and_switches_the_stance():
    from ispatialgym.sampling.v2 import tabletop as TT

    g = _grasp_row()
    q_r = _q(right=[0.0, -1.0, 0.0, -1.5, 0.3 + 1.5, 0.0, 0.0], left=START_L)  # the right arm 1.5 rad of wrist from its start
    q_l = _q(right=START_R, left=RB.mirror_arm_q([0.0, -1.0, 0.0, -1.5, 0.3 + 0.4, 0.0, 0.0]))  # the left 0.4 rad
    s = _two_arm_sampler({"right": q_r, "left": q_l})
    s.ikc.feasible_sets = lambda cells, yaws, grasps, pose_at, stance, arm, ignore, eef_of: {0: [(0.0, g)]}
    manip = _manip()
    top = SimpleNamespace(z=0.75, polygon_supported=lambda fp, margin=0.0: True)
    with tempfile.TemporaryDirectory() as d:
        _band_files(d)
        rb = RB.ReachBands.load("s", "t", "apple/x", "apple__x__t", dirs=[d])
        lb = RB.ReachBands.load_arm("s", "t", "apple/x", "apple__x__t", arm="left", dirs=[d])
        pose, rec = TT.phase3_target(s, manip, np.random.default_rng(0), [g], th.zeros(8, 3), C.IDENTITY_QUAT, [], top=top, object_name="apple.n.01_1", n_positions=1,
                                     arms=["right", "left"], arm_bands={"right": rb, "left": lb})
    assert rec.arm == "left" and manip.arms == ["left"], "the stance now has the left arm working, the right idle"
    ac = rec.extras["twin_choice"]["arm_choice"]
    assert ac["chosen"] == "left" and ac["reason"] == "reach" and ac["against_default"] is True and ac["viable"] == ["left", "right"]
    assert rec.extras["arm_bands"]["left"]["mirrored_from"] == "right" and rec.extras["arm_bands"]["right"]["arm"] == "right"
    st = s.kit.stats
    assert st["arm_band_mirrored.foci"] == 1 and st["arm_band_mirrored.ik_tried"] == 1 and "arm_band_mirrored.ik_failed" not in st
    assert ("left", "left") in s.calls, "the left chain was solved at the stance with the left arm working"
    summ = TT.segment_arm_summary(rec, ["right", "left"])
    assert summ == {"arm": "left", "offered": ["right", "left"], "reason": "reach", "against_default": True, "fallback": None, "mirrored_band": True}
    assert TT.fallback_arms(rec) == ["right"]


def test_phase3_keeps_the_right_arm_on_a_near_tie():
    from ispatialgym.sampling.v2 import tabletop as TT

    g = _grasp_row()
    q_r = _q(right=[0.0, -1.0, 0.0, -1.5, 0.3 + 0.6, 0.0, 0.0], left=START_L)
    q_l = _q(right=START_R, left=RB.mirror_arm_q([0.0, -1.0, 0.0, -1.5, 0.3 + 0.4, 0.0, 0.0]))
    s = _two_arm_sampler({"right": q_r, "left": q_l})
    s.ikc.feasible_sets = lambda cells, yaws, grasps, pose_at, stance, arm, ignore, eef_of: {0: [(0.0, g)]}
    manip = _manip()
    top = SimpleNamespace(z=0.75, polygon_supported=lambda fp, margin=0.0: True)
    _, rec = TT.phase3_target(s, manip, np.random.default_rng(0), [g], th.zeros(8, 3), C.IDENTITY_QUAT, [], top=top, object_name="apple.n.01_1", n_positions=1,
                              arms=["right", "left"])
    assert rec.arm == "right" and manip.arms == ["right"] and rec.extras["twin_choice"]["arm_choice"]["reason"] == "default"


def test_the_segment_fallback_takes_the_same_pinch_with_the_other_arm():
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.sampling.v2.stances import GraspRecord

    g = _grasp_row()
    q_l = _q(right=START_R, left=START_L)
    s = _two_arm_sampler({"right": None, "left": q_l})
    manip = _manip()
    obj = (th.tensor([0.6, 0.0, 0.77]), C.IDENTITY_QUAT)
    from ispatialgym.sampling.v2 import ik as IK

    e = IK.grasp_pose(obj, g.pose)
    rec = GraspRecord("apple.n.01_1", "right", "k", "grasp_4", 0.1, IK.pre_pose(e), e, finger_q=[0.05, 0.02], eef_in_object=g.pose)
    new = TT.segment_fallback_grasp(s, manip, obj, rec, [g], "left", ignore_for_grasp=[], support_z=0.75, why="[place_unreachable] the right arm's place")
    assert new is not None and new.arm == "left" and manip.arms == ["left"] and new.grasp_name == "grasp_4" and new.yaw == 0.1
    assert new.extras["segment_fallback"] == {"from_arm": "right", "from_grasp": "grasp_4", "why": "[place_unreachable] the right arm's place"}
    assert TT.segment_arm_summary(new, ["right", "left"])["reason"] == "segment fallback"
    assert s.kit.stats["segment_fallbacks"] == 1 and s.kit.stats["segment_fallback_taken.left"] == 1
    # the other arm cannot take it either: no record, the stance unchanged
    s2 = _two_arm_sampler({"right": None, "left": None})
    manip2 = _manip()
    assert TT.segment_fallback_grasp(s2, manip2, obj, rec, [g], "left", ignore_for_grasp=[], support_z=0.75) is None and manip2.arms == ["right"]


# ------------------------------------------------------------------------------------------------------------------
# the situation predicate and its fired / not fired record
# ------------------------------------------------------------------------------------------------------------------


def test_the_arm_offer_predicate_and_its_reasons():
    from ispatialgym.sampling.v2 import tabletop as TT

    def sampler(gen=None, joint=None, goal="(ontop ?target ?partner)"):
        gp = SimpleNamespace(roles={"target": SimpleNamespace(joint=joint)}, bddl_text=f"(:init ...) (:goal (and {goal}))")
        return SimpleNamespace(ikc=SimpleNamespace(default_working_arms=["right"]), gp=gp, gen=lambda k, d=None: (gen or {}).get(k, d))

    assert TT.arm_offer(sampler()) == (["right", "left"], "")
    assert TT.arm_offer(sampler(), other_arm_roles=("push", "brace")) == (["right"], "the other arm has a role besides the segment: push, brace")
    assert TT.arm_offer(sampler(joint="j_lid"))[0] == ["right"] and "not free" in TT.arm_offer(sampler(joint="j_lid"))[1]
    assert TT.arm_offer(sampler(goal="(attached ?target ?partner)"))[0] == ["right"]
    assert TT.arm_offer(sampler(gen={"other_arm": False})) == (["right"], "generation.other_arm is off")
    assert TT.arm_offer(sampler(), free=True) == (["right", "left"], "") and TT.offered_arms(sampler()) == ["right", "left"]
    left_default = sampler()
    left_default.ikc.default_working_arms = ["left"]
    assert TT.offered_arms(left_default) == ["left", "right"]


def test_the_arm_offer_record_says_fired_or_why_not():
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.sampling.v2.stances import GraspRecord

    rec = GraspRecord("apple.n.01_1", "left", "k", "grasp_4", 0.0)
    rec.extras = {"twin_choice": {"arm_choice": {"reason": "reach"}}, "arm_bands": {"left": {"mirrored_from": "right"}}}
    r = TT.arm_offer_record(["right", "left"], ["right", "left"], rec)
    assert r == {"fired": True, "template_arms": ["right", "left"], "arms": ["right", "left"], "arm": "left", "reason": "reach", "against_default": True,
                 "mirrored_band": True}
    rec_r = GraspRecord("apple.n.01_1", "right", "k", "grasp_4", 0.0)
    assert TT.arm_offer_record(["right", "left"], ["right"], rec_r)["not_fired"] == "the other arm's band does not cover the focus"
    assert TT.arm_offer_record(["right"], ["right"], rec_r, "the other arm has a role besides the segment: push")["not_fired"].startswith("the other arm has a role")


def test_the_executor_notes_the_arm_offer_of_the_segments_actions():
    from ispatialgym.oracle.executor import Executor

    offer = {"fired": True, "arm": "left", "reason": "reach"}
    stub = SimpleNamespace(sidecar=SimpleNamespace(grasps=[SimpleNamespace(object="apple.n.01_1", arm="left", extra={"arm_offer": offer})]))
    note = Executor._arm_offer_note(stub, SimpleNamespace(object="apple.n.01_1", arms=("left",)))
    assert note == offer and note is not offer
    assert Executor._arm_offer_note(stub, SimpleNamespace(object="apple.n.01_1", arms=("right",))) is None, "another arm's action"
    assert Executor._arm_offer_note(stub, SimpleNamespace(object=None, arms=())) is None, "a navigate / transition"
    stub.sidecar.grasps[0].extra = {}
    assert Executor._arm_offer_note(stub, SimpleNamespace(object="apple.n.01_1", arms=("left",))) is None, "a record from before v2.2"


# ------------------------------------------------------------------------------------------------------------------
# the arm ranked on the segment's whole held chain (tabletop.held_arms)
# ------------------------------------------------------------------------------------------------------------------

OBJ = "brussels_sprouts.n.01_1"


def _wit(m, q=None, joint="right_arm_joint7"):
    return {"joint_limit_margin_rad": m, "joint_limit_nearest": joint, **({"q": [float(v) for v in q.tolist()]} if q is not None else {})}


def _report(held, q_pre):
    return {"mode": "expand", "actions": [
        {"name": "grasp", "objects": [OBJ], "carried": None, "witnesses": {"pre_grasp": _wit(0.3, q_pre), "grasp": _wit(0.3)}},
        {"name": "lift", "objects": [OBJ], "carried": OBJ, "witnesses": {"lift": _wit(0.3)}},
        {"name": "place", "objects": [OBJ], "carried": OBJ, "witnesses": {"carry": _wit(held), "pre_place": _wit(held), "place": _wit(0.3)}},
        {"name": "retreat", "objects": [], "carried": None, "witnesses": {}}]}


def _held_arm_stub(held, *, viable=("right", "left"), switch=True, switch_why=None, recheck_why=None):
    """A sampler after the program check of a right-arm segment: ``held[arm]`` = the held-chain margin the program check reports for
    that arm, the pre-grasp witness 0.3 rad (right) / 0.3 rad (left) from each arm's own start."""
    from ispatialgym.sampling.v2.stances import GraspRecord, SidecarDraft, Stance, Transition

    ikc = _ikc2()
    starts = {"right": _q(right=START_R, left=START_L), "left": _q(right=START_R, left=START_L)}
    ikc.q_locked = lambda stance, **kw: starts[(stance.working_arms or ["right"])[0]].clone()
    ikc.stances_collision_free = lambda sts: th.ones(len(sts), dtype=th.bool)
    q_pre = {"right": _q(right=[v + (0.3 if i == 4 else 0.0) for i, v in enumerate(START_R)], left=START_L),
             "left": _q(right=START_R, left=RB.mirror_arm_q([v + (0.3 if i == 4 else 0.0) for i, v in enumerate(START_R)]))}
    opts = [{"label": f"{a}:grasp_2", "arm": a, "ok": a in viable, "source": "ik"} for a in ("right", "left")]
    rec = GraspRecord(OBJ, "right", "k", "grasp_2", 0.0, extras={"twin_choice": {"orient": {"arm": "right", "chosen": "right:grasp_2", "options": opts}},
                                                                 "arm_offer": {"fired": True, "template_arms": ["right", "left"], "arm": "right", "reason": "default"}})
    sc = SidecarDraft("a", "f")
    manip = Stance("manipulation", (0.0, 0.0, 0.0), 0.5, 0.0, arms=["right"])
    sc.add_stance(Stance("init", (0.0, -0.5, 0.0), 0.5, 0.0))
    sc.add_stance(manip)
    sc.add_grasp(rec)
    sc.add_transition(Transition("init", "manipulation", 0.45, [0.5, 0.0], [0.5, 0.0]))
    sc.ik_witnesses = ["as checked"]
    s = SimpleNamespace(kit=ikc.kit, ikc=ikc, sidecar=sc, carry_arm_q={}, gen=lambda k, d=None: d)
    draft = SimpleNamespace(summary={"arm": {"arm": "right", "reason": "default", "against_default": False}})
    calls = {"switch": [], "recheck": [], "rerun": 0}

    def arm_switch(d, g, arm):
        calls["switch"].append(arm)
        if switch_why:
            return switch_why
        TT_adopt(manip, arm)
        sc.transitions[:] = [Transition("init", "manipulation", 0.55, [0.5, 0.0], [0.5, 0.0])]
        return None

    def recheck(d, g, q_first=None):
        calls["recheck"].append((g.arm, g.grasp_name))
        g.extras["open_pads_clear_of_support"] = True
        return recheck_why

    def rerun():
        calls["rerun"] += 1
        sc.ik_witnesses = ["re-checked"]
        return _report(held[rec.arm], q_pre[rec.arm])

    s.turn_recheck, s.verify_carry_legs = recheck, (lambda d: None)
    if switch:
        s.arm_switch = arm_switch
    return s, rec, draft, _report(held["right"], q_pre["right"]), rerun, calls


def TT_adopt(stance, arm):
    stance.arms, stance.idle_posture = [arm], None


def test_held_arms_switches_to_the_other_arm_when_only_its_whole_chain_is_safe():
    from ispatialgym.sampling.v2 import tabletop as TT

    s, rec, draft, report, rerun, calls = _held_arm_stub({"right": 0.0177, "left": 0.18})
    out = TT.held_arms(s, draft, report, rerun)
    assert rec.arm == "left" and s.sidecar.stances["manipulation"].arms == ["left"] and calls["switch"] == ["left"] and calls["rerun"] == 1
    assert out["actions"][2]["witnesses"]["carry"]["joint_limit_margin_rad"] == 0.18, "the switched segment's report is kept"
    h = rec.extras["twin_choice"]["held_arm"]
    assert h["status"] == "switched" and h["chosen"] == "left" and h["reason"] == "safety" and h["against_default"] is True
    assert rec.extras["arm_offer"]["held_chain"] == {"fired": True, "status": "switched", "why": "safety"} and rec.extras["arm_offer"]["arm"] == "left"
    assert draft.summary["arm"]["arm"] == "left" and draft.summary["arm"]["reason"] == "held chain"
    assert s.kit.stats["held_arm_checks"] == 1 and s.kit.stats["held_arm_switched"] == 1


def test_held_arms_keeps_and_restores_the_segment_when_the_other_arm_is_not_better():
    from ispatialgym.sampling.v2 import tabletop as TT

    s, rec, draft, report, rerun, calls = _held_arm_stub({"right": 0.0177, "left": 0.0100})
    extras0, tr0 = rec.extras, list(s.sidecar.transitions)
    TT.held_arms(s, draft, report, rerun)
    assert rec.arm == "right" and s.sidecar.stances["manipulation"].arms == ["right"] and calls["switch"] == ["left"]
    assert rec.extras is extras0 and "open_pads_clear_of_support" not in rec.extras, "the re-check's writes went into the discarded copy"
    assert s.sidecar.transitions == tr0 and s.sidecar.ik_witnesses == ["as checked"] and draft.summary["arm"]["arm"] == "right"
    h = rec.extras["twin_choice"]["held_arm"]
    assert h["status"] == "kept" and h["chosen"] == "right" and h["reason"] == "clearance" and s.kit.stats["held_arm_kept"] == 1
    # the other arm fails its re-check (both finger assignments): kept, restored, the reason recorded
    s2, rec2, draft2, report2, rerun2, calls2 = _held_arm_stub({"right": 0.0177, "left": 0.18}, recheck_why="descent: exact IK failed at pose 2 of 2")
    TT.held_arms(s2, draft2, report2, rerun2)
    assert rec2.arm == "right" and rec2.grasp_name == "grasp_2" and s2.sidecar.stances["manipulation"].arms == ["right"] and calls2["rerun"] == 0
    assert calls2["recheck"] == [("left", "grasp_2"), ("left", "grasp_2~r180")]
    h2 = rec2.extras["twin_choice"]["held_arm"]
    assert h2["status"] == "kept" and "descent" in h2["why"] and "the other finger assignment" in h2["why"]


def test_held_arms_is_a_no_op_unless_its_predicate_fires():
    from ispatialgym.sampling.v2 import tabletop as TT

    s, rec, draft, report, rerun, calls = _held_arm_stub({"right": 0.20, "left": 0.30})
    assert TT.held_arms(s, draft, report, rerun) is report and calls == {"switch": [], "recheck": [], "rerun": 0}
    assert rec.extras["arm_offer"]["held_chain"] == {"fired": False, "why": "the held chain is joint-limit safe"} and "held_arm" not in rec.extras["twin_choice"]
    s2, rec2, draft2, report2, rerun2, calls2 = _held_arm_stub({"right": 0.0177, "left": 0.30}, viable=("right",))
    assert TT.held_arms(s2, draft2, report2, rerun2) is report2 and calls2["switch"] == []
    assert rec2.extras["arm_offer"]["held_chain"]["fired"] is False and "held_arm_checks" not in s2.kit.stats
    # a module without the arm switch: not offered (recorded), nothing touched
    s3, rec3, draft3, report3, rerun3, calls3 = _held_arm_stub({"right": 0.0177, "left": 0.30}, switch=False)
    TT.held_arms(s3, draft3, report3, rerun3)
    h3 = rec3.extras["twin_choice"]["held_arm"]
    assert h3["status"] == "not offered" and rec3.arm == "right" and calls3["recheck"] == [] and s3.kit.stats["held_arm_not_offered"] == 1
