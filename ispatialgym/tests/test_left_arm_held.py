"""The left hand as the held-object tasks' holding arm (owner 2026-09-27, first-demo v2.2; ``sampling/v2/tasks/held_object.py``):
the mirrored left hold, the right hold / cache key / cache entry / records unchanged, the left hold's own cache key and no B7
prior, complete controller targets for the idle arm, the hold arm choice by the orientation rule and the segment fallback of
the template's goal search. A real ``IKChecker`` on a stub kit with the ``r1pro.urdf`` limits (the left's under the joint sign
map); no simulator is launched (importing the task module pulls in the OmniGibson package)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.tasks import held_object as HO

R_IDX = th.tensor([M.ALL_JOINTS.index(n) for n in M.ARM_JOINTS["right"]])
L_IDX = th.tensor([M.ALL_JOINTS.index(n) for n in M.ARM_JOINTS["left"]])
R_LO = [-4.4506, -3.1416, -2.356196, -2.0944, -2.356196, -1.047198, -1.5708]
R_HI = [1.3090, 0.1745, 2.356196, 0.3491, 2.356196, 1.047198, 1.5708]
L_LO = [lo if s > 0 else -hi for s, lo, hi in zip(RB.MIRROR_JOINT_SIGNS, R_LO, R_HI)]
L_HI = [hi if s > 0 else -lo for s, lo, hi in zip(RB.MIRROR_JOINT_SIGNS, R_LO, R_HI)]
#: the mental-rotation right hold of the B7 survey (``visibility.json`` ``hold.right_arm_q``): joint 6 2 mrad from its limit
Q_HOLD_R = [-1.3579, -0.0523, 0.5563, -0.5480, -0.0928, -1.0451, 1.0943]
#: the survey's hold record (right hand, roll 15°)
SURVEY_HOLD = {"key": "19590eb8f1c9dcea", "robot": "r1pro", "depth_m": 0.5, "trunk": [0.5, 0.0], "roll_deg": 15.0, "approach": "base +y",
               "closing": "world down rolled 15.0 deg about the approach", "right_arm_q": Q_HOLD_R, "idle_arms": "tucked"}


def _camera(yaw: float):
    """A head camera 1.2 m up and 0.3 m ahead of the base, looking along base +x pitched 10° down (the USD camera looks along -z)."""
    base_q = C.quat_from_yaw(yaw)
    look = C.quat_rotate(base_q, th.tensor([math.cos(math.radians(10.0)), 0.0, -math.sin(math.radians(10.0))]))
    return C.quat_rotate(base_q, th.tensor([0.3, 0.0, 1.2])), C.quat_with_z_axis(-look, C.qmul(base_q, C.quat_from_euler(0.0, math.radians(-10.0), 0.0)))


def _q(right=None, left=None):
    q = th.zeros(28)
    if right is not None:
        q[R_IDX] = th.tensor(right, dtype=th.float32)
    if left is not None:
        q[L_IDX] = th.tensor(left, dtype=th.float32)
    return q


def _ikc(default_arms=("right",)):
    """A real IKChecker on a stub kit with both arms' limits; ``default_arms`` = core's ``generation.arm`` mapping."""
    from ispatialgym.sampling.v2.ik import IKChecker

    lo, hi = th.full((28,), -10.0), th.full((28,), 10.0)
    lo[R_IDX], hi[R_IDX] = th.tensor(R_LO), th.tensor(R_HI)
    lo[L_IDX], hi[L_IDX] = th.tensor(L_LO), th.tensor(L_HI)
    kit = SimpleNamespace(robot=SimpleNamespace(arm_control_idx={"right": R_IDX, "left": L_IDX}, joint_lower_limits=lo, joint_upper_limits=hi), stats={})
    return IKChecker(kit, default_working_arms=list(default_arms))


# ----------------------------------------------------------------------------------------------------------------
# the hold pose
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("roll", [15.0, 90.0])
@pytest.mark.parametrize("yaw", [0.0, 0.3, -2.0])
def test_the_left_hold_is_the_right_hold_turned_pi_about_the_closing_axis(roll, yaw):
    hc = HO.HoldConstants(0.50, 0.5, 0.0, roll)
    cam = _camera(yaw)
    (o_r, q_r), pad_r = HO.hold_eef_pose(cam, yaw, hc, "right")
    (o_l, q_l), pad_l = HO.hold_eef_pose(cam, yaw, hc, "left")
    y_b = C.quat_rotate(C.quat_from_yaw(yaw), th.tensor([0.0, 1.0, 0.0]))
    assert th.allclose(pad_l, pad_r, atol=1e-6) and th.allclose(C.pad_center((o_l, q_l)), pad_l, atol=1e-6)
    # approach base -y; the same closing axis, in the base x-z plane
    assert th.allclose(C.approach_axis(q_l), -y_b, atol=1e-6)
    c = C.closing_axis(q_r)
    assert th.allclose(C.closing_axis(q_l), c, atol=1e-6) and float(th.dot(c, y_b)) == pytest.approx(0.0, abs=1e-6)
    # the right hold turned pi about the closing axis through the pad centre
    turn = C.quat_from_axis_angle(c, math.pi)
    assert C.quat_angle(C.qmul(turn, q_r), q_l) == pytest.approx(0.0, abs=1e-5)
    assert th.allclose(pad_r + C.quat_rotate(turn, o_r - pad_r), o_l, atol=1e-6)
    # the mirror image of the right hold in the base x-z plane through the pad (axes reflected, the body y flipped back to a
    # right-handed frame) is the left hold's wrist twin: the exact mirror conjugate up to the twin
    mirror = th.eye(3) - 2.0 * th.outer(y_b, y_b)
    q_mirror = C.mat_to_quat(mirror @ C.quat_to_mat(q_r).float() @ th.diag(th.tensor([1.0, -1.0, 1.0])))
    assert C.quat_angle(q_mirror, C.twin_pose((o_l, q_l))[1]) == pytest.approx(0.0, abs=1e-5)
    assert th.allclose(o_r - 2.0 * float(th.dot(o_r - pad_r, y_b)) * y_b, o_l, atol=1e-6)


def test_the_right_hold_is_unchanged():
    """The default and ``arm="right"`` are the construction before the arm choice bit for bit, and the v1 runs' hold quaternion
    is reproduced."""
    for hc in (HO.HoldConstants(), HO.HoldConstants(0.50, 0.5, 0.0, 90.0)):
        for yaw in (0.0, 0.3, -2.0):
            cam = _camera(yaw)
            cam_pos, cam_quat = C.as_pose(cam)
            pad = cam_pos + C.quat_rotate(cam_quat, th.tensor([0.0, 0.0, -1.0])) * float(hc.depth_m)
            a = C.quat_rotate(C.quat_from_yaw(yaw), th.tensor([0.0, 1.0, 0.0]))
            c = C.quat_rotate(C.quat_from_axis_angle(a, math.radians(hc.roll_deg)), th.tensor([0.0, 0.0, -1.0]))
            legacy = ((pad - C.CONTACT_CENTER_Z * a, C.eef_quat(a, c)), pad)
            for (o, q), p in (HO.hold_eef_pose(cam, yaw, hc), HO.hold_eef_pose(cam, yaw, hc, "right")):
                assert th.equal(o, legacy[0][0]) and th.equal(q, legacy[0][1]) and th.equal(p, legacy[1])
    (_, q0), _ = HO.hold_eef_pose((th.tensor([0.3, 0.0, 1.2]), C.quat_with_z_axis(th.tensor([-1.0, 0.0, 0.0]), C.IDENTITY_QUAT)), 0.0, HO.HoldConstants(), "right")
    v1 = th.tensor([-0.7010573744773865, 0.09229596704244614, 0.09229596704244614, 0.7010573744773865])
    assert C.quat_angle(q0, v1) == pytest.approx(0.0, abs=1e-3)


# ----------------------------------------------------------------------------------------------------------------
# the hold cache
# ----------------------------------------------------------------------------------------------------------------


def _cam_in_zed():
    return th.tensor([0.01234, -0.04567, 0.07891]), th.tensor([0.5, -0.5, 0.5, -0.5])


def _legacy_key(hc, cam_in_zed) -> str:
    """The hold cache key as computed before the arm choice (``hold_cache_extras`` without an arm)."""
    cam = [[round(float(v), 4) for v in cam_in_zed[0]], [round(float(v), 4) for v in cam_in_zed[1]]]
    extras = {"idle_arms": "tucked", "contact_center_z": round(float(C.CONTACT_CENTER_Z), 5), "cam_in_zed": cam, "robot_model": "r1pro"}
    return hashlib.sha256(json.dumps({"robot": "r1pro", "hold": hc.to_json(), **extras}, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def test_the_right_hold_keeps_its_cache_key_and_the_left_hold_has_its_own():
    kit = SimpleNamespace(cam_in_zed=_cam_in_zed())
    for hc in (HO.HoldConstants(), HO.HoldConstants(0.50, 0.5, 0.0, 90.0)):
        right = HO.hold_cache_extras(kit, "right")
        assert HO.hold_cache_extras(kit) == right and "arm" not in right and "approach" not in right
        assert HO.hold_cache_key("r1pro", hc, right) == _legacy_key(hc, kit.cam_in_zed)
        left = HO.hold_cache_extras(kit, "left")
        assert left == {**right, "arm": "left", "approach": "base -y"}
        key_l = HO.hold_cache_key("r1pro", hc, left)
        assert key_l != _legacy_key(hc, kit.cam_in_zed) and len(key_l) == 16
    assert HO.hold_cache_key("r1pro", HO.HoldConstants(), HO.hold_cache_extras(kit, "left")) != HO.hold_cache_key("r1pro", HO.HoldConstants(0.5, 0.5, 0.0, 90.0),
                                                                                                                  HO.hold_cache_extras(kit, "left"))


class _HoldKit:
    """``solve_hold``'s kit: the IK puts a fixed configuration per arm (the left the right's mirror image) and remembers the pose it
    reached, FK returns it; nothing collides."""

    def __init__(self):
        self.cam_in_zed = _cam_in_zed()
        self.jidx = {n: M.ALL_JOINTS.index(n) for n in M.ALL_JOINTS}
        self.reached = {}
        self.ik_calls = []

    def ik(self, pose, q0, arm="right"):
        self.ik_calls.append(arm)
        q = q0.clone()
        q[R_IDX if arm == "right" else L_IDX] = th.tensor(Q_HOLD_R if arm == "right" else RB.mirror_arm_q(Q_HOLD_R), dtype=th.float32)
        self.reached[(arm, tuple(round(float(v), 4) for v in q[R_IDX if arm == "right" else L_IDX]))] = pose
        return q

    def eef_pose(self, q, arm="right"):
        return self.reached[(arm, tuple(round(float(v), 4) for v in q[R_IDX if arm == "right" else L_IDX]))]

    def collision_free(self, q, margin=0.01):
        return th.tensor([True])

    def ignoring(self, objs):
        return contextlib.nullcontext()


def _hold_sampler(tmp_path, monkeypatch, hc=None):
    """A ``MentalRotationV2`` whose ``solve_hold`` runs on :class:`_HoldKit` with the cache in ``tmp_path``."""
    s = object.__new__(HO.MentalRotationV2)
    s.kit = _HoldKit()
    s.robot = SimpleNamespace(arm_control_idx={"right": R_IDX, "left": L_IDX})
    s.world = SimpleNamespace(scene=SimpleNamespace(objects=[]))
    s.gp = SimpleNamespace(alias="mental_rotation-test")
    hc = hc or HO.HoldConstants()
    base3 = (th.tensor([1.0, 2.0, 0.3]), C.quat_from_yaw(0.3))
    s.hold_start = lambda: (hc, (1.0, 2.0, 0.3), base3, th.zeros(28), _camera(0.3))
    path = tmp_path / "held_hold.json"
    monkeypatch.setattr(HO, "HOLD_CACHE_PATH", str(path))
    return s, path


def test_the_right_hold_writes_its_cache_entry_as_before_and_the_left_hold_its_own(tmp_path, monkeypatch):
    s, path = _hold_sampler(tmp_path, monkeypatch)
    h_r, h_l = s.solve_hold("right"), s.solve_hold("left")
    cache = json.loads(path.read_text())
    assert set(cache) == {h_r.sol["key"], h_l.sol["key"]} and h_r.sol["key"] == _legacy_key(HO.HoldConstants(), _cam_in_zed())
    right = cache[h_r.sol["key"]]
    # the entry format before the arm choice: its keys in their order, the approach / closing words, the right arm's joints
    assert list(right) == ["key", "robot", "depth_m", "trunk", "roll_deg", "approach", "closing", "eef_in_base", "pad_in_base", "camera_in_base", "right_arm_q",
                           "idle_arms"]
    assert right["approach"] == "base +y" and right["closing"] == "world down rolled 15.0 deg about the approach"
    assert right["right_arm_q"] == pytest.approx(Q_HOLD_R, abs=1e-6)
    left = cache[h_l.sol["key"]]
    assert left["arm"] == "left" and left["approach"] == "base -y" and left["closing"] == "world down rolled -15.0 deg about the approach"
    assert "right_arm_q" not in left and left["left_arm_q"] == pytest.approx(RB.mirror_arm_q(Q_HOLD_R), abs=1e-6)
    assert h_r.sol["source"] == h_l.sol["source"] == "solved" and h_r.arm == "right" and h_l.arm == "left"
    # both are hits the next time (FK verified), no IK
    s.kit.ik_calls.clear()
    again_r, again_l = s.solve_hold("right"), s.solve_hold("left")
    assert again_r.sol["source"] == again_l.sol["source"] == "cache" and s.kit.ik_calls == []
    assert th.equal(again_r.q, h_r.q) and th.equal(again_l.q, h_l.q)
    # the left hold's eef in the base frame is the right one turned pi about the closing axis through the pad
    eef_r, eef_l = h_r.eef_in_base, h_l.eef_in_base
    c = C.closing_axis(eef_r[1])
    assert C.quat_angle(C.qmul(C.quat_from_axis_angle(c, math.pi), eef_r[1]), eef_l[1]) == pytest.approx(0.0, abs=1e-5)
    assert th.allclose(h_r.pad_in_base, h_l.pad_in_base, atol=1e-6)


# ----------------------------------------------------------------------------------------------------------------
# B7 prior, records, controller targets
# ----------------------------------------------------------------------------------------------------------------


def test_a_left_hold_never_uses_the_right_hand_survey(tmp_path):
    path = tmp_path / "visibility.json"
    rec = {"grasp_key": "banana__znakxm", "grasp_name": "grasp_0", "visible_fraction": 0.2, "in_frame": True}
    path.write_text(json.dumps({"hold": SURVEY_HOLD, "records": [rec]}))
    hc = HO.HoldConstants(0.50, 0.5, 0.0, 15.0)
    assert HO.survey_arm(SURVEY_HOLD) == "right"
    assert len(HO.load_visibility_prior(str(path), hc)) == len(HO.load_visibility_prior(str(path), hc, "right")) == 1
    assert HO.load_visibility_prior(str(path), hc, "left") == {} and not HO.survey_matches_hold(SURVEY_HOLD, hc, "left")
    # a survey rendered with the left hand would apply to the left hold only
    left_rec = {**{k: v for k, v in SURVEY_HOLD.items() if k != "right_arm_q"}, "approach": "base -y", "left_arm_q": RB.mirror_arm_q(Q_HOLD_R)}
    assert HO.survey_arm(left_rec) == "left" and HO.survey_matches_hold(left_rec, hc, "left") and not HO.survey_matches_hold(left_rec, hc, "right")
    assert HO.survey_arm({**SURVEY_HOLD, "arm": "left"}) == "left"


def test_the_repo_survey_serves_the_right_mental_rotation_hold_only():
    if not os.path.exists(HO.VISIBILITY_PRIOR_PATH):
        pytest.skip("no B7 survey in this checkout")
    hc = HO.HoldConstants(0.50, 0.5, 0.0, 15.0)
    assert len(HO.load_visibility_prior(HO.VISIBILITY_PRIOR_PATH, hc, "right")) > 0
    assert HO.load_visibility_prior(HO.VISIBILITY_PRIOR_PATH, hc, "left") == {}


def _sol(arm: str):
    """A hold solution record (``solve_hold``'s ``sol``) of ``arm``, roll 15°."""
    sol = {"key": "k", "source": "cache", "robot": "r1pro", **HO.HoldConstants().to_json(), "approach": "base +y",
           "closing": "world down rolled 15.0 deg about the approach", "eef_in_base": {"position": [0.70, 0.01, 1.20], "orientation": [-0.70, 0.09, 0.09, 0.70]},
           "pad_in_base": [0.70, 0.0, 1.20], "camera_in_base": {"position": [0.24, 0.0, 1.39], "orientation": [0.39, -0.39, -0.59, 0.59]}, "right_arm_q": Q_HOLD_R,
           "idle_arms": "tucked"}
    if arm == "left":
        sol = {**{k: v for k, v in sol.items() if k != "right_arm_q"}, "arm": "left", "approach": "base -y",
               "closing": "world down rolled -15.0 deg about the approach", "left_arm_q": RB.mirror_arm_q(Q_HOLD_R)}
    return sol


def _legacy_hold_record(sol):
    """The ``hold`` constraint ``record_constraints`` wrote before the arm choice, verbatim."""
    return {"scope": {}, "type": "hold", "robot": sol["robot"], "depth_m": sol["depth_m"], "trunk": list(sol["trunk"]), "roll_deg": sol["roll_deg"],
            "approach": sol["approach"], "closing": sol["closing"], "eef_in_base": sol["eef_in_base"], "right_arm_q": sol["right_arm_q"],
            "idle_arms": sol["idle_arms"], "hard": False}


def test_the_records_of_the_right_hold_are_unchanged_and_the_left_hold_names_its_arm():
    from ispatialgym.sampling import customs as CU

    assert list(HO.hold_posture("right").items()) == [("left_arm", "reset"), ("right_arm", "hold")]
    assert HO.hold_posture("left") == {"left_arm": "hold", "right_arm": "reset"}
    assert HO.hold_init_arms("right") == "left tucked (JoyLo reset), right hold" and HO.hold_init_arms("left") == "right tucked (JoyLo reset), left hold"
    hc = HO.HoldConstants()
    assert HO.hold_json(hc) == HO.hold_json(hc, "right") == hc.to_json()
    assert HO.hold_json(hc, "left") == {**hc.to_json(), "arm": "left", "approach": "base -y"}
    sol_r, sol_l = _sol("right"), _sol("left")
    legacy = _legacy_hold_record(sol_r)
    rec_r = HO.hold_record(sol_r)
    assert list(rec_r.items()) == list(legacy.items()) and HO.hold_record(sol_r, "right") == legacy
    rec_l = HO.hold_record(sol_l, "left")
    assert rec_l["arm"] == "left" and rec_l["left_arm_q"] == sol_l["left_arm_q"] and "right_arm_q" not in rec_l and rec_l["approach"] == "base -y"
    # the write-back dedupes per arm: a written right record (no arm field) is not the left hold's, and vice versa
    assert HO.same_hold(legacy, sol_r, "right") and not HO.same_hold(legacy, sol_l, "left")
    assert HO.same_hold(rec_l, sol_l, "left") and not HO.same_hold(rec_l, sol_r, "right")
    for rec in (rec_r, rec_l):
        c = CU.Constraint.from_dict(dict(rec, source={}))
        assert c.type == "hold" and c.scope == {} and c.to_dict() == dict(rec, source={})


def test_the_constraint_write_back_is_per_holding_arm():
    written = [_legacy_hold_record(_sol("right"))]
    s = object.__new__(HO.MentalRotationV2)
    s.gp = SimpleNamespace(constraints_for=lambda ctype: [r for r in written if r["type"] == ctype])
    s.choice = SimpleNamespace(models={"target": SimpleNamespace(category="banana", model="znakxm")})
    s.hold = HO.HoldConstants()
    draft = SimpleNamespace(summary={"grasp": {"key": "banana__znakxm", "name": "grasp_0"}, "goal": {"angle_deg": 45.0}})
    s.arm, s.hold_solution = "right", _sol("right")
    recs = s.record_constraints(draft)
    # the right hold is on record already; the grasp_set prior as before
    assert list(recs[0].items()) == list({"scope": {"target": "banana/znakxm"}, "type": "grasp_set", "trunk": [0.5, 0.0], "hold": True,
                                          "feasible": [[0.0, "grasp_0"]], "goal": {"angle_deg": 45.0}, "hard": False}.items()) and len(recs) == 1
    s.arm, s.hold_solution = "left", _sol("left")
    recs = s.record_constraints(draft)
    assert [r["type"] for r in recs] == ["hold", "grasp_set"] and recs[0] == HO.hold_record(_sol("left"), "left") and recs[1]["arm"] == "left"
    written.append(recs[0])
    assert [r["type"] for r in s.record_constraints(draft)] == ["grasp_set"]


def _robot():
    """A robot view with the R1Pro joint order and the trunk / arm index groups ``oracle.action`` reads."""
    names = list(M.ALL_JOINTS)
    ids = lambda js: th.tensor([names.index(j) for j in js])  # noqa: E731
    return SimpleNamespace(joints={n: None for n in names}, trunk_control_idx=ids(M.TORSO_JOINTS), arm_control_idx={a: ids(M.ARM_JOINTS[a]) for a in M.ARMS})


def test_the_start_records_complete_trunk_idle_arm_and_holding_arm_targets():
    from ispatialgym.oracle.action import apply_recorded_targets

    robot = _robot()
    state = SimpleNamespace(trunk_target=th.tensor([1.0, 2.0, 3.0, 4.0]), arm_target={"left": th.arange(7.0) + 10.0, "right": th.arange(7.0) + 20.0})
    legacy = {}
    for js, vals in ((M.TORSO_JOINTS, state.trunk_target), (M.ARM_JOINTS["left"], state.arm_target["left"]), (M.ARM_JOINTS["right"], state.arm_target["right"])):
        legacy.update({j: float(v) for j, v in zip(js, vals.tolist())})
    right = HO.start_controller_targets(robot, state, "right")
    assert list(right.items()) == list(legacy.items())  # trunk, the idle left arm, the holding right arm: as before
    left = HO.start_controller_targets(robot, state, "left")
    assert list(left) == M.TORSO_JOINTS + M.ARM_JOINTS["right"] + M.ARM_JOINTS["left"] and left == legacy
    for targets in (right, left):
        hold = SimpleNamespace(trunk_target=None, arm_target={})
        assert apply_recorded_targets(hold, robot, targets) == ["trunk", "arm_left", "arm_right"]
        assert th.equal(hold.arm_target["left"], state.arm_target["left"]) and th.equal(hold.arm_target["right"], state.arm_target["right"])
    # the loop before v2.2 named the left arm as the idle one: a left hold recorded no right-arm group, which the loader skips
    hardcoded = {M.ALL_JOINTS[int(i)]: 0.0 for idx in (robot.trunk_control_idx, robot.arm_control_idx["left"], robot.arm_control_idx["left"]) for i in idx}
    assert apply_recorded_targets(SimpleNamespace(trunk_target=None, arm_target={}), robot, hardcoded) == ["trunk", "arm_left"]


# ----------------------------------------------------------------------------------------------------------------
# the hold arm choice (orientation rule)
# ----------------------------------------------------------------------------------------------------------------


def _choose(ikc, holds, default="right", offered=("right", "left")):
    """``hold_arm_choice`` with ``holds[arm]`` = the arm's 7 hold joints (missing: its hold IK fails); both arms start tucked."""
    targets = {a: HO.hold_eef_pose(_camera(0.0), 0.0, HO.HoldConstants(), a)[0] for a in offered}
    calls = []

    def solve(arm):
        calls.append(arm)
        if arm not in holds:
            return None, None, f"[hold_pose] no {arm}-arm IK solution for the fixed hold"
        return f"{arm} hold", _q(**{arm: holds[arm]}), ""

    chosen, dec = HO.hold_arm_choice(ikc, default, list(offered), targets, {a: th.zeros(28) for a in offered}, solve)
    return chosen, dec, calls


def test_the_mirrored_holds_tie_and_the_default_right_arm_keeps_the_hold():
    ikc = _ikc()
    chosen, dec, calls = _choose(ikc, {"right": Q_HOLD_R, "left": RB.mirror_arm_q(Q_HOLD_R)})
    assert calls == ["right", "left"] and chosen.proposal.arm == "right" and chosen.result == "right hold"
    assert dec["arm_choice"] == {"default": "right", "offered": ["right", "left"], "chosen": "right", "reason": "default", "against_default": False,
                                 "viable": ["right", "left"]}
    opts = {o["arm"]: o for o in dec["options"]}
    assert opts["right"]["margin_rad"] == opts["left"]["margin_rad"] == pytest.approx(0.0021, abs=2e-4)  # neither joint-limit safe
    assert opts["right"]["nearest_joint"] == "right_arm_joint6" and opts["left"]["nearest_joint"] == "left_arm_joint6"
    assert opts["right"]["reach_rad"] == opts["left"]["reach_rad"] == pytest.approx(1.358, abs=1e-3)
    rec = HO.hold_arm_record("right", "right", ["right", "left"], dec)
    assert (rec["arm"], rec["reason"], rec["against_default"]) == ("right", "default", False) and "segment_fallback" not in rec
    assert ikc.kit.stats["orient_arm.right"] == 1 and ikc.kit.stats["orient_arm_reason.default"] == 1


def test_the_left_arm_takes_the_hold_when_the_rule_prefers_it():
    # the right hold has no IK solution: the left one is the only feasible
    chosen, dec, _ = _choose(_ikc(), {"left": RB.mirror_arm_q(Q_HOLD_R)})
    assert chosen.proposal.arm == "left" and dec["arm_choice"]["reason"] == "only feasible" and dec["arm_choice"]["against_default"] is True
    assert dec["arm_choice"]["viable"] == ["left"]
    # a joint-limit safe left hold before an unsafe right one
    safe_l = RB.mirror_arm_q(Q_HOLD_R[:5] + [-0.90] + Q_HOLD_R[6:])
    chosen, dec, _ = _choose(_ikc(), {"right": Q_HOLD_R, "left": safe_l})
    assert chosen.proposal.arm == "left" and dec["arm_choice"]["reason"] == "safety"
    # neither safe: more clearance decides, also by a few mrad (the right hold's joint 6 at 2 mrad, the left's at 47 mrad)
    clear_l = RB.mirror_arm_q(Q_HOLD_R[:5] + [-1.0] + Q_HOLD_R[6:])
    chosen, dec, _ = _choose(_ikc(), {"right": Q_HOLD_R, "left": clear_l})
    assert chosen.proposal.arm == "left" and dec["arm_choice"]["reason"] == "clearance"
    rec = HO.hold_arm_record("left", "right", ["right", "left"], dec)
    assert (rec["arm"], rec["reason"], rec["against_default"]) == ("left", "clearance", True)


def test_generation_arm_left_makes_the_left_arm_the_default():
    """``generation.arm: left`` -> core's ``default_working_arms = ["left"]`` -> ``tabletop.default_arm`` = left: the left arm wins the tie."""
    from ispatialgym.sampling.v2 import tabletop as TT

    s = SimpleNamespace(ikc=_ikc(["left"]), gen=lambda key, default=None: default)
    assert TT.default_arm(s) == "left" and TT.offered_arms(s, free=True) == ["left", "right"]
    chosen, dec, calls = _choose(s.ikc, {"right": Q_HOLD_R, "left": RB.mirror_arm_q(Q_HOLD_R)}, default="left", offered=("left", "right"))
    assert calls == ["left", "right"] and chosen.proposal.arm == "left"
    assert dec["arm_choice"]["reason"] == "default" and dec["arm_choice"]["against_default"] is False
    s_off = SimpleNamespace(ikc=_ikc(["left"]), gen=lambda key, default=None: False if key == "other_arm" else default)
    assert TT.offered_arms(s_off, free=True) == ["left"]


def test_a_single_offered_arm_is_only_evaluated():
    chosen, dec, calls = _choose(_ikc(), {"right": Q_HOLD_R}, offered=("right",))
    assert calls == ["right"] and chosen.proposal.arm == "right" and chosen.proposal.label == "hold" and "arm_choice" not in dec
    rec = HO.hold_arm_record("right", "right", ["right"], dec)
    assert (rec["arm"], rec["reason"], rec["against_default"]) == ("right", "only arm offered", False)
    none, dec_none, _ = _choose(_ikc(), {})
    assert none is None and dec_none["arm_choice"]["chosen"] is None


# ----------------------------------------------------------------------------------------------------------------
# the template: arm choice, goal search, segment fallback
# ----------------------------------------------------------------------------------------------------------------


def _template(tmp_path, monkeypatch, holds, goal_ok, default_arms=("right",), generation=None):
    """A ``MentalRotationV2`` whose hold solve and goal search are stubbed: ``holds[arm]`` = the arm's hold joints (missing: no
    IK), ``goal_ok(arm)`` = whether the goal search with that arm's hold leaves options. The B7 survey is the right hand's."""
    survey = tmp_path / "visibility.json"
    survey.write_text(json.dumps({"hold": SURVEY_HOLD, "records": [{"grasp_key": "banana__znakxm", "grasp_name": "grasp_0", "visible_fraction": 0.9,
                                                                    "in_frame": True}]}))
    monkeypatch.setattr(HO, "VISIBILITY_PRIOR_PATH", str(survey))
    s = object.__new__(HO.MentalRotationV2)
    s.ikc = _ikc(default_arms)
    s.kit = s.ikc.kit
    s.robot, s.world = s.kit.robot, SimpleNamespace(scene=SimpleNamespace(objects=["table"]))
    s.spec = SimpleNamespace(generation=dict(generation or {}))
    s.gp = SimpleNamespace(alias="mental_rotation-test")
    s.template_meta, s.template_facts = {"init_arms": "untucked"}, {}
    s.arm, s.hold_arm, s.hold, s.prior = HO.ARM, {}, HO.HoldConstants(), {}
    pose = (th.zeros(3), C.IDENTITY_QUAT.clone())
    s.hold_start = lambda: (s.hold, (0.0, 0.0, 0.0), pose, th.zeros(28), _camera(0.0))
    log = {"solved": [], "searched": [], "scene_removed": 0}

    @contextlib.contextmanager
    def ignoring(objs):
        log["scene_removed"] += 1
        yield

    s.kit.ignoring = ignoring

    def solve_hold(arm, scene_removed=False):
        assert scene_removed, "every offered arm's hold is solved in the template's one robot-only block"
        log["solved"].append(arm)
        if arm not in holds:
            raise Reject("hold_pose", f"no {arm}-arm IK solution for the fixed hold")
        q = _q(**{arm: holds[arm]})
        return HO.HoldSolution(arm, q, q.clone(), pose, th.zeros(3), pose, {"key": f"{arm}-key", "robot": "r1pro", f"{arm}_arm_q": list(holds[arm])})

    def precompute_options():
        log["searched"].append(s.arm)
        s.template_facts["goal_options"] = {"arm": s.arm, "feasible": 5 if goal_ok(s.arm) else 0}
        if not goal_ok(s.arm):
            raise Reject("goal_infeasible", f"no library grasp has a visible hold with a feasible goal ({s.arm})")

    s.solve_hold, s.precompute_options = solve_hold, precompute_options
    return s, log


MIRRORED = {"right": Q_HOLD_R, "left": RB.mirror_arm_q(Q_HOLD_R)}


def test_the_template_keeps_the_right_hold_with_its_prior(tmp_path, monkeypatch):
    s, log = _template(tmp_path, monkeypatch, MIRRORED, lambda arm: True)
    s.choose_hold()
    assert s.arm == "right" and log == {"solved": ["right", "left"], "searched": ["right"], "scene_removed": 1}
    assert s.hold_solution == {"key": "right-key", "robot": "r1pro", "right_arm_q": Q_HOLD_R} and len(s.prior) == 1
    assert s.template_meta["init_arms"] == "left tucked (JoyLo reset), right hold"
    fact = s.template_facts["hold_arm"]
    assert (fact["arm"], fact["reason"], fact["against_default"], fact["offered"]) == ("right", "default", False, ["right", "left"])
    assert fact["arm_choice"]["chosen"] == "right" and "segment_fallback" not in fact
    assert s.arm_summary() == {"arm": "right", "reason": "default", "against_default": False}


def test_the_template_falls_back_to_the_other_arm_when_its_goal_search_leaves_nothing(tmp_path, monkeypatch):
    s, log = _template(tmp_path, monkeypatch, MIRRORED, lambda arm: arm == "left")
    s.choose_hold()
    assert s.arm == "left" and log == {"solved": ["right", "left"], "searched": ["right", "left"], "scene_removed": 1}
    assert s.hold_solution["key"] == "left-key" and s.prior == {}, "a left hold has no B7 prior"
    assert s.template_meta["init_arms"] == "right tucked (JoyLo reset), left hold"
    fact = s.template_facts["hold_arm"]
    assert (fact["arm"], fact["reason"], fact["against_default"]) == ("left", "segment fallback", True)
    fb = fact["segment_fallback"]
    assert fb["from_arm"] == "right" and "goal_infeasible" in fb["why"] and fb["goal_options"] == {"arm": "right", "feasible": 0} and "failed" not in fb
    assert s.template_facts["goal_options"] == {"arm": "left", "feasible": 5}
    assert s.arm_summary() == {"arm": "left", "reason": "segment fallback", "against_default": True}
    assert s.kit.stats["segment_fallbacks"] == 1 and s.kit.stats["segment_fallback_taken.left"] == 1


def test_the_segment_fallback_runs_once_and_only_to_a_viable_arm(tmp_path, monkeypatch):
    # both goal searches fail: one retry, then the template is rejected with both arms' reasons
    s, log = _template(tmp_path, monkeypatch, MIRRORED, lambda arm: False)
    with pytest.raises(Reject) as e:
        s.choose_hold()
    assert e.value.reason == "goal_infeasible" and "right arm:" in str(e.value) and "left arm (segment fallback):" in str(e.value)
    assert log["searched"] == ["right", "left"] and "failed" in s.template_facts["hold_arm"]["segment_fallback"]
    # the left hold has no IK solution: the right arm's goal-search rejection stands as it was, no retry
    s, log = _template(tmp_path, monkeypatch, {"right": Q_HOLD_R}, lambda arm: False)
    with pytest.raises(Reject) as e:
        s.choose_hold()
    assert str(e.value) == "[goal_infeasible] no library grasp has a visible hold with a feasible goal (right)" and log["searched"] == ["right"]
    # a rejection other than an empty goal set is not a segment fallback
    s, log = _template(tmp_path, monkeypatch, MIRRORED, lambda arm: True)

    def spec_error():
        log["searched"].append(s.arm)
        raise Reject("spec", "3 intermediate waypoints precomputed, generation.waypoints = 2")

    s.precompute_options = spec_error
    with pytest.raises(Reject) as e:
        s.choose_hold()
    assert e.value.reason == "spec" and log["searched"] == ["right"]


def test_the_template_hold_rejections(tmp_path, monkeypatch):
    # no hold for either arm: one hold_pose rejection naming both
    s, log = _template(tmp_path, monkeypatch, {}, lambda arm: True)
    with pytest.raises(Reject) as e:
        s.choose_hold()
    assert e.value.reason == "hold_pose" and "right arm:" in str(e.value) and "left arm:" in str(e.value) and log["searched"] == []
    # the other arm switched off (generation.other_arm false): the right arm alone, its rejection as before
    s, log = _template(tmp_path, monkeypatch, {}, lambda arm: True, generation={"other_arm": False})
    with pytest.raises(Reject) as e:
        s.choose_hold()
    assert str(e.value) == "[hold_pose] no right-arm IK solution for the fixed hold" and log["solved"] == ["right"]
    s, log = _template(tmp_path, monkeypatch, MIRRORED, lambda arm: True, generation={"other_arm": False})
    s.choose_hold()
    assert s.arm == "right" and log["solved"] == ["right"] and s.template_facts["hold_arm"]["reason"] == "only arm offered"
    # generation.arm left: the left arm is the default and keeps the tie
    s, log = _template(tmp_path, monkeypatch, MIRRORED, lambda arm: True, default_arms=("left",))
    s.choose_hold()
    assert s.arm == "left" and log["solved"] == ["left", "right"] and s.arm_summary() == {"arm": "left", "reason": "default", "against_default": False}
