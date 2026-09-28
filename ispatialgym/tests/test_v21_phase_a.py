"""first-demo v2.1 Phase A (owner findings on v2.0): each rule fires only in its situation, detected at run time.

A1 (cap 0927-040231): an attach place opens the fingers just enough when they would meet the partner before the part seats.
A2 (block design 0927-040243): a travel leg keeps an arm that holds the recorded hand pose in another IK branch when it stays clear
and the next contact tracks from there; the next contact then runs on the kept branch's configurations.
A3 (bar soap 0927-034001): a library pinch whose fingers would reach below the object's bottom is lifted; a support place of an
object that hangs lower in the hand than recorded is raised by the difference.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.sidecar import Pose
from ispatialgym.robot import conventions as C
from ispatialgym.tests.test_v2_oracle import FakeBackend, iors_world, make_executor


def _pose(xyz, quat=(0.0, 0.0, 0.0, 1.0)):
    return Pose(position=[float(v) for v in xyz], orientation=[float(v) for v in quat])


# ------------------------------------------------------------------------------------------------ A3


def _grasp_setup(bottom=0.80, top=0.86):
    obj = SimpleNamespace(aabb=(th.tensor([0.5, -0.05, bottom]), th.tensor([0.6, 0.05, top])))
    state = SimpleNamespace(object=lambda name: obj)
    action = SimpleNamespace(kind="library", object="soap", arm="right")
    grasp = _pose([0.55, 0.0, 0.83], C.quat_from_yaw(0.0).tolist())
    pre = _pose([0.55, 0.0, 0.93])
    targets = {"pre_grasp": pre, "grasp": grasp, "approach": [pre, grasp]}
    return action, state, targets


def test_a3_grasp_lifted_by_the_finger_deficit_below_the_object(monkeypatch):
    action, state, targets = _grasp_setup()
    monkeypatch.setattr(EX, "_finger_low_z", lambda pos, quat, arm: 0.794)  # 6 mm under the object's bottom
    ctx = {"next_place_kind": "support"}
    out = EX.lift_grasp_off_support(action, None, state, targets, ctx, EX.ExecuteConfig())
    need = 0.80 + EX.ExecuteConfig().grasp_support_clearance_m - 0.794
    assert ctx["grasp_support_lift"]["applied"] and ctx["grasp_support_lift"]["lift_m"] == pytest.approx(need, abs=1e-4)
    assert out["grasp"].position[2] == pytest.approx(targets["grasp"].position[2] + need, abs=1e-6)
    assert out["grasp"].position[:2] == pytest.approx(targets["grasp"].position[:2])
    assert out["pre_grasp"] is targets["pre_grasp"]  # the transit to the pre-grasp stays the recorded one (sandal 0927-030120)
    # the approach ramps from the recorded pre-grasp up to the lifted grasp
    assert [p.position[2] for p in out["approach"]] == pytest.approx([targets["approach"][0].position[2], targets["approach"][1].position[2] + need])
    mid = EX._ramped(_pose([0.0, 0.0, 1.0]), [_pose([0.0, 0.0, 0.9]), _pose([0.0, 0.0, 0.8])], _pose([0.0, 0.0, 0.6]), 0.01)
    assert [p.position[2] for p in mid] == pytest.approx([0.9025, 0.805])


def test_a3_grasp_kept_when_clear_or_when_the_pads_would_leave_the_object(monkeypatch):
    action, state, targets = _grasp_setup()
    monkeypatch.setattr(EX, "_finger_low_z", lambda pos, quat, arm: 0.81)  # already 10 mm above the bottom
    ctx = {}
    assert EX.lift_grasp_off_support(action, None, state, targets, ctx, EX.ExecuteConfig()) is targets and not ctx["grasp_support_lift"]["applied"]
    action, state, targets = _grasp_setup(top=0.835)  # a 5 mm-tall part: a lift would take the pads off it
    monkeypatch.setattr(EX, "_finger_low_z", lambda pos, quat, arm: 0.79)
    ctx = {}
    assert EX.lift_grasp_off_support(action, None, state, targets, ctx, EX.ExecuteConfig()) is targets
    assert not ctx["grasp_support_lift"]["applied"] and "pads" in ctx["grasp_support_lift"]["reason"]
    ctx = {}
    assert EX.lift_grasp_off_support(SimpleNamespace(kind="cube", object="b", arm="right"), None, state, targets, ctx, EX.ExecuteConfig()) is targets
    assert ctx == {}


def test_a3_grasp_kept_when_the_lifted_approach_does_not_track_or_failed_before(monkeypatch):
    """Sandal 0927-030120 replayed on a pod: its pre-grasp and grasp witnesses lie in different IK branches; the approach to the grasp
    lifted 13.7 mm stalled at waypoint 8 / 12 and the finish toward the grasp witness (3.1 rad away) was refused, twice."""
    action, state, targets = _grasp_setup()
    monkeypatch.setattr(EX, "_finger_low_z", lambda pos, quat, arm: 0.794)
    seen = {}

    def tracks(motions, arm, q_hint, poses):
        seen["z"] = [float(p.position[2]) for p in poses]
        return False

    monkeypatch.setattr(EX, "_approach_tracks", tracks)
    ctx = {"next_place_kind": "support"}
    assert EX.lift_grasp_off_support(action, None, state, targets, ctx, EX.ExecuteConfig()) is targets
    assert not ctx["grasp_support_lift"]["applied"] and "does not track" in ctx["grasp_support_lift"]["reason"]
    assert seen["z"][0] == pytest.approx(targets["pre_grasp"].position[2], abs=1e-6) and seen["z"][-1] == pytest.approx(targets["grasp"].position[2] + 0.010, abs=1e-6)
    # a lifted grasp whose attempt failed: the retry of that action runs the recorded grasp
    monkeypatch.setattr(EX, "_approach_tracks", lambda motions, arm, q_hint, poses: True)
    ctx = {"protect": {"index": 7}, "next_place_kind": "support"}
    assert EX.lift_grasp_off_support(action, SimpleNamespace(support_lift_refused={7}), state, targets, ctx, EX.ExecuteConfig()) is targets
    assert not ctx["grasp_support_lift"]["applied"] and "previous attempt" in ctx["grasp_support_lift"]["reason"]
    ctx = {"protect": {"index": 8}, "next_place_kind": "attach"}
    assert EX.lift_grasp_off_support(action, SimpleNamespace(support_lift_refused={7}), state, targets, ctx, EX.ExecuteConfig())["grasp"] is not targets["grasp"]
    assert ctx["grasp_support_lift"]["applied"]


def test_a3_grasp_kept_when_the_object_is_dropped_next(monkeypatch):
    """Sandal 0927-030120 on 503be065: lifted 14 mm at its grasp, the drop landed 9.6 deg / 7 mm off the recorded one and the push
    under the table turned it 78 deg (v2.0: 2 deg) - a drop does not follow the in-hand pose, so its grasp is not lifted."""
    action, state, targets = _grasp_setup()
    monkeypatch.setattr(EX, "_finger_low_z", lambda pos, quat, arm: 0.794)
    monkeypatch.setattr(EX, "_approach_tracks", lambda motions, arm, q_hint, poses: True)
    for nxt in ("drop", None):
        ctx = {"next_place_kind": nxt}
        assert EX.lift_grasp_off_support(action, None, state, targets, ctx, EX.ExecuteConfig()) is targets
        assert not ctx["grasp_support_lift"]["applied"] and "next place" in ctx["grasp_support_lift"]["reason"]


def test_executor_next_place_kind_follows_the_grasped_object():
    from ispatialgym.oracle.executor import Executor

    ex = SimpleNamespace(program_actions=None)

    def grasp(arm, obj):
        a = EX.Grasp.__new__(EX.Grasp)
        object.__setattr__(a, "arm", arm)
        object.__setattr__(a, "object", obj)
        return a

    def place(arm, obj, kind):
        a = EX.Place.__new__(EX.Place)
        for k, v in (("arm", arm), ("object", obj), ("kind", kind)):
            object.__setattr__(a, k, v)
        return a

    ex.program_actions = [grasp("right", "sandal"), place("left", "sandal", "support"), place("right", "sandal", "drop"), grasp("right", "cube"),
                          place("right", "cube", "support")]
    assert Executor.next_place_kind(ex, 0, ex.program_actions[0]) == "drop"
    assert Executor.next_place_kind(ex, 3, ex.program_actions[3]) == "support"
    ex.program_actions = [grasp("right", "sandal"), grasp("right", "cube"), place("right", "sandal", "support")]
    assert Executor.next_place_kind(ex, 0, ex.program_actions[0]) is None  # the arm grasps again first



def _posture_motions(reaches_here=False, plan_ok=True, violations=()):
    """A motions stub for ``execute._into_witness_posture``: the arm at a posture 2.9 rad from the pre-grasp witness."""
    calls = []
    q_now = th.zeros(28)
    q_now[th.arange(7)] = 2.9
    backend = SimpleNamespace(q_full=lambda: q_now.clone(), update_world=lambda: calls.append(("world",)),
                              solve_ik=lambda emb, targets, q_seed=None, sync_world=False: C.as_tensor(q_seed).clone() + 0.01,
                              plan_joint=lambda emb, q, q_start=None, **kw: (calls.append(("plan", C.as_tensor(q).clone())) or
                                                                           SimpleNamespace(success=plan_ok, q_traj=th.stack([q_start, C.as_tensor(q)]), status="ok")))

    def execute(q_traj, arms=None):
        calls.append(("execute", C.as_tensor(q_traj)[-1].clone()))
        yield th.zeros(3)

    m = SimpleNamespace(backend=backend, robot=SimpleNamespace(arm_control_idx={"right": th.arange(7)}), cfg=SimpleNamespace(witness_refine_max_jump=0.6, plan_attempts=2),
                        along_reaches=lambda arm, wps: reaches_here, _protect_violations=lambda arms, q_traj, emb, P=None: list(violations),
                        execute_joint_trajectory=execute)
    return m, calls


def test_grasp_moves_into_the_witness_posture_when_the_approach_tracks_only_from_there(monkeypatch):
    """Sandal on pods (0927-154128 / 0927-165511): the via-above left the arm 2.91 rad from the pre-grasp witness and the approach
    stalled 56 mm short; from the witness it tracks - the arm moves into the witness posture first."""
    _, _, targets = _grasp_setup()
    w = th.zeros(28)
    monkeypatch.setattr(EX, "_approach_tracks", lambda motions, arm, q_hint, poses: True)
    m, calls = _posture_motions()
    ctx = {"witnesses": {"pre_grasp": w}}
    list(EX._into_witness_posture(m, targets, ctx, "right"))
    assert ctx["witness_posture"]["result"] == "moved into the witness posture" and ctx["witness_posture"]["gap_rad"] == pytest.approx(2.89, abs=1e-3)
    plan = [c[1] for c in calls if c[0] == "plan"][0]
    assert float(plan[0]) == pytest.approx(0.01)  # the witness refined by the IK of the pre-grasp pose
    assert [c[0] for c in calls] == ["world", "plan", "execute"]


def test_grasp_keeps_the_transit_posture_outside_that_situation(monkeypatch):
    _, _, targets = _grasp_setup()
    ctx = {"witnesses": {"pre_grasp": th.zeros(28)}}
    monkeypatch.setattr(EX, "_approach_tracks", lambda motions, arm, q_hint, poses: True)
    m, calls = _posture_motions(reaches_here=True)  # the approach tracks from where the arm is (v2.0, the local replays: 0.91 rad)
    list(EX._into_witness_posture(m, targets, ctx, "right"))
    assert calls == [] and "witness_posture" not in ctx
    monkeypatch.setattr(EX, "_approach_tracks", lambda motions, arm, q_hint, poses: False)  # nor from the witness: nothing to gain
    m, calls = _posture_motions()
    list(EX._into_witness_posture(m, targets, ctx, "right"))
    assert calls == [] and "witness_posture" not in ctx
    monkeypatch.setattr(EX, "_approach_tracks", lambda motions, arm, q_hint, poses: True)
    for kw, why in (({"plan_ok": False}, "no plan"), ({"violations": [{"object": "sandal", "clearance_m": -0.01}]}, "passes sandal")):
        m, calls = _posture_motions(**kw)
        ctx = {"witnesses": {"pre_grasp": th.zeros(28)}}
        list(EX._into_witness_posture(m, targets, ctx, "right"))
        assert why in ctx["witness_posture"]["result"] and "execute" not in [c[0] for c in calls]
    m, calls = _posture_motions()
    list(EX._into_witness_posture(m, targets, {"witnesses": {}}, "right"))  # no pre-grasp witness
    assert calls == []

def _place_setup(hang_extra):
    """Recorded: the hand 5 cm above the object's origin; live: 5 cm + ``hang_extra`` (the object hangs lower in the hand)."""
    obj = SimpleNamespace(name="soap")
    eef_now = (th.tensor([0.4, 0.1, 1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    obj_now = (th.tensor([0.4, 0.1, 1.0 - 0.05 - hang_extra]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    state = SimpleNamespace(object=lambda n: obj, held_object=lambda arm: obj, eef_pose=lambda arm: eef_now, object_pose=lambda n: obj_now)
    g = SimpleNamespace(object="soap", arm="right", extra={"eef_in_object": {"position": [0.0, 0.0, 0.05], "orientation": [0.0, 0.0, 0.0, 1.0]}})
    sidecar = SimpleNamespace(grasps=[g])
    place, pre = _pose([0.7, 0.0, 0.88]), _pose([0.7, 0.0, 0.98])
    targets = {"pre_place": pre, "place": place, "descent": [pre, place], "retreat": [place, pre], "carry": [_pose([0.5, 0.0, 1.0]), pre]}
    return SimpleNamespace(kind="support", object="soap", arm="right"), state, sidecar, targets


def test_a3_place_raised_when_the_object_hangs_lower_in_the_hand():
    action, state, sidecar, targets = _place_setup(0.015)
    ctx = {}
    out = EX.raise_place_for_inhand_pose(action, state, sidecar, targets, ctx, EX.ExecuteConfig())
    assert ctx["place_inhand_raise"]["applied"] and ctx["place_inhand_raise"]["raised_m"] == pytest.approx(0.015, abs=1e-4)
    for k in ("place", "pre_place"):
        assert out[k].position[2] == pytest.approx(targets[k].position[2] + 0.015, abs=1e-5)
    assert out["carry"][-1].position[2] == pytest.approx(targets["carry"][-1].position[2] + 0.015, abs=1e-5)
    assert out["carry"][0].position[2] < targets["carry"][0].position[2] + 0.015  # the carry ramps up to the raised pre-place


def test_a3_place_never_lowered_and_only_for_support_places():
    action, state, sidecar, targets = _place_setup(-0.010)  # higher in the hand: the recorded place stands
    ctx = {}
    assert EX.raise_place_for_inhand_pose(action, state, sidecar, targets, ctx, EX.ExecuteConfig()) is targets and not ctx["place_inhand_raise"]["applied"]
    action = SimpleNamespace(kind="attach", object="soap", arm="right")
    ctx = {}
    assert EX.raise_place_for_inhand_pose(action, state, sidecar, targets, ctx, EX.ExecuteConfig()) is targets and ctx == {}
    action, state, sidecar, targets = _place_setup(0.012)  # a drop: never lifted at its grasp, released as recorded
    ctx = {}
    assert EX.raise_place_for_inhand_pose(SimpleNamespace(kind="drop", object="soap", arm="right"), state, sidecar, targets, ctx, EX.ExecuteConfig()) is targets
    assert ctx == {}


# ------------------------------------------------------------------------------------------------ A1


def _box(cx, cy, z0, z1, hw=0.002, hd=0.002):
    xs, ys, zs = np.meshgrid([cx - hw, cx + hw], [cy - hd, cy + hd], [z0, z1], indexing="ij")
    return np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)


def test_a1_attach_place_opens_the_fingers_just_enough(monkeypatch):
    """Two finger boxes 26 mm either side of the eef axis reaching 20 mm below the eef, a partner disc of radius 27 mm whose top the
    fingertips would reach 5 mm before the seat: the fingers open by the smallest 0.5 mm step whose rays miss the disc."""
    fingers = [(_box(0.0, 0.026, -0.02, 0.0), np.array([0.0, 1.0, 0.0])), (_box(0.0, -0.026, -0.02, 0.0), np.array([0.0, -1.0, 0.0]))]
    monkeypatch.setattr(EX, "finger_surface_points", lambda arm, spacing: fingers)
    top, r_rim = 0.815, 0.027  # the seat puts the eef at z 0.83: fingertips at 0.81, 5 mm below the partner's top

    def raycast(starts, ends):
        out = []
        for s, e in zip(starts, ends):
            s, e = np.asarray(s), np.asarray(e)
            r = float(np.hypot(s[0], s[1]))
            if r <= r_rim and s[2] > top >= e[2]:
                out.append([{"hit": True, "distance": float(s[2] - top), "rigidBody": "/World/partner/base_link"}])
            else:
                out.append([{"hit": False}])
        return out

    monkeypatch.setattr(EX, "_raycast_all", raycast)
    partner = SimpleNamespace(prim_path="/World/partner", name="partner")
    part = SimpleNamespace(name="cap")
    robot = SimpleNamespace(gripper_control_idx={"right": th.tensor([0, 1])}, get_joint_positions=lambda: th.tensor([0.0, 0.0]))
    emitted = []
    motions = SimpleNamespace(robot=robot, state=SimpleNamespace(gripper_cmd={}, stop_base=lambda: None), cfg=SimpleNamespace(gripper_preshape_steps=3),
                              _gripper_limits=lambda arm: (0.0, 0.05), gripper_command=lambda arm, q: 2 * q / 0.05 - 1, _emit=lambda: emitted.append(1) or th.zeros(1))
    state = SimpleNamespace(object=lambda n: partner if n == "bottle" else part, held_object=lambda arm: part)
    action = SimpleNamespace(arm="right", on="bottle", object="cap")
    targets = {"pre_place": _pose([0.0, 0.0, 0.93]), "place": _pose([0.0, 0.0, 0.83])}
    ctx = {}
    list(EX.clear_fingers_over_partner(action, motions, state, targets, ctx, EX.ExecuteConfig()))
    rec = ctx["finger_clearance"]
    assert rec["before_seat_m"] == pytest.approx(0.005, abs=1e-4)
    # the inner corners stand at r = hypot(2, 24) mm: 3 mm (six steps) takes them past the 27 mm rim
    assert rec["applied"] and rec["opened_m"] == pytest.approx(0.003, abs=1e-6) and rec["before_seat_after_m"] is None
    assert motions.state.gripper_cmd["right"] == pytest.approx(2 * 0.003 / 0.05 - 1) and len(emitted) == 3 and rec["held"]


def test_a1_attach_place_leaves_clear_fingers_alone(monkeypatch):
    fingers = [(_box(0.0, 0.030, -0.02, 0.0), np.array([0.0, 1.0, 0.0])), (_box(0.0, -0.030, -0.02, 0.0), np.array([0.0, -1.0, 0.0]))]
    monkeypatch.setattr(EX, "finger_surface_points", lambda arm, spacing: fingers)
    monkeypatch.setattr(EX, "_raycast_all", lambda starts, ends: [[{"hit": False}] for _ in starts])
    partner = SimpleNamespace(prim_path="/World/partner", name="partner")
    robot = SimpleNamespace(gripper_control_idx={"right": th.tensor([0, 1])}, get_joint_positions=lambda: th.tensor([0.0, 0.0]))
    motions = SimpleNamespace(robot=robot, state=SimpleNamespace(gripper_cmd={}, stop_base=lambda: None), cfg=SimpleNamespace(gripper_preshape_steps=3),
                              _gripper_limits=lambda arm: (0.0, 0.05), gripper_command=lambda arm, q: 0.0, _emit=lambda: th.zeros(1))
    state = SimpleNamespace(object=lambda n: partner, held_object=lambda arm: None)
    ctx = {}
    list(EX.clear_fingers_over_partner(SimpleNamespace(arm="right", on="bottle", object="cap"), motions, state,
                                        {"pre_place": _pose([0.0, 0.0, 0.93]), "place": _pose([0.0, 0.0, 0.83])}, ctx, EX.ExecuteConfig()))
    assert not ctx["finger_clearance"]["applied"] and ctx["finger_clearance"]["before_seat_m"] is None and motions.state.gripper_cmd == {}


# ------------------------------------------------------------------------------------------------ A2


class _BranchBackend:
    """Toy kinematics with two IK branches per hand pose: the hand sits at (|q0|, q1, q2) of the arm's first three joints, so q0 and -q0
    hold the same pose; the IK keeps the seed's sign of q0."""

    def __init__(self, robot, arm="right"):
        self.robot, self.arm = robot, arm
        self.idx = robot.arm_control_idx[arm]
        self.q = robot.q.clone()
        self._jidx = {n: i for i, n in enumerate(M.ALL_JOINTS)}
        self.collide = False
        self.fail_path = False

    def q_full(self):
        return self.q.clone()

    def fk(self, frames, q=None):
        q = self.q if q is None else q
        a = q[self.idx]
        return {f: (th.stack([a[0].abs(), a[1], a[2]]), th.tensor([0.0, 0.0, 0.0, 1.0])) for f in frames}

    def _solve(self, pose, seed):
        q = seed.clone()
        p = C.as_tensor(pose[0]).float()
        sign = 1.0 if float(seed[self.idx][0]) >= 0.0 else -1.0
        q[self.idx[0]], q[self.idx[1]], q[self.idx[2]] = sign * float(p[0]), float(p[1]), float(p[2])
        return q

    def check_collision(self, qs, sync_world=False):
        return th.full((len(qs),), bool(self.collide))

    def solve_ik(self, emb, targets, q_seed=None, sync_world=True):
        return self._solve(next(iter(targets.values())), q_seed)

    def solve_ik_path(self, emb, waypoints, q_start=None, **kw):
        if self.fail_path:
            return SimpleNamespace(success=False, q_traj=None, status="stalled (fake)")
        q, out = q_start.clone(), []
        for wp in waypoints:
            q = self._solve(next(iter(wp.values())), q)
            out.append(q.clone())
        return SimpleNamespace(success=True, q_traj=th.stack(out), status="ok")

    def world_to_root(self, pose):
        return pose


def _branch_setup(x0=0.5, first=(0.6, 0.1, 0.2), second=(0.65, 0.1, 0.15)):
    from ispatialgym.oracle.motions import MotionsConfig
    from ispatialgym.tests.test_v2_oracle import FakeRobot

    robot = FakeRobot()
    backend = _BranchBackend(robot)
    idx = backend.idx
    backend.q[idx[0]], backend.q[idx[1]], backend.q[idx[2]] = -x0, 0.1, 0.2  # the arm holds the pose in the "negative" branch
    q_rec = backend.q[idx].clone()
    q_rec[0] = x0  # the recorded joints: the same hand pose, the other branch
    motions = SimpleNamespace(backend=backend, cfg=MotionsConfig(), _densify=lambda start, wps: list(wps))
    state = SimpleNamespace(robot=robot)
    rec_q = backend.q.clone()
    rec_q[idx[0]] = first[0]  # the sampler's witness of the next pre-grasp: the recorded ("positive") branch

    def w(action, key, xyz):
        return {"action": action, "key": key, "pose": (th.tensor(xyz, dtype=th.float32), th.tensor([0.0, 0.0, 0.0, 1.0])), "q": rec_q, "margin": 0.2, "base": [0.0, 0.0, 0.0]}

    nxt = {"index": 5, "name": "grasp", "object": None, "chain": [w(5, "pre_grasp", first), w(5, "grasp", second), w(6, "lift", second)],
           "stance": ([0.0, 0.0, 0.0], robot.q[robot.trunk_control_idx].clone()), "margin": 0.2}
    return motions, state, backend, q_rec, nxt


def test_a2_keep_arm_branch_keeps_the_arm_and_tracks_the_next_contact_from_it():
    motions, state, backend, q_rec, nxt = _branch_setup()
    f = EX.keep_arm_branch(motions, state, "right", q_rec, [backend.q_full()], nxt, {})
    assert f["keep"], f
    assert f["joint_gap_rad"] == pytest.approx(1.0, abs=1e-3) and f["hand_gap_m"] == pytest.approx(0.0, abs=1e-6)
    assert f["reach_rad"] == pytest.approx(0.1, abs=1e-4) and f["recorded_reach_rad"] == pytest.approx(1.1, abs=1e-4)
    idx0 = backend.idx[0]
    assert sorted(f["witnesses"]) == [5, 6] and sorted(f["witnesses"][5]) == ["grasp", "pre_grasp"]
    assert float(f["witnesses"][5]["pre_grasp"][idx0]) == pytest.approx(-0.6) and float(f["witnesses"][6]["lift"][idx0]) == pytest.approx(-0.65)


def test_a2_keep_arm_branch_refuses_outside_its_situation():
    motions, state, backend, q_rec, nxt = _branch_setup()
    far = q_rec.clone()
    far[0] = 0.56  # 6 cm from the hand: another hand pose
    assert EX.keep_arm_branch(motions, state, "right", far, [], nxt, {})["why"] == "the hand is not at the recorded pose"
    assert "no grasp" in EX.keep_arm_branch(motions, state, "right", q_rec, [], None, {})["why"]
    # the arm's next contact is a push (sandal 0927-030120: the push under the table from the kept branch turned the sandal 49 deg)
    push = {"index": 5, "name": "push", "object": None, "chain": [], "why": "the arm's next contact is a push, not a grasp"}
    assert EX.keep_arm_branch(motions, state, "right", q_rec, [], push, {})["why"] == push["why"]
    backend.collide = True
    assert "collides" in EX.keep_arm_branch(motions, state, "right", q_rec, [backend.q_full()], nxt, {})["why"]
    backend.collide, backend.fail_path = False, True
    assert "does not track" in EX.keep_arm_branch(motions, state, "right", q_rec, [], nxt, {})["why"]
    motions, state, backend, q_rec, nxt = _branch_setup(second=(3.135, 0.1, 0.15))  # the chain from here ends 5 mrad from joint 1's limit
    f = EX.keep_arm_branch(motions, state, "right", q_rec, [], nxt, {})
    assert not f["keep"] and "joint limit" in f["why"] and f["chain_margin_rad"] < 0.01
    motions, state, backend, q_rec, nxt = _branch_setup(first=(0.6, 1.5, 0.2))  # the recorded witness (0.6, 0.1, 0.2) lies nearer than the IK from here
    f = EX.keep_arm_branch(motions, state, "right", q_rec, [], nxt, {})
    assert not f["keep"] and "no nearer" in f["why"] and f["reach_rad"] > f["recorded_reach_rad"]


def test_a2_move_idle_arms_leaves_a_kept_arm_and_hands_its_witnesses_over(monkeypatch):
    sc, env = iors_world()
    robot = env.robot
    ridx = robot.arm_control_idx["right"]
    manip = sc.stance("manipulation")
    ex, motions, backend, rec = make_executor(env, sc)
    here = robot.q[ridx].clone()
    want = here + th.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    lidx = robot.arm_control_idx["left"]
    arms_q = {"right": want.tolist(), "left": robot.q[lidx].tolist()}
    over = {3: {"pre_grasp": th.zeros(28)}}
    monkeypatch.setattr(EX, "keep_arm_branch", lambda motions, state, arm, q_rec, sweep, nxt, ctx: {"keep": True, "hand_gap_m": 0.003, "joint_gap_rad": 1.0, "witnesses": over})
    ctx = {"next_contact": {"right": {"index": 3, "name": "grasp"}}}
    for a in EX.move_idle_arms(motions, ex.state, manip, ctx, arms_q=arms_q, keep_sweep=lambda: []):
        env.step(a)
    assert ctx["arms_posture"]["moved"] == [] and th.allclose(robot.q[ridx], here, atol=1e-6)
    assert ctx["branch_kept"]["right"]["keep"] and "witnesses" not in ctx["branch_kept"]["right"] and ctx["witness_overrides"] == {"right": over}
    monkeypatch.setattr(EX, "keep_arm_branch", lambda motions, state, arm, q_rec, sweep, nxt, ctx: {"keep": False, "why": "the hand is not at the recorded pose"})
    ctx = {"next_contact": {"right": {"index": 3, "name": "grasp"}}}
    for a in EX.move_idle_arms(motions, ex.state, manip, ctx, arms_q=arms_q, keep_sweep=lambda: []):
        env.step(a)
    assert ctx["arms_posture"]["moved"] == ["right"] and th.allclose(robot.q[ridx], want, atol=1e-4) and "witness_overrides" not in ctx


def test_a2_executor_looks_ahead_past_travel_legs_and_applies_the_kept_witnesses():
    from ispatialgym.oracle.programs import expand

    sc, env = iors_world()
    ex, motions, backend, rec = make_executor(env, sc)
    ex.program_actions = expand(sc.alias, sc)
    names = [a.name for a in ex.program_actions]
    assert names[:5] == ["transition", "grasp", "lift", "place", "retreat"]
    q = [0.0] * 28

    def wit(key, margin):
        return {key: {"q": q, "pose": {"position": [0.5, 0.0, 1.0], "orientation": [0.0, 0.0, 0.0, 1.0]}, "base_pose2d": [0.1, 0.2, 0.3], "joint_limit_margin_rad": margin}}

    sc.extra["ik_witnesses"] = [{"action": 1, "arm": "right", "targets": {**wit("pre_grasp", 0.3), **wit("grasp", 0.2)}}, {"action": 2, "arm": "right", "targets": wit("lift", 0.1)}]
    nc = ex.next_contact(0, "right")
    assert nc["index"] == 1 and nc["name"] == "grasp" and [(c["action"], c["key"]) for c in nc["chain"]] == [(1, "pre_grasp"), (1, "grasp"), (2, "lift")]
    assert nc["margin"] == pytest.approx(0.1) and nc["stance"][0] == [0.1, 0.2, 0.3]
    assert ex.next_contact(0, "left") is None  # the next action is the right arm's
    q_kept = th.full((28,), 0.5)
    ex._absorb_branch_decisions(0, {"branch_kept": {"right": {"keep": True}}, "arms_posture": {"moved": []}, "witness_overrides": {"right": {1: {"pre_grasp": q_kept}}}})
    assert ex._witness_overrides[1]["leg"] == 0 and th.equal(ex._witness_overrides[1]["witnesses"]["pre_grasp"], q_kept)
    ex._absorb_branch_decisions(3, {"arms_posture": {"moved": ["right"]}})
    assert ex._witness_overrides == {}


# ------------------------------------------------------------------------------------------------ carton close brace (chooser follow-up)


def test_wide_stop_keeps_the_push_line_between_its_fingertips():
    """Carton close 0927-133502: the far-face stop at the robot-side end + 5 cm inset stood 65 mm beside the push line (the fingertips
    +-4 cm around the pad) and the push turned the box 1.23 deg; a stop the line passes 1 cm inside keeps its place."""
    from ispatialgym.sampling.v2.tasks.articulation_small import wide_stop_positions

    ends = (-0.115, 0.115)
    assert wide_stop_positions(-0.066, -0.041, 0.04, ends) == [-0.066]  # the line 25 mm from the pad: >= 1 cm inside the 4 cm tips
    assert wide_stop_positions(-0.066, -0.035, 0.04, ends)[0] == pytest.approx(-0.035)  # 31 mm: beside the tips' 3 cm reach - moved
    pos = wide_stop_positions(-0.066, -0.0035, 0.04, ends)
    assert pos == pytest.approx([-0.0035, -0.0135, -0.0235, -0.0335])  # from the line toward the corner, the line >= 1 cm inside the tips
    assert all(abs(p - (-0.0035)) <= 0.03 + 1e-9 for p in pos)
    assert wide_stop_positions(0.07, 0.0, 0.04, ends) == pytest.approx([0.0, 0.01, 0.02, 0.03])
    # a face too short for a stop around the line: nothing to try (the next brace kind)
    assert wide_stop_positions(0.04, -0.02, 0.04, (-0.045, 0.045)) == []



# ------------------------------------------------------------------------------------------------ joint jump at the tracking resolution


def test_arm_along_retries_a_joint_jump_at_half_the_waypoint_spacing(tmp_path):
    """Shampoo 0927-183304 on 052de5eb: the carry to the floor jumped 0.23 rad (> max_joint_step) between two waypoints in both
    attempts; at half the spacing a continuous fast turn stays under the limit - a branch flip does not, and still fails."""
    from ispatialgym.oracle.motions import SkillError
    from ispatialgym.tests.test_v2_oracle import drive

    sc, env = iors_world()
    ex, motions, backend, _ = make_executor(env, sc, tmp_path)
    arm = "right"
    p0, q0 = motions.eef_pose(arm)
    target = (p0 + th.tensor([0.10, 0.0, 0.0]), q0)
    orig = backend.solve_ik_path
    seen = []

    def jump_once(emb, waypoints, q_start=None, **kw):
        seen.append(len(waypoints))
        if len(seen) == 1:
            return SimpleNamespace(success=False, q_traj=None, status="joint jump 0.23 rad at waypoint 1", diagnostics={})
        return orig(emb, waypoints, q_start=q_start, **kw)

    backend.solve_ik_path = jump_once
    drive(env, motions.arm_along(arm, [target]))
    assert seen[1] == 2 * seen[0] and float(th.linalg.norm(motions.eef_pose(arm)[0] - target[0])) < 5e-3
    seen.clear()
    backend.solve_ik_path = lambda emb, waypoints, q_start=None, **kw: (seen.append(len(waypoints)) or
                                                                        SimpleNamespace(success=False, q_traj=None, status="joint jump 3.1 rad at waypoint 1", diagnostics={}))
    with pytest.raises(SkillError):
        drive(env, motions.arm_along(arm, [(p0, q0)]))
    assert len(seen) == 2  # one retry at the finer spacing, then the leg fails as before
