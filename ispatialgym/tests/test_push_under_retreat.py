"""The push-under retreat (back along the push at its height, then up), the program check's push-under ignore sets and the yaw-free
push-under orientation (owner 2026-10-13 ~04:30 item 1, under_large#01: c3_k0's demos 17837980 / 17838477 ran all 12 actions with
``under`` satisfied, then ``left_realsense_link`` met bed_rrcvaq_0 one step after the straight-up retreat; the sandal ended 0.605 rad off
its goal orientation, 34.8 deg of yaw lying flat) (``pytest ispatialgym/tests/test_push_under_retreat.py``)."""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle import programs as PR
from ispatialgym.oracle.actions import Push, Retreat
from ispatialgym.oracle.executor import Executor, ExecutorConfig, vertical_angle
from ispatialgym.oracle.motions import MotionsConfig
from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.sidecar import Sidecar, SidecarError
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import program_check as PC

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "sidecar_variants", "inter_object_relation_satisfaction-under_large_scale__push_under.json")
TARGET, FLOOR, PARTNER = "briefcase.n.01_1", "floor.n.01_1", "breakfast_table.n.01_1"
NO_HAND = SimpleNamespace(object_pose=lambda n: None)  # no live hand: the legs leave from the recorded push goal (the program check's view)

#: under_large#01 c3_k0 (alias 133, code_27ee9069): the recorded push-under contact - the 'pitch 0-' blade along +x, the bed's -x face at 22.24
C3K0_PUSH = {"pre": [21.86641502380371, 17.534568786621094, 0.14036628603935242], "init": [21.86641502380371, 17.534568786621094, 0.040366280823946],
             "mid": [22.01763153076172, 17.534568786621094, 0.040366280823946], "goal": [22.16884994506836, 17.53456687927246, 0.040366280823946]}
C3K0_QUAT = [-0.7071067690849304, 0.7071067690849304, 0.0, 0.0]
BED_EDGE_X = 22.2398  # bed_rrcvaq_0's AABB, -x face
CAMERA_AHEAD_M = 0.075  # the 'pitch 0-' wrist camera's spheres reach up to 7.5 cm toward the bed from the eef


def _raw() -> Dict[str, Any]:
    with open(FIXTURE) as f:
        return json.load(f)


def _fixture(raw=None) -> Sidecar:
    return Sidecar.from_dict(raw or _raw())


def _c3_k0() -> Sidecar:
    d = _raw()
    p = d["contacts"]["push"]
    for k, v in C3K0_PUSH.items():
        p[k] = {"position": list(v), "orientation": list(C3K0_QUAT)}
    p.update(direction=[1.0, -0.0, 0.0], distance_m=0.2924346923828125, pitch_deg=0.0, ground_ignored=True)
    return Sidecar.from_dict(d)


def _yaw(a: float) -> th.Tensor:
    return th.tensor([0.0, 0.0, math.sin(a / 2), math.cos(a / 2)])


def _roll(a: float) -> th.Tensor:
    return th.tensor([math.sin(a / 2), 0.0, 0.0, math.cos(a / 2)])


# ----------------------------------------------------------------------------------------------------------------
# (a) the retreat: back along the push at its height, then up
# ----------------------------------------------------------------------------------------------------------------


def test_push_under_retreat_backs_out_along_the_push_then_lifts():
    sc = _fixture()
    acts = expand(sc.alias, sc)
    assert [a.name for a in acts] == ["transition", "grasp", "lift", "navigate", "place", "retreat", "push", "retreat"]
    push, ret = acts[-2], acts[-1]
    assert isinstance(push, Push) and push.kind == "push_under" and isinstance(ret, Retreat) and ret.after is push
    assert ret.distance == pytest.approx(0.10) and ret.lift_after_m == pytest.approx(0.10) and ret.lift_first_m is None
    assert ret.slide_dir == pytest.approx([0.0, -1.0, 0.0])  # the fixture pushes along +y
    leg = ret.targets(sc, NO_HAND)["retreat"]
    pts = [list(p.position) for p in leg]
    assert len(pts) == 6
    # the first three at the push height (z 0.05), back along -y from the goal (6.5, 2.995): the corridor the push swept
    for p, y in zip(pts[:3], (2.995 - 0.1 / 3, 2.995 - 0.2 / 3, 2.895)):
        assert p == pytest.approx([6.5, y, 0.05], abs=1e-5)
    # then straight up, 10 cm
    for p, z in zip(pts[3:], (0.05 + 0.1 / 3, 0.05 + 0.2 / 3, 0.15)):
        assert p == pytest.approx([6.5, 2.895, z], abs=1e-5)
    assert all(list(p.orientation) == pytest.approx(list(push.contact.goal.orientation), abs=1e-6) for p in leg)
    assert list(ret.end_pose(sc, NO_HAND).position) == pytest.approx([6.5, 2.895, 0.15], abs=1e-5)
    assert ret.annotate(sc)[1].endswith("10 cm back along the push at its height, then 10 cm up")
    # the drop's retreat and every other program keep their retreats
    assert acts[5].lift_after_m is None and acts[5].slide_dir is None


def test_push_under_retreat_leaves_from_where_the_hand_is_for_the_recorded_poses():
    """The demo's leg starts where the push left the hand (2 cm on past the goal after its residual re-push, the loaded pad 4 mm low) and
    ends at the poses taken from the recorded goal - the poses the program check solved and recorded the witness of."""
    sc = _fixture()
    push, ret = expand(sc.alias, sc)[-2:]
    q = th.tensor(push.contact.goal.orientation)
    here = th.tensor(push.contact.goal.position) + th.tensor([0.003, 0.02, -0.004])
    st = SimpleNamespace(eef_pose=lambda arm: (here.clone(), q.clone()), object_pose=lambda n: None)
    pts = [list(p.position) for p in ret.targets(sc, st)["retreat"]]
    mid = th.tensor([6.5, 2.895, 0.05])
    assert pts[0] == pytest.approx((here + (mid - here) / 3).tolist(), abs=1e-5)
    assert pts[2] == pytest.approx(mid.tolist(), abs=1e-5) and pts[-1] == pytest.approx([6.5, 2.895, 0.15], abs=1e-5)


def test_push_under_retreat_stays_in_the_swept_corridor():
    def contact(direction, init_y=2.745, goal_y=2.995):
        return SimpleNamespace(direction=direction, init=SimpleNamespace(position=[6.5, init_y, 0.05]), goal=SimpleNamespace(position=[6.5, goal_y, 0.05]))

    assert PR.push_under_retreat(contact([0.0, 1.0, 0.0])) == {"distance": pytest.approx(0.10), "slide_dir": [-0.0, -1.0, 0.0], "lift_after_m": pytest.approx(0.10)}
    # a blade that swept 6 cm backs out 6 cm: no farther than the corridor it swept
    assert PR.push_under_retreat(contact([0.0, 1.0, 0.0], init_y=2.935))["distance"] == pytest.approx(0.06)
    # the direction's floor-plane part, normalised
    assert PR.push_under_retreat(contact([0.6, 0.8, 0.05]))["slide_dir"] == pytest.approx([-0.6, -0.8, 0.0])
    with pytest.raises(SidecarError, match="along the floor"):
        PR.push_under_retreat(contact([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="lift_after_m"):
        Retreat(arm="right", object=TARGET, after=None, lift_after_m=0.1)  # no slide direction
    with pytest.raises(ValueError, match="lift_after_m"):
        Retreat(arm="right", object=TARGET, after=None, lift_after_m=0.1, slide_dir=[0.0, -1.0, 0.0], lift_first_m=0.03)


def test_c3_k0_record_retreats_clear_of_the_bed_edge():
    """c3_k0's record (written by code_27ee9069) expands at demo time to the new retreat: 10 cm back to x 22.069 at the push height
    (z 0.040), then up to z 0.140 - the wrist camera's spheres 9.6 cm short of the bed's -x face while the hand rises, where the straight lift
    at x 22.169 put them at the face (z ~0.23)."""
    sc = _c3_k0()
    push, ret = expand(sc.alias, sc)[-2:]
    leg = [list(p.position) for p in ret.targets(sc, NO_HAND)["retreat"]]
    gx, gz = C3K0_PUSH["goal"][0], C3K0_PUSH["goal"][2]
    assert all(p[0] <= gx + 1e-6 for p in leg)  # never toward the bed
    rising = [p for p in leg if p[2] > gz + 1e-4]
    assert len(rising) == 3 and all(p[0] + CAMERA_AHEAD_M <= BED_EDGE_X - 0.09 for p in rising)
    assert leg[-1] == pytest.approx([22.0688, 17.5346, 0.1404], abs=1e-4)


class _StubMotions:
    cfg = MotionsConfig()
    press_hold = None

    def __init__(self):
        self.legs: List[List[List[float]]] = []
        self.finishes: List[Any] = []

    def arm_along(self, arm, leg, finish=None, soft_pos_tol=None, **kw):
        self.legs.append([[float(v) for v in (p[0] if isinstance(p, (tuple, list)) else p.position)] for p in leg])
        self.finishes.append(finish)
        if False:
            yield None

    def gripper(self, *a, **k):
        if False:
            yield None

    def hold(self, n):
        if False:
            yield None


def _run_retreat(sc, ret, witnesses=None, witness_poses=None):
    push = ret.after
    here = th.tensor(push.contact.goal.position, dtype=th.float32)
    quat = th.tensor(push.contact.goal.orientation, dtype=th.float32)
    obj = SimpleNamespace(name="target_x_0")
    state = SimpleNamespace(eef_pose=lambda arm: (here.clone(), quat.clone()), object_pose=lambda n: (here.clone(), quat.clone()), object=lambda n: obj,
                            held_object=lambda arm: None, robot=None)
    ctx: Dict[str, Any] = {"cfg": EX.ExecuteConfig(), "witnesses": dict(witnesses or {}), "witness_poses": dict(witness_poses or {}), "hooks": {}}
    motions = _StubMotions()
    list(EX.execute_retreat(ret, motions, state, ret.targets(sc, state), ctx))
    return motions, ctx


def test_execute_retreat_runs_the_slide_then_lift_leg():
    sc = _fixture()
    ret = expand(sc.alias, sc)[-1]
    motions, ctx = _run_retreat(sc, ret)
    assert ctx["retreat_escape"] == {"skipped": "push_under: back along the push at its height, then up"}
    leg = motions.legs[-1]
    assert [p[2] for p in leg[:3]] == pytest.approx([0.05] * 3, abs=1e-5) and leg[2][1] == pytest.approx(2.895, abs=1e-5)
    assert leg[-1] == pytest.approx([6.5, 2.895, 0.15], abs=1e-5)


def test_retreat_witness_of_another_pose_is_not_used():
    """An old record's witness of the push-under retreat belongs to the straight lift off the goal (10 cm from the new leg's end): the
    finish must not plan back to it. A witness of the leg's own end is used, and the other retreats keep theirs as before."""
    sc = _fixture()
    acts = expand(sc.alias, sc)
    ret = acts[-1]
    q = th.arange(28, dtype=th.float32) / 100.0
    goal = th.tensor(ret.after.contact.goal.position, dtype=th.float32)
    quat = th.tensor(ret.after.contact.goal.orientation, dtype=th.float32)
    old = (goal + th.tensor([0.0, 0.0, 0.10]), quat)  # code_27ee9069's retreat witness pose: pre_pose(goal, 0.10)
    new = (th.tensor([6.5, 2.895, 0.15]), quat)
    motions, ctx = _run_retreat(sc, ret, {"retreat": q}, {"retreat": old})
    assert motions.finishes[-1].q_hint is None
    assert ctx["retreat_witness_dropped"] == {"recorded_pose_off_m": pytest.approx(0.10, abs=1e-4), "tolerance_m": 0.02}
    motions, ctx = _run_retreat(sc, ret, {"retreat": q}, {})  # a witness without its pose: not trusted either
    assert motions.finishes[-1].q_hint is None and ctx["retreat_witness_dropped"]["recorded_pose_off_m"] is None
    motions, ctx = _run_retreat(sc, ret, {"retreat": q}, {"retreat": new})
    assert motions.finishes[-1].q_hint is q and "retreat_witness_dropped" not in ctx
    straight = Retreat(arm=ret.arm, object=ret.object, after=ret.after)  # the default leg keeps its witness whatever pose it names
    motions, ctx = _run_retreat(sc, straight, {"retreat": q}, {"retreat": new})
    assert motions.finishes[-1].q_hint is q and "retreat_witness_dropped" not in ctx


def test_executor_reads_the_recorded_witness_poses():
    ex = Executor.__new__(Executor)
    ex.sidecar = SimpleNamespace(extra={"ik_witnesses": [
        {"action": 11, "name": "retreat", "targets": {"retreat": {"pose": {"position": [22.16885, 17.53457, 0.14037], "orientation": C3K0_QUAT}, "q": [0.0] * 28}}},
        {"action": 10, "name": "push", "targets": {"pre": {"pose": {"position": [21.86642, 17.53457, 0.14037], "orientation": C3K0_QUAT}, "q": [0.0] * 28},
                                                  "waypoints": {"q": [0.0] * 28}}},
    ]})
    poses = ex.witness_poses_for(11)
    assert list(poses) == ["retreat"] and poses["retreat"][0].tolist() == pytest.approx([22.16885, 17.53457, 0.14037])
    assert list(ex.witness_poses_for(10)) == ["pre"] and ex.witness_poses_for(3) == {}


# ----------------------------------------------------------------------------------------------------------------
# (b) the program check: the push-under legs after the drop keep the partner in the world
# ----------------------------------------------------------------------------------------------------------------


def _sampler(supports):
    objs = {n: SimpleNamespace(name=n) for n in (TARGET, FLOOR, PARTNER, "nightstand.n.01_1")}

    class Sampler:
        objects = objs

        @staticmethod
        def support_objects():
            return [objs[n] for n in supports]

    return Sampler, objs


def test_program_check_push_under_legs_never_ignore_the_partner():
    """On a bed-top init the init surface is the partner (``support_objects()`` = bed_rrcvaq_0, with the supports united with its top): the
    push and the retreats after the drop and after the push leave out the moved object and the floor only."""
    sc = _fixture()
    place, ret1, push, ret2 = expand(sc.alias, sc)[4:]
    sampler, objs = _sampler([PARTNER, "nightstand.n.01_1"])
    for a in (ret1, push, ret2):
        assert PC._push_under_floor(a) == FLOOR
        assert PC._ignore_for(a, sampler, None) == ([], [objs[TARGET], objs[FLOOR]]), a.name
    # the drop itself is unchanged: its descent ignores the supports and the floor it lands on
    assert PC._push_under_floor(place) is None
    last, every = PC._ignore_for(place, sampler, None)
    assert objs[PARTNER] in every and objs[FLOOR] in every
    # a plain push and its retreat keep the supports (block design, articulation, touching ...)
    plain = Push(contact=replace(push.contact, extra={**push.contact.extra, "kind": "push"}), base_yaw=0.0)
    assert PC._push_under_floor(plain) is None and objs[PARTNER] in PC._ignore_for(plain, sampler, None)[1]
    assert objs[PARTNER] in PC._ignore_for(Retreat(arm="right", object=TARGET, after=plain), sampler, None)[1]
    touching = Push(contact=replace(push.contact, extra={**push.contact.extra, "kind": "push_touching"}), base_yaw=0.0)
    assert PC._push_under_floor(touching) is None
    # the floor init (no supports): the same sets
    sampler0, objs0 = _sampler([])
    assert PC._ignore_for(push, sampler0, None) == ([], [objs0[TARGET], objs0[FLOOR]])


def test_program_check_slab_off_for_the_retreat_that_slides_at_the_push_height():
    sc = _fixture()
    place, ret1, push, ret2 = expand(sc.alias, sc)[4:]
    assert not PC._slab_off(push) and not PC._slab_off(ret2)
    push.contact.extra["ground_ignored"] = True  # the fingertips run below the slab's padding (< 4 cm above the floor)
    assert PC._slab_off(push) and PC._slab_off(ret2)
    assert not PC._slab_off(Retreat(arm="right", object=TARGET, after=push))  # a straight lift starts at the goal and rises
    assert not PC._slab_off(ret1) and not PC._slab_off(place)


# ----------------------------------------------------------------------------------------------------------------
# (c) yaw-free orientation of a push-under target
# ----------------------------------------------------------------------------------------------------------------


def test_vertical_angle_is_the_tilt_modulo_yaw():
    ident = th.tensor([0.0, 0.0, 0.0, 1.0])
    assert vertical_angle(_yaw(0.605), ident) == pytest.approx(0.0, abs=1e-4)
    assert vertical_angle(_roll(0.3), ident) == pytest.approx(0.3, abs=1e-4)
    assert vertical_angle(C.qmul(_yaw(0.605), _roll(0.12)), ident) == pytest.approx(0.12, abs=1e-4)
    goal = C.qmul(_yaw(0.0025), _roll(0.1))  # a goal tilted 0.1 rad: any world yaw of it is the goal
    assert vertical_angle(C.qmul(_yaw(1.0), goal), goal) == pytest.approx(0.0, abs=1e-4)
    assert vertical_angle(ident, goal) == pytest.approx(0.1, abs=1e-4)
    flipped = _roll(math.pi)
    assert vertical_angle(flipped, ident) == pytest.approx(math.pi, abs=1e-3)
    assert vertical_angle(flipped, ident, [th.tensor([1.0, 0.0, 0.0, 0.0])]) == pytest.approx(0.0, abs=1e-4)


def _gate_executor(actions, q_now, category="briefcase", model="lwbctq", goal=None):
    ex = Executor.__new__(Executor)
    ex.program_actions = actions
    ex.cfg = ExecutorConfig(demo_gating=("placed_orientation",))
    tgt = SimpleNamespace(frame="world", orientation=_yaw(0.0025) if goal is None else goal, orientation_tolerance=0.2, orientation_metric="full",
                          orientation_equivalences=[])
    obj = SimpleNamespace(name=f"target_{category}_{model}_0", category=category, model=model)
    ex.state = SimpleNamespace(pose_target=lambda o: tgt, object_pose=lambda o: (th.zeros(3), q_now), object=lambda o: obj)
    return ex


def test_placed_orientation_judges_a_push_under_target_modulo_yaw():
    sc = _fixture()
    acts = expand(sc.alias, sc)
    full = 0.605
    isg = {"errors": {"pose_errors": {TARGET: {"orientation_error": full, "gating": False}}}}
    ex = _gate_executor(acts, _yaw(0.0025 + full))  # c3_k0: 34.8 deg of yaw, lying flat
    assert ex.pushed_under_objects() == {TARGET}
    rep = ex.placed_orientation_report(isg)[TARGET]
    assert rep["within"] is True and rep["orientation_error"] < 1e-3 and rep["full_error"] == full and rep["tolerance"] == 0.2
    assert "world vertical" in rep["equivalence"] and "revolution_axis" not in rep
    # the tilt stays gated: 0.25 rad off the vertical fails, yawed or not
    rep = _gate_executor(acts, C.qmul(_yaw(0.6), _roll(0.25))).placed_orientation_report(isg)[TARGET]
    assert rep["within"] is False and rep["orientation_error"] == pytest.approx(0.25, abs=1e-3)
    # a program without the push-under push (a set-down under the partner) keeps the full error
    no_push = [a for a in acts if not isinstance(a, Push)]
    rep = _gate_executor(no_push, _yaw(0.0025 + full)).placed_orientation_report(isg)[TARGET]
    assert rep["within"] is False and rep["orientation_error"] == full and "equivalence" not in rep


def test_placed_orientation_takes_the_smaller_reading_for_a_pushed_cylinder(monkeypatch):
    from ispatialgym.oracle import executor as EXE

    monkeypatch.setattr(EXE, "revolution_axes", lambda path=EXE.REVOLUTION_AXES_PATH: {"trash_can/abc": {"axis": "x", "tags": ["xinf"]}})
    sc = _fixture()
    acts = expand(sc.alias, sc)
    isg = {"errors": {"pose_errors": {TARGET: {"orientation_error": 1.6, "gating": False}}}}
    # yawed: the long axis swings in the floor plane (revolution reading 0.6), the vertical one is ~0
    rep = _gate_executor(acts, _yaw(0.6), "trash_can", "abc", goal=th.tensor([0.0, 0.0, 0.0, 1.0])).placed_orientation_report(isg)[TARGET]
    assert rep["within"] is True and rep["orientation_error"] < 1e-3 and "world vertical" in rep["equivalence"]
    # rolled about its own long axis: the revolution reading (~0) stays, the vertical one (1.6) is larger
    rep = _gate_executor(acts, _roll(1.6), "trash_can", "abc", goal=th.tensor([0.0, 0.0, 0.0, 1.0])).placed_orientation_report(isg)[TARGET]
    assert rep["within"] is True and rep["revolution_axis"] == "x" and rep["orientation_error"] < 1e-3


def test_goal_override_judges_a_push_under_target_modulo_yaw():
    sc = _fixture()
    acts = expand(sc.alias, sc)
    goal_p, goal_q = th.tensor([6.5, 3.2, 0.03]), _yaw(0.0025)
    q_now = _yaw(0.0025 + 0.605)
    obj = SimpleNamespace(name="target_briefcase_lwbctq_0", links={}, aabb_extent=th.tensor([0.40, 0.12, 0.30]),
                          get_position_orientation=lambda: (goal_p + th.tensor([0.004, -0.003, 0.0]), q_now.clone()))
    tgt = SimpleNamespace(frame="world", link=None, position=goal_p, orientation=goal_q, position_metric="full", orientation_metric="full",
                          position_tolerance=0.03, orientation_tolerance=0.2, orientation_equivalences=[])
    compiled = SimpleNamespace(is_gating=lambda pred, inst: False)
    ex = Executor.__new__(Executor)
    ex.program_actions, ex.cfg, ex.state = acts, ExecutorConfig(), SimpleNamespace(object=lambda inst: obj)
    rec = ex._override_pose_target(TARGET, tgt, compiled)
    assert rec["applied"] is True and rec["orientation_error"] < 1e-3 and rec["orientation_error_full"] == pytest.approx(0.605, abs=1e-3)
    assert rec["orientation"] == pytest.approx(q_now.tolist(), abs=1e-6) and "world vertical" in rec["equivalence"]
    ex.program_actions = [a for a in acts if not isinstance(a, Push)]
    rec = ex._override_pose_target(TARGET, tgt, compiled)
    assert rec["applied"] is False and rec["orientation_error"] == pytest.approx(0.605, abs=1e-3) and "equivalence" not in rec
