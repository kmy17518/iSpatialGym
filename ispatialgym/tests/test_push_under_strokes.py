"""The push-under push that does not turn the object (owner 2026-10-13 ~04:30 item 2, under_large#01): the oracle's strokes with re-aim and
the two-point gripper against the fake world of ``test_v2_oracle`` (a sandal whose round heel the blade meets, turning with a contact off its
centre line), the stroke helpers, and the record's stroke plan / contact forms (no simulator)."""

from __future__ import annotations

import math
import re
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle.actions import Push
from ispatialgym.oracle.executor import SimState
from ispatialgym.oracle.motions import Motions, MotionsConfig
from ispatialgym.oracle.sidecar import Pose, PushContact
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2.tasks import iors_large as IL
from ispatialgym.tests.test_v2_oracle import FakeBackend, FakeEnv, FakeObject, FakeRobot

OBJ = "sandal.n.01_1"
HEEL_R, HEEL_C, HEEL_TOP = 0.035, 0.093, 0.035  # the heel: a half disc of radius 3.5 cm whose centre is 9.3 cm behind the root, 3.5 cm high
TIP_Z = 0.0216
DOWN = th.tensor([0.0, 0.0, -1.0])


def sandal_points(n: int = 19) -> th.Tensor:
    """The synthetic sandal chgcrm in its root frame (heel 0.128 m behind the root up to 3.5 cm, toe 0.117 m ahead up to 4.5 cm, the strap
    2.7 cm behind the root 7.2 cm up)."""
    pts = []
    for k in range(n):
        a = math.pi / 2 + math.pi * k / (n - 1)
        pts += [[-HEEL_C + HEEL_R * math.cos(a), HEEL_R * math.sin(a), z] for z in (0.0, HEEL_TOP)]
        b = -math.pi / 2 + math.pi * k / (n - 1)
        pts += [[0.077 + 0.04 * math.cos(b), 0.04 * math.sin(b), z] for z in (0.0, 0.045)]
    pts += [[-0.027, s * 0.04, 0.072] for s in (-1, 1)] + [[0.03, s * 0.045, 0.072] for s in (-1, 1)]
    return th.tensor(pts, dtype=th.float32)


class _Link:
    def __init__(self, obj, live: bool):
        self.obj, self.live = obj, live
        self.prim_path, self.body_name = f"/World/{obj.name}/base_link", "base_link"

    def get_position_orientation(self):
        return self.obj.pos.clone(), self.obj.quat.clone()

    @property
    def collision_boundary_points_world(self):
        if not self.live:
            return None
        return C.transform_points((self.obj.pos, self.obj.quat), sandal_points())


def sandal_world(yaw_deg: float = 5.33, off_m: float = 0.0085, turn_k: float = 150.0, live: bool = True):
    """The left blade pushing a sandal along +x on the floor (the robot's base at the origin facing +x): the sandal's root at (0.6, off_m),
    turned ``yaw_deg``. Each step the blade's 14 mm face (or a two-point hand's fingertips) that reaches past the heel's arc moves the sandal
    along +x by the overlap and turns it by ``-turn_k x (contact lateral - root lateral) x overlap`` (a contact right of the root's line turns
    it counter-clockwise, as c3_k0's did); ``live``: the sandal exposes its collision points (else the oracle uses the record's offset)."""
    robot = FakeRobot(base=(0.0, 0.0, 0.0))
    sandal = FakeObject("target_sandal_chgcrm_0", [0.6, off_m, 0.0], quat=C.quat_from_yaw(math.radians(yaw_deg)).tolist(), category="sandal",
                        extent=(0.245, 0.09, 0.072))
    sandal.graspable = False
    sandal.links = {"base_link": _Link(sandal, live)}
    env = FakeEnv(robot, {OBJ: sandal})
    goal = th.tensor([0.89, 0.0, 0.0])
    env.task._tro = SimpleNamespace(meta=SimpleNamespace(pose_targets={OBJ: SimpleNamespace(
        frame="world", position=goal, orientation=C.IDENTITY_QUAT.clone(), position_tolerance=0.03, orientation_tolerance=0.2, orientation_equivalences=[],
        orientation_metric="full")}))
    hand = {"strips": EX.PUSH_BLADE_STRIPS_M, "face": EX.PUSH_BLADE_FACE_HALF_M}

    def react(_action):
        p, q = robot.eef_pose("left")
        yaw = C.yaw_of(sandal.quat)
        u = th.tensor([math.cos(yaw), math.sin(yaw)])
        cx, cy = (sandal.pos[:2] - HEEL_C * u).tolist()
        tip_z = float(p[2]) + C.FINGERTIP_LENGTH * float(C.approach_axis(q)[2])
        if tip_z > HEEL_TOP:
            return
        hits = []
        for c, hw in hand["strips"]:
            y1, y2 = float(p[1]) + c - hw, float(p[1]) + c + hw
            ye = min(max(cy, y1), y2)
            if abs(ye - cy) >= HEEL_R:
                continue
            pen = float(p[0]) + hand["face"] - (cx - math.sqrt(HEEL_R ** 2 - (ye - cy) ** 2))
            if pen > 0:
                hits.append((pen, ye))
        if not hits:
            return
        pen = max(h[0] for h in hits)
        touching = [h[1] for h in hits if h[0] > pen - 1e-3]  # fingertips that both touch share the load
        ye = sum(touching) / len(touching)
        sandal.pos = sandal.pos + th.tensor([pen, 0.0, 0.0])
        sandal.quat = C.quat_from_yaw(yaw - turn_k * (ye - float(sandal.pos[1])) * pen)

    env.after_step = react
    return env, robot, sandal, goal, hand


def push_action(extra=None, form_quat=None, standoff=0.023, off=0.128):
    """The c3_k0 push-under record moved to the fake world: the drop cell (0.6, 0), the goal cell (0.89, 0), the blade 2.3 cm behind the
    heel at the recorded tip height."""
    q = form_quat if form_quat is not None else C.eef_quat(DOWN, th.tensor([-1.0, 0.0, 0.0]))
    z = TIP_Z + C.FINGERTIP_LENGTH
    start = th.tensor([0.6 - (off + standoff), 0.0, z])
    goal = th.tensor([0.89 - (off + standoff - 0.01), 0.0, z])
    pose = lambda p: Pose(position=[float(v) for v in p], orientation=[float(v) for v in q])  # noqa: E731
    ex = {"kind": "push_under", "blade_standoff_m": standoff, "half_extent_m": off, "fingertip_z_m": TIP_Z, "push_height_m": z, "drop_cell": [0.6, 0.0],
          "goal_cell": [0.89, 0.0], "partner": "bed.n.01_1", **(extra or {})}
    c = PushContact(arm="left", object=OBJ, pre=pose(start + th.tensor([0.0, 0.0, 0.10])), init=pose(start), mid=pose(0.5 * (start + goal)), goal=pose(goal),
                    direction=[1.0, 0.0, 0.0], distance_m=0.29, extra=ex)
    return Push(contact=c, base_yaw=0.0, touches=("floor.n.01_1",), avoid=("bed.n.01_1",))


def run_push(env, robot, push):
    backend = FakeBackend(robot)
    motions = Motions(env, backend, MotionsConfig(settle_steps=5, gripper_steps=5, cartesian_step=0.01, cartesian_rot_step=0.3))
    state = SimState(env, env.task, robot)
    ctx: dict = {}
    n = 0
    for a in EX.run_action(push, motions, state, push.targets(None, state), ctx):
        env.step(a)
        n += 1
        assert n < 20000
    return ctx, motions


def plan(gain: float = 0.0, **kw):
    return IL.push_stroke_plan(0.29, 0.12, backoff_m=0.03, reaim_max_m=0.03, yaw_gain_m_per_rad=gain, yaw_deadband_rad=math.radians(2.0), **kw)


# ----------------------------------------------------------------------------------------------------------------
# the helpers
# ----------------------------------------------------------------------------------------------------------------


def test_stroke_plan_from_the_key_and_the_oracle_reading():
    p = plan()
    assert p["n"] == 3 and p["ends_m"] == pytest.approx([0.09667, 0.19333, 0.29], abs=1e-5)
    assert IL.push_stroke_plan(0.29, 0.15)["n"] == 2 and IL.push_stroke_plan(0.29, 0.05)["n"] == IL.PUSH_STROKES_MAX
    assert IL.push_stroke_plan(0.29, 0.05, n_max=5)["n"] == 5 and IL.push_stroke_plan(0.10, 0.12)["n"] == 1
    got = EX.push_stroke_plan({"strokes": {"n": 2, "ends_m": [0.1, 0.2]}})
    assert got == {"n": 2, "ends_m": [0.1, 0.2], **EX.PUSH_STROKE_DEFAULTS}  # the defaults fill what the record leaves out
    assert EX.push_stroke_plan({"strokes": {**p, "yaw_gain_m_per_rad": 0.2}})["yaw_gain_m_per_rad"] == 0.2
    for bad in ({}, {"strokes": None}, {"strokes": {"n": 0, "ends_m": []}}, {"strokes": {"n": 2, "ends_m": [0.2]}},
                {"strokes": {"n": 2, "ends_m": [0.2, 0.1]}}, {"strokes": {"n": "x"}}):
        assert EX.push_stroke_plan(bad) is None


def test_stroke_target_follows_the_root_and_turns_the_object_back():
    p = EX.push_stroke_plan({"strokes": plan(gain=0.15)})
    # c3_k0's landing: the root 8.5 mm left of the recorded line, turned 5.33 deg counter-clockwise -> the contact left of the root by 0.15 x 0.093
    target, off, term = EX.push_stroke_aim((0.6, 0.0085), (0.4, 0.0), (1.0, 0.0), math.radians(5.33), p)
    assert off == pytest.approx(0.0085) and term == pytest.approx(0.15 * math.radians(5.33)) and target == pytest.approx(off + term)
    assert EX.push_stroke_aim((0.6, 0.0085), (0.4, 0.0), (1.0, 0.0), math.radians(1.5), p)[2] == 0.0  # inside the deadband: no yaw term
    assert EX.push_stroke_aim((0.6, 0.0085), (0.4, 0.0), (1.0, 0.0), -math.radians(5.33), p)[2] < 0.0  # a clockwise error: the contact to the right
    # another push direction: +y, the root 1 cm toward -x = left of the push
    assert EX.push_stroke_aim((-0.01, 0.6), (0.0, 0.4), (0.0, 1.0), 0.0, p)[1] == pytest.approx(0.01)


def test_contact_aim_puts_the_first_contact_on_the_target():
    yaw = math.radians(5.33)
    pts = C.transform_points((th.tensor([0.6, 0.0085, 0.0]), C.quat_from_yaw(yaw)), sandal_points(37))
    rear_lat = 0.0085 - HEEL_C * math.sin(yaw)  # the heel's rearmost point: behind its arc centre
    band = (TIP_Z, TIP_Z + 0.05)
    # the target (the root's line) 8.6 mm left of the heel's rearmost point: the blade's right edge on it
    aim, along, lat = EX.push_contact_aim(pts, (0.4, 0.0), (1.0, 0.0), 0.0085, EX.PUSH_BLADE_STRIPS_M, *band)
    assert lat == pytest.approx(0.0085, abs=6e-4) and aim == pytest.approx(0.0085 + 0.007, abs=6e-4)
    # a target on the rearmost point: the blade centred on it; right of it: the blade's left edge on it
    aim, _, lat = EX.push_contact_aim(pts, (0.4, 0.0), (1.0, 0.0), rear_lat, EX.PUSH_BLADE_STRIPS_M, *band)
    assert lat == pytest.approx(rear_lat, abs=6e-4) and aim == pytest.approx(rear_lat, abs=1e-3)
    aim, _, lat = EX.push_contact_aim(pts, (0.4, 0.0), (1.0, 0.0), -0.015, EX.PUSH_BLADE_STRIPS_M, *band)
    assert lat == pytest.approx(-0.015, abs=6e-4) and aim == pytest.approx(-0.015 - 0.007, abs=6e-4)
    # a two-point hand straddles the target; nothing in reach: None
    two = ((-0.023, 0.0042), (0.023, 0.0042))
    assert EX.push_contact_aim(pts, (0.4, 0.0), (1.0, 0.0), 0.0085, two, *band)[0] == pytest.approx(0.0085)
    assert EX.push_contact_aim(pts, (0.4, 0.0), (1.0, 0.0), 0.3, EX.PUSH_BLADE_STRIPS_M, *band) is None


def test_back_along_meets_the_round_heel_within_the_blade():
    pts = C.transform_points((th.tensor([0.6, 0.0, 0.0]), C.IDENTITY_QUAT), sandal_points())
    line = (0.4, 0.0)
    # the blade on the centre line meets the heel's back 0.128 m behind the root; 2 cm aside the arc is 6.3 mm farther ahead
    assert EX.push_back_along(pts, line, (1.0, 0.0), 0.0, EX.PUSH_BLADE_STRIPS_M, TIP_Z, TIP_Z + 0.05) == pytest.approx(0.2 - 0.128, abs=2e-4)
    side = 0.6 - HEEL_C - math.sqrt(HEEL_R ** 2 - 0.013 ** 2) - 0.4  # the strip's inner edge 1.3 cm off the heel's centre line
    assert EX.push_back_along(pts, line, (1.0, 0.0), 0.02, EX.PUSH_BLADE_STRIPS_M, TIP_Z, TIP_Z + 0.05) == pytest.approx(side, abs=1e-3)
    assert EX.push_back_along(pts, line, (1.0, 0.0), 0.2, EX.PUSH_BLADE_STRIPS_M, TIP_Z, TIP_Z + 0.05) is None  # beside the sandal
    assert EX.push_back_along(pts, line, (1.0, 0.0), 0.0, EX.PUSH_BLADE_STRIPS_M, 0.08, 0.1) is None  # nothing in the band


def test_contact_forms_geometry():
    pts = sandal_points()
    blade = IL.push_form_contact("blade", pts, (1.0, 0.0), TIP_Z, TIP_Z + 0.05)
    assert blade["off_m"] == pytest.approx(0.128, abs=2e-4) and blade["face_half_m"] == pytest.approx(0.0084) and blade["strips_m"] == [[0.0, 0.007]]
    turned = IL.push_form_contact("closed_turned", pts, (1.0, 0.0), TIP_Z, TIP_Z + 0.05)
    assert turned["off_m"] == pytest.approx(0.128, abs=2e-4) and turned["face_half_m"] == pytest.approx(0.007) and turned["strips_m"] == [[0.0, 0.0084]]
    two = IL.push_form_contact("two_point", pts, (1.0, 0.0), TIP_Z, TIP_Z + 0.05)
    # the heel 2 cm ahead of its back is 2 x sqrt(3.5^2 - 1.5^2) = 6.3 cm wide -> the fingertips 0.6 x 6.3 = 3.8 cm apart, each meeting the arc
    # where it is (1.9 + 0.42) cm off the centre line
    w = 2.0 * math.sqrt(HEEL_R ** 2 - (HEEL_R - IL.PUSH_TWO_POINT_DEPTH_M) ** 2)
    assert two["opening_m"] == pytest.approx(IL.PUSH_TWO_POINT_WIDTH_FRAC * w, abs=1.5e-3)
    inner = 0.5 * two["opening_m"]
    assert two["off_m"] == pytest.approx(HEEL_C + math.sqrt(HEEL_R ** 2 - inner ** 2), abs=1e-3)
    assert two["strips_m"][1][0] == pytest.approx(inner + 0.0042, abs=1e-5) and two["fingers_m"] == pytest.approx(IL.finger_joint_for_opening(two["opening_m"]))
    fixed = IL.push_form_contact("two_point", pts, (1.0, 0.0), TIP_Z, TIP_Z + 0.05, opening=0.05)
    assert fixed["opening_m"] == 0.05
    assert IL.push_form_contact("two_point", pts, (1.0, 0.0), 0.08, 0.1) is None  # nothing in the band
    assert IL.finger_joint_for_opening(C.GRIPPER_MAX_OPENING_M) == pytest.approx(C.FINGER_OPEN)
    assert IL.finger_joint_for_opening(0.0) == pytest.approx(0.5 * (2 * C.FINGER_OPEN - C.GRIPPER_MAX_OPENING_M))
    assert IL.finger_joint_for_opening(0.2) == C.FINGER_OPEN


# ----------------------------------------------------------------------------------------------------------------
# the oracle against the fake sandal
# ----------------------------------------------------------------------------------------------------------------


def _yaw_deg(sandal) -> float:
    return math.degrees(C.yaw_of(sandal.quat))


def test_single_stroke_turns_the_sandal_on_the_recorded_line():
    env, robot, sandal, goal, _ = sandal_world()
    ctx, _ = run_push(env, robot, push_action())
    assert "push_strokes" not in ctx
    assert _yaw_deg(sandal) > 25.0  # the fake's turning rule: the heel met right of the root's line all the way


def test_strokes_reaim_at_the_root_line_and_keep_the_yaw(caplog):
    env, robot, sandal, goal, _ = sandal_world()
    with caplog.at_level("INFO", logger="ispatialgym.oracle.execute"):
        ctx, _ = run_push(env, robot, push_action({"strokes": plan()}))
    rep = ctx["push_strokes"]
    assert rep["n"] == 3 and len(rep["strokes"]) == 3 and rep["yaw_reference"] == "goal" and rep["yaw_start_deg"] == pytest.approx(5.33, abs=0.01)
    s1 = rep["strokes"][0]
    assert s1["contact"] == "the object's back" and s1["root_off_line_m"] == pytest.approx(0.0085, abs=1e-4) and s1["target_m"] == pytest.approx(0.0085, abs=1e-4)
    # the heel's rearmost point lies 8.6 mm right of the root's line (its arc centre 9.3 cm behind the root, turned 5.33 deg): the 14 mm
    # blade stands with its right edge on the root's line, so the first contact is there and not at the blade's centre
    assert s1["contact_lateral_m"] == pytest.approx(0.0085, abs=6e-4) and s1["aim_m"] == pytest.approx(0.0085 + 0.007, abs=6e-4)
    assert s1["yaw_term_m"] == 0.0 and "reaim_failed" not in s1
    # every stroke pushed a third of the way, the last one to the goal line; the yaw stayed within 2 deg of the start (the contact within
    # 1 mm of the root's line; the single stroke: > 25 deg)
    assert [round(s["pushed_m"], 2) for s in rep["strokes"][:2]] == [pytest.approx(0.097, abs=0.01)] * 2
    assert abs(_yaw_deg(sandal) - 5.33) < 2.0
    assert float(goal[0] - sandal.pos[0]) < 0.01  # the residual / re-push / base nudge after the strokes as before
    assert re.search(r"push\(sandal\.n\.01_1\): stroke 1/3: yaw \+5\.3 deg vs the goal, root \+8\.5 mm off the line \(yaw term \+0\.0 mm\) -> contact \+[89]\.\d mm, "
                     r"blade \+1[45]\.\d mm across \(contact from the object's back\); pushed 9\d\.\d mm, yaw \+\d\.\d deg", caplog.text)
    assert "push(sandal.n.01_1): 3 of 3 strokes, yaw +5.3 -> " in caplog.text


def test_strokes_yaw_gain_moves_the_contact():
    # the knob only (sim check 17838968: on the sandal's round heel +0.15 and -0.15 both turned it further; the default is 0)
    env, robot, sandal, goal, _ = sandal_world()
    ctx, _ = run_push(env, robot, push_action({"strokes": plan(gain=0.15)}))
    s1 = ctx["push_strokes"]["strokes"][0]
    assert s1["yaw_term_m"] == pytest.approx(0.15 * math.radians(5.33), abs=1e-4) and s1["target_m"] == pytest.approx(0.0085 + s1["yaw_term_m"], abs=1e-4)
    assert s1["contact_lateral_m"] == pytest.approx(s1["target_m"], abs=1.5e-3)  # the aim search's 1 mm bins, the record's 0.1 mm rounding


def test_strokes_keep_the_blade_within_the_reaim_band():
    env, robot, sandal, goal, _ = sandal_world(yaw_deg=0.0, off_m=0.05)
    ctx, _ = run_push(env, robot, push_action({"strokes": plan()}))
    s1 = ctx["push_strokes"]["strokes"][0]
    assert s1["target_m"] == pytest.approx(0.05, abs=1e-4) and s1["aim_m"] == pytest.approx(0.03)  # reaim_max_m from the recorded line


def test_strokes_without_live_points_use_the_recorded_offset():
    env, robot, sandal, goal, _ = sandal_world(live=False)
    ctx, _ = run_push(env, robot, push_action({"strokes": plan()}))
    assert {s["contact"] for s in ctx["push_strokes"]["strokes"]} == {"the recorded contact offset"}
    assert float(goal[0] - sandal.pos[0]) < 0.01


def test_failed_reaim_falls_back_to_the_recorded_line(monkeypatch):
    env, robot, sandal, goal, _ = sandal_world()
    real = Motions.arm_along
    seen = {"n": 0}

    def arm_along(self, arm, waypoints, loaded=False, **kw):
        p = as_world(waypoints[-1])
        if not loaded and abs(float(p[1])) > 1e-6 and seen["n"] == 0:
            seen["n"] += 1
            raise EX.SkillError("plan_failed", "the re-aimed start is out of reach (test)")
        return (yield from real(self, arm, waypoints, loaded=loaded, **kw))

    monkeypatch.setattr(Motions, "arm_along", arm_along)
    ctx, _ = run_push(env, robot, push_action({"strokes": plan()}))
    s1 = ctx["push_strokes"]["strokes"][0]
    assert "out of reach" in s1["reaim_failed"] and s1["aim_m"] == 0.0 and len(ctx["push_strokes"]["strokes"]) == 3


def as_world(pose) -> th.Tensor:
    return th.as_tensor(pose.position if hasattr(pose, "position") else pose[0], dtype=th.float32)


def test_two_point_push_opens_the_gripper_to_the_record():
    env, robot, sandal, goal, hand = sandal_world(yaw_deg=0.0, off_m=0.0)
    fc = IL.push_form_contact("two_point", sandal_points(), (1.0, 0.0), TIP_Z, TIP_Z + 0.05)
    hand.update({"strips": [tuple(s) for s in fc["strips_m"]], "face": fc["face_half_m"]})
    q = C.eef_quat(DOWN, th.tensor([0.0, 1.0, 0.0]))
    push = push_action({"push_contact_form": "two_point", "fingers_opening_m": fc["fingers_m"], "contact_strips_m": fc["strips_m"], "face_half_m": fc["face_half_m"]},
                       form_quat=q, standoff=0.019, off=fc["off_m"])
    ctx, motions = run_push(env, robot, push)
    assert ctx["push_contact_form"] == {"form": "two_point", "fingers_opening_m": pytest.approx(fc["fingers_m"])}
    assert float(robot.q[robot.gripper_control_idx["left"]].mean()) == pytest.approx(fc["fingers_m"], abs=1e-3)
    assert motions.gripper_intent_closed["left"] is False
    assert float(goal[0] - sandal.pos[0]) < 0.01 and abs(_yaw_deg(sandal)) < 0.5


def test_without_the_keys_the_push_is_unchanged():
    env, robot, sandal, goal, _ = sandal_world(yaw_deg=0.0, off_m=0.0)
    ctx, motions = run_push(env, robot, push_action())
    assert "push_strokes" not in ctx and "push_contact_form" not in ctx and motions.gripper_intent_closed["left"] is True


# ----------------------------------------------------------------------------------------------------------------
# the sampler: the keys, the chain's contact forms and the record
# ----------------------------------------------------------------------------------------------------------------


class _GP:
    def __init__(self, **gen):
        self.variant, self.generation = "inter_object_relation_satisfaction-under_large_scale-sandal-bed-bed_iorsl", dict(gen)

    def gen(self, key, default=None):
        return self.generation.get(key, default)


def _sampler(fails=(), **gen):
    """The blade chain of ``IORSLarge._push_under_chain`` on the synthetic sandal (tips at push_tip_frac 0.3 = 2.16 cm) with a stub IK that
    fails every proposal whose label starts with one of ``fails``; ``s.ik_calls`` keeps (label, fingers) per call."""
    import contextlib

    s = IL.IORSLarge.__new__(IL.IORSLarge)
    hull = sandal_points()
    s.gp, s.hull, s.floor_z, s.seg_arm = _GP(push_tip_frac=0.3, **gen), hull, 0.0, "left"
    s.place = {"height": float(hull[:, 2].max() - hull[:, 2].min())}
    s.obj = lambda role: role
    s.kit = SimpleNamespace(floor_z=0.0, stats={})
    s.ik_calls = []
    s.ikc = SimpleNamespace(ignoring=lambda objs, ground=True: contextlib.nullcontext(), q_locked=lambda place, **kw: None,
                            exact=lambda chain, place, arm, **kw: SimpleNamespace(ok=True, reason=""))
    s.fails = tuple(fails)
    return s


@pytest.fixture
def chooser(monkeypatch):
    """``orient.choose`` taking the first proposal the stub IK passes (None when it fails them all); records every call's labels."""
    calls = []

    def choose(ikc, arm, props, evaluate, q_start, camera_idx=None, stage="", downstream=None):
        calls.append([p.label for p in props])
        for p in props:
            ok = not any(p.label.startswith(f) for f in getattr(choose, "fails", ()))
            evaluate(p)
            if ok:
                return SimpleNamespace(proposal=p), {"options": [{"label": p.label, "ok": True}]}
        return None, {"options": [{"label": p.label, "ok": False, "why": "stub IK"} for p in props]}

    monkeypatch.setattr(IL.O, "choose", choose)
    choose.calls = calls
    return choose


PLACE_C = SimpleNamespace(stand_off=0.8, lateral=-0.15, translate=2.0, tilt=0.0, base_pose2d=(0.0, 0.0, 0.0))
DROP_C, GOAL_C = (0.6, 0.0), (0.89, 0.0)


def _fingers_seen(s, monkeypatch):
    seen = []
    real = s.ikc.exact

    def exact(chain, place, arm, **kw):
        seen.append(kw.get("fingers"))
        return real(chain, place, arm, **kw)

    s.ikc.exact = exact
    return seen


def test_contact_forms_key():
    assert _sampler().push_contact_forms() == ["blade"]
    assert _sampler(push_under_contact="two_point").push_contact_forms() == ["two_point", "blade"]
    assert _sampler(push_under_contact=["closed_turned", "two_point"]).push_contact_forms() == ["closed_turned", "two_point", "blade"]
    assert _sampler(push_under_contact=["blade", "two_point"]).push_contact_forms() == ["blade", "two_point"]  # the list places the blade
    for bad in ("wide", ["two_point", "flat"], [], 3):
        with pytest.raises(IL.Reject) as e:
            _sampler(push_under_contact=bad).push_contact_forms()
        assert e.value.reason == "spec"


def test_without_the_contact_key_the_blade_chain_is_unchanged(chooser, monkeypatch):
    s = _sampler()
    seen = _fingers_seen(s, monkeypatch)
    ch = s._push_under_chain(PLACE_C, C.IDENTITY_QUAT, DROP_C, GOAL_C, (1.0, 0.0))
    assert "record" not in ch and chooser.calls[0] == ["pitch 0", "pitch 0-"] and set(seen) == {C.FINGER_CLOSED}
    assert ch["half_extent_m"] == pytest.approx(0.128, abs=2e-4) and ch["blade_standoff_m"] == pytest.approx(0.018 + IL.PUSH_UNDER_CLEARANCE_M)
    assert C.closing_axis(ch["start"][1]).tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)  # the closing axis along the push


def test_two_point_chain_and_record(chooser, monkeypatch, caplog):
    s = _sampler(push_under_contact="two_point")
    seen = _fingers_seen(s, monkeypatch)
    with caplog.at_level("INFO", logger="ispatialgym.sampling.v2"):
        ch = s._push_under_chain(PLACE_C, C.IDENTITY_QUAT, DROP_C, GOAL_C, (1.0, 0.0))
    fc = IL.push_form_contact("two_point", sandal_points(), (1.0, 0.0), 0.0216, 0.0716)
    rec = ch["record"]
    assert rec["push_contact_form"] == "two_point" and rec["fingers_opening_m"] == pytest.approx(fc["fingers_m"], abs=1e-5)
    assert rec["contact_width_m"] == pytest.approx(fc["opening_m"], abs=1e-4) and rec["contact_strips_m"] == fc["strips_m"]
    assert rec["face_half_m"] == pytest.approx(0.007)
    assert chooser.calls[0] == ["two_point pitch 0", "two_point pitch 0-"] and seen and all(f == pytest.approx(fc["fingers_m"], abs=1e-5) for f in seen)
    # the hand stands the side profile's 14 mm + 5 mm behind where its first fingertip meets the heel, the closing axis across the push
    assert ch["half_extent_m"] == pytest.approx(fc["off_m"]) and ch["blade_standoff_m"] == pytest.approx(0.014 + IL.PUSH_UNDER_CLEARANCE_M)
    assert float(ch["start"][0][0]) == pytest.approx(0.6 - fc["off_m"] - 0.019, abs=1e-5)
    assert C.closing_axis(ch["start"][1]).tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    assert C.approach_axis(ch["start"][1]).tolist() == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)
    assert s.kit.stats == {"push_contact_form.two_point": 1}
    assert "push-under contact form two_point (generation.push_under_contact ['two_point', 'blade']): pitch 0, wrist +1, stand-off 0.019 m" in caplog.text
    # the record carries it to the oracle and the goal check: the opening, through the sidecar
    push = IL.PushUnderRecord("left", ch["pre"], ch["start"], ch["mid"], ch["goal"], ch["direction"], ch["distance_m"], OBJ, pitch_deg=0.0,
                              blade_standoff_m=ch["blade_standoff_m"], half_extent_m=ch["half_extent_m"], fingertip_z_m=ch["fingertip_z_m"], **rec)
    pc = PushContact.from_dict(push.to_sidecar(), "contacts.push")
    assert pc.extra["push_contact_form"] == "two_point" and IL.push_record_fingers(pc.extra) == pytest.approx(fc["fingers_m"], abs=1e-5)
    assert IL.push_record_fingers({}) == C.FINGER_CLOSED and IL.push_record_fingers({"push_contact_form": "closed_turned"}) == C.FINGER_CLOSED


def test_contact_forms_fall_back_in_order(chooser, monkeypatch):
    chooser.fails = ("two_point",)
    s = _sampler(push_under_contact=["two_point", "closed_turned"])
    ch = s._push_under_chain(PLACE_C, C.IDENTITY_QUAT, DROP_C, GOAL_C, (1.0, 0.0))
    assert ch["record"]["push_contact_form"] == "closed_turned" and "fingers_opening_m" not in ch["record"]
    # pitch 45 / 60: the 7.2 cm sandal's face reaches the side profile's palm section (blade_standoff side=True): not tried
    assert [c[0] for c in chooser.calls] == ["two_point pitch 0", "two_point pitch 30", "closed_turned pitch 0"]
    assert ch["half_extent_m"] == pytest.approx(0.128, abs=2e-4) and ch["blade_standoff_m"] == pytest.approx(0.014 + IL.PUSH_UNDER_CLEARANCE_M)
    chooser.fails = ("two_point", "closed_turned")
    ch = _sampler(push_under_contact=["two_point", "closed_turned"])._push_under_chain(PLACE_C, C.IDENTITY_QUAT, DROP_C, GOAL_C, (1.0, 0.0))
    assert ch["record"] == {"push_contact_form": "blade"} and ch["blade_standoff_m"] == pytest.approx(0.023)  # today's blade, last


def test_stroke_key_and_record():
    assert _sampler().push_strokes_record(0.29) == {}
    rec = _sampler(push_under_stroke_m=0.12, push_under_reaim_yaw_gain=0.1)
    plan_ = rec.push_strokes_record(0.2924)["strokes"]
    assert plan_["n"] == 3 and plan_["ends_m"][-1] == pytest.approx(0.2924) and plan_["yaw_gain_m_per_rad"] == pytest.approx(0.1)
    assert plan_["backoff_m"] == pytest.approx(0.03) and plan_["reaim_max_m"] == pytest.approx(0.03) and plan_["yaw_deadband_rad"] == pytest.approx(math.radians(2.0), abs=1e-5)
    assert _sampler(push_under_stroke_m=0.15).push_strokes_record(0.29)["strokes"]["n"] == 2
    # the open two-point hand pushes in one stroke (sim check 17838968: re-aimed, one fingertip met the heel first and the sandal turned 80 deg)
    assert rec.push_strokes_record(0.29, "two_point") == {} and rec.push_strokes_record(0.29, "closed_turned")["strokes"]["n"] == 3
    assert _sampler(push_under_stroke_m=0.05, push_under_strokes_max=4).push_strokes_record(0.29)["strokes"]["n"] == 4
    for bad in (0.0, -0.1, "0.1", True):
        with pytest.raises(IL.Reject):
            _sampler(push_under_stroke_m=bad).push_strokes_record(0.29)
    # the plan reaches the oracle unchanged through the sidecar
    q = C.eef_quat(DOWN, th.tensor([-1.0, 0.0, 0.0]))
    pose = (th.tensor([0.45, 0.0, 0.04]), q)
    d = IL.PushUnderRecord("left", pose, pose, pose, pose, th.tensor([1.0, 0.0, 0.0]), 0.2924, OBJ, **rec.push_strokes_record(0.2924)).to_sidecar()
    assert EX.push_stroke_plan(PushContact.from_dict(d, "contacts.push").extra) == pytest.approx(plan_)


def test_program_check_two_point_fingers():
    from ispatialgym.sampling.v2 import program_check as PC

    two = push_action({"push_contact_form": "two_point", "fingers_opening_m": 0.0204})
    assert PC._two_point_fingers(two, "push", None) == pytest.approx(0.0204)
    assert PC._two_point_fingers(SimpleNamespace(after=two), "retreat", "push") == pytest.approx(0.0204)
    assert PC._two_point_fingers(push_action(), "push", None) is None
    assert PC._two_point_fingers(push_action({"push_contact_form": "closed_turned"}), "push", None) is None
    assert PC._two_point_fingers(SimpleNamespace(after=None), "retreat", "place") is None
