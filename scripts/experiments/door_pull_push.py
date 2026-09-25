"""Door pull-back → push-open feasibility experiment (owner request 2026-09-19 night).

Run through ``run_isg`` with ``ISG_PROGRAM_HOOK=<this file>:hook`` on a large-articulation *door open* instance whose
recorded program is ``navigate, transition, grasp, base_follow, retreat``. The hook keeps the first three actions and
replaces the recorded (hybrid) follow by one experimental action :class:`PullPushOpen`:

1. **pull** — the arm is locked on the handle (its joint targets frozen at the grasp configuration) and the base
   drives straight back along its own −x in ``pull_step_m`` steps; the eef follows the base rigidly, the door is
   dragged open along its arc, the arm winds up laterally. The pull stops when the arm's joint wind-up exceeds
   ``pull_deflection_rad`` ("the arm is at its limit"), the base tracker stalls, the grasp is lost, the door stops
   moving or ``pull_max_m`` is travelled.
2. **release** — gripper open + assisted grasp released, the hand backs off 0.10 m along −a, then the gripper closes
   (the closed hand is the pusher).
3. **reposition** — the base drives to the free-edge side of the leaf (the robot's left for this door) to a stance
   from which the closed hand reaches around the free edge onto the **inside** face.
4. **push** — closed-hand contact on the inside face ``push_inset_m`` from the free edge at ``contact_height_m``, then
   an arc about the hinge by ``push_delta_rad`` (the leaf swings further open, away from the robot).

The outcome (door angle after every phase, the stop reason of the pull, IK / tracker failures) goes to the action's
``notes["pull_push"]`` in ``meta.json``. The episode's own success flag is judged against the *recorded* goal angle
and is not the experiment's verdict.

Geometry: the hinge is fitted from the sidecar's handle contact points at three joint values (a circle in the plane
⊥ the axis); the leaf width and thickness come from the leaf link's AABB at the closed pose at run time. Every knob has
an ``ISG_PP_*`` environment override (see :func:`hook`).

The door *close* by pushing the outer face is the registry's program for a room door (``PushClose`` in
``oracle.actions`` / ``oracle.execute``, ``programs.program_articulation_large``); :func:`hook_close` only applies the
``ISG_PP_*`` knobs to it.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle import executor as EXR
from ispatialgym.oracle.actions import Action
from ispatialgym.oracle.execute import _leaf_dims, leaf_frame, leaf_width, rot_z
from ispatialgym.oracle.motions import Motions, SkillError
from ispatialgym.robot import conventions as C

log = logging.getLogger("ispatialgym.experiments.door_pull_push")


def _env(name: str, default: float) -> float:
    v = os.environ.get(f"ISG_PP_{name}")
    return float(v) if v not in (None, "") else float(default)


# ----------------------------------------------------------------------------------------------------------------
# geometry helpers (pure)
# ----------------------------------------------------------------------------------------------------------------


def fit_hinge_xy(p0: Sequence[float], p1: Sequence[float], p2: Sequence[float]) -> Tuple[float, float, float]:
    """Centre (x, y) and radius of the circle through three planar points (the handle contact at three joint values)."""
    (x0, y0), (x1, y1), (x2, y2) = p0[:2], p1[:2], p2[:2]
    a = th.tensor([[2 * (x1 - x0), 2 * (y1 - y0)], [2 * (x2 - x0), 2 * (y2 - y0)]], dtype=th.float64)
    b = th.tensor([x1 * x1 + y1 * y1 - x0 * x0 - y0 * y0, x2 * x2 + y2 * y2 - x0 * x0 - y0 * y0], dtype=th.float64)
    c = th.linalg.solve(a, b)
    r = math.hypot(x0 - float(c[0]), y0 - float(c[1]))
    return float(c[0]), float(c[1]), r


def recorded_base_bounds(sidecar, margin: float = 0.0) -> Optional[Tuple[float, float, float, float]]:
    """(x_min, x_max, y_min, y_max) of every base pose the sampler recorded (stances, base_follow, the hybrid path) grown by
    ``margin``: the floor the sampler proved free of walls. A push stance outside it is not tried (0920-022454: the outer-face
    stance at the hinge side put the wrist camera into the garden strip's side wall)."""
    pts: List[Tuple[float, float]] = []
    try:
        for st in (getattr(sidecar, "stances", None) or {}).values():
            b = getattr(st, "base_pose2d", None) or (st.get("base_pose2d") if isinstance(st, dict) else None)
            if b is not None:
                pts.append((float(b[0]), float(b[1])))
        art = sidecar.articulation
        bf = art["base_follow"] if isinstance(art, dict) else getattr(art, "base_follow", None)
        for b in (bf or {}).values():
            pts.append((float(b[0]), float(b[1])))
        path = art.get("base_follow_path") if isinstance(art, dict) else getattr(art, "base_follow_path", None)
        poses = (path or {}).get("poses") if isinstance(path, dict) else getattr(path, "poses", None)
        for pz in poses or []:
            b = pz["base_pose2d"] if isinstance(pz, dict) else getattr(pz, "base_pose2d", None)
            if b is not None:
                pts.append((float(b[0]), float(b[1])))
    except Exception as e:  # noqa: BLE001
        log.warning("recorded_base_bounds: %r", e)
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)


def within(bounds: Optional[Tuple[float, float, float, float]], x: float, y: float) -> bool:
    return bounds is None or (bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3])


def tuck_arms(motions: Motions, rep: Dict[str, Any], ignore_objects=()) -> Iterator[th.Tensor]:
    """Both arms straight down before the base drives beside a wall (0920-022239: the idle left gripper brushed the strip's
    side wall with the arms in the manipulation posture). ``ignore_objects``: the door the hand has just let go of -
    its handle is centimetres from the fingers and cuRobo refuses a start state "in collision: world" (0921-060044)."""
    for posture in ("ready",):  # "tucked" (arms straight down) self-collides with the base on this robot (owner 2026-09-20 18:43: ready only)
        try:
            yield from motions.arm_to(("left", "right"), q={"left": EX.posture_q(motions.robot, posture, "left"), "right": EX.posture_q(motions.robot, posture, "right")},
                                      ignore_objects=list(ignore_objects))
            rep["tucked"] = posture
            return
        except SkillError as e:
            rep["tucked"] = f"{posture} failed: {(e.message or e.reason)[:120]}"
            log.warning("tuck_arms(%s) failed: %s", posture, e)


# ----------------------------------------------------------------------------------------------------------------
# the experimental action
# ----------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class PullPushOpen(Action):
    name: str = "pull_push_open"
    arm: str
    object: str
    joint: str
    hinge_xy: Tuple[float, float]
    handle_xy_closed: Tuple[float, float]
    approach_closed: Tuple[float, float, float]
    handle_radius_m: float
    handle_z: float
    leaf_link: str = "leaf"
    # pull
    pull_step_m: float = 0.05
    pull_max_m: float = 1.20
    pull_deflection_rad: float = 0.25  # (legacy) arm wind-up that ended the base pull
    pull_step_deg: float = 5.0  # arm-mode pull: arc step per tracked waypoint
    pull_max_deg: float = 60.0
    pull_min_progress_rad: float = 0.005
    pull_speed_mps: float = 0.08
    pull_settle_steps: int = 15
    pull_compliant: bool = True  # re-target the *wrist* to its measured joints after every base step: the hand turns with the leaf
    pull_compliant_joints: Tuple[int, ...] = (4, 5, 6)  # arm joints 5-7 (0-based) follow the leaf; 1-4 stay rigid and do the pulling
    # (0920-012838: re-targeting all 7 left the arm slack - it stretched to 0.83 m while the door stalled at 19 deg)
    pull_limit_margin_rad: float = 0.06  # a joint this close to a limit = "the arm is at its limit"
    pull_max_extension_m: float = 0.85  # eef this far (xy) from the base centre = fully extended
    min_wall_clear_m: float = 0.62  # base centre at least this far from the door's wall plane (the "ready" left elbow touched it at 0.50, 0920-195727)
    min_push_angle_deg: float = 8.0  # below this the wedge behind the leaf is too thin for the hand: report and stop
    # push
    push_inset_m: float = 0.05  # hook point this far inside the free edge (owner 2026-09-20 23:58: 5 cm, not 10)
    push_height_drop_m: float = 0.05  # contact this far below the handle
    hook_gap_m: float = 0.035  # the hand's centre this far off the inside face (half the gripper width + 1 cm; the arc closes the gap)
    push_delta_deg: float = 20.0
    push_pre_m: float = 0.10
    push_n: int = 6
    # push stance: base at E + along·d + out·m from the free edge E (leaf direction d, outer normal m at the pulled angle)
    stance_along_m: float = 0.35
    stance_out_m: float = 0.30
    stance_candidates: int = 6
    base_bounds: Optional[Tuple[float, float, float, float]] = None  # kept for the record only; no clamp (owner 2026-09-20 18:43)
    push_arms: Tuple[str, ...] = ("left", "right")  # arms tried for the wedge entry, in order
    # the sampler's first arm segment (sequential policy: recorded handle poses + IK witnesses per <= 10 deg step) - the
    # pull tracks the true handle arc between them, the witness taking over where the differential IK gives up
    pull_poses: Optional[List[Dict[str, Any]]] = None
    pull_contacts: Optional[List[List[float]]] = None  # recorded handle contacts (the arc axis, as the executor fits it)
    qa_exempt: frozenset = frozenset({"failed_grasp"})
    TARGET_KEYS = ()

    def __post_init__(self):
        self.arms = ("left", "right")
        self.objects = (self.object,)
        super().__post_init__()

    def targets(self, sidecar, state):
        return {}

    def annotate(self, sidecar):
        obj = sidecar.language.phrase(self.object)
        fine = f"open {obj} further: pull it back with the base, then push its inside face with the closed hand"
        return fine, f"{fine} (pull ≤ {self.pull_max_m:.2f} m, push {self.push_delta_deg:.0f}° about the hinge)"

    def end_pose(self, sidecar, state):
        return None


def _arm_windup(motions: Motions, arm: str) -> float:
    q = motions.robot.get_joint_positions()[motions.robot.arm_control_idx[arm]].float()
    tgt = motions.state.arm_target[arm]
    return float(th.max(th.abs(q - tgt)))


def _limit_margin(motions: Motions, arm: str) -> Tuple[float, int]:
    """Smallest distance (rad) of the arm's joints to a limit and the index of that joint."""
    robot = motions.robot
    idx = robot.arm_control_idx[arm]
    q = robot.get_joint_positions()[idx].float()
    lo = th.as_tensor(robot.joint_lower_limits)[idx].float()
    hi = th.as_tensor(robot.joint_upper_limits)[idx].float()
    m = th.minimum(q - lo, hi - q)
    j = int(th.argmin(m))
    return float(m[j]), j


def _extension(motions: Motions, arm: str) -> float:
    bx, by, _ = motions.base_pose2d()
    p = C.as_tensor(motions.eef_pose(arm)[0])
    return math.hypot(float(p[0]) - bx, float(p[1]) - by)


def execute_pull_push(action: PullPushOpen, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    arm, name = action.arm, action.object
    rep: Dict[str, Any] = {"phases": []}
    ctx["pull_push"] = rep
    hooks = ctx.get("hooks") or {}
    if "fov" in hooks:  # the hand leaves the camera frame during the reposition: the FOV rule does not apply here
        hooks["fov"](False)

    def q_now() -> float:
        return float(state.joint_position(name, action.joint))

    obj = state.object(name)
    d0, m0, sgn = leaf_frame(action.hinge_xy, action.handle_xy_closed, action.approach_closed)
    q0 = q_now()
    _w_aabb, thick = _leaf_dims(obj, action.hinge_xy, rot_z(d0, sgn * q0), rot_z(m0, sgn * q0), action.leaf_link)
    width = leaf_width(action.handle_radius_m)
    rep["geometry"] = {"hinge_xy": list(action.hinge_xy), "d0": d0.tolist(), "m0": m0.tolist(), "swing_sign": sgn, "leaf_width_m": round(width, 3),
                       "thickness_m": round(thick, 3), "q_start_rad": round(q0, 4)}
    log.info("pull_push: hinge %s, leaf width %.2f m, thickness %.3f m, swing sign %+.0f, door at %.1f deg", [round(v, 3) for v in action.hinge_xy], width, thick, sgn, math.degrees(q0))

    # ---------------------------------------------------------------- 1. arm-mode pull to the arm's limit (owner 2026-09-20 23:58)
    # the hand tracks the handle arc about the hinge (base still), 5 deg per step, until a joint nears its limit or the
    # differential IK gives up; the door stays where the arm stops (the base never moves with the grasp held)
    robot = motions.robot
    idx = robot.arm_control_idx[arm]
    hinge3 = th.tensor([action.hinge_xy[0], action.hinge_xy[1], 0.0])
    pull: Dict[str, Any] = {"steps": [], "stop": None}
    q_start = q_now()
    step_rad = math.radians(action.pull_step_deg)
    # 1a. the recorded first arm segment (a sequential instance): its poses are the arm-only reach the sampler proved
    # from this stance (door 0921-045829: 0 -> 12.3 deg), tracked along the arc with the executor's own follow logic
    if action.pull_poses:
        axis = EX.joint_axis_from_contacts([th.as_tensor(c, dtype=th.float32) for c in action.pull_contacts]) if action.pull_contacts and len(action.pull_contacts) >= 3 else None
        eef_prev = motions.eef_pose(arm)
        seg_rec: Dict[str, Any] = {"n_poses": len(action.pull_poses), "witness_ramps": 0, "stop": None}
        for p in action.pull_poses:
            if p.get("eef") is None:
                continue
            target = EX.as_world_pose(p["eef"])
            wps = EX.arc_between(eef_prev, target, axis[0], axis[1]) if axis is not None else [target]
            try:
                yield from motions.arm_along(arm, list(wps), loaded=True, speed=motions.cfg.follow_eef_speed, converge=False)
            except SkillError as e:
                if e.details.get("stage") != "diff_ik" or p.get("arm_q") is None:
                    seg_rec["stop"] = f"segment pose q = {float(p.get('q') or 0.0):.3f}: {(e.message or e.reason)[:120]}"
                    break
                d_w = yield from EX._ramp_arm_to_q(motions, arm, p["arm_q"], loaded=True)
                seg_rec["witness_ramps"] += 1
                log.warning("pull: segment pose q = %.3f: %s -> ramped onto the recorded IK witness (%.3f rad)", float(p.get("q") or 0.0), (e.message or e.reason)[:120], d_w)
            eef_prev = target
            yield from motions.hold(3)
            q = q_now()
            margin, j_lim = _limit_margin(motions, arm)
            pull["steps"].append({"step": f"seg q {float(p.get('q') or 0.0):.3f}", "q_deg": round(math.degrees(q), 2), "limit_margin_rad": round(margin, 4), "held": state.held_object(arm) is not None})
            log.info("pull (recorded segment) -> q %.3f: door %.1f deg, limit margin %.3f (joint %d)", float(p.get("q") or 0.0), math.degrees(q), margin, j_lim)
            if state.held_object(arm) is None:
                seg_rec["stop"] = "grasp_lost"
                break
        pull["recorded_segment"] = seg_rec
        if seg_rec["stop"] is not None:
            pull["stop"] = seg_rec["stop"]
    # 1b. beyond the recorded segment (or without one): 5 deg arc steps until a joint nears its limit / the diff-IK gives up
    for i in range(1, int(math.ceil(math.radians(action.pull_max_deg) / step_rad)) + 1):
        if pull["stop"] is not None:
            break
        eef = motions.eef_pose(arm)
        eef = (C.as_tensor(eef[0]), C.as_tensor(eef[1]))
        wp = C.arc_waypoints(eef, hinge3 + th.tensor([0.0, 0.0, float(eef[0][2])]), th.tensor([0.0, 0.0, 1.0]), sgn * step_rad, n=1)
        try:
            yield from motions.arm_along(arm, list(wp), loaded=True, speed=motions.cfg.follow_eef_speed, converge=False)
        except SkillError as e:
            pull["stop"] = f"arm limit: {(e.message or e.reason)[:120]}"
            break
        yield from motions.hold(3)
        q = q_now()
        margin, j_lim = _limit_margin(motions, arm)
        held = state.held_object(arm) is not None
        pull["steps"].append({"step": i, "q_deg": round(math.degrees(q), 2), "limit_margin_rad": round(margin, 4), "held": held})
        log.info("pull %d: door %.1f deg, limit margin %.3f (joint %d), held %s", i, math.degrees(q), margin, j_lim, held)
        if not held:
            pull["stop"] = "grasp_lost"
            break
        if margin <= action.pull_limit_margin_rad:
            pull["stop"] = f"arm_at_limit: joint {j_lim} within {margin:.3f} rad of its limit"
            break
    if pull["stop"] is None:
        pull["stop"] = f"pull_max_deg {action.pull_max_deg:.0f} reached"
    # release: gradual unwind, hold until still, full opening, settle, retreat 0.10 m along -a
    total = motions.unwind(arm)
    for _ in range(11):
        yield from motions.hold(1)
        total += motions.unwind(arm)
    pull["unwind_rad"] = round(total, 4)
    yield from motions.settle(max_steps=30)
    q_pull = q_now()
    pull.update({"q_end_deg": round(math.degrees(q_pull), 2), "gain_deg": round(math.degrees(q_pull - q_start), 2)})
    rep["pull"] = pull
    rep["phases"].append("pull")
    log.info("pull done: %s; door at %.1f deg", pull["stop"], math.degrees(q_pull))

    # ---------------------------------------------------------------- 2. release, hand back, fist
    yield from motions.gripper(arm, close=False, staged=False)
    EX._release(motions, state, arm)
    if "fov" in hooks:
        hooks["fov"](False)  # the grasp -> release span is over (qa rule (b)); the wedge entry is a contact, not a grasp
    yield from motions.hold(12)
    eef = motions.eef_pose(arm)
    handle_at_release = C.as_tensor(eef[0]).clone()
    rep["release"] = {}
    try:
        yield from motions.arm_along(arm, [C.pre_pose(eef, 0.10)], soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
        rep["release"]["back_off_m"] = 0.10
    except SkillError as e:
        rep["release"]["back_off_failed"] = (e.message or e.reason)[:160]
        log.warning("release back-off failed: %s", e)
        # the differential IK gave up (0921-060044: 11.8 mm at waypoint 4/11, no joint at a limit); a planned leg, the door
        # ignored (its handle is between the open fingers), takes the hand 0.15 m back along -a
        try:
            yield from motions.arm_to(arm, pose=C.pre_pose(eef, 0.15), ignore_objects=[obj])
            rep["release"]["back_off_planned_m"] = 0.15
        except SkillError as e2:
            rep["release"]["back_off_planned_failed"] = (e2.message or e2.reason)[:160]
            log.warning("release back-off (planned) failed too: %s", e2)
    yield from motions.settle(max_steps=20)
    motions.state.arm_target[arm] = robot.get_joint_positions()[idx].clone().float()
    clear_m = float(th.linalg.norm(C.as_tensor(motions.eef_pose(arm)[0]) - handle_at_release))
    rep["release"]["hand_clear_of_handle_m"] = round(clear_m, 3)
    if clear_m >= 0.06:
        yield from motions.gripper(arm, close=True)  # the fist for the push
    else:
        # a fist closing on the handle re-engages the assisted grasp: the door then follows the base and the ghost-hand
        # monitor aborts ("drifted 5.8 cm / 4 deg off the right hand while grasped", 0921-060044) - stay open here
        log.warning("release: the hand is only %.0f mm from the handle - the gripper stays open until the wedge stance", clear_m * 1000)
    rep["release"]["q_deg"] = round(math.degrees(q_now()), 2)
    rep["phases"].append("release")


    # ---------------------------------------------------------------- 3. base to the opening side, facing the gap
    # (owner 2026-09-20 13:43: "move the robot left circle-wise so that it sees the opened opening of the door - that pose
    # gives the inside face enough room, then push outwards or move the base to the right")
    q1 = q_now()
    if math.degrees(q1) < action.min_push_angle_deg:
        rep["verdict"] = f"pull opened the door only {math.degrees(q1):.1f} deg (< {action.min_push_angle_deg:.0f}): the wedge behind the leaf is too thin for the hand - no push"
        rep["final_q_deg"] = round(math.degrees(q1), 2)
        log.warning("pull_push: %s", rep["verdict"])
        return
    d1, m1 = rot_z(d0, sgn * q1), rot_z(m0, sgn * q1)
    hinge = th.tensor([action.hinge_xy[0], action.hinge_xy[1], 0.0])
    edge = hinge + width * d1
    z_c = action.handle_z - action.push_height_drop_m
    # the hand goes into the wedge between the leaf and the frame plane, along the leaf toward the hinge (a = -d), its
    # centre hook_gap_m off the inside face; the arc about the hinge then carries hand and leaf outward (+m)
    p_hand = hinge + (width - action.push_inset_m) * d1 - (0.5 * thick + action.hook_gap_m) * m1 + th.tensor([0.0, 0.0, z_c])
    rep["push_geometry"] = {"free_edge_xy": [round(float(v), 3) for v in edge[:2]], "hand": [round(float(v), 3) for v in p_hand], "leaf_dir": [round(float(v), 3) for v in d1],
                            "outer_normal": [round(float(v), 3) for v in m1], "q_deg": round(math.degrees(q1), 2), "wedge_width_m": round((width - action.push_inset_m) * math.sin(q1), 3)}
    yield from tuck_arms(motions, rep, ignore_objects=[obj])
    # stances on the opening side (beyond the free edge along +d), out from the frame plane along +m, facing the hand point;
    # the base path bulges outward (+m) on its way: a circle-wise move around the free edge
    # stances beside the door on the free-edge side, behind the wedge's pre point along the wedge axis (+d) so the hand
    # enters along it (world heading ~109 deg at 19 deg open), then the earlier out-in-front variants
    p_pre_w = p_hand + (action.push_inset_m + 0.12) * d1
    cands: List[Tuple[float, float]] = [(0.45, 0.0), (0.50, 0.05), (0.40, 0.10), (0.55, 0.0), (0.45, 0.20), (0.15, 0.45), (0.10, 0.50)]
    moved, reached, contact_pose, closing_used = None, None, None, None
    x_here, y_here, _ = motions.base_pose2d()
    for along, out in cands:
        b = p_pre_w + along * d1 + out * m1 if out < 0.4 else edge + along * d1 + out * m1
        clear = float(th.dot(b - hinge, m0))
        if clear < action.min_wall_clear_m:
            b = b + (action.min_wall_clear_m - clear) * m0
        bx, by = float(b[0]), float(b[1])
        # no clamp to the sampler's recorded poses (owner 2026-09-20 18:43: the "1.65 m strip" was wrong - open deck on the
        # free-edge side; the collision monitor aborts on a real contact)
        aim = p_hand - th.tensor([bx, by, float(p_hand[2])])
        yaw = math.atan2(float(aim[1]), float(aim[0]))
        target = (bx, by, yaw)  # facing the hand point; each arm gets its own shoulder-offset yaw below
        mid = (0.5 * (x_here + bx) + 0.15 * float(m1[0]), 0.5 * (y_here + by) + 0.15 * float(m1[1]), 0.5 * (yaw + math.atan2(math.sin(yaw), math.cos(yaw))))
        log.info("wedge push: stance candidate %s via %s (along %.2f, out %.2f)", [round(v, 2) for v in target], [round(v, 2) for v in mid], along, out)
        try:
            yield from motions.track_path([mid, target])
        except SkillError as e:
            rep.setdefault("reposition_failures", []).append({"stance": [round(v, 3) for v in target], "error": e.message or e.reason})
            continue
        moved = target
        yield from motions.settle(max_steps=20)
        # the wedge axis runs almost along the door's wall, which is on the robot's LEFT from the opening side: the left
        # arm enters it naturally, the right arm would cross the body (0920-184642: 42 mm / 0.22 rad off with the right)
        for push_arm in action.push_arms:
            yield from motions.gripper(push_arm, close=True)
            for pre_extra in (0.12, 0.18, 0.06):  # how far beyond the free edge the entry starts
                for closing in (th.tensor([0.0, 0.0, 1.0]), m1.clone()):
                    cp = C.contact_eef(p_hand, -d1, closing=closing)
                    pre = (C.as_tensor(cp[0]) + (action.push_inset_m + pre_extra) * d1, C.as_tensor(cp[1]))
                    try:
                        yield from motions.arm_to(push_arm, pose=pre, ignore_objects=[obj])
                        reached, contact_pose, closing_used = target, cp, f"{push_arm} arm, pre +{pre_extra:.2f}, " + ("pads vertical" if float(closing[2]) > 0.5 else "pads along the normal")
                        arm = push_arm  # the rest of the action (entry, arc, drag, back-off) uses this arm
                        break
                    except SkillError as e:
                        rep.setdefault("prep_failures", []).append({"stance": [round(v, 3) for v in target], "arm": push_arm, "pre_extra": pre_extra, "closing": closing.tolist(), "error": (e.message or e.reason)[:160]})
                if reached is not None:
                    break
            if reached is not None:
                break
        if reached is not None:
            break
    rep["reposition"] = {"stance": None if moved is None else [round(v, 3) for v in moved], "prep_reached": reached is not None, "closing": closing_used}
    rep["phases"].append("reposition")
    if reached is None or contact_pose is None:
        rep["verdict"] = "wedge entry pose unreachable from every opening-side stance"
        raise SkillError("plan_failed", "pull_push: the wedge entry pose behind the leaf is unreachable from every candidate stance", stage="push_prep")

    # ---------------------------------------------------------------- 4. slide into the wedge (gated by the wrist camera), then swing the leaf outward
    q_before = q_now()
    push: Dict[str, Any] = {"q_before_deg": round(math.degrees(q_before), 2), "delta_deg": action.push_delta_deg}
    cam_link = f"{arm}_realsense_link"
    edge_plane = float(th.dot(edge, d1))  # the free-edge plane: x . d = edge . d (deeper into the wedge = smaller x . d)

    def cam_depth() -> Optional[float]:
        """How far the wrist camera is past the free-edge plane toward the hinge (m; < 0 = still outside)."""
        link = (getattr(robot, "links", {}) or {}).get(cam_link)
        if link is None:
            return None
        pos = C.as_tensor(link.get_position_orientation()[0])
        return edge_plane - float(th.dot(pos, d1))

    # step in along -d in 2 cm increments; stop when the wrist camera would pass the free-edge plane or the hook pose is reached
    entry_steps: List[Dict[str, Any]] = []
    hooked_ok = False
    try:
        while True:
            eef = motions.eef_pose(arm)
            p_eef = C.as_tensor(eef[0])
            remaining = float(th.dot(p_eef - C.as_tensor(contact_pose[0]), d1))  # > 0: still short of the hook pose along -d
            if remaining <= 0.005:
                hooked_ok = True
                break
            step = min(0.02, remaining)
            yield from motions.arm_along(arm, [(p_eef - step * d1, C.as_tensor(eef[1]))], soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
            depth_hand = edge_plane - float(th.dot(C.as_tensor(motions.eef_pose(arm)[0]), d1))
            dc = cam_depth()
            entry_steps.append({"hand_depth_m": round(depth_hand, 3), "cam_depth_m": None if dc is None else round(dc, 3)})
            if dc is not None and dc > -0.01:
                push["entry_stop"] = f"wrist camera at the free-edge plane (hand {depth_hand:.3f} m in)"
                hooked_ok = depth_hand > 0.02
                break
        push["hook"] = "in the wedge" if hooked_ok else "not inserted"
    except SkillError as e:
        push["hook"] = f"failed: {e.message or e.reason}"
        log.warning("wedge entry failed: %s", e)
    push["insertion_depth_m"] = entry_steps[-1]["hand_depth_m"] if entry_steps else 0.0
    push["entry_steps"] = entry_steps
    if push["hook"] == "in the wedge":
        hooked = motions.eef_pose(arm)
        cpose = (C.as_tensor(hooked[0]), C.as_tensor(hooked[1]))
        delta = sgn * math.radians(action.push_delta_deg)
        arc = C.arc_waypoints(cpose, hinge + th.tensor([0.0, 0.0, float(cpose[0][2])]), th.tensor([0.0, 0.0, 1.0]), delta, n=action.push_n)
        try:
            yield from motions.arm_along(arm, list(arc), loaded=True, partial=True)
            push["arc"] = "complete"
        except SkillError as e:
            push["arc"] = f"stopped: {e.message or e.reason}"
        yield from motions.settle(max_steps=40)
        push["q_after_arm_deg"] = round(math.degrees(q_now()), 2)
        if q_now() - q_before < math.radians(3.0):
            # the owner's alternative: with the hand hooked behind the leaf, drive the base outward (+m) - the hand drags the leaf
            motions.state.arm_target[arm] = robot.get_joint_positions()[robot.arm_control_idx[arm]].clone().float()
            bx, by, byaw = motions.base_pose2d()
            tgt = (bx + 0.25 * float(m1[0]), by + 0.25 * float(m1[1]), byaw)
            try:
                yield from motions.track_path([tgt], tracker_cfg=replace(motions.cfg.base_tracker, v_max=(0.06, 0.06, 0.3)))
                push["base_drag"] = "0.25 m outward"
            except SkillError as e:
                push["base_drag"] = f"stopped: {e.message or e.reason}"
            yield from motions.settle(max_steps=40)
    else:
        push["arc"] = "not attempted"
    q_after = q_now()
    push["q_after_deg"] = round(math.degrees(q_after), 2)
    push["gain_deg"] = round(math.degrees(q_after - q_before), 2)
    try:  # hand out of the wedge (back along +m), then let the leaf settle
        eef = motions.eef_pose(arm)
        yield from motions.arm_along(arm, [(C.as_tensor(eef[0]) + 0.10 * m1, C.as_tensor(eef[1]))], soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
    except SkillError as e:
        push["back_off_failed"] = e.message or e.reason
    yield from motions.settle(max_steps=60)
    push["q_settled_deg"] = round(math.degrees(q_now()), 2)
    rep["push"] = push
    rep["phases"].append("push")
    rep["final_q_deg"] = push["q_settled_deg"]
    rep["verdict"] = ("push opened the door further" if q_after - q_before > math.radians(3) else "push did not move the door")
    mon = ctx.get("monitors")
    if mon is not None:
        rep["contacts"] = [dict(e) for e in getattr(mon, "collision_events", [])[-40:]]
        log.info("pull_push contacts (collision monitor): %s", rep["contacts"])
    log.info("pull_push done: door %.1f -> %.1f -> %.1f deg (pull, push, settled); %s", math.degrees(q_pull), math.degrees(q_after), math.degrees(q_now()), rep["verdict"])



def post_pull_push(action: PullPushOpen, state, targets, ctx) -> Optional[str]:
    return None  # the experiment's outcome is in ctx["pull_push"]; the episode's goal check judges the door angle


EX.EXECUTORS[PullPushOpen] = (execute_pull_push, post_pull_push)
if "pull_push" not in EXR.NOTE_KEYS:
    EXR.NOTE_KEYS = tuple(EXR.NOTE_KEYS) + ("pull_push",)


# ----------------------------------------------------------------------------------------------------------------
# the program hooks
# ----------------------------------------------------------------------------------------------------------------


def _door_geometry(sidecar):
    art = sidecar.articulation

    def field(o, key):
        return o[key] if isinstance(o, dict) else getattr(o, key)

    wps = field(art, "waypoints")
    keys = sorted(wps.keys(), key=float)
    pts = [list(field(wps[k], "contact")) for k in (keys[0], keys[len(keys) // 2], keys[-1])]
    hx, hy, r = fit_hinge_xy(*pts)
    # the closed-leaf frame: the waypoint whose joint value is the lower limit (open: k = 0; close: k = 1)
    q_init, q_goal = float(field(art, "q_init")), float(field(art, "q_goal"))
    k_closed = keys[0] if q_init <= q_goal else keys[-1]
    c0, a0 = list(field(wps[k_closed], "contact")), list(field(wps[k_closed], "approach"))
    return {"hinge_xy": (hx, hy), "handle_radius_m": r, "handle_xy_closed": (float(c0[0]), float(c0[1])), "approach_closed": tuple(float(v) for v in a0),
            "handle_z": float(c0[2]), "object": field(art, "object"), "joint": field(art, "joint"), "q_init": q_init, "q_goal": q_goal}


def hook(actions: List[Action], sidecar, loader) -> List[Action]:
    """Keep ``navigate, transition, grasp`` of the recorded program, replace the rest by :class:`PullPushOpen`."""
    keep = []
    for a in actions:
        if a.name in ("navigate", "transition", "grasp"):
            keep.append(a)
        else:
            break
    if not keep or keep[-1].name != "grasp":
        raise RuntimeError(f"door_pull_push: the program must start with navigate/transition/grasp, got {[a.name for a in actions]}")
    g = _door_geometry(sidecar)
    arm = keep[-1].arms[0]
    # a sequential instance: the first recorded arm segment is the pull (owner 2026-09-20 23:58 item 2: "the arm-mode
    # pull of item 1 ... to the arm's reach limit from the manipulation stance")
    art = sidecar.articulation
    extra = (art.get("extra") if isinstance(art, dict) else getattr(art, "extra", None)) or {}
    path = extra.get("base_follow_path") or {}
    pull_poses = None
    if path.get("policy") == "sequential" and path.get("segments") and path["segments"][0].get("type") == "arm":
        pull_poses = [dict(p) for p in path["segments"][0].get("poses") or []]
    wps = art["waypoints"] if isinstance(art, dict) else art.waypoints
    contacts = [list((w["contact"] if isinstance(w, dict) else w.contact)) for _, w in sorted(wps.items(), key=lambda kv: float(kv[0]))]
    act = PullPushOpen(arm=arm, object=g["object"], joint=g["joint"], hinge_xy=g["hinge_xy"], handle_xy_closed=g["handle_xy_closed"],
                       pull_poses=pull_poses, pull_contacts=contacts if len(contacts) >= 3 else None,
                       approach_closed=g["approach_closed"], handle_radius_m=g["handle_radius_m"], handle_z=g["handle_z"],
                       pull_step_m=_env("PULL_STEP_M", 0.05), pull_max_m=_env("PULL_MAX_M", 1.2), pull_deflection_rad=_env("PULL_DEFLECTION_RAD", 0.25),
                       pull_speed_mps=_env("PULL_SPEED_MPS", 0.08), min_wall_clear_m=_env("MIN_WALL_CLEAR_M", 0.62),
                       pull_compliant=_env("PULL_COMPLIANT", 1.0) > 0, pull_limit_margin_rad=_env("PULL_LIMIT_MARGIN_RAD", 0.06), pull_max_extension_m=_env("PULL_MAX_EXTENSION_M", 0.85),
                       push_inset_m=_env("PUSH_INSET_M", 0.05), push_height_drop_m=_env("PUSH_HEIGHT_DROP_M", 0.05), push_delta_deg=_env("PUSH_DELTA_DEG", 20.0), hook_gap_m=_env("HOOK_GAP_M", 0.035),
                       stance_along_m=_env("STANCE_ALONG_M", 0.35), stance_out_m=_env("STANCE_OUT_M", 0.30), base_bounds=recorded_base_bounds(sidecar))
    log.info("door_pull_push hook: hinge %s, handle radius %.3f m, arm %s, %s/%s", [round(v, 3) for v in g["hinge_xy"]], g["handle_radius_m"], arm, g["object"], g["joint"])
    return keep + [act]


def hook_close(actions: List[Action], sidecar, loader) -> List[Action]:
    """A room door's *close* instance: the registry's own program (``programs.program_articulation_large``: navigate,
    transition, :class:`~ispatialgym.oracle.actions.PushClose`) with the ``ISG_PP_*`` knobs applied to the push."""
    from ispatialgym.oracle.actions import PushClose
    from ispatialgym.oracle.programs import expand

    program = expand(sidecar.alias, sidecar)
    push = program[-1] if program else None
    if not isinstance(push, PushClose):
        raise RuntimeError(f"door_pull_push.hook_close: the registry's program is {[a.name for a in program]}, not a push-close")
    push = replace(push, push_inset_m=_env("PUSH_INSET_M", push.push_inset_m), push_height_drop_m=_env("PUSH_HEIGHT_DROP_M", push.push_height_drop_m),
                   chunk_deg=_env("CHUNK_DEG", push.chunk_deg), closed_tol_rad=_env("CLOSED_TOL_RAD", push.closed_tol_rad))
    log.info("door_pull_push hook_close: hinge %s, door at %.1f deg -> closed, arm %s", [round(v, 3) for v in push.hinge_xy], math.degrees(push.q_from), push.arm)
    return program[:-1] + [push]
