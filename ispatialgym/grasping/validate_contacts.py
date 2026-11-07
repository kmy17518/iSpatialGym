"""Physics validation of the PUSH and BRACE contact libraries (i7-contact 2026-11-07, step 2; :mod:`contact_library`), the way
:mod:`validate_tabletop` validates grasps - and by running THE ORACLE'S OWN CODE on each entry, so validation is what the oracle executes.

World (:class:`KitContactWorld`, one persistent Isaac Sim world per shard and arm): ``validate_tabletop.TabletopWorld`` - the JoyLo R1Pro with
the evaluator's 250 kg base, the AG rules, the sleep threshold / assist, the fixed slab under the object (``slab_top_z``), cuRobo - with the
base yaw fixed to the rule frame (the robot facing +x: the entry's push direction / brace approach keep their robot-relative meaning) and
the object placed at its stable rest turned to the entry's ``yaw_deg``, dropped and settled (``TabletopWorld.settle``: the measured rest
saved BEFORE any contact). Per entry:

1. **placement**: ``TabletopWorld.plan`` puts the eef at the entry's EXECUTED pre-contact (``pre.executed``: the oracle's lowered pre of a
   straight push) - batched IK over base offsets x trunks, whole-robot contact policy vs the slab, arm clear of the object, the most
   comfortable solution; the fingers at the entry's opening (closed / ``fingers_opening_m``);
2. **trial** (state restored from the rest snapshot): the entry becomes the oracle's action - a ``Push`` (``PushContact``: the RECORDED
   ``pre`` - ``execute_push`` lowers it itself -, ``init`` = the contact, ``mid`` for an arc, ``goal`` = contact + stroke x direction,
   ``extra`` = the record's extras with its ``pose_class``) or a ``Brace`` (``BraceContact``: pre, contact, approach) - and is run through
   ``execute.run_action`` with the oracle's ``Motions`` and ``executor.SimState`` (the gripper closure / opening, the pre-contact, the push
   legs and their speed, the palm lift, the strokes, the brace's touch stop and back-off are all the oracle's); then
   ``execute.check_postcondition``. A brace is then LOADED: a kinematic pusher (a box ``brace_pusher_size_m`` against the partner's face
   opposite the brace, ``brace_pusher_gap_m`` off it) drives the partner along the entry's load direction (``load.direction``: the touching
   push) at ``brace_pusher_speed_m_s`` until ``brace_press_m`` past first contact, and holds ``brace_hold_steps``;
3. **acceptance** (:func:`judge_push` / :func:`judge_brace`; force series from ``validate_tabletop.ForceProbe``):

   * a PUSH passes when the object moved along the push direction by >= ``stroke - stop clearance - push_short_tol_m`` (the stroke minus the
     fingertips' start gap), it did not topple (tilt change <= ``topple_tilt_rad``), the hand did not slip off (sideways drift <=
     ``push_lateral_max_m``, yaw change <= ``push_yaw_max_rad``, the fingertips within ``contact_lost_m`` of the object's face at the goal),
     no non-finger robot link loaded the slab beyond ``non_finger_contact_N``, the oracle's postcondition is clean and the action ran to its
     end (``max_steps``);
   * a BRACE passes when the oracle's brace reached its contact (postcondition), and under the pusher's load the partner moved along the
     load by <= ``brace_hold_max_m`` (the 2 mm touch stop + 4 mm back-off + slack), did not topple, the brace hand stayed within
     ``brace_eef_drift_m`` of its contact pose, and no non-finger link loaded the slab.

Trials: ``trials_per_entry`` (1: the physics is deterministic at one configuration; the yaw slots of a class are separate entries), at most
``max_entries`` entries per model and kind in library order. Timing is recorded per model, rest and kind (``validation_timing``: setup, settle,
plan, trials, total). The thresholds are PROVISIONAL (the recorded measures are for calibrating them).

    python -m ispatialgym.grasping.validate_contacts --root ROOT --kind push|brace --glib GLIB [--jobs J.json] [--arm right|left] [--shard i/n] [--only ID ...]
"""

from __future__ import annotations

import argparse
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.grasping import schema as S

TRIAL_SCHEMA = "isg-contact-trial/1"


@dataclass
class ContactValidateConfig:
    work_arm: str = "right"
    brace_arm: str = "left"
    trials_per_entry: int = 1
    max_entries: int = 48
    #: the object in the base frame (ahead of the slab edge, to the arm's side): the right arm's; the left arm's lateral is mirrored
    base_offsets: Tuple[Tuple[float, float], ...] = ((0.0, -0.15), (0.0, -0.28), (0.08, -0.05), (0.08, -0.22))
    max_steps: int = 1800
    # push acceptance
    push_short_tol_m: float = 0.01
    push_lateral_max_m: float = 0.02
    push_yaw_max_rad: float = 0.2
    contact_lost_m: float = 0.02
    topple_tilt_rad: float = math.radians(10.0)
    non_finger_contact_N: float = 0.5
    # brace load / acceptance
    brace_pusher_size_m: Tuple[float, float, float] = (0.03, 0.08, 0.05)
    brace_pusher_gap_m: float = 0.01
    brace_press_m: float = 0.01
    brace_pusher_speed_m_s: float = 0.02
    brace_hold_steps: int = 30
    brace_hold_max_m: float = 0.012
    brace_eef_drift_m: float = 0.01


# ------------------------------------------------------------------------------------------------ the entry as the oracle's action


def _sc_pose(pose):
    from ispatialgym.oracle.sidecar import Pose

    return Pose([float(v) for v in th.as_tensor(pose[0]).float().tolist()], [float(v) for v in th.as_tensor(pose[1]).float().tolist()])


def _T(pose) -> np.ndarray:
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    return CPZ._mat(pose)


def entry_action(entry: Dict[str, Any], T_world_obj, obj_name: str, arm: str, *, base_yaw: float = 0.0, touches: Sequence[str] = ()):
    """The oracle's action of a schema entry with the object at ``T_world_obj``: a ``Push`` (``PushContact``: the RECORDED pre - the
    oracle's ``push_lowered_pre`` lowers a straight push's itself -, init = the contact, mid for an arc, goal = the stroke along the direction,
    the record's extras + ``pose_class``) or a ``Brace`` (``BraceContact``: the recorded pre, the contact, the approach)."""
    from ispatialgym.oracle.actions import Brace, Push
    from ispatialgym.oracle.sidecar import BraceContact, PushContact
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    w = CPZ.entry_world(entry, T_world_obj)
    extras = dict(entry.get("extras") or {})
    if entry["kind"] == "push":
        extras.setdefault("pose_class", extras.get("pose_class") or entry["class"])
        mid = goal = None
        arc = entry.get("arc")
        if arc is not None:
            To = _T(T_world_obj)
            from ispatialgym.grasping import schema as SS

            def wo(d):
                T = To @ SS.pose_of(d)
                return th.tensor(T[:3, 3], dtype=th.float32), th.tensor(SS.mat_to_quat(T[:3, :3]), dtype=th.float32)

            mid, goal = wo(arc["mid_object"]), wo(arc["goal_object"])
        else:
            goal = w["goal"]
        if goal is None:
            raise ValueError(f"push entry {entry['class']}: no stroke / goal")
        c = PushContact(arm=arm, object=obj_name, pre=_sc_pose(w["pre"]), init=_sc_pose(w["contact"]), mid=None if mid is None else _sc_pose(mid),
                        goal=_sc_pose(goal), direction=[float(v) for v in w["direction"].tolist()], distance_m=float(entry["push"]["stroke_m"]), extra=extras)
        return Push(contact=c, base_yaw=base_yaw, touches=tuple(touches))
    a = th.tensor(_T(w["contact"])[:3, 2], dtype=th.float32)
    c = BraceContact(arm=arm, object=obj_name, kind=str(extras.get("pose_class") or entry["class"]), pre=_sc_pose(w["pre"]), contact=_sc_pose(w["contact"]),
                     direction=[float(v) for v in a.tolist()], extra=extras)
    return Brace(contact=c, base_yaw=base_yaw)


def run_oracle(action, motions, state, step, max_steps: int) -> Dict[str, Any]:
    """``execute.run_action`` of ``action`` stepping the world with ``step(a)`` (at most ``max_steps``), then ``execute.check_postcondition``:
    ``{"steps", "completed", "postcondition", "ctx"}``."""
    from ispatialgym.oracle import execute as EX

    targets = action.targets(None, state)
    ctx: Dict[str, Any] = {}
    n, completed = 0, True
    gen = EX.run_action(action, motions, state, targets, ctx)
    for a in gen:
        step(a)
        n += 1
        if n >= max_steps:
            gen.close()
            completed = False
            break
    post = EX.check_postcondition(action, state, targets, ctx) if completed else "incomplete"
    return {"steps": n, "completed": completed, "postcondition": post, "ctx": {k: v for k, v in ctx.items() if isinstance(v, (int, float, str, bool, dict, list))}}


# ------------------------------------------------------------------------------------------------ acceptance (pure)


def _yaw(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def _tilt(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle between the object's up axis (its frame's world vertical at ``Ra``) before and after."""
    za, zb = Ra.T @ np.array([0.0, 0.0, 1.0]), Rb.T @ np.array([0.0, 0.0, 1.0])
    return float(math.acos(max(-1.0, min(1.0, float(za @ zb)))))


def tip_gap(entry: Dict[str, Any], T_obj0: np.ndarray, T_obj: np.ndarray, T_eef: np.ndarray, pts_obj: np.ndarray) -> float:
    """How far the object's rear face (its extreme against the push direction) lies ahead of the fingertips along the push (m; <= 0: the tips
    on / past it) - the hand slipping off the face shows as a growing gap while the object no longer follows."""
    from ispatialgym.robot import conventions as C

    d = T_obj0[:3, :3] @ np.asarray(entry["push"]["direction"], dtype=float)
    d[2] = 0.0
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    tip = T_eef[:3, 3] + C.FINGERTIP_LENGTH * T_eef[:3, 2]
    pts = pts_obj @ T_obj[:3, :3].T + T_obj[:3, 3]
    return float((pts @ d).min() - tip @ d)


def judge_push(entry: Dict[str, Any], T_obj0: np.ndarray, T_obj1: np.ndarray, T_eef1: np.ndarray, pts_obj: np.ndarray, cfg: ContactValidateConfig,
               run: Dict[str, Any], other_slab_N: float = 0.0, tip_gap_m: float = None) -> Dict[str, Any]:
    """The push acceptance (module docstring): ``{"verdict", "reasons", "measures"}``. ``tip_gap_m``: the fingertips' gap to the face at the
    push's farthest object position (the trial's per-step :func:`tip_gap`; the oracle backs the hand off after the push), else at the end."""
    d = T_obj0[:3, :3] @ np.asarray(entry["push"]["direction"], dtype=float)
    d[2] = 0.0
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    dp = T_obj1[:3, 3] - T_obj0[:3, 3]
    along = float(dp @ d)
    lateral = float(np.linalg.norm(dp[:2] - along * d[:2]))
    dyaw = abs(math.atan2(math.sin(_yaw(T_obj1[:3, :3]) - _yaw(T_obj0[:3, :3])), math.cos(_yaw(T_obj1[:3, :3]) - _yaw(T_obj0[:3, :3]))))
    tilt = _tilt(T_obj0[:3, :3], T_obj1[:3, :3])
    clearance = float((entry.get("extras") or {}).get("stop_clearance_m") or 0.005)
    stroke = float(entry["push"]["stroke_m"])
    need = stroke - clearance - cfg.push_short_tol_m
    gap = tip_gap(entry, T_obj0, T_obj1, T_eef1, pts_obj) if tip_gap_m is None else float(tip_gap_m)
    reasons = []
    if not run.get("completed", True):
        reasons.append("incomplete")
    if run.get("postcondition"):
        reasons.append(f"oracle:{run['postcondition']}")
    if along < need:
        reasons.append("short")
    if tilt > cfg.topple_tilt_rad:
        reasons.append("topple")
    if lateral > cfg.push_lateral_max_m or dyaw > cfg.push_yaw_max_rad:
        reasons.append("slip")
    if gap > cfg.contact_lost_m:
        reasons.append("contact_lost")
    if other_slab_N > cfg.non_finger_contact_N:
        reasons.append("non_finger_contact")
    return {"verdict": "invalid" if reasons else "valid", "reasons": reasons,
            "measures": {"moved_along_m": round(along, 5), "needed_m": round(need, 5), "lateral_m": round(lateral, 5), "yaw_change_rad": round(dyaw, 5),
                         "tilt_change_rad": round(tilt, 5), "tip_gap_m": round(gap, 5), "other_slab_N": round(float(other_slab_N), 4)}}


class PeakTracker:
    """Per control step of a push trial: the object's displacement along the push and the fingertips' :func:`tip_gap` at the step it was
    farthest (``gap``: None before any step)."""

    def __init__(self, entry: Dict[str, Any], T_obj0: np.ndarray, pts_obj: np.ndarray):
        self.entry, self.T0, self.pts = entry, T_obj0, pts_obj
        d = T_obj0[:3, :3] @ np.asarray(entry["push"]["direction"], dtype=float)
        d[2] = 0.0
        self.d = d / max(float(np.linalg.norm(d)), 1e-12)
        self.best, self.gap = -math.inf, None

    def __call__(self, T_obj: np.ndarray, T_eef: np.ndarray) -> None:
        along = float((T_obj[:3, 3] - self.T0[:3, 3]) @ self.d)
        if self.gap is None or along > self.best + 1e-5:  # the first step at the farthest position (the back-off after it does not count)
            self.best, self.gap = along, tip_gap(self.entry, self.T0, T_obj, T_eef, self.pts)


def judge_brace(entry: Dict[str, Any], T_obj0: np.ndarray, T_obj1: np.ndarray, T_eef0: np.ndarray, T_eef1: np.ndarray, load_dir_world: np.ndarray,
                cfg: ContactValidateConfig, run: Dict[str, Any], other_slab_N: float = 0.0) -> Dict[str, Any]:
    """The brace acceptance (module docstring): ``T_obj0`` / ``T_eef0`` when the brace stands (before the load), ``*1`` after the load."""
    d = np.asarray(load_dir_world, dtype=float).copy()
    d[2] = 0.0
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    moved = float((T_obj1[:3, 3] - T_obj0[:3, 3]) @ d)
    tilt = _tilt(T_obj0[:3, :3], T_obj1[:3, :3])
    drift = float(np.linalg.norm(T_eef1[:3, 3] - T_eef0[:3, 3]))
    reasons = []
    if not run.get("completed", True):
        reasons.append("incomplete")
    if run.get("postcondition"):
        reasons.append(f"oracle:{run['postcondition']}")
    if moved > cfg.brace_hold_max_m:
        reasons.append("not_held")
    if tilt > cfg.topple_tilt_rad:
        reasons.append("topple")
    if drift > cfg.brace_eef_drift_m:
        reasons.append("hand_pushed")
    if other_slab_N > cfg.non_finger_contact_N:
        reasons.append("non_finger_contact")
    return {"verdict": "invalid" if reasons else "valid", "reasons": reasons,
            "measures": {"moved_along_load_m": round(moved, 5), "tilt_change_rad": round(tilt, 5), "eef_drift_m": round(drift, 5), "other_slab_N": round(float(other_slab_N), 4)}}


def pusher_path(entry: Dict[str, Any], T_obj: np.ndarray, pts_obj: np.ndarray, cfg: ContactValidateConfig, dt: float) -> List[np.ndarray]:
    """The brace load's pusher centres (world, one per control step): from ``brace_pusher_gap_m`` off the partner's face opposite the load
    (its extreme against ``load.direction``) at the partner's mid height to ``brace_press_m`` past it, at ``brace_pusher_speed_m_s``."""
    d = np.asarray((entry.get("load") or {}).get("direction") or [0.0, 1.0, 0.0], dtype=float)
    d[2] = 0.0
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    pts = pts_obj @ T_obj[:3, :3].T + T_obj[:3, 3]
    face = float((pts[:, :2] @ d[:2]).min())
    c = pts.mean(axis=0)
    z = 0.5 * float(pts[:, 2].min() + pts[:, 2].max())
    half = 0.5 * float(cfg.brace_pusher_size_m[0])
    start = np.array([c[0], c[1], z]) + d * (face - float(c[:2] @ d[:2]) - cfg.brace_pusher_gap_m - half)
    travel = cfg.brace_pusher_gap_m + cfg.brace_press_m
    n = max(1, int(math.ceil(travel / max(cfg.brace_pusher_speed_m_s * dt, 1e-6))))
    return [start + d * travel * k / n for k in range(n + 1)]


# ------------------------------------------------------------------------------------------------ one model (any world with the KitContactWorld surface)


def validate_library(world, lib_k: Dict[str, Any], obj, geom, pts_obj: np.ndarray, cfg: ContactValidateConfig, arm: str) -> Dict[str, Any]:
    """Validate the entries of one library (``lib_k``: one kind) of a loaded object in ``world`` (the :class:`KitContactWorld` surface:
    ``settle(obj, geom, quat)``, ``plan_entry(obj, geom, entry, obj_pose, arm)``, ``trial_push / trial_brace(obj, entry, plan, state, rest,
    arm)``); writes each entry's ``validation`` and returns the timing (per model / rest / kind)."""
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    t_all = time.time()
    tim = {"settle_s": 0.0, "plan_s": 0.0, "trials_s": 0.0, "n_trials": 0, "n_entries": 0}
    rest_q = np.asarray(lib_k["object"]["rest"]["quat_xyzw"], dtype=float)
    settled: Dict[float, Tuple[Dict[str, Any], Any]] = {}
    for e in lib_k["entries"][: cfg.max_entries]:
        yaw = round(float(e.get("yaw_deg", 0.0)), 4)
        t0 = time.time()
        if yaw not in settled:
            q0 = S.quat_mul(S.quat_yaw(math.radians(yaw)), rest_q)
            settled[yaw] = world.settle(obj, geom, q0)
        rest, state = settled[yaw]
        tim["settle_s"] += time.time() - t0
        T_obj = S.pose_mat(rest["world_position"], rest["world_quat_xyzw"]) if "world_position" in rest else None
        t0 = time.time()
        plan = world.plan_entry(obj, geom, e, T_obj, arm)
        tim["plan_s"] += time.time() - t0
        tim["n_entries"] += 1
        if plan is None or plan.get("q") is None:
            e["validation"] = {"schema": TRIAL_SCHEMA, "verdict": "invalid", "reasons": ["no_placement"], "plan_fail": (plan or {}).get("fail"), "trials": []}
            continue
        trials = []
        t0 = time.time()
        for _k in range(cfg.trials_per_entry):
            trials.append(world.trial_push(obj, e, plan, state, rest, arm) if lib_k["kind"] == "push" else world.trial_brace(obj, e, plan, state, rest, arm))
            tim["n_trials"] += 1
        tim["trials_s"] += time.time() - t0
        ok = all(t["verdict"] == "valid" for t in trials)
        e["validation"] = {"schema": TRIAL_SCHEMA, "verdict": "valid" if ok else "invalid", "reasons": sorted({r for t in trials for r in t["reasons"]}),
                           "trials": trials, "config": asdict(cfg), "entry_digest": e.get("digest"), "calibrated": False}
    tim = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in tim.items()}
    tim["total_s"] = round(time.time() - t_all, 2)
    tim["rest_id"] = lib_k["object"].get("rest_id")
    tim["kind"] = lib_k["kind"]
    lib_k["validation_timing"] = tim
    lib_k["validated"] = {"schema": TRIAL_SCHEMA, "config": asdict(cfg), "arm": arm, "oracle_config": {k: getattr(CPZ.oracle_config(), k) for k in
                                                                                                       ("push_pre_dist_m", "brace_touch_stop_m", "brace_touch_backoff_m")},
                          "created": S.now()}
    return tim


# ------------------------------------------------------------------------------------------------ the Isaac Sim world (GPU node; not unit-tested)


class KitContactWorld:
    """``validate_tabletop.TabletopWorld`` for contacts: the base yaw fixed to the rule frame (``base_yaws`` 1: the robot facing +x), the base
    offsets on the arm's side, the oracle's ``Motions`` with the cuRobo backend as its planner, ``executor.SimState`` over a task stub (the
    object and the slab in scope), ``ForceProbe`` for the slab loads, a kinematic pusher box for the brace load."""

    def __init__(self, cfg: ContactValidateConfig, arm: str):
        from ispatialgym.grasping import validate_tabletop as VT

        offs = cfg.base_offsets if arm == "right" else tuple((a, -b) for a, b in cfg.base_offsets)
        self.vcfg = VT.ValidateConfig(arm=arm, base_yaws=1, base_offsets=offs)
        self.tw = VT.TabletopWorld(self.vcfg)
        self.cfg, self.arm, self.VT = cfg, arm, VT
        self._pusher = None

    def load(self, job: Dict[str, Any], model_dir: str):
        return self.tw.load(job, model_dir)

    def unload(self, obj):
        self.tw.unload(obj)

    def settle(self, obj, geom, quat0):
        rest, state = self.tw.settle(obj, geom, np.asarray(quat0, dtype=float))
        p, q = (x.clone() for x in obj.get_position_orientation())
        rest = dict(rest, world_position=[float(v) for v in p.tolist()], world_quat_xyzw=[float(v) for v in q.tolist()])  # the restore keys stay
        return rest, state

    def plan_entry(self, obj, geom, entry, T_obj, arm):
        from ispatialgym.sampling.v2 import contact_poses as CPZ

        cand = {"name": entry["digest"], "T_object_eef": entry["pre"]["executed"], "preshape_q": float(entry["fingers"]["opening_m"])}
        p, q = obj.get_position_orientation()
        best = self.tw.plan(obj, geom, [cand], (p.numpy(), q.numpy()))
        b = best.get(entry["digest"])
        if b is not None and b.get("q") is not None:
            b = dict(b, T_world_obj=CPZ._mat((p, q)))
        return b

    def _motions_state(self, obj):
        from ispatialgym.oracle.executor import SimState
        from ispatialgym.oracle.motions import Motions, MotionsConfig

        motions = Motions(self.tw.env, self.tw.backend, cfg=MotionsConfig(), robot=self.tw.robot)
        motions.state.hold_current()
        task = SimpleNamespace(object_scope={obj.name: obj, self.tw.slab.name: self.tw.slab}, _tro=None)
        return motions, SimState(self.tw.env, task, self.tw.robot)

    def _start(self, obj, entry, plan, state, rest):
        th_ = self.tw.th
        rec = self.tw.restore(obj, state, rest)
        q = plan["q"].clone()
        q[self.tw.robot.gripper_control_idx[self.arm]] = float(entry["fingers"]["opening_m"])
        self.tw.robot.set_joint_positions(q)
        self.tw.robot.keep_still()
        obj.wake()
        probe = self.VT.ForceProbe(self.tw.robot, self.tw.slab, obj, self.arm)
        probe.install()
        return rec, probe, th_

    def _step_fn(self, probe, series):
        def step(a):
            self.tw.step(a)
            f = probe.take()
            series.append(float(f.get("other_slab", 0.0) or 0.0))
        return step

    def trial_push(self, obj, entry, plan, state, rest, arm):
        t0 = time.time()
        rec, probe, _ = self._start(obj, entry, plan, state, rest)
        series: List[float] = []
        try:
            T0 = _T(obj.get_position_orientation())
            from ispatialgym.grasping.contact_library import hull_points

            pts = hull_points(self._geom_of(obj))
            motions, sim = self._motions_state(obj)
            act = entry_action(entry, T0, obj.name, arm, touches=(self.tw.slab.name,))
            track = PeakTracker(entry, T0, pts)
            step0 = self._step_fn(probe, series)

            def step(a):
                step0(a)
                track(_T(obj.get_position_orientation()), _T(sim.eef_pose(arm)))

            run = run_oracle(act, motions, sim, step, self.cfg.max_steps)
            T1 = _T(obj.get_position_orientation())
            Te = _T(sim.eef_pose(arm))
        finally:
            probe.uninstall()
        out = judge_push(entry, T0, T1, Te, pts, self.cfg, run, max(series) if series else 0.0, tip_gap_m=track.gap)
        return dict(out, schema=TRIAL_SCHEMA, restore=rec, base=plan.get("base"), trunk=plan.get("trunk"), run=run, wall_s=round(time.time() - t0, 3))

    def trial_brace(self, obj, entry, plan, state, rest, arm):
        t0 = time.time()
        rec, probe, th_ = self._start(obj, entry, plan, state, rest)
        series: List[float] = []
        try:
            T_start = _T(obj.get_position_orientation())
            motions, sim = self._motions_state(obj)
            act = entry_action(entry, T_start, obj.name, arm)
            run = run_oracle(act, motions, sim, self._step_fn(probe, series), self.cfg.max_steps)
            T0, Te0 = _T(obj.get_position_orientation()), _T(sim.eef_pose(arm))
            from ispatialgym.grasping.contact_library import hull_points

            pts = hull_points(self._geom_of(obj))
            pusher = self._pusher_obj()
            step = self._step_fn(probe, series)
            for c in pusher_path(entry, T0, pts, self.cfg, 1.0 / 30.0) + [None] * self.cfg.brace_hold_steps:
                if c is not None:  # the pusher one step further along the load; the robot holds (the brace arm stays at its contact)
                    pusher.set_position_orientation(th_.tensor(c, dtype=th_.float32), th_.tensor([0.0, 0.0, 0.0, 1.0]))
                for a in motions.hold(1):
                    step(a)
            pusher.set_position_orientation(th_.tensor([5.0, 5.0, -1.0]), th_.tensor([0.0, 0.0, 0.0, 1.0]))
            T1, Te1 = _T(obj.get_position_orientation()), _T(sim.eef_pose(arm))
        finally:
            probe.uninstall()
        load = np.asarray((entry.get("load") or {}).get("direction") or [0.0, 1.0, 0.0], dtype=float)
        out = judge_brace(entry, T0, T1, Te0, Te1, load, self.cfg, run, max(series) if series else 0.0)
        return dict(out, schema=TRIAL_SCHEMA, restore=rec, base=plan.get("base"), trunk=plan.get("trunk"), run=run, wall_s=round(time.time() - t0, 3))

    def _geom_of(self, obj):
        return getattr(obj, "_isg_geom")

    def _pusher_obj(self):
        """The brace load's pusher: a kinematic box (``kinematic_only``: set_position drives it into the partner, PhysX pushes it out)."""
        if self._pusher is None:
            from omnigibson.objects.primitive_object import PrimitiveObject

            sx, sy, sz = self.cfg.brace_pusher_size_m
            self._pusher = PrimitiveObject(name="isg_brace_pusher", primitive_type="Cube", size=1.0, kinematic_only=True, rgba=(0.2, 0.6, 0.9, 1.0),
                                           scale=self.tw.th.tensor([sy, sx, sz]))
            self.tw.scene.add_object(self._pusher)
            self._pusher.set_position_orientation(self.tw.th.tensor([5.0, 5.0, -1.0]), self.tw.th.tensor([0.0, 0.0, 0.0, 1.0]))
        return self._pusher


# ------------------------------------------------------------------------------------------------ CLI


def validate_cli(a) -> None:
    from ispatialgym.grasping import contact_library as CL

    cfg = ContactValidateConfig()
    arm = a.arm or (cfg.work_arm if a.kind == "push" else cfg.brace_arm)
    base = os.path.join(a.root, a.kind, "models")
    if getattr(a, "jobs", None):
        # the build's partition (the manifest's job ids of this shard): a shard validates the libraries it built
        ids = [j["id"] for j in CL._jobs(a.jobs, a.only, a.shard) if os.path.exists(CL.library_path(a.root, a.kind, j["id"]))]
    else:
        ids = sorted(os.listdir(base)) if os.path.isdir(base) else []
        if a.only:
            ids = [i for i in ids if i in set(a.only)]
        if a.shard:
            i, n = (int(v) for v in a.shard.split("/"))
            ids = [x for k, x in enumerate(ids) if k % n == i]
    if not ids:
        CL._log(a.root, "validate", a.shard, {"kind": a.kind, "ok": True, "reason": "no library in this shard"})
        print("GB_CONTACT_VALIDATE_DONE", flush=True)
        return
    t0 = time.time()
    world = KitContactWorld(cfg, arm)
    setup_s = round(time.time() - t0, 2)
    for eid in ids:
        path = CL.library_path(a.root, a.kind, eid)
        t1 = time.time()
        try:
            lib_k = CL.load_library(path)
            o = lib_k["object"]
            job = {"id": eid, "category": o["category"], "model": o["model"], "scale": o["scale"], "entry": o["entry"]}
            mlib, _rid, geom = CL.load_object(a.glib, job)
            obj = world.load(dict(job, rest_hint_wxyz=None), os.path.dirname(mlib.path))
            obj._isg_geom = geom
            try:
                tim = validate_library(world, lib_k, obj, geom, CL.hull_points(geom), cfg, arm)
            finally:
                world.unload(obj)
            tim["setup_s"] = setup_s
            S.save_json(path, lib_k)
            v = [e.get("validation", {}).get("verdict") for e in lib_k["entries"]]
            CL._log(a.root, "validate", a.shard, {"id": eid, "kind": a.kind, "arm": arm, "ok": True, "s": round(time.time() - t1, 2), "timing": tim,
                                                  "valid": v.count("valid"), "invalid": v.count("invalid"), "skipped": v.count(None)})
        except Exception as e:  # noqa: BLE001
            CL._log(a.root, "validate", a.shard, {"id": eid, "kind": a.kind, "ok": False, "error": repr(e), "tb": traceback.format_exc()[-1200:]})
    print("GB_CONTACT_VALIDATE_DONE", flush=True)
    import omnigibson as og

    og.shutdown()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--kind", required=True, choices=["push", "brace"])
    ap.add_argument("--glib", required=True)
    ap.add_argument("--arm", default=None, choices=["right", "left"], help="default: push right, brace left")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--jobs", default=None, help="the build manifest: validate the shard's job ids (the build's partition)")
    validate_cli(ap.parse_args(argv))


if __name__ == "__main__":
    main()
