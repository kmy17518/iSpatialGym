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
    python -m ispatialgym.grasping.validate_contacts replay --instance <...oracle.json ...> --kind push|brace --model-dir <v2 model dir>
"""

from __future__ import annotations

import argparse
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.grasping import schema as S

TRIAL_SCHEMA = "isg-contact-trial/1"


#: the base placement of a trial per family (owner 2026-11-07, the second cluster run: "reproduce the stance the SAMPLER uses"): the CONTACT
#: point ahead of / lateral to the base (base frame: +x ahead, +y left; the base yaw = the rule frame's) and the trunks, as measured on the
#: successful demos' sidecars ($OFH/out/*/<alias>/.../instances/*.oracle.json, contacts.<kind> vs stances.<its stance>.base_pose2d):
STANCE_RELATIONS: Dict[str, Dict[str, Any]] = {
    # 227 touching_01/02/04 demos: the push contact 0.69-0.84 m ahead, 0.06-0.20 m right of the push stance; trunks of those stances
    "touching_push": {"ahead_m": (0.72, 0.78, 0.84, 0.69), "lateral_m": (-0.08, -0.12, -0.15, -0.20),
                      "trunks": ((0.0, -0.61), (0.25, -0.53), (0.25, -0.27), (0.5, -0.45), (0.5, -0.23), (0.0, -0.31), (0.25, -0.8), (0.0, -0.92)),
                      "source": "227 touching_01/02/04 demos: push contact 0.69-0.84 m ahead, 0.06-0.20 m right of the push stance"},
    # the same demos' braces: the brace contact at the push's depth, 0.02-0.18 m LEFT of the same stance (one two-arm stance)
    "touching_brace": {"ahead_m": (0.72, 0.78, 0.84, 0.69), "lateral_m": (0.02, 0.07, 0.12, 0.18),
                       "trunks": ((0.0, -0.61), (0.25, -0.53), (0.25, -0.27), (0.5, -0.45), (0.5, -0.23), (0.0, -0.31), (0.25, -0.8), (0.0, -0.92)),
                       "source": "the touching demos' braces: 0.69-0.84 m ahead, 0.02-0.18 m left of the push stance"},
    # 207 axial_01-05 demos: the push init 0.49-1.10 m ahead, 0.41 m right to 0.03 m left of the manipulation stance
    "axial_push": {"ahead_m": (0.70, 0.80, 0.62, 0.90), "lateral_m": (-0.10, -0.18, 0.0, -0.28),
                   "trunks": ((0.0, -0.92), (0.0, -0.61), (0.25, -0.8), (0.25, -0.53), (0.5, -0.45), (0.5, -0.68), (0.75, -0.38), (0.75, -0.19), (0.95, -0.16),
                              (0.95, -0.31)),
                   "source": "207 axial_01-05 demos: push init 0.49-1.10 m ahead, -0.41..+0.03 m lateral of the manipulation stance"},
    # no braced axial demo (0 / 107): the axial stance's depths, the stop face to the left arm's side (the near-stop end)
    "axial_brace": {"ahead_m": (0.70, 0.80, 0.62, 0.90), "lateral_m": (0.0, 0.08, 0.16, 0.25),
                    "trunks": ((0.0, -0.92), (0.0, -0.61), (0.25, -0.8), (0.25, -0.53), (0.5, -0.45), (0.5, -0.68), (0.75, -0.38), (0.75, -0.19), (0.95, -0.16),
                               (0.95, -0.31)),
                    "source": "derived (0 braced axial demos): the axial stance's depths, the stop face on the left arm's side"},
}


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
    #: an ARC push (the axial push): the object turned >= |turn| - this (the axial orientation tolerance), its root drifted <= arc_drift_max_m
    push_turn_tol_rad: float = 0.2
    arc_drift_max_m: float = 0.05
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
    extras = world_extras(entry, T_world_obj)
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


#: record extras holding WORLD values (the sampler's records): the library's are in the rule frame the builder ran in (``T_rule_object``)
WORLD_Z_KEYS = ("fingertip_z", "contact_z")
WORLD_POINT_KEYS = ("contact_point",)


def world_extras(entry: Dict[str, Any], T_world_obj) -> Dict[str, Any]:
    """The entry's record extras with their WORLD values (``fingertip_z``, ``contact_z``, ``contact_point``) moved from the rule frame the
    builder ran in (``T_rule_object``: the object on a support at z 0) to the trial's object pose - the oracle reads them as world values
    (``execute_push``'s palm lift compares ``fingertip_z`` with the object's top: a rule-frame 2 cm raised every such push ~0.8 m)."""
    extras = dict(entry.get("extras") or {})
    if entry.get("T_rule_object") is None:
        return extras
    M = _T(T_world_obj) @ np.linalg.inv(S.pose_of(entry["T_rule_object"]))
    for k in WORLD_Z_KEYS:
        if isinstance(extras.get(k), (int, float)):
            extras[k] = round(float(extras[k]) + float(M[2, 3]), 6)  # a yaw + translation: z moves by the translation alone
    for k in WORLD_POINT_KEYS:
        v = extras.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 3:
            extras[k] = [round(float(x), 6) for x in (M[:3, :3] @ np.asarray(v, dtype=float) + M[:3, 3])]
    return extras


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
    if entry.get("arc") is not None:
        return judge_arc_push(entry, T_obj0, T_obj1, cfg, run, other_slab_N)
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


def judge_arc_push(entry: Dict[str, Any], T_obj0: np.ndarray, T_obj1: np.ndarray, cfg: ContactValidateConfig, run: Dict[str, Any],
                   other_slab_N: float = 0.0) -> Dict[str, Any]:
    """An ARC push (the axial push turns the target about its centre of mass): it passes when the object turned in the arc's sense by at least
    ``|turn| - push_turn_tol_rad``, did not topple, its root drifted <= ``arc_drift_max_m``, no non-finger slab load, the oracle's postcondition
    clean and the action finished (the straight push's along / sideways / tip-gap rules do not apply to a rotation)."""
    turn = float(entry["arc"]["turn_rad"])
    dyaw = math.atan2(math.sin(_yaw(T_obj1[:3, :3]) - _yaw(T_obj0[:3, :3])), math.cos(_yaw(T_obj1[:3, :3]) - _yaw(T_obj0[:3, :3])))
    turned = dyaw * (1.0 if turn >= 0 else -1.0)
    drift = float(np.linalg.norm((T_obj1[:3, 3] - T_obj0[:3, 3])[:2]))
    tilt = _tilt(T_obj0[:3, :3], T_obj1[:3, :3])
    need = abs(turn) - cfg.push_turn_tol_rad
    reasons = []
    if not run.get("completed", True):
        reasons.append("incomplete")
    if run.get("postcondition"):
        reasons.append(f"oracle:{run['postcondition']}")
    if turned < need:
        reasons.append("short_turn")
    if tilt > cfg.topple_tilt_rad:
        reasons.append("topple")
    if drift > cfg.arc_drift_max_m:
        reasons.append("drift")
    if other_slab_N > cfg.non_finger_contact_N:
        reasons.append("non_finger_contact")
    return {"verdict": "invalid" if reasons else "valid", "reasons": reasons,
            "measures": {"turned_rad": round(turned, 5), "needed_rad": round(need, 5), "turn_rad": round(turn, 5), "root_drift_m": round(drift, 5),
                         "tilt_change_rad": round(tilt, 5), "other_slab_N": round(float(other_slab_N), 4)}}


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
    (its extreme against ``load.direction``) to ``brace_press_m`` past it, at ``brace_pusher_speed_m_s``, on the line through ``load.point_object``
    (the axial push's arriving end) or the partner's centre, at its mid height (the box's bottom above the support)."""
    d = np.asarray((entry.get("load") or {}).get("direction") or [0.0, 1.0, 0.0], dtype=float)
    d[2] = 0.0
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    pts = pts_obj @ T_obj[:3, :3].T + T_obj[:3, 3]
    face = float((pts[:, :2] @ d[:2]).min())
    load = entry.get("load") or {}
    # on the line through the load point (the axial push's arriving end, goal object frame), else through the partner's centre
    c = T_obj[:3, :3] @ np.asarray(load["point_object"], dtype=float) + T_obj[:3, 3] if load.get("point_object") is not None else pts.mean(axis=0)
    z = max(0.5 * float(pts[:, 2].min() + pts[:, 2].max()), float(pts[:, 2].min()) + 0.5 * float(cfg.brace_pusher_size_m[2]) + 0.001)  # above the support
    half = 0.5 * float(cfg.brace_pusher_size_m[0])
    start = np.array([c[0], c[1], z]) + d * (face - float(c[:2] @ d[:2]) - cfg.brace_pusher_gap_m - half)
    travel = cfg.brace_pusher_gap_m + cfg.brace_press_m
    n = max(1, int(math.ceil(travel / max(cfg.brace_pusher_speed_m_s * dt, 1e-6))))
    return [start + d * travel * k / n for k in range(n + 1)]


# ------------------------------------------------------------------------------------------------ one model (any world with the KitContactWorld surface)


def entry_order(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The validation order: the entries the sampler would try (not ``sampler_screen``ed) first, their yaw slots round-robin (each slot's in
    library order = the sampler's order), then the screened ones - so the ``max_entries`` cap samples every yaw's first candidates (the first
    run validated 48 of the spray bottle's 1104 entries, all at yaw 0; the second 48 screened napkins blades the sampler rejects too)."""
    def round_robin(es):
        by_yaw: Dict[float, List[Dict[str, Any]]] = {}
        for e in es:
            by_yaw.setdefault(round(float(e.get("yaw_deg", 0.0)), 4), []).append(e)
        out: List[Dict[str, Any]] = []
        lists = list(by_yaw.values())
        for i in range(max((len(v) for v in lists), default=0)):
            out.extend(v[i] for v in lists if i < len(v))
        return out

    screened = [e for e in entries if (e.get("sampler_screen") or {}).get("hand_overlap")]
    live = [e for e in entries if not (e.get("sampler_screen") or {}).get("hand_overlap")]
    return round_robin(live) + screened  # the sampler's candidates first (it never records a screened class)


def _error_trial(e: BaseException) -> Dict[str, Any]:
    """A trial the oracle could not run: an unreachable motion (``SkillError`` ``plan_failed`` / an IK failure) is ``unreachable``, anything
    else ``error`` - the entry fails alone, the model goes on."""
    reason = getattr(e, "reason", None)
    unreachable = reason in ("plan_failed", "ik_failed", "unreachable") or "IK failed" in str(e)
    return {"schema": TRIAL_SCHEMA, "verdict": "unreachable" if unreachable else "invalid",
            "reasons": ["unreachable" if unreachable else f"error:{type(e).__name__}"] + ([f"oracle:{reason}"] if reason else []), "error": str(e)[:600]}


def validate_library(world, lib_k: Dict[str, Any], obj, geom, pts_obj: np.ndarray, cfg: ContactValidateConfig, arm: str) -> Dict[str, Any]:
    """Validate the entries of one library (``lib_k``: one kind) of a loaded object in ``world`` (the :class:`KitContactWorld` surface:
    ``settle(obj, geom, quat)``, ``plan_entry(obj, geom, entry, T_world_obj, arm)``, ``trial_push / trial_brace(obj, entry, plan, state, rest,
    arm)``); writes each entry's ``validation`` (verdict ``valid`` / ``invalid`` / ``unreachable``) and returns the timing (per model / rest /
    kind). Every entry is planned against its yaw's SETTLED rest pose and tried from the restored rest snapshot, and an entry that cannot be
    placed or run fails alone (``no_placement``, ``chain_unreachable``, ``unreachable``, ``error:...``)."""
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    t_all = time.time()
    tim = {"settle_s": 0.0, "plan_s": 0.0, "trials_s": 0.0, "n_trials": 0, "n_entries": 0, "n_errors": 0}
    rest_q = np.asarray(lib_k["object"]["rest"]["quat_xyzw"], dtype=float)
    settled: Dict[float, Tuple[Dict[str, Any], Any]] = {}
    n_live = 0
    for e in entry_order(lib_k["entries"]):
        yaw = round(float(e.get("yaw_deg", 0.0)), 4)
        screen = (e.get("sampler_screen") or {}).get("hand_overlap")
        if screen:
            # the sampler never records this class here (its hand_overlap rule): not a contact the oracle would execute; not counted
            e["validation"] = {"schema": TRIAL_SCHEMA, "verdict": "screened", "reasons": ["sampler_hand_overlap"], "screen": screen, "trials": []}
            tim["n_screened"] = tim.get("n_screened", 0) + 1
            continue
        if n_live >= cfg.max_entries:
            continue
        n_live += 1
        tim["n_entries"] += 1
        try:
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
        except Exception as ex:  # noqa: BLE001 - this entry fails, the model goes on
            tim["n_errors"] += 1
            e["validation"] = dict(_error_trial(ex), trials=[], stage="settle/plan")
            continue
        if plan is None or plan.get("q") is None:
            e["validation"] = {"schema": TRIAL_SCHEMA, "verdict": "invalid", "reasons": ["no_placement"], "plan_fail": (plan or {}).get("fail"), "trials": []}
            continue
        if plan.get("chain_unreachable"):
            e["validation"] = {"schema": TRIAL_SCHEMA, "verdict": "unreachable", "reasons": ["chain_unreachable"], "plan": _plan_facts(plan), "trials": []}
            continue
        trials = []
        t0 = time.time()
        for _k in range(cfg.trials_per_entry):
            try:
                trials.append(world.trial_push(obj, e, plan, state, rest, arm) if lib_k["kind"] == "push" else world.trial_brace(obj, e, plan, state, rest, arm))
            except Exception as ex:  # noqa: BLE001 - the oracle's SkillError (an unreachable pre-contact) or anything else: this entry alone
                tim["n_errors"] += 1
                trials.append(dict(_error_trial(ex), plan=_plan_facts(plan)))
            tim["n_trials"] += 1
        tim["trials_s"] += time.time() - t0
        verdicts = {t["verdict"] for t in trials}
        verdict = "valid" if verdicts == {"valid"} else ("unreachable" if verdicts == {"unreachable"} else "invalid")
        e["validation"] = {"schema": TRIAL_SCHEMA, "verdict": verdict, "reasons": sorted({r for t in trials for r in t["reasons"]}),
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


def placement_offsets(family: str, contact_xy, obj_xy, support_half_x: float, edge_clear: float) -> Tuple[Tuple[float, float], ...]:
    """The family's stance relation (:data:`STANCE_RELATIONS`: the contact ``ahead`` / ``lateral`` of the base, yaw 0) as
    ``TabletopWorld.plan``'s object-relative base offsets (it puts the base ``support half + edge clear + off[0]`` behind the OBJECT and
    ``off[1]`` to its side): the base at ``contact - (ahead, lateral)``."""
    rel = STANCE_RELATIONS[family]
    out = []
    for ah in rel["ahead_m"]:
        for la in rel["lateral_m"]:
            bx, by = float(contact_xy[0]) - ah, float(contact_xy[1]) - la
            out.append((round(float(obj_xy[0]) - bx - float(support_half_x) - float(edge_clear), 5), round(float(obj_xy[1]) - by, 5)))
    return tuple(out)


def start_pose_of(entry: Dict[str, Any], pts_obj: np.ndarray, T_obj: np.ndarray, margin: float = 0.003) -> Optional[str]:
    """Where a trial sets the robot (it is teleported there, no approach): the EXECUTED pre-contact when the hand clears the object there (the
    hand beyond the fingers by ``contact_poses.hand_overlap`` against its hull, the finger spheres at the entry's opening), else the RECORDED one
    (the oracle plans its own way down to the executed pre), else None (``start_in_object``: no trial)."""
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.v2 import tabletop as TT

    w = CPZ.entry_world(entry, T_obj)
    hull = th.as_tensor(pts_obj, dtype=th.float32)
    obj_pose = (th.tensor(T_obj[:3, 3], dtype=th.float32), th.tensor(S.mat_to_quat(T_obj[:3, :3]), dtype=th.float32))
    sec = CPZ.HullSections(C.transform_points(obj_pose, hull))
    q = float(entry["fingers"]["opening_m"])
    # the fingers too (``hand_overlap`` covers the hand beyond them): the finger spheres moved ``q`` outward along the closing axis
    fingers = th.tensor([[x, y + math.copysign(q, y), z] for (x, y, z), _r in TT.FINGER_SPHERES_EEF], dtype=th.float32)
    radii = th.tensor([r for _c, r in TT.FINGER_SPHERES_EEF], dtype=th.float32)
    for key, pose in (("executed", w["pre_executed"]), ("recorded", w["pre"])):
        if CPZ.hand_overlap(pose, hull, obj_pose, margin=margin, against="hull") is not None:
            continue
        d = sec.distance(C.transform_points((th.as_tensor(pose[0]).float(), th.as_tensor(pose[1]).float()), fingers))
        if d is not None and bool((d < radii + margin).any()):
            continue
        return key
    return None


def _entry_world(entry: Dict[str, Any], T_obj: np.ndarray) -> Dict[str, Any]:
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    return CPZ.entry_world(entry, T_obj)


def _plan_facts(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {k: plan.get(k) for k in ("base", "trunk", "cost", "start_lift_m", "chain_reach", "chain_unreachable", "start", "relation")}


def _pose7(T: np.ndarray) -> List[float]:
    return [round(float(v), 4) for v in T[:3, 3]] + [round(float(v), 4) for v in S.mat_to_quat(T[:3, :3])]


class TrialTrace:
    """The per-trial debug record (the first cluster run's spray bottle was flung with no trace of how): the object / hand poses at the start,
    at the contact (the first step the eef is within ``contact_tol_m`` of the contact pose), at the first object motion (> ``motion_tol_m``)
    and at the end, the step and the hand's distance to the contact when the object first moved, the probe's loads at that step (fingers on
    the object, a non-finger link on the slab), and every ``every``-th step (object xyz + tilt, eef xyz, loads)."""

    def __init__(self, T_obj0: np.ndarray, contact_T: np.ndarray, every: int = 10, contact_tol_m: float = 0.01, motion_tol_m: float = 0.005):
        self.T0, self.Tc, self.every, self.ctol, self.mtol = T_obj0, contact_T, int(every), float(contact_tol_m), float(motion_tol_m)
        self.k = 0
        self.rec: Dict[str, Any] = {"obj_start": _pose7(T_obj0), "contact_pose": _pose7(contact_T), "at_contact": None, "first_motion": None, "trace": []}

    def __call__(self, T_obj: np.ndarray, T_eef: np.ndarray, loads: Optional[Dict[str, Any]] = None) -> None:
        self.k += 1
        moved = float(np.linalg.norm(T_obj[:3, 3] - self.T0[:3, 3]))
        d_c = float(np.linalg.norm(T_eef[:3, 3] - self.Tc[:3, 3]))
        loads = {k: (round(float(v), 3) if isinstance(v, (int, float)) else v) for k, v in (loads or {}).items()}
        if self.rec["at_contact"] is None and d_c <= self.ctol:
            self.rec["at_contact"] = {"step": self.k, "obj": _pose7(T_obj), "eef": _pose7(T_eef), "obj_moved_m": round(moved, 4)}
        if self.rec["first_motion"] is None and moved > self.mtol:
            self.rec["first_motion"] = {"step": self.k, "obj": _pose7(T_obj), "eef": _pose7(T_eef), "eef_to_contact_m": round(d_c, 4),
                                        "before_contact": self.rec["at_contact"] is None, "loads": loads}
        if self.k % self.every == 0:
            self.rec["trace"].append([self.k, _pose7(T_obj)[:3], round(_tilt(self.T0[:3, :3], T_obj[:3, :3]), 4), _pose7(T_eef)[:3],
                                      loads.get("finger_obj"), loads.get("other_slab"), loads.get("other_link")])

    def end(self, T_obj: np.ndarray, T_eef: np.ndarray) -> Dict[str, Any]:
        self.rec.update(obj_end=_pose7(T_obj), eef_end=_pose7(T_eef), steps=self.k)
        return self.rec


# ------------------------------------------------------------------------------------------------ the Isaac Sim world (GPU node; not unit-tested)


class KitContactWorld:
    """``validate_tabletop.TabletopWorld`` for contacts: the base yaw fixed to the rule frame (``base_yaws`` 1: the robot facing +x), the base
    offsets on the arm's side, the oracle's ``Motions`` with the cuRobo backend as its planner, ``executor.SimState`` over a task stub (the
    object and the slab in scope), the world's own ``ForceProbe`` (``TabletopWorld.step``) and its sleep assist, a kinematic pusher box for
    the brace load created ONCE before any object (adding / removing prims under a live probe invalidated the physics tensor views)."""

    def __init__(self, cfg: ContactValidateConfig, arm: str):
        from ispatialgym.grasping import validate_tabletop as VT

        offs = cfg.base_offsets if arm == "right" else tuple((a, -b) for a, b in cfg.base_offsets)
        self.vcfg = VT.ValidateConfig(arm=arm, base_yaws=1, base_offsets=offs)
        self.tw = VT.TabletopWorld(self.vcfg)
        self.cfg, self.arm, self.VT = cfg, arm, VT
        self._pusher = None
        self._pusher_obj()  # before any object: no prim is added while an object's probe view is live

    def load(self, job: Dict[str, Any], model_dir: str):
        return self.tw.load(job, model_dir)

    def unload(self, obj):
        self.tw.unload(obj)

    def settle(self, obj, geom, quat0):
        rest, state = self.tw.settle(obj, geom, np.asarray(quat0, dtype=float))
        p, q = (x.clone() for x in obj.get_position_orientation())
        rest = dict(rest, world_position=[float(v) for v in p.tolist()], world_quat_xyzw=[float(v) for v in q.tolist()])  # the restore keys stay
        return rest, state

    def clean_world(self, obj) -> None:
        """The cuRobo collision world of the validator's own IK (the placement, the chain check): the robot and the ground only, as the
        sampler's exact IK at a contact (the support and the pushed / braced object ignored at the contact pose; the contact policy and the
        arm-object check are explicit). The oracle's planning syncs the world with the slab and the object in it (Motions.arm_to), and the
        stale meshes made every IK after a model's first trial fail (run C: 1 valid, then 47 no_placement / chain_unreachable)."""
        ignore = [self.tw.slab, obj] + ([self._pusher] if self._pusher is not None else [])
        self.tw.backend.update_world(ignore_objects=ignore)

    def set_support_height(self, z: Optional[float]) -> None:
        """The slab top of the next library (the alias family's support height: its reach and the arm's posture depend on it); None = the
        world's default (``ValidateConfig.slab_top_z``)."""
        import dataclasses

        self.tw.cfg = dataclasses.replace(self.vcfg, slab_top_z=float(z) if z is not None else self.vcfg.slab_top_z)

    def plan_entry(self, obj, geom, entry, T_obj, arm):
        """The configuration at the trial's start pose (:func:`start_pose_of`) against the SETTLED rest pose ``T_obj`` (not the live object, which
        a previous trial may have moved), the base placed by the family's measured stance relation (:data:`STANCE_RELATIONS`: the contact
        ahead / lateral of the base, the trunks of the demos; ``TabletopWorld.plan``: batched IK, the whole robot's contact policy against the
        slab, the arm clear of the object, the most comfortable solution), then the chain's other poses (the contact, an arc's mid and goal,
        a straight push's goal) by IK from that base / trunk (``chain_reach``; ``chain_unreachable`` names the ones without a solution).
        ``no_placement`` = no relation x trunk had an IK solution of the start pose passing those checks (``fail``: the worst link, or none)."""
        import dataclasses

        from ispatialgym.grasping.contact_library import hull_points
        from ispatialgym.sampling.v2 import contact_poses as CPZ

        if T_obj is None:
            T_obj = CPZ._mat(obj.get_position_orientation())
        self.clean_world(obj)
        start = start_pose_of(entry, hull_points(geom), T_obj)
        if start is None:
            return {"q": None, "fail": "start_in_object: the hand meets the object at both pre-contacts (the robot is set there)"}
        w = CPZ.entry_world(entry, T_obj)
        fam = entry.get("family")
        cfg0 = self.tw.cfg
        if fam in STANCE_RELATIONS:
            offs = placement_offsets(fam, w["contact"][0][:2].tolist(), T_obj[:2, 3], float(self.tw.support.half[0]), cfg0.slab_edge_clear_m)
            self.tw.cfg = dataclasses.replace(cfg0, base_yaws=1, base_offsets=offs, trunks=tuple(STANCE_RELATIONS[fam]["trunks"]))
        try:
            cand = {"name": entry["digest"], "T_object_eef": entry["pre"][start], "preshape_q": float(entry["fingers"]["opening_m"])}
            best = self.tw.plan(obj, geom, [cand], (T_obj[:3, 3].copy(), S.mat_to_quat(T_obj[:3, :3])))
        finally:
            self.tw.cfg = cfg0
        b = best.get(entry["digest"])
        if b is None:
            return {"q": None, "fail": f"no IK of the {start} pre-contact from any of the {fam} stance relation's placements x trunks"}
        if b.get("q") is None:
            return dict(b, fail=f"IK found, the contact policy / arm-object check failed: {b.get('fail')}")
        b = dict(b, T_world_obj=T_obj, start=start, relation=fam)
        poses = {"contact": w["contact"]}
        if entry.get("arc") is not None:
            To = CPZ._mat(T_obj) if not isinstance(T_obj, np.ndarray) else T_obj
            for k in ("mid", "goal"):
                Tk = To @ S.pose_of(entry["arc"][f"{k}_object"])
                poses[k] = (th.tensor(Tk[:3, 3], dtype=th.float32), th.tensor(S.mat_to_quat(Tk[:3, :3]), dtype=th.float32))
        elif w["goal"] is not None:
            poses["goal"] = w["goal"]
        try:
            q_seed = b["q"].clone()
            q_seed[self.tw.robot.gripper_control_idx[arm]] = float(entry["fingers"]["opening_m"])
            succ, _qs = self.tw.backend.solve_ik_batch(f"{arm}_arm", self.tw.M.EEF_LINK[arm], list(poses.values()), q_seed, batch_size=8, seeds=self.vcfg.ik_seeds)
            reach = {k: bool(v) for k, v in zip(poses, succ.tolist())}
        except Exception as ex:  # noqa: BLE001 - the reach check is advisory: the trial decides
            reach = {"error": str(ex)[:200]}
        b["chain_reach"] = reach
        b["chain_unreachable"] = [k for k, v in reach.items() if v is False]
        return b

    def _motions_state(self, obj):
        from ispatialgym.oracle.executor import SimState
        from ispatialgym.oracle.motions import Motions, MotionsConfig

        motions = Motions(self.tw.env, self.tw.backend, cfg=MotionsConfig(), robot=self.tw.robot)
        motions.state.hold_current()
        task = SimpleNamespace(object_scope={obj.name: obj, self.tw.slab.name: self.tw.slab}, _tro=None)
        return motions, SimState(self.tw.env, task, self.tw.robot)

    def _start(self, obj, entry, plan, state, rest):
        """The rest snapshot restored (verified), the robot set at the planned configuration with the entry's fingers, the drive targets held,
        two hold steps with the sleep assist (as ``validate_tabletop``'s trial)."""
        rec = self.tw.restore(obj, state, rest)
        q = plan["q"].clone()
        q[self.tw.robot.gripper_control_idx[self.arm]] = float(entry["fingers"]["opening_m"])
        self.tw.robot.set_joint_positions(q)
        self.tw.robot.keep_still()
        obj.wake()
        sa = self.tw.SA.SleepAssist(self.tw.robot, {obj.name: obj}, "gc") if self.vcfg.sleep_assist else None
        self.tw.probe.take()  # the restore's own contact samples are not the trial's
        return rec, sa

    def _stepper(self, obj, sim, arm, sa, series, trace):
        k = [0]

        def step(a):
            f = self.tw.step(a, sa, k[0])  # the world's probe (TabletopWorld.load), the sleep assist
            k[0] += 1
            series.append(float(f.get("other_slab", 0.0) or 0.0))
            if trace is not None:
                trace(_T(obj.get_position_orientation()), _T(sim.eef_pose(arm)), f)
            return f
        return step

    def _hold(self, motions, step, n: int) -> None:
        for a in motions.hold(n):
            step(a)

    def trial_push(self, obj, entry, plan, state, rest, arm):
        t0 = time.time()
        rec, sa = self._start(obj, entry, plan, state, rest)
        series: List[float] = []
        from ispatialgym.grasping.contact_library import hull_points

        pts = hull_points(self._geom_of(obj))
        motions, sim = self._motions_state(obj)
        T_start = _T(obj.get_position_orientation())
        w = _entry_world(entry, T_start)
        trace = TrialTrace(T_start, _T(w["contact"]))
        trace.rec.update(obj_rest=[round(float(v), 4) for v in rest.get("world_position", [])] + [round(float(v), 4) for v in rest.get("world_quat_xyzw", [])],
                         eef_start=_pose7(_T(sim.eef_pose(arm))), restore=rec)
        step = self._stepper(obj, sim, arm, sa, series, trace)
        self._hold(motions, step, self.vcfg.init_hold_steps)
        T0 = _T(obj.get_position_orientation())
        act = entry_action(entry, T0, obj.name, arm, touches=(self.tw.slab.name,))
        track = PeakTracker(entry, T0, pts)

        def step_track(a):
            step(a)
            track(_T(obj.get_position_orientation()), _T(sim.eef_pose(arm)))

        try:
            run = run_oracle(act, motions, sim, step_track, self.cfg.max_steps)
        except Exception as ex:  # noqa: BLE001 - the entry fails alone, with its trace
            out = _error_trial(ex)
            return dict(out, restore=rec, base=plan.get("base"), trunk=plan.get("trunk"), debug=trace.end(_T(obj.get_position_orientation()), _T(sim.eef_pose(arm))),
                        wall_s=round(time.time() - t0, 3))
        T1, Te = _T(obj.get_position_orientation()), _T(sim.eef_pose(arm))
        out = judge_push(entry, T0, T1, Te, pts, self.cfg, run, max(series) if series else 0.0, tip_gap_m=track.gap)
        return dict(out, schema=TRIAL_SCHEMA, restore=rec, base=plan.get("base"), trunk=plan.get("trunk"), run=run, debug=trace.end(T1, Te),
                    wall_s=round(time.time() - t0, 3))

    def trial_brace(self, obj, entry, plan, state, rest, arm):
        t0 = time.time()
        rec, sa = self._start(obj, entry, plan, state, rest)
        series: List[float] = []
        th_ = self.tw.th
        from ispatialgym.grasping.contact_library import hull_points

        pts = hull_points(self._geom_of(obj))
        motions, sim = self._motions_state(obj)
        T_start = _T(obj.get_position_orientation())
        w = _entry_world(entry, T_start)
        trace = TrialTrace(T_start, _T(w["contact"]))
        trace.rec.update(eef_start=_pose7(_T(sim.eef_pose(arm))), restore=rec)
        step = self._stepper(obj, sim, arm, sa, series, trace)
        self._hold(motions, step, self.vcfg.init_hold_steps)
        act = entry_action(entry, _T(obj.get_position_orientation()), obj.name, arm)
        try:
            run = run_oracle(act, motions, sim, step, self.cfg.max_steps)
            T0, Te0 = _T(obj.get_position_orientation()), _T(sim.eef_pose(arm))
            trace.rec["braced"] = {"obj": _pose7(T0), "eef": _pose7(Te0)}
            pusher = self._pusher_obj()
            for c in pusher_path(entry, T0, pts, self.cfg, 1.0 / 30.0) + [None] * self.cfg.brace_hold_steps:
                if c is not None:  # the pusher one step further along the load; the robot holds (the brace arm stays at its contact)
                    pusher.set_position_orientation(th_.tensor(c, dtype=th_.float32), th_.tensor([0.0, 0.0, 0.0, 1.0]))
                self._hold(motions, step, 1)
        except Exception as ex:  # noqa: BLE001 - the entry fails alone, with its trace
            out = _error_trial(ex)
            return dict(out, restore=rec, base=plan.get("base"), trunk=plan.get("trunk"), debug=trace.end(_T(obj.get_position_orientation()), _T(sim.eef_pose(arm))),
                        wall_s=round(time.time() - t0, 3))
        finally:
            self._park_pusher()
        T1, Te1 = _T(obj.get_position_orientation()), _T(sim.eef_pose(arm))
        load = np.asarray((entry.get("load") or {}).get("direction") or [0.0, 1.0, 0.0], dtype=float)
        out = judge_brace(entry, T0, T1, Te0, Te1, load, self.cfg, run, max(series) if series else 0.0)
        return dict(out, schema=TRIAL_SCHEMA, restore=rec, base=plan.get("base"), trunk=plan.get("trunk"), run=run, debug=trace.end(T1, Te1),
                    wall_s=round(time.time() - t0, 3))

    def replay(self, sit: Dict[str, Any], obj, geom) -> Dict[str, Any]:
        """A recorded demo's situation (:func:`replay_situation`) rebuilt: the slab at the recorded support height, the object settled at
        its recorded stance-frame pose, the robot at the oracle's own witness configuration of the pre-contact (base at the origin), then the
        validator's chain IK of the record's poses from there (the clean world) and the ORACLE's action of the record itself, judged as
        the library entries are. ``divergence`` names what differs from the real run."""
        from ispatialgym.sampling.v2 import contact_poses as CPZ

        self.set_support_height(sit["support_z"])
        T_o = sit["T_base_object"]
        rest, state = self.tw.settle(obj, geom, S.mat_to_quat(T_o[:3, :3]), obj_xy=(float(T_o[0, 3]), float(T_o[1, 3])))
        p, q = (x.clone() for x in obj.get_position_orientation())
        rest = dict(rest, world_position=[float(v) for v in p.tolist()], world_quat_xyzw=[float(v) for v in q.tolist()])
        T_settled = CPZ._mat((p, q))
        entry = record_entry(dict(sit, T_base_object=T_settled))
        out: Dict[str, Any] = {"situation": {k: sit[k] for k in ("kind", "pose_class", "support_z", "stance", "relation", "object_from")},
                               "settle_shift_m": round(float(np.linalg.norm(T_settled[:3, 3] - T_o[:3, 3])), 5)}
        self.clean_world(obj)
        if sit.get("witness_q") is None:
            return dict(out, error="no witness configuration in the record's ik_witnesses")
        qw = th.tensor(sit["witness_q"], dtype=th.float32)
        plan = {"q": qw, "base": [0.0, 0.0, 0.0], "trunk": [sit["stance"]["translate"], sit["stance"]["tilt"]]}
        w = CPZ.entry_world(entry, T_settled)
        poses = {"contact": w["contact"]}
        if entry.get("arc") is not None:
            for k in ("mid", "goal"):
                Tk = T_settled @ S.pose_of(entry["arc"][f"{k}_object"])
                poses[k] = (th.tensor(Tk[:3, 3], dtype=th.float32), th.tensor(S.mat_to_quat(Tk[:3, :3]), dtype=th.float32))
        elif w["goal"] is not None:
            poses["goal"] = w["goal"]
        succ, _qs = self.tw.backend.solve_ik_batch(f"{self.arm}_arm", self.tw.M.EEF_LINK[self.arm], list(poses.values()), qw, batch_size=8, seeds=self.vcfg.ik_seeds)
        out["chain_reach"] = {k: bool(v) for k, v in zip(poses, succ.tolist())}
        trial = self.trial_push(obj, entry, plan, state, rest, self.arm) if sit["kind"] == "push" else self.trial_brace(obj, entry, plan, state, rest, self.arm)
        out["trial"] = trial
        div = []
        if not all(out["chain_reach"].values()):
            div.append(f"chain IK from the oracle's own configuration fails for {[k for k, v in out['chain_reach'].items() if not v]}")
        if trial.get("verdict") != "valid":
            div.append(f"the oracle's action of the record is {trial.get('verdict')} here: {trial.get('reasons')}")
        if out["settle_shift_m"] > 0.005:
            div.append(f"the object settles {out['settle_shift_m']:.3f} m from its recorded pose (the slab vs the real support)")
        out["divergence"] = div or ["none: the replay passes as the demo did"]
        return out

    def _geom_of(self, obj):
        return getattr(obj, "_isg_geom")

    def _park_pusher(self):
        if self._pusher is not None:
            self._pusher.set_position_orientation(self.tw.th.tensor([5.0, 5.0, -1.0]), self.tw.th.tensor([0.0, 0.0, 0.0, 1.0]))

    def _pusher_obj(self):
        """The brace load's pusher: a kinematic box (``kinematic_only``: set_position drives it into the partner, PhysX pushes it out), parked
        below the floor 5 m away between loads."""
        if self._pusher is None:
            from omnigibson.objects.primitive_object import PrimitiveObject

            sx, sy, sz = self.cfg.brace_pusher_size_m
            self._pusher = PrimitiveObject(name="isg_brace_pusher", primitive_type="Cube", size=1.0, kinematic_only=True, rgba=(0.2, 0.6, 0.9, 1.0),
                                           scale=self.tw.th.tensor([sy, sx, sz]))
            self.tw.scene.add_object(self._pusher)
            self._park_pusher()
            self.tw.og.sim.step()
        return self._pusher


# ------------------------------------------------------------------------------------------------ replay of a recorded demo situation


def _mat2d(base_pose2d) -> np.ndarray:
    x, y, yaw = (float(v) for v in base_pose2d)
    T = np.eye(4)
    T[:3, :3] = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    T[:2, 3] = [x, y]
    return T


def _pose_T(d: Dict[str, Any]) -> np.ndarray:
    return S.pose_mat(d["position"], d["orientation"])


def _pose_dict(T: np.ndarray) -> Dict[str, List[float]]:
    return {"position": [float(v) for v in T[:3, 3]], "orientation": [float(v) for v in S.mat_to_quat(T[:3, :3])]}


def replay_situation(oracle: Dict[str, Any], kind: str, tro: Dict[str, Any]) -> Dict[str, Any]:
    """The exact situation of a recorded demo's push / brace (``oracle`` = the instance's ``*.oracle.json`` sidecar, ``tro`` = its
    ``-tro_state.json``) in the record's STANCE frame (the base at the origin facing +x, z kept): the base pose and trunk of the record's
    stance, the arm configuration of the oracle's witness at the pre-contact (``ik_witnesses``), the object's pose at the action (a pushed
    target the program placed first: its place = ``grasps[].place`` o inv(``eef_in_object``); else its program-start pose), the support height
    (the object's start AABB bottom), the record moved into the stance frame (poses, direction, world extras), its stance relation (the
    contact ahead / lateral of the base)."""
    rec = dict((oracle.get("contacts") or {})[kind])
    st = (oracle.get("stances") or {})[rec.get("stance") or "manipulation"]
    Tb = _mat2d(st["base_pose2d"])
    Tbi = np.linalg.inv(Tb)
    obj = rec["object"]
    g = next((g for g in oracle.get("grasps") or [] if g.get("object") == obj and g.get("place") and g.get("eef_in_object")), None)
    if g is not None and kind == "push":
        T_obj = _pose_T(g["place"]) @ np.linalg.inv(_pose_T(g["eef_in_object"]))
        obj_from = "place"
    else:
        rl = tro[obj]["root_link"]
        T_obj = S.pose_mat(rl["pos"], rl["ori"])
        obj_from = "program start"
    support_z = float(tro[obj]["non_kin"]["AABB"]["aabb"][2])
    keys = [k for k in ("pre", "init", "mid", "goal", "contact") if isinstance(rec.get(k), dict)]
    rec_b = dict(rec)
    for k in keys:
        rec_b[k] = _pose_dict(Tbi @ _pose_T(rec[k]))
    if rec.get("direction") is not None:
        rec_b["direction"] = [float(v) for v in (Tbi[:3, :3] @ np.asarray(rec["direction"], dtype=float))]
    for k in WORLD_POINT_KEYS:
        if isinstance(rec.get(k), list) and len(rec[k]) == 3:
            rec_b[k] = [float(v) for v in (Tbi[:3, :3] @ np.asarray(rec[k], dtype=float) + Tbi[:3, 3])]
    wit = next((w for w in oracle.get("ik_witnesses") or [] if w.get("name") == kind and w.get("arm") == rec.get("arm") and "pre" in (w.get("targets") or {})), None)
    q = None if wit is None else [float(v) for v in wit["targets"]["pre"]["q"]]
    if q is not None:  # the base joints (x, y, z, rx, ry, yaw) into the stance frame: the base at the origin
        q[0], q[1], q[5] = 0.0, 0.0, 0.0
    contact = rec_b.get("init") or rec_b.get("contact")
    return {"kind": kind, "arm": rec.get("arm"), "object": obj, "object_from": obj_from, "T_base_object": Tbi @ T_obj, "support_z": support_z,
            "stance": {"name": rec.get("stance"), "base_pose2d": list(st["base_pose2d"]), "translate": st.get("translate"), "tilt": st.get("tilt"),
                       "stand_off": st.get("stand_off")}, "witness_q": q, "record": rec_b, "record_world": rec,
            "relation": {"contact_ahead_m": round(float(contact["position"][0]), 4), "contact_lateral_m": round(float(contact["position"][1]), 4)},
            "pose_class": rec.get("pose_class") or rec.get("kind")}


def replay_entry_check(sit: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    """The library entry of the replayed record's class at the replayed object pose vs the record: the contact eef's position / angle error
    and the pre-contact's (the library frame against what the sampler recorded)."""
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    w = CPZ.entry_world(entry, sit["T_base_object"])
    rec = sit["record"]
    c_rec = _pose_T(rec.get("init") or rec["contact"])
    c_lib = _T(w["contact"])
    p_rec, p_lib = _pose_T(rec["pre"]), _T(w["pre"])
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(c_rec[:3, :3].T @ c_lib[:3, :3])) - 1.0) / 2.0))))
    return {"class": entry["class"], "contact_err_m": round(float(np.linalg.norm(c_rec[:3, 3] - c_lib[:3, 3])), 5), "contact_angle_deg": round(ang, 3),
            "pre_err_m": round(float(np.linalg.norm(p_rec[:3, 3] - p_lib[:3, 3])), 5)}


def record_entry(sit: Dict[str, Any]) -> Dict[str, Any]:
    """The replayed record as a schema entry at its object pose (the oracle action of the replay is the RECORD itself; the entry form feeds
    the validator's judge and chain check)."""
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    rec = sit["record"]
    T_obj = sit["T_base_object"]
    extras = {k: v for k, v in rec.items() if k not in ("arm", "object", "pre", "init", "mid", "goal", "contact", "direction", "distance_m", "stance", "kind", "orient")}
    if sit["kind"] == "push":
        e = CPZ.contact_entry("push", sit["pose_class"] or "push", "replay", "work", T_obj, _poseq(rec["init"]), _poseq(rec["pre"]), extras,
                              direction=rec["direction"], stroke_m=rec.get("distance_m"), arc=rec.get("mid") is not None)
        if rec.get("mid") is not None:
            Ti = np.linalg.inv(T_obj)
            turn = math.atan2(*(lambda Rm: (Rm[1, 0], Rm[0, 0]))(_pose_T(rec["goal"])[:3, :3] @ _pose_T(rec["init"])[:3, :3].T))
            e["arc"] = {"mid_object": CPZ._pose_dict(Ti @ _pose_T(rec["mid"])), "goal_object": CPZ._pose_dict(Ti @ _pose_T(rec["goal"])), "turn_rad": round(turn, 6)}
    else:
        e = CPZ.contact_entry("brace", sit["pose_class"] or "brace", "replay", "brace", T_obj, _poseq(rec["contact"]), _poseq(rec["pre"]), extras)
    e["T_rule_object"] = CPZ._pose_dict(T_obj)  # the record's world extras are the stance frame's once the object stands there
    return e


def _poseq(d: Dict[str, Any]):
    return th.tensor(d["position"], dtype=th.float32), th.tensor(d["orientation"], dtype=th.float32)


# ------------------------------------------------------------------------------------------------ CLI


def library_job(lib_k: Dict[str, Any], eid: str) -> Dict[str, Any]:
    """The object job of a library: its model directory as the BUILD read it (``object.source``: a manifest's ``model_dir``, e.g. a roles_all
    entry read in place under $GBH), else ``entry`` under ``--glib``."""
    o = lib_k["object"]
    job = {"id": eid, "category": o["category"], "model": o["model"], "scale": o["scale"], "entry": o["entry"]}
    if o.get("source") and os.path.isdir(o["source"]):
        job["model_dir"] = o["source"]
    return job


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
            job = library_job(lib_k, eid)
            mlib, _rid, geom = CL.load_object(a.glib, job)
            world.set_support_height((lib_k.get("validate") or {}).get("support_z"))
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
                                                  "valid": v.count("valid"), "invalid": v.count("invalid"), "unreachable": v.count("unreachable"), "skipped": v.count(None)})
        except Exception as e:  # noqa: BLE001
            CL._log(a.root, "validate", a.shard, {"id": eid, "kind": a.kind, "ok": False, "error": repr(e), "tb": traceback.format_exc()[-1200:]})
    print("GB_CONTACT_VALIDATE_DONE", flush=True)
    import omnigibson as og

    og.shutdown()


def replay_cli(argv) -> None:
    """``validate_contacts replay --instance <instances/...oracle.json> --kind push|brace --model-dir <v2 model dir of the contacted object>``:
    one recorded demo's situation rebuilt and run (:meth:`KitContactWorld.replay`); one ``GB_CONTACT_REPLAY {json}`` line."""
    import json as _json

    from ispatialgym.grasping import contact_library as CL

    ap = argparse.ArgumentParser(prog="validate_contacts replay")
    ap.add_argument("--instance", nargs="+", required=True, help="the instances' *.oracle.json (their -tro_state.json beside them)")
    ap.add_argument("--kind", required=True, choices=["push", "brace"])
    ap.add_argument("--model-dir", required=True)
    a = ap.parse_args(argv)
    cfg = ContactValidateConfig()
    world = None
    for path in a.instance:
        oracle = _json.load(open(path))
        tro = _json.load(open(path[: -len(".oracle.json")] + "-tro_state.json"))
        sit = replay_situation(oracle, a.kind, tro)
        if world is None:
            world = KitContactWorld(cfg, sit["arm"])
        job = {"id": os.path.basename(a.model_dir), "model_dir": a.model_dir}
        mlib, _rid, geom = CL.load_object(None, job)
        obj = world.load({"id": job["id"], "category": mlib.data.get("category"), "model": mlib.data.get("model"), "scale": mlib.data.get("scale"),
                          "rest_hint_wxyz": None}, a.model_dir)
        obj._isg_geom = geom
        try:
            res = world.replay(sit, obj, geom)
        except Exception as e:  # noqa: BLE001
            import traceback

            res = {"error": repr(e), "tb": traceback.format_exc()[-1500:]}
        finally:
            world.unload(obj)
        print("GB_CONTACT_REPLAY " + _json.dumps({"instance": os.path.basename(path), **res}, default=str), flush=True)
    import omnigibson as og

    og.shutdown()


def main(argv=None):
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "replay":
        replay_cli(argv[1:])
        return
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
