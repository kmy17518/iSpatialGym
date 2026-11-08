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

   * a PUSH WITH A TASK GOAL (``goal_target``: every touching / axial push entry and replay since run D, calibrated on the replays of the
     demos) passes when the TASK's goal check passes (:func:`judge_goal`): the object within the goal's position tolerance (touching 0.03 m;
     axial none) and orientation tolerance (touching 0.2, axial 0.1 rad: the full rotation, so a topple or a slip fails it), a touching
     push's target touching its partner (the stop's face <= ``touching_gap_max_m`` + ``stop_slack_m`` away: the BDDL ``touching``), the
     oracle's action ran to its end with a clean postcondition, no non-finger link loaded the slab. The world carries what the demos have:
     the goal is the object's TRO pose target (``SimState.pose_target``: the oracle's closed-loop push stops at it / re-pushes toward it),
     and a touching push meets a kinematic partner stand-in (``stop_size_m``) at the place gap (``push.stop_gap_m``). The stroke measures
     below stay in the record as diagnostics (``diagnostic_reasons``);
   * any other PUSH passes when the object moved along the push direction by >= ``stroke - stop clearance - push_short_tol_m`` (the stroke
     minus the fingertips' start gap), it did not topple (tilt change <= ``topple_tilt_rad``), the hand did not slip off (sideways drift <=
     ``push_lateral_max_m``, yaw change <= ``push_yaw_max_rad``, the fingertips within ``contact_lost_m`` of the object's face at the goal),
     no non-finger robot link loaded the slab beyond ``non_finger_contact_N``, the oracle's postcondition is clean and the action ran to its
     end (``max_steps``);
   * a BRACE passes when the oracle's brace reached its contact (postcondition), and under the pusher's load the partner moved along the
     load by <= ``brace_hold_max_m`` (the 2 mm touch stop + 4 mm back-off + slack), did not topple, the brace hand stayed within
     ``brace_eef_drift_m`` of its contact pose, and no non-finger link loaded the slab.

Trials: ``trials_per_entry`` (1: the physics is deterministic at one configuration; the yaw slots of a class are separate entries), at most
``max_entries`` entries per model and kind in library order - or STRATIFIED (owner A6, ``--per-group-valid 3``: per (yaw x axial end x
spin) group until 3 are valid or ``per_group_max`` were tried, :class:`GroupQuota`; the maps then read the VALID entries only). Timing is recorded per model, rest and kind (``validation_timing``: setup, settle,
plan, trials, total). The thresholds are PROVISIONAL (the recorded measures are for calibrating them).

    python -m ispatialgym.grasping.validate_contacts --root ROOT --kind push|brace --glib GLIB [--jobs J.json] [--arm right|left] [--shard i/n] [--only ID ...]
    python -m ispatialgym.grasping.validate_contacts replay --instance <...oracle.json ...> --kind push|brace --model-dir <v2 model dir>
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.grasping import schema as S

TRIAL_SCHEMA = "isg-contact-trial/1"
#: i7-contactfu 2026-11-08 (owner A2): the touching push's partner stand-in is KINEMATIC - OmniGibson honours ``kinematic_only`` only with a
#: fixed base (``KitContactWorld._stop_obj``)
STOP_STANDIN_KINEMATIC = {"fixed_base": True, "kinematic_only": True}
#: i7-brace 2026-11-08 (owner 3.3): the brace load's pusher is built the same way (it was ``kinematic_only`` without a fixed base: dynamic)
PUSHER_KINEMATIC = STOP_STANDIN_KINEMATIC


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
    # i7-brace (owner 3.4): 26 braced world_vertical demos ($OFH/out/*/alignment-world_vertical_0*, 2026-11-08): the brace contact 0.75-0.88 m
    # ahead of the manipulation stance (square to the wall, stand-off 0.80-0.90), 0.05-0.35 m to the BRACE ARM's side (left +, right -:
    # ``mirror_right``), trunks (0 | 0.25, 0 .. -0.12)
    "wall_brace": {"ahead_m": (0.76, 0.81, 0.86, 0.73), "lateral_m": (0.05, 0.10, 0.20, 0.30), "mirror_right": True,
                   "trunks": ((0.25, 0.0), (0.0, 0.0), (0.0, -0.1), (0.25, -0.1), (0.5, 0.0)),
                   "source": "26 braced world_vertical demos: brace contact 0.75-0.88 m ahead, 0.05-0.35 m to the brace arm's side"},
}
#: i7-brace (owner 3.4): the families validated in the WALL world (:meth:`KitContactWorld.hang`: the target hung on its COM hinge at a
#: kinematic wall, no slab) - each entry with its own brace arm (``wall.brace_arm``: a validation run of one arm skips the other's entries)
WALL_FAMILIES = ("wall_brace",)
#: the hinge height (m above the floor) of a wall library without ``validate.support_z`` (world_vertical's hang_height_m)
WALL_HANG_DEFAULT_M = 1.45


@dataclass
class ContactValidateConfig:
    work_arm: str = "right"
    brace_arm: str = "left"
    trials_per_entry: int = 1
    max_entries: int = 48
    #: DIAGNOSTIC (run E, owner question 2): validate the entries the sampler's hand_overlap screen removes INSTEAD of the live ones - is the
    #: screen over-strict physically? Their verdicts carry ``diagnostic``; the screen itself is unchanged (a library run with it goes to its own root)
    diagnose_screened: bool = False
    #: keep the verdicts an earlier run wrote (``validation.verdict`` set): those entries are neither re-run nor counted against
    #: ``max_entries`` - a re-run in a new order adds its sample to the old one (i7-contact 2026-11-07: the axial (end, spin) groups)
    skip_validated: bool = False
    #: owner item 3 switch (default OFF; the owner decides): the sampler's hand_overlap screen is a ranking hint - the screened entries are
    #: validated and counted after the live ones (``entry_order``) instead of being marked ``screened`` (the sampler: ISPATIALGYM_HAND_OVERLAP_RANK)
    screen_as_rank: bool = False
    #: owner A6 (2026-11-08, i7-contactfu): STRATIFIED validation - per (yaw slot x axial end x spin) group (:func:`entry_group`) the live
    #: entries are validated in library order (round-robin over the groups) until ``per_group_valid`` are physics-valid or ``per_group_max``
    #: were tried; 0 = off (the ``max_entries`` cap as before). A group with fewer than ``per_group_min`` valid entries is reported ``short``.
    #: Without ``skip_validated`` the run is FRESH: an entry it does not run loses an earlier verdict (kept as ``validation_prev``), so a map
    #: built from the library reads only this run's verdicts
    per_group_valid: int = 0
    #: i7-brace 2026-11-08 (owner 3.3): RE-validate exactly the entries an earlier run validated (verdict valid / invalid / unreachable; the
    #: earlier record kept as ``validation_prev``), no cap - the per-entry verdict changes of a validator fix; other entries untouched
    revalidate: bool = False
    per_group_min: int = 2
    per_group_max: int = 8
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
    #: the touching push's partner stand-in (thickness along the push, width, height): a partner-sized face the target is pushed into
    stop_size_m: Tuple[float, float, float] = (0.04, 0.20, 0.30)
    #: the stop's near face this far past the goal (the push's stop-at-goal ends the push first; the stop catches an overshoot)
    stop_slack_m: float = 0.001
    #: a touching push's target touches its partner: the gap to the stop's face at the end <= this (the touching predicate's geometric
    #: branch, ``sampling.predicates.TOUCHING_GAP_MAX_M``) plus ``stop_slack_m``
    touching_gap_max_m: float = 0.003
    brace_pusher_gap_m: float = 0.01
    brace_press_m: float = 0.01
    brace_pusher_speed_m_s: float = 0.02
    brace_hold_steps: int = 30
    brace_hold_max_m: float = 0.012
    brace_eef_drift_m: float = 0.01
    # i7-brace (owner 3.4): the wall world (WALL_FAMILIES) - the kinematic wall box (thickness along its normal, width, height from the floor),
    # the hang's settle steps on the hinge and the drift it may show
    wall_size_m: Tuple[float, float, float] = (0.10, 3.0, 2.6)
    wall_hang_steps: int = 20
    wall_hang_drift_max_rad: float = 0.01
    #: the braced point must move at least this far under the load (else ``not_loaded``: the pusher never reached the target)
    wall_loaded_min_m: float = 0.001
    #: the brace fingers must press the target at least this hard during the load (else ``not_braced``: it went past the fist)
    wall_braced_min_N: float = 1.0
    #: the braced point may travel at most the stop's gap (``wall.brace_offset_m``) + the hand's own displacement + this (``slipped_past``)
    wall_slip_m: float = 0.004
    #: the wall load's contact: the target turned this far (rad) under the sweep
    wall_contact_turn_rad: float = 0.002
    wall_pusher_travel_max_m: float = 0.06
    # i7-brace 5w.3 (owner 20261108h, DIAGNOSIS ONLY: :func:`brace_load_switches`, default off) - the touching / axial brace load measured
    #: the load point counts as met once it moved this far along the load (the measured first contact)
    brace_contact_m: float = 0.0005
    #: MEASURED mode: the pusher may travel at most this far (gap + press + the contact found late)
    brace_pusher_travel_max_m: float = 0.06


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


def oracle_ctx(variant: Optional[str] = None) -> Dict[str, Any]:
    """The oracle action's ``ctx`` at its start: the variant as the sidecar's (``execute.variant_generation`` reads its customs
    ``generation``; the oracle's other sidecar reads are ``getattr(..., None)``), else empty."""
    return {} if not variant else {"sidecar": SimpleNamespace(variant=str(variant))}


#: the families (and replayed variants) whose demos run on the task's vertical COM hinge (``AlignmentAxial.after_settle``)
AXIAL_FAMILIES = ("axial_push", "axial_brace")


def axial_hinged(entry: Dict[str, Any], variant: Optional[str]) -> bool:
    """Whether a trial runs on the axial task's COM hinge: an axial library entry, or a replayed record of an alignment-axial variant."""
    return entry.get("family") in AXIAL_FAMILIES or str(variant or "").startswith("alignment-axial")


def axial_hinge_spec(variant: Optional[str], mass_kg: float, footprint_long_short: Tuple[float, float]) -> Dict[str, Any]:
    """The hinge ``AlignmentAxial.after_settle`` installs, from the variant's customs ``generation`` over the alignment-axial defaults:
    ``sink_m`` (``hinge_sink_m``: the anchor that far below the support top, on the COM's vertical) and the resistance
    (``AlignmentAxial.hinge_resistance``: None - frictionless - unless the variant sets one of ``HINGE_RESISTANCE_KEYS``; then
    ``constraints.hinge_resistance`` of the mass over the footprint, the world vertical's defaults for the keys left out)."""
    from ispatialgym.oracle import execute as EX
    from ispatialgym.sampling import constraints as CN
    from ispatialgym.sampling import generation_defaults as GD

    keys = ("hinge_stop_decel_rad_s2", "hinge_viscous_tau_s", "hinge_brake_omega_rad_s")  # alignment_axial.HINGE_RESISTANCE_KEYS
    gen = dict(GD.defaults_for("alignment-axial"))
    gen.update(EX.variant_generation(oracle_ctx(variant)) if variant else {})
    vals = {k: gen.get(k) for k in keys}
    res = None
    if any(v is not None for v in vals.values()):
        d = dict(zip(keys, (CN.HINGE_STOP_DECEL_RAD_S2, CN.HINGE_VISCOUS_TAU_S, CN.HINGE_BRAKE_OMEGA_RAD_S)))
        res = CN.hinge_resistance(float(mass_kg), tuple(float(v) for v in footprint_long_short), decel=float(vals[keys[0]] if vals[keys[0]] is not None else d[keys[0]]),
                                  tau=float(vals[keys[1]] if vals[keys[1]] is not None else d[keys[1]]),
                                  omega_brake=float(vals[keys[2]] if vals[keys[2]] is not None else d[keys[2]]))
    return {"sink_m": float(gen.get("hinge_sink_m", 0.001)), "resistance": res}


def run_oracle(action, motions, state, step, max_steps: int, variant: Optional[str] = None) -> Dict[str, Any]:
    """``execute.run_action`` of ``action`` stepping the world with ``step(a)`` (at most ``max_steps``), then ``execute.check_postcondition``:
    ``{"steps", "completed", "postcondition", "ctx"}``. ``variant``: the customs variant whose ``generation`` the oracle reads
    (``execute.variant_generation`` via ``ctx["sidecar"].variant``: e.g. ``push_arc_stall_ends_arc`` of alignment-axial-board_game-
    breakfast_table, which the demos ran with - run D's axial_01 replay ran without it)."""
    from ispatialgym.oracle import execute as EX

    targets = action.targets(None, state)
    ctx = oracle_ctx(variant)
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
               run: Dict[str, Any], other_slab_N: float = 0.0, tip_gap_m: float = None, goal: Optional[Dict[str, Any]] = None,
               touch_gap_m: Optional[float] = None) -> Dict[str, Any]:
    """The push acceptance (module docstring): ``{"verdict", "reasons", "measures"}``. ``tip_gap_m``: the fingertips' gap to the face at the
    push's farthest object position (the trial's per-step :func:`tip_gap`; the oracle backs the hand off after the push), else at the end.
    An entry with a task goal (``goal_target``) is judged by :func:`judge_goal` (``goal``: the trial's, else :func:`goal_pose` at ``T_obj0``;
    ``touch_gap_m``: a touching push's end gap to the stop); the stroke / arc measures stay as diagnostics."""
    if entry.get("goal_target"):
        diag = (judge_arc_push(entry, T_obj0, T_obj1, cfg, run, other_slab_N) if entry.get("arc") is not None else
                judge_push(dict(entry, goal_target=None), T_obj0, T_obj1, T_eef1, pts_obj, cfg, run, other_slab_N, tip_gap_m))
        return judge_goal(entry, T_obj0, T_obj1, cfg, run, other_slab_N, extra_measures=dict(diag["measures"], diagnostic_reasons=diag["reasons"]),
                          goal=goal, touch_gap_m=touch_gap_m)
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


def goal_pose(entry: Dict[str, Any], T_obj0: np.ndarray) -> Optional[Dict[str, Any]]:
    """The task goal an entry's push serves (``entry["goal_target"]``), at the object's start pose ``T_obj0``: ``{"T": the goal pose (4x4),
    "position_tolerance" (None: orientation only), "orientation_tolerance"}`` - a touching push leaves the target touching its partner (the
    start pose moved ``along_push_m`` along the push), an axial push turns it ``turn_rad`` about the vertical (orientation only)."""
    g = entry.get("goal_target")
    if not g:
        return None
    T = np.array(T_obj0, dtype=float)
    if g["kind"] == "translate":
        d = T_obj0[:3, :3] @ np.asarray(entry["push"]["direction"], dtype=float)
        d[2] = 0.0
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        T[:3, 3] = T_obj0[:3, 3] + d * float(g["along_push_m"])
    elif g["kind"] == "turn":
        a = float(g["turn_rad"])
        T[:3, :3] = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]) @ T_obj0[:3, :3]
    else:
        raise ValueError(f"goal_target kind {g['kind']!r}")
    return {"T": T, "position_tolerance": g.get("position_tolerance"), "orientation_tolerance": float(g["orientation_tolerance"]), "kind": g["kind"]}


def goal_pose_target(name: str, goal: Dict[str, Any]):
    """The goal as the oracle's ``eval.targets.PoseTarget`` (what ``SimState.pose_target`` hands its closed-loop pushes: the re-push, the
    goal follow, the arc's residual), from ``tro_writer.pose_target``'s record."""
    from ispatialgym.eval.targets import PoseTarget
    from ispatialgym.sampling.tro_writer import pose_target

    T = goal["T"]
    pos = None if goal["position_tolerance"] is None else [float(v) for v in T[:3, 3]]
    return PoseTarget.from_dict(name, pose_target(position=pos, orientation=[float(v) for v in S.mat_to_quat(T[:3, :3])],
                                                  position_tolerance=goal["position_tolerance"], orientation_tolerance=goal["orientation_tolerance"]))


def _quat_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(math.acos(max(-1.0, min(1.0, (float(np.trace(Ra.T @ Rb)) - 1.0) / 2.0))))


def judge_goal(entry: Dict[str, Any], T_obj0: np.ndarray, T_obj1: np.ndarray, cfg: ContactValidateConfig, run: Dict[str, Any], other_slab_N: float = 0.0,
               extra_measures: Optional[Dict[str, Any]] = None, goal: Optional[Dict[str, Any]] = None, touch_gap_m: Optional[float] = None) -> Dict[str, Any]:
    """A push with a task goal (``goal_target``) passes when the TASK's goal check passes - the object within the goal's position tolerance
    (when it has one) and orientation tolerance (the full rotation: a topple or a slip fails it; the TRO pose target the oracle aims at),
    a touching push's target touching its partner (``touch_gap_m`` <= ``touching_gap_max_m`` + ``stop_slack_m``: the task's BDDL goal) - the
    oracle's action ran to its end with a clean postcondition, and no non-finger link loaded the slab: the validator is no stricter than
    what makes a demo succeed. The straight / arc measures stay as diagnostics (``extra_measures``)."""
    goal = goal if goal is not None else goal_pose(entry, T_obj0)
    pos_err = float(np.linalg.norm(T_obj1[:3, 3] - goal["T"][:3, 3]))
    ori_err = _quat_angle(T_obj1[:3, :3], goal["T"][:3, :3])
    reasons = []
    if not run.get("completed", True):
        reasons.append("incomplete")
    if run.get("postcondition"):
        reasons.append(f"oracle:{run['postcondition']}")
    if goal["position_tolerance"] is not None and pos_err > float(goal["position_tolerance"]):
        reasons.append("goal_position")
    if ori_err > goal["orientation_tolerance"]:
        reasons.append("goal_orientation")
    touching = (entry.get("push") or {}).get("stop_gap_m") is not None and goal["kind"] == "translate"
    if touching and (touch_gap_m is None or float(touch_gap_m) > cfg.touching_gap_max_m + cfg.stop_slack_m):
        reasons.append("not_touching")
    if other_slab_N > cfg.non_finger_contact_N:
        reasons.append("non_finger_contact")
    meas = {"touch_gap_m": None if touch_gap_m is None else round(float(touch_gap_m), 5),"goal_position_err_m": round(pos_err, 5), "goal_orientation_err_rad": round(ori_err, 5), "position_tolerance": goal["position_tolerance"],
            "orientation_tolerance": goal["orientation_tolerance"], "tilt_change_rad": round(_tilt(T_obj0[:3, :3], T_obj1[:3, :3]), 5),
            "other_slab_N": round(float(other_slab_N), 4), **(extra_measures or {})}
    return {"verdict": "invalid" if reasons else "valid", "reasons": reasons, "measures": meas}


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


def stop_pose(entry: Dict[str, Any], T_obj: np.ndarray, pts_obj: np.ndarray, cfg: ContactValidateConfig) -> Optional[Dict[str, Any]]:
    """Where the touching push's partner stand-in stands (``push.stop_gap_m`` set): its near face ``stop_gap_m`` beyond the object's leading
    face along the push, on the object's centre line, its bottom 1 mm above the support, turned square to the push. None without a gap."""
    gap = (entry.get("push") or {}).get("stop_gap_m")
    if gap is None:
        return None
    d = T_obj[:3, :3] @ np.asarray(entry["push"]["direction"], dtype=float)
    d[2] = 0.0
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    pts = pts_obj @ T_obj[:3, :3].T + T_obj[:3, 3]
    lead = float((pts[:, :2] @ d[:2]).max())
    c = pts.mean(axis=0)
    sx, _sy, sz = cfg.stop_size_m
    face = lead + float(gap) + float(cfg.stop_slack_m)
    centre = np.array([c[0], c[1], float(pts[:, 2].min()) + 0.5 * float(sz) + 0.001]) + d * (face - float(c[:2] @ d[:2]) + 0.5 * float(sx))
    yaw = math.atan2(d[1], d[0])
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])  # its x (thickness) along the push
    return {"centre": centre, "R": R, "gap_m": round(float(gap), 5), "d": d, "face_along_m": face}


def touch_gap(stop: Optional[Dict[str, Any]], T_obj: np.ndarray, pts_obj: np.ndarray) -> Optional[float]:
    """The gap between the object's leading face (``T_obj``) and the stop's near face (:func:`stop_pose`) along the push (<= 0: touching or
    pressed into it); None without a stop."""
    if stop is None:
        return None
    pts = pts_obj @ T_obj[:3, :3].T + T_obj[:3, 3]
    return float(stop["face_along_m"] - (pts[:, :2] @ np.asarray(stop["d"])[:2]).max())


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


def wall_entry(entry: Dict[str, Any]) -> bool:
    """i7-brace: a wall-world entry (:data:`WALL_FAMILIES`)."""
    return entry.get("family") in WALL_FAMILIES


def brace_load_switches() -> Dict[str, bool]:
    """i7-brace 5w.3 (owner 20261108h: diagnose and COUNT only - no default, no rebuild) - the touching / axial brace load (not the wall's):
    ``ISPATIALGYM_GC_BRACE_LOAD_DIAG=1`` records the load's facts in the trial (``debug.load_diag``: the object asleep at the load's start /
    per step, the sleep assist's sleeps / wakes, the load point's travel along the load, the measured first contact vs the computed face);
    ``ISPATIALGYM_GC_BRACE_LOAD_WAKE=1`` wakes the braced object at every pusher step (the wall fix 220ff7db: the sleep assist sleeps a still
    partner while the brace stands and a teleported kinematic pusher does not wake it); ``ISPATIALGYM_GC_BRACE_LOAD_MEASURED=1`` stops the
    pusher ``brace_press_m`` after the MEASURED first contact (the load point moved ``brace_contact_m``; the wall fix fae1e4d8: the computed
    face from the hull's vertices ignores the contact offsets), extending the path up to ``brace_pusher_travel_max_m``. Either correction
    implies the record. All unset = today's load, bit-identical."""
    on = {k: os.environ.get(f"ISPATIALGYM_GC_BRACE_LOAD_{k.upper()}", "0") == "1" for k in ("diag", "wake", "measured")}
    on["diag"] = on["diag"] or on["wake"] or on["measured"]
    return on


def _asleep(obj) -> Optional[bool]:
    try:
        a = getattr(obj, "is_asleep", None)
        return bool(a() if callable(a) else a)
    except Exception:  # noqa: BLE001
        return None


def brace_load_point(entry: Dict[str, Any], T_obj: np.ndarray) -> np.ndarray:
    """The world point the brace load drives (:func:`pusher_path`'s line: ``load.point_object`` - the axial push's arriving end - else the
    object's root): on the axial task's COM hinge the root does not translate, the arriving end does."""
    po = (entry.get("load") or {}).get("point_object")
    return T_obj[:3, :3] @ np.asarray(po, dtype=float) + T_obj[:3, 3] if po is not None else T_obj[:3, 3].copy()


def measured_load_stop(travels: Sequence[float], moved: Sequence[float], press_m: float, contact_m: float) -> Tuple[Optional[int], Optional[float]]:
    """(stop index, contact travel): the first step whose load point moved >= ``contact_m`` along the load is the measured contact, its pusher
    travel less that motion the contact travel; the stop is the first step whose travel >= contact travel + ``press_m``. (None, None) before
    a contact; (None, contact) before the stop."""
    c = next((i for i, m in enumerate(moved) if m >= contact_m), None)
    if c is None:
        return None, None
    s_c = float(travels[c]) - float(moved[c])
    k = next((i for i in range(c, len(travels)) if travels[i] >= s_c + press_m - 1e-9), None)
    return k, s_c


def wall_arm_entries(entries: Sequence[Dict[str, Any]], arm: str) -> List[Dict[str, Any]]:
    """The entries a validation run of ``arm`` validates: every non-wall entry, and the wall entries whose brace arm (``wall.brace_arm``) it is."""
    return [e for e in entries if not wall_entry(e) or str((e.get("wall") or {}).get("brace_arm") or "left") == arm]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def hull_edge_points(pts: np.ndarray, per_edge: int = 24) -> np.ndarray:
    """The convex hull's vertices plus ``per_edge`` points along each of its edges: the first contact of a flat face sweeping the hull lies on an
    edge between vertices (2026-11-08 run 4: a box-like hull had only its two corner vertices within the pusher's width, the bottom edge
    between them stood up to 7 mm in front of the face plane - the load over-turned the target by 0.02 rad)."""
    from scipy.spatial import ConvexHull

    P = np.asarray(pts, dtype=float).reshape(-1, 3)
    try:
        hull = ConvexHull(P)
    except Exception:  # noqa: BLE001 - degenerate (flat) point sets: the vertices alone
        return P
    edges = set()
    for tri in hull.simplices:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.add((min(a, b), max(a, b)))
    E = np.array(sorted(edges), dtype=int)
    w = np.linspace(0.0, 1.0, int(per_edge) + 2)[1:-1]
    seg = P[E[:, 0]][:, None, :] * (1.0 - w)[None, :, None] + P[E[:, 1]][:, None, :] * w[None, :, None]
    return np.concatenate([P[hull.vertices], seg.reshape(-1, 3)])


def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
    k = _unit(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def wall_pusher_path(entry: Dict[str, Any], T_obj: np.ndarray, pts_obj: np.ndarray, cfg: ContactValidateConfig, dt: float
                     ) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    """The wall brace load's pusher (i7-brace, owner 3.4): ``(centres, xyzw quaternions - one per control step -, facts)``. The leveling push
    carried on past level: the pusher's face meets the PUSHED corner (``load.point_object``) square to its tangent (``load.direction``, in the
    wall plane) and is carried along the corner's ARC about the hinge (``wall.hinge_object``, the wall normal), from ``brace_pusher_gap_m``
    before the material to the end of the sweep (``wall_pusher_travel_max_m``); the trial (:func:`wall_load_stop`) stops it ``phi_press`` = ``brace_press_m`` / r_brace
    after the target first moves - the braced point (``load.brace_point_object``) driven ``brace_press_m`` past level unbraced, the touching
    / axial calibration (their object travels ``brace_press_m`` past first contact); the contact is MEASURED (run 5: the target turned from
    ~7 mm before the computed face - the physics' contact offsets - 0.09 rad for 0.056). A straight path along the tangent over-turned the target
    (2026-11-08 run 3: the contact slid inward along the flat face, every trial 1.59 cm at the braced point for 1 cm meant), so the arc makes
    the travel exact whatever the brace does. The material it meets: the hull points within the box's half-width ACROSS the line in the wall
    plane (any offset along the normal: the corner is on the mid-plane, the hull's vertices off it); none raises (no load placed: the trial
    errors, never a vacuous pass). The box: local z along the travel (5 cm), y across it in the wall plane (3 cm: the push fist's width),
    x along the wall normal (8 cm)."""
    load = entry.get("load") or {}
    wall = entry.get("wall") or {}
    R, t = T_obj[:3, :3], T_obj[:3, 3]
    d = _unit(R @ np.asarray(load["direction"], dtype=float))
    P = R @ np.asarray(load["point_object"], dtype=float) + t
    n = _unit(R @ np.asarray(wall.get("normal_object") or [1.0, 0.0, 0.0], dtype=float))
    H = R @ np.asarray(wall.get("hinge_object") or [0.0, 0.0, 0.0], dtype=float) + t
    pts = hull_edge_points(pts_obj) @ R.T + t
    rel = pts - P
    along = rel @ d
    y = _unit(np.cross(d, n))
    r_line = 0.5 * float(cfg.brace_pusher_size_m[0]) + 0.002
    near = np.abs(rel @ y) <= r_line
    if not near.any():
        raise ValueError(f"wall load: no hull point within {r_line:.3f} m across the pushed corner's line (no load can be placed)")
    face = float(along[near].min())
    half = 0.5 * float(cfg.brace_pusher_size_m[2])
    F0 = P + d * face  # the material's face point on the corner's line
    rF = F0 - H
    rF = rF - n * float(rF @ n)
    r_F = max(float(np.linalg.norm(rF)), 1e-3)
    axis = n if float(np.cross(n, rF) @ d) >= 0 else -n  # turning about +axis carries the corner along d
    r_B = float(load.get("r_brace_m") or r_F)
    phi0 = -cfg.brace_pusher_gap_m / r_F
    phi_press = cfg.brace_press_m / max(r_B, 0.05)
    phi1 = (cfg.brace_pusher_gap_m + cfg.wall_pusher_travel_max_m) / r_F + phi0  # the sweep's end; the trial stops it phi_press after contact
    k = max(1, int(math.ceil((phi1 - phi0) * r_F / max(cfg.brace_pusher_speed_m_s * dt, 1e-6))))
    R0 = np.stack([np.cross(y, d), y, d], axis=1)
    centres, quats, phis = [], [], []
    for i in range(k + 1):
        phi = phi0 + (phi1 - phi0) * i / k
        Rp = _rot(axis, phi)
        Fi = H + Rp @ (F0 - H)
        centres.append(Fi - (Rp @ d) * half)
        quats.append(np.asarray(S.mat_to_quat(Rp @ R0), dtype=float))
        phis.append(phi)
    facts = {"start": [round(float(v), 4) for v in centres[0]], "direction": [round(float(v), 4) for v in d], "face_along_m": round(face, 5),
             "arc_rad": [round(phi0, 5), round(phi1, 5)], "phi_press_rad": round(phi_press, 6), "phis": phis, "r_face_m": round(r_F, 5), "steps": k,
             "axis": [float(v) for v in axis],
             "n_line_points": int(near.sum())}
    return centres, quats, facts


def signed_turn(R0: np.ndarray, R1: np.ndarray, axis) -> float:
    """The rotation from ``R0`` to ``R1`` about ``axis`` (rad, signed: + = the sense ``axis`` turns; the rotation vector's component)."""
    Rr = np.asarray(R1, dtype=float) @ np.asarray(R0, dtype=float).T
    c = max(-1.0, min(1.0, (float(np.trace(Rr)) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return 0.0
    w = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]])
    nw = float(np.linalg.norm(w))
    if nw < 1e-12:  # a half turn: the axis from the symmetric part
        return ang
    return ang * float((w / nw) @ _unit(axis))


def wall_load_stop(phis: Sequence[float], turns: Sequence[float], phi_press: float, contact_rad: float) -> Optional[int]:
    """i7-brace: the step of the wall load's sweep at which the pusher stops - the first step whose angle reaches the contact angle (the
    pusher's angle at the first step the target had turned >= ``contact_rad`` PAST LEVEL - ``turns`` signed, :func:`signed_turn`) +
    ``phi_press``; None while that is not reached (``phis`` / ``turns`` per step so far)."""
    c = next((i for i, t in enumerate(turns) if t >= contact_rad), None)
    if c is None:
        return None
    stop = next((i for i in range(c, len(phis)) if phis[i] >= phis[c] + phi_press - 1e-12), None)
    return stop


def judge_wall_brace(entry: Dict[str, Any], T_obj0: np.ndarray, T_obj1: np.ndarray, T_eef0: np.ndarray, T_eef1: np.ndarray, cfg: ContactValidateConfig,
                     run: Dict[str, Any], other_N: float = 0.0, finger_N: Optional[float] = None) -> Dict[str, Any]:
    """The wall brace acceptance (i7-brace, owner 3.4; ``*0`` the brace standing, ``*1`` after the load): the oracle's brace ran to its end
    with a clean postcondition; under the load the BRACED point moved along its own load direction (``load.brace_direction``: past level)
    by <= ``brace_hold_max_m`` (``not_held``) and >= ``wall_loaded_min_m`` (``not_loaded``: no load reached it), the brace fingers pressed the target >= ``wall_braced_min_N`` during it (``not_braced``: the
    braced point went past the fist), it travelled at most the stop's gap + the hand's displacement + ``wall_slip_m`` (``slipped_past``: it
    slid past the fist), the hinge axis did not turn off the wall normal by more than ``topple_tilt_rad``
    (``topple``: off its hinge), the brace hand stayed within ``brace_eef_drift_m`` (``hand_pushed``), and no non-finger robot link loaded the
    wall beyond ``non_finger_contact_N``. Measures: the turn about the wall normal (rad, + past level) and the braced point's motion."""
    load = entry.get("load") or {}
    wall = entry.get("wall") or {}
    R0, R1 = T_obj0[:3, :3], T_obj1[:3, :3]
    Bo = np.asarray(load.get("brace_point_object") or wall.get("B_object"), dtype=float)
    B0, B1 = R0 @ Bo + T_obj0[:3, 3], R1 @ Bo + T_obj1[:3, 3]
    vB = _unit(R0 @ np.asarray(load.get("brace_direction"), dtype=float))
    n = _unit(R0 @ np.asarray(wall.get("normal_object") or [1.0, 0.0, 0.0], dtype=float))
    moved = float((B1 - B0) @ vB)
    Rr = R1 @ R0.T
    twist = float(math.acos(max(-1.0, min(1.0, float(n @ (Rr @ n))))))
    # the signed turn about n: the braced point's lever (B - hinge) turned toward vB is + (past level)
    H = R0 @ np.asarray(wall.get("hinge_object") or [0.0, 0.0, 0.0], dtype=float) + T_obj0[:3, 3]
    r0 = B0 - H
    r0 = r0 - n * float(r0 @ n)
    r1 = Rr @ r0
    turn = math.atan2(float(np.cross(r0, r1) @ n), float(r0 @ r1))
    turn *= 1.0 if float(np.cross(n, r0) @ vB) >= 0 else -1.0
    drift = float(np.linalg.norm(T_eef1[:3, 3] - T_eef0[:3, 3]))
    reasons = []
    if not run.get("completed", True):
        reasons.append("incomplete")
    if run.get("postcondition"):
        reasons.append(f"oracle:{run['postcondition']}")
    if moved > cfg.brace_hold_max_m:
        reasons.append("not_held")
    if moved < cfg.wall_loaded_min_m:
        reasons.append("not_loaded")  # the load never reached the target (the stop's gap alone is >= 1 mm): no verdict on the brace
    gap_b = float(wall.get("brace_offset_m", 0.005))
    slip = moved - drift - gap_b
    if slip > cfg.wall_slip_m:
        reasons.append("slipped_past")  # the braced point went farther than the stop's gap + the hand's own displacement: it slid past the fist
    if finger_N is not None and finger_N < cfg.wall_braced_min_N:
        reasons.append("not_braced")  # the braced point went past and never met the fist (2026-11-08 run 3: the 25 deg wrists, 0.1 mm drift)
    if twist > cfg.topple_tilt_rad:
        reasons.append("topple")
    if drift > cfg.brace_eef_drift_m:
        reasons.append("hand_pushed")
    if other_N > cfg.non_finger_contact_N:
        reasons.append("non_finger_contact")
    return {"verdict": "invalid" if reasons else "valid", "reasons": reasons,
            "measures": {"brace_point_moved_m": round(moved, 5), "turn_past_level_rad": round(turn, 5), "hinge_axis_twist_rad": round(twist, 5),
                         "finger_obj_N": None if finger_N is None else round(float(finger_N), 3), "slip_m": round(slip, 5),
                         "eef_drift_m": round(drift, 5), "other_wall_N": round(float(other_N), 4)}}


def judge_wall_standing(run: Dict[str, Any], cfg: ContactValidateConfig, other_N: float = 0.0) -> Dict[str, Any]:
    """The wall brace BEFORE the load (i7-brace: "valid before / after the load"): the oracle's brace reached its stop (ran to its end, clean
    postcondition) with no non-finger robot link loading the wall."""
    reasons = []
    if not run.get("completed", True):
        reasons.append("incomplete")
    if run.get("postcondition"):
        reasons.append(f"oracle:{run['postcondition']}")
    if other_N > cfg.non_finger_contact_N:
        reasons.append("non_finger_contact")
    return {"verdict": "invalid" if reasons else "valid", "reasons": reasons, "other_wall_N": round(float(other_N), 4)}


# ------------------------------------------------------------------------------------------------ one model (any world with the KitContactWorld surface)


def axial_group(entry: Dict[str, Any]) -> str:
    """An axial entry's (pushed end, turn sense) group - the class label's ``+end+spin`` / ``+end-spin`` / ``-end+spin`` / ``-end-spin``
    prefix (``contact_library.build_axial_push`` / ``build_axial_brace``); ``""`` for every other family."""
    m = re.match(r"^([+-]end[+-]spin)(_opp_)?", str(entry.get("class", "")))
    # i7-brace (owner 3.4): an opposite-end entry of the clear-turn build is its own group (the stratified quota of the current-end
    # candidates, first in library order, otherwise never reaches it)
    return (m.group(1) + ("_opp" if m.group(2) else "")) if m else ""


def entry_group(entry: Dict[str, Any]) -> Tuple[float, str]:
    """The stratification group of an entry (owner A6): its yaw slot and, for an axial entry, its (pushed end, turn sense)."""
    return round(float(entry.get("yaw_deg", 0.0)), 4), axial_group(entry)


def group_key_str(g: Tuple[float, str]) -> str:
    return f"y{g[0]:g}{g[1]}"


class GroupQuota:
    """Owner A6 (i7-contactfu 2026-11-08): the stratified validation's schedule over the live entries (pure; :func:`validate_library` runs
    it). Groups (:func:`entry_group`) in their first appearance in library order; within a group the library order (the sampler's). Each
    round takes the next entry of every group that still needs one (fewer than ``per_valid`` valid and fewer than ``per_max`` tried);
    ``kept`` verdicts (``skip_validated``) count first and are not re-run. :meth:`next_round` -> the entries to run now;
    :meth:`done(entry, verdict)` books one."""

    def __init__(self, entries: Sequence[Dict[str, Any]], per_valid: int, per_max: int, keep_verdicts: bool = False):
        self.per_valid, self.per_max = int(per_valid), int(per_max)
        self.groups: Dict[Tuple[float, str], List[Dict[str, Any]]] = {}
        for e in entries:
            self.groups.setdefault(entry_group(e), []).append(e)
        self.state = {g: {"valid": 0, "tried": 0, "kept": 0} for g in self.groups}
        self.pos = {g: 0 for g in self.groups}
        self.kept_ids = set()
        if keep_verdicts:
            for g, es in self.groups.items():
                for e in es:
                    v = (e.get("validation") or {}).get("verdict")
                    if v in ("valid", "invalid", "unreachable"):
                        self.kept_ids.add(id(e))
                        self._book(g, v, kept=True)

    def _book(self, g, verdict: str, kept: bool = False) -> None:
        st = self.state[g]
        st["tried"] += 1
        st["kept"] += int(kept)
        st["valid"] += int(verdict == "valid")

    def needs(self, g) -> bool:
        st = self.state[g]
        return st["valid"] < self.per_valid and st["tried"] < self.per_max

    def next_round(self) -> List[Dict[str, Any]]:
        out = []
        for g, es in self.groups.items():
            if not self.needs(g):
                continue
            while self.pos[g] < len(es) and id(es[self.pos[g]]) in self.kept_ids:
                self.pos[g] += 1
            if self.pos[g] < len(es):
                out.append(es[self.pos[g]])
                self.pos[g] += 1
        return out

    def done(self, entry: Dict[str, Any], verdict: Optional[str]) -> None:
        self._book(entry_group(entry), verdict or "invalid")

    def summary(self, per_min: int) -> Dict[str, Any]:
        groups = {group_key_str(g): dict(st, short=st["valid"] < per_min, exhausted=self.pos[g] >= len(self.groups[g]) and self.needs(g))
                  for g, st in self.state.items()}
        return {"per_group_valid": self.per_valid, "per_group_max": self.per_max, "per_group_min": per_min, "groups": groups,
                "n_groups": len(groups), "n_short": sum(1 for v in groups.values() if v["short"]),
                "valid_hist": {str(k): sum(1 for v in groups.values() if min(v["valid"], self.per_valid) == k) for k in range(self.per_valid + 1)}}


def entry_order(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The validation order: the entries the sampler would try (not ``sampler_screen``ed) first, round-robin over their (yaw slot, axial
    group) - each group in library order = the sampler's order -, then the screened ones - so the ``max_entries`` cap samples every yaw's
    and every axial (end, spin)'s first candidates (the first run validated 48 of the spray bottle's 1104 entries, all at yaw 0; the second
    48 screened napkins blades the sampler rejects too; the axial run on the hinge 48 entries of the ``+end+spin`` group only - the library
    loops yaw -> end -> spin -> class, 6 per yaw -, so its maps held no other end or turn sense: i7-contact 2026-11-07)."""
    def round_robin(es):
        groups: Dict[Tuple[float, str], List[Dict[str, Any]]] = {}
        for e in es:
            groups.setdefault((round(float(e.get("yaw_deg", 0.0)), 4), axial_group(e)), []).append(e)
        # within a round: group by group, each over the yaw slots in library order (a non-axial library: one group, the order as before)
        names = sorted({g for _y, g in groups})
        yaws = list(dict.fromkeys(y for y, _g in groups))
        lists = [groups[k] for k in sorted(groups, key=lambda k: (names.index(k[1]), yaws.index(k[0])))]
        out: List[Dict[str, Any]] = []
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
    placed or run fails alone (``no_placement``, ``chain_unreachable``, ``unreachable``, ``error:...``). A WALL library (i7-brace,
    :data:`WALL_FAMILIES`): the entries of ``arm``'s brace only (:func:`wall_arm_entries`), each from the target hung at its level pose
    (``world.hang``: the entry's ``T_rule_object`` at the library's hinge height ``validate.support_z``) instead of the settled rest."""
    # i7-brace (owner 3.4): a wall library is validated per brace arm - this run validates its arm's entries only (the others untouched)
    all_entries = lib_k["entries"]
    is_wall = any(wall_entry(e) for e in all_entries)
    if is_wall:
        lib_k["entries"] = wall_arm_entries(all_entries, arm)
    try:
        tim = _validate_library(world, lib_k, obj, geom, pts_obj, cfg, arm, is_wall)
    finally:
        lib_k["entries"] = all_entries
    if is_wall:  # the two arms' runs keep their own records
        lib_k.setdefault("validation_timing_by_arm", {})[arm] = lib_k["validation_timing"]
        lib_k.setdefault("validated_by_arm", {})[arm] = lib_k["validated"]
    return tim


def _validate_library(world, lib_k: Dict[str, Any], obj, geom, pts_obj: np.ndarray, cfg: ContactValidateConfig, arm: str, is_wall: bool) -> Dict[str, Any]:
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    t_all = time.time()
    tim = {"settle_s": 0.0, "plan_s": 0.0, "trials_s": 0.0, "n_trials": 0, "n_entries": 0, "n_errors": 0}
    rest_q = np.asarray(lib_k["object"]["rest"]["quat_xyzw"], dtype=float)
    settled: Dict[float, Tuple[Dict[str, Any], Any]] = {}
    n_live = 0
    hang_z = float((lib_k.get("validate") or {}).get("support_z") or WALL_HANG_DEFAULT_M)

    def run_entry(e: Dict[str, Any], screen) -> Optional[str]:
        """Validate one entry (its ``validation`` written); returns the verdict."""
        yaw = round(float(e.get("yaw_deg", 0.0)), 4)
        tim["n_entries"] += 1
        try:
            t0 = time.time()
            if yaw not in settled:
                if is_wall:  # the level (goal) pose at the wall, the hinge at hang_z (the same for every entry of the library)
                    settled[yaw] = world.hang(obj, geom, S.pose_of(e["T_rule_object"]), hang_z, e)
                else:
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
            return e["validation"]["verdict"]
        if plan is None or plan.get("q") is None:
            e["validation"] = {"schema": TRIAL_SCHEMA, "verdict": "invalid", "reasons": ["no_placement"], "plan_fail": (plan or {}).get("fail"), "trials": []}
            return "invalid"
        if plan.get("chain_unreachable"):
            e["validation"] = {"schema": TRIAL_SCHEMA, "verdict": "unreachable", "reasons": ["chain_unreachable"], "plan": _plan_facts(plan), "trials": []}
            return "unreachable"
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
        if screen and cfg.screen_as_rank and not cfg.diagnose_screened:
            e["validation"].update(screen=screen, screen_as_rank=True)
        elif screen:
            e["validation"].update(diagnostic="a screened entry validated (the sampler's hand_overlap removes it: never executed)", screen=screen)
        return verdict

    def screen_of(e):
        return (e.get("sampler_screen") or {}).get("hand_overlap")

    if cfg.per_group_valid > 0 and not cfg.diagnose_screened:
        # owner A6: stratified - the live entries (screened ones only with screen_as_rank, after them) per (yaw x end x spin) group
        live = [e for e in lib_k["entries"] if not screen_of(e)]
        ranked = [e for e in lib_k["entries"] if screen_of(e)] if cfg.screen_as_rank else []
        for e in lib_k["entries"]:
            if screen_of(e) and not cfg.screen_as_rank:
                e["validation"] = {"schema": TRIAL_SCHEMA, "verdict": "screened", "reasons": ["sampler_hand_overlap"], "screen": screen_of(e), "trials": []}
                tim["n_screened"] = tim.get("n_screened", 0) + 1
        quota = GroupQuota(live + ranked, cfg.per_group_valid, cfg.per_group_max, keep_verdicts=cfg.skip_validated)
        tim["n_kept"] = len(quota.kept_ids)
        ran = set(quota.kept_ids)
        while True:
            batch = quota.next_round()
            if not batch:
                break
            for e in batch:
                quota.done(e, run_entry(e, screen_of(e)))
                ran.add(id(e))
        if not cfg.skip_validated:
            # FRESH: an entry this run did not reach keeps no verdict of an earlier run (a map reads only this run's verdicts)
            for e in live + ranked:
                if id(e) not in ran and e.get("validation") is not None:
                    e["validation_prev"] = e.pop("validation")
                    tim["n_cleared"] = tim.get("n_cleared", 0) + 1
        lib_k["validation_groups"] = quota.summary(cfg.per_group_min)
    elif cfg.revalidate:
        for e in entry_order(lib_k["entries"]):
            prev = e.get("validation") or {}
            if prev.get("verdict") not in ("valid", "invalid", "unreachable"):
                continue
            e["validation_prev"] = e.pop("validation")
            run_entry(e, screen_of(e))
            tim["n_revalidated"] = tim.get("n_revalidated", 0) + 1
            if e["validation"]["verdict"] != prev.get("verdict"):
                tim["n_changed"] = tim.get("n_changed", 0) + 1
    else:
        for e in entry_order(lib_k["entries"]):
            screen = screen_of(e)
            if cfg.diagnose_screened and not screen:
                continue  # the diagnostic validates the screened entries only; the live ones keep their record
            if screen and not cfg.diagnose_screened and not cfg.screen_as_rank:
                # the sampler never records this class here (its hand_overlap rule): not a contact the oracle would execute; not counted
                e["validation"] = {"schema": TRIAL_SCHEMA, "verdict": "screened", "reasons": ["sampler_hand_overlap"], "screen": screen, "trials": []}
                tim["n_screened"] = tim.get("n_screened", 0) + 1
                continue
            if cfg.skip_validated and (e.get("validation") or {}).get("verdict") is not None:
                tim["n_kept"] = tim.get("n_kept", 0) + 1
                continue
            if n_live >= cfg.max_entries:
                continue
            n_live += 1
            run_entry(e, screen)
    tim = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in tim.items()}
    tim["total_s"] = round(time.time() - t_all, 2)
    tim["rest_id"] = lib_k["object"].get("rest_id")
    tim["kind"] = lib_k["kind"]
    lib_k["validation_timing"] = tim
    lib_k["validated"] = {"schema": TRIAL_SCHEMA, "config": asdict(cfg), "arm": arm, "oracle_config": {k: getattr(CPZ.oracle_config(), k) for k in
                                                                                                       ("push_pre_dist_m", "brace_touch_stop_m", "brace_touch_backoff_m")},
                          "created": S.now()}
    return tim


def placement_offsets(family: str, contact_xy, obj_xy, support_half_x: float, edge_clear: float, arm: str = "left") -> Tuple[Tuple[float, float], ...]:
    """The family's stance relation (:data:`STANCE_RELATIONS`: the contact ``ahead`` / ``lateral`` of the base, yaw 0) as
    ``TabletopWorld.plan``'s object-relative base offsets (it puts the base ``support half + edge clear + off[0]`` behind the OBJECT and
    ``off[1]`` to its side): the base at ``contact - (ahead, lateral)``. A relation with ``mirror_right`` (the wall brace: either arm braces)
    has its laterals mirrored for the right arm."""
    rel = STANCE_RELATIONS[family]
    flip = -1.0 if (rel.get("mirror_right") and arm == "right") else 1.0
    out = []
    for ah in rel["ahead_m"]:
        for la in (flip * v for v in rel["lateral_m"]):
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
        self.variant: Optional[str] = None
        self._pusher = None
        self._pusher_obj()  # before any object: no prim is added while an object's probe view is live
        self._stop = None
        self._stop_obj()
        # i7-brace (owner 3.4): the wall world's kinematic wall (created before any object, parked; WALL_FAMILIES libraries place it)
        self._wall = None
        self._wall_on = False
        self._wall_support = None
        self._wall_obj()

    def load(self, job: Dict[str, Any], model_dir: str):
        return self.tw.load(job, model_dir)

    def unload(self, obj):
        if self._wall_on:  # the next library is a slab library until it hangs its own target
            self._wall_on = False
            self._park_wall()
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
        ignore = [self.tw.slab, obj] + [o for o in (self._pusher, getattr(self, "_stop", None), getattr(self, "_wall", None)) if o is not None]
        self.tw.backend.update_world(ignore_objects=ignore)

    def set_support_height(self, z: Optional[float]) -> None:
        """The slab top of the next library (the alias family's support height: its reach and the arm's posture depend on it); None = the
        world's default (``ValidateConfig.slab_top_z``)."""
        import dataclasses

        self.tw.cfg = dataclasses.replace(self.vcfg, slab_top_z=float(z) if z is not None else self.vcfg.slab_top_z)

    def set_variant(self, variant: Optional[str]) -> None:
        """The customs variant of the next library / replay (its ``generation`` knobs reach the oracle: :func:`run_oracle`); None = none."""
        self.variant = str(variant) if variant else None

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
        if self._wall_on:  # the wall is the support the whole robot's contact policy is checked against (a trial's restore set the slab)
            self.tw.support = self._wall_support
        self.clean_world(obj)
        start = start_pose_of(entry, hull_points(geom), T_obj)
        if start is None:
            return {"q": None, "fail": "start_in_object: the hand meets the object at both pre-contacts (the robot is set there)"}
        w = CPZ.entry_world(entry, T_obj)
        fam = entry.get("family")
        cfg0 = self.tw.cfg
        if fam in STANCE_RELATIONS:
            offs = placement_offsets(fam, w["contact"][0][:2].tolist(), T_obj[:2, 3], float(self.tw.support.half[0]), cfg0.slab_edge_clear_m, arm=arm)
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

    def _motions_state(self, obj, goal: Optional[Dict[str, Any]] = None):
        """The oracle's ``Motions`` (the cuRobo backend as its planner) and ``SimState`` over a task stub: the object and the slab in scope, and
        the push's task goal as the object's TRO pose target (``SimState.pose_target``: the oracle's closed-loop push - the re-push, the goal
        follow, an arc's residual - aims at it, as in the demos; without it run D's axial replays stopped short: 'skipped: beyond the
        recorded arc')."""
        from ispatialgym.oracle.executor import SimState
        from ispatialgym.oracle.motions import Motions, MotionsConfig

        motions = Motions(self.tw.env, self.tw.backend, cfg=MotionsConfig(), robot=self.tw.robot)
        motions.state.hold_current()
        tro = None if goal is None else SimpleNamespace(meta=SimpleNamespace(pose_targets={obj.name: goal_pose_target(obj.name, goal)}))
        scope = {obj.name: obj, self.tw.slab.name: self.tw.slab}
        if self._wall_on:  # i7-brace: the wall in the task scope - SimState.contact_supports gives it to the oracle's planner as the target's support
            scope[self._wall.name] = self._wall
        task = SimpleNamespace(object_scope=scope, _tro=tro)
        return motions, SimState(self.tw.env, task, self.tw.robot)

    def _start(self, obj, entry, plan, state, rest):
        """The rest snapshot restored (verified), the robot set at the planned configuration with the entry's fingers, the drive targets held,
        two hold steps with the sleep assist (as ``validate_tabletop``'s trial)."""
        rec = self.tw.restore(obj, state, rest)
        if self._wall_on:
            self.tw.support = self._wall_support
        q = plan["q"].clone()
        q[self.tw.robot.gripper_control_idx[self.arm]] = float(entry["fingers"]["opening_m"])
        self.tw.robot.set_joint_positions(q)
        self.tw.robot.keep_still()
        obj.wake()
        sa = self.tw.SA.SleepAssist(self.tw.robot, {obj.name: obj}, "gc") if self.vcfg.sleep_assist else None
        self.tw.probe.take()  # the restore's own contact samples are not the trial's
        return rec, sa

    def _stepper(self, obj, sim, arm, sa, series, trace, finger_series: Optional[List[float]] = None):
        k = [0]

        def step(a):
            f = self.tw.step(a, sa, k[0])  # the world's probe (TabletopWorld.load), the sleep assist
            k[0] += 1
            series.append(float(f.get("other_slab", 0.0) or 0.0))
            if finger_series is not None:  # i7-brace: the acting arm's fingers on the object (the wall brace's load reaching the fist)
                finger_series.append(float(f.get("finger_obj", 0.0) or 0.0))
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
        T_start = _T(obj.get_position_orientation())
        # the task's goal (the oracle's pose target, the judge's goal) and the touching push's partner stand-in, at the settled start
        goal = goal_pose(entry, T_start)
        motions, sim = self._motions_state(obj, goal)
        stop = self._place_stop(entry, T_start, pts)
        hinge = self._hinge_on(obj, entry, pts)  # the axial task's COM hinge (the demos' world)
        w = _entry_world(entry, T_start)
        trace = TrialTrace(T_start, _T(w["contact"]))
        trace.rec.update(obj_rest=[round(float(v), 4) for v in rest.get("world_position", [])] + [round(float(v), 4) for v in rest.get("world_quat_xyzw", [])],
                         eef_start=_pose7(_T(sim.eef_pose(arm))), restore=rec,
                         stop=None if stop is None else {"centre": [round(float(v), 4) for v in stop["centre"]], "gap_m": stop["gap_m"]},
                         goal=None if goal is None else {"pose": _pose7(goal["T"]), "position_tolerance": goal["position_tolerance"],
                                                         "orientation_tolerance": goal["orientation_tolerance"]})
        step = self._stepper(obj, sim, arm, sa, series, trace)
        self._hold(motions, step, self.vcfg.init_hold_steps)
        T0 = _T(obj.get_position_orientation())
        act = entry_action(entry, T0, obj.name, arm, touches=(self.tw.slab.name,))
        track = PeakTracker(entry, T0, pts)

        def step_track(a):
            step(a)
            track(_T(obj.get_position_orientation()), _T(sim.eef_pose(arm)))

        try:
            run = run_oracle(act, motions, sim, step_track, self.cfg.max_steps, variant=self.variant)
        except Exception as ex:  # noqa: BLE001 - the entry fails alone, with its trace
            self._park(self._stop, 1)
            self._hinge_off(hinge, obj)
            out = _error_trial(ex)
            return dict(out, restore=rec, base=plan.get("base"), trunk=plan.get("trunk"), debug=trace.end(_T(obj.get_position_orientation()), _T(sim.eef_pose(arm))),
                        wall_s=round(time.time() - t0, 3))
        T1, Te = _T(obj.get_position_orientation()), _T(sim.eef_pose(arm))
        self._park(self._stop, 1)  # after the end pose is read (a target leaning on it is judged where it stands)
        self._hinge_off(hinge, obj)
        trace.rec["hinge"] = None if hinge is None else {k: v for k, v in hinge.items() if k != "hinge"}
        out = judge_push(entry, T0, T1, Te, pts, self.cfg, run, max(series) if series else 0.0, tip_gap_m=track.gap, goal=goal,
                         touch_gap_m=touch_gap(stop, T1, pts))
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
        hinge = self._hinge_on(obj, entry, pts)  # the axial task's COM hinge (the demos' world)
        w = _entry_world(entry, T_start)
        trace = TrialTrace(T_start, _T(w["contact"]))
        trace.rec.update(eef_start=_pose7(_T(sim.eef_pose(arm))), restore=rec, hinge=None if hinge is None else {k: v for k, v in hinge.items() if k != "hinge"})
        fingers: List[float] = []
        step = self._stepper(obj, sim, arm, sa, series, trace, fingers)
        self._hold(motions, step, self.vcfg.init_hold_steps)
        act = entry_action(entry, _T(obj.get_position_orientation()), obj.name, arm)
        try:
            run = run_oracle(act, motions, sim, step, self.cfg.max_steps, variant=self.variant)
            T0, Te0 = _T(obj.get_position_orientation()), _T(sim.eef_pose(arm))
            trace.rec["braced"] = {"obj": _pose7(T0), "eef": _pose7(Te0)}
            pusher = self._pusher_obj()
            if wall_entry(entry):  # i7-brace: the leveling push carried on past level (3-D, in the wall plane), judged before and after it
                pre_load = judge_wall_standing(run, self.cfg, max(series) if series else 0.0)
                n_pre = len(series)
                path, q_push, push_facts = wall_pusher_path(entry, T0, pts, self.cfg, 1.0 / 30.0)
                trace.rec["pusher"] = {k: v for k, v in push_facts.items() if k != "phis"}
            else:
                path = pusher_path(entry, T0, pts, self.cfg, 1.0 / 30.0)
                q_push = [np.array([0.0, 0.0, 0.0, 1.0])] * len(path)
            if wall_entry(entry):
                # i7-brace: the arc sweep, stopped phi_press after the target first moved (wall_load_stop); the sleep assist sleeps the still
                # hung target while the brace stands and a teleported kinematic pusher does not wake it (run 2: 6 / 12 signpost trials did not
                # move) - the push hand it stands in for would: woken every step
                turns: List[float] = []
                stop = None
                for i, c in enumerate(path):
                    pusher.set_position_orientation(th_.tensor(c, dtype=th_.float32), th_.tensor(q_push[i], dtype=th_.float32))
                    obj.wake()
                    self._hold(motions, step, 1)
                    # SIGNED (past level +): an unsigned turn read the target's small backward settle after the brace as the contact (run 6:
                    # 10 / 17 painting trials stopped at the sweep's first step, the load never reached them)
                    turns.append(signed_turn(T0[:3, :3], _T(obj.get_position_orientation())[:3, :3], push_facts["axis"]))
                    stop = wall_load_stop(push_facts["phis"], turns, push_facts["phi_press_rad"], self.cfg.wall_contact_turn_rad)
                    if stop is not None and i >= stop:
                        break
                trace.rec["pusher"] = dict({k: v for k, v in push_facts.items() if k != "phis"}, stop_step=stop, swept_steps=len(turns),
                                           contact_step=next((j for j, t in enumerate(turns) if t >= self.cfg.wall_contact_turn_rad), None))
                for _ in range(self.cfg.brace_hold_steps):
                    self._hold(motions, step, 1)
            elif brace_load_switches()["diag"]:  # i7-brace 5w.3: the diagnostic load (record / wake / measured), default off
                trace.rec["load_diag"] = self._brace_load_diag(obj, entry, T0, path, pusher, motions, step, sa, brace_load_switches())
            else:
                for i, c in enumerate(path + [None] * self.cfg.brace_hold_steps):
                    if c is not None:  # the pusher one step further along the load; the robot holds (the brace arm stays at its contact)
                        pusher.set_position_orientation(th_.tensor(c, dtype=th_.float32), th_.tensor(q_push[i], dtype=th_.float32))
                    self._hold(motions, step, 1)
        except Exception as ex:  # noqa: BLE001 - the entry fails alone, with its trace
            self._park_pusher()
            self._hinge_off(hinge, obj)
            out = _error_trial(ex)
            return dict(out, restore=rec, base=plan.get("base"), trunk=plan.get("trunk"), debug=trace.end(_T(obj.get_position_orientation()), _T(sim.eef_pose(arm))),
                        wall_s=round(time.time() - t0, 3))
        finally:
            self._park_pusher()
        T1, Te1 = _T(obj.get_position_orientation()), _T(sim.eef_pose(arm))
        self._hinge_off(hinge, obj)
        if wall_entry(entry):
            out = judge_wall_brace(entry, T0, T1, Te0, Te1, self.cfg, run, max(series[n_pre:]) if len(series) > n_pre else 0.0,
                                   finger_N=max(fingers[n_pre:]) if len(fingers) > n_pre else 0.0)
            out["pre_load"] = pre_load
            return dict(out, schema=TRIAL_SCHEMA, restore=rec, base=plan.get("base"), trunk=plan.get("trunk"), run=run, debug=trace.end(T1, Te1),
                        hinge=None if hinge is None else {k: v for k, v in hinge.items() if k != "hinge"}, wall_s=round(time.time() - t0, 3))
        load = np.asarray((entry.get("load") or {}).get("direction") or [0.0, 1.0, 0.0], dtype=float)
        out = judge_brace(entry, T0, T1, Te0, Te1, load, self.cfg, run, max(series) if series else 0.0)
        return dict(out, schema=TRIAL_SCHEMA, restore=rec, base=plan.get("base"), trunk=plan.get("trunk"), run=run, debug=trace.end(T1, Te1),
                    wall_s=round(time.time() - t0, 3))

    def _brace_load_diag(self, obj, entry, T0, path, pusher, motions, step, sa, sw: Dict[str, bool]) -> Dict[str, Any]:
        """i7-brace 5w.3 (diagnosis): today's brace load (:func:`pusher_path`, the same steps and hold), recorded - and with ``sw["wake"]``
        the object woken at every pusher step, with ``sw["measured"]`` the pusher stopped ``brace_press_m`` after the measured contact
        (:func:`measured_load_stop`, the path extended at the same step up to ``brace_pusher_travel_max_m``)."""
        th_ = self.tw.th
        cfg = self.cfg
        d = np.asarray((entry.get("load") or {}).get("direction") or [0.0, 1.0, 0.0], dtype=float)
        d[2] = 0.0
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        n = len(path) - 1
        ds = (cfg.brace_pusher_gap_m + cfg.brace_press_m) / max(n, 1)
        pts = list(path)
        if sw["measured"]:
            n_max = int(math.ceil(cfg.brace_pusher_travel_max_m / max(ds, 1e-9)))
            pts = [path[0] + d * ds * k for k in range(n_max + 1)]
        p0 = brace_load_point(entry, T0)
        name = obj.name
        facts: Dict[str, Any] = {"switches": dict(sw), "asleep_at_load": _asleep(obj),
                                 "sa_sleeps_before": list((sa.sleeps.get(name) or []) if sa is not None else []),
                                 "sa_wakes_before": list((sa.wakes.get(name) or []) if sa is not None else []),
                                 "computed_contact_m": round(float(cfg.brace_pusher_gap_m), 5), "step_m": round(float(ds), 6), "path_steps": n}
        travels, moved, asleep_n = [], [], 0
        stop = s_c = None
        for k, c in enumerate(pts):
            pusher.set_position_orientation(th_.tensor(c, dtype=th_.float32), th_.tensor([0.0, 0.0, 0.0, 1.0], dtype=th_.float32))
            if sw["wake"]:
                obj.wake()
            self._hold(motions, step, 1)
            asleep_n += int(bool(_asleep(obj)))
            travels.append(ds * k)
            moved.append(float((brace_load_point(entry, _T(obj.get_position_orientation())) - p0) @ d))
            if sw["measured"]:
                stop, s_c = measured_load_stop(travels, moved, cfg.brace_press_m, cfg.brace_contact_m)
                if stop is not None:
                    break
        if not sw["measured"]:
            _st, s_c = measured_load_stop(travels, moved, cfg.brace_press_m, cfg.brace_contact_m)
        hold_moved = []
        for _ in range(cfg.brace_hold_steps):
            self._hold(motions, step, 1)
            asleep_n += int(bool(_asleep(obj)))
            hold_moved.append(float((brace_load_point(entry, _T(obj.get_position_orientation())) - p0) @ d))
        facts.update(swept_steps=len(travels), stop_step=stop, travel_end_m=round(travels[-1], 5) if travels else None,
                     contact_travel_m=None if s_c is None else round(s_c, 5),
                     early_m=None if s_c is None else round(float(cfg.brace_pusher_gap_m) - s_c, 5),
                     load_point_moved_max_m=round(max(moved + hold_moved), 5) if moved else None,
                     load_point_moved_end_m=round((hold_moved or moved or [0.0])[-1], 5),
                     load_point_moved_at_path_end_m=round(moved[min(n, len(moved) - 1)], 5) if moved else None,
                     asleep_steps=asleep_n, loaded=bool(moved and max(moved + hold_moved) >= cfg.wall_loaded_min_m),
                     sa_sleeps=list((sa.sleeps.get(name) or []) if sa is not None else []),
                     sa_wakes=list((sa.wakes.get(name) or []) if sa is not None else []))
        return facts

    def probe_wake(self, obj, entry, plan, state, rest, arm, cond: str, speed: float, presleep_steps: int = 150, press_m: float = 0.015) -> Dict[str, Any]:
        """i7-brace 5w.3(b) PROBE (diagnosis only; ``validate_contacts probe``): does the REAL ARM's first contact wake a sleeping partner?
        The rest restored, the robot at the entry's pre-contact (still) until the partner sleeps, then the hand PRESSED straight along the
        entry's approach from the pre-contact to ``press_m`` past the contact pose at ``speed`` m/s (``Motions.arm_along``: the oracle's
        own Cartesian tracking; the oracle's brace stops short of the face by design, so it never touches a still partner). ``cond``:
        ``native`` no assist (PhysX's own sleep at the 0.005 threshold); ``assist`` the sleep assist running (it sleeps the partner only if
        PhysX did not first; it proactively wakes only a body IT slept); ``assist_marked`` as ``assist`` with the sleep booked as the
        assist's (the proactive wake armed: a moving eef >= 0.3 mm/step within 5 cm of the box); ``awake`` no assist, woken every step (the
        control); ``tp_pose`` / ``tp_state`` / ``tp_state_newsa`` asleep and booked, then teleported in place (``set_position_orientation`` of
        its own pose: a TRO object state load) / the scene state restored (``TabletopWorld.restore``, the robot set back), the last with a
        NEW sleep assist (a new episode's: the sleep not booked). Per step: asleep, speeds, eef distance to the object's box / speed, finger
        load. Returns the timeline facts: who woke it (assist / physx / never), the steps of wake / touch / first motion, speeds after it."""
        th_ = self.tw.th
        from ispatialgym.oracle import sleep_assist as SAM
        from ispatialgym.sampling.v2 import contact_poses as CPZ

        rec, sa = self._start(obj, entry, plan, state, rest)
        if cond in ("native", "awake"):
            sa = None
        motions, sim = self._motions_state(obj)
        series: List[float] = []
        rows: List[Dict[str, Any]] = []

        def box_dist(pe):
            lo, hi = (th_.as_tensor(v) for v in obj.aabb)
            gap = th_.maximum(th_.maximum(lo - pe, pe - hi), th_.zeros(3))
            return float(th_.linalg.norm(gap))

        cur = {"base": self._stepper(obj, sim, arm, sa, series, None), "sa": sa}
        eprev = [None]

        def step(a):
            if cond == "awake":
                obj.wake()
            f = cur["base"](a)
            pe = th_.as_tensor(self.tw.robot.get_eef_position(arm)).detach().clone()
            sp = float(th_.linalg.norm(pe - eprev[0])) if eprev[0] is not None else 0.0
            eprev[0] = pe
            p = obj.get_position_orientation()[0]
            s_ = cur["sa"]
            rows.append({"k": len(rows), "asleep": _asleep(obj), "v": float(th_.linalg.norm(obj.get_linear_velocity())),
                         "w": float(th_.linalg.norm(obj.get_angular_velocity())), "p": [float(x) for x in p.tolist()],
                         "eef_box_m": box_dist(pe), "eef_speed_mm": sp * 1000.0, "finger_N": float(f.get("finger_obj", 0.0) or 0.0),
                         "sa_wakes": len((s_.wakes.get(obj.name) or [])) if s_ is not None else 0,
                         "sa_sleeps": len((s_.sleeps.get(obj.name) or [])) if s_ is not None else 0})
            return f

        k_asleep = None
        for _ in range(int(presleep_steps)):
            self._hold(motions, step, 1)
            if cond != "awake" and rows[-1]["asleep"]:
                k_asleep = rows[-1]["k"]
                break
        slept_by = None if k_asleep is None else ("assist" if rows[-1]["sa_sleeps"] > 0 else "physx")

        def book(s_):  # the sleep booked as the assist's own (its proactive wake armed)
            if s_ is not None and k_asleep is not None:
                s_.slept[obj.name] = True
                s_.sleeps.setdefault(obj.name, []).append(len(rows))

        tp = {}
        if cond in ("assist_marked", "tp_pose", "tp_state"):
            book(cur["sa"])
        if cond.startswith("tp_") and k_asleep is not None:
            if cond == "tp_pose":
                p, q = obj.get_position_orientation()
                obj.set_position_orientation(p.clone(), q.clone())
            else:  # the scene state (robot included) restored, the robot set back at the pre-contact (as _start), the object NOT woken
                self.tw.restore(obj, state, rest)
                q = plan["q"].clone()
                q[self.tw.robot.gripper_control_idx[self.arm]] = float(entry["fingers"]["opening_m"])
                self.tw.robot.set_joint_positions(q)
                self.tw.robot.keep_still()
                if cond == "tp_state_newsa":
                    cur["sa"] = SAM.SleepAssist(self.tw.robot, {obj.name: obj}, "gc_new")
                    cur["base"] = self._stepper(obj, sim, arm, cur["sa"], series, None)
            tp = {"asleep_after_teleport": _asleep(obj)}
        k0 = len(rows)
        p0 = np.asarray(rows[-1]["p"]) if rows else None
        w = CPZ.entry_world(entry, _T(obj.get_position_orientation()))
        cpos, cq = w["contact"]
        To = _T(obj.get_position_orientation())
        app = To[:3, :3] @ np.asarray(entry.get("approach") or [0.0, 0.0, -1.0], dtype=float)
        app = app / max(float(np.linalg.norm(app)), 1e-12)
        target = (cpos + th_.tensor(app * float(press_m), dtype=th_.float32), cq)
        err = None
        try:
            for a in motions.arm_along(arm, [target], speed=float(speed), converge=False):
                step(a)
            self._hold(motions, step, 30)
        except Exception as ex:  # noqa: BLE001
            err = repr(ex)[:300]
        tl = rows[k0:]
        mv = next((r["k"] for r in tl if p0 is not None and float(np.linalg.norm(np.asarray(r["p"]) - p0)) > 0.0005), None)
        woke = next((r["k"] for r in tl if r["asleep"] is False), None) if k_asleep is not None else None
        sa_w0 = rows[k0 - 1]["sa_wakes"] if k0 else 0
        sa_w = next((r["k"] for r in tl if r["sa_wakes"] > sa_w0), None)
        touch = next((r["k"] for r in tl if r["finger_N"] > 0.05), None)
        after = [r for r in rows if mv is not None and mv <= r["k"] < mv + 15]
        e_sp = [r["eef_speed_mm"] for r in tl[5:] if touch is None or r["k"] < touch]
        return {"cond": cond, "speed_m_s": speed, "restore": rec, "presleep_asleep_k": k_asleep, "slept_by": slept_by, **tp, "press_start_k": k0,
                "sa_wake_k": sa_w, "first_awake_k": woke, "first_touch_k": touch, "first_motion_k": mv,
                "asleep_steps_touched": sum(1 for r in tl if r["asleep"] and r["finger_N"] > 0.05),
                "touch_N_max_while_asleep": max((r["finger_N"] for r in tl if r["asleep"]), default=0.0),
                "woken_by": None if k_asleep is None else ("assist" if sa_w is not None and (woke is None or sa_w <= woke) else ("physx" if woke is not None else "never")),
                "eef_box_m_at_wake": next((round(r["eef_box_m"], 4) for r in tl if woke is not None and r["k"] == woke), None),
                "eef_box_m_at_touch": next((round(r["eef_box_m"], 4) for r in tl if touch is not None and r["k"] == touch), None),
                "eef_speed_mm_approach_med": round(float(np.median(e_sp)), 4) if e_sp else None,
                "obj_v_max_after_motion": None if not after else round(max(r["v"] for r in after), 4),
                "obj_w_max_after_motion": None if not after else round(max(r["w"] for r in after), 4),
                "moved_total_m": round(float(np.linalg.norm(np.asarray(rows[-1]["p"]) - p0)), 5) if p0 is not None else None,
                "error": err, "steps": len(rows)}

    def replay(self, sit: Dict[str, Any], obj, geom) -> Dict[str, Any]:
        """A recorded demo's situation (:func:`replay_situation`) rebuilt: the slab at the recorded support height, the object settled at
        its recorded stance-frame pose, the robot at the oracle's own witness configuration of the pre-contact (base at the origin), then the
        validator's chain IK of the record's poses from there (the clean world) and the ORACLE's action of the record itself, judged as
        the library entries are. ``divergence`` names what differs from the real run."""
        from ispatialgym.sampling.v2 import contact_poses as CPZ

        self.set_support_height(sit["support_z"])
        self.set_variant(sit.get("variant"))
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

    def _reprobe(self, obj, fn):
        """``fn()`` (a hinge prim added / removed: ``og.sim.update_handles``) with the world's contact probe uninstalled before and rebuilt
        after - the physics views a prim change invalidates (run C, finding (d))."""
        try:
            self.tw.probe.uninstall()
        except Exception:  # noqa: BLE001
            pass
        try:
            return fn()
        finally:
            # i7-brace: in the wall world the probe's support column is the wall (a non-finger link loading it)
            self.tw.probe = self.VT.ForceProbe(self.tw.robot, self._wall if getattr(self, "_wall_on", False) else self.tw.slab, obj, self.arm)
            self.tw.probe.install()

    def physics_facts(self, obj) -> Dict[str, Any]:
        """The contact physics of the trial (run E diagnosis): the object's mass and the physics materials bound to its and the slab's
        collision meshes (static / dynamic friction, restitution; None = the PhysX default material, 0.5 / 0.5 / 0)."""
        def mats(o):
            out = []
            for link in o.links.values():
                for name, m in link.collision_meshes.items():
                    pm = m.get_applied_physics_material()
                    out.append({"mesh": name, "material": None if pm is None else
                                {"static_friction": float(pm.get_static_friction()), "dynamic_friction": float(pm.get_dynamic_friction()),
                                 "restitution": float(pm.get_restitution()), "path": pm.prim_path}})
            return out

        return {"object_mass_kg": round(float(obj.mass), 5), "object": mats(obj), "slab": mats(self.tw.slab)}

    def set_slab_friction(self, obj, mu: float) -> None:
        """DIAGNOSTIC: bind a physics material of friction ``mu`` (static = dynamic, restitution 0) to the slab's collision meshes."""
        import omnigibson.lazy as lazy

        def bind():
            # the material prim is a USD edit (outside editing_usd it raises: sweep 18368438-41); the binding opens its own editing_usd
            # (GeomPrim.apply_physics_material) and nesting raises (sweep 18369352-55), so it runs after the first block closes
            with self.tw.og.sim.editing_usd():
                pm = lazy.isaacsim.core.api.materials.physics_material.PhysicsMaterial(
                    prim_path=f"{self.tw.slab.prim_path}/Looks/diag_physics_mat_{int(round(mu * 1000))}", name=f"diag_physics_mat_{int(round(mu * 1000))}",
                    static_friction=float(mu), dynamic_friction=float(mu), restitution=0.0)
            for link in self.tw.slab.links.values():
                for m in link.collision_meshes.values():
                    m.apply_physics_material(pm)
            self.tw.og.sim.update_handles()
            self.tw.og.sim.step()

        self._reprobe(obj, bind)

    def _hinge_on(self, obj, entry, pts) -> Optional[Dict[str, Any]]:
        """The axial task's COM hinge (``AlignmentAxial.after_settle``: ``constraints.create_hinge`` at the object's PhysX COM on its vertical,
        ``hinge_sink_m`` below the slab top, parented to the slab, the variant's resistance) for an axial trial at the object's start pose;
        None for every other trial."""
        if wall_entry(entry):
            return self._wall_hinge_on(obj, entry)
        if not axial_hinged(entry, self.variant):
            return None
        from ispatialgym.sampling import constraints as CN
        from ispatialgym.sampling.v2.tasks.alignment_axial import _com_local

        T = _T(obj.get_position_orientation())
        com = T[:3, :3] @ np.asarray(_com_local(obj).tolist(), dtype=float) + T[:3, 3]
        ext = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
        spec = axial_hinge_spec(self.variant, float(obj.mass), (float(ext.max()), float(ext.min())))
        anchor = [float(com[0]), float(com[1]), float(self.tw.cfg.slab_top_z) - spec["sink_m"]]
        res = spec["resistance"]
        h = self._reprobe(obj, lambda: CN.create_hinge(obj, self.tw.th.tensor(anchor), self.tw.th.tensor([0.0, 0.0, 1.0]), parent_link=self.tw.slab.root_link,
                                                       name="isg_validate_hinge", friction=None if res is None else res["friction"],
                                                       damping=None if res is None else res["damping"]))
        return {"hinge": h, "anchor": [round(v, 4) for v in anchor], "sink_m": spec["sink_m"],
                "resistance": None if res is None else {k: v for k, v in res.items() if isinstance(v, (int, float))}}

    def _hinge_off(self, hinge: Optional[Dict[str, Any]], obj) -> None:
        if hinge is None:
            return
        from ispatialgym.sampling import constraints as CN

        self._reprobe(obj, lambda: CN.remove_hinge(hinge["hinge"]))

    def _stop_obj(self):
        """The touching push's PARTNER stand-in: a kinematic box (``stop_size_m``) the target is pushed into - the stop at the place gap that
        ends a touching push in the demos (without it the target ran the whole push margin on and a tall narrow one tipped over: run D's
        touching_02 replay, tilt 0.82 rad), parked below the floor between trials.

        i7-contactfu 2026-11-08 (owner A2): ``fixed_base=True`` with ``kinematic_only=True`` - OmniGibson makes an object kinematic only when it
        has a fixed base (``usd_utils.compute_kinematic_only``: ``if not fixed_base: return False``), so until now the stand-in was a DYNAMIC
        2.4 kg box 4 cm thick and 30 cm tall (scene probe $OFH/contactfu_20261108/a2: rigidBodyEnabled, kinematicEnabled 0, mass 2.4) that
        fell below the floor while parked and gave way under the tilting target (touching_02 replay: the target leaned 0.46 rad on it). The
        demo's partner is braced by the left gripper and stays put (fix2 300008: it moved 5 mm, tilted 8 deg, the target came back level);
        the slab is built the same way (fixed_base, kinematic)."""
        if self._stop is None:
            from omnigibson.objects.primitive_object import PrimitiveObject

            sx, sy, sz = self.cfg.stop_size_m
            self._stop = PrimitiveObject(name="isg_push_stop", primitive_type="Cube", size=1.0, **STOP_STANDIN_KINEMATIC, rgba=(0.9, 0.5, 0.2, 1.0),
                                         scale=self.tw.th.tensor([sx, sy, sz]))  # its x (thickness) along the push: stop_pose turns it so
            self.tw.scene.add_object(self._stop)
            self._park(self._stop, 1)
            self.tw.og.sim.step()
        return self._stop

    def _park(self, o, k: int = 0):
        o.set_position_orientation(self.tw.th.tensor([5.0 + 0.5 * k, 5.0, -1.0]), self.tw.th.tensor([0.0, 0.0, 0.0, 1.0]))

    def _place_stop(self, entry: Dict[str, Any], T_obj: np.ndarray, pts_obj: np.ndarray) -> Optional[Dict[str, Any]]:
        """The stop at :func:`stop_pose` (None: the entry has no ``push.stop_gap_m``, the stop stays parked)."""
        c = stop_pose(entry, T_obj, pts_obj, self.cfg)
        if c is None:
            return None
        th_ = self.tw.th
        self._stop.set_position_orientation(th_.tensor(c["centre"], dtype=th_.float32), th_.tensor(S.mat_to_quat(c["R"]), dtype=th_.float32))
        return c

    # ------------------------------------------------------------------ i7-brace (owner 3.4): the wall world

    def _wall_obj(self):
        """The wall of the WALL_FAMILIES libraries: a KINEMATIC box (``PUSHER_KINEMATIC``: fixed_base + kinematic_only, as the pusher fix),
        category ``walls`` (the oracle's ``SimState.contact_supports`` then hands it to the planner as the hung target's support, as a scene
        wall), ``wall_size_m``; parked far below the floor between wall libraries."""
        if self._wall is None:
            from omnigibson.objects.primitive_object import PrimitiveObject

            sx, sy, sz = self.cfg.wall_size_m
            self._wall = PrimitiveObject(name="isg_validate_wall", primitive_type="Cube", category="walls", size=1.0, **PUSHER_KINEMATIC,
                                         rgba=(0.85, 0.85, 0.8, 1.0), scale=self.tw.th.tensor([sx, sy, sz]))
            self.tw.scene.add_object(self._wall)
            self._park_wall()
            self.tw.og.sim.step()
        return self._wall

    def _park_wall(self):
        if self._wall is not None:
            self._wall.set_position_orientation(self.tw.th.tensor([-30.0, 30.0, -10.0]), self.tw.th.tensor([0.0, 0.0, 0.0, 1.0]))

    def hang(self, obj, geom, T_rule: np.ndarray, hang_z: float, entry: Dict[str, Any]):
        """The wall world's "settle" (i7-brace, owner 3.4): the slab out of reach, the wall at the rule frame's plane (x = 0, facing -x: the robot
        faces it from x < 0), the target at its LEVEL pose ``T_rule`` with the hinge ``hang_z`` above the floor, hung on the task's COM hinge
        (:meth:`_wall_hinge_on`) for ``wall_hang_steps`` (the robot parked), the hinge removed again (each trial installs its own) and the scene
        snapshot taken - ``(rest, state)`` as :meth:`settle` returns them (the restore keys, ``slab`` = where the slab was parked, ``hang`` =
        the facts: the drift on the hinge, the PhysX COM's offset from the library's hinge point)."""
        th_ = self.tw.th
        cfg = self.cfg
        sx, sy, sz = cfg.wall_size_m
        centre = [0.5 * sx, 0.0, 0.5 * sz]
        self._wall_on = True
        self._wall.set_position_orientation(th_.tensor(centre, dtype=th_.float32), th_.tensor([0.0, 0.0, 0.0, 1.0]))
        from ispatialgym.grasping import contact_policy as CP

        self._wall_support = CP.Support.box(centre, [0.5 * sx, 0.5 * sy, 0.5 * sz], name="wall")
        self.tw.park()
        slab_c, slab_d = [-20.0, -20.0, -5.0], [0.6, 0.6, float(self.tw.cfg.slab_thickness_m)]
        self.tw.place_slab(slab_c, slab_d)
        self.tw.support = self._wall_support
        self._reprobe(obj, lambda: None)  # the probe's support column: the wall
        T = np.array(T_rule, dtype=float)
        T[2, 3] += float(hang_z)
        q_i = S.mat_to_quat(T[:3, :3])
        obj.set_position_orientation(th_.tensor(T[:3, 3], dtype=th_.float32), th_.tensor(q_i, dtype=th_.float32))
        obj.keep_still()
        hinge = self._wall_hinge_on(obj, entry)
        a = self.tw.hold_action()
        for _ in range(int(cfg.wall_hang_steps)):
            self.tw.step(a)
        p, q = (x.clone() for x in obj.get_position_orientation())
        drift = _quat_angle(S.quat_to_mat(q.numpy().astype(float)), T[:3, :3])
        shift = float(np.linalg.norm(p.numpy().astype(float) - T[:3, 3]))
        self._hinge_off(hinge, obj)
        obj.keep_still()
        state = self.tw.scene.dump_state(serialized=False)
        rest = {"schema": S.REST, "rest_kind": "wall_hang", "quat_xyzw": [round(float(v), 7) for v in q.tolist()],
                "h_root": round(float(p[2]) - float(self.tw.cfg.slab_top_z), 6), "slab": {"centre": slab_c, "dims": slab_d},
                "world_position": [float(v) for v in p.tolist()], "world_quat_xyzw": [float(v) for v in q.tolist()],
                "hang": {"hang_z": float(hang_z), "drift_rad": round(drift, 5), "shift_m": round(shift, 5), "steps": int(cfg.wall_hang_steps),
                         "stable": bool(drift <= cfg.wall_hang_drift_max_rad), "wall_centre": centre, "wall_size_m": list(cfg.wall_size_m),
                         "hinge": None if hinge is None else {k: v for k, v in hinge.items() if k != "hinge"}}}
        return rest, state

    def _wall_hinge_on(self, obj, entry: Dict[str, Any]) -> Dict[str, Any]:
        """The world_vertical task's hinge (``AlignmentWall.apply_draft_extras``): ``constraints.create_hinge`` at the target's PhysX COM about
        the wall normal, parented to the wall, the resistance of ``AlignmentWall.hinge_resistance`` (the mass over the width x height, the
        variant's ``hinge_*`` knobs over the world_vertical defaults)."""
        from ispatialgym.oracle import execute as EX
        from ispatialgym.sampling import constraints as CN
        from ispatialgym.sampling import generation_defaults as GD
        from ispatialgym.sampling.v2.tasks.alignment_wall import AlignmentWall

        wall = entry.get("wall") or {}
        T = _T(obj.get_position_orientation())
        com_l = np.asarray(AlignmentWall._com_local(None, obj).tolist(), dtype=float)
        com = T[:3, :3] @ com_l + T[:3, 3]
        axis = _unit(T[:3, :3] @ np.asarray(wall.get("normal_object") or [1.0, 0.0, 0.0], dtype=float))
        gen = dict(GD.defaults_for("alignment-world_vertical"))
        gen.update(EX.variant_generation(oracle_ctx(self.variant)) if self.variant else {})
        ext = wall.get("extents_perp_m") or [0.3, 0.3]
        res = CN.hinge_resistance(float(obj.mass), (float(ext[0]), float(ext[1])), decel=float(gen.get("hinge_stop_decel_rad_s2", CN.HINGE_STOP_DECEL_RAD_S2)),
                                  tau=float(gen.get("hinge_viscous_tau_s", CN.HINGE_VISCOUS_TAU_S)),
                                  omega_brake=float(gen.get("hinge_brake_omega_rad_s", CN.HINGE_BRAKE_OMEGA_RAD_S)))
        th_ = self.tw.th
        h = self._reprobe(obj, lambda: CN.create_hinge(obj, th_.tensor(com, dtype=th_.float32), th_.tensor(axis, dtype=th_.float32), parent_link=self._wall.root_link,
                                                       name="isg_validate_wall_hinge", friction=res["friction"], damping=res["damping"]))
        off = float(np.linalg.norm(com_l - np.asarray(wall.get("hinge_object") or com_l, dtype=float)))
        return {"hinge": h, "anchor": [round(float(v), 4) for v in com], "axis": [round(float(v), 4) for v in axis], "com_offset_m": round(off, 5),
                "resistance": {k: v for k, v in res.items() if isinstance(v, (int, float))}}

    def _park_pusher(self):
        if self._pusher is not None:
            self._pusher.set_position_orientation(self.tw.th.tensor([5.0, 5.0, -1.0]), self.tw.th.tensor([0.0, 0.0, 0.0, 1.0]))

    def _pusher_obj(self):
        """The brace load's pusher: a kinematic box (set_position drives it into the partner, PhysX pushes the partner out), parked below the
        floor 5 m away between loads. i7-brace 2026-11-08 (owner 3.3): built with ``PUSHER_KINEMATIC`` (fixed_base + kinematic_only) -
        OmniGibson honours ``kinematic_only`` only with a fixed base, so until now the pusher was a DYNAMIC box (as the touching stand-in
        was, 021aa89c): gravity and the partner's contact moved it during the load."""
        if self._pusher is None:
            from omnigibson.objects.primitive_object import PrimitiveObject

            sx, sy, sz = self.cfg.brace_pusher_size_m
            self._pusher = PrimitiveObject(name="isg_brace_pusher", primitive_type="Cube", size=1.0, **PUSHER_KINEMATIC, rgba=(0.2, 0.6, 0.9, 1.0),
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
    # the object's TRO pose target (the task's goal, what the demo's oracle closed its push on), into the stance frame
    pt = dict(((tro.get("ispatialgym") or {}).get("pose_targets") or {}).get(obj) or {}) or None
    if pt is not None:
        if pt.get("position") is not None:
            pt["position"] = [float(v) for v in (Tbi[:3, :3] @ np.asarray(pt["position"], dtype=float) + Tbi[:3, 3])]
        if pt.get("orientation") is not None:
            pt["orientation"] = [float(v) for v in S.mat_to_quat(Tbi[:3, :3] @ S.pose_mat([0.0, 0.0, 0.0], pt["orientation"])[:3, :3])]
    return {"kind": kind, "arm": rec.get("arm"), "object": obj, "object_from": obj_from, "variant": oracle.get("variant") or "", "T_base_object": Tbi @ T_obj, "support_z": support_z,
            "pose_target": pt,
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
    goal = record_goal(sit, e)
    if goal is not None:
        stop = goal.pop("stop_gap_m", None)
        if stop is not None:
            e["push"]["stop_gap_m"] = stop
        e["goal_target"] = goal
    return e


def record_goal(sit: Dict[str, Any], entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The replayed push's task goal (``goal_target``, :func:`goal_pose`), at the replay's object start pose:

    - a touching push (the variant's ``touching``, a straight push): the partner's face where the demo had it - the record's push distance
      less the task's ``push_margin_m`` (``iors_small``: push distance = the gap at the place + 0.035) - is the stop (``stop_gap_m``), the
      goal the target touching it (``along_push_m`` = that gap; the TRO's tolerances 0.03 / 0.2). The TRO's own goal position, measured from
      the replay's start (the place pose before its settle), is kept as ``tro_along_push_m`` (touching_01: 0.0235 vs gap 0.0217; touching_02:
      0.0445 vs 0.0493 - the goal was drafted against the partner before the target's 3 mm settle);
    - an arc (axial) push: the turn of the TRO's orientation target from the start (``turn_rad``; tolerance the TRO's, 0.1), else the
      record's arc (init -> goal eef turn).

    None for any other push / a brace."""
    from ispatialgym.grasping.contact_library import AXIAL_GOAL_TOL, TOUCH_GOAL_TOL, TOUCH_PUSH_MARGIN_M

    rec = sit["record"]
    if sit["kind"] != "push":
        return None
    T0 = sit["T_base_object"]
    pt = sit.get("pose_target") or {}
    if "touching" in str(sit.get("variant") or "") and rec.get("mid") is None and rec.get("distance_m") is not None:
        gap = max(0.0, float(rec["distance_m"]) - TOUCH_PUSH_MARGIN_M)
        tol = dict(TOUCH_GOAL_TOL)
        if pt.get("position_tolerance") is not None:
            tol = {"position_tolerance": float(pt["position_tolerance"]), "orientation_tolerance": float(pt.get("orientation_tolerance") or tol["orientation_tolerance"])}
        g = {"kind": "translate", "along_push_m": round(gap, 5), **tol, "stop_gap_m": round(gap, 5), "source": "the record's push distance - push_margin"}
        if pt.get("position") is not None:
            d = np.asarray(rec["direction"], dtype=float)
            d[2] = 0.0
            d = d / max(float(np.linalg.norm(d)), 1e-12)
            g["tro_along_push_m"] = round(float((np.asarray(pt["position"]) - T0[:3, 3]) @ d), 5)
        return g
    if entry.get("arc") is not None:
        tol = float(pt["orientation_tolerance"]) if pt.get("orientation_tolerance") is not None else AXIAL_GOAL_TOL["orientation_tolerance"]
        if pt.get("orientation") is not None:
            Rg = S.pose_mat([0.0, 0.0, 0.0], pt["orientation"])[:3, :3] @ T0[:3, :3].T
            return {"kind": "turn", "turn_rad": round(math.atan2(Rg[1, 0], Rg[0, 0]), 6), "orientation_tolerance": tol, "source": "the TRO's orientation target",
                    "arc_turn_rad": entry["arc"]["turn_rad"]}
        return {"kind": "turn", "turn_rad": entry["arc"]["turn_rad"], "orientation_tolerance": tol, "source": "the record's arc: init -> goal eef turn"}
    return None


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

    cfg = ContactValidateConfig(diagnose_screened=bool(getattr(a, "diagnose_screened", False)), skip_validated=bool(getattr(a, "skip_validated", False)),
                                 screen_as_rank=bool(getattr(a, "screen_as_rank", False)), per_group_valid=int(getattr(a, "per_group_valid", 0) or 0),
                                 per_group_min=int(getattr(a, "per_group_min", 2) or 2), per_group_max=int(getattr(a, "per_group_max", 8) or 8),
                                 revalidate=bool(getattr(a, "revalidate", False)))
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
            world.set_variant((lib_k.get("validate") or {}).get("variant"))
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
                                                  "valid": v.count("valid"), "invalid": v.count("invalid"), "unreachable": v.count("unreachable"), "skipped": v.count(None),
                                                  **({"groups": {k: lib_k["validation_groups"][k] for k in ("n_groups", "n_short", "valid_hist")}}
                                                     if lib_k.get("validation_groups") else {})})
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
    ap.add_argument("--slab-friction", type=float, default=None,
                    help="DIAGNOSTIC (run E, touching_02's tipping): a physics material of this static = dynamic friction on the slab; default the PhysX default")
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
            if a.slab_friction is not None:
                world.set_slab_friction(obj, a.slab_friction)
            phys = world.physics_facts(obj)
            res = dict(world.replay(sit, obj, geom), physics=phys)
        except Exception as e:  # noqa: BLE001
            import traceback

            res = {"error": repr(e), "tb": traceback.format_exc()[-1500:]}
        finally:
            world.unload(obj)
        print("GB_CONTACT_REPLAY " + _json.dumps({"instance": os.path.basename(path), **res}, default=str), flush=True)
    import omnigibson as og

    og.shutdown()


def probe_cli(argv) -> None:
    """i7-brace 5w.3(b) (owner 20261108h, diagnosis only): ``validate_contacts probe --root R --glib G --jobs J --only ID.. --per-object K
    --conds native assist assist_marked awake tp_pose tp_state tp_state_newsa --speeds 0.003 0.01 0.1 --out OUT.jsonl``: for the first K VALID brace
    entries of each library, :meth:`KitContactWorld.probe_wake` per condition x approach speed; one JSON line per probe. The libraries are
    read, never written."""
    import json as _json

    from ispatialgym.grasping import contact_library as CL

    ap = argparse.ArgumentParser(prog="validate_contacts probe")
    ap.add_argument("--root", required=True)
    ap.add_argument("--glib", required=True)
    ap.add_argument("--jobs", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--arm", default="left")
    ap.add_argument("--per-object", type=int, default=1)
    ap.add_argument("--conds", nargs="+", default=["assist", "native", "awake"])
    ap.add_argument("--speeds", nargs="+", type=float, default=[0.1])
    ap.add_argument("--tp-speed", type=float, default=0.035, help="the approach speed of the teleport conditions")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = ContactValidateConfig()
    ids = [j["id"] for j in CL._jobs(a.jobs, a.only, None)] if a.jobs else list(a.only or [])
    world = KitContactWorld(cfg, a.arm)
    out = open(a.out, "a")
    for eid in ids:
        path = CL.library_path(a.root, "brace", eid)
        lib_k = CL.load_library(path)
        job = library_job(lib_k, eid)
        mlib, _rid, geom = CL.load_object(a.glib, job)
        world.set_support_height((lib_k.get("validate") or {}).get("support_z"))
        world.set_variant((lib_k.get("validate") or {}).get("variant"))
        obj = world.load(dict(job, rest_hint_wxyz=None), os.path.dirname(mlib.path))
        obj._isg_geom = geom
        rest_q = np.asarray(lib_k["object"]["rest"]["quat_xyzw"], dtype=float)
        try:
            ents = [e for e in lib_k["entries"] if (e.get("validation") or {}).get("verdict") == "valid" and not wall_entry(e)][: a.per_object]
            for e in ents:
                q0 = S.quat_mul(S.quat_yaw(math.radians(float(e.get("yaw_deg", 0.0)))), rest_q)
                rest, state = world.settle(obj, geom, q0)
                T_obj = S.pose_mat(rest["world_position"], rest["world_quat_xyzw"])
                plan = world.plan_entry(obj, geom, e, T_obj, a.arm)
                if plan is None or plan.get("q") is None:
                    out.write(_json.dumps({"id": eid, "class": e["class"], "error": "no_placement"}) + "\n")
                    continue
                todo = [(c, v) for c in a.conds if not c.startswith("tp_") for v in a.speeds] + [(c, a.tp_speed) for c in a.conds if c.startswith("tp_")]
                for c, v in todo:
                    try:
                        r = world.probe_wake(obj, e, plan, state, rest, a.arm, c, v)
                    except Exception as ex:  # noqa: BLE001
                        r = {"cond": c, "speed_m_s": v, "error": repr(ex)[:300], "tb": traceback.format_exc()[-800:]}
                    out.write(_json.dumps({"id": eid, "class": e["class"], "yaw": e.get("yaw_deg"), "family": e.get("family"), **r}, default=str) + "\n")
                    out.flush()
        finally:
            world.unload(obj)
    out.close()
    print("GB_CONTACT_PROBE_DONE", flush=True)
    import omnigibson as og

    og.shutdown()


def main(argv=None):
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "replay":
        replay_cli(argv[1:])
        return
    if argv and argv[0] == "probe":
        probe_cli(argv[1:])
        return
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--kind", required=True, choices=["push", "brace"])
    ap.add_argument("--glib", required=True)
    ap.add_argument("--arm", default=None, choices=["right", "left"], help="default: push right, brace left")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--diagnose-screened", action="store_true", help="DIAGNOSTIC: validate the hand_overlap-screened entries (their own root)")
    ap.add_argument("--screen-as-rank", action="store_true", help="owner item 3 switch (default off): validate the hand_overlap-screened entries "
                    "after the live ones, counted, instead of marking them screened")
    ap.add_argument("--skip-validated", action="store_true", help="keep the entries an earlier run validated (not re-run, not counted against max_entries)")
    ap.add_argument("--per-group-valid", type=int, default=0, help="owner A6: STRATIFIED - validate per (yaw x end x spin) group until this many "
                    "are valid (0: off, the max_entries cap); without --skip-validated the run is fresh (unreached entries lose old verdicts)")
    ap.add_argument("--revalidate", action="store_true", help="owner 3.3: re-run exactly the entries an earlier run validated (old record -> "
                    "validation_prev), uncapped: the per-entry verdict changes of a validator fix")
    ap.add_argument("--per-group-min", type=int, default=2, help="a group with fewer valid entries is reported short")
    ap.add_argument("--per-group-max", type=int, default=8, help="at most this many entries tried per group (kept verdicts count)")
    ap.add_argument("--jobs", default=None, help="the build manifest: validate the shard's job ids (the build's partition)")
    validate_cli(ap.parse_args(argv))


if __name__ == "__main__":
    main()
