"""The v2 executor loop (``oracle_primitives_design.md`` §3.8; TODO F1 / F3 / F4).

One loop serves the oracle / replay / policy modes of §6.5::

    for action in actions:
        checkpoint = snapshot(simulator state, executor state)
        for attempt in (0, 1):
            verdict = run(action)                 # steps the env; the QA monitors watch every step
            if verdict is None: verdict = action.postcondition(state)
            if verdict is None: break
            restore(checkpoint); log(action, verdict, attempt)
        else: fail the episode with verdict
        record action.annotate(sidecar)
    epilogue: binary success -> max(50, 20 % of the action steps) no-op -> pose-settled -> goal image ->
              if success within tolerance: TRO goal poses overridden with the settled values

* **Stepping is the executor's**: :meth:`Executor.step` is the only ``env.step(action, n_render_iterations=1)``;
  the recorder and the QA monitors see the same frames in all three modes.
* **Checkpoint contents** (:class:`Checkpoint`): the serialized simulator state (``og.sim.dump_state(serialized=True)``
  with the dump filter off) and the executor state — the Layer-1 hold targets / gripper commands / trunk scalars /
  tracker index (``Motions.export_state``), the backend's attached objects, the assisted-grasp release / grasp
  window counters, and the last observation (re-paired with the first action after a rollback).
* **Assisted-grasp round-trip (§8 open item, verified in code)**: OmniGibson's ``Robot._dump_state`` /
  ``_load_state`` (and ``serialize`` / ``deserialize``) carry ``ag_obj_constraint_params`` per arm and the controller
  state; ``_load_state`` releases / re-creates the AG joint so ``_ag_obj_in_hand`` and ``_ag_obj_constraints`` are
  restored. Not carried: the transient ``_ag_release_counter`` / ``_ag_grasp_counter`` windows and everything the
  planner knows (``R1ProCuRobo._attached``). :meth:`Executor.restore` therefore re-applies the counters and the
  backend attachments explicitly, *verifies* the in-hand objects after ``load_state`` and re-creates a missing AG
  joint from the checkpoint's constraint params (reported in ``checkpoint_reports``), then applies the restored hold
  targets with one recorded hold step (the PhysX drive targets are not part of the state; without that step the
  first physics step would pull the joints toward the failed attempt's last targets).
* **The epilogue belongs to the executor**: ``not_reached`` / ``tolerance`` / ``not_settled`` verdicts, the goal image
  after ≥ 3 renders (10) with the physics-synced sensor pose, and the TRO goal override (:meth:`goal_override`)
  computed here and written by ``run_isg``.

Simulator-free by construction: ``env`` / ``task`` / ``sim`` are duck-typed and the tests drive the loop with fakes.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym import ASSETS_ROOT
from ispatialgym import single_env as SE
from ispatialgym.fastflags import fast_on, fast_value  # 2026-10-31 train-fast switches
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle import sleep_assist as SA
from ispatialgym.oracle.actions import Action, Brace, Grasp, Navigate, Place, Push, Transition
from ispatialgym.oracle.execute import (aim_push_past_goal, check_postcondition, forbidden_contacts, fov_monitored, intended_contacts, is_manipulation, offset_brace,
                                        run_action, shorten_push)
from ispatialgym.oracle.motions import Motions, SkillError
from ispatialgym.oracle.qa import QA, RULE_METRICS, ArmStutter, CameraModel, Monitors, QAStats, Verdict, category_of, is_hard_qa
from ispatialgym.oracle.recorder import PHASE_GOAL, PHASE_ORACLE, PHASE_SETTLE, Recorder, to_uint8_rgb
from ispatialgym.robot import conventions as C

log = logging.getLogger(__name__)

AGENT_INSTANCE = "agent.n.01_1"

#: cylinder tags of the placed-orientation demo gate (owner 2026-09-16 17:10 item 4 fallback): ``{"<category>/<model>":
#: {"axis": "x"|"y"|"z", "tags": [...], ...}}`` built by ``scripts/symmetry/revolution_axes.py`` from the ICL symmetry
#: survey — the models with a body-of-revolution tag (``xinf`` / ``yinf`` / ``zinf``) about their long axis
REVOLUTION_AXES_PATH = os.path.join(ASSETS_ROOT, "icl", "revolution_axes.json")


@functools.lru_cache(maxsize=None)
def revolution_axes(path: str = REVOLUTION_AXES_PATH) -> Dict[str, Dict[str, Any]]:
    """The revolution-axis table (empty when the asset is missing or unreadable — then every object keeps the full error)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        log.warning("revolution axes table %s not readable (%r): placed objects keep the full orientation error", path, e)
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, dict) and v.get("axis") in ("x", "y", "z")}


def revolution_axis(category: str, model: str, path: str = REVOLUTION_AXES_PATH) -> Optional[Dict[str, Any]]:
    """The table entry of ``category/model`` (``axis`` = the object-frame long axis it may roll about), None otherwise."""
    return revolution_axes(path).get(f"{category}/{model}")


#: owner 2026-11-08 (answers b, item 2; i7-contact3): a braced push whose brace is a TOP PRESS (``contacts.brace.kind == "top_press"``,
#: small-scale articulation's push-close) and whose residual lies within the task's tolerance is accepted as it is - no improvement rung, so
#: ``offset_brace`` never raises the press off the object (close_02: the +10 mm rung lifted the press and the carton slid until the
#: 10 mm object_tilt QA). Every other brace kind, and every push out of tolerance, keeps the retry ladder. "0" = the ladder for all.
#: Default "1" since the 2026-11-08 A/B (the owner decided the rule): close_02 at 300006-009, off -> on 0 / 4 -> 3 / 4 (fresh v2only) and
#: 0 / 4 -> 3 / 4 (the int1_6 instances re-collected); object_tilt 3 -> 0, the carton's peak 10.0-11.7 -> 3.3-6.2 mm.
TOP_PRESS_ACCEPT_ENV = "ISPATIALGYM_TOP_PRESS_ACCEPT"
TOP_PRESS_ACCEPT_DEFAULT = "1"


def top_press_accept_on() -> bool:
    """:data:`TOP_PRESS_ACCEPT_ENV` (unset / empty: :data:`TOP_PRESS_ACCEPT_DEFAULT`) is "1"."""
    return (os.environ.get(TOP_PRESS_ACCEPT_ENV) or TOP_PRESS_ACCEPT_DEFAULT).strip() == "1"


@dataclass
class ExecutorConfig:
    n_render_iterations: int = 1
    #: episode step budget (the QA monitors' own budget is ``QAConfig.max_episode_steps``); a program's actions get
    #: ``max_steps - settle_steps`` of it, the rest is its epilogue's
    max_steps: int = 20000
    #: seconds the actions may take (None = no limit): a stream (replay / policy) ends there and the task's metrics judge
    #: the episode, as at the end of its steps; a program gets a ``timeout`` verdict
    max_wall_s: Optional[float] = None
    #: one retry from the checkpoint (§3.8)
    retries: int = 1
    #: the epilogue of a program: exactly this many no-op steps (``run_isg --settle-steps``, the task's ``pose_settled`` window),
    #: reserved in ``max_steps``; a stream has none
    settle_steps: int = 45
    #: what ``success`` follows (``run_isg --success-metric``): the run mode's ``geometric`` (default) or ``symbolic`` success
    success_metric: str = "geometric"
    #: a stream ends at the run mode's geometric success (the episode's termination; policy mode)
    stop_on_success: bool = False
    #: (2026-09-22, 0918-000750) before the no-op window every arm's hold target is relaxed onto its measured joints over
    #: ``epilogue_relax_steps`` steps (``Motions.relax_arms``; a pressing fist of an open ``press_hold`` is left alone): a
    #: drive still fighting an unseen contact at the end of the program (the folded elbow on its own upper arm: joint
    #: velocities of 2.2 rad/s reported for the whole window, ``pose_settled(agent)`` never true) stops pushing, so the
    #: window judges the robot at rest
    epilogue_relax_arms: bool = True
    epilogue_relax_steps: int = 6
    #: goal image
    goal_image: bool = False
    goal_image_renders: int = 10
    #: brace retry ladder (owner 2026-09-16 05:26 / 05:39): retries of a braced push whose residual is out of tolerance,
    #: with the brace moved by these offsets (m) outward (short) / inward (overshoot) along its recorded direction
    #: (rush 2026-09-17: 4 retries; when the residual has flipped sign across retries the next offset bisects the last
    #: overshoot / short offsets - the painting demo alternated -0.27 -> +0.14 rad on the two-rung ladder)
    brace_retries: int = 4
    brace_retry_offsets_m: Tuple[float, ...] = (0.005, 0.010, 0.015, 0.020)
    #: (rush) the ladder keeps climbing while retries remain until the residual lies within aim_frac x tolerance of the aim
    #: centre; for arcs the centre is biased to the overshoot side by aim_bias_rad (the two-pad squeeze relaxes toward
    #: the short side by ~0.07 rad when the pads leave: painting 0917-220043 push +0.053 -> final 0.129)
    brace_retry_aim_frac: float = 0.4
    brace_retry_aim_bias_rad: float = 0.05
    #: (rush) an arc overshoot with no brace catch shortens the recorded arc (shorten_push) instead of moving the brace
    push_shorten_on_free_overshoot: bool = True
    #: (owner 2026-09-26) the mirror for a short with no brace catch: out of tolerance -> the arc is lengthened and the pad aims
    #: past the goal by the measured sag (aim_push_past_goal); in tolerance -> accepted as it is (the epilogue's settled check decides)
    push_extend_on_free_short: bool = True
    #: ``run_isg --qa`` (``ispatialgym.oracle.qa.QA``): the demo QA metrics, logged in ``result.qa``; a hard metric fails an
    #: oracle demo at once (no retry), a stream only logs them. The oracle's retry rules (``Monitors``) run either way.
    qa: bool = True
    #: the placed-orientation fact of an oracle demo (``result.placed_orientation``, no verdict): every placed object (place /
    #: drop / attach, not pushes) against its recorded orientation target's tolerance (``orientation_tolerance``, 0.2 rad
    #: default; nearest equivalence, the evaluator's own error). Cylinders (owner decision 2026-09-16 17:10 item 4): an object
    #: with a continuous symmetry about its long axis per the ICL symmetry survey (``assets/icl/revolution_axes.json``:
    #: marker, pencil, bottles, cans ...) is compared **modulo rotation about that axis**; ``demo_revolution_axes`` switches
    #: that equivalence off (full error for everything)
    demo_orientation_tolerance_default: float = 0.2
    demo_revolution_axes: bool = True
    #: renders after a checkpoint restore before the (recorded) hold step, so no frame of the rolled-back attempt
    #: survives in the video / dataset (:meth:`Executor.render_restored`). ``restore_renders`` frames are always
    #: rendered; from then on the head camera is read after every render until its image is **fresh** - closer to the
    #: checkpoint's own observation than to the last observation of the rolled-back attempt - and **settled** - within
    #: ``restore_frame_tol`` (mean abs rgb difference, 0-255 scale) of the checkpoint observation, or a new frame whose
    #: distance to the checkpoint observation shrank by at most ``restore_frame_gain_tol`` since the previous new frame
    #: (the RTX temporal accumulation ghosts the rolled-back scene into the first fresh frames and fades it
    #: geometrically: 0924-171522 set_torso 28.95 stale -> 3.51 -> 2.86 -> 2.72 -> 2.52 ..., 0924-170128 navigate
    #: 61.7 stale -> 13.0 -> 6.0 -> 5.7; the steady grain that remains varies by ~1.0 per frame and only converges
    #: over tens of frames, so neither a small absolute distance nor a small frame-to-frame change can be waited for)
    #: - or ``restore_renders_max`` frames were rendered (then ``checkpoint_reports[].restore_renders.converged =
    #: False``; a warning when the last frame was not even fresh). A fixed count is not enough: the first read after
    #: ``load_state`` returned the rolled-back scene whether 1 or 3 frames had been rendered before it (0924-164130 /
    #: 0924-170128 / 0924-171522) - the annotator hands frames out with a load-dependent lag of 1-2 *reads* on top of
    #: the 1-frame pose -> renderer lag (``sampling.sim_utils.SimWorld.seg_instance``); with the 3 fixed renders of
    #: 2026-09-16 the hold step still recorded the failed attempt's last arm pose in ~half of the retried demos of
    #: 09-18 .. 09-22 (the one-frame flash of the hand at the failed grasp in 0919-062149 / 0920-193713 / 0921-021337 ...).
    #: Minimum 30 (owner decision 2026-09-24 17:30): at the 5-8 renders the settle rule stops at, the hold step's frame is
    #: still ~2 mean-abs softer than its neighbours (the accumulation grain of the reset scene: a visible one-frame blink
    #: in 0924-165803 / 0924-170537); the grain converges over ~30 frames (~8 s at 1080 x 1080, not recorded: renders are
    #: not steps, a retry adds exactly the one hold step to the demo)
    restore_renders: int = 30
    restore_renders_max: int = 40
    restore_frame_tol: float = 1.0
    restore_frame_gain_tol: float = 0.25
    #: TRO goal override thresholds of a target without a tolerance of its own (every loaded target has the profile's)
    override_min_pos: float = 0.01
    override_rot: float = 0.20
    #: record the serialized sim state every step (needed for the BEHAVIOR-replayable hdf5)
    record_state: bool = True
    #: checkpoints, rollbacks and the recorded per-step states go through the env's own scene (``scene.dump_state`` /
    #: ``load_state``) instead of the simulator: in a one-scene simulator the serialized state is the same tensor
    #: (``og.sim.serialize`` concatenates the scenes'), and with several scenes in one simulator (ispatialgym.multienv) it is
    #: the only state that belongs to this env. Default: ISPATIALGYM_PER_SCENE_STATE=1 (off when unset)
    per_scene_state: bool = field(default_factory=lambda: os.environ.get("ISPATIALGYM_PER_SCENE_STATE", "").lower() in ("1", "true", "yes"))
    #: physics-only collection (:mod:`ispatialgym.oracle.physics_only`): the episode steps physics without rendering, with
    #: Fabric's transforms synced after every step and after a rollback's load (camera poses read as in a rendered run),
    #: and a rollback does not re-render. Default: ISPATIALGYM_PHYSICS_ONLY=1 (off when unset)
    physics_only: bool = field(default_factory=lambda: os.environ.get("ISPATIALGYM_PHYSICS_ONLY", "").lower() in ("1", "true", "yes"))


class SupportLoad:
    """QA **fact** (owner 2026-09-16 23:00, no verdict): how much the fingers load the support during a demo — the
    support's pose drift over the episode (displacement and up-axis tilt vs its pose at the first step, max and at the
    end), the control steps in which a finger link of either arm touches a support body (by action name, from
    ``RigidContactAPI``) and the peak contact force between the finger links and the support (the contact force matrix
    of the last physics substep of each control step, N). ``supports`` = the tracked objects no program action
    manipulates (the table under a tabletop task); ``fixed_base`` records the object's flag in the loaded scene (a
    dynamic table is what the fingers can rock). Written as ``meta.support_disturbance``; every simulator call is guarded
    so a missing contact view leaves the pose drift alone."""

    def __init__(self, robot, supports: Dict[str, Any], scene_idx: int = 0):
        self.robot = robot
        self.supports = dict(supports)
        self.scene_idx = int(scene_idx)
        self.pose0: Dict[str, Tuple[th.Tensor, th.Tensor]] = {}
        self.max_disp: Dict[str, float] = {}
        self.max_tilt: Dict[str, float] = {}
        self.end_disp: Dict[str, float] = {}
        self.end_tilt: Dict[str, float] = {}
        self.contact_steps: Counter = Counter()
        self.contact_steps_total = 0
        self.peak_force: float = 0.0
        self.peak_force_action: Optional[str] = None
        self.peak_force_by_action: Dict[str, float] = {}
        self.fixed_base: Dict[str, Optional[bool]] = {}
        self.contact_api = None
        self.contact_failed = False
        self.force_failed = False
        self.finger_links: List[Any] = []
        self.support_links: List[Any] = []
        self.steps = 0
        for name, obj in self.supports.items():
            try:
                self.fixed_base[name] = bool(getattr(obj, "fixed_base")) if getattr(obj, "fixed_base", None) is not None else None
            except Exception:  # noqa: BLE001
                self.fixed_base[name] = None
            try:
                self.support_links += list(getattr(obj, "links", {}).values())
            except Exception:  # noqa: BLE001
                pass
        try:
            for arm in M.ARMS:
                self.finger_links += list(getattr(robot, "finger_links", {}).get(arm, []))
        except Exception:  # noqa: BLE001
            self.finger_links = []
        if not self.finger_links or not self.support_links:
            self.contact_failed = True  # no finger / support bodies to query (no simulator): contacts not measured

    def _pose(self, obj) -> Optional[Tuple[th.Tensor, th.Tensor]]:
        try:
            p, q = obj.get_position_orientation()
            return p.float().clone(), q.float().clone()
        except Exception:  # noqa: BLE001
            return None

    def observe(self, action_name: Optional[str]) -> None:
        self.steps += 1
        for name, obj in self.supports.items():
            pose = self._pose(obj)
            if pose is None:
                continue
            if name not in self.pose0:
                self.pose0[name] = pose
            p0, q0 = self.pose0[name]
            disp = float(th.linalg.norm(pose[0] - p0))
            up0, up = C.quat_rotate(q0, th.tensor([0.0, 0.0, 1.0])), C.quat_rotate(pose[1], th.tensor([0.0, 0.0, 1.0]))
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(th.dot(up, up0))))))
            self.max_disp[name] = max(self.max_disp.get(name, 0.0), disp)
            self.max_tilt[name] = max(self.max_tilt.get(name, 0.0), tilt)
            self.end_disp[name], self.end_tilt[name] = disp, tilt
        if not self.finger_links or not self.support_links or self.contact_failed:
            return
        action = action_name or "epilogue"
        try:
            if self.contact_api is None:
                from omnigibson.utils.usd_utils import RigidContactAPI

                self.contact_api = RigidContactAPI
            pairs = self.contact_api.get_contact_pairs(self.scene_idx, self.finger_links, self.support_links, current_only=True)
        except Exception as e:  # noqa: BLE001
            log.warning("support load monitor: contact query failed (%r); contact steps not measured", e)
            self.contact_failed = True
            return
        if not pairs:
            return
        self.contact_steps[action] += 1
        self.contact_steps_total += 1
        if self.force_failed:
            return
        try:
            import omnigibson as og

            view = self.contact_api._CONTACT_VIEW[self.scene_idx]
            forces = view.get_contact_force_matrix(dt=og.sim.get_physics_dt())
            rows = self.contact_api.get_contact_row_indices(self.scene_idx, self.finger_links)
            cols = self.contact_api.get_contact_col_indices(self.scene_idx, self.support_links)
            f = th.as_tensor(forces)[rows][:, cols]
            peak = float(th.linalg.norm(f.reshape(-1, 3), dim=1).max()) if f.numel() else 0.0
        except Exception as e:  # noqa: BLE001
            log.warning("support load monitor: contact force read failed (%r); peak force not measured", e)
            self.force_failed = True
            return
        self.peak_force_by_action[action] = max(self.peak_force_by_action.get(action, 0.0), peak)
        if peak > self.peak_force:
            self.peak_force, self.peak_force_action = peak, action

    def report(self) -> Dict[str, Any]:
        if not self.supports:
            return {}
        by_action = {a: {"contact_steps": int(n), "peak_force_N": round(self.peak_force_by_action.get(a, 0.0), 2) if not self.force_failed else None}
                     for a, n in sorted(self.contact_steps.items(), key=lambda kv: -kv[1])}
        return {
            "supports": sorted(self.supports),
            "fixed_base": dict(self.fixed_base),
            "max_displacement_m": round(max(self.max_disp.values(), default=0.0), 5),
            "end_displacement_m": round(max(self.end_disp.values(), default=0.0), 5),
            "max_tilt_deg": round(max(self.max_tilt.values(), default=0.0), 4),
            "end_tilt_deg": round(max(self.end_tilt.values(), default=0.0), 4),
            "per_support": {n: {"max_displacement_m": round(self.max_disp.get(n, 0.0), 5), "max_tilt_deg": round(self.max_tilt.get(n, 0.0), 4)} for n in self.supports},
            "contact_steps": None if self.contact_failed else int(self.contact_steps_total),
            "contact_steps_by_action": None if self.contact_failed else {a: int(n) for a, n in self.contact_steps.items()},
            "peak_force_N": None if (self.contact_failed or self.force_failed) else round(self.peak_force, 2),
            "peak_force_action": None if (self.contact_failed or self.force_failed) else self.peak_force_action,
            "by_action": None if self.contact_failed else by_action,
            "steps_observed": int(self.steps),
            "force_sampling": "last physics substep of each control step",
        }


@dataclass
class Checkpoint:
    action_index: int
    step: int
    sim_state: th.Tensor
    exec_state: Dict[str, Any]
    attached: Dict[str, str]
    ag_in_hand: Dict[str, Optional[str]]
    ag_params: Dict[str, Any]
    ag_counters: Dict[str, Dict[str, Any]]
    obs: Any = None
    info: Any = None
    recorder_steps: int = 0
    env_step: Optional[int] = None
    task_tracking: Dict[str, Any] = field(default_factory=dict)
    #: the gripper-in-FOV rule's spans open at the checkpoint (``Monitors.fov_spans``)
    fov_spans: Dict[str, bool] = field(default_factory=dict)
    #: ``QA.snapshot``: the gripper-in-view spans open at the checkpoint, the articulated roots' readings, the step streaks
    qa: Optional[Dict[str, Any]] = None


#: ``ISpatialGymTask`` tracking that a rollback must restore (its symbolic / geometric success trackers, a copy)
TASK_TRACKING_ATTRS = ("metrics_state",)


#: ctx keys the execute functions write for the record (meta.json ``actions[].notes``)
NOTE_KEYS = ("push_residual", "push_pivot_offset_m", "push_object_drift_m", "approach_finish", "descent_finish", "lift_finish", "retreat_finish", "pre_pose_standoff", "plan_relaxed_links", "push_axis", "carry_leg", "idle_arms_moved", "arms_posture", "fingers_set", "plan_ignored", "object_motion",
             "camera_error_before_correction", "camera_error", "attached", "preshape_opening_m", "closure_rate_m_per_step", "follow", "attach", "reorient_finish", "approach_standoff", "partner_shift", "pre_grasp_transit",
             "pre_grasp_cartesian", "navigate_arms_reposed", "navigate_arms_source", "retreat_partial_m", "retreat_start_contacts", "retreat_start_offenders", "brace_allowed",
             "push_strokes", "push_contact_form")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, th.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    return str(x)


#: attachment (2026-09-20 22:30): the place re-aimed for the part's live in-hand pose at the pre-place (execute._reaim_for_inhand_pose)
NOTE_KEYS = NOTE_KEYS + ("inhand_reaim",)
#: room door push-close: the per-chunk record of execute.execute_push_close
NOTE_KEYS = NOTE_KEYS + ("push_close",)
#: owner 2026-10-12 (nextto_large): the base push of execute.execute_base_push - pushed distance, sideways drift, NextTo at the end
NOTE_KEYS = NOTE_KEYS + ("base_push",)
NOTE_KEYS = NOTE_KEYS + ("protection",)  # oracle.protection (owner 2026-09-26): the transits of the action and their protected objects
NOTE_KEYS = NOTE_KEYS + ("repose",)  # block design: the joint-space re-pose before a grasp after a base move (execute.repose_arm)
NOTE_KEYS = NOTE_KEYS + ("retreat_planned_fallback",)  # owner 2026-09-26 (axial 0926-061151): a failed retreat leg replaced by a planned transit
NOTE_KEYS = NOTE_KEYS + ("ag_joint_type",)  # the assisted-grasp joint a handle grasp got (the task's rule for links behind a movable joint)
NOTE_KEYS = NOTE_KEYS + ("retreat_slide_first",)  # root_moved (2026-09-27, carton lever): the lever fist slides out before it lifts
NOTE_KEYS = NOTE_KEYS + ("retreat_escape", "retreat_witness_dropped")  # the push-under retreat (owner 2026-10-13 ~04:30): its leg, a stale witness dropped
#: first-demo v2.1 A3 (bar soap 0927-034001): a library pinch lifted off the support under it, a support place raised for the in-hand pose
NOTE_KEYS = NOTE_KEYS + ("grasp_support_lift", "place_inhand_raise")
#: owner audit 2026-10-09: a grasp approach stopped on the support, a free-hanging drop released at once, a next-to / touching place moved
#: off the partner for the release and the push extended by it
NOTE_KEYS = NOTE_KEYS + ("approach_support_stop", "partner_release_clearance", "push_place_shift_extended_m", "drop_lowered")
#: oracle-fix 2026-11-03: the touch-stopped descent, the in-hand place re-aim and the push re-contact rounds
NOTE_KEYS = NOTE_KEYS + ("place_touch_stop", "place_inhand_reaim", "push_recontact")
#: first-demo v2.1 A2 (block design 0927-040243): a travel leg kept an arm in its IK branch; the next contact ran on the kept branch
NOTE_KEYS = NOTE_KEYS + ("branch_kept", "witnesses_rebranched")
#: first-demo v2.1 (sandal on pods): the arm moved into the pre-grasp witness's posture before an approach that tracks only from there
NOTE_KEYS = NOTE_KEYS + ("witness_posture",)
#: first-demo v3.0 F4 (cork 0928-041525): an attach place kept the holding arm's hold from the descent until the attachment existed
NOTE_KEYS = NOTE_KEYS + ("attach_hold",)
#: first-demo v3.0 F1 (world vertical 0928-043205): a transition kept a free arm as it was for its brace / push at the arriving stance
NOTE_KEYS = NOTE_KEYS + ("contact_arms_kept",)
#: first-demo v3.0 F5 (nextto marker 0928-050212): a grasp transit whose plan dipped below both ends ran as a Cartesian leg
NOTE_KEYS = NOTE_KEYS + ("transit_dip",)
#: owner 2026-10-11: an action on an object on an interior shelf kept the support in the world but for the board pieces under it
NOTE_KEYS = NOTE_KEYS + ("interior_support",)
#: owner 2026-10-12 (A2b): the parts / distractors a grasp may touch - on the source tray after the grasp / lift, how far they and the
#: target moved
NOTE_KEYS = NOTE_KEYS + ("may_touch",)
#: owner 2026-10-12 (IORS touching large-scale): where a push stopped (goal reached / leg end, the object's position, the residual) and
#: how far a followed partner moved (execute.push_stop_facts)
NOTE_KEYS = NOTE_KEYS + ("push_stop",)
#: a straight push's lowered pre-contact (``execute.push_lowered_pre``): where the transit went and the first contact pose under it
NOTE_KEYS = NOTE_KEYS + ("push_pre_lowered",)
#: the collision world the transit to a push pre-contact was planned in (``execute.reach_push_pre``): the pushed object present or left out, why
NOTE_KEYS = NOTE_KEYS + ("push_transit",)
#: owner 2026-10-13 (disembedding): an axis-aligned leg's driven heading / corridor and its moves (``execute.post_navigate``)
NOTE_KEYS = NOTE_KEYS + ("axis_leg",)
#: owner 2026-10-14: a push's recorded pad speed (``execute.push_speed_of``) and, per base nudge with a brace standing, the brace hand held
#: in the world (``execute._NudgeBraceHold``: its drift, the IK misses, why the nudge stopped)
NOTE_KEYS = NOTE_KEYS + ("push_speed_m_s", "push_nudge_brace_hold")
NOTE_KEYS = NOTE_KEYS + ("preshape_set", "preshape_kept_from_transition_m")  # oracle-fix 2026-11-06: the shelf back-off's pre-shape (transition / grasp)
NOTE_KEYS = NOTE_KEYS + ("push_gripper",)  # oracle-fix 2026-11-06: an open-hand push's opening (execute_push; touching's generation.push_gripper_open)
NOTE_KEYS = NOTE_KEYS + ("overpath", "transit_book_clearance")  # i7-overpath 2026-11-08: the overhang pick's recorded transit / its book clearance


@dataclass
class ActionRecord:
    index: int
    name: str
    attempts: List[Optional[Dict[str, Any]]] = field(default_factory=list)
    verdict: Optional[Dict[str, Any]] = None
    step_start: int = 0
    step_end: int = 0
    targets_keys: List[str] = field(default_factory=list)
    #: what the execute function reported about its closure (push residual, approach finish, idle arms moved, ...)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "name": self.name, "attempts": list(self.attempts), "verdict": self.verdict,
                "step_start": self.step_start, "step_end": self.step_end, "n_steps": self.step_end - self.step_start,
                "targets_keys": list(self.targets_keys), "notes": _jsonable(self.notes)}


# ==========================================================================================
# privileged state
# ==========================================================================================


class SimState:
    """The privileged view the actions' postconditions, ``targets()`` and the epilogue read (never write)."""

    def __init__(self, env, task, robot, cam_key: Optional[str] = None, cam_in_zed=None):
        self.env, self.task, self.robot, self.cam_key = env, task, robot, cam_key
        self.cam_in_zed = cam_in_zed

    def object(self, name: str):
        scope = getattr(self.task, "object_scope", None) or {}
        ent = scope.get(name)
        if ent is None and getattr(self.env, "scene", None) is not None:
            ent = self.env.scene.object_registry("name", name, None)
        if ent is None or not hasattr(ent, "get_position_orientation"):
            raise KeyError(f"no physical scene object for {name!r}")
        return ent

    def object_pose(self, name: str) -> Tuple[th.Tensor, th.Tensor]:
        p, q = self.object(name).get_position_orientation()
        return p.float().clone(), q.float().clone()

    def object_velocity(self, name: str) -> float:
        obj = self.object(name)
        try:
            return float(th.linalg.norm(obj.get_linear_velocity().float()))
        except Exception:  # noqa: BLE001
            return 0.0

    def eef_pose(self, arm: str) -> Tuple[th.Tensor, th.Tensor]:
        p, q = self.robot.eef_links[arm].get_position_orientation()
        return p.float().clone(), q.float().clone()

    def base_pose2d(self) -> Tuple[float, float, float]:
        p, q = self.robot.get_position_orientation()
        return float(p[0]), float(p[1]), C.yaw_of(q.float())

    def pose_target(self, name: str):
        """The TRO pose target of ``name`` (``eval.targets.PoseTarget``: world / relative frame, position, orientation,
        tolerances) or None — what a push closes the loop on (spec §8: the goal, not the arc)."""
        tro = getattr(self.task, "_tro", None)
        meta = getattr(tro, "meta", None)
        return (getattr(meta, "pose_targets", None) or {}).get(name)

    def touching(self, name: str) -> List[Any]:
        """Scene objects of the task scope in contact with ``name`` (its support / partner; OmniGibson ``Touching``).
        What a finishing ``arm_to`` of a grasp approach leaves out of the collision world next to the object itself
        (the sampler's "support ignored for the grasp pose"). ``[]`` without the simulator."""
        try:
            from omnigibson.object_states import Touching
        except Exception:  # noqa: BLE001
            return []
        try:
            obj = self.object(name)
        except KeyError:
            return []
        out: List[Any] = []
        for inst, ent in (getattr(self.task, "object_scope", None) or {}).items():
            if ent is obj or ent is self.robot or not hasattr(ent, "states") or Touching not in getattr(ent, "states", {}):
                continue
            try:
                if bool(obj.states[Touching].get_value(ent)):
                    out.append(ent)
            except Exception:  # noqa: BLE001
                continue
        return out

    #: structural categories a target rests against without a contact report (a painting attached to its wall)
    STRUCTURE_CATEGORIES = ("walls", "floors", "ceilings", "window", "door")

    def contact_supports(self, name: str, margin: float = 0.10, depth: int = 2) -> List[Any]:
        """The objects ``name`` rests in / on / against — what the sampler ignores for that object's contact poses
        (spec §3.6 "support ignored for the grasp pose", disembedding's ``[support, src]`` = the table and the source
        tray, the wall along the whole brace chain of wall alignment): the ``touching`` objects plus the structural
        task-scope objects (walls, floors, ...) whose AABB lies within ``margin`` of the object's, followed
        transitively for ``depth`` levels (the tray's own support, the table, is within cuRobo's activation distance
        of fingertips on the tray floor). Held objects and the robot are never included. ``[]`` without the simulator."""
        held = [o for o in self.held_objects()]
        out: List[Any] = []
        frontier = [name]
        for _ in range(max(1, int(depth))):
            nxt: List[str] = []
            for n in frontier:
                for ent in self._supports_of(n, margin):
                    if any(ent is o for o in out) or any(ent is h for h in held):
                        continue
                    try:
                        if ent is self.object(name):
                            continue
                    except KeyError:
                        pass
                    out.append(ent)
                    nxt.append(ent.name)
            frontier = nxt
            if not frontier:
                break
        return out

    def _supports_of(self, name: str, margin: float) -> List[Any]:
        """Objects that carry ``name``: touching objects whose footprint contains the object's centre (a table, tray,
        bowl, saucer — not a neighbour lying next to it: the nextto partner touched the marker end to end and was
        swept aside by a pre-place plan that ignored it), plus structural objects within ``margin``."""
        try:
            obj = self.object(name)
            lo, hi = obj.aabb
        except Exception:  # noqa: BLE001
            return list(self.touching(name))
        centre = 0.5 * (th.as_tensor(lo).float() + th.as_tensor(hi).float())
        out: List[Any] = []
        for ent in self.touching(name):
            try:
                lo2, hi2 = ent.aabb
                inside = bool(th.all(centre[:2] >= th.as_tensor(lo2).float()[:2] - 0.01)) and bool(th.all(centre[:2] <= th.as_tensor(hi2).float()[:2] + 0.01))
            except Exception:  # noqa: BLE001
                inside = True
            if inside or str(getattr(ent, "category", "")) in self.STRUCTURE_CATEGORIES:
                out.append(ent)
        for inst, ent in (getattr(self.task, "object_scope", None) or {}).items():
            if ent is obj or ent is self.robot or any(ent is o for o in out) or str(getattr(ent, "category", "")) not in self.STRUCTURE_CATEGORIES:
                continue
            try:
                lo2, hi2 = ent.aabb
                gap = th.maximum(th.as_tensor(lo2).float() - th.as_tensor(hi).float(), th.as_tensor(lo).float() - th.as_tensor(hi2).float())
                if float(gap.max()) <= margin:
                    out.append(ent)
            except Exception:  # noqa: BLE001
                continue
        return out

    def hand_contacts(self, arm: str) -> List[Any]:
        """Scene objects the ``arm``'s finger / gripper links are in contact with right now (``RigidContactAPI``, current
        step; the floor and other ground categories left out) — what a leg that starts from a contact is allowed to touch
        along its way (owner 2026-09-24 item 2: the push pad against the partner it just pushed the target onto, the brace
        hand on the braced object). ``[]`` without the simulator / a contact view."""
        try:
            from omnigibson.utils.usd_utils import RigidContactAPI
        except Exception:  # noqa: BLE001
            return []
        scene = getattr(self.env, "scene", None)
        if scene is None or not hasattr(self.robot, "prim_path"):
            return []
        links = set(M.FINGER_LINKS.get(arm, ())) | {M.GRIPPER_LINK.get(arm)}
        prefix = self.robot.prim_path + "/"
        try:
            pairs = RigidContactAPI.get_contact_pairs(scene.idx, [self.robot], None, current_only=True)
        except Exception:  # noqa: BLE001
            return []
        out: List[Any] = []
        for own, other in pairs:
            if not str(own).startswith(prefix) or str(other).startswith(prefix):
                continue
            if str(own).rsplit("/", 1)[-1] not in links:
                continue
            try:
                obj = scene.object_registry("prim_path", str(other).rsplit("/", 1)[0], None)
            except Exception:  # noqa: BLE001
                obj = None
            if obj is None or obj is self.robot or str(getattr(obj, "category", "")) in ("floors", "ceilings", "lawn", "roof"):
                continue
            if not any(obj is o for o in out):
                out.append(obj)
        return out

    def held_object(self, arm: str):
        return getattr(self.robot, "_ag_obj_in_hand", {}).get(arm)

    def held_objects(self) -> List[Any]:
        ag = getattr(self.robot, "_ag_obj_in_hand", {}) or {}
        return [o for o in ag.values() if o is not None]

    def joint_position(self, name: str, joint: str) -> float:
        obj = self.object(name)
        names = list(obj.joints.keys())
        if joint not in names:
            raise KeyError(f"{name} has no joint {joint!r} ({names})")
        return float(obj.get_joint_positions()[names.index(joint)])

    def camera_pose(self) -> Tuple[th.Tensor, th.Tensor]:
        if self.cam_key is None:
            raise KeyError("no head camera key")
        p, q = self.robot.sensors[self.cam_key].get_position_orientation()
        return p.float().clone(), q.float().clone()

    def trunk(self) -> Tuple[float, float]:
        q = self.robot.get_joint_positions()[self.robot.trunk_control_idx].float()
        return C.torso_q_to_translate_tilt(q)

    def attached(self, child: str, parent: str) -> bool:
        from omnigibson.object_states import AttachedTo

        c, p = self.object(child), self.object(parent)
        if AttachedTo not in c.states:
            return False
        return bool(c.states[AttachedTo].get_value(p))

    def attachment_links(self, child: str, parent: str):
        """``(male link of child, female link of parent)`` OmniGibson would join — the first candidate pair whose female
        link is free, in ``AttachedTo._find_attachment_links``'s own order; ``None`` when the pair has no matching
        attachment meta links."""
        from omnigibson.object_states import AttachedTo

        c, p = self.object(child), self.object(parent)
        if AttachedTo not in c.states or AttachedTo not in p.states:
            return None
        cands = c.states[AttachedTo]._get_parent_candidates(p) or {}
        for child_name, parent_names in cands.items():
            for parent_name in sorted(parent_names):
                if p.states[AttachedTo].children.get(parent_name) is None:
                    return c.states[AttachedTo].links[child_name], p.states[AttachedTo].links[parent_name]
        return None

    def attachment_offset(self, child: str, parent: str) -> Optional[Dict[str, Any]]:
        """Live male-vs-female meta-link offset of the attachment pair **modulo the child's recorded symmetry**
        (``predicates.attachment_alignment``, the evaluator's own rule): ``pos_m`` / ``rot_rad`` and ``fix`` = the world
        transform carrying the best equivalent male frame onto the female one (applied to the hand pose it is the
        closed-loop correction of the place; a zinf cap is not turned to the knocked bottle's yaw - 0920-164504 rotated the
        hand 35 deg for nothing and dragged the cap into the neck). ``rot_raw_rad`` keeps the full-SO(3) difference."""
        from ispatialgym.sampling import predicates as PRED

        pair = self.attachment_links(child, parent)
        if pair is None:
            return None
        c, p = self.object(child), self.object(parent)
        a = PRED.attachment_alignment(c, p, with_fix=True)
        if not a.get("fix"):
            return None
        male, female = pair
        mp, _ = (x.float().clone() for x in male.get_position_orientation())
        fp, _ = (x.float().clone() for x in female.get_position_orientation())
        return {
            "pos_m": float(a["pos_m"]),
            "rot_rad": float(a["rot_rad"]),
            "rot_raw_rad": float(a.get("rot_raw_rad", a["rot_rad"])),
            "equivalence": a.get("equivalence"),
            "delta_pos": [round(float(v), 4) for v in (fp - mp).tolist()],
            "male_link": male.body_name,
            "female_link": female.body_name,
            "fix": a["fix"],
        }

    def init_pose(self, name: str) -> Optional[Tuple[th.Tensor, th.Tensor]]:
        """The object's pose at the start of the program (``Executor.run_program`` snapshot), None when unknown."""
        p = (getattr(self, "init_poses", None) or {}).get(name)
        return None if p is None else (p[0].clone(), p[1].clone())

    def attach(self, child: str, parent: str) -> bool:
        """The BEHAVIOR attachment of ``child`` to ``parent`` by the evaluator's own rule (``predicates.attach_if_aligned``,
        owner 2026-10-10): released, in contact with the parent and aligned modulo the child's recorded symmetry, then joined
        in the configured mode; otherwise nothing changes and False is returned. ``self.last_attach`` keeps the alignment record."""
        from omnigibson.object_states import AttachedTo

        from ispatialgym.sampling import predicates as PRED

        c, p = self.object(child), self.object(parent)
        if AttachedTo not in c.states:
            return False
        if c.states[AttachedTo].get_value(p):
            return True
        rec = PRED.attach_if_aligned(c, p)
        self.last_attach = {k: v for k, v in rec.items()}
        return bool(rec.get("attached"))

    def q28(self) -> th.Tensor:
        return self.robot.get_joint_positions().clone().float()


class MotionSnapshot:
    """Robot configuration + task-object poses, to tell motion over a window."""

    def __init__(self, robot, objects: Dict[str, Any]):
        self.q = robot.get_joint_positions().clone().float()
        self.base = tuple(x.clone().float() for x in robot.get_position_orientation())
        self.objects = {k: tuple(x.clone().float() for x in o.get_position_orientation()) for k, o in objects.items()}

    def delta(self, other: "MotionSnapshot") -> Dict[str, float]:
        d = {
            "joint_rad": float(th.max(th.abs(other.q - self.q))) if len(self.q) else 0.0,
            "base_m": float(th.linalg.norm(other.base[0] - self.base[0])),
            "base_rad": C.quat_angle(other.base[1], self.base[1]),
        }
        for k, (p, q) in self.objects.items():
            if k in other.objects:
                p1, q1 = other.objects[k]
                d[f"{k}_m"] = float(th.linalg.norm(p1 - p))
                d[f"{k}_rad"] = C.quat_angle(q1, q)
        return d


# ==========================================================================================
# executor
# ==========================================================================================


class Executor:
    """The §3.8 loop for one episode. ``sim`` is ``og.sim`` (imported lazily when ``None``)."""

    def __init__(self, env, task, sidecar, recorder: Optional[Recorder] = None, motions: Optional[Motions] = None,
                 monitors: Optional[Monitors] = None, cfg: Optional[ExecutorConfig] = None, sim=None, state: Optional[SimState] = None,
                 tracked_objects: Optional[Dict[str, Any]] = None, cam_key: Optional[str] = None):
        self.env, self.task, self.sidecar = env, task, sidecar
        self.cfg = cfg or ExecutorConfig()
        self.robot = env.scene.robots[0]
        self.cam_key = cam_key or (state.cam_key if state is not None else None)
        self.recorder = recorder
        self.motions = motions
        self.monitors = monitors
        self._sim = sim
        self.state = state or SimState(env, task, self.robot, self.cam_key)
        self.tracked = dict(tracked_objects or {})
        self.n = 0
        self.last_obs = None
        self.last_info: Dict[str, Any] = {}
        self.phase_steps: Dict[int, int] = {PHASE_ORACLE: 0, PHASE_SETTLE: 0, PHASE_GOAL: 0}
        self._wall_t0: Optional[float] = None  # program mode: when the program started (cfg.max_wall_s)
        self.wall_clock_exceeded = False
        #: every assisted-grasp release of the episode: it must come from OmniGibson's rule at a recorded gripper command of
        #: exactly +1 (what an evaluated policy can do); a release between steps (a direct call) is a violation - rollbacks
        #: reset the tracking
        self.release_check: Dict[str, Any] = {"releases": [], "violations": []}
        self._ag_last: Optional[Dict[str, Optional[str]]] = None
        self.stats: QAStats = monitors.stats if monitors is not None else QAStats()
        self.action_records: List[ActionRecord] = []
        self.last_ctx: Dict[str, Any] = {}
        self._defer_commit = False
        self.brace_retries: List[Dict[str, Any]] = []
        #: the oracle's own counters of the episode (``result.counters``): ``push_outcome.top_press_in_tolerance_accepted``
        self.counters: Dict[str, int] = {}
        self._stream = False  # a replay / policy stream: the retry rules are off (only the step budget)
        # oracle.protection (owner 2026-09-26): the goal objects protected during transits ([(bddl name, scene object)], set by
        # run_isg), and the actions whose retry takes the via-above re-plan at once (after a run-time protected verdict)
        self.protected_goal: List[Tuple[str, Any]] = []
        self._protect_force: set = set()
        #: the interior support surface of the instance's template (``execute.interior_support_surface``, set by run_isg; None on a
        #: support top): every action gets it as ``ctx["support_surface"]`` (``execute.interior_split``)
        self.support_surface: Optional[Dict[str, Any]] = None
        # first-demo v2.1 A2: the witnesses of a kept arm's next contact, tracked from the branch a travel leg kept it in
        # ({action index: {"arm", "witnesses", "leg"}}, execute.keep_arm_branch)
        self._witness_overrides: Dict[int, Dict[str, Any]] = {}
        self.program_actions: List[Action] = []
        self.checkpoint_reports: List[Dict[str, Any]] = []
        self.timings: Dict[str, float] = {}
        self.result: Dict[str, Any] = {}
        self.verdict: Optional[Verdict] = None
        self.support_load: Optional[SupportLoad] = None  # the support loading fact (meta.support_disturbance), built by run_program
        self.arm_stutter: Optional[ArmStutter] = None  # the commanded arm jolts (meta.arm_stutter, gate arm_stutter), built per episode
        #: ``--sleep-assist`` (``oracle.sleep_assist``, as run_isg_vec's slots): the tracked task objects at rest put to sleep, built per
        #: episode when the driver's flag is on (``ISG_SLEEP_ASSIST``)
        self.sleep_assist: Optional[SA.SleepAssist] = None
        #: ``--qa`` (``cfg.qa``): the episode's QA metrics, built by run_program / run_stream
        self.qa: Optional[QA] = None
        #: a stream's program (the sidecar's expanded actions, set by run_isg): its operated articulated roots are what
        #: ``object_tilt`` watches in a replay / policy episode
        self.qa_program: Optional[Sequence[Action]] = None
        #: the recorded phase of each streamed action (a replayed demo's ``isg/phase``): its settle rows are no action steps
        #: for the QA metrics and the stalled rule
        self.stream_phases: Optional[Sequence[int]] = None

    # ------------------------------------------------------------------ sim access

    @property
    def sim(self):
        if self._sim is None:
            import omnigibson as og

            self._sim = og.sim
        return self._sim

    def enable_recording_filter(self, enabled: bool = True) -> None:
        """BEHAVIOR's per-step dump filter (active, initialized objects only) for the recorded states; checkpoints
        always dump the full state."""
        self._recording_filter = bool(enabled)
        registry = getattr(getattr(self.env, "scene", None), "object_registry", None)
        if enabled and fast_on("RECORD_CHANGED_ONLY"):
            self._active_filter = self._changed_filter()  # the instance attribute shadows the static method
        if registry is not None and hasattr(registry, "set_dump_filter"):
            registry.set_dump_filter(self._active_filter if enabled else (lambda obj: True))

    @staticmethod
    def _active_filter(obj) -> bool:
        return bool(getattr(obj, "is_active", True) and getattr(obj, "initialized", True))

    #: RECORD_CHANGED_ONLY: an awake object enters a recorded state only when its root pose or joints moved more than this since the
    #: state it was last recorded with (m / quaternion components / joint units)
    RECORD_CHANGE_TOL = 1e-5

    def _changed_filter(self):
        """train-fast 2026-11-01 (RECORD_CHANGED_ONLY): BEHAVIOR's per-step filter (awake, initialized objects) plus: only objects whose root
        pose or joint positions changed by more than :data:`RECORD_CHANGE_TOL` since their last recorded state. In the full house scene
        ~50 static objects stay awake (resting-contact chatter: bushes, sofas, pot plants) and were written every step; a replay then
        re-posed them every step - half the replay speed, and the renderer's temporal history disturbed on static edges (nextto_02
        100322: 3.2 grey levels from the cropped path vs 1.9 run-to-run). An object left out keeps its last loaded state on replay, as a
        sleeping object always did; the comparison is against the last RECORDED state, so a slow drift is still recorded once it adds up."""
        last: Dict[str, th.Tensor] = {}
        tol = float(fast_value("RECORD_CHANGE_TOL", self.RECORD_CHANGE_TOL))
        base = Executor._active_filter

        def flt(obj) -> bool:
            if not base(obj):
                return False
            try:
                p, q = obj.get_position_orientation()
                parts = [th.as_tensor(p, dtype=th.float32).reshape(-1), th.as_tensor(q, dtype=th.float32).reshape(-1)]
                if int(getattr(obj, "n_joints", 0) or 0) > 0:
                    parts.append(th.as_tensor(obj.get_joint_positions(), dtype=th.float32).reshape(-1))
                fp = th.cat(parts)
            except Exception:  # noqa: BLE001 - anything without a pose: recorded as before
                return True
            prev = last.get(obj.name)
            if prev is not None and prev.shape == fp.shape and float((fp - prev).abs().max()) <= tol:
                return False
            last[obj.name] = fp.clone()
            return True

        return flt

    @property
    def state_owner(self):
        """What checkpoints and recorded states are dumped from and loaded into: the env's scene
        (``ExecutorConfig.per_scene_state``) or the simulator."""
        return self.env.scene if self.cfg.per_scene_state else self.sim

    def _dump_state(self, full: bool) -> th.Tensor:
        owner = self.state_owner
        registry = getattr(getattr(self.env, "scene", None), "object_registry", None)
        if full and getattr(self, "_recording_filter", False) and registry is not None and hasattr(registry, "set_dump_filter"):
            registry.set_dump_filter(lambda obj: True)
            try:
                return owner.dump_state(serialized=True)
            finally:
                registry.set_dump_filter(self._active_filter)
        return owner.dump_state(serialized=True)

    @property
    def isg(self) -> Dict[str, Any]:
        info = self.last_info or {}
        return (info.get("done") or {}).get("ispatialgym") or info.get("ispatialgym") or {}

    @property
    def action_budget(self) -> int:
        """The steps a program's actions may take: ``max_steps`` less its epilogue's ``settle_steps``."""
        return max(0, int(self.cfg.max_steps) - int(self.cfg.settle_steps))

    # ------------------------------------------------------------------ stepping (the only env.step)

    def _ag_held(self) -> Dict[str, Optional[str]]:
        """arm -> name of the object the assisted-grasp joint holds (None: no joint)."""
        cons = getattr(self.robot, "_ag_obj_constraints", None) or {}
        held = getattr(self.robot, "_ag_obj_in_hand", None) or {}
        return {arm: (getattr(held.get(arm), "name", "?") if c is not None else None) for arm, c in cons.items()}

    def _check_releases(self, before: Dict[str, Optional[str]], after: Dict[str, Optional[str]], action: th.Tensor) -> None:
        if self._ag_last is not None:
            for arm, obj in self._ag_last.items():
                if obj is not None and before.get(arm) is None:
                    self.release_check["violations"].append({"step": self.n + 1, "arm": arm, "object": obj, "kind": "released_between_steps"})
        for arm, obj in before.items():
            if obj is not None and after.get(arm) is None:
                cmd = float(action[self.robot.controller_action_idx[f"gripper_{arm}"]].flatten()[0])
                rec = {"step": self.n, "arm": arm, "object": obj, "gripper_command": round(cmd, 4)}
                # the fork's pad-clear release (Robot.ag_release_on_pad_clear): how long the weld outlived the open command
                # and what ended it ("pads_clear" or "cap")
                log_ = getattr(self.robot, "ag_release_log", None)
                ev = next((e for e in reversed(log_ or []) if e.get("arm") == arm and e.get("object") == obj), None)
                if ev is not None:
                    rec.update({"weld_delay_s": round(float(ev["delay_s"]), 4), "weld_delay_physics_steps": int(ev["delay_steps"]), "weld_released_by": ev["by"]})
                    log_.remove(ev)
                self.release_check["releases"].append(rec)
                if cmd < 1.0 - 1e-6:
                    self.release_check["violations"].append({"step": self.n, "arm": arm, "object": obj, "kind": "gripper_command_below_open",
                                                             "gripper_command": round(cmd, 4)})

    def step(self, action: th.Tensor, phase: int = PHASE_ORACLE, action_index: int = -1, attempt: int = 0, monitor: bool = True,
             acting: Optional[bool] = None) -> Optional[Verdict]:
        """One control step. ``acting``: whether it is an action step for the QA metrics and the stalled rule (default: not a
        settle-window step; False for a settle row of a replayed demo)."""
        settling = False
        if not self._stream and self.motions is not None:
            settling = bool(getattr(self.motions, "settling", False))
            self.motions.settling = False
        if acting is None:
            acting = phase != PHASE_SETTLE
        # 2026-11-01 train-fast (always on): per-step cost split - env step (physics + obs + task), monitors, state dump, recorder, QA
        sc = self.__dict__.setdefault("step_costs", {"steps": 0, "env_step_s": 0.0, "monitors_s": 0.0, "dump_s": 0.0, "record_s": 0.0, "qa_s": 0.0,
                                                     "dump_floats": 0})
        _t0 = time.perf_counter()
        ag_before = self._ag_held()
        obs, reward, terminated, truncated, info = SE.step(self.env, action, n_render_iterations=self.cfg.n_render_iterations)
        _t1 = time.perf_counter()
        sc["steps"] += 1
        sc["env_step_s"] += _t1 - _t0
        self.n += 1
        ag_after = self._ag_held()
        n_violations = len(self.release_check["violations"])
        self._check_releases(ag_before, ag_after, action)
        new_violations = self.release_check["violations"][n_violations:]
        self._ag_last = ag_after
        self.phase_steps[phase] = self.phase_steps.get(phase, 0) + 1
        self.last_obs, self.last_info = obs, info
        if self.support_load is not None:
            try:
                name = self.program_actions[action_index].name if 0 <= action_index < len(self.program_actions) else None
                self.support_load.observe(name)
            except Exception as e:  # noqa: BLE001  (a fact, never a failure)
                log.warning("support load monitor failed (%r); disabled", e)
                self.support_load = None
        if self.arm_stutter is not None:
            try:
                in_action = action_index >= 0 and phase == PHASE_ORACLE
                self.arm_stutter.observe(self.n, action, action_index * 1000 + attempt if in_action else -1,
                                         self.program_actions[action_index].name if 0 <= action_index < len(self.program_actions) else None)
            except Exception as e:  # noqa: BLE001  (a fact; the QA metric then has nothing to judge)
                log.warning("arm stutter monitor failed (%r); disabled", e)
                self.arm_stutter = None
        _t2 = time.perf_counter()
        sc["monitors_s"] += _t2 - _t1
        if self.recorder is not None:
            state = self._dump_state(full=False) if (self.cfg.record_state and self.recorder.cfg.hdf5) else None
            _t3 = time.perf_counter()
            sc["dump_s"] += _t3 - _t2
            if state is not None:
                sc["dump_floats"] += int(state.numel())
            sim_time = None
            try:
                sim_time = float(self.sim.current_time)
            except Exception:  # noqa: BLE001
                pass
            try:
                rew = float(reward)
            except Exception:  # noqa: BLE001
                rew = 0.0
            self.recorder.record(self.n, action, obs, info, state=state, phase=phase, action_index=action_index, attempt=attempt,
                                 reward=rew, terminated=bool(terminated), truncated=bool(truncated), sim_time=sim_time)
            sc["record_s"] += time.perf_counter() - _t3
        if self.sleep_assist is not None:
            _t5 = time.perf_counter()
            self.sleep_assist.step(self.n)  # after the step's record, before QA observes it (run_isg_vec: SlotRecorder.record, then qa.observe)
            sc["sleep_assist_s"] = sc.get("sleep_assist_s", 0.0) + (time.perf_counter() - _t5)
        verdict = None
        _t4 = time.perf_counter()
        if self.qa is not None:
            verdict = self.qa.observe(self.n, action, acting=acting, live=monitor, held=[n for n in ag_after.values() if n],
                                      release_violations=new_violations, settling=settling)
            if verdict is not None:
                verdict.action_index, verdict.attempt = action_index, attempt
                self.stats.record(verdict)
        if verdict is None and monitor and self.monitors is not None:
            if self._stream:
                verdict = self.monitors.check_timeout_only(self.n, settle=not acting)
                pop = getattr(self.monitors, "pop_logged", None)
                for v in (pop() if pop is not None else ()):
                    if self.qa is not None:
                        self.qa.note_rule(v)
            else:
                verdict = self.monitors.check(self.n, settle=settling)
        sc["qa_s"] += time.perf_counter() - _t4
        if verdict is None and not self._stream and phase != PHASE_SETTLE and self.n > self.action_budget:
            verdict = Verdict("timeout", f"the program exceeded its {self.action_budget} action steps (max_steps {self.cfg.max_steps} less the "
                                         f"{self.cfg.settle_steps}-step epilogue)", step=self.n)
            self.stats.record(verdict)
        if (verdict is None and phase != PHASE_SETTLE and self._wall_t0 is not None and self.cfg.max_wall_s
                and time.time() - self._wall_t0 > self.cfg.max_wall_s):
            self.wall_clock_exceeded = True
            verdict = Verdict("timeout", f"the program exceeded the wall clock of {self.cfg.max_wall_s:.0f} s", step=self.n)
            self.stats.record(verdict)
        return verdict

    def run_generator(self, gen: Iterator[th.Tensor], phase: int = PHASE_ORACLE, action_index: int = -1, attempt: int = 0, monitor: bool = True) -> Optional[Verdict]:
        """Step every action a generator yields; returns the first monitor verdict (the generator is then closed)."""
        for a in gen:
            v = self.step(a, phase, action_index, attempt, monitor=monitor)
            if v is not None:
                gen.close()
                return v
        return None

    # ------------------------------------------------------------------ checkpoints

    def _backend(self):
        return getattr(self.motions, "backend", None)

    def _arm_offer_note(self, action) -> Optional[Dict[str, Any]]:
        """The sampler's arm offer of the segment an action belongs to (first-demo v2.2; fired / not fired): ``grasps[].arm_offer`` of
        the grasp record of the action's object made by the action's arm; None for any other action."""
        obj, arms = getattr(action, "object", None), tuple(getattr(action, "arms", ()) or ())
        g = next((g for g in (getattr(self.sidecar, "grasps", None) or []) if g.object == obj and g.arm in arms), None) if obj and arms else None
        ex = getattr(g, "extra", None) if g is not None else None
        offer = ex.get("arm_offer") if isinstance(ex, dict) else None
        return dict(offer) if isinstance(offer, dict) else None

    def checkpoint(self, action_index: int) -> Checkpoint:
        robot = self.robot
        backend = self._backend()
        attached = {}
        if backend is not None:
            for arm, info in getattr(backend, "_attached", {}).items():
                attached[arm] = info["obj"].name
        ag = getattr(robot, "_ag_obj_in_hand", {}) or {}
        params = getattr(robot, "_ag_obj_constraint_params", {}) or {}
        cp = Checkpoint(
            action_index=action_index,
            step=self.n,
            sim_state=self._dump_state(full=True).clone(),
            exec_state=self.motions.export_state() if self.motions is not None else {},
            attached=attached,
            ag_in_hand={arm: (o.name if o is not None else None) for arm, o in ag.items()},
            ag_params={arm: (dict(p) if p is not None else None) for arm, p in params.items()},
            ag_counters={
                "release": dict(getattr(robot, "_ag_release_counter", {}) or {}),
                "grasp": dict(getattr(robot, "_ag_grasp_counter", {}) or {}),
            },
            obs=self.last_obs,
            info=self.last_info,
            recorder_steps=self.recorder.n_total if self.recorder is not None else self.n,
            env_step=getattr(self.env, "_current_step", None),
            task_tracking={k: getattr(self.task, k) for k in TASK_TRACKING_ATTRS if hasattr(self.task, k)},
            fov_spans=self.monitors.fov_spans() if self.monitors is not None and hasattr(self.monitors, "fov_spans") else {},
            qa=self.qa.snapshot() if self.qa is not None else None,
        )
        return cp

    def restore(self, cp: Checkpoint, action_index: int = -1, attempt: int = 0) -> Dict[str, Any]:
        """Roll the simulator and the executor back to ``cp``; returns the round-trip report."""
        robot = self.robot
        report: Dict[str, Any] = {"action_index": cp.action_index, "ag_roundtrip_ok": True, "ag_fixed": [], "attached_fixed": []}
        # what the camera showed last: the rolled-back attempt's final scene (nothing to compare without renders)
        stale = None if self.cfg.physics_only else self.head_rgb_of(self.last_obs)
        self.state_owner.load_state(cp.sim_state, serialized=True)
        if self.cfg.physics_only:
            from ispatialgym.oracle import physics_only as PO

            PO.sync_transforms(self.sim)
        self._ag_last = None  # the restore re-establishes / drops grasps itself (not a release of the demo)
        for k in ("releases", "violations"):  # the rolled-back attempt's steps leave the recording, and its releases with them
            self.release_check[k] = [r for r in self.release_check[k] if r["step"] <= cp.step]
        if self.arm_stutter is not None:
            self.arm_stutter.trim(cp.step)
        if self.sleep_assist is not None:
            self.sleep_assist.trim(cp.step)
        # -- assisted grasp: verify OmniGibson's own round-trip, re-apply what it does not carry
        ag = getattr(robot, "_ag_obj_in_hand", None)
        if ag is not None:
            for arm, want in cp.ag_in_hand.items():
                have = ag.get(arm)
                have_name = have.name if have is not None else None
                if have_name != want:
                    report["ag_roundtrip_ok"] = False
                    report["ag_fixed"].append({"arm": arm, "restored": have_name, "checkpoint": want})
                    if have is not None:
                        robot.release_grasp_immediately(arm)
                    params = cp.ag_params.get(arm)
                    if want is not None and params is not None:
                        p = dict(params)
                        if isinstance(p.get("target_obj"), str):
                            p["target_obj"] = self.env.scene.object_registry("name", p["target_obj"])
                        robot._create_assisted_grasp_joint(arm, p)
            for arm in cp.ag_counters["release"]:
                if hasattr(robot, "_ag_release_counter"):
                    robot._ag_release_counter[arm] = cp.ag_counters["release"][arm]
                if hasattr(robot, "_ag_grasp_counter"):
                    robot._ag_grasp_counter[arm] = cp.ag_counters["grasp"].get(arm)
        # -- planner attachments
        backend = self._backend()
        if backend is not None:
            have = {arm: info["obj"].name for arm, info in getattr(backend, "_attached", {}).items()}
            for arm in set(have) - set(cp.attached):
                backend.detach_object(arm)
                report["attached_fixed"].append({"arm": arm, "detached": have[arm]})
            for arm, name in cp.attached.items():
                if have.get(arm) != name:
                    if arm in have:
                        backend.detach_object(arm)
                    obj = self.env.scene.object_registry("name", name)
                    backend.attach_object(arm, obj)
                    report["attached_fixed"].append({"arm": arm, "attached": name})
        # -- executor state
        if self.motions is not None and cp.exec_state:
            self.motions.import_state(cp.exec_state)
        if self.monitors is not None:
            self.monitors.reset_ag_reference()
            if hasattr(self.monitors, "restore_fov_spans"):
                self.monitors.restore_fov_spans(cp.fov_spans)
        if self.qa is not None and cp.qa is not None:
            self.qa.restore(cp.qa, cp.step)
        for obj in self.tracked.values():
            if hasattr(obj, "wake"):
                obj.wake()
        # -- episode bookkeeping: step counters and the task's sticky success tracking
        self.n = cp.step
        if cp.env_step is not None and hasattr(self.env, "_current_step"):
            self.env._current_step = cp.env_step
        for k, v in cp.task_tracking.items():
            try:
                setattr(self.task, k, v)
            except Exception:  # noqa: BLE001
                pass
        self.last_obs, self.last_info = cp.obs, cp.info
        if self.recorder is not None:
            self.recorder.set_observation(cp.obs)
        # -- the renderer trails the physics: the first frames after ``load_state`` still show the pre-restore scene
        #    (a sensor read lags the last render by a load-dependent 1-2 frames plus the USD sync, B10), so the failed
        #    attempt's last arm pose leaked into the committed video as the restore step's frame. Render until the
        #    head camera shows the restored scene (= the checkpoint's observation) before anything is recorded again.
        report["restore_renders"] = self.render_restored(cp, stale)
        self.checkpoint_reports.append(report)
        # -- the PhysX drive targets are not state: apply the restored hold targets with one (recorded) hold step;
        #    the monitors are off for it (their per-action whitelist is not set up yet)
        if self.motions is not None:
            self.run_generator(self.motions.hold(1), PHASE_ORACLE, action_index, attempt, monitor=False)
        return report

    def head_rgb_of(self, obs) -> Optional[np.ndarray]:
        """The head camera's rgb of an observation as uint8 (H, W, 3); None when the observation has none."""
        try:
            robs = obs.get(self.robot.name, obs) if isinstance(obs, dict) else None
            return to_uint8_rgb(robs[self.cam_key]["rgb"]) if robs else None
        except Exception:  # noqa: BLE001
            return None

    def head_rgb_now(self) -> Optional[np.ndarray]:
        """The head camera's rgb read from the sensor now (the annotator buffer as it currently stands; a read does
        not render) as uint8 (H, W, 3); None when there is no readable camera (fakes without one)."""
        try:
            sensor = (getattr(self.robot, "sensors", None) or {}).get(self.cam_key) if self.cam_key else None
            if sensor is None or not hasattr(sensor, "get_obs"):
                return None
            out = sensor.get_obs()
            robs = out[0] if isinstance(out, tuple) else out
            return to_uint8_rgb(robs["rgb"])
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def frame_difference(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        """Mean absolute rgb difference of two uint8 frames (0-255 scale); None when the shapes differ."""
        if a is None or b is None or a.shape != b.shape:
            return None
        return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())

    def render_restored(self, cp: Checkpoint, stale: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Render after ``load_state`` until the head camera shows the restored scene (``ExecutorConfig.restore_*``):
        at least ``restore_renders`` frames, then one frame at a time until the sensor image is *fresh* (closer to the
        checkpoint observation ``cp.obs`` than to ``stale``, the last observation of the rolled-back attempt; always
        fresh without a ``stale`` reference) and *settled* (within ``restore_frame_tol`` of the checkpoint observation,
        or a new frame whose distance to it shrank by at most ``restore_frame_gain_tol`` since the previous new frame:
        the ghost of the rolled-back scene has faded), up to ``restore_renders_max`` frames. Returns ``{"n", "diff",
        "diff_stale", "change", "gain", "fresh", "converged", "checked"}`` - of the last frame read: its distance to the
        checkpoint observation / to the rolled-back one / to the previous read, the shrinkage of ``diff`` since the
        previous new frame, freshness; ``checked`` False = no checkpoint observation / no readable camera: the fixed
        minimum was rendered. A frame within ``restore_frame_tol`` of the checkpoint observation is fresh whatever the
        rolled-back scene looked like (an attempt that barely moved leaves the two references alike)."""
        if self.cfg.physics_only:
            # no frames are recorded: nothing to wait for (the transforms were synced after the load)
            return {"n": 0, "diff": None, "diff_stale": None, "change": None, "gain": None, "fresh": None, "converged": None,
                    "checked": False, "physics_only": True}
        n_min, n_max = int(self.cfg.restore_renders), max(int(self.cfg.restore_renders), int(self.cfg.restore_renders_max))
        tol, gain_tol = float(self.cfg.restore_frame_tol), float(self.cfg.restore_frame_gain_tol)
        ref = self.head_rgb_of(cp.obs)
        out: Dict[str, Any] = {"n": 0, "diff": None, "diff_stale": None, "change": None, "gain": None, "fresh": None, "converged": None, "checked": ref is not None}
        prev: Optional[np.ndarray] = None
        prev_diff: Optional[float] = None
        for k in range(1, n_max + 1):
            self.sim.render()
            out["n"] = k
            if k < n_min:
                continue
            if ref is None:
                break
            cur = self.head_rgb_now()
            diff = self.frame_difference(cur, ref)
            if diff is None:  # the sensor cannot be read: nothing to wait for beyond the minimum
                out["checked"] = False
                break
            out["diff"], out["diff_stale"], out["change"] = diff, self.frame_difference(cur, stale), self.frame_difference(cur, prev)
            out["fresh"] = diff <= tol or out["diff_stale"] is None or diff <= out["diff_stale"]
            # an exactly repeated read is the queue handing out the same frame twice (the GPU behind), not a new frame
            # (0924-170128: 6.04 -> 5.68 -> 5.68 duplicate; 0924-171522: 2.52 -> 2.52); it neither settles nor counts
            new_frame = out["change"] is None or out["change"] > 0.0
            out["gain"] = None if (prev_diff is None or not new_frame) else prev_diff - diff
            settled = diff <= tol or (out["gain"] is not None and out["gain"] <= gain_tol)
            if out["fresh"] and settled:
                out["converged"] = True
                break
            if new_frame:
                prev, prev_diff = cur, diff
        if out["checked"] and not out["converged"]:
            out["converged"] = False
            if out["fresh"]:
                log.info("restore of action %d: the head camera shows the restored scene after %d renders but its distance to the checkpoint observation is still shrinking (%.2f, by %s per frame)",
                         cp.action_index, out["n"], out["diff"], None if out["gain"] is None else round(out["gain"], 2))
            else:
                log.warning("restore of action %d: after %d renders the head camera still shows the rolled-back scene (diff %.2f to the checkpoint observation, %.2f to the rolled-back attempt's last frame); the recorded hold step may show the rolled-back attempt",
                            cp.action_index, out["n"], out["diff"], out["diff_stale"])
        return out

    # ------------------------------------------------------------------ one action

    def witnesses_for(self, index: int) -> Dict[str, th.Tensor]:
        """The sampler's IK witnesses of action ``index`` (sidecar ``ik_witnesses[].targets[key].q``, the 28-D joint
        solution of every checked end pose, written by the program check) as ``{target key: q}``; ``{}`` when the
        sidecar has none (pre-2026-09-16 sidecars)."""
        entries = (getattr(self.sidecar, "extra", None) or {}).get("ik_witnesses") or []
        out: Dict[str, th.Tensor] = {}
        for e in entries:
            if not isinstance(e, dict) or e.get("action") != index:
                continue
            for key, w in (e.get("targets") or {}).items():
                q = w.get("q") if isinstance(w, dict) else None
                if q is not None and len(q) == 28:
                    out[key] = th.tensor([float(v) for v in q], dtype=th.float32)
        return out

    def witness_poses_for(self, index: int) -> Dict[str, Tuple[th.Tensor, th.Tensor]]:
        """The eef poses the witnesses of action ``index`` were recorded for (``ik_witnesses[].targets[key].pose``) as ``{target key: (pos,
        quat)}``; a key without a recorded pose is left out."""
        entries = (getattr(self.sidecar, "extra", None) or {}).get("ik_witnesses") or []
        out: Dict[str, Tuple[th.Tensor, th.Tensor]] = {}
        for e in entries:
            if not isinstance(e, dict) or e.get("action") != index:
                continue
            for key, w in (e.get("targets") or {}).items():
                p = w.get("pose") if isinstance(w, dict) else None
                if isinstance(p, dict) and p.get("position") is not None and p.get("orientation") is not None:
                    out[key] = (th.tensor([float(v) for v in p["position"]], dtype=th.float32), th.tensor([float(v) for v in p["orientation"]], dtype=th.float32))
        return out

    def next_contact(self, index: int, arm: str) -> Optional[Dict[str, Any]]:
        """The next contact of ``arm`` after the travel legs that follow action ``index`` (``execute.keep_arm_branch``, first-demo v2.1
        A2): the first action past the transitions / navigates, when it moves ``arm``, with its chain - the actions of ``arm`` up to the
        next travel leg - as the sampler's witnesses in order (``[{action, key, pose, q, margin, base}]``), the contact's stance (the base
        pose and the trunk joints of its first witness) and the chain's smallest recorded joint-limit margin. None when another action
        comes first or no witness with a stance is recorded."""
        acts = list(self.program_actions)
        entries = (getattr(self.sidecar, "extra", None) or {}).get("ik_witnesses") or []
        j = index + 1
        while j < len(acts) and isinstance(acts[j], (Transition, Navigate)):
            j += 1
        if j >= len(acts) or arm not in tuple(getattr(acts[j], "arms", ()) or ()):
            return None
        if not isinstance(acts[j], Grasp):
            # a push / brace acts through the arm's posture around its obstacles (sandal 0927-030120 replayed with the arm kept: the push
            # under the table from the other branch turned the sandal 49 deg): its recorded arm posture stays
            return {"index": j, "name": acts[j].name, "object": None, "chain": [], "why": f"the arm's next contact is a {acts[j].name}, not a grasp"}
        if getattr(acts[j], "repose", None) is not None:
            # block design after a base move: the grasp re-poses the arm to its recorded pre-grasp configuration (``Grasp.repose``) - its
            # witnesses stay the program check's, no branch kept from the travel replaces them
            return {"index": j, "name": acts[j].name, "object": None, "chain": [], "why": "the arm's next grasp re-poses it to its recorded pre-grasp configuration"}
        chain: List[Dict[str, Any]] = []
        k = j
        while k < len(acts) and not isinstance(acts[k], (Transition, Navigate)):
            if arm in tuple(getattr(acts[k], "arms", ()) or ()):
                for e in entries:
                    if not isinstance(e, dict) or e.get("action") != k or (e.get("arm") or arm) != arm:
                        continue
                    for key, w in (e.get("targets") or {}).items():
                        if not isinstance(w, dict) or w.get("q") is None or len(w["q"]) != 28 or not isinstance(w.get("pose"), dict):
                            continue
                        p = w["pose"]
                        chain.append({"action": k, "key": key, "q": th.tensor([float(v) for v in w["q"]], dtype=th.float32), "margin": w.get("joint_limit_margin_rad"),
                                      "pose": (th.tensor([float(v) for v in p["position"]]), th.tensor([float(v) for v in p["orientation"]])), "base": w.get("base_pose2d")})
            k += 1
        if not chain or chain[0]["base"] is None:
            return None
        margins = [c["margin"] for c in chain if c["margin"] is not None]
        return {"index": j, "name": acts[j].name, "object": (tuple(getattr(acts[j], "objects", ()) or ()) or (None,))[0], "chain": chain,
                "stance": ([float(v) for v in chain[0]["base"]], chain[0]["q"][self.robot.trunk_control_idx]), "margin": min(margins) if margins else None}

    def stance_contact(self, index: int, arm: str) -> Optional[Dict[str, Any]]:
        """first-demo v3.0 F1: the first action of ``arm`` at the stance the travel legs after action ``index`` arrive at (up to the next
        travel leg), when it is a brace or a push: ``{"index", "name", "object", "targets", "witnesses"}`` - its recorded targets and the
        sampler's IK witnesses of it in order (``witnesses_for``). None when the arm's first action there is another one, or there is none."""
        acts = list(self.program_actions)
        j = index + 1
        while j < len(acts) and isinstance(acts[j], (Transition, Navigate)):
            j += 1
        while j < len(acts) and not isinstance(acts[j], (Transition, Navigate)):
            a = acts[j]
            if arm in tuple(getattr(a, "arms", ()) or ()):
                if not isinstance(a, (Brace, Push)):
                    return None
                try:
                    targets = a.targets(self.sidecar, self.state)
                except Exception:  # noqa: BLE001
                    return None
                return {"index": j, "name": a.name, "object": a.contact.object, "targets": targets, "witnesses": self.witnesses_for(j)}
            j += 1
        return None

    def next_place_kind(self, index: int, action: Grasp) -> Optional[str]:
        """The kind of the place that puts down what the grasp ``index`` picks (the first ``Place`` of its arm and object before that
        arm grasps again); None when none follows."""
        acts = list(self.program_actions)
        for a in acts[index + 1:]:
            if isinstance(a, Grasp) and a.arm == action.arm:
                return None
            if isinstance(a, Place) and a.arm == action.arm and a.object == action.object:
                return a.kind
        return None
    def _absorb_branch_decisions(self, index: int, ctx: Dict[str, Any]) -> None:
        """After a travel leg passed: a kept arm's next contact runs on the configurations tracked from it (``_witness_overrides``); an
        arm the leg re-posed to the recorded joints drops them (its next contact runs on the sampler's witnesses)."""
        reposed = set((ctx.get("arms_posture") or {}).get("moved") or []) | set(ctx.get("navigate_arms_reposed") or [])
        reposed |= {arm for arm, f in (ctx.get("branch_kept") or {}).items() if not f.get("keep")}
        if reposed:
            self._witness_overrides = {i: o for i, o in self._witness_overrides.items() if o["arm"] not in reposed}
        for arm, per_action in (ctx.get("witness_overrides") or {}).items():
            for i, ws in per_action.items():
                self._witness_overrides[int(i)] = {"arm": arm, "witnesses": ws, "leg": index}

    def _scene_names(self, instances: Iterable[str]) -> set:
        out = set()
        for inst in instances:
            try:
                out.add(self.state.object(inst).name)
            except Exception:  # noqa: BLE001
                out.add(inst)
        return out

    def brace_allowed_pairs(self, index: int) -> List[Tuple[List[str], str]]:
        """Contacts the collision monitor must not flag while a brace is active (owner 2026-09-24 item 1): the brace hand's
        finger / gripper links against the braced object, from the ``Brace`` action through the brace arm's releasing
        ``Retreat`` (``programs._brace_push_retreat``: brace -> push -> push-arm retreat -> brace-arm retreat). Read off the
        program, so a checkpoint restore / retry needs no state: ``[(link names, scene object name), ...]`` for the braces
        whose span contains action ``index``. The braced object was never in the push's ``intended`` set and the monitor
        exempts the acting arm's fingers only; with the closed-loop push arc and the 250 kg chassis the brace pad presses on
        its object on every attempt (touching 0925-113253: ``left_gripper_finger_link1 in contact with partner_box_of_tissues``)."""
        out: List[Tuple[List[str], str]] = []
        actions = list(getattr(self, "program_actions", None) or [])
        active: Dict[str, Tuple[int, str]] = {}
        for j, a in enumerate(actions):
            name = getattr(a, "name", "")
            if name == "brace":
                contact = getattr(a, "contact", None)
                arm, obj = getattr(contact, "arm", None), getattr(contact, "object", None)
                if arm and obj:
                    active[arm] = (j, obj)
            elif name == "retreat" and getattr(a, "release", False) and getattr(a, "arm", None) in active:
                start, obj = active.pop(a.arm)
                if start <= index <= j:
                    out.append((list(M.FINGER_LINKS[a.arm]) + [M.GRIPPER_LINK[a.arm]], obj))
        for arm, (start, obj) in active.items():  # a brace with no releasing retreat recorded: active to the program's end
            if start <= index:
                out.append((list(M.FINGER_LINKS[arm]) + [M.GRIPPER_LINK[arm]], obj))
        resolved: List[Tuple[List[str], str]] = []
        for links, obj in out:
            names = self._scene_names([obj])
            for n in names:
                resolved.append((links, n))
        return resolved

    def braces_held(self, index: int) -> List[Dict[str, Any]]:
        """The braces standing while action ``index`` runs: from each ``Brace`` action to its arm's next ``Retreat`` (releasing
        or not: a wide stop lifts off with its fingers open), exclusive at both ends; a brace with no retreat stands to the
        program's end. Read off the program like :meth:`brace_allowed_pairs`, so a checkpoint restore / retry needs no state:
        ``[{"arm", "object", "brace_index"}, ...]`` (``execute_push`` holds them in the world during a base nudge)."""
        out: List[Dict[str, Any]] = []
        active: Dict[str, Tuple[int, str]] = {}
        for j, a in enumerate(list(getattr(self, "program_actions", None) or [])):
            name = getattr(a, "name", "")
            if name == "brace":
                contact = getattr(a, "contact", None)
                arm, obj = getattr(contact, "arm", None), getattr(contact, "object", None)
                if arm and obj:
                    active[arm] = (j, obj)
            elif name == "retreat" and getattr(a, "arm", None) in active:
                start, obj = active.pop(a.arm)
                if start < index < j:
                    out.append({"arm": a.arm, "object": obj, "brace_index": start})
        out += [{"arm": arm, "object": obj, "brace_index": start} for arm, (start, obj) in active.items() if start < index]
        return out

    def attempt(self, index: int, action: Action, attempt: int, watch: Sequence[Tuple[Sequence[str], str]] = ()) -> Tuple[Optional[Verdict], Dict[str, Any]]:
        """Execute ``action`` once under the monitors and check its postcondition. ``watch``: contact pairs the
        monitors log for the caller (``ctx["watch_hits"]``, episode steps)."""
        ctx: Dict[str, Any] = {}
        try:
            targets = action.targets(self.sidecar, self.state)
        except Exception as e:  # noqa: BLE001
            v = Verdict("targets_failed", f"{action.name}.targets() raised {e!r}", action=action.name, action_index=index, attempt=attempt, step=self.n,
                        details={"traceback": traceback.format_exc()})
            self.stats.record(v)
            return v, ctx
        ctx["targets_keys"] = sorted(targets.keys())
        ctx["witnesses"] = self.witnesses_for(index)
        ctx["witness_poses"] = self.witness_poses_for(index)
        ov = self._witness_overrides.get(index)
        if ov:
            ctx["witnesses"].update(ov["witnesses"])
            ctx["witnesses_rebranched"] = {"arm": ov["arm"], "keys": sorted(ov["witnesses"]), "leg": ov["leg"]}
        if isinstance(action, (Transition, Navigate)) and targets.get("arms_q"):
            ctx["next_contact"] = {arm: self.next_contact(index, arm) for arm in targets["arms_q"]}
        if isinstance(action, Transition):
            ctx["stance_contacts"] = {arm: self.stance_contact(index, arm) for arm in M.ARMS}
        if isinstance(action, Grasp):
            ctx["next_place_kind"] = self.next_place_kind(index, action)
        ctx["sidecar"] = self.sidecar  # attachment: the hover alignment's regime travels in sidecar.extra["attach_correction"] (execute.attach_correction_regime)
        ctx["braces_held"] = self.braces_held(index)  # execute_push holds them in the world during a base nudge
        ctx["protect"] = {"goal": list(self.protected_goal), "force": index in self._protect_force, "task": getattr(self.sidecar, "variant", None) if self.sidecar is not None else None,
                          "index": index}
        if self.support_surface is not None:
            ctx["support_surface"] = self.support_surface
        intended = self._scene_names(intended_contacts(action))
        hooks: Dict[str, Any] = {}
        if self.qa is not None or self.monitors is not None:
            hooks["fov"] = self._set_fov
        if self.qa is not None:
            self.qa.begin_action(action, intended=intended, fov=fov_monitored(action))
        if self.monitors is not None:
            hooks["transit"] = lambda on, objects=(), label=None: self.monitors.begin_transit(objects, label) if on else self.monitors.end_transit()
            ctx["monitors"] = self.monitors  # experiments read collision_events for their notes (owner 2026-09-20 23:58)
            allowed = self.brace_allowed_pairs(index)
            ctx["brace_allowed"] = [(sorted(links), obj) for links, obj in allowed]
            self.monitors.begin_action(action, index, attempt, intended=intended, held=self.state.held_objects(),
                                       manipulation=is_manipulation(action), fov=fov_monitored(action), forbidden=self._scene_names(forbidden_contacts(action)),
                                       watch=[(links, next(iter(self._scene_names([obj])))) for links, obj in watch], allowed=allowed)
        if hooks:
            ctx["hooks"] = hooks
        try:
            gen = run_action(action, self.motions, self.state, targets, ctx)
            v = self.run_generator(gen, PHASE_ORACLE, index, attempt)
            if self.monitors is not None:
                ctx["watch_hits"] = list(getattr(self.monitors, "watch_hits", []))
            if v is not None:
                return v, ctx
            cat = check_postcondition(action, self.state, targets, ctx)
            if cat is not None:
                v = Verdict(cat, f"postcondition of {action.name} failed", action=action.name, action_index=index, attempt=attempt, step=self.n,
                            details={k: val for k, val in ctx.items() if k not in ("targets_keys", "cfg")})
                self.stats.record(v)
                return v, ctx
            return None, ctx
        except SkillError as e:
            v = Verdict(category_of(e), e.message or str(e), action=action.name, action_index=index, attempt=attempt, step=self.n, details=dict(e.details))
            self.stats.record(v)
            return v, ctx
        except Exception as e:  # noqa: BLE001
            log.exception("action %s[%d] attempt %d raised", action.name, index, attempt)
            v = Verdict("error", repr(e), action=action.name, action_index=index, attempt=attempt, step=self.n, details={"traceback": traceback.format_exc()})
            self.stats.record(v)
            return v, ctx
        finally:
            if self.monitors is not None:
                self.monitors.end_action()
            if self.qa is not None:
                self.qa.end_action()

    def _set_fov(self, enabled: bool = True) -> None:
        """``ctx["hooks"]["fov"]``: open / close the acting arms' gripper-in-FOV span, for the retry rule and the QA log."""
        if self.monitors is not None and hasattr(self.monitors, "set_fov_enabled"):
            self.monitors.set_fov_enabled(enabled)
        if self.qa is not None:
            self.qa.set_fov_enabled(enabled)

    def _commit(self) -> None:
        """Commit the recorder's pending segment unless a group (braced push) defers commits to its end."""
        if self.recorder is not None and not self._defer_commit:
            self.recorder.commit()

    def run_action(self, index: int, action: Action, watch: Sequence[Tuple[Sequence[str], str]] = ()) -> Optional[Verdict]:
        """Checkpoint → attempt → (restore → retry) → annotate. Returns the verdict that fails the episode, or None."""
        rec = ActionRecord(index=index, name=action.name, step_start=self.n)
        self.action_records.append(rec)
        fine, numbers = "", ""
        try:
            fine, numbers = action.annotate(self.sidecar)
        except Exception as e:  # noqa: BLE001
            log.warning("annotate() of %s failed: %r", action.name, e)
            fine = action.summary() if hasattr(action, "summary") else action.name
            numbers = fine
        if self.recorder is not None:
            self.recorder.begin_action_annotation(index, action, fine, numbers)
        cp = self.checkpoint(index)
        mark = self.recorder.mark() if self.recorder is not None else 0  # pending steps of an enclosing group stay
        verdict: Optional[Verdict] = None
        for k in range(self.cfg.retries + 1):
            t0 = time.time()
            verdict, ctx = self.attempt(index, action, k, watch=watch)
            self.last_ctx = ctx
            rec.targets_keys = ctx.get("targets_keys", [])
            rec.notes = {key: val for key, val in ctx.items() if key in NOTE_KEYS}
            offer = self._arm_offer_note(action)
            if offer is not None:
                rec.notes["arm_offer"] = offer
            left = getattr(self.motions, "last_witness_residual", None) if self.motions is not None else None
            if left is not None:  # an arm_to left a small witness residual unclosed (a witness on a joint limit)
                rec.notes["witness_residual_left"] = {"pos_m": round(float(left[0]), 4), "rot_rad": round(float(left[1]), 4)}
                self.motions.last_witness_residual = None
            events = list(getattr(self.motions, "release_events", None) or []) if self.motions is not None else []
            if events:  # the gripper events of this action: contact holds, releases (instant_open)
                rec.notes["release"] = events
                self.motions.release_events.clear()
            via = list(getattr(self.motions, "witness_via_posture_events", None) or []) if self.motions is not None else []
            seen = getattr(self, "_via_seen", 0)
            rec.notes["witness_via_posture"] = via[seen:] if len(via) > seen else "not fired"
            self._via_seen = len(via)
            rec.attempts.append(None if verdict is None else verdict.to_dict())
            if verdict is None:
                self._absorb_branch_decisions(index, ctx)
                self._commit()
                log.info("action %d %s ok in %d steps (%.1fs)", index, action.summary() if hasattr(action, "summary") else action.name, self.n - rec.step_start, time.time() - t0)
                break
            log.warning("action %d %s attempt %d failed: %s", index, action.name, k, verdict)
            if self.qa is not None and verdict.category in RULE_METRICS:
                self.qa.note_rule(verdict, retried=k < self.cfg.retries, arms=tuple(getattr(action, "arms", ()) or ()),
                                  objects=tuple(getattr(action, "objects", ()) or ()))
            if is_hard_qa(verdict):
                # a QA hard metric fails the demo at once: no retry, the attempt's steps stay in the record for diagnosis
                if self.recorder is not None:
                    self.recorder.commit()
                break
            if verdict.category in ("protected_contact", "protected_moved"):
                self._protect_force.add(index)  # the retry re-plans the transit via the point above its goal (oracle.protection)
            if (ctx.get("grasp_support_lift") or {}).get("applied") and self.motions is not None:
                # first-demo v2.1 A3: a grasp lifted off its support failed - the retry runs the recorded grasp
                if not hasattr(self.motions, "support_lift_refused"):
                    self.motions.support_lift_refused = set()
                self.motions.support_lift_refused.add(index)
            if k < self.cfg.retries:
                # roll back: the failed attempt leaves no trace in the demo (kept under rollbacks), retry from the checkpoint
                if self.recorder is not None:
                    self.recorder.rollback(index, k, verdict.to_dict(), state_before=cp.sim_state, mark=mark)
                    self.recorder.note_rollback(verdict.to_dict())
                self.restore(cp, index, k + 1)
            elif self.recorder is not None:
                # the last attempt failed: the episode fails with its verdict; its steps stay in the record for diagnosis
                self.recorder.commit()
        rec.step_end = self.n
        rec.verdict = None if verdict is None else verdict.to_dict()
        if self.recorder is not None:
            self.recorder.end_action_annotation(rec.verdict, attempts=len(rec.attempts))
        return verdict

    # ------------------------------------------------------------------ the loop

    def _init_sleep_assist(self) -> None:
        """``--sleep-assist`` (``ISG_SLEEP_ASSIST``, set by the driver's flags; off: None): the episode's tracked task objects, as
        run_isg_vec's ``SlotRecorder`` (the robot itself is never slept: ``sleep_assist.is_free_rigid``)."""
        self.sleep_assist = SA.SleepAssist(self.robot, self.tracked, "s0") if SA.sleep_assist_config() is not None else None

    def _init_support_load(self) -> None:
        """The support loading fact: every tracked object no program action manipulates (the table of a tabletop task)."""
        acted = set()
        for a in self.program_actions:
            acted.update(getattr(a, "objects", ()) or ())
        supports = {inst: obj for inst, obj in self.tracked.items() if inst not in acted and obj is not self.robot}
        if not supports:
            self.support_load = None
            return
        scene_idx = 0
        try:
            scene_idx = int(getattr(getattr(self.env, "scene", None), "idx", 0) or 0)
        except Exception:  # noqa: BLE001
            scene_idx = 0
        self.support_load = SupportLoad(self.robot, supports, scene_idx=scene_idx)

    def _init_qa(self, oracle: bool, actions: Optional[Sequence[Action]] = None) -> None:
        """``cfg.qa``: the episode's QA metrics (``object_tilt`` watches the articulated roots ``actions`` work). The arm stutter
        fact (``qa.ArmStutter``, ``result.arm_stutter``) is built whether ``--qa`` is on or not; the QA metric ``arm_stutter`` judges it."""
        self.qa = None
        try:
            self.arm_stutter = ArmStutter(self.robot)
        except Exception as e:  # noqa: BLE001  (a robot without arm controllers: no fact, no metric)
            log.warning("arm stutter monitor unavailable (%r)", e)
            self.arm_stutter = None
        if not self.cfg.qa:
            return
        watched, skipped = self.operated_roots(actions) if actions else ({}, {})
        camera = getattr(self.monitors, "camera", None)
        if camera is None and self.cam_key is not None:
            try:
                camera = CameraModel.from_sensor(self.robot.sensors[self.cam_key])
            except Exception:  # noqa: BLE001
                camera = None
        cam_in_zed = getattr(self.monitors, "cam_in_zed", None)
        if cam_in_zed is None:
            cam_in_zed = getattr(self.state, "cam_in_zed", None)
        pairs = (lambda: self.monitors._contact_pairs(self.n)) if self.monitors is not None else None
        limits = self.monitors.rule_limits() if self.monitors is not None and hasattr(self.monitors, "rule_limits") else None
        self.qa = QA(self.robot, oracle=oracle, scene=getattr(self.env, "scene", None), camera=camera, cam_in_zed=cam_in_zed,
                     roots=watched, roots_skipped=skipped, task_objects=[getattr(o, "name", n) for n, o in self.tracked.items() if o is not self.robot],
                     contact_pairs=pairs, rules=self.monitors is not None, rule_limits=limits)

    def operated_roots(self, actions: Optional[Sequence[Action]] = None) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """``(watched, skipped)``: the tracked articulated objects whose joint the program (``actions``, default the running
        program) works without moving them as a whole - the ``articulation.object`` of a handle grasp / base follow, the object
        of a push / brace / push-close carrying an ``articulation`` record (a lid push, a flap press, a push-close segment) -
        minus any object a place, a body grasp or a plain push moves on purpose. ``skipped`` names the operated objects not
        watched: fixed to the scene (an appliance, a room door: the root cannot move) or not tracked."""
        operated, moved = set(), set()
        for a in (self.program_actions if actions is None else actions):
            art = getattr(a, "articulation", None)
            if art is not None:
                operated.add(art.object)
            contact = getattr(a, "contact", None)
            articulated = contact is not None and (getattr(contact, "extra", None) or {}).get("articulation") is not None
            if articulated:
                operated.add(contact.object)
            if getattr(a, "name", None) == "push_close":
                operated.add(a.object)
            if isinstance(a, Place):
                moved.add(a.object)
            elif isinstance(a, Push) and not articulated:
                moved.add(a.contact.object)
            elif getattr(a, "name", None) == "grasp" and art is None:
                moved.update(getattr(a, "objects", ()) or ())
        watched: Dict[str, Any] = {}
        skipped: Dict[str, str] = {}
        for inst in sorted(operated - moved):
            obj = self.tracked.get(inst)
            if obj is None or obj is self.robot:
                skipped[inst] = "not tracked"
            elif bool(getattr(obj, "fixed_base", False)):
                skipped[inst] = "fixed base"
            else:
                watched[inst] = obj
        return watched, skipped

    def run_program(self, actions: Sequence[Action]) -> Dict[str, Any]:
        if self.cfg.physics_only:
            from ispatialgym.oracle import physics_only as PO

            with PO.episode(self.sim):
                return self._run_program(actions)
        return self._run_program(actions)

    def _run_program(self, actions: Sequence[Action]) -> Dict[str, Any]:
        t0 = time.time()
        self._wall_t0 = t0
        actions = list(actions)
        self.program_actions = actions
        self._stream = False
        self._init_support_load()
        self._init_qa(oracle=True, actions=actions)
        self._init_sleep_assist()
        # every tracked object's pose at the start (the loaded init): the reference for a partner displaced during the run
        # (attachment 0920-164504: the bottle knocked 4.7 cm during the grasp phase, the place aimed at its recorded pose)
        init_poses: Dict[str, Tuple[th.Tensor, th.Tensor]] = {}
        for inst, obj in self.tracked.items():
            if obj is self.robot:
                continue
            try:
                init_poses[inst] = tuple(x.clone().float() for x in obj.get_position_orientation())
            except Exception:  # noqa: BLE001
                continue
        self.state.init_poses = init_poses
        if self.motions is not None:
            self.motions.keep_hold_arms = set()
        verdict: Optional[Verdict] = None
        i = 0
        while i < len(actions):
            action = actions[i]
            if i == len(actions) - 1:
                self._keep_final_hold(action)
            if isinstance(action, Brace) and i + 1 < len(actions) and isinstance(actions[i + 1], Push) and self.cfg.brace_retries > 0:
                verdict = self.run_braced_push(i, action, actions[i + 1])
                i += 2
            else:
                verdict = self.run_action(i, action)
                i += 1
            if verdict is not None:
                break
        self.timings["actions_s"] = time.time() - t0
        n_action_steps = self.n
        return self.epilogue(verdict, n_action_steps)

    # ------------------------------------------------------------------ braced push: the brace retry ladder

    def push_outcome(self, push: Push, ctx: Dict[str, Any], brace: Optional[Brace] = None) -> Dict[str, Any]:
        """How the push ended against the task goal: ``residual`` (signed in the push sense: > 0 short, < 0
        overshoot), ``tolerance`` (the TRO target's, None when the object has no gating target of the push's kind),
        ``in_tolerance``, ``brace_hits`` (steps the watched brace fingers touched the target before the goal first
        held) and the ladder ``reason``: ``"short"`` (out of tolerance, short, the target touched the stop before the
        goal), ``"overshoot"`` (out of tolerance, the residual flipped sign) or None. ``brace`` (the push's brace): with
        :func:`top_press_accept_on` an in-tolerance push braced by a ``top_press`` is accepted as it is (``accepted``; no rung)."""
        res = ctx.get("push_residual") or {}
        value = res.get("final", res.get("after", res.get("value")))  # the residual the object is left with after the back-off
        kind = res.get("kind", "line")
        tgt = self.state.pose_target(push.contact.object) if hasattr(self.state, "pose_target") else None
        tol = None
        if tgt is not None:
            tol = getattr(tgt, "orientation_tolerance", None) if kind == "arc" else getattr(tgt, "position_tolerance", None)
        out: Dict[str, Any] = {"residual": value, "kind": kind, "tolerance": tol, "in_tolerance": None, "brace_hits": 0, "reason": None}
        if value is None or tol is None:
            return out
        signed = float(value) * (1.0 if kind == "line" else math.copysign(1.0, float(res.get("angle_sign", 1.0))))
        out["residual"] = signed
        out["in_tolerance"] = abs(signed) <= float(tol)
        first = self.isg.get("goal_first_step")
        hits = [h for h in (ctx.get("watch_hits") or []) if first is None or h < first]
        out["brace_hits"] = len(hits)
        # rush 2026-09-17: the ladder aims at a fraction of the tolerance (brace_retry_aim_frac) - a two-pad squeeze
        # relaxes by ~0.07 rad when the pads leave (painting 0917-220043: push ended at +0.053 of 0.08, final 0.129);
        # a residual inside the tolerance but above the aim still climbs the ladder while retries remain
        # arcs: the aim is biased to the overshoot side by the expected relaxation (brace_retry_aim_bias_rad; the pads'
        # penetration recovers toward the short side once they leave); lines keep the aim centred on the goal
        centre = -float(getattr(self.cfg, "brace_retry_aim_bias_rad", 0.0)) if kind == "arc" else 0.0
        half = float(tol) * float(getattr(self.cfg, "brace_retry_aim_frac", 1.0))
        out["aim"] = {"centre": centre, "half": half}
        out["aimed"] = abs(signed - centre) <= half
        if out["in_tolerance"] and out["aimed"]:
            return out
        if out["in_tolerance"] and brace is not None and str(getattr(brace.contact, "kind", "")) == "top_press" and top_press_accept_on():
            # owner 2026-11-08 (answers b, item 2): a top press holds the object down, it is no stop on the push path - an improvement rung
            # moves it "outward" along -direction, i.e. UP, off the object (close_02: +10 mm and the carton slid); in tolerance is accepted
            out["accepted"] = "top press in tolerance: accepted as it is (no improvement rung lifts the press; owner 2026-11-08)"
            self.counters["push_outcome.top_press_in_tolerance_accepted"] = self.counters.get("push_outcome.top_press_in_tolerance_accepted", 0) + 1
            log.info("braced push: top press, residual %.4f within the tolerance %.4f (aim %+.3f +- %.3f): accepted, no retry", signed, float(tol), centre, half)
            return out
        if signed < centre:
            out["reason"] = "overshoot"
        elif hits:
            out["reason"] = "short"  # the target touched the brace before the goal: the stop stands too close - move it outward
        elif out["in_tolerance"]:
            # owner 2026-09-26 (world vertical 0925-193257 / 225052 / 235643: every rung 'short' with brace hits 0 - the pad stops at
            # the goal and the painting sags back 0.015-0.035 rad when it leaves; moving the brace outward cannot change that, the
            # extra pushes were only more chances for a plan failure): a free short inside the tolerance is accepted as it is
            out["accepted"] = "in tolerance, no brace catch: accepted without improvement rungs (the epilogue's settled check decides)"
        else:
            out["reason"] = "short_free"  # out of tolerance, the brace untouched: push further (aim_push_past_goal by the measured sag)
        return out

    def run_braced_push(self, index: int, brace: Brace, push: Push) -> Optional[Verdict]:
        """``brace`` then ``push`` as one group with the **brace retry ladder** (owner decisions 2026-09-16 05:26 /
        05:39): when the push ends outside the goal tolerance and (a) the target touched the brace finger before the
        goal (``short``: the stop too close) the group is rolled back to the checkpoint taken before the brace and
        re-run with the brace +5 mm then +10 mm outward along −``direction``; (b) the signed residual flipped
        (``overshoot``: the stop too far) → −5 mm then −10 mm (inward); two retries at most, then the epilogue's
        ``tolerance`` verdict. Recorded in ``meta.brace_retry``."""
        cp = self.checkpoint(index)
        mark = self.recorder.mark() if self.recorder is not None else 0
        first_record = len(self.action_records)
        ladder = {"short": self.cfg.brace_retry_offsets_m, "overshoot": tuple(-o for o in self.cfg.brace_retry_offsets_m)}
        retries: List[Dict[str, Any]] = []
        history: List[Tuple[float, str]] = []  # (brace offset, reason) of every out-of-tolerance push so far
        offset = 0.0
        shorten = 0.0  # rad the recorded arc is shortened by (overshoot with no brace catch: push less, brace unchanged)
        extend = 0.0  # rad the pad aims past the goal (short with no brace catch, out of tolerance: push further, brace unchanged)
        verdict: Optional[Verdict] = None
        best_in_tol: Optional[Tuple[float, float, float]] = None  # (offset, shorten, extend) of the last rung whose push ended IN tolerance
        fell_back = False
        self._defer_commit = True
        try:
            k = 0
            while k <= self.cfg.brace_retries:
                brace_k = brace if offset == 0.0 else offset_brace(brace, offset)
                push_k = shorten_push(push, shorten) if shorten > 0 else (aim_push_past_goal(push, extend) if extend > 0 else push)
                verdict = self.run_action(index, brace_k)
                if verdict is None:
                    watch = [(list(M.FINGER_LINKS[brace.contact.arm]) + [M.GRIPPER_LINK[brace.contact.arm]], push.contact.object)]
                    verdict = self.run_action(index + 1, push_k, watch=watch)
                if verdict is not None:
                    if best_in_tol is not None and not fell_back and (offset, shorten, extend) != best_in_tol and not is_hard_qa(verdict):
                        # owner 2026-09-25 (world vertical 0925-225052, wave 3): three rungs ended IN tolerance (0.025-0.035 rad of the
                        # 0.08) but above the aim, the fourth re-brace at -15 mm left the push arm no plan ('trajectory optimization to the
                        # joint goal failed' twice) and the demo failed although the painting had stood aligned three times. An improvement
                        # rung that fails outright goes back to the rung that was in tolerance: rolled back and re-run once, whatever it yields
                        log.warning("braced push: the improvement rung at %+.0f mm failed (%s) - back to the rung that was in tolerance (%+.0f mm, arc shortened %.3f rad)",
                                    -offset * 1000, verdict.category, -best_in_tol[0] * 1000, best_in_tol[1])
                        retries.append({"attempt": k, "offset_m": offset, "reason": f"rung failed: {verdict.category}", "residual": None, "tolerance": None,
                                        "brace_hits": None, "actions": [r.to_dict() for r in self.action_records[first_record:]], "next_offset_m": best_in_tol[0]})
                        del self.action_records[first_record:]
                        if self.recorder is not None:
                            self.recorder.rollback(index, k, {"category": verdict.category, "message": f"braced push rung failed: {verdict.message}"}, state_before=cp.sim_state, mark=mark)
                            self.recorder.note_rollback({"category": verdict.category, "brace_retry": "rung failed"})
                        self.restore(cp, index, k + 1)
                        offset, shorten, extend = best_in_tol
                        fell_back, verdict = True, None
                        k += 1
                        continue
                    break
                outcome = self.push_outcome(push, self.last_ctx, brace=brace)
                outcome["shortened_rad"] = shorten
                self.action_records[-1].notes["push_outcome"] = outcome
                if outcome.get("in_tolerance"):
                    best_in_tol = (offset, shorten, extend)
                if outcome["reason"] is None or k >= self.cfg.brace_retries or fell_back:
                    if outcome["reason"] is not None:
                        log.warning("braced push: %s residual %.4f > tolerance %.4f after %d brace retries; the epilogue decides", outcome["reason"], outcome["residual"], outcome["tolerance"], k)
                    break
                if outcome["reason"] != "short_free":
                    history.append((offset, outcome["reason"]))
                centre = float((outcome.get("aim") or {}).get("centre", 0.0))
                if outcome["reason"] == "short_free":
                    if not (self.cfg.push_extend_on_free_short and push.contact.mid is not None):
                        log.warning("braced push: short with no brace catch (residual %.4f > tolerance %.4f) and no arc to lengthen; the epilogue decides", outcome["residual"], outcome["tolerance"])
                        break
                    # the painting sags back by about the final residual once the pad leaves: aim the pad that far past the goal
                    extend += max(float(outcome["residual"]) - centre, 0.0)
                    new_offset = offset
                    log.warning("braced push: short with no brace catch (residual %.4f, tolerance %.4f) -> the pad aims %.3f rad past the goal and re-pushes (brace unchanged)",
                                outcome["residual"], outcome["tolerance"], extend)
                elif outcome["reason"] == "overshoot" and outcome["brace_hits"] == 0 and self.cfg.push_shorten_on_free_overshoot and push.contact.mid is not None:
                    # the target swept past the goal without touching the brace: the pad's arc rotates it too far - push
                    # less (aim at the ladder centre), keep the brace where it is
                    shorten += abs(float(outcome["residual"])) - abs(centre)
                    new_offset = offset
                    log.warning("braced push: overshoot (residual %.4f, tolerance %.4f, no brace catch) -> the arc is shortened by %.3f rad in total and re-pushed",
                                outcome["residual"], outcome["tolerance"], shorten)
                else:
                    over = [o for o, r in history if r == "overshoot"]
                    short = [o for o, r in history if r == "short"]
                    if over and short:
                        new_offset = 0.5 * (over[-1] + short[-1])  # the residual flipped sign: bisect the bracket
                    else:
                        same = [o for o, r in history if r == outcome["reason"]]
                        rungs = ladder[outcome["reason"]]
                        new_offset = float(rungs[len(same) - 1]) if len(same) <= len(rungs) else float(same[-1] + (rungs[-1] - rungs[-2]))
                    log.warning("braced push: %s (residual %.4f, tolerance %.4f, brace hits %d) -> re-brace at %+.0f mm along the brace direction and re-push",
                                outcome["reason"], outcome["residual"], outcome["tolerance"], outcome["brace_hits"], -new_offset * 1000)
                dropped = [r.to_dict() for r in self.action_records[first_record:]]
                del self.action_records[first_record:]
                retries.append({"attempt": k, "offset_m": offset, "reason": outcome["reason"], "residual": outcome["residual"], "tolerance": outcome["tolerance"],
                                "brace_hits": outcome["brace_hits"], "actions": dropped, "next_offset_m": new_offset})
                if self.recorder is not None:
                    self.recorder.rollback(index, k, {"category": "tolerance", "message": f"braced push {outcome['reason']}: residual {outcome['residual']:.4f}"}, state_before=cp.sim_state, mark=mark)
                    self.recorder.note_rollback({"category": "tolerance", "brace_retry": outcome["reason"]})
                self.restore(cp, index, k + 1)
                offset = new_offset
                k += 1
        finally:
            self._defer_commit = False
        if retries:
            self.brace_retries.append({"action_index": index, "retries": retries, "final_offset_m": offset,
                                       "final": (self.action_records[-1].notes.get("push_outcome") if len(self.action_records) > first_record else None),
                                       "verdict": None if verdict is None else verdict.to_dict()})
        if verdict is None:
            self._commit()
        return verdict

    def run_stream(self, actions: Iterable[th.Tensor], phase: int = PHASE_ORACLE) -> Dict[str, Any]:
        """Replay / policy modes: step a stream of 23-D actions (no checkpoints, no program, no epilogue: a replay streams its
        demo's settle rows, a policy has none) until it ends, ``max_steps``, the wall clock or - ``stop_on_success`` - the run
        mode's geometric success; then the task's metrics judge the episode (:meth:`epilogue`)."""
        t0 = time.time()
        verdict: Optional[Verdict] = None
        self._stream = True
        self._init_qa(oracle=False, actions=self.qa_program)
        self._init_sleep_assist()
        if self.monitors is not None:
            self.monitors.begin_action(_StreamAction(), 0, 0, intended=(), held=self.state.held_objects(), manipulation=False, fov=False)
        phases = self.stream_phases
        try:
            for k, a in enumerate(actions):
                if self.n >= self.cfg.max_steps:
                    log.info("stream: max_steps %d reached; the task's metrics judge the episode", self.cfg.max_steps)
                    break
                if self.cfg.max_wall_s and time.time() - t0 > self.cfg.max_wall_s:
                    self.wall_clock_exceeded = True
                    log.info("stream: wall clock of %.0f s reached after %d steps; the task's metrics judge the episode", self.cfg.max_wall_s, self.n)
                    break
                a = th.as_tensor(a, dtype=th.float32).reshape(-1)
                self._last_stream_action = a.clone()
                settle_row = phases is not None and k < len(phases) and int(phases[k]) == PHASE_SETTLE
                verdict = self.step(a, PHASE_SETTLE if settle_row else phase, 0, 0)  # a replayed settle row is recorded as one
                if verdict is not None:
                    break
                if self.cfg.stop_on_success and self.isg.get("official_success") and self.isg.get("success_final", True):
                    # the evaluated success step is chosen within the settle window after the first success (eval.metrics)
                    log.info("stream: geometric success (first at step %s, evaluated at step %s) final; the episode ends",
                             self.isg.get("geometric_first_step"), self.isg.get("geometric_success_step"))
                    break
        finally:
            if self.monitors is not None:
                self.monitors.end_action()
        if self.recorder is not None:
            self.recorder.commit()
        self.timings["actions_s"] = time.time() - t0
        return self.epilogue(verdict, self.n)

    def _keep_final_hold(self, action: Action) -> None:
        """The program's last action moves an arm that holds an object and does not place it (the held-object
        reorient): that hand keeps its object to the end, so neither the leg's jam probe nor the epilogue relaxes it."""
        arm = getattr(action, "arm", None)
        if self.motions is None or arm not in M.ARMS or isinstance(action, Place) or self.state.held_object(arm) is None:
            return
        self.motions.keep_hold_arms.add(arm)

    def _epilogue_hold(self, steps: int, relax_info: Dict[str, Any]) -> Iterator[th.Tensor]:
        """The first no-op window of the epilogue (``steps`` steps in all): the arms' hold targets relaxed onto their
        measured joints over its first ``epilogue_relax_steps`` steps (``epilogue_relax_arms``, program mode - a stream
        has no Layer-1 targets to relax), then no-op steps. An arm still holding an object is not relaxed (the payload
        would sag). ``relax_info`` receives ``{arm: largest target change (rad)}``."""
        motions = self.motions
        n_relax = 0
        if self.cfg.epilogue_relax_arms and motions is not None and getattr(self, "_last_stream_action", None) is None and hasattr(motions, "relax_arms"):
            hold = getattr(motions, "press_hold", None) or {}
            arms = [a for a in M.ARMS if hold.get("arm") != a and a not in getattr(motions, "keep_hold_arms", ()) and self.state.held_object(a) is None]
            n_relax = min(int(self.cfg.epilogue_relax_steps), int(steps))
            if arms and n_relax > 0:
                relaxed = yield from motions.relax_arms(arms, steps=n_relax)
                relax_info.update({a: round(float(v), 5) for a, v in (relaxed or {}).items()})
                moved = {a: v for a, v in relax_info.items() if v > 1e-3}
                if moved:
                    log.info("epilogue: arm hold targets relaxed onto the measured joints by %s rad (a drive was still pressing)", moved)
            else:
                n_relax = 0
        yield from self._noop_actions(int(steps) - n_relax)

    def _noop_actions(self, steps: int) -> Iterator[th.Tensor]:
        """The epilogue's no-op: the Layer-1 hold (program mode) or the last streamed action with the base stopped."""
        last = getattr(self, "_last_stream_action", None)
        if last is not None:
            hold = last.clone()
            idx = getattr(self.robot, "controller_action_idx", {}).get("base")
            if idx is not None:
                hold[idx] = 0.0
            for _ in range(int(steps)):
                yield hold.clone()
        elif self.motions is not None:
            yield from self.motions.hold(steps)
        else:
            yield from _hold_actions(self.robot, steps)

    # ------------------------------------------------------------------ epilogue (§3.8 / §6.2)

    def placed_objects(self) -> List[str]:
        """The bddl instances the program places (``Place`` of any kind: set down, drop, attach), in program order."""
        out: List[str] = []
        for a in self.program_actions:
            if isinstance(a, Place) and a.object not in out:
                out.append(a.object)
        return out

    def placed_orientation_report(self, isg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Per placed object: the evaluator's final orientation error (its metric and equivalences) against the
        recorded TRO orientation target and tolerance (``demo_orientation_tolerance_default`` when the target has
        none); ``within`` is None when the object has no orientation target or no error report.

        **Cylinders** (owner 2026-09-16 17:10 item 4, fallback): a placed object whose model carries a continuous symmetry
        about its long axis in the ICL symmetry survey (:func:`revolution_axis`) is judged modulo rotation about that
        axis — ``orientation_error`` becomes the angle between the axis now and at the goal (nearest recorded
        equivalence, :meth:`axis_error`), ``full_error`` keeps the evaluator's reading, ``equivalence`` names the axis;
        every other object keeps the full error.

        **Push-under targets** (owner 2026-10-13 ~04:30 item 1c): an object the program pushes under its partner (a ``Push`` of kind
        ``push_under``, read off the program the record expands to) is judged modulo rotation about the world vertical - the yaw the
        floor push gives it is free, its tilt stays gated: ``orientation_error`` becomes :meth:`vertical_error` (the smaller of the two
        readings for a cylinder), ``full_error`` and ``equivalence`` as for the cylinders."""
        errs = (isg.get("errors") or {}).get("pose_errors") or {}
        out: Dict[str, Dict[str, Any]] = {}
        pushed_under = self.pushed_under_objects()
        for obj in self.placed_objects():
            tgt = self.state.pose_target(obj) if hasattr(self.state, "pose_target") else None
            rep = errs.get(obj) if isinstance(errs.get(obj), dict) else {}
            err = rep.get("orientation_error")
            has_target = tgt is not None and getattr(tgt, "orientation", None) is not None
            if not has_target or err is None:
                out[obj] = {"orientation_error": None if err is None else float(err), "tolerance": None, "within": None,
                            "note": "no orientation target" if not has_target else "no orientation error reported"}
                continue
            tol = getattr(tgt, "orientation_tolerance", None)
            tol = float(self.cfg.demo_orientation_tolerance_default if tol is None else tol)
            entry: Dict[str, Any] = {"orientation_error": float(err), "tolerance": tol, "within": float(err) <= tol, "metric": getattr(tgt, "orientation_metric", None),
                                     "target_in_goal": bool(rep.get("in_goal", False))}
            axis = self.revolution_axis_of(obj) if self.cfg.demo_revolution_axes else None
            if axis is not None and str(getattr(tgt, "frame", "world")) == "world":
                ax_err = self.axis_error(obj, tgt, axis["axis"])
                if ax_err is not None:
                    entry.update({"orientation_error": ax_err, "within": ax_err <= tol, "full_error": float(err),
                                  "equivalence": f"rotation about the {axis['axis']} axis (revolution, survey tags {axis.get('tags')})",
                                  "revolution_axis": axis["axis"], "model": axis.get("key")})
            if obj in pushed_under and str(getattr(tgt, "frame", "world")) == "world":
                v_err = self.vertical_error(obj, tgt)
                if v_err is not None and ("revolution_axis" not in entry or v_err < float(entry["orientation_error"])):
                    entry.update({"orientation_error": v_err, "within": v_err <= tol, "full_error": float(err),
                                  "equivalence": "rotation about the world vertical (push-under: the yaw is free, the tilt is gated)"})
            out[obj] = entry
        return out

    def pushed_under_objects(self) -> set:
        """The bddl instances the program pushes under their partner (a ``Push`` of kind ``push_under``)."""
        return {a.contact.object for a in self.program_actions if isinstance(a, Push) and a.kind == "push_under"}

    def vertical_error(self, obj: str, tgt) -> Optional[float]:
        """:func:`vertical_angle` of a placed object's orientation now against its recorded goal orientation and equivalences."""
        try:
            _, q_cur = self.state.object_pose(obj)
        except Exception:  # noqa: BLE001
            return None
        return vertical_angle(q_cur, getattr(tgt, "orientation"), getattr(tgt, "orientation_equivalences", None) or ())

    def revolution_axis_of(self, obj: str) -> Optional[Dict[str, Any]]:
        """The :func:`revolution_axis` entry of a placed object's model (``category`` / ``model`` of the scene object,
        else parsed from its name ``<category>_<model>_<i>``), None without a tag or without the object."""
        try:
            o = self.state.object(obj)
        except Exception:  # noqa: BLE001
            return None
        cat, model = getattr(o, "category", None), getattr(o, "model", None)
        name = str(getattr(o, "name", "") or "")
        if (not cat or not model) and name:
            parts = name.rsplit("_", 2)
            if len(parts) == 3 and parts[2].isdigit():
                cat, model = cat or parts[0], model or parts[1]
        if not cat or not model:
            return None
        entry = revolution_axis(str(cat), str(model))
        if entry is None:
            return None
        return {**entry, "key": f"{cat}/{model}"}

    def axis_error(self, obj: str, tgt, axis: str) -> Optional[float]:
        """The angle (rad) between a placed object's object-frame ``axis`` now and at the recorded goal orientation,
        minimised over the target's orientation equivalences (the error modulo rotation about that axis)."""
        try:
            _, q_cur = self.state.object_pose(obj)
        except Exception:  # noqa: BLE001
            return None
        a = th.zeros(3)
        a["xyz".index(axis)] = 1.0
        goal_q = C.as_tensor(getattr(tgt, "orientation")).float()
        a_goal = C.quat_rotate(goal_q, a)
        best = None
        for e in [None] + list(getattr(tgt, "orientation_equivalences", None) or []):
            q = q_cur if e is None else C.qmul(C.as_tensor(q_cur).float(), C.as_tensor(e).float())
            a_now = C.quat_rotate(C.as_tensor(q).float(), a)
            ang = math.acos(max(-1.0, min(1.0, float(th.dot(a_now, a_goal)))))
            best = ang if best is None or ang < best else best
        return None if best is None else float(best)

    def orientation_targets(self, isg: Optional[Dict[str, Any]] = None) -> set:
        """Instances whose goal has an orientation target: the evaluator's report says so (``pose_errors[inst]
        .orientation_error`` is a number, None for a position-only / relation-only target), else the TRO meta."""
        isg = self.isg if isg is None else isg
        out = set()
        rep = ((isg.get("errors") or {}).get("pose_errors") or {})
        for inst, r in rep.items():
            if isinstance(r, dict) and r.get("orientation_error") is not None:
                out.add(inst)
        tro = getattr(self.task, "_tro", None)
        meta = getattr(tro, "meta", None)
        for inst, tgt in (getattr(meta, "pose_targets", None) or {}).items():
            if getattr(tgt, "orientation", None) is not None:
                out.add(inst)
        return out

    def _reset_predicate_branches(self) -> None:
        """Clear the D-007 predicate branch log (``ISpatialGymTask.reset_predicate_branches``) so the next evaluations are
        the ones ``_predicate_branches`` reports; a task without the wrapper is left alone."""
        fn = getattr(self.task, "reset_predicate_branches", None)
        if callable(fn):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                log.warning("reset_predicate_branches failed: %r", e)

    def _predicate_branches(self) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """``({"under(saucer, teacup)": "rays", ...}, log tail)``: the wrapper branch that decided each under / ontop atom
        in the latest evaluation (``ISpatialGymTask.predicate_branches``; ``{}`` without the wrapper) and the last six
        under / ontop log entries (``ispatialgym.sampling.predicates.branches``) for the ledger."""
        fn = getattr(self.task, "predicate_branches", None)
        branch: Dict[str, str] = {}
        if callable(fn):
            try:
                branch = {str(k): str(v) for k, v in (fn() or {}).items()}
            except Exception as e:  # noqa: BLE001
                log.warning("predicate_branches failed: %r", e)
        tail: List[Dict[str, Any]] = []
        if branch:
            try:
                from ispatialgym.sampling import predicates as PRED  # lazy: the sampler package

                tail = [_jsonable(e) for e in PRED.branches() if e.get("predicate") in ("under", "ontop")][-6:]
            except Exception:  # noqa: BLE001
                tail = []
        return branch, tail

    def metric_verdict(self, isg: Dict[str, Any]) -> Optional[Verdict]:
        """The episode's verdict from the task's metrics (``ispatialgym.eval.metrics``; the run's settle-eval mode): None when the
        run mode's ``success_metric`` success was reached; ``symbolic_only`` - the goal held (symbolic success) but the measured
        targets were never within tolerance at the same step(s); ``not_settled`` (settle-eval on) - the goal held but never
        through a settled window; ``not_reached`` - the goal never held."""
        geometric, symbolic = bool(isg.get("geometric_success")), bool(isg.get("symbolic_success"))
        if (geometric if self.cfg.success_metric == "geometric" else symbolic):
            return None
        evaluation = isg.get("evaluation") or {}
        if symbolic:
            outside = [m["bddl"] for m in (evaluation.get("measured") or []) if not m.get("within_tolerance")]
            return Verdict("symbolic_only", f"the goal held (symbolic success at step {isg.get('symbolic_success_step')}) but the measured targets were "
                                            "not within tolerance at the same step(s); at the evaluation step outside: " + (", ".join(outside) or "none"),
                           step=self.n, details={"outside": outside, "symbolic_success_step": isg.get("symbolic_success_step"),
                                                 "error_metric": evaluation.get("error_metric")})
        if isg.get("settle_eval") and isg.get("goal_first_step") is not None:
            return Verdict("not_settled", f"the goal held (first at step {isg.get('goal_first_step')}) but never through a settled window of "
                                          f"{isg.get('settle_steps')} steps", step=self.n, details={"goal_first_step": isg.get("goal_first_step")})
        return Verdict("not_reached", "the goal never held; unsatisfied at the last step: " + ", ".join(_unsatisfied(isg)), step=self.n)

    def epilogue(self, verdict: Optional[Verdict], n_action_steps: int) -> Dict[str, Any]:
        """The end of an episode: a program's settle window (exactly ``settle_steps`` no-op steps, reserved in the step budget;
        none after a timeout or a hard QA verdict, none for a stream), then the verdict - the first one the actions or the window
        gave, else the task's metrics (:meth:`metric_verdict`) - the QA metrics and the result."""
        t0 = time.time()
        if self.monitors is not None:
            self.monitors.begin_epilogue(held=self.state.held_objects())
        window = 0 if self._stream else int(self.cfg.settle_steps)
        runs_window = window > 0 and (verdict is None or (verdict.category != "timeout" and not is_hard_qa(verdict)))
        before = MotionSnapshot(self.robot, self.tracked)
        settle_verdict = None
        relax_info: Dict[str, Any] = {}
        self._reset_predicate_branches()  # D-007: the branch log of the final evaluation starts with the settle window
        if runs_window:
            settle_verdict = self.run_generator(self._epilogue_hold(window, relax_info), PHASE_SETTLE)
            if self.recorder is not None:
                self.recorder.commit()
        motion = before.delta(MotionSnapshot(self.robot, self.tracked))
        isg = self.isg
        binary_branch, branch_log = self._predicate_branches()  # D-007: which wrapper branch decided each under / ontop atom at the last step
        if binary_branch:
            log.info("epilogue: goal atoms decided by %s", binary_branch)
        if self.qa is not None and settle_verdict is not None and settle_verdict.category in RULE_METRICS:
            self.qa.note_rule(settle_verdict, retried=False)  # the settle window has no retry
        if verdict is None and settle_verdict is not None:
            verdict = settle_verdict
        if verdict is None:
            verdict = self.metric_verdict(isg)
            if verdict is not None:
                self.stats.record(verdict)
        stutter_report = self.arm_stutter.report() if self.arm_stutter is not None else {}
        # -- the placed objects' orientation against their recorded targets: a fact of an oracle demo, no verdict
        placed_orientation = self.placed_orientation_report(isg) if (self.program_actions and not self._stream) else None
        self.timings["settle_s"] = time.time() - t0
        goal_obs = None
        # 2026-10-31 train-fast (PHYSICS_ONLY_TRIM): in a physics-only run the goal image (10 renders) and the goal-occlusion depth
        # renders are the last renders left; neither is taken - the result notes it and the gate records "skipped" without judging
        po_trim = bool(self.cfg.physics_only) and fast_on("PHYSICS_ONLY_TRIM")
        if verdict is None and self.cfg.goal_image and po_trim:
            goal_obs = {"skipped": "physics_only", "step": self.n}
            log.info("train-fast PHYSICS_ONLY_TRIM: goal image not rendered (physics-only: no frame is recorded)")
        elif verdict is None and self.cfg.goal_image:
            t1 = time.time()
            try:
                goal_obs = self.capture_goal_image()
            except Exception as e:  # noqa: BLE001
                log.warning("goal image not captured: %r", e)
                goal_obs = {"error": repr(e)}
            self.timings["goal_image_s"] = time.time() - t1
        # -- QA: the episode measures (jerky motion, head camera upright, goal occlusion, arm stutter); a hard one fails an oracle demo
        qa_report: Dict[str, Any] = {"enabled": False}
        if self.qa is not None:
            from ispatialgym.eval import goal_occlusion as GO

            t1 = time.time()
            measure = self.goal_occlusion_report if (GO.enabled() and (self._stream or verdict is None)) else None
            if measure is not None and po_trim:
                # train-fast PHYSICS_ONLY_TRIM: the goal-occlusion depth renders are not taken; recorded as skipped, never judged
                measure = lambda: {"skipped": "physics_only", "occluded": None, "method": "none (physics-only: no depth render)", "step": self.n}  # noqa: E731
                log.info("train-fast PHYSICS_ONLY_TRIM: goal_occlusion not measured (it would render depth; physics-only) - the demo is not judged by it")
            self.qa.note_arm_stutter(stutter_report)
            qa_verdict = self.qa.finish(goal_occlusion=measure, step=self.n)
            self.timings["qa_finish_s"] = time.time() - t1
            if verdict is None and qa_verdict is not None:
                verdict = qa_verdict
                self.stats.record(verdict)
            qa_report = self.qa.report()
        self.verdict = verdict
        evaluation = isg.get("evaluation") or {}
        self.result = {
            "success": verdict is None,
            "verdict": None if verdict is None else verdict.to_dict(),
            "reason": None if verdict is None else verdict.category,
            "success_metric": self.cfg.success_metric,
            "settle_eval": isg.get("settle_eval"),
            "settle_steps": isg.get("settle_steps"),
            "symbolic_success": bool(isg.get("symbolic_success")),
            "symbolic_success_step": isg.get("symbolic_success_step"),
            "geometric_success": bool(isg.get("geometric_success")),
            "geometric_success_step": isg.get("geometric_success_step"),
            "symbolic_first_step": isg.get("symbolic_first_step"),
            "geometric_first_step": isg.get("geometric_first_step"),
            "evaluation_step": evaluation.get("step"),
            "evaluation_kind": evaluation.get("kind"),
            "evaluation_settled": evaluation.get("settled"),
            "success_final": isg.get("success_final"),
            "closest_approach": isg.get("closest_approach"),
            "closest_goal_approach": isg.get("closest_goal_approach"),
            # the errors the run mode is judged by (at its evaluated geometric step, else at its evaluated symbolic step, else at the
            # last step) and the last step's
            "errors": evaluation.get("errors") or isg.get("errors") or {},
            "errors_final_step": isg.get("errors") or {},
            "error_metric": evaluation.get("error_metric"),
            "measured": evaluation.get("measured"),
            "other_mode": isg.get("other_mode"),
            "goal_first_step": isg.get("goal_first_step"),
            # D-007 (owner 2026-09-16 22:05 item 4): the wrapper branch (omnigibson / rays / geometry / none) that decided every
            # under / ontop atom at the final evaluation; ``binary_branch_fallback`` lists the atoms a fallback branch passed
            "binary_branch": binary_branch,
            "binary_branch_fallback": sorted(k for k, v in binary_branch.items() if v not in ("omnigibson",)),
            "binary_branch_log": branch_log,
            # owner 2026-09-16 23:00 (fact, no verdict): the support's pose drift and the finger-support contact loading
            "support_disturbance": self.support_load.report() if self.support_load is not None else {},
            "wall_clock_exceeded": bool(self.wall_clock_exceeded),
            # owner 2026-10-13: repeated jolts of an arm's commanded joint motion (the QA hard metric arm_stutter)
            "arm_stutter": stutter_report,
            # owner 2026-11-01: run_isg_vec's sleep assist in the demos too (what it slept / woke, per tracked object; None when off)
            "sleep_assist": self.sleep_assist.report() if self.sleep_assist is not None else None,
            "pose_settled_config": isg.get("pose_settled_config"),
            "conditions": isg.get("conditions"),
            "group_assignments": isg.get("group_assignments"),
            "n_steps": self.n,
            "n_action_steps": n_action_steps,
            "n_settle_steps": self.phase_steps.get(PHASE_SETTLE, 0),
            "epilogue_steps": window if runs_window else 0,
            "epilogue_relax": relax_info,
            "kept_hold_arms": sorted(getattr(self.motions, "keep_hold_arms", ()) or ()),
            "jams": _jsonable(list(getattr(self.motions, "jams", []) or [])),
            "witness_branch_accepted": _jsonable(list(getattr(self.motions, "witness_branch_accepted_events", []) or [])),
            "witness_via_posture": _jsonable(list(getattr(self.motions, "witness_via_posture_events", []) or [])),
            # 2026-10-31 train-fast (STRAIGHT_ARM): straight ramps taken in place of trajopt plans, and the fall-backs by cause
            "straight_arm": _jsonable(dict(getattr(self.motions, "straight_arm_stats", None) or {})),
            "protection": _jsonable({"protected": [getattr(o, "name", n) for n, o in self.protected_goal], "plan_checks": int(getattr(self.motions, "protect_checks", 0) or 0),
                                     "fired": list(getattr(self.motions, "protect_events", []) or []) + list(getattr(self.monitors, "protect_events", []) or [])}),
            "placed_orientation": placed_orientation,
            "motion_delta": motion,
            "actions": [r.to_dict() for r in self.action_records],
            # the oracle's retry rules (verdict counts, the collision rule's contacts) and the --qa metrics
            "monitors": self.monitors.report() if self.monitors is not None else {"stats": self.stats.to_dict()},
            "qa": qa_report,
            "checkpoint_reports": list(self.checkpoint_reports),
            "brace_retry": _jsonable(self.brace_retries),
            "counters": dict(self.counters),
            "goal_observation": goal_obs,
            "timings": dict(self.timings),
            "release_check": _jsonable(self.release_check),
        }
        return self.result

    # ------------------------------------------------------------------ goal image

    def capture_goal_image(self) -> Dict[str, Any]:
        """The rgb of the head camera after ``goal_image_renders`` renders with everything stationary, with the
        physics-synced sensor pose (``goal_observation.camera_pose_world``) and the pinhole intrinsics."""
        for _ in range(int(self.cfg.goal_image_renders)):
            self.sim.render()
        obs, _info = SE.get_obs(self.env)
        robs = obs.get(self.robot.name, obs) if isinstance(obs, dict) else obs
        rgb = to_uint8_rgb(robs[self.cam_key]["rgb"])
        sensor = self.robot.sensors[self.cam_key]
        cp, cq = sensor.get_position_orientation()
        try:
            intr = CameraModel.from_sensor(sensor).intrinsics()
        except Exception:  # noqa: BLE001
            intr = None
        return {
            "rgb": rgb,
            "sensor_key": self.cam_key,
            "resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
            "camera_pose_world": {"position": cp.float().tolist(), "orientation": cq.float().tolist()},
            "intrinsics": intr,
            "n_renders": int(self.cfg.goal_image_renders),
            "step": self.n,
        }

    def goal_occlusion_report(self) -> Dict[str, Any]:
        """The QA hard metric ``goal_occlusion`` (``ispatialgym.eval.goal_occlusion``) on the settled last frame: the head camera's
        depth, the box screen of every tracked target / partner (an inside goal's target measured, not judged: the sidecar's variant
        names the subtask) and the panel screen of a tracked fixture (a second depth render with the robot hidden; the panel = the
        links its revolute joints move, at the final joint state). Runs after the last recorded frame (``ispatialgym.oracle.head_depth``:
        the camera's observations stay as loaded) and leaves the robot visible."""
        from omnigibson.utils.constants import JointType

        from ispatialgym.eval import goal_occlusion as GO

        from ispatialgym.oracle.head_depth import settled_depth

        sensor = self.robot.sensors[self.cam_key]
        cm = CameraModel.from_sensor(sensor)
        intr = GO.intrinsics_of(cm)
        judged = [o for o in self.tracked.values() if GO.role(getattr(o, "name", ""))]
        depth_without = None
        depth = settled_depth(sensor, self.sim.render)
        cp, cq = sensor.get_position_orientation()
        if any(GO.role(o.name) == "fixture" for o in judged):
            self.robot.visible = False
            try:
                depth_without = settled_depth(sensor, self.sim.render)
            finally:
                self.robot.visible = True
                for _ in range(int(self.cfg.goal_image_renders)):
                    self.sim.render()
        cam = (cp.float().tolist(), cq.float().tolist())
        objects = GO.screen_scene(depth, cam, intr, judged, depth_without_robot=depth_without, revolute=JointType.JOINT_REVOLUTE,
                                  subtask=getattr(self.sidecar, "variant", None))
        return GO.report(objects, method="live head-camera depth", step=self.n, camera_key=self.cam_key, resolution=[intr[0], intr[1]])

    # ------------------------------------------------------------------ TRO goal override (§3.8, spec §8)

    def goal_override(self, tro_meta, compiled_task=None) -> Dict[str, Any]:
        """Settled values for every TRO target within its tolerance (the one tolerance per target of the ``isg-v2`` profile, which
        the loaded TRO carries in memory); a push-under target's orientation is judged modulo rotation about the world vertical
        (:func:`vertical_angle`, as the ``placed_orientation`` gate). Returns
        ``{"pose_targets": {inst: {...}}, "groups": {...}, "articulation_targets": {...}, "applied": [...]}``; the
        caller writes the TRO."""
        out: Dict[str, Any] = {"pose_targets": {}, "groups": {}, "articulation_targets": {}, "applied": [], "skipped": []}
        compiled = compiled_task if compiled_task is not None else getattr(self.task, "compiled_task", None)
        isg = self.isg
        for inst, tgt in tro_meta.pose_targets.items():
            try:
                rec = self._override_pose_target(inst, tgt, compiled)
            except Exception as e:  # noqa: BLE001
                out["skipped"].append({"target": inst, "reason": repr(e)})
                continue
            if rec is None:
                continue
            out["pose_targets"][inst] = rec
            if rec["applied"]:
                out["applied"].append(inst)
        assignments = isg.get("group_assignments") or {}
        for name, g in tro_meta.groups.items():
            assign = assignments.get(name) if isinstance(assignments, dict) else None
            if not assign or not g.target_slots:
                out["skipped"].append({"target": name, "reason": "no assignment / slots"})
                continue
            rec = self._override_group(name, g, assign)
            out["groups"][name] = rec
            if rec["applied"]:
                out["applied"].append(name)
        for inst, art in tro_meta.articulation_targets.items():
            try:
                obj = self.state.object(inst)
            except Exception as e:  # noqa: BLE001
                out["skipped"].append({"target": inst, "reason": repr(e)})
                continue
            names = list(obj.joints.keys())
            q = obj.get_joint_positions()
            joints = {}
            ok = True
            for jn, (pos, tol) in art.joints.items():
                if jn not in names:
                    ok = False
                    break
                val = float(q[names.index(jn)])
                joints[jn] = {"position": val, "was": pos, "error": abs(val - pos), "tolerance": tol}
                ok = ok and abs(val - pos) <= tol
            out["articulation_targets"][inst] = {"joints": joints, "applied": ok}
            if ok:
                out["applied"].append(inst)
        return out

    def _override_threshold(self, inst: str, tgt, compiled) -> Tuple[float, float]:
        return float(tgt.position_tolerance or self.cfg.override_min_pos), float(tgt.orientation_tolerance or self.cfg.override_rot)

    def _measured_pose(self, inst: str, tgt) -> Tuple[th.Tensor, th.Tensor]:
        if inst == AGENT_INSTANCE:
            if tgt.link and tgt.link in getattr(self.robot, "sensors", {}):
                p, q = self.robot.sensors[tgt.link].get_position_orientation()
            elif tgt.link and tgt.link in getattr(self.robot, "links", {}):
                p, q = self.robot.links[tgt.link].get_position_orientation()
            else:
                p, q = self.robot.get_position_orientation()
            return p.float(), q.float()
        obj = self.state.object(inst)
        if tgt.link and tgt.link in getattr(obj, "links", {}):
            p, q = obj.links[tgt.link].get_position_orientation()
        else:
            p, q = obj.get_position_orientation()
        return p.float(), q.float()

    def _override_pose_target(self, inst: str, tgt, compiled) -> Optional[Dict[str, Any]]:
        if tgt.frame != "world":
            return {"applied": False, "reason": f"frame {tgt.frame!r} is not world (depth targets are not overridden)"}
        p, q = self._measured_pose(inst, tgt)
        pos_tol, rot_tol = self._override_threshold(inst, tgt, compiled)
        rec: Dict[str, Any] = {"applied": False, "position_tolerance": pos_tol, "orientation_tolerance": rot_tol}
        ok = True
        if tgt.position_metric == "full" and tgt.position is not None:
            dp = float(th.linalg.norm(p - tgt.position.float()))
            rec.update({"position": p.tolist(), "position_was": tgt.position.tolist(), "position_error": dp})
            ok = ok and dp <= pos_tol
        if tgt.orientation_metric == "full" and tgt.orientation is not None:
            cands = [tgt.orientation.float()] + [C.qmul(tgt.orientation.float(), C.qinv(e.float())) for e in (tgt.orientation_equivalences or [])]
            da = min(C.quat_angle(q, c) for c in cands)
            rec.update({"orientation": q.tolist(), "orientation_was": tgt.orientation.tolist(), "orientation_error": da})
            if inst in self.pushed_under_objects():  # judged as the placed_orientation gate judges it: the yaw is free
                da = vertical_angle(q, tgt.orientation, tgt.orientation_equivalences or ())
                rec.update({"orientation_error_full": rec["orientation_error"], "orientation_error": da, "equivalence": "rotation about the world vertical (push-under)"})
            ok = ok and da <= rot_tol
        rec["applied"] = ok
        return rec

    def _override_group(self, name: str, g, assign: Dict[str, str]) -> Dict[str, Any]:
        pos_tol, rot_tol = g.pose_tolerances or (self.cfg.override_min_pos, self.cfg.override_rot)
        slots: Dict[str, Dict[str, Any]] = {}
        ok = True
        for member, slot_id in assign.items():
            slot = g.target_slots.get(slot_id)
            if slot is None:
                ok = False
                continue
            p, q = self.state.object_pose(member)
            dp = float(th.linalg.norm(p - slot.position.float()))
            cands = [slot.orientation.float()] + [C.qmul(slot.orientation.float(), C.qinv(e.float())) for e in (g.orientation_equivalences or [])]
            da = min(C.quat_angle(q, c) for c in cands)
            slots[slot_id] = {"member": member, "position": p.tolist(), "orientation": q.tolist(), "position_error": dp, "orientation_error": da}
            ok = ok and dp <= pos_tol and da <= rot_tol
        relations = []
        if ok and g.layout_relations:
            for rel in g.layout_relations:
                a, b = slots.get(rel.from_slot), slots.get(rel.to_slot)
                if a is None or b is None:
                    ok = False
                    break
                rp, rq = C.relative((th.tensor(b["position"]), th.tensor(b["orientation"])), (th.tensor(a["position"]), th.tensor(a["orientation"])))
                relations.append({"from": rel.from_slot, "to": rel.to_slot, "position": rp.tolist(), "orientation": rq.tolist()})
        return {"applied": ok, "target_slots": slots, "layout_relations": relations, "position_tolerance": pos_tol, "orientation_tolerance": rot_tol}


# ==========================================================================================
# helpers
# ==========================================================================================


class _StreamAction:
    """Stand-in action for the replay / policy streams (QA monitors need a name and arms)."""

    name = "stream"
    arms: Tuple[str, ...] = ()
    objects: Tuple[str, ...] = ()


def _hold_actions(robot, steps: int) -> Iterator[th.Tensor]:
    """Zero-velocity hold when no Motions instance exists (replay / policy modes): current joint targets."""
    from ispatialgym.oracle.action import ActionState

    st = ActionState(robot)
    st.hold_current()
    for _ in range(int(steps)):
        yield st.action()


def vertical_angle(q_cur, goal_q, equivalences: Sequence[Any] = ()) -> float:
    """The angle (rad) between the world vertical and the object-frame axis that is vertical at ``goal_q``, at the orientation ``q_cur``
    (minimised over the object-frame ``equivalences``): the orientation error modulo rotation about the world vertical, the tilt off the goal."""
    up = th.tensor([0.0, 0.0, 1.0])
    a = C.quat_rotate(C.qinv(C.as_tensor(goal_q).float()), up)
    best = math.pi
    for e in [None] + list(equivalences or []):
        q = C.as_tensor(q_cur).float() if e is None else C.qmul(C.as_tensor(q_cur).float(), C.as_tensor(e).float())
        best = min(best, math.acos(max(-1.0, min(1.0, float(th.dot(C.quat_rotate(q, a), up))))))
    return float(best)


def _unsatisfied(isg: Dict[str, Any]) -> List[str]:
    return [c.get("bddl", "?") for c in (isg.get("conditions") or []) if c.get("kind") == "binary" and not c.get("satisfied")]


def apply_goal_override(tro_raw: Dict[str, Any], override: Dict[str, Any], tro_key: str = "ispatialgym") -> List[str]:
    """Write the ``applied`` values of :meth:`Executor.goal_override` into a loaded TRO json dict; returns the keys
    changed."""
    meta = tro_raw[tro_key]
    changed = []
    for inst, rec in override.get("pose_targets", {}).items():
        if not rec.get("applied"):
            continue
        tgt = meta["pose_targets"][inst]
        if "position" in rec:
            tgt["position"] = [float(v) for v in rec["position"]]
        if "orientation" in rec:
            tgt["orientation"] = [float(v) for v in rec["orientation"]]
        changed.append(f"pose_targets.{inst}")
    for name, rec in override.get("groups", {}).items():
        if not rec.get("applied"):
            continue
        g = meta["groups"][name]
        for slot_id, s in rec["target_slots"].items():
            g["target_slots"][slot_id]["position"] = [float(v) for v in s["position"]]
            g["target_slots"][slot_id]["orientation"] = [float(v) for v in s["orientation"]]
        if rec.get("layout_relations"):
            g["layout_relations"] = rec["layout_relations"]
        changed.append(f"groups.{name}")
    for inst, rec in override.get("articulation_targets", {}).items():
        if not rec.get("applied"):
            continue
        joints = meta["articulation_targets"][inst]["joints"]
        for jn, j in rec["joints"].items():
            joints[jn]["position"] = float(j["position"])
        changed.append(f"articulation_targets.{inst}")
    return changed


__all__ = [
    "ExecutorConfig", "Checkpoint", "ActionRecord", "SimState", "MotionSnapshot", "Executor", "apply_goal_override", "AGENT_INSTANCE", "vertical_angle",
]
