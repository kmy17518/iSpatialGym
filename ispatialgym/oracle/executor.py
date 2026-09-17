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
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym import ASSETS_ROOT
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.actions import Action, Brace, Place, Push
from ispatialgym.oracle.execute import check_postcondition, forbidden_contacts, fov_monitored, intended_contacts, is_manipulation, offset_brace, run_action
from ispatialgym.oracle.motions import Motions, SkillError
from ispatialgym.oracle.qa import CameraModel, Monitors, QAStats, Verdict, category_of
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


@dataclass
class ExecutorConfig:
    n_render_iterations: int = 1
    #: episode step budget (the QA monitors' own budget is ``QAConfig.max_episode_steps``)
    max_steps: int = 20000
    #: one retry from the checkpoint (§3.8)
    retries: int = 1
    #: epilogue no-op window: max(settle_min_steps, settle_fraction * action steps)
    settle_min_steps: int = 50
    settle_fraction: float = 0.2
    #: second chance: when the first window shows motion (or the binary goal is lost), a second no-op window of this
    #: many times the first runs and *its* motion / end state decide (0 = no second window)
    settle_second_window_factor: float = 2.0
    #: motion over the no-op window that counts as "not settled" (robot base / joints / task objects)
    settled_pos_tol: float = 0.005
    settled_rot_tol: float = 0.01
    settled_joint_tol: float = 0.02
    #: also require the evaluator's own ``settled_success`` (PoseSettled of every settled entity) at the last step.
    #: Off by default: PoseSettled's 1 mm/s root-velocity threshold is never met by an object held at the end of a
    #: position-controlled arm (0 / 74 v1 held-object demos ever settled by that measure while being motionless
    #: within 0.1 mm over the window), so the §6.4 "motion after the final no-op window" is measured here instead.
    require_task_settled: bool = False
    #: goal image
    goal_image: bool = False
    goal_image_renders: int = 10
    #: brace retry ladder (owner 2026-09-16 05:26 / 05:39): retries of a braced push whose residual is out of tolerance,
    #: with the brace moved by these offsets (m) outward (short) / inward (overshoot) along its recorded direction
    brace_retries: int = 2
    brace_retry_offsets_m: Tuple[float, ...] = (0.005, 0.010)
    #: demo-collection gates applied after the epilogue (``run_isg --demo-gating``; oracle mode: ``placed_orientation``):
    #: ``placed_orientation`` = every placed object (place / drop / attach, not pushes) must end within its recorded
    #: orientation target's tolerance (``orientation_tolerance``, 0.2 rad default; nearest equivalence, the evaluator's
    #: own error) — the TRO target keeps ``gating: false`` for the policy evaluation, position stays non-gating.
    #: Cylinders (owner decision 2026-09-16 17:10 item 4, fallback applied after the closure investigation): an object
    #: with a continuous symmetry about its long axis per the ICL symmetry survey (``assets/icl/revolution_axes.json``:
    #: marker, pencil, bottles, cans ...) is compared **modulo rotation about that axis** — the angle between its long
    #: axis now and at the recorded goal (nearest equivalence) against the same tolerance; every other object keeps the
    #: full error. ``demo_revolution_axes`` switches that equivalence off (full error for everything).
    demo_gating: Tuple[str, ...] = ()
    demo_orientation_tolerance_default: float = 0.2
    demo_revolution_axes: bool = True
    #: ``placed_tilt`` (retracted as the default, owner 2026-09-16 17:10 item 4; the code path stays unused): the tilt of
    #: a placed object's principal axis against the recorded goal's (rad)
    demo_tilt_tolerance: float = math.radians(15.0)
    #: renders after a checkpoint restore before the (recorded) hold step, so no frame of the rolled-back attempt
    #: survives in the video / dataset (the renderer trails ``load_state`` by 1-2 frames)
    restore_renders: int = 3
    #: TRO goal override thresholds for measured-only targets (spec §8): max(min_pos, extent_fraction × smallest
    #: horizontal extent) / rot
    override_min_pos: float = 0.01
    override_extent_fraction: float = 0.25
    override_rot: float = 0.20
    #: record the serialized sim state every step (needed for the BEHAVIOR-replayable hdf5)
    record_state: bool = True


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
    #: the gripper-in-FOV spans open at the checkpoint (rule (b): a span runs from the pre-grasp to the release across
    #: actions; a retry re-runs the action with the spans as they were)
    fov_spans: Dict[str, bool] = field(default_factory=dict)


#: ``ISpatialGymTask`` success-tracking attributes that a rollback must restore (sticky "first success" steps)
TASK_TRACKING_ATTRS = ("_first_success_step", "_first_settled_success_step", "_errors_at_first_success", "_last_goal_eval")


#: ctx keys the execute functions write for the record (meta.json ``actions[].notes``)
NOTE_KEYS = ("push_residual", "push_pivot_offset_m", "push_object_drift_m", "approach_finish", "descent_finish", "lift_finish", "retreat_finish", "pre_pose_standoff", "plan_relaxed_links", "push_axis", "carry_leg", "idle_arms_moved", "arms_posture", "fingers_set", "plan_ignored", "object_motion",
             "camera_error_before_correction", "camera_error", "attached", "preshape_opening_m", "closure_rate_m_per_step", "release_stage_m")


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
        self.robot = env.robots[0]
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
        self.stats: QAStats = monitors.stats if monitors is not None else QAStats()
        self.action_records: List[ActionRecord] = []
        self.last_ctx: Dict[str, Any] = {}
        self._defer_commit = False
        self.brace_retries: List[Dict[str, Any]] = []
        self.program_actions: List[Action] = []
        self.checkpoint_reports: List[Dict[str, Any]] = []
        self.timings: Dict[str, float] = {}
        self.result: Dict[str, Any] = {}
        self.verdict: Optional[Verdict] = None

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
        if registry is not None and hasattr(registry, "set_dump_filter"):
            registry.set_dump_filter(self._active_filter if enabled else (lambda obj: True))

    @staticmethod
    def _active_filter(obj) -> bool:
        return bool(getattr(obj, "is_active", True) and getattr(obj, "initialized", True))

    def _dump_state(self, full: bool) -> th.Tensor:
        registry = getattr(getattr(self.env, "scene", None), "object_registry", None)
        if full and getattr(self, "_recording_filter", False) and registry is not None and hasattr(registry, "set_dump_filter"):
            registry.set_dump_filter(lambda obj: True)
            try:
                return self.sim.dump_state(serialized=True)
            finally:
                registry.set_dump_filter(self._active_filter)
        return self.sim.dump_state(serialized=True)

    @property
    def isg(self) -> Dict[str, Any]:
        info = self.last_info or {}
        return (info.get("done") or {}).get("ispatialgym") or info.get("ispatialgym") or {}

    # ------------------------------------------------------------------ stepping (the only env.step)

    def step(self, action: th.Tensor, phase: int = PHASE_ORACLE, action_index: int = -1, attempt: int = 0, monitor: bool = True) -> Optional[Verdict]:
        obs, reward, terminated, truncated, info = self.env.step(action, n_render_iterations=self.cfg.n_render_iterations)
        self.n += 1
        self.phase_steps[phase] = self.phase_steps.get(phase, 0) + 1
        self.last_obs, self.last_info = obs, info
        if self.recorder is not None:
            state = self._dump_state(full=False) if (self.cfg.record_state and self.recorder.cfg.hdf5) else None
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
        verdict = self.monitors.check(self.n) if (monitor and self.monitors is not None) else None
        if verdict is None and self.n >= self.cfg.max_steps:
            verdict = Verdict("timeout", f"episode exceeded max_steps {self.cfg.max_steps}", step=self.n)
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
        )
        return cp

    def restore(self, cp: Checkpoint, action_index: int = -1, attempt: int = 0) -> Dict[str, Any]:
        """Roll the simulator and the executor back to ``cp``; returns the round-trip report."""
        robot = self.robot
        report: Dict[str, Any] = {"action_index": cp.action_index, "ag_roundtrip_ok": True, "ag_fixed": [], "attached_fixed": []}
        self.sim.load_state(cp.sim_state, serialized=True)
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
        self.checkpoint_reports.append(report)
        # -- the renderer trails the physics: the first frame after ``load_state`` still shows the pre-restore scene
        #    (a sensor read lags the last render by 1-2 frames plus the USD sync, B10), so the failed attempt's last
        #    arm pose leaked into the committed video as the restore step's frame. Render until the buffers show the
        #    restored scene before anything is recorded again.
        for _ in range(int(self.cfg.restore_renders)):
            self.sim.render()
        # -- the PhysX drive targets are not state: apply the restored hold targets with one (recorded) hold step;
        #    the monitors are off for it (their per-action whitelist is not set up yet)
        if self.motions is not None:
            self.run_generator(self.motions.hold(1), PHASE_ORACLE, action_index, attempt, monitor=False)
        return report

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

    def release_stage_opening(self, action: Place) -> Optional[float]:
        """The first-stage opening (m per finger) of the staged release of ``action.object`` (owner 2026-09-16 21:18):
        the library pinch width / 2 + ``MotionsConfig.gripper_preshape_margin`` (the pre-shape width, ~1 cm past the
        pinch), capped by the sampler's ``grasps[k].release_sweep.release_opening_max_m`` / 2 (core ``385de1e``: the
        largest symmetric opening that still clears the partner by 1 cm) and never below pinch / 2 + 5 mm. None when
        the sidecar has no library grasp with a pinch width for the object (a cube / handle grasp): ``Motions`` then
        falls back to the remembered pre-shape, else opens fully."""
        g = next((g for g in (getattr(self.sidecar, "grasps", None) or []) if g.object == action.object), None)
        if g is None:
            return None
        try:
            width = (C.grasp_entry(g.grasp_key, g.grasp_name).meta or {}).get("width")
            width = float(width)
        except Exception:  # noqa: BLE001
            return None
        if not math.isfinite(width) or width <= 0:
            return None
        margin = getattr(self.motions.cfg, "gripper_preshape_margin", None) if self.motions is not None else None
        stage = 0.5 * width + float(0.010 if margin is None else margin)
        sweep = (getattr(g, "extra", None) or {}).get("release_sweep")
        cap = sweep.get("release_opening_max_m") if isinstance(sweep, dict) else None
        if isinstance(cap, (int, float)) and math.isfinite(float(cap)) and float(cap) > 0:
            stage = min(stage, 0.5 * float(cap))
        return max(stage, 0.5 * width + 0.005)

    def _scene_names(self, instances: Iterable[str]) -> set:
        out = set()
        for inst in instances:
            try:
                out.add(self.state.object(inst).name)
            except Exception:  # noqa: BLE001
                out.add(inst)
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
        if isinstance(action, Place) and self.motions is not None and hasattr(self.motions, "release_opening"):
            # the staged release (Motions.gripper): first stage = pinch / 2 + margin, capped by the sampler's release sweep
            stage = self.release_stage_opening(action)
            if stage is not None:
                self.motions.release_opening[action.arm] = stage
            ctx["release_stage_m"] = stage
        if self.monitors is not None:
            ctx["hooks"] = {"fov": self.monitors.set_fov_enabled}
            self.monitors.begin_action(action, index, attempt, intended=self._scene_names(intended_contacts(action)), held=self.state.held_objects(),
                                       manipulation=is_manipulation(action), fov=fov_monitored(action), forbidden=self._scene_names(forbidden_contacts(action)),
                                       watch=[(links, next(iter(self._scene_names([obj])))) for links, obj in watch])
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
            left = getattr(self.motions, "last_witness_residual", None) if self.motions is not None else None
            if left is not None:  # an arm_to left a small witness residual unclosed (a witness on a joint limit)
                rec.notes["witness_residual_left"] = {"pos_m": round(float(left[0]), 4), "rot_rad": round(float(left[1]), 4)}
                self.motions.last_witness_residual = None
            events = list(getattr(self.motions, "release_events", None) or []) if self.motions is not None else []
            if events:  # staged release: the first stage of this release and / or the deferred full opening a retreat completed
                rec.notes["release"] = events
                self.motions.release_events.clear()
            rec.attempts.append(None if verdict is None else verdict.to_dict())
            if verdict is None:
                self._commit()
                log.info("action %d %s ok in %d steps (%.1fs)", index, action.summary() if hasattr(action, "summary") else action.name, self.n - rec.step_start, time.time() - t0)
                break
            log.warning("action %d %s attempt %d failed: %s", index, action.name, k, verdict)
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

    def run_program(self, actions: Sequence[Action]) -> Dict[str, Any]:
        t0 = time.time()
        actions = list(actions)
        self.program_actions = actions
        verdict: Optional[Verdict] = None
        i = 0
        while i < len(actions):
            action = actions[i]
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

    def push_outcome(self, push: Push, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """How the push ended against the task goal: ``residual`` (signed in the push sense: > 0 short, < 0
        overshoot), ``tolerance`` (the TRO target's, None when the object has no gating target of the push's kind),
        ``in_tolerance``, ``brace_hits`` (steps the watched brace fingers touched the target before the goal first
        held) and the ladder ``reason``: ``"short"`` (out of tolerance, short, the target touched the stop before the
        goal), ``"overshoot"`` (out of tolerance, the residual flipped sign) or None."""
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
        first = int(self.isg.get("first_success_step", -1) or -1)
        hits = [h for h in (ctx.get("watch_hits") or []) if first < 0 or h < first]
        out["brace_hits"] = len(hits)
        if out["in_tolerance"]:
            return out
        if signed < 0:
            out["reason"] = "overshoot"
        elif hits:
            out["reason"] = "short"
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
        offset = 0.0
        verdict: Optional[Verdict] = None
        self._defer_commit = True
        try:
            for k in range(self.cfg.brace_retries + 1):
                brace_k = brace if offset == 0.0 else offset_brace(brace, offset)
                verdict = self.run_action(index, brace_k)
                if verdict is not None:
                    break
                watch = [(list(M.FINGER_LINKS[brace.contact.arm]) + [M.GRIPPER_LINK[brace.contact.arm]], push.contact.object)]
                verdict = self.run_action(index + 1, push, watch=watch)
                if verdict is not None:
                    break
                outcome = self.push_outcome(push, self.last_ctx)
                self.action_records[-1].notes["push_outcome"] = outcome
                if outcome["reason"] is None or k == self.cfg.brace_retries:
                    if outcome["reason"] is not None:
                        log.warning("braced push: %s residual %.4f > tolerance %.4f after %d brace retries; the epilogue decides", outcome["reason"], outcome["residual"], outcome["tolerance"], k)
                    break
                new_offset = float(ladder[outcome["reason"]][k])
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
        """Replay / policy modes: step a stream of 23-D actions (no checkpoints, no program), then the epilogue. The
        epilogue's no-op window repeats the **last streamed action** with the base stopped (the Layer-1 hold targets
        are not updated by a stream; holding them would drive the robot back to its initial configuration)."""
        t0 = time.time()
        verdict: Optional[Verdict] = None
        if self.monitors is not None:
            self.monitors.begin_action(_StreamAction(), 0, 0, intended=(), held=self.state.held_objects(), manipulation=False)
        try:
            for a in actions:
                a = th.as_tensor(a, dtype=th.float32).reshape(-1)
                self._last_stream_action = a.clone()
                verdict = self.step(a, phase, 0, 0)
                if verdict is not None:
                    break
        finally:
            if self.monitors is not None:
                self.monitors.end_action()
        if self.recorder is not None:
            self.recorder.commit()
        self.timings["actions_s"] = time.time() - t0
        return self.epilogue(verdict, self.n)

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
        every other object keeps the full error."""
        errs = (isg.get("errors") or {}).get("pose_errors") or {}
        out: Dict[str, Dict[str, Any]] = {}
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
                                     "target_gating": bool(rep.get("gating", False))}
            axis = self.revolution_axis_of(obj) if self.cfg.demo_revolution_axes else None
            if axis is not None and str(getattr(tgt, "frame", "world")) == "world":
                ax_err = self.axis_error(obj, tgt, axis["axis"])
                if ax_err is not None:
                    entry.update({"orientation_error": ax_err, "within": ax_err <= tol, "full_error": float(err),
                                  "equivalence": f"rotation about the {axis['axis']} axis (revolution, survey tags {axis.get('tags')})",
                                  "revolution_axis": axis["axis"], "model": axis.get("key")})
            out[obj] = entry
        return out

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

    @staticmethod
    def principal_axis(extent) -> Tuple[th.Tensor, str]:
        """The object-frame axis whose tilt says whether a placed object tipped: the **long** axis of an elongated
        object (one native extent >= 2x both others: a marker, a pencil, a bottle - lying or standing, it may roll
        about that axis freely), the **thin** axis (the face normal) of a flat one (one extent <= half of both
        others: a plate, a book, a board - it may spin on its face), else the object's **up** axis (+z: a cup, a
        bowl, an apple - it may yaw). Returns ``(unit axis in the object frame, rule)``."""
        e = [float(v) for v in C.as_tensor(extent).reshape(-1)[:3].tolist()]
        order = sorted(range(3), key=lambda k: e[k])
        lo, mid, hi = e[order[0]], e[order[1]], e[order[2]]
        axis = th.zeros(3)
        if hi >= 2.0 * mid and hi > 0:
            axis[order[2]] = 1.0
            return axis, "long"
        if lo <= 0.5 * mid and mid > 0:
            axis[order[0]] = 1.0
            return axis, "thin"
        axis[2] = 1.0
        return axis, "up"

    def placed_tilt_report(self) -> Dict[str, Dict[str, Any]]:
        """Per placed object: the angle between its principal axis now and at the recorded goal orientation
        (:meth:`principal_axis` from the native bounding box: the long axis of a marker, the face normal of a plate,
        +z otherwise), minimised over the target's orientation equivalences; ``within`` against
        ``demo_tilt_tolerance`` (None without an orientation target). A roll of a lying marker about its own axis,
        the yaw of a cup, the spin of a plate on its face do not tilt that axis; a fallen cup or a marker standing on
        its end do. The sampler's ``goal_tipped`` guard in oracle terms."""
        out: Dict[str, Dict[str, Any]] = {}
        for obj in self.placed_objects():
            tgt = self.state.pose_target(obj) if hasattr(self.state, "pose_target") else None
            goal_q = getattr(tgt, "orientation", None) if tgt is not None else None
            if goal_q is None or str(getattr(tgt, "frame", "world")) != "world":
                out[obj] = {"tilt_rad": None, "tolerance_rad": None, "within": None, "note": "no world orientation target"}
                continue
            try:
                _, q_cur = self.state.object_pose(obj)
            except Exception:  # noqa: BLE001
                out[obj] = {"tilt_rad": None, "tolerance_rad": None, "within": None, "note": "no object pose"}
                continue
            extent = None
            try:
                o = self.state.object(obj)
                extent = getattr(o, "native_bbox", None)
                if extent is None:
                    extent = getattr(o, "aabb_extent", None)
            except Exception:  # noqa: BLE001
                extent = None
            axis, rule = self.principal_axis(extent) if extent is not None else (th.tensor([0.0, 0.0, 1.0]), "up")
            goal_q = C.as_tensor(goal_q).float()
            a_goal = C.quat_rotate(goal_q, axis)
            best = None
            for e in [None] + list(getattr(tgt, "orientation_equivalences", None) or []):
                q = q_cur if e is None else C.qmul(q_cur, C.as_tensor(e).float())
                a_now = C.quat_rotate(q, axis)
                ang = math.acos(max(-1.0, min(1.0, float(th.dot(a_now, a_goal)))))
                best = ang if best is None or ang < best else best
            tol = float(self.cfg.demo_tilt_tolerance)
            out[obj] = {"tilt_rad": float(best), "tilt_deg": math.degrees(best), "tolerance_rad": tol, "tolerance_deg": math.degrees(tol), "within": float(best) <= tol,
                        "axis": "xyz"[int(th.argmax(axis))], "rule": rule, "extent": None if extent is None else [round(float(v), 4) for v in C.as_tensor(extent).reshape(-1)[:3].tolist()]}
        return out

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

    def _moved(self, motion: Dict[str, float], isg: Dict[str, Any]) -> bool:
        """The epilogue's motion-window rule: the robot (joints, base) must be still; a tracked object's **position**
        must be still; its **rotation** only counts when the goal has an orientation target for it (owner rule
        2026-09-16: a near-spherical brussels sprout spinning in the bowl — 0.245 rad at 0.7 mm — is not a failed
        IORS goal; the relation and the position decide)."""
        if motion.get("joint_rad", 0.0) > self.cfg.settled_joint_tol or motion.get("base_m", 0.0) > self.cfg.settled_pos_tol or motion.get("base_rad", 0.0) > self.cfg.settled_rot_tol:
            return True
        oriented = self.orientation_targets(isg)
        for k, v in motion.items():
            if k in ("joint_rad", "base_m", "base_rad") or not isinstance(v, (int, float)):
                continue
            if k.endswith("_m"):
                if v > self.cfg.settled_pos_tol:
                    return True
            elif k.endswith("_rad"):
                if k[: -len("_rad")] in oriented and v > self.cfg.settled_rot_tol:
                    return True
        return False

    def epilogue(self, verdict: Optional[Verdict], n_action_steps: int) -> Dict[str, Any]:
        t0 = time.time()
        if self.monitors is not None:
            self.monitors.begin_epilogue(held=self.state.held_objects())
        window = max(int(self.cfg.settle_min_steps), int(math.ceil(self.cfg.settle_fraction * n_action_steps)))
        before = MotionSnapshot(self.robot, self.tracked)
        settle_verdict = None
        windows = [window]
        if verdict is None or verdict.category != "timeout":
            settle_verdict = self.run_generator(self._noop_actions(window), PHASE_SETTLE)
            if self.recorder is not None:
                self.recorder.commit()
        after = MotionSnapshot(self.robot, self.tracked)
        motion = before.delta(after)
        isg = self.isg
        moved = self._moved(motion, isg)
        binary_final = bool(isg.get("binary_success", False))
        if verdict is None and settle_verdict is None and (moved or not binary_final) and self.cfg.settle_second_window_factor > 0:
            # second chance (owner rule 2026-09-16): a longer no-op window before the verdict; the motion over *this*
            # window and the binary goal at its end decide (a near-spherical object still rolling in a bowl, a
            # released object still relaxing)
            window2 = int(math.ceil(self.cfg.settle_second_window_factor * window))
            log.info("epilogue: motion after the first no-op window (%s) - second window of %d steps", {k: round(v, 4) for k, v in motion.items() if (v > 1e-4)}, window2)
            before2 = MotionSnapshot(self.robot, self.tracked)
            settle_verdict = self.run_generator(self._noop_actions(window2), PHASE_SETTLE)
            if self.recorder is not None:
                self.recorder.commit()
            motion_first = motion
            motion = before2.delta(MotionSnapshot(self.robot, self.tracked))
            motion["first_window"] = motion_first
            isg = self.isg
            moved = self._moved(motion, isg)
            binary_final = bool(isg.get("binary_success", False))
            windows.append(window2)
        binary_ever = bool(isg.get("binary_success_ever", isg.get("binary_success", False)))
        settled_task = bool(isg.get("settled_success", False))
        out_of_tol = _out_of_tolerance(isg)
        if verdict is None and settle_verdict is not None:
            verdict = settle_verdict
        if verdict is None and not (binary_ever or binary_final):
            unsatisfied = _unsatisfied(isg)
            # a goal that never held only because gating pose targets are outside their tolerance is a tolerance
            # failure (the brace retry ladder ends there), not a relation that was never established
            only_tolerance = bool(out_of_tol) and all(_pose_reached_of(c) in out_of_tol for c in unsatisfied)
            if only_tolerance:
                verdict = Verdict("tolerance", "measured error above tolerance for " + ", ".join(out_of_tol) + " (the goal never held)", step=self.n,
                                  details={"targets": out_of_tol, "never_reached": True})
            else:
                verdict = Verdict("not_reached", "binary goal not reached; unsatisfied: " + ", ".join(unsatisfied), step=self.n)
            self.stats.record(verdict)
        if verdict is None and out_of_tol:
            verdict = Verdict("tolerance", "measured error above tolerance for " + ", ".join(out_of_tol), step=self.n, details={"targets": out_of_tol})
            self.stats.record(verdict)
        if verdict is None and (moved or not binary_final or (self.cfg.require_task_settled and not settled_task)):
            if moved:
                msg = "robot / object motion after the final no-op window"
            elif not binary_final:
                msg = "the binary goal no longer holds at the last step"
            else:
                msg = "the evaluator's pose_settled atoms do not hold at the last step"
            verdict = Verdict("not_settled", msg, step=self.n, details={"motion": motion, "task_settled_success": settled_task})
            self.stats.record(verdict)
        # -- demo-collection gates (owner decision 2026-09-16 10:15): the placed objects' orientation against their
        #    recorded (non-gating) targets decides the *demo* verdict; the TRO / policy evaluation is untouched
        demo_gating: Dict[str, Any] = {"enabled": list(self.cfg.demo_gating)}
        if self.program_actions and ("placed_orientation" in self.cfg.demo_gating or "placed_tilt" in self.cfg.demo_gating):
            # both reports are always written when a gate is on, so the strict and the tilt reading can be compared
            demo_gating["placed_orientation"] = self.placed_orientation_report(isg)
            demo_gating["placed_tilt"] = self.placed_tilt_report()
        if "placed_tilt" in self.cfg.demo_gating:
            rep = demo_gating["placed_tilt"]
            failed = [o for o, r in rep.items() if r.get("within") is False]
            if verdict is None and failed:
                msg = ", ".join(f"{o} tilted {rep[o]['tilt_deg']:.1f} deg > {rep[o]['tolerance_deg']:.0f}" for o in failed)
                verdict = Verdict("placed_tilt", "placed object tipped off its recorded up axis: " + msg, step=self.n, details={"targets": failed, "errors": {o: rep[o] for o in failed}})
                self.stats.record(verdict)
        if "placed_orientation" in self.cfg.demo_gating:
            rep = demo_gating["placed_orientation"]
            failed = [o for o, r in rep.items() if r.get("within") is False]
            if verdict is None and failed:
                msg = ", ".join(f"{o} {rep[o]['orientation_error']:.3f} rad > {rep[o]['tolerance']:.2f}" for o in failed)
                verdict = Verdict("placed_orientation", "placed object orientation outside its recorded target tolerance: " + msg, step=self.n,
                                  details={"targets": failed, "errors": {o: rep[o] for o in failed}})
                self.stats.record(verdict)
        self.timings["settle_s"] = time.time() - t0
        goal_obs = None
        if verdict is None and self.cfg.goal_image:
            t1 = time.time()
            try:
                goal_obs = self.capture_goal_image()
            except Exception as e:  # noqa: BLE001
                log.warning("goal image not captured: %r", e)
                goal_obs = {"error": repr(e)}
            self.timings["goal_image_s"] = time.time() - t1
        self.verdict = verdict
        success = verdict is None
        errors = isg.get("final_errors") or isg.get("errors") or {}
        last_errors = isg.get("errors") or {}
        self.result = {
            "success": success,
            "verdict": None if verdict is None else verdict.to_dict(),
            "reason": None if verdict is None else verdict.category,
            "binary_success_ever": binary_ever,
            "binary_success_final": binary_final,
            "settled_success": settled_task,
            "settled_success_ever": int(isg.get("first_settled_success_step", -1)) >= 0,
            "first_success_step": isg.get("first_success_step"),
            "first_settled_success_step": isg.get("first_settled_success_step"),
            "settle_latency_steps": isg.get("settle_latency_steps"),
            "errors": errors,
            "errors_final_step": last_errors,
            "conditions": isg.get("conditions"),
            "measured": isg.get("measured"),
            "group_assignments": isg.get("group_assignments"),
            "n_steps": self.n,
            "n_action_steps": n_action_steps,
            "n_settle_steps": self.phase_steps.get(PHASE_SETTLE, 0),
            "settle_window": window,
            "settle_windows": windows,
            "demo_gating": demo_gating,
            "motion_delta": motion,
            "actions": [r.to_dict() for r in self.action_records],
            "qa": self.monitors.report() if self.monitors is not None else {"stats": self.stats.to_dict()},
            "checkpoint_reports": list(self.checkpoint_reports),
            "brace_retry": _jsonable(self.brace_retries),
            "goal_observation": goal_obs,
            "timings": dict(self.timings),
        }
        return self.result

    # ------------------------------------------------------------------ goal image

    def capture_goal_image(self) -> Dict[str, Any]:
        """The rgb of the head camera after ``goal_image_renders`` renders with everything stationary, with the
        physics-synced sensor pose (``goal_observation.camera_pose_world``) and the pinhole intrinsics."""
        for _ in range(int(self.cfg.goal_image_renders)):
            self.sim.render()
        obs, _info = self.env.get_obs()
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

    # ------------------------------------------------------------------ TRO goal override (§3.8, spec §8)

    def goal_override(self, tro_meta, compiled_task=None) -> Dict[str, Any]:
        """Settled values for every TRO target within its override threshold: gating targets use their own tolerance,
        measured-only targets spec §8's max(0.01 m, 0.25 × smallest horizontal extent) / 0.20 rad. Returns
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
        gating = bool(compiled.is_gating("pose_reached", inst)) if compiled is not None else True
        if gating:
            return float(tgt.position_tolerance or self.cfg.override_min_pos), float(tgt.orientation_tolerance or self.cfg.override_rot)
        extent = None
        try:
            obj = self.state.object(inst)
            ext = obj.aabb_extent
            extent = float(min(float(ext[0]), float(ext[1])))
        except Exception:  # noqa: BLE001
            pass
        pos = max(self.cfg.override_min_pos, self.cfg.override_extent_fraction * extent) if extent is not None else self.cfg.override_min_pos
        return pos, self.cfg.override_rot

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


def _unsatisfied(isg: Dict[str, Any]) -> List[str]:
    return [c.get("bddl", "?") for c in (isg.get("conditions") or []) if c.get("kind") == "binary" and not c.get("satisfied")]


def _pose_reached_of(bddl: str) -> Optional[str]:
    """The target instance of a ``(pose_reached ?x)`` atom (``?`` and parentheses stripped), None for any other atom."""
    s = str(bddl).strip().strip("()").split()
    if len(s) == 2 and s[0] == "pose_reached":
        return s[1].lstrip("?")
    return None


def _out_of_tolerance(isg: Dict[str, Any]) -> List[str]:
    """**Gating** targets whose current error report says ``within_tolerance: False`` (pose and layout targets). The
    measured-only ``(:measure ...)`` targets of disembedding / IORS (``gating: False``) are reported, never a verdict:
    their tolerance is the reporting / goal-override band of spec §8, the relation predicate is what gates."""
    errs = isg.get("errors") or {}
    out = []
    for kind in ("pose_errors", "layout_errors"):
        for inst, rep in (errs.get(kind) or {}).items():
            if isinstance(rep, dict) and rep.get("within_tolerance") is False and rep.get("gating", True):
                out.append(inst)
    return out


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
    "ExecutorConfig", "Checkpoint", "ActionRecord", "SimState", "MotionSnapshot", "Executor", "apply_goal_override", "AGENT_INSTANCE",
]
