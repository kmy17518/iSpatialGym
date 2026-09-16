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

import logging
import math
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.oracle.actions import Action
from ispatialgym.oracle.execute import check_postcondition, fov_monitored, is_manipulation, run_action
from ispatialgym.oracle.motions import Motions, SkillError
from ispatialgym.oracle.qa import CameraModel, Monitors, QAStats, Verdict, category_of
from ispatialgym.oracle.recorder import PHASE_GOAL, PHASE_ORACLE, PHASE_SETTLE, Recorder, to_uint8_rgb
from ispatialgym.robot import conventions as C

log = logging.getLogger(__name__)

AGENT_INSTANCE = "agent.n.01_1"


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


#: ``ISpatialGymTask`` success-tracking attributes that a rollback must restore (sticky "first success" steps)
TASK_TRACKING_ATTRS = ("_first_success_step", "_first_settled_success_step", "_errors_at_first_success", "_last_goal_eval")


@dataclass
class ActionRecord:
    index: int
    name: str
    attempts: List[Optional[Dict[str, Any]]] = field(default_factory=list)
    verdict: Optional[Dict[str, Any]] = None
    step_start: int = 0
    step_end: int = 0
    targets_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "name": self.name, "attempts": list(self.attempts), "verdict": self.verdict,
                "step_start": self.step_start, "step_end": self.step_end, "n_steps": self.step_end - self.step_start,
                "targets_keys": list(self.targets_keys)}


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
        # -- the PhysX drive targets are not state: apply the restored hold targets with one (recorded) hold step;
        #    the monitors are off for it (their per-action whitelist is not set up yet)
        if self.motions is not None:
            self.run_generator(self.motions.hold(1), PHASE_ORACLE, action_index, attempt, monitor=False)
        return report

    # ------------------------------------------------------------------ one action

    def _scene_names(self, instances: Iterable[str]) -> set:
        out = set()
        for inst in instances:
            try:
                out.add(self.state.object(inst).name)
            except Exception:  # noqa: BLE001
                out.add(inst)
        return out

    def attempt(self, index: int, action: Action, attempt: int) -> Tuple[Optional[Verdict], Dict[str, Any]]:
        """Execute ``action`` once under the monitors and check its postcondition."""
        ctx: Dict[str, Any] = {}
        try:
            targets = action.targets(self.sidecar, self.state)
        except Exception as e:  # noqa: BLE001
            v = Verdict("targets_failed", f"{action.name}.targets() raised {e!r}", action=action.name, action_index=index, attempt=attempt, step=self.n,
                        details={"traceback": traceback.format_exc()})
            self.stats.record(v)
            return v, ctx
        ctx["targets_keys"] = sorted(targets.keys())
        if self.monitors is not None:
            self.monitors.begin_action(action, index, attempt, intended=self._scene_names(action.objects), held=self.state.held_objects(), manipulation=is_manipulation(action),
                                       fov=fov_monitored(action))
        try:
            gen = run_action(action, self.motions, self.state, targets, ctx)
            v = self.run_generator(gen, PHASE_ORACLE, index, attempt)
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

    def run_action(self, index: int, action: Action) -> Optional[Verdict]:
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
        verdict: Optional[Verdict] = None
        for k in range(self.cfg.retries + 1):
            t0 = time.time()
            verdict, ctx = self.attempt(index, action, k)
            rec.targets_keys = ctx.get("targets_keys", [])
            rec.attempts.append(None if verdict is None else verdict.to_dict())
            if verdict is None:
                if self.recorder is not None:
                    self.recorder.commit()
                log.info("action %d %s ok in %d steps (%.1fs)", index, action.summary() if hasattr(action, "summary") else action.name, self.n - rec.step_start, time.time() - t0)
                break
            log.warning("action %d %s attempt %d failed: %s", index, action.name, k, verdict)
            if k < self.cfg.retries:
                # roll back: the failed attempt leaves no trace in the demo (kept under rollbacks), retry from the checkpoint
                if self.recorder is not None:
                    self.recorder.rollback(index, k, verdict.to_dict(), state_before=cp.sim_state)
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
        verdict: Optional[Verdict] = None
        for i, action in enumerate(actions):
            verdict = self.run_action(i, action)
            if verdict is not None:
                break
        self.timings["actions_s"] = time.time() - t0
        n_action_steps = self.n
        return self.epilogue(verdict, n_action_steps)

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

    def epilogue(self, verdict: Optional[Verdict], n_action_steps: int) -> Dict[str, Any]:
        t0 = time.time()
        if self.monitors is not None:
            self.monitors.begin_epilogue(held=self.state.held_objects())
        window = max(int(self.cfg.settle_min_steps), int(math.ceil(self.cfg.settle_fraction * n_action_steps)))
        before = MotionSnapshot(self.robot, self.tracked)
        settle_verdict = None
        if verdict is None or verdict.category != "timeout":
            settle_verdict = self.run_generator(self._noop_actions(window), PHASE_SETTLE)
            if self.recorder is not None:
                self.recorder.commit()
        after = MotionSnapshot(self.robot, self.tracked)
        motion = before.delta(after)
        isg = self.isg
        binary_ever = bool(isg.get("binary_success_ever", isg.get("binary_success", False)))
        binary_final = bool(isg.get("binary_success", False))
        settled_task = bool(isg.get("settled_success", False))
        moved = motion.get("joint_rad", 0.0) > self.cfg.settled_joint_tol or motion.get("base_m", 0.0) > self.cfg.settled_pos_tol or motion.get("base_rad", 0.0) > self.cfg.settled_rot_tol or any(
            (v > self.cfg.settled_pos_tol) if k.endswith("_m") else (v > self.cfg.settled_rot_tol) for k, v in motion.items() if k not in ("joint_rad", "base_m", "base_rad")
        )
        out_of_tol = _out_of_tolerance(isg)
        if verdict is None and settle_verdict is not None:
            verdict = settle_verdict
        if verdict is None and not (binary_ever or binary_final):
            verdict = Verdict("not_reached", "binary goal not reached; unsatisfied: " + ", ".join(_unsatisfied(isg)), step=self.n)
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
            "motion_delta": motion,
            "actions": [r.to_dict() for r in self.action_records],
            "qa": self.monitors.report() if self.monitors is not None else {"stats": self.stats.to_dict()},
            "checkpoint_reports": list(self.checkpoint_reports),
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


def _out_of_tolerance(isg: Dict[str, Any]) -> List[str]:
    """Targets whose current error report says ``within_tolerance: False`` (pose and layout targets)."""
    errs = isg.get("errors") or {}
    out = []
    for kind in ("pose_errors", "layout_errors"):
        for inst, rep in (errs.get(kind) or {}).items():
            if isinstance(rep, dict) and rep.get("within_tolerance") is False:
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
