"""Resting light objects and the sleep they never reach (vec_next 2026-10-30, ``docs/VEC_NEXT.md`` section 1, "small-object
trembling"), shared by both drivers so that they run the same physics: ``run_isg`` (one scene; demo generation) and ``run_isg_vec``
(N scenes; evaluation) take the same flags with the same defaults (:func:`add_arguments`, :func:`export_env`) and apply them at the
same points of the loader contract.

* The rigid sleep threshold: ``physxRigidBody:sleepThreshold`` of every free, non-articulated object (:func:`is_free_rigid`) = 0.005,
  the PhysX default (OmniGibson sets 5e-5 on everything, at which a placed light object's TGS residual keeps its stack awake for good) -
  the scene's own objects after the world build (:func:`apply_to_scenes`, simulator stopped), the task objects at every TRO restore
  (:func:`apply_to_objects`). Articulations and the robot keep 5e-5.
* The sleep assist (:class:`SleepAssist`, after every control step): a free task object that is awake, moved less than 1 mm / 3 mrad over
  the last 30 control steps, does not touch the robot, has no MOVING end effector within 5 cm of its box and no awake tracked body resting
  on it, and weighs at most 5 kg, is put to sleep; a slept object is woken when a moving end effector comes within 5 cm of its box.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch as th

#: the defaults of both drivers (adopted by vec_next on 2026-10-30, round D)
RIGID_SLEEP_THRESHOLD = 0.005
SLEEP_ASSIST = "30,1,3"
SLEEP_ASSIST_MODE = "sleep"
SLEEP_ASSIST_MAX_KG = 5.0
SLEEP_ASSIST_WAKE_M = 0.05


def add_arguments(g) -> None:
    """The rigid-body / sleep-assist flags of both drivers on the argparse group (or parser) ``g``."""
    g.add_argument("--rigid-sleep-threshold", type=float, default=RIGID_SLEEP_THRESHOLD,
                   help="physxRigidBody:sleepThreshold (mass-normalised kinetic energy) of every free non-articulated object; articulations and the robot keep "
                        "OmniGibson's 5e-5. Default 0.005 = the PhysX default (adopted 2026-10-30: at 5e-5 a placed light object's TGS rocking keeps its stack awake "
                        "forever and in some clone configurations the rocking never decays - the 'not_settled' failures; at 0.005 the stack sleeps within 0.4 s). "
                        "0 = stock 5e-5")
    g.add_argument("--rigid-solver-iterations", default=None, metavar="POS,VEL",
                   help="solver position,velocity iteration counts of every free non-articulated object (scene objects at the build, task objects at the "
                        "restore); articulations and the robot keep OmniGibson's 32,1. vec_next 2026-10-30: the teapot's convex-hull base chatters on the "
                        "bar forever at 32,1 (never asleep at any sleep threshold up to 0.05) and sleeps within 0.5 s at 64,8 (probe R7). None = stock")
    g.add_argument("--sleep-assist", default=SLEEP_ASSIST, metavar="W,POS_MM,ROT_MRAD",
                   help="put a free non-articulated task object to sleep once it has moved less than POS_MM / ROT_MRAD over the last W control steps and does "
                        "not touch the robot (vec_next 2026-10-30: PhysX's sleep filter never fires on a resting convex hull's 120 Hz chatter, so the stack "
                        "rocks for the rest of the episode in some clones; a forced sleep sticks). Default 30,1,3 = 1 s, 1 mm, 3 mrad (adopted 2026-10-30: "
                        "the lid-on-teapot alias 8/8 slots at step 1002 like N=1 and the demo; stock 3/8). Recorded per object in meta.sleep_assist. 'off' = stock")
    g.add_argument("--sleep-assist-mode", default=SLEEP_ASSIST_MODE, choices=["sleep", "clamp"],
                   help="sleep (default): PhysX putToSleep once at rest, woken again when a gripper comes within --sleep-assist-wake-radius of its box (without "
                        "the wake a gripper touching a slept body kicks it: round D sleep mode, sprout 180-385 mrad/step); clamp: zero the velocities every "
                        "control step and stay awake (holds the lid-on-teapot stack in 5 of 8 slots only)")
    g.add_argument("--sleep-assist-max-mass", type=float, default=SLEEP_ASSIST_MAX_KG, metavar="KG",
                   help="sleep assist: objects heavier than this are never assist-slept (free furniture; a support slept under awake bodies jolts them). 0 = no cap")
    g.add_argument("--sleep-assist-wake-radius", type=float, default=SLEEP_ASSIST_WAKE_M, metavar="M",
                   help="sleep mode: wake a slept object when an end effector is within this distance of its bounding box, and do not sleep it while one is "
                        "(default 0.05 m: at 0.15 the gripper hovering over the released lid kept the teapot stack awake for the whole settle tail; 0 = never)")


def export_env(args) -> None:
    """The sleep-assist flags into the environment :class:`SleepAssist` reads (``ISG_SLEEP_ASSIST*``)."""
    if getattr(args, "sleep_assist", None) is not None:
        os.environ["ISG_SLEEP_ASSIST"] = str(args.sleep_assist)  # read by SleepAssist (sleep_assist_config); 'off' / '0' disable
    os.environ["ISG_SLEEP_ASSIST_MODE"] = str(getattr(args, "sleep_assist_mode", "sleep"))
    os.environ["ISG_SLEEP_ASSIST_WAKE_M"] = str(getattr(args, "sleep_assist_wake_radius", 0.05))
    os.environ["ISG_SLEEP_ASSIST_MAX_KG"] = str(getattr(args, "sleep_assist_max_mass", 5.0))


def rigid_overrides(args) -> Dict[str, Any]:
    """``--rigid-sleep-threshold`` / ``--rigid-solver-iterations`` as the ``rigid_*`` physics overrides (empty: stock)."""
    ov: Dict[str, Any] = {}
    if getattr(args, "rigid_sleep_threshold", 0.0):
        ov["rigid_sleep_threshold"] = float(args.rigid_sleep_threshold)
    if getattr(args, "rigid_solver_iterations", None):
        pos_it, _, vel_it = str(args.rigid_solver_iterations).replace("+", ",").replace(":", ",").partition(",")  # 64+8 for sbatch --export
        if pos_it.strip():
            ov["rigid_solver_position_iterations"] = int(pos_it)
        if vel_it.strip():
            ov["rigid_solver_velocity_iterations"] = int(vel_it)
    return ov


def sleep_assist_config() -> Optional[Tuple[int, float, float]]:
    """``ISG_SLEEP_ASSIST=W,POS_MM,ROT_MRAD`` (set from ``--sleep-assist``): a free task object that moved less than POS_MM / ROT_MRAD over the
    last W control steps and does not touch the robot is put to sleep (vec_next 2026-10-30: PhysX's own sleep filter never fires on the
    convex-hull chatter of a resting object - probe F1: a forced sleep sticks). None = off."""
    raw = os.environ.get("ISG_SLEEP_ASSIST", "").strip()
    if not raw or raw in ("0", "off"):
        return None
    w, pos_mm, rot_mrad = (raw.replace("+", ",").replace(":", ",").split(",") + ["", ""])[:3]
    return int(w or 30), float(pos_mm or 1.0) / 1000.0, float(rot_mrad or 3.0) / 1000.0


_ASSIST_DEBUG = os.environ.get("ISG_SLEEP_ASSIST_DEBUG", "0") == "1"


def sleep_assist_mode() -> str:
    """``ISG_SLEEP_ASSIST_MODE`` (``--sleep-assist-mode``): ``sleep`` (default) calls ``obj.sleep()`` (PhysX putToSleep) and wakes the object
    again when a moving gripper comes within :func:`sleep_assist_wake_radius` of it (without that wake a body the robot touches again kicks:
    round D, the brussels sprout spun 180-385 mrad per step when the gripper came back to the bowl); ``clamp`` zeroes the resting object's
    velocities every control step and leaves it awake."""
    m = os.environ.get("ISG_SLEEP_ASSIST_MODE", "sleep").strip().lower()
    return m if m in ("clamp", "sleep") else "sleep"


def sleep_assist_max_mass() -> float:
    """``ISG_SLEEP_ASSIST_MAX_KG`` (``--sleep-assist-max-mass``): heavier free objects (free furniture such as a breakfast table) are never
    assist-slept - they do not chatter, and putting a support to sleep under awake bodies jolts them (kielbasa: the pack rotated 20-90 mrad
    per step when the table under its platter was slept at step 751, and landed 0.43 rad off in 2 of 8 slots)."""
    try:
        return float(os.environ.get("ISG_SLEEP_ASSIST_MAX_KG", "5"))
    except ValueError:
        return 5.0


def sleep_assist_wake_radius() -> float:
    """``ISG_SLEEP_ASSIST_WAKE_M`` (``--sleep-assist-wake-radius``): a slept object is woken (PhysX wakeUp) as soon as one of the robot's
    end effectors comes within this distance of its bounding box, so it is awake BEFORE the gripper touches it (the kick of round D: the
    gripper, an articulation link, does not wake a slept body in time; the contact then resolves a deep penetration at once)."""
    try:
        return float(os.environ.get("ISG_SLEEP_ASSIST_WAKE_M", "0.05"))
    except ValueError:
        return 0.05


def is_free_rigid(obj) -> bool:
    """A free (not fixed, not kinematic), non-articulated (no movable joint) object that is not the robot: the objects the rigid-body
    overrides apply to. Fixed-joint-only assets (the meta-link articulations of the teapot or a lid) count as rigid."""
    if getattr(obj, "fixed_base", False) or getattr(obj, "kinematic_only", False) or getattr(obj, "category", "") == "agent":
        return False
    return not (int(getattr(obj, "n_dof", 0) or 0) > 0 or len(getattr(obj, "joints", {}) or {}) > 0)


def apply_to_objects(objs: Iterable[Any], ov: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The rigid-body overrides of ``ov`` (``rigid_sleep_threshold``, ``rigid_solver_position_iterations`` /
    ``rigid_solver_velocity_iterations``) on every free, non-articulated object of ``objs`` - a TRO restore's task objects, which the
    build-time pass (:func:`apply_to_scenes`) does not reach. Returns the report entry (None when no override is on)."""
    ov = ov or {}
    thr = ov.get("rigid_sleep_threshold")
    pos_it, vel_it = ov.get("rigid_solver_position_iterations"), ov.get("rigid_solver_velocity_iterations")
    if thr is None and pos_it is None and vel_it is None:
        return None
    names, skipped = [], []
    for obj in objs:
        if not is_free_rigid(obj):
            skipped.append(obj.name)
            continue
        try:
            if thr is not None:
                obj.sleep_threshold = float(thr)
            if pos_it is not None:
                obj.solver_position_iteration_count = int(pos_it)
            if vel_it is not None:
                obj.solver_velocity_iteration_count = int(vel_it)
            names.append(obj.name)
        except Exception as ex:  # noqa: BLE001
            skipped.append(f"{obj.name}!{type(ex).__name__}")
    out: Dict[str, Any] = {"set": names, "skipped": skipped}
    if thr is not None:
        out["threshold"] = float(thr)
    if pos_it is not None or vel_it is not None:
        out["solver_iterations"] = {"position": None if pos_it is None else int(pos_it), "velocity": None if vel_it is None else int(vel_it)}
    return out


def apply_to_scenes(scenes: Iterable[Any], ov: Dict[str, Any]) -> Dict[str, Any]:
    """The rigid-body overrides of ``ov`` on every free, non-articulated object of ``scenes`` after the world build (the simulator
    stopped). Returns ``{"rigid_sleep_threshold": ..., "rigid_solver_iterations": ...}`` for the overrides that are on."""
    out: Dict[str, Any] = {}
    scenes = list(scenes)
    if "rigid_sleep_threshold" in ov:
        # vec_next (2026-10-30, the small-object trembling): physxRigidBody:sleepThreshold of every FREE, NON-ARTICULATED object (no joints,
        # not fixed_base, not kinematic, not the robot) - the resting light objects whose stack never sleeps at OmniGibson's 5e-5 and whose
        # TGS residual then drives a limit cycle in some clone configurations. Articulations (doors, drawers, robots) keep the stock value.
        thr = float(ov["rigid_sleep_threshold"])
        n_set, names = 0, []
        for scene in scenes:
            for obj in scene.objects:
                if not is_free_rigid(obj):
                    continue
                try:
                    obj.sleep_threshold = thr
                    n_set += 1
                    if scene.idx == 0:
                        names.append(obj.name)
                except Exception:  # noqa: BLE001
                    pass
        out["rigid_sleep_threshold"] = {"threshold": thr, "objects": n_set, "scene0_names": names[:40]}
    if "rigid_solver_position_iterations" in ov or "rigid_solver_velocity_iterations" in ov:
        # vec_next (2026-10-30, the teapot-on-bar chatter): solver iteration counts of every free non-articulated scene object (the task
        # objects get theirs at the restore, apply_to_objects); articulations and the robot keep OmniGibson's 32 / 1
        pos_it, vel_it = ov.get("rigid_solver_position_iterations"), ov.get("rigid_solver_velocity_iterations")
        n_set = 0
        for scene in scenes:
            for obj in scene.objects:
                if not is_free_rigid(obj):
                    continue
                try:
                    if pos_it is not None:
                        obj.solver_position_iteration_count = int(pos_it)
                    if vel_it is not None:
                        obj.solver_velocity_iteration_count = int(vel_it)
                    n_set += 1
                except Exception:  # noqa: BLE001
                    pass
        out["rigid_solver_iterations"] = {"position": pos_it, "velocity": vel_it, "objects": n_set}
    return out


class SleepAssist:
    """``--sleep-assist`` for one scene's robot and tracked task objects: :meth:`step` after every control step (``run_isg_vec``'s
    ``SlotRecorder.record``, ``run_isg``'s ``Executor.step``), :meth:`trim` when the executor rolls the world back to a checkpoint,
    :meth:`report` for ``meta.sleep_assist``. ``name`` prefixes the log lines (the slot)."""

    def __init__(self, robot, tracked: Dict[str, Any], name: str, config: Optional[Tuple[int, float, float]] = None):
        self.robot, self.tracked, self.name = robot, tracked, name
        self.config = config if config is not None else sleep_assist_config()
        self.mode = sleep_assist_mode()
        self.wake_m = sleep_assist_wake_radius()
        self.max_kg = sleep_assist_max_mass()
        self.eef_prev: Dict[str, Any] = {}
        self.eef_speed: Dict[str, float] = {}
        self.slept: Dict[str, bool] = {}
        self.wakes: Dict[str, List[int]] = {}
        self.clamped: Dict[str, int] = {}
        self.hist: Dict[str, list] = {}
        self.free: Dict[str, bool] = {}
        self.sleeps: Dict[str, List[int]] = {}
        self.rollbacks: List[int] = []
        #: wall time of step() by phase (report: cost_ms_per_step) and the number of steps
        self.cost_s: Dict[str, float] = {}
        self.n_steps = 0

    def trim(self, step: int) -> None:
        """The executor rolled the world back to its checkpoint after control step ``step`` (``Executor.restore``; ``run_isg_vec`` never
        does): the rolled-back attempt's sleeps and wakes leave the record, as its steps leave the recording (``ArmStutter.trim``); every
        rest window restarts and the end effectors' speeds are measured anew. An object the assist put to sleep at or before ``step`` and
        did not wake since is still the assist's to wake when a gripper approaches."""
        step = int(step)
        for events in (self.sleeps, self.wakes):
            for inst in list(events):
                events[inst] = [s for s in events[inst] if s <= step]
                if not events[inst]:
                    del events[inst]
        self.hist.clear()
        self.eef_prev.clear()
        self.eef_speed.clear()
        self.slept = {inst: True for inst, s in self.sleeps.items() if self.mode == "sleep" and s[-1] > max(self.wakes.get(inst) or [-1])}
        self.rollbacks.append(step)

    def report(self) -> Dict[str, Any]:
        W, pos_tol, rot_tol = self.config
        out = {"mode": self.mode, "window_steps": W, "pos_tol_mm": pos_tol * 1000, "rot_tol_mrad": rot_tol * 1000,
               "wake_radius_m": self.wake_m, "max_mass_kg": self.max_kg, "wakes": {k: v[:50] for k, v in self.wakes.items()},
               "sleeps" if self.mode == "sleep" else "clamp_run_steps": {k: v[:50] for k, v in self.sleeps.items()},
               "clamped_steps": dict(self.clamped), "free_objects": [k for k, v in self.free.items() if v]}
        if self.rollbacks:
            out["rollbacks"] = self.rollbacks[:50]  # run_isg: the checkpoint steps the executor rolled back to (events after them trimmed)
        if self.n_steps:
            out["cost_ms_per_step"] = {k: round(1000.0 * v / self.n_steps, 3) for k, v in sorted(self.cost_s.items())}
            out["cost_steps"] = self.n_steps
        return out

    def step(self, step: int) -> None:
        """--sleep-assist (vec_next 2026-10-30, the small-object trembling): every free, non-articulated tracked object that is awake, has
        moved less than the configured distance / angle over the last W control steps and does not touch the robot is put to sleep
        (``obj.sleep()``: PhysX zeroes its velocities and keeps it still until an awake body touches it or a joint to one moves it). The
        windowed net-motion test is the ``pose_settled`` notion; PhysX's own filter sums the per-frame velocities and never fires on a
        convex-hull chatter that has no net motion. Recorded per object in ``meta.sleep_assist``."""
        W, pos_tol, rot_tol = self.config
        self.n_steps += 1
        cost, t_last = self.cost_s, [time.perf_counter()]

        def lap(name: str) -> None:  # the time since the last lap, to the phase that just ended
            now = time.perf_counter()
            cost[name] = cost.get(name, 0.0) + (now - t_last[0])
            t_last[0] = now

        try:
            from omnigibson.object_states import Touching
        except Exception:  # noqa: BLE001
            Touching = None
        # the end effectors' positions and speeds this step (a PARKED gripper 3 cm from a released pack must not block its sleep for the
        # whole tail - kielbasa; only a MOVING gripper within the radius wakes / blocks)
        eefs: Dict[str, Any] = {}
        try:
            for arm in self.robot.arm_names:
                pe = th.as_tensor(self.robot.get_eef_position(arm)).detach().clone()
                prev = self.eef_prev.get(arm)
                self.eef_speed[arm] = float(th.linalg.norm(pe - prev)) if prev is not None else 0.0
                self.eef_prev[arm] = pe
                eefs[arm] = pe
        except Exception:  # noqa: BLE001
            eefs = {}
        lap("eef")
        for inst, obj in self.tracked.items():
            free = self.free.get(inst)
            if free is None:
                try:
                    free = bool(is_free_rigid(obj))
                    if free and self.max_kg > 0:
                        try:
                            free = float(obj.mass) <= self.max_kg  # free furniture is never assist-slept (see sleep_assist_max_mass)
                        except Exception:  # noqa: BLE001
                            pass
                except Exception:  # noqa: BLE001
                    free = False
                self.free[inst] = free
            if not free:
                continue
            try:
                pos, quat = obj.get_position_orientation()
            except Exception:  # noqa: BLE001
                continue
            hist = self.hist.setdefault(inst, [])
            hist.append((pos.detach().clone(), quat.detach().clone()))
            if len(hist) > W:
                del hist[0]
            lap("pose")
            try:
                asleep = obj.is_asleep() if callable(getattr(obj, "is_asleep", None)) else bool(getattr(obj, "is_asleep", False))
            except Exception:  # noqa: BLE001
                asleep = False
            lap("asleep")
            # proactive wake: a gripper approaching a slept object wakes it before the contact (sleep mode)
            near = False
            if self.mode == "sleep" and self.wake_m > 0:
                try:
                    lo, hi = obj.aabb  # the distance from each MOVING end effector to the object's axis-aligned box (0 inside)
                    lo, hi = th.as_tensor(lo), th.as_tensor(hi)
                    dmin = float("inf")
                    for arm, pe in eefs.items():
                        if self.eef_speed.get(arm, 0.0) < 0.0003:  # parked (< 0.3 mm per control step = 1 cm/s): no approach
                            continue
                        gap = th.maximum(th.maximum(lo - pe, pe - hi), th.zeros(3))
                        dmin = min(dmin, float(th.linalg.norm(gap)))
                    near = dmin < self.wake_m
                except Exception:  # noqa: BLE001
                    near = False
                lap("near")
                if near and asleep and self.slept.get(inst):
                    try:
                        obj.wake()
                        self.slept[inst] = False
                        self.wakes.setdefault(inst, []).append(step)
                        if len(self.wakes[inst]) <= 3:
                            print(f"SLEEP_ASSIST {self.name} step {step}: {inst} woken (a moving gripper {dmin * 100:.1f} cm from its box, radius {self.wake_m * 100:.0f} cm)", flush=True)
                    except Exception as ex:  # noqa: BLE001
                        print(f"SLEEP_ASSIST {self.name} step {step}: {inst} wake() failed: {ex!r}", flush=True)
                    continue
            debug = _ASSIST_DEBUG and step % 30 == 0
            # the rest window must be free of robot contact (vec_next 2026-10-31): an object held still by a motionless gripper
            # otherwise passes the window and is slept on the very step the fingers let go (tablespoon slot 1, step 549), freezing
            # the release mid-way. Every contact with the robot restarts the window.
            if not asleep and Touching is not None and hasattr(obj, "states") and Touching in obj.states:
                try:
                    if bool(obj.states[Touching].get_value(self.robot)):
                        hist.clear()
                        if debug:
                            print(f"SLEEP_ASSIST_DEBUG {self.name} step {step} {inst}: touching the robot (window restarted)", flush=True)
                        lap("touch")
                        continue
                except Exception as ex:  # noqa: BLE001
                    if debug:
                        print(f"SLEEP_ASSIST_DEBUG {self.name} step {step} {inst}: Touching(robot) failed {ex!r}", flush=True)
            lap("touch")
            if asleep or near or len(hist) < W:
                if debug:
                    print(f"SLEEP_ASSIST_DEBUG {self.name} step {step} {inst}: asleep={asleep} near={near} hist={len(hist)}", flush=True)
                continue
            p0, q0 = hist[0]
            dp = max(float(th.linalg.norm(pp - p0)) for pp, _ in hist)
            dq = max(2.0 * math.acos(min(1.0, abs(float(th.dot(q0, qq))))) for _, qq in hist)
            lap("rest")
            if dp >= pos_tol or dq >= rot_tol:
                if debug:
                    print(f"SLEEP_ASSIST_DEBUG {self.name} step {step} {inst}: moving dp={dp * 1000:.2f}mm dq={dq * 1000:.2f}mrad", flush=True)
                continue
            if Touching is not None and hasattr(obj, "states") and Touching in obj.states:
                try:
                    if bool(obj.states[Touching].get_value(self.robot)):
                        if debug:
                            print(f"SLEEP_ASSIST_DEBUG {self.name} step {step} {inst}: at rest but touching the robot", flush=True)
                        continue
                except Exception as ex:  # noqa: BLE001
                    if debug:
                        print(f"SLEEP_ASSIST_DEBUG {self.name} step {step} {inst}: Touching(robot) failed {ex!r}", flush=True)
                # never sleep a SUPPORT while an awake tracked free body rests on it: putting the support to sleep under it jolts it
                # (kielbasa, step 751). "Rests on" = touching and higher (its box centre above this object's); the top of a stack sleeps
                # first, the support follows once everything on it is asleep (a symmetric rule deadlocked pack and platter).
                blocked = False
                try:
                    z_self = float(th.as_tensor(obj.aabb_center)[2])
                except Exception:  # noqa: BLE001
                    z_self = float(pos[2])
                for inst2, obj2 in self.tracked.items():
                    if inst2 == inst or not self.free.get(inst2):
                        continue
                    try:
                        asleep2 = obj2.is_asleep() if callable(getattr(obj2, "is_asleep", None)) else bool(getattr(obj2, "is_asleep", False))
                        if asleep2:
                            continue
                        z2 = float(th.as_tensor(obj2.aabb_center)[2])
                        if z2 > z_self and bool(obj.states[Touching].get_value(obj2)):
                            blocked = True
                            break
                    except Exception:  # noqa: BLE001
                        continue
                lap("support")
                if blocked:
                    if debug:
                        print(f"SLEEP_ASSIST_DEBUG {self.name} step {step} {inst}: at rest but an awake tracked body rests on it", flush=True)
                    continue
            try:
                if self.mode == "sleep":
                    obj.sleep()
                    self.slept[inst] = True
                    self.sleeps.setdefault(inst, []).append(step)
                    if len(self.sleeps[inst]) <= 3:
                        print(f"SLEEP_ASSIST {self.name} step {step}: {inst} put to sleep (net motion over {W} steps {dp * 1000:.2f} mm / {dq * 1000:.2f} mrad)", flush=True)
                else:
                    obj.keep_still()  # zero the velocities, stay awake: the chatter cannot grow into rocking, contacts keep responding normally
                    n = self.clamped.get(inst, 0)
                    self.clamped[inst] = n + 1
                    runs = self.sleeps.setdefault(inst, [])
                    if not runs or step - runs[-1] > 1:  # the start of a clamp run (consecutive steps are one run)
                        if len(runs) < 60:
                            runs.append(step)
                        if len(runs) <= 3:
                            print(f"SLEEP_ASSIST {self.name} step {step}: {inst} clamp run starts (net motion over {W} steps {dp * 1000:.2f} mm / {dq * 1000:.2f} mrad)", flush=True)
                    else:
                        runs[-1] = step  # extend: the list keeps run starts as [start, ..., last] pairs implicitly via the counter below
            except Exception as ex:  # noqa: BLE001
                print(f"SLEEP_ASSIST {self.name} step {step}: {inst} assist failed: {ex!r}", flush=True)
            lap("act")
