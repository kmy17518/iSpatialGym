"""``run_isg_vec`` -- ``run_isg`` for N instances per Isaac Sim process: ``num_envs`` cloned scenes of one template
(:class:`~ispatialgym.oracle.vec_loader.VecInstanceLoader`), one batched ``env.step`` per control step
(:class:`~ispatialgym.oracle.vec_stream.VecStreamExecutor`), the single-env evaluator's loader contract, step budget, metrics
and verdicts per slot::

    python -m ispatialgym.oracle.run_isg_vec --num-envs 8 --mode replay --task object_scaling --instance-indices 0 1 2 --out runs/vec
    python -m ispatialgym.oracle.run_isg_vec --num-envs 8 --mode replay --task object_scaling --instance-indices 0 --same-instance-all-slots --out runs/check
    python -m ispatialgym.oracle.run_isg_vec --num-envs 8 --mode policy --instances jobs/val_objscale.json --policy-server ws://127.0.0.1:8000 --out runs/eval

Instances: ``--task NAME --instance-indices I...`` (the eval dataset, as run_isg) or ``--instances FILE.json`` (a list of
``{"tro", "name", "demo", "length", "action_steps", "settle_steps", "goal_png", "task_id", "task_family", "task", "subtask",
"alias", "episode_index"}``; the val instances of a training run). All instances of one invocation share one template (one
world); they run in batches of ``--num-envs`` (``--same-instance-all-slots``: every slot of a batch runs the same instance - the
replay exactness check; ``--repeat K`` runs every batch K times).

Modes: ``replay`` (each slot's own demo, open loop: its action rows then the first ``--settle-steps`` of its settle rows, as
run_isg) and ``policy`` (a websocket policy server, BEHAVIOR's NumPy-MessagePack protocol: ACT's ``b1k_server.py`` or openpi's
``serve_b1k.py``; the slots of a batch share one session and one batched request per step,
:class:`~ispatialgym.eval.ws_policy.BatchedWSPolicy`). The robot carries the training data's observation layout
(``behavior_lerobot.robot_overrides``: head 720 x 720 at 40 mm, wrists 480 x 480, BEHAVIOR's 61-D proprioception; no depth).

Outputs: ``<out>/<name>[.slotK][.repR]/`` per slot episode with ``meta.json`` (run_isg's fields plus ``vec``), ``tiled_rgb.mp4``
(left wrist | head | right wrist over the external camera, streamed while the episode runs), ``init_rgb.png`` /
``final_rgb.png``, ``actions.npy``, ``trajectory.npz`` (per step: base pose, joints, eef poses, head camera pose, tracked object
poses, the task's error metric) and ``goal_rgb.png`` (policy: the goal image sent); per batch ``<out>/_batches/batch_NNNN_grid.mp4``
(every slot's tile in a grid) and ``batch_NNNN.json``; ``<out>/results.jsonl`` (one line per slot episode) and one
``RUN_ISG_RESULT`` line per slot on stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OMNIGIBSON_CHANGE_DETECTION_NUMPY", "1")
# the task's reward potential re-evaluates every goal predicate a second time per step per env (BehaviorTask.get_potential);
# the codebase's opt-in reuses the termination's evaluation (ispatialgym_task.SHARED_GOAL_EVAL) - on by default here
os.environ.setdefault("ISPATIALGYM_SHARED_GOAL_EVAL", "1")
# batched, bit-identical symmetry search of the goal check (geometry_fast; 73 x 73 equivalence pairs per attachment step)
os.environ.setdefault("ISPATIALGYM_FAST_GEOMETRY", "1")
os.environ.setdefault("ISPATIALGYM_FAST_ASSIGN", "1")

import numpy as np  # noqa: E402
import torch as th  # noqa: E402

from ispatialgym.oracle.run_isg import EXTERNAL_CAMERA_NAME, provenance, replay_counts, replay_rows, resolve_instance  # noqa: E402  (omnigibson import order)

import omnigibson as og  # noqa: E402

from ispatialgym.eval import instances as INST  # noqa: E402
from ispatialgym.eval.clone_frames import scene_translation  # noqa: E402
from ispatialgym.oracle.executor import ExecutorConfig, SimState  # noqa: E402
from ispatialgym.oracle.qa import Monitors, QAConfig  # noqa: E402
from ispatialgym.oracle.recorder import PHASE_ORACLE, PHASE_SETTLE, dump_json, json_default, save_png  # noqa: E402
from ispatialgym.oracle.vec_loader import RENDER_SETTING_KEYS, VecInstanceLoader  # noqa: E402
from ispatialgym.oracle.vec_rooms import FrameDumper, RoomVisibility, parse_dump_spec, resolve_rooms_override  # noqa: E402

_SLEEP_TRACE_EVERY = int(os.environ.get("ISG_SLEEP_TRACE_EVERY", "30")) if os.environ.get("ISG_SLEEP_TRACE", "0") == "1" else 0


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
    """``ISG_SLEEP_ASSIST_MODE`` (``--sleep-assist-mode``): ``clamp`` (default) zeroes the resting object's velocities every control step and
    leaves it awake; ``sleep`` calls ``obj.sleep()`` (PhysX putToSleep) - a body the robot touches again later wakes with a violent kick
    (round D sleep mode: the brussels sprout spun 180-385 mrad per step when the gripper came back to the bowl), so sleep is for experiments."""
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
from ispatialgym.oracle.vec_stream import Slot, VecStreamExecutor, isg_of  # noqa: E402
from ispatialgym.oracle.vec_video import VideoSink, grid_frame, label, tiled_frame, to_uint8_rgb  # noqa: E402
from ispatialgym.robot import conventions as C  # noqa: E402

log = logging.getLogger("ispatialgym.run_isg_vec")

EXIT_SUCCESS, EXIT_ORACLE_FAILED, EXIT_WORLDS, EXIT_ERROR = 0, 2, 3, 4
TRO_SUFFIX = "-tro_state.json"


# ==========================================================================================
# instance specs
# ==========================================================================================


def _names_from_alias(alias: Optional[str]) -> Dict[str, Optional[str]]:
    if not alias:
        return {}
    from ispatialgym.eval import names as NAMES

    try:
        return {"task_family": NAMES.task_family(alias), "task": NAMES.task_name(alias), "subtask": NAMES.subtask_name(alias)}
    except Exception:  # noqa: BLE001
        return {}


def specs_from_dataset(task: str, indices: Sequence[int], root: Optional[str], goal_dir: Optional[str], task_ids: Dict[str, int]) -> List[dict]:
    specs = []
    for i, rec in zip(indices, INST.resolve(task, indices, root)):
        alias = (rec.source or {}).get("variant")
        goal = None
        if goal_dir:
            cand = os.path.join(goal_dir, rec.instance_id + "-goal_rgb.png")
            goal = cand if os.path.exists(cand) else None
        specs.append({"name": rec.instance_id, "tro": rec.path("tro_state"), "demo": rec.demo_path(), "length": rec.episode.get("length"),
                      "action_steps": rec.episode.get("action_steps"), "settle_steps": rec.episode.get("settle_steps"), "instance_max_steps": rec.max_steps,
                      "task_family": rec.task_family, "task": rec.task, "subtask": rec.subtask, "instance": rec.instance, "alias": alias,
                      "goal_png": goal or rec.path("goal_rgb"), "goal_png_released": rec.path("goal_rgb"), "task_id": task_ids.get(alias) if alias else None,
                      "dataset_index": int(i), "source": "eval-dataset"})
    return specs


def specs_from_file(path: str) -> List[dict]:
    with open(path) as f:
        doc = json.load(f)
    items = doc.get("instances", doc) if isinstance(doc, dict) else doc
    specs = []
    for k, it in enumerate(items):
        if "tro" not in it:
            raise ValueError(f"{path}[{k}]: an instance needs 'tro'")
        it = dict(it)
        it.setdefault("name", os.path.basename(it["tro"])[: -len(TRO_SUFFIX)] if it["tro"].endswith(TRO_SUFFIX) else f"instance_{k}")
        it.setdefault("source", "file")
        if not it.get("task_family") and it.get("alias"):
            it.update({k2: v for k2, v in _names_from_alias(it["alias"]).items() if not it.get(k2)})
        specs.append(it)
    return specs


# ==========================================================================================
# sources
# ==========================================================================================


class ReplaySource:
    """A slot's demo rows (``run_isg.replay_rows``): None once they are exhausted."""

    def __init__(self, demo: str, episode: int, n_action: int, n_settle: int):
        self.it: Iterator[th.Tensor] = replay_rows(demo, episode, n_action, n_settle)
        self.phases = [PHASE_ORACLE] * int(n_action) + [PHASE_SETTLE] * int(n_settle)
        self.streamed = 0
        self.stop_reason: Optional[str] = None

    def __call__(self, obs, info):
        try:
            a = next(self.it)
        except StopIteration:
            self.stop_reason = "demo_end"
            return None
        self.streamed += 1
        return th.as_tensor(a, dtype=th.float32).reshape(-1)


class PolicySource:
    """A slot's action of the step, computed for the whole batch by :class:`BatchedPolicyStep` in the executor's ``pre_step``."""

    def __init__(self, step: "BatchedPolicyStep", idx: int):
        self.step, self.idx = step, idx
        self.streamed = 0
        self.actions: List[np.ndarray] = []
        self.stop_reason: Optional[str] = None

    def __call__(self, obs, info):
        a = self.step.pending.pop(self.idx, None)
        if a is None:
            self.stop_reason = self.step.error or "policy_none"
            return None
        self.streamed += 1
        self.actions.append(np.asarray(a, dtype=np.float32))
        return th.as_tensor(a, dtype=th.float32).reshape(-1)


class BatchedPolicyStep:
    """The executor's ``pre_step``: one batched request for every streaming slot's observation."""

    def __init__(self, client):
        self.client = client
        self.pending: Dict[int, np.ndarray] = {}
        self.error: Optional[str] = None

    def __call__(self, slots: List[Slot]) -> None:
        if self.error is not None:
            return
        try:
            self.pending = self.client.act({s.idx: s.last_obs for s in slots})
        except Exception as e:  # noqa: BLE001
            log.error("policy request failed: %r", e)
            traceback.print_exc()
            self.error = f"policy_error: {e!r}"
            self.pending = {}


# ==========================================================================================
# per-slot recording (rows + videos)
# ==========================================================================================


class SlotRecorder:
    """The per-step facts and videos of one slot: trajectory rows (base, joints, eef, head camera, tracked objects, the task's
    error metric), the tiled frame (streamed to ``tiled_rgb.mp4``) and the last head frame. ``record`` / ``commit`` are what
    :class:`VecStreamExecutor` calls."""

    def __init__(self, out_dir: str, robot, cam_key: str, tracked: Dict[str, Any], scene, sink: Optional[VideoSink], views: Sequence[str],
                 tile_px: int, external_rows: Optional[int], name: str, external_camera: Optional[str], compose: str = "cpu",
                 dump_steps: Optional[set] = None, timers: Optional[Dict[str, float]] = None, video_every: int = 1,
                 dump_spec=None, seg_visibility: bool = False, seg_every: int = 1):
        """``compose``: ``cpu`` composes the tile on the stepping thread (the 10-23 path); ``thread`` hands the raw frames to the
        sink's thread, which composes the tile (and the batch grid) there (lever S3-recording). ``dump_steps``: steps (and
        ``"last"``) whose head / wrist / external frames are written losslessly to ``<out>/frames/`` (the tiled-exposure
        experiment). ``timers``: shared dict receiving this recorder's wall time per stage. ``dump_spec`` / ``seg_visibility`` /
        ``seg_every``: campaign rooms' probes (``vec_rooms``: lossless frame dumps by a step spec, the room-visibility probe from
        instance segmentation)."""
        self.out_dir, self.robot, self.cam_key, self.tracked, self.scene = out_dir, robot, cam_key, dict(tracked), scene
        self.sink, self.views, self.tile_px, self.external_rows, self.name = sink, tuple(views), int(tile_px), external_rows, name
        self.external_camera = external_camera
        self.compose = str(compose)
        self.video_every = max(1, int(video_every))  # the per-slot videos (and the grid) keep every k-th step (2026-10-27: the tile is not composed in between)
        self.dump_steps = set(dump_steps) if dump_steps else None
        self.timers = timers if timers is not None else {}
        self.rows: Dict[str, list] = defaultdict(list)
        self.last_rgb: Optional[np.ndarray] = None
        self.last_tile: Optional[np.ndarray] = None
        self.last_frames: Optional[Dict[str, Optional[np.ndarray]]] = None
        self.n = 0
        self._wrist: Dict[str, Optional[str]] = {}
        self._assist = sleep_assist_config()
        self._assist_mode = sleep_assist_mode()
        self._assist_wake_m = sleep_assist_wake_radius()
        self._assist_max_kg = sleep_assist_max_mass()
        self._assist_eef_prev: Dict[str, Any] = {}
        self._assist_eef_speed: Dict[str, float] = {}
        self._assist_slept: Dict[str, bool] = {}
        self._assist_wakes: Dict[str, List[int]] = {}
        self._assist_radius: Dict[str, float] = {}
        self._assist_clamped: Dict[str, int] = {}
        self._assist_hist: Dict[str, list] = {}
        self._assist_free: Dict[str, bool] = {}
        self._assist_sleeps: Dict[str, List[int]] = {}
        self._offset = scene_translation(scene)
        # campaign rooms (2026-10-28): lossless frame dumps at chosen steps and the per-camera room-visibility probe
        self.dumper = FrameDumper(out_dir, dump_spec) if dump_spec is not None else None
        self.visibility: Optional[RoomVisibility] = None
        self._seg_visibility, self._seg_every = bool(seg_visibility), int(seg_every)
        if sink is not None:
            for v in self.views:
                sink.open(f"{name}:{v}", os.path.join(out_dir, f"{v}_rgb.mp4"))
            if self.compose == "thread":
                sink.register_tile(name, tile=self.tile_px, external_rows=self.external_rows)

    def dump(self, obs, step: int, tag: Optional[str] = None) -> None:
        """Lossless PNGs of this step's camera frames (``frames/step_XXXX_<camera>.png``)."""
        fr = self._frames(obs)
        d = os.path.join(self.out_dir, "frames")
        os.makedirs(d, exist_ok=True)
        for cam in ("head", "left", "right", "external"):
            if fr.get(cam) is not None:
                save_png(os.path.join(d, f"step_{int(step):04d}_{cam}.png"), fr[cam])
                if tag:
                    save_png(os.path.join(d, f"{tag}_{cam}.png"), fr[cam])

    def _sleep_assist(self, step: int) -> None:
        """--sleep-assist (vec_next 2026-10-30, the small-object trembling): every free, non-articulated tracked object that is awake, has
        moved less than the configured distance / angle over the last W control steps and does not touch the robot is put to sleep
        (``obj.sleep()``: PhysX zeroes its velocities and keeps it still until an awake body touches it or a joint to one moves it). The
        windowed net-motion test is the ``pose_settled`` notion; PhysX's own filter sums the per-frame velocities and never fires on a
        convex-hull chatter that has no net motion. Recorded per object in ``meta.sleep_assist``."""
        W, pos_tol, rot_tol = self._assist
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
                prev = self._assist_eef_prev.get(arm)
                self._assist_eef_speed[arm] = float(th.linalg.norm(pe - prev)) if prev is not None else 0.0
                self._assist_eef_prev[arm] = pe
                eefs[arm] = pe
        except Exception:  # noqa: BLE001
            eefs = {}
        for inst, obj in self.tracked.items():
            free = self._assist_free.get(inst)
            if free is None:
                try:
                    from ispatialgym.oracle.vec_loader import VecInstanceLoader

                    free = bool(VecInstanceLoader._is_free_rigid(obj))
                    if free and self._assist_max_kg > 0:
                        try:
                            free = float(obj.mass) <= self._assist_max_kg  # free furniture is never assist-slept (see sleep_assist_max_mass)
                        except Exception:  # noqa: BLE001
                            pass
                except Exception:  # noqa: BLE001
                    free = False
                self._assist_free[inst] = free
            if not free:
                continue
            try:
                pos, quat = obj.get_position_orientation()
            except Exception:  # noqa: BLE001
                continue
            hist = self._assist_hist.setdefault(inst, [])
            hist.append((pos.detach().clone(), quat.detach().clone()))
            if len(hist) > W:
                del hist[0]
            try:
                asleep = obj.is_asleep() if callable(getattr(obj, "is_asleep", None)) else bool(getattr(obj, "is_asleep", False))
            except Exception:  # noqa: BLE001
                asleep = False
            # proactive wake: a gripper approaching a slept object wakes it before the contact (sleep mode)
            near = False
            if self._assist_mode == "sleep" and self._assist_wake_m > 0:
                try:
                    lo, hi = obj.aabb  # the distance from each MOVING end effector to the object's axis-aligned box (0 inside)
                    lo, hi = th.as_tensor(lo), th.as_tensor(hi)
                    dmin = float("inf")
                    for arm, pe in eefs.items():
                        if self._assist_eef_speed.get(arm, 0.0) < 0.0003:  # parked (< 0.3 mm per control step = 1 cm/s): no approach
                            continue
                        gap = th.maximum(th.maximum(lo - pe, pe - hi), th.zeros(3))
                        dmin = min(dmin, float(th.linalg.norm(gap)))
                    near = dmin < self._assist_wake_m
                except Exception:  # noqa: BLE001
                    near = False
                if near and asleep and self._assist_slept.get(inst):
                    try:
                        obj.wake()
                        self._assist_slept[inst] = False
                        self._assist_wakes.setdefault(inst, []).append(step)
                        if len(self._assist_wakes[inst]) <= 3:
                            print(f"SLEEP_ASSIST {self.name} step {step}: {inst} woken (a moving gripper {dmin * 100:.1f} cm from its box, radius {self._assist_wake_m * 100:.0f} cm)", flush=True)
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
                        continue
                except Exception as ex:  # noqa: BLE001
                    if debug:
                        print(f"SLEEP_ASSIST_DEBUG {self.name} step {step} {inst}: Touching(robot) failed {ex!r}", flush=True)
            if asleep or near or len(hist) < W:
                if debug:
                    print(f"SLEEP_ASSIST_DEBUG {self.name} step {step} {inst}: asleep={asleep} near={near} hist={len(hist)}", flush=True)
                continue
            p0, q0 = hist[0]
            dp = max(float(th.linalg.norm(pp - p0)) for pp, _ in hist)
            dq = max(2.0 * math.acos(min(1.0, abs(float(th.dot(q0, qq))))) for _, qq in hist)
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
                    if inst2 == inst or not self._assist_free.get(inst2):
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
                if blocked:
                    if debug:
                        print(f"SLEEP_ASSIST_DEBUG {self.name} step {step} {inst}: at rest but an awake tracked body rests on it", flush=True)
                    continue
            try:
                if self._assist_mode == "sleep":
                    obj.sleep()
                    self._assist_slept[inst] = True
                    self._assist_sleeps.setdefault(inst, []).append(step)
                    if len(self._assist_sleeps[inst]) <= 3:
                        print(f"SLEEP_ASSIST {self.name} step {step}: {inst} put to sleep (net motion over {W} steps {dp * 1000:.2f} mm / {dq * 1000:.2f} mrad)", flush=True)
                else:
                    obj.keep_still()  # zero the velocities, stay awake: the chatter cannot grow into rocking, contacts keep responding normally
                    n = self._assist_clamped.get(inst, 0)
                    self._assist_clamped[inst] = n + 1
                    runs = self._assist_sleeps.setdefault(inst, [])
                    if not runs or step - runs[-1] > 1:  # the start of a clamp run (consecutive steps are one run)
                        if len(runs) < 60:
                            runs.append(step)
                        if len(runs) <= 3:
                            print(f"SLEEP_ASSIST {self.name} step {step}: {inst} clamp run starts (net motion over {W} steps {dp * 1000:.2f} mm / {dq * 1000:.2f} mrad)", flush=True)
                    else:
                        runs[-1] = step  # extend: the list keeps run starts as [start, ..., last] pairs implicitly via the counter below
            except Exception as ex:  # noqa: BLE001
                print(f"SLEEP_ASSIST {self.name} step {step}: {inst} assist failed: {ex!r}", flush=True)

    def _sleep_trace(self, step: int) -> None:
        """ISG_SLEEP_TRACE=1 (vec_next, the small-object trembling): one stdout line per slot every ISG_SLEEP_TRACE_EVERY steps with the tracked
        objects' sleep state, speeds and whether the robot touches them, plus the robot's own sleep state."""
        try:
            from omnigibson.object_states import Touching
        except Exception:  # noqa: BLE001
            Touching = None
        parts = []
        for inst, obj in self.tracked.items():
            try:
                asleep = obj.is_asleep() if callable(getattr(obj, "is_asleep", None)) else getattr(obj, "is_asleep", None)
                v = float(th.linalg.norm(obj.get_linear_velocity())) * 1000
                w = float(th.linalg.norm(obj.get_angular_velocity())) * 1000
                touch = ""
                if Touching is not None and hasattr(obj, "states") and Touching in obj.states:
                    touch = f" robot_touch={bool(obj.states[Touching].get_value(self.robot))}"
                parts.append(f"{inst.split('.')[0]}: asleep={asleep} v={v:.2f}mm/s w={w:.1f}mrad/s{touch}")
            except Exception as ex:  # noqa: BLE001
                parts.append(f"{inst.split('.')[0]}: n/a ({type(ex).__name__})")
        try:
            r_asleep = self.robot.is_asleep() if callable(getattr(self.robot, "is_asleep", None)) else getattr(self.robot, "is_asleep", None)
        except Exception:  # noqa: BLE001
            r_asleep = "n/a"
        print(f"SLEEP_TRACE {self.name} step {step} | robot asleep={r_asleep} | " + " | ".join(parts), flush=True)

    def _probes(self, step: int, obs, info) -> None:
        if self.dumper is None and not self._seg_visibility:
            return
        robs = self._robot_obs(obs)
        if not self._wrist:
            for arm in ("left", "right"):
                self._wrist[arm] = next((k for k in robs if k.endswith(f"{arm}_realsense_link:Camera:0")), None)
        if self._seg_visibility and self.visibility is None:
            self.visibility = RoomVisibility(self.scene, self.robot.name, {"head": self.cam_key, "left": self._wrist.get("left"), "right": self._wrist.get("right")},
                                             every=self._seg_every)
        if self.visibility is not None:
            self.visibility.observe(step, obs, info)
        if self.dumper is not None and self.dumper.spec.wants(step):
            self.dumper.maybe(step, self._frames(obs))

    def _robot_obs(self, obs) -> Dict[str, Any]:
        return (obs or {}).get(self.robot.name) or {}

    def _frames(self, obs, convert: bool = True) -> Dict[str, Optional[np.ndarray]]:
        """The step's camera images as uint8 HWC RGB host arrays (``convert``), or the raw observation arrays / tensors (RGBA, host
        or device) for a consumer that converts them itself (the sink's thread)."""
        robs = self._robot_obs(obs)
        if not self._wrist:
            for arm in ("left", "right"):
                self._wrist[arm] = next((k for k in robs if k.endswith(f"{arm}_realsense_link:Camera:0")), None)
        cv = to_uint8_rgb if convert else (lambda x: x)
        out = {"head": cv((robs.get(self.cam_key) or {}).get("rgb")) if self.cam_key in robs else None}
        for arm in ("left", "right"):
            k = self._wrist.get(arm)
            out[arm] = cv(robs[k]["rgb"]) if k and k in robs else None
        ext = None
        if self.external_camera and isinstance(obs, dict) and "external" in obs:
            try:
                ext = cv(obs["external"][self.external_camera]["rgb"])
            except Exception:  # noqa: BLE001
                ext = None
        out["external"] = ext
        return out

    def _pose(self, pos, quat) -> np.ndarray:
        p = th.as_tensor(pos, dtype=th.float32).reshape(3)
        if self._offset is not None:
            p = p - self._offset
        return np.concatenate([p.detach().cpu().numpy(), th.as_tensor(quat, dtype=th.float32).reshape(4).detach().cpu().numpy()]).astype(np.float32)

    def frame_step(self, obs, step_text: str = "", step: int = 0) -> Optional[np.ndarray]:
        """Compose (and queue) this step's videos from ``obs``; returns the tiled frame (``compose="cpu"``) or None when the sink's
        thread composes it or the step is skipped by ``video_every`` (the head frame is still kept for ``final_rgb.png``)."""
        t0 = time.time()
        if self.video_every > 1 and (int(step) % self.video_every) and self.sink is not None:
            head = self._frames(obs, convert=False).get("head")
            if head is not None:
                self.last_rgb = head
            self.timers["record_frames_s"] = self.timers.get("record_frames_s", 0.0) + (time.time() - t0)
            return None
        thread = self.sink is not None and self.views and self.compose == "thread"
        fr = self._frames(obs, convert=not thread)
        if fr["head"] is not None:
            self.last_rgb = fr["head"]  # raw in thread mode; save_png converts
        self.timers["record_frames_s"] = self.timers.get("record_frames_s", 0.0) + (time.time() - t0)
        t0 = time.time()
        tile = None
        if thread:
            # the raw frames (fresh arrays of this step: the annotator copies per call, nothing else writes them) go to the sink's thread,
            # which converts them, composes the tile for the per-slot video and keeps it for the batch grid
            self.sink.put_raw(self.name, fr, step_text, self.views)
        elif self.sink is not None and self.views:
            for v in self.views:
                if v == "tiled":
                    tile = tiled_frame(fr["left"], fr["head"], fr["right"], fr["external"], tile=self.tile_px, external_rows=self.external_rows if fr["external"] is not None else None)
                    if step_text:
                        label(tile, step_text)
                    self.sink.put(f"{self.name}:tiled", tile)
                elif v == "head":
                    self.sink.put(f"{self.name}:head", fr["head"])
                elif v == "external":
                    self.sink.put(f"{self.name}:external", fr["external"])
                elif v in ("left_wrist", "right_wrist"):
                    self.sink.put(f"{self.name}:{v}", fr[v.split("_")[0]])
        elif fr["head"] is not None and self.sink is not None:
            tile = tiled_frame(fr["left"], fr["head"], fr["right"], fr["external"], tile=self.tile_px, external_rows=self.external_rows if fr["external"] is not None else None)
        if tile is not None:
            self.last_tile = tile
        self.timers["record_tile_s"] = self.timers.get("record_tile_s", 0.0) + (time.time() - t0)
        return tile

    def start(self, obs0, info0=None) -> None:
        self.frame_step(obs0, "step 0")
        if self.dump_steps and 0 in self.dump_steps:
            self.dump(obs0, 0)
        self._probes(0, obs0, info0)

    def record(self, step: int, action, obs, info, state=None, phase: int = PHASE_ORACLE, action_index: int = 0, attempt: int = 0, reward: float = 0.0,
               terminated: bool = False, truncated: bool = False, sim_time=None) -> None:
        t0 = time.time()
        r = self.rows
        robot = self.robot
        r["step"].append(int(step))
        r["phase"].append(int(phase))
        r["action"].append(np.asarray(th.as_tensor(action).detach().cpu(), dtype=np.float32))
        r["joint_positions"].append(np.asarray(robot.get_joint_positions().detach().cpu(), dtype=np.float32))
        r["base_pose"].append(self._pose(*robot.get_position_orientation()))
        for arm in ("left", "right"):
            r[f"eef_pose_{arm}"].append(self._pose(*robot.eef_links[arm].get_position_orientation()))
        if self.cam_key in getattr(robot, "sensors", {}):
            r["head_camera_pose"].append(self._pose(*robot.sensors[self.cam_key].get_position_orientation()))
        for inst, obj in self.tracked.items():
            r[f"object_pose/{inst}"].append(self._pose(*obj.get_position_orientation()))
        isg = isg_of(info)
        r["eval/symbolic"].append(bool(isg.get("goal", False)))
        r["eval/settled"].append(bool(isg.get("settled", False)))
        r["eval/geometric_within"].append(bool(isg.get("measure_within", False)))
        metric = isg.get("error_metric") or {}
        for key in ("position_error", "orientation_error"):
            v = metric.get(key)
            r[f"eval/{key}"].append(float("nan") if v is None else float(v))
        self.n = int(step)
        if self._assist is not None:
            self._sleep_assist(int(step))
        if _SLEEP_TRACE_EVERY and int(step) % _SLEEP_TRACE_EVERY == 0:
            self._sleep_trace(int(step))
        self.timers["record_rows_s"] = self.timers.get("record_rows_s", 0.0) + (time.time() - t0)
        self.frame_step(obs, f"step {step}", step=int(step))
        if self.dump_steps and int(step) in self.dump_steps:
            self.dump(obs, int(step))
        if self.dump_steps and "last" in self.dump_steps:
            self._last_obs_for_dump = obs
        self._probes(int(step), obs, info)

    def commit(self) -> int:
        if self.dump_steps and "last" in self.dump_steps and getattr(self, "_last_obs_for_dump", None) is not None:
            self.dump(self._last_obs_for_dump, self.n, tag="last")
            self._last_obs_for_dump = None
        return self.n

    def close(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self._assist is not None:
            W, pos_tol, rot_tol = self._assist
            out["sleep_assist"] = {"mode": self._assist_mode, "window_steps": W, "pos_tol_mm": pos_tol * 1000, "rot_tol_mrad": rot_tol * 1000,
                                   "wake_radius_m": self._assist_wake_m, "max_mass_kg": self._assist_max_kg, "wakes": {k: v[:50] for k, v in self._assist_wakes.items()},
                                   "sleeps" if self._assist_mode == "sleep" else "clamp_run_steps": {k: v[:50] for k, v in self._assist_sleeps.items()},
                                   "clamped_steps": dict(self._assist_clamped), "free_objects": [k for k, v in self._assist_free.items() if v]}
        out.update(self._close_rest())
        return out

    def _close_rest(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.sink is not None:
            for v in self.views:
                self.sink.close_name(f"{self.name}:{v}")
            out["videos"] = {v: os.path.join(self.out_dir, f"{v}_rgb.mp4") for v in self.views}
        if self.visibility is not None:
            out["rooms_visibility"] = self.visibility.write(self.out_dir)
            s = self.visibility.summary()
            out["rooms_seen"] = {r: {"frames": v["frames"], "max_frac": round(v["max_frac"], 6)} for r, v in s["rooms"].items()}
        if self.dumper is not None:
            out["frames_dumped"] = self.dumper.n
        if self.rows:
            arrays = {k.replace("/", "__"): np.asarray(v) for k, v in self.rows.items()}
            np.savez_compressed(os.path.join(self.out_dir, "trajectory.npz"), **arrays)
            out["trajectory"] = os.path.join(self.out_dir, "trajectory.npz")
            np.save(os.path.join(self.out_dir, "actions.npy"), np.asarray(self.rows["action"], dtype=np.float32))
        return out


# ==========================================================================================
# texture wait for a batch
# ==========================================================================================


def wait_for_textures_batch(env, loaded: Sequence[int], robot_name: str, cam_key: str, max_s: float, min_frames: int = 5, stable_frames: int = 12,
                            rel_tol: float = 0.01) -> Tuple[Dict[str, Any], Any, Any]:
    """``texture_wait.wait_for_textures`` for every loaded clone: render (no physics) until no USD file load is in flight and
    every slot's head image sharpness has stopped rising; returns (status, obs_list, info_list) of a fresh ``get_obs``."""
    from ispatialgym.oracle import texture_wait as TW

    t0 = time.time()
    idx = th.tensor(list(loaded), dtype=th.long)
    renders, stable = 0, 0
    best: Dict[int, float] = {}
    status: Dict[str, Any] = {"complete": False, "why": None}
    obs_list, info_list = env.get_obs(env_indices=idx)
    while True:
        try:
            import omnigibson.lazy as lazy

            ctx = lazy.omni.usd.get_context()
            files, total = ctx.get_stage_loading_status()[1:3]
            loading = (files or 0) > 0 or (total or 0) > 0
        except Exception:  # noqa: BLE001
            loading = False
        risen = False
        for k, i in enumerate(loaded):
            img = obs_list[k][robot_name][cam_key]["rgb"]
            s = TW.sharpness(img)
            if s > best.get(i, 0.0) * (1.0 + rel_tol):
                risen = True
            best[i] = max(best.get(i, 0.0), s)
        stable = 0 if (loading or risen) else stable + 1
        renders += 1
        if renders >= min_frames and stable >= stable_frames:
            status.update({"complete": True, "why": "stable"})
            break
        if time.time() - t0 > max_s:
            status.update({"complete": False, "why": "max_s"})
            break
        og.sim.render()
        obs_list, info_list = env.get_obs(env_indices=idx)
    status.update({"renders": renders, "seconds": round(time.time() - t0, 1), "sharpness": {int(i): round(v, 2) for i, v in best.items()}})
    return status, obs_list, info_list


# ==========================================================================================
# arguments
# ==========================================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("instances")
    g.add_argument("--task", default=None, help="eval dataset: a task family, task or subtask name (ispatialgym.eval.instances)")
    g.add_argument("--instance-indices", type=int, nargs="+", default=[0])
    g.add_argument("--instances-root", default=None)
    g.add_argument("--instances", default=None, help="JSON list of instance specs (module docstring)")
    g.add_argument("--goal-dir", default=None, help="eval dataset: pre-saved goal images <dir>/<instance id>-goal_rgb.png (default: the released goal image)")
    g.add_argument("--task-ids", default=None, help="JSON {alias: task_id} for eval dataset instances (policy mode)")
    g.add_argument("--same-instance-all-slots", action="store_true", help="every slot of a batch runs the same instance (replay exactness / repeat rollouts)")
    g.add_argument("--repeat", type=int, default=1, help="run every batch this many times (outputs suffixed .repK)")
    g.add_argument("--limit-batches", type=int, default=None)
    g.add_argument("--world-group", type=int, default=None,
                   help="When the instances span several worlds (templates that are not the same world up to the TRO's task objects), "
                        "run only the K-th group (instance order); without it such a set exits with RUN_ISG_VEC_WORLDS and code 3")
    r = p.add_argument_group("run")
    r.add_argument("--num-envs", type=int, default=4)
    r.add_argument("--mode", default="replay", choices=["replay", "policy"])
    r.add_argument("--out", default="runs/vec")
    r.add_argument("--overwrite", action="store_true", help="re-run slot episodes whose meta.json exists")
    r.add_argument("--max-steps", type=int, default=None, help="run_isg's: default max(300, round(1.5 x demo length))")
    r.add_argument("--settle-steps", type=int, default=None, help="run_isg's: replay = the demo's settle rows streamed (default all), policy = the window (45)")
    r.add_argument("--stall-timeout", type=float, default=1800.0,
                   help="stuck-episode guard (2026-10-29): end the process (exit 75, RUN_ISG_VEC_STALL) when no batch step completes for this many seconds "
                        "during a batch's stepping - a hung simulator; episodes are limited by their step budget only. 0 = off")
    r.add_argument("--wallclock", type=float, default=0.0,
                   help="seconds a slot's actions may take before its episode is cut (0 = never, the default here: run_isg's 1 s x max_steps assumes a "
                        "single env at ~10 steps/s, while a batch step on a shared GPU takes 1-2 s and cut 17%% of the 2026-10-23 policy episodes early)")
    r.add_argument("--settle-eval", default="auto", choices=["auto", "on", "off"])
    r.add_argument("--success-metric", default="geometric", choices=["geometric", "symbolic"])
    r.add_argument("--tolerance-scale", type=float, default=None, help="default 3 with --mode policy, else 1")
    r.add_argument("--qa", action=argparse.BooleanOptionalAction, default=False, help="log the --qa metrics per slot (run_isg's meta.qa; costs per-step contact queries)")
    r.add_argument("--stop-on-success", action=argparse.BooleanOptionalAction, default=None, help="end a stream at the geometric success (default: policy on, replay off)")
    w = p.add_argument_group("world")
    w.add_argument("--settle-physics-steps", type=int, default=25)
    w.add_argument("--warm-renders", type=int, default=100, help="renders after the batch reset (the renderer's view adaptation) before the texture wait")
    w.add_argument("--texture-wait-s", type=float, default=float(os.environ.get("ISG_TEXTURE_WAIT_MAX_S", "240")), help="max seconds of the texture wait (0: none)")
    w.add_argument("--enhanced-determinism", action=argparse.BooleanOptionalAction, default=None)
    w.add_argument("--stock-handle-ag", action="store_true")
    w.add_argument("--scene-margin", type=float, default=5.0,
                   help="gap between neighbouring clones' live boxes in m (default 5 with the centred grid = the 2026-10-27 clone-placement fix, adopted "
                        "2026-10-29; OmniGibson's 10 = the 10-23 production placement: --layout grid --scene-margin 10)")
    w.add_argument("--layout", default="grid-centered", choices=["grid", "grid-centered", "line"],
                   help="clone placement: grid-centered (default since 2026-10-29: scene 0 at the origin, the clones on the cells around it nearest first; "
                        "the vec_speed ladder re-verification), grid (the 2026-10-23 production layout growing towards +x/+y), line (OmniGibson's row along +x)")
    w.add_argument("--rigid-sleep-threshold", type=float, default=0.005,
                   help="physxRigidBody:sleepThreshold (mass-normalised kinetic energy) of every free non-articulated object; articulations and the robot keep "
                        "OmniGibson's 5e-5. Default 0.005 = the PhysX default (adopted 2026-10-30: at 5e-5 a placed light object's TGS rocking keeps its stack awake "
                        "forever and in some clone configurations the rocking never decays - the 'not_settled' failures; at 0.005 the stack sleeps within 0.4 s). "
                        "0 = stock 5e-5")
    w.add_argument("--rigid-solver-iterations", default=None, metavar="POS,VEL",
                   help="solver position,velocity iteration counts of every free non-articulated object (scene objects at the build, task objects at the "
                        "restore); articulations and the robot keep OmniGibson's 32,1. vec_next 2026-10-30: the teapot's convex-hull base chatters on the "
                        "bar forever at 32,1 (never asleep at any sleep threshold up to 0.05) and sleeps within 0.5 s at 64,8 (probe R7). None = stock")
    w.add_argument("--sleep-assist", default="30,1,3", metavar="W,POS_MM,ROT_MRAD",
                   help="put a free non-articulated task object to sleep once it has moved less than POS_MM / ROT_MRAD over the last W control steps and does "
                        "not touch the robot (vec_next 2026-10-30: PhysX's sleep filter never fires on a resting convex hull's 120 Hz chatter, so the stack "
                        "rocks for the rest of the episode in some clones; a forced sleep sticks). Default 30,1,3 = 1 s, 1 mm, 3 mrad (adopted 2026-10-30: "
                        "the lid-on-teapot alias 8/8 slots at step 1002 like N=1 and the demo; stock 3/8). Recorded per object in meta.sleep_assist. 'off' = stock")
    w.add_argument("--sleep-assist-mode", default="sleep", choices=["sleep", "clamp"],
                   help="sleep (default): PhysX putToSleep once at rest, woken again when a gripper comes within --sleep-assist-wake-radius of its box (without "
                        "the wake a gripper touching a slept body kicks it: round D sleep mode, sprout 180-385 mrad/step); clamp: zero the velocities every "
                        "control step and stay awake (holds the lid-on-teapot stack in 5 of 8 slots only)")
    w.add_argument("--sleep-assist-max-mass", type=float, default=5.0, metavar="KG",
                   help="sleep assist: objects heavier than this are never assist-slept (free furniture; a support slept under awake bodies jolts them). 0 = no cap")
    w.add_argument("--sleep-assist-wake-radius", type=float, default=0.05, metavar="M",
                   help="sleep mode: wake a slept object when an end effector is within this distance of its bounding box, and do not sleep it while one is "
                        "(default 0.05 m: at 0.15 the gripper hovering over the released lid kept the teapot stack awake for the whole settle tail; 0 = never)")
    w.add_argument("--physics-override", action="append", default=[], metavar="KEY=VALUE",
                   help="EXPERIMENT (vec_next, the small-object trembling): physics settings - solver=TGS|PGS, broadphase=MBP|SAP|GPU, gpu_dynamics=0|1, "
                        "num_threads=N, enhanced_determinism=0|1 (scene, applied stopped after the build); sleep_threshold=F, solver_position_iterations=N, "
                        "solver_velocity_iterations=N (macros applied before the build); stabilization=F (scene flag + every free object's threshold)")
    w.add_argument("--scene-offsets", default=None, metavar="X,Y,Z;X,Y,Z;...",
                   help="EXPERIMENT: explicit clone offsets in m for scenes 1, 2, ... (overrides --layout; the last entry repeats); e.g. '0,0,30' stacks the clone 30 m above the origin scene")
    w.add_argument("--load-rooms", default="template",
                   help="room instances loaded for the world: 'template' (the template's load_room_instances: task rooms + every adjacent room), "
                        "'task' (the template's task rooms only; an experiment), or a comma list of room instances")
    w.add_argument("--rooms-override", default=None, metavar="JSON",
                   help="{alias: {rooms, variant, ...}} - the rooms to load for the world's task alias when listed (campaign rooms, 2026-10-28: the rooms "
                        "the demos can see; docs/ROOMS_OVERRIDE_HOWTO.md); an alias without an entry keeps --load-rooms. ISG_ROOMS_OVERRIDE=<file> does the "
                        "same by template variant for both evaluators")
    w.add_argument("--seg-visibility", action="store_true",
                   help="instance segmentation on every robot camera + the room-visibility probe (rooms_visibility.json per episode); "
                        "pair with --render-override /rtx/raytracing/showLights=0 (the segmentation node's crash guard)")
    w.add_argument("--seg-every", type=int, default=1, help="sample the segmentation every k-th step")
    w.add_argument("--dump-frames-spec", default=None, metavar="SPEC",
                   help="lossless PNGs of the robot cameras at these steps into <episode>/frames/ (campaign rooms' dumper): '0-60:5,25' = steps 0..60 every 5 and every 25th")
    w.add_argument("--image-size", type=int, default=None, help="(not used with the BEHAVIOR layout; kept for the single-env profile)")
    v = p.add_argument_group("videos")
    v.add_argument("--record-view", default="tiled", help="comma list of tiled,head,external,left_wrist,right_wrist per slot; '' for none")
    v.add_argument("--tile-px", type=int, default=240, help="tile size of the per-slot tiled video")
    v.add_argument("--external-camera", action=argparse.BooleanOptionalAction, default=None,
                   help="BEHAVIOR's over-the-shoulder camera on every clone (tiled / external views; one render product per clone). Default: off in --mode policy "
                        "(adopted 2026-10-29, lever noext: the videos lose their 4th tile), on in --mode replay")
    v.add_argument("--external-size", type=int, default=320, help="external camera height in px (width 1.5 x)")
    v.add_argument("--grid-video", action=argparse.BooleanOptionalAction, default=True, help="one grid video of every slot per batch")
    v.add_argument("--grid-width", type=int, default=1920)
    v.add_argument("--video-crf", type=int, default=23)
    v.add_argument("--video-every", type=int, default=1, help="record every k-th step in the videos (1 = every control step)")
    pol = p.add_argument_group("policy")
    pol.add_argument("--policy-server", default="ws://127.0.0.1:8000")
    pol.add_argument("--policy-protocol", default="auto", choices=["auto", "act", "openpi"])
    pol.add_argument("--action-chunk", type=int, default=None, help="openpi: actions executed per request (default: the server's action_horizon)")
    pol.add_argument("--task-id", type=int, default=None, help="task id for every instance (else per spec / --task-ids)")
    pol.add_argument("--goal-views", default="zed_link", help="comma list of the goal views the policy takes when the server does not say")
    pol.add_argument("--policy-timeout", type=float, default=300.0)
    pol.add_argument("--client-image-size", type=int, nargs=2, default=(224, 224), metavar=("H", "W"),
                     help="ACT: frames and goals resized + quantized on the client with the training-time path (ws_policy.resize_quantize) and sent at this "
                          "size (the checkpoint's adapter image_size; 224 224 for the ISG runs); 0 0 sends the native frames (openpi always does)")
    pol.add_argument("--policy-tag", default=None, help="recorded label of the policy (run / checkpoint)")
    pol.add_argument("--policy-sweep", default=None,
                     help="JSON list of policies run one after the other on the same world, each {tag, launch: [argv with {port} / {ckpt}], ckpt, cwd, env, port, "
                          "server}: the driver starts the server (waits for /healthz), runs every batch into <out>/<tag>/, stops it (checkpoints: latest first)")
    pol.add_argument("--server-start-timeout", type=float, default=900.0)
    sp = p.add_argument_group("speed levers and experiments (campaign vec_speed, 2026-10-27; every default = the production path)")
    sp.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=False,
                    help="one tiled render product per camera name for all clones (gm.ENABLE_TILED_RENDERING); EXPERIMENT: the tiles share one "
                         "exposure (lab gate 3d) - never for policy results without the camera check (docs/TILED_EXPOSURE_EXPERIMENT.md)")
    sp.add_argument("--tiled-group", default="name", choices=["name", "clone", "external", "name+external", "clone+external"],
                    help="which cameras share tiled render products (overlay ISG_VEC_TILED_GROUP, needs --tiled): 'name' = per camera name "
                         "across clones (upstream; verification / benchmark runs only - the non-primary tiles render 10-25 %% darker), "
                         "'clone' = per clone (2026-10-29, withdrawn: a rotated camera in the same product fails like a translated one), "
                         "'external' = only the N external cameras in one product (2026-10-29: the policy cameras keep their per-camera "
                         "products and frames; the clones' external video tiles carry the post-processing error), '+external' combines")
    sp.add_argument("--tiled-sensors", default="",
                    help="EXPERIMENT (2026-10-29): comma list of substrings; with --tiled-group name only robot cameras whose sensor name contains "
                         "one of them are tiled (e.g. 'zed_link' = head only, 'realsense' = wrists only): a single tiled resolution per session")
    sp.add_argument("--render-override", action="append", default=[], metavar="KEY=VALUE",
                    help="carb render setting applied after the simulator launch, before the world (VecInstanceLoader(render_overrides)); "
                         "EXPERIMENT only (production renders keep OmniGibson's defaults). Repeatable; bool / int / float parsed")
    sp.add_argument("--log-render-settings", action="store_true", help="read and log the exposure / tonemap / AA / DLSS / render-mode carb settings after the launch")
    sp.add_argument("--dump-frames", default=None, metavar="STEPS",
                    help="comma list of steps (and 'last') at which every slot's head / wrist / external frames are saved losslessly as PNG under <slot>/frames/")
    sp.add_argument("--monitors", action=argparse.BooleanOptionalAction, default=None,
                    help="per-slot QA monitors (contact query, drift / stall statistics) on every step; default on (the 10-23 path); --no-monitors = lever "
                         "S3-monitors (the statistics only feed the QA log and the meta's monitors report; the step budget is the executor's)")
    sp.add_argument("--video-compose", default="thread", choices=["cpu", "thread"],
                    help="where the per-slot tile and the batch grid are composed: cpu = on the stepping thread (the 10-23 path), thread = on the video "
                         "sink's thread from the raw frames (lever S3-recording)")
    sp.add_argument("--lean-step", action=argparse.BooleanOptionalAction, default=True,
                    help="ISG_VEC_LEAN_STEP=1 for the overlay's Simulator._non_physics_step: no per-object update_visuals() loop (lever S2-lean)")
    sp.add_argument("--stop-render-finished", action=argparse.BooleanOptionalAction, default=False,
                    help="turn a slot's render products off when its stream ends, back on at the next batch restore (lever S1-finished)")
    sp.add_argument("--render-switch-method", default="hydra-sync", choices=["hydra-sync", "hydra-nosync", "detach"],
                    help="how a slot's cameras are paused (vec_loader.set_render_enabled): hydra-sync = HydraTexture.set_updates_enabled inside "
                         "og.sim.editing_usd() (10-27), hydra-nosync = the same without the Fabric sync, detach = the annotators detached instead")
    sp.add_argument("--stop-render-at", default=None, metavar="STEP:SLOT[,SLOT]",
                    help="EXPERIMENT: switch the given slots' cameras off at batch step STEP without ending their streams (the other slots' "
                         "frames show whether the switch perturbs the renderer), e.g. 200:1,2")
    sp.add_argument("--gpu-obs", action=argparse.BooleanOptionalAction, default=False,
                    help="ISG_VEC_GPU_OBS=1 for the overlay's VisionSensor: RGB annotators on cuda:0 without a host copy (lever S2-gpu-obs)")
    sp.add_argument("--physics-threads", type=int, default=None,
                    help="PhysX CPU worker threads (/physics/numThreads via VecInstanceLoader(physics_overrides)); EXPERIMENT (lever S2-split: two processes "
                         "per GPU must not oversubscribe the cores); default: Isaac Sim's")
    sp.add_argument("--bench-configs", default=None,
                    help="comma list of lever configurations (BENCH_CONFIGS) run one after the other on the same world, each into <out>/<config>/ "
                         "(same batches, same demos: the before/after timings and the bit-identity evidence come from one process)")
    p.add_argument("--tag", default=None)
    p.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    return p


#: runtime lever configurations of --bench-configs (argument overrides applied per configuration; the world is built once)
BENCH_CONFIGS: Dict[str, Dict[str, Any]] = {
    "base": {"monitors": True, "video_compose": "cpu", "lean_step": False, "stop_render_finished": False},  # the 10-23 path
    "base_again": {"monitors": True, "video_compose": "cpu", "lean_step": False, "stop_render_finished": False},  # repeat: timing noise floor, batch-to-batch identity
    "nomon": {"monitors": False},
    "vidthread": {"video_compose": "thread"},
    "vid2": {"video_every": 2},
    "novid": {"record_view": (), "grid_video": False},
    "lean": {"lean_step": True},
    "stoprender": {"stop_render_finished": True},
    "nomon_vidthread": {"monitors": False, "video_compose": "thread"},
    "nomon_vidthread_lean": {"monitors": False, "video_compose": "thread", "lean_step": True},
    "nomon_vidthread_lean_stoprender": {"monitors": False, "video_compose": "thread", "lean_step": True, "stop_render_finished": True},
}


def _parse_override(s: str):
    key, _, raw = s.partition("=")
    if not key or not _:
        raise argparse.ArgumentTypeError(f"--render-override needs KEY=VALUE, got {s!r}")
    v = raw.strip()
    if v.lower() in ("true", "false"):
        return key.strip(), v.lower() == "true"
    try:
        return key.strip(), int(v)
    except ValueError:
        pass
    try:
        return key.strip(), float(v)
    except ValueError:
        return key.strip(), v


# the render-setting keys the experiment logs: ispatialgym.oracle.vec_loader.RENDER_SETTING_KEYS


def parse_args(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    if getattr(args, "sleep_assist", None) is not None:
        os.environ["ISG_SLEEP_ASSIST"] = str(args.sleep_assist)  # read by SlotRecorder (sleep_assist_config); 'off' / '0' disable
    os.environ["ISG_SLEEP_ASSIST_MODE"] = str(getattr(args, "sleep_assist_mode", "sleep"))
    os.environ["ISG_SLEEP_ASSIST_WAKE_M"] = str(getattr(args, "sleep_assist_wake_radius", 0.05))
    os.environ["ISG_SLEEP_ASSIST_MAX_KG"] = str(getattr(args, "sleep_assist_max_mass", 5.0))
    if (args.task is None) == (args.instances is None):
        p.error("exactly one of --task and --instances")
    if args.num_envs < 1:
        p.error("--num-envs >= 1")
    args.record_view = tuple(v.strip() for v in args.record_view.split(",") if v.strip())
    unknown = [v for v in args.record_view if v not in ("tiled", "head", "external", "left_wrist", "right_wrist")]
    if unknown:
        p.error(f"unknown --record-view {unknown}")
    if args.tolerance_scale is None:
        args.tolerance_scale = 3.0 if args.mode == "policy" else 1.0
    if args.stop_on_success is None:
        args.stop_on_success = args.mode == "policy"
    if args.monitors is None:
        args.monitors = bool(args.qa)  # lever S3-monitors adopted 2026-10-29: the per-step QA monitors only with --qa
    if args.external_camera is None:
        args.external_camera = args.mode != "policy"
    args.render_override = dict(_parse_override(s) for s in (args.render_override or []))
    if args.dump_frames:
        parts = [s.strip() for s in str(args.dump_frames).split(",") if s.strip()]
        args.dump_frames = {int(s) for s in parts if s != "last"} | ({"last"} if "last" in parts else set())
    else:
        args.dump_frames = None
    if args.bench_configs:
        names = [s.strip() for s in args.bench_configs.split(",") if s.strip()]
        unknown = [n for n in names if n not in BENCH_CONFIGS]
        if unknown:
            p.error(f"unknown --bench-configs {unknown}; known: {sorted(BENCH_CONFIGS)}")
        args.bench_configs = names
    # the overlay switches read at the simulator launch / every step (the overlay files are no-ops without them)
    os.environ["ISG_VEC_LEAN_STEP"] = "1" if args.lean_step else "0"
    os.environ["ISG_VEC_GPU_OBS"] = "1" if args.gpu_obs else "0"
    return args


# ==========================================================================================
# policy server processes (--policy-sweep)
# ==========================================================================================


class PolicyServerProcess:
    """A policy server started by the driver: ``launch`` argv with ``{port}`` / ``{ckpt}`` placeholders, its log under ``log_dir``,
    ready when ``GET /healthz`` answers."""

    def __init__(self, entry: dict, log_dir: str, start_timeout: float = 900.0):
        import socket

        self.entry = dict(entry)
        self.tag = str(entry.get("tag") or "policy")
        self.port = int(entry.get("port") or 0)
        if not self.port:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                self.port = sock.getsockname()[1]
        self.url = entry.get("server") or f"ws://127.0.0.1:{self.port}"
        self.proc = None
        self.log_path = os.path.join(log_dir, f"server_{self.tag}.log")
        self.start_timeout = float(start_timeout)
        os.makedirs(log_dir, exist_ok=True)

    def start(self) -> str:
        import subprocess
        import urllib.request

        launch = self.entry.get("launch")
        if launch:
            argv = [str(a).format(port=self.port, ckpt=self.entry.get("ckpt", "")) for a in launch]
            env = dict(os.environ)
            if self.entry.get("clear_pythonpath"):
                env.pop("PYTHONPATH", None)
            env.update({k: str(v) for k, v in (self.entry.get("env") or {}).items()})
            log.info("policy server %s: %s", self.tag, " ".join(argv))
            self.proc = subprocess.Popen(argv, cwd=self.entry.get("cwd") or None, env=env, stdout=open(self.log_path, "ab"), stderr=subprocess.STDOUT, start_new_session=True)
        from urllib.parse import urlparse

        u = urlparse(self.url)
        health = f"http://{u.hostname}:{u.port}/healthz"
        t0 = time.time()
        while True:
            try:
                with urllib.request.urlopen(health, timeout=3) as r:
                    if r.status == 200:
                        log.info("policy server %s ready at %s after %.0fs", self.tag, self.url, time.time() - t0)
                        return self.url
            except Exception:  # noqa: BLE001
                pass
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"policy server {self.tag} exited with {self.proc.returncode} (log {self.log_path})")
            if time.time() - t0 > self.start_timeout:
                self.stop()
                raise RuntimeError(f"policy server {self.tag} not ready after {self.start_timeout:.0f}s (log {self.log_path})")
            time.sleep(2.0)

    def stop(self) -> None:
        import signal

        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        self.proc = None


# ==========================================================================================
# main
# ==========================================================================================


def _slot_name(spec: dict, slot: int, rep: int, args) -> str:
    name = spec["name"]
    if args.same_instance_all_slots:
        name += f".slot{slot}"
    if args.repeat > 1:
        name += f".rep{rep}"
    return name


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("ispatialgym").setLevel(args.log_level)
    t_proc = time.time()
    task_ids: Dict[str, int] = {}
    if args.task_ids:
        with open(args.task_ids) as f:
            task_ids = {k: int(v) for k, v in json.load(f).items()}
    specs = specs_from_dataset(args.task, args.instance_indices, args.instances_root, args.goal_dir, task_ids) if args.task else specs_from_file(args.instances)
    if args.task_id is not None:
        for s in specs:
            s["task_id"] = int(args.task_id)
    if not specs:
        print("no instances", flush=True)
        return EXIT_SUCCESS
    # one world: every instance's template must be the first one's (the batch builder groups instances by template)
    from ispatialgym.oracle.vec_loader import template_paths, tro_task_objects, world_payload, worlds_compatible

    worlds: List[dict] = []
    for s in specs:
        s["iid"], s["template_dir"], s["tmeta"], s["activity"], s["did"] = resolve_instance(s["tro"])
        s["template_file"] = os.path.abspath(template_paths(s["template_dir"], s["tmeta"])[0])
        for w in worlds:
            if s["template_file"] == w["file"]:
                w["names"].append(s["name"])
                break
            ok, bad = worlds_compatible(w["payload"], world_payload(s["template_dir"], s["tmeta"], ignore_objects=set(w["task_objects"]) | set(tro_task_objects(s["tro"]))))
            if ok:
                w["names"].append(s["name"])
                break
        else:
            worlds.append({"file": s["template_file"], "payload": world_payload(s["template_dir"], s["tmeta"], ignore_objects=tro_task_objects(s["tro"])),
                           "task_objects": tro_task_objects(s["tro"]), "names": [s["name"]]})
    if len(worlds) > 1:
        sizes = [len(w["names"]) for w in worlds]
        print(f"RUN_ISG_VEC_WORLDS {len(worlds)} sizes={json.dumps(sizes)}", flush=True)
        if args.world_group is None:
            print(f"the instances span {len(worlds)} worlds; one invocation runs one world (--world-group K): "
                  f"{json.dumps({w['file']: w['names'][:3] for w in worlds}, indent=1)}", flush=True)
            return EXIT_WORLDS
        if not 0 <= args.world_group < len(worlds):
            raise SystemExit(f"--world-group {args.world_group}: {len(worlds)} worlds")
        keep = set(worlds[args.world_group]["names"])
        specs = [s for s in specs if s["name"] in keep]
        print(f"RUN_ISG_VEC world group {args.world_group}/{len(worlds)}: {len(specs)} instances", flush=True)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "_batches"), exist_ok=True)
    N = int(args.num_envs)
    # batches
    todo = []
    for s in specs:
        for rep in range(args.repeat):
            if args.same_instance_all_slots:
                names = [_slot_name(s, k, rep, args) for k in range(N)]
                if args.overwrite or not all(os.path.exists(os.path.join(args.out, n, "meta.json")) for n in names):
                    todo.append(("same", s, rep))
            elif args.overwrite or not os.path.exists(os.path.join(args.out, _slot_name(s, 0, rep, args), "meta.json")):
                todo.append(("one", s, rep))
    batches: List[List[Tuple[dict, int]]] = []
    if args.same_instance_all_slots:
        batches = [[(s, rep)] * N for _, s, rep in todo]
    else:
        for rep in range(args.repeat):
            items = [(s, r) for kind, s, r in todo if r == rep]
            for i in range(0, len(items), N):
                batches.append(items[i:i + N])
    if args.limit_batches:
        batches = batches[: args.limit_batches]
    sweep = None
    if args.policy_sweep:
        with open(args.policy_sweep) as f:
            sweep = json.load(f)
        sweep = sweep.get("sweep", sweep) if isinstance(sweep, dict) else sweep
        if args.mode != "policy":
            raise SystemExit("--policy-sweep needs --mode policy")
    print(f"RUN_ISG_VEC mode={args.mode} num_envs={N} instances={len(specs)} batches={len(batches)} sweep={len(sweep) if sweep else None} out={args.out}", flush=True)
    if not batches:
        return EXIT_SUCCESS

    prov = provenance()
    prov.update({"vectorized": True, "num_envs": N, "driver": "ispatialgym.oracle.run_isg_vec"})
    settle_eval = args.settle_eval == "on" or (args.settle_eval == "auto" and args.mode != "policy")
    external_hw = (int(args.external_size), int(round(args.external_size * 1.5)) // 2 * 2) if args.external_camera and ({"tiled", "external"} & set(args.record_view) or args.grid_video) else None
    # -- the world -------------------------------------------------------------------------------------------------
    t0 = time.time()
    os.environ["ISG_VEC_TILED_GROUP"] = args.tiled_group
    if args.scene_offsets:
        os.environ["ISG_SCENE_OFFSETS"] = args.scene_offsets.replace("+", ",").replace("_", ";")  # '+' and '_' spellings survive sbatch --export's comma split
    os.environ["ISG_VEC_TILED_SENSORS"] = args.tiled_sensors
    # room loading (campaign rooms, 2026-10-28): --rooms-override (per alias) > --load-rooms list > --load-rooms task > the template's
    # load_room_instances (which build_env_config replaces through ISG_ROOMS_OVERRIDE, by template variant; the loader reports what the world loads)
    alias0 = specs[0].get("alias")
    load_rooms = resolve_rooms_override(args.rooms_override, alias0)
    rooms_source = f"override {args.rooms_override}" if load_rooms is not None else None
    if load_rooms is None and args.load_rooms not in ("template", "task"):
        load_rooms = [r.strip() for r in args.load_rooms.split(",") if r.strip()]
        rooms_source = "--load-rooms list"
    if load_rooms is None and args.load_rooms == "task":
        tm = resolve_instance(os.path.abspath(specs[0]["tro"]))[2]
        load_rooms = list(tm.get("task_rooms") or tm.get("rooms") or [])
        rooms_source = "--load-rooms task"
        if not load_rooms:
            raise SystemExit("--load-rooms task: the template meta names no task_rooms / rooms")
    loader = VecInstanceLoader(num_envs=N, behavior_output=True, tiled_rendering=bool(args.tiled), settle_physics_steps=args.settle_physics_steps, warm_renders=args.warm_renders,
                               enhanced_determinism=args.enhanced_determinism, fixed_handle_ag=not args.stock_handle_ag, scene_margin=args.scene_margin,
                               drop_depth=True, external_camera_hw=external_hw, layout=args.layout, load_rooms=load_rooms, seg_instance=bool(args.seg_visibility),
                               render_overrides=args.render_override or None, log_render_settings=bool(args.log_render_settings or args.render_override or args.tiled),
                               physics_overrides=_physics_overrides(args))
    loader.build_env(specs[0]["tro"])
    if rooms_source:
        loader.report["load_rooms"]["source"] = rooms_source
    print(f"RUN_ISG_VEC_ROOMS alias={alias0} {json.dumps(loader.report.get('load_rooms'))}", flush=True)
    build_s = time.time() - t0
    env, task, cam_key = loader.env, loader.task, loader.cam_key
    fps = int(loader.cfg["env"]["action_frequency"])
    robot_name = loader.robots[0].name
    print(f"RUN_ISG_VEC world built in {build_s:.1f}s: template={loader.template_dir} scene={loader.tmeta.get('scene')} offsets={loader.report.get('scene_offsets')}", flush=True)
    if loader.report.get("render_settings") is not None:
        print(f"RUN_ISG_VEC render_settings {json.dumps(loader.report['render_settings'], default=str)}", flush=True)
    if loader.report.get("render_overrides"):
        print(f"RUN_ISG_VEC render_overrides {json.dumps(loader.report['render_overrides'], default=str)}", flush=True)
    print(f"RUN_ISG_VEC levers tiled={loader.report.get('tiled_rendering')} gpu_obs={os.environ.get('ISG_VEC_GPU_OBS')} lean_step={os.environ.get('ISG_VEC_LEAN_STEP')} "
          f"monitors={args.monitors} video_compose={args.video_compose} stop_render_finished={args.stop_render_finished} external_camera={external_hw is not None}", flush=True)
    dump_json(os.path.join(args.out, "_batches", "loader_report.json"), {"report": loader.report, "argv": sys.argv[1:], "levers": {
        "tiled": bool(args.tiled), "tiled_group": args.tiled_group, "gpu_obs": bool(args.gpu_obs), "lean_step": bool(args.lean_step), "monitors": bool(args.monitors), "video_compose": args.video_compose,
        "stop_render_finished": bool(args.stop_render_finished), "external_camera": external_hw is not None, "video_every": args.video_every}})
    names = {k: v for k, v in (loader.tmeta.get("eval") or {}).items() if k in ("task_family", "task", "subtask")}
    exit_code = EXIT_SUCCESS
    n_done = 0
    root_out = args.out
    if args.bench_configs:
        # the same batches once per lever configuration on this world (before / after on one host, one build): <out>/<config>/
        import copy as _copy

        for k, cname in enumerate(args.bench_configs):
            cargs = _copy.copy(args)
            for key, value in BENCH_CONFIGS[cname].items():
                setattr(cargs, key, value)
            cargs.bench_config = cname
            os.environ["ISG_VEC_LEAN_STEP"] = "1" if cargs.lean_step else "0"
            out_k = os.path.join(root_out, cname)
            os.makedirs(os.path.join(out_k, "_batches"), exist_ok=True)
            print(f"RUN_ISG_VEC bench {k + 1}/{len(args.bench_configs)} {cname}: {BENCH_CONFIGS[cname]} (ISG_VEC_LEAN_STEP={os.environ['ISG_VEC_LEAN_STEP']})", flush=True)
            code, n = run_batches(cargs, loader, batches, out_k, prov, settle_eval, external_hw, build_s, names, policy_server=args.policy_server,
                                  policy_extra={"bench_config": cname, "bench_overrides": BENCH_CONFIGS[cname]})
            exit_code = max(exit_code, code)
            n_done += n
        print(f"RUN_ISG_VEC_DONE episodes={n_done} batches={len(batches) * len(args.bench_configs)} wall={time.time() - t_proc:.1f}s", flush=True)
        return exit_code
    if sweep:
        for k, entry in enumerate(sweep):
            tag = str(entry.get("tag") or f"policy{k}")
            out_k = os.path.join(root_out, tag)
            os.makedirs(os.path.join(out_k, "_batches"), exist_ok=True)
            # skip a sweep entry whose every slot episode exists (resume)
            pending = [b_ for b_ in batches if any(args.overwrite or not os.path.exists(os.path.join(out_k, _slot_name(sp, i, rep, args), "meta.json"))
                                                   for i, (sp, rep) in enumerate(b_))]
            if not pending:
                print(f"RUN_ISG_VEC sweep {k + 1}/{len(sweep)} {tag}: done already", flush=True)
                continue
            server = PolicyServerProcess(entry, os.path.join(root_out, "_servers"), start_timeout=args.server_start_timeout)
            t_sw = time.time()
            try:
                url = server.start()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                print(f"RUN_ISG_VEC sweep {k + 1}/{len(sweep)} {tag}: server failed: {e!r}", flush=True)
                exit_code = EXIT_ERROR
                continue
            try:
                code, n = run_batches(args, loader, pending, out_k, prov, settle_eval, external_hw, build_s, names, policy_server=url, policy_tag=tag,
                                      policy_extra={"checkpoint": entry.get("ckpt"), "sweep_entry": {kk: v for kk, v in entry.items() if kk != "env"}})
            finally:
                server.stop()
            n_done += n
            exit_code = max(exit_code, code)
            print(f"RUN_ISG_VEC sweep {k + 1}/{len(sweep)} {tag}: {n} episodes in {time.time() - t_sw:.0f}s", flush=True)
    else:
        exit_code, n_done = run_batches(args, loader, batches, root_out, prov, settle_eval, external_hw, build_s, names, policy_server=args.policy_server,
                                        policy_tag=args.policy_tag, policy_extra={})
    print(f"RUN_ISG_VEC_DONE episodes={n_done} batches={len(batches)} wall={time.time() - t_proc:.1f}s", flush=True)
    return exit_code


def _physics_overrides(args) -> Optional[Dict[str, Any]]:
    """``--physics-threads`` + ``--physics-override KEY=VALUE`` into VecInstanceLoader(physics_overrides); None when nothing is set."""
    ov: Dict[str, Any] = {}
    if getattr(args, "physics_threads", None):
        ov["num_threads"] = int(args.physics_threads)
    if getattr(args, "rigid_sleep_threshold", 0.0):
        ov["rigid_sleep_threshold"] = float(args.rigid_sleep_threshold)  # an explicit --physics-override rigid_sleep_threshold=... below wins
    if getattr(args, "rigid_solver_iterations", None):
        pos_it, _, vel_it = str(args.rigid_solver_iterations).replace("+", ",").replace(":", ",").partition(",")  # 64+8 for sbatch --export
        if pos_it.strip():
            ov["rigid_solver_position_iterations"] = int(pos_it)
        if vel_it.strip():
            ov["rigid_solver_velocity_iterations"] = int(vel_it)
    for item in getattr(args, "physics_override", None) or []:
        key, _, val = item.partition("=")
        key = key.strip()
        v: Any = val.strip()
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        if key in ("gpu_dynamics", "enhanced_determinism") and isinstance(v, int):
            v = bool(v)
        ov[key] = v
    return ov or None


def run_batches(args, loader, batches, out_root, prov, settle_eval, external_hw, build_s, names, policy_server=None, policy_tag=None, policy_extra=None) -> Tuple[int, int]:
    """Every batch of ``batches`` on the loaded world into ``out_root``; returns (exit code, slot episodes written)."""
    N = int(args.num_envs)
    env, task, cam_key = loader.env, loader.task, loader.cam_key
    fps = int(loader.cfg["env"]["action_frequency"])
    robot_name = loader.robots[0].name
    os.makedirs(os.path.join(out_root, "_batches"), exist_ok=True)
    results_path = os.path.join(out_root, "results.jsonl")
    exit_code = EXIT_SUCCESS
    n_done = 0
    policy_meta: Optional[dict] = None
    policy_extra = policy_extra or {}

    for b, batch in enumerate(batches):
        t_batch = time.time()
        placement: Dict[int, Tuple[dict, int]] = {k: item for k, item in enumerate(batch)}
        tro_paths = [placement[i][0]["tro"] if i in placement else None for i in range(N)]
        seed0 = int(placement[0][0].get("episode_index") or placement[0][0].get("dataset_index") or 0)
        th.manual_seed(seed0)
        np.random.seed(seed0 % (2 ** 32))
        random.seed(seed0)
        # budgets
        budgets: Dict[int, dict] = {}
        budget_errors: Dict[int, str] = {}
        for i, (spec, rep) in placement.items():
            try:
                demo_settle = spec.get("settle_steps")
                if args.mode == "replay" and spec.get("demo") and (spec.get("action_steps") is None or demo_settle is None):
                    n_action, demo_settle = replay_counts(spec["demo"], 0, {"action_steps": spec.get("action_steps"), "settle_steps": demo_settle})
                    spec["action_steps"] = n_action
                    spec["settle_steps"] = demo_settle
                budgets[i] = INST.step_budget(spec.get("length"), args.max_steps, args.settle_steps, args.wallclock or None, mode=args.mode,
                                              demo_settle_steps=demo_settle if args.mode == "replay" else None, instance_max_steps=spec.get("instance_max_steps"))
                if not args.wallclock:
                    budgets[i]["wallclock_s"] = 0  # no wall-clock cut (ExecutorConfig.max_wall_s 0 = disabled); the worker's stall watchdog guards a hung process
            except Exception as e:  # noqa: BLE001
                budget_errors[i] = repr(e)
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            obs_by, info_by = loader.restore_batch(tro_paths, settle_eval=settle_eval, settle_steps=[budgets.get(i, {}).get("settle_steps") for i in range(N)], mode=args.mode,
                                                   tolerance_scale=args.tolerance_scale)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            for i, (spec, rep) in placement.items():
                out_dir = os.path.join(out_root, _slot_name(spec, i, rep, args))
                os.makedirs(out_dir, exist_ok=True)
                meta = {"status": "error", "error": f"restore_batch: {e!r}", "traceback": traceback.format_exc(), "instance_id": spec["name"], "slot": i, "batch": b, "num_envs": N,
                        "provenance": prov, "started_utc": started}
                dump_json(os.path.join(out_dir, "meta.json"), meta)
            exit_code = EXIT_ERROR
            continue
        load_s = time.time() - t_batch
        loaded = sorted(obs_by)
        tex_status = None
        if loaded and args.texture_wait_s > 0:
            tex_status, obs_list_fresh, info_list_fresh = wait_for_textures_batch(env, loaded, robot_name, cam_key, args.texture_wait_s)
            for k, i in enumerate(loaded):
                obs_by[i], info_by[i] = obs_list_fresh[k], {"obs_info": info_list_fresh[k]}
        # per slot setup
        sink = VideoSink(fps=max(1, fps // max(1, args.video_every)), crf=args.video_crf) if (args.record_view or args.grid_video) else None
        rec_timers: Dict[str, float] = {"record_frames_s": 0.0, "record_tile_s": 0.0, "record_rows_s": 0.0, "grid_s": 0.0}  # the recorder's share of record_s / post_step_s
        slots: List[Slot] = []
        per: Dict[int, dict] = {}
        policy_client = None
        policy_step = None
        if args.mode == "policy":
            from ispatialgym.eval.ws_policy import BatchedWSPolicy, load_rgb

            goals: Dict[int, Dict[str, np.ndarray]] = {}
            tids: Dict[int, int] = {}
            for i in loaded:
                spec = placement[i][0]
                if i in loader.failed or i in budget_errors:
                    continue
                if spec.get("task_id") is None:
                    budget_errors[i] = "no task_id for the policy (spec task_id / --task-ids / --task-id)"
                    continue
                if not spec.get("goal_png"):
                    budget_errors[i] = "no goal image (spec goal_png / --goal-dir)"
                    continue
                try:
                    goals[i] = {"zed_link": load_rgb(spec["goal_png"])}
                except Exception as e:  # noqa: BLE001
                    budget_errors[i] = f"goal image {spec['goal_png']}: {e!r}"
                    continue
                tids[i] = int(spec["task_id"])
            if tids:
                try:
                    cis = tuple(args.client_image_size) if args.client_image_size and all(v > 0 for v in args.client_image_size) else None
                    policy_client = BatchedWSPolicy(policy_server, sorted(tids), tids, goals, robot=robot_name, protocol=args.policy_protocol, action_chunk=args.action_chunk,
                                                    timeout=args.policy_timeout, client_image_size=cis)
                    policy_client.default_goal_views = [v.strip() for v in args.goal_views.split(",") if v.strip()]
                    policy_meta = policy_client.connect()
                    policy_step = BatchedPolicyStep(policy_client)
                    print(f"RUN_ISG_VEC policy server {policy_server}: protocol={policy_client.protocol} slots={sorted(tids)} meta_keys={sorted(policy_meta)[:12]}", flush=True)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    for i in tids:
                        budget_errors[i] = f"policy server: {e!r}"
                    policy_client = None
        for i in range(N):
            if i not in placement:
                continue
            spec, rep = placement[i]
            out_dir = os.path.join(out_root, _slot_name(spec, i, rep, args))
            os.makedirs(out_dir, exist_ok=True)
            d = {"spec": spec, "rep": rep, "out_dir": out_dir, "t0": time.time(), "budget": budgets.get(i)}
            per[i] = d
            meta: Dict[str, Any] = {"status": "error", "mode": args.mode, "argv": sys.argv[1:], "hostname": os.uname().nodename, "started_utc": started, "provenance": prov,
                                    "instance_id": spec["name"], "instance_name": spec["name"], "tro_path": spec["tro"], "template_dir": loader.template_dir,
                                    "activity": loader.activity, "activity_definition_id": loader.did, "scene": loader.tmeta.get("scene"), "robot_name": robot_name,
                                    "out_dir": out_dir, "budget": budgets.get(i), "settle_eval": settle_eval, "settle_steps": (budgets.get(i) or {}).get("settle_steps"),
                                    "success_metric": args.success_metric, "tolerance_scale": args.tolerance_scale, "stop_on_success": bool(args.stop_on_success),
                                    "vec": {"num_envs": N, "slot": i, "batch": b, "rep": rep, "same_instance_all_slots": bool(args.same_instance_all_slots),
                                            "scene_offset": (loader.report.get("scene_offsets") or [None] * N)[i], "world_build_s": round(build_s, 1), "reused_world": b > 0},
                                    "texture_wait": tex_status, "tag": args.tag, "policy_tag": policy_tag,
                                    **{k: spec.get(k) for k in ("task_family", "task", "subtask", "alias", "episode_index", "task_id", "goal_png", "demo", "length")},
                                    **{k: v for k, v in names.items() if not spec.get(k)}}
            d["meta"] = meta
            if i in loader.failed:
                fail = loader.failed[i]
                meta.update({"status": "oracle_failed", "reason": fail["reason"], "load_error": fail["message"], "loader": loader.slot_reports[i]})
                continue
            if i in budget_errors:
                meta.update({"status": "error", "error": budget_errors[i], "loader": loader.slot_reports[i]})
                continue
            if i not in obs_by:
                meta.update({"status": "error", "error": "no observation after the restore", "loader": loader.slot_reports[i]})
                continue
            budget = budgets[i]
            robot, scene = loader.robots[i], env.scenes[i]
            try:
                obs0, info0 = obs_by[i], info_by[i]
                meta["loader"] = {k: v for k, v in loader.slot_reports[i].items() if k != "trav_cells_blocked"}
                meta["tolerance_profile"] = loader.slot_reports[i].get("tolerance_profile")
                meta["robot_base_init"] = loader.slot_reports[i].get("robot_base_init_scene")
                try:
                    save_png(os.path.join(out_dir, "init_rgb.png"), obs0[robot_name][cam_key]["rgb"])
                except Exception as ex:  # noqa: BLE001
                    log.warning("init render not saved: %r", ex)
                tracked = loader.tracked(i)
                meta["tracked_objects"] = {k: v.name for k, v in tracked.items()}
                if args.mode == "replay":
                    n_action, n_settle = int(spec["action_steps"]), int(budget["settle_steps"])
                    source = ReplaySource(spec["demo"], 0, n_action, n_settle)
                    phases = source.phases
                    meta["replay"] = spec["demo"]
                    meta["replay_rows"] = {"action": n_action, "settle": n_settle, "demo_settle": spec.get("settle_steps")}
                else:
                    source = PolicySource(policy_step, i)
                    phases = None
                    meta["policy"] = {"server": policy_server, "protocol": policy_client.protocol if policy_client else None, "metadata": policy_meta,
                                      "task_id": spec.get("task_id"), "goal_png": spec.get("goal_png"), "tag": policy_tag, **policy_extra,
                                      "goal_sha256": (policy_client.goal_sha256.get(i) if policy_client else None), "action_chunk": policy_client.action_chunk if policy_client else None}
                    try:
                        import shutil

                        shutil.copyfile(spec["goal_png"], os.path.join(out_dir, "goal_rgb.png"))
                    except Exception:  # noqa: BLE001
                        pass
                cam_in_zed = C.camera_offset_in_zed(robot, cam_key)
                meta["camera_in_zed"] = {"position": cam_in_zed[0].tolist(), "orientation": cam_in_zed[1].tolist()}
                view, tview = loader.view(i), loader.task_view(i)
                state = SimState(view, tview, robot, cam_key, cam_in_zed=cam_in_zed)
                qa_cfg = QAConfig(max_episode_steps=budget["episode_cap"], max_action_steps=budget["episode_cap"])
                # lever S3-monitors (2026-10-27): without --qa / --monitors no Monitors object - its per-step contact query, drift and stall
                # rules only feed the QA log and the meta's `monitors` report; the step budget is the executor's (vec_stream: s.n >= max_steps,
                # reached before the monitor's own timeout rule could fire), so the verdicts are unchanged
                monitors = Monitors(view, robot, qa_cfg, cam_key=cam_key, cam_in_zed=cam_in_zed) if (args.qa or args.monitors) else None
                recorder = SlotRecorder(out_dir, robot, cam_key, tracked, scene, sink, args.record_view, args.tile_px, 2 * args.tile_px if external_hw else None,
                                        name=f"s{i}", external_camera=EXTERNAL_CAMERA_NAME if external_hw else None, compose=args.video_compose,
                                        dump_steps=args.dump_frames, timers=rec_timers, video_every=args.video_every,
                                        dump_spec=parse_dump_spec(args.dump_frames_spec), seg_visibility=bool(args.seg_visibility), seg_every=args.seg_every)
                meta["load_rooms"] = loader.report.get("load_rooms")
                recorder.start(obs0, info0)
                cfg = ExecutorConfig(max_steps=budget["episode_cap"], max_wall_s=budget["wallclock_s"], settle_steps=budget["settle_steps"], success_metric=args.success_metric,
                                     stop_on_success=bool(args.stop_on_success), goal_image=False, record_state=False, qa=bool(args.qa))
                slots.append(Slot(idx=i, robot=robot, source=source, cfg=cfg, task=tview, monitors=monitors, recorder=recorder, tracked=tracked, held=state.held_objects,
                                  phases=phases))
                d.update({"source": source, "recorder": recorder, "monitors": monitors})
            except Exception as ex:  # noqa: BLE001
                meta["error"] = repr(ex)
                meta["traceback"] = traceback.format_exc()
                traceback.print_exc()
        # grid video of the batch
        grid_name = f"batch_{b:04d}"
        grid_path = os.path.join(out_root, "_batches", grid_name + "_grid.mp4")
        if sink is not None and args.grid_video and slots:
            sink.open("grid", grid_path)

        t_progress = [time.time()]
        _switch_step = [int(args.stop_render_at.split(":")[0])] if args.stop_render_at else [-1]
        _switch_slots = [int(x) for x in args.stop_render_at.split(":")[1].split(",")] if args.stop_render_at else []

        def post_step(step: int, stepped: List[Slot], _slots=slots, _sink=sink, _per=per) -> None:
            if time.time() - t_progress[0] > 60:
                t_progress[0] = time.time()
                print(f"RUN_ISG_VEC_PROGRESS batch={b} step={step} streaming={len(stepped)} done={sum(1 for s_ in _slots if s_.done)}/{len(_slots)} "
                      f"elapsed={time.time() - t_run:.0f}s pending_video={_sink.pending if _sink is not None else 0}", flush=True)
            if args.stop_render_at and step == _switch_step[0]:
                for i_ in _switch_slots:
                    n_ = loader.set_render_enabled(i_, False)
                    print(f"RUN_ISG_VEC_SWITCH step={step} slot={i_} products_off={n_} method={loader.render_switch_method}", flush=True)
            if _sink is None or not args.grid_video or (step % max(1, args.video_every)):
                return
            t_g = time.time()
            if args.video_compose == "thread":
                # lever S3-recording: the sink's thread composes the grid from the slots' last tiles (which it also composes); the
                # stepping thread only hands over the done-labels
                done_labels = {s.idx: f"done: {'success' if s.result.get('success') else (s.result.get('reason') or 'fail')}" for s in _slots if s.done}
                _sink.put_grid("grid", [f"s{s.idx}" for s in _slots], done_labels, max_width=args.grid_width)
            else:
                tiles = []
                for s in _slots:
                    rec = _per[s.idx]["recorder"]
                    t = rec.last_tile
                    if t is not None and s.done:
                        t = t.copy()
                        label(t, f"done: {'success' if s.result.get('success') else (s.result.get('reason') or 'fail')}", org=(6, t.shape[0] - 10))
                    tiles.append(t)
                _sink.put("grid", grid_frame(tiles, max_width=args.grid_width))
            rec_timers["grid_s"] += time.time() - t_g

        # lever S1-finished (2026-10-27): a slot whose stream ended stops rendering its cameras until the next batch restore
        loader.render_switch_method = args.render_switch_method
        on_stream_end = (lambda s: loader.set_render_enabled(s.idx, False)) if args.stop_render_finished else None

        idle = {i: loader.hold_action(i) for i in range(N) if i not in {s.idx for s in slots}}
        obs_list = [obs_by.get(i) for i in range(N)]
        info_list = [info_by.get(i) for i in range(N)]
        t_run = time.time()
        t_progress[0] = t_run
        results: Dict[int, dict] = {}
        timings_loop: Dict[str, float] = {}
        batch_steps = 0
        if slots:
            timeline = []
            try:
                vex = VecStreamExecutor(env, slots, sim=og.sim, n_render_iterations=1, num_envs=N, idle_actions=idle, qa=bool(args.qa), pre_step=policy_step, post_step=post_step,
                                        on_stream_end=on_stream_end, stall_timeout_s=args.stall_timeout)
                for s, r in zip(slots, vex.run(obs_list, info_list)):
                    results[s.idx] = r
                timings_loop = dict(vex.timings)
                timings_loop.update({k: v for k, v in rec_timers.items()})
                batch_steps = vex.steps
                timeline = list(vex.timeline)  # per batch step: (env_step s, slots still streaming)
            except Exception as ex:  # noqa: BLE001
                traceback.print_exc()
                for s in slots:
                    per[s.idx]["meta"]["error"] = per[s.idx]["meta"].get("error") or f"vec run: {ex!r}"
                    per[s.idx]["meta"]["traceback"] = traceback.format_exc()
                exit_code = EXIT_ERROR
        episode_s = time.time() - t_run
        if policy_client is not None:
            policy_client.close()
        # -- per slot: finalize -------------------------------------------------------------------------------------
        video_info = sink.close() if sink is not None else {}
        for i in sorted(per):
            d = per[i]
            spec, rep, out_dir, meta = d["spec"], d["rep"], d["out_dir"], d["meta"]
            episode = results.get(i) or {}
            try:
                rec = d.get("recorder")
                if rec is not None:
                    if rec.last_rgb is not None:
                        save_png(os.path.join(out_dir, "final_rgb.png"), rec.last_rgb)
                    paths = rec.close()
                    meta["videos"] = {v: p for v, p in (paths.get("videos") or {}).items()}
                    meta["video_errors"] = {k: v for k, v in (video_info.get("errors") or {}).items() if k.startswith(f"s{i}:")} or None
                    meta["trajectory"] = paths.get("trajectory")
                    for k in ("rooms_visibility", "rooms_seen", "frames_dumped", "sleep_assist"):
                        if k in paths:
                            meta[k] = paths[k]
                if episode:
                    meta.update({k: v for k, v in episode.items() if k not in ("goal_observation", "timings")})
                    meta["status"] = "success" if episode["success"] else "oracle_failed"
                    meta["qa_reason"] = episode.get("reason")
                    src = d.get("source")
                    meta["stream"] = {"streamed": getattr(src, "streamed", None), "stop_reason": getattr(src, "stop_reason", None) or
                                      ("success" if (args.stop_on_success and episode.get("success")) else ("budget" if episode.get("n_steps", 0) >= (d["budget"] or {}).get("episode_cap", 10**9) else None))}
                    meta["timings"] = {"load_s": round(load_s, 1), "episode_s": round(episode_s, 1), "batch_steps": batch_steps, "loop": {k: round(v, 2) for k, v in timings_loop.items()},
                                       **{f"executor_{k}": v for k, v in episode.get("timings", {}).items()}, "total_s": round(time.time() - d["t0"], 1)}
                    if policy_client is not None:
                        meta.setdefault("policy", {})["stats"] = policy_client.stats()
                elif meta.get("status") == "error" and not meta.get("error"):
                    meta["error"] = "no episode result"
                meta["wall_time_s"] = round(time.time() - d["t0"], 1)
                meta["grid_video"] = grid_path if (sink is not None and args.grid_video) else None
            except Exception as ex:  # noqa: BLE001
                meta["error"] = meta.get("error") or repr(ex)
                meta["traceback"] = traceback.format_exc()
                traceback.print_exc()
            dump_json(os.path.join(out_dir, "meta.json"), meta)
            success = bool(meta.get("success")) if meta.get("status") in ("success", "oracle_failed") else False
            result = {"instance": spec["name"], "status": meta.get("status"), "success": success, "reason": meta.get("reason") or meta.get("error"),
                      "success_metric": args.success_metric, "settle_eval": settle_eval, "settle_steps": (d["budget"] or {}).get("settle_steps"),
                      **{k: episode.get(k) for k in ("symbolic_success", "symbolic_success_step", "symbolic_first_step", "geometric_success", "geometric_success_step",
                                                     "geometric_first_step", "evaluation_step", "evaluation_kind", "evaluation_settled", "success_final",
                                                     "error_metric", "closest_approach", "closest_goal_approach", "errors", "errors_final_step", "measured",
                                                     "other_mode")},
                      "n_steps": episode.get("n_steps"), "n_action_steps": episode.get("n_action_steps"), "max_steps": (d["budget"] or {}).get("max_steps"),
                      "wall_clock_exceeded": episode.get("wall_clock_exceeded"), "wall_time_s": meta.get("wall_time_s"), "out": out_dir, "slot": i, "batch": b, "rep": rep,
                      "num_envs": N, "mode": args.mode, "task_family": meta.get("task_family"), "task": meta.get("task"), "subtask": meta.get("subtask"), "alias": meta.get("alias"),
                      "episode_index": spec.get("episode_index"), "task_id": spec.get("task_id"), "policy_tag": policy_tag, "tag": args.tag,
                      "exit_code": EXIT_SUCCESS if success else (EXIT_ORACLE_FAILED if meta.get("status") == "oracle_failed" else EXIT_ERROR),
                      "host": os.uname().nodename, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with open(results_path, "a") as f:
                f.write(json.dumps(result, default=json_default) + "\n")
            print("RUN_ISG_RESULT " + json.dumps(result, default=json_default), flush=True)
            n_done += 1
            if meta.get("status") == "error":
                exit_code = EXIT_ERROR
        batch_rec = {"batch": b, "slots": {i: per[i]["spec"]["name"] for i in per}, "load_s": round(load_s, 1), "episode_s": round(episode_s, 1), "batch_steps": batch_steps,
                     "loop": {k: round(v, 2) for k, v in timings_loop.items()}, "texture_wait": tex_status, "grid_video": grid_path if (sink is not None and args.grid_video) else None,
                     "env_steps_per_s": round(batch_steps * len(slots) / episode_s, 2) if episode_s > 0 else None,
                     "timeline": timeline,  # [(env_step s, streaming slots)] per batch step (lever S1-finished: does the step get cheaper as slots finish?)
                     "batch_steps_per_s": round(batch_steps / episode_s, 2) if episode_s > 0 else None,
                     "ms_per_batch_step": {k: round(1000.0 * v / batch_steps, 1) for k, v in timings_loop.items()} if batch_steps else None,
                     "levers": {"bench_config": getattr(args, "bench_config", None), "monitors": bool(args.qa or args.monitors), "video_compose": args.video_compose,
                                "video_every": args.video_every, "record_view": list(args.record_view), "grid_video": bool(args.grid_video),
                                "lean_step": os.environ.get("ISG_VEC_LEAN_STEP") == "1", "stop_render_finished": bool(args.stop_render_finished),
                                "gpu_obs": os.environ.get("ISG_VEC_GPU_OBS") == "1", "tiled": bool(loader.report.get("tiled_rendering")),
                                "external_camera": external_hw is not None, "num_envs": N, "gpu": th.cuda.get_device_name(0) if th.cuda.is_available() else None},
                     "video_pending_at_end": video_info.get("max_pending") if isinstance(video_info, dict) else None}
        if os.environ.get("ISG_VEC_GPU_OBS_CHECK") == "1":
            try:
                from omnigibson.sensors import vision_sensor as _vs

                batch_rec["gpu_obs_check"] = dict(getattr(_vs, "ISG_GPU_OBS_CHECK", {}))
            except Exception as ex:  # noqa: BLE001
                batch_rec["gpu_obs_check"] = {"error": repr(ex)}
        dump_json(os.path.join(out_root, "_batches", grid_name + ".json"), batch_rec)
        print(f"RUN_ISG_VEC_BATCH {b + 1}/{len(batches)} slots={len(slots)} load={load_s:.1f}s run={episode_s:.1f}s batch_steps={batch_steps} "
              f"batch_steps/s={batch_rec['batch_steps_per_s']} env_steps/s={batch_rec['env_steps_per_s']} loop={batch_rec['loop']}", flush=True)
        try:
            import gc

            gc.collect()
            th.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return exit_code, n_done


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # no og.shutdown(): Kit's teardown segfaults after a short run (SIGSEGV after RUN_ISG_VEC_DONE, 2026-10-24: a two-world
    # unit's first group lost its exit code and the second group never ran); the OS reclaims the process, the per-unit app
    # data is deleted by the worker anyway
    os._exit(code)
