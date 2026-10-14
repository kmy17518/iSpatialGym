"""Kinematic replayer: BEHAVIOR LeRobot v3 shards from physics-only ``demo.hdf5`` files (``ISPATIALGYM_PHYSICS_ONLY=1``
collection, :mod:`ispatialgym.oracle.physics_only`).

    python -m ispatialgym.multienv.replay --demos <demo dir>[,<demo dir>...] --root <lerobot root> --task <variant>
        [--batched-usd-edits] [--slots 1] [--tiled] [--repo-prefix replay] [--async-flush] [--first-renders 10]
        [--renders 1] [--warmup-s 60] [--proprio recorded|loaded] [--results <jsonl>]
    python -m ispatialgym.multienv.replay --watch <demo root> --done-file <path> ...   (a pipeline stage: see below)

One process per variant, ``--slots`` scenes of the variant's scene in one simulator, each slot playing a different demo (a
slot takes the next demo of the list when its demo ends). The demos must share a template (the same scene objects), as
production's one-world replay (``replay_obs_isg.py --one-world``). Per tick every slot loads its demo's next recorded
state into its scene (no physics step), the cameras render once, and each slot's cameras feed that slot's BEHAVIOR
:class:`~ispatialgym.oracle.behavior_lerobot.ShardWriter`: one LeRobot v3 dataset per slot, ``<root>/<repo
prefix>_slot<k>``, BEHAVIOR's encoders, background flush with ``--async-flush``. Measured (dishwasher, 12 CPUs): one slot
with ``--batched-usd-edits`` replays ~2x faster than plain loads with the same images; several slots in one process
were not faster than one (a state load costs more per slot in a bigger stage), and ``--tiled`` changes the rgb images
(see its help), so a GPU is best served by several one-slot processes.

The environment is production's aligned-replay one (``DataPlaybackWrapper.create_from_hdf5`` with replay_obs's settings:
the recorded env config and scene file, visual-only objects, replay_obs's sensors, modalities and proprioception keys)
with ``num_envs`` = slots and ``DummyTask`` (``ISpatialGymTask`` binds one environment; it adds nothing visible).

Frames pair as production's ``AlignedLeRobotPlaybackWrapper`` and live collection: frame i = the observation of state i
with action i (reward / terminated / truncated i). ``observation.state`` is the collection-time proprioception
(``isg/behavior_proprio``, recorded by physics-only collection); ``--proprio loaded`` takes the robot's proprioception
of the loaded state instead. Renders per tick: ``--first-renders`` when a slot has just loaded the first state of a demo
(production's replay renders 10 times there), else ``--renders`` (production's ``n_render_iterations`` = 1).

``--watch``: the demos are the successful runs appearing under ``<demo root>`` (``meta.json`` with status success, written
by run_isg after demo.hdf5), taken in the order they finish; the environment is built from the first one, idle slots
take new demos as they appear, and the process ends when ``--done-file`` exists and every finished demo is replayed.

Prints ``REPLAY_DEMO {json}`` per finished demo and ``REPLAY_SUMMARY {json}`` (env-steps/s, per-tick time split, CPU,
GPU memory) at the end; ``--results`` appends the same lines as json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch as th

BEHAVIOR_PROPRIO_KEY = "isg/behavior_proprio"


class Demo:
    """The recorded arrays of one demo's committed episode (``data/demo_0``)."""

    def __init__(self, demo_dir: str):
        self.dir = demo_dir
        self.name = os.path.basename(os.path.normpath(demo_dir))
        with h5py.File(os.path.join(demo_dir, "demo.hdf5"), "r") as f:
            g = f["data/demo_0"]
            self.scene_file = json.loads(f["data"].attrs["scene_file"])
            self.state = g["state"][()]
            self.state_size = g["state_size"][()]
            self.action = th.as_tensor(g["action"][()], dtype=th.float32)
            self.reward = g["reward"][()]
            self.terminated = g["terminated"][()]
            self.truncated = g["truncated"][()]
            self.proprio = g[BEHAVIOR_PROPRIO_KEY][()] if BEHAVIOR_PROPRIO_KEY in g else None
        self.T = int(self.action.shape[0])
        if self.state.shape[0] != self.T:
            raise ValueError(f"{demo_dir}: {self.state.shape[0]} states for {self.T} actions")

    def state_at(self, i: int) -> th.Tensor:
        return th.as_tensor(self.state[i, : int(self.state_size[i])], dtype=th.float32)


class DemoSource:
    """The demos to replay: a fixed list, or (``watch``) the successful runs appearing under a demo root until
    ``done_file`` exists."""

    def __init__(self, demo_dirs: List[str], watch: Optional[str] = None, done_file: Optional[str] = None, poll_s: float = 5.0,
                 claims: Optional[str] = None):
        self.pending: List[str] = list(demo_dirs)
        self.watch, self.done_file, self.poll_s = watch, done_file, float(poll_s)
        self.claims = claims
        if claims:
            os.makedirs(claims, exist_ok=True)
        self.seen = set(demo_dirs)
        self._last_scan = 0.0

    def _claim(self, demo_dir: str) -> bool:
        """``claims``: several replayers watching one demo root take each demo once (an exclusive create)."""
        if not self.claims:
            return True
        try:
            os.close(os.open(os.path.join(self.claims, os.path.basename(os.path.normpath(demo_dir))), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            return False

    def _scan(self, force: bool = False) -> None:
        if self.watch is None or (not force and time.time() - self._last_scan < self.poll_s):
            return
        self._last_scan = time.time()
        found = []
        for name in os.listdir(self.watch) if os.path.isdir(self.watch) else []:
            d = os.path.join(self.watch, name)
            meta = os.path.join(d, "meta.json")
            if d in self.seen or not os.path.exists(meta) or not os.path.exists(os.path.join(d, "demo.hdf5")):
                continue
            self.seen.add(d)
            try:
                with open(meta) as f:
                    ok = json.load(f).get("status") == "success"
            except (OSError, ValueError):
                self.seen.discard(d)  # meta.json being written: next scan
                continue
            if ok:
                found.append((os.path.getmtime(meta), d))
        self.pending += [d for _, d in sorted(found)]

    def next(self, force: bool = False) -> Optional[Demo]:
        self._scan(force)
        while self.pending:
            d = self.pending.pop(0)
            if self._claim(d):
                return Demo(d)
        return None

    def finished(self) -> bool:
        """No demo left and none to come."""
        if self.watch is None:
            return not self.pending
        if not (self.done_file and os.path.exists(self.done_file)):
            return False
        self._scan(force=True)
        return not self.pending

    def first(self) -> Demo:
        while True:
            d = self.next(force=True)
            if d is not None:
                return d
            if self.finished():
                raise SystemExit("no demo to replay")
            time.sleep(self.poll_s)


def env_config(first: Demo, demos: List[Demo], first_dir: str, n_slots: int) -> Dict[str, Any]:
    """``create_from_hdf5``'s config edits for replay_obs's settings (include_contacts=False, include_task_obs=False,
    replay_obs's robot sensors / modalities / proprioception), with ``n_slots`` envs and DummyTask."""
    from omnigibson.envs.data_wrapper import _add_recorded_non_kin_states_to_scene_file

    from ispatialgym.oracle import behavior_lerobot as BL

    with h5py.File(os.path.join(first_dir, "demo.hdf5"), "r") as f:
        config = json.loads(f["data"].attrs["config"])
    config["env"]["flatten_obs_space"] = True
    config["env"]["num_envs"] = int(n_slots)
    scene_file = json.loads(json.dumps(first.scene_file))
    _add_recorded_non_kin_states_to_scene_file(scene_file=scene_file, recorded_scene_files=[d.scene_file for d in demos])
    config["scene"]["scene_file"] = scene_file
    config["task"] = {"type": "DummyTask", "include_obs": False}
    config["objects"] = []
    for robot_cfg in config["robots"]:
        robot_cfg.update(BL.robot_overrides())
    return config


def make_slot_view(env, k: int):
    """Slot ``k`` of the multi-env environment as the one-scene, one-robot environment BEHAVIOR's writer expects."""
    from omnigibson.envs.env_wrapper import EnvironmentWrapper

    class SlotView(EnvironmentWrapper):
        def __init__(self, env, k):
            super().__init__(env=env)
            object.__setattr__(self, "_slot", int(k))

        @property
        def num_envs(self):
            return 1

        @property
        def scene(self):
            return self.env.scenes[self._slot]

        @property
        def scenes(self):
            return [self.env.scenes[self._slot]]

        @property
        def robots(self):
            return self.env.scenes[self._slot].robots

        @property
        def external_sensors(self):
            return None

    return SlotView(env, k)


@contextlib.contextmanager
def individual_render_products(env):
    """With tiled rendering the environment pauses every camera's own render product; BEHAVIOR's writer reads each
    camera's intrinsics from it (the camera-params annotator) when it creates its dataset. Resume them meanwhile."""
    from omnigibson.sensors.vision_sensor import VisionSensor

    import omnigibson as og

    paused = []
    if getattr(env, "_tiled_sensor", None) is not None:
        with og.sim.editing_usd():
            for scene in env.scenes:
                for sensor in scene.robots[0].sensors.values():
                    if isinstance(sensor, VisionSensor) and not sensor.render_product.hydra_texture.updates_enabled:
                        sensor.render_product.hydra_texture.updates_enabled = True
                        paused.append(sensor)
    try:
        yield
    finally:
        if paused:
            with og.sim.editing_usd():
                for sensor in paused:
                    sensor.render_product.hydra_texture.updates_enabled = False


def process_gpu_mib() -> Optional[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            pid, mib = [x.strip() for x in line.split(",")]
            if int(pid) == os.getpid():
                return int(mib)
    except Exception:  # noqa: BLE001
        return None
    return None


def _same(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, th.Tensor) or isinstance(b, th.Tensor):
        return isinstance(a, th.Tensor) and isinstance(b, th.Tensor) and a.shape == b.shape and th.equal(a, b)
    return a == b


class Slot:
    def __init__(self, k: int, scene, writer, robot_name: str):
        self.k = k
        self.scene = scene
        self.writer = writer
        self.proprio_key = f"{robot_name}::proprio"
        self.demo: Optional[Demo] = None
        self.i = 0
        self.t0 = 0.0
        self.last_obs = None
        #: --delta-load: every object's recorded state as last loaded into this slot (None: load everything next)
        self.loaded: Optional[Dict[str, Any]] = None

    def start(self, demo: Demo) -> None:
        from omnigibson.envs.data_wrapper import _align_scene_object_states_with_recorded_schema

        self.scene.restore(demo.scene_file, update_initial_file=True)
        _align_scene_object_states_with_recorded_schema(scene=self.scene, recorded_scene_file=demo.scene_file)
        self.demo, self.i, self.t0 = demo, 0, time.time()
        self.loaded = None

    def load(self, flat: th.Tensor, delta: bool) -> Tuple[int, int]:
        """Load a recorded state into the slot's scene; returns (objects in the state, objects loaded). ``delta``: objects
        whose recorded state equals the one last loaded into this slot are left as they are (nothing else changes a
        slot's objects between loads: no physics step), which leaves the scene as a full load would."""
        if not delta:
            self.scene.load_state(flat, serialized=True)
            return 0, 0
        state, _ = self.scene.deserialize(flat)
        objs = state["registry"]["object_registry"]
        last = self.loaded
        changed = {n: s for n, s in objs.items() if last is None or n not in last or not _same(s, last[n])}
        registry = dict(state["registry"])
        registry["object_registry"] = changed
        self.scene.load_state({**state, "registry": registry}, serialized=False)
        self.loaded = dict(objs) if last is None else {**last, **objs}
        return len(objs), len(changed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", default="", help="demo dirs (demo.hdf5 inside), comma-separated, or @file with one per line")
    ap.add_argument("--watch", default=None, help="replay the successful runs appearing under this demo root (see the module doc)")
    ap.add_argument("--done-file", default=None, help="--watch: no more demos come once this file exists")
    ap.add_argument("--claims", default=None, help="--watch: claim directory shared by several replayers of one demo root")
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--root", required=True, help="LeRobot root dir: one dataset per slot, <root>/<repo prefix>_slot<k>")
    ap.add_argument("--task", required=True, help="task name written into the datasets (the variant)")
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--tiled", action="store_true",
                    help="with several slots, one tiled render product per camera name. Depth matches collection, rgb does not: the "
                         "renderer's post-processing (exposure, denoising) runs over the whole tiled image, so the tiles come out "
                         "darker than a camera's own render product (head camera 12.7 dB PSNR vs a collection frame)")
    ap.add_argument("--repo-prefix", default="replay")
    ap.add_argument("--async-flush", action="store_true", help="the writers' 32-step segments go to a background thread")
    ap.add_argument("--first-renders", type=int, default=10)
    ap.add_argument("--renders", type=int, default=1)
    ap.add_argument("--delta-load", action="store_true",
                    help="per slot, load only the objects whose recorded state differs from the one last loaded into it. The scene "
                         "state matches a full load (poses bitwise, only PhysX sleep flags differ) but the images do not quite: "
                         "right-wrist depth 0.33 mm mean error vs collection (0.12 with full loads), and with --batched-usd-edits the "
                         "moving parts render one state late")
    ap.add_argument("--batched-usd-edits", action="store_true",
                    help="load each tick's states inside og.sim.batched_usd_edits(): one Fabric synchronization per tick instead "
                         "of one per USD edit (~150 per state); state, poses and images as with plain loads, load 2.2x faster")
    ap.add_argument("--warmup-s", type=float, default=60.0,
                    help="render the slots' first states this long before the first frame: a fresh renderer's images keep changing for "
                         "~300 frames (texture streaming, exposure), which a collection process has behind it after its first demo")
    ap.add_argument("--proprio", choices=("recorded", "loaded"), default="recorded")
    ap.add_argument("--results", default=None, help="append REPLAY_DEMO / REPLAY_SUMMARY json lines here")
    ap.add_argument("--profile-ticks", default=None, help="START:COUNT: cProfile those ticks, stats to <results>.prof.txt (or stdout)")
    args = ap.parse_args(argv)

    if args.demos.startswith("@"):
        with open(args.demos[1:]) as f:
            demo_dirs = [line.strip() for line in f if line.strip()]
    else:
        demo_dirs = [d for d in args.demos.split(",") if d]
    if not demo_dirs and not args.watch:
        ap.error("--demos or --watch")
    if args.watch and not args.done_file:
        ap.error("--watch needs --done-file")
    n_slots = max(1, int(args.slots) if args.watch else min(int(args.slots), len(demo_dirs)))
    tiled = args.tiled and n_slots > 1
    t_start = time.time()
    source = DemoSource(demo_dirs, args.watch, args.done_file, args.poll_s, claims=args.claims)
    first = source.first()

    def usable(demo: Demo) -> Demo:
        if args.proprio == "recorded" and demo.proprio is None:
            raise SystemExit(f"no {BEHAVIOR_PROPRIO_KEY} in {demo.dir} (record with ISPATIALGYM_PHYSICS_ONLY=1 or "
                             "ISPATIALGYM_RECORD_BEHAVIOR_PROPRIO=1, or use --proprio loaded)")
        return demo

    usable(first)

    import omnigibson as og
    from omnigibson.macros import gm

    for name, value in (("ENABLE_TRANSITION_RULES", False), ("ENABLE_TILED_RENDERING", tiled), ("VISUAL_ONLY", True)):
        if getattr(gm, name) != value:
            with gm.unlocked():
                setattr(gm, name, value)

    from ispatialgym.oracle import behavior_lerobot as BL

    # the environment of the first demo, as production's one-world replay: the others align their object-state schema
    # when their slot starts them
    config = env_config(first, [first], first.dir, n_slots)
    env = og.Environment(configs=config)
    load_s = time.time() - t_start
    robot_name = env.scenes[0].robots[0].name
    with individual_render_products(env):
        writers = [BL.ShardWriter(make_slot_view(env, k), args.root, f"{args.repo_prefix}_slot{k}", args.task, async_flush=args.async_flush)
                   for k in range(n_slots)]
    slots = [Slot(k, env.scenes[k], writers[k], robot_name) for k in range(n_slots)]
    slots[0].start(first)
    n_demos = 1

    def fill_idle(force: bool = False) -> None:
        nonlocal n_demos
        for s in slots:
            if s.demo is None:
                d = source.next(force)
                if d is None:
                    return
                s.start(usable(d))
                n_demos += 1

    fill_idle(force=True)
    t_warm = time.time()
    n_warm = 0
    if args.warmup_s > 0:
        for s in slots:
            if s.demo is not None:
                s.scene.load_state(s.demo.state_at(0), serialized=True)
        while time.time() - t_warm < args.warmup_s:
            og.sim.render()
            n_warm += 1
    warmup = {"s": round(time.time() - t_warm, 1), "renders": n_warm}
    out = open(args.results, "a") if args.results else None

    def emit(tag: str, rec: Dict[str, Any]) -> None:
        line = f"{tag} {json.dumps(rec)}"
        print(line, flush=True)
        if out is not None:
            out.write(json.dumps({"tag": tag, **rec}) + "\n")
            out.flush()

    t_ticks = defaultdict(float)
    n_ticks = n_steps = 0
    objs_in = objs_loaded = 0
    idle_s = 0.0
    r0, w0 = resource.getrusage(resource.RUSAGE_SELF), time.time()
    gpu_peak = process_gpu_mib() or 0
    prof_window = tuple(int(v) for v in args.profile_ticks.split(":")) if args.profile_ticks else None
    prof = None
    while True:
        if prof_window is not None:
            if n_ticks == prof_window[0]:
                import cProfile

                prof = cProfile.Profile()
                prof.enable()
            elif prof is not None and n_ticks == prof_window[0] + prof_window[1]:
                prof.disable()
                import io
                import pstats

                buf = io.StringIO()
                pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(50)
                pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(30)
                target = (args.results + ".prof.txt") if args.results else None
                if target:
                    with open(target, "w") as f:
                        f.write(buf.getvalue())
                else:
                    print(buf.getvalue(), flush=True)
                prof, prof_window = None, None
        active = [s for s in slots if s.demo is not None]
        if not active:
            if source.finished():
                break
            t_idle = time.time()
            time.sleep(args.poll_s)
            fill_idle(force=True)
            idle_s += time.time() - t_idle
            continue
        t1 = time.perf_counter()
        with og.sim.batched_usd_edits() if args.batched_usd_edits else contextlib.nullcontext():
            for s in active:
                n_in, n_loaded = s.load(s.demo.state_at(s.i), args.delta_load)
                objs_in += n_in
                objs_loaded += n_loaded
        t2 = time.perf_counter()
        for _ in range(args.first_renders if any(s.i == 0 for s in active) else args.renders):
            og.sim.render()
        t3 = time.perf_counter()
        obs_list, _ = env.get_obs()
        t4 = time.perf_counter()
        for s in active:
            d, i = s.demo, s.i
            # tiled camera buffers are persistent CUDA tensors (per-slot views), and the writer converts frames with .numpy()
            obs = {key: ((v.detach().cpu() if v.is_cuda else v.clone()) if isinstance(v, th.Tensor) else v) for key, v in obs_list[s.k].items()}
            if args.proprio == "recorded":
                obs[s.proprio_key] = th.as_tensor(d.proprio[i], dtype=th.float32)
            if i == 0:
                s.writer.begin(obs)
            else:
                s.writer.step(d.action[i - 1], obs, float(d.reward[i - 1]), bool(d.terminated[i - 1]), bool(d.truncated[i - 1]))
            s.i += 1
            n_steps += 1
            if s.i == d.T:
                # the last action; the observation it led to is not a frame (the writer pairs it with no action)
                s.writer.step(d.action[d.T - 1], obs, float(d.reward[d.T - 1]), bool(d.terminated[d.T - 1]), bool(d.truncated[d.T - 1]))
                index = s.writer.end(save=True)
                emit("REPLAY_DEMO", {"demo": d.dir, "slot": s.k, "frames": d.T, "episode_index": index, "wall_s": round(time.time() - s.t0, 2),
                                     "dataset": os.path.join(args.root, f"{args.repo_prefix}_slot{s.k}")})
                s.demo = None
        fill_idle()
        t5 = time.perf_counter()
        t_ticks["load_states"] += t2 - t1
        t_ticks["render"] += t3 - t2
        t_ticks["get_obs"] += t4 - t3
        t_ticks["writers"] += t5 - t4
        n_ticks += 1
        if n_ticks % 200 == 0:
            gpu_peak = max(gpu_peak, process_gpu_mib() or 0)
    t_flush = time.perf_counter()
    for w in writers:
        w.close()
    close_s = time.perf_counter() - t_flush
    w1, r1 = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    wall = w1 - w0
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    busy = wall - idle_s
    emit("REPLAY_SUMMARY", {
        "slots": n_slots, "tiled": tiled, "demos": n_demos, "env_steps": n_steps, "ticks": n_ticks, "load_s": round(load_s, 1), "warmup": warmup,
        "replay_wall_s": round(wall, 2), "idle_s": round(idle_s, 2), "close_s": round(close_s, 2),
        "env_steps_per_s": round(n_steps / max(wall, 1e-9), 2), "env_steps_per_busy_s": round(n_steps / max(busy, 1e-9), 2),
        "ms_per_tick": {k: round(1000 * v / max(1, n_ticks), 2) for k, v in t_ticks.items()},
        "cpu_cores_busy": round(cpu / max(wall, 1e-9), 2), "cpu_ms_per_env_step": round(1000 * cpu / max(1, n_steps), 1),
        "gpu_mib_peak": max(gpu_peak, process_gpu_mib() or 0), "async_flush": bool(args.async_flush), "proprio": args.proprio,
        "delta_load": bool(args.delta_load), "batched_usd_edits": bool(args.batched_usd_edits),
        "objects_per_state": round(objs_in / max(1, n_steps), 1) if args.delta_load else None,
        "objects_loaded_per_state": round(objs_loaded / max(1, n_steps), 1) if args.delta_load else None,
        "renders": args.renders, "first_renders": args.first_renders, "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "cpus_allowed": len(os.sched_getaffinity(0)), "host": os.uname().nodename, "pid": os.getpid()})
    if out is not None:
        out.close()
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    try:
        import omnigibson as og

        og.shutdown()
    except OSError:  # Kit's temp-dir cleanup at exit; the datasets are written by then
        pass
    sys.exit(rc)
