"""Step-cost benchmark: N copies of one instance's scene in one OmniGibson process (a measurement tool, not used by
production).

    python -m ispatialgym.multienv.bench_step --tro <iid>-tro_state.json --demo <demo dir> --num-envs N [--tiled]
        [--steps 150] [--sync-dir DIR --sync-n K] [--out result.json]

The robot and cameras are run_isg's (``build_env_config``: R1Pro, head + wrist cameras at the evaluation profile's
1080 px unless ``ISPATIALGYM_HEAD_CAMERA_ONLY``); the task is a ``DummyTask`` (``ISpatialGymTask`` is single-env). Every
scene gets the demo's recorded state at step ``i * --offset`` (``demo.hdf5`` state[0], then the filtered state of that
step) and replays the demo's recorded actions from there, so the envs are not in lockstep. Each phase starts from those
states again, so every phase replays the same segment:

  record   ``env.step(actions)`` as the executor calls it (physics + render + observations + task), then every scene's
           serialized state with the per-step recording filter (``Executor._dump_state(full=False)``), timed apart
  parts    apply_action / ``og.sim.step()`` / ``env.get_obs()`` timed apart, then physics-only steps (no render) and
           ``og.sim.render()`` alone

each with change detection (``gm.TRACK_STATE_UPDATES``) on and off: the flag is read every step, so one load serves
both. With ``--sync-dir`` the process waits after loading until ``--sync-n`` processes are ready (the separate-process
baseline: several single-env processes timed together on one GPU). GPU utilization and memory are sampled with
nvidia-smi during each phase. Prints ``BENCH {json}`` lines and writes ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time

import torch as th

from ispatialgym.oracle import run_isg as R  # noqa: F401  (curobo before omnigibson)

import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402


class GpuSampler:
    """``nvidia-smi`` utilization / memory samples every ``period_ms`` in a background thread."""

    def __init__(self, period_ms: int = 200):
        self.samples = []
        self._proc = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", f"-lms={period_ms}"],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        for line in self._proc.stdout:
            try:
                u, m = (float(x) for x in line.strip().split(","))
            except ValueError:
                continue
            self.samples.append((time.time(), u, m))

    def window(self, t0: float, t1: float):
        s = [(u, m) for t, u, m in self.samples if t0 <= t <= t1]
        if not s:
            return {"util_mean": None, "mem_max_mib": None, "n": 0}
        return {"util_mean": round(sum(u for u, _ in s) / len(s), 1), "mem_max_mib": max(m for _, m in s), "n": len(s)}

    def stop(self):
        self._proc.terminate()


def process_gpu_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return sum(int(l.split(",")[1]) for l in out.strip().splitlines() if l.split(",")[0].strip() == str(os.getpid()))


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return None
    return {"mean_ms": round(1000 * sum(xs) / len(xs), 2), "median_ms": round(1000 * xs[len(xs) // 2], 2), "n": len(xs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tro", required=True)
    ap.add_argument("--demo", required=True, help="demo dir (demo.hdf5 inside) of the same instance")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--tiled", action="store_true", help="gm.ENABLE_TILED_RENDERING (one render product for every env's cameras)")
    ap.add_argument("--steps", type=int, default=150, help="batched steps per timed loop")
    ap.add_argument("--parts-steps", type=int, default=60)
    ap.add_argument("--offset", type=int, default=50, help="env i starts this many demo steps after env i-1")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--image-size", type=int, default=R.DEFAULT_IMAGE_SIZE)
    ap.add_argument("--track", default="1,0", help="change-detection settings to time, in order")
    ap.add_argument("--sync-dir", default=None)
    ap.add_argument("--sync-n", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import h5py

    for name, value in (("ENABLE_OBJECT_STATES", True), ("USE_GPU_DYNAMICS", False), ("ENABLE_TRANSITION_RULES", False), ("ENABLE_TILED_RENDERING", bool(args.tiled))):
        if getattr(gm, name) != value:
            with gm.unlocked():
                setattr(gm, name, value)
    N = int(args.num_envs)
    tro_path = os.path.abspath(args.tro)
    iid, template_dir, tmeta, activity, did = R.resolve_instance(tro_path)
    tro = R.load_tro(tro_path)
    cfg = R.build_env_config(template_dir, tmeta, tro, tro_path, activity, did, R.robot_name_from_tro(tro), args.image_size)
    cfg["task"] = {"type": "DummyTask"}
    cfg["env"]["num_envs"] = N
    with h5py.File(os.path.join(args.demo, "demo.hdf5"), "r") as f:
        d = f["data/demo_0"]
        actions = th.as_tensor(d["action"][()], dtype=th.float32)
        sizes = d["state_size"][()]
        states = d["state"][()]
    T = int(actions.shape[0])
    offsets = [(i * args.offset) % max(1, T - args.steps - args.warmup - 1) for i in range(N)]
    res = {"gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
           "host": os.uname().nodename, "num_envs": N, "tiled": bool(args.tiled), "image_size": args.image_size,
           "head_camera_only": os.environ.get("ISPATIALGYM_HEAD_CAMERA_ONLY", ""), "tro": tro_path, "demo": args.demo, "offsets": offsets,
           "omp_threads": os.environ.get("OMP_NUM_THREADS"), "isg_thread_cap": os.environ.get("ISG_THREAD_CAP"),
           "cpus_allowed": len(os.sched_getaffinity(0)), "pid": os.getpid()}
    gpu = GpuSampler()
    t0 = time.time()
    env = og.Environment(configs=cfg)
    res["load_s"] = round(time.time() - t0, 1)
    robots = [s.robots[0] for s in env.scenes]
    res["cameras_per_robot"] = sum(1 for k in robots[0].sensors if "Camera" in k)
    res["gpu_mib_after_load"] = process_gpu_mib()
    print("BENCH " + json.dumps({"event": "loaded", **res}), flush=True)

    def reset_to_offsets():
        s0 = th.as_tensor(states[0][: int(sizes[0])], dtype=th.float32)
        for scene, off in zip(env.scenes, offsets):
            scene.load_state(s0, serialized=True)
            if off > 0:
                scene.load_state(th.as_tensor(states[off][: int(sizes[off])], dtype=th.float32), serialized=True)
        for _ in range(2):
            og.sim.step()

    def batch(t):
        return th.stack([actions[min(off + t, T - 1)] for off in offsets])

    def active(obj):
        return bool(getattr(obj, "is_active", True) and getattr(obj, "initialized", True))

    for scene in env.scenes:
        scene.object_registry.set_dump_filter(active)
    reset_to_offsets()
    for t in range(args.warmup):
        env.step(batch(t))
    if args.sync_dir:
        os.makedirs(args.sync_dir, exist_ok=True)
        open(os.path.join(args.sync_dir, f"ready.{os.getpid()}"), "w").close()
        deadline = time.time() + 3600
        while len([p for p in os.listdir(args.sync_dir) if p.startswith("ready.")]) < args.sync_n and time.time() < deadline:
            time.sleep(0.5)
        res["sync_wait_end"] = time.time()

    phases = []
    for track in [s.strip() for s in args.track.split(",") if s.strip()]:
        with gm.unlocked():
            gm.TRACK_STATE_UPDATES = track == "1"
        # -- record: env.step + the recorded per-scene state, as the executor does per step
        reset_to_offsets()
        th.cuda.synchronize()
        step_t, dump_t = [], []
        r0, w0 = resource.getrusage(resource.RUSAGE_SELF), time.time()
        for t in range(args.steps):
            a = batch(args.warmup + t)
            t1 = time.perf_counter()
            env.step(a)
            t2 = time.perf_counter()
            for scene in env.scenes:
                scene.dump_state(serialized=True)
            t3 = time.perf_counter()
            step_t.append(t2 - t1)
            dump_t.append(t3 - t2)
        w1, r1 = time.time(), resource.getrusage(resource.RUSAGE_SELF)
        cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
        ph = {"phase": "record", "track_state_updates": track == "1", "env_step": stats(step_t), "dump_all_scenes": stats(dump_t),
              "wall_s": round(w1 - w0, 2), "t0": w0, "t1": w1, "cpu_cores_busy": round(cpu / (w1 - w0), 2), "gpu": gpu.window(w0, w1),
              "ms_per_env_step": round(1000 * (sum(step_t) + sum(dump_t)) / (args.steps * N), 2)}
        phases.append(ph)
        print("BENCH " + json.dumps(ph), flush=True)
        # -- parts: where a step's time goes
        reset_to_offsets()
        th.cuda.synchronize()
        app_t, sim_t, obs_t = [], [], []
        w0 = time.time()
        for t in range(args.parts_steps):
            a = batch(args.warmup + t)
            t1 = time.perf_counter()
            for i, r in enumerate(robots):
                r.apply_action(a[i])
            t2 = time.perf_counter()
            og.sim.step()
            t3 = time.perf_counter()
            env.get_obs()
            t4 = time.perf_counter()
            app_t.append(t2 - t1)
            sim_t.append(t3 - t2)
            obs_t.append(t4 - t3)
        w1 = time.time()
        phys_t, rend_t = [], []
        render_on_step = og.sim._render_on_step
        og.sim._render_on_step = False
        try:
            for t in range(args.parts_steps):
                a = batch(args.warmup + args.parts_steps + t)
                for i, r in enumerate(robots):
                    r.apply_action(a[i])
                t1 = time.perf_counter()
                og.sim.step()
                th.cuda.synchronize()
                t2 = time.perf_counter()
                og.sim.render()
                th.cuda.synchronize()
                t3 = time.perf_counter()
                phys_t.append(t2 - t1)
                rend_t.append(t3 - t2)
        finally:
            og.sim._render_on_step = render_on_step
        w2 = time.time()
        ph = {"phase": "parts", "track_state_updates": track == "1", "apply_action": stats(app_t), "sim_step_with_render": stats(sim_t), "get_obs": stats(obs_t),
              "physics_only_step": stats(phys_t), "render_only": stats(rend_t), "gpu_with_render": gpu.window(w0, w1), "gpu_split": gpu.window(w1, w2)}
        phases.append(ph)
        print("BENCH " + json.dumps(ph), flush=True)
    res["phases"] = phases
    res["gpu_mib_end"] = process_gpu_mib()
    res["torch_max_reserved_gib"] = round(th.cuda.max_memory_reserved() / 2**30, 2)
    gpu.stop()
    print("BENCH " + json.dumps({"event": "done", "load_s": res["load_s"], "gpu_mib_end": res["gpu_mib_end"]}), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
