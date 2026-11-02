"""Where a batched ``env.step`` spends its time: the vectorized world of one eval instance in every slot, ``--steps`` steps of
its demo, cProfile over the loop plus manual timers of the step's stages (apply_action, og.sim.step, get_obs, task.step).

    python scripts/vec_eval/profile_step.py --task object_scaling --instance-index 0 --num-envs 8 --steps 40 --out /tmp/prof
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OMNIGIBSON_CHANGE_DETECTION_NUMPY", "1")
os.environ.setdefault("ISPATIALGYM_SHARED_GOAL_EVAL", "1")
# batched, bit-identical symmetry search of the goal check (geometry_fast; 73 x 73 equivalence pairs per attachment step)
os.environ.setdefault("ISPATIALGYM_FAST_GEOMETRY", "1")
os.environ.setdefault("ISPATIALGYM_FAST_ASSIGN", "1")

import numpy as np  # noqa: E402
import torch as th  # noqa: E402

from ispatialgym.oracle.run_isg import replay_rows  # noqa: E402

import omnigibson as og  # noqa: E402

from ispatialgym.eval import instances as INST  # noqa: E402
from ispatialgym.oracle.vec_loader import VecInstanceLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="object_scaling")
    ap.add_argument("--instance-index", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--external-size", type=int, default=320)
    ap.add_argument("--no-external", action="store_true")
    ap.add_argument("--warm-renders", type=int, default=20)
    ap.add_argument("--out", default="/tmp/prof")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rec = INST.resolve(args.task, [args.instance_index])[0]
    N = args.num_envs
    ext = None if args.no_external else (args.external_size, int(args.external_size * 1.5) // 2 * 2)
    t0 = time.time()
    loader = VecInstanceLoader(num_envs=N, behavior_output=True, tiled_rendering=False, settle_physics_steps=25, warm_renders=args.warm_renders, drop_depth=True,
                               external_camera_hw=ext)
    loader.build_env(rec.path("tro_state"))
    print(f"world built in {time.time() - t0:.1f}s", flush=True)
    env, task = loader.env, loader.task
    t0 = time.time()
    obs_by, info_by = loader.restore_batch([rec.path("tro_state")] * N, settle_eval=True, settle_steps=[None] * N, mode="replay", tolerance_scale=1.0)
    print(f"restore_batch {time.time() - t0:.1f}s; loaded {sorted(obs_by)}", flush=True)
    rows = list(replay_rows(rec.demo_path(), 0, int(rec.episode["action_steps"]), int(rec.episode["settle_steps"])))
    acts = [th.as_tensor(a, dtype=th.float32) for a in rows[: args.steps]]
    while len(acts) < args.steps:
        acts.append(acts[-1].clone())

    # -- GPU utilization sampler ------------------------------------------------------------------------------------------
    import subprocess
    import threading

    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                u, m = r.stdout.strip().splitlines()[0].split(",")
                samples.append((int(u), int(m)))
            except Exception:  # noqa: BLE001
                pass
            stop.wait(0.5)

    th_s = threading.Thread(target=sampler, daemon=True)
    th_s.start()
    # -- manual stage timers (one pass) ---------------------------------------------------------------------------------
    stages = {"apply_action": 0.0, "sim_step": 0.0, "get_obs": 0.0, "task_step": 0.0, "total": 0.0}
    for k in range(args.steps // 2):
        a = th.stack([acts[k]] * N)
        t_all = time.time()
        t = time.time()
        for i in range(N):
            env.scenes[i].robots[0].apply_action(a[i])
        stages["apply_action"] += time.time() - t
        t = time.time()
        og.sim.step()
        stages["sim_step"] += time.time() - t
        t = time.time()
        obs_list, info_list = env.get_obs()
        stages["get_obs"] += time.time() - t
        t = time.time()
        rewards, dones, infos = env.task.step(env, a)
        stages["task_step"] += time.time() - t
        stages["total"] += time.time() - t_all
    n = max(1, args.steps // 2)
    per = {k: round(1000 * v / n, 1) for k, v in stages.items()}
    print("STAGES ms/step:", json.dumps(per), flush=True)
    # physics vs render inside og.sim.step
    t = time.time()
    for _ in range(10):
        og.sim.render()
    render_ms = 100 * (time.time() - t)
    t = time.time()
    for _ in range(10):
        og.sim.step_physics()
    phys_ms = 100 * (time.time() - t)
    print(f"RENDER-ONLY {render_ms:.1f} ms per render; PHYSICS-ONLY {phys_ms:.1f} ms per physics substep (x4 per control step)", flush=True)
    # per-env get_obs: robot sensors vs proprio vs external
    t = time.time()
    for i in range(N):
        for s in env.scenes[i].robots[0].sensors.values():
            s.get_obs()
    sens_ms = 1000 * (time.time() - t)
    t = time.time()
    for i in range(N):
        env.scenes[i].robots[0].get_proprioception()
    prop_ms = 1000 * (time.time() - t)
    print(f"GET_OBS parts: robot sensors {sens_ms:.1f} ms, proprio {prop_ms:.1f} ms (all {N} envs)", flush=True)

    # -- cProfile (second pass) ------------------------------------------------------------------------------------------
    pr = cProfile.Profile()
    pr.enable()
    t = time.time()
    for k in range(args.steps // 2, args.steps):
        a = th.stack([acts[k]] * N)
        env.step(a, n_render_iterations=1)
    wall = time.time() - t
    pr.disable()
    print(f"PROFILED {args.steps - args.steps // 2} env.step calls: {1000 * wall / max(1, args.steps - args.steps // 2):.1f} ms/step", flush=True)
    stop.set()
    if samples:
        us = [u for u, _ in samples]
        print(f"GPU util during the loops: mean {sum(us) / len(us):.0f}% max {max(us)}% over {len(samples)} samples; memory used max {max(m for _, m in samples)} MiB", flush=True)
    # a pure render loop with GPU sampling
    samples.clear(); stop.clear()
    th_r = threading.Thread(target=sampler, daemon=True); th_r.start()
    t = time.time()
    for _ in range(40):
        og.sim.render()
    r_ms = 1000 * (time.time() - t) / 40
    stop.set()
    us = [u for u, _ in samples]
    print(f"RENDER LOOP 40 renders: {r_ms:.1f} ms each; GPU util mean {sum(us) / max(1, len(us)):.0f}%", flush=True)
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(70)
    open(os.path.join(args.out, f"profile_n{N}.txt"), "w").write(s.getvalue())
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(50)
    open(os.path.join(args.out, f"profile_n{N}_tottime.txt"), "w").write(s2.getvalue())
    pr.dump_stats(os.path.join(args.out, f"profile_n{N}.prof"))
    print(s2.getvalue()[:6000], flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
    os._exit(0)
