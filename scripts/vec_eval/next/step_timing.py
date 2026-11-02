"""Where a vectorized control step goes (campaign vec_next, 2026-11-01): builds VecInstanceLoader(num_envs=N) for the first instance of
an instance list, restores it into every slot (robots holding), then times, over repeated calls:
  physics  - one og.sim.step_physics() (a single PhysX substep, no render)
  render   - one og.sim.render()
  sim_step - one og.sim.step() (the control step's substeps + render + OmniGibson's non-physics update)
  env_step - one env.step(hold actions) (sim_step + task / termination checks + observations)
Nothing is recorded in the record.

    python scripts/vec_eval/next/step_timing.py --instances <alias.json> --num-envs 8 [--reps 30]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import omnigibson as og  # noqa: F401  (import order)
import torch as th

from ispatialgym.oracle.vec_loader import VecInstanceLoader  # noqa: E402


def _time(fn, reps):
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instances", required=True)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--external-size", type=int, default=240)
    a = p.parse_args()
    tro = json.load(open(a.instances))["instances"][0]["tro"]
    ext = (a.external_size, int(round(a.external_size * 1.5)) // 2 * 2)
    loader = VecInstanceLoader(num_envs=a.num_envs, behavior_output=True, tiled_rendering=False, drop_depth=True, external_camera_hw=ext,
                               layout="grid-centered", scene_margin=5.0)
    loader.build_env(tro)
    loader.restore_batch([tro] * a.num_envs)
    env = loader.env
    acts = th.stack([loader.hold_action(i) for i in range(a.num_envs)])
    n_sub = og.sim.n_physics_timesteps_per_render
    for _ in range(5):  # warm up every path once
        env.step(acts)
    res = {
        "physics_substep": _time(og.sim.step_physics, a.reps * n_sub),
        "render": _time(og.sim.render, a.reps),
        "sim_step": _time(og.sim.step, a.reps),
        "env_step": _time(lambda: env.step(acts), a.reps),
    }
    med = {k: statistics.median(v) for k, v in res.items()}
    print(f"STEP_TIMING num_envs={a.num_envs} substeps_per_control_step={n_sub} " + " ".join(f"{k}_ms={1000 * v:.1f}" for k, v in med.items())
          + f" physics_per_control_step_ms={1000 * med['physics_substep'] * n_sub:.1f}", flush=True)
    og.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
