"""Fabric-direct replay feasibility prototype (a measurement tool, not used by production): does the renderer draw link
poses written straight into Fabric (no USD edits, no PhysX writes), and how fast is a tick?

    python -m ispatialgym.multienv.replay_fabric --demo <physics-only demo dir> --out <dir> [--start 100] [--frames 200]

One slot (the replayer's environment, :func:`ispatialgym.multienv.replay.env_config`).
  pass 1 (plain): every recorded state loaded as the exact replayer does (``og.sim.batched_usd_edits()``), one render,
           the robot cameras' rgb read, and every link prim's Fabric world transform
           (``og.sim.fabric_hierarchy.get_world_xform``) kept where it changed since the previous state. These stand in
           for the per-step link poses a collector would record (an opt-in ``isg/link_poses`` dataset).
  pass 2 (fabric): the first state loaded once the plain way, then per state only those transforms written with
           ``fabric_hierarchy.set_world_xform`` and propagated to the links' descendants (cameras, visual meshes) with
           ``update_world_xforms``, one render, the cameras read.
Saves both passes' frames to ``<out>/fabric.npz`` and ``<out>/fabric.json`` (ms per tick: load / write / render / read,
changed links per state, rgb PSNR of pass 2 against pass 1 per camera, or the error if a call is not available).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback

import numpy as np

from ispatialgym.multienv import replay as RP


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup-s", type=float, default=60.0)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    demo = RP.Demo(args.demo)

    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.macros import gm
    from omnigibson.sensors.vision_sensor import VisionSensor

    for name, value in (("ENABLE_TRANSITION_RULES", False), ("ENABLE_TILED_RENDERING", False), ("VISUAL_ONLY", True)):
        if getattr(gm, name) != value:
            with gm.unlocked():
                setattr(gm, name, value)
    env = og.Environment(configs=RP.env_config(demo, [demo], demo.dir, 1))
    scene = env.scenes[0]
    robot = scene.robots[0]
    slot = RP.Slot(0, scene, None, robot.name)
    slot.start(demo)
    cams = [n for n, s in robot.sensors.items() if isinstance(s, VisionSensor)]
    paths = sorted({link.prim_path for obj in scene.objects for link in obj.links.values()})
    fh = og.sim.fabric_hierarchy
    sdf = lazy.usdrt.Sdf.Path
    rep = {"demo": demo.dir, "links": len(paths), "start": args.start}

    def read(o):
        out = {}
        for n in cams:
            key = next(k for k in o if k.startswith(robot.name + "::") and k.split("::")[-2] == n and k.endswith("::rgb"))
            out[n] = o[key][..., :3].detach().cpu().numpy().astype(np.uint8)
        return out

    slot.scene.load_state(demo.state_at(0), serialized=True)
    t_w = time.time()
    while time.time() - t_w < args.warmup_s:
        og.sim.render()
    end = min(demo.T, args.start + args.frames)
    # pass 1: plain (batched) loads, transforms recorded
    with og.sim.batched_usd_edits():
        slot.scene.load_state(demo.state_at(args.start), serialized=True)
    og.sim.render()
    prev = {p: np.array(fh.get_world_xform(sdf(p))) for p in paths}
    changes, frames1 = [], {n: [] for n in cams}
    t_load = t_render = t_read = 0.0
    for i in range(args.start, end):
        t0 = time.perf_counter()
        with og.sim.batched_usd_edits():
            slot.scene.load_state(demo.state_at(i), serialized=True)
        t1 = time.perf_counter()
        og.sim.render()
        t2 = time.perf_counter()
        f = read(env.get_obs()[0][0])
        t3 = time.perf_counter()
        for n in cams:
            frames1[n].append(f[n])
        ch = {}
        for p in paths:
            m = fh.get_world_xform(sdf(p))
            a = np.array(m)
            if not np.array_equal(a, prev[p]):
                ch[p] = m
                prev[p] = a
        changes.append(ch)
        t_load += t1 - t0
        t_render += t2 - t1
        t_read += t3 - t2
    n1 = end - args.start
    rep["plain_ms_per_tick"] = {"load": round(1000 * t_load / n1, 1), "render": round(1000 * t_render / n1, 1), "read": round(1000 * t_read / n1, 1)}
    rep["changed_links_per_state"] = round(float(np.mean([len(c) for c in changes])), 1)
    # pass 2: fabric writes only
    frames2 = {n: [] for n in cams}
    try:
        with og.sim.batched_usd_edits():
            slot.scene.load_state(demo.state_at(args.start), serialized=True)
        og.sim.render()
        t_write = t_upd = t_render2 = t_read2 = 0.0
        for k, i in enumerate(range(args.start, end)):
            t0 = time.perf_counter()
            for p, m in changes[k].items():
                fh.set_world_xform(sdf(p), m)
            t1 = time.perf_counter()
            fh.update_world_xforms()
            t2 = time.perf_counter()
            og.sim.render()
            t3 = time.perf_counter()
            f = read(env.get_obs()[0][0])
            t4 = time.perf_counter()
            for n in cams:
                frames2[n].append(f[n])
            t_write += t1 - t0
            t_upd += t2 - t1
            t_render2 += t3 - t2
            t_read2 += t4 - t3
        rep["fabric_ms_per_tick"] = {"write": round(1000 * t_write / n1, 1), "update_world_xforms": round(1000 * t_upd / n1, 1),
                                     "render": round(1000 * t_render2 / n1, 1), "read": round(1000 * t_read2 / n1, 1)}
        psnr = {}
        for n in cams:
            a, b = np.stack(frames1[n]).astype(np.float64), np.stack(frames2[n]).astype(np.float64)
            mse = ((a - b) ** 2).reshape(len(a), -1).mean(axis=1)
            p = 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-10))
            psnr[n.split(":")[1]] = {"mean": round(float(p.mean()), 2), "min": round(float(p.min()), 2),
                                     "first_10": round(float(p[:10].mean()), 2), "last_10": round(float(p[-10:].mean()), 2)}
        rep["fabric_vs_plain_psnr"] = psnr
    except Exception as e:  # noqa: BLE001
        rep["fabric_error"] = repr(e)
        rep["fabric_trace"] = traceback.format_exc()[-1500:]
    np.savez_compressed(os.path.join(args.out, "fabric.npz"), **{f"plain::{n}": np.stack(v) for n, v in frames1.items()},
                        **{f"fabric::{n}": np.stack(v) for n, v in frames2.items() if v})
    with open(os.path.join(args.out, "fabric.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("FABRIC " + json.dumps({k: v for k, v in rep.items() if k != "fabric_trace"}), flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    with contextlib.suppress(Exception):
        import omnigibson as og

        og.shutdown()
    os._exit(rc)
