"""Replay-load lag diagnosis (a measurement tool, not used by production): why ``--batched-usd-edits --delta-load``
renders some parts one recorded state late while plain and batched-only loads are exact.

    python -m ispatialgym.multienv.replay_lag --demo <physics-only demo dir> --out <dir> --variants a,b,...
        [--start 100] [--frames 200] [--warmup-s 60]

One slot (the replayer's environment, :func:`ispatialgym.multienv.replay.env_config`), every variant replaying states
--start .. --start+--frames-1 after loading the states before them the same way. Per frame and camera of the robot: the
rgb frame, the render-side view matrix (camera-params annotator ``cameraViewTransform``, the transform the renderer drew
with) and the PhysX-side world pose of the camera's link; per frame the names of the objects the load wrote (delta).
Saved to ``<out>/<variant>.npz``.

Variants (load mode, then what happens between the loads and the read):
  plain                   scene.load_state, render, read (the reference)
  batched                 loads inside og.sim.batched_usd_edits()
  delta                   only objects whose recorded state changed since the slot's last load
  batched_delta           both (the lagging combination)
  batched_delta_r2        both, two renders before the read
  batched_delta_sync      both, then physics_only.sync_transforms (PhysX -> Fabric) before the render
  batched_delta_robot     both, but the robot is always loaded in full, even if unchanged
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch as th

from ispatialgym.multienv import replay as RP


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup-s", type=float, default=60.0)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    demo = RP.Demo(args.demo)

    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.sensors.vision_sensor import VisionSensor

    from ispatialgym.oracle import physics_only as PO

    for name, value in (("ENABLE_TRANSITION_RULES", False), ("ENABLE_TILED_RENDERING", False), ("VISUAL_ONLY", True)):
        if getattr(gm, name) != value:
            with gm.unlocked():
                setattr(gm, name, value)
    env = og.Environment(configs=RP.env_config(demo, [demo], demo.dir, 1))
    robot = env.scenes[0].robots[0]
    slot = RP.Slot(0, env.scenes[0], None, robot.name)
    slot.start(demo)
    cams = {n: s for n, s in robot.sensors.items() if isinstance(s, VisionSensor)}
    for s in cams.values():
        s.camera_parameters  # attach the camera-params annotator
    slot.scene.load_state(demo.state_at(0), serialized=True)
    t_w = time.time()
    while time.time() - t_w < args.warmup_s:
        og.sim.render()
    end = min(demo.T, args.start + args.frames)
    report = {"demo": demo.dir, "start": args.start, "end": end, "variants": {}}

    def load(i, kind):
        batched = kind.startswith("batched")
        delta = "delta" in kind
        with og.sim.batched_usd_edits() if batched else contextlib.nullcontext():
            if not delta:
                slot.scene.load_state(demo.state_at(i), serialized=True)
                return []
            flat = demo.state_at(i)
            state, _ = slot.scene.deserialize(flat)
            objs = state["registry"]["object_registry"]
            last = slot.loaded
            changed = {n: s for n, s in objs.items() if last is None or n not in last or not RP._same(s, last[n])}
            if kind == "batched_delta_robot":
                changed[robot.name] = objs[robot.name]
            registry = dict(state["registry"])
            registry["object_registry"] = changed
            slot.scene.load_state({**state, "registry": registry}, serialized=False)
            slot.loaded = dict(objs) if last is None else {**last, **objs}
            return sorted(changed)

    for vname in [v for v in args.variants.split(",") if v]:
        slot.loaded = None
        for i in range(0, args.start):  # the same history before the recorded segment
            load(i, vname)
        frames, views, links, loaded = {n: [] for n in cams}, {n: [] for n in cams}, {n: [] for n in cams}, []
        t0 = time.time()
        for i in range(args.start, end):
            loaded.append(load(i, vname))
            if vname == "batched_delta_sync":
                PO.sync_transforms(og.sim)
            for _ in range(2 if vname == "batched_delta_r2" else 1):
                og.sim.render()
            obs_list, _ = env.get_obs()
            o = obs_list[0]
            for n, s in cams.items():
                key = next(k for k in o if k.startswith(robot.name + "::") and k.split("::")[-2] == n and k.endswith("::rgb"))
                frames[n].append(o[key][..., :3].detach().cpu().numpy().astype(np.uint8))
                views[n].append(np.asarray(s.camera_parameters["cameraViewTransform"], dtype=np.float64).reshape(-1))
                link = robot.links.get(n.split(":")[1])
                p, q = (link if link is not None else s).get_position_orientation()
                links[n].append(np.concatenate([np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)]))
        np.savez_compressed(os.path.join(args.out, f"{vname}.npz"),
                            **{f"frames::{n}": np.stack(v) for n, v in frames.items()},
                            **{f"view::{n}": np.stack(v) for n, v in views.items()},
                            **{f"link::{n}": np.stack(v) for n, v in links.items()})
        report["variants"][vname] = {"wall_s": round(time.time() - t0, 1),
                                     "objects_loaded_per_frame": round(float(np.mean([len(x) for x in loaded])), 2) if loaded and loaded[0] else None,
                                     "loaded_first": loaded[:3]}
        print("LAG_VARIANT " + json.dumps({"variant": vname, **report["variants"][vname]}), flush=True)
        with open(os.path.join(args.out, "lag.json"), "w") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    with contextlib.suppress(Exception):
        import omnigibson as og

        og.shutdown()
    os._exit(rc)
