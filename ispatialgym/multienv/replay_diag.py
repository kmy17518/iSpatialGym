"""Multi-slot replay diagnosis (a measurement tool, not used by production): why slot 0's wrist cameras render
differently in a stage holding several scenes.

    python -m ispatialgym.multienv.replay_diag --demo <demo dir> --slots N --out <dir> --variants a,b,...
        [--frames 557] [--stride 4] [--dummies 9] [--warmup-s 60]

Builds the replayer's environment (:func:`ispatialgym.multienv.replay.env_config`: production's aligned-replay settings,
replay_obs's cameras) with N scenes, every slot playing the same demo in lockstep (as ``replay --slots N`` with the
demo repeated, the n4nt run), and runs the variants one after the other in this process, each over frames
0..--frames-1: every slot loads the recorded state (batched USD edits), the cameras render (10 renders on frame 0, then
1) and the observations are read through ``env.get_obs()``. Per variant ``<out>/<variant>.npz`` holds slot 0's rgb
frames (every --stride-th, uint8) and every frame's mean rgb per camera; ``<out>/diag.json`` the per-variant tick times,
the light prims per scene, and the renderer settings (carb ``/rtx``, ``/persistent/rtx``, ``/rtx-transient``,
``/app/renderer``, ``/app/hydra``, ``/omni/replicator``; every RenderProduct / RenderSettings prim's attributes) read
after loading and after each variant.

Variants (names may repeat with a suffix ``#k`` to rerun one later in the process):
  all              every slot's cameras render (the n4nt setup)
  only0            the other slots' render products paused (hydra texture updates off)
  only0_nolights   only0, and the other slots' light prims at intensity 0
  all_nolights     every slot's cameras render, the other slots' lights at intensity 0
  base             slot 0's cameras and nothing else (with --dummies: the dummies paused)
  dummies          slot 0's cameras plus --dummies external cameras rendering and read (3 at 720 px, the rest at
                   480 px, rgb + depth_linear, as replay_obs's head / wrists), placed around slot 0's robot
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import torch as th

from ispatialgym.multienv import replay as RP

SETTINGS_ROOTS = ("/rtx", "/persistent/rtx", "/rtx-transient", "/app/renderer", "/app/hydra", "/omni/replicator", "/renderer",
                  "/app/asyncRendering", "/app/asyncRenderingLowLatency", "/persistent/app/viewport")


def read_settings(stage):
    import carb

    s = carb.settings.get_settings()
    out = {}
    for root in SETTINGS_ROOTS:
        try:
            out[root] = s.get(root)
        except Exception as e:  # noqa: BLE001
            out[root] = f"<error {e!r}>"
    prims = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() in ("RenderProduct", "RenderSettings", "RenderVar"):
            attrs = {}
            for a in prim.GetAttributes():
                try:
                    attrs[a.GetName()] = str(a.Get())
                except Exception:  # noqa: BLE001
                    attrs[a.GetName()] = "<unreadable>"
            prims[str(prim.GetPath())] = {"type": prim.GetTypeName(), "attrs": attrs}
    out["prims"] = prims
    return json.loads(json.dumps(out, default=str))


def light_prims(stage, scenes):
    """Light prims (UsdLux LightAPI) per scene (by prim path prefix), plus those outside every scene."""
    from pxr import UsdLux

    roots = [(k, str(sc.prim_path)) for k, sc in enumerate(scenes)]
    by = {str(k): [] for k, _ in roots}
    by["outside"] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        p = str(prim.GetPath())
        owner = next((str(k) for k, r in roots if p == r or p.startswith(r + "/")), "outside")
        by[owner].append({"path": p, "type": prim.GetTypeName()})
    return by


def dummy_sensor_configs(n):
    out = []
    for i in range(n):
        size = 720 if i < 3 else 480
        out.append({"sensor_type": "VisionSensor", "name": f"diag_dummy{i}", "relative_prim_path": f"/diag_dummy{i}",
                    "modalities": ["rgb", "depth_linear"], "include_in_obs": True,
                    "sensor_kwargs": {"image_height": size, "image_width": size}, "position": [0.0, 0.0, 1.2 + 0.1 * i]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--frames", type=int, default=557)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--dummies", type=int, default=0)
    ap.add_argument("--warmup-s", type=float, default=60.0)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    demo = RP.Demo(args.demo)
    n = max(1, int(args.slots))

    import omnigibson as og
    import omnigibson.utils.transform_utils as T
    from omnigibson.macros import gm
    from omnigibson.sensors.vision_sensor import VisionSensor

    for name, value in (("ENABLE_TRANSITION_RULES", False), ("ENABLE_TILED_RENDERING", False), ("VISUAL_ONLY", True)):
        if getattr(gm, name) != value:
            with gm.unlocked():
                setattr(gm, name, value)
    config = RP.env_config(demo, [demo], demo.dir, n)
    if args.dummies:
        config["env"]["external_sensors"] = dummy_sensor_configs(args.dummies)
    env = og.Environment(configs=config)
    load_s = time.time() - t_start
    robot_name = env.scenes[0].robots[0].name
    slots = [RP.Slot(k, env.scenes[k], None, robot_name) for k in range(n)]
    for s in slots:
        s.start(demo)
    stage = og.sim.stage
    diag = {"demo": demo.dir, "slots": n, "dummies": args.dummies, "load_s": round(load_s, 1), "host": os.uname().nodename,
            "pid": os.getpid(), "lights": light_prims(stage, env.scenes), "settings": {"after_load": read_settings(stage)},
            "variants": {}}
    print("DIAG lights per scene:", {k: len(v) for k, v in diag["lights"].items()}, flush=True)

    def robot_cameras(k):
        return [x for x in env.scenes[k].robots[0].sensors.values() if isinstance(x, VisionSensor)]

    dummies = [x for sensors in (env.external_sensors or []) for x in sensors.values() if isinstance(x, VisionSensor)]
    light_orig = {}
    for k, lights in diag["lights"].items():
        if k == "outside":
            continue
        for rec in lights:
            prim = stage.GetPrimAtPath(rec["path"])
            attr = prim.GetAttribute("inputs:intensity")
            if attr and attr.IsValid():
                light_orig[rec["path"]] = (int(k), attr, attr.Get())

    def set_products(enabled_slots, dummies_on):
        with og.sim.editing_usd():
            for k in range(n):
                for cam in robot_cameras(k):
                    cam.render_product.hydra_texture.updates_enabled = k in enabled_slots
            for cam in dummies:
                cam.render_product.hydra_texture.updates_enabled = bool(dummies_on)

    def set_other_lights(on):
        with og.sim.editing_usd():
            for path, (k, attr, value) in light_orig.items():
                if k != 0:
                    attr.Set(value if on else 0.0)

    def place_dummies():
        if not dummies:
            return
        pos, _ = env.scenes[0].robots[0].get_position_orientation()
        for i, cam in enumerate(dummies):
            yaw = 2 * math.pi * i / len(dummies)
            # a camera looking horizontally along world (cos yaw, sin yaw), up = world z: camera -Z = view direction
            d = th.tensor([math.cos(yaw), math.sin(yaw), 0.0])
            z = -d
            y = th.tensor([0.0, 0.0, 1.0])
            x = th.linalg.cross(y, z)
            q = T.mat2quat(th.stack([x, y, z], dim=1))
            cam.set_position_orientation(position=pos + th.tensor([0.3 * math.cos(yaw), 0.3 * math.sin(yaw), 1.2]), orientation=q)

    # first state everywhere, then the replayer's warm-up
    with og.sim.batched_usd_edits():
        for s in slots:
            s.scene.load_state(demo.state_at(0), serialized=True)
    place_dummies()
    t_w = time.time()
    n_w = 0
    while time.time() - t_w < args.warmup_s:
        og.sim.render()
        n_w += 1
    diag["warmup"] = {"s": round(time.time() - t_w, 1), "renders": n_w}

    T_frames = min(int(args.frames), demo.T)
    for vname in [v for v in args.variants.split(",") if v]:
        kind = vname.split("#")[0]
        if kind == "all":
            set_products(set(range(n)), False)
            set_other_lights(True)
        elif kind == "only0":
            set_products({0}, False)
            set_other_lights(True)
        elif kind == "only0_nolights":
            set_products({0}, False)
            set_other_lights(False)
        elif kind == "all_nolights":
            set_products(set(range(n)), False)
            set_other_lights(False)
        elif kind == "base":
            set_products({0}, False)
            set_other_lights(True)
        elif kind == "dummies":
            set_products({0}, True)
            set_other_lights(True)
        else:
            raise SystemExit(f"unknown variant {vname}")
        frames, means = {}, {}
        t_load = t_render = t_obs = 0.0
        t_v = time.time()
        for i in range(T_frames):
            t1 = time.perf_counter()
            with og.sim.batched_usd_edits():
                for s in slots:
                    s.scene.load_state(demo.state_at(i), serialized=True)
            if i == 0:
                place_dummies()
            t2 = time.perf_counter()
            for _ in range(10 if i == 0 else 1):
                og.sim.render()
            t3 = time.perf_counter()
            obs_list, _ = env.get_obs()
            t4 = time.perf_counter()
            o = obs_list[0]
            for key, v in o.items():
                if not key.startswith(robot_name + "::") or not key.endswith("::rgb") or not isinstance(v, th.Tensor):
                    continue
                rgb = v[..., :3]
                means.setdefault(key, []).append(float(rgb.float().mean()))
                if i % args.stride == 0:
                    frames.setdefault(key, []).append(rgb.detach().cpu().numpy().astype(np.uint8))
            t_load += t2 - t1
            t_render += t3 - t2
            t_obs += t4 - t3
        cams = sorted(frames)
        np.savez_compressed(os.path.join(args.out, f"{vname.replace('#', '_')}.npz"),
                            **{f"frames::{c}": np.stack(frames[c]) for c in cams}, **{f"means::{c}": np.array(means[c]) for c in cams},
                            frame_index=np.arange(0, T_frames, args.stride))
        rec = {"frames": T_frames, "stride": args.stride, "wall_s": round(time.time() - t_v, 1), "cameras": cams,
               "ms_per_tick": {"load": round(1000 * t_load / T_frames, 1), "render": round(1000 * t_render / T_frames, 1),
                               "obs": round(1000 * t_obs / T_frames, 1)},
               "mean_rgb": {c: round(float(np.mean(means[c])), 3) for c in cams}}
        diag["variants"][vname] = rec
        diag["settings"][f"after_{vname}"] = read_settings(stage)
        print("DIAG_VARIANT " + json.dumps({"variant": vname, **rec}), flush=True)
        with open(os.path.join(args.out, "diag.json"), "w") as f:
            json.dump(diag, f, indent=1)
    print("DIAG_DONE " + json.dumps({"load_s": diag["load_s"], "variants": list(diag["variants"])}), flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    with contextlib.suppress(Exception):
        import omnigibson as og

        og.shutdown()
    os._exit(rc)
