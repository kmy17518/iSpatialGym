#!/usr/bin/env python3
"""First head-camera frame (and proprioception) of every regenerated instance of one alias, as the collector would have
recorded it: ``run_isg``'s ``InstanceLoader`` with BEHAVIOR's sensor set (``behavior_output``: head 720 px, 40 mm), one world
per alias, every instance attached / restored in turn (the ``--queue`` path). Writes ``<out>/<iid>.png`` (720 x 720 RGB),
``<out>/<iid>.proprio.npy`` (61) and ``<out>/index.json``.

    python lab_tools/first_frames.py --alias <alias> --inst-root <job dir> --out datasets/first_frames/<alias>

Runs inside the ``ispatialgym-v2.0-vectorized`` conda env (Isaac Sim).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def tro_paths(inst_root: str):
    """``[(key, tro path)]``: strict instances (key = instance id) and lenient candidates (key = ``<iid>@a<attempt>``)."""
    out = []
    for p in sorted(glob.glob(os.path.join(inst_root, "inst*", "*", "*", "instances", "*-tro_state.json"))):
        out.append((os.path.basename(p)[: -len("-tro_state.json")], p))
    for p in sorted(glob.glob(os.path.join(inst_root, "candidates", "*", "*", "instances", "*-tro_state.json"))):
        iid = os.path.basename(p)[: -len("-tro_state.json")]
        out.append((f"{iid}@{os.path.basename(os.path.dirname(os.path.dirname(p)))}", p))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--inst-root", required=True, help="job dir of lab_tools/sample_chosen.py (inst*/ roots)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="instance ids to (re)do")
    ap.add_argument("--redo-missing-depth", action="store_true", help="also redo entries without a <iid>.depth.npy")
    ap.add_argument("--tro-glob", default=None, help="instead of <inst-root>/inst*/...: a glob of *-tro_state.json files (e.g. the cluster's copy)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    if args.tro_glob:
        paths = [(os.path.basename(p)[: -len("-tro_state.json")], p) for p in sorted(glob.glob(args.tro_glob))]
    else:
        paths = tro_paths(args.inst_root)
    todo = []
    for iid, p in paths:
        if args.only and iid not in args.only:
            continue
        done = iid in index and os.path.exists(os.path.join(args.out, iid + ".png"))
        if args.redo_missing_depth and not os.path.exists(os.path.join(args.out, iid + ".depth.npy")):
            done = False
        if done and not args.only:
            continue
        todo.append((iid, p))
    print(f"FIRST_FRAMES alias={args.alias} instances={len(paths)} todo={len(todo)}", flush=True)
    if not todo:
        return 0

    import numpy as np
    import torch as th

    from ispatialgym.oracle import run_isg as R
    from ispatialgym.oracle.recorder import save_png

    shared = {}
    t_world = time.time()
    for k, (iid, tro) in enumerate(todo):
        t0 = time.time()
        rec = {"tro": tro, "ok": False}
        try:
            loader = R.InstanceLoader(tro)
            loader.behavior_output = True
            loader.external_camera = False
            if shared.get("env") is not None:
                shared["template_dir"] = loader.template_dir  # lenient candidates: a copy of the same template under another path
                loader.attach(shared)
            else:
                loader.build_env()
                loader.share(shared)
                rec["world_load_s"] = round(time.time() - t_world, 1)
            shared["loader"] = loader
            obs, info = loader.restore()
            robot = loader.robot
            cam_key = loader.cam_key
            rgb = obs[robot.name][cam_key]["rgb"]
            rgb = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
            save_png(os.path.join(args.out, iid + ".png"), rgb)
            depth = obs[robot.name][cam_key].get("depth_linear")
            if depth is not None:
                depth = depth.cpu().numpy() if hasattr(depth, "cpu") else np.asarray(depth)
                np.save(os.path.join(args.out, iid + ".depth.npy"), np.clip(np.asarray(depth, dtype=np.float32).reshape(depth.shape[0], depth.shape[1]), 0.0, 20.0).astype(np.float16))
                rec["depth_shape"] = list(depth.shape)
            prop = obs[robot.name].get("proprio")
            if prop is not None:
                prop = prop.cpu().numpy() if hasattr(prop, "cpu") else np.asarray(prop)
                np.save(os.path.join(args.out, iid + ".proprio.npy"), np.asarray(prop, dtype=np.float32))
                rec["proprio_dim"] = int(np.asarray(prop).size)
            p, q = robot.get_position_orientation()
            rec.update({"ok": True, "rgb_shape": list(rgb.shape), "cam_key": cam_key, "robot": robot.name,
                        "robot_base": {"position": [float(v) for v in p], "orientation": [float(v) for v in q]},
                        "activity": loader.activity, "template_dir": loader.template_dir, "wall_s": round(time.time() - t0, 1)})
            print(f"FIRST_FRAMES {k + 1}/{len(todo)} {iid} ok shape={rec['rgb_shape']} proprio={rec.get('proprio_dim')} {rec['wall_s']}s", flush=True)
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)
            rec["traceback"] = traceback.format_exc()
            print(f"FIRST_FRAMES {k + 1}/{len(todo)} {iid} ERROR {e!r}", flush=True)
            traceback.print_exc()
        index[iid] = rec
        with open(index_path, "w") as f:
            json.dump(index, f, indent=1)
    try:
        import omnigibson as og

        og.shutdown()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
