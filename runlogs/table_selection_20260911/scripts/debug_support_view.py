#!/usr/bin/env python3
"""Debug one (scene, table, problem) template setup with head-camera renders.

Runs ``TaskSampler.setup()`` exactly like validate_tables.py, but every ``SimWorld.visible_pixels`` call additionally
saves the head camera RGB (``<out>/<tag>__<n>.png``) and prints which spawned objects the instance segmentation
lists. Used to understand 'destination tray visibility out of range (dest 0 px, source 0 px)' rejections.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from ispatialgym.sampling.spec import load_task_spec  # noqa: E402
import validate_tables as vt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--problem", default="disembedding-0")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "debug"))
    ap.add_argument("--n-instances", type=int, default=0)
    ap.add_argument("--pre-problem", default=None, help="run this problem (setup + 1 instance) first in the same process, then --problem")
    ap.add_argument("--seg-fix", action="store_true", help="apply the validate_tables seg-refresh wrapper around visible_pixels")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    spec_file = load_task_spec(vt.DEFAULT_SPEC)
    rows = vt.load_inventory(os.path.join(HERE, "..", "inventory.json"), args.scene)
    t = {r["name"]: r for r in rows}[args.table]

    from ispatialgym.sampling.robot_utils import RobotKit
    from ispatialgym.sampling.sim_utils import Reject, SimWorld
    from ispatialgym.sampling.tasks import get_sampler_class
    from ispatialgym.sampling.tasks.base import SamplerContext

    import omnigibson as og
    from PIL import Image

    rooms = [r for r in t["rooms"].split(";") if r]
    world = SimWorld(args.scene, rooms)
    kit = RobotKit(world, prebuild=("right_arm",))
    tag = f"{args.scene}__{args.table}__{args.problem}"
    counter = {"n": 0}
    orig = world.visible_pixels

    def patched(n_renders=3, exclude_structure=True):
        px = orig(n_renders, exclude_structure)
        counter["n"] += 1
        rgb = world.rgb(1).numpy().astype(np.uint8)
        path = os.path.join(args.out, f"{tag}__{counter['n']:02d}.png")
        Image.fromarray(rgb).save(path)
        seg, mapping = world.seg_instance(1)
        spawned = {n: px.get(n, 0) for n in world.spawned}
        top = sorted(px.items(), key=lambda kv: -kv[1])[:6]
        print(f"[debug] top labels by pixels: {top}", flush=True)
        listed = sorted(v for v in mapping.values() if any(v == n for n in world.spawned))
        cam_p, cam_q = world.camera_pose()
        print(f"[debug] render {counter['n']:02d}: spawned px {spawned}; spawned in seg mapping {listed}; "
              f"camera {[round(float(v), 3) for v in cam_p]} q {[round(float(v), 3) for v in cam_q]}; robot base "
              f"{[round(float(v), 3) for v in world.robot.get_position_orientation()[0]]}; saved {path}", flush=True)
        for n, o in world.spawned.items():
            p, _ = o.get_position_orientation()
            print(f"[debug]    {n}: pos {[round(float(v), 3) for v in p]} visible={o.visible} asleep={getattr(o, 'is_asleep', None)}", flush=True)
        return px

    if args.seg_fix:
        vt.install_seg_refresh(world, kit)
        orig = world.visible_pixels
    world.visible_pixels = patched

    def run(problem, n_inst):
        pspec = vt.spec_for_table(spec_file[problem], args.scene, t)
        scene_spec = pspec.scenes[args.scene]
        choices = pspec.template_choices(np.random.default_rng(0))[0]
        ptag = f"{args.scene}__{args.table}__{problem}"
        out_dir = os.path.join("/tmp/isg_table_sel/debug_work", ptag)
        os.makedirs(out_dir, exist_ok=True)
        ctx = SamplerContext(pspec, scene_spec, choices, world, kit, out_dir, ptag, spec_file.sha256, base_seed=0)
        sampler = get_sampler_class(problem)(ctx)
        t0 = time.time()
        try:
            sampler.setup()
            print(f"[debug] {problem} setup ok in {time.time() - t0:.1f}s: {json.dumps(vt.template_meta_digest(sampler.template_meta), default=str)[:1500]}", flush=True)
            if n_inst:
                written = sampler.run(n_inst, max_attempts=6)
                print(f"[debug] {problem} wrote {len(written)} instances; rejections {sampler.rejections}", flush=True)
        except Reject as e:
            print(f"[debug] {problem} setup REJECTED after {time.time() - t0:.1f}s: {e}", flush=True)
        except Exception:
            traceback.print_exc()
        finally:
            for o in list(world.spawned.values()):
                world.remove(o)
            kit.sync_world()

    try:
        if args.pre_problem:
            run(args.pre_problem, 1)
        run(args.problem, args.n_instances)
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
