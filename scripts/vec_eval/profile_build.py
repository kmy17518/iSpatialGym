"""Profile the vectorized world build (campaign vec_next, 2026-10-29): builds ``VecInstanceLoader(num_envs=N)`` for the first
instance of an instance list exactly as run_isg_vec does, under cProfile, and prints the build time, the per-clone slope and the
hottest functions. Nothing is stepped; nothing is recorded in the record.

    python scripts/vec_eval/profile_build.py --instances <alias.json> --num-envs 2 [--layout grid-centered --scene-margin 5]
        [--out <dir>]  ->  <out>/build_profile_n<N>.{txt,pstats}
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

import omnigibson as og  # noqa: F401  (import order: before ispatialgym.oracle)

from ispatialgym.oracle.vec_loader import VecInstanceLoader  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instances", required=True)
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--layout", default="grid-centered")
    p.add_argument("--scene-margin", type=float, default=5.0)
    p.add_argument("--external-size", type=int, default=240)
    p.add_argument("--no-external-camera", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--top", type=int, default=45)
    p.add_argument("--dump-state", default=None, metavar="NPZ", help="save every scene's dump_state(serialized=True) after the build (bit-identity check of build-only changes)")
    p.add_argument("--no-profile", action="store_true", help="skip cProfile (blind past the Kit launch anyway); timers / dump only")
    args = p.parse_args()
    with open(args.instances) as f:
        spec = json.load(f)
    tro = spec["instances"][0]["tro"]
    ext = None if args.no_external_camera else (args.external_size, int(round(args.external_size * 1.5)) // 2 * 2)
    out = os.path.abspath(args.out or os.path.join(os.environ.get("ISG_VEC_ROOT", os.getcwd()), "next", "profile"))
    if args.dump_state:
        args.dump_state = os.path.abspath(args.dump_state)
    os.makedirs(out, exist_ok=True)
    loader = VecInstanceLoader(num_envs=args.num_envs, behavior_output=True, tiled_rendering=False, drop_depth=True, external_camera_hw=ext,
                               layout=args.layout, scene_margin=args.scene_margin)
    t0 = time.time()
    pr = cProfile.Profile()
    if not args.no_profile:
        pr.enable()
    loader.build_env(tro)
    if not args.no_profile:
        pr.disable()
    total = time.time() - t0
    if args.dump_state:
        import numpy as np
        states = {}
        for i, sc in enumerate(loader.env.scenes):
            st = sc.dump_state(serialized=True)
            states[f"scene_{i}"] = np.asarray(st.cpu() if hasattr(st, "cpu") else st)
        np.savez(args.dump_state, **states)
        print(f"BUILD_STATE {args.dump_state} " + " ".join(f"{k}:{v.shape[0]}" for k, v in states.items()), flush=True)
    base = os.path.join(out, f"build_profile_n{args.num_envs}")
    s = io.StringIO()
    if not args.no_profile:
        pr.dump_stats(base + ".pstats")
        st = pstats.Stats(pr, stream=s)
        st.sort_stats("cumulative").print_stats(args.top)
        st.sort_stats("tottime").print_stats(args.top)
    rep = loader.report
    from omnigibson.objects.usd_object import USDObject
    hits = getattr(USDObject, "_PREPARED_ASSETS_HITS", 0)
    n_prepared = len(getattr(USDObject, "_PREPARED_ASSETS", {}))
    head = (f"BUILD_PROFILE num_envs={args.num_envs} build_s={rep.get('build_s')} wall_s={total:.1f} objects_per_scene={len(loader.env.scenes[0].objects)} "
            f"prepare_cache={{'entries': {n_prepared}, 'hits': {hits}, 'enabled': {os.environ.get('ISG_VEC_PREPARE_CACHE', '1')}}} "
            f"lazy_systems={os.environ.get('ISG_VEC_LAZY_SYSTEMS', '1')} layout={rep.get('clone_layout')} scene_margin={rep.get('scene_margin')}")
    print(head, flush=True)
    with open(base + ".txt", "w") as f:
        f.write(head + "\n\n" + s.getvalue())
    print(f"profile written: {base}.txt", flush=True)
    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
