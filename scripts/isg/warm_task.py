#!/usr/bin/env python3
"""The sampler and the demo of one task in ONE warm Isaac process (engineering track, owner 2026-09-27) - what ``first_demo.py run
--warm`` runs in place of its two-process chain (``sample_isg``, then ``run_isg`` on the TRO it wrote).

Stage 1 (this script): the demo still builds its own environment after ``og.clear()``, so one process saves the second process's
imports and Isaac boot (~20 s measured 2026-09-27), not the demo's scene load; taking over the sampler's loaded world (the ~2 min
saving) is the next stage and needs the resets listed in the engineering notes. Either way the demo records its start state
(``run_isg`` ``meta.start_state``), and a pristine copy of the TRO and its sidecar is kept before the demo (a successful demo
rewrites the TRO's targets and the sidecar's sha): ``<iid>_pristine-tro_state.json`` / ``<iid>_pristine.oracle.json`` next to them,
which a fresh process loads with ``run_isg --load-only`` for the comparison (``scripts/isg/start_state_compare.py``). Release evidence
stays on the two-process chain until that comparison has passed on a representative alias set.

The stdout markers are the chain's (``SAMPLING_REPORT``, ``SAMPLE_EXIT``, ``TRO=``, ``RUN_ISG_RESULT``, ``DEMO_EXIT``); the output
layout under ``$SMOKE_OUT`` is the chain's plus ``warm_timings.json``.

usage: SMOKE_OUT=<dir> python scripts/isg/warm_task.py --alias A [--scene S] [--customs C] [--max-attempts N] [--seed 0]
                                                       [--run-isg-args "--dev"] [--sample-only]
"""
import argparse
import glob
import json
import os
import shlex
import shutil
import sys
import time


def pristine_copy(tro: str) -> str:
    """Copy the TRO and its sidecar next to them as ``<iid>_pristine-...`` (same template directory, so the loader resolves it);
    returns the copied TRO."""
    base = tro[: -len("-tro_state.json")]
    dst = f"{base}_pristine-tro_state.json"
    shutil.copy2(tro, dst)
    sc = f"{base}.oracle.json"
    if os.path.exists(sc):
        with open(sc) as f:
            d = json.load(f)
        with open(f"{base}_pristine.oracle.json", "w") as f:
            json.dump(d, f, indent=1)
    return dst


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alias", required=True)
    p.add_argument("--scene", default=None)
    p.add_argument("--customs", default=None)
    p.add_argument("--max-attempts", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-isg-args", default="")
    p.add_argument("--sample-only", action="store_true")
    args = p.parse_args(argv)
    out = os.environ["SMOKE_OUT"]
    timings = {}
    t0 = time.time()
    from ispatialgym.sampling import sample_isg

    s_argv = ["--alias", args.alias, "--templates", "1", "--instances-per-template", "1", "--max-attempts", str(args.max_attempts), "--seed", str(args.seed),
              "--no-write-back", "--out", os.path.join(out, "inst"), "--log-level", "INFO"]
    if args.scene:
        s_argv += ["--scene", args.scene]
    if args.customs:
        s_argv += ["--customs", args.customs]
    try:
        code, _, _ = sample_isg.main(s_argv, shutdown=False)
    except SystemExit as e:  # an argument error
        code = int(e.code or 1)
    timings["sample_s"] = round(time.time() - t0, 1)
    print(f"SAMPLE_EXIT {code}", flush=True)
    tros = sorted(glob.glob(os.path.join(out, "inst", "*", "*", "instances", "*-tro_state.json")))
    tros = [t for t in tros if not t.endswith("_pristine-tro_state.json")]
    dcode = None
    import omnigibson as og

    try:
        if tros:
            tro = tros[0]
            print(f"TRO={tro}", flush=True)
            if not args.sample_only:
                timings["pristine"] = os.path.relpath(pristine_copy(tro), out)
                t1 = time.time()
                og.clear()  # the demo builds its own environment (stage 1); the Isaac app stays up
                timings["clear_s"] = round(time.time() - t1, 1)
                from ispatialgym.oracle import run_isg

                t2 = time.time()
                dcode = run_isg.main(["--instance", tro, "--mode", "oracle", "--out", os.path.join(out, "demo"), "--video", "--video-size", "540"]
                                     + shlex.split(args.run_isg_args))
                timings["demo_s"] = round(time.time() - t2, 1)
                print(f"DEMO_EXIT {dcode}", flush=True)
    finally:
        timings["total_s"] = round(time.time() - t0, 1)
        with open(os.path.join(out, "warm_timings.json"), "w") as f:
            json.dump(timings, f, indent=1)
        sys.stdout.flush()
        sys.stderr.flush()
        og.shutdown()
    return 0 if code == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
