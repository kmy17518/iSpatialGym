#!/usr/bin/env python3
"""Parity gate item: first-frame verification unchanged. Re-renders the first frames of a few restored training instances with
lab_tools/first_frames.py (the collector's sensor set, the current evaluator code + patches) into a scratch dir and compares them
with the stored frames under datasets/first_frames_train/ (mean abs diff; identical renders differ only by RTX accumulation grain).

    python lab_tools/first_frames_check.py --instances camera_relocalization-standard__house_single_floor__1115 ... [--out lab/diagnostics/eval-fixes/first_frames_check.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDA_PY = os.path.expanduser("~/miniconda3/envs/ispatialgym-v2.0-vectorized/bin/python")
TASK = "camera_relocalization-standard"
STORED = os.path.join(WT, "datasets", "first_frames_train", TASK)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--scratch", default=os.path.join(WT, "runlogs", "diagnostics", "eval-fixes", "first_frames_check"))
    ap.add_argument("--out", default=os.path.join(WT, "lab", "diagnostics", "eval-fixes", "first_frames_check.json"))
    ap.add_argument("--threshold", type=float, default=3.0, help="mean abs diff (0-255) above which a re-render counts as changed")
    args = ap.parse_args(argv)
    from PIL import Image

    stored_index = json.load(open(os.path.join(STORED, "index.json")))
    tros = [stored_index[i]["tro"] for i in args.instances]
    os.makedirs(args.scratch, exist_ok=True)
    # one process: the TROs by glob is awkward across slices; pass them through a temporary glob-compatible list via --tro-glob per call is wasteful,
    # so run first_frames.py once with a brace-free glob covering the bundle and --only the requested ids
    glob_pat = os.path.join(WT, "datasets", "isg-reloc-train-instances", "prod_0927", TASK, "i1*", "inst", TASK, "house_single_floor", "instances", "*-tro_state.json")
    t0 = time.time()
    cmd = [CONDA_PY, os.path.join(WT, "lab_tools", "first_frames.py"), "--variant", TASK, "--inst-root", args.scratch, "--tro-glob", glob_pat, "--out", args.scratch, "--only", *args.instances]
    env = dict(os.environ, OMNIGIBSON_HEADLESS="1", OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1")
    rc = subprocess.run(cmd, cwd=WT, stdout=open(os.path.join(args.scratch, "render.log"), "w"), stderr=subprocess.STDOUT, env=env).returncode
    rows = []
    for i in args.instances:
        a = os.path.join(STORED, i + ".png")
        b = os.path.join(args.scratch, i + ".png")
        row = {"instance": i, "tro": stored_index[i]["tro"], "rendered": os.path.exists(b)}
        if os.path.exists(a) and os.path.exists(b):
            ia = np.asarray(Image.open(a).convert("RGB"), dtype=np.float32)
            ib = np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)
            row["mean_abs_diff"] = float(np.abs(ia - ib).mean()) if ia.shape == ib.shape else None
            row["unchanged"] = row["mean_abs_diff"] is not None and row["mean_abs_diff"] <= args.threshold
            pa, pb = os.path.join(STORED, i + ".proprio.npy"), os.path.join(args.scratch, i + ".proprio.npy")
            if os.path.exists(pa) and os.path.exists(pb):
                row["proprio_max_abs_diff"] = float(np.abs(np.load(pa) - np.load(pb)).max())
        rows.append(row)
    out = {"rc": rc, "wall_s": round(time.time() - t0, 1), "threshold_mean_abs_diff": args.threshold, "rows": rows, "pass": bool(rows) and all(r.get("unchanged") for r in rows), "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
