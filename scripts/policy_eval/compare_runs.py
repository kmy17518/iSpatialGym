#!/usr/bin/env python3
"""Equivalence of two runs of the same instances (e.g. world reuse: run_isg --instance-list vs a fresh world per instance; or a fresh
run against a repeat of itself, the noise floor): per instance present in both, the verdict, steps, evaluated step, final error
metric, closest approach, final tracked-object poses and the dumped observations (obs_dump.npz: proprio max |diff|, image PSNR).

    compare_runs.py <run dir A> <run dir B>
"""
import json
import math
import os
import sys

import numpy as np


def load(d):
    out = {}
    for name in sorted(os.listdir(d)):
        m = os.path.join(d, name, "meta.json")
        if os.path.isfile(m):
            out[name] = (json.load(open(m)), os.path.join(d, name))
    return out


def pose_diff(a, b):
    worst = 0.0
    for k, va in (a or {}).items():
        vb = (b or {}).get(k)
        if not isinstance(va, dict) or not isinstance(vb, dict):
            continue
        for f in ("position", "pos"):
            if f in va and f in vb:
                worst = max(worst, float(np.abs(np.asarray(va[f], float) - np.asarray(vb[f], float)).max()))
    return worst


def psnr(x, y):
    mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def main():
    A, B = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(A) & set(B))
    print(f"{len(common)} instances in both ({len(A)} / {len(B)})")
    for name in common:
        (ma, da), (mb, db) = A[name], B[name]
        ea, eb = ma.get("error_metric") or {}, mb.get("error_metric") or {}
        row = {"success": (ma.get("success"), mb.get("success")), "n_steps": (ma.get("n_steps"), mb.get("n_steps")),
               "eval_step": (ma.get("evaluation_step"), mb.get("evaluation_step")),
               "reused": (ma.get("reused_world"), mb.get("reused_world"))}
        for k in ("position_error", "orientation_error"):
            if ea.get(k) is not None and eb.get(k) is not None:
                row[k + "_diff"] = round(abs(float(ea[k]) - float(eb[k])), 6)
        ca, cb = (ma.get("closest_approach") or {}).get("score"), (mb.get("closest_approach") or {}).get("score")
        if ca is not None and cb is not None:
            row["closest_score_diff"] = round(abs(ca - cb), 6)
        row["final_pose_max_diff_m"] = round(pose_diff(ma.get("final_target_poses"), mb.get("final_target_poses")), 6)
        fa, fb = os.path.join(da, "obs_dump.npz"), os.path.join(db, "obs_dump.npz")
        if os.path.exists(fa) and os.path.exists(fb):
            xa, xb = np.load(fa), np.load(fb)
            n = min(len(xa["step"]), len(xb["step"]))
            row["proprio_max_diff"] = float(np.abs(xa["proprio"][:n] - xb["proprio"][:n]).max())
            row["psnr_head"] = [round(psnr(xa["zed_link"][k], xb["zed_link"][k]), 1) for k in range(n)]
        same = row["success"][0] == row["success"][1] and row["n_steps"][0] == row["n_steps"][1] and row["eval_step"][0] == row["eval_step"][1]
        print(("SAME " if same else "DIFF ") + name[:70], json.dumps(row))


if __name__ == "__main__":
    main()
