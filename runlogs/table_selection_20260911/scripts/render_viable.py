#!/usr/bin/env python3
"""Render the init state of one viable template per (problem, scene, support) for a visual check.

Stages the generated template directories (``<work>/<scene>/<tag>``) as ``<stage>/<problem>/<tag>`` symlinks and runs
``ispatialgym.sampling.render_instances`` on the stage (BEHAVIOR replay layout: left wrist | head | right wrist, external
over-the-shoulder camera below). The PNGs are copied to ``<out>/renders/<tag>__<iid>-init.png`` and one contact sheet
per problem to ``<out>/renders/<problem>.png``.

``--max-per-problem-scene N`` limits the supports rendered per (problem, scene) (default: all viable ones).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="/tmp/isg_table_sel")
    ap.add_argument("--out", default=ROOT)
    ap.add_argument("--problems", default=None, help="comma separated (default: all)")
    ap.add_argument("--max-per-problem-scene", type=int, default=0)
    ap.add_argument("--python", default=os.path.expanduser("~/miniconda3/envs/ispatialgym/bin/python"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stage = os.path.join(args.work, "render_stage")
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    n = 0
    wanted = set(args.problems.split(",")) if args.problems else None
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        if p.endswith("__precheck.json"):
            continue
        with open(p) as f:
            r = json.load(f)
        if r.get("status") not in ("ok", "ok_other_support") or not r.get("work_dir"):
            continue
        if wanted and r["problem"] not in wanted:
            continue
        if not os.path.exists(os.path.join(r["work_dir"], "template_meta.json")):
            continue
        pdir = os.path.join(stage, r["problem"])
        os.makedirs(pdir, exist_ok=True)
        if args.max_per_problem_scene and sum(1 for d in os.listdir(pdir) if d.startswith(r["scene"] + "__")) >= args.max_per_problem_scene:
            continue
        os.symlink(r["work_dir"], os.path.join(pdir, os.path.basename(r["work_dir"])))
        n += 1
    print(f"staged {n} templates under {stage}")
    if args.dry_run or n == 0:
        return
    env = dict(os.environ, OMNIGIBSON_HEADLESS="1")
    snap = os.path.join(args.work, "snapshot")
    if os.path.isdir(os.path.join(snap, "ispatialgym")):
        env["PYTHONPATH"] = snap + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [args.python, "-m", "ispatialgym.sampling.render_instances", "--out", stage]
    print(" ".join(cmd), flush=True)
    rc = subprocess.call(cmd, env=env, cwd="/tmp")
    print("render_instances exit", rc)
    out_dir = os.path.join(args.out, "renders")
    os.makedirs(out_dir, exist_ok=True)
    k = 0
    for png in glob.glob(os.path.join(stage, "*", "*", "instances", "*-init.png")):
        tag = os.path.basename(os.path.dirname(os.path.dirname(png)))
        shutil.copy(png, os.path.join(out_dir, f"{tag}__{os.path.basename(png)}"))
        k += 1
    for png in glob.glob(os.path.join(stage, "renders", "*.png")):
        shutil.copy(png, os.path.join(out_dir, os.path.basename(png)))
    print(f"copied {k} instance renders to {out_dir}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
