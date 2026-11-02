#!/usr/bin/env python3
"""Demo collection on the pod pool for the pilot's complete instances (C5: demos in parallel with the sampling fleet).

    python scripts/isg/pilot_demos.py --collected runlogs/runpod/pilot --out runlogs/pilot_20260915/demos \
        [--subtasks mental_rotation object_scaling ...] [--parallel 4] [--only-missing]

Every v2 template directory found under ``<collected>/<pod>/v2__<variant>/<variant>/<template>/`` is staged (TRO + sidecar +
template files + ``activity_definitions``, no renders) under ``runlogs/smoke_inputs/pilot_instances/<variant>/<template>/`` —
a path ``remote_run.py`` syncs — and every instance gets one ``run_isg --mode oracle --video`` on a free pool pod through
``remote_run.py`` (``--parallel`` workers); results are pulled to ``<out>/<iid>/`` (meta.json, annotations.json,
demo.hdf5, head_rgb.mp4, lerobot/). ``demos.jsonl`` in ``<out>`` records one line per run (status, success, reason, wall).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
STAGE = os.path.join(REPO, "runlogs", "smoke_inputs", "pilot_instances")
REMOTE_RUN = os.path.join(REPO, "scripts", "runpod", "remote_run.py")


def subtask_of(variant: str, customs: dict) -> str:
    e = customs.get(variant) or {}
    st = e.get("subtask")
    if st:
        return st
    iddl = e.get("iddl", "")
    fam, fn = iddl.split("/", 1) if "/" in iddl else ("", iddl)
    fn = os.path.splitext(fn)[0]
    return fam if fam in ("object_scaling", "mental_rotation", "disembedding") and fn == fam else (f"{fam}-{fn}" if fam else fn)


def stage_templates(collected: str, customs: dict, subtasks):
    """Copy every v2 template dir (without renders / validation) into the staging tree; return [(variant, template_dir, tro)]."""
    items = []
    for tdir in sorted(glob.glob(os.path.join(collected, "*", "v2__*", "*", "*", "instances"))):
        tdir = os.path.dirname(tdir)
        variant = os.path.basename(os.path.dirname(tdir))
        st = subtask_of(variant, customs)
        if subtasks and st not in subtasks:
            continue
        dst = os.path.join(STAGE, variant, os.path.basename(tdir))
        if not os.path.exists(os.path.join(dst, "template.json")):
            shutil.copytree(tdir, dst, ignore=shutil.ignore_patterns("renders", "validation", "*.png"), dirs_exist_ok=True)
        for tro in sorted(glob.glob(os.path.join(dst, "instances", "*-tro_state.json"))):
            items.append((variant, st, dst, tro))
    return items


def run_one(item, out_root: str, timeout: int):
    variant, st, tdir, tro = item
    iid = os.path.basename(tro).replace("-tro_state.json", "")
    rel_tro = os.path.relpath(tro, REPO)
    out_dir = os.path.join(out_root, iid)
    os.makedirs(out_dir, exist_ok=True)
    cmd = (f"python -m ispatialgym.oracle.run_isg --instance {rel_tro} --mode oracle --out $SMOKE_OUT --video --video-size 540; "
           f"echo DEMO_EXIT $?")
    t0 = time.time()
    with open(os.path.join(out_dir, "remote.log"), "w") as log:
        r = subprocess.run([sys.executable, REMOTE_RUN, "--timeout", str(timeout), "--pull", "$SMOKE_OUT", out_dir, "--", cmd],
                           stdout=log, stderr=subprocess.STDOUT, text=True, cwd=REPO, timeout=timeout + 3600)
    meta = None
    for m in glob.glob(os.path.join(out_dir, "*", "meta.json")):
        try:
            meta = json.load(open(m))
        except Exception:
            pass
    rec = {"iid": iid, "variant": variant, "subtask": st, "rc": r.returncode, "wall_s": round(time.time() - t0),
           "status": (meta or {}).get("status"), "success": (meta or {}).get("success"), "reason": (meta or {}).get("reason"),
           "n_steps": (meta or {}).get("n_steps"), "out": out_dir}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collected", default=os.path.join(REPO, "runlogs", "runpod", "pilot"))
    ap.add_argument("--customs", default=os.path.join(REPO, "ispatialgym", "assets", "task_specs", "isg_customs.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--subtasks", nargs="*", default=None)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--only-missing", action="store_true", help="skip instances that already have a meta.json under --out")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    customs = json.load(open(args.customs))
    items = stage_templates(args.collected, customs, args.subtasks)
    if args.only_missing:
        items = [it for it in items if not glob.glob(os.path.join(args.out, os.path.basename(it[3]).replace("-tro_state.json", ""), "*", "meta.json"))]
    if args.limit:
        items = items[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(items)} demos to run on the pool ({args.parallel} parallel)", flush=True)
    ledger = open(os.path.join(args.out, "demos.jsonl"), "a")
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for rec in ex.map(lambda it: run_one(it, args.out, args.timeout), items):
            ledger.write(json.dumps(rec) + "\n")
            ledger.flush()
            print(f"{rec['subtask'][:40]:40s} {rec['iid'][-40:]:40s} {rec['status']} {rec['reason'] or ''} {rec['wall_s']}s", flush=True)
    print("PILOT_DEMOS_DONE")


if __name__ == "__main__":
    main()
