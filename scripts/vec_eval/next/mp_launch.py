"""Run one instance list as K concurrent ``run_isg_vec`` processes on one GPU (campaign vec_next, 2026-10-31, option 1).

One process builds and steps its world on essentially one CPU thread; K smaller processes build in parallel and step on K cores
(lambda RTX 4090, apple, 8 scenes: 1 x 8 finished in 466 s, 2 x 4 in 249 s, 4 x 2 in 178 s; 8 x 1 ran out of the 24 GB). The
instance list is split round-robin into K shards; process k replays / evaluates shard k with ``--num-envs N`` and its own Kit
app-data folder (one folder per Kit process, never shared): a copy of ``--appdata-template`` with the texture cache linked file
by file (read-only store; a texture a process converts itself is written next to the links).

    python scripts/vec_eval/next/mp_launch.py --instances <alias.json> --procs 4 --num-envs 2 --out <dir> \\
        --appdata-template $OMNIGIBSON_APPDATA_PATH [--same-instance-shards] -- <further run_isg_vec args>

``--same-instance-shards`` gives every process the full list (the same demo in every process, for throughput measurements).
Output: <out>/p<k>/ (run_isg_vec's output of shard k), <out>/p<k>.log, <out>/mp_launch.json (per-process rc, wall, build, rates).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time


def _copy_appdata(template: str, dst: str) -> None:
    """``template`` into ``dst``: everything copied except global/cache/texturecache, whose files are symlinked."""
    if os.path.exists(os.path.join(dst, ".ready")):
        return
    tex_rel = os.path.join("global", "cache", "texturecache")
    for root, dirs, files in os.walk(template):
        rel = os.path.relpath(root, template)
        in_tex = rel == tex_rel or rel.startswith(tex_rel + os.sep)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            src, out = os.path.join(root, f), os.path.join(dst, rel, f)
            if os.path.lexists(out):
                continue
            if in_tex:
                os.symlink(src, out)
            else:
                shutil.copy2(src, out, follow_symlinks=True)
    open(os.path.join(dst, ".ready"), "w").close()


def _summary(log: str) -> dict:
    out: dict = {}
    try:
        text = open(log, errors="replace").read()
    except OSError:
        return out
    for key, pat in (("world_build_s", r"world built in ([0-9.]+)s"), ("env_steps_per_s", r"env_steps/s=([0-9.]+)"),
                     ("batch_steps_per_s", r"batch_steps/s=([0-9.]+)")):
        m = re.findall(pat, text)
        if m:
            out[key] = float(m[-1])
    out["results"] = text.count("RUN_ISG_RESULT")
    out["done"] = "RUN_ISG_VEC_DONE" in text
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instances", required=True)
    p.add_argument("--procs", type=int, required=True)
    p.add_argument("--num-envs", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--appdata-template", default=os.environ.get("OMNIGIBSON_APPDATA_PATH"))
    p.add_argument("--appdata-root", default=None, help="where the per-process copies go (default <out>/_appdata; node-local disk)")
    p.add_argument("--same-instance-shards", action="store_true")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    spec = json.load(open(a.instances))
    insts = spec["instances"]
    shards = [list(insts) for _ in range(a.procs)] if a.same_instance_shards else [insts[k::a.procs] for k in range(a.procs)]
    app_root = a.appdata_root or os.path.join(a.out, "_appdata")
    procs = []
    t0 = time.time()
    for k, shard in enumerate(shards):
        if not shard:
            continue
        shard_path = os.path.join(a.out, f"shard_{k}.json")
        with open(shard_path, "w") as f:
            json.dump({**spec, "instances": shard}, f)
        env = dict(os.environ)
        if a.appdata_template:
            app = os.path.join(app_root, f"p{k}")
            _copy_appdata(a.appdata_template, app)
            env["OMNIGIBSON_APPDATA_PATH"] = app
            env["TMPDIR"] = os.path.join(app, "tmp")
            os.makedirs(env["TMPDIR"], exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "ispatialgym.oracle.run_isg_vec", "--instances", shard_path, "--num-envs", str(a.num_envs),
               "--out", os.path.join(a.out, f"p{k}"), *extra]
        log = os.path.join(a.out, f"p{k}.log")
        procs.append((k, log, subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT), time.time()))
        print(f"MP_LAUNCH p{k}: {len(shard)} instances, {a.num_envs} envs -> {log}", flush=True)
    report = {"procs": a.procs, "num_envs": a.num_envs, "instances": len(insts), "same_instance_shards": a.same_instance_shards, "runs": {}}
    rc_all = 0
    for k, log, proc, ts in procs:
        rc = proc.wait()
        rc_all = rc_all or rc
        report["runs"][f"p{k}"] = {"rc": rc, "wall_s": round(time.time() - ts, 1), **_summary(log)}
    report["wall_s"] = round(time.time() - t0, 1)
    report["env_steps_per_s_sum"] = round(sum(r.get("env_steps_per_s", 0.0) for r in report["runs"].values()), 2)
    with open(os.path.join(a.out, "mp_launch.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("MP_LAUNCH_DONE " + json.dumps(report), flush=True)
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
