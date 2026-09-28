#!/usr/bin/env python3
"""first_demo runs on a Slurm partition: one sbatch job per run (sample one alias -> ``run_isg --mode oracle --video``),
all of them in parallel.

    python scripts/isg/slurm_sweep.py submit --aliases A B ... | --all  [--copies N] [--sweep NAME] [--note "..."]
    python scripts/isg/slurm_sweep.py status --sweep NAME
    python scripts/isg/slurm_sweep.py collect --sweep NAME [--wait]     # ledger rows + board for the runs that ended

A job runs first_demo's chain (the same ``sample_isg`` / ``run_isg`` command lines; the first-demo-v2.0 sweep's budgets,
:func:`budget`) in the submitting shell's conda env and ``HOME``. ``TMPDIR`` is the node-local scratch Slurm provides
when it has room (OmniGibson decrypts every object's USD into it: on NFS the scene loads several times slower and the
``.nfs`` placeholders of still-open files break its temp-dir cleanup at shutdown). Run directories use first_demo's
layout (``runlogs/pilot_20260915/first_demo/<subtask>/<run id>/``: ``inst/``, ``demo/``, ``local.log``) plus
``chain.sh``, ``job.sh``, ``slurm.log`` and ``slurm_result.json``; the sweep manifest is
``runlogs/slurm_sweeps/<sweep>.json``. ``collect`` appends one ledger row per ended run (kind ``sample+demo (slurm)``)
from one process, then rebuilds the board.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import first_demo as FDM  # noqa: E402

REPO = FDM.REPO
SWEEPS = os.path.join(REPO, "runlogs", "slurm_sweeps")
#: first-demo-v2.0 sweep budgets (runlogs/first-demo-v2.0/pins/driver/v2_all_jobs.json): 8 attempts / 9000 s, except
BUDGET_OVERRIDES = {
    "configuration_matching-attachment-cap-lemonade_bottle-breakfast_table": (16, 9000),
    "configuration_matching-attachment-cork-wine_bottle-breakfast_table": (20, 9000),
    "inter_object_relation_satisfaction-ontop_large_scale-shampoo_dispenser-countertop-floor": (16, 14400),
}
DEFAULT_BUDGET = (8, 9000)
#: the 28 first-demo aliases of first-demo-v2.0 (runlogs/first-demo-v2.0/manifest.json)
V2_ALIASES = [
    "alignment-axial-board_game-breakfast_table",
    "alignment-world_vertical-painting_s070-walls",
    "camera_relocalization-obstructed-footstool_1",
    "camera_relocalization-standard",
    "configuration_matching-articulation_close_large_scale-dishwasher",
    "configuration_matching-articulation_close_large_scale-door",
    "configuration_matching-articulation_close_small_scale-blender-breakfast_table",
    "configuration_matching-articulation_close_small_scale-carton_cdmmwy-breakfast_table",
    "configuration_matching-articulation_open_large_scale-dishwasher",
    "configuration_matching-articulation_open_large_scale-door",
    "configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table",
    "configuration_matching-articulation_open_small_scale-carton_cdmmwy_lever-breakfast_table",
    "configuration_matching-attachment-cap-lemonade_bottle-breakfast_table",
    "configuration_matching-attachment-cork-wine_bottle-breakfast_table",
    "configuration_matching-attachment-lid-teapot-breakfast_table",
    "configuration_matching-block_design-block_4-breakfast_table_board",
    "disembedding-bar_soap-bar_soap_2_biscuit_1-breakfast_table",
    "disembedding-bottle_of_milk-bottle_of_milk_1-breakfast_table",
    "inter_object_relation_satisfaction-inside-brussels_sprouts-bowl-breakfast_table",
    "inter_object_relation_satisfaction-nextto-marker-colored_pencil-breakfast_table",
    "inter_object_relation_satisfaction-ontop_large_scale-shampoo_dispenser-countertop-floor",
    "inter_object_relation_satisfaction-ontop_small_scale-apple-plate-breakfast_table",
    "inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table",
    "inter_object_relation_satisfaction-touching-box_of_sanitary_napkins-box_of_tissues-breakfast_table",
    "inter_object_relation_satisfaction-under_large_scale-sandal-breakfast_table-breakfast_table",
    "inter_object_relation_satisfaction-under_small_scale-teacup-saucer-breakfast_table",
    "mental_rotation-banana",
    "object_scaling-canned_food",
]
ENDED = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE")


def budget(alias):
    return BUDGET_OVERRIDES.get(alias, DEFAULT_BUDGET)


def chain(alias, max_attempts, seed, sample_only=False):
    """first_demo ``cmd_run``'s command chain (``$SMOKE_OUT`` = the run directory)."""
    cmd = (f"python -m ispatialgym.sampling.sample_isg --alias {alias} --templates 1 --instances-per-template 1 --max-attempts {max_attempts} "
           f"--seed {seed} --no-write-back --out $SMOKE_OUT/inst --log-level INFO; echo SAMPLE_EXIT $?; "
           f"tro=$(ls $SMOKE_OUT/inst/*/*/instances/*-tro_state.json 2>/dev/null | head -1); ")
    if sample_only:
        return cmd + "if [ -n \"$tro\" ]; then echo TRO=$tro; fi"
    return cmd + ("if [ -n \"$tro\" ]; then echo TRO=$tro; python -m ispatialgym.oracle.run_isg --instance \"$tro\" --mode oracle --out $SMOKE_OUT/demo "
                  "--video --video-size 540; echo DEMO_EXIT $?; fi")


def job_script(runs, args):
    """One sbatch job for ``runs`` = ``[(run_dir, timeout), ...]``: they run concurrently on the job's GPU (``--pack``),
    each with its own ``TMPDIR`` (and cold caches), ``local.log`` and ``slurm_result.json``."""
    q = shlex.quote
    min_kb = int(args.min_scratch_gb * 1024 * 1024)
    # after the env's activation hooks (which may point these at a shared cache): each run starts cold, like a pool pod,
    # and concurrent runs never rewrite one held_hold.json; reach bands and the reach volume stay shared
    cold = ('export ISPATIALGYM_CACHE_DIR="$TMPDIR/isg_cache" ISPATIALGYM_SAMPLING_CACHE="$TMPDIR/isg_cache/sampling" '
            'ISPATIALGYM_HELD_HOLD_CACHE="$TMPDIR/isg_cache/held_hold.json"; ') if args.cold_caches else ""
    if args.local_appdata:  # Kit's portable root (caches, logs) per run instead of the checkout's, shared by every job on NFS
        cold += 'export OMNIGIBSON_APPDATA_PATH="$TMPDIR/appdata"; '
    launches = "\n".join(f"run_one {q(d)} {int(t)} &" for d, t in runs)
    return f"""#!/bin/bash
export HOME={q(args.home)}
if [ -n "${{TMPDIR:-}}" ] && [ -d "$TMPDIR" ] && [ -w "$TMPDIR" ] && [ "$(df -Pk "$TMPDIR" | awk 'NR==2{{print $4}}')" -gt {min_kb} ]; then
  JOB_TMP="$TMPDIR/isg_$SLURM_JOB_ID"
else
  JOB_TMP={q(args.fallback_tmp)}/isg_$SLURM_JOB_ID
fi
source {q(args.conda_sh)}
conda activate {q(args.env)}
cd {q(REPO)} || exit 2
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES
gpu="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
run_one() {{
  local d="$1" t="$2"
  (
    export TMPDIR="$JOB_TMP/$(basename "$d")" SMOKE_OUT="$d"
    mkdir -p "$TMPDIR"
    {cold}t0=$(date +%s)
    timeout --kill-after=120 "$t" bash "$d/chain.sh" > "$d/local.log" 2>&1
    rc=$?
    printf '{{"job_id": "%s", "rc": %d, "wall_s": %d, "host": "%s", "gpu": "%s", "tmpdir": "%s", "packed": %d}}\\n' "$SLURM_JOB_ID" "$rc" \\
      "$(( $(date +%s) - t0 ))" "$(hostname -s)" "$gpu" "$TMPDIR" {len(runs)} > "$d/slurm_result.json"
    rm -rf "$TMPDIR"
  )
}}
{launches}
wait
rm -rf "$JOB_TMP"
"""


def slurm_env():
    """The submitting shell's environment without the SLURM_* variables of an enclosing allocation (an interactive job's
    would leak into the new job's defaults)."""
    return {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}


def manifest_path(sweep):
    return os.path.join(SWEEPS, f"{sweep}.json")


def load_manifest(sweep):
    with open(manifest_path(sweep)) as f:
        return json.load(f)


def save_manifest(m):
    os.makedirs(SWEEPS, exist_ok=True)
    tmp = manifest_path(m["sweep"]) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=1)
    os.replace(tmp, manifest_path(m["sweep"]))


def cmd_submit(args):
    aliases = list(V2_ALIASES) if args.all else list(args.aliases or [])
    if not aliases:
        raise SystemExit("slurm_sweep submit: --aliases or --all")
    customs = json.load(open(FDM.CUSTOMS))
    sweep = args.sweep or time.strftime("sweep_%m%d-%H%M%S")
    if os.path.exists(manifest_path(sweep)):
        raise SystemExit(f"slurm_sweep: sweep {sweep!r} exists ({manifest_path(sweep)})")
    commit, dirty = FDM.git("rev-parse", "HEAD"), bool(FDM.git("status", "--porcelain", "--", "ispatialgym", "scripts"))
    m = {"sweep": sweep, "submitted": FDM.now(), "code_commit": commit, "dirty": dirty, "env": args.env, "home": args.home, "note": args.note,
         "sbatch": {"partition": args.partition, "account": args.account, "gres": args.gres, "cpus": args.cpus, "mem": args.mem, "exclude": args.exclude},
         "runs": []}
    stamp = time.strftime("%m%d-%H%M%S")
    runs = []
    for k, alias in enumerate(a for a in aliases for _ in range(args.copies)):
        max_attempts, timeout = budget(alias)
        max_attempts = args.max_attempts or max_attempts
        timeout = args.timeout or timeout
        st = FDM.subtask_of(alias, customs)
        run_id = f"{stamp}-s{k:02d}"
        run_dir = os.path.join(FDM.FD, st, run_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "chain.sh"), "w") as f:
            f.write(chain(alias, max_attempts, args.seed, args.sample_only) + "\n")
        runs.append({"alias": alias, "subtask": st, "run_id": run_id, "out": os.path.relpath(run_dir, REPO), "job_id": None,
                     "max_attempts": max_attempts, "timeout": timeout, "seed": args.seed, "sample_only": bool(args.sample_only), "collected": False})
    pack = max(1, int(args.pack))
    mem_gb = float(str(args.mem).rstrip("Gg"))
    for i in range(0, len(runs), pack):
        group = runs[i:i + pack]
        dirs = [(os.path.join(REPO, r["out"]), r["timeout"]) for r in group]
        script = os.path.join(dirs[0][0], "job.sh")
        with open(script, "w") as f:
            f.write(job_script(dirs, args))
        name = "isg_" + re.sub(r"[^A-Za-z0-9_.-]", "_", group[0]["alias"])[:40] + (f"_x{len(group)}" if len(group) > 1 else "")
        cmd = ["sbatch", "--parsable", "-p", args.partition, "-A", args.account, f"--gres={args.gres}", "-c", str(args.cpus * len(group)),
               f"--mem={int(mem_gb * len(group))}G", f"--time={int(max(t for _, t in dirs) / 60) + 30}", "-J", name, "-o", os.path.join(dirs[0][0], "slurm.log")]
        if args.exclude:
            cmd.append(f"--exclude={args.exclude}")
        cmd.append(script)
        r = subprocess.run(cmd, capture_output=True, text=True, env=slurm_env())
        if r.returncode != 0:
            raise SystemExit(f"sbatch failed for {[g['alias'] for g in group]}: {r.stderr.strip()}")
        job_id = r.stdout.strip().split(";")[0]
        for g in group:
            g["job_id"] = job_id
            print(f"{job_id}  {g['alias']}  -> {g['out']}")
    m["runs"] = runs
    m["pack"] = pack
    save_manifest(m)
    print(f"sweep {sweep}: {len(runs)} runs in {len({r['job_id'] for r in runs})} jobs ({manifest_path(sweep)})")


def job_states(job_ids):
    """``{job id: (state, elapsed, node)}`` from sacct (the allocation line of each job)."""
    if not job_ids:
        return {}
    r = subprocess.run(["sacct", "-n", "-P", "-X", "-j", ",".join(job_ids), "--format=JobID,State,Elapsed,NodeList"], capture_output=True, text=True, env=slurm_env())
    out = {}
    for line in r.stdout.strip().splitlines():
        jid, state, elapsed, node = (line.split("|") + ["", "", "", ""])[:4]
        out[jid] = (state.split()[0] if state else "UNKNOWN", elapsed, node)
    return out


def summarize(run_dir):
    samp = FDM.parse_sampling(os.path.join(run_dir, "inst"))
    demo = FDM.parse_meta(os.path.join(run_dir, "demo"))
    res = {}
    rp = os.path.join(run_dir, "slurm_result.json")
    if os.path.exists(rp):
        try:
            res = json.load(open(rp))
        except Exception:  # noqa: BLE001
            res = {}
    return samp, demo, res


def cmd_status(args):
    m = load_manifest(args.sweep)
    states = job_states([r["job_id"] for r in m["runs"]])
    for r in m["runs"]:
        state, elapsed, node = states.get(r["job_id"], ("?", "", ""))
        samp, demo, res = summarize(os.path.join(REPO, r["out"]))
        print(f"{r['job_id']:>10} {state:<11} {elapsed:>9} {node:<9} n_written={samp['n_written']} demo={(demo or {}).get('status')} "
              f"steps={(demo or {}).get('n_steps')} wall={res.get('wall_s')}  {r['alias']}")


def cmd_collect(args):
    m = load_manifest(args.sweep)
    while True:
        states = job_states([r["job_id"] for r in m["runs"]])
        pending = [r for r in m["runs"] if states.get(r["job_id"], ("?",))[0] not in ENDED]
        for r in m["runs"]:
            if r["collected"] or r in pending:
                continue
            run_dir = os.path.join(REPO, r["out"])
            samp, demo, res = summarize(run_dir)
            state = states.get(r["job_id"], ("?", "", ""))
            rec = {"time": FDM.now(), "kind": ("sample-only" if r["sample_only"] else "sample+demo") + " (slurm)", "subtask": r["subtask"], "alias": r["alias"],
                   "run_id": r["run_id"], "out": r["out"], "rc": res.get("rc"), "wall_s": res.get("wall_s"), "code_commit": m["code_commit"], "dirty": m["dirty"],
                   "sampler_version": samp["sampler_version"], "sampling": {k: samp[k] for k in ("n_written", "rejections", "error", "time_s")},
                   "tro": samp["tro"] and os.path.relpath(samp["tro"], REPO), "demo": demo and {**demo, "meta": os.path.relpath(demo["meta"], REPO)},
                   "note": m.get("note", ""), "slurm": {"sweep": m["sweep"], "job_id": r["job_id"], "state": state[0], "node": res.get("host") or state[2],
                                                        "gpu": res.get("gpu"), "tmpdir": res.get("tmpdir")}}
            FDM.append_jsonl(FDM.LEDGER, rec)
            r["collected"] = True
            save_manifest(m)
        if not pending or not args.wait:
            break
        time.sleep(args.poll)
    FDM.build_board()
    cmd_status(args)
    done = [r for r in m["runs"] if r["collected"]]
    print(f"collected {len(done)}/{len(m['runs'])} runs of sweep {m['sweep']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--aliases", nargs="*", default=None)
    s.add_argument("--all", action="store_true", help="the 28 aliases of first-demo-v2.0")
    s.add_argument("--copies", type=int, default=1)
    s.add_argument("--sweep", default=None, help="sweep name (default sweep_<timestamp>)")
    s.add_argument("--note", default="")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-attempts", type=int, default=None, help="override every alias's budget")
    s.add_argument("--timeout", type=int, default=None, help="override every alias's timeout (s)")
    s.add_argument("--sample-only", action="store_true")
    s.add_argument("--partition", default="viscam")
    s.add_argument("--account", default="viscam")
    s.add_argument("--gres", default="gpu:1")
    s.add_argument("--cpus", type=int, default=12, help="per run (viscam GPU nodes: 96-128 CPUs for 8 GPUs; with 8 a node of eight runs was CPU-bound)")
    s.add_argument("--mem", default="48G")
    s.add_argument("--exclude", default="viscam-hgx-1,viscam-hgx-2,viscam1,viscam9",
                   help="nodes to avoid: H200s (no RT cores), the 11 GB 2080 Ti node, a node whose /tmp is full")
    s.add_argument("--env", default=os.environ.get("CONDA_DEFAULT_ENV", ""), help="conda env of the jobs (default: the active one)")
    s.add_argument("--conda-sh", default=os.path.join(os.path.dirname(os.path.dirname(os.environ.get("CONDA_EXE", "/opt/conda/bin/conda"))), "etc", "profile.d", "conda.sh"))
    s.add_argument("--home", default=os.environ.get("HOME", ""), help="HOME of the jobs (caches; default: the submitting shell's)")
    s.add_argument("--fallback-tmp", default=os.environ.get("TMPDIR", "/tmp"), help="TMPDIR when the node scratch is missing or short of space")
    s.add_argument("--min-scratch-gb", type=float, default=50.0)
    s.add_argument("--pack", type=int, default=1, help="runs per job, concurrently on its one GPU (a run peaks at ~10 GB: 2 on 24 GB cards, "
                                                        "3-4 on 46-48 GB); --cpus and --mem are per run")
    s.add_argument("--local-appdata", action="store_true", help="a cold Kit appdata per run in the node scratch (~3 min of cache building "
                                                                "per launch) instead of the checkout's warm one, which every job shares on NFS")
    s.add_argument("--shared-caches", dest="cold_caches", action="store_false",
                   help="use the env's sampling / held-hold caches instead of a fresh per-job cache in the node scratch")
    s.set_defaults(fn=cmd_submit)
    s = sub.add_parser("status")
    s.add_argument("--sweep", required=True)
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("collect")
    s.add_argument("--sweep", required=True)
    s.add_argument("--wait", action="store_true", help="poll until every job has ended")
    s.add_argument("--poll", type=int, default=120)
    s.set_defaults(fn=cmd_collect)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
