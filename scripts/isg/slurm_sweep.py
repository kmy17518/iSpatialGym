#!/usr/bin/env python3
"""first_demo runs on a Slurm partition: one sbatch job per run (sample one variant -> ``run_isg --mode oracle --video``),
all of them in parallel.

    python scripts/isg/slurm_sweep.py submit --variants A B ... | --all  [--copies N] [--sweep NAME] [--note "..."]
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
import glob
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
#: the 28 first-demo variants of first-demo-v2.0 (runlogs/first-demo-v2.0/manifest.json)
V2_VARIANTS = [
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


def budget(variant):
    return BUDGET_OVERRIDES.get(variant, DEFAULT_BUDGET)


def chain(variant, max_attempts, seed, sample_only=False):
    """first_demo ``cmd_run``'s command chain (``$SMOKE_OUT`` = the run directory)."""
    cmd = (f"python -m ispatialgym.sampling.sample_isg --variant {variant} --templates 1 --instances-per-template 1 --max-attempts {max_attempts} "
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
    stage = ""
    if args.stage_datasets:
        # one copy per node, made by the first job that lands there (the others wait on the lock); OmniGibson then reads the
        # scenes and objects from the local disk instead of NFS
        src = os.path.realpath(os.path.join(REPO, "Behavior-1K", "datasets"))
        dst = args.stage_datasets
        stage = f"""mkdir -p {q(os.path.dirname(dst.rstrip('/')))}
( flock 9; [ -f {q(dst)}/.isg_staged ] || {{ rsync -aL --delete {q(src)}/ {q(dst)}/ && touch {q(dst)}/.isg_staged; }} ) 9>{q(dst.rstrip('/') + '.lock')}
[ -f {q(dst)}/.isg_staged ] && export OMNIGIBSON_DATA_PATH={q(dst)}
"""
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
{stage}gpu="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
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


def prod_job_script(job_dir, variant, start, count, max_attempts, args):
    """One production job (``produce``): ``prod_worker.py`` for instances ``[start, start + count)`` of ``variant`` - the sampler
    plus ``--demo-workers`` concurrent demos on the job's GPU. The job works in a private copy of the committed reach bands
    (``cp -p``: the loader's newest-file tie break is unchanged) because the samplers write feasible sets back into the band
    files they read, in place."""
    q = shlex.quote
    min_kb = int(args.min_scratch_gb * 1024 * 1024)
    appdata = 'export OMNIGIBSON_APPDATA_PATH="$TMPDIR/appdata"\n' if args.local_appdata else ""
    data = q(os.path.realpath(args.data_path or os.path.join(REPO, "Behavior-1K", "datasets")))
    if getattr(args, "snapshot_link", None):
        # everything but the job's parameters lives in the snapshot's scripts/isg/prod_job.sh
        env = {"PROD_VARIANT": variant, "PROD_START": start, "PROD_COUNT": count, "PROD_OUT": job_dir, "PROD_SEED": args.seed,
               "PROD_MAX_ATTEMPTS": max_attempts, "PROD_DEMO_WORKERS": args.demo_workers, "PROD_DEMO_MODE": args.demo_mode,
               "PROD_HOURS": int(args.hours), "PROD_GRACE_MIN": int(args.grace_min), "PROD_THREADS": int(args.cpus_per_proc),
               "PROD_HOME": args.home, "PROD_CONDA_SH": args.conda_sh, "PROD_ENV": args.env,
               "PROD_DATA_PATH": os.path.realpath(args.data_path or os.path.join(REPO, "Behavior-1K", "datasets")),
               "PROD_FALLBACK_TMP": args.fallback_tmp, "PROD_MIN_SCRATCH_KB": min_kb}
        return ("#!/bin/bash\n" + f"export PROD_SNAP=\"$(readlink -f {q(args.snapshot_link)})\"\n"
                + "".join(f"export {k}={q(str(v))}\n" for k, v in env.items())
                + 'exec bash "$PROD_SNAP/ispatialgym/scripts/isg/prod_job.sh"\n')
    resolve = f"CODE={q(os.path.abspath(args.code_root or REPO))}\n"
    snapshot = ""
    if args.b1k_root:
        # a frozen checkout of both repos: its packages shadow the env's editable installs (their finders come after sys.path)
        b1k = q(os.path.abspath(args.b1k_root))
        snapshot = (f'export PYTHONPATH="$CODE":{b1k}/OmniGibson:{b1k}/bddl3${{PYTHONPATH:+:$PYTHONPATH}}\n'
                    f'export OMNIGIBSON_DATA_PATH={data}\n')
    threads = int(args.cpus_per_proc)
    worker = ["--variant", variant, "--start", str(start), "--count", str(count), "--out", job_dir, "--seed", str(args.seed),
              "--max-attempts", str(max_attempts), "--demo-workers", str(args.demo_workers), "--demo-mode", args.demo_mode]
    return f"""#!/bin/bash
export HOME={q(args.home)}
if [ -n "${{TMPDIR:-}}" ] && [ -d "$TMPDIR" ] && [ -w "$TMPDIR" ] && [ "$(df -Pk "$TMPDIR" | awk 'NR==2{{print $4}}')" -gt {min_kb} ]; then
  JOB_TMP="$TMPDIR/isg_$SLURM_JOB_ID"
elif [ -w /dev/shm ] && [ "$(df -Pk /dev/shm | awk 'NR==2{{print $4}}')" -gt {min_kb} ]; then
  JOB_TMP="/dev/shm/isg_$SLURM_JOB_ID"  # a node whose /tmp is full: RAM-backed scratch (charged to the job's --mem)
else
  JOB_TMP={q(args.fallback_tmp)}/isg_$SLURM_JOB_ID
fi
source {q(args.conda_sh)}
conda activate {q(args.env)}
{resolve}cd "$CODE" || exit 2
{snapshot}export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES
export TMPDIR="$JOB_TMP"
mkdir -p "$TMPDIR/reach_bands"
cp -p "$CODE"/runlogs/reach_bands_20260915/bands/* "$TMPDIR/reach_bands/" 2>/dev/null
export ISPATIALGYM_REACH_BANDS="$TMPDIR/reach_bands" ISPATIALGYM_CACHE_DIR="$TMPDIR/isg_cache" ISPATIALGYM_SAMPLING_CACHE="$TMPDIR/isg_cache/sampling" \\
  ISPATIALGYM_HELD_HOLD_CACHE="$TMPDIR/isg_cache/held_hold.json"
export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads}
{appdata}echo "PROD_JOB $SLURM_JOB_ID host=$(hostname -s) gpu=\\"$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)\\" TMPDIR=$TMPDIR code=$CODE start=$(date -Is)"
deadline=$(( $(date +%s) + {int(args.hours * 3600) - int(args.grace_min * 60)} ))
python "$CODE/scripts/isg/prod_worker.py" {' '.join(q(w) for w in worker)} --deadline $deadline
rc=$?
echo "PROD_JOB_END $SLURM_JOB_ID rc=$rc end=$(date -Is)"
case "$JOB_TMP" in /vision/*) ;; *) rm -rf "$JOB_TMP" ;; esac
exit $rc
"""


def cmd_produce(args):
    """Submit ``--per-task`` instances + demos per variant as jobs of ``--slice`` instances each, or the jobs of ``--plan`` (see
    :func:`prod_job_script`)."""
    variants = list(V2_VARIANTS) if args.all else list(args.variants or [])
    if not variants and not args.plan:
        raise SystemExit("slurm_sweep produce: --variants, --all or --plan")
    sweep = args.sweep or time.strftime("prod_%m%d-%H%M%S")
    root = os.path.join(args.root, sweep)
    if os.path.exists(os.path.join(root, "manifest.json")):
        raise SystemExit(f"slurm_sweep produce: {root} exists")
    os.makedirs(root, exist_ok=True)
    code = os.path.abspath(args.code_root or REPO)
    git = lambda *a: subprocess.run(["git", "-C", code, *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--", "ispatialgym", "scripts"))
    b1k_commit = subprocess.run(["git", "-C", args.b1k_root or os.path.join(REPO, "Behavior-1K"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    m = {"sweep": sweep, "kind": "produce", "submitted": FDM.now(), "code_root": code, "code_commit": commit, "dirty": dirty,
         "b1k_root": args.b1k_root, "b1k_commit": b1k_commit, "demo_mode": args.demo_mode, "env": args.env, "note": args.note,
         "per_task": args.per_task, "slice": args.slice, "seed": args.seed, "demo_workers": args.demo_workers, "hours": args.hours,
         "sbatch": {"partition": args.partition, "account": args.account, "gres": args.gres, "cpus_per_proc": args.cpus_per_proc,
                    "mem_per_proc": args.mem_per_proc, "exclude": args.exclude}, "jobs": []}
    m["args"] = {k: v for k, v in vars(args).items() if k != "fn"}
    if args.plan:
        # --plan: {"classes": {name: {gres, constraint, cpus_per_proc, mem_per_proc, demo_workers}}, "jobs": [{variant, start, count, class}]}
        plan = json.load(open(args.plan))
        m["plan"] = plan
        jobs = [(j["variant"], int(j["start"]), int(j["count"]), plan["classes"][j["class"]], j["class"]) for j in plan["jobs"]]
    else:
        base = {"gres": args.gres, "constraint": args.constraint, "cpus_per_proc": args.cpus_per_proc, "mem_per_proc": args.mem_per_proc,
                "demo_workers": args.demo_workers}
        jobs = [(variant, start, min(args.slice, args.first_index + args.per_task - start), base, None)
                for variant in variants for start in range(args.first_index, args.first_index + args.per_task, args.slice)]
    for variant, start, count, cls, cls_name in jobs:
        max_attempts = args.max_attempts or budget(variant)[0]
        jargs = argparse.Namespace(**{**vars(args), **{k: v for k, v in cls.items() if k != "constraint"}})
        nproc = 1 + int(jargs.demo_workers)
        mem_gb = float(str(jargs.mem_per_proc).rstrip("Gg"))
        job_dir = os.path.join(root, variant, f"i{start:04d}")
        os.makedirs(job_dir, exist_ok=True)
        script = os.path.join(job_dir, "job.sh")
        with open(script, "w") as f:
            f.write(prod_job_script(job_dir, variant, start, count, max_attempts, jargs))
        name = "isgp_" + re.sub(r"[^A-Za-z0-9_.-]", "_", variant)[:36] + f"_{start}"
        cmd = ["sbatch", "--parsable", "-p", args.partition, "-A", args.account, f"--gres={jargs.gres}", "-c", str(jargs.cpus_per_proc * nproc),
               f"--mem={int(mem_gb * nproc)}G", f"--time={int(args.hours * 60)}", "-J", name, "-o", os.path.join(job_dir, "slurm.log")]
        if cls.get("constraint"):
            cmd.append(f"--constraint={cls['constraint']}")
        if args.exclude:
            cmd.append(f"--exclude={args.exclude}")
        cmd.append(script)
        r = subprocess.run(cmd, capture_output=True, text=True, env=slurm_env())
        if r.returncode != 0:
            raise SystemExit(f"sbatch failed for {variant} [{start}, {start + count}): {r.stderr.strip()}")
        job_id = r.stdout.strip().split(";")[0]
        m["jobs"].append({"variant": variant, "start": start, "count": count, "max_attempts": max_attempts, "dir": job_dir, "job_id": job_id,
                          "class": cls_name})
        print(f"{job_id}  {variant}  [{start}, {start + count})  {cls_name or ''}")
        with open(os.path.join(root, "manifest.json"), "w") as f:
            json.dump(m, f, indent=1)
    print(f"produce {sweep}: {len(m['jobs'])} jobs -> {root}")


def cmd_requeue(args):
    """Cancel the produce run's jobs that have not started and submit them again with a regenerated ``job.sh`` (sbatch keeps the
    script it was given, so a pending job would otherwise run the code it was submitted with): ``--snapshot-link`` makes the
    new scripts resolve a snapshot symlink when they start. Running and ended jobs are left alone."""
    root = os.path.join(args.root, args.sweep)
    mpath = os.path.join(root, "manifest.json")
    m = json.load(open(mpath))
    overrides = {k: v for k, v in (("snapshot_link", args.snapshot_link), ("demo_mode", args.demo_mode)) if v}
    pargs = argparse.Namespace(**{**m["args"], **overrides})
    classes = (m.get("plan") or {}).get("classes") or {}
    states = job_states([j["job_id"] for j in m["jobs"]])
    n = 0
    for j in m["jobs"]:
        state, elapsed, _ = states.get(j["job_id"], ("?", "", ""))
        # --cancelled: jobs cancelled before they started; --empty: jobs that ended without writing a single instance (their
        # directory moves aside: the sampler skips a template directory that exists)
        never_ran = args.cancelled and state == "CANCELLED" and elapsed in ("00:00:00", "0:00", "")
        empty = (args.empty and state in ENDED and state != "CANCELLED"
                 and not glob.glob(os.path.join(j["dir"], "inst*", "*", "*", "instances", "*.summary.json")))
        if not (state == "PENDING" or never_ran or empty):
            continue
        if state == "PENDING":
            subprocess.run(["scancel", j["job_id"]], env=slurm_env())
        if empty:
            os.rename(j["dir"], f"{j['dir']}.empty-{j['job_id']}")
            os.makedirs(j["dir"])
        cls = classes.get(j.get("class")) or {"gres": pargs.gres, "constraint": pargs.constraint, "cpus_per_proc": pargs.cpus_per_proc,
                                               "mem_per_proc": pargs.mem_per_proc, "demo_workers": pargs.demo_workers}
        jargs = argparse.Namespace(**{**vars(pargs), **{k: v for k, v in cls.items() if k != "constraint"}})
        nproc = 1 + int(jargs.demo_workers)
        script = os.path.join(j["dir"], "job.sh")
        with open(script, "w") as f:
            f.write(prod_job_script(j["dir"], j["variant"], j["start"], j["count"], j["max_attempts"], jargs))
        name = "isgp_" + re.sub(r"[^A-Za-z0-9_.-]", "_", j["variant"])[:36] + f"_{j['start']}"
        cmd = ["sbatch", "--parsable", "-p", pargs.partition, "-A", pargs.account, f"--gres={jargs.gres}", "-c", str(jargs.cpus_per_proc * nproc),
               f"--mem={int(float(str(jargs.mem_per_proc).rstrip('Gg')) * nproc)}G", f"--time={int(pargs.hours * 60)}", "-J", name,
               "-o", os.path.join(j["dir"], "slurm.log")]
        if cls.get("constraint"):
            cmd.append(f"--constraint={cls['constraint']}")
        if pargs.exclude:
            cmd.append(f"--exclude={pargs.exclude}")
        cmd.append(script)
        r = subprocess.run(cmd, capture_output=True, text=True, env=slurm_env())
        if r.returncode != 0:
            print(f"sbatch failed for {j['variant']} {j['start']}: {r.stderr.strip()}")
            continue
        j.setdefault("previous_job_ids", []).append(j["job_id"])
        j["job_id"] = r.stdout.strip().split(";")[0]
        n += 1
    m["args"].update(overrides)
    with open(mpath, "w") as f:
        json.dump(m, f, indent=1)
    print(f"requeued {n} jobs of {args.sweep}")


_SNAP_FACTS = {}


def snapshot_facts(snap):
    """Commits and behaviour of a produce snapshot: queue-mode scene restore in its run_isg, the change-detection switch in
    its OmniGibson and the default its prod_job.sh gives it (None: no prod_job.sh)."""
    if snap not in _SNAP_FACTS:
        git = lambda root: subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None  # noqa: E731
        read = lambda p: open(p).read() if os.path.exists(p) else ""  # noqa: E731
        job_sh = read(os.path.join(snap, "ispatialgym", "scripts", "isg", "prod_job.sh"))
        m = re.search(r"PROD_TRACK_STATE_UPDATES:-(\d)", job_sh)
        _SNAP_FACTS[snap] = {
            "isg_commit": git(os.path.join(snap, "ispatialgym")), "b1k_commit": git(os.path.join(snap, "Behavior-1K")),
            "queue_scene_restore": "scene_state0" in read(os.path.join(snap, "ispatialgym", "ispatialgym", "oracle", "run_isg.py")),
            "switch": "TRACK_STATE_UPDATES" in read(os.path.join(snap, "Behavior-1K", "OmniGibson", "omnigibson", "macros.py")),
            "prod_job_default": None if m is None else m.group(1) == "1",
        }
    return _SNAP_FACTS[snap]


def job_provenance(j):
    """Snapshot, commits and settings a produce job ran with, from its slurm.log (the PROD_JOB line) or its job.sh."""
    log = os.path.join(j["dir"], "slurm.log")
    text = open(log, errors="replace").read(200000) if os.path.exists(log) else ""
    line = next((ln for ln in text.splitlines() if ln.startswith("PROD_JOB ")), "")
    code = re.search(r" code=(\S+)", line)
    started = re.search(r" start=(\S+)", line)
    if code:
        snap, source = os.path.dirname(code.group(1)), "slurm.log"
    else:
        sh = open(os.path.join(j["dir"], "job.sh")).read() if os.path.exists(os.path.join(j["dir"], "job.sh")) else ""
        hit = re.search(r"(/\S*/isg_prod/code_[0-9a-f]{8})", sh) if line else None  # a fat job.sh with the snapshot written in
        if not hit:
            return {"snapshot": None, "source": "not started" if not line else "unknown"}
        snap, source = hit.group(1), "job.sh"
    f = snapshot_facts(snap)
    logged = re.search(r" track_state_updates=(\d)", line)
    if logged:
        track = logged.group(1) == "1"
    elif not f["switch"]:
        track = True  # an OmniGibson without the switch always tracks
    elif source == "slurm.log" and "appdata=" in line and f["prod_job_default"] is not None:
        track = f["prod_job_default"]  # thin job.sh: prod_job.sh's default (job.sh never sets PROD_TRACK_STATE_UPDATES)
    else:
        track = True  # a fat job.sh never set OMNIGIBSON_TRACK_STATE_UPDATES: the macro's default
    return {"snapshot": os.path.basename(snap), "snapshot_path": snap, "isg_commit": f["isg_commit"], "b1k_commit": f["b1k_commit"],
            "track_state_updates": track, "queue_scene_restore": f["queue_scene_restore"], "started": started.group(1) if started else None,
            "source": source}


def _rewrite_json(path, fn, jsonl=False, tries=5):
    """Rewrite a (jsonl) file through ``fn`` atomically; retried when the file grows meanwhile (a runner appending a row)."""
    for _ in range(tries):
        size = os.path.getsize(path)
        with open(path) as fh:
            data = [json.loads(ln) for ln in fh if ln.strip()] if jsonl else json.load(fh)
        new = fn(data)
        if new is None:
            return False
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            if jsonl:
                fh.writelines(json.dumps(r) + "\n" for r in new)
            else:
                json.dump(new, fh, indent=1)
        if os.path.getsize(path) == size:
            os.replace(tmp, path)
            return True
        os.remove(tmp)
    return False


def cmd_provenance(args):
    """Per-job provenance into the manifest, provenance + provisional into every results row and demo meta.json lacking them,
    and <root>/<sweep>/demos_index.jsonl (one row per demo, with the scene audit of --audit files when given)."""
    root = os.path.join(args.root, args.sweep)
    mpath = os.path.join(root, "manifest.json")
    m = json.load(open(mpath))
    audit = {}
    for p in sorted(glob.glob(args.audit)) if args.audit else []:
        for ln in open(p):
            if ln.strip():
                a = json.loads(ln)
                audit[(a["job_dir"], a["instance"])] = "no_hdf5" if a.get("hdf5_error") else ("disturbed" if a.get("disturbed") else "clean")
    n_rows = n_meta = 0
    latest = {}
    for j in m["jobs"]:
        prov = job_provenance(j)
        j["provenance"] = prov
        rp = os.path.join(j["dir"], "results.jsonl")
        if not prov.get("snapshot") or not os.path.exists(rp):
            continue
        base = {k: prov[k] for k in ("snapshot", "isg_commit", "b1k_commit", "track_state_updates", "queue_scene_restore")}

        def fill(rows):
            changed = False
            for r in rows:
                if "provenance" not in r or r["provenance"] is None:
                    r["provenance"] = dict(base, source="backfill:" + prov["source"])
                    changed = True
                if "provisional" not in r:
                    r["provisional"] = bool(r.get("reused_world")) and not r["provenance"].get("queue_scene_restore", False)
                    changed = True
            return rows if changed else None

        before = sum(1 for ln in open(rp) if ln.strip())
        if _rewrite_json(rp, fill, jsonl=True):
            n_rows += before
        for r in (json.loads(ln) for ln in open(rp) if ln.strip()):
            meta = os.path.join(r.get("out") or "", "meta.json")
            if r.get("out") and os.path.exists(meta):
                def fill_meta(mm, r=r):
                    if "provenance" in mm:
                        return None
                    mm["provenance"] = r["provenance"]
                    mm.setdefault("reused_world", bool(r.get("reused_world")))
                    mm["provisional"] = bool(r.get("provisional"))
                    return mm
                if _rewrite_json(meta, fill_meta):
                    n_meta += 1
            pr = r.get("provenance") or {}
            b = r.get("behavior_lerobot") or {}
            # one entry per demo: a requeued slice appends the row of a demo recorded again, the latest row stands
            latest[(j["dir"], r["instance"])] = {
                "variant": j["variant"], "job_id": r.get("job_id") or j["job_id"], "job_dir": j["dir"], "instance": r["instance"], "success": bool(r.get("success")),
                "status": r.get("status"), "reason": r.get("reason"), "n_steps": r.get("n_steps"), "reused_world": bool(r.get("reused_world")),
                "provisional": bool(r.get("provisional")), "snapshot": pr.get("snapshot") and os.path.basename(str(pr.get("snapshot"))),
                "isg_commit": pr.get("isg_commit"), "b1k_commit": pr.get("b1k_commit"), "track_state_updates": pr.get("track_state_updates"),
                "queue_scene_restore": pr.get("queue_scene_restore"), "hdf5_ok": r.get("hdf5_ok"),
                "behavior_shard": b.get("shard"), "behavior_episode": b.get("episode_index"), "behavior_status": b.get("status"),
                "audit": audit.get((j["dir"], r["instance"]), "unchecked"), "out": r.get("out")}
    index = list(latest.values())
    m["code_commit_note"] = "code_commit is the checkout that submitted the first jobs; each job's code is jobs[].provenance"
    with open(mpath + ".tmp", "w") as fh:
        json.dump(m, fh, indent=1)
    os.replace(mpath + ".tmp", mpath)
    with open(os.path.join(root, "demos_index.jsonl.tmp"), "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in index)
    os.replace(os.path.join(root, "demos_index.jsonl.tmp"), os.path.join(root, "demos_index.jsonl"))
    by_snap = {}
    for j in m["jobs"]:
        s = (j.get("provenance") or {}).get("snapshot") or (j.get("provenance") or {}).get("source")
        by_snap[s] = by_snap.get(s, 0) + 1
    print(f"jobs per snapshot: {by_snap}")
    print(f"results rows backfilled: {n_rows}; meta.json backfilled: {n_meta}; index: {len(index)} demos "
          f"({sum(r['provisional'] for r in index)} provisional, {sum(not r['track_state_updates'] for r in index if r['track_state_updates'] is not None)} with change detection off)")


def cmd_pstatus(args):
    """Per variant: instances sampled, demos finished / successful, jobs by state."""
    root = os.path.join(args.root, args.sweep)
    m = json.load(open(os.path.join(root, "manifest.json")))
    states = job_states([j["job_id"] for j in m["jobs"]])
    per = {}
    for j in m["jobs"]:
        a = per.setdefault(j["variant"], {"jobs": {}, "inst": 0, "demos": 0, "ok": 0, "steps": []})
        st = states.get(j["job_id"], ("?",))[0]
        a["jobs"][st] = a["jobs"].get(st, 0) + 1
        a["inst"] += len(glob.glob(os.path.join(j["dir"], "inst", "*", "*", "instances", "*.summary.json")))
        rp = os.path.join(j["dir"], "results.jsonl")
        if os.path.exists(rp):
            for line in open(rp):
                if line.strip():
                    row = json.loads(line)
                    a["demos"] += 1
                    a["ok"] += int(bool(row.get("success")))
    tot = {"inst": 0, "demos": 0, "ok": 0}
    for variant, a in sorted(per.items()):
        for k in tot:
            tot[k] += a[k]
        print(f"{a['ok']:>4} ok / {a['demos']:>4} demos / {a['inst']:>4} inst  {' '.join(f'{k}:{v}' for k, v in sorted(a['jobs'].items())):<34} {variant}")
    print(f"total: {tot['ok']} successful demos, {tot['demos']} demos, {tot['inst']} instances ({m['per_task']} per task requested)")


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
    variants = list(V2_VARIANTS) if args.all else list(args.variants or [])
    if not variants:
        raise SystemExit("slurm_sweep submit: --variants or --all")
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
    for k, variant in enumerate(a for a in variants for _ in range(args.copies)):
        max_attempts, timeout = budget(variant)
        max_attempts = args.max_attempts or max_attempts
        timeout = args.timeout or timeout
        st = FDM.subtask_of(variant, customs)
        run_id = f"{stamp}-s{k:02d}"
        while os.path.exists(os.path.join(FDM.FD, st, run_id)):  # another sweep submitted in the same second
            run_id += "b"
        run_dir = os.path.join(FDM.FD, st, run_id)
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "chain.sh"), "w") as f:
            f.write(chain(variant, max_attempts, args.seed, args.sample_only) + "\n")
        runs.append({"variant": variant, "subtask": st, "run_id": run_id, "out": os.path.relpath(run_dir, REPO), "job_id": None,
                     "max_attempts": max_attempts, "timeout": timeout, "seed": args.seed, "sample_only": bool(args.sample_only), "collected": False})
    pack = max(1, int(args.pack))
    mem_gb = float(str(args.mem).rstrip("Gg"))
    for i in range(0, len(runs), pack):
        group = runs[i:i + pack]
        dirs = [(os.path.join(REPO, r["out"]), r["timeout"]) for r in group]
        script = os.path.join(dirs[0][0], "job.sh")
        with open(script, "w") as f:
            f.write(job_script(dirs, args))
        name = "isg_" + re.sub(r"[^A-Za-z0-9_.-]", "_", group[0]["variant"])[:40] + (f"_x{len(group)}" if len(group) > 1 else "")
        cmd = ["sbatch", "--parsable", "-p", args.partition, "-A", args.account, f"--gres={args.gres}", "-c", str(args.cpus * len(group)),
               f"--mem={int(mem_gb * len(group))}G", f"--time={int(max(t for _, t in dirs) / 60) + 30}", "-J", name, "-o", os.path.join(dirs[0][0], "slurm.log")]
        if args.exclude:
            cmd.append(f"--exclude={args.exclude}")
        cmd.append(script)
        r = subprocess.run(cmd, capture_output=True, text=True, env=slurm_env())
        if r.returncode != 0:
            raise SystemExit(f"sbatch failed for {[g['variant'] for g in group]}: {r.stderr.strip()}")
        job_id = r.stdout.strip().split(";")[0]
        for g in group:
            g["job_id"] = job_id
            print(f"{job_id}  {g['variant']}  -> {g['out']}")
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
              f"steps={(demo or {}).get('n_steps')} wall={res.get('wall_s')}  {r['variant']}")


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
            rec = {"time": FDM.now(), "kind": ("sample-only" if r["sample_only"] else "sample+demo") + " (slurm)", "subtask": r["subtask"], "variant": r["variant"],
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
    s.add_argument("--variants", nargs="*", default=None)
    s.add_argument("--all", action="store_true", help="the 28 variants of first-demo-v2.0")
    s.add_argument("--copies", type=int, default=1)
    s.add_argument("--sweep", default=None, help="sweep name (default sweep_<timestamp>)")
    s.add_argument("--note", default="")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-attempts", type=int, default=None, help="override every variant's budget")
    s.add_argument("--timeout", type=int, default=None, help="override every variant's timeout (s)")
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
    s.add_argument("--stage-datasets", default=None, metavar="NODE_DIR",
                   help="node-local directory (e.g. /tmp/$USER/isg_datasets) that the first job on a node fills with a copy of "
                        "Behavior-1K/datasets (~35 GB) and every job then reads through OMNIGIBSON_DATA_PATH; the copy stays on the "
                        "node for later sweeps (remove it when done). The first copy is slow from NFS (~6.5 MB/s measured from "
                        "/vision on viscam12 2026-09-27, ~1.5 h): worth it only for nodes reused across many sweeps")
    s.add_argument("--local-appdata", action="store_true", help="a cold Kit appdata per run in the node scratch (~3 min of cache building "
                                                                "per launch) instead of the checkout's warm one, which every job shares on NFS")
    s.add_argument("--shared-caches", dest="cold_caches", action="store_false",
                   help="use the env's sampling / held-hold caches instead of a fresh per-job cache in the node scratch")
    s.set_defaults(fn=cmd_submit)
    s = sub.add_parser("produce", help="many instances + demos per variant: jobs of --slice instances, each a sampler with concurrent demos")
    s.add_argument("--variants", nargs="*", default=None)
    s.add_argument("--all", action="store_true", help="the 28 variants of first-demo-v2.0")
    s.add_argument("--per-task", type=int, default=100, help="instances (and demos) per variant")
    s.add_argument("--slice", type=int, default=20, help="instances per job")
    s.add_argument("--first-index", type=int, default=0, help="instance index of the first slice (a top-up run continues after an earlier one)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-attempts", type=int, default=None, help="per instance (default: the first-demo-v2.0 budget of the variant)")
    s.add_argument("--demo-workers", type=int, default=1, help="concurrent demos per job next to its sampler (GPU memory: a process peaks at ~10 GB)")
    s.add_argument("--hours", type=float, default=11.0, help="job time limit; no demo starts in the last --grace-min minutes")
    s.add_argument("--grace-min", type=float, default=40.0)
    s.add_argument("--root", default=os.path.join(os.path.dirname(REPO), "isg_prod"), help="output root (outside the checkout)")
    s.add_argument("--sweep", default=None)
    s.add_argument("--note", default="")
    s.add_argument("--partition", default="viscam")
    s.add_argument("--account", default="viscam")
    s.add_argument("--gres", default="gpu:1")
    s.add_argument("--cpus-per-proc", type=int, default=8)
    s.add_argument("--mem-per-proc", default="32G")
    s.add_argument("--exclude", default="viscam-hgx-1,viscam-hgx-2,viscam1,viscam9")
    s.add_argument("--env", default=os.environ.get("CONDA_DEFAULT_ENV", ""))
    s.add_argument("--conda-sh", default=os.path.join(os.path.dirname(os.path.dirname(os.environ.get("CONDA_EXE", "/opt/conda/bin/conda"))), "etc", "profile.d", "conda.sh"))
    s.add_argument("--home", default=os.environ.get("HOME", ""))
    s.add_argument("--fallback-tmp", default=os.environ.get("TMPDIR", "/tmp"))
    s.add_argument("--min-scratch-gb", type=float, default=50.0)
    s.add_argument("--local-appdata", action="store_true")
    s.add_argument("--demo-mode", default="queue", choices=["queue", "fresh"], help="prod_worker --demo-mode")
    s.add_argument("--constraint", default=None, help="sbatch --constraint (viscam node features: 24G / 48G, turing / ampere / ada)")
    s.add_argument("--plan", default=None, help="JSON {classes: {name: {gres, constraint, cpus_per_proc, mem_per_proc, demo_workers}}, "
                                                "jobs: [{variant, start, count, class}]}: the jobs to submit, in order (replaces "
                                                "--variants / --per-task / --slice)")
    s.add_argument("--code-root", default=None, help="iSpatialGym checkout the jobs run (default: this one); a frozen worktree keeps "
                                                     "edits made while the jobs run out of them")
    s.add_argument("--b1k-root", default=None, help="BEHAVIOR-1K checkout whose OmniGibson / bddl the jobs import (PYTHONPATH) instead "
                                                    "of the env's editable installs")
    s.add_argument("--data-path", default=None, help="OMNIGIBSON_DATA_PATH with --b1k-root (default: this checkout's Behavior-1K/datasets)")
    s.add_argument("--snapshot-link", default=None, help="instead of --code-root / --b1k-root: a symlink to <snap>/{ispatialgym,Behavior-1K} "
                                                         "that each job resolves when it starts (repoint it to roll out a new snapshot)")
    s.set_defaults(fn=cmd_produce)
    s = sub.add_parser("requeue", help="resubmit the pending jobs of a produce run with regenerated job scripts")
    s.add_argument("--sweep", required=True)
    s.add_argument("--root", default=os.path.join(os.path.dirname(REPO), "isg_prod"))
    s.add_argument("--snapshot-link", default=None, help="symlink to <snap>/{ispatialgym,Behavior-1K} the new job scripts resolve at start")
    s.add_argument("--cancelled", action="store_true", help="also resubmit jobs that were cancelled before they started")
    s.add_argument("--empty", action="store_true", help="also resubmit jobs that ended without writing an instance (e.g. a GPU crash at start)")
    s.add_argument("--demo-mode", default=None, choices=["queue", "fresh"], help="override the run's prod_worker --demo-mode")
    s.set_defaults(fn=cmd_requeue)
    s = sub.add_parser("provenance", help="per-job code provenance into the manifest, results rows, meta.json and demos_index.jsonl")
    s.add_argument("--sweep", required=True)
    s.add_argument("--root", default=os.path.join(os.path.dirname(REPO), "isg_prod"))
    s.add_argument("--audit", default=None, help="glob of check_provisional.py reports (job_dir, instance, disturbed)")
    s.set_defaults(fn=cmd_provenance)
    s = sub.add_parser("pstatus", help="progress of a produce run")
    s.add_argument("--sweep", required=True)
    s.add_argument("--root", default=os.path.join(os.path.dirname(REPO), "isg_prod"))
    s.set_defaults(fn=cmd_pstatus)
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
