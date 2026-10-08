#!/usr/bin/env python3
"""One production job: a ``sample_isg`` slice of one alias (instances ``[start, start + count)`` of its first template) with
``run_isg`` demos running next to it on the job's GPU.

    python scripts/isg/prod_worker.py --alias A --start S --count N --out DIR [--seed 0] [--max-attempts 8] [--demo-workers 2]

The sampler writes an instance's TRO, oracle sidecar and summary, the summary last. As soon as an instance's summary
exists it is replayed by first_demo's ``run_isg --mode oracle --video --video-size 540`` command in a process of its
own (a fresh world per demo, as in first_demo), up to ``--demo-workers`` at a time. ``DIR/results.jsonl`` gets one line
per demo (the ``RUN_ISG_RESULT`` record plus timing); ``DIR/sample.log`` and ``DIR/demo_logs/<instance>.log`` are the
process logs. After ``--deadline`` (epoch seconds) no demo is started and the sampler is stopped; running demos finish.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

SUMMARY_SUFFIX = ".summary.json"
TRO_SUFFIX = "-tro_state.json"


def result_record(log_path):
    """The last ``RUN_ISG_RESULT`` record of a demo log, or None."""
    rec = None
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                if line.startswith("RUN_ISG_RESULT "):
                    try:
                        rec = json.loads(line[len("RUN_ISG_RESULT "):])
                    except ValueError:
                        pass
    except OSError:
        pass
    return rec


#: the speed settings prod_job.sh exports (PROD_SPEEDUPS), recorded with the job's provenance
SPEEDUP_ENV = ("OMP_NUM_THREADS", "OMNIGIBSON_CHANGE_DETECTION_NUMPY", "ISPATIALGYM_FAST_GEOMETRY", "ISPATIALGYM_FAST_SUPPORT", "ISPATIALGYM_FAST_ASSIGN",
               "ISPATIALGYM_SHARED_GOAL_EVAL", "ISPATIALGYM_GC_FREEZE", "ISPATIALGYM_BATCHED_WORLD_SYNC", "ISPATIALGYM_GOAL_OCCLUSION")


def job_provenance():
    """The job's snapshot (PROD_SNAP from job.sh) with its commits and the change-detection / speed settings of prod_job.sh."""
    snap = os.environ.get("PROD_SNAP")

    def git(root):
        try:
            return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60).stdout.strip() or None
        except Exception:  # noqa: BLE001
            return None

    return {"snapshot": snap, "isg_commit": git(os.path.join(snap, "ispatialgym")) if snap else None,
            "b1k_commit": git(os.path.join(snap, "Behavior-1K")) if snap else None,
            "track_state_updates_env": os.environ.get("OMNIGIBSON_TRACK_STATE_UPDATES"),
            "speedups_env": {k: os.environ.get(k) for k in SPEEDUP_ENV if os.environ.get(k) is not None}}


def job_gpus():
    """The job's GPU indices (``CUDA_VISIBLE_DEVICES`` as Slurm set it), ``[]`` when unset."""
    return [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]


def proc_env(role, slot=0):
    """Environment of a child process: with two or more GPUs in the job the sampler gets the first one and the demo runners
    the others (round robin); with one GPU every process shares it."""
    env = dict(os.environ)
    gpus = job_gpus()
    if len(gpus) >= 2:
        env["CUDA_VISIBLE_DEVICES"] = gpus[0] if role == "sampler" else gpus[1 + slot % (len(gpus) - 1)]
    return env


def ready_instances(inst_root):
    """``{instance id: tro path}`` of the instances whose summary (written last) exists and whose template meta is there."""
    out = {}
    for s in glob.glob(os.path.join(inst_root, "*", "*", "instances", "*" + SUMMARY_SUFFIX)):
        iid = os.path.basename(s)[: -len(SUMMARY_SUFFIX)]
        tro = os.path.join(os.path.dirname(s), iid + TRO_SUFFIX)
        meta = os.path.join(os.path.dirname(os.path.dirname(s)), "template_meta.json")
        if os.path.exists(tro) and os.path.exists(meta):
            out[iid] = tro
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--start", type=int, required=True, help="first instance index of the slice")
    ap.add_argument("--count", type=int, required=True, help="instances in the slice")
    ap.add_argument("--out", required=True, help="job directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=8, help="per instance")
    ap.add_argument("--demo-workers", type=int, default=2)
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds: no demo starts after it and the sampler is stopped")
    ap.add_argument("--poll-s", type=float, default=10.0)
    ap.add_argument("--demo-mode", default="queue", choices=["queue", "fresh", "sample"],
                    help="queue: --demo-workers long-lived 'run_isg --queue' processes, each replaying instance after instance in one "
                         "loaded world; fresh: one run_isg process (its own world) per demo, as first_demo; sample: the sampler only")
    args = ap.parse_args(argv)
    if args.demo_mode in ("queue", "sample"):
        return run_queue_mode(args)

    out = os.path.abspath(args.out)
    inst_glob = os.path.join(out, "inst") + "*"
    logs = os.path.join(out, "demo_logs")
    os.makedirs(logs, exist_ok=True)
    results_path = os.path.join(out, "results.jsonl")
    done = done_instances(results_path)  # a requeued slice replays what has no demo standing, sampling continues after its last index
    resume = resume_sampler(out, args.start, args.count)
    t_job = time.time()
    sampler = None
    if resume is not None:
        with open(os.path.join(out, "sample.log"), "a") as slog:
            sampler = subprocess.Popen(sampler_command(args, *resume), stdout=slog, stderr=subprocess.STDOUT, env=proc_env("sampler"))
    running = {}  # iid -> (Popen, started, log path, tro)
    stopping = {"flag": False}

    def on_term(signum, _frame):
        stopping["flag"] = True
        for p in ([sampler] if sampler is not None else []) + [r[0] for r in running.values()]:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    sampler_rc = None if sampler is not None else 0
    while True:
        if args.deadline and time.time() > args.deadline and not stopping["flag"]:
            stopping["flag"] = True
            if sampler is not None and sampler.poll() is None:
                sampler.send_signal(signal.SIGTERM)
        if sampler_rc is None and sampler.poll() is not None:
            sampler_rc = sampler.returncode
        for iid, (p, t0, lp, tro) in list(running.items()):
            if p.poll() is None:
                continue
            rec = result_record(lp) or {"status": "error", "reason": "no RUN_ISG_RESULT line"}
            row = {"instance": iid, "alias": args.alias, "rc": p.returncode, "status": rec.get("status"), "success": bool(rec.get("success")),
                   "reason": rec.get("reason"), "n_steps": rec.get("n_steps"), "wall_s": round(time.time() - t0, 1), "tro": tro,
                   "out": rec.get("out"), "reused_world": False, "provisional": False, "provenance": rec.get("provenance"),
                   "hdf5_ok": rec.get("hdf5_ok"), "behavior_lerobot": rec.get("behavior_lerobot"),
                   "host": os.uname().nodename, "job_id": os.environ.get("SLURM_JOB_ID"), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with open(results_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            done.add(iid)
            del running[iid]
        if not stopping["flag"]:
            for iid, tro in sorted(ready_instances(inst_glob).items()):
                if len(running) >= args.demo_workers:
                    break
                if iid in done or iid in running:
                    continue
                lp = os.path.join(logs, iid + ".log")
                slot = len(running)
                cmd = [sys.executable, "-m", "ispatialgym.oracle.run_isg", "--instance", tro, "--mode", "oracle", "--out", os.path.join(out, "demo"),
                       *recording_args(out, slot)]
                with open(lp, "w") as f:
                    running[iid] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=proc_env("demo", slot)), time.time(), lp, tro)
        pending = [i for i in ready_instances(inst_glob) if i not in done and i not in running]
        if sampler_rc is not None and not running and (stopping["flag"] or not pending):
            break
        time.sleep(args.poll_s)

    summary = {"provenance": job_provenance(), "alias": args.alias, "start": args.start, "count": args.count, "seed": args.seed, "sampler_rc": sampler_rc,
               "n_instances": len(ready_instances(inst_glob)), "n_demos": len(done), "wall_s": round(time.time() - t_job, 1),
               "stopped": stopping["flag"]}
    with open(os.path.join(out, "worker_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("PROD_WORKER " + json.dumps(summary), flush=True)
    return 0


EXIT_QUEUE_RESTART = 75  # run_isg.EXIT_QUEUE_RESTART
SAMPLER_RESTARTS = 3
_INDEX_RE = re.compile(r"__(\d{4,})$")


def last_written_index(inst_root):
    """Highest instance index with a summary under the sampler's root, or None."""
    idx = [int(m.group(1)) for iid in ready_instances(inst_root) for m in [_INDEX_RE.search(iid)] if m]
    return max(idx) if idx else None


HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def hdf5_readable(path):
    try:
        with open(path, "rb") as f:
            return f.read(8) == HDF5_SIGNATURE
    except OSError:
        return False


def slice_rows(results_path):
    """``{instance: latest results row}`` of the job directory."""
    rows = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["instance"]] = r
    return rows


def rerecord(row):
    """A successful demo that does not count: recorded with OmniGibson's change detection off, or without a readable
    demo.hdf5 (failed demos stay failed: their instance was attempted)."""
    if not row.get("success"):
        return False
    if (row.get("provenance") or {}).get("track_state_updates") is False:
        return True
    return not hdf5_readable(os.path.join(row.get("out") or "", "demo.hdf5"))


def done_instances(results_path):
    """Instances whose demo stands: the latest row is a failure, or a success that counts (:func:`rerecord`)."""
    return {iid for iid, r in slice_rows(results_path).items() if not rerecord(r)}


def resume_sampler(out, start, count):
    """Where a job continues the slice's sampling: ``(first index, number of instances, root)``, or None when every index
    of ``[start, start + count)`` is written. A slice with instances continues after the last written index under a root of
    its own (sample_isg skips a template directory that exists and excludes its model combination from the draw)."""
    last = last_written_index(os.path.join(out, "inst") + "*")
    nxt = start if last is None or last < start else last + 1
    if nxt >= start + count:
        return None
    k = 0
    while os.path.exists(os.path.join(out, "inst" if k == 0 else f"inst_r{k}")):
        k += 1
    return nxt, start + count - nxt, os.path.join(out, "inst" if k == 0 else f"inst_r{k}")


def sampler_command(args, first, n, root):
    return [sys.executable, "-m", "ispatialgym.sampling.sample_isg", "--alias", args.alias, "--templates", "1",
            "--instances-per-template", str(n), "--start-index", str(first), "--seed", str(args.seed),
            "--max-attempts", str(args.max_attempts), "--no-write-back", "--out", root, "--log-level", "INFO"]


def recording_args(out, slot):
    """demo.hdf5 (change detection on: prod_job.sh) + one BEHAVIOR LeRobot v3 shard per runner slot (a replacement runner of
    the slot resumes it)."""
    return ["--record", "hdf5", "--behavior-shard", os.path.join(out, "lerobot", f"shard_{os.environ.get('SLURM_JOB_ID', 'local')}_{slot}")]


def release_orphan_claims(claims, results_path, live_pids, rerun=()):
    """Claims of runners that died mid-instance (their pid is gone and no result row names the instance) go back to the queue,
    and so do the claims of ``rerun`` (successful demos to record again: :func:`rerecord`)."""
    done = set(slice_rows(results_path)) - set(rerun)
    released = []
    for iid in os.listdir(claims):
        if iid in done:
            continue
        try:
            with open(os.path.join(claims, iid)) as f:
                pid = int(f.read().split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if pid not in live_pids:
            os.remove(os.path.join(claims, iid))
            released.append(iid)
    return released


def run_queue_mode(args):
    """The sampler plus ``--demo-workers`` ``run_isg --queue`` runners sharing the job's instance queue (claim files). A runner
    that exits while instances are still coming or unclaimed (another template, an error) is replaced by a fresh one."""
    out = os.path.abspath(args.out)
    inst_root, claims = os.path.join(out, "inst"), os.path.join(out, "claims")
    inst_glob = inst_root + "*"  # inst/ plus the inst_r<k>/ of restarted samplers
    logs = os.path.join(out, "demo_logs")
    os.makedirs(logs, exist_ok=True)
    os.makedirs(claims, exist_ok=True)
    done_file = os.path.join(out, "sampler.done")
    results_path = os.path.join(out, "results.jsonl")
    t_job = time.time()
    # a requeued slice: its instances stay, sampling continues after the last written index, orphaned claims (runners of the
    # previous job) and the claims of successful demos that do not count go back to the queue
    rerun = sorted(iid for iid, r in slice_rows(results_path).items() if rerecord(r))
    released = release_orphan_claims(claims, results_path, set(), rerun=rerun)
    resume = resume_sampler(out, args.start, args.count)
    sampler, sampler_cmd = None, None
    with open(os.path.join(out, "sample.log"), "a") as slog:
        slog.write(f"\nPROD_WORKER start: {len(ready_instances(inst_glob))} instances, {len(released)} claims released "
                   f"({len(rerun)} demos to record again), sampler {resume}\n")
        if resume is not None:
            sampler_cmd = sampler_command(args, *resume)
            sampler = subprocess.Popen(sampler_cmd, stdout=slog, stderr=subprocess.STDOUT, env=proc_env("sampler"))
        else:
            open(done_file, "w").close()
    runners = {}  # slot -> (Popen, log path)
    n_started = 0
    stopping = {"flag": False}

    def on_term(signum, _frame):
        stopping["flag"] = True
        for p in ([sampler] if sampler is not None else []) + [r[0] for r in runners.values()]:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    def unclaimed():
        return [iid for iid in ready_instances(inst_glob) if not os.path.exists(os.path.join(claims, iid))]

    sampler_rc = None if sampler is not None else 0
    sampler_restarts = 0
    while True:
        if args.deadline and time.time() > args.deadline and not stopping["flag"]:
            stopping["flag"] = True
            if sampler is not None and sampler.poll() is None:
                sampler.send_signal(signal.SIGTERM)
        if sampler_rc is None and sampler.poll() is not None:
            rc = sampler.returncode
            resume = resume_sampler(out, args.start, args.count)
            if rc not in (0, 2) and not stopping["flag"] and sampler_restarts < SAMPLER_RESTARTS and resume is not None:
                # a crashed sampler (GPU device lost, segfault) continues after the last instance it wrote
                sampler_restarts += 1
                with open(os.path.join(out, "sample.log"), "a") as slog:
                    slog.write(f"\nPROD_WORKER sampler exited {rc}; restart {sampler_restarts}: {resume}\n")
                    slog.flush()
                    sampler = subprocess.Popen(sampler_command(args, *resume), stdout=slog, stderr=subprocess.STDOUT, env=proc_env("sampler"))
            else:
                sampler_rc = rc
                open(done_file, "w").close()
        if args.demo_mode == "sample":
            if sampler_rc is not None:
                break
            time.sleep(args.poll_s)
            continue
        dead = False
        for slot, (p, lp) in list(runners.items()):
            if p.poll() is not None:
                del runners[slot]
                dead = True
        if dead:
            release_orphan_claims(claims, results_path, {p.pid for p, _ in runners.values()})
        work_left = bool(unclaimed()) or sampler_rc is None
        # once the sampler is done its GPU share takes one more runner (its memory ~ a runner's; on a job of two or more
        # GPUs the sampler's card would otherwise idle while the runners drain the queue)
        n_slots = args.demo_workers + (1 if sampler_rc is not None else 0)
        if not stopping["flag"] and work_left:
            for slot in range(n_slots):
                if slot in runners:
                    continue
                if sampler_rc is not None and len(unclaimed()) <= len(runners):
                    break  # the queue is drained by the runners already alive
                if slot >= args.demo_workers and len(unclaimed()) < 3:
                    break  # the extra runner loads its world first (~5 min): only for a queue worth it
                n_started += 1
                lp = os.path.join(logs, f"runner{slot}_{n_started:03d}.log")
                cmd = [sys.executable, "-m", "ispatialgym.oracle.run_isg", "--queue", inst_glob, "--claims", claims, "--done-file", done_file,
                       "--results", results_path, "--mode", "oracle", "--out", os.path.join(out, "demo"), *recording_args(out, slot),
                       "--poll-s", str(args.poll_s)]
                if args.deadline:
                    cmd += ["--deadline", str(args.deadline)]
                env = proc_env("sampler") if slot >= args.demo_workers else proc_env("demo", slot)
                with open(lp, "w") as f:
                    runners[slot] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env), lp)
        if sampler_rc is not None and not runners and (stopping["flag"] or not unclaimed()):
            break
        time.sleep(args.poll_s)

    n_demos = sum(1 for line in open(results_path) if line.strip()) if os.path.exists(results_path) else 0
    summary = {"provenance": job_provenance(), "alias": args.alias, "start": args.start, "count": args.count, "seed": args.seed, "sampler_rc": sampler_rc,
               "demo_mode": args.demo_mode, "sampler_restarts": sampler_restarts, "n_instances": len(ready_instances(inst_glob)), "n_demos": n_demos,
               "runner_processes": n_started, "claims_released_at_start": len(released), "rerecorded": rerun,
               "wall_s": round(time.time() - t_job, 1), "stopped": stopping["flag"]}
    with open(os.path.join(out, "worker_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("PROD_WORKER " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
