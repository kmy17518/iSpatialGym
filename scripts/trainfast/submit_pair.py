#!/usr/bin/env python3
"""trainfast packing test (2026-11-01): two fast-flow jobs of one alias on ONE GPU (scripts/job_pair.sh).

    submit_pair.py <key> <n per member> --code <worktree> --tag <tag> [--kcollect K] [--gres l40s] [--acct viscam] [--part viscam]
                   [--cpus 24] [--mem 64G] [--sample-limit-s S] [--time-s T] [--extra-env NAME=V ...] [--dry-run]

Both members' job dirs, job.env and ledger rows come from submit_fast.submit (train indices from the ledger as for any job), with its
sbatch replaced by a stub (the ledger's jobid column reads "pair"); then one sbatch of job_pair.sh with the pair's --cpus / --mem.
"""
import argparse
import os
import shlex
import subprocess
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import submit_fast as SF  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("n", type=int)
    ap.add_argument("--code", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--kcollect", type=int, default=1)
    ap.add_argument("--gres", default="l40s")
    ap.add_argument("--acct", default="viscam")
    ap.add_argument("--part", default="viscam")
    ap.add_argument("--cpus", type=int, default=24)
    ap.add_argument("--mem", default="64G")
    ap.add_argument("--sample-limit-s", type=int, default=None)
    ap.add_argument("--time-s", type=int, default=None)
    ap.add_argument("--extra-env", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    captured = []
    real_run = SF.subprocess.run

    def stub(cmd, **kw):
        captured.append(list(cmd))
        return types.SimpleNamespace(returncode=0, stdout="pair", stderr="")

    member = argparse.Namespace(key=a.key, n=a.n, flow="fast", code=a.code, gres=a.gres, acct=a.acct, part=a.part, cpus=a.cpus // 2,
                                mem=a.mem, kcollect=a.kcollect, sample_limit_s=a.sample_limit_s, time_s=a.time_s, tag=a.tag,
                                extra_env=a.extra_env, dry_run=a.dry_run)
    if a.dry_run:
        return SF.submit(member)
    SF.subprocess.run = stub
    try:
        SF.submit(member)
        SF.submit(member)
    finally:
        SF.subprocess.run = real_run
    dirs = [c[-1] for c in captured]
    cmd = captured[0][:-2]  # ... -o <log>, then the member's script and job dir
    i = cmd.index("-J")
    cmd[i + 1] = "isgp_" + cmd[i + 1][len("isgf_"):]
    cmd[cmd.index("-c") + 1] = str(a.cpus)
    cmd = [x if not x.startswith("--mem=") else f"--mem={a.mem}" for x in cmd]
    cmd += [os.path.join(SF.F, "scripts", "job_pair.sh")] + dirs
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        for d in dirs:
            os.rename(d, d + ".failed_submit")
        raise SystemExit(f"sbatch failed: {res.stderr.strip()}")
    jobid = res.stdout.strip().split(";")[0]
    for d in dirs:
        with open(os.path.join(d, "pair.txt"), "a") as fh:
            fh.write(f"PAIR_SUBMITTED {jobid} members={','.join(dirs)} cmd={' '.join(shlex.quote(x) for x in cmd)}\n")
    print(f"SUBMITTED_PAIR {jobid} {a.key} k={a.kcollect} {' '.join(os.path.basename(d) for d in dirs)} {a.acct}/{a.part} {a.gres} "
          f"cpus={a.cpus} mem={a.mem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
