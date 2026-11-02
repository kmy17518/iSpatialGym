"""The campaign verdict file of one alias's N-slot replay verification (campaign vec_next, 2026-10-29), in the shape of the record's
``$V/mirror/verify/<alias>.json`` (task, pass, num_envs, runs{n1, n8same{pass, reasons, instances}}) plus the host, the code
state and the previous record's verdict it supersedes for this code. Reads ``<runs>/{n1,n8same}/`` (``*.run.json`` written by
verify.sbatch, the episodes' meta.json, ``n8same/verify.json`` from verify_rules).

    python scripts/vec_eval/next/verify_record.py --task <alias> --runs <runs dir> --num-envs 8 --out <verify json> [--previous <mirror json>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import subprocess
import time


def _load(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def _run_summary(d):
    out = dict(_load(d + ".run.json", {}) or {})
    metas = sorted(glob.glob(os.path.join(d, "**", "meta.json"), recursive=True))
    out["episodes"] = len(metas)
    m0 = _load(metas[0]) if metas else None
    if m0:
        out["load_rooms"] = m0.get("load_rooms")
        out["levers"] = m0.get("levers")
    lr = _load(os.path.join(d, "_batches", "loader_report.json"), {}) or {}
    lr = lr.get("report", lr) if isinstance(lr, dict) else {}
    for k in ("physics_overrides", "sleep_threshold", "solver_position_iterations", "solver_velocity_iterations", "clone_layout", "scene_margin"):
        if k in lr:
            out[k] = lr[k]
    batches = sorted(glob.glob(os.path.join(d, "_batches", "batch_*.json")))
    b0 = _load(batches[0]) if batches else None
    if b0:
        out["batch0"] = {k: b0.get(k) for k in ("load_s", "run_s", "batch_steps", "batch_steps_per_s", "env_steps_per_s", "world_build_s", "build_s") if k in b0}
    log = d + ".log"
    if os.path.exists(log):
        with open(log, errors="replace") as f:
            for line in f:
                if "world built in" in line:
                    try:
                        out["world_build_s"] = float(line.split("world built in")[1].split("s")[0])
                    except Exception:  # noqa: BLE001
                        pass
                    break
    out["ok"] = out.get("rc") == 0 and out["episodes"] > 0
    out["reason"] = "" if out["ok"] else f"rc={out.get('rc')} episodes={out['episodes']}"
    return out


def _run_host(runs):
    """The host the replays ran on (their ``<phase>.run.json`` ``host``), so a verdict re-composed elsewhere keeps it."""
    for name in ("n8same.run.json", "n1.run.json"):
        h = (_load(os.path.join(runs, name), {}) or {}).get("host")
        if h:
            return h
    return None


def _overlay_files(b1k):
    """The modified / new files of the Behavior-1K checkout = the overlay state the runs imported."""
    if not b1k:
        return None
    try:
        out = subprocess.run(["git", "-C", b1k, "status", "--short", "--untracked-files=all"], capture_output=True, text=True, timeout=30).stdout
        return sorted(line[3:] for line in out.splitlines() if line.strip() and not line.endswith("/"))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--previous", default=None)
    ap.add_argument("--worktree", default=os.environ.get("ISG_EVAL_ROOT", os.getcwd()))
    ap.add_argument("--b1k", default=None, help="the Behavior-1K checkout the runs imported (its overlay state is the code under test)")
    a = ap.parse_args()
    n1 = _run_summary(os.path.join(a.runs, "n1"))
    nn = _run_summary(os.path.join(a.runs, "n8same"))
    verify = _load(os.path.join(a.runs, "n8same", "verify.json"), {}) or {}
    nn.update({"pass": bool(verify.get("pass", False)), "inconclusive": bool(verify.get("inconclusive", False)),
               "reasons": verify.get("reasons", ["no verify.json"] if not verify else []), "instances": verify.get("instances", {}),
               "pass_relaxed": verify.get("pass_relaxed"), "relaxed_reasons": verify.get("relaxed_reasons")})
    ok = n1["ok"] and nn["ok"] and nn["pass"]
    # every slot reproduces the reference's verdicts and success steps, only the strict rotation rule fails: the comparator, not the physics
    slot_problems = [p for inst in nn["instances"].values() for sl in inst.get("slots", []) for p in sl.get("problems", [])]
    comparator_only = (not ok) and n1["ok"] and nn["ok"] and bool(nn.get("pass_relaxed")) and not slot_problems
    prev = _load(a.previous) if a.previous else None
    try:
        head = subprocess.run(["git", "-C", a.worktree, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        head = None
    n1_parity = None
    for f in glob.glob(os.path.join(a.runs, "n1", "verify_vs_*.json")):
        v = _load(f, {}) or {}
        n1_parity = {"reference": os.path.basename(f)[len("verify_vs_"):-5], "pass": v.get("pass"), "reasons": v.get("reasons", [])[:3]}
    rec = {"task": a.task, "campaign": "vec_next", "unit": f"vec_next_verify__{a.task}", "date": time.strftime("%Y-%m-%d %H:%M %Z"),
           "n1_parity": n1_parity, "extra_args": os.environ.get("ISG_EXTRA_ARGS"),
           "pass_relaxed": bool(n1["ok"] and nn["ok"] and nn.get("pass_relaxed")), "comparator_only": comparator_only,
           "host": _run_host(a.runs) or socket.gethostname(), "composed_on": socket.gethostname(),
           "pass": ok, "num_envs": a.num_envs if ok else None, "tested_num_envs": a.num_envs,
           "runs": {"n1": n1, "n8same": nn},
           "code": {"worktree": a.worktree, "head": head, "b1k_dir": a.b1k, "overlay_files": _overlay_files(a.b1k),
                    "rooms_override": os.environ.get("ISG_ROOMS_OVERRIDE")},
           "previous": None if not prev else {"file": a.previous, "pass": prev.get("pass"), "num_envs": prev.get("num_envs")}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(f"VERIFY_RECORD {a.task} pass={ok} pass_relaxed={rec['pass_relaxed']} comparator_only={comparator_only} num_envs={rec['num_envs']} previous={rec['previous'] and rec['previous']['num_envs']} -> {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
