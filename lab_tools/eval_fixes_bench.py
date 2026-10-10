#!/usr/bin/env python3
"""Part 2 (evaluator fixes) benchmarks and parity gate (user, 2026-10-08). Everything under lab/diagnostics/eval-fixes/.

  server    --ckpt PATH --port 8002 [--extra --temporal-agg]      start / replace the benchmark ACT server (pid file server_<port>.json)
  variation --port 8002                                           old path (eval_episodes.py process), default execution: r1-early-copy05-st0-s0 @100k on 105, 113, twice
  worker    --slot 0 [--no-planner] [--cleanup]                   start a persistent worker (lab_tools/eval_worker.py) for the benchmark (jobs under <out>/workers/<slot>)
  job       --slot 0 --kind eval|replay --split train --episodes ... --port 8002 --name parity-default   dispatch a job to the benchmark worker
  wait      --slot 0                                              wait for the slot's jobs
  summarize                                                       parity table + memory curves + episodes/hour -> summary.json

Benchmark records go to <out>/<name>/camera_relocalization-standard/episode_XXXXXX.json (eval) or <out>/<name>/episode_XXXXXX.json (replay).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
ACT = os.path.join(WT, "act-isg")
CONDA_PY = os.path.expanduser("~/miniconda3/envs/ispatialgym-v2.0-vectorized/bin/python")
OUT = os.path.join(WT, "lab", "diagnostics", "eval-fixes")
RAW = os.path.join(WT, "runlogs", "diagnostics", "eval-fixes")
TASK = "camera_relocalization-standard"
RUN, STEP = "r1-early-copy05-st0-s0", 100000
CKPT = os.path.join(ACT, "ckpts", RUN, "eval", "step-00100000.pt")
REPO, FILE = "kmy17518/act-isg-r1-early-copy05-st0-s0", "eval/step-00100000.pt"
SPLITS = {"eval50": {"dataset": os.path.join(WT, "datasets", "isg-init-eval"), "instances": os.path.join(WT, "lab", "instances.json")},
          "train": {"dataset": os.path.join(WT, "datasets", "isg-init"), "instances": os.path.join(WT, "lab", "instances-train.json")}}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_of(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def cmd_server(args):
    os.makedirs(RAW, exist_ok=True)
    pidf = os.path.join(RAW, f"server_{args.port}.json")
    if os.path.exists(pidf):
        old = json.load(open(pidf))
        try:
            cmdline = open(f"/proc/{old['pid']}/cmdline").read()
            if "serve_b1k.py" in cmdline:
                os.kill(old["pid"], 15)
                for _ in range(30):
                    if not os.path.exists(f"/proc/{old['pid']}"):
                        break
                    time.sleep(1)
        except Exception:  # noqa: BLE001
            pass
    extra = args.extra or []
    sargs = ["scripts/b1k/serve_b1k.py", "--model-path", os.path.relpath(args.ckpt, ACT), "--port", str(args.port), "--device", "cuda", *extra]
    logf = open(os.path.join(RAW, f"server_{args.port}.log"), "a")
    proc = subprocess.Popen([os.path.join(ACT, ".venv", "bin", "python"), *sargs], cwd=ACT, stdout=logf, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"), start_new_session=True)
    for _ in range(120):
        if healthy(args.port):
            break
        if proc.poll() is not None:
            raise SystemExit("server died")
        time.sleep(2)
    time.sleep(2)
    rec = {"pid": proc.pid, "port": args.port, "ckpt": args.ckpt, "cmd": "CUDA_VISIBLE_DEVICES=0 .venv/bin/python " + " ".join(sargs), "extra": extra, "started_at": utc()}
    json.dump(rec, open(pidf, "w"), indent=1)
    print(json.dumps(rec))


def server_rec(port):
    return json.load(open(os.path.join(RAW, f"server_{port}.json")))


def cmd_variation(args):
    """Old path: two eval_episodes.py processes in sequence, each running 105 and 113 once (default execution)."""
    srv = server_rec(args.port)
    for rep in (1, 2):
        lab_out = os.path.join(OUT, "variation", f"run{rep}")
        raw_out = os.path.join(RAW, "variation", f"run{rep}")
        os.makedirs(os.path.join(raw_out, "logs"), exist_ok=True)
        cmd = [CONDA_PY, os.path.join(WT, "lab_tools", "eval_episodes.py"), "--run", RUN, "--step", str(STEP), "--ckpt-repo", REPO, "--ckpt-file", FILE, "--ckpt-sha256", sha256_of(CKPT),
               "--ckpt-path", CKPT, "--server", f"ws://127.0.0.1:{args.port}", "--server-args", srv["cmd"], "--action-horizon", "16", "--task", TASK, "--split", "train",
               "--dataset", SPLITS["train"]["dataset"], "--instances", SPLITS["train"]["instances"], "--lab-out", lab_out, "--raw-out", raw_out, "--horizon", "1500",
               "--video-size", "540", "--episodes", "105", "113", "--tag", f"variation-run{rep}", "--overwrite"]
        logf = open(os.path.join(raw_out, "logs", f"eval.{time.strftime('%Y%m%d-%H%M%S')}.log"), "w")
        env = dict(os.environ, OMNIGIBSON_HEADLESS="1", OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1")
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=WT, stdout=logf, stderr=subprocess.STDOUT, env=env).returncode
        print(f"variation run{rep}: rc={rc} wall={time.time() - t0:.0f}s", flush=True)


def worker_dir(slot):
    return os.path.join(OUT, "workers", str(slot))


def cmd_worker(args):
    d = worker_dir(args.slot)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "STOP")):
        os.remove(os.path.join(d, "STOP"))
    logf = open(os.path.join(d, f"stdout.{time.strftime('%Y%m%d-%H%M%S')}.log"), "w")
    env = dict(os.environ, OMNIGIBSON_HEADLESS="1", OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1")
    cmd = [CONDA_PY, os.path.join(WT, "lab_tools", "eval_worker.py"), "--worker-id", str(args.slot), "--jobs-dir", d, "--max-gpu-mib", "20000"]
    proc = subprocess.Popen(cmd, cwd=WT, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    print(f"worker {args.slot} pid {proc.pid} jobs {d}")


def cmd_job(args):
    d = worker_dir(args.slot)
    os.makedirs(d, exist_ok=True)
    n = len(glob.glob(os.path.join(d, "job_*.json")))
    sp = args.split
    job = {"kind": args.kind, "run": RUN, "step": STEP, "episodes": args.episodes, "horizon": 1500, "task": TASK, "split": sp, "dataset": SPLITS[sp]["dataset"],
           "instances": SPLITS[sp]["instances"], "video_size": 540, "overwrite": True, "tag": args.name, "no_planner": bool(args.no_planner), "cleanup": bool(args.cleanup),
           "created_at": utc(), "entry_key": f"bench/{args.name}"}
    if args.kind == "eval":
        srv = server_rec(args.port)
        job.update({"ckpt_repo": REPO, "ckpt_file": FILE, "ckpt_sha256": sha256_of(CKPT), "ckpt_path": CKPT, "server": f"ws://127.0.0.1:{args.port}", "server_args": srv["cmd"],
                    "action_horizon": 1 if "--temporal-agg" in (srv.get("extra") or []) else 16, "lab_out": os.path.join(OUT, args.name), "raw_out": os.path.join(RAW, args.name)})
    else:
        job.update({"lab_out": os.path.join(OUT, args.name, sp), "raw_out": os.path.join(RAW, args.name, sp)})
    path = os.path.join(d, f"job_{n:04d}.json")
    json.dump(job, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)
    print(f"dispatched {path}: {args.kind} {sp} {args.episodes} -> {job['lab_out']}")


def cmd_wait(args):
    d = worker_dir(args.slot)
    while True:
        jobs = [p for p in glob.glob(os.path.join(d, "job_*.json")) if not p.endswith(".done.json")]
        pending = [p for p in jobs if not os.path.exists(p[:-5] + ".done.json")]
        if not pending:
            break
        time.sleep(10)
    for p in sorted(glob.glob(os.path.join(d, "job_*.done.json"))):
        dn = json.load(open(p))
        print(os.path.basename(p), "rc", dn.get("rc"), "episodes", dn.get("episodes"), "wall", dn.get("wall_s"), "mem", dn.get("memory"), "scene loads", dn.get("scene_loads_this_process"))


def recs(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        r = json.load(open(p))
        out[int(r["episode_index"])] = r
    return out


def cmd_summarize(args):
    import statistics

    summary = {"format": "isg-lab-eval-fixes/v1", "created_at": utc()}
    # run-to-run variation (old path)
    v1 = recs(os.path.join(OUT, "variation", "run1", TASK, "episode_*.json"))
    v2 = recs(os.path.join(OUT, "variation", "run2", TASK, "episode_*.json"))
    base = recs(os.path.join(WT, "lab", "tierB-train", RUN, f"step_{STEP:08d}", TASK, "episode_*.json"))
    var_rows = []
    for e in sorted(set(v1) | set(v2)):
        runs = [r for r in (base.get(e), v1.get(e), v2.get(e)) if r]
        pes = [(r.get("score") or {}).get("position_error_m") for r in runs]
        oes = [(r.get("score") or {}).get("orientation_error_deg") for r in runs]
        var_rows.append({"episode": e, "success": [r.get("success") for r in runs], "position_error_m": pes, "orientation_error_deg": oes,
                         "position_spread_m": (max(pes) - min(pes)) if all(p is not None for p in pes) else None, "orientation_spread_deg": (max(oes) - min(oes)) if all(o is not None for o in oes) else None})
    spread_pos = [r["position_spread_m"] for r in var_rows if r["position_spread_m"] is not None]
    spread_ori = [r["orientation_spread_deg"] for r in var_rows if r["orientation_spread_deg"] is not None]
    summary["run_to_run_variation"] = {"rows": var_rows, "max_position_spread_m": max(spread_pos) if spread_pos else None, "max_orientation_spread_deg": max(spread_ori) if spread_ori else None,
                                       "note": "existing record + two fresh runs of the old path, same seeds, default execution"}
    tol_pos = max([0.1] + spread_pos) if spread_pos else 0.1
    tol_ori = max([10.0] + spread_ori) if spread_ori else 10.0
    # parity: new path vs existing records (default and ta)
    parity = {}
    for name, ref_dir in (("parity-default", os.path.join(WT, "lab", "tierB-train", RUN, f"step_{STEP:08d}", TASK)), ("parity-ta", os.path.join(WT, "lab", "diagnostics", "exec", "ta", RUN, f"step_{STEP:08d}", TASK))):
        new = recs(os.path.join(OUT, name, TASK, "episode_*.json"))
        ref = recs(os.path.join(ref_dir, "episode_*.json"))
        rows = []
        for e in sorted(new):
            n, r = new[e], ref.get(e)
            if r is None:
                continue
            pe_n, pe_r = (n.get("score") or {}).get("position_error_m"), (r.get("score") or {}).get("position_error_m")
            oe_n, oe_r = (n.get("score") or {}).get("orientation_error_deg"), (r.get("score") or {}).get("orientation_error_deg")
            rows.append({"episode": e, "success_ref": r.get("success"), "success_new": n.get("success"), "verdict_same": bool(r.get("success")) == bool(n.get("success")),
                         "position_error_ref_m": pe_r, "position_error_new_m": pe_n, "orientation_error_ref_deg": oe_r, "orientation_error_new_deg": oe_n,
                         "within_variation": (pe_n is not None and pe_r is not None and abs(pe_n - pe_r) <= tol_pos and oe_n is not None and oe_r is not None and abs(oe_n - oe_r) <= tol_ori),
                         "setting_new": (n.get("server_metadata") or {}).get("execution_mode"), "action_horizon_new": (n.get("server_metadata") or {}).get("action_horizon"),
                         "memory_new": n.get("memory"), "error_new": n.get("error")})
        parity[name] = {"rows": rows, "n": len(rows), "verdicts_same": sum(1 for x in rows if x["verdict_same"]), "within_variation": sum(1 for x in rows if x["within_variation"]),
                        "pass": bool(rows) and all(x["verdict_same"] and x["within_variation"] and not x["error_new"] for x in rows)}
    # replays on the new path
    rep_rows = []
    for sp in ("eval50", "train"):
        for e, r in recs(os.path.join(OUT, "parity-replay", sp, "episode_*.json")).items():
            rep_rows.append({"split": sp, "episode": e, "success": r.get("success"), "binary_success_ever": r.get("binary_success_ever"), "first_success_step": r.get("first_success_step"),
                             "position_error_m": (r.get("score") or {}).get("position_error_m"), "error": r.get("error")})
    summary["parity"] = {**parity, "replays": {"rows": rep_rows, "n": len(rep_rows), "reached_tolerance": sum(1 for x in rep_rows if x["binary_success_ever"]), "pass": len(rep_rows) == 14 and all(x["binary_success_ever"] for x in rep_rows)},
                         "tolerance_used": {"position_m": tol_pos, "orientation_deg": tol_ori}}
    ff = os.path.join(OUT, "first_frames_check.json")
    if os.path.exists(ff):
        summary["parity"]["first_frames"] = json.load(open(ff))
    summary["parity"]["pass"] = all(parity[k]["pass"] for k in parity) and summary["parity"]["replays"]["pass"] and (summary["parity"].get("first_frames", {}).get("pass", True))
    # memory curves: production baseline (memlog of the morning process) and the new path (worker records in order)
    mem_new = []
    for name in ("parity-replay/eval50", "parity-replay/train", "parity-default", "parity-ta"):
        pat = os.path.join(OUT, name, "episode_*.json") if "replay" in name else os.path.join(OUT, name, TASK, "episode_*.json")
        for p in sorted(glob.glob(pat)):
            r = json.load(open(p))
            m = r.get("memory") or {}
            if m.get("gpu_mib"):
                mem_new.append({"name": name, "episode": r["episode_index"], "finished_at": r.get("finished_at"), "worker": r.get("worker"), "gpu_mib": m.get("gpu_mib"), "rss_mib": m.get("rss_mib"),
                                "torch_allocated_mib": m.get("torch_allocated_mib"), "torch_reserved_mib": m.get("torch_reserved_mib")})
    mem_new.sort(key=lambda x: (str(x["worker"]), x["finished_at"]))
    by_worker = {}
    for x in mem_new:
        by_worker.setdefault(str(x["worker"]), []).append(x)
    summary["memory"] = {"new_path_per_episode": mem_new,
                         "new_path_growth_after_warmup": {w: {"n_episodes": len(v), "gpu_first_after_warmup_mib": v[1]["gpu_mib"] if len(v) > 1 else None, "gpu_last_mib": v[-1]["gpu_mib"],
                                                              "growth_mib": (v[-1]["gpu_mib"] - v[1]["gpu_mib"]) if len(v) > 1 else None, "rss_growth_mib": (v[-1]["rss_mib"] - v[1]["rss_mib"]) if len(v) > 1 and v[1].get("rss_mib") and v[-1].get("rss_mib") else None}
                                                          for w, v in by_worker.items()}}
    memlog = os.path.join(OUT, "memlog.jsonl")
    if os.path.exists(memlog):
        samples = [json.loads(l) for l in open(memlog)]
        pids = {}
        for s in samples:
            for p in s["procs"]:
                if p["kind"] == "eval":
                    pids.setdefault(p["pid"], []).append((s["ts"], p["gpu_mib"], p["rss_mib"]))
        summary["memory"]["production_baseline"] = {str(pid): {"n_samples": len(v), "first": v[0], "last": v[-1], "gpu_after_first_3min_mib": next((x[1] for x in v if x[0] >= v[0][0] and (time.mktime(time.strptime(x[0], "%Y-%m-%dT%H:%M:%SZ")) - time.mktime(time.strptime(v[0][0], "%Y-%m-%dT%H:%M:%SZ"))) > 180), None)}
                                                     for pid, v in pids.items() if len(v) > 10}
    # throughput: before (records of the last 24 h of production) vs after (parity jobs)
    walls_before = [json.load(open(p)).get("wall_s") for p in glob.glob(os.path.join(WT, "lab", "tierB-train", "*", "step_*", TASK, "episode_*.json"))]
    walls_before = [w for w in walls_before if w]
    walls_after = [x for x in (json.load(open(p)).get("wall_s") for p in glob.glob(os.path.join(OUT, "parity-*", TASK, "episode_*.json"))) if x]
    summary["throughput"] = {"before_median_wall_s_per_episode": statistics.median(walls_before) if walls_before else None, "after_median_wall_s_per_episode": statistics.median(walls_after) if walls_after else None,
                             "note": "episodes per hour per process = 3600 / median wall; the scene load (~120 s per fresh process) and the planner build (~27 s) are no longer paid per entry"}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps({"variation": summary["run_to_run_variation"]["max_position_spread_m"], "parity": {k: (v.get("pass"), v.get("n")) for k, v in summary["parity"].items() if isinstance(v, dict)},
                      "memory_growth": summary["memory"]["new_path_growth_after_warmup"], "throughput": summary["throughput"]}, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["server", "variation", "worker", "job", "wait", "summarize"])
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--extra", nargs="*", default=None)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--kind", default="eval", choices=["eval", "replay"])
    ap.add_argument("--split", default="train", choices=["train", "eval50"])
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--name", default="parity-default")
    ap.add_argument("--no-planner", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    return {"server": cmd_server, "variation": cmd_variation, "worker": cmd_worker, "job": cmd_job, "wait": cmd_wait, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
