#!/usr/bin/env python3
"""Persistent Tier B evaluator worker (evaluator fix 2.2, 2026-10-08): one Isaac Sim process that keeps its loaded world across
queue entries and runs the episode jobs the loop hands it.

The loop writes ``<jobs-dir>/job_<n>.json`` files (the fields of ``eval_episodes.py``'s arguments: run, step, checkpoint
repo / file / sha256 / path, server url + command line, action horizon, task, split, dataset, instances, episodes, lab_out,
raw_out, horizon, tag, video size, overwrite, no_planner, cleanup) and the worker processes them in order with
:func:`eval_episodes.run_job` on one shared world; every episode record is written exactly as by ``eval_episodes.py`` (plus
``memory`` and ``worker``). When a job is done the worker writes ``job_<n>.done.json`` (summary: episodes, errors, memory).
``<jobs-dir>/STOP`` makes the worker exit after the current job; it also exits (code 75, logged) when its GPU memory after
an episode exceeds ``--max-gpu-mib`` (the logged safety restart) - the loop respawns it.

    python lab_tools/eval_worker.py --worker-id 0 --jobs-dir runlogs/tierB/workers/0 [--max-gpu-mib 20000]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)

EXIT_RESTART = 75
DEFAULTS = {"ckpt_path": None, "server": "ws://127.0.0.1:8000", "server_args": "", "action_horizon": 16, "task": "camera_relocalization-standard",
            "instances": os.path.join(WT, "lab", "instances.json"), "dataset": os.path.join(WT, "datasets", "isg-init-eval"), "split": "eval50", "goals_root": None,
            "episodes": None, "limit": None, "horizon": 1500, "max_steps": 6000, "settle_steps": 50, "video_size": 540, "external_video": False,
            "goal_from_episode": None, "goal_swap": None, "no_stop_on_success": False, "keep_depth": False, "include_unverified": False, "tag": None,
            "overwrite": False, "no_planner": False, "cleanup": False, "stock_handle_ag": False}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(jobs_dir, msg):
    line = f"{utc()} [worker {os.getpid()}] {msg}"
    print(line, flush=True)
    with open(os.path.join(jobs_dir, "worker.log"), "a") as f:
        f.write(line + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--jobs-dir", required=True)
    ap.add_argument("--max-gpu-mib", type=int, default=20000, help="safety restart: exit 75 once this process's GPU memory after an episode exceeds this")
    ap.add_argument("--poll-s", type=float, default=2.0)
    ap.add_argument("--idle-exit-s", type=float, default=0, help="exit when no job arrived for this long (0 = never)")
    args = ap.parse_args(argv)
    os.makedirs(args.jobs_dir, exist_ok=True)
    from lab_tools import eval_episodes as EE

    shared: dict = {}
    scene_loads = 0
    n_jobs = 0
    n_episodes = 0
    t_idle = time.time()
    log(args.jobs_dir, f"worker {args.worker_id} up (pid {os.getpid()}); waiting for jobs in {args.jobs_dir}")
    with open(os.path.join(args.jobs_dir, "worker.pid"), "w") as f:
        f.write(f"{os.getpid()}\n")
    while True:
        if os.path.exists(os.path.join(args.jobs_dir, "STOP")):
            log(args.jobs_dir, "STOP seen; exiting")
            break
        jobs = sorted(p for p in glob.glob(os.path.join(args.jobs_dir, "job_*.json")) if not p.endswith(".done.json") and not os.path.exists(p[:-5] + ".done.json"))
        if not jobs:
            if args.idle_exit_s and time.time() - t_idle > args.idle_exit_s:
                log(args.jobs_dir, "idle too long; exiting")
                break
            time.sleep(args.poll_s)
            continue
        path = jobs[0]
        try:
            job = json.load(open(path))
        except Exception as e:  # noqa: BLE001
            log(args.jobs_dir, f"unreadable job {path}: {e!r}; waiting")
            time.sleep(args.poll_s)
            continue
        cfg = dict(DEFAULTS)
        cfg.update({k: v for k, v in job.items() if k in DEFAULTS or k in ("run", "step", "ckpt_repo", "ckpt_file", "ckpt_sha256", "lab_out", "raw_out")})
        cfg["worker_id"] = args.worker_id
        ns = argparse.Namespace(**cfg)
        had_world = shared.get("env") is not None or shared.get("vec_loader") is not None
        t0 = time.time()
        done = {"job": os.path.basename(path), "run": job.get("run"), "step": job.get("step"), "tag": job.get("tag"), "episodes": job.get("episodes"), "started_at": utc(),
                "worker": args.worker_id, "pid": os.getpid(), "world_loaded_before": had_world}
        try:
            if job.get("vectorized") or job.get("kind") == "vec":
                # the vectorized evaluator (N instances per batch on this process's cloned world; vec_eval.run_vec_job)
                from lab_tools import vec_eval as VE

                vns = VE.args_from_job(job, worker_id=args.worker_id)
                try:
                    rc = VE.run_vec_job(vns, shared=shared, shutdown=False)
                except VE.WorldMismatch as e:
                    log(args.jobs_dir, f"job {path}: {e}; this process restarts for it (exit {EXIT_RESTART})")
                    done.update({"rc": EXIT_RESTART, "error": str(e), "finished_at": utc(), "wall_s": round(time.time() - t0, 1), "world_mismatch": True})
                    with open(path[:-5] + ".done.json", "w") as f:
                        json.dump(done, f, indent=1)
                    try:
                        import omnigibson as og

                        og.shutdown()
                    except Exception:  # noqa: BLE001
                        pass
                    return EXIT_RESTART
            elif shared.get("vec_loader") is not None:
                log(args.jobs_dir, f"job {path}: a single-env job on a process holding a vectorized world; restarting (exit {EXIT_RESTART})")
                done.update({"rc": EXIT_RESTART, "error": "single-env job on a vectorized world", "finished_at": utc(), "wall_s": round(time.time() - t0, 1), "world_mismatch": True})
                with open(path[:-5] + ".done.json", "w") as f:
                    json.dump(done, f, indent=1)
                try:
                    import omnigibson as og

                    og.shutdown()
                except Exception:  # noqa: BLE001
                    pass
                return EXIT_RESTART
            elif job.get("kind") == "replay":
                from lab_tools import replay_episodes as REP

                rns = argparse.Namespace(split=job["split"], dataset=job["dataset"], instances=job["instances"], episodes=job["episodes"], horizon=job["horizon"], max_steps=6000,
                                         settle_steps=50, lab_out=job["lab_out"], raw_out=job["raw_out"], video_size=job.get("video_size", 540), no_stop_on_success=False, keep_depth=False,
                                         include_unverified=False, tag=job.get("tag"), overwrite=bool(job.get("overwrite")), no_planner=bool(job.get("no_planner")))
                rc = REP.run_replay_job(rns, shared=shared, shutdown=False)
            else:
                rc = EE.run_job(ns, shared=shared, shutdown=False)
            done["rc"] = rc
        except SystemExit as e:
            done["rc"] = int(e.code) if isinstance(e.code, int) else 1
            done["error"] = f"SystemExit {e.code}"
        except Exception as e:  # noqa: BLE001
            done["rc"] = 1
            done["error"] = repr(e)
            done["traceback"] = traceback.format_exc()
            log(args.jobs_dir, f"job {path} failed: {e!r}")
            traceback.print_exc()
        if not had_world and (shared.get("env") is not None or shared.get("vec_loader") is not None):
            scene_loads += 1
            log(args.jobs_dir, f"scene loaded (load #{scene_loads} of this process)")
        n_jobs += 1
        n_episodes += len(job.get("episodes") or [])
        mem = EE.memory_snapshot()
        done.update({"finished_at": utc(), "wall_s": round(time.time() - t0, 1), "memory": mem, "scene_loads_this_process": scene_loads, "jobs_this_process": n_jobs,
                     "episodes_this_process": n_episodes})
        with open(path[:-5] + ".done.json", "w") as f:
            json.dump(done, f, indent=1)
        log(args.jobs_dir, f"job {os.path.basename(path)} done rc={done.get('rc')} episodes={job.get('episodes')} gpu={mem.get('gpu_mib')} MiB rss={mem.get('rss_mib')} MiB (scene loads {scene_loads}, episodes {n_episodes})")
        t_idle = time.time()
        if mem.get("gpu_mib") and mem["gpu_mib"] > args.max_gpu_mib:
            log(args.jobs_dir, f"SAFETY RESTART: GPU memory {mem['gpu_mib']} MiB > {args.max_gpu_mib} MiB after {n_episodes} episodes; exiting {EXIT_RESTART}")
            try:
                import omnigibson as og

                og.shutdown()
            except Exception:  # noqa: BLE001
                pass
            return EXIT_RESTART
    try:
        import omnigibson as og

        og.shutdown()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
