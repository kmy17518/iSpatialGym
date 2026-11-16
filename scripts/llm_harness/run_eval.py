#!/usr/bin/env python3
"""Run a policy on eval instances (one, several, or a recorded subset) and collect every instance's results in one folder.

    python scripts/llm_harness/run_eval.py --policy my_module:MyPolicy --results results/iors_goal \\
        (--subset iors | --subset full | --instances <id or index> [...] | --instance-file <file>) \\
        [--condition goal | cross_goal | text] [--name <label>] [--policy-kwargs '<json>'] [--gpus 0,1] [--jobs 2]

Each instance runs in its own ``python -m ispatialgym.run_isg`` process (the policy entry point the campaign's evals used) (its own simulator and world: the 100
selected instances are 100 different tasks), with the settings of the campaign's policy evals: 224 px cameras, settled
evaluation (``--settle-eval on --settle-steps 45``), the instance's step budget (max(300, 1.5 x its demo length)) as the only
end of an episode. Results: ``<results>/<name>/<instance id>/`` (meta.json, result.json, init / goal / final images, ...) and
``<results>/<name>/_logs/<instance id>.log``; ``<results>/<name>/ckpt.json`` lists what was asked for (the summary reports what
is missing). Instances with a finished ``meta.json`` are skipped (rerun the same command to resume; ``--rerun`` to redo them).
Afterwards: ``bash scripts/llm_harness/summarize.sh <results>``.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
#: the run_isg settings of the campaign's policy evals (scripts/policy_eval/make_eval_run.py DEFAULT_ARGS): no video by default
RUN_ISG_ARGS = ["--mode", "policy", "--image-size", "224", "--settle-eval", "on", "--settle-steps", "45", "--wallclock", "1000000",
                "--record", "video"]
#: the simulator environment of those evals (scripts/policy_eval/vast/eval_worker.py proc_env): the same physics / evaluation switches
EVAL_ENV = {
    "OMNIGIBSON_HEADLESS": "1", "OMNI_KIT_ACCEPT_EULA": "YES", "HDF5_USE_FILE_LOCKING": "FALSE", "OMNI_CRASHREPORTER_ENABLED": "0",
    "OMNIGIBSON_TRACK_STATE_UPDATES": "1", "OMP_NUM_THREADS": "1", "OMNIGIBSON_CHANGE_DETECTION_NUMPY": "1",
    "ISPATIALGYM_FAST_GEOMETRY": "1", "ISPATIALGYM_FAST_SUPPORT": "1", "ISPATIALGYM_FAST_ASSIGN": "1", "ISPATIALGYM_SHARED_GOAL_EVAL": "1",
    "ISPATIALGYM_GC_FREEZE": "1", "ISPATIALGYM_BATCHED_WORLD_SYNC": "1", "ISPATIALGYM_SHARD_FORMAT": "legacy",
    "ISPATIALGYM_SEEDED_SPHERE_FIT": "off", "NUMPY_MADVISE_HUGEPAGE": "0", "ISPATIALGYM_CUROBO_WARMUP_ITERS": "0", "PYTHONHASHSEED": "0",
    "ISPATIALGYM_EVAL_IGNORE_COLLECTION_LIGHTING": "1",
}
#: switches of other campaign phases that must be off in an eval
EVAL_UNSET = ("ISPATIALGYM_PHYSICS_ONLY", "ISPATIALGYM_HEAD_CAMERA_ONLY", "ISPATIALGYM_EXECUTOR_MESH_CACHE", "ISPATIALGYM_EVAL_BDDL_OVERLAY",
              "ISPATIALGYM_ASYNC_SHARD", "ISPATIALGYM_PER_SCENE_STATE", "ISPATIALGYM_ASYNC_FRAMES", "ISPATIALGYM_LAZY_FRAMES",
              "ISPATIALGYM_MESH_CACHE", "ISPATIALGYM_RECORD_BEHAVIOR_PROPRIO", "ISPATIALGYM_SAMPLER_MESH_CACHE")


def load_selection():
    return json.load(open(HERE / "instances_100.json"))["instances"]


def resolve(a, dataset):
    """[(instance id, tro path, selection row or None)] for the requested instances."""
    recs = [json.loads(l) for l in open(dataset / "metadata" / "instances.jsonl") if l.strip()]
    by_id = {os.path.basename(r["files"]["tro_state"])[: -len("-tro_state.json")]: (i, r) for i, r in enumerate(recs)}
    sel = {x["instance_id"]: x for x in load_selection()}
    if a.subset:
        ids = [l.strip() for l in open(HERE / "subsets" / {"iors": "iors_35.txt", "full": "full_100.txt"}[a.subset]) if l.strip()]
    elif a.instance_file:
        ids = [l.strip() for l in open(a.instance_file) if l.strip() and not l.startswith("#")]
    else:
        ids = list(a.instances)
    out = []
    for x in ids:
        if x.isdigit():
            if int(x) >= len(recs):
                raise SystemExit(f"index {x}: the dataset has {len(recs)} instances")
            x = os.path.basename(recs[int(x)]["files"]["tro_state"])[: -len("-tro_state.json")]
        if x not in by_id:
            raise SystemExit(f"unknown instance {x!r} (an instance id or a dataset index; scripts/llm_harness/instances_100.csv)")
        out.append((x, str(dataset / by_id[x][1]["files"]["tro_state"]), sel.get(x), by_id[x]))
    return out


def record_request(out_root, a, tag, run_rec):
    """Add this invocation's request to ``<out_root>/ckpt.json`` (read-modify-write under a lock, atomic replace: several runners
    may share one results folder)."""
    import fcntl

    cj = out_root / "ckpt.json"
    with open(out_root / ".ckpt.json.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
        except OSError:  # a file system without locks (some NFS mounts): the atomic replace below still keeps the file whole
            pass
        try:
            info = json.load(open(cj)) if cj.exists() else None
        except ValueError:
            print(f"warning: {cj} unreadable, starting it afresh")
            info = None
        info = info or {"ckpt": a.name, "runs": {}}
        info["spec"] = {"policy": a.policy, "policy_kwargs": json.loads(a.policy_kwargs), "condition": a.condition}
        info["runs"][tag] = run_rec
        tmp = cj.with_name(f".ckpt.json.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(info, f, indent=1)
        os.replace(tmp, cj)


def run_one(iid, tro, cmd_base, env, out_root, gpu, timeout):
    log = out_root / "_logs" / f"{iid}.log"
    env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu)) if gpu is not None else dict(env)
    t0 = time.time()
    with open(log, "w") as fh:
        fh.write(f"# {dt.datetime.now().isoformat(timespec='seconds')} gpu={gpu}\n# {shlex.join(cmd_base + ['--instance', tro])}\n")
        fh.flush()
        try:
            rc = subprocess.run(cmd_base + ["--instance", tro], env=env, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    res = out_root / iid / "result.json"
    r = json.load(open(res)) if res.exists() else {}
    return iid, rc, r.get("success"), r.get("reason"), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--subset", choices=["iors", "full"], help="iors = the 35 inter-object-relation tasks, full = all 100 (instances_100.csv)")
    g.add_argument("--instances", nargs="+", help="instance ids or dataset indices (one or many)")
    g.add_argument("--instance-file", help="a file of instance ids / indices, one per line")
    ap.add_argument("--policy", required=True, help="module:Class, e.g. my_harness.policy:MyPolicy (an ispatialgym.eval.llm_policy.HarnessPolicy)")
    ap.add_argument("--policy-kwargs", default="{}", help="JSON keyword arguments of the policy class")
    ap.add_argument("--condition", choices=["goal", "cross_goal", "text"], default="goal",
                    help="goal = own goal image (default); cross_goal = cross-episode goal image; text = text instruction, no image (iors only)")
    ap.add_argument("--results", required=True, help="results folder (one per eval, e.g. results/iors_goal)")
    ap.add_argument("--name", default="llm", help="label of the policy / run inside the results folder (default llm)")
    ap.add_argument("--dataset", default=os.environ.get("ISPATIALGYM_INSTANCES_PATH"),
                    help="ispatialgym-eval-instances root (default $ISPATIALGYM_INSTANCES_PATH)")
    ap.add_argument("--gpus", default=None, help="comma list of GPU ids; instances go round-robin (default: the visible GPU)")
    ap.add_argument("--jobs", type=int, default=1, help="instances at once (each needs ~9 GB GPU memory + ~10 GB RAM + 4 CPUs)")
    ap.add_argument("--rerun", action="store_true", help="also rerun instances that already have results")
    ap.add_argument("--timeout", type=float, default=4 * 3600, help="seconds per instance before it is killed")
    ap.add_argument("--extra-args", default="", help="more run_isg arguments, e.g. '--record-view head' for a head-camera video")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dataset:
        raise SystemExit("--dataset (or ISPATIALGYM_INSTANCES_PATH): the ispatialgym-eval-instances folder")
    dataset = Path(a.dataset).resolve()
    json.loads(a.policy_kwargs)
    items = resolve(a, dataset)
    if a.condition == "text":
        prompts = json.load(open(HERE / "text_prompts_iors.json"))["prompts"]
        bad = [iid for iid, _, _, (_, r) in items if r["source"]["class_id"] not in prompts]
        if bad:
            raise SystemExit(f"--condition text: no text prompt for {len(bad)} instance(s) (only the 35 iors tasks have one): {bad[:3]}")
    out_root = Path(a.results).resolve() / a.name
    run_args = RUN_ISG_ARGS + (["--cross-goal"] if a.condition == "cross_goal" else []) + shlex.split(a.extra_args)
    cmd_base = [sys.executable, "-u", "-m", "ispatialgym.run_isg", "--out", str(out_root), "--policy", a.policy,
                "--policy-kwargs", a.policy_kwargs] + run_args
    env = {k: v for k, v in os.environ.items() if k not in EVAL_UNSET}
    env.update(EVAL_ENV)
    env.update({"ISPATIALGYM_INSTANCES_PATH": str(dataset), "ISG_HARNESS_CONDITION": a.condition})
    # the request, for summarize.py (missing instances) and for the record
    tag = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    run_rec = {"run_isg_args": run_args, "dataset": str(dataset), "results_dir": str(out_root.parent), "condition": a.condition,
               "expected": [{"index": i, "instance_id": iid, "class_id": r["source"]["class_id"], "task_family": r["task_family"],
                             "subtask": r["subtask"]} for iid, _, _, (i, r) in items]}
    todo = [(iid, tro) for iid, tro, _, _ in items if a.rerun or not (out_root / iid / "meta.json").exists()]
    print(f"{len(items)} instance(s), {len(items) - len(todo)} already done, running {len(todo)}; condition {a.condition}; "
          f"results {out_root}")
    if a.dry_run:
        for iid, tro in todo:
            print(shlex.join(cmd_base + ["--instance", tro]))
        return
    (out_root / "_logs").mkdir(parents=True, exist_ok=True)
    record_request(out_root, a, tag, run_rec)
    gpus = [g.strip() for g in a.gpus.split(",")] if a.gpus else [None]
    n_ok = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        futs = [pool.submit(run_one, iid, tro, cmd_base, env, out_root, gpus[k % len(gpus)], a.timeout) for k, (iid, tro) in enumerate(todo)]
        for k, f in enumerate(cf.as_completed(futs), 1):
            iid, rc, success, reason, secs = f.result()
            n_ok += bool(success)
            note = "" if (out_root / iid / "meta.json").exists() else f"  NO RESULT (see {out_root / '_logs' / (iid + '.log')})"
            print(f"[{k}/{len(todo)}] {iid}: success={success} reason={reason} rc={rc} {secs / 60:.1f} min{note}", flush=True)
    print(f"done: {n_ok}/{len(todo)} successful (the run's verdict: settled geometric). Summary: bash scripts/llm_harness/summarize.sh {a.results}")


if __name__ == "__main__":
    main()
