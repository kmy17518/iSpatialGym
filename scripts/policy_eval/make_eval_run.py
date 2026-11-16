#!/usr/bin/env python3
"""Write a policy-eval run: ``$EV/runs/<run>/spec.json`` (what eval_worker.py reads; upload it to ``$JOB_S3/runs/<run>/spec.json``)
and ``units.json`` (vastctl_eval.py's progress: ``frames`` = the unit's step budget).

    make_eval_run.py --run <name> --ckpts ckpts.json [--instances <eval-instances root>] [--select all | n_per_class:K | family:F,...]
                     [--run-isg-args "..."] [--code eval_<sha>]

``ckpts.json``: ``{tag: {"kind": "pi" | "act" | "dp" | "gt", "repo": "kmy17518/<run>", "step": N, "task_alias": <any training alias:
pi's --task>, "server_args": [...], "servers_per_gpu": n, "procs_per_gpu": n, "policy_kwargs": {...}, "run_isg_args": [...],
"families": [...] (only these task families)}}``. Units: one per checkpoint x selected instance, ``<tag>__<dataset index>``, budget =
the instance's max_steps (``max(300, round(1.5 x demo length))``).
"""
import argparse
import collections
import json
import os
import shlex
from pathlib import Path

EV = Path(os.environ.get("EV", "/vision/u/minyeong/projects/isg_home/campaign_20261110/eval_policy_r1"))
DEFAULT_INSTANCES = "/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1/merged/v2/eval-layout/ispatialgym-eval-instances"
#: the owner's eval settings (2026-11-16): settled evaluation, 45 settle steps, 224 px cameras; the step budget alone ends an episode
DEFAULT_ARGS = "--image-size 224 --settle-eval on --settle-steps 45 --wallclock 1000000 --record video"
POLICY_ENVS = {"pi": "bundle/policy/policy_env_pi.tar.zst", "act": "bundle/policy/policy_env_act.tar.zst", "dp": "bundle/policy/policy_env_dp.tar.zst"}
_P = "/home/minyeongk/Desktop/projects/isg-train224-eval"
POLICY_PATHS = {"pi": {"cwd": f"{_P}/openpi", "python": f"{_P}/openpi/.venv-jax/bin/python"},
                "act": {"cwd": f"{_P}/act", "python": f"{_P}/act/.venv/bin/python"},
                "dp": {"cwd": f"{_P}/diffusion_policy", "python": f"{_P}/diffusion_policy/.venv/bin/python"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--instances", default=DEFAULT_INSTANCES)
    ap.add_argument("--select", default="all")
    ap.add_argument("--run-isg-args", default=DEFAULT_ARGS)
    ap.add_argument("--code", default="eval_9eec9441")
    ap.add_argument("--dataset", default="bundle/data/eval_v2.tar.zst")
    a = ap.parse_args()
    ckpts = json.load(open(a.ckpts))
    recs = [json.loads(l) for l in open(os.path.join(a.instances, "metadata/instances.jsonl")) if l.strip()]
    sel = list(range(len(recs)))
    if a.select.startswith("n_per_class:"):
        k = int(a.select.split(":")[1])
        seen = collections.Counter()
        sel = []
        for i, r in enumerate(recs):
            c = r["source"]["class_id"]
            if seen[c] < k:
                seen[c] += 1
                sel.append(i)
    elif a.select.startswith("family:"):
        fams = set(a.select.split(":", 1)[1].split(","))
        sel = [i for i, r in enumerate(recs) if r["task_family"] in fams]
    elif a.select.startswith("indices:"):
        sel = [int(x) for x in a.select.split(":", 1)[1].split(",")]
    units = []
    for tag, c in ckpts.items():
        fams = set(c.get("families") or [])
        for i in sel:
            r = recs[i]
            if fams and r["task_family"] not in fams:
                continue
            budget = int(r.get("max_steps") or max(300, round(1.5 * int(r["length"]))))
            units.append({"unit": f"{tag}__{i:04d}", "ckpt": tag, "index": i, "tro": r["files"]["tro_state"], "budget": budget, "frames": budget,
                          "scene": r["scene"], "task_family": r["task_family"], "subtask": r["subtask"], "class_id": r["source"]["class_id"]})
    spec = {"run": a.run, "code": a.code, "dataset": a.dataset, "policy_envs": POLICY_ENVS, "policy_paths": POLICY_PATHS, "ckpts": ckpts,
            "run_isg_args": shlex.split(a.run_isg_args), "units": units}
    d = EV / "runs" / a.run
    d.mkdir(parents=True, exist_ok=True)
    json.dump(spec, open(d / "spec.json", "w"), indent=1)
    json.dump({"units": units}, open(d / "units.json", "w"))
    per = collections.Counter(u["ckpt"] for u in units)
    print(f"{a.run}: {len(units)} units {dict(per)}, {sum(u['budget'] for u in units)} budget steps -> {d}/spec.json")


if __name__ == "__main__":
    main()
