#!/usr/bin/env python3
"""Write a policy-eval run: ``$EV/runs/<run>/spec.json`` (what eval_worker.py reads; upload it to ``$JOB_S3/runs/<run>/spec.json``)
and ``units.json`` (vastctl_eval.py's progress: ``frames`` = the unit's step budget).

    make_eval_run.py --run <name> --ckpts ckpts.json [--instances <eval-instances root>] [--select all | n_per_class:K | family:F,...]
                     [--run-isg-args "..."] [--code eval_<sha>] [--results-dir <dir>]

``--results-dir``: where ``vast/collect.sh`` gathers the run's results (default ``$EV/runs/<run>/results``): ``<dir>/<ckpt>/<instance id>/``
holds every artifact of the instance's episode; ``summarize.py <dir>`` reports total / family / task class / task metrics.

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
#: the 2026-11-16 re-export (= kmy17518/isg-standard-eval-224-instances-v2 revision cf94275c; same index order as the first export)
DEFAULT_INSTANCES = str(EV / "dataset/eval-layout-v2r/ispatialgym-eval-instances")
#: the owner's eval settings (2026-11-16): settled evaluation, 45 settle steps, 224 px cameras; the step budget alone ends an episode.
#: No videos by default (owner 2026-11-16); add e.g. --record-view head to a run's --run-isg-args for <instance id>/head_rgb.mp4.
DEFAULT_ARGS = "--image-size 224 --settle-eval on --settle-steps 45 --wallclock 1000000 --record video"
POLICY_ENVS = {"pi": ["bundle/policy/policy_env_pi.tar.zst", "bundle/policy/openpi_data.tar.zst"], "act": "bundle/policy/policy_env_act.tar.zst", "dp": "bundle/policy/policy_env_dp.tar.zst"}
_P = "/home/minyeongk/Desktop/projects/isg-train224-eval"
POLICY_PATHS = {"pi": {"cwd": f"{_P}/openpi", "python": f"{_P}/openpi/.venv-jax/bin/python", "openpi_data": f"{_P}/openpi_data"},
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
    ap.add_argument("--worlds", default=None, help="world groups (world_groups.py --json): a unit is a chunk of one group, run in one world "
                    "(run_isg --instance-list); without it a unit is one instance (one world each)")
    ap.add_argument("--chunk-steps", type=int, default=8000, help="--worlds: a unit's instances sum to at most this step budget")
    ap.add_argument("--results-dir", default=None, help="where collect.sh gathers the results (default $EV/runs/<run>/results)")
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
    def budget_of(r):
        return int(r.get("max_steps") or max(300, round(1.5 * int(r["length"]))))

    def inst(i):
        r = recs[i]
        return {"index": i, "tro": r["files"]["tro_state"], "budget": budget_of(r), "task_family": r["task_family"], "subtask": r["subtask"],
                "class_id": r["source"]["class_id"]}

    groups = json.load(open(a.worlds)) if a.worlds else [[i] for i in range(len(recs))]
    chosen = set(sel)
    units = []
    for tag, c in ckpts.items():
        fams = set(c.get("families") or [])
        for g in groups:
            members = [i for i in g if i in chosen and (not fams or recs[i]["task_family"] in fams)]
            # chunks of at most --chunk-steps (longest first, so each chunk's world build is amortised over as many steps as allowed)
            chunks, cur, tot = [], [], 0
            for i in sorted(members, key=lambda i: -budget_of(recs[i])):
                if cur and tot + budget_of(recs[i]) > a.chunk_steps:
                    chunks.append(cur)
                    cur, tot = [], 0
                cur.append(i)
                tot += budget_of(recs[i])
            if cur:
                chunks.append(cur)
            for ch in chunks:
                ins = [inst(i) for i in sorted(ch)]
                budget = sum(x["budget"] for x in ins)
                units.append({"unit": f"{tag}__{ins[0]['index']:04d}" + (f"-{len(ins)}" if len(ins) > 1 else ""), "ckpt": tag, "index": ins[0]["index"],
                              "instances": ins, "budget": budget, "frames": budget, "scene": recs[ins[0]["index"]]["scene"],
                              "task_family": ins[0]["task_family"], "subtask": ins[0]["subtask"], "class_id": ins[0]["class_id"],
                              "tro": ins[0]["tro"]})
    spec = {"run": a.run, "code": a.code, "dataset": a.dataset, "policy_envs": POLICY_ENVS, "policy_paths": POLICY_PATHS, "ckpts": ckpts,
            "run_isg_args": shlex.split(a.run_isg_args), "units": units}
    d = EV / "runs" / a.run
    spec["results_dir"] = os.path.abspath(a.results_dir) if a.results_dir else str(d / "results")
    d.mkdir(parents=True, exist_ok=True)
    json.dump(spec, open(d / "spec.json", "w"), indent=1)
    json.dump({"units": units}, open(d / "units.json", "w"))
    per = collections.Counter(u["ckpt"] for u in units)
    print(f"{a.run}: {len(units)} units {dict(per)}, {sum(len(u['instances']) for u in units)} instances, {sum(u['budget'] for u in units)} "
          f"budget steps, largest unit {max(u['budget'] for u in units)} -> {d}/spec.json")


if __name__ == "__main__":
    main()
