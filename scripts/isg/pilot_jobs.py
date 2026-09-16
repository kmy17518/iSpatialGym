#!/usr/bin/env python3
"""E1 pilot manifests for the v2 pipeline on the fleet (`scripts/runpod/dispatch_samp.py` CMD jobs).

    python scripts/isg/pilot_jobs.py sample  [--customs ispatialgym/assets/task_specs/isg_customs.json] [--subtasks ...]
                                             [--templates 1] [--instances 2] --out runlogs/pilot_<date>/jobs_sample.json
    python scripts/isg/pilot_jobs.py demo    --done runlogs/runpod/<purpose>/done.json --out .../jobs_demo_<pod>.json

``sample``: one job per grounded alias of the customs list — ``sample_isg --alias <alias> --templates T
--instances-per-template N --no-write-back --out $OUT`` (the write-back happens when the results are merged:
``merge_constraints`` below reads every pulled ``task_custom_lists.json`` copy and appends its blacklists /
constraints to the source customs file). ``demo``: for every finished sampling job with instances, one job running
``run_isg --mode oracle`` on each ``<iid>-tro_state.json`` of that job's output (on the pod that holds it, like
``make_demo_jobs.py``). Weights order the queue: disembedding / large-scale first.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CUSTOMS = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "isg_customs.json")
WEIGHT = {"disembedding": 4.0, "inter_object_relation_satisfaction-ontop_large_scale": 4.0, "inter_object_relation_satisfaction-under_large_scale": 4.0,
          "configuration_matching-articulation_open_large_scale": 3.5, "configuration_matching-articulation_close_large_scale": 3.5,
          "configuration_matching-block_design": 3.0, "configuration_matching-attachment": 3.0, "alignment-world_vertical": 3.0,
          "alignment-world_horizontal": 3.0, "alignment-axial": 3.0, "camera_relocalization-obstructed": 2.0}
SAMPLE_CMD = ("cd /root/ispatialgym && source /root/miniforge3/etc/profile.d/conda.sh && conda activate ispatialgym && "
              "OMNIGIBSON_HEADLESS=1 timeout {timeout} python -m ispatialgym.sampling.sample_isg --alias {alias} --templates {templates} "
              "--instances-per-template {instances} --seed {seed} --no-write-back --out \"$OUT\" --log-level INFO")
DEMO_CMD = ("cd /root/ispatialgym && source /root/miniforge3/etc/profile.d/conda.sh && conda activate ispatialgym && "
            "for tro in /root/samp/{jid}/*/*/instances/*-tro_state.json; do "
            "OMNIGIBSON_HEADLESS=1 timeout 1800 python -m ispatialgym.oracle.run_isg --instance \"$tro\" --mode oracle --out \"$OUT\" --video --video-size 540; "
            "echo \"DEMO_EXIT $? $tro\"; done")


def subtask_of(entry: dict) -> str:
    iddl = entry.get("iddl", "")
    return os.path.splitext(os.path.basename(iddl))[0] if "/" not in iddl else {
        # iddl paths are <family>/<file>.iddl; the subtask id is the alias prefix up to the first category slot
    }.get(iddl, entry.get("subtask") or "")


def load_entries(path: str):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for alias, e in data.items():
        if alias.startswith("_") or not isinstance(e, dict) or "roles" not in e:
            continue
        st = e.get("subtask")
        if not st:
            iddl = e.get("iddl", "")
            fam, fn = iddl.split("/", 1) if "/" in iddl else ("", iddl)
            fn = os.path.splitext(fn)[0]
            st = fam if fam in ("object_scaling", "mental_rotation", "disembedding") and fn == fam else (f"{fam}-{fn}" if fam else fn)
        out[alias] = (st, e)
    return out


def cmd_sample(args):
    entries = load_entries(args.customs)
    jobs = []
    for alias, (st, e) in sorted(entries.items()):
        if args.subtasks and st not in args.subtasks:
            continue
        jid = "v2__" + alias[:150]
        jobs.append({"id": jid, "cmd": SAMPLE_CMD.format(alias=alias, templates=args.templates, instances=args.instances, seed=args.seed, timeout=args.timeout),
                     "weight": WEIGHT.get(st, 2.0), "subtask": st, "alias": alias, "slot": "representative"})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(jobs, f, indent=1)
    by = collections.Counter(j["subtask"] for j in jobs)
    print(f"{len(jobs)} sample jobs -> {args.out}")
    for k, v in sorted(by.items()):
        print(f"  {k}: {v}")


def cmd_demo(args):
    done = json.load(open(args.done))
    jobs = [j for j in json.load(open(args.manifest))]
    by_id = {j["id"]: j for j in jobs}
    per_pod = collections.defaultdict(list)
    for jid, d in done.items():
        j = by_id.get(jid)
        if j is None or d.get("n_written", 0) <= 0:
            continue
        per_pod[d["pod"]].append({"id": "demo__" + jid, "cmd": DEMO_CMD.format(jid=jid), "weight": 1.0, "subtask": j["subtask"], "alias": j["alias"],
                                  "slot": j.get("slot", ""), "source_job": jid})
    os.makedirs(args.out, exist_ok=True)
    for pod, js in per_pod.items():
        with open(os.path.join(args.out, f"jobs_demo_{pod}.json"), "w") as f:
            json.dump(js, f, indent=1)
        print(f"{pod}: {len(js)} demo jobs")


def cmd_merge_constraints(args):
    """Append the blacklists / constraints of every pulled ``task_custom_lists.json`` copy to the source customs file."""
    import glob

    with open(args.customs) as f:
        src = json.load(f)
    n_bl = n_c = 0
    for copy in glob.glob(os.path.join(args.pulled_root, "*", "v2__*", "task_custom_lists.json")):
        try:
            d = json.load(open(copy))
        except Exception:
            continue
        for alias, e in d.items():
            if alias.startswith("_") or alias not in src or not isinstance(e, dict):
                continue
            for scene, sd in (e.get("scenes") or {}).items():
                tgt = src[alias].setdefault("scenes", {}).setdefault(scene, {}).setdefault("blacklist", {})
                for role, cats in (sd.get("blacklist") or {}).items():
                    for cat, models in cats.items():
                        for m, why in models.items():
                            if m not in tgt.setdefault(role, {}).setdefault(cat, {}):
                                tgt[role][cat][m] = why
                                n_bl += 1
            have = {json.dumps(c, sort_keys=True) for c in src[alias].get("constraints", [])}
            for c in e.get("constraints", []) or []:
                if json.dumps(c, sort_keys=True) not in have:
                    src[alias].setdefault("constraints", []).append(c)
                    n_c += 1
    with open(args.customs, "w") as f:
        json.dump(src, f, indent=1)
    print(f"merged {n_bl} blacklist entries and {n_c} constraints into {args.customs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--customs", default=CUSTOMS)
    s.add_argument("--subtasks", nargs="*", default=None)
    s.add_argument("--templates", type=int, default=1)
    s.add_argument("--instances", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--timeout", type=int, default=5400)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_sample)
    s = sub.add_parser("demo")
    s.add_argument("--done", required=True, help="done.json of samp_summary.py (job id -> pod, n_written)")
    s.add_argument("--manifest", required=True, help="the sample manifest")
    s.add_argument("--out", required=True, help="directory for the per-pod demo manifests")
    s.set_defaults(fn=cmd_demo)
    s = sub.add_parser("merge-constraints")
    s.add_argument("--customs", default=CUSTOMS)
    s.add_argument("--pulled-root", required=True, help="e.g. runlogs/runpod/pilot")
    s.set_defaults(fn=cmd_merge_constraints)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
