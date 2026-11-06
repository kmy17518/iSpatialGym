#!/usr/bin/env python3
"""trainfast / one trunk (owner 2026-11-05): register an alias that train85's counted table lacks (sampled only with job_val.sh
KIND=eval, or re-bound) so that submit_fast.py can submit it, from a run that sampled it at an eval seed (sample_isg --seed-split eval:
the template-making step that stays of KIND=eval).

    python register_alias.py <key> <run dir> --root <dir> --pick <pick> --tag <s3_55tasks|s2_3tasks|iors_large> [--iid IID]
        [--band-stores DIR ...] [--extra-env NAME=VALUE ...] [--customs trunk|template] [--note TEXT]

Writes <root>/templates/<key>/inst/<alias>/<template_id>/ (the run's template without its instances, + the run's scene_variants/ and
task_custom_lists.json, as train85's template copies) and <root>/eval_index/<key>.json (every instance of that template in the run, as
isg-instance-index-v1 records: the train-repeat rule keeps train draws away from them), and prints the registry.tsv row (append it to the
trunk's scripts/trainfast/registry.tsv; the costs come from the run's sample.log / demo meta and set the job limits as for a counted
alias). Writes nothing in the run dir.
"""
import argparse
import glob
import json
import os
import re
import shlex
import shutil
import sys

COLS = ["key", "alias", "pick", "tag", "subtask", "family", "template_id", "template", "counted_template", "eval_index", "start_per_attempt",
        "customs", "limit_s", "budget_s", "attempts", "attempts_s", "world_load_s", "setup_s", "bands_missing_s", "demo_load_s", "band_stores",
        "extra_env", "source_run", "source_code", "note"]


def family_of(subtask, tag):  # = train85 submit_train.family_of
    for f in ("camera_relocalization", "object_scaling", "mental_rotation", "alignment", "disembedding"):
        if subtask.startswith(f):
            return f
    if subtask.startswith("configuration_matching"):
        return "configuration_matching"
    if subtask.startswith("inter_object_relation_satisfaction"):
        return "iors_large" if tag == "iors_large" else "iors_small"
    return subtask


def job_env(run):
    p = os.path.join(run, "job.env")
    env = {}
    for l in open(p) if os.path.exists(p) else []:
        if "=" in l and not l.startswith("#"):
            k, v = l.rstrip("\n").split("=", 1)
            env[k] = (shlex.split(v) or [""])[0]
    return env


def costs(run):
    """(attempts to the first written instance, their seconds, world load s, demo load s) from the run's logs."""
    log = os.path.join(run, "sample.log")
    att, att_s, world = 1, 300.0, 400.0
    if os.path.exists(log):
        totals, first = [], None
        for l in open(log, errors="replace"):
            m = re.search(r"instance (\d+) attempt (\d+) stages: .*total=([\d.]+)s", l)
            if m:
                totals.append(float(m.group(3)))
            if first is None and re.search(r"instance \d+ written \(attempt (\d+)", l):
                first = len(totals)
            if l.startswith("SAMPLING_REPORT "):
                try:
                    world = float(json.loads(l.split(" ", 1)[1])["world"]["load_time_s"])
                except (ValueError, KeyError, TypeError):
                    pass
        if totals:
            n = first or len(totals)
            att, att_s = n, sum(totals[:n])
    demo = 400.0
    for m in glob.glob(os.path.join(run, "demo", "*", "meta.json")):
        t = (json.load(open(m)).get("timings") or {})
        if t.get("load_s"):
            demo = float(t["load_s"])
            break
    return att, att_s, world, demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("run")
    ap.add_argument("--root", required=True)
    ap.add_argument("--pick", required=True)
    ap.add_argument("--tag", required=True, choices=["s3_55tasks", "s2_3tasks", "iors_large"])
    ap.add_argument("--iid", default=None, help="the template holding this instance (default: the first template with an instance)")
    ap.add_argument("--band-stores", nargs="*", default=[])
    ap.add_argument("--extra-env", nargs="*", default=[])
    ap.add_argument("--customs", default="trunk", choices=["trunk", "template"])
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    tdirs = sorted(os.path.dirname(m) for m in glob.glob(os.path.join(run, "inst", "*", "*", "template_meta.json")))
    tdirs = [t for t in tdirs if glob.glob(os.path.join(t, "instances", "*-tro_state.json"))]
    if a.iid:
        tdirs = [t for t in tdirs if os.path.exists(os.path.join(t, "instances", a.iid + "-tro_state.json"))]
    if not tdirs:
        raise SystemExit(f"no template with an instance{' ' + a.iid if a.iid else ''} under {run}/inst")
    src = tdirs[0]
    meta = json.load(open(os.path.join(src, "template_meta.json")))
    # <run>/inst/<alias>/<template id>/ (template_meta.json of the v2 sampler names neither)
    alias, tid = os.path.basename(os.path.dirname(src)), os.path.basename(src)
    subtask = meta.get("subtask") or ""
    dst = os.path.join(a.root, "templates", a.key, "inst", alias, tid)
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists")
    shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns("instances", "runs"))
    inst_root = os.path.join(run, "inst")
    for extra in ("scene_variants", "task_custom_lists.json"):
        s = os.path.join(inst_root, extra)
        d = os.path.join(a.root, "templates", a.key, "inst", extra)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks=True, dirs_exist_ok=True)
        elif os.path.exists(s):
            shutil.copy2(s, d)
    from ispatialgym.sampling import instance_separation as SEP
    recs = [SEP.read_instance(p) for p in sorted(glob.glob(os.path.join(src, "instances", "*-tro_state.json")))]
    idx = os.path.join(a.root, "eval_index", a.key + ".json")
    os.makedirs(os.path.dirname(idx), exist_ok=True)
    SEP.write_index(idx, recs)
    env = job_env(run)
    att, att_s, world, demo = costs(run)
    row = {"key": a.key, "alias": alias, "pick": a.pick, "tag": a.tag, "subtask": subtask, "family": family_of(subtask, a.tag), "template_id": tid,
           "template": dst, "counted_template": src, "eval_index": idx, "start_per_attempt": env.get("START_PER_ATTEMPT", "0") or "0",
           "customs": a.customs, "limit_s": env.get("DEMO_LIMIT_S", "1800") or "1800", "budget_s": env.get("BUDGET_S", "1800") or "1800",
           "attempts": att, "attempts_s": round(att_s, 1), "world_load_s": round(world, 1), "setup_s": 100, "bands_missing_s": 0,
           "demo_load_s": round(demo, 1), "band_stores": " ".join(a.band_stores), "extra_env": " ".join(a.extra_env), "source_run": run,
           "source_code": "/vision/u/minyeong/projects/ispatialgym-oracle-fix", "note": a.note.replace("\t", " ")}
    print("REGISTRY_ROW\t" + "\t".join(str(row[c]) for c in COLS))
    print(f"registered {a.key}: {len(recs)} eval records, template {dst}", file=sys.stderr)


if __name__ == "__main__":
    main()
