"""Summary of a grasp-bands run directory (stdlib only: runs on the cluster login node):

    python3 scripts/grasp_bands/aggregate.py --run $GBH/out/<tag> --jobs <manifest.json> [--out summary.json]

Per source (legacy key / floating key / new model, and ``retry:<source>`` for the derived jobs the retry stage made at a measured
other rest - model directories absent from the manifest): jobs, jobs with >= 1 physics-valid grasp, representatives tested / valid,
rejection reasons, rests (stable, other rest), per-model time; coverage per object model (category, model) over all its jobs; reach
maps: the latest log record per (job, rest, height) - a rerun supersedes an earlier failure - with time and IK problems per map."""

import argparse
import collections
import glob
import json
import os
import statistics as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    jobs = {j["id"]: j for j in json.load(open(a.jobs))}
    src_of = {(j["category"], j["model"]): j.get("source", "?") for j in jobs.values()}
    for jid in sorted(os.listdir(os.path.join(a.run, "glib", "models"))):  # retry jobs: derived at a measured other rest
        c = os.path.join(a.run, "glib", "models", jid, "candidates.json")
        if jid not in jobs and os.path.exists(c):
            cd = json.load(open(c))
            jobs[jid] = {"id": jid, "category": cd["category"], "model": cd["model"], "derived": True,
                         "source": "retry:" + src_of.get((cd["category"], cd["model"]), "?")}
    by_src = collections.defaultdict(lambda: collections.Counter())
    reasons = collections.Counter()
    t_model, no_cand, rest_other = [], [], 0
    per = {}
    for jid, j in jobs.items():
        d = os.path.join(a.run, "glib", "models", jid)
        c = os.path.join(d, "candidates.json")
        v = os.path.join(d, "validation.json")
        lib = os.path.join(d, "library.json")
        src = j.get("source", "?")
        b = by_src[src]
        b["jobs"] += 1
        if not os.path.exists(c):
            b["no_candidates_file"] += 1
            continue
        cd = json.load(open(c))
        if not cd.get("representatives"):
            b["no_representatives"] += 1
            no_cand.append(jid)
        if not os.path.exists(v):
            b["not_validated"] += 1
            continue
        vd = json.load(open(v))
        tr = [t for t in vd.get("trials", []) if "verdict" in t]
        nv = sum(1 for t in tr if t["verdict"] == "valid")
        b["reps_tested"] += len(tr)
        b["reps_valid"] += nv
        b["models_with_valid"] += nv > 0
        for t in tr:
            if t["verdict"] != "valid":
                for r in t.get("reasons") or []:
                    reasons[r.split(":")[0] + (":" + r.split(":")[1].split("@")[0] if r.startswith("geometry_at_measured_rest") else "")] += 1
        if os.path.exists(lib):
            ld = json.load(open(lib))
            rest_other += any(k != "r0" for k in (ld.get("rests") or {}))
            b["unstable_rest"] += any(not r.get("stable") for r in (ld.get("rests") or {}).values())
        tm = (vd.get("timing") or {}).get("model_s")
        if tm:
            t_model.append(tm)
        per[jid] = {"valid": nv, "tested": len(tr)}
    cover = collections.defaultdict(lambda: [False, False])  # (category, model) -> [valid on a manifest job, valid on any job]
    for jid, j in jobs.items():
        v = per.get(jid, {}).get("valid", 0) > 0
        c = cover[(j["category"], j["model"])]
        c[0] |= v and not j.get("derived")
        c[1] |= v
    records = []
    for f in sorted(glob.glob(os.path.join(a.run, "logs", "maps_*.log"))):  # appended across runs: file order = time order per shard
        records += [json.loads(l[len("GB_MAP "):]) for l in open(f, errors="replace") if l.startswith("GB_MAP {")]
    latest = {}
    for m in records:
        if m.get("rest") is not None:  # a failure logged before the rest was known (an exception: no rest, maybe no height) ends with a rerun
            latest.pop((m["id"], None, m.get("h")), None)
            latest.pop((m["id"], None, None), None)
        latest[(m["id"], m.get("rest"), m.get("h"))] = m
    maps = list(latest.values())
    ok = [m for m in maps if m.get("ok")]
    out = {"by_source": {k: dict(v) for k, v in sorted(by_src.items())}, "reasons": dict(reasons.most_common()),
           "object_models": {"n": len(cover), "with_valid": sum(c[1] for c in cover.values()),
                             "rescued_by_retry": sum(c[1] and not c[0] for c in cover.values())},
           "model_s": {"n": len(t_model), "median": st.median(t_model) if t_model else None, "p90": sorted(t_model)[int(0.9 * len(t_model))] if t_model else None,
                       "sum_h": round(sum(t_model) / 3600, 2)},
           "rests_other_than_intended": rest_other, "no_representatives": len(no_cand),
           "maps": {"n": len(maps), "ok": len(ok), "failed": len(maps) - len(ok), "log_records": len(records), "fail_reasons": dict(collections.Counter((m.get("reason") or m.get("error") or "")[:60] for m in maps if not m.get("ok"))),
                    "total_s_median": st.median([m["timing"]["total_s"] for m in ok]) if ok else None,
                    "ik_problems_median": st.median([m["timing"]["ik_problems"] for m in ok]) if ok else None,
                    "sum_h": round(sum(m["timing"]["total_s"] for m in ok) / 3600, 2)}}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump({**out, "per_model": per, "no_representatives_ids": no_cand}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
