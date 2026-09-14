"""Aggregate validation results + reviewer flags -> per-model / per-category / per-task counts."""
import json, glob, os, collections
OUT = "/tmp/glib"
jobs = {j["key"]: j for j in json.load(open(f"{OUT}/models.json"))}
for j in json.load(open(f"{OUT}/models_scaled.json")):
    j = dict(j); j["category"] = j["category"] + " (scaled)"; jobs[j["key"]] = j
results = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{OUT}/results/*.json")}
review = {}
for p in sorted(glob.glob(f"{OUT}/review/*.json")):          # later waves override earlier reviews of the same key
    try:
        d = json.load(open(p))
    except Exception as e:
        print("bad review file", p, e); continue
    for m in d.get("models", []):
        review[m["key"]] = m
rows = []
for key, j in jobs.items():
    r = results.get(key); rv = review.get(key) if (r and r.get("n_valid")) else None      # a review only counts for the current validation result
    n_valid = r["n_valid"] if r else 0
    tested = len(r["tested"]) if r else 0
    flagged = set(); unsure = set(); flagged_strict = set(); reviewed = rv is not None
    if rv:
        for g in rv.get("grasps", []):
            fl = set(g.get("flags", []))
            if g.get("verdict") == "flag":
                flagged_strict.add(g["rank"])
                if fl - {"one_finger"}: flagged.add(g["rank"])          # lenient: one_finger alone is an AG-closing artifact
            elif g.get("verdict") == "unsure": unsure.add(g["rank"])
    n_clean = max(0, n_valid - len(flagged)) if reviewed else None
    n_clean_strict = max(0, n_valid - len(flagged_strict)) if reviewed else None
    rows.append({**j, "n_tested": tested, "n_valid": n_valid, "reviewed": reviewed,
                 "n_flagged": len(flagged), "n_flagged_strict": len(flagged_strict), "n_unsure": len(unsure), "n_clean": n_clean, "n_clean_strict": n_clean_strict, "flags": sorted({f for g in (rv or {}).get("grasps", []) for f in g.get("flags", [])}), "error": (r or {}).get("error")})
json.dump(rows, open(f"{OUT}/aggregate.json", "w"), indent=1)

def summarize(rows, label):
    n = len(rows); v1 = sum(1 for r in rows if r["n_valid"] >= 1); v5 = sum(1 for r in rows if r["n_valid"] >= 5)
    rev = [r for r in rows if r["reviewed"]]
    c1 = sum(1 for r in rev if (r["n_clean"] or 0) >= 1); c3 = sum(1 for r in rev if (r["n_clean"] or 0) >= 3); c5 = sum(1 for r in rev if (r["n_clean"] or 0) >= 5)
    s1 = sum(1 for r in rev if (r["n_clean_strict"] or 0) >= 1); s3 = sum(1 for r in rev if (r["n_clean_strict"] or 0) >= 3)
    return f"{label:24s} models={n:3d} | AG>=1 {v1:3d}  AG=5 {v5:3d} | reviewed {len(rev):3d} | lenient clean>=1 {c1:3d} >=3 {c3:3d} =5 {c5:3d} | strict clean>=1 {s1:3d} >=3 {s3:3d}"

print(summarize(rows, "ALL"))
for task in sorted({r["task"] for r in rows}):
    print(summarize([r for r in rows if r["task"] == task], task))
print()
cats = collections.defaultdict(list)
for r in rows: cats[r["category"]].append(r)
print(f"{'category':26s} {'task':22s} models AG>=1 clean>=1 clean>=3  flags")
for c, rs in sorted(cats.items(), key=lambda kv: (kv[1][0]['task'], kv[0])):
    fl = collections.Counter(f for r in rs for f in r["flags"])
    print(f"{c:26s} {rs[0]['task']:22s} {len(rs):6d} {sum(1 for r in rs if r['n_valid']>=1):6d} {sum(1 for r in rs if (r['n_clean'] or 0)>=1):8d} {sum(1 for r in rs if (r['n_clean'] or 0)>=3):8d}  {dict(fl.most_common(4))}")
fl_all = collections.Counter(f for r in rows for f in r["flags"]); print("\nflag frequency (models with >=1 grasp flagged for it):", dict(fl_all.most_common()))
