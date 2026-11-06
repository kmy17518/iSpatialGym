import json, glob, collections, csv
F = "/vision/u/minyeong/projects/isg_home/trainfast"; T = "/vision/u/minyeong/projects/isg_home/train85"
rows = list(csv.DictReader(open(f"{T}/manifests/manifest.tsv"), delimiter="\t"))
for k in ("inter_object_relation_satisfaction-nextto_02", "inter_object_relation_satisfaction-touching_01"):
    for t in ("base", "flags", "fast", "fast2", "pilot"):
        c = collections.Counter(); n = 0
        for f in glob.glob(f"{F}/out/{t}/{k}/g*/results.jsonl"):
            for l in open(f):
                r = json.loads(l); n += 1; c[r.get("reason") or "success"] += 1
        print(f"{k[35:]:14s} {t:6s} n={n:3d} {dict(c.most_common())}")
    sel = [r for r in rows if r["key"] == k and r["demo_s"]]
    print(f"{k[35:]:14s} train85 n={len(sel)} {dict(collections.Counter(r['status'] for r in sel))}")
