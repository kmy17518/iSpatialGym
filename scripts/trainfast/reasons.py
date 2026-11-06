import json, glob, collections
F = "/vision/u/minyeong/projects/isg_home/trainfast"
tags = ["base", "flags", "fast", "fast2", "fast_c24", "pilot", "prebuild", "large", "large_base"]
print(f"{'tag':10s} {'demos':>6s} {'succ':>5s}  failure reasons")
for t in tags:
    c = collections.Counter(); n = s = 0
    for f in glob.glob(f"{F}/out/{t}/*/g*/results.jsonl"):
        if ".cancelled" in f: continue
        for l in open(f):
            r = json.loads(l); n += 1; s += bool(r.get("success"))
            if not r.get("success"): c[r.get("reason") or r.get("status")] += 1
    print(f"{t:10s} {n:6d} {s:5d}  {dict(c.most_common())}")
c = collections.Counter()
for f in glob.glob(f"{F}/out/pilot/*/g*/results.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r.get("reason") == "robot_stutter": c[f.split("/")[-3]] += 1
print("pilot robot_stutter by alias:", dict(c))
