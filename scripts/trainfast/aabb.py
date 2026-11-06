import json, glob, sys
F = "/vision/u/minyeong/projects/isg_home/trainfast"
k = sys.argv[1]
env = dict(l.strip().split("=", 1) for l in open(sorted(glob.glob(f"{F}/out/fast/{k}/g*"))[0] + "/job.env") if "=" in l)
paths = {"counted": env["COUNTED_TEMPLATE"].strip("'") + "/template.json"}
for t in ("base", "fast", "fast2", "pilot"):
    g = sorted(glob.glob(f"{F}/out/{t}/{k}/g*/inst/*/*/template.json"))
    if g:
        paths[t] = g[0]
for n in sys.argv[2:]:
    print("==", n)
    for name, p in paths.items():
        o = json.load(open(p))["state"]["registry"]["object_registry"][n]
        rl = o["root_link"]
        print(f"  {name:8s} ori {[round(v, 4) for v in rl['ori']]} aabb {[round(v, 3) for v in o['non_kin']['AABB']['aabb']]} lin_vel {[round(v, 4) for v in rl.get('lin_vel', [])]} asleep {o.get('is_asleep')} keys {sorted(o)}")
