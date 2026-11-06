import json, glob, os, math, sys
F = "/vision/u/minyeong/projects/isg_home/trainfast"; T = "/vision/u/minyeong/projects/isg_home/train85"
k = sys.argv[1]; names = sys.argv[2:]
env = dict(l.strip().split("=", 1) for l in open(sorted(glob.glob(f"{F}/out/fast/{k}/g*"))[0] + "/job.env") if "=" in l)
src = {"counted": env["COUNTED_TEMPLATE"].strip("'"), "sampler input (TEMPLATE)": env["TEMPLATE"].strip("'")}
for j in sorted(glob.glob(f"{F}/out/*/{k}/g*")) + sorted(glob.glob(f"{T}/out/{k}/g*"))[:3]:
    t = glob.glob(j + "/inst/*/*/template.json")
    if t:
        src[j.replace(F + "/out/", "").replace(T + "/out/", "train85/")] = os.path.dirname(t[0])
for name, td in src.items():
    d = json.load(open(td + "/template.json"))
    reg = d["state"]["registry"]["object_registry"]
    r = reg.get("robot_r1"); rp = r["root_link"]["pos"] if r else None
    parts = []
    for n in names:
        o = reg.get(n)
        if o is None:
            parts.append(f"{n}: -"); continue
        op = o["root_link"]["pos"]
        parts.append(f"{n.split('_')[0]} {[round(v, 4) for v in op]}" + (f" ({math.dist(rp[:2], op[:2]):.2f} m from robot)" if rp else ""))
    print(f"{name[:46]:46s} robot {None if rp is None else [round(v, 2) for v in rp]} | " + " | ".join(parts))
