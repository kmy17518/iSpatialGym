import json, glob, sys, os
F = "/vision/u/minyeong/projects/isg_home/trainfast"; T = "/vision/u/minyeong/projects/isg_home/train85"
k = sys.argv[1]
env = dict(l.strip().split("=", 1) for l in open(sorted(glob.glob(f"{F}/out/fast/{k}/g*"))[0] + "/job.env") if "=" in l)
paths = {"counted": env["COUNTED_TEMPLATE"].strip("'"), "TEMPLATE (sampler input)": env["TEMPLATE"].strip("'")}
for j in sorted(glob.glob(f"{F}/out/*/{k}/g*")) + sorted(glob.glob(f"{T}/out/{k}/g*"))[:2]:
    g = glob.glob(j + "/inst/*/*/template.json")
    if g:
        paths[j.split("/out/")[-1][:40] + (" (train85)" if T in j else "")] = os.path.dirname(g[0])
for name, td in paths.items():
    d = json.load(open(td + "/template.json")); m = json.load(open(td + "/template_meta.json"))
    reg = d["state"]["registry"]["object_registry"]
    mo = ((d.get("metadata") or {}).get("ispatialgym") or {}).get("moved_objects") or {}
    chairs = {n: [round(v, 2) for v in reg[n]["root_link"]["pos"]] for n in sorted(reg) if "chair" in n}
    print(f"{name:42s} variant={m.get('scene_variant')} moved_objects={sorted(mo)} chairs={chairs}")
