import glob, json, re, os, datetime as dt, statistics as st
F = "/vision/u/minyeong/projects/isg_home/trainfast"
RX_REJ = re.compile(r"rejected in (\S+) after ([\d.]+)s")
RX_W = re.compile(r" written \(attempt (\d+), ([\d.]+)s\)")
RX_WL = re.compile(r"world loaded for \S+ in ([\d.]+)s")
def job_stats(J):
    log = open(J + "/sample.log", errors="replace").read()
    rej = [float(m.group(2)) for m in RX_REJ.finditer(log)]; wr = [float(m.group(2)) for m in RX_W.finditer(log)]
    wl = [float(m.group(1)) for m in RX_WL.finditer(log)]
    res = [json.loads(l) for l in open(J + "/results.jsonl")] if os.path.exists(J + "/results.jsonl") else []
    txt = open(J + "/job.txt").read()
    wall = int(re.search(r"ISGF_END \S+ wall_s=(\d+)", txt).group(1)) if "ISGF_END" in txt else None
    sw = int(re.search(r"SAMPLE_EXIT \S+ wall_s=(\d+)", txt).group(1)) if "SAMPLE_EXIT" in txt else None
    demo_s = [r["wall_s"] for r in res]; steps = [r["n_steps"] for r in res]
    return dict(rej=rej, wr=wr, wl=wl[0] if wl else None, res=res, wall=wall, sample_wall=sw, demo_s=demo_s, steps=steps)
def show(tag, keys=None):
    print(f"\n== {tag}")
    print(f"{'alias':26s} {'load':>5s} {'att/w':>5s} {'s/att':>5s} {'samp s/inst':>11s} {'demo s':>6s} {'steps':>5s} {'coll s/inst(k=2)':>16s} {'bottleneck':>10s} {'succ%':>5s} {'job s/succ':>10s}")
    by = {}
    for J in sorted(glob.glob(f"{F}/out/{tag}/*/g*")):
        if J.endswith(".cancelled") or not os.path.exists(J + "/job.txt"): continue
        k = os.path.basename(os.path.dirname(J)); by.setdefault(k, []).append(job_stats(J))
    for k, js in by.items():
        if keys and not any(x in k for x in keys): continue
        rej = sum((j["rej"] for j in js), []); wr = sum((j["wr"] for j in js), []); res = sum((j["res"] for j in js), [])
        if not wr or not res: continue
        n_att = len(rej) + len(wr); s_att = (sum(rej) + sum(wr)) / n_att
        samp = (sum(rej) + sum(wr)) / len(wr)
        demo = st.mean(r["wall_s"] for r in res); steps = st.mean(r["n_steps"] for r in res)
        coll = demo / 2
        succ = sum(1 for r in res if r["success"]) / len(res)
        walls = [j["wall"] for j in js if j["wall"]]
        nsucc = sum(1 for r in res if r["success"])
        jps = sum(walls) / nsucc if walls and nsucc else None
        name = k.replace("inter_object_relation_satisfaction-", "iors-").replace("configuration_matching-", "cm-").replace("camera_relocalization-obstructed-", "cr-")[:26]
        print(f"{name:26s} {st.mean(j['wl'] for j in js if j['wl'])/60:4.1f}m {n_att/len(wr):5.1f} {s_att:5.0f} {samp:11.0f} {demo:6.0f} {steps:5.0f} {coll:16.0f} {('sampler' if samp > coll else 'collector'):>10s} {100*succ:5.0f} {jps if jps is None else round(jps):>10}")
show("pilot")
show("large", ["touching_large"]); show("large_ab_noik", ["touching_large"]); show("large_ab_noramp", ["touching_large"])
