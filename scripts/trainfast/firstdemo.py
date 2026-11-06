import glob, json, re, os, datetime as dt
F = "/vision/u/minyeong/projects/isg_home/trainfast"
RX_REJ = re.compile(r"instance (\d+) attempt (\d+) rejected in (\S+) after ([\d.]+)s: \[?([^\]\n]*)")
RX_W = re.compile(r"instance (\d+) written \(attempt (\d+), ([\d.]+)s\)")
RX_WL = re.compile(r"world loaded for \S+ in ([\d.]+)s")
print(f"{'alias':22s} {'world':>6s} {'to 1st inst':>11s} {'attempts':>8s} {'rejections before 1st instance':40s} {'1st demo':>8s} {'steps':>6s} {'total':>6s}")
for J in sorted(glob.glob(f"{F}/out/prebuild/*/g*")):
    if J.endswith(".cancelled") or not os.path.exists(J + "/results.jsonl"):
        continue
    log = open(J + "/sample.log", errors="replace").read()
    wl = float(RX_WL.search(log).group(1))
    s_att, n_att, rej = 0.0, 0, []
    for line in log.splitlines():
        m = RX_REJ.search(line)
        if m:
            s_att += float(m.group(4)); n_att += 1; rej.append(f"{m.group(3)}:{m.group(5).split(':')[0].split(',')[0][:18]}")
            continue
        m = RX_W.search(line)
        if m:
            s_att += float(m.group(3)); n_att += 1
            break
    first = json.loads(open(J + "/results.jsonl").readline())
    meta = json.load(open(glob.glob(f"{J}/demo/{first['instance']}/meta.json")[0]))
    start = dt.datetime.fromisoformat(re.search(r"start=(\S+)", open(J + "/job.txt").readline()).group(1)).timestamp()
    fin = dt.datetime.fromisoformat(first["finished"]).timestamp()
    k = os.path.basename(os.path.dirname(J)).replace("inter_object_relation_satisfaction-", "iors-")[:22]
    print(f"{k:22s} {wl/60:5.1f}m {s_att/60:10.1f}m {n_att:8d} {', '.join(rej)[:40]:40s} {meta['timings']['episode_s']/60:7.1f}m {meta['n_steps']:6d} {(fin-start)/60:5.1f}m")
