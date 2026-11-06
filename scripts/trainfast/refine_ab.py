"""Band refinement A/B, pooled per arm (refine_on = bracket start, refine_nearest, refine_off) and alias: written / attempts with Wilson
95 % intervals, two-sided Fisher p of each refine arm against refine_off, sampler s per attempt, the refine.* / band_evidence.* /
arm_band_mirrored.empty_F counters, the rejection mix, and the spread of the written instances' focus distance ahead of the
manipulation base (m; only written instances record their stance).
    refine_ab.py"""
import collections, glob, json, math, os, re, statistics as st
F = "/vision/u/minyeong/projects/isg_home/trainfast"
ARMS = ("refine_on", "refine_nearest", "refine_off")
KEYS = ("inter_object_relation_satisfaction-ontop_small_scale_01", "inter_object_relation_satisfaction-touching_01")
RX_CNT = re.compile(r"((?:refine|band_evidence|tilt_snap|arm_band_mirrored\.empty_F)[a-z_.]*) \((\d+)\)")
RX_REJ = re.compile(r"attempt \d+ rejected in \w+ after ([\d.]+)s: \[(\w+)\](?: candidate \d+: \[(\w+)\])?")
RX_OK = re.compile(r" written \(attempt \d+, ([\d.]+)s\)")

def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = w / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d

def fisher(a, n1, b, n2):
    """Two-sided Fisher exact p for a/n1 vs b/n2."""
    K, N = a + b, n1 + n2
    pmf = lambda x: math.comb(n1, x) * math.comb(n2, K - x) / math.comb(N, K)
    p0 = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(max(0, K - n2), min(K, n1) + 1) if pmf(x) <= p0 * (1 + 1e-9)))

def arm_stats(arm, key):
    r = dict(w=0, n=0, jobs=0, s=0.0, cnt=collections.Counter(), rej=collections.Counter(), ahead=[], per_job=[])
    for J in glob.glob(f"{F}/out/{arm}/{key}/g*"):
        p = os.path.join(J, "sample.log")
        if not os.path.exists(p):
            continue
        r["jobs"] += 1
        w0, n0 = r["w"], r["n"]
        for line in open(p, errors="replace"):
            m = RX_OK.search(line)
            if m:
                r["w"] += 1; r["n"] += 1; r["s"] += float(m.group(1))
            m = RX_REJ.search(line)
            if m:
                r["n"] += 1; r["s"] += float(m.group(1)); r["rej"][m.group(3) or m.group(2)] += 1
            if " kit: " in line:
                for k, v in RX_CNT.findall(line):
                    r["cnt"][k] += int(v)
        r["per_job"].append(f"{r['w'] - w0}/{r['n'] - n0}")
        for o in glob.glob(f"{J}/inst/*/*/instances/*.oracle.json"):
            try:
                m = (json.load(open(o)).get("stances") or {}).get("manipulation") or {}
                bx, by, yaw = m["base_pose2d"]; fx, fy = m["focus"][:2]
                r["ahead"].append(math.cos(yaw) * (fx - bx) + math.sin(yaw) * (fy - by))
            except Exception:
                pass
    return r

for key in KEYS:
    print(f"== {key.replace('inter_object_relation_satisfaction-', '')}")
    R = {arm: arm_stats(arm, key) for arm in ARMS}
    off = R["refine_off"]
    for arm in ARMS:
        r = R[arm]; w, n = r["w"], r["n"]
        lo, hi = wilson(w, n)
        fp = "" if arm == "refine_off" else f"  Fisher p vs off {fisher(w, n, off['w'], off['n']):.2g}"
        a = sorted(r["ahead"])
        sp = (f"focus ahead m: median {st.median(a):.2f}, range {a[0]:.2f}-{a[-1]:.2f} (n {len(a)})" if a else "focus ahead: -")
        print(f"  {arm:15s} jobs {r['jobs']:2d}  written {w}/{n} = {100 * w / max(n, 1):.1f} % (95 % {100 * lo:.1f}-{100 * hi:.1f}){fp}  "
              f"s/attempt {r['s'] / max(n, 1):.0f}  written per sampler-h {3600 * w / max(r['s'], 1):.1f}  {sp}")
        print("     per job: " + ", ".join(r["per_job"]))
        if r["cnt"]:
            print("     counters: " + ", ".join(f"{k}={v}" for k, v in sorted(r["cnt"].items())))
        rj = n - w
        if rj:
            print("     rejections: " + ", ".join(f"{k} {100 * v / rj:.0f} %" for k, v in r["rej"].most_common(6)))
