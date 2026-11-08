"""i7-diag (2026-11-08, owner request "two diagnoses"): CPU-only tabulation of fast-flow job dirs, per alias and job.
Read-only. Run on the cluster (sacct for GPU-h):  ssh minyeong@svl11 python3 - floor|t04 [tag ...] < diag_floor_t04.py

floor: per (tag, alias, job) of the six IORS large floor aliases - trunk, slurm state, att / written / ok, GPU-h, rejections,
       demo failures, the setup error, missing_map / MAP_REQUEST / LEGACY_BAND_BLOCKED lines.
t04:   per job of the given tags - rejections per alias, and for target_placement the sub-reasons ("why" strings) counted
       from sample.log, with the time each rejected attempt cost (ATTEMPT lines)."""
import collections, glob, json, os, re, subprocess, sys

ROOTS = ["/vision/u/minyeong/projects/isg_home/oracle_fix/out", "/vision/u/shiyuc/minyeong/isgf_root/out"]
FLOOR = ("under_large_scale_01", "under_large_scale_02", "under_large_scale_03", "ontop_large_scale_01", "ontop_large_scale_02",
         "touching_large_scale_02")


def job_dirs(tag, pat="*"):
    out, seen = [], set()
    for r in ROOTS:
        for d in sorted(glob.glob(f"{r}/{tag}/{pat}/g*")):
            rel = d[len(r):]
            if rel in seen:
                continue
            seen.add(rel)
            out.append(d)
    return out


def sacct(ids):
    if not ids:
        return {}
    res = subprocess.run(["sacct", "-j", ",".join(ids), "-X", "-n", "-P", "-o", "JobID,State,ElapsedRaw"], capture_output=True,
                         text=True).stdout
    out = {}
    for l in res.splitlines():
        p = l.split("|")
        if len(p) == 3 and p[2].isdigit():
            out[p[0]] = (p[1], int(p[2]))
    return out


def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


def trunk_of(d):
    m = re.search(r"trunk_[0-9a-f]+", read(d + "/job.env"))
    return m.group(0) if m else "?"


def results(d):
    fails = collections.Counter()
    for l in read(d + "/results.jsonl").splitlines():
        try:
            r = json.loads(l)
        except Exception:
            continue
        if not r.get("success"):
            fails[f"{r.get('status')}:{r.get('reason')}"] += 1
    return fails


def floor(tags):
    for tag in tags:
        ds = [d for d in job_dirs(tag) if any(d.split("/")[-2].endswith(a) for a in FLOOR)]
        ids = {d: read(d + "/jobid").strip() for d in ds}
        sa = sacct([i for i in ids.values() if i])
        for d in ds:
            alias = d.split("/")[-2].replace("inter_object_relation_satisfaction-", "")
            s = {}
            if os.path.exists(d + "/job_stats.json"):
                s = json.load(open(d + "/job_stats.json"))
            st, el = sa.get(ids[d], ("?", 0))
            lg = read(d + "/sample.log")
            setup = re.findall(r'"error": "(setup rejected[^"]{0,400})', lg)
            mr = sorted(set(re.findall(r"MAP_REQUEST (\{[^}]*\})", lg)))
            mm = collections.Counter(re.findall(r"\[missing_map\] [^:]*: ([^\"\n]{0,160})", lg))
            lbb = lg.count("LEGACY_BAND_BLOCKED")
            where = "shiyuc" if d.startswith("/vision/u/shiyuc") else ("copied" if os.path.exists(d + "/COPIED_FROM_SHIYUC") else "minyeong")
            ok = s.get("successes", 0) or 0
            gh = el / 3600.0
            print(f"{tag:14s} {alias:22s} {d.split('/')[-1]:15s} job {ids[d]} {st:10s} {where:8s} {trunk_of(d):15s} "
                  f"att {s.get('attempts', '-')} wr {s.get('written', '-')} ok {s.get('successes', '-')} gpu-h {gh:.2f}"
                  + (f" = {gh / ok:.2f}/success" if ok else ""))
            if s.get("rejections"):
                print(f"    rejections {sorted(s['rejections'].items(), key=lambda x: -x[1])[:6]}")
            f = results(d)
            if f:
                print(f"    demo failures {f.most_common(5)}")
            for e in setup[:1]:
                print(f"    setup: {e[:400]}")
            for m in mr[:4]:
                print(f"    MAP_REQUEST {m}")
            for k, v in mm.most_common(3):
                print(f"    missing_map x{v}: {k}")
            if lbb:
                print(f"    LEGACY_BAND_BLOCKED lines {lbb}")


ATT = re.compile(r"instance (\d+) attempt (\d+) rejected in (\w+) after ([\d.]+)s: \[(\w+)\] (.*)$")
CAND = re.compile(r"candidate (\d+): (?:\[(\w+)\] )?(.*?)(?=; candidate \d+: |$)")
MT = re.compile(r"MAP_TILT .*?: empty F at so([\d.]+)\|t([\d.-]+)\|k([\d.-]+) ([\d.-]+),([\d.-]+) arm (\w+) -> (\w+)")
WR = re.compile(r"instance (\d+) attempt (\d+) (?:written|accepted)")


def why_class(reason, why):
    if "empty feasible set F at the focus" in why:
        return "grasp_unreachable: empty F at the focus (target init grasp, every (yaw, grasp) fails exact IK)"
    m = re.search(r"push stances: \{'candidates': (\d+), 'map_feasible': (\d+), 'collision_free': (\d+)\}", why)
    if "touching partner draw: no push stance" in why:
        if m and int(m.group(3)) == 0 and int(m.group(2)) > 0:
            return "contact_unreachable: partner draw - push stances fit the touch_push map but none collision-free"
        if m and int(m.group(2)) == 0:
            return "contact_unreachable: partner draw - no push stance fits the touch_push map"
        return "contact_unreachable: partner draw - other"
    w = re.sub(r"-?\d+\.\d+", "#", why)
    w = re.sub(r"\b\d+\b", "#", w)
    return f"{reason}: {w[:150]}"


def t04(tags, pat="*touching_04"):
    for tag in tags:
        for d in job_dirs(tag, pat):
            s = json.load(open(d + "/job_stats.json")) if os.path.exists(d + "/job_stats.json") else {}
            jid = read(d + "/jobid").strip()
            st, el = sacct([jid]).get(jid, ("?", 0)) if jid else ("?", 0)
            lg = read(d + "/sample.log").splitlines()
            rej_t = collections.defaultdict(list)
            sub = collections.Counter(); sub_att = collections.Counter(); ncand = collections.Counter()
            mt = collections.Counter(); mt_x = collections.Counter(); mt_k = collections.Counter()
            for l in lg:
                m = ATT.search(l)
                if m:
                    reason, rest, t = m.group(5), m.group(6), float(m.group(4))
                    rej_t[reason].append(t)
                    if reason == "target_placement":
                        cs = CAND.findall(rest)
                        ncand[len(cs)] += 1
                        seen = set()
                        for _ci, cr, cw in cs:
                            k = why_class(cr or "-", cw)
                            sub[k] += 1
                            seen.add(k)
                        for k in seen:
                            sub_att[k] += 1
                        if not cs:
                            sub[why_class("-", rest)] += 1
                    continue
                m = MT.search(l)
                if m:
                    mt[m.group(7)] += 1
                    mt_x[round(float(m.group(4)), 2)] += 1
                    mt_k[f"t{m.group(2)}/k{m.group(3)}"] += 1
            n_att = s.get("attempts", "-")
            gh = el / 3600.0
            print(f"== {tag} {d.split('/')[-1]} job {jid} {st} {trunk_of(d)} att {n_att} wr {s.get('written', '-')} ok {s.get('successes', '-')} "
                  f"gpu-h {gh:.2f} attempts_s {s.get('attempts_s', 0):.0f} reject_s {s.get('reject_s', 0):.0f}")
            for r, ts in sorted(rej_t.items(), key=lambda x: -len(x[1])):
                ts2 = sorted(ts)
                print(f"   {r:28s} {len(ts):4d}  s/attempt mean {sum(ts) / len(ts):6.1f} median {ts2[len(ts2) // 2]:6.1f} total {sum(ts):7.0f}")
            if sub:
                print(f"   target_placement: candidates shown per rejection (last 3 of <= 6 tried) {sorted(ncand.items())}")
                for k, v in sub.most_common(8):
                    print(f"     {v:4d} cand / {sub_att[k]:3d} att  {k}")
            if mt:
                print(f"   MAP_TILT empty F (inherited columns): {dict(mt)}; cell depth x {sorted(mt_x.items())}; cols {mt_k.most_common(6)}")
            f = results(d)
            if f:
                print(f"   demo failures {f.most_common(5)}")


PDC = re.compile(r"place depth cap \(owner 3.6\): none of (\d+) candidates .*?\(depths ([\d.]+)-([\d.]+) m, caps (?:([\d.]+)-([\d.]+) m|none)\)")


def kit_counters(d, prefix):
    """The job's kit counters with ``prefix``, summed over the per-attempt kit lines of sample.log ("name (n)")."""
    k = collections.Counter()
    for l in read(d + "/sample.log").splitlines():
        if " kit: " in l:
            for name, n in re.findall(r"(" + re.escape(prefix) + r"[\w.]*) \((\d+)\)", l):
                k[name] += int(n)
    return k


def pdc(tags, pat="*touching_04"):
    """i7-contact3's place depth cap (owner 3.6) on touching_04: the cap counters and the all-removed depth / cap ranges."""
    for tag in tags:
        for d in job_dirs(tag, pat):
            k = kit_counters(d, "place_depth_cap")
            depths, caps = collections.Counter(), []
            for l in read(d + "/sample.log").splitlines():
                m = PDC.search(l)
                if m:
                    depths[f"{m.group(2)}-{m.group(3)}"] += 1
                    caps.append(None if m.group(4) is None else (float(m.group(4)), float(m.group(5))))
            print(f"== {tag} {d.split('/')[-1]}: {dict(sorted(k.items()))}")
            if depths:
                cc = [c for c in caps if c]
                print(f"   all_removed attempts {len(caps)}: focus depths {dict(depths)}; caps min {min(c[0] for c in cc):.2f} max {max(c[1] for c in cc):.2f}; "
                      f"no cap {sum(1 for c in caps if c is None)}")


if __name__ == "__main__":
    mode, tags = sys.argv[1], sys.argv[2:]
    if mode == "floor":
        floor(tags)
    elif mode == "pdc":
        pdc(tags)
    elif mode == "t04":
        t04(tags)
    elif mode == "t04any":
        t04(tags[1:], tags[0])
