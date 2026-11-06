"""trainfast: camera relocalization A/B - per (tag, alias): attempts, written, s per attempt, sampler s per written
instance, rejections by reason, summed candidate failures, the field timing, demos / successes.
    python3 cr_ab.py [tag ...]   (default: cr_fast cr_off pilot)"""
import ast, collections, glob, json, os, re, sys
F = "/vision/u/minyeong/projects/isg_home/trainfast"
RX_REJ = re.compile(r"attempt (\d+) rejected in (\S+) after ([\d.]+)s: \[([^\]]+)\] ?(.*)$")
RX_WR = re.compile(r" written \(attempt (\d+), ([\d.]+)s\)")
tags = sys.argv[1:] or ["cr_fast", "cr_off", "pilot"]
for tag in tags:
    for kd in sorted(glob.glob(f"{F}/out/{tag}/camera_relocalization*")):
        att, wr, rej_s, reason, fails, field, ret = [], [], 0.0, collections.Counter(), collections.Counter(), [], 0
        demos = succ = tb = 0
        for J in glob.glob(f"{kd}/g*"):
            p = f"{J}/sample.log"
            if os.path.exists(p):
                for line in open(p, errors="replace"):
                    m = RX_REJ.search(line)
                    if m:
                        att.append(float(m.group(3))); reason[m.group(4)] += 1; rej_s += float(m.group(3))
                        mm = re.search(r"(\{[^{}]*\})", m.group(5))
                        if mm:
                            try:
                                for k, v in ast.literal_eval(mm.group(1)).items():
                                    if isinstance(v, int): fails[k] += v
                            except Exception:
                                pass
                        continue
                    m = RX_WR.search(line)
                    if m:
                        att.append(float(m.group(2))); wr.append(float(m.group(2)))
                    if line.startswith("Traceback"):
                        tb += 1
            for s in glob.glob(f"{J}/inst/*/*/instances/*.summary.json"):
                try:
                    d = json.load(open(s))
                except Exception:
                    continue
                cf = d.get("cr_fast") or {}
                if cf.get("field"): field.append(cf["field"]["s"])
                for k, v in (d.get("candidate_failures") or {}).items():
                    if k.startswith("obstacle"): ret += v
            p = f"{J}/results.jsonl"
            if os.path.exists(p):
                for line in open(p):
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    demos += 1; succ += bool(r.get("success"))
        n = len(att)
        tot = sum(att)
        print(f"{tag:8s} {os.path.basename(kd)[22:]:28s} attempts {n:4d} written {len(wr):3d} "
              f"s/att {tot / n if n else 0:6.1f} sampler-s/written {tot / len(wr) if wr else float('nan'):7.0f} "
              f"demos {demos:3d} succ {succ:3d} tb {tb} field_s {sum(field) / len(field) if field else float('nan'):.2f} "
              f"obstacle-retries-in-written {ret}")
        print(f"         rejections {dict(reason.most_common(6))}")
        print(f"         candidate failures {dict(fails.most_common(9))}")
