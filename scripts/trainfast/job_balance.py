"""Sampler vs collector balance per fast-flow job (job.txt walls + demo meta timings): collector busy share = the job's demo
total_s / the collectors' walls; the drain = job wall - sampler wall (collectors alone at the end). Per family, medians over jobs.
    job_balance.py <tag> [<tag> ...]"""
import collections, glob, json, re, statistics as st, sys

F = "/vision/u/minyeong/projects/isg_home/trainfast"
RX_S = re.compile(r"SAMPLE_EXIT \d+ wall_s=(\d+) written=(\d+)")
RX_C = re.compile(r"COLLECT_EXIT c\d+ q=\d+ rc=\d+ wall_s=(\d+) results=(\d+)")
RX_E = re.compile(r"ISGF_END \d+ wall_s=(\d+)")


def family(key):
    a = key.split("-")[0]
    f = {"inter_object_relation_satisfaction": "iors", "camera_relocalization": "cr", "articulation": "artic"}.get(a, a)
    return f + ("_large" if "large" in key else "")


def main(tags):
    by = collections.defaultdict(list)
    for tag in tags:
        for d in glob.glob(f"{F}/out/{tag}/*/g*"):
            try:
                txt = open(f"{d}/job.txt").read()
            except OSError:
                continue
            s, cs, e = RX_S.search(txt), RX_C.findall(txt), RX_E.search(txt)
            if not (s and cs and e):
                continue
            busy = 0.0
            nd = 0
            for p in glob.glob(f"{d}/demo/*/meta.json"):
                try:
                    t = json.load(open(p)).get("timings") or {}
                except Exception:  # noqa: BLE001
                    continue
                busy += float(t.get("total_s") or 0)
                nd += 1
            cw = sum(int(w) for w, _ in cs)
            if not cw or not nd:
                continue
            key = d.split("/out/")[1].split("/")[1]
            by[family(key)].append({"busy": busy / cw, "sw": int(s.group(1)), "jw": int(e.group(1)), "written": int(s.group(2)),
                                    "k": len(cs), "nd": nd, "tag": tag})
    allj = [j for v in by.values() for j in v]
    print(f"{len(allj)} jobs; collector busy share = demo total_s / collector walls; drain = job wall - sampler wall")
    for f in sorted(by, key=lambda f: -len(by[f])) + ["ALL"]:
        js = allj if f == "ALL" else by[f]
        print(f"{f:<24} jobs {len(js):3d}  collector busy median {100*st.median(j['busy'] for j in js):4.0f} % "
              f"(p10-p90 {100*sorted(j['busy'] for j in js)[len(js)//10]:3.0f}-{100*sorted(j['busy'] for j in js)[(9*len(js))//10]:3.0f})  "
              f"sampler wall {st.median(j['sw'] for j in js):5.0f} s  job wall {st.median(j['jw'] for j in js):5.0f} s  "
              f"drain {st.median(j['jw'] - j['sw'] for j in js):4.0f} s  demos/job {st.median(j['nd'] for j in js):3.0f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["pilot2", "pilot3", "pilot4", "crop2", "crop4"])
