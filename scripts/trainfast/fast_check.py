#!/usr/bin/env python3
"""trainfast: per-job statistics after the job (job_stats.json) - sampler attempts / rejections / timings from sample.log,
demo results from results.jsonl + meta.json, process walls from job.txt. No provenance rule (the code differs from the counted demo).

    fast_check.py <job dir>
"""
import glob
import json
import os
import re
import sys

RX_WRITTEN = re.compile(r"\] (\S+): instance (\d+) written \(attempt (\d+), ([\d.]+)s\)")
RX_REJ = re.compile(r"\] (\S+): instance (\d+) attempt (\d+) rejected in (\S+) after ([\d.]+)s: \[?([^\]\n]*)")
RX_WORLD = re.compile(r"world loaded for (\S+) in ([\d.]+)s")
RX_START = re.compile(r"start_per_attempt: start drawn in ([\d.]+)s")
RX_FAMILY = re.compile(r"instance family: (\S+) child (\d+) of (\S+) in ([\d.]+)s")


def main(job):
    st = {"job": os.path.basename(job), "attempts": 0, "attempts_s": 0.0, "written": 0, "written_s": 0.0, "rejections": {}, "reject_stages": {},
          "reject_s": 0.0, "world_load_s": [], "start_draw_s": [], "family_children": 0}
    p = os.path.join(job, "sample.log")
    if os.path.exists(p):
        for line in open(p, errors="replace"):
            m = RX_REJ.search(line)
            if m:
                st["attempts"] += 1
                s = float(m.group(5))
                st["attempts_s"] += s
                st["reject_s"] += s
                st["reject_stages"][m.group(4)] = st["reject_stages"].get(m.group(4), 0) + 1
                reason = m.group(6).split(",")[0].split(":")[0].strip()
                st["rejections"][reason] = st["rejections"].get(reason, 0) + 1
                continue
            m = RX_WRITTEN.search(line)
            if m:
                st["attempts"] += 1
                st["written"] += 1
                st["attempts_s"] += float(m.group(4))
                st["written_s"] += float(m.group(4))
                continue
            m = RX_WORLD.search(line)
            if m:
                st["world_load_s"].append(float(m.group(2)))
                continue
            m = RX_START.search(line)
            if m:
                st["start_draw_s"].append(float(m.group(1)))
                continue
            if RX_FAMILY.search(line):
                st["family_children"] += 1
    res = []
    p = os.path.join(job, "results.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                res.append(json.loads(line))
            except ValueError:
                pass
    st["demos"] = len(res)
    st["successes"] = sum(1 for r in res if r.get("success"))
    st["demo_wall_s"] = [r.get("wall_s") for r in res]
    st["demo_steps"] = [r.get("n_steps") for r in res]
    timings = []
    for m in glob.glob(os.path.join(job, "demo", "*", "meta.json")):
        try:
            d = json.load(open(m))
            t = d.get("timings") or {}
            timings.append({"instance": d.get("instance_id"), "success": d.get("success"), "wall_s": d.get("wall_time_s"), "n_steps": d.get("n_steps"),
                            "n_action_steps": d.get("n_action_steps"), "n_settle_steps": d.get("n_settle_steps"),
                            "timings": {k: v for k, v in t.items() if isinstance(v, (int, float))}})
        except Exception as e:  # noqa: BLE001
            timings.append({"meta": m, "error": repr(e)})
    st["demo_meta"] = timings
    walls = {"sample_wall_s": None, "demo_walls_s": [], "job_wall_s": None, "collector_walls_s": []}
    p = os.path.join(job, "job.txt")
    if os.path.exists(p):
        for line in open(p):
            if line.startswith("SAMPLE_EXIT"):
                m = re.search(r"wall_s=(\d+)", line)
                walls["sample_wall_s"] = int(m.group(1)) if m else None
            elif line.startswith("DEMO_EXIT"):
                m = re.search(r"wall_s=(\d+)", line)
                walls["demo_walls_s"].append(int(m.group(1)) if m else None)
            elif line.startswith("COLLECT_EXIT"):
                m = re.search(r"wall_s=(\d+)", line)
                walls["collector_walls_s"].append(int(m.group(1)) if m else None)
            elif line.startswith("ISGF_END"):
                m = re.search(r"wall_s=(\d+)", line)
                walls["job_wall_s"] = int(m.group(1)) if m else None
    st.update(walls)
    for k in ("world_load_s", "start_draw_s"):
        v = st[k]
        st[k] = {"n": len(v), "sum": round(sum(v), 1), "mean": round(sum(v) / len(v), 2) if v else None}
    json.dump(st, open(os.path.join(job, "job_stats.json"), "w"), indent=1)
    print(f"FAST_CHECK attempts={st['attempts']} written={st['written']} attempts_s={st['attempts_s']:.0f} demos={st['demos']} successes={st['successes']} "
          f"sample_wall={st['sample_wall_s']} demo_walls={st['demo_walls_s']} job_wall={st['job_wall_s']} rejections={st['rejections']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(os.path.abspath(sys.argv[1])))
