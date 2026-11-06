"""Collector per-step cost split (meta fast.step_costs / fast.world_sync, always on since 2026-11-01 06:55) over the demos of the given
tags: per family and overall, ms per env step of env step / monitors / state dump / recorder / QA, the planning + rest share of the
episode, world syncs, and the load. Medians over demos; shares = sums over demos (time-weighted).
    step_costs.py <tag> [<tag> ...]"""
import collections, glob, json, os, statistics as st, sys

F = "/vision/u/minyeong/projects/isg_home/trainfast"
PARTS = ("env_step_s", "monitors_s", "dump_s", "record_s", "qa_s")


def family(alias):
    a = alias.split("-")[0] if alias else "?"
    return {"inter_object_relation_satisfaction": "iors", "camera_relocalization": "cr", "articulation": "artic"}.get(a, a)


def rows(tags):
    for tag in tags:
        for p in glob.glob(f"{F}/out/{tag}/*/g*/demo/*/meta.json"):
            try:
                m = json.load(open(p))
            except Exception:  # noqa: BLE001 - a demo still being written
                continue
            fa = m.get("fast") or {}
            sc, t = fa.get("step_costs"), m.get("timings") or {}
            if not sc or not sc.get("steps") or not t.get("episode_s"):
                continue
            key = p.split("/out/")[1].split("/")[1]
            large = "large" in key
            yield {"tag": tag, "key": key, "fam": family(m.get("alias") or key) + ("_large" if large else ""), "sc": sc,
                   "ws": fa.get("world_sync") or {}, "t": t, "crop": (m.get("scene_crop") or {}).get("mode", "?") if isinstance(m.get("scene_crop"), dict) else str(m.get("scene_crop")),
                   "host": m.get("hostname", "?"), "ok": bool(m.get("success"))}


def summary(rs, label):
    n = len(rs)
    if not n:
        return
    steps = sum(r["sc"]["steps"] for r in rs)
    ep = sum(r["t"]["episode_s"] for r in rs)
    tot = sum(r["t"].get("total_s", 0) for r in rs)
    load = sum(r["t"].get("load_s", 0) for r in rs)
    parts = {k: sum(r["sc"].get(k, 0.0) for r in rs) for k in PARTS}
    per_step = sum(parts.values())
    ws = sum(r["ws"].get("s", 0.0) for r in rs)
    rest = ep - per_step - ws
    ms = lambda k: 1000 * parts[k] / steps
    med_steps = st.median(r["sc"]["steps"] for r in rs)
    floats = sum(r["sc"].get("dump_floats", 0) for r in rs) / steps
    print(f"{label:<22} demos {n:4d}  steps/demo {med_steps:5.0f}  per step ms: env {ms('env_step_s'):5.1f}  qa {ms('qa_s'):5.1f}  "
          f"record {ms('record_s'):4.1f}  dump {ms('dump_s'):4.1f}  monitors {ms('monitors_s'):4.1f}  | dump floats/step {floats:6.0f}")
    print(f"{'':<22} episode share: env {100*parts['env_step_s']/ep:4.1f} %  qa {100*parts['qa_s']/ep:4.1f} %  record {100*parts['record_s']/ep:4.1f} %  "
          f"dump {100*parts['dump_s']/ep:4.1f} %  monitors {100*parts['monitors_s']/ep:4.1f} %  world sync {100*ws/ep:4.1f} %  "
          f"planning+rest {100*rest/ep:4.1f} %  | episode {ep/n:5.0f} s/demo, load {load/n:4.0f} s/demo, total {tot/n:5.0f} s/demo"
          f"  world syncs/demo {sum(r['ws'].get('calls', 0) for r in rs)/n:4.1f} ({1000*ws/max(1, sum(r['ws'].get('calls', 0) for r in rs)):4.0f} ms each)")


def main(tags):
    rs = list(rows(tags))
    print(f"tags {' '.join(tags)}: {len(rs)} demos with step_costs")
    summary(rs, "ALL")
    by = collections.defaultdict(list)
    for r in rs:
        by[r["fam"]].append(r)
    for f in sorted(by, key=lambda f: -len(by[f])):
        summary(by[f], f)
    print("-- by scene crop mode")
    byc = collections.defaultdict(list)
    for r in rs:
        byc[r["crop"]].append(r)
    for c in sorted(byc):
        summary(byc[c], f"crop={c}")
    print("-- by GPU host class")
    byh = collections.defaultdict(list)
    for r in rs:
        byh["viscam (L40S)" if r["host"].startswith("viscam") else "svl (TITAN RTX)"].append(r)
    for h in sorted(byh):
        summary(byh[h], h)


if __name__ == "__main__":
    main(sys.argv[1:] or ["pilot2", "pilot3", "pilot4", "crop2", "crop4"])
