#!/usr/bin/env python
"""Campaign rooms: from the segmentation pass (<seg>/<task>/<instance>/rooms_visibility.json, one per demo) to the per-alias
room decision. A neighbour room is *seen* when any non-always-loaded object of it (floor, window, light, furniture) covered
at least --min-frac of a camera frame in any sampled step of any demo of the task; the candidate set = the template's task
rooms + the seen neighbours (+ --keep rooms). Writes <out>/visibility.md and <out>/rooms_override_candidate.json.

    python analyze_visibility.py [--root $ISG_VEC_ROOT] [--campaign <root>/campaigns/rooms] [--seg <campaign>/seg]
                                 [--out <campaign>/summary] [--tasks a,b] [--min-frac 0.0002] [--keep garden_0]

Inputs (every default = the 2026-10-28 campaign's layout): <root>/instances/task_ids.json (the task list, unless --tasks),
<root>/instances/eval/<task>.json (the eval instance; its TRO names the template whose -meta.json holds task_rooms /
load_room_instances / scene), <seg>/<task>/*/rooms_visibility.json (run_isg_vec --seg-visibility, one per episode).
"""
import argparse, collections, glob, json, os, time

DEFAULT_ROOT = os.environ.get("ISG_VEC_ROOT", "/vision/u/minyeong/projects/isg_home/vec_eval85")


def template_meta(instances_dir, task):
    ev = json.load(open(os.path.join(instances_dir, "eval", f"{task}.json")))["instances"][0]
    tro = ev["tro"]
    tdir = os.path.dirname(os.path.dirname(tro))
    tid = os.path.basename(tro).replace("-tro_state.json", "")
    return json.load(open(os.path.join(tdir, tid + "-meta.json")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="the vec_eval root: <root>/instances/{task_ids.json,eval/<task>.json} (env ISG_VEC_ROOT)")
    ap.add_argument("--campaign", default=None, help="the campaign folder (default <root>/campaigns/rooms)")
    ap.add_argument("--instances", default=None, help="the instance lists folder (default <root>/instances)")
    ap.add_argument("--seg", default=None, help="the segmentation pass outputs, <seg>/<task>/<episode>/rooms_visibility.json (default <campaign>/seg)")
    ap.add_argument("--out", default=None, help="where visibility.md and rooms_override_candidate.json go (default <campaign>/summary)")
    ap.add_argument("--tasks", default="", help="comma list of task aliases (default: every key of <instances>/task_ids.json)")
    ap.add_argument("--min-frac", type=float, default=0.0002, help="a room counts as seen above this fraction of a frame's pixels (0.0002 = 104 px of 720^2)")
    ap.add_argument("--keep", default="", help="comma list of room instances always kept (e.g. garden_0: the exterior, never dropped since the sunflower gate)")
    a = ap.parse_args()
    campaign = a.campaign or os.path.join(a.root, "campaigns", "rooms")
    instances = a.instances or os.path.join(a.root, "instances")
    seg = a.seg or os.path.join(campaign, "seg")
    out_dir = a.out or os.path.join(campaign, "summary")
    os.makedirs(out_dir, exist_ok=True)
    only = [t for t in a.tasks.split(",") if t]
    tasks = sorted(only) if only else sorted(json.load(open(os.path.join(instances, "task_ids.json"))))
    keep_always = {r for r in a.keep.split(",") if r}
    out = {"min_frac": a.min_frac, "generated": time.strftime("%Y-%m-%dT%H:%M"), "overrides": {}}
    md = [f"# Room visibility per task (segmentation pass; a room is seen above {a.min_frac * 100:.3f} % of a frame in any demo)\n",
          "| task | scene | task rooms | template rooms | seen neighbours (max % of a frame, demos seen / demos) | dropped | demos |", "|---|---|---|---|---|---|---|"]
    n_tasks_done = 0
    for t in tasks:
        files = sorted(glob.glob(f"{seg}/{t}/*/rooms_visibility.json"))
        meta = template_meta(instances, t)
        task_rooms = list(meta.get("task_rooms") or meta.get("rooms") or [])
        tmpl = list(meta.get("load_room_instances") or [])
        if not files:
            md.append(f"| `{t}` | {meta.get('scene')} | {task_rooms} | {len(tmpl)} | *no segmentation pass yet* | | 0 |")
            continue
        n_tasks_done += 1
        seen = collections.defaultdict(lambda: {"max_frac": 0.0, "demos": 0, "frames": 0, "cams": set()})
        for f in files:
            d = json.load(open(f))
            for r, st in (d.get("rooms") or {}).items():
                s = seen[r]
                s["max_frac"] = max(s["max_frac"], float(st.get("max_frac") or 0.0))
                s["frames"] += int(st.get("frames") or 0)
                s["cams"] |= set(st.get("cams") or [])
                if float(st.get("max_frac") or 0.0) >= a.min_frac:
                    s["demos"] += 1
        neighbours = [r for r in tmpl if r not in task_rooms]
        kept = [r for r in neighbours if seen[r]["max_frac"] >= a.min_frac or r in keep_always]
        dropped = [r for r in neighbours if r not in kept]
        rooms = task_rooms + kept
        # rooms seen that the template did not load (should not happen: not loaded -> not visible); report anyway
        extra = [r for r in seen if r not in tmpl and seen[r]["max_frac"] >= a.min_frac]
        out["overrides"][t] = {"rooms": rooms, "task_rooms": task_rooms, "template_rooms": tmpl, "dropped": dropped, "scene": meta.get("scene"), "demos": len(files),
                               "seen": {r: {"max_frac": round(seen[r]["max_frac"], 6), "demos_seen": seen[r]["demos"], "cams": sorted(seen[r]["cams"])} for r in neighbours},
                               "seen_not_in_template": extra}
        cell = ", ".join(f"{r} ({seen[r]['max_frac'] * 100:.2f} %, {seen[r]['demos']}/{len(files)})" for r in kept) or "—"
        md.append(f"| `{t}` | {meta.get('scene')} | {', '.join(task_rooms)} | {len(tmpl)} | {cell} | {', '.join(dropped) or '—'} | {len(files)} |")
    md.append(f"\n{n_tasks_done}/{len(tasks)} tasks with a segmentation pass. Rooms dropped in total: "
              f"{sum(len(v['dropped']) for v in out['overrides'].values())} of {sum(len(v['template_rooms']) - len(v['task_rooms']) for v in out['overrides'].values())} neighbour loads.")
    open(os.path.join(out_dir, "visibility.md"), "w").write("\n".join(md) + "\n")
    json.dump(out, open(os.path.join(out_dir, "rooms_override_candidate.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
