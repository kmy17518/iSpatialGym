"""Aggregate validation results + latest reviewer verdicts over the selection manifest (rounds 1-3).
Writes selection_manifest.csv (one row per (task, category, model) selection) and rounds_summary.md."""
import json, glob, os, csv, collections
OUT = "/tmp/glib"
manifest = json.load(open(f"{OUT}/manifest.json"))
results = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{OUT}/results/*.json")}
review = {}
for p in sorted(glob.glob(f"{OUT}/review/*.json")):            # later waves override earlier reviews of the same key
    try:
        for m in json.load(open(p))["models"]: review[m["key"]] = m
    except Exception as e: print("bad review file", p, e)


def grade(key):
    r = results.get(key); rv = review.get(key) if (r and r.get("n_valid")) else None
    n_valid = r["n_valid"] if r else 0; tested = len(r["tested"]) if r else 0
    n_clean = n_strict = None; flags = []
    if rv:
        gs = rv.get("grasps", []); flags = sorted({f for g in gs for f in g.get("flags", [])})
        n_clean = sum(1 for g in gs if g.get("verdict") != "flag" or set(g.get("flags", [])) <= {"one_finger"})
        n_strict = sum(1 for g in gs if g.get("verdict") != "flag")
    return dict(tested=tested, n_valid=n_valid, n_clean=n_clean, n_clean_strict=n_strict, flags=flags, reviewed=rv is not None, error=(r or {}).get("error"))


rows = []
for m in manifest:
    key = m["reuse_of"] or m["key"]
    g = grade(key) if key else dict(tested=None, n_valid=None, n_clean=None, n_clean_strict=None, flags=[], reviewed=False, error=None)
    rows.append({**m, "result_key": key, **g})
with open(f"{OUT}/selection_manifest.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["round", "task", "relation", "category", "model", "result_key", "pipeline", "mode", "constraint", "scale", "reuse_of", "grasp_test", "partner", "tested", "n_valid", "n_clean", "n_clean_strict", "flags", "error"])
    for r in rows:
        w.writerow([r["round"], r["task"], r["relation"] or "", r["category"], r["model"], r["result_key"] or "", r["pipeline"] or "", r["mode"] or "", r["constraint"] or "", json.dumps(r["scale"]) if r["scale"] else "", r["reuse_of"] or "", r["grasp_test"], r["partner"] or "", r["tested"] if r["tested"] is not None else "", r["n_valid"] if r["n_valid"] is not None else "", r["n_clean"] if r["n_clean"] is not None else "", r["n_clean_strict"] if r["n_clean_strict"] is not None else "", ",".join(r["flags"]), r["error"] or ""])
json.dump(rows, open(f"{OUT}/selection_manifest.json", "w"), indent=1)


def verdict(n):
    return "ok" if n >= 5 else "thin" if n >= 3 else "weak" if n >= 1 else "drop"


L = ["# Category selection rounds 1-3: outcome", "",
     "round 1 = original plan (validated in the first two runs); round 2 = categories reused from another task (grasps re-tested only where the grasp method differs); round 3 = never-used categories, single-model categories allowed.",
     "L1 / L3 / L5 = models with >= 1 / >= 3 / 5 reviewer-clean grasps (lenient: `one_finger` ignored); S1 = models with >= 1 strictly clean grasp. Articulation rows have no grasp test (handle grasps on articulated links are a separate pipeline).", ""]
L += ["## Per task and round", "", "| task | round | categories | models | grasp tests run | AG-valid | L1 | L3 | L5 | S1 | categories ok / thin / weak / drop |", "|---|---|---|---|---|---|---|---|---|---|---|"]
tot = collections.Counter()
for task in ["object_scaling", "mental_rotation", "disembedding", "configuration_matching", "articulation", "IORS"]:
    for rnd in (1, 2, 3):
        rs = [r for r in rows if r["task"] == task and r["round"] == rnd]
        if not rs: continue
        cats = collections.defaultdict(list)
        for r in rs: cats[r["category"]].append(r)
        if task == "articulation":
            L.append(f"| {task} | {rnd} | {len(cats)} | {len(rs)} | n/a | | | | | | |"); continue
        v = collections.Counter(verdict(sum(1 for r in cc if (r["n_clean"] or 0) >= 1)) for cc in cats.values())
        d = dict(AG=sum(1 for r in rs if (r["n_valid"] or 0) >= 1), L1=sum(1 for r in rs if (r["n_clean"] or 0) >= 1), L3=sum(1 for r in rs if (r["n_clean"] or 0) >= 3), L5=sum(1 for r in rs if (r["n_clean"] or 0) >= 5), S1=sum(1 for r in rs if (r["n_clean_strict"] or 0) >= 1))
        tests = sum(1 for r in rs if r["grasp_test"] and not r["reuse_of"] and rnd > 1)
        L.append(f"| {task} | {rnd} | {len(cats)} | {len(rs)} | {tests if rnd > 1 else len(rs)} | {d['AG']} | {d['L1']} | {d['L3']} | {d['L5']} | {d['S1']} | {v['ok']} / {v['thin']} / {v['weak']} / {v['drop']} |")
L += ["", "## Per category", ""]
for task in ["object_scaling", "mental_rotation", "disembedding", "configuration_matching", "IORS", "articulation"]:
    L += [f"### {task}", "", "| round | relation | category | partner / note | models | tests | AG | L1 | L3 | L5 | S1 | verdict | flags |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rnd in (1, 2, 3):
        rs = [r for r in rows if r["task"] == task and r["round"] == rnd]
        cats = collections.defaultdict(list)
        for r in rs: cats[r["category"]].append(r)
        for c, cc in cats.items():
            if task == "articulation":
                L.append(f"| {rnd} | | {c} | {cc[0]['partner'] or ''} | {len(cc)} | n/a | | | | | | | |"); continue
            n1 = sum(1 for r in cc if (r["n_clean"] or 0) >= 1)
            fl = collections.Counter(f for r in cc for f in r["flags"])
            L.append(f"| {rnd} | {cc[0]['relation'] or ''} | {c} | {cc[0]['partner'] or ''} | {len(cc)} | {sum(1 for r in cc if r['grasp_test'] and not r['reuse_of'])} | {sum(1 for r in cc if (r['n_valid'] or 0) >= 1)} | {n1} | {sum(1 for r in cc if (r['n_clean'] or 0) >= 3)} | {sum(1 for r in cc if (r['n_clean'] or 0) >= 5)} | {sum(1 for r in cc if (r['n_clean_strict'] or 0) >= 1)} | {verdict(n1)} | {', '.join(f'{k} {v}' for k, v in fl.items())} |")
    L.append("")
usage = collections.Counter((r["category"], r["model"]) for r in rows)
L += ["## Model usage balance", "", f"Distinct models selected: {len(usage)}. Selection counts per model: " + ", ".join(f"{n}x: {c}" for n, c in sorted(collections.Counter(usage.values()).items())) + ".",
      "Categories used by more than one task: " + ", ".join(sorted({r['category'] for r in rows if len({x['task'] for x in rows if x['category'] == r['category']}) > 1})) + ".", ""]
L += ["## Notes on the selection", "",
      "* Round 2 picks stay disjoint across tasks (a category is reused by at most one extra task), and inside a reused category the least-used models are taken first, so no model is selected more than twice.",
      "* Round 2 grasp tests were skipped only where the grasp method was identical (bell_pepper and potato: GraspGen floating grasps already validated for object scaling are reused for mental rotation). Everything else in round 2 changes pipeline or constraint (tabletop -> floating, floating -> tabletop, tabletop -> insert / attachment_free) and was re-tested.",
      "* Attachment round 2 has only 2 categories: no category used by another task carries a male attachment meta-link; bottle_of_water and soda_cup are female parts (the cap attaches to them) and both failed (task_violation / no AG). Attachment round 3 has 10 categories but 9 of them have exactly one model with an attachment link; menu / bow / fire_alarm / antlers are sheets or thin discs and fail as license_plate did.",
      "* Articulation round 2 is empty (no category used by another task is openable + articulated); round 3 lists 10 unused openable articulated categories (door, openable_window, sliding_door with models in the challenge scenes; recycling_bin, blender, cedar_chest, toolbox, cooler, violin_case, laptop). No grasp tests: handle grasps on articulated links are a separate pipeline.",
      "* Round 3 flowers: lily needed its own stem window (55-85% of the head-to-stem axis, blooms and leaves occupy the top 60%); 6 of 10 models produce stem candidates, 4 hold.",
      "* Round 3 categories with a single ready model are counted as `weak` by the verdict rule (>= 1 clean model); the user allowed single-model categories in this round, so `weak` is acceptable there.", ""]
open(f"{OUT}/rounds_summary.md", "w").write("\n".join(L))
print("\n".join(L[:40]))
