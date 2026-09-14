"""Build the grasp-library job list from the curated plan: (task, category, model) with pipeline, GraspGen mode and
task/object-specific constraint type. Cap 12 models per category, preferring gripper-friendly models."""
import json, re
OUT = "/tmp/glib"; EXP = "/tmp/grasp_exp"
src = open(f"{EXP}/curated_plan.py").read()
ns = {}; exec(src.split("used = {}")[0], ns)
PLAN, P, C = ns["PLAN"], ns["P"], ns["C"]
O = json.load(open(f"{EXP}/category_table.json"))["objects"]
CAP = 12
FLAGGED_OK = {"lily", "dahlia_flower"}          # not 'ready' only because of the automatic non-VRay material flag; render fine

# GraspGen scoring modes (from the taxonomy + overrides), only for the in-hand tasks
MODES = {}
for task, spec in PLAN.items():
    for k, v in spec.get("mode", {}).items():
        MODES[k] = v.split(" ")[0]

# constraint type per task / category
def constraint_for(task, cat):
    if task.startswith(("object_scaling", "mental_rotation")):
        return "floating"                                   # in-hand tasks: no table, GraspGen
    if "attachment" in task:
        return "attachment_free"                            # keep the mating side (attachment meta-link) uncovered
    if "inside" in task and cat in ("sunflower", "lily", "dahlia_flower"):
        return "flower_stem_top"                            # grasp high on the stem / head so the stem can go into the vase
    if "inside" in task:
        return "tabletop_insert"                            # tabletop + fingertips above the object bottom (already implied) + compact
    return "tabletop"                                       # ontop / under / nextto / disembedding


def pick_models(cat):
    ms = [o for o in O.values() if o["category"] == cat and o["bbox"] and (o["ready"] or cat in FLAGGED_OK)]
    def key(o):
        a, b, c = sorted(o["bbox"])
        return (0 if b <= 0.095 else 1, 0 if a <= 0.085 else 1, a)      # width-fit first, then pinchable, then thinnest
    ms.sort(key=key)
    return ms[:CAP]


jobs = []
for task, spec in PLAN.items():
    if "articulation" in task:
        continue
    pipeline = "graspgen" if task.startswith(("object_scaling", "mental_rotation")) else "a1"
    for entry in spec["main"]:
        cat = entry.split(" (")[0].split(" +")[0]
        cats = [cat] if "+" not in entry else [x.strip() for x in entry.split("(")[0].split("+")]
        for c in cats:
            for o in pick_models(c):
                jobs.append({"task": task.split(" (")[0].split(" -")[0].strip(), "category": c, "model": o["model"], "key": f"{c}__{o['model']}",
                             "pipeline": pipeline, "mode": MODES.get(c, "C") if pipeline == "graspgen" else None,
                             "constraint": constraint_for(task, c), "bbox": o["bbox"], "ready": o["ready"]})
json.dump(jobs, open(f"{OUT}/models.json", "w"), indent=1)
import collections
print(len(jobs), "jobs;", len({j['key'] for j in jobs}), "unique models;", collections.Counter(j["pipeline"] for j in jobs), collections.Counter(j["constraint"] for j in jobs))
print(collections.Counter(j["task"] for j in jobs))
