#!/usr/bin/env python3
"""Which object categories the other iSpatialGym tasks already use, per selection round (for the IORS-4 / touching
selection; the current IORS-4 rows of the sheet are excluded because they are what is being re-selected).

Sources (same recipe as runlogs/world_alignment_selection_20260911/scripts/used_categories.py):
* ``runlogs/grasp_library_20260905/selection_manifest.csv`` - every (task, round, category, model) of the grasp-library
  selection rounds 1-3 (targets of object scaling, mental rotation, disembedding, attachment, IORS; articulation).
* ``runlogs/grasp_library_20260905/select/task_object_selection.csv`` - partner / female / pool / spec rows (a partner
  category inherits the round of its target; spec-pinned fixed roles count as round 1).
* ``ispatialgym/assets/task_specs/task_custom_lists.json`` - pinned models of every role (round 1).
* ``runlogs/world_alignment_selection_20260911/candidate_categories.json`` - the world-alignment selection that runs in
  parallel (its ``groups`` are recorded as round 1 candidates so that the touching round 1 avoids them too).

Output: ``used_categories.json`` = {category: [{"task": ..., "round": n, "role": ...}, ...]} and a summary on stdout.
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LIB = os.path.join(REPO, "runlogs", "grasp_library_20260905")
SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
WA = os.path.join(REPO, "runlogs", "world_alignment_selection_20260911", "candidate_categories.json")
OUT = os.path.join(HERE, "..", "used_categories.json")

TASK_OF_MANIFEST = {"object_scaling": "object_scaling-0", "mental_rotation": "mental_rotation-0", "disembedding": "disembedding-0",
                    "configuration_matching": "configuration_matching-3", "articulation": "configuration_matching-1/-2"}
IORS_ID = {"ontop": 0, "under": 1, "inside": 2, "nextto": 3, "touching": 4}
SELF = "inter_object_relation_satisfaction-4"


def main():
    used = defaultdict(list)

    def add(cat, task, rnd, role):
        if not cat or task == SELF:
            return
        e = {"task": task, "round": int(rnd), "role": role}
        if e not in used[cat]:
            used[cat].append(e)

    with open(os.path.join(LIB, "selection_manifest.csv"), newline="") as f:
        for r in csv.DictReader(f):
            task = TASK_OF_MANIFEST.get(r["task"], r["task"])
            if r["task"] == "IORS":
                task = f"inter_object_relation_satisfaction-{IORS_ID[r['relation']]}" if r["relation"] in IORS_ID else "inter_object_relation_satisfaction"
            add(r["category"], task, r["round"], "target")
    with open(os.path.join(LIB, "select", "task_object_selection.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    task = role = None
    for r in rows:
        if r["task"]:
            task, role = r["task"].split(" ")[0], r["role"]
            continue
        if task is None or task == SELF:
            continue
        cat = r["category"]
        if not cat or re.match(r"^\d+ \(\d+\)$", cat):
            continue
        m = re.search(r"round (\d)", r["note"])
        rnd = int(m.group(1)) if m else 1
        if role in ("partner", "female", "tray_source", "tray_dest", "same_category_pool", "distractor_pool", "blocks", "artic"):
            add(cat, task, rnd if role in ("partner", "female") else 1, role)
        elif role in ("target", "male"):
            add(cat, task, rnd, role)
    task = role = None
    for r in rows:
        if r["task"]:
            task, role = r["task"].split(" ")[0], r["role"]
            continue
        if task is None or task == SELF or role not in ("target", "male"):
            continue
        m = re.search(r"round (\d)", r["note"])
        rnd = int(m.group(1)) if m else 1
        for p in re.findall(r"-> ([a-z_]+)(?:/[a-z]+)?", r["pair"]):
            add(p, task, rnd, "partner")
    with open(SPEC) as f:
        spec = json.load(f)
    for prob, body in spec.items():
        if prob.startswith("_") or prob == SELF:
            continue
        for role, cfg in (body.get("roles") or {}).items():
            for cat in (cfg.get("models") or {}):
                add(cat, prob, 1, f"spec:{role}")
    if os.path.exists(WA):
        with open(WA) as f:
            wa = json.load(f)
        for cat, d in wa.items():
            for g in d.get("groups", []):
                add(cat, f"world_alignment-{g[-1]}", d.get("round", 1), "candidate of the parallel selection, not final")
    out = {c: sorted(v, key=lambda e: (e["round"], e["task"])) for c, v in sorted(used.items())}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    by_round = defaultdict(set)
    for c, es in out.items():
        for e in es:
            by_round[e["round"]].add(c)
    print(f"{len(out)} categories used by other tasks; per round: " + ", ".join(f"r{k}: {len(v)}" for k, v in sorted(by_round.items())))
    print("round-1 categories:", " ".join(sorted(by_round[1])))
    print("round-2 categories:", " ".join(sorted(by_round[2])))
    print("round-3 categories:", " ".join(sorted(by_round[3])))
    return out


if __name__ == "__main__":
    main()
