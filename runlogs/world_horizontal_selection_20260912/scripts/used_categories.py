#!/usr/bin/env python3
"""Which object categories the other iSpatialGym tasks already use, per selection round (for the world-horizontal
round rules: round 1 must not overlap any other task's round-1 categories).

Same sources as runlogs/world_alignment_selection_20260911/scripts/used_categories.py (grasp-library selection
manifest, the sheet's partner / pool / spec rows, the spec pins) **plus** the world-alignment rounds 1-3 of that run
(``selection.json``: table surface and world vertical are other tasks from the point of view of world horizontal).

Output: ``used_categories.json`` = {category: [{"task": ..., "round": n, "role": ...}, ...]}.
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
WA = os.path.join(REPO, "runlogs", "world_alignment_selection_20260911")
SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
OUT = os.path.join(HERE, "..", "used_categories.json")

TASK_OF_MANIFEST = {"object_scaling": "object_scaling-0", "mental_rotation": "mental_rotation-0", "disembedding": "disembedding-0",
                    "configuration_matching": "configuration_matching-3", "articulation": "configuration_matching-1/-2"}
IORS_ID = {"ontop": 0, "under": 1, "inside": 2, "nextto": 3, "touching": 4}
OWN = "world_alignment-2"


def main():
    used = defaultdict(list)

    def add(cat, task, rnd, role):
        if not cat or task.startswith(OWN):
            return
        e = {"task": task, "round": int(rnd), "role": role}
        if e not in used[cat]:
            used[cat].append(e)

    # 1. grasp-library selection rounds
    with open(os.path.join(LIB, "selection_manifest.csv"), newline="") as f:
        for r in csv.DictReader(f):
            task = TASK_OF_MANIFEST.get(r["task"], r["task"])
            if r["task"] == "IORS":
                task = f"inter_object_relation_satisfaction-{IORS_ID[r['relation']]}" if r["relation"] in IORS_ID else "inter_object_relation_satisfaction"
            add(r["category"], task, r["round"], "target")
    # 2. the sheet: partner / female / spec / pool rows (round from the note of the target block when present)
    with open(os.path.join(LIB, "select", "task_object_selection.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    task = role = None
    for r in rows:
        if r["task"]:
            task, role = r["task"].split(" ")[0], r["role"]
            continue
        if task is None or task.startswith("world_alignment"):
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
        if task is None or task.startswith("world_alignment") or role not in ("target", "male"):
            continue
        m = re.search(r"round (\d)", r["note"])
        rnd = int(m.group(1)) if m else 1
        for p in re.findall(r"-> ([a-z_]+)(?:/[a-z]+)?", r["pair"]):
            add(p, task, rnd, "partner")
    # 3. world-alignment table surface / world vertical rounds 1-3
    with open(os.path.join(WA, "selection.json")) as f:
        wa = json.load(f)
    for t in wa["tasks"].values():
        for rnd, recs in t["rounds"].items():
            for rec in recs:
                add(rec["category"], t["problem"], rnd, "target")
    # 4. spec pins (round 1)
    with open(SPEC) as f:
        spec = json.load(f)
    for prob, body in spec.items():
        if prob.startswith("_") or prob.startswith(OWN):
            continue
        for role, cfg in (body.get("roles") or {}).items():
            for cat in (cfg.get("models") or {}):
                add(cat, prob, 1, f"spec:{role}")
    out = {c: sorted(v, key=lambda e: (e["round"], e["task"])) for c, v in sorted(used.items())}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    by_round = defaultdict(set)
    for c, es in out.items():
        for e in es:
            by_round[e["round"]].add(c)
    print(f"{len(out)} categories used by other tasks; per round: " + ", ".join(f"r{k}: {len(v)}" for k, v in sorted(by_round.items())))
    return out


if __name__ == "__main__":
    main()
