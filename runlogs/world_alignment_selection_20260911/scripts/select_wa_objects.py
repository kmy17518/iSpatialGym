#!/usr/bin/env python3
"""Turn the reviewer verdicts (``review/*.json``, contract ``review_contract.md``) into the round 1 / 2 / 3 category and
model selection for ``world_alignment-0`` (table surface, ``wa0``) and ``world_alignment-1`` (world vertical, ``wa1``).

Rules (from the request of 2026-09-11):
* round 1: 10 categories per task, none used by another task's round 1 (we take categories not used by any other
  task at all when enough qualify, then the ones only used in other tasks' rounds 2 / 3), >= 2 valid models each;
* round 2: 5 categories per task, may overlap with other tasks' round-1 categories, >= 2 valid models each;
* round 3: 5 categories per task, may overlap with any other task's categories, >= 1 valid model.
* wa0 and wa1 stay disjoint in rounds 1 and 2 (a category reviewed for both goes to the task where it scores higher);
  the spec-pinned categories (keyboard for wa0, painting for wa1) are round 1.
* valid model = reviewer ``valid`` (V >= 3, O/U >= 2, F/H >= 1); categories are ranked by the mean score of their two
  best valid models (single-model categories by that model's score minus 2), ties by the number of valid models.
* Picks per category = the two best valid models (reviewer picks when given), alternate = the next valid model; every
  other valid model is a candidate.

Output: ``selection.json`` and ``selection.md``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")

TASKS = {"wa0": "world_alignment-0", "wa1": "world_alignment-1"}
PINNED_CATEGORY = {"wa0": "keyboard", "wa1": "painting"}
ROUND_SIZES = {1: 10, 2: 5, 3: 5}

#: manual adjustments of the automatic ranking (category -> (task, round or None to exclude, reason)); applied after the
#: automatic assignment, with the same validity rules
OVERRIDES = {
    # a sneaker on a table is the least natural table object of the round-1 set (reviewer F = 1 for every model) and the
    # newspapers are natural, thin and readable: swap them between rounds 1 and 2
    ("wa0", "gym_shoe"): (2, "shoe on a table is only borderline natural (F = 1); moved to round 2"),
    ("wa0", "newspaper"): (1, "natural flat table object with large mastheads; promoted to round 1"),
    # paper-thin (< 1 mm) menu sheets are fragile to push and hinge; the rigid clipboard / the heart wreath take the slot
    ("wa0", "menu"): (None, "paper-thin (< 1 mm) sheets: fragile for pushing / hinging on the table; not selected"),
    ("wa0", "clipboard"): (3, "rigid 0.32 m board with a clip and a printed form; takes the round-3 slot of menu"),
    ("wa1", "menu"): (None, "paper-thin (< 1 mm) sheets are not a wall object; not selected"),
    ("wa1", "valentine_wreath"): (3, "heart-shaped door / wall wreath with an obvious up; takes the round-3 slot of menu"),
}


def load_reviews():
    out = {"wa0": {}, "wa1": {}}
    for p in sorted(glob.glob(os.path.join(ROOT, "review", "*.json"))):
        with open(p) as f:
            r = json.load(f)
        task = r.get("task")
        if task not in out:
            continue
        cat = r["category"]
        if cat in out[task]:
            print(f"WARNING duplicate review for {task}/{cat}: {p}")
        r["file"] = os.path.relpath(p, ROOT)
        out[task][cat] = r
    return out


def model_score_key(task):
    return ("O", "F") if task == "wa0" else ("U", "H")


def category_record(task, cat, review, info, used, pinned_models):
    a, b = model_score_key(task)
    models = []
    for m in review["models"]:
        valid = bool(m.get("valid"))
        # re-derive validity from the rubric so that a reviewer slip cannot let an invalid model through
        rule = m.get("V", 0) >= 3 and m.get(a, 0) >= 2 and m.get(b, 0) >= 1
        if valid != rule:
            valid = rule
        score = 2 * m.get("V", 0) + 2 * m.get(a, 0) + m.get(b, 0)
        models.append({**m, "valid": valid, "score": score})
    valid_models = sorted([m for m in models if m["valid"]], key=lambda m: (-m["score"], m["model"]))
    # reviewer picks first (when valid), then by score
    picks = [p for p in (review.get("picks") or []) if any(m["model"] == p and m["valid"] for m in models)]
    for m in valid_models:
        if len(picks) >= 2:
            break
        if m["model"] not in picks:
            picks.append(m["model"])
    alt = review.get("alternate")
    if alt is not None and not any(m["model"] == alt and m["valid"] for m in models):
        alt = None
    if alt is None:
        rest = [m["model"] for m in valid_models if m["model"] not in picks]
        alt = rest[0] if rest else None
    if len(valid_models) >= 2:
        cat_score = (valid_models[0]["score"] + valid_models[1]["score"]) / 2.0
    elif valid_models:
        cat_score = valid_models[0]["score"] - 2.0
    else:
        cat_score = -1.0
    if review.get("category_verdict") == "weak" and len(valid_models) >= 2:
        cat_score -= 1.0
    n_ready = info.get("n_ready", len(models))
    n_reviewed = len(models)
    used_by = used.get(cat, [])
    return {
        "task": task, "category": cat, "n_ready": n_ready, "n_reviewed": n_reviewed, "n_valid": len(valid_models),
        "n_unreviewed": max(0, n_ready - n_reviewed), "category_score": round(cat_score, 2),
        "verdict": review.get("category_verdict"), "category_note": review.get("category_note", ""),
        "picks": picks[:2], "alternate": alt, "models": models, "valid_models": [m["model"] for m in valid_models],
        "used_by_other_tasks": used_by, "used_in_round1_elsewhere": any(u["round"] == 1 for u in used_by),
        "used_elsewhere": bool(used_by), "pinned_models": sorted(pinned_models.get((task, cat), [])),
        "review_file": review["file"],
    }


def assign_rounds(records, used):
    """records: {task: {cat: rec}} -> {task: {round: [cat, ...]}} following the rules in the module docstring."""
    assigned = {t: {1: [], 2: [], 3: []} for t in TASKS}
    taken = {}  # cat -> (task, round) for rounds 1-2 disjointness

    def rank(recs):
        return sorted(recs, key=lambda r: (-r["category_score"], -r["n_valid"], r["category"]))

    # both tasks compete for categories reviewed twice: give each to the task where it scores higher
    prefer = {}
    for cat in set(records["wa0"]) & set(records["wa1"]):
        prefer[cat] = "wa0" if records["wa0"][cat]["category_score"] >= records["wa1"][cat]["category_score"] else "wa1"

    def eligible(task, rec, rnd):
        cat = rec["category"]
        if cat in taken:
            return False
        if cat in prefer and prefer[cat] != task and rnd <= 2:
            return False
        if rnd == 1:
            return rec["n_valid"] >= 2 and not rec["used_elsewhere"]
        if rnd == "1b":
            return rec["n_valid"] >= 2 and not rec["used_in_round1_elsewhere"]
        if rnd == 2:
            return rec["n_valid"] >= 2
        return rec["n_valid"] >= 1

    for task in TASKS:
        recs = rank(records[task].values())
        # round 1: the pinned category first, then unused categories, then categories only used in other rounds
        pin = PINNED_CATEGORY[task]
        if pin in records[task] and records[task][pin]["n_valid"] >= 2:
            assigned[task][1].append(pin)
            taken[pin] = (task, 1)
        for stage in (1, "1b"):
            for r in recs:
                if len(assigned[task][1]) >= ROUND_SIZES[1]:
                    break
                if eligible(task, r, stage):
                    assigned[task][1].append(r["category"])
                    taken[r["category"]] = (task, 1)
    for task in TASKS:
        recs = rank(records[task].values())
        for r in recs:
            if len(assigned[task][2]) >= ROUND_SIZES[2]:
                break
            if eligible(task, r, 2):
                assigned[task][2].append(r["category"])
                taken[r["category"]] = (task, 2)
    for task in TASKS:
        recs = rank(records[task].values())
        for r in recs:
            if len(assigned[task][3]) >= ROUND_SIZES[3]:
                break
            cat = r["category"]
            if cat in taken and taken[cat][0] == task:
                continue  # already in this task's rounds 1/2
            if cat in taken and taken[cat][1] <= 2 and r["n_valid"] == 0:
                continue
            if r["n_valid"] >= 1:
                assigned[task][3].append(cat)
    # manual overrides
    for (task, cat), (rnd, reason) in OVERRIDES.items():
        if cat not in records[task]:
            continue
        for k in (1, 2, 3):
            if cat in assigned[task][k]:
                assigned[task][k].remove(cat)
        if rnd is not None:
            rec = records[task][cat]
            if rec["n_valid"] < (2 if rnd <= 2 else 1):
                raise SystemExit(f"override {task}/{cat} -> round {rnd} violates the validity rule ({rec['n_valid']} valid)")
            assigned[task][rnd].append(cat)
        records[task][cat]["override"] = reason
    for task in TASKS:
        for k in (1, 2, 3):
            assigned[task][k].sort(key=lambda c: (-records[task][c]["category_score"], -records[task][c]["n_valid"], c))
            if len(assigned[task][k]) > ROUND_SIZES[k]:
                print(f"WARNING {task} round {k} has {len(assigned[task][k])} categories (> {ROUND_SIZES[k]})")
    return assigned


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=ROOT)
    args = ap.parse_args()
    reviews = load_reviews()
    with open(os.path.join(ROOT, "candidate_categories.json")) as f:
        info = json.load(f)
    with open(os.path.join(ROOT, "used_categories.json")) as f:
        used = json.load(f)
    with open(SPEC) as f:
        spec = json.load(f)
    pinned_models = defaultdict(set)
    for task, prob in TASKS.items():
        for cat, ms in (spec.get(prob, {}).get("roles", {}).get("target", {}).get("models") or {}).items():
            pinned_models[(task, cat)] |= set(ms)
    records = {t: {} for t in TASKS}
    for task in TASKS:
        for cat, rev in reviews[task].items():
            records[task][cat] = category_record(task, cat, rev, info.get(cat, {}), used, pinned_models)
    rounds = assign_rounds(records, used)
    out = {"tasks": {}, "rules": __doc__.strip().splitlines()[3:14]}
    md = ["# World-alignment object selection (rounds 1-3)", "",
          "Scores: wa0 `2V + 2O + F`, wa1 `2V + 2U + H` (max 18; contract `review_contract.md`). valid = V >= 3, O/U >= 2, F/H >= 1. "
          "`cat score` = mean of the two best valid models. Frames: wa0 `long_axis` / `front_axis` (side facing the robot when aligned), "
          "wa1 `front_axis` (out of the wall) / `up_axis`.", ""]
    for task, prob in TASKS.items():
        t = {"problem": prob, "rounds": {}, "not_selected": []}
        selected = set()
        md.append(f"## {prob} ({task})\n")
        for rnd in (1, 2, 3):
            cats = rounds[task][rnd]
            t["rounds"][str(rnd)] = []
            md.append(f"### round {rnd} ({len(cats)} categories)\n")
            md.append("| category | cat score | valid / reviewed / ready | picks (score) | alternate | frames of the picks | used by other tasks | verdict | note |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for cat in cats:
                r = records[task][cat]
                selected.add(cat)
                by = {m["model"]: m for m in r["models"]}
                a, b = model_score_key(task)
                picks_txt = ", ".join(f"{p} ({by[p]['score']})" for p in r["picks"])
                if task == "wa0":
                    frames = "; ".join(f"{p}: long {by[p].get('long_axis')}, front {by[p].get('front_axis')}" + ("" if by[p].get("rest_ok", True) else ", rest NOT canonical") for p in r["picks"])
                else:
                    frames = "; ".join(f"{p}: front {by[p].get('front_axis')}, up {by[p].get('up_axis')}, thin {by[p].get('thin_axis')}" for p in r["picks"])
                usedtxt = ", ".join(sorted({f"{u['task']} r{u['round']}" for u in r["used_by_other_tasks"]})) or "-"
                note = r['category_note'] + (f" (override: {r['override']})" if r.get('override') else "")
                md.append(f"| {cat} | {r['category_score']} | {r['n_valid']} / {r['n_reviewed']} / {r['n_ready']} | {picks_txt} | {r['alternate'] or '-'} | {frames} | {usedtxt} | {r['verdict']} | {note} |")
                t["rounds"][str(rnd)].append({**r, "round": rnd})
            md.append("")
        rest = sorted((r for c, r in records[task].items() if c not in selected), key=lambda r: (-r["category_score"], r["category"]))
        md.append("### not selected\n")
        md.append("| category | cat score | valid / reviewed / ready | verdict | used by other tasks | note |")
        md.append("|---|---|---|---|---|---|")
        for r in rest:
            usedtxt = ", ".join(sorted({f"{u['task']} r{u['round']}" for u in r["used_by_other_tasks"]})) or "-"
            note = r['category_note'] + (f" (override: {r['override']})" if r.get('override') else "")
            md.append(f"| {r['category']} | {r['category_score']} | {r['n_valid']} / {r['n_reviewed']} / {r['n_ready']} | {r['verdict']} | {usedtxt} | {note} |")
            t["not_selected"].append(r)
        md.append("")
        out["tasks"][task] = t
    # spec fragments: the picks of every selected category as roles.target.models plus the per-model canonical frames
    # in the dict form that sampler >= 1.4.1 accepts for generation.front_axis / long_axis / up_axis
    frags = {}
    for task, prob in TASKS.items():
        models = {}
        axes = {"front_axis": {"default": "+x"}, ("long_axis" if task == "wa0" else "up_axis"): {"default": "y" if task == "wa0" else "+z"}}
        rounds_of = {}
        for rnd in ("1", "2", "3"):
            for rec in out["tasks"][task]["rounds"][rnd]:
                cat = rec["category"]
                models[cat] = list(rec["picks"])
                rounds_of[cat] = int(rnd)
                for m in rec["models"]:
                    if not m["valid"]:
                        continue
                    key = f"{cat}/{m['model']}"
                    if m.get("front_axis"):
                        axes["front_axis"][key] = m["front_axis"]
                    if task == "wa0" and m.get("long_axis"):
                        axes["long_axis"][key] = m["long_axis"]
                    if task == "wa1" and m.get("up_axis"):
                        axes["up_axis"][key] = m["up_axis"]
        frags[prob] = {"_comment": ("roles.target.models = pick 1 / pick 2 of every selected category (all rounds; rounds in _rounds); generation "
                                    "axes are per model ('<category>/<model>') as read from the renders, in the dict form of sampler 1.4.1. Merge into "
                                    "task_custom_lists.json[problem] as needed (the test spec keeps keyboard / painting only)."),
                       "_rounds": rounds_of, "roles": {"target": {"models": models}}, "generation": axes}
    with open(os.path.join(args.out, "spec_fragments.json"), "w") as f:
        json.dump(frags, f, indent=1)
    with open(os.path.join(args.out, "selection.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(args.out, "selection.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    for task in TASKS:
        print(task, {k: v for k, v in rounds[task].items()})


if __name__ == "__main__":
    main()
