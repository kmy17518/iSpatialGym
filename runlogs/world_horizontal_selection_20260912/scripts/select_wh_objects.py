#!/usr/bin/env python3
"""Turn the reviewer verdicts (``review/*.json``, contract ``review_contract.md``) into the round 1 / 2 / 3 category and
model selection for ``world_alignment-2`` (world horizontal, ``wa2``).

Rules (the round rules of the 2026-09-11 request, applied to the new problem):
* round 1: 10 categories, none used by another task's round 1 (categories used by no other task at all are taken
  first, then the ones only used in other tasks' rounds 2 / 3), >= 2 valid models each;
* round 2: 5 categories, may overlap with other tasks' round-1 categories (including the world-vertical round 1:
  towel_rack, hanger, signpost, ... - wall-hangable categories are scarce), >= 2 valid models each;
* round 3: 5 categories, may overlap with any other task's categories, >= 1 valid model.
* valid model = reviewer ``valid`` re-derived from the rubric (V >= 3, L >= 2, H >= 1); categories are ranked by the
  mean score of their two best valid models (single-model categories by that model's score minus 2, ``weak`` verdicts
  with >= 2 valid models minus 1), ties by the number of valid models.
* Picks per category = the two best valid models (reviewer picks when given), alternate = the next valid model; every
  other valid model is a candidate.

Output: ``selection.json``, ``selection.md`` and ``spec_fragments.json`` (``roles.target.models`` = the picks of every
selected category, ``generation.front_axis`` / ``up_axis`` / ``level_axis`` / ``symmetric_180`` per model in the dict
form the sampler accepts).
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

TASK = "wa2"
PROBLEM = "world_alignment-2"
ROUND_SIZES = {1: 10, 2: 5, 3: 5}
AXES = ("x", "y", "z")

#: manual adjustments of the automatic ranking (category -> (round or None to exclude, reason)); applied after the
#: automatic assignment, with the same validity rules
OVERRIDES = {
    # the reviewer gave H = 1 to the two 0.20 m-deep track lights (bkaivt, jhpygl) although the contract sets H = 0 above
    # 0.15 m thickness (flagged in the review notes); strictly only the vertical tube sconce etqbld is valid (< 2 models),
    # and 0.2 m-deep fixtures cannot hang flat on the hinge
    "wall_mounted_light": (None, "track lights are 0.20 m deep (contract: H = 0 above 0.15 m); only the tube sconce remains valid, < 2 models; not selected"),
}


def load_reviews():
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "review", "*.json"))):
        with open(p) as f:
            r = json.load(f)
        if r.get("task") != TASK:
            continue
        cat = r["category"]
        if cat in out:
            print(f"WARNING duplicate review for {cat}: {p}")
        r["file"] = os.path.relpath(p, ROOT)
        out[cat] = r
    return out


def frame_ok(m) -> bool:
    """front / level / up name three different axes."""
    f = str(m.get("front_axis", "")).lstrip("+-")
    lv = str(m.get("level_axis", "")).lstrip("+-")
    u = str(m.get("up_axis", "")).lstrip("+-")
    return f in AXES and lv in AXES and u in AXES and len({f, lv, u}) == 3


def category_record(cat, review, info, used, pinned_models):
    models = []
    for m in review["models"]:
        rule = m.get("V", 0) >= 3 and m.get("L", 0) >= 2 and m.get("H", 0) >= 1
        score = 2 * m.get("V", 0) + 2 * m.get("L", 0) + m.get("H", 0)
        rec = {**m, "valid": bool(rule), "score": score, "frame_ok": frame_ok(m)}
        if rule and not rec["frame_ok"]:
            print(f"WARNING {cat}/{m['model']}: inconsistent frame {m.get('front_axis')} / {m.get('level_axis')} / {m.get('up_axis')}")
        models.append(rec)
    valid_models = sorted([m for m in models if m["valid"]], key=lambda m: (-m["score"], m["model"]))
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
    used_by = used.get(cat, [])
    return {
        "task": TASK, "category": cat, "n_ready": n_ready, "n_reviewed": len(models), "n_valid": len(valid_models),
        "n_unreviewed": max(0, n_ready - len(models)), "category_score": round(cat_score, 2),
        "verdict": review.get("category_verdict"), "category_note": review.get("category_note", ""), "group": info.get("group"),
        "picks": picks[:2], "alternate": alt, "models": models, "valid_models": [m["model"] for m in valid_models],
        "used_by_other_tasks": used_by, "used_in_round1_elsewhere": any(u["round"] == 1 for u in used_by),
        "used_elsewhere": bool(used_by), "pinned_models": sorted(pinned_models.get(cat, [])), "review_file": review["file"],
    }


def assign_rounds(records):
    assigned = {1: [], 2: [], 3: []}
    taken = set()

    def rank(recs):
        return sorted(recs, key=lambda r: (-r["category_score"], -r["n_valid"], r["category"]))

    def eligible(rec, rnd):
        if rec["category"] in taken:
            return False
        if rnd == 1:
            return rec["n_valid"] >= 2 and not rec["used_elsewhere"]
        if rnd == "1b":
            return rec["n_valid"] >= 2 and not rec["used_in_round1_elsewhere"]
        if rnd == 2:
            return rec["n_valid"] >= 2
        return rec["n_valid"] >= 1

    recs = rank(records.values())
    for stage in (1, "1b"):
        for r in recs:
            if len(assigned[1]) >= ROUND_SIZES[1]:
                break
            if eligible(r, stage):
                assigned[1].append(r["category"])
                taken.add(r["category"])
    for r in recs:
        if len(assigned[2]) >= ROUND_SIZES[2]:
            break
        if eligible(r, 2):
            assigned[2].append(r["category"])
            taken.add(r["category"])
    for r in recs:
        if len(assigned[3]) >= ROUND_SIZES[3]:
            break
        if eligible(r, 3):
            assigned[3].append(r["category"])
            taken.add(r["category"])
    for cat, (rnd, reason) in OVERRIDES.items():
        if cat not in records:
            continue
        for k in (1, 2, 3):
            if cat in assigned[k]:
                assigned[k].remove(cat)
        if rnd is not None:
            if records[cat]["n_valid"] < (2 if rnd <= 2 else 1):
                raise SystemExit(f"override {cat} -> round {rnd} violates the validity rule ({records[cat]['n_valid']} valid)")
            assigned[rnd].append(cat)
        records[cat]["override"] = reason
    for k in (1, 2, 3):
        assigned[k].sort(key=lambda c: (-records[c]["category_score"], -records[c]["n_valid"], c))
        if len(assigned[k]) > ROUND_SIZES[k]:
            print(f"WARNING round {k} has {len(assigned[k])} categories (> {ROUND_SIZES[k]})")
    return assigned


def frame_txt(m):
    s = f"front {m.get('front_axis')}, level {m.get('level_axis')}, up {m.get('up_axis')}, thin {m.get('thin_axis')}"
    if m.get("symmetric_180"):
        s += ", symmetric 180"
    return s


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
    pinned = defaultdict(set)
    for cat, ms in (spec.get(PROBLEM, {}).get("roles", {}).get("target", {}).get("models") or {}).items():
        pinned[cat] |= set(ms)
    missing = sorted(set(info) - set(reviews))
    if missing:
        print(f"WARNING no review for: {missing}")
    records = {cat: category_record(cat, rev, info.get(cat, {}), used, pinned) for cat, rev in reviews.items()}
    rounds = assign_rounds(records)
    out = {"problem": PROBLEM, "task": TASK, "rules": __doc__.strip().splitlines()[3:15], "rounds": {}, "not_selected": []}
    md = ["# World-horizontal object selection (rounds 1-3)", "",
          "Scores: `2V + 2L + H` (max 18; contract `review_contract.md`): V visual, L level readability, H hangability. valid = V >= 3, L >= 2, "
          "H >= 1. `cat score` = mean of the two best valid models (single valid model: score - 2; weak verdict: - 1). Frames: `front` = axis "
          "out of the wall, `level` = axis along the wall when aligned, `up` = axis pointing up when hung, `thin` = thinnest bbox axis; "
          "`symmetric 180` = looks the same upside down (the sampler adds the half turn about the front axis to the goal's orientation "
          "equivalences).", "", f"## {PROBLEM} ({TASK})", ""]
    selected = set()
    for rnd in (1, 2, 3):
        cats = rounds[rnd]
        out["rounds"][str(rnd)] = []
        md.append(f"### round {rnd} ({len(cats)} categories)\n")
        md.append("| category | cat score | valid / reviewed / ready | picks (score) | alternate | frames of the picks | used by other tasks | verdict | note |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for cat in cats:
            r = records[cat]
            selected.add(cat)
            by = {m["model"]: m for m in r["models"]}
            picks_txt = ", ".join(f"{p} ({by[p]['score']})" for p in r["picks"])
            frames = "; ".join(f"{p}: {frame_txt(by[p])}" for p in r["picks"])
            usedtxt = ", ".join(sorted({f"{u['task']} r{u['round']}" for u in r["used_by_other_tasks"]})) or "-"
            note = r["category_note"] + (f" (override: {r['override']})" if r.get("override") else "")
            md.append(f"| {cat} | {r['category_score']} | {r['n_valid']} / {r['n_reviewed']} / {r['n_ready']} | {picks_txt} | {r['alternate'] or '-'} | {frames} | {usedtxt} | {r['verdict']} | {note} |")
            out["rounds"][str(rnd)].append({**r, "round": rnd})
        md.append("")
    rest = sorted((r for c, r in records.items() if c not in selected), key=lambda r: (-r["category_score"], r["category"]))
    md.append("### not selected\n")
    md.append("| category | cat score | valid / reviewed / ready | verdict | used by other tasks | note |")
    md.append("|---|---|---|---|---|---|")
    for r in rest:
        usedtxt = ", ".join(sorted({f"{u['task']} r{u['round']}" for u in r["used_by_other_tasks"]})) or "-"
        note = r["category_note"] + (f" (override: {r['override']})" if r.get("override") else "")
        md.append(f"| {r['category']} | {r['category_score']} | {r['n_valid']} / {r['n_reviewed']} / {r['n_ready']} | {r['verdict']} | {usedtxt} | {note} |")
        out["not_selected"].append(r)
    md.append("")
    # spec fragment: picks of every selected category + per-model frames / symmetry in the sampler's dict form
    models, rounds_of = {}, {}
    axes = {"front_axis": {"default": "+x"}, "up_axis": {"default": "+z"}, "level_axis": {}, "symmetric_180": {"default": False}}
    for rnd in ("1", "2", "3"):
        for rec in out["rounds"][rnd]:
            cat = rec["category"]
            models[cat] = list(rec["picks"])
            rounds_of[cat] = int(rnd)
            for m in rec["models"]:
                if not m["valid"]:
                    continue
                key = f"{cat}/{m['model']}"
                for k in ("front_axis", "up_axis", "level_axis"):
                    if m.get(k):
                        axes[k][key] = str(m[k]) if k != "level_axis" else str(m[k]).lstrip("+-")
                if m.get("symmetric_180"):
                    axes["symmetric_180"][key] = True
    frags = {PROBLEM: {"_comment": ("roles.target.models = pick 1 / pick 2 of every selected category (all rounds; rounds in _rounds); generation "
                                    "axes / symmetric_180 are per valid model ('<category>/<model>') as read from the renders, in the dict form "
                                    "of the sampler (level_axis is a consistency check against front_axis / up_axis). Merge into "
                                    "task_custom_lists.json[problem] as needed (the test spec pins one category)."),
                       "_rounds": rounds_of, "roles": {"target": {"models": models}}, "generation": axes}}
    with open(os.path.join(args.out, "spec_fragments.json"), "w") as f:
        json.dump(frags, f, indent=1)
    with open(os.path.join(args.out, "selection.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(args.out, "selection.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print({k: v for k, v in rounds.items()})


if __name__ == "__main__":
    main()
