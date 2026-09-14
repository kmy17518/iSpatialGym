#!/usr/bin/env python3
"""Turn the reviewer verdicts (``review/*.json``, contract ``review_contract.md``) into the obstacle category / model
selection for ``camera_relocalization-0`` (task id ``obst``).

Rules:
* valid model = re-derived from the rubric (V >= 3, N >= 2, S >= 1, rest_ok); score = 2V + 2N + S (max 18);
* categories are ranked by the mean score of their two best valid models (single-model categories by that model's
  score minus 2, ``weak`` verdicts with >= 2 valid models minus 1), ties by the number of valid models, then by the
  number of the seven scenes the category is natural in;
* round 1 = the 10 best categories with >= 2 valid models, round 2 = the next 5 with >= 2 valid models, round 3 = the
  next 5 with >= 1 valid model. The obstacle is never manipulated, so - unlike the target selections of the other
  tasks - no overlap rule with other tasks' categories is applied; overlaps are only recorded (``used_categories.json``).
  Every other reviewed category is listed under ``not_selected`` with its ranking, so the pool can be extended later;
* picks per category = the two best valid models (reviewer picks when given), alternate = the next valid model; every
  other valid model is a candidate.

Output: ``selection.json``, ``selection.md`` and ``spec_fragments.json`` (``roles.obstacle.models`` = the picks of every
selected category plus ``_pool`` = every valid model per selected category, ``_natural_scenes`` per category).
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

TASK = "obst"
PROBLEM = "camera_relocalization-0"
ROUND_SIZES = {1: 10, 2: 5, 3: 5}
SCENES = ["house_single_floor", "house_double_floor_upper", "house_double_floor_lower", "Rs_int", "restaurant_diner",
          "hotel_suite_large", "office_cubicles_right"]

#: manual adjustments of the automatic ranking (category -> (round or None to exclude, reason)); applied after the
#: automatic assignment, with the same validity rules
OVERRIDES: dict = {
    # obstacle-type diversity: the automatic ranking put three stool categories (taboret, music_stool, highchair) among the
    # 20; taboret (6 valid models) already covers stools, so music_stool (2 models) swaps with vacuum (3 valid models,
    # natural in all seven scenes, a distinct obstacle type) and highchair (both models are counter stools, not high
    # chairs, diner / kitchen only) gives way to backpack (score 16, ideal size, natural in all seven scenes; not ranked
    # higher only because the category has a single ready model)
    "vacuum": (1, "swapped with music_stool for obstacle-type diversity (taboret already covers stools)"),
    "music_stool": (2, "swapped with vacuum: second stool category next to taboret"),
    "highchair": (None, "third stool category (both models are counter stools, not high chairs); replaced by backpack"),
    "backpack": (3, "replaces highchair: classic mid-path obstacle, score 16, natural in all seven scenes (single ready model)"),
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
        # some reviewers wrote the ids as printed in the strip headers ("<category>/<model>"): strip the prefix
        def mid(x):
            return x.split("/", 1)[1] if isinstance(x, str) and x.startswith(cat + "/") else x

        for m in r["models"]:
            m["model"] = mid(m["model"])
        r["picks"] = [mid(x) for x in (r.get("picks") or [])]
        r["alternate"] = mid(r.get("alternate"))
        r["file"] = os.path.relpath(p, ROOT)
        out[cat] = r
    return out


def category_record(cat, review, info, used):
    models = []
    for m in review["models"]:
        rest_ok = bool(m.get("rest_ok", False))
        rule = m.get("V", 0) >= 3 and m.get("N", 0) >= 2 and m.get("S", 0) >= 1 and rest_ok
        score = 2 * m.get("V", 0) + 2 * m.get("N", 0) + m.get("S", 0)
        if m.get("score") != score or bool(m.get("valid")) != bool(rule):
            print(f"note {cat}/{m['model']}: reviewer score/valid {m.get('score')}/{m.get('valid')} re-derived as {score}/{rule}")
        models.append({**m, "rest_ok": rest_ok, "valid": bool(rule), "score": score})
    bboxes = info.get("bbox", {})
    for m in models:
        m["bbox"] = bboxes.get(m["model"])
    valid_models = sorted([m for m in models if m["valid"]], key=lambda m: (-m["score"], m["model"]))
    picks = [p for p in (review.get("picks") or []) if any(m["model"] == p and m["valid"] for m in models)]
    for m in valid_models:
        if len(picks) >= 2:
            break
        if m["model"] not in picks:
            picks.append(m["model"])
    alt = review.get("alternate")
    if alt is not None and (alt in picks or not any(m["model"] == alt and m["valid"] for m in models)):
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
    natural = [s for s in (review.get("natural_scenes") or info.get("natural_scenes") or []) if s in SCENES]
    if not natural:
        natural = [s for s in info.get("natural_scenes", []) if s in SCENES]
    n_ready = info.get("n_ready", len(models))
    used_by = used.get(cat, [])
    return {
        "task": TASK, "category": cat, "group": info.get("group"), "n_ready": n_ready, "n_reviewed": len(models),
        "n_valid": len(valid_models), "n_unreviewed": max(0, n_ready - len(models)), "category_score": round(cat_score, 2),
        "verdict": review.get("category_verdict"), "category_note": review.get("category_note", ""),
        "natural_scenes": natural, "in_scene": info.get("in_scene", {}),
        "picks": picks[:2], "alternate": alt, "models": models, "valid_models": [m["model"] for m in valid_models],
        "used_by_other_tasks": used_by, "used_in_round1_elsewhere": any(u["round"] == 1 for u in used_by),
        "used_elsewhere": bool(used_by), "review_file": review["file"],
    }


def assign_rounds(records):
    assigned = {1: [], 2: [], 3: []}
    taken = set()

    def key(r):
        return (-r["category_score"], -r["n_valid"], -len(r["natural_scenes"]), r["category"])

    recs = sorted(records.values(), key=key)
    for rnd in (1, 2, 3):
        need = 2 if rnd <= 2 else 1
        for r in recs:
            if len(assigned[rnd]) >= ROUND_SIZES[rnd]:
                break
            if r["category"] in taken or r["n_valid"] < need:
                continue
            assigned[rnd].append(r["category"])
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
        assigned[k].sort(key=lambda c: key(records[c]))
        if len(assigned[k]) > ROUND_SIZES[k]:
            print(f"WARNING round {k} has {len(assigned[k])} categories (> {ROUND_SIZES[k]})")
    return assigned


def bbox_txt(b):
    return f"{b[0]:.2f} x {b[1]:.2f} x {b[2]:.2f} m" if b else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=ROOT)
    args = ap.parse_args()
    reviews = load_reviews()
    with open(os.path.join(ROOT, "candidate_categories.json")) as f:
        info = json.load(f)
    with open(os.path.join(ROOT, "used_categories.json")) as f:
        used = json.load(f)
    missing = sorted(set(info) - set(reviews))
    if missing:
        print(f"WARNING no review for: {missing}")
    for cat, rev in reviews.items():
        rendered = set(info.get(cat, {}).get("rendered", []))
        seen = {m["model"] for m in rev["models"]}
        if rendered and rendered != seen:
            print(f"WARNING {cat}: reviewed models differ from rendered ones: missing {sorted(rendered - seen)}, extra {sorted(seen - rendered)}")
    records = {cat: category_record(cat, rev, info.get(cat, {}), used) for cat, rev in reviews.items()}
    rounds = assign_rounds(records)
    out = {"problem": PROBLEM, "task": TASK, "rules": __doc__.strip().splitlines()[3:16], "rounds": {}, "not_selected": []}
    md = ["# Obstacle selection for camera_relocalization-0 (rounds 1-3)", "",
          "Scores: `2V + 2N + S` (max 18; contract `review_contract.md`): V visual, N naturalness as a mid-path floor obstacle, S size fit; "
          "valid = V >= 3, N >= 2, S >= 1 and rest_ok (stands stably on the floor in its canonical pose). `cat score` = mean of the two best "
          "valid models (single valid model: score - 2; weak verdict: - 1). Rounds = ranking tiers (10 / 5 / 5; no overlap rule: the obstacle "
          "is never manipulated). `natural in` = the scenes (of the seven) the reviewer confirmed the category is natural in; `*` marks scenes "
          "whose BEHAVIOR scene files already contain the category.", "", f"## {PROBLEM} ({TASK})", ""]
    selected = set()

    def scenes_txt(r):
        return ", ".join(s + ("*" if s in r["in_scene"] else "") for s in r["natural_scenes"]) or "-"

    for rnd in (1, 2, 3):
        cats = rounds[rnd]
        out["rounds"][str(rnd)] = []
        md.append(f"### round {rnd} ({len(cats)} categories)\n")
        md.append("| category | group | cat score | valid / reviewed / ready | picks (score, bbox) | alternate | natural in | used by other tasks | verdict | note |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for cat in cats:
            r = records[cat]
            selected.add(cat)
            by = {m["model"]: m for m in r["models"]}
            picks_txt = ", ".join(f"{p} ({by[p]['score']}, {bbox_txt(by[p]['bbox'])})" for p in r["picks"])
            usedtxt = ", ".join(sorted({f"{u['task']} r{u['round']}" for u in r["used_by_other_tasks"]})) or "-"
            note = r["category_note"] + (f" (override: {r['override']})" if r.get("override") else "")
            md.append(f"| {cat} | {r['group']} | {r['category_score']} | {r['n_valid']} / {r['n_reviewed']} / {r['n_ready']} | {picks_txt} | "
                      f"{r['alternate'] or '-'} | {scenes_txt(r)} | {usedtxt} | {r['verdict']} | {note} |")
            out["rounds"][str(rnd)].append({**r, "round": rnd})
        md.append("")
    rest = sorted((r for c, r in records.items() if c not in selected), key=lambda r: (-r["category_score"], -r["n_valid"], r["category"]))
    md.append("### not selected (ranked; valid models of these categories can extend the pool)\n")
    md.append("| category | group | cat score | valid / reviewed / ready | best valid models | natural in | verdict | note |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rest:
        by = {m["model"]: m for m in r["models"]}
        best = ", ".join(f"{m} ({by[m]['score']})" for m in r["valid_models"][:3]) or "-"
        note = r["category_note"] + (f" (override: {r['override']})" if r.get("override") else "")
        md.append(f"| {r['category']} | {r['group']} | {r['category_score']} | {r['n_valid']} / {r['n_reviewed']} / {r['n_ready']} | {best} | "
                  f"{scenes_txt(r)} | {r['verdict']} | {note} |")
        out["not_selected"].append(r)
    md.append("")
    # spec fragment: picks + full valid pool of every selected category
    models, pool, rounds_of, natural = {}, {}, {}, {}
    for rnd in ("1", "2", "3"):
        for rec in out["rounds"][rnd]:
            cat = rec["category"]
            models[cat] = list(rec["picks"])
            pool[cat] = list(rec["valid_models"])
            rounds_of[cat] = int(rnd)
            natural[cat] = list(rec["natural_scenes"])
    frags = {PROBLEM: {"_comment": ("Obstacle role for camera_relocalization-0 (sampler support pending: the sampler currently binds the floor only). "
                                    "roles.obstacle.models = pick 1 / pick 2 of every selected category (rounds in _rounds); _pool = every valid "
                                    "model per selected category; _natural_scenes = the scenes (of the seven candidates) in which the category is "
                                    "natural, i.e. where an instance may draw it. Merge into task_custom_lists.json[problem] once the sampler "
                                    "places obstacles."),
                       "_rounds": rounds_of, "_natural_scenes": natural, "_pool": pool, "roles": {"obstacle": {"models": models}}}}
    with open(os.path.join(args.out, "spec_fragments.json"), "w") as f:
        json.dump(frags, f, indent=1)
    with open(os.path.join(args.out, "selection.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(args.out, "selection.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print({k: v for k, v in rounds.items()})
    n_valid_all = sum(r["n_valid"] for r in records.values())
    print(f"{len(records)} categories reviewed, {sum(1 for r in records.values() if r['n_valid'])} with >= 1 valid model, {n_valid_all} valid models; "
          f"selected {sum(len(v) for v in rounds.values())} categories with {sum(len(pool[c]) for c in pool)} valid models")


if __name__ == "__main__":
    main()
