#!/usr/bin/env python3
"""Prepare the reviewer round of the symmetry survey (``runlogs/symmetry_20260915``).

Review set = every model with a ``symmetry_near`` entry (canonical model, and the scaled variant as ``<key>__scaled``
when its ``symmetry_near_scaled`` is non-empty) plus every alignment target whose ``symmetric_180`` reviewer flag
disagrees with the survey's half turn about the alignment axis. Per model the reviewer judges the candidate tags from
the contact sheet ``renders/<key>.jpg``; a sheet that lacks a row for a candidate (or does not exist because the model
had no geometric candidate) is (re-)rendered here with the review tags forced onto it (``survey_symmetry.texture_one``
with ``extra_tags``), then ``report`` is re-run so ``symmetry.json`` points at the new sheets.

Outputs: ``assign/rev_XX.json`` (batches of <= ``--batch-size`` models: key, sheet, candidate tags with the survey's
scores and classes, ``symmetric_180`` questions, uses, scale), ``assign/index.json``, ``assign/review_tags.json`` (what
was re-rendered). The contract the reviewers follow is ``review_contract.md``; they write ``review/<batch_id>.json``;
``aggregate_review.py`` merges the verdicts.

Env: ``base`` conda env (the re-render needs pyrender / EGL, ~0.3 GB of GPU)::

    python scripts/symmetry/prepare_review.py [--out runlogs/symmetry_20260915] [--batch-size 30] [--no-render]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import symcore as S  # noqa: E402
import survey_symmetry as SV  # noqa: E402

VERDICTS = ("preserved", "broken", "unsure")


def rel(path: str) -> str:
    return os.path.relpath(path, REPO)


def tag_context(scores: Dict[str, dict], tag: str, exact_raw: List[str], near_raw: List[str]) -> dict:
    """What the survey measured for one candidate tag (for the assignment entry)."""
    s = scores.get(tag)
    if s is None:
        return {"survey_geometry": "not a candidate (mean > 3 % of the bbox diagonal)", "survey_texture": "not rendered"}
    g = s["geom"]
    gcls = "exact" if tag in exact_raw else ("near" if tag in near_raw else "none")
    out = {"survey_geometry": f"{gcls} (mean {g['mean'] * 100:.2f} %, p95 {g['p95'] * 100:.2f} %, frac2 {g['frac2'] * 100:.1f} % of the bbox diagonal; about the {s['centre']} centre)",
           "worst_rotation": s.get("worst", "")}
    if "tex" in s:
        t = s["tex"]
        out["survey_texture"] = (f"{s.get('tex_class', '?')} (MAD {t['mad'] * 255:.1f}, p95 {t['p95'] * 255:.1f}, p99 {t.get('p99', 0) * 255:.0f} /255 on the "
                                 f"fully covered pixels; union-mask MAD {t.get('mad_union', 0) * 255:.1f}, silhouette IoU {t.get('iou', 0):.3f})")
        out["sheet_row"] = f"{tag}  ({t.get('worst', '')})"
    else:
        out["survey_texture"] = "not rendered"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=SV.DEFAULT_OUT)
    ap.add_argument("--icl", default=SV.DEFAULT_ICL)
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--no-render", action="store_true", help="only list the sheets that lack rows, do not render them")
    args = ap.parse_args()

    sym = json.load(open(os.path.join(args.out, "symmetry.json")))
    icl = list(csv.DictReader(open(args.icl)))
    models = {m["key"]: m for m in json.load(open(os.path.join(args.out, "models.json")))}

    # ---- review set ------------------------------------------------------------------------------------------------
    entries: Dict[str, dict] = collections.OrderedDict()

    def entry(key: str, variant: str) -> dict:
        base_key = key[: -len("__scaled")] if variant == "scaled" else key
        rec = sym[base_key]
        if key not in entries:
            src = rec if variant == "base" else rec["scaled"]
            entries[key] = {"key": key, "category": rec["category"], "model": rec["model"], "variant": variant,
                            "scale": rec["scale"] if variant == "scaled" else None,
                            "sheet": rel(os.path.join(args.out, "renders", key + ".jpg")), "bbox_m": src.get("bbox"), "n_faces": rec.get("n_faces"),
                            "uses": rec["uses"],
                            "survey": {"symmetry": rec["symmetry"] if variant == "base" else rec["symmetry_scaled"],
                                       "symmetry_near": rec["symmetry_near"] if variant == "base" else rec["symmetry_near_scaled"]},
                            "candidates": collections.OrderedDict(), "symmetric_180": []}
        return entries[key]

    def add_tag(key: str, variant: str, tag: str, reason: str):
        e = entry(key, variant)
        base_key = key[: -len("__scaled")] if variant == "scaled" else key
        rec = sym[base_key]
        src = rec if variant == "base" else rec["scaled"]
        if tag not in e["candidates"]:
            exact_raw = src.get("geom_exact_raw", S.expand_tags(src.get("geom_exact", [])))
            near_raw = src.get("geom_near_raw", S.expand_tags(src.get("geom_near", [])))
            e["candidates"][tag] = {"asked_because": reason, **tag_context(src["scores"], tag, exact_raw, near_raw)}
        else:
            e["candidates"][tag]["asked_because"] += "; " + reason

    for key, rec in sym.items():
        for t in rec["symmetry_near"]:
            add_tag(key, "base", t, "symmetry_near")
        if rec.get("symmetry_near_scaled"):
            for t in rec["symmetry_near_scaled"]:
                add_tag(key + "__scaled", "scaled", t, "symmetry_near of the spawn-scaled variant")
    n_near = len(entries)
    n_dis_rows = 0
    for h in SV.alignment_half_turns(icl, sym):
        if not h["disagree"]:
            continue
        n_dis_rows += 1
        survey = "in symmetry (exact geometry, preserved texture)" if h["has"] else ("in symmetry_near" if h["near"] else "not symmetric")
        add_tag(h["key"], "base", h["tag"], f"symmetric_180 disagreement ({h['subtask']})")
        e = entries[h["key"]]
        axis_role = h["axis_note"].split("(")[1].rstrip(")")  # "up" | "front"
        e["symmetric_180"].append({
            "subtask": h["subtask"], "tag": h["tag"], "axis": h["axis_note"], "reviewer_value": h["symmetric_180"], "survey": survey,
            "question": (f"The selection reviewer marked symmetric_180 = {h['symmetric_180'] or '(empty = not symmetric)'} for the half turn about the "
                         f"{h['axis_note']} axis of the canonical frame (the {axis_role} axis; for {h['subtask']} this is the axis the alignment "
                         f"rotates about); the survey says: {survey}. Judge the tag {h['tag']} from the sheet and say in its note who is right - "
                         f"the reviewer's value or the survey - and about which axis you judged.")})
    print(f"review set: {len(entries)} entries ({n_near} with a symmetry_near entry incl. scaled variants, "
          f"{len(entries) - n_near} added by {n_dis_rows} symmetric_180 disagreement rows)")

    # ---- sheets: which candidate rows are missing? -----------------------------------------------------------------
    need: Dict[str, Dict[str, List[str]]] = {}
    for key, e in entries.items():
        base_key = key[: -len("__scaled")] if e["variant"] == "scaled" else key
        gp, tp = os.path.join(args.out, "geom", base_key + ".json"), os.path.join(args.out, "tex", base_key + ".json")
        g = json.load(open(gp))[e["variant"]]
        tex = json.load(open(tp)).get(e["variant"], {}) if os.path.exists(tp) else {}
        sheet = os.path.join(REPO, e["sheet"])
        shown = SV.sheet_rows(g, tex) if os.path.exists(sheet) else []
        missing = [t for t in e["candidates"] if t not in shown]
        if missing:
            need.setdefault(base_key, {})[e["variant"]] = list(e["candidates"])
    os.makedirs(os.path.join(args.out, "assign"), exist_ok=True)
    with open(os.path.join(args.out, "assign", "review_tags.json"), "w") as f:
        json.dump(need, f, indent=1)
    print(f"sheets lacking a candidate row (or missing): {len(need)} -> assign/review_tags.json")
    if need and not args.no_render:
        renderer = S.Renderer(size=192, ss=2)
        for base_key, extra in need.items():
            g = json.load(open(os.path.join(args.out, "geom", base_key + ".json")))
            status = SV.texture_one(models[base_key], g, renderer, os.path.join(args.out, "tex"), os.path.join(args.out, "renders"), True, extra)
            print(f"  re-rendered {base_key} {extra}: {status}")
        renderer.close()
        # the sheets changed -> refresh symmetry.json / report.md (render paths); verdicts are unchanged
        SV.stage_report(argparse.Namespace(out=args.out, icl=args.icl, only=None, limit=None))
        for key, e in entries.items():
            base_key = key[: -len("__scaled")] if e["variant"] == "scaled" else key
            if base_key in need:
                g = json.load(open(os.path.join(args.out, "geom", base_key + ".json")))[e["variant"]]
                tex = json.load(open(os.path.join(args.out, "tex", base_key + ".json"))).get(e["variant"], {})
                shown = SV.sheet_rows(g, tex, first=need[base_key].get(e["variant"], []))
                still = [t for t in e["candidates"] if t not in shown]
                if still or not os.path.exists(os.path.join(REPO, e["sheet"])):
                    print(f"  WARNING {key}: rows still missing after the re-render: {still}")

    # ---- batches -----------------------------------------------------------------------------------------------------
    order = sorted(entries.values(), key=lambda e: (e["category"], e["model"], e["variant"]))
    n_batches = math.ceil(len(order) / args.batch_size)
    per = math.ceil(len(order) / n_batches)
    index = []
    for b in range(n_batches):
        chunk = order[b * per: (b + 1) * per]
        bid = f"rev_{b:02d}"
        doc = {"batch_id": bid, "contract": rel(os.path.join(args.out, "review_contract.md")),
               "output": rel(os.path.join(args.out, "review", bid + ".json")),
               "n_models": len(chunk), "n_candidates": sum(len(e["candidates"]) for e in chunk), "models": chunk}
        with open(os.path.join(args.out, "assign", bid + ".json"), "w") as f:
            json.dump(doc, f, indent=1)
        index.append({"batch_id": bid, "assignment": rel(os.path.join(args.out, "assign", bid + ".json")), "output": doc["output"],
                      "n_models": len(chunk), "n_candidates": doc["n_candidates"],
                      "n_symmetric_180": sum(1 for e in chunk if e["symmetric_180"]), "categories": sorted({e["category"] for e in chunk})})
    with open(os.path.join(args.out, "assign", "index.json"), "w") as f:
        json.dump({"n_entries": len(order), "n_batches": n_batches, "batch_size": per, "verdicts": list(VERDICTS), "batches": index}, f, indent=1)
    n_cand = sum(len(e["candidates"]) for e in order)
    print(f"{n_batches} batches of <= {per} models -> assign/rev_XX.json ({len(order)} entries, {n_cand} candidate tags, "
          f"{sum(1 for e in order if e['symmetric_180'])} with a symmetric_180 question, {sum(1 for e in order if e['variant'] == 'scaled')} scaled variants)")


if __name__ == "__main__":
    main()
