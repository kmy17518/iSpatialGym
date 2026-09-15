#!/usr/bin/env python3
"""Prepare the reviewer round of the symmetry survey (``runlogs/symmetry_20260915``).

Review set = every model with a ``symmetry_near`` entry (canonical model, and the scaled variant as ``<key>__scaled``
when its ``symmetry_near_scaled`` is non-empty) plus every alignment target whose ``symmetric_180`` reviewer flag
disagrees with the survey's half turn about the alignment axis. Per model the reviewer judges the candidate tags from
the contact sheet ``renders/<key>.jpg``; a sheet that lacks a row for a candidate (or does not exist because the model
had no geometric candidate) is (re-)rendered here with the review tags forced onto it (``survey_symmetry.texture_one``
with ``extra_tags``), then ``report`` is re-run so ``symmetry.json`` points at the new sheets.

Modes::

    python scripts/symmetry/prepare_review.py                 # first round: assign/rev_00.json ... (all candidates)
    python scripts/symmetry/prepare_review.py --supplement    # after a survey re-run: only the (key, tag) candidates that
                                                              # no existing batch asked -> assign/rev_NN.json from the
                                                              # next free number (existing batches are never rewritten)
    python scripts/symmetry/prepare_review.py --layout        # write assign/render_layout.json: the sheet row order and
                                                              # displayed rotations of the issued batches (from the
                                                              # superseded v1 texture scores), for
                                                              # survey_symmetry.py texture --layout-file

Outputs: ``assign/rev_XX.json`` (batches of <= ``--batch-size`` models: key, sheet, candidate tags with the survey's
scores and classes, ``symmetric_180`` questions, uses, scale), ``assign/index.json``, ``assign/review_tags.json`` (what
was re-rendered). The contract the reviewers follow is ``review_contract.md``; they write ``review/<batch_id>.json``;
``aggregate_review.py`` merges the verdicts.

Env: ``base`` conda env (the re-render needs pyrender / EGL, ~0.3 GB of GPU).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import symcore as S  # noqa: E402
import survey_symmetry as SV  # noqa: E402

VERDICTS = ("preserved", "broken", "unsure")


def rel(path: str) -> str:
    return os.path.relpath(path, REPO)


def split_key(key: str) -> Tuple[str, str]:
    """entry key -> (model key, variant)."""
    return (key[: -len("__scaled")], "scaled") if key.endswith("__scaled") else (key, "base")


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
        blob = (f"; largest difference blob {t['blob_mm']:.0f} mm ({t['blob_px'] / max(t.get('n6', 1), 1):.1f}x a 6 x 6 mm patch, {t['blob_frac'] * 100:.1f} % of the "
                f"{t['blob_view']} view, mean {t.get('blob_d', 0) * 255:.0f}/255)" if t.get("blob_px") else "; no difference blob >= 3 x 3 mm")
        sil = (f"; silhouette difference {t['sil_mm']:.0f} mm ({t['sil_px'] / max(t.get('n6', 1), 1):.1f}x a 6 x 6 mm patch, {t['sil_view']} view)"
               if t.get("sil_px") else "")
        out["survey_texture"] = (f"{s.get('tex_class', '?')} (MAD {t['mad'] * 255:.1f}, p95 {t['p95'] * 255:.1f}, p99 {t.get('p99', 0) * 255:.0f} /255 on the "
                                 f"blurred, fully covered pixels{blob}{sil}; union-mask MAD {t.get('mad_union', 0) * 255:.1f}, silhouette IoU {t.get('iou', 0):.3f})")
        out["sheet_row"] = f"{tag}  ({t.get('shown', t.get('worst', ''))})"
    else:
        out["survey_texture"] = "not rendered"
    return out


def build_review_set(sym: dict, icl: List[dict], out_dir: str) -> Tuple[Dict[str, dict], int]:
    """Entries of the review set from the current symmetry.json (+ the symmetric_180 disagreements); returns
    (entries, number of disagreement rows)."""
    entries: Dict[str, dict] = collections.OrderedDict()

    def entry(key: str, variant: str) -> dict:
        base_key = split_key(key)[0]
        rec = sym[base_key]
        if key not in entries:
            src = rec if variant == "base" else rec["scaled"]
            entries[key] = {"key": key, "category": rec["category"], "model": rec["model"], "variant": variant,
                            "scale": rec["scale"] if variant == "scaled" else None,
                            "sheet": rel(os.path.join(out_dir, "renders", key + ".jpg")), "bbox_m": src.get("bbox"), "n_faces": rec.get("n_faces"),
                            "uses": rec["uses"],
                            "survey": {"symmetry": rec["symmetry"] if variant == "base" else rec["symmetry_scaled"],
                                       "symmetry_near": rec["symmetry_near"] if variant == "base" else rec["symmetry_near_scaled"]},
                            "candidates": collections.OrderedDict(), "symmetric_180": []}
        return entries[key]

    def add_tag(key: str, variant: str, tag: str, reason: str):
        e = entry(key, variant)
        rec = sym[split_key(key)[0]]
        src = rec if variant == "base" else rec["scaled"]
        if tag not in e["candidates"]:
            exact_raw = src.get("geom_exact_raw", S.expand_tags(src.get("geom_exact", [])))
            near_raw = src.get("geom_near_raw", S.expand_tags(src.get("geom_near", [])))
            e["candidates"][tag] = {"asked_because": reason, **tag_context(src["scores"], tag, exact_raw, near_raw)}
        else:
            e["candidates"][tag]["asked_because"] += "; " + reason

    skipped_flat = 0
    for key, rec in sym.items():
        if rec.get("flat_render"):  # the sheet cannot show this asset's look (reflection-map pattern): no review until a textured render exists
            skipped_flat += 1
            continue
        for t in rec["symmetry_near"]:
            add_tag(key, "base", t, "symmetry_near")
        if rec.get("symmetry_near_scaled"):
            for t in rec["symmetry_near_scaled"]:
                add_tag(key + "__scaled", "scaled", t, "symmetry_near of the spawn-scaled variant")
    if skipped_flat:
        print(f"{skipped_flat} flat_render models left out of the review set (their symmetries wait for a render with the reflection maps)")
    n_dis_rows = 0
    for h in SV.alignment_half_turns(icl, sym):
        if not h["disagree"] or sym[h["key"]].get("flat_render"):
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
    return entries, n_dis_rows


def load_assignments(out_dir: str) -> Tuple[Dict[str, dict], List[str]]:
    """Existing assignment entries by key (merged candidates across batches) and the batch files."""
    asked: Dict[str, dict] = {}
    files = sorted(glob.glob(os.path.join(out_dir, "assign", "rev_*.json")))
    for f in files:
        d = json.load(open(f))
        for e in d["models"]:
            a = asked.setdefault(e["key"], {"candidates": set(), "symmetric_180": set(), "batches": []})
            a["candidates"] |= set(e["candidates"])
            a["symmetric_180"] |= {(q["subtask"], q["tag"]) for q in e.get("symmetric_180", [])}
            a["batches"].append(d["batch_id"])
    return asked, files


def sheet_rows_of(out_dir: str, key: str, variant: str, first: List[str] = (), tex_dir: str = "tex", classify=None) -> List[str]:
    """Rows a contact sheet shows (recomputed from the geometry and texture records)."""
    base_key = split_key(key)[0] if variant == "scaled" else key
    g = json.load(open(os.path.join(out_dir, "geom", base_key + ".json")))[variant]
    tp = os.path.join(out_dir, tex_dir, base_key + ".json")
    tex = json.load(open(tp)).get(variant, {}) if os.path.exists(tp) else {}
    rec = json.load(open(tp)) if os.path.exists(tp) else {}
    if rec.get("sheet_rows", {}).get(variant):  # v2 records store the rows they were drawn with
        return rec["sheet_rows"][variant]
    return SV.sheet_rows(g, tex, first=first, classify=classify) if tex else []


def render_missing(out_dir: str, models: Dict[str, dict], need: Dict[str, Dict[str, List[str]]], layout: Optional[dict] = None):
    """Re-render the sheets in ``need`` ({model key: {variant: [tags]}}) with those tags forced (after the rows a layout
    file prescribes, so issued batches keep their row order), then refresh symmetry.json / report.md."""
    if not need:
        return
    renderer = S.Renderer(size=384, ss=2)
    for base_key, extra in need.items():
        g = json.load(open(os.path.join(out_dir, "geom", base_key + ".json")))
        lay = (layout or {}).get(base_key, {})
        first = {v: list(dict.fromkeys(list(lay.get(v, {}).get("first", [])) + tags)) for v, tags in extra.items()}
        pins = {v: lay.get(v, {}).get("pin", {}) for v in extra}
        status = SV.texture_one(models[base_key], g, renderer, os.path.join(out_dir, "tex"), os.path.join(out_dir, "renders"), True, first, pins)
        print(f"  re-rendered {base_key} {first}: {status}")
    renderer.close()


def write_batches(out_dir: str, entries: List[dict], start: int, batch_size: int) -> List[dict]:
    order = sorted(entries, key=lambda e: (e["category"], e["model"], e["variant"]))
    n_batches = math.ceil(len(order) / batch_size) if order else 0
    per = math.ceil(len(order) / n_batches) if n_batches else 0
    index = []
    for b in range(n_batches):
        chunk = order[b * per: (b + 1) * per]
        bid = f"rev_{start + b:02d}"
        doc = {"batch_id": bid, "contract": rel(os.path.join(out_dir, "review_contract.md")),
               "output": rel(os.path.join(out_dir, "review", bid + ".json")),
               "n_models": len(chunk), "n_candidates": sum(len(e["candidates"]) for e in chunk), "models": chunk}
        with open(os.path.join(out_dir, "assign", bid + ".json"), "w") as f:
            json.dump(doc, f, indent=1)
        index.append({"batch_id": bid, "assignment": rel(os.path.join(out_dir, "assign", bid + ".json")), "output": doc["output"],
                      "n_models": len(chunk), "n_candidates": doc["n_candidates"],
                      "n_symmetric_180": sum(1 for e in chunk if e["symmetric_180"]), "categories": sorted({e["category"] for e in chunk})})
    return index


def write_index(out_dir: str):
    index = []
    for f in sorted(glob.glob(os.path.join(out_dir, "assign", "rev_*.json"))):
        d = json.load(open(f))
        index.append({"batch_id": d["batch_id"], "assignment": rel(f), "output": d["output"], "n_models": d["n_models"],
                      "n_candidates": d["n_candidates"], "n_symmetric_180": sum(1 for e in d["models"] if e["symmetric_180"]),
                      "categories": sorted({e["category"] for e in d["models"]})})
    with open(os.path.join(out_dir, "assign", "index.json"), "w") as f:
        json.dump({"n_entries": sum(i["n_models"] for i in index), "n_batches": len(index), "verdicts": list(VERDICTS), "batches": index}, f, indent=1)


def mode_layout(args):
    """assign/render_layout.json: for every model with a v1 texture record, the v1 sheet rows (v1 texture classes,
    v1 forced tags) as ``first`` and, for the tags the issued batches ask about, the v1 displayed rotation as ``pin``."""
    v1_dir = os.path.join(args.out, "superseded", "v1")
    asked, _ = load_assignments(args.out)
    forced_v1 = json.load(open(os.path.join(v1_dir, "review_tags.json"))) if os.path.exists(os.path.join(v1_dir, "review_tags.json")) else {}
    layout: Dict[str, dict] = {}
    for tp in sorted(glob.glob(os.path.join(v1_dir, "tex", "*.json"))):
        key = os.path.splitext(os.path.basename(tp))[0]
        rec = json.load(open(tp))
        g_all = json.load(open(os.path.join(args.out, "geom", key + ".json")))
        for v in ("base", "scaled"):
            if v not in rec or v not in g_all:
                continue
            ekey = key if v == "base" else key + "__scaled"
            has_sheet = os.path.exists(os.path.join(args.out, "renders", ekey + ".jpg"))
            first = SV.sheet_rows(g_all[v], rec[v], first=forced_v1.get(key, {}).get(v, []), classify=S.classify_tex_v1) if has_sheet else []
            pin = {t: rec[v][t]["worst"] for t in asked.get(ekey, {}).get("candidates", set()) if t in rec[v]}
            if first or pin:
                layout.setdefault(key, {})[v] = {"first": first, "pin": pin}
    path = os.path.join(args.out, "assign", "render_layout.json")
    with open(path, "w") as f:
        json.dump(layout, f, indent=1)
    print(f"{rel(path)}: {len(layout)} models, {sum(len(d.get('pin', {})) for m in layout.values() for d in m.values())} pinned rows "
          f"(v1 sheet layout for the re-render; issued batches: {len(asked)} entries)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=SV.DEFAULT_OUT)
    ap.add_argument("--icl", default=SV.DEFAULT_ICL)
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--no-render", action="store_true", help="only list the sheets that lack rows, do not render them")
    ap.add_argument("--supplement", action="store_true", help="add batches for the candidates no existing batch asked about")
    ap.add_argument("--layout", action="store_true", help="write assign/render_layout.json from the superseded v1 records and exit")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "assign"), exist_ok=True)
    if args.layout:
        mode_layout(args)
        return

    sym = json.load(open(os.path.join(args.out, "symmetry.json")))
    icl = list(csv.DictReader(open(args.icl)))
    models = {m["key"]: m for m in json.load(open(os.path.join(args.out, "models.json")))}
    entries, n_dis_rows = build_review_set(sym, icl, args.out)
    n_near = sum(1 for e in entries.values() if any(c["asked_because"].startswith("symmetry_near") for c in e["candidates"].values()))
    print(f"review set: {len(entries)} entries ({n_near} with a symmetry_near entry incl. scaled variants, "
          f"{len(entries) - n_near} added by {n_dis_rows} symmetric_180 disagreement rows)")

    start = 0
    if args.supplement:  # keep only what no issued batch asked; batch numbers continue
        asked, files = load_assignments(args.out)
        start = len(files)
        kept: Dict[str, dict] = collections.OrderedDict()
        for key, e in entries.items():
            a = asked.get(key, {"candidates": set(), "symmetric_180": set()})
            new_tags = [t for t in e["candidates"] if t not in a["candidates"]]
            new_q = [q for q in e["symmetric_180"] if (q["subtask"], q["tag"]) not in a["symmetric_180"]]
            if not new_tags and not new_q:
                continue
            e2 = dict(e)
            e2["candidates"] = collections.OrderedDict((t, e["candidates"][t]) for t in e["candidates"] if t in new_tags or any(q["tag"] == t for q in new_q))
            e2["symmetric_180"] = new_q
            e2["supplement_of"] = a.get("batches", [])
            if e2["candidates"]:
                kept[key] = e2
        print(f"supplement: {len(kept)} entries with {sum(len(e['candidates']) for e in kept.values())} candidate tags not asked in {start} issued batches")
        entries = kept

    # ---- sheets: which candidate rows are missing? -----------------------------------------------------------------
    layout_path = os.path.join(args.out, "assign", "render_layout.json")
    layout = json.load(open(layout_path)) if os.path.exists(layout_path) else {}
    need: Dict[str, Dict[str, List[str]]] = {}
    for key, e in entries.items():
        base_key, variant = split_key(key)
        sheet = os.path.join(REPO, e["sheet"])
        shown = sheet_rows_of(args.out, key, variant) if os.path.exists(sheet) else []
        missing = [t for t in e["candidates"] if t not in shown]
        if missing:
            need.setdefault(base_key, {})[variant] = list(shown) + missing if shown else list(e["candidates"])
    tags_path = os.path.join(args.out, "assign", "review_tags.json" if not args.supplement else f"review_tags_supplement_{start:02d}.json")
    with open(tags_path, "w") as f:
        json.dump(need, f, indent=1)
    print(f"sheets lacking a candidate row (or missing): {len(need)} -> {rel(tags_path)}")
    if need and not args.no_render:
        render_missing(args.out, models, need, layout)
        SV.stage_report(argparse.Namespace(out=args.out, icl=args.icl, only=None, limit=None))
        for key, e in entries.items():
            base_key, variant = split_key(key)
            if base_key in need:
                shown = sheet_rows_of(args.out, key, variant)
                still = [t for t in e["candidates"] if t not in shown]
                if still or not os.path.exists(os.path.join(REPO, e["sheet"])):
                    print(f"  WARNING {key}: rows still missing after the re-render: {still}")

    # ---- batches -----------------------------------------------------------------------------------------------------
    new_index = write_batches(args.out, list(entries.values()), start, args.batch_size)
    write_index(args.out)
    n_cand = sum(len(e["candidates"]) for e in entries.values())
    print(f"{len(new_index)} batches ({', '.join(i['batch_id'] for i in new_index) or '-'}) -> assign/ ({len(entries)} entries, {n_cand} candidate tags, "
          f"{sum(1 for e in entries.values() if e['symmetric_180'])} with a symmetric_180 question, "
          f"{sum(1 for e in entries.values() if e['variant'] == 'scaled')} scaled variants)")


if __name__ == "__main__":
    main()
