#!/usr/bin/env python3
"""Merge the reviewer verdicts of the symmetry review round into ``review_verdicts.json``.

Inputs (``runlogs/symmetry_20260915`` by default): ``symmetry.json`` (survey), ``assign/rev_XX.json`` (the review set:
candidate tags, ``symmetric_180`` questions), ``review/rev_XX.json`` (reviewer output per the contract:
``{"<key>": {"<tag>": {"verdict": preserved | broken | unsure, "note": ...}}}``; the directory may be empty or absent),
``review/overrides.json`` (coordinator decisions, same schema plus ``"by"``; applied after the reviewer files, an
override wins over a reviewer verdict, may name a tag no batch asked about, and is listed in the summary).

Per review-set entry: final ``symmetry`` = survey ``symmetry`` ∪ tags reviewed ``preserved`` (reduced to generators);
``symmetry_near`` = survey ``symmetry_near`` minus ``preserved`` / ``broken`` tags minus what the new ``symmetry``
implies (``unsure`` stays). ``<key>__scaled`` entries update ``symmetry_scaled`` / ``symmetry_near_scaled``. The verdict
on a ``symmetric_180`` question tag becomes ``value`` ``"1"`` (preserved) / ``"0"`` (broken) / ``null`` (unsure or not
reviewed) per alignment subtask. Only models with at least one verdict are listed under ``models`` (the survey values
stand for everything else; unreviewed review-set entries are listed under ``unreviewed``).

Outputs: ``review_verdicts.json`` (consumed by ``scripts/icl/build_icl.py --symmetry`` when it sits next to
``symmetry.json``), ``review_summary.md`` (verdict counts per tag and batch, changed models, ``symmetric_180``
corrections, unreviewed entries); both are also printed. Malformed input is reported, never silently dropped: unknown
keys / tags are ignored with a warning, an invalid verdict counts as ``unsure``.

    python scripts/symmetry/aggregate_review.py [--out runlogs/symmetry_20260915]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import symcore as S  # noqa: E402

VERDICTS = ("preserved", "broken", "unsure")


def tags_of(v) -> List[str]:
    return [t for t in (v or []) if t != "none"]


def merge(survey_sym: List[str], survey_near: List[str], reviewed: Dict[str, dict], flat_render: bool = False) -> Tuple[List[str], List[str]]:
    """Human verdicts win over the survey: ``preserved`` joins ``symmetry``; ``broken`` leaves both ``symmetry`` and
    ``symmetry_near`` (also when a re-run of the survey later called the tag preserved); ``unsure`` on a tag the
    survey calls preserved moves it to ``symmetry_near`` (a symmetry only takes effect once it is confirmed). For a
    ``flat_render`` model (its look lives in a reflection map the sheets cannot show) a ``preserved`` verdict is held in
    ``symmetry_near`` as well - nothing is confirmed until a render with those maps exists."""
    preserved = {t for t, r in reviewed.items() if r["verdict"] == "preserved"}
    broken = {t for t, r in reviewed.items() if r["verdict"] == "broken"}
    unsure = {t for t, r in reviewed.items() if r["verdict"] == "unsure"}
    held = preserved if flat_render else set()
    sym_raw = (set(S.expand_tags(tags_of(survey_sym))) | preserved) - broken - unsure - held
    sym = S.reduce_tags(sym_raw)
    implied = set(S.expand_tags(sym))
    near = [t for t in tags_of(survey_near) if t not in preserved and t not in broken and t not in implied]
    near += [t for t in unsure if t in set(S.expand_tags(tags_of(survey_sym))) and t not in implied and t not in near]
    near += [t for t in held if t not in implied and t not in near]
    return sym or ["none"], S.reduce_tags(near) if near else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "runlogs", "symmetry_20260915"))
    args = ap.parse_args()
    sym = json.load(open(os.path.join(args.out, "symmetry.json")))
    assign_files = sorted(glob.glob(os.path.join(args.out, "assign", "rev_*.json")))
    if not assign_files:
        sys.exit(f"no assignments in {os.path.join(args.out, 'assign')} - run prepare_review.py first")
    entries: Dict[str, dict] = {}
    batch_of: Dict[str, str] = {}
    for f in assign_files:
        d = json.load(open(f))
        for e in d["models"]:
            if e["key"] in entries:  # a supplementary batch adds candidates to a model an earlier batch already asked about
                prev = entries[e["key"]]
                merged = dict(prev)
                merged["candidates"] = {**prev["candidates"], **e["candidates"]}
                merged["symmetric_180"] = prev.get("symmetric_180", []) + [q for q in e.get("symmetric_180", [])
                                                                          if (q["subtask"], q["tag"]) not in {(p["subtask"], p["tag"]) for p in prev.get("symmetric_180", [])}]
                entries[e["key"]] = merged
                batch_of[e["key"]] = f"{batch_of[e['key']]}+{d['batch_id']}"
            else:
                entries[e["key"]] = e
                batch_of[e["key"]] = d["batch_id"]
    # ---- read the verdicts (reviewer batches review/rev_*.json, then review/overrides.json: a coordinator override wins)
    warnings: List[str] = []
    reviewed: Dict[str, Dict[str, dict]] = collections.defaultdict(dict)
    review_files = sorted(glob.glob(os.path.join(args.out, "review", "rev_*.json")))
    overrides_path = os.path.join(args.out, "review", "overrides.json")
    per_batch = collections.Counter()
    overrides: List[dict] = []
    for f in review_files + ([overrides_path] if os.path.exists(overrides_path) else []):
        bid = os.path.splitext(os.path.basename(f))[0]
        is_override = bid == "overrides"
        try:
            d = json.load(open(f))
        except Exception as ex:  # noqa: BLE001
            warnings.append(f"{os.path.basename(f)}: unreadable json ({ex})")
            continue
        if not isinstance(d, dict):
            warnings.append(f"{os.path.basename(f)}: top level is not an object")
            continue
        for key, tags in d.items():
            if key not in entries:
                warnings.append(f"{bid}: unknown key {key} ignored")
                continue
            if not isinstance(tags, dict):
                warnings.append(f"{bid}: {key} is not an object of tags")
                continue
            for tag, v in tags.items():
                if tag not in entries[key]["candidates"] and not is_override:
                    warnings.append(f"{bid}: {key} tag {tag} was not asked - ignored")
                    continue
                verdict = (v or {}).get("verdict") if isinstance(v, dict) else None
                if verdict not in VERDICTS:
                    warnings.append(f"{bid}: {key}/{tag} invalid verdict {verdict!r} -> unsure")
                    verdict = "unsure"
                note = str((v or {}).get("note", ""))[:400] if isinstance(v, dict) else ""
                if is_override:
                    prev = reviewed[key].get(tag)
                    overrides.append({"key": key, "tag": tag, "verdict": verdict, "note": note, "by": (v or {}).get("by", "coordinator"),
                                      "replaces": f"{prev['verdict']} ({prev['batch']})" if prev else "-"})
                    reviewed[key][tag] = {"verdict": verdict, "note": note, "batch": "overrides", "by": (v or {}).get("by", "coordinator")}
                    continue
                if tag in reviewed[key] and reviewed[key][tag]["batch"] != bid:
                    warnings.append(f"{bid}: {key}/{tag} also reviewed in {reviewed[key][tag]['batch']} - later file wins")
                reviewed[key][tag] = {"verdict": verdict, "note": note, "batch": bid}
                per_batch[bid] += 1
    # ---- merge
    models: Dict[str, dict] = {}
    unreviewed, partial, stale = [], [], []
    changed, corrections = [], []
    verdict_counts = collections.Counter()
    tag_counts: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for key, e in sorted(entries.items()):
        base_key = key[: -len("__scaled")] if e["variant"] == "scaled" else key
        rec = sym[base_key]
        rv = reviewed.get(key, {})
        missing = [t for t in e["candidates"] if t not in rv]
        if not rv:
            unreviewed.append(key)
        elif missing:
            partial.append((key, missing))
        cur_sym = rec["symmetry"] if e["variant"] == "base" else (rec.get("symmetry_scaled") or [])
        cur_near = rec["symmetry_near"] if e["variant"] == "base" else (rec.get("symmetry_near_scaled") or [])
        still_candidate = set(S.expand_tags(tags_of(cur_sym))) | set(cur_near) | set(S.expand_tags(cur_near))
        for t, r in rv.items():
            verdict_counts[r["verdict"]] += 1
            tag_counts[t][r["verdict"]] += 1
            if t not in still_candidate:  # a survey re-run classified the tag broken since the batch was issued: kept, applied, flagged
                stale.append((key, t, r["verdict"], r["batch"]))
        if not rv:
            continue  # unreviewed: not listed in "models" -> build_icl.py keeps the survey values
        m = models.setdefault(base_key, {"category": rec["category"], "model": rec["model"], "scale": rec["scale"],
                                         "symmetry": rec["symmetry"], "symmetry_near": rec["symmetry_near"],
                                         "symmetry_scaled": rec.get("symmetry_scaled"), "symmetry_near_scaled": rec.get("symmetry_near_scaled"),
                                         "reviewed": {}, "reviewed_scaled": {}, "symmetric_180": [], "batch": batch_of.get(base_key),
                                         "flat_render": bool(rec.get("flat_render"))})
        if e["variant"] == "base":
            m["symmetry"], m["symmetry_near"] = merge(rec["symmetry"], rec["symmetry_near"], rv, bool(rec.get("flat_render")))
            m["reviewed"] = rv
            if m["symmetry"] != rec["symmetry"] or m["symmetry_near"] != rec["symmetry_near"]:
                changed.append((key, rec["symmetry"], rec["symmetry_near"], m["symmetry"], m["symmetry_near"]))
            for q in e.get("symmetric_180", []):
                r = rv.get(q["tag"])
                value = {"preserved": "1", "broken": "0"}.get(r["verdict"]) if r else None
                m["symmetric_180"].append({"subtask": q["subtask"], "tag": q["tag"], "axis": q["axis"], "reviewer_value": q["reviewer_value"],
                                           "verdict": r["verdict"] if r else None, "value": value, "note": r["note"] if r else ""})
                if value is not None and value != (q["reviewer_value"] or "0"):
                    corrections.append((key, q["subtask"], q["tag"], q["reviewer_value"] or "(empty)", value))
        else:
            m["symmetry_scaled"], m["symmetry_near_scaled"] = merge(rec["symmetry_scaled"], rec["symmetry_near_scaled"], rv, bool(rec.get("flat_render")))
            m["reviewed_scaled"] = rv
            m["batch_scaled"] = batch_of.get(key)
            if m["symmetry_scaled"] != rec["symmetry_scaled"] or m["symmetry_near_scaled"] != rec["symmetry_near_scaled"]:
                changed.append((key, rec["symmetry_scaled"], rec["symmetry_near_scaled"], m["symmetry_scaled"], m["symmetry_near_scaled"]))
    n_rev = len(entries) - len(unreviewed)
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "survey": "symmetry.json", "contract": "review_contract.md",
           "n_review_set": len(entries), "n_reviewed": n_rev, "n_partially_reviewed": len(partial), "n_unreviewed": len(unreviewed),
           "verdict_counts": dict(verdict_counts), "warnings": warnings, "models": models, "unreviewed": unreviewed,
           "partially_reviewed": {k: miss for k, miss in partial},
           "verdicts_on_non_candidates": [{"key": k, "tag": t, "verdict": v, "batch": b} for k, t, v, b in stale],
           "overrides": overrides,
           "batches": {os.path.splitext(os.path.basename(f))[0]: {"n_entries": len(json.load(open(f))["models"]),
                                                                 "n_verdicts": per_batch.get(os.path.splitext(os.path.basename(f))[0], 0),
                                                                 "reviewed": os.path.exists(os.path.join(args.out, "review", os.path.basename(f)))}
                       for f in assign_files}}
    with open(os.path.join(args.out, "review_verdicts.json"), "w") as f:
        json.dump(out, f, indent=1)
    # ---- summary
    L = [f"# Symmetry review — aggregation ({out['generated']})\n",
         f"Review set {len(entries)} entries ({sum(1 for e in entries.values() if e['variant'] == 'scaled')} scaled variants); reviewed {n_rev}, "
         f"partially reviewed {len(partial)}, unreviewed {len(unreviewed)}; verdicts {dict(verdict_counts) or 'none yet'}.\n",
         "| batch | entries | verdicts | file present |\n|---|---|---|---|"]
    for bid, b in out["batches"].items():
        L.append(f"| {bid} | {b['n_entries']} | {b['n_verdicts']} | {'yes' if b['reviewed'] else 'no'} |")
    L.append("\n| tag | preserved | broken | unsure |\n|---|---|---|---|")
    for t in S.TAG_ORDER:
        if t in tag_counts:
            c = tag_counts[t]
            L.append(f"| `{t}` | {c['preserved']} | {c['broken']} | {c['unsure']} |")
    L.append(f"\n## Models whose verdict changed ({len(changed)})\n\n| key | survey `symmetry` / `symmetry_near` | final `symmetry` / `symmetry_near` |\n|---|---|---|")
    for key, s0, n0, s1, n1 in changed:
        L.append(f"| `{key}` | `{' '.join(s0)}` / `{' '.join(n0) or '-'}` | `{' '.join(s1)}` / `{' '.join(n1) or '-'}` |")
    L.append(f"\n## `symmetric_180` corrections ({len(corrections)})\n\n| key | subtask | tag | reviewer value | review value |\n|---|---|---|---|---|")
    for key, st, tag, old, new in corrections:
        L.append(f"| `{key}` | {st} | `{tag}` | {old} | {new} |")
    L.append(f"\n## Coordinator overrides ({len(overrides)}, `review/overrides.json`, applied after the reviewer files)\n")
    if overrides:
        L.append("| key | tag | verdict | replaces | by | note |\n|---|---|---|---|---|---|")
        L += [f"| `{o['key']}` | `{o['tag']}` | {o['verdict']} | {o['replaces']} | {o['by']} | {o['note'][:120]} |" for o in overrides]
    else:
        L.append("none")
    L.append(f"\n## Unreviewed entries ({len(unreviewed)})\n")
    L.append(", ".join(f"`{k}` ({batch_of[k]})" for k in unreviewed) or "none")
    if partial:
        L.append(f"\n## Partially reviewed entries ({len(partial)})\n")
        L += [f"- `{k}` missing {miss}" for k, miss in partial]
    if stale:
        L.append(f"\n## Verdicts on tags that are no longer survey candidates ({len(stale)})\n\n"
                 "The survey re-run classified these tags broken after the batch was issued; the verdicts are kept and applied "
                 "(a human `preserved` still joins `symmetry`), listed here for a second look.\n")
        L += [f"- `{k}` `{t}`: reviewer {v} ({b})" for k, t, v, b in stale]
    if warnings:
        L.append(f"\n## Warnings ({len(warnings)})\n")
        L += [f"- {w}" for w in warnings]
    with open(os.path.join(args.out, "review_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    if len(L) > 40:
        print(f"... ({len(L) - 40} more lines in review_summary.md)")
    print(f"\nreview_verdicts.json: {len(models)} models ({n_rev} of {len(entries)} entries reviewed, {len(unreviewed)} unreviewed, "
          f"{len(changed)} changed, {len(corrections)} symmetric_180 corrections, {len(warnings)} warnings)")


if __name__ == "__main__":
    main()
