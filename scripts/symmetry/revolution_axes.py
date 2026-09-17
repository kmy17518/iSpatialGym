#!/usr/bin/env python3
"""Cylinder tags for the placed-orientation demo gate (owner decision 2026-09-16 17:10 item 4, fallback applied 20:00).

The closure investigation (``runlogs/closure_probe_20260916/report.md``) ended without meeting its criterion, so the
pre-decided fallback applies: in the ``placed_orientation`` demo gate an object with a **continuous symmetry about its
long axis** (a marker, a pencil, a bottle — anything that rolls like a wheel while the fingers close) is compared
**modulo rotation about that axis**; every other object keeps the full orientation error. This script derives the tag
table the gate reads (``ispatialgym/assets/icl/revolution_axes.json``) from the ICL symmetry survey
(``runlogs/symmetry_20260915/symmetry.json``; the icl ``symmetry`` / ``symmetry_near`` columns are the reviewed
subset of the same tags):

* a model qualifies when one of its survey tag lists — ``symmetry`` (geometry and texture), ``symmetry_near``,
  ``geom_exact``, ``geom_near`` (geometry alone; the texture of a printed marker breaks the *visual* symmetry but not the
  rolling) and the ``_scaled`` variants — carries a body-of-revolution tag ``xinf`` / ``yinf`` / ``zinf`` **about the
  model's long axis**: the largest native bbox extent, at least ``--long-ratio`` (1.5) × the middle one. A plate
  (``zinf`` about its thin axis), a bowl, a puck-shaped lid keep the full error;
* the unscaled tags are used for the models the survey also evaluated at a spawn scale: a non-uniformly scaled pencil
  (2 cm thick) fails the survey's millimetre thresholds although it is the same prism about the same axis.

Output: ``{"<category>/<model>": {"axis": "x"|"y"|"z", "ratio": <long / middle extent>, "tags": [...], "sources": [...],
"bbox": [...], "uses": [...]}}`` plus a ``_meta`` block; the executor (``oracle.executor.revolution_axis``) looks a placed
object up by its ``category`` / ``model``. Models without a tag are absent (full error). Re-run after a survey update:

    python scripts/symmetry/revolution_axes.py [--survey runlogs/symmetry_20260915/symmetry.json] [--out ispatialgym/assets/icl/revolution_axes.json]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

SURVEY = os.path.join(REPO, "runlogs", "symmetry_20260915", "symmetry.json")
OUT = os.path.join(REPO, "ispatialgym", "assets", "icl", "revolution_axes.json")
TAG_LISTS = ("symmetry", "symmetry_near", "geom_exact", "geom_near", "symmetry_scaled", "symmetry_near_scaled")
AXES = "xyz"


def long_axis(bbox: List[float], ratio: float) -> Optional[str]:
    """The axis of the largest extent when it is at least ``ratio`` × the middle extent, else None."""
    if not bbox or len(bbox) < 3:
        return None
    e = [float(v) for v in bbox[:3]]
    order = sorted(range(3), key=lambda k: e[k])
    lo, mid, hi = e[order[0]], e[order[1]], e[order[2]]
    if hi <= 0 or mid <= 0 or hi < ratio * mid:
        return None
    return AXES[order[2]]


def derive(survey: Dict[str, dict], ratio: float) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for key, e in sorted(survey.items()):
        cat, model = e.get("category"), e.get("model")
        if not cat or not model:
            continue
        axis = long_axis(e.get("bbox") or [], ratio)
        if axis is None:
            continue
        sources, tags = [], set()
        for name in TAG_LISTS:
            lst = e.get(name) or []
            if not isinstance(lst, list):
                continue
            hit = [t for t in lst if isinstance(t, str) and t == f"{axis}inf"]
            if hit:
                sources.append(name)
                tags.update(t for t in lst if isinstance(t, str) and t != "none")
        if not sources:
            continue
        bb = [float(v) for v in e["bbox"][:3]]
        order = sorted(bb)
        out[f"{cat}/{model}"] = {
            "axis": axis,
            "ratio": round(order[2] / order[1], 3),
            "tags": sorted(tags),
            "sources": sources,
            "bbox": [round(v, 5) for v in bb],
            "uses": list(e.get("uses") or []),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--survey", default=SURVEY)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--long-ratio", type=float, default=1.5, help="largest / middle native extent for a 'long axis'")
    args = ap.parse_args()
    with open(args.survey) as f:
        survey = json.load(f)
    table = derive(survey, args.long_ratio)
    meta = {
        "_meta": {
            "built": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "survey": os.path.relpath(args.survey, REPO),
            "rule": (f"a body-of-revolution tag (xinf / yinf / zinf) about the model's long axis (largest native extent >= "
                     f"{args.long_ratio} x the middle one) in any of {list(TAG_LISTS)}; the placed_orientation demo gate compares such an "
                     "object modulo rotation about that axis (owner 2026-09-16 17:10 item 4 fallback), every other object keeps the full error"),
            "n_models": len(table),
            "categories": dict(sorted(collections.Counter(k.split("/")[0] for k in table).items())),
        }
    }
    meta.update(table)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=False)
        f.write("\n")
    cats = meta["_meta"]["categories"]
    print(f"{len(table)} models with a revolution axis about their long axis -> {os.path.relpath(args.out, REPO)}")
    print(", ".join(f"{c} {n}" for c, n in sorted(cats.items(), key=lambda kv: -kv[1])[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
