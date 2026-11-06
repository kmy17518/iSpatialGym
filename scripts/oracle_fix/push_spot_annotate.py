#!/usr/bin/env python3
"""Push-spot annotations of floor-standing push targets (oracle-fix 2026-11-03, owner decision 3a; the B6 contacts slot of the icl).

    python scripts/oracle_fix/push_spot_annotate.py --dumps <dir of <category>__<model>.npz> --out runlogs/contacts_<date> [--models K ...]

Input: the target's collision pieces in its root frame, dumped by the sampler (``ISPATIALGYM_PUSH_FACE_DUMP=<dir>``,
``iors_large_nextto.dump_push_geometry``). For every face (+x / -x / +y / -y of the root frame) and every height of the robot's push
band (``--heights``, above the lowest point) the face is probed with the sampler's own section helpers (``horizontal_section``,
``outermost_on_line``): the outermost material along the outward normal at lateral steps across the face. A one-hand spot = a height
where the centre line and ``+-probe_lat`` / ``+-probe_z`` around it are flat within ``--flat`` and at most ``--inset`` inside the AABB;
a two-hand spot = the widest spacing of ``--spacings`` whose two spots are flat and within ``--depth-tol`` of each other. Per face the
chosen height is the one with the least inset (all up to ``--inset-ok`` equal), then the nearest to min(``--prefer-height``, ``--prefer-frac`` x the object's height). Writes ``<out>/<category>__<model>.json``
(``{"key", "category", "model", "scale", "contacts": [{"name", "role": "base_push", "face", "position", "height_m", "inset_m", "spacing_m",
"spots", "approach", "direction", "shape_class", "verdict", "flags", "note"}]}``, root frame, metres) - the file
``scripts/icl/build_icl.py build_contacts`` reads - and ``<out>/push_spots.md`` (a review table of every face and height).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch as th

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ispatialgym.sampling.v2.tasks import iors_large_nextto as NM  # noqa: E402

FACE_NAMES = ("+x", "-x", "+y", "-y")


def load(path: str):
    d = np.load(path)
    pts, off = d["pieces"], d["offsets"]
    pieces = [th.as_tensor(pts[int(off[i]):int(off[i + 1])]).float() for i in range(len(off) - 1) if int(off[i + 1]) > int(off[i])]
    return pieces, th.as_tensor(d["lo"]).float(), th.as_tensor(d["hi"]).float()


def probe(pieces, face, lo, hi, h, lat, probe_lat, probe_z) -> Optional[Dict[str, float]]:
    return NM.face_probe(pieces, face, lo, hi, float(h), probe_lat, probe_z, lateral=float(lat))


def best_pair(pieces, face, lo, hi, h, outer, half_w, a):
    """The two-hand pair at height ``h``: the widest spacing of ``--spacings`` whose two spots are flat (``--flat``), within ``--inset`` of
    the AABB plane and within ``--depth-tol`` of each other along the normal. With ``--off-centre`` (2026-11-04, the swivel chair: its
    armrests / backrest are not symmetric about the face centre line) the pair's midpoint may lie anywhere on the face
    (``--mid-step`` grid), the one nearest the centre line winning at a spacing; else the midpoint is the centre line."""
    def ok(s):
        return s is not None and s["spread"] <= a.flat and outer - s["coord"] <= a.inset

    cache = {}

    def at(t):
        k = round(float(t), 4)
        if k not in cache:
            cache[k] = probe(pieces, face, lo, hi, h, k, a.probe_lat, a.probe_z)
        return cache[k]

    for sp in sorted(a.spacings, reverse=True):
        room = half_w - 0.5 * sp - a.probe_lat
        if room < 0:
            continue
        mids = [0.0]
        if a.off_centre:
            n = int(room / a.mid_step)
            mids = sorted({0.0, *[k * a.mid_step for k in range(-n, n + 1)]}, key=abs)
        for m in mids:
            s1, s2 = at(m + 0.5 * sp), at(m - 0.5 * sp)
            if ok(s1) and ok(s2) and abs(s1["coord"] - s2["coord"]) <= a.depth_tol:
                return (sp, s1, s2)
    return None


def annotate(key: str, pieces, lo, hi, a) -> Dict[str, Any]:
    category, model = key.split("__", 1)
    rows: List[Dict[str, Any]] = []
    contacts: List[Dict[str, Any]] = []
    height_total = float(hi[2] - lo[2])
    prefer = min(float(a.prefer_height), float(a.prefer_frac) * height_total)  # low objects: below the COM, so the push does not tip them
    for face in FACE_NAMES:
        outer = NM.face_outer(face, lo, hi)
        half_w = NM.face_half_width(face, lo, hi)
        cands = []  # (score, h, one-hand record or None, pair or None) of every acceptable height
        for h in np.arange(a.heights[0], min(a.heights[1], height_total - 0.002) + 1e-9, a.height_step):
            f = probe(pieces, face, lo, hi, h, 0.0, a.probe_lat, a.probe_z)
            ok1 = f is not None and f["spread"] <= a.flat and outer - f["coord"] <= a.inset
            pair = best_pair(pieces, face, lo, hi, h, outer, half_w, a)
            rows.append({"face": face, "h": round(float(h), 3), "one": None if f is None else (round(outer - f["coord"], 3), round(f["spread"], 3)), "ok1": ok1,
                         "pair": None if pair is None else pair[0]})
            if not (ok1 or pair):
                continue
            inset = outer - (f["coord"] if ok1 else min(pair[1]["coord"], pair[2]["coord"]))
            # insets up to --inset-ok count as flush (the preferred height decides: a high push tips a chair; 2026-11-03 the swivel
            # chair's backrest won at 0.75 m by 1 mm of inset)
            score = (0 if pair else 1, 0.0 if inset <= a.inset_ok else round(inset, 3), abs(float(h) - prefer))
            cands.append((score, float(h), f if ok1 else None, pair))
        if not cands:
            continue
        # the chosen height, then up to ``--alternatives`` more at least ``--alt-sep`` from every height already kept (the stance search
        # gets more than one reach; owner 3b: the swivel chair is dominated by stance_reach)
        ok_heights = sorted(c[1] for c in cands)
        chosen = []
        for c in sorted(cands, key=lambda c: c[0]):
            if len(chosen) > a.alternatives:
                break
            if all(abs(c[1] - k[1]) >= a.alt_sep - 1e-9 for k in chosen):
                chosen.append(c)
        nx, ny = NM.FACES[face]
        for _, h, f, pair in chosen:
            run = [h, h]  # the contiguous run of acceptable heights around h (the height range of the spot)
            while any(abs(x - (run[0] - a.height_step)) < 1e-6 for x in ok_heights):
                run[0] -= a.height_step
            while any(abs(x - (run[1] + a.height_step)) < 1e-6 for x in ok_heights):
                run[1] += a.height_step
            contacts.append(contact(face, outer, h, f, pair, run, nx, ny, a))
    return {"key": key, "category": category, "model": model, "scale": None, "contacts": contacts,
            "box_local": [lo.tolist(), hi.tolist()], "source": "oracle-fix 2026-11-03 push_spot_annotate.py"}, rows


def contact(face: str, outer: float, h: float, f, pair, run, nx: float, ny: float, a) -> Dict[str, Any]:
    """One B6 push contact: the one-hand spot (or the pair midpoint), the pair, the height and its range (root frame, metres)."""

    def pt(rec):
        return [round(float(v), 4) for v in NM.contact_local(face, rec["coord"], rec["t0"], rec["z"]).tolist()]

    c = {"name": f"push_{face}_h{h:.3f}", "role": "base_push", "face": face, "height_m": round(h, 4), "height_range_m": [round(run[0], 4), round(run[1], 4)],
         "approach": [-float(nx), -float(ny), 0.0], "direction": [-float(nx), -float(ny), 0.0], "shape_class": "face", "verdict": "ok", "flags": []}
    if f is not None:
        c["position"] = pt(f)
        c["inset_m"] = round(outer - f["coord"], 4)
    else:
        mid = {"coord": 0.5 * (pair[1]["coord"] + pair[2]["coord"]), "t0": 0.5 * (pair[1]["t0"] + pair[2]["t0"]), "z": pair[1]["z"]}
        c["position"] = pt(mid)
        c["inset_m"] = round(outer - mid["coord"], 4)
        c["flags"].append("two_hand_only")
    if pair is not None:
        c["spacing_m"] = round(pair[0], 3)
        c["spots"] = [pt(pair[1]), pt(pair[2])]
    c["note"] = (f"derived from the collision pieces by scripts/oracle_fix/push_spot_annotate.py (flat {a.flat} m, inset <= {a.inset} m, probes "
                 f"+-{a.probe_lat} / +-{a.probe_z} m, heights {a.heights[0]}-{a.heights[1]} m step {a.height_step}); review before production")
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--heights", type=float, nargs=2, default=(0.15, 0.75))
    ap.add_argument("--height-step", type=float, default=0.01)
    ap.add_argument("--prefer-height", type=float, default=0.45)
    ap.add_argument("--prefer-frac", type=float, default=0.6, help="... and at most this fraction of the object's height")
    ap.add_argument("--probe-lat", type=float, default=0.03)
    ap.add_argument("--probe-z", type=float, default=0.01)
    ap.add_argument("--flat", type=float, default=0.02)
    ap.add_argument("--inset", type=float, default=0.15)
    ap.add_argument("--depth-tol", type=float, default=0.03)
    ap.add_argument("--off-centre", action="store_true", help="two-hand pairs with the midpoint anywhere on the face")
    ap.add_argument("--mid-step", type=float, default=0.02)
    ap.add_argument("--inset-ok", type=float, default=0.05, help="insets up to this count as flush when ranking heights")
    ap.add_argument("--spacings", type=float, nargs="+", default=[0.40, 0.34, 0.30, 0.26, 0.22])
    ap.add_argument("--alternatives", type=int, default=2, help="extra heights per face besides the chosen one")
    ap.add_argument("--alt-sep", type=float, default=0.05, help="... at least this far from every height kept")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    md = ["# push spots (oracle-fix 2026-11-03)", "", "per model: the chosen spot per face; then every probed height (inset m, spread m of the one-hand probe; two-hand spacing)", ""]
    for path in sorted(glob.glob(os.path.join(a.dumps, "*.npz"))):
        key = os.path.splitext(os.path.basename(path))[0]
        if a.models and key not in a.models:
            continue
        pieces, lo, hi = load(path)
        ann, rows = annotate(key, pieces, lo, hi, a)
        with open(os.path.join(a.out, key + ".json"), "w") as f:
            json.dump(ann, f, indent=1)
        md += [f"## {key}", "", f"box {[round(v, 3) for v in lo.tolist()]} .. {[round(v, 3) for v in hi.tolist()]}, {len(pieces)} pieces", ""]
        md += [f"probes +-{a.probe_lat} / +-{a.probe_z} m, heights {a.heights[0]}-{a.heights[1]} step {a.height_step}", "",
               "| face | height m | range m | position (root) | inset m | spacing m |", "|---|---|---|---|---|---|"]
        for c in ann["contacts"]:
            md.append(f"| {c['face']} | {c['height_m']} | {c['height_range_m']} | {c.get('position')} | {c.get('inset_m')} | {c.get('spacing_m', '-')} |")
        if not ann["contacts"]:
            md.append("| (none) | | | | |")
        md += ["", "<details><summary>all heights</summary>", "", "| face | h | one-hand (inset, spread) | ok | two-hand spacing |", "|---|---|---|---|---|"]
        md += [f"| {r['face']} | {r['h']} | {r['one']} | {'y' if r['ok1'] else ''} | {r['pair'] or ''} |" for r in rows]
        md += ["", "</details>", ""]
        print(key, [(c["face"], c["height_m"], c.get("inset_m"), c.get("spacing_m")) for c in ann["contacts"]])
    md_path = os.path.join(a.out, "push_spots.md" if not a.models else "push_spots__" + "_".join(a.models) + ".md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
