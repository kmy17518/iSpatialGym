#!/usr/bin/env python3
"""Aggregate the per-scene inventories of ``ispatialgym.sampling.scene_inventory`` into ``<root>/summary.json`` and
``<root>/README.md`` (tables per scene: supports, large partners, articulated objects with the joint that needs a review).

    python scripts/inventory/summarize.py [--root runlogs/scene_inventory_20260915]

No simulator import; runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ROOT = os.path.join(REPO, "runlogs", "scene_inventory_20260915")
SCENES = ("house_single_floor", "house_double_floor_upper", "house_double_floor_lower", "Rs_int", "hotel_suite_large", "restaurant_diner", "office_cubicles_right")


def load(root: str, scene: str, part: str) -> Dict[str, dict]:
    d = os.path.join(root, scene, part)
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn)) as f:
                out[fn[:-5]] = json.load(f)
    return out


def best_edge(rec: dict, s: str = "0.45"):
    best = None
    for e in rec.get("edges", []):
        r = e["runs"].get(s, {})
        v = r.get("longest_free_run_m", 0.0)
        if best is None or v > best[1]:
            best = (e["name"], v, r.get("free_fraction_chairs_removed", 0.0), e.get("blocked_in_front"))
    return best


def fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()
    summary: Dict[str, Any] = {"scenes": {}}
    lines: List[str] = ["# Scene inventories (B4)", "",
                        "Produced by `python -m ispatialgym.sampling.scene_inventory --scene <scene>` (one process per scene, full scene, in place).",
                        "Per scene: `supports/<name>.json` (top raster, hull, four edges with free-floor runs at stand-offs 0.35 / 0.45 / 0.55 m with the",
                        "chairs within 1 m removed, walls in front, chairs, occupants), `partners/<name>.json` (same + under-clearance map + supports within",
                        "3.5 m with geodesic distance), `artic/<name>.json` (instance scale, joints measured in the instance frame, handle candidates of the",
                        "canonical heuristic, world heights, free floor in front, prior reviewer verdict of the canonical model) and `artic/strips/<name>.png`",
                        "(review renders: overview / per-joint zoom with lettered markers / opened 60 %).", "",
                        "Edge run = longest contiguous lateral run (m) of base positions on free floor in front of the edge at stand-off 0.45 m (chairs removed).", ""]
    for scene in SCENES:
        sup, par, art = load(args.root, scene, "supports"), load(args.root, scene, "partners"), load(args.root, scene, "artic")
        if not (sup or par or art):
            continue
        sc: Dict[str, Any] = {"supports": {}, "partners": {}, "artic": {}}
        lines += [f"## {scene}", ""]
        # supports
        lines += [f"### supports ({len(sup)})", "", "| name | category | rooms | top z | area m² | free cells | best edge run 0.45 (edge, m, free frac) | blocked in front | chairs ≤ 1 m | in union |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in sorted(sup.items(), key=lambda kv: (-kv[1]["top"]["n_free_cells"])):
            be = best_edge(r)
            sc["supports"][name] = {"category": r["category"], "rooms": r["rooms"], "top_z": r["top"]["z"], "area_m2": r["top"]["area_m2"], "n_free_cells": r["top"]["n_free_cells"],
                                    "best_edge": be[0] if be else None, "best_run_m": be[1] if be else 0.0, "chairs": r["chairs_within_1m"], "in_union": r.get("in_union", False)}
            blocked = (be[3] or {}).get("category") if be else None
            lines.append(f"| {name} | {r['category']} | {','.join(r['rooms'])} | {fmt(r['top']['z'])} | {fmt(r['top']['area_m2'])} | {r['top']['n_free_cells']} | "
                         f"{be[0] if be else '-'}, {fmt(be[1]) if be else '-'}, {fmt(be[2]) if be else '-'} | {blocked or '-'} | {len(r['chairs_within_1m'])} | {'yes' if r.get('in_union') else ''} |")
        lines.append("")
        # partners
        lines += [f"### large partners ({len(par)})", "", "| name | category | rooms | face z | face cells | under ≥ 0.15 m | under ≥ 0.25 m | max clearance | best edge run 0.45 | supports ≤ 3 m (geodesic) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in sorted(par.items()):
            uc = r.get("under_clearance", {})
            near = [f"{s['support']} ({s['geodesic_m']})" for s in r.get("nearby_supports", []) if (s.get("geodesic_m") or 99) <= 3.0]
            be = best_edge(r)
            sc["partners"][name] = {"category": r["category"], "rooms": r["rooms"], "face_z": r["top"]["z"], "face_cells": r["top"]["n_free_cells"],
                                    "under_ge_0.15": uc.get("fraction_ge", {}).get("0.15"), "under_ge_0.25": uc.get("fraction_ge", {}).get("0.25"),
                                    "max_clearance_m": uc.get("max_clearance_m"), "supports_within_3m": near, "best_run_m": be[1] if be else 0.0}
            lines.append(f"| {name} | {r['category']} | {','.join(r['rooms'])} | {fmt(r['top']['z'])} | {r['top']['n_free_cells']} | {fmt(uc.get('fraction_ge', {}).get('0.15'))} | "
                         f"{fmt(uc.get('fraction_ge', {}).get('0.25'))} | {fmt(uc.get('max_clearance_m'))} | {fmt(be[1]) if be else '-'} | {'; '.join(near) or '-'} |")
        lines.append("")
        # articulated objects
        lines += [f"### articulated objects ({len(art)})", "",
                  "| name | category/model | scale | rooms | joint | kind | status | best handle (kind, height m, width mm) | front free m (eroded) | canonical annotation | prior verdict | strip |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in sorted(art.items()):
            joints = r.get("joints", [])
            sc["artic"][name] = {"category": r["category"], "model": r["model"], "scale": r.get("scale"), "rooms": r.get("rooms"), "canonical_annotation": r.get("canonical_annotation"),
                                 "joints": [{"joint": j["joint"], "kind": j.get("kind"), "status": j.get("status"),
                                             "handles": [(h["kind"], h.get("height_world"), round(1000 * h["width"])) for h in j.get("handles", [])],
                                             "fallback": (j.get("fallback") or {}).get("kind"), "front_free_m": j.get("front_free_m"), "front_free_eroded_m": j.get("front_free_eroded_m"),
                                             "prior_review": (j.get("prior_review") or {}).get("chosen")} for j in joints],
                                 "review_needed": (not r.get("canonical_annotation")) and any(j.get("status") in ("handle", "edge") for j in joints)}
            if not joints:
                lines.append(f"| {name} | {r['category']}/{r['model']} | {r.get('scale')} | {','.join(r.get('rooms') or [])} | - | - | no openable joint | | | {'yes' if r.get('canonical_annotation') else 'no'} | | |")
            for j in joints:
                hs = j.get("handles", [])
                best = None
                if hs:
                    h = hs[0]
                    best = f"{h['kind']}, {fmt(h.get('height_world'))}, {round(1000 * h['width'])}"
                elif j.get("fallback"):
                    fb = j["fallback"]
                    best = f"{fb['kind']}{'' if fb.get('exposed') else ' (not exposed)'}, {fmt(fb.get('height_world'))}"
                pr = j.get("prior_review") or {}
                lines.append(f"| {name} | {r['category']}/{r['model']} | {r.get('scale')} | {','.join(r.get('rooms') or [])} | {j['joint']} | {j.get('kind', '-')} | {j.get('status')} | "
                             f"{best or '-'} | {fmt(j.get('front_free_m'))} ({fmt(j.get('front_free_eroded_m'))}) | {'yes' if r.get('canonical_annotation') else 'NO'} | "
                             f"{(pr.get('chosen') or '-') + ('/' + pr.get('result', '') if pr else '')} | {r.get('strip') or '-'} |")
        # review workload: distinct (category, model, scale) groups among the objects without a canonical annotation
        groups: Dict[tuple, List[str]] = {}
        for name, a in sc["artic"].items():
            if a["review_needed"]:
                groups.setdefault((a["category"], a["model"], tuple(round(x, 2) for x in (a.get("scale") or []))), []).append(name)
        sc["review_groups"] = [{"category": k[0], "model": k[1], "scale": list(k[2]), "objects": v} for k, v in sorted(groups.items())]
        lines += [f"Review needed (no canonical handle annotation, graspable joint): {sum(1 for a in sc['artic'].values() if a['review_needed'])} objects in "
                  f"{len(groups)} distinct (model, scale) groups: " + "; ".join(f"{k[0]}/{k[1]} x{len(v)} scale {list(k[2])}" for k, v in sorted(groups.items())), ""]
        summary["scenes"][scene] = sc
    with open(os.path.join(args.root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.root, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.root}/summary.json and README.md for {len(summary['scenes'])} scenes")


if __name__ == "__main__":
    main()
