#!/usr/bin/env python3
"""Execution-settings test (user, 2026-10-07 18:30Z, item 1): success, position error and orientation error per checkpoint,
episode and setting (ta = --temporal-agg, ah1 = --action-horizon 1, ah4 = --action-horizon 4) against the default (16).

Reads lab/diagnostics/exec/<tag>/<run>/step_XXXXXXXX/camera_relocalization-standard/episode_*.json and the default train records
under lab/tierB-train/<run>/step_XXXXXXXX/camera_relocalization-standard/. Writes lab/diagnostics/exec/table.md and table.json.

    python lab_tools/exec_table.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = "camera_relocalization-standard"
TAGS = ["default", "ta", "ah1", "ah4"]


def load(pattern):
    out = {}
    for p in glob.glob(pattern):
        r = json.load(open(p))
        if r.get("task") != TASK:
            continue
        out[(r["run"], int(r["step"]), int(r["episode_index"]))] = r
    return out


def cell(r):
    if r is None:
        return "–"
    if r.get("error"):
        return "ERR"
    sc = r.get("score") or {}
    pe, oe = sc.get("position_error_m"), sc.get("orientation_error_deg")
    mark = "OK" if r.get("success") else "x"
    return f"{mark} {pe:.2f} m / {oe:.0f}°" if pe is not None and oe is not None else mark


def main(argv=None) -> int:
    recs = {"default": load(os.path.join(WT, "lab", "tierB-train", "*", "step_*", TASK, "episode_*.json"))}
    for tag in TAGS[1:]:
        recs[tag] = load(os.path.join(WT, "lab", "diagnostics", "exec", tag, "*", "step_*", TASK, "episode_*.json"))
    keys = sorted({k for tag in TAGS[1:] for k in recs[tag]})
    runs = sorted({(k[0], k[1]) for k in keys})
    lines = ["| checkpoint | episode | " + " | ".join(TAGS) + " |", "|---|---|" + "|".join("---" for _ in TAGS) + "|"]
    table = []
    for run, step in runs:
        eps = sorted(k[2] for k in keys if k[0] == run and k[1] == step)
        for e in eps:
            row = {"run": run, "step": step, "episode": e}
            cells = []
            for tag in TAGS:
                r = recs[tag].get((run, step, e))
                cells.append(cell(r))
                sc = (r or {}).get("score") or {}
                row[tag] = None if r is None else {"success": r.get("success"), "error": r.get("error"), "position_error_m": sc.get("position_error_m"),
                                                   "orientation_error_deg": sc.get("orientation_error_deg"), "geodesic_distance_m": sc.get("geodesic_distance_m"), "steps": r.get("steps"),
                                                   "closest_approach_m": (sc.get("closest_approach") or {}).get("min_position_error_m")}
            lines.append(f"| {run} @{step // 1000}k | {e} | " + " | ".join(cells) + " |")
            table.append(row)
        # per-checkpoint summary row
        cells = []
        for tag in TAGS:
            rs = [recs[tag].get((run, step, e)) for e in eps]
            rs = [r for r in rs if r is not None and not r.get("error")]
            if not rs:
                cells.append("–")
                continue
            n_ok = sum(1 for r in rs if r.get("success"))
            pes = [r["score"]["position_error_m"] for r in rs if (r.get("score") or {}).get("position_error_m") is not None]
            oes = [r["score"]["orientation_error_deg"] for r in rs if (r.get("score") or {}).get("orientation_error_deg") is not None]
            cells.append(f"**{n_ok}/{len(rs)}** med {sorted(pes)[len(pes) // 2]:.2f} m / {sorted(oes)[len(oes) // 2]:.0f}°" if pes and oes else f"**{n_ok}/{len(rs)}**")
        lines.append(f"| {run} @{step // 1000}k | all | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    out_dir = os.path.join(WT, "lab", "diagnostics", "exec")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "table.md"), "w") as f:
        f.write("# Execution-settings test: success / final position error / final orientation error per episode (OK = success, x = miss)\n\n" + md + "\n")
    with open(os.path.join(out_dir, "table.json"), "w") as f:
        json.dump({"tags": TAGS, "rows": table}, f, indent=1)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
