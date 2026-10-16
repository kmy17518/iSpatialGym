"""Benchmark metrics over a folder of runs: ``python -m ispatialgym.eval.summarize <runs dir> [--csv out.csv]``.

Per task family and overall: the geometric success rate (primary), the symbolic success rate, and the mean position /
orientation error of the error metric over all episodes and over the symbolically successful ones (an episode without a
component is left out of that mean). Settle-eval-on runs also report their off mode. An episode its policy marked as a
``forced_failure`` counts as failed. Runs that ended in an error are listed as unscored and excluded.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Any

RATES = (("geometric", "geometric"), ("symbolic", "symbolic"))
OTHER_RATES = (("other_geometric", "off-mode geometric"), ("other_symbolic", "off-mode symbolic"))


def episode_row(path: str, root: str) -> dict[str, Any] | None:
    """The scoring columns of one run's ``meta.json``; None for any other JSON file."""
    try:
        with open(path) as f:
            m = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(m, dict) or m.get("mode") != "policy" or "status" not in m:
        return None
    row = {
        "episode": os.path.relpath(os.path.dirname(path), root),
        "family": m.get("task_family") or "?",
        "subtask": m.get("subtask"),
    }
    if m["status"] == "error":
        return {**row, "status": "unscored", "error": m.get("error")}
    metric = m.get("error_metric") or {}
    forced = bool(m.get("forced_failure"))
    row.update(
        {
            "status": "scored",
            "geometric": bool(m.get("geometric_success")) and not forced,
            "symbolic": bool(m.get("symbolic_success")) and not forced,
            "position_error": metric.get("position_error"),
            "orientation_error": metric.get("orientation_error"),
            "reason": m.get("reason"),
            "n_steps": m.get("n_steps"),
        }
    )
    other = m.get("other_mode") or {}
    if m.get("settle_eval") and other:
        row.update(
            {
                "other_geometric": bool(other.get("geometric_success")) and not forced,
                "other_symbolic": bool(other.get("symbolic_success")) and not forced,
            }
        )
    return row


def episode_rows(root: str) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(root, "**", "meta.json"), recursive=True))
    return [row for row in (episode_row(p, root) for p in paths) if row is not None]


def _rate(rows: list[dict[str, Any]], key: str) -> tuple[int, int] | None:
    values = [bool(r[key]) for r in rows if r.get(key) is not None]
    return (sum(values), len(values)) if values else None


def _fmt_rate(rate: tuple[int, int] | None) -> str:
    return "-" if rate is None else f"{rate[0]}/{rate[1]} {100.0 * rate[0] / rate[1]:5.1f}%"


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def _fmt_errors(rows: list[dict[str, Any]], key: str) -> str:
    def fmt(v):
        return "-" if v is None else f"{v:.4f}"

    return f"{fmt(_mean(rows, key))} / {fmt(_mean([r for r in rows if r.get('symbolic')], key))}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark metrics of a folder of run_isg runs.")
    parser.add_argument("root")
    parser.add_argument("--csv", default=None, help="also write one row per episode to this file")
    args = parser.parse_args(argv)

    rows = episode_rows(args.root)
    scored = [r for r in rows if r["status"] == "scored"]
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in scored:
        families[r["family"]].append(r)
    rates = RATES + (OTHER_RATES if any(r.get("other_geometric") is not None for r in scored) else ())

    print(f"episodes: {len(rows)} found, {len(scored)} scored, {len(rows) - len(scored)} unscored")
    header = f"  {'family':<36} " + " ".join(f"{label:>18}" for _, label in rates)
    print(header + f"  {'position error (all / symbolic)':>32}  {'orientation error (all / symbolic)':>34}")
    for family, fam_rows in sorted(families.items()):
        cells = " ".join(f"{_fmt_rate(_rate(fam_rows, key)):>18}" for key, _ in rates)
        print(
            f"  {family:<36} {cells}  {_fmt_errors(fam_rows, 'position_error'):>32}  {_fmt_errors(fam_rows, 'orientation_error'):>34}"
        )
    if scored:
        cells = " ".join(f"{_fmt_rate(_rate(scored, key)):>18}" for key, _ in rates)
        print(
            f"  {'overall':<36} {cells}  {_fmt_errors(scored, 'position_error'):>32}  {_fmt_errors(scored, 'orientation_error'):>34}"
        )
    for r in rows:
        if r["status"] == "unscored":
            print(f"  unscored: {r['episode']}  {str(r.get('error'))[:160]}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
