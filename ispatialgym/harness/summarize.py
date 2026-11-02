"""Aggregate episodes: ``python -m ispatialgym.harness.summarize <root> [--csv out.csv]``.

``<root>`` holds harness episodes (``harness_config.json``, the run under ``run/<iid>/meta.json``) or plain ``run_isg`` runs
(``<iid>/meta.json``, e.g. replays of the eval instances, or ``run_isg --policy ispatialgym.harness.policy:HarnessPolicy``
runs, whose harness columns come from ``meta.json``'s ``harness`` block). Per episode: the run mode's geometric and symbolic success, the other
settle-eval mode's two for a settle-eval-on run (its off mode, ``ispatialgym.eval.metrics``; a settle-eval-off run has none, whatever
its ``meta.json`` recorded), the error metric, how the episode ended, calls, repairs, tokens and wall time; then per task family and
overall the geometric rate first, then the symbolic rate, then the other mode's two (over the settle-eval-on episodes; left out when
there are none), and per family the mean position / orientation error of the error metric over all episodes and over the
symbolically successful ones (an episode without a component is left out of that mean). A harness episode the planner stopped is
forced to a failure.

Episodes run_isg recorded as ``status: error`` (a provider outage, a harness bug) are **unscored**: listed, not counted in any
rate, and resubmitted with ``harness.run --rerun-unscored``. A ``meta.json`` written before the 2026-10-14 scoring change (no
``geometric_success``) counts as symbolic only (its ``success``) and is flagged."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

RATES = (("geometric", "geometric"), ("symbolic", "symbolic"))
#: only a settle-eval-on episode has these (its off mode)
OTHER_RATES = (("other_geometric", "other mode geometric"), ("other_symbolic", "other mode symbolic"))


def run_meta(ep: str) -> Dict[str, Any]:
    metas = sorted(glob.glob(os.path.join(ep, "run", "*", "meta.json")))
    return json.load(open(metas[0])) if metas else {}


def score_fields(m: Dict[str, Any], forced: bool = False) -> Dict[str, Any]:
    """The scoring columns of a ``meta.json``: geometric / symbolic success of the run mode, of the other mode for a settle-eval-on
    run only (``other_geometric`` / ``other_symbolic``), the error metric; an old ``meta.json`` (no ``geometric_success``) gives its
    ``success`` as the symbolic success and nothing else, flagged."""
    if "geometric_success" not in m:
        return {"old_meta": True, "geometric": None, "symbolic": bool(m.get("success")) and not forced,
                "settle_eval": m.get("settle_eval"), "position_error": None, "orientation_error": None, "error_case": None}
    metric = m.get("error_metric") or {}
    out = {"old_meta": False, "geometric": bool(m.get("geometric_success")) and not forced, "symbolic": bool(m.get("symbolic_success")) and not forced,
           "settle_eval": m.get("settle_eval"), "position_error": metric.get("position_error"), "orientation_error": metric.get("orientation_error"),
           "error_case": metric.get("case")}
    other = m.get("other_mode") or {}
    if m.get("settle_eval") and other:
        out.update({"other_geometric": bool(other.get("geometric_success")) and not forced,
                    "other_symbolic": bool(other.get("symbolic_success")) and not forced})
    return out


def _family(m: Dict[str, Any], variant: str = "") -> str:
    return m.get("task_family") or (variant or m.get("variant") or "").split("-")[0] or "?"


def harness_fields(m: Dict[str, Any]) -> Dict[str, Any]:
    """The harness's columns of a ``meta.json`` (its ``harness`` block): how the planner stopped, calls, repairs, tokens."""
    h = m.get("harness") or {}
    pl = h.get("planner") or {}
    usage = pl.get("usage") or {}
    return {"stop": (h.get("stop") or {}).get("reason"), "failure_kind": h.get("failure_kind"), "calls": pl.get("calls"),
            "repairs": pl.get("repairs"), "unreachable": pl.get("unreachable"), "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens")}


def episode_rows(root: str) -> List[Dict[str, Any]]:
    rows = []
    harness_dirs = []
    for cfg_path in sorted(glob.glob(os.path.join(root, "**", "harness_config.json"), recursive=True)):
        ep = os.path.dirname(cfg_path)
        harness_dirs.append(os.path.abspath(ep) + os.sep)
        alias = json.load(open(cfg_path)).get("alias") or ""
        row: Dict[str, Any] = {"episode": os.path.relpath(ep, root), "alias": alias, "family": alias.split("-")[0], "status": "not run"}
        m = run_meta(ep)
        if m:
            h = m.get("harness") or {}
            row.update({**harness_fields(m), "reason": m.get("qa_reason") or m.get("reason"), "n_steps": m.get("n_steps"), "wall_s": m.get("wall_time_s")})
            if m.get("status") == "error":
                row.update({"status": "unscored", "error": m.get("error"), "failure_kind": h.get("failure_kind") or "infrastructure"})
            else:
                forced = bool(h.get("forced_failure") or m.get("forced_failure"))
                row.update({"status": "scored", "success": bool(m.get("success")) and not forced, "forced_failure": forced, **score_fields(m, forced)})
        rows.append(row)
    for meta_path in sorted(glob.glob(os.path.join(root, "**", "meta.json"), recursive=True)):
        if any(os.path.abspath(meta_path).startswith(d) for d in harness_dirs):
            continue
        try:
            m = json.load(open(meta_path))
        except (OSError, ValueError):
            continue
        if not isinstance(m, dict) or "mode" not in m or "status" not in m:
            continue  # not a run_isg meta.json
        row = {"episode": os.path.relpath(os.path.dirname(meta_path), root), "alias": m.get("subtask") or m.get("variant") or "", "family": _family(m),
               "mode": m.get("mode"), "reason": m.get("qa_reason") or m.get("reason"), "n_steps": m.get("n_steps"), "wall_s": m.get("wall_time_s")}
        h = m.get("harness")  # run_isg --policy ispatialgym.harness.policy:HarnessPolicy run without an episode directory
        if h:
            row.update(harness_fields(m))
        if m.get("status") == "error":
            row.update({"status": "unscored", "error": m.get("error")})
            if h:
                row["failure_kind"] = h.get("failure_kind") or "infrastructure"
        else:
            forced = bool((h or {}).get("forced_failure") or m.get("forced_failure"))
            row.update({"status": "scored", "success": bool(m.get("success")) and not forced, **score_fields(m, forced)})
            if h or m.get("forced_failure") is not None:
                row["forced_failure"] = forced
        rows.append(row)
    return rows


def _rate(rows: List[Dict[str, Any]], key: str) -> Optional[tuple]:
    vals = [bool(r[key]) for r in rows if r.get(key) is not None]
    return (sum(vals), len(vals)) if vals else None


def _fmt_rate(r: Optional[tuple]) -> str:
    return "-" if r is None else f"{r[0]}/{r[1]} {100.0 * r[0] / r[1]:5.1f}%"


def _mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt_err(rows: List[Dict[str, Any]], key: str) -> str:
    sym = [r for r in rows if r.get("symbolic")]
    a, b = _mean(rows, key), _mean(sym, key)
    return f"{'-' if a is None else f'{a:.4f}'} / {'-' if b is None else f'{b:.4f}'}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--csv", default=None)
    a = p.parse_args(argv)
    rows = episode_rows(a.root)
    scored = [r for r in rows if r["status"] == "scored"]
    unscored = [r for r in rows if r["status"] == "unscored"]
    fam: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in scored:
        fam[r["family"]].append(r)
    print(f"episodes: {len(rows)} found, {len(scored)} scored, {len(unscored)} unscored, {len(rows) - len(scored) - len(unscored)} not run")
    rates = RATES + (OTHER_RATES if any(r.get("other_geometric") is not None for r in scored) else ())
    header = f"  {'family':<36} " + " ".join(f"{label:>18}" for _k, label in rates) + f"  {'position error (all / symbolic)':>32}  {'orientation error (all / symbolic)':>34}"
    print(header)
    for f, rs in sorted(fam.items()):
        print(f"  {f:<36} " + " ".join(f"{_fmt_rate(_rate(rs, k)):>18}" for k, _l in rates)
              + f"  {_fmt_err(rs, 'position_error'):>32}  {_fmt_err(rs, 'orientation_error'):>34}")
    if scored:
        print(f"  {'overall (micro)':<36} " + " ".join(f"{_fmt_rate(_rate(scored, k)):>18}" for k, _l in rates))
        means = []
        for k, label in rates:
            per = [_rate(rs, k) for rs in fam.values()]
            per = [x[0] / x[1] for x in per if x is not None]
            means.append(f"{label} {100.0 * sum(per) / len(per):.1f}%" if per else f"{label} -")
        print("  family mean: " + ", ".join(means))
        print("  how scored episodes ended: " + ", ".join(f"{k} {v}" for k, v in Counter(r.get("stop") or r.get("reason") or "success" for r in scored).most_common()))
        old = [r for r in scored if r.get("old_meta")]
        if old:
            print(f"  flagged: {len(old)} meta.json written before the 2026-10-14 scoring change (no geometric_success) counted as symbolic only: "
                  + ", ".join(r["episode"] for r in old[:10]) + (" ..." if len(old) > 10 else ""))
    for r in unscored:
        print(f"  unscored: {r['episode']}  {str(r.get('error'))[:160]}")
    if a.csv:
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
