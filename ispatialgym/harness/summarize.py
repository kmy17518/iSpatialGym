"""Aggregate harness episodes: ``python -m ispatialgym.harness.summarize <root> [--csv out.csv]``.

Per episode: success (run_isg's epilogue verdict, the official success of every policy, forced to a failure when the
planner stopped), how the episode ended, calls, repairs, tokens and wall time; then success rates per task family and
overall. Episodes run_isg recorded as ``status: error`` (a provider outage, a harness bug) are **unscored**: listed, not
counted in any rate, and resubmitted with ``harness.run --rerun-unscored``."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List


def run_meta(ep: str) -> Dict[str, Any]:
    metas = sorted(glob.glob(os.path.join(ep, "run", "*", "meta.json")))
    return json.load(open(metas[0])) if metas else {}


def episode_rows(root: str) -> List[Dict[str, Any]]:
    rows = []
    for cfg_path in sorted(glob.glob(os.path.join(root, "**", "harness_config.json"), recursive=True)):
        ep = os.path.dirname(cfg_path)
        alias = json.load(open(cfg_path)).get("alias") or ""
        row: Dict[str, Any] = {"episode": os.path.relpath(ep, root), "alias": alias, "family": alias.split("-")[0], "status": "not run"}
        m = run_meta(ep)
        if m:
            h = m.get("harness") or {}
            pl = h.get("planner") or {}
            usage = pl.get("usage") or {}
            row.update({"stop": (h.get("stop") or {}).get("reason"), "failure_kind": h.get("failure_kind"), "calls": pl.get("calls"),
                        "repairs": pl.get("repairs"), "unreachable": pl.get("unreachable"), "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"), "n_steps": m.get("n_steps"), "wall_s": m.get("wall_time_s")})
            if m.get("status") == "error":
                row.update({"status": "unscored", "error": m.get("error"), "failure_kind": h.get("failure_kind") or "infrastructure"})
            else:
                row.update({"status": "scored", "success": bool(m.get("success")), "binary_success_ever": m.get("binary_success_ever"),
                            "forced_failure": h.get("forced_failure")})
        rows.append(row)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--csv", default=None)
    a = p.parse_args(argv)
    rows = episode_rows(a.root)
    scored = [r for r in rows if r["status"] == "scored"]
    unscored = [r for r in rows if r["status"] == "unscored"]
    fam: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in scored:
        fam[r["family"]][0] += int(r["success"])
        fam[r["family"]][1] += 1
    print(f"episodes: {len(rows)} prepared, {len(scored)} scored, {len(unscored)} unscored, {len(rows) - len(scored) - len(unscored)} not run")
    for f, (s, n) in sorted(fam.items()):
        print(f"  {f:<40} {s}/{n} = {100.0 * s / n:.1f}%")
    if scored:
        s = sum(int(r["success"]) for r in scored)
        macro = sum(v[0] / v[1] for v in fam.values()) / len(fam)
        print(f"  overall (micro) {s}/{len(scored)} = {100.0 * s / len(scored):.1f}%   (family mean {100.0 * macro:.1f}%)")
        print("  how scored episodes ended: " + ", ".join(f"{k} {v}" for k, v in Counter(r.get("stop") or "env_end" for r in scored).most_common()))
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
