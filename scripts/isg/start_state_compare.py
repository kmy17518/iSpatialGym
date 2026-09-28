#!/usr/bin/env python3
"""Compare the start states two demo runs recorded (``run_isg`` ``meta.start_state``; engineering track, owner 2026-09-27): a warm
demo (``first_demo.py run --warm``) against a fresh-process load of the same instance (``first_demo.py demo-only --tro
<iid>_pristine-tro_state.json --run-isg-args=--load-only``). Pass = every pose / joint within ``oracle.start_state.TOLERANCES`` (the
exact reload tolerances, 0.1 mm / 0.1 mrad) and every discrete item equal.

usage: start_state_compare.py <warm meta.json | run dir> <fresh meta.json | run dir> [--json [out.json]]
       start_state_compare.py --pairs pairs.json [--json [out.json]]      # [[warm, fresh], ...]
``--json`` without a path prints the rows to stdout.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ispatialgym.oracle import start_state as S  # noqa: E402


def meta_of(p: str) -> dict:
    if os.path.isdir(p):
        hits = glob.glob(os.path.join(p, "**", "meta.json"), recursive=True)
        if not hits:
            raise SystemExit(f"no meta.json under {p}")
        p = hits[0]
    with open(p) as f:
        return json.load(f)


def main() -> int:
    argv = sys.argv[1:]
    out = None
    if "--json" in argv:
        i = argv.index("--json")
        out = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else "-"
    if "--pairs" in argv:
        pairs = json.load(open(argv[argv.index("--pairs") + 1]))
    else:
        pos = [a for a in argv if not a.startswith("--") and a != out]
        if len(pos) < 2:
            raise SystemExit(__doc__)
        pairs = [pos[:2]]
    rows, ok_all = [], True
    for warm, fresh in pairs:
        a, b = meta_of(warm), meta_of(fresh)
        if "start_state" not in a or "start_state" not in b:
            rows.append({"warm": warm, "fresh": fresh, "ok": False, "fails": ["a run recorded no start_state"]})
            ok_all = False
            continue
        rep = S.compare(a["start_state"], b["start_state"])
        rows.append({"warm": warm, "fresh": fresh, "alias": a.get("alias"), **rep})
        ok_all = ok_all and rep["ok"]
        if out != "-":
            print(f"{'OK ' if rep['ok'] else 'BAD'} {str(a.get('alias'))[:70]:70s} identical {rep['identical']} max {rep['max']} {rep['fails'][:3]}")
    if out == "-":
        print(json.dumps(rows, indent=1))
    elif out:
        json.dump(rows, open(out, "w"), indent=1)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
