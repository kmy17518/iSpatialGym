#!/usr/bin/env python3
"""The replay-verification outcomes (verify/<task>.json mirrored or fetched) as a markdown table: python verification_table.py DIR > out.md"""
import glob, json, os, sys

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    d = json.load(open(f))
    runs = d.get("runs") or {}
    ladder = " / ".join(f"N={k[1:-4]}:{'pass' if v.get('pass') else 'fail'}" for k, v in runs.items() if k.endswith("same") and k != "n1")
    why = ""
    for k, v in runs.items():
        if k.endswith("same") and v.get("pass") is False and v.get("reasons"):
            r = str(v["reasons"][0])
            why = ("verdict flips" if "success" in r else "trajectory deviation" if "deviation" in r else r[:60]) + f" at N={k[1:-4]}"
            break
    rows.append((d["task"], d.get("num_envs"), ladder, why, round((d.get("seconds") or 0) / 60)))
print("| task | policy N | ladder | first failure | minutes |")
print("|---|---|---|---|---|")
for t, n, ladder, why, mins in rows:
    print(f"| {t} | {n} | {ladder} | {why} | {mins} |")
import collections
print(f"\n{len(rows)} tasks verified; policy N: {dict(sorted(collections.Counter(n for _, n, *_ in rows).items(), reverse=True))}")
