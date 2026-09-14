#!/usr/bin/env python3
"""Delete disembedding results that were skipped because IORS ontop found the support top unreachable from its one
auto-chosen edge (disembedding searches every edge itself), so that the next --skip-done round redoes them."""
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
n = 0
for p in sorted(glob.glob(os.path.join(HERE, "..", "results", "*__disembedding-0.json"))):
    with open(p) as f:
        d = json.load(f)
    if d.get("status") == "skipped" and "no reachable cell" in str(d.get("reason", "")):
        os.remove(p)
        n += 1
        print("requeued", os.path.basename(p))
print(n, "disembedding jobs requeued")
