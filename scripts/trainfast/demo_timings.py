#!/usr/bin/env python3
"""trainfast: one line per finished demo (meta.json) - success, steps, wall, timings, straight-arm counters, plan lines of its collector log.

    demo_timings.py [--tag base --tag flags ...] [--key <substring>]
"""
import argparse, glob, json, os, re, sys
F = "/vision/u/minyeong/projects/isg_home/trainfast"
ap = argparse.ArgumentParser(); ap.add_argument("--tag", action="append", default=None); ap.add_argument("--key", default="")
a = ap.parse_args()
tags = a.tag or sorted(os.path.basename(d) for d in glob.glob(f"{F}/out/*"))
for tag in tags:
    for m in sorted(glob.glob(f"{F}/out/{tag}/*{a.key}*/g*/demo/*/meta.json"), key=os.path.getmtime):
        try:
            d = json.load(open(m))
        except Exception as e:
            print(tag, m, repr(e)); continue
        t = d.get("timings") or {}
        key = m.split("/")[-4].replace("inter_object_relation_satisfaction-", "iors-").replace("configuration_matching-", "cm-")[:34]
        tt = {k: round(v) for k, v in t.items() if isinstance(v, (int, float)) and v}
        sa = (d.get("fast") or {}).get("straight_arm") or {}
        sa_txt = f"straight ok={sa.get('straight_arm_ok')} fb={sa.get('straight_arm_fallback')} {sa.get('fallback_why')}" if sa else ""
        print(f"{tag:6s} {key:34s} succ={d.get('success')!s:5s} steps={d.get('n_steps')} act={d.get('n_action_steps')} settle={d.get('n_settle_steps')} wall={round(d.get('wall_time_s') or 0)} {tt} {sa_txt}")
