"""Compare evaluator records episode by episode (the cluster sanity battery, lab_tools/cluster/MERGE_SANITY_PROMPT.md).

* ``--a A --b B``: the same battery run twice on one machine (A: the unmerged lab-vec-eval, B: the merge). Records are paired
  by their path below A and B (``sanity_battery.sh`` writes the same layout for every label). Compared: the counters
  (``vec_gate_3b.COUNTERS``: success, reason, stop reason, binary success ever / final, streamed steps, first success step,
  total steps, settle windows) and the per-step arrays of the raw dirs (``vec_gate_3b.ARRAYS``: base x / y / yaw, head
  camera pose, pose errors, proprio base velocity, joints, tracked objects). ``--strict`` requires both to be identical:
  open-loop replays are deterministic on one machine, so any difference means the merge changed the evaluator path.
  Without it vec_gate_3b's rule applies (bit-identity for one environment; for clones the counters within a step and
  the trajectories within --tol-pos / --tol-rot / --tol-err).
* ``--a A --expected lab_tools/cluster/expected_replays.json``: the lab machine's counters of the same replays. Another
  machine's floating point may move a tolerance crossing by a step or two (``--step-slack``, default 3); the verdicts
  must match, except for entries marked ``borderline`` (they ended within 0.1 mm of a tolerance) and entries whose
  reference was made under other settings (``reference_settings_differ``: only the verdict is compared).
* ``--summarize DIR``: success / error counts per run directory (the closed-loop records: their outcomes are chaotic under
  the renderer's run-to-run noise - 0.14-0.29 per-episode flip rate between two runs of the same code - so they are
  counted, never paired).

    python lab_tools/cluster/compare_replays.py --a sanity/base/records --b sanity/merged/records --strict --out sanity/compare_base_vs_merged.json
    python lab_tools/cluster/compare_replays.py --a sanity/merged/records --expected lab_tools/cluster/expected_replays.json --out sanity/compare_merged_vs_lab.json
    python lab_tools/cluster/compare_replays.py --summarize sanity/base/closedloop sanity/merged/closedloop
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WT)

from lab_tools.vec_gate_3b import COUNTERS, compare_arrays, counters_close, judge  # noqa: E402


def records(root: str) -> dict:
    return {os.path.relpath(p, root): p for p in glob.glob(os.path.join(root, "**", "episode_*.json"), recursive=True)}


def raw_dir(rec: dict):
    rd = rec.get("raw_dir")
    if not rd:
        return None
    return rd if os.path.isabs(rd) else os.path.join(WT, rd)


def pair_mode(args) -> dict:
    A, B = records(args.a), records(args.b)
    items, ok_all = [], bool(A) and bool(B)
    for rel in sorted(set(A) | set(B)):
        item = {"record": rel}
        if rel not in A or rel not in B:
            item.update({"pass": False, "status": f"only in {'a' if rel in A else 'b'}"})
        else:
            ra, rb = json.load(open(A[rel])), json.load(open(B[rel]))
            item.update({"episode": rb.get("episode_index"), "task": rb.get("task"), "slot": rb.get("slot"), "num_envs": rb.get("num_envs"),
                         "counters": {k: [ra.get(k), rb.get(k)] for k in COUNTERS}})
            if ra.get("error") or rb.get("error"):
                item.update({"pass": False, "status": "error", "errors": [ra.get("error"), rb.get("error")]})
            else:
                da, db = raw_dir(ra), raw_dir(rb)
                cmp = compare_arrays(db, da) if da and db else {}
                if args.strict:
                    bad = [f"{k}: {ra.get(k)} vs {rb.get(k)}" for k in COUNTERS if ra.get(k) != rb.get(k)]
                    ok, reasons = judge(cmp, (not bad, bad), int(rb.get("slot") or 0), 0.0, 0.0, 0.0, num_envs=1)
                else:
                    n = int(rb.get("num_envs") or 1)
                    ok, reasons = judge(cmp, counters_close(rb, ra, num_envs=n), int(rb.get("slot") or 0), args.tol_pos, args.tol_rot, args.tol_err, num_envs=n)
                if not cmp:
                    ok, reasons = False, reasons + ["raw dir missing"]
                first = {f: r.get("first_diff_step") for f, r in cmp.items() if r.get("first_diff_step") is not None}
                item.update({"pass": ok, "status": "compared", "reasons": reasons, "first_diff_step": first,
                             "max_deviation": {f: r.get("max_abs") or [r.get("max_pos"), r.get("max_quat")] for f, r in cmp.items() if r.get("status") == "ok"}})
        ok_all = ok_all and item["pass"]
        items.append(item)
    return {"mode": "pair", "strict": args.strict, "a": args.a, "b": args.b, "n": len(items), "n_pass": sum(i["pass"] for i in items), "pass": ok_all, "records": items}


def expected_mode(args) -> dict:
    exp = json.load(open(args.expected))
    want = {(e["split"], e["task"], int(e["episode"])): e for e in exp["episodes"]}
    seen, items = set(), []
    for rel, p in sorted(records(args.a).items()):
        r = json.load(open(p))
        key = (r.get("replay_split") or r.get("split"), r.get("task"), int(r.get("episode_index", -1)))
        e = want.get(key)
        if e is None:
            continue
        seen.add(key)
        reasons, s = [], args.step_slack
        if r.get("error"):
            reasons.append(f"error: {str(r.get('error'))[:200]}")
        elif r.get("success") != e["success"] and not e.get("borderline"):
            reasons.append(f"success {r.get('success')} vs lab {e['success']} (reason {r.get('reason')} vs {e.get('reason')})")
        elif e["success"] and not r.get("binary_success_ever"):
            reasons.append("the goal never held (lab: it did)")
        elif not e.get("reference_settings_differ"):
            # a borderline verdict decides the epilogue's second window, so only the first success step is comparable
            keys = (("first_success_step", s),) if e.get("borderline") else (("first_success_step", s), ("n_steps_total", 2 * s + 1))
            for k, slack in keys:
                a, b = r.get(k), e.get(k)
                if a is None or b is None or abs(int(a) - int(b)) > slack:
                    reasons.append(f"{k} {a} vs lab {b} (slack {slack})")
        items.append({"record": rel, "split": key[0], "task": key[1], "episode": key[2], "slot": r.get("slot"), "num_envs": r.get("num_envs"),
                      "pass": not reasons, "reasons": reasons, "success": [r.get("success"), e["success"]],
                      "first_success_step": [r.get("first_success_step"), e.get("first_success_step")],
                      "n_steps_total": [r.get("n_steps_total"), e.get("n_steps_total")], "note": e.get("note")})
    missing = [f"{k[0]} {k[1]} episode {k[2]}" for k in sorted(want) if k not in seen]
    return {"mode": "expected", "a": args.a, "expected": args.expected, "step_slack": args.step_slack, "n": len(items),
            "n_pass": sum(i["pass"] for i in items), "not_run": missing, "pass": bool(items) and all(i["pass"] for i in items), "records": items}


def summarize(dirs) -> dict:
    out = {}
    for d in dirs:
        runs = {}
        for rel, p in records(d).items():
            r = json.load(open(p))
            run = os.path.dirname(rel) or "."
            s = runs.setdefault(run, {"n": 0, "success": 0, "errors": 0, "binary_success_ever": 0, "episodes": []})
            s["n"] += 1
            s["success"] += bool(r.get("success"))
            s["errors"] += bool(r.get("error"))
            s["binary_success_ever"] += bool(r.get("binary_success_ever"))
            s["episodes"].append([r.get("episode_index"), r.get("slot"), r.get("success"), (r.get("score") or {}).get("position_error_m")])
        out[d] = runs
    return {"mode": "summarize", "dirs": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--expected")
    ap.add_argument("--summarize", nargs="+")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--tol-pos", type=float, default=5e-3)
    ap.add_argument("--tol-rot", type=float, default=5e-3)
    ap.add_argument("--tol-err", type=float, default=5e-3)
    ap.add_argument("--step-slack", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.summarize:
        res = summarize(args.summarize)
    elif args.a and args.b:
        res = pair_mode(args)
    elif args.a and args.expected:
        res = expected_mode(args)
    else:
        ap.error("--a with --b or --expected, or --summarize")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    if res["mode"] == "summarize":
        for d, runs in res["dirs"].items():
            for run, s in sorted(runs.items()):
                print(f"{d}/{run}: {s['success']}/{s['n']} success, {s['binary_success_ever']} reached, {s['errors']} errors")
        return 0
    for it in res["records"]:
        if not it["pass"] or args.out is None:
            print(f"  {'ok  ' if it['pass'] else 'FAIL'} {it['record']}: {'; '.join(it.get('reasons') or []) or it.get('status', '')}")
    if res.get("not_run"):
        print("  not run:", ", ".join(res["not_run"]))
    print(f"{res['mode']}{' (strict)' if res.get('strict') else ''}: {res['n_pass']}/{res['n']} pass -> {'PASS' if res['pass'] else 'FAIL'}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
