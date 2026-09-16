#!/usr/bin/env python3
"""E1 pilot ledger + yield audit from the pulled pod directories.

    python scripts/isg/pilot_ledger.py --manifest runlogs/pilot_20260915/jobs_sample.json --live runlogs/runpod/pilot/live \
        [--killed runlogs/pilot_20260915/killed_at_stop.json] [--requested 2] --out runlogs/pilot_20260915

Writes ``progress.json`` (per alias: pod, rc, wall, n_written, instance ids / seeds, rejection histogram, exception line,
state ∈ {complete, partial, zero, crashed, killed, never_started}, demo outcomes) and ``audit.md`` (per subtask: aliases,
jobs done, n_written histogram, crashes grouped by exception, dominant attempt-level and candidate-level rejections,
median / max wall, attempts per written instance, demo verdicts). ``scene_reset`` is a counter of scene resets, not a
rejection, and is reported apart.

Sources per pod dir ``<live>/<pod name>/``: ``done.txt`` (``<jid> rc=<rc> wall=<s> n_written=<n>``), ``<jid>/summary.json``
(v2 ``isg-sampling-summary-v2``: ``problems.<alias>.<template>.{n_written,time_s,rejections,rejection_examples,seeds,
written,error}``), ``<jid>/run.log`` (``... instance i attempt j rejected in <stage> after <s>s: [<reason>] <message>``,
``Traceback`` blocks, ``RUN_ISG_RESULT {...}`` of the demo jobs).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import statistics

REJ_RE = re.compile(r"(?:\] )?([A-Za-z0-9_\-]+): instance (\d+) attempt (\d+) rejected in (\w+) after ([\d.]+)s: \[(\w+)\] (.*)")
EXC_RE = re.compile(r"^(?:[A-Za-z_.]*(?:Error|Exception|Exit)|KeyboardInterrupt)\b.*")
COUNTERS = {"scene_reset"}


def read_done(pod_dir):
    out = {}
    p = os.path.join(pod_dir, "done.txt")
    if not os.path.exists(p):
        return out
    for line in open(p):
        if " rc=" not in line:
            continue
        jid, rest = line.split(" rc=", 1)
        parts = dict(kv.split("=") for kv in ("rc=" + rest).split())
        out.setdefault(jid.strip(), []).append({"rc": int(parts["rc"]), "wall": int(parts["wall"]), "n_written": int(parts.get("n_written", -1))})
    return out


def parse_run_log(path):
    rej, exc, results, attempts = [], [], [], set()
    if not os.path.exists(path):
        return rej, exc, results, 0
    lines = open(path, errors="replace").read().splitlines()
    for i, line in enumerate(lines):
        m = REJ_RE.search(line)
        if m:
            rej.append({"alias": m.group(1), "instance": int(m.group(2)), "attempt": int(m.group(3)), "stage": m.group(4), "s": float(m.group(5)),
                        "reason": m.group(6), "msg": m.group(7)[:300]})
            attempts.add((m.group(1), int(m.group(2)), int(m.group(3))))
        if line.startswith("Traceback"):
            for j in range(i + 1, min(i + 60, len(lines))):
                if EXC_RE.match(lines[j]):
                    exc.append(lines[j][:200])
                    break
        if "RUN_ISG_RESULT" in line:
            try:
                results.append(json.loads(line.split("RUN_ISG_RESULT", 1)[1].strip()))
            except Exception:
                pass
    return rej, exc, results, len(attempts)


def _alias_entry(jid, alias, subtask, pod, summ, arec, runs, rej, exc, n_attempts, killed_jids, requested):
    n_written, time_s, rejections, examples, seeds, written, error = 0, None, {}, {}, [], [], None
    for prob in (summ.get("problems") or {}).values():
        for t in prob.values():
            if not isinstance(t, dict):
                continue
            n_written += int(t.get("n_written", 0))
            time_s = (time_s or 0) + float(t.get("time_s") or 0)
            for k, v in (t.get("rejections") or {}).items():
                rejections[k] = rejections.get(k, 0) + int(v)
            for k, v in ((t.get("counters") or {}).items()):
                rejections.setdefault(k, 0)
            examples.update(t.get("rejection_examples") or {})
            seeds += list(t.get("seeds") or [])
            written += [os.path.basename(w).replace("-tro_state.json", "") for w in (t.get("written") or [])]
            error = error or t.get("error")
    if arec:
        n_written = max(n_written, int(arec.get("n_written") or 0))
        time_s = time_s or arec.get("time_s")
        error = error or arec.get("error")
    if n_attempts is None:
        n_attempts = len({(r["instance"], r["attempt"]) for r in rej})
    last = runs[-1] if runs else None
    if jid in killed_jids and n_written < requested:
        state = "killed"
    elif not runs and not summ.get("problems") and not arec:
        state = "never_started"
    elif n_written >= requested:
        state = "complete"
    elif n_written > 0:
        state = "partial"
    elif error or (exc and not arec):
        state = "crashed"
    else:
        state = "zero"
    return {"job": jid, "subtask": subtask, "pod": pod, "runs": runs, "rc": last["rc"] if last else None, "wall": last["wall"] if last else None,
            "n_written": n_written, "time_s": time_s, "instances": written, "seeds": seeds, "rejections": rejections, "rejection_examples": examples,
            "attempts_logged": n_attempts, "rejected_attempts": rej, "exceptions": exc if not arec else ([error] if error else []), "error": error, "state": state}


def _merge(ledger, alias, entry):
    rec = ledger.get(alias)
    if rec is None or (entry["n_written"], entry["state"] == "complete") > (rec["n_written"], rec["state"] == "complete"):
        if rec is not None:
            entry["other_runs"] = rec.get("other_runs", []) + [{"pod": rec["pod"], "job": rec["job"], "state": rec["state"], "n_written": rec["n_written"]}]
        ledger[alias] = entry
    else:
        rec.setdefault("other_runs", []).append({"pod": entry["pod"], "job": entry["job"], "state": entry["state"], "n_written": entry["n_written"]})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="one or more manifests, comma-separated (single-alias jobs or C5 chunks)")
    ap.add_argument("--customs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ispatialgym", "assets", "task_specs", "isg_customs.json"))
    ap.add_argument("--live", required=True)
    ap.add_argument("--killed", default=None)
    ap.add_argument("--requested", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jobs = {}
    for mpath in args.manifest.split(","):
        for j in json.load(open(mpath)):
            if j.get("aliases"):  # C5 chunk: one job, several aliases
                for a in j["aliases"]:
                    jobs.setdefault(j["id"], {"id": j["id"], "alias": None, "subtask": None, "aliases": j["aliases"], "subtasks": j.get("subtasks")})
            else:
                jobs[j["id"]] = j
    customs = json.load(open(args.customs)) if args.customs else {}

    def subtask_of(alias):
        e = customs.get(alias) or {}
        if e.get("subtask"):
            return e["subtask"]
        iddl = e.get("iddl", "")
        fam, fn = iddl.split("/", 1) if "/" in iddl else ("", iddl)
        fn = os.path.splitext(fn)[0]
        return fam if fam in ("object_scaling", "mental_rotation", "disembedding") and fn == fam else (f"{fam}-{fn}" if fam else fn)
    killed = json.load(open(args.killed)) if args.killed and os.path.exists(args.killed) else {}
    killed_jids = set(killed.values())
    ledger = {}
    demos = collections.defaultdict(list)
    for pod_dir in sorted(glob.glob(os.path.join(args.live, "*"))):
        pod = os.path.basename(pod_dir)
        done = read_done(pod_dir)
        for jdir in sorted(glob.glob(os.path.join(pod_dir, "*__*"))):
            jid = os.path.basename(jdir)
            rej, exc, results, n_attempts = parse_run_log(os.path.join(jdir, "run.log"))
            if jid.startswith("demo__"):
                src = jid[len("demo__"):]
                for r in results:
                    demos[src].append({"pod": pod, "instance": r.get("instance"), "status": r.get("status"), "success": r.get("success"),
                                       "reason": r.get("reason"), "wall_s": r.get("wall_time_s") or r.get("wall_s"), "n_steps": r.get("n_steps"),
                                       "qa": r.get("qa") or r.get("qa_summary")})
                if not results:
                    demos[src].append({"pod": pod, "status": "killed" if jid in killed_jids else ("no result" if done.get(jid) else "not run"),
                                       "exception": exc[-1] if exc else None})
                continue
            j = jobs.get(jid)
            if j is None:
                continue
            summ = {}
            sp = os.path.join(jdir, "summary.json")
            if os.path.exists(sp):
                try:
                    summ = json.load(open(sp))
                except Exception:
                    summ = {}
            alias_list = j.get("aliases") or [j["alias"]]
            for alias in alias_list:
                sub_summ = {"problems": {k: v for k, v in (summ.get("problems") or {}).items() if k == alias}}
                arec = (summ.get("aliases") or {}).get(alias) or {}
                sub_rej = [r for r in rej if alias in r.get("msg", "") or True] if len(alias_list) == 1 else [r for r in rej if r.get("alias") == alias]
                entry = _alias_entry(jid, alias, subtask_of(alias) if j.get("aliases") else j["subtask"], pod, sub_summ, arec, done.get(jid, []),
                                     sub_rej, exc, n_attempts if len(alias_list) == 1 else None, killed_jids, args.requested)
                _merge(ledger, alias, entry)
            continue
            n_written, time_s, rejections, examples, seeds, written, error = 0, None, {}, {}, [], [], None
            for prob in (summ.get("problems") or {}).values():
                for t in prob.values():
                    if not isinstance(t, dict):
                        continue
                    n_written += int(t.get("n_written", 0))
                    time_s = (time_s or 0) + float(t.get("time_s") or 0)
                    for k, v in (t.get("rejections") or {}).items():
                        rejections[k] = rejections.get(k, 0) + int(v)
                    examples.update(t.get("rejection_examples") or {})
                    seeds += list(t.get("seeds") or [])
                    written += [os.path.basename(w).replace("-tro_state.json", "") for w in (t.get("written") or [])]
                    error = error or t.get("error")
            runs = done.get(jid, [])
            last = runs[-1] if runs else None
            if jid in killed_jids and not runs:
                state = "killed"
            elif jid in killed_jids and runs and n_written < args.requested:
                state = "killed"  # re-run was in flight
            elif not runs and not summ:
                state = "never_started"
            elif n_written >= args.requested:
                state = "complete"
            elif n_written > 0:
                state = "partial"
            elif error or exc:
                state = "crashed"
            else:
                state = "zero"
            rec = ledger.get(j["alias"])
            entry = {"job": jid, "subtask": j["subtask"], "pod": pod, "runs": runs, "rc": last["rc"] if last else None, "wall": last["wall"] if last else None,
                     "n_written": n_written, "time_s": time_s, "instances": written, "seeds": seeds, "rejections": rejections, "rejection_examples": examples,
                     "attempts_logged": n_attempts, "rejected_attempts": rej, "exceptions": exc, "error": error, "state": state}
            # an alias may have run on several pods (re-dispatch after a crash): keep the best outcome, remember the others
            if rec is None or (entry["n_written"], entry["state"] == "complete") > (rec["n_written"], rec["state"] == "complete"):
                if rec is not None:
                    entry["other_runs"] = rec.get("other_runs", []) + [{"pod": rec["pod"], "state": rec["state"], "n_written": rec["n_written"], "exceptions": rec["exceptions"]}]
                ledger[j["alias"]] = entry
            else:
                rec.setdefault("other_runs", []).append({"pod": pod, "state": state, "n_written": n_written, "exceptions": exc})
    for jid, j in jobs.items():
        for alias in (j.get("aliases") or [j["alias"]]):
            if alias in ledger:
                continue
            ledger[alias] = {"job": jid, "subtask": subtask_of(alias) if j.get("aliases") else j["subtask"], "pod": None, "runs": [], "rc": None, "wall": None,
                             "n_written": 0, "time_s": None, "instances": [], "seeds": [], "rejections": {}, "rejection_examples": {}, "attempts_logged": 0,
                             "rejected_attempts": [], "exceptions": [], "error": None, "state": "killed" if jid in killed_jids else "never_started"}
        continue
        if j["alias"] not in ledger:
            ledger[j["alias"]] = {"job": jid, "subtask": j["subtask"], "pod": None, "runs": [], "rc": None, "wall": None, "n_written": 0, "time_s": None,
                                  "instances": [], "seeds": [], "rejections": {}, "rejection_examples": {}, "attempts_logged": 0, "rejected_attempts": [],
                                  "exceptions": [], "error": None, "state": "killed" if jid in killed_jids else "never_started"}
    for alias, rec in ledger.items():
        rec["demos"] = demos.get(rec["job"], [])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "progress.json"), "w") as f:
        json.dump({"requested_per_alias": args.requested, "aliases": ledger}, f, indent=1)

    # ---- audit ----
    by = collections.defaultdict(list)
    for alias, rec in ledger.items():
        by[rec["subtask"]].append(rec)
    md = ["# E1 pilot — yield audit", "", f"{len(ledger)} aliases; states: " + ", ".join(f"{k} {v}" for k, v in sorted(collections.Counter(r['state'] for r in ledger.values()).items())), "",
          "| subtask | aliases | ran | n_written 0/1/2 | crashed | killed / never | median wall s | max wall s | attempts per instance | dominant attempt rejections (excl. scene_reset) | demos ok / run |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    detail = []
    for st in sorted(by):
        recs = by[st]
        ran = [r for r in recs if r["state"] in ("complete", "partial", "zero", "crashed")]
        hist = collections.Counter(min(r["n_written"], args.requested) for r in ran)
        crashed = [r for r in recs if r["state"] == "crashed"]
        kn = sum(1 for r in recs if r["state"] in ("killed", "never_started"))
        walls = [r["time_s"] for r in ran if r["time_s"]]
        rej = collections.Counter()
        for r in ran:
            for k, v in r["rejections"].items():
                if k not in COUNTERS:
                    rej[k] += v
        n_inst = sum(r["n_written"] for r in ran)
        n_att = sum(r["attempts_logged"] for r in ran) + n_inst
        api = f"{n_att / n_inst:.1f}" if n_inst else "-"
        dem = [d for r in recs for d in r["demos"]]
        dem_ok = sum(1 for d in dem if d.get("success"))
        dem_run = sum(1 for d in dem if d.get("status") not in (None, "killed", "not run", "no result"))
        md.append(f"| {st} | {len(recs)} | {len(ran)} | {hist.get(0,0)}/{hist.get(1,0)}/{hist.get(2,0)} | {len(crashed)} | {kn} | "
                  f"{statistics.median(walls):.0f} | {max(walls):.0f} | {api} | " if walls else f"| {st} | {len(recs)} | {len(ran)} | {hist.get(0,0)}/{hist.get(1,0)}/{hist.get(2,0)} | {len(crashed)} | {kn} | - | - | {api} | ")
        md[-1] += ", ".join(f"{k} {v}" for k, v in rej.most_common(4)) + f" | {dem_ok} / {dem_run} |"
        # detail: exceptions, candidate-level messages, per-alias lines
        exc = collections.Counter(e for r in recs for e in r["exceptions"])
        errs = collections.Counter((r["error"] or "")[:110] for r in recs if r["error"] and not r["exceptions"])
        cand = collections.Counter()
        for r in ran:
            for a in r["rejected_attempts"]:
                cand[f"[{a['reason']}] " + re.sub(r"[-\d.]+", "#", a["msg"])[:90]] += 1
        detail.append(f"\n## {st}\n")
        if exc:
            detail.append("Crashes (Traceback): " + "; ".join(f"`{k}` ×{v}" for k, v in exc.most_common(5)))
        if errs:
            detail.append("Setup errors: " + "; ".join(f"`{k}` ×{v}" for k, v in errs.most_common(5)))
        if cand:
            detail.append("Candidate-level (log) rejections: " + "; ".join(f"`{k}` ×{v}" for k, v in cand.most_common(6)))
        for r in sorted(recs, key=lambda r: r["state"]):
            demo_s = ", ".join(f"{d.get('status')}{'/' + str(d.get('reason')) if d.get('reason') else ''}" for d in r["demos"]) or "-"
            detail.append(f"- `{r['job'][4:]}`: **{r['state']}** n={r['n_written']} wall={r['time_s'] and round(r['time_s'])} pod={r['pod']} "
                          f"rej={dict(collections.Counter({k: v for k, v in r['rejections'].items() if k not in COUNTERS}).most_common(3))} "
                          f"{('exc=' + r['exceptions'][-1][:80]) if r['exceptions'] else ''} {('err=' + r['error'][:80]) if r['error'] and not r['exceptions'] else ''} demos: {demo_s}")
    with open(os.path.join(args.out, "audit.md"), "w") as f:
        f.write("\n".join(md + detail) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
