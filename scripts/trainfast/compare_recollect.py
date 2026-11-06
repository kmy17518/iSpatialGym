#!/usr/bin/env python3
"""merge_vecnext: a collector rerun (job_recollect.sh) against the job it re-ran, instance by instance.

    compare_recollect.py <rerun job dir> [--traj]

The source is recollect.json's. Per instance: verdict / reason before and after, steps, the per-step costs (meta.fast.step_costs:
env step, QA, record, sleep assist; ms per step), the sleep assist's sleeps / wakes, the load time; --traj also compares the two
demo.hdf5 step by step (driver_agreement.py). Prints a table and a SUMMARY line.
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def metas(job):
    out = {}
    for m in glob.glob(os.path.join(job, "demo", "*", "meta.json")):
        j = json.load(open(m))
        out[j.get("instance_id") or os.path.basename(os.path.dirname(m))] = (os.path.dirname(m), j)
    return out


def costs(j):
    sc = ((j.get("fast") or {}).get("step_costs") or {})
    n = max(1, int(sc.get("steps") or 0))
    return {k: 1000.0 * float(sc.get(k) or 0.0) / n for k in ("env_step_s", "qa_s", "record_s", "sleep_assist_s")}, int(sc.get("steps") or 0)


def main(job, traj=False):
    src = json.load(open(os.path.join(job, "recollect.json")))["source"]
    a, b = metas(src), metas(job)
    rows, agree = [], {}
    print(f"{'instance':>8} {'before':>26} {'after':>26} {'steps':>11} {'env ms':>13} {'qa ms':>11} {'assist ms':>9} {'load s':>11} sleeps")
    for iid in sorted(set(a) | set(b)):
        ja, jb = (a.get(iid) or (None, {}))[1], (b.get(iid) or (None, {}))[1]
        ca, na = costs(ja)
        cb, nb = costs(jb)
        sa = jb.get("sleep_assist") or {}
        ev = sum(len(v) for v in (sa.get("sleeps") or {}).values())
        wk = sum(len(v) for v in (sa.get("wakes") or {}).values())
        la, lb = (ja.get("timings") or {}).get("load_s"), (jb.get("timings") or {}).get("load_s")
        r = {"iid": iid, "before": (ja.get("status"), ja.get("reason")), "after": (jb.get("status"), jb.get("reason")), "steps": (ja.get("n_steps"), jb.get("n_steps")),
             "env_ms": (ca["env_step_s"], cb["env_step_s"]), "qa_ms": (ca["qa_s"], cb["qa_s"]), "assist_ms": cb["sleep_assist_s"], "load_s": (la, lb), "sleeps": ev, "wakes": wk,
             "reused": jb.get("reused_world"), "driver_physics": jb.get("driver_physics")}
        rows.append(r)
        fmt = lambda s: f"{s[0] or '-'}:{s[1] or ''}"[:26]  # noqa: E731
        f1 = lambda x: "-" if x is None else f"{x:.1f}"  # noqa: E731
        print(f"{iid[-6:]:>8} {fmt(r['before']):>26} {fmt(r['after']):>26} {str(r['steps'][0]):>5}/{str(r['steps'][1]):<5} {ca['env_step_s']:6.1f}/{cb['env_step_s']:<6.1f} "
              f"{ca['qa_s']:5.1f}/{cb['qa_s']:<5.1f} {cb['sleep_assist_s']:9.2f} {f1(la):>5}/{f1(lb):<5} {ev}s {wk}w{' reused' if jb.get('reused_world') else ''}")
        if traj and iid in a and iid in b:
            p = subprocess.run([sys.executable, os.path.join(HERE, "driver_agreement.py"), a[iid][0], b[iid][0]], capture_output=True, text=True)
            agree[iid] = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr.strip().splitlines()[-1]
            print(f"{'':>8} {agree[iid]}")
    ok_a = sum(1 for r in rows if r["before"][0] == "success")
    ok_b = sum(1 for r in rows if r["after"][0] == "success")
    same = sum(1 for r in rows if r["before"] == r["after"])
    w = lambda k, i: [r[k][i] for r in rows if r[k][i]]  # noqa: E731
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    print(f"SUMMARY {os.path.basename(os.path.dirname(job))}: success {ok_a}/{len(a)} -> {ok_b}/{len(b)}, verdicts equal {same}/{len(rows)}, "
          f"env step {mean(w('env_ms', 0)):.1f} -> {mean(w('env_ms', 1)):.1f} ms, qa {mean(w('qa_ms', 0)):.1f} -> {mean(w('qa_ms', 1)):.1f} ms, "
          f"assist {mean([r['assist_ms'] for r in rows if r['assist_ms']]):.2f} ms/step, demos with sleeps {sum(1 for r in rows if r['sleeps'])}"
          + (f", trajectories identical {sum(1 for v in agree.values() if 'identical=True' in v)}/{len(agree)}" if traj else ""))
    with open(os.path.join(job, "compare.json"), "w") as f:
        json.dump({"rows": rows, "agreement": agree}, f, indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1], "--traj" in sys.argv[2:])
