"""Gate 3c of the vectorized evaluator: cross-talk between slots.

Three checks on the replay runs of gate 3b (``lab/diagnostics/vec-eval/3b/<run>``):

* neighbour independence: the same episode in the same slot of two worlds whose other slots run different episodes
  (``--neighbours runA:runB``, e.g. n4 vs n4same: slot 0 runs episode 2 next to 3/33/34 and next to 2/2/2) must give
  **bit-identical** trajectories (base, camera, errors, proprio base velocity, joints, objects);
* slot-position independence: the same episode in every slot of one world (``--same runX``) must give identical counters
  and trajectories within the clone tolerances (float32 world coordinates at the clone offsets);
* idle slots / free robots: a batch with fewer episodes than slots (``--idle runY:refdir``, e.g. the alignment hinge task in
  slots 0-1 with idle robots in slots 2-3) must match the single-env reference like a full batch (counters identical,
  trajectories within tolerance), i.e. an idle neighbour - a robot merely holding its joints - changes nothing.

    python lab_tools/vec_gate_3c.py --neighbours n4:n4same --neighbours n4warm:n4samewarm --same n4same --same n4samewarm \
        --idle align-n4:lab/diagnostics/vec-eval/3b/ref-tasks/alignment-axial-board_game-breakfast_table --out lab/diagnostics/vec-eval/3c/summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)

from lab_tools.vec_gate_3b import COUNTERS, compare_arrays, counters_close, judge  # noqa: E402

ROOT = os.path.join(WT, "lab", "diagnostics", "vec-eval", "3b")


def records(run: str):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, run, "*", "episode_*.json"))):
        r = json.load(open(p))
        if r.get("raw_dir") and not r.get("error"):
            out.append(r)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neighbours", action="append", default=[], help="runA:runB - same episode & slot, different neighbours")
    ap.add_argument("--same", action="append", default=[], help="a same-episode-all-slots run")
    ap.add_argument("--idle", action="append", default=[], help="run:refdir - a partial batch against the single-env reference")
    ap.add_argument("--tol-pos", type=float, default=5e-3)
    ap.add_argument("--tol-rot", type=float, default=5e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = {"format": "isg-lab-vec-gate-3c/v1", "neighbours": [], "same_episode": [], "idle": [], "pass": True}
    for pair in args.neighbours:
        a, b = pair.split(":")
        ra = {(r["episode_index"], int(r.get("slot", 0))): r for r in records(a)}
        rb = {(r["episode_index"], int(r.get("slot", 0))): r for r in records(b)}
        for key in sorted(set(ra) & set(rb)):
            cmp = compare_arrays(os.path.join(WT, ra[key]["raw_dir"]), os.path.join(WT, rb[key]["raw_dir"]))
            identical = all(rec.get("identical") for rec in cmp.values() if rec.get("status") == "ok")
            counters = all(ra[key].get(k) == rb[key].get(k) for k in COUNTERS)
            item = {"runs": [a, b], "episode": key[0], "slot": key[1], "bit_identical": identical, "counters_identical": counters,
                    "neighbours": [sorted(r["episode_index"] for r in records(a) if int(r.get("slot", 0)) != key[1]), sorted(r["episode_index"] for r in records(b) if int(r.get("slot", 0)) != key[1])],
                    "arrays": {f: {k: v for k, v in rec.items() if k in ("identical", "max_abs", "max_pos", "max_quat")} for f, rec in cmp.items() if rec.get("status") == "ok"}}
            item["pass"] = identical and counters
            out["neighbours"].append(item)
            out["pass"] = out["pass"] and item["pass"]
    for run in args.same:
        recs = sorted(records(run), key=lambda r: int(r.get("slot", 0)))
        if not recs:
            continue
        base = recs[0]
        for r in recs[1:]:
            cmp = compare_arrays(os.path.join(WT, r["raw_dir"]), os.path.join(WT, base["raw_dir"]))
            counters = counters_close(r, base, num_envs=int(r.get("num_envs") or 2))
            ok, reasons = judge(cmp, counters, int(r.get("slot", 0)), args.tol_pos, args.tol_rot, args.tol_pos, num_envs=int(r.get("num_envs") or 2))
            item = {"run": run, "episode": r["episode_index"], "slot": int(r.get("slot", 0)), "vs_slot": int(base.get("slot", 0)), "counters_identical": all(r.get(k) == base.get(k) for k in COUNTERS), "pass": ok, "reasons": reasons,
                    "arrays": {f: {k: v for k, v in rec.items() if k in ("identical", "max_abs", "max_pos", "max_quat")} for f, rec in cmp.items() if rec.get("status") == "ok"}}
            out["same_episode"].append(item)
            out["pass"] = out["pass"] and ok
    for spec in args.idle:
        run, refdir = spec.split(":", 1)
        refdir = refdir if os.path.isabs(refdir) else os.path.join(WT, refdir)
        for r in records(run):
            e = r["episode_index"]
            refp = os.path.join(refdir, f"episode_{e:06d}.json")
            if not os.path.exists(refp):
                out["idle"].append({"run": run, "episode": e, "slot": r.get("slot"), "pass": False, "reasons": ["no reference"]})
                out["pass"] = False
                continue
            ref = json.load(open(refp))
            cmp = compare_arrays(os.path.join(WT, r["raw_dir"]), os.path.join(WT, ref["raw_dir"]))
            counters = counters_close(r, ref, num_envs=int(r.get("num_envs") or 2))
            ok, reasons = judge(cmp, counters, int(r.get("slot", 0)), args.tol_pos, args.tol_rot, args.tol_pos, num_envs=int(r.get("num_envs") or 2))
            item = {"run": run, "episode": e, "slot": int(r.get("slot", 0)), "num_envs": r.get("num_envs"), "idle_slots": int(r.get("num_envs") or 0) - len(records(run)), "counters_identical": all(r.get(k) == ref.get(k) for k in COUNTERS),
                    "pass": ok, "reasons": reasons, "arrays": {f: {k: v for k, v in rec.items() if k in ("identical", "max_abs", "max_pos", "max_quat")} for f, rec in cmp.items() if rec.get("status") == "ok"}}
            out["idle"].append(item)
            out["pass"] = out["pass"] and ok
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for it in out["neighbours"]:
        print(f"neighbours {it['runs']} ep {it['episode']} slot {it['slot']}: bit-identical={it['bit_identical']} counters={it['counters_identical']} neighbours={it['neighbours']}")
    for it in out["same_episode"]:
        print(f"same {it['run']} ep {it['episode']} slot {it['slot']} vs {it['vs_slot']}: pass={it['pass']} {it['reasons']}")
    for it in out["idle"]:
        print(f"idle {it['run']} ep {it['episode']} slot {it['slot']} (idle slots {it.get('idle_slots')}): pass={it['pass']} {it['reasons']}")
    print("GATE 3c:", "PASS" if out["pass"] else "FAIL")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
