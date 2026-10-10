"""Gate 3e of the vectorized evaluator: closed-loop distribution parity.

Closed-loop ACT rollouts are chaotic (the same episode on the same path flips between success and failure from run to run:
lab/diagnostics/eval-fixes/variation), so parity is judged on distributions, not per-episode equality. For every setting
(default execution / temporal aggregation) the vectorized sets (N = 4, N = 1) are compared with the single-env reference
sets of the same checkpoint and episodes:

* success counts against the binomial 95 % interval of the pooled single-env success rate (and vice versa);
* per-episode success flips (vec vs single-env) against the single-env run-to-run flip rate measured on the repeated
  episodes; geodesic / position / orientation error distributions (median, mean, quartiles) side by side;
* streamed steps, first success steps, stop reasons, verdict categories, evaluator errors, memory, wall time per episode.

    python lab_tools/vec_gate_3e.py --out lab/diagnostics/vec-eval/3e/summary.json \
        --setting default --vec default-n4=lab/diagnostics/vec-eval/3e/default-n4/camera_relocalization-standard \
                          --vec default-n4-rep2=... --vec default-n1=... \
                          --ref production=lab/tierB-train/r1-early-copy05-st0-s0/step_00100000/camera_relocalization-standard \
                          --ref parity-default=lab/diagnostics/eval-fixes/parity-default/camera_relocalization-standard ...
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_set(d: str) -> Dict[int, dict]:
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "episode_*.json"))):
        try:
            r = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        if r.get("episode_index") is None:
            continue
        out[int(r["episode_index"])] = r
    return out


def stats(recs: Dict[int, dict]) -> dict:
    vals = {"geodesic_distance_m": [], "position_error_m": [], "orientation_error_deg": [], "steps": [], "first_success_step": [], "wall_s": [], "gpu_mib": []}
    succ, errors, reasons, stops = 0, 0, {}, {}
    for r in recs.values():
        if r.get("error"):
            errors += 1
            continue
        succ += int(bool(r.get("success")))
        sc = r.get("score") or {}
        for k in ("geodesic_distance_m", "position_error_m", "orientation_error_deg"):
            if sc.get(k) is not None:
                vals[k].append(float(sc[k]))
        if r.get("steps") is not None:
            vals["steps"].append(float(r["steps"]))
        if r.get("first_success_step") is not None and r["first_success_step"] >= 0:
            vals["first_success_step"].append(float(r["first_success_step"]))
        if r.get("wall_s") is not None:
            vals["wall_s"].append(float(r["wall_s"]))
        if (r.get("memory") or {}).get("gpu_mib"):
            vals["gpu_mib"].append(float(r["memory"]["gpu_mib"]))
        reasons[str(r.get("reason"))] = reasons.get(str(r.get("reason")), 0) + 1
        stops[str(r.get("stop_reason"))] = stops.get(str(r.get("stop_reason")), 0) + 1
    n = len(recs) - errors

    def q(v):
        if not v:
            return None
        a = np.asarray(v)
        return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)), "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75)), "min": float(a.min()), "max": float(a.max())}

    return {"n": n, "errors": errors, "successes": succ, "success_rate": (succ / n) if n else None, "reasons": reasons, "stop_reasons": stops, **{k: q(v) for k, v in vals.items()}}


def binomial_interval(p: float, n: int, z: float = 1.96) -> List[int]:
    """Approximate 95 % interval of successes in n trials at rate p (normal with continuity; clipped)."""
    mu, sd = n * p, math.sqrt(max(n * p * (1 - p), 1e-9))
    return [int(max(0, math.floor(mu - z * sd - 0.5))), int(min(n, math.ceil(mu + z * sd + 0.5)))]


def flips(a: Dict[int, dict], b: Dict[int, dict]) -> dict:
    common = [e for e in a if e in b and not a[e].get("error") and not b[e].get("error")]
    if not common:
        return {"n_common": 0}
    fl = [e for e in common if bool(a[e].get("success")) != bool(b[e].get("success"))]
    geo = []
    for e in common:
        ga, gb = (a[e].get("score") or {}).get("geodesic_distance_m"), (b[e].get("score") or {}).get("geodesic_distance_m")
        if ga is not None and gb is not None:
            geo.append(abs(float(ga) - float(gb)))
    return {"n_common": len(common), "n_flips": len(fl), "flip_rate": len(fl) / len(common), "flipped_episodes": fl,
            "abs_geodesic_diff_median": float(np.median(geo)) if geo else None, "abs_geodesic_diff_p90": float(np.percentile(geo, 90)) if geo else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--vec", action="append", default=[], help="name=dir of a vectorized set")
    ap.add_argument("--ref", action="append", default=[], help="name=dir of a single-env reference set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-flip-excess", type=float, default=0.15, help="vec-vs-ref per-episode flip rate may exceed the single-env run-to-run flip rate by at most this")
    args = ap.parse_args(argv)
    vec = {k: load_set(os.path.join(WT, v) if not os.path.isabs(v) else v) for k, v in (x.split("=", 1) for x in args.vec)}
    ref = {k: load_set(os.path.join(WT, v) if not os.path.isabs(v) else v) for k, v in (x.split("=", 1) for x in args.ref)}
    out = {"format": "isg-lab-vec-gate-3e/v1", "setting": args.setting, "sets": {}, "pairs": {}, "pass": True, "reasons": []}
    for name, recs in {**{f"vec:{k}": v for k, v in vec.items()}, **{f"ref:{k}": v for k, v in ref.items()}}.items():
        out["sets"][name] = stats(recs)
        out["sets"][name]["episodes"] = sorted(recs)
    # the single-env envelope: flips between reference sets on their common episodes
    ref_names = list(ref)
    env_flips = []
    for i in range(len(ref_names)):
        for j in range(i + 1, len(ref_names)):
            f = flips(ref[ref_names[i]], ref[ref_names[j]])
            if f.get("n_common"):
                out["pairs"][f"ref:{ref_names[i]} vs ref:{ref_names[j]}"] = f
                env_flips.append((f["n_flips"], f["n_common"]))
    ref_flip_rate = (sum(a for a, _ in env_flips) / sum(b for _, b in env_flips)) if env_flips and sum(b for _, b in env_flips) else None
    out["single_env_run_to_run_flip_rate"] = ref_flip_rate
    # the pooled single-env success rate on the episodes the vectorized sets cover
    all_vec_eps = set().union(*[set(v) for v in vec.values()]) if vec else set()
    pooled_s, pooled_n = 0, 0
    for recs in ref.values():
        for e, r in recs.items():
            if e in all_vec_eps and not r.get("error"):
                pooled_n += 1
                pooled_s += int(bool(r.get("success")))
    pooled_rate = (pooled_s / pooled_n) if pooled_n else None
    out["single_env_pooled"] = {"successes": pooled_s, "n": pooled_n, "rate": pooled_rate}
    for vname, vrecs in vec.items():
        st = out["sets"][f"vec:{vname}"]
        if st["errors"]:
            out["pass"] = False
            out["reasons"].append(f"{vname}: {st['errors']} evaluator errors")
        if pooled_rate is not None and st["n"]:
            lo, hi = binomial_interval(pooled_rate, st["n"])
            st["expected_successes_95"] = [lo, hi]
            if not (lo <= st["successes"] <= hi):
                out["pass"] = False
                out["reasons"].append(f"{vname}: {st['successes']}/{st['n']} successes outside the single-env 95 % interval [{lo}, {hi}] at rate {pooled_rate:.3f}")
        for rname, rrecs in ref.items():
            f = flips(vrecs, rrecs)
            if f.get("n_common"):
                out["pairs"][f"vec:{vname} vs ref:{rname}"] = f
                if ref_flip_rate is not None and f["n_common"] >= 5 and f["flip_rate"] > ref_flip_rate + args.max_flip_excess:
                    out["pass"] = False
                    out["reasons"].append(f"{vname} vs {rname}: per-episode flip rate {f['flip_rate']:.2f} > single-env run-to-run {ref_flip_rate:.2f} + {args.max_flip_excess}")
        for v2, r2 in vec.items():
            if v2 > vname:
                f = flips(vrecs, r2)
                if f.get("n_common"):
                    out["pairs"][f"vec:{vname} vs vec:{v2}"] = f
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for name, st in out["sets"].items():
        g = st.get("geodesic_distance_m") or {}
        print(f"{name}: n={st['n']} errors={st['errors']} successes={st['successes']} rate={st['success_rate'] and round(st['success_rate'], 3)} expected95={st.get('expected_successes_95')} "
              f"geodesic median={g.get('median') and round(g['median'], 3)} mean={g.get('mean') and round(g['mean'], 3)} steps mean={(st.get('steps') or {}).get('mean') and round(st['steps']['mean'])} "
              f"wall/ep={(st.get('wall_s') or {}).get('mean') and round(st['wall_s']['mean'])}s")
    for k, f in out["pairs"].items():
        print(f"  {k}: common={f['n_common']} flips={f.get('n_flips')} rate={f.get('flip_rate') and round(f['flip_rate'], 2)} |dgeo| median={f.get('abs_geodesic_diff_median') and round(f['abs_geodesic_diff_median'], 3)}")
    print("single-env run-to-run flip rate:", ref_flip_rate)
    print("GATE 3e (" + args.setting + "):", "PASS" if out["pass"] else "FAIL", out["reasons"])
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
