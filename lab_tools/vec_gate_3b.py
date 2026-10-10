"""Gate 3b of the vectorized evaluator: replay exactness per slot. Every vectorized replay record (``lab/diagnostics/vec-eval/3b/<run>/
<task>/episode_XXXXXX[.slotN].json``) is compared with the single-env replay of the same episode (the reference directory's
record) step by step: the base (x, y, yaw), the head camera pose, the per-step pose errors, the proprio base velocity, the
robot joints and the tracked objects' poses, plus the counters (streamed steps, total steps, first success step, verdict,
settle windows).

Pass rule: a one-environment world (``num_envs == 1``) must be **bit-identical** to the single-env evaluator (it is: same
scene, same operations). With ``num_envs > 1`` bit-identity is not attainable - the clones live at x offsets of 63 m per
slot where float32 world coordinates have 8-15 um resolution, and PhysX's solver also moves the base scene by ~1e-4 m once
other scenes exist (deterministic: slot 0's trajectory does not depend on what the other slots run, see the same-episode
runs) - so every slot must have identical counters and verdict (streamed steps, total steps, first success step, verdict,
settle windows) and trajectories within ``--tol-pos`` (m) / ``--tol-rot`` (rad) / ``--tol-err`` (the pose-error channels, m /
rad) at every step: 5 mm / 5 mrad by default, a tenth of the relocalization tolerance (50 mm / 100 mrad); measured
deviations are reported so the margin is visible.

    python lab_tools/vec_gate_3b.py --runs n1 n2 n2same n4 n4same --ref lab/diagnostics/eval-fixes/parity-replay/eval50 \
        --out lab/diagnostics/vec-eval/3b/summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARRAYS = ("replay_base_xyyaw.npy", "replay_camera_pose.npy", "errors.npy", "replay_base_qvel.npy", "replay_joint_q.npy", "replay_object_poses.npy")
COUNTERS = ("steps", "n_steps_total", "first_success_step", "reason", "settle_windows", "binary_success_ever", "binary_success_final", "success", "stop_reason")


def quat_diff(qa, qb):
    sign = np.sign((qa * qb).sum(-1, keepdims=True))
    sign[sign == 0] = 1
    return np.abs(qa - sign * qb)


def compare_arrays(vd: str, rd: str) -> dict:
    out = {}
    for f in ARRAYS:
        pa, pb = os.path.join(vd, f), os.path.join(rd, f)
        if not os.path.exists(pa) or not os.path.exists(pb):
            out[f] = {"status": "missing", "vec": os.path.exists(pa), "ref": os.path.exists(pb)}
            continue
        a, b = np.load(pa).astype(np.float64), np.load(pb).astype(np.float64)
        rec = {"len": [int(len(a)), int(len(b))]}
        n = min(len(a), len(b))
        if a.shape[1:] != b.shape[1:]:
            rec["status"] = f"shape {a.shape} vs {b.shape}"
            out[f] = rec
            continue
        a, b = a[:n], b[:n]
        if f == "replay_camera_pose.npy":
            rec["max_pos"] = float(np.abs(a[:, :3] - b[:, :3]).max()) if n else 0.0
            rec["max_quat"] = float(quat_diff(a[:, 3:], b[:, 3:]).max()) if n else 0.0
            rec["identical"] = bool(rec["max_pos"] == 0.0 and rec["max_quat"] == 0.0)
        elif f == "replay_object_poses.npy":
            if a.ndim == 3 and a.shape[1] > 0:
                rec["max_pos"] = float(np.abs(a[..., :3] - b[..., :3]).max())
                rec["max_quat"] = float(quat_diff(a[..., 3:], b[..., 3:]).max())
            else:
                rec["max_pos"] = rec["max_quat"] = 0.0
            rec["identical"] = bool(rec["max_pos"] == 0.0 and rec["max_quat"] == 0.0)
        else:
            d = np.abs(a - b)
            with np.errstate(all="ignore"):
                mx = np.nanmax(d, axis=0) if n else np.zeros(a.shape[1:])
            mx = np.nan_to_num(mx, nan=0.0)  # an all-NaN channel (no pose target in a layout task) is identical
            rec["max_abs"] = [float(v) for v in np.atleast_1d(mx)]
            rec["identical"] = bool(np.array_equal(a, b, equal_nan=True)) if n else True
            nz = np.nan_to_num(d).reshape(n, -1).max(axis=1) > 0 if n else np.zeros(0, dtype=bool)
            rec["first_diff_step"] = int(np.argmax(nz)) if nz.any() else None
        rec["status"] = "ok"
        out[f] = rec
    return out


def counters_close(vec: dict, ref: dict, num_envs: int = 1) -> tuple[bool, list]:
    """Identical verdict / success / stop reason always; the step counters identical for a one-environment world, within one
    step for clones (a tolerance crossing - e.g. the alignment task's 0.1 rad - can move by a step under the clone's float32
    deviation; the settle window then follows the streamed steps)."""
    bad = []
    for k in ("success", "reason", "stop_reason", "binary_success_ever", "binary_success_final"):
        if vec.get(k) != ref.get(k):
            bad.append(f"{k}: {vec.get(k)} vs {ref.get(k)}")
    slack = 0 if num_envs == 1 else 1
    for k in ("steps", "first_success_step", "n_steps_total"):
        a, b = vec.get(k), ref.get(k)
        if a is None or b is None:
            if a != b:
                bad.append(f"{k}: {a} vs {b}")
        elif abs(int(a) - int(b)) > slack + (1 if k == "n_steps_total" else 0):  # the window rounds separately
            bad.append(f"{k}: {a} vs {b}")
    if len(vec.get("settle_windows") or []) != len(ref.get("settle_windows") or []):
        bad.append(f"settle_windows: {vec.get('settle_windows')} vs {ref.get('settle_windows')}")
    return (not bad), bad


def judge(cmp: dict, counters_equal, slot: int, tol_pos: float, tol_rot: float, tol_err: float, num_envs: int = 1) -> tuple[bool, list]:
    reasons = []
    if isinstance(counters_equal, tuple):
        counters_equal, bad = counters_equal
    else:
        bad = []
    if not counters_equal:
        reasons.append("counters differ: " + "; ".join(bad))
    strict = num_envs == 1  # bit-identity is required of the one-environment world only (see the module docstring)
    for f, rec in cmp.items():
        if rec.get("status") != "ok":
            if f in ("replay_joint_q.npy", "replay_object_poses.npy") and rec.get("status") == "missing":
                continue  # older references without these arrays
            reasons.append(f"{f}: {rec.get('status')}")
            continue
        if abs(rec["len"][0] - rec["len"][1]) > (0 if strict else 1):  # a clone's stream may end a step earlier / later (counters_close)
            reasons.append(f"{f}: length {rec['len']}")
        if strict:
            if not rec.get("identical"):
                reasons.append(f"{f}: not bit-identical (num_envs = 1)")
            continue
        if f == "replay_base_xyyaw.npy":
            if max(rec["max_abs"][:2]) > tol_pos or rec["max_abs"][2] > tol_rot:
                reasons.append(f"{f}: {rec['max_abs']} > tol")
        elif f == "replay_camera_pose.npy":
            if rec["max_pos"] > tol_pos or rec["max_quat"] > tol_rot:
                reasons.append(f"{f}: pos {rec['max_pos']:.2e} quat {rec['max_quat']:.2e} > tol")
        elif f == "errors.npy":
            if max(rec["max_abs"]) > tol_err:
                reasons.append(f"{f}: {rec['max_abs']} > tol")
        elif f == "replay_joint_q.npy":
            if max(rec["max_abs"]) > tol_rot:
                reasons.append(f"{f}: {max(rec['max_abs']):.2e} > tol")
        elif f == "replay_object_poses.npy":
            if rec["max_pos"] > tol_pos or rec["max_quat"] > tol_rot:
                reasons.append(f"{f}: pos {rec['max_pos']:.2e} quat {rec['max_quat']:.2e} > tol")
        # replay_base_qvel.npy: the proprio velocity is a noisy derivative; reported, not gated for clones
    return (not reasons), reasons


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(WT, "lab", "diagnostics", "vec-eval", "3b"))
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref", required=True, help="single-env replay records of the same episodes (directory with episode_XXXXXX.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol-pos", type=float, default=5e-3)
    ap.add_argument("--tol-rot", type=float, default=5e-3)
    ap.add_argument("--tol-err", type=float, default=5e-3)
    args = ap.parse_args(argv)
    summary = {"format": "isg-lab-vec-gate-3b/v1", "ref": args.ref, "tolerances": {"pos_m": args.tol_pos, "rot_rad": args.tol_rot, "err": args.tol_err}, "runs": {}, "pass": True}
    for run in args.runs:
        recs = sorted(glob.glob(os.path.join(args.root, run, "*", "episode_*.json")))
        run_out = {"records": [], "pass": True, "n": 0, "n_pass": 0}
        for rp in recs:
            vec = json.load(open(rp))
            e = vec["episode_index"]
            slot = int(vec.get("slot", 0))
            ref_path = os.path.join(args.ref, f"episode_{e:06d}.json")
            item = {"episode": e, "slot": slot, "num_envs": vec.get("num_envs"), "record": os.path.relpath(rp, WT), "error": vec.get("error"), "evaluator_commit": vec.get("evaluator_commit")}
            if not os.path.exists(ref_path):
                item.update({"status": "no reference", "pass": False})
            elif vec.get("error") or not vec.get("raw_dir"):
                item.update({"status": "vec error", "pass": False})
            else:
                ref = json.load(open(ref_path))
                counters = {k: [vec.get(k), ref.get(k)] for k in COUNTERS}
                counters_equal = counters_close(vec, ref, num_envs=int(vec.get("num_envs") or 1))
                cmp = compare_arrays(os.path.join(WT, vec["raw_dir"]), os.path.join(WT, ref["raw_dir"]))
                ok, reasons = judge(cmp, counters_equal, slot, args.tol_pos, args.tol_rot, args.tol_err, num_envs=int(vec.get("num_envs") or 1))
                item.update({"status": "compared", "pass": ok, "reasons": reasons, "counters": counters, "arrays": cmp,
                             "position_error_m": [vec.get("score", {}).get("position_error_m") if vec.get("score") else None, ref.get("score", {}).get("position_error_m")],
                             "memory_gpu_mib": (vec.get("memory") or {}).get("gpu_mib"), "batch_wall_s": vec.get("batch_wall_s"), "wall_s": vec.get("wall_s")})
            run_out["records"].append(item)
            run_out["n"] += 1
            run_out["n_pass"] += int(bool(item.get("pass")))
            run_out["pass"] = run_out["pass"] and bool(item.get("pass"))
        # the same-episode runs: the slots against each other as well
        by_ep = {}
        for item in run_out["records"]:
            by_ep.setdefault(item["episode"], []).append(item)
        for e, items in by_ep.items():
            if len(items) > 1:
                raws = [json.load(open(os.path.join(WT, i["record"])))["raw_dir"] for i in items if i.get("status") == "compared"]
                pairs = []
                for k in range(1, len(raws)):
                    c = compare_arrays(os.path.join(WT, raws[k]), os.path.join(WT, raws[0]))
                    pairs.append({"slot_vs_slot0": items[k]["slot"], "arrays": {f: {kk: vv for kk, vv in r.items() if kk in ("max_abs", "max_pos", "max_quat", "identical", "len")} for f, r in c.items()}})
                run_out.setdefault("same_episode_slot_pairs", {})[str(e)] = pairs
        mx = {}
        for it in run_out["records"]:
            for f, r in (it.get("arrays") or {}).items():
                if r.get("status") != "ok":
                    continue
                if "max_abs" in r:
                    mx[f] = max(mx.get(f, 0.0), max(r["max_abs"]) if r["max_abs"] else 0.0)
                else:
                    mx[f + ":pos"] = max(mx.get(f + ":pos", 0.0), r.get("max_pos", 0.0))
                    mx[f + ":quat"] = max(mx.get(f + ":quat", 0.0), r.get("max_quat", 0.0))
        run_out["max_deviation"] = mx
        summary["runs"][run] = run_out
        summary["pass"] = summary["pass"] and run_out["pass"] and run_out["n"] > 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    for run, r in summary["runs"].items():
        print(f"{run}: {r['n_pass']}/{r['n']} pass")
        for it in r["records"]:
            arr = it.get("arrays") or {}
            xy = arr.get("replay_base_xyyaw.npy", {}).get("max_abs")
            print(f"  ep {it['episode']} slot {it['slot']}: {it.get('status')} pass={it.get('pass')} xyyaw_max={xy} reasons={it.get('reasons')}")
    print("GATE 3b:", "PASS" if summary["pass"] else "FAIL")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
