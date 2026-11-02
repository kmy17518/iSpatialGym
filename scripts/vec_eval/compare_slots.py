"""Compare the slot episodes of a ``run_isg_vec --same-instance-all-slots`` run (and optionally a reference run of the same
instance at N=1): verdicts, success steps, per-step trajectories (base pose, joints, eef poses, tracked object poses) and the
task's error metric. Prints a table and writes ``<run>/compare.json``.

    python scripts/vec_eval/compare_slots.py <run dir> [--ref <N=1 run dir>]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np



def _first_step(meta, kind):
    """The step ``kind``'s success was FIRST reached (verification compares first steps: the evaluated step, chosen within the
    settle window by the lowest error since 2026-11-01, can move far on a tiny error difference). Metas written before that
    change carry only ``*_success_step``, which was the first step."""
    key = f"{kind}_first_step"
    return meta.get(key) if key in meta else meta.get(f"{kind}_success_step")

def load(ep_dir):
    meta = json.load(open(os.path.join(ep_dir, "meta.json")))
    traj = np.load(os.path.join(ep_dir, "trajectory.npz")) if os.path.exists(os.path.join(ep_dir, "trajectory.npz")) else None
    return meta, traj


def quat_angle(q1, q2):
    d = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(d)


def diff(a, b):
    """Max deviation between two trajectory sets over their common length: position (m) and rotation (rad) per key."""
    out = {}
    n = min(len(a["step"]), len(b["step"]))
    for k in a.files:
        if k not in b.files or k in ("step", "phase") or k.startswith("eval"):
            continue
        x, y = a[k][:n], b[k][:n]
        if x.ndim == 2 and x.shape[1] == 7:
            out[k] = {"pos_max_m": float(np.max(np.linalg.norm(x[:, :3] - y[:, :3], axis=1))), "rot_max_rad": float(np.max(quat_angle(x[:, 3:], y[:, 3:])))}
        elif x.ndim == 2 and k == "joint_positions":
            # revolute joints are reported in (-pi, pi]: a joint crossing +-180 deg one step apart in two runs (the base yaw
            # of camera_relocalization-standard, 2026-10-23: -3.1428 vs +3.1403) is a 0.0003 rad difference, not 6.28; wrapping
            # a prismatic joint is harmless (a > pi m gap already fails the 0.02 m pose checks)
            d = np.abs(x - y)
            d = np.minimum(d, 2.0 * np.pi - d)
            out[k] = {"max_abs": float(np.max(d)), "max_abs_unwrapped": float(np.max(np.abs(x - y)))}
        elif x.ndim == 2:
            out[k] = {"max_abs": float(np.max(np.abs(x - y)))}
    e1, e2 = a["eval__position_error"][:n], b["eval__position_error"][:n]
    m = np.isfinite(e1) & np.isfinite(e2)
    out["eval_position_error"] = {"max_abs": float(np.max(np.abs(e1[m] - e2[m]))) if m.any() else None}
    out["common_steps"] = int(n)
    out["length"] = [int(len(a["step"])), int(len(b["step"]))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref", default=None, help="an N=1 run of the same instance(s)")
    args = ap.parse_args()
    eps = sorted(d for d in glob.glob(os.path.join(args.run, "*")) if os.path.isdir(d) and os.path.exists(os.path.join(d, "meta.json")))
    by_inst = {}
    for d in eps:
        name = os.path.basename(d)
        base = name.split(".slot")[0] if ".slot" in name else name
        by_inst.setdefault(base, []).append(d)
    report = {}
    for base, dirs in by_inst.items():
        rows = []
        metas = {}
        trajs = {}
        for d in dirs:
            m, t = load(d)
            metas[d] = m
            trajs[d] = t
            rows.append({"slot": m.get("vec", {}).get("slot"), "status": m.get("status"), "success": m.get("success"), "reason": m.get("reason"),
                         "wall_clock_exceeded": bool(m.get("wall_clock_exceeded")), "max_steps": m.get("max_steps"),
                         "n_steps": m.get("n_steps"), "geo_step": _first_step(m, "geometric"), "sym_step": _first_step(m, "symbolic"),
                         "geo_eval_step": m.get("geometric_success_step"), "sym_eval_step": m.get("symbolic_success_step"),
                         "pos_err_final": ((m.get("errors_final_step") or {}).get("pose_errors") or {}) and next(iter(((m.get("errors_final_step") or {}).get("pose_errors") or {}).values())).get("position_error"),
                         "wall_s": m.get("wall_time_s")})
        rows.sort(key=lambda r: (r["slot"] is None, r["slot"]))
        ref_dir = None
        if args.ref:
            cands = [d for d in glob.glob(os.path.join(args.ref, base + "*")) if os.path.isdir(d)]
            ref_dir = cands[0] if cands else None
        anchor = ref_dir or dirs[0]
        a_meta, a_traj = (load(anchor) if ref_dir else (metas[anchor], trajs[anchor]))
        devs = {}
        for d in dirs:
            if trajs[d] is None or a_traj is None:
                continue
            devs[os.path.basename(d)] = diff(a_traj, trajs[d])
        verdict_set = {(r["success"], r["geo_step"], r["n_steps"]) for r in rows}
        report[base] = {"anchor": "ref N=1" if ref_dir else os.path.basename(anchor), "anchor_success": a_meta.get("success"), "anchor_geo_step": _first_step(a_meta, "geometric"),
                        "anchor_geo_eval_step": a_meta.get("geometric_success_step"),
                        "anchor_n_steps": a_meta.get("n_steps"), "anchor_wall_clock_exceeded": bool(a_meta.get("wall_clock_exceeded")),
                        "rows": rows, "identical_verdicts": len(verdict_set) == 1 and (not ref_dir or (a_meta.get("success"), _first_step(a_meta, "geometric"), a_meta.get("n_steps")) in verdict_set),
                        "deviations": devs}
        print(f"== {base}  anchor={report[base]['anchor']} success={a_meta.get('success')} geo_step={_first_step(a_meta, 'geometric')} n_steps={a_meta.get('n_steps')}")
        for r in rows:
            print(f"  slot {r['slot']}: success={r['success']} reason={r['reason']} n_steps={r['n_steps']} geo_step={r['geo_step']} sym_step={r['sym_step']} pos_err_final={r['pos_err_final']} wall={r['wall_s']}")
        for name, dv in devs.items():
            worst_pos = max((v.get("pos_max_m") or 0.0) for v in dv.values() if isinstance(v, dict))
            worst_rot = max((v.get("rot_max_rad") or 0.0) for v in dv.values() if isinstance(v, dict))
            jm = (dv.get("joint_positions") or {}).get("max_abs")
            print(f"  dev {name}: max pos {worst_pos:.2e} m, max rot {worst_rot:.2e} rad, joints {jm}, err metric {dv['eval_position_error']['max_abs']}, steps {dv['length']}")
        print(f"  identical verdicts: {report[base]['identical_verdicts']}")
    with open(os.path.join(args.run, "compare.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)


if __name__ == "__main__":
    main()
