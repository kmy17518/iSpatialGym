#!/usr/bin/env python3
"""merge_vecnext: one instance's episode in two runs, step by step (run_isg vs run_isg_vec, or two runs of one driver).

    driver_agreement.py <run A instance dir> <run B instance dir> [--out agreement.json]

A run_isg instance dir holds demo.hdf5 (data/demo_0/isg/*) + meta.json, a run_isg_vec one trajectory.npz + meta.json; either may be
A or B. Compared per step: joints, base, end effectors, head camera, every tracked object's pose, the task's eval rows; per
episode: verdict, the first / evaluated success steps and the sleep assist's sleeps / wakes. Prints one AGREEMENT line.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROWS = ("joint_positions", "base_pose", "eef_pose_left", "eef_pose_right", "head_camera_pose", "eval/symbolic", "eval/settled",
        "eval/geometric_within", "eval/position_error", "eval/orientation_error")
META = ("success", "reason", "symbolic_first_step", "geometric_first_step", "evaluation_step", "symbolic_success_step",
        "geometric_success_step", "n_steps")


def rows_of(d):
    npz = os.path.join(d, "trajectory.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        return "run_isg_vec", {k.replace("__", "/", 1): z[k] for k in z.files}
    import h5py

    h = glob.glob(os.path.join(d, "*.hdf5"))
    assert h, f"no trajectory.npz or *.hdf5 in {d}"
    out = {}
    with h5py.File(h[0], "r") as f:
        g = f["data"][sorted(f["data"].keys())[0]]["isg"]
        g.visititems(lambda name, ds: out.__setitem__(name, ds[()]) if hasattr(ds, "shape") else None)
    return "run_isg", out


def compare(a, b):
    sa, sb = [int(s) for s in a["step"]], [int(s) for s in b["step"]]
    common = sorted(set(sa) & set(sb))
    ia, ib = {s: i for i, s in enumerate(sa)}, {s: i for i, s in enumerate(sb)}
    ka, kb = [ia[s] for s in common], [ib[s] for s in common]
    out = {"steps_a": len(sa), "steps_b": len(sb), "common_steps": len(common), "rows": {}}
    keys = list(ROWS) + sorted({k for k in a if k.startswith("object_pose/")} | {k for k in b if k.startswith("object_pose/")})
    for k in keys:
        if k not in a or k not in b:
            out["rows"][k] = {"missing_in": "a" if k not in a else "b"}
            continue
        x, y = np.asarray(a[k])[ka].astype(np.float64), np.asarray(b[k])[kb].astype(np.float64)
        same = np.array_equal(x, y, equal_nan=True)
        d = np.abs(x - y)
        d = np.where(np.isnan(x) & np.isnan(y), 0.0, d)
        d = d.reshape(len(common), -1).max(axis=1) if d.ndim > 1 else d
        first = next((common[i] for i, v in enumerate(d) if not v <= 1e-6), None)
        out["rows"][k] = {"identical": bool(same), "max_abs": float(np.nanmax(d)) if len(d) else 0.0, "first_step_over_1e-6": first}
    out["identical"] = all(r.get("identical") for r in out["rows"].values()) and len(sa) == len(sb)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    (da, a), (db, b) = rows_of(args.a), rows_of(args.b)
    ma, mb = json.load(open(os.path.join(args.a, "meta.json"))), json.load(open(os.path.join(args.b, "meta.json")))
    res = {"a": {"dir": args.a, "driver": da}, "b": {"dir": args.b, "driver": db}, **compare(a, b), "meta": {}}
    for k in META:
        res["meta"][k] = {"a": ma.get(k), "b": mb.get(k), "same": ma.get(k) == mb.get(k)}
    sla, slb = ma.get("sleep_assist") or {}, mb.get("sleep_assist") or {}
    res["sleep_assist"] = {"a": {k: sla.get(k) for k in ("sleeps", "wakes", "rollbacks")}, "b": {k: slb.get(k) for k in ("sleeps", "wakes", "rollbacks")},
                           "same_events": (sla.get("sleeps"), sla.get("wakes")) == (slb.get("sleeps"), slb.get("wakes"))}
    res["driver_physics_a"], res["driver_physics_b"] = ma.get("driver_physics"), mb.get("driver_physics")
    worst = max(((k, r.get("max_abs", 0.0)) for k, r in res["rows"].items()), key=lambda t: t[1], default=(None, 0.0))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
    print(f"AGREEMENT identical={res['identical']} steps {res['steps_a']}/{res['steps_b']} worst {worst[0]} {worst[1]:.3g} "
          f"first_div {min([r['first_step_over_1e-6'] for r in res['rows'].values() if r.get('first_step_over_1e-6') is not None], default=None)} "
          f"verdict {ma.get('success')}/{mb.get('success')} eval_step {ma.get('evaluation_step')}/{mb.get('evaluation_step')} "
          f"sleep_events_same={res['sleep_assist']['same_events']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
