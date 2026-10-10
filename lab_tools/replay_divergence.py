"""Item 1b: where does an open-loop replay depart from the recorded demo?

The demo's ``observation.state`` (61-D proprioception: arm / gripper / trunk joint positions, base velocity) is compared step
by step with the replay's recorded joints (``replay_joint_q.npy``, 28 joints before every policy call; index t = the pose the
demo's ``observation.state[t]`` was taken at) and the replay's tracked object poses (``replay_object_poses.npy``). Reports, per
joint group, the first step at which the replay departs by more than a threshold, the step-0 differences (loader / restore),
and the largest departures over the stream; writes a JSON next to the record.

    python lab_tools/replay_divergence.py --record lab/diagnostics/vec-eval/3b/ref-tasks/<task>/episode_000100.json --dataset datasets/isg-init-eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)

from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from lab_tools.replay_episodes import load_episode_arrays  # noqa: E402

J = {n: i for i, n in enumerate(M.ALL_JOINTS)}
GROUPS = {
    "trunk": ([J[n] for n in M.TORSO_JOINTS], slice(53, 57)),
    "arm_left": ([J[n] for n in M.ARM_JOINTS["left"]], slice(3, 10)),
    "arm_right": ([J[n] for n in M.ARM_JOINTS["right"]], slice(28, 35)),
    "gripper_left": ([J[n] for n in M.FINGER_JOINTS["left"]], slice(24, 26)),
    "gripper_right": ([J[n] for n in M.FINGER_JOINTS["right"]], slice(49, 51)),
}


def first_exceed(d: np.ndarray, thr: float):
    idx = np.where(d > thr)[0]
    return int(idx[0]) if len(idx) else None


def analyze(record_path: str, dataset: str, thr_rad: float = 0.01, thr_gripper: float = 0.002) -> dict:
    rec = json.load(open(record_path))
    e = int(rec["episode_index"])
    raw = os.path.join(WT, rec["raw_dir"]) if not os.path.isabs(rec["raw_dir"]) else rec["raw_dir"]
    actions, states, length = load_episode_arrays(dataset, e)
    jq = np.load(os.path.join(raw, "replay_joint_q.npy")).astype(np.float64)  # (T_r, 28), t = before policy call t
    n = min(len(jq), len(states))
    out = {"episode": e, "record": os.path.relpath(record_path, WT), "success": rec.get("success"), "reason": rec.get("reason"), "steps_streamed": rec.get("steps"),
           "demo_length": int(length), "compared_steps": int(n), "groups": {}}
    for g, (idx, sl) in GROUPS.items():
        rep = jq[:n, idx]
        demo = states[:n, sl].astype(np.float64)
        if rep.shape != demo.shape:
            out["groups"][g] = {"error": f"shape {rep.shape} vs {demo.shape}"}
            continue
        d = np.abs(rep - demo)
        dmax = d.max(axis=1)
        thr = thr_gripper if g.startswith("gripper") else thr_rad
        first = first_exceed(dmax, thr)
        out["groups"][g] = {
            "step0_max_abs": float(dmax[0]), "step0_per_joint": [float(v) for v in d[0]],
            "first_step_over_threshold": first, "threshold": thr,
            "max_abs": float(dmax.max()), "max_abs_step": int(dmax.argmax()),
            "mean_abs_first_100": float(dmax[:100].mean()), "mean_abs_all": float(dmax.mean()),
            "profile": {str(t): float(dmax[t]) for t in [0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900] if t < n},
        }
    # the commanded (action) vs the measured demo state: how closely did the demo track its own targets, and the replay?
    try:
        demo_track = {}
        rep_track = {}
        amap = {"trunk": (slice(3, 7), slice(53, 57)), "arm_left": (slice(7, 14), slice(3, 10)), "arm_right": (slice(15, 22), slice(28, 35))}
        for g, (asl, ssl) in amap.items():
            a = actions[: n - 1, asl].astype(np.float64)
            demo_track[g] = float(np.abs(states[1:n, ssl] - a).mean())
            rep_track[g] = float(np.abs(jq[1:n, GROUPS[g][0]] - a).mean())
        out["tracking_mean_abs_target_minus_next_state"] = {"demo": demo_track, "replay": rep_track}
    except Exception as ex:  # noqa: BLE001
        out["tracking_error"] = repr(ex)
    # base velocity: the demo's proprio base_qvel vs the replay's (both proprio samples)
    try:
        bq = np.load(os.path.join(raw, "replay_base_qvel.npy")).astype(np.float64)
        m = min(len(bq), n)
        d = np.abs(bq[:m] - states[:m, 0:3])
        out["base_qvel"] = {"max_abs": [float(v) for v in d.max(axis=0)], "mean_abs": [float(v) for v in d.mean(axis=0)], "first_step_over_0.05": first_exceed(d.max(axis=1), 0.05)}
    except Exception as ex:  # noqa: BLE001
        out["base_qvel"] = {"error": repr(ex)}
    # tracked objects (replay only): pose change over the stream (the demo has no object record)
    try:
        op = np.load(os.path.join(raw, "replay_object_poses.npy")).astype(np.float64)
        names = json.load(open(os.path.join(raw, "replay_objects.json")))
        objs = {}
        for k, name in enumerate(names):
            p = op[:, k, :3]
            objs[name] = {"pos0": [float(v) for v in p[0]], "max_disp_from_start_m": float(np.linalg.norm(p - p[0], axis=1).max()),
                          "first_step_moved_1mm": first_exceed(np.linalg.norm(p - p[0], axis=1), 1e-3)}
        out["objects"] = objs
    except Exception as ex:  # noqa: BLE001
        out["objects"] = {"error": repr(ex)}
    meta_p = os.path.join(raw, "meta.json")
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        out["loader"] = {k: meta.get("loader", {}).get(k) for k in ("constraints", "grasps", "controller_targets_restored", "after_reset", "base_link_mass_kg")}
        out["errors_final_step"] = meta.get("errors_final_step")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", nargs="+", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None, help="JSON summary (default: next to the first record, divergence_<ep>.json)")
    args = ap.parse_args(argv)
    results = [analyze(r, args.dataset) for r in args.record]
    out = args.out or os.path.join(os.path.dirname(args.record[0]), f"divergence_{results[0]['episode']:06d}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    for r in results:
        print(f"episode {r['episode']} success={r['success']} reason={r['reason']} streamed={r['steps_streamed']} demo_length={r['demo_length']}")
        for g, v in r["groups"].items():
            if "error" in v:
                print(f"  {g}: {v['error']}")
                continue
            print(f"  {g}: step0 {v['step0_max_abs']:.4f}  first>{v['threshold']} at step {v['first_step_over_threshold']}  max {v['max_abs']:.4f} @ {v['max_abs_step']}  mean(first100) {v['mean_abs_first_100']:.4f} mean(all) {v['mean_abs_all']:.4f}")
        print("  tracking (|target - next state| mean):", r.get("tracking_mean_abs_target_minus_next_state"))
        print("  base_qvel:", r.get("base_qvel"))
        print("  objects:", r.get("objects"))
    print("written", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
