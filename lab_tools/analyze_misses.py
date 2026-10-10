#!/usr/bin/env python3
"""Relocalization misses: where does the final error come from (user, 2026-10-07 18:30Z, item 4), and stalls (item 6).

For every relocalization episode record under lab/ (tierB, tierB-train, diagnostics/exec, diagnostics/replay):
* position error split into horizontal (camera xy vs goal xy, ``score.euclidean_xy_m`` / ``position_error_xy_m``) and vertical
  (``position_error_z_m`` when logged, else sqrt(total^2 - xy^2), sign unknown);
* orientation error split into yaw (``heading_error_deg`` = yaw of goal^-1 * current) and the pitch/roll remainder
  (``orientation_error_ypr_deg`` when logged, else sqrt(total^2 - yaw^2));
* the last commanded trunk targets (actions.npy, trunk slots of the 23-D action) vs the instance's goal trunk configuration
  (sidecar stance 'goal' -> torso_q), and the final measured trunk / base vs goal when the record logged them
  (``score.base_trunk_vs_goal``, episodes run after 2026-10-07 18:40Z);
* closest approach (``score.closest_approach`` from errors.npy, new episodes only);
* stall: from actions.npy, the first step after which the base command stays below ``--base-eps`` and every joint target moves
  less than ``--joint-eps`` rad per step until the end of the stream.

Writes lab/diagnostics/misses.json (per-episode rows + aggregates) and prints a markdown table.

    python lab_tools/analyze_misses.py [--base-eps 0.02] [--joint-eps 0.002] [--stall-steps 150]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
TASK = "camera_relocalization-standard"
TRUNK = slice(3, 7)  # 23-D action: base 0:3, trunk 3:7, left arm 7:14, left gripper 14, right arm 15:22, right gripper 22


def torso_q(translate, tilt):
    from ispatialgym.robot.conventions import torso_q as tq

    return tq(float(translate), float(tilt)).numpy()


def goal_stance(tro: str):
    p = tro.replace("-tro_state.json", ".oracle.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))["stances"]["goal"]
    except Exception:  # noqa: BLE001
        return None


def stall_step(actions: np.ndarray, base_eps: float, joint_eps: float, window: int = 150):
    """Parked robot: the first step s such that every trailing window of ``window`` steps ending at t >= s + window has a mean
    |base command| (max over vx, vy, wz; [-1, 1] scale) below ``base_eps`` and a joint-target excursion (max - min over the
    window, every trunk / arm / gripper slot) below ``joint_eps`` rad; None when the robot was still moving at the end. The
    16-step re-plan sawtooth of chunked execution keeps per-step target changes non-zero, hence window statistics."""
    if actions is None or len(actions) < window + 1:
        return None
    T = len(actions)
    base = np.abs(actions[:, 0:3]).max(axis=1)
    cs = np.r_[0.0, np.cumsum(base)]
    quiet = np.zeros(T, dtype=bool)
    for t in range(window, T):
        if (cs[t + 1] - cs[t + 1 - window]) / window >= base_eps:
            continue
        w = actions[t + 1 - window:t + 1, 3:]
        if float((w.max(0) - w.min(0)).max()) < joint_eps:
            quiet[t] = True
    if not quiet[-1]:
        return None
    t = T - 1
    while t > window and quiet[t - 1]:
        t -= 1
    return int(t - window + 1)  # the window ending at t starts here: the robot has been parked since this step


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-eps", type=float, default=0.02, help="base command (|a[0:3]|, [-1, 1] scale) below this = stopped (0.02 x 0.75 m/s = 1.5 cm/s)")
    ap.add_argument("--joint-eps", type=float, default=0.1, help="joint-target excursion (max - min) over the window below this = parked (rad)")
    ap.add_argument("--stall-steps", type=int, default=150, help="window of the parked test and the proposed rule: end the stream once the robot has been parked for this many steps (150 = 5 s)")
    ap.add_argument("--out", default=os.path.join(WT, "lab", "diagnostics", "misses.json"))
    args = ap.parse_args(argv)

    paths = sorted(set(glob.glob(os.path.join(WT, "lab", "tierB*", "*", "step_*", TASK, "episode_*.json")) +
                       glob.glob(os.path.join(WT, "lab", "diagnostics", "exec", "*", "*", "step_*", TASK, "episode_*.json"))))
    inst_maps = {}
    for sp, f in (("eval50", "instances.json"), ("train", "instances-train.json")):
        p = os.path.join(WT, "lab", f)
        if os.path.exists(p):
            d = json.load(open(p))
            inst_maps[sp] = d.get("episodes", d)
    rows = []
    for p in paths:
        r = json.load(open(p))
        if r.get("error") or r.get("task") != TASK:
            continue
        sc = r.get("score") or {}
        sp = r.get("split") or "eval50"
        row = {"run": r["run"], "step": r["step"], "split": sp, "tag": r.get("tag"), "episode": r["episode_index"], "instance": r.get("instance"), "success": r.get("success"),
               "reason": r.get("reason"), "steps": r.get("steps"), "horizon": r.get("horizon"), "binary_success_ever": r.get("binary_success_ever"),
               "position_error_m": sc.get("position_error_m"), "orientation_error_deg": sc.get("orientation_error_deg"), "geodesic_distance_m": sc.get("geodesic_distance_m")}
        pe, xy = sc.get("position_error_m"), sc.get("position_error_xy_m", sc.get("euclidean_xy_m"))
        row["position_error_xy_m"] = xy
        if sc.get("position_error_z_m") is not None:
            row["position_error_z_m"] = sc["position_error_z_m"]
            row["z_signed"] = True
        elif pe is not None and xy is not None:
            row["position_error_z_m"] = math.sqrt(max(0.0, pe * pe - xy * xy))
            row["z_signed"] = False
        oe, yaw = sc.get("orientation_error_deg"), sc.get("heading_error_deg")
        row["yaw_error_deg"] = yaw
        ypr = sc.get("orientation_error_ypr_deg")
        if ypr:
            row["pan_deg"], row["tilt_deg"], row["roll_deg"] = ypr.get("pan_deg"), ypr.get("tilt_deg"), ypr.get("roll_deg")
            row["pitch_roll_residual_deg"] = math.sqrt(max(0.0, (ypr.get("tilt_deg") or 0) ** 2 + (ypr.get("roll_deg") or 0) ** 2))
        elif oe is not None and yaw is not None:
            row["pitch_roll_residual_deg"] = math.sqrt(max(0.0, oe * oe - yaw * yaw))
        # trunk: last commanded targets vs the goal trunk configuration
        tro = (inst_maps.get(sp, {}).get(str(r["episode_index"])) or {}).get("tro")
        gs = goal_stance(os.path.join(WT, tro)) if tro else None
        act_path = os.path.join(WT, r["actions_npy"]) if r.get("actions_npy") else None
        actions = np.load(act_path) if act_path and os.path.exists(act_path) else None
        if gs is not None and gs.get("translate") is not None:
            gq = torso_q(gs["translate"], gs.get("tilt", 0.0))
            row["goal_trunk_q"] = [round(float(v), 4) for v in gq]
            row["goal_trunk_translate_tilt"] = [round(float(gs["translate"]), 4), round(float(gs.get("tilt", 0.0)), 4)]
            if gs.get("base_pose2d"):
                row["goal_base_xyyaw"] = [round(float(v), 4) for v in gs["base_pose2d"]]
            if actions is not None and len(actions):
                last = actions[-1, TRUNK]
                row["trunk_last_cmd_q"] = [round(float(v), 4) for v in last]
                row["trunk_last_cmd_vs_goal_dq_rad"] = [round(float(v), 4) for v in (last - gq)]
                row["trunk_last_cmd_vs_goal_max_abs_dq_rad"] = round(float(np.abs(last - gq).max()), 4)
        btg = sc.get("base_trunk_vs_goal") or {}
        for k in ("base_final_xyyaw", "base_final_vs_goal_xy_m", "base_final_vs_goal_yaw_deg", "trunk_final_q", "trunk_final_vs_goal_max_abs_dq_rad"):
            if btg.get(k) is not None:
                row[k] = btg[k]
        ca = sc.get("closest_approach") or {}
        if ca.get("min_position_error_m") is not None:
            row["closest_approach"] = {k: ca.get(k) for k in ("min_position_error_m", "step", "orientation_error_at_min_pos_deg", "min_orientation_error_deg", "min_orientation_step", "first_step_within_tolerance")}
        # stall
        if actions is not None:
            s = stall_step(actions, args.base_eps, args.joint_eps, args.stall_steps)
            row["n_actions"] = int(len(actions))
            row["stall_step"] = s
            row["quiet_steps_at_end"] = None if s is None else int(len(actions) - s)
        rows.append(row)

    misses = [x for x in rows if x["success"] is False and x["split"] != "replay"]

    def agg(xs, key):
        v = [x[key] for x in xs if x.get(key) is not None]
        return {"n": len(v), "median": float(np.median(v)) if v else None, "mean": float(np.mean(v)) if v else None, "max": float(np.max(v)) if v else None}

    base_dominated = [x for x in misses if x.get("position_error_xy_m") is not None and x.get("position_error_z_m") is not None and x["position_error_xy_m"] > abs(x["position_error_z_m"])]
    trunk_dominated = [x for x in misses if x.get("position_error_xy_m") is not None and x.get("position_error_z_m") is not None and x["position_error_xy_m"] <= abs(x["position_error_z_m"])]
    far = [x for x in misses if (x.get("position_error_xy_m") or 0) > 0.5]
    near = [x for x in misses if x.get("position_error_xy_m") is not None and x["position_error_xy_m"] <= 0.5]
    near_trunk_off = [x for x in near if (x.get("trunk_last_cmd_vs_goal_max_abs_dq_rad") or 0) > 0.1]
    yaw_dominated = [x for x in misses if x.get("yaw_error_deg") is not None and x.get("pitch_roll_residual_deg") is not None and x["yaw_error_deg"] > x["pitch_roll_residual_deg"]]
    stalled = [x for x in misses if x.get("stall_step") is not None and x.get("n_actions") and x["n_actions"] >= (x.get("horizon") or 0) and x["quiet_steps_at_end"] >= args.stall_steps]
    saving = sum(max(0, x["n_actions"] - (x["stall_step"] + args.stall_steps)) for x in stalled)
    total_steps = sum(x["n_actions"] for x in misses if x.get("n_actions"))
    out = {"format": "isg-lab-misses/v1", "n_records": len(rows), "n_misses": len(misses), "rows": rows,
           "aggregates": {
               "position_error_m": agg(misses, "position_error_m"), "position_error_xy_m": agg(misses, "position_error_xy_m"), "position_error_z_abs_m": {**agg([{"z": abs(x["position_error_z_m"])} for x in misses if x.get("position_error_z_m") is not None], "z")},
               "orientation_error_deg": agg(misses, "orientation_error_deg"), "yaw_error_deg": agg(misses, "yaw_error_deg"), "pitch_roll_residual_deg": agg(misses, "pitch_roll_residual_deg"),
               "trunk_last_cmd_vs_goal_max_abs_dq_rad": agg(misses, "trunk_last_cmd_vs_goal_max_abs_dq_rad"),
               "n_horizontal_dominated": len(base_dominated), "n_vertical_dominated": len(trunk_dominated), "n_far_xy_gt_0_5m": len(far), "n_near_xy_le_0_5m": len(near),
               "n_near_with_trunk_cmd_off_gt_0_1rad": len(near_trunk_off), "n_yaw_dominated": len(yaw_dominated),
               "stall": {"rule": f"end the stream when the trailing {args.stall_steps}-step window has mean |base cmd| < {args.base_eps} (x 0.75 m/s = {args.base_eps * 0.75 * 100:.1f} cm/s) and every joint target's excursion over the window < {args.joint_eps} rad (the robot has parked); the epilogue then runs as usual",
                         "n_misses_at_horizon": sum(1 for x in misses if x.get("n_actions") and x["n_actions"] >= (x.get("horizon") or 0)),
                         "n_stalled_misses": len(stalled), "stall_steps_median": float(np.median([x["stall_step"] for x in stalled])) if stalled else None,
                         "steps_saved": int(saving), "total_miss_steps": int(total_steps), "fraction_saved": (saving / total_steps) if total_steps else None,
                         "per_episode": [{"run": x["run"], "step": x["step"], "tag": x.get("tag"), "episode": x["episode"], "stall_step": x["stall_step"], "n_actions": x["n_actions"]} for x in stalled]},
           }}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    # markdown table of the misses
    print("| run | step | tag | ep | pos m | xy m | |z| m | ori deg | yaw deg | pitch/roll deg | trunk cmd dq max rad | base xy vs goal m | stall step |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in sorted(misses, key=lambda x: (x["run"], x["step"], str(x.get("tag")), x["episode"])):
        f = lambda v, n=3: "" if v is None else f"{v:.{n}f}"  # noqa: E731
        print(f"| {x['run']} | {x['step']} | {x.get('tag') or ''} | {x['episode']} | {f(x.get('position_error_m'))} | {f(x.get('position_error_xy_m'))} | "
              f"{f(abs(x['position_error_z_m']) if x.get('position_error_z_m') is not None else None)} | {f(x.get('orientation_error_deg'), 1)} | {f(x.get('yaw_error_deg'), 1)} | "
              f"{f(x.get('pitch_roll_residual_deg'), 1)} | {f(x.get('trunk_last_cmd_vs_goal_max_abs_dq_rad'))} | {f(x.get('base_final_vs_goal_xy_m'))} | {x.get('stall_step') if x.get('stall_step') is not None else ''} |")
    print(json.dumps(out["aggregates"], indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
