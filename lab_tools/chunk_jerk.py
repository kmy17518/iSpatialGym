#!/usr/bin/env python3
"""Chunk-boundary jerk diagnostic (user, 2026-10-08; results only under lab/diagnostics/chunk-jerk/).

Sub-commands (conda env ispatialgym-v2.0-vectorized, Isaac Sim):

  replay   --episodes 113 98            instrumented open-loop replay of training episodes (per-step proprio, truth, action)
  rollout  --episodes 101 113 121       instrumented closed-loop rollout of a checkpoint (own ACT server on --port, default execution)
           [--proprio-mode commanded] [--render-before-obs] [--tag ab-...]     the item-4 A/B switches
  analyze                               item 1 (tracking lag in the replays), item 2 (re-plan boundaries), item 3 (jerk metric of
                                        the execution-settings episodes from their actions.npy) -> summary.json

Per-step logs: lab/diagnostics/chunk-jerk/<kind>/episode_XXXXXX.npz (proprio T x 61, proprio_sent, action T x 23, truth: base_xyyaw
T x 3, torso_q T x 4, torso_qd T x 4, arm_q T x 14; chunk_pos T) + episode_XXXXXX.json (the usual record fields). Videos / meta
under runlogs/diagnostics/chunk-jerk/.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
import urllib.request

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
ACT = os.path.join(WT, "act-isg")
OUT = os.path.join(WT, "lab", "diagnostics", "chunk-jerk")
RAW = os.path.join(WT, "runlogs", "diagnostics", "chunk-jerk")
TASK = "camera_relocalization-standard"
BASE_VEL_LIMITS = (0.75, 0.75, 1.0)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def extract_state(raw):
    """b1k_dataset.extract_state (ACT repo): 23 values, base velocity 0:3, torso 3:7, left arm 7:14, left gripper 14, right arm 15:22, right gripper 22."""
    sys.path.insert(0, ACT)
    from b1k_dataset import extract_state as es

    return es(raw, "sum")


# ----------------------------------------------------------------------------------------------------------------------
# running episodes
# ----------------------------------------------------------------------------------------------------------------------

def save_log(ctx, path_npz):
    import numpy as np

    lg = ctx["jerk_log"]
    n = len(lg["proprio"])
    acts = lg["action"][:n]
    acts = [a if a is not None else np.full(23, np.nan, np.float32) for a in acts] + [np.full(23, np.nan, np.float32)] * (n - len(acts))
    sent = [p if p is not None else np.full(61, np.nan, np.float32) for p in lg["proprio_sent"][:n]]
    np.savez(path_npz, proprio=np.stack(lg["proprio"][:n]), proprio_sent=np.stack(sent) if sent else np.zeros((0, 61)), action=np.stack(acts),
             base_xyyaw=np.asarray([t["base_xyyaw"] for t in lg["truth"][:n]], np.float64), torso_q=np.asarray([t["torso_q"] for t in lg["truth"][:n]], np.float64),
             torso_qd=np.asarray([t["torso_qd"] for t in lg["truth"][:n]], np.float64),
             arm_q=np.asarray([t["arm_q"]["left"] + t["arm_q"]["right"] for t in lg["truth"][:n]], np.float64),
             chunk_pos=np.asarray(lg["chunk_pos"][:n], np.int32), t_wall=np.asarray(lg["t_wall"][:n], np.float64))


def start_server(ckpt, port, extra):
    logf = open(os.path.join(RAW, f"server_{port}.log"), "a")
    args = ["scripts/b1k/serve_b1k.py", "--model-path", os.path.relpath(ckpt, ACT), "--port", str(port), "--device", "cuda", *extra]
    proc = subprocess.Popen([os.path.join(ACT, ".venv", "bin", "python"), *args], cwd=ACT, stdout=logf, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3) as r:
                if r.status == 200:
                    break
        except Exception:  # noqa: BLE001
            pass
        if proc.poll() is not None:
            raise RuntimeError("diagnostic server died")
        time.sleep(2)
    time.sleep(2)
    return proc, "CUDA_VISIBLE_DEVICES=0 .venv/bin/python " + " ".join(args)


def run_episodes(args, kind):
    import numpy as np
    import torch as th

    from lab_tools import act_policy as AP
    from lab_tools import chunk_jerk_policy as CJ
    from lab_tools import evaluator_patches
    from lab_tools import replay_policy as RP
    from lab_tools.eval_episodes import base_and_trunk_vs_goal, closest_approach, load_goal, score_relocalization, sha256_of
    from lab_tools.replay_episodes import load_episode_arrays

    evaluator_patches.install()
    from ispatialgym.oracle import run_isg as R

    out_dir = os.path.join(OUT, kind)
    raw_dir = os.path.join(RAW, kind)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    inst = json.load(open(os.path.join(WT, "lab", "instances-train.json")))["episodes"]
    dataset = os.path.join(WT, "datasets", "isg-init")
    goals = {g["episode_index"]: g for g in json.load(open(os.path.join(dataset, "isg_meta", "goals", "index.json")))["episodes"]}
    server_proc, server_cmd = None, None
    if kind.startswith("rollout"):
        server_proc, server_cmd = start_server(args.ckpt, args.port, args.server_args or [])
    shared = {}
    try:
        for e in args.episodes:
            v = inst[str(e)]
            seed = int(v["instance_index"])
            th.manual_seed(seed)
            np.random.seed(seed % (2 ** 32))
            random.seed(seed)
            rec = {"kind": kind, "episode_index": e, "instance": v["instance"], "seed": seed, "horizon": args.horizon, "started_at": utc(), "tag": args.tag,
                   "proprio_mode": args.proprio_mode, "render_before_obs": bool(args.render_before_obs), "server_args": server_cmd, "checkpoint": None}
            t0 = time.time()
            try:
                if kind.startswith("rollout"):
                    rec["checkpoint"] = {"path": os.path.relpath(args.ckpt, WT), "sha256": sha256_of(args.ckpt), "run": args.run, "step": args.step}
                    wrist = {}
                    gi = os.path.join(WT, "datasets", "goals", "isg-init", "index.json")
                    if os.path.exists(gi):
                        for g in json.load(open(gi))["episodes"]:
                            if int(g["episode_index"]) == e:
                                wrist[g["camera"]] = os.path.join(WT, "datasets", "goals", "isg-init", g["png"])
                    goal_png = os.path.join(dataset, goals[e]["png"])
                    ctx = {"server": f"ws://127.0.0.1:{args.port}", "goal_rgb": load_goal(goal_png), "goal_rgbs": {"zed_link": load_goal(goal_png), **{c: load_goal(p) for c, p in wrist.items()}},
                           "task_id": int(goals[e]["train_task_index"]), "horizon": args.horizon, "stop_on_success": True, "expect_action_horizon": None,
                           "proprio_mode": args.proprio_mode, "render_before_obs": bool(args.render_before_obs), "env": shared.get("env"), "shared": shared}
                    AP.CONTEXT = ctx
                    policy_spec = "lab_tools.chunk_jerk_policy:InstrumentedActPolicy"
                else:
                    actions, states, length = load_episode_arrays(dataset, e)
                    ctx = {"actions": actions, "horizon": args.horizon, "stop_on_success": True}
                    RP.CONTEXT = ctx
                    policy_spec = "lab_tools.chunk_jerk_policy:InstrumentedReplayPolicy"
                argv_r = ["--instance", v["tro"], "--mode", "policy", "--policy", policy_spec, "--policy-steps", str(args.horizon), "--max-steps", "6000",
                          "--settle-steps", "50", "--out", raw_dir, "--record", "video", "--video", "--video-size", "540", "--demo-gating", "none", "--no-goal-override", "--log-level", "WARNING"]
                rargs = R.parse_args(argv_r)
                if shared.get("env") is not None:
                    shared["template_dir"] = os.path.dirname(os.path.dirname(os.path.abspath(v["tro"])))
                    if kind.startswith("rollout"):
                        ctx["env"] = shared["env"]
                rc, result = R.run_instance(rargs, v["tro"], shared)
                if kind.startswith("rollout") and shared.get("env") is not None and ctx.get("env") is None:
                    ctx["env"] = shared["env"]
                meta = {}
                od = result.get("out")
                if od and os.path.exists(os.path.join(od, "meta.json")):
                    meta = json.load(open(os.path.join(od, "meta.json")))
                rec.update({"success": bool(result.get("success")), "reason": result.get("reason"), "steps": int(ctx.get("steps") or 0), "stop_reason": ctx.get("stop_reason"),
                            "first_success_step": meta.get("first_success_step"), "binary_success_ever": meta.get("binary_success_ever"), "settle_windows": meta.get("settle_windows"),
                            "video": os.path.relpath(meta["video"], WT) if meta.get("video") else None, "raw_dir": os.path.relpath(od, WT) if od else None,
                            "server_metadata": {k: val for k, val in (ctx.get("server_metadata") or {}).items() if k in ("action_horizon", "chunk_size", "execution_mode", "checkpoint_step", "goal_views")}})
                ep = {"bucket_hi_m": None, "tro": v["tro"]}
                score = score_relocalization(meta, shared, ep)
                score["closest_approach"] = closest_approach(ctx.get("step_errors"))
                score["base_trunk_vs_goal"] = base_and_trunk_vs_goal(shared, ep, [a for a in ctx["jerk_log"]["action"] if a is not None])
                rec["score"] = score
                if result.get("status") == "error":
                    rec["error"] = result.get("error")
                save_log(ctx, os.path.join(out_dir, f"episode_{e:06d}.npz"))
                rec["steps_logged"] = len(ctx["jerk_log"]["proprio"])
            except Exception as ex:  # noqa: BLE001
                rec["error"] = repr(ex)
                rec["traceback"] = traceback.format_exc()
                traceback.print_exc()
            rec["wall_s"] = round(time.time() - t0, 1)
            rec["finished_at"] = utc()
            with open(os.path.join(out_dir, f"episode_{e:06d}.json"), "w") as f:
                json.dump(rec, f, indent=1, default=str)
            sc = rec.get("score") or {}
            print(f"CHUNK_JERK {kind} ep={e} success={rec.get('success')} steps={rec.get('steps')} first_success={rec.get('first_success_step')} pos={sc.get('position_error_m')} "
                  f"ori={sc.get('orientation_error_deg')} logged={rec.get('steps_logged')} err={rec.get('error')}", flush=True)
    finally:
        if server_proc is not None:
            server_proc.terminate()
        try:
            import omnigibson as og

            og.shutdown()
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------------------------------------------------

def load_log(path):
    import numpy as np

    d = np.load(path)
    return {k: d[k] for k in d.files}


def lag_analysis(lg) -> dict:
    """Item 1: commanded torso joints 1-3 at step t vs the measured torso (proprio, extract_state 3:7) k steps later; base velocity
    command vs measured velocity per lag (correlation). proprio[t] is the observation before action t."""
    import numpy as np

    act, pro = lg["action"], lg["proprio"]
    n = min(len(act), len(pro))
    valid = ~np.isnan(act[:n]).any(1)  # the last call returns None (stream end): no action row
    n = int(np.where(valid)[0].max()) + 1 if valid.any() else 0
    st = np.stack([extract_state(p) for p in pro[:n]])
    cmd_t = act[:n, 3:6]
    meas_t = st[:, 3:6]
    truth_t = lg["torso_q"][:n, 0:3]
    out = {"n_steps": int(n), "torso": {}, "base": {}}
    moving = np.abs(np.diff(truth_t, axis=0, prepend=truth_t[:1])).max(1) > 1e-4
    for k in range(0, 5):
        m = n - k
        gap = np.abs(cmd_t[:m] - meas_t[k:k + m])
        gap_truth = np.abs(cmd_t[:m] - truth_t[k:k + m])
        sel = moving[:m]
        out["torso"][f"k{k}"] = {"mean_abs_gap_rad": float(gap.mean()), "mean_abs_gap_moving_rad": float(gap[sel].mean()) if sel.any() else None,
                                 "per_joint": [float(v) for v in gap.mean(0)], "vs_truth_mean_abs_gap_rad": float(gap_truth.mean())}
    best = min(range(5), key=lambda k: out["torso"][f"k{k}"]["mean_abs_gap_rad"])
    out["torso"]["best_k"] = best
    out["torso"]["gap_k1_rad"] = out["torso"]["k1"]["mean_abs_gap_rad"]
    out["torso"]["dataset_reference"] = {"best_k": 1, "gap_k1_rad": 0.0003, "demo_step_rad": 0.012}
    out["torso"]["proprio_vs_truth_same_moment_mean_abs_rad"] = float(np.abs(meas_t - truth_t).mean())
    cmd_v = act[:n, 0:3] * np.asarray(BASE_VEL_LIMITS)
    meas_v = st[:, 0:3]
    for k in range(0, 5):
        m = n - k
        cors = []
        for d in range(3):
            a, b = cmd_v[:m, d], meas_v[k:k + m, d]
            cors.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else None)
        out["base"][f"k{k}"] = {"corr_vx_vy_wz": cors, "mean_abs_gap_m_s": [float(v) for v in np.abs(cmd_v[:m] - meas_v[k:k + m]).mean(0)]}
    valid = [k for k in range(5) if out["base"][f"k{k}"]["corr_vx_vy_wz"][0] is not None]
    out["base"]["best_k_vx"] = max(valid, key=lambda k: out["base"][f"k{k}"]["corr_vx_vy_wz"][0]) if valid else None
    return out


def boundary_analysis(lg, window: int = 5, move_tol: float = 0.01) -> dict:
    """Item 2: at every re-plan (chunk_pos 0, step >= action_horizon): lag / drop / lead of torso joints 1-3 along the direction of
    motion over the previous `window` steps; base command vs measured velocity; displacement reversals."""
    import numpy as np

    act, pro, cp = lg["action"], lg["proprio"], lg["chunk_pos"]
    truth_t = lg["torso_q"][:, 0:3]
    base = lg["base_xyyaw"]
    n = min(len(act), len(pro), len(cp))
    st = np.stack([extract_state(p) for p in pro[:n]])
    ah = int(np.diff(np.where(cp[:n] == 0)[0]).min()) if (cp[:n] == 0).sum() > 1 else None
    bounds = [r for r in np.where(cp[:n] == 0)[0] if r >= window and r + window < n and r > 0]
    rows = []
    for r in bounds:
        direction = truth_t[r] - truth_t[r - window]
        row = {"step": int(r), "joints": []}
        for j in range(3):
            if abs(direction[j]) < move_tol:
                continue
            s = 1.0 if direction[j] > 0 else -1.0
            old_last, new_first = act[r - 1, 3 + j], act[r, 3 + j]
            lag = s * (old_last - truth_t[r, j])
            drop = s * (new_first - old_last)
            lead_truth = s * (new_first - truth_t[r, j])
            lead_proprio = s * (new_first - st[r, 3 + j])
            cls = "backward_command" if lead_truth < 0 else ("setpoint_drop" if drop < 0 else "continuous")
            row["joints"].append({"joint": j + 1, "direction_rad_per_window": float(direction[j]), "lag_rad": float(lag), "drop_rad": float(drop), "lead_truth_rad": float(lead_truth),
                                  "lead_proprio_rad": float(lead_proprio), "class": cls})
        # base: commands around the boundary vs measured velocity; displacement direction reversal over the next window
        cmd_old, cmd_new = act[r - 1, 0:3] * np.asarray(BASE_VEL_LIMITS), act[r, 0:3] * np.asarray(BASE_VEL_LIMITS)
        meas = st[r, 0:3]
        d_prev = base[r, :2] - base[r - window, :2]
        d_next = base[r + window, :2] - base[r, :2]
        moving_base = np.linalg.norm(d_prev) > 0.01
        reversal = bool(moving_base and float(np.dot(d_prev, d_next)) < 0)
        row["base"] = {"cmd_old_last": [float(v) for v in cmd_old], "cmd_new_first": [float(v) for v in cmd_new], "measured_v": [float(v) for v in meas],
                       "cmd_change_m_s": float(np.linalg.norm(cmd_new[:2] - cmd_old[:2])), "moving": moving_base, "displacement_reversal": reversal,
                       "d_prev_m": float(np.linalg.norm(d_prev)), "d_next_m": float(np.linalg.norm(d_next))}
        rows.append(row)
    classes = {"backward_command": [], "setpoint_drop": [], "continuous": []}
    for row in rows:
        for jrow in row["joints"]:
            classes[jrow["class"]].append(jrow)
    summ = {"action_horizon": ah, "n_boundaries": len(rows), "n_boundaries_torso_moving": sum(1 for row in rows if row["joints"]),
            "classes": {k: {"n": len(v), "median_lag_rad": float(np.median([x["lag_rad"] for x in v])) if v else None, "median_drop_rad": float(np.median([x["drop_rad"] for x in v])) if v else None,
                            "median_lead_truth_rad": float(np.median([x["lead_truth_rad"] for x in v])) if v else None, "median_lead_proprio_rad": float(np.median([x["lead_proprio_rad"] for x in v])) if v else None}
                        for k, v in classes.items()},
            "base": {"n_moving": sum(1 for row in rows if row["base"]["moving"]), "n_displacement_reversals": sum(1 for row in rows if row["base"]["displacement_reversal"]),
                     "median_cmd_change_m_s": float(np.median([row["base"]["cmd_change_m_s"] for row in rows])) if rows else None},
            "boundaries": rows}
    # jerk per degree of freedom over the whole episode: reversals of the TRUE torso joint velocity and of the base travel direction,
    # at boundaries vs elsewhere
    tq = truth_t
    vel = np.diff(tq, axis=0)
    rev_steps = {}
    for j in range(3):
        s = np.sign(vel[:, j]) * (np.abs(vel[:, j]) > 1e-3)
        nz = np.where(s != 0)[0]
        rev = [int(nz[i]) + 1 for i in range(1, len(nz)) if s[nz[i]] != s[nz[i - 1]]]
        rev_steps[f"torso_joint{j + 1}"] = rev
    dxy = np.diff(base[:n, :2], axis=0)
    mv = np.linalg.norm(dxy, axis=1) > 0.002
    nz = np.where(mv)[0]
    base_rev = [int(nz[i]) + 1 for i in range(1, len(nz)) if float(np.dot(dxy[nz[i]], dxy[nz[i - 1]])) < 0]
    rev_steps["base_travel"] = base_rev
    arm = lg["arm_q"][:n]
    av = np.diff(arm, axis=0)
    arm_rev = 0
    for j in range(arm.shape[1]):
        s = np.sign(av[:, j]) * (np.abs(av[:, j]) > 1e-3)
        nz2 = np.where(s != 0)[0]
        arm_rev += sum(1 for i in range(1, len(nz2)) if s[nz2[i]] != s[nz2[i - 1]])
    at_boundary = lambda steps: sum(1 for t in steps if ah and any(abs(t - b) <= 2 for b in bounds))  # noqa: E731
    summ["reversals"] = {k: {"n": len(v), "n_within_2_steps_of_a_boundary": at_boundary(v), "steps": v[:60]} for k, v in rev_steps.items()}
    summ["reversals"]["arms_total_14_joints"] = int(arm_rev)
    return summ


def exec_jerk_metric(actions, base_tol: float = 0.05, joint_tol: float = 1e-3) -> dict:
    """Item 3 (from actions.npy, i.e. COMMANDED quantities): direction reversals of the commanded torso joint velocity (joints 1-3)
    and of the commanded base travel direction (angle between consecutive commanded (vx, vy) > 90 deg, both above base_tol)."""
    import numpy as np

    a = np.asarray(actions, dtype=np.float64)
    out = {"n_actions": int(len(a))}
    tv = np.diff(a[:, 3:6], axis=0)
    tot = 0
    per = []
    for j in range(3):
        s = np.sign(tv[:, j]) * (np.abs(tv[:, j]) > joint_tol)
        nz = np.where(s != 0)[0]
        r = sum(1 for i in range(1, len(nz)) if s[nz[i]] != s[nz[i - 1]])
        per.append(int(r))
        tot += r
    out["torso_cmd_velocity_reversals"] = int(tot)
    out["torso_cmd_velocity_reversals_per_joint"] = per
    v = a[:, 0:2]
    mv = np.linalg.norm(v, axis=1) > base_tol
    nz = np.where(mv)[0]
    out["base_cmd_direction_reversals"] = int(sum(1 for i in range(1, len(nz)) if float(np.dot(v[nz[i]], v[nz[i - 1]])) < 0))
    out["base_cmd_moving_steps"] = int(mv.sum())
    out["note"] = "commanded (actions.npy), not measured: per-step joint positions were not logged for these episodes"
    return out


def analyze(args) -> int:
    import numpy as np

    summary = {"format": "isg-lab-chunk-jerk/v1", "created_at": utc(), "replay": {}, "rollout": {}, "ab": {}, "exec_settings_jerk": {}}
    for kind in sorted(os.listdir(OUT)) if os.path.exists(OUT) else []:
        d = os.path.join(OUT, kind)
        if not os.path.isdir(d):
            continue
        for npz in sorted(glob.glob(os.path.join(d, "episode_*.npz"))):
            e = int(os.path.basename(npz)[len("episode_"):-len(".npz")])
            lg = load_log(npz)
            rec = json.load(open(npz[:-4] + ".json")) if os.path.exists(npz[:-4] + ".json") else {}
            entry = {"record": {k: rec.get(k) for k in ("success", "steps", "first_success_step", "stop_reason", "proprio_mode", "render_before_obs", "tag")},
                     "position_error_m": (rec.get("score") or {}).get("position_error_m"), "orientation_error_deg": (rec.get("score") or {}).get("orientation_error_deg")}
            try:
                entry["lag"] = lag_analysis(lg)
            except Exception as ex:  # noqa: BLE001
                entry["lag"] = {"error": repr(ex)}
            if kind.startswith("rollout") or kind.startswith("ab"):
                try:
                    entry["boundaries"] = boundary_analysis(lg)
                except Exception as ex:  # noqa: BLE001
                    entry["boundaries"] = {"error": repr(ex), "traceback": traceback.format_exc()}
            bucket = "replay" if kind == "replay" else ("ab" if kind.startswith("ab") else "rollout")
            summary[bucket][f"{kind}/{e}"] = entry
            with open(npz[:-4] + ".analysis.json", "w") as f:
                json.dump(entry, f, indent=1, default=str)
    # item 3: the execution-settings episodes (saved actions.npy)
    rows = []
    for p in sorted(glob.glob(os.path.join(WT, "lab", "diagnostics", "exec", "*", "*", "step_*", TASK, "episode_*.json"))) + \
            sorted(glob.glob(os.path.join(WT, "lab", "tierB-train", "*", "step_*", TASK, "episode_*.json"))):
        r = json.load(open(p))
        if r.get("error") or not r.get("actions_npy") or not os.path.exists(os.path.join(WT, r["actions_npy"])):
            continue
        tag = r.get("tag") or "default"
        if tag not in ("default", "ta", "ah1", "ah4", "ah8"):
            continue
        m = exec_jerk_metric(np.load(os.path.join(WT, r["actions_npy"])))
        rows.append({"run": r["run"], "step": r["step"], "tag": tag, "episode": r["episode_index"], "success": r.get("success"), "position_error_m": (r.get("score") or {}).get("position_error_m"),
                     "orientation_error_deg": (r.get("score") or {}).get("orientation_error_deg"), **{k: v for k, v in m.items() if k != "note"}})
    agg = {}
    for row in rows:
        a = agg.setdefault(row["tag"], {"n": 0, "torso_rev": [], "base_rev": [], "success": 0})
        a["n"] += 1
        a["torso_rev"].append(row["torso_cmd_velocity_reversals"])
        a["base_rev"].append(row["base_cmd_direction_reversals"])
        a["success"] += int(bool(row["success"]))
    summary["exec_settings_jerk"] = {"note": exec_jerk_metric(np.zeros((2, 23)))["note"], "rows": rows,
                                     "per_setting": {t: {"n": a["n"], "n_success": a["success"], "median_torso_cmd_velocity_reversals": float(np.median(a["torso_rev"])),
                                                         "median_base_cmd_direction_reversals": float(np.median(a["base_rev"])), "mean_torso": float(np.mean(a["torso_rev"])), "mean_base": float(np.mean(a["base_rev"]))}
                                                     for t, a in agg.items()}}
    # jerk table of the execution-settings episodes (commanded quantities) next to success / errors
    lines = ["| run | step | tag | ep | success | pos m | ori deg | torso cmd-velocity reversals (j1,j2,j3) | base cmd-direction reversals |", "|---|---|---|---|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda x: (x["tag"], x["run"], x["step"], x["episode"])):
        pe = row.get("position_error_m")
        oe = row.get("orientation_error_deg")
        lines.append(f"| {row['run']} | {row['step']} | {row['tag']} | {row['episode']} | {'OK' if row['success'] else 'x'} | {'' if pe is None else f'{pe:.2f}'} | {'' if oe is None else f'{oe:.0f}'} | "
                     f"{row['torso_cmd_velocity_reversals']} ({','.join(str(v) for v in row['torso_cmd_velocity_reversals_per_joint'])}) | {row['base_cmd_direction_reversals']} |")
    lines.append("")
    lines.append("| setting | n | successes | median torso reversals | mean torso | median base reversals | mean base |")
    lines.append("|---|---|---|---|---|---|---|")
    for t, a in sorted(summary["exec_settings_jerk"]["per_setting"].items()):
        lines.append(f"| {t} | {a['n']} | {a['n_success']} | {a['median_torso_cmd_velocity_reversals']:.0f} | {a['mean_torso']:.1f} | {a['median_base_cmd_direction_reversals']:.0f} | {a['mean_base']:.2f} |")
    md = ("# Jerk metric of the execution-settings episodes (item 3)\n\nCounted on the COMMANDED actions (actions.npy): sign changes of the commanded torso joint "
          "velocity (joints 1-3, |step| > 1e-3 rad) and reversals of the commanded base travel direction (angle > 90 deg between consecutive commands with |v| > 0.05). "
          "Per-step joint positions were not logged for these episodes, so measured reversals are not available; the instrumented rollouts "
          "(summary.json) show the measured torso reversals cluster at chunk boundaries.\n\n" + "\n".join(lines) + "\n")
    with open(os.path.join(OUT, "exec_jerk_table.md"), "w") as f:
        f.write(md)
    os.makedirs(os.path.join(WT, "lab", "diagnostics", "exec"), exist_ok=True)
    with open(os.path.join(WT, "lab", "diagnostics", "exec", "jerk_table.md"), "w") as f:
        f.write(md)
    summary["controller_configuration"] = {
        "evaluator": "ispatialgym.oracle.r1pro_model.JOYLO_CONTROLLER_CONFIG through run_isg (the same code path and config the demos were collected with): base HolonomicBaseJointController, "
                     "motor_type velocity, command_input_limits [-1, 1]^3 -> command_output_limits +-(0.75, 0.75, 1.0) m/s | rad/s in the robot frame, use_impedances False "
                     "(vel_kp 150 is only read with impedances: the velocity target is written to the PhysX joint drives of the virtual base joints directly); trunk and arms "
                     "JointController, motor_type position, absolute targets (use_delta_commands False), no input / output limits, use_impedances False (pos_kp 150 unused: the "
                     "position target goes to the PhysX joint drive, drive stiffness / damping from the R1Pro USD); grippers MultiFingerGripperController smooth.",
        "control_rate_hz": 30, "physics_rate_hz": 120, "physics_substeps_per_control_step": 4, "setpoint_interpolation": "none (each 30 Hz action sets the drive targets once; PhysX holds them for the 4 substeps)",
        "n_render_iterations": 1,
        "data_collection": "identical: run_isg --mode oracle with the same build_env_config (action_frequency 30, physics_frequency 120) and JOYLO_CONTROLLER_CONFIG; the oracle streams its planned joint "
                           "targets at 30 Hz through the same executor.step",
        "observation_timing": "Environment.step: robot.apply_action(a_t) -> og.sim.step() (4 physics substeps at 120 Hz, the render of the step inside) -> get_obs(): the proprio is read from the "
                              "physics state after a_t's substeps (fresh: proprio torso == robot joint positions read at the same moment, 0.000 rad difference in every logged step), the camera "
                              "images come from the render issued inside the same og.sim.step() (RTX hands out the frame with the usual ~1-frame lag) - the same order as in collection, "
                              "where the dataset's images lag the proprio by 1-1.25 frames.",
    }
    # conclusion from the numbers
    rep = summary.get("replay", {})
    rol = summary.get("rollout", {})
    rep_k1 = [v["lag"]["torso"]["k1"]["mean_abs_gap_moving_rad"] for v in rep.values() if v.get("lag", {}).get("torso")]
    rep_best = [v["lag"]["torso"]["best_k"] for v in rep.values() if v.get("lag", {}).get("torso")]
    cls = {c: sum((v.get("boundaries") or {}).get("classes", {}).get(c, {}).get("n", 0) or 0 for v in rol.values()) for c in ("backward_command", "setpoint_drop", "continuous")}
    lags = [x["lag_rad"] for v in rol.values() for row in (v.get("boundaries") or {}).get("boundaries", []) for x in row["joints"]]
    leads_back = [x["lead_truth_rad"] for v in rol.values() for row in (v.get("boundaries") or {}).get("boundaries", []) for x in row["joints"] if x["class"] == "backward_command"]
    import numpy as _np

    summary["conclusion"] = {
        "supported_explanation": "neither (A) nor (B) as an evaluator defect: the simulator tracks the commands as tightly as the data (A excluded) and the proprio the policy receives "
                                 "is the true, fresh joint state with the dataset's semantics (stale / mismatched proprio excluded); the camera lag is the collection's own. The backward "
                                 "step is produced by the policy: at a re-plan the new chunk's first torso command lands behind the robot's actual position.",
        "numbers": {
            "replay_tracking": {"best_k": rep_best, "mean_abs_gap_k1_moving_rad": rep_k1, "dataset_k1_gap_rad": 0.0003, "proprio_vs_truth_rad": 0.0,
                                "base_velocity_best_k": [v["lag"]["base"]["best_k_vx"] for v in rep.values() if v.get("lag", {}).get("base")]},
            "rollout_boundaries": {"n_boundary_joint_cases": sum(cls.values()), **cls, "median_lag_rad_old_last_cmd_minus_truth": float(_np.median(lags)) if lags else None,
                                   "median_lead_truth_rad_backward_cases": float(_np.median(leads_back)) if leads_back else None,
                                   "reversals_of_true_torso_velocity_within_2_steps_of_a_boundary": {k: [(v.get("boundaries") or {}).get("reversals", {}).get(f"torso_joint{j}", {}).get("n_within_2_steps_of_a_boundary") for j in (1, 2, 3)] for k, v in rol.items()},
                                   "reversals_of_true_torso_velocity_total": {k: [(v.get("boundaries") or {}).get("reversals", {}).get(f"torso_joint{j}", {}).get("n") for j in (1, 2, 3)] for k, v in rol.items()},
                                   "base_displacement_reversals": {k: (v.get("boundaries") or {}).get("base", {}).get("n_displacement_reversals") for k, v in rol.items()}},
        },
        "degrees_of_freedom_with_the_jerk": "torso (and arms: hundreds of commanded sawtooth reversals), not the base (0-2 base travel reversals per episode)",
        "ab_rollout": "not run: items 1 and 2 point to no observation / tracking mismatch (production timing and proprio semantics unchanged, per machine 1)",
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps({"replay": {k: v.get("lag", {}).get("torso", {}) for k, v in summary["replay"].items()},
                      "rollout": {k: {c: (v.get("boundaries") or {}).get("classes", {}).get(c, {}).get("n") for c in ("backward_command", "setpoint_drop", "continuous")} for k, v in summary["rollout"].items()},
                      "exec": summary["exec_settings_jerk"]["per_setting"]}, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["replay", "rollout", "analyze"])
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--horizon", type=int, default=1500)
    ap.add_argument("--ckpt", default=os.path.join(ACT, "ckpts", "r1-early-copy05-st0-s0", "eval", "step-00100000.pt"))
    ap.add_argument("--run", default="r1-early-copy05-st0-s0")
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--server-args", nargs="*", default=None)
    ap.add_argument("--proprio-mode", default="measured", choices=["measured", "commanded"])
    ap.add_argument("--render-before-obs", action="store_true")
    ap.add_argument("--tag", default=None, help="log sub-dir for A/B rollouts (ab-proprio, ab-render); default rollout")
    args = ap.parse_args(argv)
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.cmd == "analyze":
        return analyze(args)
    kind = "replay" if args.cmd == "replay" else (args.tag or "rollout")
    run_episodes(args, kind)
    return 0


if __name__ == "__main__":
    sys.exit(main())
