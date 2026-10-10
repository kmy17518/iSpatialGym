#!/usr/bin/env python3
"""Positive control: open-loop replay of recorded relocalization episodes through the Tier B evaluator path.

For every listed episode of a LeRobot dataset (test set ``datasets/isg-init-eval`` with ``lab/instances.json``, or training set
``datasets/isg-init`` with ``lab/instances-train.json``) the mapped instance is restored and the episode's recorded actions
(``action`` column) are streamed by :mod:`lab_tools.replay_policy` through ``run_isg --mode policy`` with the evaluator's
settings (:mod:`lab_tools.evaluator_patches`: BEHAVIOR sensor set, timeout-only stream QA, the never-held epilogue), the
queue's horizon and stop-on-success. The record (``<lab-out>/episode_XXXXXX.json``) carries the evaluator's verdict and
scores like a policy episode plus ``replay``: the demo's length, how many actions were streamed, and the deviation of the
replayed base trajectory from the demo's own (dead-reckoned from ``observation.state[0:3]`` = robot-frame base velocity at
30 Hz, from the restored init pose), with a dead-reckoning self-check on the replay's own velocities.

    python lab_tools/replay_episodes.py --split eval50 --dataset datasets/isg-init-eval --instances lab/instances.json \
        --episodes 2 3 33 34 45 --horizon 1500 --lab-out lab/diagnostics/replay/eval50 --raw-out runlogs/tierB/replay/eval50
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
import traceback

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
TASK = "camera_relocalization-standard"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_episode_arrays(root: str, episode: int):
    """(actions (T, 23), states (T, 61), frame count) of one episode from the dataset's parquet files."""
    import numpy as np
    import pandas as pd

    meta = None
    for p in sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "file-*.parquet"))):
        df = pd.read_parquet(p, columns=["episode_index", "length", "data/chunk_index", "data/file_index"])
        hit = df[df["episode_index"] == episode]
        if len(hit):
            meta = hit.iloc[0]
            break
    if meta is None:
        raise KeyError(f"episode {episode} not in {root}")
    data = pd.read_parquet(os.path.join(root, "data", f"chunk-{int(meta['data/chunk_index']):03d}", f"file-{int(meta['data/file_index']):03d}.parquet"),
                           columns=["episode_index", "frame_index", "action", "observation.state"])
    data = data[data["episode_index"] == episode].sort_values("frame_index")
    actions = np.stack([np.asarray(a, dtype=np.float32) for a in data["action"]])
    states = np.stack([np.asarray(s, dtype=np.float32) for s in data["observation.state"]])
    assert len(actions) == int(meta["length"]), (len(actions), int(meta["length"]))
    return actions, states, int(meta["length"])


def dead_reckon(base_qvel, x0: float, y0: float, yaw0: float, dt: float = 1.0 / 30.0):
    """Integrate robot-frame base velocities (vx, vy, wz per step) from (x0, y0, yaw0): pose before each step, T + 1 rows."""
    import numpy as np

    out = [(x0, y0, yaw0)]
    x, y, yaw = x0, y0, yaw0
    for vx, vy, wz in np.asarray(base_qvel, dtype=np.float64):
        c, s = math.cos(yaw), math.sin(yaw)
        x += (c * vx - s * vy) * dt
        y += (s * vx + c * vy) * dt
        yaw += wz * dt
        out.append((x, y, yaw))
    return np.asarray(out)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def velocity_semantics(xyyaw, base_qvel, dt: float = 1.0 / 30.0) -> dict:
    """What the recorded ``base_qvel`` (proprio[0:3]) measures, from the replay's own simulated poses: the finite-difference
    base velocity between consecutive control steps (world frame, then rotated into the robot frame at the step's start /
    end yaw) is compared with the measured sample after the step (lag 0) and the one before it (lag 1); the best-fitting
    frame / lag and the scale between measured and finite-difference yaw rate and speed are reported."""
    import numpy as np

    P = np.asarray(xyyaw, dtype=np.float64)
    V = np.asarray(base_qvel, dtype=np.float64)
    n = min(len(P), len(V)) - 1
    if n < 10:
        return {"n": int(max(n, 0))}
    d = P[1:n + 1] - P[:n]
    d[:, 2] = wrap(d[:, 2])
    fd_world = d / dt  # velocity over step t -> t+1
    out = {"n": int(n), "dt_s": dt, "fits": {}}
    moving = np.hypot(fd_world[:, 0], fd_world[:, 1]) > 0.05
    turning = np.abs(fd_world[:, 2]) > 0.1
    for frame in ("world", "robot_at_start", "robot_at_end"):
        if frame == "world":
            fd = fd_world
        else:
            yaw = P[:n, 2] if frame == "robot_at_start" else P[1:n + 1, 2]
            c, s = np.cos(yaw), np.sin(yaw)
            fd = np.stack([c * fd_world[:, 0] + s * fd_world[:, 1], -s * fd_world[:, 0] + c * fd_world[:, 1], fd_world[:, 2]], axis=1)
        for lag, name in ((0, "measured_after_step"), (1, "measured_before_step")):
            m = V[1:n + 1] if lag == 0 else V[:n]
            err = np.sqrt(np.mean(np.sum((m[:, :2] - fd[:, :2]) ** 2, axis=1))) if n else None
            out["fits"][f"{frame}/{name}"] = {"rms_xy_m_s": round(float(err), 4)}
    best = min(out["fits"], key=lambda k: out["fits"][k]["rms_xy_m_s"])
    out["best_fit"] = best
    m0 = V[1:n + 1]
    sp_fd = np.hypot(fd_world[:, 0], fd_world[:, 1])
    sp_m = np.hypot(m0[:, 0], m0[:, 1])
    if moving.any():
        out["speed_ratio_measured_over_fd_median"] = round(float(np.median(sp_m[moving] / sp_fd[moving])), 4)
    if turning.any():
        out["yaw_rate_ratio_measured_over_fd_median"] = round(float(np.median(m0[turning, 2] / fd_world[turning, 2])), 4)
    out["yaw_total_fd_deg"] = round(math.degrees(float(d[:, 2].sum())), 2)
    out["yaw_total_measured_deg"] = round(math.degrees(float(m0[:, 2].sum() * dt)), 2)
    out["path_length_fd_m"] = round(float(sp_fd.sum() * dt), 4)
    out["path_length_measured_m"] = round(float(sp_m.sum() * dt), 4)
    return out


def yaw_rate_scale(xyyaw, base_qvel, dt: float = 1.0 / 30.0, min_rate: float = 0.05):
    """Least-squares scale k with ``measured_wz ~= k * true_wz`` on the replay's own samples (the proprio's rz virtual-joint
    velocity over-reads the base yaw rate in this simulator; the linear part is exact), or None when the base never turned."""
    import numpy as np

    P = np.asarray(xyyaw, dtype=np.float64)
    V = np.asarray(base_qvel, dtype=np.float64)
    n = min(len(P), len(V)) - 1
    if n < 10:
        return None
    fd = np.diff(np.unwrap(P[:n + 1, 2])) / dt
    m = V[1:n + 1, 2]
    sel = np.abs(fd) > min_rate
    if sel.sum() < 5 or float(np.sum(fd[sel] ** 2)) <= 0:
        return None
    return float(np.sum(m[sel] * fd[sel]) / np.sum(fd[sel] ** 2))


def deviation_stats(rep, ref) -> dict:
    import numpy as np

    rep = np.asarray(rep, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = min(len(rep), len(ref))
    if n == 0:
        return {"n": 0}
    d = np.hypot(rep[:n, 0] - ref[:n, 0], rep[:n, 1] - ref[:n, 1])
    dyaw = np.abs(wrap(rep[:n, 2] - ref[:n, 2]))
    k = int(np.argmax(d))
    return {"n": int(n), "max_xy_m": round(float(d[k]), 4), "max_xy_step": k, "end_xy_m": round(float(d[n - 1]), 4), "mean_xy_m": round(float(d.mean()), 4),
            "max_yaw_deg": round(math.degrees(float(dyaw.max())), 3), "end_yaw_deg": round(math.degrees(float(dyaw[n - 1])), 3)}


def trajectory_block(rep, replay_qvel, demo_qvel, x0: float, y0: float, yaw0: float, actions=None, streamed: int = 0) -> dict:
    """The replayed base trajectory against the demo's own, dead-reckoned from the demo's recorded base velocities
    (observation.state[0:3]: robot-frame vx, vy, wz at 30 Hz) from the same init pose - raw, and with the proprio yaw-rate
    scale measured on the replay's own samples taken out (``calibrated``); the same integration of the replay's own measured
    velocities against its simulated pose gives the method's floor (``self_check``)."""
    import numpy as np

    rep = np.asarray(rep, dtype=np.float64)
    demo_qvel = np.asarray(demo_qvel, dtype=np.float64)
    replay_qvel = np.asarray(replay_qvel if replay_qvel is not None else np.zeros((0, 3)), dtype=np.float64)
    out = {"replay_steps_recorded": int(len(rep)), "demo_steps": int(len(demo_qvel)), "init_xyyaw": [round(float(x0), 4), round(float(y0), 4), round(float(yaw0), 4)]}
    demo_dr = dead_reckon(demo_qvel, x0, y0, yaw0)
    out["raw"] = {"method": "demo base_qvel integrated as recorded", **deviation_stats(rep, demo_dr),
                  "demo_end_xyyaw": [round(float(c), 4) for c in demo_dr[-1]], "demo_yaw_change_deg": round(math.degrees(float(demo_dr[-1, 2] - yaw0)), 2)}
    out["raw"]["self_check"] = deviation_stats(rep, dead_reckon(replay_qvel, x0, y0, yaw0)) if len(replay_qvel) else None
    k = yaw_rate_scale(rep, replay_qvel)
    out["yaw_rate_scale_measured_over_true"] = None if k is None else round(k, 4)
    if k:
        q = demo_qvel.copy()
        q[:, 2] /= k
        cal = dead_reckon(q, x0, y0, yaw0)
        out["calibrated"] = {"method": f"demo base_qvel integrated with wz / {k:.4f} (the scale between the proprio yaw-rate sample and the finite-difference yaw rate of the replay's own simulated pose)",
                             **deviation_stats(rep, cal), "demo_end_xyyaw": [round(float(c), 4) for c in cal[-1]], "demo_yaw_change_deg": round(math.degrees(float(cal[-1, 2] - yaw0)), 2)}
        rq = replay_qvel.copy()
        rq[:, 2] /= k
        out["calibrated"]["self_check"] = deviation_stats(rep, dead_reckon(rq, x0, y0, yaw0))
        out["demo_dead_reckoned_calibrated"] = cal
    out["demo_dead_reckoned_raw"] = demo_dr
    try:
        out["velocity_semantics"] = velocity_semantics(rep, replay_qvel)
    except Exception as ex:  # noqa: BLE001
        out["velocity_semantics"] = {"error": repr(ex)}
    if actions is not None and streamed:
        try:
            from ispatialgym.oracle.r1pro_model import BASE_VEL_LIMITS

            cmd_dr = dead_reckon(np.asarray(actions[:streamed, 0:3], dtype=np.float64) * np.asarray(BASE_VEL_LIMITS, dtype=np.float64), x0, y0, yaw0)
            out["commanded"] = {"method": "commanded base velocity (action[0:3] x BASE_VEL_LIMITS, robot frame) integrated", "vs_simulated": deviation_stats(rep, cmd_dr),
                                "vs_demo_raw": deviation_stats(cmd_dr, demo_dr), "base_vel_limits": list(BASE_VEL_LIMITS)}
        except Exception as ex:  # noqa: BLE001
            out["commanded"] = {"error": repr(ex)}
    out["replay_end_xyyaw"] = [round(float(c), 4) for c in rep[-1]] if len(rep) else None
    return out


def goal_base_pose(tro_path: str):
    """The instance's goal base pose (x, y, yaw) from the sampler's ``<iid>.summary.json`` next to the TRO (the demo ended within
    the camera tolerance of the goal camera pose, so this is the demo's own end pose up to that tolerance); None if absent."""
    p = tro_path.replace("-tro_state.json", ".summary.json")
    if not os.path.exists(p):
        return None
    try:
        g = json.load(open(p)).get("goal_base_pose2d")
        return [float(v) for v in g] if g else None
    except Exception:  # noqa: BLE001
        return None


def end_vs_goal(rep, goal_xyyaw) -> dict | None:
    if goal_xyyaw is None or not len(rep):
        return None
    x, y, yaw = rep[-1]
    return {"goal_base_xyyaw": [round(v, 4) for v in goal_xyyaw], "replay_end_xyyaw": [round(float(x), 4), round(float(y), 4), round(float(yaw), 4)],
            "xy_m": round(math.hypot(float(x) - goal_xyyaw[0], float(y) - goal_xyyaw[1]), 4), "yaw_deg": round(math.degrees(abs(wrap(float(yaw) - goal_xyyaw[2]))), 3),
            "note": "the replay's final base pose vs the instance's goal base pose (exact reference; the demo itself ended within 5 cm / 0.1 rad of the goal camera pose)"}


def headline_deviation(traj: dict) -> dict:
    """Top-level record fields: the largest deviation of the replayed base trajectory from the demo's (calibrated dead reckoning
    when the yaw-rate scale could be measured, raw otherwise) and the method's own floor."""
    cal = traj.get("calibrated") or {}
    raw = traj.get("raw") or {}
    use = cal if cal.get("max_xy_m") is not None else raw
    evg = traj.get("end_vs_goal_base") or {}
    return {"base_trajectory_max_deviation_m": use.get("max_xy_m"), "base_trajectory_end_deviation_m": use.get("end_xy_m"),
            "base_trajectory_max_yaw_deviation_deg": use.get("max_yaw_deg"),
            "base_trajectory_deviation_method": "calibrated" if use is cal else "raw",
            "base_trajectory_method_floor_m": (use.get("self_check") or {}).get("max_xy_m"),
            "base_trajectory_max_deviation_raw_m": raw.get("max_xy_m"),
            "base_end_vs_goal_base_m": evg.get("xy_m"), "base_end_vs_goal_base_yaw_deg": evg.get("yaw_deg")}


def trajectory_report(replay_xyyaw, demo_xyyaw, replay_dr_xyyaw) -> dict:
    """Largest deviation of the replayed base trajectory from the demo's dead-reckoned one over the common steps."""
    import numpy as np

    rep = np.asarray(replay_xyyaw, dtype=np.float64)
    dem = np.asarray(demo_xyyaw, dtype=np.float64)
    n = min(len(rep), len(dem))
    out = {"n_compared_steps": int(n)}
    if n == 0:
        return out
    d = np.hypot(rep[:n, 0] - dem[:n, 0], rep[:n, 1] - dem[:n, 1])
    dyaw = np.abs(wrap(rep[:n, 2] - dem[:n, 2]))
    k = int(np.argmax(d))
    out.update({"max_xy_deviation_m": round(float(d[k]), 4), "max_xy_deviation_step": k, "xy_deviation_at_end_m": round(float(d[n - 1]), 4),
                "max_yaw_deviation_deg": round(math.degrees(float(dyaw.max())), 3), "yaw_deviation_at_end_deg": round(math.degrees(float(dyaw[n - 1])), 3),
                "mean_xy_deviation_m": round(float(d.mean()), 4), "replay_steps_recorded": int(len(rep)), "demo_steps": int(len(dem)) - 1})
    rdr = np.asarray(replay_dr_xyyaw, dtype=np.float64)
    m = min(len(rep), len(rdr))
    if m:
        e = np.hypot(rep[:m, 0] - rdr[:m, 0], rep[:m, 1] - rdr[:m, 1])
        out["dead_reckoning_self_check_max_m"] = round(float(e.max()), 4)
        out["dead_reckoning_self_check_end_m"] = round(float(e[m - 1]), 4)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["eval50", "train"], help="which dataset the episodes come from")
    ap.add_argument("--dataset", required=True, help="LeRobot root (datasets/isg-init-eval or datasets/isg-init)")
    ap.add_argument("--instances", required=True, help="episode -> instance map (lab/instances.json or lab/instances-train.json)")
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--horizon", type=int, default=1500)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--settle-steps", type=int, default=50)
    ap.add_argument("--lab-out", required=True)
    ap.add_argument("--raw-out", required=True)
    ap.add_argument("--video-size", type=int, default=540)
    ap.add_argument("--no-stop-on-success", action="store_true")
    ap.add_argument("--keep-depth", action="store_true")
    ap.add_argument("--include-unverified", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-planner", action="store_true", help="evaluator patch: no cuRobo planner (streams never plan)")
    ap.add_argument("--task", default=TASK, help="the instances' task (default relocalization; other tasks: articulation / alignment scoring)")
    ap.add_argument("--fixed-handle-ag", dest="fixed_handle_ag", action="store_true", default=True, help="evaluator patch (default on since 2026-10-08): the oracle's FixedJoint assisted grasp for links behind movable joints")
    ap.add_argument("--stock-handle-ag", dest="fixed_handle_ag", action="store_false", help="OmniGibson's stock assisted-grasp joint rule")
    args = ap.parse_args(argv)
    return run_replay_job(args, shared={}, shutdown=True)


def run_replay_job(args, shared=None, shutdown=True) -> int:
    """The replays of one job in this process; ``shared`` carries the loaded world across jobs (persistent worker)."""
    import numpy as np
    import torch as th

    from lab_tools import evaluator_patches
    from lab_tools import replay_policy as RP
    from lab_tools.eval_episodes import git_head, memory_snapshot, score_relocalization, sha256_of

    evaluator_patches.install(keep_depth=getattr(args, "keep_depth", False), no_planner=bool(getattr(args, "no_planner", False)), fixed_handle_ag=bool(getattr(args, "fixed_handle_ag", False)))
    from ispatialgym.oracle import run_isg as R

    inst_doc = json.load(open(args.instances))
    inst_map = inst_doc.get("episodes", inst_doc)
    buckets = {}
    csv_path = os.path.join(args.dataset, "isg_meta", "camera_relocalization_distances.csv")
    if os.path.exists(csv_path):
        import csv

        with open(csv_path) as f:
            for r in csv.DictReader(f):
                buckets[int(r["episode_index"])] = r
    os.makedirs(args.lab_out, exist_ok=True)
    os.makedirs(args.raw_out, exist_ok=True)
    task_name = getattr(args, "task", None) or TASK
    todo = []
    for e in args.episodes:
        v = inst_map.get(str(e))
        if v is None or v.get("task") != task_name or not v.get("instance"):
            print(f"REPLAY episode {e}: no relocalization instance in {args.instances}", flush=True)
            continue
        if not v.get("verified") and not args.include_unverified:
            print(f"REPLAY episode {e}: instance not first-frame verified; skipped (--include-unverified runs it)", flush=True)
            continue
        if not args.overwrite and os.path.exists(os.path.join(args.lab_out, f"episode_{e:06d}.json")):
            continue
        todo.append((e, v))
    print(f"REPLAY split={args.split} dataset={args.dataset} episodes={args.episodes} todo={[e for e, _ in todo]}", flush=True)
    if not todo:
        return 0
    provenance = {"evaluator_commit": git_head(WT), "evaluator_root": WT, "b1k_commit": git_head(os.path.join(WT, "Behavior-1K")),
                  "bridge_sha256": {f: sha256_of(os.path.join(WT, "lab_tools", f)) for f in ("replay_policy.py", "replay_episodes.py", "evaluator_patches.py")}}
    import omnigibson as og

    provenance["omnigibson_version"] = og.__version__
    shared = {} if shared is None else shared
    t_proc = time.time()
    for k, (e, v) in enumerate(todo):
        t0 = time.time()
        seed = int(v["instance_index"]) if v.get("instance_index") is not None else e
        th.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        random.seed(seed)
        record = {"run": "replay", "step": None, "checkpoint": None, "split": "replay", "replay_split": args.split, "dataset": os.path.basename(os.path.normpath(args.dataset)),
                  "task": task_name, "episode_index": e, "instance": v["instance"], "instance_index": v.get("instance_index"), "seed": seed, "success": None, "steps": None,
                  "horizon": args.horizon, "wall_s": None, "evaluator_commit": provenance["evaluator_commit"], "act_commit": None, "server_args": None, "error": None,
                  "score": None, "tag": args.tag, "stop_on_success": not args.no_stop_on_success, "instance_verified": v.get("verified"),
                  "first_frame_mean_abs_diff": v.get("first_frame_mean_abs_diff"), "started_at": utc(), "finished_at": None, "provenance": provenance,
                  "policy": "lab_tools.replay_policy:ReplayPolicy (recorded actions, open loop)", "fixed_handle_ag": bool(getattr(args, "fixed_handle_ag", False))}
        try:
            actions, states, length = load_episode_arrays(args.dataset, e)
            ctx = {"actions": actions, "horizon": args.horizon, "stop_on_success": not args.no_stop_on_success, "shared": shared}
            RP.CONTEXT = ctx
            argv_r = ["--instance", v["tro"], "--mode", "policy", "--policy", "lab_tools.replay_policy:ReplayPolicy", "--policy-steps", str(args.horizon),
                      "--max-steps", str(args.max_steps), "--settle-steps", str(args.settle_steps), "--out", args.raw_out, "--record", "video", "--video",
                      "--video-size", str(args.video_size), "--demo-gating", "none", "--no-goal-override", "--log-level", "WARNING"]
            rargs = R.parse_args(argv_r)
            if shared.get("env") is not None:
                shared["template_dir"] = os.path.dirname(os.path.dirname(os.path.abspath(v["tro"])))
            rc, result = R.run_instance(rargs, v["tro"], shared)
            out_dir = result.get("out")
            meta = {}
            if out_dir and os.path.exists(os.path.join(out_dir, "meta.json")):
                with open(os.path.join(out_dir, "meta.json")) as f:
                    meta = json.load(f)
            record.update({
                "success": bool(result.get("success")), "steps": int(ctx.get("steps") or result.get("n_action_steps") or 0), "stop_reason": ctx.get("stop_reason"),
                "exit_code": rc, "status": result.get("status"), "reason": result.get("reason"), "n_steps_total": result.get("n_steps"),
                "n_action_steps": result.get("n_action_steps"), "first_success_step": meta.get("first_success_step"), "binary_success_ever": meta.get("binary_success_ever"),
                "binary_success_final": meta.get("binary_success_final"), "settled_success": meta.get("settled_success"), "settle_windows": meta.get("settle_windows"),
                "video": os.path.relpath(meta["video"], WT) if meta.get("video") else None, "raw_dir": os.path.relpath(out_dir, WT) if out_dir else None,
                "timings": meta.get("timings"),
            })
            if result.get("status") == "error" or rc == R.EXIT_ERROR:
                record["error"] = result.get("error") or "evaluator error"
            ep = {"bucket_hi_m": float(buckets[e]["bucket_hi_m"]) if e in buckets else None, "tro": v["tro"]}
            if task_name == TASK:
                score = score_relocalization(meta, shared, ep)
            elif task_name.startswith("alignment"):
                from lab_tools.eval_episodes import score_alignment

                score = score_alignment(meta)
            elif task_name.startswith(("mental_rotation", "object_scaling")):
                from lab_tools.eval_episodes import score_pose_target

                score = score_pose_target(meta)
            else:
                from lab_tools.eval_episodes import score_articulation

                score = score_articulation(meta)
            try:
                from lab_tools.eval_episodes import base_and_trunk_vs_goal, closest_approach

                score["closest_approach"] = closest_approach(ctx.get("step_errors"))
                score["base_trunk_vs_goal"] = base_and_trunk_vs_goal(shared, ep, [np.asarray(a) for a in actions[: int(ctx.get("actions_streamed") or 0)]])
                if ctx.get("step_errors") and result.get("out"):
                    np.save(os.path.join(result["out"], "errors.npy"), np.asarray(ctx["step_errors"], dtype=np.float32))
            except Exception as ex:  # noqa: BLE001
                score["closest_approach_error"] = repr(ex)
            qa = meta.get("qa") or {}
            score["native"] = {"max_position_error": meta.get("max_position_error"), "max_orientation_error": meta.get("max_orientation_error"),
                               "errors_final_step": meta.get("errors_final_step"), "reason": result.get("reason"), "verdict": meta.get("verdict"),
                               "contacts": (qa.get("collisions") or {}).get("n_events") if isinstance(qa.get("collisions"), dict) else result.get("collisions")}
            # the replayed base trajectory vs the demo's own (dead-reckoned from its recorded base velocities, from the restored init pose)
            rep = ctx.get("base_xyyaw") or []
            init = meta.get("robot_base_init") or {}
            if rep:
                x0, y0, yaw0 = rep[0]
            else:
                p = init.get("position") or [0, 0, 0]
                x0, y0, yaw0 = float(p[0]), float(p[1]), RP._yaw(init.get("orientation") or [0, 0, 0, 1])
            streamed = int(ctx.get("actions_streamed") or 0)
            replay_qvel = np.asarray(ctx.get("base_qvel") or np.zeros((0, 3)), dtype=np.float64)
            traj = trajectory_block(rep, replay_qvel, states[:, 0:3], x0, y0, yaw0, actions=actions, streamed=streamed)
            traj["end_vs_goal_base"] = end_vs_goal(rep, goal_base_pose(v["tro"]))
            demo_dr = traj.pop("demo_dead_reckoned_raw")
            demo_cal = traj.pop("demo_dead_reckoned_calibrated", None)
            final_cam = (ctx.get("camera_pose") or [None])[-1]
            score["replay"] = {"demo_length": length, "demo_program_length_note": "the recorded episode = oracle program + its settle window (isg_meta/settle_windows.json)",
                               "actions_streamed": streamed, "stop_reason": ctx.get("stop_reason"), "trajectory": traj,
                               "init_pose_source": "simulator pose at the first callable call" if rep else "meta.robot_base_init", "replay_end_camera_pose": final_cam}
            record["score"] = score
            record.update(headline_deviation(traj))
            record["demo_length"] = length
            record["actions_streamed"] = streamed
            if out_dir:
                np.save(os.path.join(out_dir, "replay_base_xyyaw.npy"), np.asarray(rep, dtype=np.float32))
                np.save(os.path.join(out_dir, "replay_base_qvel.npy"), replay_qvel.astype(np.float32))
                np.save(os.path.join(out_dir, "replay_camera_pose.npy"), np.asarray(ctx.get("camera_pose") or np.zeros((0, 7)), dtype=np.float32))
                np.save(os.path.join(out_dir, "demo_base_xyyaw_dead_reckoned.npy"), np.asarray(demo_dr, dtype=np.float32))
                if demo_cal is not None:
                    np.save(os.path.join(out_dir, "demo_base_xyyaw_dead_reckoned_calibrated.npy"), np.asarray(demo_cal, dtype=np.float32))
                np.save(os.path.join(out_dir, "demo_base_qvel.npy"), np.asarray(states[:, 0:3], dtype=np.float32))
                if ctx.get("joint_q"):
                    np.save(os.path.join(out_dir, "replay_joint_q.npy"), np.asarray(ctx["joint_q"], dtype=np.float32))
                if ctx.get("object_poses"):
                    np.save(os.path.join(out_dir, "replay_object_poses.npy"), np.asarray(ctx["object_poses"], dtype=np.float32))
                    with open(os.path.join(out_dir, "replay_objects.json"), "w") as f:
                        json.dump(ctx.get("object_names") or [], f)
                    record["replay_objects"] = ctx.get("object_names")
        except Exception as ex:  # noqa: BLE001
            record["error"] = repr(ex)
            record["traceback"] = traceback.format_exc()
            traceback.print_exc()
        record["memory"] = memory_snapshot()
        record["wall_s"] = round(time.time() - t0, 1)
        record["finished_at"] = utc()
        name = f"episode_{e:06d}.json"
        with open(os.path.join(args.lab_out, name + ".tmp"), "w") as f:
            json.dump(record, f, indent=1, default=str)
        os.replace(os.path.join(args.lab_out, name + ".tmp"), os.path.join(args.lab_out, name))
        sc = record.get("score") or {}
        tr = (sc.get("replay") or {}).get("trajectory") or {}
        print(f"REPLAY_EPISODE {k + 1}/{len(todo)} ep={e} inst={v['instance'].rsplit('__', 1)[-1]} success={record['success']} reason={record.get('reason')} "
              f"steps={record['steps']} stop={record.get('stop_reason')} first_success={record.get('first_success_step')} wall={record['wall_s']}s "
              f"geodesic={sc.get('geodesic_distance_m')} pos={sc.get('position_error_m')} ori_deg={sc.get('orientation_error_deg')} "
              f"max_dev={tr.get('max_xy_deviation_m')} err={record['error']}", flush=True)
    print(f"REPLAY_DONE split={args.split} episodes={len(todo)} wall={round(time.time() - t_proc, 1)}s", flush=True)
    if shutdown:
        try:
            og.shutdown()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
