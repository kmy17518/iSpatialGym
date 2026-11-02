#!/usr/bin/env python3
"""Write lab/env.json: evaluator and ACT commits, OmniGibson / Isaac Sim versions, horizons, the (single) action horizon,
measured minutes per episode and the number of parallel evaluators.

    python lab_tools/write_env.py --minutes-per-episode 3.2 --parallel-evaluators 1 [--note "..."]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = [
    "camera_relocalization-standard",
    "configuration_matching-articulation_open_large_scale-dishwasher",
    "configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table",
    "alignment-axial-board_game-breakfast_table",
]


def git(root, *a):
    try:
        return subprocess.run(["git", "-C", root, *a], capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def pip_version(python, pkg):
    try:
        out = subprocess.run([python, "-m", "pip", "show", pkg], capture_output=True, text=True, timeout=60).stdout
        for line in out.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        return None


def measured_episode_stats():
    """minutes per episode and steps/s from every finished episode record under lab/ (smoke + tierB)."""
    walls, rates = [], []
    for p in glob.glob(os.path.join(WT, "lab", "**", "episode_*.json"), recursive=True):
        try:
            r = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        if r.get("error") or not r.get("wall_s"):
            continue
        walls.append(r["wall_s"])
        t = (r.get("timings") or {})
        if t.get("episode_s") and r.get("n_steps_total"):
            rates.append(r["n_steps_total"] / t["episode_s"])
    out = {"episodes_measured": len(walls)}
    if walls:
        walls.sort()
        out.update({"minutes_per_episode_mean": round(sum(walls) / len(walls) / 60, 2), "minutes_per_episode_median": round(walls[len(walls) // 2] / 60, 2),
                    "minutes_per_episode_max": round(walls[-1] / 60, 2)})
    if rates:
        out["sim_steps_per_s_mean"] = round(sum(rates) / len(rates), 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes-per-episode", type=float, default=None)
    ap.add_argument("--parallel-evaluators", type=int, default=1)
    ap.add_argument("--action-horizon", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=1500, help="the queue's horizon_steps of the current phase (relocalization phase: 1500)")
    ap.add_argument("--gpu-memory", default=None, help="measured GPU memory note")
    ap.add_argument("--note", default=None)
    ap.add_argument("--vectorized", type=int, default=0, help="> 0: the vectorized evaluator with this many environments per process (lab_tools/vec_eval.py on the lab-vec-eval branch)")
    ap.add_argument("--switched-at", default=None, help="when the queue switched to the vectorized evaluator (UTC)")
    args = ap.parse_args(argv)
    conda_py = os.path.expanduser("~/miniconda3/envs/ispatialgym-v2.0-vectorized/bin/python")
    act_py = os.path.join(WT, "act-isg", ".venv", "bin", "python")
    stats = measured_episode_stats()
    mpe = args.minutes_per_episode if args.minutes_per_episode is not None else stats.get("minutes_per_episode_mean")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        gpu = None
    env = {
        "format": "isg-lab-env/v1",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "gpu": gpu,
        "evaluator": {
            "repo": "git@github.com:kmy17518/iSpatialGym.git", "branch": "v2.0-vectorized", "commit": git(WT, "rev-parse", "HEAD"),
            "entry": "ispatialgym.oracle.run_isg --mode policy (one world per task, instances attached / restored in turn)",
            "process_model": "since 2026-10-08 ~11:10Z: persistent evaluator workers (lab_tools/eval_worker.py, one Isaac Sim process per slot, the loaded world kept across queue "
                             "entries and jobs; the ACT server alone is restarted per checkpoint / server_args); 2 workers (environments per process: 1); a worker restarts itself "
                             "(logged) above 20 GB of GPU memory. Before: one eval_episodes.py process per queue entry and per 10 episodes (a scene load each).",
            "patches": ["InstanceLoader.behavior_output forced on (BEHAVIOR sensor set, no LeRobot shard)", "no depth_linear annotators", "stream QA: timeout verdict only (contacts counted)",
                        "epilogue: 10-step no-op window when the goal never held", "NoPlanner (2026-10-08): no cuRobo planner / collision world in policy and replay streams (streams never plan; "
                        "-27 s and ~5 GB of GPU memory per process)", "after_episode_cleanup (2026-10-08): gc.collect + torch.cuda.empty_cache between episodes"],
            "environments_per_process": 1, "processes": 2,
            "bridge": {f: sha(os.path.join(WT, "lab_tools", f)) for f in ("act_policy.py", "eval_episodes.py", "evaluator_patches.py", "replay_policy.py", "tierb_loop.py", "vec_eval.py", "eval_worker.py")
                       if os.path.exists(os.path.join(WT, "lab_tools", f))},
            "behavior_1k": {"repo": "https://github.com/kmy17518/BEHAVIOR-1K.git", "branch": "v2.0-vectorized", "commit": git(os.path.join(WT, "Behavior-1K"), "rev-parse", "HEAD")},
            "omnigibson_version": pip_version(conda_py, "omnigibson"), "isaacsim_version": pip_version(conda_py, "isaacsim"),
            "torch_version": pip_version(conda_py, "torch"), "bddl_version": pip_version(conda_py, "bddl"), "curobo_version": pip_version(conda_py, "nvidia-curobo"),
        },
        "act": {"repo": "https://github.com/kmy17518/act.git", "branch": "isg-wave2-dev", "commit": git(os.path.join(WT, "act-isg"), "rev-parse", "HEAD"),
                "server": "scripts/b1k/serve_b1k.py --port 8000 --device cuda (default action horizon)", "torch_version": pip_version(act_py, "torch")},
        "protocol": {"name": "behavior-numpy-msgpack websocket", "connection": "one per episode", "control_hz": 30,
                     "observation": ["robot_r1::proprio (61, raw)", "robot_r1::robot_r1:{zed_link,left_realsense_link,right_realsense_link}:Camera:0::rgb (uint8 HWC native: 720/480/480)",
                                     "goal::robot_r1::robot_r1:zed_link:Camera:0::rgb (the episode's goal PNG, 720x720)", "task_id = train_task_index"],
                     "action": "23-D as the dataset's `action`, applied through run_isg's executor (JoyLo controller config of the demos)"},
        "action_horizon": args.action_horizon,
        "horizon_steps": {TASKS[0]: args.horizon},
        "horizon_note": f"relocalization phase (queue isg-tierB-queue/v2 horizon_steps, from 2026-10-07): {args.horizon} policy steps ({args.horizon / 30.0:.0f} s at 30 Hz) "
                        "for every entry (replay, train, eval50); recorded per episode as `horizon`. Wave 1 (2026-10-04/05, four tasks, lab/tierB/*/step_00050000/summary.json) "
                        "used run_isg's default 3000. Horizons are never mixed within one comparison. The stream also ends at the task's first official_success (stop on success).",
        "horizon_history": [{"phase": "wave1", "tasks": TASKS, "horizon_steps": 3000, "until": "2026-10-07"}, {"phase": "relocalization", "tasks": [TASKS[0]], "horizon_steps": args.horizon, "from": "2026-10-07"}],
        "success_criterion": {
            TASKS[0]: {
                "goal": "BDDL (:goal (and (pose_reached ?agent.n.01_1) (pose_settled ?agent.n.01_1))) of camera_relocalization problem18 (variant camera_relocalization-standard)",
                "binary_goal": "pose_reached(agent): OmniGibson PoseReached on the robot's head camera sensor `robot_r1:zed_link:Camera:0`, world frame, against the TRO pose target "
                               "(ispatialgym.pose_targets['agent.n.01_1']: the goal camera pose = the camera pose of the demo's goal view)",
                "position_error_m": "full 3-D Euclidean norm |p_camera - p_goal| (position_metric 'full', pos_axes None)",
                "orientation_error_rad": "full quaternion angular distance 2*acos(|<q_camera, q_goal>|) (orientation_metric 'full', no equivalences; the record's "
                                         "orientation_error_deg is this angle in degrees)",
                "position_tolerance_m": 0.05,
                "orientation_tolerance_rad": 0.1,
                "orientation_tolerance_deg": round(math.degrees(0.1), 3),
                "within_tolerance": "position_error <= 0.05 m AND orientation_error <= 0.1 rad (both inclusive; PoseReached._within_tolerance); this is `score.within_tolerance`, read "
                                    "from errors at the LAST step of the episode (after the epilogue)",
                "errors_not_in_the_rule": "geodesic_distance_m (trav-map shortest path camera xy -> goal camera xy) and heading_error_deg (yaw of goal^-1 * current) are reported "
                                          "only; distance_bucket_m is the dataset CSV's bucket",
                "pose_settled": "PoseSettled(agent) with DEFAULT_POSE_SETTLED_CONFIG n_steps 45, pos 0.002 m, ori 0.005 rad, joint 0.005 rad (configuration drift over the "
                                "window); informational (require_pose_settled False): it enters neither official_success nor the executor's verdict (ExecutorConfig."
                                "require_task_settled False)",
                "official_success_stream": "first step t with the binary goal true (first_success_step); the policy stream stops there (stop_on_success) or at the horizon",
                "epilogue": {
                    "no_op": "the last streamed action repeated with the base velocity command zeroed (Executor._noop_actions); arms / trunk hold their last commanded "
                             "position targets",
                    "goal_held_at_some_step": "window_1 = max(50, ceil(0.20 * n_streamed_steps)) no-op steps; if over window_1 the robot moved (base > 0.005 m or > 0.01 rad, "
                                              "any joint > 0.02 rad) or the binary goal does not hold at its end: window_2 = 2 * window_1 more no-op steps, and the motion over "
                                              "window_2 + the state at its end decide",
                    "goal_never_held": "evaluator patch (lab_tools, recorded in every episode): one 10-step no-op window, no second window; the verdict is not_reached "
                                       "(or tolerance) either way, the final errors are read where the policy left the robot",
                    "verdicts": "not_reached (binary goal never held), tolerance (gating pose target out of tolerance at the last step), not_settled (motion over the final "
                                "window or the goal lost at the last step), timeout (QA: > QAConfig.max_action_steps 3000 stream steps or > max_steps 6000 episode steps)",
                    "success": "success == True iff no verdict: the binary goal held at some step, holds at the last step (within tolerance), and the robot was still over the "
                               "final no-op window",
                },
                "binary_success_ever": "the binary goal held at some step of the episode (stream or epilogue)",
                "first_success_step": "env step index (1-based count of env.step calls since the restore) of the first binary success; -1 / null when never",
                "stream_qa": "during policy / replay streams only the timeout verdict ends an episode; robot-object contacts are counted (score.native.contacts) and never fail it "
                             "(machine 1, 2026-10-04T18:33:48Z)",
            },
        },
        "goal_images": {
            "views": "one goal image per view in the server metadata's goal_views / goal_observation_keys (zed_link for most checkpoints; zed_link + left_realsense_link + "
                     "right_realsense_link for the 'wrist' runs trained with --goal-views ... --goal-source episode_last); the bridge refuses to start an episode when a view is missing",
            "zed_link": "the dataset's isg_meta/goals/<task>/episode_XXXXXX.png (720 x 720, machine 1's decode of the episode's last head frame)",
            "wrist": "lab_tools/wrist_goals.py: the episode's last recorded frame of observation.rgb.<camera>_camera_0 at native 480 x 480, decoded like b1k_dataset.B1KDataset._goal_frame / "
                     "_decode for episode_last goals (PyAV: seek backward to the row timestamp + the stream's from_timestamp, nearest pts, stop at the first frame at or after); "
                     "PNGs under datasets/goals/<dataset>/<camera>/episode_XXXXXX.png with index.json (frame, timestamp, sha256)",
            "bit_exact_check": "lab_tools/check_wrist_goals.py: quantize_image(preprocess_image(PNG, adapter image_size 224)) == B1KDataset(goal_views=3 cameras, goal_source=episode_last)"
                               ".goal_for(position)[view] on the lab machine: wrist PNGs equal for 100/100 train and 50/50 eval50 relocalization episodes and both cameras; the zed PNGs "
                               "(made on machine 1) differ from the lab machine's decode of the same frame by at most 2/255 per channel (mean ~0.55/255, ~90 % of pixels): a PyAV / "
                               "ffmpeg yuv->rgb build difference between machines, so the lab's wrist goals sit within that same tolerance of what the training machines decoded",
        },
        "replay_positive_control": {
            "actions": "the LeRobot `action` column of the episode (23-D: base vx, vy, wz in [-1, 1] of the HolonomicBaseJointController velocity command "
                       "(output limits BASE_VEL_LIMITS, robot frame); trunk 4 + arms 2x7 absolute joint position targets; grippers 2x1), streamed open-loop at 30 Hz "
                       "through the same run_isg executor path as the policy (lab_tools.replay_policy), same stop-on-success and epilogue",
            "base_trajectory_deviation": "max over streamed steps of |base_xy(replay, t) - base_xy(demo, t)|, the demo trajectory dead-reckoned from its own "
                                         "observation.state base_qvel (robot-frame vx, vy, wz = 61-D proprio [0:3]) at 1/30 s from the restored init pose, the yaw rate "
                                         "divided by the scale measured in the same replay between the proprio yaw-rate sample and the finite-difference yaw rate of the "
                                         "simulated base (the rz virtual-joint velocity over-reads by 3-17 % and reads ~+0.01 rad/s while driving straight; the linear part "
                                         "is exact to ~1 cm/s). Reported with the method's floor (the same integration of the replay's own recorded velocities vs its "
                                         "simulated path) and with the exact end-point checks: final camera pose vs the goal camera pose (score.position_error_m / "
                                         "orientation_error_deg) and final base pose vs the instance's goal base pose (base_end_vs_goal_base_m). Details in "
                                         "score.replay.trajectory of each replay record.",
            "stop_on_success_effect": "the stream stops at the FIRST step within tolerance and the epilogue holds the last streamed action (base stopped, trunk / arm "
                                      "targets of that action), so a successful replay (or policy) ends near the tolerance boundary: the 5 eval50 replays end at 4.7-4.9 cm "
                                      "/ 2.6-5.3 deg although the demos' own remaining actions would bring the camera to 0.2-1.7 cm / 0.4-1.5 deg "
                                      "(lab/diagnostics/replay/eval50/fullstream/). Final errors of successful episodes are therefore not a precision measure.",
        },
        "episode_rules": {
            "stop_on_success": True,
            "epilogue": "executor default (no-op window max(50, 20% of steps), then a 2x second window if the goal does not hold / something moved) when the binary goal held at some step of the stream; a 10-step no-op window when it never held (the verdict is not_reached / tolerance either way; final errors read where the policy left the robot)",
            "simulation": "OMNIGIBSON_TRACK_STATE_UPDATES=0 (per-step change detection only feeds the HDF5 recorder, not used), depth_linear annotators off (the policy consumes RGB + proprio); physics 120 Hz, control 30 Hz, n_render_iterations 1",
            "qa_verdicts_during_policy_stream": "timeout only; contacts are counted (score.native.contacts) but do not end the episode",
            "success": "run_isg epilogue verdict: binary goal reached, within tolerance, settled (not_reached / tolerance / not_settled otherwise)",
            "seed": "per instance: its instance index (torch / numpy / random), same order (episode_index) for every checkpoint",
            "sensors": "BEHAVIOR replay_obs set (behavior_lerobot.robot_overrides): head zed_link 720 px 40 mm, wrists 480 px, rgb + depth_linear + proprio",
            "video": "head camera mp4 per episode (run_isg --video, 540 px) under runlogs/tierB/<run>/step_XXXXXXXX/<task>/<instance>/head_rgb.mp4 on the lab machine",
        },
        "scores": {
            TASKS[0]: "at the last step of the episode: geodesic_distance_m (trav-map shortest path from the final head-camera xy to the goal camera xy; euclidean fallback flagged in "
                      "geodesic_source), position_error_m and orientation_error_deg (the evaluator's own PoseReached errors, see success_criterion), within_tolerance "
                      "(their tolerance test), heading_error_deg (yaw of goal^-1 * current), distance_bucket_m (dataset CSV)",
            TASKS[1]: "angle_error_deg = |final - target| of the target's revolute joint (door), final_angle_deg, target_angle_deg",
            TASKS[2]: "angle_error_deg = |final - target| of the target's revolute joint (lid), final_angle_deg, target_angle_deg",
            TASKS[3]: "alignment_error_deg (PoseReached orientation error of the board game), steps_to_success (first_success_step)",
            "native": "score.native: errors_final_step, verdict, contacts",
        },
        "minutes_per_episode": mpe,
        "parallel_evaluators": args.parallel_evaluators,
        "vectorized": None,
        "measured": stats,
        "gpu_memory_note": args.gpu_memory,
        "instances": "lab/instances.json (test set: regenerated with sample_isg seed 1003, first-frame verified); lab/instances-train.json (training episodes of "
                     "kmy17518/isg-init mapped to recovered instances, first-frame verified the same way)",
        "splits": {"eval50": "kmy17518/isg-init-eval episodes (lab/instances.json) -> lab/tierB/<run>/step_XXXXXXXX/camera_relocalization-standard/",
                   "train": "kmy17518/isg-init training episodes (lab/instances-train.json) -> lab/tierB-train/<run>/step_XXXXXXXX/camera_relocalization-standard/",
                   "replay": "open-loop replay of the recorded actions -> lab/diagnostics/replay/<eval50|train>/",
                   "summary": "one summary.json per (run, step, split) in the same directory as its episode files"},
        "note": args.note,
    }
    os.makedirs(os.path.join(WT, "lab"), exist_ok=True)
    if args.vectorized and args.vectorized > 0:
        n = int(args.vectorized)
        gates = {}
        for name, path in (("3a", "lab/diagnostics/vec-eval/3a_unit_tests.txt"), ("3b", "lab/diagnostics/vec-eval/3b/summary.json"), ("3c", "lab/diagnostics/vec-eval/3c/summary.json"),
                           ("3d", "lab/diagnostics/vec-eval/3d/summary.json"), ("3e", "lab/diagnostics/vec-eval/3e/summary.json"), ("3g", "lab/diagnostics/vec-eval/3g/summary.json")):
            full = os.path.join(WT, path)
            if os.path.exists(full):
                try:
                    gates[name] = {"file": path, "pass": json.load(open(full)).get("pass")} if path.endswith(".json") else {"file": path}
                except Exception:  # noqa: BLE001
                    gates[name] = {"file": path}
        env["evaluator"].update({
            "branch": git(WT, "rev-parse", "--abbrev-ref", "HEAD"),
            "pinned_commit_before": "221c6e03b0403ba9d3662f9197248966f2e5ae5b",
            "entry": "lab_tools/vec_eval.py (ispatialgym.oracle.vec_loader.VecInstanceLoader + vec_stream.VecStreamExecutor + tasks.vec_task.VecISpatialGymTask): one og.Environment with "
                     f"{n} cloned scenes, one batched env.step per control step, every environment with its own instance, scope, targets, stream, QA, recorder, verdict and ACT websocket session",
            "process_model": f"since {args.switched_at or env['updated_at']}: one persistent evaluator worker (lab_tools/eval_worker.py, vectorized jobs) holding a {n}-environment world across queue "
                             "entries and jobs; batches of up to N episodes of one task per restore; the single-env path (environments_per_process 1) stays selectable with "
                             "tierb_loop.py --num-envs 1 and is not routed to by default",
            "environments_per_process": n, "processes": 1,
            "vectorized": {
                "rendering": "per-camera render products (tiled rendering off: the tiles share one exposure), 100 warm renders after a batch's reset before the first observation",
                "clone_frames": "clones are translated copies (+x, identity rotation); object root positions, the robot base, world-frame pose targets and hinge anchors are translated into the clone's "
                                "frame, orientations untouched; everything reported is in the base scene's frame",
                "exactness": "num_envs = 1 is bit-identical to the single-env evaluator (replays: base, camera, errors, proprio equal); with 2-4 clones trajectories deviate <= 1.3 mm / 0.7 mrad over "
                             "an episode (float32 world coordinates at 63-190 m offsets; PhysX moves the base scene by <= 0.1 mm once other scenes exist), counters / verdicts / first success "
                             "steps identical in every replay; slot results are independent of the neighbours' episodes (bit-identical slot 0 across neighbour sets)",
                "gates": gates,
            },
        })
        env["vectorized"] = {"num_envs": n, "switched_at": args.switched_at}

    with open(os.path.join(WT, "lab", "env.json"), "w") as f:
        json.dump(env, f, indent=1)
    print(json.dumps({k: env[k] for k in ("action_horizon", "minutes_per_episode", "parallel_evaluators", "measured")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
