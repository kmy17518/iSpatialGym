#!/usr/bin/env python3
"""Tier B: closed-loop episodes of one task against a running ACT policy server, through ``run_isg --mode policy``.

One process = one task (one template, one loaded world; every instance is attached / restored in turn as in ``--queue``).
For each test episode of the task (``lab/instances.json``: episode -> regenerated instance, verified by its first frame):

1. the instance is restored, the policy (:mod:`lab_tools.act_policy`) opens a new websocket session, sends the raw
   61-D proprioception, the three native-size RGB images and the episode's goal PNG every control step and applies
   the returned 23-D action through the executor (the same controller configuration as the demos);
2. the stream ends at the horizon or at the task's first ``official_success``; the executor's epilogue (no-op settle
   window, tolerance / settled checks) decides ``success``;
3. the episode record goes to ``<lab-out>/<task>/episode_XXXXXX.json`` and the head-camera video to
   ``<raw-out>/<iid>/head_rgb.mp4`` (``run_isg --record video --video``).

Evaluator settings that differ from the oracle's QA (recorded in every episode and in lab/env.json):
* stream QA verdicts: only ``timeout``; contacts are counted (``score.native.contacts``) but do not end a policy
  episode (BEHAVIOR-challenge style);
* epilogue: the default windows (max(50, 20 %) + a 2x second window) when the binary goal held at some step; when the
  goal never held during the stream the second window is skipped (the verdict is ``not_reached`` either way).

    python lab_tools/eval_episodes.py --run R --step 50000 --ckpt-repo ... --ckpt-file ... --ckpt-sha256 ... \
        --server ws://127.0.0.1:8000 --task <alias> --instances lab/instances.json --dataset datasets/isg-init-eval \
        --lab-out lab/tierB/R/step_00050000 --raw-out runlogs/tierB/R/step_00050000 [--episodes 0 1 ...] [--horizon 3000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# OmniGibson's per-step tensorized-state change detection only feeds BaseObject.is_active, which only the HDF5 recorder's
# dump filter reads; the evaluator records video only, and the detection costs ~half of an env step (bc7180af).
os.environ.setdefault("OMNIGIBSON_TRACK_STATE_UPDATES", "0")

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = {
    "camera_relocalization-standard": "reloc",
    "configuration_matching-articulation_open_large_scale-dishwasher": "dishwasher",
    "configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table": "blender",
    "alignment-axial-board_game-breakfast_table": "board_game",
}


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head(root: str):
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_goal(png_path: str):
    import numpy as np
    from PIL import Image

    return np.ascontiguousarray(np.asarray(Image.open(png_path).convert("RGB"), dtype=np.uint8))


def yaw_error_deg(q_cur, q_goal) -> float:
    """Heading error: the yaw of the relative rotation goal^-1 * current (deg)."""
    import torch as th

    import omnigibson.utils.transform_utils as T

    q_cur = th.as_tensor(q_cur, dtype=th.float32).reshape(4)
    q_goal = th.as_tensor(q_goal, dtype=th.float32).reshape(4)
    rel = T.quat_multiply(T.quat_inverse(q_goal), q_cur)
    rel = rel / th.norm(rel)
    euler = T.quat2euler(rel).reshape(-1)
    return abs(math.degrees(float(euler[2])))


def relative_ypr_deg(q_cur, q_goal) -> dict:
    """Yaw / pitch / roll (deg) of the relative rotation goal^-1 * current, expressed in the GOAL camera frame (USD camera:
    optical axis -Z, up +Y): about the camera's up axis (Y: pan / yaw), its right axis (X: tilt / pitch) and its optical
    axis (Z: roll). Also the world-frame z-up yaw of the same rotation."""
    import torch as th

    import omnigibson.utils.transform_utils as T

    q_cur = th.as_tensor(q_cur, dtype=th.float32).reshape(4)
    q_goal = th.as_tensor(q_goal, dtype=th.float32).reshape(4)
    rel = T.quat_multiply(T.quat_inverse(q_goal), q_cur)
    rel = rel / th.norm(rel)
    e = T.quat2euler(rel).reshape(-1)  # intrinsic xyz of the goal-camera-frame rotation
    rel_world = T.quat_multiply(q_cur, T.quat_inverse(q_goal))  # the same rotation seen in the world frame
    ew = T.quat2euler(rel_world / th.norm(rel_world)).reshape(-1)
    return {"pan_deg": math.degrees(float(e[1])), "tilt_deg": math.degrees(float(e[0])), "roll_deg": math.degrees(float(e[2])),
            "world_yaw_deg": math.degrees(float(ew[2])), "world_pitch_deg": math.degrees(float(ew[1])), "world_roll_deg": math.degrees(float(ew[0])),
            "total_deg": math.degrees(2.0 * math.acos(min(1.0, abs(float(th.sum(q_cur * q_goal))))))}


def closest_approach(step_errors) -> dict | None:
    """Minimum camera position error over the stream (and the orientation error at that step), and the minimum orientation error."""
    import numpy as np

    if not step_errors:
        return None
    a = np.asarray(step_errors, dtype=np.float64)
    pos, ori = a[:, 0], a[:, 1]
    ok = ~np.isnan(pos)
    if not ok.any():
        return None
    ip = int(np.nanargmin(pos))
    io = int(np.nanargmin(ori)) if (~np.isnan(ori)).any() else None
    both = np.where((pos <= 0.05) & (ori <= 0.1))[0]
    return {"min_position_error_m": float(pos[ip]), "step": ip, "orientation_error_at_min_pos_deg": None if np.isnan(ori[ip]) else math.degrees(float(ori[ip])),
            "min_orientation_error_deg": None if io is None else math.degrees(float(ori[io])), "min_orientation_step": io,
            "position_error_at_min_ori_m": None if io is None else float(pos[io]),
            "first_step_within_tolerance": int(both[0]) if len(both) else None, "n_steps_within_tolerance": int(len(both)), "n_steps": int(len(pos))}


def base_and_trunk_vs_goal(shared: dict, ep: dict, actions) -> dict:
    """Final base pose vs the instance's goal base pose (sidecar stance 'goal'); final measured trunk joints and the last
    commanded trunk targets vs the goal trunk configuration (torso_q(translate, tilt))."""
    import numpy as np

    from ispatialgym.robot.conventions import torso_q

    out: dict = {}
    robot = shared.get("robot")
    tro = ep.get("tro")
    side = tro.replace("-tro_state.json", ".oracle.json") if tro else None
    goal_stance = None
    if side and os.path.exists(side):
        try:
            goal_stance = json.load(open(side))["stances"]["goal"]
        except Exception:  # noqa: BLE001
            goal_stance = None
    if robot is not None:
        pos, quat = robot.get_position_orientation()
        pos, quat = pos.detach().cpu().numpy(), quat.detach().cpu().numpy()
        x, y, z, w = [float(v) for v in quat]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        out["base_final_xyyaw"] = [float(pos[0]), float(pos[1]), yaw]
        if goal_stance and goal_stance.get("base_pose2d"):
            gx, gy, gyaw = [float(v) for v in goal_stance["base_pose2d"]]
            out["goal_base_xyyaw"] = [gx, gy, gyaw]
            out["base_final_vs_goal_xy_m"] = math.hypot(float(pos[0]) - gx, float(pos[1]) - gy)
            out["base_final_vs_goal_yaw_deg"] = math.degrees(abs((yaw - gyaw + math.pi) % (2 * math.pi) - math.pi))
        try:
            idx = robot.controller_action_idx.get("trunk")
            tj = robot.get_joint_positions()[robot.trunk_control_idx].detach().cpu().numpy()
            out["trunk_final_q"] = [float(v) for v in tj]
            if goal_stance is not None and goal_stance.get("translate") is not None:
                gq = torso_q(float(goal_stance["translate"]), float(goal_stance.get("tilt", 0.0))).numpy()
                out["goal_trunk_q"] = [float(v) for v in gq]
                out["goal_trunk_translate_tilt"] = [float(goal_stance["translate"]), float(goal_stance.get("tilt", 0.0))]
                out["trunk_final_vs_goal_max_abs_dq_rad"] = float(np.abs(tj - gq).max())
                if actions is not None and len(actions) and idx is not None:
                    last_cmd = np.asarray(actions[-1])[np.asarray(idx.detach().cpu().numpy() if hasattr(idx, "detach") else idx)]
                    out["trunk_last_cmd_q"] = [float(v) for v in last_cmd]
                    out["trunk_last_cmd_vs_goal_max_abs_dq_rad"] = float(np.abs(last_cmd - gq).max())
                    out["trunk_last_cmd_vs_goal_dq_rad"] = [float(v) for v in (last_cmd - gq)]
        except Exception as e:  # noqa: BLE001
            out["trunk_error"] = repr(e)
    return out


def score_relocalization(meta: dict, shared: dict, ep: dict) -> dict:
    import torch as th

    from omnigibson.object_states import PoseReached

    robot = shared["robot"]
    env = shared["env"]
    errs = (meta.get("errors_final_step") or meta.get("errors") or {}).get("pose_errors") or {}
    rep = next(iter(errs.values()), {}) if errs else {}
    out = {"geodesic_distance_m": None, "position_error_m": rep.get("position_error"), "heading_error_deg": None,
           "orientation_error_deg": None if rep.get("orientation_error") is None else math.degrees(rep["orientation_error"]),
           "within_tolerance": rep.get("within_tolerance"), "distance_bucket_m": ep.get("bucket_hi_m"), "geodesic_source": None}
    try:
        st = robot.states[PoseReached]
        goal_pos, goal_ori = st._goal_pos, st._goal_ori
        cur_pos, cur_ori = st._get_current_pose()
        cur_pos, cur_ori = st._pose_in_frame(cur_pos, cur_ori)
        out["euclidean_xy_m"] = float(th.norm(th.as_tensor(cur_pos[:2]) - th.as_tensor(goal_pos[:2])))
        # decomposition of the final errors: horizontal / vertical position; yaw / pitch / roll of goal^-1 * current
        out["position_error_xy_m"] = out["euclidean_xy_m"]
        out["position_error_z_m"] = float(cur_pos[2] - goal_pos[2])
        out["camera_pose_final"] = {"position": [float(v) for v in cur_pos], "orientation_xyzw": [float(v) for v in cur_ori]}
        out["camera_pose_goal"] = {"position": [float(v) for v in goal_pos], "orientation_xyzw": [float(v) for v in goal_ori]} if goal_ori is not None else None
        if goal_ori is not None and cur_ori is not None:
            out["heading_error_deg"] = yaw_error_deg(cur_ori, goal_ori)
            out["orientation_error_ypr_deg"] = relative_ypr_deg(cur_ori, goal_ori)
        try:
            path, dist = env.scene.trav_map.get_shortest_path(0, th.as_tensor(cur_pos[:2], dtype=th.float32), th.as_tensor(goal_pos[:2], dtype=th.float32), entire_path=True)
            if dist is not None:
                out["geodesic_distance_m"] = float(dist)
                out["geodesic_source"] = "trav_map(camera xy -> goal camera xy)"
        except Exception as e:  # noqa: BLE001
            out["geodesic_error"] = repr(e)
        if out["geodesic_distance_m"] is None:
            out["geodesic_distance_m"] = out["euclidean_xy_m"]
            out["geodesic_source"] = "euclidean_xy_fallback"
    except Exception as e:  # noqa: BLE001
        out["score_error"] = repr(e)
        if out["geodesic_distance_m"] is None and rep.get("position_error") is not None:
            out["geodesic_distance_m"] = rep["position_error"]
            out["geodesic_source"] = "position_error_fallback"
    return out


def score_articulation(meta: dict) -> dict:
    errs = (meta.get("errors_final_step") or meta.get("errors") or {}).get("layout_errors") or {}
    out = {"angle_error_deg": None, "final_angle_deg": None, "target_angle_deg": None, "within_tolerance": None, "joint": None}
    for inst, rep in errs.items():
        joints = rep.get("joints") or {}
        for jn, j in joints.items():
            if "Prismatic" in str(j.get("joint_type")):
                continue
            out.update({"angle_error_deg": math.degrees(float(j["error"])), "final_angle_deg": math.degrees(float(j["position"])),
                        "target_angle_deg": math.degrees(float(j["goal"])), "within_tolerance": j.get("within_tolerance"),
                        "joint": f"{inst}:{jn}", "tolerance_deg": math.degrees(float(j["tolerance"])) if j.get("tolerance") is not None else None})
            return out
        if rep.get("orientation_error") is not None and out["angle_error_deg"] is None:
            out.update({"angle_error_deg": math.degrees(float(rep["orientation_error"])), "within_tolerance": rep.get("within_tolerance"), "joint": inst})
    return out


def score_pose_target(meta: dict) -> dict:
    """The evaluator's own pose-target errors as the headline (mental_rotation: the held object's orientation vs the target, tolerance
    0.15 rad; object_scaling: the object's depth in the head camera frame vs the target depth, tolerance ~2 cm)."""
    errs = (meta.get("errors_final_step") or meta.get("errors") or {}).get("pose_errors") or {}
    out = {"object": None, "position_metric": None, "orientation_metric": None, "position_error_m": None, "orientation_error_deg": None, "depth_m": None, "goal_depth_m": None,
           "within_tolerance": None, "steps_to_success": meta.get("first_success_step")}
    for inst, rep in errs.items():
        out.update({"object": inst, "position_metric": rep.get("position_metric"), "orientation_metric": rep.get("orientation_metric"), "position_error_m": rep.get("position_error"),
                    "orientation_error_deg": None if rep.get("orientation_error") is None else math.degrees(float(rep["orientation_error"])), "depth_m": rep.get("depth"),
                    "goal_depth_m": rep.get("goal_depth"), "within_tolerance": rep.get("within_tolerance")})
        break
    return out


def score_alignment(meta: dict) -> dict:
    errs = (meta.get("errors_final_step") or meta.get("errors") or {}).get("pose_errors") or {}
    out = {"alignment_error_deg": None, "steps_to_success": meta.get("first_success_step"), "within_tolerance": None, "object": None}
    for inst, rep in errs.items():
        if rep.get("orientation_error") is not None:
            out.update({"alignment_error_deg": math.degrees(float(rep["orientation_error"])), "within_tolerance": rep.get("within_tolerance"), "object": inst})
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--ckpt-repo", required=True)
    ap.add_argument("--ckpt-file", required=True)
    ap.add_argument("--ckpt-sha256", required=True)
    ap.add_argument("--ckpt-path", default=None, help="local checkpoint path (sha256 re-verified when given)")
    ap.add_argument("--server", default="ws://127.0.0.1:8000")
    ap.add_argument("--server-args", default="", help="the server's command line, recorded")
    ap.add_argument("--action-horizon", type=int, default=16, help="expected server action horizon (checked against the handshake)")
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--instances", default=os.path.join(WT, "lab", "instances.json"), help="episode -> instance map (lab/instances.json: test set; lab/instances-train.json: training episodes)")
    ap.add_argument("--dataset", default=os.path.join(WT, "datasets", "isg-init-eval"), help="LeRobot root of the episodes (goal PNGs under isg_meta/goals)")
    ap.add_argument("--split", default="eval50", help="recorded split name: eval50 (test set) or train (training episodes)")
    ap.add_argument("--goals-root", default=None, help="wrist goal PNGs (lab_tools/wrist_goals.py output; default datasets/goals/<dataset dir name>)")
    ap.add_argument("--lab-out", required=True, help="lab/tierB/<run>/step_XXXXXXXX (the task subdir is added)")
    ap.add_argument("--raw-out", required=True, help="runlogs/tierB/<run>/step_XXXXXXXX (videos, meta.json per instance)")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="episode indices (default: every episode of the task)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=3000, help="policy steps (the evaluator's default: run_isg --policy-steps 3000)")
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--settle-steps", type=int, default=50)
    ap.add_argument("--video-size", type=int, default=540)
    ap.add_argument("--external-video", action="store_true", help="also record the over-the-shoulder external camera (run_isg --dev)")
    ap.add_argument("--goal-from-episode", type=int, default=None, help="plumbing check: use another episode's goal PNG for every episode")
    ap.add_argument("--goal-swap", type=int, nargs=2, metavar=("EPISODE", "OTHER"), default=None,
                    help="plumbing check in the same world: after the normal episodes, run EPISODE again with OTHER's goal PNG -> episode_EPISODE.goalswap_OTHER.json")
    ap.add_argument("--no-stop-on-success", action="store_true")
    ap.add_argument("--keep-depth", action="store_true", help="keep the depth_linear annotators of BEHAVIOR's sensor set (slower; the policy does not use depth)")
    ap.add_argument("--include-unverified", action="store_true", help="also run episodes whose instance mapping is not first-frame verified (default: verified only)")
    ap.add_argument("--tag", default=None, help="free text recorded in every episode (smoke, ...)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-planner", action="store_true", help="evaluator patch: no cuRobo planner in policy mode (streams never plan; saves ~5 GB of GPU memory and the build)")
    ap.add_argument("--stock-handle-ag", action="store_true", help="OmniGibson's stock assisted-grasp joint rule (default since 2026-10-08: the oracle's FixedJoint for links behind movable joints)")
    ap.add_argument("--cleanup", action="store_true", help="after every episode: gc.collect() + torch.cuda.empty_cache() (evaluator_patches.after_episode_cleanup)")
    args = ap.parse_args(argv)
    return run_job(args, shared={}, shutdown=True)


def memory_snapshot() -> dict:
    """This process's GPU memory (nvidia-smi) and host RSS, plus torch's allocator counters, after an episode."""
    out = {"pid": os.getpid()}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    out["rss_mib"] = int(line.split()[1]) // 1024
    except Exception:  # noqa: BLE001
        pass
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        for line in q.strip().splitlines():
            pid, mem = [x.strip() for x in line.split(",")[:2]]
            if int(pid) == os.getpid():
                out["gpu_mib"] = int(float(mem))
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch as th

        out["torch_allocated_mib"] = int(th.cuda.memory_allocated() // (1 << 20))
        out["torch_reserved_mib"] = int(th.cuda.memory_reserved() // (1 << 20))
    except Exception:  # noqa: BLE001
        pass
    return out


def run_job(args, shared=None, shutdown=True) -> int:
    """The episodes of one (checkpoint, split, episode list) job in this process; ``shared`` carries the loaded world across jobs
    (the persistent evaluator worker passes the same dict for every job; the CLI passes a fresh one)."""
    import numpy as np
    import torch as th

    sys.path.insert(0, WT)
    from lab_tools import act_policy as AP
    from lab_tools import evaluator_patches

    # -- evaluator patches (policy mode): BEHAVIOR sensor set, no depth, timeout-only stream QA, never-held epilogue ----
    evaluator_patches.install(keep_depth=args.keep_depth, no_planner=bool(getattr(args, "no_planner", False)), fixed_handle_ag=not getattr(args, "stock_handle_ag", False))
    from ispatialgym.oracle import run_isg as R

    # -- episodes ------------------------------------------------------------------------------------------------------
    inst_doc = json.load(open(args.instances))
    inst_map = inst_doc.get("episodes", inst_doc)
    goals = {g["episode_index"]: g for g in json.load(open(os.path.join(args.dataset, "isg_meta", "goals", "index.json")))["episodes"]}
    # wrist goal images (checkpoints with goal_views beyond zed_link): {episode: {camera: png path}}
    goals_root = args.goals_root or os.path.join(WT, "datasets", "goals", os.path.basename(os.path.normpath(args.dataset)))
    wrist_goals = {}
    if os.path.exists(os.path.join(goals_root, "index.json")):
        for g in json.load(open(os.path.join(goals_root, "index.json")))["episodes"]:
            wrist_goals.setdefault(int(g["episode_index"]), {})[g["camera"]] = os.path.join(goals_root, g["png"])
    buckets = {}
    csv_path = os.path.join(args.dataset, "isg_meta", "camera_relocalization_distances.csv")
    if os.path.exists(csv_path):
        import csv

        with open(csv_path) as f:
            for r in csv.DictReader(f):
                buckets[int(r["episode_index"])] = r
    episodes = []
    for k, v in inst_map.items():
        e = int(k)
        if v.get("task") != args.task or not v.get("instance"):
            continue
        if not v.get("verified") and not args.include_unverified:
            continue
        if args.episodes is not None and e not in set(args.episodes):
            continue
        g = goals[e]
        episodes.append({"episode_index": e, "instance": v["instance"], "tro": v["tro"], "train_task_index": int(g["train_task_index"]),
                         "goal_png": os.path.join(args.dataset, g["png"]), "verified": v.get("verified"), "first_frame_mean_abs_diff": v.get("first_frame_mean_abs_diff"),
                         "instance_index": v.get("instance_index"), "bucket_hi_m": float(buckets[e]["bucket_hi_m"]) if e in buckets else None})
    episodes.sort(key=lambda d: d["episode_index"])
    lab_dir = os.path.join(args.lab_out, args.task)
    raw_dir = os.path.join(args.raw_out, args.task)
    os.makedirs(lab_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    todo = [ep for ep in episodes if args.overwrite or not os.path.exists(os.path.join(lab_dir, f"episode_{ep['episode_index']:06d}.json"))]
    if args.limit:
        todo = todo[: args.limit]  # episodes per process (the simulator process grows by ~2.5 GB of GPU memory per hour)
    print(f"EVAL run={args.run} step={args.step} task={args.task} episodes={len(episodes)} todo={len(todo)} server={args.server}", flush=True)
    if not todo and not args.goal_swap:
        return 0
    if args.ckpt_path:
        actual = sha256_of(args.ckpt_path)
        if actual != args.ckpt_sha256:
            raise SystemExit(f"checkpoint sha256 mismatch: {actual} != {args.ckpt_sha256}")

    provenance = {"evaluator_commit": git_head(WT), "evaluator_root": WT, "b1k_commit": git_head(os.path.join(WT, "Behavior-1K")),
                  "act_commit": git_head(os.path.join(WT, "act-isg")), "bridge_sha256": {f: sha256_of(os.path.join(WT, "lab_tools", f)) for f in ("act_policy.py", "eval_episodes.py")}}
    import omnigibson as og

    provenance["omnigibson_version"] = og.__version__
    goal_override = None
    if args.goal_from_episode is not None:
        g = goals[args.goal_from_episode]
        goal_override = (args.goal_from_episode, os.path.join(args.dataset, g["png"]))

    shared = {} if shared is None else shared
    t_proc = time.time()

    def run_one(ep, k, n, goal_png, suffix="", goal_from=None, extra_raw_sub=None):
        e = ep["episode_index"]
        t0 = time.time()
        seed = int(ep["instance_index"]) if ep.get("instance_index") is not None else e
        th.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        random.seed(seed)
        out_root = raw_dir if extra_raw_sub is None else os.path.join(raw_dir, extra_raw_sub)
        goal_pngs = {"zed_link": goal_png, **{cam: p for cam, p in wrist_goals.get(goal_from if goal_from is not None else e, {}).items()}}
        ctx = {"server": args.server, "goal_rgb": load_goal(goal_png), "goal_rgbs": {cam: load_goal(p) for cam, p in goal_pngs.items()},
               "task_id": ep["train_task_index"], "horizon": args.horizon,
               "stop_on_success": not args.no_stop_on_success, "expect_action_horizon": args.action_horizon}
        AP.CONTEXT = ctx
        argv_r = ["--instance", ep["tro"], "--mode", "policy", "--policy", "lab_tools.act_policy:ActPolicy", "--policy-steps", str(args.horizon),
                  "--max-steps", str(args.max_steps), "--settle-steps", str(args.settle_steps), "--out", out_root, "--record", "video", "--video",
                  "--video-size", str(args.video_size), "--demo-gating", "none", "--no-goal-override", "--log-level", "WARNING"]
        if args.external_video:
            argv_r.append("--dev")
        rargs = R.parse_args(argv_r)
        record = {"run": args.run, "step": args.step, "checkpoint": {"repo": args.ckpt_repo, "file": args.ckpt_file, "sha256": args.ckpt_sha256},
                  "split": args.split, "dataset": os.path.basename(os.path.normpath(args.dataset)),
                  "task": args.task, "episode_index": e, "instance": ep["instance"], "instance_index": ep.get("instance_index"), "seed": seed,
                  "success": None, "steps": None, "horizon": args.horizon, "wall_s": None, "evaluator_commit": provenance["evaluator_commit"],
                  "act_commit": provenance["act_commit"], "server_args": args.server_args, "error": None, "score": None,
                  "goal_png": os.path.relpath(goal_png, args.dataset), "goal_pngs": {cam: os.path.relpath(p, WT) for cam, p in goal_pngs.items()},
                  "goal_from_episode": goal_from, "tag": args.tag,
                  "stop_on_success": ctx["stop_on_success"], "instance_verified": ep.get("verified"), "first_frame_mean_abs_diff": ep.get("first_frame_mean_abs_diff"),
                  "fixed_handle_ag": not getattr(args, "stock_handle_ag", False), "started_at": utc(), "finished_at": None, "provenance": provenance}
        try:
            if shared.get("env") is not None:
                shared["template_dir"] = os.path.dirname(os.path.dirname(os.path.abspath(ep["tro"])))  # same template, possibly another path (lenient candidates)
            rc, result = R.run_instance(rargs, ep["tro"], shared)
            out_dir = result.get("out")
            meta = {}
            if out_dir and os.path.exists(os.path.join(out_dir, "meta.json")):
                with open(os.path.join(out_dir, "meta.json")) as f:
                    meta = json.load(f)
            record.update({
                "success": bool(result.get("success")), "steps": int(ctx.get("steps") or result.get("n_action_steps") or 0),
                "stop_reason": ctx.get("stop_reason"), "exit_code": rc, "status": result.get("status"), "reason": result.get("reason"),
                "n_steps_total": result.get("n_steps"), "n_action_steps": result.get("n_action_steps"),
                "first_success_step": meta.get("first_success_step"), "binary_success_ever": meta.get("binary_success_ever"),
                "binary_success_final": meta.get("binary_success_final"), "settled_success": meta.get("settled_success"),
                "settle_windows": meta.get("settle_windows"), "video": os.path.relpath(meta["video"], WT) if meta.get("video") else None,
                "raw_dir": os.path.relpath(out_dir, WT) if out_dir else None,
                "server_metadata": {k2: v for k2, v in (ctx.get("server_metadata") or {}).items() if k2 in ("policy", "action_horizon", "chunk_size", "execution_mode", "checkpoint_step", "goal_fusion", "goal_views", "task_map")},
                "goal_views_sent": ctx.get("goal_views_sent"),
                "timings": meta.get("timings"), "release_check": meta.get("release_check"),
            })
            if result.get("status") == "error" or rc == R.EXIT_ERROR:
                record["error"] = result.get("error") or "evaluator error"
            if args.task == "camera_relocalization-standard":
                score = score_relocalization(meta, shared, ep)
            elif args.task.startswith("alignment"):
                score = score_alignment(meta)
            elif args.task.startswith(("mental_rotation", "object_scaling")):
                score = score_pose_target(meta)
            else:
                score = score_articulation(meta)
            qa = meta.get("qa") or {}
            score["native"] = {
                "max_position_error": meta.get("max_position_error"), "max_orientation_error": meta.get("max_orientation_error"),
                "errors_final_step": meta.get("errors_final_step"), "reason": result.get("reason"), "verdict": meta.get("verdict"),
                "contacts": (qa.get("collisions") or {}).get("n_events") if isinstance(qa.get("collisions"), dict) else result.get("collisions"),
                "collisions_by_object": qa.get("collisions_by_object") or (qa.get("collisions") or {}).get("by_object") if isinstance(qa, dict) else None,
            }
            if args.task == "camera_relocalization-standard":
                try:
                    score["closest_approach"] = closest_approach(ctx.get("step_errors"))
                except Exception as e3:  # noqa: BLE001
                    score["closest_approach"] = {"error": repr(e3)}
                try:
                    score["base_trunk_vs_goal"] = base_and_trunk_vs_goal(shared, ep, ctx.get("actions"))
                except Exception as e3:  # noqa: BLE001
                    score["base_trunk_vs_goal"] = {"error": repr(e3)}
            record["score"] = score
            record["server_request_s_mean"] = ctx.get("request_s_mean")
            record["server_request_s_max"] = ctx.get("request_s_max")
            if ctx.get("actions") and out_dir:
                np.save(os.path.join(out_dir, "actions.npy"), np.stack(ctx["actions"]).astype(np.float32))
                record["actions_npy"] = os.path.relpath(os.path.join(out_dir, "actions.npy"), WT)
            if ctx.get("step_errors") and out_dir:
                np.save(os.path.join(out_dir, "errors.npy"), np.asarray(ctx["step_errors"], dtype=np.float32))  # per step: position_error_m, orientation_error_rad
                record["errors_npy"] = os.path.relpath(os.path.join(out_dir, "errors.npy"), WT)
        except Exception as e2:  # noqa: BLE001
            record["error"] = repr(e2)
            record["traceback"] = traceback.format_exc()
            traceback.print_exc()
        if getattr(args, "cleanup", False):
            try:
                evaluator_patches.after_episode_cleanup()
            except Exception as e4:  # noqa: BLE001
                record["cleanup_error"] = repr(e4)
        record["memory"] = memory_snapshot()
        record["worker"] = getattr(args, "worker_id", None)
        record["wall_s"] = round(time.time() - t0, 1)
        record["finished_at"] = utc()
        name = f"episode_{e:06d}{suffix}.json"
        with open(os.path.join(lab_dir, name + ".tmp"), "w") as f:
            json.dump(record, f, indent=1, default=str)
        os.replace(os.path.join(lab_dir, name + ".tmp"), os.path.join(lab_dir, name))
        sc = record.get("score") or {}
        headline = {k2: v for k2, v in sc.items() if k2 in ("geodesic_distance_m", "heading_error_deg", "angle_error_deg", "alignment_error_deg")}
        print(f"EVAL_EPISODE {k + 1}/{n} ep={e}{suffix} inst={ep['instance'].rsplit('__', 1)[-1]} success={record['success']} reason={record.get('reason')} "
              f"steps={record['steps']} stop={record.get('stop_reason')} wall={record['wall_s']}s score={json.dumps(headline)} err={record['error']}", flush=True)
        return record

    for k, ep in enumerate(todo):
        run_one(ep, k, len(todo), goal_override[1] if goal_override else ep["goal_png"], goal_from=goal_override[0] if goal_override else None)
    if args.goal_swap:
        e_swap, other = args.goal_swap
        ep = next((d for d in episodes if d["episode_index"] == e_swap), None)
        if ep is None:
            print(f"EVAL goal-swap: episode {e_swap} is not in this task's list", flush=True)
        else:
            g = goals[other]
            run_one(ep, 0, 1, os.path.join(args.dataset, g["png"]), suffix=f".goalswap_{other:06d}", goal_from=other, extra_raw_sub=f"goalswap_{other:06d}")
    print(f"EVAL_DONE task={args.task} episodes={len(todo)} wall={round(time.time() - t_proc, 1)}s", flush=True)
    if shutdown:
        try:
            og.shutdown()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
