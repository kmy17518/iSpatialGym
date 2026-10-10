"""Vectorized multi-instance evaluator (lab): N relocalization instances per Isaac Sim process, one batched ``env.step`` per
control step (``ispatialgym.oracle.vec_loader.VecInstanceLoader`` + ``vec_stream.VecStreamExecutor`` + ``tasks.vec_task``).

Modes: ``policy`` (the ACT server, one websocket session per environment: ``lab_tools.act_policy.ActPolicy(ctx=...)``) and
``replay`` (the demo's recorded actions, open loop). Per episode the outputs are the single-env evaluator's: ``<raw>/<task>/<iid>/
{meta.json, head_rgb.mp4, init_rgb.png, final_rgb.png, actions.npy, errors.npy, annotations.json}`` (replay: the trajectory
arrays too) and ``<lab>/<task>/episode_XXXXXX.json`` with the same fields plus ``slot``, ``num_envs``, ``batch`` and the
vectorized provenance. All poses in records and scores are **scene-relative** (= the base scene's world frame, what the
single-env records hold); the geodesic distance is computed on the clone's own traversability map in scene coordinates.

    python lab_tools/vec_eval.py --mode replay --split eval50 --dataset datasets/isg-init-eval --instances lab/instances.json \
        --episodes 2 3 33 34 --num-envs 4 --lab-out lab/diagnostics/vec-eval/3b/n4 --raw-out runlogs/diagnostics/vec-eval/3b/n4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)

from lab_tools.eval_episodes import closest_approach, git_head, load_goal, memory_snapshot, relative_ypr_deg, sha256_of, utc, yaw_error_deg  # noqa: E402

TASK = "camera_relocalization-standard"
BRIDGE_FILES = ("vec_eval.py", "act_policy.py", "replay_policy.py", "evaluator_patches.py")
EVALUATOR_FILES = ("ispatialgym/oracle/vec_stream.py", "ispatialgym/oracle/vec_loader.py", "ispatialgym/tasks/vec_task.py", "ispatialgym/oracle/executor.py",
                   "ispatialgym/tasks/ispatialgym_task.py")


def _yaw(q) -> float:
    x, y, z, w = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("policy", "replay"), default="policy")
    ap.add_argument("--run", default="replay")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--split", default="eval50", help="eval50 | train | (replay: the split replayed)")
    ap.add_argument("--dataset", required=True, help="LeRobot root (isg-init-eval for eval50, isg-init for train)")
    ap.add_argument("--instances", required=True, help="lab/instances.json or lab/instances-train.json")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--horizon", type=int, default=1500)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--tiled", dest="tiled", action="store_true", default=False, help="one tiled render product for all clones (off: the tiles share one exposure, gate 3d)")
    ap.add_argument("--no-tiled", dest="tiled", action="store_false")
    ap.add_argument("--warm-renders", type=int, default=100, help="renders after the batch reset before the first observation (the renderer's view adaptation)")
    ap.add_argument("--enhanced-determinism", dest="enhanced_determinism", action="store_true", default=None, help="PhysX enhanced determinism (default: on for num_envs > 1)")
    ap.add_argument("--no-enhanced-determinism", dest="enhanced_determinism", action="store_false")
    ap.add_argument("--stock-handle-ag", action="store_true", help="OmniGibson's stock assisted-grasp joint rule (default: the oracle's FixedJoint for links behind movable joints)")
    ap.add_argument("--physics", default=None, help="item-2 experiments: JSON of physics overrides, e.g. '{\"broadphase\": \"SAP\", \"num_threads\": 1}'")
    ap.add_argument("--dummy-bodies", type=int, default=0, help="item-2 experiments: static cubes added to scene 0 at x = 63 m")
    ap.add_argument("--render", default=None, help="item-1c experiments: JSON of carb render settings applied after launch, e.g. '{\"/rtx/post/aa/op\": 0}'")
    ap.add_argument("--scene-margin", type=float, default=None, help="gap between cloned scenes along +x in m (OmniGibson default 10; spacing = margin + scene width ~53 m)")
    ap.add_argument("--server", default="ws://127.0.0.1:8000")
    ap.add_argument("--server-args", default=None, help="the serve_b1k.py extra arguments (recorded)")
    ap.add_argument("--action-horizon", type=int, default=None, help="expected server action_horizon (checked)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--lab-out", required=True)
    ap.add_argument("--raw-out", required=True)
    ap.add_argument("--video-size", type=int, default=480)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--stream-qa", default="off", choices=["off", "stop", "log"],
                    help="VecStreamExecutor stream_qa: off = only the timeout ends a stream (the evaluator's rule), stop = every QA monitor, "
                         "log = the verdicts are recorded per slot and the stream goes on")
    ap.add_argument("--no-stop-on-success", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--include-unverified", action="store_true")
    ap.add_argument("--limit-batches", type=int, default=None)
    ap.add_argument("--ckpt-repo", default=None)
    ap.add_argument("--ckpt-file", default=None)
    ap.add_argument("--ckpt-sha256", default=None)
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument("--goals-root", default=None)
    ap.add_argument("--keep-depth", action="store_true")
    ap.add_argument("--slot-order", default="natural", choices=("natural", "reverse", "rotate"), help="which slot an episode takes (cross-talk tests)")
    ap.add_argument("--same-episode-all-slots", action="store_true", help="every slot of a batch runs the batch's first episode (replay exactness per slot)")
    ap.add_argument("--repeat", type=int, default=1, help="run every batch this many times in the same process (records / raw dirs suffixed .repK; reproducibility experiments)")
    ap.add_argument("--worker-id", default=None)
    ap.add_argument("--no-task-subdir", action="store_true", help="write records to --lab-out / --raw-out directly (the replay layout) instead of a <task>/ subdirectory")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------------------------------------- scene-relative


def scene_relative(scene, pos, quat):
    """A world pose in the base scene's frame: the clone's translation subtracted (clones are translated copies, so the
    orientation is untouched); the base scene's pose is returned as it is - bit for bit what the single-env records hold."""
    import torch as th

    from ispatialgym.eval.clone_frames import scene_translation

    pos = th.as_tensor(pos, dtype=th.float32).reshape(3)
    quat = th.as_tensor(quat, dtype=th.float32).reshape(4)
    t = scene_translation(scene)
    if t is None:
        return pos, quat
    return pos - t, quat


class SceneRelativeRobot:
    """A robot whose ``get_position_orientation()`` is scene-relative (the base scene's frame); everything else forwarded."""

    def __init__(self, robot, scene):
        self._robot, self._scene = robot, scene

    def get_position_orientation(self, *a, **k):
        pos, quat = self._robot.get_position_orientation(*a, **k)
        return scene_relative(self._scene, pos, quat)

    def __getattr__(self, name):
        return getattr(self._robot, name)


def score_relocalization_env(meta: dict, robot, scene, ep: dict) -> dict:
    """``eval_episodes.score_relocalization`` for one clone: errors from its PoseReached state, camera poses converted to scene
    coordinates, the geodesic on the clone's own traversability map in scene coordinates."""
    import torch as th
    from omnigibson.object_states import PoseReached

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
        cur_pos, cur_ori = scene_relative(scene, cur_pos, cur_ori if cur_ori is not None else [0, 0, 0, 1])
        if goal_ori is not None:
            goal_pos, goal_ori = scene_relative(scene, goal_pos, goal_ori)
        else:
            goal_pos, _ = scene_relative(scene, goal_pos, [0, 0, 0, 1])
        out["euclidean_xy_m"] = float(th.norm(th.as_tensor(cur_pos[:2]) - th.as_tensor(goal_pos[:2])))
        out["position_error_xy_m"] = out["euclidean_xy_m"]
        out["position_error_z_m"] = float(cur_pos[2] - goal_pos[2])
        out["camera_pose_final"] = {"position": [float(v) for v in cur_pos], "orientation_xyzw": [float(v) for v in cur_ori]}
        out["camera_pose_goal"] = {"position": [float(v) for v in goal_pos], "orientation_xyzw": [float(v) for v in goal_ori]} if st._goal_ori is not None else None
        if st._goal_ori is not None:
            out["heading_error_deg"] = yaw_error_deg(cur_ori, goal_ori)
            out["orientation_error_ypr_deg"] = relative_ypr_deg(cur_ori, goal_ori)
        try:
            path, dist = scene.trav_map.get_shortest_path(0, th.as_tensor(cur_pos[:2], dtype=th.float32), th.as_tensor(goal_pos[:2], dtype=th.float32), entire_path=True)
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


# ---------------------------------------------------------------------------------------------------------- sources


class PoseTracer:
    """Per-step trace of one slot, read before every policy call (index t = the pose ``observation.state[t]`` was taken at): the
    scene-relative base (x, y, yaw), the proprio base velocity, the head camera pose, the robot joints and the tracked objects'
    poses - the replay comparison material (gate 3b) and the per-slot traces of the closed-loop runs (gate 3f)."""

    def __init__(self, ctx: dict, robot, scene, cam, tracked: Optional[Dict[str, Any]] = None):
        self.ctx, self.robot, self.scene, self.cam = ctx, robot, scene, cam
        self.objects = [(k, v) for k, v in (tracked or {}).items() if v is not None and hasattr(v, "get_position_orientation")]
        ctx.update({"base_xyyaw": [], "base_qvel": [], "camera_pose": [], "joint_q": [], "object_poses": [], "object_names": [k for k, _ in self.objects]})

    def record(self, obs):
        import numpy as np

        pos, quat = scene_relative(self.scene, *self.robot.get_position_orientation())
        pos, quat = pos.detach().cpu().numpy(), quat.detach().cpu().numpy()
        self.ctx["base_xyyaw"].append([float(pos[0]), float(pos[1]), _yaw(quat)])
        try:
            pr = obs[self.robot.name]["proprio"]
            pr = pr.detach().cpu().numpy() if hasattr(pr, "detach") else np.asarray(pr)
            self.ctx["base_qvel"].append([float(v) for v in np.asarray(pr).reshape(-1)[0:3]])
        except Exception:  # noqa: BLE001
            pass
        if self.cam is not None:
            cp, cq = scene_relative(self.scene, *self.cam.get_position_orientation())
            self.ctx["camera_pose"].append([*[float(v) for v in cp.detach().cpu().numpy()], *[float(v) for v in cq.detach().cpu().numpy()]])
        try:
            self.ctx["joint_q"].append([float(v) for v in self.robot.get_joint_positions().detach().cpu().numpy()])
            row = []
            for _, o in self.objects:
                op, oq = scene_relative(self.scene, *o.get_position_orientation())
                row.append([*[float(v) for v in op.detach().cpu().numpy()], *[float(v) for v in oq.detach().cpu().numpy()]])
            self.ctx["object_poses"].append(row)
        except Exception:  # noqa: BLE001
            pass

    def save(self, out_dir: str, prefix: str = "replay") -> Dict[str, str]:
        import numpy as np

        ctx, out = self.ctx, {}
        for key, name in (("base_xyyaw", f"{prefix}_base_xyyaw.npy"), ("base_qvel", f"{prefix}_base_qvel.npy"), ("camera_pose", f"{prefix}_camera_pose.npy"),
                          ("joint_q", f"{prefix}_joint_q.npy"), ("object_poses", f"{prefix}_object_poses.npy")):
            if ctx.get(key):
                np.save(os.path.join(out_dir, name), np.asarray(ctx[key], dtype=np.float32))
                out[key] = os.path.relpath(os.path.join(out_dir, name), WT)
        if ctx.get("object_names"):
            with open(os.path.join(out_dir, f"{prefix}_objects.json"), "w") as f:
                json.dump(ctx["object_names"], f)
        return out


class TracedPolicy:
    """A slot's policy callable with a :class:`PoseTracer` read before every call (the closed-loop per-slot trace)."""

    def __init__(self, policy, tracer: PoseTracer):
        self.policy, self.tracer = policy, tracer

    def __call__(self, obs, info):
        self.tracer.record(obs)
        return self.policy(obs, info)


class ReplaySource:
    """``replay_policy.ReplayPolicy`` for one slot: the recorded actions, the pose trace before every step, the per-step pose errors."""

    def __init__(self, ctx: dict, robot, scene, cam, tracked: Optional[Dict[str, Any]] = None):
        from lab_tools.act_policy import record_step_error, task_info

        self.ctx = ctx
        self._task_info, self._record_step_error = task_info, record_step_error
        self.steps = 0
        self.tracer = PoseTracer(ctx, robot, scene, cam, tracked)
        ctx.update({"stop_reason": None, "actions_streamed": 0, "step_errors": []})

    def record_pose(self, obs):
        self.tracer.record(obs)

    def end(self, reason):
        self.ctx["stop_reason"] = reason
        self.ctx["steps"] = self.steps
        return None

    def __call__(self, obs, info):
        import numpy as np

        ctx = self.ctx
        self.record_pose(obs)
        actions = ctx["actions"]
        if self.steps >= int(ctx["horizon"]):
            return self.end("horizon")
        if self.steps >= len(actions):
            return self.end("demo_end")
        isg = self._task_info(info)
        self._record_step_error(ctx, isg)
        if ctx.get("stop_on_success", True) and isg.get("official_success"):
            return self.end("success")
        a = np.asarray(actions[self.steps], dtype=np.float32).reshape(-1)
        if a.shape != (23,) or not np.isfinite(a).all():
            raise RuntimeError(f"bad recorded action at step {self.steps}: shape {a.shape}")
        self.steps += 1
        ctx["steps"] = self.steps
        ctx["actions_streamed"] = self.steps
        if self.steps >= int(ctx["horizon"]) or self.steps >= len(actions):
            ctx["stop_reason"] = "horizon" if self.steps >= int(ctx["horizon"]) else "demo_end"
        return a


# ---------------------------------------------------------------------------------------------------------- main


def args_from_job(job: dict, worker_id=None):
    """The job dict of the persistent worker (``eval_worker.py``: the fields of this tool's arguments) as parsed arguments."""
    argv = ["--mode", job.get("mode") or ("replay" if job.get("kind") == "replay" else "policy"), "--dataset", job["dataset"], "--instances", job["instances"],
            "--lab-out", job["lab_out"], "--raw-out", job["raw_out"], "--num-envs", str(job.get("num_envs", 4)), "--split", str(job.get("split", "eval50")),
            "--horizon", str(job.get("horizon", 1500)), "--task", job.get("task") or TASK, "--server", job.get("server") or "ws://127.0.0.1:8000",
            "--video-size", str(job.get("video_size", 540)), "--max-steps", str(job.get("max_steps", 6000)), "--settle-steps", str(job.get("settle_steps", 50)),
            "--warm-renders", str(job.get("warm_renders", 100))]
    if job.get("run"):
        argv += ["--run", str(job["run"])]
    if job.get("step") is not None:
        argv += ["--step", str(job["step"])]
    if job.get("episodes"):
        argv += ["--episodes", *[str(e) for e in job["episodes"]]]
    for key, flag in (("server_args", "--server-args"), ("action_horizon", "--action-horizon"), ("tag", "--tag"), ("stream_qa", "--stream-qa"), ("ckpt_repo", "--ckpt-repo"), ("ckpt_file", "--ckpt-file"),
                      ("ckpt_sha256", "--ckpt-sha256"), ("ckpt_path", "--ckpt-path"), ("goals_root", "--goals-root")):
        if job.get(key) not in (None, ""):
            argv += [flag, str(job[key])]
    for key, flag in (("overwrite", "--overwrite"), ("no_stop_on_success", "--no-stop-on-success"), ("tiled", "--tiled"), ("keep_depth", "--keep-depth"),
                      ("include_unverified", "--include-unverified"), ("same_episode_all_slots", "--same-episode-all-slots"), ("no_task_subdir", "--no-task-subdir"),
                      ("stock_handle_ag", "--stock-handle-ag")):
        if job.get(key):
            argv.append(flag)
    if worker_id is not None:
        argv += ["--worker-id", str(worker_id)]
    return parse_args(argv)


class WorldMismatch(RuntimeError):
    """The process's world (template signature / num_envs) cannot host this job: the worker restarts for it."""


def main(argv=None) -> int:
    return run_vec_job(parse_args(argv), shared={}, shutdown=True)


def run_vec_job(args, shared=None, shutdown=True) -> int:
    """The episodes of one job on this process's vectorized world; ``shared["vec_loader"]`` carries the world across jobs (the
    persistent worker passes the same dict for every job; the CLI a fresh one). A job whose template / num_envs differ from the
    loaded world raises :class:`WorldMismatch`."""
    import numpy as np
    import torch as th

    shared = {} if shared is None else shared

    from lab_tools import act_policy as AP
    from lab_tools import evaluator_patches

    evaluator_patches.install(keep_depth=args.keep_depth, no_planner=True)  # BEHAVIOR sensor set, no depth (QA: --stream-qa)
    import omnigibson as og
    from omnigibson.object_states import PoseReached  # noqa: F401

    from ispatialgym.oracle.executor import ExecutorConfig, SimState
    from ispatialgym.oracle.qa import Monitors, QAConfig
    from ispatialgym.oracle.recorder import Recorder, RecorderConfig, dump_json, save_png
    from ispatialgym.oracle.vec_loader import VecInstanceLoader
    from ispatialgym.oracle.vec_stream import Slot, VecStreamExecutor
    from ispatialgym.robot import conventions as C

    # -- episodes ----------------------------------------------------------------------------------------------------
    inst_doc = json.load(open(args.instances))
    inst_map = inst_doc.get("episodes", inst_doc)
    goals_index = os.path.join(args.dataset, "isg_meta", "goals", "index.json")
    goals = {g["episode_index"]: g for g in json.load(open(goals_index))["episodes"]} if os.path.exists(goals_index) else {}
    goals_root = args.goals_root or os.path.join(WT, "datasets", "goals", os.path.basename(os.path.normpath(args.dataset)))
    wrist_goals: Dict[int, Dict[str, str]] = {}
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
        g = goals.get(e, {})
        episodes.append({"episode_index": e, "instance": v["instance"], "tro": v["tro"], "train_task_index": int(g["train_task_index"]) if g else None,
                         "goal_png": os.path.join(args.dataset, g["png"]) if g else None, "verified": v.get("verified"),
                         "first_frame_mean_abs_diff": v.get("first_frame_mean_abs_diff"), "instance_index": v.get("instance_index"),
                         "bucket_hi_m": float(buckets[e]["bucket_hi_m"]) if e in buckets else None})
    episodes.sort(key=lambda d: d["episode_index"])
    lab_dir = args.lab_out if args.no_task_subdir else os.path.join(args.lab_out, args.task)
    raw_dir = args.raw_out if args.no_task_subdir else os.path.join(args.raw_out, args.task)
    os.makedirs(lab_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    todo = [ep for ep in episodes if args.overwrite or not os.path.exists(os.path.join(lab_dir, f"episode_{ep['episode_index']:06d}.json"))]
    print(f"VEC_EVAL mode={args.mode} run={args.run} step={args.step} split={args.split} episodes={len(episodes)} todo={len(todo)} num_envs={args.num_envs} tiled={args.tiled}", flush=True)
    if not todo:
        return 0
    if args.ckpt_path:
        actual = sha256_of(args.ckpt_path)
        if actual != args.ckpt_sha256:
            raise SystemExit(f"checkpoint sha256 mismatch: {actual} != {args.ckpt_sha256}")
    provenance = {"evaluator_commit": git_head(WT), "evaluator_root": WT, "b1k_commit": git_head(os.path.join(WT, "Behavior-1K")),
                  "act_commit": git_head(os.path.join(WT, "act-isg")) if args.mode == "policy" else None,
                  "bridge_sha256": {f: sha256_of(os.path.join(WT, "lab_tools", f)) for f in BRIDGE_FILES},
                  "evaluator_sha256": {f: sha256_of(os.path.join(WT, f)) for f in EVALUATOR_FILES},
                  "omnigibson_version": og.__version__, "vectorized": True, "num_envs": args.num_envs, "tiled_rendering": args.tiled, "warm_renders": args.warm_renders,
                  "enhanced_determinism": args.enhanced_determinism if args.enhanced_determinism is not None else (args.num_envs > 1), "fixed_handle_ag": not args.stock_handle_ag}

    # -- batches (one world signature per process; the first TRO decides the template) -------------------------------
    N = int(args.num_envs)
    batches: List[List[dict]] = []
    if args.same_episode_all_slots:
        for ep in todo:
            batches.append([ep] * N)
    else:
        for i in range(0, len(todo), N):
            batches.append(todo[i:i + N])
    if args.limit_batches:
        batches = batches[: args.limit_batches]
    if args.repeat and args.repeat > 1:
        batches = [b for b in batches for _ in range(args.repeat)]

    loader = shared.get("vec_loader")
    if loader is not None:
        from ispatialgym.oracle.vec_loader import world_payload, worlds_compatible

        tdir = os.path.dirname(os.path.dirname(os.path.abspath(batches[0][0]["tro"])))
        ok, bad = worlds_compatible(loader.payload, world_payload(tdir)) if os.path.abspath(tdir) != os.path.abspath(loader.template_dir) else (True, [])
        if loader.num_envs != N or not ok:
            raise WorldMismatch(f"loaded world: num_envs={loader.num_envs} template={loader.template_dir}; job needs num_envs={N} template={tdir}: {bad[:3]}")
        print(f"VEC_EVAL reusing the loaded world (batches so far {loader.batches}, scene loads {loader.scene_loads})", flush=True)
    else:
        loader = VecInstanceLoader(num_envs=N, behavior_output=True, tiled_rendering=args.tiled, settle_physics_steps=args.settle_steps, warm_renders=args.warm_renders,
                                   enhanced_determinism=args.enhanced_determinism, fixed_handle_ag=not args.stock_handle_ag,
                                   physics_overrides=json.loads(args.physics) if args.physics else None, dummy_bodies=args.dummy_bodies,
                                   render_overrides=json.loads(args.render) if args.render else None, scene_margin=args.scene_margin)
        t_build = time.time()
        loader.build_env(batches[0][0]["tro"])
        shared["vec_loader"] = loader
        print(f"VEC_EVAL world built in {round(time.time() - t_build, 1)}s: {loader.report}", flush=True)
    env, task = loader.env, loader.task
    cam_key = loader.cam_key
    fps = int(loader.cfg["env"]["action_frequency"])
    demo_cache: Dict[int, Any] = {}

    def slot_of(k: int) -> int:
        if args.slot_order == "reverse":
            return N - 1 - k
        if args.slot_order == "rotate":
            return (k + 1) % N
        return k

    t_proc = time.time()
    n_done = 0
    for b, batch in enumerate(batches):
        rep_suffix = f".rep{b % args.repeat}" if args.repeat and args.repeat > 1 else ""
        t_batch = time.time()
        seed0 = int(batch[0]["instance_index"]) if batch[0].get("instance_index") is not None else int(batch[0]["episode_index"])
        th.manual_seed(seed0)
        np.random.seed(seed0 % (2 ** 32))
        random.seed(seed0)
        placement: Dict[int, dict] = {slot_of(k): ep for k, ep in enumerate(batch)}
        tro_paths = [placement[i]["tro"] if i in placement else None for i in range(N)]
        started = utc()
        try:
            obs_by, info_by = loader.restore_batch(tro_paths)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            for i, ep in placement.items():
                rec = {"run": args.run, "step": args.step, "split": args.split, "task": args.task, "episode_index": ep["episode_index"], "instance": ep["instance"],
                       "error": f"restore_batch: {e!r}", "traceback": traceback.format_exc(), "slot": i, "num_envs": N, "batch": b, "tag": args.tag, "started_at": started,
                       "finished_at": utc(), "provenance": provenance, "success": None}
                with open(os.path.join(lab_dir, f"episode_{ep['episode_index']:06d}.json"), "w") as f:
                    json.dump(rec, f, indent=1, default=str)
            continue
        load_s = time.time() - t_batch
        for i, fail in loader.failed.items():
            ep = placement[i]
            rec = {"run": args.run, "step": args.step, "split": args.split, "task": args.task, "episode_index": ep["episode_index"], "instance": ep["instance"],
                   "instance_index": ep.get("instance_index"), "success": False, "status": "oracle_failed", "reason": fail["reason"], "error": fail["message"],
                   "exit_code": 2, "slot": i, "num_envs": N, "batch": b, "tag": args.tag, "started_at": started, "finished_at": utc(), "provenance": provenance,
                   "loader": loader.slot_reports[i]}
            with open(os.path.join(lab_dir, f"episode_{ep['episode_index']:06d}.json"), "w") as f:
                json.dump(rec, f, indent=1, default=str)
            print(f"VEC_EPISODE batch={b} slot={i} ep={ep['episode_index']} load failed: {fail['reason']}: {fail['message']}", flush=True)
        slots: List[Slot] = []
        per: Dict[int, dict] = {}
        for i, ep in placement.items():
            if i in loader.failed:
                continue
            e = ep["episode_index"]
            robot, scene = loader.robots[i], env.scenes[i]
            iid = os.path.basename(ep["tro"])[: -len("-tro_state.json")]
            out_dir = os.path.join(raw_dir, (iid if not args.same_episode_all_slots else f"{iid}.slot{i}") + rep_suffix)
            os.makedirs(out_dir, exist_ok=True)
            seed = int(ep["instance_index"]) if ep.get("instance_index") is not None else e
            record = {"run": args.run, "step": args.step, "checkpoint": {"repo": args.ckpt_repo, "file": args.ckpt_file, "sha256": args.ckpt_sha256} if args.mode == "policy" else None,
                      "split": "replay" if args.mode == "replay" else args.split, "replay_split": args.split if args.mode == "replay" else None,
                      "dataset": os.path.basename(os.path.normpath(args.dataset)), "task": args.task, "episode_index": e, "instance": ep["instance"],
                      "instance_index": ep.get("instance_index"), "seed": seed, "batch_seed": seed0, "success": None, "steps": None, "horizon": args.horizon, "wall_s": None,
                      "evaluator_commit": provenance["evaluator_commit"], "act_commit": provenance["act_commit"], "server_args": args.server_args if args.mode == "policy" else None,
                      "error": None, "score": None, "tag": args.tag, "stop_on_success": not args.no_stop_on_success, "instance_verified": ep.get("verified"),
                      "first_frame_mean_abs_diff": ep.get("first_frame_mean_abs_diff"), "started_at": started, "finished_at": None, "provenance": provenance,
                      "slot": i, "num_envs": N, "batch": b, "vectorized": True, "tiled_rendering": loader.report.get("tiled_rendering"), "stream_qa": args.stream_qa}
            if args.mode == "replay":
                record["policy"] = "lab_tools.vec_eval:ReplaySource (recorded actions, open loop)"
            per[i] = {"ep": ep, "record": record, "out_dir": out_dir, "t0": time.time()}
            try:
                obs0, info0 = obs_by[i], info_by[i]
                try:
                    save_png(os.path.join(out_dir, "init_rgb.png"), obs0[robot.name][cam_key]["rgb"])
                except Exception as ex:  # noqa: BLE001
                    print(f"init render not saved: {ex!r}", flush=True)
                cam = robot.sensors.get(cam_key)
                if args.mode == "policy":
                    goal_pngs = {"zed_link": ep["goal_png"], **{c: p for c, p in wrist_goals.get(e, {}).items()}}
                    ctx = {"server": args.server, "goal_rgb": load_goal(ep["goal_png"]), "goal_rgbs": {c: load_goal(p) for c, p in goal_pngs.items()},
                           "task_id": ep["train_task_index"], "horizon": args.horizon, "stop_on_success": not args.no_stop_on_success,
                           "expect_action_horizon": args.action_horizon, "robot": robot.name}
                    source = TracedPolicy(AP.ActPolicy(ctx=ctx), PoseTracer(ctx, robot, scene, cam, loader.tracked(i)))
                    record["goal_png"] = os.path.relpath(ep["goal_png"], args.dataset)
                    record["goal_pngs"] = {c: os.path.relpath(p, WT) for c, p in goal_pngs.items()}
                    record["goal_from_episode"] = None
                else:
                    if e not in demo_cache:
                        from lab_tools.replay_episodes import load_episode_arrays

                        demo_cache[e] = load_episode_arrays(args.dataset, e)
                    actions_demo, states_demo, length = demo_cache[e]
                    ctx = {"actions": actions_demo, "horizon": args.horizon, "stop_on_success": not args.no_stop_on_success}
                    source = ReplaySource(ctx, robot, scene, cam, tracked=loader.tracked(i))
                per[i]["ctx"] = ctx
                per[i]["source"] = source  # this slot's source (its tracer saves this slot's arrays)
                cam_in_zed = C.camera_offset_in_zed(robot, cam_key)
                view, tview = loader.view(i), loader.task_view(i)  # this clone's scene / scope for the monitors, state and recorder
                state = SimState(view, tview, robot, cam_key, cam_in_zed=cam_in_zed)
                monitors = Monitors(view, robot, QAConfig(max_episode_steps=args.max_steps), cam_key=cam_key, cam_in_zed=cam_in_zed)
                tracked = loader.tracked(i)
                rec_cfg = RecorderConfig(hdf5=False, lerobot=False, video=True, video_size=args.video_size, all_cameras=False)
                recorder = Recorder(out_dir, view, robot, cam_key, tracked, fps=fps, cfg=rec_cfg, task_name=loader.activity.replace("_", " "), instance_id=iid)
                recorder.start(obs0, info0, state0=None)
                cfg = ExecutorConfig(max_steps=args.max_steps, settle_min_steps=args.settle_steps, goal_image=False, record_state=False, demo_gating=())
                slots.append(Slot(idx=i, robot=robot, source=source, cfg=cfg, task=tview, monitors=monitors, recorder=recorder, tracked=tracked,
                                  held=state.held_objects, never_held_window=10))
                per[i].update({"monitors": monitors, "recorder": recorder, "cam_in_zed": cam_in_zed, "tracked": tracked, "iid": iid})
            except Exception as ex:  # noqa: BLE001
                record["error"] = repr(ex)
                record["traceback"] = traceback.format_exc()
                traceback.print_exc()
        idle = {i: loader.hold_action(i) for i in range(N) if i not in {s.idx for s in slots}}
        obs_list = [obs_by.get(i) for i in range(N)]
        info_list = [info_by.get(i) for i in range(N)]
        t_run = time.time()
        results: Dict[int, dict] = {}
        try:
            vex = VecStreamExecutor(env, slots, sim=og.sim, n_render_iterations=1, num_envs=N, idle_actions=idle, stream_qa=args.stream_qa)
            for s, r in zip(slots, vex.run(obs_list, info_list)):
                results[s.idx] = r
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            for s in slots:
                per[s.idx]["record"]["error"] = per[s.idx]["record"].get("error") or f"vec run: {ex!r}"
                per[s.idx]["record"]["traceback"] = traceback.format_exc()
        episode_s = time.time() - t_run
        # -- per slot: close the recorder, meta.json, score, record ----------------------------------------------------
        for s in slots:
            i = s.idx
            d = per[i]
            ep, record, out_dir, ctx = d["ep"], d["record"], d["out_dir"], d["ctx"]
            source = d["source"]
            e = ep["episode_index"]
            robot, scene = loader.robots[i], env.scenes[i]
            episode = results.get(i) or {}
            meta: Dict[str, Any] = {"status": "error", "mode": args.mode, "argv": sys.argv[1:], "hostname": os.uname().nodename, "started_utc": started, "provenance": provenance,
                                    "reused_world": loader.batches > 1, "instance_id": d.get("iid"), "template_dir": loader.slot_reports[i].get("template_dir"),
                                    "activity": loader.activity, "activity_definition_id": loader.did, "scene": loader.tmeta.get("scene"), "tro_path": ep["tro"],
                                    "robot_name": robot.name, "out_dir": out_dir, "loader": {**{k: v for k, v in loader.report.items() if k != "trav_cells_blocked"}, **loader.slot_reports[i]},
                                    "robot_base_init": loader.slot_reports[i].get("robot_base_init_scene"), "tracked_objects": {k: v.name for k, v in d.get("tracked", {}).items()},
                                    "vec": {"num_envs": N, "env_idx": i, "batch": b, "batch_steps": episode.get("batch_steps"), "tiled_rendering": loader.report.get("tiled_rendering"),
                                            "scene_offset": loader.report.get("scene_offsets", [None] * N)[i]}}
            try:
                cz = d.get("cam_in_zed")
                if cz is not None:
                    meta["camera_in_zed"] = {"position": cz[0].tolist(), "orientation": cz[1].tolist(), "constant_dev_mm": 1000.0 * float(th.linalg.norm(cz[0] - C.CAMERA_IN_ZED[0]))}
                recorder = d.get("recorder")
                paths = {}
                if recorder is not None:
                    if recorder.last_rgb is not None:
                        save_png(os.path.join(out_dir, "final_rgb.png"), recorder.last_rgb)
                    qa = (episode.get("qa") or {}).get("stats") if episode else None
                    paths = recorder.close(meta, qa=qa, success=bool(episode.get("success")))
                if episode:
                    meta.update({k: v for k, v in episode.items() if k not in ("goal_observation", "timings")})
                    meta["status"] = "success" if episode["success"] else "oracle_failed"
                    meta["qa_reason"] = episode.get("reason")
                    meta["timings"] = {"load_s": load_s, "planner_build_s": 0.0, "episode_s": episode_s, **{f"executor_{k}": v for k, v in episode.get("timings", {}).items()},
                                       "total_s": time.time() - d["t0"]}
                meta["video"] = paths.get("video")
                meta["video_error"] = paths.get("video_error")
                meta["wall_time_s"] = time.time() - d["t0"]
                if record.get("error") and not episode:
                    meta["error"] = record["error"]
                dump_json(os.path.join(out_dir, "meta.json"), meta)
                if episode:
                    errors = episode.get("errors") or {}
                    record.update({
                        "success": bool(episode.get("success")), "steps": int(ctx.get("steps") or episode.get("n_action_steps") or 0), "stop_reason": ctx.get("stop_reason"),
                        "exit_code": 0 if episode.get("success") else 1, "status": meta["status"], "reason": episode.get("reason"), "n_steps_total": episode.get("n_steps"),
                        "n_action_steps": episode.get("n_action_steps"), "first_success_step": episode.get("first_success_step"), "binary_success_ever": episode.get("binary_success_ever"),
                        "binary_success_final": episode.get("binary_success_final"), "settled_success": episode.get("settled_success"), "settle_windows": episode.get("settle_windows"),
                        "video": os.path.relpath(meta["video"], WT) if meta.get("video") else None, "raw_dir": os.path.relpath(out_dir, WT), "timings": meta.get("timings"),
                        "batch_steps": episode.get("batch_steps"), "release_check": episode.get("release_check"), "stream_qa_log": episode.get("stream_qa"),
                    })
                    if args.mode == "policy":
                        record["server_metadata"] = {k2: v for k2, v in (ctx.get("server_metadata") or {}).items()
                                                     if k2 in ("policy", "action_horizon", "chunk_size", "execution_mode", "checkpoint_step", "goal_fusion", "goal_views", "task_map")}
                        record["goal_views_sent"] = ctx.get("goal_views_sent")
                        record["goal_sha256"] = ctx.get("goal_sha256")
                        record["task_id_sent"] = ctx.get("task_id_sent")
                        record["server_request_s_mean"] = ctx.get("request_s_mean")
                        record["server_request_s_max"] = ctx.get("request_s_max")
                    if args.task == TASK:
                        score = score_relocalization_env(meta, robot, scene, ep)
                    elif args.task.startswith("alignment"):
                        from lab_tools.eval_episodes import score_alignment

                        score = score_alignment(meta)
                    elif args.task.startswith(("mental_rotation", "object_scaling")):
                        from lab_tools.eval_episodes import score_pose_target

                        score = score_pose_target(meta)
                    else:
                        from lab_tools.eval_episodes import score_articulation

                        score = score_articulation(meta)
                    qa = meta.get("qa") or {}
                    score["native"] = {"max_position_error": errors.get("max_position_error"), "max_orientation_error": errors.get("max_orientation_error"),
                                       "errors_final_step": episode.get("errors_final_step"), "reason": episode.get("reason"), "verdict": episode.get("verdict"),
                                       "contacts": (qa.get("collisions") or {}).get("n_events") if isinstance(qa.get("collisions"), dict) else None,
                                       "collisions_by_object": qa.get("collisions_by_object") or (qa.get("collisions") or {}).get("by_object") if isinstance(qa, dict) else None}
                    if args.task == TASK:
                        try:
                            score["closest_approach"] = closest_approach(ctx.get("step_errors"))
                        except Exception as e3:  # noqa: BLE001
                            score["closest_approach"] = {"error": repr(e3)}
                        try:
                            from lab_tools.eval_episodes import base_and_trunk_vs_goal

                            acts = ctx.get("actions") if args.mode == "policy" else [np.asarray(a) for a in ctx["actions"][: int(ctx.get("actions_streamed") or 0)]]
                            score["base_trunk_vs_goal"] = base_and_trunk_vs_goal({"robot": SceneRelativeRobot(robot, scene)}, ep, acts)
                        except Exception as e3:  # noqa: BLE001
                            score["base_trunk_vs_goal"] = {"error": repr(e3)}
                    if args.mode == "replay":
                        from lab_tools.replay_episodes import end_vs_goal, goal_base_pose, headline_deviation, trajectory_block

                        actions_demo, states_demo, length = demo_cache[e]
                        rep = ctx.get("base_xyyaw") or []
                        if rep:
                            x0, y0, yaw0 = rep[0]
                        else:
                            p = (meta.get("robot_base_init") or {}).get("position") or [0, 0, 0]
                            x0, y0, yaw0 = float(p[0]), float(p[1]), _yaw((meta.get("robot_base_init") or {}).get("orientation") or [0, 0, 0, 1])
                        streamed = int(ctx.get("actions_streamed") or 0)
                        replay_qvel = np.asarray(ctx.get("base_qvel") or np.zeros((0, 3)), dtype=np.float64)
                        traj = trajectory_block(rep, replay_qvel, states_demo[:, 0:3], x0, y0, yaw0, actions=actions_demo, streamed=streamed)
                        traj["end_vs_goal_base"] = end_vs_goal(rep, goal_base_pose(ep["tro"]))
                        demo_dr = traj.pop("demo_dead_reckoned_raw")
                        demo_cal = traj.pop("demo_dead_reckoned_calibrated", None)
                        score["replay"] = {"demo_length": length, "actions_streamed": streamed, "stop_reason": ctx.get("stop_reason"), "trajectory": traj,
                                           "init_pose_source": "simulator pose at the first callable call" if rep else "meta.robot_base_init",
                                           "replay_end_camera_pose": (ctx.get("camera_pose") or [None])[-1]}
                        record.update(headline_deviation(traj))
                        record["demo_length"] = length
                        record["actions_streamed"] = streamed
                        source.tracer.save(out_dir, prefix="replay")
                        np.save(os.path.join(out_dir, "demo_base_xyyaw_dead_reckoned.npy"), np.asarray(demo_dr, dtype=np.float32))
                        if demo_cal is not None:
                            np.save(os.path.join(out_dir, "demo_base_xyyaw_dead_reckoned_calibrated.npy"), np.asarray(demo_cal, dtype=np.float32))
                        np.save(os.path.join(out_dir, "demo_base_qvel.npy"), np.asarray(states_demo[:, 0:3], dtype=np.float32))
                        record["replay_objects"] = ctx.get("object_names")
                    else:
                        record["trace"] = source.tracer.save(out_dir, prefix="trace")  # per-slot closed-loop trace (gate 3f)
                        record["trace_objects"] = ctx.get("object_names")
                    record["score"] = score
                    if args.mode == "policy" and ctx.get("actions"):
                        np.save(os.path.join(out_dir, "actions.npy"), np.stack(ctx["actions"]).astype(np.float32))
                        record["actions_npy"] = os.path.relpath(os.path.join(out_dir, "actions.npy"), WT)
                    if ctx.get("step_errors"):
                        np.save(os.path.join(out_dir, "errors.npy"), np.asarray(ctx["step_errors"], dtype=np.float32))
                        record["errors_npy"] = os.path.relpath(os.path.join(out_dir, "errors.npy"), WT)
            except Exception as ex:  # noqa: BLE001
                record["error"] = record.get("error") or repr(ex)
                record["traceback"] = traceback.format_exc()
                traceback.print_exc()
            record["memory"] = memory_snapshot()
            record["worker"] = args.worker_id
            record["wall_s"] = round(time.time() - d["t0"], 1)
            record["batch_wall_s"] = round(time.time() - t_batch, 1)
            record["finished_at"] = utc()
            name = f"episode_{e:06d}{'' if not args.same_episode_all_slots else f'.slot{i}'}{rep_suffix}.json"
            with open(os.path.join(lab_dir, name + ".tmp"), "w") as f:
                json.dump(record, f, indent=1, default=str)
            os.replace(os.path.join(lab_dir, name + ".tmp"), os.path.join(lab_dir, name))
            n_done += 1
            sc = record.get("score") or {}
            print(f"VEC_EPISODE batch={b} slot={i} ep={e} inst={ep['instance'].rsplit('__', 1)[-1]} success={record['success']} reason={record.get('reason')} "
                  f"steps={record['steps']} stop={record.get('stop_reason')} first_success={record.get('first_success_step')} geodesic={sc.get('geodesic_distance_m')} "
                  f"pos={sc.get('position_error_m')} wall={record['wall_s']}s err={record['error']}", flush=True)
        mem = memory_snapshot()
        print(f"VEC_BATCH {b + 1}/{len(batches)} slots={len(slots)} load={round(load_s, 1)}s run={round(episode_s, 1)}s batch_steps={max([r.get('batch_steps') or 0 for r in results.values()] or [0])} "
              f"gpu={mem.get('gpu_mib')} rss={mem.get('rss_mib')}", flush=True)
        try:
            evaluator_patches.after_episode_cleanup()
        except Exception:  # noqa: BLE001
            pass
    print(f"VEC_DONE episodes={n_done} batches={len(batches)} wall={round(time.time() - t_proc, 1)}s", flush=True)
    if shutdown:
        try:
            og.shutdown()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
