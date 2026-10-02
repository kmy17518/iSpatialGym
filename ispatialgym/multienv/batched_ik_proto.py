"""Prototype (measurement only, not used by the sampler): screening IK for several placement attempts in one cuRobo call.

    python -m ispatialgym.multienv.batched_ik_proto [--attempts 4] [--poses 64] [--seeds 12] [--repeats 10] [--out result.json]

Today the sampler screens one batch of eef poses per placement attempt (``R1ProCuRobo.solve_ik_batch``: an IKSolver with
``max_batch_size = 64`` and 12 seeds over the one shared world), syncing the world between attempts. Here every attempt
has its own world (the same static meshes plus the attempt's placed object at its own pose) and two layouts are timed:

  sequential  one world store (num_envs = 1), the backend's solver; per attempt: move the placed object, solve its poses.
  multi-env   a world store with num_envs = A (env a = attempt a's world) and one solver call of A x poses problems.
              cuRobo's ``multi_env`` gives every problem its own environment (num_envs = max_batch_size); the goal
              buffer's ``idxs_env`` is set to problem // poses instead (``GoalRegistry.create_idx`` patched while this
              runs), so A worlds serve A x poses problems; the robot's kinematics keeps max_batch_size identical sphere
              sets (cuRobo's multi_env layout), each well under a megabyte.

Standalone: cuRobo + the R1Pro model (``r1pro_model``: the derived URDF, v1 spheres, the right-arm embodiment with the
JoyLo reset posture locked), no simulator. Synthetic worlds: a table, boxes and cylinders on it, a shelf above, the
placed object at an attempt-specific spot on the table; the goals are forward-kinematics poses of random arm
configurations near the placed object (so success depends on collisions). Reports per layout the median solve time per
attempt over ``--repeats`` rounds (CUDA graphs captured in the warm-up rounds), torch's peak allocated GPU memory and the
device's used memory, and the agreement of the two layouts' success masks; ``--blocked`` fills attempt 1's world with a
box over the workspace (every one of its poses must fail in both layouts: the env mapping is in effect).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time

import numpy as np
import torch as th
import trimesh

from curobo._src.geom.collision import SceneCollisionCfg, create_collision_checker
from curobo._src.rollout import goal_registry as GR
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.util.warp import init_warp
from curobo.scene import Cuboid, Mesh, Scene
from curobo.types import DeviceCfg, GoalToolPose, JointState
from curobo.types import Pose as CuPose

from ispatialgym.oracle import r1pro_model as M

FRAME = M.EEF_LINK["right"]
MESH_QUERY_MIN_DIMS = 0.4  # as R1ProCuRobo._patch_mesh_query_dims
_ENV_PER = [0]


def _create_idx(cls, pose_batch_size, multi_env, num_seeds, device_cfg, seed_goal_state=None, repeat_seed_idx_buffers=False):
    g = _ORIG_CREATE_IDX(cls, pose_batch_size, multi_env, num_seeds, device_cfg, seed_goal_state, repeat_seed_idx_buffers)
    if multi_env and _ENV_PER[0]:
        g.idxs_env = (g.idxs_link_pose // _ENV_PER[0]).to(g.idxs_env.dtype)
    return g


_ORIG_CREATE_IDX = GR.GoalRegistry.create_idx.__func__
GR.GoalRegistry.create_idx = classmethod(_create_idx)


def device_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return int(out.stdout.split()[0])
    except Exception:  # noqa: BLE001
        return None


def box_mesh(name, extents, pos):
    tm = trimesh.creation.box(extents=extents)
    return Mesh(name=name, pose=[*pos, 1.0, 0.0, 0.0, 0.0], vertices=np.asarray(tm.vertices, np.float32), faces=np.asarray(tm.faces, np.int64))


def cyl_mesh(name, radius, height, pos):
    tm = trimesh.creation.cylinder(radius=radius, height=height, sections=24)
    return Mesh(name=name, pose=[*pos, 1.0, 0.0, 0.0, 0.0], vertices=np.asarray(tm.vertices, np.float32), faces=np.asarray(tm.faces, np.int64))


def static_meshes(rng):
    out = [box_mesh("table_top", [0.7, 1.4, 0.05], [0.8, 0.0, 0.725]), box_mesh("shelf", [0.4, 1.0, 0.03], [0.95, 0.0, 1.45])]
    for i in range(10):
        x, y = rng.uniform(0.55, 1.05), rng.uniform(-0.6, 0.6)
        if i % 2:
            h = rng.uniform(0.08, 0.3)
            out.append(cyl_mesh(f"cyl_{i}", rng.uniform(0.03, 0.06), h, [x, y, 0.75 + h / 2]))
        else:
            e = rng.uniform(0.05, 0.2, size=3)
            out.append(box_mesh(f"box_{i}", e.tolist(), [x, y, 0.75 + e[2] / 2]))
    return out


def placed_pose(rng):
    return [rng.uniform(0.55, 0.85), rng.uniform(-0.45, 0.15), 0.75 + 0.125]


_WORLD_OPTS = {"ground": True}


def world(statics, placed, blocker=False):
    meshes = list(statics) + [box_mesh("placed", [0.12, 0.12, 0.25], placed)]
    if blocker:
        meshes.append(box_mesh("blocker", [1.6, 2.4, 1.6], [0.6, 0.0, 1.0]))
    ground = [Cuboid(name="ground", dims=[60.0, 60.0, 0.1], pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0])] if _WORLD_OPTS["ground"] else None
    return Scene(mesh=meshes, cuboid=ground)


def make_solver(robot_dict, device_cfg, checker, batch, seeds, multi_env):
    cfg = IKSolverCfg.create(robot=robot_dict, scene_model=None, collision_cache=None, self_collision_check=True, device_cfg=device_cfg,
                             num_seeds=seeds, position_tolerance=0.005, orientation_tolerance=0.05, use_cuda_graph=True,
                             optimizer_collision_activation_distance=0.01, max_batch_size=batch, multi_env=multi_env)
    return IKSolver(cfg, checker)


def patch_dims(checker):
    meshes = getattr(checker.data, "meshes", None)
    if meshes is not None and meshes.dims is not None:
        with th.no_grad():
            meshes.dims[..., :3].clamp_(min=MESH_QUERY_MIN_DIMS)


def solve(ik, pos, quat, q0):
    b = pos.shape[0]
    goal = GoalToolPose.from_poses({FRAME: CuPose(position=pos, quaternion=quat)}, ordered_tool_frames=list(ik.kinematics.tool_frames), num_goalset=1)
    cur = JointState.from_position(q0.unsqueeze(0).repeat(b, 1).clone(), joint_names=list(ik.joint_names))
    res = ik.solve_pose(goal, current_state=cur, seed_config=cur.position.view(b, 1, -1).clone(), return_seeds=1)
    return res.success.reshape(-1)[:b].clone()


def goals_near(ik, centre, n, rng, q0, device):
    lim = ik.kinematics.get_joint_limits().position
    lo, hi = lim[0], lim[1]
    pos_out, quat_out = [], []
    centre = th.tensor(centre, device=device)
    while sum(p.shape[0] for p in pos_out) < n:
        q = lo + (hi - lo) * th.tensor(rng.uniform(size=(4096, lo.shape[0])), device=device, dtype=lo.dtype)
        st = ik.kinematics.compute_kinematics(JointState.from_position(q, joint_names=list(ik.joint_names)))
        p = st.tool_poses.get_link_pose(FRAME)
        pos, quat = p.position.reshape(-1, 3), p.quaternion.reshape(-1, 4)
        keep = th.linalg.norm(pos - centre, dim=-1) < 0.3
        pos_out.append(pos[keep])
        quat_out.append(quat[keep])
    return th.cat(pos_out)[:n].contiguous(), th.cat(quat_out)[:n].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--poses", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mesh-cache", type=int, default=64)
    ap.add_argument("--blocked", action="store_true")
    ap.add_argument("--ground", action="store_true",
                    help="add the backend's ground slab (top at z = 0): with this standalone robot setup its wheel spheres rest on "
                         "it and every IK fails, so it is off by default")
    ap.add_argument("--empty", action="store_true", help="no static meshes (the placed object only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A, P = args.attempts, args.poses
    _WORLD_OPTS["ground"] = args.ground
    device_cfg = DeviceCfg(device=th.device("cuda:0"), dtype=th.float32)
    init_warp(device_cfg=device_cfg)
    rng = np.random.default_rng(0)
    emb = M.EMBODIMENTS["right_arm"]
    default_q = dict(zip(M.ALL_JOINTS, M.JOYLO_RESET_JOINT_POS))
    robot_dict = M.build_curobo_robot_dict(emb, M.generate_holonomic_urdf(), {n: default_q[n] for n in emb.locked_joints}, default_q,
                                           M.load_v1_kinematics_yaml())
    statics = [] if args.empty else static_meshes(rng)
    placed = [placed_pose(rng) for _ in range(A)]
    res = {"attempts": A, "poses": P, "seeds": args.seeds, "mesh_cache": args.mesh_cache, "static_meshes": len(statics), "blocked": args.blocked,
           "gpu": th.cuda.get_device_name(0)}
    dev0 = device_used_mib()

    # sequential: the backend's layout
    th.cuda.reset_peak_memory_stats()
    checker = create_collision_checker(SceneCollisionCfg(device_cfg=device_cfg, scene_model=None, cache={"mesh": args.mesh_cache, "cuboid": 4}))
    checker.load_collision_model(world(statics, placed[0], blocker=args.blocked))
    patch_dims(checker)
    ik = make_solver(robot_dict, device_cfg, checker, P, args.seeds, multi_env=False)
    q0 = th.tensor([default_q[n] for n in ik.joint_names], device=device_cfg.device, dtype=th.float32)
    goals = [goals_near(ik, placed[a], P, rng, q0, device_cfg.device) for a in range(A)]

    def seq_round():
        out = []
        for a in range(A):
            checker.data.meshes.update_pose("placed", w_obj_pose=CuPose(position=th.tensor([placed[a]], device=device_cfg.device),
                                                                      quaternion=th.tensor([[1.0, 0.0, 0.0, 0.0]], device=device_cfg.device)))
            if args.blocked:
                checker.data.meshes.set_enabled("blocker", a == 1)
            out.append(solve(ik, goals[a][0], goals[a][1], q0))
        th.cuda.synchronize()
        return th.stack(out)

    t_seq = []
    for r in range(args.warmup + args.repeats):
        t0 = time.perf_counter()
        succ_seq = seq_round()
        if r >= args.warmup:
            t_seq.append(time.perf_counter() - t0)
    res["sequential"] = {"round_s_median": round(statistics.median(t_seq), 4), "per_attempt_ms": round(1000 * statistics.median(t_seq) / A, 2),
                         "torch_peak_mib": round(th.cuda.max_memory_allocated() / 2**20, 1), "device_used_mib": device_used_mib(),
                         "success_per_attempt": succ_seq.sum(dim=1).tolist()}
    print("BATCHED_IK_PROTO " + json.dumps({"layout": "sequential", **res["sequential"]}), flush=True)
    del ik, checker
    th.cuda.empty_cache()

    # multi-env: one call, A worlds
    th.cuda.reset_peak_memory_stats()
    _ENV_PER[0] = P
    checker = create_collision_checker(SceneCollisionCfg(device_cfg=device_cfg, scene_model=None, cache={"mesh": args.mesh_cache, "cuboid": 4},
                                                         num_envs=A))
    for a in range(A):
        checker.load_collision_model(world(statics, placed[a], blocker=args.blocked and a == 1), env_idx=a)
    patch_dims(checker)
    ik = make_solver(robot_dict, device_cfg, checker, A * P, args.seeds, multi_env=True)
    pos_all = th.cat([g[0] for g in goals])
    quat_all = th.cat([g[1] for g in goals])

    t_me = []
    for r in range(args.warmup + args.repeats):
        t0 = time.perf_counter()
        succ_me = solve(ik, pos_all, quat_all, q0).reshape(A, P)
        th.cuda.synchronize()
        if r >= args.warmup:
            t_me.append(time.perf_counter() - t0)
    agree = (succ_me.cpu() == succ_seq.cpu()).float().mean().item()
    res["multi_env"] = {"round_s_median": round(statistics.median(t_me), 4), "per_attempt_ms": round(1000 * statistics.median(t_me) / A, 2),
                        "torch_peak_mib": round(th.cuda.max_memory_allocated() / 2**20, 1), "device_used_mib": device_used_mib(),
                        "success_per_attempt": succ_me.sum(dim=1).tolist()}
    res["agreement"] = round(agree, 4)
    res["speedup"] = round(statistics.median(t_seq) / statistics.median(t_me), 2)
    res["device_used_mib_before"] = dev0
    print("BATCHED_IK_PROTO " + json.dumps({"layout": "multi_env", **res["multi_env"], "agreement": res["agreement"], "speedup": res["speedup"]}), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
