"""Static probe of the velocity readings PoseSettled judges the R1Pro by (owner 2026-09-24: "test the articulation on a pod").

Background (alignment-world_vertical 0923-000539-fixes4): with the robot physically at rest (base pose drift 0.1 um, joint drift
< 0.2 mrad over 191 steps) the base-footprint linear velocity reads 1.5-2.5 mm/s on every step (PoseSettled gate: 1 mm/s) and
the two wrist joint-5 velocities 0.037 rad/s (gate: 0.02 rad/s), so ``pose_settled(agent)`` never holds. The virtual base joints
show a constant, non-integrating -1.0 mm/s (x) / -0.5 mm/s (z) plus zero-mean chatter flipping sign every other step. Hypothesis:
the reading is the solver's per-substep contact / drive correction residual of a 250 kg chassis (the BEHAVIOR-1K evaluator mass,
applied since 2026-09-21; 47.6 kg stock) standing on three wheel spheres, with the articulation running 32 position / 1 velocity
solver iterations - not motion.

The probe holds the robot still (the run_isg controller configuration, the JoyLo reset posture, fists closed, base commanded
zero) and records per control step what PoseSettled reads - ``robot.get_linear_velocity()`` (= the base footprint link),
``get_angular_velocity()``, every joint velocity - together with the pose / joint drift, for these variants:

* ``stock_47kg``      - the stock chassis mass, 32 / 1 iterations (the 0918-000750 setting)
* ``eval_250kg``      - the evaluator mass, 32 / 1 (the current run_isg setting)
* ``eval_250kg_vi8``  / ``eval_250kg_vi32`` - more velocity iterations (the residual should shrink if it is a solver residual)
* ``eval_250kg_pi64`` - more position iterations
* ``eval_250kg_nograv`` - gravity switched off on the chassis (no wheel-floor load): isolates the contact mechanism

on the empty floor plane (fast) and, with ``--scenes empty,house``, in the demo's living room (house_single_floor, the stance
of the painting demo). Writes ``<out>/results.json`` and ``<out>/results.md``.

    python scripts/experiments/probe_base_velocity_artifact.py --out $SMOKE_OUT/probe_base_velocity --scenes empty,house
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import torch as th  # noqa: E402

#: the robot entry run_isg builds (demo.hdf5 ``config`` of 0923-000539-fixes4, proprio only)
ROBOT_CFG: Dict[str, Any] = {
    "name": "robot_r1", "model": "r1pro", "obs_modalities": ["proprio"], "action_normalize": False, "self_collisions": True,
    "grasping_mode": "assisted",
    "reset_joint_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.025, -1.45, -0.47, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.05, 0.05, 0.05, 0.05],
    "controller_config": {
        "base": {"name": "HolonomicBaseJointController", "motor_type": "velocity", "vel_kp": 150, "command_input_limits": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                 "command_output_limits": [[-0.75, -0.75, -1.0], [0.75, 0.75, 1.0]], "use_impedances": False},
        "trunk": {"name": "JointController", "motor_type": "position", "pos_kp": 150, "command_input_limits": None, "command_output_limits": None,
                  "use_impedances": False, "use_delta_commands": False},
        "arm_left": {"name": "JointController", "motor_type": "position", "pos_kp": 150, "command_input_limits": None, "command_output_limits": None,
                     "use_impedances": False, "use_delta_commands": False},
        "arm_right": {"name": "JointController", "motor_type": "position", "pos_kp": 150, "command_input_limits": None, "command_output_limits": None,
                      "use_impedances": False, "use_delta_commands": False},
        "gripper_left": {"name": "MultiFingerGripperController", "mode": "smooth", "command_input_limits": "default", "command_output_limits": "default"},
        "gripper_right": {"name": "MultiFingerGripperController", "mode": "smooth", "command_input_limits": "default", "command_output_limits": "default"},
    },
}
ENV_CFG = {"action_frequency": 30, "rendering_frequency": 30, "physics_frequency": 120, "device": None, "automatic_reset": False,
           "flatten_action_space": False, "flatten_obs_space": False}
#: the painting demo's stance in house_single_floor / living_room_0 (base facing +y)
HOUSE_POSE = ([17.3247, 3.1685, 0.005277], [-5.44e-05, 6.09e-05, 0.62133718, 0.78354335])
EMPTY_POSE = ([0.0, 0.0, 0.005277], [0.0, 0.0, 0.0, 1.0])

#: postures of the painting demo 0923-000539-fixes4 (joint_positions at its first / last step): torso 4, left arm 7, right arm 7.
#: The JoyLo reset posture (every arm joint 0) is NOT usable here: with self-collisions on it reads constant-sign velocities of
#: 0.03-0.3 rad/s on the arm joints and 11 mm/s on the base with zero motion (arms against the torso; local check 2026-09-24).
POSTURES: Dict[str, Dict[str, List[float]]] = {
    "settle": {"torso": [0.7375, -0.925, -0.2608, 0.0], "left": [-0.3875, 0.1737, -0.386, -2.0433, -0.0091, -0.4372, 0.6868],
               "right": [0.0731, -2.2405, -0.7271, -1.4505, 2.0892, 0.5759, -0.605], "fingers_closed": True},
    "init": {"torso": [0.45, -0.4, 0.0, 0.0], "left": [0.0, 1.57, 0.0, -1.57, 1.57, 0.0, 0.0], "right": [0.0001, -1.57, 0.0, -1.57, -1.57, 0.0, 0.0],
             "fingers_closed": False},
}

STOCK_MASS = 47.63
EVAL_MASS = 250.0
VARIANTS: List[Dict[str, Any]] = [
    {"name": "stock_47kg", "mass": STOCK_MASS, "vel_iters": 1, "pos_iters": 32, "gravity": True},
    {"name": "eval_250kg", "mass": EVAL_MASS, "vel_iters": 1, "pos_iters": 32, "gravity": True},
    {"name": "eval_250kg_vi8", "mass": EVAL_MASS, "vel_iters": 8, "pos_iters": 32, "gravity": True},
    {"name": "eval_250kg_vi32", "mass": EVAL_MASS, "vel_iters": 32, "pos_iters": 32, "gravity": True},
    {"name": "eval_250kg_pi64", "mass": EVAL_MASS, "vel_iters": 1, "pos_iters": 64, "gravity": True},
    {"name": "eval_250kg_nograv", "mass": EVAL_MASS, "vel_iters": 1, "pos_iters": 32, "gravity": False},
    # the x / y / rz velocity drives hold the chassis with the default damping 1e7 (controller_base.DEFAULT_ISAAC_KP for velocity control) -
    # redundant with the wheel friction; softer drives test whether the redundancy is what the solver leaves as a residual
    {"name": "eval_250kg_kd1e5", "mass": EVAL_MASS, "vel_iters": 1, "pos_iters": 32, "gravity": True, "base_kd": 1e5},
    {"name": "eval_250kg_kd1e3", "mass": EVAL_MASS, "vel_iters": 1, "pos_iters": 32, "gravity": True, "base_kd": 1e3},
]
#: PoseSettled's gates (omnigibson.object_states.pose_settled macros)
LIN_TOL, ANG_TOL, JOINT_TOL = 0.001, 0.005, 0.02


def hold_action(robot, fingers_closed: bool = True) -> th.Tensor:
    """The Layer-1 hold: base velocity 0, trunk / arm position targets = the measured joints, fists closed (JoyLo smooth -1) or open (+1)."""
    a = th.zeros(int(robot.action_dim))
    q = robot.get_joint_positions()
    idx = robot.controller_action_idx
    a[idx["trunk"]] = q[robot.trunk_control_idx]
    for arm in ("left", "right"):
        a[idx[f"arm_{arm}"]] = q[robot.arm_control_idx[arm]]
        a[idx[f"gripper_{arm}"]] = -1.0 if fingers_closed else 1.0
    return a


def set_posture(robot, posture: Dict[str, Any]) -> None:
    """Put the trunk and the arms at ``posture`` (a collision-free posture of the demo) and stop every joint."""
    q = robot.get_joint_positions().clone()
    q[robot.trunk_control_idx] = th.tensor(posture["torso"], dtype=q.dtype)
    for arm in ("left", "right"):
        q[robot.arm_control_idx[arm]] = th.tensor(posture[arm], dtype=q.dtype)
    for arm in ("left", "right"):
        q[robot.gripper_control_idx[arm]] = 0.0 if posture.get("fingers_closed", True) else 0.05
    robot.set_joint_positions(q, drive=False)
    robot.set_joint_velocities(th.zeros_like(q), drive=False)
    robot.keep_still()


def set_chassis(robot, mass: float, gravity: bool) -> Dict[str, float]:
    import omnigibson as og

    link = robot.base_footprint_link
    before = float(link.mass)
    if abs(before - mass) > 1e-6:
        og.sim.stop()
        link.mass = float(mass)
        og.sim.play()
        if hasattr(og.sim, "update_handles"):
            og.sim.update_handles()
    if gravity:
        link.enable_gravity()
    else:
        link.disable_gravity()
    return {"mass_before": before, "mass_after": float(link.mass)}


def set_solver(robot, vel_iters: int, pos_iters: int) -> Dict[str, Any]:
    if int(robot.solver_velocity_iteration_count) != int(vel_iters):
        robot.solver_velocity_iteration_count = int(vel_iters)
    if int(robot.solver_position_iteration_count) != int(pos_iters):
        robot.solver_position_iteration_count = int(pos_iters)
    return {"velocity_iterations": int(robot.solver_velocity_iteration_count), "position_iterations": int(robot.solver_position_iteration_count)}


def set_base_drive_damping(robot, kd: Optional[float]) -> Optional[float]:
    """Damping of the x / y / rz velocity drives (None = leave the controller's default, 1e7). Applied after ``env.reset``:
    a stop / play re-applies the controller defaults (``update_controller_mode``)."""
    if kd is None:
        return None
    from omnigibson.controllers import ControlType

    for comp in ("x", "y", "rz"):
        robot.joints[f"base_footprint_{comp}_joint"].set_control_type(ControlType.VELOCITY, kd=float(kd))
    return float(kd)


def wheel_min_z(robot) -> Optional[float]:
    names = list(getattr(robot, "floor_touching_base_link_names", []) or [])
    zs = []
    for n in names:
        link = robot.links.get(n)
        if link is None:
            continue
        try:
            zs.append(float(link.aabb[0][2]))
        except Exception:  # noqa: BLE001
            continue
    return min(zs) if zs else None


def measure(env, robot, warmup: int, steps: int, posture: Dict[str, Any]) -> Dict[str, Any]:
    from omnigibson.object_states import PoseSettled

    set_posture(robot, posture)
    act = hold_action(robot, fingers_closed=bool(posture.get("fingers_closed", True)))
    for _ in range(int(warmup)):
        env.step(act)
    st = robot.states.get(PoseSettled) if hasattr(robot, "states") else None
    if st is not None:
        st.activate()
    joints = list(robot.joints.keys())
    q0 = robot.get_joint_positions().clone()
    p0, _ = robot.get_position_orientation()
    lin, ang, jv_max, jv_all, settled, qs, ps = [], [], [], [], [], [], []
    base_idx = robot.base_idx.tolist()
    body_idx = [i for i in range(len(joints)) if i not in base_idx]
    for _ in range(int(steps)):
        env.step(act)
        v = robot.get_linear_velocity()
        w = robot.get_angular_velocity()
        jv = robot.get_joint_velocities()
        lin.append(float(th.linalg.norm(v)))
        ang.append(float(th.linalg.norm(w)))
        jv_all.append(jv.clone())
        jv_max.append(float(jv[body_idx].abs().max()))
        settled.append(bool(st.is_settled) if st is not None else None)
        qs.append(robot.get_joint_positions().clone())
        ps.append(robot.get_position_orientation()[0].clone())
    lin_t, ang_t, jvm_t = th.tensor(lin), th.tensor(ang), th.tensor(jv_max)
    JV = th.stack(jv_all)
    Q = th.stack(qs)
    P = th.stack(ps)
    ok = (lin_t <= LIN_TOL) & (ang_t <= ANG_TOL) & (jvm_t <= JOINT_TOL)
    per_joint = {joints[i]: {"median_abs": round(float(JV[:, i].abs().median()), 5), "p95_abs": round(float(JV[:, i].abs().quantile(0.95)), 5),
                             "mean": round(float(JV[:, i].mean()), 5), "sign_flip_frac": round(float(((th.sign(JV[1:, i]) * th.sign(JV[:-1, i])) < 0).float().mean()), 2)}
                 for i in range(len(joints)) if float(JV[:, i].abs().quantile(0.95)) > 0.005}

    def longest(mask) -> int:
        best = cur = 0
        for m in mask.tolist():
            cur = cur + 1 if m else 0
            best = max(best, cur)
        return best

    return {
        "steps": int(steps), "warmup": int(warmup),
        "base_lin_vel_mm_s": {"median": round(float(lin_t.median()) * 1000, 3), "p95": round(float(lin_t.quantile(0.95)) * 1000, 3), "max": round(float(lin_t.max()) * 1000, 3),
                              "frac_within_1mm_s": round(float((lin_t <= LIN_TOL).float().mean()), 3)},
        "base_ang_vel_mrad_s": {"median": round(float(ang_t.median()) * 1000, 3), "p95": round(float(ang_t.quantile(0.95)) * 1000, 3),
                                "frac_within_5mrad_s": round(float((ang_t <= ANG_TOL).float().mean()), 3)},
        "body_joint_vel_rad_s": {"median_max": round(float(jvm_t.median()), 4), "p95_max": round(float(jvm_t.quantile(0.95)), 4),
                                 "frac_all_within_0.02": round(float((jvm_t <= JOINT_TOL).float().mean()), 3)},
        "all_gates_frac": round(float(ok.float().mean()), 3), "all_gates_longest_run": longest(ok),
        "pose_settled_state_frac": None if st is None else round(sum(1 for s in settled if s) / max(len(settled), 1), 3),
        "drift": {"base_pos_um": round(float(th.linalg.norm(P[-1] - p0)) * 1e6, 2), "base_pos_max_um": round(float(th.linalg.norm(P - p0, dim=1).max()) * 1e6, 2),
                  "joint_max_mrad": round(float((Q - q0).abs().max()) * 1000, 4)},
        "virtual_base_joints": {joints[i]: {"mean": round(float(JV[:, i].mean()), 5), "median_abs": round(float(JV[:, i].abs().median()), 5),
                                            "sign_flip_frac": round(float(((th.sign(JV[1:, i]) * th.sign(JV[:-1, i])) < 0).float().mean()), 2),
                                            "pos": round(float(Q[-1, i]), 6)} for i in base_idx},
        "hot_joints": per_joint,
        "wheel_min_z_m": wheel_min_z(robot),
    }


def run_scene(scene: str, variants: List[Dict[str, Any]], warmup: int, steps: int, posture_name: str) -> Dict[str, Any]:
    import omnigibson as og

    posture = POSTURES[posture_name]

    robot = dict(ROBOT_CFG)
    if scene == "house":
        scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "house_single_floor", "load_room_instances": ["living_room_0"]}
        robot["position"], robot["orientation"] = HOUSE_POSE
    else:
        scene_cfg = {"type": "Scene"}
        robot["position"], robot["orientation"] = EMPTY_POSE
    t0 = time.time()
    env = og.Environment(configs={"env": dict(ENV_CFG), "scene": scene_cfg, "robots": [robot]})
    r = env.scene.robots[0]
    out: Dict[str, Any] = {"scene": scene, "posture": posture_name, "load_s": round(time.time() - t0, 1), "physics_dt": float(og.sim.get_physics_dt()),
                           "solver_stock": {"velocity_iterations": int(r.solver_velocity_iteration_count), "position_iterations": int(r.solver_position_iteration_count)},
                           "chassis_mass_stock_kg": float(r.base_footprint_link.mass), "variants": {}}
    print(f"[probe] {scene}: loaded in {out['load_s']} s; stock solver {out['solver_stock']}, chassis {out['chassis_mass_stock_kg']:.2f} kg, posture {posture_name}", flush=True)
    for v in variants:
        t1 = time.time()
        solver = set_solver(r, v["vel_iters"], v["pos_iters"])
        chassis = set_chassis(r, v["mass"], v["gravity"])
        env.reset()
        kd = set_base_drive_damping(r, v.get("base_kd"))
        rec = measure(env, r, warmup, steps, posture)
        if kd is not None:
            set_base_drive_damping(r, 1e7)  # back to the controller default for the next variant
        rec.update({"variant": dict(v), "solver": solver, "chassis": chassis, "base_drive_kd": kd, "wall_s": round(time.time() - t1, 1)})
        out["variants"][v["name"]] = rec
        print(f"[probe] {scene}/{v['name']}: base lin {rec['base_lin_vel_mm_s']['median']:.2f} mm/s (p95 {rec['base_lin_vel_mm_s']['p95']:.2f}), "
              f"ang {rec['base_ang_vel_mrad_s']['median']:.2f} mrad/s, joints max {rec['body_joint_vel_rad_s']['median_max']:.3f} rad/s, "
              f"gates {rec['all_gates_frac']:.2f} (run {rec['all_gates_longest_run']}), PoseSettled {rec['pose_settled_state_frac']}, "
              f"drift {rec['drift']['base_pos_max_um']:.1f} um / {rec['drift']['joint_max_mrad']:.3f} mrad, wheel z {rec['wheel_min_z_m']}", flush=True)
    # back to the stock setting for the next scene
    set_solver(r, out["solver_stock"]["velocity_iterations"], out["solver_stock"]["position_iterations"])
    set_chassis(r, out["chassis_mass_stock_kg"], True)
    env.close()
    return out


def markdown(results: Dict[str, Any]) -> str:
    lines = ["| scene | variant | mass kg | vel/pos iters | gravity | base kd | base lin mm/s med (p95) | ≤1 mm/s | base ang mrad/s | joints max rad/s med | ≤0.02 | all gates | PoseSettled | drift µm / mrad |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sc in results["scenes"]:
        if sc.get("error"):
            lines.append(f"| {sc['scene']} | (failed: {sc['error']}) | | | | | | | | | | | | |")
        for name, rec in sc["variants"].items():
            v = rec["variant"]
            kd = rec.get("base_drive_kd")
            lines.append(f"| {sc['scene']} | {name} | {v['mass']:.1f} | {rec['solver']['velocity_iterations']}/{rec['solver']['position_iterations']} | {'on' if v['gravity'] else 'off'} "
                         f"| {'1e7' if kd is None else f'{kd:.0e}'} "
                         f"| {rec['base_lin_vel_mm_s']['median']:.2f} ({rec['base_lin_vel_mm_s']['p95']:.2f}) | {rec['base_lin_vel_mm_s']['frac_within_1mm_s']:.2f} "
                         f"| {rec['base_ang_vel_mrad_s']['median']:.2f} | {rec['body_joint_vel_rad_s']['median_max']:.3f} | {rec['body_joint_vel_rad_s']['frac_all_within_0.02']:.2f} "
                         f"| {rec['all_gates_frac']:.2f} | {rec['pose_settled_state_frac']} | {rec['drift']['base_pos_max_um']:.1f} / {rec['drift']['joint_max_mrad']:.3f} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="empty", help="comma list of empty,house")
    ap.add_argument("--variants", default=",".join(v["name"] for v in VARIANTS), help="comma list of variant names")
    ap.add_argument("--warmup", type=int, default=90)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--posture", default="settle", choices=sorted(POSTURES), help="the held posture (the demo's settle-window or initial posture)")
    args = ap.parse_args()
    names = [s.strip() for s in args.variants.split(",") if s.strip()]
    variants = [v for v in VARIANTS if v["name"] in names]
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    os.makedirs(args.out, exist_ok=True)
    results: Dict[str, Any] = {"gates": {"lin_m_s": LIN_TOL, "ang_rad_s": ANG_TOL, "joint_rad_s": JOINT_TOL}, "posture": args.posture, "scenes": []}
    if len(scenes) > 1:
        # one simulator per process: closing an Environment and creating the next one in the same process segfaulted on the
        # pod (exit 139, 2026-09-24) - every scene runs in a child process and the results are merged here
        import subprocess
        import sys

        for scene in scenes:
            sub_out = os.path.join(args.out, scene)
            cmd = [sys.executable, os.path.abspath(__file__), "--out", sub_out, "--scenes", scene, "--variants", args.variants, "--warmup", str(args.warmup),
                   "--steps", str(args.steps), "--posture", args.posture]
            rc = subprocess.run(cmd, check=False).returncode
            path = os.path.join(sub_out, "results.json")
            if os.path.exists(path):
                with open(path) as f:
                    results["scenes"].extend(json.load(f)["scenes"])
            else:
                results["scenes"].append({"scene": scene, "error": f"child exit {rc}", "variants": {}})
            with open(os.path.join(args.out, "results.json"), "w") as f:
                json.dump(results, f, indent=1)
        md = markdown(results)
        with open(os.path.join(args.out, "results.md"), "w") as f:
            f.write(md + "\n")
        print(md, flush=True)
        return 0
    results["scenes"].append(run_scene(scenes[0], variants, args.warmup, args.steps, args.posture))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    md = markdown(results)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(md + "\n")
    print(md, flush=True)
    import omnigibson as og

    og.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
