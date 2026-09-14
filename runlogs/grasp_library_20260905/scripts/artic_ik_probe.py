import os, math, json, numpy as np, torch as th
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.skills import Skills
from ispatialgym.oracle.trunk import TrunkState
import omnigibson as og, omnigibson.utils.transform_utils as T
env = og.Environment(configs={"scene": {"type": "Scene"}, "robots": [M.joylo_robot_config("robot", obs_modalities=["rgb"])]})
robot = env.robots[0]
for _ in range(5): og.sim.step()
skills = Skills(env, robot)
q = robot.get_joint_positions().clone()
for arm in ("left", "right"): q[robot.arm_control_idx[arm]] = th.tensor(M.ARM_READY_Q[arm]); q[robot.gripper_control_idx[arm]] = 0.05
robot.set_joint_positions(q); robot.keep_still(); og.sim.step()
skills.backend.update_world(); qf = skills.backend.q_full(); emb = skills._emb_for("right")
def quat_from(approach, pinch):
    z = np.array(approach, float); z /= np.linalg.norm(z); y = np.array(pinch, float); y = y - np.dot(y, z) * z; y /= np.linalg.norm(y); x = np.cross(y, z)
    return T.mat2quat(th.as_tensor(np.stack([x, y, z], axis=1), dtype=th.float32))
ORIENTS = {"fwd_pinchZ_only": ([1, 0, 0], [0, 0, 1])} if True else {"topdown_pinchX": ([0, 0, -1], [1, 0, 0]), "topdown_pinchY": ([0, 0, -1], [0, 1, 0]), "fwd_pinchZ": ([1, 0, 0], [0, 0, 1]), "fwd_pinchNegZ": ([1, 0, 0], [0, 0, -1]), "fwd_pinchY": ([1, 0, 0], [0, 1, 0]),
           "fwd_down20_pinchZ": ([math.cos(math.radians(20)), 0, -math.sin(math.radians(20))], [0, 0, 1]), "fwd_down45_pinchZ": ([0.707, 0, -0.707], [0, 0, 1]), "back_pinchZ": ([-1, 0, 0], [0, 0, 1])}
print("robot at origin facing +x; targets at (x ahead, y=-0.2 (right arm side), z); torso translate levels tested: 0.0 0.5 1.0 1.5 2.0 (2.0 = ground crouch)", flush=True)
for name, (ap, pi) in ORIENTS.items():
    quat = quat_from(ap, pi); row = []
    for z in (0.35, 0.5, 0.6, 0.7, 0.9, 1.1, 1.3):
        for x in (0.5,):
            ok = False
            oks = []
            for tr in (0.0, 0.5, 1.0, 1.5):
                for tilt in (-0.5, 0.0, 0.5, 1.0):
                    qk = qf.clone(); qk[robot.trunk_control_idx] = TrunkState(tr, tilt).q(skills._joint_limits)
                    if skills.backend.solve_ik(emb, {M.EEF_LINK["right"]: (th.tensor([x, -0.2, z]), quat)}, q_seed=qk, sync_world=False) is not None: oks.append(f"t{tr}/{tilt:+.1f}")
            row.append(f"z{z}/x{x}:{('+'.join(oks)) if oks else '.'}")
    print(f"{name:20s} " + " ".join(row), flush=True)
og.shutdown()
