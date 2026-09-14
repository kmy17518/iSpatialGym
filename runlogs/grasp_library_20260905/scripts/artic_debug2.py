import json, os, math, numpy as np, torch as th
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.skills import Skills
from ispatialgym.oracle.trunk import TrunkState
import omnigibson as og, omnigibson.utils.transform_utils as T
env = og.Environment(configs={"scene": {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}, "robots": [M.joylo_robot_config("robot", obs_modalities=["rgb"])]})
robot = env.robots[0]; scene = env.scene
for _ in range(5): og.sim.step()
skills = Skills(env, robot)
print("robot spawn", robot.get_position_orientation()[0].numpy().round(2), flush=True)
q = robot.get_joint_positions().clone()
for arm in ("left", "right"): q[robot.arm_control_idx[arm]] = th.tensor(M.ARM_READY_Q[arm]); q[robot.gripper_control_idx[arm]] = 0.05
robot.set_joint_positions(q); robot.keep_still(); og.sim.step()
skills.backend.update_world(); qf = skills.backend.q_full()
def coll_at(x, y, yaw, translate=0.5):
    qk = skills.q_with_base_pose(qf, x, y, yaw); qk[robot.trunk_control_idx] = TrunkState(translate, 0.0).q(skills._joint_limits)
    return bool(skills.backend.check_collision(qk.unsqueeze(0), emb_name="whole_body", margin=0.01)[0])
sp = robot.get_position_orientation()[0].numpy()
print("collision at spawn pose:", coll_at(float(sp[0]), float(sp[1]), 0.0), flush=True)
for (x, y) in [(-1.63, 1.68), (-1.63, 2.0), (-1.63, 2.3), (0.0, 0.0), (-0.5, 1.0)]:
    print(f"collision at ({x},{y}) yaw 1.57:", coll_at(x, y, 1.57), "| yaw 0:", coll_at(x, y, 0.0), flush=True)
# objects near (-1.63, 2.0)
for o in scene.objects:
    lo, hi = o.aabb; lo = lo.numpy(); hi = hi.numpy()
    if lo[0] < -1.0 and hi[0] > -2.3 and lo[1] < 2.6 and hi[1] > 1.4 and lo[2] < 1.5: print("  near:", o.category, o.name, lo.round(2), hi.round(2), flush=True)
# collision with everything ignored except floors
skills.backend.update_world(ignore_objects=[o for o in scene.objects if o.category not in ("floors",)])
print("collision at (-1.63,2.0) with only floors in world:", coll_at(-1.63, 2.0, 1.57), flush=True)
skills.backend.update_world(ignore_objects=[o for o in scene.objects])
print("collision at (-1.63,2.0) with empty world (ground slab only):", coll_at(-1.63, 2.0, 1.57), flush=True)
og.shutdown()
