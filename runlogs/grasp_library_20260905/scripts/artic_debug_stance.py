import json, os, math, numpy as np, torch as th
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.skills import Skills, SkillError, run
from ispatialgym.oracle.trunk import TrunkState
import omnigibson as og, omnigibson.utils.transform_utils as T
A = "/tmp/glib/artic"; CONTACT_Z = json.load(open("/tmp/grasp_exp/r1pro_right_gripper_eef.json"))["contact_center_z"]
env = og.Environment(configs={"scene": {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}, "robots": [M.joylo_robot_config("robot", obs_modalities=["rgb"])]})
robot = env.robots[0]; scene = env.scene
for _ in range(5): og.sim.step()
skills = Skills(env, robot); obj = scene.object_registry("name", "bottom_cabinet_bamfsz_0")
H = json.load(open(f"{A}/handles/bottom_cabinet__bamfsz.json")); j = H["joints"][3]; cand = j["handles"][0]
rp, rq = obj.root_link.get_position_orientation(); rp = rp.numpy().astype(float); Rr = T.quat2mat(rq).numpy().astype(float); sc = obj.scale.numpy().astype(float)
contact_w = rp + Rr @ (np.array(cand["contact"]) * sc); approach_w = Rr @ np.array(cand["approach_dir"]); pinch_w = Rr @ np.array(cand["pinch_axis"])
print("contact", contact_w.round(3), "approach", approach_w.round(3), "pinch", pinch_w.round(3), "cabinet aabb", obj.aabb[0].numpy().round(2), obj.aabb[1].numpy().round(2))
def eef_quat_from(approach, pinch, sign):
    z = approach / np.linalg.norm(approach); y = pinch * sign; y = y - np.dot(y, z) * z; y /= np.linalg.norm(y); x = np.cross(y, z)
    return T.mat2quat(th.as_tensor(np.stack([x, y, z], axis=1), dtype=th.float32))
quat = eef_quat_from(approach_w, pinch_w, 1.0); z_eef = T.quat2mat(quat)[:, 2].numpy(); origin = contact_w - z_eef * CONTACT_Z
grasp = (th.as_tensor(origin, dtype=th.float32), quat); pre = (th.as_tensor(origin - z_eef * 0.10, dtype=th.float32), quat)
n_xy = -approach_w; n_xy[2] = 0; n_xy /= np.linalg.norm(n_xy); theta = math.atan2(-n_xy[1], -n_xy[0])
start = contact_w[:2] + n_xy[:2] * 1.1
robot.set_position_orientation(th.tensor([start[0], start[1], 0.0]), th.tensor([0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2)]))
q = robot.get_joint_positions().clone()
for arm in ("left", "right"): q[robot.arm_control_idx[arm]] = th.tensor(M.ARM_READY_Q[arm]); q[robot.gripper_control_idx[arm]] = 0.05
robot.set_joint_positions(q); robot.keep_still(); og.sim.step()
skills.backend.update_world(ignore_objects=[]); qf = skills.backend.q_full(); emb = skills._emb_for("right")
print("robot start", robot.get_position_orientation()[0].numpy().round(2), "theta", round(theta, 2))
rows = []
for translate in (0.0, 0.25, 0.5, 0.75, 1.0):
    for d in (0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9):
        for lateral in (-0.2, 0.0, -0.35):
            bx = contact_w[0] - d * math.cos(theta) - lateral * (-math.sin(theta)); by = contact_w[1] - d * math.sin(theta) - lateral * math.cos(theta)
            qk = skills.q_with_base_pose(qf, bx, by, theta); qk[robot.trunk_control_idx] = TrunkState(translate, 0.0).q(skills._joint_limits)
            coll = bool(skills.backend.check_collision(qk.unsqueeze(0), emb_name="whole_body", margin=0.01)[0])
            rows.append((translate, d, lateral, coll, qk))
free = [r for r in rows if not r[3]]
print(len(rows), "stances,", len(free), "collision-free; by translate:", {t: sum(1 for r in free if r[0] == t) for t in (0.0, 0.25, 0.5, 0.75, 1.0)}, "by distance:", {d: sum(1 for r in free if r[1] == d) for d in (0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9)})
skills.backend.update_world(ignore_objects=[obj])
ok = 0
for translate, d, lateral, coll, qk in free[:60]:
    r1 = skills.backend.solve_ik(emb, {M.EEF_LINK["right"]: grasp}, q_seed=qk, sync_world=False) is not None
    r2 = skills.backend.solve_ik(emb, {M.EEF_LINK["right"]: pre}, q_seed=qk, sync_world=False) is not None
    if r1 or r2: print(f"  translate {translate} d {d} lateral {lateral}: grasp IK {r1} pre IK {r2}")
    ok += r1 and r2
print("stances with grasp+pre IK:", ok)
# IK without collision world for the grasp pose alone from a good base pose
skills.backend.update_world(ignore_objects=list(scene.objects))
for translate, d, lateral, coll, qk in free[:8]:
    r1 = skills.backend.solve_ik(emb, {M.EEF_LINK["right"]: grasp}, q_seed=qk, sync_world=False) is not None
    print(f"  no-obstacle IK translate {translate} d {d} lateral {lateral}: {r1}")
og.shutdown()
