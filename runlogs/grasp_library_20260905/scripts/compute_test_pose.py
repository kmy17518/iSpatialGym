"""One-off: find a right-arm configuration with the gripper pointing straight down (clear of the body) and save it."""
import json, os
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import torch as th
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.skills import Skills, SkillError
from ispatialgym.oracle.poses import top_down_quat
import omnigibson as og

env = og.Environment(configs={"scene": {"type": "Scene"}, "robots": [M.joylo_robot_config("robot", obs_modalities=["rgb"], sensor_config={"VisionSensor": {"sensor_kwargs": {"image_height": 480, "image_width": 480}}})]})
robot = env.robots[0]; sk = Skills(env, prebuild=("right_arm",))
def run(gen):
    for a in gen: env.step(a)
run(sk.ready_arms(arms=("right",)))
p, q = sk.eef_pose("right"); q_down = top_down_quat(q); eef = M.EEF_LINK["right"]
for pos in ([0.45, -0.25, 0.95], [0.42, -0.28, 0.90], [0.48, -0.20, 0.88], [0.40, -0.30, 0.85], p.tolist()):
    target = (th.tensor(pos), q_down)
    if sk.backend.solve_ik("right_arm", {eef: target}) is None: continue
    try:
        run(sk.move_eef("right", target)); break
    except SkillError as e:
        print("failed", pos, e)
run(sk.open_gripper("right")); run(sk.hold(10))
qpos = robot.get_joint_positions().tolist(); ep, eq = sk.eef_pose("right")
obs, _ = robot.get_obs()
print("sensor keys:", list(obs.keys()))
json.dump({"q": qpos, "eef_pos": ep.tolist(), "eef_quat": eq.tolist(), "sensor_keys": list(obs.keys())}, open("/tmp/glib/test_pose.json", "w"), indent=1)
print("TEST POSE SAVED", ep.tolist())
og.shutdown()
