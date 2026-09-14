"""Assisted-grasp validation + multi-view rendering for the grasp library.

usage: python ag_validate_render.py <shard> <nshards> [max_models]
For every model of the shard: load the object, iterate the ranked candidates, teleport the object into the open gripper
at the candidate pose (gravity off), close, check that OmniGibson's assisted grasp attaches and holds under gravity, and
for valid grasps render 3 external views + the right wrist camera. Stops after 5 valid grasps per model."""
import json, os, sys, time, traceback
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import numpy as np, torch as th, yaml
from PIL import Image
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.action import ActionState
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject

OUT = "/tmp/glib"; NEED = 5; MAX_TESTS = 12
shard, nshard = int(sys.argv[1]), int(sys.argv[2]); max_models = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
jobs = [j for i, j in enumerate(json.load(open(os.environ.get("JOBS", f"{OUT}/models.json")))) if i % nshard == shard][:max_models]
pose = json.load(open(f"{OUT}/test_pose.json"))
CONTACT_Z = json.load(open("/tmp/grasp_exp/r1pro_right_gripper_eef.json"))["contact_center_z"]
WRIST = "robot:right_realsense_link:Camera:0"

env = og.Environment(configs={"scene": {"type": "Scene"}, "robots": [M.joylo_robot_config("robot", obs_modalities=["rgb"], sensor_config={"VisionSensor": {"sensor_kwargs": {"image_height": 480, "image_width": 480}}})]})
robot = env.robots[0]; scene = env.scene
robot.set_joint_positions(th.tensor(pose["q"])); og.sim.step()
state = ActionState(robot); state.hold_current(); state.set_gripper("right", close=False)
for _ in range(15):
    env.step(state.action())
eef_pos, eef_quat = robot.eef_links["right"].get_position_orientation(); T_eef = T.pose2mat((eef_pos, eef_quat)); R_eef = T_eef[:3, :3]
print("eef", eef_pos.tolist(), "approach", R_eef[:, 2].tolist(), flush=True)
cam = og.sim.viewer_camera
# warm up the viewer camera (the first frame after a camera move is rendered from the old pose otherwise)
cam.set_position_orientation(eef_pos + th.tensor([0.4, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
for _ in range(6): og.sim.render()


def look_at(cam_pos, target, up=th.tensor([0.0, 0.0, 1.0])):
    f = target - cam_pos; f = f / th.linalg.norm(f); r = th.linalg.cross(f, up)
    if th.linalg.norm(r) < 1e-3: r = th.linalg.cross(f, th.tensor([1.0, 0.0, 0.0]))
    r = r / th.linalg.norm(r); u = th.linalg.cross(r, f); return T.mat2quat(th.stack([r, u, -f], dim=1))


def grab(cam_pos, target, path, size=360):
    cam.set_position_orientation(cam_pos, look_at(cam_pos, target))
    for _ in range(3): og.sim.render()
    rgb = cam.get_obs()[0]["rgb"][..., :3].cpu().numpy(); h, w = rgb.shape[:2]; s = min(h, w)
    Image.fromarray(rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]).resize((size, size), Image.LANCZOS).save(path)


def wrist(path, size=360):
    for _ in range(2): og.sim.render()
    obs, _ = robot.get_obs(); rgb = obs[WRIST]["rgb"][..., :3].cpu().numpy()
    Image.fromarray(rgb).resize((size, size), Image.LANCZOS).save(path)


def load_cands(key):
    p = f"{OUT}/cands/{key}.yaml"
    if not os.path.exists(p): return None, None
    d = yaml.safe_load(open(p)); out = []
    for k, g in (d.get("grasps") or {}).items():
        q = g["orientation"]; out.append({"name": k, "pos": th.tensor(g["position"], dtype=th.float32), "quat": th.tensor([*q["xyz"], q["w"]], dtype=th.float32), "score": g.get("confidence"), "meta": g.get("ispatialgym", {})})
    return out, d.get("ispatialgym", {})


ex, ey, ez = (th.tensor(R_eef[:, i], dtype=th.float32) for i in range(3))
c_w = eef_pos + ez * CONTACT_Z                    # world position of the pad centre
t_start = time.time(); n_done = 0
for j in jobs:
    key = j["key"]; res_path = f"{OUT}/results/{key}.json"
    if os.path.exists(res_path): continue
    cands, meta = load_cands(key)
    result = {**j, "candidates_meta": meta, "tested": [], "n_valid": 0, "error": None}
    if not cands:
        result["error"] = "no candidates"; json.dump(result, open(res_path, "w"), indent=1); print(f"{key}: no candidates", flush=True); continue
    obj = None; t0 = time.time()
    try:
        obj = DatasetObject(name=f"o_{key}", category=j["category"], model=j["model"], fixed_base=False, include_default_states=False, **({"scale": th.tensor(j["scale"], dtype=th.float32)} if j.get("scale") else {}))
        scene.add_object(obj)
        park = th.tensor([3.0, 3.0, 0.6]); obj.set_position_orientation(park, th.tensor([0.0, 0.0, 0.0, 1.0])); obj.disable_gravity(); obj.keep_still()
        for _ in range(3): env.step(state.action())
        os.makedirs(f"{OUT}/frames/{key}", exist_ok=True)
        size = float(max(obj.aabb_extent))
        for gi, g in enumerate(cands[:MAX_TESTS]):
            if result["n_valid"] >= NEED: break
            tg0 = time.time()
            T_g = T.pose2mat((g["pos"], g["quat"])); T_obj = T_eef @ th.linalg.inv(T_g); opos, oquat = T.mat2pose(T_obj)
            state.set_gripper("right", close=False)
            for _ in range(8):
                obj.set_position_orientation(opos, oquat); obj.keep_still(); env.step(state.action())
            p0 = obj.get_position_orientation()[0].clone()
            for _ in range(4): obj.keep_still(); env.step(state.action())
            push = float(th.linalg.norm(obj.get_position_orientation()[0] - p0))
            state.set_gripper("right", close=True); attached_step = None
            for i in range(60):
                if robot._ag_obj_in_hand["right"] is None: obj.keep_still()
                env.step(state.action())
                if robot._ag_obj_in_hand["right"] is obj: attached_step = i; break
            attached = robot._ag_obj_in_hand["right"] is obj
            obj.enable_gravity()
            for _ in range(30): env.step(state.action())
            held = robot._ag_obj_in_hand["right"] is obj
            fingers = robot.get_joint_positions()[robot.gripper_control_idx["right"]].tolist()
            rec = {"grasp": g["name"], "idx": gi, "ag_attached": bool(attached), "attached_step": attached_step, "held_after_gravity": bool(held), "placement_push_m": round(push, 4), "finger_q": [round(v, 4) for v in fingers], "score": g["score"], "meta": g["meta"], "frames": {}}
            if attached and held:
                d = float(np.clip(1.4 * size + 0.12, 0.28, 0.6))          # camera distance scales with the object
                tgt = c_w + ez * (0.3 * size)                               # look a bit past the pads toward the object body
                grab(c_w + ex * d - ez * 0.10, tgt, f"{OUT}/frames/{key}/g{gi}_v1.png")        # profile (pinch axis left-right)
                grab(c_w - ex * d - ez * 0.10, tgt, f"{OUT}/frames/{key}/g{gi}_v2.png")        # opposite profile
                grab(c_w + (ex * 0.55 + ey * 0.8) / 0.97 * d + ez * (0.25 * d), tgt, f"{OUT}/frames/{key}/g{gi}_v3.png")   # oblique from below
                wrist(f"{OUT}/frames/{key}/g{gi}_wrist.png")
                rec["frames"] = {f"v{k}": f"{OUT}/frames/{key}/g{gi}_v{k}.png" for k in (1, 2, 3)}; rec["frames"]["wrist"] = f"{OUT}/frames/{key}/g{gi}_wrist.png"
                result["n_valid"] += 1
            rec["t_s"] = round(time.time() - tg0, 2); result["tested"].append(rec)
            # release and park
            state.set_gripper("right", close=False); obj.disable_gravity()
            for _ in range(15): env.step(state.action())
            obj.set_position_orientation(park, th.tensor([0.0, 0.0, 0.0, 1.0])); obj.keep_still()
            for _ in range(3): env.step(state.action())
    except Exception as e:
        result["error"] = repr(e); traceback.print_exc()
    finally:
        if obj is not None:
            try: scene.remove_object(obj)
            except Exception as e: print("remove failed", key, repr(e))
        for _ in range(2): env.step(state.action())
    result["t_model_s"] = round(time.time() - t0, 1); json.dump(result, open(res_path, "w"), indent=1); n_done += 1
    print(f"[{n_done}/{len(jobs)}] {key}: valid {result['n_valid']}/{len(result['tested'])} tested, {result['t_model_s']}s ({np.mean([r['t_s'] for r in result['tested']]) if result['tested'] else 0:.1f} s/grasp)", flush=True)
print(f"SHARD DONE {shard}/{nshard}: {n_done} models in {time.time()-t_start:.0f}s", flush=True)
og.shutdown()
