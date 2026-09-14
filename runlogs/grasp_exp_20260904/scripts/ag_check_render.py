"""Assisted-grasp (AG) check + rendering of grasp sets in OmniGibson.

For every object and every grasp set (a1 = sampler+filter, a2raw = GraspGen top-5, a2tab = GraspGen top-5 with
approach/table constraints): teleport the object into the (open) right gripper at the stored relative pose, close the
gripper, check that OmniGibson's assisted grasp attaches the object, re-enable gravity and step, then render a
snapshot. Produces per-object PNG strips (rows = grasp sets, columns = 5 grasps)."""
import json, math, os, time
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import numpy as np
import torch as th
import yaml
from PIL import Image, ImageDraw, ImageFont

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.skills import Skills, SkillError
from ispatialgym.oracle.poses import top_down_quat
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject

OUT = "/tmp/grasp_exp"
SETS = [("a1", "A1 sampler+filter"), ("a2raw", "A2 GraspGen raw top-5"), ("a2tab", "A2 GraspGen +table top-5")]
TAG = os.environ.get("AG_TAG", "")
if os.environ.get("AG_SETS"):
    SETS = [tuple(x.split(":")) for x in os.environ["AG_SETS"].split(",")]
CELL = 256
objects = json.load(open(f"{OUT}/objects_exported.json"))
os.makedirs(f"{OUT}/strips{TAG}", exist_ok=True); os.makedirs(f"{OUT}/frames{TAG}", exist_ok=True)


def run(env, gen):
    for a in gen:
        env.step(a)


def load_set(path):
    if not os.path.exists(path):
        return []
    d = yaml.safe_load(open(path))
    out = []
    for k, g in (d.get("grasps") or {}).items():
        q = g["orientation"]; quat_xyzw = th.tensor([*q["xyz"], q["w"]], dtype=th.float32)
        out.append({"name": k, "pos": th.tensor(g["position"], dtype=th.float32), "quat": quat_xyzw, "conf": g.get("confidence"), "meta": g.get("ispatialgym", {})})
    return out


def look_at(cam_pos, target, up=th.tensor([0.0, 0.0, 1.0])):
    f = target - cam_pos; f = f / th.linalg.norm(f)
    r = th.linalg.cross(f, up); r = r / th.linalg.norm(r)
    u = th.linalg.cross(r, f)
    R = th.stack([r, u, -f], dim=1)  # camera looks along -z, +y up (USD convention)
    return T.mat2quat(R)


def snapshot(cam, eef_pos, path):
    target = eef_pos + th.tensor([0.0, 0.0, -0.03])
    cam_pos = eef_pos + th.tensor([0.30, -0.34, -0.10])
    cam.set_position_orientation(cam_pos, look_at(cam_pos, target))
    for _ in range(3):
        og.sim.render()
    rgb = cam.get_obs()[0]["rgb"][..., :3].cpu().numpy()
    h, w = rgb.shape[:2]; s = min(h, w); x0 = (w - s) // 2; y0 = (h - s) // 2
    rgb = rgb[y0:y0 + s, x0:x0 + s]
    Image.fromarray(rgb).save(path)
    return rgb


env = og.Environment(configs={"scene": {"type": "Scene"}, "robots": [M.joylo_robot_config("robot")]})
robot = env.robots[0]
sk = Skills(env, prebuild=("right_arm",))
cam = og.sim.viewer_camera
# NOTE: resizing the viewer camera invalidates the robot articulation view in this OG build -> keep 1280x720, crop centre

# arm test pose: gripper pointing straight down, in front of the robot, clear of the body
run(env, sk.ready_arms(arms=("right",)))
p, q = sk.eef_pose("right")
q_down = top_down_quat(q)
eef_frame = M.EEF_LINK["right"]
reached = False
for pos in ([0.45, -0.25, 0.95], [0.42, -0.28, 0.90], [0.48, -0.20, 0.88], [0.40, -0.30, 0.85], p.tolist()):
    target = (th.tensor(pos), q_down)
    if sk.backend.solve_ik("right_arm", {eef_frame: target}) is None:
        print("test pose IK infeasible:", pos); continue
    try:
        run(env, sk.move_eef("right", target)); reached = True; print("test pose reached:", pos); break
    except SkillError as e:
        print("move_eef to test pose failed:", pos, e)
if not reached:
    print("WARNING: using ready pose for the AG test")
run(env, sk.open_gripper("right"))
run(env, sk.hold(10))
eef_pos, eef_quat = sk.eef_pose("right")
T_eef = T.pose2mat((eef_pos, eef_quat))
print("test eef pose:", eef_pos.tolist(), "approach axis:", T_eef[:3, 2].tolist())

results = {}
t_start = time.time()
for oi, o in enumerate(objects):
    name = o["name"]
    obj = DatasetObject(name=name, category=o["category"], model=o["model"], fixed_base=False, include_default_states=False)
    env.scene.add_object(obj)
    park = th.tensor([3.0 + oi, 3.0, 0.5])
    obj.set_position_orientation(park, th.tensor([0.0, 0.0, 0.0, 1.0]))
    obj.disable_gravity(); obj.keep_still()
    run(env, sk.hold(3))
    results[name] = {}
    for set_key, set_label in SETS:
        grasps = load_set(f"{OUT}/{set_key}_{name}.yaml")
        results[name][set_key] = []
        for gi, g in enumerate(grasps):
            t0 = time.time()
            T_g = T.pose2mat((g["pos"], g["quat"]))           # eef in object frame
            T_obj = T_eef @ th.linalg.inv(T_g)                # object in world so that the eef sits at the grasp
            opos, oquat = T.mat2pose(T_obj)
            # open gripper, place the object (gravity off, held still), let contacts settle
            sk.state.set_gripper("right", close=False)
            for _ in range(8):
                obj.set_position_orientation(opos, oquat); obj.keep_still()
                env.step(sk.state.action())
            p0 = obj.get_position_orientation()[0].clone()
            for _ in range(4):
                obj.keep_still(); env.step(sk.state.action())
            push = float(th.linalg.norm(obj.get_position_orientation()[0] - p0))  # displacement caused by finger/palm interpenetration
            # close and wait for AG (GRASP_WINDOW = 0.3 s of physics time)
            sk.state.set_gripper("right", close=True)
            attached_step = None
            for i in range(60):
                if sk.held_object("right") is None:
                    obj.keep_still()
                env.step(sk.state.action())
                if sk.held_object("right") is obj:
                    attached_step = i; break
            attached = sk.held_object("right") is obj
            # gravity on, hold, verify the object stays attached
            obj.enable_gravity()
            for _ in range(30):
                env.step(sk.state.action())
            held_after = sk.held_object("right") is obj
            fingers = robot.get_joint_positions()[robot.gripper_control_idx["right"]].tolist()
            frame = f"{OUT}/frames{TAG}/{name}__{set_key}__{gi}.png"
            snapshot(cam, eef_pos, frame)
            rel = T.relative_pose_transform(*obj.get_position_orientation(), *sk.eef_pose("right"))
            drift = float(th.linalg.norm(rel[0] - T.mat2pose(th.linalg.inv(T_g))[0]))
            # release and park
            sk.state.set_gripper("right", close=False)
            obj.disable_gravity()
            for _ in range(15):
                env.step(sk.state.action())
            obj.set_position_orientation(park, th.tensor([0.0, 0.0, 0.0, 1.0])); obj.keep_still()
            for _ in range(3):
                env.step(sk.state.action())
            r = {"grasp": g["name"], "ag_attached": bool(attached), "attached_step": attached_step, "held_after_gravity": bool(held_after), "placement_push_m": round(push, 4),
                 "drift_m": round(drift, 4), "finger_q": [round(v, 4) for v in fingers], "tilt_deg": g["meta"].get("tilt_deg"), "conf": g["conf"], "frame": frame, "t_s": round(time.time() - t0, 2)}
            results[name][set_key].append(r)
            print(f"{name:26s} {set_key:6s} #{gi} AG={'Y' if attached else 'n'} held={'Y' if held_after else 'n'} push={push*1000:5.1f}mm drift={drift*1000:5.1f}mm fingers={[round(v,3) for v in fingers]} ({r['t_s']}s)")
    json.dump(results, open(f"{OUT}/ag_results{TAG}.json", "w"), indent=1)
    # remove object from the scene to keep the world small
    env.scene.remove_object(obj)
    run(env, sk.hold(2))

print(f"AG checks done in {time.time()-t_start:.0f} s")
if TAG:
    og.shutdown(); raise SystemExit(0)

# ------------------------------------------------------------------ strips
font = ImageFont.load_default()
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except Exception:
    font_b = font
LABEL_W = 170
a1 = json.load(open(f"{OUT}/a1_results.json")); a2 = json.load(open(f"{OUT}/a2_franka_panda_results.json")); a2b = json.load(open(f"{OUT}/a2_franka_panda_results_big.json"))
for o in objects:
    name = o["name"]
    strip = Image.new("RGB", (LABEL_W + 5 * CELL, 3 * CELL + 28), (255, 255, 255))
    d = ImageDraw.Draw(strip)
    d.text((6, 6), f"{name}   bbox {np.round(np.array(o['coll_bbox_max'])-np.array(o['coll_bbox_min']),3).tolist()} m", fill=(0, 0, 0), font=font_b)
    for ri, (set_key, set_label) in enumerate(SETS):
        y0 = 28 + ri * CELL
        tm = {"a1": f"{a1[name]['t_total_s']*1000:.0f} ms", "a2raw": f"{a2[name]['t_inference_s']*1000:.0f} ms", "a2tab": f"{a2b[name]['t_inference_s']*1000:.0f} ms"}[set_key]
        d.text((6, y0 + 8), set_label, fill=(0, 0, 0), font=font)
        d.text((6, y0 + 30), f"gen time {tm}", fill=(60, 60, 60), font=font)
        rs = results[name][set_key]
        d.text((6, y0 + 52), f"AG ok {sum(r['ag_attached'] for r in rs)}/{len(rs)}", fill=(0, 0, 0), font=font)
        for ci in range(5):
            x0 = LABEL_W + ci * CELL
            if ci < len(rs):
                im = Image.open(rs[ci]["frame"]).resize((CELL, CELL)); strip.paste(im, (x0, y0))
                ok = rs[ci]["ag_attached"] and rs[ci]["held_after_gravity"]
                col = (0, 160, 0) if ok else (200, 0, 0)
                d.rectangle([x0, y0, x0 + CELL - 1, y0 + CELL - 1], outline=col, width=4)
                tilt = rs[ci]["tilt_deg"]; tilt_s = f" tilt {tilt:.0f}" if tilt is not None else ""
                d.text((x0 + 6, y0 + 4), f"AG {'OK' if ok else 'FAIL'}{tilt_s}", fill=col, font=font_b)
                if rs[ci]["placement_push_m"] > 0.005:
                    d.text((x0 + 6, y0 + CELL - 20), f"push {rs[ci]['placement_push_m']*1000:.0f} mm", fill=(200, 100, 0), font=font)
            else:
                d.rectangle([x0, y0, x0 + CELL - 1, y0 + CELL - 1], fill=(235, 235, 235)); d.text((x0 + 60, y0 + CELL // 2), "no candidate", fill=(120, 120, 120), font=font)
    strip.save(f"{OUT}/strips/{name}.png")
print("STRIPS DONE")
og.shutdown()
