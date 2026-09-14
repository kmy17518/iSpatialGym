"""Robot-in-the-loop opening test for NON-SCENE articulated objects (and anything that opens like a lid): the object stands on
the floor or on a table in front of R1Pro; for every openable joint with a reviewed/heuristic handle the robot takes a stance
(base pose + trunk height so the gripper is at handle height), grasps the handle (assisted grasping), then pulls along the
opening path (straight for drawers, an arc about the hinge for doors/lids). Records reachability, AG attach and the achieved
joint motion, and renders before/after frames.

usage: python artic_validate_table.py <shard> <nshards>"""
import json, os, sys, time, math, traceback, glob
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import numpy as np, torch as th
from PIL import Image, ImageDraw, ImageFont
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.skills import Skills, SkillError, run
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson import object_states

A = "/tmp/glib/artic"; os.makedirs(f"{A}/results", exist_ok=True); os.makedirs(f"{A}/frames", exist_ok=True); os.makedirs(f"{A}/vstrips", exist_ok=True)
CONTACT_Z = json.load(open("/tmp/grasp_exp/r1pro_right_gripper_eef.json"))["contact_center_z"]
MANUAL = json.load(open(f"{A}/manual_handles.json")) if os.path.exists(f"{A}/manual_handles.json") else {}
NON_SCENE = {"blender", "carton", "cedar_chest", "cooler", "hinged_jar", "laptop", "recycling_bin", "toolbox", "violin_case", "mailbox", "backpack"}
TABLE_TOP = 0.75; OBJ_X = 1.25
keys = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{A}/handles/*.json") if os.path.basename(p).split("__")[0] in NON_SCENE)
shard, nshard = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
keys = [k for i, k in enumerate(keys) if i % nshard == shard]
if os.environ.get("ONLY"): keys = [k for k in keys if k.startswith(os.environ["ONLY"])]
REPEAT = int(os.environ.get("REPEAT", "1")); keys = [k for k in keys for _ in range(REPEAT)]

env = og.Environment(configs={"scene": {"type": "Scene"}, "robots": [M.joylo_robot_config("robot", obs_modalities=["rgb"])]})
robot = env.robots[0]; scene = env.scene
for _ in range(5): og.sim.step()
skills = Skills(env, robot)
cam = og.sim.viewer_camera; cam.set_position_orientation(th.tensor([2.5, -2.0, 1.6]), th.tensor([0.0, 0.0, 0.0, 1.0]))
for _ in range(6): og.sim.render()
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except Exception: FONT = ImageFont.load_default()


def look_at(cam_pos, target, up=th.tensor([0.0, 0.0, 1.0])):
    f = target - cam_pos; f = f / th.linalg.norm(f); r = th.linalg.cross(f, up)
    if th.linalg.norm(r) < 1e-3: r = th.linalg.cross(f, th.tensor([1.0, 0.0, 0.0]))
    r = r / th.linalg.norm(r); u = th.linalg.cross(r, f); return T.mat2quat(th.stack([r, u, -f], dim=1))


def grab(cam_pos, target, path, size=420):
    cam.set_position_orientation(th.as_tensor(cam_pos, dtype=th.float32), look_at(th.as_tensor(cam_pos, dtype=th.float32), th.as_tensor(target, dtype=th.float32)))
    for _ in range(3): og.sim.render()
    rgb = cam.get_obs()[0]["rgb"][..., :3].cpu().numpy(); h, w = rgb.shape[:2]; s = min(h, w)
    Image.fromarray(rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]).resize((size, size), Image.LANCZOS).save(path)


def yaw_quat(yaw): return th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def eef_quat_from(approach, pinch, sign):
    z = np.asarray(approach, float); z /= np.linalg.norm(z); y = np.asarray(pinch, float) * sign; y = y - np.dot(y, z) * z; y /= np.linalg.norm(y); x = np.cross(y, z)
    return T.mat2quat(th.as_tensor(np.stack([x, y, z], axis=1), dtype=th.float32))


def orientation_options(cand, kind):
    """Grasp orientation alternatives for a handle candidate: (label, approach_dir, pinch_axis) in the object frame.
    front = as inferred; front_down45 / topdown for protruding bars and knobs (pads straddle the bar from above); a 30 deg
    downward variant for edge pinches."""
    a0 = np.asarray(cand["approach_dir"], float); p0 = np.asarray(cand["pinch_axis"], float); z = np.array([0.0, 0.0, 1.0])
    def unit(v): n = np.linalg.norm(v); return v / n if n > 1e-9 else v
    opts = [("front", a0, p0)]
    ck = cand.get("kind", "")
    if ck in ("bar", "knob"):
        depth = float(cand.get("depth", 0.0)); u = unit(np.cross(-a0, p0))                    # bar long axis
        if depth >= 0.02 and abs(a0[2]) < 0.7:
            a45 = unit(a0 - z); p45 = unit(np.cross(u, a45))
            if np.linalg.norm(p45) > 0.5: opts.append(("front_down45", a45, p45))
        if depth >= 0.025 and abs(u[2]) < 0.7:
            ptd = unit(np.cross(u, -z))
            if np.linalg.norm(ptd) > 0.5: opts.append(("topdown", -z, ptd))
    elif ck.startswith("edge") and abs(a0[2]) < 0.7:
        opts.append(("edge_down30", unit(a0 - 0.58 * z), p0))
    return opts


def selected_grasp(H, joint, review):
    """Return (candidate dict, source): manual override > review > heuristic; None if ungraspable."""
    mj = MANUAL.get(H["key"], {}).get(joint["joint"])
    if mj is not None:
        if mj.get("skip"): return None, f"manual skip: {mj['skip']}"
        if mj.get("use") == "F" and joint.get("fallback"): return joint["fallback"], "manual:F"
        if "contact" in mj: return mj, "manual"
    rj = next((r for r in (review or {}).get("joints", []) if r["joint"] == joint["joint"]), None)
    if rj is not None:
        if not rj.get("graspable", True) or rj.get("chosen") in (None, "none"): return None, f"review: {rj.get('handle_description', 'ungraspable')}"
        ch = rj["chosen"]
        if ch == "F" and joint.get("fallback"): return joint["fallback"], "review:F"
        idx = "ABCD".find(ch)
        if 0 <= idx < len(joint.get("handles", [])): return joint["handles"][idx], f"review:{ch}"
    if joint.get("handles"): return joint["handles"][0], "heuristic:A"
    if joint.get("fallback") and joint["fallback"].get("exposed"): return joint["fallback"], "heuristic:F"
    return None, "heuristic: ungraspable"


def reset_robot():
    robot.set_position_orientation(th.tensor([0.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    q = robot.get_joint_positions().clone()
    for arm in ("left", "right"): q[robot.arm_control_idx[arm]] = th.zeros(len(robot.arm_control_idx[arm])); q[robot.gripper_control_idx[arm]] = 0.05   # arms hanging (JoyLo reset): lets the stance search stand close to counters
    robot.set_joint_positions(q); robot.set_joint_velocities(th.zeros_like(q)); robot.keep_still()
    skills.state.hold_current(); skills.state.set_gripper("right", close=False); skills.state.set_gripper("left", close=False)
    for _ in range(5): env.step(skills.state.action())


table = None
def ensure_table(need):
    global table
    if need and table is None:
        og.sim.stop(); table = PrimitiveObject(name="table", primitive_type="Cube", scale=th.tensor([0.5, 1.2, TABLE_TOP]), fixed_base=True, rgba=[0.6, 0.6, 0.6, 1.0]); scene.add_object(table); og.sim.play()
        table.set_position_orientation(th.tensor([OBJ_X, 0.0, TABLE_TOP / 2]), th.tensor([0.0, 0.0, 0.0, 1.0])); og.sim.step()
    if table is not None:
        table.set_position_orientation(th.tensor([OBJ_X, 0.0, TABLE_TOP / 2]) if need else th.tensor([20.0, 20.0, -5.0]), th.tensor([0.0, 0.0, 0.0, 1.0])); og.sim.step()


for rep, key in enumerate(keys):
    if REPEAT == 1 and os.path.exists(f"{A}/results/{key}.json"): continue
    H = json.load(open(f"{A}/handles/{key}.json")); inv = json.load(open(f"{A}/inv/{key}.json"))
    review = json.load(open(f"{A}/review/{key}.json")) if os.path.exists(f"{A}/review/{key}.json") else None
    result = {"key": key, "category": H["category"], "model": H["model"], "aabb_extent": H["aabb_extent"], "support": None, "joints": [], "error": None}
    obj = None; t0 = time.time()
    try:
        ext = np.array(H["aabb_extent"]); on_floor = ext[2] >= 0.6; result["support"] = "floor" if on_floor else "table"
        ensure_table(not on_floor)
        # orientation: the first graspable joint's outward normal faces the robot (-x); for top lids the free edge faces the robot
        n0 = np.array([1.0, 0.0, 0.0]); jsel = None
        for j in H["joints"]:
            c, src = selected_grasp(H, j, review)
            if c is not None: jsel = (j, c); break
        if jsel is not None:
            j, c = jsel; n0 = np.array(c["approach_dir"]) * -1.0                 # outward = opposite of the approach
            if abs(n0[2]) > 0.7:                                                  # lid: use the horizontal component of the approach (edge grasp) or hinge->edge
                hz = np.array(c["approach_dir"]); hz[2] = 0
                if np.linalg.norm(hz) < 1e-3 and j.get("hinge_point_obj") is not None:
                    hz = np.array(c["contact"]) - np.array(j["hinge_point_obj"]); hz[2] = 0
                n0 = -hz / max(np.linalg.norm(hz), 1e-9) if np.linalg.norm(hz) > 1e-3 else np.array([1.0, 0.0, 0.0])
        yaw = math.pi - math.atan2(n0[1], n0[0])                                  # rotate so that n0 -> -x (towards the robot)
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
        zmin = inv["aabb_min_rel_root"][2]; root = np.array([OBJ_X, 0.0, (0.0 if on_floor else TABLE_TOP) - zmin + 0.002])
        og.sim.stop(); obj = DatasetObject(name=f"o_{key}", category=H["category"], model=H["model"], fixed_base=True); scene.add_object(obj); og.sim.play()
        obj.set_position_orientation(th.as_tensor(root, dtype=th.float32), yaw_quat(yaw))
        names = list(obj.joints.keys()); q = obj.get_joint_positions().clone()
        for j in H["joints"]:
            if j["joint"] in names: q[names.index(j["joint"])] = j["closed_q"]
        obj.set_joint_positions(q); obj.keep_still(); og.sim.step()
        reset_robot(); skills.backend.update_world()
        rp = obj.root_link.get_position_orientation()[0].numpy().astype(np.float64)
        def to_world(p): return rp + Rz @ np.asarray(p, float)
        def dir_world(d): return Rz @ np.asarray(d, float)
        for j in H["joints"]:
            cand, src = selected_grasp(H, j, review)
            rec = {"joint": j["joint"], "kind": j.get("kind"), "grasp_source": src, "stance": None, "ag": False, "dq": 0.0, "dq_target": None, "opened": False, "open_state": None, "steps": [], "error": None}
            result["joints"].append(rec)
            if cand is None: rec["error"] = "no graspable handle"; continue
            cand_list = [cand] + ([cand['alternative']] if isinstance(cand, dict) and cand.get('alternative') else [])
            for ci, cand in enumerate(cand_list):
                contact_w = to_world(cand["contact"]); approach_w = dir_world(cand["approach_dir"]); pinch_w = dir_world(cand["pinch_axis"]); pull_w = dir_world(cand["pull_dir"])
                rec["contact_world"] = contact_w.round(3).tolist(); rec["handle_height"] = round(float(contact_w[2]), 3)
                # try both pinch signs (wrist roll); pick the first with a feasible stance
                ji = names.index(j["joint"]); q0 = float(obj.get_joint_positions()[ji])
                span = j["upper"] - j["lower"]; is_prism = j["type"] == "PrismaticJoint"
                target = min(0.15, 0.6 * span) if is_prism else min(math.radians(45), 0.6 * span); rec["dq_target"] = round(target, 3)
                done_grasp = False
                for olabel, o_app, o_pinch in orientation_options(cand, j.get('kind')):
                  approach_w = dir_world(o_app); pinch_w = dir_world(o_pinch)
                  for sign in (1.0, -1.0):
                      quat = eef_quat_from(approach_w, pinch_w, sign); z_eef = T.quat2mat(quat)[:, 2].numpy()
                      origin = contact_w - z_eef * CONTACT_Z; grasp = (th.as_tensor(origin, dtype=th.float32), quat); pre = (th.as_tensor(origin - z_eef * 0.10, dtype=th.float32), quat)
                      stance = None; last_err = None
                      for lateral in (-0.2, 0.0, -0.35):
                          try:
                              reset_robot()
                              stance = skills.stance_for(th.as_tensor(contact_w, dtype=th.float32), arm="right", eef_candidates=[pre, grasp], distances=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8), translates=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 0.25, 0.0), lateral=lateral, ignore_objects=[obj]); break
                          except SkillError as e: last_err = e
                      try:
                          if stance is None: raise last_err
                      except SkillError as e:
                          rec["steps"].append(f"sign {sign:+.0f}: no stance ({e})")
                          try:                                                                   # classify: collision-bound or IK-bound
                              from ispatialgym.oracle.trunk import TrunkState
                              qf = skills.backend.q_full(); emb = skills._emb_for("right"); theta = math.atan2(float(contact_w[1] - robot.get_position_orientation()[0][1]), float(contact_w[0] - robot.get_position_orientation()[0][0]))
                              free = 0; ik_free_world = 0; ik_no_obst = 0; n = 0
                              skills.backend.update_world(ignore_objects=[obj])
                              for tr in (0.0, 0.5, 1.0):
                                  for d in (0.5, 0.6, 0.7, 0.8):
                                      bx = contact_w[0] - d * math.cos(theta) + 0.2 * (-math.sin(theta)); by = contact_w[1] - d * math.sin(theta) + 0.2 * math.cos(theta)
                                      qk = skills.q_with_base_pose(qf, bx, by, theta); qk[robot.trunk_control_idx] = TrunkState(tr, 0.0).q(skills._joint_limits); n += 1
                                      if not bool(skills.backend.check_collision(qk.unsqueeze(0), emb_name="whole_body", margin=0.01)[0]):
                                          free += 1
                                          if skills.backend.solve_ik(emb, {M.EEF_LINK["right"]: grasp}, q_seed=qk, sync_world=False) is not None: ik_free_world += 1
                              skills.backend.update_world(ignore_objects=list(scene.objects))
                              for tr in (0.0, 0.5, 1.0):
                                  for d in (0.5, 0.6, 0.7, 0.8):
                                      bx = contact_w[0] - d * math.cos(theta) + 0.2 * (-math.sin(theta)); by = contact_w[1] - d * math.sin(theta) + 0.2 * math.cos(theta)
                                      qk = skills.q_with_base_pose(qf, bx, by, theta); qk[robot.trunk_control_idx] = TrunkState(tr, 0.0).q(skills._joint_limits)
                                      if skills.backend.solve_ik(emb, {M.EEF_LINK["right"]: grasp}, q_seed=qk, sync_world=False) is not None: ik_no_obst += 1
                              skills.backend.update_world()
                              rec["diag_stance"] = {"probed": n, "collision_free": free, "grasp_ik_from_free_stance": ik_free_world, "grasp_ik_without_obstacles": ik_no_obst, "handle_height": round(float(contact_w[2]), 3), "approach": [round(float(v), 2) for v in approach_w]}
                          except Exception as e2: rec["diag_stance"] = {"error": repr(e2)}
                          continue
                      rec["stance"] = {"base": [round(float(v), 3) for v in stance[0]], "translate": round(float(stance[1]), 2), "sign": sign, "orientation": olabel}
                      try:
                          try:
                              run(env, skills.go_to_stance(stance))
                          except SkillError as e_st:
                              if e_st.reason not in ("planning", "collision"): raise
                              (bx_, by_, yaw_), tr_ = stance; run(env, skills.navigate_to((bx_, by_, yaw_), ignore_objects=[obj])); run(env, skills.set_torso(translate=tr_, auto_ready_arms=False))
                          rec["steps"].append("stance ok")
                          grab(rp + np.array([-1.6, -1.4, 1.2]), rp + np.array([0, 0, ext[2] / 2]), f"{A}/frames/{key}__{j['joint']}_0.png")
                          run(env, skills.grasp("right", obj=obj, grasp_pose=grasp, pregrasp_dist=0.10, lift_dist=0.0, torso=False))
                          rec["ag"] = True; rec["steps"].append("grasped"); done_grasp = True
                          grab(rp + np.array([-1.6, -1.4, 1.2]), rp + np.array([0, 0, ext[2] / 2]), f"{A}/frames/{key}__{j['joint']}_1.png")
                          # pull along the opening path in small chunks, watching the joint
                          eef_p, eef_q = skills.eef_pose("right"); eef_p = eef_p.numpy().astype(np.float64); Rq = T.quat2mat(eef_q).numpy()
                          n_chunks = 6
                          for k in range(1, n_chunks + 1):
                              frac = k / n_chunks
                              if is_prism:
                                  wp_p = eef_p + pull_w * target * frac; wp_q = eef_q
                              else:
                                  hinge_w = to_world(j["hinge_point_obj"]); ax = dir_world(j["axis_dir_obj"]) * j["open_sign"]; ang = target * frac
                                  Rrot = T.quat2mat(T.axisangle2quat(th.as_tensor(ax * ang, dtype=th.float32))).numpy()
                                  wp_p = hinge_w + Rrot @ (eef_p - hinge_w); wp_q = T.mat2quat(th.as_tensor(Rrot @ Rq, dtype=th.float32))
                              ok = False
                              for attempt in range(3):
                                  try:
                                      run(env, skills.move_eef_cartesian("right", pose=(th.as_tensor(wp_p, dtype=th.float32), wp_q), check_collisions=False, ignore_objects=[obj], speed=0.08, torso=True)); ok = True; break
                                  except SkillError as e:
                                      # arm (+torso) cannot follow: back the base off along the pull direction and retry
                                      bx, by, byaw = skills.base_pose2d(); pxy = np.array([pull_w[0], pull_w[1]]); pn = np.linalg.norm(pxy)
                                      if pn < 1e-3: rec["steps"].append(f"pull chunk {k}: {e}"); break
                                      pxy /= pn
                                      try:
                                          run(env, skills.navigate_to((bx + 0.12 * pxy[0], by + 0.12 * pxy[1], byaw), ignore_objects=[obj])); rec["base_shifts"] = rec.get("base_shifts", 0) + 1
                                      except SkillError as e2:
                                          rec["steps"].append(f"pull chunk {k}: base shift failed: {e2}"); break
                              if not ok: rec["steps"].append(f"pull chunk {k}: arm+base could not follow"); break
                              qk = float(obj.get_joint_positions()[ji]); rec["dq"] = round(abs(qk - q0), 4)
                              if skills.held_object("right") is not obj: rec["steps"].append(f"pull chunk {k}: object released"); break
                          for _ in range(10): env.step(skills.state.action())
                          rec["dq"] = round(abs(float(obj.get_joint_positions()[ji]) - q0), 4)
                          rec["opened"] = bool(rec["dq"] >= 0.5 * target)
                          try: rec["open_state"] = bool(obj.states[object_states.Open].get_value())
                          except Exception: rec["open_state"] = None
                          grab(rp + np.array([-1.6, -1.4, 1.2]), rp + np.array([0, 0, ext[2] / 2]), f"{A}/frames/{key}__{j['joint']}_2.png")
                          run(env, skills.open_gripper("right"))
                          break
                      except SkillError as e:
                          rec["steps"].append(f"sign {sign:+.0f}: {e.reason}: {e}"); rec["error"] = f"{e.reason}: {e}"
                          try:
                              ep, eq = skills.eef_pose("right"); fq = robot.get_joint_positions()[robot.gripper_control_idx["right"]].tolist()
                              cands_set, links = robot._find_gripper_contacts(arm="right")
                              rec["diag"] = {"eef_pos_err_mm": round(float(th.linalg.norm(ep - th.as_tensor(origin, dtype=th.float32))) * 1000, 1), "eef_ang_err_deg": round(math.degrees(2 * math.acos(min(1.0, abs(float(th.dot(eq, quat)))))), 1),
                                             "finger_q_mm": [round(v * 1000) for v in fq], "gripper_contacts": sorted({c.split("/")[-2] + "/" + c.split("/")[-1] for c in cands_set})[:6], "contact_links": {k.split("/")[-1]: sorted(x.split("/")[-1] for x in v) for k, v in list(links.items())[:4]}}
                              # side view along the pinch axis so both fingers are visible
                              side = np.cross(approach_w, [0, 0, 1.0]); side = side / max(np.linalg.norm(side), 1e-9)
                              grab(contact_w + side * 0.45 - approach_w * 0.15 + np.array([0, 0, 0.12]), contact_w, f"{A}/frames/{key}__{j['joint']}_failside{int(sign)}.png")
                          except Exception as e2: rec["diag"] = {"error": repr(e2)}
                          grab(rp + np.array([-1.0, -0.9, 0.9]) if ext[2] < 0.6 else rp + np.array([-1.6, -1.4, 1.2]), np.array(contact_w), f"{A}/frames/{key}__{j['joint']}_fail{int(sign)}.png")
                          grab(contact_w + np.array([-0.5, -0.5, 0.35]), np.array(contact_w), f"{A}/frames/{key}__{j['joint']}_failzoom{int(sign)}.png")
                          try: run(env, skills.open_gripper("right"))
                          except Exception: pass
                          if e.reason != "planning": break
                  if done_grasp: break
                if done_grasp: break
            if not done_grasp and rec["stance"] is None: rec["error"] = rec["error"] or "unreachable: no stance"
            # restore the closed joint state for the next joint
            obj.set_joint_positions(q); obj.set_joint_velocities(th.zeros_like(q)); obj.keep_still(); og.sim.step()
        result["time_s"] = round(time.time() - t0, 1)
        print(f"{key}: " + "; ".join(f"{r['joint']}: {'AG' if r['ag'] else 'noAG'} dq={r['dq']}/{r['dq_target']} {'OPEN' if r['opened'] else '-'} {r['error'] or ''}" for r in result["joints"]), flush=True)
    except Exception as e:
        result["error"] = repr(e); print(f"FAILED {key}: {e!r}", flush=True); traceback.print_exc()
    finally:
        json.dump(result, open(f"{A}/results/{key}{'' if REPEAT == 1 else f'__rep{rep}'}.json", "w"), indent=1)
        try:
            skills.backend.detach_object("right") if hasattr(skills.backend, "detach_object") else None
        except Exception: pass
        try:
            og.sim.stop()
            if obj is not None: scene.remove_object(obj)
            og.sim.play(); og.sim.step()
        except Exception as e: print("cleanup failed", key, repr(e))
print("TABLE VALIDATION DONE", flush=True); og.shutdown()
