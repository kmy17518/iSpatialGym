"""Robot-in-the-loop opening test for SCENE articulated objects. For one challenge scene: load it with R1Pro, and for every test
assigned to this scene (artic/scene_assign.json) put the model under test at its host instance (same model: use as is; other
model: swap it in scaled to the host's AABB, AABB centres aligned), close its joints, and for every openable joint with a
reviewed/heuristic handle: teleport the robot in front of the handle, find a stance (base pose + trunk height), grasp the handle
(assisted grasping) and pull along the opening path, driving the base backwards along the pull direction whenever the arm alone
cannot follow. Unreachable handles are recorded with the reason.

usage: python artic_validate_scene.py <scene_name>"""
import json, os, sys, time, math, traceback
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import numpy as np, torch as th
from PIL import Image
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.skills import Skills, SkillError, run
from ispatialgym.oracle.curobo_planner import R1ProCuRobo
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson import object_states

A = "/tmp/glib/artic"; os.makedirs(f"{A}/results", exist_ok=True); os.makedirs(f"{A}/frames", exist_ok=True)
CONTACT_Z = json.load(open("/tmp/grasp_exp/r1pro_right_gripper_eef.json"))["contact_center_z"]
MANUAL = json.load(open(f"{A}/manual_handles.json")) if os.path.exists(f"{A}/manual_handles.json") else {}
scene_name = sys.argv[1]
assign = [a for a in json.load(open(f"{A}/scene_assign.json")) if a["scene"] == scene_name]
if os.environ.get("ONLY"): assign = [a for a in assign if f"{a['category']}__{a['model']}".startswith(os.environ["ONLY"])]
assign = [a for a in assign if not os.path.exists(f"{A}/results/{a['category']}__{a['model']}.json")]
LIMIT = int(os.environ.get("LIMIT", "1000")); assign = assign[:LIMIT]                           # fresh process every LIMIT tests (stop/play cycles degrade the sim)
print(f"{scene_name}: {len(assign)} tests in this process", flush=True)
if not assign: sys.exit(0)

env = og.Environment(configs={"scene": {"type": "InteractiveTraversableScene", "scene_model": scene_name}, "robots": [M.joylo_robot_config("robot", obs_modalities=["rgb"])]})
robot = env.robots[0]; scene = env.scene
for _ in range(5): og.sim.step()
FLOOR_Z = float(robot.get_position_orientation()[0][2])
skills = Skills(env, robot, backend=R1ProCuRobo(robot, prebuild=("right_arm", "base"), mesh_cache_size=8000))   # many swaps add meshes; removed ones stay cached
cam = og.sim.viewer_camera
for _ in range(6): og.sim.render()


def look_at(cam_pos, target, up=th.tensor([0.0, 0.0, 1.0])):
    f = target - cam_pos; f = f / th.linalg.norm(f); r = th.linalg.cross(f, up)
    if th.linalg.norm(r) < 1e-3: r = th.linalg.cross(f, th.tensor([1.0, 0.0, 0.0]))
    r = r / th.linalg.norm(r); u = th.linalg.cross(r, f); return T.mat2quat(th.stack([r, u, -f], dim=1))


def grab(cam_pos, target, path, size=420):
    cam.set_position_orientation(th.as_tensor(cam_pos, dtype=th.float32), look_at(th.as_tensor(cam_pos, dtype=th.float32), th.as_tensor(target, dtype=th.float32)))
    for _ in range(3): og.sim.render()
    rgb = cam.get_obs()[0]["rgb"][..., :3].cpu().numpy(); h, w = rgb.shape[:2]; s = min(h, w)
    Image.fromarray(rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]).resize((size, size), Image.LANCZOS).save(path)


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


def selected_grasp(joint, review, key=None):
    mj = MANUAL.get(key or "", {}).get(joint["joint"])
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


def reset_robot(pos_xy, yaw):
    robot.set_position_orientation(th.tensor([pos_xy[0], pos_xy[1], FLOOR_Z]), th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]))
    q = robot.get_joint_positions().clone()
    for arm in ("left", "right"): q[robot.arm_control_idx[arm]] = th.zeros(len(robot.arm_control_idx[arm])); q[robot.gripper_control_idx[arm]] = 0.05   # arms hanging (JoyLo reset): lets the stance search stand close to counters
    robot.set_joint_positions(q); robot.set_joint_velocities(th.zeros_like(q)); robot.keep_still()
    skills.state.hold_current(); skills.state.set_gripper("right", close=False); skills.state.set_gripper("left", close=False)
    for _ in range(5): env.step(skills.state.action())


hosts = {}                                                                                   # original host geometry per host name
current = {}                                                                                 # host name -> object currently standing there
expanded = []
for a in assign:
    for k, hn in enumerate([a["host"]] + a.get("host_alternatives", [])): expanded.append((a, hn, k))
results_by_key = {}
for a, hname, host_rank in expanded:
    key = f"{a['category']}__{a['model']}"
    prev = results_by_key.get(key)
    if prev is not None and any(j.get("reachable") for j in prev["joints"]): continue      # primary host worked: skip alternatives
    H = json.load(open(f"{A}/handles/{key}.json"))
    review = json.load(open(f"{A}/review/{key}.json")) if os.path.exists(f"{A}/review/{key}.json") else None
    result = {"key": key, "category": a["category"], "model": a["model"], "scene": scene_name, "host": hname, "host_rank": host_rank, "host_model": a["host_model"] if host_rank == 0 else None, "swap": None, "joints": [], "error": None, "previous_hosts": ([prev["host"]] + prev.get("previous_hosts", [])) if prev else []}
    obj = None; t0 = time.time()
    try:
        if hname not in hosts:
            h = scene.object_registry("name", hname)
            if h is None: raise RuntimeError(f"host {hname} not in scene")
            lo, hi = h.aabb; hosts[hname] = {"center": ((lo + hi) / 2).clone(), "extent": (hi - lo).clone(), "ori": h.get_position_orientation()[1].clone(), "obj": h, "model": h.model}
            current[hname] = h
        hd = hosts[hname]; result["swap"] = hd["model"] != a["model"]; result["host_model"] = hd["model"]
        if hd["model"] == a["model"] and current[hname] is hd["obj"]:
            obj = hd["obj"]
        else:
            ext_h = th.clamp(hd["extent"], min=0.02)                                                        # never a zero extent (singular scale)
            ext_t = th.tensor(H["aabb_extent"]); factors = ext_h / th.clamp(ext_t, min=1e-3)
            if float(factors.max()) > 5.0 or float(factors.min()) < 0.2:
                raise RuntimeError(f"swap skipped: host/test aspect mismatch, scale factors {[round(float(x), 2) for x in factors]}")
            og.sim.stop()
            if current[hname] is not None:
                try: scene.remove_object(current[hname])
                except Exception as e: print("remove failed", hname, repr(e), flush=True)
            current[hname] = None
            stale = scene.object_registry("name", f"o_{key}")
            if stale is not None:
                try: scene.remove_object(stale)
                except Exception as e: print("stale remove failed", key, repr(e), flush=True)
            obj = DatasetObject(name=f"o_{key}", category=a["category"], model=a["model"], bounding_box=ext_h, fixed_base=True)
            scene.add_object(obj); og.sim.play()
            for _ in range(2): og.sim.step()
            obj.set_position_orientation(hd["center"], hd["ori"]); og.sim.step()
            lo, hi = obj.aabb; obj.set_position_orientation(obj.get_position_orientation()[0] + (hd["center"] - (lo + hi) / 2), hd["ori"]); og.sim.step()
            current[hname] = obj
        result["scale"] = [round(float(x), 3) for x in obj.scale]
        names = list(obj.joints.keys()); q = obj.get_joint_positions().clone()
        for j in H["joints"]:
            if j["joint"] in names: q[names.index(j["joint"])] = j["closed_q"]
        obj.set_joint_positions(q); obj.set_joint_velocities(th.zeros_like(q)); obj.keep_still(); og.sim.step()
        rp, rq = obj.root_link.get_position_orientation(); rp = rp.numpy().astype(np.float64); Rr = T.quat2mat(rq).numpy().astype(np.float64); sc = obj.scale.numpy().astype(np.float64)
        def to_world(p): return rp + Rr @ (np.asarray(p, float) * sc)
        def dir_world(d): v = Rr @ np.asarray(d, float); return v / max(np.linalg.norm(v), 1e-9)
        ext = (obj.aabb[1] - obj.aabb[0]).numpy(); ctr = ((obj.aabb[0] + obj.aabb[1]) / 2).numpy()
        for j in H["joints"]:
            cand, src = selected_grasp(j, review, key)
            rec = {"joint": j["joint"], "kind": j.get("kind"), "grasp_source": src, "stance": None, "reachable": None, "ag": False, "dq": 0.0, "dq_target": None, "opened": False, "open_state": None, "base_shifts": 0, "steps": [], "error": None}
            result["joints"].append(rec)
            if cand is None: rec["error"] = "no graspable handle"; continue
            cand_list = [cand] + ([cand['alternative']] if isinstance(cand, dict) and cand.get('alternative') else [])
            for ci, cand in enumerate(cand_list):
                contact_w = to_world(cand["contact"]); approach_w = dir_world(cand["approach_dir"]); pinch_w = dir_world(cand["pinch_axis"]); pull_w = dir_world(cand["pull_dir"])
                rec["contact_world"] = contact_w.round(3).tolist(); rec["handle_height_above_floor"] = round(float(contact_w[2] - FLOOR_Z), 3)
                n_w = -approach_w if abs(approach_w[2]) < 0.7 else -dir_world(np.array(cand["approach_dir"]) * np.array([1, 1, 0])) if np.linalg.norm(np.array(cand["approach_dir"])[:2]) > 1e-3 else np.array([1.0, 0, 0])
                n_xy = np.array([n_w[0], n_w[1], 0.0]); n_xy /= max(np.linalg.norm(n_xy), 1e-9)
                ji = names.index(j["joint"]); q0 = float(obj.get_joint_positions()[ji]); span = j["upper"] - j["lower"]; is_prism = j["type"] == "PrismaticJoint"
                target = min(0.15, 0.6 * span) if is_prism else min(math.radians(45), 0.6 * span); rec["dq_target"] = round(target, 3)
                start_xy = contact_w[:2] + n_xy[:2] * 1.1; yaw0 = math.atan2(-n_xy[1], -n_xy[0])
                done_grasp = False
                for olabel, o_app, o_pinch in orientation_options(cand, j.get('kind')):
                  approach_w = dir_world(o_app); pinch_w = dir_world(o_pinch)
                  for sign in (1.0, -1.0):
                      quat = eef_quat_from(approach_w, pinch_w, sign); z_eef = T.quat2mat(quat)[:, 2].numpy()
                      origin = contact_w - z_eef * CONTACT_Z; grasp = (th.as_tensor(origin, dtype=th.float32), quat); pre = (th.as_tensor(origin - z_eef * 0.10, dtype=th.float32), quat)
                      stance = None; last_err = None
                      for lateral, dang in ((-0.2, 0.0), (0.0, 0.0), (-0.2, 0.5), (-0.2, -0.5)):
                          try:
                              # approach direction rotated by dang about the handle: lets the robot stand at an angle in tight kitchens
                              ca, sa = math.cos(dang), math.sin(dang); nrot = np.array([ca * n_xy[0] - sa * n_xy[1], sa * n_xy[0] + ca * n_xy[1]])
                              reset_robot(contact_w[:2] + nrot * 1.1, math.atan2(-nrot[1], -nrot[0])); skills.backend.update_world()
                              stance = skills.stance_for(th.as_tensor(contact_w, dtype=th.float32), arm="right", eef_candidates=[pre, grasp], distances=(0.45, 0.55, 0.65, 0.8), translates=(0.5, 1.0, 0.0, 1.5, 2.0), lateral=lateral, ignore_objects=[obj]); break
                          except SkillError as e: last_err = e
                      if stance is None:
                          rec["steps"].append(f"sign {sign:+.0f}: no stance ({last_err})")
                          try:                                                                   # classify: collision-bound or IK-bound
                              from ispatialgym.oracle.trunk import TrunkState
                              qf = skills.backend.q_full(); emb = skills._emb_for("right"); bp = robot.get_position_orientation()[0]
                              theta = math.atan2(float(contact_w[1] - bp[1]), float(contact_w[0] - bp[0])); free = 0; ik_free = 0; ik_no = 0; n = 0
                              skills.backend.update_world(ignore_objects=[obj])
                              probes = [(tr, d) for tr in (0.0, 0.5, 1.0, 1.5, 2.0) for d in (0.45, 0.6, 0.75)]
                              qs = []
                              for tr, d in probes:
                                  bx = contact_w[0] - d * math.cos(theta) + 0.2 * (-math.sin(theta)); by = contact_w[1] - d * math.sin(theta) + 0.2 * math.cos(theta)
                                  qk = skills.q_with_base_pose(qf, bx, by, theta); qk[robot.trunk_control_idx] = TrunkState(tr, 0.0).q(skills._joint_limits); qs.append(qk); n += 1
                              coll = skills.backend.check_collision(th.stack(qs), emb_name="whole_body", margin=0.01).tolist()
                              for qk, c in zip(qs, coll):
                                  if not c:
                                      free += 1
                                      if skills.backend.solve_ik(emb, {M.EEF_LINK["right"]: grasp}, q_seed=qk, sync_world=False) is not None: ik_free += 1
                              skills.backend.update_world(ignore_objects=list(scene.objects))
                              for qk in qs:
                                  if skills.backend.solve_ik(emb, {M.EEF_LINK["right"]: grasp}, q_seed=qk, sync_world=False) is not None: ik_no += 1
                              skills.backend.update_world()
                              rec["diag_stance"] = {"probed": n, "collision_free": free, "grasp_ik_from_free_stance": ik_free, "grasp_ik_without_obstacles": ik_no}
                          except Exception as e2: rec["diag_stance"] = {"error": repr(e2)}
                          continue
                      rec["stance"] = {"base": [round(float(v), 3) for v in stance[0]], "translate": round(float(stance[1]), 2), "sign": sign, "orientation": olabel}; rec["reachable"] = True
                      try:
                          # this harness tests grasping/opening, not navigation: teleport the base to the stance, then set the trunk
                          (bx_, by_, yaw_), tr_ = stance; reset_robot((bx_, by_), yaw_)
                          try: run(env, skills.set_torso(translate=tr_))
                          except SkillError as e_st:
                              if e_st.reason not in ("planning", "collision"): raise
                              run(env, skills.set_torso(translate=tr_, auto_ready_arms=False))
                          rec["steps"].append("stance ok (teleported)")
                          cam_pos = contact_w + n_xy * 1.8 + np.cross(n_xy, [0, 0, 1.0]) * 1.2 + np.array([0, 0, 0.9])
                          grab(cam_pos, contact_w, f"{A}/frames/{key}__{j['joint']}_0.png")
                          run(env, skills.grasp("right", obj=obj, grasp_pose=grasp, pregrasp_dist=0.10, lift_dist=0.0, torso=False))
                          rec["ag"] = True; rec["steps"].append("grasped"); done_grasp = True
                          grab(cam_pos, contact_w, f"{A}/frames/{key}__{j['joint']}_1.png")
                          eef_p, eef_q = skills.eef_pose("right"); eef_p = eef_p.numpy().astype(np.float64); Rq = T.quat2mat(eef_q).numpy()
                          n_chunks = 6
                          big_arc = (not is_prism) and (a["category"] in ("door", "fridge", "oven", "dishwasher", "clothes_dryer") or np.linalg.norm((eef_p - to_world(j["hinge_point_obj"]))[:2]) > 0.45)
                          base0 = np.array(skills.base_pose2d(), float)
                          for k in range(1, n_chunks + 1):
                              frac = k / n_chunks
                              if is_prism: wp_p = eef_p + pull_w * target * frac; wp_q = eef_q
                              else:
                                  hinge_w = to_world(j["hinge_point_obj"]); ax = dir_world(np.array(j["axis_dir_obj"]) * j["open_sign"]); ang = target * frac
                                  Rrot = T.quat2mat(T.axisangle2quat(th.as_tensor(ax * ang, dtype=th.float32))).numpy()
                                  wp_p = hinge_w + Rrot @ (eef_p - hinge_w); wp_q = T.mat2quat(th.as_tensor(Rrot @ Rq, dtype=th.float32))
                              if big_arc:
                                  # drive the base along with the handle (JoyLo-style): rotate the base about a vertical hinge, or back off by the
                                  # horizontal handle displacement for horizontal hinges; the AG joint drags the door while the arm holds its posture
                                  if abs(ax[2]) > 0.7:
                                      R2 = Rrot[:2, :2]; nb = hinge_w[:2] + R2 @ (base0[:2] - hinge_w[:2]); nyaw = base0[2] + ang * np.sign(ax[2])
                                  else:
                                      d_h = (wp_p - eef_p); d_h[2] = 0; nb = base0[:2] + d_h[:2]; nyaw = base0[2]
                                  try:
                                      run(env, skills.navigate_to((float(nb[0]), float(nb[1]), float(nyaw)), ignore_objects=[obj])); rec["base_shifts"] += 1
                                  except SkillError as e_b: rec["steps"].append(f"pull chunk {k}: base follow failed: {e_b}")
                              ok = False
                              for attempt in range(3):
                                  try:
                                      run(env, skills.move_eef_cartesian("right", pose=(th.as_tensor(wp_p, dtype=th.float32), wp_q), check_collisions=False, ignore_objects=[obj], speed=0.08, torso=True)); ok = True; break
                                  except SkillError as e:
                                      # arm cannot follow: back the base off along the pull direction and retry (JoyLo operators drive backwards while holding the handle)
                                      bx, by, byaw = skills.base_pose2d(); pxy = np.array([pull_w[0], pull_w[1]]); pxy = pxy / max(np.linalg.norm(pxy), 1e-9)
                                      try:
                                          run(env, skills.navigate_to((bx + 0.15 * pxy[0], by + 0.15 * pxy[1], byaw), ignore_objects=[obj])); rec["base_shifts"] += 1
                                      except SkillError as e2:
                                          rec["steps"].append(f"pull chunk {k}: base shift failed: {e2}"); break
                              if not ok: rec["steps"].append(f"pull chunk {k}: arm+base could not follow"); break
                              qk = float(obj.get_joint_positions()[ji]); rec["dq"] = round(abs(qk - q0), 4)
                              if skills.held_object("right") is not obj: rec["steps"].append(f"pull chunk {k}: object released"); break
                          for _ in range(10): env.step(skills.state.action())
                          rec["dq"] = round(abs(float(obj.get_joint_positions()[ji]) - q0), 4); rec["opened"] = bool(rec["dq"] >= 0.5 * target)
                          try: rec["open_state"] = bool(obj.states[object_states.Open].get_value())
                          except Exception: rec["open_state"] = None
                          grab(cam_pos, contact_w, f"{A}/frames/{key}__{j['joint']}_2.png")
                          run(env, skills.open_gripper("right"))
                          break
                      except SkillError as e:
                          rec["steps"].append(f"sign {sign:+.0f}: {e.reason}: {e}"); rec["error"] = f"{e.reason}: {e}"
                          grab(contact_w + n_xy * 0.6 + np.cross(n_xy, [0, 0, 1.0]) * 0.5 + np.array([0, 0, 0.35]), contact_w, f"{A}/frames/{key}__{j['joint']}_fail{int(sign)}.png")
                          try: run(env, skills.open_gripper("right"))
                          except Exception: pass
                          if e.reason != "planning": break
                  if done_grasp: break
                if done_grasp: break
            if rec["stance"] is None: rec["reachable"] = False; rec["error"] = rec["error"] or "unreachable: no collision-free stance with IK to the handle"
            obj.set_joint_positions(q); obj.set_joint_velocities(th.zeros_like(q)); obj.keep_still(); og.sim.step()
            try: run(env, skills.open_gripper("right"))
            except Exception: pass
        result["time_s"] = round(time.time() - t0, 1)
        print(f"{key} @ {a['host']}: " + "; ".join(f"{r['joint']}: {'reach' if r['reachable'] else 'UNREACH'} {'AG' if r['ag'] else 'noAG'} dq={r['dq']}/{r['dq_target']} {'OPEN' if r['opened'] else '-'} shifts={r['base_shifts']} {r['error'] or ''}" for r in result["joints"]), flush=True)
    except Exception as e:
        result["error"] = repr(e); print(f"FAILED {key}: {e!r}", flush=True); traceback.print_exc()
    finally:
        results_by_key[key] = result
        json.dump(result, open(f"{A}/results/{key}.json", "w"), indent=1)
print(f"SCENE {scene_name} DONE", flush=True); og.shutdown()
