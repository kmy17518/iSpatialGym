"""Articulated-object inventory: for every articulation model in the manifest, load it in OmniGibson (fixed base, canonical
orientation, all joints closed), measure each openable joint empirically (axis direction, hinge point, opening sign), and export
the movable link's collision pieces + bounding data in the OBJECT frame at the closed state.

Outputs /tmp/glib/artic/inv/<cat>__<model>.json and <cat>__<model>__<link>.obj (collision pieces of the movable link)."""
import json, os, sys, time, traceback
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import numpy as np, torch as th
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.usd_utils import mesh_prim_shape_to_trimesh_mesh

OUT = "/tmp/glib/artic/inv"; os.makedirs(OUT, exist_ok=True)
ASSETS = "/home/minyeongk/Desktop/projects/ispatialgym/Behavior-1K/datasets/behavior-1k-assets/objects"
manifest = json.load(open("/tmp/glib/manifest.json"))
jobs = sorted({(m["category"], m["model"]) for m in manifest if m["task"] == "articulation"})
shard, nshard = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
jobs = [j for i, j in enumerate(jobs) if i % nshard == shard]
env = og.Environment(configs={"scene": {"type": "Scene"}}); scene = env.scene
for _ in range(3): og.sim.step()


def link_pieces_local(link):
    """Collision pieces of @link expressed in the link frame (list of (name, verts, faces))."""
    lp, lq = link.get_position_orientation(); out = []
    for cm_name, cm in link.collision_meshes.items():
        pos, quat = cm.get_position_orientation(); pos, quat = T.relative_pose_transform(pos, quat, lp, lq)
        if cm.geom_type == "Mesh": verts = cm.points.numpy().astype(np.float64); faces = cm.faces.numpy().astype(np.int64)
        else:
            tm = mesh_prim_shape_to_trimesh_mesh(cm.prim); verts = np.asarray(tm.vertices, dtype=np.float64); faces = np.asarray(tm.faces, dtype=np.int64)
        scale = cm.get_world_scale().numpy().astype(np.float64); R = T.quat2mat(quat).numpy().astype(np.float64)
        out.append((cm_name, (verts * scale) @ R.T + pos.numpy().astype(np.float64), faces))
    return out


def to_obj_frame(p, q, rp, rq):
    pos, quat = T.relative_pose_transform(th.as_tensor(p, dtype=th.float32), th.as_tensor(q, dtype=th.float32), rp, rq)
    return pos.numpy().astype(np.float64), quat.numpy().astype(np.float64)


done = 0; t_start = time.time()
for cat, model in jobs:
    key = f"{cat}__{model}"
    if os.path.exists(f"{OUT}/{key}.json"): continue
    t0 = time.time(); obj = None
    try:
        og.sim.stop()
        obj = DatasetObject(name=f"o_{key}", category=cat, model=model, fixed_base=True, include_default_states=False)
        scene.add_object(obj); obj.set_position_orientation(th.tensor([0.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
        og.sim.play()
        for _ in range(2): og.sim.step()
        # put the object so that its AABB bottom sits on z=0 (canonical upright), all joints at the closed limit
        alo, ahi = obj.aabb; rp0, rq0 = obj.root_link.get_position_orientation()
        obj.set_position_orientation(th.tensor([0.0, 0.0, float(rp0[2] - alo[2])]), th.tensor([0.0, 0.0, 0.0, 1.0]))
        jl = obj.joint_lower_limits.clone(); ju = obj.joint_upper_limits.clone()
        md_path = f"{ASSETS}/{cat}/{model}/misc/metadata.json"; md = json.load(open(md_path)) if os.path.exists(md_path) else {}
        oj = md.get("openable_joint_ids") or []
        joint_dir = {}
        for e in oj:                                       # [(id, name)] or [(id, name, direction)]
            if isinstance(e, (list, tuple)) and len(e) >= 2: joint_dir[e[1]] = (e[2] if len(e) > 2 else 1)
        if not joint_dir:                                  # no annotation: every joint counts (as the Open state does)
            joint_dir = {jn: 1 for jn in obj.joints}
        # closed state: for direction +1 the closed end is the lower limit, for -1 the upper limit
        q_closed = jl.clone()
        for i, (jn, j) in enumerate(obj.joints.items()):
            if joint_dir.get(jn, 1) < 0: q_closed[i] = ju[i]
        obj.set_joint_positions(q_closed); obj.set_joint_velocities(th.zeros_like(q_closed)); obj.keep_still(); og.sim.step()
        rp, rq = obj.root_link.get_position_orientation(); alo, ahi = obj.aabb
        info = {"category": cat, "model": model, "key": key, "root_pos_world": rp.tolist(), "aabb_extent": (ahi - alo).tolist(), "aabb_min_rel_root": (alo - rp).tolist(), "aabb_max_rel_root": (ahi - rp).tolist(),
                "link_tags": {k: list(v) for k, v in (md.get("link_tags") or {}).items()}, "openable_joint_ids": oj, "joints": [], "base_link": {}}
        # base link bbox (all pieces) in object frame
        base_pieces = link_pieces_local(obj.root_link); bl_p, bl_q = to_obj_frame(*obj.root_link.get_position_orientation(), rp, rq)
        Rb = T.quat2mat(th.as_tensor(bl_q, dtype=th.float32)).numpy()
        bv = np.concatenate([v @ Rb.T + bl_p for _, v, _ in base_pieces]) if base_pieces else np.zeros((1, 3))
        info["base_link"] = {"bbox_min": bv.min(0).round(4).tolist(), "bbox_max": bv.max(0).round(4).tolist(), "n_pieces": len(base_pieces)}
        names = list(obj.joints.keys())
        for i, jn in enumerate(names):
            j = obj.joints[jn]
            if jn not in joint_dir: continue
            child = j.body1.split("/")[-1]; link = obj.links[child]
            # empirical axis: perturb the joint by a small delta towards the open side and watch the child link
            q1 = q_closed.clone(); d = joint_dir[jn]; span = float(ju[i] - jl[i])
            delta = min(0.05 if j.joint_type == "PrismaticJoint" else 0.25, 0.5 * span) * d
            p0, o0 = link.get_position_orientation()
            q1[i] = q_closed[i] + delta; obj.set_joint_positions(q1); obj.keep_still(); og.sim.step()
            p1, o1 = link.get_position_orientation()
            obj.set_joint_positions(q_closed); obj.keep_still(); og.sim.step()
            p0o, o0o = to_obj_frame(p0, o0, rp, rq); p1o, o1o = to_obj_frame(p1, o1, rp, rq)
            entry = {"joint": jn, "type": j.joint_type, "child_link": child, "lower": float(jl[i]), "upper": float(ju[i]), "closed_q": float(q_closed[i]), "open_sign": int(d),
                     "link_pos_obj": p0o.round(5).tolist(), "link_quat_obj": o0o.round(5).tolist()}
            if j.joint_type == "PrismaticJoint":
                v = p1o - p0o; entry["axis_dir_obj"] = (v / max(np.linalg.norm(v), 1e-9)).round(5).tolist(); entry["travel_per_unit"] = float(np.linalg.norm(v) / abs(delta))
            else:
                R0 = T.quat2mat(th.as_tensor(o0o, dtype=th.float32)).numpy(); R1 = T.quat2mat(th.as_tensor(o1o, dtype=th.float32)).numpy()
                Rrel = R1 @ R0.T; ang = float(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
                ax = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]]); ax = ax / max(np.linalg.norm(ax), 1e-9)
                # hinge point: solve (I - Rrel) c = p1 - Rrel p0 (least squares, direction along ax is free)
                A = np.eye(3) - Rrel; b = p1o - Rrel @ p0o; c, *_ = np.linalg.lstsq(A, b, rcond=None)
                c = c - ax * np.dot(c - p0o, ax) + ax * np.dot(p0o - c, ax) * 0  # keep as is (any point on the axis)
                entry["axis_dir_obj"] = ax.round(5).tolist(); entry["hinge_point_obj"] = c.round(5).tolist(); entry["measured_angle"] = round(ang, 4); entry["expected_angle"] = round(abs(delta), 4)
            # movable link pieces in object frame (closed state)
            pieces = link_pieces_local(link); lp, lq = to_obj_frame(p0, o0, rp, rq); Rl = T.quat2mat(th.as_tensor(lq, dtype=th.float32)).numpy()
            path = f"{OUT}/{key}__{child}.obj"
            with open(path, "w") as f:
                off = 0
                for pname, v, fc in pieces:
                    vw = v @ Rl.T + lp; f.write(f"o {pname}\n")
                    for x, y, z in vw: f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                    for a, b_, c_ in fc: f.write(f"f {a+1+off} {b_+1+off} {c_+1+off}\n")
                    off += len(vw)
            allv = np.concatenate([v @ Rl.T + lp for _, v, _ in pieces]) if pieces else np.zeros((1, 3))
            entry.update({"pieces_obj": path, "n_pieces": len(pieces), "link_bbox_min": allv.min(0).round(4).tolist(), "link_bbox_max": allv.max(0).round(4).tolist(),
                          "link_tags": list((md.get("link_tags") or {}).get(child, []))})
            # extra child links attached to this link (e.g. door handle links) -> their bboxes
            subs = []
            for jn2, j2 in obj.joints.items():
                if j2.body0.split("/")[-1] == child:
                    sl = obj.links[j2.body1.split("/")[-1]]; sp = link_pieces_local(sl); slp, slq = to_obj_frame(*sl.get_position_orientation(), rp, rq)
                    Rs = T.quat2mat(th.as_tensor(slq, dtype=th.float32)).numpy(); sv = np.concatenate([v @ Rs.T + slp for _, v, _ in sp]) if sp else None
                    if sv is not None: subs.append({"link": sl.name.split("/")[-1] if "/" in sl.name else j2.body1.split("/")[-1], "joint": jn2, "bbox_min": sv.min(0).round(4).tolist(), "bbox_max": sv.max(0).round(4).tolist()})
            entry["sub_links"] = subs
            info["joints"].append(entry)
        json.dump(info, open(f"{OUT}/{key}.json", "w"), indent=1)
        done += 1; print(f"[{done}/{len(jobs)}] {key}: {len(info['joints'])} openable joints, aabb {np.round(ahi - alo, 2).tolist()} ({time.time() - t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"FAILED {key}: {e!r}", flush=True); traceback.print_exc(); json.dump({"category": cat, "model": model, "error": repr(e)}, open(f"{OUT}/{key}.failed.json", "w"))
    finally:
        try:
            og.sim.stop()
            if obj is not None: scene.remove_object(obj)
            og.sim.play(); og.sim.step()
        except Exception as e: print("cleanup failed", key, repr(e))
print(f"INVENTORY DONE {done} in {time.time() - t_start:.0f}s", flush=True); og.shutdown()
