"""Export collision meshes (object base_link frame) of the selected BEHAVIOR models + the R1Pro
gripper sphere model in the right eef frame. Headless OmniGibson."""
import json, os, sys, time
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import numpy as np
import torch as th
import yaml

from ispatialgym.oracle import r1pro_model as M  # noqa: F401  (import before omnigibson)
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.usd_utils import mesh_prim_shape_to_trimesh_mesh

OUT = "/tmp/grasp_exp"
objects = json.load(open(f"{OUT}/objects.json"))

env = og.Environment(configs={"scene": {"type": "Scene"}, "robots": [M.joylo_robot_config("robot")]})
robot = env.robots[0]
scene = env.scene
for _ in range(5):
    og.sim.step()

# ---------------------------------------------------------------- gripper model (right arm) in eef frame
sph_yaml = yaml.safe_load(open(os.path.join(og.__path__[0], "..", "..", "datasets", "omnigibson-robot-assets", "models", "r1pro", "curobo", "r1pro_description_curobo_default.yaml"))) if False else None
sph_path = "/home/minyeongk/Desktop/projects/ispatialgym/Behavior-1K/datasets/omnigibson-robot-assets/models/r1pro/curobo/r1pro_description_curobo_default.yaml"
spheres = yaml.safe_load(open(sph_path))["robot_cfg"]["kinematics"]["collision_spheres"]
eef = robot.eef_links["right"]

def link_in_eef(link_name):
    lp, lq = robot.links[link_name].get_position_orientation()
    ep, eq = eef.get_position_orientation()
    return T.relative_pose_transform(lp, lq, ep, eq)

def gripper_spheres(finger_q):
    q = robot.get_joint_positions().clone()
    for j in M.FINGER_JOINTS["right"]:
        q[M.ALL_JOINTS.index(j)] = finger_q
    robot.set_joint_positions(q)
    og.sim.render()
    out = []
    for link in ["right_gripper_link", "right_arm_link7", "right_gripper_finger_link1", "right_gripper_finger_link2"]:
        p, qq = link_in_eef(link)
        R = T.quat2mat(qq)
        for s in spheres.get(link, []):
            c = (R @ th.tensor(s["center"], dtype=th.float32) + p).tolist()
            out.append({"link": link, "center": [round(v, 5) for v in c], "radius": s["radius"]})
    return out

gripper = {
    "open_q": 0.05, "closed_q": 0.0,
    "spheres_open": gripper_spheres(0.05),
    "spheres_closed": gripper_spheres(0.0),
    "eef_to_fingertip_lengths": {k: float(v) for k, v in robot.eef_to_fingertip_lengths["right"].items()} if isinstance(robot.eef_to_fingertip_lengths["right"], dict) else float(robot.eef_to_fingertip_lengths["right"]),
}
# finger pad z-extent in eef frame (from the finger spheres) -> contact centre
fz = [s["center"][2] for s in gripper["spheres_open"] if "finger" in s["link"]]
gripper["pad_z_range"] = [min(fz), max(fz)]
gripper["contact_center_z"] = float(np.mean(fz))
fy = [abs(s["center"][1]) - s["radius"] for s in gripper["spheres_open"] if "finger" in s["link"]]
gripper["max_opening"] = float(2 * min(fy))
json.dump(gripper, open(f"{OUT}/r1pro_right_gripper_eef.json", "w"), indent=1)
print("gripper model:", json.dumps({k: v for k, v in gripper.items() if "spheres" not in k}))

# ---------------------------------------------------------------- objects
results = []
for i, o in enumerate(objects):
    name = f"{o['category']}_{o['model']}"
    t0 = time.time()
    obj = DatasetObject(name=name, category=o["category"], model=o["model"], fixed_base=False, include_default_states=False)
    scene.add_object(obj)
    obj.set_position_orientation(th.tensor([2.0 + i * 1.0, 3.0, 1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    og.sim.step()
    rp, rq = obj.root_link.get_position_orientation()
    pieces = []
    for link_name, link in obj.links.items():
        for cm_name, cm in link.collision_meshes.items():
            pos, quat = cm.get_position_orientation()
            pos, quat = T.relative_pose_transform(pos, quat, rp, rq)
            if cm.geom_type == "Mesh":
                verts = cm.points.numpy().astype(np.float64); faces = cm.faces.numpy().astype(np.int64)
            else:
                tm = mesh_prim_shape_to_trimesh_mesh(cm.prim); verts = np.asarray(tm.vertices, dtype=np.float64); faces = np.asarray(tm.faces, dtype=np.int64)
            scale = cm.get_world_scale().numpy().astype(np.float64)
            R = T.quat2mat(quat).numpy().astype(np.float64)
            v = (verts * scale) @ R.T + pos.numpy().astype(np.float64)
            pieces.append((f"{link_name}__{cm_name}", v, faces))
    # write OBJ with one group per collision piece
    path = f"{OUT}/meshes/{name}.obj"
    with open(path, "w") as f:
        off = 0
        for pname, v, fc in pieces:
            f.write(f"o {pname}\n")
            for x, y, z in v: f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for a, b, c in fc: f.write(f"f {a+1+off} {b+1+off} {c+1+off}\n")
            off += len(v)
    allv = np.concatenate([p[1] for p in pieces])
    lo, hi = allv.min(0), allv.max(0)
    aabb_lo, aabb_hi = obj.aabb
    info = {**o, "name": name, "obj": path, "n_pieces": len(pieces), "n_links": len(obj.links), "mass": float(obj.mass),
            "coll_bbox_min": lo.round(4).tolist(), "coll_bbox_max": hi.round(4).tolist(), "z_min": float(lo[2]),
            "aabb_extent_world": (aabb_hi - aabb_lo).tolist(), "export_s": round(time.time() - t0, 2)}
    results.append(info)
    print(f"[{i}] {name}: pieces={len(pieces)} verts={len(allv)} bbox={np.round(hi-lo,3).tolist()} mass={info['mass']:.3f} ({info['export_s']}s)")
json.dump(results, open(f"{OUT}/objects_exported.json", "w"), indent=1)
print("EXPORT DONE")
og.shutdown()
