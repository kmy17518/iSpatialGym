"""Export collision meshes (object root-link frame, canonical orientation) + metadata for all job models."""
import json, os, sys, time, traceback
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import numpy as np
import torch as th
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.usd_utils import mesh_prim_shape_to_trimesh_mesh

OUT = "/tmp/glib"; ASSETS = "/home/minyeongk/Desktop/projects/ispatialgym/Behavior-1K/datasets/behavior-1k-assets/objects"
jobs = json.load(open(os.environ.get("JOBS", f"{OUT}/models.json")))
shard, nshard = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
jobs = [j for i, j in enumerate(jobs) if i % nshard == shard]
env = og.Environment(configs={"scene": {"type": "Scene"}})
scene = env.scene
for _ in range(3):
    og.sim.step()
done = 0; t_start = time.time()
for j in jobs:
    key = j["key"]; path = f"{OUT}/meshes/{key}.obj"
    if os.path.exists(path) and os.path.exists(f"{OUT}/meshes/{key}.json"):
        continue
    t0 = time.time(); obj = None
    try:
        obj = DatasetObject(name=f"o_{key}", category=j["category"], model=j["model"], fixed_base=False, include_default_states=False, **({"scale": th.tensor(j["scale"], dtype=th.float32)} if j.get("scale") else {}))
        scene.add_object(obj)
        obj.set_position_orientation(th.tensor([3.0, 3.0, 1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
        obj.disable_gravity(); obj.keep_still()
        og.sim.step()
        rp, rq = obj.root_link.get_position_orientation()
        pieces = []
        for link_name, link in obj.links.items():
            for cm_name, cm in link.collision_meshes.items():
                pos, quat = cm.get_position_orientation(); pos, quat = T.relative_pose_transform(pos, quat, rp, rq)
                if cm.geom_type == "Mesh":
                    verts = cm.points.numpy().astype(np.float64); faces = cm.faces.numpy().astype(np.int64)
                else:
                    tm = mesh_prim_shape_to_trimesh_mesh(cm.prim); verts = np.asarray(tm.vertices, dtype=np.float64); faces = np.asarray(tm.faces, dtype=np.int64)
                scale = cm.get_world_scale().numpy().astype(np.float64); R = T.quat2mat(quat).numpy().astype(np.float64)
                pieces.append((f"{link_name}__{cm_name}", (verts * scale) @ R.T + pos.numpy().astype(np.float64), faces))
        with open(path, "w") as f:
            off = 0
            for pname, v, fc in pieces:
                f.write(f"o {pname}\n")
                for x, y, z in v: f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                for a, b, c in fc: f.write(f"f {a+1+off} {b+1+off} {c+1+off}\n")
                off += len(v)
        allv = np.concatenate([p[1] for p in pieces]); lo, hi = allv.min(0), allv.max(0)
        alo, ahi = obj.aabb
        meta_path = f"{ASSETS}/{j['category']}/{j['model']}/misc/metadata.json"
        md = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        info = {**j, "obj": path, "n_pieces": len(pieces), "n_links": len(obj.links), "mass": float(obj.mass), "coll_bbox_min": lo.round(4).tolist(), "coll_bbox_max": hi.round(4).tolist(),
                "z_min": float(lo[2]), "visual_aabb_extent": (ahi - alo).tolist(), "visual_aabb_center_rel_root": ((alo + ahi) / 2 - rp).tolist(),
                "meta_links": md.get("meta_links", {}), "link_names": list(obj.links.keys()), "export_s": round(time.time() - t0, 2)}
        json.dump(info, open(f"{OUT}/meshes/{key}.json", "w"))
        done += 1
        print(f"[{done}/{len(jobs)}] {key} pieces={len(pieces)} links={len(obj.links)} bbox={np.round(hi-lo,3).tolist()} ({info['export_s']}s)", flush=True)
    except Exception as e:
        print(f"FAILED {key}: {e!r}", flush=True); traceback.print_exc()
        json.dump({**j, "error": repr(e)}, open(f"{OUT}/meshes/{key}.failed.json", "w"))
    finally:
        if obj is not None:
            try:
                scene.remove_object(obj)
            except Exception as e:
                print("remove failed", key, repr(e))
        og.sim.step()
print(f"EXPORT DONE {done} in {time.time()-t_start:.0f}s", flush=True)
og.shutdown()
