"""Approach 2: GraspGen (Franka Panda checkpoint, pure diffusion planner) on the exported collision meshes.
Runs in the `graspgen` conda env. Writes raw grasps (Panda hand frame in the object frame) as isaac_grasp YAML."""
import json, os, sys, time
import numpy as np
import torch
import trimesh
import trimesh.transformations as tra

sys.path.insert(0, "/home/minyeongk/Desktop/projects/graspgen/GraspGen")
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.dataset.eval_utils import save_to_isaac_grasp_format

OUT = "/tmp/grasp_exp"
CKPT = "/home/minyeongk/Desktop/projects/graspgen/GraspGenModels/checkpoints"
GRIPPER = sys.argv[1] if len(sys.argv) > 1 else "franka_panda"
NUM_GRASPS, TOPK, NUM_POINTS = int(os.environ.get("GG_N", 400)), int(os.environ.get("GG_TOPK", 100)), 2000

t0 = time.time()
cfg = load_grasp_cfg(f"{CKPT}/graspgen_{GRIPPER}.yml")
sampler = GraspGenSampler(cfg)
t_load = time.time() - t0
print(f"model load: {t_load:.1f} s  (gripper={cfg.data.gripper_name})")

results = {}
for obj in json.load(open(f"{OUT}/objects_exported.json")):
    name = obj["name"]
    t1 = time.time()
    mesh = trimesh.load(obj["obj"], force="mesh", process=False)
    xyz, _ = trimesh.sample.sample_surface(mesh, NUM_POINTS)
    xyz = np.asarray(xyz, dtype=np.float32)
    T_center = tra.translation_matrix(-xyz.mean(axis=0))
    pc = tra.transform_points(xyz, T_center)
    t_prep = time.time() - t1
    torch.cuda.synchronize(); t2 = time.time()
    grasps, conf = GraspGenSampler.run_inference(pc, sampler, grasp_threshold=-1.0, num_grasps=NUM_GRASPS, topk_num_grasps=TOPK, remove_outliers=False)
    torch.cuda.synchronize(); t_inf = time.time() - t2
    grasps = grasps.cpu().numpy() if len(grasps) else np.zeros((0, 4, 4), np.float32)
    conf = conf.cpu().numpy() if len(conf) else np.zeros((0,), np.float32)
    grasps = np.array([tra.inverse_matrix(T_center) @ g for g in grasps])  # back to the object frame
    order = np.argsort(-conf); grasps, conf = grasps[order], conf[order]
    save_to_isaac_grasp_format(grasps, conf, f"{OUT}/a2_{GRIPPER}_{name}_raw{os.environ.get('GG_TAG','')}.yaml")
    np.savez(f"{OUT}/a2_{GRIPPER}_{name}_raw{os.environ.get('GG_TAG','')}.npz", T=grasps, conf=conf)
    results[name] = {"n_returned": int(len(grasps)), "conf_min": float(conf.min()) if len(conf) else None, "conf_max": float(conf.max()) if len(conf) else None,
                     "t_prep_s": round(t_prep, 3), "t_inference_s": round(t_inf, 3)}
    print(f"{name:28s} {len(grasps):4d} grasps  conf {results[name]['conf_min']}-{results[name]['conf_max']}  prep {t_prep*1000:5.0f} ms  inference {t_inf*1000:6.0f} ms")
results["_model_load_s"] = round(t_load, 2)
json.dump(results, open(f"{OUT}/a2_{GRIPPER}_results{os.environ.get('GG_TAG','')}.json", "w"), indent=1)
print("A2 DONE")
