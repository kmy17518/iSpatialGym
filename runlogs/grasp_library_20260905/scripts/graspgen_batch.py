"""GraspGen (Franka Panda checkpoint, diffusion planner) on all graspgen-pipeline models -> raw npz (Panda hand frame,
object root frame). Run in the `graspgen` conda env."""
import json, os, sys, time
import numpy as np, torch, trimesh, trimesh.transformations as tra
sys.path.insert(0, "/home/minyeongk/Desktop/projects/graspgen/GraspGen")
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
OUT = "/tmp/glib"; CKPT = "/home/minyeongk/Desktop/projects/graspgen/GraspGenModels/checkpoints"
jobs = [j for j in json.load(open(os.environ.get("JOBS", f"{OUT}/models.json"))) if j["pipeline"] == "graspgen"]
t0 = time.time(); cfg = load_grasp_cfg(f"{CKPT}/graspgen_franka_panda.yml"); sampler = GraspGenSampler(cfg); print(f"model load {time.time()-t0:.1f}s")
os.makedirs(f"{OUT}/graspgen", exist_ok=True)
times = []
for i, j in enumerate(jobs):
    key = j["key"]; out = f"{OUT}/graspgen/{key}.npz"
    if os.path.exists(out):
        continue
    info_path = f"{OUT}/meshes/{key}.json"
    if not os.path.exists(info_path):
        print("no mesh for", key); continue
    mesh = trimesh.load(f"{OUT}/meshes/{key}.obj", force="mesh", process=False)
    xyz, _ = trimesh.sample.sample_surface(mesh, 2000); xyz = np.asarray(xyz, np.float32)
    Tc = tra.translation_matrix(-xyz.mean(0)); pc = tra.transform_points(xyz, Tc)
    torch.cuda.synchronize(); t1 = time.time()
    try:
        grasps, conf = GraspGenSampler.run_inference(pc, sampler, grasp_threshold=-1.0, num_grasps=2000, topk_num_grasps=500, remove_outliers=False)
    except Exception as e:
        print("FAILED", key, repr(e)); continue
    torch.cuda.synchronize(); dt = time.time() - t1; times.append(dt)
    grasps = grasps.cpu().numpy() if len(grasps) else np.zeros((0, 4, 4), np.float32); conf = conf.cpu().numpy() if len(conf) else np.zeros((0,), np.float32)
    grasps = np.array([tra.inverse_matrix(Tc) @ g for g in grasps]) if len(grasps) else grasps
    np.savez(out, T=grasps, conf=conf, t_inference=dt)
    if i % 20 == 0:
        print(f"[{i}/{len(jobs)}] {key}: {len(grasps)} grasps, {dt*1000:.0f} ms", flush=True)
print(f"GRASPGEN DONE: {len(times)} models, mean inference {np.mean(times)*1000:.0f} ms, total {np.sum(times):.0f} s")
