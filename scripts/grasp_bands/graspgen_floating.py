"""GraspGen proposals for the floating entries of a grasp-library v2 manifest (i7-floating 2026-11-07; port of the legacy
``runlogs/grasp_library_20260905/scripts/graspgen_batch.py``). Runs in the ``graspgen`` conda env (GraspGen 2dd8852, the Franka Panda
checkpoint - the legacy run's env and checkpoint), not in the Isaac env: it imports only numpy / torch / trimesh / grasp_gen.

Per floating job (``constraint == "floating"``) of ``--jobs``: the collision geometry the v2 export stage wrote (``<root>/models/<id>/
geometry.json``: the PhysX collision pieces in the base_link frame at the applied scale - no visual asset is decoded or written), 2000
surface points of their union centred on their mean, GraspGen's diffusion planner (2000 grasps, ``grasp_threshold`` -1, the top 500 by the
discriminator, no outlier removal - the legacy settings) -> ``graspgen.npz`` (``T`` (N, 4, 4) Panda base poses in the OBJECT frame,
``conf`` (N,)) and ``graspgen.json`` (settings, seed, timings, checkpoint). Seeded per entry (numpy + torch, ``--seed`` + the entry id's
digest), so a rerun reproduces the proposals on the same GPU / library versions.

    source /vision/u/minyeong/projects/graspgen/env_graspgen.sh
    python scripts/grasp_bands/graspgen_floating.py --jobs M.json --root <glib root> [--shard i/n] [--force]

Re-seed (owner 2026-11-07 item 4.1, entries with few rows): ``--n-points`` (surface points), ``--n-seeds`` (inference runs, each its own
seed and point sample; the proposals are concatenated), ``--num-grasps`` / ``--topk`` (generated / kept per run). The defaults are the
legacy settings (2000, 1, 2000, 500).
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

GRASPGEN_ROOT = os.environ.get("GRASPGEN_ROOT", "/vision/u/minyeong/projects/graspgen")
CHECKPOINT = os.path.join(GRASPGEN_ROOT, "GraspGenModels", "checkpoints", "graspgen_franka_panda.yml")
SETTINGS = {"n_points": 2000, "num_grasps": 2000, "topk_num_grasps": 500, "grasp_threshold": -1.0, "remove_outliers": False}


def entry_dir(root, job):
    return os.path.join(root, "models", job["id"])


def points_of(geometry_path, n, rng_seed):
    import trimesh

    with open(geometry_path) as f:
        d = json.load(f)
    pieces = [trimesh.Trimesh(np.asarray(p["vertices"], dtype=np.float64), np.asarray(p["faces"], dtype=np.int64), process=False) for p in d["pieces"]]
    union = trimesh.util.concatenate(pieces)
    xyz, _ = trimesh.sample.sample_surface(union, n, seed=rng_seed)
    return np.asarray(xyz, np.float32), d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--root", required=True, help="the library root (models/<id>/geometry.json)")
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-points", type=int, default=SETTINGS["n_points"])
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--num-grasps", type=int, default=SETTINGS["num_grasps"])
    ap.add_argument("--topk", type=int, default=SETTINGS["topk_num_grasps"])
    a = ap.parse_args(argv)
    with open(a.jobs) as f:
        jobs = [j for j in json.load(f) if (j.get("constraint") or "tabletop") == "floating"]
    if a.shard:
        i, n = (int(v) for v in a.shard.split("/"))
        jobs = [j for k, j in enumerate(jobs) if k % n == i]
    import torch

    sys.path.insert(0, os.path.join(GRASPGEN_ROOT, "GraspGen"))
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

    commit = None
    try:
        import subprocess

        commit = subprocess.check_output(["git", "-C", os.path.join(GRASPGEN_ROOT, "GraspGen"), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    sampler = GraspGenSampler(load_grasp_cfg(CHECKPOINT))
    t_load = time.time() - t0
    print(f"GG model load {t_load:.1f}s gpu {torch.cuda.get_device_name(0)} jobs {len(jobs)}", flush=True)
    for j in jobs:
        d = entry_dir(a.root, j)
        out, meta_p = os.path.join(d, "graspgen.npz"), os.path.join(d, "graspgen.json")
        rec = {"id": j["id"]}
        try:
            if os.path.exists(out) and not a.force:
                rec.update(ok=True, skipped=True)
                print("GG " + json.dumps(rec), flush=True)
                continue
            geo = os.path.join(d, "geometry.json")
            if not os.path.exists(geo):
                raise FileNotFoundError(f"{geo} (run the export stage first)")
            settings = {**SETTINGS, "n_points": a.n_points, "num_grasps": a.num_grasps, "topk_num_grasps": a.topk, "n_seeds": a.n_seeds}
            seed0 = (a.seed + int(hashlib.sha1(j["id"].encode()).hexdigest()[:8], 16)) % (2 ** 31)
            Gs, Cs, centres, dt = [], [], [], 0.0
            for r in range(a.n_seeds):
                seed = (seed0 + r) % (2 ** 31)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                xyz, gd = points_of(geo, a.n_points, seed)
                centre = xyz.mean(0)
                pc = xyz - centre
                torch.cuda.synchronize()
                t1 = time.time()
                grasps, conf = GraspGenSampler.run_inference(pc, sampler, grasp_threshold=SETTINGS["grasp_threshold"], num_grasps=a.num_grasps,
                                                             topk_num_grasps=a.topk, remove_outliers=SETTINGS["remove_outliers"])
                torch.cuda.synchronize()
                dt += time.time() - t1
                G1 = grasps.cpu().numpy().astype(np.float64) if len(grasps) else np.zeros((0, 4, 4))
                C1 = conf.cpu().numpy().astype(np.float64) if len(conf) else np.zeros((0,))
                G1[:, :3, 3] += centre[None].astype(np.float64)  # back to the object frame (the point cloud was centred on its mean)
                Gs.append(G1)
                Cs.append(C1)
                centres.append([round(float(v), 6) for v in centre])
            G, C = np.concatenate(Gs), np.concatenate(Cs)
            seed, centre = seed0, np.asarray(centres[0])
            np.savez(out, T=G, conf=C)
            meta = {"schema": "isg-graspgen-floating/1", "id": j["id"], "checkpoint": CHECKPOINT, "graspgen_commit": commit, "settings": settings,
                    "seed": seed, "seeds": [(seed0 + r) % (2 ** 31) for r in range(a.n_seeds)], "centres": centres, "n": int(len(G)),
                    "conf": {"max": round(float(C.max()), 5) if len(C) else None, "min": round(float(C.min()), 5) if len(C) else None},
                    "geometry": {"format": gd.get("format"), "n_pieces": len(gd.get("pieces", [])), "scale": gd.get("scale")},
                    "frames": "T = Panda base (GraspGen: approach +z, closing x) in the object base_link frame", "t_inference_s": round(dt, 3),
                    "t_model_load_s": round(t_load, 1), "gpu": torch.cuda.get_device_name(0), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with open(meta_p, "w") as f:
                json.dump(meta, f, indent=1)
            rec.update(ok=True, n=int(len(G)), conf_max=meta["conf"]["max"], s=round(dt, 3))
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, error=repr(e))
        print("GG " + json.dumps(rec), flush=True)
    print("GB_GRASPGEN_DONE", flush=True)


if __name__ == "__main__":
    main()
