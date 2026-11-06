"""Partial-scene validation (2026-11-01): the same instance collected in the cropped world (partial) and the full scene (full) -
outcome, length, and how far the robot's recorded trajectories drift apart (joint positions, right eef, base) per instance.
    python compare_pv.py [<pv root>]"""
import glob, json, os, sys
import h5py
import numpy as np
root = sys.argv[1] if len(sys.argv) > 1 else "/vision/u/minyeong/projects/isg_home/trainfast/out/pv"
SIDE_A = sys.argv[2] if len(sys.argv) > 2 else "partial"
SIDE_B = sys.argv[3] if len(sys.argv) > 3 else "full"

def load(d):
    m = json.load(open(os.path.join(d, "meta.json")))
    out = {"success": bool(m.get("success")), "reason": m.get("reason"), "steps": None}
    p = os.path.join(d, "demo.hdf5")
    if os.path.exists(p):
        with h5py.File(p, "r") as f:
            g = f["data/demo_0"]
            out.update(steps=len(g["action"]), q=g["isg/joint_positions"][:], eef=g["isg/eef_pose_right"][:, :3], base=g["isg/base_pose"][:, :3],
                       action=g["action"][:], crop=f["data"].attrs.get("isg_scene_crop"))
    return out

print(f"{SIDE_A} vs {SIDE_B}")
print(f"{'alias':42s} {'inst':7s} {SIDE_A:16s} {SIDE_B:16s} {'steps p/f':11s} {'action max':10s} {'joint max/mean (rad)':22s} {'eef max (mm)':12s} {'base max (mm)':12s}")
for key_dir in sorted(glob.glob(os.path.join(root, "*"))):
    key = os.path.basename(key_dir)
    for pd in sorted(glob.glob(os.path.join(key_dir, SIDE_A, "demo", "*"))):
        inst = os.path.basename(pd).rsplit("__", 1)[-1]
        fd = os.path.join(key_dir, SIDE_B, "demo", os.path.basename(pd))
        if not os.path.exists(os.path.join(pd, "meta.json")) or not os.path.exists(os.path.join(fd, "meta.json")):
            print(f"{key[:42]:42s} {inst:7s} (one side missing)"); continue
        a, b = load(pd), load(fd)
        fmt = lambda r: ("ok" if r["success"] else str(r["reason"]))[:16]
        if a.get("q") is None or b.get("q") is None:
            print(f"{key[:42]:42s} {inst:7s} {fmt(a):16s} {fmt(b):16s} (no hdf5)"); continue
        n = min(a["steps"], b["steps"])
        dq = np.abs(a["q"][:n] - b["q"][:n]); de = np.linalg.norm(a["eef"][:n] - b["eef"][:n], axis=1) * 1000
        db = np.linalg.norm(a["base"][:n] - b["base"][:n], axis=1) * 1000; da = np.abs(a["action"][:n] - b["action"][:n]).max()
        print(f"{key[:42]:42s} {inst:7s} {fmt(a):16s} {fmt(b):16s} {a['steps']:>4}/{b['steps']:<5} {da:10.4f} {dq.max():8.4f} / {dq.mean():8.5f}    {de.max():10.1f}   {db.max():10.1f}")
