"""Orientation / position drift of the resting object over the final settle window, per slot, for the placement experiments."""
import glob, json, os, sys
import numpy as np
def qang(q1, q2):
    d = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0, 1); return 2 * np.arccos(d)
root = sys.argv[1]; obj = sys.argv[2] if len(sys.argv) > 2 else None
for run in sorted(glob.glob(f"{root}/*/")):
    name = os.path.basename(run.rstrip("/"))
    eps = sorted(d for d in glob.glob(f"{run}/*.slot*") if os.path.isdir(d))
    if not eps: continue
    out = [name]
    for d in eps:
        m = json.load(open(f"{d}/meta.json")); t = np.load(f"{d}/trajectory.npz")
        key = f"object_pose__{obj}" if obj and f"object_pose__{obj}" in t.files else [k for k in t.files if k.startswith("object_pose__")][0]
        n = m["n_steps"]; w = m["pose_settled_config"]["n_steps"]; pos = t[key][n - w:, :3]; quat = t[key][n - w:, 3:]
        g = int(m.get("goal_first_step") or 0); speed = np.linalg.norm(np.diff(t[key][g:, :3], axis=0), axis=1) * 30
        out.append(f"slot{d[-1]} off={[round(v,1) for v in m['vec']['scene_offset']]} {'PASS' if m['success'] else 'FAIL'} ori {qang(quat, quat[0]).max()*1000:5.2f} mrad pos {np.linalg.norm(pos-pos[0],axis=1).max()*1000:4.2f} mm speed p90 {np.percentile(speed,90)*1000:5.2f} mm/s")
    print(" | ".join(out))
