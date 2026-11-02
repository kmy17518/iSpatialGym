import glob, json, sys
import numpy as np
root = sys.argv[1]; obj = sys.argv[2]
def qang(q1, q2):
    d = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0, 1); return 2 * np.arccos(d)
for run in sorted(glob.glob(f"{root}/*/")):
    name = run.rstrip("/").split("/")[-1]
    for d in sorted(glob.glob(f"{run}/*.slot*/")):
        m = json.load(open(d + "meta.json")); t = np.load(d + "trajectory.npz"); P = t[f"object_pose__{obj}"]; n = m["n_steps"]; w = m["pose_settled_config"]["n_steps"]; g = int(m.get("goal_first_step") or 0)
        dp = np.abs(np.diff(P, axis=0)).max(axis=1); frozen_from = n
        for i in range(n - 2, -1, -1):
            if dp[i] == 0.0: frozen_from = i + 1
            else: break
        pos = P[n - w:, :3]; quat = P[n - w:, 3:]; wstep = qang(P[g + 1:, 3:], P[g:-1, 3:]) * 30.0
        print(f"{name:8} slot{d.rstrip('/')[-1]} off={[round(v) for v in m['vec']['scene_offset']]} {'PASS' if m['success'] else 'FAIL'} drift ori {qang(quat, quat[0]).max()*1000:6.2f} mrad | asleep from {frozen_from} ({n-frozen_from}) | omega p90 after goal {np.percentile(wstep,90)*1000:6.1f} mrad/s")
