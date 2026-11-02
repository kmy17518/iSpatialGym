import glob, json, sys
import numpy as np
C = "/vision/u/minyeong/projects/isg_home/vec_eval85/campaigns/vec_next"
T = sys.argv[1] if len(sys.argv) > 1 else "inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table"
obj = sys.argv[2] if len(sys.argv) > 2 else "pack__of__kielbasa.n.01_1"
def qang(q1, q2):
    d = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0, 1); return 2 * np.arccos(d)
for rnd in ("runs_B", "runs_C"):
    for d in sorted(glob.glob(f"{C}/{rnd}/{T}/n1/*/") + glob.glob(f"{C}/{rnd}/{T}/n8same/*.slot*/")):
        if "_batches" in d: continue
        m = json.load(open(d + "meta.json")); t = np.load(d + "trajectory.npz"); P = t[f"object_pose__{obj}"]; n = m["n_steps"]; w = m["pose_settled_config"]["n_steps"]
        g = int(m.get("goal_first_step") or 0)
        dp = np.abs(np.diff(P, axis=0)).max(axis=1); frozen_from = n
        for i in range(n - 2, -1, -1):
            if dp[i] == 0.0: frozen_from = i + 1
            else: break
        pos = P[n - w:, :3]; quat = P[n - w:, 3:]
        wstep = qang(P[g + 1:, 3:], P[g:-1, 3:]) * 30.0  # rad/s from consecutive quaternions after the goal
        vstep = np.linalg.norm(np.diff(P[g:, :3], axis=0), axis=1) * 30.0
        label = "n1   " if "/n1/" in d else "slot" + d.rstrip("/")[-1]
        print(f"{rnd} {label} {'PASS' if m['success'] else 'FAIL'} host={m.get('hostname','?')[:8]} | last-window drift ori {qang(quat, quat[0]).max()*1000:6.2f} mrad pos {np.linalg.norm(pos-pos[0],axis=1).max()*1000:5.2f} mm | asleep from {frozen_from} ({n-frozen_from}) | after goal: omega p50/p90/max {np.median(wstep)*1000:6.1f}/{np.percentile(wstep,90)*1000:6.1f}/{wstep.max()*1000:7.1f} mrad/s, v p90 {np.percentile(vstep,90)*1000:5.1f} mm/s")
