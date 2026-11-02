import numpy as np, sys, glob, os, re
root, obj, t0, t1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
def ang(q0, q1): return 2 * np.arccos(np.clip(np.abs(np.sum(q0 * q1, -1)), 0, 1))
log = root + ".log"
sleeps = {}
for m in re.finditer(r"SLEEP_ASSIST s(\d) step (\d+): ([a-z_]+)", open(log).read()):
    sleeps.setdefault(int(m.group(1)), []).append(f"{m.group(3)[:6]}@{m.group(2)}")
for d in sorted(glob.glob(os.path.join(root, "*.slot*"))):
    z = np.load(d + "/trajectory.npz", allow_pickle=True); s = int(d[-1])
    X = z[f"object_pose__{obj}"]; q = X[:, 3:7]; p = X[:, :3]
    dq = ang(q[t0:t1 - 1], q[t0 + 1:t1]) * 1e3
    big = [t0 + i for i, v in enumerate(dq) if v > 5]
    net = ang(q[t0:t0 + 1], q[t1 - 1:t1])[0] * 1e3
    eef = min(np.linalg.norm(z["eef_pose_right"][t0:t1, :3] - p[t0:t1], axis=1).min(), np.linalg.norm(z["eef_pose_left"][t0:t1, :3] - p[t0:t1], axis=1).min())
    print(f"slot{s}: {obj.split('.')[0]} rot {t0}-{t1}: net {net:.0f} mrad, per-step max {dq.max():.1f} mrad, steps>5mrad {big[:6]}{'...' if len(big)>6 else ''}; min eef dist {eef*100:.0f} cm | sleeps {sleeps.get(s, [])}")
