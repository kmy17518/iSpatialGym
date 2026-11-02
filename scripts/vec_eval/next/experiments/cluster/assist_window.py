"""Would the sleep assist (net motion < POS mm / ROT mrad over W steps, robot away) have fired after the release? Per slot: first qualifying window end."""
import numpy as np, sys, glob, os
root, W, POS, ROT = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) / 1000, float(sys.argv[4]) / 1000
def ang(q0, q1): return 2 * np.arccos(np.clip(np.abs(np.sum(q0 * q1, -1)), 0, 1))
for d in sorted(glob.glob(os.path.join(root, "*.slot*"))):
    z = np.load(d + "/trajectory.npz", allow_pickle=True)
    T = len(z["step"]); out = []
    for name in ("teapot.n.01_1", "lid.n.02_1"):
        X = z[f"object_pose__{name}"]; p, q = X[:, :3], X[:, 3:7]
        eefR = z["eef_pose_right"][:, :3]; eefL = z["eef_pose_left"][:, :3]
        first = None
        for t in range(W, T):
            w = slice(t - W, t)
            dp = np.linalg.norm(p[w] - p[t - W], axis=1).max(); dq = ang(q[w], np.broadcast_to(q[t - W], q[w].shape)).max()
            far = min(np.linalg.norm(eefR[t] - p[t]), np.linalg.norm(eefL[t] - p[t])) > 0.12
            if t > 770 and dp < POS and dq < ROT and far:
                first = t; break
        # how big is the rocking right after the release (steps 780-810) and at the end?
        w1 = slice(780, 810); w2 = slice(T - 30, T)
        rock1 = ang(q[w1][:-1], q[w1][1:]).max() * 1e3; rock2 = ang(q[w2][:-1], q[w2][1:]).max() * 1e3
        out.append(f"{name.split('.')[0]}: first qualifying window end={first} | per-step rot max 780-810={rock1:.2f} mrad, last30={rock2:.2f} mrad")
    print(f"{os.path.basename(d)[-5:]} settled_end={bool(z['eval__settled'][-1])} | " + " | ".join(out))
