import glob, sys, numpy as np
C = "/vision/u/minyeong/projects/isg_home/vec_eval85/campaigns/vec_next"
R, a = sys.argv[1], sys.argv[2]
def ld(p): z = np.load(p, allow_pickle=True); return {k: z[k] for k in z.files}
def ang(q0, q1): return 2 * np.arccos(np.clip(np.abs(np.sum(q0 * q1, -1)), 0, 1))
Z = [ld(d + "/trajectory.npz") for d in sorted(glob.glob(f"{C}/{R}/{a}/n8same/*.slot*"))]
z1 = ld(glob.glob(f"{C}/{R}/{a}/n1/*template*")[0] + "/trajectory.npz")
zd = ld(glob.glob(f"{C}/runs_D/{a}/n1/*template*")[0] + "/trajectory.npz")
T = min(len(z1["step"]), *[len(z["step"]) for z in Z])
keys = [k for k in z1 if k.startswith("object_pose__") or k == "joint_positions"]
def dev(z, t): return max(float(np.abs(z[k][t] - z1[k][t]).max()) for k in keys)
objs = [k for k in z1 if k.startswith("object_pose__") and not k.endswith("table.n.01_1")]
print(R, a, "T", T, "objects", [o.replace("object_pose__", "") for o in objs])
for name, z in [("D-n1", zd)] + [(f"s{i}", z) for i, z in enumerate(Z)]:
    TT = min(T, len(z["step"]))
    d = np.array([dev(z, t) for t in range(TT)])
    first = [int(np.argmax(d > th)) if (d > th).any() else None for th in (0, 1e-6, 1e-4, 1e-3, 1e-2)]
    fin = {o.replace("object_pose__", "")[:10]: round(float(ang(z[o][TT - 1, 3:7], z1[o][TT - 1, 3:7])) * 1e3, 1) for o in objs}
    print(f"  {name:5s} first step with any diff >0 / 1e-6 / 1e-4 / 1e-3 / 1e-2: {first} | final rot vs n1 (mrad) {fin}")
