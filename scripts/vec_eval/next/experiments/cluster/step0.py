import glob, numpy as np
def ld(p): z = np.load(p, allow_pickle=True); return {k: z[k] for k in z.files}
for a in ["inter_object_relation_satisfaction-nextto-tablespoon-chopstick-breakfast_table",
          "inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table"]:
    print("==", a.split("-")[2:4])
    R = {}
    for r in ["runs_B", "runs_C", "runs_D", "runs_D_sleepmode"]:
        S = [ld(d + "/trajectory.npz") for d in sorted(glob.glob(f"{r}/{a}/n8same/*.slot*"))]
        n1 = glob.glob(f"{r}/{a}/n1/*template*/trajectory.npz")
        R[r] = (S, ld(n1[0]) if n1 else None)
    keys = [k for k in R["runs_D"][0][0] if k.startswith("object_pose__") or k == "joint_positions"]
    def full(z, t): return np.concatenate([z[k][t].ravel() for k in keys])
    for r1, r2 in [("runs_D", "runs_D_sleepmode"), ("runs_C", "runs_D"), ("runs_B", "runs_C")]:
        S1, S2 = R[r1][0], R[r2][0]
        d = [float(np.abs(full(x, 0) - full(y, 0)).max()) for x, y in zip(S1, S2)]
        n = float(np.abs(full(R[r1][1], 0) - full(R[r2][1], 0)).max()) if R[r1][1] is not None and R[r2][1] is not None else None
        # first step where the two runs of each slot differ
        fs = []
        for x, y in zip(S1, S2):
            T = min(len(x["step"]), len(y["step"])); dd = np.array([np.abs(full(x, t) - full(y, t)).max() for t in range(T)])
            fs.append(int(np.argmax(dd > 0)) if (dd > 0).any() else None)
        print(f"  {r1} vs {r2}: step-0 max abs diff per slot {['%.1e' % v for v in d]} | n1 {n} | first differing step per slot {fs}")
    S, z1 = R["runs_D"]
    print("  runs_D slot vs own n1 at step 0:", ['%.1e' % float(np.abs(full(x, 0) - full(z1, 0)).max()) for x in S])
