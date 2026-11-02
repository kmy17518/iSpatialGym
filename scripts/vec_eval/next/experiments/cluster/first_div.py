import glob, sys, numpy as np
C = sys.argv[1]
def ld(p): z = np.load(p, allow_pickle=True); return {k: z[k] for k in z.files}
def ang(q0, q1): return 2 * np.arccos(np.clip(np.abs(np.sum(q0 * q1, -1)), 0, 1))
for a in ["inter_object_relation_satisfaction-nextto-tablespoon-chopstick-breakfast_table",
          "inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table"][:1]:
    Z = [ld(d + "/trajectory.npz") for d in sorted(glob.glob(f"{C}/runs_D/{a}/n8same/*.slot*"))]
    z1 = ld(glob.glob(f"{C}/runs_D/{a}/n1/*template*")[0] + "/trajectory.npz")
    zb = ld(glob.glob(f"{C}/runs_B/{a}/n1/*template*")[0] + "/trajectory.npz")
    T = min(len(z1["step"]), len(zb["step"]), *[len(z["step"]) for z in Z])
    objs = [k for k in z1 if k.startswith("object_pose__") and "breakfast_table" not in k]
    print("==", a, "T", T)
    for name, z in [("B-n1", zb)] + [(f"s{i}", z) for i, z in enumerate(Z)]:
        parts = []
        for k in objs + ["joint_positions"]:
            if k == "joint_positions":
                d = np.abs(z[k][:T] - z1[k][:T]).max(1) * 1e3
            else:
                d = np.maximum(np.linalg.norm(z[k][:T, :3] - z1[k][:T, :3], axis=1) * 1e3, ang(z[k][:T, 3:7], z1[k][:T, 3:7]) * 1e3)
            f = [int(np.argmax(d > th)) if (d > th).any() else None for th in (1e-3, 0.1, 1, 10)]
            parts.append(f"{k.replace('object_pose__','')[:10]}: " + " ".join(f"{th}@{x}" for th, x in zip(("1e-3", ".1", "1", "10"), f)))
        print(f"  {name:5s}", " | ".join(parts))
    ph = z1["phase"]; print("  n1 phase changes:", [(t, str(ph[t])) for t in range(T) if t == 0 or ph[t] != ph[t - 1]][:30])
    for k in objs + ["joint_positions"]:
        if k == "joint_positions":
            d0 = [np.abs(z[k][0] - z1[k][0]).max() * 1e3 for z in Z]
        else:
            d0 = [max(np.linalg.norm(z[k][0, :3] - z1[k][0, :3]) * 1e3, ang(z[k][0, 3:7], z1[k][0, 3:7]) * 1e3) for z in Z]
        print("  step-0 diff", k[:28], [f"{x:.4f}" for x in d0])
    k = [x for x in objs if "tablespoon" in x][0]
    for t in (540, 545, 548, 550, 552, 555, 560, 570, 600, 700, T - 1):
        print(f"  spoon step {t}: rot vs n1 mrad", [round(float(ang(z[k][t, 3:7], z1[k][t, 3:7]) * 1e3), 2) for z in Z], " B-n1", round(float(ang(zb[k][t, 3:7], z1[k][t, 3:7]) * 1e3), 2))
