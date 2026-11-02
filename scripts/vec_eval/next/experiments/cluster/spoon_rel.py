import glob, numpy as np
def ld(p): z = np.load(p, allow_pickle=True); return {k: z[k] for k in z.files}
def ang(q0, q1): return 2 * np.arccos(np.clip(np.abs(np.sum(q0 * q1, -1)), 0, 1))
a = "inter_object_relation_satisfaction-nextto-tablespoon-chopstick-breakfast_table"
Z = [ld(d + "/trajectory.npz") for d in sorted(glob.glob(f"runs_D/{a}/n8same/*.slot*"))]
z1 = ld(glob.glob(f"runs_D/{a}/n1/*template*")[0] + "/trajectory.npz")
k = "object_pose__tablespoon.n.02_1"
for name, z in [("n1", z1), ("s0", Z[0]), ("s1", Z[1]), ("s2", Z[2])]:
    print(name, "spoon z(mm) / step rot(mrad) / step move(mm) / eefL-z, eefR-z; left/right gripper joints last 4")
    for t in range(544, 560):
        dz = z[k][t, 2] * 1e3; r = ang(z[k][t, 3:7], z[k][t - 1, 3:7]) * 1e3; m = np.linalg.norm(z[k][t, :3] - z[k][t - 1, :3]) * 1e3
        print(f"  {t}: z {dz:8.2f} rot/step {r:6.2f} move/step {m:6.2f} | eefL {z['eef_pose_left'][t,2]*1e3:7.1f} eefR {z['eef_pose_right'][t,2]*1e3:7.1f} | dist spoon-eefL {np.linalg.norm(z[k][t,:3]-z['eef_pose_left'][t,:3])*1e3:6.1f} eefR {np.linalg.norm(z[k][t,:3]-z['eef_pose_right'][t,:3])*1e3:6.1f}")
