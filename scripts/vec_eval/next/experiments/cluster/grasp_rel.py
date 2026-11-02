import glob, sys, numpy as np
from scipy.spatial.transform import Rotation as R
C = sys.argv[1]
def rel(z, k, eef, t):
    po, qo = z[k][t, :3], z[k][t, 3:7]; pe, qe = z[eef][t, :3], z[eef][t, 3:7]
    Re = R.from_quat(qe); return Re.inv().apply(po - pe), (Re.inv() * R.from_quat(qo))
for a, k, eef in [("configuration_matching-attachment-cap-lemonade_bottle-breakfast_table", None, "eef_pose_right"),
                  ("inter_object_relation_satisfaction-inside-sunflower-vase-coffee_table", "object_pose__sunflower.n.01_1", "eef_pose_right")]:
    Z = [np.load(d + "/trajectory.npz", allow_pickle=True) for d in sorted(glob.glob(f"{C}/runs_D/{a}/n8same/*.slot*"))]
    z1 = np.load(glob.glob(f"{C}/runs_D/{a}/n1/*template*")[0] + "/trajectory.npz", allow_pickle=True)
    if k is None:
        print([x for x in z1.files][:40]); k = [x for x in z1.files if x.startswith("object_pose__cap")][0]
    print("==", a, k, "keys with grasp:", [x for x in z1.files if "grasp" in x.lower() or "ag" == x[:2]])
    T = min(len(z1["step"]), *[len(z["step"]) for z in Z])
    # object world speed in n1 to locate the pickup
    for eefk in ("eef_pose_left", "eef_pose_right"):
        d = [np.linalg.norm(z1[k][t, :3] - z1[eefk][t, :3]) for t in range(T)]
        print(f"  n1 dist obj-{eefk} min {min(d)*100:.1f} cm at step {int(np.argmin(d))}")
    mv = np.linalg.norm(np.diff(z1[k][:T, :3], axis=0), axis=1) * 1e3
    first = int(np.argmax(mv > 0.5)); print("  n1 object first moves >0.5 mm/step at", first)
    for t in list(range(max(first - 6, 0), first + 14, 2)) + [first + 30, first + 60, first + 120]:
        if t >= T: break
        row = []
        for z in Z:
            p, q = rel(z, k, eef, t); p1, q1 = rel(z1, k, eef, t)
            row.append(f"{np.linalg.norm(p-p1)*1e3:5.2f}mm/{(q1.inv()*q).magnitude()*1e3:5.1f}")
        wd = max(np.linalg.norm(z[k][t, :3] - z1[k][t, :3]) * 1e3 for z in Z)
        print(f"  step {t:4d} rel-to-gripper vs n1 per slot: {' '.join(row)} | max world pos diff {wd:.2f} mm")
