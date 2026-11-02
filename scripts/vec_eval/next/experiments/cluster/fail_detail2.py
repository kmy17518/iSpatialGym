import json, glob, sys, numpy as np
C = sys.argv[1]
def ang(q0, q1): return 2 * np.arccos(np.clip(np.abs(np.sum(q0 * q1, -1)), 0, 1))
a = "inter_object_relation_satisfaction-inside-sunflower-vase-coffee_table"
Z = [np.load(d + "/trajectory.npz", allow_pickle=True) for d in sorted(glob.glob(f"{C}/runs_D/{a}/n8same/*.slot*"))]
z1 = np.load(glob.glob(f"{C}/runs_D/{a}/n1/*template*")[0] + "/trajectory.npz", allow_pickle=True)
T = min(len(z1["step"]), min(len(z["step"]) for z in Z))
k = "object_pose__sunflower.n.01_1"
print("SUNFLOWER slot0 vs n1: rot (mrad) / eefR pos diff (mm) / sunflower pos diff (mm) at steps:")
for t in [100, 300, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, T-1]:
    r = ang(Z[0][k][t, 3:7], z1[k][t, 3:7]) * 1e3; e = np.linalg.norm(Z[0]["eef_pose_right"][t, :3] - z1["eef_pose_right"][t, :3]) * 1e3
    p = np.linalg.norm(Z[0][k][t, :3] - z1[k][t, :3]) * 1e3; j = np.abs(Z[0]["joint_positions"][t] - z1["joint_positions"][t]).max()
    print(f"  {t}: rot {r:6.1f} | eefR {e:5.2f} | pos {p:5.1f} | max joint diff {j*1e3:.2f} mrad")
# where is the sunflower relative to the vase at the end, per slot: height of the sunflower origin above the vase origin and the tilt of its axis from vertical
def tilt(q):  # angle between the object's local z axis and world z (xyzw)
    x, y, z, w = q; zz = 1 - 2 * (x * x + y * y); return float(np.degrees(np.arccos(np.clip(zz, -1, 1))))
print("SUNFLOWER tilt of the local z axis from vertical at the end (deg): n1", round(tilt(z1[k][T-1, 3:7]), 1), "slots", [round(tilt(z[k][T-1, 3:7]), 1) for z in Z])
print("  at step 450 (held):", round(tilt(z1[k][450, 3:7]), 1), [round(tilt(z[k][450, 3:7]), 1) for z in Z], " at 600:", round(tilt(z1[k][600, 3:7]), 1), [round(tilt(z[k][600, 3:7]), 1) for z in Z])
a = "inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table"
Z = [np.load(d + "/trajectory.npz", allow_pickle=True) for d in sorted(glob.glob(f"{C}/runs_D/{a}/n8same/*.slot*"))]
z1 = np.load(glob.glob(f"{C}/runs_D/{a}/n1/*template*")[0] + "/trajectory.npz", allow_pickle=True); k = "object_pose__pack__of__kielbasa.n.01_1"
for t in (700, 745, 770, 830, 949):
    print(f"KIELBASA pack rot vs n1 at {t} (mrad):", [round(float(ang(z[k][t, 3:7], z1[k][t, 3:7]) * 1e3)) for z in Z], " pos (mm):", [round(float(np.linalg.norm(z[k][t, :3] - z1[k][t, :3]) * 1e3), 1) for z in Z])
# yaw of the pack at the end vs n1 (rotation about vertical)
def yaw(q):
    x, y, z, w = q; return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
print("KIELBASA pack yaw at the end (deg): n1", round(yaw(z1[k][949, 3:7]), 1), "slots", [round(yaw(z[k][949, 3:7]), 1) for z in Z])
print("KIELBASA settled flag at the last step per slot:", [int(z["eval__settled"][-1]) for z in Z], "n1", int(z1["eval__settled"][-1]))
