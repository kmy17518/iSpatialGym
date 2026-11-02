"""Float32 hypothesis: the resting object's orientation drift over the final settle window vs the ULP of its world coordinates."""
import glob, json, math, os
import numpy as np
C = "/vision/u/minyeong/projects/isg_home/vec_eval85/campaigns/vec_next"
pairs = [("configuration_matching-attachment-lid-teapot-bar", "lid.n.02_1"), ("inter_object_relation_satisfaction-inside-brussels_sprouts-bowl-breakfast_table", "brussels_sprouts.n.01_1"),
         ("inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table", "pack__of__kielbasa.n.01_1"), ("inter_object_relation_satisfaction-inside-sunflower-vase-coffee_table", "sunflower.n.01_1"),
         ("configuration_matching-attachment-cap-lemonade_bottle-breakfast_table", "cap.n.02_1"), ("inter_object_relation_satisfaction-under_small_scale-teacup-saucer-breakfast_table", "teacup.n.02_1"),
         ("inter_object_relation_satisfaction-nextto-salt_shaker-pepper_shaker-breakfast_table", "salt_shaker.n.01_1")]
def ulp(v): return float(np.spacing(np.float32(abs(v))))
def qang(q1, q2):
    d = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0, 1); return 2 * np.arccos(d)
rows = []
for task, obj in pairs:
    R = f"{C}/runs/{task}"
    if not os.path.isdir(R): continue
    print(f"\n## {task}")
    for d in sorted(glob.glob(f"{R}/n8same/*.slot*")):
        m = json.load(open(f"{d}/meta.json")); t = np.load(f"{d}/trajectory.npz")
        key = f"object_pose__{obj}"
        if key not in t.files: key = [k for k in t.files if k.startswith("object_pose__")][0]
        n = m["n_steps"]; w = m["pose_settled_config"]["n_steps"]
        pos = t[key][n - w:, :3]; quat = t[key][n - w:, 3:]
        drift_ori = qang(quat, quat[0]).max() * 1000
        x, y, z = [float(v) for v in pos[-1]]
        u = ulp(x) + ulp(y)
        print(f"  slot {d[-1]} success={str(m['success']):5} world x={x:7.2f} y={y:7.2f} | ulp(x)={ulp(x)*1e6:4.1f} um ulp(y)={ulp(y)*1e6:4.1f} um | ori drift {drift_ori:6.2f} mrad")
        rows.append((u * 1e6, drift_ori, m["success"]))
rows = np.array(rows, dtype=float)
print("\n== over all slots of these 7 tasks: Spearman rank correlation of (ulp(x)+ulp(y)) vs ori drift:")
from scipy.stats import spearmanr
print(spearmanr(rows[:, 0], rows[:, 1]))
for lo, hi in ((0, 5), (5, 9), (9, 13), (13, 20)):
    sel = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
    if sel.any(): print(f"  ulp sum {lo}-{hi} um: n={int(sel.sum())} median drift {np.median(rows[sel,1]):.2f} mrad, fail rate {1-rows[sel,2].mean():.2f}")
