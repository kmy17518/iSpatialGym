import glob, json, os
import numpy as np
from scipy.stats import spearmanr
C = "/vision/u/minyeong/projects/isg_home/vec_eval85/campaigns/vec_next"
pairs = [("configuration_matching-attachment-lid-teapot-bar", "lid.n.02_1"), ("inter_object_relation_satisfaction-inside-brussels_sprouts-bowl-breakfast_table", "brussels_sprouts.n.01_1"),
         ("inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table", "pack__of__kielbasa.n.01_1"), ("inter_object_relation_satisfaction-inside-sunflower-vase-coffee_table", "sunflower.n.01_1"),
         ("configuration_matching-attachment-cap-lemonade_bottle-breakfast_table", "cap.n.02_1"), ("inter_object_relation_satisfaction-under_small_scale-teacup-saucer-breakfast_table", "teacup.n.02_1"),
         ("inter_object_relation_satisfaction-nextto-salt_shaker-pepper_shaker-breakfast_table", "salt_shaker.n.01_1"), ("inter_object_relation_satisfaction-under_small_scale-plate-place_mat-breakfast_table", "plate.n.04_1"),
         ("inter_object_relation_satisfaction-nextto-tablespoon-chopstick-breakfast_table", "tablespoon.n.02_1")]
def ulp(v): return float(np.spacing(np.float32(abs(v)))) * 1e6
def qang(q1, q2):
    d = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0, 1); return 2 * np.arccos(d)
rows = []
for task, obj in pairs:
    R = f"{C}/runs/{task}"
    if not os.path.isdir(R): continue
    print(f"\n## {task.split('-')[-3] if 'inter_object' in task else task.split('-')[-2]}  ({obj})")
    for d in sorted(glob.glob(f"{R}/n8same/*.slot*")):
        m = json.load(open(f"{d}/meta.json")); t = np.load(f"{d}/trajectory.npz"); off = m["vec"]["scene_offset"]
        key = f"object_pose__{obj}"
        if key not in t.files: key = [k for k in t.files if k.startswith("object_pose__")][0]
        n = m["n_steps"]; w = m["pose_settled_config"]["n_steps"]; pos = t[key][n - w:, :3]; quat = t[key][n - w:, 3:]
        drift = qang(quat, quat[0]).max() * 1000
        x, y = float(pos[-1, 0] + off[0]), float(pos[-1, 1] + off[1]); r = (x * x + y * y) ** 0.5
        print(f"  slot {d[-1]} {'PASS' if m['success'] else 'FAIL'} world x={x:7.1f} y={y:7.1f} |r|={r:5.1f} ulp x/y={ulp(x):3.1f}/{ulp(y):3.1f} um | ori drift {drift:6.2f} mrad")
        rows.append((abs(x), abs(y), r, ulp(x) + ulp(y), drift, m["success"]))
rows = np.array(rows, dtype=float)
print("\nSpearman: |r| vs drift", spearmanr(rows[:, 2], rows[:, 4])[0].round(3), "| ulp(x)+ulp(y) vs drift", spearmanr(rows[:, 3], rows[:, 4])[0].round(3), "| max(|x|,|y|) vs drift", spearmanr(np.maximum(rows[:, 0], rows[:, 1]), rows[:, 4])[0].round(3))
for lo, hi in ((0, 20), (20, 55), (55, 64), (64, 70), (70, 100)):
    sel = (np.maximum(rows[:, 0], rows[:, 1]) >= lo) & (np.maximum(rows[:, 0], rows[:, 1]) < hi)
    if sel.any(): print(f"  max(|x|,|y|) in [{lo},{hi}) m: n={int(sel.sum())} median drift {np.median(rows[sel,4]):.2f} mrad, p90 {np.percentile(rows[sel,4],90):.2f}, fail rate {1-rows[sel,5].mean():.2f}")
