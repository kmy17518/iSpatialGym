"""Are the resting objects asleep? A sleeping PhysX body has a bit-identical pose from step to step; count exactly-zero pose deltas over the final window."""
import glob, json, os
import numpy as np
C = "/vision/u/minyeong/projects/isg_home/vec_eval85/campaigns/vec_next"
pairs = [("configuration_matching-attachment-lid-teapot-bar", "lid.n.02_1"), ("inter_object_relation_satisfaction-inside-brussels_sprouts-bowl-breakfast_table", "brussels_sprouts.n.01_1"),
         ("inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table", "pack__of__kielbasa.n.01_1"), ("inter_object_relation_satisfaction-inside-sunflower-vase-coffee_table", "sunflower.n.01_1"),
         ("inter_object_relation_satisfaction-nextto-salt_shaker-pepper_shaker-breakfast_table", "salt_shaker.n.01_1"), ("inter_object_relation_satisfaction-under_small_scale-plate-place_mat-breakfast_table", "plate.n.04_1")]
for task, obj in pairs:
    R = f"{C}/runs/{task}"; print(f"\n## {obj}")
    for d in sorted(glob.glob(f"{R}/n8same/*.slot*")):
        m = json.load(open(f"{d}/meta.json")); t = np.load(f"{d}/trajectory.npz"); key = f"object_pose__{obj}"
        n = m["n_steps"]; w = m["pose_settled_config"]["n_steps"]; P = t[key][n - w:]
        dp = np.abs(np.diff(P, axis=0)).max(axis=1)
        frozen = float(np.mean(dp == 0.0)); longest = 0; cur = 0
        for z in (dp == 0.0):
            cur = cur + 1 if z else 0; longest = max(longest, cur)
        print(f"  slot {d[-1]} off={[round(v) for v in m['vec']['scene_offset'][:2]]} {'PASS' if m['success'] else 'FAIL'}: steps with bit-identical pose {frozen*100:5.1f}% of {w}, longest frozen run {longest:3d}, p50 |dpos| {np.median(np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1))*1e6:6.1f} um")
