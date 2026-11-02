import glob, json, os
import numpy as np
C = "/vision/u/minyeong/projects/isg_home/vec_eval85/campaigns/vec_next"
pairs = [("configuration_matching-attachment-lid-teapot-bar", "lid.n.02_1"), ("inter_object_relation_satisfaction-inside-brussels_sprouts-bowl-breakfast_table", "brussels_sprouts.n.01_1"),
         ("inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table", "pack__of__kielbasa.n.01_1"), ("inter_object_relation_satisfaction-inside-sunflower-vase-coffee_table", "sunflower.n.01_1"),
         ("configuration_matching-attachment-cap-lemonade_bottle-breakfast_table", "cap.n.02_1")]
def qang(q1, q2):
    d = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0, 1); return 2 * np.arccos(d)
for task, obj in pairs:
    R = f"{C}/runs/{task}"; print(f"\n## {task}")
    for d in sorted(glob.glob(f"{R}/n8same/*.slot*")):
        m = json.load(open(f"{d}/meta.json")); t = np.load(f"{d}/trajectory.npz")
        off = (m.get("vec") or {}).get("scene_offset"); g = int(m.get("goal_first_step") or 0); n = m["n_steps"]; w = m["pose_settled_config"]["n_steps"]
        pos = t[f"object_pose__{obj}"][n - w:, :3]; quat = t[f"object_pose__{obj}"][n - w:, 3:]
        drift_pos = np.linalg.norm(pos - pos[0], axis=1).max(); drift_ori = qang(quat, quat[0]).max()
        base = t["base_pose"][n - w:, :3]; bdrift = np.linalg.norm(base - base[0], axis=1).max()
        jdrift = np.abs(t["joint_positions"][n - w:] - t["joint_positions"][n - w]).max()
        errs = m.get("errors_final_step") or m.get("errors")
        print(f"  slot {d[-1]} offset={[round(v,1) for v in off] if off else None} success={m['success']}: last-window drift {obj}: pos {drift_pos*1000:.2f} mm (tol 2), ori {drift_ori*1000:.2f} mrad (tol 5); robot base {bdrift*1000:.2f} mm, joints {jdrift*1000:.2f} mrad (tol 5)"
              + (f" | errors_final_step: {json.dumps(errs)[:160]}" if (not m['success'] and errs) else ""))
