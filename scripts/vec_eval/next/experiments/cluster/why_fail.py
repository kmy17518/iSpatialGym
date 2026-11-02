"""Per failing alias (round dir): slot verdicts and reasons, divergence onset of every tracked object vs the N=1 reference, AG weld timeline."""
import glob, json, os, re, sys
import numpy as np
C = "/vision/u/minyeong/projects/isg_home/vec_eval85/campaigns/vec_next"
rnd = sys.argv[1] if len(sys.argv) > 1 else "runs"
tasks = sys.argv[2:] or ["configuration_matching-attachment-cap-lemonade_bottle-breakfast_table", "configuration_matching-attachment-lid-teapot-bar",
                         "inter_object_relation_satisfaction-inside-brussels_sprouts-bowl-breakfast_table", "inter_object_relation_satisfaction-inside-sunflower-vase-coffee_table",
                         "inter_object_relation_satisfaction-nextto-tablespoon-chopstick-breakfast_table", "inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table"]
def quat_angle(q1, q2):
    d = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0, 1); return 2 * np.arccos(d)
def load(d):
    m = json.load(open(os.path.join(d, "meta.json"))); t = np.load(os.path.join(d, "trajectory.npz")); return m, t
def onset(x, y, thr):
    n = min(len(x), len(y)); d = np.linalg.norm(x[:n, :3] - y[:n, :3], axis=1); i = np.argmax(d > thr) if np.any(d > thr) else None
    return (int(i) if i is not None else None), float(d.max()), int(np.argmax(d))
def ag_events(log):
    ev = []
    for line in open(log, errors="replace"):
        if "ISG_AG_SHARED_CONTACT_VIEW built" in line:
            m = re.search(r"t=([0-9.]+),.*reads/hits/empty_fresh (\d+)/(\d+)/(\d+)", line)
            if m: ev.append((float(m.group(1)), int(m.group(2)), int(m.group(3))))
    return [e for e in ev if e[0] > 1.0]  # rebuilds during the episode (the build-time ones are at t < 1 s)
for task in tasks:
    R = f"{C}/{rnd}/{task}"
    if not os.path.isdir(R): print(f"## {task}: no {rnd} dir"); continue
    ref_dirs = [d for d in glob.glob(f"{R}/n1/*") if os.path.isdir(d) and not d.endswith("_batches")]
    if not ref_dirs: print(f"## {task}: no n1"); continue
    rm, rt = load(ref_dirs[0])
    print(f"\n## {task}  (ref: success={rm['success']} geo_step={rm.get('geometric_success_step')} goal_first={rm.get('goal_first_step')} n={rm['n_steps']} budget_cap={(rm.get('budget') or {}).get('episode_cap')} settle_min={(rm.get('pose_settled_config') or {}).get('settle_min_steps') if isinstance(rm.get('pose_settled_config'), dict) else rm.get('pose_settled_config')})")
    objs = [k for k in rt.files if k.startswith("object_pose__")]
    for d in sorted(glob.glob(f"{R}/n8same/*.slot*")):
        m, t = load(d); slot = d.rsplit("slot", 1)[1]
        verd = (m.get("monitors") or {}).get("stats", {}).get("verdicts", [])
        msg = verd[0]["message"][:110] if verd else ""
        line = f"  slot {slot}: success={m['success']} reason={m.get('qa_reason')} geo_step={m.get('geometric_success_step')} goal_first={m.get('goal_first_step')} n={m['n_steps']} | {msg}"
        print(line)
        devs = []
        for k in objs:
            o5, mx, imx = onset(t[k], rt[k], 0.005); o2, _, _ = onset(t[k], rt[k], 0.02)
            devs.append((k.replace("object_pose__", ""), o5, o2, mx, imx))
        devs.sort(key=lambda z: -z[3])
        print("      objects (first step >5 mm, >2 cm, max dev m @step): " + "; ".join(f"{n}: {o5}, {o2}, {mx:.3f}@{imx}" for n, o5, o2, mx, imx in devs[:3]))
        oe, mxe, ie = onset(t["eef_pose_right"], rt["eef_pose_right"], 0.005); ol, mxl, il = onset(t["eef_pose_left"], rt["eef_pose_left"], 0.005)
        jd = np.abs(t["joint_positions"][:min(len(t["step"]), len(rt["step"]))] - rt["joint_positions"][:min(len(t["step"]), len(rt["step"]))]).max()
        print(f"      eef right first>5mm {oe} (max {mxe:.3f}@{ie}), left {ol} (max {mxl:.3f}@{il}); joints max dev {jd:.4f}")
    ev8 = ag_events(f"{R}/n8same.log"); ev1 = ag_events(f"{R}/n1.log")
    print(f"  AG weld-triggered view rebuilds (t, cache reads, hits): n1 {ev1[:8]}  | n8same {ev8[:16]}")
