import glob, json, os, sys
import numpy as np
C = "/vision/u/minyeong/projects/isg_home/vec_eval85/campaigns/vec_next"
pairs = [("configuration_matching-attachment-lid-teapot-bar", "lid.n.02_1"), ("inter_object_relation_satisfaction-inside-brussels_sprouts-bowl-breakfast_table", "brussels_sprouts.n.01_1"),
         ("inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table", "pack__of__kielbasa.n.01_1"), ("inter_object_relation_satisfaction-inside-sunflower-vase-coffee_table", "sunflower.n.01_1")]
def runs(d):
    m = json.load(open(os.path.join(d, "meta.json"))); t = np.load(os.path.join(d, "trajectory.npz")); return m, t
def longest_true(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0; best = max(best, cur)
    return best
for task, obj in pairs:
    R = f"{C}/runs/{task}"
    ref = [d for d in glob.glob(f"{R}/n1/*") if os.path.isdir(d) and not d.endswith("_batches")][0]
    rm, rt = runs(ref)
    print(f"\n## {task}")
    psc = rm.get("pose_settled_config"); print("  pose_settled_config:", json.dumps(psc)[:300]); print("  settle_steps:", rm.get("settle_steps"), "n_settle_steps:", rm.get("n_settle_steps"), "length:", rm.get("length"), "epilogue_steps:", rm.get("epilogue_steps"))
    g = int(rm.get("goal_first_step") or 0)
    def stats(m, t, label):
        s = t["eval__settled"][g:].astype(bool); sym = t["eval__symbolic"][g:].astype(bool)
        pos = t[f"object_pose__{obj}"][:, :3]; speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 30.0  # m/s at 30 Hz
        w = speed[g:]
        print(f"  {label}: success={m['success']} settled steps after goal {int(s.sum())}/{len(s)} (longest run {longest_true(s)}), symbolic True {int(sym.sum())}/{len(sym)}; "
              f"{obj} speed after goal: median {np.median(w)*1000:.2f} mm/s p90 {np.percentile(w,90)*1000:.2f} max {w.max()*1000:.2f} mm/s; final pos err {m.get('error_metric',{}).get('position_error')}")
    stats(rm, rt, "ref n1   ")
    for d in sorted(glob.glob(f"{R}/n8same/*.slot*")):
        m, t = runs(d); stats(m, t, f"slot {d[-1]}  ")
# tablespoon: the verify reasons
v = json.load(open(f"{C}/verify/inter_object_relation_satisfaction-nextto-tablespoon-chopstick-breakfast_table.json"))
print("\n## tablespoon verify reasons:"); [print("  -", r[:260]) for r in v["runs"]["n8same"]["reasons"]]
cj = json.load(open(f"{C}/runs/inter_object_relation_satisfaction-nextto-tablespoon-chopstick-breakfast_table/n8same/compare.json"))
for name, rep in cj.items():
    for row in rep.get("rows", [])[:8]:
        dv = row.get("deviation") or row.get("dev") or {}
        print("  slot", row.get("slot"), {k: (round(v, 4) if isinstance(v, float) else v) for k, v in dv.items()} if isinstance(dv, dict) else dv)
    print("  keys of a row:", list(rep.get("rows", [{}])[0].keys())[:30])
