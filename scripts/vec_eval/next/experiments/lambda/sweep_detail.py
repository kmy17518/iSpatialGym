import glob, json, os, sys
import numpy as np
root = sys.argv[1]; obj = sys.argv[2]
def qang(q1, q2):
    d = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0, 1); return 2 * np.arccos(d)
for run in sorted(glob.glob(f"{root}/*/")):
    name = os.path.basename(run.rstrip("/"))
    for d in sorted(glob.glob(f"{run}/*.slot*")):
        m = json.load(open(f"{d}/meta.json")); t = np.load(f"{d}/trajectory.npz"); key = f"object_pose__{obj}"
        P = t[key]; n = m["n_steps"]; g = int(m.get("goal_first_step") or 0); w = m["pose_settled_config"]["n_steps"]
        dp = np.abs(np.diff(P, axis=0)).max(axis=1)
        # first step from which the pose is bit-identical to the end (frozen = asleep)
        frozen_from = n
        for i in range(n - 2, -1, -1):
            if dp[i] == 0.0: frozen_from = i + 1
            else: break
        settled = t["eval__settled"].astype(bool); sym = t["eval__symbolic"].astype(bool)
        verd = (m.get("monitors") or {}).get("stats", {}).get("verdicts", []); msg = verd[0]["message"][:120] if verd else ""
        errs = m.get("errors_final_step") or {}
        pe = (errs.get("pose_errors") or {}).get(obj) or {}
        print(f"{name} slot{d[-1]}: success={m['success']} reason={m.get('qa_reason')} goal_first={g} geo_step={m.get('geometric_success_step')} n={n} window={w} | frozen from step {frozen_from} ({n-frozen_from} steps) | settled flags true {int(settled[g:].sum())}/{n-g} symbolic true {int(sym[g:].sum())}/{n-g} | final pos err {m.get('error_metric',{}).get('position_error')} | {msg} | pose_err {json.dumps(pe)[:120]}")
