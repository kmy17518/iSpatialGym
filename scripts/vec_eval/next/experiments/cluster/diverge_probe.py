"""When do the slots diverge? Per object: angle vs slot 0 over time (every 100 steps) and the tail per-step rotation per slot."""
import numpy as np, sys, glob, os
def ang(q0,q1): return 2*np.arccos(np.clip(np.abs(np.sum(q0*q1,-1)),0,1))
for root in sys.argv[1:]:
    dirs = sorted(glob.glob(os.path.join(root, "*.slot*")))
    if not dirs: print("##", root, "no slots"); continue
    Z = [np.load(d+"/trajectory.npz", allow_pickle=True) for d in dirs]
    T = min(len(z["step"]) for z in Z)
    print(f"######## {os.path.basename(os.path.dirname(root))}/{os.path.basename(root)} slots={len(Z)} T={T} settled_end={[int(z['eval__settled'][-1]) for z in Z]} symbolic_end={[int(z['eval__symbolic'][-1]) for z in Z]}")
    for k in [k for k in Z[0].keys() if k.startswith("object_pose__")]:
        n = k[len("object_pose__"):]
        P = [z[k][:T] for z in Z]
        if any(np.isnan(p).all() for p in P): continue
        rows = []
        for t in list(range(0, T, 100)) + [T-1]:
            d = max(ang(P[j][t,3:7], P[0][t,3:7]) for j in range(1,len(P))) * 1e3
            dp = max(np.linalg.norm(P[j][t,:3]-P[0][t,:3]) for j in range(1,len(P))) * 1e3
            rows.append(f"{t}:{d:.1f}/{dp:.1f}")
        tail = slice(T-150, T)
        rock = [ang(p[tail][:-1,3:7], p[tail][1:,3:7]).mean()*1e3 for p in P]
        mv = [np.linalg.norm(np.diff(p[tail][:, :3], axis=0), axis=1).mean()*1e3 for p in P]
        print(f"  {n:22s} max-vs-slot0 (mrad/mm) at steps: {' '.join(rows)}")
        print(f"  {'':22s} tail150 per-step rot mean per slot (mrad): {[round(r,3) for r in rock]} | pos (mm): {[round(m,3) for m in mv]}")
