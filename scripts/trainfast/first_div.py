"""Where two collections of one instance first diverge: the first step whose joint positions or actions differ, its program phase and
action index, and the per-step gap growth after it."""
import glob, os, sys
import h5py
import numpy as np
root, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
for key_dir in sorted(glob.glob(os.path.join(root, "*"))):
    for pa in sorted(glob.glob(os.path.join(key_dir, A, "demo", "*", "demo.hdf5"))):
        pb = pa.replace(f"/{A}/demo/", f"/{B}/demo/")
        if not os.path.exists(pb):
            continue
        with h5py.File(pa, "r") as fa, h5py.File(pb, "r") as fb:
            ga, gb = fa["data/demo_0"], fb["data/demo_0"]
            n = min(len(ga["action"]), len(gb["action"]))
            dq = np.abs(ga["isg/joint_positions"][:n] - gb["isg/joint_positions"][:n]).max(axis=1)
            da = np.abs(ga["action"][:n] - gb["action"][:n]).max(axis=1)
            st_a, st_b = ga["state"][:n], gb["state"][:n]
            ds = np.abs(st_a - st_b).max(axis=1) if st_a.shape == st_b.shape else None
            idx = np.nonzero((dq > 1e-5) | (da > 1e-5))[0]
            sidx = np.nonzero(ds > 1e-5)[0] if ds is not None else []
            inst = os.path.basename(os.path.dirname(pa)).rsplit("__", 1)[-1]
            if len(idx) == 0 and len(sidx) == 0:
                print(f"{os.path.basename(key_dir)[:40]:40s} {inst}: identical over {n} steps"); continue
            i = int(idx[0]) if len(idx) else n
            j = int(sidx[0]) if len(sidx) else None
            ph, ai = int(ga["isg/phase"][min(i, n - 1)]), int(ga["isg/action_index"][min(i, n - 1)])
            print(f"{os.path.basename(key_dir)[:40]:40s} {inst}: robot/action first differ at step {i}/{n} (phase {ph}, action {ai}); "
                  f"full state first differs at step {j} ({'same layout' if ds is not None else 'state layouts differ'}); "
                  f"joint gap at +10/+50/+200 steps: {[round(float(dq[min(i + d, n - 1)]), 4) for d in (10, 50, 200)] if i < n else '-'}")
