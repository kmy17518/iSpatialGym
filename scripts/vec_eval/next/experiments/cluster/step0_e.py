import glob, sys, numpy as np
C = "/vision/u/minyeong/projects/isg_home/vec_eval85/campaigns/vec_next"
R, a = sys.argv[1], sys.argv[2]
def ld(p): z = np.load(p, allow_pickle=True); return {k: z[k] for k in z.files}
Z = [ld(d + "/trajectory.npz") for d in sorted(glob.glob(f"{C}/{R}/{a}/n8same/*.slot*"))]
z1 = ld(glob.glob(f"{C}/{R}/{a}/n1/*template*")[0] + "/trajectory.npz")
for t in (0, 1):
    for i in (0, 1, 5):
        z = Z[i]
        out = []
        for k in z1:
            if k in ("step", "phase") or z1[k].dtype.kind not in "fiub" or z1[k].ndim < 2: continue
            d = np.abs(z[k][t].astype(float) - z1[k][t].astype(float))
            if d.max() > 0: out.append(f"{k[:34]}={d.max():.1e}@{int(d.argmax())}")
        print(f"step {t} slot {i}:", "; ".join(out)[:900])
