import glob, json, os, numpy as np
C = "/vision/u/minyeong/projects/isg_home/vec_eval85/campaigns/vec_next"
def ld(p): z = np.load(p, allow_pickle=True); return {k: z[k] for k in z.files}
def first_diff(z, z1):
    keys = [k for k in z1 if (k.startswith("object_pose__") or k in ("joint_positions", "base_pose")) and k in z]
    T = min(len(z1["step"]), len(z["step"]))
    for t in range(T):
        if any(not np.array_equal(z[k][t], z1[k][t]) for k in keys): return t, T
    return None, T
rows = []
for f in sorted(glob.glob(f"{C}/verify_F/*.json")):
    a = os.path.basename(f)[:-5]
    try: e = json.load(open(f)); d = json.load(open(f"{C}/verify_E/{a}.json"))
    except Exception: continue
    res = {}
    for R in ("runs_E", "runs_F"):
        try:
            z1 = ld(glob.glob(f"{C}/{R}/{a}/n1/*/trajectory.npz")[0]); z0 = ld(sorted(glob.glob(f"{C}/{R}/{a}/n8same/*.slot0/trajectory.npz"))[0])
            res[R] = first_diff(z0, z1)
        except Exception as ex:
            res[R] = (f"err", 0)
    rows.append((a, d["pass"], d.get("pass_relaxed"), e["pass"], e.get("pass_relaxed"), (e.get("n1_parity") or {}).get("pass"), res["runs_E"], res["runs_F"]))
print(f"{'alias':60s} E(strict/relaxed) F(strict/relaxed) N1parity_vs_E | slot0 == N=1 until step (E -> F) / T")
ident_D = ident_E = 0
for a, dp, dr, ep, er, par, rd, re_ in rows:
    fd = "all" if rd[0] is None else rd[0]; fe = "all" if re_[0] is None else re_[0]
    ident_D += rd[0] is None; ident_E += re_[0] is None
    print(f"{a[:60]:60s} {int(dp)}/{int(bool(dr))}  ->  {int(ep)}/{int(bool(er))}   parity={par}   slot0: {fd} -> {fe} / {re_[1]}")
print(f"n={len(rows)} strict E {sum(r[1] for r in rows)} F {sum(r[3] for r in rows)} | relaxed E {sum(bool(r[2]) for r in rows)} E {sum(bool(r[4]) for r in rows)} | N=1 parity vs E {sum(bool(r[5]) for r in rows)} | slot 0 identical to N=1 for the whole episode: E {ident_D} F {ident_E}")
