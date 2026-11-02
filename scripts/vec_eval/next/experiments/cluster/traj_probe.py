import numpy as np, sys, glob, os
root = sys.argv[1]; TAIL = int(sys.argv[2]) if len(sys.argv) > 2 else 300
dirs = sorted(glob.glob(os.path.join(root, "*.slot*")))
def qang(q0, q1):
    d = np.abs(np.sum(q0*q1, axis=-1)).clip(0,1); return 2*np.arccos(d)
def relrot(qa, qb):
    xa,ya,za,wa = qa.T; xb,yb,zb,wb = qb.T
    return 2*np.stack([wa*xb - xa*wb - ya*zb + za*yb, wa*yb + xa*zb - ya*wb - za*xb, wa*zb - xa*yb + ya*xb - za*wb], 1)
ends = {}
for d in dirs:
    z = np.load(os.path.join(d, "trajectory.npz"), allow_pickle=True)
    T = len(z["step"]); ph = z["phase"]
    settled = z["eval__settled"]; sym = z["eval__symbolic"]
    print(f"== {os.path.basename(d)[-5:]} T={T} phases={np.unique(ph).tolist()} last-phase-start={int(np.argmax(ph==ph[-1]))} settled[-1]={bool(settled[-1])} symbolic[-1]={bool(sym[-1])} settled-true-steps={int(settled.sum())} first-settled={int(np.argmax(settled)) if settled.any() else -1}")
    for k in z.keys():
        if not k.startswith("object_pose__"): continue
        n = k[len("object_pose__"):]; X = z[k]; p, q = X[:, :3], X[:, 3:7]
        if np.isnan(p).all(): continue
        tail = slice(max(0, T-TAIL), T)
        dp = np.linalg.norm(np.diff(p[tail], axis=0), axis=1)
        dq = qang(q[tail][:-1], q[tail][1:])
        net_p = np.linalg.norm(p[-1]-p[tail.start]); net_q = qang(q[tail.start:tail.start+1], q[-1:])[0]
        rv = relrot(q[tail][:-1], q[tail][1:]); ax = int(np.argmax(np.abs(rv).sum(0))); s = np.sign(rv[:, ax]); flips = int((s[1:]*s[:-1] < 0).sum())
        # dominant period of the tilt oscillation: zero crossings of rv[:,ax] - mean
        print(f"  {n:18s} tail{TAIL}: net dp={net_p*1e3:.3f}mm rot={net_q*1e3:.3f}mrad | per-step dp mean={dp.mean()*1e3:.4f} max={dp.max()*1e3:.4f}mm  rot mean={dq.mean()*1e3:.4f} max={dq.max()*1e3:.4f}mrad | dom axis={ax} signflips={flips}/{len(s)}")
        ends.setdefault(n, []).append((p[-1], q[-1]))
print("== across-slot spread of the FINAL pose")
for n, L in ends.items():
    P = np.array([a for a,_ in L]); Q = np.array([b for _,b in L])
    dpos = np.linalg.norm(P - P[0], axis=1); drot = qang(Q[0:1].repeat(len(Q),0), Q)
    print(f"  {n:18s} vs slot0: dpos(mm)={np.round(dpos*1e3,3).tolist()} drot(mrad)={np.round(drot*1e3,2).tolist()}")
