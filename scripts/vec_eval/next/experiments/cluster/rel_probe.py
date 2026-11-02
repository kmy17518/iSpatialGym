"""Lid-vs-teapot relative motion per slot: is the lid jittering relative to the teapot (attachment joint) or is the whole stack rocking?"""
import numpy as np, sys, glob, os
def qmul(a, b):  # xyzw
    x1,y1,z1,w1 = a.T; x2,y2,z2,w2 = b.T
    return np.stack([w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2, w1*w2-x1*x2-y1*y2-z1*z2], 1)
def qconj(a): return a*np.array([-1,-1,-1,1])
def qrot(q, v):  # rotate vectors v by quats q
    u = q[:, :3]; s = q[:, 3:4]
    return 2*np.sum(u*v,1,keepdims=True)*u + (s*s - np.sum(u*u,1,keepdims=True))*v + 2*s*np.cross(u, v)
def ang(q): return 2*np.arccos(np.clip(np.abs(q[:,3]),0,1))
for root in sys.argv[1:]:
    dirs = sorted(glob.glob(os.path.join(root, "*.slot*")))
    print("########", root, len(dirs), "slots")
    for d in dirs:
        z = np.load(os.path.join(d, "trajectory.npz"), allow_pickle=True)
        L = z["object_pose__lid.n.02_1"]; Tp = z["object_pose__teapot.n.01_1"]; T = len(L)
        pl, ql = L[:, :3].astype(np.float64), L[:, 3:7].astype(np.float64); pt, qt = Tp[:, :3].astype(np.float64), Tp[:, 3:7].astype(np.float64)
        # lid pose in the teapot frame
        rel_q = qmul(qconj(qt), ql); rel_p = qrot(qconj(qt), pl - pt)
        # attachment moment: first step after which the relative pose stays within 2 mm / 10 mrad of its final value
        fin_p, fin_q = rel_p[-1], rel_q[-1]
        dp_fin = np.linalg.norm(rel_p - fin_p, axis=1); dq_fin = ang(qmul(qconj(rel_q), np.broadcast_to(fin_q, rel_q.shape)))
        close = (dp_fin < 0.004) & (dq_fin < 0.03)
        t_att = T - 1
        while t_att > 0 and close[t_att-1]: t_att -= 1
        tail = slice(max(t_att+10, T-150), T)
        rp, rq = rel_p[tail], rel_q[tail]
        drp = np.linalg.norm(np.diff(rp, axis=0), axis=1); drq = ang(qmul(qconj(rq[:-1]), rq[1:]))
        # absolute teapot & lid per-step rotation in the same window
        dqt = ang(qmul(qconj(qt[tail][:-1]), qt[tail][1:])); dql = ang(qmul(qconj(ql[tail][:-1]), ql[tail][1:]))
        settled = z["eval__settled"]; 
        print(f"  {os.path.basename(d)[-5:]}: lid-in-teapot-frame stable from step {t_att} (of {T}); window[{tail.start}:{T}] REL per-step |dp| mean={drp.mean()*1e3:.3f} max={drp.max()*1e3:.3f} mm, rot mean={drq.mean()*1e3:.3f} max={drq.max()*1e3:.3f} mrad; REL spread in window: pos {np.ptp(rp,0).max()*1e3:.2f} mm rot {ang(qmul(qconj(rq), np.broadcast_to(rq[-1], rq.shape))).max()*1e3:.2f} mrad | ABS teapot rot/step mean={dqt.mean()*1e3:.3f} max={dqt.max()*1e3:.3f}; lid mean={dql.mean()*1e3:.3f} max={dql.max()*1e3:.3f} mrad | settled[-1]={bool(settled[-1])} rel_p_final={np.round(fin_p*1e3,1).tolist()}mm")
