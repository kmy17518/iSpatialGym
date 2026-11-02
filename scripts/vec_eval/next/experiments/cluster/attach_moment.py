import numpy as np, sys, glob, os
d = sorted(glob.glob(os.path.join(sys.argv[1], "*.slot0")))[0]
z = np.load(d + "/trajectory.npz", allow_pickle=True)
L = z["object_pose__lid.n.02_1"]; Tp = z["object_pose__teapot.n.01_1"]; el = z["eef_pose_left"]; er = z["eef_pose_right"]
for t in list(range(740, 790, 2)) + [850, 900, 1000]:
    dl = np.linalg.norm(el[t,:3]-L[t,:3]); dr = np.linalg.norm(er[t,:3]-L[t,:3])
    rel = L[t,:3]-Tp[t,:3]
    print(f"t={t} lid-teapot rel={np.round(rel*1e3,1)}mm  lid-eefL={dl*1e3:.0f}mm lid-eefR={dr*1e3:.0f}mm  lid step dp={np.linalg.norm(L[t,:3]-L[t-1,:3])*1e3:.2f}mm teapot dz={ (Tp[t,2]-Tp[t-1,2])*1e3:.2f}mm phase={int(z['phase'][t])} sym={int(z['eval__symbolic'][t])}")
