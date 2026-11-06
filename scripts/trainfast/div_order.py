"""Which diverges first between the cropped and full collections of an instance: the recorded joints (physics) or the actions (plan)."""
import glob, os, sys
import h5py
import numpy as np
root = sys.argv[1]
for pa in sorted(glob.glob(root + "/*/partial/demo/*/demo.hdf5")):
    pb = pa.replace("/partial/demo/", "/full/demo/")
    if not os.path.exists(pb):
        continue
    with h5py.File(pa) as fa, h5py.File(pb) as fb:
        ga, gb = fa["data/demo_0"], fb["data/demo_0"]
        n = min(len(ga["action"]), len(gb["action"]))
        dq = np.abs(ga["isg/joint_positions"][:n] - gb["isg/joint_positions"][:n]).max(axis=1)
        da = np.abs(ga["action"][:n] - gb["action"][:n]).max(axis=1)
        iq, ia = np.nonzero(dq > 1e-6)[0], np.nonzero(da > 1e-6)[0]
        if not len(iq) and not len(ia):
            continue
        q0 = int(iq[0]) if len(iq) else None
        a0 = int(ia[0]) if len(ia) else None
        if q0 is not None and (a0 is None or q0 < a0):
            order = "physics first"
        elif a0 is not None and (q0 is None or a0 < q0):
            order = "action (plan) first"
        else:
            order = "same step"
        name = pa.split("/pv/")[1].split("/")[0].replace("inter_object_relation_satisfaction-", "iors-")
        inst = os.path.basename(os.path.dirname(pa)).rsplit("__", 1)[-1]
        jq = dq[q0] if q0 is not None else 0.0
        ja = da[a0] if a0 is not None else 0.0
        print(f"{name[:32]:32s} {inst}: joints first differ at {q0}, actions at {a0} -> {order}; first gaps joint {jq:.2e} action {ja:.2e}")
