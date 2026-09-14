"""Do articulated joints hold a mid-way position when released? Set each openable joint to 50 % open, step 3 s of physics, measure drift."""
import os, json, glob
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import torch as th, omnigibson as og
from omnigibson.objects.dataset_object import DatasetObject
tests = [("hinged_jar", "vzwhbg"), ("hinged_jar", "gqtsam"), ("carton", "msfzpz"), ("toolbox", "redmtl"), ("cooler", "pnkiag"), ("top_cabinet", "dmwxyl"), ("fridge", "petcxr"), ("bottom_cabinet", "bamfsz"), ("oven", "kfftgk"), ("door", "cbgtjl"), ("laptop", "nvulcs"), ("cedar_chest", "fwstpx")]
env = og.Environment(configs={"scene": {"type": "Scene"}}); scene = env.scene
for _ in range(3): og.sim.step()
for cat, model in tests:
    og.sim.stop(); obj = DatasetObject(name=f"o_{model}", category=cat, model=model, fixed_base=True); scene.add_object(obj); og.sim.play()
    for _ in range(2): og.sim.step()
    H = json.load(open(f"/tmp/glib/artic/handles/{cat}__{model}.json"))
    names = list(obj.joints.keys()); q = obj.get_joint_positions().clone()
    for j in H["joints"]:
        if j["joint"] in names: q[names.index(j["joint"])] = j["closed_q"] + j["open_sign"] * 0.5 * (j["upper"] - j["lower"])
    obj.set_joint_positions(q); obj.set_joint_velocities(th.zeros_like(q)); og.sim.step()
    q0 = obj.get_joint_positions().clone()
    for _ in range(int(3.0 / og.sim.get_physics_dt())): og.sim.step()
    q1 = obj.get_joint_positions()
    out = []
    for j in H["joints"]:
        if j["joint"] not in names: continue
        i = names.index(j["joint"]); jp = obj.joints[j["joint"]]
        drift = float(q1[i] - q0[i]); span = j["upper"] - j["lower"]
        out.append(f"{j['joint']}({j['type'][:4]}): set {float(q0[i]):.3f} -> after 3 s {float(q1[i]):.3f} (drift {drift:+.3f} = {100 * drift / span:+.0f}% of range; friction={float(jp.friction) if hasattr(jp, 'friction') else 'n/a'} damping={float(jp.damping) if hasattr(jp, 'damping') else 'n/a'})")
    print(f"{cat}/{model}: " + "; ".join(out), flush=True)
    og.sim.stop(); scene.remove_object(obj); og.sim.play(); og.sim.step()
og.shutdown()
