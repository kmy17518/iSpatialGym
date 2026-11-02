import json, glob, sys
F = "/vision/u/minyeong/projects/isg_home/trainfast"
for m in sorted(glob.glob(f"{F}/out/large/*touching*/g*/demo/*/meta.json")):
    d = json.load(open(m))
    if d.get("reason") != "robot_stutter":
        continue
    qa = d.get("qa") or {}
    print("==", d["instance_id"][-6:], "reason", d.get("reason"), "verdict", str(d.get("verdict"))[:300])
    for k, v in qa.items():
        s = json.dumps(v, default=str)
        if "stutter" in k.lower() or "stutter" in s.lower():
            print("   qa.", k, s[:700])
    print("   straight_arm", (d.get("fast") or {}).get("straight_arm"))
    acts = d.get("actions") or []
    for a in acts:
        if isinstance(a, dict) and ("stutter" in json.dumps(a).lower() or a.get("status") not in (None, "ok")):
            print("   action", json.dumps(a, default=str)[:400])
    break
