import json, glob, os, sys, collections
F = "/vision/u/minyeong/projects/isg_home/trainfast"
STRUCT = {"walls", "floors", "ceilings", "roof", "fence", "door", "sliding_door", "window", "driveway", "lawn", "background", "carpet"}
def boxes(t):
    reg = t["state"]["registry"]["object_registry"]; out = {}
    for n, s in reg.items():
        a = ((s.get("non_kin") or {}).get("AABB") or {}).get("aabb")
        if a and len(a) == 6: out[n] = (a[:3], a[3:])
    return out
def overlap(b, lo, hi):
    return all(b[1][i] >= lo[i] and b[0][i] <= hi[i] for i in range(2))
print(f"{'alias':42s} {'loaded':>6s} {'task rm':>7s} {'floor+1m':>8s} {'supp+2.5m':>9s} {'partial_rooms':>13s}  load_s")
for key in sys.argv[1:]:
    tj = sorted(glob.glob(f"{F}/out/pilot/{key}/g*/inst/*/*/template.json")) or sorted(glob.glob(f"{F}/out/*/{key}/g*/inst/*/*/template.json"))
    if not tj: print(key, "no template"); continue
    d = os.path.dirname(tj[0]); t = json.load(open(tj[0])); m = json.load(open(d + "/template_meta.json"))
    init = t["objects_info"]["init_info"]; B = boxes(t); rooms = set(m.get("task_rooms") or m.get("rooms") or [])
    names = [n for n in init if not n.startswith("robot")]
    in_task = [n for n in names if set(init[n]["args"].get("in_rooms") or []) & rooms]
    fl = [B[n] for n in names if init[n]["args"].get("category") == "floors" and set(init[n]["args"].get("in_rooms") or []) & rooms and n in B]
    if fl:
        lo = [min(b[0][i] for b in fl) - 1.0 for i in range(3)]; hi = [max(b[1][i] for b in fl) + 1.0 for i in range(3)]
        floor1 = [n for n in names if n not in B or overlap(B[n], lo, hi)]
    else: floor1 = names
    i2n = m.get("inst_to_name") or {}
    sup = None
    for b in m.get("bindings") or []:
        if isinstance(b, dict) and b.get("role") == "support":
            sup = i2n.get(b.get("name"), b.get("name"))
    if sup not in B:
        cands = [v for k, v in i2n.items() if v in B and not v.startswith(("target", "partner", "robot", "floors", "walls")) and not k.startswith("floor")]
        sup = cands[0] if cands else None
    if sup in B:
        slo = [B[sup][0][i] - 2.5 for i in range(3)]; shi = [B[sup][1][i] + 2.5 for i in range(3)]
        reg = [n for n in names if n not in B or overlap(B[n], slo, shi)]
    else: reg = []
    pr = d + "/template-partial_rooms.json"
    npr = len([n for n in json.load(open(pr))["objects_info"]["init_info"] if not n.startswith("robot")]) if os.path.exists(pr) else -1
    print(f"{key[:42]:42s} {len(names):6d} {len(in_task):7d} {len(floor1):8d} {len(reg) if reg else '-':>9} {npr:13d}  {m.get('scene_load_s', 0):.0f}  support={sup}")
