import json, glob, os, sys, collections
F = "/vision/u/minyeong/projects/isg_home/trainfast"
key = sys.argv[1]; R = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
tj = sorted(glob.glob(f"{F}/out/pilot/{key}/g*/inst/*/*/template.json"))[0]; d = os.path.dirname(tj)
t = json.load(open(tj)); m = json.load(open(d + "/template_meta.json"))
init = t["objects_info"]["init_info"]; reg = t["state"]["registry"]["object_registry"]
B = {n: (s["non_kin"]["AABB"]["aabb"][:3], s["non_kin"]["AABB"]["aabb"][3:]) for n, s in reg.items() if (s.get("non_kin") or {}).get("AABB")}
i2n = m["inst_to_name"]; sup = next(i2n.get(b["name"]) for b in m["bindings"] if b["role"] == "support")
rooms = set(m["task_rooms"])
slo = [B[sup][0][i] - R for i in range(2)]; shi = [B[sup][1][i] + R for i in range(2)]
def ov(b): return all(b[1][i] >= slo[i] and b[0][i] <= shi[i] for i in range(2))
cat = lambda n: init[n]["args"].get("category")
allc = collections.Counter(cat(n) for n in init if not n.startswith("robot"))
inreg = [n for n in init if not n.startswith("robot") and (n not in B or ov(B[n]))]
regc = collections.Counter(cat(n) for n in inreg)
print(f"{key}: support {sup} box {[round(v,2) for v in B[sup][0]]}..{[round(v,2) for v in B[sup][1]]}, region +-{R} m")
print("  loaded by category (top):", allc.most_common(12))
print("  in region by category:", regc.most_common(20))
big = [(n, round(B[n][1][0]-B[n][0][0],1), round(B[n][1][1]-B[n][0][1],1)) for n in inreg if n in B and cat(n) in ("walls","ceilings","floors","roof")]
print("  structure in region:", len(big), "e.g.", big[:6])
print("  non-structure in region:", sorted(n for n in inreg if cat(n) not in ("walls","ceilings","floors","roof","door","window"))[:40])
