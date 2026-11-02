import json, glob, os, sys, collections
F = "/vision/u/minyeong/projects/isg_home/trainfast"
STRUCT = {"walls", "floors", "ceilings", "roof", "fence", "door", "sliding_door", "garage_door", "gate", "railing"}  # loader: building structure + doors
def box(s):
    a = ((s.get("non_kin") or {}).get("AABB") or {}).get("aabb")
    return (a[:3], a[3:]) if a and len(a) == 6 else None
print(f"{'alias':34s} {'scene json':>10s} {'loaded':>6s} {'(struct)':>8s} {'task room':>9s} {'floor+1m':>8s} {'supp+2.5m':>9s} {'supp+1.5m':>9s}")
for key in sys.argv[1:]:
    tj = sorted(glob.glob(f"{F}/out/pilot/{key}/g*/inst/*/*/template.json")) or sorted(glob.glob(f"{F}/out/*/{key}/g*/inst/*/*/template.json"))
    d = os.path.dirname(tj[0]); t = json.load(open(tj[0])); m = json.load(open(d + "/template_meta.json"))
    init = t["objects_info"]["init_info"]; reg = t["state"]["registry"]["object_registry"]
    loaded_rooms = set(m.get("load_room_instances") or []); rooms = set(m.get("task_rooms") or [])
    cat = lambda n: init[n]["args"].get("category"); inr = lambda n: set(init[n]["args"].get("in_rooms") or [])
    names = [n for n in init if not n.startswith("robot")]
    spawned = [n for n in names if n.startswith(("target_", "partner_", "obstacles_", "distractor", "source_", "destination_", "board_", "block_"))]
    loaded = [n for n in names if (inr(n) & loaded_rooms) or cat(n) in STRUCT or n in spawned or not loaded_rooms]
    B = {n: box(reg[n]) for n in loaded if n in reg and box(reg[n])}
    task = [n for n in loaded if inr(n) & rooms or n in spawned]
    fl = [B[n] for n in loaded if cat(n) == "floors" and inr(n) & rooms and n in B]
    def within(lo, hi): return [n for n in loaded if n not in B or all(B[n][1][i] >= lo[i] and B[n][0][i] <= hi[i] for i in range(2))]
    floor1 = within([min(b[0][i] for b in fl) - 1 for i in range(2)], [max(b[1][i] for b in fl) + 1 for i in range(2)]) if fl else loaded
    i2n = m.get("inst_to_name") or {}
    sup = next((i2n.get(b["name"]) for b in (m.get("bindings") or []) if b.get("role") == "support"), None)
    def around(R):
        if sup not in B: return "-"
        return len(within([B[sup][0][i] - R for i in range(2)], [B[sup][1][i] + R for i in range(2)]))
    nst = sum(1 for n in loaded if cat(n) in STRUCT)
    print(f"{key[:34]:34s} {len(names):10d} {len(loaded):6d} {nst:8d} {len(task):9d} {len(floor1):8d} {around(2.5):>9} {around(1.5):>9}")
