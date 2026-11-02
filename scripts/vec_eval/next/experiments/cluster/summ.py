import json, glob, sys
C = "/vision/u/minyeong/projects/isg_home/vec_eval85/campaigns/vec_next"
def summ(d):
    rows = []
    for f in sorted(glob.glob(f"{C}/{d}/*.json")):
        j = json.load(open(f)); n8 = j["runs"].get("n8same", {})
        rows.append((j["task"], j["pass"], (j.get("previous") or {}).get("num_envs"), (n8.get("reasons") or [""])[0][:110], n8.get("world_build_s"), j.get("host")))
    return rows
for d in sys.argv[1:] or ("verify", "verify_B"):
    rows = summ(d); print(f"== {d}: {len(rows)} verdicts, {sum(1 for r in rows if r[1])} PASS")
    for r in rows:
        if (not r[1]) or ((r[2] or 8) < 8):
            print("  " + ("PASS" if r[1] else "FAIL") + f" prev N={r[2]!s:4} {r[0][:72]:72} {r[3]}")
