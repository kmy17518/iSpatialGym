import json, sys, math
d = sys.argv[1]
def load(p): return [json.loads(l) for l in open(p)]
n1 = {r["phase"]: r for r in load(f"{d}/n1.jsonl")}
n8 = load(f"{d}/n8same.jsonl")
def diff(a, b):
    out = {}
    for k in a:
        if k not in b or k == "error": continue
        x, y = a[k], b[k]
        if k == "joints":
            out[k] = max(abs(u - v) for u, v in zip(x, y)) * 1e3  # mrad
        else:
            dp = math.dist(x[:3], y[:3]) * 1e3  # mm
            dq = 2 * math.acos(min(1.0, abs(sum(u * v for u, v in zip(x[3:], y[3:]))))) * 1e3
            out[k] = max(dp, dq)
    return out
print("N=1 phases:", [(r, round(n1[r]["t"], 4)) for r in n1])
for r in n8:
    ph = r["phase"]
    ref_ph = ph
    for s, st in r["slots"].items():
        if ph.endswith(f"_{s}") and ph.rsplit("_", 1)[0] in ("before_load", "after_load", "after_slot"):
            ref_ph = ph.rsplit("_", 1)[0] + "_0"
    line = []
    for s, st in sorted(r["slots"].items(), key=lambda kv: int(kv[0])):
        base, _, kk = ph.rpartition("_")
        if base in ("before_load", "after_load", "after_slot"):
            k = int(kk)
            if k == int(s): rp = base + "_0"
            elif k > int(s): rp = "after_slot_0"   # slot s already loaded, waiting for the later slots
            else: rp = "before_load_0"           # slot s not loaded yet (template state)
        else:
            rp = ph
        if rp not in n1 or "0" not in n1[rp]["slots"]:
            continue
        dd = diff(st, n1[rp]["slots"]["0"])
        if dd:
            worst = max(dd.items(), key=lambda kv: kv[1])
            line.append(f"s{s}:{worst[1]:.2e}({worst[0][:10]})")
    if line:
        print(f"{ph:28s} t={r['t']:.4f} | " + " ".join(line))
