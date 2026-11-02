import json, glob, sys
C = sys.argv[1]
for f in sorted(glob.glob(f"{C}/*.json")):
    j = json.load(open(f))
    if j.get("pass"): continue
    n8 = j["runs"].get("n8same", {}); n1p = j.get("n1_parity") or {}
    reasons = [r[:100] for r in (n8.get("reasons") or [])][:2]
    rr = [r[:80] for r in (j.get("relaxed_reasons") or [])][:2]
    print(f"{j['task'][:72]:72s} pass={j['pass']} relaxed={j.get('pass_relaxed')} comparator_only={j.get('comparator_only')} n1_parity={n1p.get('pass')} host={j.get('host')}")
    print(f"    reasons={reasons}")
    if rr: print(f"    relaxed_reasons={rr}")
