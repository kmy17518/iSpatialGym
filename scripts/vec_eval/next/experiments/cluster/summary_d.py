"""Round D summary table -> <verify dir>/SUMMARY.md (and stdout)."""
import json, glob, sys, os, time
V = sys.argv[1]
J = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(V, "*.json")))]
n = len(J); p = sum(j["pass"] for j in J); r = sum(bool(j.get("pass_relaxed")) for j in J)
co = sum(bool(j.get("comparator_only")) for j in J); par = sum((j.get("n1_parity") or {}).get("pass") is True for j in J)
lines = [f"# Round D ({os.path.basename(V)}) - {time.strftime('%Y-%m-%d %H:%M %Z')}", "",
         f"{n} aliases at N=8 (same demo in every slot) vs the same code at N=1: **{p} PASS strict**, {r} PASS relaxed (rotation counted only above 0.3 rad), "
         f"{co} comparator-only (identical verdicts and success steps, only the strict rotation rule fails), N=1 parity vs round B's N=1 (stock physics): {par} PASS.", "",
         "| alias | strict | relaxed | comparator-only | N=1 parity vs B | previous N | host | first reason |", "|---|---|---|---|---|---|---|---|"]
for j in J:
    n8 = j["runs"].get("n8same", {}); reason = (n8.get("reasons") or [""])[0]
    reason = reason.split(": ", 1)[-1][:90] if reason else ""
    prev = (j.get("previous") or {}).get("num_envs")
    lines.append(f"| {j['task']} | {'PASS' if j['pass'] else 'FAIL'} | {'PASS' if j.get('pass_relaxed') else 'FAIL'} | {'yes' if j.get('comparator_only') else ''} | "
                 f"{'PASS' if (j.get('n1_parity') or {}).get('pass') else 'FAIL'} | {prev if prev is not None else ''} | {j.get('host')} | {reason} |")
prev_low = [(j["task"], (j.get("previous") or {}).get("num_envs")) for j in J if j["pass"] and ((j.get("previous") or {}).get("num_envs") or 8) < 8]
lines += ["", f"Previously verified below N=8 ({len(prev_low)}), now PASS at 8: " + ", ".join(f"{t} (was {k})" for t, k in prev_low)]
out = "\n".join(lines) + "\n"
open(os.path.join(V, "SUMMARY.md"), "w").write(out); print(out[:3000])
