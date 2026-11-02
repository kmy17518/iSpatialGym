"""Round D failure detail per alias: verdict reasons, per-slot outcomes, N=1 outcome, divergence timeline of the task objects vs slot 0 and vs n1."""
import json, glob, os, sys
import numpy as np
C = sys.argv[1]
def ang(q0, q1): return 2 * np.arccos(np.clip(np.abs(np.sum(q0 * q1, -1)), 0, 1))
for a in sys.argv[2:]:
    j = json.load(open(f"{C}/verify_D/{a}.json")); n8 = j["runs"]["n8same"]
    print(f"######## {a}")
    print(f"  strict={j['pass']} relaxed={j.get('pass_relaxed')} comparator_only={j.get('comparator_only')} n1_parity={(j.get('n1_parity') or {}).get('pass')} prev_N={(j.get('previous') or {}).get('num_envs')} host={j.get('host')}")
    for r in (n8.get("reasons") or [])[:3]: print("  reason:", r.split(": ", 1)[-1][:170])
    for r in ((j.get("n1_parity") or {}).get("reasons") or [])[:2]: print("  n1 parity:", r.split(": ", 1)[-1][:150])
    r1 = [json.loads(l) for l in open(f"{C}/runs_D/{a}/n1/results.jsonl")][0]
    print(f"  n1: success={r1.get('success')} reason={r1.get('reason')} symbolic_step={r1.get('symbolic_success_step')} geometric_step={r1.get('geometric_success_step')} steps={r1.get('n_steps')}")
    rows = sorted([json.loads(l) for l in open(f"{C}/runs_D/{a}/n8same/results.jsonl")], key=lambda r: r.get("slot", 0))
    print("  n8 slots:", [(r.get("slot"), "ok" if r.get("success") else (r.get("reason") or "fail")[:14], r.get("symbolic_success_step")) for r in rows])
    # divergence of each tracked object: angle / position vs slot 0 and vs n1 at the end, and the step where the spread first exceeds 20 mrad
    dirs = sorted(glob.glob(f"{C}/runs_D/{a}/n8same/*.slot*")); Z = [np.load(d + "/trajectory.npz", allow_pickle=True) for d in dirs]
    z1 = np.load(glob.glob(f"{C}/runs_D/{a}/n1/*template*")[0] + "/trajectory.npz", allow_pickle=True)
    T = min(min(len(z["step"]) for z in Z), len(z1["step"]))
    for k in [k for k in Z[0].keys() if k.startswith("object_pose__")]:
        name = k[13:]; P = [z[k][:T] for z in Z]; P1 = z1[k][:T]
        if any(np.isnan(p).all() for p in P): continue
        spread_t = [max(ang(P[s][t, 3:7], P[0][t, 3:7]) for s in range(1, len(P))) * 1e3 for t in range(0, T, 10)]
        first = next((t * 10 for t, v in enumerate(spread_t) if v > 20), None)
        end_rot = [round(float(ang(P[s][T-1, 3:7], P1[T-1, 3:7]) * 1e3)) for s in range(len(P))]
        end_pos = [round(float(np.linalg.norm(P[s][T-1, :3] - P1[T-1, :3]) * 1e3), 1) for s in range(len(P))]
        tail = slice(T - 100, T); still = [round(float(ang(p[tail][:-1, 3:7], p[tail][1:, 3:7]).mean() * 1e3), 3) for p in P]
        print(f"  {name:26s} vs n1 at the end: rot(mrad)={end_rot} pos(mm)={end_pos}; slot spread first >20 mrad at step {first}; tail100 per-step rot mean={still}")
