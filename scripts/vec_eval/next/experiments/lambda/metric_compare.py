"""New-metric replays vs the same code's replays before the change (2026-11-01): trajectories must be bit-identical, verdicts
unchanged, the new *_first_step = the old *_success_step, the new *_success_step within the selection window."""
import glob, json, sys, numpy as np
task = sys.argv[1]
new, old = f"/home/minyeongk/vec_eval85/next/metric_change/{task}", f"/home/minyeongk/vec_eval85/next/opt3/{task}"
ok = True
for run in ("n1", "n8same"):
    for dn in sorted(glob.glob(f"{new}/{run}/*/meta.json")):
        ep = dn.split("/")[-2]; do = f"{old}/{run}/{ep}/meta.json"
        mn, mo = json.load(open(dn)), json.load(open(do))
        tn, to = np.load(dn.replace("meta.json", "trajectory.npz")), np.load(do.replace("meta.json", "trajectory.npz"))
        same_traj = set(tn.files) == set(to.files) and all(tn[k].shape == to[k].shape and np.array_equal(tn[k], to[k], equal_nan=tn[k].dtype.kind == "f") for k in tn.files if tn[k].dtype.kind != "O")
        W = mn.get("settle_steps")
        checks = {
            "trajectory_bit_identical": same_traj,
            "success_same": mn.get("success") == mo.get("success"),
            "n_steps_same": mn.get("n_steps") == mo.get("n_steps"),
            "geo_first==old_geo_step": mn.get("geometric_first_step") == mo.get("geometric_success_step"),
            "sym_first==old_sym_step": mn.get("symbolic_first_step") == mo.get("symbolic_success_step"),
        }
        if mn.get("settle_eval"):  # a settled success is judged exactly as before
            checks["settled: geo step == old"] = mn.get("geometric_success_step") == mo.get("geometric_success_step")
            checks["settled: sym step == old"] = mn.get("symbolic_success_step") == mo.get("symbolic_success_step")
            checks["settled: error_metric == old"] = json.dumps(mn.get("error_metric"), sort_keys=True) == json.dumps(mo.get("error_metric"), sort_keys=True)
            checks["settled: errors == old"] = json.dumps(mn.get("errors"), sort_keys=True, default=str) == json.dumps(mo.get("errors"), sort_keys=True, default=str)
        for k in ("geometric", "symbolic"):
            f, b = mn.get(f"{k}_first_step"), mn.get(f"{k}_success_step")
            if f is None:
                checks[f"{k}_window"] = b is None
            elif mn.get("settle_eval"):
                checks[f"{k}_window"] = b == f
            else:
                checks[f"{k}_window"] = f <= b <= f + W - 1
        ca, cg = mn.get("closest_approach"), mn.get("closest_goal_approach")
        checks["closest records present"] = ca is not None and cg is not None
        if ca and cg:
            em = mn.get("error_metric") or {}
            tol = next(iter(((mn.get("errors") or {}).get("pose_errors") or {}).values()), {}).get("tolerances", {}).get("position")
            judged = (em.get("position_error") or 0.0) / tol if tol else None
            checks["closest <= goal-closest <= judged"] = ca["score"] <= cg["score"] and (judged is None or cg["score"] <= judged + 1e-12)
        bad = [k for k, v in checks.items() if not v]
        ok &= not bad
        print(f"{run}/{ep[-12:]}: {'OK' if not bad else 'FAIL ' + str(bad)} | settle_eval={mn.get('settle_eval')} W={W} geo first {mn.get('geometric_first_step')} -> evaluated {mn.get('geometric_success_step')} "
              f"(old {mo.get('geometric_success_step')}), err {mo.get('error_metric')} -> {mn.get('error_metric')}, evaluation_settled={mn.get('evaluation_settled')}")
print("METRIC_COMPARE", "ALL OK" if ok else "PROBLEMS")
