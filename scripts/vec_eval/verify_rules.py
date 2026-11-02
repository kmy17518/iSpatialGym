"""The pass rule of the N-slot open-loop replay verification (``compare_slots.py``'s ``compare.json`` of a
``--same-instance-all-slots`` run against its N=1 reference): every slot must reproduce the reference's verdicts -
``success``, ``symbolic_success``, ``geometric_success`` (settle-eval on, the replay default) - and its success steps within
``STEP_TOL`` steps, and no tracked object / robot trajectory may deviate from the reference by more than ``POS_TOL_M`` /
``ROT_TOL_RAD`` (PhysX's multi-scene resting-contact jitter is ~1e-4 m; a divergence beyond centimetres means the slot
saw other physics). ``decide(compare_json, meta_dirs)`` returns ``{"pass", "reasons", "slots", ...}``.

    python scripts/vec_eval/verify_rules.py <run dir of the same-instance run> [--ref <N=1 run dir>]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

STEP_TOL = 5          # success-step agreement (steps); contact chaos moves a crossing by a step or two
POS_TOL_M = 0.02      # max position deviation of any tracked object / base / eef over the episode
ROT_TOL_RAD = 0.05    # max rotation deviation (quaternion angle)
JOINT_TOL_RAD = 0.05  # robot joints (the replayed robot follows the same commands: a larger gap means other contacts)
# Companion verdict ``pass_relaxed`` (vec_next, 2026-10-30): the rotation rule counts only above ROT_RELAXED_RAD (a real tip-over / roll-over) -
# the position rule (POS_TOL_M) and the joint rule stay as they are. A free spin / roll of a symmetric object (a flower in a vase, a spoon on the
# table, a shaker frozen at a slightly different yaw when it falls asleep) at centimetre position agreement and identical verdicts is not a
# physical divergence. ``pass`` stays the 10-23 strict rule (comparable with the record); both are reported.
ROT_RELAXED_RAD = 0.3


def decide(compare: dict) -> dict:
    """``inconclusive`` (2026-10-27): an episode the driver cut by its wall clock (``wall_clock_exceeded``: the 1 s x max_steps
    cap in force until 2026-10-24 03:56) before the demo could reach its success step says nothing about the clones - the
    run must be repeated with ``--wallclock 0``; it is never counted as a divergence (4 of the 17 tasks verified below N=8 on
    2026-10-23 were demoted by such cuts). The verdict is ``pass: False`` with ``inconclusive: True`` and the cut slots named."""
    out = {"pass": True, "inconclusive": False, "reasons": [], "instances": {}, "pass_relaxed": True, "relaxed_reasons": []}
    if not compare:
        return {"pass": False, "inconclusive": False, "reasons": ["no comparison"], "instances": {}}
    for name, rep in compare.items():
        rows = rep.get("rows") or []
        ref_success, ref_geo = rep.get("anchor_success"), rep.get("anchor_geo_step")
        inst = {"anchor": rep.get("anchor"), "ref_success": ref_success, "ref_geo_step": ref_geo, "slots": []}
        if not rows:
            out["pass"] = False
            out["reasons"].append(f"{name}: no slot episodes")
        if rep.get("anchor_wall_clock_exceeded"):
            out["pass"], out["inconclusive"] = False, True
            out["reasons"].append(f"{name}: the N=1 reference was cut by the wall clock at step {rep.get('anchor_n_steps')} (inconclusive: re-run with --wallclock 0)")
        cut = [r for r in rows if r.get("wall_clock_exceeded")]
        if cut:
            out["pass"], out["inconclusive"] = False, True
            out["reasons"].append(f"{name}: slots {[r.get('slot') for r in cut]} cut by the wall clock at step "
                                  f"{sorted({r.get('n_steps') for r in cut})} of {sorted({r.get('max_steps') for r in cut})} (inconclusive: re-run with --wallclock 0)")
        for r in rows:
            problems = []
            if r.get("wall_clock_exceeded"):
                inst["slots"].append({"slot": r.get("slot"), "problems": ["wall clock cut (inconclusive)"], "inconclusive": True,
                                      **{k: r.get(k) for k in ("success", "geo_step", "sym_step", "n_steps", "reason")}})
                continue
            if r.get("status") not in ("success", "oracle_failed"):
                problems.append(f"status {r.get('status')}")
            if bool(r.get("success")) != bool(ref_success):
                problems.append(f"success {r.get('success')} != ref {ref_success}")
            if r.get("sym_step") is None and r.get("geo_step") is None and ref_geo is not None:
                problems.append("no success step")
            if ref_geo is not None and r.get("geo_step") is not None and abs(int(r["geo_step"]) - int(ref_geo)) > STEP_TOL:
                problems.append(f"geometric success step {r['geo_step']} vs ref {ref_geo}")
            if ref_geo is None and r.get("geo_step") is not None:
                problems.append(f"geometric success at {r['geo_step']} where the reference had none")
            inst["slots"].append({"slot": r.get("slot"), "problems": problems, **{k: r.get(k) for k in ("success", "geo_step", "sym_step", "n_steps", "reason")}})
            if problems:
                out["pass"] = False
                out["reasons"].append(f"{name} slot {r.get('slot')}: " + "; ".join(problems))
                out["pass_relaxed"] = False
                out["relaxed_reasons"].append(f"{name} slot {r.get('slot')}: " + "; ".join(problems))
        worst = {"pos_m": 0.0, "rot_rad": 0.0, "joints_rad": 0.0}
        for sname, dv in (rep.get("deviations") or {}).items():
            for k, v in dv.items():
                if not isinstance(v, dict):
                    continue
                if v.get("pos_max_m") is not None:
                    worst["pos_m"] = max(worst["pos_m"], float(v["pos_max_m"]))
                if v.get("rot_max_rad") is not None:
                    worst["rot_rad"] = max(worst["rot_rad"], float(v["rot_max_rad"]))
                if k == "joint_positions" and v.get("max_abs") is not None:
                    worst["joints_rad"] = max(worst["joints_rad"], float(v["max_abs"]))
            if (dv.get("length") or [0, 0])[0] != (dv.get("length") or [0, 0])[1] and not (cut or rep.get("anchor_wall_clock_exceeded")):
                out["pass"] = False
                out["reasons"].append(f"{name} {sname}: episode lengths {dv.get('length')}")
                out["pass_relaxed"] = False
                out["relaxed_reasons"].append(f"{name} {sname}: episode lengths {dv.get('length')}")
        inst["worst_deviation"] = worst
        if worst["pos_m"] > POS_TOL_M or worst["rot_rad"] > ROT_TOL_RAD or worst["joints_rad"] > JOINT_TOL_RAD:
            out["pass"] = False
            out["reasons"].append(f"{name}: trajectory deviation {worst} beyond {POS_TOL_M} m / {ROT_TOL_RAD} rad / {JOINT_TOL_RAD} rad")
        rot_counts = worst["rot_rad"] > ROT_RELAXED_RAD
        if worst["pos_m"] > POS_TOL_M or rot_counts or worst["joints_rad"] > JOINT_TOL_RAD:
            out["pass_relaxed"] = False
            out["relaxed_reasons"].append(f"{name}: trajectory deviation {worst} beyond {POS_TOL_M} m / {ROT_RELAXED_RAD} rad / {JOINT_TOL_RAD} rad joints")
        out["instances"][name] = inst
    if any(c.get("wall_clock_exceeded") for rep in compare.values() for c in (rep.get("rows") or [])) or any(rep.get("anchor_wall_clock_exceeded") for rep in compare.values()):
        out["pass_relaxed"] = False
    out["thresholds"] = {"step_tol": STEP_TOL, "pos_tol_m": POS_TOL_M, "rot_tol_rad": ROT_TOL_RAD, "joint_tol_rad": JOINT_TOL_RAD,
                         "relaxed_rot_tol_rad": ROT_RELAXED_RAD}
    return out


def verdict_flags(run_dir: str) -> dict:
    """Per slot episode: the three verdict flags, their settle-eval mode and the other mode (what the user asked to check)."""
    out = {}
    for m in glob.glob(os.path.join(run_dir, "*", "meta.json")):
        d = json.load(open(m))
        out[os.path.basename(os.path.dirname(m))] = {k: d.get(k) for k in ("success", "symbolic_success", "geometric_success", "settle_eval", "symbolic_success_step",
                                                                           "geometric_success_step", "n_steps", "reason")} | {"other_mode": d.get("other_mode")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref", default=None)
    args = ap.parse_args()
    cj = os.path.join(args.run, "compare.json")
    if not os.path.exists(cj):
        raise SystemExit(f"{cj} missing: run compare_slots.py first")
    d = decide(json.load(open(cj)))
    d["verdicts"] = verdict_flags(args.run)
    if args.ref:
        d["verdicts_ref"] = verdict_flags(args.ref)
    json.dump(d, open(os.path.join(args.run, "verify.json"), "w"), indent=1, default=str)
    print(json.dumps({k: d[k] for k in ("pass", "reasons")}, indent=1))


if __name__ == "__main__":
    main()
