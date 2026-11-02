#!/usr/bin/env python
"""Campaign rooms: the gate decision per task from compare_gate.py's outputs (gate_template_vs_override.json and
gate_template_vs_template_repeat.json), the final override file (passing tasks only) and the before / after video index.

Rule (per task, over its gate episodes):
  physics  - every episode: same verdicts (success / symbolic / geometric / geometric success step); trajectory deviation of
             the override vs the template <= POS_TOL m / ROT_TOL rad (the replay-verification thresholds of EVAL_RECORD §5);
  frames   - for the robot cameras (head, left, right) on the first-60-step dumps (steps 0..60 every 5: the robot has barely
             moved, so neither a physics divergence nor the renderer's load-dependent frame latency enters the pixels), at the
             policy's input resolution (224^2 bilinear): per step the *excess* of the override's mean |diff| over the same-host
             repeat's mean |diff| (both vs the template run); pass when the mean excess <= EXCESS_MEAN and the p90 excess
             <= EXCESS_MAX (in /255; single-frame spikes are the frame-latency effect on a moving arm), i.e. the override is within the renderer's own run-to-run noise.
  Later frames (every 25th step) are reported but do not decide: with identical physics they still show one-frame shifts of
  fast-moving close objects in the wrist cameras (the observation lags the physics by a load-dependent 1-2 frames, lab notes
  2026-09-15; a lighter scene renders faster), and after a physics divergence they carry it.

    python decide_gate.py [--root $ISG_VEC_ROOT] [--campaign <root>/campaigns/rooms] [--gate <campaign>/gate]
                          [--candidate <campaign>/summary/rooms_override_candidate.json] [--out <campaign>/summary] [--source TEXT]

Outputs: <out>/gate.md, <out>/rooms_override.json ({generated, source, rule, overrides: {alias: {rooms, variant, template_rooms,
dropped, scene, evidence}}}; ``variant`` from the template meta of <instances>/eval/<task>.json - run_isg.rooms_for_template
matches by it), <out>/gate_rows.json (every gated task, pass or fail).
"""
import argparse, glob, json, os, time

import numpy as np

DEFAULT_ROOT = os.environ.get("ISG_VEC_ROOT", "/vision/u/minyeong/projects/isg_home/vec_eval85")
DEFAULT_SOURCE = "campaign rooms 2026-10-28 (segmentation pass + same-host frame / physics gate)"
POS_TOL, ROT_TOL = 0.02, 0.05
EXCESS_MEAN, EXCESS_MAX = 0.25, 0.60
CAMS = ("head", "left", "right")


def excess_stats(g, r, early=True):
    """Per camera: the per-step excess (override mean|diff| - repeat mean|diff|) over the common dumped steps of all episodes."""
    out = {}
    for cam in CAMS + ("external",):
        ex = []
        for name, e in g["episodes"].items():
            re_ = (r["episodes"].get(name) or {}).get("frames", {}).get(cam, {}).get("rows", [])
            rr = {x["step"]: x.get("mean_abs_224", x["mean_abs"]) for x in re_}
            for x in e.get("frames", {}).get(cam, {}).get("rows", []):
                if (x["step"] <= 60) != early:
                    continue
                if x["step"] in rr:
                    ex.append(x.get("mean_abs_224", x["mean_abs"]) - rr[x["step"]])
        if ex:
            out[cam] = {"n": len(ex), "mean_excess": float(np.mean(ex)), "max_excess": float(np.max(ex)), "p90_excess": float(np.percentile(ex, 90))}
    return out


def template_meta(instances_dir, task):
    ev = json.load(open(os.path.join(instances_dir, "eval", f"{task}.json")))["instances"][0]
    tro = ev["tro"]
    return json.load(open(os.path.join(os.path.dirname(os.path.dirname(tro)), os.path.basename(tro).replace("-tro_state.json", "") + "-meta.json")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="the vec_eval root: <root>/instances/eval/<task>.json (env ISG_VEC_ROOT)")
    ap.add_argument("--campaign", default=None, help="the campaign folder (default <root>/campaigns/rooms)")
    ap.add_argument("--instances", default=None, help="the instance lists folder (default <root>/instances)")
    ap.add_argument("--gate", default=None, help="compare_gate.py's outputs, <gate>/<task>/gate_template_vs_{override,template_repeat}.json (default <campaign>/gate)")
    ap.add_argument("--candidate", default=None, help="analyze_visibility.py's output (default <campaign>/summary/rooms_override_candidate.json)")
    ap.add_argument("--out", default=None, help="where gate.md, rooms_override.json and gate_rows.json go (default <campaign>/summary)")
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="the 'source' field written into rooms_override.json (name the campaign / date)")
    a = ap.parse_args()
    campaign = a.campaign or os.path.join(a.root, "campaigns", "rooms")
    instances = a.instances or os.path.join(a.root, "instances")
    gate = a.gate or os.path.join(campaign, "gate")
    candidate = a.candidate or os.path.join(campaign, "summary", "rooms_override_candidate.json")
    out_dir = a.out or os.path.join(campaign, "summary")
    os.makedirs(out_dir, exist_ok=True)
    cand = json.load(open(candidate))
    rows, final = [], {"generated": time.strftime("%Y-%m-%dT%H:%M"), "source": a.source,
                       "rule": {"pos_tol_m": POS_TOL, "rot_tol_rad": ROT_TOL, "excess_mean_255": EXCESS_MEAN, "excess_p90_255": EXCESS_MAX, "frames": "steps 0..60 every 5, robot cameras, 224^2 bilinear (the policy input)"},
                       "overrides": {}}
    md = ["# Gate: replays under the candidate rooms vs the template rooms (same host; the repeat = the host's noise floor)\n",
          "| task | dropped | physics (verdicts, pos cm, rot) | head excess mean / p90 / max (224², /255) | left | right | later frames (head mean excess) | result |", "|---|---|---|---|---|---|---|---|"]
    for t in sorted(cand["overrides"]):
        e = cand["overrides"][t]
        if not e["dropped"]:
            continue
        pg, pr = f"{gate}/{t}/gate_template_vs_override.json", f"{gate}/{t}/gate_template_vs_template_repeat.json"
        if not (os.path.exists(pg) and os.path.exists(pr)):
            md.append(f"| `{t}` | {len(e['dropped'])} | *gate not finished* | | | | | pending |")
            continue
        g, r = json.load(open(pg)), json.load(open(pr))
        s = g["summary"]
        phys_ok = bool(s["verdicts_equal"]) and (s["traj_pos_max_m"] or 0) <= POS_TOL and (s["traj_rot_max_rad"] or 0) <= ROT_TOL
        early, late = excess_stats(g, r, True), excess_stats(g, r, False)
        # p90 instead of max: single-frame spikes come from the renderer's load-dependent frame latency on a moving arm (gym_shoe, crab, mug)
        fr_ok = all(early.get(c, {}).get("mean_excess", 9) <= EXCESS_MEAN and early.get(c, {}).get("p90_excess", 9) <= EXCESS_MAX for c in CAMS)
        ok = phys_ok and fr_ok
        res = "PASS" if ok else ("FAIL (frames)" if phys_ok else ("FAIL (physics)" if fr_ok else "FAIL (physics, frames)"))
        cell = lambda c: f"{early[c]['mean_excess']:+.2f} / {early[c]['p90_excess']:+.2f} / {early[c]['max_excess']:+.2f}" if c in early else "—"
        md.append(f"| `{t}` | {', '.join(e['dropped'])} | {'=' if s['verdicts_equal'] else '≠'}, {(s['traj_pos_max_m'] or 0) * 100:.2f}, {s['traj_rot_max_rad'] or 0:.3f} | {cell('head')} | {cell('left')} | {cell('right')} | "
                  f"{late.get('head', {}).get('mean_excess', float('nan')):+.2f} | **{res}** |")
        rows.append({"task": t, "result": res, "physics_ok": phys_ok, "frames_ok": fr_ok, "early": early, "late": late, "summary": s, "dropped": e["dropped"], "rooms": e["rooms"]})
        if ok:
            final["overrides"][t] = {"rooms": e["rooms"], "variant": None, "template_rooms": e["template_rooms"], "dropped": e["dropped"], "scene": e["scene"],
                                     "evidence": {"segmentation_demos": e["demos"], "gate_episodes": s["episodes"], "head_excess_mean_255": early.get("head", {}).get("mean_excess"),
                                                  "traj_pos_max_m": s["traj_pos_max_m"], "traj_rot_max_rad": s["traj_rot_max_rad"]}}
    # the template variant of each passing task (run_isg.rooms_for_template matches by variant)
    for t, e in final["overrides"].items():
        e["variant"] = template_meta(instances, t).get("variant")
    n_pass = sum(1 for x in rows if x["result"] == "PASS")
    md.append(f"\n{len(rows)} gated tasks: **{n_pass} pass**, {len(rows) - n_pass} fail (rule: verdicts equal, trajectory <= {POS_TOL * 100:.0f} cm / {ROT_TOL} rad, "
              f"first-60-step robot-camera excess over the same-host repeat <= {EXCESS_MEAN}/255 mean and {EXCESS_MAX}/255 p90). Rooms dropped by the passing tasks: "
              f"{sum(len(x['dropped']) for x in rows if x['result'] == 'PASS')}.")
    # before / after videos
    md.append("\n## Before / after videos (tiled per episode; grid per batch) — template rooms vs candidate rooms\n")
    for x in rows:
        t = x["task"]
        md.append(f"### `{t}` — {x['result']} — dropped: {', '.join(x['dropped'])}")
        for ep in sorted(os.listdir(f"{gate}/{t}/template")):
            pa, pb = f"{gate}/{t}/template/{ep}/tiled_rgb.mp4", f"{gate}/{t}/override/{ep}/tiled_rgb.mp4"
            if os.path.exists(pa) and os.path.exists(pb):
                md.append(f"- `{ep[-60:]}`: [before]({pa}) · [after]({pb}) · [repeat]({gate}/{t}/template_repeat/{ep}/tiled_rgb.mp4)")
        md.append("")
    open(os.path.join(out_dir, "gate.md"), "w").write("\n".join(md) + "\n")
    json.dump(final, open(os.path.join(out_dir, "rooms_override.json"), "w"), indent=1)
    json.dump(rows, open(os.path.join(out_dir, "gate_rows.json"), "w"), indent=1, default=str)
    print("\n".join(md[: len(rows) + 6]))


if __name__ == "__main__":
    main()
