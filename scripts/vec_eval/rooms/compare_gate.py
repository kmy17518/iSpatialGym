#!/usr/bin/env python
"""Campaign rooms: the gate between two replays of the same demos under different room loadings (<gate>/<task>/template vs
<gate>/<task>/override, or template vs template_repeat for the noise floor): per dumped step and camera the mean absolute
difference (0-255, at the dump resolution and at the policy's 224^2 input), the fraction of pixels differing by > 2 / > 8,
the max; per episode the verdicts and the trajectory deviation (compare_slots.diff). Writes <out> (default
<gate>/<task>/gate_<a>_vs_<b>.json) and prints one line per task.

    python compare_gate.py --task <alias> [--a template --b override] [--root $ISG_VEC_ROOT] [--campaign <root>/campaigns/rooms]
                           [--gate <campaign>/gate] [--scripts-dir <this checkout>/scripts/vec_eval] [--no-dataset] [--dataset-root DIR]

Inputs: <gate>/<task>/<a|b>/<episode>/{meta.json, trajectory.npz, frames/step_NNNNN_<cam>.png} (run_isg_vec --dump-frames);
<instances>/all/<task>.json (the val instances' video_file / goal_timestamp / length, for the optional comparison of the head
frames with the recorded demo frames of the LeRobot dataset under --dataset-root); compare_slots.py and
build_instance_lists.py from --scripts-dir.
"""
import argparse, glob, json, os, sys

import numpy as np

DEFAULT_ROOT = os.environ.get("ISG_VEC_ROOT", "/vision/u/minyeong/projects/isg_home/vec_eval85")
DEFAULT_DATASET = "/vision/u/minyeong/projects/isg_home/train85/datasets/isg-1020-b1k"
DEFAULT_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/vec_eval of this checkout
FPS = 30.0


def down224(x):
    from PIL import Image
    return np.asarray(Image.fromarray(x.astype(np.uint8)).resize((224, 224), Image.BILINEAR, reducing_gap=None)).astype(np.float32)


def load_png(p):
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32)


def episodes(d):
    return {os.path.basename(e): e for e in glob.glob(f"{d}/*") if os.path.isdir(e) and os.path.exists(f"{e}/meta.json")}


def dataset_head_frame(spec, step, dataset_root):
    """The recorded head frame of a val demo at ``step`` (the dataset video of the episode; ``goal_timestamp`` is the last
    frame's video timestamp, frames are FPS apart), as uint8 HxWx3 - or None."""
    try:
        from build_instance_lists import decode_frame
        ts = float(spec["goal_timestamp"]) - (int(spec["length"]) - 1 - int(step)) / FPS
        return decode_frame(os.path.join(dataset_root, spec["video_file"]), ts).astype(np.float32)
    except Exception:  # noqa: BLE001
        return None


def val_specs(instances_dir, task):
    d = json.load(open(os.path.join(instances_dir, "all", f"{task}.json")))["instances"]
    return {s["name"]: s for s in d if s.get("video_file")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="the vec_eval root: <root>/instances/all/<task>.json (env ISG_VEC_ROOT)")
    ap.add_argument("--campaign", default=None, help="the campaign folder (default <root>/campaigns/rooms)")
    ap.add_argument("--instances", default=None, help="the instance lists folder (default <root>/instances)")
    ap.add_argument("--gate", default=None, help="the gate runs, <gate>/<task>/{template,override,template_repeat}/ (default <campaign>/gate)")
    ap.add_argument("--a", default="template")
    ap.add_argument("--b", default="override")
    ap.add_argument("--out", default=None, help="default <gate>/<task>/gate_<a>_vs_<b>.json")
    ap.add_argument("--scripts-dir", default=DEFAULT_SCRIPTS, help="folder with compare_slots.py and build_instance_lists.py (default: this checkout's scripts/vec_eval)")
    ap.add_argument("--dataset", action=argparse.BooleanOptionalAction, default=True, help="also compare the val episodes' head frames with the recorded demo frames")
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET, help="the LeRobot dataset the val demos were recorded in (video_file paths are relative to it)")
    a = ap.parse_args()
    campaign = a.campaign or os.path.join(a.root, "campaigns", "rooms")
    instances = a.instances or os.path.join(a.root, "instances")
    gate = a.gate or os.path.join(campaign, "gate")
    sys.path.insert(0, a.scripts_dir)
    from compare_slots import diff as traj_diff  # noqa: E402

    specs = val_specs(instances, a.task)
    da, db = f"{gate}/{a.task}/{a.a}", f"{gate}/{a.task}/{a.b}"
    ea, eb = episodes(da), episodes(db)
    common = sorted(set(ea) & set(eb))
    rep = {"task": a.task, "a": da, "b": db, "episodes": {}, "summary": {}}
    cams_all = {}
    for name in common:
        ma, mb = json.load(open(f"{ea[name]}/meta.json")), json.load(open(f"{eb[name]}/meta.json"))
        # the first geometric success step is compared (the evaluated step, chosen by the lowest error within the settle window since
        # 2026-11-01, can move far on a tiny error difference); metas before that change carry only geometric_success_step (= the first)
        def _verdict(m):
            v = {k: m.get(k) for k in ("success", "symbolic_success", "geometric_success", "geometric_success_step", "n_steps", "reason", "status")}
            v["geometric_first_step"] = m.get("geometric_first_step") if "geometric_first_step" in m else m.get("geometric_success_step")
            return v
        e = {"verdict_a": _verdict(ma), "verdict_b": _verdict(mb),
             "rooms_a": (ma.get("load_rooms") or {}).get("rooms"), "rooms_b": (mb.get("load_rooms") or {}).get("rooms"), "frames": {}}
        e["verdicts_equal"] = all(e["verdict_a"][k] == e["verdict_b"][k] for k in ("success", "symbolic_success", "geometric_success", "geometric_first_step"))
        ta, tb = f"{ea[name]}/trajectory.npz", f"{eb[name]}/trajectory.npz"
        if os.path.exists(ta) and os.path.exists(tb):
            dv = traj_diff(np.load(ta), np.load(tb))
            e["traj"] = {"pos_max_m": max((v.get("pos_max_m") or 0.0) for v in dv.values() if isinstance(v, dict)),
                         "rot_max_rad": max((v.get("rot_max_rad") or 0.0) for v in dv.values() if isinstance(v, dict)),
                         "joints_rad": (dv.get("joint_positions") or {}).get("max_abs"), "length": dv.get("length"),
                         "per_key": {k: v for k, v in dv.items() if isinstance(v, dict) and "pos_max_m" in v}}
        fa = sorted(glob.glob(f"{ea[name]}/frames/step_*_*.png"))
        per_cam = {}
        spec = specs.get(name)
        ds_rows = []
        for pa in fa:
            base = os.path.basename(pa)
            pb = f"{eb[name]}/frames/{base}"
            if not os.path.exists(pb):
                continue
            step = int(base.split("_")[1])
            cam = base.split("_", 2)[2][:-4]
            A, B = load_png(pa), load_png(pb)
            if A.shape != B.shape:
                continue
            d = np.abs(A - B)
            # at the policy's input resolution (ACT: 224^2 bilinear antialiased): a one-pixel edge shift at 720^2 (renderer
            # realisation noise on a close object) nearly vanishes, a changed region (a room's content, lighting) stays
            d224 = np.abs(down224(A) - down224(B))
            per_cam.setdefault(cam, []).append({"step": step, "mean_abs": float(d.mean()), "mean_abs_224": float(d224.mean()), "max_224": float(d224.max()),
                                                "frac_gt2": float((d.max(-1) > 2).mean()), "frac_gt8": float((d.max(-1) > 8).mean()), "max": float(d.max())})
            # the recorded demo frame (val episodes, head camera): how far each run is from the dataset
            if cam == "head" and spec is not None and a.dataset and step < int(spec["length"]):
                D = dataset_head_frame(spec, step, a.dataset_root)
                if D is not None and D.shape == A.shape:
                    ds_rows.append({"step": step, "a_vs_demo": float(np.abs(A - D).mean()), "b_vs_demo": float(np.abs(B - D).mean())})
        if ds_rows:
            e["head_vs_demo"] = {"n": len(ds_rows), "a_mean": float(np.mean([r["a_vs_demo"] for r in ds_rows])), "b_mean": float(np.mean([r["b_vs_demo"] for r in ds_rows])),
                                 "rows": ds_rows}
        for cam, rows in per_cam.items():
            rows.sort(key=lambda r: r["step"])
            e["frames"][cam] = {"n": len(rows), "mean_abs_mean": float(np.mean([r["mean_abs"] for r in rows])), "mean_abs_max": float(max(r["mean_abs"] for r in rows)),
                                "mean_abs_224_mean": float(np.mean([r["mean_abs_224"] for r in rows])), "mean_abs_224_max": float(max(r["mean_abs_224"] for r in rows)),
                                "frac_gt8_max": float(max(r["frac_gt8"] for r in rows)), "worst_step": max(rows, key=lambda r: r["mean_abs"])["step"], "rows": rows}
            c = cams_all.setdefault(cam, {"mean_abs": [], "frac_gt8": []})
            c["mean_abs"] += [r["mean_abs"] for r in rows]
            c["frac_gt8"] += [r["frac_gt8"] for r in rows]
        rep["episodes"][name] = e
    ds = [e["head_vs_demo"] for e in rep["episodes"].values() if e.get("head_vs_demo")]
    rep["summary"] = {"head_vs_demo": {"episodes": len(ds), "a_mean": float(np.mean([x["a_mean"] for x in ds])), "b_mean": float(np.mean([x["b_mean"] for x in ds]))} if ds else None,
                      "episodes": len(common), "verdicts_equal": all(e["verdicts_equal"] for e in rep["episodes"].values()) if common else None,
                      "traj_pos_max_m": max((e.get("traj") or {}).get("pos_max_m") or 0.0 for e in rep["episodes"].values()) if common else None,
                      "traj_rot_max_rad": max((e.get("traj") or {}).get("rot_max_rad") or 0.0 for e in rep["episodes"].values()) if common else None,
                      "cams": {cam: {"frames": len(c["mean_abs"]), "mean_abs_mean": float(np.mean(c["mean_abs"])), "mean_abs_max": float(max(c["mean_abs"])),
                                     "frac_gt8_max": float(max(c["frac_gt8"]))} for cam, c in cams_all.items()}}
    out = a.out or f"{gate}/{a.task}/gate_{a.a}_vs_{a.b}.json"
    json.dump(rep, open(out, "w"), indent=1, default=str)
    s = rep["summary"]
    cams = " ".join(f"{cam}: mean {v['mean_abs_mean']:.3f} max {v['mean_abs_max']:.3f} (>8: {v['frac_gt8_max'] * 100:.2f} %)" for cam, v in sorted(s["cams"].items()))
    dsm = s.get("head_vs_demo")
    ds_txt = f" | head vs demo: {a.a} {dsm['a_mean']:.2f} {a.b} {dsm['b_mean']:.2f} ({dsm['episodes']} val ep)" if dsm else ""
    print(f"{a.task} [{a.a} vs {a.b}] episodes {s['episodes']} verdicts_equal {s['verdicts_equal']} traj pos {(s['traj_pos_max_m'] or 0) * 100:.2f} cm rot {s['traj_rot_max_rad'] or 0:.4f} | frames/255: {cams}{ds_txt}")


if __name__ == "__main__":
    main()
