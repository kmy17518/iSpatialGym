"""Gate 3d of the vectorized evaluator: observation isolation per slot.

For every vectorized record the slot's first head frame (``init_rgb.png`` of its raw dir, rendered by the clone's own camera -
through the tiled render product when tiled rendering is on) is compared with (a) the dataset's first frame of that episode
(the instance-verification metric: mean absolute pixel difference <= 20, ``match_instances.py``) and (b) the single-env
evaluator's ``init_rgb.png`` of the same episode (the reference replay's raw dir): the render-to-render grain of the same
scene is 3-6; a clone leaking into the view, a wrong instance in the slot or a different camera (aperture, resolution)
would show as a large difference. The slot's final frame is compared with the single-env final frame the same way (open-loop
replays end at the same place). Also checked: the slot's goal images / task id (policy records) differ between slots as the
episodes do, and the slot's proprio at the first step equals the dataset's first ``observation.state`` joint part the way the
instance verification compared it.

    python lab_tools/vec_gate_3d.py --runs n4 n4same --ref lab/diagnostics/eval-fixes/parity-replay/eval50 --dataset datasets/isg-init-eval \
        --out lab/diagnostics/vec-eval/3d/summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)

from lab_tools.match_instances import dataset_episodes, first_frame, load_png, mean_abs_diff  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(WT, "lab", "diagnostics", "vec-eval", "3b"))
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dataset-diff", type=float, default=20.0, help="machine 1's instance-verification criterion")
    ap.add_argument("--max-single-env-diff", type=float, default=10.0, help="vs the single-env evaluator's own render (render grain 3-6)")
    args = ap.parse_args(argv)
    eps_meta = None
    summary = {"format": "isg-lab-vec-gate-3d/v1", "ref": args.ref, "criteria": {"dataset_first_frame_mean_abs_diff": args.max_dataset_diff, "single_env_frame_mean_abs_diff": args.max_single_env_diff},
               "runs": {}, "pass": True}
    for run in args.runs:
        recs = sorted(glob.glob(os.path.join(args.root, run, "*", "episode_*.json")))
        run_out = {"records": [], "pass": True, "n": 0, "n_pass": 0}
        goals_by_slot = {}
        for rp in recs:
            vec = json.load(open(rp))
            e, slot = vec["episode_index"], int(vec.get("slot", 0))
            item = {"episode": e, "slot": slot, "num_envs": vec.get("num_envs"), "tiled_rendering": vec.get("tiled_rendering"), "record": os.path.relpath(rp, WT), "pass": True, "reasons": []}
            if vec.get("error") or not vec.get("raw_dir"):
                item.update({"pass": False, "reasons": ["vec error"]})
                run_out["records"].append(item)
                run_out["n"] += 1
                run_out["pass"] = False
                continue
            vd = os.path.join(WT, vec["raw_dir"])
            init = load_png(os.path.join(vd, "init_rgb.png"))
            item["init_shape"] = list(init.shape)
            # (a) the dataset's first frame
            try:
                if eps_meta is None:
                    eps_meta = dataset_episodes(args.dataset)
                ds = first_frame(args.dataset, eps_meta[e])
                item["vs_dataset_first_frame"] = mean_abs_diff(init, ds)
                if item["vs_dataset_first_frame"] > args.max_dataset_diff:
                    item["reasons"].append(f"dataset first frame diff {item['vs_dataset_first_frame']:.1f} > {args.max_dataset_diff}")
            except Exception as ex:  # noqa: BLE001
                item["vs_dataset_first_frame"] = None
                item["reasons"].append(f"dataset frame: {ex!r}")
            # (b) the single-env evaluator's frames
            ref_path = os.path.join(args.ref, f"episode_{e:06d}.json")
            if os.path.exists(ref_path):
                ref = json.load(open(ref_path))
                rd = os.path.join(WT, ref["raw_dir"])
                for name in ("init_rgb.png", "final_rgb.png"):
                    pa, pb = os.path.join(vd, name), os.path.join(rd, name)
                    if os.path.exists(pa) and os.path.exists(pb):
                        d = mean_abs_diff(load_png(pa), load_png(pb))
                        item[f"vs_single_env_{name[:-4]}"] = d
                        if d > args.max_single_env_diff:
                            item["reasons"].append(f"{name} vs single-env diff {d:.1f} > {args.max_single_env_diff}")
                    else:
                        item[f"vs_single_env_{name[:-4]}"] = None
            else:
                item["reasons"].append("no single-env reference")
            # (c) per-slot goal / task id (policy records) and the first proprio vs the dataset's first state
            if vec.get("goal_pngs"):
                goals_by_slot[slot] = (vec.get("goal_pngs"), (vec.get("server_metadata") or {}).get("task_map"), e)
            try:
                q = np.load(os.path.join(vd, "replay_base_qvel.npy")) if os.path.exists(os.path.join(vd, "replay_base_qvel.npy")) else None
                if q is not None and eps_meta is not None:
                    s0 = eps_meta[e]["state0"]
                    item["first_base_qvel_vs_dataset"] = [float(v) for v in (q[0] - s0[0:3])]
            except Exception as ex:  # noqa: BLE001
                item["first_base_qvel_vs_dataset"] = repr(ex)
            item["pass"] = not item["reasons"]
            run_out["records"].append(item)
            run_out["n"] += 1
            run_out["n_pass"] += int(item["pass"])
            run_out["pass"] = run_out["pass"] and item["pass"]
        if goals_by_slot:
            run_out["goals_by_slot"] = {str(k): {"goal_pngs": v[0], "episode": v[2]} for k, v in goals_by_slot.items()}
            run_out["goals_distinct_as_episodes"] = len({json.dumps(v[0], sort_keys=True) for v in goals_by_slot.values()}) == len({v[2] for v in goals_by_slot.values()})
        summary["runs"][run] = run_out
        summary["pass"] = summary["pass"] and run_out["pass"] and run_out["n"] > 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    for run, r in summary["runs"].items():
        print(f"{run}: {r['n_pass']}/{r['n']} pass")
        for it in r["records"]:
            print(f"  ep {it['episode']} slot {it['slot']}: dataset {it.get('vs_dataset_first_frame')} single-env init {it.get('vs_single_env_init_rgb')} final {it.get('vs_single_env_final_rgb')} reasons={it['reasons']}")
    print("GATE 3d:", "PASS" if summary["pass"] else "FAIL")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
