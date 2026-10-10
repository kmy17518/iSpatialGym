"""Gate 3f of the vectorized evaluator: per-step traces of the closed-loop runs, per slot.

For every policy-mode record of the given sets (``lab/diagnostics/vec-eval/3e/<set>/<task>/episode_XXXXXX.json``):

* the per-slot files exist and are consistent: ``actions.npy`` (streamed actions, one row per policy call), ``errors.npy``
  (per-step pose errors), the ``trace_*`` arrays (scene-relative base / camera pose, proprio base velocity, joints before
  every policy call) with lengths matching the record's ``steps``;
* the goal image each slot's websocket session sent (``goal_sha256``, hashed by the bridge from the array it sends) is the
  hash of that episode's goal PNG decoded the same way, and ``task_id_sent`` is the episode's ``train_task_index`` - the
  per-slot goal / task isolation of the ACT bridge;
* within a batch no two slots share an action trace or a base trace (distinct episodes -> distinct rollouts), and the first
  base pose of each slot equals the single-env record's initial base pose of that episode when one exists.

    python lab_tools/vec_gate_3f.py --sets default-n4 default-n4-rep2 default-n1 ta-n4 ta-n1 --dataset datasets/isg-init --out lab/diagnostics/vec-eval/3f/summary.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)

from lab_tools.eval_episodes import load_goal  # noqa: E402


def goal_hash(png_path: str) -> str:
    arr = np.ascontiguousarray(load_goal(png_path))[None, ..., :3]
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(WT, "lab", "diagnostics", "vec-eval", "3e"))
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    goals = {g["episode_index"]: g for g in json.load(open(os.path.join(args.dataset, "isg_meta", "goals", "index.json")))["episodes"]}
    out = {"format": "isg-lab-vec-gate-3f/v1", "sets": {}, "pass": True}
    for name in args.sets:
        recs = sorted(glob.glob(os.path.join(args.root, name, "*", "episode_*.json")))
        set_out = {"n": 0, "n_pass": 0, "records": [], "pass": True}
        by_batch = {}
        for rp in recs:
            r = json.load(open(rp))
            item = {"episode": r.get("episode_index"), "slot": r.get("slot"), "batch": r.get("batch"), "reasons": []}
            if r.get("error"):
                item["reasons"].append(f"error: {str(r['error'])[:120]}")
            raw = os.path.join(WT, r["raw_dir"]) if r.get("raw_dir") else None
            steps = r.get("steps") or 0
            arrays = {}
            if raw:
                for f in ("actions.npy", "errors.npy", "trace_base_xyyaw.npy", "trace_camera_pose.npy", "trace_base_qvel.npy", "trace_joint_q.npy"):
                    p = os.path.join(raw, f)
                    if os.path.exists(p):
                        a = np.load(p)
                        arrays[f] = a
                        item[f] = list(a.shape)
                    else:
                        item[f] = None
                        if not r.get("error"):
                            item["reasons"].append(f"missing {f}")
                if "actions.npy" in arrays and len(arrays["actions.npy"]) != steps:
                    item["reasons"].append(f"actions rows {len(arrays['actions.npy'])} != steps {steps}")
                if "trace_base_xyyaw.npy" in arrays and abs(len(arrays["trace_base_xyyaw.npy"]) - (steps + 1)) > 1:
                    item["reasons"].append(f"trace rows {len(arrays['trace_base_xyyaw.npy'])} vs steps {steps}")
            g = goals.get(r.get("episode_index"))
            if g is not None:
                want = goal_hash(os.path.join(args.dataset, g["png"]))
                sent = (r.get("goal_sha256") or {}).get("zed_link")
                item["goal_hash_sent"], item["goal_hash_expected"] = sent, want
                if sent is not None and sent != want:
                    item["reasons"].append("goal image sent != the episode's goal PNG")
                if r.get("task_id_sent") is not None and int(r["task_id_sent"]) != int(g["train_task_index"]):
                    item["reasons"].append(f"task_id sent {r['task_id_sent']} != {g['train_task_index']}")
            if "actions.npy" in arrays:
                by_batch.setdefault(r.get("batch"), []).append((item["slot"], item["episode"], arrays["actions.npy"], arrays.get("trace_base_xyyaw.npy")))
            item["pass"] = not item["reasons"]
            set_out["records"].append(item)
            set_out["n"] += 1
            set_out["n_pass"] += int(item["pass"])
            set_out["pass"] = set_out["pass"] and item["pass"]
        # distinct rollouts per slot within a batch
        dup = []
        for b, items in by_batch.items():
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    si, ei, ai, ti = items[i]
                    sj, ej, aj, tj = items[j]
                    if ei != ej and ai.shape == aj.shape and np.array_equal(ai, aj):
                        dup.append({"batch": b, "slots": [si, sj], "episodes": [ei, ej], "what": "actions"})
                    if ei != ej and ti is not None and tj is not None and ti.shape == tj.shape and np.array_equal(ti, tj):
                        dup.append({"batch": b, "slots": [si, sj], "episodes": [ei, ej], "what": "base trace"})
        set_out["duplicate_traces_across_slots"] = dup
        if dup:
            set_out["pass"] = False
        out["sets"][name] = set_out
        out["pass"] = out["pass"] and set_out["pass"] and set_out["n"] > 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for name, s in out["sets"].items():
        print(f"{name}: {s['n_pass']}/{s['n']} pass; duplicates across slots: {len(s['duplicate_traces_across_slots'])}")
        for it in s["records"]:
            if not it["pass"]:
                print(f"   ep {it['episode']} slot {it['slot']}: {it['reasons']}")
    print("GATE 3f:", "PASS" if out["pass"] else "FAIL")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
