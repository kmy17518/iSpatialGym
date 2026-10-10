"""Item 2 (localize the multi-scene deviation): the base scene's full state after ``restore_batch`` and after each of the first K
steps of a recorded episode, for an N-environment world (the other slots idle: no episode, robots holding). Saved as a pickle of
flattened state leaves per step; ``--compare A B`` prints the first step and the first quantity at which two traces differ.

    python lab_tools/exp_state_trace.py --dataset datasets/isg-init-eval --instances lab/instances.json --episode 2 --num-envs 1 --steps 10 --out runlogs/diagnostics/item2/trace_n1.pkl
    python lab_tools/exp_state_trace.py --dataset datasets/isg-init-eval --instances lab/instances.json --episode 2 --num-envs 2 --steps 10 --out runlogs/diagnostics/item2/trace_n2idle.pkl
    python lab_tools/exp_state_trace.py --compare runlogs/diagnostics/item2/trace_n1.pkl runlogs/diagnostics/item2/trace_n2idle.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)


def flatten(x, prefix="", out=None):
    out = {} if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            flatten(v, f"{prefix}/{k}", out)
    elif isinstance(x, (list, tuple)):
        if x and all(isinstance(v, (int, float, bool)) for v in x):
            out[prefix] = np.asarray(x, dtype=np.float64)
        else:
            for i, v in enumerate(x):
                flatten(v, f"{prefix}[{i}]", out)
    elif hasattr(x, "detach"):
        out[prefix] = np.asarray(x.detach().cpu(), dtype=np.float64).reshape(-1)
    elif isinstance(x, np.ndarray):
        out[prefix] = x.astype(np.float64).reshape(-1)
    elif isinstance(x, (int, float, bool)):
        out[prefix] = np.asarray([float(x)])
    else:
        out[prefix] = str(x)
    return out


def record(args) -> int:
    from lab_tools import evaluator_patches

    evaluator_patches.install(keep_depth=False, no_planner=True, fixed_handle_ag=True)
    import omnigibson as og
    import torch as th

    from ispatialgym.oracle.vec_loader import VecInstanceLoader
    from lab_tools.replay_episodes import load_episode_arrays

    inst = json.load(open(args.instances))
    eps = inst.get("episodes", inst)
    tro = eps[str(args.episode)]["tro"]
    actions, states, length = load_episode_arrays(args.dataset, args.episode)
    loader = VecInstanceLoader(num_envs=args.num_envs, tiled_rendering=False, warm_renders=args.warm_renders, physics_overrides=json.loads(args.physics) if args.physics else None,
                               dummy_bodies=args.dummy_bodies, enhanced_determinism=args.enhanced_determinism)
    loader.build_env(tro)
    env = loader.env
    obs, infos = loader.restore_batch([tro] + [None] * (args.num_envs - 1))
    scene0 = env.scenes[0]
    robot0 = loader.robots[0]

    def snapshot(tag):
        st = scene0.dump_state(serialized=False)
        flat = flatten(st, "")
        flat["/robot/joint_positions"] = np.asarray(robot0.get_joint_positions().detach().cpu(), dtype=np.float64)
        flat["/robot/joint_velocities"] = np.asarray(robot0.get_joint_velocities().detach().cpu(), dtype=np.float64)
        p, q = robot0.get_position_orientation()
        flat["/robot/pos"] = np.asarray(p.detach().cpu(), dtype=np.float64)
        flat["/robot/quat"] = np.asarray(q.detach().cpu(), dtype=np.float64)
        try:
            flat["/robot/joint_efforts"] = np.asarray(robot0.get_joint_efforts().detach().cpu(), dtype=np.float64)
        except Exception:  # noqa: BLE001
            pass
        try:
            from omnigibson.utils.usd_utils import RigidContactAPI

            pairs = RigidContactAPI.get_contact_pairs(scene0.idx, [robot0], None, current_only=False)
            flat["/robot/contact_pairs"] = sorted(f"{a}|{b}" for a, b in pairs)
        except Exception as e:  # noqa: BLE001
            flat["/robot/contact_pairs"] = f"error {e!r}"
        return {"tag": tag, "flat": flat}

    trace = [snapshot("after_restore")]
    idle = {i: loader.hold_action(i) for i in range(1, args.num_envs)}
    for t in range(args.steps):
        a = th.zeros((args.num_envs, 23), dtype=th.float32)
        a[0] = th.as_tensor(actions[t], dtype=th.float32)
        for i, h in idle.items():
            a[i] = h
        env.step(a)
        trace.append(snapshot(f"step_{t}"))
    meta = {"num_envs": args.num_envs, "episode": args.episode, "steps": args.steps, "physics": args.physics, "dummy_bodies": args.dummy_bodies, "report": {k: v for k, v in loader.report.items() if k != "trav_cells_blocked"}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"meta": meta, "trace": trace}, f)
    print("TRACE_WRITTEN", args.out, "snapshots", len(trace), flush=True)
    og.shutdown()
    return 0


def compare(a_path: str, b_path: str, top: int = 12) -> int:
    A, B = pickle.load(open(a_path, "rb")), pickle.load(open(b_path, "rb"))
    print("A:", A["meta"], "\nB:", B["meta"])
    first = None
    for sa, sb in zip(A["trace"], B["trace"]):
        fa, fb = sa["flat"], sb["flat"]
        keys = sorted(set(fa) | set(fb))
        diffs = []
        for k in keys:
            va, vb = fa.get(k), fb.get(k)
            if va is None or vb is None:
                diffs.append((k, "missing", None))
                continue
            if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
                if va.shape != vb.shape:
                    diffs.append((k, "shape", None))
                    continue
                d = np.abs(va - vb)
                if np.nanmax(d) > 0 if d.size else False:
                    diffs.append((k, float(np.nanmax(d)), int(np.nanargmax(d))))
            elif va != vb:
                diffs.append((k, "value", None))
        n_num = sum(1 for d in diffs if isinstance(d[1], float))
        print(f"{sa['tag']}: {len(diffs)} differing leaves ({n_num} numeric)")
        if diffs and first is None:
            first = sa["tag"]
            diffs_sorted = sorted([d for d in diffs if isinstance(d[1], float)], key=lambda d: -d[1])
            for k, v, i in diffs_sorted[:top]:
                print(f"   {k}: max |diff| {v:.3e} at index {i}")
            for k, v, i in [d for d in diffs if not isinstance(d[1], float)][:top]:
                print(f"   {k}: {v}")
    print("FIRST_DIFFERING_SNAPSHOT:", first)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--dataset", default=os.path.join(WT, "datasets", "isg-init-eval"))
    ap.add_argument("--instances", default=os.path.join(WT, "lab", "instances.json"))
    ap.add_argument("--episode", type=int, default=2)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warm-renders", type=int, default=0)
    ap.add_argument("--physics", default=None)
    ap.add_argument("--dummy-bodies", type=int, default=0)
    ap.add_argument("--enhanced-determinism", dest="enhanced_determinism", action="store_true", default=None)
    ap.add_argument("--no-enhanced-determinism", dest="enhanced_determinism", action="store_false")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    assert args.out, "--out is required"
    return record(args)


if __name__ == "__main__":
    sys.exit(main())
