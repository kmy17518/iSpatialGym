"""Work units of the fleet evaluation, in waves (the 2026-10-23 11:50 plan).

One (run, task) pair = that task's eval instance + its 10 val instances on one world. Its checkpoints are split into
units that share the output key ``out/<run>__<task>/step-XXXXXXXX/`` (so finished checkpoints are skipped whatever unit
they came from):

  wave 1 (priority 1)   ACT: ``step-00200000``; pi05: ``[5000, <latest 10k-stride step on HF>]`` and, gated on its
                        upload (``requires_hf_step``), ``[100000]`` -- every pair's wave-1 unit is claimed before any
                        wave-2 unit
  wave 2 (priority 2.x) the remaining checkpoints, latest first, in chunks of <= 3 per unit; the priority grows with
                        the chunk's distance from the latest step, so the whole fleet advances together
                        (180k -> 160k -> ... -> 20k) instead of one GPU grinding one task's history

Step budget per episode: max(300, round(1.5 x the demo's length)). Also writes the merged per-task instance lists
``<out>/instances/all/<task>.json`` and the replay-verification units (priority 0).

    python scripts/vec_eval/fleet/make_units.py --out /vision/u/minyeong/projects/isg_home/vec_eval85 [--pi-latest 90000]
"""

from __future__ import annotations

import argparse
import json
import os

HF_USER = "kmy17518"
CHUNK = 3
BUDGET_FACTOR = 1.5


def budget(length: int) -> int:
    return max(300, int(round(BUDGET_FACTOR * int(length))))


def chunks(steps, size):
    steps = sorted(steps, reverse=True)
    return [steps[i:i + size] for i in range(0, len(steps), size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default=None, help="comma list of runs (default all of instances/runs.json)")
    ap.add_argument("--pi-latest", type=int, default=90000, help="the latest 10k-stride pi05 step already on HF (wave 1 with 5k)")
    ap.add_argument("--pi-pending", type=int, nargs="*", default=[100000], help="pi05 steps not yet on HF: their own wave-1 units, gated on the upload")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="checkpoints per wave-2 unit (1 since 2026-10-23 17:20: the 4-hour deadline)")
    ap.add_argument("--units-out", default=None, help="where to write units.json (default <out>/fleet/units.json); a campaign folder's units.json")
    ap.add_argument("--plan", default=None, help="the plan note written into units.json")
    ap.add_argument("--no-verify-units", action="store_true",
                    help="no replay-verification units: the campaign reuses the first campaign's verify/<task>.json records (copied to its prefix)")
    ap.add_argument("--split-n1-verify", default=None, metavar="DIR",
                    help="a dir of verify/<task>.json records: a task verified at N=1 gets one checkpoint per wave-2 unit whatever --chunk "
                         "(its world build is ~6 min, so chunking saves nothing there and only lengthens the campaign's tail)")
    ap.add_argument("--single", type=int, nargs="*", default=[],
                    help="steps of an explicit order that always get their own wave-2 unit, after the chunks (no HF gate): steps added "
                         "to a plan whose chunk units are already claimable, e.g. pi05 90k / 99999 of campaign c2")
    args = ap.parse_args()
    n1_tasks = set()
    if args.split_n1_verify:
        for f in os.listdir(args.split_n1_verify):
            if f.endswith(".json"):
                v = json.load(open(os.path.join(args.split_n1_verify, f)))
                if v.get("pass") and int(v.get("num_envs") or 0) == 1:
                    n1_tasks.add(v.get("task") or f[:-5])
        print(f"{len(n1_tasks)} tasks verified at N=1 get single-checkpoint wave-2 units")
    inst = os.path.join(args.out, "instances")
    runs = json.load(open(os.path.join(inst, "runs.json")))
    os.makedirs(os.path.join(inst, "all"), exist_ok=True)
    units_out = args.units_out or os.path.join(args.out, "fleet", "units.json")
    os.makedirs(os.path.dirname(os.path.abspath(units_out)), exist_ok=True)
    only = set(args.runs.split(",")) if args.runs else None
    units = []
    pairs = 0
    for run, spec in runs["runs"].items():
        if only and run not in only:
            continue
        # the serving checkout: runs.json's code_dir when given (act-piaug for runs with image_augment), else the first campaign's rule
        code_dir = spec.get("code_dir") or ("act-side" if "ddp4" in run else ("openpi-todo1" if spec["kind"] == "openpi" else "act-todo1"))
        steps = sorted(int(s) for s in spec["steps"])
        latest = max(steps)
        pending = [s for s in args.pi_pending if s in steps] if spec["kind"] == "openpi" else []
        if spec.get("order"):
            # an explicit evaluation order (campaign c2): wave 1 = the first step, wave 2 = the rest in that order, one per unit
            order = [int(s) for s in spec["order"]]
            assert sorted(order) == steps, f"{run}: order {order} != steps {steps}"
            wave1 = [[order[0]]]
            rest = order[1:]
        elif spec["kind"] == "openpi":
            wave1 = [[5000, args.pi_latest]] + [[s] for s in pending]
            rest = [s for s in steps if s not in (5000, args.pi_latest) and s not in pending]
        else:
            wave1 = [[latest]]
            rest = [s for s in steps if s != latest]
        for task in spec["tasks"]:
            pairs += 1
            val = json.load(open(os.path.join(inst, "val", task + ".json")))["instances"]
            ev = json.load(open(os.path.join(inst, "eval", task + ".json")))["instances"]
            merged = ev + val  # the eval instance first: the world is built from its (released) template
            all_path = os.path.join(inst, "all", task + ".json")
            all_rec = {"task": task, "task_id": runs["tasks"][task]["task_id"], "instances": merged}
            if os.path.exists(all_path):
                # the merged lists are frozen inputs of the campaigns that used them: never rewrite one that differs
                if json.load(open(all_path)) != all_rec:
                    raise SystemExit(f"{all_path} differs from eval + val of {task}; the instance lists are frozen (rebuild them deliberately)")
            else:
                with open(all_path, "w") as f:
                    json.dump(all_rec, f, indent=1)
            scene = None
            try:
                tro = json.load(open(ev[0]["tro"]))
                scene = ((tro.get("ispatialgym") or tro.get("metadata") or {}).get("scene") or {}).get("scene_model")
            except Exception:  # noqa: BLE001
                pass
            if scene is None:
                scene = ev[0]["tro"].split("/scenes/")[1].split("/")[0] if "/scenes/" in ev[0]["tro"] else None
            budgets = [budget(int(s.get("length") or 0)) for s in merged]
            base = {"run": run, "kind": spec["kind"], "code_dir": code_dir, "task": task, "task_id": runs["tasks"][task]["task_id"], "scene": scene,
                    "hf_repo": f"{HF_USER}/{run}", "instances": f"instances/all/{task}.json", "n_instances": len(merged), "n_val": len(val), "n_eval": len(ev),
                    "max_budget_steps": max(budgets), "sum_budget_steps": sum(budgets), "budget_factor": BUDGET_FACTOR,
                    "out_key": f"{run}__{task}", "requires_verify": task}

            def unit(name, chunk, priority, **extra):
                if extra.get("wave") != 1 and not spec.get("order"):
                    chunk = sorted(chunk, reverse=True)  # wave 1 (and an explicit order) keeps its given order (pi05: 5k first, then the latest)
                if spec["kind"] == "openpi" and "requires_hf_step" not in extra:
                    # claimable once the step is on HF; with an explicit order, also once the run's first step is (so the
                    # order holds while the training's uploads are still landing)
                    gates = sorted({s for s in chunk if s in pending} | ({int(spec["order"][0])} if spec.get("order") and int(spec["order"][0]) in pending else set()))
                    if gates:
                        extra["requires_hf_step"] = gates[0] if len(gates) == 1 else gates
                units.append({"unit": f"{run}__{task}__{name}", **base, "steps": chunk, "priority": priority, "frames": sum(budgets) * len(chunk), **extra})

            for k, chunk in enumerate(wave1):
                extra = {"requires_hf_step": chunk[0]} if (spec["kind"] == "openpi" and chunk[0] in args.pi_pending) else {}
                unit("w1" if k == 0 else f"w1_{chunk[0] // 1000}k", chunk, 1.0, wave=1, **extra)
            if spec.get("order"):
                # the given order, in chunks of --chunk checkpoints that keep it (campaign c2, user 2026-10-24 14:05: one
                # wave-2 unit per pair holding its 4 remaining checkpoints saves a world build per checkpoint); a step still
                # gated on its HF upload gets its own unit after the chunks so a chunk is never blocked by an upload
                open_steps = [s for s in rest if s not in pending and s not in args.single]
                gated = [s for s in rest if s in pending or s in args.single]  # own units after the chunks (a gate only when pending)
                size = 1 if task in n1_tasks else args.chunk
                groups = [open_steps[i:i + size] for i in range(0, len(open_steps), size)]
                for k, chunk in enumerate(groups):
                    name = f"s{chunk[0] // 1000}k" if len(chunk) == 1 else f"c{chunk[0] // 1000}k-{chunk[-1] // 1000}k"
                    unit(name, chunk, round(2.0 + 0.1 * (k + 1), 3), wave=2)
                for k, step in enumerate(gated):
                    unit(f"s{step // 1000}k", [step], round(2.0 + 0.1 * (len(groups) + k + 1), 3), wave=2)
            else:
                for chunk in chunks(rest, args.chunk):
                    # 2.0 + the chunk's distance from the latest step (fraction of it): 180k -> 2.1, 160k.. -> 2.2 ... 20k -> 2.9
                    name = f"s{chunk[0] // 1000}k" if len(chunk) == 1 else f"c{chunk[0] // 1000}k-{chunk[-1] // 1000}k"
                    unit(name, chunk, round(2.0 + (1.0 - chunk[0] / latest), 3), wave=2)
    # the replay verification units: one per eval-dataset instance (the task's), N=1 reference + N=8 same instance in every slot
    verify_units = []
    for task in (sorted({u["task"] for u in units}) if not args.no_verify_units else []):
        ev = json.load(open(os.path.join(inst, "eval", task + ".json")))["instances"]
        L = int(ev[0].get("length") or 1000)
        verify_units.append({"unit": f"verify__{task}", "kind": "replay_verify", "task": task, "task_id": runs["tasks"][task]["task_id"],
                             "scene": next(u["scene"] for u in units if u["task"] == task), "instances": f"instances/eval/{task}.json", "n_instances": 1,
                             "num_envs_ladder": [8, 4, 2], "frames": 9 * L, "demo_length": L, "priority": 0, "steps": [], "run": "verify"})
    units.sort(key=lambda u: (u["priority"], -u["frames"]))
    verify_units.sort(key=lambda u: -u["frames"])
    with open(units_out, "w") as f:
        json.dump({"units": verify_units + units, "plan": args.plan or "2026-10-23 11:50: waves, 1.5x budget, ACT 20k stride, pi05 5k + 10k stride to 100k"}, f, indent=1)
    if verify_units:
        print(f"{len(verify_units)} replay-verification units (priority 0), longest demo {max(u['demo_length'] for u in verify_units)} steps")
    w1 = [u for u in units if u["wave"] == 1]
    w2 = [u for u in units if u["wave"] == 2]
    print(f"{pairs} (run, task) pairs; {len(w1)} wave-1 units ({sum(len(u['steps']) for u in w1)} checkpoints), {len(w2)} wave-2 units "
          f"({sum(len(u['steps']) for u in w2)} checkpoints); episodes {sum(u['n_instances'] * len(u['steps']) for u in units)}; "
          f"budget steps {sum(u['frames'] for u in units) / 1e6:.1f} M; scenes {sorted({u['scene'] for u in units})}")
    by_run = {}
    for u in units:
        by_run.setdefault(u["run"], []).append(u)
    for run, us in by_run.items():
        print(f"  {run}: {len(us)} units, {sum(u['n_instances'] * len(u['steps']) for u in us)} episodes, max budget/batch {max(u['max_budget_steps'] for u in us)}, "
              f"priorities {sorted({u['priority'] for u in us})}")


if __name__ == "__main__":
    main()
