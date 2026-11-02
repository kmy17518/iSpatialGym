#!/usr/bin/env python
"""Campaign rooms: gate units (kind vec_runs) for every task whose candidate room set drops at least one neighbour: on ONE
GPU, in sequence, the task's first three instances of <instances>/all/<task>.json (eval + 2 val) replayed under the template
rooms, under the candidate rooms (--load-rooms <list>) and once more under the template rooms (the host's run-to-run noise
floor), each with lossless frame dumps and videos. Appends the units to units.json (existing units kept) and uploads nothing.

    python make_gate_units.py [--root $ISG_VEC_ROOT] [--campaign <root>/campaigns/rooms] [--candidate <campaign>/summary/rooms_override_candidate.json]
                              [--units <campaign>/units.json] [--all] [--tasks a,b]

The units need the fleet worker's ``vec_runs`` kind (eval_worker.run_cmd_unit of the rooms worktree: several run_isg_vec
invocations on one host, outputs out/<out_key>/<name>/). Without that worker, run the three ``runs`` of a unit by hand with
``run_isg_vec --out <campaign>/gate/<task>/<name> <args>`` on one host (README.md).
"""
import argparse, hashlib, json, os, time

# vec_next (2026-10-29): the rooms dumper's flag is --dump-frames-spec in this checkout ($R: --dump-frames)
DEFAULT_ROOT = os.environ.get("ISG_VEC_ROOT", "/vision/u/minyeong/projects/isg_home/vec_eval85")
COMMON = ["--mode", "replay", "--num-envs", "1", "--dump-frames-spec", "0-60:5,25", "--record-view", "tiled", "--tile-px", "240", "--external-size", "240", "--grid-video"]
# tasks whose instances span two worlds (the eval instance's template is not the val instances'): one run per world group
TWO_WORLDS = {"configuration_matching-attachment-lid-stockpot-desk": [("wg0", ["--world-group", "0", "--limit-batches", "1"]), ("wg1", ["--world-group", "1", "--limit-batches", "2"])]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="the vec_eval root: <root>/instances/all/<task>.json (env ISG_VEC_ROOT)")
    ap.add_argument("--campaign", default=None, help="the campaign folder (default <root>/campaigns/rooms)")
    ap.add_argument("--instances", default=None, help="the instance lists folder (default <root>/instances); the paths written into the units are absolute")
    ap.add_argument("--candidate", default=None, help="analyze_visibility.py's output (default <campaign>/summary/rooms_override_candidate.json)")
    ap.add_argument("--units", default=None, help="the units.json to append to (default <campaign>/units.json; created as {units: [], plan: ''} when missing)")
    ap.add_argument("--all", action="store_true", help="also tasks without dropped rooms (noise-floor only)")
    ap.add_argument("--tasks", default="", help="comma list: only these tasks")
    a = ap.parse_args()
    campaign = a.campaign or os.path.join(a.root, "campaigns", "rooms")
    instances = os.path.abspath(a.instances or os.path.join(a.root, "instances"))
    candidate = a.candidate or os.path.join(campaign, "summary", "rooms_override_candidate.json")
    units_p = a.units or os.path.join(campaign, "units.json")
    cand = json.load(open(candidate))["overrides"]
    only = {t for t in a.tasks.split(",") if t}
    d = json.load(open(units_p)) if os.path.exists(units_p) else {"units": [], "plan": ""}
    have = {u["unit"] for u in d["units"]}
    new = []
    for t, e in sorted(cand.items()):
        if only and t not in only:
            continue
        if not e["dropped"] and not a.all:
            continue
        uid = f"gate__{t}"
        if uid in have:
            continue
        inst = os.path.join(instances, "all", f"{t}.json")
        groups = TWO_WORLDS.get(t, [("", ["--limit-batches", "3"])])
        runs = []
        for mode in ("template", "override", "template_repeat"):
            for gname, gargs in groups:
                args = COMMON + ["--instances", inst] + gargs
                if mode == "override":
                    args = args + ["--load-rooms", ",".join(e["rooms"])]
                runs.append({"name": f"{mode}{'_' + gname if gname else ''}", "args": args})
        n_inst = 3
        L = 0
        for s in json.load(open(inst))["instances"][:n_inst]:
            L += int(s["length"])
        new.append({"unit": uid, "kind": "vec_runs", "task": t, "scene": e["scene"], "runs": runs, "out_key": f"gate/{t}", "n_instances": n_inst,
                    "frames": 3 * L, "priority": 0.5, "steps": [], "run": "rooms-gate", "wave": 0,
                    "rooms": {"template": e["template_rooms"], "override": e["rooms"], "dropped": e["dropped"]}})
    d["units"] += new
    d["units"].sort(key=lambda u: (u.get("priority", 0), -u["frames"]))
    d["plan"] = d.get("plan", "") + f" | gate units added {time.strftime('%Y-%m-%d %H:%M')}: {len(new)} tasks with dropped rooms, 3 runs each on one host"
    os.makedirs(os.path.dirname(os.path.abspath(units_p)), exist_ok=True)
    json.dump(d, open(units_p, "w"), indent=1)
    print(f"{len(new)} gate units added ({sum(u['frames'] for u in new)} frames); {units_p} now {len(d['units'])} units, sha1 {hashlib.sha1(open(units_p, 'rb').read()).hexdigest()[:12]}")
    for u in new:
        print(f"  {u['unit']}: drop {u['rooms']['dropped']}")


if __name__ == "__main__":
    main()
