#!/usr/bin/env python3
"""E1 pilot manifests for the v2 pipeline on the fleet (`scripts/runpod/dispatch_samp.py` CMD jobs).

    python scripts/isg/pilot_jobs.py sample  [--customs ispatialgym/assets/task_specs/isg_customs.json] [--subtasks ...]
                                             [--templates 1] [--instances 2] --out runlogs/pilot_<date>/jobs_sample.json
    python scripts/isg/pilot_jobs.py demo    --done runlogs/runpod/<purpose>/done.json --out .../jobs_demo_<pod>.json

``sample``: one job per grounded alias of the customs list — ``sample_isg --alias <alias> --templates T
--instances-per-template N --no-write-back --out $OUT`` (the write-back happens when the results are merged:
``merge_constraints`` below reads every pulled ``task_custom_lists.json`` copy and appends its blacklists /
constraints to the source customs file). ``demo``: for every finished sampling job with instances, one job running
``run_isg --mode oracle`` on each ``<iid>-tro_state.json`` of that job's output (on the pod that holds it, like
``make_demo_jobs.py``). Weights order the queue: disembedding / large-scale first.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CUSTOMS = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "isg_customs.json")
WEIGHT = {"disembedding": 4.0, "inter_object_relation_satisfaction-ontop_large_scale": 4.0, "inter_object_relation_satisfaction-under_large_scale": 4.0,
          "configuration_matching-articulation_open_large_scale": 3.5, "configuration_matching-articulation_close_large_scale": 3.5,
          "configuration_matching-block_design": 3.0, "configuration_matching-attachment": 3.0, "alignment-world_vertical": 3.0,
          "alignment-world_horizontal": 3.0, "alignment-axial": 0.5, "camera_relocalization-obstructed": 2.0}
SAMPLE_CMD = ("cd /root/ispatialgym && source /root/miniforge3/etc/profile.d/conda.sh && conda activate ispatialgym && "
              "OMNIGIBSON_HEADLESS=1 timeout {timeout} python -m ispatialgym.sampling.sample_isg --alias {alias} --templates {templates} "
              "--instances-per-template {instances} --max-attempts {max_attempts} --seed {seed} --no-write-back --out \"$OUT\" --log-level INFO")
DEMO_CMD = ("cd /root/ispatialgym && source /root/miniforge3/etc/profile.d/conda.sh && conda activate ispatialgym && "
            "for tro in /root/samp/{jid}/*/*/instances/*-tro_state.json; do "
            "OMNIGIBSON_HEADLESS=1 timeout 1800 python -m ispatialgym.oracle.run_isg --instance \"$tro\" --mode oracle --out \"$OUT\" --video --video-size 540; "
            "echo \"DEMO_EXIT $? $tro\"; done")


def subtask_of(entry: dict) -> str:
    iddl = entry.get("iddl", "")
    return os.path.splitext(os.path.basename(iddl))[0] if "/" not in iddl else {
        # iddl paths are <family>/<file>.iddl; the subtask id is the alias prefix up to the first category slot
    }.get(iddl, entry.get("subtask") or "")


def _superseded(e: dict) -> bool:
    """D1: a replaced entry carries a top-level ``superseded_by`` (older files: ``fallback_params.superseded_by``); it is dormant."""
    return bool(e.get("superseded_by") or (e.get("fallback_params") or {}).get("superseded_by"))


def load_entries(path: str):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for alias, e in data.items():
        if alias.startswith("_") or not isinstance(e, dict) or "roles" not in e:
            continue
        st = e.get("subtask")
        if not st:
            iddl = e.get("iddl", "")
            fam, fn = iddl.split("/", 1) if "/" in iddl else ("", iddl)
            fn = os.path.splitext(fn)[0]
            st = fam if fam in ("object_scaling", "mental_rotation", "disembedding") and fn == fam else (f"{fam}-{fn}" if fam else fn)
        out[alias] = (st, e)
    return out


def cmd_sample(args):
    entries = load_entries(args.customs)
    jobs = []
    for alias, (st, e) in sorted(entries.items()):
        if args.subtasks and st not in args.subtasks:
            continue
        jid = "v2__" + alias[:150]
        jobs.append({"id": jid, "cmd": SAMPLE_CMD.format(alias=alias, templates=args.templates, instances=args.instances, seed=args.seed, timeout=args.timeout,
                                                        max_attempts=args.max_attempts),
                     "weight": WEIGHT.get(st, 2.0), "subtask": st, "alias": alias, "slot": "representative"})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(jobs, f, indent=1)
    by = collections.Counter(j["subtask"] for j in jobs)
    print(f"{len(jobs)} sample jobs -> {args.out}")
    for k, v in sorted(by.items()):
        print(f"  {k}: {v}")


def cmd_demo(args):
    """Per-pod demo manifests (``jobs_demo_<pod id>.json`` + ``pods_demo.txt``); ``done.json`` names the pod, the fleet
    registry maps the name to the live pod id. Jobs already listed in ``--skip`` manifests are not repeated."""
    done = json.load(open(args.done))
    jobs = [j for j in json.load(open(args.manifest))]
    by_id = {j["id"]: j for j in jobs}
    fleet = json.load(open(os.path.join(REPO, "runlogs", "runpod", "fleet.json")))
    name_to_pid = {i["name"]: pid for pid, i in fleet["pods"].items() if not i.get("terminated_at")}
    skip = set()
    for m in args.skip or []:
        skip |= {j["id"] for j in json.load(open(m))}
    per_pod = collections.defaultdict(list)
    for jid, d in done.items():
        j = by_id.get(jid)
        if j is None or d.get("n_written", 0) <= 0 or ("demo__" + jid) in skip:
            continue
        pid = name_to_pid.get(d["pod"])
        if pid is None:
            print(f"skip {jid}: pod {d['pod']} is gone")
            continue
        per_pod[pid].append({"id": "demo__" + jid, "cmd": DEMO_CMD.format(jid=jid), "weight": 1.0, "subtask": j["subtask"], "alias": j["alias"],
                             "slot": j.get("slot", ""), "source_job": jid, "pod": pid})
    os.makedirs(args.out, exist_ok=True)
    for pid, js in per_pod.items():
        with open(os.path.join(args.out, f"jobs_demo_{pid}.json"), "w") as f:
            json.dump(js, f, indent=1)
        print(f"{pid}: {len(js)} demo jobs")
    with open(os.path.join(args.out, "pods_demo.txt"), "w") as f:
        f.write("\n".join(per_pod) + "\n")
    all_jobs = [j for js in per_pod.values() for j in js]
    with open(os.path.join(args.out, "jobs_demo_all.json"), "w") as f:
        json.dump(all_jobs, f, indent=1)
    print(f"{len(all_jobs)} demo jobs on {len(per_pod)} pods")


#: expected minutes per alias at 10 attempts (pilot medians incl. the per-alias share of the scene load; C5 decision)
MINUTES = {"alignment-axial": 35.0, "alignment-world_vertical": 25.0, "alignment-world_horizontal": 25.0, "camera_relocalization-standard": 15.0,
           "camera_relocalization-obstructed": 15.0, "mental_rotation": 8.0, "object_scaling": 7.0, "disembedding": 20.0,
           "configuration_matching-block_design": 45.0, "configuration_matching-attachment": 30.0,
           "configuration_matching-articulation_open_small_scale": 20.0, "configuration_matching-articulation_close_small_scale": 20.0,
           "configuration_matching-articulation_open_large_scale": 40.0, "configuration_matching-articulation_close_large_scale": 40.0,
           "inter_object_relation_satisfaction-ontop_small_scale": 15.0, "inter_object_relation_satisfaction-under_small_scale": 15.0,
           "inter_object_relation_satisfaction-inside": 20.0, "inter_object_relation_satisfaction-nextto": 15.0,
           "inter_object_relation_satisfaction-touching": 25.0, "inter_object_relation_satisfaction-ontop_large_scale": 30.0,
           "inter_object_relation_satisfaction-under_large_scale": 30.0}
CHUNK_CMD = ("cd /root/ispatialgym && source /root/miniforge3/etc/profile.d/conda.sh && conda activate ispatialgym && "
             "OMNIGIBSON_HEADLESS=1 timeout {timeout} python -m ispatialgym.sampling.sample_isg {aliases} --templates {templates} "
             "--instances-per-template {instances} --max-attempts {max_attempts} --seed {seed} --no-write-back --out \"$OUT\" --log-level INFO")


def world_keys(aliases):
    """``sample_isg``'s WorldPool key per alias without the simulator: (scene, task rooms, removed chairs)."""
    import json as _json

    from ispatialgym.sampling import sample_isg as S
    from ispatialgym.sampling.sim_utils import stable_scene_path
    from ispatialgym.sampling.v2.problem import GroundedProblem

    stables, keys = {}, {}
    for a in aliases:
        try:
            gp = GroundedProblem.load(a)
            sc = gp.scene_names()[0]
            if sc not in stables:
                stables[sc] = _json.load(open(stable_scene_path(sc)))
            rooms = S.rooms_for(gp, sc, stables[sc])
            removed = S.chairs_to_remove(gp, sc, stables[sc], rooms)
            keys[a] = f"{sc}|{','.join(sorted(rooms))}|{','.join(sorted(removed))}"
        except Exception as e:  # noqa: BLE001
            keys[a] = f"unknown|{type(e).__name__}"
    return keys


def cmd_chunks(args):
    """C5 manifests: aliases not complete, grouped by world key, chunked (mixed subtasks, time budget), coverage wave first."""
    entries = load_entries(args.customs)
    progress = json.load(open(args.progress))["aliases"] if args.progress else {}
    requested = args.instances
    complete = {a for a, r in progress.items() if r.get("n_written", 0) >= requested}
    fallback_ids = {a for a, (st, e) in entries.items() if e.get("fallback_of") or _superseded(e)}
    import re as _re

    excl = _re.compile(args.exclude_alias_regex) if args.exclude_alias_regex else None
    todo = [a for a in entries if a not in complete and a not in fallback_ids and (not args.subtasks or entries[a][0] in args.subtasks)
            and not (excl and excl.search(a))]
    keys = world_keys(todo)
    by_st = collections.defaultdict(list)
    for a in todo:
        by_st[entries[a][0]].append(a)
    done_per_st = collections.Counter(entries[a][0] for a in complete if a in entries)
    # coverage wave: up to (coverage - complete) aliases per subtask; prefer aliases that never ran, then zero / crashed / killed
    order = {"never_started": 0, "killed": 1, "partial": 2, "zero": 3, "crashed": 4}
    coverage, rest = [], []
    for st, als in by_st.items():
        als = sorted(als, key=lambda a: order.get(progress.get(a, {}).get("state", "never_started"), 5))
        need = max(0, args.coverage - done_per_st.get(st, 0))
        coverage += als[:need]
        rest += als[need:]

    def chunk(aliases, tag):
        groups = collections.defaultdict(list)
        for a in aliases:
            groups[keys[a]].append(a)
        jobs = []
        for key, als in groups.items():
            # round-robin over subtasks so every chunk mixes subtasks
            per = collections.defaultdict(collections.deque)
            for a in als:
                per[entries[a][0]].append(a)
            queue = []
            while any(per.values()):
                for st in list(per):
                    if per[st]:
                        queue.append(per[st].popleft())
            cur, cur_min = [], 0.0
            for a in queue:
                m = MINUTES.get(entries[a][0], 20.0)
                if cur and (cur_min + m > args.chunk_minutes or len(cur) >= args.chunk_max):
                    jobs.append((key, cur, cur_min))
                    cur, cur_min = [], 0.0
                cur.append(a)
                cur_min += m
            if cur:
                jobs.append((key, cur, cur_min))
        out = []
        for i, (key, als, mins) in enumerate(jobs):
            jid = f"{tag}_{i:03d}__" + "+".join(sorted({entries[a][0].split('-')[-1][:12] for a in als}))[:60]
            out.append({"id": jid, "cmd": CHUNK_CMD.format(aliases=" ".join("--alias " + a for a in als), templates=args.templates, instances=requested,
                                                           max_attempts=args.max_attempts, seed=args.seed, timeout=int(max(1800, mins * 60 * 2.5))),
                        "weight": 10.0 - i * 1e-3, "world_key": key, "aliases": als, "subtasks": sorted({entries[a][0] for a in als}), "minutes": mins,
                        "wave": tag})
        return out

    cov_jobs, rest_jobs = chunk(coverage, "cov"), chunk(rest, "rest")
    os.makedirs(args.out, exist_ok=True)
    for name, js in (("jobs_coverage.json", cov_jobs), ("jobs_rest.json", rest_jobs)):
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(js, f, indent=1)
    with open(os.path.join(args.out, "world_keys.json"), "w") as f:
        json.dump(keys, f, indent=1)
    print(f"{len(todo)} aliases to run ({len(complete)} complete excluded, {len(fallback_ids)} fallbacks deferred); "
          f"coverage {len(coverage)} aliases in {len(cov_jobs)} chunks ({sum(j['minutes'] for j in cov_jobs):.0f} min), "
          f"rest {len(rest)} aliases in {len(rest_jobs)} chunks ({sum(j['minutes'] for j in rest_jobs):.0f} min); world keys: "
          + ", ".join(f"{k.split('|')[1] or k}: {v}" for k, v in collections.Counter(keys.values()).items()))


def cmd_fallbacks(args):
    """Fallback aliases (customs ``fallback_of``) whose primary yielded nothing so far -> one chunk manifest."""
    entries = load_entries(args.customs)
    progress = json.load(open(args.progress))["aliases"]
    # the primary's ``fallbacks`` list is authoritative (D1: an entry with ``fallback_params.superseded_by`` is dormant)
    ranks = collections.defaultdict(list)
    for a, (st, e) in entries.items():
        live = e.get("fallbacks")
        if live:
            for r, fb in enumerate(live, 1):
                if fb in entries and not _superseded(entries[fb][1]):
                    ranks[a].append((r, fb))
    for a, (st, e) in entries.items():  # entries without a ``fallbacks`` list: fall back to fallback_of / fallback_rank
        if e.get("fallback_of") and e["fallback_of"] in entries and not entries[e["fallback_of"]][1].get("fallbacks") \
                and not _superseded(e):
            ranks[e["fallback_of"]].append((int(e.get("fallback_rank", 1)), a))
    todo = []
    for primary, fbs in ranks.items():
        chain = [primary] + [a for _, a in sorted(set(fbs))]
        # the first alias of the chain without a result that has not yielded yet
        for prev, nxt in zip(chain, chain[1:]):
            r = progress.get(prev)
            if r is None or r.get("state") in ("never_started", "killed"):
                break  # the primary / earlier fallback has not run yet
            if r.get("n_written", 0) > 0:
                break  # yielded: no fallback needed
            if progress.get(nxt) is None:
                todo.append(nxt)
                break
    args.subtasks = None
    keys = world_keys(todo)
    jobs = [{"id": f"fb__{a[:110]}", "cmd": SAMPLE_CMD.format(alias=a, templates=args.templates, instances=args.instances, seed=args.seed,
                                                              timeout=args.timeout, max_attempts=args.max_attempts),
             "weight": 5.0, "subtask": entries[a][0], "alias": a, "slot": "fallback", "world_key": keys[a]} for a in todo]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(jobs, f, indent=1)
    print(f"{len(jobs)} fallback jobs -> {args.out}")


def cmd_merge_constraints(args):
    """Append the blacklists / constraints of every pulled ``task_custom_lists.json`` copy to the source customs file."""
    import glob

    with open(args.customs) as f:
        src = json.load(f)
    n_bl = n_c = 0
    for copy in glob.glob(os.path.join(args.pulled_root, "*", "v2__*", "task_custom_lists.json")):
        try:
            d = json.load(open(copy))
        except Exception:
            continue
        for alias, e in d.items():
            if alias.startswith("_") or alias not in src or not isinstance(e, dict):
                continue
            for scene, sd in (e.get("scenes") or {}).items():
                tgt = src[alias].setdefault("scenes", {}).setdefault(scene, {}).setdefault("blacklist", {})
                for role, cats in (sd.get("blacklist") or {}).items():
                    for cat, models in cats.items():
                        for m, why in models.items():
                            if m not in tgt.setdefault(role, {}).setdefault(cat, {}):
                                tgt[role][cat][m] = why
                                n_bl += 1
            have = {json.dumps(c, sort_keys=True) for c in src[alias].get("constraints", [])}
            for c in e.get("constraints", []) or []:
                if json.dumps(c, sort_keys=True) not in have:
                    src[alias].setdefault("constraints", []).append(c)
                    n_c += 1
    with open(args.customs, "w") as f:
        json.dump(src, f, indent=1)
    print(f"merged {n_bl} blacklist entries and {n_c} constraints into {args.customs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--customs", default=CUSTOMS)
    s.add_argument("--subtasks", nargs="*", default=None)
    s.add_argument("--templates", type=int, default=1)
    s.add_argument("--instances", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--timeout", type=int, default=5400)
    s.add_argument("--max-attempts", type=int, default=10, help="sample_isg --max-attempts per instance (pilot: keep non-yielding aliases short)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_sample)
    s = sub.add_parser("demo")
    s.add_argument("--done", required=True, help="done.json of samp_summary.py (job id -> pod, n_written)")
    s.add_argument("--manifest", required=True, help="the sample manifest")
    s.add_argument("--out", required=True, help="directory for the per-pod demo manifests")
    s.add_argument("--skip", nargs="*", default=None, help="earlier demo manifests whose jobs are already dispatched")
    s.set_defaults(fn=cmd_demo)
    s = sub.add_parser("chunks", help="C5: world-key chunks, coverage wave first, complete aliases excluded")
    s.add_argument("--customs", default=CUSTOMS)
    s.add_argument("--progress", default=None, help="runlogs/pilot_<date>/progress.json (states from the ledger)")
    s.add_argument("--subtasks", nargs="*", default=None)
    s.add_argument("--templates", type=int, default=1)
    s.add_argument("--instances", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-attempts", type=int, default=10)
    s.add_argument("--coverage", type=int, default=5, help="templates (aliases) per subtask in the coverage wave")
    s.add_argument("--chunk-minutes", type=float, default=100.0)
    s.add_argument("--chunk-max", type=int, default=6)
    s.add_argument("--exclude-alias-regex", default=None, help="skip aliases matching this regex (e.g. supports whose bands do not exist yet)")
    s.add_argument("--out", required=True, help="directory for jobs_coverage.json / jobs_rest.json / world_keys.json")
    s.set_defaults(fn=cmd_chunks)
    s = sub.add_parser("fallbacks", help="B: fallback aliases whose primary yielded nothing")
    s.add_argument("--customs", default=CUSTOMS)
    s.add_argument("--progress", required=True)
    s.add_argument("--templates", type=int, default=1)
    s.add_argument("--instances", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-attempts", type=int, default=10)
    s.add_argument("--timeout", type=int, default=5400)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_fallbacks)
    s = sub.add_parser("merge-constraints")
    s.add_argument("--customs", default=CUSTOMS)
    s.add_argument("--pulled-root", required=True, help="e.g. runlogs/runpod/pilot")
    s.set_defaults(fn=cmd_merge_constraints)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
