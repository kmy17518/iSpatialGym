#!/usr/bin/env python3
"""Validate table / work-surface candidates with the instance samplers (final pass of the table pre-selection).

For one scene, every candidate support of ``inventory.json`` (or ``--tables``) is tried as the fixed ``table`` role
of every table-based problem of the test spec, exactly the way ``ispatialgym.sampling.sample_instances`` would
generate instances there: the spec entry is rewritten so that ``scenes.<scene>`` binds ``table`` to that one scene
object (rooms = the object's ``in_rooms``, floor = its first room), then ``TaskSampler.setup()`` (support raster,
stance, view, reachability tables) and ``TaskSampler.run(n)`` (init settle + BEHAVIOR validation, ``:init`` true,
goal false, goal feasibility by teleport + settle) decide. A support is *viable* for a problem when at least one
instance was written.

``--under`` runs ``inter_object_relation_satisfaction-1`` instead: every (partner, table) pair of the scene, with the
partner (bed / sofa) bound as the fixed ``partner`` role.

One OmniGibson process per scene; supports are grouped by room set so that the scene is loaded once per room set.
Results go to ``<out>/results/<scene>__<table>__<problem>.json`` (one file per run, written immediately so that a
crashed process can be resumed with ``--skip-done``); templates / TROs go to ``--work`` (large, outside the repo).

Example::

    OMNIGIBSON_HEADLESS=1 python runlogs/table_selection_20260911/scripts/validate_tables.py --scene Rs_int \
        --out runlogs/table_selection_20260911 --work /tmp/isg_table_sel --n-instances 2 --max-attempts 6 --skip-done
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shutil
import sys
import time
import traceback

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
#: the sampler code + spec to validate with. Default: the repo. ``ISG_TABLE_REPO`` points at a frozen copy of the
#: working tree (``/tmp/isg_table_sel/snapshot``, made with rsync + symlinked runlogs / Behavior-1K) so that a long
#: batch does not pick up edits made to the sampler while it runs (the spec schema changed once mid-run).
REPO = os.environ.get("ISG_TABLE_REPO") or os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

from ispatialgym.sampling.spec import ProblemSpec, SceneSpec, load_task_spec  # noqa: E402

DEFAULT_SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")

#: table-based problems, most generic first
TABLE_PROBLEMS = [
    "inter_object_relation_satisfaction-0",  # ontop: two small objects
    "disembedding-0",  # two trays side by side, destination barely visible
    "configuration_matching-0",  # block design
    "configuration_matching-3",  # attachment
    "world_alignment-0",  # keyboard, front edge in view
    "inter_object_relation_satisfaction-2",  # inside
    "inter_object_relation_satisfaction-3",  # nextto
    "inter_object_relation_satisfaction-4",  # touching
    "configuration_matching-1",  # articulation open (toolbox on the table)
    "configuration_matching-2",  # articulation close
]
#: a problem that is only worth running when another one produced instances on the same support
DEPENDS_ON = {"configuration_matching-2": "configuration_matching-1"}
UNDER_PROBLEM = "inter_object_relation_satisfaction-1"

log = logging.getLogger("table_selection")


def say(msg: str):
    """Progress line on stdout (OmniGibson re-configures logging after import; print survives)."""
    print(f"[table_selection {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------------------------
# candidate selection (from inventory.json)
# ----------------------------------------------------------------------------------------------------------------


def load_inventory(path: str, scene: str):
    with open(path) as f:
        rows = json.load(f)
    return [r for r in rows if r["scene"] == scene]


def candidate_tables(rows, include_low: bool, min_low_top: float = 0.40):
    out = []
    for r in rows:
        if r["kind"] == "under_partner":
            continue
        if not r.get("rooms"):
            continue
        if any(s in r["rooms"] for s in ("garden", "patio", "backyard", "yard", "pool")):
            continue  # outdoor rooms crash the seg renderer (see instance_sampling_spec.md section 8)
        v = r["verdict"]
        if v == "keep" or (v == "high" and r["category"] == "cubicle"):
            out.append(r)
        elif include_low and v == "low" and float(r.get("top_z", 0)) >= min_low_top:
            out.append(r)
    return out


def under_pairs(rows, max_dist: float = 4.0):
    """(partner, table) pairs: bed / sofa partners with a table-like object whose AABB centre is within ``max_dist``."""
    partners = [r for r in rows if r["kind"] == "under_partner"]
    tables = [r for r in rows if r["kind"] != "under_partner" and r["verdict"] in ("keep", "low") and r.get("rooms")
              and float(r.get("top_z", 0)) >= 0.40 and r["category"] not in ("shelf", "bench", "ottoman", "booth")]
    pairs = []
    for p in partners:
        for t in tables:
            d = math.hypot(p["pos_x"] - t["pos_x"], p["pos_y"] - t["pos_y"])
            if d <= max_dist:
                pairs.append((p, t, d))
    pairs.sort(key=lambda x: (x[0]["name"], x[2]))
    return pairs


# ----------------------------------------------------------------------------------------------------------------
# spec rewriting
# ----------------------------------------------------------------------------------------------------------------


def spec_for_table(pspec: ProblemSpec, scene: str, table_row, partner_row=None) -> ProblemSpec:
    """Copy of the problem spec whose only scene entry binds ``table`` (and ``partner``) to the given scene objects."""
    rooms = [r for r in table_row["rooms"].split(";") if r]
    fixed = {"table": [table_row["name"]], "floor": {"room": rooms[0]}}
    if partner_row is not None:
        p_rooms = [r for r in partner_row["rooms"].split(";") if r]
        fixed["partner"] = [partner_row["name"]]
        fixed["floor"] = {"room": p_rooms[0]}
        rooms = p_rooms + [r for r in rooms if r not in p_rooms]
    # keep scene options of the test entry (e.g. table_front_edge) only if it was for the same object
    old = pspec.scenes.get(scene)
    options = {}
    if old is not None and isinstance(old.fixed.get("table"), list) and old.fixed["table"] == [table_row["name"]]:
        options = dict(old.options)
        if old.fixed.get("table_front_edge"):
            fixed["table_front_edge"] = old.fixed["table_front_edge"]
    scene_spec = SceneSpec(scene, rooms, fixed, options)
    new = ProblemSpec(name=pspec.name, alias=pspec.alias, roles=pspec.roles, scenes={scene: scene_spec},
                      generation=copy.deepcopy(pspec.generation), raw=copy.deepcopy(pspec.raw))
    return new


def template_meta_digest(meta: dict) -> dict:
    keys = ("support", "stance", "place_stance", "view_arms", "yaw_adjust_deg", "dest_lateral_m", "dest_side", "source_lateral_offset",
            "n_reachable_cells", "n_grasp_cells", "n_grasp_combos", "n_place_combos", "n_region_cells", "n_visible_cells", "n_cells",
            "n_reachable_strip_cells", "stand_off", "edge", "approach_forward", "distractors", "load_room_instances", "task_rooms",
            "rejections", "n_written", "scene_variant", "init_arms")
    out = {k: meta[k] for k in keys if k in meta}
    t = meta.get("timings", {})
    out["timings"] = {k: round(v, 2) for k, v in t.items() if isinstance(v, (int, float))}
    return out


def instance_digest(out_dir: str) -> list:
    inst_dir = os.path.join(out_dir, "instances")
    out = []
    if not os.path.isdir(inst_dir):
        return out
    for fn in sorted(os.listdir(inst_dir)):
        if not fn.endswith(".summary.json"):
            continue
        with open(os.path.join(inst_dir, fn)) as f:
            s = json.load(f)
        d = {"summary": os.path.join(inst_dir, fn), "wall_time_s": round(s.get("wall_time_s", 0), 1),
             "init_conditions_all_true": (s.get("bddl_at_init") or {}).get("init_conditions_all_true"),
             "goal_false_at_init": not all(c["satisfied"] for c in (s.get("bddl_at_init") or {}).get("goal", []) if "pose_settled" not in c["bddl"]) if s.get("bddl_at_init") else None,
             "goal_true_at_goal": all(c["satisfied"] for c in (s.get("goal_check") or {}).get("goal", []) if "pose_settled" not in c["bddl"]) if s.get("goal_check") else None,
             "base_pose2d": s.get("base_pose2d"), "trunk": s.get("trunk")}
        for k in ("occluded_variant", "grasp", "dest_pixels", "source_pixels", "view"):
            if k in s:
                d[k] = s[k]
        if "initial_occlusion" in s:
            d["occlusion_ratio"] = s["initial_occlusion"].get("ratio")
        out.append(d)
    return out


# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--inventory", default=os.path.join(HERE, "..", "inventory.json"))
    ap.add_argument("--out", default=os.path.join(HERE, ".."), help="where results/ go (small json files)")
    ap.add_argument("--work", default="/tmp/isg_table_sel", help="where templates / TROs go (large)")
    ap.add_argument("--tables", default=None, help="comma separated scene object names (default: inventory candidates)")
    ap.add_argument("--problems", default=None, help="comma separated problem names (default: all table problems)")
    ap.add_argument("--under", action="store_true", help="run the IORS under problem over (partner, table) pairs instead")
    ap.add_argument("--partners", default=None, help="comma separated partner names for --under (default: inventory)")
    ap.add_argument("--n-instances", type=int, default=1)
    ap.add_argument("--max-attempts", type=int, default=6, help="instance attempts per template (disembedding: at least 10)")
    ap.add_argument("--retries", type=int, default=1, help="extra disembedding template seeds (re-drawn distractor set) when a template yields no instance")
    ap.add_argument("--no-early-skip", action="store_true", help="run every problem even when the support has no stance at all")
    ap.add_argument("--include-low", action="store_true", help="also try 'low' supports with top >= 0.40 m")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout, force=True)
    spec_file = load_task_spec(args.spec)
    rows = load_inventory(args.inventory, args.scene)
    by_name = {r["name"]: r for r in rows}
    res_dir = os.path.join(args.out, "results")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)

    # ---- work list: (table_row, partner_row | None, problem)
    jobs = []
    if args.under:
        pairs = under_pairs(rows)
        if args.partners:
            keep = set(args.partners.split(","))
            pairs = [p for p in pairs if p[0]["name"] in keep]
        if args.tables:
            keep = set(args.tables.split(","))
            pairs = [p for p in pairs if p[1]["name"] in keep]
        for p, t, d in pairs:
            jobs.append((t, p, UNDER_PROBLEM))
        say(f"{args.scene}: {len(pairs)} (partner, table) pairs for {UNDER_PROBLEM}")
    else:
        if args.tables:
            tables = [by_name[n] for n in args.tables.split(",")]
        else:
            tables = candidate_tables(rows, args.include_low)
        problems = args.problems.split(",") if args.problems else list(TABLE_PROBLEMS)
        for t in tables:
            for p in problems:
                jobs.append((t, None, p))
        say(f"{args.scene}: {len(tables)} tables x {len(problems)} problems: {[t['name'] for t in tables]}")

    def result_path(t, p, problem):
        tag = f"{args.scene}__{t['name']}" + (f"__under_{p['name']}" if p is not None else "") + f"__{problem}"
        return os.path.join(res_dir, tag + ".json"), tag

    if args.skip_done:
        before = len(jobs)
        remaining = []
        for j in jobs:
            path, tag = result_path(j[0], j[1], j[2])
            if os.path.exists(path):
                continue
            if os.path.exists(path + ".inprogress"):
                # a previous process died (segfault / kill) while running this job: record it, do not retry
                with open(path + ".inprogress") as f:
                    started = f.read().strip()
                with open(path, "w") as f:
                    json.dump({"scene": args.scene, "table": j[0]["name"], "category": j[0]["category"], "model": j[0]["model"],
                               "partner": j[1]["name"] if j[1] else None, "problem": j[2], "status": "crashed",
                               "error": f"process died while running this job (started {started})", "n_written": 0}, f, indent=1)
                os.remove(path + ".inprogress")
                say(f"{tag}: previous process died during this job; recorded as crashed")
                continue
            remaining.append(j)
        jobs = remaining
        say(f"skip-done: {before - len(jobs)} of {before} jobs already have results")
    if not jobs:
        say("nothing to do")
        return

    # ---- group by (room set, scene variant) so that the scene is loaded once per group
    def rooms_of(t, p):
        rooms = [r for r in t["rooms"].split(";") if r]
        if p is not None:
            pr = [r for r in p["rooms"].split(";") if r]
            rooms = pr + [r for r in rooms if r not in pr]
        return tuple(rooms)

    # heavy imports after the CLI parsed (omnigibson launch)
    from ispatialgym.sampling.robot_utils import RobotKit
    from ispatialgym.sampling.sim_utils import Reject, SimWorld, stable_scene_path
    from ispatialgym.sampling.tasks import get_sampler_class
    from ispatialgym.sampling.tasks.base import SamplerContext, TaskSampler

    import omnigibson as og

    # scene variants (chairs around the work table removed) exactly as sample_instances.py decides them; the sampler
    # version in use may not have them yet (older snapshot), in which case nothing is removed
    try:
        from ispatialgym.sampling import scene_variants as SV
        from ispatialgym.sampling.sample_instances import chairs_to_remove
    except ImportError:
        SV = None
        chairs_to_remove = None
    with open(stable_scene_path(args.scene)) as f:
        stable = json.load(f)

    def removed_of(t, p, problem):
        if chairs_to_remove is None:
            return tuple()
        pspec = spec_for_table(spec_file[problem], args.scene, t, p)
        return tuple(chairs_to_remove(pspec, pspec.scenes[args.scene], stable, list(rooms_of(t, p))))

    say(f"sampler code: {REPO} (scene variants {'available' if SV is not None else 'not available'})")
    jobs.sort(key=lambda j: (rooms_of(j[0], j[1]), removed_of(*j), j[0]["name"], j[1]["name"] if j[1] else "",
                             TABLE_PROBLEMS.index(j[2]) if j[2] in TABLE_PROBLEMS else 99))

    world = kit = None
    current_rooms = None  # (rooms, removed chairs) of the loaded world
    passed = {}  # (table, partner) -> set(problems with >= 1 instance)
    precheck_cache = {}
    t_start = time.time()
    try:
        for ji, (t, p, problem) in enumerate(jobs):
            path, tag = result_path(t, p, problem)
            rooms = rooms_of(t, p)
            removed = removed_of(t, p, problem)
            rec = {"scene": args.scene, "table": t["name"], "category": t["category"], "model": t["model"], "rooms": list(rooms),
                   "partner": p["name"] if p else None, "problem": problem, "inventory": {k: t.get(k) for k in ("top_z", "bbox_x", "bbox_y", "bbox_z", "verdict")},
                   "removed_chairs": list(removed), "sampler_code": REPO,
                   "n_requested": args.n_instances, "max_attempts": args.max_attempts, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
            dep = DEPENDS_ON.get(problem)
            if dep is not None and dep not in passed.get((t["name"], p["name"] if p else None), set()):
                dep_path, _ = result_path(t, p, dep)
                dep_ok = False
                if os.path.exists(dep_path):
                    with open(dep_path) as f:
                        dep_ok = json.load(f).get("n_written", 0) > 0
                if not dep_ok:
                    rec.update({"status": "skipped", "reason": f"{dep} produced no instance on this support"})
                    with open(path, "w") as f:
                        json.dump(rec, f, indent=1)
                    say(f"[{ji + 1}/{len(jobs)}] {tag} skipped ({rec['reason']})")
                    continue
            # ---- (re)load the scene for this (room set, scene variant)
            if current_rooms != (rooms, removed):
                try:
                    if world is not None:
                        og.clear()
                    t0 = time.time()
                    variant = None
                    if removed:
                        vpath, _ = SV.write_variant(stable_scene_path(args.scene), list(removed), os.path.join(args.work, "scene_variants"))
                        variant = {"path": vpath, "removed_objects": list(removed), "freed_footprints": SV.freed_footprints(stable, list(removed))}
                    world = SimWorld(args.scene, list(rooms), scene_variant=variant) if variant else SimWorld(args.scene, list(rooms))
                    kit = RobotKit(world, prebuild=("right_arm",))
                    install_seg_refresh(world, kit)
                    current_rooms = (rooms, removed)
                    precheck_cache.clear()  # the precheck depends on the variant (chairs)
                    say(f"scene {args.scene} rooms {list(rooms)} loaded in {time.time() - t0:.1f}s" + (f"; scene variant without {list(removed)}" if removed else ""))
                except Exception as e:
                    # a failed scene load (CUDA OOM while another process allocates, Isaac Sim hiccup) leaves the
                    # simulator in an undefined state: do not record anything, let the wrapper restart the process
                    traceback.print_exc()
                    say(f"scene load failed for {args.scene} {list(rooms)}: {e!r}; exiting for a restart")
                    os._exit(3)
            # ---- cheap shared precheck per support: raster + stance + view (what every table task starts with)
            pk = (t["name"], rooms)
            if pk not in precheck_cache:
                precheck_cache[pk] = precheck_support(world, kit, spec_file, args.scene, t, TaskSampler, SamplerContext, Reject)
                with open(os.path.join(res_dir, f"{args.scene}__{t['name']}__precheck.json"), "w") as f:
                    json.dump({"scene": args.scene, "table": t["name"], "rooms": list(rooms), "removed_chairs": list(removed), **precheck_cache[pk]}, f, indent=1)
                pc = precheck_cache[pk]
                say(f"precheck {t['name']}: z {pc.get('support', {}).get('z')} free cells {pc.get('support', {}).get('n_free_cells')} occupants "
                    f"{pc.get('support', {}).get('occupants')} stance {'ok ' + pc['stance']['edge'] if pc.get('stance') else 'NONE: ' + str(pc.get('stance_error') or pc.get('error'))[:300]}")
            rec["precheck"] = precheck_cache[pk]
            # ---- early skip: the most lenient problem (IORS ontop: four stand-offs, generic top-down grasp) found no
            # stance at all around the support, or no reachable cell on its top from any stance -> the other table
            # tasks (subset of stances, same reach machinery at the same height) cannot work there either. Other IORS
            # failures (placement region, view, partner) are problem specific and do not skip anything
            key = (t["name"], p["name"] if p else None)
            if not args.no_early_skip and problem != TABLE_PROBLEMS[0]:
                first_path, _ = result_path(t, p, TABLE_PROBLEMS[0])
                skip_reason = None
                if os.path.exists(first_path):
                    with open(first_path) as f:
                        first = json.load(f)
                    tags = option_tags(str(first.get("error", "")))
                    # disembedding searches every edge and several lateral offsets itself, so an 'unreachable' verdict
                    # of IORS ontop (one auto-chosen edge per stand-off) does not transfer to it; a support with no
                    # stance at all does
                    allowed = {"no_stance", "stance_collision"} if problem == "disembedding-0" else {"no_stance", "stance_collision", "unreachable"}
                    if first.get("status") == "setup_failed" and (precheck_cache[pk].get("stance") is None or tags <= allowed):
                        what = "no stance around the support" if tags <= {"no_stance", "stance_collision"} else "no stance / no reachable cell on the support top"
                        skip_reason = (f"{TABLE_PROBLEMS[0]} found {what} at any stand-off"
                                       + (" (precheck: no stance either)" if precheck_cache[pk].get("stance") is None else ""))
                if skip_reason:
                    rec.update({"status": "skipped", "reason": skip_reason})
                    with open(path, "w") as f:
                        json.dump(rec, f, indent=1)
                    say(f"[{ji + 1}/{len(jobs)}] {tag} skipped ({rec['reason']})")
                    continue
            # ---- the sampler itself; a template that yields nothing is retried with other seeds (re-drawn distractor
            # sets / layouts) before the support is declared unusable for the problem
            with open(path + ".inprogress", "w") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
            t_job = time.time()
            attempts = []
            # only the disembedding template depends on the seed (distractor set drawn per template); every other
            # sampler already tries ``max_attempts`` layouts per instance, so extra template seeds add little
            is_dis = problem == "disembedding-0"
            max_attempts = max(args.max_attempts, 10) if is_dis else args.max_attempts
            for k in range(1 + (args.retries if is_dis else 0)):
                seed = args.seed + 1000 * k
                a = run_one(problem, t, p, spec_file, args, world, kit, seed, ji, tag + (f"__s{k}" if k else ""), max_attempts,
                            get_sampler_class, SamplerContext, Reject)
                attempts.append(a)
                if a["status"] in ("ok", "ok_other_support", "error"):
                    break
            final = attempts[-1]
            err_l = str(final.get("error", "")).lower()
            corrupt = final["status"] == "error" and ("memory" in err_l or "not a unit quaternion" in err_l or "nan" in err_l.replace("tensorlist", ""))
            if corrupt:
                # GPU memory pressure from a neighbouring process, or a PhysX blow-up that leaves NaN poses in the whole
                # simulator (survives og.clear(); seen in the diner after 'Illegal BroadPhaseUpdateData'): not a property of
                # the support. Restart the process and redo the job; after two restarts on the same job, record the error.
                n_restart = 0
                if os.path.exists(path + ".restarts"):
                    with open(path + ".restarts") as f:
                        n_restart = int(f.read().strip() or 0)
                if n_restart < 2:
                    with open(path + ".restarts", "w") as f:
                        f.write(str(n_restart + 1))
                    if os.path.exists(path + ".inprogress"):
                        os.remove(path + ".inprogress")
                    say(f"[{ji + 1}/{len(jobs)}] {tag}: simulator-state error ({final['error'][:200]}); exiting for a restart ({n_restart + 1}/2)")
                    os._exit(3)
                rec["restarts"] = n_restart
            rec.update({k_: final[k_] for k_ in ("status", "n_written", "rejections", "template_meta", "instances", "work_dir", "bound_table",
                                                  "setup_s", "run_s", "reject_reason", "error") if k_ in final})
            rec["attempts"] = [{k_: a.get(k_) for k_ in ("seed", "status", "n_written", "rejections", "reject_reason", "setup_s", "run_s")} for a in attempts]
            rec["job_s"] = round(time.time() - t_job, 1)
            rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if rec.get("n_written"):
                passed.setdefault(key, set()).add(problem)
            with open(path, "w") as f:
                json.dump(rec, f, indent=1, default=str)
            if os.path.exists(path + ".inprogress"):
                os.remove(path + ".inprogress")
            say(f"[{ji + 1}/{len(jobs)}] {tag} -> {rec['status']} (written {rec.get('n_written', 0)}, rejections {rec.get('rejections')}, "
                f"{len(attempts)} template seed(s), {rec['job_s']:.0f}s; elapsed {(time.time() - t_start) / 60:.0f} min)"
                + (f" :: {rec.get('error', '')[:300]}" if rec["status"] in ("setup_failed", "error") else ""))
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        og.shutdown()


def option_tags(err: str) -> set:
    """The innermost ``[reason]`` tag of every tried option in a sampler ``Reject`` message.

    The samplers join their options with ``; `` (``stand_off 0.45 edge None: [unreachable] no reachable cell ...``);
    nested messages carry several tags, the last one names the actual failure of that option.
    """
    tags = set()
    for part in err.split("; "):
        found = [part[i + 1:part.index("]", i)] for i in range(len(part)) if part[i] == "[" and "]" in part[i:]]
        if found:
            tags.add(found[-1])
    return tags


def install_seg_refresh(world, kit=None):
    """Wrap ``SimWorld.visible_pixels`` so that a placed, visible spawned object whose centre projects into the image
    but that the instance segmentation does not list triggers ``SimWorld.ensure_segmented`` and one re-render.

    Isaac Sim's replicator label map lags behind object spawning / removal (non-deterministic; the sampler already
    applies this remedy to the disembedding target). When several problems run in one process (objects spawned and
    removed between them) the trays of a later disembedding template sometimes read 0 px although they fill the
    image, which rejected the support with 'destination tray visibility out of range (dest 0 px, source 0 px)'.
    """
    import torch as th

    from ispatialgym.sampling.sim_utils import PARK_POSITION

    # more renders before every segmentation read: the batch runs (three simulator processes sharing the GPU) read
    # frames in which freshly (re)placed trays were still missing, consistently within a process, while a single
    # process on the same support saw them (debug_support_view.py); the asynchronous renderer's readback latency is
    # what the 3-render default absorbs on an idle GPU
    orig_seg = world.seg_instance
    min_renders = int(os.environ.get("ISG_TABLE_MIN_RENDERS", "6"))

    def seg_more(n_renders=3):
        return orig_seg(max(int(n_renders), min_renders))

    world.seg_instance = seg_more
    orig = world.visible_pixels

    def refreshed(n_renders=3, exclude_structure=True):
        px = orig(n_renders, exclude_structure)
        try:
            # camera pose from the physics joint state (FK): the sensor prim pose only updates on a full simulator step
            cam = kit.camera_pose(kit.q_current()) if kit is not None else world.camera_pose()
            missing = []
            for o in world.spawned.values():
                if not o.visible or px.get(o.name, 0) > 0:
                    continue
                p = o.get_position_orientation()[0]
                if float(th.linalg.norm(p[:2] - PARK_POSITION[:2])) < 5.0:
                    continue  # parked
                if bool(world.camera_model.inside(cam, o.aabb_center.float().unsqueeze(0), 0.0)[0]):
                    missing.append(o)
            if missing:
                fixed = [o.name for o in missing if world.ensure_segmented(o)]
                px2 = orig(n_renders, exclude_structure)
                say(f"seg refresh: {[o.name for o in missing]} were in view but unlabelled; re-registered {fixed}; "
                    f"pixels before {[px.get(o.name, 0) for o in missing]} after {[px2.get(o.name, 0) for o in missing]}")
                px = px2
        except Exception:
            traceback.print_exc()
        return px

    world.visible_pixels = refreshed


def run_one(problem, t, p, spec_file, args, world, kit, seed, ji, tag, max_attempts, get_sampler_class, SamplerContext, Reject) -> dict:
    """One template of ``problem`` on support ``t`` (partner ``p``): setup + run, exactly like sample_instances.py."""
    pspec = spec_for_table(spec_file[problem], args.scene, t, p)
    scene_spec = pspec.scenes[args.scene]
    rng = np.random.default_rng(seed)
    choices = pspec.template_choices(rng)[0]
    out_dir = os.path.join(args.work, args.scene, tag)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ctx = SamplerContext(pspec, scene_spec, choices, world, kit, out_dir, tag, spec_file.sha256, base_seed=seed * 7919 + ji)
    sampler = get_sampler_class(problem)(ctx)
    rec = {"seed": seed, "template_id": tag}
    t0 = time.time()
    written = []
    try:
        sampler.setup()
        rec["setup_s"] = round(time.time() - t0, 1)
        bound = [b.scene_object_name for b in sampler.bindings.roles.get("table", [])]
        rec["bound_table"] = bound[0] if bound else None
        t1 = time.time()
        for k in range(3):
            try:
                written = sampler.run(args.n_instances, max_attempts=max_attempts)
                break
            except Exception as e:  # a sampler exception in one attempt path (latent bug): re-run the instance loop with other seeds
                if k == 2:
                    raise
                traceback.print_exc()
                rec.setdefault("run_exceptions", []).append(repr(e)[:300])
                sampler.ctx.base_seed += 7919
                say(f"{tag}: exception in the instance loop ({e!r}); retrying the instances with other seeds")
        rec["run_s"] = round(time.time() - t1, 1)
        rec["status"] = "ok" if written else "no_instances"
        if written and rec["bound_table"] not in (None, t["name"]):
            rec["status"] = "ok_other_support"  # e.g. IORS under fell back to the partner's own top
    except Reject as e:
        rec.update({"status": "setup_failed", "reject_reason": e.reason, "error": str(e)[:2000], "setup_s": round(time.time() - t0, 1)})
    except Exception as e:  # keep going with the next job
        traceback.print_exc()
        rec.update({"status": "error", "error": repr(e)[:2000], "setup_s": round(time.time() - t0, 1)})
    finally:
        rec["n_written"] = len(written)
        rec["rejections"] = dict(sampler.rejections)
        rec["template_meta"] = template_meta_digest(sampler.template_meta)
        rec["instances"] = instance_digest(out_dir)
        rec["work_dir"] = out_dir
        try:
            if hasattr(sampler, "teardown"):
                sampler.teardown()
        except Exception:
            traceback.print_exc()
        for o in list(world.spawned.values()):
            try:
                world.remove(o)
            except Exception:
                traceback.print_exc()
        try:
            kit.sync_world()
        except Exception:
            traceback.print_exc()
    return rec


def precheck_support(world, kit, spec_file, scene, table_row, TaskSampler, SamplerContext, Reject):
    """Support raster + canonical stance + viewing trunk with the generic parameters (IORS ontop spec): the checks every
    table task starts with. Returns a small dict (support z / free cells / occupants / edges; stance or failure)."""
    pspec = spec_for_table(spec_file["inter_object_relation_satisfaction-0"], scene, table_row)
    ctx = SamplerContext(pspec, pspec.scenes[scene], {}, world, kit, "/tmp", "precheck", spec_file.sha256, base_seed=0)
    s = TaskSampler(ctx)
    out = {}
    t0 = time.time()
    try:
        world.park_robot(world.robot_park_xy, floor_z=world.floor_z)
        kit.sync_world()
        table = world.obj(table_row["name"])
        lo, hi = table.aabb
        out["aabb"] = [[round(float(v), 3) for v in lo], [round(float(v), 3) for v in hi]]
        top = s.support_top(table)
        out["support"] = top.to_json()
        out["edges"] = [{"name": n, "mid": [round(float(v), 3) for v in mid], "length": round(float(length), 3)} for n, mid, _, length in top.edges()]
        if int(top.free.sum()) < 50:
            out["stance"] = None
            out["stance_error"] = "too little free top surface"
        else:
            try:
                st = s.stance_with_view(top, 0.25)
                out["stance"] = st.to_json()
                out["view_arms"] = s.template_meta.get("view_arms")
            except Reject as e:
                out["stance"] = None
                out["stance_error"] = str(e)[:1200]
    except Exception as e:
        traceback.print_exc()
        out["error"] = repr(e)[:800]
    out["precheck_s"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    main()
