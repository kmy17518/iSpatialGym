"""``sample_isg``: BDDL + task customs list -> N templates (model combinations) per alias, each with its instance 0
(spec §9, artefact layout §1).

Example::

    OMNIGIBSON_HEADLESS=1 python -m ispatialgym.sampling.sample_isg --alias <alias> [--customs task_custom_lists.json] \
        [--scene house_single_floor] [--templates 3] [--instances-per-template 1] [--seed 0] [--out <root>]

Per alias: ``GroundedProblem.load`` (customs entry + grounded BDDL) -> per candidate scene: one OmniGibson world (reused
across the templates of a scene, chairs within 1 m of the support removed as a scene variant) -> the §9 model draw
(``v2.problem.iter_template_choices``: all whitelisted models before reuse, unique across roles, look-alike pairs
rejected, groups per ``generation.groups``) -> per template ``SamplerV2.setup()`` (template facts) and
``run_instances(instances_per_template)``. A template whose setup / instance 0 fails moves the models responsible to
the scene's blacklist with the rejection reason (whitelist pruning); every successful instance appends constraints.
Blacklists and constraints are written back into the source customs file (``--no-write-back`` keeps it untouched) and
the copy under ``<out>/task_custom_lists.json`` is refreshed. The run ends with the ``SAMPLING_REPORT`` stdout line and
``<out>/summary.json`` (``problems.<alias>.<template>.{n_written, time_s, rejections, ...}``).

``--v1-spec <spec.json> --v1-problem <activity>-<def_id> --subtask <id>`` builds the grounded problem from a v1 spec
entry through the facade (``GroundedProblem.from_v1_spec``) until ``ground_isg`` (D1) produces customs entries.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np  # noqa: E402

from ispatialgym import REPO_ROOT  # noqa: E402
from ispatialgym.sampling.v2 import SAMPLER_VERSION  # noqa: E402
from ispatialgym.sampling.v2.problem import (  # noqa: E402
    DEFAULT_CUSTOMS,
    GroundedProblem,
    TemplateChoice,
    iter_template_choices,
    new_run_summary,
)

DEFAULT_OUT = os.path.join(REPO_ROOT, "datasets", "ispatialgym-task-instances-v2")
#: the reach bands B3's fleet writes (``ReachBands.load`` searches ``ISPATIALGYM_REACH_BANDS`` first, then the runlog root
#: and ``~/.cache``); a hit there skips the 5–11 min in-process band computation per (support, object)
FLEET_BANDS = os.path.join(REPO_ROOT, "runlogs", "reach_bands_20260915", "bands")
if os.path.isdir(FLEET_BANDS):
    os.environ.setdefault("ISPATIALGYM_REACH_BANDS", FLEET_BANDS)

log = logging.getLogger("ispatialgym.sampling.v2")


# ----------------------------------------------------------------------------------------------------------------
# world management (scene reuse across templates, as the v1 driver)
# ----------------------------------------------------------------------------------------------------------------


def rooms_for(problem: GroundedProblem, scene: str, stable_scene: dict) -> List[str]:
    """Task rooms = ``scenes.<scene>.rooms`` + the rooms of the bound scene objects (§3.1)."""
    rooms = list(problem.rooms(scene))
    init_info = stable_scene["objects_info"]["init_info"]
    for role, req in problem.bindings(scene).items():
        names = [n for n in req if isinstance(n, str)] if isinstance(req, list) else []
        for n in names:
            if n in init_info:
                for r in init_info[n]["args"].get("in_rooms", []) or []:
                    if r not in rooms:
                        rooms.append(r)
        if isinstance(req, dict) and req.get("room") and req["room"] not in rooms:
            rooms.append(req["room"])
    return rooms


def chairs_to_remove(problem: GroundedProblem, scene: str, stable_scene: dict, rooms: List[str]) -> List[str]:
    """§3.1: table tasks sample in the stable scene minus the chairs within 1.0 m of the support."""
    from ispatialgym.sampling import scene_variants as SV

    req = problem.bindings(scene).get("support")
    names = [n for n in req if isinstance(n, str)] if isinstance(req, list) else []
    if not names or problem.gen("remove_chairs_near_tables", True) is False:
        return []
    tables = SV.tables_in_rooms(stable_scene, rooms, names)
    if not tables:
        return []
    cats = problem.gen("chair_categories") or SV.CHAIR_CATEGORIES
    return SV.chairs_near_tables(stable_scene, rooms, tables, distance=float(problem.gen("chair_distance", 1.0)), categories=tuple(cats))


def world_key(problem: GroundedProblem, scene: str, stable: Optional[dict] = None) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    """The **world key** of an alias in a scene — ``(scene, task rooms, removed chairs)`` — what decides whether two aliases
    can share one loaded world (C5 reschedule: aliases are grouped by this key). Needs only the stable scene json."""
    if stable is None:
        from omnigibson.utils.asset_utils import get_dataset_path  # package import only, no simulator launch

        path = os.path.join(get_dataset_path("2026-challenge-task-instances"), "scenes", scene, "json", f"{scene}_stable.json")
        with open(path) as f:
            stable = json.load(f)
    rooms = rooms_for(problem, scene, stable)
    removed = chairs_to_remove(problem, scene, stable, rooms)
    return scene, tuple(sorted(rooms)), tuple(removed)


def world_key_str(key: Tuple[str, Tuple[str, ...], Tuple[str, ...]]) -> str:
    scene, rooms, removed = key
    return f"{scene}|rooms={','.join(rooms)}|removed={','.join(removed)}"


class WorldPool:
    """One live OmniGibson world at a time, re-created when the world key (scene, task rooms, removed chairs) changes;
    aliases with the same key reuse it (``last_action`` = ``"loaded"`` / ``"reused"`` after :meth:`get`)."""

    def __init__(self, out_root: str):
        self.out_root = out_root
        self.world = None
        self.kit = None
        self.key = None
        self.last_action = None
        self.n_loads = 0
        self.load_time_s = 0.0

    def get(self, problem: GroundedProblem, scene: str):
        from ispatialgym.sampling.robot_utils import RobotKit
        from ispatialgym.sampling.sim_utils import SimWorld, stable_scene_path

        import omnigibson as og

        with open(stable_scene_path(scene)) as f:
            stable = json.load(f)
        key = world_key(problem, scene, stable)
        removed = list(key[2])
        rooms = list(key[1])
        if self.key == key and self.world is not None:
            self.last_action = "reused"
            log.info("world reused for %s: %s", problem.alias, world_key_str(key))
            return self.world, self.kit
        if self.world is not None:
            og.clear()
            self.world, self.kit = None, None
        t0 = time.time()
        variant = None
        if removed:
            from ispatialgym.sampling import scene_variants as SV

            vpath, _ = SV.write_variant(stable_scene_path(scene), removed, os.path.join(self.out_root, "scene_variants"))
            variant = {"path": vpath, "removed_objects": removed, "freed_footprints": SV.freed_footprints(stable, removed)}
            log.info("scene variant for %s: %d objects removed (%s)", scene, len(removed), removed)
        self.world = SimWorld(scene, rooms, scene_variant=variant)
        self.kit = RobotKit(self.world, prebuild=("right_arm",))  # the left-arm planner is built on first use (brace tasks)
        self.key = key
        self.last_action = "loaded"
        self.n_loads += 1
        self.load_time_s += time.time() - t0
        log.info("world loaded for %s in %.1fs: %s", problem.alias, time.time() - t0, world_key_str(key))
        return self.world, self.kit

    def cleanup_template(self):
        if self.world is None:
            return
        for o in list(self.world.spawned.values()):
            self.world.remove(o)
        self.kit.sync_world()


# ----------------------------------------------------------------------------------------------------------------
# one template
# ----------------------------------------------------------------------------------------------------------------


def run_template(problem: GroundedProblem, choice: TemplateChoice, world, kit, out_root: str, n_instances: int, seed: int, max_attempts: int,
                 variant: Optional[str] = None, customs_sha256: Optional[str] = None, start_index: Optional[int] = None) -> Dict[str, Any]:
    """Build the sampler of the alias's subtask for ``choice``, run ``setup()`` + ``run_instances``; returns the
    ``problems.<alias>.<template_id>`` summary entry (``error`` set when setup raised)."""
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2.core import ContextV2, get_sampler

    out_dir = os.path.join(out_root, problem.alias, choice.template_id)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    ctx = ContextV2(problem, choice, world, kit, out_dir, base_seed=seed, customs_sha256=customs_sha256, variant=variant)
    sampler = get_sampler(problem.subtask)(ctx)
    try:
        sampler.setup()
        written = sampler.run_instances(n_instances, max_attempts=max_attempts, start_index=start_index)
        entry = sampler.template_summary(len(written), time.time() - t0)
        entry["written"] = [os.path.relpath(p, out_root) for p in written]
        return entry
    except Reject as e:
        sampler.rejection_log.add(e.reason, str(e), phase="setup")
        entry = sampler.template_summary(0, time.time() - t0)
        entry["error"] = f"setup rejected: {e}"
        entry["reject_reason"] = e.reason
        return entry
    except Exception as e:  # keep going with the next template
        log.error("traceback", exc_info=True)
        entry = sampler.template_summary(0, time.time() - t0)
        entry["error"] = repr(e)
        return entry
    finally:
        try:
            sampler.write_template_meta() if sampler.template_paths is None else None
        except Exception:
            pass


def blacklist_after_failure(problem: GroundedProblem, choice: TemplateChoice, entry: Dict[str, Any]) -> List[str]:
    """§9 whitelist pruning: when a template produced no instance, the model(s) responsible move to the scene's
    blacklist with the rejection reason. The dominant rejection reason decides which role is responsible (grasp /
    placement reasons -> target; partner reasons -> partner; stance / support reasons -> every sampleable model of the
    template, since the pair failed together)."""
    reasons = entry.get("rejections") or {}
    reason = entry.get("reject_reason") or (max(reasons, key=reasons.get) if reasons else "no_instance")
    date = time.strftime("%Y-%m-%d")
    roles: List[str]
    if any(k in reason for k in ("grasp", "target", "place", "pushable")):
        roles = ["target"]
    elif "partner" in reason or "pair" in reason or "inside" in reason or "nextto" in reason:
        roles = ["partner"]
    else:
        roles = list(choice.models)
    out = []
    for role in roles:
        c = choice.models.get(role)
        if c is None:
            continue
        problem.add_blacklist(choice.scene, role, c.category, c.model, f"sample_isg {date}: {reason}")
        out.append(f"{role}={c.category}/{c.model}")
    return out


# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------


def load_problem_from_args(args) -> GroundedProblem:
    if args.v1_spec:
        from ispatialgym.sampling.spec import load_task_spec

        spec_file = load_task_spec(args.v1_spec, check_ready=False)
        pspec = spec_file[args.v1_problem]
        if not args.subtask:
            raise SystemExit("--v1-spec needs --subtask <subtask id>")
        gp = GroundedProblem.from_v1_spec(pspec, args.subtask, alias=args.alias, scene=args.scene)
        gp.customs_path = None
        return gp
    return GroundedProblem.load(args.alias, args.customs, args.bddl_root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alias", action="append", required=True, help="customs alias; repeatable")
    parser.add_argument("--customs", default=DEFAULT_CUSTOMS)
    parser.add_argument("--bddl-root", default=None, help="activity_definitions root of the grounded BDDL (default: assets)")
    parser.add_argument("--scene", default=None, help="restrict to one scene")
    parser.add_argument("--templates", type=int, default=None, help="templates per alias and scene (default: generation.templates_max)")
    parser.add_argument("--instances-per-template", type=int, default=None, help="default: generation.instances_per_template")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--variant", default=None, help="task sub-variant (open / close, ...)")
    parser.add_argument("--no-write-back", action="store_true", help="do not append blacklists / constraints to the source customs file")
    parser.add_argument("--v1-spec", default=None, help="facade: v1 task spec json (until ground_isg lands)")
    parser.add_argument("--v1-problem", default=None, help="facade: <activity>-<def_id> key in --v1-spec")
    parser.add_argument("--subtask", default=None, help="facade: subtask id of the v1 entry")
    parser.add_argument("--world-key", action="store_true", help="print '<alias>\\t<world key>' per alias (no simulator) and exit; the "
                                                                 "scheduler groups aliases with equal keys into one process")
    parser.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout, force=True)
    if args.world_key:
        return print_world_keys(args)
    os.makedirs(args.out, exist_ok=True)
    customs_copy = os.path.join(args.out, "task_custom_lists.json")
    if not args.v1_spec and os.path.exists(args.customs):
        shutil.copy(args.customs, customs_copy)
    import hashlib

    customs_sha = hashlib.sha256(open(args.customs, "rb").read()).hexdigest() if (not args.v1_spec and os.path.exists(args.customs)) else None
    report = new_run_summary(SAMPLER_VERSION, None if args.v1_spec else args.customs, args.out, vars(args))
    report["aliases"] = {}  # per alias: wall time, world action, n_written, error
    pool = WorldPool(args.out)
    t_start = time.time()
    import omnigibson as og

    try:
        for alias in list(args.alias):
            t_alias = time.time()
            arec = report["aliases"].setdefault(alias, {"n_written": 0, "templates": 0, "world": None, "time_s": None, "error": None})
            try:  # a failing alias never aborts the rest of the chunk
                run_alias(args, alias, pool, report, customs_copy, customs_sha)
            except Exception as e:
                log.error("alias %s failed: %r", alias, e, exc_info=True)
                arec["error"] = repr(e)
                report["errors"].append({"alias": alias, "error": repr(e)})
            arec["time_s"] = round(time.time() - t_alias, 2)
            arec["world"] = pool.last_action
            arec["n_written"] = sum(int(e.get("n_written", 0)) for e in report["problems"].get(alias, {}).values())
            arec["templates"] = len(report["problems"].get(alias, {}))
    except BaseException:
        log.error("traceback", exc_info=True)
        raise
    finally:
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        report["wall_time_s"] = round(time.time() - t_start, 2)
        report["world"] = {"n_loads": pool.n_loads, "load_time_s": round(pool.load_time_s, 2), "key": world_key_str(pool.key) if pool.key else None}
        report["n_written"] = sum(int(e.get("n_written", 0)) for temps in report["problems"].values() for e in temps.values())
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        print("SAMPLING_REPORT " + json.dumps(_compact(report), default=str), flush=True)
        og.shutdown()
    # exit code: 0 when at least one instance was written, 2 when every template failed (chains / job runners stop)
    sys.exit(0 if report["n_written"] > 0 else 2)


def run_alias(args, alias: str, pool: WorldPool, report: Dict[str, Any], customs_copy: str, customs_sha: Optional[str]) -> None:
    """All templates of one alias (every candidate scene unless ``--scene``): world from the pool (reused when the world key
    matches), the §9 model draw, one ``run_template`` per choice, whitelist pruning, customs write-back."""
    args.alias = alias
    gp = load_problem_from_args(args)
    scenes = [args.scene] if args.scene else gp.scene_names()
    rng = np.random.default_rng(args.seed)
    n_inst = args.instances_per_template if args.instances_per_template is not None else int(gp.gen("instances_per_template", 5))
    for scene in scenes:
        if scene not in gp.scene_names():
            report["errors"].append({"alias": alias, "scene": scene, "error": "scene not in the customs entry"})
            continue
        try:
            world, kit = pool.get(gp, scene)
        except Exception as e:
            log.error("traceback", exc_info=True)
            report["errors"].append({"alias": alias, "scene": scene, "error": f"scene load failed: {e!r}"})
            pool.world, pool.kit, pool.key = None, None, None
            continue
        exclude = _existing_templates(args.out, gp, scene)
        for ti, choice in enumerate(iter_template_choices(gp, scene, rng, exclude=exclude, max_templates=args.templates)):
            out_dir = os.path.join(args.out, gp.alias, choice.template_id)
            if os.path.exists(os.path.join(out_dir, "template.json")) and not args.overwrite:
                log.info("skip existing template %s", out_dir)
                continue
            if os.path.exists(out_dir) and args.overwrite:
                shutil.rmtree(out_dir)
            entry = run_template(gp, choice, world, kit, args.out, n_inst, args.seed * 7919 + ti, args.max_attempts, args.variant, customs_sha)
            report["problems"].setdefault(gp.alias, {})[choice.template_id] = entry
            if entry.get("n_written", 0) == 0:
                entry["blacklisted"] = blacklist_after_failure(gp, choice, entry)
                log.warning("template %s produced no instance; blacklisted %s", choice.template_id, entry["blacklisted"])
            else:
                log.info("template %s done: %d instances in %.1fs (rejections %s)", choice.template_id, entry["n_written"], entry["time_s"], entry["rejections"])
            pool.cleanup_template()
            if not args.no_write_back and gp.customs_path:
                gp.save_customs()  # §1: sample_isg appends blacklists / constraints to the source customs list
            gp.save_customs(customs_copy)  # the copy under <out> always carries what this run learned


def print_world_keys(args) -> None:
    """``--world-key``: one ``<alias>\\t<scene>|rooms=...|removed=...`` line per (alias, candidate scene) from the customs
    entry and the stable scene json; no simulator. Aliases with equal keys can share one ``sample_isg`` process."""
    for alias in list(args.alias):
        args.alias = alias
        try:
            gp = load_problem_from_args(args)
        except Exception as e:
            print(f"{alias}\tERROR {e!r}")
            continue
        for scene in ([args.scene] if args.scene else gp.scene_names()):
            try:
                print(f"{alias}\t{world_key_str(world_key(gp, scene))}", flush=True)
            except Exception as e:
                print(f"{alias}\tERROR {scene}: {e!r}")


def _existing_templates(out_root: str, gp: GroundedProblem, scene: str) -> List[Dict[str, str]]:
    """Model combinations already produced under ``<out>/<alias>`` (the representative template of ``ground_isg``, or
    an interrupted run) are excluded from the draw (§9)."""
    d = os.path.join(out_root, gp.alias)
    out = []
    if not os.path.isdir(d):
        return out
    for name in os.listdir(d):
        meta = os.path.join(d, name, "template_meta.json")
        if not os.path.exists(meta):
            continue
        try:
            with open(meta) as f:
                m = json.load(f)
            if m.get("scene") != scene:
                continue
            models = (m.get("choice") or {}).get("models") or {}
            out.append({r: f"{c['category']}/{c['model']}" for r, c in models.items()})
        except Exception:
            continue
    return out


def _compact(report: Dict[str, Any]) -> Dict[str, Any]:
    out = {"sampler_version": report["sampler_version"], "n_written": report.get("n_written", 0), "wall_time_s": report.get("wall_time_s"),
           "world": report.get("world"), "aliases": report.get("aliases", {}), "problems": {}, "errors": report.get("errors", [])}
    for alias, temps in report["problems"].items():
        out["problems"][alias] = {t: {k: v for k, v in e.items() if k in ("n_written", "time_s", "rejections", "counters", "error", "blacklisted")} for t, e in temps.items()}
    return out


if __name__ == "__main__":
    main()
