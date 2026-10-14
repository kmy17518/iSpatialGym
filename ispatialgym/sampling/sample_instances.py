"""Generate iSpatialGym task instances (templates + TROs) from a task spec.

Example::

    OMNIGIBSON_HEADLESS=1 python -m ispatialgym.sampling.sample_instances \
        --spec ispatialgym/assets/task_specs/task_custom_lists.json \
        --problem inter_object_relation_satisfaction-0 --n-instances 3 --seed 0

One OmniGibson process handles the listed problems sequentially; templates of the same scene share the
loaded scene and the cuRobo planners.
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

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np  # noqa: E402

from ispatialgym import REPO_ROOT  # noqa: E402
from ispatialgym.sampling import SAMPLER_VERSION  # noqa: E402
from ispatialgym.sampling.spec import ProblemSpec, SceneSpec, load_task_spec  # noqa: E402

DEFAULT_SPEC = os.path.join(REPO_ROOT, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
DEFAULT_OUT = os.path.join(REPO_ROOT, "datasets", "ispatialgym-task-instances")

log = logging.getLogger("ispatialgym.sampling")


def rooms_for(problem: ProblemSpec, scene_spec: SceneSpec, stable_scene: dict) -> list:
    """Rooms to load: the spec's rooms plus the rooms of explicitly named fixed scene objects."""
    rooms = list(scene_spec.rooms)
    init_info = stable_scene["objects_info"]["init_info"]
    for role, req in scene_spec.fixed.items():
        # lists of scene object names; ``wall_regions`` is a list of dicts (surveyed wall segments) and is skipped
        names = [n for n in req if isinstance(n, str)] if isinstance(req, list) else []
        for n in names:
            if n in init_info:
                for r in init_info[n]["args"].get("in_rooms", []) or []:
                    if r not in rooms:
                        rooms.append(r)
        if isinstance(req, dict) and req.get("room") and req["room"] not in rooms:
            rooms.append(req["room"])
    return rooms


def chairs_to_remove(problem: ProblemSpec, scene_spec: SceneSpec, stable_scene: dict, rooms: list) -> list:
    """Chairs around the work table(s) of a table task (``generation.remove_chairs_near_tables``, default on when the
    problem has a fixed ``table`` role; ``distance`` 1.0 m; ``chair_categories`` overrides the default list)."""
    from ispatialgym.sampling import scene_variants as SV

    if "table" not in scene_spec.fixed and problem.gen("remove_chairs_near_tables") is None:
        return []
    if problem.gen("remove_chairs_near_tables", True) is False:
        return []
    req = scene_spec.fixed.get("table")
    if isinstance(req, list):
        wanted = list(req)
    elif isinstance(req, dict):
        wanted = list(req.get("category", []))
    else:
        wanted = []
    tables = SV.tables_in_rooms(stable_scene, rooms, wanted) if wanted else []
    if not tables:
        return []
    cats = problem.gen("chair_categories") or SV.CHAIR_CATEGORIES
    return SV.chairs_near_tables(stable_scene, rooms, tables, distance=float(problem.gen("chair_distance", 1.0)), categories=tuple(cats))


def template_id_for(scene: str, choices: dict) -> str:
    parts = [f"{role}-{c.category}-{c.model}" for role, c in sorted(choices.items())]
    return "__".join([scene] + parts) if parts else scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", default=DEFAULT_SPEC)
    parser.add_argument("--problem", action="append", required=True, help="<activity>-<def_id>; repeatable")
    parser.add_argument("--scene", default=None, help="restrict to one scene model")
    parser.add_argument("--n-instances", type=int, default=None, help="per template (default: spec instances_per_template)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--sub-variant", action="append", default=None, help="task sub-variant override (e.g. block pattern id, 'open'/'close', 'occluded')")
    parser.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout, force=True)
    spec_file = load_task_spec(args.spec)
    os.makedirs(args.out, exist_ok=True)
    shutil.copy(args.spec, os.path.join(args.out, "task_custom_lists.json"))

    # heavy imports after the CLI parsed (omnigibson launch)
    from ispatialgym.sampling.robot_utils import RobotKit
    from ispatialgym.sampling.sim_utils import SimWorld, stable_scene_path
    from ispatialgym.sampling.tasks import get_sampler_class
    from ispatialgym.sampling.tasks.base import SamplerContext

    import omnigibson as og

    report = {"sampler_version": SAMPLER_VERSION, "problems": {}}
    current_world = None
    current_kit = None
    current_key = None
    try:
        for problem_name in args.problem:
            pspec = spec_file[problem_name]
            sampler_cls = get_sampler_class(problem_name)
            scenes = [args.scene] if args.scene else list(pspec.scenes)
            rng = np.random.default_rng(args.seed)
            for scene in scenes:
                scene_spec = pspec.scenes[scene]
                with open(stable_scene_path(scene)) as f:
                    stable = json.load(f)
                rooms = rooms_for(pspec, scene_spec, stable)
                removed = chairs_to_remove(pspec, scene_spec, stable, rooms)
                key = (scene, tuple(sorted(rooms)), tuple(removed))
                if current_key != key:
                    try:
                        if current_world is not None:
                            og.clear()
                        variant = None
                        if removed:
                            from ispatialgym.sampling import scene_variants as SV

                            vpath, _ = SV.write_variant(stable_scene_path(scene), removed, os.path.join(args.out, "scene_variants"), mode="remove")
                            variant = {"path": vpath, "removed_objects": removed, "freed_footprints": SV.freed_footprints(stable, removed)}
                            log.info("scene variant for %s: %d objects removed (%s)", scene, len(removed), removed)
                        current_world = SimWorld(scene, rooms, scene_variant=variant)
                        current_kit = RobotKit(current_world, prebuild=("right_arm",))
                        current_key = key
                    except Exception as e:  # keep going with the next scene
                        traceback.print_exc()
                        log.error("loading %s %s failed: %r", scene, rooms, e)
                        report["problems"].setdefault(problem_name, {})[scene] = {"error": f"scene load failed: {e!r}"}
                        current_world, current_kit, current_key = None, None, None
                        continue
                choices_list = pspec.template_choices(rng)
                # specs with per-template random draws (distractor pools, ``pick: "pool"`` roles) get several templates
                # per model choice (``__v1``, ``__v2``, ...: another distractor set each)
                want = pspec.gen("templates_per_scene")
                if want and (pspec.distractor_spec() is not None or any(r.pick == "pool" for r in pspec.spawned_roles())) and choices_list:
                    base = list(choices_list)
                    while len(choices_list) < int(want):
                        choices_list.append(base[len(choices_list) % len(base)])
                if args.max_templates is not None:
                    choices_list = choices_list[: args.max_templates]
                n_inst = args.n_instances if args.n_instances is not None else int(pspec.gen("instances_per_template", 5))
                seen_ids = {}
                for ti, choices in enumerate(choices_list):
                    template_id = template_id_for(scene, choices)
                    if template_id in seen_ids:
                        seen_ids[template_id] += 1
                        template_id = f"{template_id}__v{seen_ids[template_id]}"
                    else:
                        seen_ids[template_id] = 0
                    out_dir = os.path.join(args.out, problem_name, template_id)
                    if os.path.exists(os.path.join(out_dir, "template.json")) and not args.overwrite:
                        log.info("skip existing template %s", out_dir)
                        continue
                    if os.path.exists(out_dir) and args.overwrite:
                        shutil.rmtree(out_dir)
                    os.makedirs(out_dir, exist_ok=True)
                    t0 = time.time()
                    ctx = SamplerContext(pspec, scene_spec, choices, current_world, current_kit, out_dir, template_id, spec_file.sha256,
                                         base_seed=args.seed * 7919 + ti)
                    sampler = sampler_cls(ctx)
                    if args.sub_variant:
                        sampler.sub_variant_overrides = list(args.sub_variant)
                    try:
                        sampler.setup()
                        written = sampler.run(n_inst, max_attempts=args.max_attempts)
                        report["problems"].setdefault(problem_name, {})[template_id] = {
                            "n_written": len(written), "rejections": sampler.rejections, "time_s": time.time() - t0}
                        log.info("template %s done: %d instances in %.1fs (rejections %s)", template_id, len(written), time.time() - t0, sampler.rejections)
                    except Exception as e:  # keep going with the next template
                        traceback.print_exc()
                        report["problems"].setdefault(problem_name, {})[template_id] = {"error": repr(e), "time_s": time.time() - t0}
                        log.error("template %s failed: %r", template_id, e)
                    finally:
                        sampler.teardown() if hasattr(sampler, "teardown") else None
                        for o in list(current_world.spawned.values()):
                            current_world.remove(o)
                        current_kit.sync_world()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        with open(os.path.join(args.out, f"sampling_report_{int(time.time())}.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        print("SAMPLING_REPORT " + json.dumps(report, default=str), flush=True)
        og.shutdown()


if __name__ == "__main__":
    main()
