"""``multiply_isg``: one template -> N instances (spec §9); ``--split train``: the train instances of eval templates.

Example::

    OMNIGIBSON_HEADLESS=1 python -m ispatialgym.sampling.multiply_isg --template <out>/<variant>/<template_id> --instances 5 \
        [--path-buckets 3:10,5:10,10:5] [--seed 1 | --split eval] [--customs task_custom_lists.json]

Reads ``template_meta.json`` (variant, scene, model choice, customs path), rebuilds the world of the template (same rooms
and scene variant), runs the sampler's ``setup()`` (template facts, cached) and ``run_instances(N)`` continuing the
instance numbering after the existing instances (instance 0 exists from ``sample_isg``). Per instance the per-instance
parts of the rules run (draws, exact IK, transitions, paths) with the customs constraints as priors; camera
relocalization fills the path buckets (``--path-buckets max_m:n,...`` overrides ``generation.path_buckets``); every
success appends constraints (written back to the customs source unless ``--no-write-back``); rejections are counted per
reason in ``summary.json`` (``problems.<variant>.<template>.{n_written, time_s, rejections}``) and the run ends with the
``SAMPLING_REPORT`` stdout line. A template sampled in a room its customs entry does not list (``sample_isg --room`` /
``--scene-from-split``) is continued in that room.

Train policy (owner 2026-10-11)::

    python -m ispatialgym.sampling.multiply_isg --split train --template <eval template> [--template ...] --instances N \
        --out <train root> --eval-run <eval run / index> [--start-index G] [--only-scene S [--only-room R]] [--plan]

* seeds per variant and split: ``split_seeds.split_seed("train", variant, --seed-index)`` (``--seed`` overrides), never an eval
  seed; the attempt seeds run over the train instance index, so every train instance has its own stream;
* train instances never go into the eval run: ``--out`` (required; not an eval template's root) gets
  ``<out>/<variant>/<template_id>/instances``, one template per (scene, room), created by its first instance and continued by
  later runs;
* a train instance must not repeat an eval instance of its task: with ``--eval-run`` (eval run directories, TRO files or
  ``instance_separation`` index files) an instance whose start and goal poses are all within ``--repeat-pos-m`` (0.25 m) and
  ``--repeat-ang-deg`` (15 deg) of an eval instance of the same task and scene is removed and its attempt rejected
  (``repeats_eval``, ``instance_separation.find_repeat``); similar instances are kept;
* no distance rule among train instances (the eval 1 m separation does not apply to them);
* camera relocalization: the path buckets fill by the sampler's deficit draw in the train template (``--path-buckets`` /
  ``generation.path_buckets``); object scaling: the start and goal depths are drawn as the sampler draws them (toward or away
  at random); disembedding: the destination side is drawn per template (the eval variants' ``destination_side`` is dropped);
* object scaling / mental rotation: every train instance draws its own (scene, room), ``draw_scene_room("train", variant, g)``
  for train instance g, and goes to the template of that scene / room with the eval template's models;
  ``--keep-template-scene`` keeps the eval template's scene and room. The draw picks one room among the eligible rooms of all
  candidate scenes (``split_seeds.eligible_rooms``: any room but an outdoor one, owner 2026-10-11 13:31, with at least 1 m^2 of
  robot-reachable floor), weighted by its reachable floor capped at 3x the median eligible room (owner 2026-10-11 ~13:50).

The train instance indices g continue after the variant's train instances under ``--out``; ``--start-index`` fixes them (give
the same value to processes that split one plan by world with ``--only-scene`` / ``--only-room``). ``--plan`` prints the plan
(``PLAN <variant> <g> <scene> <room> <template dir>``) without the simulator. The instances of one (scene, room) run in one world,
shared by every template of the process.
"""

from __future__ import annotations

import argparse
import glob
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

from ispatialgym.sampling import instance_separation as SEP  # noqa: E402
from ispatialgym.sampling import split_seeds as SS  # noqa: E402
from ispatialgym.sampling import support_rules as SR  # noqa: E402
from ispatialgym.sampling.sample_isg import FLEET_BANDS  # noqa: E402,F401  (sets ISPATIALGYM_REACH_BANDS to the fleet bands when present)
from ispatialgym.sampling.v2 import SAMPLER_VERSION  # noqa: E402
from ispatialgym.sampling.v2.problem import GroundedProblem, TemplateChoice, new_run_summary  # noqa: E402

log = logging.getLogger("ispatialgym.sampling.v2")


#: customs generation keys that only steer eval instances (dropped for --split train; IORS small ``partner_side``: train draws the side;
#: disembedding ``destination_side``: train draws it per template)
EVAL_ONLY_GENERATION_KEYS = ("requested_bucket_m", "requested_direction", "start_depth_band", "partner_side", "destination_side")


def prepare_train_generation(gp) -> None:
    """Keep a campaign's validated tray side while applying ordinary train draws.

    Campaign validation and goal groups also use the train execution driver.
    Their explicitly chosen tray side is part of the feasible template layout,
    so dropping it can rebuild a different, unreachable scene before any draw.
    Legacy training still draws its destination side; all other train keys keep
    their existing behavior.
    """
    from ispatialgym.sampling import campaign_runtime

    preserve_side = False
    if campaign_runtime.active():
        _, template, _, _ = campaign_runtime.configuration()
        preserve_side = template.task == 'disembedding' and gp.subtask == 'disembedding'
    for key in EVAL_ONLY_GENERATION_KEYS:
        if key == 'destination_side' and preserve_side:
            continue
        gp.generation.pop(key, None)


def parse_path_buckets(text: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """``"3:10,5:10,10:5"`` -> ``[{"max_m": 3, "n": 10}, ...]`` (§2.3 ``generation.path_buckets``)."""
    if not text:
        return None
    out = []
    for part in text.split(","):
        m, n = part.split(":")
        out.append({"max_m": float(m), "n": int(n)})
    return out


def load_template(template_dir: str) -> Dict[str, Any]:
    from ispatialgym.oracle.sidecar import from_legacy_keys

    with open(os.path.join(template_dir, "template_meta.json")) as f:
        return from_legacy_keys(json.load(f))


def problem_for_template(meta: Dict[str, Any], customs: Optional[str], bddl_root: Optional[str]) -> GroundedProblem:
    """The grounded problem of the template: from the customs file (``--customs``, else the one recorded in the meta,
    else the copy next to the variant directory)."""
    variant = meta["variant"]
    template_dir = meta.get("_dir")
    candidates = [customs, meta.get("customs")]
    if template_dir:
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(template_dir))), "task_custom_lists.json"))
    for c in candidates:
        if c and os.path.exists(c):
            try:
                return GroundedProblem.load(variant, c, bddl_root)
            except Exception as e:  # the copy may lack the grounded BDDL (v1 facade run): fall through
                log.warning("customs %s: %s", c, e)
                with open(c) as f:
                    data = json.load(f)
                if variant in data:
                    gp = GroundedProblem(variant, data[variant], customs_path=c, bddl_root=bddl_root, subtask=meta.get("subtask"))
                    return gp
    raise SystemExit(f"no customs entry for variant {variant!r} (tried {candidates})")


def choice_for_template(gp: GroundedProblem, meta: Dict[str, Any]) -> TemplateChoice:
    """The template's model choice; a floor-only template's single room is (re)applied to the problem, so a room the customs entry
    does not list (``sample_isg --room`` / ``--scene-from-split``) is sampled again (``GroundedProblem.override_scene_room``). A
    template bound to a support the support rule of its subtask excludes (``support_rules``) is refused."""
    choice = TemplateChoice.from_json(meta["choice"])
    req = choice.bindings.get("support")
    _, excluded = SR.split_supports(gp.subtask, choice.scene, [n for n in req if isinstance(n, str)] if isinstance(req, list) else [])
    if excluded:
        raise SystemExit(f"{gp.variant}: template {choice.template_id} is bound to supports the support rules exclude ({excluded}); "
                         f"re-sample it on an admitted support")
    if len(choice.rooms) == 1 and gp.floor_only():
        gp.override_scene_room(choice.scene, choice.rooms[0], note=f"multiply_isg: template {choice.template_id}")
    return choice


def choice_in_room(gp: GroundedProblem, choice: TemplateChoice, scene: str, room: Optional[str]) -> TemplateChoice:
    """``choice``'s models in ``scene`` / ``room`` (a floor-only problem's scene block overridden to the room); ``choice`` itself when
    no room is given or the problem has other scene roles."""
    if room is None or not gp.floor_only():
        return choice
    gp.override_scene_room(scene, room, note="multiply_isg --split train")
    return TemplateChoice(scene, dict(choice.models), {r: list(ms) for r, ms in choice.groups.items()}, gp.bindings(scene), gp.rooms(scene),
                          gp.room_tag(scene))


def next_index(template_dir: str) -> int:
    """One past the largest instance index under ``<template_dir>/instances`` (0 when none; robust to gaps)."""
    best = -1
    for p in glob.glob(os.path.join(template_dir, "instances", "*-tro_state.json")):
        tail = os.path.basename(p)[: -len("-tro_state.json")].rsplit("__", 1)[-1]
        if tail.isdigit():
            best = max(best, int(tail))
    return best + 1


def next_train_index(out_root: str, variant: str) -> int:
    """One past the largest instance index of the variant's templates under ``out_root`` (train indices are per variant)."""
    return max([0] + [next_index(d) for d in glob.glob(os.path.join(out_root, variant, "*")) if os.path.isdir(d)])


def template_seed(args, variant: str) -> Tuple[int, Dict[str, Any]]:
    """``--seed``, else ``split_seed(--split, variant, --seed-index)``, else 1; with its record."""
    if args.seed is not None:
        return int(args.seed), {"split": args.split, "index": None, "seed": int(args.seed), "seed_source": "--seed"}
    if args.split:
        s = SS.split_seed(args.split, variant, args.seed_index)
        return s, {"split": args.split, "index": int(args.seed_index), "seed": s, "seed_source": "split"}
    return 1, {"split": None, "index": None, "seed": 1, "seed_source": "default"}


def train_plan(variant: str, choice: TemplateChoice, n: int, start: int, scene_draw: bool) -> List[Tuple[int, str, Optional[str]]]:
    """``(g, scene, room)`` of the ``n`` train instances from index ``start``: with ``scene_draw`` each draws its own (scene, room)
    (``draw_scene_room("train", variant, g)``), otherwise all stay in the template's scene and room."""
    room = choice.rooms[0] if len(choice.rooms) == 1 else None
    out: List[Tuple[int, str, Optional[str]]] = []
    for g in range(int(start), int(start) + int(n)):
        scene, r = SS.draw_scene_room("train", variant, g) if scene_draw else (choice.scene, room)
        out.append((g, scene, r))
    return out


def _under(path: str, root: str) -> bool:
    path, root = os.path.abspath(path), os.path.abspath(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def log_fast_switches() -> None:
    """train-fast 2026-10-31: the ``ISPATIALGYM_FAST*`` switches set in the environment (``ispatialgym.fastflags``: every switch is on
    unless ``=0``; ``ISPATIALGYM_FAST=0`` turns all of them off) and the cuRobo mesh store setting, once at startup, so a job log
    shows its configuration before the first path logs itself."""
    fast = {k: v for k, v in sorted(os.environ.items()) if k.startswith("ISPATIALGYM_FAST")}
    log.info("fast switches set in the environment: %s (an unset switch is on; each path logs itself when first consulted)",
             " ".join(f"{k}={v}" for k, v in fast.items()) or "none")
    cache = os.environ.get("ISPATIALGYM_SAMPLER_MESH_CACHE", "").strip()
    log.info("cuRobo mesh store: ISPATIALGYM_SAMPLER_MESH_CACHE=%s", cache or "unset (cuRobo's default 8192; 'auto' = the world's collision meshes + headroom)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", required=True, action="append", help="<out>/<variant>/<template_id> directory (several: --split train)")
    parser.add_argument("--instances", type=int, required=True, help="instances to add per template (instance 0 exists)")
    parser.add_argument("--path-buckets", default=None, help="camera relocalization: max_m:n,... (overrides generation.path_buckets)")
    parser.add_argument("--seed", type=int, default=None, help="default: split_seeds.split_seed(--split, variant, --seed-index) with --split, else 1")
    parser.add_argument("--split", choices=SS.SPLITS, default=None, help="per-variant split seeds; train: the train policy (module docstring)")
    parser.add_argument("--seed-index", type=int, default=0, help="k of the split seed")
    parser.add_argument("--out", default=None, help="root the instances go to (default: the template's root; required with --split train)")
    parser.add_argument("--eval-run", action="append", default=[],
                        help="--split train: eval instances a train instance must not repeat (run directory, TRO file or instance index file); repeatable")
    parser.add_argument("--repeat-pos-m", type=float, default=SEP.REPEAT_POS_M, help="train-repeat rule: position tolerance (m)")
    parser.add_argument("--repeat-ang-deg", type=float, default=SEP.REPEAT_ANG_DEG, help="train-repeat rule: angle tolerance (deg)")
    parser.add_argument("--start-index", type=int, default=None, help="--split train: first train instance index (default: after the variant's under --out)")
    parser.add_argument("--keep-template-scene", action="store_true", help="--split train, held-object variants: no per-instance (scene, room) draw")
    parser.add_argument("--only-scene", default=None, help="--split train: only the planned instances in this scene")
    parser.add_argument("--only-room", default=None, help="--split train: only the planned instances in this room")
    parser.add_argument("--plan", action="store_true", help="--split train: print the plan and exit (no simulator)")
    parser.add_argument("--customs", default=os.environ.get("ISG_CUSTOMS") or None, help="customs JSON (default: $ISG_CUSTOMS, else the template's copy)")
    parser.add_argument("--bddl-root", default=None)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--no-write-back", action="store_true")
    parser.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)
    train = args.split == "train"
    if not train and (len(args.template) > 1 or args.eval_run or args.plan or args.only_scene or args.only_room or args.start_index is not None):
        parser.error("several --template, --eval-run, --plan, --only-scene / --only-room and --start-index need --split train")
    if train and not args.out:
        parser.error("--split train needs --out (the train root; never the eval run)")

    from ispatialgym.sampling.sample_isg import configure_logging

    configure_logging(args.log_level)
    log_fast_switches()
    # i7-memory (owner 2026-11-08 part C): the per-alias outcome memory ($ISPATIALGYM_ALIAS_MEMORY) steers TRAIN generation only - never an
    # eval instance (alias_memory.enable_for_split: any other split leaves it off whatever the environment says)
    from ispatialgym.sampling import alias_memory as AM

    AM.enable_for_split(args.split)
    if train:
        return main_train(args)
    template_dir = os.path.abspath(args.template[0])
    meta = load_template(template_dir)
    meta["_dir"] = template_dir
    out_root = os.path.abspath(args.out) if args.out else os.path.dirname(os.path.dirname(template_dir))
    gp = problem_for_template(meta, args.customs, args.bddl_root)
    buckets = parse_path_buckets(args.path_buckets)
    if buckets is not None:
        gp.generation["path_buckets"] = buckets
    choice = choice_for_template(gp, meta)
    if choice.template_id != os.path.basename(template_dir):
        log.warning("template id of the meta (%s) differs from the directory (%s)", choice.template_id, os.path.basename(template_dir))
    seed, seed_rec = template_seed(args, gp.variant)
    target_dir = os.path.join(out_root, gp.variant, choice.template_id)
    report = new_run_summary(SAMPLER_VERSION, gp.customs_path, out_root, vars(args))
    import omnigibson as og

    from ispatialgym.sampling.sample_isg import WorldPool, run_template

    pool = WorldPool(out_root)
    try:
        world, kit = pool.get(gp, choice.scene)
        existing = next_index(target_dir)
        entry = run_template(gp, choice, world, kit, out_root, args.instances, seed * 7919 + existing, args.max_attempts,
                             None, start_index=existing, meta_extra={"split": seed_rec} if args.split else None)
        entry["start_index"] = existing
        entry["split"] = seed_rec
        report["problems"].setdefault(gp.variant, {})[choice.template_id] = entry
        pool.cleanup_template()
        if not args.no_write_back and gp.customs_path:
            gp.save_customs()  # §9: every successful instance appends its constraints to the source customs list
        copy = os.path.join(out_root, "task_custom_lists.json")
        if not gp.customs_path or os.path.abspath(copy) != os.path.abspath(gp.customs_path):
            gp.save_customs(copy)  # the copy under <root> always carries what this run learned
    except BaseException:
        log.error("traceback", exc_info=True)
        raise
    finally:
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        report["n_written"] = sum(int(e.get("n_written", 0)) for temps in report["problems"].values() for e in temps.values())
        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, f"multiply_summary_{int(time.time())}.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        print("SAMPLING_REPORT " + json.dumps({"sampler_version": SAMPLER_VERSION, "n_written": report["n_written"],
                                              "problems": {a: {t: {k: v for k, v in e.items() if k in ('n_written', 'time_s', 'rejections', 'error')}
                                                               for t, e in ts.items()} for a, ts in report["problems"].items()}}, default=str), flush=True)
        og.shutdown()
    sys.exit(0 if report["n_written"] > 0 else 2)


def main_train(args) -> int:
    """``--split train`` (module docstring): plan every template's train instances, group them by (scene, room) and run each group
    in one world."""
    out_root = os.path.abspath(args.out)
    buckets = parse_path_buckets(args.path_buckets)
    jobs: List[Dict[str, Any]] = []
    for t in args.template:
        template_dir = os.path.abspath(t)
        eval_root = os.path.dirname(os.path.dirname(template_dir))
        if _under(out_root, eval_root) or _under(eval_root, out_root):
            raise SystemExit(f"--out {out_root} and the eval template's root {eval_root} overlap; train instances never go into the eval run")
        meta = load_template(template_dir)
        meta["_dir"] = template_dir
        if any(j["gp"].variant == meta["variant"] for j in jobs):
            raise SystemExit(f"two templates of {meta['variant']}: one --template per variant (the train indices are per variant)")
        gp = problem_for_template(meta, args.customs, args.bddl_root)
        if buckets is not None:
            gp.generation["path_buckets"] = buckets
        # the eval-only draws of the customs entry (owner 2026-10-11): the eval camera bucket and object-scaling direction / start
        # band; train fills the buckets by the deficit draw and draws the direction at random (IORS small, owner 2026-10-12: the
        # partner side among the sides with a start cell; disembedding: the destination side per template)
        prepare_train_generation(gp)
        choice = choice_for_template(gp, meta)
        seed, seed_rec = template_seed(args, gp.variant)
        scene_draw = SS.scene_draw_applies(gp.subtask) and not args.keep_template_scene
        start = args.start_index if args.start_index is not None else next_train_index(out_root, gp.variant)
        plan = [p for p in train_plan(gp.variant, choice, args.instances, start, scene_draw)
                if (args.only_scene is None or p[1] == args.only_scene) and (args.only_room is None or p[2] == args.only_room)]
        jobs.append({"gp": gp, "meta": meta, "choice": choice, "dir": template_dir, "seed": seed, "seed_rec": seed_rec, "scene_draw": scene_draw,
                     "start": start, "plan": plan})
    groups: Dict[Tuple[str, str], List[Tuple[Dict[str, Any], List[int]]]] = {}
    for job in jobs:
        by_room: Dict[Tuple[str, str], List[int]] = {}
        for g, scene, room in job["plan"]:
            by_room.setdefault((scene, room or ""), []).append(g)
        for key, idxs in by_room.items():
            groups.setdefault(key, []).append((job, sorted(idxs)))
    if args.plan:
        for job in jobs:
            for g, scene, room in job["plan"]:
                tid = choice_in_room(job["gp"], job["choice"], scene, room).template_id
                print(f"PLAN\t{job['gp'].variant}\t{g}\t{scene}\t{room or ''}\t{os.path.join(out_root, job['gp'].variant, tid)}")
        for (scene, room), members in sorted(groups.items()):
            print(f"WORLD\t{scene}\t{room}\t{sum(len(i) for _, i in members)} instances\t{len(members)} variants")
        return 0
    notes: List[str] = []
    eval_recs = SEP.load_instances(args.eval_run, notes=notes) if args.eval_run else []
    for n in notes:
        log.info("eval instances: %s", n)
    if not args.eval_run:
        log.warning("--split train without --eval-run: the train-repeat rule is off")
    report = new_run_summary(SAMPLER_VERSION, args.customs, out_root, vars(args))
    report["train"] = {"eval_instances": len(eval_recs), "repeat_pos_m": args.repeat_pos_m, "repeat_ang_deg": args.repeat_ang_deg,
                       "plan": {j["gp"].variant: [[g, s, r] for g, s, r in j["plan"]] for j in jobs},
                       "seeds": {j["gp"].variant: j["seed_rec"] for j in jobs}}
    import omnigibson as og

    from ispatialgym.sampling.sample_isg import WorldPool, run_template

    pool = WorldPool(out_root)
    try:
        for (scene, room), members in sorted(groups.items()):
            for job, idxs in members:
                gp = job["gp"]
                choice = choice_in_room(gp, job["choice"], scene, room or None)
                try:
                    world, kit = pool.get(gp, scene)
                except Exception as e:
                    log.error("traceback", exc_info=True)
                    report["errors"].append({"variant": gp.variant, "scene": scene, "room": room, "error": f"scene load failed: {e!r}"})
                    pool.world, pool.kit, pool.key = None, None, None
                    continue
                filt = SEP.repeat_filter(eval_recs, task=gp.activity, pos_m=args.repeat_pos_m, ang_deg=args.repeat_ang_deg) if eval_recs else None
                split = {**job["seed_rec"], "scene": scene, "room": room or None, "scene_source": "split" if job["scene_draw"] else "template",
                         "source_template": job["dir"]}
                entry = run_template(gp, choice, world, kit, out_root, len(idxs), job["seed"] * 7919, args.max_attempts, None,
                                     indices=idxs, instance_filter=filt, meta_extra={"split": split})
                entry["indices"] = idxs
                entry["split"] = split
                report["problems"].setdefault(gp.variant, {})[choice.template_id] = entry
                pool.cleanup_template()
                if not args.no_write_back and gp.customs_path:
                    gp.save_customs()
                gp.save_customs(os.path.join(out_root, "task_custom_lists.json"))
    except BaseException:
        log.error("traceback", exc_info=True)
        raise
    finally:
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        report["n_written"] = sum(int(e.get("n_written", 0)) for temps in report["problems"].values() for e in temps.values())
        os.makedirs(out_root, exist_ok=True)
        with open(os.path.join(out_root, f"multiply_summary_{int(time.time())}.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        print("SAMPLING_REPORT " + json.dumps({"sampler_version": SAMPLER_VERSION, "n_written": report["n_written"], "errors": report["errors"],
                                              "problems": {a: {t: {k: v for k, v in e.items() if k in ('n_written', 'time_s', 'rejections', 'error', 'indices')}
                                                               for t, e in ts.items()} for a, ts in report["problems"].items()}}, default=str), flush=True)
        og.shutdown()
    sys.exit(0 if report["n_written"] > 0 else 2)


if __name__ == "__main__":
    main()
