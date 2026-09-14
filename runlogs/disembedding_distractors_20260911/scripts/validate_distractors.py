#!/usr/bin/env python3
"""Distractor validation for the disembedding task with the instance sampler (the "initialization script").

Two phases, one OmniGibson process (house_single_floor, dining_room_0, the test-spec table and trays):

``--phase render``
    Head-camera renders of every candidate model (``candidates.json``, ``unique_models``) standing alone at the
    centre of the source tray, seen from the disembedding init stance of the test template (robot at the stance,
    trunk at the viewing configuration, trays where the sampler puts them). Four yaws (0 / 90 / 180 / 270 deg)
    for the same-category candidates, two for the others. The crops (``renders/<category>__<model>__y<k>.png``,
    the source-tray region of the 1080x1080 head image) are the evidence for the "identical from the init view"
    grouping; ``renders/meta/<category>__<model>.json`` records the settling check (drift / tilt after 60 physics
    steps, still on the tray floor, segmentation pixels) of the model resting on the tray.

``--phase validate``
    For every target category: the disembedding template is set up with the category's test pick as target
    (``DisembeddingSampler.setup``: stance, trays, view, grasp / placement tables - exactly what
    ``sample_instances.py`` does) with ONE candidate distractor; then every other candidate is swapped in as the
    distractor (same bindings / BDDL rebuild the sampler would do for a new template) and ``TaskSampler.run(1)``
    generates one instance: sample -> apply -> settle -> BEHAVIOR validation -> ``:init`` true -> goal false ->
    occlusion measurement -> goal feasibility by teleport + settle. A distractor is *viable* for the category when an
    instance was written. Results: ``results/<category>__<dcategory>-<dmodel>.json`` (status, rejections, the
    instance summary digest, occlusion, footprint budget); TROs / summaries go to ``--work``.

Both phases are resumable (``--skip-done``); a job that was in progress when the process died is recorded as
``crashed`` on the next run.
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
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "runlogs", "table_selection_20260911", "scripts"))

from ispatialgym.sampling.spec import ModelChoice, ProblemSpec, RoleSpec, load_task_spec  # noqa: E402

DEFAULT_SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
PROBLEM = "disembedding-0"
SCENE = "house_single_floor"
SAME_YAWS = [0.0, 90.0, 180.0, 270.0]
DIFF_YAWS = [0.0, 90.0]


def say(msg: str):
    print(f"[dis_distractors {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------------------------
# spec helpers
# ----------------------------------------------------------------------------------------------------------------


def problem_for(base: ProblemSpec, target_cat: str, target_model: str, grasp_key: str, d_cat: str, d_model: str) -> ProblemSpec:
    """Test-spec disembedding entry with one target model and a ``distractors`` section (sampler >= 1.4.0) holding
    exactly one distractor: the same-category pool when ``d_cat == target_cat``, otherwise an other-category pool."""
    roles = dict(base.roles)
    roles["target"] = RoleSpec("target", models={target_cat: [target_model]}, grasps={"key": grasp_key})
    roles.pop("same_category_pool", None)
    roles.pop("distractor_pool", None)
    gen = copy.deepcopy(base.generation)
    same = d_cat == target_cat
    gen.update({"same_category_prob": 1.0 if same else 0.0, "max_distractors": 1, "instances_per_template": 1, "templates_per_scene": None})
    raw = copy.deepcopy(base.raw)
    pools = {target_cat: {"same_category": [d_model] if same else [], "other_categories": [] if same else [d_cat]}}
    if not same:
        pools[d_cat] = {"same_category": [d_model], "other_categories": []}
    raw["distractors"] = {"pools": pools, "lookalike_groups": []}
    return ProblemSpec(name=base.name, alias=base.alias, roles=roles, scenes=base.scenes, generation=gen, raw=raw)


def rooms_of(spec_file):
    from ispatialgym.sampling.sample_instances import rooms_for
    from ispatialgym.sampling.sim_utils import stable_scene_path

    pspec = spec_file[PROBLEM]
    with open(stable_scene_path(SCENE)) as f:
        stable = json.load(f)
    return rooms_for(pspec, pspec.scenes[SCENE], stable)


def scene_variant_of(spec_file, rooms, work_dir):
    """The scene variant sample_instances.py (sampler >= 1.4.0) samples table tasks in: the stable scene minus the
    chairs around the work table (``generation.remove_chairs_near_tables``). None when the sampler has no such step."""
    try:
        from ispatialgym.sampling import scene_variants as SV
        from ispatialgym.sampling.sample_instances import chairs_to_remove
    except ImportError:
        return None
    from ispatialgym.sampling.sim_utils import stable_scene_path

    pspec = spec_file[PROBLEM]
    with open(stable_scene_path(SCENE)) as f:
        stable = json.load(f)
    removed = chairs_to_remove(pspec, pspec.scenes[SCENE], stable, rooms)
    if not removed:
        return None
    vpath, _ = SV.write_variant(stable_scene_path(SCENE), removed, os.path.join(work_dir, "scene_variants"))
    return {"path": vpath, "removed_objects": removed, "freed_footprints": SV.freed_footprints(stable, removed)}


# ----------------------------------------------------------------------------------------------------------------
# render phase
# ----------------------------------------------------------------------------------------------------------------


def asset_path(cat: str, model: str) -> str:
    from omnigibson.utils.asset_utils import get_dataset_path

    return os.path.join(get_dataset_path("behavior-1k-assets"), "objects", cat, model, "usd", f"{model}.encrypted.usd")


def spawn_batch(world, items):
    """Spawn several DatasetObjects with one stop / play cycle (SimWorld.spawn does one per object).

    Returns ``(objects, failed)``; ``failed`` = ``[(name, category, model, error), ...]`` for models whose asset is
    missing on disk (checked before touching the stage) or whose ``add_object`` raised. A raise inside the stopped
    simulator left the process unusable in the 09:00 run (every later spawn failed), so callers exit for a restart
    after a batch with an ``add_object`` failure.
    """
    import omnigibson as og
    from omnigibson.objects import DatasetObject

    objs, failed, todo = [], [], []
    for name, cat, model in items:
        p = asset_path(cat, model)
        if not os.path.exists(p):
            failed.append((name, cat, model, f"asset missing on disk: {p}"))
        else:
            todo.append((name, cat, model))
    state = og.sim.dump_state()
    og.sim.stop()
    raised = False
    for name, cat, model in todo:
        try:
            kwargs = {"in_rooms": list(world.rooms)} if world.rooms else {}
            obj = DatasetObject(name=name, category=cat, model=model, dataset_name="behavior-1k-assets", fixed_base=False, **kwargs)
            world.scene.add_object(obj)
            objs.append(obj)
        except Exception as e:
            traceback.print_exc()
            failed.append((name, cat, model, f"add_object failed: {e!r}"))
            raised = True
    og.sim.play()
    try:
        og.sim.load_state(state)
    except Exception:
        traceback.print_exc()
    for obj in objs:
        world.spawned[obj.name] = obj
        world.park(obj, len(world.spawned) - 1)
    og.sim.step()
    return objs, failed, raised


def remove_batch(world, objs):
    import omnigibson as og

    state = og.sim.dump_state()
    og.sim.stop()
    for obj in objs:
        world.spawned.pop(obj.name, None)
        world.scene.remove_object(obj)
    og.sim.play()
    try:
        og.sim.load_state(state)
    except Exception:
        pass
    og.sim.step()


def render_phase(args, cands, spec_file, world, kit):
    import torch as th
    from PIL import Image

    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling.tasks import get_sampler_class
    from ispatialgym.sampling.tasks.base import SamplerContext

    out_dir = os.path.join(args.out, "renders")
    meta_dir = os.path.join(out_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    same_models = {(e["category"], e["model"]) for lst in cands["same"].values() for e in lst}
    todo = []
    for e in cands["unique_models"]:
        key = (e["category"], e["model"])
        yaws = SAME_YAWS if key in same_models else DIFF_YAWS
        if args.categories and e["category"] not in args.categories:
            continue
        if args.skip_done and os.path.exists(os.path.join(meta_dir, f"{e['category']}__{e['model']}.json")):
            continue
        todo.append((e["category"], e["model"], yaws))
    # same-category candidates first: their grids feed the look-alike review, which can start while the rest renders
    todo.sort(key=lambda t: (0 if (t[0], t[1]) in same_models else 1, t[0], t[1]))
    say(f"render phase: {len(todo)} models to render ({len(cands['unique_models'])} unique candidates)")
    if not todo:
        return

    # reference template (test spec: cereal box + trays) gives the stance and the tray poses
    pspec = spec_file[PROBLEM]
    choices = pspec.template_choices(np.random.default_rng(0))[0]
    ctx = SamplerContext(pspec, pspec.scenes[SCENE], choices, world, kit, os.path.join(args.work, "render_ref"), "render_ref", spec_file.sha256, base_seed=0)
    sampler = get_sampler_class(PROBLEM)(ctx)
    t0 = time.time()
    sampler.setup()
    say(f"reference template set up in {time.time() - t0:.0f}s: stance {sampler.stance.to_json()}")
    st = sampler.stance
    for o in world.spawned.values():  # park the reference target / distractor, keep the trays
        if o.name not in (sampler.obj("tray_source").name, sampler.obj("tray_dest").name):
            world.park(o, list(world.spawned).index(o.name))
    kit.apply(st.q_stance, st.base_pose2d)
    world.step_physics(5, still=[world.robot])
    cam = kit.camera_pose(st.q_stance)
    floor = sampler.src_floor
    cells = floor.cells()
    centre = th.tensor([float(cells[:, 0].mean()), float(cells[:, 1].mean())])
    # crop window: the source tray's footprint corners at the floor and 0.35 m above it
    src = sampler.obj("tray_source")
    lo, hi = src.aabb
    corners = G.aabb_corners(lo, hi)
    corners_hi = corners.clone()
    corners_hi[:, 2] = float(floor.z) + 0.35
    uv, _ = world.camera_model.project(cam, th.cat([corners, corners_hi]))
    m = 30
    u0, v0 = max(0, int(uv[:, 0].min()) - m), max(0, int(uv[:, 1].min()) - m)
    u1, v1 = min(world.camera_model.width, int(uv[:, 0].max()) + m), min(world.camera_model.height, int(uv[:, 1].max()) + m)
    with open(os.path.join(out_dir, "render_meta.json"), "w") as f:
        json.dump({"stance": st.to_json(), "camera_pose": [cam[0].tolist(), cam[1].tolist()], "crop_uv": [u0, v0, u1, v1],
                   "tray_source": {"model": src.model, "pose": [src.get_position_orientation()[0].tolist(), src.get_position_orientation()[1].tolist()]},
                   "floor_z": float(floor.z), "centre_xy": centre.tolist(), "yaws_same": SAME_YAWS, "yaws_diff": DIFF_YAWS,
                   "robot_trunk": list(kit.trunk_of(st.q_stance)), "image_size": world.camera_model.width,
                   "camera_model": {"fx": world.camera_model.fx, "fy": world.camera_model.fy, "cx": world.camera_model.cx, "cy": world.camera_model.cy},
                   "tray_dest": {"model": sampler.obj("tray_dest").model, "pose": [sampler.obj("tray_dest").get_position_orientation()[0].tolist(),
                                                                                  sampler.obj("tray_dest").get_position_orientation()[1].tolist()]}}, f, indent=1)
    say(f"crop window u {u0}..{u1} v {v0}..{v1}; tray centre {centre.tolist()} z {floor.z:.3f}")

    def crop_rgb(n_renders=3):
        img = world.rgb(n_renders).numpy().astype(np.uint8)
        return img[v0:v1, u0:u1]

    batch = int(args.batch)
    n_done = 0
    t_start = time.time()
    for bi in range(0, len(todo), batch):
        chunk = todo[bi:bi + batch]
        spawned, failed, raised = spawn_batch(world, [(f"rnd_{c}_{m}", c, m) for c, m, _ in chunk])
        by_name = {o.name: o for o in spawned}
        objs = [by_name.get(f"rnd_{c}_{m}") for c, m, _ in chunk]
        for name, c, m, err in failed:
            say(f"spawn failed for {c}/{m}: {err[:160]}")
            with open(os.path.join(meta_dir, f"{c}__{m}.json"), "w") as f:
                json.dump({"category": c, "model": m, "status": "spawn_failed", "error": err[:500]}, f, indent=1)
            n_done += 1
        for (c, m, yaws), obj in zip(chunk, objs):
            if obj is None:
                continue
            rec = {"category": c, "model": m, "status": "ok", "yaws": yaws, "renders": {}}
            try:
                hull = world.local_hull(obj)
                if len(hull) == 0:
                    raise RuntimeError("no collision hull")
                for k, yaw_deg in enumerate(yaws):
                    quat = G.quat_from_yaw(math.radians(yaw_deg))
                    pos = world.place(obj, centre, quat, float(floor.z), local_hull=hull)
                    obj.keep_still()
                    world.step_physics(60, still=[world.robot])
                    p1, q1 = obj.get_position_orientation()
                    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(G.rotate(q1, th.tensor([0.0, 0.0, 1.0]))[2])))))
                    drift = float(th.linalg.norm(p1[:2] - pos[:2]))
                    on_floor = bool(floor.contains(p1[:2], 0.0)) and float(p1[2]) > float(floor.z) - 0.02
                    world.wake([obj])
                    obj.keep_still()
                    # physics-only steps do not push poses to the render stage: two full (rendering) simulator steps
                    # sync the parked previous object and the settled new one before the frames are read (measured:
                    # without them every frame showed the previous placement)
                    for _ in range(2):
                        world.robot.keep_still()
                        world.full_step(render=True)
                    world.robot.keep_still()
                    obj.keep_still()
                    path = os.path.join(out_dir, f"{c}__{m}__y{k}.png")
                    Image.fromarray(crop_rgb(4)).save(path)
                    r = {"yaw_deg": yaw_deg, "png": os.path.relpath(path, ROOT), "drift_m": round(drift, 4), "tilt_deg": round(tilt, 1), "on_tray": on_floor,
                         "settled_z": round(float(p1[2]) - float(floor.z), 4)}
                    if k == 0:
                        # the instance-segmentation label map lags behind spawning (non-deterministic); retry with
                        # full steps before giving up, and record None (unknown) rather than 0 for a placed object
                        px = world.object_pixels(obj)
                        for _ in range(3):
                            if px > 0:
                                break
                            world.ensure_segmented(obj)
                            world.robot.keep_still()
                            world.full_step(render=True)
                            px = world.object_pixels(obj)
                        r["seg_pixels"] = int(px) if px > 0 else None
                        r["aabb_extent"] = [round(float(v), 3) for v in (obj.aabb[1] - obj.aabb[0])]
                    rec["renders"][f"y{k}"] = r
                    world.park(obj, list(world.spawned).index(obj.name))
                r0 = rec["renders"]["y0"]
                rec["stable"] = bool(r0["on_tray"] and r0["drift_m"] < 0.03 and r0["tilt_deg"] < 20.0)
                rec["visible"] = None if r0.get("seg_pixels") is None else int(r0["seg_pixels"]) >= 200
            except Exception as e:
                traceback.print_exc()
                rec.update({"status": "error", "error": repr(e)[:800]})
                try:
                    world.park(obj, list(world.spawned).index(obj.name))
                except Exception:
                    pass
            with open(os.path.join(meta_dir, f"{c}__{m}.json"), "w") as f:
                json.dump(rec, f, indent=1)
            n_done += 1
            if n_done % 10 == 0 or n_done == len(todo):
                say(f"rendered {n_done}/{len(todo)} ({(time.time() - t_start) / max(n_done, 1):.1f} s/model); last {c}/{m}: "
                    f"stable {rec.get('stable')} px {rec.get('renders', {}).get('y0', {}).get('seg_pixels')}")
        try:
            remove_batch(world, [o for o in objs if o is not None])
        except Exception:
            traceback.print_exc()
            say("batch removal failed; removing one by one")
            for o in objs:
                if o is not None and o.name in world.spawned:
                    try:
                        world.remove(o)
                    except Exception:
                        traceback.print_exc()
        if raised:
            say("an add_object call raised inside the stopped simulator; exiting for a clean restart (results so far are kept)")
            os._exit(3)
    # tear down the reference template
    for o in list(world.spawned.values()):
        try:
            world.remove(o)
        except Exception:
            traceback.print_exc()
    kit.sync_world()


# ----------------------------------------------------------------------------------------------------------------
# validation phase
# ----------------------------------------------------------------------------------------------------------------


def jobs_for_category(cands, cat, args):
    """Ordered candidate distractors for the category: same-category models (size ok, ready) then related ones.
    With ``--pairs`` (json ``{target category: [[category, model], ...]}``, e.g. the preliminary pools + reserves of
    assemble_pools.py) only the listed candidates are validated, in the listed order."""
    same = [e for e in cands["same"][cat] if e["size_ok"] and e["ready"]]
    if args.pairs is not None:
        want = [tuple(p) for p in args.pairs.get(cat, [])]
        by_key = {(e["category"], e["model"]): ("same", e) for e in same}
        by_key.update({(e["category"], e["model"]): ("diff", e) for e in cands["diff"][cat] if e["size_ok"] and e["ready"]})
        return [by_key[k] for k in want if k in by_key]
    n_same_groups = len({tuple(e["exact_group"]) if e["exact_group"] else (e["model"],) for e in same})
    n_diff = min(int(cands["limits"]["n_diff_validate"]), 20 - min(n_same_groups, 10) + 6) if args.n_diff is None else args.n_diff
    diff = [e for e in cands["diff"][cat] if e["size_ok"] and e["ready"]][:n_diff]
    return [("same", e) for e in same] + [("diff", e) for e in diff]


def swap_distractor(sampler, choice: ModelChoice):
    """Replace the template's distractor by ``choice``: remove the old object, spawn / measure the new one and rebuild
    the concrete BDDL, exactly the state ``setup()`` leaves for a template that drew this distractor."""
    from ispatialgym.sampling.bddl_writer import build_problem
    from ispatialgym.sampling.spec import sha256_of_text

    for b in sampler.bindings.many("distractors"):
        obj = sampler.objects.pop(b.name, None)
        if obj is not None and obj.name in sampler.world.spawned:
            sampler.world.remove(obj)
        for d in (sampler.hull, sampler.radius, sampler.height):
            d.pop(b.name, None)
    sampler.bindings.roles["distractors"] = []
    sampler.spawn_role("distractors", choice)
    b = sampler.bindings.many("distractors")[0]
    sampler.kit.sync_world()
    sampler._measure(b.name, sampler.objects[b.name])
    sampler.problem = build_problem(sampler.activity, sampler.definition_id, sampler.bindings)
    sampler.compiled = sampler.problem.compile()
    sampler.template_meta["bddl_sha256"] = sha256_of_text(sampler.problem.text)
    sampler.build_scope()
    return b


def footprint_fraction(target_bbox, d_bbox, tray_bbox):
    a = (target_bbox[0] + 0.02) * (target_bbox[1] + 0.02) + (d_bbox[0] + 0.02) * (d_bbox[1] + 0.02)
    return a / (tray_bbox[0] * tray_bbox[1])


def validate_phase(args, cands, spec_file, world, kit):
    from validate_tables import install_seg_refresh, instance_digest, template_meta_digest  # noqa: E402  (table-selection helpers)

    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.tasks import get_sampler_class
    from ispatialgym.sampling.tasks.base import SamplerContext

    from ispatialgym.sampling import knowledge as K

    install_seg_refresh(world)
    res_dir = os.path.join(args.out, "results")
    os.makedirs(res_dir, exist_ok=True)
    base = spec_file[PROBLEM]
    tray_bbox = cands["tray_source"]["bbox"]
    cats = list(cands["targets"]) if not args.categories else [c for c in cands["targets"] if c in args.categories]
    t_start = time.time()
    for ci, cat in enumerate(cats):
        tinfo = cands["targets"][cat]
        jobs = jobs_for_category(cands, cat, args)

        def rpath(e):
            return os.path.join(res_dir, f"{cat}__{e['category']}-{e['model']}.json")

        if args.skip_done:
            kept = []
            for kind, e in jobs:
                p = rpath(e)
                if os.path.exists(p):
                    continue
                if os.path.exists(p + ".inprogress"):
                    with open(p + ".inprogress") as f:
                        started = f.read().strip()
                    with open(p, "w") as f:
                        json.dump({"target_category": cat, "category": e["category"], "model": e["model"], "kind": kind, "status": "crashed",
                                   "error": f"process died while running this job (started {started})", "n_written": 0}, f, indent=1)
                    os.remove(p + ".inprogress")
                    say(f"{cat} / {e['category']}-{e['model']}: previous process died here; recorded as crashed")
                    continue
                kept.append((kind, e))
            jobs = kept
        if not jobs:
            say(f"[{ci + 1}/{len(cats)}] {cat}: nothing to do")
            continue
        say(f"[{ci + 1}/{len(cats)}] {cat}: {len(jobs)} distractor candidates ({sum(1 for k, _ in jobs if k == 'same')} same-category)")
        # ---- target: the pick-1 test instance. The pick-1 model itself is not validated as a distractor (a second
        # template per category costs ~4 min of IK tables for one pair); it is the target of every other pair of the
        # category, which proves it stands / is graspable on the tray (recorded as status ``is_target``).
        targets = [(tinfo["pick1"], tinfo["pick1_key"])]
        if args.pick2_template and tinfo.get("pick2"):
            same_rows = {e["model"]: e for e in cands["same"][cat]}
            p2 = same_rows.get(tinfo["pick2"])
            targets.append((tinfo["pick2"], p2["grasp_key"] if p2 and p2.get("grasp_key") else f"{cat}__{tinfo['pick2']}"))
        by_target = {}
        for kind, e in jobs:
            tm = None
            for t_model, t_key in targets:
                if not (e["category"] == cat and e["model"] == t_model):
                    tm = (t_model, t_key)
                    break
            if tm is None:
                with open(rpath(e), "w") as f:
                    json.dump({"target_category": cat, "category": e["category"], "model": e["model"], "kind": kind, "status": "is_target",
                               "error": "the category's test target itself: it is the target of every other validated pair of the category "
                                        "(stands on the tray, graspable, placeable), not re-validated as a distractor", "n_written": 0}, f, indent=1)
                continue
            by_target.setdefault(tm, []).append((kind, e))
        for (t_model, t_key), tjobs in by_target.items():
            sampler = None
            first_kind, first = tjobs[0]
            pspec = problem_for(base, cat, t_model, t_key, first["category"], first["model"])
            choices = pspec.template_choices(np.random.default_rng(0))[0]
            tag = f"{cat}-{t_model}"
            ctx = SamplerContext(pspec, pspec.scenes[SCENE], choices, world, kit, os.path.join(args.work, tag), tag, spec_file.sha256, base_seed=args.seed * 7919 + ci)
            sampler = get_sampler_class(PROBLEM)(ctx)
            sampler.variant_overrides = ["same_category" if first["category"] == cat else "other_category"]
            t0 = time.time()
            setup_err = None
            try:
                sampler.setup()
            except Reject as e:
                setup_err = f"[{e.reason}] {e}"
            except Exception as e:
                traceback.print_exc()
                setup_err = repr(e)
                if "memory" in repr(e).lower():
                    say("out-of-memory during template setup; exiting for a restart")
                    os._exit(3)
            setup_s = time.time() - t0
            if setup_err is not None:
                say(f"  template setup with target {cat}/{t_model} FAILED after {setup_s:.0f}s: {setup_err[:400]}")
                for kind, e in tjobs:
                    with open(rpath(e), "w") as f:
                        json.dump({"target_category": cat, "target_model": t_model, "category": e["category"], "model": e["model"], "kind": kind,
                                   "status": "setup_failed", "error": setup_err[:2000], "n_written": 0, "setup_s": round(setup_s, 1)}, f, indent=1)
                for o in list(world.spawned.values()):
                    world.remove(o)
                kit.sync_world()
                continue
            say(f"  template with target {cat}/{t_model} set up in {setup_s:.0f}s (stance {sampler.stance.edge_name}, view arms {sampler.template_meta.get('view_arms')}, "
                f"yaw adjust {sampler.template_meta.get('yaw_adjust_deg', 0):.1f} deg, {len(sampler.cells)} grasp cells)")
            tmeta = template_meta_digest(sampler.template_meta)
            tmeta["setup_s"] = round(setup_s, 1)
            # templates are not saved (evidence = summaries / TROs); keeps the work dir small
            sampler.template_paths = ("-", "-")
            sampler.template_sha = "not-saved"
            for ji, (kind, e) in enumerate(tjobs):
                p = rpath(e)
                dtag = f"{tag}__{e['category']}-{e['model']}"
                rec = {"target_category": cat, "target_model": t_model, "target_key": t_key, "category": e["category"], "model": e["model"], "kind": kind,
                       "tier": e.get("tier"), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "template": tmeta,
                       "footprint_fraction_with_target": round(footprint_fraction(K.canonical_bbox(t_model), e["bbox"], tray_bbox), 3)}
                with open(p + ".inprogress", "w") as f:
                    f.write(rec["started"])
                t1 = time.time()
                out_dir = os.path.join(args.work, dtag)
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                os.makedirs(out_dir, exist_ok=True)
                written = []
                try:
                    if ji > 0:
                        swap_distractor(sampler, ModelChoice(e["category"], e["model"]))
                    sampler.ctx.out_dir = out_dir
                    sampler.ctx.template_id = dtag
                    sampler.template_meta["template_id"] = dtag
                    sampler.template_meta["distractors"] = {"same_category": kind == "same", "models": [f"{e['category']}-{e['model']}"], "requested": 1,
                                                            "footprint_fraction": rec["footprint_fraction_with_target"]}
                    sampler.rejections = {}
                    sampler.variant_overrides = []
                    written = sampler.run(1, max_attempts=args.max_attempts)
                    if not written and sampler.rejections.get("occlusion"):
                        # a short distractor cannot hide a tall target and half of the draws ask for an occluded
                        # instance: confirm that the pair works as an unoccluded instance before giving it up
                        rec["occlusion_retry"] = dict(sampler.rejections)
                        sampler.variant_overrides = ["unoccluded"]
                        sampler.ctx.base_seed += 1
                        written = sampler.run(1, max_attempts=max(2, args.max_attempts // 2))
                        rec["viable_unoccluded_only"] = bool(written)
                    rec["status"] = "ok" if written else "no_instances"
                except Reject as ex:
                    rec.update({"status": "setup_failed", "reject_reason": ex.reason, "error": str(ex)[:2000]})
                except Exception as ex:
                    traceback.print_exc()
                    rec.update({"status": "error", "error": repr(ex)[:2000]})
                    if "memory" in repr(ex).lower():
                        if os.path.exists(p + ".inprogress"):
                            os.remove(p + ".inprogress")
                        say("out-of-memory error; exiting for a restart")
                        os._exit(3)
                    # an object that failed to spawn / measure leaves no distractor binding: re-spawn a known-good one
                    try:
                        for b in sampler.bindings.many("distractors"):
                            o = sampler.objects.pop(b.name, None)
                            if o is not None and o.name in world.spawned:
                                world.remove(o)
                        sampler.bindings.roles["distractors"] = []
                    except Exception:
                        traceback.print_exc()
                finally:
                    try:
                        for arm in ("left", "right"):
                            if getattr(world.robot, "_ag_obj_in_hand", {}).get(arm) is not None:
                                world.robot.release_grasp_immediately(arm)
                    except Exception:
                        pass
                rec.update({"n_written": len(written), "rejections": dict(sampler.rejections), "instances": instance_digest(out_dir),
                            "work_dir": out_dir, "job_s": round(time.time() - t1, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
                if rec["instances"]:
                    inst = rec["instances"][0]
                    rec["occluded_variant"] = inst.get("occluded_variant")
                    rec["occlusion_ratio"] = inst.get("occlusion_ratio")
                with open(p, "w") as f:
                    json.dump(rec, f, indent=1, default=str)
                if os.path.exists(p + ".inprogress"):
                    os.remove(p + ".inprogress")
                say(f"  [{ji + 1}/{len(tjobs)}] {cat} <- {e['category']}/{e['model']} ({kind}) -> {rec['status']} (written {rec['n_written']}, "
                    f"rejections {rec['rejections']}, {rec['job_s']:.0f}s; elapsed {(time.time() - t_start) / 60:.0f} min)"
                    + (f" :: {rec.get('error', '')[:200]}" if rec["status"] in ("setup_failed", "error") else ""))
            # ---- tear down the template
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
            kit.sync_world()


# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["render", "validate", "both"], default="both")
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--candidates", default=os.path.join(ROOT, "candidates.json"))
    ap.add_argument("--out", default=ROOT)
    ap.add_argument("--work", default="/tmp/isg_dis_distractors")
    ap.add_argument("--categories", default=None, help="comma separated target categories (validate) / candidate categories (render)")
    ap.add_argument("--n-diff", type=int, default=None, help="override the number of related-category candidates validated per target category")
    ap.add_argument("--pairs", default=None, help="json {target category: [[category, model], ...]}: validate only these candidates (e.g. pools + reserves)")
    ap.add_argument("--pick2-template", action="store_true", help="also set up a template with the pick-2 target to validate the pick-1 model as a distractor")
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--batch", type=int, default=12, help="render phase: models spawned per stop/play cycle")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "WARNING"))
    args = ap.parse_args(argv)
    args.categories = set(args.categories.split(",")) if args.categories else None
    if args.pairs:
        with open(args.pairs) as f:
            args.pairs = json.load(f)

    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout, force=True)
    spec_file = load_task_spec(args.spec)
    with open(args.candidates) as f:
        cands = json.load(f)
    os.makedirs(args.work, exist_ok=True)
    rooms = rooms_of(spec_file)
    variant = scene_variant_of(spec_file, rooms, args.work)
    say(f"scene variant: {len(variant['removed_objects'])} objects removed {variant['removed_objects']}" if variant else "scene variant: none (stable scene)")

    from ispatialgym.sampling.robot_utils import RobotKit
    from ispatialgym.sampling.sim_utils import SimWorld

    import omnigibson as og

    try:
        t0 = time.time()
        try:
            world = SimWorld(SCENE, rooms, scene_variant=variant) if variant else SimWorld(SCENE, rooms)
            kit = RobotKit(world, prebuild=("right_arm",))
        except Exception as e:
            traceback.print_exc()
            say(f"scene load failed: {e!r}; exiting for a restart")
            os._exit(3)
        say(f"scene {SCENE} rooms {rooms} loaded in {time.time() - t0:.0f}s")
        if args.phase in ("render", "both"):
            render_phase(args, cands, spec_file, world, kit)
        if args.phase in ("validate", "both"):
            validate_phase(args, cands, spec_file, world, kit)
        say("done")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
