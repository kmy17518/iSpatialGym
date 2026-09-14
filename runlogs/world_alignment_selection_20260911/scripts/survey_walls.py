#!/usr/bin/env python3
"""Wall survey for ``world_alignment-1`` (world vertical): where in each candidate scene can the instance sampler hang
the target and initialise the task?

For one scene, every indoor room instance is loaded on its own (like ``sample_instances.py`` does for a task room) and
the ``WorldVerticalSampler`` is run with a *dense* wall-spot search instead of its 24 rays from the room centre:

1. **Spots** (tier 1, geometric): horizontal rays at the hanging height from a 0.5 m lattice of free floor cells of the
   room, 24 directions each, hits on ``walls`` objects whose front (0.8 m in front of the hit) lies in the room; one
   spot per 0.25 m of wall. Every spot is tested with the sampler's own ``_flat_patch`` / ``_front_clear`` at square
   patch sizes 0.4 .. 1.2 m (``max_patch_m`` = largest size that is flat and unobstructed; the sampler needs
   ``object + 0.10 m`` flat and ``object`` unobstructed 0.35 m deep), then with the sampler's ``_arc_bases``
   (traversable, in-room, whole-body collision-free base poses on the 0.6-0.95 m arc, capped for the survey) and
   ``_reach_contact`` (pushing reach of the aligned target's lower corners, right arm, IK; the survey stops after 3
   pushing-capable bases, tier 2 recomputes the full set). A spot is *viable* when the reference target passes all of it.
2. **Regions**: consecutive viable spots of one wall segment (same wall link and normal) are merged into regions
   (``p0``-``p1`` in world xy, normal, length, min ``max_patch_m``).
3. **Instances** (tier 2): for every region the sampler generates one full instance at the region's middle spot
   (``TaskSampler.run``: init pose, robot on the arc, settle, BEHAVIOR validation, ``:init`` true, goal false, >= 5
   objects visible from the head camera, hinge drift, goal feasibility by teleport + settle, TRO written). Long regions
   also get their quarter points. This is exactly what ``sample_instances.py`` does, so an instance written here means
   the region can initialise the task.

Results: ``<out>/results/<scene>__<room>.json`` (spots, regions, instances with the sampler's rejection reasons);
templates / TROs under ``--work``. ``aggregate_walls.py`` turns the results into the wall candidate spec, tables and maps.

Example::

    OMNIGIBSON_HEADLESS=1 python survey_walls.py --scene house_single_floor --out .. --work /tmp/isg_wall_survey
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
from typing import Any, Dict, List

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

from ispatialgym.sampling.spec import ProblemSpec, SceneSpec, load_task_spec  # noqa: E402

DEFAULT_SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
PROBLEM = "world_alignment-1"
OUTDOOR = ("garden", "patio", "backyard", "yard", "pool", "balcony", "terrace", "porch")
PATCH_SIZES = (0.4, 0.6, 0.8, 1.0, 1.2)


def say(msg: str):
    print(f"[wall_survey {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def indoor_rooms(scene: str) -> List[str]:
    from omnigibson.maps.segmentation_map import SegmentationMap
    from omnigibson.utils.asset_utils import get_scene_path

    seg = SegmentationMap(scene_dir=get_scene_path(scene), map_resolution=0.1)
    names = sorted(seg.room_ins_name_to_ins_id)
    return [n for n in names if not any(s in n for s in OUTDOOR)]


def spec_for_room(pspec: ProblemSpec, scene: str, room: str, target: Dict[str, List[str]]) -> ProblemSpec:
    raw = copy.deepcopy(pspec.raw)
    roles = copy.deepcopy(pspec.roles)
    roles["target"].models = {k: list(v) for k, v in target.items()}
    scene_spec = SceneSpec(scene, [room], {"floor": {"room": room}}, {})
    return ProblemSpec(name=pspec.name, alias=pspec.alias, roles=roles, scenes={scene: scene_spec}, generation=copy.deepcopy(pspec.generation), raw=raw)


def make_sampler_class():
    """Built after the OmniGibson launch (the sampler modules import omnigibson)."""
    import torch as th

    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling.bddl_writer import build_problem
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.tasks.world_alignment import WorldVerticalSampler

    class WallSurveySampler(WorldVerticalSampler):
        SPOT_STEP = 0.25  # m along the wall
        LATTICE = 0.3  # m between ray origins
        N_DIRS = 24
        MAX_BASES = 6  # arc bases kept per spot for the survey reach test (the sampler itself uses all 28)
        ENOUGH_REACH = 3  # stop the per-spot reach test after this many pushing-capable bases (tier 2 recomputes all)
        MAX_REACH_SPOTS = 60  # flat + clear spots per room that get the arc / reach test (evenly strided along the walls beyond that)

        def __init__(self, ctx):
            super().__init__(ctx)
            self.survey: List[Dict[str, Any]] = []
            self.all_good: List[Dict[str, Any]] = []
            self.n_ray_origins = 0

        # ---- dense spot search ---------------------------------------------------------------------------
        def _ray_origins(self, z: float) -> th.Tensor:
            """Ray origins on a lattice over the room's free floor (un-eroded traversability map: the rays only have to
            start inside the room, the robot's own footprint is validated later by the arc-base collision check)."""
            trav = self.world.scene.trav_map
            seg = self.world.scene.seg_map
            pts = []
            step = max(1, int(round(self.LATTICE / trav.map_resolution)))
            free = th.nonzero(trav.floor_map[0] == 255)
            for c in free:
                if int(c[0]) % step or int(c[1]) % step:
                    continue
                xy = trav.map_to_world(c)
                if seg.get_room_instance_by_point(xy) != self.room:
                    continue
                pts.append(th.tensor([float(xy[0]), float(xy[1]), z]))
            return th.stack(pts) if pts else th.zeros(0, 3)

        def _find_wall_spots(self) -> List[Dict[str, Any]]:
            floor = self.obj("floor")
            lo, hi = floor.aabb
            z = float(hi[2]) + float(self.gen("height", 1.45))
            origins = self._ray_origins(z)
            self.n_ray_origins = int(len(origins))
            if len(origins) == 0:
                return []
            dirs = th.stack([th.tensor([math.cos(a), math.sin(a), 0.0]) for a in th.linspace(0, 2 * math.pi, self.N_DIRS + 1)[:-1].tolist()])
            starts = origins.repeat_interleave(len(dirs), dim=0)
            ends = starts + dirs.repeat(len(origins), 1) * 12.0
            hits = self.world.raycast(starts, ends)
            walls = {l.prim_path: o for o in self.world.scene.objects if o.category == "walls" for l in o.links.values()}
            thickness, width, height = self._object_dims()
            uniq: Dict[tuple, Dict[str, Any]] = {}
            for h in hits:
                if not h["hit"] or h["rigidBody"] not in walls:
                    continue
                point = G.t(h["position"]).clone()
                normal = G.t(h["normal"]).clone()
                normal[2] = 0.0
                if float(th.linalg.norm(normal)) < 0.5:
                    continue
                normal = normal / th.linalg.norm(normal)
                if not self._in_room(point[:2] + normal[:2] * 0.8):
                    continue
                yaw = math.degrees(math.atan2(float(normal[1]), float(normal[0])))
                yaw_b = int(round(yaw / 15.0)) % 24
                tangent = th.tensor([-float(normal[1]), float(normal[0])])
                t = float(point[:2] @ tangent)
                key = (h["rigidBody"], yaw_b, int(math.floor(t / self.SPOT_STEP)))
                if key in uniq:
                    continue
                point[2] = z  # all spots at the hanging height
                uniq[key] = {"wall_name": walls[h["rigidBody"]].name, "wall_link": h["rigidBody"], "point": point, "normal": normal,
                             "yaw_bucket": yaw_b, "tangent_t": t}
            spots = []
            for key, s in uniq.items():
                wall = walls[s["wall_link"]]
                max_patch = None
                flat = {}
                for size in PATCH_SIZES:
                    ok = self._flat_patch(s["point"], s["normal"], size, size, wall) and self._front_clear(s["point"], s["normal"], size - 0.10, size - 0.10)
                    flat[str(size)] = bool(ok)
                    if ok:
                        max_patch = size
                    else:
                        break
                s["flat"] = flat
                s["max_patch_m"] = max_patch
                s["target_ok"] = bool(self._flat_patch(s["point"], s["normal"], width + 0.10, height + 0.10, wall) and self._front_clear(s["point"], s["normal"], width, height))
                self.survey.append(s)
                if s["target_ok"]:
                    spots.append(s)
            say(f"{self.room}: {self.n_ray_origins} ray origins, {len(uniq)} wall spots, {len(spots)} flat + clear for the target "
                f"({width:.2f} x {height:.2f} m)")
            return spots

        def _arc_bases(self, spot):
            bases = super()._arc_bases(spot)
            spot["n_bases_all"] = len(bases)
            if len(bases) > self.MAX_BASES:
                idx = np.linspace(0, len(bases) - 1, self.MAX_BASES).round().astype(int)
                bases = [bases[i] for i in sorted(set(idx.tolist()))]
            return bases

        # ---- keep every wall, record everything ----------------------------------------------------------------
        def prepare_bindings(self):
            target = self.obj("target")
            self.hull = self.world.local_hull(target)
            self.com_local = self._com_local(target)
            self.kit.sync_world()
            spots = self._find_wall_spots()
            # long corridors / halls yield hundreds of flat spots: stride them evenly along the walls so that the
            # (expensive) arc / reach test stays bounded; regions are merged with the strided step
            self.spot_stride = 1
            if len(spots) > self.MAX_REACH_SPOTS:
                spots.sort(key=lambda sp: (sp["wall_link"], sp["yaw_bucket"], sp["tangent_t"]))
                self.spot_stride = int(math.ceil(len(spots) / self.MAX_REACH_SPOTS))
                for i, sp in enumerate(spots):
                    if i % self.spot_stride:
                        sp["skipped"] = True
                spots = [sp for sp in spots if not sp.get("skipped")]
                say(f"{self.room}: reach test strided to every {self.spot_stride}. spot ({len(spots)} spots)")
            good = []
            for s in spots:
                t0 = time.time()
                bases = self._arc_bases(s)
                s["n_bases"] = len(bases)
                s["n_reach"] = 0
                if not bases:
                    s["reject"] = "no_arc_base"
                    continue
                p_goal, q_goal = self._hang_pose(s, 0.0)
                corners = [self._lowest_corner((p_goal, q_goal), s["normal"], side) for side in (1.0, -1.0)]
                keep, reach = [], []
                n_tested = 0
                for b in bases:
                    tr = None
                    for c in corners:
                        tr = self._reach_contact(b, c, s["normal"])
                        if tr is not None:
                            break
                    n_tested += 1
                    if tr is not None:
                        keep.append(b)
                        reach.append(tr)
                        if len(keep) >= self.ENOUGH_REACH:
                            break
                s["n_reach"] = len(keep)
                s["n_reach_tested"] = n_tested
                s["reach_s"] = round(time.time() - t0, 1)
                if not keep:
                    s["reject"] = "goal_corner_unreachable"
                    continue
                s["bases"], s["goal_reach"] = keep, reach
                s["viable"] = True
                good.append(s)
            self.all_good = good
            # bind the wall with the most viable spots so that the problem compiles; validate_spot rebinds per spot
            counts: Dict[str, int] = {}
            for s in good:
                counts[s["wall_name"]] = counts.get(s["wall_name"], 0) + 1
            if counts:
                wall_name = max(counts, key=counts.get)
            else:
                walls = [o for o in self.world.scene.objects if o.category == "walls" and set(o.in_rooms or []) & {self.room}]
                if not walls:
                    walls = [o for o in self.world.scene.objects if o.category == "walls"]
                if not walls:
                    raise Reject("no_wall", "no walls object loaded")
                wall_name = sorted(o.name for o in walls)[0]
            self._rebind_wall(self.world.obj(wall_name))
            self.wall_spots = [s for s in good if s["wall_name"] == wall_name]
            say(f"{self.room}: {len(good)} viable spots on {len(counts)} wall objects {counts}")

        def _rebind_wall(self, wall_obj):
            old = self.bindings.roles.get("wall", [])
            for b in old:
                self.objects.pop(b.name, None)
                self.bindings._counters[b.synset] = max(0, self.bindings._counters.get(b.synset, 1) - 1)
            self.bindings.roles["wall"] = []
            self.bind_scene_object("wall", wall_obj)

        def rebuild_problem(self):
            self.problem = build_problem(self.activity, self.definition_id, self.bindings)
            self.compiled = self.problem.compile()
            self.build_scope()

        # ---- tier 2: one instance at a given spot ------------------------------------------------------------
        def validate_spot(self, spot: Dict[str, Any], out_dir: str, max_attempts: int = 4) -> Dict[str, Any]:
            wall_obj = self.world.obj(spot["wall_name"])
            if self.bindings.one("wall").scene_object_name != wall_obj.name:
                self._rebind_wall(wall_obj)
                self.rebuild_problem()
            # the real sampler uses every arc base of the spot (not the survey cap)
            full = WorldVerticalSampler._arc_bases(self, spot)
            if full:
                p_goal, q_goal = self._hang_pose(spot, 0.0)
                corners = [self._lowest_corner((p_goal, q_goal), spot["normal"], side) for side in (1.0, -1.0)]
                keep, reach = [], []
                for b in full:
                    tr = None
                    for c in corners:
                        tr = self._reach_contact(b, c, spot["normal"])
                        if tr is not None:
                            break
                    if tr is not None:
                        keep.append(b)
                        reach.append(tr)
                if keep:
                    spot["bases"], spot["goal_reach"] = keep, reach
            self.wall_spots = [spot]
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            self.ctx.out_dir = out_dir
            self.template_paths = None
            self.template_sha = None
            self.rejections = {}
            t0 = time.time()
            written = self.run(1, max_attempts=max_attempts)
            rec = {"n_written": len(written), "rejections": dict(self.rejections), "run_s": round(time.time() - t0, 1), "work_dir": out_dir,
                   "n_bases_full": len(spot.get("bases", []))}
            inst_dir = os.path.join(out_dir, "instances")
            if os.path.isdir(inst_dir):
                for fn in sorted(os.listdir(inst_dir)):
                    if fn.endswith(".summary.json"):
                        with open(os.path.join(inst_dir, fn)) as f:
                            s = json.load(f)
                        rec["instance"] = {k: s.get(k) for k in ("base_pose2d", "trunk", "deviation_deg", "arc", "view", "push_reach_translates", "visible_objects", "wall_time_s")}
                        rec["instance"]["summary"] = os.path.join(inst_dir, fn)
                        rec["instance"]["n_visible_objects"] = len(s.get("visible_objects") or [])
            return rec

    return WallSurveySampler


def regions_of(spots: List[Dict[str, Any]], step: float) -> List[Dict[str, Any]]:
    """Merge consecutive viable spots of one wall segment (same link + normal bucket) into regions."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for s in spots:
        if not s.get("viable"):
            continue
        groups.setdefault((s["wall_link"], s["yaw_bucket"]), []).append(s)
    out = []
    for (link, yb), lst in groups.items():
        lst.sort(key=lambda s: s["tangent_t"])
        run = [lst[0]]
        runs = []
        for s in lst[1:]:
            if s["tangent_t"] - run[-1]["tangent_t"] <= 1.6 * step:
                run.append(s)
            else:
                runs.append(run)
                run = [s]
        runs.append(run)
        for r in runs:
            n = np.mean([s["normal"][:2] for s in r], axis=0)
            n = n / (np.linalg.norm(n) + 1e-9)
            p0, p1 = r[0]["point"][:2], r[-1]["point"][:2]
            out.append({"wall_name": r[0]["wall_name"], "wall_link": link, "normal": [round(float(v), 4) for v in n],
                        "p0": [round(float(v), 3) for v in p0], "p1": [round(float(v), 3) for v in p1],
                        "length_m": round(float(np.linalg.norm(np.asarray(p1) - np.asarray(p0))) + step, 2),
                        "hang_z": round(float(r[0]["point"][2]), 3), "n_spots": len(r),
                        "max_patch_m": min((s["max_patch_m"] or 0.0) for s in r),
                        "n_bases": int(sum(s.get("n_bases", 0) for s in r)), "n_reach": int(sum(s.get("n_reach", 0) for s in r)),
                        "spot_indices": [s["idx"] for s in r]})
    out.sort(key=lambda r: (r["wall_name"], -r["length_m"]))
    for i, r in enumerate(out):
        r["id"] = f"r{i:02d}"
    return out


def jsonable(x):
    import torch as th

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items() if k not in ("bases", "goal_reach")}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, th.Tensor):
        return [round(float(v), 4) for v in x.flatten().tolist()] if x.numel() > 1 else round(float(x), 4)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--out", default=os.path.join(HERE, ".."))
    ap.add_argument("--work", default="/tmp/isg_wall_survey")
    ap.add_argument("--rooms", default=None, help="comma separated room instances (default: every indoor room)")
    ap.add_argument("--target", default="painting/dacihj", help="reference target category/model")
    ap.add_argument("--max-instances-per-room", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--no-instances", action="store_true", help="tier 1 only")
    ap.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout, force=True)
    spec_file = load_task_spec(args.spec)
    cat, model = args.target.split("/")
    res_dir = os.path.join(args.out, "results")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)

    import omnigibson as og  # noqa: F401  (launch)

    rooms = args.rooms.split(",") if args.rooms else indoor_rooms(args.scene)
    say(f"{args.scene}: rooms {rooms}")
    from ispatialgym.sampling.robot_utils import RobotKit
    from ispatialgym.sampling.sim_utils import Reject, SimWorld
    from ispatialgym.sampling.tasks.base import SamplerContext

    Sampler = make_sampler_class()
    t_start = time.time()
    try:
        for ri, room in enumerate(rooms):
            path = os.path.join(res_dir, f"{args.scene}__{room}.json")
            if args.skip_done and os.path.exists(path):
                say(f"[{ri + 1}/{len(rooms)}] {room}: done already")
                continue
            if os.path.exists(path + ".inprogress") and args.skip_done:
                with open(path + ".inprogress") as f:
                    started = f.read().strip()
                with open(path, "w") as f:
                    json.dump({"scene": args.scene, "room": room, "status": "crashed", "error": f"process died while surveying this room (started {started})"}, f, indent=1)
                os.remove(path + ".inprogress")
                say(f"[{ri + 1}/{len(rooms)}] {room}: previous process died here; recorded as crashed")
                continue
            with open(path + ".inprogress", "w") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
            rec: Dict[str, Any] = {"scene": args.scene, "room": room, "target": args.target, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                   "spot_step_m": Sampler.SPOT_STEP, "patch_sizes_m": list(PATCH_SIZES)}
            t_room = time.time()
            world = None
            try:
                t0 = time.time()
                world = SimWorld(args.scene, [room])
                kit = RobotKit(world, prebuild=("right_arm",))
                rec["load_s"] = round(time.time() - t0, 1)
                rec["walls"] = sorted(o.name for o in world.scene.objects if o.category == "walls" and room in (o.in_rooms or []))
                rec["floor"] = world.floor_of_room(room).name
                pspec = spec_for_room(spec_file[PROBLEM], args.scene, room, {cat: [model]})
                choices = pspec.template_choices(np.random.default_rng(0))[0]
                tag = f"{args.scene}__{room}"
                out_dir = os.path.join(args.work, args.scene, tag)
                ctx = SamplerContext(pspec, pspec.scenes[args.scene], choices, world, kit, out_dir, tag, spec_file.sha256, base_seed=ri)
                s = Sampler(ctx)
                t1 = time.time()
                s.setup()
                rec["setup_s"] = round(time.time() - t1, 1)
                for i, sp in enumerate(s.survey):
                    sp["idx"] = i
                regions = regions_of(s.survey, Sampler.SPOT_STEP * getattr(s, "spot_stride", 1))
                rec["spot_stride"] = getattr(s, "spot_stride", 1)
                rec["n_ray_origins"] = s.n_ray_origins
                rec["n_spots"] = len(s.survey)
                rec["n_target_ok"] = sum(1 for sp in s.survey if sp.get("target_ok"))
                rec["n_viable"] = sum(1 for sp in s.survey if sp.get("viable"))
                rec["regions"] = regions
                say(f"{room}: {rec['n_spots']} spots, {rec['n_target_ok']} flat+clear, {rec['n_viable']} viable -> {len(regions)} regions "
                    f"(setup {rec['setup_s']}s)")
                rec["walls"] = sorted({sp["wall_name"] for sp in s.survey}) or rec["walls"]
                # tier 2: one instance per region at its middle spot; if that fails, the spot of the region with the
                # most pushing-capable bases; long regions also get their quarter points
                if not args.no_instances and regions:
                    todo = []  # (region id, [spot indices to try in order])
                    for r in sorted(regions, key=lambda r: -r["length_m"]):
                        idxs = r["spot_indices"]
                        mid = idxs[len(idxs) // 2]
                        best = max(idxs, key=lambda i: s.survey[i].get("n_reach", 0))
                        todo.append((r["id"], [mid] + ([best] if best != mid else [])))
                        if r["length_m"] >= 2.0 and len(idxs) >= 4:
                            todo.append((r["id"], [idxs[len(idxs) // 4]]))
                            todo.append((r["id"], [idxs[(3 * len(idxs)) // 4]]))
                    todo = todo[: args.max_instances_per_room]
                    inst = []
                    for k, (rid, spot_idxs) in enumerate(todo):
                        for si in spot_idxs:
                            sp = s.survey[si]
                            say(f"{room}: instance {k + 1}/{len(todo)} region {rid} spot {si} on {sp['wall_name']} at {[round(float(v), 2) for v in sp['point'][:2]]}")
                            try:
                                r_inst = s.validate_spot(sp, os.path.join(out_dir, f"{rid}_spot{si}"), max_attempts=args.max_attempts)
                            except Reject as e:
                                r_inst = {"n_written": 0, "rejections": {e.reason: 1}, "error": str(e)[:600]}
                            except Exception as e:
                                traceback.print_exc()
                                r_inst = {"n_written": 0, "error": repr(e)[:600]}
                                if "memory" in repr(e).lower():
                                    say("out of memory; exiting for a restart")
                                    os.remove(path + ".inprogress")
                                    os._exit(3)
                            r_inst.update({"region": rid, "spot": si, "wall_name": sp["wall_name"], "point": [round(float(v), 3) for v in sp["point"].tolist()]})
                            inst.append(r_inst)
                            say(f"{room}: region {rid} spot {si} -> written {r_inst['n_written']} rejections {r_inst.get('rejections')} {r_inst.get('error', '')[:200]}")
                            if r_inst["n_written"]:
                                break
                            err = str(r_inst.get("error", ""))
                            if any(k in err for k in ("not a unit quaternion", "singular", "nan")):
                                # a physics blow-up (NaN pose) corrupts the simulator state for the rest of the room:
                                # stop here, the room can be re-run (delete its result json) once the process restarts
                                rec["sim_error"] = err[:300]
                                say(f"{room}: simulator state corrupt after a physics blow-up; skipping the remaining instances of this room")
                                break
                        if rec.get("sim_error"):
                            break
                    rec["instances"] = inst
                    ok_regions = {i["region"] for i in inst if i["n_written"]}
                    for r in regions:
                        r["validated"] = r["id"] in ok_regions
                        r["tested"] = r["id"] in {i["region"] for i in inst}
                rec["spots"] = jsonable(s.survey)
                rec["template_meta"] = jsonable({k: v for k, v in s.template_meta.items() if k in ("timings", "rejections", "n_written")})
                rec["status"] = "ok"
                try:
                    s.teardown()
                except Exception:
                    pass
            except Reject as e:
                rec.update({"status": "setup_failed", "reject_reason": e.reason, "error": str(e)[:2000]})
                say(f"{room}: setup failed: {e}")
            except Exception as e:
                traceback.print_exc()
                rec.update({"status": "error", "error": repr(e)[:2000]})
                if "memory" in repr(e).lower():
                    say("out of memory; exiting for a restart")
                    if os.path.exists(path + ".inprogress"):
                        os.remove(path + ".inprogress")
                    os._exit(3)
            finally:
                rec["room_s"] = round(time.time() - t_room, 1)
                with open(path, "w") as f:
                    json.dump(jsonable(rec), f, indent=1, default=str)
                if os.path.exists(path + ".inprogress"):
                    os.remove(path + ".inprogress")
                if world is not None:
                    try:
                        for o in list(world.spawned.values()):
                            world.remove(o)
                    except Exception:
                        traceback.print_exc()
                    try:
                        og.clear()
                    except Exception:
                        traceback.print_exc()
            say(f"[{ri + 1}/{len(rooms)}] {room} -> {rec.get('status')} ({rec['room_s']:.0f}s; elapsed {(time.time() - t_start) / 60:.0f} min)")
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
