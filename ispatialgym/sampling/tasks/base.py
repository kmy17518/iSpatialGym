"""Base class shared by every task sampler: role binding, spawning, concrete BDDL, the per-instance
settle / validate / goal-feasibility pipeline and the TRO / template writers.

Task-specific samplers implement :meth:`TaskSampler.bind_fixed_roles`, :meth:`prepare_template` (cached
geometry) and :meth:`sample` (one :class:`Draft`), and may override the hooks marked *hook* below.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling import robot_utils  # noqa: F401  (warp pinning before omnigibson)

import omnigibson as og  # noqa: E402
from ispatialgym.sampling import predicates as PRED  # noqa: E402  (robust under / ontop, D-007 option A)

from ispatialgym.activity.problem import AGENT_INSTANCE  # noqa: E402
from ispatialgym.eval.group_entity import GroupEntity  # noqa: E402
from ispatialgym.eval.tro import apply_targets_to_scope, physical_instances  # noqa: E402
from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from ispatialgym.sampling import geometry as G  # noqa: E402
from ispatialgym.sampling import knowledge as K  # noqa: E402
from ispatialgym.sampling.bddl_writer import Bindings, Bound, build_problem  # noqa: E402
from ispatialgym.sampling.cache import cache_get, cache_key, cache_put  # noqa: E402
from ispatialgym.sampling.robot_utils import RobotKit, ViewSolution  # noqa: E402
from ispatialgym.sampling.sim_utils import Reject, SimWorld, SupportTop  # noqa: E402
from ispatialgym.sampling.spec import ModelChoice, ProblemSpec, SceneSpec, sha256_of_file, sha256_of_text  # noqa: E402
from ispatialgym.sampling.tro_writer import build_tro, meta_from_dict, validate_and_write  # noqa: E402

log = logging.getLogger("ispatialgym.sampling")

TABLE_CATEGORIES = ("breakfast_table", "desk", "coffee_table", "conference_table", "pedestal_table", "countertop", "bar", "nightstand", "console_table", "side_table")

#: a path component is at most 255 bytes on ext4 (and most filesystems); the longest name derived from an instance id is
#: ``<iid>-tro_state.json`` (2026-09-25: the 3-distractor milk alias' id ran to 241 characters and the TRO write failed
#: with ENAMETOOLONG after an attempt had passed every check, 0925-044627)
INSTANCE_FILE_NAME_MAX = 255
INSTANCE_ID_MAX = INSTANCE_FILE_NAME_MAX - len("-tro_state.json")


def make_instance_id(alias: str, template_id: str, index: int) -> str:
    """``<alias>__<template_id>__<index:04d>``; when that exceeds :data:`INSTANCE_ID_MAX` bytes the template component is
    cut and suffixed with ``~<sha1[:8] of the full template id>`` (deterministic, still unique per template). The alias
    stays the first ``__`` component (``first_demo`` / the schedulers read it from the file name) and the consumers derive
    the id from the file name and the template dir from the path, never from the id's structure."""
    iid = f"{alias}__{template_id}__{index:04d}"
    if len(iid.encode("utf-8")) <= INSTANCE_ID_MAX:
        return iid
    import hashlib

    digest = hashlib.sha1(template_id.encode("utf-8")).hexdigest()[:8]
    keep = max(INSTANCE_ID_MAX - len(f"{alias}____{index:04d}".encode("utf-8")) - len(digest) - 1, 0)
    return f"{alias}__{template_id[:keep]}~{digest}__{index:04d}"


@dataclass
class Draft:
    """Everything one instance attempt decided; filled progressively by the sampler and the pipeline."""

    object_poses: Dict[str, G.Pose] = field(default_factory=dict)  # bddl instance -> init root pose
    object_joints: Dict[str, th.Tensor] = field(default_factory=dict)
    robot_q: Optional[th.Tensor] = None  # 28-D (base virtual joints ignored)
    base_pose2d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    pose_targets: Dict[str, Any] = field(default_factory=dict)
    groups: Dict[str, Any] = field(default_factory=dict)
    articulation_targets: Dict[str, Any] = field(default_factory=dict)
    goal_poses: Dict[str, G.Pose] = field(default_factory=dict)
    goal_joints: Dict[str, th.Tensor] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)  # grasps / constraints / attachment_targets / initial_occlusion
    summary: Dict[str, Any] = field(default_factory=dict)
    settle_free: Optional[List[str]] = None  # objects released during settling (default: all spawned)
    layout_template_id: Optional[str] = None


@dataclass
class Stance:
    edge_name: str
    edge_mid: th.Tensor  # world xy of the front-edge midpoint
    inward: th.Tensor  # unit xy normal pointing from the edge into the support
    tangent: th.Tensor  # unit xy along the edge (robot's left = +tangent when facing inward)
    base_pose2d: Tuple[float, float, float]
    q_stance: th.Tensor  # 28-D with base virtual joints encoding base_pose2d (root frame)
    view: Optional[ViewSolution] = None

    @property
    def base_xy(self) -> th.Tensor:
        return th.tensor(self.base_pose2d[:2])

    @property
    def edge_depth(self) -> float:
        """Distance from the base centre to the front edge along the inward direction."""
        return float(th.dot(self.edge_mid - self.base_xy, self.inward))

    def local(self, xy) -> Tuple[float, float]:
        """(depth into the support from the front edge, lateral offset from the robot's centre line; +lateral = robot's left)."""
        rel = G.t(xy)[:2] - self.base_xy
        return float(rel @ self.inward) - self.edge_depth, float(rel @ self.tangent)

    def world(self, depth: float, lateral: float) -> th.Tensor:
        return self.base_xy + self.inward * (self.edge_depth + depth) + self.tangent * lateral

    def to_json(self):
        d = {"edge": self.edge_name, "edge_mid": self.edge_mid.tolist(), "inward": self.inward.tolist(), "tangent": self.tangent.tolist(),
             "base_pose2d": list(self.base_pose2d), "edge_depth": self.edge_depth}
        if self.view is not None:
            d["view"] = {"translate": self.view.translate, "tilt": self.view.tilt, "axis_angle": self.view.angle}
        return d


class SamplerContext:
    def __init__(self, problem_spec: ProblemSpec, scene_spec: SceneSpec, choices: Dict[str, ModelChoice], world: SimWorld, kit: RobotKit,
                 out_dir: str, template_id: str, spec_sha256: str, base_seed: int):
        self.problem_spec = problem_spec
        self.scene_spec = scene_spec
        self.choices = choices
        self.world = world
        self.kit = kit
        self.out_dir = out_dir
        self.template_id = template_id
        self.spec_sha256 = spec_sha256
        self.base_seed = base_seed


class TaskSampler:
    activity: str = ""
    definition_id: int = 0
    #: settling steps and tolerances shared by all tasks
    DEFAULT_SETTLE_STEPS = 90
    #: trunk translates tried for reaching ("sitting" levels). Measured on the R1Pro over a 0.80 m table: translates
    #: beyond 0.75 never added a reachable cell and each one costs a full IK pass, so the default stops there
    #: (``generation.reach_translates`` overrides, e.g. for low supports).
    DEFAULT_TRANSLATES = [0.5, 0.75]

    def __init__(self, ctx: SamplerContext):
        self.ctx = ctx
        self.spec = ctx.problem_spec
        self.scene_spec = ctx.scene_spec
        self.choices = ctx.choices
        self.world: SimWorld = ctx.world
        self.kit: RobotKit = ctx.kit
        self.robot = self.world.robot
        self.activity = self.spec.activity
        self.definition_id = self.spec.definition_id
        self.bindings = Bindings()
        self.objects: Dict[str, Any] = {}  # bddl instance -> simulator object
        self.problem = None
        self.compiled = None
        self.scope: Dict[str, Any] = {}
        self.template_meta: Dict[str, Any] = {"template_id": ctx.template_id, "scene": self.world.scene_model, "timings": {}}
        self.template_paths: Optional[Tuple[str, str]] = None
        self.template_sha: Optional[str] = None
        self.rejections: Dict[str, int] = {}

    # ------------------------------------------------------------------ spec helpers

    def gen(self, key: str, default=None):
        return self.spec.generation.get(key, default)

    #: arm configuration of the robot at init for every task unless the spec says otherwise (``generation.init_arms``):
    #: BEHAVIOR's reset pose. "ready" (hands in front, elbows bent) and "tucked" (hanging) are the alternatives.
    DEFAULT_INIT_ARMS = "untucked"

    def _scene_variant_meta(self) -> Optional[dict]:
        """Scene variant used (chairs removed, ...): scene file relative to the template dir, removed objects (``moved_objects``:
        parked outside the scene, with their pose in the scene file) and the traversability-map footprints to free at
        evaluation time (``scene_variants.patch_trav_map_from_meta``)."""
        sv = getattr(self.world, "scene_variant", None) or {}
        if not sv:
            return None
        out = {"scene_file": os.path.relpath(sv["path"], self.ctx.out_dir), "removed_objects": list(sv.get("removed_objects", [])),
               "freed_footprints": sv.get("freed_footprints")}
        if sv.get("moved_objects"):
            out["moved_objects"] = sv["moved_objects"]
        return out

    def init_arms(self) -> str:
        """Init arm mode from the spec; fixed per task so that every instance of a task starts in the same arm pose
        (no silent fallback: ``generation.view_arm_fallbacks`` must list alternatives explicitly)."""
        return str(self.gen("init_arms", self.DEFAULT_INIT_ARMS))

    def tol(self, key: str, default: float) -> float:
        return float((self.gen("tolerances") or {}).get(key, default))

    # ------------------------------------------------------------------ template phase

    def setup(self):
        t0 = time.time()
        k0 = self.kit.stats_snapshot()
        # the previous template in this world may have nudged furniture within the 5 cm tolerance after its last reset
        # (goal teleports, the stop / play cycle of object removal); the stance search must see the stable scene
        n_reset = self.world.reset_scene_objects()
        if n_reset:
            log.info("%s: %d scene objects reset before the template setup", self.spec.name, n_reset)
        # and the robot must be out of the way: left at the previous template's stance, its untucked arms hover over
        # the table and the support raster's downward rays hit them (seen: 32 cells of the dining table "occupied",
        # the source tray then had no supported position at the only usable end)
        self.world.park_robot(self.world.robot_park_xy, floor_z=self.world.floor_z)
        self.template_meta["init_arms"] = self.init_arms()
        self.bind_fixed_roles()
        self.spawn_roles()
        self.kit.sync_world()
        self.prepare_bindings()
        self.problem = build_problem(self.activity, self.definition_id, self.bindings)
        self.compiled = self.problem.compile()
        self.template_meta["bddl_sha256"] = sha256_of_text(self.problem.text)
        self.build_scope()
        self.template_meta["timings"]["bind_spawn_s"] = time.time() - t0
        k1 = self.kit.stats_snapshot()
        self.template_meta["timings"]["bind_spawn_kit"] = self.kit.stats_delta(k0, k1)
        t1 = time.time()
        self.kit.sync_world()
        self.prepare_template()
        self.template_meta["timings"]["prepare_template_s"] = time.time() - t1
        self.template_meta["timings"]["prepare_template_kit"] = self.kit.stats_delta(k1, self.kit.stats_snapshot())
        self.template_meta["bindings"] = [b.__dict__ for b in self.bindings.all()]

    def bind_fixed_roles(self):  # hook
        self.bind_floor()

    def prepare_bindings(self):  # hook: bindings that need the spawned objects' geometry (e.g. choosing the support)
        pass

    def prepare_template(self):  # hook: cached geometry once the problem is compiled
        pass

    def spawn_roles(self):
        """Spawn every non-fixed role of the spec (``count`` copies) and bind it."""
        for role in self.spec.spawned_roles():
            if role.name not in self.choices:
                continue
            self.spawn_role(role.name, self.choices[role.name], count=role.count)

    def spawn_role(self, role: str, choice: ModelChoice, count: int = 1, name_prefix: Optional[str] = None, extra_abilities=None) -> List[Any]:
        objs = []
        for i in range(count):
            idx = len(self.bindings.roles.get(role, []))
            name = f"{name_prefix or role}_{choice.category}_{choice.model}_{idx}"
            obj = self.world.spawn(name, choice.category, choice.model, dataset=choice.dataset, scale=choice.scale, extra_abilities=extra_abilities)
            b = self.bindings.add(role, choice.synset, choice.category, choice.model, obj.name, choice.dataset)
            self.objects[b.name] = obj
            objs.append(obj)
        return objs

    def bind_scene_object(self, role: str, obj) -> Bound:
        b = self.bindings.add(role, K.synset_of_category(obj.category), obj.category, obj.model, obj.name)
        self.objects[b.name] = obj
        return b

    def alias_role(self, role: str, bound: Bound):
        self.bindings.roles[role] = [bound]

    def bind_floor(self, role: str = "floor", room: Optional[str] = None):
        req = self.scene_spec.fixed.get(role)
        if isinstance(req, dict) and "room" in req:
            room = req["room"]
        if isinstance(req, list) and req:
            obj = self.world.obj(req[0])
        else:
            room = room or (self.scene_spec.rooms[0] if self.scene_spec.rooms else None)
            if room is None:
                raise Reject("spec", f"{self.spec.name}: no room / floor binding for role {role!r}")
            obj = self.world.floor_of_room(room)
        return self.bind_scene_object(role, obj)

    def fixed_candidates(self, role: str, default_categories: Sequence[str] = ()) -> List[Any]:
        """Scene objects the fixed ``role`` may bind to, in spec order."""
        req = self.scene_spec.fixed.get(role)
        if isinstance(req, list):
            return [self.world.obj(n) for n in req]
        cats = list((req or {}).get("category", default_categories)) if isinstance(req, dict) or req is None else []
        # category candidates come from the task rooms only (neighbouring rooms are loaded for their walls / furniture)
        rooms = set(self.world.rooms or [])
        out = [o for o in self.world.scene.objects if o.category in cats and o.name in self.world.scene_object_names
               and (not rooms or set(o.in_rooms or []) & rooms)]
        return sorted(out, key=lambda o: o.name)

    def build_scope(self):
        self.scope = {AGENT_INSTANCE: self.robot}
        for inst in self.problem.instances:
            if inst != AGENT_INSTANCE:
                self.scope[inst] = self.objects[inst]
        for g, members in self.problem.groups.items():
            self.scope[g] = GroupEntity(g, members, resolve_member=lambda n: self.scope.get(n))

    @property
    def spawned_instances(self) -> List[str]:
        return [b.name for b in self.bindings.all() if b.scene_object_name in self.world.spawned]

    @property
    def physical(self) -> List[str]:
        return physical_instances(self.compiled)

    def obj(self, role: str):
        return self.objects[self.bindings.one(role).name]

    def objs(self, role: str) -> List[Any]:
        return [self.objects[b.name] for b in self.bindings.many(role)]

    # ------------------------------------------------------------------ predicate evaluation

    def _evaluate(self, predicate_cls, *names):
        return PRED.evaluate_predicate(predicate_cls, *[self.scope[n] for n in names])

    def _refresh_for_predicates(self):
        """Wake every task object and run one *full* simulator step so contact / kinematic / AABB caches see the
        current state: ``step_physics`` alone leaves the per-step caches (collision boundary points, contact matrix)
        stale, and sleeping objects are treated as unchanged by ``KinematicsMixin``."""
        self.world.wake(self.objects.values())
        self.robot.keep_still()
        self.world.full_step()
        self.robot.keep_still()
        for e in self.scope.values():
            if isinstance(e, GroupEntity):
                e.invalidate()

    def init_holds(self) -> Tuple[bool, Any]:
        self._refresh_for_predicates()
        return self.compiled.check_initial_conditions(self._evaluate)

    def binary_goal_holds(self) -> Tuple[bool, Any]:
        self._refresh_for_predicates()
        PRED.reset_branches()
        _, status = self.compiled.check_goal(self._evaluate)
        return set(self.compiled.binary_goal_head_idxs) <= set(status["satisfied"]), status

    def configure_targets(self, draft: Draft):
        meta = meta_from_dict(self.tro_meta_block(draft, instance_id="draft", seed=0))
        apply_targets_to_scope(meta, self.scope)

    # ------------------------------------------------------------------ instance pipeline

    def sample(self, rng: np.random.Generator) -> Draft:  # abstract
        raise NotImplementedError

    def apply_draft(self, draft: Draft):  # hook
        for name, pose in draft.object_poses.items():
            obj = self.objects[name]
            obj.set_position_orientation(G.t(pose[0]), G.t(pose[1]))
            if obj.name in self.world.spawned:
                self.world.activate(obj)
            obj.keep_still()
        for name, q in draft.object_joints.items():
            obj = self.objects[name]
            obj.set_joint_positions(G.t(q))
            obj.set_joint_velocities(th.zeros_like(G.t(q)))
            obj.keep_still()
        self.kit.apply(draft.robot_q, draft.base_pose2d)
        for name in draft.object_poses:
            self.objects[name].keep_still()
        self.apply_draft_extras(draft)

    def apply_draft_extras(self, draft: Draft):  # hook (grasps, hinges, ...)
        pass

    def settle_entities(self, draft: Draft) -> List[Any]:
        names = draft.settle_free if draft.settle_free is not None else self.spawned_instances
        return [self.objects[n] for n in names]

    def settle_still_entities(self, draft: Draft) -> List[Any]:  # hook
        """Entities held at rest during the settle / drift / trembling windows (2026-09-25: the scene support the task
        objects rest on - a free-standing table chatters on the floor mesh once the dropped trays nudge it and shakes
        everything on it; the demo loads it at rest and never kicks it). Default: none."""
        return []

    def settle_and_validate(self, draft: Draft):
        free = self.settle_entities(draft)
        # the residual-velocity report of the settle is advisory since 2026-09-24: PhysX reports 0.1-0.9 rad/s of
        # sign-alternating contact velocity on lying cylinders that do not move at all (32/32 bottle drops on tray uekqey
        # stayed within 1.5 mm over 5 s while never reading below 0.05 rad/s), so the 0.5 rad/s gate rejected still
        # scenes; anything that really moves fails the pose gate below (2 cm/s of rolling = 30 mm over its window)
        still = self.settle_still_entities(draft)
        vel_err = self.world.settle(free, steps=int(self.gen("settle_steps", self.DEFAULT_SETTLE_STEPS)), still=still)
        draft.summary["residual_velocity"] = vel_err
        # 2026-09-24: visible stillness over 1.5 s of frames (``SimWorld.tremble_after``) before the poses are recorded: the
        # velocity report above and ``drift_after`` below let PhysX contact chatter through (the milk bottles lying across the
        # seams of the gsxbym tray floor trembled from step 0 of every demo and never satisfied ``pose_settled``)
        W = self.world
        err = W.tremble_after(free, frames=int(self.gen("tremble_frames", W.TREMBLE_FRAMES)),
                              pos_tol=float(self.gen("tremble_pos_tol_m", W.TREMBLE_POS_TOL)), ang_tol=float(self.gen("tremble_ang_tol_rad", W.TREMBLE_ANG_TOL)),
                              drift_pos_tol=float(self.gen("tremble_drift_pos_tol_m", W.TREMBLE_DRIFT_POS_TOL)),
                              drift_ang_tol=float(self.gen("tremble_drift_ang_tol_rad", W.TREMBLE_DRIFT_ANG_TOL)), still=still)
        draft.summary["tremble"] = dict(getattr(W, "last_tremble", {}) or {})
        if err:
            # diagnostic (2026-09-25): what touches what at this moment - the robot's contacts and the free objects' contacts
            contacts = ""
            try:
                from omnigibson.utils.usd_utils import RigidContactAPI

                idx = W.scene.idx
                pairs = set(RigidContactAPI.get_contact_pairs(idx, [W.robot] + list(free), None, current_only=True))
                short = sorted({(a.split("/")[-2] + ":" + a.split("/")[-1], b.split("/")[-2] + ":" + b.split("/")[-1]) for a, b in pairs})
                contacts = f" [contacts: {short[:24]}]"
            except Exception as ce:  # noqa: BLE001
                contacts = f" [contacts unavailable: {ce!r}]"
            raise Reject("unstable_init" if vel_err else "trembling", f"{err}" + (f"; residual velocity: {vel_err}" if vel_err else "") + contacts)
        if vel_err:
            log.info("%s: residual velocity after the settle ignored, the objects are visually still: %s", getattr(getattr(self, "gp", None), "alias", self.__class__.__name__), vel_err)
        for name in draft.object_poses:
            p, q = self.objects[name].get_position_orientation()
            draft.object_poses[name] = (p.clone(), q.clone())
        for name in draft.object_joints:
            draft.object_joints[name] = self.objects[name].get_joint_positions().clone()
        err = self.world.check_scene_unchanged([o.name for o in self.objects.values()])
        if err:
            raise Reject("scene_disturbed", err)
        self.after_settle(draft)
        ok, status = self.init_holds()
        if not ok:
            raise Reject("init_false", str(status))
        self.configure_targets(draft)
        holds, status = self.binary_goal_holds()
        if holds:
            raise Reject("goal_satisfied_at_init", str(status))
        draft.summary["bddl_at_init"] = {"init_conditions_all_true": True, "goal": self.goal_report(status)}
        err = self.world.drift_after(free, steps=10, still=still)
        if err:
            raise Reject("drift", err)
        self.extra_checks(draft)

    def after_settle(self, draft: Draft):  # hook: refresh draft data that depends on settled poses
        pass

    def extra_checks(self, draft: Draft):  # hook: renders, reachability, ...
        pass

    def goal_entities(self, draft: Draft) -> List[Any]:
        return [self.objects[n] for n in list(draft.goal_poses) + list(draft.goal_joints)]

    def apply_goal(self, draft: Draft):  # hook
        for name, pose in draft.goal_poses.items():
            obj = self.objects[name]
            obj.set_position_orientation(G.t(pose[0]), G.t(pose[1]))
            obj.keep_still()
        for name, q in draft.goal_joints.items():
            obj = self.objects[name]
            obj.set_joint_positions(G.t(q))
            obj.set_joint_velocities(th.zeros_like(G.t(q)))
            obj.keep_still()

    def check_goal_feasible(self, draft: Draft):
        """Teleport the objects to the goal, settle, require the binary goal to hold, restore. The robot is parked
        out of the way meanwhile: the goal must be feasible *excluding* the robot, and a hand hovering where the goal
        object lands would eject it (seen in disembedding). The snapshot restore puts the robot back."""
        if not draft.goal_poses and not draft.goal_joints:
            return
        state = self.world.snapshot()
        try:
            self.world.park_robot(self.world.robot_park_xy, floor_z=self.world.floor_z)
            self.apply_goal(draft)
            ents = self.goal_entities(draft)
            err = self.world.settle(ents, steps=int(self.gen("goal_settle_steps", 2 * self.DEFAULT_SETTLE_STEPS)), still=self.settle_still_entities(draft))
            if err:
                raise Reject("goal_unstable", err)
            self.refine_targets_from_goal(draft)
            self.configure_targets(draft)
            holds, status = self.binary_goal_holds()
            settled = {o.name: [[round(v, 3) for v in o.get_position_orientation()[0].tolist()], [round(v, 3) for v in o.aabb[0].tolist()], [round(v, 3) for v in o.aabb[1].tolist()]] for o in ents}
            if not holds:
                failed = [c["bddl"] for c in self.goal_report(status) if not c["satisfied"] and "pose_settled" not in c["bddl"]]
                raise Reject("goal_infeasible", f"unsatisfied {failed}; goal objects (pos, aabb lo, aabb hi) {settled}")
            draft.summary["goal_check"] = {
                "settled_poses": {o.name: [o.get_position_orientation()[0].tolist(), o.get_position_orientation()[1].tolist()] for o in ents},
                "goal": self.goal_report(status),  # every binary goal atom evaluated true at the teleported+settled goal
            }
        finally:
            self.world.restore(state)
            self.world.keep_still(self.objects.values())
            self.robot.keep_still()
            self.configure_targets(draft)

    def refine_targets_from_goal(self, draft: Draft):  # hook: snap targets to the settled goal poses
        for name in draft.goal_poses:
            tgt = draft.pose_targets.get(name)
            if tgt is None:
                continue
            p, q = self.objects[name].get_position_orientation()
            if tgt.get("position_metric") == "full" and tgt.get("frame", "world") == "world":
                tgt["position"] = p.tolist()
            if tgt.get("orientation_metric") == "full":
                tgt["orientation"] = q.tolist()

    def goal_report(self, status):
        satisfied = set(status["satisfied"])
        return [{"bddl": " ".join(str(self.problem.goal[i]).split()), "satisfied": i in satisfied} for i in range(len(self.problem.goal))]

    # ------------------------------------------------------------------ writing

    def object_bindings(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for b in self.bindings.all():
            d = {"scene_object_name": b.scene_object_name, "category": b.category, "model": b.model}
            if b.dataset and b.dataset != "behavior-1k-assets":
                d["dataset_name"] = b.dataset
            out[b.name] = d
        return out

    def scene_block(self) -> Dict[str, Any]:
        return {
            "model": self.world.scene_model,
            "floor": 0,
            "template_path": "../template.json",
            "template_sha256": self.template_sha,
            "load_room_instances": list(self.world.eval_rooms or []),  # task rooms + neighbours (what evaluation loads)
            "task_rooms": list(self.world.rooms or []),
            "sampled_with_rooms": list(self.world.loaded_rooms or []),
            "scene_variant": self._scene_variant_meta(),
            "required_scene_objects": sorted({b.scene_object_name for b in self.bindings.all() if b.scene_object_name in self.world.scene_object_names}),
        }

    def tro_meta_block(self, draft: Draft, instance_id: str, seed: int) -> Dict[str, Any]:
        robot_state = {
            "bddl_instance": AGENT_INSTANCE,
            "model": "r1pro",
            "joint_positions": self.kit.joint_dict(),
            "joint_velocities": {n: 0.0 for n in M.ALL_JOINTS if n not in M.BASE_VIRTUAL_JOINTS},
            "controller_targets": {},
        }
        p, q = self.robot.get_position_orientation()
        tro = build_tro(
            problem_name=self.spec.name,
            activity=self.activity,
            definition_id=self.definition_id,
            task_variant=self.problem.alias,
            instance_id=instance_id,
            generation_seed=seed,
            bddl_sha256=self.template_meta["bddl_sha256"],
            task_spec_sha256=self.ctx.spec_sha256,
            scene=self.scene_block(),
            object_bindings=self.object_bindings(),
            object_states={},
            robot_pose={"position": p.tolist(), "orientation": q.tolist()},
            robot_state=robot_state,
            pose_targets=draft.pose_targets,
            groups=draft.groups,
            articulation_targets=draft.articulation_targets,
            grasps=draft.extras.get("grasps"),
            constraints=draft.extras.get("constraints"),
            attachment_targets=draft.extras.get("attachment_targets"),
            initial_occlusion=draft.extras.get("initial_occlusion"),
            layout_template_id=draft.layout_template_id,
        )
        return tro["ispatialgym"]

    def instance_id(self, index: int) -> str:
        return make_instance_id(self.spec.name, self.ctx.template_id, index)

    def save_template(self):
        inst_to_name = {b.name: b.scene_object_name for b in self.bindings.all()}
        inst_to_name[AGENT_INSTANCE] = self.robot.name
        p, q = self.robot.get_position_orientation()
        robot_poses = {"robot": [{"position": p.tolist(), "orientation": q.tolist()}]}
        t0 = time.time()
        # the template must equal the stable scene except for the task objects and the robot
        n_reset = self.world.reset_scene_objects([o.name for o in self.objects.values()], context=f"{self.spec.name}: before saving the template")
        if n_reset:
            log.warning("%s: %d non-task scene objects were reset right before saving the template", self.spec.name, n_reset)
        partial, full = self.world.save_template(self.ctx.out_dir, inst_to_name, robot_poses)
        self.template_paths = (partial, full)
        self.template_sha = sha256_of_file(full)
        bddl_dir = os.path.join(self.ctx.out_dir, "activity_definitions", self.activity)
        os.makedirs(bddl_dir, exist_ok=True)
        with open(os.path.join(bddl_dir, f"problem{self.definition_id}.bddl"), "w") as f:
            f.write(self.problem.text)
        self.template_meta["timings"]["save_template_s"] = time.time() - t0
        self.template_meta.update({
            "template_sha256": self.template_sha,
            "template_path": full,
            "template_partial_path": partial,
            "load_room_instances": list(self.world.eval_rooms or []),
            "task_rooms": list(self.world.rooms or []),
            "sampled_with_rooms": list(self.world.loaded_rooms or []),
            "scene_variant": self._scene_variant_meta(),
            "problem": self.spec.name,
            "alias": self.problem.alias,
            "choices": {r: c.__dict__ for r, c in self.choices.items()},
            "inst_to_name": inst_to_name,
            "scene_load_s": self.world.load_time,
        })

    def write_template_meta(self):
        with open(os.path.join(self.ctx.out_dir, "template_meta.json"), "w") as f:
            json.dump(_jsonable(self.template_meta), f, indent=2)

    def write_instance(self, draft: Draft, index: int, seed: int) -> str:
        iid = self.instance_id(index)
        self.world.keep_still(self.objects[inst] for inst in self.physical)  # a state at rest (see v2/core.write_instance)
        states = {inst: self.objects[inst].dump_state(serialized=False) for inst in self.physical}
        meta = self.tro_meta_block(draft, iid, seed)
        tro = dict(states)
        p, q = self.robot.get_position_orientation()
        tro["robot_poses"] = {"robot": [{"position": p.tolist(), "orientation": q.tolist()}]}
        tro["ispatialgym"] = meta
        path = os.path.join(self.ctx.out_dir, "instances", f"{iid}-tro_state.json")
        validate_and_write(tro, self.compiled, path)
        summary = dict(draft.summary)
        summary.update({"instance_id": iid, "seed": seed, "base_pose2d": list(draft.base_pose2d), "trunk": list(self.kit.trunk_of(draft.robot_q)),
                        "rejections_so_far": dict(self.rejections)})
        with open(os.path.join(self.ctx.out_dir, "instances", f"{iid}.summary.json"), "w") as f:
            json.dump(_jsonable(summary), f, indent=2)
        return path

    # ------------------------------------------------------------------ main loop

    def run(self, n_instances: int, max_attempts: int = 30) -> List[str]:
        written = []
        for index in range(n_instances):
            ok = False
            for attempt in range(max_attempts):
                seed = int(self.ctx.base_seed * 1000003 + index * 1009 + attempt)
                rng = np.random.default_rng(seed)
                t0 = time.time()
                k0 = self.kit.stats_snapshot()
                try:
                    # every attempt starts from the stable scene: whatever physics did to non-task objects during the
                    # previous attempt must not leak into this instance (or into the template)
                    n_reset = self.world.reset_scene_objects([o.name for o in self.objects.values()])
                    if n_reset:
                        self.rejections["scene_reset"] = self.rejections.get("scene_reset", 0) + n_reset
                        log.info("%s: %d non-task scene objects had drifted and were reset", self.spec.name, n_reset)
                    draft = self.sample(rng)
                    t_sample = time.time() - t0
                    self.apply_draft(draft)
                    t_apply = time.time() - t0 - t_sample
                    self.settle_and_validate(draft)
                    t_settle = time.time() - t0 - t_sample - t_apply
                    self.check_goal_feasible(draft)
                    if self.template_paths is None:
                        self.save_template()
                    draft.summary["wall_time_s"] = time.time() - t0
                    draft.summary["phase_s"] = {"sample": round(t_sample, 2), "apply": round(t_apply, 2), "settle_validate": round(t_settle, 2),
                                                "goal_check_and_write": round(time.time() - t0 - t_sample - t_apply - t_settle, 2)}
                    draft.summary["kit"] = self.kit.stats_delta(k0, self.kit.stats_snapshot())
                    written.append(self.write_instance(draft, index, seed))
                    log.info("%s: instance %d written (attempt %d, %.1fs)", self.spec.name, index, attempt, time.time() - t0)
                    ok = True
                    break
                except Reject as e:
                    self.rejections[e.reason] = self.rejections.get(e.reason, 0) + 1
                    log.info("%s: instance %d attempt %d rejected: %s", self.spec.name, index, attempt, e)
                    self.on_reject(e)
            if not ok:
                log.error("%s: instance %d failed after %d attempts; rejections=%s", self.spec.name, index, max_attempts, self.rejections)
        self.template_meta["rejections"] = dict(self.rejections)
        self.template_meta["n_written"] = len(written)
        if self.template_paths is not None:
            self.write_template_meta()
        return written

    def on_reject(self, err: Reject):  # hook
        self.park_spawned_objects(only_moved=False)

    def park_spawned_objects(self, only_moved: bool = True) -> int:
        """Release any assisted grasp and park every spawned object of the template where ``SimWorld.spawn`` parked it (slot ``i``
        of ``PARK_POSITION``), then one physics step - the ``on_reject`` parking, shared since train-fast 2026-10-31 so that
        ``PARK_SPAWNED`` can run it at the start of every attempt (after a written instance the objects would otherwise still sit on
        the support while the next attempt samples). ``only_moved`` (the reset-phase call): an object already within 1 cm of its
        slot is left alone and the step is skipped when nothing moved; ``on_reject`` parks all and always steps (as before).
        Returns the number of objects parked."""
        from ispatialgym.sampling.sim_utils import PARK_POSITION

        try:
            for arm in M.ARMS:
                if self.robot._ag_obj_in_hand.get(arm) is not None:
                    self.robot.release_grasp_immediately(arm)
        except Exception:
            pass
        n = 0
        # ``getattr``: a sampler driven without a live world (the pure-python run_instances tests stub ``world``) has nothing to park
        for i, o in enumerate((getattr(self.world, "spawned", None) or {}).values()):
            if only_moved:
                try:
                    at = o.get_position_orientation()[0].float()
                    if float(th.linalg.norm(at - (PARK_POSITION + th.tensor([i * 1.0, 0.0, 0.0])))) <= 0.01:
                        continue
                except Exception:  # noqa: BLE001 - unreadable pose: park it
                    pass
            self.world.park(o, i)
            n += 1
        if n or not only_moved:
            og.sim.step_physics()
        return n

    # ------------------------------------------------------------------ shared geometry helpers

    def support_top(self, support, ignore: Sequence[Any] = ()) -> SupportTop:
        return self.world.support_top(support, resolution=float(self.gen("support_resolution", 0.02)), ignore=list(ignore) + list(self.world.spawned.values()))

    def choose_stance(self, top: SupportTop, focus_xy: Optional[th.Tensor] = None, stand_off: Optional[float] = None, arms: str = "untucked",
                      translates: Optional[Sequence[float]] = None, preferred_edge: Optional[str] = None, lateral_step: float = 0.10,
                      edge_margin: float = 0.25) -> Stance:
        """Base pose in front of one edge of the support (facing inward), on the traversability map and whole-body
        collision free at every trunk translate in ``translates`` (viewing and reaching levels).

        Candidates are all (edge, lateral position) pairs; they are collision-checked in one batch. Preference: the
        edge named by ``preferred_edge`` / the spec (``fixed.<role>_front_edge``), then the lateral position closest to
        ``focus_xy`` (or to the edge centre), then the free run behind the robot.
        """
        stand_off = float(self.gen("stand_off", 0.45)) if stand_off is None else stand_off
        translates = list(self.gen("reach_translates", self.DEFAULT_TRANSLATES)) if translates is None else list(translates)
        if M.DEFAULT_TRUNK_TRANSLATE not in translates:
            translates = [M.DEFAULT_TRUNK_TRANSLATE] + translates
        q0 = self.kit.q_config(arms=arms, translate=M.DEFAULT_TRUNK_TRANSLATE)
        cands = []
        # the top raster can be inset from the object's footprint (bed frames, footboards, table aprons): measure the
        # stand-off from whichever is further out along the edge normal
        support = self.world.obj(top.name)
        lo, hi = support.aabb
        aabb_xy = G.aabb_corners(lo, hi)[:, :2]
        for name, mid, normal, length in top.edges():
            if preferred_edge is not None and name != preferred_edge:
                continue
            inward = -normal
            tangent = th.tensor([-float(inward[1]), float(inward[0])])  # robot's left when facing inward
            yaw = math.atan2(float(inward[1]), float(inward[0]))
            half = max(0.0, length / 2 - edge_margin)
            n_lat = max(1, int(round(2 * half / lateral_step)) + 1)
            want = float(th.dot(G.t(focus_xy) - mid, tangent)) if focus_xy is not None else 0.0
            overhang = max(0.0, float((aabb_xy @ normal).max()) - float(th.dot(mid, normal)))
            if overhang < 0.05:
                overhang = 0.0  # bevelled table edges: the arm reach over the table is marginal, keep the stance as is
            for lat in th.linspace(-half, half, n_lat).tolist():
                base_xy = mid + normal * (stand_off + overhang) + tangent * lat
                if not self.kit.trav_free(base_xy):
                    continue
                cands.append((name, mid, inward, tangent, length, lat, abs(lat - want), base_xy, yaw))
        if not cands:
            raise Reject("no_stance", f"no traversable stance around {top.name}")
        qs = []
        for c in cands:
            for tr in translates:
                qs.append(self.kit.q_at_base_pose(self.kit.with_trunk(q0, tr, 0.0), float(c[7][0]), float(c[7][1]), c[8]))
        free = self.kit.collision_free(th.stack(qs), margin=0.01).view(len(cands), len(translates))
        log.info("choose_stance(%s): %d candidates, %d free at default translate, %d free at all translates",
                 top.name, len(cands), int(free[:, 0].sum()), int(free.all(dim=1).sum()))
        good = []
        for c, f in zip(cands, free):
            if not bool(f[0]):  # must be free at the default (viewing) translate
                continue
            n_free_tr = int(f.sum())
            back = 0.0
            for d in (0.2, 0.4, 0.6, 0.8):
                if self.kit.trav_free(c[7] - c[2] * d):
                    back = d
                else:
                    break
            good.append((-n_free_tr, c[6], -back, c))
        if not good:
            raise Reject("no_stance", f"no collision-free stance around {top.name}")
        good.sort(key=lambda g: (g[0], g[1], g[2]))
        _, _, _, (name, mid, inward, tangent, length, lat, _, base_xy, yaw) = good[0]
        q = self.kit.q_at_base_pose(q0, float(base_xy[0]), float(base_xy[1]), yaw)
        st = Stance(name, mid, inward, tangent, (float(base_xy[0]), float(base_xy[1]), yaw), q)
        st.lateral = lat  # type: ignore[attr-defined]
        return st

    #: arm configuration assumed while the robot "sits" (deep trunk translates) to reach: hanging arms stay clear of the
    #: table top, whereas the untucked / ready hands would sweep through it. The oracle lowers the arms before sitting.
    REACH_ARMS = "tucked"

    def stance_free_translates(self, stance: Stance, translates: Sequence[float], arms: Optional[str] = None) -> List[float]:
        q0 = self.kit.q_config(arms=arms or self.REACH_ARMS)
        qs = th.stack([self.kit.q_at_base_pose(self.kit.with_trunk(q0, tr, 0.0), *stance.base_pose2d) for tr in translates])
        free = self.kit.collision_free(qs, margin=0.01)
        return [float(tr) for tr, f in zip(translates, free.tolist()) if f]

    def reach_seed(self, stance: Stance, translate: float) -> th.Tensor:
        """Configuration used as IK seed / locked-joint source when testing reachability at ``translate``."""
        q0 = self.kit.q_config(arms=self.REACH_ARMS, translate=translate)
        return self.kit.q_at_base_pose(q0, *stance.base_pose2d)

    def stance_with_view(self, top: SupportTop, depth_mid: float, arms: Optional[str] = None, stand_off: Optional[float] = None,
                         preferred_edge: Optional[str] = None, focus_xy: Optional[th.Tensor] = None, focus_z: Optional[float] = None,
                         prefer_translate: Optional[float] = None) -> Stance:
        """Stance plus the trunk (translate, tilt) that centres the point at ``depth_mid`` in front of the robot.
        ``prefer_translate`` (default spec ``view_prefer_translate`` or the JoyLo default 0.5) biases the trunk: low
        translates keep the camera high and behind the table edge (wide objects), high ones bring it close.

        ``arms`` defaults to the spec's ``init_arms``. Alternatives listed in ``generation.view_arm_fallbacks`` (none by
        default, so that all instances of a task share the arm pose) are tried when no viewing configuration with
        ``arms`` is collision free; the arm mode used is recorded in ``template_meta``."""
        arms = self.init_arms() if arms is None else arms
        errors = []
        for arm_mode in ([arms] + [a for a in self.gen("view_arm_fallbacks", []) if a != arms]):
            try:
                st = self._stance_with_view(top, depth_mid, arm_mode, stand_off, preferred_edge, focus_xy, focus_z, prefer_translate)
                self.template_meta["view_arms"] = arm_mode
                return st
            except Reject as e:
                errors.append(f"{arm_mode}: {e}")
        raise Reject("stance_collision", "; ".join(errors))

    def _stance_with_view(self, top: SupportTop, depth_mid: float, arms: str, stand_off, preferred_edge, focus_xy, focus_z, prefer_translate) -> Stance:
        st = self.choose_stance(top, focus_xy=focus_xy, stand_off=stand_off, arms=arms, preferred_edge=preferred_edge)
        base_xy = th.tensor(st.base_pose2d[:2])
        # point on the robot's centre line, depth_mid into the support from the edge
        d_edge = float(th.dot(st.edge_mid - base_xy, st.inward))
        focus = th.tensor([*(base_xy + st.inward * (d_edge + depth_mid)).tolist(), top.z + (0.05 if focus_z is None else focus_z)])
        translates = th.tensor(self.gen("view_translates", [0.0, 0.25, 0.5, 0.75, 1.0]))
        pref = float(self.gen("view_prefer_translate", M.DEFAULT_TRUNK_TRANSLATE)) if prefer_translate is None else float(prefer_translate)
        chosen = None
        n_cand = 0
        # exact centring first; if every centred configuration collides (low furniture close to the robot), accept
        # views that put the focus within 20 deg of the optical axis
        for max_angle in (math.radians(8.0), math.radians(20.0)):
            views = self.kit.solve_view(st.q_stance, focus, translates=translates, top_k=24, prefer_translate=pref, max_angle=max_angle)
            qs = th.stack([self.kit.with_trunk(st.q_stance, v.translate, v.tilt) for v in views])
            free = self.kit.collision_free(qs, margin=0.01)
            n_cand += len(views)
            for v, q, f in zip(views, qs, free.tolist()):
                if f:
                    chosen = (v, q)
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise Reject("stance_collision", f"no collision-free viewing trunk configuration ({n_cand} candidates)")
        st.view, st.q_stance = chosen
        st.view.cam_pose = self.kit.camera_pose(st.q_stance)
        st.focus = focus  # type: ignore[attr-defined]
        return st

    def region_cells(self, top: SupportTop, stance: Stance, depth_range: Tuple[float, float], lateral_half: float, margin: float = 0.04,
                     lateral_center: float = 0.0) -> th.Tensor:
        """Free cells of ``top`` within ``depth_range`` from the front edge and ``|lateral - lateral_center| <= lateral_half``
        (lateral measured from the robot's centre line, + = robot's left)."""
        cells = top.cells(free_only=True)
        rel = cells - stance.base_xy
        depth = rel @ stance.inward - stance.edge_depth
        lat = rel @ stance.tangent
        keep = (depth >= depth_range[0]) & (depth <= depth_range[1]) & ((lat - lateral_center).abs() <= lateral_half)
        if len(cells) and int(keep.sum()) == 0:
            log.info("region_cells(%s): %d free cells, depth %.2f..%.2f, lateral %.2f..%.2f; none within depth %s lateral %.2f+-%.2f",
                     top.name, len(cells), float(depth.min()), float(depth.max()), float(lat.min()), float(lat.max()), list(depth_range), lateral_center, lateral_half)
        cells = cells[keep]
        if len(cells) == 0:
            return cells
        kept = [c for c in cells if top.contains(c, margin)]
        return th.stack(kept) if kept else th.zeros(0, 2)

    def _cache_context(self, stance: Stance) -> Dict[str, Any]:
        """Everything the collision world / IK setup depends on: the scene (model + loaded rooms fix the scene objects),
        spawned objects that are already placed (e.g. trays), the stance and the IK settings."""
        from ispatialgym.sampling.sim_utils import PARK_POSITION

        placed = []
        for o in self.world.spawned.values():
            p, q = o.get_position_orientation()
            if float(th.linalg.norm(p[:2] - PARK_POSITION[:2])) > 1.0:
                placed.append((o.category, o.model, p, q))
        return {
            "scene": self.world.scene_model,
            "rooms": sorted(self.world.loaded_rooms or []),
            "base": list(stance.base_pose2d),
            "arms": self.REACH_ARMS,
            "ignore": sorted(o.name for o in self.grasp_ignore()),
            "seeds": int(getattr(self.kit.backend, "batch_ik_seeds", 12)),
            "floor_z": self.kit.floor_z,
            "placed": placed,
        }

    def reachable_cells(self, cells: th.Tensor, stance: Stance, z: float, translates=None, yaws=None, tilts=(0.0, 30.0)) -> Tuple[th.Tensor, List[Optional[float]]]:
        translates = list(self.gen("reach_translates", self.DEFAULT_TRANSLATES)) if translates is None else list(translates)
        translates = self.stance_free_translates(stance, translates)
        if not translates:
            return th.zeros(len(cells), dtype=th.bool), [None] * len(cells)
        yaw0 = stance.base_pose2d[2]
        yaws = [yaw0, yaw0 + math.pi / 2] if yaws is None else yaws
        key = cache_key("reach_grid", {**self._cache_context(stance), "cells": cells, "z": z, "translates": translates, "yaws": list(yaws), "tilts": list(tilts)})
        hit = cache_get(key)
        if hit is not None and len(hit.get("mask", [])) == len(cells):
            log.info("reach grid: cache hit (%d cells)", len(cells))
            tr = [None if i is None else translates[int(i)] for i in hit["tr"]]
            return th.tensor(hit["mask"], dtype=th.bool), tr
        q_seed = self.reach_seed(stance, translates[0])
        mask, tr = self.kit.reach_grid(q_seed, cells, z, yaws, translates, toward=stance.inward, tilts=tilts, ignore_for_grasp=self.grasp_ignore())
        cache_put(key, {"mask": mask.tolist(), "tr": [None if t is None else translates.index(t) for t in tr]})
        return mask, tr

    def grasp_ignore(self) -> List[Any]:
        """Objects ignored for the *grasp* pose collision check (the support: fingertips may touch / dip into it, the
        oracle's final approach is not collision checked). Overridden by samplers that know their support."""
        return [self.objects[b.name] for b in self.bindings.roles.get("table", [])]

    def reachable_poses(self, stance: Stance, eef_poses: Sequence[G.Pose], translates=None, ignore_for_last=None) -> Optional[float]:
        """Exact (32-seed) check that every eef pose is IK-feasible from the stance at some collision-free translate
        (the last pose is checked with the support -- or ``ignore_for_last`` -- ignored)."""
        translates = list(self.gen("reach_translates", self.DEFAULT_TRANSLATES)) if translates is None else list(translates)
        ignore = self.grasp_ignore() if ignore_for_last is None else list(ignore_for_last)
        for tr in self.stance_free_translates(stance, translates):
            if self.kit.reachable(eef_poses, self.reach_seed(stance, tr), ignore_for_last=ignore) is not None:
                return tr
        return None

    def grasp_combo_table(self, stance: Stance, cells: th.Tensor, pose_at, grasps, translates: Sequence[float],
                          yaws: Optional[Sequence[float]] = None, cell_translates: Optional[Sequence[Optional[float]]] = None) -> Dict[int, List[Tuple[float, Any, float]]]:
        """Per cell index: feasible (yaw, grasp, translate) combinations of the library grasps (screening IK), computed
        once per template so that instance sampling never has to search for a reachable grasp.

        ``pose_at(cell_xy, yaw) -> (pos, quat)`` gives the object pose. Translates are tried shallowest first; a cell
        stops being evaluated once it has combinations.
        """
        yaws = list(th.linspace(-math.pi, math.pi, 9)[:-1].tolist()) if yaws is None else list(yaws)
        table: Dict[int, List[Tuple[float, Any, float]]] = {}
        free_tr = self.stance_free_translates(stance, translates)
        ignore = self.grasp_ignore()
        by_name = {g.name: g for g in grasps}
        # the object pose per (cell, yaw) is what the eef poses derive from; the poses at yaw 0 pin down the height / rest
        key = cache_key("grasp_combos", {**self._cache_context(stance), "cells": cells, "translates": list(translates), "yaws": yaws,
                                         "grasps": [(g.key, g.name, g.position, g.quat) for g in grasps],
                                         "poses0": [list(pose_at(c, 0.0)[0]) + list(pose_at(c, 0.0)[1]) for c in cells]})
        hit = cache_get(key)
        if hit is not None:
            log.info("grasp combination table: cache hit (%d cells)", len(hit))
            for i_str, lst in hit.items():
                table[int(i_str)] = [(yaws[int(yi)], by_name[gn], float(translates[int(ti)])) for yi, gn, ti in lst if gn in by_name]
            return table
        for tr in free_tr:
            q_seed = self.reach_seed(stance, tr)
            todo = [i for i in range(len(cells)) if i not in table]
            if not todo:
                break
            pairs, index = [], []
            for i in todo:
                for yaw in yaws:
                    obj_pose = pose_at(cells[i], yaw)
                    for g in grasps:
                        pairs.append(self.grasp_eef_poses(obj_pose, g))
                        index.append((i, yaw, g))
            if not pairs:
                continue
            feas = self.kit.pairs_feasible(pairs, q_seed, ignore)
            for (i, yaw, g), s in zip(index, feas.tolist()):
                if s:
                    table.setdefault(i, []).append((yaw, g, float(tr)))
        cache_put(key, {str(i): [(yaws.index(yaw), g.name, list(translates).index(tr)) for yaw, g, tr in lst] for i, lst in table.items()})
        return table

    def coarse_cells(self, cells: th.Tensor, resolution: float, raster: float) -> th.Tensor:
        """Sub-sample raster cells (spacing ``raster``) to a coarser lattice of spacing ``resolution``."""
        if len(cells) == 0:
            return cells
        k = max(1, int(round(resolution / raster)))
        origin = cells.min(dim=0).values
        idx = th.round((cells - origin) / raster).long()
        keep = (idx[:, 0] % k == 0) & (idx[:, 1] % k == 0)
        return cells[keep]

    def visible_cells(self, cells: th.Tensor, cam_pose: G.Pose, z: float, height: float, margin: float = 0.05, radius: float = 0.0) -> th.Tensor:
        """Mask of cells whose object footprint (radius ``radius`` around the cell, from ``z`` to ``z + height``) projects
        inside the image with ``margin`` (fraction of the image size). Vectorised frustum test, no renders."""
        if len(cells) == 0:
            return th.zeros(0, dtype=th.bool)
        offsets = [(0.0, 0.0)] if radius <= 0 else [(radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)]
        ok = th.ones(len(cells), dtype=th.bool)
        for dx, dy in offsets:
            for zz in (z, z + height):
                pts = th.cat([cells + th.tensor([dx, dy]), th.full((len(cells), 1), float(zz))], dim=1)
                ok &= self.world.camera_model.inside(cam_pose, pts, margin)
        return ok

    def feasible_yaw_grasps(self, stance: Stance, xy, z_of_yaw, grasps, translates: Sequence[float], yaws: Optional[Sequence[float]] = None,
                            rng=None) -> List[Tuple[float, Any, float]]:
        """(yaw, grasp, translate) combinations for an object at ``xy`` whose library grasp (pre-grasp + grasp) passes the
        screening IK, evaluated in one batched call per translate. ``z_of_yaw(yaw) -> (pos, quat)`` gives the object pose.

        Returns the feasible combinations shuffled (if ``rng``) so that callers can take the first that also passes the
        support / view checks; each should be confirmed with :meth:`reachable_poses` (32 seeds) before use.
        """
        yaws = list(th.linspace(-math.pi, math.pi, 13)[:-1].tolist()) if yaws is None else list(yaws)
        out = []
        for tr in translates:
            q_seed = self.reach_seed(stance, tr)
            poses, index = [], []
            for yaw in yaws:
                obj_pose = z_of_yaw(yaw)
                for g in grasps:
                    poses.extend(self.grasp_eef_poses(obj_pose, g))
                    index.append((yaw, g))
            succ = self.kit.ik_batch(poses, q_seed).view(-1, 2).all(dim=1)
            for (yaw, g), s in zip(index, succ.tolist()):
                if s:
                    out.append((yaw, g, float(tr)))
            if out:
                break  # prefer the shallowest translate at which anything is feasible
        if rng is not None and out:
            perm = rng.permutation(len(out))
            out = [out[i] for i in perm]
        return out

    def grasp_eef_poses(self, obj_pose: G.Pose, grasp, pregrasp: float = 0.10) -> List[G.Pose]:
        """[pre-grasp, grasp] world eef poses for a library grasp (eef in object frame) at ``obj_pose``."""
        eef = G.compose(obj_pose, grasp.pose)
        axis = G.rotate(eef[1], th.tensor([0.0, 0.0, 1.0]))
        return [(eef[0] - axis * pregrasp, eef[1]), eef]

    def verify_grasp_reachable(self, stance: Stance, obj_pose: G.Pose, grasps, translates=None):
        """First library grasp whose pre-grasp and grasp are IK-feasible from the stance; returns (grasp, translate)."""
        for g in grasps:
            tr = self.reachable_poses(stance, self.grasp_eef_poses(obj_pose, g), translates)
            if tr is not None:
                return g, tr
        raise Reject("grasp_unreachable", "no library grasp reachable from the stance")

    def frustum_ok(self, cam_pose: G.Pose, points: th.Tensor, margin: float = 0.05) -> bool:
        return self.world.camera_model.all_inside(cam_pose, points, margin)

    def hull_corners_world(self, local_hull: th.Tensor, pose: G.Pose) -> th.Tensor:
        pts = self.world.hull_at(local_hull, pose)
        lo, hi = pts.min(dim=0).values, pts.max(dim=0).values
        return G.aabb_corners(lo, hi)

    def non_structural_visible(self, min_pixels: int = 200) -> Dict[str, int]:
        px = self.world.visible_pixels(exclude_structure=True)
        return {k: v for k, v in px.items() if v >= min_pixels and k != self.robot.name}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, th.Tensor):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "__dict__") and not isinstance(x, type):
        return _jsonable(vars(x))
    return x
