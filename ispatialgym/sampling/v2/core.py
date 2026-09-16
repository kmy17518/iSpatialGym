"""``SamplerV2``: the v2 pipeline over the v1 ``TaskSampler`` machinery (spec §3.5, §3.11, §7, §9; README contract 7).

Kept from v1 (``sampling/tasks/base.py``): spawning, the BDDL scope, ``apply_draft``, ``settle_and_validate`` (settle 90 +
10, scene unchanged, ``:init`` true, binary goal false, drift), ``check_goal_feasible`` (teleport + settle with the robot
parked), ``save_template`` / ``write_instance`` (TRO written and validated against the BDDL), the summary fields.

New here: the ``GroundedProblem`` / ``TemplateChoice`` inputs (customs entry + grounded BDDL), the draft sidecar
(``self.sidecar``), the §3.3 parking rule, the program check as the last stage of every attempt, TRO + sidecar
(``<iid>.oracle.json`` with ``tro_sha256``), rejections per reason with the first three examples, the constraints
write-back, ``template_meta.json`` with the stances / cached geometry / spec hash, and the task-module hooks:

.. code-block:: python

    @register("inter_object_relation_satisfaction-ontop_small_scale")
    class IORSOntopSmall(SamplerV2):
        def bind_scene_roles(self): ...        # supports / partners / walls / rooms from customs.scenes[scene].bindings
        def sample_template(self): ...         # per-template facts: bands, stable joint set S, pair admissibility
        def sample_instance(self, rng): ...    # per-instance draws -> Draft (object poses, stances, sidecar blocks)
        def goal(self, draft): ...             # goal poses / joint targets -> pose_targets / articulation_targets

Optional hooks: ``after_goal_check(draft)`` (checks that need the settled goal pose, e.g. the place IK), ``record_
constraints(draft)`` (typed records appended to the customs entry), ``support_objects()`` (ignored for the final grasp
pose), ``v1_role_aliases()`` (facade BDDL rendering until D1).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling import robot_utils  # noqa: F401  (warp pinning before omnigibson)

import omnigibson as og  # noqa: E402

from ispatialgym.activity.problem import AGENT_INSTANCE, ISGProblem  # noqa: E402
from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from ispatialgym.robot import conventions as C  # noqa: E402
from ispatialgym.sampling import geometry as G  # noqa: E402
from ispatialgym.sampling import knowledge as K  # noqa: E402
from ispatialgym.sampling.bddl_writer import Bindings, Bound, build_problem  # noqa: E402
from ispatialgym.sampling.robot_utils import RobotKit  # noqa: E402
from ispatialgym.sampling.sim_utils import Reject, SimWorld  # noqa: E402
from ispatialgym.sampling.spec import ModelChoice, sha256_of_file  # noqa: E402
from ispatialgym.sampling.tasks.base import Draft, SamplerContext, TaskSampler, _jsonable  # noqa: E402
from ispatialgym.sampling.tro_writer import validate_and_write  # noqa: E402
from ispatialgym.sampling.v2 import SAMPLER_VERSION  # noqa: E402
from ispatialgym.sampling.v2 import navigation as NAV  # noqa: E402
from ispatialgym.sampling.v2 import program_check as PC  # noqa: E402
from ispatialgym.sampling.v2.ik import IKChecker  # noqa: E402
from ispatialgym.sampling.v2.problem import GroundedProblem, RejectionLog, TemplateChoice, template_summary  # noqa: E402
from ispatialgym.sampling.v2.stances import PathRecord, SidecarDraft, Stance, Transition  # noqa: E402

log = logging.getLogger("ispatialgym.sampling.v2")

# ----------------------------------------------------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------------------------------------------------

REGISTRY: Dict[str, type] = {}


def register(*subtasks: str) -> Callable[[type], type]:
    """Class decorator: ``@register("<subtask id>", ...)`` makes the class the sampler of those subtasks."""

    def deco(cls: type) -> type:
        for st in subtasks:
            if st in REGISTRY and REGISTRY[st] is not cls:
                log.warning("subtask %s: sampler %s replaces %s", st, cls.__name__, REGISTRY[st].__name__)
            REGISTRY[st] = cls
        cls.subtasks = tuple(subtasks)
        return cls

    return deco


def get_sampler(subtask: str) -> type:
    """The registered ``SamplerV2`` subclass for ``subtask`` (imports the task modules on first use)."""
    if subtask not in REGISTRY:
        from ispatialgym.sampling.v2 import load_tasks

        load_tasks()
    if subtask not in REGISTRY:
        raise KeyError(f"no v2 sampler registered for subtask {subtask!r}; have {sorted(REGISTRY)}")
    return REGISTRY[subtask]


# ----------------------------------------------------------------------------------------------------------------
# context
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class ContextV2:
    """What one template's sampler is built from."""

    problem: GroundedProblem
    choice: TemplateChoice
    world: SimWorld
    kit: RobotKit
    out_dir: str  # <out_root>/<alias>/<template_id>
    base_seed: int = 0
    customs_sha256: Optional[str] = None
    variant: Optional[str] = None  # task sub-variant (open / close, ...)

    @property
    def scene(self) -> str:
        return self.choice.scene

    @property
    def template_id(self) -> str:
        return self.choice.template_id

    @property
    def alias_dir(self) -> str:
        return os.path.dirname(os.path.abspath(self.out_dir))


#: v2 role names -> the v1 ``bddl_writer`` role names (facade rendering until the grounded BDDL exists)
V1_ROLE_MAP = {"support": "table"}


# ----------------------------------------------------------------------------------------------------------------
# sampler
# ----------------------------------------------------------------------------------------------------------------


class SamplerV2(TaskSampler):
    """Base of every v2 task module (README "How a task module plugs in")."""

    subtasks: Tuple[str, ...] = ()
    #: settle steps of §3.5 (90 + 10 in ``SimWorld.settle``); goal check 120
    DEFAULT_SETTLE_STEPS = 90
    GOAL_SETTLE_STEPS = 120

    def __init__(self, ctx: ContextV2):
        self.ctx2 = ctx
        self.gp: GroundedProblem = ctx.problem
        self.choice: TemplateChoice = ctx.choice
        self.scene: str = ctx.scene
        self.subtask: str = self.gp.subtask
        v1ctx = SamplerContext(self.gp, self.gp.scenes[self.scene], self.choice.choices, ctx.world, ctx.kit, ctx.out_dir, ctx.template_id,
                               ctx.customs_sha256 or self.gp.spec_hash(), ctx.base_seed)
        super().__init__(v1ctx)
        self.activity = self.gp.activity
        self.definition_id = self.gp.definition_id
        self.sidecar = SidecarDraft(self.gp.alias, self.gp.family)
        arm = self.gp.gen("arm", "right")
        default_arms = ["right"] if arm in (None, "auto", "right") else (["left"] if arm == "left" else ["right", "left"])
        self.ikc = IKChecker(self.kit, {"scene": self.world.scene_model, "rooms": sorted(self.world.loaded_rooms or []), "floor_z": self.kit.floor_z},
                             default_working_arms=default_arms)
        self.rejection_log = RejectionLog(keep=3)
        self.counters: Dict[str, int] = {}  # scene_reset, attempts, ... (not rejections)
        self.instance_seeds: List[int] = []
        self.template_facts: Dict[str, Any] = {}  # cached geometry of sample_template (written to template_meta)
        self.template_meta.update({"alias": self.gp.alias, "subtask": self.subtask, "scene": self.scene, "sampler_version": SAMPLER_VERSION,
                                   "spec_hash": self.gp.spec_hash(), "choice": self.choice.to_json(), "rooms": list(self.gp.rooms(self.scene)),
                                   "customs": self.gp.customs_path})
        self.variant = ctx.variant
        self.last_program_check: Optional[Dict[str, Any]] = None
        self._park_xy: Optional[th.Tensor] = None

    # ------------------------------------------------------------------ hooks (task modules)

    def bind_scene_roles(self):  # hook
        """Bind the scene roles (``support`` / ``partner`` / ``wall`` / ``floor``) from ``customs.scenes[scene].bindings``.
        The default binds ``floor`` (room of the bindings or the first task room) and every other scene role whose
        binding is a list of scene object names (first name). Task modules override to choose among candidates."""
        self.bind_floor_v2()
        for role in self.gp.scene_roles():
            if role == "floor" or self.bindings.has(role):
                continue
            req = self.choice.bindings.get(role)
            if isinstance(req, list) and req and isinstance(req[0], str):
                self.bind_scene_object(role, self.world.obj(req[0]))

    def sample_template(self):  # hook
        """Per-template facts (bands, stable joint sets, pair admissibility, cached geometry) into ``self.template_facts``."""

    def sample_instance(self, rng: np.random.Generator) -> Draft:  # hook
        raise NotImplementedError

    def goal(self, draft: Draft):  # hook
        """Goal poses / joint targets -> ``draft.goal_poses`` / ``draft.goal_joints`` and ``draft.pose_targets`` /
        ``draft.articulation_targets``; the sidecar ``goal`` block."""

    def after_goal_check(self, draft: Draft):  # hook
        """Checks that need the settled goal pose (place IK from the manipulation stance, partner unmoved, ...)."""

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:  # hook
        """Typed, scoped constraint records (§2.3) learned from this successful instance (``source`` added by core)."""
        return []

    def support_objects(self) -> List[Any]:  # hook
        """Objects ignored for the final grasp / contact pose only (§3.6): the bound support(s)."""
        out = []
        for role in ("support", "tray_source", "tray_dest", "wall"):
            if self.bindings.has(role):
                out += [self.objects[b.name] for b in self.bindings.many(role)]
        return out

    def v1_role_aliases(self) -> Dict[str, str]:  # hook (facade only)
        """Extra v1 role names bound to v2 roles when the BDDL is rendered through ``bddl_writer`` (e.g. IORS small-scale:
        ``{"partner_support": "support"}``)."""
        return {}

    # ------------------------------------------------------------------ v1 overrides

    def grasp_ignore(self) -> List[Any]:
        return self.support_objects()

    def bind_fixed_roles(self):
        self.bind_scene_roles()

    def prepare_template(self):
        self.sample_template()

    def sample(self, rng: np.random.Generator) -> Draft:
        self.sidecar.reset_instance()
        draft = self.sample_instance(rng)
        if draft.robot_q is None and "init" in self.sidecar.stances:
            st = self.sidecar.stance("init")
            draft.robot_q = st.q if st.q is not None else self.ikc.q_locked(st)
            draft.base_pose2d = tuple(st.base_pose2d)
        self.goal(draft)
        return draft

    def bind_floor_v2(self, role: str = "floor"):
        req = self.choice.bindings.get(role)
        room = None
        if isinstance(req, dict) and req.get("room"):
            room = req["room"]
        elif isinstance(req, list) and req:
            return self.bind_scene_object(role, self.world.obj(req[0]))
        rooms = self.gp.rooms(self.scene)
        room = room or (rooms[0] if rooms else None)
        if room is None:
            raise Reject("spec", f"{self.gp.alias}: no room / floor binding in scene {self.scene}")
        return self.bind_scene_object(role, self.world.floor_of_room(room))

    # ------------------------------------------------------------------ spawning

    def spawn_roles(self):
        """Spawn every sampleable role and every group member of the template choice (scale from ``roles.<role>.scale``)."""
        for role in self.gp.sampleable_roles():
            if role in self.choice.models:
                self.spawn_role(role, self.choice.models[role])
        for role in self.gp.group_roles():
            for member in self.choice.groups.get(role, []):
                self.spawn_role(role, member)

    # ------------------------------------------------------------------ BDDL

    def build_bddl(self) -> ISGProblem:
        """The grounded BDDL of the alias with the bindings mapped onto its instances; when the alias has none yet (v1
        facade), render it with the v1 ``bddl_writer`` from the v1 role names."""
        if self.gp.bddl is not None:
            want = {i for i in self.gp.bddl.instances if i != AGENT_INSTANCE}
            have = {b.name for b in self.bindings.all()}
            if want != have:
                raise Reject("spec", f"bindings {sorted(have)} do not match the grounded BDDL instances {sorted(want)}")
            problem = self.gp.bddl
            problem.text = self.gp.bddl_text  # type: ignore[attr-defined]
            return problem
        v1 = Bindings()
        v1._counters = dict(self.bindings._counters)
        aliases = {**self.v1_role_aliases()}
        for role, lst in self.bindings.roles.items():
            v1.roles[V1_ROLE_MAP.get(role, role)] = list(lst)
        for v1_role, v2_role in aliases.items():
            if v2_role in self.bindings.roles:
                v1.roles[v1_role] = list(self.bindings.roles[v2_role])
        return build_problem(self.activity, self.definition_id, v1)

    # ------------------------------------------------------------------ template phase

    def setup(self):
        t0 = time.time()
        k0 = self.kit.stats_snapshot()
        n_reset = self.world.reset_scene_objects()
        if n_reset:
            log.info("%s: %d scene objects reset before the template setup", self.gp.alias, n_reset)
        self.park_robot()
        self.template_meta["init_arms"] = "untucked"
        self.bind_scene_roles()
        self.spawn_roles()
        self.kit.sync_world()
        self.prepare_bindings()
        self.problem = self.build_bddl()
        self.compiled = self.problem.compile()
        self.template_meta["bddl_sha256"] = _sha_text(self.problem.text)
        self.build_scope()
        self.template_meta["timings"]["bind_spawn_s"] = time.time() - t0
        k1 = self.kit.stats_snapshot()
        self.template_meta["timings"]["bind_spawn_kit"] = self.kit.stats_delta(k0, k1)
        t1 = time.time()
        self.kit.sync_world()
        self.run_template()
        self.template_meta["timings"]["sample_template_s"] = time.time() - t1
        self.template_meta["timings"]["sample_template_kit"] = self.kit.stats_delta(k1, self.kit.stats_snapshot())
        self.template_meta["bindings"] = [b.__dict__ for b in self.bindings.all()]

    def run_template(self):
        """Template facts once: bands, caches, representative test (``sample_template``); recorded in ``template_meta``."""
        self.park_robot()
        self.adopt_existing_template()
        try:
            self.sample_template()
        finally:
            # the facts gathered so far stay in template_meta (a rejected template documents where it failed)
            self.template_meta["template_facts"] = _jsonable(self.template_facts)

    def adopt_existing_template(self) -> bool:
        """``multiply_isg``: when ``<out_dir>/template.json`` exists (written by ``sample_isg`` with instance 0), keep it —
        the new instances must reference the same template (``scene.template_sha256`` in every TRO) — and continue its
        ``template_meta.json`` instead of re-saving. Returns True when a template was adopted."""
        full = os.path.join(self.ctx.out_dir, "template.json")
        partial = os.path.join(self.ctx.out_dir, "template-partial_rooms.json")
        meta_path = os.path.join(self.ctx.out_dir, "template_meta.json")
        if not (os.path.exists(full) and os.path.exists(meta_path)):
            return False
        with open(meta_path) as f:
            old = json.load(f)
        self.template_paths = (partial if os.path.exists(partial) else full, full)
        self.template_sha = sha256_of_file(full)
        keep = {k: v for k, v in old.items() if k not in ("timings", "rejections", "rejection_examples", "seeds", "kit_stats")}
        self.template_meta = {**keep, **self.template_meta, "timings": {**old.get("timings", {}), **self.template_meta.get("timings", {})}}
        self.template_meta["n_written_before"] = int(old.get("n_written", 0))
        self.template_meta["n_written"] = int(old.get("n_written", 0))
        for k, v in (old.get("rejections") or {}).items():
            if k == "scene_reset":  # written under rejections by the first runs; a counter since 2026-09-16
                self.counters[k] = self.counters.get(k, 0) + int(v)
                continue
            self.rejection_log.counts[k] = self.rejection_log.counts.get(k, 0) + int(v)
        for k, v in (old.get("counters") or {}).items():
            self.counters[k] = self.counters.get(k, 0) + int(v)
        self.instance_seeds = list(old.get("seeds") or [])
        log.info("%s: continuing template %s (%d instances so far, sha %s)", self.gp.alias, self.ctx.template_id, self.template_meta["n_written"], self.template_sha[:12])
        return True

    # ------------------------------------------------------------------ parking (§3.3)

    def park_robot(self) -> th.Tensor:
        """§3.3: tucked at a free cell of the task rooms' floor, >= 1 m from every spawned object and support."""
        keep = list(self.world.spawned.values()) + self.support_objects()
        try:
            xy = NAV.park_pose(self.kit, self.gp.rooms(self.scene) or (self.world.rooms or []), keep_away=keep)
        except NAV.NavigationError as e:
            log.warning("%s: %s; parking at the room centre", self.gp.alias, e)
            xy = self.world.robot_park_xy
        self._park_xy = xy
        self.world.park_robot(xy, floor_z=self.world.floor_z)
        self.kit.sync_world()
        return xy

    def check_goal_feasible(self, draft: Draft):
        """§3.5: teleport + settle (120 steps) with the robot parked per §3.3, every binary goal atom true, restore."""
        if not draft.goal_poses and not draft.goal_joints:
            return
        state = self.world.snapshot()
        try:
            self.park_robot()
            self.apply_goal(draft)
            ents = self.goal_entities(draft)
            err = self.world.settle(ents, steps=int(self.gen("goal_settle_steps", self.GOAL_SETTLE_STEPS)))
            if err:
                raise Reject("goal_unstable", err)
            self.refine_targets_from_goal(draft)
            self.configure_targets(draft)
            holds, status = self.binary_goal_holds()
            settled = {o.name: [[round(v, 3) for v in o.get_position_orientation()[0].tolist()]] for o in ents}
            if not holds:
                failed = [c["bddl"] for c in self.goal_report(status) if not c["satisfied"] and "pose_settled" not in c["bddl"]]
                raise Reject("goal_infeasible", f"unsatisfied {failed}; goal objects {settled}")
            draft.summary["goal_check"] = {
                "settled_poses": {o.name: [o.get_position_orientation()[0].tolist(), o.get_position_orientation()[1].tolist()] for o in ents},
                "goal": self.goal_report(status),
                "robot_parked_at": None if self._park_xy is None else [float(v) for v in self._park_xy],
            }
        finally:
            self.world.restore(state)
            self.world.keep_still(self.objects.values())
            self.robot.keep_still()
            self.kit.sync_world()
            self.configure_targets(draft)

    def check_stances_with_objects(self, draft: Draft):
        """§4.6 step 5: whole-body collision check of every recorded stance (init, manipulation, placement, ...) with the
        objects placed (the Phase-2 scan ran without them); ``Reject("stance_collision_with_objects")``. Runs after the
        settle (objects at their settled = recorded poses, robot at init)."""
        stances = [s for s in self.sidecar.stances.values() if s.name != "goal"]
        if not stances:
            return
        self.kit.sync_world()
        bad = []
        for st in stances:
            # a stance that carries an object is checked with it attached (posed in the hand) and its support ignored
            obj, arm, eef_in_object = self.carried_attachment(st.carried)
            with NAV.attached(self.kit, arm if obj is not None else None, obj, eef_in_object=eef_in_object):
                with self.ikc.ignoring([obj] if obj is not None else []):
                    if not self.ikc.stance_collision_free(st):
                        bad.append(st.name)
        if bad:
            raise Reject("stance_collision_with_objects", f"stances {bad} collide with the objects placed (trunks {[(s.translate, round(s.tilt, 2)) for s in stances if s.name in bad]})")
        # §3.8 with the objects placed: the recorded transitions are swept again (the smallest free s_T may change)
        for i, tr in enumerate(list(self.sidecar.transitions)):
            frm, to = self.sidecar.stances.get(tr.frm), self.sidecar.stances.get(tr.to)
            if frm is None or to is None:
                continue
            obj, arm, eef_in_object = self.carried_attachment(to.carried)
            with self.ikc.ignoring([obj] if obj is not None else []):
                new = NAV.transition(self.kit, frm, to, carried_obj=obj, carried_arm=arm, eef_in_object=eef_in_object)
            if new is None:
                raise Reject("no_transition", f"{tr.frm} -> {tr.to}: no swept-free s_T with the objects placed")
            self.sidecar.transitions[i] = new

    # ------------------------------------------------------------------ stances / transitions / paths helpers

    def stance_q(self, stance: Stance, **arm_q) -> th.Tensor:
        """28-D configuration of a stance, recomputed at every call: the virtual base joints are relative to the robot's
        articulation root at call time (``RobotKit.q_at_base_pose``), so a configuration must not outlive a teleport."""
        return self.ikc.q_locked(stance, **arm_q)

    def carried_attachment(self, carried: Optional[str], arm: Optional[str] = None) -> Tuple[Any, str, Optional[Any]]:
        """``(object, arm, T_obj→eef)`` of a carried bddl instance from its grasp record (``sidecar.grasps``) — what the
        checks attach and pose in the hand; ``(None, arm, None)`` when nothing is carried."""
        if not carried:
            return None, arm or "right", None
        g = self.sidecar.grasp_of(carried)
        obj = self.objects.get(carried)
        if g is None or g.eef_in_object is None:
            log.warning("%s: carried %s has no grasp record with eef_in_object; attached at its current pose", self.gp.alias, carried)
            return obj, arm or (g.arm if g else "right"), None
        return obj, arm or g.arm, g.eef_in_object

    def record_transition(self, frm: Stance, to: Stance, carried: Optional[str] = None, arm: Optional[str] = None) -> Transition:
        """§3.8 transition between two recorded stances (carried object attached, posed in the hand per its grasp record);
        ``Reject("no_transition")``."""
        carried = carried or to.carried
        obj, arm, eef_in_object = self.carried_attachment(carried, arm)
        tr = NAV.transition(self.kit, frm, to, carried_obj=obj, carried_arm=arm, eef_in_object=eef_in_object)
        if tr is None:
            raise Reject("no_transition", f"{frm.name} -> {to.name}: no s_T in {NAV.transition_standoffs(to.stand_off or 0.0)} is swept collision-free")
        return self.sidecar.add_transition(tr)

    def record_path(self, frm: Stance, to: Stance, carried: Optional[str] = None, arm: Optional[str] = None, max_length_m: Optional[float] = None,
                    blocked: Sequence[Any] = (), force_global: bool = False, trunk: Optional[Tuple[float, float]] = None) -> PathRecord:
        """§3.7 path between two recorded stances, checked at the departing stance's trunk (``trunk`` overrides) with the
        carried object attached (posed in the hand per its grasp record); ``force_global`` plans on the eroded map end to
        end; ``Reject(no_path | path_collision | path_too_long)``."""
        carried = carried or to.carried or frm.carried
        obj, arm, eef_in_object = self.carried_attachment(carried, arm)
        try:
            p = NAV.plan_path(self.kit, frm, to, carried=carried, carried_obj=obj, carried_arm=arm, rooms=self.gp.rooms(self.scene) or self.world.rooms,
                              blocked=blocked, max_length_m=max_length_m, trunk=trunk, eef_in_object=eef_in_object, force_global=force_global)
        except NAV.NavigationError as e:
            raise Reject(e.reason, str(e))
        return self.sidecar.add_path(p)

    def camera_pose_at(self, stance: Stance) -> G.Pose:
        return self.kit.camera_pose(self.stance_q(stance))

    # ------------------------------------------------------------------ program check (§3.5)

    def program_check(self, draft: Draft) -> Dict[str, Any]:
        report = PC.run(self.gp.alias, self.sidecar, self, subtask=self.subtask)
        self.last_program_check = report
        draft.summary["program_check"] = report
        return report

    def oracle_state(self):
        """State object handed to ``Action.targets(sidecar, state)`` (F6 contract): object poses by bddl instance."""
        return {"objects": {n: [x.tolist() for x in o.get_position_orientation()] for n, o in self.objects.items()},
                "robot": {"base_pose2d": list(self.kit.world_base_pose2d())}}

    # ------------------------------------------------------------------ writing

    def save_template(self):
        """v1 template writer; the grounded BDDL goes to ``<alias>/activity_definitions/<activity>/problem<N>.bddl`` (§1).
        The v1 copy inside the template directory stays until D5 moves ``validate_instances.py`` to the §1 layout
        (its loader reads ``<template>/activity_definitions``)."""
        super().save_template()
        bddl_dir = os.path.join(self.ctx2.alias_dir, "activity_definitions", self.activity)
        os.makedirs(bddl_dir, exist_ok=True)
        with open(os.path.join(bddl_dir, f"problem{self.definition_id}.bddl"), "w") as f:
            f.write(self.problem.text)
        self.template_meta["activity_definition"] = os.path.relpath(os.path.join(bddl_dir, f"problem{self.definition_id}.bddl"), self.ctx.out_dir)
        # v1 tools split ``problem`` as <activity>-<def_id>; the alias lives under ``alias``
        self.template_meta["problem"] = f"{self.activity}-{self.definition_id}"
        self.template_meta["alias"] = self.gp.alias
        self.template_meta["bddl_alias"] = self.problem.alias

    def write_instance(self, draft: Draft, index: int, seed: int) -> str:
        iid = self.instance_id(index)
        states = {inst: self.objects[inst].dump_state(serialized=False) for inst in self.physical}
        meta = self.tro_meta_block(draft, iid, seed)
        tro = dict(states)
        p, q = self.robot.get_position_orientation()
        tro["robot_poses"] = {"robot": [{"position": p.tolist(), "orientation": q.tolist()}]}
        tro["ispatialgym"] = meta
        inst_dir = os.path.join(self.ctx.out_dir, "instances")
        path = os.path.join(inst_dir, f"{iid}-tro_state.json")
        validate_and_write(tro, self.compiled, path)
        sidecar_path = os.path.join(inst_dir, f"{iid}.oracle.json")
        if not NAV.attached_checks_enabled():
            self.sidecar.waypoint_policy = "checked_without_attached_object"
        self.annotate_stance_arms()
        self.sidecar.finalize(iid, path)
        self.sidecar.write(sidecar_path)
        summary = dict(draft.summary)
        summary.update({"instance_id": iid, "seed": seed, "base_pose2d": list(draft.base_pose2d), "trunk": list(self.kit.trunk_of(draft.robot_q)),
                        "rejections_so_far": dict(self.rejection_log.counts), "sidecar": os.path.basename(sidecar_path),
                        "stances": {k: v.to_meta() for k, v in self.sidecar.stances.items()}, "sampler_version": SAMPLER_VERSION,
                        "checks": self.check_flags()})
        with open(os.path.join(inst_dir, f"{iid}.summary.json"), "w") as f:
            json.dump(_jsonable(summary), f, indent=2)
        if "stances" not in self.template_meta:
            self.template_meta["stances"] = {k: v.to_meta() for k, v in self.sidecar.stances.items()}
        return path

    def annotate_stance_arms(self) -> None:
        """A6: every recorded stance carries the arm joint values the checks used (``arms_q = {"right": [7], "left": [7]}``,
        ``fingers`` when not the open default) next to ``idle_posture`` in the sidecar, so the oracle can move the idle
        arm there before the trunk motion (unknown keys land in the F6 ``Sidecar.extra``)."""
        for st in self.sidecar.stances.values():
            r, l, f = self.ikc.arm_vectors(st)
            st.extras["arms_q"] = {"right": [round(float(v), 5) for v in r.tolist()], "left": [round(float(v), 5) for v in l.tolist()]}
            if abs(f - float(C.FINGER_OPEN)) > 1e-6:
                st.extras["fingers"] = f

    def constraints_write_back(self, draft: Draft, iid: str) -> List[Dict[str, Any]]:
        """§2.3 / §9: append the task module's records (+ the stance record core derives) with ``source``."""
        recs = list(self.record_constraints(draft))
        st = self.sidecar.stances.get("manipulation") or self.sidecar.stances.get("init")
        if st is not None and st.edge is not None and self.bindings.has("support"):
            recs.append({"scope": {"scene": self.scene, "support": self.bindings.one("support").scene_object_name}, "type": "stance",
                         "edges": [st.edge.name], "stand_off": [st.stand_off, st.stand_off], "lateral": [st.lateral, st.lateral],
                         "trunk": {"translate": [st.translate, st.translate], "tilt": [st.tilt, st.tilt]}, "hard": False})
        out = [self.gp.add_constraint(r, source_instance=iid, sampler_version=SAMPLER_VERSION) for r in recs]
        self.template_meta["n_constraints_recorded"] = self.template_meta.get("n_constraints_recorded", 0) + len(out)
        return out

    def check_flags(self) -> Dict[str, Any]:
        """Which checks ran in full (development switches are recorded, never silent): ``attached_object`` (the carried
        object attached for lift / carry / place / path / transition checks), ``program_check`` mode."""
        return {"attached_object": NAV.attached_checks_enabled(),
                "program_check": (self.last_program_check or {}).get("mode", "expand" if os.environ.get("ISG_PROGRAM_CHECK", "").lower() != "stub" else "stub")}

    def write_template_meta(self):
        self.template_meta["checks"] = self.check_flags()
        self.template_meta["rejections"] = dict(self.rejection_log.counts)
        self.template_meta["rejection_examples"] = {k: list(v) for k, v in self.rejection_log.examples.items()}
        self.template_meta["counters"] = dict(self.counters)
        self.template_meta["seeds"] = list(self.instance_seeds)
        self.template_meta["kit_stats"] = self.kit.stats_snapshot()
        super().write_template_meta()

    def template_summary(self, n_written: int, time_s: float) -> Dict[str, Any]:
        return template_summary(n_written, time_s, self.rejection_log, self.instance_seeds,
                                {"scene": self.scene, "template_id": self.ctx.template_id, "kit": self.kit.stats_snapshot(), "counters": dict(self.counters),
                                 "checks": self.check_flags()})

    # ------------------------------------------------------------------ main loop

    def existing_instances(self) -> int:
        d = os.path.join(self.ctx.out_dir, "instances")
        if not os.path.isdir(d):
            return 0
        return len([f for f in os.listdir(d) if f.endswith("-tro_state.json")])

    def run_instances(self, n: int, max_attempts: int = 30, start_index: Optional[int] = None) -> List[str]:
        """Per attempt: sample (task module) -> apply -> settle + BEHAVIOR validation -> goal feasibility (robot parked per
        §3.3) -> ``after_goal_check`` -> program check -> TRO + sidecar -> constraints write-back. Rejections are counted
        per reason with the first three examples. Returns the TRO paths written."""
        written: List[str] = []
        start = self.existing_instances() if start_index is None else int(start_index)
        for index in range(start, start + int(n)):
            ok = False
            for attempt in range(max_attempts):
                seed = int(self.ctx.base_seed * 1000003 + index * 1009 + attempt)
                rng = np.random.default_rng(seed)
                t0 = time.time()
                k0 = self.kit.stats_snapshot()
                phase = "reset"
                try:
                    n_reset = self.world.reset_scene_objects([o.name for o in self.objects.values()])
                    if n_reset:
                        # a counter (scene objects put back before the attempt), not a rejection
                        self.counters["scene_reset"] = self.counters.get("scene_reset", 0) + n_reset
                    self.park_robot()
                    self.ikc.solutions.clear()
                    phase = "sample"
                    draft = self.sample(rng)
                    t_sample = time.time() - t0
                    phase = "apply"
                    self.apply_draft(draft)
                    t_apply = time.time() - t0 - t_sample
                    phase = "settle_validate"
                    self.settle_and_validate(draft)
                    # §4.6 step 5 with the objects at their settled (= recorded) poses
                    self.check_stances_with_objects(draft)
                    t_settle = time.time() - t0 - t_sample - t_apply
                    phase = "goal_check"
                    self.check_goal_feasible(draft)
                    self.after_goal_check(draft)
                    phase = "program_check"
                    self.program_check(draft)
                    phase = "write"
                    if self.template_paths is None:
                        self.save_template()
                    draft.summary["wall_time_s"] = time.time() - t0
                    draft.summary["phase_s"] = {"sample": round(t_sample, 2), "apply": round(t_apply, 2), "settle_validate": round(t_settle, 2),
                                                "goal_check_program_check_write": round(time.time() - t0 - t_sample - t_apply - t_settle, 2)}
                    draft.summary["kit"] = self.kit.stats_delta(k0, self.kit.stats_snapshot())
                    draft.summary["attempt"] = attempt
                    path = self.write_instance(draft, index, seed)
                    self.constraints_write_back(draft, self.instance_id(index))
                    self.instance_seeds.append(seed)
                    written.append(path)
                    log.info("%s: instance %d written (attempt %d, %.1fs)", self.gp.alias, index, attempt, time.time() - t0)
                    ok = True
                    break
                except Reject as e:
                    self.rejection_log.add(e.reason, str(e), phase=phase, index=index, attempt=attempt, seed=seed, **_plain_details(e.details))
                    log.debug("rejection traceback", exc_info=True)
                    self.rejections[e.reason] = self.rejections.get(e.reason, 0) + 1
                    log.info("%s: instance %d attempt %d rejected in %s after %.1fs: %s", self.gp.alias, index, attempt, phase, time.time() - t0, e)
                    self.on_reject(e)
            if not ok:
                log.error("%s: instance %d failed after %d attempts; rejections=%s", self.gp.alias, index, max_attempts, self.rejection_log.counts)
        self.template_meta["n_written"] = self.template_meta.get("n_written", 0) + len(written)
        if self.template_paths is not None:
            self.write_template_meta()
        return written

    def run(self, n_instances: int, max_attempts: int = 30) -> List[str]:  # v1 name
        return self.run_instances(n_instances, max_attempts)

    def on_reject(self, err: Reject):
        super().on_reject(err)
        for arm in M.ARMS:
            try:
                self.kit.backend.detach_object(arm)
            except Exception:
                pass
        self.kit.sync_world()


def _plain_details(d: Dict[str, Any]) -> Dict[str, Any]:
    reserved = {"phase", "index", "attempt", "seed", "message"}
    return {(f"detail_{k}" if k in reserved else k): (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)) for k, v in (d or {}).items()}


def _sha_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
