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
import contextlib
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
from ispatialgym.sampling import predicates as PRED  # noqa: E402
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


# ----------------------------------------------------------------------------------------------------------------
# A/B knob ``generation.contact_height_extra_m`` (owner 2026-09-16 23:00): a constant lift of every recorded contact /
# grasp height above today's pose (default 0.0 = today's behaviour)
# ----------------------------------------------------------------------------------------------------------------

#: the pad centre of a lifted grasp stays this far below the object's pinch band top; a lifted blade's fingertip stays this
#: far below the face top
PAD_BAND_MARGIN = 0.005
FACE_TOP_MARGIN = 0.010
PAD_HALF_HEIGHT = 0.012


def requested_height_extra(sampler_or_gp) -> float:
    """``generation.contact_height_extra_m`` of the alias (0.0 when absent)."""
    gen = getattr(sampler_or_gp, "gen", None)
    try:
        return max(0.0, float(gen("contact_height_extra_m", 0.0) or 0.0)) if gen is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def contact_height_extra(sampler_or_gp, kind: str, *, fingertip_z: Optional[float] = None, support_z: Optional[float] = None,
                         band_top_z: Optional[float] = None, pad_center_z: Optional[float] = None) -> Tuple[float, str]:
    """``(effective lift m, clip reason)`` of the A/B knob for one record. ``kind``:

    * ``grasp`` — the offset applies to the **pads**, not the tips: raise only until the fingertip is >= x above the support
      (``fingertip_z`` / ``support_z``; a fingertip already that clear needs no lift), never above the object's pinch band
      (``pad_center_z`` + lift <= ``band_top_z`` - ``PAD_BAND_MARGIN``) — thin lying objects (closure probe: +5 mm lost the
      marker's AG attach, +10 mm the pads missed);
    * ``blade`` / ``stop`` — a top-down blade beside a face: the fingertip stays >= ``FACE_TOP_MARGIN`` below ``band_top_z``;
    * ``push`` / ``brace`` — pads on a face: the pad centre + ``PAD_HALF_HEIGHT`` stays below ``band_top_z``;
    * ``release`` (and anything else) — the full offset.
    Reasons: ``"off"`` (knob 0), ``"full"``, ``"fingertip already clear"``, ``"clipped: pinch band"``, ``"clipped: face top"``.
    """
    x = requested_height_extra(sampler_or_gp)
    if x <= 0.0:
        return 0.0, "off"
    eff, why = x, "full"
    if kind == "grasp":
        if fingertip_z is not None and support_z is not None:
            clearance = float(fingertip_z) - float(support_z)
            eff = min(x, max(0.0, x - clearance))
            if eff <= 0.0:
                return 0.0, "fingertip already clear"
            if eff < x:
                why = "partial: fingertip clearance"
        if band_top_z is not None and pad_center_z is not None:
            room = max(0.0, float(band_top_z) - PAD_BAND_MARGIN - float(pad_center_z))
            if room < eff:
                eff, why = room, "clipped: pinch band"
    elif kind in ("blade", "stop"):
        if fingertip_z is not None and band_top_z is not None:
            room = max(0.0, float(band_top_z) - FACE_TOP_MARGIN - float(fingertip_z))
            if room < eff:
                eff, why = room, "clipped: face top"
    elif kind in ("push", "brace"):
        if pad_center_z is not None and band_top_z is not None:
            room = max(0.0, float(band_top_z) - PAD_HALF_HEIGHT - float(pad_center_z))
            if room < eff:
                eff, why = room, "clipped: face top"
    return float(eff), why


def lift_contact_height(sampler, z: float, kind: str, *, face_top_z: Optional[float] = None, fingertip_below: float = C.FINGERTIP_LENGTH) -> float:
    """One-line form for the task modules: the eef-origin height ``z`` of a top-down push / blade / stop pose lifted by the
    knob (fingertip = ``z - fingertip_below``, kept below ``face_top_z``); the verdict is kept on ``sampler.height_extra_last``
    for ``annotate_height_extra``."""
    eff, why = contact_height_extra(sampler, "blade" if kind in ("blade", "stop", "push") else kind, fingertip_z=float(z) - float(fingertip_below),
                                    band_top_z=face_top_z)
    sampler.height_extra_last = {"height_extra_m": requested_height_extra(sampler), "height_extra_effective_m": round(eff, 4), "height_extra_clip_reason": why}
    return float(z) + eff


def annotate_height_extra(record, sampler, effective: Optional[float] = None, reason: Optional[str] = None):
    """Write the knob's facts (``height_extra_m`` requested, ``height_extra_effective_m``, ``height_extra_clip_reason``) on a
    grasp / contact record's ``extras`` (from ``sampler.height_extra_last`` when not given); returns the record."""
    facts = dict(getattr(sampler, "height_extra_last", None) or {"height_extra_m": requested_height_extra(sampler), "height_extra_effective_m": 0.0,
                                                                  "height_extra_clip_reason": "off" if requested_height_extra(sampler) <= 0 else "not applied"})
    if effective is not None:
        facts["height_extra_effective_m"] = round(float(effective), 4)
    if reason is not None:
        facts["height_extra_clip_reason"] = reason
    extras = getattr(record, "extras", None)
    if isinstance(extras, dict):
        extras.update(facts)
    return record


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
        margin = self.gp.gen("joint_limit_margin_rad", None)  # None = IKChecker.JOINT_LIMIT_MARGIN (0.05)
        reach = self.gp.gen("witness_residual_at_draw_m", None)  # (a) reach quality at the draw: last witness within this of its pose
        self.ikc = IKChecker(self.kit, {"scene": self.world.scene_model, "rooms": sorted(self.world.loaded_rooms or []), "floor_z": self.kit.floor_z},
                             default_working_arms=default_arms, joint_limit_margin=None if margin is None else float(margin),
                             witness_residual_at_draw=None if reach is None else float(reach))
        self.rejection_log = RejectionLog(keep=3)
        self.counters: Dict[str, int] = {}  # scene_reset, attempts, ... (not rejections)
        self.instance_seeds: List[int] = []
        self.template_facts: Dict[str, Any] = {}  # cached geometry of sample_template (written to template_meta)
        self.height_extra_last: Optional[Dict[str, Any]] = None  # the A/B knob's last verdict (``lift_contact_height`` / ``annotate_height_extra``)
        self.template_meta.update({"alias": self.gp.alias, "subtask": self.subtask, "scene": self.scene, "sampler_version": SAMPLER_VERSION,
                                   "spec_hash": self.gp.spec_hash(), "choice": self.choice.to_json(), "rooms": list(self.gp.rooms(self.scene)),
                                   "customs": self.gp.customs_path, "joint_limit_margin_rad": float(self.ikc.JOINT_LIMIT_MARGIN)})
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

    def turn_recheck(self, draft: Draft, grasp: Any, q_first: Optional[th.Tensor] = None) -> Optional[str]:  # hook
        """``tabletop.held_turns``: the module's checks that depend on the wrist turn of a library pinch, run for ``grasp`` just recorded
        as its other finger assignment (``q_first``: the recorded pre-grasp witness wrist-flipped, a start for the pick). None when they
        pass, else the first failure; ``tabletop.TurnNotOffered`` when the module cannot re-check that held chain. Default: not offered."""
        from ispatialgym.sampling.v2 import tabletop as TT

        raise TT.TurnNotOffered("no turn re-check in this task module")

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
            self.template_facts["contact_height_extra_m"] = requested_height_extra(self)  # A/B knob (owner 2026-09-16 23:00)
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
        # the task room's measured floor (``SimWorld.measure_floor_z``: raycast, cached per room); the kit's copy follows
        self.world.floor_z = self.world.measure_floor_z(xy)
        self.kit.floor_z = self.world.floor_z
        self.world.park_robot(xy, floor_z=self.world.floor_z)
        self.kit.sync_world()
        return xy

    def settle_still_entities(self, draft: Draft) -> List[Any]:
        """Entities held at rest during the settle / trembling / goal windows (hook of ``TaskSampler``). None by default:
        holding the scene support was tried on 2026-09-25 against the disembedding bottles' motion and made no difference
        (the excitation was the trays' own drop onto the table, see ``disembedding.TRAY_RELEASE_GAP_M``), and the demo
        holds nothing."""
        return []

    def check_goal_feasible(self, draft: Draft):
        """§3.5: teleport + settle (120 steps) with the robot parked per §3.3, every binary goal atom true, restore."""
        if not draft.goal_poses and not draft.goal_joints:
            return
        state = self.world.snapshot()
        try:
            self.park_robot()
            self.apply_goal(draft)
            ents = self.goal_entities(draft)
            err = self.world.settle(ents, steps=int(self.gen("goal_settle_steps", self.GOAL_SETTLE_STEPS)), still=self.settle_still_entities(draft))
            if err:
                raise Reject("goal_unstable", err)
            self.check_goal_upright(draft)
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
                # which branch decided each under / ontop atom (robust predicate wrapper, D-007 option A)
                "binary_branch": PRED.decided_by(),
                "binary_branch_log": [e for e in PRED.branches() if e["predicate"] in ("under", "ontop")][-6:],
            }
        finally:
            self.world.restore(state)
            self.world.keep_still(self.objects.values())
            self.robot.keep_still()
            self.kit.sync_world()
            self.configure_targets(draft)

    #: FOV rule (b) (owner 2026-09-16 17:10 item 5, spec 3.9): the carry leg lift end -> pre-place is a straight Cartesian
    #: leg at constant orientation; screened at this many waypoints, and when it fails one carry waypoint is searched among
    #: these raises (m) and lateral offsets (m) of its midpoint, in frame and IK-checked like the 3.9 waypoints
    CARRY_LEG_WAYPOINTS = 5
    CARRY_RAISES = (0.05, 0.10, 0.15, 0.20)
    CARRY_LATERAL = (0.0, 0.05, -0.05, 0.10, -0.10)

    def verify_carry_legs(self, draft: Draft):
        """For every grasp record that places (``grasp`` + ``pre_place`` + ``carry_height``, same orientation at both ends):
        the straight leg from the lift end to the pre-place must be trackable with the object attached (screening IK at
        ``CARRY_LEG_WAYPOINTS`` waypoints, exact ends; supports removed, everything else — partner, scene — in the world).
        When it is not, one carry waypoint (midpoint raised / shifted laterally, inside the manipulation image) whose two
        sub-legs pass is recorded as ``grasps[k].carry = [pose]`` (the oracle carries through it; ``waypoint_policy``
        notes it); none → ``Reject("carry_infeasible")``. ``summary.carry`` records the outcome per object."""
        out: Dict[str, Any] = {}
        if "placement" in self.sidecar.stances:
            draft.summary["carry"] = {"leg": "skipped (the place is made from the placement stance after a base move)"}
            return
        for g in list(self.sidecar.grasps):
            if g.grasp is None or g.pre_place is None or g.carry_height is None:
                continue
            if G.quat_angle(G.t(g.grasp[1]), G.t(g.pre_place[1])) > 0.05:
                out[g.object] = {"leg": "skipped (orientation changes along the carry)"}
                continue
            stance = self.sidecar.stances.get(g.stance) or self.sidecar.stance("manipulation")
            obj = self.objects.get(g.object)
            lift_end = (G.t(g.grasp[0]) + th.tensor([0.0, 0.0, float(g.carry_height)]), G.t(g.grasp[1]))
            pre_place = (G.t(g.pre_place[0]), G.t(g.pre_place[1]))
            supports = self.support_objects()
            cam = self.camera_pose_at(stance)

            def leg_ok(a: G.Pose, b: G.Pose) -> Tuple[bool, str]:
                wps = [tuple(w) for w in C.line_waypoints(a, b, self.CARRY_LEG_WAYPOINTS)]
                with NAV.attached(self.kit, g.arm, obj, eef_in_object=g.eef_in_object):
                    r = self.ikc.waypoints_ok([a] + wps, stance, g.arm, ignore_all=supports, cache=False, approached=[])
                return r.ok, r.reason

            ok, why = leg_ok(lift_end, pre_place)
            if ok:
                out[g.object] = {"leg": "straight", "length_m": round(float(th.linalg.norm(pre_place[0] - lift_end[0])), 3)}
                continue
            mid = 0.5 * (lift_end[0] + pre_place[0])
            d = pre_place[0] - lift_end[0]
            horiz = th.tensor([-float(d[1]), float(d[0]), 0.0])
            horiz = horiz / max(float(th.linalg.norm(horiz)), 1e-6)
            found = None
            tried = 0
            for raise_m in self.CARRY_RAISES:
                for lat in self.CARRY_LATERAL:
                    wp = (mid + th.tensor([0.0, 0.0, raise_m]) + horiz * lat, G.t(g.grasp[1]))
                    tried += 1
                    if not self.ikc.eef_in_frame(wp, cam):
                        continue
                    ok1, _ = leg_ok(lift_end, wp)
                    if not ok1:
                        continue
                    ok2, _ = leg_ok(wp, pre_place)
                    if ok2:
                        found = (wp, raise_m, lat)
                        break
                if found is not None:
                    break
            if found is None:
                raise Reject("carry_infeasible", f"{g.object}: the straight carry leg lift end -> pre-place fails ({why}) and none of {tried} carry waypoints "
                                                 f"(raises {self.CARRY_RAISES}, lateral {self.CARRY_LATERAL}) is in frame with both sub-legs trackable")
            wp, raise_m, lat = found
            g.carry = [wp]
            out[g.object] = {"leg": "carry waypoint", "raise_m": raise_m, "lateral_m": lat, "straight_leg_failure": why}
            self.sidecar.waypoint_policy = (self.sidecar.waypoint_policy + "; " if self.sidecar.waypoint_policy else "") + \
                f"carry: straight lift-end -> pre-place leg infeasible, one carry waypoint recorded (raise {raise_m:.2f} m, lateral {lat:+.2f} m), both sub-legs screened at {self.CARRY_LEG_WAYPOINTS} waypoints"
            log.info("%s: carry waypoint recorded for %s (raise %.2f m, lateral %+.2f m): straight leg %s", self.gp.alias, g.object, raise_m, lat, why)
        if out:
            draft.summary["carry"] = out

    #: up-axis guard (owner item 4, 2026-09-16 10:15): after the goal settle an object's up axis must stay within this
    #: of the up axis of the goal pose it was released at (= its init orientation for IORS)
    GOAL_TIP_MAX_DEG = 15.0

    def check_goal_upright(self, draft: Draft):
        """After the goal-feasibility settle every object with a goal pose must keep its up axis within
        ``GOAL_TIP_MAX_DEG`` of the goal pose's up axis (a cup that fell rim-to-well, a box that toppled off a plate) —
        ``Reject("goal_tipped")``; the measured tilts go to ``summary.goal_tilt_deg`` and the counter ``goal_tipped``."""
        up = th.tensor([0.0, 0.0, 1.0])
        tilts: Dict[str, float] = {}
        max_deg = float(self.gen("goal_tip_max_deg", self.GOAL_TIP_MAX_DEG))
        for name, pose in draft.goal_poses.items():
            obj = self.objects.get(name)
            if obj is None:
                continue
            _, q_now = obj.get_position_orientation()
            u_goal = G.rotate(G.t(pose[1]), up)
            u_now = G.rotate(q_now.float(), up)
            c = float(th.clamp(th.dot(u_goal, u_now), -1.0, 1.0))
            tilts[name] = round(math.degrees(math.acos(c)), 2)
        draft.summary["goal_tilt_deg"] = tilts
        bad = {n: d for n, d in tilts.items() if d > max_deg}
        if bad:
            self.counters["goal_tipped"] = self.counters.get("goal_tipped", 0) + 1
            raise Reject("goal_tipped", f"up axis after the goal settle off by {bad} deg (> {max_deg:.0f})")

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
            # a stance that carries an object is checked with it attached (posed in the hand) and its support ignored; the holding
            # arm at its lift configuration when the task recorded one (``carry_arm_q``: the object rides where the oracle holds it)
            obj, arm, eef_in_object = self.carried_attachment(st.carried)
            arm_q = {f"{arm}_arm": self.carry_arm_q[st.carried]} if (obj is not None and st.carried in self.carry_arm_q) else {}
            with self.object_joints_of(st):  # a stance the task recorded with an articulated object's joint held elsewhere (push-close re-stance)
                with NAV.attached(self.kit, arm if obj is not None else None, obj, eef_in_object=eef_in_object):
                    with self.ikc.ignoring([obj] if obj is not None else []):
                        if not self.ikc.stance_collision_free(st, **arm_q):
                            bad.append(st.name)
        if bad:
            raise Reject("stance_collision_with_objects", f"stances {bad} collide with the objects placed (trunks {[(s.translate, round(s.tilt, 2)) for s in stances if s.name in bad]})")
        # §3.8 with the objects placed: the recorded transitions are swept again — trunk sweep and drive-in with the
        # arriving arms — starting at the *recorded* s_T (never lowered: the recorded path ends at its drive-back pose when
        # ``prompt0.connect`` planned it; a larger s_T adds a drive-back from the path's end, checked at the departing trunk)
        for i, tr in enumerate(list(self.sidecar.transitions)):
            frm, to = self.sidecar.stances.get(tr.frm), self.sidecar.stances.get(tr.to)
            if frm is None or to is None:
                continue
            carried_name = tr.holding or (frm.carried if tr.departure else to.carried)
            obj, arm, eef_in_object = self.carried_attachment(carried_name)
            q_carry = self.carry_arm_q.get(carried_name) if (obj is not None and carried_name) else None
            override = {arm: q_carry} if q_carry is not None else None
            with self.object_joints_of(to), self.ikc.ignoring([obj] if obj is not None else []):
                if tr.departure:
                    # a departure transition is re-swept at its recorded stand-off only (the recorded path starts at that pose)
                    new = NAV.departure_transition(self.kit, frm, to, tr.trunk_to, carried_obj=obj, carried_arm=arm, eef_in_object=eef_in_object,
                                                   min_standoff=tr.stand_off, arm_q_override=override)
                    if new is not None and abs(new.stand_off - tr.stand_off) > 1e-6:
                        new = None
                else:
                    dep = self.departure_of(tr.frm, tr.to)
                    s_to = float(to.stand_off if to.stand_off is not None else 0.0)
                    new = NAV.transition(self.kit, frm, to, carried_obj=obj, carried_arm=arm, eef_in_object=eef_in_object,
                                         min_standoff=tr.stand_off, path_end_standoff=self.path_end_standoff(frm, to), arm_q_override=override,
                                         trunk_from=dep.trunk_to if dep is not None else None,
                                         extra_max=max(NAV.TRANSITION_EXTRA_MAX, float(tr.stand_off) - s_to + NAV.TRANSITION_EXTRA_MAX))
            if new is None:
                joints = {n: [round(float(v), 3) for v in o.get_joint_positions().tolist()] for n, o in self.objects.items() if getattr(o, "n_joints", 0)}
                raise Reject("no_transition", f"{tr.frm} -> {tr.to}{' (departure)' if tr.departure else ''}: no swept-free s_T >= the recorded {tr.stand_off:.2f} "
                                              f"with the objects placed (to: stand-off {float(to.stand_off or 0):.2f} trunk ({to.translate:.2f}, {to.tilt:+.2f}) arms {to.arms}; "
                                              f"from trunk ({frm.translate:.2f}, {frm.tilt:+.2f}); object joints {joints})")
            if abs(new.stand_off - tr.stand_off) > 1e-6:
                log.info("%s -> %s: transition stand-off %.2f -> %.2f with the objects placed", tr.frm, tr.to, tr.stand_off, new.stand_off)
            self.sidecar.transitions[i] = new

    def recheck_carried_travel(self, carried: str, arm_q: Optional[Dict[str, th.Tensor]] = None, stances: bool = True) -> Optional[str]:
        """The recorded sweeps that carry ``carried``, checked again as recorded and read-only with the object posed in the hand per its
        CURRENT grasp record (``tabletop.held_turns``: after ``tabletop.flip_grasp`` the other finger assignment rides there): with
        ``stances`` its carried stances, and its transitions at their recorded stand-off, as :meth:`check_stances_with_objects` sweeps
        them; its paths at their recorded strides as ``NAV.plan_path`` / ``NAV.plan_edge_slide`` swept them (yaw-lag envelope, edge
        slide), at ``generation.carry_path_margin_m``. ``arm_q`` (per arm): the arms a task drove its carried path with (touching: this
        assignment's lift configuration); default as :meth:`record_path` (``carry_arm_q``, else the departing stance's arms). Returns why
        a sweep fails, None when every one passes."""
        from ispatialgym.sampling.v2.ik import stance_arm_vectors

        obj, arm, eef_in_object = self.carried_attachment(carried)
        if obj is None:
            return None
        q_carry = self.carry_arm_q.get(carried)
        for st in (self.sidecar.stances.values() if stances else ()):
            if st.name == "goal" or st.carried != carried:
                continue
            kw = {f"{arm}_arm": q_carry} if q_carry is not None else {}
            with self.object_joints_of(st), NAV.attached(self.kit, arm, obj, eef_in_object=eef_in_object), self.ikc.ignoring([obj]):
                if not self.ikc.stance_collision_free(st, **kw):
                    return f"stance {st.name} collides with {carried} in the hand"
        for tr in self.sidecar.transitions:
            frm, to = self.sidecar.stances.get(tr.frm), self.sidecar.stances.get(tr.to)
            if frm is None or to is None or (tr.holding or (frm.carried if tr.departure else to.carried)) != carried:
                continue
            override = {arm: q_carry} if q_carry is not None else None
            with self.object_joints_of(to), self.ikc.ignoring([obj]):
                if tr.departure:
                    new = NAV.departure_transition(self.kit, frm, to, tr.trunk_to, carried_obj=obj, carried_arm=arm, eef_in_object=eef_in_object,
                                                   min_standoff=tr.stand_off, arm_q_override=override)
                else:
                    dep = self.departure_of(tr.frm, tr.to)
                    s_to = float(to.stand_off if to.stand_off is not None else 0.0)
                    new = NAV.transition(self.kit, frm, to, carried_obj=obj, carried_arm=arm, eef_in_object=eef_in_object, min_standoff=tr.stand_off,
                                         path_end_standoff=self.path_end_standoff(frm, to), arm_q_override=override, trunk_from=dep.trunk_to if dep is not None else None,
                                         extra_max=max(NAV.TRANSITION_EXTRA_MAX, float(tr.stand_off) - s_to + NAV.TRANSITION_EXTRA_MAX))
            if new is None or abs(new.stand_off - tr.stand_off) > 1e-6:
                return f"transition {tr.frm} -> {tr.to} is not swept free at its recorded s_T {tr.stand_off:.2f} with {carried} in the hand"
        margin = float(self.gp.gen("carry_path_margin_m", 0.01))
        for p in self.sidecar.paths:
            frm = self.sidecar.stances.get(p.frm)
            if p.carried != carried or frm is None or len(p.waypoints) < 2:
                continue
            if p.mode == NAV.EDGE_SLIDE_MODE:
                poses = NAV.stride_poses(p.waypoints, p.stride_m)
            else:
                poses = NAV.stride_poses(p.waypoints, p.stride_m, yaw_envelope=True)
            q_arms = arm_q
            if q_arms is None:
                r, l, _ = stance_arm_vectors(self.kit, frm, p.posture if p.posture != "untucked" else None)
                q_arms = {"right": r, "left": l, **({arm: q_carry} if q_carry is not None else {})}
            trunk = (float(p.trunk[0]), float(p.trunk[1])) if p.trunk is not None else (frm.translate, frm.tilt)
            with NAV.attached(self.kit, arm, obj, eef_in_object=eef_in_object):
                ok, bad = NAV.path_collision_free(self.kit, poses, trunk, p.posture, margin=margin, q_arms=q_arms)
            if not ok:
                return f"path {p.frm} -> {p.to} collides at swept pose {bad}/{len(poses)} with {carried} in the hand"
        return None

    @contextlib.contextmanager
    def object_joints_of(self, st: Optional[Stance]):
        """The articulated objects' joints as the task recorded them for a stance (``stance.extras["object_joints"] =
        {instance: {joint: value}}``: a push-close re-stance is made with the flap held part-way by the brake; the settled
        world has it at q_init) - set kinematically for the checks under the block, restored after. No-op without the field."""
        oj = (getattr(st, "extras", None) or {}).get("object_joints") if st is not None else None
        if not oj:
            yield
            return
        saved = []
        try:
            for name, joints in oj.items():
                obj = self.objects.get(name)
                if obj is None or not getattr(obj, "n_joints", 0):
                    continue
                qs = obj.get_joint_positions().clone()
                saved.append((obj, qs.clone()))
                names = list(obj.joints)
                for jn, v in (joints or {}).items():
                    if jn in names:
                        qs[names.index(jn)] = float(v)
                obj.set_joint_positions(qs)
                obj.set_joint_velocities(th.zeros_like(qs))
                obj.keep_still()
            self.kit.sync_world()
            yield
        finally:
            for obj, qs in saved:
                obj.set_joint_positions(qs)
                obj.set_joint_velocities(th.zeros_like(qs))
                obj.keep_still()
            if saved:
                self.kit.sync_world()

    def departure_of(self, frm: str, to: str) -> Optional[Transition]:
        """The recorded departure transition ``frm -> to`` (None when the pair has only the §3.8 arrival one)."""
        return next((t for t in self.sidecar.transitions if t.frm == frm and t.to == to and t.departure), None)

    @property
    def carry_arm_q(self) -> Dict[str, th.Tensor]:
        """bddl instance -> 7 arm-joint values the carrying arm holds while the object is carried (its lift configuration,
        recorded by the task module with :meth:`record_carry_arm_q`); the transition / path checks with that object
        attached use it in place of the stances' recorded arm posture (the oracle never moves a holding arm)."""
        return self.__dict__.setdefault("_carry_arm_q", {})

    def record_carry_arm_q(self, carried: str, arm: str, q_arm: Optional[th.Tensor]) -> None:
        if q_arm is None:
            self.carry_arm_q.pop(carried, None)
            self.carry_end_q.pop(carried, None)
        else:
            self.carry_arm_q[carried] = q_arm.clone()
        self.__dict__.setdefault("_carry_arm_of", {})[carried] = arm

    @property
    def carry_end_q(self) -> Dict[str, th.Tensor]:
        """carried instance -> the arm configuration the oracle's straight carry leg reaches the PRE-PLACE in (the end of the
        differential-IK tracking the task module ran from ``carry_arm_q``); the program check seeds the place witnesses from it."""
        return self.__dict__.setdefault("_carry_end_q", {})

    def path_end_standoff(self, frm: Stance, to: Stance) -> float:
        """Stand-off at which the recorded path ``frm -> to`` ends: the stance's own ``s`` when the path reaches the
        stance, ``s + back`` when it ends at a drive-back pose along -inward (``prompt0.connect``)."""
        s = float(to.stand_off if to.stand_off is not None else 0.0)
        for p in self.sidecar.paths:
            if p.frm == frm.name and p.to == to.name and p.waypoints:
                end = p.waypoints[-1]
                dx, dy = float(to.base_pose2d[0]) - float(end[0]), float(to.base_pose2d[1]) - float(end[1])
                back = dx * float(to.inward[0]) + dy * float(to.inward[1])  # positive: the path ends behind the stance
                return s + max(0.0, back)
        return s

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

    def record_transition(self, frm: Stance, to: Stance, carried: Optional[str] = None, arm: Optional[str] = None,
                          arm_q: Optional[Dict[str, th.Tensor]] = None, trunk_from: Optional[Tuple[float, float]] = None,
                          extra_max: Optional[float] = None) -> Transition:
        """§3.8 (arrival) transition between two recorded stances (carried object attached, posed in the hand per its grasp
        record); ``arm_q`` = arms that keep their configuration (the carrying arm at its lift pose); ``trunk_from`` = the
        trunk the robot arrives with when a departure transition straightened it; ``extra_max`` = how far beyond the
        stand-off the s_T candidates go (§3.8 0.5 m; a carry into a deep crouch may back off further);
        ``Reject("no_transition")``."""
        carried = carried or to.carried
        obj, arm, eef_in_object = self.carried_attachment(carried, arm)
        if arm_q is None and carried and obj is not None and carried in self.carry_arm_q:
            arm_q = {arm: self.carry_arm_q[carried]}
        extra = NAV.TRANSITION_EXTRA_MAX if extra_max is None else float(extra_max)
        # a carried sweep is checked with more clearance than the 1 cm of an empty-handed one: the executed trunk sweep lags its
        # targets and the loaded arm droops (0920-143723: the held sandal brushed the table top for 2 steps of an arrival the
        # sampler had passed at 1 cm)
        margin = float(self.gp.gen("carry_collision_margin_m", 0.04)) if (carried and obj is not None) else 0.01
        tr = NAV.transition(self.kit, frm, to, carried_obj=obj, carried_arm=arm, eef_in_object=eef_in_object, arm_q_override=arm_q, trunk_from=trunk_from,
                            extra_max=extra, blame=self.ikc._collision_blame, margin=margin)
        if tr is None:
            raise Reject("no_transition", f"{frm.name} -> {to.name}: no s_T in {NAV.transition_standoffs(to.stand_off or 0.0, extra_max=extra)} is swept collision-free")
        tr.holding = carried or tr.holding  # ``transitions[].holding``: the bddl instance carried through (attachment's second base pose)
        return self.sidecar.add_transition(tr)

    def record_departure_transition(self, frm: Stance, to: Stance, travel_trunk: Tuple[float, float], carried: Optional[str] = None,
                                    arm: Optional[str] = None, arm_q: Optional[Dict[str, th.Tensor]] = None, min_standoff: Optional[float] = None) -> Transition:
        """Owner 2026‑09‑19 21:56: the departure transition at ``frm`` (drive back along its own -forward to s_T, sweep
        the trunk to ``travel_trunk``, no drive-in); the path ``frm -> to`` recorded after it must start at
        ``NAV.departure_pose(frm, tr.stand_off)`` and be driven at ``travel_trunk`` (:meth:`record_path` ``start_pose2d``
        / ``trunk``). ``Reject("no_transition")``."""
        carried = carried or frm.carried
        obj, arm, eef_in_object = self.carried_attachment(carried, arm)
        if arm_q is None and carried and obj is not None and carried in self.carry_arm_q:
            arm_q = {arm: self.carry_arm_q[carried]}
        margin = float(self.gp.gen("carry_collision_margin_m", 0.04)) if (carried and obj is not None) else 0.01
        tr = NAV.departure_transition(self.kit, frm, to, travel_trunk, carried_obj=obj, carried_arm=arm, eef_in_object=eef_in_object, arm_q_override=arm_q, margin=margin,
                                      min_standoff=min_standoff)
        if tr is None:
            raise Reject("no_transition", f"{frm.name} -> {to.name} (departure): no s_T in {NAV.transition_standoffs(frm.stand_off or 0.0)}"
                                          + (f" >= {float(min_standoff):.2f}" if min_standoff is not None else "") + " admits the drive-back + "
                                          f"trunk sweep {(frm.translate, round(frm.tilt, 2))} -> {tuple(round(float(v), 2) for v in travel_trunk)}")
        tr.holding = carried or tr.holding
        return self.sidecar.add_transition(tr)

    def record_path(self, frm: Stance, to: Stance, carried: Optional[str] = None, arm: Optional[str] = None, max_length_m: Optional[float] = None,
                    blocked: Sequence[Any] = (), force_global: bool = False, trunk: Optional[Tuple[float, float]] = None,
                    arm_q: Optional[Dict[str, th.Tensor]] = None, start_pose2d: Optional[Tuple[float, float, float]] = None,
                    edge_slide: bool = False, edge_slide_back_m: float = 0.0) -> PathRecord:
        """§3.7 path between two recorded stances, checked at the departing stance's trunk (``trunk`` overrides) with the
        carried object attached (posed in the hand per its grasp record); ``arm_q`` (per arm) fixes the arm posture of the
        drive (a carry: the lift configuration the oracle drives with; default for a carried object: its recorded
        ``carry_arm_q``); ``start_pose2d`` = where the path starts when it is not ``frm``'s base pose (the drive-back pose
        of a departure transition); ``force_global`` plans on the eroded map end to end; ``edge_slide`` (owner 2026-09-24)
        tries the same-edge slide (``NAV.plan_edge_slide``: body facing the edge, no turns, ``edge_slide_back_m`` extra
        stand-off for the slide) before the grid planner; ``Reject(no_path | path_collision | path_too_long)``."""
        carried = carried or to.carried or frm.carried
        obj, arm, eef_in_object = self.carried_attachment(carried, arm)
        if arm_q is None and carried and obj is not None and carried in self.carry_arm_q:
            from ispatialgym.sampling.v2.ik import stance_arm_vectors

            r, l, _ = stance_arm_vectors(self.kit, frm)
            arm_q = {"right": r, "left": l, arm: self.carry_arm_q[carried]}
        start = frm
        if start_pose2d is not None:
            start = Stance(frm.name, tuple(float(v) for v in start_pose2d), frm.translate, frm.tilt, frm.stand_off, frm.edge, frm.face, frm.lateral, frm.focus,
                           frm.arms, list(frm.contacts), frm.q, frm.carried, dict(frm.extras), frm.fingers, frm.idle_posture)
        # ``paths[].posture`` = the posture the drive was checked at: a tucked init stance (the floor class of IORS large-scale,
        # ``1dd3820``) drives tucked - the oracle re-poses empty hands to ``untucked`` only when the record says so
        posture = "tucked" if (frm.posture == "tucked" and arm_q is None) else "untucked"
        try:
            # a CARRIED drive may ask for more clearance than the 1 cm of the whole-body stride checks (generation.carry_path_margin_m;
            # 0921-040940: the idle arm brushed a wardrobe on a 0.54 m carry the 1 cm check had passed) - a local path that fails it
            # falls back to the eroded global map in the callers' path_collision retry
            margin = float(self.gp.gen("carry_path_margin_m", 0.01)) if (carried and obj is not None) else 0.01
            # owner 2026-09-26: ``generation.base_path_style`` (default "recorded") - a family that opts in has its UNCARRIED drives checked
            # and recorded as the face_travel schedule (recorded fallback when the schedule collides); carried legs stay recorded, the
            # schedule only probed at 1 cm for the morning report
            is_carried = bool(carried and obj is not None)
            style = "recorded" if is_carried else str(self.gp.gen("base_path_style", "recorded") or "recorded")
            p = NAV.plan_path(self.kit, start, to, posture=posture, carried=carried, carried_obj=obj, carried_arm=arm, rooms=self.gp.rooms(self.scene) or self.world.rooms,
                              blocked=blocked, max_length_m=max_length_m, trunk=trunk, eef_in_object=eef_in_object, force_global=force_global, arm_q=arm_q, margin=margin,
                              edge_slide=edge_slide, edge_slide_back_m=edge_slide_back_m, style=style, probe_face_travel=is_carried)
        except NAV.NavigationError as e:
            raise Reject(e.reason, str(e))
        return self.sidecar.add_path(p)

    def camera_pose_at(self, stance: Stance) -> G.Pose:
        """Head-camera pose at a stance through the **shared** camera FK (``conventions.camera_pose``: torso chain from the
        stance's trunk joints, base at the stance pose on the measured floor, the kit's measured sensor offset in
        ``zed_link``) — the model the oracle's gripper-in-FOV monitor projects with (owner 2026-09-16 10:15 item 1b), so a
        pre-pose that passes here is not flagged at execution."""
        q = self.stance_q(stance)
        x, y, yaw = (float(v) for v in stance.base_pose2d)
        base = (th.tensor([x, y, float(self.kit.floor_z) + C.NOMINAL_BASE_Z]), G.quat_from_yaw(yaw))
        return C.camera_pose(q, base_pose=base, cam_in_zed=self.kit.cam_in_zed)

    # ------------------------------------------------------------------ program check (§3.5)

    def program_check(self, draft: Draft) -> Dict[str, Any]:
        from ispatialgym.sampling.v2 import tabletop as TT

        report = PC.run(self.gp.alias, self.sidecar, self, subtask=self.subtask)
        # the wrist turn of a library pinch ranked on its whole held chain (owner 2026-09-27; ranking only, never a rejection)
        report = TT.held_turns(self, draft, report, lambda: PC.run(self.gp.alias, self.sidecar, self, subtask=self.subtask))
        # the arm of the segment ranked on its whole held chain (owner 2026-09-27, first-demo v2.2; ranking only, never a rejection)
        report = TT.held_arms(self, draft, report, lambda: PC.run(self.gp.alias, self.sidecar, self, subtask=self.subtask))
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
        # the TRO records a state at rest: the physics steps of the checks after the settle (``attached`` teleports, program
        # check) leave residual contact-chatter velocities on the objects the checks did not hold (2026-09-24: 8-50 mm/s on
        # the disembedding distractor and source tray while the target read 0)
        self.world.keep_still(self.objects[inst] for inst in self.physical)
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

    def final_configuration(self) -> Optional[Tuple[th.Tensor, Tuple[float, float, float], Dict[str, Any]]]:
        """Where the program check leaves the robot: ``(q, base_pose2d, source)`` - the last action's last witness (its
        retreat / end pose) at that action's stance, or that stance's own posture when the last action records no
        witness (a closed-loop push-close). None without a checked program."""
        acts = (self.last_program_check or {}).get("actions") or []
        if not acts:
            return None
        last = acts[-1]
        stance = self.sidecar.stances.get(last.get("stance") or "")
        if stance is None:
            return None
        wit = last.get("witnesses") or {}
        key = "retreat" if "retreat" in wit else (list(wit)[-1] if wit else None)
        if key is not None and wit[key].get("q") is not None:
            q = th.tensor(wit[key]["q"], dtype=th.float32)
            source = {"action": last.get("name"), "index": last.get("index"), "witness": key, "stance": last.get("stance")}
        else:
            q = self.stance_q(stance)
            source = {"action": last.get("name"), "index": last.get("index"), "witness": None, "stance": last.get("stance")}
        return q, tuple(float(v) for v in stance.base_pose2d), source

    def check_goal_occlusion(self, draft: Draft, tro_path: str) -> None:
        """Pre-generation goal-image occlusion (``ispatialgym.eval.goal_occlusion``): the goal state as the goal check
        settled it (its settled poses, the goal joints) with the robot where the program check ends
        (:meth:`final_configuration`), screened as the oracle screens a demo's last frame - the head camera's depth, the box
        screen per target / partner, the panel screen per fixture (a second depth render with the robot hidden). It runs
        after the instance is written, so the TRO is the one an unchecked run writes; a target / partner at least half
        hidden removes the instance's files and rejects the attempt (``goal_occluded``) before any demo is recorded."""
        from omnigibson.utils.constants import JointType

        from ispatialgym.eval import goal_occlusion as GO

        if not GO.enabled() or GO.exempt(self.subtask):
            return
        judged = [o for o in dict.fromkeys(self.objects.values()) if GO.role(o.name)]
        final = self.final_configuration()
        if not judged or final is None or self.world.cam is None:
            draft.summary["goal_occlusion"] = {"skipped": "no judged object" if not judged else ("no checked program" if final is None else "no camera")}
            return
        q, base_pose2d, source = final
        settled = (draft.summary.get("goal_check") or {}).get("settled_poses") or {}
        t0 = time.time()
        state = self.world.snapshot()
        try:
            for name, pose in draft.goal_poses.items():
                obj = self.objects[name]
                p = settled.get(obj.name, pose)
                obj.set_position_orientation(G.t(p[0]), G.t(p[1]))
            for name, qj in draft.goal_joints.items():
                obj = self.objects[name]
                obj.set_joint_positions(G.t(qj))
                obj.set_joint_velocities(th.zeros_like(G.t(qj)))
            self.kit.apply(q, base_pose2d)
            # a body teleported while asleep keeps its old render transform until it is awake in a rendering step
            # (SimWorld.ensure_segmented): wake, one rendering step, everything held still
            task_objects = list(dict.fromkeys(self.objects.values()))
            self.world.wake(task_objects)
            self.robot.keep_still()
            self.world.full_step(render=True)
            self.world.keep_still(task_objects)
            self.robot.keep_still()
            intr = GO.intrinsics_of(self.world.camera_model)
            depth = self.world.depth_linear().float().numpy()
            cp, cq = self.world.camera_pose()
            depth_without = None
            if any(GO.role(o.name) == "fixture" for o in judged):
                self.robot.visible = False
                try:
                    depth_without = self.world.depth_linear().float().numpy()
                finally:
                    self.robot.visible = True
            objects = GO.screen_scene(depth, (cp.tolist(), cq.tolist()), intr, judged, depth_without_robot=depth_without, revolute=JointType.JOINT_REVOLUTE)
            rep = GO.report(objects, method="predicted: settled goal poses, robot at the program check's end, head-camera depth", robot=source,
                            seconds=round(time.time() - t0, 2))
        except Exception as e:  # noqa: BLE001 - a failed prediction keeps the instance; the oracle's gate still judges its last frame
            log.warning("%s: goal occlusion not predicted: %r", self.gp.alias, e)
            rep = {"error": repr(e), "occluded": False, "robot": source}
        finally:
            self.world.restore(state)
            self.world.keep_still(self.objects.values())
            self.robot.keep_still()
            self.kit.sync_world()
            self.configure_targets(draft)
        draft.summary["goal_occlusion"] = rep
        summary_path = tro_path.replace("-tro_state.json", ".summary.json")
        sidecar_path = tro_path.replace("-tro_state.json", ".oracle.json")
        if rep["occluded"]:
            for p in (tro_path, sidecar_path, summary_path):
                if os.path.exists(p):
                    os.remove(p)
            if not self.instance_seeds:
                self.template_meta.pop("stances", None)
            self.counters["goal_occluded"] = self.counters.get("goal_occluded", 0) + 1
            raise Reject("goal_occluded", f"{rep['worst']} predicted {rep['max_occluded']:.0%} hidden in the goal image (>= {rep['threshold']:.0%})",
                         objects={k: v.get("occluded") for k, v in rep["objects"].items()})
        with open(summary_path) as f:
            summary = json.load(f)
        summary["goal_occlusion"] = _jsonable(rep)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def annotate_stance_arms(self) -> None:
        """A6: every recorded stance carries the arm joint values the checks used (``arms_q = {"right": [7], "left": [7]}``,
        ``fingers`` when not the open default) next to ``idle_posture`` in the sidecar, so the oracle can move the idle
        arm there before the trunk motion (unknown keys land in the F6 ``Sidecar.extra``)."""
        for st in self.sidecar.stances.values():
            r, l, f = self.ikc.arm_vectors(st)
            st.extras["arms_q"] = {"right": [round(float(v), 5) for v in r.tolist()], "left": [round(float(v), 5) for v in l.tolist()]}
            if abs(f - float(C.FINGER_OPEN)) > 1e-6:
                st.extras["fingers"] = f
            if getattr(st, "q", None) is not None:
                # the opening per arm as the recorded configuration has it (owner 2026-09-22 11:27: a re-stance recorded with the fist
                # closed must not close the idle hand too)
                robot = self.kit.robot
                working = set(getattr(st, "working_arms", []) or [])
                st.extras["fingers_by_arm"] = {a: (round(float(st.q[robot.gripper_control_idx[a]][0]), 4) if (not working or a in working) else None) for a in ("right", "left")}

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
                                 "checks": self.check_flags(),
                                 "floor": {"floor_z": round(float(self.world.floor_z), 4), "by_room": {k: round(v, 4) for k, v in getattr(self.world, "floor_z_by_room", {}).items()},
                                           "probe": dict(getattr(self.world, "floor_probe", {}) or {})}})

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
                    self.ikc.attempt = attempt  # ``IKChecker.LIMIT_RETRY_RANDOM_FROM_ATTEMPT``: attempt 0 solves without the random re-solves
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
                    self.verify_carry_legs(draft)
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
                    phase = "goal_occlusion"
                    self.check_goal_occlusion(draft, path)
                    self.constraints_write_back(draft, self.instance_id(index))
                    self.instance_seeds.append(seed)
                    written.append(path)
                    # run_isg loads an instance through the template meta; demos replay instances while the rest of the
                    # template is still being sampled
                    self.write_template_meta()
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
