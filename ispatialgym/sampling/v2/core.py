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
pose), ``v1_role_variants()`` (facade BDDL rendering until D1).
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

from ispatialgym.fastflags import fast_on  # noqa: E402  (train-fast switches, 2026-10-31: ISPATIALGYM_FAST_<NAME>, default on, =0 off)
from ispatialgym.sampling import instance_separation as SEP  # noqa: E402  (pure python: the early repeat filter's record)

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
from ispatialgym.sampling.v2 import neighbors as NB  # noqa: E402
from ispatialgym.sampling.v2 import program_check as PC  # noqa: E402
from ispatialgym.sampling.v2.ik import IKChecker  # noqa: E402
from ispatialgym.sampling.v2.problem import GroundedProblem, RejectionLog, TemplateChoice, template_summary  # noqa: E402
from ispatialgym.sampling.v2.stances import PathRecord, SidecarDraft, Stance, Transition  # noqa: E402

log = logging.getLogger("ispatialgym.sampling.v2")


def start_per_attempt() -> bool:
    """train85 (owner 2026-10-14 ~20:45): ``ISPATIALGYM_START_PER_ATTEMPT=1``, set only by the train jobs, makes every attempt draw
    its own start (:meth:`SamplerV2.spa_redraw_start`) instead of the start ``sample_template`` drew once per template from the
    template seed; each instance's start is in its own TRO / sidecar. Unset: nothing changes."""
    return os.environ.get("ISPATIALGYM_START_PER_ATTEMPT") == "1"


#: i7-attachdef 2026-11-08 (lead decision: a sampler-only bug fix, default OFF until the owner decides): the collector's queue
#: (``run_isg.queue_ready``) takes an instance as ready once its TRO, its summary and the template meta exist, and the summary was written
#: BEFORE the post-write screens (``check_instance_filter``, ``check_goal_occlusion``) - a collector polling in that window claimed an attempt
#: the screen then rejected (campaign: disembedding_04 / 09 ran a goal-occluded layout and wrote its goal override into the TRO the sampler
#: re-sampled over it; 2 mental_rotation episodes hit FileNotFoundError). With ``ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS=1`` the summary is
#: written as ``<iid>.summary.json.pending`` and renamed (atomic) to ``<iid>.summary.json`` only after every post-write screen passed; a
#: rejected attempt's files are removed with it, so it is never claimable. Counters (SAMPLING_REPORT) ``publish_after_screens.{published,withheld}``.
SUMMARY_SUFFIX = ".summary.json"
PENDING_SUFFIX = ".pending"


def publish_after_screens() -> bool:
    return fast_on("PUBLISH_AFTER_SCREENS", default=False)


def instance_summary_path(tro_path: str, pending: Optional[bool] = None) -> str:
    """The summary file of the instance whose TRO is ``tro_path``: the pending name while the switch holds it back."""
    base = tro_path[: -len("-tro_state.json")] + SUMMARY_SUFFIX if tro_path.endswith("-tro_state.json") else tro_path + SUMMARY_SUFFIX
    return base + PENDING_SUFFIX if (publish_after_screens() if pending is None else pending) else base


def remove_instance_files(tro_path: str) -> List[str]:
    """A rejected attempt's files: the TRO, the sidecar, the summary (published or pending). Returns the removed paths."""
    out = []
    for p in (tro_path, tro_path.replace("-tro_state.json", ".oracle.json"), instance_summary_path(tro_path, False), instance_summary_path(tro_path, True)):
        if os.path.exists(p):
            os.remove(p)
            out.append(p)
    return out


def publish_instance(tro_path: str) -> bool:
    """After the post-write screens: the pending summary -> the summary the queue reads (``os.replace``: atomic on one filesystem).
    False when there was nothing pending (the switch off: the summary was written in place)."""
    pend = instance_summary_path(tro_path, True)
    if not os.path.exists(pend):
        return False
    os.replace(pend, instance_summary_path(tro_path, False))
    return True


class TemplateExhausted(Reject):
    """A rejection after which no attempt of the template can succeed (IORS small once every start-region candidate failed its
    init -> manipulation travel deterministically): ``run_instances`` counts it like any rejection and ends the template's attempts
    instead of spending the time budget on them."""


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
    """``generation.contact_height_extra_m`` of the variant (0.0 when absent)."""
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
    out_dir: str  # <out_root>/<variant>/<template_id>
    base_seed: int = 0
    customs_sha256: Optional[str] = None

    @property
    def scene(self) -> str:
        return self.choice.scene

    @property
    def template_id(self) -> str:
        return self.choice.template_id

    @property
    def variant_dir(self) -> str:
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
    #: ``(tro_path) -> None | why`` run on every instance just written (``multiply_isg --split train``: the train-repeat rule against
    #: the eval instances, ``instance_separation.repeat_filter``); a why removes the instance and rejects the attempt (``repeats_eval``)
    instance_filter: Optional[Callable[[str], Optional[str]]] = None

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
        self.sidecar = SidecarDraft(self.gp.variant, self.gp.family)
        arm = self.gp.gen("arm", "right")
        default_arms = ["right"] if arm in (None, "auto", "right") else (["left"] if arm == "left" else ["right", "left"])
        margin = self.gp.gen("joint_limit_margin_rad", None)  # None = IKChecker.JOINT_LIMIT_MARGIN (0.05)
        reach = self.gp.gen("witness_residual_at_draw_m", None)  # (a) reach quality at the draw: last witness within this of its pose
        ik_scope = {"scene": self.world.scene_model, "rooms": sorted(self.world.loaded_rooms or []), "floor_z": self.kit.floor_z}
        crop = self._world_scene_crop()
        if crop.get("mode", "full") != "full":
            # train-fast SCENE_CROP (2026-11-01): a cropped world is another collision world - its own cache namespace (full keeps today's)
            ik_scope["scene_crop"] = f"{crop['mode']}:{crop.get('center_object')}:{crop.get('radius_m')}"
            # IK_CACHE_SHARE_CROP (2026-11-01): the rooms the full world of this scene loads, which the exact-IK key uses instead of the
            # crop's (verified hits: the cropped and the full world share them; ik.IKChecker.exact_cache_scope)
            ik_scope["full_rooms"] = sorted(getattr(self.world, "eval_rooms", None) or self.world.loaded_rooms or [])
        self.ikc = IKChecker(self.kit, ik_scope,
                             default_working_arms=default_arms, joint_limit_margin=None if margin is None else float(margin),
                             witness_residual_at_draw=None if reach is None else float(reach))
        self.rejection_log = RejectionLog(keep=3)
        self.counters: Dict[str, int] = {}  # scene_reset, attempts, ... (not rejections)
        self.instance_seeds: List[int] = []
        self.template_facts: Dict[str, Any] = {}  # cached geometry of sample_template (written to template_meta)
        self.height_extra_last: Optional[Dict[str, Any]] = None  # the A/B knob's last verdict (``lift_contact_height`` / ``annotate_height_extra``)
        self.template_meta.update({"variant": self.gp.variant, "subtask": self.subtask, "scene": self.scene, "sampler_version": SAMPLER_VERSION,
                                   "spec_hash": self.gp.spec_hash(), "choice": self.choice.to_json(), "rooms": list(self.gp.rooms(self.scene)),
                                   "customs": self.gp.customs_path, "joint_limit_margin_rad": float(self.ikc.JOINT_LIMIT_MARGIN)})
        self.last_program_check: Optional[Dict[str, Any]] = None
        self._park_xy: Optional[th.Tensor] = None
        # train-fast (2026-10-31): PARK_ONCE's (key, xy, floor_z) of the last computed park; the one-line "first time taken" log keys
        self._park_cache: Optional[Tuple[Tuple, th.Tensor, float]] = None
        self._fast_logged: set = set()

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

    def spa_redraw_start(self, rng: np.random.Generator) -> None:  # hook (start_per_attempt)
        """This attempt's own start drawn from the attempt's ``rng`` (:func:`start_per_attempt`): the task modules whose
        ``sample_template`` fixes the start for the template override it; the others draw the start per attempt already."""

    def spa_template_start(self, errors: List[str], draws: int = 8) -> None:
        """start_per_attempt: the template's own start draw found no start with the template seed's draw; up to ``draws`` more draws
        from derived streams (every attempt draws its own start anyway: the template's start only shows that one exists, and one
        unlucky draw would otherwise reject the template for every train job of the variant). Leaves the start unset when every draw
        fails (the caller's rejection follows)."""
        for k in range(1, int(draws) + 1):
            try:
                self.spa_redraw_start(np.random.default_rng([int(self.ctx.base_seed), k]))
                log.info("%s: start_per_attempt: template start found on setup draw %d: %s", self.gp.variant, k, getattr(self, "_spa_facts", None))
                return
            except Reject as e:
                errors.append(f"start_per_attempt setup draw {k}: [{e.reason}] {str(e)[:200]}")

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
        """Objects ignored for the final grasp / contact pose only (§3.6): the bound support(s) and the neighbouring supports the
        template's top is united with (owner 2026-10-12 A4)."""
        out = []
        for role in ("support", "tray_source", "tray_dest", "wall"):
            if self.bindings.has(role):
                out += [self.objects[b.name] for b in self.bindings.many(role)]
        return out + NB.neighbor_objects(self)

    def _evaluate(self, predicate_cls, *names):
        """``TaskSampler._evaluate`` with a support united with its neighbours standing for any of its tops in ``ontop``
        (``neighbors.evaluate_on_tops``; identical without neighbours)."""
        return NB.evaluate_on_tops(self, predicate_cls, [self.scope[n] for n in names], PRED.evaluate_predicate)

    def v1_role_variants(self) -> Dict[str, str]:  # hook (facade only)
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
        from ispatialgym.sampling.v2 import tabletop as TT

        self.sidecar.reset_instance()
        if start_per_attempt():
            full = getattr(self, "_split_cone_cands", None)
            # oracle-fix 2026-11-06 (zero-yield survey, ontop_small_04 / inside_04): a task that drew this attempt's start before calling here
            # (``_spa_drawn``: iors_small.sample, whose spa_redraw_start rebuilds manip_candidates for the new start) keeps that list - the
            # restore below put the PREVIOUS start's candidates back over it, so every attempt after a process's first drove from its own
            # init to the first start's manipulation stances (path_too_long 3.1-4.3 m, path_collision, partner frame 0/82, an empty
            # candidate list: 92 % target_placement on ontop_small_04, 2 written in 615 attempts)
            if full is not None and fast_on("SPLIT_CONE_RESTORE") and not getattr(self, "_spa_drawn", False):
                # train-fast 2026-11-01 (SPLIT_CONE_RESTORE): split_cone_draw narrowed manip_candidates to the last attempt's position; the
                # reset below rebuilds the pools from manip_candidates, so without this every attempt after the first kept that one
                # position (only iors_small's spa_redraw_start rebuilds the list): the template's full list goes back first
                self.manip_candidates = list(full)
            if not getattr(self, "_spa_drawn", False):
                t_spa = time.time()
                self.spa_redraw_start(rng)
                self._spa_draw_s = time.time() - t_spa
            self._spa_drawn = False
            self._split_cone_cands = self._split_cone_pools = None
            log.info("%s: start_per_attempt: start drawn in %.1fs: %s", self.gp.variant, getattr(self, "_spa_draw_s", 0.0), getattr(self, "_spa_facts", None))
        TT.split_cone_draw(self, rng)
        draft = self.sample_instance(rng)
        if start_per_attempt():
            draft.summary["start_per_attempt"] = {**(getattr(self, "_spa_facts", None) or {}), "draw_s": round(float(getattr(self, "_spa_draw_s", 0.0)), 2)}
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
            raise Reject("spec", f"{self.gp.variant}: no room / floor binding in scene {self.scene}")
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
        """The grounded BDDL of the variant with the bindings mapped onto its instances; when the variant has none yet (v1
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
        variants = {**self.v1_role_variants()}
        for role, lst in self.bindings.roles.items():
            v1.roles[V1_ROLE_MAP.get(role, role)] = list(lst)
        for v1_role, v2_role in variants.items():
            if v2_role in self.bindings.roles:
                v1.roles[v1_role] = list(self.bindings.roles[v2_role])
        return build_problem(self.activity, self.definition_id, v1)

    # ------------------------------------------------------------------ template phase

    def setup(self):
        t0 = time.time()
        k0 = self.kit.stats_snapshot()
        n_reset = self.world.reset_scene_objects(context=f"{self.gp.variant}: before the template setup")
        if n_reset:
            log.info("%s: %d scene objects reset before the template setup", self.gp.variant, n_reset)
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
        # train-fast SCENE_CROP (2026-11-01): the adopted template-partial_rooms.json is the dump of the world that saved it; a meta from
        # before the crop existed was a full load. The new instances may come from a world cropped otherwise (same task, same full scene).
        self.template_meta.setdefault("scene_crop", {"mode": "full"})
        live = self._world_scene_crop()
        rec = self.template_meta["scene_crop"]
        if (live.get("mode"), live.get("center_object"), live.get("radius_m")) != (rec.get("mode"), rec.get("center_object"), rec.get("radius_m")):
            log.warning("%s: template %s was saved from a SCENE_CROP %s world; this world is %s (template_meta.scene_crop keeps the template's)",
                        self.gp.variant, self.ctx.template_id, rec.get("mode"), live.get("mode"))
        log.info("%s: continuing template %s (%d instances so far, sha %s)", self.gp.variant, self.ctx.template_id, self.template_meta["n_written"], self.template_sha[:12])
        return True

    # ------------------------------------------------------------------ parking (§3.3)

    def park_keep_away(self) -> List[Any]:  # hook
        """Scene objects the parked robot also keeps 1 m from, beyond the spawned objects and the supports (default none)."""
        return []

    def path_blocked(self) -> List[Any]:  # hook
        """Floor footprints (world xy polygons) every recorded path plans around on the local map (default none)."""
        return []

    def gen_seeded(self, key: str, default=None):
        """``generation.<key>`` of the customs entry, else the subtask's ``generation_defaults`` seed of the key, else ``default``
        (oracle-fix 2026-11-06, owner feedback on the inside_large#02 demo): for a knob whose default the owner set per subtask in
        ``generation_defaults`` - ``GroundedProblem.gen`` reads the entry alone, and the counted templates read their own customs copy,
        seeded before the key existed."""
        v = self.gp.gen(key)
        if v is not None:
            return v
        from ispatialgym.sampling import generation_defaults as GD

        try:
            return GD.defaults_for(str(getattr(self.gp, "subtask", "") or "")).get(key, default)
        except KeyError:
            return default

    def base_path_style(self, carried: bool) -> str:
        """How a drive is checked and recorded (``robot.base_path.STYLES``): ``generation.base_path_style`` (owner 2026-09-26) for an
        empty-handed one, ``generation.carried_base_path_style`` (oracle-fix 2026-11-06, owner feedback on the inside_large#02 demo: "does
        this use the same navigation module as camera relocalization? the path doesn't seem smooth") for a carried one; "recorded" unless
        the entry or its subtask's defaults (:meth:`gen_seeded`; IORS large scale: "face_travel") say otherwise."""
        return str(self.gen_seeded("carried_base_path_style" if carried else "base_path_style", "recorded") or "recorded")

    def path_keep_off_footprints(self) -> List[Any]:
        """Opt-in ``generation.path_keep_off`` (oracle-fix 2026-11-04 round 4, pillow 158): ``[{"name": <object>, "margin_m": m}, ...]`` - every
        recorded path keeps its base centre ``margin_m`` off the object's AABB footprint, on the local AND the eroded global map (probe_pillow:
        the global path turned 0.41 m from bed_kddswt_1's side, its wheel sphere - 0.404 m from the base centre - 8.7 mm from the bed)."""
        out = []
        for rec in (self.gp.gen("path_keep_off") or []):
            obj = self.world.scene.object_registry("name", str(rec.get("name"))) if hasattr(self.world.scene, "object_registry") else None
            if obj is None:
                continue
            lo, hi = obj.aabb
            from ispatialgym.sampling.scene_variants import aabb_footprint

            out.append(aabb_footprint(lo, hi, float(rec.get("margin_m", 0.4))))
        return out

    def _log_once(self, key: str, msg: str, *args, level: int = logging.INFO) -> None:
        """One log line per sampler the first time a train-fast path is taken (``key``)."""
        seen = getattr(self, "_fast_logged", None)  # a sampler built without __init__ (the pure-python run_instances tests) gets the set here
        if seen is None:
            seen = self._fast_logged = set()
        if key not in seen:
            seen.add(key)
            log.log(level, msg, *args)

    def park_robot(self) -> th.Tensor:
        """§3.3: tucked at a free cell of the task rooms' floor, >= 1 m from every spawned object and support (and :meth:`park_keep_away`).

        train-fast PARK_ONCE (2026-10-31): the cell and the floor height are cached under :func:`park_key` (rooms, the keep-away and
        task objects' poses / joints), so a second park with nothing moved is the teleport + the five physics steps without
        ``NAV.park_pose``; the cuRobo re-sync after it is :meth:`_sync_after_park`. Switch off: today's path."""
        keep = list(self.world.spawned.values()) + self.support_objects() + self.park_keep_away()
        rooms = self.gp.rooms(self.scene) or (self.world.rooms or [])
        park_once = fast_on("PARK_ONCE")
        key = park_key(_dedupe(keep + list((getattr(self, "objects", None) or {}).values())), rooms) if park_once else None  # a stand-in without bindings (tests): keep-away set only
        cached = getattr(self, "_park_cache", None)  # a stand-in built without __init__ (tests) has no cache
        if park_once and cached is not None and cached[0] == key:
            _, xy, floor_z = cached
            self.counters["park_cell_reused"] = self.counters.get("park_cell_reused", 0) + 1
            self._log_once("park_once", "%s: PARK_ONCE: park cell (%.2f, %.2f) / floor z %.4f reused while the keep-away set is unchanged (no park_pose)",
                           self.gp.variant, float(xy[0]), float(xy[1]), floor_z)
        else:
            try:
                xy = NAV.park_pose(self.kit, rooms, keep_away=keep)
            except NAV.NavigationError as e:
                log.warning("%s: %s; parking at the room centre", self.gp.variant, e)
                xy = self.world.robot_park_xy
            # the task room's measured floor (``SimWorld.measure_floor_z``: raycast, cached per room); the kit's copy follows
            floor_z = self.world.measure_floor_z(xy)
            if park_once:
                self._park_cache = (key, xy, floor_z)
        self._park_xy = xy
        self.world.floor_z = floor_z
        self.kit.floor_z = self.world.floor_z
        self.world.park_robot(xy, floor_z=self.world.floor_z)
        if park_once:
            self._sync_after_park()
        else:
            self.kit.sync_world()
        return xy

    def _sync_after_park(self) -> None:
        """PARK_ONCE: the cuRobo re-sync after a park. The kit stores obstacles relative to the robot root, so a root that moved (every
        park after an ``apply_draft``: the robot comes from its stance) takes the full pass (``RobotKit.sync_world`` does so itself
        when the root drifted > 1 cm). When the root did not move - the previous attempt was rejected while the robot was still parked
        here (a ``sample`` rejection, the common case) - only the objects that can have moved since the last full sync are refreshed:
        the task objects (spawned ones re-parked, bound scene objects, articulated ones whose joints an attempt set) and the scene
        objects ``reset_scene_objects`` just put back (``SimWorld.last_reloaded``). Every other scene object is within 1 mm / 0.005 rad
        of the scene file (else it would have been reset and listed) - below the 1 cm collision margin of the checks."""
        backend = getattr(self.kit, "backend", None)
        if backend is None or not hasattr(backend, "root_pose"):  # a kit stand-in (tests): the plain sync
            self.kit.sync_world()
            return
        root = backend.root_pose()[0]
        prev = getattr(self.kit, "_synced_root", None)
        if prev is None or float(th.linalg.norm(root - prev)) > 0.01 or not getattr(backend, "_world_loaded", False):
            self.kit.sync_world()
            return
        objs = _dedupe(list((getattr(self, "objects", None) or {}).values()) + list((getattr(self.world, "spawned", None) or {}).values()))
        for name in getattr(self.world, "last_reloaded", None) or []:
            o = self.world.scene.object_registry("name", name)
            if o is not None and o not in objs:
                objs.append(o)
        self.counters["park_sync_partial"] = self.counters.get("park_sync_partial", 0) + 1
        self._log_once("park_once_sync", "%s: PARK_ONCE: robot root unmoved since the last full cuRobo sync; refreshing %d task / reset objects only",
                       self.gp.variant, len(objs))
        self.kit.sync_world(only_objects=objs)

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
            lift_end = PC.lift_end_pose(g)
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
            log.info("%s: carry waypoint recorded for %s (raise %.2f m, lateral %+.2f m): straight leg %s", self.gp.variant, g.object, raise_m, lat, why)
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

    def stance_check_ignores(self, stance) -> List[Any]:
        """Objects out of the world in :meth:`check_stances_with_objects` for ``stance`` (hook; none by default - oracle-fix 2026-11-04 round 3: the
        large IORS base push / push-under skip the target there, which stands at its INIT pose while the robot is elsewhere in the program)."""
        return []

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
                    extra = [o for o in self.stance_check_ignores(st) if o is not obj]  # after the sync above (a re-sync would undo an outer ignore)
                    with self.ikc.ignoring(([obj] if obj is not None else []) + extra):
                        if not self.ikc.stance_collision_free(st, **arm_q):
                            bad.append(st.name)
                            if os.environ.get("ISPATIALGYM_STANCE_BLAME"):  # oracle-fix 2026-11-04 round 3: which body the stance hits
                                try:
                                    log.info("%s: STANCE_BLAME settle stance %s (base %s trunk (%.2f, %.2f), posture %s): %s", self.gp.variant, st.name,
                                             [round(float(v), 2) for v in st.base_pose2d], st.translate, st.tilt, getattr(st, "idle_posture", None),
                                             self.ikc._collision_blame(self.ikc.q_locked(st, **arm_q), [obj] if obj is not None else [], 0.01)[:300])
                                except Exception as e:  # noqa: BLE001 - a diagnostic never fails the draw
                                    log.info("%s: STANCE_BLAME settle stance blame failed (%r)", self.gp.variant, e)
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
            extra_t = [o for o in self.stance_check_ignores(to) if o is not obj]  # oracle-fix 2026-11-04 round 3: as the stances above
            with self.object_joints_of(to), self.ikc.ignoring(([obj] if obj is not None else []) + extra_t):
                if tr.departure:
                    # a departure transition is re-swept at its recorded stand-off only (the recorded path starts at that pose)
                    new = NAV.departure_transition(self.kit, frm, to, tr.trunk_to, carried_obj=obj, carried_arm=arm, eef_in_object=eef_in_object,
                                                   min_standoff=tr.stand_off, arm_q_override=override)
                    if new is not None and abs(new.stand_off - tr.stand_off) > 1e-6:
                        new = None
                else:
                    dep = self.departure_of(tr.frm, tr.to)
                    s_to = float(to.stand_off if to.stand_off is not None else 0.0)
                    # oracle-fix 2026-11-06 (the shelf back-off): a record that carries the arms the oracle moves to after the drive-back
                    # (``arms_q``) and the pre-shaped fingers (``preshape_m``) is re-swept with them, and keeps them
                    arms_to_q, fingers_rec = NAV.recorded_arms_to(tr.extras)
                    new = NAV.transition(self.kit, frm, to, carried_obj=obj, carried_arm=arm, eef_in_object=eef_in_object,
                                         min_standoff=tr.stand_off, path_end_standoff=self.path_end_standoff(frm, to), arm_q_override=override,
                                         trunk_from=dep.trunk_to if dep is not None else None,
                                         extra_max=max(NAV.TRANSITION_EXTRA_MAX, float(tr.stand_off) - s_to + NAV.TRANSITION_EXTRA_MAX),
                                         arms_to_q=arms_to_q, fingers=fingers_rec)
                    if new is not None and arms_to_q:
                        new.holding = tr.holding
                        new.extras.update(tr.extras)
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
        slide; a face_travel record's schedule, ``NAV.record_sweep_poses``), at ``generation.carry_path_margin_m``. ``arm_q`` (per
        arm): the arms a task drove its carried path with (touching: this assignment's lift configuration); default as :meth:`record_path`
        (``carry_arm_q``, else the departing stance's arms). Returns why a sweep fails, None when every one passes."""
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
            # oracle-fix 2026-11-06: a face_travel record is swept as its schedule (+ the yaw-lag envelope), what the oracle drives
            poses = NAV.record_sweep_poses(p, yaw_envelope=p.mode != NAV.EDGE_SLIDE_MODE)
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
            log.warning("%s: carried %s has no grasp record with eef_in_object; attached at its current pose", self.gp.variant, carried)
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

    def record_stance_travel(self, frm: Stance, to: Stance) -> Transition:
        """The empty-handed move init -> manipulation: :meth:`record_transition`, unless ``to`` comes from the split cone
        (``tabletop.phase2_split_cone``): then the records of the pair are replaced by the transition when both stances share the edge
        lateral, else by the move along the edge of :meth:`record_edge_travel`, and ``to.extras["init_travel"]`` records the move.
        Returns the (arrival) transition."""
        if "_position" not in to.extras:
            return self.record_transition(frm, to)
        self.drop_travel_records(frm.name, to.name)
        lateral = float(th.dot(to.base_xy - frm.base_xy, frm.tangent))
        path, dep = None, None
        if abs(lateral) < 0.02:
            tr = self.record_transition(frm, to)
        else:
            path, dep, tr = self.record_edge_travel(frm, to)
        to.extras["init_travel"] = {"lateral_m": round(lateral, 4), "base_m": round(float(th.linalg.norm(to.base_xy - frm.base_xy)), 4),
                                    "path_m": None if path is None else round(float(path.length_m), 4),
                                    "departure_stand_off_m": None if dep is None else round(float(dep.stand_off), 4), "transition_stand_off_m": round(float(tr.stand_off), 4)}
        return tr

    def record_edge_travel(self, frm: Stance, to: Stance) -> Tuple[PathRecord, Optional[Transition], Transition]:
        """An empty-handed move along the edge, as block design's ``_travel_between``: when either trunk bows forward (tilt <=
        ``travel_straighten_tilt``) the departure transition first - drive back at least ``travel_back_min_m`` beyond ``frm``'s
        stand-off (0921-023009: a lateral drive 0.35 m from the edge brushed it), sweep to the travel trunk (``frm``'s translate, else
        ``travel_trunk_translate``, tilt 0) - then the path at that trunk and the arrival transition; otherwise, or when that fails, the
        §3.7 path to the drive-back pose plus the transition (``prompt0.connect``). Returns (path, departure or None, arrival)."""
        from ispatialgym.sampling.v2 import prompt0 as P0
        from ispatialgym.sampling.v2.ik import stance_arm_vectors

        max_len = float(self.gen("stance_path_max_m", 3.0))
        tilt_limit = float(self.gen("travel_straighten_tilt", -0.05))
        if float(frm.tilt) <= tilt_limit or float(to.tilt) <= tilt_limit:
            back_min = float(frm.stand_off or 0.0) + float(self.gen("travel_back_min_m", 0.10))
            for travel in ((float(frm.translate), 0.0), (float(self.gen("travel_trunk_translate", 0.95)), 0.0)):
                try:
                    dep = self.record_departure_transition(frm, to, travel, min_standoff=back_min)
                    r, l, _ = stance_arm_vectors(self.kit, to)  # the departure sweep puts the arms in the arriving stance's posture
                    path = self.record_path(frm, to, trunk=travel, arm_q={"right": r, "left": l}, start_pose2d=NAV.departure_pose(frm, dep.stand_off),
                                            max_length_m=max_len)
                    # oracle-fix 2026-11-06 (owner, blender close ep 3136: "why does it go to pre close twice?"): the path is swept with the
                    # arms the departure puts in the arriving stance's posture, so the record carries them - the oracle's navigate keeps an
                    # empty arm at a record's arms_q; without them it re-posed the arm to untucked and the arrival moved it out again
                    # (~360 of 887 steps in 78 / 101 isg-1020 blender-close demos)
                    if not getattr(path, "carried", False) and getattr(path, "arms_q", None) is None:
                        path.arms_q = {"right": [round(float(v), 5) for v in r.tolist()], "left": [round(float(v), 5) for v in l.tolist()]}
                    return path, dep, self.record_transition(frm, to, trunk_from=travel)
                except Reject as e:
                    log.info("%s: travel %s -> %s at trunk %.2f/%+.2f after the drive-back: %s", self.gp.variant, frm.name, to.name, travel[0], travel[1], e)
                    self.drop_travel_records(frm.name, to.name)
        path, tr = P0.connect(self, frm, to, path_max_m=max_len)
        return path, None, tr

    def drop_travel_records(self, frm: str, to: str) -> None:
        """Remove the paths / transitions (arrival and departure) recorded from stance ``frm`` to ``to``."""
        self.sidecar.paths[:] = [p for p in self.sidecar.paths if not (str(p.frm) == frm and str(p.to) == to)]
        self.sidecar.transitions[:] = [t for t in self.sidecar.transitions if not (str(t.frm) == frm and str(t.to) == to)]

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
                    edge_slide: bool = False, edge_slide_back_m: float = 0.0, axis_leg: Optional[Sequence[float]] = None,
                    arrive_straight_m: float = 0.0) -> PathRecord:
        """§3.7 path between two recorded stances, checked at the departing stance's trunk (``trunk`` overrides) with the
        carried object attached (posed in the hand per its grasp record); ``arm_q`` (per arm) fixes the arm posture of the
        drive (a carry: the lift configuration the oracle drives with; default for a carried object: its recorded
        ``carry_arm_q``); ``start_pose2d`` = where the path starts when it is not ``frm``'s base pose (the drive-back pose
        of a departure transition); ``force_global`` plans on the eroded map end to end; ``edge_slide`` (owner 2026-09-24)
        tries the same-edge slide (``NAV.plan_edge_slide``: body facing the edge, no turns, ``edge_slide_back_m`` extra
        stand-off for the slide) before the grid planner; ``axis_leg`` (owner 2026-10-13, the drive-backs to try): the edge
        slide alone (``NAV.plan_axis_leg``: the heading kept, forward / backward and lateral moves only), no grid planner;
        ``blocked`` (+ :meth:`path_blocked`) = footprints kept off the local map; ``arrive_straight_m`` (oracle-fix 2026-11-06,
        ``NAV.plan_path``): the grid path ends that far behind ``to`` along its axis and drives straight in; ``Reject(no_path |
        path_collision | path_too_long | place_leg_collision | place_leg_heading)``."""
        keep_off = self.path_keep_off_footprints()
        blocked = list(blocked) + list(self.path_blocked()) + list(keep_off)
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
            # oracle-fix 2026-11-04 round 3: ``generation.path_margin_m`` the same for the empty-handed legs (default 1 cm; the briefcase / cabinet #03
            # demos failed `collision` on navigate legs: an arm link on a wall, a finger on the neighbouring cabinet)
            margin = float(self.gp.gen("carry_path_margin_m", 0.01)) if (carried and obj is not None) else float(self.gp.gen("path_margin_m", 0.01))
            # owner 2026-09-26: ``generation.base_path_style`` (default "recorded") - a family that opts in has its UNCARRIED drives checked
            # and recorded as the face_travel schedule (recorded fallback when the schedule collides)
            # oracle-fix 2026-11-06 (owner feedback on the inside_large#02 demo: "the path doesn't seem smooth"; the jar's carried leg as recorded:
            # 17.2 s, 409 deg of rotation for a net -1 deg, the body 59 deg off the travel direction on average - 9.0 s / 102 deg as face_travel):
            # ``generation.carried_base_path_style`` the same for the CARRIED drives (default "recorded": the schedule only probed at 1 cm for the
            # morning report); ``generation.face_travel_fallback_cfgs`` (opt-in) = schedule configurations tried after the default one collides;
            # ``generation.face_travel_max_extra_rotation_deg``: a schedule turning the body more than the polyline (+ this) is not used
            is_carried = bool(carried and obj is not None)
            if axis_leg is not None:
                p = NAV.plan_axis_leg(self.kit, start, to, back_offs=list(axis_leg), posture=posture, carried=carried, carried_obj=obj, carried_arm=arm,
                                      max_length_m=max_length_m, trunk=trunk, eef_in_object=eef_in_object, arm_q=arm_q, margin=margin)
            else:
                style = self.base_path_style(is_carried)
                ft_cfgs = list(self.gen_seeded("face_travel_fallback_cfgs") or []) if style == "face_travel" else []
                ft_rot = self.gen_seeded("face_travel_max_extra_rotation_deg") if style == "face_travel" else None
                p = NAV.plan_path(self.kit, start, to, posture=posture, carried=carried, carried_obj=obj, carried_arm=arm, rooms=self.gp.rooms(self.scene) or self.world.rooms,
                                  blocked=blocked, max_length_m=max_length_m, trunk=trunk, eef_in_object=eef_in_object, force_global=force_global, arm_q=arm_q,
                                  margin=margin, edge_slide=edge_slide, edge_slide_back_m=edge_slide_back_m, style=style,
                                  probe_face_travel=is_carried and style != "face_travel", **({"global_blocked": keep_off} if keep_off else {}),
                                  **({"face_travel_cfgs": ft_cfgs} if ft_cfgs else {}),
                                  **({"face_travel_max_extra_rotation_deg": float(ft_rot)} if ft_rot is not None else {}),
                                  **({"arrive_straight_m": float(arrive_straight_m)} if arrive_straight_m else {}))
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

        report = PC.run(self.gp.variant, self.sidecar, self, subtask=self.subtask)
        # the wrist turn of a library pinch ranked on its whole held chain (owner 2026-09-27; ranking only, never a rejection)
        report = TT.held_turns(self, draft, report, lambda: PC.run(self.gp.variant, self.sidecar, self, subtask=self.subtask))
        # the arm of the segment ranked on its whole held chain (owner 2026-09-27, first-demo v2.2; ranking only, never a rejection)
        report = TT.held_arms(self, draft, report, lambda: PC.run(self.gp.variant, self.sidecar, self, subtask=self.subtask))
        self.last_program_check = report
        draft.summary["program_check"] = report
        # of-v2maps 2026-11-06: the grasps the opt-in support-policy fallback took (tabletop.support_policy_fallback: raised / relaxed, depth_mm)
        sp = [{"object": g.object, "grasp": g.grasp_name, **g.extras["support_policy"]} for g in self.sidecar.grasps
              if isinstance((getattr(g, "extras", None) or {}).get("support_policy"), dict)]
        if sp:
            draft.summary["support_policy"] = sp
        return report

    def oracle_state(self):
        """State object handed to ``Action.targets(sidecar, state)`` (F6 contract): object poses by bddl instance."""
        return {"objects": {n: [x.tolist() for x in o.get_position_orientation()] for n, o in self.objects.items()},
                "robot": {"base_pose2d": list(self.kit.world_base_pose2d())}}

    # ------------------------------------------------------------------ writing

    def save_template(self):
        """v1 template writer; the grounded BDDL goes to ``<variant>/activity_definitions/<activity>/problem<N>.bddl`` (§1).
        The v1 copy inside the template directory stays until D5 moves ``validate_instances.py`` to the §1 layout
        (its loader reads ``<template>/activity_definitions``)."""
        super().save_template()
        # train-fast SCENE_CROP (2026-11-01): what template-partial_rooms.json holds (the collector loads it with scene_crop.loaded_rooms)
        self.template_meta["scene_crop"] = self._world_scene_crop()
        bddl_dir = os.path.join(self.ctx2.variant_dir, "activity_definitions", self.activity)
        os.makedirs(bddl_dir, exist_ok=True)
        with open(os.path.join(bddl_dir, f"problem{self.definition_id}.bddl"), "w") as f:
            f.write(self.problem.text)
        self.template_meta["activity_definition"] = os.path.relpath(os.path.join(bddl_dir, f"problem{self.definition_id}.bddl"), self.ctx.out_dir)
        # v1 tools split ``problem`` as <activity>-<def_id>; the variant lives under ``variant``
        self.template_meta["problem"] = f"{self.activity}-{self.definition_id}"
        self.template_meta["variant"] = self.gp.variant
        self.template_meta["bddl_variant"] = self.problem.variant

    def check_witness_reach(self) -> None:
        """i7-attachfix (owner 2026-11-08 item 3): every recorded IK witness reaches its pose (an independent numpy FK,
        :mod:`witness_check`) before the instance is written; a miss rejects the attempt (``witness_unreachable``)."""
        from ispatialgym.sampling.v2 import witness_check as WC

        wit = getattr(getattr(self, "sidecar", None), "ik_witnesses", None)  # a sampler stub (tests) has no sidecar
        if not WC.enabled() or not wit:
            return
        bad = WC.witness_violations(wit)
        if bad:
            raise Reject("witness_unreachable", f"{len(bad)} recorded IK witness target(s) miss their pose", worst=sorted(bad, key=lambda e: -e["pos_m"])[:3])

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
        if not getattr(self.world, "cameras", True):
            # train-fast SAMPLER_CAMERAS (2026-10-31): this world has no head camera; the pixel screens of this instance are still to run
            summary["visibility"] = "pending"
        with open(instance_summary_path(path), "w") as f:  # i7-attachdef: pending until the post-write screens pass (switch)
            json.dump(_jsonable(summary), f, indent=2)
        if start_per_attempt():
            self.template_meta["start_per_attempt"] = True  # no template-level start: each instance carries its own (TRO, sidecar)
        elif "stances" not in self.template_meta:
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

    def instance_record(self, draft: Draft) -> SEP.InstanceRecord:
        """train-fast EARLY_REPEAT_FILTER (2026-10-31): the ``instance_separation.InstanceRecord`` of the draft as the written instance
        would read it - a minimal TRO dict (``robot_poses.robot[0]`` from ``draft.base_pose2d`` at z 0 with the yaw's quaternion, the
        ``ispatialgym`` block ``tro_meta_block`` builds for ``configure_targets`` before every write: activity, task_variant, scene
        model / task_rooms, ``pose_targets`` / ``grasps`` from the draft, and the draft's object poses as ``root_link`` states), the
        sidecar as it stands after ``sample`` (``goal.base_pose2d`` / ``goal.eef`` from the ``goal()`` hook) and the draft summary.
        No settle, IK or disk I/O: the poses are the draft's, millimetres from the settled ones the rule compares at 0.25 m / 15 deg."""
        x, y, yaw = (float(v) for v in draft.base_pose2d)
        tro: Dict[str, Any] = {"robot_poses": {"robot": [{"position": [x, y, 0.0], "orientation": [float(v) for v in G.quat_from_yaw(yaw).tolist()]}]},
                               "ispatialgym": self.tro_meta_block(draft, instance_id="draft", seed=0)}
        for name, pose in draft.object_poses.items():
            tro[name] = {"root_link": {"pos": [float(v) for v in G.t(pose[0]).tolist()], "ori": [float(v) for v in G.t(pose[1]).tolist()]}}
        # REPEAT_MEASURED (2026-11-01): the written TRO keys object states by the task's instance names - the draft's object poses are
        # keyed by scene object; both keys, so the measured objects' start states are found
        try:
            for b in self.bindings.all():
                if b.scene_object_name in draft.object_poses and b.name not in tro:
                    tro[b.name] = tro[b.scene_object_name]
        except Exception:  # noqa: BLE001 - a stand-in sampler without bindings (tests): record_complete then sends it to the post-write check
            pass
        return SEP.record_from_dicts(tro, self.sidecar.to_dict(), draft.summary, self.template_meta, path="draft")

    def check_early_repeat(self, draft: Draft) -> None:
        """EARLY_REPEAT_FILTER: :attr:`instance_filter`'s ``check_record`` on :meth:`instance_record` right after ``sample`` - the same
        ``Reject("repeats_eval")`` the post-write check raises, before the apply / settle / goal check / program check / write of an
        instance that would be deleted anyway. The record must carry every pose the written record compares
        (``instance_separation.record_complete``: start base + goal base; camera: goal camera; held tasks: the object's start and goal
        pose) - fewer poses would compare fewer constraints and could reject what the post-write check keeps - else, or when the
        filter has no ``check_record`` or the record cannot be built, this task falls back to the post-write check (logged once).
        The post-write check stays as the final guard."""
        if self.instance_filter is None or not fast_on("EARLY_REPEAT_FILTER"):
            return
        check_record = getattr(self.instance_filter, "check_record", None)
        if check_record is None:
            self._log_once("early_filter_off", "%s: EARLY_REPEAT_FILTER: the instance filter has no check_record; post-write check only", self.gp.variant)
            return
        try:
            rec = self.instance_record(draft)
            incomplete = SEP.record_complete(rec)
        except Exception as e:  # noqa: BLE001 - never lose an attempt to the early check: the post-write check decides
            self._log_once(f"early_filter_error:{self.activity}", "%s: EARLY_REPEAT_FILTER: draft record not built (%r); post-write check only for %s",
                           self.gp.variant, e, self.activity, level=logging.WARNING)
            return
        if incomplete:
            self._log_once(f"early_filter_incomplete:{self.activity}", "%s: EARLY_REPEAT_FILTER: draft record of %s is %s; post-write check only",
                           self.gp.variant, self.activity, incomplete)
            return
        self._log_once("early_filter_on", "%s: EARLY_REPEAT_FILTER: repeat rule checked on the draft after sample (poses %s -> %s) against %d eval instances",
                       self.gp.variant, [p.name for p in rec.start], [p.name for p in rec.goal], len(getattr(self.instance_filter, "pool", None) or []))
        why = check_record(rec)
        if why:
            self.counters["repeats_eval_early"] = self.counters.get("repeats_eval_early", 0) + 1
            raise Reject("repeats_eval", f"early: {why}")

    def check_instance_filter(self, tro_path: str) -> None:
        """:attr:`instance_filter` on the instance just written: when it gives a reason the instance's files are removed and the
        attempt is rejected (``repeats_eval``). With EARLY_REPEAT_FILTER (:meth:`check_early_repeat`, train-fast 2026-10-31) this is
        the cheap final guard on the files as written."""
        if self.instance_filter is None:
            return
        why = self.instance_filter(tro_path)
        if not why:
            return
        remove_instance_files(tro_path)
        self._count_withheld()
        if not self.instance_seeds:
            self.template_meta.pop("stances", None)
        raise Reject("repeats_eval", str(why))

    def _count_withheld(self) -> None:
        """i7-attachdef: a post-write rejection the switch kept from the queue (its summary was still pending)."""
        if publish_after_screens():
            self.counters["publish_after_screens.withheld"] = self.counters.get("publish_after_screens.withheld", 0) + 1

    def check_goal_occlusion(self, draft: Draft, tro_path: str) -> None:
        """Pre-generation goal-image occlusion (``ispatialgym.eval.goal_occlusion``): the goal state as the goal check
        settled it (its settled poses, the goal joints) with the robot where the program check ends
        (:meth:`final_configuration`), screened as the oracle screens a demo's last frame - the head camera's depth, the box
        screen per target / partner, the panel screen per fixture (a second depth render with the robot hidden). It runs
        after the instance is written, so the TRO is the one an unchecked run writes; a target / partner at least half
        hidden (an inside goal: the partner, its target is hidden by design) removes the instance's files and rejects the attempt
        (``goal_occluded``) before any demo is recorded."""
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
            objects = GO.screen_scene(depth, (cp.tolist(), cq.tolist()), intr, judged, depth_without_robot=depth_without, revolute=JointType.JOINT_REVOLUTE,
                                      subtask=self.subtask)
            rep = GO.report(objects, method="predicted: settled goal poses, robot at the program check's end, head-camera depth", robot=source,
                            seconds=round(time.time() - t0, 2))
        except Exception as e:  # noqa: BLE001 - a failed prediction keeps the instance; the oracle's gate still judges its last frame
            log.warning("%s: goal occlusion not predicted: %r", self.gp.variant, e)
            rep = {"error": repr(e), "occluded": False, "robot": source}
        finally:
            self.world.restore(state)
            self.world.keep_still(self.objects.values())
            self.robot.keep_still()
            self.kit.sync_world()
            self.configure_targets(draft)
        draft.summary["goal_occlusion"] = rep
        summary_path = instance_summary_path(tro_path)
        if rep["occluded"]:
            remove_instance_files(tro_path)
            self._count_withheld()
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

    def _world_scene_crop(self) -> Dict[str, Any]:
        """``template_meta.scene_crop`` of the live world (train-fast SCENE_CROP; ``{"mode": "full"}`` for a world without one)."""
        fn = getattr(self.world, "scene_crop_meta", None)
        try:
            out = fn() if callable(fn) else None
        except Exception:  # noqa: BLE001 - a stand-in world (tests)
            out = None
        return out if isinstance(out, dict) else {"mode": "full"}

    def write_template_meta(self):
        # train-fast SCENE_CROP: a template that never got saved still documents its world; a saved / adopted one keeps the record of
        # the world its template-partial_rooms.json came from
        self.template_meta.setdefault("scene_crop", self._world_scene_crop())
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

    def log_attempt_stages(self, index: int, attempt: int, clock: "_StageClock") -> None:
        """STAGE_TIMING: two INFO lines per attempt - the stage seconds and the attempt's kit counters (every ``*_calls`` / ``*_s`` /
        ``*_problems`` / ``*_configs`` key of ``RobotKit.stats_snapshot``, whatever the kit counts)."""
        log.info("%s: instance %d attempt %d stages: %s", self.gp.variant, index, attempt, clock.stages_text())
        log.info("%s: instance %d attempt %d kit: %s", self.gp.variant, index, attempt, format_kit_delta(clock.total_kit()))
        prof = getattr(self, "_sampler_profile", None)
        if prof is not None:  # the profile so far after every attempt (a cancelled job keeps it); dump_stats disables the profiler
            prof.dump_stats(os.environ["ISPATIALGYM_SAMPLER_PROFILE"])
            prof.enable()

    def run_instances(self, n: int, max_attempts: int = 30, start_index: Optional[int] = None, indices: Optional[Sequence[int]] = None) -> List[str]:
        """Per attempt: sample (task module) -> apply -> settle + BEHAVIOR validation -> goal feasibility (robot parked per
        §3.3) -> ``after_goal_check`` -> program check -> TRO + sidecar -> constraints write-back. Rejections are counted
        per reason with the first three examples; a :class:`TemplateExhausted` one ends the template's attempts (counter
        ``template_exhausted``). ``indices``: the instance indices to sample (default ``n`` from ``start_index``,
        else after the instances written). Returns the TRO paths written."""
        written: List[str] = []
        if indices is None:
            start = self.existing_instances() if start_index is None else int(start_index)
            indices = range(start, start + int(n))
        deadline = getattr(self, "deadline", None)
        out_of_time = exhausted = False
        for i_index, index in enumerate(indices):
            ok = False
            for attempt in range(max_attempts):
                # the variant's time budget (sample_isg --time-budget-s) ends the run between attempts only: an attempt in progress
                # always finishes, and a template whose setup finished gets its first attempt
                if deadline is not None and time.time() >= deadline and (attempt > 0 or i_index > 0):
                    out_of_time = True
                    self.counters["time_budget_stops"] = self.counters.get("time_budget_stops", 0) + 1
                    log.warning("%s: time budget used up; no attempt %d of instance %d", self.gp.variant, attempt, index)
                    break
                seed = int(self.ctx.base_seed * 1000003 + index * 1009 + attempt)
                rng = np.random.default_rng(seed)
                t0 = time.time()
                k0 = self.kit.stats_snapshot()
                # train-fast STAGE_TIMING (2026-10-31): seconds and kit counters per stage, logged after every attempt (success or rejection)
                clock = _StageClock(self.kit)
                phase = "reset"
                try:
                    n_reset = self.world.reset_scene_objects([o.name for o in self.objects.values()],
                                                             context=f"{self.gp.variant}: instance {index} attempt {attempt}")
                    if n_reset:
                        # a counter (scene objects put back before the attempt), not a rejection
                        self.counters["scene_reset"] = self.counters.get("scene_reset", 0) + n_reset
                    if fast_on("PARK_SPAWNED"):
                        # train-fast PARK_SPAWNED (2026-10-31): every attempt starts with the template's spawned objects parked (the on_reject
                        # parking; after a written instance they would otherwise still sit on the support while the next attempt samples)
                        n_parked = self.park_spawned_objects()
                        if n_parked:
                            self.counters["spawned_parked_at_reset"] = self.counters.get("spawned_parked_at_reset", 0) + n_parked
                            self._log_once("park_spawned", "%s: PARK_SPAWNED: %d spawned objects parked before the attempt (one physics step)", self.gp.variant, n_parked)
                    self.park_robot()
                    self.ikc.solutions.clear()
                    self.ikc.attempt = attempt  # ``IKChecker.LIMIT_RETRY_RANDOM_FROM_ATTEMPT``: attempt 0 solves without the random re-solves
                    clock.mark("reset")
                    phase = "sample"
                    draft = self.sample(rng)
                    clock.mark("sample")
                    t_sample = time.time() - t0
                    phase = "instance_filter"
                    self.check_early_repeat(draft)  # EARLY_REPEAT_FILTER: Reject("repeats_eval", "early: ...") before any physics / IK / write
                    clock.mark("instance_filter")
                    phase = "apply"
                    self.apply_draft(draft)
                    clock.mark("apply")
                    t_apply = time.time() - t0 - t_sample
                    phase = "settle_validate"
                    self.settle_and_validate(draft)
                    # §4.6 step 5 with the objects at their settled (= recorded) poses
                    self.check_stances_with_objects(draft)
                    clock.mark("settle_validate")
                    t_settle = time.time() - t0 - t_sample - t_apply
                    phase = "goal_check"
                    self.check_goal_feasible(draft)
                    self.after_goal_check(draft)
                    self.verify_carry_legs(draft)
                    clock.mark("goal_check")
                    phase = "program_check"
                    self.program_check(draft)
                    clock.mark("program_check")
                    phase = "witness_check"
                    self.check_witness_reach()
                    phase = "write"
                    if self.template_paths is None:
                        self.save_template()
                    draft.summary["wall_time_s"] = time.time() - t0
                    # the four legacy keys as before (train scripts read them) + the split stages (STAGE_TIMING)
                    draft.summary["phase_s"] = {"sample": round(t_sample, 2), "apply": round(t_apply, 2), "settle_validate": round(t_settle, 2),
                                                "goal_check_program_check_write": round(time.time() - t0 - t_sample - t_apply - t_settle, 2),
                                                **{k: round(v, 2) for k, v in clock.seconds.items() if k not in ("sample", "apply", "settle_validate")}}
                    draft.summary["kit"] = self.kit.stats_delta(k0, self.kit.stats_snapshot())
                    draft.summary["kit_attempt"] = {"by_stage": {k: dict(v) for k, v in clock.kit_by_stage.items()}, "total": clock.total_kit()}
                    draft.summary["attempt"] = attempt
                    path = self.write_instance(draft, index, seed)
                    clock.mark("write")
                    phase = "instance_filter"
                    self.check_instance_filter(path)
                    clock.mark("instance_filter")
                    phase = "goal_occlusion"
                    self.check_goal_occlusion(draft, path)
                    self.constraints_write_back(draft, self.instance_id(index))
                    clock.mark("goal_occlusion")
                    if publish_instance(path):  # i7-attachdef: only now can a collector claim it (ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS)
                        self.counters["publish_after_screens.published"] = self.counters.get("publish_after_screens.published", 0) + 1
                    self.instance_seeds.append(seed)
                    written.append(path)
                    # run_isg loads an instance through the template meta; demos replay instances while the rest of the
                    # template is still being sampled
                    self.write_template_meta()
                    log.info("%s: instance %d written (attempt %d, %.1fs)", self.gp.variant, index, attempt, time.time() - t0)
                    self.log_attempt_stages(index, attempt, clock)
                    ok = True
                    break
                except Reject as e:
                    clock.mark(phase)
                    self.rejection_log.add(e.reason, str(e), phase=phase, index=index, attempt=attempt, seed=seed, **_plain_details(e.details))
                    log.debug("rejection traceback", exc_info=True)
                    self.rejections[e.reason] = self.rejections.get(e.reason, 0) + 1
                    log.info("%s: instance %d attempt %d rejected in %s after %.1fs: %s", self.gp.variant, index, attempt, phase, time.time() - t0, e)
                    self.log_attempt_stages(index, attempt, clock)  # STAGE_TIMING: separate lines, the rejection line above keeps its format
                    self.on_reject(e)
                    if isinstance(e, TemplateExhausted) and not start_per_attempt():
                        exhausted = True
                        self.counters["template_exhausted"] = self.counters.get("template_exhausted", 0) + 1
                        log.warning("%s: template exhausted at instance %d attempt %d; no further attempt", self.gp.variant, index, attempt)
                        break
            if out_of_time or exhausted:
                break
            if not ok:
                log.error("%s: instance %d failed after %d attempts; rejections=%s", self.gp.variant, index, max_attempts, self.rejection_log.counts)
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


# ----------------------------------------------------------------------------------------------------------------
# train-fast (2026-10-31): per-stage timing of an attempt, the park cache key, compact kit-counter formatting
# ----------------------------------------------------------------------------------------------------------------

#: stage -> short name in the per-attempt ``stages:`` log line (STAGE_TIMING)
STAGE_SHORT = {"reset": "reset", "sample": "sample", "instance_filter": "filter", "apply": "apply", "settle_validate": "settle", "goal_check": "goal",
               "program_check": "program", "write": "write", "goal_occlusion": "occlusion"}


class _StageClock:
    """STAGE_TIMING (always on): wall seconds and kit-counter deltas (``RobotKit.stats_snapshot``, any key) per stage of one attempt.
    ``mark(stage)`` closes the stage that ran since the previous mark (a stage marked twice accumulates: the repeat filter runs
    early and after the write)."""

    def __init__(self, kit):
        self.kit = kit
        self.t0 = time.time()
        self.seconds: Dict[str, float] = {}
        self.kit_by_stage: Dict[str, Dict[str, float]] = {}
        self._t = self.t0
        self._k0 = self._snap()
        self._k = dict(self._k0)

    def _snap(self) -> Dict[str, float]:
        snap = getattr(self.kit, "stats_snapshot", None)  # a stub kit (tests) may have no counters
        return dict(snap()) if callable(snap) else {}

    @staticmethod
    def _delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        return {k: round(float(after.get(k, 0.0)) - float(before.get(k, 0.0)), 3) for k in set(before) | set(after)}

    def mark(self, stage: str) -> None:
        now, k = time.time(), self._snap()
        self.seconds[stage] = round(self.seconds.get(stage, 0.0) + (now - self._t), 3)
        delta = {kk: v for kk, v in self._delta(self._k, k).items() if v}
        if delta:
            acc = self.kit_by_stage.setdefault(stage, {})
            for kk, v in delta.items():
                acc[kk] = round(acc.get(kk, 0.0) + v, 3)
        self._t, self._k = now, k

    def total_kit(self) -> Dict[str, float]:
        return {k: v for k, v in self._delta(self._k0, self._snap()).items() if v}

    def elapsed(self) -> float:
        return time.time() - self.t0

    def stages_text(self) -> str:
        order = list(STAGE_SHORT) + [k for k in self.seconds if k not in STAGE_SHORT]
        parts = [f"{STAGE_SHORT.get(k, k)}={self.seconds[k]:.1f}s" for k in order if k in self.seconds]
        return " ".join(parts) + f" total={self.elapsed():.1f}s"


def format_kit_delta(delta: Dict[str, float]) -> str:
    """``{"ik_calls": 12, "ik_s": 1.4, "ik_batch_calls": 3, "ik_batch_problems": 210, ...}`` -> ``"ik 12 calls 1.4s, ik_batch 3 calls 9.8s
    (210 problems), ..."``; generic over the counter names (``<name>_calls`` / ``_s`` / ``_problems`` / ``_configs``, anything else
    printed as is); counters at zero are left out."""
    groups: Dict[str, Dict[str, float]] = {}
    for k, v in (delta or {}).items():
        for suf in ("_calls", "_problems", "_configs", "_s"):
            if k.endswith(suf) and len(k) > len(suf):
                groups.setdefault(k[: -len(suf)], {})[suf] = v
                break
        else:
            groups.setdefault(k, {})[""] = v
    parts = []
    for name, g in groups.items():
        if not any(abs(float(v)) > 0 for v in g.values()):
            continue
        txt = name
        if "_calls" in g:
            txt += f" {int(g['_calls'])} calls"
        if "_s" in g:
            txt += f" {float(g['_s']):.1f}s"
        extras = [f"{int(g[suf])} {suf[1:]}" for suf in ("_problems", "_configs") if g.get(suf)]
        if "" in g:
            extras.append(f"{g['']:g}" if isinstance(g[""], float) else str(g[""]))
        if extras:
            txt += " (" + ", ".join(extras) + ")"
        parts.append(txt)
    return ", ".join(parts) if parts else "none"


def _dedupe(objs: Sequence[Any]) -> List[Any]:
    """The objects in order without repeats, by identity (an unhashable stand-in in the tests is fine; train-fast 2026-10-31)."""
    seen: set = set()
    out: List[Any] = []
    for o in objs:
        if id(o) not in seen:
            seen.add(id(o))
            out.append(o)
    return out


def park_key(objs: Sequence[Any], rooms: Sequence[str]) -> Tuple:
    """PARK_ONCE: what ``park_robot``'s answer depends on besides the (fixed) scene - the task rooms and, per object the parked robot
    keeps away from or that an attempt can move, its name, root pose (5 cm / 0.05 quaternion units) and joint positions (0.05 rad).
    A key that differs recomputes the cell (never wrong, only slower); the rounding makes a settled object's millimetre jitter reuse it."""
    parts: List[Any] = [tuple(rooms)]
    for o in objs:
        try:
            p, q = o.get_position_orientation()
            item: Tuple[Any, ...] = (o.name, tuple(round(float(v) / 0.05) for v in p.tolist()), tuple(round(float(v) / 0.05) for v in q.tolist()))
            if int(getattr(o, "n_joints", 0) or 0) > 0:
                item += (tuple(round(float(v) / 0.05) for v in o.get_joint_positions().tolist()),)
        except Exception:  # noqa: BLE001 - an object without a readable pose: never reuse
            item = (getattr(o, "name", repr(o)), id(o), time.time())
        parts.append(item)
    return tuple(parts)
