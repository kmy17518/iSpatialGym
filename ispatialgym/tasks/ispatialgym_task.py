"""``ISpatialGymTask``: BEHAVIOR-compatible task that scores iSpatialGym problems (owner 2026-10-14, ``ispatialgym.eval.metrics``).

Per step ``t``: ``G(t)`` - every goal atom except ``pose_settled`` holds (``pose_reached`` / ``layout_reached`` are configured
from the TRO and may compare a full pose, only the depth along the head-camera axis, only the orientation, joint positions, or
relative poses between group members); ``M(t)`` - every ``(:measure ...)`` atom is within its tolerance; ``S(t)`` - every
``pose_settled`` atom holds (its window of ``settle_steps`` steps ending at ``t`` is full and within the drift thresholds).

**Symbolic** success is the BDDL ``:goal``; **geometric** success (the primary metric) is symbolic success plus every
``:measure`` atom within tolerance at the same step(s). ``settle_eval`` off: symbolic = the first ``t`` with ``G(t)``,
geometric = the first with ``G(t)`` and ``M(t)``; on: the first ``t`` with ``S(t)`` where ``G`` (``G`` and ``M``) held at every
step of the window. Both modes are tracked; the run's mode gives ``info["success"]``, ``official_success`` and the termination
(its geometric success), the other mode is reported in ``other_mode``. The tolerances are the ``isg-v2`` profile's
(``ispatialgym.eval.tolerance_profile``, :meth:`ISpatialGymTask.apply_tolerance_profile` at load).

Everything is surfaced in ``info["ispatialgym"]`` of ``env.step``; see :meth:`ISpatialGymTask._populate_isg_info`.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import LayoutReached, PoseReached, PoseSettled
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.tasks.task_base import BaseTask
from omnigibson.termination_conditions.predicate_goal import PredicateGoal
from omnigibson.termination_conditions.timeout import Timeout
from ispatialgym.sampling import predicates as PRED  # robust under / ontop (D-007 option A): the same wrapper the sampler's goal check uses
from omnigibson.utils.python_utils import classproperty, recursively_convert_to_torch

from ispatialgym.activity.problem import (
    ACTIVITY_DEFINITIONS_ROOT,
    AGENT_INSTANCE,
    ISGProblem,
    canonical_activity,
    is_agent_instance,
    load_problem,
    to_bddl,
)
from ispatialgym.eval import metrics as MET
from ispatialgym.eval import tolerance_profile as TP
from ispatialgym.eval.group_entity import GroupEntity
from ispatialgym.eval.tro import (
    TROFile,
    apply_targets_to_scope,
    load_tro,
    physical_instances,
    validate_tro_against_problem,
)

log = logging.getLogger(__name__)

#: ``PoseSettled`` configuration of every settled entity when the task gets none (owner 2026-09-24). The stock state
#: (10 steps = 0.33 s, drift 10 mm / 50 mrad / 50 mrad) was gated by its per-step velocity thresholds; those never hold for
#: a body in resting contact or on a driven joint (the contact-solver residual reads 1-2.5 mm/s on the robot's base and
#: 0.02-0.04 rad/s on its wrists, ~1-3 mm/s on an attached or resting object, with micrometres of motion), so the
#: evaluator decides ``pose_settled`` by the window's configuration drift alone (``sampling.predicates``,
#: ``DRIFT_SETTLED_ENTITIES``) and this window carries the rest criterion:
#:
#: * ``n_steps`` 45 control steps (1.5 s) - the default ``settle_steps`` of oracle and policy runs (2026-10-14: the window is
#:   ``run_isg --settle-steps`` in every mode, ``ISpatialGymTask.settle_steps`` overrides this value);
#: * ``pos_tolerance`` 2 mm, ``ori_tolerance`` 5 mrad, ``joint_tolerance`` 5 mrad (metres for prismatic joints) - the tightest
#:   round values every settled entity of those 28 demos clears at its last step with margin: their end-of-episode drift
#:   over 1.5 s is at most 0.074 mm, 0.98 mrad (the float32 quaternion resolution) and 0.036 rad/1000 on the robot's
#:   joints, with no measurable creep. A painting creeping 0.25 deg/s on its hinge (run 0922-180242) drifts 6.5 mrad over
#:   the window and fails; an object rolling 3 mm/s drifts 4.5 mm and fails.
DEFAULT_POSE_SETTLED_CONFIG: Dict[str, float] = {"n_steps": 45, "pos_tolerance": 0.002, "ori_tolerance": 0.005, "joint_tolerance": 0.005}

#: ISPATIALGYM_SHARED_GOAL_EVAL=1: within one task step the potential reward reuses the termination's goal evaluation (the
#: same goal heads on the same state) and appends that evaluation's branch-log entries again, as its own evaluation would;
#: a step whose termination evaluation created an attachment joint evaluates again (the second evaluation then logs the
#: atom as decided by OmniGibson). =check evaluates again on every step, compares the potential and the log entries and
#: prints SHARED_GOAL_CHECK counts.
SHARED_GOAL_EVAL = os.environ.get("ISPATIALGYM_SHARED_GOAL_EVAL", "")
_SHARED_COUNTS = {"steps": 0, "reused": 0, "joint_steps": 0, "mismatches": 0, "potential_s": 0.0}

#: The assisted-grasp joint of a link behind a movable joint (a lid, a door leaf, a flap), set on every robot of the env
#: (our Behavior-1K fork's ``Robot.ag_moving_link_joint_type``): fixed instead of OmniGibson's spherical one, which couples
#: position only - a handle pinched across a 13 mm tab could not turn a blender lid on its 4 cm lever (2026-09-18). The
#: demos are recorded under this rule, and an evaluated policy meets the same one.
AG_MOVING_LINK_JOINT_TYPE = "FixedJoint"
#: The assisted grasp's release, set on every robot of the env (our Behavior-1K fork's ``Robot.ag_release_on_pad_clear`` /
#: ``ag_release_max_delay_s``, owner 2026-10-10): after the open command (+1, binary) the weld stays until a pad has left the
#: object, at most this long. Let go at the command itself, the pads still clamped it: r14 tipped a block-design die to
#: 88 deg and turned the free-standing carton 0.86 deg in one step. Demos are recorded and policies evaluated under it.
AG_RELEASE_ON_PAD_CLEAR = True
AG_RELEASE_MAX_DELAY_S = 0.2


def _compact(expr) -> str:
    return " ".join(to_bddl(expr).split())


def _robots(robots):
    """The robots of ``env.robots`` (one list per scene in the vectorized environment)."""
    stack = list(robots or [])
    while stack:
        robot = stack.pop()
        if isinstance(robot, (list, tuple)):
            stack.extend(robot)
        else:
            yield robot


def apply_ag_joint_rule(robot) -> None:
    """The assisted-grasp joint rule ``initialize_activity`` sets on every robot of the env, on one robot."""
    robot.ag_moving_link_joint_type = AG_MOVING_LINK_JOINT_TYPE


def apply_environment_rules(robots) -> int:
    """The rules ``initialize_activity`` applies to the env's robots - the attach rule's release hook, the assisted-grasp
    joint rule and its pad-clear release - for a task that re-implements it (``VecISpatialGymTask``). ``robots``:
    ``env.robots`` (one list per scene in the vectorized environment). Returns the number of robots."""
    PRED.install_release_hook(robots)
    n = 0
    for robot in _robots(robots):
        apply_ag_joint_rule(robot)
        robot.ag_release_on_pad_clear = AG_RELEASE_ON_PAD_CLEAR
        robot.ag_release_max_delay_s = AG_RELEASE_MAX_DELAY_S
        n += 1
    return n


class ISpatialGymTask(BehaviorTask):
    """
    Args:
        activity_name (str): iSpatialGym activity (directory under ``activity_definitions_root``). A retired name
            (``world_alignment``) is accepted and reported under its canonical name (``alignment``); a legacy root
            whose directory still carries the retired name is found either way.
        activity_definition_id (int): problem file number (``problemN.bddl``).
        activity_instance_id (int): instance id (bookkeeping only; the TRO file is given explicitly).
        activity_definitions_root (None or str): override for the activity definitions directory.
        tro_path (None or str): TRO file whose ``ispatialgym`` block configures the evaluation targets. The object
            states it stores are applied by :meth:`load_tro_instance` (requires a playing simulator).
        object_bindings (None or dict): ``{bddl_instance: scene_object_name}`` for every physical instance. If
            None, the TRO ``object_bindings`` are used, else the scene task metadata ``inst_to_name``.
        settle_eval (bool): the run's settle-eval mode (module docstring): its geometric success is ``task.success`` and
            ends the episode; the other mode is reported.
        settle_steps (int): the ``pose_settled`` window (``PoseSettled`` ``n_steps``) and the settle-eval window, in steps.
        pose_settled_config (None or dict): kwargs forwarded to ``PoseSettled.activate`` for every settled entity, with
            ``n_steps`` = ``settle_steps`` (None = :data:`DEFAULT_POSE_SETTLED_CONFIG`'s drift thresholds; ``{}`` =
            OmniGibson's stock state).
        highlight_task_relevant_objects, termination_config, reward_config, include_obs: see ``BehaviorTask``.
    """

    def __init__(
        self,
        activity_name,
        activity_definition_id=0,
        activity_instance_id=0,
        activity_definitions_root=None,
        tro_path=None,
        object_bindings=None,
        settle_eval=False,
        settle_steps=DEFAULT_POSE_SETTLED_CONFIG["n_steps"],
        pose_settled_config=None,
        highlight_task_relevant_objects=False,
        termination_config=None,
        reward_config=None,
        include_obs=False,
    ):
        assert gm.ENABLE_OBJECT_STATES, "Must set gm.ENABLE_OBJECT_STATES=True in order to use ISpatialGymTask!"

        # --- BehaviorTask attributes (BehaviorTask.__init__ is bypassed: it requires a knowledge-base activity) ---
        self.activity_name = activity_name
        self.activity_definition_id = int(activity_definition_id)
        self.activity_instance_id = activity_instance_id
        self.compiled_task = None
        self.activity_initial_conditions = None
        self.activity_goal_conditions = None
        self.ground_goal_state_options = None
        self.feedback = None
        self.sampler = None
        self.scene_name = None
        self.online_object_sampling = False
        self.use_presampled_robot_pose = False
        self.randomize_presampled_pose = False
        self.sampling_whitelist = None
        self.sampling_blacklist = None
        self.highlight_task_relevant_objs = highlight_task_relevant_objects
        self.object_scope = None
        self.object_instance_to_category = None
        self.future_obj_instances = set()
        self._pose_settled_insts = []
        self._other_goal_head_idxs = []
        self._pose_settled_other_success = [{}]  # BehaviorTask keeps this per env; an ISpatialGymTask drives one env
        self._t_other_success = [-1]
        self.instruction_order = None
        self.currently_viewed_index = None
        self.currently_viewed_instruction = None
        self.activity_natural_language_initial_conditions = None
        self.activity_natural_language_goal_conditions = None

        # --- iSpatialGym ---
        self.activity_definitions_root = activity_definitions_root or ACTIVITY_DEFINITIONS_ROOT
        self.problem: Optional[ISGProblem] = None
        self.task_variant = None
        self.groups: Dict[str, list] = {}
        self.settle_eval = bool(settle_eval)
        self.settle_steps = int(settle_steps)
        self._settle_steps_envs: list = [self.settle_steps]
        self.pose_settled_config = {**(DEFAULT_POSE_SETTLED_CONFIG if pose_settled_config is None else pose_settled_config), "n_steps": self.settle_steps}
        self._config_object_bindings = dict(object_bindings) if object_bindings else None
        self._tro: Optional[TROFile] = None
        self._tro_path = tro_path
        self._targets_configured = False
        self._target_metas: Dict[int, Any] = {}
        self._measure_atoms: list = []
        self._pose_settled_head_idxs = []
        #: the ``isg-v2`` record of :meth:`apply_tolerance_profile` (meta.json ``tolerance_profile``), per env
        self.tolerance_profiles: Dict[int, Dict[str, Any]] = {}
        self._reset_tracking()

        BaseTask.__init__(self, termination_config=termination_config, reward_config=reward_config, include_obs=include_obs)

    # ------------------------------------------------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------------------------------------------------

    @classmethod
    def verify_scene_and_task_config(cls, scene_cfg, task_cfg):
        # No scene-instance rewriting: iSpatialGym instances name their scene explicitly (TRO ``scene``).
        BaseTask.verify_scene_and_task_config(scene_cfg=scene_cfg, task_cfg=task_cfg)

    @property
    def object_scopes(self):
        """The per-env scopes the (vectorized) ``BehaviorTask`` reads: an ISpatialGymTask binds exactly one env, whose
        scope is :attr:`object_scope`."""
        return None if self.object_scope is None else [self.object_scope]

    @object_scopes.setter
    def object_scopes(self, scopes):
        self.object_scope = None if scopes is None else scopes[0]

    def _load(self, env):
        assert env.num_envs == 1, f"ISpatialGymTask binds one environment, got num_envs={env.num_envs}"
        self.update_activity(env=env, activity_name=self.activity_name, activity_definition_id=self.activity_definition_id)
        if self._tro_path is not None:
            self._tro = load_tro(self._tro_path)
            validate_tro_against_problem(self._tro.meta, self.compiled_task, require_bindings=False)
        success, self.feedback = self.initialize_activity(env=env)
        assert success, f"Failed to initialize iSpatialGym activity. Feedback:\n{self.feedback}"
        self.scene_name = env.scene.scene_model if isinstance(env.scene, TraversableScene) else None

        if self.highlight_task_relevant_objs:
            for inst, entity in self.physical_object_scope.items():
                if entity is not None and not is_agent_instance(inst):
                    entity.highlighted = True

        callback_name = f"{self.activity_name}_{self.activity_definition_id}_refresh"
        og.sim.add_callback_on_add_obj(name=callback_name, callback=self._update_bddl_scope_from_added_obj)
        og.sim.add_callback_on_remove_obj(name=callback_name, callback=self._update_bddl_scope_from_removed_obj)

    def update_activity(self, env, activity_name, activity_definition_id):
        # The evaluator keys on the canonical activity name; retired names (``world_alignment``) are accepted on input
        # and resolve to the same problem files (``activity.problem.LEGACY_ACTIVITIES``).
        self.activity_name = canonical_activity(activity_name)
        if self.activity_name != activity_name:
            log.info(f"Activity {activity_name!r} is a retired name of {self.activity_name!r}")
        self.activity_definition_id = int(activity_definition_id)
        self.problem = load_problem(activity_name, activity_definition_id, root=self.activity_definitions_root)
        self.compiled_task = self.problem.compile()
        self.task_variant = self.problem.variant
        self.groups = {g: list(members) for g, members in self.problem.groups.items()}

        # Scope: agent first, then declared instances, then the nonphysical group entities.
        self.object_scope = {AGENT_INSTANCE: None}
        for inst in self.problem.instances:
            self.object_scope.setdefault(inst, None)
        for g, members in self.groups.items():
            self.object_scope[g] = GroupEntity(g, members, resolve_member=self._resolve_member)
        self.object_instance_to_category = dict(self.problem.instance_to_synset)

        self.activity_initial_conditions = self.compiled_task.initial_conditions
        self.activity_goal_conditions = self.compiled_task.goal_conditions
        self.ground_goal_state_options = self.compiled_task.ground_goal_state_options
        self._pose_settled_insts = list(self.compiled_task.pose_settled_insts)
        self._pose_settled_head_idxs = list(self.compiled_task.pose_settled_head_idxs)
        self._other_goal_head_idxs = list(self.compiled_task.binary_goal_head_idxs)
        self._measure_atoms = MET.measure_atoms(self.problem)
        MET.error_metric_case(self._measure_atoms, self.problem_label)  # an episode is judged by it: fail at load, not at the end

        n_goals = len(self.compiled_task.conditions.parsed_goal_conditions)
        self.instruction_order = th.arange(n_goals)
        self.instruction_order = self.instruction_order[th.randperm(self.instruction_order.size(0))]
        self.currently_viewed_index = 0
        self.currently_viewed_instruction = self.instruction_order[self.currently_viewed_index]
        self.activity_natural_language_initial_conditions = self.compiled_task.natural_language_initial_conditions
        self.activity_natural_language_goal_conditions = self.compiled_task.natural_language_goal_conditions
        self._targets_configured = False

    def _resolve_member(self, inst):
        return self.object_scope.get(inst) if self.object_scope else None

    def initialize_activity(self, env):
        """Bind every declared instance to a scene entity (no sampling) and configure targets if a TRO is loaded."""
        bindings = self._object_bindings_source(env)
        missing = []
        for inst in physical_instances(self.compiled_task):
            name = bindings.get(inst)
            if isinstance(name, dict):
                name = name.get("scene_object_name")
            if name is None:
                missing.append(inst)
                continue
            entity = env.scene.object_registry("name", name)
            if entity is None:
                return False, f"Object binding {inst!r} -> {name!r} does not exist in the scene"
            self.object_scope[inst] = entity
        if missing:
            return False, f"No object binding for instances {missing}"
        self.object_scope[AGENT_INSTANCE] = self.get_agent(env)
        # the attach rule also acts at the instant a robot's assisted grasp lets go (our Behavior-1K fork's release hook): a
        # part released aligned with its partner is joined before it can fall through the collision-filtered partner
        PRED.install_release_hook(getattr(env, "robots", None))
        for robot in _robots(getattr(env, "robots", None)):
            robot.ag_moving_link_joint_type = AG_MOVING_LINK_JOINT_TYPE
            robot.ag_release_on_pad_clear = AG_RELEASE_ON_PAD_CLEAR
            robot.ag_release_max_delay_s = AG_RELEASE_MAX_DELAY_S
        if self._tro is not None:
            self.configure_targets(self._tro.meta)
        return True, None

    def _object_bindings_source(self, env) -> Dict[str, Any]:
        if self._config_object_bindings:
            return self._config_object_bindings
        if self._tro is not None and self._tro.meta.object_bindings:
            return self._tro.meta.object_bindings
        try:
            return env.scene.get_task_metadata(key="inst_to_name") or {}
        except Exception:
            return {}

    def configure_targets(self, meta):
        """Push TRO pose / group / articulation targets into the bound scope entities (strict)."""
        apply_targets_to_scope(meta, self.object_scope)
        self._targets_configured = True
        self._target_metas[0] = meta
        for entity in self.group_entities.values():
            entity.invalidate()
        PRED.set_attachment_context(self._attachment_pairs(meta))

    def _attachment_pairs(self, meta) -> list:
        """The TRO's ``attachment_targets`` as the predicate wrapper's context (owner 2026-09-19 17:10): scene object names of
        the child / parent, the meta-link pair, and the child's symmetry — its pose target's ``orientation_equivalences``
        when it has one, else those of the group both parts belong to (the sampler writes the male's mating-axis symmetry
        there). With it the ``attached`` atom is decided modulo that symmetry (``predicates.attach_if_aligned``)."""
        out = []
        for rec in (meta.extras or {}).get("attachment_targets") or []:
            child_inst, parent_inst = rec.get("child"), rec.get("parent")
            child, parent = self.object_scope.get(child_inst), self.object_scope.get(parent_inst)
            if child is None or parent is None or isinstance(child, GroupEntity) or isinstance(parent, GroupEntity):
                log.warning(f"attachment_targets: {child_inst!r} -> {parent_inst!r} is not a bound pair of physical instances; skipped")
                continue
            eqs = None
            pt = meta.pose_targets.get(child_inst)
            if pt is not None and pt.orientation_equivalences:
                eqs = pt.orientation_equivalences
            else:
                for g in meta.groups.values():
                    if child_inst in g.members and parent_inst in g.members and g.orientation_equivalences:
                        eqs = g.orientation_equivalences
                        break
            out.append({"child": child.name, "parent": parent.name, "child_link": rec.get("child_link"), "parent_link": rec.get("parent_link"),
                        "equivalences": [[float(v) for v in th.as_tensor(q).reshape(-1).tolist()] for q in (eqs or [])]})
        return out

    def load_tro_instance(self, env, tro=None, settle_steps=25, apply_object_states=True):
        """Load a TRO instance into the running environment (simulator must be playing).

        Applies the serialized object states and the robot root pose stored in the TRO, restores robot joint
        positions from ``robot_state`` when present, configures the evaluation targets, settles the scene while
        keeping every task-relevant entity still, and snapshots the result as the scene's initial state so that
        ``env.reset()`` returns to it.

        Args:
            env (og.Environment): active environment.
            tro (None, str or TROFile): TRO to load; defaults to the constructor's ``tro_path``.
            settle_steps (int): physics steps used to settle before snapshotting.
            apply_object_states (bool): whether to overlay the stored object states / robot pose.
        """
        if tro is None:
            tro = self._tro if self._tro is not None else self._tro_path
        if tro is None:
            raise ValueError("No TRO given and no tro_path configured")
        if isinstance(tro, str):
            tro = load_tro(tro)
        assert og.sim.is_playing(), "load_tro_instance requires a playing simulator"

        validate_tro_against_problem(tro.meta, self.compiled_task, require_bindings=False)
        self._tro = tro
        self._verify_object_bindings(tro)
        robot = self.get_agent(env)

        if apply_object_states:
            for inst, state in tro.object_states.items():
                entity = self.object_scope.get(inst)
                if entity is None or isinstance(entity, GroupEntity):
                    raise ValueError(f"TRO stores a state for {inst!r}, which is not a bound physical instance")
                entity.load_state(recursively_convert_to_torch(copy.deepcopy(state)), serialized=False)
            if tro.robot_poses:
                poses = {k.lower(): v for k, v in tro.robot_poses.items()}
                if "robot" in poses:
                    available = poses["robot"]
                elif robot.model in poses:
                    available = poses[robot.model]
                else:
                    raise KeyError(f"No generic or model-specific presampled robot pose for {robot.model}")
                pose = available[0]
                robot.set_position_orientation(th.tensor(pose["position"]), th.tensor(pose["orientation"]))
                env.scene.write_task_metadata(key="robot_poses", data=tro.robot_poses)
            self._restore_robot_state(robot, tro.meta.robot_state)
            for key in ("grasps", "attachment_targets", "constraints"):
                if tro.meta.extras.get(key):
                    log.warning(f"TRO '{key}' records are not restored by ISpatialGymTask.load_tro_instance yet")

        self.configure_targets(tro.meta)

        og.sim.update_handles()
        for _ in range(settle_steps):
            og.sim.step_physics()
            for entity in self.physical_object_scope.values():
                if entity is not None:
                    entity.keep_still()
        env.scene.update_initial_file()
        env.scene.reset()
        self._reset_tracking()

    def _verify_object_bindings(self, tro):
        for inst, binding in tro.meta.object_bindings.items():
            entity = self.object_scope.get(inst)
            if entity is None:
                raise ValueError(f"TRO object_bindings names {inst!r}, which is not bound in the scope")
            expected = binding.get("scene_object_name") if isinstance(binding, dict) else binding
            if expected is not None and entity.name != expected:
                raise ValueError(f"object_bindings[{inst!r}] = {expected!r} but the scope holds {entity.name!r}")
            if isinstance(binding, dict):
                for key, attr in (("category", "category"), ("model", "model")):
                    if binding.get(key) is not None and getattr(entity, attr, None) != binding[key]:
                        raise ValueError(
                            f"object_bindings[{inst!r}].{key} = {binding[key]!r} but the scene object has "
                            f"{getattr(entity, attr, None)!r}"
                        )

    @staticmethod
    def _restore_robot_state(robot, robot_state):
        if not robot_state:
            return
        joint_positions = robot_state.get("joint_positions")
        if joint_positions:
            names = list(robot.joints.keys())
            unknown = sorted(set(joint_positions) - set(names))
            if unknown:
                raise ValueError(f"robot_state.joint_positions names unknown joints {unknown}")
            current = robot.get_joint_positions()
            target = current.clone()
            for i, name in enumerate(names):
                if name in joint_positions:
                    target[i] = float(joint_positions[name])
            robot.set_joint_positions(target)
            velocities = robot_state.get("joint_velocities")
            vel = th.zeros_like(target)
            if velocities:
                for i, name in enumerate(names):
                    if name in velocities:
                        vel[i] = float(velocities[name])
            robot.set_joint_velocities(vel)
        if robot_state.get("controller_targets"):
            log.info("robot_state.controller_targets: applied by the instance loader's hold (oracle.run_isg.InstanceLoader.restore), not here")

    # ------------------------------------------------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------------------------------------------------

    @property
    def group_entities(self) -> Dict[str, GroupEntity]:
        return {k: v for k, v in (self.object_scope or {}).items() if isinstance(v, GroupEntity)}

    @property
    def n_envs(self) -> int:
        return 1

    def scope_of(self, env_idx: int = 0) -> Dict[str, Any]:
        return self.object_scope

    def group_entities_of(self, env_idx: int = 0) -> Dict[str, GroupEntity]:
        return {k: v for k, v in (self.scope_of(env_idx) or {}).items() if isinstance(v, GroupEntity)}

    @property
    def problem_label(self) -> str:
        return f"{self.activity_name}-{self.activity_definition_id} ({self.task_variant})"

    @property
    def physical_object_scope(self) -> Dict[str, Any]:
        return {k: v for k, v in (self.object_scope or {}).items() if not isinstance(v, GroupEntity)}

    def evaluate_predicate(self, predicate_name, *entities):
        """``evaluate_fn`` of ``compiled_task.check_goal`` / ``check_initial_conditions`` against this task's scope."""
        return PRED.evaluate_predicate(predicate_name, *[self.object_scope[ent] for ent in entities])

    def _evaluate_predicate(self, env_idx, predicate_name, *entities):
        # BehaviorTask's per-env evaluator (potential reward, goal options); an ISpatialGymTask has one env
        return self.evaluate_predicate(predicate_name, *entities)

    @staticmethod
    def predicate_branches() -> Dict[str, str]:
        """``binary_branch``: which wrapper branch (omnigibson / rays / geometry / none) decided each under / ontop atom in
        the latest evaluations (``ispatialgym.sampling.predicates``; ``reset_predicate_branches`` clears the log)."""
        return PRED.decided_by()

    @staticmethod
    def reset_predicate_branches() -> None:
        PRED.reset_branches()

    def update_bddl_scope_metadata(self, env):
        env.scene.write_task_metadata(
            key="inst_to_name",
            data={inst: entity.name for inst, entity in self.physical_object_scope.items() if entity is not None},
        )

    def _update_bddl_scope_from_added_obj(self, obj):
        for inst, entity in self.physical_object_scope.items():
            if entity is None and self._object_bindings_source_name(inst) == obj.name:
                self.object_scope[inst] = obj
                return

    def _object_bindings_source_name(self, inst):
        bindings = self._config_object_bindings or (self._tro.meta.object_bindings if self._tro else {})
        name = bindings.get(inst)
        return name.get("scene_object_name") if isinstance(name, dict) else name

    def _update_bddl_scope_from_removed_obj(self, obj):
        for inst, entity in self.physical_object_scope.items():
            if entity is not None and obj.name == entity.name:
                self.object_scope[inst] = None
                return

    def _update_bddl_scope_from_system_init(self, system):
        return

    def _update_bddl_scope_from_system_clear(self, system):
        return

    def _get_obs(self, env, env_idx=0):
        # Same as BehaviorTask but restricted to physical entities (groups have no pose).
        group_names = set(self.group_entities)
        scope_backup = self.object_scope
        try:
            self.object_scope = {k: v for k, v in scope_backup.items() if k not in group_names}
            return super()._get_obs(env, env_idx)
        finally:
            self.object_scope = scope_backup

    def save_task(self, env, save_dir=None, override=False, task_relevant_only=False, suffix=None):
        group_names = set(self.group_entities)
        scope_backup = self.object_scope
        try:
            self.object_scope = {k: v for k, v in scope_backup.items() if k not in group_names}
            return super().save_task(env, save_dir=save_dir, override=override, task_relevant_only=task_relevant_only, suffix=suffix)
        finally:
            self.object_scope = scope_backup

    # ------------------------------------------------------------------------------------------------------------
    # Termination / two-level evaluation
    # ------------------------------------------------------------------------------------------------------------

    def _create_termination_conditions(self):
        terminations = dict()
        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        terminations["predicate"] = PredicateGoal(check_goal_fn=self._check_goal)
        return terminations

    def step(self, env, action):
        self._step_env = env  # the goal evaluation reads each env's episode step (the tracking advances once per step)
        if not SHARED_GOAL_EVAL:
            return super().step(env, action)
        self._shared_armed, self._shared_goal = True, None
        try:
            return super().step(env, action)
        finally:
            self._shared_armed, self._shared_goal = False, None

    def _check_goal(self, env_idx=0):
        if not (SHARED_GOAL_EVAL and getattr(self, "_shared_armed", False) and getattr(self, "_shared_goal", None) is None):
            return self._check_goal_impl(env_idx)
        PRED._RECORD, joints0 = [], PRED._JOINTS[0]
        try:
            out = self._check_goal_impl(env_idx)
        finally:
            entries, PRED._RECORD = PRED._RECORD, None
        self._shared_goal = {"status": out[1], "entries": entries, "joints": PRED._JOINTS[0] - joints0}
        return out

    def get_potential(self, env, env_idx):
        shared = getattr(self, "_shared_goal", None) if SHARED_GOAL_EVAL and getattr(self, "_shared_armed", False) else None
        if shared is None:
            return super().get_potential(env, env_idx)
        self._shared_goal = None
        _SHARED_COUNTS["steps"] += 1
        if shared["joints"]:
            _SHARED_COUNTS["joint_steps"] += 1
            return super().get_potential(env, env_idx)
        n_sat, n_unsat = len(shared["status"]["satisfied"]), len(shared["status"]["unsatisfied"])
        value = -(n_sat / (n_sat + n_unsat))
        if SHARED_GOAL_EVAL == "check":
            PRED._RECORD = []
            t0 = time.perf_counter()
            try:
                ref = super().get_potential(env, env_idx)
            finally:
                real, PRED._RECORD = PRED._RECORD, None
                _SHARED_COUNTS["potential_s"] += time.perf_counter() - t0
            if ref != value or json.dumps(real, sort_keys=True, default=str) != json.dumps(shared["entries"], sort_keys=True, default=str):
                _SHARED_COUNTS["mismatches"] += 1
                if _SHARED_COUNTS["mismatches"] <= 20:
                    print(f"SHARED_GOAL_CHECK mismatch: potential {ref!r} vs shared {value!r}; log {real!r} vs {shared['entries']!r}", flush=True)
            if _SHARED_COUNTS["steps"] % 500 == 0:
                print(f"SHARED_GOAL_CHECK {json.dumps(_SHARED_COUNTS)}", flush=True)
            return ref
        PRED.replay_log(shared["entries"])
        _SHARED_COUNTS["reused"] += 1
        return value

    def _check_goal_impl(self, env_idx=0):
        """Evaluate all goal heads and the error reports of env ``env_idx`` once; advance its tracking (once per episode step)
        and decide termination.

        Returns ``(done, goal_status)`` like ``CompiledTask.check_goal``; ``done`` is the run mode's geometric success.
        """
        _, status = self.compiled_task.check_goal(lambda name, *ents: self._evaluate_predicate(env_idx, name, *ents))
        satisfied = set(status["satisfied"])
        goal = set(self._other_goal_head_idxs).issubset(satisfied)
        settled = set(self._pose_settled_head_idxs).issubset(satisfied)
        errors = self.collect_errors(env_idx)
        within = MET.measure_within(errors, self._measure_atoms)
        tracker = self._trackers[env_idx]
        env = getattr(self, "_step_env", None)
        if env is not None:
            t = int(env.episode_steps[env_idx])
            if tracker.last is None or tracker.last[0] != t:
                tracker.update(t, goal, within, settled, errors)
        self._last_goal_evals[env_idx] = (goal, within, settled, status)
        self._last_errors[env_idx] = errors
        return tracker.done, status

    def settle_steps_of(self, env_idx: int = 0) -> int:
        """Env ``env_idx``'s ``pose_settled`` / settle-eval window (:meth:`configure_evaluation`)."""
        return int(self._settle_steps_envs[env_idx])

    def pose_settled_config_of(self, env_idx: int = 0) -> Dict[str, Any]:
        return {**self.pose_settled_config, "n_steps": self.settle_steps_of(env_idx)}

    def _reset_tracking(self, env_indices=None):
        n = self.n_envs
        if len(self._settle_steps_envs) != n:
            self._settle_steps_envs = [self.settle_steps] * n
        if env_indices is None or len(getattr(self, "_trackers", ())) != n:
            self._trackers = [None] * n
            self._last_goal_evals = [None] * n
            self._last_errors = [None] * n
            self._pose_settled_other_success = [{} for _ in range(n)]
            self._t_other_success = [-1] * n
            env_indices = range(n)
        for i in env_indices:
            i = int(i)
            self._trackers[i] = MET.SuccessTracker(self.settle_eval, self.settle_steps_of(i))
            self._last_goal_evals[i] = None
            self._last_errors[i] = None

    def _reset_variables(self, env, env_indices):
        super()._reset_variables(env, env_indices)
        self._reset_tracking(env_indices)

    @property
    def metrics_state(self) -> Dict[str, Any]:
        """The tracking a checkpoint rollback restores (a copy; the executor's ``TASK_TRACKING_ATTRS``)."""
        return {"trackers": [t.state() for t in self._trackers], "last_goal_evals": list(self._last_goal_evals), "last_errors": list(self._last_errors)}

    @metrics_state.setter
    def metrics_state(self, state: Dict[str, Any]) -> None:
        for tracker, s in zip(self._trackers, state["trackers"]):
            tracker.load(s)
        self._last_goal_evals = list(state["last_goal_evals"])
        self._last_errors = list(state["last_errors"])

    def configure_evaluation(self, settle_eval: bool, settle_steps: int, env_indices=None) -> None:
        """The run's settle-eval mode and the window of envs ``env_indices`` (default all; ``run_isg`` / the vectorized loader,
        per instance, before its reset): their ``PoseSettled`` windows get ``n_steps = settle_steps`` and their tracking restarts."""
        if int(settle_steps) < 1:
            raise ValueError(f"settle_steps must be >= 1, got {settle_steps}")
        self.settle_eval = bool(settle_eval)
        if env_indices is None:
            self.settle_steps = int(settle_steps)
            self.pose_settled_config = {**self.pose_settled_config, "n_steps": self.settle_steps}
            self._settle_steps_envs = [self.settle_steps] * self.n_envs
            env_indices = range(self.n_envs)
        else:
            env_indices = [int(i) for i in env_indices]
            if len(self._settle_steps_envs) != self.n_envs:
                self._settle_steps_envs = [self.settle_steps] * self.n_envs
            for i in env_indices:
                self._settle_steps_envs[i] = int(settle_steps)
        self._reset_tracking(env_indices)
        if self.object_scopes is not None:
            self._reset_pose_settled_tracking(env_indices)

    def apply_tolerance_profile(self, mode: str, scale: Optional[float] = None, customs_path: Optional[str] = None) -> Dict[str, Any]:
        """The ``isg-v2`` tolerances of the loaded TRO from the current state (after the loader's settle, before the first step),
        written into the TRO in memory with the targets re-configured; returns the record (kept in :attr:`tolerance_profiles`)."""
        self.tolerance_profiles[0] = TP.apply_to_task(self, mode, scale=scale, customs_path=customs_path)
        return self.tolerance_profiles[0]

    def reset(self, env, env_indices=None):
        super().reset(env, env_indices=env_indices)
        for entity in self.group_entities.values():
            entity.invalidate()

    def _reset_pose_settled_tracking(self, env_indices=None):
        self._pose_settled_other_success = [{}]
        self._t_other_success = [-1]
        for inst in self._pose_settled_insts:
            entity = self.object_scope.get(inst)
            if entity is not None and not isinstance(entity, GroupEntity) and PoseSettled in entity.states:
                entity.states[PoseSettled].activate(**self.pose_settled_config_of(0))

    def success_of(self, env_idx: int = 0) -> bool:
        """The run mode's geometric success reached in env ``env_idx``."""
        return self._trackers[env_idx].done

    @property
    def official_success(self) -> bool:
        return self.success_of(0)

    def _step_termination(self, env, action, infos=None):
        dones, infos = BaseTask._step_termination(self, env=env, action=action, infos=infos)
        goal_status = self._termination_conditions["predicate"].goal_status
        for env_idx in range(self.n_envs):
            info = infos[env_idx]
            info["goal_status"] = goal_status[env_idx]
            success = self.success_of(env_idx)
            info["success"] = success
            info["termination_conditions"]["predicate"]["success"] = success
            self._populate_isg_info(env, info, env_idx)
            if self._pose_settled_insts:
                self._populate_pose_settled_info(info, env_idx)
        return dones, infos

    # ------------------------------------------------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------------------------------------------------

    def _is_configured(self, inst, state_cls, env_idx: int = 0):
        entity = (self.scope_of(env_idx) or {}).get(inst)
        if entity is None:
            return False
        if isinstance(entity, GroupEntity):
            if entity.target is None:
                return False
            return entity.target.supports_pose_reached if state_cls is PoseReached else entity.target.supports_layout_reached
        return state_cls in entity.states and entity.states[state_cls].has_goal

    @property
    def measured_pose_instances(self):
        """Entities whose pose error is reported: ``pose_reached`` arguments of the goal (gating) and of
        ``(:measure ...)`` (non-gating), once their TRO target has been configured."""
        if self.compiled_task is None:
            return []
        return [inst for inst in self.compiled_task.all_pose_reached_args if self._is_configured(inst, PoseReached)]

    @property
    def measured_layout_instances(self):
        if self.compiled_task is None:
            return []
        return [inst for inst in self.compiled_task.all_layout_reached_args if self._is_configured(inst, LayoutReached)]

    def collect_errors(self, env_idx: int = 0) -> Dict[str, Any]:
        """Current error reports of every configured pose / group / articulation target of env ``env_idx`` (the ``pose_reached``
        / ``layout_reached`` arguments of ``:goal`` and ``:measure``): ``{"pose_errors": {inst: report}, "layout_errors": {...}}``,
        each report with its ``tolerances`` (``tolerance_profile.target_tolerances``), ``in_goal`` and ``in_measure``."""
        out: Dict[str, Any] = {"pose_errors": {}, "layout_errors": {}}
        if self.compiled_task is None:
            return out
        scope, meta = self.scope_of(env_idx), self._target_metas.get(env_idx)
        for predicate, state_cls, args, key in (("pose_reached", PoseReached, self.compiled_task.all_pose_reached_args, "pose_errors"),
                                                ("layout_reached", LayoutReached, self.compiled_task.all_layout_reached_args, "layout_errors")):
            for inst in args:
                if not self._is_configured(inst, state_cls, env_idx):
                    continue
                report = scope[inst].states[state_cls].get_error_report()
                if report is None:
                    continue
                report.update({"tolerances": TP.target_tolerances(meta, inst, predicate), "in_goal": bool(self.compiled_task.is_gating(predicate, inst)),
                               "in_measure": bool(self.compiled_task.in_measure(predicate, inst))})
                out[key][inst] = report
        return out

    # ------------------------------------------------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------------------------------------------------

    def goal_conditions_report(self, status):
        satisfied = set(status["satisfied"])
        settle = set(self._pose_settled_head_idxs)
        return [
            {
                "index": i,
                "bddl": _compact(self.problem.goal[i]),
                "kind": "pose_settled" if i in settle else "binary",
                "satisfied": i in satisfied,
            }
            for i in range(len(self.problem.goal))
        ]

    def measured_conditions_report(self, errors):
        """The ``(:measure ...)`` atoms with their within-tolerance status in ``errors`` (``in_goal``: the atom is in ``:goal`` too)."""
        out = []
        for lit in self.problem.measure:
            pred, arg = lit[0], lit[1].lstrip("?")
            reports = (errors.get("pose_errors") if pred == "pose_reached" else errors.get("layout_errors")) or {}
            report = reports.get(arg)
            out.append(
                {
                    "bddl": _compact(lit),
                    "kind": "measure",
                    "in_goal": bool(self.compiled_task.is_gating(pred, arg)),
                    "configured": report is not None,
                    "within_tolerance": None if report is None else report.get("within_tolerance"),
                }
            )
        return out

    def _populate_isg_info(self, env, info, env_idx: int = 0):
        """``info["ispatialgym"]`` of env ``env_idx``: this step's facts (``goal`` = G, ``measure_within`` = M, ``settled`` = S, its
        ``errors`` / ``error_metric`` / ``measured``), the run mode's symbolic and geometric success (steps: 0-based episode steps,
        None while not reached), ``evaluation`` (the errors the run mode is judged by: at its geometric success, else this step's),
        ``other_mode`` and the goal conditions."""
        goal, within, settled, status = self._last_goal_evals[env_idx] or (False, False, False, info["goal_status"])
        errors = self._last_errors[env_idx]
        if errors is None:
            errors = self.collect_errors(env_idx)
        where = self.problem_label
        ev = self._trackers[env_idx].report(errors, self._measure_atoms, where)
        groups = self.group_entities_of(env_idx)
        info["ispatialgym"] = {
            "activity": self.activity_name,
            "activity_definition_id": self.activity_definition_id,
            "task_variant": self.task_variant,
            "env_idx": env_idx,
            "step": int(env.episode_steps[env_idx]),
            "settle_eval": ev["settle_eval"],
            "settle_steps": ev["settle_steps"],
            "pose_settled_config": self.pose_settled_config_of(env_idx),
            "goal": bool(goal),
            "measure_within": bool(within),
            "settled": bool(settled),
            "error_metric": MET.error_metric(errors, self._measure_atoms, where),
            "official_success": ev["geometric_success"],
            "symbolic_success": ev["symbolic_success"],
            "symbolic_success_step": ev["symbolic_success_step"],
            "geometric_success": ev["geometric_success"],
            "geometric_success_step": ev["geometric_success_step"],
            "goal_first_step": ev["goal_first_step"],
            "other_mode": ev["other_mode"],
            "evaluation": {"errors": ev["errors"], "error_metric": ev["error_metric"], "measured": self.measured_conditions_report(ev["errors"])},
            "conditions": self.goal_conditions_report(status),
            "measured": self.measured_conditions_report(errors),
            "errors": errors,
            "group_assignments": {
                name: (entity.evaluate().assignment if entity.evaluate() is not None else None)
                for name, entity in groups.items()
            },
            "tolerance_profile": (self.tolerance_profiles.get(env_idx) or {}).get("name"),
        }

    # ------------------------------------------------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------------------------------------------------

    @property
    def name(self):
        variant = f"_{self.task_variant}" if self.task_variant else ""
        return f"{self.__class__.__name__}_{self.activity_name}_{self.activity_definition_id}{variant}_{self.activity_instance_id}"

    @classproperty
    def default_termination_config(cls):
        return {"max_steps": 500}

    @classproperty
    def default_reward_config(cls):
        return {"r_potential": 1.0}
