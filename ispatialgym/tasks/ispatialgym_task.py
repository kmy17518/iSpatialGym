"""``ISpatialGymTask``: a BEHAVIOR task that scores iSpatialGym problems (:mod:`ispatialgym.eval.metrics`).

Every step the task evaluates the goal (``G``: all atoms but ``pose_settled``), the measured targets (``M``) and the
settle atoms (``S``), and advances a :class:`~ispatialgym.eval.metrics.SuccessTracker`. The run's geometric success is
``info["success"]`` and ends the episode. ``pose_reached`` / ``layout_reached`` are configured from the instance's TRO
with the ``isg-v2`` tolerances (:meth:`ISpatialGymTask.apply_tolerance_profile`). Everything is reported in the
``ispatialgym`` block of the step info (:meth:`ISpatialGymTask._populate_isg_info`).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import omnigibson as og
import torch as th
from omnigibson.macros import gm
from omnigibson.object_states import LayoutReached, PoseReached, PoseSettled
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.tasks.task_base import BaseTask
from omnigibson.termination_conditions.predicate_goal import PredicateGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.python_utils import classproperty, recursively_convert_to_torch

from ispatialgym.activity.problem import (
    ACTIVITY_DEFINITIONS_ROOT,
    AGENT_INSTANCE,
    ISGProblem,
    is_agent_instance,
    load_problem,
    to_bddl,
)
from ispatialgym.eval import metrics as MET
from ispatialgym.eval import predicates as PRED
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

#: ``PoseSettled`` window of every settled entity: ``n_steps`` is the run's settle window; an entity is settled when its
#: drift over the window stays within 2 mm / 5 mrad (joints 5 mrad, or 5 mm for prismatic joints)
DEFAULT_POSE_SETTLED_CONFIG: dict[str, float] = {
    "n_steps": 45,
    "pos_tolerance": 0.002,
    "ori_tolerance": 0.005,
    "joint_tolerance": 0.005,
}

#: assisted grasps of a link behind a movable joint (a lid, a door leaf) use a fixed joint instead of OmniGibson's
#: spherical one, which couples position only
AG_MOVING_LINK_JOINT_TYPE = "FixedJoint"
#: after the open command the assisted-grasp weld holds until a pad has left the object, at most this long
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


class ISpatialGymTask(BehaviorTask):
    """
    Args:
        activity_name (str): activity (directory under ``activity_definitions_root``).
        activity_definition_id (int): problem number (``problemN.bddl``).
        activity_instance_id (int): instance id (bookkeeping only).
        activity_definitions_root (None or str): override of the activity definitions directory.
        tro_path (None or str): TRO file whose ``ispatialgym`` block configures the evaluation targets.
        object_bindings (None or dict): ``{bddl_instance: scene_object_name}``; default the TRO's ``object_bindings``,
            else the scene's ``inst_to_name`` metadata.
        settle_eval (bool): the run's settle-eval mode (:mod:`ispatialgym.eval.metrics`).
        settle_steps (int): the ``pose_settled`` and settle-eval window, in steps.
        pose_settled_config (None or dict): ``PoseSettled.activate`` kwargs (default :data:`DEFAULT_POSE_SETTLED_CONFIG`).
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

        # BehaviorTask attributes (BehaviorTask.__init__ requires a knowledge-base activity and is bypassed)
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
        self._pose_settled_other_success = [{}]
        self._t_other_success = [-1]
        self.instruction_order = None
        self.currently_viewed_index = None
        self.currently_viewed_instruction = None
        self.activity_natural_language_initial_conditions = None
        self.activity_natural_language_goal_conditions = None

        self.activity_definitions_root = activity_definitions_root or ACTIVITY_DEFINITIONS_ROOT
        self.problem: ISGProblem | None = None
        self.task_variant = None
        self.groups: dict[str, list] = {}
        self.settle_eval = bool(settle_eval)
        self.settle_steps = int(settle_steps)
        base_config = DEFAULT_POSE_SETTLED_CONFIG if pose_settled_config is None else pose_settled_config
        self.pose_settled_config = {**base_config, "n_steps": self.settle_steps}
        self._config_object_bindings = dict(object_bindings) if object_bindings else None
        self._tro: TROFile | None = None
        self._tro_path = tro_path
        self._target_meta = None
        self._measure_atoms: list = []
        self._pose_settled_head_idxs = []
        self._step_env = None
        self.tolerance_profile: dict[str, Any] = {}
        self._reset_tracking()

        BaseTask.__init__(
            self, termination_config=termination_config, reward_config=reward_config, include_obs=include_obs
        )

    # ------------------------------------------------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------------------------------------------------

    @classmethod
    def verify_scene_and_task_config(cls, scene_cfg, task_cfg):
        BaseTask.verify_scene_and_task_config(scene_cfg=scene_cfg, task_cfg=task_cfg)

    @property
    def object_scopes(self):
        """The per-env scopes ``BehaviorTask`` reads: an ISpatialGymTask binds one environment."""
        return None if self.object_scope is None else [self.object_scope]

    @object_scopes.setter
    def object_scopes(self, scopes):
        self.object_scope = None if scopes is None else scopes[0]

    @property
    def tro(self) -> TROFile | None:
        return self._tro

    def _load(self, env):
        assert env.num_envs == 1, f"ISpatialGymTask binds one environment, got num_envs={env.num_envs}"
        self.update_activity(
            env=env, activity_name=self.activity_name, activity_definition_id=self.activity_definition_id
        )
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
        self.activity_name = activity_name
        self.activity_definition_id = int(activity_definition_id)
        self.problem = load_problem(activity_name, activity_definition_id, root=self.activity_definitions_root)
        self.compiled_task = self.problem.compile()
        self.task_variant = self.problem.variant
        self.groups = {g: list(members) for g, members in self.problem.groups.items()}

        # agent first, then the declared instances, then the nonphysical group entities
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
        MET.error_metric_case(self._measure_atoms, self.problem_label)  # fail at load, not at the end of an episode

        n_goals = len(self.compiled_task.conditions.parsed_goal_conditions)
        self.instruction_order = th.arange(n_goals)[th.randperm(n_goals)]
        self.currently_viewed_index = 0
        self.currently_viewed_instruction = self.instruction_order[self.currently_viewed_index]
        self.activity_natural_language_initial_conditions = self.compiled_task.natural_language_initial_conditions
        self.activity_natural_language_goal_conditions = self.compiled_task.natural_language_goal_conditions

    def _resolve_member(self, inst):
        return self.object_scope.get(inst) if self.object_scope else None

    def initialize_activity(self, env):
        """Bind every declared instance to its scene object (no sampling) and configure the TRO's targets."""
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
        for robot in _robots(getattr(env, "robots", None)):
            robot.ag_moving_link_joint_type = AG_MOVING_LINK_JOINT_TYPE
            robot.ag_release_on_pad_clear = AG_RELEASE_ON_PAD_CLEAR
            robot.ag_release_max_delay_s = AG_RELEASE_MAX_DELAY_S
        if self._tro is not None:
            self.configure_targets(self._tro.meta)
        return True, None

    def _object_bindings_source(self, env) -> dict[str, Any]:
        if self._config_object_bindings:
            return self._config_object_bindings
        if self._tro is not None and self._tro.meta.object_bindings:
            return self._tro.meta.object_bindings
        try:
            return env.scene.get_task_metadata(key="inst_to_name") or {}
        except Exception:  # noqa: BLE001 - a scene without task metadata
            return {}

    def configure_targets(self, meta):
        """Push the TRO's pose / group / articulation targets into the bound scope entities."""
        apply_targets_to_scope(meta, self.object_scope)
        self._target_meta = meta
        for entity in self.group_entities.values():
            entity.invalidate()

    def load_tro_instance(self, env, tro=None, settle_steps=25):
        """Apply a TRO's object states, robot pose and robot joints, configure its targets, settle with every task
        entity held still, and snapshot the result as the scene's initial state (the simulator must be playing)."""
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

        for inst, state in tro.object_states.items():
            entity = self.object_scope.get(inst)
            if entity is None or isinstance(entity, GroupEntity):
                raise ValueError(f"TRO stores a state for {inst!r}, which is not a bound physical instance")
            entity.load_state(recursively_convert_to_torch(copy.deepcopy(state)), serialized=False)
        if tro.robot_poses:
            poses = {k.lower(): v for k, v in tro.robot_poses.items()}
            available = poses.get("robot") or poses.get(robot.model)
            if available is None:
                raise KeyError(f"No generic or model-specific robot pose for {robot.model}")
            robot.set_position_orientation(th.tensor(available[0]["position"]), th.tensor(available[0]["orientation"]))
            env.scene.write_task_metadata(key="robot_poses", data=tro.robot_poses)
        self._restore_robot_state(robot, tro.meta.robot_state)
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
                for key in ("category", "model"):
                    if binding.get(key) is not None and getattr(entity, key, None) != binding[key]:
                        raise ValueError(
                            f"object_bindings[{inst!r}].{key} = {binding[key]!r} but the scene object has {getattr(entity, key, None)!r}"
                        )

    @staticmethod
    def _restore_robot_state(robot, robot_state):
        joint_positions = (robot_state or {}).get("joint_positions")
        if not joint_positions:
            return
        names = list(robot.joints.keys())
        unknown = sorted(set(joint_positions) - set(names))
        if unknown:
            raise ValueError(f"robot_state.joint_positions names unknown joints {unknown}")
        target = robot.get_joint_positions().clone()
        velocities = robot_state.get("joint_velocities") or {}
        vel = th.zeros_like(target)
        for i, name in enumerate(names):
            if name in joint_positions:
                target[i] = float(joint_positions[name])
            if name in velocities:
                vel[i] = float(velocities[name])
        robot.set_joint_positions(target)
        robot.set_joint_velocities(vel)

    # ------------------------------------------------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------------------------------------------------

    @property
    def group_entities(self) -> dict[str, GroupEntity]:
        return {k: v for k, v in (self.object_scope or {}).items() if isinstance(v, GroupEntity)}

    @property
    def physical_object_scope(self) -> dict[str, Any]:
        return {k: v for k, v in (self.object_scope or {}).items() if not isinstance(v, GroupEntity)}

    @property
    def problem_label(self) -> str:
        return f"{self.activity_name}-{self.activity_definition_id} ({self.task_variant})"

    def evaluate_predicate(self, predicate_name, *entities):
        """``evaluate_fn`` of ``compiled_task.check_goal`` against this task's scope."""
        return PRED.evaluate_predicate(predicate_name, *[self.object_scope[ent] for ent in entities])

    def _evaluate_predicate(self, env_idx, predicate_name, *entities):
        return self.evaluate_predicate(predicate_name, *entities)

    def update_bddl_scope_metadata(self, env, env_idx=0):
        env.scene.write_task_metadata(
            key="inst_to_name",
            data={inst: entity.name for inst, entity in self.physical_object_scope.items() if entity is not None},
        )

    def _update_bddl_scope_from_added_obj(self, obj):
        bindings = self._config_object_bindings or (self._tro.meta.object_bindings if self._tro else {})
        for inst, entity in self.physical_object_scope.items():
            name = bindings.get(inst)
            name = name.get("scene_object_name") if isinstance(name, dict) else name
            if entity is None and name == obj.name:
                self.object_scope[inst] = obj
                return

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
        scope = self.object_scope
        try:
            self.object_scope = self.physical_object_scope  # group entities have no pose
            return super()._get_obs(env, env_idx)
        finally:
            self.object_scope = scope

    def save_task(self, env, save_dir=None, override=False, task_relevant_only=False, suffix=None, env_idx=0):
        scope = self.object_scope
        try:
            self.object_scope = self.physical_object_scope
            return super().save_task(
                env,
                save_dir=save_dir,
                override=override,
                task_relevant_only=task_relevant_only,
                suffix=suffix,
                env_idx=env_idx,
            )
        finally:
            self.object_scope = scope

    # ------------------------------------------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------------------------------------------

    def _create_termination_conditions(self):
        return {
            "timeout": Timeout(max_steps=self._termination_config["max_steps"]),
            "predicate": PredicateGoal(check_goal_fn=self._check_goal),
        }

    def step(self, env, action):
        self._step_env = env  # the goal evaluation reads the episode step
        return super().step(env, action)

    def _check_goal(self, env_idx=0):
        """Evaluate the goal heads and the error reports once, advance the tracker (once per episode step) and return
        ``(done, goal_status)`` like ``CompiledTask.check_goal``, ``done`` being the run's geometric success."""
        _, status = self.compiled_task.check_goal(self.evaluate_predicate)
        satisfied = set(status["satisfied"])
        goal = set(self._other_goal_head_idxs).issubset(satisfied)
        settled = set(self._pose_settled_head_idxs).issubset(satisfied)
        errors = self.collect_errors()
        within = MET.measure_within(errors, self._measure_atoms)
        if self._step_env is not None:
            t = int(self._step_env.episode_steps[env_idx])
            if self._tracker.last is None or self._tracker.last[0] != t:
                self._tracker.update(t, goal, within, settled, errors)
        self._last_goal_eval = (goal, within, settled, status)
        self._last_errors = errors
        return self._tracker.done, status

    def _reset_tracking(self):
        self._tracker = MET.SuccessTracker(self.settle_eval, self.settle_steps)
        self._last_goal_eval = None
        self._last_errors = None

    def _reset_variables(self, env, env_indices):
        super()._reset_variables(env, env_indices)
        self._reset_tracking()

    def configure_evaluation(self, settle_eval: bool, settle_steps: int) -> None:
        """Set the run's settle-eval mode and window (before the instance's reset activates the settle windows)."""
        if int(settle_steps) < 1:
            raise ValueError(f"settle_steps must be >= 1, got {settle_steps}")
        self.settle_eval = bool(settle_eval)
        self.settle_steps = int(settle_steps)
        self.pose_settled_config = {**self.pose_settled_config, "n_steps": self.settle_steps}
        self._reset_tracking()
        if self.object_scope is not None:
            self._reset_pose_settled_tracking()

    def apply_tolerance_profile(self, scale: float = TP.DEFAULT_POLICY_SCALE) -> dict[str, Any]:
        """Apply the ``isg-v2`` tolerances to the loaded TRO from the current state; returns the record."""
        self.tolerance_profile = TP.apply_to_task(self, scale=scale)
        return self.tolerance_profile

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
                entity.states[PoseSettled].activate(**self.pose_settled_config)

    def _step_termination(self, env, action, infos=None):
        dones, infos = BaseTask._step_termination(self, env=env, action=action, infos=infos)
        goal_status = self._termination_conditions["predicate"].goal_status
        for env_idx, info in enumerate(infos):
            info["goal_status"] = goal_status[env_idx]
            info["success"] = self._tracker.done
            info["termination_conditions"]["predicate"]["success"] = self._tracker.done
            self._populate_isg_info(env, info, env_idx)
            if self._pose_settled_insts:
                self._populate_pose_settled_info(info, env_idx)
        return dones, infos

    def _is_configured(self, inst, state_cls) -> bool:
        entity = (self.object_scope or {}).get(inst)
        if entity is None:
            return False
        if isinstance(entity, GroupEntity):
            if entity.target is None:
                return False
            return (
                entity.target.supports_pose_reached
                if state_cls is PoseReached
                else entity.target.supports_layout_reached
            )
        return state_cls in entity.states and entity.states[state_cls].has_goal

    def collect_errors(self) -> dict[str, Any]:
        """Error reports of every configured pose / group / articulation target: ``{"pose_errors": {inst: report},
        "layout_errors": {...}}``, each with its ``tolerances``, ``in_goal`` and ``in_measure``."""
        out: dict[str, Any] = {"pose_errors": {}, "layout_errors": {}}
        if self.compiled_task is None:
            return out
        for predicate, state_cls, args, key in (
            ("pose_reached", PoseReached, self.compiled_task.all_pose_reached_args, "pose_errors"),
            ("layout_reached", LayoutReached, self.compiled_task.all_layout_reached_args, "layout_errors"),
        ):
            for inst in args:
                if not self._is_configured(inst, state_cls):
                    continue
                report = self.object_scope[inst].states[state_cls].get_error_report()
                if report is None:
                    continue
                report.update(
                    {
                        "tolerances": TP.target_tolerances(self._target_meta, inst, predicate),
                        "in_goal": bool(self.compiled_task.is_gating(predicate, inst)),
                        "in_measure": bool(self.compiled_task.in_measure(predicate, inst)),
                    }
                )
                out[key][inst] = report
        return out

    def goal_conditions_report(self, status):
        satisfied, settle = set(status["satisfied"]), set(self._pose_settled_head_idxs)
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
        """The ``(:measure ...)`` atoms with their within-tolerance status in ``errors``."""
        out = []
        for lit in self.problem.measure:
            pred, arg = lit[0], lit[1].lstrip("?")
            report = ((errors.get("pose_errors") if pred == "pose_reached" else errors.get("layout_errors")) or {}).get(
                arg
            )
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

    def _populate_isg_info(self, env, info, env_idx=0):
        """``info["ispatialgym"]``: this step's facts (``goal``, ``measure_within``, ``settled`` and the errors), the run's
        symbolic / geometric success (0-based steps, None until reached), ``evaluation`` (the errors the run is judged
        by: at its geometric success, else its symbolic success, else this step's), ``other_mode`` and the goal conditions."""
        goal, within, settled, status = self._last_goal_eval or (False, False, False, info["goal_status"])
        errors = self._last_errors if self._last_errors is not None else self.collect_errors()
        ev = self._tracker.report(errors, self._measure_atoms, self.problem_label)
        group_assignments = {}
        for name, entity in self.group_entities.items():
            evaluation = entity.evaluate()
            group_assignments[name] = None if evaluation is None else evaluation.assignment
        info["ispatialgym"] = {
            "activity": self.activity_name,
            "activity_definition_id": self.activity_definition_id,
            "task_variant": self.task_variant,
            "step": int(env.episode_steps[env_idx]),
            "settle_eval": ev["settle_eval"],
            "settle_steps": ev["settle_steps"],
            "pose_settled_config": dict(self.pose_settled_config),
            "goal": bool(goal),
            "measure_within": bool(within),
            "settled": bool(settled),
            "error_metric": MET.error_metric(errors, self._measure_atoms, self.problem_label),
            "official_success": ev["geometric_success"],
            "symbolic_success": ev["symbolic_success"],
            "symbolic_success_step": ev["symbolic_success_step"],
            "geometric_success": ev["geometric_success"],
            "geometric_success_step": ev["geometric_success_step"],
            "goal_first_step": ev["goal_first_step"],
            "other_mode": ev["other_mode"],
            "evaluation": {
                "errors": ev["errors"],
                "error_metric": ev["error_metric"],
                "measured": self.measured_conditions_report(ev["errors"]),
            },
            "conditions": self.goal_conditions_report(status),
            "measured": self.measured_conditions_report(errors),
            "errors": errors,
            "group_assignments": group_assignments,
            "tolerance_profile": self.tolerance_profile.get("name"),
        }

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
