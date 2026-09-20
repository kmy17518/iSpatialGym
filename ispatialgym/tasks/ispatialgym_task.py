"""``ISpatialGymTask``: BEHAVIOR-compatible task that evaluates iSpatialGym problems at two levels.

Level 1 -- **binary success** ``G(t)``: every goal atom except ``pose_settled`` holds (``pose_reached`` /
``layout_reached`` are configured from the TRO and may compare a full pose, only the depth along the head-camera
axis, only the orientation, joint positions, or relative poses between group members).

Level 2 -- once ``G`` has held for the first time (``first_success_step``), the ``pose_settled`` atoms are tracked:
``settle_latency_steps = first_settled_success_step - first_success_step`` where the settled-success step is the
first ``t >= first_success_step`` with ``G(t) and S(t)``. Position / orientation errors of every configured pose,
group and articulation target are reported per step; the representative values (``final_errors``) are the errors
at the first binary success, or at the current (terminal) step if ``G`` was never reached.

Everything is surfaced in ``info["ispatialgym"]`` of ``env.step``; see :meth:`ISpatialGymTask._populate_isg_info`.
"""

from __future__ import annotations

import copy
import logging
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
from ispatialgym.eval.group_entity import GroupEntity
from ispatialgym.eval.tro import (
    TROFile,
    apply_targets_to_scope,
    load_tro,
    physical_instances,
    validate_tro_against_problem,
)

log = logging.getLogger(__name__)


def _compact(expr) -> str:
    return " ".join(to_bddl(expr).split())


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
        require_pose_settled (bool): if True, official success (``task.success``) additionally requires every
            ``pose_settled`` goal atom to hold together with the binary goals; otherwise official success is the
            binary success (reached at any step of the episode).
        terminate_on_binary_success (bool): end the episode at the first binary success instead of waiting for
            the settled success (settle latency is then not measurable).
        pose_settled_config (None or dict): kwargs forwarded to ``PoseSettled.activate`` for every settled entity.
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
        require_pose_settled=False,
        terminate_on_binary_success=False,
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
        self._pose_settled_other_success = {}
        self._t_other_success = -1
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
        self.require_pose_settled = bool(require_pose_settled)
        self.terminate_on_binary_success = bool(terminate_on_binary_success)
        self.pose_settled_config = dict(pose_settled_config or {})
        self._config_object_bindings = dict(object_bindings) if object_bindings else None
        self._tro: Optional[TROFile] = None
        self._tro_path = tro_path
        self._targets_configured = False
        self._pose_settled_head_idxs = []
        self._reset_tracking()

        BaseTask.__init__(self, termination_config=termination_config, reward_config=reward_config, include_obs=include_obs)

    # ------------------------------------------------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------------------------------------------------

    @classmethod
    def verify_scene_and_task_config(cls, scene_cfg, task_cfg):
        # No scene-instance rewriting: iSpatialGym instances name their scene explicitly (TRO ``scene``).
        BaseTask.verify_scene_and_task_config(scene_cfg=scene_cfg, task_cfg=task_cfg)

    def _load(self, env):
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
        self.task_variant = self.problem.alias
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
            log.warning("robot_state.controller_targets are not restored yet")

    # ------------------------------------------------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------------------------------------------------

    @property
    def group_entities(self) -> Dict[str, GroupEntity]:
        return {k: v for k, v in (self.object_scope or {}).items() if isinstance(v, GroupEntity)}

    @property
    def physical_object_scope(self) -> Dict[str, Any]:
        return {k: v for k, v in (self.object_scope or {}).items() if not isinstance(v, GroupEntity)}

    def _evaluate_predicate(self, predicate_name, *entities):
        return PRED.evaluate_predicate(predicate_name, *[self.object_scope[ent] for ent in entities])

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

    def _get_obs(self, env):
        # Same as BehaviorTask but restricted to physical entities (groups have no pose).
        group_names = set(self.group_entities)
        scope_backup = self.object_scope
        try:
            self.object_scope = {k: v for k, v in scope_backup.items() if k not in group_names}
            return super()._get_obs(env)
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

    def _check_goal(self):
        """Evaluate all goal heads once; decide termination.

        Returns ``(done, goal_status)`` like ``CompiledTask.check_goal``. ``done`` is the first settled success
        (``G and S``) when the goal has ``pose_settled`` atoms and ``terminate_on_binary_success`` is False,
        otherwise the first binary success.
        """
        _, status = self.compiled_task.check_goal(self._evaluate_predicate)
        satisfied = set(status["satisfied"])
        binary = set(self._other_goal_head_idxs).issubset(satisfied)
        settled = set(self._pose_settled_head_idxs).issubset(satisfied)
        self._last_goal_eval = (binary, settled, status)
        if not self._pose_settled_head_idxs or self.terminate_on_binary_success:
            return binary, status
        return (binary and settled), status

    def _reset_tracking(self):
        self._last_goal_eval = None
        self._first_success_step = -1
        self._first_settled_success_step = -1
        self._errors_at_first_success = None
        self._settled_history = {}
        self._pose_settled_other_success = {}
        self._t_other_success = -1

    def _reset_variables(self, env):
        super()._reset_variables(env)
        self._reset_tracking()

    def reset(self, env):
        super().reset(env)
        for entity in self.group_entities.values():
            entity.invalidate()

    def _reset_pose_settled_tracking(self):
        self._pose_settled_other_success = {}
        self._t_other_success = -1
        for inst in self._pose_settled_insts:
            entity = self.object_scope.get(inst)
            if entity is not None and not isinstance(entity, GroupEntity) and PoseSettled in entity.states:
                entity.states[PoseSettled].activate(**self.pose_settled_config)

    @property
    def official_success(self) -> bool:
        """Binary success reached at some step (or settled success when ``require_pose_settled``)."""
        if self.require_pose_settled:
            return self._first_settled_success_step >= 0
        return self._first_success_step >= 0

    def _step_termination(self, env, action, info=None):
        done, info = BaseTask._step_termination(self, env=env, action=action, info=info)
        info["goal_status"] = self._termination_conditions["predicate"].goal_status
        self._update_success_tracking(env)
        info["success"] = self.official_success
        info["termination_conditions"]["predicate"]["success"] = self.official_success
        self._populate_isg_info(env, info)
        if self._pose_settled_insts:
            self._populate_pose_settled_info(info)
        return done, info

    def _update_success_tracking(self, env):
        if self._last_goal_eval is None:
            return
        binary, settled, _ = self._last_goal_eval
        t = env.episode_steps
        if binary and self._first_success_step < 0:
            self._first_success_step = t
            self._errors_at_first_success = self.collect_errors()
        if binary and settled and self._first_success_step >= 0 and self._first_settled_success_step < 0:
            self._first_settled_success_step = t

    # ------------------------------------------------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------------------------------------------------

    def _is_configured(self, inst, state_cls):
        entity = (self.object_scope or {}).get(inst)
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

    def collect_errors(self) -> Dict[str, Any]:
        """Current errors of every configured target plus the max position / orientation errors.

        ``max_position_error`` / ``max_orientation_error`` range over the pose targets (``pose_reached``-type:
        world/camera-frame pose, depth or orientation); ``max_layout_position_error`` /
        ``max_layout_orientation_error`` range over the layout targets (group relations, prismatic / revolute joints).
        """
        pose_errors, layout_errors = {}, {}
        for inst in self.measured_pose_instances:
            report = self.object_scope[inst].states[PoseReached].get_error_report()
            if report is not None:
                report["gating"] = self.compiled_task.is_gating("pose_reached", inst)
                pose_errors[inst] = report
        for inst in self.measured_layout_instances:
            report = self.object_scope[inst].states[LayoutReached].get_error_report()
            if report is not None:
                report["gating"] = self.compiled_task.is_gating("layout_reached", inst)
                layout_errors[inst] = report

        def _max(reports, key):
            vals = [r[key] for r in reports.values() if r.get(key) is not None]
            return max(vals) if vals else None

        return {
            "pose_errors": pose_errors,
            "layout_errors": layout_errors,
            "max_position_error": _max(pose_errors, "position_error"),
            "max_orientation_error": _max(pose_errors, "orientation_error"),
            "max_layout_position_error": _max(layout_errors, "position_error"),
            "max_layout_orientation_error": _max(layout_errors, "orientation_error"),
        }

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
        """The non-gating ``(:measure ...)`` atoms with their current within-tolerance status (informational)."""
        out = []
        for lit in self.problem.measure:
            pred, arg = lit[0], lit[1].lstrip("?")
            reports = errors["pose_errors"] if pred == "pose_reached" else errors["layout_errors"]
            report = reports.get(arg)
            out.append(
                {
                    "bddl": _compact(lit),
                    "kind": "measure",
                    "configured": report is not None,
                    "within_tolerance": None if report is None else report.get("within_tolerance"),
                }
            )
        return out

    def _populate_isg_info(self, env, info):
        binary, settled, status = self._last_goal_eval if self._last_goal_eval else (False, False, info["goal_status"])
        current_errors = self.collect_errors()
        first, first_settled = self._first_success_step, self._first_settled_success_step
        latency = (first_settled - first) if (first >= 0 and first_settled >= 0) else -1
        dt = og.sim.get_sim_step_dt()
        final_errors = self._errors_at_first_success if self._errors_at_first_success is not None else current_errors
        info["ispatialgym"] = {
            "activity": self.activity_name,
            "activity_definition_id": self.activity_definition_id,
            "task_variant": self.task_variant,
            "official_success": self.official_success,
            "require_pose_settled": self.require_pose_settled,
            "binary_success": bool(binary),
            "binary_success_ever": first >= 0,
            "first_success_step": first,
            "settled": bool(settled),
            "settled_success": bool(binary and settled),
            "first_settled_success_step": first_settled,
            "settle_latency_steps": latency,
            "settle_latency_seconds": (latency * dt) if latency >= 0 else None,
            "step": env.episode_steps,
            "conditions": self.goal_conditions_report(status),
            "measured": self.measured_conditions_report(current_errors),
            "errors": current_errors,
            "errors_at_first_success": self._errors_at_first_success,
            "final_errors": final_errors,
            "max_position_error": final_errors["max_position_error"],
            "max_orientation_error": final_errors["max_orientation_error"],
            "max_layout_position_error": final_errors["max_layout_position_error"],
            "max_layout_orientation_error": final_errors["max_layout_orientation_error"],
            "group_assignments": {
                name: (entity.evaluate().assignment if entity.evaluate() is not None else None)
                for name, entity in self.group_entities.items()
            },
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
