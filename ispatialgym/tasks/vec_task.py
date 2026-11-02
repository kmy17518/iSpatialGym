"""``VecISpatialGymTask``: :class:`~ispatialgym.tasks.ispatialgym_task.ISpatialGymTask` for a vectorized ``og.Environment``
(``num_envs`` cloned scenes), every environment with its own scope, targets and success tracking (the base class tracks, reports
and fills ``infos[env_idx]`` per env; this class binds the scopes and targets per env).

Design (BEHAVIOR's ``omnigibson/eval/evaluator.py``): one task object, ``object_scopes[env_idx]`` bound to ``env.scenes[env_idx]``'s
objects and robot, the goal evaluated per env through ``_evaluate_predicate(env_idx, ...)`` (``PredicateGoal`` passes
``env_idx``), ``infos[env_idx]`` filled per env. A TRO is authored in env 0's frame (the base scene): when it is loaded
into clone ``env_idx`` its object root poses, robot pose and **world-frame pose targets** are converted with
``scene.convert_scene_relative_pose_to_world`` (clones are translated copies, so a goal left in base-scene coordinates
would sit in env 0's geometry - unreachable from the clone, or reachable at the wrong place).

With ``num_envs == 1`` every method reduces to the single-env class's behaviour (same scope, same info). The single-env
class is untouched; this one is selected by ``task.type = "VecISpatialGymTask"`` in the env config.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Optional

import torch as th

import omnigibson as og
from omnigibson.object_states import LayoutReached, PoseReached, PoseSettled
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.tasks.task_base import BaseTask
from omnigibson.utils.python_utils import recursively_convert_to_torch

from ispatialgym.activity.problem import AGENT_INSTANCE, is_agent_instance
from ispatialgym.eval.clone_frames import offset_object_state, offset_targets_meta, scene_translation
from ispatialgym.eval import tolerance_profile as TP
from ispatialgym.eval.group_entity import GroupEntity
from ispatialgym.eval.tro import apply_targets_to_scope, load_tro, physical_instances, validate_tro_against_problem
from ispatialgym.sampling import predicates as PRED
from ispatialgym.tasks.ispatialgym_task import ISpatialGymTask, apply_environment_rules

log = logging.getLogger(__name__)


class VecISpatialGymTask(ISpatialGymTask):
    """Per-environment iSpatialGym task (see the module docstring)."""

    def __init__(self, activity_name, activity_definition_id=0, activity_instance_id=0, activity_definitions_root=None, tro_path=None, object_bindings=None,
                 settle_eval=False, settle_steps=45, pose_settled_config=None, highlight_task_relevant_objects=False, termination_config=None,
                 reward_config=None, include_obs=False):
        # the explicit signature: OmniGibson's registry picks the config keys from it
        self._n = 1
        self._scopes: Optional[List[Dict[str, Any]]] = None
        super().__init__(activity_name, activity_definition_id=activity_definition_id, activity_instance_id=activity_instance_id, activity_definitions_root=activity_definitions_root,
                         tro_path=tro_path, object_bindings=object_bindings, settle_eval=settle_eval, settle_steps=settle_steps,
                         pose_settled_config=pose_settled_config, highlight_task_relevant_objects=highlight_task_relevant_objects, termination_config=termination_config,
                         reward_config=reward_config, include_obs=include_obs)

    @property
    def n_envs(self) -> int:
        return self._n

    # ------------------------------------------------------------------ scopes (per env)

    @property
    def object_scopes(self):
        return self._scopes

    @object_scopes.setter
    def object_scopes(self, scopes):
        self._scopes = None if scopes is None else list(scopes)

    @property
    def object_scope(self):
        """Env 0's scope (the single-env helpers of the base class read it)."""
        return None if not self._scopes else self._scopes[0]

    @object_scope.setter
    def object_scope(self, scope):
        if scope is None:
            self._scopes = None
        elif not self._scopes:
            self._scopes = [scope]
        else:
            self._scopes[0] = scope

    def scope_of(self, env_idx: int) -> Dict[str, Any]:
        return self._scopes[env_idx]

    def _resolve_member(self, inst):
        return self.object_scope.get(inst) if self.object_scope else None

    # ------------------------------------------------------------------ loading

    def _load(self, env):
        self._n = int(env.num_envs)
        self.update_activity(env=env, activity_name=self.activity_name, activity_definition_id=self.activity_definition_id)
        # update_activity built env 0's template scope; the clones get their own copies (group entities bound per env)
        template = self._scopes[0]
        scopes = []
        for env_idx in range(self._n):
            scope = {}
            for k, v in template.items():
                if isinstance(v, GroupEntity):
                    scope[k] = GroupEntity(k, list(self.groups[k]), resolve_member=lambda inst, i=env_idx: (self._scopes[i].get(inst) if self._scopes else None))
                else:
                    scope[k] = None
            scopes.append(scope)
        self._scopes = scopes
        self._reset_tracking()
        if self._tro_path is not None:
            self._tro = load_tro(self._tro_path)
            validate_tro_against_problem(self._tro.meta, self.compiled_task, require_bindings=False)
        for env_idx in range(self._n):
            ok, self.feedback = self.initialize_activity(env=env, env_idx=env_idx)
            assert ok, f"Failed to initialize iSpatialGym activity in env {env_idx}. Feedback:\n{self.feedback}"
        self.scene_name = env.scenes[0].scene_model if isinstance(env.scenes[0], TraversableScene) else None
        if self.highlight_task_relevant_objs:
            for scope in self._scopes:
                for inst, entity in scope.items():
                    if entity is not None and not isinstance(entity, GroupEntity) and not is_agent_instance(inst):
                        entity.highlighted = True

    def initialize_activity(self, env, env_idx: int = 0):
        bindings = self._object_bindings_source(env)
        scene = env.scenes[env_idx]
        scope = self._scopes[env_idx]
        missing = []
        for inst in physical_instances(self.compiled_task):
            name = bindings.get(inst)
            if isinstance(name, dict):
                name = name.get("scene_object_name")
            if name is None:
                missing.append(inst)
                continue
            entity = scene.object_registry("name", name)
            if entity is None:
                return False, f"Object binding {inst!r} -> {name!r} does not exist in scene {env_idx}"
            scope[inst] = entity
        if missing:
            return False, f"No object binding for instances {missing}"
        scope[AGENT_INSTANCE] = self.get_agent(env, env_idx)
        if env_idx > 0:
            # the clones are copies of env 0's scene file: every bound instance must carry the same scene object name (the
            # TRO's object_bindings, the attachment context and the recorder's names are shared by name)
            for inst, entity in scope.items():
                e0 = self._scopes[0].get(inst)
                if entity is not None and e0 is not None and not isinstance(entity, GroupEntity) and getattr(entity, "name", None) != getattr(e0, "name", None):
                    return False, f"env {env_idx}: {inst!r} is bound to {entity.name!r}, env 0 to {e0.name!r}"
        apply_environment_rules(getattr(env, "robots", None))
        if self._tro is not None:
            self.configure_targets(self._tro.meta, env_idx=env_idx, scene=scene)
        return True, None

    def _object_bindings_source(self, env) -> Dict[str, Any]:
        if self._config_object_bindings:
            return self._config_object_bindings
        if self._tro is not None and self._tro.meta.object_bindings:
            return self._tro.meta.object_bindings
        try:
            return env.scenes[0].get_task_metadata(key="inst_to_name") or {}
        except Exception:  # noqa: BLE001
            return {}

    def configure_targets(self, meta, env_idx: int = 0, scene=None):
        """TRO targets into env ``env_idx``'s scope, the world-frame pose targets placed in that clone's world frame."""
        scope = self._scopes[env_idx]
        apply_targets_to_scope(offset_targets_meta(meta, scene), scope)
        self._targets_configured = True
        self._target_metas[env_idx] = meta
        for entity in scope.values():
            if isinstance(entity, GroupEntity):
                entity.invalidate()
        # the attachment pairs (scene object names, links, the child's symmetry) registered for this clone's scene: the names
        # are the clone's own (initialize_activity binds every env's scope and asserts the names match env 0's, so the
        # base class's env-0 pairs are this env's pairs); the predicate wrapper looks them up by the entity's scene index
        PRED.set_attachment_context(self._attachment_pairs(meta), scene_idx=getattr(scene, "idx", None) if scene is not None else None)

    def load_tro_instance(self, env, tro=None, settle_steps=25, apply_object_states=True, env_idx: int = 0, defer_reset_step: bool = False):
        """The TRO into clone ``env_idx``: object states with their root poses moved into the clone's frame, the robot pose set
        scene-relative, the robot joints, the targets; then exactly the single-env sequence - ``settle_steps`` physics steps with
        the scope's entities kept still, the scene's initial file updated, the scene reset, the tracking of that env reset. The
        vectorized loader calls it with ``settle_steps=0`` (as ``run_isg.InstanceLoader.restore`` does) and settles the batch.

        ``defer_reset_step`` (vec_next 2026-10-31): the scene is restored from its new initial file WITHOUT the physics step that
        ``Scene.reset`` ends with - that step advances every scene of the world, so restoring N clones one after another gave the clone
        restored first N-1 extra unheld physics steps (the per-slot step-0 offsets from N=1). The caller restores every clone this way
        and then calls :meth:`finish_deferred_reset` once: per clone the N=1 sequence (restore, one physics step, tracking reset)."""
        if tro is None:
            tro = self._tro if self._tro is not None else self._tro_path
        if tro is None:
            raise ValueError("No TRO given and no tro_path configured")
        if isinstance(tro, str):
            tro = load_tro(tro)
        assert og.sim.is_playing(), "load_tro_instance requires a playing simulator"
        validate_tro_against_problem(tro.meta, self.compiled_task, require_bindings=False)
        self._tro = tro
        scene = env.scenes[env_idx]
        scope = self._scopes[env_idx]
        self._verify_object_bindings_env(tro, scope)
        robot = self.get_agent(env, env_idx)
        if apply_object_states:
            offset = scene_translation(scene)
            for inst, state in tro.object_states.items():
                entity = scope.get(inst)
                if entity is None or isinstance(entity, GroupEntity):
                    raise ValueError(f"TRO stores a state for {inst!r}, which is not a bound physical instance")
                st = offset_object_state(recursively_convert_to_torch(copy.deepcopy(state)), scene)
                entity.load_state(st, serialized=False)
            if tro.robot_poses:
                poses = {k.lower(): v for k, v in tro.robot_poses.items()}
                if "robot" in poses:
                    available = poses["robot"]
                elif robot.model in poses:
                    available = poses[robot.model]
                else:
                    raise KeyError(f"No generic or model-specific presampled robot pose for {robot.model}")
                pose = available[0]
                # the base scene: the single-env call bit for bit; a clone: the position translated, the orientation untouched (the
                # scene-frame setter's matrix round trip would perturb the quaternion)
                position = th.tensor(pose["position"]) if offset is None else th.tensor(pose["position"], dtype=th.float32) + offset
                robot.set_position_orientation(position, th.tensor(pose["orientation"]))
                scene.write_task_metadata(key="robot_poses", data=tro.robot_poses)
            self._restore_robot_state(robot, tro.meta.robot_state)
            for key in ("grasps", "attachment_targets", "constraints"):
                if tro.meta.extras.get(key):
                    log.warning(f"TRO '{key}' records are not restored by VecISpatialGymTask.load_tro_instance (the loader restores them per env)")
        self.configure_targets(tro.meta, env_idx=env_idx, scene=scene)
        og.sim.update_handles()
        for _ in range(settle_steps):
            og.sim.step_physics()
            for entity in scope.values():
                if entity is not None and not isinstance(entity, GroupEntity):
                    entity.keep_still()
        scene.update_initial_file()
        if defer_reset_step:
            self._restore_scene_no_step(scene)
            return
        scene.reset()
        self._reset_tracking([env_idx])

    @staticmethod
    def _restore_scene_no_step(scene) -> None:
        """``Scene.reset(hard=True)`` without its closing ``og.sim.step_physics()`` (the step that advances the whole world)."""
        assert og.sim.is_playing(), "Simulator must be playing in order to reset the scene!"
        assert scene._initial_file is not None
        scene.restore(scene_file=scene._initial_file)

    def finish_deferred_reset(self, env_indices) -> None:
        """The one physics step and the tracking reset after :meth:`load_tro_instance` (``defer_reset_step=True``) of every listed env."""
        og.sim.step_physics()
        self._reset_tracking([int(i) for i in env_indices])

    def _reset_scene(self, env, env_indices):
        """``BaseTask._reset_scene`` (each scene's ``reset()``, each ending with a physics step of the WHOLE world) as one batch: every
        listed scene restored, then one physics step - per scene the N=1 sequence. ``ISG_BATCHED_SCENE_RESET=0`` restores the stock loop."""
        if os.environ.get("ISG_BATCHED_SCENE_RESET", "1") == "0":
            return BehaviorTask._reset_scene(self, env, env_indices)
        for idx in env_indices:
            self._restore_scene_no_step(env.scenes[int(idx)])
        og.sim.step_physics()

    def _verify_object_bindings_env(self, tro, scope):
        for inst, binding in tro.meta.object_bindings.items():
            entity = scope.get(inst)
            if entity is None:
                raise ValueError(f"TRO object_bindings names {inst!r}, which is not bound in the scope")
            expected = binding.get("scene_object_name") if isinstance(binding, dict) else binding
            if expected is not None and entity.name != expected:
                raise ValueError(f"object_bindings[{inst!r}] = {expected!r} but the scope holds {entity.name!r}")
            if isinstance(binding, dict):
                for key, attr in (("category", "category"), ("model", "model")):
                    if binding.get(key) is not None and getattr(entity, attr, None) != binding[key]:
                        raise ValueError(f"object_bindings[{inst!r}].{key} = {binding[key]!r} but the scene object has {getattr(entity, attr, None)!r}")

    # ------------------------------------------------------------------ scope helpers (per env)

    def group_entities_of(self, env_idx: int) -> Dict[str, GroupEntity]:
        return {k: v for k, v in (self._scopes[env_idx] or {}).items() if isinstance(v, GroupEntity)}

    def physical_scope_of(self, env_idx: int) -> Dict[str, Any]:
        return {k: v for k, v in (self._scopes[env_idx] or {}).items() if not isinstance(v, GroupEntity)}

    def _evaluate_predicate(self, env_idx, predicate_name, *entities):
        return PRED.evaluate_predicate(predicate_name, *[self._scopes[env_idx][ent] for ent in entities])

    def evaluate_predicate(self, predicate_name, *entities):
        return self._evaluate_predicate(0, predicate_name, *entities)

    def update_bddl_scope_metadata(self, env):
        for env_idx in range(self._n):
            env.scenes[env_idx].write_task_metadata(key="inst_to_name", data={inst: e.name for inst, e in self.physical_scope_of(env_idx).items() if e is not None})

    def _update_bddl_scope_from_added_obj(self, obj):
        return  # the evaluator never adds objects during an episode

    def _update_bddl_scope_from_removed_obj(self, obj):
        return

    # ------------------------------------------------------------------ termination / tracking (per env)

    def _check_goal(self, env_idx=0):
        # per env, without the single-env class's shared goal evaluation (ISPATIALGYM_SHARED_GOAL_EVAL keeps one slot)
        return self._check_goal_impl(env_idx)

    def _reset_variables(self, env, env_indices):
        BaseTask._reset_variables(self, env, env_indices)
        self._reset_tracking(env_indices)

    def _reset_pose_settled_tracking(self, env_indices=None):
        idx = range(self._n) if env_indices is None else [int(i) for i in env_indices]
        for i in idx:
            self._pose_settled_other_success[i] = {}
            self._t_other_success[i] = -1
            for inst in self._pose_settled_insts:
                entity = self._scopes[i].get(inst)
                if entity is not None and not isinstance(entity, GroupEntity) and PoseSettled in entity.states:
                    entity.states[PoseSettled].activate(**self.pose_settled_config_of(i))

    @property
    def success(self):
        return th.tensor([self.success_of(i) for i in range(self._n)], dtype=th.bool)

    def apply_tolerance_profile(self, mode: str, scale: Optional[float] = None, customs_path: Optional[str] = None, env_idx: int = 0, scene=None,
                                meta=None) -> Dict[str, Any]:
        """:meth:`ISpatialGymTask.apply_tolerance_profile` for env ``env_idx`` (its TRO ``meta``, default the last loaded one, and
        its ``scene``)."""
        meta = meta if meta is not None else self._target_metas.get(env_idx, getattr(self._tro, "meta", None))
        self.tolerance_profiles[env_idx] = TP.apply_to_task(self, mode, scale=scale, customs_path=customs_path, meta=meta, env_idx=env_idx, scene=scene)
        return self.tolerance_profiles[env_idx]

    # ------------------------------------------------------------------ errors / info (per env)

    def measured_pose_instances_of(self, env_idx: int):
        if self.compiled_task is None:
            return []
        return [inst for inst in self.compiled_task.all_pose_reached_args if self._is_configured(inst, PoseReached, env_idx)]

    def measured_layout_instances_of(self, env_idx: int):
        if self.compiled_task is None:
            return []
        return [inst for inst in self.compiled_task.all_layout_reached_args if self._is_configured(inst, LayoutReached, env_idx)]

    def reset(self, env, env_indices=None):
        BehaviorTask.reset(self, env, env_indices=env_indices)
        for scope in self._scopes or []:
            for entity in scope.values():
                if isinstance(entity, GroupEntity):
                    entity.invalidate()

    def _get_obs(self, env, env_idx=0):
        group_names = set(self.group_entities_of(env_idx))
        backup = self._scopes
        try:
            self._scopes = [{k: v for k, v in s.items() if k not in group_names} for s in backup]
            return BehaviorTask._get_obs(self, env, env_idx)
        finally:
            self._scopes = backup


__all__ = ["VecISpatialGymTask"]
