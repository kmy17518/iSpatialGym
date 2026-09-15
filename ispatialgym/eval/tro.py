"""Loading of iSpatialGym TRO files and configuration of evaluator targets from them.

A TRO file keeps the stock BEHAVIOR layout (one serialized object state per BDDL instance plus ``robot_poses``) and
adds the reserved ``ispatialgym`` block (:mod:`ispatialgym.eval.targets`). :func:`load_tro` splits the two,
:func:`validate_tro_against_problem` checks the block against a compiled problem, and
:func:`apply_targets_to_scope` pushes the numeric targets into the object states of the bound scope entities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ispatialgym.activity.problem import ISGCompiledTask, is_agent_instance, same_activity
from ispatialgym.eval.targets import TRO_KEY, ISGTroMeta, TROValidationError, parse_frame

ROBOT_POSES_KEY = "robot_poses"


@dataclass
class TROFile:
    path: Optional[str]
    object_states: Dict[str, Any]  # BDDL instance -> serialized BEHAVIOR object state (plain JSON values)
    robot_poses: Optional[Dict[str, Any]]
    meta: ISGTroMeta

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: Optional[str] = None) -> "TROFile":
        if TRO_KEY not in data:
            raise TROValidationError(f"TRO file has no reserved '{TRO_KEY}' block")
        meta = ISGTroMeta.from_dict(data[TRO_KEY])
        object_states = {k: v for k, v in data.items() if k not in (TRO_KEY, ROBOT_POSES_KEY)}
        return cls(path=path, object_states=object_states, robot_poses=data.get(ROBOT_POSES_KEY), meta=meta)


def load_tro(path: str) -> TROFile:
    with open(path, "r") as f:
        data = json.load(f)
    return TROFile.from_dict(data, path=path)


# ----------------------------------------------------------------------------------------------------------------
# Validation against a problem
# ----------------------------------------------------------------------------------------------------------------


def physical_instances(compiled: ISGCompiledTask):
    """Declared instances that are simulator objects: everything except the agent (groups are not declared)."""
    return [inst for inst in compiled.problem.instances if not is_agent_instance(inst)]


def validate_tro_against_problem(meta: ISGTroMeta, compiled: ISGCompiledTask, require_bindings: bool = True) -> None:
    """Raise :class:`TROValidationError` if ``meta`` does not fit the compiled problem.

    Checks: activity / definition id / alias agreement (a retired activity name such as ``world_alignment`` equals its
    canonical name ``alignment``, so the v1 TROs still validate against the renamed problems); every ``pose_reached``
    / ``layout_reached`` atom of the goal (gating) and of ``(:measure ...)`` (non-gating) has a matching target of
    the right kind; conversely every TRO target corresponds to one of those atoms (the BDDL is the single statement
    of what is evaluated, the TRO only supplies the numbers); target keys are declared instances or groups; group
    members agree; frame references are declared instances; and (optionally) every physical instance has an object
    binding.
    """
    problem = compiled.problem
    if not same_activity(meta.activity, problem.activity) or meta.activity_definition_id != problem.definition_id:
        raise TROValidationError(
            f"TRO is for {meta.name!r} but the problem is {problem.name!r}"
        )
    if meta.task_variant != problem.alias:
        raise TROValidationError(f"TRO task_variant {meta.task_variant!r} != problem alias {problem.alias!r}")

    declared = set(problem.instances)
    groups = problem.groups
    pose_args = set(compiled.all_pose_reached_args)
    layout_args = set(compiled.all_layout_reached_args)
    hint = "add it to the goal (gating) or to (:measure ...) (reported only)"

    for inst in meta.pose_targets:
        if inst not in declared:
            raise TROValidationError(f"pose_targets[{inst!r}]: not a declared instance (groups go under 'groups')")
        if inst not in pose_args:
            raise TROValidationError(f"pose_targets[{inst!r}] has no (pose_reached {inst}) atom in the problem; {hint}")
        frame_inst, _ = parse_frame(meta.pose_targets[inst].frame)
        if frame_inst is not None and frame_inst not in declared:
            raise TROValidationError(f"pose_targets[{inst!r}].frame references undeclared instance {frame_inst!r}")
    for name, target in meta.groups.items():
        if name not in groups:
            raise TROValidationError(f"groups[{name!r}] is not a declared BDDL group; have {sorted(groups)}")
        if set(target.members) != set(groups[name]):
            raise TROValidationError(
                f"groups[{name!r}].members {sorted(target.members)} != BDDL members {sorted(groups[name])}"
            )
        if name not in pose_args and name not in layout_args:
            raise TROValidationError(
                f"groups[{name!r}] has neither a (pose_reached {name}) nor a (layout_reached {name}) atom; {hint}"
            )
    for inst in meta.articulation_targets:
        if inst not in declared:
            raise TROValidationError(f"articulation_targets[{inst!r}] is not a declared instance")
        if inst not in layout_args:
            raise TROValidationError(
                f"articulation_targets[{inst!r}] has no (layout_reached {inst}) atom in the problem; {hint}"
            )

    for arg in compiled.all_pose_reached_args:
        where = "goal" if compiled.is_gating("pose_reached", arg) else ":measure"
        if arg in groups:
            g = meta.groups.get(arg)
            if g is None or not g.supports_pose_reached:
                raise TROValidationError(
                    f"(pose_reached {arg}) in {where} needs groups[{arg!r}] with target_slots and pose_tolerances"
                )
        elif arg not in meta.pose_targets:
            raise TROValidationError(f"(pose_reached {arg}) in {where} has no pose_targets[{arg!r}] entry")
    for arg in compiled.all_layout_reached_args:
        where = "goal" if compiled.is_gating("layout_reached", arg) else ":measure"
        if arg in groups:
            g = meta.groups.get(arg)
            if g is None or not g.supports_layout_reached:
                raise TROValidationError(
                    f"(layout_reached {arg}) in {where} needs groups[{arg!r}] with layout_relations and layout_tolerances"
                )
        elif arg not in meta.articulation_targets:
            raise TROValidationError(f"(layout_reached {arg}) in {where} has no articulation_targets[{arg!r}] entry")

    physical = set(physical_instances(compiled))
    extra = set(meta.object_bindings) - physical
    if extra:
        raise TROValidationError(f"object_bindings has entries that are not physical instances: {sorted(extra)}")
    if require_bindings:
        missing = physical - set(meta.object_bindings)
        if missing:
            raise TROValidationError(f"object_bindings is missing physical instances {sorted(missing)}")


# ----------------------------------------------------------------------------------------------------------------
# Applying targets to bound scope entities
# ----------------------------------------------------------------------------------------------------------------


def resolve_frame(frame: str, object_scope: Dict[str, Any]) -> str:
    """Turn a TRO frame (``"world"`` / ``"<inst>"`` / ``"<inst>::<link>"``) into the scene-name form that
    ``PoseReached`` understands (``"<scene object name>::<link>"``)."""
    inst, link = parse_frame(frame)
    if inst is None:
        return "world"
    entity = object_scope.get(inst)
    if entity is None:
        raise TROValidationError(f"frame {frame!r}: instance {inst!r} is not bound to a scene entity")
    return entity.name if link is None else f"{entity.name}::{link}"


def apply_targets_to_scope(meta: ISGTroMeta, object_scope: Dict[str, Any]) -> None:
    """Configure ``PoseReached`` / ``LayoutReached`` / group targets on the bound entities of ``object_scope``.

    Every pose target is configured, whether or not a ``pose_reached`` goal atom references it, so that its errors
    are measured. Unknown links / sensors / joints raise (strict, no fallbacks).
    """
    from omnigibson.object_states import LayoutReached, PoseReached

    from ispatialgym.eval.group_entity import GroupEntity

    for inst, target in meta.pose_targets.items():
        entity = object_scope.get(inst)
        if entity is None:
            raise TROValidationError(f"pose_targets[{inst!r}]: instance is not bound to a scene entity")
        if target.link is not None:
            sensors = getattr(entity, "sensors", None) or {}
            links = getattr(entity, "links", None) or {}
            if target.link not in sensors and target.link not in links:
                raise TROValidationError(
                    f"pose_targets[{inst!r}].link {target.link!r} is neither a sensor nor a link of {entity.name}"
                )
        kwargs = target.pose_reached_kwargs(resolve_frame(target.frame, object_scope))
        entity.states[PoseReached].set_goal(**kwargs)

    for name, target in meta.groups.items():
        entity = object_scope.get(name)
        if not isinstance(entity, GroupEntity):
            raise TROValidationError(f"groups[{name!r}]: scope entry is not a GroupEntity ({entity!r})")
        entity.set_target(target)

    for inst, target in meta.articulation_targets.items():
        entity = object_scope.get(inst)
        if entity is None:
            raise TROValidationError(f"articulation_targets[{inst!r}]: instance is not bound to a scene entity")
        entity.states[LayoutReached].set_goal(
            {name: {"position": pos, "tolerance": tol} for name, (pos, tol) in target.joints.items()}
        )
