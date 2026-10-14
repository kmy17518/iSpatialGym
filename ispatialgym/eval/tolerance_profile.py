"""Evaluation tolerances, profile ``isg-v2`` (owner 2026-10-14): one tolerance per target, used wherever a target is judged - the
goal predicates (``pose_reached`` / ``layout_reached``), geometric success, the oracle's push loops (they read the task's loaded TRO)
and the goal override.

:func:`apply_to_task` computes them at load, after the loader's settle and before the first step, overwrites the loaded TRO's
tolerances in memory and re-configures the ``PoseReached`` / ``LayoutReached`` / group targets (every later re-application reads the
same TRO, so it keeps them). Nothing is written to the dataset: the TRO's own tolerance fields stay (``ispatialgym.eval.targets``
requires them) and are superseded at evaluation time.

Scale groups (:func:`scale_group`, by the instance's variant and the target's kind):

* ``micro``: pose targets of object_scaling, mental_rotation and alignment; layout targets of articulation (joints), block_design
  and attachment (group relations);
* ``small``: pose targets of articulation small-scale (the root), block_design and attachment (the group's slots), disembedding and
  IORS small-scale (target and partner);
* ``large``: pose targets of camera_relocalization and IORS large-scale.

Values at scale 1: micro poses max(0.01 m, 10 % of the change) / max(0.05 rad, 10 % of the change); micro joints per joint,
prismatic max(0.01 m, 10 % of |dq|), revolute max(0.05 rad, 10 % of |dq|); micro group relations 0.01 m / 0.05 rad (floors only);
small 0.03 m / 0.10 rad; large 0.05 m / 0.15 rad. The change is the target's error at the first step of the episode (after the
loader's settle, before any action), from the evaluation's own error code (metric, frame, symmetry equivalences); per joint for an
articulation, so a joint the task does not move gets the floor.

``scale`` (``run_isg --tolerance-scale``: 3 with ``--mode policy``, else 1) multiplies every tolerance, overrides included.
Overrides: the ``evaluation.tolerances`` block of the subtask's entry in ``isg_customs.json`` (keyed by
``ispatialgym.eval.names.subtask_name`` of the variant; empty unless someone tunes it): ``pose`` and ``layout`` take absolute
``position`` (m) / ``orientation`` (rad), ``joints`` takes the ``revolute`` / ``prismatic`` floors and the ``fraction``; an absent
key keeps the default. ``generation.tolerances`` is generation-only and never read here.

Object scaling compares the full position in the head-camera frame: its target's ``position_metric`` is set to ``"full"`` at load
(the TRO stores the 3-D camera-frame position; the frame stays).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

PROFILE_NAME = "isg-v2"

#: micro-scale floors and the share of the change above them
MICRO_FLOORS = {"position": 0.01, "orientation": 0.05}
JOINT_FLOORS = {"prismatic": 0.01, "revolute": 0.05}
FRACTION = 0.10
#: micro-scale group relations: floors only
RELATION_TOLERANCES = {"position": 0.01, "orientation": 0.05}
FIXED_TOLERANCES = {"small": {"position": 0.03, "orientation": 0.10}, "large": {"position": 0.05, "orientation": 0.15}}
#: ``evaluation.tolerances`` sections and their keys
OVERRIDE_KEYS = {"pose": ("position", "orientation"), "layout": ("position", "orientation"), "joints": ("revolute", "prismatic", "fraction")}
#: target kinds: a pose target, a group's slots (group pose_reached), a group's relations (group layout_reached), an articulation's joints
KINDS = ("pose", "group_pose", "group_layout", "joints")
#: ``run_isg --tolerance-scale`` default per mode
DEFAULT_SCALE = {"policy": 3.0}

MICRO_POSE_FAMILIES = ("object_scaling", "mental_rotation", "alignment")
IORS = "inter_object_relation_satisfaction"


class ToleranceProfileError(ValueError):
    """A target the profile has no rule for, or a malformed ``evaluation.tolerances`` block."""


def default_scale(mode: str) -> float:
    return float(DEFAULT_SCALE.get(mode, 1.0))


def family_task(variant: str) -> Tuple[str, str]:
    """``(task family, task)`` of a variant: ``configuration_matching-articulation_open_small_scale-blender-...`` ->
    ``("configuration_matching", "articulation_open_small_scale")``; the task of a held-object variant is its family."""
    parts = str(variant or "").split("-")
    if len(parts) < 2 or not parts[0]:
        raise ToleranceProfileError(f"variant {variant!r} names no task family and task")
    family = parts[0]
    task = family if family in ("mental_rotation", "object_scaling") else parts[1]
    return family, task


def scale_group(variant: str, kind: str) -> str:
    """The scale group (``micro`` / ``small`` / ``large``) of a target of ``kind`` (:data:`KINDS`) in an instance of ``variant``."""
    if kind not in KINDS:
        raise ToleranceProfileError(f"unknown target kind {kind!r}; expected one of {KINDS}")
    family, task = family_task(variant)
    if kind == "joints":
        if family == "configuration_matching" and task.startswith("articulation"):
            return "micro"
    elif kind == "group_layout":
        if family == "configuration_matching" and task in ("block_design", "attachment"):
            return "micro"
    elif kind == "group_pose":
        if family == "configuration_matching" and task in ("block_design", "attachment"):
            return "small"
    else:
        if family in MICRO_POSE_FAMILIES:
            return "micro"
        if family == "configuration_matching" and task.startswith("articulation") and task.endswith("small_scale"):
            return "small"
        if family == "disembedding":
            return "small"
        if family == IORS:
            return "large" if task.endswith("_large_scale") else "small"
        if family == "camera_relocalization":
            return "large"
    raise ToleranceProfileError(f"no isg-v2 scale group for a {kind} target of {variant!r}")


def validate_overrides(tolerances: Optional[Dict[str, Any]], where: str = "evaluation.tolerances") -> Dict[str, Dict[str, float]]:
    """``evaluation.tolerances`` checked: known sections and keys, positive numbers."""
    out: Dict[str, Dict[str, float]] = {}
    for section, values in (tolerances or {}).items():
        if section not in OVERRIDE_KEYS:
            raise ToleranceProfileError(f"{where}: unknown section {section!r}; expected {sorted(OVERRIDE_KEYS)}")
        if not isinstance(values, dict):
            raise ToleranceProfileError(f"{where}.{section} must be an object")
        for key, value in values.items():
            if key not in OVERRIDE_KEYS[section]:
                raise ToleranceProfileError(f"{where}.{section}: unknown key {key!r}; expected {list(OVERRIDE_KEYS[section])}")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not value > 0:
                raise ToleranceProfileError(f"{where}.{section}.{key} must be a positive number, got {value!r}")
        out[section] = {k: float(v) for k, v in values.items()}
    return out


def overrides_for(variant: str, customs_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """The ``evaluation.tolerances`` block of the variant's subtask entry (empty when absent)."""
    from ispatialgym.eval.names import subtask_name
    from ispatialgym.sampling.customs import DEFAULT_CUSTOMS_PATH

    path = customs_path or DEFAULT_CUSTOMS_PATH
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    subtask = subtask_name(variant)
    entry = data.get(subtask) if isinstance(data, dict) else None
    tolerances = ((entry or {}).get("evaluation") or {}).get("tolerances")
    return validate_overrides(tolerances, where=f"{os.path.basename(path)}[{subtask!r}].evaluation.tolerances")


def _pose_values(group: str, change: Dict[str, Optional[float]], components: List[str], override: Dict[str, float], scale: float,
                 relation: bool = False) -> Tuple[Dict[str, Optional[float]], Dict[str, bool]]:
    tol: Dict[str, Optional[float]] = {"position": None, "orientation": None}
    overridden = {"position": False, "orientation": False}
    for comp in components:
        if comp in override:
            value = override[comp]
            overridden[comp] = True
        elif relation:
            value = RELATION_TOLERANCES[comp]
        elif group == "micro":
            value = max(MICRO_FLOORS[comp], FRACTION * float(change.get(comp) or 0.0))
        else:
            value = FIXED_TOLERANCES[group][comp]
        tol[comp] = float(value) * scale
    return tol, overridden


def plan(variant: str, targets: List[Dict[str, Any]], scale: float = 1.0, overrides: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, Any]]:
    """The tolerance of every target, simulator-free. ``targets``: ``{"name", "kind", "components": ["position", "orientation"],
    "change": {"position", "orientation"}}`` for a pose / group target, ``{"name", "kind": "joints", "joints": {joint: {"type":
    "prismatic" | "revolute", "change": |dq|}}}`` for an articulation; ``"instance"`` (default the name): the BDDL instance or group
    the target belongs to. Returns ``{name: {"kind", "instance", "scale_group", "change", "tolerances", "override"}}``."""
    overrides = validate_overrides(overrides)
    scale = float(scale)
    if not scale > 0:
        raise ToleranceProfileError(f"tolerance scale must be > 0, got {scale}")
    out: Dict[str, Dict[str, Any]] = {}
    for t in targets:
        kind, name = t["kind"], t["name"]
        group = scale_group(variant, kind)
        if kind == "joints":
            ov = overrides.get("joints", {})
            fraction = ov.get("fraction", FRACTION)
            change, tol, overridden = {}, {}, {}
            for joint, spec in t["joints"].items():
                jtype = spec["type"]
                if jtype not in JOINT_FLOORS:
                    raise ToleranceProfileError(f"{name}: joint {joint!r} has type {jtype!r}; expected one of {sorted(JOINT_FLOORS)}")
                dq = abs(float(spec.get("change") or 0.0))
                floor = ov.get(jtype, JOINT_FLOORS[jtype])
                change[joint] = dq
                tol[joint] = max(floor, fraction * dq) * scale
                overridden[joint] = jtype in ov or "fraction" in ov
            out[name] = {"kind": kind, "instance": t.get("instance", name), "scale_group": group, "change": change, "tolerances": tol,
                         "override": overridden}
            continue
        section = "layout" if kind == "group_layout" else "pose"
        tol, overridden = _pose_values(group, t.get("change") or {}, list(t.get("components") or []), overrides.get(section, {}), scale,
                                       relation=kind == "group_layout")
        out[name] = {"kind": kind, "instance": t.get("instance", name), "scale_group": group, "change": dict(t.get("change") or {}), "tolerances": tol,
                     "override": overridden}
    return out


# ----------------------------------------------------------------------------------------------------------------
# Applying the profile to a loaded task
# ----------------------------------------------------------------------------------------------------------------


def _configure(task, meta, env_idx: Optional[int], scene) -> None:
    if env_idx is None:
        task.configure_targets(meta)
    else:
        task.configure_targets(meta, env_idx=env_idx, scene=scene)


def _joint_type(value) -> str:
    from omnigibson.utils.constants import JointType

    return "prismatic" if value == JointType.JOINT_PRISMATIC else "revolute"


def _components(target) -> List[str]:
    return [c for c, metric in (("position", target.position_metric), ("orientation", target.orientation_metric)) if metric is not None]


def target_tolerances(meta, inst: str, predicate: str) -> Optional[Dict[str, Any]]:
    """The tolerances a target is judged with (the loaded TRO's, which :func:`apply_to_task` overwrites): ``{"position",
    "orientation"}`` of a pose target (None for a component it does not compare) or of a group's slots / relations,
    ``{"joints": {joint: tolerance}}`` of an articulation; None for an instance without a target."""
    if meta is None:
        return None
    if predicate == "pose_reached":
        t = meta.pose_targets.get(inst)
        if t is not None:
            return {"position": t.position_tolerance if t.position_metric is not None else None,
                    "orientation": t.orientation_tolerance if t.orientation_metric is not None else None}
        g = meta.groups.get(inst)
        tol = g.pose_tolerances if g is not None else None
    else:
        art = meta.articulation_targets.get(inst)
        if art is not None:
            return {"joints": {j: float(tol) for j, (_pos, tol) in art.joints.items()}}
        g = meta.groups.get(inst)
        tol = g.layout_tolerances if g is not None else None
    return None if not tol else {"position": float(tol[0]), "orientation": float(tol[1])}


def apply_to_task(task, mode: str, scale: Optional[float] = None, customs_path: Optional[str] = None, meta=None, env_idx: Optional[int] = None,
                  scene=None) -> Dict[str, Any]:
    """Compute the profile's tolerances for the task's loaded TRO (``meta``: default ``task._tro.meta``; ``env_idx`` / ``scene``: the
    environment of a ``VecISpatialGymTask``) from the current state - after the loader's settle, before the first step - write them
    into the TRO in memory and re-configure the targets. Returns the record ``meta.json`` keeps (``tolerance_profile``; its
    ``targets`` are keyed by instance for a pose target, ``<group>:pose`` / ``<group>:layout`` for a group, ``<instance>:joints``
    for an articulation, whose instance may also have a pose target)."""
    meta = meta if meta is not None else task._tro.meta
    variant = meta.task_variant
    family, _ = family_task(variant)
    scale = default_scale(mode) if scale is None else float(scale)
    notes: Dict[str, Dict[str, Any]] = {}
    if family == "object_scaling":
        for inst, target in meta.pose_targets.items():
            if target.position_metric == "depth":
                target.position_metric = "full"
                notes[inst] = {"position_metric": "full", "was": "depth"}
    _configure(task, meta, env_idx, scene)
    errors = task.collect_errors() if env_idx is None else task.collect_errors(env_idx)
    pose_errors, layout_errors = errors.get("pose_errors") or {}, errors.get("layout_errors") or {}
    targets: List[Dict[str, Any]] = []
    for inst, target in meta.pose_targets.items():
        rep = pose_errors.get(inst)
        if rep is None:
            raise ToleranceProfileError(f"pose target {inst!r} of {variant!r} reports no error at load")
        targets.append({"name": inst, "kind": "pose", "components": _components(target),
                        "change": {"position": rep.get("position_error"), "orientation": rep.get("orientation_error")}})
    for name, g in meta.groups.items():
        if g.supports_pose_reached:
            rep = pose_errors.get(name) or {}
            targets.append({"name": f"{name}:pose", "instance": name, "kind": "group_pose", "components": ["position", "orientation"],
                            "change": {"position": rep.get("position_error"), "orientation": rep.get("orientation_error")}})
        if g.supports_layout_reached:
            rep = layout_errors.get(name) or {}
            targets.append({"name": f"{name}:layout", "instance": name, "kind": "group_layout", "components": ["position", "orientation"],
                            "change": {"position": rep.get("position_error"), "orientation": rep.get("orientation_error")}})
    for inst, art in meta.articulation_targets.items():
        joints = (layout_errors.get(inst) or {}).get("joints") or {}
        missing = sorted(set(art.joints) - set(joints))
        if missing:
            raise ToleranceProfileError(f"articulation target {inst!r} of {variant!r}: no error at load for joints {missing}")
        targets.append({"name": f"{inst}:joints", "instance": inst, "kind": "joints",
                        "joints": {j: {"type": _joint_type(joints[j]["joint_type"]), "change": joints[j]["error"]} for j in art.joints}})
    overrides = overrides_for(variant, customs_path)
    planned = plan(variant, targets, scale=scale, overrides=overrides)
    for inst, target in meta.pose_targets.items():
        tol = planned[inst]["tolerances"]
        if target.position_metric is not None:
            target.position_tolerance = tol["position"]
        if target.orientation_metric is not None:
            target.orientation_tolerance = tol["orientation"]
    for name, g in meta.groups.items():
        if g.supports_pose_reached:
            tol = planned[f"{name}:pose"]["tolerances"]
            g.pose_tolerances = (tol["position"], tol["orientation"])
        if g.supports_layout_reached:
            tol = planned[f"{name}:layout"]["tolerances"]
            g.layout_tolerances = (tol["position"], tol["orientation"])
    for inst, art in meta.articulation_targets.items():
        tol = planned[f"{inst}:joints"]["tolerances"]
        art.joints = {j: (pos, tol[j]) for j, (pos, _old) in art.joints.items()}
    _configure(task, meta, env_idx, scene)
    for name, note in notes.items():
        planned[name]["note"] = note
    return {"name": PROFILE_NAME, "mode": mode, "scale": scale, "overrides": overrides, "targets": planned}


__all__ = ["PROFILE_NAME", "ToleranceProfileError", "KINDS", "default_scale", "family_task", "scale_group", "validate_overrides",
           "overrides_for", "plan", "apply_to_task", "target_tolerances"]
