"""Evaluation tolerances, profile ``isg-v2``: one tolerance per target, used wherever the target is judged.

:func:`apply_to_task` computes them once the instance is loaded and settled, before the first step, and writes them into
the loaded TRO in memory (the dataset files are never modified).

Scale groups (:func:`scale_group`), by task and target kind:

* ``micro``: pose targets of object_scaling, mental_rotation and alignment; the joints of articulation tasks; the group
  relations of block_design and attachment;
* ``small``: pose targets of articulation small-scale, block_design and attachment slots, disembedding and IORS small-scale;
* ``large``: pose targets of camera_relocalization and IORS large-scale.

Values at scale 1: micro poses ``max(0.01 m, 10 % of the change)`` / ``max(0.05 rad, 10 % of the change)``; micro joints
``max(0.01 m, 10 % of |dq|)`` (prismatic) or ``max(0.05 rad, 10 % of |dq|)`` (revolute); micro group relations 0.01 m /
0.05 rad; small 0.03 m / 0.10 rad; large 0.05 m / 0.15 rad. The change is the target's error at the first step, measured
with the evaluator's own metric. ``scale`` (``run_isg --tolerance-scale``, 3 for policy runs) multiplies every tolerance.

Object scaling compares the full camera-frame position: its depth-only target is switched to ``"full"`` at load.
"""

from __future__ import annotations

from typing import Any

PROFILE_NAME = "isg-v2"
DEFAULT_POLICY_SCALE = 3.0

MICRO_FLOORS = {"position": 0.01, "orientation": 0.05}
JOINT_FLOORS = {"prismatic": 0.01, "revolute": 0.05}
FRACTION = 0.10
RELATION_TOLERANCES = {"position": 0.01, "orientation": 0.05}
FIXED_TOLERANCES = {"small": {"position": 0.03, "orientation": 0.10}, "large": {"position": 0.05, "orientation": 0.15}}
#: a pose target, a group's slots, a group's relations, an articulation's joints
KINDS = ("pose", "group_pose", "group_layout", "joints")

MICRO_POSE_FAMILIES = ("object_scaling", "mental_rotation", "alignment")
IORS = "inter_object_relation_satisfaction"


class ToleranceProfileError(ValueError):
    """A target the profile has no rule for."""


def family_task(variant: str) -> tuple[str, str]:
    """``(task family, task)`` of a variant; a held-object family's task is the family itself."""
    parts = str(variant or "").split("-")
    if len(parts) < 2 or not parts[0]:
        raise ToleranceProfileError(f"variant {variant!r} names no task family and task")
    family = parts[0]
    return family, family if family in ("mental_rotation", "object_scaling") else parts[1]


def scale_group(variant: str, kind: str) -> str:
    """The scale group (``micro`` / ``small`` / ``large``) of a ``kind`` target in an instance of ``variant``."""
    if kind not in KINDS:
        raise ToleranceProfileError(f"unknown target kind {kind!r}; expected one of {KINDS}")
    family, task = family_task(variant)
    configuration = family == "configuration_matching"
    if kind == "joints":
        if configuration and task.startswith("articulation"):
            return "micro"
    elif kind == "group_layout":
        if configuration and task in ("block_design", "attachment"):
            return "micro"
    elif kind == "group_pose":
        if configuration and task in ("block_design", "attachment"):
            return "small"
    else:
        if family in MICRO_POSE_FAMILIES:
            return "micro"
        if configuration and task.startswith("articulation") and task.endswith("small_scale"):
            return "small"
        if family == "disembedding":
            return "small"
        if family == IORS:
            return "large" if task.endswith("_large_scale") else "small"
        if family == "camera_relocalization":
            return "large"
    raise ToleranceProfileError(f"no {PROFILE_NAME} scale group for a {kind} target of {variant!r}")


def _pose_tolerances(
    group: str, change: dict[str, float | None], components: list[str], scale: float, relation: bool
) -> dict[str, float | None]:
    tol: dict[str, float | None] = {"position": None, "orientation": None}
    for comp in components:
        if relation:
            value = RELATION_TOLERANCES[comp]
        elif group == "micro":
            value = max(MICRO_FLOORS[comp], FRACTION * float(change.get(comp) or 0.0))
        else:
            value = FIXED_TOLERANCES[group][comp]
        tol[comp] = float(value) * scale
    return tol


def plan(variant: str, targets: list[dict[str, Any]], scale: float = 1.0) -> dict[str, dict[str, Any]]:
    """The tolerance of every target (simulator-free).

    ``targets``: ``{"name", "kind", "components": [...], "change": {"position", "orientation"}}`` for a pose or group
    target, ``{"name", "kind": "joints", "joints": {joint: {"type", "change"}}}`` for an articulation, with an optional
    ``"instance"`` (default the name). Returns ``{name: {"kind", "instance", "scale_group", "change", "tolerances"}}``.
    """
    scale = float(scale)
    if not scale > 0:
        raise ToleranceProfileError(f"tolerance scale must be > 0, got {scale}")
    out: dict[str, dict[str, Any]] = {}
    for t in targets:
        kind, name = t["kind"], t["name"]
        group = scale_group(variant, kind)
        if kind == "joints":
            change, tol = {}, {}
            for joint, spec in t["joints"].items():
                if spec["type"] not in JOINT_FLOORS:
                    raise ToleranceProfileError(
                        f"{name}: joint {joint!r} has type {spec['type']!r}; expected one of {sorted(JOINT_FLOORS)}"
                    )
                change[joint] = abs(float(spec.get("change") or 0.0))
                tol[joint] = max(JOINT_FLOORS[spec["type"]], FRACTION * change[joint]) * scale
        else:
            change = dict(t.get("change") or {})
            tol = _pose_tolerances(
                group, change, list(t.get("components") or []), scale, relation=kind == "group_layout"
            )
        out[name] = {
            "kind": kind,
            "instance": t.get("instance", name),
            "scale_group": group,
            "change": change,
            "tolerances": tol,
        }
    return out


def _joint_type(value) -> str:
    from omnigibson.utils.constants import JointType

    return "prismatic" if value == JointType.JOINT_PRISMATIC else "revolute"


def _components(target) -> list[str]:
    return [
        c
        for c, metric in (("position", target.position_metric), ("orientation", target.orientation_metric))
        if metric is not None
    ]


def target_tolerances(meta, inst: str, predicate: str) -> dict[str, Any] | None:
    """The tolerances a target is judged with: ``{"position", "orientation"}`` of a pose or group target (None for a
    component it does not compare), ``{"joints": {joint: tolerance}}`` of an articulation, None without a target."""
    if meta is None:
        return None
    if predicate == "pose_reached":
        t = meta.pose_targets.get(inst)
        if t is not None:
            return {
                "position": t.position_tolerance if t.position_metric is not None else None,
                "orientation": t.orientation_tolerance if t.orientation_metric is not None else None,
            }
        g = meta.groups.get(inst)
        tol = g.pose_tolerances if g is not None else None
    else:
        art = meta.articulation_targets.get(inst)
        if art is not None:
            return {"joints": {j: float(tol) for j, (_pos, tol) in art.joints.items()}}
        g = meta.groups.get(inst)
        tol = g.layout_tolerances if g is not None else None
    return None if not tol else {"position": float(tol[0]), "orientation": float(tol[1])}


def apply_to_task(task, scale: float = DEFAULT_POLICY_SCALE) -> dict[str, Any]:
    """Compute the profile's tolerances of the task's loaded TRO from the current state, write them into the TRO in
    memory and re-configure the targets. Returns the record kept in ``meta.json`` (targets keyed by instance for a pose
    target, ``<group>:pose`` / ``<group>:layout`` for a group, ``<instance>:joints`` for an articulation)."""
    meta = task.tro.meta
    variant = meta.task_variant
    family, _ = family_task(variant)
    notes: dict[str, dict[str, Any]] = {}
    if family == "object_scaling":
        for inst, target in meta.pose_targets.items():
            if target.position_metric == "depth":
                target.position_metric = "full"
                notes[inst] = {"position_metric": "full", "was": "depth"}
    task.configure_targets(meta)
    errors = task.collect_errors()
    pose_errors, layout_errors = errors.get("pose_errors") or {}, errors.get("layout_errors") or {}

    targets: list[dict[str, Any]] = []
    for inst, target in meta.pose_targets.items():
        rep = pose_errors.get(inst)
        if rep is None:
            raise ToleranceProfileError(f"pose target {inst!r} of {variant!r} reports no error at load")
        change = {"position": rep.get("position_error"), "orientation": rep.get("orientation_error")}
        targets.append({"name": inst, "kind": "pose", "components": _components(target), "change": change})
    for name, g in meta.groups.items():
        for kind, supported, reports in (
            ("pose", g.supports_pose_reached, pose_errors),
            ("layout", g.supports_layout_reached, layout_errors),
        ):
            if supported:
                rep = reports.get(name) or {}
                change = {"position": rep.get("position_error"), "orientation": rep.get("orientation_error")}
                targets.append(
                    {
                        "name": f"{name}:{kind}",
                        "instance": name,
                        "kind": f"group_{kind}",
                        "components": ["position", "orientation"],
                        "change": change,
                    }
                )
    for inst, art in meta.articulation_targets.items():
        joints = (layout_errors.get(inst) or {}).get("joints") or {}
        missing = sorted(set(art.joints) - set(joints))
        if missing:
            raise ToleranceProfileError(
                f"articulation target {inst!r} of {variant!r}: no error at load for joints {missing}"
            )
        specs = {j: {"type": _joint_type(joints[j]["joint_type"]), "change": joints[j]["error"]} for j in art.joints}
        targets.append({"name": f"{inst}:joints", "instance": inst, "kind": "joints", "joints": specs})

    planned = plan(variant, targets, scale=scale)
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
    task.configure_targets(meta)
    for name, note in notes.items():
        planned[name]["note"] = note
    return {"name": PROFILE_NAME, "scale": float(scale), "targets": planned}
