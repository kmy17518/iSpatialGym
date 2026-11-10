"""Immutable conceptual pools and exact active disembedding variants.

The pool is an admission contract, not a sampled template. It lists every model
the world may preload. A physical Template and its TRO contain only selected
objects. Each physical variant still needs its own exact ValidationKey, demo,
and replay; sharing this conceptual identity never grants validation reuse.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from scripts.campaign_20261110.contracts import (
    ObjectRef, Template, _sha, canonical_json, digest, file_digest, require,
    write_json_atomic,
)

POOL_SCHEMA = "isg-binding-pool-v1"
REFERENCE_SCHEMA = "isg-active-binding-pool-v1"


@dataclass(frozen=True)
class PoolCandidate:
    candidate_id: str
    object: ObjectRef
    max_count: int = 1

    def __post_init__(self):
        require(bool(self.candidate_id), "pool candidate has no id")
        require(self.object.role in ("target", "distractor"), "only target/distractor choices may vary")
        require(self.object.ready and bool(self.object.readiness_source), "unready pool candidate")
        require(type(self.max_count) is int and self.max_count > 0, "invalid pool candidate multiplicity")
        require(self.object.semantic_kind == "object", "variable candidate must be an ordinary object")


@dataclass(frozen=True)
class BindingPool:
    conceptual_template_id: str
    class_id: str
    split: str
    scene: str
    room: str
    fixed_objects: tuple[ObjectRef, ...]
    candidates: tuple[PoolCandidate, ...]
    role_limits: Mapping[str, tuple[int, int]]
    source_definition_sha256: str
    task_rooms: tuple[str, ...] = ()
    fixed_layout: bool = False
    fixed_selection: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema: str = POOL_SCHEMA
    task: str = "disembedding"

    def __post_init__(self):
        require(self.schema == POOL_SCHEMA and self.task == "disembedding", "unsupported binding pool")
        require(bool(self.conceptual_template_id and self.class_id), "pool identity is incomplete")
        _sha(self.source_definition_sha256, "pool source definition")
        require(len({o.instance for o in self.fixed_objects}) == len(self.fixed_objects), "duplicate fixed pool binding")
        require(all(o.role == "support" for o in self.fixed_objects), "fixed pool objects must be supports")
        require(all(o.ready and o.readiness_source for o in self.fixed_objects), "unready fixed pool object")
        require(len({c.candidate_id for c in self.candidates}) == len(self.candidates), "duplicate pool candidate id")
        require(bool(self.candidates), "empty candidate pool")
        limits = {role: tuple(value) for role, value in self.role_limits.items()}
        require(set(limits) == {"target", "distractor"}, "pool must bound target and distractor counts")
        require(limits["target"] == (1, 1), "disembedding needs exactly one active target")
        for role, bounds in limits.items():
            require(len(bounds) == 2 and all(type(n) is int for n in bounds) and 0 <= bounds[0] <= bounds[1],
                    "invalid pool role count bounds")
            capacity = sum(c.max_count for c in self.candidates if c.object.role == role)
            require(bounds[1] <= capacity, "role limit exceeds admitted maximum pool")
        require(limits["distractor"][1] > 0, "disembedding pool has no distractors")
        object.__setattr__(self, "role_limits", limits)
        require(all(isinstance(room, str) and room for room in self.task_rooms), "invalid pool rooms")
        require(type(self.fixed_layout) is bool, "fixed layout must be explicit boolean")
        require(bool(self.fixed_selection) == self.fixed_layout, "only fixed-layout pools require an exact selection")
        canonical_json(self.provenance)
        # Reuse readiness/scene/split validation without claiming runtime support.
        self.audit_template()
        if self.fixed_layout:
            self.check_selection(self.fixed_selection)

    @property
    def fingerprint(self):
        return digest(asdict(self))

    def audit_template(self):
        """All roles/models, including candidates unused by a given instance."""
        objects = list(self.fixed_objects)
        objects.extend(replace(candidate.object, instance="__pool_candidate_"+str(i))
                       for i, candidate in enumerate(self.candidates))
        return Template(self.conceptual_template_id, self.task, self.class_id, self.split,
                        self.scene, self.room, tuple(objects), self.source_definition_sha256,
                        {"task_rooms": list(self.task_rooms)})

    def check_selection(self, selection):
        require(isinstance(selection, Mapping) and all(isinstance(n, str) and n for n in selection),
                "active selection must map binding names to candidate ids")
        require(not set(selection) & {o.instance for o in self.fixed_objects}, "candidate overwrites fixed support")
        candidates = {candidate.candidate_id: candidate for candidate in self.candidates}
        require(all(isinstance(value, str) and value in candidates for value in selection.values()),
                "active selection uses a model outside the maximum pool")
        counts = Counter(selection.values())
        require(all(n <= candidates[key].max_count for key, n in counts.items()), "active selection exceeds candidate multiplicity")
        roles = Counter(candidates[key].object.role for key in selection.values())
        require(all(low <= roles[role] <= high for role, (low, high) in self.role_limits.items()),
                "active selection violates role count bounds")
        if self.fixed_layout:
            require(dict(selection) == dict(self.fixed_selection), "fixed puzzle layout changed active objects")
        return roles


def pool_from_dict(value):
    data = dict(value)
    data["fixed_objects"] = tuple(ObjectRef(**row) for row in data["fixed_objects"])
    data["candidates"] = tuple(PoolCandidate(row["candidate_id"], ObjectRef(**row["object"]), row["max_count"])
                               for row in data["candidates"])
    data["task_rooms"] = tuple(data.get("task_rooms", ()))
    return BindingPool(**data)


def write_pool(path, pool):
    """Publish once; a changed candidate set requires a new manifest path/id."""
    path = Path(path).resolve()
    if path.exists():
        require(pool_from_dict(json.loads(path.read_text())).fingerprint == pool.fingerprint,
                "immutable maximum pool already exists with different content")
    else:
        write_json_atomic(path, asdict(pool))
    return path


def _same_object(actual, candidate):
    return asdict(actual) == asdict(replace(candidate, instance=actual.instance))


def validate_active_variant(template, pool, selection):
    require(template.task == pool.task and template.class_id == pool.class_id, "variant task/class differs from pool")
    require(template.split == pool.split or {template.split, pool.split} <= {"train", "eval_standard"},
            "variant split differs from pool")
    require(template.scene == pool.scene and template.room == pool.room, "variant scene/room differs from pool")
    rooms = set(template.scene_setup.get("task_rooms", ())) | {template.room}
    require(rooms == set(pool.task_rooms) | {pool.room}, "variant task rooms differ from pool")
    roles = pool.check_selection(selection)
    fixed = {obj.instance: obj for obj in pool.fixed_objects}
    candidates = {candidate.candidate_id: candidate for candidate in pool.candidates}
    require({obj.instance for obj in template.objects} == set(fixed) | set(selection),
            "physical template must contain only and all active bindings")
    for obj in template.objects:
        expected = fixed[obj.instance] if obj.instance in fixed else candidates[selection[obj.instance]].object
        require(_same_object(obj, expected), "active model/hash/scale/role/readiness differs from maximum pool")
    return dict(roles)


def bind_variant(template, pool_path, selection):
    """Reference a full pool without embedding inactive models in the template."""
    path = Path(pool_path).resolve()
    pool = pool_from_dict(json.loads(path.read_text()))
    roles = validate_active_variant(template, pool, selection)
    reference = {"schema": REFERENCE_SCHEMA, "conceptual_template_id": pool.conceptual_template_id,
                 "pool_sha256": pool.fingerprint, "manifest": str(path), "manifest_sha256": file_digest(path),
                 "active_selection": dict(selection), "active_selection_sha256": digest(selection),
                 "active_role_counts": roles, "fixed_layout": pool.fixed_layout}
    return replace(template, scene_setup={**template.scene_setup, "binding_pool": reference})


def load_pool_for_template(template, *, root=None):
    reference = template.scene_setup.get("binding_pool")
    if reference is None:
        return None
    require(isinstance(reference, dict) and reference.get("schema") == REFERENCE_SCHEMA, "invalid active pool reference")
    path = Path(reference["manifest"])
    require(path.is_absolute(), "pool manifest must have an absolute immutable location")
    if root is not None:
        require(path.resolve().is_relative_to(Path(root).resolve()), "pool manifest outside campaign")
    require(file_digest(path) == reference["manifest_sha256"], "immutable maximum pool bytes changed")
    pool = pool_from_dict(json.loads(path.read_text()))
    require(pool.fingerprint == reference["pool_sha256"] and pool.conceptual_template_id == reference["conceptual_template_id"],
            "maximum pool identity changed")
    require(digest(reference["active_selection"]) == reference["active_selection_sha256"], "active selection digest changed")
    roles = validate_active_variant(template, pool, reference["active_selection"])
    require(roles == reference["active_role_counts"] and pool.fixed_layout == reference["fixed_layout"], "active pool summary changed")
    return pool


def conceptual_identity(template, *, root=None):
    """Pool identity for quota/template counting only, never a ValidationKey."""
    pool = load_pool_for_template(template, root=root)
    return digest({"schema": POOL_SCHEMA, "conceptual_template_id": pool.conceptual_template_id,
                   "pool_sha256": pool.fingerprint}) if pool else None


def audit_objects(template, *, root=None):
    """The entire admitted maximum pool participates in cross-role exclusion."""
    from scripts.campaign_20261110.camera_context import load_camera_context
    pool = load_pool_for_template(template, root=root)
    objects = pool.audit_template().objects if pool else template.objects
    return objects + load_camera_context(template, root=root)


def canonical_pool_perturbations(template, perturbations):
    """Default distractor copies are start variation, not new canonical goals."""
    pool = load_pool_for_template(template)
    require(pool is not None, "canonical pool perturbations require a binding pool")
    selection = template.scene_setup["binding_pool"]["active_selection"]
    changes = {}
    for name, state in perturbations.scales.items():
        if state.applied == state.default:
            continue
        identity = "candidate:"+selection[name] if name in selection else "fixed:"+name
        changes.setdefault(identity, set()).add(canonical_json(asdict(state)))
    return {"scales": {identity: [json.loads(value) for value in sorted(values)]
                       for identity, values in sorted(changes.items())},
            "removed_objects": list(perturbations.removed_objects), "lighting": perturbations.lighting}
