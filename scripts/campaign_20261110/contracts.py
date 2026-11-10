"""Simulator-free, fail-closed contracts for the fresh November 10 campaign.

This module does not change legacy sampling, oracle execution, or simulator loading.
The campaign adapter must apply and verify a restoration plan before recording a
successful validation. Metadata alone is never evidence of simulator support.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "isg-campaign-v1"
SPLITS = frozenset({"train", "eval_standard", "eval_model", "eval_category"})
ROLES = frozenset({"target", "partner", "support", "distractor"})
SCENES = frozenset({"house_single_floor", "house_double_floor_upper", "house_double_floor_lower",
                    "Rs_int", "hotel_suite_large", "restaurant_diner", "office_cubicles_right"})
PENDING_ALIASES = frozenset({"inter_object_relation_satisfaction-touching_03",
                            "inter_object_relation_satisfaction-under_large_scale_01",
                            "inter_object_relation_satisfaction-under_large_scale_02",
                            "inter_object_relation_satisfaction-under_large_scale_03",
                            "inter_object_relation_satisfaction-nextto_04"})


class ContractError(ValueError):
    """An incomplete or inconsistent record must not enter production."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _sha(value: str, name: str) -> None:
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            f"{name} must be a full SHA-256 digest")


def _scale(value: Sequence[float]) -> tuple[float, float, float]:
    require(len(value) == 3, "scale must contain three absolute dimensions")
    result = tuple(float(x) for x in value)
    require(all(math.isfinite(x) and x > 0 for x in result), "scale dimensions must be finite and positive")
    return result


@dataclass(frozen=True)
class TaskQuota:
    task: str
    standard_classes: int
    train_instances: int
    model_instances_per_class: int = 9
    category_instances: int | None = None
    standard_buckets: tuple[int, ...] = ()
    model_buckets: tuple[int, ...] = ()
    category_buckets: tuple[int, ...] = ()
    goal_groups: bool = True
    model_bucket_policy: Mapping[str, Any] = field(default_factory=dict)

    @property
    def model_classes(self) -> int:
        return 0 if self.task == "camera_relocalization-standard" else self.standard_classes


def task_quotas(*, articulation_category_count: str | None = "stated_totals",
                block_category_instances: int | None = 1) -> tuple[TaskQuota, ...]:
    """Owner-resolved November 10 quotas; explicit None preserves a pending choice.

    Owner chose block model holdout nine/category one, and category articulation
    two per small-scale task/three per large-scale task. Axial has five classes.
    """
    require(articulation_category_count in (None, "stated_totals", "balanced_four"),
            "unknown articulation category count decision")
    if block_category_instances is not None:
        require(block_category_instances > 0, "block category count must be positive")
    q = [
        TaskQuota("camera_relocalization-standard", 1, 1000, 0, 10, (3, 4, 3), (), (3, 4, 3), False),
        TaskQuota("camera_relocalization-obstructed", 9, 9000, 9, 9, (3, 4, 3), (3, 3, 3), (3, 3, 3), False),
        TaskQuota("object_scaling", 10, 10000, 9, 10, (3, 4, 3), (3, 3, 3), (3, 4, 3)),
        TaskQuota("mental_rotation", 10, 10000, 9, 10, (3, 4, 3), (3, 3, 3), (3, 4, 3)),
        TaskQuota("disembedding", 10, 10000, 9, 10, (3, 4, 3), (3, 3, 3), (3, 4, 3)),
        TaskQuota("alignment-world_vertical", 5, 5000, 9, 5, (3, 4, 3), (3, 3, 3), (2, 2, 1)),
        TaskQuota("alignment-axial", 5, 5000, 9, 5, (3, 4, 3), (3, 3, 3), (2, 2, 1)),
        TaskQuota("configuration_matching-block_design", 1, 1000, 9, block_category_instances,
                  (2, 3, 3, 2), (2, 3, 2, 2), model_bucket_policy={
                      "method": "proportional_largest_remainder", "source_counts": [2, 3, 3, 2],
                      "target_total": 9, "tie_break": "lower_bucket_index"}),
        TaskQuota("configuration_matching-attachment", 4, 4000, 9, 4),
    ]
    for scale, count in (("small", 2), ("large", 3)):
        n = None if articulation_category_count is None else (count if articulation_category_count == "stated_totals" else 4)
        buckets = () if n is None else ((1, 1) if n == 2 else (2, 1) if n == 3 else (2, 2))
        for direction in ("close", "open"):
            q.append(TaskQuota(f"configuration_matching-articulation_{direction}_{scale}_scale", count,
                               count * 1000, 9, n, (5, 5), (5, 4), buckets))
    for scale, count in (("small", 4), ("large", 3)):
        for relation in ("ontop", "under", "inside", "nextto", "touching"):
            q.append(TaskQuota(f"inter_object_relation_satisfaction-{relation}_{scale}_scale", count,
                               count * 1000, 9, count))
    return tuple(q)


TASKS = frozenset(q.task for q in task_quotas())


def task_for_alias(alias: str) -> str:
    """Normalize legacy small IORS aliases without guessing a new task identity."""
    for task in sorted(TASKS, key=len, reverse=True):
        if alias == task or alias.startswith(task + "_") or alias.startswith(task + "-"):
            return task
    for relation in ("inside", "nextto", "touching"):
        prefix = f"inter_object_relation_satisfaction-{relation}"
        if alias == prefix or alias.startswith(prefix + "_") or alias.startswith(prefix + "-"):
            return prefix + "_small_scale"
    raise ContractError(f"unknown task alias {alias!r}")


@dataclass(frozen=True)
class ObjectRef:
    instance: str
    role: str
    category: str
    model: str
    asset_sha256: str
    default_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    joint: str | None = None
    semantic_kind: str = "object"
    ready: bool = False
    readiness_source: str = ""

    def __post_init__(self) -> None:
        require(self.role in ROLES, f"unknown role {self.role!r}")
        require(self.semantic_kind in ("object", "tray", "board", "wall", "floor"), "unknown semantic kind")
        require(isinstance(self.ready, bool), "readiness must be explicit boolean")
        require(not self.ready or bool(self.readiness_source), "ready object needs readiness provenance")
        require(bool(self.instance and self.category and self.model), "object identity is incomplete")
        _sha(self.asset_sha256, "asset_sha256")
        object.__setattr__(self, "default_scale", _scale(self.default_scale))

    @property
    def model_key(self) -> str:
        return f"{self.category}/{self.model}"


@dataclass(frozen=True)
class Template:
    template_id: str
    task: str
    class_id: str
    split: str
    scene: str
    room: str
    objects: tuple[ObjectRef, ...]
    definition_sha256: str
    scene_setup: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(self.task in TASKS, f"unknown task {self.task!r}")
        require(self.split in SPLITS, f"unknown split {self.split!r}")
        require(self.scene in SCENES, f"scene {self.scene!r} is outside the seven scenes")
        require(bool(self.template_id and self.class_id and self.room), "template identity is incomplete")
        require(len({o.instance for o in self.objects}) == len(self.objects), "duplicate object instance")
        require(all(o.ready and o.readiness_source for o in self.objects), "template contains an object without confirmed readiness")
        if self.task != "camera_relocalization-standard":
            require(any(o.role == "target" for o in self.objects), "template has no target")
        _sha(self.definition_sha256, "definition_sha256")
        canonical_json(self.scene_setup)

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))


def audit_splits(templates: Iterable[Template]) -> list[str]:
    """Check held-out resources globally across all tasks and all four roles.

    Sharing within a holdout pool is allowed. Asset hashes also catch a copied
    model renamed into another category. Structural floors/walls are exempt from
    model/category checks; trays/boards are exempt only from category checks
    (owner November 10). Rooms are checked regardless of these exceptions.
    """
    pools: dict[str, dict[str, set[str]]] = {
        split: {k: set() for k in ("models", "hashes", "categories", "rooms")} for split in SPLITS
    }
    seen: dict[str, str] = {}
    errors = []
    for t in templates:
        if t.template_id in seen and seen[t.template_id] != t.fingerprint:
            errors.append(f"template id has conflicting definitions: {t.template_id}")
        seen[t.template_id] = t.fingerprint
        p = pools[t.split]
        rooms = set(t.scene_setup.get("task_rooms", ())) | {t.room}
        require(all(isinstance(room, str) and room for room in rooms), "invalid template task_rooms")
        p["rooms"].update(t.scene + "/" + room for room in rooms)
        for obj in t.objects:
            if obj.semantic_kind in ("wall", "floor"):
                continue
            p["models"].add(obj.model_key)
            p["hashes"].add(obj.asset_sha256)
            if obj.semantic_kind not in ("tray", "board"):
                p["categories"].add(obj.category)
    held_in = {key: pools["train"][key] | pools["eval_standard"][key] for key in pools["train"]}
    for key in ("models", "hashes"):
        other = held_in[key] | pools["eval_category"][key]
        for overlap in sorted(pools["eval_model"][key] & other):
            errors.append(f"eval_model {key} leak across roles/tasks: {overlap}")
    for key in ("categories", "models", "hashes", "rooms"):
        other = held_in[key] | pools["eval_model"][key]
        for overlap in sorted(pools["eval_category"][key] & other):
            errors.append(f"eval_category {key} leak across roles/tasks: {overlap}")
    return errors


@dataclass(frozen=True)
class ScaleState:
    default: tuple[float, float, float]
    applied: tuple[float, float, float]
    mode: str = "uniform"

    def __post_init__(self) -> None:
        object.__setattr__(self, "default", _scale(self.default))
        object.__setattr__(self, "applied", _scale(self.applied))
        require(self.mode in ("uniform", "height", "fixed"), "unknown scale mode")
        ratio = [a / d for a, d in zip(self.applied, self.default)]
        if self.mode == "height":
            require(ratio[:2] == [1.0, 1.0], "height perturbation must preserve x/y scale")
        elif self.mode == "uniform":
            require(max(ratio) - min(ratio) < 1e-12, "uniform perturbation must preserve aspect ratio")
        else:
            require(self.applied == self.default, "fixed scale cannot change")
        require(all(0.8 - 1e-12 <= x <= 1.2 + 1e-12 for x in ratio), "scale perturbation outside [0.8, 1.2]")

    @property
    def perturbed(self) -> bool:
        return self.default != self.applied


def scale_candidates(default: Sequence[float], *, seed: int, height_only: bool = False) -> tuple[ScaleState, ...]:
    """Two reproducible perturbations relative to the actual default, not 1.0."""
    base = _scale(default)
    rng = random.Random(seed)
    result = []
    for _ in range(2):
        factor = rng.uniform(0.8, 1.2)
        applied = (base[0], base[1], base[2] * factor) if height_only else tuple(x * factor for x in base)
        result.append(ScaleState(base, applied, "height" if height_only else "uniform"))
    return tuple(result)


@dataclass(frozen=True)
class Perturbations:
    scales: Mapping[str, ScaleState] = field(default_factory=dict)
    removed_objects: tuple[str, ...] = ()
    lighting: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, split: str, relevant_objects: Iterable[str]) -> None:
        require(split in SPLITS, "unknown split")
        require(not (set(self.removed_objects) & set(relevant_objects)), "cannot remove a task-relevant object")
        require(len(set(self.removed_objects)) == len(self.removed_objects), "duplicate removed object")
        canonical_json(asdict(self))  # Reject NaN and non-JSON metadata before it reaches the simulator.
        if split != "train":
            require(not self.perturbed, "evaluation instances must have no perturbations")

    @property
    def perturbed(self) -> bool:
        return any(s.perturbed for s in self.scales.values()) or bool(self.removed_objects or self.lighting)

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))

    @property
    def scale_fingerprint(self) -> str:
        return digest({name: asdict(state) for name, state in self.scales.items()})


def scale_cache_key(obj: ObjectRef, scale: ScaleState, *, support_height_m: float | None,
                    robot_sha256: str, builder_sha256: str, physics_sha256: str) -> str:
    """Exact absolute scale/height keys prevent reuse of incompatible grasps/maps."""
    require(scale.default == obj.default_scale, "cache key default differs from object default")
    for value, name in ((robot_sha256, "robot"), (builder_sha256, "builder"), (physics_sha256, "physics")):
        _sha(value, name)
    if support_height_m is not None:
        require(math.isfinite(support_height_m) and support_height_m > 0, "invalid support height")
    return digest({"asset": obj.asset_sha256, "model": obj.model_key, "joint": obj.joint,
                   "absolute_scale": scale.applied, "support_height_m": support_height_m,
                   "robot": robot_sha256, "builder": builder_sha256, "physics": physics_sha256})


@dataclass(frozen=True)
class ValidationKey:
    template_sha256: str
    code_sha256: str
    physics_sha256: str
    data_sha256: str
    scale_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _sha(value, name)

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class ValidationEvidence:
    key: ValidationKey
    kind: str
    success: bool
    qa_passed: bool
    created_at: float
    report_path: str
    report_sha256: str
    demo_sha256: str
    instance_id: str
    restoration_verified: bool

    def __post_init__(self) -> None:
        require(self.kind in ("demo", "open_loop_replay"), "validation must be demo or open-loop replay")
        require(math.isfinite(self.created_at) and self.created_at > 0, "invalid evidence time")
        _sha(self.report_sha256, "report_sha256")
        _sha(self.demo_sha256, "demo_sha256")
        require(bool(self.report_path and self.instance_id), "validation evidence has no report/instance")


def require_validation(key: ValidationKey, evidence: Iterable[ValidationEvidence], *, campaign_started_at: float,
                       verify_artifacts: bool = True) -> tuple[ValidationEvidence, ValidationEvidence]:
    """One successful fresh demo and its successful action replay unlock a template.

    Both reports must refer to the exact same demo and the requested code,
    template, maps, physics, and scale. A replay of a different demo is not a gate.
    """
    eligible = []
    for e in evidence:
        if (e.key != key or not e.success or not e.qa_passed or not e.restoration_verified
                or e.created_at < campaign_started_at):
            continue
        if verify_artifacts:
            require(Path(e.report_path).is_file(), f"missing validation report {e.report_path}")
            require(file_digest(e.report_path) == e.report_sha256, f"modified validation report {e.report_path}")
        eligible.append(e)
    for demo in eligible:
        if demo.kind != "demo":
            continue
        for replay in eligible:
            if (replay.kind == "open_loop_replay" and replay.demo_sha256 == demo.demo_sha256
                    and replay.instance_id == demo.instance_id and replay.created_at >= demo.created_at):
                return demo, replay
    raise ContractError("production requires one fresh successful demo and its verified successful open-loop replay")


@dataclass(frozen=True)
class GroupMember:
    instance_id: str
    campaign_id: str
    template_sha256: str
    goal_sha256: str
    start_grasp_sha256: str
    perturbations: Perturbations
    created_at: float
    demo_success: bool


def allocate_goal_group(members: Sequence[GroupMember], *, campaign_id: str, campaign_started_at: float,
                        standard_eval_requested: bool, max_members: int = 3) -> dict[str, str]:
    """Promote exactly one unperturbed member only after a distinct sibling exists.

    Call when the group budget expires or the group fills. One success is useful
    training data. This never relabels a historical instance or copies one.
    """
    require(1 <= len(members) <= max_members <= 5, "goal-group size is outside the configured limit")
    require(len({m.instance_id for m in members}) == len(members), "duplicate instance in goal group")
    require(len({m.template_sha256 for m in members}) == 1, "goal group crosses templates")
    require(len({m.goal_sha256 for m in members}) == 1, "goal group has different goals")
    require(len({m.perturbations.fingerprint for m in members}) == 1, "goal group has inconsistent perturbations")
    require(len({m.start_grasp_sha256 for m in members}) == len(members), "goal group duplicates a start/grasp configuration")
    for m in members:
        require(m.campaign_id == campaign_id and m.created_at >= campaign_started_at, "historical instance in fresh campaign")
        require(m.demo_success, "unsuccessful demo cannot enter an instance split")
        for value, name in ((m.template_sha256, "template"), (m.goal_sha256, "goal"), (m.start_grasp_sha256, "start/grasp")):
            _sha(value, name)
    result = {m.instance_id: "train" for m in members}
    if standard_eval_requested and len(members) >= 2:
        require(not any(m.perturbations.perturbed for m in members), "eval goal group must remain unperturbed")
        result[min(members, key=lambda m: m.instance_id).instance_id] = "eval_standard"
    return result


def group_budget_seconds(successful_attempt_seconds: Sequence[float], *, minimum: float = 300,
                         maximum: float = 1800, multiplier: float = 3) -> float:
    """Bound one group independently of each sampler attempt's existing limit."""
    import statistics
    require(successful_attempt_seconds and all(math.isfinite(x) and x > 0 for x in successful_attempt_seconds),
            "group budget needs at least one measured successful attempt")
    require(0 < minimum <= maximum and multiplier > 0, "invalid group budget settings")
    return min(maximum, max(minimum, multiplier * statistics.median(successful_attempt_seconds)))


def attach_tro_metadata(tro: Mapping[str, Any], *, campaign_id: str, split: str, template: Template,
                        created_at: float, seed: int, perturbations: Perturbations,
                        validation_key: ValidationKey, goal_group_id: str | None = None) -> dict[str, Any]:
    """Return a copy with metadata; call BEFORE the oracle sidecar hashes the TRO."""
    require("ispatialgym" in tro and isinstance(tro["ispatialgym"], Mapping), "not an iSpatialGym TRO")
    require(bool(campaign_id) and math.isfinite(created_at) and created_at > 0, "incomplete campaign provenance")
    require(isinstance(seed, int) and seed >= 0, "invalid seed")
    require(split == template.split or (template.split == "eval_standard" and split == "train"),
            "instance split conflicts with template split")
    require(validation_key.template_sha256 == template.fingerprint, "validation key is for another template")
    require(validation_key.scale_sha256 == perturbations.scale_fingerprint, "validation key is for another scale configuration")
    perturbations.validate(split, (o.instance for o in template.objects))
    require(set(perturbations.scales) == {o.instance for o in template.objects},
            "persist exact default/applied scale for every task-relevant object")
    for obj in template.objects:
        require(perturbations.scales[obj.instance].default == obj.default_scale,
                f"default scale mismatch for {obj.instance}")
    result = copy.deepcopy(dict(tro))
    result["ispatialgym"]["campaign"] = {
        "schema": SCHEMA, "campaign_id": campaign_id, "split": split, "task": template.task,
        "class_id": template.class_id, "template_id": template.template_id,
        "template_sha256": template.fingerprint, "created_at": created_at, "seed": seed,
        "goal_group_id": goal_group_id, "perturbations": asdict(perturbations),
        "scene_setup": copy.deepcopy(dict(template.scene_setup)),
        "validation_key": asdict(validation_key),
        "restoration_order": ["scene_setup", "scales", "object_states", "removed_objects", "lighting"],
    }
    # The live sampler's object states contain tensors. The established TRO
    # writer applies TorchEncoder to those; only our new metadata is JSON here.
    validate_campaign_block(result["ispatialgym"]["campaign"], object_bindings=result["ispatialgym"].get("object_bindings"))
    return result


def validate_campaign_block(block: Mapping[str, Any], *, object_bindings: Mapping[str, Any] | None = None) -> None:
    """Validate namespaced TRO provenance without importing the simulator.

    Called by the actual TRO parser as well as the campaign writer. Binding keys
    are BDDL instance names, never scene-object names. The existing loader still
    has to verify live state; valid provenance is not restoration evidence.
    """
    require(isinstance(block, Mapping), "campaign metadata must be an object")
    require(block.get("schema") == SCHEMA, "unknown campaign metadata schema")
    for name in ("campaign_id", "template_id", "class_id"):
        require(isinstance(block.get(name), str) and bool(block[name].strip()), f"invalid campaign {name}")
    require(block.get("task") in TASKS, "invalid campaign task")
    require(block.get("split") in SPLITS, "invalid campaign split")
    _sha(block.get("template_sha256"), "template_sha256")
    created = block.get("created_at")
    require(isinstance(created, (float, int)) and not isinstance(created, bool)
            and math.isfinite(created) and created > 0, "invalid campaign creation time")
    seed = block.get("seed")
    require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, "invalid campaign seed")
    group = block.get("goal_group_id")
    require(group is None or (isinstance(group, str) and bool(group.strip())), "invalid goal group id")
    require(isinstance(block.get("validation_key"), Mapping), "missing validation key")
    require(set(block["validation_key"]) == set(ValidationKey.__dataclass_fields__), "invalid validation-key fields")
    key = ValidationKey(**block["validation_key"])
    require(key.template_sha256 == block["template_sha256"], "campaign template hash differs from validation key")
    p = block.get("perturbations")
    require(isinstance(p, Mapping) and set(p) == {"scales", "removed_objects", "lighting"}, "invalid perturbation fields")
    require(isinstance(p["scales"], Mapping), "missing per-object scale states")
    scales = {}
    for name, state in p["scales"].items():
        require(isinstance(name, str) and bool(name) and isinstance(state, Mapping), "invalid object scale record")
        require(set(state) == {"default", "applied", "mode"}, f"incomplete scale record for {name}")
        scales[name] = ScaleState(**state)
    require(isinstance(p["removed_objects"], (list, tuple)) and all(isinstance(x, str) and x for x in p["removed_objects"]),
            "invalid removed-object list")
    require(isinstance(p["lighting"], Mapping), "invalid lighting record")
    perturbations = Perturbations(scales, tuple(p["removed_objects"]), p["lighting"])
    if object_bindings is not None:
        require(isinstance(object_bindings, Mapping), "invalid TRO object bindings")
        structural = {name for name, binding in object_bindings.items()
                      if (isinstance(binding, Mapping) and binding.get("category") in ("floor", "floors", "wall", "walls"))
                      or re.fullmatch(r"(?:floor|wall)\.n\.01_\d+", name)}
        require(set(scales) <= set(object_bindings), "campaign scale names are not TRO object bindings")
        require(set(object_bindings) - structural <= set(scales),
                "campaign scales must cover every nonstructural TRO object binding")
    perturbations.validate(block["split"], scales)
    require(key.scale_sha256 == perturbations.scale_fingerprint, "campaign scales differ from validation key")
    require(isinstance(block.get("scene_setup"), Mapping), "missing deterministic scene setup")
    require(block.get("restoration_order") == ["scene_setup", "scales", "object_states", "removed_objects", "lighting"],
            "invalid campaign restoration order")
    if "runtime" in block:
        require(isinstance(block["runtime"], Mapping), "invalid campaign runtime record")
    canonical_json(block)


def restoration_plan(tro: Mapping[str, Any]) -> dict[str, Any]:
    """Decode desired restoration; caller must execute it, then verify live state.

    A legacy loader does not call this function. Publishing this metadata as a
    completed perturbation implementation would be incorrect.
    """
    meta = tro.get("ispatialgym", {}).get("campaign")
    require(isinstance(meta, dict) and meta.get("schema") == SCHEMA, "missing campaign restoration metadata")
    validate_campaign_block(meta, object_bindings=tro["ispatialgym"].get("object_bindings"))
    p = meta["perturbations"]
    perturbations = Perturbations({k: ScaleState(**v) for k, v in p["scales"].items()},
                                 tuple(p["removed_objects"]), p["lighting"])
    perturbations.validate(meta["split"], tro["ispatialgym"].get("object_bindings", {}))
    return {"order": list(meta["restoration_order"]), "scene_setup": copy.deepcopy(meta["scene_setup"]),
            "perturbations": asdict(perturbations),
            "fingerprint": perturbations.fingerprint}


def write_json_atomic(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=target.parent, prefix=target.name + ".", delete=False) as stream:
        temp = stream.name
        try:
            stream.write(canonical_json(data) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            os.unlink(temp)
            raise
    os.replace(temp, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articulation-category-count", choices=("stated_totals", "balanced_four"), default="stated_totals")
    parser.add_argument("--block-category-instances", type=int, default=1)
    args = parser.parse_args()
    quotas = task_quotas(articulation_category_count=args.articulation_category_count,
                         block_category_instances=args.block_category_instances)
    pending = [q.task for q in quotas if q.category_instances is None]
    print(json.dumps({"schema": SCHEMA, "tasks": [asdict(q) for q in quotas],
                      "totals": {"tasks": len(quotas), "standard_classes": sum(q.standard_classes for q in quotas),
                                 "standard_instances": 10 * sum(q.standard_classes for q in quotas),
                                 "model_classes": sum(q.model_classes for q in quotas),
                                 "model_instances": sum(q.model_classes * q.model_instances_per_class for q in quotas),
                                 "category_classes": len(quotas),
                                 "category_instances": None if pending else sum(q.category_instances for q in quotas),
                                 "train_instances": sum(q.train_instances for q in quotas)},
                      "pending_category_counts": pending}, indent=2))


if __name__ == "__main__":
    main()
