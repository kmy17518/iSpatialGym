"""Explicit retained proposal strategies for two verified campaign recipes.

These strategies rank the existing checked candidates. They neither synthesize
poses nor replace runtime goal geometry, native feasibility checks or the oracle.
The caller resolves the durable route and persists the returned trace in the
instance. No ambient environment variable selects a strategy here.
"""
from __future__ import annotations

import hashlib
import json
import math


_ROUTES = {
    "attachment03_retained_source_stance_v1": {
        "class_id": "configuration_matching-attachment_03",
        "task": "configuration_matching-attachment",
        "template_sha256": "40c59c6a6af0fc4c2f4e535f9ed0d45e9dcdb4f1243a74685c05e02f58e14a1a",
        "scale_sha256": "9b41364f7f868e8dc08102685fd964c1b2461eebcb33ae6d0c82ab04164b153e",
        "source_revision": "1af23ba8537b",
        "source_group": "standard_recovery_attachment03_manipulation_hint_r3",
        "historical_max_draws": 32,
    },
    "under_small03_retained_source_stance_v1": {
        "class_id": "inter_object_relation_satisfaction-under_small_scale_03",
        "task": "inter_object_relation_satisfaction-under_small_scale",
        "template_sha256": "68b7fa5314a82859ef4b0f7e934295b56f06ee6fc120fc10d0dc279357580b80",
        "scale_sha256": "6be4910af7cd0cf1fcf319fb3a99e55f88ec4d7ff2c4113f8efc33bc0a55d5d8",
        "source_revision": "fd95d329793e",
        "source_group": "current100_r1_d0458fd30b8d6c26325d",
        "historical_max_draws": 15,
    },
}
DATA_SHA = "759a02240d5175b78036137e5e74ff92c59a6ac858b5535d24105839ab7219a9"
PHYSICS_SHA = "74f1aba526a6a9080301a3a45df9be47011dcc3da5404196b04d0fbd550ef1a9"

ORIGIN_EVIDENCE = {
    "path": "/vision/u/minyeong/projects/isg_home/campaign_20261110/diagnostics/accepted_class_recipe_recovery_r1/origin_evidence_r1.json",
    "sha256": "26781c1a5b7d0a42eee51ca98be8fe46bef64bd0081f4a760643d2e6b6ff844a",
}
COMPARISON_EVIDENCE = {
    "path": "/vision/u/minyeong/projects/isg_home/campaign_20261110/diagnostics/accepted_class_recipe_recovery_r1/comparison_r1.json",
    "sha256": "2ce9ae87a7f5443b1c98ef1709818c06d99cedcfebab764b5129f5c36b999f97",
}

PROFILES = {
    value["class_id"]: dict(value, route_id=route_id, version=1,
        source_evidence=ORIGIN_EVIDENCE, comparison_evidence=COMPARISON_EVIDENCE,
        historical_validation_is_current_acceptance=False,
        applicability={"splits": ["eval_standard", "train"], "template_sha256": value["template_sha256"],
                       "scale_sha256": value["scale_sha256"], "data_sha256": DATA_SHA,
                       "physics_sha256": PHYSICS_SHA, "variant_inheritance": "exact_reviewed_binding_only"})
    for route_id, value in _ROUTES.items()
}


#: train 2026-11-14 (train_r4): a train sub-template of a recipe class that is not the reviewed canonical selection (another
#: scene / room / model / scale) keeps the recipe - its physics and data fingerprints, its goal plans and the native checks are
#: unchanged - and binds the profile by class + task + recipe applicability instead of the eval template/scale fingerprints.
TRAIN_VARIANT_BINDING = "train_class_recipe_v1"


def _binding(config, class_id, task, template_sha256, scale_sha256):
    """``"exact"`` (the reviewed canonical selection, eval or train), ``TRAIN_VARIANT_BINDING`` (a train sub-template of the
    class with the recipe's data/physics fingerprints) or None (incompatible)."""
    template = config.get("template", {})
    key = config.get("validation_key", {})
    if (config.get("split") not in ("eval_standard", "train") or template.get("class_id") != class_id
            or template.get("task") != task or key.get("data_sha256") != DATA_SHA or key.get("physics_sha256") != PHYSICS_SHA):
        return None
    if key.get("template_sha256") == template_sha256 and key.get("scale_sha256") == scale_sha256:
        return "exact"
    if config.get("split") == "train" and template.get("split") in ("train", "eval_standard") and all(
            isinstance(key.get(k), str) and len(key[k]) == 64 for k in ("template_sha256", "scale_sha256")):
        return TRAIN_VARIANT_BINDING
    return None


def profile_for_config(config):
    """Resolve selected canonical profiles without importing a simulator. A train sub-template other than the canonical
    selection gets the same recipe with its binding recorded (``train_variant``); eval binds the exact fingerprints only."""
    profile = PROFILES.get(config.get("template", {}).get("class_id"))
    if profile is None:
        return None
    binding = _binding(config, profile["class_id"], profile["task"], profile["template_sha256"], profile["scale_sha256"])
    if binding is None:
        raise ValueError("selected class has no compatible retained recipe profile")
    result = json.loads(json.dumps(profile))
    if binding == TRAIN_VARIANT_BINDING:
        key = config["validation_key"]
        result["train_variant"] = {"binding": TRAIN_VARIANT_BINDING, "template_sha256": key["template_sha256"],
                                   "scale_sha256": key["scale_sha256"], "split": "train",
                                   "rule": "class + task + recipe data/physics fingerprints; goal plan bound to this template"}
    return result


def _value(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_recipe_inputs(route_id, config, plan, *, phase, trace):
    """Fail closed on incompatible variants and record the actual phase call.

    Template/scale digests bind the canonical cork/bottle/desk or nonuniform
    plate/place-mat/table selection. Other variants need a separately reviewed
    profile; they cannot inherit canonical artifacts solely from their class.
    The normal campaign loader remains responsible for verifying decoded inputs
    against these fingerprints and for native library/map lookup validation.
    """
    if route_id not in _ROUTES:
        raise ValueError("unknown retained class recipe")
    if phase not in ("sampling", "oracle", "replay"):
        raise ValueError("unknown retained recipe phase")
    if trace is None or not callable(getattr(trace, "append", None)):
        raise ValueError("retained recipe requires a persistent execution trace sink")
    profile = _ROUTES[route_id]
    key = config.get("validation_key", {})
    binding = _binding(config, profile["class_id"], profile["task"], profile["template_sha256"], profile["scale_sha256"])
    # the goal plan belongs to the configuration's own template (the reviewed one, or the train sub-template's)
    plan_template = profile["template_sha256"] if binding == "exact" else key.get("template_sha256")
    if binding is None or _value(plan, "contract_template_sha256") != plan_template:
        raise ValueError("retained recipe does not match exact reviewed configuration")
    event = {"schema": "isg-retained-class-recipe-trace-v1", "route_id": route_id,
             "phase": phase, "class_id": profile["class_id"],
             "configuration_sha256": _digest(config),
             "template_sha256": key["template_sha256"], "scale_sha256": key["scale_sha256"],
             "plan_sha256": _value(plan, "sha256") or _digest(plan),
             "source_revision": profile["source_revision"], "source_group": profile["source_group"],
             "native_checks_unchanged": True,
             **({"train_variant_binding": binding} if binding != "exact" else {}),
             "action": "binding_verified" if phase == "sampling" else ("oracle_native_unchanged" if phase == "oracle" else "replay_recorded_route_verified")}
    trace.append(event)
    return event


def _hint(value):
    if not isinstance(value, dict):
        raise ValueError("retained source stance is missing")
    for key, size in (("base_pose2d", 3), ("focus", 3)):
        v = value.get(key)
        if not isinstance(v, (list, tuple)) or len(v) != size or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v):
            raise ValueError("invalid retained source stance " + key)
    for key in ("translate", "tilt"):
        v = value.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise ValueError("invalid retained source stance " + key)
    return value


def _rank(order, candidates, hints, event):
    # Exact historical prioritize_written_stances policy: four ordinary draws,
    # then two nearest source/written candidates, followed by the remaining pool.
    from ispatialgym.sampling.campaign_tabletop_starts import prioritize_written_stances
    hints = [_hint(value) for value in hints]
    before = [int(i) for i in order]
    if len(set(before)) != len(before) or any(i < 0 or i >= len(candidates) for i in before):
        raise ValueError("invalid native candidate permutation")
    result = prioritize_written_stances(before, candidates, hints, explore=4, exploit=2)
    if sorted(result) != sorted(before):
        raise ValueError("retained strategy changed native candidate membership")
    event.update(action="source_stance_ranked", candidate_count=len(before),
                 input_order_sha256=_digest(before), output_order_sha256=_digest(result),
                 hint_sha256=_digest(hints), explore=4, exploit=2)
    return result


def attachment03_order_stances(route_id, config, plan, order, candidates, *, trace):
    if route_id != "attachment03_retained_source_stance_v1":
        raise ValueError("attachment strategy received a different route")
    event = validate_recipe_inputs(route_id, config, plan, phase="sampling", trace=trace)
    return _rank(order, candidates, [_value(plan, "anchor_manipulation_stance")], event)


def under_small03_order_stances(route_id, config, plan, order, candidates, written=(), *, trace):
    if route_id != "under_small03_retained_source_stance_v1":
        raise ValueError("under strategy received a different route")
    event = validate_recipe_inputs(route_id, config, plan, phase="sampling", trace=trace)
    hints = list(written) or [_value(plan, "anchor_manipulation_stance")]
    event["hint_source"] = "current_group_written_member" if written else "current_plan_anchor"
    return _rank(order, candidates, hints, event)
