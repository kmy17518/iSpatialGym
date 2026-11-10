"""Campaign provenance and live scale checks, isolated from legacy runs.

Sampling opts in with ISPATIALGYM_CAMPAIGN_CONFIG. Replaying a campaign TRO
always checks its recorded scale, without depending on that environment flag.
Scene and lighting augmentation are deliberately rejected until their physical
adapters are installed; metadata cannot stand in for applying a perturbation.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path

from scripts.campaign_20261110.contracts import (
    ObjectRef, Perturbations, ScaleState, Template, ValidationKey,
    attach_tro_metadata, canonical_json, require,
)


def active() -> bool:
    return bool(os.environ.get("ISPATIALGYM_CAMPAIGN_CONFIG"))


@lru_cache(maxsize=32)
def _load(path: str):
    with open(path) as stream:
        config = json.load(stream)
    data = dict(config["template"])
    data["objects"] = tuple(ObjectRef(**obj) for obj in data["objects"])
    template = Template(**data)
    p = config["perturbations"]
    perturbations = Perturbations({key: ScaleState(**value) for key, value in p["scales"].items()},
                                 tuple(p.get("removed_objects", [])), p.get("lighting", {}))
    key = ValidationKey(**config["validation_key"])
    require(key.scale_sha256 == perturbations.scale_fingerprint, "validation key scale mismatch")
    require(config["split"] == template.split or (template.split == "eval_standard" and config["split"] == "train"),
            "campaign configuration split mismatch")
    perturbations.validate(config["split"], (obj.instance for obj in template.objects))
    _require_supported(perturbations)
    return config, template, perturbations, key


def configuration():
    return _load(str(Path(os.environ["ISPATIALGYM_CAMPAIGN_CONFIG"]).resolve()))


def training_soft_buckets() -> bool:
    return active() and configuration()[0]["split"] == "train"


def choose_soft_bucket(rng, buckets, counts):
    """Favor underfilled train buckets while keeping every bucket eligible.

    The sampler still checks the physical bounds of its chosen draw. Unlike eval,
    an already-filled bucket stays available when another bucket has low yield.
    """
    require(bool(buckets), "training bucket list is empty")
    maxima = [float(b["max_m"]) for b in buckets]
    targets = [max(0, int(b.get("n", 0))) for b in buckets]
    require(all(math.isfinite(v) and v > 0 for v in maxima), "invalid bucket bounds")
    deficits = [max(0, n - int(counts.get(hi, 0))) for n, hi in zip(targets, maxima)]
    # Reserve 20% of draws for uniform exploration even for large outstanding
    # quotas. A fixed pseudocount would vanish beside a 10,000-instance deficit.
    total = sum(deficits)
    probabilities = ([0.2 / len(buckets) + 0.8 * deficit / total for deficit in deficits]
                     if total else [1.0 / len(buckets)] * len(buckets))
    return int(rng.choice(len(buckets), p=probabilities))


def _require_supported(perturbations):
    require(not perturbations.removed_objects and not perturbations.lighting,
            "scene/light perturbations need an installed and verified runtime adapter")


def _vector(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (float, int)):
        value = [value] * 3
    result = [float(v) for v in value]
    require(len(result) == 3 and all(math.isfinite(v) and v > 0 for v in result), "invalid live object scale")
    return result


def verify_scales(objects, scales):
    observed = {}
    for name, state in scales.items():
        obj = objects.get(name)
        require(obj is not None, f"campaign task object absent: {name}")
        actual = _vector(obj.scale)
        expected = state.applied if isinstance(state, ScaleState) else state["applied"]
        require(all(abs(a - float(b)) <= 1e-6 * max(1.0, abs(float(b))) for a, b in zip(actual, expected)),
                f"campaign scale mismatch for {name}: expected {expected}, loaded {actual}")
        observed[name] = actual
    return observed


def annotate(sampler, tro, draft, seed):
    config, template, perturbations, key = configuration()
    require(sampler.scene == template.scene, "campaign template scene differs from sampler scene")
    for ref in template.objects:
        obj = sampler.objects.get(ref.instance)
        require(obj is not None, f"campaign binding missing: {ref.instance}")
        require(obj.category == ref.category and obj.model == ref.model,
                f"campaign asset identity mismatch: {ref.instance}")
    observed = verify_scales(sampler.objects, perturbations.scales)
    group = draft.summary.get("campaign_goal_group") or {}
    if group:
        require(group.get("campaign_id") == config["campaign_id"], "goal plan belongs to another campaign")
        require(group.get("split") == config["split"], "goal plan belongs to another split")
    result = attach_tro_metadata(tro, campaign_id=config["campaign_id"], split=config["split"],
                                 template=template, created_at=time.time(), seed=seed, perturbations=perturbations,
                                 validation_key=key, goal_group_id=group.get("group_id"))
    result["ispatialgym"]["campaign"]["runtime"] = {
        "observed_scales": observed, "goal_group": group,
        "purpose": config.get("purpose", "candidate"),
        "adapters": {"scale": "template_scene_with_live_verification", "scene": "none", "lighting": "none"},
    }
    return result


def verify_loaded(meta, physical):
    """Check a loaded campaign instance before any recorded action is executed."""
    require(meta.get("schema") == "isg-campaign-v1", "unknown campaign TRO schema")
    p = meta["perturbations"]
    perturbations = Perturbations({name: ScaleState(**value) for name, value in p["scales"].items()},
                                 tuple(p.get("removed_objects", [])), p.get("lighting", {}))
    perturbations.validate(meta["split"], physical)
    _require_supported(perturbations)
    require(meta["validation_key"]["scale_sha256"] == perturbations.scale_fingerprint,
            "TRO validation key scale mismatch")
    return {"observed_scales": verify_scales(physical, perturbations.scales), "scale_verified": True,
            "scene_augmentation": False, "lighting_augmentation": False}


def _plain(value):
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def record_failure(sampler, draft, *, index, attempt, seed, phase, error):
    config, template, perturbations, _ = configuration()
    group = (draft.summary.get("campaign_goal_group") or {}) if draft is not None else {}
    proposed = getattr(sampler, "_campaign_latest_proposal", None) or {}
    record = {
        "schema": "isg-campaign-failure-v1", "campaign_id": config["campaign_id"], "created_at": time.time(),
        "template_id": template.template_id, "class_id": template.class_id, "task": template.task,
        "index": index, "attempt": attempt, "seed": seed, "failure_kind": "sampling", "phase": phase,
        "reason": error.reason, "detail": str(error), "diagnostics": _plain(getattr(error, "details", {})),
        "scale_sha256": perturbations.scale_fingerprint, "goal_group_id": group.get("group_id", proposed.get("group_id")),
        "proposal_available": draft is not None or bool(proposed),
        "start_poses": _plain(draft.object_poses if draft is not None else proposed.get("start_poses")),
        "goal_poses": _plain(draft.goal_poses if draft is not None else proposed.get("goal_poses")),
        "start_joints": _plain(draft.object_joints) if draft is not None else None,
        "goal_joints": _plain(draft.goal_joints) if draft is not None else None,
        "base_pose2d": _plain(draft.base_pose2d if draft is not None else proposed.get("base_pose2d")),
    }
    path = Path(sampler.ctx.out_dir) / "campaign_sampling_failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(canonical_json(record) + "\n")
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)
