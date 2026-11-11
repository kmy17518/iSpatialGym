"""Persist exact common-scale draws for mating and custom multi-object tasks.

These identities resolve preparation inputs only. They do not prove mating,
layout, reachability, task semantics, or successful native validation.
"""
from __future__ import annotations

import hashlib
import json
import math
import random


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def coupled_profile(kind, objects):
    """Use the established attachment identity; other families get distinct seeds."""
    if kind not in ("attachment", "block_design", "puzzle"):
        raise ValueError("unsupported coupled scale family")
    if "target" not in objects or len(objects) < 2:
        raise ValueError("coupled profile requires target and companion objects")
    if kind == "attachment" and set(objects) != {"target", "partner"}:
        raise ValueError("attachment profile requires exactly target and partner")
    if kind == "block_design" and set(objects) != {"target", "board"}:
        raise ValueError("block profile requires exactly target and board")
    identity = {}
    for role, obj in sorted(objects.items()):
        scale = obj["default_scale"]
        if len(scale) != 3 or not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in scale):
            raise ValueError("invalid measured native default scale")
        identity[role] = {k: obj[k] for k in ("category", "model", "asset_sha256", "default_scale")}
    key = digest(identity)
    # Exactly matches training_attachment_dependencies.coupled_profile so a
    # larger inventory cannot silently replace the four already-authored pairs.
    seed = {"campaign": "campaign_20261110", "kind": kind + "_coupled_scale", "pair": key}
    rng = random.Random(int(digest(seed), 16))
    factors = [rng.uniform(.8, 1.2), rng.uniform(.8, 1.2)]
    scales = {role: [[round(float(v) * f, 4) for v in obj["default_scale"]] for f in factors]
              for role, obj in identity.items()}
    if any(value[0] == value[1] for value in scales.values()):
        raise ValueError("scale quantization merged distinct coupled draws")
    return {"id": key, "kind": kind, "objects": identity,
            "sampled_uniform_factors": factors, "absolute_scales": scales,
            "maximum_absolute_scales": {role: [round(float(v) * 1.2, 4) for v in obj["default_scale"]]
                                        for role, obj in identity.items()},
            "precision_rule": "native v2 absolute scale precision 1e-4; one relative multiplier for every coupled object",
            "physical_validation": "unverified"}


def candidate_profile(candidate, assets):
    """Return role identities and a profile only after every native default exists."""
    task = candidate["task"]
    refs = []
    if task == "configuration_matching-attachment":
        kind = "attachment"
        refs = [("target", candidate.get("target")), ("partner", candidate.get("partner"))]
    elif task == "configuration_matching-block_design":
        kind = "block_design"
        boards = [r for r in candidate.get("extra_roles", []) if r["role"] == "board"]
        if len(boards) != 1:
            return None, {}
        refs = [("target", candidate.get("target")), ("board", boards[0])]
    elif task == "disembedding" and (candidate.get("target") or {}).get("category") == "jigsaw_puzzle_piece":
        kind = "puzzle"
        refs = [("target", candidate["target"])]
        refs += [(r["role"] + ":" + r["category"] + "/" + r["model"], r)
                 for r in candidate.get("extra_roles", []) if r["category"] == "jigsaw_puzzle_piece"]
        if len(refs) < 2:
            return None, {}
    else:
        return None, {}
    objects, role_keys = {}, {}
    for role_key, obj in refs:
        if not obj:
            return None, {}
        key = obj["category"] + "/" + obj["model"]
        asset = assets[key]
        native = asset.get("native_default_geometry")
        if not native:
            return None, {}
        objects[role_key] = {k: asset[k] for k in ("category", "model", "asset_sha256")}
        objects[role_key]["default_scale"] = native["default_scale"]
        role_keys[(role_key.split(":", 1)[0], key)] = role_key
    return coupled_profile(kind, objects), role_keys
