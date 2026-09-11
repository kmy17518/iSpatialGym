"""BEHAVIOR knowledge-base lookups used by the sampler (no simulator needed).

* category -> BDDL synset (``bddl.object_taxonomy``),
* model readiness (no unprocessed complaint in ``generated_data/complaints.json``, not in the deletion queue),
* canonical bounding boxes and meta-link / attachment data (``generated_data/object_inventory.json``).
"""

from __future__ import annotations

import csv
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional

import bddl

GENERATED_DATA = os.path.join(os.path.dirname(bddl.__file__), "generated_data")

#: never counted as "objects" in visibility criteria (building shell and fixtures that belong to it)
STRUCTURE_CATEGORIES = {"walls", "floors", "ceilings", "roof", "baseboard", "downlight", "room_light", "electric_switch"}


@lru_cache(maxsize=1)
def taxonomy():
    from bddl.object_taxonomy import ObjectTaxonomy

    return ObjectTaxonomy()


def synset_of_category(category: str) -> str:
    syn = taxonomy().get_synset_from_category(category)
    if syn is None:
        raise KeyError(f"category {category!r} has no BDDL synset")
    return syn


def abilities_of_category(category: str) -> Dict[str, dict]:
    """Default object abilities (what ``DatasetObject`` uses to pick its object states)."""
    try:
        return dict(taxonomy().get_abilities(synset_of_category(category)) or {})
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _inventory() -> dict:
    with open(os.path.join(GENERATED_DATA, "object_inventory.json")) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _unprocessed_complaints() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with open(os.path.join(GENERATED_DATA, "complaints.json")) as f:
        for c in json.load(f):
            if str(c.get("processed")).lower() != "true":
                out.setdefault(c["object"].split("-")[-1], []).append(c["type"])
    return out


@lru_cache(maxsize=1)
def _deleted_models() -> set:
    path = os.path.join(GENERATED_DATA, "deletion_queue.csv")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {r["Object"].strip().split("-")[-1] for r in csv.DictReader(f) if r.get("Object")}


@lru_cache(maxsize=1)
def _model_to_category() -> Dict[str, str]:
    out = {}
    for name in _inventory()["providers"]:
        cat, mid = name.rsplit("-", 1)
        out[mid] = cat
    return out


def is_ready(model: str) -> bool:
    """A model is *ready* when it has no unprocessed knowledge-base complaint and is not queued for deletion."""
    return model not in _unprocessed_complaints() and model not in _deleted_models()


def complaints(model: str) -> List[str]:
    return sorted(set(_unprocessed_complaints().get(model, [])))


def canonical_bbox(model: str) -> Optional[List[float]]:
    inv = _inventory()
    bb = inv["bounding_box_sizes"].get(model)
    if bb is None:
        cat = _model_to_category().get(model)
        if cat is not None:
            bb = inv["bounding_box_sizes"].get(f"{cat}-{model}")
    return list(bb) if bb is not None else None


def meta_links(category: str, model: str) -> List[str]:
    return list(_inventory()["meta_links"].get(f"{category}-{model}", []))


def attachment_pairs(category: str, model: str) -> Dict[str, List[str]]:
    return dict(_inventory()["attachment_pairs"].get(f"{category}-{model}", {}))


def models_of_category(category: str, ready_only: bool = True) -> List[str]:
    out = [m for m, c in _model_to_category().items() if c == category]
    if ready_only:
        out = [m for m in out if is_ready(m)]
    return sorted(out)
