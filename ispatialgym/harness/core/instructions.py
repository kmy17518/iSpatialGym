"""Task statements from the instance's alias: one template per iSpatialGym family, always paired with the goal image.

``level="family"`` names no objects; ``level="alias"`` names the object categories the alias carries (as a language
instruction would). Relation directions follow the BDDL goals: ``under_small_scale-<t>-<p>`` is ``(under p t)`` (the
partner ends up under the target), ``under_large_scale-<t>-<p>`` is ``(under t p)``; ontop / inside / nextto /
touching read ``(rel t p)``."""

from __future__ import annotations

import csv
import importlib.util
import os
import re
from typing import Dict, FrozenSet, List, Optional

_ID_TAIL = re.compile(r"_[a-z]{6}$")  # object model ids: bar_soap_utgixp, drop_in_sink_lkklqs_0
_SCALE_TAIL = re.compile(r"_s\d{3}$")  # scaled copies: box_of_cereal_s080
_VARIANT_TAILS = ("_iorsl", "_pitched", "_place")
_CATEGORIES: Optional[FrozenSet[str]] = None


def _categories() -> FrozenSet[str]:
    """BEHAVIOR object categories (bddl's category_mapping.csv), so a model id is told apart from a six-letter word."""
    global _CATEGORIES
    if _CATEGORIES is None:
        names = set()
        spec = importlib.util.find_spec("bddl")
        for root in (spec.submodule_search_locations or []) if spec else []:
            path = os.path.join(root, "generated_data", "category_mapping.csv")
            if os.path.isfile(path):
                with open(path, newline="") as f:
                    names = {row["category"] for row in csv.DictReader(f) if row.get("category")}
                break
        _CATEGORIES = frozenset(names)
    return _CATEGORIES


def clean(token: str) -> str:
    t = token
    for tail in _VARIANT_TAILS:
        if t.endswith(tail):
            t = t[: -len(tail)]
    t = _SCALE_TAIL.sub("", t)
    t = re.sub(r"_\d+$", "", t)
    cats, m = _categories(), _ID_TAIL.search(t)
    if m and t not in cats and (not cats or t[: m.start()] in cats):
        t = t[: m.start()]
    if t.endswith("_board"):
        t = t[: -len("_board")] + " (with a board)"
    return t.replace("_", " ").strip()


def _distractors(token: str) -> List[str]:
    return [clean(p) for p in re.split(r"_\d+_?", token) if p]


def parse_alias(alias: str) -> Dict[str, object]:
    parts = alias.split("-")
    fam = parts[0]
    out: Dict[str, object] = {"family": fam, "alias": alias}
    if fam == "camera_relocalization":
        out["variant"] = parts[1] if len(parts) > 1 else "standard"
        if out["variant"] == "obstructed" and len(parts) > 2:
            out["obstacle"] = clean(parts[2])
    elif fam in ("mental_rotation", "object_scaling"):
        out["target"] = clean(parts[1])
    elif fam == "alignment":
        out["variant"], out["target"], out["support"] = parts[1], clean(parts[2]), clean(parts[3])
    elif fam == "configuration_matching":
        kind = parts[1]
        if kind.startswith("articulation_"):
            out["variant"] = "open" if "_open_" in kind else "close"
            out["scale"] = "large" if kind.endswith("large_scale") else "small"
            out["target"] = clean(parts[2])
            if len(parts) > 3:
                out["support"] = clean(parts[3])
        elif kind == "attachment":
            out["variant"], out["target"], out["partner"], out["support"] = "attachment", clean(parts[2]), clean(parts[3]), clean(parts[4])
        elif kind == "block_design":
            out["variant"], out["support"] = "block_design", clean(parts[-1])
    elif fam == "disembedding":
        out["target"], out["distractors"], out["support"] = clean(parts[1]), _distractors(parts[2]), clean(parts[3])
    elif fam == "inter_object_relation_satisfaction":
        rel = parts[1]
        out["relation"] = rel.split("_")[0]
        out["scale"] = "large" if rel.endswith("large_scale") else "small"
        out["target"], out["partner"] = clean(parts[2]), clean(parts[3])
        if len(parts) > 4:
            out["support"] = clean(parts[4])
    return out


def instruction_for(alias: Optional[str], level: str = "alias") -> str:
    """The task statement; the goal image always comes with it."""
    if not alias:
        return "Make the scene match the goal image."
    a = parse_alias(alias)
    named = level == "alias"

    def n(key: str, generic: str) -> str:
        v = a.get(key)
        return f"the {v}" if (named and v) else generic

    fam = a["family"]
    if fam == "camera_relocalization":
        s = "Drive the robot so that its head camera sees the view shown in the goal image: the same place and the same viewing direction."
        if a.get("variant") == "obstructed":
            s += f" {('A ' + str(a['obstacle'])) if (named and a.get('obstacle')) else 'An obstacle'} may be in the way."
        return s
    if fam == "mental_rotation":
        return (f"You are holding {n('target', 'an object')} in one gripper. Rotate it in place so that its orientation matches the goal image, "
                "and keep holding it.")
    if fam == "object_scaling":
        return (f"You are holding {n('target', 'an object')} in one gripper. Move it toward or away from the head camera so that it appears "
                "the same size as in the goal image, and keep holding it.")
    if fam == "alignment":
        if a.get("variant") == "world_vertical":
            return (f"{n('target', 'An object').capitalize()} is mounted on a wall and can only rotate in the plane of the wall. "
                    "Rotate it until its orientation matches the goal image.")
        return (f"{n('target', 'An object').capitalize()} lies on {n('support', 'a surface')} and can only rotate about a vertical axis "
                "through a fixed point. Rotate it until its orientation matches the goal image.")
    if fam == "configuration_matching":
        v = a.get("variant")
        if v in ("open", "close"):
            verb = "Open" if v == "open" else "Close"
            return f"{verb} {n('target', 'the articulated object')} until it matches the goal image."
        if v == "attachment":
            return f"Attach {n('target', 'the part')} to {n('partner', 'its base object')} as shown in the goal image."
        if v == "block_design":
            return f"Arrange the blocks on {n('support', 'the table')} to match the layout shown in the goal image."
    if fam == "disembedding":
        among = (" (it sits among " + ", ".join(a["distractors"]) + ")") if (named and a.get("distractors")) else ""  # type: ignore[arg-type]
        return (f"Take {n('target', 'the target object')} out of the tray it is in{among} and put it into the other tray, as shown in the "
                "goal image. Only that object may end up in the other tray.")
    if fam == "inter_object_relation_satisfaction":
        t, p, rel = n("target", "the object"), n("partner", "the other object"), a.get("relation")
        if rel == "under":
            if a.get("scale") == "small":
                return f"Move {t} so that {p} is under it, as shown in the goal image."
            return f"Move {t} under {p}, as shown in the goal image."
        phrase = {"ontop": "on top of", "inside": "inside", "nextto": "next to", "touching": "so that it touches"}.get(str(rel), str(rel))
        return f"Move {t} {phrase} {p}, as shown in the goal image."
    return "Make the scene match the goal image."
