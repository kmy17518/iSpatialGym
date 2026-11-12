"""Support rules of the alignment tasks: rectangular tops (axial) and opaque walls (world vertical).

Owner 2026-10-13 ~18:40 (eval85 ``COMPLAINTS.md``, global rules for new instances): "Alignment axial: the support is rectangular; round
tables (e.g. the round coffee tables) are not supports." and "Alignment world vertical: transparent walls are not supports."
(alignment-axial#04 had been sampled on the round ``coffee_table_mtchfd_0``, alignment-world_vertical#02 on the glass partition
``walls_dmcumz_0``.)

**Rectangular top** (:data:`RECTANGULAR_TOP`, alignment-axial). The convex outline of the support's top covers at least
:data:`MIN_TOP_RECTANGULARITY` of its minimum-area bounding rectangle (:func:`rectangularity`). The outline is the convex hull of the
support's B4 top raster (``scene_inventory/<scene>/supports/<name>.json`` ``top.hull_xy``: the 2 cm cells whose downward ray first hits
the support at its top height, the surface the sampler places on). A disc or an ellipse covers pi/4 = 0.785, a regular octagon 0.83,
a stadium (oval) of aspect a covers 1 - (1 - pi/4) / a (0.89 at 2 : 1), an L-shaped corner desk 0.82-0.84 (the cut corner of its
outline); a rectangle covers 1, one with corners rounded to radius r 1 - (4 - pi) r^2 / (w h) (0.94 at r = 0.22 m on 1.2 x 0.6 m).
On the seven scenes the rectangular axial candidates measure >= 0.966, the round / oval / L-shaped / curved ones <= 0.921.

**Opaque wall** (:data:`OPAQUE_WALL`, alignment-world_vertical). The wall's BEHAVIOR-1K asset tags no link ``glass``
(``objects/<category>/<model>/misc/metadata.json`` ``link_tags``; OmniGibson binds the visual meshes of such a link to its glass
material). A wall with a glass link is excluded as a whole, its opaque frame included.

A support the rule of its subtask cannot evaluate (no B4 top record, no asset metadata) is excluded as well: the rules admit only
the supports they have shown to be rectangular / opaque. Other subtasks have no support rule. Where the rules apply: spec §3.2
"Alignment support rules".
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym import ASSETS_ROOT, REPO_ROOT

RECTANGULAR_TOP = "rectangular_top"
OPAQUE_WALL = "opaque_wall"
SUBTASK_RULES = {"alignment-axial": RECTANGULAR_TOP, "alignment-world_vertical": OPAQUE_WALL}
MIN_TOP_RECTANGULARITY = 0.94
GLASS_TAG = "glass"
#: the ``ground_isg`` skip reason of a pick row whose support a rule excludes (``support_candidates`` status ``not_grounded: ...``)
SKIP_REASON = "excluded_support"
DEFAULT_INVENTORY = os.path.join(ASSETS_ROOT, "icl", "require", "lib", "scene_inventory")
INVENTORY_MANIFEST_ENV = "ISPATIALGYM_SUPPORT_INVENTORY_MANIFEST"
INVENTORY_MANIFEST_SHA_ENV = "ISPATIALGYM_SUPPORT_INVENTORY_SHA256"


class SupportRuleError(ValueError):
    """A sampler was handed a support its subtask's rule excludes."""


def rule_of(subtask: Optional[str]) -> Optional[str]:
    """The support rule of ``subtask`` (:data:`SUBTASK_RULES`), None when it has none."""
    return SUBTASK_RULES.get(subtask or "")


def data_root() -> str:
    """The BEHAVIOR-1K datasets root: ``$OMNIGIBSON_DATA_PATH``, else the repo's ``Behavior-1K/datasets``."""
    return os.environ.get("OMNIGIBSON_DATA_PATH") or os.path.join(REPO_ROOT, "Behavior-1K", "datasets")


# ----------------------------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------------------------


def convex_hull(points) -> np.ndarray:
    """Monotone-chain convex hull of (N, 2) points, counter-clockwise, without collinear points."""
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    upper: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(poly) -> float:
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def min_area_rectangle(hull) -> Tuple[float, float, float]:
    """``(area, side_a, side_b)`` of the minimum-area rectangle enclosing a convex polygon (one side lies on a polygon edge)."""
    p = np.asarray(hull, dtype=float)
    best = (float("inf"), 0.0, 0.0)
    for i in range(len(p)):
        e = p[(i + 1) % len(p)] - p[i]
        n = float(np.hypot(e[0], e[1]))
        if n < 1e-12:
            continue
        u = e / n
        v = np.array([-u[1], u[0]])
        a, b = p @ u, p @ v
        sa, sb = float(a.max() - a.min()), float(b.max() - b.min())
        if sa * sb < best[0]:
            best = (sa * sb, sa, sb)
    return best


def rectangularity(points) -> float:
    """Area of the convex hull of ``points`` over the area of its minimum-area bounding rectangle (1 = a rectangle); 0 for fewer than
    three distinct points or a degenerate hull."""
    hull = convex_hull(points)
    if len(hull) < 3:
        return 0.0
    rect = min_area_rectangle(hull)[0]
    return polygon_area(hull) / rect if 0.0 < rect < float("inf") else 0.0


# ----------------------------------------------------------------------------------------------------------------
# data: B4 top records, BEHAVIOR-1K asset metadata
# ----------------------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _top_record(inventory: str, scene: str, name: str) -> Optional[dict]:
    path = os.path.join(inventory, scene, "supports", f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("top") or None


def _bound_private_top_record(scene: str, name: str) -> Optional[dict]:
    """Use an exact measured B4 input only in the private scene that produced it.

    The manifest locates evidence; it does not replace the rectangularity rule.
    Original inventory files and every unlisted scene/support remain unchanged.
    """
    path = os.environ.get(INVENTORY_MANIFEST_ENV)
    expected = os.environ.get(INVENTORY_MANIFEST_SHA_ENV)
    if not path and not expected:
        return None
    if not path or not expected:
        raise SupportRuleError("private B4 inventory requires its manifest and digest")

    def bound_json(reference):
        source = Path(reference["path"])
        if not source.is_absolute():
            raise SupportRuleError("private B4 evidence path must be absolute")
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
            raise SupportRuleError("private B4 evidence digest differs: " + str(source))
        return json.loads(payload)

    try:
        manifest = bound_json({"path": path, "sha256": expected})
        if manifest["schema"] != "isg-private-support-inventory-v1":
            raise SupportRuleError("unsupported private B4 inventory schema")
        entries = [row for row in manifest["entries"] if (row["scene"], row["name"]) == (scene, name)]
        if not entries:
            return None
        if len(entries) != 1:
            raise SupportRuleError("ambiguous private B4 inventory entry")
        entry = entries[0]
        scene_dir = os.environ.get("ISPATIALGYM_STABLE_SCENE_DIR")
        if not scene_dir or (Path(scene_dir) / (scene + "_stable.json")).resolve() != Path(entry["scene_file"]["path"]).resolve():
            raise SupportRuleError("private B4 inventory is bound to a different stable scene")
        bound_json(entry["scene_file"])
        record = bound_json(entry["inventory_record"])
        measured = bound_json(entry["measurement"])
        if any(record[key] != entry[key] for key in ("scene", "name", "category", "model", "scale")):
            raise SupportRuleError("private B4 support identity differs")
        if (measured["native_inventory_record"] != entry["inventory_record"] or
                measured["source_scene"] != entry["scene_file"] or
                measured["object_name"] != name or measured["actual_scale"] != entry["scale"] or
                measured["category"] != entry["category"] or measured["model"] != entry["model"]):
            raise SupportRuleError("private B4 measurement identity differs")
        top = record["top"]
        hull = np.asarray(top["hull_xy"], dtype=float)
        if top["resolution"] != .02 or hull.ndim != 2 or hull.shape[1] != 2 or len(hull) < 3 or not np.isfinite(hull).all():
            raise SupportRuleError("private B4 measurement has no finite native top outline")
        return top
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        raise SupportRuleError("invalid private B4 inventory evidence: " + str(exc)) from exc


def top_rectangularity(scene: str, name: str, inventory: Optional[str] = None) -> Optional[float]:
    """:func:`rectangularity` of the B4 top outline of support ``name`` in ``scene``; None without a B4 record with a top outline."""
    top = _bound_private_top_record(scene, name) if inventory is None else None
    if top is None:
        top = _top_record(inventory or DEFAULT_INVENTORY, scene, name)
    hull = (top or {}).get("hull_xy") or []
    return rectangularity(hull) if len(hull) >= 3 else None


def split_object_name(name: str) -> Optional[Tuple[str, str]]:
    """``(category, model)`` of a BEHAVIOR scene object name ``<category>_<model>_<index>`` (``walls_dmcumz_0`` -> walls, dmcumz)."""
    parts = (name or "").rsplit("_", 2)
    if len(parts) != 3 or not all(parts) or not parts[2].isdigit():
        return None
    return parts[0], parts[1]


@functools.lru_cache(maxsize=None)
def _link_tags(root: str, category: str, model: str) -> Optional[Dict[str, List[str]]]:
    path = os.path.join(root, "behavior-1k-assets", "objects", category, model, "misc", "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return dict(json.load(f).get("link_tags") or {})


def glass_links(name: str, root: Optional[str] = None) -> Optional[List[str]]:
    """The links of scene object ``name``'s BEHAVIOR-1K asset tagged ``glass`` ([] = none); None when its metadata is not found."""
    cm = split_object_name(name)
    tags = _link_tags(root or data_root(), *cm) if cm else None
    if tags is None:
        return None
    return sorted(link for link, t in tags.items() if GLASS_TAG in (t or []))


# ----------------------------------------------------------------------------------------------------------------
# the rules
# ----------------------------------------------------------------------------------------------------------------


def exclusion_reason(subtask: Optional[str], scene: Optional[str], name: Optional[str], inventory: Optional[str] = None,
                     root: Optional[str] = None) -> Optional[str]:
    """Why the rule of ``subtask`` excludes support ``name`` of ``scene`` (``<code>: <detail>``, code ``not_rectangular`` /
    ``transparent_wall`` / ``unchecked``); None when the support is admitted or the subtask has no rule. The floor (``"floor"``, a
    floor-bound support) is never excluded."""
    rule = rule_of(subtask)
    if rule is None or not name or name == "floor":
        return None
    if rule == RECTANGULAR_TOP:
        r = top_rectangularity(scene or "", name, inventory)
        if r is None:
            return f"unchecked: no B4 top outline of {name} in {scene} (the rectangular-top rule needs it)"
        if r < MIN_TOP_RECTANGULARITY:
            return (f"not_rectangular: its top covers {r:.3f} of its minimum bounding rectangle (< {MIN_TOP_RECTANGULARITY}: a round, oval, "
                    f"L-shaped or curved top)")
        return None
    if rule == OPAQUE_WALL:
        links = glass_links(name, root)
        if links is None:
            return f"unchecked: no BEHAVIOR-1K asset metadata for {name} (the transparent-wall rule needs its link tags)"
        if links:
            return f"transparent_wall: its asset marks link(s) {', '.join(links)} as glass"
        return None
    raise ValueError(f"unknown support rule {rule!r} of {subtask}")


def split_supports(subtask: Optional[str], scene: Optional[str], names: Sequence[str], **kw) -> Tuple[List[str], Dict[str, str]]:
    """``(admitted names in order, {excluded name: reason})`` of the support names of one scene block."""
    keep: List[str] = []
    dropped: Dict[str, str] = {}
    for n in names:
        why = exclusion_reason(subtask, scene, n, **kw)
        if why:
            dropped[n] = why
        else:
            keep.append(n)
    return keep, dropped


def check_support(subtask: Optional[str], scene: Optional[str], name: Optional[str], **kw) -> None:
    """:class:`SupportRuleError` when the rule of ``subtask`` excludes support ``name`` of ``scene``."""
    why = exclusion_reason(subtask, scene, name, **kw)
    if why:
        raise SupportRuleError(f"{subtask}: support {name} of {scene} is excluded by the {rule_of(subtask)} rule - {why}")
