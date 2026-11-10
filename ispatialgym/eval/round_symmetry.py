"""Yaw free in the success measure for rotationally symmetric objects (owner decision 2026-11-09 A4; i7-measure).

The success measure (``pose_reached``: the ``PoseReached`` state of every TRO pose target, :func:`ispatialgym.eval.tro.apply_targets_to_scope`)
compares a target's orientation with its goal orientation in full, modulo the target's recorded ``orientation_equivalences`` only. A round
object - under_large_03's trash can (``trash_can/jlawet``) - has no meaningful yaw, yet it was gated on it: a can pushed under the desk that
turned 0.22 rad on the way ended ``symbolic_only`` (i7-floorpush A). The demo QA's ``placed_orientation`` fact already treated such
rotations as free in two places (``oracle.executor.Executor.placed_orientation_report``): a cylinder of ``assets/icl/revolution_axes.json``
modulo rotation about its long axis, and any push-under target modulo the world vertical.

With ``ISPATIALGYM_SUCCESS_YAW_FREE_SYMMETRIC=1`` (default OFF; it flips only in the eval-measure commit, together with A2's
``ISPATIALGYM_BRIEFCASE_LYING``) one symmetry notion serves both:

* the table: :data:`ROUND_AXES_PATH` (``scripts/symmetry/revolution_axes.py --rule round``) - every model whose ICL ``symmetry`` column
  (``assets/icl/roles/candidates.csv``: the rotations under which geometry AND texture are identical, the 2026-09-15 review applied; the
  column the customs roles' ``orientation_equivalences`` already come from) holds a body-of-revolution tag (``xinf`` / ``yinf`` / ``zinf``):
  166 models - round cans, plates, bowls, glasses, lids, caps, plain cylinders. The geometry-only survey lists of the QA's cylinder table
  are not used: a wall clock's face or a printed marker turns visibly;
* the success measure: every pose target with an orientation metric whose bound object is such a model gets the continuous symmetry about
  that object-frame axis added to its ``orientation_equivalences`` (:func:`apply_yaw_free`) - the survey's own convention for a continuous
  axis, :func:`ispatialgym.sampling.customs.equivalence_group` (72 rotations at 5 deg: a pure turn about the axis leaves at most 2.5 deg =
  0.044 rad of error), composed with the target's recorded equivalences. For an object standing on its axis that is its yaw. The TRO file is
  not changed: the equivalences are added where the targets are configured, so demo QA (``oracle.run_isg`` -> ``tasks.ispatialgym_task``),
  replay, the sampler's goal checks (``sampling.tasks.base``) and policy eval (``tasks.vec_task``) all read the same measure;
* the QA: ``Executor.revolution_axis_of`` reads the same table (:func:`round_axis`) for a model its cylinder table lacks, so
  ``placed_orientation`` frees every rotation the measure frees (the 43 models in both tables have the same axis); it keeps freeing the
  printed cylinders' roll in the fingers, which the measure does not.

A target that already carries the rotation (a customs role tagged ``zinf``) is left as it is (``acted`` False). Kit-style counts per process:
:data:`COUNTS` (``targets`` / ``acted`` / ``already``), one INFO line per acted target.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th

log = logging.getLogger(__name__)

SUCCESS_YAW_FREE_ENV = "ISPATIALGYM_SUCCESS_YAW_FREE_SYMMETRIC"
#: default OFF (owner: ship with another eval-measure change, so scores change once - A2's lying briefcase; one commit flips both)
SUCCESS_YAW_FREE_DEFAULT = "0"
_ON = ("1", "true", "yes", "on")

_ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
ROUND_AXES_PATH = os.path.join(_ASSETS, "icl", "round_axes.json")
#: the enumeration step of a continuous axis (the survey's / customs' ``CONTINUOUS_STEP_DEG``)
STEP_DEG = 5.0

COUNTS: Dict[str, int] = {}


def yaw_free_on() -> bool:
    """:data:`SUCCESS_YAW_FREE_ENV` (unset / empty: :data:`SUCCESS_YAW_FREE_DEFAULT`) is on."""
    return (os.environ.get(SUCCESS_YAW_FREE_ENV) or SUCCESS_YAW_FREE_DEFAULT).strip().lower() in _ON


@functools.lru_cache(maxsize=None)
def round_axes(path: str = ROUND_AXES_PATH) -> Dict[str, Dict[str, Any]]:
    """The round-axis table ``{"<category>/<model>": {"axis": "x"|"y"|"z", ...}}`` (empty when the asset is missing or unreadable)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        log.warning("round axes table %s not readable (%r): no object gets a free rotation", path, e)
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, dict) and v.get("axis") in ("x", "y", "z")}


def round_axis(category: Optional[str], model: Optional[str], path: str = ROUND_AXES_PATH) -> Optional[Dict[str, Any]]:
    """The table entry of ``category/model`` (``axis`` = the object-frame axis it is a body of revolution about), None otherwise."""
    if not category or not model:
        return None
    return round_axes(path).get(f"{category}/{model}")


@functools.lru_cache(maxsize=None)
def axis_group(axis: str, step_deg: float = STEP_DEG) -> Tuple[Tuple[float, ...], ...]:
    """The rotations about the object-frame ``axis`` (xyzw), identity excluded: ``customs.equivalence_group([f"{axis}inf"])``."""
    from ispatialgym.sampling.customs import equivalence_group

    group = equivalence_group([f"{axis}inf"], step_deg=step_deg) or []
    return tuple(tuple(float(c) for c in q) for q in group[1:])


def _qmul(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float, float]:
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _key(q: Sequence[float]) -> Tuple[int, ...]:
    """Dedupe key of a rotation (q and -q alike), 1e-4 resolution."""
    v = [float(c) for c in q]
    n = sum(c * c for c in v) ** 0.5 or 1.0
    v = [c / n for c in v]
    for c in (v[3], v[2], v[1], v[0]):
        if abs(c) > 1e-6:
            if c < 0:
                v = [-c_ for c_ in v]
            break
    return tuple(int(round(c * 1e4)) for c in v)


def with_axis(equivalences: Sequence[Any], axis: str) -> Tuple[List[Tuple[float, ...]], int]:
    """``equivalences`` (object-frame xyzw, identity implied) composed with the continuous rotation about ``axis``: every ``e * g`` (e in
    {identity} + equivalences, g in {identity} + :func:`axis_group`) but the identity, deduplicated. Returns ``(the list, how many are new)``."""
    ident = (0.0, 0.0, 0.0, 1.0)
    base = [ident] + [tuple(float(c) for c in (e.tolist() if hasattr(e, "tolist") else e)) for e in (equivalences or [])]
    seen = {_key(ident)}
    old = {_key(e) for e in base[1:]}
    out: List[Tuple[float, ...]] = []
    for e in base:
        for g in (ident,) + axis_group(axis):
            r = _qmul(e, g)
            k = _key(r)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
    return out, sum(1 for r in out if _key(r) not in old)


def entity_model(entity: Any) -> Tuple[Optional[str], Optional[str]]:
    """``(category, model)`` of a bound scene object (its attributes, else parsed from its name ``<category>_<model>_<i>``)."""
    cat, model = getattr(entity, "category", None), getattr(entity, "model", None)
    name = str(getattr(entity, "name", "") or "")
    if (not cat or not model) and name:
        parts = name.rsplit("_", 2)
        if len(parts) == 3 and parts[2].isdigit():
            cat, model = cat or parts[0], model or parts[1]
    return (str(cat) if cat else None), (str(model) if model else None)


def apply_yaw_free(meta: Any, object_scope: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """With :func:`yaw_free_on`: every pose target of ``meta`` with an orientation metric, in the world frame, whose bound entity's model has
    a :func:`round_axis` gets that axis's continuous rotation in its ``orientation_equivalences`` (:func:`with_axis`, in place - the meta the
    executor's QA reads is the same object). Returns ``{inst: {"model", "axis", "acted", "n_equivalences"}}`` (empty when off); idempotent."""
    if not yaw_free_on():
        return {}
    report: Dict[str, Dict[str, Any]] = {}
    for inst, tgt in (getattr(meta, "pose_targets", None) or {}).items():
        if getattr(tgt, "orientation", None) is None or getattr(tgt, "orientation_metric", None) is None:
            continue
        if str(getattr(tgt, "frame", "world") or "world") != "world":
            continue
        cat, model = entity_model(object_scope.get(inst))
        entry = round_axis(cat, model)
        if entry is None:
            continue
        axis = str(entry["axis"])
        COUNTS["targets"] = COUNTS.get("targets", 0) + 1
        eqs, n_new = with_axis(list(getattr(tgt, "orientation_equivalences", None) or []), axis)
        # "acted": this switch added the rotation (sticky over repeated calls - the task configures its targets more than once)
        rec = {"model": f"{cat}/{model}", "axis": axis, "acted": n_new > 0 or bool(getattr(tgt, "_yaw_free_acted", False)), "n_equivalences": len(eqs)}
        if n_new > 0:
            tensors = [th.tensor(q, dtype=th.float32) for q in eqs]
            cur = getattr(tgt, "orientation_equivalences", None)
            if isinstance(cur, list):
                cur[:] = tensors  # in place: a shallow copy of the target (``clone_frames.offset_targets_meta``) shares the list
            else:
                tgt.orientation_equivalences = tensors
            try:
                tgt._yaw_free_acted = True
            except Exception:  # noqa: BLE001 - a slotted target keeps no flag
                pass
            COUNTS["acted"] = COUNTS.get("acted", 0) + 1
            log.info("SUCCESS_YAW_FREE_SYMMETRIC: %s (%s) - rotation about its %s axis free in the success measure (%d equivalences, %d new)",
                     inst, rec["model"], axis, len(eqs), n_new)
        else:
            COUNTS["already"] = COUNTS.get("already", 0) + 1
        report[inst] = rec
    return report


__all__ = ["COUNTS", "ROUND_AXES_PATH", "SUCCESS_YAW_FREE_ENV", "apply_yaw_free", "axis_group", "entity_model", "round_axes", "round_axis",
           "with_axis", "yaw_free_on"]
