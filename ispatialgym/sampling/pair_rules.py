"""Target / partner pair rules of the IORS small-scale tasks, decided at grounding time (owner 2026-10-11); pure, no simulator.

**Fit scale** (ontop / under / inside small scale). A target bigger than its partner is shrunk by a uniform spawn scale before
sampling (``ground_isg`` writes it into ``roles.target.scale``), from the models' bounding boxes:

* extents at rest (:func:`rest_extents`): the native bbox (m, canonical frame) times the model's base spawn scale, rotated by the
  rest orientation of its grasp key (the sampler places the target at that rest, its footprint is what lands on the partner);
* ontop / under (:func:`fit_on_partner`): the target is set down centred on the partner (under: the partner ends under the
  target, so the geometry is the same and height does not enter). *Bigger* = the pair fails every tier of
  ``generation.fit_tiers`` of the sampler's §6.11 pair check (``iors_small.pair_admissibility``: ``any_yaw`` diagonal
  ``+ 2m <= p_short``, ``aligned`` ``t_short + 2m <= p_short`` and ``t_long + 2m <= p_long``, ``centred`` ``t_short + 2m <=
  p_short`` and ``t_long <= p_long``; ``m = partner_footprint_margin_m``), i.e. exactly the pairs the sampler would reject
  ``partner_too_small``. The scale makes the target fit the ``aligned`` tier, a margin ``m`` on every side:
  ``s = min((p_short - 2m) / t_short, (p_long - 2m) / t_long)``;
* inside (:func:`fit_inside`): *does not fit* = ``t_short + om > p_short`` or ``t_long + om > p_long`` (the sampler's opening
  check: the partner's outer footprint less ``opening_margin_m``), or the target cannot sink deep enough for the drop test's
  protrusion rule (``(1 - f) * t_height > p_height``, ``f = max_protrusion_fraction``; the partner height bounds its depth).
  ``s = min((p_short - om) / t_short, (p_long - om) / t_long, p_height / ((1 - f) * t_height))``;
* :func:`floor_scale` rounds the scale down to 0.01 (the fit still holds); below ``target_fit_scale_min`` (default
  :data:`TARGET_FIT_SCALE_MIN`) a shrink is not a fix (a lying sunflower in a tube vase needs x0.41) and the model is withheld.

**Handedness** (nextto, owner 2026-10-11): a mirror pair (left / right shoe) ends with the right one on the robot's right of the
left one. :func:`handed_target_side` gives the side of the partner the target must end on; :func:`partner_side_for` the side
the partner is placed on at the start (``tabletop.place_partner``: +1 = the robot's left of the target).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

#: Smallest uniform shrink applied (owner 2026-10-11 rule; ``generation.target_fit_scale_min`` overrides): a target that needs
#: less than this to fit is withheld instead of shrunk.
TARGET_FIT_SCALE_MIN = 0.5
#: Fit tiers of the sampler's §6.11 ontop / under pair check, in the order it tries them.
FIT_TIERS = ("any_yaw", "aligned", "centred")
HANDS = ("left", "right")


@dataclass
class Fit:
    """Outcome of a fit check: ``bigger`` (the pair fails the check), ``scale`` (the largest uniform factor of the target that
    passes, unrounded; 1.0 when it fits as is), ``tier`` (ontop / under: the sampler tier that passes as is), ``limit`` (what
    bounds the scale)."""

    bigger: bool
    scale: float = 1.0
    tier: Optional[str] = None
    limit: Optional[str] = None
    facts: Dict[str, Any] = field(default_factory=dict)


def _quat_wxyz_matrix(q: Sequence[float]):
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def rest_extents(bbox: Sequence[float], scale: Optional[Sequence[float]] = None, rest_quat_wxyz: Optional[Sequence[float]] = None) -> list:
    """World-axis extents ``[x, y, z]`` (m) of the box ``bbox`` (native extents, canonical frame) spawned at ``scale`` (None = unit)
    and turned by ``rest_quat_wxyz`` (None = canonical): ``e_i = sum_j |R_ij| * bbox_j * scale_j``."""
    s = [1.0, 1.0, 1.0] if scale is None else [float(v) for v in scale]
    h = [float(b) * sv for b, sv in zip(bbox, s)]
    if rest_quat_wxyz is None:
        return h
    R = _quat_wxyz_matrix(rest_quat_wxyz)
    return [sum(abs(R[i][j]) * h[j] for j in range(3)) for i in range(3)]


def _sorted_xy(ext: Sequence[float]):
    a, b = sorted((float(ext[0]), float(ext[1])))
    return a, b


def on_partner_tier(target_ext: Sequence[float], partner_ext: Sequence[float], margin: float, tiers: Sequence[str] = FIT_TIERS) -> Optional[str]:
    """The first tier of the sampler's §6.11 ontop / under pair check that admits the footprints (``iors_small.pair_admissibility``),
    else None (the sampler rejects ``partner_too_small``)."""
    ts, tl = _sorted_xy(target_ext)
    ps, pl = _sorted_xy(partner_ext)
    m = float(margin)
    tiers = list(tiers)
    if "any_yaw" in tiers and math.hypot(ts, tl) + 2 * m <= ps:
        return "any_yaw"
    if "aligned" in tiers and ts + 2 * m <= ps and tl + 2 * m <= pl:
        return "aligned"
    if "centred" in tiers and ts + 2 * m <= ps and tl <= pl:
        return "centred"
    return None


def fit_on_partner(target_ext: Sequence[float], partner_ext: Sequence[float], margin: float = 0.01, tiers: Sequence[str] = FIT_TIERS) -> Fit:
    """Ontop / under small scale: whether the target is bigger than the partner's top and the uniform factor that sets it down
    with ``margin`` on every side (the ``aligned`` tier)."""
    tier = on_partner_tier(target_ext, partner_ext, margin, tiers)
    ts, tl = _sorted_xy(target_ext)
    ps, pl = _sorted_xy(partner_ext)
    facts = {"target_footprint_m": [round(ts, 4), round(tl, 4)], "partner_footprint_m": [round(ps, 4), round(pl, 4)], "margin_m": float(margin)}
    if tier is not None:
        return Fit(False, 1.0, tier, None, facts)
    m = float(margin)
    by_short = (ps - 2 * m) / ts if ts > 0 else math.inf
    by_long = (pl - 2 * m) / tl if tl > 0 else math.inf
    s = max(0.0, min(by_short, by_long))
    return Fit(True, s, None, "partner_short_side" if by_short <= by_long else "partner_long_side", facts)


def fit_inside(target_ext: Sequence[float], partner_ext: Sequence[float], opening_margin: float = 0.01, max_protrusion_fraction: float = 0.5) -> Fit:
    """Inside small scale: whether the target passes the partner's opening (outer footprint less ``opening_margin``, the sampler's
    pair and release checks) and can sink deep enough for the protrusion rule, and the uniform factor that makes both hold."""
    ts, tl = _sorted_xy(target_ext)
    ps, pl = _sorted_xy(partner_ext)
    th, ph = float(target_ext[2]), float(partner_ext[2])
    om, f = float(opening_margin), float(max_protrusion_fraction)
    facts = {"target_footprint_m": [round(ts, 4), round(tl, 4)], "opening_m": [round(ps - om, 4), round(pl - om, 4)], "target_height_m": round(th, 4),
             "partner_height_m": round(ph, 4), "opening_margin_m": om, "max_protrusion_fraction": f}
    sink = (1.0 - f) * th
    if ts + om <= ps and tl + om <= pl and sink <= ph:
        return Fit(False, 1.0, None, None, facts)
    limits = {"opening_short_side": (ps - om) / ts if ts > 0 else math.inf, "opening_long_side": (pl - om) / tl if tl > 0 else math.inf,
              "protrusion": ph / sink if sink > 0 else math.inf}
    limit = min(limits, key=limits.get)
    return Fit(True, max(0.0, limits[limit]), None, limit, facts)


def floor_scale(scale: float, step: float = 0.01) -> float:
    """``scale`` rounded down to ``step`` (0.7046 -> 0.70): the fit still holds at the rounded scale."""
    return round(math.floor(float(scale) / step + 1e-6) * step, 4)


def uniform_factor(scale: Optional[Sequence[float]], base: Optional[Sequence[float]], tol: float = 1e-3) -> Optional[float]:
    """The factor ``f`` with ``scale = f * base`` on every axis (None = unit), else None (not a uniform multiple)."""
    s = [1.0, 1.0, 1.0] if scale is None else [float(v) for v in scale]
    b = [1.0, 1.0, 1.0] if base is None else [float(v) for v in base]
    ratios = [sv / bv for sv, bv in zip(s, b) if bv]
    if len(ratios) != 3 or max(ratios) - min(ratios) > tol * max(1.0, max(ratios)):
        return None
    return sum(ratios) / 3.0


def handed_target_side(target_hand: Optional[str], partner_hand: Optional[str]) -> Optional[str]:
    """The side of the partner (robot frame, ``"left"`` / ``"right"``) the target must end on: its own hand (the right shoe right
    of the left shoe). None when either hand is unknown; ValueError for two objects of the same hand."""
    if target_hand not in HANDS or partner_hand not in HANDS:
        return None
    if target_hand == partner_hand:
        raise ValueError(f"target and partner are both {target_hand}")
    return target_hand


def partner_side_for(target_side: str) -> float:
    """``place_partner``'s side of the partner (+1 = the robot's left of the target, -1 = its right) that leaves the target on
    ``target_side`` of the partner."""
    return 1.0 if target_side == "right" else -1.0
