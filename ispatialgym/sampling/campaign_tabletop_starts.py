"""Campaign-only proposal ordering near a physically proven goal viewpoint.

These helpers choose which native candidates to try. The existing floor,
collision, frame, map, IK and trajectory checks still decide feasibility.
"""
import math
from ispatialgym.sampling.campaign_held_goals import _require


def anchor_base(raw):
    value = raw.get('anchor_base_pose2d')
    if value is None:
        return None
    _require(isinstance(value, (list, tuple)) and len(value) == 3
             and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value),
             'invalid campaign anchor base')
    return tuple(float(v) for v in value)


def edge_lateral(edge, anchor):
    return sum((anchor[i]-float(edge.mid[i]))*float(edge.tangent[i]) for i in range(2))


def ordered_edges(rows, anchor):
    if anchor is None:
        return rows
    def score(row):
        edge, stance = row
        lateral = min(float(edge.length)/2, max(-float(edge.length)/2, edge_lateral(edge, anchor)))
        xy = edge.base_xy(stance.stand_off, lateral)
        yaw = float(stance.base_pose2d[2])
        angle = abs(math.atan2(math.sin(yaw-anchor[2]), math.cos(yaw-anchor[2])))
        return math.dist([float(xy[0]),float(xy[1])],anchor[:2]) + .3*angle
    return sorted(rows, key=score)


def near_lateral(rng, lo, hi, preferred, radius=.08):
    _require(all(math.isfinite(v) for v in (lo, hi, preferred, radius)) and lo <= hi and radius > 0,
             'invalid campaign lateral proposal bounds')
    centre = min(hi, max(lo, preferred))
    return float(rng.uniform(max(lo, centre-radius), min(hi, centre+radius)))
