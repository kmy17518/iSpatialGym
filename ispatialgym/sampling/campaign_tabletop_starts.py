"""Campaign-only proposal ordering near a physically proven goal viewpoint.

These helpers choose which native candidates to try. The existing floor,
collision, frame, map, IK and trajectory checks still decide feasibility.
"""
import math
from ispatialgym.sampling.campaign_held_goals import _require


def source_stance_hint_enabled(relation, scale, class_id):
    """Scope source candidate ranking to the measured pair recoveries."""
    return scale == 'small' and (relation == 'inside' or
        (relation == 'nextto' and class_id == 'inter_object_relation_satisfaction-nextto_02') or
        (relation == 'touching' and class_id == 'inter_object_relation_satisfaction-touching_01'))


def anchor_base(raw):
    value = raw.get('anchor_base_pose2d')
    if value is None:
        return None
    _require(isinstance(value, (list, tuple)) and len(value) == 3
             and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value),
             'invalid campaign anchor base')
    return tuple(float(v) for v in value)


def anchor_manipulation(raw):
    """Read a proposal hint bound to the successful source's audited summary."""
    value = raw.get('anchor_manipulation_stance')
    if value is None:
        return None
    _require(isinstance(value, dict), 'invalid anchor manipulation stance')
    sha = raw.get('anchor_sources', {}).get('summary', {}).get('sha256')
    _require(isinstance(sha, str) and len(sha) == 64
             and all(c in '0123456789abcdef' for c in sha)
             and value.get('source_summary_sha256') == sha,
             'anchor manipulation stance lacks matching summary provenance')
    for key, size in [('base_pose2d', 3), ('focus', 3)]:
        vector = value.get(key)
        _require(isinstance(vector, (tuple, list)) and len(vector) == size
                 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vector),
                 'invalid anchor manipulation '+key)
    _require(all(isinstance(value.get(k), (int, float)) and not isinstance(value[k], bool)
                 and math.isfinite(value[k]) for k in ('translate', 'tilt')),
             'invalid anchor manipulation torso')
    return {k:value[k] for k in ('base_pose2d','focus','translate','tilt')}


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


def prioritize_written_stances(order, candidates, written, *, explore=4, exploit=2):
    """Keep exploration and try checked candidates near a fully written member.

    This never adds a stance or removes one from the native checked pool.
    Callers key ``written`` by the immutable goal-plan hash.
    """
    if not written:
        return order
    order = [int(i) for i in order]
    def distance(index):
        c = candidates[index]
        def one(prior):
            base = prior['base_pose2d']
            angle = abs(math.atan2(math.sin(float(c.base_pose2d[2])-base[2]),
                                   math.cos(float(c.base_pose2d[2])-base[2])))
            score = math.dist([float(v) for v in c.base_pose2d[:2]], base[:2]) + .3*angle
            score += abs(float(c.translate)-prior['translate']) + abs(float(c.tilt)-prior['tilt'])
            if c.focus is not None and prior.get('focus') is not None:
                score += math.dist([float(v) for v in c.focus[:2]], prior['focus'][:2])
            return score
        return min(one(prior) for prior in written)
    prefix = order[:explore]
    promoted = sorted((i for i in order if i not in prefix), key=distance)[:exploit]
    return prefix + promoted + [i for i in order if i not in prefix and i not in promoted]
