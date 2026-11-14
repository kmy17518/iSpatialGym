"""Physical goal and saved-start comparisons, independent of oracle execution.

IDs, seeds, timestamps, and starting configurations never distinguish goals.
Start differences use saved object, grasp, and robot states, not proposal labels.
"""
import hashlib
import json
import math


def canonical(value):
    if isinstance(value, dict):
        return {k: quaternion(v) if k in ('orientation', 'ori') else canonical(v)
                for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and all(isinstance(v, (tuple, list)) for v in value) and len(value[0]) == 3 and len(value[1]) == 4:
            return [canonical(value[0]), quaternion(value[1])]
        return [canonical(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('nonfinite physical state')
        return round(value, 6) or 0.0
    return value


def quaternion(q):
    if len(q) != 4 or not all(math.isfinite(x) for x in q):
        raise ValueError('invalid quaternion')
    norm = math.sqrt(sum(x*x for x in q))
    if norm == 0:
        raise ValueError('zero quaternion')
    sign = next((1 if x > 0 else -1 for x in reversed(q) if abs(x) > 1e-12), 1)
    return [round(sign*x/norm, 6) or 0.0 for x in q]


def physical_goal(record):
    goal = record.get('requested_goal')
    if goal:
        cid = record['class_id']
        native = record.get('goal_group') or {}
        if cid.startswith('inter_object_relation_satisfaction-') and not {'target', 'partner'} <= goal.keys():
            raise ValueError('IORS goal lacks target or partner')
        if cid.startswith(('mental_rotation-', 'object_scaling-')):
            # HeldGoalPlan.summary() exposes the selected base separately from
            # requested_goal. This is goal context, not the sampled robot start.
            base = native.get('base_pose2d')
            if not isinstance(base, (tuple, list)) or len(base) != 3 or not all(math.isfinite(x) for x in base):
                raise ValueError('held-object native goal lacks selected base_pose2d')
            yaw = math.atan2(math.sin(base[2]), math.cos(base[2]))
            if abs(abs(yaw)-math.pi) < 1e-12:
                yaw = -math.pi
            return canonical({'target': goal, 'base_pose2d': [base[0], base[1], yaw]})
        if cid.startswith('configuration_matching-block_design'):
            pattern = native.get('pattern')
            if not isinstance(pattern, (tuple, list)) or len(pattern) != 4 or not all(isinstance(x, str) and x for x in pattern):
                raise ValueError('block native goal lacks desired four-slot pattern')
            if not {'slots', 'board'} <= goal.keys() or len(goal['slots']) != 4:
                raise ValueError('block native goal lacks board and four goal slots')
            # pattern_id labels alone cannot create diversity: compare the
            # actual per-slot desired face/color values from BlockGoalPlan.
            return canonical({**goal, 'pattern': list(pattern)})
        return canonical(goal)
    if record['class_id'].startswith('camera_relocalization-'):
        targets = record.get('pose_targets') or {}
        if not targets:
            raise ValueError('camera goal pose absent')
        return canonical({k: {f: v[f] for f in ('position', 'orientation')} for k, v in targets.items()})
    raise ValueError('requested physical goal absent')


def goal_key(record):
    return hashlib.sha256(json.dumps([record['class_id'], physical_goal(record)], sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def equivalent_goal(a, b, *, position_m=.001, angle_rad=math.pi/1800,
                    joint_units=.0001):
    """Reject numerical goal jitter as diversity, with sign-invariant rotations.

    These are duplicate-rejection tolerances, not task success tolerances.
    A different class never shares a quota. Structure changes fail closed.
    """
    if a['class_id'] != b['class_id']:
        return False

    def same(x, y, field=None):
        if isinstance(x, dict):
            if not isinstance(y, dict) or set(x) != set(y):
                raise ValueError('physical goal structures differ')
            return all([same(x[k], y[k], k) for k in x])
        if isinstance(x, list):
            if not isinstance(y, list) or len(x) != len(y):
                raise ValueError('physical goal vector shapes differ')
            if field in ('position', 'pos'):
                return math.dist(x, y) <= position_m
            if field in ('orientation', 'ori'):
                return angle(x, y) <= angle_rad
            if field == 'base_pose2d':
                return (math.dist(x[:2], y[:2]) <= position_m and
                        abs(math.atan2(math.sin(x[2]-y[2]), math.cos(x[2]-y[2]))) <= angle_rad)
            if len(x) == 2 and all(isinstance(v, list) for v in x) and len(x[0]) == 3 and len(x[1]) == 4:
                return same(x[0], y[0], 'position') and same(x[1], y[1], 'orientation')
            return all([same(u, v, field) for u, v in zip(x, y)])
        if isinstance(x, (float, int)) and not isinstance(x, bool):
            if not isinstance(y, (float, int)) or isinstance(y, bool):
                raise ValueError('physical goal scalar types differ')
            return abs(x-y) <= joint_units
        return x == y

    return same(physical_goal(a), physical_goal(b))


def angle(a, b):
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        raise ValueError('zero quaternion')
    return 2*math.acos(min(1., abs(sum(x*y for x, y in zip(a, b))/(na*nb))))


def pose_delta(a, b):
    if not a or not b:
        return None
    ap = a.get('position', a.get('pos')); bp = b.get('position', b.get('pos'))
    aq = a.get('orientation', a.get('ori')); bq = b.get('orientation', b.get('ori'))
    if any(x is None for x in (ap, bp, aq, bq)):
        return None
    return {'position_m': math.dist(ap, bp), 'angle_rad': angle(aq, bq)}


def vector_delta(a, b):
    if a is None or b is None:
        return None
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return None
        return max((abs(a[k]-b[k]) for k in a), default=0.)
    if len(a) != len(b):
        return None
    return max((abs(x-y) for x, y in zip(a, b)), default=0.)


def start_comparison(a, b, *, position_m=.01, angle_rad=math.pi/180, joint_units=.01):
    """Conservative operational thresholds, explicitly recorded for review.

    Joint units are the native saved units (angular or prismatic); no joint type
    is inferred. Missing state is unresolved, never silently called a duplicate.
    """
    deltas = {}; missing = []; reasons = []
    def pose(name, x, y):
        delta = pose_delta(x, y)
        if delta is None:
            missing.append(name); return
        deltas[name] = delta
        if delta['position_m'] > position_m or delta['angle_rad'] > angle_rad:
            reasons.append(name)
    ao, bo = a.get('object_states', {}), b.get('object_states', {})
    if set(ao) != set(bo):
        missing.append('object_inventory')
    for name in sorted(set(ao) & set(bo)):
        pose('object:'+name, ao[name].get('root_link'), bo[name].get('root_link'))
        if ao[name].get('joint_pos') is not None or bo[name].get('joint_pos') is not None:
            delta = vector_delta(ao[name].get('joint_pos'), bo[name].get('joint_pos'))
            if delta is None:
                missing.append('object_joints:'+name)
            else:
                deltas['object_joints:'+name] = delta
                if delta > joint_units:
                    reasons.append('object_joints:'+name)
    ar = (a.get('robot_poses') or {}).get('robot', [])
    br = (b.get('robot_poses') or {}).get('robot', [])
    pose('robot_base', ar[0] if len(ar) == 1 else None, br[0] if len(br) == 1 else None)
    delta = vector_delta((a.get('robot_state') or {}).get('joint_positions'),
                         (b.get('robot_state') or {}).get('joint_positions'))
    if delta is None:
        missing.append('robot_joints')
    else:
        deltas['robot_joints'] = delta
        if delta > joint_units:
            reasons.append('robot_joints')
    ag = {(g['object'], g['arm']): g for g in a.get('actual_grasps', [])}
    bg = {(g['object'], g['arm']): g for g in b.get('actual_grasps', [])}
    if set(ag) != set(bg):
        reasons.append('grasp_object_or_arm')
    for name in sorted(set(ag) & set(bg)):
        pose('grasp:'+str(name), ag[name].get('eef_in_object'), bg[name].get('eef_in_object'))
    ah, bh = a.get('actual_articulation_grasp'), b.get('actual_articulation_grasp')
    if ah is not None or bh is not None:
        if not ah or not bh:
            missing.append('actual_articulation_grasp')
        else:
            if (ah.get('object'), ah.get('arm')) != (bh.get('object'), bh.get('arm')):
                reasons.append('actual_articulation_object_or_arm')
            for key in ('contact', 'approach', 'pinch'):
                delta = vector_delta((ah.get('initial_waypoint') or {}).get(key),
                                     (bh.get('initial_waypoint') or {}).get(key))
                if delta is None:
                    missing.append('actual_articulation:'+key)
                else:
                    deltas['actual_articulation:'+key] = delta
                    if delta > (.01 if key == 'contact' else .02):
                        reasons.append('actual_articulation:'+key)
    # Articulation may use contacts rather than a grasp sidecar. Preserve the
    # actual object-frame contact geometry as a separate comparison.
    aa = (a.get('requested_start') or {}).get('articulation_grasp')
    ba = (b.get('requested_start') or {}).get('articulation_grasp')
    if aa is not None or ba is not None:
        if not isinstance(aa, dict) or not isinstance(ba, dict):
            missing.append('articulation_grasp')
        else:
            if (aa.get('object'), aa.get('arm')) != (ba.get('object'), ba.get('arm')):
                reasons.append('articulation_grasp_object_or_arm')
            for key in ('contact_obj', 'approach_dir_obj', 'pinch_axis_obj'):
                delta = vector_delta(aa.get(key), ba.get(key))
                if delta is None:
                    missing.append('articulation_grasp:'+key)
                else:
                    deltas['articulation_grasp:'+key] = delta
                    if delta > (.01 if key == 'contact_obj' else .02):
                        reasons.append('articulation_grasp:'+key)
    above_noise = any((v['position_m'] > .0001 or v['angle_rad'] > math.pi/1800)
                      if isinstance(v, dict) else v > .0001 for v in deltas.values())
    return {'different': bool(reasons), 'comparable': not missing,
            'within_noise': not above_noise and not reasons and not missing, 'reasons': reasons,
            'missing': missing, 'deltas': deltas,
            'thresholds': {'position_m': position_m, 'angle_rad': angle_rad, 'joint_native_units': joint_units}}
