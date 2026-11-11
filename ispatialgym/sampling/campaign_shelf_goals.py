"""Audited shelf-tip context for touching03 fixed-goal siblings.

The context fixes the physical compartment and partner, not a successful
trajectory. Every new start, grasp, carry and release still runs native checks.
"""
import math

from ispatialgym.sampling.campaign_held_goals import _require
from ispatialgym.sampling.campaign_tray_goals import pose, pose_error

CLASS = 'inter_object_relation_satisfaction-touching_03'


def checked_context(raw, *, relation, scale, release, place):
    value = raw.get('shelf_tip_context')
    if value is None:
        return None
    _require(relation == 'touching' and scale == 'small' and raw.get('class_id') == CLASS,
             'shelf context belongs only to touching03')
    _require(isinstance(value, dict) and value.get('mode') == 'native_shelf_tip', 'invalid shelf release mode')
    for key in ('summary', 'sidecar'):
        sha = raw.get('anchor_sources', {}).get(key, {}).get('sha256')
        _require(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha)
                 and value.get('source_' + key + '_sha256') == sha, 'shelf context lacks exact ' + key + ' provenance')
    for key in ('support_name', 'grasp_key', 'edge_name'):
        _require(isinstance(value.get(key), str) and value[key], 'shelf context missing ' + key)
    _require(value.get('target_board') in ('left', 'right') and value.get('arm') in ('left', 'right'),
             'shelf context has invalid board/arm')
    for key in ('row_z', 'bay_m', 'target_lean_deg', 'partner_lean_deg', 'spine_behind_edge_m'):
        x = value.get(key)
        _require(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x), 'invalid shelf ' + key)
    _require(0 < value['target_lean_deg'] < 45 and 0 < value['partner_lean_deg'] < 45
             and 0 <= value['spine_behind_edge_m'] <= .5, 'shelf lean/depth outside native bounds')
    start = pose(value.get('source_target_start'), 'shelf source target start')
    _require(place is not None and pose_error(release, place)[0] < .001
             and pose_error(release, place)[1] < .001, 'shelf tip place must equal the audited release')
    return dict(value, source_target_start=start)


def author_context(anchor, *, target_name, support_name):
    summary = anchor['summary']
    start = summary['shelf_start']
    board = start['board_draw']
    _require(summary.get('touching', {}).get('tip_release') is True, 'shelf source is not a native tip release')
    source = anchor['planned_start_poses'][target_name]
    depth = start['spine_behind_edge_m']
    proposal = start.get('target_depth_proposal')
    if proposal is not None:
        # A successful grouped member may have moved only the target's start.
        # Reusing that draw as the fixed partner's depth makes every next group
        # fail the unchanged native board-supported partner check.
        _require(proposal.get('partner_depth_unchanged') is True and proposal.get('goal_unchanged') is True,
                 'shelf source depth proposal changed partner or goal')
        fixed, drawn = proposal.get('source_depth_m'), proposal.get('target_depth_m')
        _require(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= .5
                     for x in (fixed, drawn, depth)), 'invalid shelf source depth provenance')
        # Native summary writes the drawn start depth to four decimal places.
        _require(abs(drawn-depth) <= .00005001, 'shelf drawn depth differs from source summary')
        depth = fixed
        # The drawn member start is not the geometry reference of its fixed
        # release. Its audited group summary retains that original reference.
        group = summary.get('campaign_goal_group', {})
        inherited = group.get('shelf_tip_context')
        _require(group.get('phase') == 'settled_goal_verified' and
                 group.get('relation') == 'touching' and group.get('target_name') == target_name and
                 isinstance(inherited, dict), 'shelf refill lacks a verified parent geometry context')
        parent_sha = group.get('plan_sha256')
        _require(isinstance(parent_sha, str) and len(parent_sha) == 64 and
                 all(c in '0123456789abcdef' for c in parent_sha), 'shelf parent plan identity missing')
        _require(inherited.get('support_name') == support_name and
                 inherited.get('spine_behind_edge_m') == fixed,
                 'shelf parent support or fixed depth differs')
        release = pose(group.get('requested_release'), 'shelf parent release')
        actual_release = anchor.get('planned_goal_poses', {}).get(target_name)
        _require(actual_release is not None and pose_error(release, actual_release)[0] < .001 and
                 pose_error(release, actual_release)[1] < .001,
                 'shelf parent reference does not produce the source release')
        result = dict(inherited,
            source_summary_sha256=anchor['sources']['summary']['sha256'],
            source_sidecar_sha256=anchor['sources']['sidecar']['sha256'],
            inherited_group_plan_sha256=parent_sha)
        inherited_start = result.get('source_target_start')
        if isinstance(inherited_start, (list, tuple)) and len(inherited_start) == 2:
            # Native summaries serialize the parsed pose as [position, quaternion].
            result['source_target_start'] = {'position': inherited_start[0], 'orientation': inherited_start[1]}
        checked = checked_context({'class_id': CLASS, 'anchor_sources': anchor['sources'],
                                   'shelf_tip_context': result}, relation='touching', scale='small',
                                  release=release, place=pose(group.get('requested_place'), 'shelf parent place'))
        reference = checked['source_target_start']
        result['source_target_start'] = {'position': list(reference[0]), 'orientation': list(reference[1])}
        return result
    return {'mode': 'native_shelf_tip', 'support_name': support_name,
            'grasp_key': board['grasp_key'], 'row_z': board['row_z'], 'bay_m': start['bay_m'],
            'target_board': start['target_board'], 'arm': start['arm'],
            'target_lean_deg': start['target_lean_deg'], 'partner_lean_deg': start['partner_lean_deg'],
            'spine_behind_edge_m': depth,
            'edge_name': summary['manipulation_stance']['edge'],
            'source_target_start': {'position': source[0], 'orientation': source[1]},
            'source_summary_sha256': anchor['sources']['summary']['sha256'],
            'source_sidecar_sha256': anchor['sources']['sidecar']['sha256']}


def same_start_grasp(first_pose, first_grasp, second_pose, second_grasp):
    """Conservatively avoid repeating a near-identical start with the same grasp.

    Acceptance independently compares the actual recorded start/grasp identity.
    """
    dp, dq = pose_error(first_pose, second_pose)
    return first_grasp == second_grasp and dp < .01 and dq < math.radians(5)


def positive_rest_arms(context, states, row, *, other_arm):
    """Retain native mapped arm choices without moving the fixed compartment.

    The source arm proves its original rest is available; it is not a goal
    constraint. The optional other arm remains subject to the native flag,
    positive row map, idle-arm collision, grasp, carry and release checks.
    """
    mapped = [s for s in states if row in s['rows']]
    _require(any(s['key'] == context['grasp_key']
                 and context['arm'] in s.get('arms', {}).get(row, []) for s in mapped),
             'source shelf rest/arm lacks a positive native row map')
    options = [(s, arm) for s in mapped for arm in s.get('arms', {}).get(row, [])
               if arm == context['target_board'] or other_arm]
    _require(bool(options), 'fixed shelf compartment has no permitted mapped arm')
    return options


def target_depth_proposal(context, rng):
    """Propose only the movable book's start depth near its proven source.

    Partner, compartment, row and goal remain fixed. The native shelf ray,
    support, full-object frame, grasp, carry and settled checks judge the draw.
    This does not move a final goal or change a tolerance.
    """
    depth = context['spine_behind_edge_m']
    _require(type(depth) in (int, float) and math.isfinite(depth) and 0 <= depth <= .5,
             'invalid source shelf depth')
    lo, hi = max(0., depth - .03), min(.5, depth + .03)
    value = float(rng.uniform(lo, hi))
    _require(math.isfinite(value) and lo <= value <= hi, 'invalid target shelf depth proposal')
    return value, {'source_depth_m': depth, 'interval_m': [lo, hi], 'target_depth_m': value,
                   'partner_depth_unchanged': True, 'goal_unchanged': True}
