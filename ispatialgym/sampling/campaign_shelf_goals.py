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
    return {'mode': 'native_shelf_tip', 'support_name': support_name,
            'grasp_key': board['grasp_key'], 'row_z': board['row_z'], 'bay_m': start['bay_m'],
            'target_board': start['target_board'], 'arm': start['arm'],
            'target_lean_deg': start['target_lean_deg'], 'partner_lean_deg': start['partner_lean_deg'],
            'spine_behind_edge_m': start['spine_behind_edge_m'],
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
