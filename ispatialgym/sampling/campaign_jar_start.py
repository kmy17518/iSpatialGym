"""Audited free-start proposals for the ready jar/fridge recovery."""
from scripts.campaign_20261110.contracts import require


VARIANT = 'inter_object_relation_satisfaction-inside_large_scale-jar_of_spaghetti_sauce-fridge-countertop'


def rank_initial_edges(sampler, candidates):
    """Prefer an observed placement component; keep every original edge.

    The reference is an exact native navigation diagnostic, not an accepted
    trajectory. This only orders the existing checked support-edge proposals;
    the current draw still recomputes its placement, carry and full path checks.
    """
    cfg = sampler.gp.gen('free_start')
    hint = cfg.get('prefer_placement_component') if isinstance(cfg, dict) else None
    if hint is None:
        return candidates
    require(sampler.gp.variant == VARIANT, 'placement component hint belongs to the audited jar variant')
    require(isinstance(hint, dict) and set(hint) == {'source_path', 'source_sha256'},
            'placement component hint needs one exact diagnostic source')
    from pathlib import Path
    import hashlib
    import json
    from ispatialgym.sampling.v2 import navigation as nav

    path = Path(hint['source_path']).resolve()
    require(path.is_relative_to('/vision/u/minyeong/projects/isg_home/campaign_20261110/diagnostics'),
            'placement component diagnostic is outside the owned campaign')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == hint['source_sha256'], 'placement component diagnostic changed')
    proof = json.loads(raw)
    require(proof.get('campaign_id') == 'campaign_20261110'
            and proof.get('status') == 'native_disconnection_captured', 'placement component diagnostic is incomplete')
    require(proof.get('carried') == 'jar__of__spaghetti_sauce.n.01_1'
            and {'countertop_tpuwys_0', 'fridge_xyejdx_0'} <= set(proof.get('objects', {})),
            'placement component diagnostic object identities differ')
    goal = proof.get('goal_pose')
    require(isinstance(goal, list) and len(goal) == 3, 'placement component diagnostic lacks native goal base')
    return _rank_component_edges(sampler, candidates, goal[:2], nav)


def _rank_component_edges(sampler, candidates, goal_xy, nav):
    free = nav.eroded_map(sampler.kit)
    goal = nav.nearest_free_cell(free, nav._to_map(sampler.kit, goal_xy), 20)
    rows = list(candidates)
    ranked = []
    for ordinal, (edge, stance) in enumerate(rows):
        start = nav.nearest_free_cell(free, nav._to_map(sampler.kit, stance.base_pose2d[:2]), 20)
        connected = start is not None and goal is not None and nav.grid_shortest_path(free, start, goal) is not None
        ranked.append((not connected, ordinal, rows[ordinal]))
    ranked.sort(key=lambda item: item[:2])
    sampler.template_facts['placement_component_edge_order'] = {
        'native_goal_base_xy': list(goal_xy), 'original': [edge.name for edge, _ in rows],
        'ordered': [item[2][0].name for item in ranked],
        'connected': [item[2][0].name for item in ranked if not item[0]],
        'all_original_candidates_retained': True, 'full_native_navigation_required': True}
    return [item[2] for item in ranked]


def connection_filter(sampler, manipulation):
    """Require a component containing a native B, without certifying navigation.

    The unchanged free-floor draw could accept a visible A south of the
    countertop while every offered B stood north of it. The subsequent native
    path checks rejected those disconnected pairs. This opt-in proposal screen
    retains the original view, body, sweep and full navigation checks.
    """
    cfg = sampler.gp.gen('free_start')
    enabled = cfg.get('connected_to_manipulation', False) if isinstance(cfg, dict) else False
    require(type(enabled) is bool, 'connected free start must be boolean')
    if not enabled:
        return lambda stance: True
    require(sampler.gp.variant == VARIANT,
            'connected free start belongs to the audited ready jar/fridge variant')
    require(manipulation, 'connected free start requires native manipulation candidates')
    from ispatialgym.sampling.v2 import navigation as nav

    free = nav.eroded_map(sampler.kit)
    goals = set()
    for stance in manipulation:
        cell = nav.nearest_free_cell(free, nav._to_map(sampler.kit, stance.base_pose2d[:2]), 20)
        if cell is not None:
            goals.add(tuple(cell))
    require(goals, 'native manipulation pool has no eroded-map anchors')
    stats = {'tested_cells': 0, 'disconnected_cells': 0,
             'manipulation_anchor_cells': [list(c) for c in sorted(goals)],
             'full_navigation_check_required': True}
    sampler.template_facts['connected_free_start'] = stats
    cache = {}

    def accepts(stance):
        start = nav.nearest_free_cell(free, nav._to_map(sampler.kit, stance.base_pose2d[:2]), 20)
        key = None if start is None else tuple(start)
        if key not in cache:
            stats['tested_cells'] += 1
            cache[key] = key is not None and any(
                nav.grid_shortest_path(free, key, goal) is not None for goal in sorted(goals))
            stats['disconnected_cells'] += int(not cache[key])
        return cache[key]

    return accepts
