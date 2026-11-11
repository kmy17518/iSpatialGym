"""Audited free-start proposals for the ready jar/fridge recovery."""
from scripts.campaign_20261110.contracts import require


VARIANT = 'inter_object_relation_satisfaction-inside_large_scale-jar_of_spaghetti_sauce-fridge-countertop'


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
