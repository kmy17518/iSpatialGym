"""A bounded alternate pre-contact pose; every physical checker is reused."""
from dataclasses import replace


def choose_push_with_pre_fallback(proposals, choose, pre_pose, *, enabled):
    """Keep the original winning chain, or check a 5 cm approach from scratch.

    The first call uses the unchanged 10 cm proposals. Only a failed choice can
    reach the fallback. Contact, arc, goal, ranking and evaluator are identical;
    the selected full chain is recorded for the ordinary oracle to execute.
    """
    chosen, decision = choose(proposals)
    if chosen is not None or not enabled:
        return chosen, decision
    shorter = [replace(p, chain=[pre_pose(p.chain[1], .05), *p.chain[1:]],
                       label=p.label+'_pre05', q_first=None, source='ik')
               for p in proposals]
    selected, alternative = choose(shorter)
    return selected, {**alternative, 'pre_approach_fallback': {
        'original_distance_m': .10, 'alternate_distance_m': .05,
        'original_decision': decision, 'physical_checks_unchanged': True}}
