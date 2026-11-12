"""A bounded alternate pre-contact pose; every physical checker is reused."""
from dataclasses import replace
import os


def short_pre_fallback_enabled():
    # The integrated fallback has not yet reproduced the positive isolated
    # 5 cm experiment. Keep it diagnostic-only until native/demo/replay pass.
    return bool(os.environ.get('ISPATIALGYM_CAMPAIGN_CONFIG')) and os.environ.get(
        'ISPATIALGYM_AXIAL_SHORT_PRE_FALLBACK', '0') == '1'


def short_pre_only_enabled():
    # Reproduce the positive 5 cm-only diagnostic without first perturbing the
    # solver with failed 10 cm checks. This remains an explicit campaign canary.
    return bool(os.environ.get('ISPATIALGYM_CAMPAIGN_CONFIG')) and os.environ.get(
        'ISPATIALGYM_AXIAL_SHORT_PRE_ONLY', '0') == '1'


def choose_push_with_pre_fallback(proposals, choose, pre_pose, *, enabled, short_only=False):
    """Keep the original winning chain, or check a 5 cm approach from scratch.

    The first call uses the unchanged 10 cm proposals. Only a failed choice can
    reach the fallback. Contact, arc, goal, ranking and evaluator are identical;
    the selected full chain is recorded for the ordinary oracle to execute.
    """
    if short_only:
        shorter = [replace(p, chain=[pre_pose(p.chain[1], .05), *p.chain[1:]],
                           q_first=None, source='ik') for p in proposals]
        chosen, decision = choose(shorter)
        return chosen, {**decision, 'pre_approach_program': {
            'distance_m': .05, 'original_checked_first': False,
            'physical_checks_unchanged': True}}
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
