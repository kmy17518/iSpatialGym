"""A bounded alternate pre-contact pose; every physical checker is reused."""
from dataclasses import replace
import os


def program_witness_selection_enabled():
    if not (os.environ.get('ISPATIALGYM_CAMPAIGN_CONFIG') and
            os.environ.get('ISPATIALGYM_AXIAL_PROGRAM_WITNESS_SELECTION', '0') == '1'):
        return False
    from ispatialgym.sampling import campaign_runtime
    return campaign_runtime.configuration()[1].class_id == 'alignment-axial_04'


def path_selection_enabled():
    return program_witness_selection_enabled() and os.environ.get(
        'ISPATIALGYM_AXIAL_PATH_SELECTION', '0') == '1'


def checked_stance_limits():
    """Bound a prospective search after the strict endpoint/path screens.

    The earlier wider search used the failing 10 cm/open-finger endpoint
    checks. R10's corrected checks reached an arc but could not retreat from
    its nearest stances. Permit more *existing checked* Phase-2 candidates;
    every map, IK, path and settled-program check still applies.
    """
    if path_selection_enabled() and os.environ.get('ISPATIALGYM_AXIAL_CHECKED_STANCE_SEARCH', '0') == '1':
        return 256, 24
    return 64, 6


def check_program_push_path(sampler, chain, stance, direction, target, support):
    """Reject proposals that already fail the unchanged native program path.

    Axial04 r9 passed endpoint checks but failed the escape from the arc end,
    including the alternative branch the program would actually choose. Try
    the next existing orientation/stance instead of committing that proposal.
    The complete settled-world program check still runs afterward.
    """
    from ispatialgym.sampling.v2 import program_check as PC
    from ispatialgym.oracle.actions import RETREAT_DIST
    from ispatialgym.oracle.execute import push_speed_of
    from types import SimpleNamespace

    checker = sampler.ikc
    key = ('right', checker._pose_key(chain[0]))
    q_pre = checker.solutions.get(key)
    if q_pre is None:
        return False, 'program path proposal has no pre-contact witness', {}
    tol = (sampler.gp.gen('tolerances') or {}).get('orientation')
    band = float(tol) * PC._push_stop_frac_arc() if tol else .005
    lag = float(sampler.gp.gen('push_arc_replay_lag_rad') or 0.)
    speed = push_speed_of(SimpleNamespace(extra={'push_speed_m_s': sampler.gp.gen('push_speed_m_s')}))
    ignored = [support, target]
    reports = []

    def replay(q_from, line=False):
        ok, why, report = PC.replay_arc_push(
            sampler, 'right', q_from, chain[0], chain[1:], direction, target,
            list(sampler.support_objects()), ignored, band, RETREAT_DIST,
            speed_m_s=speed, line=line, lag_allowance_rad=lag)
        reports.append({'ok': bool(ok), 'reason': why, 'report': report})
        return ok, why, report

    ok, why, _ = replay(q_pre)
    if not ok:
        return False, why, {'checks': reports}
    single_q = {'pre': q_pre}
    branch = PC._rewitness_flipped_push(sampler, chain, stance, 'right', q_pre,
                                      0., replay, {}, single_q, 'pre', ignored)
    if branch is not None and not branch[0]:
        return False, branch[1], {'checks': reports}
    checker.solutions[key] = single_q['pre'].clone()
    return True, '', {'checks': reports, 'branch': branch}


def polish_program_endpoints(checker, chain, *, arm, ignored):
    """Select only endpoints that survive the existing program witness check.

    Native polishing already retains a candidate only after its collision and
    joint-margin checks. Publish both solution updates only if both endpoints
    meet the unchanged program tolerance. This is proposal rejection, never a
    replacement for the complete program check.
    """
    selected = []
    for label, pose in [('pre', chain[0]), ('goal', chain[-1])]:
        key = (arm, checker._pose_key(pose))
        witness = checker.solutions.get(key)
        if witness is None:
            return False, label + ': missing endpoint witness'
        polished, residual = checker.polish_witness(witness, pose, arm, ignore=ignored)
        if residual > checker.WITNESS_RESIDUAL_MAX:
            return False, (f'{label}: program witness FK residual {residual * 1000:.3f}mm '
                           f'> {checker.WITNESS_RESIDUAL_MAX * 1000:.3f}mm')
        selected.append((key, polished))
    for key, polished in selected:
        checker.solutions[key] = polished.clone()
    return True, ''


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
