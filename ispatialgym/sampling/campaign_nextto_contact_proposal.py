"""Rank existing swivel-chair push candidates by one audited successful contact.

This opt-in proposal never changes a contact, stance, goal or feasibility check.
"""
import math
import os

from ispatialgym.sampling.campaign_held_goals import _require

FIELD = 'nextto_contact_source_proposal'
CLASS_ID = 'inter_object_relation_satisfaction-nextto_large_scale_01'


def checked_hint(template):
    hint = template.scene_setup.get(FIELD)
    if hint is None:
        return None
    from ispatialgym.sampling.campaign_runtime import canonical_scope
    # train_r6: the train clone of the exact canonical template keeps its audited contact (canonical_scope).
    _require(template.class_id == CLASS_ID and canonical_scope(template)
             and template.task == 'inter_object_relation_satisfaction-nextto_large_scale',
             'nextto contact proposal has wrong class/task/split')
    target = [o for o in template.objects if o.role == 'target']
    _require(len(target) == 1 and target[0].category == 'swivel_chair'
             and target[0].model == 'iiihwn', 'nextto contact proposal target differs')
    _require(isinstance(hint, dict), 'invalid nextto contact proposal')
    for key in ('tro_sha256', 'summary_sha256', 'sidecar_sha256', 'demo_sha256', 'own_replay_gate_sha256'):
        h = hint.get(key)
        _require(isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h),
                 'nextto contact proposal lacks ' + key)
    _require(hint.get('face') in ('+x', '-x', '+y', '-y') and hint.get('hands') == 2,
             'nextto contact proposal lacks exact two-hand face')
    for key in ('height_m', 'spacing_m', 'stand_off_m', 'translate', 'tilt'):
        v = hint.get(key)
        _require(type(v) in (float, int) and math.isfinite(v), 'invalid nextto proposal ' + key)
    _require(all(hint[k] > 0 for k in ('height_m', 'spacing_m', 'stand_off_m')),
             'nextto contact dimensions must be positive')
    return dict(hint)


def configured_hint():
    if not os.environ.get('ISPATIALGYM_CAMPAIGN_CONFIG'):
        return None
    from ispatialgym.sampling.campaign_runtime import configuration
    return checked_hint(configuration()[1])


def initial_stance_batches(template):
    """Spend a bounded independent start search before discarding good contacts.

    A start is independent of the manipulation contact. The old loop spent 30
    draws per contact and discarded feasible low contacts before a later batch
    found a visible start. Reuse is local to one unchanged target/goal pair.
    """
    enabled = template.scene_setup.get('recovery_options', {}).get('nextto_reuse_initial_stance', False)
    _require(type(enabled) is bool, 'nextto initial stance option must be boolean')
    if not enabled:
        return 1
    _require(checked_hint(template) is not None, 'initial stance reuse requires audited contact proposal')
    return 12


class CheckedInitialStance:
    """Cache only a successfully checked start for one target/goal pair."""
    def __init__(self, batches):
        self.batches = batches
        self._value = None
        self._failure = None

    def get(self, sample, rejection):
        if self.batches == 1:
            return sample()
        if self._failure is not None:
            raise self._failure
        if self._value is None:
            for _ in range(self.batches):
                try:
                    self._value = sample()
                    break
                except rejection as failure:
                    self._failure = failure
            if self._value is None:
                raise self._failure
            self._failure = None
        return self._value


def prioritize(candidates, face, hint):
    """Move at most two exact-contact existing candidates first; retain every other.

    The caller has already checked floor, reach, camera and body collisions. All
    ordinary IK, complete swept path, navigation and final checks still follow.
    """
    if hint is None or face != hint['face']:
        return candidates
    eligible = []
    for i, (st, info) in enumerate(candidates):
        pair = info.get('pair')
        if not isinstance(pair, dict) or len(info.get('arms', [])) != 2:
            continue
        if (abs(float(pair['height']) - hint['height_m']) > 1e-6
                or abs(float(pair['spacing']) - hint['spacing_m']) > 1e-6):
            continue
        score = (abs(float(st.stand_off) - hint['stand_off_m']),
                 abs(float(st.translate) - hint['translate']) + abs(float(st.tilt) - hint['tilt']))
        eligible.append((score, i))
    chosen = [i for _, i in sorted(eligible)[:2]]
    if not chosen:
        return candidates
    used = set(chosen)
    return [candidates[i] for i in chosen] + [c for i, c in enumerate(candidates) if i not in used]
