"""Proven source proposals for the fixed briefcase goal, never feasibility shortcuts."""
import math
import json
from pathlib import Path

from ispatialgym.sampling.campaign_held_goals import _require
from ispatialgym.sampling.campaign_tabletop_starts import prioritize_written_stances

CLASS_ID = 'inter_object_relation_satisfaction-under_large_scale_02'
FIELD = 'floor_push_source_proposal'


def checked_hint(raw, *, relation, scale):
    hint = raw.get(FIELD)
    if hint is None:
        return None
    _require(relation == 'under' and scale == 'large' and raw.get('class_id') == CLASS_ID,
             'floor push proposal belongs to the audited briefcase class')
    _require(isinstance(hint, dict), 'invalid floor push source proposal')
    sha = raw.get('anchor_sources', {}).get('summary', {}).get('sha256')
    _require(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha)
             and hint.get('source_summary_sha256') == sha, 'floor push proposal source hash mismatch')
    for key, length in [('base_pose2d', 3), ('focus', 3)]:
        value = hint.get(key)
        _require(isinstance(value, (list, tuple)) and len(value) == length and all(
            type(v) in (int, float) and math.isfinite(v) for v in value), 'invalid floor push proposal '+key)
    _require(all(type(hint.get(k)) in (int, float) and math.isfinite(hint[k]) for k in ('translate', 'tilt')),
             'invalid floor push proposal torso')
    _require(hint.get('arm') in ('left', 'right') and isinstance(hint.get('grasp_name'), str)
             and bool(hint['grasp_name']), 'floor push proposal lacks source arm/grasp provenance')
    return dict(hint)


def author_hint(anchor, class_id):
    if class_id != CLASS_ID:
        return None
    from scripts.campaign_20261110.contracts import file_digest, require
    proof = anchor['sources']['summary']
    require(file_digest(proof['path']) == proof['sha256'], 'floor push source summary changed after audit')
    summary = anchor['summary']
    require(json.loads(Path(proof['path']).read_text()) == summary, 'floor push proposal differs from audited summary')
    stance = summary['manipulation_stance']
    hint = {**{k: stance[k] for k in ('base_pose2d', 'focus', 'translate', 'tilt')},
            'arm': summary['arm']['arm'], 'grasp_name': summary['grasp']['name'],
            'source_summary_sha256': proof['sha256']}
    return checked_hint({'class_id': class_id, 'anchor_sources': anchor['sources'], FIELD: hint},
                        relation='under', scale='large')


def source_hint(sampler):
    plan = getattr(sampler, '_campaign_pair_active', None)
    return getattr(plan, FIELD, None) if plan is not None else None


def preferred_index(candidates, hint):
    """Choose an existing candidate; never synthesize a source stance."""
    return prioritize_written_stances(list(range(len(candidates))), candidates, [hint], explore=0, exploit=1)[0]


def ranked_prescreen(pre, hint):
    """Try two screened candidates near the source, then retain the native order."""
    if hint is None:
        return pre
    order = prioritize_written_stances(list(range(len(pre))), [r[1] for r in pre], [hint], explore=0, exploit=2)
    return [pre[i] for i in order]
