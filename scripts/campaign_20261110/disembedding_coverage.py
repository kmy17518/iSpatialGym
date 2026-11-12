"""Bind exact measured geometry and a descriptive tray coverage label."""
from dataclasses import asdict, replace
import math
from pathlib import Path
import json

from scripts.campaign_20261110.contracts import digest, file_digest, require

FIELD = 'standard_native_coverage'
SCHEMA = 'isg-standard-native-coverage-eligibility-v1'


def measure(template, profile):
    require(profile.get('schema') == 'isg-native-tray-coverage-profile-v1', 'missing native tray coverage profile')
    objects = {o.instance: o for o in template.objects if o.role in ('target', 'distractor')}
    rows = [profile['target']] + profile['distractors']
    require(len(rows) == len(objects) and {r['instance'] for r in rows} == set(objects), 'native coverage active bindings differ')
    for row in rows:
        obj = objects[row['instance']]
        require((row['category'], row['model']) == (obj.category, obj.model), 'native coverage model differs')
        require(len(row['scale']) == 3 and all(math.isclose(a, b, abs_tol=1e-6) for a,b in zip(row['scale'], obj.default_scale)),
                'native coverage default scale differs')
        require(math.isfinite(row['area_m2']) and row['area_m2'] > 0, 'invalid native object area')
    floor = profile['source_floor_area_m2']
    require(math.isfinite(floor) and floor > 0, 'invalid native tray floor area')
    coverage = sum(r['area_m2'] for r in rows) / floor
    require(0 < coverage <= 1, 'native coverage outside physical tray area')
    bucket = 0 if coverage < .30 else 1 if coverage < .45 else 2
    return coverage, bucket, digest([asdict(o) for o in template.objects if o.role in ('target', 'distractor')])


def bind_standard_coverage(template, anchor):
    require(template.task == 'disembedding' and template.split == 'eval_standard', 'coverage eligibility requires standard disembedding')
    metadata = anchor['template_meta']; path = Path(anchor['template_dir'])/'template_meta.json'
    require(json.loads(path.read_text()) == metadata, 'native coverage metadata changed')
    profile = metadata.get('template_facts', {}).get('campaign_coverage_profile', {})
    coverage, bucket, objects = measure(template, profile)
    if FIELD in template.scene_setup:
        require(eligible_buckets(template) == (bucket,), 'active selection native coverage bucket changed')
        return template
    record = {'schema': SCHEMA, 'bucket': bucket, 'coverage': coverage, 'active_objects_sha256': objects,
              'source': {'path': str(path.resolve()), 'sha256': file_digest(path)}, 'profile_sha256': digest(profile),
              'meaning': 'Exact fixed active default-scale geometry; bucket label is descriptive, not an allocation limit.'}
    return replace(template, scene_setup={**template.scene_setup, FIELD: record})


def eligible_buckets(template):
    record = getattr(template, 'scene_setup', {}).get(FIELD)
    if record is None or template.split != 'eval_standard':
        return None
    require(template.task == 'disembedding' and record.get('schema') == SCHEMA, 'invalid native coverage eligibility')
    path = Path(record['source']['path'])
    require(file_digest(path) == record['source']['sha256'], 'native coverage source bytes changed')
    profile = json.loads(path.read_text())['template_facts']['campaign_coverage_profile']
    coverage, bucket, objects = measure(template, profile)
    require(digest(profile) == record['profile_sha256'] and objects == record['active_objects_sha256']
            and coverage == record['coverage'] and bucket == record['bucket'], 'native coverage eligibility differs from measured source')
    return (bucket,)
