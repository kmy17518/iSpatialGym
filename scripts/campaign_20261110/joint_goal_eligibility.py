"""Bind a fixed, physically probed opening goal and its descriptive bucket."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from scripts.campaign_20261110.contracts import digest, file_digest, require

FIELD = 'standard_native_joint_goal'
SCHEMA = 'isg-standard-native-opening-eligibility-v1'


def measure(template, tro, metadata):
    require(template.split in ('eval_standard', 'eval_model', 'eval_category')
            and template.task == 'configuration_matching-articulation_open_small_scale',
            'fixed opening eligibility requires evaluated small articulation opening')
    targets = [o for o in template.objects if o.role == 'target']
    require(len(targets) == 1 and targets[0].joint, 'ambiguous opening target joint')
    target = targets[0]
    raw = tro['ispatialgym']; binding = raw['object_bindings'][target.instance]
    require((binding['category'], binding['model']) == (target.category, target.model), 'opening source model differs')
    runtime = raw['campaign']['runtime']; names = runtime['joint_names'][target.instance]
    values = runtime['planned_goal_joints'][target.instance]
    require(len(names) == len(values) and target.joint in names, 'opening source joint vector differs')
    goal = float(values[names.index(target.joint)])
    facts = metadata['template_facts']; lower, upper = facts['limits']
    require(facts['direction'] == 'open' and facts['joint'] == target.joint
            and all(math.isfinite(v) for v in (lower, upper, goal)) and lower < goal <= upper,
            'invalid native opening goal/limits')
    pairs = facts['artsmall_draw']['pairs']
    matching = [p for p in pairs if abs(p['q_goal']-goal) <= 1e-6 and p['q_init'] < p['q_goal']]
    require(matching, 'fixed opening goal was not physically probed')
    fraction = (goal-lower)/(upper-lower)
    bucket = 0 if fraction <= .5 else 1
    return {'goal':goal,'limits':[lower,upper],'fraction':fraction,'bucket':bucket,
            'target_sha256':digest(asdict(target))}


def bind_opening(template, anchor):
    meta = Path(anchor['template_dir'])/'template_meta.json'
    tro = Path(anchor['sources']['tro']['path'])
    require(file_digest(tro) == anchor['sources']['tro']['sha256'] and json.loads(meta.read_text()) == anchor['template_meta'],
            'opening source changed after anchor audit')
    measured = measure(template, json.loads(tro.read_text()), anchor['template_meta'])
    if FIELD in template.scene_setup:
        require(eligible_buckets(template) == (measured['bucket'],), 'fixed opening bucket changed')
        return template
    record = {'schema':SCHEMA,**measured,'sources':{
        'tro':{'path':str(tro.resolve()),'sha256':file_digest(tro)},
        'metadata':{'path':str(meta.resolve()),'sha256':file_digest(meta)}},
        'meaning':'Immutable physically probed goal; bucket label is descriptive, not an allocation limit.'}
    return replace(template,scene_setup={**template.scene_setup,FIELD:record})


def bind_standard_opening(template, anchor):
    require(template.split == 'eval_standard', 'standard opening author requires standard split')
    return bind_opening(template, anchor)


def eligible_buckets(template):
    record = getattr(template,'scene_setup',{}).get(FIELD)
    if record is None or template.split not in ('eval_standard', 'eval_model', 'eval_category'):return None
    require(record.get('schema') == SCHEMA, 'invalid native opening eligibility')
    sources={}
    for name,source in record['sources'].items():
        require(file_digest(source['path']) == source['sha256'], 'native opening source bytes changed')
        sources[name]=json.loads(Path(source['path']).read_text())
    measured=measure(template,sources['tro'],sources['metadata'])
    require(all(record[k] == v for k,v in measured.items()), 'native opening eligibility differs from source')
    return (measured['bucket'],)
