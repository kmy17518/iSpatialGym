#!/usr/bin/env python3
"""Materialize 29 reserved room-family model holds from measured default loads."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
B1K = Path('/home/minyeongk/Desktop/projects/Behavior-1K-grasp-bands-6a6194600')
sys.path[:0] = [str(ROOT), str(B1K/'bddl3')]
from contracts import ObjectRef, Template, digest
from ispatialgym.grasping.manifest import floating_job_of
from ispatialgym.grasping.schema import entry_id
from ispatialgym.sampling.bddl_from_iddl import RoleGrounding, render
from ispatialgym.sampling.campaign_obstacles import with_stationary_goals
from ispatialgym.sampling.iddl import parse_file
from ispatialgym.sampling.knowledge import synset_of_category
from ispatialgym.sampling.v2.problem import GroundedProblem, SUBTASK_FAMILY


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--defaults', type=Path, required=True)
    p.add_argument('--visual-review', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('new immutable output directory required')
    inv = ROOT/'tmp/campaign_20261110/inventory'
    reservation_path = inv/'split_reservations.json'
    reservations = json.loads(reservation_path.read_text())
    measured = json.loads(a.defaults.read_text())
    if measured['status'] != 'native_defaults_and_geometry_recorded' or measured['reservation_sha256'] != sha(reservation_path):
        raise ValueError('successful default-load measurement bound to exact reservations required')
    rows = measured['objects']
    if len(rows) != 29 or len({r['class_id'] for r in rows}) != 29:
        raise ValueError('incomplete measured room-family holds')
    review=json.loads(a.visual_review.read_text())
    if review['default_geometry_sha256']!=sha(a.defaults):raise ValueError('visual review source mismatch')
    standards = {t['class_id']:t for t in json.loads((inv/'standard_contract_ready.json').read_text())['templates']}
    records = {r['key']:r for r in json.loads((inv/'cluster_templates_with_scales.json').read_text())['records']}
    source_path = inv/'frozen_customs_f24ea424.json'
    originals = json.loads(source_path.read_text())
    reserved_models = set(reservations['model_holdout_models'])
    category_models = set(reservations['category_holdout_models'])
    category_rooms = {(r['scene'],r['room']) for r in reservations['category_holdout_rooms']}
    definitions, jobs, contracts, floating, blocked = {}, [], [], [], []
    a.out.mkdir(parents=True)
    for row in rows:
        class_id, task = row['class_id'], row['task']
        record, standard = records[class_id], standards[class_id]
        alias = record['row']['alias']
        entry = copy.deepcopy(originals[alias])
        iddl = parse_file(str(ROOT/'ispatialgym/assets/iddl'/entry['iddl']))
        role, category, model = row['role'], row['category'], row['model']
        key = category+'/'+model
        decision=review['objects'][key]
        if decision['status']!='passed':
            blocked.append({'class_id':class_id,'model_key':key,**decision})
            continue
        if key not in reserved_models or key in category_models or not row['ready']:
            raise ValueError('model hold violates frozen split: '+key)
        scale = row['default_scale']
        if len(scale) != 3 or not all(float(x)>0 for x in scale):
            raise ValueError('invalid measured default scale')
        spec = entry['roles'][role]
        spec.update(representative=model, dataset='behavior-1k-assets', scale={key:scale})
        for stale in ('grasp_keys','orientation_equivalences'):
            spec.pop(stale,None)
        grasp_key = None
        if task in {'mental_rotation','object_scaling'}:
            # Reuse only the category's established candidate ranking mode;
            # geometry, candidates and physics grasps are freshly generated.
            source_key = record['metadata']['choices']['target']['grasp_key']
            source = floating_job_of(source_key,str(ROOT/'runlogs/grasp_library_20260905'),held_tasks=[task])
            if source is None or source.get('constraint') != 'floating':
                raise ValueError('source floating ranking metadata absent: '+class_id)
            grasp_key = entry_id(category,model,scale,'floating')
            spec['grasp_keys'] = {key:grasp_key}
            floating.append({'id':grasp_key,'legacy_key':grasp_key,'category':category,'model':model,'scale':scale,
                'dataset':'behavior-1k-assets','constraint':'floating','kind':'floating','rest_hint_wxyz':None,'rest_pose':None,
                'floating_mode':source['floating_mode'],'held_tasks':[task], 'source':'campaign_reserved_model',
                'source_ranking_metadata_key':source_key,'ready':True,'readiness_source':row['readiness_source'],
                'asset_sha256':row['asset_sha256'],'legacy_key_note':'Unique routing alias; no legacy grasp or geometry import.'})
        scene,room = standard['scene'],standard['room']
        if (scene,room) in category_rooms:
            raise ValueError('category-heldout room cannot host model hold')
        entry['scenes'] = {scene:{'rooms':[room], 'bindings':{'floor':{'room':room}},
            'whitelist':{role:{category:[model]}},'blacklist':{}}}
        entry['constraints'] = []
        for k in list(entry):
            if k.startswith('fallback') or k=='superseded_by':entry.pop(k)
        for k in list(entry['generation']):
            if k.startswith('_') or k=='remove_objects':entry['generation'].pop(k)
        entry['generation'].update(templates_max=1,instances_per_template=1)
        entry['roles'] = {r:entry['roles'][r] for r in iddl.roles if r in entry['roles']} | {
            r:s for r,s in entry['roles'].items() if r not in iddl.roles}
        definition_id = 22000+len(definitions)
        entry['definition_id'] = definition_id
        groundings = {}
        for name in iddl.roles:
            role_spec = entry['roles'][name]
            group = entry['generation'].get('groups',{}).get(name)
            if name=='floor':
                groundings[name] = RoleGrounding(name,'scene',same_as='floor')
            elif group:
                members=[(c,synset_of_category(c)) for c,n in group['members'].items() for _ in range(n)]
                groundings[name]=RoleGrounding(name,'group',members=members)
            else:
                cat=role_spec['category']
                groundings[name]=RoleGrounding(name,'sampleable',category=cat,synset=synset_of_category(cat))
        activity=SUBTASK_FAMILY[task][0]
        rendered=render(iddl,groundings,definition_id,alias,activity,comment='Reserved model candidate; fresh geometry/demo/replay required.')
        text=rendered.text
        if task=='camera_relocalization-obstructed':
            obstacle_instances=rendered.instances['obstacles.members']
            text=with_stationary_goals(text,obstacle_instances)
        bddl=a.out/'bddl'/activity/f'problem{definition_id}.bddl'
        bddl.parent.mkdir(parents=True,exist_ok=True);bddl.write_text(text)
        entry['campaign_provenance']={'reservation_sha256':sha(reservation_path),'default_geometry_sha256':sha(a.defaults),
            'source_customs_sha256':sha(source_path),'source_class':class_id,'status':'candidate_not_released'}
        definitions[alias]=entry
        refs=[]
        for original in standard['objects']:
            data=copy.deepcopy(original)
            if original['semantic_kind'] not in {'floor','wall'}:
                if original['role']!='target':raise ValueError('unexpected nonstructural secondary role')
                data.update(category=category,model=model,default_scale=scale,asset_sha256=row['asset_sha256'],
                            ready=True,readiness_source=row['readiness_source'])
            refs.append(ObjectRef(**{**data,'default_scale':tuple(data['default_scale'])}))
        tid='held_model__'+digest({'class':class_id,'scene':scene,'room':room,'objects':[asdict(o) for o in refs]})[:20]
        template=Template(template_id=tid,task=task,class_id=class_id,split='eval_model',scene=scene,room=room,objects=tuple(refs),
            definition_sha256=digest({'entry':entry,'bddl_sha256':sha(bddl)}),scene_setup={
                'task_rooms':[room],'runtime_bddl_sha256':sha(bddl),'customs_alias':alias,'release_status':'candidate_only',
                'reservation_sha256':sha(reservation_path),'default_geometry_sha256':sha(a.defaults),
                'model_visual_review_sha256':sha(a.visual_review),
                'fresh_metadata_binding_check':'required','fresh_demo':'pending','fresh_replay':'pending',
                'perturbations':False})
        contracts.append({**asdict(template),'fingerprint':template.fingerprint})
        jobs.append({'candidate_id':tid,'class_id':class_id,'task':task,'alias':alias,'scene':scene,'room':room,
            'split':'eval_model','template_fingerprint':template.fingerprint,'bddl_sha256':sha(bddl),
            'bddl_instances':rendered.instances,'production_released':False,
            'status':'awaiting_fresh_floating_grasp_library' if grasp_key else 'ready_for_fresh_geometry_validation',
            'required_new_grasp_key':grasp_key,'environment':{'ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL':'1'} if task=='camera_relocalization-obstructed' else {},
            'sample_argv':['python','-m','ispatialgym.sampling.sample_isg','--variant',alias,'--customs','${MATERIALIZED}/candidate_customs.json',
                '--bddl-root','${MATERIALIZED}/bddl','--scene',scene,'--room',room,'--templates','1','--instances-per-template','1',
                '--max-attempts','3','--seed','20261110','--no-write-back','--out','${CAMPAIGN_OUT}/validation/'+class_id]})
    custom_path=a.out/'candidate_customs.json';custom_path.write_text(json.dumps(definitions,indent=2)+'\n')
    for job in jobs:
        gp=GroundedProblem.load(job['alias'],str(custom_path),str(a.out/'bddl'))
        gp.override_scene_room(job['scene'],job['room'])
        if gp.rooms(job['scene']) != [job['room']]:raise ValueError('native room mismatch')
    for name,data in [('validation_jobs.json',{'jobs':jobs,'blocked':blocked,'customs_sha256':sha(custom_path),'reservation_sha256':sha(reservation_path)}),
                      ('contract_candidates.json',{'templates':contracts}),('floating_jobs.json',floating)]:
        (a.out/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'model_hold_candidates':len(jobs),'fresh_floating_jobs':len(floating),'blocked':blocked,'production_released':0}))


if __name__=='__main__':main()
