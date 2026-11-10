#!/usr/bin/env python3
"""Compile reviewed extra train models across the existing ten-room plans."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
B1K=Path('/home/minyeongk/Desktop/projects/Behavior-1K-grasp-bands-6a6194600')
sys.path[:0]=[str(ROOT),str(B1K/'bddl3')]
from contracts import ObjectRef,Template,digest
from readiness_catalog import ReadinessCatalog
from ispatialgym.grasping.manifest import floating_job_of
from ispatialgym.grasping.schema import entry_id
from ispatialgym.sampling.bddl_from_iddl import RoleGrounding,render
from ispatialgym.sampling.campaign_obstacles import with_stationary_goals
from ispatialgym.sampling.iddl import parse_file
from ispatialgym.sampling.knowledge import synset_of_category
from ispatialgym.sampling.v2.problem import GroundedProblem,SUBTASK_FAMILY


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--defaults',type=Path,required=True)
    p.add_argument('--visual-review',type=Path,required=True)
    p.add_argument('--reservations',type=Path,required=True)
    p.add_argument('--reserved-asset-hashes',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new immutable output required')
    inv=ROOT/'tmp/campaign_20261110/inventory';respath=a.reservations;reserved=json.loads(respath.read_text())
    hashpath=a.reserved_asset_hashes
    hashes=json.loads(hashpath.read_text());measured=json.loads(a.defaults.read_text());review=json.loads(a.visual_review.read_text())
    if hashes['reservation_sha256']!=sha(respath):raise ValueError('reservation and actual asset hashes do not match')
    if measured['status']!='native_defaults_and_geometry_recorded' or measured['scope']!='train_extra_rooms' or measured['reservation_sha256']!=sha(inv/'split_reservations.json'):
        raise ValueError('native defaults do not match frozen reservation')
    # Measurements predate the monotonic reservation extension. Re-audit every
    # model and hash against the supplied current policy below before use.
    original_policy=json.loads((inv/'split_reservations.json').read_text())
    for key in ('model_holdout_models','category_holdout_models','train_exclude_models','train_exclude_categories'):
        if not set(original_policy[key])<=set(reserved[key]):raise ValueError('reservation policy is not a monotonic extension')
    if original_policy['category_holdout_rooms']!=reserved['category_holdout_rooms']:raise ValueError('room policy changed since native measurements')
    if review['default_geometry_sha256']!=sha(a.defaults):raise ValueError('visual review does not match native measurement')
    if len(measured['objects'])!=64:raise ValueError('expected64 selected model measurements')
    records={r['key']:r for r in json.loads((inv/'cluster_templates_with_scales.json').read_text())['records']}
    source_path=inv/'frozen_customs_f24ea424.json';originals=json.loads(source_path.read_text())
    planpath=inv/'train_room_plan.json';plan=json.loads(planpath.read_text())
    readiness=ReadinessCatalog(B1K/'bddl3/bddl/generated_data');scene_cache={};floor_cache={}
    excluded=set(reserved['train_exclude_models']);excluded_categories=set(reserved['train_exclude_categories']);excluded_hashes=set(hashes['reserved_asset_hashes'])
    held_rooms={(r['scene'],r['room']) for r in reserved['category_holdout_rooms']}
    a.out.mkdir(parents=True);jobs,templates,floating,blocked=[],[],[],[]
    for index,row in enumerate(measured['objects']):
        cls,task,role=row['class_id'],row['task'],row['role'];cat,model=row['category'],row['model'];key=cat+'/'+model
        if key in excluded or cat in excluded_categories or row['asset_sha256'] in excluded_hashes or not row['ready']:
            raise ValueError('unready or reserved train candidate')
        if review['objects'][key]['status']!='passed':blocked.append({'model_key':key,'review':review['objects'][key]});continue
        record=records[cls];alias=record['row']['alias'];entry=copy.deepcopy(originals[alias]);iddl=parse_file(str(ROOT/'ispatialgym/assets/iddl'/entry['iddl']))
        spec=entry['roles'][role];spec.update(representative=model,dataset='behavior-1k-assets',scale={key:row['default_scale']})
        spec.pop('grasp_keys',None);spec.pop('orientation_equivalences',None);gkey=None
        if task=='object_scaling':
            source_key=record['metadata']['choices']['target']['grasp_key'];base=floating_job_of(source_key,str(ROOT/'runlogs/grasp_library_20260905'),held_tasks=[task])
            if base is None:raise ValueError('missing source floating ranking metadata')
            gkey=entry_id(cat,model,row['default_scale'],'floating');spec['grasp_keys']={key:gkey}
            floating.append({'id':gkey,'legacy_key':gkey,'category':cat,'model':model,'scale':row['default_scale'],'dataset':'behavior-1k-assets',
                'constraint':'floating','kind':'floating','rest_hint_wxyz':None,'rest_pose':None,'floating_mode':base['floating_mode'],'held_tasks':[task],
                'source':'campaign_extra_train_model','asset_sha256':row['asset_sha256'],'ready':True,'readiness_source':row['readiness_source'],
                'legacy_key_note':'Unique new routing alias, no legacy candidates/grasps imported.'})
        entry['constraints']=[]
        for k in list(entry):
            if k.startswith('fallback') or k=='superseded_by':entry.pop(k)
        for k in list(entry['generation']):
            if k.startswith('_') or k=='remove_objects':entry['generation'].pop(k)
        entry['generation'].update(templates_max=1,instances_per_template=1)
        entry['roles']={r:entry['roles'][r] for r in iddl.roles if r in entry['roles']}|{r:s for r,s in entry['roles'].items() if r not in iddl.roles}
        entry['scenes']={}
        for anchor in plan['tasks'][task]:
            scene,room=anchor['scene'],anchor['room']
            if (scene,room) in held_rooms:raise ValueError('train room is category-heldout')
            sb=entry['scenes'].setdefault(scene,{'rooms':[],'bindings':{},'whitelist':{role:{cat:[model]}},'blacklist':{}})
            sb['rooms'].append(room);sb['bindings']['floor']={'room':sb['rooms'][0]}
        did=25000+index;entry['definition_id']=did;groundings={}
        for name in iddl.roles:
            group=entry['generation'].get('groups',{}).get(name)
            if group:groundings[name]=RoleGrounding(name,'group',members=[(c,synset_of_category(c)) for c,n in group['members'].items() for _ in range(n)])
            else:
                c=entry['roles'][name]['category'];groundings[name]=RoleGrounding(name,'sampleable',category=c,synset=synset_of_category(c))
        activity=SUBTASK_FAMILY[task][0];rendered=render(iddl,groundings,did,alias,activity);text=rendered.text
        if task.endswith('obstructed'):text=with_stationary_goals(text,rendered.instances['obstacles.members'])
        directory=a.out/model;bddl=directory/'bddl'/activity/f'problem{did}.bddl';bddl.parent.mkdir(parents=True);bddl.write_text(text)
        entry['campaign_provenance']={'source_class':cls,'source_customs_sha256':sha(source_path),'reservation_sha256':sha(respath),
            'default_geometry_sha256':sha(a.defaults),'visual_review_sha256':sha(a.visual_review),'room_plan_sha256':sha(planpath),'status':'candidate_only'}
        custom=directory/'candidate_customs.json';custom.write_text(json.dumps({alias:entry},indent=2)+'\n')
        for anchor in plan['tasks'][task]:
            scene,room=anchor['scene'],anchor['room']
            if scene not in scene_cache:
                sp=B1K/'datasets/2026-challenge-task-instances/scenes'/scene/'json'/f'{scene}_stable.json'
                scene_cache[scene]=(json.loads(sp.read_text())['objects_info']['init_info'],str(sp),sha(sp))
            init,scene_path,scene_hash=scene_cache[scene]
            if (scene,room) not in floor_cache:
                floor=next(v['args'] for v in init.values() if v['args'].get('category')=='floors' and room in v['args'].get('in_rooms',[]))
                st=readiness.status('floors',floor['model'])
                if not st['ready']:raise ValueError('unready actual scene floor')
                usd=B1K/'datasets/behavior-1k-assets/objects/floors'/floor['model']/'usd'/(floor['model']+'.encrypted.usd')
                floor_cache[scene,room]=(floor,st,sha(usd))
            floor,st,floor_hash=floor_cache[scene,room]
            refs=[ObjectRef(instance='floor.n.01_1',role='support',category='floors',model=floor['model'],asset_sha256=floor_hash,
                default_scale=tuple(floor['scale']),semantic_kind='floor',ready=True,readiness_source=st['readiness_source'])]
            names=rendered.instances.get(role+'.members') or [rendered.instances[role]]
            for name in names:refs.append(ObjectRef(instance=name,role='target',category=cat,model=model,asset_sha256=row['asset_sha256'],default_scale=tuple(row['default_scale']),ready=True,readiness_source=row['readiness_source']))
            tid='train_extra_room__'+digest({'class_id':cls,'model':model,'scene':scene,'room':room,'objects':[asdict(o) for o in refs]})[:20]
            template=Template(template_id=tid,task=task,class_id=cls,split='train',scene=scene,room=room,objects=tuple(refs),
                definition_sha256=digest({'entry':entry,'bddl_sha256':sha(bddl)}),scene_setup={'task_rooms':[room],'inst_to_name':{'floor.n.01_1':floor['name']},
                    'runtime_bddl_sha256':sha(bddl),'customs_alias':alias,'room_plan':anchor,'native_scene_path':scene_path,'native_scene_sha256':scene_hash,
                    'release_status':'candidate_only','fresh_metadata_binding_check':'required','fresh_demo':'pending','fresh_replay':'pending'})
            templates.append({**asdict(template),'fingerprint':template.fingerprint})
            jobs.append({'candidate_id':tid,'class_id':cls,'task':task,'alias':alias,'scene':scene,'room':room,'model':model,
                'relative_materialized':model,'template_fingerprint':template.fingerprint,'production_released':False,
                'status':'awaiting_fresh_floating_library' if gkey else 'ready_for_fresh_geometry','required_new_grasp_key':gkey,
                'environment':{'ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL':'1'} if task.endswith('obstructed') else {},
                'sample_argv':['python','-m','ispatialgym.sampling.sample_isg','--variant',alias,'--customs','${MATERIALIZED}/'+model+'/candidate_customs.json',
                    '--bddl-root','${MATERIALIZED}/'+model+'/bddl','--scene',scene,'--room',room,'--templates','1','--instances-per-template','1',
                    '--max-attempts','3','--seed','20261110','--no-write-back','--out','${CAMPAIGN_OUT}/validation/'+tid]})
            gp=GroundedProblem.load(alias,str(custom),str(directory/'bddl'));gp.override_scene_room(scene,room)
            if gp.rooms(scene)!=[room]:raise ValueError('native room override failed')
    if len({t['template_id'] for t in templates})!=len(templates):raise ValueError('duplicate concrete template identity')
    for filename,data in [('validation_jobs.json',{'jobs':jobs,'blocked':blocked,'reservation_sha256':sha(respath)}),
        ('contract_candidates.json',{'templates':templates}),('floating_jobs.json',floating)]:
        (a.out/filename).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'new_models':len(templates)//10,'new_room_candidates':len(templates),'fresh_floating_jobs':len(floating),'blocked':blocked,'production_released':0}))


if __name__=='__main__':main()
