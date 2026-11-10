#!/usr/bin/env python3
"""Compile first four category-heldout tasks in the frozen heldout room."""
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
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new private output required')
    inv=ROOT/'tmp/campaign_20261110/inventory';respath=inv/'split_reservations.json';reserved=json.loads(respath.read_text())
    defaults=json.loads(a.defaults.read_text());review=json.loads(a.visual_review.read_text())
    if defaults['status']!='native_defaults_and_geometry_recorded' or defaults['scope']!='category_rooms' or defaults['reservation_sha256']!=sha(respath):
        raise ValueError('measured category defaults do not match reservations')
    if review['default_geometry_sha256']!=sha(a.defaults):raise ValueError('visual review source mismatch')
    measured={r['task']:r for r in defaults['objects']}
    if len(measured)!=3:raise ValueError('three category room models required')
    standards={r['class_id']:r for r in json.loads((inv/'standard_contract_ready.json').read_text())['templates']}
    records={r['key']:r for r in json.loads((inv/'cluster_templates_with_scales.json').read_text())['records']}
    custom_path=inv/'frozen_customs_f24ea424.json';originals=json.loads(custom_path.read_text())
    scene,room='office_cubicles_right','conference_hall_0'
    if {'scene':scene,'room':room} not in reserved['category_holdout_rooms']:raise ValueError('unreserved category room')
    scene_path=B1K/'datasets/2026-challenge-task-instances/scenes'/scene/'json'/f'{scene}_stable.json'
    init=json.loads(scene_path.read_text())['objects_info']['init_info']
    floor=next(r['args'] for r in init.values() if r['args'].get('category')=='floors' and room in r['args'].get('in_rooms',[]))
    readiness=ReadinessCatalog(B1K/'bddl3/bddl/generated_data');status=readiness.status('floors',floor['model'])
    if not status['ready']:raise ValueError('heldout room floor not ready')
    floor_hash=sha(B1K/'datasets/behavior-1k-assets/objects/floors'/floor['model']/'usd'/(floor['model']+'.encrypted.usd'))
    sources={'camera_relocalization-standard':'camera_relocalization-standard',
        'camera_relocalization-obstructed':'camera_relocalization-obstructed-hamper_1',
        'mental_rotation':'mental_rotation-carving_knife','object_scaling':'object_scaling-apple'}
    definitions,jobs,templates,floating={},{},[],[];a.out.mkdir(parents=True)
    for task,source_class in sources.items():
        record=records[source_class];entry=copy.deepcopy(originals[record['row']['alias']]);iddl=parse_file(str(ROOT/'ispatialgym/assets/iddl'/entry['iddl']))
        row=measured.get(task);alias=task if row is None else row['class_id'];class_id=alias;role=None
        if row:
            cat,model=row['category'],row['model'];key=cat+'/'+model;role='obstacles' if task.endswith('obstructed') else 'target'
            if cat not in reserved['category_holdout_categories'] or key not in reserved['category_holdout_models'] or review['objects'][key]['status']!='passed':
                raise ValueError('unreserved or unreviewed category model')
            spec=entry['roles'][role];spec.update(category=cat,representative=model,dataset='behavior-1k-assets',scale={key:row['default_scale']})
            spec.pop('grasp_keys',None);spec.pop('orientation_equivalences',None)
            if role=='obstacles':entry['generation']['groups']['obstacles']={'size':1,'draw':'unique','members':{cat:1}}
            else:
                source_key=record['metadata']['choices']['target']['grasp_key']
                base=floating_job_of(source_key,str(ROOT/'runlogs/grasp_library_20260905'),held_tasks=[task])
                if base is None:raise ValueError('source floating mode unavailable')
                gkey=entry_id(cat,model,row['default_scale'],'floating');spec['grasp_keys']={key:gkey}
                floating.append({'id':gkey,'legacy_key':gkey,'category':cat,'model':model,'scale':row['default_scale'],'dataset':'behavior-1k-assets',
                    'constraint':'floating','kind':'floating','rest_hint_wxyz':None,'rest_pose':None,'floating_mode':base['floating_mode'],
                    'held_tasks':[task],'source':'campaign_category_holdout','ready':True,'readiness_source':row['readiness_source'],
                    'asset_sha256':row['asset_sha256'],'legacy_key_note':'Unique new alias; fresh candidates and physics only.'})
        entry['scenes']={scene:{'rooms':[room],'bindings':{'floor':{'room':room}},'whitelist':{} if row is None else {role:{cat:[model]}},'blacklist':{}}}
        entry['constraints']=[]
        for k in list(entry):
            if k.startswith('fallback') or k=='superseded_by':entry.pop(k)
        for k in list(entry['generation']):
            if k.startswith('_') or k=='remove_objects':entry['generation'].pop(k)
        entry['generation'].update(templates_max=1,instances_per_template=1)
        entry['roles']={r:entry['roles'][r] for r in iddl.roles if r in entry['roles']}|{r:s for r,s in entry['roles'].items() if r not in iddl.roles}
        did=24200+len(definitions);entry['definition_id']=did;groundings={}
        for name in iddl.roles:
            group=entry['generation'].get('groups',{}).get(name)
            if group:groundings[name]=RoleGrounding(name,'group',members=[(c,synset_of_category(c)) for c,n in group['members'].items() for _ in range(n)])
            else:
                c=entry['roles'][name]['category'];groundings[name]=RoleGrounding(name,'sampleable',category=c,synset=synset_of_category(c))
        activity=SUBTASK_FAMILY[task][0];rendered=render(iddl,groundings,did,alias,activity);text=rendered.text
        if task.endswith('obstructed'):text=with_stationary_goals(text,rendered.instances['obstacles.members'])
        bddl=a.out/'bddl'/activity/f'problem{did}.bddl';bddl.parent.mkdir(parents=True,exist_ok=True);bddl.write_text(text)
        refs=[ObjectRef(instance='floor.n.01_1',role='support',category='floors',model=floor['model'],asset_sha256=floor_hash,
            default_scale=tuple(floor['scale']),semantic_kind='floor',ready=True,readiness_source=status['readiness_source'])]
        if row:
            names=rendered.instances.get(role+'.members') or [rendered.instances[role]]
            for name in names:refs.append(ObjectRef(instance=name,role='target',category=cat,model=model,asset_sha256=row['asset_sha256'],
                default_scale=tuple(row['default_scale']),ready=True,readiness_source=row['readiness_source']))
        entry['campaign_provenance']={'source_class':source_class,'reservation_sha256':sha(respath),'native_scene_sha256':sha(scene_path),
            'defaults_sha256':sha(a.defaults),'visual_review_sha256':sha(a.visual_review),'status':'candidate_only'}
        definitions[alias]=entry;tid='held_category__'+class_id
        template=Template(template_id=tid,task=task,class_id=class_id,split='eval_category',scene=scene,room=room,objects=tuple(refs),
            definition_sha256=digest({'entry':entry,'bddl_sha256':sha(bddl)}),scene_setup={'task_rooms':[room],'inst_to_name':{'floor.n.01_1':floor['name']},
                'runtime_bddl_sha256':sha(bddl),'customs_alias':alias,'release_status':'candidate_only','fresh_metadata_binding_check':'required',
                'fresh_demo':'pending','fresh_replay':'pending','perturbations':False})
        templates.append({**asdict(template),'fingerprint':template.fingerprint})
        jobs[task]={'candidate_id':tid,'class_id':class_id,'task':task,'scene':scene,'room':room,'alias':alias,'split':'eval_category',
            'production_released':False,'template_fingerprint':template.fingerprint,'status':'awaiting_fresh_floating_library' if task in {'object_scaling','mental_rotation'} else 'ready_for_fresh_geometry',
            'environment':{'ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL':'1'} if task.endswith('obstructed') else {},
            'sample_argv':['python','-m','ispatialgym.sampling.sample_isg','--variant',alias,'--customs','${MATERIALIZED}/candidate_customs.json',
                '--bddl-root','${MATERIALIZED}/bddl','--scene',scene,'--room',room,'--templates','1','--instances-per-template','1','--max-attempts','3',
                '--seed','20261110','--no-write-back','--out','${CAMPAIGN_OUT}/validation/'+tid]}
    output_custom=a.out/'candidate_customs.json';output_custom.write_text(json.dumps(definitions,indent=2)+'\n')
    for job in jobs.values():
        gp=GroundedProblem.load(job['alias'],str(output_custom),str(a.out/'bddl'));gp.override_scene_room(scene,room)
        if gp.rooms(scene)!=[room]:raise ValueError('native heldout room mismatch')
    for filename,data in [('validation_jobs.json',{'jobs':list(jobs.values())}),('contract_candidates.json',{'templates':templates}),('floating_jobs.json',floating)]:
        (a.out/filename).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'category_candidates':len(jobs),'fresh_floating_jobs':len(floating),'production_released':0}))


if __name__=='__main__':main()
