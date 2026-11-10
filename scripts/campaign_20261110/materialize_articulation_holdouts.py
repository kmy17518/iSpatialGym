#!/usr/bin/env python3
"""Freeze native measured articulated model holds independently of blocked models."""
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
from ispatialgym.sampling.bddl_from_iddl import RoleGrounding,render
from ispatialgym.sampling.iddl import parse_file
from ispatialgym.sampling.knowledge import synset_of_category
from ispatialgym.sampling.v2.problem import GroundedProblem,SUBTASK_FAMILY


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('defaults','visual-review','reservations','reserved-asset-hashes','out'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new immutable output required')
    inv=ROOT/'tmp/campaign_20261110/inventory';policy=json.loads(a.reservations.read_text());hashes=json.loads(a.reserved_asset_hashes.read_text());doc=json.loads(a.defaults.read_text());review=json.loads(a.visual_review.read_text())
    if hashes['reservation_sha256']!=sha(a.reservations) or doc['reservation_sha256']!=sha(a.reservations) or doc['status']!='native_defaults_and_geometry_recorded':raise ValueError('measured defaults/policy mismatch')
    if review['default_geometry_sha256']!=sha(a.defaults):raise ValueError('visual evidence mismatch')
    measured={r['category']+'/'+r['model']:r for r in doc['objects']};records={r['key']:r for r in json.loads((inv/'cluster_templates_with_scales.json').read_text())['records']};source=inv/'frozen_customs_f24ea424.json';originals=json.loads(source.read_text())
    readiness=ReadinessCatalog(B1K/'bddl3/bddl/generated_data');definitions={};jobs=[];templates=[];requests=[];blocked=[];a.out.mkdir(parents=True)
    selected_joint={'door':'j_left_leaf','dishwasher':'j_link_1','bottom_cabinet':'j_link_2','blender':'j_lid'}
    for proposal in policy['model_holdout_templates']:
        if 'articulation' not in proposal['task']:continue
        cls,task=proposal['class_id'],proposal['task'];chosen=next(x for x in proposal['roles'] if x['role']=='target');key=chosen['category']+'/'+chosen['model'];row=measured.get(key)
        if row is None or review['objects'][key]['status']!='passed':blocked.append({'class_id':cls,'model_key':key,'reason':'model default geometry/review blocked'});continue
        category,model=row['category'],row['model'];joint=selected_joint[category]
        if joint not in row['joints'] or row['joints'][joint]['lower_limit']>=row['joints'][joint]['upper_limit']:raise ValueError('selected native joint invalid')
        large='large_scale' in task;entry=copy.deepcopy(originals[records[cls]['row']['alias']]);iddl=parse_file(str(ROOT/'ispatialgym/assets/iddl'/entry['iddl']))
        scene,room=(row['scene'],row['room']) if large else ('Rs_int','living_room_0')
        sp=B1K/'datasets/2026-challenge-task-instances/scenes'/scene/'json'/f'{scene}_stable.json';state=json.loads(sp.read_text());init=state['objects_info']['init_info'];floor=next(x['args'] for x in init.values() if x['args'].get('category')=='floors' and room in x['args'].get('in_rooms',[]))
        name=row['scene_default_scale_source']['object'] if large else None;role_key=scene+'/'+name if large else key
        spec=entry['roles']['target'];spec.update(joint={role_key:joint},handle_key={role_key:category+'__'+model});spec.pop('handle_row',None)
        if not large:spec.update(representative=model,dataset='behavior-1k-assets',scale={key:row['default_scale']})
        bindings={'floor':{'room':room}};whitelist={}
        if large:bindings['target']=[name]
        else:
            support='breakfast_table_skczfi_0';bindings.update(support=[support],support_surfaces=[{'name':support,'init':'ontop','top_z':.7588,'row':'campaign_articulation_holdout'}]);whitelist={'target':{category:[model]}}
        entry['scenes']={scene:{'rooms':[room],'bindings':bindings,'whitelist':whitelist,'blacklist':{}}};entry['constraints']=[]
        for k in list(entry):
            if k.startswith('fallback') or k=='superseded_by':entry.pop(k)
        for k in list(entry['generation']):
            if k.startswith('_') or k=='remove_objects':entry['generation'].pop(k)
        entry['generation'].update(templates_max=1,instances_per_template=1)
        entry['roles']={r:entry['roles'][r] for r in iddl.roles if r in entry['roles']}|{r:s for r,s in entry['roles'].items() if r not in iddl.roles}
        alias=records[cls]['row']['alias'];did=27000+len(definitions);entry['definition_id']=did;groundings={}
        for role in iddl.roles:
            if role=='floor':groundings[role]=RoleGrounding(role,'scene',same_as='floor')
            else:
                c=entry['roles'][role]['category'];groundings[role]=RoleGrounding(role,'sampleable' if entry['roles'][role].get('sampleable') else 'scene',category=c,synset=synset_of_category(c))
        activity=SUBTASK_FAMILY[task][0];rendered=render(iddl,groundings,did,alias,activity);bddl=a.out/'bddl'/activity/f'problem{did}.bddl';bddl.parent.mkdir(parents=True,exist_ok=True);bddl.write_text(rendered.text)
        refs=[];names={}
        for role,obj in [('floor',floor)]+([] if large else [('support',init[support]['args'])]):
            st=readiness.status(obj['category'],obj['model']);usd=B1K/'datasets/behavior-1k-assets/objects'/obj['category']/obj['model']/'usd'/(obj['model']+'.encrypted.usd')
            if not st['ready']:raise ValueError('unready native scene role')
            refs.append(ObjectRef(instance=rendered.instances[role],role='support',category=obj['category'],model=obj['model'],asset_sha256=sha(usd),default_scale=tuple(obj['scale']),semantic_kind='floor' if role=='floor' else 'object',ready=True,readiness_source=st['readiness_source']));names[rendered.instances[role]]=obj['name']
        refs.append(ObjectRef(instance=rendered.instances['target'],role='target',category=category,model=model,asset_sha256=row['asset_sha256'],default_scale=tuple(row['default_scale']),joint=joint,ready=row['ready'],readiness_source=row['readiness_source']))
        if large:names[rendered.instances['target']]=name
        for obj in refs:
            if obj.semantic_kind=='floor':continue
            k=obj.category+'/'+obj.model
            if k not in policy['train_exclude_models'] or hashes['ready_assets'][k]['asset_sha256']!=obj.asset_sha256:raise ValueError('nonreserved articulated role')
        entry['campaign_provenance']={'reservation_sha256':sha(a.reservations),'native_defaults_sha256':sha(a.defaults),'visual_review_sha256':sha(a.visual_review),'source_customs_sha256':sha(source),'native_scene_sha256':sha(sp),'status':'candidate_only'};definitions[alias]=entry
        template=Template(template_id='held_model__'+cls,task=task,class_id=cls,split='eval_model',scene=scene,room=room,objects=tuple(refs),definition_sha256=digest({'entry':entry,'bddl_sha256':sha(bddl)}),scene_setup={'task_rooms':[room],'inst_to_name':names,'runtime_bddl_sha256':sha(bddl),'customs_alias':alias,'native_scene_sha256':sha(sp),'native_joint':row['joints'][joint],'release_status':'candidate_only','fresh_metadata_binding_check':'required','fresh_demo':'pending','fresh_replay':'pending','perturbations':False})
        templates.append({**asdict(template),'fingerprint':template.fingerprint})
        jobs.append({'candidate_id':template.template_id,'class_id':cls,'task':task,'alias':alias,'scene':scene,'room':room,'split':'eval_model','template_fingerprint':template.fingerprint,'production_released':False,'status':'requires_exact_native_handle_motion_and_prefix_maps','sample_argv':['python','-m','ispatialgym.sampling.sample_isg','--variant',alias,'--customs','${MATERIALIZED}/candidate_customs.json','--bddl-root','${MATERIALIZED}/bddl','--scene',scene,'--templates','1','--instances-per-template','1','--max-attempts','3','--seed','20261110','--no-write-back','--out','${CAMPAIGN_OUT}/validation/'+cls]})
        requests.append({'class_id':cls,'category':category,'model':model,'scale':row['default_scale'],'asset_sha256':row['asset_sha256'],'joint':joint,'native_joint':row['joints'][joint],'handle_key':category+'__'+model,'scene':scene,'room':room,'scene_object':name,'scene_root_state':state['state']['registry']['object_registry'][name]['root_link'] if large else None,'support_height_key':None if large else '0.7588','operation':'pull_open' if '_open_' in task else 'push_close','direction':'open' if '_open_' in task else 'close','generation':entry['generation'],'status':'dependency_request_not_validated'})
    custom=a.out/'candidate_customs.json';custom.write_text(json.dumps(definitions,indent=2)+'\n')
    for job in jobs:
        gp=GroundedProblem.load(job['alias'],str(custom),str(a.out/'bddl'))
        if gp.rooms(job['scene'])!=[job['room']]:raise ValueError('native room mismatch')
    for filename,data in [('validation_jobs.json',{'jobs':jobs,'blocked':blocked}),('contract_candidates.json',{'templates':templates}),('articulation_dependency_requests.json',requests)]:
        (a.out/filename).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'model_hold_candidates':len(jobs),'blocked':blocked,'manifest_sha256':sha(a.out/'validation_jobs.json')}))


if __name__=='__main__':main()
