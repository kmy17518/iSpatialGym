#!/usr/bin/env python3
"""Compile generated train block/LEF canaries with exact native group bindings."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
B1K=Path('/home/minyeongk/Desktop/projects/Behavior-1K-grasp-bands-6a6194600')
REMOTE='/vision/u/minyeong/projects/isg_home/campaign_20261110'
sys.path[:0]=[str(ROOT),str(B1K/'bddl3')]
from contracts import ObjectRef,Template,digest
from ispatialgym.grasping.schema import entry_id
from ispatialgym.sampling.bddl_from_iddl import RoleGrounding,render
from ispatialgym.sampling.iddl import parse_file
from ispatialgym.sampling.knowledge import synset_of_category
from ispatialgym.sampling.v2.problem import GroundedProblem,SUBTASK_FAMILY


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--readiness',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new campaign output required')
    inv=ROOT/'tmp/campaign_20261110/inventory'
    reserved=json.loads((inv/'split_reservations.json').read_text())
    excluded=set(reserved['train_exclude_models'])
    generated=json.loads(a.readiness.read_text())['objects']
    standards={r['class_id']:r for r in json.loads((inv/'standard_contract_ready.json').read_text())['templates']}
    records={r['key']:r for r in json.loads((inv/'cluster_templates_with_scales.json').read_text())['records']}
    customs_path=inv/'frozen_customs_f24ea424.json';originals=json.loads(customs_path.read_text())
    lef_root=ROOT/'tmp/campaign_20261110/assets/lef_variants_v3'
    lef_index=json.loads((lef_root/'asset_builds.json').read_text())
    cases=[('block',f'isgk{i:02d}',None) for i in range(1,10)]
    for row in lef_index['models']:
        if row['role']=='target' and row['split_reservation']=='train':cases.append(('lef',row['model'],row))
    jobs,templates,grasp_jobs=[],[],[]
    a.out.mkdir(parents=True)
    for case_index,(kind,model,build) in enumerate(cases):
        cls='configuration_matching-block_design_01' if kind=='block' else 'disembedding_01'
        standard=standards[cls];record=records[cls];alias=record['row']['alias'];entry=copy.deepcopy(originals[alias])
        task=standard['task'];scene=standard['scene'];room=standard['room'];path=a.out/model;path.mkdir()
        iddl=parse_file(str(ROOT/'ispatialgym/assets/iddl'/entry['iddl']))
        role_objects={}
        for binding in record['metadata']['bindings']:
            source=next(o for o in standard['objects'] if o['instance']==binding['name'])
            role_objects.setdefault(binding['role'],[]).append(copy.deepcopy(source))
        group_role='targets' if kind=='block' else 'target'
        category='toy_dice' if kind=='block' else 'jigsaw_puzzle_piece'
        asset=generated[category+'/'+model]
        if not asset['ready'] or category+'/'+model in excluded or asset['split_reservation']!='train':
            raise ValueError('generated model is not admitted for train')
        target_ref=role_objects[group_role][0]
        target_ref.update(category=category,model=model,default_scale=asset['default_scale'],asset_sha256=asset['asset_sha256'],
                          ready=True,readiness_source=asset['readiness_source'])
        role_objects[group_role]=[copy.deepcopy(target_ref) for _ in range(4 if kind=='block' else 1)]
        entry['roles'][group_role].update(representative=model,dataset=asset['dataset'])
        face_map=None;assembly=None;new_grasp_key=None
        if kind=='block':
            build_path=ROOT/f'tmp/campaign_20261110/assets/block_variants/{model}/model/{model}/build_manifest.json'
            face_map=json.loads(build_path.read_text())['faces']
            needed={color for pattern in entry['generation']['patterns'].values() for color in pattern}
            if not needed.issubset(face_map.values()):raise ValueError('native pattern color absent from model faces')
        else:
            assembly_path=lef_root/build['assembly_file'];assembly=json.loads(assembly_path.read_text())
            if sha(assembly_path)!=build['assembly_sha256']:raise ValueError('assembly source changed')
            parts=[m for m in assembly['models'] if m!=model]
            entry['generation']['groups']['distractors']={'size':len(parts),'draw':'unique','members':{category:len(parts)}}
            entry['generation']['fixed_layout']={'file':REMOTE+'/assets_sources/lef_v1/'+build['assembly_file'],
                'sha256':build['assembly_sha256'],'target':category+'/'+model,'parts':[category+'/'+m for m in parts]}
            entry['generation'].pop('distractor_budget',None)
            original_part=role_objects['distractors'][0];role_objects['distractors']=[]
            for part in parts:
                source=generated[category+'/'+part]
                if not source['ready'] or category+'/'+part in excluded:raise ValueError('unready or reserved context')
                ref=copy.deepcopy(original_part);ref.update(category=category,model=part,asset_sha256=source['asset_sha256'],
                    default_scale=source['default_scale'],readiness_source=source['readiness_source'],ready=True)
                role_objects['distractors'].append(ref)
            entry['roles']['distractors']['dataset']=asset['dataset']
            tokens={r:entry['roles'][r]['category'] for r in iddl.variant_slots if r!='distractors'}
            tokens['distractors']=category+'_'+str(len(parts))
            alias=iddl.variant_template.format(**tokens)
            new_grasp_key=entry_id(category,model,asset['default_scale'],'tabletop')
            entry['roles']['target']['grasp_keys']={category+'/'+model:new_grasp_key}
            grasp_jobs.append({'id':new_grasp_key,'legacy_key':new_grasp_key,'category':category,'model':model,
                'scale':asset['default_scale'],'dataset':asset['dataset'],'constraint':'tabletop','rest_hint_wxyz':None,
                'rest_pose':'canonical','kind':'tabletop','source':'campaign_generated_asset','asset_sha256':asset['asset_sha256'],
                'ready':True,'readiness_source':asset['readiness_source'],'campaign_classes':[cls],
                'measured_support_height_key':'0.8126','legacy_key_note':'New routing alias; no legacy grasps imported.'})
        block=copy.deepcopy(entry['scenes'][scene]);block['rooms']=[room]
        block['bindings']=copy.deepcopy(record['metadata']['choice']['bindings']);block['whitelist']={};block['blacklist']={}
        for role,objects in role_objects.items():
            spec=entry['roles'][role]
            if not spec.get('sampleable'):continue
            pool={};scales={}
            for ref in objects:
                pool.setdefault(ref['category'],[]).append(ref['model']);scales[ref['category']+'/'+ref['model']]=ref['default_scale']
            block['whitelist'][role]={c:sorted(set(ms)) for c,ms in pool.items()};spec['scale']=scales
        entry['scenes']={scene:block};entry['constraints']=[]
        for k in list(entry):
            if k.startswith('fallback') or k=='superseded_by':entry.pop(k)
        for k in list(entry['generation']):
            if k.startswith('_'):entry['generation'].pop(k)
        entry['generation'].update(templates_max=1,instances_per_template=1)
        entry['roles']={r:entry['roles'][r] for r in iddl.roles if r in entry['roles']} | {r:s for r,s in entry['roles'].items() if r not in iddl.roles}
        # A separate customs file permits the exact native alias for every new
        # model, without adding tokens that task handlers could misinterpret.
        definition_id=23000+case_index;entry['definition_id']=definition_id;groundings={}
        for role in iddl.roles:
            spec=entry['roles'][role];group=entry['generation'].get('groups',{}).get(role)
            if role=='floor':groundings[role]=RoleGrounding(role,'scene',same_as='floor')
            elif group:
                groundings[role]=RoleGrounding(role,'group',members=[(c,synset_of_category(c)) for c,n in group['members'].items() for _ in range(n)])
            else:
                c=spec['category'];groundings[role]=RoleGrounding(role,'sampleable' if spec.get('sampleable') else 'scene',category=c,synset=synset_of_category(c))
        activity=SUBTASK_FAMILY[task][0];rendered=render(iddl,groundings,definition_id,alias,activity)
        definition_text=rendered.text;source_definition=None
        if kind=='block':
            # The board variant is hand-grounded: its goal is on the board,
            # whereas the family IDDL alone targets the tabletop. Preserve the
            # exact existing board-specific init/goal/measure and object names.
            source_definition=ROOT/'ispatialgym/assets/bddl/activity_definitions/configuration_matching/problem83.bddl'
            definition_text=re.sub(r'\(problem configuration_matching-83\)',f'(problem configuration_matching-{definition_id})',source_definition.read_text(),count=1)
            rendered.instances['board']=next(b['name'] for b in record['metadata']['bindings'] if b['role']=='board')
        bddl=path/'bddl'/activity/f'problem{definition_id}.bddl';bddl.parent.mkdir(parents=True);bddl.write_text(definition_text)
        entry['campaign_provenance']={'source_class':cls,'source_customs_sha256':sha(customs_path),'readiness_sha256':sha(a.readiness),
            'reservation_sha256':sha(inv/'split_reservations.json'),'model_faces':face_map,'color_to_face':None if face_map is None else {v:k for k,v in face_map.items()},
            'source_definition_sha256':None if source_definition is None else sha(source_definition),
            'fresh_geometry_demo_replay_required':True}
        custom=path/'candidate_customs.json';custom.write_text(json.dumps({alias:entry},indent=2)+'\n')
        gp=GroundedProblem.load(alias,str(custom),str(path/'bddl'))
        if gp.rooms(scene)!=[room]:raise ValueError('native task room differs')
        refs=[]
        for role,objects in role_objects.items():
            names=rendered.instances.get(role+'.members') or [rendered.instances[role]]
            if len(names)!=len(objects):raise ValueError('native group count mismatch: '+role)
            for name,obj in zip(names,objects):refs.append(ObjectRef(**{**obj,'instance':name,'default_scale':tuple(obj['default_scale'])}))
        tid='generated_train__'+model
        template=Template(template_id=tid,task=task,class_id=cls,split='train',scene=scene,room=room,objects=tuple(refs),
            definition_sha256=digest({'entry':entry,'bddl_sha256':sha(bddl)}),scene_setup={'task_rooms':[room],
                'runtime_bddl_sha256':sha(bddl),'customs_alias':alias,'model_faces':face_map,'release_status':'candidate_only',
                'fresh_metadata_binding_check':'required','fresh_demo':'pending','fresh_replay':'pending'})
        templates.append({**asdict(template),'fingerprint':template.fingerprint})
        jobs.append({'candidate_id':tid,'class_id':cls,'model':model,'task':task,'scene':scene,'room':room,'alias':alias,
            'relative_materialized':model,'production_released':False,'template_fingerprint':template.fingerprint,
            'status':'awaiting_fresh_tabletop_grasps_and_maps' if new_grasp_key else 'ready_for_fresh_geometry_validation',
            'required_new_grasp_key':new_grasp_key,'measured_support_height_key':'0.8126','bddl_sha256':sha(bddl),
            'sample_argv':['python','-m','ispatialgym.sampling.sample_isg','--variant',alias,'--customs','${MATERIALIZED}/'+model+'/candidate_customs.json',
                '--bddl-root','${MATERIALIZED}/'+model+'/bddl','--scene',scene,'--templates','1','--instances-per-template','1',
                '--max-attempts','3','--seed','20261110','--no-write-back','--out','${CAMPAIGN_OUT}/validation/'+tid]})
    for name,data in [('validation_jobs.json',{'jobs':jobs}),('contract_candidates.json',{'templates':templates}),('lef_tabletop_jobs.json',grasp_jobs)]:
        (a.out/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'generated_train_candidates':len(jobs),'block_canaries':9,'lef_grasp_jobs':len(grasp_jobs),'production_released':0}))


if __name__=='__main__':main()
