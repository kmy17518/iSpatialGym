#!/usr/bin/env python3
"""Freeze the tenth embedded figure with globally reserved trays and support."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
B1K=Path('/home/minyeongk/Desktop/projects/Behavior-1K-grasp-bands-6a6194600')
REMOTE='/vision/u/minyeong/projects/isg_home/campaign_20261110'
sys.path[:0]=[str(ROOT),str(B1K/'bddl3')]
from contracts import ObjectRef,Template,digest
from readiness_catalog import ReadinessCatalog
from ispatialgym.grasping.schema import entry_id
from ispatialgym.sampling.bddl_from_iddl import RoleGrounding,render
from ispatialgym.sampling.iddl import parse_file
from ispatialgym.sampling.knowledge import synset_of_category
from ispatialgym.sampling.v2.problem import GroundedProblem,SUBTASK_FAMILY


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('defaults','visual-review','readiness','reservations','reserved-asset-hashes','out'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new immutable output required')
    inv=ROOT/'tmp/campaign_20261110/inventory';policy=json.loads(a.reservations.read_text());hashes=json.loads(a.reserved_asset_hashes.read_text())
    defaults=json.loads(a.defaults.read_text());review=json.loads(a.visual_review.read_text());generated=json.loads(a.readiness.read_text())['objects']
    if hashes['reservation_sha256']!=sha(a.reservations) or defaults['reservation_sha256']!=sha(a.reservations) or defaults['status']!='native_defaults_and_geometry_recorded':raise ValueError('reservation or measured defaults mismatch')
    if review['default_geometry_sha256']!=sha(a.defaults):raise ValueError('review source mismatch')
    measured={r['category']+'/'+r['model']:r for r in defaults['objects']}
    cls='disembedding_01';record=next(r for r in json.loads((inv/'cluster_templates_with_scales.json').read_text())['records'] if r['key']==cls)
    original=inv/'frozen_customs_f24ea424.json';entry=copy.deepcopy(json.loads(original.read_text())[record['row']['alias']])
    iddl=parse_file(str(ROOT/'ispatialgym/assets/iddl'/entry['iddl']));cat='jigsaw_puzzle_piece';model='ief09t'
    buildroot=ROOT/'tmp/campaign_20261110/assets/lef_variants_v3';build=next(r for r in json.loads((buildroot/'asset_builds.json').read_text())['models'] if r['model']==model)
    assemblypath=buildroot/build['assembly_file'];assembly=json.loads(assemblypath.read_text());parts=[m for m in assembly['models'] if m!=model]
    if sha(assemblypath)!=build['assembly_sha256']:raise ValueError('assembly hash mismatch')
    scene,room,support='Rs_int','living_room_0','breakfast_table_skczfi_0'
    scenepath=B1K/'datasets/2026-challenge-task-instances/scenes'/scene/'json'/f'{scene}_stable.json'
    init=json.loads(scenepath.read_text())['objects_info']['init_info'];floor=next(v['args'] for v in init.values() if v['args'].get('category')=='floors' and room in v['args'].get('in_rooms',[]));table=init[support]['args']
    ready=ReadinessCatalog(B1K/'bddl3/bddl/generated_data');role_rows={}
    for role,actual in [('floor',floor),('support',table)]:
        st=ready.status(actual['category'],actual['model'])
        if not st['ready']:raise ValueError('unready native scene support')
        role_rows[role]=[{'category':actual['category'],'model':actual['model'],'default_scale':actual['scale'],'asset_sha256':sha(B1K/'datasets/behavior-1k-assets/objects'/actual['category']/actual['model']/'usd'/(actual['model']+'.encrypted.usd')),'ready':True,'readiness_source':st['readiness_source'],'scene_object_name':actual['name']}]
    role_rows['target']=[generated[cat+'/'+model]];role_rows['distractors']=[generated[cat+'/'+m] for m in parts]
    for role,m in [('source','gsxbym'),('destination','mhhoga')]:
        key='tray/'+m
        if review['objects'][key]['status']!='passed':raise ValueError('tray visual review missing')
        role_rows[role]=[measured[key]]
    for role,rows in role_rows.items():
        for r in rows:
            if not r['ready']:raise ValueError('unready heldout role')
            key=r['category']+'/'+r['model']
            if role!='floor' and (key not in policy['train_exclude_models'] or hashes['ready_assets'][key]['asset_sha256']!=r['asset_sha256']):raise ValueError('heldout role not globally reserved: '+key)
    for role in ('target','distractors','source','destination'):
        rows=role_rows[role];entry['roles'][role].update(representative=rows[0]['model'],dataset=rows[0].get('dataset','behavior-1k-assets'),scale={r['category']+'/'+r['model']:r['default_scale'] for r in rows})
    gkey=entry_id(cat,model,generated[cat+'/'+model]['default_scale'],'tabletop');entry['roles']['target']['grasp_keys']={cat+'/'+model:gkey}
    entry['generation']['groups']['distractors']={'size':len(parts),'draw':'unique','members':{cat:len(parts)}}
    entry['generation']['fixed_layout']={'file':REMOTE+'/assets_sources/lef_v1/'+build['assembly_file'],'sha256':build['assembly_sha256'],'target':cat+'/'+model,'parts':[cat+'/'+m for m in parts]}
    entry['generation'].pop('distractor_budget',None);entry['generation'].pop('remove_objects',None)
    entry['generation'].update(templates_max=1,instances_per_template=1)
    for k in list(entry['generation']):
        if k.startswith('_'):entry['generation'].pop(k)
    for k in list(entry):
        if k.startswith('fallback') or k=='superseded_by':entry.pop(k)
    pool={role:{r[0]['category']:[x['model'] for x in r]} for role,r in role_rows.items() if role not in {'floor','support'}}
    entry['scenes']={scene:{'rooms':[room],'bindings':{'floor':{'room':room},'support':[support],'support_surfaces':[{'name':support,'init':'ontop','top_z':.7588,'row':'campaign_lef_holdout'}]},'whitelist':pool,'blacklist':{}}};entry['constraints']=[]
    entry['roles']={r:entry['roles'][r] for r in iddl.roles if r in entry['roles']}|{r:s for r,s in entry['roles'].items() if r not in iddl.roles}
    alias=iddl.variant_template.format(target=cat,distractors=cat+'_'+str(len(parts)),support='breakfast_table');entry['definition_id']=26020
    groundings={}
    for role in iddl.roles:
        if role=='floor':groundings[role]=RoleGrounding(role,'scene',same_as='floor')
        elif role=='distractors':groundings[role]=RoleGrounding(role,'group',members=[(cat,synset_of_category(cat)) for _ in parts])
        else:
            c=role_rows[role][0]['category'];groundings[role]=RoleGrounding(role,'sampleable' if entry['roles'][role].get('sampleable') else 'scene',category=c,synset=synset_of_category(c))
    rendered=render(iddl,groundings,26020,alias,SUBTASK_FAMILY['disembedding'][0]);a.out.mkdir(parents=True);bddl=a.out/'bddl/disembedding/problem26020.bddl';bddl.parent.mkdir(parents=True);bddl.write_text(rendered.text)
    entry['campaign_provenance']={'reservation_sha256':sha(a.reservations),'readiness_sha256':sha(a.readiness),'tray_defaults_sha256':sha(a.defaults),'tray_visual_review_sha256':sha(a.visual_review),'native_scene_sha256':sha(scenepath),'source_customs_sha256':sha(original),'assembly_sha256':sha(assemblypath),'status':'candidate_only'}
    custom=a.out/'candidate_customs.json';custom.write_text(json.dumps({alias:entry},indent=2)+'\n')
    refs=[];names={}
    for role,rows in role_rows.items():
        instances=rendered.instances.get(role+'.members') or [rendered.instances[role]]
        if len(instances)!=len(rows):raise ValueError('native role count mismatch')
        for instance,r in zip(instances,rows):
            kind='floor' if role=='floor' else ('tray' if role in {'source','destination'} else 'object')
            refs.append(ObjectRef(instance=instance,role='target' if role=='target' else ('distractor' if role=='distractors' else 'support'),category=r['category'],model=r['model'],asset_sha256=r['asset_sha256'],default_scale=tuple(r['default_scale']),semantic_kind=kind,ready=True,readiness_source=r['readiness_source']))
            if r.get('scene_object_name'):names[instance]=r['scene_object_name']
    template=Template(template_id='held_model__lef_ief09_reserved_trays',task='disembedding',class_id=cls,split='eval_model',scene=scene,room=room,objects=tuple(refs),definition_sha256=digest({'entry':entry,'bddl_sha256':sha(bddl)}),scene_setup={'task_rooms':[room],'inst_to_name':names,'runtime_bddl_sha256':sha(bddl),'customs_alias':alias,'reservation_sha256':sha(a.reservations),'fixed_layout_sha256':sha(assemblypath),'release_status':'candidate_only','fresh_metadata_binding_check':'required','fresh_demo':'pending','fresh_replay':'pending','perturbations':False})
    gp=GroundedProblem.load(alias,str(custom),str(a.out/'bddl'))
    if gp.rooms(scene)!=[room]:raise ValueError('native room mismatch')
    job={'candidate_id':template.template_id,'class_id':cls,'task':'disembedding','alias':alias,'scene':scene,'room':room,'split':'eval_model','template_fingerprint':template.fingerprint,'required_new_grasp_key':gkey,'measured_table_height_key':'0.7588','required_tray_floor_height_key':'must raycast actual source tray in template','production_released':False,'sample_argv':['python','-m','ispatialgym.sampling.sample_isg','--variant',alias,'--customs','${MATERIALIZED}/candidate_customs.json','--bddl-root','${MATERIALIZED}/bddl','--scene',scene,'--templates','1','--instances-per-template','1','--max-attempts','3','--seed','20261110','--no-write-back','--out','${CAMPAIGN_OUT}/validation/'+template.template_id]}
    grasp={'id':gkey,'legacy_key':gkey,'category':cat,'model':model,'scale':generated[cat+'/'+model]['default_scale'],'dataset':generated[cat+'/'+model]['dataset'],'constraint':'tabletop','rest_hint_wxyz':None,'rest_pose':'canonical','kind':'tabletop','source':'campaign_generated_model_holdout','asset_sha256':generated[cat+'/'+model]['asset_sha256'],'ready':True,'readiness_source':generated[cat+'/'+model]['readiness_source'],'campaign_classes':[cls]}
    for name,data in [('validation_jobs.json',{'jobs':[job]}),('contract_candidates.json',{'templates':[{**asdict(template),'fingerprint':template.fingerprint}]}),('tabletop_jobs.json',[grasp])]:
        (a.out/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'candidate':template.template_id,'manifest_sha256':sha(a.out/'validation_jobs.json')}))


if __name__=='__main__':main()
