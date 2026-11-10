#!/usr/bin/env python3
"""Compile the audited asymmetric isgm10 model hold after reservation extension."""
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
    p.add_argument('--reservations',type=Path,required=True)
    p.add_argument('--reserved-asset-hashes',type=Path,required=True)
    p.add_argument('--generated-readiness',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new immutable output required')
    inv=ROOT/'tmp/campaign_20261110/inventory';key='classroom_mock_up/isgm10';cls='mental_rotation-classroom_mock_up'
    policy=json.loads(a.reservations.read_text());hashes=json.loads(a.reserved_asset_hashes.read_text())
    asset=json.loads(a.generated_readiness.read_text())['objects'][key]
    if hashes['reservation_sha256']!=sha(a.reservations):raise ValueError('policy/hash mismatch')
    if key not in policy['model_holdout_models'] or key not in policy['train_exclude_models']:raise ValueError('new mental model not reserved')
    if 'classroom_mock_up/tpfmpb' not in policy['train_exclude_models']:raise ValueError('prior reservation was dropped')
    if not asset['ready'] or asset['split_reservation']!='eval_model':raise ValueError('project readiness or split assignment missing')
    if hashes['ready_assets'][key]['asset_sha256']!=asset['asset_sha256']:raise ValueError('asset hash mismatch')
    standard=next(t for t in json.loads((inv/'standard_contract_ready.json').read_text())['templates'] if t['class_id']==cls)
    record=next(r for r in json.loads((inv/'cluster_templates_with_scales.json').read_text())['records'] if r['key']==cls)
    source=inv/'frozen_customs_f24ea424.json';alias=record['row']['alias'];entry=copy.deepcopy(json.loads(source.read_text())[alias])
    scene,room=standard['scene'],standard['room'];cat,model=asset['category'],asset['model'];scale=asset['default_scale']
    if any(r['scene']==scene and r['room']==room for r in policy['category_holdout_rooms']):raise ValueError('model hold uses category-held room')
    grasp_key=cat+'__'+model+'__f'
    entry['roles']['target'].update(representative=model,dataset=asset['dataset'],scale={key:scale},grasp_keys={key:grasp_key},orientation_equivalences={key:[]})
    entry['scenes']={scene:{'rooms':[room],'bindings':{'floor':{'room':room}},'whitelist':{'target':{cat:[model]}},'blacklist':{}}}
    entry['constraints']=[]
    for k in list(entry):
        if k.startswith('fallback') or k=='superseded_by':entry.pop(k)
    for k in list(entry['generation']):
        if k.startswith('_') or k=='remove_objects':entry['generation'].pop(k)
    entry['generation'].update(templates_max=1,instances_per_template=1)
    iddl=parse_file(str(ROOT/'ispatialgym/assets/iddl'/entry['iddl']))
    entry['roles']={r:entry['roles'][r] for r in iddl.roles if r in entry['roles']}|{r:s for r,s in entry['roles'].items() if r not in iddl.roles}
    entry['definition_id']=26010
    groundings={r:RoleGrounding(r,'sampleable',category=entry['roles'][r]['category'],synset=synset_of_category(entry['roles'][r]['category'])) for r in iddl.roles}
    activity=SUBTASK_FAMILY['mental_rotation'][0];rendered=render(iddl,groundings,26010,alias,activity)
    a.out.mkdir(parents=True);bddl=a.out/'bddl'/activity/'problem26010.bddl';bddl.parent.mkdir(parents=True);bddl.write_text(rendered.text)
    entry['campaign_provenance']={'source_customs_sha256':sha(source),'reservation_sha256':sha(a.reservations),'generated_readiness_sha256':sha(a.generated_readiness),'reason':'Audited asymmetric replacement; prior tpfmpb remains reserved unused.','status':'candidate_only'}
    custom=a.out/'candidate_customs.json';custom.write_text(json.dumps({alias:entry},indent=2)+'\n')
    sp=B1K/'datasets/2026-challenge-task-instances/scenes'/scene/'json'/f'{scene}_stable.json'
    floor=next(x['args'] for x in json.loads(sp.read_text())['objects_info']['init_info'].values() if x['args'].get('category')=='floors' and room in x['args'].get('in_rooms',[]))
    status=ReadinessCatalog(B1K/'bddl3/bddl/generated_data').status('floors',floor['model'])
    if not status['ready']:raise ValueError('scene floor not ready')
    refs=[ObjectRef(instance='floor.n.01_1',role='support',category='floors',model=floor['model'],asset_sha256=sha(B1K/'datasets/behavior-1k-assets/objects/floors'/floor['model']/'usd'/(floor['model']+'.encrypted.usd')),default_scale=tuple(floor['scale']),semantic_kind='floor',ready=True,readiness_source=status['readiness_source']),
        ObjectRef(instance=rendered.instances['target'],role='target',category=cat,model=model,asset_sha256=asset['asset_sha256'],default_scale=tuple(scale),ready=True,readiness_source=asset['readiness_source'])]
    tid='held_model__'+digest({'class':cls,'scene':scene,'room':room,'objects':[asdict(o) for o in refs]})[:20]
    t=Template(template_id=tid,task='mental_rotation',class_id=cls,split='eval_model',scene=scene,room=room,objects=tuple(refs),definition_sha256=digest({'entry':entry,'bddl_sha256':sha(bddl)}),scene_setup={'task_rooms':[room],'runtime_bddl_sha256':sha(bddl),'customs_alias':alias,'reservation_sha256':sha(a.reservations),'native_scene_sha256':sha(sp),'inst_to_name':{'floor.n.01_1':floor['name']},'orientation_equivalences':[],'symmetry_evidence':'Generated occupancy has24 distinct proper-cube rotations; model physical and three-view visual reports are bound by readiness_source.','release_status':'candidate_only','fresh_metadata_binding_check':'required','fresh_demo':'pending','fresh_replay':'pending','perturbations':False})
    job={'candidate_id':tid,'class_id':cls,'task':'mental_rotation','alias':alias,'scene':scene,'room':room,'split':'eval_model','template_fingerprint':t.fingerprint,'bddl_sha256':sha(bddl),'required_new_grasp_key':grasp_key,'production_released':False,'status':'awaiting_exact_floating_library_reader_gate','sample_argv':['python','-m','ispatialgym.sampling.sample_isg','--variant',alias,'--customs','${MATERIALIZED}/candidate_customs.json','--bddl-root','${MATERIALIZED}/bddl','--scene',scene,'--room',room,'--templates','1','--instances-per-template','1','--max-attempts','3','--seed','20261110','--no-write-back','--out','${CAMPAIGN_OUT}/validation/'+cls]}
    gp=GroundedProblem.load(alias,str(custom),str(a.out/'bddl'));gp.override_scene_room(scene,room)
    if gp.rooms(scene)!=[room]:raise ValueError('native room mismatch')
    for name,data in [('validation_jobs.json',{'jobs':[job],'reservation_sha256':sha(a.reservations)}),('contract_candidates.json',{'templates':[{**asdict(t),'fingerprint':t.fingerprint}]})]:
        (a.out/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'candidate':tid,'class_id':cls,'required_grasp_key':grasp_key,'manifest_sha256':sha(a.out/'validation_jobs.json')}))


if __name__=='__main__':main()
