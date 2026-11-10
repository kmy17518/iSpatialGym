#!/usr/bin/env python3
"""Freeze the ready Kohs model hold with its reserved board and native support."""
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
sys.path[:0]=[str(ROOT),str(B1K/'bddl3')]
from contracts import ObjectRef,Template,digest
from readiness_catalog import ReadinessCatalog
from ispatialgym.sampling.v2.problem import GroundedProblem


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--readiness',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new private output required')
    inv=ROOT/'tmp/campaign_20261110/inventory';respath=inv/'split_reservations.json';reservations=json.loads(respath.read_text())
    hashes=json.loads((ROOT/'tmp/campaign_20261110/remote_evidence/manifests/inventory_revision3/reservation_asset_hashes_r3.json').read_text())
    if hashes['reservation_sha256']!=sha(respath):raise ValueError('reservation hash mismatch')
    generated=json.loads(a.readiness.read_text())['objects']
    standard=next(t for t in json.loads((a.source/'contract_candidates.json').read_text())['templates'] if t['template_id']=='generated_train__isgk01')
    source_custom=a.source/'isgk01/candidate_customs.json';d=json.loads(source_custom.read_text());alias=next(iter(d));entry=copy.deepcopy(d[alias])
    block=generated['toy_dice/isgk10'];board=generated['tray/isgbdm']
    for asset in (block,board):
        if not asset['ready'] or asset['split_reservation']!='eval_model':raise ValueError('asset not ready for model hold')
    scene,room='Rs_int','living_room_0';support='breakfast_table_skczfi_0'
    scene_path=B1K/'datasets/2026-challenge-task-instances/scenes'/scene/'json'/f'{scene}_stable.json'
    init=json.loads(scene_path.read_text())['objects_info']['init_info'];support_init=init[support]['args']
    floor_init=next(v['args'] for v in init.values() if v['args'].get('category')=='floors' and room in v['args'].get('in_rooms',[]))
    ready=ReadinessCatalog(B1K/'bddl3/bddl/generated_data')
    refs=[];inst_to_name={}
    for obj in standard['objects']:
        data=copy.deepcopy(obj)
        if data['role']=='target':asset=block
        elif data['semantic_kind']=='tray':asset=board
        else:asset=None
        if asset:
            data.update(category=asset['category'],model=asset['model'],default_scale=asset['default_scale'],asset_sha256=asset['asset_sha256'],ready=True,readiness_source=asset['readiness_source'])
        else:
            actual=floor_init if data['semantic_kind']=='floor' else support_init
            status=ready.status(actual['category'],actual['model'])
            if not status['ready']:raise ValueError('unready native support/floor')
            usd=B1K/'datasets/behavior-1k-assets/objects'/actual['category']/actual['model']/'usd'/(actual['model']+'.encrypted.usd')
            data.update(category=actual['category'],model=actual['model'],default_scale=actual['scale'],asset_sha256=sha(usd),ready=True,readiness_source=status['readiness_source'])
            inst_to_name[data['instance']]=actual['name']
        key=data['category']+'/'+data['model']
        if data['semantic_kind']!='floor' and (key not in hashes['ready_assets'] or data['asset_sha256']!=hashes['ready_assets'][key]['asset_sha256']):
            raise ValueError('nonreserved or mismatched model hold: '+key)
        refs.append(ObjectRef(**{**data,'default_scale':tuple(data['default_scale'])}))
    for role,asset in [('targets',block),('board',board)]:
        entry['roles'][role].update(representative=asset['model'],dataset=asset['dataset'],scale={asset['category']+'/'+asset['model']:asset['default_scale']})
    entry['scenes']={scene:{'rooms':[room],'bindings':{'floor':{'room':room},'support':[support],
        'support_surfaces':[{'name':support,'init':'ontop','top_z':.7588,'row':'campaign_block_holdout'}]},
        'whitelist':{'targets':{'toy_dice':['isgk10']},'board':{'tray':['isgbdm']}},'blacklist':{}}}
    entry['generation'].pop('remove_objects',None)
    faces_path=ROOT/'tmp/campaign_20261110/assets/block_variants/isgk10/model/isgk10/build_manifest.json'
    faces=json.loads(faces_path.read_text())['faces'];required={x for pattern in entry['generation']['patterns'].values() for x in pattern}
    if not required.issubset(faces.values()):raise ValueError('heldout pattern colors absent')
    entry['definition_id']=24000
    source_bddl=next((a.source/'isgk01/bddl/configuration_matching').glob('problem*.bddl'))
    text=re.sub(r'\(problem configuration_matching-\d+\)','(problem configuration_matching-24000)',source_bddl.read_text(),count=1)
    a.out.mkdir(parents=True);bddl=a.out/'bddl/configuration_matching/problem24000.bddl';bddl.parent.mkdir(parents=True);bddl.write_text(text)
    entry['campaign_provenance']={'reservation_sha256':sha(respath),'source_customs_sha256':sha(source_custom),
        'native_scene_path':str(scene_path),'native_scene_sha256':sha(scene_path),'generated_readiness_sha256':sha(a.readiness),
        'model_faces':faces,'color_to_face':{v:k for k,v in faces.items()},'status':'model_holdout_candidate_not_released'}
    customs=a.out/'candidate_customs.json';customs.write_text(json.dumps({alias:entry},indent=2)+'\n')
    gp=GroundedProblem.load(alias,str(customs),str(a.out/'bddl'))
    if gp.rooms(scene)!=[room]:raise ValueError('native room mismatch')
    template=Template(template_id='held_model__block_isgk10_isgbdm_skczfi',task=standard['task'],class_id=standard['class_id'],split='eval_model',scene=scene,room=room,objects=tuple(refs),
        definition_sha256=digest({'entry':entry,'bddl_sha256':sha(bddl)}),scene_setup={'task_rooms':[room],'inst_to_name':inst_to_name,
            'runtime_bddl_sha256':sha(bddl),'customs_alias':alias,'model_faces':faces,'release_status':'candidate_only',
            'fresh_metadata_binding_check':'required','fresh_demo':'pending','fresh_replay':'pending','perturbations':False})
    job={'candidate_id':template.template_id,'class_id':template.class_id,'task':template.task,'alias':alias,'scene':scene,'room':room,'split':'eval_model',
        'production_released':False,'status':'requires_generic_map_and_fresh_geometry','measured_support_height_key':'0.7588','template_fingerprint':template.fingerprint,
        'sample_argv':['python','-m','ispatialgym.sampling.sample_isg','--variant',alias,'--customs','${MATERIALIZED}/candidate_customs.json',
            '--bddl-root','${MATERIALIZED}/bddl','--scene',scene,'--templates','1','--instances-per-template','1','--max-attempts','3','--seed','20261110',
            '--no-write-back','--out','${CAMPAIGN_OUT}/validation/'+template.template_id]}
    (a.out/'contract_candidates.json').write_text(json.dumps({'templates':[{**asdict(template),'fingerprint':template.fingerprint}]},indent=2,sort_keys=True)+'\n')
    (a.out/'validation_jobs.json').write_text(json.dumps({'jobs':[job]},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'model_holdout_candidates':1,'all_nonstructural_models_reserved':True,'production_released':0}))


if __name__=='__main__':main()
