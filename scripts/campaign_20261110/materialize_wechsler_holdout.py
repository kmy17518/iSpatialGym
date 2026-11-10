#!/usr/bin/env python3
"""Compile the one-instance Wechsler category hold on a measured private table."""
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
from ispatialgym.sampling.knowledge import synset_of_category
from ispatialgym.sampling.v2.problem import GroundedProblem


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('support-probe','support-scene','support-visual-review','readiness','reservations','reserved-asset-hashes','out'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new immutable output required')
    inv=ROOT/'tmp/campaign_20261110/inventory';policy=json.loads(a.reservations.read_text());hashes=json.loads(a.reserved_asset_hashes.read_text())
    generated=json.loads(a.readiness.read_text())['objects'];probe=json.loads(a.support_probe.read_text());review=json.loads(a.support_visual_review.read_text())
    if hashes['reservation_sha256']!=sha(a.reservations) or probe['reservation_sha256']!=sha(a.reservations):raise ValueError('reservation mismatch')
    if probe['status']!='private_support_geometry_and_reload_passed' or probe['native_scene_sha256']!=sha(a.support_scene):raise ValueError('private support geometry not admitted')
    if review['status']!='passed' or review['support_probe_sha256']!=sha(a.support_probe):raise ValueError('support visual review missing')
    scene,room=probe['scene'],probe['room'];support=probe['object_name'];block=generated['dice/isgwch'];board=generated['tray/isgbdw']
    if {'scene':scene,'room':room} not in policy['category_holdout_rooms']:raise ValueError('unreserved category room')
    source=inv/'frozen_customs_f24ea424.json';old_alias='configuration_matching-block_design-block_4-breakfast_table_board';entry=copy.deepcopy(json.loads(source.read_text())[old_alias])
    init=json.loads(a.support_scene.read_text())['objects_info']['init_info'];table=init[support]['args'];floor=next(x['args'] for x in init.values() if x['args'].get('category')=='floors' and room in x['args'].get('in_rooms',[]))
    ready=ReadinessCatalog(B1K/'bddl3/bddl/generated_data');refs=[];names={}
    for role,obj in [('support',table),('floor',floor)]:
        st=ready.status(obj['category'],obj['model'])
        if not st['ready']:raise ValueError('unready native scene object')
        category=obj['category'];synset='floor.n.01' if role=='floor' else synset_of_category(category);instance=synset+'_1'
        usd=B1K/'datasets/behavior-1k-assets/objects'/category/obj['model']/'usd'/(obj['model']+'.encrypted.usd')
        refs.append(ObjectRef(instance=instance,role='support',category=category,model=obj['model'],asset_sha256=sha(usd),default_scale=tuple(obj['scale']),semantic_kind='floor' if role=='floor' else 'object',ready=True,readiness_source=st['readiness_source']));names[instance]=obj['name']
    target_synset=synset_of_category(block['category'])
    for n in range(1,5):refs.append(ObjectRef(instance=target_synset+'_'+str(n),role='target',category=block['category'],model=block['model'],asset_sha256=block['asset_sha256'],default_scale=tuple(block['default_scale']),ready=block['ready'],readiness_source=block['readiness_source']))
    refs.append(ObjectRef(instance='tray.n.01_1',role='support',category=board['category'],model=board['model'],asset_sha256=board['asset_sha256'],default_scale=tuple(board['default_scale']),semantic_kind='board',ready=board['ready'],readiness_source=board['readiness_source']))
    for obj in refs:
        if obj.semantic_kind=='floor':continue
        key=obj.category+'/'+obj.model
        if key not in policy['train_exclude_models'] or hashes['ready_assets'][key]['asset_sha256']!=obj.asset_sha256:raise ValueError('nonreserved category-hold asset')
        if obj.semantic_kind!='board' and obj.category not in policy['category_holdout_categories']:raise ValueError('nonheld category role')
    for role,asset in [('targets',block),('board',board)]:
        if not asset['ready'] or asset['split_reservation']!='eval_category':raise ValueError('generated category asset not ready')
        entry['roles'][role].update(category=asset['category'],representative=asset['model'],dataset=asset['dataset'],scale={asset['category']+'/'+asset['model']:asset['default_scale']})
    entry['roles']['support']['category']=table['category']
    entry['generation']['groups']['targets']={'size':4,'draw':'same_model','members':{block['category']:4}}
    # Every listed face is present on this physical Wechsler model. Duplicate
    # colors use the same deterministic last-face map as the native planner.
    facespath=ROOT/'tmp/campaign_20261110/assets/block_variants/isgwch/model/isgwch/build_manifest.json';faces=json.loads(facespath.read_text())['faces']
    entry['generation']['patterns']={'red_white_checker':['red','white','white','red'],'white_red_checker':['white','red','red','white'],'diagonal_pair':['red|white','red','white','red|white']}
    if any(c not in faces.values() for pattern in entry['generation']['patterns'].values() for c in pattern):raise ValueError('pattern face not in Wechsler asset')
    entry['generation'].update(templates_max=1,instances_per_template=1);entry['generation'].pop('remove_objects',None)
    for k in list(entry['generation']):
        if k.startswith('_'):entry['generation'].pop(k)
    for k in list(entry):
        if k.startswith('fallback') or k=='superseded_by':entry.pop(k)
    entry['constraints']=[];entry['scenes']={scene:{'rooms':[room],'bindings':{'floor':{'room':room},'support':[support],'support_surfaces':[{'name':support,'init':'ontop','top_z':probe['measured_height_world'],'row':'campaign_wechsler_category'}]},'whitelist':{'targets':{block['category']:[block['model']]},'board':{'tray':[board['model']]}},'blacklist':{}}}
    alias='configuration_matching-block_design-dice_4-pedestal_table_board';entry['definition_id']=26030
    src=ROOT/'ispatialgym/assets/bddl/activity_definitions/configuration_matching/problem83.bddl'
    text=src.read_text().replace('configuration_matching-83','configuration_matching-26030').replace(old_alias,alias).replace('toy_dice.n.01',target_synset).replace('breakfast_table.n.01',synset_of_category(table['category']))
    a.out.mkdir(parents=True);bddl=a.out/'bddl/configuration_matching/problem26030.bddl';bddl.parent.mkdir(parents=True);bddl.write_text(text)
    entry['campaign_provenance']={'source_customs_sha256':sha(source),'source_bddl_sha256':sha(src),'model_faces':faces,'color_to_face':{c:f for f,c in faces.items()},'reservation_sha256':sha(a.reservations),'generated_readiness_sha256':sha(a.readiness),'support_probe_sha256':sha(a.support_probe),'support_visual_review_sha256':sha(a.support_visual_review),'status':'candidate_only'}
    custom=a.out/'candidate_customs.json';custom.write_text(json.dumps({alias:entry},indent=2)+'\n');gp=GroundedProblem.load(alias,str(custom),str(a.out/'bddl'))
    if gp.rooms(scene)!=[room]:raise ValueError('native room mismatch')
    template=Template(template_id='held_category__wechsler_dice_isgwch',task='configuration_matching-block_design',class_id='configuration_matching-block_design-wechsler',split='eval_category',scene=scene,room=room,objects=tuple(refs),definition_sha256=digest({'entry':entry,'bddl_sha256':sha(bddl)}),scene_setup={'task_rooms':[room],'inst_to_name':names,'runtime_bddl_sha256':sha(bddl),'customs_alias':alias,'native_scene_path':probe['native_scene_path'],'native_scene_sha256':sha(a.support_scene),'model_faces':faces,'color_to_face':{c:f for f,c in faces.items()},'release_status':'candidate_only','fresh_metadata_binding_check':'required','fresh_demo':'pending','fresh_replay':'pending','perturbations':False})
    job={'candidate_id':template.template_id,'class_id':template.class_id,'task':template.task,'alias':alias,'scene':scene,'room':room,'split':'eval_category','template_fingerprint':template.fingerprint,'production_released':False,'quota':1,'measured_support_height_key':probe['actual_height_key'],'environment':{'ISPATIALGYM_STABLE_SCENE_DIR':str(Path(probe['native_scene_path']).parent)},'status':'requires_campaign_support_navigation_hook_and_generic_cube_map','sample_argv':['python','-m','ispatialgym.sampling.sample_isg','--variant',alias,'--customs','${MATERIALIZED}/candidate_customs.json','--bddl-root','${MATERIALIZED}/bddl','--scene',scene,'--templates','1','--instances-per-template','1','--max-attempts','3','--seed','20261110','--no-write-back','--out','${CAMPAIGN_OUT}/validation/'+template.template_id]}
    for name,data in [('validation_jobs.json',{'jobs':[job]}),('contract_candidates.json',{'templates':[{**asdict(template),'fingerprint':template.fingerprint}]})]:(a.out/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'candidate':template.template_id,'manifest_sha256':sha(a.out/'validation_jobs.json'),'quota':1}))


if __name__=='__main__':main()
