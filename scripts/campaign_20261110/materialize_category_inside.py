#!/usr/bin/env python3
"""Compile the category comic-book placement into a measured wardrobe shelf."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
B1K=ROOT/'Behavior-1K'
if not (B1K/'bddl3').exists():
    B1K=Path('/home/minyeongk/Desktop/projects/Behavior-1K-grasp-bands-6a6194600')
sys.path[:0]=[str(ROOT),str(B1K/'bddl3')]
from scripts.campaign_20261110.contracts import ObjectRef,Template,digest,file_digest,require,write_json_atomic
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.readiness_catalog import ReadinessCatalog
from ispatialgym.sampling.bddl_from_iddl import RoleGrounding,render
from ispatialgym.sampling.iddl import parse_file
from ispatialgym.sampling.knowledge import synset_of_category
from ispatialgym.sampling.v2.problem import GroundedProblem,iter_template_choices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inventory','defaults','model-review','scene-probe','scene-review','support-probe','interior-probe','reservations','reserved-asset-hashes','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--height-key',default='0.7860')
    a=p.parse_args();require(not a.out.exists(),'new immutable output required')
    paths=[a.defaults,a.model_review,a.scene_probe,a.scene_review,a.support_probe,a.interior_probe,a.reservations,a.reserved_asset_hashes]
    defaults,review,scene_proof,visual,support,interior,policy,hashes=[json.loads(x.read_text()) for x in paths]
    require(hashes['reservation_sha256']==file_digest(a.reservations),'policy companion differs')
    require(defaults['status']=='native_defaults_and_geometry_recorded' and review['default_geometry_sha256']==file_digest(a.defaults),'native defaults not reviewed')
    require(scene_proof['status']=='private_context_geometry_and_reload_passed' and visual['status']=='passed'
        and visual['context_probe_sha256']==file_digest(a.scene_probe),'private scene review incomplete')
    require(interior['status']=='measured_planes_require_target_geometry_and_maps' and interior['scene_probe_sha256']==file_digest(a.scene_probe),'native interior provenance differs')
    require(support['status']=='private_support_geometry_and_reload_passed' and support['native_scene_sha256']==scene_proof['source_scene_sha256'],'support source differs')
    scene_path=Path(scene_proof['native_scene_path']);require(file_digest(scene_path)==scene_proof['native_scene_sha256']==interior['native_scene_sha256'],'scene bytes differ')
    plane=next(x for x in interior['planes'] if x['height_key']==a.height_key)
    require(plane['n_free_cells']>0 and plane['cells_headroom_at_least_5cm']>0 and plane['open_edges']==['+u'],'unsupported or closed interior')
    require(abs(float(plane['yaw'])-1.5707963267948966)<1e-4,'native opening is not the expected world +y side')
    target=next(x for x in defaults['objects'] if x['category']=='comic_book' and x['model']=='nekxsh')
    require(target['ready'] and review['objects']['comic_book/nekxsh']['status']=='passed','comic book not ready/reviewed')
    scene,room=scene_proof['scene'],scene_proof['room'];init=json.loads(scene_path.read_text())['objects_info']['init_info']
    partner=init[interior['object']]['args'];table=init[support['object_name']]['args']
    require(partner['scale']==interior['default_scale'] and table['scale']==support['default_scale'],'actual support/partner scale differs')
    floor=next(x['args'] for x in init.values() if x['args'].get('category')=='floors' and room in x['args'].get('in_rooms',[]))
    task='inter_object_relation_satisfaction-inside_large_scale';cls=task+'-comic_book-wardrobe';source_class=task+'_02'
    source=a.inventory/'frozen_customs_f24ea424.json';originals=json.loads(source.read_text());records=json.loads((a.inventory/'cluster_templates_with_scales.json').read_text())['records']
    source_alias=next(x['row']['alias'] for x in records if x['key']==source_class);entry=copy.deepcopy(originals[source_alias]);iddl=parse_file(str(ROOT/'ispatialgym/assets/iddl'/entry['iddl']))
    gkey='comic_book__nekxsh__s1'
    entry['roles']['target'].update(category='comic_book',representative='nekxsh',dataset=target['dataset'],require={'grasps':'tabletop'},
        scale={'comic_book/nekxsh':target['default_scale']},grasp_keys={'comic_book/nekxsh':gkey})
    entry['roles']['target'].pop('orientation_equivalences',None)
    entry['roles']['partner']['category']='wardrobe';entry['roles']['support']['category']='pedestal_table'
    entry['roles']={key:entry['roles'][key] for key in iddl.roles};entry['constraints']=[]
    entry['scenes']={scene:dict(rooms=[room],bindings={'floor':{'room':room},'partner':[partner['name']],'support':[table['name']],
        'support_surfaces':[dict(name=table['name'],init='ontop',top_z=support['measured_height_world'],row='campaign_category_inside')]},
        whitelist={'target':{'comic_book':['nekxsh']}},blacklist={})}
    for key in list(entry):
        if key.startswith('fallback') or key=='superseded_by':entry.pop(key)
    for key in list(entry['generation']):
        if key.startswith('_') or key=='remove_objects' or key.startswith('inside_region_'):entry['generation'].pop(key)
    entry['generation'].update(templates_max=1,instances_per_template=1,inside_surface_z_m=plane['height_above_floor'],
        inside_stance_side='+y',partner_joint_init=interior['requested_joints'])
    did=30300;entry['definition_id']=did;alias=iddl.variant_template.format(target='comic_book',partner='wardrobe',support='pedestal_table')
    grounds={key:RoleGrounding(key,'sampleable' if key=='target' else 'scene',category=entry['roles'][key]['category'],synset=synset_of_category(entry['roles'][key]['category'])) for key in iddl.roles if key!='floor'}
    grounds['floor']=RoleGrounding('floor','scene',same_as='floor');rendered=render(iddl,grounds,did,alias,'inter_object_relation_satisfaction')
    a.out.mkdir(parents=True);bddl=a.out/'bddl/inter_object_relation_satisfaction'/f'problem{did}.bddl';bddl.parent.mkdir(parents=True);bddl.write_text(rendered.text)
    ready=ReadinessCatalog(B1K/'bddl3/bddl/generated_data');refs=[];names={}
    for role,obj in [('target',target),('partner',partner),('support',table),('floor',floor)]:
        cat,model=obj['category'],obj['model'];status=ready.status(cat,model);key=cat+'/'+model
        asset=B1K/'datasets/behavior-1k-assets/objects'/cat/model/'usd'/(model+'.encrypted.usd');asset_hash=file_digest(asset)
        require(status['ready'],'unready category role')
        if cat!='floors':require(key in policy['category_holdout_models'] and asset_hash==hashes['ready_assets'][key]['asset_sha256'],'category role/hash not reserved')
        refs.append(ObjectRef(instance=rendered.instances[role],role=role if role in ('target','partner') else 'support',category=cat,model=model,
            asset_sha256=asset_hash,default_scale=tuple(obj['default_scale'] if role=='target' else obj['scale']),semantic_kind='floor' if cat=='floors' else 'object',ready=True,readiness_source=status['readiness_source']))
        if role!='target':names[rendered.instances[role]]=obj['name']
    proof=dict(source_class=source_class,source_customs_sha256=file_digest(source),native_defaults_sha256=file_digest(a.defaults),scene_probe_sha256=file_digest(a.scene_probe),
        scene_visual_review_sha256=file_digest(a.scene_review),interior_probe_sha256=file_digest(a.interior_probe),native_scene_sha256=file_digest(scene_path),reservation_sha256=file_digest(a.reservations),production_released=False)
    entry['campaign_provenance']=proof;custom=a.out/'candidate_customs.json';write_json_atomic(custom,{alias:entry})
    template=Template('held_category__comic_book_wardrobe_shelf'+a.height_key.replace('.','_'),task,cls,'eval_category',scene,room,tuple(refs),digest(dict(entry=entry,bddl_sha256=file_digest(bddl))),
        {**proof,'task_rooms':[room],'inst_to_name':names,'runtime_bddl_sha256':file_digest(bddl),'customs_alias':alias,'fresh_geometry':'pending','fresh_demo':'pending','fresh_replay':'pending','perturbations':False})
    check_reservations(template,policy,hashes)
    import numpy as np
    native=GroundedProblem.load(alias,str(custom),str(a.out/'bddl'));choices=list(iter_template_choices(native,scene,np.random.default_rng(20261110),max_templates=1))
    require(len(choices)==1 and choices[0].models['target'].model=='nekxsh','native category target enumeration differs')
    job=dict(candidate_id=template.template_id,class_id=cls,task=task,alias=alias,scene=scene,room=room,split='eval_category',template_fingerprint=template.fingerprint,
        source_manifest=str(a.out/'validation_jobs.json'),required_new_grasp_key=gkey,environment={'ISPATIALGYM_STABLE_SCENE_DIR':str(scene_path.parent)},production_released=False,
        sample_argv=['python','-m','ispatialgym.sampling.sample_isg','--variant',alias,'--customs',str(custom),'--bddl-root',str(a.out/'bddl'),'--scene',scene,'--templates','1','--instances-per-template','1','--max-attempts','3','--seed','20261110','--no-write-back','--out','${CAMPAIGN_OUT}/validation/'+cls])
    write_json_atomic(a.out/'validation_jobs.json',{'jobs':[job]});write_json_atomic(a.out/'contract_candidates.json',{'templates':[{**asdict(template),'fingerprint':template.fingerprint}]})
    write_json_atomic(a.out/'map_dependency_requests.json',[dict(class_id=cls,grasp_key=gkey,support_height_key=support['actual_height_key'],goal_height_key=plane['height_key'],
        native_plane=plane,native_joints=interior['requested_joints'],interior_probe=str(a.interior_probe),interior_probe_sha256=file_digest(a.interior_probe),
        interior_spec_status='requires native interiors.measure registry before map generation',arms=['right','left'])])
    print(json.dumps(dict(candidates=1,manifest_sha256=file_digest(a.out/'validation_jobs.json'))))


if __name__=='__main__':
    main()
