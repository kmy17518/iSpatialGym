#!/usr/bin/env python3
"""Select 64 ready nonreserved room models by semantics and dimension screening.

The native defaults/visual probe must still pass before materialization.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
B1K=Path('/home/minyeongk/Desktop/projects/Behavior-1K-grasp-bands-6a6194600')
REMOTE_ASSETS='/vision/u/minyeong/projects/ispatialgym/Behavior-1K/datasets/behavior-1k-assets'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inv=ROOT/'tmp/campaign_20261110/inventory';rpath=inv/'split_reservations.json';reserved=json.loads(rpath.read_text())
    hashes=json.loads((ROOT/'tmp/campaign_20261110/remote_evidence/manifests/inventory_revision3/reservation_asset_hashes_r3.json').read_text())
    blocked_models=set(reserved['train_exclude_models']);blocked_categories=set(reserved['train_exclude_categories']);blocked_hashes=set(hashes['reserved_asset_hashes'])
    catalog_path=inv/'category_inventory.json';catalog=json.loads(catalog_path.read_text())['categories']
    source=json.loads((inv/'cluster_templates_with_scales.json').read_text())['records'];queues={};rejected=[]
    for record in source:
        cls=record['key']
        if not cls.startswith(('camera_relocalization-obstructed','object_scaling')):continue
        task='object_scaling' if cls.startswith('object_scaling') else 'camera_relocalization-obstructed'
        role='target' if task=='object_scaling' else 'obstacles'
        binding=next(b for b in record['metadata']['bindings'] if b['role']==role);category=binding['category'];original=binding['model']
        base=next(r for r in catalog[category] if r['model']==original)['bbox_m'];pool=[]
        if category in blocked_categories:raise ValueError('source class is category-reserved')
        for candidate in catalog[category]:
            model=candidate['model'];key=category+'/'+model
            if model==original or not candidate['ready'] or not candidate['usd_present'] or key in blocked_models:continue
            extent=candidate['bbox_m']
            if len(extent)!=3 or not all(v>0 for v in extent):continue
            if task=='object_scaling':
                valid=sorted(extent)[1]<=.085 and max(extent)<=.32 and min(extent)>=.006
                rule='middle bbox dimension <=85mm, longest <=320mm, shortest >=6mm; coarse prefilter only, fresh in-hand grasps required'
            else:
                valid=.12<=max(extent[:2])<=.8 and .1<=extent[2]<=1.7
                rule='household obstacle category; native task footprint .12-.80m and height .10-1.70m bounds'
            if not valid:
                rejected.append({'class_id':cls,'model_key':key,'bbox_m':extent,'reason':'dimension_screen'});continue
            usd=B1K/'datasets/behavior-1k-assets/objects'/category/model/'usd'/(model+'.encrypted.usd');asset_hash=sha(usd)
            if asset_hash in blocked_hashes:continue
            score=sum(abs(math.log(a/b)) for a,b in zip(sorted(extent),sorted(base)))
            pool.append({'class_id':cls,'task':task,'role':role,'category':category,'model':model,'dataset':'behavior-1k-assets',
                'ready':True,'readiness_source':candidate['readiness_source'],'asset_sha256':asset_hash,
                'asset_path':REMOTE_ASSETS+'/objects/'+category+'/'+model+'/usd/'+model+'.encrypted.usd',
                'catalog_bbox_m':extent,'source_model':original,'dimension_distance':score,'screen_rule':rule,
                'default_scale':None,'default_scale_status':'must measure native initialized scale before materialization'})
        queues[cls]=sorted(pool,key=lambda r:(r['dimension_distance'],r['model']))[:9]
    selected=[]
    for rank in range(9):
        for cls in sorted(queues):
            if rank<len(queues[cls]) and len(selected)<64:selected.append(queues[cls][rank])
    if len(selected)!=64:raise ValueError(f'only {len(selected)} dimension-screened ready models')
    output={'schema':'campaign-extra-room-models-v1','reservation_sha256':sha(rpath),'category_inventory_sha256':sha(catalog_path),
        'reservation_asset_hashes_sha256':sha(ROOT/'tmp/campaign_20261110/remote_evidence/manifests/inventory_revision3/reservation_asset_hashes_r3.json'),
        'objects':selected,'rejected_dimension_candidates':rejected,'status':'requires_native_default_geometry_and_visual_review','production_released':0}
    path=inv/'extra_room_models_v1.json'
    if path.exists():raise ValueError('candidate selection is immutable; choose new version')
    path.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'models':len(selected),'per_task':{task:sum(r['task']==task for r in selected) for task in sorted({r['task'] for r in selected})},
        'new_room_candidates':len(selected)*10,'max_extra_models_per_class':max(sum(r['class_id']==cls for r in selected) for cls in queues)}))


if __name__=='__main__':main()
