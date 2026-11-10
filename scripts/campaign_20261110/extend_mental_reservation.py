#!/usr/bin/env python3
"""Propose a monotonic reservation extension; allocator adoption is separate."""
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REMOTE='/vision/u/minyeong/projects/isg_home/campaign_20261110/manifests/reservation_revision4'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    if path.exists():raise ValueError('immutable output exists: '+str(path))
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def main():
    inv=ROOT/'tmp/campaign_20261110/inventory';out=inv/'reservation_revision4';out.mkdir(parents=True,exist_ok=True)
    basepath=inv/'split_reservations.json';base=json.loads(basepath.read_text())
    oldhashpath=ROOT/'tmp/campaign_20261110/remote_evidence/manifests/inventory_revision3/reservation_asset_hashes_r3.json'
    oldhashes=json.loads(oldhashpath.read_text());readypath=ROOT/'tmp/campaign_20261110/remote_evidence/manifests/generated_project_readiness_r2.json'
    registry=json.loads(readypath.read_text());key='classroom_mock_up/isgm10';obj=registry['objects'][key]
    usepath=out/'pre_adoption_use_audit.json';use=json.loads(usepath.read_text())
    if not use['no_accepted_or_group_config_use'] or use['model_key']!=key or use['asset_sha256']!=obj['asset_sha256']:
        raise ValueError('zero-use audit missing/mismatched')
    if not obj['ready'] or oldhashes['reservation_sha256']!=sha(basepath):raise ValueError('readiness/base policy mismatch')
    if key in base['train_exclude_models']:raise ValueError('extension already in base')
    policy=copy.deepcopy(base)
    for field in ['model_holdout_models','train_exclude_models']:policy[field]=sorted(set(policy[field])|{key})
    policy['readiness_evidence'][key]=obj['readiness_source']
    row=next(t for t in policy['model_holdout_templates'] if t['class_id']=='mental_rotation-classroom_mock_up')
    target=next(r for r in row['roles'] if r['role']=='target')
    if target['model']!='tpfmpb':raise ValueError('unexpected prior classroom hold')
    target.update(model='isgm10',dataset=obj['dataset'],ready=True,readiness_source=obj['readiness_source'],
        asset_sha256=obj['asset_sha256'],default_scale=obj['default_scale'],geometry_validated=False,
        replaced_assignment='classroom_mock_up/tpfmpb remains globally reserved but unused')
    policy['policy_extension']={'revision':4,'parent_reservation_sha256':sha(basepath),'added_model_keys':[key],
        'status':'requires_allocator_adoption','unchanged_policy':'Existing reserved models, categories and rooms remain reserved.'}
    newpath=out/'split_reservations.json';save(newpath,policy)
    hashes=copy.deepcopy(oldhashes);hashes['reservation_sha256']=sha(newpath)
    hashes['reserved_model_keys']=sorted(set(hashes['reserved_model_keys'])|{key})
    hashes['ready_assets'][key]={'asset_sha256':obj['asset_sha256'],'path':obj['usd'],'ready':True,'readiness_provenance':obj['readiness_source']}
    hashes['reserved_asset_hashes']=sorted(set(hashes['reserved_asset_hashes'])|{obj['asset_sha256']})
    hashpath=out/'reservation_asset_hashes.json';save(hashpath,hashes)
    registry['objects'][key]['split_reservation']='eval_model'
    registry['objects'][key]['split_assignment_provenance']={'reservation_sha256':sha(newpath),'extension_audit':REMOTE+'/extension_audit.json',
        'prior_split':'train proposal only; zero accepted/group-config use verified'}
    registry['effective_reservation_sha256']=sha(newpath)
    readynew=out/'generated_project_readiness_r3.json';save(readynew,registry)
    audit={'schema':'campaign-monotonic-reservation-extension-v1','old_reservation_sha256':sha(basepath),'new_reservation_sha256':sha(newpath),
        'old_asset_hashes_sha256':sha(oldhashpath),'new_asset_hashes_sha256':sha(hashpath),'added_model_keys':[key],
        'added_asset_sha256':{key:obj['asset_sha256']},'added_categories':[],'added_rooms':[],
        'reason':'Replace visually ambiguous molecular-lattice mental-rotation candidate with a generated asymmetric wood model; retain tpfmpb as globally reserved and unused.',
        'readiness_source':obj['readiness_source'],'source_readiness_registry_sha256':sha(readypath),'new_readiness_registry_sha256':sha(readynew),
        'pre_adoption_use_audit_sha256':sha(usepath),'pre_adoption_use_audit':REMOTE+'/pre_adoption_use_audit.json',
        'old_model_stays_reserved':'classroom_mock_up/tpfmpb','removed_unreleased_train_candidates':10,
        'raw_library_reuse':'Only exact same model/USD hash/default scale/constraint physical prebuild may be reused; no train instances accepted.',
        'status':'proposal_requires_atomic_allocator_and_camera_policy_adoption'}
    for field in ['model_holdout_models','category_holdout_models','train_exclude_models']:
        if not set(base[field]).issubset(policy[field]):raise ValueError('nonmonotonic model policy')
    for field in ['category_holdout_categories','train_exclude_categories','category_holdout_rooms','category_only_exempt_semantic_kinds','structural_model_category_exempt']:
        if base[field]!=policy[field]:raise ValueError('existing category/room policy changed')
    save(out/'extension_audit.json',audit)
    print(json.dumps({'models':len(hashes['reserved_model_keys']),'new_policy_sha256':sha(newpath),'new_asset_hashes_sha256':sha(hashpath),'extension_audit_sha256':sha(out/'extension_audit.json')}))


if __name__=='__main__':main()
