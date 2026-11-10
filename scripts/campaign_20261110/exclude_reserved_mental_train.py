#!/usr/bin/env python3
"""Publish a filtered train-room revision after a monotonic holdout extension."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reservations',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new candidate revision required')
    policy=json.loads(a.reservations.read_text());key='classroom_mock_up/isgm10'
    if key not in policy['train_exclude_models']:raise ValueError('new reservation missing')
    definitions=json.loads((a.source/'candidate_customs.json').read_text())
    omitted_aliases={alias for alias,entry in definitions.items() if 'isgm10' in json.dumps(entry)}
    if len(omitted_aliases)!=1:raise ValueError('expected exactly one generated model variant')
    definitions={k:v for k,v in definitions.items() if k not in omitted_aliases}
    jobs=json.loads((a.source/'validation_jobs.json').read_text());old_jobs=jobs['jobs']
    removed=[j for j in old_jobs if j['alias'] in omitted_aliases];jobs['jobs']=[j for j in old_jobs if j['alias'] not in omitted_aliases]
    if len(removed)!=10:raise ValueError('expected ten unreleased model-room candidates')
    old_templates=json.loads((a.source/'contract_candidates.json').read_text())['templates']
    templates=[t for t in old_templates if not any(o['category']+'/'+o['model']==key for o in t['objects'])]
    if len(templates)!=len(jobs['jobs']) or len(old_templates)-len(templates)!=10:raise ValueError('contract removal count mismatch')
    floating=json.loads((a.source/'generated_mental_floating_jobs.json').read_text());floating=[j for j in floating if j['category']+'/'+j['model']!=key]
    if len(floating)!=9:raise ValueError('expected nine remaining train mental floating jobs')
    a.out.mkdir(parents=True);custom=a.out/'candidate_customs.json';custom.write_text(json.dumps(definitions,indent=2)+'\n')
    for entry in definitions.values():
        paths=list((a.source/'bddl').glob('*/problem'+str(entry['definition_id'])+'.bddl'))
        if len(paths)!=1:raise ValueError('missing or duplicated native definition')
        target=a.out/'bddl'/paths[0].parent.name/paths[0].name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(paths[0],target)
    jobs.update(customs_sha256=sha(custom),effective_reservation_sha256=sha(a.reservations),
        superseded_jobs_sha256=sha(a.source/'validation_jobs.json'),removed_candidate_ids=[j['candidate_id'] for j in removed],
        excluded_new_model_key=key,production_released=0)
    for filename,data in [('validation_jobs.json',jobs),('contract_candidates.json',{'templates':templates}),('generated_mental_floating_jobs.json',floating)]:
        (a.out/filename).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    (a.out/'revision.json').write_text(json.dumps({'source_manifest_sha256':sha(a.source/'validation_jobs.json'),
        'new_manifest_sha256':sha(a.out/'validation_jobs.json'),'effective_reservation_sha256':sha(a.reservations),
        'removed_model_key':key,'removed_candidate_ids':[j['candidate_id'] for j in removed],
        'remaining_candidate_fingerprints':'unchanged; no runtime definition, object, room or scale changes to retained candidates'},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'remaining_train_candidates':len(templates),'removed':len(removed),'floating_train_jobs':len(floating)}))


if __name__=='__main__':main()
