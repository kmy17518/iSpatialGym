"""Clone four held pairs with adopted, measured same-category role replacements."""
import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

from scripts.campaign_20261110.bind_fresh_candidates import load_contract
from scripts.campaign_20261110.contracts import digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import active_policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','extension-audit','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();require(not a.out.exists() and a.out.resolve().is_relative_to(a.root.resolve()),'new owned output required')
    pointer,policy,hashes=active_policy(a.root);audit=json.loads(a.extension_audit.read_text())
    require(pointer['reservations']['sha256']==audit['new_reservation_sha256'],'replacement policy not adopted')
    native=Path(audit['native_defaults']['path']);visual=Path(audit['visual_review']['path'])
    require(file_digest(native)==audit['native_defaults']['sha256'] and file_digest(visual)==audit['visual_review']['sha256'],'native proof changed')
    measured={r['category']+'/'+r['model']:r for r in json.loads(native.read_text())['objects']}
    changes={r['class_id']:r for r in audit['changed_candidate_assignments']}
    contracts={r['class_id']:load_contract(r) for r in json.loads((a.source/'contract_candidates.json').read_text())['templates']}
    source_customs=a.source/'candidate_customs.json';original=json.loads(source_customs.read_text());customs={};templates=[];jobs=[]
    for job0 in json.loads((a.source/'validation_jobs.json').read_text())['jobs']:
        cls=job0['class_id']
        if cls not in changes:continue
        change=changes[cls];role=change['role'];key=change['new'];row=measured[key];template=contracts[cls]
        entry=copy.deepcopy(original[job0['alias']]);require(entry['roles'][role]['category']==row['category'],'replacement changes category')
        require(row['ready'] and row['default_scale']==[1.,1.,1.] and file_digest(Path(row['asset_path']))==row['asset_sha256'],'replacement readiness or default identity changed')
        entry['roles'][role].update(representative=row['model'],dataset=row['dataset'],scale={key:row['default_scale']})
        for name in ('rest_pose','rest_hint_wxyz','orientation_equivalences','grasp_keys','frame'):
            entry['roles'][role].pop(name,None)
        if role=='target':entry['roles'][role]['grasp_keys']={key:row['category']+'__'+row['model']+'__s1'}
        for scene in entry['scenes'].values():scene['whitelist'][role]={row['category']:[row['model']]}
        # Remove only stale diagnostics for a historical, unrelated target; no
        # native fit rule or default-scale choice is changed.
        if entry['generation'].get('target_fit'):
            entry['generation']['target_fit']={k:v for k,v in entry['generation']['target_fit'].items() if k==key}
        objects=[];count=0
        for obj in template.objects:
            if obj.role==role:
                require(obj.model_key==change['previous'],'source role differs from audited replacement');count+=1
                obj=replace(obj,model=row['model'],asset_sha256=row['asset_sha256'],default_scale=tuple(row['default_scale']),ready=True,readiness_source=row['readiness_source'])
            objects.append(obj)
        require(count==1,'replacement role ambiguous')
        proof=dict(path=str(a.extension_audit),sha256=file_digest(a.extension_audit),source_customs_sha256=file_digest(source_customs),change=change)
        entry.setdefault('campaign_provenance',{})['ready_pair_replacement']=proof
        setup={**template.scene_setup,'reservation_sha256':pointer['reservations']['sha256'],'ready_pair_replacement':proof}
        updated=replace(template,template_id=template.template_id+'__ready_pair_replacement_v1',objects=tuple(objects),
            definition_sha256=digest(dict(source_definition=template.definition_sha256,entry=entry,proof=proof)),scene_setup=setup)
        check_reservations(updated,policy,hashes)
        job=copy.deepcopy(job0);job.update(candidate_id=updated.template_id,template_fingerprint=updated.fingerprint)
        job['sample_argv']=[x.replace(str(a.source),'${MATERIALIZED}') for x in job['sample_argv']]
        job['blockers']=['Fresh exact replacement maps/contact dependencies and native task geometry are required.']
        customs[job['alias']]=entry;templates.append({**asdict(updated),'fingerprint':updated.fingerprint});jobs.append(job)
    require(len(jobs)==4,'four audited classes required');a.out.mkdir();shutil.copytree(a.source/'bddl',a.out/'bddl')
    write_json_atomic(a.out/'candidate_customs.json',customs)
    from ispatialgym.sampling.v2.problem import GroundedProblem,iter_template_choices
    import numpy as np
    for job in jobs:
        gp=GroundedProblem.load(job['alias'],str(a.out/'candidate_customs.json'),str(a.out/'bddl'))
        choices=list(iter_template_choices(gp,job['scene'],np.random.default_rng(20261110),max_templates=1))
        require(len(choices)==1,'native uniqueness/lookalike preflight failed')
        change=changes[job['class_id']];model=choices[0].models[change['role']]
        require(model.category+'/'+model.model==change['new'] and list(model.scale)==measured[change['new']]['default_scale'],'native chosen model/default scale differs')
    write_json_atomic(a.out/'validation_jobs.json',dict(jobs=jobs,production_released=False))
    write_json_atomic(a.out/'contract_candidates.json',dict(templates=templates))
    print(json.dumps(dict(candidates=4,manifest=str(a.out/'validation_jobs.json'),sha256=file_digest(a.out/'validation_jobs.json'))))


if __name__=='__main__':main()
