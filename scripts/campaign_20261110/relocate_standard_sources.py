#!/usr/bin/env python3
"""Copy exact historical BDDL source bytes into private provenance directories.

Only contracts whose source metadata points to a missing file are rewritten.
Expected BDDL hashes and original full-template hashes must both match. Runtime
semantic comparison remains required by the normal authoring pipeline.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contracts',type=Path,required=True);p.add_argument('--frozen-bddl',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new immutable output required')
    original=json.loads(a.contracts.read_text());doc=copy.deepcopy(original);index={}
    for path in a.frozen_bddl.rglob('*.bddl'):index.setdefault(sha(path),path)
    a.out.mkdir(parents=True);rows=[];unresolved=[]
    for template in doc['templates']:
        setup=template['scene_setup'];source=Path(setup['source_template']);meta_path=source/'template_meta.json'
        if not meta_path.is_file():unresolved.append({'class_id':template['class_id'],'reason':'source metadata absent'});continue
        meta=json.loads(meta_path.read_text());declared=(source/meta['activity_definition']).resolve()
        if declared.is_file():continue
        expected=setup['source_bddl_sha256'];found=index.get(expected)
        if found is None:
            for candidate in source.rglob('*.bddl'):
                if sha(candidate)==expected:found=candidate;break
        if found is None:unresolved.append({'class_id':template['class_id'],'reason':'exact source bytes not found','sha256':expected});continue
        full=source/'template.json'
        if not full.is_file() or sha(full)!=setup['source_template_sha256']:
            unresolved.append({'class_id':template['class_id'],'reason':'original full-template hash mismatch'});continue
        dest=a.out/'templates'/template['class_id'];dest.mkdir(parents=True)
        activity,number=meta['problem'].rsplit('-',1);bddl=dest/'activity_definitions'/activity/('problem'+number+'.bddl');bddl.parent.mkdir(parents=True)
        bddl.write_bytes(found.read_bytes());shutil.copyfile(full,dest/'template.json')
        new_meta=copy.deepcopy(meta);new_meta['activity_definition']=str(bddl.resolve())
        (dest/'template_meta.json').write_text(json.dumps(new_meta,indent=2,sort_keys=True)+'\n')
        row={'class_id':template['class_id'],'old_fingerprint':template['fingerprint'],'old_source_template':str(source),'old_meta_sha256':sha(meta_path),'old_missing_bddl':str(declared),'recovered_source':str(found),'expected_bddl_sha256':expected,'new_source_template':str(dest.resolve()),'new_meta_sha256':sha(dest/'template_meta.json'),'full_template_sha256':sha(full),'new_bddl_sha256':sha(bddl)}
        setup.update(source_template=str(dest.resolve()),source_relocation_provenance={k:v for k,v in row.items() if k!='old_fingerprint'})
        template.pop('fingerprint');template['fingerprint']=digest(template);row['new_fingerprint']=template['fingerprint'];rows.append(row)
    output=a.out/'standard_contract_ready_relocated.json';output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    audit={'schema':'campaign-source-relocation-v1','input_contracts':str(a.contracts),'input_sha256':sha(a.contracts),'output_sha256':sha(output),'repaired':rows,'unresolved':unresolved,'scope':'Only source locators/provenance changed. Every original file remains unchanged; exact source/runtime semantic comparison still required. Existing active/accepted configs are not rewritten.'}
    (a.out/'relocation_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'repaired':len(rows),'unresolved':unresolved,'output':str(output),'sha256':sha(output)}))


if __name__=='__main__':main()
