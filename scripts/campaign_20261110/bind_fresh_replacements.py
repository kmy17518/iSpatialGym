#!/usr/bin/env python3
"""Bind candidate contracts to successful fresh native templates and asset bytes."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--contracts-dir',type=Path,required=True)
    p.add_argument('--reserved-hashes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.contracts_dir))
    from contracts import ObjectRef,Template,digest
    candidates={t['class_id']:t for t in json.loads(a.candidates.read_text())['templates']}
    reserved=json.loads(a.reserved_hashes.read_text())
    excluded_hashes=set(reserved['reserved_asset_hashes'])
    excluded_models=set(reserved['reserved_model_keys'])
    output=[];skipped=[]
    for class_id,template in candidates.items():
        for path in sorted((a.root/class_id).rglob('template_meta.json')):
            meta=json.loads(path.read_text())
            if meta.get('n_written',0)<1:
                skipped.append({'class_id':class_id,'path':str(path),'reason':'no successful fresh sample'});continue
            directory=path.parent;full=directory/'template.json'
            state=json.loads(full.read_text());init_info=state['objects_info']['init_info']
            stable_path=a.code/'Behavior-1K/datasets/2026-challenge-task-instances/scenes'/meta['scene']/'json'/(meta['scene']+'_stable.json')
            stable_init=json.loads(stable_path.read_text())['objects_info']['init_info']
            if sha(full)!=meta['template_sha256']:
                raise ValueError(f'template bytes differ from native metadata: {directory}')
            matching=[b for b in (directory/'activity_definitions').rglob('*.bddl') if sha(b)==meta['bddl_sha256']]
            if len(matching)!=1 or meta['bddl_sha256']!=template['scene_setup']['runtime_bddl_sha256']:
                raise ValueError(f'fresh BDDL differs from the exact approved candidate definition: {directory}')
            by_asset={(o['role'],o['category'],o['model']):o for o in template['objects']}
            role_names={'target':'target','partner':'partner','distractors':'distractor'}
            refs=[];bindings={};actual_counts=Counter();init_subset={};seen=set();scale_resolutions=[]
            for binding in meta['bindings']:
                if binding['name'] in seen:continue
                seen.add(binding['name'])
                init=init_info[binding['scene_object_name']]['args']
                role=role_names.get(binding['role'],'support');cat,model=init['category'],init['model']
                key=(role,cat,model);expected=by_asset.get(key)
                if expected is None or not expected['ready']:
                    raise ValueError(f'unexpected/unready fresh role {class_id}/{key}')
                scale=init.get('scale')
                if not isinstance(scale,list) or len(scale)!=3:
                    raise ValueError(f'fresh object scale missing {class_id}/{key}')
                if max(abs(x-y) for x,y in zip(scale,expected['default_scale']))>1e-6:
                    frozen=stable_init.get(binding['scene_object_name'],{}).get('args',{})
                    if (frozen.get('category'),frozen.get('model'))!=(cat,model) or not frozen.get('scale') or max(abs(x-y) for x,y in zip(scale,frozen['scale']))>1e-6:
                        raise ValueError(f'fresh object scale differs from candidate and exact frozen stable scene {class_id}/{key}')
                    scale_resolutions.append({'instance':binding['name'],'candidate_scale':expected['default_scale'],
                        'actual_default_scale':scale,'frozen_stable_scale':frozen['scale'],'source':str(stable_path),
                        'source_sha256':sha(stable_path),'reason':'Actual untouched stable-scene default supersedes earlier best-scene candidate metadata.'})
                dataset=init.get('dataset_name') or 'behavior-1k-assets'
                asset_root=a.code/'ispatialgym/assets/datasets'/dataset if dataset=='ispatialgym-assets' else a.code/'Behavior-1K/datasets'/dataset
                asset=asset_root/'objects'/cat/model/'usd'/(model+('.encrypted.usd' if dataset=='behavior-1k-assets' else '.usd'))
                actual_hash=sha(asset)
                if actual_hash!=expected['asset_sha256']:
                    raise ValueError(f'actual runtime asset changed {class_id}/{key}')
                if cat not in {'walls','floors'} and (actual_hash in excluded_hashes or cat+'/'+model in excluded_models):
                    raise ValueError(f'reserved byte/model leak {class_id}/{key}')
                refs.append(ObjectRef(**{**expected,'instance':binding['name'],'default_scale':tuple(scale)}))
                actual_counts[key]+=1;bindings[binding['name']]=binding['scene_object_name'];init_subset[binding['name']]=init
            if actual_counts!=Counter((o['role'],o['category'],o['model']) for o in template['objects']):
                raise ValueError(f'fresh role multiplicities differ from candidate {class_id}')
            setup={**template['scene_setup'],'source_template':str(directory),'source_template_sha256':sha(full),
                   'runtime_template_id':meta['template_id'],'runtime_bddl_sha256':meta['bddl_sha256'],
                   'source_bddl_sha256':meta['bddl_sha256'],'source_alias':meta['variant'],'inst_to_name':bindings,
                   'task_rooms':meta['task_rooms'],'scene_variant':meta.get('scene_variant',{}),
                   'fresh_geometry':'native sampler wrote a fresh TRO','fresh_demo':'pending','fresh_replay':'pending',
                   'fresh_metadata_sha256':sha(path),'actual_init_args':init_subset,
                   'default_scale_resolution':scale_resolutions,
                   'reserved_asset_hash_manifest_sha256':sha(a.reserved_hashes)}
            record=Template(template_id=template['template_id'],task=template['task'],class_id=class_id,
                            split=template['split'],scene=meta['scene'],room=meta['task_rooms'][0],objects=tuple(refs),
                            definition_sha256=digest({'candidate_definition_sha256':template['definition_sha256'],
                                'fresh_template_sha256':sha(full),'bddl_sha256':meta['bddl_sha256'],'actual_init_args':init_subset}),scene_setup=setup)
            output.append({**asdict(record),'fingerprint':record.fingerprint})
    if a.output.exists(): raise FileExistsError('fresh contract output already exists')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'templates':output,'skipped':skipped,'production_released':0},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'fresh_contracts':len(output),'classes':[t['class_id'] for t in output],'output':str(a.output),'sha256':sha(a.output)}))


if __name__=='__main__':main()
