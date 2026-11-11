"""Materialize exact scaled-target / measured-support native validation inputs.

The original ready defaults stay in the logical contract. Applied scales live
in a separate plan and are checked against fresh native bindings before any
annotated canary can run. This is preparation, never a production release.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path

from scripts.campaign_20261110.bind_fresh_candidates import load_contract, bind_one
from scripts.campaign_20261110.contracts import ScaleState, Perturbations, digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import active_policy


def scale_plan(template, request, probe, profile, job):
    require(template.task in ('disembedding','inter_object_relation_satisfaction-under_small_scale'),
            'this materializer does not yet implement the other task scaling rules')
    require(template.class_id in request['classes'], 'dependency belongs to another class')
    require(template.scene==probe['scene'], 'support probe belongs to another scene')
    targets=[o for o in template.objects if o.role=='target']
    require(len(targets)==1, 'coupled/group targets need their dedicated scaling implementation')
    target=targets[0]
    require((target.category,target.model,target.asset_sha256)==(job['category'],job['model'],job['asset_sha256']),
            'target dependency identity differs')
    require(target.category!='jigsaw_puzzle_piece', 'LEF target/distractor coupling must use a dedicated plan')
    names=template.scene_setup['inst_to_name']
    supports=[o for o in template.objects if names.get(o.instance)==probe['object']]
    require(len(supports)==1 and supports[0].role=='support' and supports[0].semantic_kind=='object',
            'measured support does not uniquely match the native support')
    support=supports[0]
    require((support.category,support.model,support.asset_sha256)==(profile['category'],profile['model'],profile['asset_sha256']),
            'support profile identity differs')
    require(list(support.default_scale)==profile['default_scale'], 'support default differs from probed baseline')
    require(request['target_scale']==job['scale'] and request['support_scale']==probe['absolute_scale'],
            'requested applied scales differ from measured dependencies')
    ratios=[a/b for a,b in zip(job['scale'],target.default_scale)]
    require(all(.8-1e-4<=x<=1.2+1e-4 for x in ratios) and max(ratios)-min(ratios)<2e-4,
            'target scale is not a uniform draw relative to its actual default')
    states={o.instance:ScaleState(o.default_scale,o.default_scale,'fixed') for o in template.objects}
    states[target.instance]=ScaleState(target.default_scale,tuple(job['scale']),'uniform')
    states[support.instance]=ScaleState(support.default_scale,tuple(probe['absolute_scale']),'height')
    result=Perturbations(states)
    result.validate('train',states)
    return result


def bind_training_candidate(template, job, metadata_path, code, reservations, hashes, pointer):
    require(template.split=='train' and job['production_released'] is False, 'training preparation only')
    plan=job['training_scale_plan']
    require(digest(plan)==template.scene_setup['training_scale_plan_sha256'], 'training scale plan changed')
    states={name:ScaleState(**value) for name,value in plan['scales'].items()}
    require(set(states)=={o.instance for o in template.objects}, 'scale plan object set differs')
    require(all(states[o.instance].default==o.default_scale for o in template.objects), 'original defaults lost')
    physical=replace(template,objects=tuple(replace(o,default_scale=states[o.instance].applied) for o in template.objects))
    bound=bind_one(physical,job,metadata_path,code,reservations,hashes,pointer)
    originals={(o.role,o.category,o.model):o for o in template.objects}
    refs=[];mapped={}
    for o in bound.objects:
        original=originals[(o.role,o.category,o.model)]
        state=states[original.instance]
        require(tuple(o.default_scale)==state.applied, 'fresh native applied scale differs')
        refs.append(replace(o,default_scale=original.default_scale));mapped[o.instance]=state
    setup={**bound.scene_setup,'candidate_fingerprint':template.fingerprint,
        'training_scale_plan_sha256':digest(plan),'training_original_defaults_preserved':True}
    return replace(bound,objects=tuple(refs),scene_setup=setup),Perturbations(mapped)


def materialize(root, release, code, output, classes):
    root,release,code,output=map(lambda p:Path(p).resolve(),(root,release,code,output))
    require(output.is_relative_to(root) and not output.exists(), 'new private preparation directory required')
    canonical_path=root/'infra/standard_recovery/canonical_validation.json'
    canonical=json.loads(canonical_path.read_text())
    source={r['class_id']:r for r in canonical['classes']}
    release_report=release/'training_height_reader_audit.json';dep=json.loads(release_report.read_text())
    require(dep['campaign_id']=='campaign_20261110' and dep['native_reader_errors']==0, 'dependency release is not audited')
    proof=json.loads(Path(dep['audit']).read_text())
    require(file_digest(dep['audit'])==dep['audit_sha256'], 'source dependency audit changed')
    profiles=json.loads((root/'manifests/support_height_profiles.json').read_text())
    jobs={j['id']:j for j in json.loads((root/'manifests/train_scale_glib_jobs.json').read_text())}
    pointer,policy,hashes=active_policy(root)
    require(proof['active_policy_sha256']==pointer['policy_sha256'], 'holdout policy changed')
    require(all(source[c]['validated'] is True for c in classes), 'canonical source has not validated')
    output.mkdir()
    records=[];templates=[]
    for req in proof['checked']:
        selected=set(req['classes']).intersection(classes)
        if not selected:continue
        probe=json.loads(Path(req['probe']['path']).read_text())
        profile=profiles['profiles'][probe['profile_id']]
        for cls in sorted(selected):
            baseline=load_contract(source[cls]['configuration'])
            plan=scale_plan(baseline,req,probe,profile,jobs[req['entry_id']])
            source_dir=Path(baseline.scene_setup['source_template']);meta=json.loads((source_dir/'template_meta.json').read_text())
            # Freeze the exact source declaration, including role order. Fresh
            # geometry still checks the altered scale, support and full layout.
            customs_path=Path(meta['customs']);customs=json.loads(customs_path.read_text())
            alias=meta['alias'];entry=copy.deepcopy(customs[alias])
            identity=digest({'class':cls,'dependency':req,'scale_plan':asdict(plan)})[:20]
            out=output/identity;out.mkdir()
            block=copy.deepcopy(entry['scenes'][baseline.scene]);block['rooms']=[baseline.room]
            block['bindings']=copy.deepcopy(meta['choice']['bindings']);block['blacklist']={};block['whitelist']={}
            require(block['bindings']['support']==[probe['object']], 'support binding differs from measured scene')
            block['bindings'].pop('support_regions',None)
            block['bindings']['support_surfaces']=[{'name':probe['object'],'init':'ontop',
                'top_z':probe['measured_height_world'],'row':'training_height_'+identity}]
            for role,models in [(role,[obj])for role,obj in meta['choice']['models'].items()]+list(meta['choice'].get('groups',{}).items()):
                spec=entry['roles'][role]
                for model in models:
                    category,mid=model['category'],model['model'];key=category+'/'+mid
                    refs=[o for o in baseline.objects if (o.category,o.model)==(category,mid)]
                    require(refs and len({plan.scales[o.instance].applied for o in refs})==1,'model has ambiguous applied scales')
                    scale=plan.scales[refs[0].instance].applied
                    spec.setdefault('scale',{})[key]=list(scale)
                    block['whitelist'].setdefault(role,{}).setdefault(category,[])
                    if mid not in block['whitelist'][role][category]:block['whitelist'][role][category].append(mid)
                    if role=='target':spec.setdefault('grasp_keys',{})[key]=dep['routes'][req['entry_id']]['alias']
            entry['scenes']={baseline.scene:block};entry['constraints']=[]
            entry['generation'].update(templates_max=1,instances_per_template=1)
            for key in list(entry['generation']):
                if key.startswith('_'):entry['generation'].pop(key)
            for key in list(entry):
                if key.startswith('fallback')or key=='superseded_by':entry.pop(key)
            entry['campaign_provenance']={'status':'native_training_geometry_validation_only','class_id':cls,
                'source_customs_sha256':file_digest(customs_path),'dependency_audit_sha256':dep['audit_sha256'],
                'original_default_scales_preserved':True,'scale_plan':asdict(plan),'production_released':False}
            custom=out/'customs.json';custom.write_text(json.dumps({alias:entry},indent=2)+'\n')
            matches=[p for p in (source_dir/'activity_definitions').rglob('*.bddl') if file_digest(p)==meta['bddl_sha256']]
            require(len(matches)==1,'exact source BDDL missing or ambiguous')
            bddl=out/'bddl'/matches[0].parent.name/matches[0].name;bddl.parent.mkdir(parents=True);bddl.write_bytes(matches[0].read_bytes())
            setup={**baseline.scene_setup,'runtime_bddl_sha256':meta['bddl_sha256'],
                'native_scene_sha256':req['stable_scene']['sha256'],'training_scale_plan_sha256':digest(asdict(plan)),
                'training_height_dependency':req,'training_dependency_release':str(release),
                'training_dependency_release_sha256':file_digest(release/'release_manifest.json'),
                'release_status':'native_geometry_pending','production_released':False}
            candidate=replace(baseline,template_id='train_height_'+identity,split='train',scene_setup=setup,
                definition_sha256=digest({'source':baseline.definition_sha256,'customs':file_digest(custom),'bddl':meta['bddl_sha256']}))
            check_reservations(candidate,policy,hashes)
            contract=out/'contract_candidates.json';write_json_atomic(contract,{'templates':[asdict(candidate)]})
            job={'candidate_id':candidate.template_id,'class_id':cls,'alias':alias,'scene':baseline.scene,'room':baseline.room,
                'source_manifest':str(contract),'training_scale_plan':asdict(plan),'production_released':False,
                'dependency_release':str(release),'environment':{'ISPATIALGYM_STABLE_SCENE_DIR':str(Path(req['stable_scene']['path']).parent),
                  'ISPATIALGYM_GRASP_LIBRARY_V2':str(release/'glib'),'ISPATIALGYM_REACH_MAPS':str(release/'maps'),
                  'ISPATIALGYM_REACH_MAP_ANCHORS':str(release/'anchors.json'),'ISPATIALGYM_V2_ONLY':'1'},
                'sample_argv':['python','-m','ispatialgym.sampling.sample_isg','--variant',alias,'--customs',str(custom),
                  '--bddl-root',str(out/'bddl'),'--scene',baseline.scene,'--room',baseline.room,'--templates','1',
                  '--instances-per-template','1','--max-attempts','3','--seed','2026111108','--no-write-back',
                  '--out','${CAMPAIGN_OUT}/validation/'+identity],
                'remaining':['native maximum scaled layout and binding','annotated training canary','own physics replay',
                             'scene/light augmentation restoration','distinct group sampling','distribution compliance']}
            write_json_atomic(out/'validation_jobs.json',{'jobs':[job]})
            records.append(job);templates.append(asdict(candidate))
    require(records,'no matching audited requests')
    write_json_atomic(output/'validation_jobs.json',{'jobs':records,'production_released':False,
        'canonical_observation_sha256':file_digest(canonical_path),'dependency_release_sha256':file_digest(release/'release_manifest.json')})
    print(json.dumps({'candidates':len(records),'classes':sorted(classes),'production_released':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','release','code','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--class-id',action='append',required=True)
    a=p.parse_args();materialize(a.root,a.release,a.code,a.output,set(a.class_id))
