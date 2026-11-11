"""Prepare coupled attachment scales and stream exact dependency requests.

This is dependency preparation only. A ready pair or support inventory entry is
not a fitted layout, successful demo, or training production authorization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random

from scripts.campaign_20261110.contracts import digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.goal_groups import read_env
from scripts.campaign_20261110.height_dependency_jobs import measured_probe, positive_library, evidence
from scripts.campaign_20261110.reservation_policy import active_policy


def permitted(obj, policy):
    require(obj.get('ready') is True and obj.get('readiness_source') and obj.get('asset_sha256'),
            'ready asset provenance missing')
    require(obj['category'] not in policy['category_holdout_categories'] and
            obj['category']+'/'+obj['model'] not in policy['train_exclude_models'],
            'attachment dependency is held out from training')


def coupled_profile(target, partner):
    identity = {role: {k: obj[k] for k in ('category','model','asset_sha256','default_scale')}
                for role,obj in (('target',target),('partner',partner))}
    key = digest(identity)
    rng = random.Random(int(digest({'campaign':'campaign_20261110','kind':'attachment_coupled_scale',
                                  'pair':key}),16))
    factors = [rng.uniform(.8,1.2),rng.uniform(.8,1.2)]
    scales = {role:[[round(float(v)*factor,4) for v in obj['default_scale']] for factor in factors]
              for role,obj in (('target',target),('partner',partner))}
    require(scales['target'][0] != scales['target'][1] and scales['partner'][0] != scales['partner'][1],
            'scale quantization merged distinct pair draws')
    for role,obj in (('target',target),('partner',partner)):
        require(len(obj['default_scale'])==3 and all(math.isfinite(x) and x>0 for x in obj['default_scale']),
                'invalid original default scale')
        for slot,factor in enumerate(factors):
            require(all(abs(s-d*factor)<=.00005000001 for s,d in zip(scales[role][slot],obj['default_scale'])),
                    'coupled scale rounding differs from native schema')
    return dict(id=key,objects=identity,sampled_uniform_factors=factors,absolute_scales=scales,
                maximum_absolute_scales={role:[round(float(v)*1.2,4) for v in obj['default_scale']]
                    for role,obj in (('target',target),('partner',partner))},
                precision_rule='native v2 absolute scale precision 1e-4; common relative multiplier for both ports',
                mating_geometry_status='requires fresh native scaled attachment spec')


def shared_support_requests(profiles, classes, pair_profiles, policy):
    selected={}
    for binding_id,binding in profiles['bindings'].items():
        if not binding.get('shared_anchor'): continue
        profile=profiles['profiles'][binding['profile_id']]
        permitted(profile,policy)
        identity=(binding['scene'],binding['name'])
        require(identity not in selected,'shared support object duplicated')
        selected[identity]=dict(binding_id=binding_id,scene=binding['scene'],object=binding['name'],
            profile_id=binding['profile_id'],classes=sorted(classes),pair_profiles=sorted(pair_profiles),
            maximum_layout_scale_factor=1.2,maximum_layout_fit='pending_native_free_space_and_pair_footprint',
            usable_edges='pending_native_chair_removal_and_navigation',production_admitted=False)
    require(len(selected)==10,'attachment training requires ten reserved support candidates')
    return list(selected.values())


def verify_pair_request(job, pair, configuration):
    refs={role:next(o for o in configuration['objects'] if o['role']==role) for role in ('target','partner')}
    profile=coupled_profile(refs['target'],refs['partner']);slot=pair['scale_slot']
    require(type(slot) is int and slot in (0,1),'invalid coupled scale slot')
    require(pair['class_id']==configuration['class_id'] and job['campaign_classes']==[pair['class_id']],
            'pair class differs from accepted source')
    require(pair['profile_id']==profile['id']==job['campaign_profile'],'coupled profile identity changed')
    require(job['campaign_scale_slot']==slot and job['scale']==profile['absolute_scales']['target'][slot]
            and pair['partner']['scale']==profile['absolute_scales']['partner'][slot], 'pair scales are not the coupled draw')
    for actual,ref in ((job,refs['target']),(pair['partner'],refs['partner'])):
        require(all(actual[k]==ref[k] for k in ('category','model','asset_sha256','ready','readiness_source')),
                'pair dependency differs from ready source asset')


def verify_candidate_axis(job,candidates):
    measured=candidates.get('log',{}).get('attach_axis',{})
    frames=measured.get('mates',[])
    require(frames,'candidate mating-frame provenance missing')
    if job['attach_axis']=='seat':
        from ispatialgym.grasping.candidates import attach_seat_record
        require(job.get('attach_seat_proof'),'scaled seat-rule entry lacks exact native seat proof')
        require(attach_seat_record(job) is not None,'scaled native seat proof is unavailable')
        require(measured.get('seat_record') is True and all(f.get('source')=='seat' for f in frames),
                'scaled seat-rule candidates used a metadata fallback')
    else:
        require(job['attach_axis']=='meta_z' and all(f.get('source') in ('meta_z','bottom_centre') for f in frames),
                'metadata-axis candidates used a different mating rule')
    return measured


def prepare(root, code, output, classes, *, seat_facts=None):
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling import grasp_library as G
    root,code,output=map(lambda p:Path(p).resolve(),(root,code,output))
    require(output.is_relative_to(root) and not output.exists(),'new campaign dependency directory required')
    canonical_path=root/'infra/standard_recovery/canonical_validation.json'
    canonical=json.loads(canonical_path.read_text()); rows={r['class_id']:r for r in canonical['classes']}
    pointer,policy,_=active_policy(root);require(pointer is not None,'active reservations unavailable')
    profile_path=root/'manifests/support_height_profiles.json';supports=json.loads(profile_path.read_text())
    jobs,pairs,profiles,sources=[],[],{},[]
    for cls in sorted(classes):
        row=rows[cls];require(cls.startswith('configuration_matching-attachment_') and row['validated'] is True,
                              'attachment canonical source has not validated')
        config=row['configuration'];objects={role:next(o for o in config['objects'] if o['role']==role)
                                           for role in ('target','partner')}
        for obj in objects.values():
            permitted(obj,policy)
            asset=code/'Behavior-1K/datasets/behavior-1k-assets/objects'/obj['category']/obj['model']/'usd'/(obj['model']+'.encrypted.usd')
            require(file_digest(asset)==obj['asset_sha256'],'ready asset changed')
        source=next(e for e in row['evidence'] if e['split']=='eval_standard')
        for name in ('tro','report','sidecar'):
            ref=source['artifacts'][name];require(file_digest(ref['path'])==ref['sha256'],'accepted source artifact changed')
        gate=source['gate'];require(file_digest(gate['path'])==gate['sha256'],'source replay gate changed')
        require(json.loads(Path(gate['path']).read_text()).get('success') is True,'source replay is not true')
        tro=Path(source['artifacts']['tro']['path']);job_dir=next(p for p in tro.parents if (p/'job.env').is_file())
        meta=json.loads((tro.parent.parent/'template_meta.json').read_text())
        env=read_env(job_dir/'job.env');glib=env['ISPATIALGYM_GRASP_LIBRARY_V2']
        alias=meta['choice']['models']['target']['grasp_key']
        os.environ['ISPATIALGYM_GRASP_LIBRARY_V2']=glib;G._v2_index.cache_clear()
        lib=G.v2_entry(alias);require(lib is not None and bool(lib.grasps()),'native source grasp route is not positive')
        target,partner=objects['target'],objects['partner'];data=lib.data
        require((data['category'],data['model'],tuple(data['scale']))==
                (target['category'],target['model'],S.canonical_scale(target['default_scale'])),
                'native source library does not match target default')
        S.check_compatible(lib,scale=target['default_scale'],rest_kind='stable')
        require(data.get('constraint')=='attachment_free','source is not an attachment grasp constraint')
        profile=coupled_profile(target,partner);profiles[profile['id']]=profile
        source_proof=dict(class_id=cls,configuration=config,validation_key=row['validation_key'],
            accepted_source=source,job_env=evidence(job_dir/'job.env'),library=evidence(lib.path),
            source_alias=alias,canonical_observation=evidence(canonical_path))
        sources.append(source_proof)
        for slot in (0,1):
            scale=profile['absolute_scales']['target'][slot]
            job=dict(category=target['category'],model=target['model'],scale=scale,dataset=data['dataset'],
                constraint=data['constraint'],rest_hint_wxyz=data.get('rest_hint_wxyz'),rest_pose='canonical',
                attach_axis=lib.attach_axis(),kind='tabletop',legacy_key=None,
                source='campaign_attachment_coupled_scale',campaign_classes=[cls],
                campaign_profile=profile['id'],campaign_scale_slot=slot,ready=True,
                readiness_source=target['readiness_source'],asset_sha256=target['asset_sha256'])
            job['cand_overrides']={'attach_axis':job['attach_axis']}
            if job['attach_axis']=='seat' and seat_facts:
                from ispatialgym.grasping.candidates import attach_seat_record
                seat_path=Path(seat_facts).resolve();require(seat_path.is_relative_to(root),'seat facts outside campaign')
                job['attach_seat_proof']={'path':str(seat_path),'sha256':file_digest(seat_path),
                    'partner':{'category':partner['category'],'model':partner['model'],
                               'scale':profile['absolute_scales']['partner'][slot]}}
                require(attach_seat_record(job) is not None,'exact paired native seat proof missing')
            job['id']=S.entry_id(job['category'],job['model'],scale,job['constraint'],job['rest_hint_wxyz'])
            jobs.append(job)
            pairs.append(dict(id=cls+'__'+profile['id'][:12]+'__s'+str(slot),class_id=cls,
                male_entry=job['id'],partner={k:partner[k] for k in ('category','model','asset_sha256','ready','readiness_source')}
                    |dict(scale=profile['absolute_scales']['partner'][slot]),profile_id=profile['id'],scale_slot=slot,
                status='awaiting_positive_scaled_library_and_measured_support',production_admitted=False))
    fit_requests=shared_support_requests(supports,classes,profiles,policy)
    probes,failures={},[]
    paths=[*root.glob('height_probes/*/height_probe.json'),*root.glob('height_probes_v2/*/height_probe.json')]
    for path in sorted(paths,key=lambda p:p.stat().st_mtime):
        try:
            probe,binding,profile=measured_probe(path,supports,file_digest(profile_path),root)
            permitted(profile,policy)
            if not binding.get('shared_anchor'):continue
            probes[(probe['binding_id'],probe['slot'])]=dict(probe=evidence(path),binding_id=probe['binding_id'],
                height_slot=probe['slot'],actual_height_key=probe['actual_height_key'],
                stable_scene=dict(path=probe['stable_scene_path'],sha256=probe['stable_scene_sha256']))
        except (OSError,ValueError,KeyError,TypeError) as exc:failures.append(dict(path=str(path),reason=str(exc)))
    output.mkdir()
    for name,value in [('jobs.json',jobs),('pairs_pending.json',pairs),('profiles.json',profiles),
                       ('sources.json',sources),('support_fit_requests.json',fit_requests)]:write_json_atomic(output/name,value)
    report=dict(schema='campaign-training-attachment-dependencies-v1',campaign_id='campaign_20261110',
        observed_at=datetime.now(timezone.utc).isoformat(),canonical_observation=evidence(canonical_path),
        support_profiles=evidence(profile_path),reservation_policy=pointer,jobs=len(jobs),pairs=len(pairs),
        measured_support_slots=list(probes.values()),probe_failures=failures,production_admitted=False,
        pipeline='advance each positive exact scaled library to measured-height attachment specs/maps independently',
        remaining=['positive physics-tested scaled libraries','native scaled mating-port geometry',
            'positive exact-height tabletop and attachment maps','maximum scaled layout fit and usable edges',
            'native demo and own physics replay','augmentation restoration and batched replay',
            'shared plus applicable-path plus exact-template production gates'])
    write_json_atomic(output/'dependencies.json',report)
    return report


def ready_map_requests(root, batch):
    """Publish each completed entry immediately, independent of other shards."""
    from ispatialgym.grasping import schema as S
    root,batch=Path(root).resolve(),Path(batch).resolve()
    require(batch.is_relative_to(root),'dependency batch outside campaign')
    dep=json.loads((batch/'dependencies.json').read_text())
    pointer,policy,_=active_policy(root)
    require(pointer['policy_sha256']==dep['reservation_policy']['policy_sha256'],'reservation policy changed')
    profile_ref=dep['support_profiles'];require(file_digest(profile_ref['path'])==profile_ref['sha256'],'support profiles changed')
    profiles=json.loads(Path(profile_ref['path']).read_text())
    pairs={p['male_entry']:p for p in json.loads((batch/'pairs_pending.json').read_text())}
    sources={s['class_id']:s for s in json.loads((batch/'sources.json').read_text())}
    ready,blocked=[],[]
    for shard,job in enumerate(json.loads((batch/'jobs.json').read_text())):
        pair=pairs[job['id']];output=batch/'ready_map_requests'/pair['id']
        if (output/'request.json').exists():
            ready.append(evidence(output/'request.json'));continue
        try:
            source=sources[pair['class_id']]['configuration']
            verify_pair_request(job,pair,source)
            log=batch/'build/logs'/('glib_'+str(shard)+'.log')
            require(log.is_file() and 'GB_VALIDATE_DONE' in log.read_text(),'entry physics validation is incomplete')
            lp=batch/'build/glib/models'/job['id']/'library.json'
            libdata=positive_library(job,lp,policy)
            axis=verify_candidate_axis(job,json.loads(lp.with_name('candidates.json').read_text()))
            lib=S.load_model(job['category'],job['model'],job['scale'],str(batch/'build/glib'),
                constraint=job['constraint'],rest_hint_wxyz=job.get('rest_hint_wxyz'))
            require(lib is not None and Path(lib.path).resolve()==lp.resolve() and bool(lib.grasps()),
                    'native reader did not resolve exact positive scaled entry')
            S.check_compatible(lib,scale=job['scale'],rest_kind='stable')
            permitted(pair['partner'],policy)
            names=source['scene_setup']['inst_to_name']
            source_supports={names[o['instance']] for o in source['objects'] if o['role']=='support'
                             and o.get('semantic_kind')=='object'}
            probes=[]
            for ref in dep['measured_support_slots']:
                require(file_digest(ref['probe']['path'])==ref['probe']['sha256'],'measured probe changed')
                probe,binding,profile=measured_probe(ref['probe']['path'],profiles,profile_ref['sha256'],root)
                permitted(profile,policy)
                if binding['scene']==source['scene'] and binding['name'] in source_supports:
                    probes.append(ref)
            require(probes,'no admitted scaled-height probe on canonical support; other supports need a new binding proof')
            heights=sorted({p['actual_height_key'] for p in probes});anchors=[];surfaces=[]
            for height in heights:
                matching=[p for p in probes if p['actual_height_key']==height]
                ids=[p['binding_id']+'/slot'+str(p['height_slot']) for p in matching]
                anchors.append(dict(key=height,height=float(height),precision=.0001,
                    range=[float(height)-.00005,float(height)+.00005],surfaces=ids,status=['measured']))
                for ref,sid in zip(matching,ids):
                    native=json.loads(Path(ref['probe']['path']).read_text())
                    surfaces.append(dict(id=sid,key=height,height=native['measured_height_above_floor'],
                        scene=native['scene'],support=native['object'],surface='top',precision=.0001,
                        source=ref['probe']['path'],floor_z=native['floor_z_world'],status='measured'))
            output.mkdir(parents=True)
            attach_pair={**pair,'heights':heights};attach_pair.pop('status',None)
            for name,value in [('jobs.json',[job]),('pairs.json',[attach_pair]),('heights_by_id.json',{job['id']:heights}),
                ('anchors.json',dict(schema='isg-height-anchors/1',max_approx_m=.00005,anchors=anchors,surfaces=surfaces))]:
                write_json_atomic(output/name,value)
            request=dict(schema='campaign-training-attachment-map-request-v1',class_id=pair['class_id'],
                pair=attach_pair,library=evidence(lp),library_validation_log=evidence(log),
                candidate_axis=axis,candidate_axis_evidence=evidence(lp.with_name('candidates.json')),
                positive_grasps=len(libdata['grasps']),measured_support_slots=probes,
                reservation_sha256=pointer['policy_sha256'],parent_dependencies=evidence(batch/'dependencies.json'),
                input_files=[evidence(batch/name) for name in ('jobs.json','pairs_pending.json','profiles.json','sources.json')],
                requires=dict(tabletop_arms=['right','left'],attachment_arms=['right'],attachment_rules=['box','swept']),
                status='ready_for_exact_native_map_build',production_admitted=False)
            write_json_atomic(output/'request.json',request);ready.append(evidence(output/'request.json'))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            blocked.append(dict(entry_id=job['id'],class_id=pair['class_id'],reason=str(exc)))
    report=dict(observed_at=datetime.now(timezone.utc).isoformat(),ready=ready,blocked=blocked,production_admitted=False)
    write_json_atomic(batch/'map_request_status.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--batch',type=Path)
    p.add_argument('--seat-facts',type=Path)
    for name in ('code','output'):p.add_argument('--'+name,type=Path)
    p.add_argument('--class-id',action='append',default=[]);a=p.parse_args()
    if a.batch:
        report=ready_map_requests(a.root,a.batch)
        print(json.dumps({'ready':len(report['ready']),'blocked':report['blocked'],'production_admitted':False}));return
    require(a.code and a.output and a.class_id,'preparation needs code, output and class IDs')
    report=prepare(a.root,a.code,a.output,set(a.class_id),seat_facts=a.seat_facts)
    print(json.dumps({k:report[k] for k in ('jobs','pairs','production_admitted')}|{
        'measured_support_slots':len(report['measured_support_slots'])}))


if __name__=='__main__':main()
