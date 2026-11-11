"""Audit and freeze one ready coupled attachment configuration independently."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import numpy as np

from scripts.campaign_20261110.contracts import digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.height_dependency_jobs import evidence, measured_probe, positive_library
from scripts.campaign_20261110.reservation_policy import active_policy
from scripts.campaign_20261110.training_attachment_dependencies import permitted, verify_pair_request


def positive_lattice(native, statuses):
    require(np.isfinite(native.cells).all() and np.isfinite(native.yaws).all(),'nonfinite native map lattice')
    arrays=[v for k,v in native.arrays.items() if k.startswith('s')]
    require(arrays and all(set(np.unique(v))<=set(statuses.values()) for v in arrays),'invalid native map states')
    n=sum(int((v==statuses['feasible']).sum()) for v in arrays)
    require(n>0,'native map has no feasible cells')
    return n


def audit(root,batch,request_dir,height_slot):
    from ispatialgym.grasping import schema as S, contact_policy as CP
    from ispatialgym.sampling import attach_maps as AM, grasp_library as G, reach_maps as RM
    from ispatialgym.sampling.generation_defaults import defaults_for
    root,batch,request_dir=map(lambda p:Path(p).resolve(),(root,batch,request_dir))
    require(batch.is_relative_to(root) and request_dir.is_relative_to(batch),'request outside campaign batch')
    req=json.loads((request_dir/'request.json').read_text());pair=req['pair'];entry=pair['male_entry']
    pointer,policy,_=active_policy(root);require(pointer['policy_sha256']==req['reservation_sha256'],'reservation policy changed')
    for ref in [req['library'],req['library_validation_log'],req['parent_dependencies'],*req['input_files']]:
        require(file_digest(ref['path'])==ref['sha256'],'dependency input changed: '+ref['path'])
    jobs={x['id']:x for x in json.loads((batch/'jobs.json').read_text())};job=jobs[entry]
    sources={x['class_id']:x for x in json.loads((batch/'sources.json').read_text())}
    verify_pair_request(job,pair,sources[pair['class_id']]['configuration']);permitted(pair['partner'],policy)
    lp=Path(req['library']['path']);data=positive_library(job,lp,policy)
    lib=S.load_model(job['category'],job['model'],job['scale'],str(lp.parent.parent.parent),
        constraint=job['constraint'],rest_hint_wxyz=job.get('rest_hint_wxyz'))
    require(lib is not None and Path(lib.path).resolve()==lp.resolve() and bool(lib.grasps()),'native scaled library differs')
    S.check_compatible(lib,scale=job['scale'],rest_kind='stable')
    matches=[r for r in req['measured_support_slots'] if r['height_slot']==height_slot]
    require(len(matches)==1,'measured support slot missing or ambiguous');support=matches[0]
    pp=root/'manifests/support_height_profiles.json';profiles=json.loads(pp.read_text())
    require(file_digest(support['probe']['path'])==support['probe']['sha256'],'support probe changed')
    probe,binding,profile=measured_probe(support['probe']['path'],profiles,file_digest(pp),root);permitted(profile,policy)
    height=float(probe['actual_height_key']);table_maps=[];attach_maps=[]
    for rest in sorted({g['rest_id'] for g in data['grasps']}):
        for arm in ('right','left'):
            native=RM.ReachMap.find(str(request_dir/'tabletop/maps'/entry),lib,height,rest_id=rest,
                arm=arm,policy_key=CP.DEFAULT.key(arm),robot=RM.robot_digest())
            require(native is not None,'missing exact tabletop map '+arm)
            require(native.meta['kind']=='tabletop' and native.meta['key']==RM.map_key(native.meta)
                    and abs(native.meta['height']['value']-height)<1e-8,'tabletop map identity differs')
            table_maps.append({**evidence(native.path),'npz':evidence(Path(native.path).with_suffix('.npz')),
                'positive_cells':positive_lattice(native,RM.STATUS),'rest_id':rest,'arm':arm,'height':height,
                'object_key':lib.identity_key()})
    sp=request_dir/'attachment/specs'/pair['id']/'attach_spec.json';spec=AM.load_spec(str(sp))
    require(spec is not None and spec.data['id']==pair['id'],'native mating spec missing')
    require(all(spec.data['partner'][k]==pair['partner'][k] for k in ('category','model','scale')),
            'native mating partner differs')
    require(spec.data['male']['entry']==entry,'native mating male differs')
    records=[]
    for log in (request_dir/'attachment/logs').glob('build_right_*.log'):
        records += [json.loads(line.split(' ',1)[1]) for line in log.read_text(errors='replace').splitlines()
                    if line.startswith(('ATTACH_SPEC ','ATTACH_MAP '))]
    require(any(r.get('ok') is True and r.get('digest')==spec.digest() for r in records),
            'native successful spec measurement receipt missing')
    partner=pair['partner']
    for rule in ('box','swept'):
        native=AM.AttachMap.find(str(request_dir/'attachment/maps'/entry),height,
            partner=(partner['category'],partner['model'],partner['scale']),rest_id=G.v2_rest_id(lib),
            object_key=lib.identity_key(),arm='right',hover=AM.oracle_hover_params(rule),
            pre_back_m=defaults_for('configuration_matching-attachment')['pre_attach_back_m'],
            policy_key=CP.DEFAULT.key('right'),robot=RM.robot_digest())
        require(native is not None,'missing exact attachment map '+rule)
        require(native.meta['spec']['digest']==spec.digest() and native.meta['male']['entry']==entry,
                'attachment map does not use measured scaled mating geometry')
        require(any(r.get('ok') is True and r.get('path')==native.path and r.get('rule')==rule
                    and r.get('h')==probe['actual_height_key'] for r in records),'successful map build receipt missing')
        attach_maps.append({**evidence(native.path),'npz':evidence(Path(native.path).with_suffix('.npz')),
            'positive_cells':positive_lattice(native,AM.STATUS),'rule':rule,'height':height,'arm':'right',
            'rest_id':G.v2_rest_id(lib),'object_key':lib.identity_key(),'spec_digest':spec.digest()})
    row=dict(entry_id=entry,classes=[pair['class_id']],target_scale=job['scale'],target_scale_slot=pair['scale_slot'],
        coupled_pair=pair,support_binding=probe['binding_id'],support_height_slot=height_slot,
        support_scale=probe['absolute_scale'],actual_height_key=probe['actual_height_key'],
        probe=support['probe'],stable_scene=support['stable_scene'],library=req['library'],
        positive_grasps=len(data['grasps']),arms=['right','left'],maps=table_maps,attachment_maps=attach_maps,
        attachment_spec=evidence(sp),partner_geometry=evidence(sp.with_name('partner_geometry.json')),
        attachment_spec_digest=spec.digest(),status='native_dependencies_verified_layout_pending',production_admitted=False)
    refs=[row[k] for k in ('probe','stable_scene','library','attachment_spec','partner_geometry')]
    refs += [ref for record in table_maps+attach_maps for ref in (record,record['npz'])]
    return dict(schema='isg-training-attachment-native-audit-v1',campaign_id='campaign_20261110',
        observed_at=datetime.now(timezone.utc).isoformat(),active_policy_sha256=pointer['policy_sha256'],
        request=evidence(request_dir/'request.json'),requests=1,verified=1,failures=[],checked=[row],
        files={ref['path']:ref['sha256'] for ref in refs},production_admitted=False,
        remaining=['maximum scaled layout fit','fresh native binding and demo','own physics replay',
            'augmentation restoration','batched replay','shared, applicable-path and exact-template production gates'])


def verify_release(release,proof,routes):
    from ispatialgym.grasping import schema as S, contact_policy as CP
    from ispatialgym.sampling import attach_maps as AM, grasp_library as G, reach_maps as RM
    from ispatialgym.sampling.generation_defaults import defaults_for
    release=Path(release)
    os.environ['ISPATIALGYM_GRASP_LIBRARY_V2']=str(release/'glib');G._v2_index.cache_clear()
    for row in proof['checked']:
        entry=row['entry_id'];route=routes[entry];lib=G.v2_entry(route['alias'])
        require(lib is not None and Path(lib.path).resolve()==(release/'glib/models'/entry/'library.json').resolve(),
                'merged attachment scale alias resolves to wrong entry')
        require(file_digest(lib.path)==route['library_sha256'] and G.grasps_from_v2(lib,route['alias']),
                'merged native scale route is not positive')
        S.check_compatible(lib,scale=row['target_scale'],rest_kind='stable')
        for record in row['maps']:
            native=RM.ReachMap.find(str(release/'maps'/entry),lib,record['height'],
                rest_id=record['rest_id'],arm=record['arm'],policy_key=CP.DEFAULT.key(record['arm']),robot=RM.robot_digest())
            require(native is not None and file_digest(native.path)==record['sha256'],'merged tabletop map differs')
        partner=row['coupled_pair']['partner']
        for record in row['attachment_maps']:
            native=AM.AttachMap.find(str(release/'maps'/entry),record['height'],
                partner=(partner['category'],partner['model'],partner['scale']),rest_id=G.v2_rest_id(lib),
                object_key=lib.identity_key(),arm=record['arm'],hover=AM.oracle_hover_params(record['rule']),
                pre_back_m=defaults_for('configuration_matching-attachment')['pre_attach_back_m'],
                policy_key=CP.DEFAULT.key(record['arm']),robot=RM.robot_digest())
            require(native is not None and file_digest(native.path)==record['sha256'] and
                    native.meta['spec']['digest']==row['attachment_spec_digest'],'merged attachment map differs')


def publish(root,previous,audit_path,destination):
    from scripts.campaign_20261110.merge_height_dependency_release import physical_library
    root,previous,audit_path,destination=map(lambda p:Path(p).resolve(),(root,previous,audit_path,destination))
    require(all(p.is_relative_to(root) for p in (previous,audit_path,destination)) and not destination.exists(),
            'new private campaign release required')
    proof=json.loads(audit_path.read_text());before=json.loads((previous/'release_manifest.json').read_text())
    require(proof['schema']=='isg-training-attachment-native-audit-v1' and
            proof['verified']==proof['requests']==1 and not proof['failures'],'one fully verified attachment dependency required')
    pointer,_,_=active_policy(root);require(proof['active_policy_sha256']==pointer['policy_sha256'],'reservation policy changed')
    for path,sha in proof['files'].items():require(file_digest(path)==sha,'audited dependency changed')
    destination.mkdir(mode=0o750);write_json_atomic(destination/'release_block.json',{'reason':'attachment native readers pending'})
    files={};routes={}
    def copy(source,relative,expected):
        source=Path(source).resolve();target=destination/relative
        require(source.is_relative_to(root) and target.resolve().is_relative_to(destination),'dependency path escapes campaign')
        require(file_digest(source)==expected,'dependency bytes changed')
        if target.exists():require(file_digest(target)==expected,'conflicting dependency bytes')
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        files[str(relative)]=expected
    for relative,sha in before['files'].items():copy(previous/relative,relative,sha)
    for row in proof['checked']:
        entry=row['entry_id'];lp=Path(row['library']['path'])
        for source in lp.parent.rglob('*'):
            if source.is_file():copy(source,Path('glib/models')/entry/source.relative_to(lp.parent),file_digest(source))
        library=destination/'glib/models'/entry/'library.json';original=json.loads(library.read_text())
        alias='campaign_train__'+entry;updated={**original,'legacy_keys':list(dict.fromkeys([*(original.get('legacy_keys')or[]),alias]))}
        require(physical_library(updated)==physical_library(original),'alias modified physical library')
        write_json_atomic(library,updated);files[str(library.relative_to(destination))]=file_digest(library)
        routes[entry]=dict(alias=alias,library=str(library),library_sha256=file_digest(library),
            source_library_sha256=row['library']['sha256'],physical_payload_sha256=digest(physical_library(original)),scale=row['target_scale'])
        for record in row['maps']+row['attachment_maps']:
            for ref in (record,record['npz']):copy(ref['path'],Path('maps')/entry/Path(ref['path']).name,ref['sha256'])
        for name in ('attachment_spec','partner_geometry'):
            ref=row[name];copy(ref['path'],Path('evidence/attachment_training')/entry/Path(ref['path']).name,ref['sha256'])
    anchors=json.loads((previous/'anchors.json').read_text());by_key={r['key']:r for r in anchors['anchors']}
    for row in proof['checked']:
        key=row['actual_height_key'];height=float(key)
        if key not in by_key:by_key[key]=dict(key=key,height=height,precision=.0001,surfaces=[],status=['measured_native_dependency'])
        require(abs(by_key[key]['height']-height)<1e-8,'anchor differs from measurement')
        by_key[key]['surfaces'].append(row['support_binding']+'/slot'+str(row['support_height_slot']))
    ordered=sorted(by_key.values(),key=lambda r:r['height']);maximum=anchors['max_approx_m']
    for i,row in enumerate(ordered):
        lo=(ordered[i-1]['height']+row['height'])/2 if i else float('-inf')
        hi=(ordered[i+1]['height']+row['height'])/2 if i+1<len(ordered) else float('inf')
        row['range']=[max(lo,row['height']-maximum),min(hi,row['height']+maximum)]
    anchors['anchors']=ordered;write_json_atomic(destination/'anchors.json',anchors);files['anchors.json']=file_digest(destination/'anchors.json')
    verify_release(destination,proof,routes)
    report=dict(schema='isg-training-attachment-release-v1',campaign_id='campaign_20261110',
        observed_at=datetime.now(timezone.utc).isoformat(),audit=str(audit_path),audit_sha256=file_digest(audit_path),
        routes=routes,native_reader_errors=0,production_admitted=False,remaining=proof['remaining'])
    write_json_atomic(destination/'training_height_reader_audit.json',report)
    files['training_height_reader_audit.json']=file_digest(destination/'training_height_reader_audit.json')
    write_json_atomic(destination/'release_manifest.json',{**before,'created_at':report['observed_at'],
        'previous_release':str(previous),'files':files,'production_admitted':False})
    (destination/'release_block.json').unlink()
    for relative,sha in before['files'].items():require(file_digest(previous/relative)==sha,'previous release changed')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','batch','request','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--height-slot',type=int,choices=(0,1),required=True);a=p.parse_args()
    require(a.output.resolve().is_relative_to(a.root.resolve()) and not a.output.exists(),'new campaign audit required')
    proof=audit(a.root,a.batch,a.request,a.height_slot);write_json_atomic(a.output,proof)
    print(json.dumps({'verified':proof['verified'],'production_admitted':False,'output':str(a.output)}))


if __name__=='__main__':main()
