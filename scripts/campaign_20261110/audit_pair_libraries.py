#!/usr/bin/env python3
"""Admit completed positive pair libraries independently for exact-height maps."""
import argparse
import json
import os
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from scripts.campaign_20261110.reservation_policy import active_policy,policy_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'secondary-root', 'out'):
        p.add_argument('--' + name,type=Path,required=True)
    p.add_argument('--frozen-reservations',type=Path)
    p.add_argument('--frozen-asset-hashes',type=Path)
    a=p.parse_args();root=a.root.resolve();require(not a.out.exists() and any(a.out.resolve().is_relative_to(p.resolve()) for p in (root,a.secondary_root)),'new owned private output required')
    if a.frozen_reservations or a.frozen_asset_hashes:
        require(a.frozen_reservations and a.frozen_asset_hashes,'both immutable policy sources required')
        policy,hashes=[json.loads(path.read_text()) for path in (a.frozen_reservations,a.frozen_asset_hashes)]
        require(hashes['reservation_sha256']==file_digest(a.frozen_reservations),'frozen policy companion differs')
        pointer={'policy_sha256':policy_digest(policy),'dependency_audit_only':True,
            'reservations':{'path':str(a.frozen_reservations),'sha256':file_digest(a.frozen_reservations)},
            'asset_hashes':{'path':str(a.frozen_asset_hashes),'sha256':file_digest(a.frozen_asset_hashes)}}
    else:
        pointer,policy,hashes=active_policy(root)
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling import grasp_library as G
    specs=[('model',a.secondary_root/'heldsmallpairs_prebuild_v1',root/'heldout_small_pairs_v1/tabletop_jobs.json','0.7588'),
           ('category',a.secondary_root/'category_smallpairs_prebuild_v1',root/'heldout_category_pairs_v1/tabletop_jobs.json','0.6634')]
    admitted=[];skipped=[];batches=[];a.out.mkdir(parents=True)
    for kind,output,manifest,height in specs:
        jobs=json.loads(manifest.read_text());kept=[];indices=[]
        os.environ['ISPATIALGYM_GRASP_LIBRARY_V2']=str(output/'glib');G._v2_index.cache_clear()
        for index,job in enumerate(jobs):
            base=dict(split_kind=kind,source_index=index,id=job['id'],source_manifest=str(manifest),source_manifest_sha256=file_digest(manifest))
            try:
                key=job['category']+'/'+job['model']
                require(job['ready'] and bool(job['readiness_source']),'source model is not ready')
                require(key in policy[kind+'_holdout_models'],'source model is no longer reserved')
                expected=hashes['ready_assets'][key];require(job['asset_sha256']==expected['asset_sha256']==file_digest(expected['path']),'source asset changed')
                log=output/'logs'/f'glib_{index}.log';contents=log.read_text()
                require('GB_VALIDATE_DONE' in contents and 'Simulation App Shutting Down' in contents,'native validation/process not complete')
                lib=S.load_model(job['category'],job['model'],job['scale'],str(output/'glib'),constraint=job['constraint'],rest_hint_wxyz=job.get('rest_hint_wxyz'))
                require(lib is not None and len(lib.grasps())>0,'no positive native grasps')
                require(Path(lib.path).parent.name==job['id'],'native entry ID differs')
                require(lib.data['dataset']==job['dataset'] and lib.data['constraint']==job['constraint'],'native constraint/dataset differs')
                require(lib.data.get('legacy_key')==job.get('legacy_key') and lib.rest_kind=='stable' and bool(lib.rests),'native alias/rest identity differs')
                S.check_compatible(lib,scale=job['scale'],rest_kind='stable')
                routed=G.v2_entry(job['legacy_key']);require(routed is not None and Path(routed.path)==Path(lib.path),'native alias resolver differs')
                require(len(G.grasps_from_v2(routed,job['legacy_key']))>0,'native reader exposes no positive grasps')
                record={**base,'source':str(Path(lib.path).parent),'library_sha256':file_digest(lib.path),'validation_log':str(log),
                    'validation_log_sha256':file_digest(log),'positive_grasps':len(lib.grasps()),'asset_sha256':job['asset_sha256'],
                    'scale':job['scale'],'constraint':job['constraint'],'alias':job['legacy_key'],'rest_kind':'stable',
                    'rest_ids':sorted(lib.rests),'tabletop_height_key':height,'arms':['right','left']}
                admitted.append(record);kept.append(job);indices.append(index)
            except (ValueError,OSError,KeyError) as exc:
                skipped.append({**base,'reason':str(exc)})
        kept_path=a.out/(kind+'_admitted_jobs.json');write_json_atomic(kept_path,kept)
        heights=a.out/(kind+'_heights_by_id.json');write_json_atomic(heights,{job['id']:[height] for job in kept})
        batches.append(dict(kind=kind,source_indices=indices,admitted_manifest=str(kept_path),admitted_manifest_sha256=file_digest(kept_path),
            heights_by_id=str(heights),heights_sha256=file_digest(heights),source_library_root=str(output/'glib')))
    tray=root/'heldout_probes/category_disembedding_tray_floors_v1/tray_floor_probe.json';td=json.loads(tray.read_text())
    require(td['status']=='native_tray_floor_measurement_passed','native placed tray heights missing')
    eraser=next((x for x in admitted if x['id']=='rubber_eraser__cungod__s1'),None)
    placed=[]
    if eraser:
        for model in ('hjxczh','thkphg'):
            obj=td['objects'][model]
            placed.append(dict(grasp_key=eraser['alias'],source_library=eraser['source'],library_sha256=eraser['library_sha256'],
                height_key=obj['map_height_key'],placed_key=obj['placed_key'],tray_model='tray/'+model,tray_scale=obj['default_scale'],
                tray_asset_sha256=obj['asset_sha256'],tray_floor_probe=str(tray),tray_floor_probe_sha256=file_digest(tray),arms=['right','left']))
    write_json_atomic(a.out/'placed_tray_map_requests.json',placed)
    report=dict(schema='campaign-exact-pair-library-admission-v1',active_policy_sha256=pointer['policy_sha256'],policy_provenance=pointer,admitted=admitted,skipped=skipped,
        batches=batches,placed_tray_requests=placed,task_geometry_demo_replay_still_required=True,production_released=False)
    write_json_atomic(a.out/'audit.json',report)
    print(json.dumps(dict(admitted=len(admitted),skipped=skipped,batches=batches,placed_tray_requests=len(placed),audit_sha256=file_digest(a.out/'audit.json'))))


if __name__=='__main__':
    main()
