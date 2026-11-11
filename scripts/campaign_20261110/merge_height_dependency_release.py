"""Add audited measured-height maps to a private exact-scale data release.

Only dependency readiness is established. Native template layout, augmentation,
demo, replay, acceptance and distribution gates remain mandatory.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic, digest
from scripts.campaign_20261110.reservation_policy import active_policy


def physical_library(data):
    return {k:v for k,v in data.items() if k not in ('legacy_key','legacy_keys')}


def merge(root, previous, audit, destination):
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling import grasp_library as G, reach_maps as RM
    from ispatialgym.sampling.height_anchors import AnchorSet

    root, previous, audit, destination = [Path(p).resolve() for p in (root,previous,audit,destination)]
    require(all(p.is_relative_to(root) for p in (previous,audit,destination)) and not destination.exists(),
            'new private campaign release required')
    before=json.loads((previous/'release_manifest.json').read_text())
    proof=json.loads(audit.read_text())
    require(proof['schema']=='isg-measured-height-native-audit-v1' and proof['campaign_id']=='campaign_20261110', 'wrong audit')
    require(proof['verified']==proof['requests']>0 and not proof['failures'], 'native dependency audit incomplete')
    pointer, _, _=active_policy(root)
    require(proof['active_policy_sha256']==pointer['policy_sha256'], 'reservation policy changed since audit')
    for path,sha in proof['files'].items():
        require(file_digest(path)==sha, 'audited dependency changed: '+path)
    for name,sha in before['files'].items():
        require(file_digest(previous/name)==sha, 'previous release changed: '+name)
    # Copy only the authoritative inventory. Old release directories may also
    # retain superseded libraries that are deliberately absent from the release.
    destination.mkdir(mode=0o750)
    write_json_atomic(destination/'release_block.json',{'reason':'measured-height dependency audit in progress'})
    for name in before['files']:
        target=destination/name
        require(target.resolve().is_relative_to(destination), 'unsafe release manifest path')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(previous/name,target)
    anchors=json.loads((previous/'anchors.json').read_text())
    by_key={a['key']:a for a in anchors['anchors']}
    files=dict(before['files']);routes={};maps={}
    for row in proof['checked']:
        entry=row['entry_id'];library=destination/'glib/models'/entry/'library.json'
        original=json.loads(Path(row['library']['path']).read_text())
        require(library.is_file(), 'exact scale library absent from frozen base: '+entry)
        current=json.loads(library.read_text())
        require(physical_library(current)==physical_library(original), 'scaled physical library differs from map source')
        alias='campaign_train__'+entry
        updated={**current,'legacy_keys':list(dict.fromkeys([*(current.get('legacy_keys')or[]),alias]))}
        require(physical_library(updated)==physical_library(original), 'routing changes physical data')
        write_json_atomic(library,updated)
        files[str(library.relative_to(destination))]=file_digest(library)
        routes[entry]={'alias':alias,'library':str(library),'library_sha256':file_digest(library),
                       'source_library_sha256':row['library']['sha256'],'physical_payload_sha256':digest(physical_library(original)),
                       'scale':row['target_scale']}
        key=row['actual_height_key'];height=float(key)
        if key not in by_key:
            by_key[key]={'key':key,'height':height,'precision':.0001,'surfaces':[], 'status':['measured_native_dependency']}
        require(abs(by_key[key]['height']-height)<1e-8, 'height anchor differs')
        surface=row['support_binding']+'/slot_'+str(row['support_height_slot'])
        if surface not in by_key[key]['surfaces']:by_key[key]['surfaces'].append(surface)
        for mp in row['maps']:
            for src in (mp,mp['npz']):
                relative=Path('maps')/entry/Path(src['path']).name;target=destination/relative
                if target.exists():require(file_digest(target)==src['sha256'],'conflicting map bytes')
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(src['path'],target)
                files[str(relative)]=src['sha256']
            maps[mp['sha256']]={**mp,'entry':entry,'path':str(destination/'maps'/entry/Path(mp['path']).name)}
    ordered=sorted(by_key.values(),key=lambda a:a['height']);maximum=anchors['max_approx_m']
    for i,a in enumerate(ordered):
        lo=(ordered[i-1]['height']+a['height'])/2 if i else float('-inf')
        hi=(ordered[i+1]['height']+a['height'])/2 if i+1<len(ordered) else float('inf')
        a['range']=[max(lo,a['height']-maximum),min(hi,a['height']+maximum)]
    anchors['anchors']=ordered;write_json_atomic(destination/'anchors.json',anchors)
    files['anchors.json']=file_digest(destination/'anchors.json');aset=AnchorSet(ordered,maximum)
    os.environ['ISPATIALGYM_GRASP_LIBRARY_V2']=str(destination/'glib');G._v2_index.cache_clear()
    for entry,row in routes.items():
        lib=G.v2_entry(row['alias'])
        require(lib is not None and Path(lib.path)==Path(row['library']), 'native scale alias resolved incorrectly')
        S.check_compatible(lib,scale=row['scale'])
        require(G.grasps_from_v2(lib,row['alias']), 'native sampler sees no scaled grasps')
        for mp in maps.values():
            if mp['entry']!=entry:continue
            resolved=RM.ReachMap.find(str(destination/'maps'/entry),lib,mp['height'],rest_id=mp['rest_id'],arm=mp['arm'],anchors=aset)
            anchor,how=aset.lookup(mp['height'])
            require(resolved is not None and file_digest(resolved.path)==mp['sha256'] and how=='exact'
                    and abs(anchor['height']-mp['height'])<1e-8
                    and abs(resolved.meta['height']['value']-mp['height'])<1e-8,
                    'native merged map is not the audited exact-height dependency')
    report={'schema':'isg-training-height-release-v1','campaign_id':'campaign_20261110','production_admitted':False,
            'observed_at':datetime.now(timezone.utc).isoformat(),'audit':str(audit),'audit_sha256':file_digest(audit),
            'routes':routes,'maps':list(maps.values()),'native_reader_errors':0,'remaining':proof['remaining']}
    write_json_atomic(destination/'training_height_reader_audit.json',report)
    files['training_height_reader_audit.json']=file_digest(destination/'training_height_reader_audit.json')
    write_json_atomic(destination/'release_manifest.json',{**before,'created_at':report['observed_at'],
        'previous_release':str(previous),'new_measured_height_maps':len(maps),'production_admitted':False,'files':files})
    (destination/'release_block.json').unlink()
    for name,sha in before['files'].items():require(file_digest(previous/name)==sha,'source release mutated')
    print(json.dumps({'release':str(destination),'scaled_models':len(routes),'maps':len(maps),'production_admitted':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','previous','audit','destination'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();merge(a.root,a.previous,a.audit,a.destination)
