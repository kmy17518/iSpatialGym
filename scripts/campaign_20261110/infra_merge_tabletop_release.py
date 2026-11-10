"""Publish an exact-copy tabletop dependency layer after both native readers pass."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge(root, previous, inputs, destination):
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling import grasp_library as G, reach_maps as R
    from ispatialgym.sampling.height_anchors import AnchorSet
    root, previous, inputs, destination = [Path(p).resolve() for p in (root,previous,inputs,destination)]
    assert all(p.is_relative_to(root) for p in (previous,inputs,destination)) and not destination.exists()
    os.umask(0o027); destination.mkdir(mode=0o750)
    (destination/'release_block.json').write_text('{"reason":"native reader verification pending"}\n')
    before = json.loads((previous/'release_manifest.json').read_text())
    transfer = json.loads((inputs/'transfer_manifest.json').read_text())
    transfer_key = sha(inputs/'transfer_manifest.json')[:16]
    copied = {}
    def copy(path, relative, expected):
        path = Path(path); target = destination/relative
        assert target.resolve().is_relative_to(destination)
        data = path.read_bytes(); assert hashlib.sha256(data).hexdigest() == expected, 'source content changed: '+str(path)
        if target.exists(): assert sha(target) == expected, 'conflicting dependency path: '+relative
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream: stream.write(data)
        copied[relative] = expected
    for relative, expected in before['files'].items(): copy(previous/relative,relative,expected)
    for row in transfer['files']:
        relative = row['destination']
        if relative.startswith('evidence/'):
            relative = 'evidence/' + transfer_key + '/' + relative.removeprefix('evidence/')
        copy(inputs/row['destination'],relative,row['sha256'])
    transfer_audit = 'tabletop_transfer_audit_' + transfer_key + '.json'
    copy(inputs/'transfer_manifest.json',transfer_audit,sha(inputs/'transfer_manifest.json'))
    anchors = json.loads((destination/'anchors.json').read_text())
    by_key = {a['key']:a for a in anchors['anchors']}
    for row in transfer['maps']:
        height = float(row['height']); key = f'{height:.4f}'
        if key not in by_key:
            by_key[key] = {'key':key,'height':height,'precision':.0001,'surfaces':[],
                          'status':['measured_native_dependency'],'source_map_sha256':row['map_sha256']}
        assert abs(by_key[key]['height']-height)<1e-8
    ordered = sorted(by_key.values(),key=lambda x:x['height']); maximum = anchors['max_approx_m']
    for i,a in enumerate(ordered):
        lo=(ordered[i-1]['height']+a['height'])/2 if i else float('-inf')
        hi=(ordered[i+1]['height']+a['height'])/2 if i+1<len(ordered) else float('inf')
        a['range']=[max(lo,a['height']-maximum),min(hi,a['height']+maximum)]
    anchors['anchors']=ordered;(destination/'anchors.json').write_text(json.dumps(anchors,indent=2)+'\n')
    copied['anchors.json']=sha(destination/'anchors.json');aset=AnchorSet(ordered,maximum)
    os.environ['ISPATIALGYM_GRASP_LIBRARY_V2']=str(destination/'glib');G._v2_index.cache_clear()
    admission=json.loads((inputs/'evidence/library_admission.json').read_text())
    sources={row['id']:row for row in admission['admitted']}
    checked=[]
    for row in transfer['maps']:
        entry=sources[row['entry']];lib=G.v2_entry(entry['alias'])
        assert lib and Path(lib.path).parent.name==row['entry'] and sha(lib.path)==entry['library_sha256']
        S.check_compatible(lib,scale=entry['scale'])
        assert len(lib.grasps())==entry['positive_grasps'] and G.grasps_from_v2(lib,entry['alias'])
        anchor,kind=aset.lookup(row['height']);assert kind=='exact' and abs(anchor['height']-row['height'])<1e-8
        native=R.ReachMap.find(str(destination/'maps'/row['entry']),lib,row['height'],
                              rest_id=row['rest_id'],arm=row['arm'],anchors=aset)
        assert native and sha(native.path)==row['map_sha256'] and native.meta['object_key']==row['object_key']
        checked.append({'entry':row['entry'],'arm':row['arm'],'height':row['height'],'map':native.path})
    audit={'native_reader_errors':0,'libraries':transfer['libraries'],'positive_maps':len(checked),'checked':checked,
           'source_transfer_sha256':sha(inputs/'transfer_manifest.json'),'source_permissions_unchanged':True,
           'task_geometry_demo_replay_still_required':True}
    audit_name = 'tabletop_reader_audit_' + transfer_key + '.json'
    (destination/audit_name).write_text(json.dumps(audit,indent=2)+'\n')
    copied[audit_name]=sha(destination/audit_name)
    manifest={**before,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'previous_release':str(previous),'new_tabletop_libraries':transfer['libraries'],
              'new_tabletop_maps':len(checked),'new_tabletop_reader_audit':audit_name,
              'new_tabletop_transfer_audit':transfer_audit,'files':copied}
    (destination/'release_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (destination/'release_block.json').unlink()
    (root/'validation'/(destination.name+'_reader.txt')).write_text('Native library/alias/exact scale/anchor/map reader\nERRORS 0\n')
    print(json.dumps({'release':str(destination),'libraries':transfer['libraries'],'maps':len(checked),
                      'manifest_sha256':sha(destination/'release_manifest.json')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','previous','inputs','destination'):p.add_argument('--'+name,required=True)
    a=p.parse_args();merge(a.root,a.previous,a.inputs,a.destination)
