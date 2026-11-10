"""Read-only native audit and exact-file manifest for new campaign tabletop maps."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(admission, layers, output, root='/vision/u/shiyuc/minyeong/campaign_20261110'):
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling import reach_maps as R
    owned_root = Path(root).resolve()
    assert owned_root.name == 'campaign_20261110', 'unexpected campaign source root'
    admission = Path(admission).resolve(); document = json.loads(admission.read_text())
    assert admission.is_relative_to(owned_root) and Path(output).resolve().is_relative_to(owned_root)
    sources = {row['id']:row for row in document['admitted']}
    libraries = {}; files = {}; maps = []; coverage = {}
    def add(path, relative):
        path = Path(path).resolve()
        assert path.is_relative_to(owned_root)
        record = {'source':str(path), 'destination':relative, 'sha256':sha(path), 'bytes':path.stat().st_size}
        assert relative not in files or files[relative] == record
        files[relative] = record
    for layer in map(Path,layers):
        for path in sorted((layer/'maps').glob('*/*.json')):
            entry = path.parent.name; source = sources[entry]
            if entry not in libraries:
                libpath = Path(source['source'])/'library.json'
                assert sha(libpath) == source['library_sha256']
                data = json.loads(libpath.read_text())
                lib = S.load_model(data['category'],data['model'],source['scale'],str(libpath.parent.parent.parent),constraint=source['constraint'])
                assert lib and Path(lib.path).resolve() == libpath.resolve() and len(lib.grasps()) == source['positive_grasps'] > 0
                assert data['legacy_key'] == source['alias']
                libraries[entry] = lib
                for child in sorted(libpath.parent.iterdir()):
                    assert child.is_file() and not child.is_symlink()
                    add(child,'glib/models/'+entry+'/'+child.name)
            lib = libraries[entry]; native = R.ReachMap(str(path)); meta = native.meta
            assert meta['kind'] == 'tabletop' and meta['key'] == R.map_key(meta)
            assert meta['object_key'] == lib.identity_key() and meta['rest_id'] in lib.rests
            assert np.isfinite(native.cells).all() and np.isfinite(native.yaws).all()
            status = [v for k,v in native.arrays.items() if k.startswith('s')]
            assert status and all(set(np.unique(v)).issubset(set(R.STATUS.values())) for v in status)
            positive = sum(int((v == R.STATUS['feasible']).sum()) for v in status)
            assert positive > 0, 'zero positive cells: '+str(path)
            height = float(meta['height']['value']); arm = meta['arm']
            found = R.ReachMap.find(str(path.parent),lib,height,rest_id=meta['rest_id'],arm=arm,
                                   policy_key=meta['policy_key'],robot=meta['robot'])
            assert found and Path(found.path).resolve() == path.resolve()
            coverage.setdefault((str(layer),entry,height),set()).add(arm)
            relative = 'maps/'+entry+'/'+path.name
            add(path,relative);add(path.with_suffix('.npz'),relative[:-5]+'.npz')
            maps.append({'entry':entry,'arm':arm,'height':height,'positive_cells':positive,'map_sha256':sha(path),
                         'source':str(path),'rest_id':meta['rest_id'],'object_key':meta['object_key']})
        for log in sorted((layer/'logs').glob('maps_*.log')):
            text = log.read_text(); assert 'GB_MAP_DONE' in text and 'Simulation App Shutting Down' in text
            add(log,'evidence/'+layer.name+'_'+log.name)
    assert maps and all(arms == {'left','right'} for arms in coverage.values()), 'both exact native arm maps required'
    add(admission,'evidence/library_admission.json')
    result = {'schema':'isg-exact-tabletop-transfer-v1','source_admission':str(admission),
              'source_admission_sha256':sha(admission),'active_policy_sha256':document['active_policy_sha256'],
              'libraries':len(libraries),'positive_maps':len(maps),'maps':maps,'files':list(files.values()),
              'native_reader_errors':0,'production_released':False,'task_geometry_demo_replay_still_required':True}
    output = Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream: json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'audit':str(output),'libraries':len(libraries),'positive_maps':len(maps),'files':len(files)}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--admission',required=True)
    p.add_argument('--layer',action='append',required=True);p.add_argument('--output',required=True)
    p.add_argument('--root',default='/vision/u/shiyuc/minyeong/campaign_20261110')
    a=p.parse_args();audit(a.admission,a.layer,a.output,a.root)
