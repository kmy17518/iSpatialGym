"""Admit complete native attachment specs and positive exact-identity maps."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','batch','glib','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); assert not a.output.exists() and a.output.resolve().is_relative_to(a.root.resolve())
    from ispatialgym.grasping import schema as S, contact_policy as CP
    from ispatialgym.sampling import attach_maps as AM, grasp_library as G, reach_maps as RM
    from ispatialgym.sampling.generation_defaults import defaults_for
    admitted=[];blocked=[];files={}
    def add(path,relative):
        path=Path(path);assert path.resolve().is_relative_to(a.root.resolve()) and path.is_file()
        row=dict(source=str(path),destination=relative,sha256=sha(path),bytes=path.stat().st_size)
        assert relative not in files or files[relative]==row
        files[relative]=row
    for index in range(4):
        cls='configuration_matching-attachment_'+str(index+1).zfill(2)
        folder=a.batch/cls;manifest=a.batch/(str(index)+'_pairs.json');pair=json.loads(manifest.read_text())[0]
        row=dict(class_id=cls,entry=pair['male_entry'],partner=pair['partner'])
        try:
            logs=list((folder/'logs').glob('build_right_*.log'));assert len(logs)==1
            log=logs[0];text=log.read_text();assert 'ATTACH_DONE' in text and 'Simulation App Shutting Down' in text
            specs=[json.loads(line[len('ATTACH_SPEC '):]) for line in text.splitlines() if line.startswith('ATTACH_SPEC ')]
            maps=[json.loads(line[len('ATTACH_MAP '):]) for line in text.splitlines() if line.startswith('ATTACH_MAP ')]
            assert len(specs)==1 and specs[0]['ok'] and len(maps)==2 and all(x['ok'] for x in maps)
            spec=AM.load_spec(specs[0]['path']);assert spec.digest()==specs[0]['digest'] and spec.data['id']==pair['id']
            for k in ('category','model','scale'):assert spec.data['partner'][k]==pair['partner'][k]
            lp=a.glib/'models'/pair['male_entry']/'library.json';ld=json.loads(lp.read_text())
            lib=S.load_model(ld['category'],ld['model'],ld['scale'],str(a.glib),constraint=ld['constraint'],rest_hint_wxyz=ld.get('rest_hint_wxyz'))
            assert lib and Path(lib.path).resolve()==lp.resolve() and len(lib.grasps())>0
            positive=[]
            for rec in maps:
                assert rec['h']=='0.7588' and rec['arm']=='right'
                partner=pair['partner'];params=AM.oracle_hover_params(rec['rule'])
                mapped=AM.AttachMap.find(str(folder/'maps'/pair['male_entry']),.7588,
                    partner=(partner['category'],partner['model'],partner['scale']),rest_id=G.v2_rest_id(lib),
                    object_key=lib.identity_key(),arm='right',hover=params,
                    pre_back_m=defaults_for('configuration_matching-attachment')['pre_attach_back_m'],
                    policy_key=CP.DEFAULT.key('right'),robot=RM.robot_digest())
                assert mapped and Path(mapped.path)==Path(rec['path']) and mapped.meta['key']==rec['key']
                assert mapped.meta['spec']['digest']==spec.digest() and mapped.meta['male']['entry']==pair['male_entry']
                assert np.isfinite(mapped.cells).all() and np.isfinite(mapped.yaws).all()
                values=[v for k,v in mapped.arrays.items() if k.startswith('s')]
                assert values and all(set(np.unique(v))<=set(AM.STATUS.values()) for v in values)
                count=sum(int((v==AM.STATUS['feasible']).sum()) for v in values);assert count>0,'zero feasible native attachment cells'
                assert count==sum(sum(t.get('n_feasible',{}).values()) for t in mapped.meta['trunks'])
                path=Path(mapped.path);relative='maps/'+pair['male_entry']+'/'+path.name
                add(path,relative);add(path.with_suffix('.npz'),relative[:-5]+'.npz')
                positive.append(dict(rule=rec['rule'],path=str(path),sha256=sha(path),positive_cells=count,rest_id=G.v2_rest_id(lib),object_key=lib.identity_key()))
            for path in (Path(specs[0]['path']),Path(specs[0]['path']).with_name('partner_geometry.json'),log,manifest):
                add(path,'evidence/attachment/'+cls+'/'+path.name)
            admitted.append({**row,'status':'native_attachment_maps_admitted','maps':positive,'library_sha256':sha(lp),
                'spec_digest':spec.digest(),'spec_path':specs[0]['path'],'spec_sha256':sha(specs[0]['path'])})
        except (AssertionError,OSError,ValueError,KeyError) as exc:
            blocked.append({**row,'reason':str(exc),'status':'blocked_or_incomplete'})
    result=dict(schema='campaign-attachment-transfer-v1',admitted=admitted,blocked=blocked,files=list(files.values()),
        native_reader_errors=0,production_released=False,task_geometry_demo_replay_still_required=True)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(admitted=len(admitted),blocked=blocked,files=len(files),output=str(a.output),sha256=sha(a.output))))


if __name__=='__main__':main()
