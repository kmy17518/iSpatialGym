import json
import numpy as np
import pytest
from ispatialgym.sampling.interior_shards import select, merge, require_complete

GRID = [[0.5, 0.0], [1.0, 0.0], [1.5, 0.0]]


def shards(tmp_path):
    paths=[]
    for index in range(2):
        selected, proof=select(GRID,f'{index}/2')
        trunks=[{'translate':t,'tilt':k,'key':f't{t:.2f}_k{k:+.1f}','eligible':True}for t,k in selected]
        meta={'schema':'test','kind':'interior','exhaustive':False,'interior_trunk_shard':proof,'trunks':trunks,
              'edge_distances':[.5],'grasps':[{'name':'g'}],'status_codes':{'feasible':1,'off_support':2},
              'config':{'seeds':12,'lattice_m':.04},'height':{'value':.8246},'object_key':'unchanged',
              'joint_margin':{'rad':.05,'pre_seeds_dropped':index+1}}
        arrays={'cells':np.array([[.6,.1],[.7,.2]],dtype=np.float32),'yaws':np.array([0.]),
                'release_lift':np.array([.03]),'release_z':np.array([.9]),'tilts':np.array([0.])}
        for trunk in trunks:
            arrays[f"s0.50_{trunk['key']}"]=np.array([[[1]],[[2]]],dtype=np.uint8)
        path=tmp_path/f'{index}.json';path.write_text(json.dumps(meta));np.savez(path.with_suffix('.npz'),**arrays);paths.append(path)
    return paths


def test_merge_recovers_every_original_array_and_keeps_hashes(tmp_path):
    paths=shards(tmp_path);meta,arrays=merge(paths,GRID)
    assert meta['exhaustive'] is True and 'interior_trunk_shard'not in meta
    assert [[x['translate'],x['tilt']]for x in meta['trunks']]==GRID
    assert meta['joint_margin']['pre_seeds_dropped']==3
    assert len(meta['trunk_shard_provenance'])==2
    for t,k in GRID:
        np.testing.assert_array_equal(arrays[f's0.50_t{t:.2f}_k{k:+.1f}'],np.array([[[1]],[[2]]]))
    require_complete(meta)


@pytest.mark.parametrize('mutation',['missing','duplicate','config','grid','wrong_trunk','array_missing','lattice','unknown_status'])
def test_reject_incomplete_or_incompatible_map_join(tmp_path,mutation):
    paths=shards(tmp_path)
    if mutation=='missing':paths=paths[:1]
    elif mutation=='duplicate':paths=[paths[0],paths[0]]
    elif mutation in ('config','grid','wrong_trunk'):
        d=json.loads(paths[1].read_text())
        if mutation=='config':d['config']['lattice_m']=.08
        if mutation=='grid':d['interior_trunk_shard']['full_grid'][0][0]=.6
        if mutation=='wrong_trunk':d['trunks'][0]['translate']=.8
        paths[1].write_text(json.dumps(d))
    else:
        p=paths[1].with_suffix('.npz')
        with np.load(p)as z:a={k:z[k]for k in z.files}
        key=next(k for k in a if k.startswith('s0.'))
        if mutation=='array_missing':del a[key]
        if mutation=='lattice':a['cells'][0,0]+=.01
        if mutation=='unknown_status':a[key][0,0,0]=255
        np.savez(p,**a)
    with pytest.raises(ValueError):merge(paths,GRID)


def test_partial_reader_gate_and_unsharded_default(tmp_path):
    p=shards(tmp_path)[0]
    with pytest.raises(ValueError,match='partial'):require_complete(json.loads(p.read_text()))
    assert select(GRID,None)==([tuple(x)for x in GRID],None)
    for value in ('0/0','2/2','0/4','-1/2','x/2'):
        with pytest.raises(ValueError):select(GRID,value)
