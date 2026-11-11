"""Exact interior-map trunk partitioning and complete-grid assembly.

Partial maps are private dependency artifacts, never sampler inputs. Sharding
changes scheduling only: every original trunk, cell, yaw and grasp is retained.
"""
from __future__ import annotations
import copy
import hashlib
import json
import pathlib

ENV = 'ISPATIALGYM_INTERIOR_MAP_TRUNK_SHARD'


def select(grid, value):
    grid = [tuple(map(float, pair)) for pair in grid]
    if not value:
        return grid, None
    parts = value.split('/')
    if len(parts) != 2 or any(not p.isdigit() for p in parts):
        raise ValueError('interior trunk shard must be index/count')
    index, count = map(int, parts)
    if not 0 <= index < count <= len(grid):
        raise ValueError('interior trunk shard outside exact grid')
    selected = grid[index::count]
    return selected, {'schema': 1, 'index': index, 'count': count,
                      'full_grid': [list(x) for x in grid],
                      'selected_grid': [list(x) for x in selected]}


def require_complete(meta):
    if meta.get('interior_trunk_shard') is not None:
        raise ValueError('partial interior trunk map cannot serve sampling')


def merge(paths, expected_grid):
    """Return complete native map arrays only after an exhaustive disjoint join."""
    import numpy as np
    expected_grid = [list(map(float, x)) for x in expected_grid]
    rows = []
    for path in map(pathlib.Path, paths):
        meta = json.loads(path.read_text())
        shard = meta.get('interior_trunk_shard')
        if not shard or meta.get('exhaustive') is not False:
            raise ValueError('expected explicitly partial map')
        if shard.get('full_grid') != expected_grid:
            raise ValueError('shard full grid differs from native grid')
        chosen, proof = select(expected_grid, f"{shard['index']}/{shard['count']}")
        if shard != proof or [[t['translate'], t['tilt']] for t in meta['trunks']] != [list(x) for x in chosen]:
            raise ValueError('shard records do not match exact trunk partition')
        npz = path.with_suffix('.npz')
        with np.load(npz) as data:
            arrays = {k: data[k] for k in data.files}
        rows.append((path, meta, arrays, hashlib.sha256(path.read_bytes()).hexdigest(),
                     hashlib.sha256(npz.read_bytes()).hexdigest()))
    if not rows:
        raise ValueError('no shards')
    count = rows[0][1]['interior_trunk_shard']['count']
    if len(rows) != count or {r[1]['interior_trunk_shard']['index'] for r in rows} != set(range(count)):
        raise ValueError('missing or duplicate trunk shard')
    invariant = ('schema','kind','arm','object','object_key','rest_id','height','edge_distances','config','policy_key','policy','plane','ik','approach_check','screening','grasps','brace','status_codes','robot','interior','interior_placement','interior_place','release')
    base = rows[0][1]
    constants = ('cells','yaws','release_lift','release_z','tilts')
    combined = {k: rows[0][2][k] for k in constants}
    trunks = {}
    for _, meta, arrays, _, _ in rows:
        if (meta['interior_trunk_shard']['count'] != count
                or meta.get('joint_margin', {}).get('rad') != base.get('joint_margin', {}).get('rad')
                or any(meta.get(k) != base.get(k) for k in invariant)):
            raise ValueError('incompatible native shard inputs')
        if any(k not in arrays or not np.array_equal(arrays[k], combined[k]) for k in constants):
            raise ValueError('shard lattice or release arrays differ')
        wanted = set(constants)
        for trunk in meta['trunks']:
            pair = (trunk['translate'], trunk['tilt'])
            if pair in trunks:
                raise ValueError('duplicate trunk')
            trunks[pair] = trunk
            for distance in meta['edge_distances']:
                key = f"s{distance:.2f}_{trunk['key']}"
                wanted.add(key)
                shape = (len(arrays['cells']), len(arrays['yaws']), len(meta['grasps']))
                if key not in arrays or arrays[key].shape != shape or arrays[key].dtype != np.uint8:
                    raise ValueError('missing or invalid trunk status array')
                if not np.isin(arrays[key], list(meta['status_codes'].values())).all():
                    raise ValueError('unknown native status code')
                combined[key] = arrays[key]
        if set(arrays) != wanted:
            raise ValueError('unexpected or missing shard arrays')
    if set(trunks) != set(map(tuple, expected_grid)):
        raise ValueError('incomplete native trunk grid')
    merged = copy.deepcopy(base)
    merged.pop('interior_trunk_shard'); merged.pop('key', None)
    merged['exhaustive'] = True
    merged['trunks'] = [trunks[tuple(x)] for x in expected_grid]
    merged['timing'] = {'total_s': sum(float(m.get('timing',{}).get('total_s',0)) for _,m,*_ in rows),
                        'parallel_shards': count}
    merged['interior_sphere_hits'] = {'see_shard_provenance': True}
    if 'joint_margin' in merged:
        merged['joint_margin'] = {'rad': base['joint_margin']['rad'],
            **{k: sum(m.get('joint_margin',{}).get(k,0) for _,m,*_ in rows)
               for k in base['joint_margin'] if k != 'rad'}}
    merged['trunk_shard_provenance'] = [{'metadata': str(p), 'metadata_sha256': h,
        'arrays_sha256': nh, 'index': m['interior_trunk_shard']['index']} for p,m,_,h,nh in rows]
    return merged, combined
