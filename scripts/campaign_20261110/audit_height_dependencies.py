"""Audit executed exact-scale / measured-height maps with the native resolver.

This prepares an immutable dependency inventory. It never admits collection;
support-layout validation, actual perturbation replay and acceptance are separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from scripts.campaign_20261110.height_dependency_jobs import evidence, measured_probe, positive_library, read
from scripts.campaign_20261110.reservation_policy import active_policy


def map_identity(meta):
    return meta['object_key'], meta['rest_id'], meta['arm'], f"{meta['height']['value']:.4f}"


def audit(root, layers, code):
    from scripts.campaign_20261110.bind_fresh_candidates import asset_path
    from ispatialgym.sampling.knowledge import is_ready
    from ispatialgym.grasping import schema as S
    from ispatialgym.grasping import contact_policy as CP
    from ispatialgym.sampling import reach_maps as RM

    root = Path(root).resolve()
    pointer, reservations, hashes = active_policy(root)
    profiles_path = root/'manifests/support_height_profiles.json'
    profiles, profiles_sha = read(profiles_path), file_digest(profiles_path)
    manifest = root/'manifests/train_scale_glib_jobs.json'
    jobs = {j['id']: j for j in read(manifest)}
    requests, candidates, logs = {}, {}, []
    for layer in map(Path, layers):
        layer = layer.resolve()
        require(layer.is_relative_to(root), 'map layer outside campaign')
        for req in read(layer/'dependencies.json')['requests']:
            key = (req['entry_id'], req['support_binding'], req['support_height_slot'])
            require(key not in requests or requests[key] == req, 'conflicting measured dependency request')
            requests[key] = req
        for p in sorted((layer/'maps').glob('*/*.json')):
            require(p.resolve().is_relative_to(root), 'map path outside campaign')
            candidates.setdefault(p.parent.name, []).append(p)
        for p in sorted((layer/'logs').glob('maps_*.log')):
            logs.append({**evidence(p), 'cuda_oom': 'CUDA out of memory' in p.read_text(errors='replace')})
    checked, failures, files, loaded = [], [], {}, {}

    def record(path):
        path = Path(path).resolve()
        require(path.is_relative_to(root), 'dependency artifact outside campaign')
        files[str(path)] = file_digest(path)

    for key, req in sorted(requests.items()):
        entry, binding_id, slot = key
        try:
            for name in ('probe', 'stable_scene', 'library'):
                require(file_digest(req[name]['path']) == req[name]['sha256'], name+' evidence changed')
                record(req[name]['path'])
            probe, binding, profile = measured_probe(req['probe']['path'], profiles, profiles_sha, root)
            require((probe['binding_id'], probe['slot'], probe['actual_height_key']) ==
                    (binding_id, slot, req['actual_height_key']), 'request/probe identity differs')
            support_key = profile['category']+'/'+profile['model']
            require(support_key not in reservations['train_exclude_models'] and
                    profile['category'] not in reservations['category_holdout_categories'], 'support reserved from training')
            job = jobs[entry]
            data = positive_library(job, req['library']['path'], reservations)
            for obj in (job, profile):
                asset = asset_path(Path(code), {'category':obj['category'], 'model':obj['model'], 'dataset_name':obj.get('dataset','behavior-1k-assets')})
                require(is_ready(obj['model']), 'runtime readiness check failed')
                require(file_digest(asset) == obj['asset_sha256'], 'ready asset bytes changed')
            if entry not in loaded:
                libpath = Path(req['library']['path'])
                lib = S.load_model(job['category'], job['model'], job['scale'],
                                   str(libpath.parent.parent.parent), constraint=job['constraint'])
                require(lib is not None and Path(lib.path).resolve() == libpath.resolve(), 'native library resolver differs')
                require(len(lib.grasps()) == len(data['grasps']) > 0, 'native positive grasp count differs')
                loaded[entry] = lib
            lib = loaded[entry]
            rests = {g['rest_id'] for g in data['grasps']}
            expected = {(lib.identity_key(), rest, arm, req['actual_height_key']) for rest in rests for arm in req['arms']}
            found = {}
            for path in candidates.get(entry, []):
                meta = read(path)
                identity = map_identity(meta)
                if identity not in expected:
                    continue
                native = RM.ReachMap(str(path))
                require(meta['kind'] == 'tabletop' and meta['key'] == RM.map_key(meta), 'invalid native map key')
                require(meta['policy_key'] == CP.DEFAULT.key(meta['arm']) and meta['robot'] == RM.robot_digest(),
                        'map robot/contact policy differs')
                require(np.isfinite(native.cells).all() and np.isfinite(native.yaws).all(), 'nonfinite native map lattice')
                states = [v for k, v in native.arrays.items() if k.startswith('s')]
                require(states and all(set(np.unique(v)) <= set(RM.STATUS.values()) for v in states), 'invalid native map states')
                positive = sum(int((v == RM.STATUS['feasible']).sum()) for v in states)
                require(positive > 0, 'map has no feasible cells')
                resolved = RM.ReachMap.find(str(path.parent), lib, float(req['actual_height_key']),
                    rest_id=meta['rest_id'], arm=meta['arm'], policy_key=meta['policy_key'], robot=meta['robot'])
                require(resolved is not None and Path(resolved.path).resolve() == path.resolve(), 'native exact-height resolver differs')
                row = {**evidence(path), 'npz': evidence(path.with_suffix('.npz')), 'positive_cells': positive,
                       'rest_id': meta['rest_id'], 'arm': meta['arm'], 'height': meta['height']['value'],
                       'object_key': meta['object_key']}
                require(identity not in found or (found[identity]['sha256'], found[identity]['npz']['sha256']) ==
                        (row['sha256'], row['npz']['sha256']), 'conflicting exact map contents')
                found[identity] = row
                record(path); record(path.with_suffix('.npz'))
            require(set(found) == expected, 'missing positive exact-height maps: '+str(sorted(expected-set(found))))
            checked.append({**req, 'maps': [found[k] for k in sorted(found)],
                'status': 'native_dependencies_verified_layout_pending', 'production_admitted': False})
        except (ValueError, KeyError, OSError, TypeError, AssertionError) as exc:
            failures.append({'entry_id': entry, 'support_binding': binding_id, 'support_height_slot': slot,
                             'reason': str(exc), 'error_type': type(exc).__name__})
    return dict(schema='isg-measured-height-native-audit-v1', campaign_id='campaign_20261110',
        observed_at=datetime.now(timezone.utc).isoformat(), production_admitted=False,
        active_policy_sha256=pointer['policy_sha256'], source_profiles=evidence(profiles_path),
        source_scale_jobs=evidence(manifest), requests=len(requests), verified=len(checked),
        checked=checked, failures=failures, logs=logs, files=files,
        remaining=['maximum scaled layout fit', 'native template binding', 'demo and own physics replay',
                   'perturbation application and restoration', 'acceptance and distribution validation'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--layer',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    require(a.output.resolve().is_relative_to(a.root.resolve()) and not a.output.exists(), 'new campaign audit required')
    result=audit(a.root,a.layer,a.code)
    write_json_atomic(a.output,result)
    print(json.dumps({k: result[k] for k in ('requests','verified','failures','production_admitted')}))


if __name__=='__main__':main()
