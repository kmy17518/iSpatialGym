"""Join measured scaled-support probes to exact positive scaled grasp libraries.

Produces real reach-map inputs; never authorizes collection or declares a layout
valid. Failed geometry and missing dependencies remain explicit report rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from scripts.campaign_20261110.reservation_policy import active_policy


def read(path):
    return json.loads(Path(path).read_text())


def evidence(path):
    return dict(path=str(Path(path).resolve()), sha256=file_digest(path))


def measured_probe(path, profiles, profiles_sha, root):
    probe = read(path)
    require(probe.get('schema') == 'isg-support-height-probe-v1', 'unknown probe schema')
    require(probe.get('geometry_admitted') is True, 'support geometry did not pass')
    require(probe['profile_manifest_sha256'] == profiles_sha, 'probe belongs to another support profile release')
    binding = profiles['bindings'][probe['binding_id']]
    profile = profiles['profiles'][binding['profile_id']]
    slot = probe['slot']
    require(type(slot) is int and slot in (0, 1), 'invalid support height slot')
    scale, default = profile['absolute_scales'][slot], profile['default_scale']
    require(profile.get('ready') is True, 'support is not ready')
    require(probe['profile_id'] == binding['profile_id'], 'support profile differs')
    require((probe['scene'], probe['object']) == (binding['scene'], binding['name']), 'support binding differs')
    require(probe['absolute_scale'] == scale and scale[:2] == default[:2], 'support scale differs or changes X/Y')
    require(.8 - 1e-7 <= scale[2] / default[2] <= 1.2 + 1e-7, 'support height outside requested range')
    require(not probe.get('conservative_overlap_blockers'), 'support collision blockers remain')
    height = probe['measured_height_above_floor']
    require(math.isfinite(height) and height > 0, 'invalid measured support height')
    require(abs(height - (probe['measured_height_world'] - probe['floor_z_world'])) < 1e-7,
            'height is not measured relative to the floor')
    require(probe['actual_height_key'] == f'{height:.4f}', 'height key differs from measurement')
    require(len(probe.get('surface_hits', [])) >= 3, 'surface measurement missing')
    scene_path = Path(probe['stable_scene_path']).resolve()
    require(scene_path.is_relative_to(Path(root).resolve()), 'scaled scene outside campaign')
    require(file_digest(scene_path) == probe['stable_scene_sha256'], 'scaled scene changed since probe')
    scene = read(scene_path)
    actual = scene['objects_info']['init_info'][binding['name']]['args']
    require(actual['scale'] == scale and actual['category'] == profile['category']
            and actual['model'] == profile['model'], 'scaled scene support identity differs')
    return probe, binding, profile


def positive_library(job, path, reservations):
    key = job['category'] + '/' + job['model']
    require(job.get('ready') is True and job.get('readiness_source') and job.get('asset_sha256'),
            'target readiness or asset provenance missing')
    require(key not in reservations['train_exclude_models'] and
            job['category'] not in reservations['category_holdout_categories'], 'target reserved from training')
    library = read(path)
    require(library.get('schema') == 'isg-grasp-library/2', 'not a validated v2 library')
    require(all(library[k] == job[k] for k in ('category', 'model', 'scale', 'constraint')),
            'grasp library does not match the exact target scale/constraint')
    require(library.get('provenance', {}).get('validator') == 'isg-tabletop-validator/1',
            'tabletop physics validator provenance missing')
    require(library.get('physics_profile') and library.get('policy_key'), 'physics/contact policy missing')
    require(library.get('grasps') and all(g.get('physics') == 'tested' for g in library['grasps']),
            'no positive physics-tested grasp library')
    require(all(g['rest_id'] in library.get('rests', {}) for g in library['grasps']), 'measured stable rest missing')
    return library


def selected_libraries(path):
    """Consume exact positive native-reader rows, preserving each source manifest."""
    report = read(path)
    require(report.get('schema') == 'isg-native-library-harvest-v1', 'unknown selected-library audit')
    jobs, libraries = {}, {}
    for row in report['verified']:
        job = row['job']
        if job['constraint'] != 'tabletop':
            continue
        require(row.get('status') == 'verified' and row.get('native_reader_passed') is True,
                'selected library lacks a positive native reader audit')
        source = row['source_manifest']
        require(file_digest(source['path']) == source['sha256'], 'selected source manifest changed')
        require(read(source['path'])[row['source_index']] == job, 'selected source job differs')
        library = row['library']
        require(file_digest(library['path']) == library['sha256'], 'audited positive library changed')
        require(job['id'] not in jobs or (jobs[job['id']] == job and libraries[job['id']] == library),
                'conflicting physical library identity')
        jobs[job['id']], libraries[job['id']] = job, library
    return list(jobs.values()), libraries


def build(root, *, only_classes=(), only_bindings=(), selected_library_audit=None):
    root = Path(root).resolve()
    profile_path = root/'manifests/support_height_profiles.json'
    manifest_path = root/'manifests/train_scale_glib_jobs.json'
    profiles = read(profile_path); profile_sha = file_digest(profile_path)
    pointer, reservations, _ = active_policy(root) or (None, None, None)
    require(pointer is not None, 'active holdout reservation policy unavailable')
    if selected_library_audit is None:
        jobs, library_paths = read(manifest_path), {}
    else:
        manifest_path = Path(selected_library_audit).resolve()
        require(manifest_path.is_relative_to(root), 'selected audit outside campaign')
        jobs, library_paths = selected_libraries(manifest_path)
    probes, failures, requests = {}, [], []
    # A later successful re-probe supersedes an earlier rejected attempt, while
    # both remain in the evidence report. No analytic prediction enters a map.
    paths = [*root.glob('height_probes/*/height_probe.json'), *root.glob('height_probes_v2/*/height_probe.json')]
    for path in sorted(paths, key=lambda p: p.stat().st_mtime):
        try:
            probe, binding, profile = measured_probe(path, profiles, profile_sha, root)
            key = profile['category']+'/'+profile['model']
            require(key not in reservations['train_exclude_models'] and
                    profile['category'] not in reservations['category_holdout_categories'], 'support reserved from training')
            if only_bindings and probe['binding_id'] not in only_bindings: continue
            probes[(probe['binding_id'], probe['slot'])] = (path, probe, binding)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(dict(probe=str(path), reason=str(exc)))
    selected, by_id, anchors, surfaces = {}, {}, {}, []
    for job in jobs:
        if only_classes and not set(job['campaign_classes']).intersection(only_classes): continue
        library_path = Path(library_paths[job['id']]['path']) if job['id'] in library_paths else \
            root/'scale_prebuild/glib/models'/job['id']/'library.json'
        try:
            library = positive_library(job, library_path, reservations)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(dict(entry_id=job['id'], reason=str(exc))); continue
        for (binding_id, slot), (path, probe, binding) in sorted(probes.items()):
            relevant = set(job['campaign_classes']).intersection(binding['classes'])
            if not relevant and not binding.get('shared_anchor'): continue
            height = probe['actual_height_key']
            selected[job['id']] = job
            by_id.setdefault(job['id'], set()).add(height)
            surface_id = f"{probe['scene']}/{probe['object']}/height_slot_{slot}/{binding_id}"
            anchor = anchors.setdefault(height, dict(key=height, height=float(height), precision=.0001,
                range=[float(height)-.00005, float(height)+.00005], surfaces=[], status=['measured']))
            if surface_id not in anchor['surfaces']:
                anchor['surfaces'].append(surface_id)
                surfaces.append(dict(id=surface_id, key=height, height=probe['measured_height_above_floor'],
                    scene=probe['scene'], support=probe['object'], surface='top', precision=.0001,
                    source=str(path), floor_z=probe['floor_z_world'], status='measured'))
            requests.append(dict(entry_id=job['id'], target_scale=job['scale'], target_scale_slot=job['campaign_scale_slot'],
                support_binding=binding_id, support_height_slot=slot, support_scale=probe['absolute_scale'],
                actual_height_key=height, classes=sorted(relevant or set(job['campaign_classes'])),
                shared_support=bool(binding.get('shared_anchor')), probe=evidence(path),
                stable_scene=dict(path=probe['stable_scene_path'], sha256=probe['stable_scene_sha256']),
                library=evidence(library_path), positive_grasps=len(library['grasps']), arms=['right','left'],
                status='ready_for_reach_map_generation', production_admitted=False,
                remaining=['positive exact-height reach maps','maximum scaled layout fit','native demo',
                    'open-loop replay','perturbation restoration','acceptance and distribution audit']))
    return dict(campaign_id='campaign_20261110', observed_at=datetime.now(timezone.utc).isoformat(),
        schema='isg-measured-height-map-dependencies-v1', production_admitted=False,
        inputs=[evidence(profile_path), evidence(manifest_path)], reservation_policy=pointer,
        measured_support_slots=len(probes), requests=requests, failures=failures,
        jobs=[selected[k] for k in sorted(selected)], heights_by_id={k:sorted(v) for k,v in by_id.items()},
        anchors=dict(schema='isg-height-anchors/1', max_approx_m=.00005,
            anchors=sorted(anchors.values(), key=lambda a:a['height']), surfaces=surfaces))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--class-id',action='append',default=[])
    p.add_argument('--binding',action='append',default=[])
    p.add_argument('--selected-library-audit',type=Path)
    a=p.parse_args()
    require(a.output.resolve().is_relative_to(a.root.resolve()), 'dependency output outside campaign')
    require(not a.output.exists(), 'immutable dependency release already exists')
    value=build(a.root, only_classes=a.class_id, only_bindings=a.binding,
                selected_library_audit=a.selected_library_audit)
    require(value['jobs'], 'no verified scaled target/support dependencies')
    a.output.mkdir(parents=True)
    for name,key in [('jobs.json','jobs'),('heights_by_id.json','heights_by_id'),('anchors.json','anchors')]:
        write_json_atomic(a.output/name,value[key])
    write_json_atomic(a.output/'dependencies.json',value)
    print(json.dumps(dict(jobs=len(value['jobs']), requests=len(value['requests']),
        measured_support_slots=value['measured_support_slots'], failures=len(value['failures']), production_admitted=False)))


if __name__=='__main__':main()
