"""Reuse exact admitted target libraries for newly measured tray-floor heights."""
import argparse
import copy
import json
import os
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from scripts.campaign_20261110.reservation_policy import active_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'release', 'jobs', 'height-proof', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--entry', action='append')
    args = parser.parse_args()
    root, release, out = (path.resolve() for path in (args.root, args.release, args.out))
    require(not out.exists() and all(path.is_relative_to(root) for path in (release, out)), 'new owned output required')
    require(not (release / 'release_block.json').exists(), 'dependency release is blocked')
    manifest_path = release / 'release_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    pointer, policy, assets = active_policy(root)
    proof = json.loads(args.height_proof.read_text())
    require(proof['status'] == 'native_tray_floor_measurement_passed', 'native tray-floor proof missing')
    heights = sorted({row['map_height_key'] for row in proof['objects'].values()})
    require(heights and all(row['map_height_key'] == f"{row['map_height_above_floor']:.4f}"
                           for row in proof['objects'].values()), 'measured height identity differs')
    jobs = json.loads(args.jobs.read_text())
    require(isinstance(jobs, list), 'tabletop job list required')
    if args.entry:
        wanted = set(args.entry)
        jobs = [job for job in jobs if job['id'] in wanted]
        require({job['id'] for job in jobs} == wanted, 'selected library missing from source manifest')
    require(jobs and len({job['id'] for job in jobs}) == len(jobs), 'unique target jobs required')
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling import grasp_library as G
    os.environ['ISPATIALGYM_GRASP_LIBRARY_V2'] = str(release / 'glib')
    G._v2_index.cache_clear()
    admitted = []
    for index, job in enumerate(jobs):
        key = job['category'] + '/' + job['model']
        require(job['ready'] and key in policy['train_exclude_models'], 'target is not ready and reserved')
        expected = assets['ready_assets'][key]
        require(expected['ready'] and file_digest(expected['path']) == expected['asset_sha256'] == job['asset_sha256'],
                'target readiness or asset bytes changed')
        lib = S.load_model(job['category'], job['model'], job['scale'], str(release / 'glib'),
                           constraint=job['constraint'], rest_hint_wxyz=job.get('rest_hint_wxyz'))
        require(lib and len(lib.grasps()) > 0 and Path(lib.path).parent.name == job['id'], 'exact positive native library missing')
        S.check_compatible(lib, scale=job['scale'], rest_kind='stable')
        require(lib.data['legacy_key'] == job['legacy_key'] and lib.data['dataset'] == job['dataset'], 'library routing identity differs')
        routed = G.v2_entry(job['legacy_key'])
        require(routed and Path(routed.path).resolve() == Path(lib.path).resolve()
                and G.grasps_from_v2(routed, job['legacy_key']), 'native alias reader differs')
        for child in Path(lib.path).parent.iterdir():
            require(child.is_file() and not child.is_symlink(), 'unexpected library member')
            relative = str(child.relative_to(release))
            require(relative in manifest['files'] and file_digest(child) == manifest['files'][relative],
                    'published dependency member changed')
        admitted.append(dict(id=job['id'], source=str(Path(lib.path).parent), source_index=index,
                             source_manifest=str(args.jobs), source_manifest_sha256=file_digest(args.jobs),
                             library_sha256=file_digest(lib.path), positive_grasps=len(lib.grasps()),
                             scale=job['scale'], constraint=job['constraint'], alias=job['legacy_key'],
                             asset_sha256=job['asset_sha256'], rest_kind='stable', rest_ids=sorted(lib.rests),
                             height_keys=heights, arms=['right', 'left'],
                             prior_release=str(release), prior_release_manifest_sha256=file_digest(manifest_path)))
    anchors = json.loads((release / 'anchors.json').read_text())
    require(file_digest(release / 'anchors.json') == manifest['files']['anchors.json'], 'published anchors changed')
    by_key = {row['key']: copy.deepcopy(row) for row in anchors['anchors']}
    for height in heights:
        by_key.setdefault(height, dict(key=height, height=float(height), precision=.0001, surfaces=[],
                                      status=['measured_native_tray_floor'], proof_sha256=file_digest(args.height_proof)))
        require(abs(by_key[height]['height'] - float(height)) < 1e-8, 'anchor identity conflict')
    ordered = sorted(by_key.values(), key=lambda row: row['height'])
    maximum = anchors['max_approx_m']
    for index, row in enumerate(ordered):
        lower = (ordered[index-1]['height'] + row['height']) / 2 if index else float('-inf')
        upper = (ordered[index+1]['height'] + row['height']) / 2 if index+1 < len(ordered) else float('inf')
        row['range'] = [max(lower, row['height'] - maximum), min(upper, row['height'] + maximum)]
    anchors['anchors'] = ordered
    out.mkdir(parents=True)
    write_json_atomic(out / 'jobs.json', jobs)
    write_json_atomic(out / 'heights_by_id.json', {job['id']: heights for job in jobs})
    write_json_atomic(out / 'anchors.json', anchors)
    report = dict(schema='campaign-reused-tabletop-library-admission-v1', active_policy_sha256=pointer['policy_sha256'],
                  admitted=admitted, height_proof=dict(path=str(args.height_proof), sha256=file_digest(args.height_proof)),
                  fresh_maps_task_geometry_demo_replay_still_required=True, production_released=False)
    write_json_atomic(out / 'library_admission.json', report)
    print(json.dumps(dict(libraries=len(admitted), heights=heights, output=str(out),
                          admission_sha256=file_digest(out / 'library_admission.json'))))


if __name__ == '__main__':
    main()
