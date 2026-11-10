#!/usr/bin/env python3
"""Measure native default loads of reserved room-family objects, read-only assets.

This proves actual scale and loadable geometry, not grasp/task acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-root', type=Path, required=True)
    p.add_argument('--reservations', type=Path, required=True)
    p.add_argument('--asset-hashes', type=Path, required=True)
    p.add_argument('--category-proposal', type=Path)
    p.add_argument('--candidate-manifest', type=Path)
    p.add_argument('--reserved-candidate-manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(a.campaign_root.resolve()):
        raise ValueError('new campaign output required')
    reservations = json.loads(a.reservations.read_text())
    hashes = json.loads(a.asset_hashes.read_text())
    if hashes['reservation_sha256'] != sha(a.reservations):
        raise ValueError('reservation identity mismatch')
    families = {'mental_rotation', 'object_scaling', 'camera_relocalization-obstructed'}
    rows = []
    proposals = reservations['model_holdout_templates']
    if sum(bool(x) for x in (a.category_proposal,a.candidate_manifest,a.reserved_candidate_manifest))>1:raise ValueError('one candidate scope at a time')
    if a.reserved_candidate_manifest:
        candidate_doc=json.loads(a.reserved_candidate_manifest.read_text())
        if candidate_doc['reservation_sha256']!=sha(a.reservations):raise ValueError('candidate reservation mismatch')
        for row in candidate_doc['objects']:
            key=row['category']+'/'+row['model']; evidence=hashes['ready_assets'][key]
            if key not in reservations['train_exclude_models'] or row['asset_sha256']!=evidence['asset_sha256']:
                raise ValueError('reserved candidate identity mismatch')
            if not row['ready'] or sha(Path(row['asset_path']))!=row['asset_sha256']:raise ValueError('unready or changed reserved candidate')
            if row.get('scene_default_scale') is not None:
                source=Path(row['scene_default_scale_source']['path'])
                if sha(source)!=row['scene_default_scale_source']['sha256']:raise ValueError('scene-scale source changed')
                actual=json.loads(source.read_text())['objects_info']['init_info'][row['scene_default_scale_source']['object']]['args']
                if actual['category']+'/'+actual['model']!=key or actual['scale']!=row['scene_default_scale']:raise ValueError('scene-scale source does not match candidate')
            rows.append(row)
        proposals=[];report_scope='reserved_supports'
    elif a.candidate_manifest:
        candidate_doc=json.loads(a.candidate_manifest.read_text())
        if candidate_doc['reservation_sha256']!=sha(a.reservations):raise ValueError('candidate reservation mismatch')
        for row in candidate_doc['objects']:
            key=row['category']+'/'+row['model']
            if key in reservations['train_exclude_models'] or row['category'] in reservations['train_exclude_categories'] or row['asset_sha256'] in hashes['reserved_asset_hashes']:
                raise ValueError('train candidate leaks reserved model/category/hash')
            if not row['ready'] or sha(Path(row['asset_path']))!=row['asset_sha256']:raise ValueError('unready or changed train candidate')
            rows.append(row)
        proposals=[];report_scope='train_extra_rooms'
    elif a.category_proposal:
        proposals=[]
        for template in json.loads(a.category_proposal.read_text())['templates']:
            if template['task'] not in families:continue
            target=template['target'];role='obstacles' if template['task']=='camera_relocalization-obstructed' else 'target'
            class_id=template['task']+'-'+target['category']+('_1' if role=='obstacles' else '')
            proposals.append({'class_id':class_id,'task':template['task'],'roles':[{**target,'role':role}]})
        report_scope='category_rooms'
    else:
        report_scope='model_rooms'
    for template in proposals:
        if template['task'] not in families:
            continue
        for role in template['roles']:
            key = role['category'] + '/' + role['model']
            evidence = hashes['ready_assets'][key]
            if not role['ready'] or not evidence['ready'] or sha(Path(evidence['path'])) != evidence['asset_sha256']:
                raise ValueError('unready or changed reserved object: ' + key)
            rows.append({**role, 'class_id': template['class_id'], 'task': template['task'],
                         'asset_sha256': evidence['asset_sha256'], 'asset_path': evidence['path']})
    expected_count=len(candidate_doc['objects']) if a.reserved_candidate_manifest else (64 if a.candidate_manifest else (3 if a.category_proposal else 29))
    if not rows or len(rows) != expected_count:
        raise ValueError('unexpected number of frozen room-family reservations')
    out.mkdir(parents=True)
    report = {'schema': 'campaign-native-defaults-v1', 'reservation_sha256': sha(a.reservations),
              'scope':report_scope,'status': 'started', 'objects': [], 'task_acceptance': 'pending geometry, demo and replay'}
    import numpy as np
    import torch as th
    from PIL import Image, ImageDraw
    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.objects import DatasetObject
    from omnigibson.utils.transform_utils import mat2quat
    try:
        # Match SimWorld: transition recipes are unrelated to task geometry and
        # would consume overlapping initial imports before they are positioned.
        with gm.unlocked():
            gm.ENABLE_OBJECT_STATES = True
            gm.USE_GPU_DYNAMICS = False
            gm.ENABLE_TRANSITION_RULES = False
        report['runtime_flags'] = {'ENABLE_OBJECT_STATES': True, 'USE_GPU_DYNAMICS': False,
                                   'ENABLE_TRANSITION_RULES': False, 'source': 'sampling.sim_utils.SimWorld'}
        env = og.Environment(configs={'env': {'action_frequency': 30, 'physics_frequency': 120, 'rendering_frequency': 30},
            'render': {'viewer_width': 256, 'viewer_height': 256}, 'scene': {'type': 'Scene'}, 'robots': []})
        og.sim.stop()
        objects = []
        for row in rows:
            # Scene roles retain the exact stable-scene initialization scale.
            # Sampleable roles omit scale to observe the provider/runtime default.
            obj = DatasetObject(name='reserved_' + row['model'], category=row['category'], model=row['model'],
                                dataset_name=row.get('dataset','behavior-1k-assets'),
                                **({'scale':row['scene_default_scale']} if row.get('scene_default_scale') is not None else {}))
            env.scene.add_object(obj)
            objects.append(obj)
        og.sim.play()
        for i, (obj, row) in enumerate(zip(objects, rows)):
            obj.set_position_orientation(th.tensor([i * 10., 0., 2.]), th.tensor([0., 0., 0., 1.]))
            lo, hi = obj.aabb
            if not th.isfinite(lo).all() or not th.isfinite(hi).all() or not th.all(hi > lo):
                raise ValueError('invalid native bounds: ' + row['model'])
            collision_count = sum(len(link.collision_meshes) for link in obj.links.values())
            if not collision_count:
                raise ValueError('no collision geometry: ' + row['model'])
            result = {**row, 'default_scale': obj.scale.tolist(), 'native_extent': (hi-lo).tolist(),
                      'collision_meshes': collision_count, 'dataset': row.get('dataset','behavior-1k-assets'),
                      'default_scale_provenance': ('Exact actual stable-scene init_info scale, checked against source hash; observed initialized object.scale' if row.get('scene_default_scale') is not None else 'Native DatasetObject with scale omitted; observed initialized object.scale'),
                      'joint_positions': obj.get_joint_positions().tolist(),
                      'joints': {name:{'name':joint.name,'type':str(joint.joint_type),
                                       'lower_limit':float(joint.lower_limit),'upper_limit':float(joint.upper_limit)}
                                 for name,joint in obj.joints.items()},
                      'images': []}
            centre = ((hi + lo) * .5).cpu().numpy()
            radius = float(th.max(hi - lo)) * 1.7 + .1
            for name, direction in [('front', [1., -1., .55]), ('back', [-1., 1., .55]), ('top', [.01, 0., 1.])]:
                direction = np.array(direction); direction /= np.linalg.norm(direction)
                eye = centre + radius * direction
                back = direction; right = np.cross([0., 1., 0.], back); right /= np.linalg.norm(right)
                up = np.cross(back, right)
                og.sim.viewer_camera.set_position_orientation(th.tensor(eye, dtype=th.float32),
                    mat2quat(th.tensor(np.column_stack([right, up, back]), dtype=th.float32)))
                for _ in range(5):
                    og.sim.render()
                obs, _ = og.sim.viewer_camera.get_obs()
                rgb = obs['rgb'].cpu().numpy()[..., :3]
                path = out / (row['model'] + '_' + name + '.png')
                Image.fromarray(rgb.astype(np.uint8)).save(path)
                result['images'].append({'path': str(path), 'sha256': sha(path)})
            report['objects'].append(result)
            (out / 'default_geometry.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        for task in sorted({r['task'] for r in rows}):
            subset = [r for r in report['objects'] if r['task'] == task]
            for start in range(0,len(subset),10):
                page=subset[start:start+10]
                sheet = Image.new('RGB', (3 * 256, len(page) * 280), 'white')
                draw = ImageDraw.Draw(sheet)
                for i, row in enumerate(page):
                    draw.text((3, i*280+3), row['category'] + '/' + row['model'], fill='black')
                    for j, frame in enumerate(row['images']):
                        sheet.paste(Image.open(frame['path']), (j*256, i*280+24))
                suffix='' if len(subset)<=10 else '_'+str(start//10).zfill(2)
                path = out / (task + '_review'+suffix+'.png'); sheet.save(path)
        report['status'] = 'native_defaults_and_geometry_recorded'
    except BaseException as exc:
        report.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        (out / 'default_geometry.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        sys.stdout.flush(); sys.stderr.flush(); og.shutdown()


if __name__ == '__main__':
    main()
