#!/usr/bin/env python3
"""Build a private conference scene with measured, reserved camera landmarks."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(test, reason):
    if not test:
        raise ValueError(reason)


def overlap(a, b):
    return min(min(float(a[1][i]), float(b[1][i])) - max(float(a[0][i]), float(b[0][i])) for i in range(3))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign-root', 'defaults', 'reservations', 'asset-hashes', 'source-scene', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    require(not out.exists() and out.is_relative_to(a.campaign_root.resolve()), 'new private output required')
    policy, hashes, defaults = [json.loads(x.read_text()) for x in (a.reservations, a.asset_hashes, a.defaults)]
    require(hashes['reservation_sha256'] == sha(a.reservations), 'current reservation identity mismatch')
    require(defaults['status'] == 'native_defaults_and_geometry_recorded', 'native default measurement incomplete')
    # Native measurements may precede a monotonic policy extension. Check every
    # selected model/hash against the current full policy, rather than relabeling it.
    keys = ('armchair/pmrcex', 'footstool/sghekm')
    rows = [next(r for r in defaults['objects'] if r['category'] + '/' + r['model'] == key) for key in keys]
    for key, row in zip(keys, rows):
        require(row['ready'] and key in policy['category_holdout_models'] and row['category'] in policy['category_holdout_categories'], 'context must be ready and category-reserved')
        require(row['asset_sha256'] == hashes['ready_assets'][key]['asset_sha256'] == sha(Path(row['asset_path'])), 'context asset provenance mismatch')
    scene, room = 'office_cubicles_right', 'conference_hall_0'
    require({'scene': scene, 'room': room} in policy['category_holdout_rooms'], 'room must remain held out')
    source, source_hash = json.loads(a.source_scene.read_text()), sha(a.source_scene)
    table_meta = source.get('metadata', {}).get('ispatialgym', {}).get('campaign_added_support')
    require(table_meta and table_meta['model_key'] == 'pedestal_table/tdpuit', 'expected measured table-only private source')
    require(not source.get('metadata', {}).get('ispatialgym', {}).get('campaign_added_context_objects'), 'source already contains campaign context')
    out.mkdir(parents=True)
    report = dict(schema='campaign-category-camera-context-v1', status='started', production_released=False,
        source_scene=str(a.source_scene), source_scene_sha256=source_hash, defaults_sha256=sha(a.defaults),
        reservation_sha256=sha(a.reservations), scene=scene, room=room, objects=[])
    __import__('ispatialgym.sampling.robot_utils')
    import cv2
    import numpy as np
    import torch as th
    from PIL import Image
    import omnigibson as og
    from omnigibson.utils.config_utils import TorchEncoder
    from omnigibson.utils.transform_utils import mat2quat
    from ispatialgym.sampling.sim_utils import SimWorld, _collision_piece_aabbs
    from ispatialgym.sampling.scene_variants import blocked_footprints
    try:
        world = SimWorld(scene, [room], scene_variant={'path': str(a.source_scene)}, cameras=False)
        require(sha(Path(world.stable_path)) == source_hash, 'runtime source mismatch')
        table = world.obj(table_meta['name'])
        table_pos = table.get_position_orientation()[0].tolist()
        seg = world.scene.seg_map
        room_id = seg.room_ins_name_to_ins_id[room]
        mask = (seg.room_ins_map == room_id).cpu().numpy().astype('uint8')
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5) * seg.map_resolution
        metadata, added = [], []

        def blockers(obj):
            pieces = _collision_piece_aabbs(obj)
            return [other.name for other in world.scene.objects if other.name != obj.name and other.category != 'floors'
                    and max((overlap(x, y) for x in pieces for y in _collision_piece_aabbs(other)), default=-1.) > .002]

        for index, row in enumerate(rows):
            name = 'campaign_context_' + row['category'] + '_' + row['model'] + '_0'
            require(name not in source['objects_info']['init_info'], 'context name already exists')
            prior_poses = [(prior, prior.get_position_orientation()) for prior in added]
            obj = world.spawn(name, row['category'], row['model'], scale=row['default_scale'])
            # SimWorld.spawn parks every previously spawned object while it
            # rebuilds PhysX views. These completed context objects must return
            # to their validated poses before placement/collision checks.
            for prior, pose in prior_poses:
                world.activate(prior)
                prior.set_position_orientation(*pose)
            yaw = math.pi / 2 if index == 0 else -math.pi / 2
            orientation = th.tensor([0., 0., math.sin(yaw / 2), math.cos(yaw / 2)])
            obj.set_position_orientation(th.tensor([0., 0., 5.]), orientation)
            pos, _ = obj.get_position_orientation(); lo, hi = obj.aabb
            local_bottom = float(lo[2] - pos[2]); extent = hi - lo
            clearance = float(th.linalg.norm(extent[:2])) * .5 + .2
            indices = np.argwhere(distance >= clearance)
            desired = [table_pos[0] + (-1.5 if index == 0 else 1.4), table_pos[1] - .2]
            candidates = [(seg.map_to_world(th.tensor(ij)).tolist(), ij) for ij in indices]
            candidates.sort(key=lambda pair: sum((pair[0][k] - desired[k]) ** 2 for k in range(2)))
            tested, selected = [], None
            for xy, _ in candidates:
                if sum((xy[k] - table_pos[k]) ** 2 for k in range(2)) > 3.5 ** 2:
                    continue
                if any(sum((xy[k] - old[k]) ** 2 for k in range(2)) < .2 ** 2 for old in tested):
                    continue
                tested.append(xy)
                obj.set_position_orientation(th.tensor([*xy, float(world.floor_z) - local_bottom + .001]), orientation)
                if not blockers(obj):
                    selected = xy
                    break
                if len(tested) >= 200:
                    break
            require(selected is not None, 'no collision-free context placement near the table')
            world.activate(obj)
            before = obj.get_position_orientation()[0].clone()
            err = world.settle([obj], steps=90, still=[o for o in world.scene.objects if o.name != name])
            require(err is None, 'context did not settle: ' + str(err))
            lo, hi = obj.aabb
            require(abs(float(lo[2]) - float(world.floor_z)) < .004, 'context floor contact failed')
            require(float(th.linalg.norm(obj.get_position_orientation()[0] - before)) < .004 and not blockers(obj), 'context drift/collision after settling')
            meta = dict(name=name, model_key=row['category'] + '/' + row['model'], asset_sha256=row['asset_sha256'],
                default_scale=row['default_scale'], readiness_source=row['readiness_source'], source_scene_sha256=source_hash,
                room=room, blocked_footprints=blocked_footprints([(lo.tolist(), hi.tolist())]), production_released=False)
            metadata.append(meta); added.append(obj)
            report['objects'].append({**meta, 'position': obj.get_position_orientation()[0].tolist(), 'orientation': obj.get_position_orientation()[1].tolist(), 'tested_placements': len(tested)})
        live = json.loads(json.dumps(world.scene.save(as_dict=True), cls=TorchEncoder))
        changed = copy.deepcopy(source)
        for obj in added:
            name = obj.name
            changed['objects_info']['init_info'][name] = live['objects_info']['init_info'][name]
            changed['state']['registry']['object_registry'][name] = live['state']['registry']['object_registry'][name]
        changed.setdefault('metadata', {}).setdefault('ispatialgym', {})['campaign_added_context_objects'] = metadata
        stable = out / 'stable_scenes' / f'{scene}_stable.json'
        stable.parent.mkdir(); stable.write_text(json.dumps(changed, indent=2) + '\n')
        og.clear()
        world = SimWorld(scene, [room], scene_variant={'path': str(stable)}, cameras=False)
        for meta in metadata:
            obj = world.obj(meta['name']); lo, hi = obj.aabb
            require(all(abs(float(x) - y) < 1e-6 for x, y in zip(obj.scale, meta['default_scale'])), 'context scale changed on reload')
            require(abs(float(lo[2]) - float(world.floor_z)) < .004 and not blockers(obj), 'reloaded context geometry failed')
        for name, value in source['objects_info']['init_info'].items():
            require(changed['objects_info']['init_info'][name] == value, 'existing object definition altered')
        for name, value in source['state']['registry']['object_registry'].items():
            require(changed['state']['registry']['object_registry'][name] == value, 'existing object state altered')
        require(sha(a.source_scene) == source_hash, 'source file changed')
        centre = np.array([table_pos[0], table_pos[1], .65])
        images = []
        for label, direction in [('front', [0., -1., .65]), ('reverse', [0., 1., .65]), ('top', [.01, 0., 1.])]:
            direction = np.array(direction); direction /= np.linalg.norm(direction)
            eye = centre + 6. * direction
            right = np.cross([0., 1., 0.], direction); right /= np.linalg.norm(right); up = np.cross(direction, right)
            og.sim.viewer_camera.set_position_orientation(th.tensor(eye, dtype=th.float32), mat2quat(th.tensor(np.column_stack([right, up, direction]), dtype=th.float32)))
            for _ in range(12):
                og.sim.render()
            obs, _ = og.sim.viewer_camera.get_obs()
            png = out / (label + '.png'); Image.fromarray(obs['rgb'].cpu().numpy()[..., :3].astype('uint8')).save(png)
            images.append(dict(path=str(png), sha256=sha(png)))
        report.update(status='private_context_geometry_and_reload_passed', native_scene_path=str(stable), native_scene_sha256=sha(stable),
            original_object_values_unchanged=True, images=images, native_camera_visibility_gate='pending; no thresholds changed')
    except BaseException as exc:
        report.update(status='failed', error=str(exc), traceback=traceback.format_exc()); traceback.print_exc(); raise
    finally:
        (out / 'context_scene_probe.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        sys.stdout.flush(); sys.stderr.flush(); og.shutdown()


if __name__ == '__main__':
    main()
