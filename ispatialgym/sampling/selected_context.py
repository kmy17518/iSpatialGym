"""Exact selected-actor construction and finite native floor placement.

This is generation input, not preparation or task acceptance. The caller owns
the fresh world, its bounded actor group, and the subsequent native sampler.
"""
import copy
import math
from pathlib import Path

from ispatialgym.sampling.prepared_articulation import binding, checked
from scripts.campaign_20261110.bind_fresh_candidates import native_scale_equal
from scripts.campaign_20261110.contracts import digest, file_digest, require


def construction_plan(*, metadata_ref, selection_ref, configuration_id, variant,
                      scene_ref, numeric_ref, prototype_name, floor_name, room,
                      yaw_radians, max_candidates=128):
    """Bind native asset defaults and an explicit source fixture construction policy."""
    proof, row, choice = binding(metadata_ref, selection_ref, configuration_id, variant)
    require(row.get('handle_frame', 'canonical') == 'canonical',
            'instance handle annotation cannot be transferred to a constructed actor')
    source, numeric = checked(scene_ref), checked(numeric_ref)
    require(Path(scene_ref['path']).name == choice['scene'] + '_stable.json',
            'selected construction source is not the requested stable scene')
    target = next(role for role in choice['roles'] if role['role'] == 'target')
    require(target.get('numeric_preparation_evidence') == numeric_ref,
            'selected construction must consume its exact prepared native geometry')
    require(numeric.get('status') == 'native_defaults_and_geometry_recorded',
            'selected construction lacks native default geometry')
    matches = [obj for obj in numeric['objects'] if obj['category'] + '/' + obj['model'] == row['model_key']]
    require(len(matches) == 1, 'selected native geometry missing or ambiguous')
    asset = matches[0]
    require(asset['ready'] is True and asset['asset_sha256'] == row['asset_sha256']
            and file_digest(asset['asset_path']) == row['asset_sha256']
            and asset['default_scale'] == target['original_default_scale'],
            'selected construction asset or original native default scale changed')
    init = source['objects_info']['init_info']
    require(prototype_name in init and floor_name in init, 'native construction prototype or floor missing')
    prototype, floor = init[prototype_name]['args'], init[floor_name]['args']
    require(prototype['category'] == asset['category'] and isinstance(prototype.get('fixed_base'), bool),
            'construction needs an explicit original fixture fixed-base policy in the same category')
    require(prototype.get('dataset_name', 'behavior-1k-assets') == asset['dataset'],
            'construction prototype uses a different dataset')
    require(floor['category'] == 'floors' and room in choice['rooms']
            and room in floor.get('in_rooms', []), 'selected placement floor is outside the requested room')
    require(isinstance(max_candidates, int) and 0 < max_candidates <= 256,
            'selected placement candidate bound must be between 1 and 256')
    require(0 < len(yaw_radians) <= 4 and all(math.isfinite(float(x)) for x in yaw_radians),
            'selected placement needs one to four explicit finite yaws')
    value = {'schema': 'isg-selected-context-construction-v1', 'selection_binding': proof,
             'scene': choice['scene'], 'rooms': choice['rooms'], 'room': room,
             'source_scene': scene_ref, 'numeric_geometry': numeric_ref,
             'prototype_name': prototype_name, 'prototype_sha256': digest(init[prototype_name]),
             'fixed_base': prototype['fixed_base'], 'floor_name': floor_name,
             'floor_sha256': digest(init[floor_name]), 'asset_path': asset['asset_path'],
             'asset_sha256': asset['asset_sha256'], 'category': asset['category'], 'model': asset['model'],
             'dataset': asset['dataset'], 'original_default_scale': asset['default_scale'],
             'applied_scale': row['applied_scale'], 'joint': row['consumer_joint'],
             'selected_joint': row['selected_joint'], 'yaw_radians': list(yaw_radians),
             'max_candidates': max_candidates,
             'construction_policy': 'native_selected_asset_defaults_with_source_fixture_fixed_base',
             'placement_policy': 'selected_room_native_floor_collision_free_candidates',
             'motion_readiness_verified': False, 'production_released': False}
    value['actor_name'] = 'campaign_selected_' + digest(value)[:20]
    require(value['actor_name'] not in init, 'selected construction actor already exists')
    return value


def merge_native_actor(source, live, plan, metadata):
    """Export only the newly initialized actor; preserve existing scene entries."""
    name = plan['actor_name']
    init = live['objects_info']['init_info'][name]
    args = init['args']
    require((args['category'], args['model']) == (plan['category'], plan['model'])
            and args['fixed_base'] == plan['fixed_base']
            and args.get('dataset_name', 'behavior-1k-assets') == plan['dataset']
            and native_scale_equal(plan['applied_scale'], args['scale']),
            'native constructed actor differs from exact selected identity')
    require(name not in source['objects_info']['init_info'], 'selected context would replace an existing actor')
    require(metadata['name'] == name and metadata['model_key'] == plan['category'] + '/' + plan['model']
            and metadata['applied_scale'] == plan['applied_scale']
            and metadata['original_default_scale'] == plan['original_default_scale'],
            'constructed actor export metadata differs')
    result = copy.deepcopy(source)
    result['objects_info']['init_info'][name] = copy.deepcopy(init)
    result['state']['registry']['object_registry'][name] = copy.deepcopy(
        live['state']['registry']['object_registry'][name])
    isg = result.setdefault('metadata', {}).setdefault('ispatialgym', {})
    isg.setdefault('campaign_added_context_objects', []).append(copy.deepcopy(metadata))
    isg['selected_context_construction'] = copy.deepcopy(plan)
    return result


def construct_and_place(world, plan):
    """Create at the selected scale once, then try a finite set of floor poses.

    No grasp libraries, reach maps, additional settling loops, or model/scale
    changes on initialized actors. Later native generation must establish task
    utility, motion feasibility, demo and replay fidelity.
    """
    import cv2
    import numpy as np
    import torch
    from omnigibson.utils.sampling_utils import raytest_batch
    from omnigibson.utils.asset_utils import get_dataset_path
    from ispatialgym.sampling import knowledge as K
    from ispatialgym.sampling.sim_utils import _collision_piece_aabbs
    from ispatialgym.sampling.scene_variants import blocked_footprints

    require(world.scene_model == plan['scene'] and set(world.rooms) == set(plan['rooms'])
            and file_digest(world.stable_path) == plan['source_scene']['sha256'],
            'selected construction world does not match its source and room group')
    require(world.model_pool is None and not world.spawned,
            'selected context construction requires its own fresh generation world')
    require(file_digest(plan['asset_path']) == plan['asset_sha256'], 'selected native asset changed')
    native_asset = Path(get_dataset_path(plan['dataset'])) / 'objects' / plan['category'] / plan['model'] / 'usd'
    native_asset /= plan['model'] + ('.encrypted.usd' if plan['dataset'] == 'behavior-1k-assets' else '.usd')
    require(file_digest(native_asset) == plan['asset_sha256'] and K.is_ready(plan['model']),
            'native factory asset or readiness catalog differs from selection')
    floor = world.obj(plan['floor_name'])
    require(floor.category == 'floors' and plan['room'] in floor.in_rooms,
            'native selected floor differs')
    obj = world.spawn(plan['actor_name'], plan['category'], plan['model'],
                      dataset=plan['dataset'], scale=plan['applied_scale'], fixed_base=plan['fixed_base'])
    require(native_scale_equal(plan['applied_scale'], obj.scale.tolist()),
            'native constructor did not retain the exact selected applied scale')
    joints = [joint for name, joint in obj.joints.items()
              if name == plan['joint'] or name.endswith('joint_' + plan['joint'])]
    require(len(joints) == 1, 'selected native articulation joint missing or ambiguous')
    joint, expected = joints[0], plan['selected_joint']
    require(bool(joint.is_revolute) == ('revolute' in expected['type'].lower())
            and abs(float(joint.lower_limit) - expected['lower_limit']) < 1e-5
            and abs(float(joint.upper_limit) - expected['upper_limit']) < 1e-5,
            'selected native articulation limits or type differ')
    seg = world.scene.seg_map
    room_id = seg.room_ins_name_to_ins_id[plan['room']]
    mask = (seg.room_ins_map == room_id).cpu().numpy().astype('uint8')
    distances = cv2.distanceTransform(mask, cv2.DIST_L2, 5) * seg.map_resolution
    floor_paths = {link.prim_path for link in floor.links.values()}
    ignore = [link.prim_path for entity in (obj, world.robot) for link in entity.links.values()]
    floor_lo, floor_hi = floor.aabb
    source = checked(plan['source_scene'])
    anchor = source['state']['registry']['object_registry'][plan['prototype_name']]['root_link']['pos']
    tested = []
    for yaw in plan['yaw_radians']:
        orientation = torch.tensor([0., 0., math.sin(yaw / 2), math.cos(yaw / 2)])
        obj.set_position_orientation(torch.tensor([0., 0., -100.]), orientation)
        position, _ = obj.get_position_orientation(); lo, hi = obj.aabb
        bottom = float(lo[2] - position[2]); radius = float(torch.linalg.norm((hi - lo)[:2])) / 2 + .1
        cells = np.argwhere(distances >= radius)
        xy = seg.map_to_world(torch.as_tensor(cells)).cpu().numpy() if len(cells) else []
        candidates = sorted(xy, key=lambda point: (float(np.linalg.norm(point - np.asarray(anchor[:2]))),
                                                  float(point[0]), float(point[1])))
        per_yaw = (plan['max_candidates'] + len(plan['yaw_radians']) - 1) // len(plan['yaw_radians'])
        for point in candidates[:per_yaw]:
            if len(tested) >= plan['max_candidates']:
                break
            x, y = map(float, point)
            hits = raytest_batch([[x, y, float(floor_hi[2]) + .1]],
                                 [[x, y, float(floor_lo[2]) - .1]], only_closest=True, ignore_bodies=ignore)
            hit = hits[0] if hits else {}
            attempt = {'xy': [x, y], 'yaw': yaw, 'floor_hit': hit.get('rigidBody')}
            tested.append(attempt)
            if not hit.get('hit') or hit.get('rigidBody') not in floor_paths:
                attempt['rejected'] = 'selected_floor_not_hit'
                continue
            obj.set_position_orientation(torch.tensor([x, y, float(hit['position'][2]) - bottom]), orientation)
            pieces = _collision_piece_aabbs(obj)
            blockers = []
            for other in world.scene.objects:
                if other is obj or other is floor:
                    continue
                overlap = max((min(float(torch.min(a[1], b[1])[i] - torch.max(a[0], b[0])[i])
                                   for i in range(3)) for a in pieces for b in _collision_piece_aabbs(other)), default=-1.)
                if overlap > .002:
                    blockers.append(other.name)
            attempt['blockers'] = blockers
            if blockers:
                continue
            world.activate(obj)
            lo, hi = obj.aabb
            metadata = {'name': obj.name, 'model_key': plan['category'] + '/' + plan['model'],
                        'asset_sha256': plan['asset_sha256'],
                        'native_factory_asset_path': str(native_asset),
                        'original_default_scale': plan['original_default_scale'],
                        'applied_scale': plan['applied_scale'],
                        'source_scene_sha256': plan['source_scene']['sha256'], 'room': plan['room'],
                        'blocked_footprints': blocked_footprints([(lo.tolist(), hi.tolist())]),
                        'selected_construction_sha256': digest(plan), 'production_released': False}
            return obj, metadata, {'candidate_count': len(tested), 'attempts': tested,
                                    'position': obj.get_position_orientation()[0].tolist(),
                                    'orientation': obj.get_position_orientation()[1].tolist(),
                                    'native_motion_readiness_verified': False}
    error = ValueError('selected actor has no valid native floor placement within the declared candidate bound')
    error.placement_attempts = tested
    raise error
