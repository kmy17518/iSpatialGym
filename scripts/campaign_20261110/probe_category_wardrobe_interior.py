#!/usr/bin/env python3
"""Measure actual private wardrobe opening and shelf planes without task changes."""
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
    for name in ('campaign-root', 'scene-probe', 'defaults', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--map-height-key')
    a = p.parse_args()
    assert not a.output.exists() and a.output.resolve().is_relative_to(a.campaign_root.resolve())
    proof = json.loads(a.scene_probe.read_text()); defaults = json.loads(a.defaults.read_text())
    assert proof['status'] == 'private_context_geometry_and_reload_passed'
    row = next(x for x in defaults['objects'] if x['category'] == 'wardrobe' and x['model'] == 'brimns')
    target = next(x for x in proof['objects'] if x['model_key'] == 'wardrobe/brimns')
    assert row['ready'] and row['asset_sha256'] == target['asset_sha256'] == sha(Path(row['asset_path']))
    scene = Path(proof['native_scene_path']); assert sha(scene) == proof['native_scene_sha256']
    a.output.mkdir(parents=True)
    report = dict(schema='campaign-category-interior-native-v1', status='started', scene=proof['scene'], room=proof['room'],
        scene_probe_sha256=sha(a.scene_probe), native_scene_sha256=sha(scene), asset_sha256=row['asset_sha256'],
        default_scale=row['default_scale'], planes=[], rejected_heights=[], production_released=False)
    __import__('ispatialgym.sampling.robot_utils')
    import omnigibson as og
    from ispatialgym.sampling.sim_utils import SimWorld
    from ispatialgym.sampling import object_joints as OJ
    try:
        world = SimWorld(proof['scene'], [proof['room']], scene_variant={'path': str(scene)}, cameras=False)
        obj = world.obj(target['name'])
        assert max(abs(float(x)-y) for x,y in zip(obj.scale,row['default_scale'])) < 1e-6
        requested = {name: joint['lower_limit'] + .85 * (joint['upper_limit']-joint['lower_limit']) for name,joint in row['joints'].items()}
        spec, clamped = OJ.parse_joint_spec(requested, list(obj.joints), OJ.joint_limits(obj))
        assert not clamped
        before = OJ.link_poses(obj); OJ.set_joints(obj, spec)
        residual = world.settle([obj], steps=90); obj.keep_still(); joints = OJ.joint_dict(obj)
        assert max(abs(joints[k]-v) for k,v in spec.items()) <= .05, 'native opened joints drifted'
        moved = OJ.moved_links(before, OJ.link_poses(obj)); static = [n for n in obj.links if n not in moved]
        boxes = OJ.link_piece_aabbs(obj, static); lo, hi = OJ.static_aabb(obj, moved)
        heights = sorted({round(float(b[1][2]),3) for b in boxes if float(b[1][2]-b[0][2]) <= .15
            and float(b[1][0]-b[0][0]) >= .05 and float(b[1][1]-b[0][1]) >= .05
            and float(lo[2]) + .025 < float(b[1][2]) < float(hi[2]) - .05})
        report.update(object=obj.name, floor_z_world=float(world.floor_z), requested_joints=spec, actual_joints=joints,
            moved_links=moved, settle_residual=residual, static_aabb=[lo.tolist(),hi.tolist()], candidate_world_heights=heights)
        for height in heights:
            try:
                interior = world.support_interior(obj, height, resolution=.02, z_margin=.015)
                OJ.mask_out_boxes(interior, OJ.link_piece_aabbs(obj,moved),pad=.01)
                info = interior.to_json()
                info.update(height_above_floor=float(interior.z)-float(world.floor_z), height_key=f'{float(interior.z)-float(world.floor_z):.4f}',
                    free_area_m2=float(interior.free.sum())*.02**2, cells_headroom_at_least_5cm=int(interior.usable(.03,.02).sum()))
                report['planes'].append(info)
            except RuntimeError as exc:
                report['rejected_heights'].append(dict(height=height,error=str(exc)))
        assert report['planes'], 'no native shelf plane found'
        if a.map_height_key:
            from ispatialgym.sampling import interiors as I
            selected = next(x for x in report['planes'] if x['height_key'] == a.map_height_key)
            assert len(selected['open_edges']) == 1, 'map measurement needs an unambiguous opening'
            measured = I.measure(world, obj, selected['height_above_floor'], name='campaign_wardrobe_brimns_'+a.map_height_key.replace('.',''),
                scene=proof['scene'], edge=selected['open_edges'][0], joints=spec, moved_links=moved,
                edge_distances=[.35,.45,.55,.65], z_margin=.015, static_aabb=OJ.static_aabb(obj,moved), open_top=False)
            # This is a physically enclosed wardrobe, with a finite measured
            # ceiling above the shelf. The generic basin rim search caps at0.8m
            # and misclassifies the1.10m compartment as open-top; that path then
            # boxes its whole own carcass AABB as an external overhead obstacle.
            # Native closed-top mode retains the measured ceiling and doors.
            report['map_topology'] = {'open_top': False, 'reason': 'Enclosed wardrobe ceiling observed in native geometry and model render; retain measured headroom and ceiling plane'}
            I.write_registry(str(a.output/'interiors.json'),[measured])
            report['native_map_spec'] = measured.to_json()
            report['interiors_sha256'] = sha(a.output/'interiors.json')
        assert sha(scene) == proof['native_scene_sha256']
        report['status'] = 'measured_planes_require_target_geometry_and_maps'
    except BaseException as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc()); traceback.print_exc(); raise
    finally:
        (a.output/'interior_probe.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        sys.stdout.flush();sys.stderr.flush();og.shutdown()


if __name__ == '__main__':
    main()
