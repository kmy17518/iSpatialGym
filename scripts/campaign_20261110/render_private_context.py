#!/usr/bin/env python3
"""Render an already validated private scene from cameras inside its room."""
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
    for name in ('campaign-root', 'scene-probe', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists() and a.output.resolve().is_relative_to(a.campaign_root.resolve())
    proof = json.loads(a.scene_probe.read_text())
    assert proof['status'] == 'private_context_geometry_and_reload_passed'
    scene_path = Path(proof['native_scene_path'])
    assert sha(scene_path) == proof['native_scene_sha256']
    a.output.mkdir(parents=True)
    report = dict(schema='campaign-context-image-review-v1', status='started',
        context_probe_path=str(a.scene_probe), context_probe_sha256=sha(a.scene_probe),
        native_scene_path=str(scene_path), native_scene_sha256=sha(scene_path), images=[], production_released=False)
    __import__('ispatialgym.sampling.robot_utils')
    import cv2
    import numpy as np
    import torch as th
    from PIL import Image
    import omnigibson as og
    from omnigibson.utils.transform_utils import mat2quat
    from ispatialgym.sampling.sim_utils import SimWorld
    try:
        world = SimWorld(proof['scene'], [proof['room']], scene_variant={'path': str(scene_path)}, cameras=False)
        assert sha(Path(world.stable_path)) == proof['native_scene_sha256']
        seg = world.scene.seg_map
        mask = (seg.room_ins_map == seg.room_ins_name_to_ins_id[proof['room']]).cpu().numpy().astype('uint8')
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5) * seg.map_resolution
        cells = np.argwhere(distance >= .25)
        assert len(cells)
        xy = np.array([seg.map_to_world(th.tensor(ij)).tolist() for ij in cells])
        meta = json.loads(scene_path.read_text())['metadata']['ispatialgym']['campaign_added_support']
        names = [meta['name']] + [x['name'] for x in proof['objects']]
        bounds = [world.obj(name).aabb for name in names]
        lower = np.min([a.cpu().numpy() for a, b in bounds], axis=0)
        upper = np.max([b.cpu().numpy() for a, b in bounds], axis=0)
        centre = (lower + upper) * .5
        centre[2] = max(.55, min(centre[2], 1.))
        for label, direction in [('south', [0., -1.]), ('north', [0., 1.]), ('east', [1., 0.]), ('west', [-1., 0.])]:
            desired = centre[:2] + 2.15 * np.array(direction)
            index = int(np.argmin(np.linalg.norm(xy - desired, axis=1)))
            eye = np.array([*xy[index], 1.65 + float(world.floor_z)])
            backward = eye - centre; backward /= np.linalg.norm(backward)
            right = np.cross([0., 0., 1.], backward); right /= np.linalg.norm(right)
            up = np.cross(backward, right)
            orientation = mat2quat(th.tensor(np.column_stack([right, up, backward]), dtype=th.float32))
            og.sim.viewer_camera.set_position_orientation(th.tensor(eye, dtype=th.float32), orientation)
            for _ in range(12):
                og.sim.render()
            obs, _ = og.sim.viewer_camera.get_obs()
            path = a.output / (label + '.png')
            Image.fromarray(obs['rgb'].cpu().numpy()[..., :3].astype('uint8')).save(path)
            report['images'].append(dict(path=str(path), sha256=sha(path), eye=eye.tolist(),
                look_at=centre.tolist(), orientation=orientation.tolist(), room_cell=cells[index].tolist()))
        assert sha(scene_path) == proof['native_scene_sha256'] and sha(a.scene_probe) == report['context_probe_sha256']
        report.update(status='exact_private_scene_rendered', object_names=names, requires_direct_visual_review=True)
    except BaseException as exc:
        report.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        (a.output / 'render_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        sys.stdout.flush(); sys.stderr.flush(); og.shutdown()


if __name__ == '__main__':
    main()
