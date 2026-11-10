#!/usr/bin/env python3
"""Read actual open-fridge interior planes in an isolated simulator world."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-root',type=Path,required=True)
    p.add_argument('--jobs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root,out=a.campaign_root.resolve(),a.output.resolve()
    if not out.is_relative_to(root) or out==root or out.exists():
        raise ValueError('output must be a new private campaign directory')
    job=next(j for j in json.loads(a.jobs.read_text())['jobs'] if j['class_id']=='inter_object_relation_satisfaction-inside_large_scale_02')
    expected=next(o for o in job['expected_objects'] if o['role']=='partner')
    if not expected['ready'] or (expected['category'],expected['model'])!=('fridge','xyejdx'):
        raise ValueError('unexpected or unready partner')
    out.mkdir(parents=True)
    os.environ.pop('ISPATIALGYM_CAMPAIGN_CONFIG',None)
    os.environ.pop('ISPATIALGYM_CAMPAIGN_GOAL_PLAN',None)
    __import__('ispatialgym.sampling.robot_utils')
    import omnigibson as og
    from ispatialgym.sampling.sim_utils import SimWorld
    from ispatialgym.sampling import object_joints as OJ
    report={'status':'started','production_released':False,'scene':job['scene'],'room':job['room'],
            'class_id':job['class_id'],'jobs_sha256':hashlib.sha256(a.jobs.read_bytes()).hexdigest(),
            'planes':[],'rejected_heights':[]}
    try:
        world=SimWorld(job['scene'],[job['room']],cameras=False)
        xy=world.park_cell()
        world.floor_z=world.measure_floor_z(xy,job['room'])
        world.park_robot(xy,floor_z=world.floor_z)
        obj=world.obj(expected['scene_init_args']['name'])
        if max(abs(float(x)-y) for x,y in zip(obj.scale,expected['default_scale']))>1e-6:
            raise ValueError('live fridge default scale mismatch')
        spec,clamped=OJ.parse_joint_spec({'j_link_0':1.55,'j_link_1':1.55},list(obj.joints),OJ.joint_limits(obj))
        before=OJ.link_poses(obj)
        OJ.set_joints(obj,spec)
        residual=world.settle([obj],steps=90)
        obj.keep_still()
        joints=OJ.joint_dict(obj)
        if max(abs(joints[k]-v) for k,v in spec.items())>.05:
            raise ValueError('fridge joints drifted during native settle')
        moved=OJ.moved_links(before,OJ.link_poses(obj))
        static=[n for n in obj.links if n not in moved]
        boxes=OJ.link_piece_aabbs(obj,static)
        lo,hi=OJ.static_aabb(obj,moved)
        heights=sorted({round(float(b[1][2]),3) for b in boxes
                        if float(b[1][2]-b[0][2])<=.15 and float(b[1][0]-b[0][0])>=.05
                        and float(b[1][1]-b[0][1])>=.05 and float(lo[2])+.05<float(b[1][2])<float(hi[2])-.05})
        report.update(object=obj.name,asset_sha256=expected['asset_sha256'],scale=obj.scale.tolist(),
                      floor_z_world=float(world.floor_z),requested_joints=spec,actual_joints=joints,clamped=clamped,
                      moved_links=moved,settle_residual=residual,static_aabb=[lo.tolist(),hi.tolist()],candidate_world_heights=heights)
        for height in heights:
            try:
                interior=world.support_interior(obj,height,resolution=.02,z_margin=.015)
                OJ.mask_out_boxes(interior,OJ.link_piece_aabbs(obj,moved),pad=.01)
                info=interior.to_json()
                info.update(height_above_floor=float(interior.z)-float(world.floor_z),
                            height_key=f'{float(interior.z)-float(world.floor_z):.4f}',
                            free_area_m2=float(interior.free.sum())*.02**2,
                            cells_headroom_at_least_20cm=int(interior.usable(.18,.02).sum()))
                report['planes'].append(info)
            except RuntimeError as exc:
                report['rejected_heights'].append({'height':height,'error':str(exc)})
        if not report['planes']:
            raise ValueError('no native interior plane found')
        report['status']='measured_planes_require_target_geometry_and_maps'
    except BaseException as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        (out/'interior_probe.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        sys.stdout.flush();sys.stderr.flush()
        og.shutdown()


if __name__=='__main__':
    main()
