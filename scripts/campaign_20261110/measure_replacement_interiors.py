#!/usr/bin/env python3
"""Measure native private interior-map geometry for the two moved partners."""
import argparse
import json
from pathlib import Path
import sys
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-root',type=Path,required=True)
    p.add_argument('--jobs',type=Path,required=True)
    p.add_argument('--case',choices=['fridge','bookcase'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.campaign_root.resolve();out=a.output.resolve()
    if not out.is_relative_to(root) or out==root or out.exists():raise ValueError('new private output required')
    suffix='02' if a.case=='fridge' else '01'
    job=next(j for j in json.loads(a.jobs.read_text())['jobs'] if j['class_id']=='inter_object_relation_satisfaction-inside_large_scale_'+suffix)
    expected=next(o for o in job['expected_objects'] if o['role']=='partner')
    if not expected['ready']:raise ValueError('unready partner')
    out.mkdir(parents=True)
    __import__('ispatialgym.sampling.robot_utils')
    import omnigibson as og
    import torch as th
    from ispatialgym.sampling.sim_utils import SimWorld
    from ispatialgym.sampling import interiors as I,object_joints as OJ
    report={'case':a.case,'class_id':job['class_id'],'status':'started','production_released':False}
    try:
        world=SimWorld(job['scene'],[job['room']],cameras=False)
        xy=world.park_cell();world.floor_z=world.measure_floor_z(xy,job['room']);world.park_robot(xy,floor_z=world.floor_z)
        obj=world.obj(expected['scene_init_args']['name']);moved=[];joints={}
        if a.case=='fridge':
            joints,_=OJ.parse_joint_spec({'j_link_0':1.55,'j_link_1':1.55},list(obj.joints),OJ.joint_limits(obj))
            before=OJ.link_poses(obj);OJ.set_joints(obj,joints);world.settle([obj],steps=90)
            actual=OJ.joint_dict(obj)
            if max(abs(actual[k]-v) for k,v in joints.items())>.05:raise ValueError('fridge door drift')
            moved=OJ.moved_links(before,OJ.link_poses(obj));height=.8245680928230286;side=None;stands=[.35,.45,.55,.65]
        else:
            height=.8354;side=th.tensor([0.,1.]);stands=[.55,.65,.75,.85]
        top=world.support_interior(obj,world.floor_z+height,z_margin=.03)
        open_edges=list(top.open_edges)
        if not open_edges:raise ValueError('no native open interior edge')
        edges={n:normal for n,_mid,normal,_length in top.edges()}
        edge=max(open_edges,key=lambda n:float(edges[n]@side)) if side is not None else open_edges[0]
        root_name=getattr(obj,'root_link_name','base_link')
        measured=I.measure(world,obj,height,name='campaign_'+a.case+'_'+expected['model'],scene=job['scene'],edge=edge,
                           joints=joints,moved_links=[n for n in moved if n!=root_name],edge_distances=stands,z_margin=.03,
                           static_aabb=OJ.static_aabb(obj,moved) if moved else None,
                           lateral_m=.3 if a.case=='bookcase' else None)
        I.write_registry(str(out/'interiors.json'),[measured])
        report.update(status='native_map_geometry_measured',object=obj.name,default_scale=obj.scale.tolist(),
                      asset_sha256=expected['asset_sha256'],floor_z_world=float(world.floor_z),
                      observed_open_edges=open_edges,selected_edge=edge,spec=measured.to_json(),
                      note='Native map envelope thickness defaults remain explicit in spec source; final task checks use actual collision geometry.')
    except BaseException as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc());traceback.print_exc();raise
    finally:
        (out/'measurement.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');sys.stdout.flush();sys.stderr.flush();og.shutdown()


if __name__=='__main__':main()
