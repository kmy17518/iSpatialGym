#!/usr/bin/env python3
"""Insert one reserved default-scale table into a private heldout-room scene.

Only the new object is copied from the simulator serialization into the original
stable scene. Existing object definitions and states remain byte-equivalent as
JSON values. New support geometry, floor rest, and reload are independently
checked; this diagnostic does not release a task template.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import traceback


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(test,msg):
    if not test:raise ValueError(msg)
def depth(a,b):return min(min(float(a[1][i]),float(b[1][i]))-max(float(a[0][i]),float(b[0][i])) for i in range(3))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('campaign-root','defaults','reservations','source-scene','output'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--category',default='pedestal_table');p.add_argument('--model',default='tdpuit')
    a=p.parse_args();out=a.output.resolve();require(not out.exists() and out.is_relative_to(a.campaign_root.resolve()),'new private output required')
    scene='office_cubicles_right';room='conference_hall_0';key=a.category+'/'+a.model;name=a.category+'_'+a.model+'_campaign_0'
    policy=json.loads(a.reservations.read_text());defaults=json.loads(a.defaults.read_text())
    require(defaults['status']=='native_defaults_and_geometry_recorded' and defaults['reservation_sha256']==sha(a.reservations),'native default probe mismatch')
    row=next(r for r in defaults['objects'] if r['category']+'/'+r['model']==key)
    require(row['ready'] and key in policy['category_holdout_models'] and a.category in policy['category_holdout_categories'],'support not a ready reserved category')
    require({'scene':scene,'room':room} in policy['category_holdout_rooms'],'room not heldout')
    require(sha(Path(row['asset_path']))==row['asset_sha256'],'support asset changed')
    source=json.loads(a.source_scene.read_text());source_hash=sha(a.source_scene)
    require(name not in source['objects_info']['init_info'],'support name already exists')
    out.mkdir(parents=True);report={'schema':'campaign-category-support-scene-v1','status':'started','production_released':False,'source_scene_sha256':source_hash,'defaults_sha256':sha(a.defaults),'reservation_sha256':sha(a.reservations),'scene':scene,'room':room,'object_name':name,'model_key':key,'default_scale':row['default_scale'],'asset_sha256':row['asset_sha256'],'readiness_source':row['readiness_source']}
    __import__('ispatialgym.sampling.robot_utils')
    import cv2
    import numpy as np
    import torch as th
    from PIL import Image
    import omnigibson as og
    from omnigibson.utils.config_utils import TorchEncoder
    from omnigibson.utils.transform_utils import mat2quat
    from ispatialgym.sampling.sim_utils import SimWorld,_collision_piece_aabbs
    from ispatialgym.sampling.scene_variants import blocked_footprints
    try:
        world=SimWorld(scene,[room],cameras=False)
        require(sha(Path(world.stable_path))==source_hash,'runtime stable source differs')
        obj=world.spawn(name,a.category,a.model,scale=row['default_scale'])
        pos,_=obj.get_position_orientation();lo,hi=obj.aabb;local_bottom=float(lo[2]-pos[2]);extent=hi-lo
        seg=world.scene.seg_map;mask=(seg.room_ins_map==seg.room_ins_name_to_ins_id[room]).cpu().numpy().astype('uint8')
        distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)*seg.map_resolution
        clearance=float(th.linalg.norm(extent[:2]))*.5+.55
        indices=np.argwhere(distance>=clearance)
        require(len(indices)>0,'heldout room has no conservative support footprint plus walking clearance')
        indices=sorted(indices,key=lambda x:(-float(distance[tuple(x)]),int(x[0]),int(x[1])))
        selected=None;tested=[]
        for ij in indices:
            xy=seg.map_to_world(th.tensor(ij)).tolist()
            if any(sum((xy[k]-v[k])**2 for k in range(2))<.25**2 for v in tested):continue
            tested.append(xy)
            obj.set_position_orientation(th.tensor([*xy,float(world.floor_z)-local_bottom+.001]),th.tensor([0.,0.,0.,1.]))
            pieces=_collision_piece_aabbs(obj);bad=[]
            for other in world.scene.objects:
                if other.name==name or other.category=='floors':continue
                amount=max((depth(x,y) for x in pieces for y in _collision_piece_aabbs(other)),default=-1.)
                if amount>.002:bad.append({'name':other.name,'overlap_m':amount})
            if not bad:selected=xy;break
            if len(tested)>=200:break
        require(selected is not None,'all conservative room placements overlap existing collision pieces')
        world.activate(obj);start=obj.get_position_orientation()[0].clone()
        error=world.settle([obj],steps=90,still=[o for o in world.scene.objects if o.name!=name])
        require(error is None,'support stability failed: '+str(error))
        lo,hi=obj.aabb;require(abs(float(lo[2])-float(world.floor_z))<.004,'support not resting on floor')
        require(float(th.linalg.norm(obj.get_position_orientation()[0]-start))<.004,'support moved during settling')
        # Copy exactly one newly serialized object into the unchanged full scene.
        live=json.loads(json.dumps(world.scene.save(as_dict=True),cls=TorchEncoder))
        changed=copy.deepcopy(source)
        changed['objects_info']['init_info'][name]=live['objects_info']['init_info'][name]
        changed['state']['registry']['object_registry'][name]=live['state']['registry']['object_registry'][name]
        fps=blocked_footprints([(lo.tolist(),hi.tolist())])
        metadata={'model_key':key,'name':name,'asset_sha256':row['asset_sha256'],'default_scale':row['default_scale'],'source_scene_sha256':source_hash,'room':room,'blocked_footprints':fps,'production_released':False}
        changed.setdefault('metadata',{}).setdefault('ispatialgym',{})['campaign_added_support']=metadata
        stable=out/'stable_scenes'/f'{scene}_stable.json';stable.parent.mkdir();stable.write_text(json.dumps(changed,indent=2)+'\n')
        og.clear();world=SimWorld(scene,[room],scene_variant={'path':str(stable)},cameras=False);obj=world.obj(name)
        require(all(abs(float(v)-w)<1e-6 for v,w in zip(obj.scale,row['default_scale'])),'reloaded support scale mismatch')
        lo,hi=obj.aabb;require(abs(float(lo[2])-float(world.floor_z))<.004,'reloaded support lost floor contact')
        hits=[]
        for i in range(1,8):
            for j in range(1,8):
                xy=[float(lo[0])+(float(hi[0])-float(lo[0]))*i/8,float(lo[1])+(float(hi[1])-float(lo[1]))*j/8]
                hit=world.surface_z_under(xy,float(hi[2])+.02,depth=.08)
                if hit and hit['body'].startswith(obj.prim_path+'/') and hit['normal'][2]>=.95:hits.append(hit)
        require(len(hits)>=9,'support lacks a broad measured top plane')
        top=statistics.median(x['z'] for x in hits);require(max(abs(h['z']-top) for h in hits)<.005,'support top is not planar')
        centre=((lo+hi)*.5).cpu().numpy();direction=np.array([1.,-1.,.7]);direction/=np.linalg.norm(direction)
        eye=centre+(float(th.max(hi-lo))*2.+.8)*direction;right=np.cross([0.,1.,0.],direction);right/=np.linalg.norm(right);up=np.cross(direction,right)
        og.sim.viewer_camera.set_position_orientation(th.tensor(eye,dtype=th.float32),mat2quat(th.tensor(np.column_stack([right,up,direction]),dtype=th.float32)))
        for _ in range(8):og.sim.render()
        obs,_=og.sim.viewer_camera.get_obs();png=out/'support_in_room.png';Image.fromarray(obs['rgb'].cpu().numpy()[...,:3].astype('uint8')).save(png)
        require(sha(a.source_scene)==source_hash,'original source scene changed')
        for n,v in source['objects_info']['init_info'].items():require(changed['objects_info']['init_info'][n]==v,'original definition altered')
        for n,v in source['state']['registry']['object_registry'].items():require(changed['state']['registry']['object_registry'][n]==v,'original state altered')
        report.update(status='private_support_geometry_and_reload_passed',native_scene_path=str(stable),native_scene_sha256=sha(stable),position=obj.get_position_orientation()[0].tolist(),orientation=obj.get_position_orientation()[1].tolist(),actual_height_key=f'{top-float(world.floor_z):.4f}',measured_height_world=top,floor_z_world=float(world.floor_z),aabb_lower=lo.tolist(),aabb_upper=hi.tolist(),tested_placements=len(tested),blocked_footprints=fps,top_hits=hits,image={'path':str(png),'sha256':sha(png)},original_object_values_unchanged=True)
    except BaseException as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc());traceback.print_exc();raise
    finally:
        (out/'support_scene_probe.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');sys.stdout.flush();sys.stderr.flush();og.shutdown()


if __name__=='__main__':main()
