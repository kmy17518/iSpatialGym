"""Measure maximum-scale attachment footprints on one real cleared support.

This proves only the specified pair's initial/attached footprint fit. It does
not prove grasp reach, usable navigation edges, the whole pair universe, or a
training template's demo/replay/augmentation gates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from scripts.campaign_20261110.height_dependency_jobs import evidence, measured_probe
from scripts.campaign_20261110.reservation_policy import active_policy
from scripts.campaign_20261110.training_attachment_dependencies import permitted


def run(root,batch,profile_id,probe_path,output):
    root,batch,probe_path,output=map(lambda p:Path(p).resolve(),(root,batch,probe_path,output))
    require(all(p.is_relative_to(root) for p in (batch,probe_path,output)) and not output.exists(),
            'new private layout probe directory required')
    pair=json.loads((batch/'profiles.json').read_text())[profile_id]
    sources=json.loads((batch/'sources.json').read_text())
    source=next(s for s in sources if all(any(o['role']==role and o['category']==ref['category'] and o['model']==ref['model']
        for o in s['configuration']['objects']) for role,ref in pair['objects'].items()))
    refs={role:next(o for o in source['configuration']['objects'] if o['role']==role) for role in ('target','partner')}
    pointer,policy,_=active_policy(root)
    for ref in refs.values():permitted(ref,policy)
    pp=root/'manifests/support_height_profiles.json';profiles=json.loads(pp.read_text())
    probe,binding,support_profile=measured_probe(probe_path,profiles,file_digest(pp),root);permitted(support_profile,policy)
    from scripts.campaign_20261110.bind_fresh_candidates import asset_path
    code=Path(__file__).resolve().parents[2]
    for ref in [*refs.values(),support_profile]:
        require(file_digest(asset_path(code,{'category':ref['category'],'model':ref['model'],
                'dataset_name':'behavior-1k-assets'}))==ref['asset_sha256'],'ready native asset bytes changed')
    stable=Path(probe['stable_scene_path']);scene=json.loads(stable.read_text())
    rooms=scene['objects_info']['init_info'][probe['object']]['args']['in_rooms'];require(rooms,'support has no room')
    output.mkdir(parents=True)
    report=dict(schema='isg-attachment-maximum-layout-probe-v1',campaign_id='campaign_20261110',
        pair_profile=profile_id,pair_profile_manifest=evidence(batch/'profiles.json'),class_id=source['class_id'],
        support_probe=evidence(probe_path),stable_scene=evidence(stable),reservation_policy=pointer,
        maximum_relative_factor=1.2,absolute_scales=pair['maximum_absolute_scales'],
        pair_layout_fit=False,whole_pair_universe_validated=False,usable_edges_validated=False,production_admitted=False,
        removed_chairs=probe.get('removed_chairs',[]),remaining=['remaining category-pair universe maximum layouts',
            'native usable-edge/navigation checks','fresh task native/demo/replay','augmentation and production gates'])
    __import__('ispatialgym.sampling.robot_utils')
    import omnigibson as og
    import torch as th
    from omnigibson.object_states import AttachedTo
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.sim_utils import SimWorld
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.sampling.v2.tasks.attachment import attachment_links
    try:
        variant={'path':str(stable),'removed_objects':probe.get('removed_chairs',[]),'freed_footprints':[],
                 'moved_objects':scene.get('metadata',{}).get('ispatialgym',{}).get('moved_objects',{})}
        world=SimWorld(probe['scene'],rooms,scene_variant=variant,cameras=False)
        support=world.obj(probe['object']);objects={};hulls={}
        for role,ref in refs.items():
            objects[role]=world.spawn('maximum_'+role,ref['category'],ref['model'],
                scale=pair['maximum_absolute_scales'][role],extra_abilities={'attachable':{}})
            require(AttachedTo in objects[role].states,'maximum pair lacks physical attachment state')
            require(all(abs(float(a)-b)<1e-6 for a,b in zip(objects[role].scale,pair['maximum_absolute_scales'][role])),
                    'native maximum scale differs')
            hulls[role]=world.local_hull(objects[role]);require(len(hulls[role])>3,'maximum pair collision hull missing')
        male,female=objects['target'],objects['partner'];links=attachment_links(male,female)
        require(links is not None,'maximum scaled pair has no matching physical attachment links')
        child,parent=links
        cp,cq=child.get_position_orientation();mp,mq=male.get_position_orientation()
        root_in_child=C.relative((mp.float(),mq.float()),(cp.float(),cq.float()))
        pp_,pq=parent.get_position_orientation();fp,fq=female.get_position_orientation()
        parent_in_female=C.relative((pp_.float(),pq.float()),(fp.float(),fq.float()))
        top=world.support_top(support,ignore=list(objects.values()))
        require(abs(top.z-probe['measured_height_world'])<.01,'native free support raster uses another plane')
        cells=top.cells();require(len(cells)>0,'no free support cells')
        cells=cells[::max(1,len(cells)//180)]
        require(file_digest(source['library']['path'])==source['library']['sha256'],'source rest library changed')
        lib=json.loads(Path(source['library']['path']).read_text())
        rests={g['rest_id'] for g in lib['grasps']};require(len(rests)==1,'ambiguous source stable rest')
        male_rest=th.tensor(lib['rests'][next(iter(rests))]['quat_xyzw'],dtype=th.float32)
        # Conservative native footprints use the same 2cm support margin and
        # 4cm pair gap as the attachment sampler. Goal pose uses live links.
        choices=[];tested=0
        for role,rest in [('target',male_rest),('partner',C.IDENTITY_QUAT)]:
            valid=[]
            for xy in cells:
                for yaw in (0.,math.pi/2,math.pi,3*math.pi/2):
                    q=C.qmul(C.quat_from_yaw(yaw),rest)
                    pose=(th.tensor([float(xy[0]),float(xy[1]),top.z+world.bottom_offset(hulls[role],q)+.003]),q)
                    footprint=world.footprint(hulls[role],pose);tested+=1
                    if not top.polygon_supported(footprint,margin=TT.SUPPORT_MARGIN):continue
                    goal=None
                    if role=='partner':
                        goal=C.compose(C.compose(pose,parent_in_female),root_in_child)
                        if not top.polygon_supported(world.footprint(hulls['target'],goal),margin=TT.SUPPORT_MARGIN):continue
                    valid.append((pose,footprint,goal))
            choices.append(valid)
        require(all(choices),'maximum pair has no supported initial or attached footprint')
        selected=None
        for male_pose,male_fp,_ in choices[0]:
            for female_pose,female_fp,goal in choices[1]:
                if float(th.linalg.norm(male_pose[0][:2]-female_pose[0][:2]))<.08:continue
                if TT._polygons_overlap(male_fp,female_fp,.04):continue
                selected=(male_pose,female_pose,goal);break
            if selected:break
        require(selected is not None,'maximum pair cannot fit with required gap and centre separation')
        def pose_json(pose):return {'position':pose[0].tolist(),'orientation':pose[1].tolist()}
        report.update(pair_layout_fit=True,status='specified_maximum_pair_footprints_fit',tested_footprints=tested,
            free_cells=int(top.free.sum()),support_height=top.z,support_occupants=top.occupants,
            measured_hull_extents={role:(h.max(dim=0).values-h.min(dim=0).values).tolist() for role,h in hulls.items()},
            initial_target=pose_json(selected[0]),partner=pose_json(selected[1]),attached_target=pose_json(selected[2]),
            geometric_edges=[{'name':n,'midpoint':m.tolist(),'normal':v.tolist(),'length':float(l)} for n,m,v,l in top.edges()])
    except Exception as exc:
        report.update(status='rejected',error=str(exc),error_type=type(exc).__name__);raise
    finally:
        write_json_atomic(output/'layout_probe.json',report);og.shutdown()
    print(json.dumps({k:report[k] for k in ('class_id','pair_layout_fit','whole_pair_universe_validated','production_admitted')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','batch','probe','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profile-id',required=True);a=p.parse_args()
    run(a.root,a.batch,a.profile_id,a.probe,a.output)
