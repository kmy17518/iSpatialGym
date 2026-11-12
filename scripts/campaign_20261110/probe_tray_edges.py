"""Bounded native support-edge/navigation/visibility measurements; never writes instances.

This consumes the full eight-model geometry superset. It does not establish the
historical 28-draw selection rule, grasp reach, task program, settling or replay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def verify(request):
    q = read(request)
    assert q['schema'] == 'isg-selected-tray-edge-request-v1' and q['campaign_id'] == 'campaign_20261110'
    assert q['benchmark_credit'] == q['new_native_instances'] == 0 and q['production_authorization'] is False
    for ref in q['inputs'].values():
        assert digest(ref['path']) == ref['sha256'], ref['path']
    source = read(q['inputs']['source_request']['path']); geometry = read(q['inputs']['geometry']['path'])
    assert source['configuration_id'] == geometry['configuration_id'] == q['configuration_id']
    assert geometry['request'] == q['inputs']['source_request']
    assert source['native_runtime'] == geometry['native_runtime'] == q['native_runtime']
    assert len(source['historical_budget_models']) == 8 and source['historical_declared_draw_count'] == 28
    assert len(source['historical_superset_draws']) == len(geometry['draws']) == 56
    assert geometry['superset_draws_fit'] == 56 and not geometry['historical_selected_scope_complete']
    assert all(row['ready'] for row in source['objects'])
    for ref in [source['native_code'], source['configuration'], source['source_metadata'], source['source_customs'],
                source['stable_scene'], source['support_probe'], source['active_binding_pool']]+[r['asset']for r in source['objects']]:
        assert digest(ref['path']) == ref['sha256'], ref['path']
    return q, source, geometry


def run(request, output):
    q, source, geometry = verify(request)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    report = {'schema':'isg-selected-tray-edge-measurement-v1','campaign_id':q['campaign_id'],
        'configuration_id':q['configuration_id'],'request':{'path':str(request),'sha256':digest(request)},
        'native_runtime':q['native_runtime'],'job_id':os.environ.get('SLURM_JOB_ID'),
        'started_at':time.time(),'cpu_affinity':sorted(os.sched_getaffinity(0)), 'state':'starting',
        'benchmark_credit':0,'production_authorization':False,'new_native_instances':0,'edges':[], 'draws':[],
        'historical_selected_scope_complete':False,'complete_configuration_validated':False,
        'remaining_checks':['historical28selection_provenance','native_phase1b_phase2_grasp_queries',
                            'exact_grasp_and_carried_path','settled_layout','demo','own_physics_replay']}
    write(output/'result.json',report)
    sys.path.insert(0,q['native_runtime'])
    from ispatialgym.sampling.robot_utils import RobotKit
    import omnigibson as og
    import torch as th
    import numpy as np
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling.sim_utils import SimWorld,Reject
    from ispatialgym.sampling.constraints import apply_mass
    from ispatialgym.sampling.v2 import tabletop as TT, navigation as NAV
    from ispatialgym.sampling.v2.ik import IKChecker
    from ispatialgym.sampling.v2.stances import Stance
    from ispatialgym.sampling.v2.tasks.disembedding import Disembedding
    from ispatialgym.sampling.knowledge import is_ready
    try:
        assert Path(TT.__file__).resolve().is_relative_to(Path(q['native_runtime']))
        variant = {k:v for k,v in source['scene_variant'].items()if k!='scene_file'}
        variant['path'] = source['stable_scene']['path']
        world = SimWorld(source['scene'],source['rooms'],scene_variant=variant,cameras=True)
        assert digest(world.stable_path) == source['stable_scene']['sha256']
        support = world.obj(source['support'])
        assert np.allclose(support.scale.tolist(),source['support_scale'],rtol=0,atol=1e-7)
        top = world.support_top(support); assert abs(top.z-geometry['support_top_z']) < 1e-5
        objs = {}; hulls = {}
        for row in source['objects']:
            key = row['key']; assert is_ready(row['model'])
            if key not in geometry['objects']: continue
            actual = geometry['objects'][key]
            obj = world.spawn('edge_probe_'+row['category']+'_'+row['model'],row['category'],row['model'],
                              scale=actual['applied_native_scale']); objs[key] = obj
            assert np.allclose(obj.scale.tolist(),actual['applied_native_scale'],rtol=0,atol=1e-6)
            if row['category']=='tray':apply_mass(obj,float(source['generation']['tray_mass_kg']))
        def park_all():
            for obj in objs.values():world.park(obj,list(world.spawned).index(obj.name))
        park_all()
        for key,obj in objs.items():
            hull = world.local_hull(obj); prior = th.tensor(geometry['objects'][key]['hull'])
            assert hull.shape == prior.shape and th.allclose(hull,prior,atol=1e-5,rtol=0), 'measured hull changed '+key
            hulls[key] = hull
        kit = RobotKit(world,prebuild=(),mesh_cache_size=max(128,RobotKit.count_collision_meshes(world)+32))
        kit.sync_world(); ikc = IKChecker(kit,{'scene':world.scene_model,'rooms':world.loaded_rooms,'floor_z':kit.floor_z})
        sampler = SimpleNamespace(world=world,robot=world.robot,kit=kit,ikc=ikc,objects=objs,
            gen=lambda key,default=None:source['generation'].get(key,default),
            g=lambda key:source['generation'][key],stance_q=lambda st:ikc.q_locked(st))
        def camera(st):
            x,y,yaw=st.base_pose2d
            return C.camera_pose(ikc.q_locked(st),base_pose=(th.tensor([x,y,kit.floor_z+C.NOMINAL_BASE_Z]),G.quat_from_yaw(yaw)),cam_in_zed=kit.cam_in_zed)
        sampler.camera_pose_at = camera
        pose = lambda value:tuple(th.tensor(v)for v in value)
        src_key,dst_key,target_key='tray/uekqey','tray/hvlfig','cellulose_tape/kavsnx'
        pair=geometry['two_tray_footprint_fit']['pose']; sp,dp=pose(pair['source']),pose(pair['destination'])
        for key,pp in [(src_key,sp),(dst_key,dp)]:
            assert top.polygon_supported(world.footprint(hulls[key],pp),margin=TT.SUPPORT_MARGIN)
        assert not TT._polygons_overlap(world.footprint(hulls[src_key],sp),world.footprint(hulls[dst_key],dp),float(source['generation']['destination_gap_min_m']))
        report.update(state='measuring',support_scale=support.scale.tolist(),support_top_z=top.z,
                      layout_pose_source=geometry['two_tray_footprint_fit'],hulls_reverified=True)
        deadline=time.monotonic()+q['maximum_measurement_seconds']; usable=[]
        parked=NAV.park_pose(kit,source['rooms'],keep_away=[support]); centre=(sp[0][:2]+dp[0][:2])/2
        for edge,baseline in TT.edge_frames(sampler,top):
            edge_result={'edge':edge.to_json(),'candidates':[]}; report['edges'].append(edge_result)
            half=max(0.,edge.length/2-.30);centre_lat=float(th.dot(centre-edge.mid,edge.tangent))
            lats=sorted(set(round(max(-half,min(half,x)),4)for x in [0.,centre_lat,centre_lat-.25,centre_lat+.25]))
            for lat in lats:
                xy=edge.base_xy(TT.STAND_OFF,lat)
                for tr in (0.,.25,.5,.75,.95):
                    if time.monotonic()>deadline:break
                    st=Stance('support_edge',(float(xy[0]),float(xy[1]),edge.yaw),tr,0.,TT.STAND_OFF,edge,lateral=lat)
                    row={'stance':st.to_sidecar(),'free_floor':bool(kit.trav_free(xy))};edge_result['candidates'].append(row)
                    if not row['free_floor']:continue
                    row['whole_body_free']=ikc.stance_collision_free(st)
                    if not row['whole_body_free']:continue
                    row['trunk_sweep_free']=TT.trunk_sweep_free(sampler,st)
                    if not row['trunk_sweep_free']:continue
                    cam=camera(st);row['both_trays_inside_margin']=all(TT.in_frame(sampler,cam,G.local_points_world(hulls[key],pp),float(source['generation']['image_margin']))for key,pp in [(src_key,sp),(dst_key,dp)])
                    if not row['both_trays_inside_margin']:continue
                    start=Stance('room_park',(float(parked[0]),float(parked[1]),edge.yaw),tr,0.)
                    try:
                        path=NAV.plan_path(kit,start,st,rooms=source['rooms']);row['navigation']={'passed':True,'path':path.to_sidecar()}
                        usable.append(st)
                    except NAV.NavigationError as exc:row['navigation']={'passed':False,'reason':str(exc)}
                    write(output/'result.json',report)
            write(output/'result.json',report)
        report['navigation_visible_stances']=len(usable)
        for draw in geometry['draws']:
            if time.monotonic()>deadline:report['time_bound_reached']=True;break
            park_all(); placements={src_key:sp,dst_key:dp,target_key:G.compose(sp,pose(draw['target_pose']))}
            placements.update({key:G.compose(sp,pose(pp))for key,pp in draw['distractor_poses'].items()})
            row={'index':draw['index'],'models':draw['models'],'active_alias_draw':draw['active_alias_draw'],'attempts':[],'status':'no_verified_view'}
            for st in usable[:8]:
                cam=camera(st)
                if not all(TT.in_frame(sampler,cam,G.local_points_world(hulls[k],pp),float(source['generation']['image_margin']))for k,pp in placements.items()):continue
                attempt={'stance':st.to_sidecar()};row['attempts'].append(attempt)
                try:
                    occ=Disembedding._measure_occlusion(sampler,st,placements,target_key,draw['models'],dst_key)
                    kit.sync_world();attempt['whole_body_with_layout']=ikc.stance_collision_free(st)
                    if not attempt['whole_body_with_layout']:continue
                    start=Stance('room_park',(float(parked[0]),float(parked[1]),st.yaw),st.translate,0.)
                    path=NAV.plan_path(kit,start,st,rooms=source['rooms'])
                    attempt['navigation_with_layout']={'passed':True,'path':path.to_sidecar()}
                    attempt['native_occlusion']=occ;row['status']='native_frame_occlusion_and_stance_passed';break
                except (Reject,NAV.NavigationError) as exc:attempt['rejection']=str(exc)
            report['draws'].append(row);write(output/f'draw_{draw["index"]:03d}.json',row);write(output/'result.json',report)
            print(json.dumps({'draw':draw['index'],'status':row['status']}),flush=True)
        report.update(state='measurement_complete',draws_checked=len(report['draws']),
            draws_passed=sum(x['status']=='native_frame_occlusion_and_stance_passed'for x in report['draws']),
            geometry_superset_count=56,historical_selected_draw_count=28,finished_at=time.time())
    except BaseException as exc:
        report.update(state='failed',error=repr(exc),traceback=traceback.format_exc(),finished_at=time.time());raise
    finally:
        write(output/'result.json',report);print('TRAY_EDGE_MEASUREMENT_RECORDED',report['state'],flush=True);og.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:verify(a.request);print('exact tray-edge inputs verified')
    elif a.output:run(a.request,a.output)
    else:p.error('--output is required for native measurement')
