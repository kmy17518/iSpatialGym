"""Exact native base hull proposal screen for the measured nextto01 chair route."""
import math
import numpy as np


def first_close(base_pieces, poses, base_z, obstacles, *, insphere_margin=.005):
    """Reject near/intersecting convex pieces at the checked navigation poses.

    A negative common insphere radius measures separation of the two native
    convex pieces. The margin is conservative proposal screening, not a task
    tolerance or a replacement for whole-robot collision checks.
    """
    from scipy.spatial import ConvexHull
    from scipy.optimize import linprog
    bases=[np.asarray(v,dtype=float)for v in base_pieces]
    bodies=[np.asarray(v,dtype=float)for v in obstacles]
    if not bases or not bodies or not poses or not math.isfinite(base_z):
        raise ValueError('Missing base route geometry')
    if any(v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all()for v in bases+bodies):
        raise ValueError('Invalid native collision vertices')
    if not math.isfinite(insphere_margin) or not 0<=insphere_margin<=.01:
        raise ValueError('Invalid native hull screen margin')
    hulls=[(v,ConvexHull(v).equations)for v in bodies]
    for index,pose in enumerate(poses):
        x,y,yaw=(float(v)for v in pose)
        if not all(math.isfinite(v)for v in (x,y,yaw)):raise ValueError('Invalid base route pose')
        c,s=math.cos(yaw),math.sin(yaw);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        for bi,local in enumerate(bases):
            world=local@rot.T+[x,y,base_z];planes=None
            for oi,(body,other)in enumerate(hulls):
                gap=np.maximum(world.min(0)-body.max(0),body.min(0)-world.max(0)).max()
                if gap>2*insphere_margin:continue
                if planes is None:planes=ConvexHull(world).equations
                eq=np.vstack([planes,other]);result=linprog([0,0,0,-1],A_ub=np.c_[eq[:,:3],np.ones(len(eq))],
                    b_ub=-eq[:,3],bounds=[(None,None)]*4,method='highs')
                if not result.success:raise RuntimeError('Native base hull overlap calculation failed')
                if result.x[3]>-insphere_margin:
                    return {'pose_index':index,'base_pose2d':[x,y,yaw],'base_piece':bi,'obstacle_piece':oi,
                            'common_insphere_radius_m':float(result.x[3]),'insphere_margin_m':insphere_margin}
    return None


def native_pieces(obj, link_names=None):
    """Read actual USD collision meshes with their world scale and transforms."""
    from omnigibson.utils.usd_utils import mesh_prim_shape_to_trimesh_mesh
    import omnigibson.utils.transform_utils as T
    pieces=[]
    for name,link in obj.links.items():
        if link_names is not None and name not in link_names:continue
        for mesh in link.collision_meshes.values():
            p,q=mesh.get_position_orientation()
            points=(mesh.points.detach().cpu().numpy()if mesh.geom_type=='Mesh'
                    else np.asarray(mesh_prim_shape_to_trimesh_mesh(mesh.prim).vertices))
            scale=mesh.get_world_scale().detach().cpu().numpy()
            rot=T.quat2mat(q).detach().cpu().numpy()
            pieces.append((np.asarray(points,dtype=float)*scale)@rot.T+p.detach().cpu().numpy())
    if not pieces:raise ValueError('Native collision meshes unavailable')
    return pieces


def screen_record(sampler,record):
    from ispatialgym.sampling import campaign_runtime
    if not campaign_runtime.contact_recovery('nextto_native_base_hulls'):return None
    from ispatialgym.sampling.v2 import navigation as NAV
    from ispatialgym.robot import conventions as C
    from dataclasses import replace
    robot=sampler.robot
    chair=[o for o in sampler.world.scene.objects if o.name=='eames_chair_hpwusn_1']
    if len(chair)!=1:raise ValueError('Audited nextto chair missing or duplicated')
    p,q=robot.get_position_orientation();p=p.detach().cpu().numpy();rot=C.quat_to_mat(q).detach().cpu().numpy()
    base=[(v-p)@rot for v in native_pieces(robot,{'base_link'})]
    # Keep the original path/schedule. Denser verification poses do not add
    # waypoints or change the controller's execution.
    poses=NAV.record_sweep_poses(replace(record,stride_m=min(float(record.stride_m or .05),.02)))
    result=first_close(base,poses,float(sampler.kit.floor_z)+C.NOMINAL_BASE_Z,native_pieces(chair[0]))
    return result
