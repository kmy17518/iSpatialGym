"""Opt-in proposal screen for inside01's measured pre-grasp descent collision."""
import os
import numpy as np


def enabled():
    if not (os.environ.get('ISPATIALGYM_CAMPAIGN_CONFIG') and
            os.environ.get('ISPATIALGYM_INSIDE01_DESCENT_PROPOSAL', '0') == '1'):
        return False
    from ispatialgym.sampling import campaign_runtime
    return campaign_runtime.configuration()[1].class_id == 'inter_object_relation_satisfaction-inside_01'


def swept_hand_box(hand, position, orientation, bounds, *, rise=.12):
    """Conservative continuous vertical sweep of native hand convex pieces.

    The protected object's AABB deliberately overapproximates its geometry.
    This rejects a start proposal only; it cannot certify the full robot path.
    Full-open fingers match the measured pre-grasp transit. Existing native
    program, contact, clearance and replay checks remain authoritative.
    """
    from scipy.spatial import ConvexHull
    from scipy.optimize import linprog
    p=np.asarray(position,dtype=float);q=np.asarray(orientation,dtype=float)
    box=np.asarray(bounds,dtype=float)
    if hand.source!='usd_collision':raise ValueError('Native USD hand collision geometry required')
    if p.shape!=(3,) or q.shape!=(4,) or box.shape!=(2,3):raise ValueError('Invalid descent geometry shape')
    if not all(np.isfinite(x).all()for x in (p,q,box)) or np.any(box[0]>box[1]) or abs(np.linalg.norm(q)-1)>1e-5:
        raise ValueError('Invalid descent geometry')
    if not np.isfinite(rise) or rise<0:raise ValueError('Invalid descent rise')
    x,y,z,w=q/np.linalg.norm(q)
    rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                       [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                       [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    box_planes=np.vstack([np.column_stack([np.eye(3),-box[1]]),np.column_stack([-np.eye(3),box[0]])])
    checked=0
    for link in hand.links:
        shift=np.zeros(3) if link.axis is None else hand.finger_open*np.asarray(link.axis)
        for index,piece in enumerate(link.pieces or [link.points]):
            world=(np.asarray(piece)+shift)@rotation.T+p
            swept=np.vstack([world,world+[0,0,rise]])
            if np.any(swept.min(0)>box[1]) or np.any(swept.max(0)<box[0]):continue
            checked+=1
            planes=np.vstack([ConvexHull(swept).equations,box_planes])
            result=linprog([0,0,0,-1],A_ub=np.column_stack([planes[:,:3],np.ones(len(planes))]),
                           b_ub=-planes[:,3],bounds=[(None,None)]*4,method='highs')
            if not result.success:raise RuntimeError('Descent convex overlap calculation failed: '+result.message)
            if result.x[3]>1e-7:
                return {'overlap':True,'link':link.link,'piece':index,
                        'overlap_insphere_m':float(result.x[3]),'checked_pairs':checked,
                        'rise_m':rise,'geometry_digest':hand.digest}
    return {'overlap':False,'checked_pairs':checked,'rise_m':rise,'geometry_digest':hand.digest}
