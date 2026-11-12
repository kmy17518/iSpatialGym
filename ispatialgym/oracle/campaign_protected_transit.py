"""Opt-in geometry-derived pre-grasp waypoint; native trajectory checks remain authoritative."""
import os
import numpy as np


def enabled(context):
    if not os.environ.get('ISPATIALGYM_CAMPAIGN_CONFIG') or context.get('leg') != 'pre_grasp':
        return False
    if os.environ.get('ISPATIALGYM_PROTECT_ABOVE_GEOMETRY', '0') == '1':
        return True
    from ispatialgym.sampling import campaign_runtime
    return campaign_runtime.contact_recovery('cap_geometry_above_pregrasp')


def above_offset(position, orientation, hand_points, protected_aabbs, *, baseline, clearance):
    """Put the lowest hand hull point above every protected object's top.

    This proposes an endpoint only. It is not proof that the full arm can reach
    it or travel there without contact; all ordinary IK/path/runtime checks run.
    """
    p=np.asarray(position,dtype=float); q=np.asarray(orientation,dtype=float)
    hand=np.asarray(hand_points,dtype=float)
    if p.shape!=(3,) or q.shape!=(4,) or hand.ndim!=2 or hand.shape[1]!=3 or not len(hand):
        raise ValueError('invalid protected-transit geometry shape')
    if not all(np.isfinite(x).all() for x in (p,q,hand)) or abs(np.linalg.norm(q)-1)>1e-5:
        raise ValueError('invalid protected-transit geometry values')
    if not np.isfinite([baseline,clearance]).all() or baseline<0 or clearance<0:
        raise ValueError('invalid protected-transit clearance')
    q=q/np.linalg.norm(q); x,y,z,w=q
    zrow=np.array([2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)])
    lowest=float((hand@zrow+p[2]).min()); required=float(baseline); objects=[]
    for name,bounds in protected_aabbs:
        box=np.asarray(bounds,dtype=float)
        if box.shape!=(2,3) or not np.isfinite(box).all() or np.any(box[1]<box[0]):
            raise ValueError('invalid protected object AABB: '+name)
        rise=max(float(baseline),float(box[1,2])+float(clearance)-lowest)
        required=max(required,rise)
        objects.append({'object':name,'top_z':float(box[1,2]),'required_offset_m':rise})
    return required,{'baseline_offset_m':float(baseline),'proposed_offset_m':required,
        'hand_lowest_z_at_goal':lowest,'clearance_m':float(clearance),'objects':objects,
        'scope':'waypoint proposal only; full-arm IK, protected path and runtime checks unchanged'}
