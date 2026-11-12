"""Bounded recontact height correction using exported native hand geometry."""
import numpy as np


def bounded_lowering(hand_points, rotations, heights, support_tops, *, cap=.002, clearance=.003):
    """Keep every hand hull vertex above the original 3 mm support clearance."""
    points=np.asarray(hand_points,dtype=float); rotations=np.asarray(rotations,dtype=float)
    heights=np.asarray(heights,dtype=float); tops=np.asarray(support_tops,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=3 or len(points)==0 or
            rotations.shape!=(len(heights),3,3) or len(heights)==0 or len(tops)==0 or
            not all(np.isfinite(a).all() for a in (points,rotations,heights,tops))):
        return 0., None
    lowest=min(float(z+np.min(points@r[2])) for z,r in zip(heights,rotations))
    before=lowest-float(tops.max())
    amount=max(0.,min(float(cap),before-float(clearance)))
    return (amount if amount>=.0005 else 0.), before


def recontact_adjustment(arm, poses, supports):
    from ispatialgym.grasping import hand_geometry
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling import campaign_runtime
    if not campaign_runtime.contact_recovery('axial_magazine_contact_height'):
        return 0., None
    geometry=hand_geometry.load(arm)
    if geometry.source!='usd_collision':
        return 0., {'reason':'native hand geometry unavailable'}
    points,_=geometry.points(finger_q=(0.,0.))
    try:
        tops=[float(obj.aabb[1][2]) for obj in supports]
    except (AttributeError,TypeError,ValueError,IndexError):
        return 0., {'reason':'support top unavailable'}
    rotations=[C.quat_to_mat(C.as_tensor(q)).detach().cpu().numpy() for _,q in poses]
    amount,before=bounded_lowering(points,rotations,[float(p[2]) for p,_ in poses],tops)
    return amount, {'lowering_m':amount,'minimum_hand_support_clearance_before_m':before,
                    'minimum_required_clearance_m':.003,'maximum_lowering_m':.002,
                    'native_hand_geometry':geometry.digest,'source':geometry.source}
