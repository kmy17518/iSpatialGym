"""Opt-in proposal screen for inside01's measured pre-grasp descent collision."""
import os
import numpy as np


def enabled():
    if not (os.environ.get('ISPATIALGYM_CAMPAIGN_CONFIG') and
            os.environ.get('ISPATIALGYM_INSIDE01_DESCENT_PROPOSAL', '0') == '1'):
        return False
    from ispatialgym.sampling import campaign_runtime
    return campaign_runtime.configuration()[1].class_id == 'inter_object_relation_satisfaction-inside_01'


def pregrasp_predicate(bounds):
    """Use the same conservative screen while drawing, before losing a stance."""
    from ispatialgym.grasping import hand_geometry as HG
    hands = {}

    def allowed(arm, pre):
        if arm not in hands:
            hands[arm] = HG.load(arm)
        p, q = (value.detach().cpu().numpy() for value in pre)
        return not swept_hand_box(hands[arm], p, q, bounds)['overlap']

    return allowed


def swept_hand_box(hand, position, orientation, bounds, *, rise=.12, displacement=None):
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
    shift_path=np.asarray([0,0,rise] if displacement is None else displacement,dtype=float)
    if shift_path.shape!=(3,) or not np.isfinite(shift_path).all():raise ValueError('Invalid hand sweep displacement')
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
            swept=np.vstack([world,world+shift_path])
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



def swept_hand_pieces(hand, position, orientation, displacement, obstacles):
    """Exact convex native pieces avoid the bowl AABB's measured false positive."""
    from scipy.spatial import ConvexHull
    from scipy.optimize import linprog
    from scipy.spatial.transform import Rotation
    if hand.source!='usd_collision':raise ValueError('Native hand geometry required')
    p=np.asarray(position,dtype=float);q=np.asarray(orientation,dtype=float);delta=np.asarray(displacement,dtype=float)
    bodies=[np.asarray(v,dtype=float)for v in obstacles]
    if p.shape!=(3,) or q.shape!=(4,) or delta.shape!=(3,) or not bodies:raise ValueError('Invalid native approach shape')
    if not all(np.isfinite(v).all()for v in [p,q,delta]+bodies) or abs(np.linalg.norm(q)-1)>1e-5:
        raise ValueError('Invalid native approach geometry')
    rot=Rotation.from_quat(q).as_matrix();hulls=[(v,ConvexHull(v).equations)for v in bodies]
    for link in hand.links:
        shift=np.zeros(3)if link.axis is None else hand.finger_open*np.asarray(link.axis)
        for piece in link.pieces or [link.points]:
            world=(np.asarray(piece)+shift)@rot.T+p;sweep=np.vstack([world,world+delta]);planes=None
            for body,other in hulls:
                if np.any(sweep.min(0)>body.max(0)) or np.any(sweep.max(0)<body.min(0)):continue
                if planes is None:planes=ConvexHull(sweep).equations
                eq=np.vstack([planes,other]);r=linprog([0,0,0,-1],A_ub=np.c_[eq[:,:3],np.ones(len(eq))],
                    b_ub=-eq[:,3],bounds=[(None,None)]*4,method='highs')
                if not r.success:raise RuntimeError('Native approach overlap calculation failed')
                if r.x[3]>1e-7:return True
    return False


def grasp_segment_predicate(partner):
    """Check the entire open-hand approach against native partner geometry."""
    from ispatialgym.grasping import hand_geometry as HG
    from ispatialgym.sampling.campaign_base_hull_screen import native_pieces
    hands={};obstacles=native_pieces(partner)
    def allowed(arm,pre,grasp):
        if arm not in hands:hands[arm]=HG.load(arm)
        p,q=(v.detach().cpu().numpy()for v in pre)
        end,orientation=(v.detach().cpu().numpy()for v in grasp)
        if abs(float(np.dot(q,orientation)))<1-1e-5:
            raise ValueError('Straight grasp approach changed hand orientation')
        return not swept_hand_pieces(hands[arm],p,q,end-p,obstacles)
    return allowed
