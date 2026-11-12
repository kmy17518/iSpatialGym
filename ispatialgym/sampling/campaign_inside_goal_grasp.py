"""Keep a bounded native pick proposal only when its fixed inside goal is reachable."""
import copy
import math


def goal_stance_order(order, candidates, anchor):
    """Rank existing checked stances by the measured goal-side robot pose.

    The old source's pick focus need not suit a goal-compatible grasp. Preserve
    fresh target draws at each native candidate's own focus and all later gates.
    """
    if not anchor:
        return order
    def distance(index):
        st = candidates[int(index)]
        yaw = float(st.base_pose2d[2]) - float(anchor['base_pose2d'][2])
        return (math.dist([float(v) for v in st.base_pose2d[:2]], anchor['base_pose2d'][:2])
                + .3 * abs(math.atan2(math.sin(yaw), math.cos(yaw)))
                + abs(float(st.translate) - float(anchor['translate']))
                + abs(float(st.tilt) - float(anchor['tilt'])))
    return sorted((int(i) for i in order), key=lambda i: (distance(i), i))


def fixed_goal_grasp_predicate(s):
    from ispatialgym.sampling.v2.tasks import iors_small as I
    from ispatialgym.sampling.sim_utils import Reject
    plan = s._campaign_pair_active
    partner = tuple(I.G.t(v) for v in plan.partner)
    cache = {}

    def allowed(stance, target_pose, grasp):
        key = (tuple(float(v) for v in stance.base_pose2d), float(stance.translate), float(stance.tilt),
               grasp.arm, grasp.grasp_name, tuple(grasp.finger_q or ()),
               tuple(float(v) for part in grasp.eef_in_object for v in part),
               str(stance.arms), str(stance.idle_posture))
        if key in cache:
            return cache[key]
        stances, grasps = dict(s.sidecar.stances), list(s.sidecar.grasps)
        try:
            candidate = copy.copy(grasp)
            candidate.place_on = plan.partner_name
            s.sidecar.add_stance(stance)
            s.sidecar.grasps.clear()
            s.sidecar.add_grasp(candidate)
            goal = s.goal_pose(target_pose, partner)
            place = I.IK.grasp_pose(goal, grasp.eef_in_object)
            pre = s.pre_place_of(place)
            with I.attached(s.kit, grasp.arm, s.obj('target'), eef_in_object=grasp.eef_in_object):
                result = s.ikc.exact([pre, place], stance, grasp.arm, ignore_for_last=s.support_objects(),
                                     scale_key=s.choice.models['target'].scale, cache=False, approached=[])
            cache[key] = bool(result.ok)
        except Reject:
            cache[key] = False
        finally:
            s.sidecar.stances.clear(); s.sidecar.stances.update(stances)
            s.sidecar.grasps[:] = grasps
        return cache[key]

    return allowed
