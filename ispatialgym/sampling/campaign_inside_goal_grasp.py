"""Keep a bounded native pick proposal only when its fixed inside goal is reachable."""
import copy


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
