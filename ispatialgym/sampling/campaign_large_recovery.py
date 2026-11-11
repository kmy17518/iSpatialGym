"""Strict candidate-selection policies for measured large-object failures."""
import os
import copy

from scripts.campaign_20261110.contracts import require


SOAP_LIFT_FRAME_ENV = 'ISPATIALGYM_CAMPAIGN_SOAP_LIFT_FRAME_SCREEN'


def candidate_program_screen(plan=None):
    """Run the unchanged final check while audited alternatives remain available."""
    from ispatialgym.sampling import campaign_runtime as runtime
    if not runtime.active():
        return False
    _, template, _, _ = runtime.configuration()
    enabled = template.scene_setup.get('recovery_options', {}).get('heel_candidate_program_screen', False)
    require(type(enabled) is bool, 'heel candidate program screen must be boolean')
    if enabled:
        require(template.class_id == 'inter_object_relation_satisfaction-under_large_scale_01'
                and template.task == 'inter_object_relation_satisfaction-under_large_scale',
                'heel candidate program screen belongs to the audited sandal class')
        return True
    proposal = getattr(plan, 'floor_push_source_proposal', None)
    if proposal is not None:
        require(template.class_id == 'inter_object_relation_satisfaction-under_large_scale_02'
                and template.task == 'inter_object_relation_satisfaction-under_large_scale'
                and isinstance(proposal, dict),
                'floor push candidate screen belongs to the audited briefcase plan')
        return True
    return False


def candidate_program_rejection(sampler, draft, plan=None):
    """Screen a complete candidate without retaining provisional witness edits.

    Both audited classes repeatedly reached the late checker only after the
    stance/grasp loop had ended. Reuse that checker on an isolated sidecar so a
    failed retreat leaves the remaining native candidates searchable. Its
    world-state restoration and every original rejection remain in force.
    """
    if not candidate_program_screen(plan):
        return None
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2 import program_check

    # sample_instance precedes core.apply_draft and goal(). In particular,
    # program_check._SamplerState reads live object poses; checking here with
    # a parked target invents behind-camera grasp targets. Install this draft
    # only for the kinematic pre-screen, without activating bodies or settling.
    # The saved whole-world state also protects against the checker's attachment
    # helpers stepping physics. All post-settle/final gates still run normally.
    from ispatialgym.sampling import geometry as G
    import torch

    original = sampler.sidecar
    state = sampler.world.snapshot()
    sampler.sidecar = copy.deepcopy(original)
    provisional = copy.deepcopy(draft)
    try:
        for name, pose in provisional.object_poses.items():
            obj = sampler.objects[name]
            obj.set_position_orientation(G.t(pose[0]), G.t(pose[1]))
            obj.keep_still()
        for name, q in provisional.object_joints.items():
            obj = sampler.objects[name]
            obj.set_joint_positions(G.t(q))
            obj.set_joint_velocities(torch.zeros_like(G.t(q)))
            obj.keep_still()
        sampler.kit.apply(provisional.robot_q, provisional.base_pose2d)
        sampler.goal(provisional)
        sampler.kit.sync_world()
        with sampler._floor_relaxed():
            program_check.run(sampler.gp.variant, sampler.sidecar, sampler,
                              subtask=sampler.subtask)
        sampler.heel_tight_left_j4_check()
    except Reject as exc:
        return f'{exc.reason}: {exc}'
    finally:
        sampler.sidecar = original
        sampler.world.restore(state)
        sampler.kit.sync_world()
    return None


def heel_carry_path_screen():
    """Check the sandal's intermediate carry, not only its endpoint witnesses."""
    from ispatialgym.sampling import campaign_runtime as runtime
    if not runtime.active():
        return False
    _, template, _, _ = runtime.configuration()
    enabled = template.scene_setup.get('recovery_options', {}).get('heel_carry_path_screen', False)
    require(type(enabled) is bool, 'heel carry path option must be boolean')
    if enabled:
        require(template.class_id == 'inter_object_relation_satisfaction-under_large_scale_01'
                and template.task == 'inter_object_relation_satisfaction-under_large_scale',
                'heel carry path screen belongs to the audited sandal class')
    return enabled


def heel_carry_path_failure(sampler, arm, margin):
    """Apply the existing heel policy to every tracked left-arm configuration.

    Source 50265800's endpoints stayed 110 mrad from the limit while its
    commanded intermediate path came within 12 mrad and hit link3 against
    link5. The margin is supplied by the unchanged heel-tight policy.
    """
    if arm != 'left' or not heel_carry_path_screen() or sampler.push_heel_tight() is None:
        return None
    trajectory = getattr(sampler.ikc, 'last_track_traj', None)
    require(trajectory is not None and len(trajectory) > 0, 'heel carry path has no tracked configurations')
    robot = sampler.kit.robot
    joint = int(robot.arm_control_idx['left'][3])
    lower = float(robot.joint_lower_limits[joint])
    measured, index = min((float(q[joint]) - lower, i) for i, q in enumerate(trajectory))
    if measured < margin:
        return (f'left_arm_joint4 carry path {measured * 1000:.1f} mrad from its lower limit '
                f'at tracked configuration {index} (< {margin * 1000:.0f} mrad heel-tight policy)')
    return None


def soap_carry_oracle_screen():
    """Use the final program's continuation bounds during soap candidate search."""
    from ispatialgym.sampling import campaign_runtime as runtime
    if not runtime.active():
        return False
    _, template, _, _ = runtime.configuration()
    enabled = template.scene_setup.get('recovery_options', {}).get('soap_carry_oracle_screen', False)
    require(type(enabled) is bool, 'soap carry screen option must be boolean')
    if enabled:
        require(template.class_id == 'inter_object_relation_satisfaction-ontop_large_scale_01'
                and template.task == 'inter_object_relation_satisfaction-ontop_large_scale',
                'soap carry screen belongs to the audited soap class')
    return enabled


def soap_lift_frame_rejection(sampler, stance, grasp, lift):
    """Try another lift candidate before a known late fingertip-frame failure.

    Ready-countertop source 1347 lost two of three complete native draws at
    the program check's lift frame gate. Reuse that exact projection and its
    margins before selecting the lift. The final program check still runs.
    """
    enabled = os.environ.get(SOAP_LIFT_FRAME_ENV, '0')
    require(enabled in ('0', '1'), 'soap lift frame option must be 0 or 1')
    if enabled == '0':
        return None
    require(sampler.gp.variant in {
                'inter_object_relation_satisfaction-ontop_large_scale-soap_dispenser-countertop-floor',
                'inter_object_relation_satisfaction-ontop_large_scale-soap_dispenser-countertop-floors',
            },
            'soap lift frame recovery belongs to the audited soap class')
    from ispatialgym.sampling.v2.program_check import fingertips_in_frame

    return fingertips_in_frame(sampler, stance, 'lift', [('lift', [grasp, lift])])


def pillow_grasp_margin():
    """Demand 15 mm of reach beyond the pillow grasp, retaining its original pose.

    The native source rejected its approaches after small settling offsets left
    the wrist beyond reach. This selects more robust candidates; all original
    IK, support, collision, tracking, demo and replay checks still run.
    """
    from ispatialgym.sampling import campaign_runtime as runtime
    if not runtime.active():
        return None
    _, template, _, _ = runtime.configuration()
    enabled = template.scene_setup.get('recovery_options', {}).get('pillow_grasp_margin', False)
    require(type(enabled) is bool, 'pillow grasp recovery option must be boolean')
    if not enabled:
        return None
    require(template.class_id == 'inter_object_relation_satisfaction-ontop_large_scale_03'
            and template.task == 'inter_object_relation_satisfaction-ontop_large_scale',
            'pillow grasp recovery belongs to the audited pillow class')
    return 0.015
