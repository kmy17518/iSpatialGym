"""Strict candidate-selection policies for measured large-object failures."""
import os

from scripts.campaign_20261110.contracts import require


SOAP_LIFT_FRAME_ENV = 'ISPATIALGYM_CAMPAIGN_SOAP_LIFT_FRAME_SCREEN'


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
