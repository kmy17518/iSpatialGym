"""Strict candidate-selection policies for measured large-object failures."""
from scripts.campaign_20261110.contracts import require


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
