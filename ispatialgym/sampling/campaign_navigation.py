"""Scoped navigation corrections backed by recorded campaign trajectories."""

from scripts.campaign_20261110.contracts import require


def terminal_deceleration():
    """Brake the audited trash-can empty-handed approach before its endpoint.

    The original native line is preserved. Source1375's base followed that line
    within 2.8 mm but crossed its checked endpoint by 32.7 mm into a wall.
    The existing 0.2 m/s² corner-stop reference profile avoids the same
    overshoot in a tracker reproduction without changing controller limits.
    """
    from ispatialgym.sampling import campaign_runtime as runtime
    if not runtime.active():
        return None
    _, template, _, _ = runtime.configuration()
    enabled = template.scene_setup.get('recovery_options', {}).get('navigation_terminal_braking', False)
    require(type(enabled) is bool, 'navigation terminal braking option must be boolean')
    if not enabled:
        return None
    require(template.class_id == 'inter_object_relation_satisfaction-under_large_scale_03'
            and template.task == 'inter_object_relation_satisfaction-under_large_scale',
            'navigation terminal braking belongs to the audited trash-can class')
    return .2
