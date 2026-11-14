"""Native release geometry for the measured kielbasa/platter recovery."""


CLASS_ID = 'inter_object_relation_satisfaction-ontop_small_scale_01'
TASK = 'inter_object_relation_satisfaction-ontop_small_scale'
OPTION = 'native_ontop_release_clearance'
POLICY = 'native_ontop_clearance'


def settings(template=None):
    if template is None:
        from ispatialgym.sampling import campaign_runtime
        if not campaign_runtime.active():
            return None
        template = campaign_runtime.configuration()[1]
    enabled = template.scene_setup.get('recovery_options', {}).get(OPTION, False)
    if type(enabled) is not bool:
        raise ValueError('native ontop release clearance must be boolean')
    if not enabled:
        return None
    if (template.class_id, template.task) != (CLASS_ID, TASK):
        raise ValueError('native ontop release clearance belongs to the audited kielbasa class')
    # The same bounded native open-jaw clearance used by raised nextto releases.
    return {'partner_clearance_m': .005, 'max_raise_m': .030, 'step_m': .001}


def fresh_goal_settings(plan=None):
    """The same clearance for the online worker's fresh-goal first candidate.

    A persistent online worker draws its first candidate before any fixed-goal
    plan exists; that candidate's measured raised release is what the plan is
    then built from (``goal_groups.prepare_group``). Only that situation is
    covered: the online-groups worker is active, no fixed-goal plan is loaded
    or exported, and the audited class enables the option (``settings``).
    """
    import os
    from ispatialgym.sampling.online_groups import ENV, PLAN_ENV
    if plan is not None or not os.environ.get(ENV) or os.environ.get(PLAN_ENV):
        return None
    return settings()
