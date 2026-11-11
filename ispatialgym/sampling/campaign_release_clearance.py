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
