"""Fail-closed native visibility screening for an audited campaign recovery."""
import math


CLASS_ID = 'inter_object_relation_satisfaction-under_small_scale_04'
TASK = 'inter_object_relation_satisfaction-under_small_scale'
OPTION = 'native_goal_occlusion_screen'
AUDITED = {CLASS_ID: TASK,
    'inter_object_relation_satisfaction-ontop_large_scale_02':
        'inter_object_relation_satisfaction-ontop_large_scale',
    'inter_object_relation_satisfaction-inside_02':
        'inter_object_relation_satisfaction-inside_small_scale',
    'inter_object_relation_satisfaction-touching_03':
        'inter_object_relation_satisfaction-touching_small_scale',
    'inter_object_relation_satisfaction-under_large_scale_02':
        'inter_object_relation_satisfaction-under_large_scale'}


def required(template=None):
    if template is None:
        from ispatialgym.sampling import campaign_runtime
        if not campaign_runtime.active():
            return False
        template = campaign_runtime.configuration()[1]
    value = template.scene_setup.get('recovery_options', {}).get(OPTION, False)
    if type(value) is not bool:
        raise ValueError('native goal occlusion requirement must be boolean')
    if value and AUDITED.get(template.class_id) != template.task:
        raise ValueError('native goal occlusion recovery belongs to an audited visibility-recovery class')
    return value


def preflight(*, enabled, exempt, judged, checked_program, camera):
    missing = [name for name, ok in [('enabled', enabled), ('nonexempt', not exempt),
               ('judged objects', judged), ('checked final program', checked_program), ('camera', camera)] if not ok]
    if missing:
        raise ValueError('required native goal occlusion unavailable: ' + ', '.join(missing))


def verify_report(report):
    """The original screen's measured result must exist; its threshold is unchanged."""
    objects = report.get('objects') or {}
    if report.get('error') or not objects or type(report.get('occluded')) is not bool:
        raise ValueError('required native goal occlusion has no measured result')
    for name, obj in objects.items():
        value = obj.get('occluded')
        if (obj.get('status') != 'ok' or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('required native goal occlusion is unmeasured for ' + name)
