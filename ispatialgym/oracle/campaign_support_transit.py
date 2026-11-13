"""Exact canonical touching pre-grasp support policy; physical gates unchanged."""
import os


def required(context):
    if os.environ.get('ISPATIALGYM_TOUCHING_PREGRASP_SUPPORT_PRESENT', '0') != '1':
        return False
    if context.get('leg') != 'pre_grasp' or context.get('action') != 'grasp':
        return False
    from ispatialgym.sampling import campaign_runtime as CR
    from scripts.campaign_20261110.contracts import require
    require(CR.active(), 'support-preserving transit requires its campaign source')
    _, template, _, _ = CR.configuration()
    require((template.class_id, template.task, template.split) == (
        'inter_object_relation_satisfaction-touching_01',
        'inter_object_relation_satisfaction-touching_small_scale', 'eval_standard'),
        'support-preserving transit belongs to the audited touching class')
    expected = {
        'target': ('box_of_sanitary_napkins', 'humkqw', (1., 1., 1.)),
        'partner': ('box_of_tissues', 'ntbrtz', (1., 1., 1.)),
    }
    require(all([(o.category, o.model, tuple(o.default_scale)) for o in template.objects if o.role == role] == [spec]
                for role, spec in expected.items()),
            'support-preserving transit requires the exact canonical actor models and scales')
    require(any(o.category == 'breakfast_table' and o.model == 'rhjoby' and o.role == 'support'
                for o in template.objects), 'support-preserving transit requires the original table')
    return True
