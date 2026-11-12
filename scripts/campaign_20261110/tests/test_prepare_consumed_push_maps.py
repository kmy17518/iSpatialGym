from copy import deepcopy
from scripts.campaign_20261110.prepare_consumed_push_maps import selected_dependency


def test_required_push_selection_preserves_model_scale_arm_and_consumer():
    obj={'category':'magazine','model':'one','scale':[1.01]*3}
    dep={'kind':'contact_push_map','required_by_classes':['alignment-axial_02'],
         'inputs':{'arm':'right','target':{'category':'magazine','model':'one','applied_scale':[1.01]*3}}}
    assert selected_dependency(dep,obj)
    for change in ['scale','arm','consumer','kind']:
        wrong=deepcopy(dep)
        if change=='scale':wrong['inputs']['target']['applied_scale']=[1]*3
        if change=='arm':wrong['inputs']['arm']='left'
        if change=='consumer':wrong['required_by_classes']=['alignment-world_vertical_02']
        if change=='kind':wrong['kind']='contact_brace_map'
        assert not selected_dependency(wrong,obj)
