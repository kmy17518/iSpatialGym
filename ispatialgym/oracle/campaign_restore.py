"""Explicit canary-only restoration of serialized task root motion."""
from __future__ import annotations
import math
from scripts.campaign_20261110.contracts import require


def vector(value, label):
    require(isinstance(value,(list,tuple)) and len(value)==3 and
            all(type(v) in (float,int) and math.isfinite(v) for v in value),
            'serialized root motion missing or invalid: '+label)
    return [float(v) for v in value]


def restore_serialized_root_motion(meta, physical, object_states):
    option=meta.get('scene_setup',{}).get('restoration_options',{}).get('serialized_root_motion_v1')
    if option is None or option is False:
        return None
    require(option is True and meta.get('campaign_id')=='campaign_20261110' and meta.get('split')=='train'
            and meta.get('class_id')=='disembedding_07' and meta.get('task')=='disembedding'
            and meta.get('runtime',{}).get('purpose')=='training_pipeline_canary',
            'serialized root motion is restricted to its exact bounded canary scope')
    import torch
    # Validate every serialized field before mutating any physical object.
    pending=[];skipped=[]
    for instance,state in object_states.items():
        obj=physical.get(instance);require(obj is not None,'serialized task object absent: '+instance)
        if bool(getattr(obj,'kinematic_only',False)):
            skipped.append(instance);continue
        root=state.get('root_link') or {}
        linear=vector(root.get('lin_vel'),instance+'.lin_vel')
        angular=vector(root.get('ang_vel'),instance+'.ang_vel')
        pending.append((instance,obj,linear,angular))
    require(pending,'no serialized dynamic root motion to restore')
    observations={}
    for instance,obj,linear,angular in pending:
        before={'linear_velocity':obj.get_linear_velocity().tolist(),
                'angular_velocity':obj.get_angular_velocity().tolist()}
        obj.set_linear_velocity(torch.tensor(linear,dtype=torch.float32))
        obj.set_angular_velocity(torch.tensor(angular,dtype=torch.float32))
        after={'linear_velocity':obj.get_linear_velocity().tolist(),
               'angular_velocity':obj.get_angular_velocity().tolist()}
        for key,expected in [('linear_velocity',linear),('angular_velocity',angular)]:
            require(len(after[key])==3 and all(abs(a-b)<=1e-7*max(1.,abs(b)) for a,b in zip(after[key],expected)),
                    'serialized root motion readback differs: '+instance+'.'+key)
        observations[instance]={'serialized':{'linear_velocity':linear,'angular_velocity':angular},
                                'before':before,'after':after}
    return {'revision':'serialized_root_motion_v1','boundary':'complete_native_loader_before_actions',
            'objects':observations,'kinematic_objects_unchanged':skipped,'physics_steps_added':0,
            'pose_or_goal_changes':False,'source':'exact immutable TRO root_link lin_vel/ang_vel'}
