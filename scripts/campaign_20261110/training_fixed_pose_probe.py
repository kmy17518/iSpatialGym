"""Bounded discriminator for unnecessary fixed-pose transform round trips.

Only an explicit mechanism request installs this. Exact current native poses
are compared with incoming poses; no cold reference or tolerance is used.
"""
from contextlib import contextmanager
from pathlib import Path

from .contracts import require, write_json_atomic


@contextmanager
def idempotent_fixed_pose_scope(xform_type, kinematic_type, events):
    import torch
    original = xform_type.set_position_orientation

    def setting(obj, position=None, orientation=None, frame='world'):
        fixed = isinstance(obj, kinematic_type) or getattr(obj, 'kinematic_only', False) is True
        if fixed and frame == 'world' and position is not None and orientation is not None:
            target = (torch.as_tensor(position, dtype=torch.float32),
                      torch.as_tensor(orientation, dtype=torch.float32))
            if (target[0].shape == (3,) and target[1].shape == (4,)
                    and all(bool(torch.isfinite(v).all()) for v in target)):
                current = obj.get_position_orientation(frame=frame)
                equal = all(torch.equal(a, b) for a, b in zip(current, target))
                events.append({'prim_path':obj.prim_path, 'frame':frame,
                    'requested_pose':[v.tolist() for v in target],
                    'current_pose':[v.tolist() for v in current],
                    'bit_identical':equal, 'skipped':equal,
                    'reference_state_copied':False, 'physics_steps_added':0})
                if equal:
                    return None
        return original(obj, position=position, orientation=orientation, frame=frame)

    xform_type.set_position_orientation = setting
    try:
        yield
    finally:
        xform_type.set_position_orientation = original


def install(runner, output):
    from omnigibson.prims.xform_prim import XFormPrim
    from omnigibson.prims.rigid_kinematic_prim import RigidKinematicPrim
    output = Path(output); observations = []
    for name in ('attach', 'restore'):
        original = getattr(runner.InstanceLoader, name)

        def bounded(loader, *args, _original=original, _name=name, **kwargs):
            event = {'boundary':_name, 'instance_id':loader.iid, 'writes':[]}
            observations.append(event)
            try:
                with idempotent_fixed_pose_scope(XFormPrim, RigidKinematicPrim, event['writes']):
                    result = _original(loader, *args, **kwargs)
                event['status'] = 'completed'
                return result
            finally:
                event['native_setter_restored'] = True
                write_json_atomic(output/'fixed_pose_setter_audit.json', {
                    'observations':observations, 'benchmark_credit':0,
                    'production_released':False, 'physics_steps_added':0,
                    'comparison':'bit-identical current native pose, no tolerance or reference copies'})

        setattr(runner.InstanceLoader, name, bounded)
