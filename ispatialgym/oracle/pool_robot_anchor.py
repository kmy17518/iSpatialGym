"""Restore the incoming spawn through the original cold robot setter.

Cold Environment loading defaults to the scene frame. Even an identity scene
pose has a float32 conversion before the entity's world-to-parent conversion.
The native setter also authors the fixed joint from its converted argument,
not from the resulting entity pose. Preserve that path and its native scalar
conversion; reconstructing the attributes from Python floats changes them.
"""
import math

from scripts.campaign_20261110.contracts import require


def restore_source_cold_spawn(robot, sim, incoming, *, pose_frame):
    require(sim.is_stopped() and robot.is_holonomic_base,
            'source spawn restoration requires the existing stopped holonomic boundary')
    require(pose_frame in ('scene', 'world'), 'unsupported cold robot spawn frame')
    require(not any(getattr(robot, '_ag_obj_in_hand', {}).values()),
            'assisted grasp cleanup must finish before the pool root transition')
    joint = getattr(robot, '_world_base_fixed_joint_prim', None)
    require(joint is not None, 'native holonomic world fixed joint is absent')
    attributes = [joint.GetAttribute(name) for name in ('physics:localPos0', 'physics:localRot0')]
    require(all(attribute and attribute.Get() is not None for attribute in attributes),
            'native world joint pose attributes are incomplete')
    robot.set_position_orientation(position=incoming['position'], orientation=incoming['orientation'],
                                   frame=pose_frame)
    position, orientation = (value.tolist() for value in robot.root_link.get_position_orientation())
    require(len(position) == 3 and len(orientation) == 4
            and all(math.isfinite(v) for v in position + orientation)
            and math.dist(position, incoming['position']) <= 1e-6
            and min(math.dist(orientation, incoming['orientation']),
                    math.dist(orientation, [-v for v in incoming['orientation']])) <= 1e-6,
            'own stopped native root differs from its exact incoming spawn')
    # Read only: the cold setter intentionally produces different entity and
    # joint quaternions. Its tuple-of-tensors position write also differs from
    # a Python-float tuple in the pinned USD binding. Never rewrite either
    # attribute from this measured root or an independently initialized trace.
    joint_position = list(attributes[0].Get())
    joint_orientation = attributes[1].Get()
    joint_orientation = [*joint_orientation.GetImaginary(), joint_orientation.GetReal()]
    require(all(math.isfinite(v) for v in joint_position + joint_orientation),
            'native cold setter produced nonfinite joint attributes')
    return {'mode': 'source_cold_stopped_spawn_v1',
            'source_pose_frame': pose_frame,
            'incoming_source_pose': incoming,
            'own_native_root': [position, orientation],
            'native_joint_pose': [joint_position, joint_orientation],
            'world_joint_path': str(joint.GetPath()),
            'native_setter_calls': 1, 'joint_pose_attribute_writes': 0,
            'native_joint_pose_preserved': True,
            'reference_state_copied': False,
            'extra_physics_steps': 0, 'extra_stop_play_pairs': 0,
            'physical_equivalence_established': False}
