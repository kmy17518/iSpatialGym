"""Bounded own-frame holonomic joint authoring at an existing stopped boundary.

The native cold loader authors its fixed joint from a measured native pose.
The ordinary stopped setter instead copies its input quaternion to that joint.
This declared warm discriminator uses this world's resulting root frame; it
does not import a reference cold state or claim numerical equivalence.
"""
import math

from scripts.campaign_20261110.contracts import require


def author_native_root_joint(robot, sim, incoming, *, quaternion=None):
    require(sim.is_stopped() and robot.is_holonomic_base,
            'own-frame joint authoring requires the existing stopped holonomic boundary')
    require(not any(getattr(robot, '_ag_obj_in_hand', {}).values()),
            'assisted grasp cleanup must finish before the pool root transition')
    joint = getattr(robot, '_world_base_fixed_joint_prim', None)
    require(joint is not None, 'native holonomic world fixed joint is absent')
    attributes = [joint.GetAttribute(name) for name in ('physics:localPos0', 'physics:localRot0')]
    require(all(attribute and attribute.Get() is not None for attribute in attributes),
            'native world joint pose attributes are incomplete')
    if quaternion is None:
        import omnigibson.lazy as lazy
        quaternion = lazy.pxr.Gf.Quatf
    robot.set_position_orientation(position=incoming['position'], orientation=incoming['orientation'])
    position, orientation = (value.tolist() for value in robot.root_link.get_position_orientation())
    require(len(position) == 3 and len(orientation) == 4
            and all(math.isfinite(v) for v in position + orientation)
            and math.dist(position, incoming['position']) <= 1e-6
            and min(math.dist(orientation, incoming['orientation']),
                    math.dist(orientation, [-v for v in incoming['orientation']])) <= 1e-6,
            'own stopped native root differs from its exact incoming spawn')
    authored_position = tuple(position)
    authored_orientation = quaternion(orientation[3], *orientation[:3])
    # Match the native fixed-joint attribute types. The observed own root is
    # used instead of the setter's raw input quaternion. No lifecycle is added.
    with sim.editing_usd():
        attributes[0].Set(authored_position)
        attributes[1].Set(authored_orientation)
    require(tuple(attributes[0].Get()) == authored_position
            and attributes[1].Get() == authored_orientation,
            'own native world-joint authorship did not read back exactly')
    return {'mode': 'own_stopped_native_root_joint_v1',
            'incoming_source_pose': incoming,
            'own_native_root': [position, orientation],
            'world_joint_path': str(joint.GetPath()),
            'joint_pose_attribute_writes': 2,
            'reference_state_copied': False,
            'extra_physics_steps': 0, 'extra_stop_play_pairs': 0,
            'physical_equivalence_established': False}
