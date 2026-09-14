"""Probe wall-ornament mounting options in OmniGibson / PhysX.

This creates four thin rigid bodies at the same initial in-plane angle:

1. no mount (falls),
2. a top-edge revolute joint (gravity self-rights it),
3. a center revolute joint (gravity produces no hinge torque), and
4. a top-edge revolute joint with gravity disabled.

It then drives a small rigid sphere into the lower side of the center-mounted
body.  This approximates a velocity-controlled robot link making contact and
checks that contact can rotate the passive hinge.

Run inside the project environment:

    OMNIGIBSON_HEADLESS=1 conda run -n ispatialgym \
        python ornament_hinge_probe.py
"""

import json
import math
import os
import traceback

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson import launch
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.scenes.scene_base import Scene
from omnigibson.utils.transform_utils import euler2quat, quat2mat, quat_inverse, quat_multiply
from omnigibson.utils.usd_utils import create_joint


WIDTH = 0.60
HEIGHT = 0.80
THICKNESS = 0.04
CENTER_Z = 2.0
INITIAL_ANGLE_DEG = 25.0


def in_plane_angle_deg(obj):
    """Return the signed angle between the ornament's local up and world +Z."""
    _, quat = obj.get_position_orientation()
    local_up = getattr(obj, "_probe_local_up", th.tensor([0.0, 0.0, 1.0]))
    up = quat2mat(quat) @ local_up
    return math.degrees(math.atan2(up[0].item(), up[2].item()))


def snapshot(obj):
    pos, _ = obj.get_position_orientation()
    return {
        "angle_deg": round(in_plane_angle_deg(obj), 3),
        "position": [round(value, 3) for value in pos.tolist()],
        "angular_speed_rad_s": round(th.linalg.vector_norm(obj.get_angular_velocity()).item(), 4),
    }


def configure_body(obj, *, angular_damping=1.5, disable_gravity=False):
    api = lazy.pxr.PhysxSchema.PhysxRigidBodyAPI(obj.root_link.prim)
    with og.sim.editing_usd():
        api.GetAngularDampingAttr().Set(angular_damping)
        api.GetLinearDampingAttr().Set(0.1)
        api.GetDisableGravityAttr().Set(disable_gravity)


def configure_revolute_joint(joint):
    with og.sim.editing_usd():
        joint.GetAttribute("physics:axis").Set("Y")
        joint.GetAttribute("physics:lowerLimit").Set(-70.0)
        joint.GetAttribute("physics:upperLimit").Set(70.0)


def add_ornament(scene, name, x, mount, *, disable_gravity=False):
    obj = PrimitiveObject(
        name=name,
        relative_prim_path=f"/{name}",
        category="picture",
        primitive_type="Cube",
        scale=[WIDTH, THICKNESS, HEIGHT],
        fixed_base=False,
        kinematic_only=False,
        include_default_states=False,
        load_config={"mass": 1.0},
        rgba=(0.25, 0.45, 0.85, 1.0),
    )
    scene.add_object(obj)

    angle = math.radians(INITIAL_ANGLE_DEG)
    quat = euler2quat(th.tensor([0.0, angle, 0.0]))
    rotation = quat2mat(quat)

    if mount == "top":
        local_anchor = th.tensor([0.0, 0.0, HEIGHT / 2.0])
        world_anchor = th.tensor([x, 0.0, CENTER_Z + HEIGHT / 2.0])
        center = world_anchor - rotation @ local_anchor
    else:
        local_anchor = th.zeros(3)
        world_anchor = th.tensor([x, 0.0, CENTER_Z])
        center = world_anchor

    obj.set_position_orientation(position=center, orientation=quat)
    configure_body(obj, disable_gravity=disable_gravity)

    if mount is not None:
        joint = create_joint(
            prim_path=f"{scene.prim_path}/ornament_joints/{name}_hinge",
            joint_type="RevoluteJoint",
            body0=None,
            body1=obj.root_link.prim_path,
            joint_frame_in_parent_frame_pos=world_anchor,
            joint_frame_in_parent_frame_quat=th.tensor([0.0, 0.0, 0.0, 1.0]),
            joint_frame_in_child_frame_pos=local_anchor,
            joint_frame_in_child_frame_quat=th.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        configure_revolute_joint(joint)

    return obj


def add_dataset_picture(scene, name, x):
    # This model is natively a roughly 8.8 x 9.6 cm sheet in its local XY plane.
    # Scale its in-plane dimensions and rotate local +Y to world +Z.
    obj = DatasetObject(
        name=name,
        relative_prim_path=f"/{name}",
        category="picture",
        model="abtgvz",
        scale=[WIDTH / 0.0878600616, HEIGHT / 0.0957818985, 1.0],
        fixed_base=False,
        kinematic_only=False,
        include_default_states=False,
        load_config={"mass": 1.0},
    )
    scene.add_object(obj)

    base_quat = euler2quat(th.tensor([math.pi / 2.0, 0.0, 0.0]))
    slant_quat = euler2quat(th.tensor([0.0, math.radians(INITIAL_ANGLE_DEG), 0.0]))
    quat = quat_multiply(slant_quat, base_quat)
    center = th.tensor([x, 0.0, CENTER_Z])
    obj.set_position_orientation(position=center, orientation=quat)
    obj._probe_local_up = th.tensor([0.0, 1.0, 0.0])
    # The real asset has a very thin collision shape, so use stronger passive
    # damping to make a low-speed contact easy to stop near the target.
    configure_body(obj, angular_damping=8.0)

    joint = create_joint(
        prim_path=f"{scene.prim_path}/ornament_joints/{name}_hinge",
        joint_type="RevoluteJoint",
        body0=None,
        body1=obj.root_link.prim_path,
        joint_frame_in_parent_frame_pos=center,
        joint_frame_in_parent_frame_quat=th.tensor([0.0, 0.0, 0.0, 1.0]),
        joint_frame_in_child_frame_pos=th.zeros(3),
        # Align the joint's child frame with the unslanted wall frame.
        joint_frame_in_child_frame_quat=quat_inverse(base_quat),
    )
    configure_revolute_joint(joint)
    return obj


def add_pusher(scene):
    pusher = PrimitiveObject(
        name="robot_link_proxy",
        relative_prim_path="/robot_link_proxy",
        category="pusher",
        primitive_type="Sphere",
        radius=0.065,
        fixed_base=False,
        kinematic_only=False,
        include_default_states=False,
        load_config={"mass": 5.0},
        rgba=(0.9, 0.25, 0.15, 1.0),
    )
    scene.add_object(pusher)
    pusher.set_position_orientation(
        position=th.tensor([0.0, -2.0, CENTER_Z]),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0]),
    )
    configure_body(pusher, angular_damping=1.0, disable_gravity=True)
    return pusher


def validate_result(result):
    initial = result["initial"]
    settled = result["after_4s"]
    primitive_push = result["contact_push"]
    dataset_push = result["dataset_picture_contact_push"]

    assert initial["unmounted"]["position"][2] - settled["unmounted"]["position"][2] > 1.0
    assert abs(settled["top_hinge_gravity"]["angle_deg"]) < 3.0
    assert abs(settled["center_hinge_gravity"]["angle_deg"] - INITIAL_ANGLE_DEG) < 0.1
    assert abs(settled["top_hinge_no_gravity"]["angle_deg"] - INITIAL_ANGLE_DEG) < 0.1
    assert abs(settled["dataset_picture_center_hinge"]["angle_deg"] - INITIAL_ANGLE_DEG) < 0.1

    assert primitive_push["reached_upright_during_push"]
    assert abs(primitive_push["after_3s_settle"]["angle_deg"]) < 3.0
    assert primitive_push["after_3s_settle"]["angular_speed_rad_s"] < 0.005

    assert dataset_push["reached_upright_during_push"]
    assert abs(dataset_push["after_3s_settle"]["angle_deg"]) < 3.0
    assert dataset_push["after_3s_settle"]["angular_speed_rad_s"] < 0.005


def main():
    sim = launch()
    try:
        scene = Scene()
        sim.import_scene(scene)

        with sim.editing_usd():
            sim.stage.DefinePrim(f"{scene.prim_path}/ornament_joints", "Scope")

        ornaments = {
            "unmounted": add_ornament(scene, "unmounted", -3.0, None),
            "top_hinge_gravity": add_ornament(scene, "top_hinge_gravity", -1.0, "top"),
            "center_hinge_gravity": add_ornament(
                scene,
                "center_hinge_gravity",
                1.0,
                "center",
            ),
            "top_hinge_no_gravity": add_ornament(
                scene,
                "top_hinge_no_gravity",
                3.0,
                "top",
                disable_gravity=True,
            ),
            "dataset_picture_center_hinge": add_dataset_picture(scene, "dataset_picture_center_hinge", 5.0),
        }
        pusher = add_pusher(scene)

        sim.play()
        sim.step()
        initial = {name: snapshot(obj) for name, obj in ornaments.items()}

        # Let gravity and damping act for four simulated seconds.
        for _ in range(240):
            sim.step()
        settled = {name: snapshot(obj) for name, obj in ornaments.items()}

        # Approach the center-mounted frame's lower-left side along its local +X.
        # This gives a negative moment about the wall-normal Y hinge axis.
        target = ornaments["center_hinge_gravity"]
        center, quat = target.get_position_orientation()
        rotation = quat2mat(quat)
        inward = rotation @ th.tensor([1.0, 0.0, 0.0])
        contact = center + rotation @ th.tensor([-WIDTH / 2.0, 0.0, -HEIGHT / 4.0])
        start = contact - inward * 0.18
        pusher.set_position_orientation(position=start)
        pusher.set_linear_velocity(th.zeros(3))
        sim.step()

        before_contact_push = snapshot(target)
        reached_upright = False
        push_steps = 0
        for push_steps in range(1, 181):
            # Reissuing velocity approximates a velocity-controlled robot link.
            pusher.set_linear_velocity(inward * 0.35)
            sim.step()
            if abs(in_plane_angle_deg(target)) < 2.0:
                reached_upright = True
                break

        pusher.set_linear_velocity(th.zeros(3))
        for _ in range(180):
            sim.step()
        after_contact_push = snapshot(target)

        # Repeat the contact test on a real BEHAVIOR-1K "picture" asset.
        dataset_target = ornaments["dataset_picture_center_hinge"]
        center, quat = dataset_target.get_position_orientation()
        rotation = quat2mat(quat)
        inward = rotation @ th.tensor([1.0, 0.0, 0.0])
        contact = center + rotation @ th.tensor([-WIDTH / 2.0, -HEIGHT / 4.0, 0.0])
        start = contact - inward * 0.10
        pusher.set_position_orientation(position=start)
        pusher.set_linear_velocity(th.zeros(3))
        sim.step()

        dataset_before_push = snapshot(dataset_target)
        dataset_reached_upright = False
        dataset_push_steps = 0
        previous_angle = in_plane_angle_deg(dataset_target)
        for dataset_push_steps in range(1, 241):
            pusher.set_linear_velocity(inward * 0.08)
            sim.step()
            angle = in_plane_angle_deg(dataset_target)
            if abs(angle) < 2.0 or angle * previous_angle <= 0.0:
                dataset_reached_upright = True
                break
            previous_angle = angle

        pusher.set_linear_velocity(th.zeros(3))
        pusher.set_position_orientation(position=th.tensor([0.0, -2.0, CENTER_Z]))
        for _ in range(180):
            sim.step()
        dataset_after_push = snapshot(dataset_target)

        result = {
            "physics_dt_s": sim.get_physics_dt(),
            "initial": initial,
            "after_4s": settled,
            "contact_push": {
                "steps": push_steps,
                "reached_upright_during_push": reached_upright,
                "before": before_contact_push,
                "after_3s_settle": after_contact_push,
            },
            "dataset_picture_contact_push": {
                "model": "picture/abtgvz",
                "steps": dataset_push_steps,
                "reached_upright_during_push": dataset_reached_upright,
                "before": dataset_before_push,
                "after_3s_settle": dataset_after_push,
            },
        }
        validate_result(result)
        result["checks_passed"] = True
        print("\nORNAMENT_PROBE_RESULT=" + json.dumps(result, indent=2), flush=True)
    except BaseException:
        # Shutdown can hard-exit in this stack and hide the original Python
        # exception, so report failures before allowing process teardown.
        traceback.print_exc()
        raise
    else:
        og.shutdown()


if __name__ == "__main__":
    main()
