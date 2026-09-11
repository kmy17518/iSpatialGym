"""Revolute hinge constraints for the World Alignment tasks (TRO ``constraints`` records).

A hinge is a PhysX ``RevoluteJoint`` between a parent rigid body (support / wall link, or the world when ``parent``
is None) and the object's root link, anchored at a world point with a world axis. The joint frame's X axis is the
rotation axis (PhysX default ``physics:axis = "X"``). Joints are USD prims, not part of ``og.sim.dump_state``; the
sampler creates them for its own checks and the loader recreates them after every reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import create_joint

from ispatialgym.sampling import geometry as G


@dataclass
class Hinge:
    prim_path: str
    child_name: str
    parent_name: Optional[str]
    parent_link: Optional[str]
    anchor: th.Tensor  # world
    axis: th.Tensor  # world unit

    def to_tro(self, child_inst: str, parent_inst: Optional[str]) -> dict:
        return {
            "type": "revolute",
            "child": child_inst,
            "parent": parent_inst,
            "parent_link": self.parent_link,
            "anchor_position": [float(v) for v in self.anchor],
            "axis": [float(v) for v in self.axis],
            "limits": None,
        }


def _frame_with_x(axis: th.Tensor, ref_up: th.Tensor) -> th.Tensor:
    """Rotation matrix whose X column is ``axis``; Z chosen close to ``ref_up``."""
    x = axis / th.linalg.norm(axis)
    z = ref_up - th.dot(ref_up, x) * x
    if float(th.linalg.norm(z)) < 1e-6:
        z = th.tensor([1.0, 0.0, 0.0]) - th.dot(th.tensor([1.0, 0.0, 0.0]), x) * x
    z = z / th.linalg.norm(z)
    y = th.cross(z, x, dim=0)
    return th.stack([x, y, z], dim=1)


def create_hinge(child_obj, anchor_world, axis_world, parent_link=None, name: str = "isg_hinge") -> Hinge:
    """Create a revolute joint holding ``child_obj``'s root at ``anchor_world`` and letting it rotate about ``axis_world``.

    ``parent_link``: a link (RigidPrim) of the parent object; None anchors to the world frame.
    """
    anchor = G.t(anchor_world)
    axis = G.t(axis_world)
    axis = axis / th.linalg.norm(axis)
    R_j = _frame_with_x(axis, th.tensor([0.0, 0.0, 1.0]))
    q_j = T.mat2quat(R_j)
    cp, cq = child_obj.root_link.get_position_orientation()
    # joint frame expressed in the child link frame
    p1, q1 = T.relative_pose_transform(anchor, q_j, cp.float(), cq.float())
    if parent_link is not None:
        pp, pq = parent_link.get_position_orientation()
        p0, q0 = T.relative_pose_transform(anchor, q_j, pp.float(), pq.float())
        body0 = parent_link.prim_path
    else:
        p0, q0 = anchor, q_j
        body0 = None
    prim_path = f"{child_obj.prim_path}/{name}"
    create_joint(
        prim_path=prim_path,
        joint_type="RevoluteJoint",
        body0=body0,
        body1=child_obj.root_link.prim_path,
        joint_frame_in_parent_frame_pos=p0,
        joint_frame_in_parent_frame_quat=q0,
        joint_frame_in_child_frame_pos=p1,
        joint_frame_in_child_frame_quat=q1,
        exclude_from_articulation=True,
    )
    return Hinge(prim_path, child_obj.name, None if parent_link is None else parent_link.prim_path.split("/")[-2], None if parent_link is None else parent_link.body_name, anchor, axis)


def remove_hinge(hinge: Optional[Hinge]):
    if hinge is None:
        return
    stage = og.sim.stage
    prim = stage.GetPrimAtPath(hinge.prim_path)
    if prim is not None and prim.IsValid():
        with og.sim.editing_usd():
            stage.RemovePrim(hinge.prim_path)
        og.sim.update_handles()
