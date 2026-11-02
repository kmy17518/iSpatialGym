"""Frames of a vectorized ``og.Environment``'s scene clones (no simulator import: unit-testable).

OmniGibson loads ``num_envs`` copies of one scene translated along +x with the identity rotation (``scene.pose`` is the 4x4
placement). A TRO is authored in the base scene's frame (env 0 = scene-relative coordinates), so what goes into clone ``i``
is **translated** by the clone's placement: object root positions, the robot base position, the world-frame pose targets
(``PoseReached`` goals), hinge anchors. Orientations and axes are untouched - a translation leaves them bit-identical, which
also keeps env 0 (zero translation) bit-identical to the single-env evaluator. Anything reported back (robot / camera poses)
is translated the other way. A rotated placement is not supported and is asserted, never silently ignored.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch as th


def scene_translation(scene) -> Optional[th.Tensor]:
    """The clone's placement as a (3,) float32 translation; ``None`` for the base scene (identity placement) or no scene."""
    if scene is None:
        return None
    pose = getattr(scene, "pose", None)
    if pose is None:
        return None
    pose = th.as_tensor(pose, dtype=th.float32)
    t = pose[:3, 3]
    rot_identity = bool(th.allclose(pose[:3, :3], th.eye(3), atol=1e-6))
    if float(th.norm(t)) == 0.0 and rot_identity:
        return None
    assert rot_identity, f"scene {getattr(scene, 'idx', '?')} is rotated ({pose[:3, :3].tolist()}); only translated clones are supported"
    return t.clone()


def to_clone(scene, position):
    """A base-scene position in clone ``scene``'s world frame."""
    t = scene_translation(scene)
    p = th.as_tensor(position, dtype=th.float32).reshape(3)
    return p if t is None else p + t


def to_base(scene, position):
    """A clone-world position in the base scene's frame."""
    t = scene_translation(scene)
    p = th.as_tensor(position, dtype=th.float32).reshape(3)
    return p if t is None else p - t


def offset_targets_meta(meta, scene):
    """A shallow copy of a TRO meta whose world-frame pose targets are translated into ``scene``'s placement (object-relative
    frames move with their object and need no change); the meta itself for the base scene."""
    t_scene = scene_translation(scene)
    if t_scene is None:
        return meta
    new = copy.copy(meta)
    new.pose_targets = {}
    for inst, tgt in meta.pose_targets.items():
        t = copy.copy(tgt)
        if (t.frame in (None, "world")) and t.position is not None:
            t.position = th.as_tensor(t.position, dtype=th.float32).reshape(3) + t_scene
        new.pose_targets[inst] = t
    # group targets: the absolute slot positions of (pose_reached <group>) move with the clone (the layout relations are
    # member-relative and need no change); without this a clone's group never reaches its pose (attachment, 2026-10-23)
    groups = getattr(meta, "groups", None)
    if groups:
        new.groups = {}
        for name, group in groups.items():
            g = copy.copy(group)
            g.target_slots = {
                sid: copy.copy(slot) for sid, slot in (group.target_slots or {}).items()
            }
            for slot in g.target_slots.values():
                if slot.position is not None:
                    slot.position = th.as_tensor(slot.position, dtype=th.float32).reshape(3) + t_scene
            new.groups[name] = g
    return new


def offset_object_state(state, scene):
    """An object's ``load_state`` dict with its root link position translated into the clone's frame (in place; returned)."""
    t = scene_translation(scene)
    if t is not None and isinstance(state, dict) and "root_link" in state:
        state["root_link"]["pos"] = th.as_tensor(state["root_link"]["pos"], dtype=th.float32) + t
    return state


__all__ = ["scene_translation", "to_clone", "to_base", "offset_targets_meta", "offset_object_state"]
