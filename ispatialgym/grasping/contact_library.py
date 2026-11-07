"""PUSH and BRACE contact libraries (i7-contact 2026-11-07, step 2): per object model and stable rest, the contacts the samplers' record builders
produce (``sampling/v2/contact_poses.py``, the ONE contact schema ``isg-contact/1``), in the object frame, with the pre-pose / fingers / stop
rules AS THE ORACLE EXECUTES them, then physics-validated (:mod:`ispatialgym.grasping.validate_contacts`). Two libraries on the one schema:
``push`` and ``brace`` (attach stays with the grasp library, handles with the articulation library).

Layout (a library root, e.g. ``$OFH/contact_libs_<date>`` - never under the grasp-bands home)::

    <root>/push/models/<entry id>/library.json      schema isg-contact-library/1, kind push
    <root>/brace/models/<entry id>/library.json     schema isg-contact-library/1, kind brace
    <root>/logs/{build,validate}_<shard>.jsonl

A library holds the OBJECT (category / model / scale, the v2 entry it was read from, its stable rest: quat + h_root, the geometry digest), the
FAMILIES it was built for with their resolved knobs (the pose set: ``generation`` knobs of the alias family, the rule frame, yaw slots, stroke),
and the ENTRIES: ``contact_poses.contact_entry`` records (kind, class, family, arm role, ``T_object_eef``, contact point, approach, closing,
pitch, wrist turn, fingers, height rule, pre recorded / executed + ``pre_pose_rule``, push direction / stroke, stop rule) + ``yaw_deg`` (the
object's yaw in the rule frame: +x = the robot's forward, +y its left; the object stands on a support at z 0) and, after validation, each
entry's ``validation`` (trials, verdict, reasons, measures). The object needs collision geometry and a measured stable rest (a v2 model
directory: ``geometry.json`` + ``rests.json`` / ``library.json``) - validated GRASPS are not needed.

Families (:data:`FAMILIES`): ``touching_push`` / ``touching_brace`` (touching small scale 01/02/04: the pushed target / the braced partner;
``IORSSmall`` record builders), ``axial_push`` (alignment-axial 01-05: the blade at a long-axis end along the end's arc), ``axial_brace``
(alignment-axial 01-05: the stop brace at the goal - ``AlignmentAxial._brace_candidates``' rule; the flat targets' v2 entries have rests but no
grasps, which a brace entry does not need). Registered without a
pure builder yet (``builder`` None: the record builders are interleaved with the samplers' IK and still need factoring): ``block_push`` /
``block_brace`` (block_design_01), ``floor_blade`` / ``floor_flat`` / ``floor_pitched`` (under_large 01-03, touching_large_03: the push-under
chains need ``contact_offset_along`` / ``pitched_turns`` factored), ``base_push`` (nextto_large 01-03, touching_large_01: the face points of
``iors_large_nextto``), ``wall_push`` / ``wall_brace`` (world_vertical: the wall layout geometry).

    python -m ispatialgym.grasping.contact_library build --jobs J.json --glib GLIB --root ROOT [--shard i/n] [--only ID ...]
    python -m ispatialgym.grasping.contact_library summary --root ROOT
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch as th

from ispatialgym.grasping import schema as S

LIB_SCHEMA = "isg-contact-library/1"
LIB_KINDS = ("push", "brace")
TOUCHING_SUBTASK = "inter_object_relation_satisfaction-touching_small_scale"
#: the rule frame's push direction of the touching layout: +y (the robot's left, toward the partner); the brace approaches along -y
TOUCH_DIRECTION = (0.0, 1.0, 0.0)
#: the default push stroke of a touching push entry (m past the init along the push): within the measured 0.056-0.085 m
TOUCH_PUSH_STROKE_M = 0.08
#: the default turn of an axial push entry (rad of the end's arc; the axial deviation range is 20-70 deg)
AXIAL_TURN_DEG = 45.0
#: alignment_axial's END_INSET: the pushed end lies this far inside the long-axis extremity
AXIAL_END_INSET_M = 0.03
DEFAULT_YAWS = 8


# ------------------------------------------------------------------------------------------------ helpers


def yaw_slots(n: int = DEFAULT_YAWS) -> List[float]:
    return [2 * math.pi * k / n for k in range(int(n))]


def object_frame(rest_quat_xyzw, h_root: float, yaw: float, support_z: float = 0.0) -> np.ndarray:
    """T_world_object of the object at its rest turned ``yaw`` about z, its root at the origin xy, standing on a support at ``support_z``
    (``schema.place_object``)."""
    T = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ S.quat_to_mat(np.asarray(rest_quat_xyzw, dtype=float))
    T[2, 3] = float(support_z) + float(h_root)
    return T


def _world_points(pts_obj: np.ndarray, T: np.ndarray) -> th.Tensor:
    return th.as_tensor(pts_obj @ T[:3, :3].T + T[:3, 3], dtype=th.float32)


def _tpose(T: np.ndarray):
    return th.tensor(T[:3, 3], dtype=th.float32), th.tensor(S.mat_to_quat(T[:3, :3]), dtype=th.float32)


def _label(kind: str, extras: Dict[str, Any]) -> str:
    """The touching push's class label (``IORSSmall._touching_push_choice``): ``kind@<contact height cm>`` when it records one."""
    h = (extras or {}).get("contact_height_m")
    return kind if h is None else f"{kind}@{100.0 * float(h):.1f}cm"


# ------------------------------------------------------------------------------------------------ the families' builders


def build_touching_push(pts_obj: np.ndarray, rest, knobs: Dict[str, Any], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The touching push entries (``contact_poses.touching_push_candidates``, the open push first under ``push_gripper_open``) of the target at
    each yaw slot: push along +y, the chain's recorded pre (``push_tip_pre_dist_m`` / ``IK.PRE_DIST``) and its executed one (the oracle lowers a
    straight push's to 0.04 m), the stroke ``params["stroke_m"]``."""
    from itertools import chain

    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.v2 import contact_poses as CPZ
    from ispatialgym.sampling.v2 import tabletop as TT

    gs = CPZ.PoseGen(knobs, subtask=params.get("subtask") or TOUCHING_SUBTASK)
    sweep = bool(TT.phase2_gen(gs, "push_height_sweep", False))
    opened = bool(TT.phase2_gen(gs, "push_gripper_open", False))
    D = th.tensor(TOUCH_DIRECTION)
    stroke = float(params.get("stroke_m", TOUCH_PUSH_STROKE_M))
    out = []
    for yaw in yaw_slots(int(params.get("yaws", DEFAULT_YAWS))):
        T = object_frame(rest.quat_xyzw, rest.h_root, yaw)
        hull_w = _world_points(pts_obj, T)
        face_z = (float(hull_w[:, 2].min()), float(hull_w[:, 2].max()))
        work, _brace = CPZ.touching_points(hull_w, hull_w, D[:2])
        cands = CPZ.touching_push_candidates(gs, CPZ.MAP_FRAME, work, face_z, D, t_hull_w=hull_w if sweep else None)
        if opened:
            cands = chain(CPZ.open_push_candidates(gs, CPZ.MAP_FRAME, hull_w, face_z, D), cands)
        seen = set()
        for kind, init, extras in cands:
            cls = _label(kind, extras)
            while cls in seen:
                cls += "#"
            seen.add(cls)
            pre = C.pre_pose(init, CPZ.recorded_pre_dist(extras))
            # the sampler's screen of the class (IORSSmall._touching_push_choice: push_hand_overlap - against the hull under the sweep or with
            # open fingers, else the box; the pushed object moves with the hand, so the goal's verdict is the init's)
            against = "hull" if (sweep or CPZ.contact_fingers(extras) != 0.0) else "box"
            overlap = CPZ.hand_overlap(init, th.as_tensor(pts_obj, dtype=th.float32), _tpose(T), against=against,
                                       hull_w=hull_w if against == "hull" else None)
            e = CPZ.contact_entry("push", cls, "touching_push", "work", T, init, pre, dict(extras, pose_class=kind),
                                  face_rule="the target's -y face (the push contact of touching_points), the footprint centre line under the sweep",
                                  direction=D, stroke_m=stroke, support_z=0.0, object_height=face_z[1] - face_z[0], executor="push")
            e["yaw_deg"] = round(math.degrees(yaw), 4)
            e["sampler_screen"] = {"hand_overlap": overlap, "against": against}
            out.append(e)
    return out


def build_touching_brace(pts_obj: np.ndarray, rest, knobs: Dict[str, Any], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The touching brace entries (``contact_poses.touching_brace_candidates``) of the partner at each yaw slot: the far (+y) face approached
    along -y, the push line through the partner's centre (``push_point`` None), ``brace_open_when_wide`` from the knobs."""
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    gs = CPZ.PoseGen(knobs, subtask=params.get("subtask") or TOUCHING_SUBTASK)
    D = th.tensor(TOUCH_DIRECTION)
    out = []
    for yaw in yaw_slots(int(params.get("yaws", DEFAULT_YAWS))):
        T = object_frame(rest.quat_xyzw, rest.h_root, yaw)
        hull_w = _world_points(pts_obj, T)
        face_z = (float(hull_w[:, 2].min()), float(hull_w[:, 2].max()))
        _work, b_point = CPZ.touching_points(hull_w, hull_w, D[:2])
        seen = set()
        for pre, contact, kind, extras in CPZ.touching_brace_candidates(gs, CPZ.MAP_FRAME, b_point, face_z, -D, partner_hull_w=hull_w, push_point=None,
                                                                        open_when_wide=knobs.get("brace_open_when_wide") is True):
            cls = kind
            while cls in seen:
                cls += "#"
            seen.add(cls)
            e = CPZ.contact_entry("brace", cls, "touching_brace", "brace", T, contact, pre, dict(extras),
                                  face_rule="the partner's +y face (the brace point of touching_points), approached along -y",
                                  support_z=0.0, object_height=face_z[1] - face_z[0], pitch_deg=extras.get("pitch_deg"))
            e["yaw_deg"] = round(math.degrees(yaw), 4)
            # the sampler's screen (IORSSmall._touching_brace_choice: hand_overlap against the partner's box)
            e["sampler_screen"] = {"hand_overlap": CPZ.hand_overlap(contact, th.as_tensor(pts_obj, dtype=th.float32), _tpose(T), against="box"), "against": "box"}
            e["load"] = {"direction": [float(v) for v in TOUCH_DIRECTION], "what": "the touching push drives the target into the partner along +y"}
            out.append(e)
    return out


def axial_rest(rest) -> Any:
    """The axial program's placement: the target upright at the IDENTITY orientation turned about z (``object_pose_on(..., IDENTITY_QUAT, ...)``,
    the goal ``quat_from_yaw(yaw_goal)``) with the v2 entry's measured root height - the v2 rest must be that upright rest (tilt <= 0.05 rad),
    else ``ValueError``."""
    if S.tilt_distance(np.asarray(rest.quat_xyzw, dtype=float), np.array([0.0, 0.0, 0.0, 1.0])) > 0.05:
        raise ValueError(f"the v2 rest {list(np.round(rest.quat_xyzw, 4))} is not the axial program's upright identity rest")
    return S.Rest(rest_id=rest.rest_id, quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), h_root=float(rest.h_root), reference_yaw=0.0, record=rest.record)


def _axial_long(pts_obj: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """The target's long axis in its own frame (``params["long_axis"]``: the customs / icl frame record; default the footprint's longer side)."""
    if params.get("long_axis") is not None:
        a = np.asarray(params["long_axis"], dtype=float)
        return a / np.linalg.norm(a)
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    ang = CPZ.min_area_rect(th.as_tensor(pts_obj[:, :2], dtype=th.float32))[0]
    return np.array([math.cos(ang), math.sin(ang), 0.0])


def build_axial_brace(pts_obj: np.ndarray, rest, knobs: Dict[str, Any], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The axial STOP BRACE entries (``contact_poses.axial_stop_geometry`` + ``axial_brace_candidates`` - ``AlignmentAxial._stop_geometry`` /
    ``_brace_candidates``' rule): per GOAL yaw slot (relative to the stance; the object's root at its goal pose), long-axis end
    (``AXIAL_END_INSET_M`` inside it) and turn sense, the push turning the end ``turn_deg`` into the goal; the closed left gripper at the stop
    face the end arrives at (at 3/4 of the height): the top-down blade beside it (straight / pitched, both wrist turns; first for a target
    under 4 cm) and the pads on it (closing v / h, pad centre 5 mm outside the goal face), each clear of the target's sweep from its init
    pose. The entries are in the object frame AT THE GOAL POSE (the brace holds the object there while the push completes); ``load`` = the
    direction the end arrives in (``d_stop``, rule frame) and the end's point (goal object frame)."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.v2 import contact_poses as CPZ
    from ispatialgym.sampling.v2 import tabletop as TT

    r0 = axial_rest(rest)
    hull = th.as_tensor(pts_obj, dtype=th.float32)
    turn = math.radians(float(params.get("turn_deg", AXIAL_TURN_DEG)))
    com_local = th.tensor(params.get("com_local") if params.get("com_local") is not None else pts_obj.mean(axis=0).tolist(), dtype=th.float32)
    long_local = th.tensor(_axial_long(pts_obj, params), dtype=th.float32)
    proj = (hull - com_local) @ long_local
    h = float(pts_obj[:, 2].max() - pts_obj[:, 2].min())
    z_push, face_z, z_brace = CPZ.axial_push_height(0.0, h), (0.0, h), TT.contact_height(0.0, h)
    out = []
    for yaw_goal in yaw_slots(int(params.get("yaws", DEFAULT_YAWS))):
        for end_sign in (1.0, -1.0):
            half_len = float(proj.max() if end_sign > 0 else -proj.min()) - AXIAL_END_INSET_M
            for spin in (1.0, -1.0):
                d_goal = spin * turn
                yaw0 = yaw_goal - d_goal
                q0 = C.quat_from_yaw(yaw0)
                pose_init = (th.tensor([0.0, 0.0, float(r0.h_root)]), q0)
                com = pose_init[0] + C.quat_rotate(q0, com_local)
                end = com + C.quat_rotate(q0, long_local) * end_sign * half_len
                goal_pose, stop_pts, d_stop = CPZ.axial_stop_geometry(hull, com_local, pose_init, end, com, spin, yaw_goal, z_brace)
                T_goal = CPZ._mat(goal_pose)
                Ti = np.linalg.inv(T_goal)
                end_goal = C.quat_rotate(goal_pose[1], C.quat_rotate(C.qinv(q0), end - pose_init[0])) + goal_pose[0]
                skips: Dict[str, int] = {}
                seen = set()
                for pre, contact, kind, extras in CPZ.axial_brace_candidates(CPZ.MAP_FRAME, hull, stop_pts, d_stop, None, pose_init, com, d_goal, z_push=z_push,
                                                                             face_z=face_z, height=h, skips=skips):
                    cls = f"{'+' if end_sign > 0 else '-'}end{'+' if spin > 0 else '-'}spin_{kind}@f{int(extras.get('stop_face_fraction', 0))}"
                    while cls in seen:
                        cls += "#"
                    seen.add(cls)
                    pitch = float(str(extras["pose_class"])[len("blade_"):]) if str(extras.get("pose_class", "")).startswith("blade_") else 0.0
                    e = CPZ.contact_entry("brace", cls, "axial_brace", "brace", T_goal, contact, pre, dict(extras, kind=kind),
                                          face_rule="the goal hull's stop face the pushed end arrives at (d_stop), at 3/4 of the height; the blade at the push height",
                                          support_z=0.0, object_height=h, pitch_deg=pitch)
                    e["yaw_deg"] = round(math.degrees(yaw_goal), 4)
                    e["goal"] = {"yaw_goal_deg": round(math.degrees(yaw_goal), 4), "turn_rad": round(d_goal, 6), "end_sign": end_sign, "spin": spin,
                                 "init_yaw_deg": round(math.degrees(yaw0), 4), "skips": dict(skips)}
                    e["load"] = {"frame": "rule", "direction": [round(float(v), 6) for v in d_stop.tolist()],
                                 "point_object": [round(float(v), 6) for v in (Ti[:3, :3] @ end_goal.double().numpy() + Ti[:3, 3])],
                                 "what": "the right arm's axial push brings the end into the stop at the goal"}
                    out.append(e)
    return out


def build_axial_push(pts_obj: np.ndarray, rest, knobs: Dict[str, Any], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The axial push entries (``AlignmentAxial._sample_with``'s classes on the shared rules: ``axial_push_height``, ``axial_blade_geometry``,
    ``extreme_point``, ``axial_push_chain``) of the target at each yaw slot: per long-axis end (``AXIAL_END_INSET_M`` inside it) and turn sense,
    the closed blade beside the face the end moves away from, pitched 0 / 30 / 45 / 60 deg toward the robot's forward (+x), the closing axis
    along the push or across it, both wrist turns, along the end's arc through ``params["turn_deg"]`` about the centre of mass
    (``params["com_local"]``, default the hull centroid). The long axis: ``params["long_axis"]`` (object frame), default the footprint's
    longer side at the rest."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.v2 import contact_poses as CPZ
    from ispatialgym.sampling.v2 import tabletop as TT

    z_axis = th.tensor([0.0, 0.0, 1.0])
    turn = math.radians(float(params.get("turn_deg", AXIAL_TURN_DEG)))
    com_local = th.tensor(params.get("com_local") if params.get("com_local") is not None else pts_obj.mean(axis=0).tolist(), dtype=th.float32)
    long_local = params.get("long_axis")
    rest = axial_rest(rest)  # the axial program's upright identity placement
    out = []
    for yaw in yaw_slots(int(params.get("yaws", DEFAULT_YAWS))):
        T = object_frame(rest.quat_xyzw, rest.h_root, yaw)
        pts_w = _world_points(pts_obj, T)
        h = float(pts_w[:, 2].max() - pts_w[:, 2].min())
        z_push = CPZ.axial_push_height(0.0, h)
        R = th.tensor(T[:3, :3], dtype=th.float32)
        com = R @ com_local + th.tensor(T[:3, 3], dtype=th.float32)
        if long_local is not None:
            long_w = C.unit(R @ th.tensor(long_local, dtype=th.float32))
        else:
            ang, _l, _s, _c = CPZ.min_area_rect(pts_w[:, :2])
            long_w = th.tensor([math.cos(ang), math.sin(ang), 0.0])
        proj = (pts_w[:, :2] - com[:2]) @ long_w[:2]
        for end_sign in (1.0, -1.0):
            half_len = float(proj.max() if end_sign > 0 else -proj.min()) - AXIAL_END_INSET_M
            e_pt = com + long_w * end_sign * half_len
            r_xy = th.tensor([float(e_pt[0] - com[0]), float(e_pt[1] - com[1]), 0.0])
            for spin in (1.0, -1.0):
                d = spin * th.cross(z_axis, r_xy, dim=0)
                d = d / th.linalg.norm(d)
                side_pt = CPZ.extreme_point(pts_w, -d, near=e_pt, long_w=long_w[:2])
                side_pt = th.tensor([float(side_pt[0]), float(side_pt[1]), z_push])
                side_c = C.unit(th.cross(z_axis, d, dim=0))
                for pitch in TT.approach_pitch_order(None):
                    a_vec = TT.pitched_approach(pitch, th.tensor([1.0, 0.0]))
                    for c_axis, cname in ((d, ""), (side_c, "_side")):
                        geo = CPZ.axial_blade_geometry(z_push, (0.0, h), pitch, 0.0, side=c_axis is side_c)
                        if geo is None:
                            continue
                        for c_sign in (1.0, -1.0):
                            pre, init, mid, goal = CPZ.axial_push_chain(side_pt, d, geo[1], a_vec, c_axis, c_sign, th.tensor([float(com[0]), float(com[1]), z_push]),
                                                                       spin * turn)
                            cls = f"{'+' if end_sign > 0 else '-'}end{'+' if spin > 0 else '-'}spin_{pitch:.0f}{'' if c_sign > 0 else '-'}{cname}"
                            stroke = abs(turn) * float(th.linalg.norm(r_xy))
                            e = CPZ.contact_entry("push", cls, "axial_push", "work", T, init, pre, {"blade_half_thickness_m": geo[1], "fingertip_z": geo[2]},
                                                  face_rule="the face a long-axis end moves away from, at the push height", direction=d, stroke_m=stroke,
                                                  arc=True, support_z=0.0, object_height=h, pitch_deg=pitch, wrist_turn=c_sign, executor="push")
                            Ti = np.linalg.inv(T)
                            e["arc"] = {"centre_object": [round(float(v), 6) for v in (Ti[:3, :3] @ np.array([float(com[0]), float(com[1]), z_push]) + Ti[:3, 3])],
                                        "turn_rad": round(spin * turn, 6), "mid_object": _obj_pose(Ti, mid), "goal_object": _obj_pose(Ti, goal)}
                            e["yaw_deg"] = round(math.degrees(yaw), 4)
                            out.append(e)
    return out


def _obj_pose(Ti: np.ndarray, pose) -> Dict[str, List[float]]:
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    return CPZ._pose_dict(Ti @ CPZ._mat(pose))


#: family -> (kind, arm role, builder or None, the aliases it serves, why a builder is still missing)
FAMILIES: Dict[str, Dict[str, Any]] = {
    "touching_push": {"kind": "push", "arm_role": "work", "builder": build_touching_push, "scope": "touching small scale 01 / 02 / 04: the pushed targets"},
    "touching_brace": {"kind": "brace", "arm_role": "brace", "builder": build_touching_brace, "scope": "touching small scale 01 / 02 / 04: the braced partners"},
    "axial_push": {"kind": "push", "arm_role": "work", "builder": build_axial_push, "scope": "alignment-axial 01-05: the pushed targets"},
    "axial_brace": {"kind": "brace", "arm_role": "brace", "builder": build_axial_brace,
                    "scope": "alignment-axial 01-05: the stop brace holding the target at its goal (board_game qgvsnb 0.8126, paperback 0.6854, keyboard 0.7853, "
                             "magazine 0.7351, tablespoon tqqwys 0.8126)"},
    "block_push": {"kind": "push", "arm_role": "work", "builder": None, "scope": "block_design_01",
                   "missing": "block_design's push builder (block_design.py ~2960 / ~3110) is not factored into contact_poses yet"},
    "block_brace": {"kind": "brace", "arm_role": "brace", "builder": None, "scope": "block_design_01", "missing": "block_design's brace builder is not factored"},
    "floor_blade": {"kind": "push", "arm_role": "work", "builder": None, "scope": "under_large 01-03, touching_large_03",
                    "missing": "IORSLarge.contact_offset_along (the blade's contact offset per yaw) is a sampler method; the chain is shared (floor_push_chain)"},
    "floor_flat": {"kind": "push", "arm_role": "work", "builder": None, "scope": "push-under flat hand", "missing": "IORSLarge.flat_rungs / flat_turns are sampler methods"},
    "floor_pitched": {"kind": "push", "arm_role": "work", "builder": None, "scope": "push-under pitched-far hand",
                      "missing": "IORSLarge.pitched_turns / far_pitches are sampler methods"},
    "base_push": {"kind": "push", "arm_role": "both", "builder": None, "scope": "nextto_large 01-03, touching_large_01 (BasePush)",
                  "missing": "iors_large_nextto's face points (the push lanes) are sampler logic; the hand poses are shared (base_push_hand_poses)"},
    "wall_push": {"kind": "push", "arm_role": "work", "builder": None, "scope": "world_vertical (library only)",
                  "missing": "the wall layout (hinge corner, E_low, the arc) is alignment_wall sampler logic; the pose is shared (wall_contact_pose)"},
    "wall_brace": {"kind": "brace", "arm_role": "brace", "builder": None, "scope": "world_vertical (library only)", "missing": "as wall_push"},
}


# ------------------------------------------------------------------------------------------------ the object (v2 model directory)


def load_object(glib: Optional[str], job: Dict[str, Any]):
    """``(ModelLibrary, rest_id, geometry)`` of a job's object from a v2 model directory (``model_dir``: its absolute path; ``entry`` = the
    directory id under ``<glib>/models``; else ``schema.load_model`` by category / model / scale / constraint / rest hint); a library without grasps serves (the
    rest and the geometry are what a contact library needs). Raises ``LookupError`` naming what is missing."""
    from ispatialgym.grasping import object_geometry as OG

    if job.get("model_dir") or (job.get("entry") and glib):
        # ``model_dir``: an absolute model directory (the objects of one manifest may sit in several v2 roots)
        d = job.get("model_dir") or os.path.join(S.library_root(glib), "models", job["entry"])
        data = S.load_json(os.path.join(d, "library.json"))
        rests_d = (data or {}).get("rests") or (S.load_json(os.path.join(d, "rests.json")) or {}).get("rests", {})
        lib = None if data is None and not rests_d else S.ModelLibrary(path=os.path.join(d, "library.json"), data=data or {"category": job.get("category"),
                                                                                                                              "model": job.get("model"), "scale": job.get("scale")},
                                                                        rests={k: S.Rest.from_record(v) for k, v in rests_d.items()})
    else:
        lib = S.load_model(job["category"], job["model"], job.get("scale"), glib, constraint=job.get("constraint") or "tabletop", rest_hint_wxyz=job.get("rest_hint_wxyz"))
    if lib is None:
        raise LookupError("no v2 model directory (collision geometry + a measured rest): run the glib export + validate stages for the object")
    stable = sorted(r for r in lib.rests if lib.rests[r].stable)
    if not stable:
        raise LookupError(f"{os.path.dirname(lib.path)}: no stable measured rest")
    geom = OG.load(os.path.dirname(lib.path))
    if geom is None:
        raise LookupError(f"{os.path.dirname(lib.path)}: no exported geometry (geometry.json)")
    return lib, stable[0], geom


def geometry_digest(model_dir: str) -> Optional[str]:
    import hashlib

    p = os.path.join(model_dir, "geometry.json")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def hull_points(geom) -> np.ndarray:
    """The object's collision vertices in its own frame (unique to 1 um): the sampler's ``local_hull`` (collision boundary points) of the same
    meshes."""
    return np.unique(np.round(np.asarray(geom.union.vertices, dtype=float), 6), axis=0)


def resolved_knobs(family: str, knobs: Dict[str, Any], subtask: Optional[str]) -> Dict[str, Any]:
    """The pose set of a family: the knobs its rule reads, as the rule resolves them (the job's ``generation``, the subtask's
    generation_defaults, the code defaults)."""
    from ispatialgym.sampling.v2 import contact_poses as CPZ
    from ispatialgym.sampling.v2 import tabletop as TT

    gs = CPZ.PoseGen(knobs, subtask=subtask)
    keys = {"touching_push": ["stop_clearance_m", "push_tip_pre_dist_m", "push_blade_pitches_deg", "blade_toward_com", "blade_palm_min_pitch_deg",
                              "blade_palm_clear_fallback", "blade_lean_deg", "blade_lean_tip_above_m", "blade_palm_clear_first", "blade_palm_clear_m",
                              "contact_height_extra_m", "push_raised_max_aspect"],
            "touching_brace": ["stop_clearance_m", "brace_open_when_wide", "brace_tip_first", "push_tip_pre_dist_m", "brace_tip_min_above_m", "brace_wide_stop",
                               "brace_wide_spreads_m", "brace_tip_pitches_deg", "blade_palm_clear_m", "contact_height_extra_m"],
            "axial_push": ["contact_height_extra_m"], "axial_brace": ["contact_height_extra_m"]}.get(family, [])
    out = {k: gs.gen(k) for k in keys}
    if family == "touching_brace":
        # the wide stop's spreads as the rule reads them (ISPATIALGYM_BRACE_FULL_OPEN prepends the full-open 0.127 m)
        out.update(brace_wide_spreads_m=CPZ.brace_wide_spreads(gs), brace_full_open=CPZ.brace_full_open_on())
    if family == "touching_push":
        out.update({k: TT.phase2_gen(gs, k, None) for k in ("push_height_sweep", "push_height_step_m", "push_max_height_fraction", "push_gripper_open",
                                                              "push_open_opening_m")})
    return {k: v for k, v in out.items() if v is not None}


def build_job(job: Dict[str, Any], glib: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """The push / brace libraries (``{kind: library}``) of one job: ``{"id", category / model / scale / entry, "families": [...],
    "generation": {knobs of the alias family}, "subtask", "params": {family: {stroke_m, yaws, turn_deg, long_axis, com_local}}}``."""
    lib, rid, geom = load_object(glib, job)
    rest = lib.rests[rid]
    pts = hull_points(geom)
    model_dir = os.path.dirname(lib.path)
    obj = {"category": lib.data.get("category") or job.get("category"), "model": lib.data.get("model") or job.get("model"),
           "scale": [round(float(v), 6) for v in S.canonical_scale(lib.data.get("scale", job.get("scale")))], "entry": os.path.basename(model_dir),
           "source": model_dir, "rest_id": rid, "rest": {"quat_xyzw": [round(float(v), 7) for v in rest.quat_xyzw], "h_root": round(float(rest.h_root), 6)},
           "geometry": geometry_digest(model_dir), "n_hull_points": int(len(pts))}
    out: Dict[str, Dict[str, Any]] = {}
    for fam in job.get("families") or []:
        spec = FAMILIES.get(fam)
        if spec is None:
            raise KeyError(f"unknown family {fam!r} (one of {sorted(FAMILIES)})")
        if spec["builder"] is None:
            raise NotImplementedError(f"family {fam}: {spec['missing']}")
        knobs = dict(job.get("generation") or {})
        params = dict((job.get("params") or {}).get(fam) or {}, subtask=job.get("subtask"))
        t0 = time.time()
        entries = spec["builder"](pts, rest, knobs, params)
        lib_k = out.setdefault(spec["kind"], {"schema": LIB_SCHEMA, "kind": spec["kind"], "object": obj, "families": {}, "entries": [], "created": S.now()})
        lib_k["families"][fam] = {"arm_role": spec["arm_role"], "scope": spec["scope"], "knobs": resolved_knobs(fam, knobs, job.get("subtask") or TOUCHING_SUBTASK),
                                  "params": {k: v for k, v in params.items() if k != "subtask"}, "frame": "rule frame: +x the robot's forward, +y its left; support z 0",
                                  "yaws_deg": [round(math.degrees(y), 4) for y in yaw_slots(int(params.get("yaws", DEFAULT_YAWS)))], "n_entries": len(entries),
                                  "build_s": round(time.time() - t0, 3)}
        lib_k["entries"].extend(entries)
    for k, lib_k in out.items():
        lib_k["digest"] = _library_digest(lib_k)
    return out


def _library_digest(lib_k: Dict[str, Any]) -> str:
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    return CPZ.digest({"object": {k: v for k, v in lib_k["object"].items() if k not in ("source",)}, "families": {f: {k: v for k, v in d.items() if k != "build_s"}
                                                                                                                for f, d in lib_k["families"].items()},
                       "entries": [e["digest"] for e in lib_k["entries"]]})


def library_path(root: str, kind: str, entry_id: str) -> str:
    return os.path.join(root, kind, "models", entry_id, "library.json")


def save_library(root: str, lib_k: Dict[str, Any], job_id: Optional[str] = None) -> str:
    p = library_path(root, lib_k["kind"], job_id or lib_k["object"]["entry"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    S.save_json(p, lib_k)
    return p


def load_library(path: str) -> Dict[str, Any]:
    with open(path) as f:
        d = json.load(f)
    if d.get("schema") != LIB_SCHEMA:
        raise S.IncompatibleRecord(f"{path}: schema {d.get('schema')!r} is not {LIB_SCHEMA}")
    return d


# ------------------------------------------------------------------------------------------------ CLI


def _jobs(path: str, only: Optional[Sequence[str]], shard: Optional[str]) -> List[Dict[str, Any]]:
    with open(path) as f:
        jobs = json.load(f)
    if only:
        jobs = [j for j in jobs if j["id"] in set(only)]
    if shard:
        i, n = (int(v) for v in shard.split("/"))
        jobs = [j for k, j in enumerate(jobs) if k % n == i]
    return jobs


def _log(root: str, stage: str, shard: Optional[str], rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    with open(os.path.join(root, "logs", f"{stage}_{(shard or '0/1').replace('/', 'of')}.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(f"GB_CONTACT_{stage.upper()} {json.dumps(rec, default=str)[:800]}", flush=True)


def build_cli(a) -> None:
    for j in _jobs(a.jobs, a.only, a.shard):
        t0 = time.time()
        try:
            libs = build_job(j, a.glib)
            paths = {k: save_library(a.root, v, j["id"]) for k, v in libs.items()}
            _log(a.root, "build", a.shard, {"id": j["id"], "ok": True, "s": round(time.time() - t0, 2), "paths": paths,
                                            "entries": {k: len(v["entries"]) for k, v in libs.items()}, "families": j.get("families")})
        except Exception as e:  # noqa: BLE001
            _log(a.root, "build", a.shard, {"id": j["id"], "ok": False, "error": repr(e), "tb": traceback.format_exc()[-1000:]})
    print("GB_CONTACT_BUILD_DONE", flush=True)


def summary(root: str) -> List[Dict[str, Any]]:
    rows = []
    for kind in LIB_KINDS:
        base = os.path.join(root, kind, "models")
        for eid in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            p = os.path.join(base, eid, "library.json")
            if not os.path.exists(p):
                continue
            d = load_library(p)
            v = [e.get("validation", {}).get("verdict") for e in d["entries"]]
            rows.append({"kind": kind, "id": eid, "families": sorted(d["families"]), "entries": len(d["entries"]), "valid": v.count("valid"),
                         "invalid": v.count("invalid"), "unvalidated": v.count(None), "timing": d.get("validation_timing")})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--jobs", required=True)
    b.add_argument("--glib", required=True, help="the v2 library root the objects' model directories are in (geometry + rests)")
    b.add_argument("--root", required=True, help="the contact library root (push/, brace/, logs/)")
    b.add_argument("--only", nargs="*", default=None)
    b.add_argument("--shard", default=None)
    s = sub.add_parser("summary")
    s.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build_cli(a)
    else:
        for r in summary(a.root):
            print(json.dumps(r))


if __name__ == "__main__":
    main()
