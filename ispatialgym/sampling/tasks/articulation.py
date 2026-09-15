"""Configuration matching / articulation (``configuration_matching-1`` closed -> open, ``-2`` open -> closed).

Roles: ``artic`` (spawned non-scene object placed on the fixed ``table``, or a fixed scene object), ``floor``.
Handle geometry (contact point, pinch axis, approach direction, closed configuration) comes from one of

* the articulation pass of the grasp library, ``artic/handles/<handle_key>.json``, measured on the *unscaled*
  canonical model: every position is multiplied by the instance's per-axis scale (scene instances are non-uniformly
  scaled), directions are re-normalised after scaling and the pinch width scales with the axis along the pinch;
* a B4 scene-inventory record ``runlogs/scene_inventory_20260915/<scene>/artic/<scene object name>.json`` (the
  same joint / handle schema, measured in place -- instance frame, no further scaling): ``roles.artic.handle_file``
  names it explicitly, or it is resolved automatically for a scene object whose ``handle_key`` has no canonical file.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym import REPO_ROOT
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.grasp_library import library_root
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import TABLE_CATEGORIES, Draft, TaskSampler
from ispatialgym.sampling.tro_writer import full_pose_target

import omnigibson as og  # noqa: E402
import omnigibson.utils.transform_utils as T  # noqa: E402
from omnigibson.object_states import Open  # noqa: E402

log = logging.getLogger("ispatialgym.sampling")

#: B4 scene inventories (``ispatialgym.sampling.scene_inventory``): ``<scene>/artic/<name>.json`` per articulated scene object
SCENE_INVENTORY_ROOT = os.path.join(REPO_ROOT, "runlogs", "scene_inventory_20260915")
#: admissible pinch width after scaling (spec 5.3); outside it the handle is used but flagged
PINCH_WIDTH_RANGE = (0.015, 0.080)


def _vec(x, default=None) -> Optional[th.Tensor]:
    if x is None:
        return None if default is None else th.tensor(default, dtype=th.float32)
    return th.as_tensor(x, dtype=th.float32)


def _scaled_position(p, scale: th.Tensor):
    return (_vec(p) * scale).tolist()


def _scaled_direction(d, scale: th.Tensor):
    """Direction of a scaled edge: ``normalize(scale * d)`` (identical to ``d`` for axis-aligned directions)."""
    v = _vec(d) * scale
    n = float(th.linalg.norm(v))
    return (v / n).tolist() if n > 1e-9 else list(d)


def _length_scale(direction, scale: th.Tensor) -> float:
    """Factor a length measured along ``direction`` (canonical frame) grows by under the per-axis ``scale``."""
    d = _vec(direction)
    n = float(th.linalg.norm(d))
    if n < 1e-9:
        return 1.0
    return float(th.linalg.norm(d / n * scale))


def scale_handle(h: Dict[str, Any], scale: th.Tensor) -> Dict[str, Any]:
    """Canonical-frame handle record (``artic/handles``) expressed in the frame of an instance with per-axis ``scale``:
    positions scaled per axis, directions re-normalised, ``width`` (across the pinch) / ``depth`` (along the approach) /
    ``length`` (along the bar) scaled with the axis they are measured along."""
    out = dict(h)
    if h.get("contact") is not None:
        out["contact"] = _scaled_position(h["contact"], scale)
    if h.get("center") is not None:  # flush handles
        out["center"] = _scaled_position(h["center"], scale)
    for k in ("pinch_axis", "approach_dir", "pull_dir"):
        if h.get(k) is not None:
            out[k] = _scaled_direction(h[k], scale)
    if h.get("width") is not None and h.get("pinch_axis") is not None:
        out["width"] = float(h["width"]) * _length_scale(h["pinch_axis"], scale)
    if h.get("depth") is not None and h.get("approach_dir") is not None:
        out["depth"] = float(h["depth"]) * _length_scale(h["approach_dir"], scale)
    if h.get("length") is not None and h.get("pinch_axis") is not None and h.get("approach_dir") is not None:
        along = th.linalg.cross(_vec(h["pinch_axis"]), _vec(h["approach_dir"]))
        out["length"] = float(h["length"]) * _length_scale(along, scale)
    if h.get("face_s") is not None and h.get("approach_dir") is not None:
        out["face_s"] = float(h["face_s"]) * _length_scale(h["approach_dir"], scale)
    return out


def scale_joint_info(j: Dict[str, Any], scale: th.Tensor) -> Dict[str, Any]:
    """Canonical-frame joint record scaled like :func:`scale_handle` (outward normal / axis re-normalised, hinge point
    and link bounding box scaled per axis, nested handle candidates scaled too)."""
    out = dict(j)
    for k in ("outward_normal", "axis_dir_obj"):
        if j.get(k) is not None:
            out[k] = _scaled_direction(j[k], scale)
    for k in ("hinge_point_obj", "link_bbox_min", "link_bbox_max"):
        if j.get(k) is not None:
            out[k] = _scaled_position(j[k], scale)
    if j.get("face_s") is not None and j.get("outward_normal") is not None:
        out["face_s"] = float(j["face_s"]) * _length_scale(j["outward_normal"], scale)
    out["handles"] = [scale_handle(h, scale) for h in j.get("handles", []) or []]
    out["flush_handles"] = [scale_handle(h, scale) for h in j.get("flush_handles", []) or []]
    out["fallback"] = scale_handle(j["fallback"], scale) if j.get("fallback") else j.get("fallback")
    out["scaled_by"] = scale.tolist()
    return out


class ArticulationSampler(TaskSampler):
    #: handles sit well above the table top, so standing trunk translates are tried first
    DEFAULT_REACH = [0.0, 0.25, 0.5, 0.75, 1.0]

    def __init__(self, ctx):
        super().__init__(ctx)
        self.direction = "open" if self.definition_id == 1 else "close"
        self.scene_object = "artic" in [r.name for r in self.spec.fixed_roles()]
        self.top = None
        self.stance = None
        self.cells = None
        self.hull = None
        self.handle = None
        self.joint_name = None
        self.stable_angles: List[float] = []

    # ------------------------------------------------------------------ template

    def bind_fixed_roles(self):
        self.bind_floor()
        if self.scene_object:
            cands = self.fixed_candidates("artic", ())
            if not cands:
                raise Reject("spec", "no scene articulated object bound")
            self.bind_scene_object("artic", cands[0])
        else:
            self._table_candidates = self.fixed_candidates("table", TABLE_CATEGORIES)

    def _handle_record(self, obj, key: str) -> Tuple[Dict[str, Any], str, str]:
        """``(record, path, frame)`` of the handle annotation for ``obj``: ``roles.artic.handle_file`` when given
        (absolute, or relative to the repository root / the cwd), else the canonical ``artic/handles/<key>.json``, else --
        scene objects only -- the B4 per-instance record ``<inventory>/<scene>/artic/<name>.json``. ``frame`` is
        ``"instance"`` for records measured in place (they carry ``scene`` / ``scale``) and ``"canonical"`` otherwise."""

        def read(path):
            with open(path) as f:
                data = json.load(f)
            frame = "instance" if data.get("scene") else "canonical"
            return data, path, frame

        explicit = self.spec.roles["artic"].options.get("handle_file")
        if explicit:
            cands = [explicit] if os.path.isabs(explicit) else [os.path.join(REPO_ROOT, explicit), os.path.abspath(explicit)]
            for p in cands:
                if os.path.exists(p):
                    return read(p)
            raise Reject("spec", f"handle_file {explicit!r} for {obj.name} not found (tried {cands})")
        tried = []
        canonical = os.path.join(library_root(), "artic", "handles", f"{key}.json")
        if os.path.exists(canonical):
            return read(canonical)
        tried.append(canonical)
        if self.scene_object:
            inst = os.path.join(SCENE_INVENTORY_ROOT, self.world.scene_model, "artic", f"{obj.name}.json")
            if os.path.exists(inst):
                return read(inst)
            tried.append(inst)
        raise Reject("spec", f"no handle annotation for {key} ({obj.name}): neither the canonical file nor a B4 instance record exists ({tried})")

    def _load_handle(self, obj):
        role = self.spec.roles["artic"]
        key = role.options.get("handle_key", f"{obj.category}__{obj.model}")
        data, path, frame = self._handle_record(obj, key)
        joint_name = role.options.get("joint")
        joints = [j for j in data["joints"] if joint_name is None or j["joint"] == joint_name]
        if not joints:
            raise Reject("spec", f"joint {joint_name!r} not in handle annotation {os.path.relpath(path, REPO_ROOT)}")
        j = joints[0]
        handles = [h for h in j.get("handles", []) if h.get("kind") in ("bar", "knob")] or ([j["fallback"]] if j.get("fallback") else [])
        if not handles:
            raise Reject("spec", f"no graspable handle for {key}/{j['joint']} in {os.path.relpath(path, REPO_ROOT)}")
        scale = th.as_tensor([float(s) for s in obj.scale], dtype=th.float32) if getattr(obj, "scale", None) is not None else th.ones(3)
        if frame == "canonical":
            # the library annotation lives in the unscaled model frame; the instance (scene object, or a spawned role
            # with ``scale``) is scaled per axis about the same origin
            handle, jinfo = scale_handle(handles[0], scale), scale_joint_info(j, scale)
        else:
            handle, jinfo = dict(handles[0]), dict(j)
            if data.get("name") and data["name"] != obj.name:
                log.warning("%s: handle record %s was measured on %s, used for %s", self.spec.name, os.path.relpath(path, REPO_ROOT), data["name"], obj.name)
            rec_scale = data.get("scale")
            if rec_scale is not None and float((th.as_tensor(rec_scale, dtype=th.float32) - scale).abs().max()) > 0.01:
                log.warning("%s: handle record scale %s differs from the instance scale %s", self.spec.name, rec_scale, scale.tolist())
        jinfo["hinge_point_obj"] = self._sane_hinge_point(obj, jinfo)
        self.joint_name = j["joint"]
        self.handle = handle
        self.joint_info = jinfo
        width = handle.get("width")
        if width is not None and not (PINCH_WIDTH_RANGE[0] <= float(width) <= PINCH_WIDTH_RANGE[1]):
            log.warning("%s: %s/%s pinch width %.1f mm outside %s mm after scaling", self.spec.name, obj.name, self.joint_name, 1000 * float(width),
                        [1000 * w for w in PINCH_WIDTH_RANGE])
        self.handle_source = {"path": os.path.relpath(path, REPO_ROOT), "frame": frame, "instance_scale": scale.tolist(), "joint": self.joint_name,
                              "kind": handle.get("kind"), "contact_obj": handle.get("contact"), "pinch_axis_obj": handle.get("pinch_axis"),
                              "approach_dir_obj": handle.get("approach_dir"), "width": width, "hinge_point_obj": jinfo.get("hinge_point_obj"),
                              "axis_dir_obj": jinfo.get("axis_dir_obj")}
        self.template_meta["handle_source"] = self.handle_source
        log.info("%s: handle for %s/%s from %s (%s frame, instance scale %s): %s at %s, pinch %s, approach %s, width %.1f mm", self.spec.name, obj.name,
                 self.joint_name, self.handle_source["path"], frame, [round(s, 4) for s in scale.tolist()], handle.get("kind"),
                 [round(v, 4) for v in handle["contact"]], [round(v, 3) for v in handle["pinch_axis"]], [round(v, 3) for v in handle["approach_dir"]],
                 1000 * float(width or 0.0))

    def _sane_hinge_point(self, obj, jinfo: Dict[str, Any]):
        """Hinge point of a revolute joint on the plane through the moving link's bounding-box centre. The inventories
        recover the hinge by least squares from a small rotation, which is unconstrained along the axis: the point can
        run away along it (B4 records show z ~ 1e8 for vertical hinges), so it is projected back; a point still > 1 m off
        the link (the run-away also tilted x / y) is re-measured from the joint prim (child link pose and the joint's
        local frame on it), falling back to the projected value with a warning."""
        hinge, axis = _vec(jinfo.get("hinge_point_obj")), _vec(jinfo.get("axis_dir_obj"))
        if hinge is None or axis is None or float(th.linalg.norm(axis)) < 1e-9:
            return jinfo.get("hinge_point_obj")
        if jinfo.get("link_bbox_min") is None or jinfo.get("link_bbox_max") is None:
            return hinge.tolist()
        lo, hi = _vec(jinfo["link_bbox_min"]), _vec(jinfo["link_bbox_max"])
        centre = (lo + hi) / 2
        a = axis / th.linalg.norm(axis)
        proj = hinge if not bool(th.isfinite(hinge).all()) else hinge + a * th.dot(centre - hinge, a)
        if bool(th.isfinite(proj).all()) and float(th.linalg.norm(proj - centre)) <= 1.0:
            return proj.tolist()
        measured = self._hinge_from_prim(obj, jinfo.get("joint"))
        if measured is not None:
            m = measured + a * th.dot(centre - measured, a)
            if bool(((m >= lo - 0.15) & (m <= hi + 0.15)).all()):
                log.info("%s: hinge point of %s/%s re-measured from the joint prim: %s (record %s)", self.spec.name, obj.name, jinfo.get("joint"),
                         [round(v, 4) for v in m.tolist()], hinge.tolist())
                return m.tolist()
        log.warning("%s: hinge point of %s/%s is degenerate in the annotation (%s); keeping the projection %s", self.spec.name, obj.name,
                    jinfo.get("joint"), hinge.tolist(), proj.tolist())
        return proj.tolist()

    def _hinge_from_prim(self, obj, joint_name: Optional[str]) -> Optional[th.Tensor]:
        """A point on the joint axis in the object (root link) frame from the live joint prim: the joint frame sits at
        ``localPos1`` in the child link's frame (scaled with the link, as PhysX does)."""
        try:
            joint = obj.joints[joint_name]
            child = obj.links[joint.body1.split("/")[-1]]
            lp, lq = child.get_position_orientation()
            lscale = child.get_world_scale() if hasattr(child, "get_world_scale") else th.ones(3)
            p_world = lp + G.rotate(lq, joint.local_position_1 * lscale)
            rp, rq = obj.get_position_orientation()
            return G.rotate(G.qinv(rq), p_world - rp)
        except Exception as e:  # noqa: BLE001 -- best effort, the record's value is kept
            log.debug("hinge from prim failed for %s/%s: %r", obj.name, joint_name, e)
            return None

    def prepare_bindings(self):
        artic = self.obj("artic")
        self._load_handle(artic)
        self.hull = self.world.local_hull(artic)
        if self.scene_object:
            self._prepare_scene_object(artic)
            return
        errors = []
        for table in self._table_candidates:
            try:
                self._try_support(table)
                self.bind_scene_object("table", table)
                return
            except Reject as e:
                errors.append(f"{table.name}: {e}")
        raise Reject("no_support", "; ".join(errors))

    def _handle_eef_poses(self, obj_pose: G.Pose, pregrasp: float = 0.10) -> List[G.Pose]:
        """[pre-grasp, grasp] eef poses for the annotated handle: approach along ``approach_dir``, fingers closing
        along ``pinch_axis`` (object frame, closed configuration)."""
        h = self.handle
        contact = G.compose(obj_pose, (th.tensor(h["contact"], dtype=th.float32), G.IDENTITY_QUAT))[0]
        approach = G.rotate(obj_pose[1], th.tensor(h["approach_dir"], dtype=th.float32))
        pinch = G.rotate(obj_pose[1], th.tensor(h["pinch_axis"], dtype=th.float32))
        z = approach / th.linalg.norm(approach)
        y = pinch - th.dot(pinch, z) * z
        y = y / th.linalg.norm(y)
        x = th.cross(y, z, dim=0)
        quat = T.mat2quat(th.stack([x, y, z], dim=1))
        eef = contact - z * 0.012  # pad centre 1.2 cm behind the eef origin along the approach
        return [(eef - z * pregrasp, quat), (eef, quat)]

    def _facing_dir_local(self) -> th.Tensor:
        """Object-frame horizontal direction that should point at the robot: the outward normal of the moving face
        (doors / drawers); for lids (vertical outward normal) the direction from the hinge line to the object centre,
        so the lid swings away from the robot."""
        outward = th.tensor(self.joint_info.get("outward_normal", [1.0, 0.0, 0.0]), dtype=th.float32)
        if abs(float(outward[2])) > 0.9:
            axis = th.tensor(self.joint_info.get("axis_dir_obj", [0.0, 1.0, 0.0]), dtype=th.float32)
            hinge = th.tensor(self.joint_info.get("hinge_point_obj", [0.0, 0.0, 0.0]), dtype=th.float32)
            perp = th.cross(axis, th.tensor([0.0, 0.0, 1.0]), dim=0)
            centre = self.hull.mean(dim=0)
            if float(th.dot(perp[:2], (centre - hinge)[:2])) < 0:
                perp = -perp
            outward = perp
        outward[2] = 0.0
        return outward / th.linalg.norm(outward)

    def _try_support(self, table):
        self.kit.sync_world()
        top = self.support_top(table)
        ext = self.hull.max(dim=0).values - self.hull.min(dim=0).values
        facing = self._facing_dir_local()
        # half extents of the (yaw-aligned) object along the robot's depth / lateral directions
        pts = self.hull[:, :2]
        d_half = float((pts @ facing[:2]).abs().max())
        side = th.tensor([-float(facing[1]), float(facing[0])])
        l_half = float((pts @ side).abs().max())
        depth_range = tuple(self.gen("depth_range", [d_half + 0.03, d_half + 0.20]))
        # wide objects need the camera high and behind the edge (standing trunk) to fit in the image while the handle
        # stays within arm reach; try the stance stand-offs in order
        errors = []
        stance, cells = None, None
        for stand_off in self.gen("stand_offs", [0.45, 0.55]):
            try:
                st = self.stance_with_view(top, sum(depth_range) / 2, preferred_edge=self.scene_spec.fixed.get("table_front_edge"), focus_z=float(ext[2]) / 2,
                                           stand_off=float(stand_off), prefer_translate=float(self.gen("view_prefer_translate", 0.0)))
            except Reject as e:
                errors.append(str(e))
                continue
            # the handle is worked with the right arm, whose reachable strip lies to the robot's right of the centre
            # line: offset the object that way (lateral_center < 0 = robot's right)
            region = self.region_cells(top, st, depth_range, float(self.gen("lateral_half", 0.15)), margin=min(d_half, l_half) + 0.02,
                                       lateral_center=float(self.gen("lateral_center", -0.15)))
            if len(region) == 0:
                errors.append(f"stand_off {stand_off}: placement region empty")
                continue
            # object yaw such that ``facing`` points at the robot (= -inward); keep cells where the yawed object is fully
            # supported and its whole hull is inside the image (exact hull, not a bounding disc: the object is wide)
            face_yaw = math.atan2(float(-st.inward[1]), float(-st.inward[0])) - math.atan2(float(facing[1]), float(facing[0]))
            keep, n_vis = [], 0
            for c in region:
                pose = self._pose_on(c, face_yaw, top.z)
                if not self.frustum_ok(st.view.cam_pose, self.hull_corners_world(self.hull, pose), float(self.gen("view_margin", 0.03))):
                    continue
                n_vis += 1
                if top.polygon_supported(self.world.footprint(self.hull, pose)):
                    keep.append(c)
            if keep:
                stance, cells, self.face_yaw = st, th.stack(keep), face_yaw
                break
            errors.append(f"stand_off {stand_off}: {n_vis} visible of {len(region)} cells, none supported+visible (view translate {st.view.translate})")
        if stance is None:
            raise Reject("support", "; ".join(errors))
        # handle reachability per coarse cell (exact handle poses, one yaw). Wide objects are viewed from further back
        # than the arm reaches, so the manipulation stance may be the viewing stance driven straight forward by up to
        # ``approach_forward`` (the oracle drives before reaching); the first (forward, translate) that works is kept.
        coarse = self.coarse_cells(cells, float(self.gen("reach_resolution", 0.06)), top.resolution)
        translates = list(self.gen("reach_translates", self.DEFAULT_REACH))
        pairs = [self._handle_eef_poses(self._pose_on(xy, self.face_yaw, top.z)) for xy in coarse]
        ok_cells, tr_of, fwd_of = [], [], []
        for fwd in self.gen("approach_forward", [0.0, 0.10, 0.20]):
            st_f = self._shifted(stance, float(fwd))
            if fwd and not self.kit.trav_free(st_f.base_xy):
                continue
            for tr in self.stance_free_translates(st_f, translates):
                todo = [i for i in range(len(pairs)) if i not in ok_cells]
                if not todo:
                    break
                feas = self.kit.pairs_feasible([pairs[i] for i in todo], self.reach_seed(st_f, tr), ignore_for_grasp=[table])
                for i, s in zip(todo, feas.tolist()):
                    if s:
                        ok_cells.append(i)
                        tr_of.append(float(tr))
                        fwd_of.append(float(fwd))
            if ok_cells:
                break
        if not ok_cells:
            h0 = pairs[0][-1][0].tolist() if pairs else None
            raise Reject("unreachable", f"handle not reachable from any of {len(coarse)} cells (e.g. handle at {h0}, base {stance.base_pose2d}, "
                                        f"translates {self.stance_free_translates(stance, translates)})")
        self.top, self.stance = top, stance
        self.cells = coarse[ok_cells]
        self.reach_translates = tr_of
        self.reach_forward = fwd_of
        self.template_meta.update({"support": top.to_json(), "stance": stance.to_json(), "n_reachable_cells": len(ok_cells), "handle": self.handle, "joint": self.joint_name,
                                   "approach_forward": sorted(set(fwd_of))})

    def _shifted(self, stance, forward: float):
        """Copy of ``stance`` with the base driven ``forward`` metres toward the support."""
        if not forward:
            return stance
        from ispatialgym.sampling.tasks.base import Stance

        bxy = stance.base_xy + stance.inward * forward
        base = (float(bxy[0]), float(bxy[1]), stance.base_pose2d[2])
        return Stance(stance.edge_name, stance.edge_mid, stance.inward, stance.tangent, base, self.kit.q_at_base_pose(stance.q_stance, *base), stance.view)

    def _pose_on(self, xy, yaw: float, z: float) -> G.Pose:
        quat = G.quat_from_yaw(yaw)
        pos = th.tensor([float(xy[0]), float(xy[1]), z + self.world.bottom_offset(self.hull, quat) + 0.003])
        return pos, quat

    def _handle_world_check(self, artic, pose: G.Pose) -> Dict[str, Any]:
        """Where the annotated handle lands in the world for ``artic`` at ``pose``: contact / grasp eef / pre-grasp
        positions against the moving link's AABB (the contact must lie inside it, inflated by 3 cm, for the hand to
        land on the handle) -- logged and recorded in ``template_meta`` so that a wrong frame / scale is visible."""
        contact = G.compose(pose, (th.tensor(self.handle["contact"], dtype=th.float32), G.IDENTITY_QUAT))[0]
        pre, grasp = self._handle_eef_poses(pose)
        # the moving link plus the links hanging off it (separate handle links): their joint AABB
        names = [self.joint_info.get("child_link")] + [s.get("link") for s in self.joint_info.get("sub_links", []) or []]
        links = [artic.links[n] for n in names if n and n in artic.links]
        out = {"contact_world": [round(v, 4) for v in contact.tolist()], "grasp_eef_world": [round(v, 4) for v in grasp[0].tolist()],
               "pregrasp_eef_world": [round(v, 4) for v in pre[0].tolist()], "link": self.joint_info.get("child_link")}
        if links:
            aabbs = [l.aabb for l in links]
            lo = th.stack([a[0] for a in aabbs]).min(dim=0).values
            hi = th.stack([a[1] for a in aabbs]).max(dim=0).values
            margin = 0.03
            inside = bool(((contact >= lo - margin) & (contact <= hi + margin)).all())
            gap = th.maximum(th.maximum(lo - contact, contact - hi), th.zeros(3))
            out.update({"link_aabb": [[round(v, 4) for v in lo.tolist()], [round(v, 4) for v in hi.tolist()]], "contact_inside_link_aabb": inside,
                        "contact_outside_by_m": [round(v, 4) for v in gap.tolist()]})
        log.info("%s: handle check for %s/%s: %s", self.spec.name, artic.name, self.joint_name, out)
        self.template_meta["handle_check"] = out
        return out

    def _prepare_scene_object(self, artic):
        """Scene object: stance in front of the opening side with the handle reachable (may drive straight forward)."""
        self.kit.sync_world()
        pose = tuple(x.float() for x in artic.get_position_orientation())
        outward = G.rotate(pose[1], th.tensor(self.joint_info.get("outward_normal", [1.0, 0.0, 0.0]), dtype=th.float32))
        outward[2] = 0.0
        outward = outward / th.linalg.norm(outward)
        centre = artic.aabb_center.float()
        handle_pairs = self._handle_eef_poses(pose)
        check = self._handle_world_check(artic, pose)
        if check.get("contact_inside_link_aabb") is False:
            raise Reject("spec", f"handle contact {check['contact_world']} is outside the moving link {check['link']} AABB {check['link_aabb']} "
                                 f"(by {check['contact_outside_by_m']} m): wrong annotation frame / scale")
        self.template_meta.update({"handle": self.handle, "joint": self.joint_name})
        best = None
        # where the candidates die, for the rejection message (a tall fridge seen from 0.6-1.0 m never fits the frame:
        # the view distances and the drive-forward before reaching are spec knobs, ``scene_distances`` / ``scene_forward``)
        stats = {"candidates": 0, "trav_free": 0, "in_floor_room": 0, "base_collision_free": 0, "views": 0, "view_collision_free": 0, "in_frame": 0,
                 "reach_tested": 0}
        lo, hi = artic.aabb
        corners = G.aabb_corners(lo, hi)
        forwards = [float(f) for f in self.gen("scene_forward", [0.0, 0.15, 0.3])]
        # the init stance must stand on the bound floor: ``(ontop agent floor)`` is an init literal, and a stance that
        # backs off into the neighbouring room (open kitchens) fails it on every attempt
        floor_rooms = set(self.obj("floor").in_rooms or []) if self.bindings.has("floor") else set()
        seg_map = getattr(self.world.scene, "seg_map", None)

        def on_floor(xy) -> bool:
            if not floor_rooms or seg_map is None:
                return True
            try:
                return seg_map.get_room_instance_by_point(th.as_tensor([float(xy[0]), float(xy[1])])) in floor_rooms
            except Exception:  # noqa: BLE001
                return True

        # ``scene_face_centre``: turn the base toward the object's centre from lateral stances (a tall object seen from
        # the side leaves the frame with the face-normal yaw), by at most ``scene_max_yaw_deg`` off the face normal
        face_centre = bool(self.gen("scene_face_centre", False))
        max_yaw = math.radians(float(self.gen("scene_max_yaw_deg", 20.0)))
        yaw_normal = math.atan2(float(-outward[1]), float(-outward[0]))
        for dist in self.gen("scene_distances", [0.6, 0.7, 0.8, 0.9, 1.0]):
            for lateral in self.gen("scene_laterals", [0.0, -0.2, 0.2, -0.35]):
                stats["candidates"] += 1
                tangent = th.tensor([-float(outward[1]), float(outward[0])])
                base_xy = centre[:2] + outward[:2] * float(dist) + tangent * float(lateral)
                yaw = yaw_normal
                if face_centre:
                    to_c = centre[:2] - base_xy
                    d_yaw = G.wrap_angle(math.atan2(float(to_c[1]), float(to_c[0])) - yaw_normal)
                    yaw = G.wrap_angle(yaw_normal + max(-max_yaw, min(max_yaw, d_yaw)))
                if not self.kit.trav_free(base_xy):
                    continue
                stats["trav_free"] += 1
                if not on_floor(base_xy):
                    continue
                stats["in_floor_room"] += 1
                q_base = self.kit.q_at_base_pose(self.kit.q_config(arms=self.init_arms()), float(base_xy[0]), float(base_xy[1]), yaw)
                if not bool(self.kit.collision_free(q_base, margin=0.01)[0]):
                    continue
                stats["base_collision_free"] += 1
                views = self.kit.solve_view(q_base, centre, translates=th.tensor(self.gen("view_translates", [0.0, 0.25, 0.5, 0.75])), top_k=12)
                stats["views"] += len(views)
                for v in views:
                    q = self.kit.with_trunk(q_base, v.translate, v.tilt)
                    if not bool(self.kit.collision_free(q, margin=0.01)[0]):
                        continue
                    stats["view_collision_free"] += 1
                    cam = self.kit.camera_pose(q)
                    if not self.frustum_ok(cam, corners, float(self.gen("view_margin", 0.03))):
                        continue
                    stats["in_frame"] += 1
                    # reach: from the stance or after driving straight forward (the oracle drives before reaching)
                    for fwd in forwards:
                        bxy = base_xy - outward[:2] * fwd
                        if fwd and not self.kit.trav_free(bxy):
                            continue
                        stance_q = self.kit.q_at_base_pose(self.kit.q_config(arms=self.REACH_ARMS), float(bxy[0]), float(bxy[1]), yaw)
                        tr_ok = None
                        stats["reach_tested"] += 1
                        for tr in self.gen("reach_translates", self.DEFAULT_REACH):
                            if self.kit.reachable(handle_pairs, self.kit.with_trunk(stance_q, tr, 0.0), ignore_for_last=[artic]) is not None:
                                tr_ok = tr
                                break
                        # handle contact in the base frame (x forward, y = robot's left, z) for the reach-volume bands
                        rel = handle_pairs[-1][0] - th.tensor([float(bxy[0]), float(bxy[1]), 0.0])
                        hb = [float(rel[0] * math.cos(yaw) + rel[1] * math.sin(yaw)), float(-rel[0] * math.sin(yaw) + rel[1] * math.cos(yaw)), float(rel[2])]
                        log.info("%s: stance d=%.2f lat=%.2f yaw=%.2f view(tr=%.2f tilt=%.2f) fwd=%.2f -> handle eef in base frame %s: %s", self.spec.name,
                                 float(dist), float(lateral), yaw, v.translate, v.tilt, fwd, [round(x, 3) for x in hb],
                                 f"reachable at translate {tr_ok}" if tr_ok is not None else "unreachable")
                        if tr_ok is not None:
                            best = {"base_pose2d": (float(base_xy[0]), float(base_xy[1]), yaw), "q": q, "view": v, "forward": fwd, "translate": tr_ok,
                                    "distance": float(dist), "lateral": float(lateral)}
                            break
                    if best:
                        break
                if best:
                    break
            if best:
                break
        self.template_meta["scene_stance_search"] = stats
        if best is None:
            raise Reject("unreachable", f"no stance in front of {artic.name} sees the whole object and reaches the handle "
                                        f"(contact {check['contact_world']}); candidates surviving each stage: {stats}")
        log.info("%s: scene stance for %s: distance %.2f lateral %.2f forward %.2f translate %.2f (%s)", self.spec.name, artic.name, best["distance"],
                 best["lateral"], best["forward"], best["translate"], stats)
        self.scene_stance = best
        self.template_meta.update({"scene_stance": {k: (v if not isinstance(v, th.Tensor) else v.tolist()) for k, v in best.items() if k not in ("q", "view")}})

    def prepare_template(self):
        artic = self.obj("artic")
        self.joint = artic.joints[self.joint_name]
        self.lower, self.upper = float(self.joint.lower_limit), float(self.joint.upper_limit)
        self.stable_angles = [self.upper] if os.environ.get("ISG_SKIP_STABLE_PROBE") else self._stable_angles(artic)
        if not self.stable_angles:
            raise Reject("articulation", f"{artic.name}/{self.joint_name} holds no open angle passively")
        self.template_meta["stable_open_fractions"] = [(a - self.lower) / (self.upper - self.lower) for a in self.stable_angles]

    def _stable_angles(self, artic) -> List[float]:
        """Joint positions (fractions of the range from the spec) at which the moving link holds still for 120 physics
        steps; the upper limit is always tested last (resting on the stop)."""
        state = self.world.snapshot()
        lo, hi = self.gen("open_fraction_range", [0.5, 0.95])
        fracs = [float(f) for f in th.linspace(lo, hi, 7).tolist()] + [1.0]
        tol = self._joint_tol()
        out = []
        idx = list(artic.joints).index(self.joint_name)
        try:
            if not self.scene_object:
                # test on a stance cell (gravity direction is what matters, not the exact place)
                pose = self._pose_on(self.cells[0], self.face_yaw, self.top.z)
                artic.set_position_orientation(*pose)
                self.world.activate(artic)
                self.world.step_physics(30, still=[self.robot])
            for f in fracs:
                q = self.lower + f * (self.upper - self.lower)
                self._set_joint(artic, q)
                self.world.step_physics(120, still=[self.robot])
                q_after = float(artic.get_joint_positions()[idx])
                log.info("%s: joint %s at fraction %.2f (%.3f) -> %.3f after 120 steps", self.spec.name, self.joint_name, f, q, q_after)
                if abs(q_after - q) <= 0.5 * tol:
                    out.append(q)
        finally:
            self.world.restore(state)
            if not self.scene_object:
                self.world.park(artic, list(self.world.spawned).index(artic.name))
        return out

    def _set_joint(self, artic, q: float):
        qs = artic.get_joint_positions().clone()
        qs[list(artic.joints).index(self.joint_name)] = q
        artic.set_joint_positions(qs)
        artic.set_joint_velocities(th.zeros_like(qs))
        artic.keep_still()

    def _joint_tol(self) -> float:
        tols = self.gen("tolerances", {})
        return float(tols.get("revolute" if self.joint.is_revolute else "prismatic", 0.10 if self.joint.is_revolute else 0.02))

    # ------------------------------------------------------------------ instance

    def sample(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        aname = self.bindings.one("artic").name
        artic = self.obj("artic")
        joint_names = list(artic.joints)
        q_closed = artic.get_joint_positions().clone()
        for n, j in artic.joints.items():
            q_closed[joint_names.index(n)] = float(j.lower_limit)
        q_open_val = self.stable_angles[int(rng.integers(len(self.stable_angles)))]
        if self.direction == "open":
            q_init, q_goal_val = q_closed.clone(), q_open_val
        else:
            q_init = q_closed.clone()
            q_init[joint_names.index(self.joint_name)] = q_open_val
            q_goal_val = float(self.joint.lower_limit)
        q_goal = q_closed.clone()
        q_goal[joint_names.index(self.joint_name)] = q_goal_val
        d.object_joints[aname] = q_init
        d.goal_joints[aname] = q_goal
        tol = self._joint_tol()
        d.articulation_targets[aname] = {"joints": {n: {"position": float(q_goal[joint_names.index(n)]), "tolerance": tol} for n in joint_names}}
        if self.scene_object:
            st = self.scene_stance
            d.robot_q = st["q"]
            d.base_pose2d = st["base_pose2d"]
            d.settle_free = []
            d.summary.update({"direction": self.direction, "joint": self.joint_name, "q_open": q_open_val, "scene_stance": {k: v for k, v in st.items() if k in ("forward", "translate")}})
            return d
        k = int(rng.integers(len(self.cells)))
        pose = self._pose_on(self.cells[k], self.face_yaw + float(rng.uniform(-0.1, 0.1)), self.top.z)
        if not self.top.polygon_supported(self.world.footprint(self.hull, pose)):
            raise Reject("target_unsupported")
        if not self.frustum_ok(self.stance.view.cam_pose, self.hull_corners_world(self.hull, pose), 0.05):
            raise Reject("out_of_view")
        st_reach = self._shifted(self.stance, self.reach_forward[k])
        tr_ok = self.reachable_poses(st_reach, self._handle_eef_poses(pose), [self.reach_translates[k]] + list(self.gen("reach_translates", self.DEFAULT_REACH)))
        if tr_ok is None:
            raise Reject("grasp_unreachable", "handle not reachable at the sampled pose")
        d.object_poses[aname] = pose
        d.robot_q = self.stance.q_stance
        d.base_pose2d = self.stance.base_pose2d
        d.pose_targets[aname] = full_pose_target(pose[0], pose[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
        d.summary.update({"direction": self.direction, "joint": self.joint_name, "q_open": q_open_val, "handle_translate": tr_ok,
                          "approach_forward": self.reach_forward[k], "view": {"translate": self.stance.view.translate, "tilt": self.stance.view.tilt}})
        return d

    def apply_draft(self, draft: Draft):
        super().apply_draft(draft)
        # hold the moving link at its init position during settling (position control is not available on objects)
        artic = self.obj("artic")
        aname = self.bindings.one("artic").name
        self._set_joint(artic, float(draft.object_joints[aname][list(artic.joints).index(self.joint_name)]))

    def after_settle(self, draft: Draft):
        aname = self.bindings.one("artic").name
        if aname in draft.pose_targets:
            p, q = self.obj("artic").get_position_orientation()
            draft.pose_targets[aname] = full_pose_target(p, q, self.tol("position", 0.03), self.tol("orientation", 0.20))
        q_now = float(self.obj("artic").get_joint_positions()[list(self.obj("artic").joints).index(self.joint_name)])
        q_want = float(draft.object_joints[aname][list(self.obj("artic").joints).index(self.joint_name)])
        if abs(q_now - q_want) > 0.5 * self._joint_tol():
            raise Reject("articulation", f"init joint drifted from {q_want:.3f} to {q_now:.3f}")

    def apply_goal(self, draft: Draft):
        super().apply_goal(draft)

    def refine_targets_from_goal(self, draft: Draft):
        aname = self.bindings.one("artic").name
        artic = self.obj("artic")
        q = artic.get_joint_positions()
        names = list(artic.joints)
        tgt = draft.articulation_targets[aname]["joints"]
        for n in names:
            if abs(float(q[names.index(n)]) - tgt[n]["position"]) > 0.5 * tgt[n]["tolerance"]:
                raise Reject("goal_unstable", f"joint {n} drifted to {float(q[names.index(n)]):.3f} (target {tgt[n]['position']:.3f})")
        if aname in draft.pose_targets:
            super().refine_targets_from_goal(draft)
