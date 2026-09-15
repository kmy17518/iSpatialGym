"""Tabletop reach bands: which cells of a support's near strip the right hand can grasp an object on, per trunk.

For one (scene support, grasped object) the script computes, in **base coordinates** (x ahead, lateral + = robot's
left) at one concrete stance (edge midpoint, stand‑off 0.45 m + AABB overhang when >= 5 cm, yaw = inward normal), for
every trunk of the grid (translate {0, 0.25, 0.5, 0.75, 0.95} x tilt {0, -0.3, -0.6, -0.9}; -1.2 added when the top is
below 0.6 m):

1. whole‑body collision check of the trunk with untucked arms and open grippers, no spawned objects; colliding trunks
   are skipped;
2. an 8 cm lattice over the near strip of the top at the object's grasp height; cells farther from the right shoulder
   than (arm reach + largest grasp offset + pre‑grasp) are dropped geometrically (no IK);
3. the library‑grasp mask on the remaining cells: object posed at 8 yaws x every library grasp, pre‑grasp 10 cm back
   along the grasp approach, 12‑seed batched IK with the locked joints of the untucked init configuration at that trunk
   (idle arm untucked), the support (and objects placed before, e.g. the source tray) ignored for the grasp pose only.
   (yaw, grasp) pairs are evaluated in rounds - best‑ranked grasps and robot‑facing yaws first - with early exit per
   cell (one feasible pair marks the cell). Objects without library grasps (``--generic-pinch``: blocks, push poses) use
   the generic top‑down pinch (2 wrist yaws x 2 approach pitches);
4. output per trunk: the reachable cells (x, lateral) and, over all trunks, the union extent [lambda_min, lambda_max].

Two‑factor cache (``ISPATIALGYM_REACH_BANDS``, default ``~/.cache/ispatialgym/reach_bands``): the support factor
(collision‑free trunks, lattice, shoulder positions) is keyed by scene + support + top height + stance + trunk grid and
shared by every object; the object factor (the mask) adds the object model, grasp set, rest orientation and the placed
objects. Neither key contains the base pose: a different stand‑off only changes which cells fall on the top.

Modes::

    python -m ispatialgym.sampling.reach_bands --scene S --support T --object CATEGORY/MODEL [--grasp-key K]
    python -m ispatialgym.sampling.reach_bands ... --placed tray/gsxbym            # disembedding: lattice = source tray floor
    python -m ispatialgym.sampling.reach_bands ... --generic-pinch                 # blocks / pushes, no library grasps
    python -m ispatialgym.sampling.reach_bands --scene S --support floor --room R --object C/M [--under PARTNER]
    python -m ispatialgym.sampling.reach_bands --scene S --support T --target toolbox/redmtl --handle-key toolbox__redmtl \
        --joint j_lid --joint-values 0.0,1.48,2.1,2.97 --brace                      # small‑scale articulation

``--rooms`` restricts the loaded rooms (default: the support's rooms); chairs within 1 m of the support are removed
(scene variant) like the sampler does. ``--smoke`` uses a 16 cm lattice and two trunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling.eef_reach_volume import TILTS, TRANSLATES

log = logging.getLogger("ispatialgym.sampling.reach_bands")

LATTICE = 0.08
STAND_OFF = 0.45
PRE_GRASP = 0.10
LOW_TOP_TILT = -1.2  # added to the grid when the top is below 0.6 m (floor, low tables)
LOW_TOP_Z = 0.60
YAWS = [k * math.pi / 4 for k in range(8)]
N_YAWS_GENERIC = 2
TILTS_GENERIC = (0.0, 30.0)
STRIP_MAX_DEPTH = 0.95  # m into the support from the edge; deeper cells are never reachable
STRIP_HALF_WIDTH = 0.80
CACHE_DEFAULT = os.path.expanduser("~/.cache/ispatialgym/reach_bands")


def cache_root() -> str:
    return os.environ.get("ISPATIALGYM_REACH_BANDS", CACHE_DEFAULT)


def _key(d: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def trunk_grid(top_z_above_floor: float) -> List[Tuple[float, float]]:
    tilts = list(TILTS) + ([LOW_TOP_TILT] if top_z_above_floor < LOW_TOP_Z else [])
    return [(tr, ti) for tr in TRANSLATES for ti in tilts]


# ----------------------------------------------------------------------------------------------- result container


@dataclass
class ReachBands:
    """Reachable cells per trunk in base coordinates (x ahead, lateral + = left) plus the stance they were computed at."""

    meta: Dict[str, Any]
    cells: Dict[Tuple[float, float], np.ndarray] = field(default_factory=dict)  # (translate, tilt) -> (N, 2) reachable cells

    @staticmethod
    def load(path_npz: str) -> "ReachBands":
        d = np.load(path_npz, allow_pickle=False)
        with open(path_npz[:-4] + ".json") as f:
            meta = json.load(f)
        cells = {}
        for t in meta["trunks"]:
            if t["collides"]:
                continue
            k = f"t{t['translate']:.2f}_k{t['tilt']:+.1f}"
            cells[(t["translate"], t["tilt"])] = d[k]
        return ReachBands(meta, cells)

    def trunks(self) -> List[Tuple[float, float]]:
        return sorted(self.cells)

    def band(self, translate: float, tilt: float) -> np.ndarray:
        """Cells of the grid trunk with the same translate and the nearest tilt (empty array when that trunk collides)."""
        cands = [k for k in self.cells if abs(k[0] - translate) < 1e-6]
        if not cands:
            return np.zeros((0, 2))
        k = min(cands, key=lambda k: abs(k[1] - tilt))
        return self.cells[k]

    def union_extent(self) -> Tuple[float, float]:
        lat = np.concatenate([c[:, 1] for c in self.cells.values() if len(c)]) if self.cells else np.zeros(0)
        return (float(lat.min()), float(lat.max())) if len(lat) else (0.0, 0.0)

    def to_world(self, cells_base: np.ndarray, base_pose2d: Sequence[float]) -> np.ndarray:
        x0, y0, yaw = base_pose2d
        c, s = math.cos(yaw), math.sin(yaw)
        return np.stack([x0 + c * cells_base[:, 0] - s * cells_base[:, 1], y0 + s * cells_base[:, 0] + c * cells_base[:, 1]], axis=1)


# ----------------------------------------------------------------------------------------------- computation


class BandComputer:
    def __init__(self, world, kit, arms: Sequence[str] = ("right",), lattice: float = LATTICE, seeds: int = 12):
        from ispatialgym.oracle import r1pro_model as M
        from ispatialgym.sampling import geometry as G

        self.M, self.G = M, G
        self.world, self.kit = world, kit
        self.arms = list(arms)
        self.lattice = lattice
        self.seeds = seeds

    # -- stance -------------------------------------------------------------------------------------------------
    def stance_at_edge(self, top, support, stand_off: float = STAND_OFF, edge_name: Optional[str] = None) -> Dict[str, Any]:
        """Base pose at the midpoint of one edge of the top (preferred: named edge; else the longest edge whose stance is on
        free floor and whole‑body collision free at trunk (0.5, 0))."""
        G, kit = self.G, self.kit
        lo, hi = support.aabb
        aabb_xy = G.aabb_corners(lo, hi)[:, :2]
        cands = []
        for name, mid, normal, length in top.edges():
            if edge_name is not None and name != edge_name:
                continue
            inward = -normal
            overhang = max(0.0, float((aabb_xy @ normal).max()) - float(th.dot(mid, normal)))
            if overhang < 0.05:
                overhang = 0.0
            base_xy = mid + normal * (stand_off + overhang)
            yaw = math.atan2(float(inward[1]), float(inward[0]))
            if not kit.trav_free(base_xy):
                continue
            q = kit.q_at_base_pose(kit.q_config(arms="untucked", translate=0.5), float(base_xy[0]), float(base_xy[1]), yaw)
            free = bool(kit.collision_free(q, margin=0.01)[0])
            cands.append((free, length, name, mid, inward, base_xy, yaw, overhang))
        if not cands:
            raise RuntimeError(f"no stance on free floor around {top.name}")
        cands.sort(key=lambda c: (not c[0], -c[1]))
        free, length, name, mid, inward, base_xy, yaw, overhang = cands[0]
        if not free:
            log.warning("every edge stance of %s collides at trunk (0.5, 0); using %s anyway", top.name, name)
        tangent = th.tensor([-float(inward[1]), float(inward[0])])
        return {"edge": name, "edge_mid": mid.tolist(), "inward": inward.tolist(), "tangent": tangent.tolist(), "length_m": float(length),
                "base_pose2d": [float(base_xy[0]), float(base_xy[1]), yaw], "stand_off": stand_off, "overhang": overhang,
                "edge_depth": float(th.dot(mid - base_xy, inward))}

    def stance_free_floor(self, room: Optional[str], facing: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        """Floor mode: the room's floor centroid with the base facing +x (or ``facing``), nudged to a free cell."""
        kit, world = self.kit, self.world
        floor = world.floor_of_room(room) if room else None
        lo, hi = (floor.aabb if floor is not None else (th.tensor([-1.0, -1.0, 0.0]), th.tensor([1.0, 1.0, 0.0])))
        centre = th.tensor([float(lo[0] + hi[0]) / 2, float(lo[1] + hi[1]) / 2])
        yaw = math.atan2(float(facing[1]), float(facing[0])) if facing is not None else 0.0
        best = None
        for r in th.arange(0.0, 2.01, 0.25).tolist():
            for a in th.linspace(0.0, 2 * math.pi, 17)[:-1].tolist():
                xy = centre + th.tensor([r * math.cos(a), r * math.sin(a)])
                if not kit.trav_free(xy, eroded=True):
                    continue
                q = kit.q_at_base_pose(kit.q_config(arms="untucked", translate=0.5), float(xy[0]), float(xy[1]), yaw)
                if bool(kit.collision_free(q, margin=0.01)[0]):
                    best = xy
                    break
            if best is not None:
                break
        if best is None:
            raise RuntimeError(f"no collision-free floor stance in room {room}")
        inward = th.tensor([math.cos(yaw), math.sin(yaw)])
        return {"edge": None, "edge_mid": best.tolist(), "inward": inward.tolist(), "tangent": [-float(inward[1]), float(inward[0])],
                "base_pose2d": [float(best[0]), float(best[1]), yaw], "stand_off": 0.0, "overhang": 0.0, "edge_depth": 0.0}

    def stance_under(self, partner, stand_off: float = STAND_OFF) -> Dict[str, Any]:
        """IORS under: base square to the partner's near AABB edge (the edge with free floor in front), at ``stand_off``."""
        kit = self.kit
        lo, hi = partner.aabb
        cx, cy = float(lo[0] + hi[0]) / 2, float(lo[1] + hi[1]) / 2
        cands = []
        for normal, mid, length in ((th.tensor([1.0, 0.0]), th.tensor([float(hi[0]), cy]), float(hi[1] - lo[1])),
                                    (th.tensor([-1.0, 0.0]), th.tensor([float(lo[0]), cy]), float(hi[1] - lo[1])),
                                    (th.tensor([0.0, 1.0]), th.tensor([cx, float(hi[1])]), float(hi[0] - lo[0])),
                                    (th.tensor([0.0, -1.0]), th.tensor([cx, float(lo[1])]), float(hi[0] - lo[0]))):
            base_xy = mid + normal * stand_off
            if not kit.trav_free(base_xy):
                continue
            inward = -normal
            yaw = math.atan2(float(inward[1]), float(inward[0]))
            q = kit.q_at_base_pose(kit.q_config(arms="untucked", translate=0.5), float(base_xy[0]), float(base_xy[1]), yaw)
            cands.append((bool(kit.collision_free(q, margin=0.01)[0]), length, mid, inward, base_xy, yaw))
        if not cands:
            raise RuntimeError(f"no stance around {partner.name}")
        cands.sort(key=lambda c: (not c[0], -c[1]))
        _, length, mid, inward, base_xy, yaw = cands[0]
        return {"edge": "aabb", "edge_mid": mid.tolist(), "inward": inward.tolist(), "tangent": [-float(inward[1]), float(inward[0])], "length_m": length,
                "base_pose2d": [float(base_xy[0]), float(base_xy[1]), yaw], "stand_off": stand_off, "overhang": 0.0,
                "edge_depth": float(th.dot(mid - base_xy, inward))}

    # -- lattice ------------------------------------------------------------------------------------------------
    def base_frame(self, stance: Dict[str, Any]):
        x0, y0, yaw = stance["base_pose2d"]
        fwd = th.tensor([math.cos(yaw), math.sin(yaw)])
        left = th.tensor([-math.sin(yaw), math.cos(yaw)])
        return th.tensor([x0, y0]), fwd, left

    def lattice_on_cells(self, stance: Dict[str, Any], world_cells: th.Tensor, x_range: Tuple[float, float] = (0.0, STRIP_MAX_DEPTH + STAND_OFF + 0.2),
                         lat_half: float = STRIP_HALF_WIDTH) -> Tuple[th.Tensor, th.Tensor]:
        """Snap world cells (support top / tray floor / floor raster) to the base‑frame lattice: returns (base_xy (N,2), world_xy (N,2))."""
        origin, fwd, left = self.base_frame(stance)
        rel = world_cells - origin
        x = rel @ fwd
        lat = rel @ left
        keep = (x >= x_range[0]) & (x <= x_range[1]) & (lat.abs() <= lat_half)
        x, lat = x[keep], lat[keep]
        ix = th.round(x / self.lattice).long()
        il = th.round(lat / self.lattice).long()
        seen = {}
        for i in range(len(x)):
            k = (int(ix[i]), int(il[i]))
            if k not in seen:
                seen[k] = (float(ix[i]) * self.lattice, float(il[i]) * self.lattice)
        base = th.tensor(list(seen.values())) if seen else th.zeros(0, 2)
        world = origin + base[:, :1] * fwd + base[:, 1:] * left if len(base) else th.zeros(0, 2)
        return base, world

    # -- object poses / grasp sets ------------------------------------------------------------------------------
    def object_pose_at(self, world_xy, yaw: float, z_support: float, hull, rest_quat) -> Tuple[th.Tensor, th.Tensor]:
        G = self.G
        quat = G.qmul(G.quat_from_yaw(yaw), rest_quat)
        pos = th.tensor([float(world_xy[0]), float(world_xy[1]), z_support + self.world.bottom_offset(hull, quat) + 0.003])
        return pos, quat

    def grasp_eef_pair(self, obj_pose, grasp) -> List[Tuple[th.Tensor, th.Tensor]]:
        G = self.G
        eef = G.compose(obj_pose, grasp.pose)
        axis = G.rotate(eef[1], th.tensor([0.0, 0.0, 1.0]))
        return [(eef[0] - axis * PRE_GRASP, eef[1]), eef]

    def _shoulder_offset(self, arm: str) -> th.Tensor:
        """Shoulder (``<arm>_arm_link1`` origin) in the ``torso_link4`` frame - constant, measured once on the simulated
        robot (cuRobo's FK only knows the tool frames, the head camera, base_link and torso_link4)."""
        cache = getattr(self, "_sh_off", {})
        if arm not in cache:
            G = self.G
            t4 = self.world.robot.links["torso_link4"].get_position_orientation()
            sh = self.world.robot.links[f"{arm}_arm_link1"].get_position_orientation()
            cache[arm] = G.relative((sh[0].float(), sh[1].float()), (t4[0].float(), t4[1].float()))[0]
            self._sh_off = cache
        return cache[arm]

    def shoulder_and_reach(self, q: th.Tensor, arm: str) -> Tuple[th.Tensor, float]:
        """Shoulder position (world) at ``q`` and the arm's stretched length (shoulder -> eef with every arm joint at zero,
        the arm hanging straight)."""
        M, G = self.M, self.G
        t4 = self.kit.fk(["torso_link4"], q)["torso_link4"]
        sh = G.compose(t4, (self._shoulder_offset(arm), G.IDENTITY_QUAT))[0]
        if not hasattr(self, "_arm_len"):
            self._arm_len = {}
        if arm not in self._arm_len:
            qz = G.t(q).clone()
            qz[self.world.robot.arm_control_idx[arm]] = 0.0
            ee = self.kit.fk([M.EEF_LINK[arm]], qz)[M.EEF_LINK[arm]][0]
            self._arm_len[arm] = float(th.linalg.norm(ee - sh))
        return sh, self._arm_len[arm]

    # -- main loop ----------------------------------------------------------------------------------------------
    def compute(self, stance: Dict[str, Any], lattice_world: th.Tensor, z_support: float, pose_fn, grasp_sets, ignore_for_grasp, arm: str = "right",
                trunks: Optional[Sequence[Tuple[float, float]]] = None, max_grasp_offset: float = 0.0) -> Dict[str, Any]:
        """``pose_fn(world_xy, yaw) -> obj_pose`` and ``grasp_sets`` = list of grasps (with ``.pose``; rank order = library
        order) or ``None`` for the generic pinch. Returns the per‑trunk results."""
        kit, G = self.kit, self.G
        base_cells, world_cells = self.lattice_on_cells(stance, lattice_world)
        trunks = list(trunks) if trunks is not None else trunk_grid(z_support - self.world.floor_z)
        x0, y0, yaw0 = stance["base_pose2d"]
        _, fwd, left = self.base_frame(stance)
        results = []
        # object yaw offsets relative to the base yaw, robot-facing first (canonical +x of the object toward the robot = offset pi)
        yaw_order = sorted(YAWS, key=lambda y: abs(G.wrap_angle(y - math.pi)))
        for tr, ti in trunks:
            t0 = time.time()
            q = kit.q_at_base_pose(kit.q_config(arms="untucked", translate=tr, tilt=ti), x0, y0, yaw0)
            entry = {"translate": tr, "tilt": ti, "collides": False, "n_lattice": int(len(base_cells)), "n_after_prune": 0, "n_reachable": 0, "rounds": 0, "time_s": 0.0}
            if not bool(kit.collision_free(q, margin=0.01)[0]):
                entry["collides"] = True
                entry["time_s"] = round(time.time() - t0, 2)
                results.append((entry, np.zeros((0, 2))))
                log.info("trunk (%.2f, %+.1f): collides, skipped", tr, ti)
                continue
            shoulder, arm_len = self.shoulder_and_reach(q, arm)
            reach = arm_len + max_grasp_offset + PRE_GRASP + 0.05
            pts = th.cat([world_cells, th.full((len(world_cells), 1), z_support)], dim=1)
            near = th.linalg.norm(pts - shoulder, dim=1) <= reach
            todo = [i for i in range(len(base_cells)) if bool(near[i])]
            entry["n_after_prune"] = len(todo)
            ok = th.zeros(len(base_cells), dtype=th.bool)
            # rounds: (grasp rank, yaw) pairs in chunks; every round is one batched pre-grasp + grasp IK call over the open cells
            if grasp_sets is None:
                pair_specs = [("generic", None, None)]
            else:
                pair_specs = [(g.name, g, y) for g in grasp_sets for y in yaw_order]
            chunk = 4 if grasp_sets is not None else 1
            for r0 in range(0, len(pair_specs), chunk):
                if not todo:
                    break
                specs = pair_specs[r0:r0 + chunk]
                pairs, index = [], []
                for i in todo:
                    if grasp_sets is None:
                        centre = th.tensor([float(world_cells[i][0]), float(world_cells[i][1]), z_support])
                        for pair in kit.grasp_candidates_at(centre, [yaw0, yaw0 + math.pi / 2][:N_YAWS_GENERIC], TILTS_GENERIC, fwd):
                            pairs.append(pair)
                            index.append(i)
                    else:
                        for _, g, y in specs:
                            pairs.append(self.grasp_eef_pair(pose_fn(world_cells[i], y + yaw0), g))
                            index.append(i)
                pres = [p[0] for p in pairs]
                ok_pre = kit.ik_batch(pres, q, arm=arm)
                sel = [k for k, s in enumerate(ok_pre.tolist()) if s]
                if sel:
                    ok_g = kit.ik_batch([pairs[k][1] for k in sel], q, arm=arm, ignore=ignore_for_grasp)
                    for k, s in zip(sel, ok_g.tolist()):
                        if s:
                            ok[index[k]] = True
                entry["rounds"] += 1
                todo = [i for i in todo if not bool(ok[i])]
            cells = base_cells[ok].numpy() if int(ok.sum()) else np.zeros((0, 2))
            entry["n_reachable"] = int(ok.sum())
            entry["time_s"] = round(time.time() - t0, 2)
            log.info("trunk (%.2f, %+.1f): %d lattice, %d after prune, %d reachable, %d rounds, %.1fs", tr, ti, len(base_cells), entry["n_after_prune"],
                     entry["n_reachable"], entry["rounds"], entry["time_s"])
            results.append((entry, cells))
        return {"trunks": [e for e, _ in results], "cells": {(e["translate"], e["tilt"]): c for e, c in results}}


# ----------------------------------------------------------------------------------------------- articulation (brace mode)


def handle_pose_at_q(joint_info: Dict[str, Any], handle: Dict[str, Any], q: float, obj_pose, G):
    """Handle contact / pinch / approach moved with the moving link to joint value ``q`` (closed configuration at
    ``lower``): revolute = rotation about (hinge_point_obj, axis_dir_obj), prismatic = translation along axis_dir_obj.
    Returns [pre-grasp, grasp] eef poses in the world."""
    import omnigibson.utils.transform_utils as T

    lower = float(joint_info.get("closed_q", joint_info.get("lower", 0.0)))
    contact = th.tensor(handle["contact"], dtype=th.float32)
    approach = th.tensor(handle["approach_dir"], dtype=th.float32)
    pinch = th.tensor(handle["pinch_axis"], dtype=th.float32)
    axis = th.tensor(joint_info.get("axis_dir_obj", [0.0, 1.0, 0.0]), dtype=th.float32)
    axis = axis / th.linalg.norm(axis)
    if joint_info.get("type", "RevoluteJoint") == "RevoluteJoint":
        rot = G.quat_about_axis(axis, q - lower)
        hinge = th.tensor(joint_info.get("hinge_point_obj", [0.0, 0.0, 0.0]), dtype=th.float32)
        contact = hinge + G.rotate(rot, contact - hinge)
        approach = G.rotate(rot, approach)
        pinch = G.rotate(rot, pinch)
    else:
        contact = contact + axis * (q - lower)
    # object frame -> world
    c_w = G.compose(obj_pose, (contact, G.IDENTITY_QUAT))[0]
    a_w = G.rotate(obj_pose[1], approach)
    p_w = G.rotate(obj_pose[1], pinch)
    z = a_w / th.linalg.norm(a_w)
    y = p_w - th.dot(p_w, z) * z
    y = y / th.linalg.norm(y)
    x = th.cross(y, z, dim=0)
    quat = T.mat2quat(th.stack([x, y, z], dim=1))
    eef = c_w - z * 0.012
    return [(eef - z * PRE_GRASP, quat), (eef, quat)]


def brace_candidates(joint_info: Dict[str, Any], handle_data: Dict[str, Any], obj_pose, q_values: Sequence[float], brace_side: float, G):
    """Brace contacts on the base link (never the moving link): ``opposite_face`` (base-link face whose outward normal is
    closest to the push direction v, contact at the face centroid shifted 1/4 of the face width toward the brace arm's
    side, approach -v) and ``top_press`` (downward press on the exposed top of the base link nearest the brace arm's
    side). Contacts must stay >= 2 cm outside the moving link's swept AABB over the joint values. Returns
    [(kind, [pre, contact])] in preference order, eef poses in the world."""
    import omnigibson.utils.transform_utils as T

    bl = handle_data.get("base_link", {})
    bmin, bmax = th.tensor(bl.get("bbox_min", [-0.1, -0.1, 0.0])), th.tensor(bl.get("bbox_max", [0.1, 0.1, 0.1]))
    # push direction: horizontal handle motion toward the first waypoint; if < 30 % of the motion is horizontal (lid lifted
    # straight up) use the direction from the hinge line to the handle contact
    h0 = handle_pose_at_q(joint_info, handle_data["handle"], q_values[0], (th.zeros(3), G.IDENTITY_QUAT), G)[1][0]
    h1 = handle_pose_at_q(joint_info, handle_data["handle"], q_values[min(1, len(q_values) - 1)], (th.zeros(3), G.IDENTITY_QUAT), G)[1][0]
    d = h1 - h0
    if float(th.linalg.norm(d)) < 1e-6 or float(th.linalg.norm(d[:2])) < 0.3 * float(th.linalg.norm(d)):
        hinge = th.tensor(joint_info.get("hinge_point_obj", [0.0, 0.0, 0.0]), dtype=th.float32)
        d = th.tensor(handle_data["handle"]["contact"], dtype=th.float32) - hinge
    v = th.tensor([float(d[0]), float(d[1]), 0.0])
    v = v / max(float(th.linalg.norm(v)), 1e-6)
    # moving link swept AABB (object frame) over the joint values, from the annotated link bbox at the closed configuration
    lmin, lmax = th.tensor(joint_info.get("link_bbox_min", bmin.tolist())), th.tensor(joint_info.get("link_bbox_max", bmax.tolist()))
    corners = th.stack([th.tensor([x, y, z]) for x in (lmin[0], lmax[0]) for y in (lmin[1], lmax[1]) for z in (lmin[2], lmax[2])])
    swept = [corners]
    axis = th.tensor(joint_info.get("axis_dir_obj", [0.0, 1.0, 0.0]), dtype=th.float32)
    axis = axis / th.linalg.norm(axis)
    lower = float(joint_info.get("closed_q", joint_info.get("lower", 0.0)))
    for qv in q_values:
        if joint_info.get("type", "RevoluteJoint") == "RevoluteJoint":
            rot = G.quat_about_axis(axis, qv - lower)
            hinge = th.tensor(joint_info.get("hinge_point_obj", [0.0, 0.0, 0.0]), dtype=th.float32)
            swept.append(hinge + G.rotate(rot, corners - hinge))
        else:
            swept.append(corners + axis * (qv - lower))
    swept = th.cat(swept)
    smin, smax = swept.min(dim=0).values - 0.02, swept.max(dim=0).values + 0.02
    out = []
    faces = [(th.tensor([1.0, 0, 0]), 0, bmax[0]), (th.tensor([-1.0, 0, 0]), 0, bmin[0]), (th.tensor([0, 1.0, 0]), 1, bmax[1]), (th.tensor([0, -1.0, 0]), 1, bmin[1])]
    # opposite_face: outward normal closest to v (the face the handle motion points toward)
    n, ax, coord = max(faces, key=lambda f: float(th.dot(f[0], v)))
    other = 1 - ax
    centre = (bmin + bmax) / 2
    p = centre.clone()
    p[ax] = coord
    side_dir = th.tensor([-float(v[1]), float(v[0]), 0.0]) * brace_side  # brace arm's side along the face
    p = p + side_dir * (float(bmax[other] - bmin[other]) / 4)
    inside_swept = bool((p >= smin).all() and (p <= smax).all())
    if not inside_swept:
        out.append(("opposite_face", p, -n))
    # top_press: exposed top of the base link nearest the brace arm's side, outside the swept AABB
    top = th.tensor([float(centre[0]), float(centre[1]), float(bmax[2])]) + side_dir * (float(bmax[other] - bmin[other]) / 3)
    if not bool((top >= smin).all() and (top <= smax).all()):
        out.append(("top_press", top, th.tensor([0.0, 0.0, -1.0])))
    poses = []
    for kind, p_obj, a_obj in out:
        c_w = G.compose(obj_pose, (p_obj, G.IDENTITY_QUAT))[0]
        a_w = G.rotate(obj_pose[1], a_obj)
        z = a_w / th.linalg.norm(a_w)
        y_ref = G.rotate(obj_pose[1], th.tensor([0.0, 1.0, 0.0]))  # closing axis = object +y
        y = y_ref - th.dot(y_ref, z) * z
        if float(th.linalg.norm(y)) < 1e-3:
            y_ref = G.rotate(obj_pose[1], th.tensor([1.0, 0.0, 0.0]))
            y = y_ref - th.dot(y_ref, z) * z
        y = y / th.linalg.norm(y)
        x = th.cross(y, z, dim=0)
        quat = T.mat2quat(th.stack([x, y, z], dim=1))
        eef = c_w - z * 0.012
        poses.append((kind, [(eef - z * PRE_GRASP, quat), (eef, quat)]))
    return poses


# ----------------------------------------------------------------------------------------------- driver


def _chairs_variant(scene: str, support_name: str, rooms: List[str], out_dir: str):
    from ispatialgym.sampling import scene_variants as SV
    from ispatialgym.sampling.sim_utils import stable_scene_path

    with open(stable_scene_path(scene)) as f:
        stable = json.load(f)
    removed = SV.chairs_near_tables(stable, rooms, [support_name], distance=1.0, categories=SV.CHAIR_CATEGORIES)
    if not removed:
        return None, []
    vpath, _ = SV.write_variant(stable_scene_path(scene), removed, out_dir)
    return {"path": vpath, "removed_objects": removed, "freed_footprints": SV.freed_footprints(stable, removed)}, removed


def _rooms_of(scene: str, name: str) -> List[str]:
    from ispatialgym.sampling.sim_utils import stable_scene_path

    with open(stable_scene_path(scene)) as f:
        stable = json.load(f)
    info = stable["objects_info"]["init_info"].get(name)
    if info is None:
        raise SystemExit(f"{name} not in scene {scene}")
    rooms = info["args"].get("in_rooms") or []
    return list(rooms)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--support", required=True, help="scene object name, or 'floor' (with --room)")
    ap.add_argument("--room", default=None, help="floor mode: room instance")
    ap.add_argument("--rooms", nargs="*", default=None, help="rooms to load (default: the support's rooms)")
    ap.add_argument("--object", default=None, help="CATEGORY/MODEL of the grasped object")
    ap.add_argument("--grasp-key", default=None)
    ap.add_argument("--generic-pinch", action="store_true", help="no library grasps: generic top-down pinch (blocks, pushes)")
    ap.add_argument("--placed", nargs="*", default=[], help="objects placed before the target, CATEGORY/MODEL (disembedding source tray: lattice = its floor)")
    ap.add_argument("--under", default=None, help="floor mode: partner scene object; lattice = floor under it")
    ap.add_argument("--target", default=None, help="brace mode: articulated CATEGORY/MODEL")
    ap.add_argument("--handle-key", default=None)
    ap.add_argument("--joint", default=None)
    ap.add_argument("--joint-values", default=None, help="comma-separated joint values whose handle poses are screened")
    ap.add_argument("--brace", action="store_true")
    ap.add_argument("--edge", default=None, help="edge name (+u/-u/+v/-v); default: longest collision-free edge")
    ap.add_argument("--stand-off", type=float, default=STAND_OFF)
    ap.add_argument("--lattice", type=float, default=LATTICE)
    ap.add_argument("--out", default=None, help="directory for the .npz/.json (default: the cache dir)")
    ap.add_argument("--smoke", action="store_true", help="16 cm lattice, two trunks")
    ap.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(name)s] %(message)s", force=True)

    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling import grasp_library as GL
    from ispatialgym.sampling.robot_utils import RobotKit
    from ispatialgym.sampling.sim_utils import SimWorld

    import omnigibson as og

    lattice = 0.16 if args.smoke else args.lattice
    floor_mode = args.support == "floor"
    anchor_name = args.under if floor_mode and args.under else (None if floor_mode else args.support)
    rooms = args.rooms or ([args.room] if floor_mode and args.room else _rooms_of(args.scene, anchor_name))
    out_dir = args.out or cache_root()
    os.makedirs(out_dir, exist_ok=True)
    variant, removed = (None, [])
    if not floor_mode:
        variant, removed = _chairs_variant(args.scene, args.support, rooms, os.path.join(out_dir, "scene_variants"))
    t_load = time.time()
    world = SimWorld(args.scene, rooms, scene_variant=variant)
    arms = ("right", "left") if args.brace else ("right",)
    kit = RobotKit(world, prebuild=tuple(f"{a}_arm" for a in arms))
    kit.sync_world()
    log.info("scene %s rooms %s loaded in %.0fs (chairs removed: %s)", args.scene, rooms, time.time() - t_load, removed)
    bc = BandComputer(world, kit, arms=arms, lattice=lattice)
    meta: Dict[str, Any] = {"scene": args.scene, "rooms": rooms, "support": args.support, "removed_chairs": removed, "lattice": lattice,
                            "stand_off": args.stand_off, "arms": "untucked", "grid": {"translates": TRANSLATES, "tilts": TILTS, "low_top_tilt": LOW_TOP_TILT},
                            "seeds": 12, "pre_grasp": PRE_GRASP, "mode": "brace" if args.brace else ("floor" if floor_mode else "tabletop")}
    try:
        # ---- support geometry and stance
        placed_objs = []
        if floor_mode:
            z_support = world.floor_z
            if args.under:
                partner = world.obj(args.under)
                stance = bc.stance_under(partner, args.stand_off)
                lo, hi = partner.aabb
                res = 0.02
                xs = th.arange(float(lo[0]) + res / 2, float(hi[0]), res)
                ys = th.arange(float(lo[1]) + res / 2, float(hi[1]), res)
                gx, gy = th.meshgrid(xs, ys, indexing="ij")
                pts = th.stack([gx.flatten(), gy.flatten()], dim=1)
                starts = th.cat([pts, th.full((len(pts), 1), world.floor_z + 0.005)], dim=1)
                ends = starts.clone()
                ends[:, 2] = float(hi[2]) + 0.5
                hits = world.raycast(starts, ends)
                prims = {l.prim_path for l in partner.links.values()}
                clear = th.tensor([bool(h["hit"] and h["rigidBody"] in prims) for h in hits])
                clearance = th.tensor([float(h["position"][2]) - world.floor_z if h["hit"] else 0.0 for h in hits])
                lattice_world = pts[clear]
                meta["under_clearance_m"] = {"min": float(clearance[clear].min()) if int(clear.sum()) else None, "max": float(clearance[clear].max()) if int(clear.sum()) else None}
                meta["under_partner"] = args.under
                ignore_grasp = [world.obj(world.floor_of_room(rooms[0]).name)] if rooms else []
            else:
                stance = bc.stance_free_floor(args.room)
                origin, fwd, left = bc.base_frame(stance)
                xs = th.arange(0.2, STRIP_MAX_DEPTH + 0.5, 0.02)
                ls = th.arange(-STRIP_HALF_WIDTH, STRIP_HALF_WIDTH + 1e-6, 0.02)
                gx, gl = th.meshgrid(xs, ls, indexing="ij")
                lattice_world = origin + gx.flatten().unsqueeze(1) * fwd + gl.flatten().unsqueeze(1) * left
                lattice_world = th.stack([c for c in lattice_world if kit.trav_free(c)]) if len(lattice_world) else lattice_world
                ignore_grasp = [world.floor_of_room(args.room)] if args.room else []
            top_json = {"z": z_support, "kind": "floor"}
        else:
            support = world.obj(args.support)
            top = world.support_top(support)
            z_support = top.z
            stance = bc.stance_at_edge(top, support, args.stand_off, args.edge)
            lattice_world = top.cells(free_only=True)
            ignore_grasp = [support]
            top_json = top.to_json()
            # objects placed before the target (disembedding source tray): centred on the stance line, near edge 8 cm behind
            # the front edge, long axis along the edge; the lattice becomes the placed object's inner floor
            for spec in args.placed:
                cat, model = spec.split("/", 1)
                obj = world.spawn(f"placed_{cat}_{model}", cat, model)
                hull = world.local_hull(obj)
                ext = hull.max(dim=0).values - hull.min(dim=0).values
                tangent = th.tensor(stance["tangent"])
                inward = th.tensor(stance["inward"])
                tangent_yaw = math.atan2(float(tangent[1]), float(tangent[0]))
                yaw = tangent_yaw if float(ext[0]) >= float(ext[1]) else tangent_yaw - math.pi / 2
                quat = G.quat_from_yaw(yaw)
                fp0 = world.footprint(hull, (th.zeros(3), quat))
                half_depth = float((fp0 @ inward).abs().max())
                mid = th.tensor(stance["edge_mid"])
                xy = mid + inward * (0.08 + half_depth)
                pos = th.tensor([float(xy[0]), float(xy[1]), top.z + world.bottom_offset(hull, quat) + 0.003])
                obj.set_position_orientation(pos, quat)
                world.activate(obj)
                world.step_physics(30, still=[world.robot])
                placed_objs.append(obj)
                # inner floor raster of the placed object (lowest inner level hit by downward rays)
                lo, hi = obj.aabb
                res = 0.01
                xs = th.arange(float(lo[0]) + res / 2, float(hi[0]), res)
                ys = th.arange(float(lo[1]) + res / 2, float(hi[1]), res)
                gx, gy = th.meshgrid(xs, ys, indexing="ij")
                starts = th.stack([gx.flatten(), gy.flatten(), th.full_like(gx.flatten(), float(hi[2]) + 0.10)], dim=1)
                ends = starts.clone()
                ends[:, 2] = float(lo[2]) - 0.02
                hits = world.raycast(starts, ends)
                prims = {l.prim_path for l in obj.links.values()}
                zs = th.tensor([float(h["position"][2]) if (h["hit"] and h["rigidBody"] in prims) else float("nan") for h in hits])
                valid = zs[~th.isnan(zs)]
                floor_z = float(th.quantile(valid, 0.10))
                keep = (~th.isnan(zs)) & ((zs - floor_z).abs() < 0.004)
                lattice_world = starts[keep][:, :2]
                z_support = floor_z
                ignore_grasp.append(obj)
                meta.setdefault("placed", []).append({"category": cat, "model": model, "pose": [pos.tolist(), quat.tolist()], "floor_z": floor_z, "n_floor_cells": int(keep.sum())})
            kit.sync_world()
        meta.update({"stance": stance, "top": top_json, "z_support": z_support})

        # ---- object / grasp set
        if args.brace:
            assert args.target and args.handle_key and args.joint and args.joint_values, "--brace needs --target --handle-key --joint --joint-values"
            cat, model = args.target.split("/", 1)
            from ispatialgym.sampling.grasp_library import library_root

            with open(os.path.join(library_root(), "artic", "handles", f"{args.handle_key}.json")) as f:
                hdata = json.load(f)
            jinfo = next(j for j in hdata["joints"] if j["joint"] == args.joint)
            handles = [h for h in jinfo.get("handles", []) if h.get("kind") in ("bar", "knob")] or ([jinfo["fallback"]] if jinfo.get("fallback") else [])
            if not handles:
                raise SystemExit(f"no graspable handle for {args.handle_key}/{args.joint}")
            hdata["handle"] = handles[0]
            q_values = [float(v) for v in args.joint_values.split(",")]
            obj = world.spawn(f"target_{cat}_{model}", cat, model)
            hull = world.local_hull(obj)
            world.park(obj, len(world.spawned) - 1)
            # facing direction: moving face outward normal (lids: from the hinge line to the centre) points at the robot
            outward = th.tensor(jinfo.get("outward_normal", [1.0, 0.0, 0.0]), dtype=th.float32)
            if abs(float(outward[2])) > 0.9:
                axis = th.tensor(jinfo.get("axis_dir_obj", [0.0, 1.0, 0.0]), dtype=th.float32)
                hinge = th.tensor(jinfo.get("hinge_point_obj", [0.0, 0.0, 0.0]), dtype=th.float32)
                perp = th.cross(axis, th.tensor([0.0, 0.0, 1.0]), dim=0)
                centre = hull.mean(dim=0)
                if float(th.dot(perp[:2], (centre - hinge)[:2])) < 0:
                    perp = -perp
                outward = perp
            outward[2] = 0.0
            outward = outward / th.linalg.norm(outward)
            inward = th.tensor(stance["inward"])
            face_yaw = math.atan2(float(-inward[1]), float(-inward[0])) - math.atan2(float(outward[1]), float(outward[0]))
            # handle arm = the arm on the handle's side of the target centre (> 5 cm), else right; brace = the other
            hc = th.tensor(hdata["handle"]["contact"], dtype=th.float32)
            side_axis = th.tensor([-float(outward[1]), float(outward[0]), 0.0])  # robot's left when the face points at the robot
            lat = float(th.dot(hc - hull.mean(dim=0), side_axis))
            handle_arm = "left" if lat > 0.05 else "right"
            brace_arm = "right" if handle_arm == "left" else "left"
            brace_side = 1.0 if brace_arm == "left" else -1.0
            x0, y0, yaw0 = stance["base_pose2d"]
            face_off = G.wrap_angle(face_yaw - yaw0)  # object yaw offset (relative to the base yaw) with the moving face toward the robot
            meta.update({"target": args.target, "handle_key": args.handle_key, "joint": args.joint, "joint_values": q_values, "face_yaw_offset": face_off,
                         "handle_arm": handle_arm, "brace_arm": brace_arm, "handle": hdata["handle"]})
            base_cells, world_cells = bc.lattice_on_cells(stance, lattice_world)
            trunks = trunk_grid(z_support - world.floor_z)
            per_trunk = []
            cells_out: Dict[Tuple[float, float], np.ndarray] = {}
            for tr, ti in (trunks[:2] if args.smoke else trunks):
                t0 = time.time()
                q = kit.q_at_base_pose(kit.q_config(arms="untucked", translate=tr, tilt=ti), x0, y0, yaw0)
                entry = {"translate": tr, "tilt": ti, "collides": False, "n_lattice": int(len(base_cells)), "n_reachable": 0, "time_s": 0.0, "feasible": []}
                if not bool(kit.collision_free(q, margin=0.01)[0]):
                    entry["collides"] = True
                    per_trunk.append(entry)
                    cells_out[(tr, ti)] = np.zeros((0, 2))
                    continue
                # 3 yaws about the facing yaw (offsets relative to the base yaw); a cell is in the band when ALL joint values'
                # handle poses are feasible for some yaw and >= 1 brace kind is feasible for that yaw. Batched per trunk:
                # every handle pre-grasp of every (cell, yaw, q) in one IK call, grasps of the survivors in a second
                # (support ignored), brace [pre, contact] pairs of the survivors in a third with the brace arm.
                shoulder, arm_len = bc.shoulder_and_reach(q, handle_arm)
                pts = th.cat([world_cells, th.full((len(world_cells), 1), z_support)], dim=1)
                near = th.linalg.norm(pts - shoulder, dim=1) <= arm_len + float(th.linalg.norm(th.tensor(hdata["handle"]["contact"]))) + PRE_GRASP + 0.05
                combos = [(i, face_off + dy) for i in range(len(base_cells)) if bool(near[i]) for dy in (0.0, -0.1, 0.1)]
                obj_poses = {c: bc.object_pose_at(world_cells[c[0]], yaw0 + c[1], z_support, hull, G.IDENTITY_QUAT) for c in combos}
                handle_pairs = {c: [handle_pose_at_q(jinfo, hdata["handle"], qv, obj_poses[c], G) for qv in q_values] for c in combos}
                flat = [(c, k) for c in combos for k in range(len(q_values))]
                ok_pre = kit.ik_batch([handle_pairs[c][k][0] for c, k in flat], q, arm=handle_arm) if flat else th.zeros(0, dtype=th.bool)
                surv = [c for c in combos if all(bool(ok_pre[n]) for n, (cc, _) in enumerate(flat) if cc == c)]
                flat = [(c, k) for c in surv for k in range(len(q_values))]
                ok_g = kit.ik_batch([handle_pairs[c][k][1] for c, k in flat], q, arm=handle_arm, ignore=ignore_grasp) if flat else th.zeros(0, dtype=th.bool)
                surv = [c for c in surv if all(bool(ok_g[n]) for n, (cc, _) in enumerate(flat) if cc == c)]
                braces = {c: brace_candidates(jinfo, hdata, obj_poses[c], q_values, brace_side, G) for c in surv}
                bflat = [(c, kind) for c in surv for kind, _ in braces[c]]
                bposes = [pose for c in surv for _, pair in braces[c] for pose in pair]
                ok_b = kit.ik_batch(bposes, q, arm=brace_arm) if bposes else th.zeros(0, dtype=th.bool)
                feasible = {}
                for n, (c, kind) in enumerate(bflat):
                    if bool(ok_b[2 * n]) and bool(ok_b[2 * n + 1]):
                        feasible.setdefault(c, []).append(kind)
                feasible_cells = []
                for i in range(len(base_cells)):
                    for c in ((i, face_off), (i, face_off - 0.1), (i, face_off + 0.1)):
                        if c in feasible:
                            feasible_cells.append((i, c[1], feasible[c]))
                            break
                entry["n_after_prune"] = int(near.sum())
                entry["n_reachable"] = len(feasible_cells)
                entry["feasible"] = [{"cell": base_cells[i].tolist(), "yaw_offset": yaw, "brace_kinds": kinds} for i, yaw, kinds in feasible_cells]
                entry["time_s"] = round(time.time() - t0, 2)
                per_trunk.append(entry)
                cells_out[(tr, ti)] = base_cells[[i for i, _, _ in feasible_cells]].numpy() if feasible_cells else np.zeros((0, 2))
                log.info("brace trunk (%.2f, %+.1f): %d/%d cells feasible, %.1fs", tr, ti, len(feasible_cells), len(base_cells), entry["time_s"])
            result = {"trunks": per_trunk, "cells": cells_out}
            key = _key({"scene": args.scene, "support": args.support, "z": round(z_support, 2), "stance": stance["edge"], "s": args.stand_off, "target": args.target,
                        "handle_key": args.handle_key, "joint": args.joint, "q": q_values, "brace": True, "grid": meta["grid"], "lattice": lattice, "arms": "untucked"})
            name = f"brace__{args.scene}__{args.support}__{cat}-{model}__{key}"
        else:
            grasps, rest_quat, max_off = None, G.IDENTITY_QUAT.clone(), 0.0
            hull = None
            if args.object:
                cat, model = args.object.split("/", 1)
                obj = world.spawn(f"target_{cat}_{model}", cat, model)
                hull = world.local_hull(obj)
                world.park(obj, len(world.spawned) - 1)
                if not args.generic_pinch:
                    key = GL.resolve_key(cat, model, args.grasp_key, ["tabletop", "insert", "floating"])
                    grasps = GL.load_grasps(key)
                    rq = GL.rest_quat_xyzw(key)
                    rest_quat = rq if rq is not None else rest_quat
                    max_off = max(float(th.linalg.norm(g.position)) for g in grasps) if grasps else 0.0
                    meta.update({"object": args.object, "grasp_key": key, "n_grasps": len(grasps), "rest_quat_xyzw": rest_quat.tolist()})
                else:
                    meta.update({"object": args.object, "grasp_key": None, "generic_pinch": True})
                h = float(hull[:, 2].max() - hull[:, 2].min()) if hull is not None else 0.05
            else:
                meta.update({"object": None, "generic_pinch": True})
                h = 0.05
            if args.generic_pinch or grasps is None:
                pose_fn = None
            else:
                def pose_fn(world_xy, yaw, _hull=hull, _rq=rest_quat):
                    return bc.object_pose_at(world_xy, yaw, z_support, _hull, _rq)
            kit.sync_world()
            trunks = trunk_grid(z_support - world.floor_z)
            if args.smoke:
                trunks = trunks[:2]
            result = bc.compute(stance, lattice_world, z_support if grasps is not None else (z_support + max(0.6 * h, 0.03)), pose_fn,
                                grasps if not args.generic_pinch else None, ignore_grasp, arm="right", trunks=trunks, max_grasp_offset=max_off)
            gset = [(g.key, g.name, g.position.tolist(), g.quat.tolist()) for g in grasps] if grasps else "generic"
            key = _key({"scene": args.scene, "support": args.support, "z": round(z_support, 2), "stance": stance["edge"], "s": args.stand_off, "object": args.object,
                        "grasps": gset, "rest": rest_quat.tolist(), "placed": args.placed, "under": args.under, "grid": meta["grid"], "lattice": lattice, "arms": "untucked", "seeds": 12})
            oname = args.object.replace("/", "-") if args.object else "generic"
            name = f"{meta['mode']}__{args.scene}__{args.support}{'-under-' + args.under if args.under else ''}__{oname}__{key}"

        # ---- write
        lat_all = np.concatenate([c[:, 1] for c in result["cells"].values() if len(c)]) if any(len(c) for c in result["cells"].values()) else np.zeros(0)
        meta.update({"trunks": result["trunks"], "union_extent": [float(lat_all.min()), float(lat_all.max())] if len(lat_all) else None,
                     "n_reachable_total": int(sum(len(c) for c in result["cells"].values())), "key": key, "time_s": round(time.time() - t_load, 1)})
        arrays = {f"t{tr:.2f}_k{ti:+.1f}": c for (tr, ti), c in result["cells"].items()}
        np.savez_compressed(os.path.join(out_dir, name + ".npz"), **arrays)
        with open(os.path.join(out_dir, name + ".json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        print(f"REACH_BANDS {json.dumps({'file': os.path.join(out_dir, name + '.npz'), 'union_extent': meta['union_extent'], 'n_reachable_total': meta['n_reachable_total'], 'trunks': [(t['translate'], t['tilt'], t['n_reachable'], t['collides']) for t in result['trunks']]})}", flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        print("REACH_BANDS_FAILED", flush=True)
        raise
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
