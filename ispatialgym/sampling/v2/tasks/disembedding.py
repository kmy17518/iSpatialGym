"""Disembedding (spec §6.4, init per §4 + §4.7): find the target among distractors on the source tray and move it to the
destination tray.

Roles: ``target`` (grasped, tabletop pipeline), ``source`` / ``destination`` (the two fixed trays; the v1 facade names
them ``tray_source`` / ``tray_dest``), ``distractors`` (group, §3.12 rule 1 pools), scene ``support``, ``floor``.

Template (``sample_template``): hulls / rest orientations / heights, the target's library grasps, the trays' inner
floors (rasters in the tray frame), the reach band of the target at the tray-floor height (``BandSpec``), Phases 1a
(any edge) / 1b (H_max = tallest of target, distractors, trays) / 1c (row ``disembedding``: m = b + 0.437 + 0.02) and
Phase 2 with the source tray (centred on the focus, long axis along the edge) as the in-frame body.

Instance (``sample_instance``): a Phase-2 survivor + transition; source tray at the focus; the target inside it from
the feasible set F (footprint on the tray floor, 0.5 cm); the destination tray on a random side at the lateral
position where exactly 4 cm of it is inside the init image (>= 4 cm from the source tray, footprint on the top);
distractors on the source floor >= x from the target along the drawn grasp's pinch axis (x = pad_clearance + 0.01),
footprints grown by their pad clearance <= 0.45 of the floor; every object but the destination in frame (5 %);
whole-body checks at init and manipulation with the objects placed; occlusion <= 0.30 (two segmentation renders,
visible area >= 400 px) with the destination tray confirmed visible; the §4.7 placement stance with the place /
pre-place IK (object in hand, tray + support ignored for the place pose) and the lateral path with the object at the
carry height (>= tallest distractor + tray rim + 3 cm above the source floor). Sidecar: ``init`` / ``manipulation`` /
``placement`` stances, the transition, the path, one grasp record (pre-grasp, grasp, pre-place, place, carry_height,
``place_on`` = the destination), language descriptors; TRO ``initial_occlusion`` (``ratio``, ``visible_area``,
``reference_area``, ``occluder_ids``, ``camera_key``).
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import generation_defaults as GD
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.tro_writer import full_pose_target
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

log = logging.getLogger("ispatialgym.sampling.v2")

SUBTASK = "disembedding"


# ----------------------------------------------------------------------------------------------------------------
# tray inner floor (raster in the tray frame; §6.4 "placement raster 1 cm over the tray's inner floor")
# ----------------------------------------------------------------------------------------------------------------


#: a tray counts as resting on the top when its lowest hull point is within this of the top and its floor normal is
#: within ``REST_TILT_MAX_DEG`` of vertical after the measurement settle
REST_Z_TOL = 0.01
REST_TILT_MAX_DEG = 3.0
#: floor cells = tray hits within this band of the floor level (10 % quantile of the hit heights); at least ``FLOOR_MIN_CELLS``
FLOOR_BAND = 0.004
FLOOR_QUANTILE = 0.10
FLOOR_MIN_CELLS = 40


def repark_spawned(s) -> None:
    """Teleport every spawned object back onto its parking slot before any of its geometry is read.

    Why (E1 pilot, every disembedding alias): ``SimWorld.park`` disables gravity through the PhysX view only, and the
    ``og.sim.stop()`` / ``play()`` of every later ``spawn`` rebuilds PhysX from USD, so the objects parked earlier fall
    again (no floor under the parking slots) during the following simulator steps. ``SimWorld.local_hull`` then mixes
    the *render-side* (Fabric) link transforms, which physics-only steps leave at the last rendered pose, with the
    current PhysX root pose: the hull of a falling body comes out shifted by the distance fallen since the last
    render (+6.5 cm in z for the source tray), ``bottom_offset`` is off by the same amount and the tray is placed
    inside the table top (6 cm slab), pushed through it and found tilted 24 cm lower. A teleport writes PhysX *and*
    Fabric, zeroes the velocities and re-disables gravity; no stop / play happens during a template, so one re-park
    at the start of ``sample_template`` keeps every hull consistent for the template's lifetime."""
    for i, o in enumerate(s.world.spawned.values()):
        s.world.park(o, i)


def tilt_deg(quat) -> float:
    """Angle between the body's local z axis and the world vertical, in degrees."""
    up = C.quat_rotate(TT._t(quat), th.tensor([0.0, 0.0, 1.0]))
    return math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))


class TrayFloor:
    """Inner floor of a tray measured once in the tray's root frame (cells = local xy of the floor raster, ``z_local`` =
    floor height above the root origin); ``at(pose)`` gives the world view used for placements."""

    def __init__(self, cells_local: th.Tensor, z_local: float, res: float, rim_local: float):
        self.cells_local = cells_local
        self.z_local = float(z_local)
        self.res = float(res)
        self.rim_local = float(rim_local)  # top of the tray walls above the root origin
        lo = cells_local.min(dim=0).values
        hi = cells_local.max(dim=0).values
        self.origin = lo.clone()
        nx = int(round(float(hi[0] - lo[0]) / res)) + 1
        ny = int(round(float(hi[1] - lo[1]) / res)) + 1
        self.grid = th.zeros(nx, ny, dtype=th.bool)
        idx = th.round((cells_local - lo) / res).long()
        self.grid[idx[:, 0], idx[:, 1]] = True

    @property
    def n_cells(self) -> int:
        return int(self.grid.sum())

    def area(self) -> float:
        return self.n_cells * self.res * self.res

    def contains_local(self, xy_local, margin: float = 0.0) -> bool:
        p = xy_local if isinstance(xy_local, th.Tensor) else th.tensor(xy_local, dtype=th.float32)
        pts = [p]
        if margin > 0:
            for dx, dy in ((margin, 0), (-margin, 0), (0, margin), (0, -margin), (margin, margin), (-margin, -margin), (margin, -margin), (-margin, margin)):
                pts.append(p + th.tensor([dx, dy]))
        for q in pts:
            i = int(round(float(q[0] - self.origin[0]) / self.res))
            j = int(round(float(q[1] - self.origin[1]) / self.res))
            if i < 0 or j < 0 or i >= self.grid.shape[0] or j >= self.grid.shape[1] or not bool(self.grid[i, j]):
                return False
        return True

    def at(self, pose: TT.Pose) -> "TrayFloorWorld":
        return TrayFloorWorld(self, pose)

    @staticmethod
    def floor_cells(zs: th.Tensor, band: float = FLOOR_BAND, quantile: float = FLOOR_QUANTILE):
        """Floor level and cell mask from the per-ray tray hit heights (``nan`` = the ray missed the tray): the floor is the
        ``quantile`` of the hit heights (the lowest inner level; the rim and the wall tops are higher), a cell is a floor
        cell when its hit lies within ``band`` of it. Returns ``(floor_z, keep)``; ``floor_z`` is ``nan`` without hits."""
        valid = zs[~th.isnan(zs)]
        if len(valid) == 0:
            return float("nan"), th.zeros(len(zs), dtype=th.bool)
        floor_z = float(th.quantile(valid, quantile))
        keep = (~th.isnan(zs)) & ((zs - floor_z).abs() < band)
        return floor_z, keep

    @staticmethod
    def measure(s, tray, top, res: float = 0.01) -> "TrayFloor":
        """Place the tray on the free centre of the top, raycast a ``res`` raster from above, keep the cells whose first
        hit is the tray at its lowest inner level (10 % quantile of the hit heights + 4 mm), park the tray again.

        The tray is re-parked first so that ``local_hull`` reads a consistent body (see :func:`repark_spawned`), and after
        the settle it must rest on the top (bottom within ``REST_Z_TOL`` of the top, tilt < ``REST_TILT_MAX_DEG``): a tray
        that fell through or off the support is a ``Reject("support")`` with the geometry, not a raster of nothing."""
        s.world.park(tray, list(s.world.spawned).index(tray.name))
        hull = s.world.local_hull(tray)
        if len(hull) == 0:
            raise Reject("support", f"{tray.name}: no collision hull")
        cells = top.cells(free_only=True)
        centre = cells.mean(dim=0) if len(cells) else th.tensor([float(top.xs.mean()), float(top.ys.mean())])
        quat = C.IDENTITY_QUAT.clone()
        pos = th.tensor([float(centre[0]), float(centre[1]), top.z + s.world.bottom_offset(hull, quat) + TT.Z_OFFSET])
        tray.set_position_orientation(pos, quat)
        s.world.activate(tray)
        s.world.step_physics(20, still=[s.robot])
        try:
            rp, rq = tray.get_position_orientation()
            rp, rq = rp.float(), rq.float()
            # raster over the hull's bounds at the settled pose (``tray.aabb`` lags physics-only steps; the PhysX pose does not)
            world_hull = C.transform_points((rp, rq), hull)
            lo, hi = world_hull.min(dim=0).values, world_hull.max(dim=0).values
            tilt = tilt_deg(rq)
            where = (f"tray at {[round(float(v), 3) for v in rp.tolist()]} (placed at {[round(float(v), 3) for v in pos.tolist()]}), hull z {float(lo[2]):.3f}..{float(hi[2]):.3f}, "
                     f"tilt {tilt:.1f} deg, top z {float(top.z):.3f}")
            if abs(float(lo[2]) - float(top.z)) > REST_Z_TOL or tilt > REST_TILT_MAX_DEG:
                raise Reject("support", f"{tray.name}: not resting on the top after the settle ({where})")
            xs = th.arange(float(lo[0]) + res / 2, float(hi[0]), res)
            ys = th.arange(float(lo[1]) + res / 2, float(hi[1]), res)
            gx, gy = th.meshgrid(xs, ys, indexing="ij")
            starts = th.stack([gx.flatten(), gy.flatten(), th.full_like(gx.flatten(), float(hi[2]) + 0.10)], dim=1)
            ends = starts.clone()
            ends[:, 2] = float(lo[2]) - 0.02
            hits = s.world.raycast(starts, ends)
            prims = tuple({l.prim_path for l in tray.links.values()} | {tray.prim_path})
            zs = th.tensor([float(h["position"][2]) if (h["hit"] and str(h["rigidBody"]).startswith(prims)) else float("nan") for h in hits])
            floor_z, keep = TrayFloor.floor_cells(zs)
            n_tray = int((~th.isnan(zs)).sum())
            if n_tray == 0:
                bodies = sorted({str(h["rigidBody"]) for h in hits if h["hit"]})[:6]
                raise Reject("support", f"{tray.name}: no raycast hits on the tray ({sum(1 for h in hits if h['hit'])}/{len(hits)} rays hit {bodies}; rays from z "
                                        f"{float(starts[0, 2]):.3f} to {float(ends[0, 2]):.3f}; {where})")
            world_xy = starts[keep][:, :2]
            if len(world_xy) < FLOOR_MIN_CELLS:
                valid = zs[~th.isnan(zs)]
                raise Reject("support", f"{tray.name}: inner floor too small ({int(keep.sum())} cells within {FLOOR_BAND * 1000:.0f} mm of the floor z {floor_z:.3f}; "
                                        f"{n_tray}/{len(hits)} rays hit the tray at z {float(valid.min()):.3f}..{float(valid.max()):.3f}, raster {len(xs)}x{len(ys)} "
                                        f"at {res:.3f} m; {where})")
            local = C.transform_points(C.invert((rp, rq)), th.cat([world_xy, th.full((len(world_xy), 1), floor_z)], dim=1))
            rim = float(hi[2]) - float(rp[2])
            log.info("%s: inner floor %d cells at %.3f m (%.4f m2), floor %.3f m above the root, rim %.3f m", tray.name, len(world_xy), res, len(world_xy) * res * res,
                     float(local[:, 2].mean()), rim)
            return TrayFloor(local[:, :2].clone(), float(local[:, 2].mean()), res, rim)
        finally:
            s.world.park(tray, list(s.world.spawned).index(tray.name))


class TrayFloorWorld:
    """World view of a :class:`TrayFloor` at a tray pose: ``cells()`` (world xy), ``z`` (world floor height),
    ``polygon_supported(fp, margin)`` / ``contains(xy, margin)`` (§3.2 tray margin 0.5 cm), ``area()``, ``centre``."""

    def __init__(self, floor: TrayFloor, pose: TT.Pose):
        self.floor = floor
        self.pose = (TT._t(pose[0]), TT._t(pose[1]))
        self.inv = C.invert(self.pose)
        pts = C.transform_points(self.pose, th.cat([floor.cells_local, th.full((len(floor.cells_local), 1), floor.z_local)], dim=1))
        self._cells = pts[:, :2].clone()
        self.z = float(pts[:, 2].mean())
        self.centre = self._cells.mean(dim=0)
        self.rim_z = float(self.pose[0][2]) + floor.rim_local
        self.name = getattr(floor, "name", "tray_floor")

    def cells(self, free_only: bool = True) -> th.Tensor:
        return self._cells

    def area(self) -> float:
        return self.floor.area()

    def _local(self, xy) -> th.Tensor:
        p = th.tensor([float(xy[0]), float(xy[1]), self.z])
        return C.transform_points(self.inv, p.unsqueeze(0))[0, :2]

    def contains(self, xy, margin: float = 0.0) -> bool:
        return self.floor.contains_local(self._local(xy), margin)

    def polygon_supported(self, poly_xy: th.Tensor, margin: float = 0.005) -> bool:
        for p in poly_xy:
            if not self.contains(p, margin):
                return False
        return self.contains(poly_xy.mean(dim=0), 0.0)


# ----------------------------------------------------------------------------------------------------------------
# sampler
# ----------------------------------------------------------------------------------------------------------------


@register(SUBTASK)
class Disembedding(SamplerV2):
    def __init__(self, ctx):
        super().__init__(ctx)
        if self.gp.bddl is None:
            self.definition_id = 0  # v1 writer: disembedding-0
        self.defaults = GD.defaults_for(SUBTASK)
        self.src_role = "source" if "source" in self.gp.roles else "tray_source"
        self.dst_role = "destination" if "destination" in self.gp.roles else "tray_dest"
        self.top = None
        self.hull: Dict[str, th.Tensor] = {}
        self.rest: Dict[str, th.Tensor] = {}
        self.height: Dict[str, float] = {}
        self.grasps: List[Any] = []
        self.grasp_key: Optional[str] = None
        self.edge: Optional[EdgeFrame] = None
        self.init_stance: Optional[Stance] = None
        self.manip_candidates: List[Stance] = []
        self.src_floor: Optional[TrayFloor] = None
        self.dst_floor: Optional[TrayFloor] = None
        self.band_spec: Optional[TT.BandSpec] = None
        self.bands = None
        self.placed_key: Optional[str] = None  # band key of the stance-relative source tray (item 9)

    # ------------------------------------------------------------------ parameters (§2.3: numbers from generation)

    def g(self, key: str):
        return self.gen(key, self.defaults.get(key))

    # ------------------------------------------------------------------ roles

    def v1_role_aliases(self) -> Dict[str, str]:
        out = {}
        if self.src_role != "tray_source":
            out["tray_source"] = self.src_role
        if self.dst_role != "tray_dest":
            out["tray_dest"] = self.dst_role
        return out

    def bind_scene_roles(self):
        self.bind_floor_v2()
        req = self.choice.bindings.get("support")
        names = [n for n in (req if isinstance(req, list) else []) if isinstance(n, str)]
        if not names:
            raise Reject("spec", f"{self.gp.alias}: scenes.{self.scene}.bindings.support must list scene object names")
        self.bind_scene_object("support", self.world.obj(names[0]))

    def support_objects(self) -> List[Any]:
        """The table and both trays (ignored for the final grasp / place pose only, §3.6); bound roles only (parking runs
        before the roles are bound)."""
        return [self.obj(role) for role in ("support", self.src_role, self.dst_role) if self.bindings.has(role)]

    def spawn_roles(self):
        # owner decision 2026-09-16 4b: the source tray spawns at ``roles.source.scale`` (D1 writes 1.25; until it lands the
        # ``source_tray_scale`` default), the destination tray at 1.0. The scaled choice replaces the template's so the band
        # key, the hulls and the spawn agree
        mc = self.choice.models.get(self.src_role)
        if mc is not None and not mc.scale:
            s = self._source_scale(mc)
            if s is not None and abs(s - 1.0) > 1e-6:
                self.choice.models[self.src_role] = dataclasses.replace(mc, scale=[float(s)] * 3)
                self.template_meta["source_tray_scale"] = float(s)
        super().spawn_roles()
        if not self.bindings.has("distractors"):
            self._spawn_facade_distractors()

    def _source_scale(self, mc) -> Optional[float]:
        """Uniform spawn scale of the source tray: ``roles.<source>.scale`` of the customs entry (per model, or a scalar /
        list for the role), else the ``source_tray_scale`` generation default (1.25)."""
        role = self.gp.roles.get(self.src_role)
        raw = (self.gp.entry.get("roles") or {}).get(self.src_role, {}).get("scale") if isinstance(getattr(self.gp, "entry", None), dict) else None
        val = None
        if role is not None:
            try:
                val = role.scale_of(mc.category, mc.model)
            except AttributeError:
                val = None
        if val is None and raw is not None and not isinstance(raw, dict):
            val = raw
        if val is None:
            val = self.g("source_tray_scale")
        if isinstance(val, (list, tuple)):
            val = val[0] if val else None
        return None if val is None else float(val)

    def _tray_budget(self) -> float:
        """§6.4 distractor budget: D1's ``generation.distractor_budget`` (the grounding's maths: its ``budget`` / ``fraction``
        value, or a bare number) when present, else the ``tray_budget`` default (0.45)."""
        db = self.gen("distractor_budget", None)
        if isinstance(db, dict):
            for k in ("budget", "fraction", "max_fraction", "tray_budget"):
                if db.get(k) is not None:
                    return float(db[k])
        elif isinstance(db, (int, float)):
            return float(db)
        return float(self.g("tray_budget"))

    def _spawn_facade_distractors(self):
        """v1 facade only (no D1 ``distractors`` group in the template choice): draw the set from the v1 spec's
        ``distractors`` pools per §3.12 rule 1 — the target's category pool (or the other-category pool) minus the target
        and its look-alike group, at most one model per look-alike group, ``size`` ∈ generation.groups.distractors."""
        v1 = TT.facade_v1_spec(self)
        target = self.choice.models["target"]
        dspec = v1.distractor_spec() if v1 is not None else None
        if dspec is None:
            raise Reject("spec", f"{self.gp.alias}: no distractors group in the template choice and no v1 distractor pools")
        rng = np.random.default_rng(self.ctx.base_seed * 31 + 17)
        same = rng.random() < float(self.gen("same_category_prob", 0.5))
        pools = {"same_category": dspec.same_category_choices(target), "other_category": dspec.other_category_choices(target)}
        pool = pools["same_category" if same else "other_category"] or pools["other_category" if same else "same_category"]
        if not pool:
            raise Reject("spec", f"both distractor pools are empty for {target.category}/{target.model} after the look-alike exclusions")
        groups = [[tuple(m) for m in g] for g in dspec.lookalike_groups]
        cands = TT.exclude_lookalikes([(c.category, c.model) for c in pool], (target.category, target.model), groups, rng)
        size = self.gp.group_spec("distractors").get("size", self.defaults["groups"]["distractors"]["size"])
        n = int(rng.integers(int(size[0]), int(size[1]) + 1)) if isinstance(size, (list, tuple)) else int(size)
        picks = cands[: min(n, len(cands))]
        if not picks:
            raise Reject("spec", "no distractor candidate survives the look-alike exclusion")
        by_tag = {(c.category, c.model): c for c in pool}
        for cm in picks:
            self.spawn_role("distractors", by_tag[cm])
        self.template_meta["distractors"] = {"pool": "same_category" if same else "other_category", "models": [f"{c}/{m}" for c, m in picks],
                                             "pool_size": len(pool), "n_after_lookalikes": len(cands)}

    # ------------------------------------------------------------------ template

    @property
    def tname(self) -> str:
        return self.bindings.one("target").name

    def sample_template(self):
        tgt, src, dst = self.obj("target"), self.obj(self.src_role), self.obj(self.dst_role)
        # every hull below is read off a parked body: re-park first (consistent PhysX / render-side transforms, gravity
        # off again after the spawns' stop / play), see ``repark_spawned``
        repark_spawned(self)
        self.grasps, self.grasp_key, self.rest["target"] = TT.library_grasps(self, "target", "tabletop")
        for key, o in (("target", tgt), (self.src_role, src), (self.dst_role, dst)):
            self.hull[key] = self.world.local_hull(o)
            self.rest.setdefault(key, C.IDENTITY_QUAT.clone())
            self.height[key] = TT.height_of(self.hull[key], self.rest[key])
        for b in self.bindings.many("distractors"):
            o = self.objects[b.name]
            self.hull[b.name] = self.world.local_hull(o)
            self.rest[b.name] = C.IDENTITY_QUAT.clone()
            self.height[b.name] = TT.height_of(self.hull[b.name], self.rest[b.name])
        support = self.obj("support")
        self.top = self.support_top(support)
        if int(self.top.free.sum()) < 50:
            raise Reject("support_small", f"{support.name}: too little free top")
        res = float(self.g("tray_raster_m"))
        self.src_floor = TrayFloor.measure(self, src, self.top, res)
        self.dst_floor = TrayFloor.measure(self, dst, self.top, res)
        self.kit.sync_world()
        # band: the target grasped at the source-floor height; the tray moves with the focus, so the lattice is the top
        # itself (the tray walls are ignored for the grasp pose per §3.6; the pre-grasp is 10 cm above the rim in practice)
        mc = self.choice.models["target"]
        src_mc = self.choice.models[self.src_role]
        # band key (item 9, B3's ``reach_bands.PlacedSpec`` grammar ``CATEGORY/MODEL[*SCALE][@MODE]``): the source tray at
        # its spawn scale, mode ``focus_centred_edge_aligned`` = inner-floor centroid on the manipulation focus cell, long
        # axis along the edge — stance-relative, so every template of the alias hits the same band
        src_scale = float(src_mc.scale[0]) if src_mc.scale else 1.0
        mode = getattr(TT.RB, "PLACED_MODE_FOCUS", "focus_centred_edge_aligned")
        spec_obj = getattr(TT.RB, "PlacedSpec", None)
        if spec_obj is not None:
            self.placed_key = spec_obj.parse((f"{src_mc.category}/{src_mc.model}", src_scale if abs(src_scale - 1.0) > 1e-6 else None, mode)).canonical()
        else:
            self.placed_key = f"{src_mc.category}/{src_mc.model}" + (f"*{src_scale:g}" if abs(src_scale - 1.0) > 1e-6 else "") + f"@{mode}"
        self.band_spec = TT.BandSpec(object=f"{mc.category}/{mc.model}", grasp_key=self.grasp_key, grasps=self.grasps, hull=self.hull["target"],
                                     rest_quat=self.rest["target"], scale=list(mc.scale) if mc.scale else None, placed=[self.placed_key],
                                     z_support=self.top.z + self.src_floor.z_local + self.world.bottom_offset(self.hull[self.src_role], C.IDENTITY_QUAT),
                                     height=self.height["target"])
        H_max = max(self.height.values())
        margins = {"kind": "disembedding", "lateral_margin": float(self.g("lateral_margin_m")), "safety": 0.02, "H_max": H_max}
        rng = np.random.default_rng(self.ctx.base_seed)
        errors = []
        # §4.2 has no preference among the edges and §4.4 checks floor + collision per lateral position (Phase 1c), so
        # every edge with a free-floor run in front of it is tried (a third of its 10 cm lateral raster at s = 0.45 on
        # free floor, as the IORS reference task does) instead of ``edge_candidates``, which drops an edge when its
        # *midpoint* stance alone is blocked: on breakfast_table_rhjoby_0 that is the 2.9 m edge — the only one long
        # enough for the disembedding margins (m_R + m_L ≈ 2.26 m) — whose midpoint sits on a 0.4 m patch the scene
        # variant leaves blocked in the traversability map (baked chair footprint wider than the freed bbox)
        for edge, st in TT.edge_frames(self, self.top, rng):
            lats = th.linspace(-edge.length / 2 + 0.1, edge.length / 2 - 0.1, max(2, int(edge.length / 0.1))).tolist()
            n_free = sum(1 for lat in lats if self.kit.trav_free(edge.base_xy(st.stand_off, lat)))
            if n_free < max(1, len(lats) // 3):
                errors.append(f"edge {edge.name}: free floor at {n_free}/{len(lats)} lateral positions")
                log.info("%s: %s", self.gp.alias, errors[-1])
                continue
            try:
                bands = TT.ensure_bands(self, self.band_spec, top=self.top, edge=edge)
                init = TT.phase1b_init_trunk(self, st, H_max, bands=bands, top=self.top)
                init = TT.phase1c_lateral(self, init, margins, rng, top=self.top, bands=bands)
                init.name = "init"
                cands = TT.phase2_manipulation_stance(self, init, TT.half_extent(self.hull[self.src_role], C.IDENTITY_QUAT), target_height=self.height[self.src_role],
                                                      in_frame_fn=self._source_tray_in_frame, bands=bands, top=self.top, focus_z=self.top.z + self.src_floor.z_local)
                if not cands:
                    raise Reject("no_manipulation_stance", "Phase 2 left no survivor with the source tray in frame")
                self.edge, self.init_stance, self.manip_candidates = edge, init, cands
                break
            except Reject as e:
                errors.append(f"edge {edge.name}: {e}")
                log.info("%s: %s", self.gp.alias, errors[-1][:300])
        if self.init_stance is None:
            raise Reject("no_stance", "; ".join(errors)[:1500])
        self.template_facts.update({"support": self.top.to_json(), "edge": self.edge.to_json(), "init_stance": self.init_stance.to_meta(),
                                    "n_manipulation_candidates": len(self.manip_candidates), "grasp_key": self.grasp_key, "n_grasps": len(self.grasps),
                                    "band": getattr(self.bands, "path", None), "placed_key": self.placed_key, "source_tray_scale": src_scale,
                                    "src_floor_cells": self.src_floor.n_cells, "dst_floor_cells": self.dst_floor.n_cells,
                                    "H_max": H_max, "distractors": [b.name for b in self.bindings.many("distractors")]})

    def _tray_yaw(self, key: str, stance: Stance) -> float:
        """Yaw putting the tray's longer footprint axis along the front edge (stance tangent)."""
        ext = TT.hull_extents(self.hull[key], C.IDENTITY_QUAT)
        tangent_yaw = math.atan2(float(stance.tangent[1]), float(stance.tangent[0]))
        return tangent_yaw if float(ext[0]) >= float(ext[1]) else tangent_yaw - math.pi / 2

    def _tray_pose_centred(self, key: str, floor: TrayFloor, stance: Stance, centre_xy) -> TT.Pose:
        """Tray pose with its inner-floor centre at ``centre_xy`` and its long axis along the edge."""
        yaw = self._tray_yaw(key, stance)
        quat = C.quat_from_yaw(yaw)
        c_local = floor.cells_local.mean(dim=0)
        off = C.quat_rotate(quat, th.tensor([float(c_local[0]), float(c_local[1]), 0.0]))
        pos = th.tensor([float(centre_xy[0]) - float(off[0]), float(centre_xy[1]) - float(off[1]), self.top.z + self.world.bottom_offset(self.hull[key], quat) + TT.Z_OFFSET])
        return pos, quat

    def _source_tray_in_frame(self, s, stance: Stance, cam, focus, info) -> bool:
        """§4.5 step 3 (disembedding): the whole source tray centred on the focus, long axis along the edge, in frame (5 %)."""
        pose = self._tray_pose_centred(self.src_role, self.src_floor, stance, focus[:2])
        if not self.top.polygon_supported(self.world.footprint(self.hull[self.src_role], pose), margin=TT.SUPPORT_MARGIN):
            return False
        return TT.in_frame(self, cam, TT.hull_corners_world(self.hull[self.src_role], pose), TT.IN_FRAME_MARGIN)

    # ------------------------------------------------------------------ instance

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        tname, sname, dname = self.tname, self.bindings.one(self.src_role).name, self.bindings.one(self.dst_role).name
        dnames = [b.name for b in self.bindings.many("distractors")]
        init = self.init_stance
        errors = []
        for k in rng.permutation(len(self.manip_candidates))[:6]:
            cand = self.manip_candidates[int(k)]
            manip = cand.with_trunk(cand.translate, cand.tilt, "manipulation")
            manip.focus, manip.arms = cand.focus, ["right"]
            self.sidecar.reset_instance()
            self.sidecar.add_stance(init)
            for o in self.world.spawned.values():
                self.world.park(o, list(self.world.spawned).index(o.name))
            try:
                d = self._sample_with(rng, init, manip, tname, sname, dname, dnames, int(k))
                return d
            except Reject as e:
                errors.append(f"candidate {int(k)}: {e}")
                log.info("%s: %s", self.gp.alias, errors[-1][:300])
        raise Reject("target_placement", "; ".join(errors[-3:])[:1500])

    def _sample_with(self, rng, init: Stance, manip: Stance, tname: str, sname: str, dname: str, dnames: List[str], k: int) -> Draft:
        d = Draft()
        support = self.obj("support")
        src, dst, tgt = self.obj(self.src_role), self.obj(self.dst_role), self.obj("target")
        # §4.5 step 5: transition init -> manipulation
        self.record_transition(init, manip)
        # source tray at the focus, aligned with the edge (§4.6 step 1, disembedding)
        s_pose = self._tray_pose_centred(self.src_role, self.src_floor, manip, manip.focus[:2])
        if not self.top.polygon_supported(self.world.footprint(self.hull[self.src_role], s_pose), margin=TT.SUPPORT_MARGIN):
            raise Reject("support", "source tray footprint not on the top at the focus")
        src.set_position_orientation(*s_pose)
        src.keep_still()
        self.kit.sync_world()
        src_w = self.src_floor.at(s_pose)
        # target inside the tray from F (footprint on the tray floor, 0.5 cm), exact IK with support + tray ignored for the grasp
        t_pose, grasp = TT.phase3_target(self, manip, rng, self.grasps, self.hull["target"], self.rest["target"], [support, src], top=src_w, support_z=src_w.z,
                                         margin=float(self.g("tray_footprint_margin_m")), scale_key=self.choice.models["target"].scale, bands=self.bands)
        grasp.place_on = dname
        # destination tray: random side, exactly 4 cm inside the init image, >= 4 cm from the source tray, footprint on the top
        cam_init = self.camera_pose_at(init)
        d_pose, side, dest_diag = self._place_destination(rng, init, manip, cam_init, s_pose)
        dst_w = self.dst_floor.at(d_pose)
        # distractors on the source floor (spacing along the pinch axis, budget)
        width = TT.grasp_pinch_width(next((g for g in self.grasps if g.name == grasp.grasp_name), None))
        spacing = TT.distractor_spacing(width if width is not None else 0.0, float(self.g("distractor_spacing_extra_m")))
        pinch_axis = C.closing_axis(grasp.grasp[1])
        items = [(n, self.hull[n], self.rest[n]) for n in dnames]
        grow = {n: TT.distractor_spacing(width if width is not None else 0.0, 0.0) for n in dnames}
        dist_poses = TT.place_distractors(self, rng, src_w, src_w.z, t_pose, self.hull["target"], items, spacing_axis=pinch_axis[:2], spacing=spacing,
                                          margin=float(self.g("tray_footprint_margin_m")), gap=0.01, cam=cam_init, budget=self._tray_budget(), grow=grow)
        objects = {tname: t_pose, sname: s_pose, dname: d_pose, **dist_poses}
        # §4.6 step 4: every object in frame (5 %) in the init image except the destination tray
        items_all = [(tname, self.hull["target"], t_pose), (sname, self.hull[self.src_role], s_pose)] + [(n, self.hull[n], dist_poses[n]) for n in dnames]
        missing = TT.all_in_frame(self, cam_init, items_all, float(self.g("image_margin")))
        if missing:
            raise Reject("not_in_frame", f"{missing} not fully in the init image")
        # §4.6 step 5: whole-body checks at init and manipulation with the objects placed
        TT.whole_body_check_at(self, [init, manip], objects, require=True, reason="stance_collision")
        # §4.7 placement stance and the place IK (object in hand; destination tray + support ignored for the place pose only)
        goal = self._goal_pose(t_pose, dst_w)
        place_st = TT.placement_stance(self, manip, dst_w.centre)
        TT.whole_body_check_at(self, [place_st], {}, require=True, reason="no_placement_stance")
        place = IK.grasp_pose(goal, grasp.eef_in_object)
        pre_place = IK.pre_pose(place)
        with TT.posed_in_hand(self, "right", tgt, grasp.eef_in_object):
            with attached(self.kit, "right", tgt):
                res = self.ikc.exact([pre_place, place], place_st, "right", ignore_for_last=[support, dst], scale_key=self.choice.models["target"].scale)
            if not res.ok:
                raise Reject("place_unreachable", f"pre-place / place fail the exact IK from the placement stance: {res.reason}")
            grasp.pre_place, grasp.place = pre_place, place
            # carry height: object bottom >= tallest distractor + tray rim + 3 cm above the source floor
            rim_h = src_w.rim_z - src_w.z
            tallest = max([self.height[n] for n in dnames] + [rim_h])
            grasp.carry_height = float(tallest + float(self.g("carry_height_extra_m")))
            # §3.7 path manipulation -> placement (straight lateral segment) with the object attached
            self.record_path(manip, place_st, carried=tname, arm="right")
        cam_manip = self.camera_pose_at(manip)
        if not self.ikc.eef_in_frame(grasp.grasp, cam_manip):
            raise Reject("gripper_not_in_frame", "grasp eef pose outside the manipulation image (5 %)")
        if not self.ikc.eef_in_frame(place, self.camera_pose_at(place_st)):
            raise Reject("gripper_not_in_frame", "place eef pose outside the placement image (5 %)")
        # occlusion by renders at the init stance (§6.4): ratio <= 0.30, visible area >= 400 px, destination visible
        occ = self._measure_occlusion(init, objects, tname, dnames, dname)
        # record
        self.sidecar.add_stance(manip)
        self.sidecar.add_stance(place_st)
        self.sidecar.add_grasp(grasp)
        d.object_poses.update(objects)
        d.robot_q = self.stance_q(init)
        d.base_pose2d = tuple(init.base_pose2d)
        d.goal_poses[tname] = goal
        d.pose_targets[tname] = full_pose_target(goal[0], goal[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
        d.extras["initial_occlusion"] = occ
        d.summary.update({"manipulation_candidate": k, "grasp": {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw, "pinch_width": width},
                          "dest_side": "left" if side > 0 else "right", "destination": dest_diag, "distractor_spacing_m": spacing, "occlusion": occ,
                          "carry_height": grasp.carry_height, "placement_stance": place_st.to_meta()})
        self.sidecar.describe(tname, self.choice.models["target"].category.replace("_", " "), self._descriptor(tname, objects, cam_init))
        self.sidecar.describe(sname, "source tray", "the tray in front of the robot")
        self.sidecar.describe(dname, "destination tray", f"the tray on the {'left' if side > 0 else 'right'}")
        for n in dnames:
            b = self.bindings.by_name()[n]
            self.sidecar.describe(n, (b.category or n).replace("_", " "), self._descriptor(n, objects, cam_init))
        return d

    def _descriptor(self, name: str, objects: Dict[str, TT.Pose], cam) -> str:
        """Image-space descriptor (left / centre / right, front / back on the source tray) for the language annotations."""
        b = self.bindings.by_name()[name]
        cm = self.world.camera_model
        uv, _ = cm.project(cam, TT._t(objects[name][0]).unsqueeze(0))
        u, v = float(uv[0, 0]) / cm.width, float(uv[0, 1]) / cm.height
        col = "left" if u < 0.4 else ("right" if u > 0.6 else "centre")
        row = "front" if v > 0.6 else ("back" if v < 0.4 else "middle")
        return f"the {(b.category or name).replace('_', ' ')} at the {row} {col} of the source tray"

    def _goal_pose(self, t_pose: TT.Pose, dst_w: TrayFloorWorld) -> TT.Pose:
        """§6.4 goal: the target at the destination floor centre with its init orientation."""
        quat = TT._t(t_pose[1])
        pos = th.tensor([float(dst_w.centre[0]), float(dst_w.centre[1]), dst_w.z + self.world.bottom_offset(self.hull["target"], quat) + TT.Z_OFFSET])
        return pos, quat

    def _place_destination(self, rng, init: Stance, manip: Stance, cam, s_pose: TT.Pose):
        """§4.6 (disembedding): destination on the left or right (random), aligned with the edge and the source tray (same
        depth), at the lateral position where exactly ``destination_visible_width_m`` of it is inside the init image
        (bisected on the camera projection of the point that far inside its inner edge), >= 4 cm from the source tray,
        footprint on the top. Returns (pose, side, diagnostics); ``Reject("destination_placement")``."""
        top, cm = self.top, self.world.camera_model
        key = self.dst_role
        yaw_d = self._tray_yaw(key, manip)
        quat = C.quat_from_yaw(yaw_d)
        fp0 = self.world.footprint(self.hull[key], (th.zeros(3), quat))
        half_lat_d = float((fp0 @ manip.tangent).abs().max())
        s_fp = self.world.footprint(self.hull[self.src_role], s_pose)
        s_lat = (s_fp - manip.base_xy) @ manip.tangent
        depth_s, _ = manip.local(s_pose[0])
        w = float(self.g("destination_visible_width_m"))
        gap_min = float(self.g("destination_gap_min_m"))
        top_z_d = top.z + self.world.bottom_offset(self.hull[key], quat) + TT.height_of(self.hull[key], quat)
        sides = [1.0, -1.0] if rng.random() < 0.5 else [-1.0, 1.0]
        diag: Dict[str, Any] = {}

        def pose_at(lat):
            xy = manip.world(depth_s, lat)
            return (th.tensor([float(xy[0]), float(xy[1]), top.z + self.world.bottom_offset(self.hull[key], quat) + TT.Z_OFFSET]), quat)

        def border_gap(sign, lat):
            """<= 0 while >= w of the tray is inside the image, > 0 once less than w shows (raw border, the destination is exempt from the 5 % rule)."""
            p = manip.world(depth_s, lat - sign * (half_lat_d - w))
            uv, dep = cm.project(cam, th.tensor([[float(p[0]), float(p[1]), top_z_d]]))
            if float(dep[0]) <= 0.05:
                return 1.0
            u = float(uv[0, 0])
            return (u - cm.width) if sign < 0 else (-u)  # robot's right = image right border, robot's left = image left

        for sign in sides:
            l_min = (float(s_lat.max()) + gap_min + half_lat_d) if sign > 0 else (-float(s_lat.min()) + gap_min + half_lat_d)
            l_max = None
            for l in th.arange(l_min, 3.0, 0.02).tolist():
                if not top.polygon_supported(self.world.footprint(self.hull[key], pose_at(sign * l)), margin=TT.SUPPORT_MARGIN):
                    break
                l_max = l
            if l_max is None:
                diag[f"side_{'left' if sign > 0 else 'right'}"] = "no supported position"
                continue
            g_lo, g_hi = border_gap(sign, sign * l_min), border_gap(sign, sign * l_max)
            if not (g_lo <= 0 <= g_hi):
                diag[f"side_{'left' if sign > 0 else 'right'}"] = f"border gap {g_lo:.0f}..{g_hi:.0f} px over lateral {l_min:.2f}..{l_max:.2f}"
                continue
            lo, hi = l_min, l_max
            for _ in range(18):
                mid = 0.5 * (lo + hi)
                if border_gap(sign, sign * mid) <= 0:
                    lo = mid
                else:
                    hi = mid
            lat = sign * 0.5 * (lo + hi)
            pose = pose_at(lat)
            if not top.polygon_supported(self.world.footprint(self.hull[key], pose), margin=TT.SUPPORT_MARGIN):
                diag[f"side_{'left' if sign > 0 else 'right'}"] = "bisected position not supported"
                continue
            diag.update({"lateral_m": lat, "visible_width_m": w, "lateral_range": [l_min, l_max]})
            return pose, sign, diag
        raise Reject("destination_placement", f"no lateral position keeps exactly {w:.2f} m of the destination tray in the init image ({diag})")

    def _measure_occlusion(self, init: Stance, objects: Dict[str, TT.Pose], tname: str, dnames: List[str], dname: str) -> Dict[str, Any]:
        """§6.4 occlusion by two segmentation renders at the init stance: ``ratio = 1 − V/A`` (A = the target alone, the
        distractors hidden), ratio <= 0.30, V >= 400 px, target inside the 5 % margin (checked by the caller), the
        destination tray visible in the same render."""
        for name, pose in objects.items():
            o = self.objects[name]
            o.set_position_orientation(TT._t(pose[0]), TT._t(pose[1]))
            o.wake()
            o.keep_still()
        self.kit.apply(self.stance_q(init), init.base_pose2d)
        self.world.step_physics(1, still=[self.robot] + [self.objects[n] for n in objects])
        target = self.objects[tname]
        hidden = [self.objects[n] for n in dnames]
        vis = self.world.object_pixels(target)
        ref = self.world.object_pixels(target, hidden=hidden)
        if ref == 0:
            self.world.ensure_segmented(target)
            vis = self.world.object_pixels(target)
            ref = self.world.object_pixels(target, hidden=hidden)
        if ref == 0:
            raise Reject("seg_missing", f"{target.name} has no segmentation pixels even with the distractors hidden")
        ratio = 1.0 - vis / max(ref, 1)
        px = self.world.visible_pixels(exclude_structure=False)
        dest_px = px.get(self.objects[dname].name, 0)
        if ratio > float(self.g("max_occlusion_ratio")):
            raise Reject("occlusion", f"occlusion ratio {ratio:.2f} > {self.g('max_occlusion_ratio')}")
        if vis < int(self.g("min_visible_pixels")):
            raise Reject("occlusion", f"target visible area {vis} px < {self.g('min_visible_pixels')}")
        if dest_px <= 0:
            raise Reject("dest_not_visible", "destination tray not visible in the confirmation render")
        return {"target": tname, "occluders": list(dnames), "ratio": float(ratio), "visible_area": int(vis), "reference_area": int(ref),
                "occluder_ids": [self.objects[n].name for n in dnames], "camera_key": self.world.cam_key, "destination_pixels": int(dest_px)}

    # ------------------------------------------------------------------ goal / pipeline hooks

    def goal(self, draft: Draft):
        manip = self.sidecar.stance("manipulation")
        self.sidecar.set_goal(manip.base_pose2d, manip.translate, manip.tilt, arms="same")

    def settle_entities(self, draft: Draft):
        return [self.objects[n] for n in draft.object_poses]

    def after_settle(self, draft: Draft):
        """The trays settled: the goal is the destination floor centre at the settled destination pose with the settled
        target orientation."""
        tname, dname = self.tname, self.bindings.one(self.dst_role).name
        dst_w = self.dst_floor.at(draft.object_poses[dname])
        goal = self._goal_pose(draft.object_poses[tname], dst_w)
        draft.goal_poses[tname] = goal
        draft.pose_targets[tname] = full_pose_target(goal[0], goal[1], self.tol("position", 0.03), self.tol("orientation", 0.20))

    def extra_checks(self, draft: Draft):
        """After settling: the occlusion measured again (the loader re-measures it after its deterministic settle)."""
        tname, dname = self.tname, self.bindings.one(self.dst_role).name
        dnames = [b.name for b in self.bindings.many("distractors")]
        init = self.sidecar.stance("init")
        cam = self.camera_pose_at(init)
        if not self.frustum_ok(cam, self.hull_corners_world(self.hull["target"], draft.object_poses[tname]), float(self.g("image_margin"))):
            raise Reject("not_in_frame", "target left the 5 % image margin while settling")
        occ = self._measure_occlusion(init, draft.object_poses, tname, dnames, dname)
        draft.extras["initial_occlusion"] = occ
        draft.summary["occlusion_settled"] = occ

    def after_goal_check(self, draft: Draft):
        """§4.7: the place / pre-place at the settled goal pose from the placement stance (object in hand, destination
        tray and support ignored for the place pose only)."""
        tname = self.tname
        grasp = self.sidecar.grasp_of(tname)
        place_st = self.sidecar.stance("placement")
        goal_pos = th.tensor(draft.pose_targets[tname]["position"])
        goal_quat = th.tensor(draft.pose_targets[tname]["orientation"])
        place = IK.grasp_pose((goal_pos, goal_quat), grasp.eef_in_object)
        pre_place = IK.pre_pose(place)
        tgt = self.obj("target")
        with TT.posed_in_hand(self, "right", tgt, grasp.eef_in_object):
            with attached(self.kit, "right", tgt):
                res = self.ikc.exact([pre_place, place], place_st, "right", ignore_for_last=[self.obj("support"), self.obj(self.dst_role)],
                                     scale_key=self.choice.models["target"].scale)
        if not res.ok:
            raise Reject("place_unreachable", f"settled goal: pre-place / place fail the exact IK from the placement stance: {res.reason}")
        grasp.pre_place, grasp.place = pre_place, place

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        g = self.sidecar.grasp_of(self.tname)
        manip = self.sidecar.stance("manipulation")
        if g is None or manip.focus is None:
            return []
        mc = self.choice.models["target"]
        return [{"scope": {"target": f"{mc.category}/{mc.model}", "support_height": round(float(self.top.z - self.world.floor_z), 2)}, "type": "grasp_set",
                 "trunk": [manip.translate, round(manip.tilt, 2)], "cell": [round(v, 2) for v in manip.world_to_base(manip.focus[:2]).tolist()],
                 "feasible": [[round(g.yaw, 2), g.grasp_name]], "hard": False}]
