"""IORS inside, large-scale (spec §5.6 "Inside (large-scale)"; owner 2026-10-12 ~16:05 item 1: inside_large_scale#03 plate -> drop-in sink,
#02 jar -> fridge shelf): :class:`InsideMode`, the mode ``IORSLarge._sample_inside`` returns (``iors_large._adopt_mode`` puts it in front of
IORSLarge; stances A / B, the B -> C coupling, the carry and the records stay IORSLarge's).

Neither partner has a fillable meta link, so the goal surface is the partner's **interior surface**: ``SimWorld.support_interior`` at
``generation.inside_surface_z_m`` (m above the floor) with ``inside_surface_tol_m`` as its height tolerance - a sink's basin floor (the
downward rays start just under the rim; a generous tolerance finds the floor whatever the basin's depth) or a fridge shelf (the code of the
bookcase compartments). The same two numbers are the goal-side band's ``--surface-z`` / ``--surface-tol``: ``ReachBands.load(...,
surface_z=...)`` matches the requested height. Goal cells: the free board cells with headroom for the target (+ 2 cm), at least the target's
half extent + ``opening_margin_m`` from the board's outline (the opening shrunk; ``inside_erosion_m`` replaces that distance - a plate as wide
as a basin's flat floor rests with its rim over the flaring walls).

Stance C faces the opening: base square to a side of the interior raster (``inside_stance_side``: the world direction the opening faces, e.g.
``"+y"``; default every side, pooled nearest first), stand-offs ``inside_stand_offs_m`` (default :data:`STAND_OFFS`) + the partner's AABB overhang,
laterals along the side, trunks from the goal-side band (without one: the :data:`NO_BAND_TRUNKS` trunks per base pose
with the most reach margin to the goal cell); the goal cell = the band-admitted goal cell nearest the base; the target at the goal in frame
(5 %). Release (the small-scale drop): the target's bottom ``drop_height_m`` above the surface, raised until the open gripper's lowest point
clears the surface by ``release_clearance_m`` for every library grasp; the partner stays in the collision world (the oracle's drop keeps its
container). Goal check: teleport to the release, settle (the drop test), ``inside`` (the BDDL goal; ``predicates.inside_rays`` decides for a
container without a fillable meta link) and the target's top at most ``max_protrusion_fraction`` of its height above the rim. The oracle
program places with kind ``drop`` (``programs.program_iors_large_inside``).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import object_joints as OJ
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import prompt0 as P0
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.sampling.v2.tasks.iors_small import gripper_corners_world

log = logging.getLogger("ispatialgym.sampling.v2")

#: stand-offs of stance C from the interior raster's side (+ the partner's AABB overhang); one more than ontop's: the raster side of a basin
#: lies 10-20 cm behind the counter's front edge
STAND_OFFS = (0.35, 0.45, 0.55, 0.65)
SURFACE_TOL_M = 0.03
HEADROOM_MARGIN_M = TT.INTERIOR_HEADROOM_MARGIN_M
OPENING_MARGIN_M = 0.02
DROP_HEIGHT_M = 0.005
RELEASE_CLEARANCE_M = 0.02
MAX_PROTRUSION_FRACTION = 0.5
#: without a goal-side band: the trunks per base pose kept for the coupling, by reach margin to the goal cell (shoulder -> cell), chosen among
#: the collision-free ones with the goal in frame (the most margin = the deepest lean, which the counter run in front of a basin blocks)
NO_BAND_TRUNKS = 3
SIDE_MAX_DEG = 45.0


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (tested without a simulator)
# ----------------------------------------------------------------------------------------------------------------


def raster_rim(mask: th.Tensor, xs: th.Tensor, ys: th.Tensor) -> th.Tensor:
    """World xy (N, 2) of the raster cells of ``mask`` with a missing 4-neighbour (the board's outline)."""
    m = mask.bool()
    pad = th.nn.functional.pad(m.float()[None, None], (1, 1, 1, 1))[0, 0] > 0.5
    interior = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    idx = th.nonzero(m & ~interior)
    return th.stack([xs[idx[:, 0]], ys[idx[:, 1]]], dim=1) if len(idx) else th.zeros(0, 2)


def erode_from(cells: th.Tensor, rim: th.Tensor, margin: float) -> th.Tensor:
    """The cells (N, 2) farther than ``margin`` from every rim point."""
    if len(cells) == 0 or len(rim) == 0:
        return cells
    return cells[th.cdist(cells.float(), rim.float()).min(dim=1).values > float(margin)]


def side_vector(side: Any) -> Optional[th.Tensor]:
    """``generation.inside_stance_side``: ``"+x" / "-x" / "+y" / "-y"`` or ``[dx, dy]`` -> unit world xy; None -> None."""
    if side is None:
        return None
    if isinstance(side, str):
        s = side.strip().lower()
        table = {"+x": (1.0, 0.0), "x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "y": (0.0, 1.0), "-y": (0.0, -1.0)}
        if s not in table:
            raise ValueError(f"inside_stance_side {side!r}: expected +x / -x / +y / -y or [dx, dy]")
        v = th.tensor(table[s])
    else:
        v = th.tensor([float(side[0]), float(side[1])])
    n = float(th.linalg.norm(v))
    if n < 1e-9:
        raise ValueError(f"inside_stance_side {side!r}: zero vector")
    return v / n


def edges_facing(edges: Sequence[EdgeFrame], side: Optional[th.Tensor], max_deg: float = SIDE_MAX_DEG) -> List[EdgeFrame]:
    """The edges whose outward normal (-inward) lies within ``max_deg`` of the world direction ``side``, best aligned first; every
    edge when ``side`` is None."""
    if side is None:
        return list(edges)
    cos_min = math.cos(math.radians(max_deg)) - 1e-6
    scored = [(float(th.dot(-e.inward.float(), side.float())), e) for e in edges]
    return [e for c, e in sorted(scored, key=lambda t: -t[0]) if c >= cos_min]


def overhang(mid, outward, lo, hi) -> float:
    """AABB overhang beyond a raster side along its outward normal (>= 5 cm counts, else 0; ``iors_large.edge_overhang``'s rule)."""
    lo, hi = G.t(lo).float(), G.t(hi).float()
    corners = G.aabb_corners(lo, hi)[:, :2]
    n = G.t(outward).float()[:2]
    over = float((corners @ n).max()) - float(th.dot(G.t(mid).float()[:2], n))
    return over if over >= 0.05 else 0.0


def release_lift(finger_lows: Sequence[float], surface_z: float, clearance: float) -> float:
    """How far the release must rise so that the lowest open-gripper point of every grasp clears the surface by ``clearance``."""
    if not finger_lows:
        return 0.0
    return max(0.0, float(surface_z) + float(clearance) - min(float(v) for v in finger_lows))


def rim_height(top_z: Optional[float], surface_z: float, height: float, aabb_top: float) -> Tuple[float, str]:
    """The rim the protrusion is measured against: the partner's support top (its most populated top height) when it lies at least the
    target's height above the interior surface (a sink's deck, a fridge's top), else the partner's AABB top (permissive)."""
    if top_z is not None and float(top_z) >= float(surface_z) + float(height):
        return float(top_z), "support_top"
    return float(aabb_top), "aabb_top"


def protrusion_ok(target_top: float, rim_z: float, height: float, fraction: float) -> Tuple[bool, float]:
    """§6.11 inside drop test: the target's top at most ``fraction`` of its height above the rim; (ok, protrusion m)."""
    prot = float(target_top) - float(rim_z)
    return prot <= float(fraction) * float(height), prot


def nearest_cell(pool: th.Tensor, xy) -> th.Tensor:
    p = G.t(xy).float()[:2]
    return pool[int(th.argmin(th.linalg.norm(pool.float() - p, dim=1)))]


def best_per_base(stances: Sequence[Stance], k: int) -> List[Stance]:
    """Per base pose, the ``k`` stances with the most ``extras["reach_margin_m"]`` (bases in their first order, ties in input order)."""
    groups: Dict[Tuple[float, ...], List[Stance]] = {}
    for st in stances:
        groups.setdefault(tuple(round(float(v), 4) for v in st.base_pose2d), []).append(st)
    return [st for g in groups.values() for st in sorted(g, key=lambda s_: -float(s_.extras["reach_margin_m"]))[:k]]


# ----------------------------------------------------------------------------------------------------------------
# the mode
# ----------------------------------------------------------------------------------------------------------------


class InsideMode:
    """The inside mode of :class:`~ispatialgym.sampling.v2.tasks.iors_large.IORSLarge` (module docstring): the partner's interior surface
    as the goal geometry, stance C at the opening, the release above the surface and the drop test. Every other stage is IORSLarge's."""

    # ------------------------------------------------------------------ partner geometry for C

    def inside_surface_spec(self) -> Dict[str, float]:
        z = self.gp.gen("inside_surface_z_m")
        if z is None:
            raise Reject("spec", f"{self.gp.alias}: generation.inside_surface_z_m (the partner's interior surface, m above the floor) is required "
                                 "for IORS large-scale inside")
        return {"z_m": float(z), "tol_m": float(self.gp.gen("inside_surface_tol_m", SURFACE_TOL_M))}

    def prepare_partner(self):
        """The interior raster of the partner at the alias's height, the goal cells (headroom, the opening shrunk by the half extent +
        ``opening_margin_m``), the raster's sides as the stance-C edges (``inside_stance_side``) and the rim of the drop test."""
        partner = self.partner
        half, height = self.target_dims()
        spec = self.inside_surface_spec()
        # generation.partner_joint_init (the open door): its links are no part of the stand-off nor of the surface
        static = getattr(self, "partner_static_aabb", None)
        lo, hi = static() if static is not None else (G.t(v).float().clone() for v in partner.aabb)
        try:
            itop = self.world.support_interior(partner, self.floor_z + spec["z_m"], resolution=float(self.num("support_resolution", 0.02)), z_margin=spec["tol_m"],
                                               ignore=list(self.world.spawned.values()))
        except RuntimeError as e:
            raise Reject("inside_surface", f"{partner.name}: no interior surface at {spec['z_m']:.2f} m (+-{spec['tol_m']:.2f}): {e}")
        parts = getattr(self, "partner_part_boxes", None)
        n_part_cells = OJ.mask_out_boxes(itop, parts() if parts is not None else [], pad=0.5 * float(itop.resolution))
        if n_part_cells and not bool(itop.mask.any()):
            raise Reject("inside_surface", f"{partner.name}: every interior cell at {float(itop.z):.3f} m lies under its doors (generation.partner_joint_init)")
        usable = itop.usable(height, HEADROOM_MARGIN_M)
        idx = th.nonzero(usable)
        cells = th.stack([itop.xs[idx[:, 0]], itop.ys[idx[:, 1]]], dim=1) if len(idx) else th.zeros(0, 2)
        margin = half + float(self.num("opening_margin_m", OPENING_MARGIN_M))
        if self.gp.gen("inside_erosion_m") is not None:  # a wide flat target in a basin whose walls flare (the plate's rim above them)
            margin = float(self.gp.gen("inside_erosion_m"))
        goal_cells = erode_from(cells, raster_rim(itop.mask, itop.xs, itop.ys), margin)
        try:
            side = side_vector(self.gp.gen("inside_stance_side"))
        except ValueError as e:
            raise Reject("spec", f"{self.gp.alias}: {e}")
        all_edges = [EdgeFrame.from_edge(name, mid, normal, length, overhang(mid, normal, lo, hi)) for name, mid, normal, length in itop.edges()]
        edges = edges_facing(all_edges, side)
        try:
            top_z = float(self.support_top(partner).z)
        except (RuntimeError, ValueError):
            top_z = None
        rim_z, rim_src = rim_height(top_z, itop.z, height, float(hi[2]))
        if self.gp.gen("inside_rim_z_m") is not None:
            rim_z, rim_src = self.floor_z + float(self.gp.gen("inside_rim_z_m")), "generation.inside_rim_z_m"
        facts = {"z_requested_above_floor": spec["z_m"], "tol_m": spec["tol_m"], "z_world": round(float(itop.z), 4), "z_above_floor": round(float(itop.z) - self.floor_z, 4),
                 "n_board_cells": int(itop.mask.sum()), "n_free_cells": int(itop.free.sum()), "n_usable_cells": int(len(cells)), "n_goal_cells": int(len(goal_cells)),
                 "erosion_m": round(margin, 4), "open_edges": list(itop.open_edges), "board_pieces": len(itop.board_pieces), "cells_under_doors": n_part_cells,
                 "headroom_m": (itop.to_json() or {}).get("headroom_m"), "rim_z": round(rim_z, 4), "rim_source": rim_src,
                 "stance_side": None if side is None else [round(float(v), 3) for v in side], "stance_edges": [e.name for e in edges]}
        self.place = {"top": itop, "cells": goal_cells, "edges": edges, "z": float(itop.z), "height": height, "half": half, "surface": facts, "rim_z": rim_z,
                      "z_requested": spec["z_m"]}
        self.template_facts["inside_surface"] = facts
        self.template_facts["partner"] = {"name": partner.name, "n_goal_cells": int(len(goal_cells)), "edges": [e.to_json() for e in edges]}
        log.info("%s: inside surface of %s at %.3f m (requested %.2f +- %.2f above the floor): %d board / %d usable / %d goal cells (eroded %.2f m), rim %.3f (%s), "
                 "stance sides %s", self.gp.alias, partner.name, float(itop.z), spec["z_m"], spec["tol_m"], int(itop.mask.sum()), len(cells), len(goal_cells), margin,
                 rim_z, rim_src, [e.name for e in edges])
        if not edges:
            raise Reject("inside_no_side", f"{partner.name}: no side of the interior raster faces {self.gp.gen('inside_stance_side')!r} "
                                           f"(sides {[(e.name, [round(float(v), 2) for v in (-e.inward).tolist()]) for e in all_edges]})")
        if len(goal_cells) == 0:
            raise Reject("inside_no_goal_cell", f"{partner.name}: no interior cell at {float(itop.z):.3f} m fits the target ({len(cells)} with headroom >= "
                                                f"{height + HEADROOM_MARGIN_M:.2f} m, none {margin:.2f} m from the board's outline)")

    def goal_region_points(self, from_xy) -> th.Tensor:
        """Stance A's partner region: the goal-cell patch on the interior surface (as ontop's on the top) + the partner's nearest bottom
        edge (IORSLarge's)."""
        edge_pts = super().goal_region_points(from_xy)
        cells = self.place.get("cells")
        if cells is None or len(cells) == 0:
            return edge_pts
        g = nearest_cell(cells, from_xy)
        z = float(self.place["z"])
        r = float(self.num("goal_patch_half_m", 0.10))
        patch = [th.tensor([float(g[0]) + dx, float(g[1]) + dy, z]) for dx, dy in ((0, 0), (r, r), (r, -r), (-r, r), (-r, -r))]
        return th.cat([th.stack(patch), edge_pts], dim=0)

    # ------------------------------------------------------------------ the goal-side band

    def inside_band(self, arm: str) -> Optional[RB.ReachBands]:
        """The goal-side band of ``arm`` (the target on the partner's interior surface, ``surface_z`` = the alias's requested height): the
        arm's own band, else (the other arm) the default arm's mirrored, else None - no band filter, the reach margin orders the trunks.
        An empty band counts as none. Never computed in-process (a band job publishes it)."""
        memo = self.__dict__.setdefault("_inside_bands", {})
        if arm in memo:
            return memo[arm]
        c = self.choice.models["target"]
        z = float(self.place["z_requested"])
        rb, why = None, ""
        try:
            rb = RB.ReachBands.load(self.world.scene_model, self.partner.name, f"{c.category}/{c.model}", self.grasp_key, scale=c.scale, arm=arm, surface_z=z)
            if int(rb.meta.get("n_reachable_total", 1) or 0) == 0:
                rb, why = None, "empty band"
        except FileNotFoundError:
            why = "no band"
        arm0 = TT.default_arm(self)
        if rb is None and arm != arm0:
            base = self.inside_band(arm0)
            if base is not None and RB.mirrorable(base.meta):
                rb, why = base.mirrored(), f"the {arm0} arm's band mirrored"
        log.info("%s: inside goal-side band (%s arm, surface_z %.2f): %s", self.gp.alias, arm, z, rb.path if rb is not None and not why else why or "none")
        self.template_facts.setdefault("inside_bands", {})[arm] = None if rb is None else {**(TT.band_facts(rb) or {}), "n_cells": int(rb.meta.get("n_reachable_total", 0) or 0)}
        memo[arm] = rb
        return rb

    # ------------------------------------------------------------------ stance C

    def _placement_candidates_at(self, edge: EdgeFrame, init_xy, arm: Optional[str] = None) -> List[Stance]:
        """Stance C at one side of the interior raster (module docstring): stand-off x lateral x trunk, collision-free, the target at the
        goal cell in frame (without a band: then the :data:`NO_BAND_TRUNKS` per base pose with the most reach margin); each survivor
        carries ``goal_cell`` (and ``reach_margin_m`` without a band)."""
        arm = arm or self.seg_arm
        partner = self.partner
        half, height = float(self.place["half"]), float(self.place["height"])
        rb = self.inside_band(arm)
        self.place["band"] = rb
        self.place.setdefault("bands", {})[arm] = rb
        trunks = rb.trunks() if rb is not None else [(float(t), float(k)) for t in RB.TRANSLATES for k in RB.TILTS]
        half_run = edge.length / 2 - (half + 0.10)
        laterals = P0.lattice_values(-half_run, half_run, 0.10) if half_run > 0 else [0.0]
        goal_cells = self.place["cells"]
        z_mid = float(self.place["z"]) + height / 2
        lattice = float(rb.meta.get("lattice", RB.LATTICE)) if rb is not None else RB.LATTICE
        stats: Dict[str, Any] = {"goal_cells": int(len(goal_cells)), "band": rb is not None, "band_trunks": len(trunks), "lattice": 0, "base_ok": 0, "band_cells": 0,
                                 "goal_in_band": 0}
        cands: List[Stance] = []
        for so in [float(v) for v in (self.gp.gen("inside_stand_offs_m") or STAND_OFFS)]:
            for lat in laterals:
                xy = edge.base_xy(so, lat)
                stats["lattice"] += len(trunks)
                if not P0.base_ok(self, xy):
                    continue
                stats["base_ok"] += len(trunks)
                for tr, ti in trunks:
                    st = Stance("placement", (float(xy[0]), float(xy[1]), edge.yaw), tr, ti, so, edge, None, lat, None, [arm])
                    if rb is not None:
                        cells = self.band_cells_world(rb, st)
                        if len(cells) == 0:
                            continue
                        stats["band_cells"] += 1
                        pool = self._snap(cells, goal_cells, lattice)
                    else:
                        pool = goal_cells
                    if len(pool) == 0:
                        continue
                    stats["goal_in_band"] += 1
                    gc = nearest_cell(pool, xy)
                    st.extras["goal_cell"] = [float(gc[0]), float(gc[1])]
                    if rb is None:
                        st.extras["reach_margin_m"] = round(self.reach_margin(st, [float(gc[0]), float(gc[1]), z_mid], arm), 4)
                    cands.append(st)
        self.template_facts["placement_stats"] = stats
        if not cands:
            log.info("%s: inside stance C at %s side %s: no candidate (%s)", self.gp.alias, partner.name, edge.name, stats)
            return []
        free = P0.body_free(self, cands)
        cands = [st for st, f in zip(cands, free.tolist()) if f]
        stats["collision_free"] = len(cands)
        if not cands:
            log.info("%s: inside stance C at %s side %s: every candidate collides (%s)", self.gp.alias, partner.name, edge.name, stats)
            return []
        cone = self.gp.gen("inside_axis_cone_deg")
        classed = []
        for st, cam in zip(cands, P0.camera_poses(self, cands)):
            gc = st.extras["goal_cell"]
            if not P0.in_frame(self, cam, self.hull_corners_world(self.hull, self.goal_pose_at(gc, self.rest))):
                continue
            focus = th.tensor([gc[0], gc[1], z_mid])
            ang = P0.axis_angle(self, cam, focus)
            st.extras["axis_angle_deg"] = math.degrees(ang)
            if cone is not None and ang > math.radians(float(cone)):
                continue
            st.focus = focus
            st.extras["angle_class"] = 1
            classed.append((st, 1))
        stats["visible"] = len(classed)
        if rb is None:
            kept = {id(st) for st in best_per_base([st for st, _ in classed], int(self.num("inside_no_band_trunks", NO_BAND_TRUNKS)))}
            classed = [(st, k) for st, k in classed if id(st) in kept]
            stats["no_band_kept"] = len(classed)
        out = P0.best_class(classed)
        log.info("%s: inside stance C at %s side %s: %d survivors (%s)", self.gp.alias, partner.name, edge.name, len(out), stats)
        return out

    # ------------------------------------------------------------------ the release (the small-scale drop)

    def release_above(self, quat) -> float:
        """Height of the target's bottom above the interior surface at the release: ``drop_height_m``, raised until the open gripper's
        lowest point clears the surface by ``release_clearance_m`` for every library grasp (one height for every grasp: the F_goal
        screening poses the target before a grasp is drawn). Memoised per orientation."""
        quat = G.t(quat).float()
        key = tuple(round(float(v), 4) for v in quat.tolist())
        memo = self.__dict__.setdefault("_inside_release", {})
        if key not in memo:
            drop = float(self.num("drop_height_m", DROP_HEIGHT_M))
            z_s = float(self.place["z"])
            pos = th.tensor([0.0, 0.0, z_s + self.world.bottom_offset(self.hull, quat) + drop])
            lows = [float(gripper_corners_world(IK.grasp_pose((pos, quat), g.pose))[:, 2].min()) for g in self.grasps]
            memo[key] = drop + release_lift(lows, z_s, float(self.num("release_clearance_m", RELEASE_CLEARANCE_M)))
        return memo[key]

    def goal_pose_at(self, goal_cell, quat) -> G.Pose:
        """The release pose: the goal cell, the target's bottom :meth:`release_above` above the interior surface (the goal check settles it)."""
        quat = G.t(quat).float()
        z = float(self.place["z"]) + self.world.bottom_offset(self.hull, quat) + self.release_above(quat)
        return th.tensor([float(goal_cell[0]), float(goal_cell[1]), z]), quat

    # ------------------------------------------------------------------ goal check

    def refine_targets_from_goal(self, draft: Draft):
        """IORSLarge's (settled pose -> targets, partner unchanged), then the drop test's protrusion rule (``inside`` itself is the BDDL
        goal the goal check evaluates next)."""
        super().refine_targets_from_goal(draft)
        p, q = self.target.get_position_orientation()
        pts = self.world.hull_at(self.hull, (p.float(), q.float()))
        t_lo, t_hi = pts.min(dim=0).values, pts.max(dim=0).values
        height = float(t_hi[2] - t_lo[2])
        frac = float(self.num("max_protrusion_fraction", MAX_PROTRUSION_FRACTION))
        ok, prot = protrusion_ok(float(t_hi[2]), float(self.place["rim_z"]), height, frac)
        draft.summary["inside"] = {"protrusion_m": round(prot, 4), "max_protrusion_m": round(frac * height, 4), "rim_z": round(float(self.place["rim_z"]), 4),
                                   "rim_source": self.place["surface"].get("rim_source"), "settled_bottom_above_surface_m": round(float(t_lo[2]) - float(self.place["z"]), 4),
                                   "surface_z": round(float(self.place["z"]), 4)}
        if not ok:
            raise Reject("does_not_fit_inside", f"target protrudes {prot:.3f} m above the rim {float(self.place['rim_z']):.3f} (> {frac:.2f} x height {height:.3f})")

    def after_goal_check(self, draft: Draft):
        """The release as recorded (the place is the release above the surface, the goal the drop settled to): pre-place / place re-checked
        from C with the object attached and the partner present; the grasp in the manipulation image, the release in the placement image;
        the release facts in ``grasps[0]`` and ``summary.release``."""
        tname = self.tname
        grasp = self.sidecar.grasp_of(tname)
        place = self.sidecar.stance("placement")
        with attached(self.kit, grasp.arm, self.target, eef_in_object=grasp.eef_in_object):
            res = self.ikc.exact([grasp.pre_place, grasp.place], place, grasp.arm, ignore_for_last=[self.obj("floor")], cache=False)
        if not res.ok:
            raise Reject("place_unreachable", f"pre-place / release fail the exact IK from stance C at the goal check: {res.reason}")
        z_s = float(self.place["z"])
        clearance = float(self.num("release_clearance_m", RELEASE_CLEARANCE_M))
        low = float(gripper_corners_world(grasp.place)[:, 2].min())
        if low < z_s + 0.5 * clearance:
            raise Reject("release_too_low", f"open gripper lowest point {low:.3f} vs the surface {z_s:.3f} at the recorded release")
        rel = draft.goal_poses.get(tname)
        facts = {"place_kind": "drop", "surface_z": round(z_s, 4), "surface_z_above_floor": round(z_s - self.floor_z, 4),
                 "release_above_m": None if rel is None else round(self.release_above(rel[1]), 4), "finger_clearance_m": round(low - z_s, 4),
                 "drop_height_m": float(self.num("drop_height_m", DROP_HEIGHT_M)), "release_clearance_m": clearance,
                 "settled_goal_z": round(float(draft.pose_targets[tname]["position"][2]), 4)}
        grasp.extras.update(facts)
        draft.summary["release"] = facts
        if not self.ikc.eef_in_frame(grasp.grasp, self.camera_pose_at(self.sidecar.stance("manipulation"))):
            raise Reject("gripper_not_in_frame", "grasp eef pose outside the manipulation image (5 %)")
        if not self.ikc.eef_in_frame(grasp.place, self.camera_pose_at(place)):
            raise Reject("gripper_not_in_frame", "release eef pose outside the placement image (5 %)")
