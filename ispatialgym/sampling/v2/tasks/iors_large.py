"""IORS — ontop / under, large-scale (spec §5.6 / §6.12; oracle doc §6.2 row "IORS ontop / under (large-scale)").

Roles: ``target`` (spawned, library grasps), fixed ``partner`` (bed, sofa, desk, breakfast table, countertop), fixed
``support`` (the **init surface**: a support object within ``max_partner_geodesic`` of the partner, the partner's own
top — under only — or the floor beside the partner; the class is an explicit choice of the alias / bindings, never a
runtime fallback), ``floor``. Three recorded stances: **A** init (tabletop Phase 1 at the support + the partner
visible: ≥ 4 of its 8 AABB corners and its centre in frame), **B** manipulation (tabletop Phase 2 at the support, band
from the reach bands), **C** placement (base square to the partner edge nearest the init pose, stand-off {0.35, 0.45,
0.55}, lateral along the edge, trunk from the band; ontop: goal cell = the band cell inside the partner's top face
eroded by half-extent + 0.05 nearest the init pose; under: the floor strip under the partner where an upward ray hits
it ≥ target height + 0.03 above the floor, within 0.45 m of the near edge, clearance ≥ height + 0.17 (§6.12)).

B ↔ C coupling: one grasp for pick and place — (yaw, grasp) drawn uniformly from F_init ∩ F_goal (F_init = the feasible
set at B's focus cell, F_goal = the library grasps screened at the goal cell for the init yaw from C), exact IK at B
(pre-grasp + grasp, init surface ignored for the grasp pose) and at C (pre-place + place with the object attached;
ontop: the partner's top ignored for the place pose only; under: only the floor ignored). Paths A → B (§3.7 / §3.8)
and B → C (≤ ``max_partner_geodesic`` + 1 m, object attached) and the transitions are recorded; the sidecar carries
``stances.placement`` (+ ``goal_cell`` / ``init_surface`` / ``path_length_m`` in its extras and the language block),
``grasps[0]`` with ``pre_place`` / ``place`` / ``place_on``.

Reach bands (B3, ``sampling.reach_bands``): loaded through ``ReachBands.load`` for (support, target), (partner-as-
support, target) and ``--support floor [--under partner]``; on a cache miss they are computed in-process with the
module's ``BandComputer`` (the function the CLI runs) and written to the cache directory (``ISG_REACH_BANDS_COMPUTE=0``
turns the compute-on-miss into ``Reject("missing_band")``).

Tabletop phases 1a / 1b / 1c / 2 are ``v2/tabletop.py``'s (edge candidates, init trunk, lateral draw, Phase-2 survivors
with the (support, target) band). Floor init: Stance A's trunk is the translate whose optical axis is closest to the
floor cell with the target and the partner in frame (the §4.3 Δ solve has no edge on the floor; documented stand-in).

**Push-under (owner decision 2026‑09‑16 06:50, item 7; ``generation.under_mode`` = ``"push"`` default).** The floor under
a bed / sofa is out of the arm's reach (B3: 0.24–0.64 m ahead at the deep trunks only), so the under goal is reached in
two strokes from stance C (base square to the near footprint edge, facing inward = the push direction, stand-off
0.5–0.7 m, floor trunks): (1) a **drop** of the carried target on a free floor cell ``half + 0.10`` m outside the edge
(object bottom 3 cm above the floor; ``grasps[0].place`` / ``pre_place`` = the release, ``place_kind: "drop"``,
``place_on`` = the floor), (2) a **straight floor-level push** of the closed gripper from behind the target to the
**goal cell** 7 cm inside the footprint edge (the ``under`` predicate needs the centre column only; the cell must clear
target height + 3 cm) — ``contacts.push`` with ``kind: "push_under"``, ``pre`` / ``init`` (start) / ``mid`` / ``goal`` blade
poses (blade stand-off from ``tabletop.blade_standoff`` over the target's face height, approach pitched 0–60° toward
the push), ``direction``, ``distance_m``, ``push_height_m``, ``drop_cell`` / ``goal_cell``. Exact IK on the fly from C for
the drop (floor ignored, object attached) and the push chain (floor + target ignored, partner present). The program
appends ``push → retreat`` after ``place(drop) → retreat``; ``under`` is read in the epilogue; the goal-feasibility check
teleports the target to the goal cell as before.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym import REPO_ROOT
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.sim_utils import Reject, SupportTop
from ispatialgym.sampling.tasks.base import Draft, _jsonable
from ispatialgym.sampling.tro_writer import full_pose_target
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import prompt0 as P0
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, GraspRecord, Stance

log = logging.getLogger("ispatialgym.sampling.v2")

RELATION_OF_SUBTASK = {
    "inter_object_relation_satisfaction-ontop_large_scale": "ontop",
    "inter_object_relation_satisfaction-under_large_scale": "under",
}
V1_DEFINITION_ID = {"ontop": 0, "under": 1}
#: scene categories a fixed partner may bind to (B4 ``PARTNER_CATEGORIES``)
PARTNER_CATEGORIES = ("bed", "sofa", "couch", "desk", "breakfast_table", "countertop")
SUPPORT_CATEGORIES = ("breakfast_table", "desk", "coffee_table", "conference_table", "pedestal_table", "countertop", "bar", "nightstand", "console_table",
                      "side_table", "shelf", "bench")
INIT_SURFACE_CLASSES = ("support", "partner", "floor")
SCENE_INVENTORY_ROOT = os.path.join(REPO_ROOT, "runlogs", "scene_inventory_20260915")
STAND_OFFS = (0.35, 0.45, 0.55)
SUPPORT_MARGIN = 0.02
# ---- push-under (owner decision 2026‑09‑16 06:50, item 7): the under goal is reached by a drop just outside the partner's
# footprint edge and a straight floor-level push of the target under the edge ------------------------------------------------
PUSH_UNDER_INSIDE_M = 0.07          # goal = the target centre this far inside the footprint edge (5–10 cm; ``under`` needs the centre column)
PUSH_UNDER_DROP_EXTRA_M = 0.10      # drop centre = half extent along the push + this outside the edge
PUSH_UNDER_DROP_HEIGHT_M = 0.10     # object bottom above the floor at the release (≥ 2 cm; 10 cm keeps the release eef ≈ 0.1 m above the floor —
                                    # the sandal's side pinches sit 1 cm above its sole, and z < 0.08 m is out of the arm's reach even at translate 0.95)
PUSH_UNDER_STAND_OFFS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)  # base stand-off from the edge: the drop cell 0.25–0.6 m ahead (B3: the floor is reachable 0.24–0.64 m ahead)
PUSH_UNDER_DROP_AHEAD_M = (0.25, 0.60)
PUSH_UNDER_CLEARANCE_M = 0.005      # blade clearance to the target's back face at the push start
#: trunks that reach the floor (B3 floor bands, A6 recompute: cells only at translate ≥ 0.75, tilt ≤ −0.9)
FLOOR_TRUNKS = ((0.75, -0.9), (0.75, -1.2), (0.95, -0.9), (0.95, -1.2))


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class PushUnderRecord(ContactRecord):
    """``contacts.push`` of the push-under (F6 ``PushContact`` + extras): ``kind: "push_under"``, straight floor-level
    push ``init → mid → goal`` of the closed gripper (blade behind the target along the push direction), from the
    placement stance; ``push_height_m``, ``pitch_deg``, ``blade_standoff_m``, ``half_extent_m`` (target half extent along
    the push), ``drop_cell`` / ``goal_cell`` (target centres, world xy)."""

    def __init__(self, arm: str, pre, start, mid, goal, direction, distance_m: float, object: str, stance: str = "placement", **extras):
        super().__init__("push", arm, pre, start, mid, goal, direction, distance_m, None, object, stance)
        self.extras: Dict[str, Any] = dict(extras)

    def to_sidecar(self) -> Dict[str, Any]:
        d = super().to_sidecar()
        d["kind"] = "push_under"
        for k, v in self.extras.items():
            if k not in d:
                d[k] = _jsonable(v)
        return d
TARGET_JITTER = math.radians(22.5)
FOCUS_RADIUS = 0.05
YAWS = [k * math.pi / 4 for k in range(8)]


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (tested without a simulator)
# ----------------------------------------------------------------------------------------------------------------


#: §5.6 / owner decision A5 (2026‑09‑16): the alias-level selection order of the init surface, applied by ``ground_isg``
#: through fallback aliases (primary → fallback when the primary yields nothing); a module never switches class at run time
SELECTION_ORDER = {"under": ("support", "partner", "floor"), "ontop": ("support", "floor")}


def init_surface_class(relation: str, support_binding: Any, partner_names: Sequence[str], alias_support_token: Optional[str] = None,
                       alias_partner_token: Optional[str] = None) -> str:
    """§5.6 init-surface class fixed by the alias (A5: no runtime switching): ``"floor"`` (the binding / alias token
    says floor, or the binding is a room dict), ``"partner"`` (the support binding names the partner, or the alias
    repeats the partner token ``…-<partner>-<partner>``; under only) or ``"support"`` (a support object). ``ValueError``
    for the ontop + partner combination (the goal would hold at init)."""
    names = [n for n in support_binding if isinstance(n, str)] if isinstance(support_binding, list) else []
    token = (alias_support_token or "").lower()
    ptoken = (alias_partner_token or "").lower()
    if names == ["floor"] or token in ("floor", "floors") or (isinstance(support_binding, dict) and (support_binding.get("floor") or support_binding.get("room"))):
        return "floor"
    own_top = (names and all(n in set(partner_names) for n in names)) or (not names and token and ptoken and token == ptoken)
    if own_top:
        if relation == "ontop":
            raise ValueError("ontop with the partner's own top as the init surface: the goal would hold at init")
        return "partner"
    if not names and not isinstance(support_binding, dict) and token and token in {c for c in PARTNER_CATEGORIES} and relation == "under":
        return "partner"
    return "support"


def alias_tokens(alias: str, subtask: str) -> List[str]:
    """The ``{target}-{partner}-{support}`` tokens of ``<subtask>-{target}-{partner}-{support}`` ([] without tokens)."""
    rest = alias[len(subtask):].strip("-") if alias.startswith(subtask) else ""
    return [p for p in rest.split("-") if p]


def alias_support_token(alias: str, subtask: str) -> Optional[str]:
    """The ``{support}`` token of ``<subtask>-{target}-{partner}-{support}`` (None when the alias has no tokens)."""
    parts = alias_tokens(alias, subtask)
    return parts[-1] if len(parts) >= 3 else None


def alias_partner_token(alias: str, subtask: str) -> Optional[str]:
    """The ``{partner}`` token (None when the alias has fewer than three tokens)."""
    parts = alias_tokens(alias, subtask)
    return parts[-2] if len(parts) >= 3 else None


def nearest_edge(edges: Sequence[EdgeFrame], xy) -> EdgeFrame:
    """The partner edge whose midpoint is nearest to a world xy (§5.6: the edge nearest the target's init pose)."""
    p = th.as_tensor([float(xy[0]), float(xy[1])])
    return min(edges, key=lambda e: float(th.linalg.norm(e.mid - p)))


def aabb_edges(lo, hi, overhang_from_top: Optional[Dict[str, float]] = None) -> List[EdgeFrame]:
    """The four AABB edges of a partner as EdgeFrames (inward normals), named ``+x / -x / +y / -y``."""
    lo, hi = th.as_tensor(lo, dtype=th.float32)[:2], th.as_tensor(hi, dtype=th.float32)[:2]
    cx, cy = float(lo[0] + hi[0]) / 2, float(lo[1] + hi[1]) / 2
    out = []
    for name, mid, outward, length in (("+x", (float(hi[0]), cy), (1.0, 0.0), float(hi[1] - lo[1])), ("-x", (float(lo[0]), cy), (-1.0, 0.0), float(hi[1] - lo[1])),
                                       ("+y", (cx, float(hi[1])), (0.0, 1.0), float(hi[0] - lo[0])), ("-y", (cx, float(lo[1])), (0.0, -1.0), float(hi[0] - lo[0]))):
        out.append(EdgeFrame.from_edge(name, th.tensor(mid), th.tensor(outward), length, float((overhang_from_top or {}).get(name, 0.0))))
    return out


def edge_overhang(top_edge_mid, outward, aabb_lo, aabb_hi) -> float:
    """AABB overhang beyond a top-raster edge along its outward normal (§4.2 / §5.6: ≥ 5 cm counts, else 0)."""
    corners = G.aabb_corners(th.as_tensor(aabb_lo, dtype=th.float32), th.as_tensor(aabb_hi, dtype=th.float32))[:, :2]
    n = th.as_tensor(outward, dtype=th.float32)[:2]
    over = float((corners @ n).max()) - float(th.dot(th.as_tensor(top_edge_mid, dtype=th.float32)[:2], n))
    return over if over >= 0.05 else 0.0


def erode_cells(cells: th.Tensor, boundary_xy: th.Tensor, margin: float) -> th.Tensor:
    """Cells (N, 2) farther than ``margin`` from every point of a boundary polyline / raster (the top eroded by the
    target's half-extent + 0.05)."""
    if len(cells) == 0 or len(boundary_xy) == 0:
        return cells
    d = th.cdist(cells.float(), boundary_xy.float())
    return cells[d.min(dim=1).values > margin]


def under_clearance_cells(grid: Dict[str, Any], min_clearance: float) -> th.Tensor:
    """World xy of the B4 under-clearance raster cells (``partners/<name>.json`` ``under_clearance``) whose clearance
    ≥ ``min_clearance``."""
    res = float(grid["resolution"])
    ox, oy = (float(v) for v in grid["origin"])
    g = grid["grid"]
    out = []
    for i, row in enumerate(g):
        for j, c in enumerate(row):
            if c is not None and float(c) >= min_clearance:
                out.append((ox + (i + 0.5) * res, oy + (j + 0.5) * res))
    return th.tensor(out, dtype=th.float32) if out else th.zeros(0, 2)


def under_admissible(clearance_at_goal: float, target_height: float, extra: float = 0.17) -> bool:
    """§6.12 under admissibility: clearance ≥ target height + 0.12 (gripper below the wrist) + 0.05."""
    return float(clearance_at_goal) >= float(target_height) + float(extra)


def intersect_feasible(F_init: Sequence[Tuple[float, Any]], F_goal: Sequence[Tuple[float, Any]]) -> List[Tuple[float, Any]]:
    """F_init ∩ F_goal over (yaw, grasp) pairs (grasps compared by name)."""
    key = lambda yg: (round(float(yg[0]), 6), getattr(yg[1], "name", yg[1]))  # noqa: E731
    goal_keys = {key(yg) for yg in F_goal}
    return [yg for yg in F_init if key(yg) in goal_keys]


# ----------------------------------------------------------------------------------------------------------------
# the task module
# ----------------------------------------------------------------------------------------------------------------


@register(*RELATION_OF_SUBTASK)
class IORSLarge(SamplerV2):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.relation = RELATION_OF_SUBTASK[self.subtask]
        if self.gp.bddl is None:
            self.definition_id = V1_DEFINITION_ID[self.relation]
        self.hull: Optional[th.Tensor] = None
        self.rest: th.Tensor = G.IDENTITY_QUAT.clone()
        self.bands: Optional[RB.ReachBands] = None  # ``tabletop.ensure_bands`` caches the init-support band here
        self.band_spec: Optional[TT.BandSpec] = None  # what ``ensure_bands`` computes on a miss (set once the grasps are loaded)
        self.grasps: List[Any] = []
        self.grasp_key: Optional[str] = None
        self.surface: Dict[str, Any] = {}  # kind, obj, top, band, edge, init, manip candidates
        self.place: Dict[str, Any] = {}  # partner geometry for C: top / under cells, edges, band
        self.floor_z = 0.0

    # ------------------------------------------------------------------ roles

    def bind_scene_roles(self):
        self.bind_floor_v2()
        req = self.choice.bindings.get("partner")
        names = [n for n in (req if isinstance(req, list) else []) if isinstance(n, str)]
        if names:
            partner = self.world.obj(names[0])
        else:
            cats = list((req or {}).get("category", PARTNER_CATEGORIES)) if isinstance(req, dict) or req is None else list(PARTNER_CATEGORIES)
            rooms = set(self.world.rooms or [])
            cands = sorted([o for o in self.world.scene.objects if o.category in cats and (not rooms or set(o.in_rooms or []) & rooms)], key=lambda o: o.name)
            if not cands:
                raise Reject("spec", f"{self.gp.alias}: no partner candidate of {cats} in {sorted(rooms)}")
            partner = cands[0]
        self.bind_scene_object("partner", partner)

    @property
    def partner(self):
        return self.obj("partner")

    @property
    def target(self):
        return self.obj("target")

    @property
    def tname(self) -> str:
        return self.bindings.one("target").name

    @property
    def pname(self) -> str:
        return self.bindings.one("partner").name

    def support_objects(self) -> List[Any]:
        """The init surface when it is a piece of furniture (parking keep-away, grasp ignore); the floor is handled by
        :meth:`grasp_ignore_objects` (its AABB would forbid every parking cell)."""
        out = []
        if self.surface.get("obj") is not None:
            out.append(self.surface["obj"])
        return out

    def grasp_ignore_objects(self) -> List[Any]:
        """§5.6: the init surface ignored for the grasp pose only (the floor for the floor init)."""
        return [self.obj("floor")] if self.surface.get("kind") == "floor" else self.support_objects()

    # duck-typed attributes the shared tabletop phases read (``tabletop.py`` / the TEMP stand-in of ``iors_small``)
    @property
    def support(self):
        return self.surface.get("obj")

    @property
    def top(self) -> Optional[SupportTop]:
        return self.surface.get("top")

    def num(self, key: str, default: float) -> float:
        v = self.gp.gen(key)
        if v is None and key.endswith("_m"):
            v = self.gp.gen(key[:-2])  # v1 spelling (max_partner_geodesic)
        return float(default if v is None else v)

    # ------------------------------------------------------------------ init surface (§5.6)

    def surface_candidates(self) -> Tuple[str, List[Any]]:
        b = self.choice.bindings
        req = b.get("support")
        try:
            kind = init_surface_class(self.relation, req, [self.partner.name], alias_support_token(self.gp.alias, self.subtask),
                                      alias_partner_token(self.gp.alias, self.subtask))
        except ValueError as e:
            raise Reject("spec", str(e))
        self.template_facts["init_surface_selection_order"] = list(SELECTION_ORDER.get(self.relation, ()))
        if kind == "floor":
            return kind, [self.obj("floor")]
        if kind == "partner":
            return kind, [self.partner]
        names = [n for n in req if isinstance(n, str)] if isinstance(req, list) else []
        if names:
            return kind, [self.world.obj(n) for n in names]
        cats = list((req or {}).get("category", SUPPORT_CATEGORIES)) if isinstance(req, dict) or req is None else list(SUPPORT_CATEGORIES)
        rooms = set(self.world.rooms or [])
        cands = [o for o in self.world.scene.objects if o.category in cats and o.name in self.world.scene_object_names and (not rooms or set(o.in_rooms or []) & rooms)]
        cands = [o for o in cands if o is not self.partner]
        max_geo = self.num("max_partner_geodesic_m", 3.0)
        out = []
        for o in sorted(cands, key=lambda o: o.name):
            dist = self.geodesic_between(o, self.partner)
            if dist is not None and dist <= max_geo:
                out.append(o)
        return kind, out

    def near_side_point(self, obj, toward_xy, off: float = 0.5) -> th.Tensor:
        """A point 0.5 m outside the AABB side of ``obj`` that faces ``toward_xy`` (geodesic anchor)."""
        lo, hi = obj.aabb
        p = th.as_tensor([float(toward_xy[0]), float(toward_xy[1])])
        near = th.minimum(th.maximum(p, lo[:2].float()), hi[:2].float())
        d = p - near
        n = float(th.linalg.norm(d))
        return near + (d / n if n > 1e-6 else th.tensor([1.0, 0.0])) * off

    def geodesic_between(self, a, b) -> Optional[float]:
        ca, cb = a.aabb_center[:2].float(), b.aabb_center[:2].float()
        pa, pb = self.near_side_point(a, cb), self.near_side_point(b, ca)
        try:
            _, dist = self.kit.geodesic(pa, pb)
        except Exception:  # noqa: BLE001
            return None
        return dist

    def prepare_bindings(self):
        try:
            self._prepare_bindings()
        finally:  # the facts survive a template rejection (diagnostics in template_meta)
            self.template_meta["template_facts"] = _jsonable(self.template_facts)

    def _prepare_bindings(self):
        """Target geometry and grasps, the partner's placement geometry, then the init surface: one candidate of the
        grounded class drawn uniformly, the rest of the class in random order; the support role bound (aliased to the
        partner / floor bound when it is the same object)."""
        tgt = self.target
        self.hull = self.world.local_hull(tgt)
        self.grasps, self.grasp_key, self.rest = TT.library_grasps(self, "target")
        self.band_spec = self.make_band_spec()
        self.floor_z = float(self.obj("floor").aabb[1][2])
        self.prepare_partner()
        kind, cands = self.surface_candidates()
        if not cands:
            raise Reject("no_init_surface", f"{self.gp.alias}: no candidate of the init-surface class {kind!r}")
        rng = np.random.default_rng(self.ctx.base_seed)
        order = [cands[int(i)] for i in rng.permutation(len(cands))]
        errors = []
        for cand in order:
            try:
                self.try_surface(kind, cand, rng)
            except Reject as e:
                errors.append(f"{cand.name}: {e.reason}: {e}")
                continue
            if kind == "floor":
                self.alias_role("support", self.bindings.one("floor"))
            elif kind == "partner":
                self.alias_role("support", self.bindings.one("partner"))
            else:
                self.bind_scene_object("support", cand)
            self.template_facts["init_surface"] = {"class": kind, "name": cand.name, "tried": len(errors) + 1}
            return
        raise Reject("no_init_surface", f"{self.gp.alias}: no {kind} candidate admits stances A–C: " + "; ".join(errors[-3:]))

    # ------------------------------------------------------------------ partner geometry for C

    def partner_record(self) -> Optional[Dict[str, Any]]:
        p = os.path.join(SCENE_INVENTORY_ROOT, self.world.scene_model, "partners", f"{self.partner.name}.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    def target_dims(self) -> Tuple[float, float]:
        pts = G.local_points_world(self.hull, (th.zeros(3), self.rest))
        ext = pts.max(dim=0).values - pts.min(dim=0).values
        return float(ext[:2].max()) / 2, float(ext[2])

    def prepare_partner(self):
        partner = self.partner
        half, height = self.target_dims()
        margin = self.num("goal_cell_margin_m", 0.05) + half
        lo, hi = partner.aabb
        if self.relation == "ontop":
            top = self.support_top(partner)
            if int(top.free.sum()) < 50:
                raise Reject("partner_top_small", f"{partner.name}: too little free top")
            cells = top.cells(free_only=True)
            # boundary = raster cells with a missing neighbour (the face outline) — eroded by half-extent + 0.05
            rim = self._raster_rim(top)
            goal_cells = erode_cells(cells, rim, margin)
            edges = []
            for name, mid, normal, length in top.edges():
                edges.append(EdgeFrame.from_edge(name, mid, normal, length, edge_overhang(mid, normal, lo, hi)))
            self.place = {"top": top, "cells": goal_cells, "edges": edges, "z": top.z, "height": height, "half": half}
            if len(goal_cells) == 0:
                raise Reject("partner_too_small", f"{partner.name}: no top cell survives the erosion by {margin:.2f} m")
        else:
            rec = self.partner_record()
            need = height + self.num("under_ray_extra_m", 0.03)
            if rec is not None and rec.get("under_clearance"):
                cells = under_clearance_cells(rec["under_clearance"], need)
                clear_map = rec["under_clearance"]
            else:
                cells, clear_map = self._under_raycast(partner, need), None
            edges = aabb_edges(lo, hi)
            if self.push_under:
                # push-under: goal cells sit at the footprint edge (centre 5–10 cm inside; the target's outer part may stick
                # out) — keep the clear cells within 0.12 m inside any AABB edge
                if len(cells):
                    depth = th.stack([th.minimum(cells[:, 0] - lo[0].float(), hi[0].float() - cells[:, 0]),
                                      th.minimum(cells[:, 1] - lo[1].float(), hi[1].float() - cells[:, 1])], dim=1).min(dim=1).values
                    cells = cells[(depth >= 0.0) & (depth <= PUSH_UNDER_INSIDE_M + 0.05)]
                self.place = {"cells": cells, "edges": edges, "z": self.floor_z, "height": height, "half": half, "clearance": clear_map, "need": need}
                if len(cells) == 0:
                    raise Reject("under_clearance", f"{partner.name}: no floor cell within {PUSH_UNDER_INSIDE_M + 0.05:.2f} m of its footprint edge clears {need:.2f} m")
            else:
                # inside the footprint shrunk by half-extent + 0.05 and within 0.45 m of a near AABB edge
                fp_lo, fp_hi = lo[:2].float() + margin, hi[:2].float() - margin
                if len(cells):
                    inside = ((cells >= fp_lo) & (cells <= fp_hi)).all(dim=1)
                    cells = cells[inside]
                self.place = {"cells": cells, "edges": edges, "z": self.floor_z, "height": height, "half": half, "clearance": clear_map, "need": need}
                if len(cells) == 0:
                    raise Reject("under_clearance", f"{partner.name}: no floor cell under it clears {need:.2f} m inside the footprint eroded by {margin:.2f} m")
        self.template_facts["partner"] = {"name": partner.name, "n_goal_cells": int(len(self.place["cells"])), "edges": [e.to_json() for e in self.place["edges"]]}

    def _raster_rim(self, top: SupportTop) -> th.Tensor:
        m = top.mask
        pad = th.nn.functional.pad(m.float()[None, None], (1, 1, 1, 1))[0, 0] > 0.5
        interior = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
        rim = m & ~interior
        idx = th.nonzero(rim)
        return th.stack([top.xs[idx[:, 0]], top.ys[idx[:, 1]]], dim=1) if len(idx) else th.zeros(0, 2)

    def _under_raycast(self, partner, need: float, res: float = 0.05) -> th.Tensor:
        lo, hi = partner.aabb
        xs = th.arange(float(lo[0]) + res / 2, float(hi[0]), res)
        ys = th.arange(float(lo[1]) + res / 2, float(hi[1]), res)
        gx, gy = th.meshgrid(xs, ys, indexing="ij")
        pts = th.stack([gx.flatten(), gy.flatten()], dim=1)
        starts = th.cat([pts, th.full((len(pts), 1), self.floor_z + 0.005)], dim=1)
        ends = starts.clone()
        ends[:, 2] = float(hi[2]) + 0.5
        hits = self.world.raycast(starts, ends)
        prims = {l.prim_path for l in partner.links.values()}
        keep = [k for k, h in enumerate(hits) if h["hit"] and h["rigidBody"] in prims and float(h["position"][2]) - self.floor_z >= need]
        return pts[keep] if keep else th.zeros(0, 2)

    # ------------------------------------------------------------------ push-under helpers (owner decision 2026‑09‑16 item 7)

    @property
    def push_under(self) -> bool:
        """The under goal is reached by drop + straight floor push (``generation.under_mode`` = ``"push"`` default)."""
        return self.relation == "under" and str(self.gp.gen("under_mode", "push")) == "push"

    #: ground-like scene categories whose AABB spans rooms (the garden lawn's AABB covers the child's room floor strip)
    GROUND_CATEGORIES = ("floors", "ceilings", "walls", "lawn", "driveway", "roof", "pool", "carpet", "rug", "mat", "doormat", "fence", "rail_fence")
    GROUND_FOOTPRINT_M2 = 15.0

    def floor_blocking_objects(self) -> List[Any]:
        """Scene objects whose AABB blocks a floor cell for the target / the drop: not ground-like (categories above, or a
        footprint > 15 m²), not the robot, not spawned, ≥ 5 cm tall (flat floor coverings are part of the floor)."""
        out = []
        for o in self.world.scene.objects:
            if o.category in self.GROUND_CATEGORIES or o is self.robot or o.name in self.world.spawned:
                continue
            lo, hi = o.aabb
            if float(hi[2] - lo[2]) < 0.05:
                continue
            if float((hi[0] - lo[0]) * (hi[1] - lo[1])) > self.GROUND_FOOTPRINT_M2:
                continue
            out.append(o)
        return out

    def _free_floor_cell(self, xy, why: Optional[Dict[str, int]] = None) -> bool:
        """Free floor for the drop: un-eroded traversable, in the task rooms, ≥ 2 cm outside every object AABB (flat
        floor coverings < 5 cm ignored; the AABB list is built once per template). ``why`` counts the failing filter."""
        p = th.as_tensor([float(xy[0]), float(xy[1])])
        if not self.kit.trav_free(p):
            if why is not None:
                why["not_traversable"] = why.get("not_traversable", 0) + 1
            return False
        if not P0.in_task_rooms(self, p):
            if why is not None:
                why["outside_rooms"] = why.get("outside_rooms", 0) + 1
            return False
        aabbs = self.__dict__.get("_floor_aabbs")
        if aabbs is None:
            aabbs = [(o.name, o.aabb[0][:2].float() - 0.02, o.aabb[1][:2].float() + 0.02) for o in self.floor_blocking_objects()]
            self._floor_aabbs = aabbs
        hit = [n for n, a, b in aabbs if bool((p >= a).all() and (p <= b).all())]
        if hit:
            if why is not None:
                why[f"aabb:{hit[0]}"] = why.get(f"aabb:{hit[0]}", 0) + 1
            return False
        return True

    def drop_pose_at(self, xy, quat) -> G.Pose:
        """Release pose of the push-under drop: the target's bottom ``PUSH_UNDER_DROP_HEIGHT_M`` above the floor at ``xy``."""
        z = self.floor_z + self.world.bottom_offset(self.hull, quat) + float(self.num("drop_height_m", PUSH_UNDER_DROP_HEIGHT_M))
        return th.tensor([float(xy[0]), float(xy[1]), z]), quat

    def half_extent_along(self, quat, direction_xy) -> float:
        """Half extent of the target's footprint along a horizontal direction at orientation ``quat``."""
        pts = G.local_points_world(self.hull, (th.zeros(3), quat))[:, :2]
        d = th.as_tensor([float(direction_xy[0]), float(direction_xy[1])])
        d = d / th.linalg.norm(d)
        proj = pts @ d
        return float(proj.max() - proj.min()) / 2

    def push_height(self, pitch_deg: float) -> Tuple[float, float]:
        """``(z_push, z_tip)`` of the floor blade: the eef origin at ¾ of the target height, clamped to 6–9 cm above the
        floor (``push_height_min_m``): the fingertips then sit ≥ 4 cm above the floor — the cuRobo ground slab (which the
        floor-ignore does not remove) flags fingertip spheres closer than that (first-demo loop 2026‑09‑16: every push
        pre-pose failed the IK at z_push = 4.5 cm while the drop grasp 11 cm up was feasible from the same stance)."""
        height = float(self.place["height"])
        z_min = float(self.num("push_height_min_m", 0.06))
        z_push = self.floor_z + min(max(0.75 * height, z_min), 0.09)
        drop = C.FINGERTIP_LENGTH * math.cos(math.radians(float(pitch_deg)))
        z_push = max(z_push, self.floor_z + 0.04 + drop)
        return z_push, z_push - drop

    def _push_under_chain(self, place: Stance, quat, drop_xy, goal_xy, d2) -> Optional[Dict[str, Any]]:
        """The straight floor-level push from the drop position to the goal cell, exact IK from stance C (closed gripper,
        floor and target ignored, partner present): blade behind the target along the push direction (stand-off from
        ``tabletop.blade_standoff`` over the target's face height), approach pitched toward the push direction (0 / 30 /
        45 / 60°, either wrist turn), poses ``[pre, start, mid, goal]``. Returns the record fields or None."""
        floor, target = self.obj("floor"), self.target
        d2 = th.as_tensor([float(d2[0]), float(d2[1])])
        d2 = d2 / th.linalg.norm(d2)
        d3 = th.tensor([float(d2[0]), float(d2[1]), 0.0])
        half_d = self.half_extent_along(quat, d2)
        height = float(self.place["height"])
        cd = th.as_tensor([float(drop_xy[0]), float(drop_xy[1])])
        gc = th.as_tensor([float(goal_xy[0]), float(goal_xy[1])])
        distance = float(th.linalg.norm(gc - cd))
        why: List[str] = []
        for pitch in TT.approach_pitch_order(None):
            z_push, z_tip = self.push_height(pitch)
            so = TT.blade_standoff((self.floor_z, self.floor_z + height), z_tip, pitch, PUSH_UNDER_CLEARANCE_M)
            if so is None:
                why.append(f"pitch {pitch:.0f}: face reaches the palm")
                continue
            standoff = so[0]
            a_vec = TT.pitched_approach(pitch, d2)  # pitched toward the push direction (= the robot's forward for a frontal stance, its right for a side stance)
            start = th.tensor([float(cd[0]), float(cd[1]), z_push]) - d3 * (half_d + standoff)
            goal = th.tensor([float(gc[0]), float(gc[1]), z_push]) - d3 * (half_d + standoff)
            mid = 0.5 * (start + goal)
            for c_sign in (1.0, -1.0):
                q = C.eef_quat(a_vec, c_sign * d3)
                poses = [(start, q), (mid, q), (goal, q)]
                pre = C.pre_pose(poses[0])
                with self.ikc.ignoring([floor, target]):
                    res = self.ikc.exact([pre] + poses, place, "right", cache=False, fingers=C.FINGER_CLOSED)
                if res.ok:
                    return {"pre": pre, "start": poses[0], "mid": poses[1], "goal": poses[2], "direction": d3, "distance_m": distance, "pitch_deg": float(pitch),
                            "blade_standoff_m": float(standoff), "push_height_m": float(z_push - self.floor_z), "half_extent_m": half_d, "wrist_sign": c_sign,
                            "fingertip_z_m": float(z_tip - self.floor_z)}
                why.append(f"pitch {pitch:.0f}{'' if c_sign > 0 else '-'}: {res.reason}")
        log.info("%s: push-under chain unreachable from stance C (d %.2f λ %+.2f trunk %s; start %s, goal %s in base): %s", self.gp.alias, place.stand_off or 0.0,
                 place.lateral, (place.translate, round(place.tilt, 2)),
                 [round(float(v), 2) for v in P0.to_base_frame(th.tensor([float(cd[0]), float(cd[1]), 0.0]) - d3 * half_d, place.base_pose2d, P0.base_z(self)).tolist()],
                 [round(float(v), 2) for v in P0.to_base_frame(th.tensor([float(gc[0]), float(gc[1]), 0.0]) - d3 * half_d, place.base_pose2d, P0.base_z(self)).tolist()],
                 "; ".join(why))
        return None

    def clearance_at(self, xy) -> Optional[float]:
        cm = self.place.get("clearance")
        if not cm:
            return None
        res, (ox, oy) = float(cm["resolution"]), (float(cm["origin"][0]), float(cm["origin"][1]))
        i, j = int((float(xy[0]) - ox) / res), int((float(xy[1]) - oy) / res)
        g = cm["grid"]
        if 0 <= i < len(g) and 0 <= j < len(g[i]) and g[i][j] is not None:
            return float(g[i][j])
        return None

    # ------------------------------------------------------------------ reach bands (B3)

    def make_band_spec(self) -> TT.BandSpec:
        c = self.choice.models["target"]
        return TT.BandSpec(object=f"{c.category}/{c.model}", grasp_key=self.grasp_key, grasps=self.grasps, hull=self.hull, rest_quat=self.rest,
                           scale=list(c.scale) if c.scale else None, height=self.target_dims()[1])

    def band(self, kind: str, support_obj=None, under_partner=None) -> Optional[RB.ReachBands]:
        """The reach band of a factor set (§5.6): tabletop mode (init support / the partner's top) through
        ``tabletop.ensure_bands`` / ``compute_bands`` (``ReachBands.load`` on a hit, ``BandComputer`` on a miss), the
        floor modes (``--support floor [--under partner]``) through :meth:`compute_band` on a miss; memoised per
        (kind, support, under) for the template. ``ISG_REACH_BANDS_COMPUTE=0`` turns a miss into ``Reject("missing_band")``."""
        c = self.choice.models["target"]
        obj = f"{c.category}/{c.model}"
        support = "floor" if kind == "floor" else support_obj.name
        under = under_partner.name if under_partner is not None else None
        memo = self.__dict__.setdefault("_bands", {})
        key = (kind, support, under)
        if key in memo:
            return memo[key]
        try:
            rb = RB.ReachBands.load(self.world.scene_model, support, obj, self.grasp_key, under=under, scale=c.scale)
            log.info("%s: reach band %s (%s) from %s", self.gp.alias, support, under or "-", rb.path)
            # a band computed before the A6 posture retry can be empty (every trunk stance collided with the untucked
            # arms: the fleet's floor-under bands of the beds) — an empty band is a miss, recomputed with the current script
            if int(rb.meta.get("n_reachable_total", 1) or 0) == 0 and os.environ.get("ISG_REACH_BANDS_COMPUTE", "1") != "0":
                log.info("%s: cached reach band %s (%s) is empty (computed with %s arms, no A6 retry): recomputing in-process", self.gp.alias, support, under or "-",
                         rb.meta.get("arms"))
                raise FileNotFoundError("empty band")
        except FileNotFoundError:
            if os.environ.get("ISG_REACH_BANDS_COMPUTE", "1") == "0":
                raise Reject("missing_band", f"no reach band for scene={self.world.scene_model} support={support} object={obj} under={under} scale={c.scale}")
            if kind == "tabletop":
                top = self.surface.get("top") if (self.surface.get("obj") is support_obj and self.surface.get("top") is not None) else self.support_top(support_obj, ignore=[self.target])
                rb = TT.compute_bands(self, self.make_band_spec(), top)
            else:
                rb = self.compute_band(kind, support_obj, under_partner)
        if rb is not None and int(rb.meta.get("n_reachable_total", 1) or 0) == 0:
            # still empty after the recompute (the band stance itself collides at every trunk — e.g. the band script's
            # under stance at a blocked AABB edge of the partner): the stance search runs without the band filter and the
            # exact IK of the coupling decides (recorded in the template facts; the band owner gets the case)
            log.warning("%s: reach band %s (%s) is empty after the recompute: stance C candidates are taken from the goal cells without the band filter",
                        self.gp.alias, support, under or "-")
            self.template_facts.setdefault("empty_bands", []).append({"support": support, "under": under, "stance": rb.meta.get("stance")})
            rb = None
        memo[key] = rb
        return rb

    def compute_band(self, kind: str, support_obj=None, under_partner=None) -> RB.ReachBands:
        """Compute-on-miss of the floor-mode bands (``--support floor [--under partner]``) with the computation the CLI
        runs (the ``stance_free_floor`` / ``stance_under`` stance, the 2 cm floor lattice (under: the clear cells), the
        library grasp mask per trunk incl. tilt −1.2); written to the cache root in the CLI's layout."""
        c = self.choice.models["target"]
        bc = RB.BandComputer(self.world, self.kit, arms=("right",))
        t0 = time.time()
        parked = self.world.snapshot()
        try:
            self.world.park(self.target, list(self.world.spawned).index(self.target.name))
            self.kit.sync_world()
            if kind == "floor":
                z = self.floor_z
                if under_partner is not None:
                    stance = bc.stance_under(under_partner, RB.STAND_OFF)
                    lattice_world = self.place["cells"] if self.relation == "under" and len(self.place.get("cells", [])) else self._under_raycast(under_partner, 0.0, 0.02)
                    ignore = [self.obj("floor")]
                else:
                    room = (self.gp.rooms(self.scene) or self.world.rooms or [None])[0]
                    stance = bc.stance_free_floor(room)
                    origin, fwd, left = bc.base_frame(stance)
                    xs = th.arange(0.2, RB.STRIP_MAX_DEPTH + 0.5, 0.02)
                    ls = th.arange(-RB.STRIP_HALF_WIDTH, RB.STRIP_HALF_WIDTH + 1e-6, 0.02)
                    gx, gl = th.meshgrid(xs, ls, indexing="ij")
                    lattice_world = origin + gx.flatten().unsqueeze(1) * fwd + gl.flatten().unsqueeze(1) * left
                    lattice_world = th.stack([cell for cell in lattice_world if self.kit.trav_free(cell)]) if len(lattice_world) else lattice_world
                    ignore = [self.obj("floor")]
                top_json = {"z": z, "kind": "floor"}
            else:
                raise ValueError("compute_band handles the floor modes; tabletop bands go through tabletop.compute_bands")
            hull, rest = self.hull, self.rest
            max_off = max(float(th.linalg.norm(g.position)) for g in self.grasps) if self.grasps else 0.0

            def pose_fn(world_xy, yaw, _h=hull, _r=rest, _z=z):
                return bc.object_pose_at(world_xy, yaw, _z, _h, _r)

            trunks = RB.trunk_grid(z - self.floor_z)
            result = bc.compute(stance, lattice_world, z, pose_fn, self.grasps, ignore, arm="right", trunks=trunks, max_grasp_offset=max_off)
        finally:
            self.world.restore(parked)
            self.kit.sync_world()
        meta = {"scene": self.world.scene_model, "rooms": list(self.world.rooms or []), "support": "floor" if kind == "floor" else support_obj.name, "lattice": bc.lattice,
                "stand_off": RB.STAND_OFF, "arms": "untucked", "grid": {"translates": RB.TRANSLATES, "tilts": RB.TILTS, "low_top_tilt": RB.LOW_TOP_TILT}, "seeds": 12,
                "pre_grasp": RB.PRE_GRASP, "yaws": RB.YAWS, "mode": "floor" if kind == "floor" else "tabletop", "scale": c.scale, "placed_specs": [],
                "object": f"{c.category}/{c.model}", "grasp_key": self.grasp_key, "n_grasps": len(self.grasps), "grasp_names": [g.name for g in self.grasps],
                "rest_quat_xyzw": self.rest.tolist(), "generic_pinch": False, "stance": stance, "top": top_json, "z_support": z,
                "under_partner": under_partner.name if under_partner is not None else None, "trunks": result["trunks"], "computed_by": "iors_large compute-on-miss",
                "time_s": round(time.time() - t0, 1)}
        lat_all = np.concatenate([cc[:, 1] for cc in result["cells"].values() if len(cc)]) if any(len(cc) for cc in result["cells"].values()) else np.zeros(0)
        meta["union_extent"] = [float(lat_all.min()), float(lat_all.max())] if len(lat_all) else None
        meta["n_reachable_total"] = int(sum(len(cc) for cc in result["cells"].values()))
        cells = {k: np.asarray(v, dtype=np.float32).reshape(-1, 2) for k, v in result["cells"].items()}
        pairs = {k: np.asarray(v, dtype=np.int32).reshape(-1, 3) for k, v in result.get("pairs", {}).items()}
        rb = RB.ReachBands(meta, cells, pairs, None)
        try:
            out_dir = RB.cache_root()
            os.makedirs(out_dir, exist_ok=True)
            key = RB._key({"scene": meta["scene"], "support": meta["support"], "z": round(z, 2), "object": meta["object"], "scale": c.scale, "under": meta["under_partner"],
                           "grasp_key": self.grasp_key, "computed_by": "iors_large"})
            name = f"{meta['mode']}__{meta['scene']}__{meta['support']}{'-under-' + meta['under_partner'] if meta['under_partner'] else ''}__{c.category}-{c.model}__{key}"
            arrays = {RB.trunk_key(tr, ti): cc for (tr, ti), cc in cells.items()}
            for (tr, ti), p in pairs.items():
                arrays[RB.trunk_key(tr, ti) + "__pairs"] = p
            np.savez_compressed(os.path.join(out_dir, name + ".npz"), **arrays)
            with open(os.path.join(out_dir, name + ".json"), "w") as f:
                json.dump(meta, f, indent=1, default=str)
            rb.path = os.path.join(out_dir, name + ".npz")
        except Exception as e:  # noqa: BLE001 - the band is still usable in memory
            log.warning("%s: could not write the computed reach band: %r", self.gp.alias, e)
        log.info("%s: reach band computed on miss (%s, %d cells, %.0fs)", self.gp.alias, meta["support"], meta["n_reachable_total"], meta["time_s"])
        return rb

    def band_cells_world(self, rb: Optional[RB.ReachBands], stance: Stance) -> th.Tensor:
        """The band of the stance's grid trunk as world xy at this concrete stance (the band is base-relative)."""
        if rb is None:
            return th.zeros(0, 2)
        cells = rb.band(stance.translate, stance.tilt)
        if len(cells) == 0:
            return th.zeros(0, 2)
        return th.as_tensor(rb.to_world(cells, stance.base_pose2d), dtype=th.float32)

    # ------------------------------------------------------------------ stances A / B per surface candidate

    def try_surface(self, kind: str, cand, rng: np.random.Generator):
        """Build the template facts of stances A and B on one init-surface candidate, and the C survivors' existence."""
        self.surface = {"kind": kind, "obj": None if kind == "floor" else cand}
        if kind == "floor":
            self.try_floor_surface(rng)
        else:
            self.try_support_surface(cand, rng)
        # stance C must exist for the init pose class (checked on the surface's representative init xy)
        c_survivors = self.placement_candidates(self.surface["init_xy"])
        if not c_survivors:
            raise Reject("no_placement_stance", f"no stance C at {self.partner.name} for an init pose at {self.surface['init_xy'].tolist()}")
        self.surface["n_c"] = len(c_survivors)
        self.template_facts["stances"] = {"init": self.surface["init"].to_meta(), "n_manipulation_candidates": len(self.surface["manip"]), "n_placement_candidates": len(c_survivors)}

    def try_support_surface(self, support, rng: np.random.Generator):
        """Stances A and B on a support (tabletop Phases 1a / 1b / 1c / 2 of ``tabletop.py``): per edge (random order)
        the init trunk, the lateral draw redrawn (≤ 6) until the partner is visible (≥ 4 / 8 corners + centre), the
        Phase-2 survivors with the band of (support, target)."""
        self.kit.sync_world()
        top = self.support_top(support, ignore=[self.target])
        if int(top.free.sum()) < 50:
            raise Reject("support_small", f"{support.name}: too little free top")
        self.surface["top"] = top
        self.bands = None  # ``TT.ensure_bands`` caches one band per support on the sampler
        half, height = self.target_dims()
        margins = {"w_t": half, "w_p": 0.0, "W_p": 0.0, "D_p_max": 0.0, "safety": 0.02, "relation": self.relation, "g_NT": None, "H_max": height}
        # A5 "partner's own top": the partner is the support under the target, so the partner-visibility redraw does not
        # apply (the target cell in frame is on the partner); stances A / B come from the tabletop phases unchanged.
        # Owner decision 2026‑09‑16 (a)/(d): stance A sees the target (tabletop Phase 1), the init surface (the target's cell
        # on it) and the partner's RELEVANT REGION (ontop: the top-face patch around the goal cell + the nearest bottom
        # edge; under: the nearest bottom edge) — no "≥ 4 of 8 AABB corners + centre" any more; (b): stance B (the grasp
        # stance, Phase 2) has no partner rule — only its target-in-frame rule.
        own_top = support is self.partner
        errors: List[str] = []
        for edge, init in TT.edge_candidates(self, top, rng):
            try:
                init = TT.phase1b_init_trunk(self, init, height, top=top)
                chosen = None
                for _ in range(1 if own_top else 6):  # §5.6: redraw the lateral position (≤ 6) until the goal region is visible
                    st = TT.phase1c_lateral(self, init, margins, rng, top=top)
                    if own_top or self.goal_region_visible(st):
                        chosen = st
                        break
            except Reject as e:
                if e.reason in ("no_workspace", "edge_too_short", "stance_collision"):
                    errors.append(f"{edge.name}: {e.reason}")
                    continue
                raise
            if chosen is None:
                errors.append(f"{edge.name}: goal_region_not_visible (the partner's top patch around the goal cell / nearest bottom edge out of frame)")
                continue
            chosen.name = "init"
            manip = TT.phase2_manipulation_stance(self, chosen, half, target_height=height, top=top)
            if not manip:
                errors.append(f"{edge.name}: no_manipulation_stance")
                continue
            self.surface.update({"edge": edge, "init": chosen, "manip": manip, "init_xy": manip[0].focus[:2].clone(), "z": top.z, "band": getattr(self, "bands", None)})
            return
        raise Reject("no_free_edge", f"{support.name}: " + ("; ".join(errors) or "no edge with a free, collision-free stance"))

    def goal_region_points(self, from_xy) -> th.Tensor:
        """Owner decision 2026‑09‑16 (a)/(d): the **relevant region** of the partner stance A must see — ontop: the top-face
        patch (± 0.10 m) around the goal cell the placement will use (the goal cell nearest to ``from_xy``, the rule C
        applies) at the top's height, plus the partner's bottom edge nearest to ``from_xy`` (its two bottom corners and
        the midpoint); under: the nearest bottom edge only (the goal lies beneath the partner)."""
        partner = self.partner
        lo, hi = partner.aabb
        f = th.as_tensor([float(from_xy[0]), float(from_xy[1])])
        pts = []
        if self.relation == "ontop" and len(self.place.get("cells", [])):
            cells = self.place["cells"]
            g = cells[int(th.argmin(th.linalg.norm(cells.float() - f, dim=1)))].float()
            z = float(self.place["z"])
            r = float(self.num("goal_patch_half_m", 0.10))
            for dx, dy in ((0, 0), (r, r), (r, -r), (-r, r), (-r, -r)):
                pts.append(th.tensor([float(g[0]) + dx, float(g[1]) + dy, z]))
        # nearest bottom edge of the AABB (as seen from ``from_xy``)
        edges = aabb_edges(lo, hi)
        edge = nearest_edge(edges, f)
        half_len = edge.length / 2
        for t in (-half_len, 0.0, half_len):
            p = edge.mid + edge.tangent * t
            pts.append(th.tensor([float(p[0]), float(p[1]), float(lo[2])]))
        return th.stack(pts)

    def goal_region_visible(self, stance: Stance, from_xy=None) -> bool:
        """The relevant partner region (:meth:`goal_region_points`) in frame with the 5 % margin from ``stance``."""
        cam = self.camera_pose_at(stance)
        if from_xy is None:
            from_xy = stance.focus[:2] if stance.focus is not None else th.as_tensor(stance.base_pose2d[:2])
        return P0.in_frame(self, cam, self.goal_region_points(from_xy))

    def partner_visible(self, stance: Stance, n_min: int) -> bool:
        cam = self.camera_pose_at(stance)
        lo, hi = self.partner.aabb
        corners = G.aabb_corners(lo.float(), hi.float())
        inside = self.world.camera_model.inside(cam, corners, 0.0)
        centre_in = bool(self.world.camera_model.inside(cam, ((lo + hi) / 2).float().reshape(1, 3), 0.0)[0])
        return int(inside.sum()) >= n_min and centre_in

    def try_floor_surface(self, rng: np.random.Generator):
        """§5.6 floor init: the anchor cell c_t, stance B facing it from {0.35, 0.45, 0.55} with the trunk from the floor
        band, stance A on the eroded map 0.6–1.5 m from c_t with the target and the partner in frame."""
        fi = dict(self.gp.gen("floor_init") or {})
        cell = float(fi.get("cell_m", 0.02))
        clearance = float(fi.get("clearance_m", 0.02))
        max_geo = float(fi.get("max_geodesic_m", self.num("max_partner_geodesic_m", 3.0)))
        floor = self.obj("floor")
        partner = self.partner
        lo, hi = floor.aabb
        plo, phi = partner.aabb
        rb = self.band("floor")
        self.surface["band"] = rb
        half, height = self.target_dims()
        objs = self.floor_blocking_objects()
        aabbs = [(o.aabb[0][:2].float() - clearance, o.aabb[1][:2].float() + clearance) for o in objs]
        # candidate anchors: free floor cells within max_geo of the partner's near side, outside every AABB and the partner footprint
        xs = th.arange(float(lo[0]) + cell / 2, float(hi[0]), cell * 4)  # 8 cm search grid (the 2 cm raster is the draw's resolution)
        ys = th.arange(float(lo[1]) + cell / 2, float(hi[1]), cell * 4)
        gx, gy = th.meshgrid(xs, ys, indexing="ij")
        pts = th.stack([gx.flatten(), gy.flatten()], dim=1)
        near_c = th.tensor([float(plo[0] + phi[0]) / 2, float(plo[1] + phi[1]) / 2])
        pts = pts[th.linalg.norm(pts - near_c, dim=1) <= max_geo + 1.5]
        anchors = []
        stats = {"grid": int(len(pts)), "trav_free": 0, "in_rooms": 0, "outside_partner": 0, "outside_objects": 0}
        blockers: Dict[str, int] = {}
        for p in pts:
            if not self.kit.trav_free(p):
                continue
            stats["trav_free"] += 1
            if not P0.in_task_rooms(self, p):
                continue
            stats["in_rooms"] += 1
            if bool((p >= plo[:2].float() - half).all() and (p <= phi[:2].float() + half).all()):
                continue  # inside the partner footprint (the goal must be false at init)
            stats["outside_partner"] += 1
            hit = [o.name for o, (a, b) in zip(objs, aabbs) if bool((p >= a).all() and (p <= b).all())]
            if hit:
                blockers[hit[0]] = blockers.get(hit[0], 0) + 1
                continue
            stats["outside_objects"] += 1
            anchors.append(p)
        self.template_facts["floor_cells"] = {**stats, "blockers": dict(sorted(blockers.items(), key=lambda kv: -kv[1])[:6]), "floor": floor.name,
                                              "floor_aabb": [lo[:2].tolist(), hi[:2].tolist()]}
        if not anchors:
            raise Reject("no_floor_cell", f"no free floor cell beside the partner ({stats}; blocking AABBs {dict(sorted(blockers.items(), key=lambda kv: -kv[1])[:4])}; "
                                          f"floor {floor.name} aabb {[round(float(v), 2) for v in lo[:2].tolist()]}..{[round(float(v), 2) for v in hi[:2].tolist()]})")
        anchors = th.stack(anchors)
        order = rng.permutation(len(anchors))
        errors = []
        for i in order[:40]:
            c_t = anchors[int(i)]
            try:
                _, dist = self.kit.geodesic(c_t, self.near_side_point(partner, c_t))
            except Exception:  # noqa: BLE001
                dist = None
            if dist is None or dist > max_geo:
                continue
            try:
                manip = self.floor_stance_b(c_t, rb, half, rng)
                init = self.floor_stance_a(c_t, rng, height)
                self.record_path(init, manip[0])  # §5.6: a path A -> B exists
            except Reject as e:
                errors.append(f"{e.reason}")
                continue
            self.sidecar.paths.clear()
            self.surface.update({"c_t": c_t, "init": init, "manip": manip, "init_xy": c_t.clone(), "z": self.floor_z, "edge": None})
            self.template_facts["floor_init"] = {"c_t": c_t.tolist(), "geodesic_m": dist, "n_anchor_candidates": int(len(anchors))}
            return
        raise Reject("no_floor_cell", f"no anchor cell admits stances A / B: {sorted(set(errors))}")

    def floor_stance_b(self, c_t: th.Tensor, rb: Optional[RB.ReachBands], half: float, rng: np.random.Generator) -> List[Stance]:
        """Stance B candidates of the floor init: base facing c_t at {0.35, 0.45, 0.55} (bearing uniform), trunk from the
        floor band (c_t must be a band cell), collision-free, target in frame, axis within 8° (else 20°) of c_t."""
        trunks = rb.trunks() if rb is not None else [(t, k) for t in RB.TRANSLATES for k in list(RB.TILTS) + [RB.LOW_TOP_TILT]]
        focus = th.tensor([float(c_t[0]), float(c_t[1]), self.floor_z + 0.03])
        cands = []
        for _ in range(12):
            bearing = float(rng.uniform(-math.pi, math.pi))
            for dist in STAND_OFFS:
                xy = c_t + th.tensor([math.cos(bearing), math.sin(bearing)]) * dist
                if not P0.base_ok(self, xy):
                    continue
                yaw = math.atan2(float(c_t[1] - xy[1]), float(c_t[0] - xy[0]))
                for tr, ti in trunks:
                    st = Stance("manipulation", (float(xy[0]), float(xy[1]), yaw), tr, ti, dist, None, None, 0.0, focus.clone(), ["right"])
                    if rb is not None:
                        cells = self.band_cells_world(rb, st)
                        if len(cells) == 0 or float(th.linalg.norm(cells - c_t, dim=1).min()) > 0.51 * float(rb.meta.get("lattice", RB.LATTICE)):
                            continue
                    cands.append(st)
        if not cands:
            raise Reject("no_manipulation_stance", "floor init: c_t is in no band cell of any stance")
        free = P0.body_free(self, cands)
        cands = [st for st, f in zip(cands, free.tolist()) if f]
        if not cands:
            raise Reject("no_manipulation_stance", "floor init: every stance B collides")
        pts = G.local_points_world(self.hull, (focus, self.rest))
        classed = []
        for st, cam in zip(cands, P0.camera_poses(self, cands)):
            if not P0.in_frame(self, cam, pts):
                continue
            k = P0.angle_class(P0.axis_angle(self, cam, focus))
            if k is None:
                continue
            st.extras["angle_class"] = k
            classed.append((st, k))
        out = P0.best_class(classed)
        if not out:
            raise Reject("no_manipulation_stance", "floor init: no stance B sees c_t within 20°")
        return out

    def floor_stance_a(self, c_t: th.Tensor, rng: np.random.Generator, height: float) -> Stance:
        """Stance A of the floor init: base on the eroded map 0.6–1.5 m from c_t facing it, tilt 0, the translate whose
        optical axis is closest to c_t with the target and the partner's relevant region (goal patch / nearest bottom
        edge, owner decision (a)/(d)) in frame, collision-free."""
        fi = dict(self.gp.gen("floor_init") or {})
        d_lo, d_hi = fi.get("distance_range_m", [0.6, 1.5])
        focus = th.tensor([float(c_t[0]), float(c_t[1]), self.floor_z + height / 2])
        pts = G.local_points_world(self.hull, (th.tensor([float(c_t[0]), float(c_t[1]), self.floor_z + self.world.bottom_offset(self.hull, self.rest)]), self.rest))
        for _ in range(30):
            bearing = float(rng.uniform(-math.pi, math.pi))
            dist = float(rng.uniform(float(d_lo), float(d_hi)))
            xy = c_t + th.tensor([math.cos(bearing), math.sin(bearing)]) * dist
            if not self.kit.trav_free(xy, eroded=True) or not P0.in_task_rooms(self, xy):
                continue
            yaw = math.atan2(float(c_t[1] - xy[1]), float(c_t[0] - xy[0]))
            cands = [Stance("init", (float(xy[0]), float(xy[1]), yaw), t, 0.0, dist) for t in RB.TRANSLATES]
            cams = P0.camera_poses(self, cands)
            feas = []
            for st, cam in zip(cands, cams):
                if not P0.in_frame(self, cam, pts) or not self.goal_region_visible(st, from_xy=c_t):
                    continue
                feas.append((st, P0.axis_angle(self, cam, focus)))
            if not feas:
                continue
            free = P0.body_free(self, [st for st, _ in feas])
            feas = [(st, a) for (st, a), f in zip(feas, free.tolist()) if f]
            if not feas or not P0.trunk_sweep_free(self, (float(xy[0]), float(xy[1]), yaw)):
                continue
            st = min(feas, key=lambda sa: (round(sa[1], 6), sa[0].translate))[0]
            st.arms = "untucked"
            st.extras.update({"bearing_deg": math.degrees(bearing), "distance": dist})
            return st
        raise Reject("no_stance", "floor init: no stance A sees the target at c_t and the partner")

    # ------------------------------------------------------------------ stance C (placement)

    def placement_candidates(self, init_xy) -> List[Stance]:
        """§5.6 stance C survivors for an init pose: base square to the partner edge nearest ``init_xy``, stand-off
        {0.35, 0.45, 0.55}, lateral along the edge minus margins, trunk from the band; goal cell per relation;
        visibility per relation; the best angle class (each survivor carries its ``goal_cell`` in ``extras``)."""
        partner = self.partner
        half = self.place["half"]
        edge = nearest_edge(self.place["edges"], init_xy)
        if self.push_under:
            return self.push_under_candidates(edge, init_xy)
        rb = self.band("tabletop", partner) if self.relation == "ontop" else self.band("floor", None, partner)
        self.place["band"] = rb
        trunks = rb.trunks() if rb is not None else [(t, k) for t in RB.TRANSLATES for k in RB.TILTS]
        margin = half + 0.10
        L = edge.length
        half_run = L / 2 - margin
        laterals = P0.lattice_values(-half_run, half_run, 0.10) if half_run > 0 else [0.0]
        init_xy = th.as_tensor([float(init_xy[0]), float(init_xy[1])])
        goal_cells = self.place["cells"]
        stats: Dict[str, int] = {"goal_cells": int(len(goal_cells)), "band_trunks": len(trunks), "lattice": 0, "base_ok": 0, "band_cells": 0, "goal_in_band": 0}
        cands: List[Stance] = []
        for so in [float(v) for v in (self.gp.gen("stand_offs_m") or STAND_OFFS)]:
            for lat in laterals:
                xy = edge.base_xy(so, lat)
                stats["lattice"] += len(trunks)
                if not P0.base_ok(self, xy):
                    continue
                stats["base_ok"] += len(trunks)
                for tr, ti in trunks:
                    st = Stance("placement", (float(xy[0]), float(xy[1]), edge.yaw), tr, ti, so, edge, None, lat, None, ["right"])
                    cells = self.band_cells_world(rb, st)
                    if rb is not None and len(cells) == 0:
                        continue
                    stats["band_cells"] += 1
                    pool = goal_cells if rb is None else self._snap(cells, goal_cells, float(rb.meta.get("lattice", RB.LATTICE)))
                    if len(pool) == 0:
                        continue
                    stats["goal_in_band"] += 1
                    gc = pool[int(th.argmin(th.linalg.norm(pool - init_xy, dim=1)))]
                    st.extras["goal_cell"] = [float(gc[0]), float(gc[1])]
                    cands.append(st)
        self.template_facts["placement_stats"] = stats
        if not cands:
            log.info("%s: stance C at %s edge %s: no candidate (%s)", self.gp.alias, partner.name, edge.name, stats)
            return []
        free = P0.body_free(self, cands)
        cands = [st for st, f in zip(cands, free.tolist()) if f]
        stats["collision_free"] = len(cands)
        if not cands:
            log.info("%s: stance C at %s edge %s: every candidate collides (%s)", self.gp.alias, partner.name, edge.name, stats)
            return []
        cams = P0.camera_poses(self, cands)
        lo, hi = partner.aabb
        tangent = th.tensor([-float(edge.inward[1]), float(edge.inward[0])])
        classed = []
        for st, cam in zip(cands, cams):
            gc = st.extras["goal_cell"]
            if self.relation == "ontop":
                pose = self.goal_pose_at(gc, self.rest)
                if not P0.in_frame(self, cam, self.hull_corners_world(self.hull, pose)):
                    continue
                focus = th.tensor([gc[0], gc[1], self.place["z"] + self.place["height"] / 2])
                # owner decision 2026‑09‑16 (c): the ontop goal-visibility rule is "goal cell (the target at the goal) in frame
                # (5 %)" — the 20° optical-axis cone stays for under only (the target hides beneath the partner). Diagnosed
                # on box_of_shampoo-sofa-bar: 13/13 collision-free C candidates had the goal in frame, unoccluded, at 30–31° off
                # the axis (camera rigid on the torso, trunk (0, −1.2) from the band, stand-off 0.35 m); an optional cone
                # (``ontop_axis_cone_deg``) is kept for experiments.
                cone = self.gp.gen("ontop_axis_cone_deg")
                ang = P0.axis_angle(self, cam, focus)
                st.extras["axis_angle_deg"] = math.degrees(ang)
                k = (1 if ang <= math.radians(float(cone)) else None) if cone is not None else 1
            else:
                # the near edge's bottom line where the target passes under it — the two bottom points of the edge at the
                # goal cell's lateral position ± (half extent + 0.10) — in frame; the optical axis within 20° of the
                # (hidden) goal cell. (The spec's "two bottom AABB corners" lie ±1 m to the sides of a bed / sofa edge, 65°
                # off the optical axis at s = 0.45 m — outside the 99° hfov for every stance; wording change proposed.)
                corners = G.aabb_corners(lo.float(), hi.float())
                bottom = corners[corners[:, 2] <= float(lo[2]) + 1e-4]
                d = (bottom[:, :2] - edge.mid) @ edge.inward
                near_line = bottom[d.abs() <= float(d.abs().min()) + 0.02]
                on_line = near_line[:, :2].mean(dim=0) + tangent * float(th.dot(th.tensor([gc[0], gc[1]]) - near_line[:, :2].mean(dim=0), tangent))
                span = half + 0.10
                near = th.stack([th.cat([on_line - tangent * span, lo[2:3].float()]), th.cat([on_line + tangent * span, lo[2:3].float()])])
                if not P0.in_frame(self, cam, near):
                    continue
                focus = th.tensor([gc[0], gc[1], self.floor_z + self.place["height"] / 2])
                ang = P0.axis_angle(self, cam, focus)
                k = 1 if ang <= math.radians(P0.AXIS_LOOSE_DEG) else None
            if k is None:
                continue
            st.focus = focus
            st.extras["angle_class"] = k
            classed.append((st, k))
        stats["visible"] = len(classed)
        out = P0.best_class(classed)
        log.info("%s: stance C at %s edge %s: %d survivors (%s)", self.gp.alias, partner.name, edge.name, len(out), stats)
        if not out and cands:
            self.debug_placement_visibility(cands, cams, edge, init_xy)
        return out

    def _object_of_prim(self, prim_path: str) -> Optional[str]:
        for o in self.world.scene.objects:
            for l in o.links.values():
                if prim_path == l.prim_path or prim_path.startswith(l.prim_path + "/"):
                    return o.name
        return None

    def debug_placement_visibility(self, cands: Sequence[Stance], cams: Sequence[Any], edge: EdgeFrame, init_xy) -> None:
        """Owner decision 2026‑09‑16 (c) diagnosis: for every collision-free stance-C candidate the goal cell's frame test
        (5 %), its angle to the optical axis, the first object a ray from the camera to the goal point hits (occluder), the
        trunk / stand-off / lateral; plus the partner edges that were considered and why one was chosen (the edge nearest
        the init pose is the only one the spec uses)."""
        try:
            partner = self.partner
            z_top = float(self.place["z"])
            rows = []
            for st, cam in zip(cands, cams):
                gc = st.extras.get("goal_cell")
                if gc is None:
                    continue
                if self.relation == "ontop":
                    focus = th.tensor([gc[0], gc[1], z_top + self.place["height"] / 2])
                    pose = self.goal_pose_at(gc, self.rest)
                    corners_in = P0.in_frame(self, cam, self.hull_corners_world(self.hull, pose))
                else:
                    focus = th.tensor([gc[0], gc[1], self.floor_z + self.place["height"] / 2])
                    corners_in = None
                cell_in = P0.in_frame(self, cam, focus.reshape(1, 3))
                ang = math.degrees(P0.axis_angle(self, cam, focus))
                cam_p = G.t(cam[0]).float()
                hit = self.world.raycast(cam_p.reshape(1, 3), focus.reshape(1, 3))[0]
                occluder = None
                if hit.get("hit"):
                    name = self._object_of_prim(hit.get("rigidBody", ""))
                    d_hit = float(th.linalg.norm(th.as_tensor(hit["position"]).float() - cam_p))
                    d_goal = float(th.linalg.norm(focus - cam_p))
                    if name is not None and name != partner.name and d_hit < d_goal - 0.03:
                        occluder = f"{name} at {d_hit:.2f} m (goal at {d_goal:.2f} m)"
                    elif name == partner.name and d_hit < d_goal - 0.05:
                        occluder = f"{partner.name} itself at {d_hit:.2f} m (goal at {d_goal:.2f} m: armrest / back / cushion in front of the cell)"
                rows.append({"stand_off": round(float(st.stand_off or 0.0), 2), "lateral": round(float(st.lateral), 2), "trunk": [float(st.translate), round(float(st.tilt), 2)],
                             "goal_cell": [round(float(v), 2) for v in gc], "goal_in_frame": bool(cell_in), "goal_hull_in_frame": corners_in, "axis_angle_deg": round(ang, 1),
                             "occluder": occluder, "camera_z": round(float(cam_p[2]), 2)})
            f = th.as_tensor([float(init_xy[0]), float(init_xy[1])])
            edges = [{"name": e.name, "length_m": round(float(e.length), 2), "dist_to_init_m": round(float(th.linalg.norm(e.mid - f)), 2), "chosen": e.name == edge.name,
                      "why": "nearest to the init pose (§5.6: edge = the partner edge nearest the target's init pose)" if e.name == edge.name else "not the nearest edge"}
                     for e in self.place["edges"]]
            self.template_facts["placement_visibility_debug"] = {"candidates": rows, "edges": edges}
            n_in = sum(1 for r in rows if r["goal_in_frame"])
            n_cone = sum(1 for r in rows if r["axis_angle_deg"] <= P0.AXIS_LOOSE_DEG)
            n_occ = sum(1 for r in rows if r["occluder"])
            log.info("%s: stance C visibility debug at %s edge %s: %d candidates — goal cell in frame %d, within the 20° cone %d, occluded %d; edges %s; per candidate %s",
                     self.gp.alias, partner.name, edge.name, len(rows), n_in, n_cone, n_occ, edges, rows)
        except Exception as e:  # noqa: BLE001 — diagnostics never fail the attempt
            log.info("%s: stance C visibility debug failed: %r", self.gp.alias, e)

    def push_under_candidates(self, edge: EdgeFrame, init_xy) -> List[Stance]:
        """Stance C of the push-under (owner decision 2026‑09‑16 item 7): base square to the partner's near AABB edge,
        facing inward (= the push direction); per lateral position λ the **goal cell** g = edge point + inward·0.07
        (target centre 5–10 cm inside the footprint, clear under: ray ≥ height + 3 cm) and the **drop cell** c_d = edge
        point − inward·(half + 0.10) on free floor; stand-off s ∈ {0.5…0.7} so that c_d lies 0.25–0.6 m ahead of the
        base (B3: the floor is reachable 0.24–0.64 m ahead at the deep trunks); trunks = the floor trunks (translate ≥
        0.75, tilt ≤ −0.9); whole-body collision-free (A6 idle-arm retry), the drop footprint and the edge's bottom line
        at λ ± (half + 0.10) in frame, optical axis within 20° of the drop cell. Each survivor carries ``goal_cell``,
        ``drop_cell``, ``push_dir``, ``drop_ahead_m``."""
        partner = self.partner
        half, height = float(self.place["half"]), float(self.place["height"])
        need = float(self.place.get("need", height + 0.03))
        d2 = th.as_tensor([float(edge.inward[0]), float(edge.inward[1])])
        tangent = th.tensor([-float(d2[1]), float(d2[0])])
        inside = float(self.num("push_under_inside_m", PUSH_UNDER_INSIDE_M))
        out = half + float(self.num("push_under_drop_extra_m", PUSH_UNDER_DROP_EXTRA_M))
        margin = half + 0.10
        half_run = edge.length / 2 - margin
        laterals = P0.lattice_values(-half_run, half_run, 0.10) if half_run > 0 else [0.0]
        stand_offs = [float(v) for v in (self.gp.gen("push_under_stand_offs_m") or PUSH_UNDER_STAND_OFFS)]
        trunks = [tuple(t) for t in (self.gp.gen("floor_trunks") or FLOOR_TRUNKS)]
        lo_ahead, hi_ahead = PUSH_UNDER_DROP_AHEAD_M
        cells = self.place["cells"]
        mid = th.as_tensor([float(edge.mid[0]), float(edge.mid[1])])
        stats: Dict[str, Any] = {"laterals": len(laterals), "goal_clear": 0, "drop_free": 0, "base_ok": 0, "candidates": 0, "side_candidates": 0, "drop_blocked": {},
                                 "drop_out_m": round(out, 3), "edge_mid": [round(float(v), 2) for v in mid.tolist()], "inward": [round(float(v), 2) for v in d2.tolist()]}
        # side stances: the base stands BESIDE the push lane, facing along the edge with the partner on its right (the
        # right arm pushes sideways, toward the partner); the lane lies ``x_ahead`` ahead of the base and the base centre
        # ``b`` outside the edge — the whole lane (drop 0.22 m outside → goal 0.07 m inside) stays 0.35–0.45 m ahead, inside
        # the floor reach, and the crouched torso bends along the edge, not into the partner (the frontal stances need
        # s ≥ 0.75 m to keep the deep trunk clear of a bed, which puts the lane 0.46–0.82 m ahead: out of reach)
        side_ahead = [float(v) for v in (self.gp.gen("push_under_side_ahead_m") or (0.35, 0.45))]
        side_b = [float(v) for v in (self.gp.gen("push_under_side_standoff_m") or (0.35, 0.40))]
        yaw_side = math.atan2(float(tangent[1]), float(tangent[0]))  # facing +tangent: the robot's right = inward (the partner side)
        cands: List[Stance] = []
        for lat in laterals:
            g = mid + tangent * float(lat) + d2 * inside
            cl = self.clearance_at(g)
            clear = (cl is not None and cl >= need) or (cl is None and len(cells) > 0 and float(th.linalg.norm(cells - g, dim=1).min()) <= 0.03)
            if not clear:
                continue
            stats["goal_clear"] += 1
            c_d = mid + tangent * float(lat) - d2 * out
            if not self._free_floor_cell(c_d, stats["drop_blocked"]):
                continue
            stats["drop_free"] += 1
            common = {"goal_cell": [float(g[0]), float(g[1])], "drop_cell": [float(c_d[0]), float(c_d[1])], "push_dir": [float(d2[0]), float(d2[1])], "inside_m": inside,
                      "edge": edge.name}
            for so in stand_offs:
                ahead = so - out
                if ahead < lo_ahead or ahead > hi_ahead:
                    continue
                xy = edge.base_xy(so, float(lat))
                if not P0.base_ok(self, xy):
                    continue
                stats["base_ok"] += 1
                for tr, ti in trunks:
                    st = Stance("placement", (float(xy[0]), float(xy[1]), edge.yaw), float(tr), float(ti), so, edge, None, float(lat), None, ["right"])
                    st.focus = th.tensor([float(c_d[0]), float(c_d[1]), self.floor_z + height / 2])
                    st.extras.update(common, drop_ahead_m=float(ahead), stance_kind="frontal")
                    cands.append(st)
                    stats["candidates"] += 1
            for x_ahead in side_ahead:
                for b in side_b:
                    xy = mid + tangent * (float(lat) - x_ahead) - d2 * b
                    if not P0.base_ok(self, xy):
                        continue
                    stats["base_ok"] += 1
                    for tr, ti in trunks:
                        st = Stance("placement", (float(xy[0]), float(xy[1]), yaw_side), float(tr), float(ti), b, None, None, float(lat), None, ["right"])
                        st.focus = th.tensor([float(c_d[0]), float(c_d[1]), self.floor_z + height / 2])
                        st.extras.update(common, drop_ahead_m=float(x_ahead), stance_kind="side", side_standoff_m=b)
                        cands.append(st)
                        stats["side_candidates"] += 1
        self.template_facts["placement_stats"] = stats
        if not cands:
            log.info("%s: push-under stance C at %s edge %s: no candidate (%s)", self.gp.alias, partner.name, edge.name, stats)
            return []
        # the deep trunks (translate 0.95) are what reaches the floor (B3), and at a floor stance they trip the 1 cm margin
        # against the floor *mesh* (torso / head "within 1 cm of the floor" — B3 README; scene-specific, the floor object
        # sits a little above floor_z): the stance check runs with the floor object removed, the cuRobo ground slab
        # (which stays) keeps the robot from penetrating the floor. Documented relaxation for the push-under stance.
        with self.ikc.ignoring([self.obj("floor")]):
            free = P0.body_free(self, cands)
        deep_fail = [st for st, f in zip(cands, free.tolist()) if not f and st.translate >= 0.95]
        cands = [st for st, f in zip(cands, free.tolist()) if f]
        stats["collision_free"] = len(cands)
        if deep_fail and not any(st.translate >= 0.95 for st in cands) and not getattr(self, "_deep_debugged", False):
            # which spheres / postures fail at the floor-reaching trunks (B3: "torso / head within 1 cm of the floor")
            self._deep_debugged = True
            with self.ikc.ignoring([self.obj("floor")]):
                P0.debug_collisions(self, deep_fail[len(deep_fail) // 2:len(deep_fail) // 2 + 2], arms_excluded=False)
        if not cands:
            log.info("%s: push-under stance C at %s edge %s: every candidate collides (%s)", self.gp.alias, partner.name, edge.name, stats)
            return []
        cams = P0.camera_poses(self, cands)
        lo, hi = partner.aabb
        z_bottom = float(lo[2])
        classed = []
        for st, cam in zip(cands, cams):
            c_d = th.as_tensor(st.extras["drop_cell"])
            # the drop footprint (a disc of the half extent around the drop cell, on the floor) and the edge's bottom line at
            # the push lane in frame (5 %); the optical axis within 20° of the drop cell
            disc = th.stack([th.cat([c_d + tangent * half, th.tensor([self.floor_z])]), th.cat([c_d - tangent * half, th.tensor([self.floor_z])]),
                             th.cat([c_d + d2 * half, th.tensor([self.floor_z])]), th.cat([c_d - d2 * half, th.tensor([self.floor_z])])])
            e0 = mid + tangent * float(st.lateral)
            lane = th.cat([e0, th.tensor([z_bottom])]).reshape(1, 3)  # the edge's bottom point at the push lane
            # a side stance looks along the edge: the lane point sits at the frame's lateral border (45–49° off-axis at
            # b = 0.35–0.40 m); the drop footprint in frame is what it needs (the partner's side fills that half of the image)
            pts = disc if st.extras.get("stance_kind") == "side" else th.cat([disc, lane])
            kind = st.extras.get("stance_kind", "frontal")
            if not P0.in_frame(self, cam, pts):
                stats[f"{kind}_out_of_frame"] = stats.get(f"{kind}_out_of_frame", 0) + 1
                if kind == "side" and stats[f"{kind}_out_of_frame"] == 1:
                    log.info("%s: side stance C example out of frame: base %s trunk %s camera %s focus %s disc %s", self.gp.alias, [round(float(v), 2) for v in st.base_pose2d],
                             (st.translate, st.tilt), [round(float(v), 2) for v in cam[0].tolist()], [round(float(v), 2) for v in st.focus.tolist()],
                             [[round(float(v), 2) for v in p] for p in disc.tolist()])
                continue
            ang = P0.axis_angle(self, cam, st.focus)
            # at the crouched trunks the head hangs ≈ 0.6 m ahead of the base, i.e. over the lane: the drop cell lies almost
            # straight below the camera, 30–40° off the (74–90° pitched) optical axis while squarely in frame — the 20° rule
            # is relaxed to ``push_under_axis_max_deg`` (45°) for the push-under stance (deviation from §5.6 to record)
            if ang > math.radians(float(self.num("push_under_axis_max_deg", 45.0))):
                stats[f"{kind}_axis_off"] = stats.get(f"{kind}_axis_off", 0) + 1
                if kind == "side" and stats[f"{kind}_axis_off"] == 1:
                    log.info("%s: side stance C example axis off by %.0f deg: trunk %s camera %s focus %s", self.gp.alias, math.degrees(ang), (st.translate, st.tilt),
                             [round(float(v), 2) for v in cam[0].tolist()], [round(float(v), 2) for v in st.focus.tolist()])
                continue
            st.extras["angle_class"] = 1
            classed.append((st, 1))
        stats["visible"] = len(classed)
        out_ = P0.best_class(classed)
        stats["survivor_trunks"] = sorted({(float(st.translate), round(float(st.tilt), 2)) for st in out_})
        stats["survivor_stand_offs"] = sorted({round(float(st.stand_off), 2) for st in out_})
        stats["survivor_kinds"] = {k: sum(1 for st in out_ if st.extras.get("stance_kind") == k) for k in ("side", "frontal")}
        stats["collision_free_trunks"] = sorted({(float(st.translate), round(float(st.tilt), 2)) for st in cands})
        log.info("%s: push-under stance C at %s edge %s: %d survivors (%s)", self.gp.alias, partner.name, edge.name, len(out_), stats)
        return out_

    def debug_drop_ik(self, place: Stance, release_xy, yaws: Sequence[float]) -> str:
        """Diagnostic when F_goal is empty at the drop cell: for the first yaw and every library grasp, the drop grasp pose
        from stance C with the floor ignored vs. with every collision object ignored (reach vs. collision), and the eef
        height above the floor / distance ahead of the base."""
        try:
            floor = self.obj("floor")
            others = [o for o in self.world.scene.objects if o.name in self.world.scene_object_names]
            yaw = float(yaws[0]) if len(yaws) else 0.0
            quat = G.qmul(G.quat_from_yaw(yaw), self.rest)
            pose = self.drop_pose_at(release_xy, quat)
            parts = []
            for g in self.grasps[:3]:
                eef = IK.grasp_pose(pose, g.pose)
                r1 = self.ikc.exact([eef], place, "right", ignore_for_last=[floor], cache=False)
                with self.ikc.ignoring(others):
                    r2 = self.ikc.exact([eef], place, "right", cache=False)
                p_base = P0.to_base_frame(eef[0], place.base_pose2d, P0.base_z(self))
                parts.append(f"{g.name}: floor-ignored {r1.ok}, world-free {r2.ok}, eef in base {[round(float(v), 2) for v in p_base.tolist()]}")
            return f"stance d {place.stand_off:.2f} trunk ({place.translate}, {place.tilt:.1f}) drop ahead {place.extras.get('drop_ahead_m')}: " + "; ".join(parts)
        except Exception as e:  # noqa: BLE001 — diagnostics never fail the attempt
            return f"debug failed: {e}"

    @staticmethod
    def _snap(band_world: th.Tensor, goal_cells: th.Tensor, lattice: float) -> th.Tensor:
        """Goal cells within half a lattice step of a band cell (the band admits them)."""
        if len(band_world) == 0 or len(goal_cells) == 0:
            return th.zeros(0, 2)
        d = th.cdist(goal_cells.float(), band_world.float())
        return goal_cells[d.min(dim=1).values <= 0.51 * lattice]

    def goal_pose_at(self, goal_cell, quat) -> G.Pose:
        """§5.6 goal pose: ontop → z = partner face + bottom offset + 1 cm; under → z = floor + bottom offset + 3 mm."""
        z = self.place["z"] + self.world.bottom_offset(self.hull, quat) + self.num("goal_drop_m", 0.01 if self.relation == "ontop" else 0.003)
        return th.tensor([float(goal_cell[0]), float(goal_cell[1]), z]), quat

    # ------------------------------------------------------------------ template

    def sample_template(self):
        self.template_facts.update({"relation": self.relation, "grasp_key": self.grasp_key, "n_grasps": len(self.grasps),
                                    "band_union_extent": None if self.surface.get("band") is None else list(self.surface["band"].union_extent())})

    # ------------------------------------------------------------------ instance (§5.6)

    def pose_on(self, xy, yaw: float, z: float) -> G.Pose:
        quat = G.qmul(G.quat_from_yaw(yaw), self.rest)
        return th.tensor([float(xy[0]), float(xy[1]), z + self.world.bottom_offset(self.hull, quat) + 0.003]), quat

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        init: Stance = self.surface["init"]
        self.sidecar.add_stance(init)
        kind = self.surface["kind"]
        b_pool = list(self.surface["manip"])
        errors: List[str] = []
        for _ in range(min(6, len(b_pool))):
            kb = int(rng.integers(len(b_pool)))
            b_grid = b_pool.pop(kb)
            manip = b_grid.with_trunk(b_grid.translate, b_grid.tilt, "manipulation")
            manip.focus, manip.arms = b_grid.focus, ["right"]
            self.sidecar.transitions.clear()
            self.sidecar.paths.clear()
            try:
                # A -> B (§3.7 / §3.8)
                self.record_path(init, manip)
                self.record_transition(init, manip)
                focus_xy = self.surface["c_t"] if kind == "floor" else manip.focus[:2]
                z_init = self.surface["z"]
                F_init = self.ikc.feasible_sets([focus_xy], YAWS, self.grasps, lambda xy, yaw: self.pose_on(xy, yaw, z_init), manip, "right", self.grasp_ignore_objects())
                F_init = F_init.get(0, [])
                if not F_init:
                    raise Reject("grasp_unreachable", "empty feasible set F_init at B's focus cell")
                c_pool = self.placement_candidates(focus_xy)
                if not c_pool:
                    raise Reject("no_placement_stance", "no stance C survivor for this init pose")
                d = self._couple(rng, init, manip, c_pool, F_init, focus_xy, z_init)
                d.summary.update({"relation": self.relation, "init_surface": {"class": kind, "name": None if kind == "floor" else self.surface["obj"].name},
                                  "init_stance": P0.summarize_stance(init), "manipulation_stance": P0.summarize_stance(manip)})
                return d
            except Reject as e:
                errors.append(f"B candidate: {e.reason}: {e}")
                self.sidecar.stances.pop("manipulation", None)
                self.sidecar.stances.pop("placement", None)
                self.sidecar.grasps.clear()
        raise Reject("target_placement", "; ".join(errors[-3:]))

    def _couple(self, rng, init: Stance, manip: Stance, c_pool: List[Stance], F_init, focus_xy, z_init: float) -> Draft:
        """B ↔ C coupling: for C survivors in random order, F_goal at the goal cell for the init yaws, the draw from
        F_init ∩ F_goal, the target pose at B's focus, the exact IK at B and at C (attached), the B → C path and
        transition; the draft and the sidecar records."""
        tname, pname = self.tname, self.pname
        partner, target = self.partner, self.target
        max_len = self.num("max_partner_geodesic_m", 3.0) + self.num("path_max_extra_m", 1.0)
        order = list(rng.permutation(len(c_pool)))
        n_try = 6
        if self.push_under:
            # side stances first (the frontal ones put the lane out of the floor reach — see push_under_candidates), the
            # deepest trunk first within a kind (B3: the floor is reached at translate 0.95 / tilt −1.2 only); every side
            # survivor is tried before the frontal ones
            side = [k for k in order if c_pool[k].extras.get("stance_kind") == "side"]
            front = [k for k in order if c_pool[k].extras.get("stance_kind") != "side"]
            depth = lambda k: (-float(c_pool[k].translate), float(c_pool[k].tilt))  # noqa: E731
            order = sorted(side, key=depth) + sorted(front, key=depth)
            n_try = max(12, len(side))
        errors = []
        for kc in order[:n_try]:
            place = c_pool[int(kc)].with_trunk(c_pool[int(kc)].translate, c_pool[int(kc)].tilt, "placement")
            place.focus, place.arms = c_pool[int(kc)].focus, ["right"]
            place.extras = dict(c_pool[int(kc)].extras)
            gc = place.extras["goal_cell"]
            push = self.push_under
            cd = place.extras.get("drop_cell") if push else None
            if self.relation == "under" and not push:
                cl = self.clearance_at(gc)
                if cl is not None and not under_admissible(cl, self.place["height"], self.num("under_clearance_extra_m", 0.17)):
                    errors.append(f"C: under clearance {cl:.2f} m at the goal cell < height + 0.17")
                    continue
            yaws = sorted({float(y) for y, _ in F_init})
            ignore_place = [partner] if self.relation == "ontop" else [self.obj("floor")]
            # the release pose: the goal cell (ontop / under by placement) or the drop cell outside the edge (push-under)
            release_xy = cd if push else gc
            release_pose = (lambda xy, quat: self.drop_pose_at(xy, quat)) if push else (lambda xy, quat: self.goal_pose_at(xy, quat))
            F_goal = self.ikc.feasible_sets([release_xy], yaws, self.grasps, lambda xy, yaw: release_pose(xy, G.qmul(G.quat_from_yaw(yaw), self.rest)), place, "right",
                                            ignore_place).get(0, [])
            F = intersect_feasible(F_init, F_goal)
            if push:
                log.info("%s: push-under C try %s d %.2f λ %+.2f trunk (%.2f, %.1f): F_init %d, F_goal %d, F %d at drop %s", self.gp.alias, place.extras.get("stance_kind"),
                         place.stand_off or 0.0, place.lateral, place.translate, place.tilt, len(F_init), len(F_goal), len(F), [round(float(v), 2) for v in release_xy])
            if not F:
                if push and not F_goal and not getattr(self, "_drop_debugged", False):
                    self._drop_debugged = True
                    log.info("%s: drop IK debug — %s", self.gp.alias, self.debug_drop_ik(place, release_xy, yaws))
                errors.append(f"C: F_init ({len(F_init)}) ∩ F_goal ({len(F_goal)}) empty at {'drop' if push else 'goal'} cell {[round(v, 2) for v in release_xy]}")
                continue
            for i in rng.permutation(len(F))[:4]:
                yaw0, g = F[int(i)]
                yaw = yaw0 + float(rng.uniform(-TARGET_JITTER, TARGET_JITTER))
                t_pose = None
                for _ in range(5):
                    if self.surface["kind"] == "floor":
                        xy = focus_xy
                    else:
                        r, a = float(rng.uniform(0, FOCUS_RADIUS)), float(rng.uniform(0, 2 * math.pi))
                        xy = focus_xy + th.tensor([r * math.cos(a), r * math.sin(a)])
                        if not self.surface["top"].polygon_supported(self.world.footprint(self.hull, self.pose_on(xy, yaw, z_init)), margin=SUPPORT_MARGIN):
                            continue
                    t_pose = self.pose_on(xy, yaw, z_init)
                    break
                if t_pose is None:
                    continue
                eef = IK.grasp_pose(t_pose, g.pose)
                pre = IK.pre_pose(eef)
                res = self.ikc.exact([pre, eef], manip, "right", ignore_for_last=self.grasp_ignore_objects(), scale_key=self.choice.models["target"].scale, cache=False)
                if not res.ok:
                    continue
                goal = release_pose(release_xy, t_pose[1])
                place_eef = IK.grasp_pose(goal, g.pose)
                pre_place = IK.pre_pose(place_eef)
                with attached(self.kit, "right", target, eef_in_object=g.pose):
                    r2 = self.ikc.exact([pre_place, place_eef], place, "right", ignore_for_last=ignore_place, cache=False)
                if not r2.ok:
                    continue
                chain = None
                if push:  # the straight floor push drop → goal from the same stance (closed gripper, floor + target ignored)
                    chain = self._push_under_chain(place, t_pose[1], cd, gc, place.extras["push_dir"])
                    if chain is None:
                        errors.append("C: push chain unreachable")
                        continue
                # B -> C with the object attached (§3.7 ≤ max_partner_geodesic + 1 m; §3.8 transition)
                self.sidecar.add_stance(manip)
                self.sidecar.add_stance(place)
                try:
                    with attached(self.kit, "right", target, eef_in_object=g.pose):
                        path_bc = self.record_path(manip, place, carried=tname, arm="right", max_length_m=max_len)
                        self.record_transition(manip, place, carried=tname, arm="right")
                except Reject as e:
                    errors.append(f"C: {e.reason}")
                    self.sidecar.stances.pop("placement", None)
                    self.sidecar.paths = [p for p in self.sidecar.paths if p.to != "placement"]
                    self.sidecar.transitions = [t for t in self.sidecar.transitions if t.to != "placement"]
                    break
                floor_name = self.bindings.one("floor").name if self.bindings.has("floor") else None
                grasp = GraspRecord(tname, "right", g.key, g.name, yaw, pre, eef, pre_place, place_eef, finger_q=list(g.finger_q), eef_in_object=g.pose,
                                    place_on=(floor_name or pname) if push else pname)
                if push:  # the release is a drop on the floor outside the edge (like the inside rule's ``place_kind: drop``)
                    grasp.extras.update({"place_kind": "drop", "drop_height_m": float(self.num("drop_height_m", PUSH_UNDER_DROP_HEIGHT_M)),
                                         "drop_cell": [float(v) for v in cd], "goal_cell": [float(v) for v in gc], "then": "push_under"})
                self.sidecar.add_grasp(grasp)
                if push:
                    self.sidecar.add_contact(PushUnderRecord("right", chain["pre"], chain["start"], chain["mid"], chain["goal"], chain["direction"], chain["distance_m"],
                                                             tname, stance="placement", pitch_deg=chain["pitch_deg"], blade_standoff_m=chain["blade_standoff_m"],
                                                             push_height_m=chain["push_height_m"], half_extent_m=chain["half_extent_m"], fingertip_z_m=chain["fingertip_z_m"],
                                                             drop_cell=[float(v) for v in cd], goal_cell=[float(v) for v in gc], partner=pname))
                cam = self.camera_pose_at(init)
                if not self.frustum_ok(cam, self.hull_corners_world(self.hull, t_pose), 0.05):
                    raise Reject("not_in_frame", "target not in the init image (5 %)")
                path_ab = next((p for p in self.sidecar.paths if p.to == "manipulation"), None)
                place.extras.update({"init_surface": {"class": self.surface["kind"], "name": None if self.surface["kind"] == "floor" else self.surface["obj"].name},
                                     "path_length_m": {"A_B": None if path_ab is None else path_ab.length_m, "B_C": path_bc.length_m}})
                d = Draft()
                d.object_poses[tname] = t_pose
                d.robot_q = self.stance_q(init)
                d.base_pose2d = tuple(init.base_pose2d)
                d.summary.update({"goal_cell": gc, "grasp": {"key": g.key, "name": g.name, "yaw": yaw}, "placement_stance": P0.summarize_stance(place),
                                  "path_length_m": place.extras["path_length_m"], "F_init": len(F_init), "F_goal": len(F_goal), "F": len(F)})
                if push:
                    d.summary["push_under"] = {"drop_cell": [float(v) for v in cd], "distance_m": chain["distance_m"], "pitch_deg": chain["pitch_deg"],
                                               "blade_standoff_m": chain["blade_standoff_m"], "push_height_m": chain["push_height_m"], "half_extent_m": chain["half_extent_m"],
                                               "drop_ahead_m": place.extras.get("drop_ahead_m")}
                pp, pq = partner.get_position_orientation()
                d.pose_targets[pname] = full_pose_target(pp, pq, self.tol("position", 0.03), self.tol("orientation", 0.20))
                d.pose_targets[tname] = full_pose_target(t_pose[0], t_pose[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
                self.sidecar.describe(tname, self.choice.models["target"].category.replace("_", " "))
                self.sidecar.describe(pname, partner.category.replace("_", " "), path_length_m=path_bc.length_m)
                if push and floor_name:
                    self.sidecar.describe(floor_name, "floor", push_distance_cm=100.0 * float(chain["distance_m"]))
                return d
            errors.append("C: no (yaw, grasp) of F passes the exact IK at B and C")
        raise Reject("grasp_unreachable", "; ".join(errors[-3:]) if errors else "no placement candidate")

    # ------------------------------------------------------------------ push-under: the floor mesh and the deep trunks

    def _floor_relaxed(self):
        """Context: the floor object out of the collision world (the cuRobo ground slab stays) — the push-under stance C
        stands at the deep trunks, where the 1 cm margin against the floor mesh is what fails (see
        :meth:`push_under_candidates`); applied to the core's stance / transition re-check and to the program check."""
        return self.ikc.ignoring([self.obj("floor")]) if self.push_under else _NullContext()

    def check_stances_with_objects(self, draft: Draft):
        with self._floor_relaxed():
            super().check_stances_with_objects(draft)

    def program_check(self, draft: Draft) -> Dict[str, Any]:
        with self._floor_relaxed():
            return super().program_check(draft)

    # ------------------------------------------------------------------ goal (§5.6 / §6.12)

    def goal(self, draft: Draft):
        tname = self.tname
        t_pose = draft.object_poses[tname]
        goal = self.goal_pose_at(draft.summary["goal_cell"], t_pose[1])
        draft.goal_poses[tname] = goal
        draft.pose_targets[tname] = full_pose_target(goal[0], goal[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
        place = self.sidecar.stance("placement")
        self.sidecar.set_goal(place.base_pose2d, place.translate, place.tilt, arms="same")

    def after_settle(self, draft: Draft):
        tname = self.tname
        goal = self.goal_pose_at(draft.summary["goal_cell"], draft.object_poses[tname][1])
        draft.goal_poses[tname] = goal
        tgt = draft.pose_targets[tname]
        tgt["position"], tgt["orientation"] = [float(v) for v in goal[0]], [float(v) for v in goal[1]]

    def refine_targets_from_goal(self, draft: Draft):
        super().refine_targets_from_goal(draft)
        p, _ = self.partner.get_position_orientation()
        tgt = draft.pose_targets[self.pname]
        moved = float(th.linalg.norm(p.float() - th.tensor(tgt["position"])))
        if moved > self.num("partner_moved_max_m", 0.005):
            raise Reject("goal_partner_moved", f"partner moved {moved:.3f} m at the goal")

    def after_goal_check(self, draft: Draft):
        """Place / pre-place at the settled goal pose re-checked from C with the object attached (the record kept)."""
        tname = self.tname
        grasp = self.sidecar.grasp_of(tname)
        place = self.sidecar.stance("placement")
        goal_pos = th.tensor(draft.pose_targets[tname]["position"])
        goal_quat = th.tensor(draft.pose_targets[tname]["orientation"])
        ignore = [self.partner] if self.relation == "ontop" else [self.obj("floor")]
        if self.push_under:
            # the release is the drop outside the edge (unchanged by the settle): re-check it as recorded; the push goal blade
            # follows the settled goal centre (a few mm), the chain start → mid → goal is re-solved (closed gripper)
            with attached(self.kit, "right", self.target, eef_in_object=grasp.eef_in_object):
                res = self.ikc.exact([grasp.pre_place, grasp.place], place, "right", ignore_for_last=ignore, cache=False)
            if not res.ok:
                raise Reject("place_unreachable", f"pre-place / drop fail the exact IK from stance C at the goal check: {res.reason}")
            push = self.sidecar.contacts.get("push")
            if push is not None:
                d3 = th.as_tensor(push.direction, dtype=th.float32)
                off = float(push.extras["half_extent_m"]) + float(push.extras["blade_standoff_m"])
                z_push = float(push.contact[0][2])
                goal_b = th.tensor([float(goal_pos[0]), float(goal_pos[1]), z_push]) - d3 * off
                start_b = push.contact[0]
                mid_b = 0.5 * (start_b + goal_b)
                q = push.contact[1]
                with self.ikc.ignoring([self.obj("floor"), self.target]):
                    r3 = self.ikc.exact([push.pre, (start_b, q), (mid_b, q), (goal_b, q)], place, "right", cache=False, fingers=C.FINGER_CLOSED)
                if not r3.ok:
                    raise Reject("push_unreachable", f"push-under chain to the settled goal fails the exact IK from stance C: {r3.reason}")
                push.mid, push.goal = (mid_b, q), (goal_b, q)
                push.distance_m = float(th.linalg.norm(goal_pos[:2].float() - th.as_tensor(push.extras["drop_cell"], dtype=th.float32)))
                push.extras["goal_cell"] = [float(goal_pos[0]), float(goal_pos[1])]
        else:
            place_eef = IK.grasp_pose((goal_pos, goal_quat), grasp.eef_in_object)
            pre_place = IK.pre_pose(place_eef)
            with attached(self.kit, "right", self.target, eef_in_object=grasp.eef_in_object):
                res = self.ikc.exact([pre_place, place_eef], place, "right", ignore_for_last=ignore, cache=False)
            if not res.ok:
                raise Reject("place_unreachable", f"pre-place / place at the settled goal fail the exact IK from stance C: {res.reason}")
            grasp.pre_place, grasp.place = pre_place, place_eef
        cam_b = self.camera_pose_at(self.sidecar.stance("manipulation"))
        if not self.ikc.eef_in_frame(grasp.grasp, cam_b):
            raise Reject("gripper_not_in_frame", "grasp eef pose outside the manipulation image (5 %)")
        if self.relation == "ontop" and not self.ikc.eef_in_frame(place_eef, self.camera_pose_at(place)):
            raise Reject("gripper_not_in_frame", "place eef pose outside the placement image (5 %)")

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        g = self.sidecar.grasp_of(self.tname)
        place = self.sidecar.stances.get("placement")
        if g is None or place is None:
            return []
        c = self.choice.models["target"]
        return [{"scope": {"scene": self.scene, "partner": self.partner.name, "target": f"{c.category}/{c.model}"}, "type": "placement",
                 "edge": place.edge.name if place.edge else None, "stand_off": place.stand_off, "lateral": place.lateral, "trunk": [place.translate, round(place.tilt, 3)],
                 "goal_cell": draft.summary.get("goal_cell"), "grasp": [round(g.yaw, 2), g.grasp_name], "hard": False}]
