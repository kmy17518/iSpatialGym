"""IORS touching, large-scale (spec §5.6 "IORS large-scale", touching mode; owner 2026-10-12 ~16:05, touching_large#03).

The target ends on the floor touching a fixed scene partner (a bookcase): carried from its init surface to stance C, dropped on
the floor in front of the partner's face nearest the init pose and pushed along the floor against that face. :class:`TouchingMode`
is the mode ``IORSLarge._sample_touching`` returns (``_adopt_mode`` puts it in front of ``IORSLarge``; not registered itself). It
runs on the push-under machinery of ``iors_large`` (owner decision 2026-09-16 item 7: drop first, then the deep crouch with empty
hands, a straight floor push of the closed gripper from stance C, exact IK on the fly) and changes only what the relation needs:

* **The face** (:meth:`TouchingMode.touching_face`): per push lane λ along the partner's AABB edge, horizontal rays at the
  target's height band (a quarter, half and nine tenths of its height) and across its width find the first partner surface
  the target would meet - its offset ``f`` behind the AABB edge (a recessed plinth, the shelf board's front edge). A lane whose
  rays meet no partner surface (the target would slide under it) or another object first is not used.
* **Goal cell**: the target's footprint touching that face from outside, ``face - d (front + gap)`` with ``front`` the target's
  extent ahead of its root along the push direction ``d`` at the goal orientation (per yaw, :meth:`touching_goal_xy`) and
  ``gap`` = ``generation.touching_goal_gap_m`` (2 mm; the owner's 0-5 mm); the footprint lies fully outside the partner's and
  clear of every other object.
* **Drop cell**: the target's half extent + ``push_under_drop_extra_m`` (0.10) back from the face along ``d``, as the
  push-under drop.
* **Stance C**: the push-under's frontal / side stances at the face (the deep floor trunks, the drop footprint and the face's
  bottom point in frame), the lanes centred on the face so that the goal footprint stays in front of it; the nearest face that
  admits a stance (``generation.placement_pool_edges`` pools them as for ontop / under).
* **The push**: the push-under chain to the touching goal - floor and target ignored, the partner present (the IK's stopping
  contact for the target, never for the hand); recorded as ``contacts.push`` with ``kind: "push_touching"``, ``relation:
  "touching"``, ``partner``, ``face_point`` / ``face_offset_m`` / ``gap_m``. The oracle program
  (``programs.program_iors_large_touching``) pushes with ``Push(follow=partner)``: the pad stops when the target reaches its goal
  (``execute_push``'s stop rule), the partner may be touched.
* **Goal check**: the target nudged 1 mm at a time along ``d`` until ``touching(target, partner)`` holds (the small-scale
  touching rule, ``generation.nudge_m`` / ``max_nudges``); the TRO goal is the touching pose.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft, _jsonable
from ispatialgym.sampling.v2 import prompt0 as P0
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.sampling.v2.tasks import iors_large as IL

log = logging.getLogger("ispatialgym.sampling.v2")

#: ``contacts.push.kind`` of the floor push against the partner's face (``programs.program_iors_large_touching``)
PUSH_TOUCHING_KIND = "push_touching"
#: the goal's gap between the target's front face and the partner's face (owner 2026-10-12 ~16:05: 0-5 mm); the goal check
#: nudges the target on until the two touch
TOUCHING_GAP_M = 0.002
TOUCHING_GAP_MAX_M = 0.005
#: the face probe: ray heights as fractions of the target's height (none below 2 cm, where floor coverings lie) and lateral
#: offsets as fractions of its half extent across the lane; rays start this far outside the AABB edge
FACE_PROBE_HEIGHTS = (0.25, 0.5, 0.9)
FACE_PROBE_MIN_Z_M = 0.02
FACE_PROBE_LATERALS = (-1.0, -0.5, 0.0, 0.5, 1.0)
FACE_PROBE_OUTSIDE_M = 0.30
#: a face found deeper than this behind the AABB edge is a cavity the target would enter, not a face (``touching_face_max_inset_m``)
FACE_MAX_INSET_M = 0.10
#: lanes along the face: centred, this step; the goal footprint stays this far inside the face's ends
LANE_STEP_M = 0.10
LANE_END_MARGIN_M = 0.02
#: side stances: the base centre this far outside the face (a solid partner down to the floor keeps the base off its face;
#: the push-under's 0.35 / 0.40 stand under a bed / table edge)
SIDE_STANDOFFS_M = (0.45, 0.55)
#: the goal footprint clears every other object's AABB by this much
GOAL_CLEARANCE_M = 0.005


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (tested without a simulator)
# ----------------------------------------------------------------------------------------------------------------


def unit_xy(d) -> th.Tensor:
    v = th.as_tensor([float(d[0]), float(d[1])], dtype=th.float32)
    return v / max(float(th.linalg.norm(v)), 1e-9)


def extent_along(points_xy: th.Tensor, d) -> Tuple[float, float]:
    """``(back, front)``: how far points (xy relative to the object's root) reach behind / ahead of the root along ``d``."""
    proj = th.as_tensor(points_xy, dtype=th.float32)[:, :2] @ unit_xy(d)
    return float(-proj.min()), float(proj.max())


def touching_goal_xy(face_xy, d, front: float, gap: float) -> th.Tensor:
    """The root of a target whose front (``front`` ahead of the root along ``d``) stands ``gap`` short of the face point."""
    return th.as_tensor([float(face_xy[0]), float(face_xy[1])], dtype=th.float32) - unit_xy(d) * (float(front) + float(gap))


def touching_drop_xy(face_xy, d, half: float, extra: float) -> th.Tensor:
    """The drop centre: the target's half extent + ``extra`` back from the face point along ``d`` (the push-under drop)."""
    return th.as_tensor([float(face_xy[0]), float(face_xy[1])], dtype=th.float32) - unit_xy(d) * (float(half) + float(extra))


def face_from_hits(hits: Sequence[Tuple[float, bool, Optional[str]]], max_inset: float = FACE_MAX_INSET_M) -> Tuple[Optional[float], str]:
    """The face offset (m behind the AABB edge along the push) from the face probe's rays: ``hits`` = per ray ``(s, on_partner,
    name)`` (``s`` = the first hit's offset behind the edge, None = no hit). The face is the partner hit nearest the edge (what
    the target meets first); ``(None, why)`` when no ray meets the partner, when another object is met before (or within 5 mm
    of) it, or when it lies deeper than ``max_inset`` (a cavity)."""
    on = [float(s) for s, p, _ in hits if s is not None and p]
    if not on:
        return None, "no_face: no ray at the target's height meets the partner (the target would pass under it)"
    f = min(on)
    first = [(float(s), n) for s, p, n in hits if s is not None and not p and float(s) < f + 0.005]
    if first:
        s, n = min(first)
        return None, f"face_blocked: {n} met {f - s:.3f} m before the partner's face"
    if f > float(max_inset):
        return None, f"face_recessed: the partner's surface lies {f:.2f} m behind its AABB edge"
    return f, ""


def centred_lanes(half_run: float, step: float = LANE_STEP_M) -> List[float]:
    """Lane offsets ``0, ±step, ±2 step, ...`` within ``±half_run`` (centre first)."""
    if half_run < 0:
        return []
    n = int(math.floor(float(half_run) / float(step) + 1e-9))
    out = [0.0]
    for k in range(1, n + 1):
        out += [round(k * step, 6), round(-k * step, 6)]
    return out


def square_hits_box(c, half: float, lo, hi, margin: float = 0.0) -> bool:
    """Whether the axis-aligned square of half size ``half`` around ``c`` overlaps the box ``[lo, hi]`` grown by ``margin``."""
    return (float(c[0]) + half > float(lo[0]) - margin and float(c[0]) - half < float(hi[0]) + margin
            and float(c[1]) + half > float(lo[1]) - margin and float(c[1]) - half < float(hi[1]) + margin)


class PushTouchingRecord(IL.PushUnderRecord):
    """``contacts.push`` of the touching push (the push-under record's fields): ``kind: "push_touching"``, ``relation:
    "touching"``, ``partner`` (the instance the target ends touching; the oracle's push follows it and may touch it)."""

    def to_sidecar(self) -> Dict[str, Any]:
        d = super().to_sidecar()
        d["kind"] = PUSH_TOUCHING_KIND
        d["relation"] = "touching"
        return d

    @classmethod
    def from_push_under(cls, rec: IL.PushUnderRecord, **extras) -> "PushTouchingRecord":
        out = cls(rec.arm, rec.pre, rec.contact, rec.mid, rec.goal, rec.direction, float(rec.distance_m), rec.object, stance=rec.stance)
        out.extras = dict(rec.extras)
        out.extras.update(extras)
        return out


# ----------------------------------------------------------------------------------------------------------------
# the task module
# ----------------------------------------------------------------------------------------------------------------


class TouchingMode:
    """The touching mode of ``IORSLarge`` (module docstring): the push-under drop + floor push aimed at the partner's face. A mixin
    (``_adopt_mode``): ``super()`` reaches ``IORSLarge``."""

    @property
    def push_under(self) -> bool:
        """The drop + straight floor push of the push-under, toward the partner's face (``generation.under_mode`` is not read)."""
        return True

    @property
    def touching_gap(self) -> float:
        return min(max(float(self.num("touching_goal_gap_m", TOUCHING_GAP_M)), 0.0), TOUCHING_GAP_MAX_M)

    def _floor_relaxed(self):
        """The floor mesh out of the collision world (the cuRobo ground slab stays) for the stance / transition / program checks, as
        for the push-under: the drop and push stances C stand at the deep floor trunks (``IORSLarge._floor_relaxed`` keys it on the
        under relation and a floor init)."""
        return self.ikc.ignoring([self.obj("floor")])

    def init_support_fallback(self, kind: str) -> Optional[Dict[str, Any]]:
        """``generation.init_support_fallback`` (iors_large 2b) falls back to the floor init, as ontop's: the target ends on the floor,
        so the partner's own top (under's fallback) is no init surface for it."""
        fb = super().init_support_fallback(kind)
        return None if fb is None else dict(fb, to="floor")

    # ------------------------------------------------------------------ partner geometry for C

    def prepare_partner(self):
        """The partner's AABB edges (the faces C may stand at); the goal cells are found per lane by the face probe."""
        partner = self.partner
        half, height = self.target_dims()
        lo, hi = partner.aabb
        self.place = {"cells": th.zeros(0, 2), "edges": IL.aabb_edges(lo, hi), "z": self.floor_z, "height": height, "half": half, "clearance": None,
                      "need": 0.0, "faces": {}}
        self.template_facts["partner"] = {"name": partner.name, "relation": "touching", "edges": [e.to_json() for e in self.place["edges"]],
                                          "aabb": [[round(float(v), 3) for v in lo.tolist()], [round(float(v), 3) for v in hi.tolist()]]}

    def placement_candidates(self, init_xy, arm: Optional[str] = None) -> List[Stance]:
        """Stance C at the partner's face nearest the init pose that admits one (owner 2026-10-12: "the partner's face nearest the
        init"); ``generation.placement_pool_edges`` true pools every face's survivors as ontop / under do."""
        if self.gp.gen("placement_pool_edges") is not None:
            return super().placement_candidates(init_xy, arm)
        arm = arm or self.seg_arm
        p = th.as_tensor([float(init_xy[0]), float(init_xy[1])])
        edges = sorted(self.place["edges"], key=lambda e: float(th.linalg.norm(e.mid - p)))
        allow = self.gp.gen("placement_edges")
        if allow:
            edges = [e for e in edges if e.name in set(allow)] or edges
        tried = []
        for rank, edge in enumerate(edges):
            out = self._placement_candidates_at(edge, init_xy, arm)
            if out:
                for st in out:
                    st.extras["edge_rank"] = rank
                if tried:
                    log.info("%s: touching stance C: face(s) %s nearest the init pose admit no stance; %s taken", self.gp.alias, tried, edge.name)
                return out
            tried.append(edge.name)
        return []

    def touching_face(self, edge: EdgeFrame, lat: float) -> Tuple[Optional[float], str, Dict[str, Any]]:
        """The face probe of lane ``lat`` on ``edge`` (module docstring): ``(offset behind the AABB edge, why, facts)``; memoised per
        (edge, lane) for the template."""
        key = (edge.name, round(float(lat), 4))
        memo = self.place.setdefault("faces", {})
        if key in memo:
            return memo[key]
        from ispatialgym.sampling.sim_utils import raytest_batch

        half, height = float(self.place["half"]), float(self.place["height"])
        d2 = unit_xy(edge.inward)
        tangent = th.tensor([-float(d2[1]), float(d2[0])])
        lo, hi = self.partner.aabb
        depth = float(th.abs(d2 @ (hi[:2].float() - lo[:2].float())))
        e0 = edge.mid.float() + tangent * float(lat)
        w = max(half - 0.01, 0.0)
        starts, ends = [], []
        for fz in FACE_PROBE_HEIGHTS:
            z = self.floor_z + max(FACE_PROBE_MIN_Z_M, float(fz) * height)
            for fl in FACE_PROBE_LATERALS:
                p = e0 + tangent * (float(fl) * w)
                a, b = p - d2 * FACE_PROBE_OUTSIDE_M, p + d2 * (depth + 0.02)
                starts.append([float(a[0]), float(a[1]), z])
                ends.append([float(b[0]), float(b[1]), z])
        ignore = [l.prim_path for o in (self.robot, self.target) for l in getattr(o, "links", {}).values()]
        raw = raytest_batch(starts, ends, only_closest=True, ignore_bodies=ignore or None)
        prims = {l.prim_path for l in self.partner.links.values()}
        hits: List[Tuple[Optional[float], bool, Optional[str]]] = []
        for h in raw:
            if not h.get("hit"):
                hits.append((None, False, None))
                continue
            body = str(h.get("rigidBody"))
            s = float((th.as_tensor([float(h["position"][0]), float(h["position"][1])]) - edge.mid.float()) @ d2)
            on = body in prims or any(body.startswith(pp + "/") for pp in prims)
            hits.append((s, on, None if on else (self._object_of_prim(body) or body)))
        f, why = face_from_hits(hits, float(self.num("touching_face_max_inset_m", FACE_MAX_INSET_M)))
        facts = {"n_rays": len(hits), "n_partner": sum(1 for s, p, _ in hits if s is not None and p), "n_miss": sum(1 for s, _, _ in hits if s is None)}
        if f is not None:
            facts["offset_m"] = round(f, 4)
        memo[key] = (f, why, facts)
        return memo[key]

    def touching_goal_free(self, g: th.Tensor, half: float) -> Optional[str]:
        """Why the goal footprint (a square of the half extent around ``g``) is not free floor: outside the task rooms or over an
        object's AABB (the partner excluded: the footprint lies in front of its face by construction); None when free."""
        if not P0.in_task_rooms(self, g):
            return "outside_rooms"
        aabbs = self.__dict__.get("_touching_aabbs")
        if aabbs is None:
            aabbs = [(o.name, o.aabb[0][:2].float(), o.aabb[1][:2].float()) for o in self.floor_blocking_objects() if o is not self.partner]
            self._touching_aabbs = aabbs
        for n, a, b in aabbs:
            if square_hits_box(g, half, a, b, GOAL_CLEARANCE_M):
                return f"aabb:{n}"
        return None

    def push_under_candidates(self, edge: EdgeFrame, init_xy, arm: Optional[str] = None) -> List[Stance]:
        """Stance C of the touching push (``iors_large.push_under_candidates`` with the goal outside the face): per lane λ (centred,
        the goal footprint within the face's ends) the face point from :meth:`touching_face`, the nominal goal cell (refined per yaw
        by the chain) on free floor, the drop cell on free floor; frontal stances square to the face at the push-under stand-offs
        (the drop cell 0.25-0.6 m ahead) with the lane to the arm's side, side stances beside the lane; the deep floor trunks,
        whole-body collision-free with the floor mesh out, the drop footprint (+ the face's bottom point at the lane, frontal) in
        frame, the optical axis within ``push_under_axis_max_deg`` of the drop cell. Extras as the push-under's + ``face_point``,
        ``face_offset_m``, ``gap_m``."""
        arm = arm or self.seg_arm
        partner = self.partner
        half, height = float(self.place["half"]), float(self.place["height"])
        d2 = unit_xy(edge.inward)
        tangent = th.tensor([-float(d2[1]), float(d2[0])])
        mid = edge.mid.float()
        gap = self.touching_gap
        extra = float(self.num("push_under_drop_extra_m", IL.PUSH_UNDER_DROP_EXTRA_M))
        lanes = centred_lanes(edge.length / 2 - half - LANE_END_MARGIN_M)
        stand_offs = [float(v) for v in (self.gp.gen("push_under_stand_offs_m") or IL.PUSH_UNDER_STAND_OFFS)]
        trunks = [tuple(t) for t in (self.gp.gen("floor_trunks") or IL.FLOOR_TRUNKS)]
        lo_ahead, hi_ahead = IL.PUSH_UNDER_DROP_AHEAD_M
        side_ahead = [float(v) for v in (self.gp.gen("push_under_side_ahead_m") or (0.45, 0.55, 0.6))]
        side_b = [float(v) for v in (self.gp.gen("push_under_side_standoff_m") or SIDE_STANDOFFS_M)]
        stats: Dict[str, Any] = {"lanes": len(lanes), "face": 0, "face_why": {}, "goal_free": 0, "goal_blocked": {}, "drop_free": 0, "drop_blocked": {}, "base_ok": 0,
                                 "candidates": 0, "side_candidates": 0, "edge": edge.name, "edge_mid": [round(float(v), 2) for v in mid.tolist()],
                                 "inward": [round(float(v), 2) for v in d2.tolist()], "gap_m": gap}
        cands: List[Stance] = []
        for lat in lanes:
            f, why, facts = self.touching_face(edge, lat)
            if f is None:
                k = why.split(":")[0]
                stats["face_why"][k] = stats["face_why"].get(k, 0) + 1
                stats.setdefault("face_why_example", why)
                continue
            stats["face"] += 1
            face = mid + tangent * float(lat) + d2 * float(f)
            g = touching_goal_xy(face, d2, half, gap)
            blocked = self.touching_goal_free(g, half)
            if blocked is not None:
                stats["goal_blocked"][blocked] = stats["goal_blocked"].get(blocked, 0) + 1
                continue
            stats["goal_free"] += 1
            c_d = touching_drop_xy(face, d2, half, extra)
            if not self._free_floor_cell(c_d, stats["drop_blocked"]):
                continue
            stats["drop_free"] += 1
            out = half + extra - float(f)  # the drop cell outside the AABB edge line (the stances stand off from that line)
            common = {"goal_cell": [float(g[0]), float(g[1])], "drop_cell": [float(c_d[0]), float(c_d[1])], "push_dir": [float(d2[0]), float(d2[1])],
                      "face_point": [float(face[0]), float(face[1])], "face_offset_m": float(f), "face_probe": facts, "gap_m": gap, "edge": edge.name}
            for off in [float(v) for v in (self.gp.gen("frontal_lane_offset_m") or (0.25, 0.0))]:
                for so in stand_offs:
                    ahead = so - out
                    if ahead < lo_ahead or ahead > hi_ahead:
                        continue
                    lat_b = IL.frontal_lane_lateral(float(lat), off, arm)
                    xy = edge.base_xy(so, lat_b)
                    if not P0.base_ok(self, xy):
                        continue
                    stats["base_ok"] += 1
                    for tr, ti in trunks:
                        st = Stance("placement", (float(xy[0]), float(xy[1]), edge.yaw), float(tr), float(ti), so, edge, None, lat_b, None, [arm])
                        st.focus = th.tensor([float(c_d[0]), float(c_d[1]), self.floor_z + height / 2])
                        st.extras.update(common, drop_ahead_m=float(ahead), stance_kind="frontal", lane_offset_m=IL.arm_sign(arm) * off)
                        cands.append(st)
                        stats["candidates"] += 1
            for x_ahead in side_ahead:
                for b in side_b:
                    x_b, y_b, yaw_b = IL.side_push_pose(mid, tangent, d2, float(lat), x_ahead, b, arm)
                    xy = th.tensor([x_b, y_b])
                    if not P0.base_ok(self, xy):
                        continue
                    stats["base_ok"] += 1
                    for tr, ti in trunks:
                        st = Stance("placement", (x_b, y_b, yaw_b), float(tr), float(ti), b, None, None, float(lat), None, [arm])
                        st.focus = th.tensor([float(c_d[0]), float(c_d[1]), self.floor_z + height / 2])
                        st.extras.update(common, drop_ahead_m=float(x_ahead), stance_kind="side", side_standoff_m=b)
                        cands.append(st)
                        stats["side_candidates"] += 1
        self.template_facts["placement_stats"] = stats
        if not cands:
            log.info("%s: touching stance C at %s face %s: no candidate (%s)", self.gp.alias, partner.name, edge.name, stats)
            return []
        with self.ikc.ignoring([self.obj("floor")]):
            free = P0.body_free(self, cands)
        cands = [st for st, ok in zip(cands, free.tolist()) if ok]
        stats["collision_free"] = len(cands)
        if not cands:
            log.info("%s: touching stance C at %s face %s: every candidate collides (%s)", self.gp.alias, partner.name, edge.name, stats)
            return []
        cams = P0.camera_poses(self, cands)
        z_bottom = float(partner.aabb[0][2])
        axis_max = math.radians(float(self.num("push_under_axis_max_deg", 45.0)))
        classed = []
        for st, cam in zip(cands, cams):
            c_d = th.as_tensor(st.extras["drop_cell"])
            disc = th.stack([th.cat([c_d + tangent * half, th.tensor([self.floor_z])]), th.cat([c_d - tangent * half, th.tensor([self.floor_z])]),
                             th.cat([c_d + d2 * half, th.tensor([self.floor_z])]), th.cat([c_d - d2 * half, th.tensor([self.floor_z])])])
            fp = th.as_tensor(st.extras["face_point"])
            kind = st.extras.get("stance_kind", "frontal")
            pts = disc if kind == "side" else th.cat([disc, th.cat([fp, th.tensor([z_bottom])]).reshape(1, 3)])
            if not P0.in_frame(self, cam, pts):
                stats[f"{kind}_out_of_frame"] = stats.get(f"{kind}_out_of_frame", 0) + 1
                continue
            if P0.axis_angle(self, cam, st.focus) > axis_max:
                stats[f"{kind}_axis_off"] = stats.get(f"{kind}_axis_off", 0) + 1
                continue
            st.extras["angle_class"] = 1
            classed.append((st, 1))
        stats["visible"] = len(classed)
        out_ = P0.best_class(classed)
        stats["survivor_kinds"] = {k: sum(1 for st in out_ if st.extras.get("stance_kind") == k) for k in ("side", "frontal")}
        stats["survivor_trunks"] = sorted({(float(st.translate), round(float(st.tilt), 2)) for st in out_})
        log.info("%s: touching stance C at %s face %s: %d survivors (%s)", self.gp.alias, partner.name, edge.name, len(out_), stats)
        return out_

    # ------------------------------------------------------------------ the push chain and its records

    def touching_goal_at(self, face_xy, d2, quat) -> th.Tensor:
        """The goal cell at orientation ``quat``: the target's front at the goal orientation ``gap`` short of the face point."""
        pts = G.local_points_world(self.hull, (th.zeros(3), G.t(quat)))[:, :2]
        _, front = extent_along(pts, d2)
        return touching_goal_xy(face_xy, d2, front, self.touching_gap)

    def _push_under_chain(self, place: Stance, quat, drop_xy, goal_xy, d2) -> Optional[Dict[str, Any]]:
        """The push-under chain to the touching goal of this yaw (:meth:`touching_goal_at` at the C candidate's face point); the
        passed (nominal) goal cell is replaced. The chain facts carry ``goal_cell``."""
        face = place.extras.get("face_point")
        if face is None:
            return super()._push_under_chain(place, quat, drop_xy, goal_xy, d2)
        g = self.touching_goal_at(face, d2, quat)
        if float((g - th.as_tensor([float(drop_xy[0]), float(drop_xy[1])], dtype=th.float32)) @ unit_xy(d2)) < 0.01:
            return None  # at this yaw the target's front reaches the face from the drop cell: nothing to push
        ch = super()._push_under_chain(place, quat, drop_xy, g, d2)
        if ch is not None:
            ch["goal_cell"] = [float(g[0]), float(g[1])]
        return ch

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        d = super().sample_instance(rng)
        self.touching_records(d)
        return d

    def touching_records(self, d: Draft) -> None:
        """The push-under records of the coupling turned into the touching ones: the goal cell at the drawn goal orientation
        (the chain was solved for it), ``contacts.push`` as a :class:`PushTouchingRecord`, ``grasps[0].extras.then`` /
        ``goal_cell``, ``summary.push_touching``."""
        rec = self.sidecar.contacts.get("push")
        place = self.sidecar.stance("placement")
        face, d2 = place.extras.get("face_point"), place.extras.get("push_dir")
        if rec is None or face is None or d2 is None:
            raise Reject("spec", f"{self.gp.alias}: the touching coupling recorded no push / face point")
        gc = self.touching_goal_at(face, d2, self.goal_quat_of(d))
        drop = th.as_tensor(rec.extras.get("drop_cell"), dtype=th.float32)
        dist = float(th.linalg.norm(gc - drop))
        new = PushTouchingRecord.from_push_under(rec, goal_cell=[float(gc[0]), float(gc[1])], relation="touching", partner=self.pname,
                                                 face_point=[float(v) for v in face], face_offset_m=place.extras.get("face_offset_m"),
                                                 gap_m=place.extras.get("gap_m"), edge=place.extras.get("edge"))
        new.distance_m = dist
        self.sidecar.add_contact(new)
        g = self.sidecar.grasp_of(self.tname)
        if g is not None:
            g.extras.update({"then": PUSH_TOUCHING_KIND, "goal_cell": [float(gc[0]), float(gc[1])]})
        d.summary["goal_cell"] = [float(gc[0]), float(gc[1])]
        pu = d.summary.pop("push_under", None) or {}
        d.summary["push_touching"] = dict(pu, distance_m=dist, goal_cell=[float(gc[0]), float(gc[1])], face_point=[float(v) for v in face],
                                          face_offset_m=place.extras.get("face_offset_m"), gap_m=place.extras.get("gap_m"), edge=place.extras.get("edge"),
                                          partner=self.pname)
        floor = self.bindings.one("floor").name if self.bindings.has("floor") else None
        if floor:
            self.sidecar.describe(floor, "floor", push_distance_cm=100.0 * dist)
        log.info("%s: touching push %.3f m from the drop %s to the goal %s against %s's face %s (offset %.3f m behind its AABB edge, gap %.0f mm)", self.gp.alias,
                 dist, [round(float(v), 3) for v in drop.tolist()], [round(float(v), 3) for v in gc.tolist()], self.pname, place.extras.get("edge"),
                 float(place.extras.get("face_offset_m") or 0.0), 1000.0 * float(place.extras.get("gap_m") or 0.0))

    # ------------------------------------------------------------------ goal (touching)

    def refine_targets_from_goal(self, draft: Draft):
        self.nudge_until_touching(draft)
        super().refine_targets_from_goal(draft)

    def nudge_until_touching(self, draft: Draft) -> None:
        """The settled goal nudged ``generation.nudge_m`` (1 mm) at a time along the push until ``touching(target, partner)``
        holds, at most ``generation.max_nudges`` (25) times (the small-scale touching rule); ``summary.touching_nudge``."""
        from omnigibson.object_states import Touching

        t, p = self.target, self.partner
        push = self.sidecar.contacts.get("push")
        d = unit_xy(push.direction if push is not None else self.sidecar.stance("placement").extras["push_dir"])
        step, n = float(self.num("nudge_m", 0.001)), int(self.num("max_nudges", 25))
        for k in range(n + 1):
            self.world.wake([t, p])
            if t.states[Touching].get_value(p):
                draft.summary["touching_nudge"] = {"steps": k, "m": round(k * step, 4)}
                return
            if k == n:
                break
            pos, quat = t.get_position_orientation()
            t.set_position_orientation(pos + th.tensor([float(d[0]) * step, float(d[1]) * step, 0.0]), quat)
            t.keep_still()
            self.world.step_physics(3, still=[self.robot, p])
        raise Reject("goal_infeasible", f"touching: no contact with {p.name} after {n} nudges of {1000 * step:.0f} mm along the push")

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        out = super().record_constraints(draft)
        for r in out:
            if r.get("type") == "placement":
                r["relation"] = "touching"
                r["face"] = _jsonable((draft.summary.get("push_touching") or {}).get("edge"))
        return out
