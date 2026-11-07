"""IORS nextto, large-scale (spec §5.6 "nextto (large-scale)"; owner 2026-10-12 nextto_large item 3): a floor-standing spawned target
pushed next to a fixed partner by the base - no grasp. :class:`NextToMode` is the mode of
:class:`~ispatialgym.sampling.v2.tasks.iors_large.IORSLarge` for the relation ``nextto`` (its hook ``_sample_nextto`` returns it).

One draw per attempt (generation keys in :data:`DEFAULTS`, read by this mode only):

* **goal**: the target beside one side of the partner's AABB (outward normal n), the push direction d = -n turned by one of
  ``push_angles_deg``, the target's yaw set so that one of its four side faces (``FACES``; the faces across its longer horizontal
  extent first - the longer base along the push tips less) faces the robot; its AABB ``goal_gap`` from the partner's (``goal_gap_frac``
  of OmniGibson's NextTo bound mean(dims_a + dims_b) / 6, clamped to [``goal_gap_min_m``, ``goal_gap_max_m``]), its centre within the
  side's span less ``goal_lateral_margin_m``, its footprint on free floor ``floor_clearance_m`` clear of every obstacle; NextTo read
  true there and at 3 of the 4 probes ±``nextto_probe_lateral_m`` / ±``nextto_probe_yaw_deg`` (OmniGibson's ``NextTo`` state, the one
  the goal check reads; unreadable = left to the goal check).
* **init**: ``distance`` ~ U(``push_distance_m``) back along d: footprint on free floor with the clearance, the AABB gap beyond the
  NextTo bound + ``init_gap_extra_m`` (NextTo false), the footprint swept init -> goal every ``sweep_step_m`` free (the partner never
  overlapped) and, with ``push_single_floor`` (default true), on one ``floors`` object: a downward ray at its corners and centre at
  every sweep step hits the same floor object (a seam between two floor objects catches a pushed object's feet; any other first hit,
  e.g. a rug, counts as a different floor).
* **push face**: the target's convex collision pieces cut at each of ``push_heights_m`` above the floor; the outermost material
  on the face's centre line, flat across the closed hand (probes ±``face_probe_m``: lateral, vertical; spread <=
  ``face_flat_tol_m``) and at most ``face_max_inset_m`` inside the AABB; when no height passes, twice the flatness and 10 cm more inset
  (``face_relaxed``: the guarded approach finds the actual contact).
* **push stance** ``manipulation``: base behind the target facing d, the push face ``push_stand_offs_m`` ahead, shifted
  ``push_lateral_offsets_m`` to the robot's left (the right arm's hand on the target's centre line; mirrored for the left arm), trunk from
  ``push_trunks``; shoulder reach margin within ``push_reach_margin_m``, the contact fingertip in the head-camera frame, whole body
  collision-free with the target placed; then the exact IK (up to ``push_ik_max`` stances): the closed hand's pre-contact
  ``push_pre_m`` behind the contact (the target present), the straight approach to the contact - fingertips on the face, approach
  pitched down by one of ``push_pitches_deg``, either wrist turn - with the target ignored at the contact only (``program_check``'s
  chain), the fingertips of the pre-contact and the approach in the camera frame, the whole body swept along the line with the arm at
  the contact configuration and the target out of the world.
* **two hands** (``push_hands`` 2, owner 2026-10-12 ~20:17 item C; default 1 = the one hand above): two push spots on the face,
  symmetric about its centre line ``push_hand_spacings_m`` apart, each probed like the centre line (flat across the closed hand, inset,
  the two within ``push_hands_depth_tol_m`` along the normal), at the same ``push_heights_m``; the base centred behind their midpoint
  (no lateral shift), each hand on the spot on its own side; both reach margins and both contacts in the camera frame; the first arm's
  chain as above, then the second arm's as the second-arm rule solves a brace (``program_check.second_hand_chain``: its pre-contact with
  the first hand at its pre-contact, its approach with the first hand locked at its contact, both hands at contact whole-body free);
  the sweep with both arms at their contact configuration. A face without two spots falls back to the one hand
  (``two_hand_fallback``).
* **push end** ``placement``: the push stance moved ``distance`` along d (trunk, arms kept), collision-free with the target at init.
* **init stance** ``init``: free floor ``init_distance_m`` from the target within ``init_bearing_max_deg`` of the line's back direction,
  facing it, the most upright straight trunk that sees the target and the goal (its footprint centre, the partner's side) in frame;
  IORS large-scale's A -> B leg (path + transition) to the push stance with the target placed.

``contacts.push`` (``kind: "base_push"``): ``pre`` / ``init`` (contact) / ``goal`` (= contact + distance x d) eef poses, ``direction``,
``distance_m`` and the extras ``base_start`` / ``base_end``, ``push_height_m``, ``stand_off_m``, ``lateral_offset_m``, ``contact_point``,
``contact_in_object`` (the contact point in the target's root frame), ``face``, ``face_inset_m``, ``pitch_deg``, ``partner``, ``end_stance``,
``goal_gap_m``, ``nextto_bound_m``, ``drift_tol_m``, ``yaw_tol_rad``, ``nextto_push_on_m``, ``push_speed_m_s``, ``ignore_last`` / ``ignore_every``
(the program check's ignores: the target at the contact), ``push_hands``; with two hands also ``second_hand`` (the other arm's
``arm`` / ``pre`` / ``init`` / ``goal`` poses, ``contact_point``, ``contact_in_object``, ``pitch_deg``, ``wrist_sign``, its spot's inset /
spread), ``hand_spacing_m`` and ``face_centre_in_object`` (the midpoint of the two spots). The oracle's ``BasePush`` reads them.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.eef_reach_volume import TRANSLATES
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft, _jsonable
from ispatialgym.sampling.tro_writer import full_pose_target
from ispatialgym.sampling.v2 import contact_poses as CPZ
from ispatialgym.sampling.v2 import navigation as NAV
from ispatialgym.sampling.v2 import program_check as PC
from ispatialgym.sampling.v2 import prompt0 as P0
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, Stance, pose_to_sidecar

log = logging.getLogger("ispatialgym.sampling.v2")

#: ``contacts.push.kind`` of the base push (``oracle.actions.BASE_PUSH_KIND``)
BASE_PUSH_KIND = "base_push"
#: the target's four side faces: outward normal in its root frame
FACES: Dict[str, Tuple[float, float]] = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}
#: the stance where the push ends (``contacts.push.end_stance``)
END_STANCE = "placement"

#: generation keys of the nextto mode with their defaults (no other relation reads them)
DEFAULTS: Dict[str, Any] = {
    "push_distance_m": [1.5, 2.5],
    "push_angles_deg": [0.0, 15.0, -15.0],
    "goal_gap_frac": 0.45,
    "goal_gap_min_m": 0.06,
    "goal_gap_max_m": 0.15,
    "goal_lateral_margin_m": 0.10,
    "floor_clearance_m": 0.05,
    "init_gap_extra_m": 0.20,
    "sweep_step_m": 0.05,
    "push_single_floor": True,
    "push_heights_m": [0.45, 0.40, 0.50, 0.55, 0.35, 0.60],
    "face_probe_m": [0.03, 0.02],
    "face_flat_tol_m": 0.03,
    "face_max_inset_m": 0.15,
    "push_stand_offs_m": [0.55, 0.50, 0.60, 0.65, 0.45],
    "push_lateral_offsets_m": [0.15, 0.08, 0.0, 0.22],
    "push_trunks": [[0.5, 0.0], [0.75, 0.0], [0.5, -0.3], [0.75, -0.3], [0.95, 0.0], [0.95, -0.3], [0.25, 0.0], [0.5, -0.6]],
    "push_pitches_deg": [0.0, 20.0, 35.0],
    "push_pre_m": 0.10,
    "push_reach_margin_m": [0.03, 0.45],
    "push_ik_max": 16,
    "drift_tol_m": 0.10,
    "yaw_tol_rad": 0.61,
    "nextto_push_on_m": 0.05,
    "push_speed_m_s": 0.08,
    "init_distance_m": [1.4, 2.4],
    "init_bearing_max_deg": 60.0,
    "nextto_probe_lateral_m": 0.03,
    "nextto_probe_yaw_deg": 3.0,
    "goal_tries": 24,
    "distance_draws": 3,
    "push_hands": 1,
    "push_hand_spacings_m": [0.30, 0.26, 0.34, 0.22],
    "push_hands_depth_tol_m": 0.03,
    # oracle-fix 2026-11-03 (touching_large#01, the ottoman): the same base push against a large partner's face until TOUCHING
    # (``generation.touching_push: "base"`` on a touching_large alias, :class:`TouchingBasePushMode`): the goal's AABB gap from the partner
    # (2 mm short of contact, the storage box's touching_goal_gap_m; the goal check nudges it on to contact), the drive's push-on after
    # Touching first holds, and the goal check's nudge (1 mm steps, at most 25)
    "touching_goal_gap_m": 0.002,
    # ... and the floor test of a touching footprint skips its points within this of the partner's AABB: the goal 2 mm off the partner
    # lies in the partner's own cells of the traversability map (0.1 m cells; the first ottoman run: every goal 'not_floor'); the
    # footprint still may not overlap the partner's AABB
    "touching_floor_skip_m": 0.15,
    "touching_push_on_m": 0.0,
    # ... and (opt-in, a sectional / chaise sofa whose AABB face lies far from its material, oracle-fix 2026-11-03 22:55: sofa_hiphpn_0's
    # seat front 0.5 m inside the AABB's chaise tip) the touching goal advanced along the push from the AABB face to touching_goal_gap_m
    # short of the partner's material (its collision pieces cut at ``touching_section_heights`` fractions of the target's height), at
    # most touching_advance_max_m; the partner test of every footprint (goal, start, sweep) then uses those sections, not the AABB
    "touching_partner_surface": False,
    # oracle-fix 2026-11-04 (owner item 1b): how many push spots / two-hand pairs per face the stance searches try (the first ones of
    # _face_candidates / _face_pairs: the probe's best heights); a model with a push-spot annotation tries every annotated one
    "push_face_spots_max": 2,
    "touching_section_heights": [0.1, 0.35, 0.6, 0.85],
    "touching_advance_max_m": 1.0,
    "nudge_m": 0.001,
    "max_nudges": 25,
}


def knob(s, key: str) -> Any:
    """``generation.<key>`` of the variant, else :data:`DEFAULTS`."""
    v = s.gp.gen(key) if s is not None else None
    return DEFAULTS[key] if v is None else v


def push_relation(s) -> str:
    """The relation the base push establishes: ``touching`` for :class:`TouchingBasePushMode` (a touching_large alias), else ``nextto``."""
    return "touching" if getattr(s, "relation", None) == "touching" else "nextto"


def goal_gap(s, bound: float) -> float:
    """The goal's AABB gap from the partner: nextto - ``goal_gap_frac`` of the NextTo bound clamped to [``goal_gap_min_m``,
    ``goal_gap_max_m``]; touching - ``touching_goal_gap_m`` (the goal check nudges the target on to contact)."""
    if push_relation(s) == "touching":
        return max(float(knob(s, "touching_goal_gap_m")), 0.0)
    return min(max(float(knob(s, "goal_gap_frac")) * bound, float(knob(s, "goal_gap_min_m"))), float(knob(s, "goal_gap_max_m")))


# ----------------------------------------------------------------------------------------------------------------
# pure geometry (tested without a simulator)
# ----------------------------------------------------------------------------------------------------------------


def nextto_bound(lo_a, hi_a, lo_b, hi_b) -> float:
    """OmniGibson's NextTo distance bound (``object_states.next_to``): mean over the three axes of (dim_a + dim_b), over 6."""
    da = th.as_tensor(hi_a, dtype=th.float32) - th.as_tensor(lo_a, dtype=th.float32)
    db = th.as_tensor(hi_b, dtype=th.float32) - th.as_tensor(lo_b, dtype=th.float32)
    return float((da + db).sum()) / 18.0


def aabb_gap(lo_a, hi_a, lo_b, hi_b) -> float:
    """The AABB gap NextTo compares with its bound: the norm of the per-axis gaps max(0, max(lo) - min(hi))."""
    lo_a, hi_a, lo_b, hi_b = (th.as_tensor(v, dtype=th.float32) for v in (lo_a, hi_a, lo_b, hi_b))
    g = th.clamp(th.maximum(lo_a, lo_b) - th.minimum(hi_a, hi_b), min=0.0)
    return float(th.linalg.norm(g))


def rot2(yaw: float) -> th.Tensor:
    c, s_ = math.cos(yaw), math.sin(yaw)
    return th.tensor([[c, -s_], [s_, c]])


def footprint(lo, hi, xy, yaw: float, grow: float = 0.0) -> th.Tensor:
    """The 4 corners (CCW, world xy) of the target's local xy box ``lo..hi`` (root frame) at root ``xy`` / ``yaw``, grown by ``grow``."""
    x0, y0, x1, y1 = float(lo[0]) - grow, float(lo[1]) - grow, float(hi[0]) + grow, float(hi[1]) + grow
    loc = th.tensor([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return loc @ rot2(yaw).T + th.as_tensor([float(xy[0]), float(xy[1])])


def footprint_aabb(lo, hi, xy, yaw: float) -> Tuple[th.Tensor, th.Tensor]:
    c = footprint(lo, hi, xy, yaw)
    return c.min(dim=0).values, c.max(dim=0).values


def world_aabb(lo, hi, xy, yaw: float, z_bottom: float) -> Tuple[th.Tensor, th.Tensor]:
    """The target's world AABB at root ``xy`` / ``yaw`` standing upright with its lowest point at ``z_bottom``."""
    l2, h2 = footprint_aabb(lo, hi, xy, yaw)
    h = float(hi[2]) - float(lo[2])
    return th.tensor([float(l2[0]), float(l2[1]), z_bottom]), th.tensor([float(h2[0]), float(h2[1]), z_bottom + h])


def face_yaw(face: str, direction_xy) -> float:
    """The target's yaw that turns ``face``'s outward normal to -``direction`` (the robot behind that face pushes along ``direction``)."""
    nx, ny = FACES[face]
    return C.wrap_angle(math.atan2(float(direction_xy[1]), float(direction_xy[0])) - math.atan2(-ny, -nx))


def face_order(lo, hi) -> List[str]:
    """The side faces, those across the target's longer horizontal extent first (the push then runs along it), + before -."""
    ex, ey = float(hi[0]) - float(lo[0]), float(hi[1]) - float(lo[1])
    return ["+y", "-y", "+x", "-x"] if ey >= ex else ["+x", "-x", "+y", "-y"]


def depth_along(face: str, lo, hi) -> float:
    """The target's extent along the push of ``face``."""
    return float(hi[0]) - float(lo[0]) if face in ("+x", "-x") else float(hi[1]) - float(lo[1])


def push_direction(side: EdgeFrame, angle_deg: float) -> th.Tensor:
    """Unit push direction toward the partner through ``side`` (its inward normal) turned by ``angle_deg`` about +z."""
    a = math.radians(float(angle_deg))
    return rot2(a) @ side.inward.float()


def goal_root_xy(side: EdgeFrame, lo, hi, yaw: float, gap: float, lateral: float) -> th.Tensor:
    """Root xy of the target at ``yaw`` whose footprint AABB stands ``gap`` outside the partner's (axis-aligned) side and whose AABB centre
    lies ``lateral`` along the side's tangent from its midpoint."""
    n_out = -side.inward.float()
    rel = footprint(lo, hi, (0.0, 0.0), yaw)
    near = float((rel @ n_out).min())
    ctr = 0.5 * (rel.min(dim=0).values + rel.max(dim=0).values)
    plane = float(side.mid.float() @ n_out)
    t = side.tangent.float()
    along_t = float(side.mid.float() @ t) + float(lateral) - float(ctr @ t)
    return n_out * (plane + float(gap) - near) + t * along_t


def sweep_xys(init_xy, goal_xy, step: float) -> List[th.Tensor]:
    """Root xy samples from ``init_xy`` to ``goal_xy`` (both included) at most ``step`` apart."""
    a, b = th.as_tensor(init_xy, dtype=th.float32), th.as_tensor(goal_xy, dtype=th.float32)
    n = max(1, int(math.ceil(float(th.linalg.norm(b - a)) / max(float(step), 1e-3))))
    return [a + (b - a) * (k / n) for k in range(n + 1)]


def sweep_floor_points(lo, hi, init_xy, goal_xy, yaw: float, step: float) -> th.Tensor:
    """``(N, 2)`` world xy where the swept footprint stands: its 4 corners and its root at every :func:`sweep_xys` sample init -> goal."""
    return th.cat([th.cat([footprint(lo, hi, p, yaw), p.reshape(1, 2)]) for p in sweep_xys(init_xy, goal_xy, step)])


def single_floor(names: Sequence[Optional[str]]) -> Optional[str]:
    """Why the floor names under the swept footprint are not one floor object (None when they are; a None name - no floor object
    first under the ray - counts as another floor)."""
    seen = sorted({str(n) for n in names if n is not None})
    if any(n is None for n in names):
        return f"floor_change: no floor object under {sum(n is None for n in names)} of {len(names)} points ({', '.join(seen) or 'none'})"
    if len(seen) > 1:
        return f"floor_change: {' / '.join(seen)}"
    return None


def box_corners(lo2, hi2) -> th.Tensor:
    return th.tensor([[float(lo2[0]), float(lo2[1])], [float(hi2[0]), float(lo2[1])], [float(hi2[0]), float(hi2[1])], [float(lo2[0]), float(hi2[1])]])


def footprint_blocked(corners: th.Tensor, boxes: Sequence[Tuple[str, th.Tensor, th.Tensor]], margin: float = 0.0) -> Optional[str]:
    """The name of the first obstacle box (``(name, lo2, hi2)``) the footprint polygon overlaps with ``margin``, else None."""
    for name, lo2, hi2 in boxes:
        if G.polygons_overlap(corners, box_corners(lo2, hi2), margin):
            return name
    return None


def horizontal_section(pieces: Sequence[th.Tensor], z: float) -> List[th.Tensor]:
    """The cut of convex pieces (points in one frame, z up) by the horizontal plane at ``z``: per piece that reaches it, the CCW corners
    (xy) of the polygon where its convex hull meets the plane."""
    out: List[th.Tensor] = []
    for P in pieces:
        P = th.as_tensor(P, dtype=th.float32).reshape(-1, 3)
        if len(P) == 0:
            continue
        s_ = P[:, 2] - float(z)
        below, above = s_ < -1e-6, s_ > 1e-6
        cand = [P[~below & ~above][:, :2]]
        if bool(below.any()) and bool(above.any()):
            s_lo, s_hi = s_[below][:, None], s_[above][None, :]
            w = (s_lo / (s_lo - s_hi))[..., None]
            cand.append((P[below][:, None, :2] + (P[above][None, :, :2] - P[below][:, None, :2]) * w).reshape(-1, 2))
        pts = th.cat(cand)
        if len(pts) >= 3:
            out.append(G.convex_hull_2d(pts))
    return out


def outermost_on_line(polys: Sequence[th.Tensor], normal, tangent, t0: float) -> Optional[float]:
    """The largest coordinate along ``normal`` of the material the line {p · tangent = t0} crosses in the convex polygons ``polys``
    (CCW corners); None when the line meets none of them."""
    n = th.as_tensor(normal, dtype=th.float32)
    t = th.as_tensor(tangent, dtype=th.float32)
    best: Optional[float] = None
    for poly in polys:
        if len(poly) < 3:
            continue
        a = poly @ t - float(t0)
        b = poly @ n
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            ai, aj = float(a[i]), float(a[j])
            hits = []
            if abs(ai) <= 1e-9:
                hits.append(float(b[i]))
            if (ai < 0.0 < aj) or (aj < 0.0 < ai):
                hits.append(float(b[i]) + (float(b[j]) - float(b[i])) * ai / (ai - aj))
            for v in hits:
                best = v if best is None or v > best else best
    return best


def face_probe(pieces_local: Sequence[th.Tensor], face: str, lo, hi, height: float, probe_lat: float, probe_z: float,
               lateral: float = 0.0) -> Optional[Dict[str, float]]:
    """The push face of ``face`` at ``height`` above the target's lowest point (root-frame pieces, the target upright): the outermost
    material along the face's outward normal on the face's centre line (moved ``lateral`` along the face's tangent: a two-hand spot),
    probed at the lateral offsets 0, ±``probe_lat`` and the heights ``height``, ±``probe_z``. Returns ``{"coord", "spread", "t0", "z"}``
    (``coord`` along the outward normal from the root, ``spread`` = the largest difference between the probes) or None when a probe
    meets no material."""
    nx, ny = FACES[face]
    n = th.tensor([nx, ny])
    tng = th.tensor([-ny, nx])
    ctr = th.tensor([0.5 * (float(lo[0]) + float(hi[0])), 0.5 * (float(lo[1]) + float(hi[1]))])
    t0 = float(ctr @ tng) + float(lateral)
    z0 = float(lo[2]) + float(height)
    vals: List[float] = []
    centre: Optional[float] = None
    for dz in (0.0, probe_z, -probe_z):
        polys = horizontal_section(pieces_local, z0 + dz)
        for dt in ((0.0, probe_lat, -probe_lat) if dz == 0.0 else (0.0,)):
            v = outermost_on_line(polys, n, tng, t0 + dt)
            if v is None:
                return None
            if dz == 0.0 and dt == 0.0:
                centre = v
            vals.append(v)
    return {"coord": float(centre), "spread": float(max(vals) - min(vals)), "t0": t0, "z": z0}


def face_outer(face: str, lo, hi) -> float:
    """The coordinate of ``face``'s AABB plane along its outward normal (root frame)."""
    nx, ny = FACES[face]
    ctr = th.tensor([0.5 * (float(lo[0]) + float(hi[0])), 0.5 * (float(lo[1]) + float(hi[1]))])
    return float(th.tensor([nx, ny]) @ ctr) + 0.5 * depth_along(face, lo, hi)


def face_half_width(face: str, lo, hi) -> float:
    """Half the target's extent along ``face``'s tangent."""
    return 0.5 * (float(hi[1]) - float(lo[1]) if face in ("+x", "-x") else float(hi[0]) - float(lo[0]))


def spot_pairs(pieces_local: Sequence[th.Tensor], face: str, lo, hi, heights: Sequence[float], spacings: Sequence[float], probe_lat: float, probe_z: float,
               flat: float, inset_max: float, depth_tol: float) -> List[Dict[str, Any]]:
    """The two-hand push spots of ``face`` (item C): per height, the first spacing whose :func:`face_probe` at ±spacing / 2 off the centre
    line passes on both sides - flat within ``flat``, at most ``inset_max`` inside the AABB, the two ``coord`` within ``depth_tol`` - with
    both probes on the face's width. ``{"height", "spacing", "spots": [the spot at +spacing / 2 along the tangent, the one at -spacing / 2]}``
    (each spot a probe dict with ``inset`` and ``height``), in the order of ``heights``."""
    outer = face_outer(face, lo, hi)
    half_w = face_half_width(face, lo, hi)
    out: List[Dict[str, Any]] = []
    for h in heights:
        for sp in spacings:
            if 0.5 * float(sp) + float(probe_lat) > half_w:
                continue
            spots = []
            for sgn in (1.0, -1.0):
                f = face_probe(pieces_local, face, lo, hi, float(h), probe_lat, probe_z, lateral=sgn * 0.5 * float(sp))
                if f is None or f["spread"] > flat or outer - f["coord"] > inset_max:
                    break
                spots.append({**f, "inset": outer - f["coord"], "height": float(h)})
            if len(spots) == 2 and abs(spots[0]["coord"] - spots[1]["coord"]) <= depth_tol:
                out.append({"height": float(h), "spacing": float(sp), "spots": spots})
                break
    return out


def contact_local(face: str, coord: float, t0: float, z_local: float) -> th.Tensor:
    """The contact point in the target's root frame: ``coord`` along the face's outward normal, ``t0`` along its tangent."""
    nx, ny = FACES[face]
    return th.tensor([nx * coord - ny * t0, ny * coord + nx * t0, z_local])


# i7-contact 2026-11-07: the base push's hand poses are the shared rule (contact_poses.base_push_approach / base_push_hand_poses)
pitched_approach = CPZ.base_push_approach
push_hand_poses = CPZ.base_push_hand_poses


def push_stance_pose(point_xy, direction_xy, stand_off: float, lateral: float) -> Tuple[float, float, float]:
    """Base pose behind the push face facing along ``direction``: the face ``stand_off`` ahead, the base ``lateral`` to the robot's left of
    the line through ``point``."""
    d = th.as_tensor([float(direction_xy[0]), float(direction_xy[1])], dtype=th.float32)
    d = d / max(float(th.linalg.norm(d)), 1e-9)
    left = th.tensor([-float(d[1]), float(d[0])])
    xy = th.as_tensor([float(point_xy[0]), float(point_xy[1])]) - d * float(stand_off) + left * float(lateral)
    return float(xy[0]), float(xy[1]), math.atan2(float(d[1]), float(d[0]))


def arm_lateral(lateral: float, arm: str) -> float:
    """``push_lateral_offsets_m`` are the right arm's (base left of the line); the left arm's are mirrored."""
    return float(lateral) if arm == "right" else -float(lateral)


def spots_by_arm(points: Sequence[Any], direction_xy) -> Dict[str, int]:
    """The index of the world point each hand takes of two push spots: the right hand the one on the robot's right of ``direction``."""
    left = th.tensor([-float(direction_xy[1]), float(direction_xy[0])])
    a, b = (float(th.as_tensor(p, dtype=th.float32).reshape(-1)[:2] @ left) for p in points)
    return {"right": 0, "left": 1} if a <= b else {"right": 1, "left": 0}


def shifted_pose2d(pose2d: Sequence[float], direction_xy, distance: float) -> Tuple[float, float, float]:
    d = th.as_tensor([float(direction_xy[0]), float(direction_xy[1])], dtype=th.float32)
    d = d / max(float(th.linalg.norm(d)), 1e-9)
    return float(pose2d[0]) + float(d[0]) * float(distance), float(pose2d[1]) + float(d[1]) * float(distance), float(pose2d[2])


def translated_pose(pose: G.Pose, delta) -> G.Pose:
    return th.as_tensor(pose[0], dtype=th.float32) + th.as_tensor(delta, dtype=th.float32), th.as_tensor(pose[1], dtype=th.float32)


# ----------------------------------------------------------------------------------------------------------------
# template (called from IORSLarge._prepare_bindings)
# ----------------------------------------------------------------------------------------------------------------


#: oracle-fix 2026-11-03 (owner decision 3a): per-model push-spot annotations, the B6 contacts slot of the icl
#: (``runlogs/contacts_<date>/<category>__<model>.json``, ``scripts/icl/build_icl.py build_contacts`` -> ``require/contacts.csv``), like the
#: articulation handles of ``require/handles.csv``. A contact of role ``push`` names its ``face`` (+x / -x / +y / -y of the root frame),
#: ``position`` (the one-hand spot on the face, root frame, m), ``height_m`` (above the target's lowest point) and, for two hands,
#: ``spacing_m`` and ``spots`` (the two spots, root frame). A model with an annotation pushes ONLY at its annotated spots; without one the
#: automatic face probe runs (the serving cart).
#: the B6 rows a base push reads: ``base_push`` (a floor-standing target's push spots; distinct from the small-object ``push``
#: rows whose presence makes the grounder enforce ``(contacts target)``) and ``push``
BASE_PUSH_ROLES = ("base_push", "push")
CONTACTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "runlogs")


def push_spot_annotation(s) -> Optional[Dict[str, Any]]:
    """The newest ``runlogs/contacts_*/<category>__<model>.json`` of the target model with at least one ``push`` contact (cached on
    ``s.place``), else None."""
    if "push_annotation" in s.place:
        return s.place["push_annotation"]
    out = None
    try:
        import glob

        mc = s.choice.models["target"]
        key = f"{mc.category}__{mc.model}"
        for d in sorted(glob.glob(os.path.join(CONTACTS_ROOT, "contacts_*")), reverse=True):
            path = os.path.join(d, key + ".json")
            if os.path.exists(path):
                with open(path) as f:
                    ann = json.load(f)
                if any(c.get("role") in BASE_PUSH_ROLES and c.get("verdict", "ok") == "ok" for c in ann.get("contacts", [])):
                    out = {**ann, "path": path}
                break
    except Exception as e:  # noqa: BLE001 - no annotation: the automatic probe
        log.warning("%s: push-spot annotation lookup failed (%r)", s.gp.variant, e)
    s.place["push_annotation"] = out
    return out


def annotated_spot(face: str, lo, hi, position, height: Optional[float] = None) -> Dict[str, float]:
    """An annotated spot (root-frame ``position`` on ``face``) as a face-probe record: ``coord`` along the outward normal, ``t0`` along
    the tangent, ``z``, ``inset`` from the AABB plane, ``spread`` 0 (reviewed flat), ``height`` above the lowest point."""
    nx, ny = FACES[face]
    p = [float(v) for v in position]
    coord = p[0] * nx + p[1] * ny
    t0 = -p[0] * ny + p[1] * nx
    h = float(height) if height is not None else p[2] - float(lo[2])
    return {"coord": coord, "spread": 0.0, "t0": t0, "z": p[2], "height": h, "inset": face_outer(face, lo, hi) - coord, "relaxed": False, "annotated": True}


def annotated_faces(s, face: str) -> Optional[Tuple[List[Dict[str, float]], List[Dict[str, Any]]]]:
    """``(one-hand candidates, two-hand pairs)`` of ``face`` from the model's push-spot annotation, or None without an annotation (the
    automatic probe then runs). A face the annotation does not name gives ``([], [])``."""
    ann = push_spot_annotation(s)
    if ann is None:
        return None
    lo, hi = s.place["local_lo"], s.place["local_hi"]
    ones, pairs = [], []
    for c in ann.get("contacts", []):
        if c.get("role") not in BASE_PUSH_ROLES or c.get("face") != face or c.get("verdict", "ok") != "ok":
            continue
        h = c.get("height_m")
        if c.get("position") is not None:
            ones.append({**annotated_spot(face, lo, hi, c["position"], h), "name": c.get("name")})
        spots = c.get("spots") or []
        if len(spots) == 2 and c.get("spacing_m"):
            sp = [{**annotated_spot(face, lo, hi, q, h), "name": c.get("name")} for q in spots]
            sp.sort(key=lambda r: -r["t0"])  # the spot at +spacing / 2 along the tangent first (spot_pairs' order)
            pairs.append({"height": sp[0]["height"], "spacing": float(c["spacing_m"]), "spots": sp, "relaxed": False, "annotated": True})
    return ones, pairs


def face_spots_cap(s, spots: Sequence[Any]) -> List[Any]:
    """The spots / pairs of one face the stance search tries: all of an annotated model's, else the first ``push_face_spots_max``."""
    if push_spot_annotation(s) is not None:
        return list(spots)
    n = knob(s, "push_face_spots_max")
    return list(spots) if not n else list(spots)[: int(n)]


def note_reach_margin(s, stand_off: float, translate: float, tilt: float, margin: float, height: float) -> None:
    """oracle-fix 2026-11-03 (owner decision 3b, tuning ``push_stand_offs_m`` / ``push_trunks``): the shoulder reach margin of every
    one-hand push stance tried, min / max / count per (stand-off, trunk), in ``template_facts["nextto_reach_margins"]`` (logged with the
    draw's ``no_push``)."""
    facts = getattr(s, "template_facts", None)
    if facts is None:  # a bare sampler (unit tests)
        return
    rec = facts.setdefault("nextto_reach_margins", {})
    key = f"so{stand_off:.2f} t({translate:.2f},{tilt:+.2f})"
    r = rec.setdefault(key, {"min": margin, "max": margin, "n": 0, "heights": [height, height]})
    r["min"], r["max"], r["n"] = round(min(r["min"], margin), 3), round(max(r["max"], margin), 3), r["n"] + 1
    r["heights"] = [round(min(r["heights"][0], height), 3), round(max(r["heights"][1], height), 3)]


def dump_push_geometry(s) -> None:
    """``ISPATIALGYM_PUSH_FACE_DUMP=<dir>`` (oracle-fix 2026-11-03, for the push-spot annotations): the target's collision pieces (root
    frame) and its local box to ``<dir>/<category>__<model>.npz`` (``pieces`` concatenated, ``offsets``, ``lo``, ``hi``)."""
    out_dir = os.environ.get("ISPATIALGYM_PUSH_FACE_DUMP")
    if not out_dir:
        return
    try:
        mc = s.choice.models["target"]
        os.makedirs(out_dir, exist_ok=True)
        pieces = [th.as_tensor(p).float().reshape(-1, 3).numpy() for p in s.place["pieces"]]
        offsets = np.cumsum([0] + [len(p) for p in pieces])
        np.savez(os.path.join(out_dir, f"{mc.category}__{mc.model}.npz"), pieces=np.concatenate(pieces) if pieces else np.zeros((0, 3)), offsets=offsets,
                 lo=s.place["local_lo"].numpy(), hi=s.place["local_hi"].numpy())
        log.info("%s: push geometry of %s/%s dumped to %s (%d pieces)", s.gp.variant, mc.category, mc.model, out_dir, len(pieces))
    except Exception as e:  # noqa: BLE001
        log.warning("%s: push geometry dump failed (%r)", s.gp.variant, e)


def prepare(s) -> None:
    """The target's geometry (root-frame hull, local box, collision pieces), the partner's sides and the obstacle boxes of the floor
    the target slides on; no grasp library, no init surface: the target stands on the floor (the support role, when the variant has
    one, is the floor binding)."""
    tgt, partner = s.target, s.partner
    s.hull = s.world.local_hull(tgt)
    s.rest = G.IDENTITY_QUAT.clone()
    s.grasps, s.grasp_key = [], None
    s.floor_z = float(s.obj("floor").aabb[1][2])
    if s.hull is None or len(s.hull) == 0:
        raise Reject("spec", f"{s.gp.variant}: the target has no collision points")
    lo, hi = s.hull.min(dim=0).values, s.hull.max(dim=0).values
    height = float(hi[2] - lo[2])
    plo, phi = (v.float() for v in partner.aabb)
    from ispatialgym.sampling.v2.tasks.iors_large import aabb_edges

    sides = aabb_edges(plo, phi)
    s.surface = {}
    s.place = {"height": height, "half": 0.5 * max(float(hi[0] - lo[0]), float(hi[1] - lo[1])), "z": s.floor_z, "cells": th.zeros(0, 2), "edges": sides,
               "local_lo": lo, "local_hi": hi, "pieces": s.world.local_collision_pieces(tgt), "partner_aabb": (plo, phi)}
    s.place["boxes"] = obstacle_boxes(s, height)
    dump_push_geometry(s)
    ann = push_spot_annotation(s)
    if ann is not None:
        s.template_facts.setdefault("push_spot_annotation", {"path": ann.get("path"), "spots": [c.get("name") for c in ann.get("contacts", [])]})
    if "support" in s.gp.scene_roles() and not s.bindings.has("support"):
        s.variant_role("support", s.bindings.one("floor"))
    s.template_facts["nextto"] = {"target_box_local": [lo.tolist(), hi.tolist()], "target_height_m": round(height, 4), "partner": partner.name,
                                  "partner_aabb": [plo.tolist(), phi.tolist()], "sides": [e.to_json() for e in sides],
                                  "nextto_bound_at_yaw0_m": round(nextto_bound(*world_aabb(lo, hi, (0.0, 0.0), 0.0, s.floor_z), plo, phi), 4),
                                  "obstacles": [b[0] for b in s.place["boxes"]], "n_pieces": len(s.place["pieces"])}


def obstacle_boxes(s, height: float) -> List[Tuple[str, th.Tensor, th.Tensor]]:
    """``(name, lo xy, hi xy)`` of the scene objects the sliding target must clear: IORS large's floor-blocking objects (not ground-like,
    not the robot, not spawned, >= 5 cm tall) whose AABB reaches below the target's top + 5 cm, the partner left out (its gap is the
    goal's)."""
    out = []
    for o in s.floor_blocking_objects():
        if o is s.partner:
            continue
        lo, hi = o.aabb
        if float(lo[2]) > s.floor_z + height + 0.05:
            continue
        out.append((o.name, lo[:2].float().clone(), hi[:2].float().clone()))
    return out


# ----------------------------------------------------------------------------------------------------------------
# instance
# ----------------------------------------------------------------------------------------------------------------


def target_pose(s, xy, yaw: float) -> G.Pose:
    """The target upright at root ``xy`` / ``yaw``, its lowest point 3 mm above the floor."""
    quat = G.qmul(G.quat_from_yaw(yaw), s.rest)
    return th.tensor([float(xy[0]), float(xy[1]), s.floor_z + s.world.bottom_offset(s.hull, quat) + 0.003]), quat


def put(s, pose: G.Pose) -> None:
    tgt = s.target
    tgt.set_position_orientation(G.t(pose[0]), G.t(pose[1]))
    tgt.keep_still()


def nextto_now(s) -> Optional[bool]:
    """OmniGibson's ``NextTo(target, partner)`` where the target stands now (the state the goal check's ``(nextto ?target ?partner)``
    reads, through the shared predicate wrapper); None when it cannot be read."""
    try:
        from bddl import predicates as BP

        from ispatialgym.sampling import predicates as PRED

        return bool(PRED.evaluate_predicate(BP.NextTo, s.target, s.partner))
    except Exception as e:  # noqa: BLE001 - no verdict: the goal check decides
        log.debug("%s: NextTo unreadable (%r)", s.gp.variant, e)
        return None


def near_box(p, lo2, hi2, margin: float) -> bool:
    """Whether the xy point ``p`` lies within ``margin`` of the axis-aligned rectangle ``lo2``-``hi2`` (inside counts)."""
    dx = max(float(lo2[0]) - float(p[0]), 0.0, float(p[0]) - float(hi2[0]))
    dy = max(float(lo2[1]) - float(p[1]), 0.0, float(p[1]) - float(hi2[1]))
    return math.hypot(dx, dy) <= margin


def floor_ok(s, corners: th.Tensor, step: float = 0.08, skip_near: Optional[Tuple[Any, Any, float]] = None) -> Optional[str]:
    """Why the footprint polygon is not on free floor (None when it is): every corner, edge point and interior grid point on the
    (un-eroded) floor map and in the task rooms. ``skip_near`` = ``(lo xy, hi xy, margin)``: points within ``margin`` of that box are not
    tested on the map (a touching goal against the partner, :func:`footprint_free`)."""
    a, b, c_ = corners[0], corners[1], corners[3]
    u, v = b - a, c_ - a
    nu = max(1, int(math.ceil(float(th.linalg.norm(u)) / step)))
    nv = max(1, int(math.ceil(float(th.linalg.norm(v)) / step)))
    for i in range(nu + 1):
        for j in range(nv + 1):
            p = a + u * (i / nu) + v * (j / nv)
            if skip_near is not None and near_box(p, *skip_near):
                continue
            if not s.kit.trav_free(p):
                _floor_debug(s, "not_floor", p)
                return "not_floor"
            if not P0.in_task_rooms(s, p):
                _floor_debug(s, "outside_rooms", p)
                return "outside_rooms"
    return None


_FLOOR_DEBUG_N = [0]


def _floor_debug(s, why: str, p) -> None:
    """``ISPATIALGYM_NEXTTO_FLOOR_DEBUG=<n>`` (oracle-fix 2026-11-03, the coffee table's goal_not_floor): log the first n failing floor
    points of the process with the draw's current side / face (``s.place["_floor_ctx"]``)."""
    n = int(os.environ.get("ISPATIALGYM_NEXTTO_FLOOR_DEBUG") or 0)
    if _FLOOR_DEBUG_N[0] >= n:
        return
    _FLOOR_DEBUG_N[0] += 1
    log.info("%s: floor test %s at %s (%s)", s.gp.variant, why, [round(float(v), 3) for v in p[:2].tolist()], s.place.get("_floor_ctx"))


def floor_names_at(s, pts: th.Tensor) -> List[Optional[str]]:
    """The ``floors`` object a downward ray (floor + 0.3 m -> floor - 0.3 m) hits first at each world xy of ``pts`` (the robot and the
    target ignored); None where it hits nothing or another body."""
    from omnigibson.utils.sampling_utils import raytest_batch

    floors = s.place.get("floor_prims")
    if floors is None:
        floors = s.place["floor_prims"] = [(o.name, o.prim_path) for o in s.world.scene.objects if o.category == "floors"]
    z0 = float(s.floor_z)
    ignore = [lk.prim_path for lk in s.world.robot.links.values()] + [lk.prim_path for lk in s.target.links.values()]
    hits = raytest_batch([[float(p[0]), float(p[1]), z0 + 0.3] for p in pts], [[float(p[0]), float(p[1]), z0 - 0.3] for p in pts],
                         only_closest=True, ignore_bodies=ignore)
    out: List[Optional[str]] = []
    for h in hits:
        body = str(h.get("rigidBody") or "") if h.get("hit") else ""
        out.append(next((n for n, pp in floors if body and (body == pp or body.startswith(pp + "/"))), None))
    return out


def footprint_free(s, xy, yaw: float, clearance: float) -> Optional[str]:
    """Why the target's footprint at root ``xy`` / ``yaw`` is not free (None when it is): free floor in the task rooms, ``clearance``
    from every obstacle box, no overlap with the partner's AABB."""
    lo, hi = s.place["local_lo"], s.place["local_hi"]
    corners = footprint(lo, hi, xy, yaw)
    plo, phi = s.place["partner_aabb"]
    skip = (plo[:2], phi[:2], float(knob(s, "touching_floor_skip_m"))) if push_relation(s) == "touching" else None
    why = floor_ok(s, footprint(lo, hi, xy, yaw, grow=clearance), skip_near=skip)
    if why is not None:
        return why
    hit = footprint_blocked(corners, s.place["boxes"], clearance)
    if hit is not None:
        return f"obstacle:{hit}"
    polys = partner_sections(s)
    if polys is not None:  # a touching goal against the partner's material: its sections, not its AABB
        return "partner" if hits_sections(corners, polys) else None
    if G.polygons_overlap(corners, box_corners(plo[:2], phi[:2]), 0.0):
        return "partner"
    return None


def partner_sections(s) -> Optional[List[th.Tensor]]:
    """World-xy convex sections of the partner's collision pieces at ``touching_section_heights`` x the target's height above the floor
    (cached on ``s.place``), or None when ``touching_partner_surface`` is off / not a touching push."""
    if push_relation(s) != "touching" or not knob(s, "touching_partner_surface"):
        return None
    if "partner_sections" not in s.place:
        p = s.partner
        root = p.get_position_orientation()
        world = [G.local_points_world(th.as_tensor(P).float().reshape(-1, 3), (G.t(root[0]), G.t(root[1]))) for P in s.world.local_collision_pieces(p)]
        polys: List[th.Tensor] = []
        for f in knob(s, "touching_section_heights"):
            polys += horizontal_section(world, float(s.floor_z) + float(f) * float(s.place["height"]))
        s.place["partner_sections"] = polys
        s.template_facts.setdefault("touching_partner_sections", {"pieces": len(world), "polygons": len(polys)})
    return s.place["partner_sections"]


def hits_sections(corners: th.Tensor, polys: Sequence[th.Tensor], margin: float = 0.0) -> bool:
    return any(G.polygons_overlap(corners, P, margin) for P in polys)


def advance_to_surface(lo, hi, xy, yaw: float, d2: th.Tensor, polys: Sequence[th.Tensor], gap: float, max_m: float,
                       step: float = 0.01) -> Optional[float]:
    """How far the footprint at root ``xy`` / ``yaw`` moves along ``d2`` before it comes within ``gap`` of the material ``polys`` (bisected
    to 0.5 mm after a ``step`` scan); 0 when it already does; None when nothing is met within ``max_m``."""
    d2 = th.as_tensor(d2).float()

    def hit(t: float) -> bool:
        return hits_sections(footprint(lo, hi, th.as_tensor(xy).float() + d2 * float(t), yaw), polys, gap)

    if hit(0.0):
        return 0.0
    a = 0.0
    t = step
    while t <= max_m + 1e-9:
        if hit(t):
            b = t
            while b - a > 5e-4:
                m = 0.5 * (a + b)
                a, b = (a, m) if hit(m) else (m, b)
            return a
        a, t = t, t + step
    return None


def count(stats: Dict[str, int], key: str) -> None:
    stats[key] = stats.get(key, 0) + 1


def goal_candidates(s, rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Goal draws: (side, push angle, face, lateral) shuffled, the faces across the longer extent first within a draw."""
    lo, hi = s.place["local_lo"], s.place["local_hi"]
    angles = [float(a) for a in knob(s, "push_angles_deg")]
    faces = face_order(lo, hi)
    margin = float(knob(s, "goal_lateral_margin_m"))
    out = []
    for side in s.place["edges"]:
        half = 0.5 * float(side.length) - margin
        if half < 0.0:
            continue
        for ang in angles:
            for _ in range(2):
                lat = float(rng.uniform(-half, half))
                out.append({"side": side, "angle": ang, "lateral": lat, "faces": faces})
    order = rng.permutation(len(out))
    return [out[int(i)] for i in order]


def sample_instance(s, rng: np.random.Generator) -> Draft:
    """One nextto draw (module docstring); the target is put back where it was afterwards (``apply_draft`` places it)."""
    from ispatialgym.sampling.v2.tasks.iors_large import ATTEMPT_BUDGET_S

    s._attempt_t0 = time.time()
    s._deadline = s._attempt_t0 + float(s.num("attempt_budget_s", ATTEMPT_BUDGET_S))
    tgt = s.target
    saved = tuple(x.clone() for x in tgt.get_position_orientation())
    try:
        return _draw(s, rng)
    finally:
        tgt.set_position_orientation(*saved)
        tgt.keep_still()
        s.kit.sync_world()


def _draw(s, rng: np.random.Generator) -> Draft:
    lo, hi = s.place["local_lo"], s.place["local_hi"]
    plo, phi = s.place["partner_aabb"]
    clearance = float(knob(s, "floor_clearance_m"))
    d_lo, d_hi = (float(v) for v in knob(s, "push_distance_m"))
    stats: Dict[str, int] = {}
    errors: List[str] = []
    tries = 0
    for cand in goal_candidates(s, rng):
        if tries >= int(knob(s, "goal_tries")):
            break
        s._check_deadline(f"nextto goal candidate {tries + 1}")
        side, ang = cand["side"], cand["angle"]
        d2 = push_direction(side, ang)
        for face in cand["faces"]:
            if not _face_candidates(s, face) and not (two_hands(s) and _face_pairs(s, face)):
                count(stats, "no_push_face")
                continue
            yaw = face_yaw(face, d2)
            bound = nextto_bound(*world_aabb(lo, hi, (0.0, 0.0), yaw, s.floor_z), plo, phi)
            gap = goal_gap(s, bound)
            goal_xy = goal_root_xy(side, lo, hi, yaw, gap, cand["lateral"])
            polys = partner_sections(s)
            if polys is not None:
                adv = advance_to_surface(lo, hi, goal_xy, yaw, d2, polys, gap, float(knob(s, "touching_advance_max_m")))
                if adv is None:
                    count(stats, "goal_no_partner_surface")
                    continue
                goal_xy = th.as_tensor(goal_xy).float() + th.as_tensor(d2).float() * adv
            s.place["_floor_ctx"] = f"goal side {side.name} face {face} yaw {yaw:.2f} at {[round(float(v), 2) for v in goal_xy[:2].tolist()]}, partner {[round(float(v), 2) for v in (*plo[:2].tolist(), *phi[:2].tolist())]}"
            why = footprint_free(s, goal_xy, yaw, clearance)
            if why is not None:
                count(stats, f"goal_{why.split(':')[0]}")
                continue
            tries += 1
            init_xy, dist = None, None
            for _ in range(int(knob(s, "distance_draws"))):
                L = float(rng.uniform(d_lo, d_hi))
                xy0 = goal_xy - d2 * L
                why = footprint_free(s, xy0, yaw, clearance)
                if why is not None:
                    count(stats, f"init_{why.split(':')[0]}")
                    continue
                if aabb_gap(*world_aabb(lo, hi, xy0, yaw, s.floor_z), plo, phi) <= bound + float(knob(s, "init_gap_extra_m")):
                    count(stats, "init_nextto")
                    continue
                bad = next((p for p in sweep_xys(xy0, goal_xy, float(knob(s, "sweep_step_m")))[1:-1] if footprint_free(s, p, yaw, clearance) is not None), None)
                if bad is not None:
                    count(stats, "sweep_blocked")
                    continue
                if bool(knob(s, "push_single_floor")) and single_floor(floor_names_at(s, sweep_floor_points(lo, hi, xy0, goal_xy, yaw, float(knob(s, "sweep_step_m"))))):
                    count(stats, "floor_change")
                    continue
                init_xy, dist = xy0, L
                break
            if init_xy is None:
                continue
            goal_pose = target_pose(s, goal_xy, yaw)
            nt = _goal_nextto(s, goal_xy, yaw, d2)
            if nt is False:
                count(stats, "goal_not_nextto")
                errors.append(f"side {side.name} face {face}: NextTo false at the goal (gap {gap:.3f} m, bound {bound:.3f} m)")
                continue
            init_pose = target_pose(s, init_xy, yaw)
            put(s, init_pose)
            s.kit.sync_world()
            try:
                d = _push_and_stances(s, rng, side, face, yaw, d2, init_pose, goal_pose, dist, gap, bound, nt, stats)
            except Reject as e:
                if e.reason == "time_budget":
                    raise
                count(stats, f"push_{e.reason}")
                errors.append(f"side {side.name} face {face} L {dist:.2f}: {e.reason}: {str(e)[:300]}")
                s.sidecar.reset_instance()
                continue
            d.summary["nextto"]["draw_stats"] = dict(stats)
            return d
    s.template_facts.setdefault("nextto_draw_stats", []).append(dict(stats))
    if (getattr(s, "template_facts", None) or {}).get("nextto_reach_margins"):
        log.info("%s: push stance reach margins (want %s): %s", s.gp.variant, list(knob(s, "push_reach_margin_m")), s.template_facts["nextto_reach_margins"])
    raise Reject("no_push", f"{tries} goal candidates, none admits the push ({dict(sorted(stats.items(), key=lambda kv: -kv[1]))}): " + "; ".join(errors[-3:]))


def _goal_nextto(s, goal_xy, yaw: float, d2: th.Tensor) -> Optional[bool]:
    """NextTo at the goal and at the lateral / yaw probes: True when the goal and at least 3 of the 4 probes read true, False when one
    of them does not, None when NextTo cannot be read (the goal check decides). Touching (:class:`TouchingBasePushMode`): None - the goal
    stands ``touching_goal_gap_m`` short of the partner and the goal check nudges it on to contact."""
    if push_relation(s) == "touching":
        return None
    lat = float(knob(s, "nextto_probe_lateral_m"))
    dyaw = math.radians(float(knob(s, "nextto_probe_yaw_deg")))
    left = th.tensor([-float(d2[1]), float(d2[0])])
    probes = [(goal_xy, yaw), (goal_xy + left * lat, yaw), (goal_xy - left * lat, yaw), (goal_xy, yaw + dyaw), (goal_xy, yaw - dyaw)]
    seen: List[bool] = []
    for xy, y in probes:
        put(s, target_pose(s, xy, y))
        v = nextto_now(s)
        if v is None:
            return None
        seen.append(bool(v))
        if not seen[0] or seen.count(False) > 1:
            return False
    return True


def _face_candidates(s, face: str) -> List[Dict[str, float]]:
    """The flat push faces of ``face`` at ``push_heights_m`` (root frame; once per face and template)."""
    cache = s.place.setdefault("face_cache", {})
    if face not in cache:
        cache[face] = _face_candidates_uncached(s, face)
    return cache[face]


def _face_candidates_uncached(s, face: str) -> List[Dict[str, float]]:
    """At ``face_flat_tol_m`` / ``face_max_inset_m``; when no height passes, once more at twice the flatness tolerance and the inset +
    0.10 m (recorded ``relaxed``: the guarded approach finds the actual contact). A model with a push-spot annotation
    (:func:`push_spot_annotation`) uses its annotated spots of ``face`` instead (oracle-fix 2026-11-03)."""
    ann = annotated_faces(s, face)
    if ann is not None:
        return ann[0]
    lo, hi = s.place["local_lo"], s.place["local_hi"]
    probe_lat, probe_z = (float(v) for v in knob(s, "face_probe_m"))
    half_d = 0.5 * depth_along(face, lo, hi)
    nx, ny = FACES[face]
    outer = float(th.tensor([nx, ny]) @ th.tensor([0.5 * (float(lo[0]) + float(hi[0])), 0.5 * (float(lo[1]) + float(hi[1]))])) + half_d
    probes = [(h, face_probe(s.place["pieces"], face, lo, hi, float(h), probe_lat, probe_z)) for h in knob(s, "push_heights_m")]
    for relaxed, flat, inset_max in ((False, float(knob(s, "face_flat_tol_m")), float(knob(s, "face_max_inset_m"))),
                                     (True, 2.0 * float(knob(s, "face_flat_tol_m")), float(knob(s, "face_max_inset_m")) + 0.10)):
        out = []
        for h, f in probes:
            if f is None or f["spread"] > flat or outer - f["coord"] > inset_max:
                continue
            out.append({**f, "height": float(h), "inset": outer - f["coord"], "relaxed": relaxed})
        if out:
            return out
    return []


def two_hands(s) -> bool:
    """``generation.push_hands`` asks for the two-hand push."""
    return int(knob(s, "push_hands")) >= 2


def _face_pairs(s, face: str) -> List[Dict[str, Any]]:
    """The two-hand push spots of ``face`` (:func:`spot_pairs`; once per face and template): at ``face_flat_tol_m`` / ``face_max_inset_m``,
    else at the one hand's relaxed tolerances (each spot then ``relaxed``)."""
    cache = s.place.setdefault("pair_cache", {})
    ann = annotated_faces(s, face) if face not in cache else None
    if ann is not None:
        cache[face] = ann[1]  # the annotated two-hand spots of the face (oracle-fix 2026-11-03)
    if face not in cache:
        lo, hi = s.place["local_lo"], s.place["local_hi"]
        probe_lat, probe_z = (float(v) for v in knob(s, "face_probe_m"))
        flat, inset = float(knob(s, "face_flat_tol_m")), float(knob(s, "face_max_inset_m"))
        cache[face] = []
        for relaxed, fl, ins in ((False, flat, inset), (True, 2.0 * flat, inset + 0.10)):
            out = spot_pairs(s.place["pieces"], face, lo, hi, [float(h) for h in knob(s, "push_heights_m")], [float(v) for v in knob(s, "push_hand_spacings_m")],
                             probe_lat, probe_z, fl, ins, float(knob(s, "push_hands_depth_tol_m")))
            if out:
                cache[face] = [{**p, "relaxed": relaxed, "spots": [{**sp, "relaxed": relaxed} for sp in p["spots"]]} for p in out]
                break
    return cache[face]


def _push_and_stances(s, rng, side: EdgeFrame, face: str, yaw: float, d2: th.Tensor, init_pose: G.Pose, goal_pose: G.Pose, dist: float, gap: float,
                      bound: float, nt: Optional[bool], stats: Dict[str, int]) -> Draft:
    """The push face, the push stance with its exact IK, the push end, stance A and the A -> B leg for one (goal, init) pair; with
    ``push_hands`` 2 the two-hand push (:func:`_push_two_hands`) when the face has two spots, else the one hand (``two_hand_fallback``)."""
    fallback = None
    if two_hands(s):
        if _face_pairs(s, face):
            return _push_two_hands(s, rng, side, face, yaw, d2, init_pose, goal_pose, dist, gap, bound, nt, stats)
        count(stats, "two_hand_no_spots")
        fallback = f"face {face}: no two spots {list(knob(s, 'push_hand_spacings_m'))} m apart at {list(knob(s, 'push_heights_m'))} m"
    faces = _face_candidates(s, face)
    if not faces:
        raise Reject("no_push_face", f"face {face}: no flat material at {list(knob(s, 'push_heights_m'))} m")
    from ispatialgym.sampling.v2 import tabletop as TT

    arms = TT.offered_arms(s)
    rquat = G.t(init_pose[1])
    root = G.t(init_pose[0])
    margin_lo, margin_hi = (float(v) for v in knob(s, "push_reach_margin_m"))
    cands: List[Tuple[Stance, Dict[str, Any]]] = []
    for f in face_spots_cap(s, faces):
        p_loc = contact_local(face, f["coord"], f["t0"], f["z"])
        point = G.local_points_world(p_loc.reshape(1, 3), (root, rquat))[0]
        for arm in arms:
            for so in knob(s, "push_stand_offs_m"):
                for lat in knob(s, "push_lateral_offsets_m"):
                    pose2d = push_stance_pose(point[:2], d2, float(so), arm_lateral(float(lat), arm))
                    if not P0.base_ok(s, th.tensor(pose2d[:2])):
                        count(stats, "stance_floor")
                        continue
                    for t, k in knob(s, "push_trunks"):
                        st = Stance("manipulation", pose2d, float(t), float(k), float(so), None, None, arm_lateral(float(lat), arm), point.clone(), [arm])
                        m = s.reach_margin(st, point, arm)
                        note_reach_margin(s, float(so), float(t), float(k), float(m), float(f["height"]))
                        if not margin_lo <= m <= margin_hi:
                            count(stats, "stance_reach")
                            continue
                        if not P0.in_frame(s, s.camera_pose_at(st), point.reshape(1, 3)):
                            count(stats, "stance_contact_out_of_frame")
                            continue
                        cands.append((st, {"face": f, "point": point, "arm": arm, "reach_margin": m, "two_hand_fallback": fallback}))
    if not cands:
        raise Reject("no_push_stance", f"face {face}: no stance passes floor / reach / camera ({ {k: v for k, v in stats.items() if k.startswith('stance')} })")
    pre_m = float(knob(s, "push_pre_m"))
    return _complete(s, rng, cands, lambda st, info, why: _push_chain(s, st, info["arm"], info["point"], d2, pre_m, why), side, face, yaw, d2, init_pose,
                     goal_pose, dist, gap, bound, nt, stats)


def _push_two_hands(s, rng, side: EdgeFrame, face: str, yaw: float, d2: th.Tensor, init_pose: G.Pose, goal_pose: G.Pose, dist: float, gap: float,
                    bound: float, nt: Optional[bool], stats: Dict[str, int]) -> Draft:
    """The two-hand push of one (goal, init) pair (item C): the spots of :func:`_face_pairs` in the world, each hand on the spot on its
    own side (:func:`spots_by_arm`), the base centred behind their midpoint (``push_stand_offs_m``, no lateral shift), trunk from
    ``push_trunks``; both reach margins within ``push_reach_margin_m`` and both contacts in the head-camera frame; then both chains
    (:func:`_push_chain_pair`) and the rest as the one hand's (:func:`_complete`)."""
    from ispatialgym.sampling.v2 import tabletop as TT

    arm1 = TT.offered_arms(s)[0]
    arm2 = "left" if arm1 == "right" else "right"
    rquat, root = G.t(init_pose[1]), G.t(init_pose[0])
    margin_lo, margin_hi = (float(v) for v in knob(s, "push_reach_margin_m"))
    cands: List[Tuple[Stance, Dict[str, Any]]] = []
    for pr in face_spots_cap(s, _face_pairs(s, face)):
        pts = [G.local_points_world(contact_local(face, sp["coord"], sp["t0"], sp["z"]).reshape(1, 3), (root, rquat))[0] for sp in pr["spots"]]
        side_of = spots_by_arm(pts, d2)
        points = {a: pts[side_of[a]] for a in (arm1, arm2)}
        spots = {a: pr["spots"][side_of[a]] for a in (arm1, arm2)}
        centre = 0.5 * (pts[0] + pts[1])
        for so in knob(s, "push_stand_offs_m"):
            pose2d = push_stance_pose(centre[:2], d2, float(so), 0.0)
            if not P0.base_ok(s, th.tensor(pose2d[:2])):
                count(stats, "stance_floor")
                continue
            for t, k in knob(s, "push_trunks"):
                st = Stance("manipulation", pose2d, float(t), float(k), float(so), None, None, 0.0, centre.clone(), [arm1, arm2])
                m = {a: s.reach_margin(st, points[a], a) for a in (arm1, arm2)}
                if not all(margin_lo <= v <= margin_hi for v in m.values()):
                    count(stats, "stance_reach")
                    continue
                if not P0.in_frame(s, s.camera_pose_at(st), th.stack([points[arm1], points[arm2]])):
                    count(stats, "stance_contact_out_of_frame")
                    continue
                cands.append((st, {"pair": pr, "face": spots[arm1], "point": points[arm1], "arm": arm1, "arms": [arm1, arm2], "points": points, "spots": spots,
                                   "reach_margin": min(m.values()), "reach_margins": m}))
    if not cands:
        raise Reject("no_push_stance", f"face {face}: no two-hand stance passes floor / reach / camera ({ {k: v for k, v in stats.items() if k.startswith('stance')} })")
    pre_m = float(knob(s, "push_pre_m"))
    return _complete(s, rng, cands, lambda st, info, why: _push_chain_pair(s, st, [(a, info["points"][a]) for a in info["arms"]], d2, pre_m, why), side, face,
                     yaw, d2, init_pose, goal_pose, dist, gap, bound, nt, stats)


def _complete(s, rng, cands: List[Tuple[Stance, Dict[str, Any]]], chain_of, side: EdgeFrame, face: str, yaw: float, d2: th.Tensor, init_pose: G.Pose,
              goal_pose: G.Pose, dist: float, gap: float, bound: float, nt: Optional[bool], stats: Dict[str, int]) -> Draft:
    """The push stances that pass floor / reach / camera, whole-body collision-free with the target placed, best reach first; per stance
    (up to ``push_ik_max``) the chain (``chain_of(stance, info, why)``), the sweep along the line with the pushing arms at contact, the
    push end, stance A and the A -> B leg; the record of the first that passes them all."""
    free = s.ikc.stances_collision_free([c[0] for c in cands])
    cands = [c for c, ok in zip(cands, free.tolist()) if ok]
    if not cands:
        count(stats, "stance_collision")
        raise Reject("no_push_stance", f"face {face}: every push stance collides (target placed)")
    cands.sort(key=lambda c: -min(round(c[1]["reach_margin"] / 0.05) * 0.05, 0.25))
    why: List[str] = []
    for st, info in cands[: int(knob(s, "push_ik_max"))]:
        s._check_deadline("nextto push stance IK")
        chain = chain_of(st, info, why)
        if chain is None:
            count(stats, "push_ik")
            continue
        end2d = shifted_pose2d(st.base_pose2d, d2, dist)
        if not _sweep_ok(s, st, end2d, info.get("arms") or [info["arm"]], chain.get("q_both", chain["q"]), why):
            count(stats, "push_sweep")
            continue
        end = st.with_trunk(st.translate, st.tilt, END_STANCE)
        end.base_pose2d = end2d
        end.focus = None
        end.extras = {"push_from": "manipulation", "push_distance_m": round(dist, 4)}
        # oracle-fix 2026-11-04 round 3 (chair 155 in the lobby: 52 x "push end collides with the target at init" - the base ends where the
        # chair stood): opt-in generation.push_end_ignore_target checks the end stance with the target out of the world, as the sweep
        with (s.ikc.ignoring([s.target]) if s.gp.gen("push_end_ignore_target") else _NoCtx()):
            end_free = bool(s.ikc.stances_collision_free([end])[0])
        if not end_free:
            why.append(f"push end {tuple(round(v, 2) for v in end2d)} collides with the target at init")
            count(stats, "push_end_collision")
            continue
        try:
            init = _stance_a(s, rng, init_pose, yaw, goal_pose, d2)
            s.sidecar.add_stance(init)
            s.sidecar.add_stance(st)
            leg = s.record_leg_a_b(init, st)
        except Reject as e:
            s.sidecar.reset_instance()
            why.append(f"stance A / leg: {e.reason}: {str(e)[:200]}")
            count(stats, f"leg_{e.reason}")
            continue
        s.sidecar.add_stance(end)
        return _record(s, init, st, end, info, chain, side, face, yaw, d2, init_pose, goal_pose, dist, gap, bound, nt, leg)
    raise Reject("push_unreachable", f"face {face}: {len(cands)} collision-free push stances, none passes the IK / sweep / leg: " + "; ".join(why[-3:]))


def _push_chain(s, st: Stance, arm: str, point: th.Tensor, d2: th.Tensor, pre_m: float, why: List[str]) -> Optional[Dict[str, Any]]:
    """The closed hand's pre-contact (target present) and approach to the contact (target ignored at the contact only) by the exact IK
    from the push stance, as the program check runs them: the pre-contact's witness first, the chain tracked from it; the fingertips of
    the pre-contact and the approach in the stance camera's frame (``program_check.fingertips_in_frame``). The first of
    :func:`_hand_chains`."""
    return next(_hand_chains(s, st, arm, point, d2, pre_m, why), None)


def _push_chain_pair(s, st: Stance, hands: Sequence[Tuple[str, th.Tensor]], d2: th.Tensor, pre_m: float, why: List[str]) -> Optional[Dict[str, Any]]:
    """Both hands' chains from one stance (item C): each chain of the first arm (:func:`_hand_chains`, the second arm at the stance
    posture), then the second arm's at the same pitch, either wrist turn, its fingertips in the camera frame, solved by
    ``program_check.second_hand_chain`` (its pre-contact with the first hand at its pre-contact, its approach with the first hand locked
    at its contact, both hands at contact whole-body free). The first arm's chain with ``second`` (the other hand's poses and witnesses)
    and ``q_both`` (both hands at contact), or None."""
    (arm1, p1), (arm2, p2) = hands
    check_frame = os.environ.get("ISG_FINGERTIP_FRAME_CHECK", "on").lower() not in ("off", "0", "false")
    for c1 in _hand_chains(s, st, arm1, p1, d2, pre_m, why):
        for wrist in (c1["wrist"], -c1["wrist"]):
            pre, contact = push_hand_poses(p2, d2, c1["pitch_deg"], wrist, pre_m)
            approach = list(C.line_waypoints(pre, contact, 3))
            tag = (f"{arm2} hand pitch {c1['pitch_deg']:.0f}{'' if wrist > 0 else '-'} with the {arm1} at {c1['pitch_deg']:.0f}{'' if c1['wrist'] > 0 else '-'} "
                   f"(d {st.stand_off:.2f} trunk ({st.translate}, {st.tilt}))")
            out = PC.fingertips_in_frame(s, st, "base_push", [("second.pre", [pre]), ("second.approach", approach)]) if check_frame else None
            if out is not None:
                why.append(f"{tag}: {out['key']} fingertip out of frame")
                continue
            r = PC.second_hand_chain(s, st, arm1, c1["q_pre"], c1["q"], arm2, pre, approach, s.target)
            if not r["ok"]:
                why.append(f"{tag}: {r['why']}")
                if r["why"].startswith("pre-contact"):
                    q2 = s.ikc.q_locked(st, fingers=0.0).clone()
                    i1 = s.kit.robot.arm_control_idx[arm1]
                    q2[i1] = c1["q_pre"][i1]
                    blame_push_pre(s, st, arm2, pre, tag, r["why"], q2, check_frame)
                continue
            second = {"arm": arm2, "pre": pre, "contact": contact, "q_pre": r["q_pre"], "q": r["q"], "pitch_deg": c1["pitch_deg"], "wrist": wrist}
            return {**c1, "q_both": r["q"].clone(), "second": second}
    return None


def _hand_chains(s, st: Stance, arm: str, point: th.Tensor, d2: th.Tensor, pre_m: float, why: List[str]) -> Iterator[Dict[str, Any]]:
    """Each passing chain of :func:`_push_chain` in turn (pitch, then wrist turn): ``{"pre", "contact", "q", "q_pre", "pitch_deg", "wrist"}``."""
    tgt = s.target
    for pitch in knob(s, "push_pitches_deg"):
        for wrist in (1.0, -1.0):
            pre, contact = push_hand_poses(point, d2, float(pitch), wrist, pre_m)
            approach = list(C.line_waypoints(pre, contact, 3))
            tag = f"pitch {float(pitch):.0f}{'' if wrist > 0 else '-'} (d {st.stand_off:.2f} lat {st.lateral:+.2f} trunk ({st.translate}, {st.tilt}))"
            check_frame = os.environ.get("ISG_FINGERTIP_FRAME_CHECK", "on").lower() not in ("off", "0", "false")
            out = PC.fingertips_in_frame(s, st, "base_push", [("pre", [pre]), ("approach", approach)]) if check_frame else None
            if out is not None:
                why.append(f"{tag}: {out['key']} fingertip out of frame")
                continue
            r_pre = s.ikc.exact([pre], st, arm, cache=False, fingers=C.FINGER_CLOSED)
            if not r_pre.ok or r_pre.q is None:
                why.append(f"{tag}: pre-contact {r_pre.reason}")
                blame_push_pre(s, st, arm, pre, tag, r_pre.reason, s.ikc.q_locked(st, fingers=C.FINGER_CLOSED), check_frame)
                continue
            r = s.ikc.waypoints_ok([pre] + approach, st, arm, ignore_for_last=[tgt], cache=False, fingers=C.FINGER_CLOSED, q_first=r_pre.q)
            if not r.ok or r.q is None:
                why.append(f"{tag}: approach {r.reason}")
                continue
            yield {"pre": pre, "contact": contact, "q": r.q.clone(), "q_pre": r_pre.q.clone(), "pitch_deg": float(pitch), "wrist": wrist}


class _NoCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def blame_push_pre(s, st: Stance, arm: str, pre, tag: str, reason: str, q_start: th.Tensor, frame_checked: bool) -> None:
    """Opt-in (``ISPATIALGYM_STANCE_BLAME=<n>``, oracle-fix 2026-11-04 round 3 item 1): why the closed hand's pre-contact exact IK failed at
    this stance / pitch / wrist turn - reached with collisions off or not (mm short, rad off, the joint nearest its limit), else the robot
    links x bodies in contact at a configuration reaching it (``IORSLarge._place_ik_contacts``), and whether the pre fingertip was
    checked in the head frame before the IK (``frame_checked``: the check runs first, so a blamed pose is in frame). At most 12 x n lines
    per attempt."""
    n = os.environ.get("ISPATIALGYM_STANCE_BLAME")
    if not n:
        return
    key = getattr(s, "_attempt_t0", None)
    seen = getattr(s, "_push_pre_blame", None)
    if seen is None or seen[0] != key:
        seen = s._push_pre_blame = [key, 0]
    try:
        cap = 12 * max(1, int(n))
    except ValueError:
        cap = 12
    if seen[1] >= cap:
        return
    seen[1] += 1
    try:
        why = s._place_ik_contacts(pre, q_start, arm, [])
    except Exception as e:  # noqa: BLE001 - a diagnostic never fails the draw
        why = f"blame failed ({e!r})"
    log.info("%s: STANCE_BLAME push pre-contact %s hand, %s [%s]: frame %s; %s", s.gp.variant, arm, tag, str(reason)[:80],
             "in (checked)" if frame_checked else "not checked", why[:420])


def _sweep_ok(s, st: Stance, end2d: Tuple[float, float, float], arms: Sequence[str], q: th.Tensor, why: List[str]) -> bool:
    """The whole body swept from the push stance to the push end along the line, the pushing arms at their contact configuration (``q``),
    both hands closed, the target out of the world (it moves ahead of the hands)."""
    r_arm, l_arm, _ = s.ikc.arm_vectors(st)
    q_arms = {"right": r_arm, "left": l_arm}
    for arm in arms:
        q_arms[arm] = q[s.kit.robot.arm_control_idx[arm]].clone()
    poses = NAV.stride_poses([tuple(st.base_pose2d), tuple(end2d)], NAV.STRIDE_M)
    if not poses or any(abs(a - b) > 1e-6 for a, b in zip(poses[-1], end2d)):
        poses = list(poses) + [tuple(end2d)]
    with s.ikc.ignoring([s.target]):
        ok, bad = NAV.path_collision_free(s.kit, poses, (st.translate, st.tilt), "untucked", margin=0.01, q_arms=q_arms, fingers=C.FINGER_CLOSED)
    if not ok:
        why.append(f"sweep pose {bad}/{len(poses)} {tuple(round(float(v), 2) for v in poses[bad])} collides")
    return bool(ok)


def _stance_a(s, rng: np.random.Generator, init_pose: G.Pose, yaw: float, goal_pose: G.Pose, d2: th.Tensor) -> Stance:
    """Stance A: free floor ``init_distance_m`` from the target within ``init_bearing_max_deg`` of the line's back direction, facing the
    target, the most upright straight trunk whose camera holds the target (its AABB centre, top and bottom) and the goal (the goal
    footprint's centre on the floor, the partner's AABB centre) in frame, collision-free with the target placed."""
    partner = s.partner
    lo, hi = world_aabb(s.place["local_lo"], s.place["local_hi"], init_pose[0][:2], yaw, s.floor_z)
    ctr = 0.5 * (lo + hi)
    tpts = th.stack([ctr, th.tensor([float(ctr[0]), float(ctr[1]), float(lo[2])]), th.tensor([float(ctr[0]), float(ctr[1]), float(hi[2])])])
    plo, phi = (v.float() for v in partner.aabb)
    gpts = th.stack([th.tensor([float(goal_pose[0][0]), float(goal_pose[0][1]), s.floor_z]), 0.5 * (plo + phi)])
    d_lo, d_hi = (float(v) for v in knob(s, "init_distance_m"))
    bmax = math.radians(float(knob(s, "init_bearing_max_deg")))
    back = math.atan2(-float(d2[1]), -float(d2[0]))
    c_xy = th.tensor([float(ctr[0]), float(ctr[1])])
    for _ in range(30):
        bearing = back + float(rng.uniform(-bmax, bmax))
        dist = float(rng.uniform(d_lo, d_hi))
        xy = c_xy + th.tensor([math.cos(bearing), math.sin(bearing)]) * dist
        if not s.kit.trav_free(xy, eroded=True) or not P0.in_task_rooms(s, xy):
            continue
        yaw = math.atan2(float(c_xy[1] - xy[1]), float(c_xy[0] - xy[0]))
        cands = [Stance("init", (float(xy[0]), float(xy[1]), yaw), float(t), 0.0, dist) for t in TRANSLATES]
        feas = [st for st, cam in zip(cands, P0.camera_poses(s, cands)) if P0.in_frame(s, cam, tpts) and P0.in_frame(s, cam, gpts)]
        if not feas:
            continue
        free = P0.body_free(s, feas)
        feas = [st for st, f in zip(feas, free.tolist()) if f]
        if not feas or not P0.trunk_sweep_free(s, (float(xy[0]), float(xy[1]), yaw)):
            continue
        st = min(feas, key=lambda x: x.translate)
        st.extras.update({"bearing_deg": round(math.degrees(C.wrap_angle(bearing - back)), 2), "distance": round(dist, 4)})
        return st
    raise Reject("no_stance", "nextto: no stance A sees the target and the goal")


def _record(s, init: Stance, st: Stance, end: Stance, info: Dict[str, Any], chain: Dict[str, Any], side: EdgeFrame, face: str, yaw: float, d2: th.Tensor,
            init_pose: G.Pose, goal_pose: G.Pose, dist: float, gap: float, bound: float, nt: Optional[bool], leg: Dict[str, Any]) -> Draft:
    tname, pname = s.tname, s.pname
    arm = info["arm"]
    f = info["face"]
    d3 = th.tensor([float(d2[0]), float(d2[1]), 0.0])
    goal_eef = translated_pose(chain["contact"], d3 * dist)
    p_loc = contact_local(face, f["coord"], f["t0"], f["z"])
    extras = {"kind": BASE_PUSH_KIND, "base_start": [float(v) for v in st.base_pose2d], "base_end": [float(v) for v in end.base_pose2d],
              "push_height_m": round(float(f["height"]), 4), "stand_off_m": float(st.stand_off), "lateral_offset_m": float(st.lateral),
              "contact_point": [round(float(v), 5) for v in info["point"].tolist()], "contact_in_object": [round(float(v), 5) for v in p_loc.tolist()],
              "face": face, "face_inset_m": round(float(f["inset"]), 4), "face_spread_m": round(float(f["spread"]), 4), "face_relaxed": bool(f.get("relaxed")), "pitch_deg": chain["pitch_deg"],
              "wrist_sign": chain["wrist"], "partner": pname, "end_stance": END_STANCE, "goal_gap_m": round(gap, 4), "nextto_bound_m": round(bound, 4),
              "drift_tol_m": float(knob(s, "drift_tol_m")), "yaw_tol_rad": float(knob(s, "yaw_tol_rad")), "nextto_push_on_m": float(knob(s, "nextto_push_on_m")),
              "push_speed_m_s": float(knob(s, "push_speed_m_s")), "relation": push_relation(s),
              "reach_margin_m": round(float(info["reach_margin"]), 4), "ignore_last": [tname], "ignore_every": []}
    second = chain.get("second")
    if second is not None:
        arm2, f2 = second["arm"], info["spots"][second["arm"]]
        p2_loc = contact_local(face, f2["coord"], f2["t0"], f2["z"])
        extras.update({"push_hands": 2, "hand_spacing_m": round(float(info["pair"]["spacing"]), 4),
                       "face_centre_in_object": [round(float(v), 5) for v in (0.5 * (p_loc + p2_loc)).tolist()],
                       "reach_margins_m": {a: round(float(v), 4) for a, v in info["reach_margins"].items()},
                       "second_hand": {"arm": arm2, "pre": pose_to_sidecar(second["pre"]), "init": pose_to_sidecar(second["contact"]),
                                       "goal": pose_to_sidecar(translated_pose(second["contact"], d3 * dist)),
                                       "contact_point": [round(float(v), 5) for v in info["points"][arm2].tolist()],
                                       "contact_in_object": [round(float(v), 5) for v in p2_loc.tolist()], "pitch_deg": second["pitch_deg"], "wrist_sign": second["wrist"],
                                       "face_inset_m": round(float(f2["inset"]), 4), "face_spread_m": round(float(f2["spread"]), 4)}})
    elif two_hands(s):
        extras.update({"push_hands": 1, "two_hand_fallback": info.get("two_hand_fallback")})
    s.sidecar.add_contact(ContactRecord("push", arm, pre=chain["pre"], contact=chain["contact"], mid=None, goal=goal_eef, direction=d3, distance_m=float(dist),
                                        object=tname, stance="manipulation", extras=extras))
    st.contacts = ["push"]
    st.extras.update({"push": {"face": face, "height_m": extras["push_height_m"], "distance_m": round(dist, 4), "pitch_deg": chain["pitch_deg"]}})
    if second is not None:
        st.extras["push"]["hands"] = 2
    path_ab = next((p for p in s.sidecar.paths if p.to == "manipulation"), None)
    d = Draft()
    d.object_poses[tname] = init_pose
    d.robot_q = s.stance_q(init)
    d.base_pose2d = tuple(init.base_pose2d)
    pp, _ = s.partner.get_position_orientation()
    if push_relation(s) == "touching":
        extras["touching_push_on_m"] = float(knob(s, "touching_push_on_m"))
    d.summary.update({"relation": push_relation(s), "init_stance": P0.summarize_stance(init), "manipulation_stance": P0.summarize_stance(st), "legs": {"A_B": leg},
                      "partner_position": [float(v) for v in pp], "arm": {"arm": arm}})
    d.summary["nextto"] = {"side": side.name, "push_angle_deg": round(math.degrees(C.wrap_angle(math.atan2(float(d2[1]), float(d2[0])) - side.yaw)), 2), "face": face,
                           "target_yaw": round(yaw, 4), "push_distance_m": round(dist, 4), "goal_gap_m": round(gap, 4), "nextto_bound_m": round(bound, 4),
                           "nextto_at_goal": nt, "push_height_m": extras["push_height_m"], "stand_off_m": float(st.stand_off), "lateral_offset_m": float(st.lateral),
                           "trunk": [st.translate, st.tilt], "pitch_deg": chain["pitch_deg"], "direction": [float(v) for v in d3],
                           "init_pose": [[float(v) for v in init_pose[0]], [float(v) for v in init_pose[1]]],
                           "goal_pose": [[float(v) for v in goal_pose[0]], [float(v) for v in goal_pose[1]]],
                           "path_length_m": {"A_B": None if path_ab is None else path_ab.length_m}}
    if "push_hands" in extras:
        d.summary["nextto"].update({"push_hands": extras["push_hands"], "hand_spacing_m": extras.get("hand_spacing_m"), "two_hand_fallback": extras.get("two_hand_fallback")})
    if second is not None:
        d.summary["arm"]["arms"] = [arm, second["arm"]]
    d.pose_targets[tname] = full_pose_target(goal_pose[0], goal_pose[1], s.tol("position", 0.15), s.tol("orientation", 0.35))
    s.sidecar.describe(tname, s.choice.models["target"].category.replace("_", " "))
    s.sidecar.describe(pname, s.partner.category.replace("_", " "), push_distance_cm=100.0 * float(dist))
    hands = f"{arm} arm" if second is None else f"{arm} + {second['arm']} arms {extras['hand_spacing_m']:.2f} m apart"
    log.info("%s: nextto push drawn: side %s angle %+.0f deg face %s, %.2f m from the goal (gap %.3f, bound %.3f, NextTo at the goal %s), push at %.2f m from "
             "d %.2f lat %+.2f trunk %s pitch %.0f, %s", s.gp.variant, side.name, d.summary["nextto"]["push_angle_deg"], face, dist, gap, bound, nt, f["height"],
             st.stand_off, st.lateral, (st.translate, st.tilt), chain["pitch_deg"], hands)
    return d


# ----------------------------------------------------------------------------------------------------------------
# goal / settle (IORSLarge hooks)
# ----------------------------------------------------------------------------------------------------------------


def goal_pose_of(draft: Draft, tname: str) -> G.Pose:
    """The goal pose: the (settled) init pose moved ``push_distance_m`` along the push direction, orientation kept."""
    nt = draft.summary["nextto"]
    p0, q0 = draft.object_poses[tname]
    d3 = th.tensor([float(v) for v in nt["direction"]])
    return G.t(p0) + d3 * float(nt["push_distance_m"]), G.t(q0).clone()


def goal(s, draft: Draft) -> None:
    tname = s.tname
    g = goal_pose_of(draft, tname)
    draft.goal_poses[tname] = g
    draft.pose_targets[tname] = full_pose_target(g[0], g[1], s.tol("position", 0.15), s.tol("orientation", 0.35))
    end = s.sidecar.stance(END_STANCE)
    s.sidecar.set_goal(end.base_pose2d, end.translate, end.tilt, arms="same")


def after_settle(s, draft: Draft) -> None:
    """The settled init pose must stay near the drawn one (the push stance and contact were placed against it); the goal follows it."""
    tname = s.tname
    nt = draft.summary["nextto"]
    p0 = th.tensor(nt["init_pose"][0])
    q0 = th.tensor(nt["init_pose"][1])
    p, q = draft.object_poses[tname]
    moved = float(th.linalg.norm(G.t(p)[:2] - p0[:2]))
    turned = G.quat_angle(G.t(q), q0)
    nt["settle_shift_m"], nt["settle_turn_rad"] = round(moved, 4), round(turned, 4)
    if moved > 0.05 or turned > 0.2:
        raise Reject("init_moved", f"the target settled {moved:.3f} m / {turned:.3f} rad off its drawn init pose")
    goal(s, draft)


class NextToMode:
    """The nextto mode of :class:`~ispatialgym.sampling.v2.tasks.iors_large.IORSLarge` (``IORSLarge._sample_nextto`` returns it, ``_adopt_mode``
    puts it in front of the sampler's class): IORSLarge's role binding (partner + floor) and template facts stay; the bindings, the draw,
    the goal, the settle and the goal check are the base push's (module docstring). No grasp: no library grasps, no bands, no init
    surface, no placement constraint records."""

    def prepare_bindings(self):
        try:
            prepare(self)
        finally:  # the facts survive a template rejection (diagnostics in template_meta)
            self.template_meta["template_facts"] = _jsonable(self.template_facts)

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        return sample_instance(self, rng)

    def goal(self, draft: Draft):
        goal(self, draft)

    def after_settle(self, draft: Draft):
        after_settle(self, draft)

    def after_goal_check(self, draft: Draft):
        """Nothing to re-solve at the settled goal: the push is recorded from the init pose; the goal check read NextTo there."""

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        return []


class TouchingBasePushMode(NextToMode):
    """oracle-fix 2026-11-03 (touching_large#01: an ottoman pushed against a sofa; ``generation.touching_push: "base"`` on a touching_large
    alias, ``IORSLarge._sample_touching``): :class:`NextToMode`'s base push with the goal ``touching_goal_gap_m`` short of the partner's face
    (no NextTo probes, :func:`goal_gap`, :func:`_goal_nextto`), the record's ``relation: touching`` (the oracle stops the drive once
    Touching holds and the object went ``touching_push_on_m`` further) and, in the goal check, the settled goal nudged ``nudge_m`` at a time
    along the push until ``touching(target, partner)`` holds (the storage box's rule, ``iors_large_touching.nudge_until_touching``)."""

    def refine_targets_from_goal(self, draft: Draft):
        nudge_until_touching(self, draft)
        super().refine_targets_from_goal(draft)


def nudge_until_touching(s, draft: Draft) -> None:
    """The settled goal nudged ``nudge_m`` (1 mm) at a time along the base push's direction until ``touching(target, partner)`` holds, at
    most ``max_nudges`` times (the task's predicate through ``sampling.predicates.evaluate_predicate``); ``summary.touching_nudge``;
    ``Reject("goal_infeasible")`` without contact."""
    from bddl import predicates as BP

    from ispatialgym.sampling import predicates as PRED

    t, p = s.target, s.partner
    push = s.sidecar.contacts.get("push")
    if push is None:
        raise Reject("goal_infeasible", "touching base push: no push record to nudge along")
    d = th.as_tensor(push.direction).reshape(-1)[:2].float()
    d = d / max(float(th.linalg.norm(d)), 1e-9)
    step, n = float(knob(s, "nudge_m")), int(knob(s, "max_nudges"))

    def gap_now() -> str:  # oracle-fix 2026-11-03: the AABB gap / overlap per axis and the centre offset (the reject's diagnostic)
        try:
            tlo, thi = (th.as_tensor(v).float() for v in t.aabb)
            plo, phi = (th.as_tensor(v).float() for v in p.aabb)
            gx = max(float(plo[0] - thi[0]), float(tlo[0] - phi[0]))
            gy = max(float(plo[1] - thi[1]), float(tlo[1] - phi[1]))
            c = 0.5 * (tlo + thi) - 0.5 * (plo + phi)
            return f"AABB gap x {gx:+.3f} y {gy:+.3f} m, centre offset {[round(float(v), 3) for v in c[:2].tolist()]}"
        except Exception as e:  # noqa: BLE001
            return f"gap unreadable ({e!r})"

    start = gap_now()
    for k in range(n + 1):
        s.world.wake([t, p])
        # the task's (touching ?t ?p) through the shared wrapper (OmniGibson's contact report, then the 3 mm gap to the partner's merged
        # convex hull): a kinematic-only partner (fixed base, no joints: sofa_hiphpn_0) is in no PhysX contact report, so the raw state
        # never reads True. The hull is lenient for a sectional (its chaise pocket is inside it) - touching_partner_surface puts the goal
        # at the material first
        if PRED.evaluate_predicate(BP.Touching, t, p):
            draft.summary["touching_nudge"] = {"steps": k, "m": round(k * step, 4)}
            return
        if k == n:
            break
        pos, quat = t.get_position_orientation()
        t.set_position_orientation(pos + th.tensor([float(d[0]) * step, float(d[1]) * step, 0.0]), quat)
        t.keep_still()
        s.world.step_physics(3, still=[s.robot, p])
    raise Reject("goal_infeasible", f"touching base push: no contact with {p.name} after {n} nudges of {1000 * step:.0f} mm along the push "
                                    f"(direction {[round(float(v), 3) for v in d.tolist()]}; before: {start}; after: {gap_now()})")

