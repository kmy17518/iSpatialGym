"""Alignment — world vertical / world horizontal (spec §5.5 / §6.9; oracle doc §6.2 row "alignment — world vertical /
horizontal"): the target hangs on a wall through a COM hinge (TRO ``constraints``), starts tilted by δ in the wall
plane and is pushed level / upright with a tangential push at its low end while the other hand braces.

Roles: ``target`` (spawned at ``roles.target.scale``), the wall (scene role ``support``; ``bindings.support`` = allowed
wall names, ``bindings.support_regions`` = surveyed wall segments; the v1 facade calls them ``wall`` / ``wall_regions``),
``floor``. Prompt 0 with the wall as the face (n = wall normal, P = the hinge point on the wall at hang height).

Template (:meth:`AlignmentWall.prepare_bindings` / :meth:`sample_template`): the target's §5.5 frame from the alias'
per-model ``generation.front_axis`` / ``up_axis`` (/ ``level_axis``) record — the reviewed icl frame ``ground_isg``
writes; an alias without it is ``Reject("missing_frame")``, never a default (owner 2026‑09‑16 22:55) —, wall spots from
the regions (0.25 m seeding) or the 24-ray search, flat 5 × 5 patch within 1 cm over (w + 0.10) × (h + 0.10) (horizontal: swept extents),
front clearance 0.35 m, in-room test 0.8 m in front, one wall per template (most spots); per spot the Phase-2 step-1
survivors (arms excluded: δ-, target- and scale-independent, cached per (scene, spot)); the template filter keeps a
spot when steps 1–3 pass for δ = +30° and δ = −30° at the configured scale; none → ``Reject("no_stance")`` with the
``scale_hint`` diagnostic (uniform s ∈ {0.9, …, 0.5} on the same spots, never applied).

Instance (:meth:`sample_instance`): spot drawn weighted by its survivors, δ = sign · U(15°, 45°), Phase 1 (β_max 35°,
d 0.5–1.5 m, the whole tilted target in frame), Phase 2 steps 2–4 (contacts of §5.5: push arm at E_low with the arc
waypoints δ/2 and goal, brace kinds ``below_high`` → ``above_low`` → ``side_high``; focus C; C, E_low and B in frame),
step 5 (brace arm first with the target not ignored, then the push arm with the brace locked and the target ignored
for the contact poses, combined whole-body check), step 6 path + transition; the hinge at the COM after apply; hinge
drift ≤ 0.03 rad and ≥ 5 visible objects after settling; goal = the level pose, tolerance 0.08 rad, equivalences from
``symmetric_180`` (half turn about the front axis) and the icl symmetry tags.

Stand-ins (reported): the combined whole-body check places the push arm at its **pre-contact** solution (the contact
pose touches the target by design); the arms-excluded step-1 check as in :mod:`prompt0`.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.constraints import Hinge, create_hinge, remove_hinge
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft, _jsonable
from ispatialgym.sampling.tro_writer import pose_target
from ispatialgym.sampling.v2 import prompt0 as P0
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.sampling.v2.stances import ContactRecord, FaceFrame, Stance

log = logging.getLogger("ispatialgym.sampling.v2")

SUBTASKS = ("alignment-world_vertical", "alignment-world_horizontal")
#: v1 ``bddl_writer`` definition ids (facade)
V1_DEFINITION_ID = {"alignment-world_vertical": 1, "alignment-world_horizontal": 2}
#: §5.5 brace candidates in order (first reachable wins)
BRACE_KINDS = ("below_high", "above_low", "side_high")
#: A8 (owner decision 2026‑09‑16): |δ| ≤ 25° until the two-stroke push for larger tilts exists (TODO: two strokes with a
#: re-grasp of the push point between them, each within the arm's arc reach; the ±30° template filter then returns)
MAX_DEVIATION_DEG = 25.0
PUSH_CLASSES = ("push_up", "push_down", "push_left", "push_right")
AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
STRUCTURE = ("walls", "floors", "ceilings")


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (tested without a simulator)
# ----------------------------------------------------------------------------------------------------------------


def axis_vec(name: str) -> th.Tensor:
    name = str(name).strip()
    name = name if name[0] in "+-" else "+" + name
    return th.tensor(AXES[name], dtype=th.float32)


def frame_axes(front: str, up: str, level: Optional[str] = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Object-frame (front, side, up) unit vectors from the reviewed axis names (``front_axis`` / ``up_axis``; the
    optional ``level_axis`` must be the axis along the wall implied by the two). No defaults: the sampler passes the
    alias' per-model record (:meth:`AlignmentWall.reviewed_frame`)."""
    f, u = axis_vec(front), axis_vec(up)
    if abs(float(th.dot(f, u))) > 1e-6:
        raise ValueError(f"front_axis {front} and up_axis {up} are not orthogonal")
    side = th.cross(u, f, dim=0)
    if level is not None:
        want = str(level).strip().lstrip("+-").lower()
        got = "xyz"[int(th.argmax(side.abs()))]
        if want != got:
            raise ValueError(f"level_axis {level} is not the axis along the wall implied by front {front} / up {up} ({got})")
    return f, side, u


def hang_quat(normal, front, side, up) -> th.Tensor:
    """Orientation mapping the object's front axis onto the (horizontal) wall normal and its up axis onto world +z."""
    n = th.as_tensor(normal, dtype=th.float32).clone()
    n[2] = 0.0
    n = n / th.linalg.norm(n)
    z = th.tensor([0.0, 0.0, 1.0])
    t = th.cross(z, n, dim=0)
    W = th.stack([n, t, z], dim=1)
    O = th.stack([front, side, up], dim=1)
    return C.mat_to_quat(W @ O.T)


def swept_extent(p: float, q: float, delta: float) -> float:
    """max over a ∈ [0, δ] of p·cos a + q·sin a: the extent along one wall axis of a p × q rectangle swept between the
    aligned pose and a tilt of δ."""
    if delta <= 0:
        return p
    if math.atan2(q, p) <= delta:
        return math.hypot(p, q)
    return p * math.cos(delta) + q * math.sin(delta)


def brace_candidate_order(height: float, width: float, kinds: Sequence[str] = BRACE_KINDS) -> List[str]:
    """§5.5 brace kinds in order; ``side_high`` is admitted only when the target's height ≥ half its width (frames)."""
    out = []
    for k in kinds:
        if k == "side_high" and not (height >= 0.5 * width):
            continue
        out.append(k)
    return out


def leveling_tangent(E_low, C_, normal, delta: float) -> th.Tensor:
    """Unit push direction at E_low: the tangent of the circle about C through E_low carried by the rotation that
    brings the object from its tilt δ (about ``normal``) back to level (−δ), signed so the low end rises (t·ẑ > 0)."""
    n = th.as_tensor(normal, dtype=th.float32)
    r = th.as_tensor(E_low, dtype=th.float32) - th.as_tensor(C_, dtype=th.float32)
    t = th.cross(n, r, dim=0) * (-1.0 if delta > 0 else 1.0)
    if float(th.linalg.norm(t)) < 1e-9:
        raise ValueError("E_low coincides with the hinge")
    t = t / th.linalg.norm(t)
    if float(t[2]) < 0:
        t = -t  # the low end rises
    return t


def extreme_points(pts: th.Tensor, axis, band: float = 0.03) -> Tuple[th.Tensor, th.Tensor]:
    """Means of the hull points within ``band`` of the two extremes along ``axis`` (low first = smaller z)."""
    a = th.as_tensor(axis, dtype=th.float32)
    s = pts @ a
    lo = pts[s <= float(s.min()) + band].mean(dim=0)
    hi = pts[s >= float(s.max()) - band].mean(dim=0)
    return (lo, hi) if float(lo[2]) <= float(hi[2]) else (hi, lo)


def on_plane(p: th.Tensor, plane_point, normal) -> th.Tensor:
    """``p`` moved along ``normal`` onto the plane through ``plane_point``."""
    n = th.as_tensor(normal, dtype=th.float32)
    d = float(th.dot(th.as_tensor(plane_point, dtype=th.float32) - p, n))
    return p + n * d


def brace_point(level_pts: th.Tensor, kind: str, side_axis, high_sign: float, C_, band: float = 0.03) -> Tuple[th.Tensor, th.Tensor]:
    """(B, approach a) of a §5.5 brace candidate on the **level-pose** hull: ``below_high`` = the lowest points on
    E_high's side, a = +ẑ; ``above_low`` = the highest points on E_low's side, a = −ẑ; ``side_high`` = the outermost
    lateral points on E_high's side in the upper half, a = horizontal from B toward C."""
    s = th.as_tensor(side_axis, dtype=th.float32)
    c = th.as_tensor(C_, dtype=th.float32)
    lat = (level_pts - c) @ s
    high_side = level_pts[lat * high_sign > 0]
    low_side = level_pts[lat * high_sign < 0]
    if kind == "below_high":
        pts = high_side if len(high_side) else level_pts
        z = pts[:, 2]
        B = pts[z <= float(z.min()) + band].mean(dim=0)
        return B, th.tensor([0.0, 0.0, 1.0])
    if kind == "above_low":
        pts = low_side if len(low_side) else level_pts
        z = pts[:, 2]
        B = pts[z >= float(z.max()) - band].mean(dim=0)
        return B, th.tensor([0.0, 0.0, -1.0])
    if kind == "side_high":
        pts = high_side if len(high_side) else level_pts
        upper = pts[pts[:, 2] >= float(c[2])]
        pts = upper if len(upper) else pts
        l2 = (pts - c) @ s * high_sign
        B = pts[l2 >= float(l2.max()) - band].mean(dim=0)
        a = c - B
        a[2] = 0.0
        if float(th.linalg.norm(a)) < 1e-9:
            a = -s * high_sign
        return B, a / th.linalg.norm(a)
    raise ValueError(kind)


def push_arm_of(E_low, C_, tangent_left) -> str:
    """§5.5 push arm = the arm on E_low's side of the hinge (robot facing the wall)."""
    off = float(th.dot((th.as_tensor(E_low, dtype=th.float32) - th.as_tensor(C_, dtype=th.float32))[:2], th.as_tensor(tangent_left, dtype=th.float32)[:2]))
    return "left" if off > 0 else "right"


def other_arm(arm: str) -> str:
    return "left" if arm == "right" else "right"


def half_turn_equivalence(front) -> List[float]:
    """The ``symmetric_180`` orientation equivalence: a half turn about the object's front axis (xyzw)."""
    q = C.quat_from_axis_angle(front, math.pi)
    return [float(round(float(v), 6)) or 0.0 for v in q]


# ----------------------------------------------------------------------------------------------------------------
# the task module
# ----------------------------------------------------------------------------------------------------------------


@register(*SUBTASKS)
class AlignmentWall(SamplerV2):
    """World vertical (upright goal) and world horizontal (level goal) share everything but the patch extents."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.horizontal = "horizontal" in self.subtask
        if self.gp.bddl is None:
            self.definition_id = V1_DEFINITION_ID[self.subtask]
        self.wall_role = "support" if "support" in self.gp.roles else ("wall" if "wall" in self.gp.roles else "support")
        self.hinge: Optional[Hinge] = None
        self.hull: Optional[th.Tensor] = None
        self.com_local = th.zeros(3)
        self.front = self.side = self.up = None
        self.frame: Dict[str, Optional[str]] = {}
        self.spots: List[Dict[str, Any]] = []
        self.spot: Optional[Dict[str, Any]] = None
        self.step1: Dict[int, List[Stance]] = {}

    # ------------------------------------------------------------------ parameters

    def v1_role_aliases(self) -> Dict[str, str]:
        return {"wall": "support"} if self.wall_role == "support" else {}

    def param(self, key: str, default=None, v1_key: Optional[str] = None):
        v = self.gp.gen(key)
        if v is None and v1_key is not None:
            v = self.gp.gen(v1_key)
        return default if v is None else v

    def target_param(self, key: str, default=None):
        """``generation.<key>`` per target: a value, or a dict keyed ``<category>/<model>`` / ``<category>`` / ``default``."""
        v = self.param(key, default)
        if isinstance(v, dict):
            c = self.choice.models["target"]
            v = v.get(f"{c.category}/{c.model}", v.get(c.category, v.get("default", default)))
        return v

    def hang_height(self) -> float:
        return float(self.param("hang_height_m", 1.45, "height"))

    def deviation_range(self) -> Tuple[float, float]:
        """§5.5 δ = sign·U(lo, hi). Owner decision A8 (2026‑09‑16): |δ| ≤ ``max_deviation_deg`` (25°) until the two-stroke
        push for larger tilts exists (TODO, D list) — a wider customs range is capped here and the applied range is
        recorded in ``template_facts.deviation_range_applied_deg``."""
        lo, hi = (float(v) for v in self.param("deviation_range_deg", [15.0, 25.0]))
        cap = float(self.param("max_deviation_deg", MAX_DEVIATION_DEG))
        hi_applied = min(hi, cap)
        lo_applied = min(lo, hi_applied)
        self.template_facts["deviation_range_applied_deg"] = [lo_applied, hi_applied]
        if hi_applied < hi:
            self.template_facts["deviation_range_customs_deg"] = [lo, hi]
        return math.radians(lo_applied), math.radians(hi_applied)

    def stance_posture(self) -> Tuple[str, float]:
        """A6: the arm posture the two-arm manipulation stance records and drives in with (``generation.
        stance_arm_posture``, default ``tucked`` — hanging arms clear a wall 0.5 m ahead where untucked / ready hands do
        not) and its gripper opening (closed pads: the oracle closes the grippers before a push / brace)."""
        return str(self.param("stance_arm_posture", "tucked")), float(C.FINGER_CLOSED)

    def symmetric_180(self) -> bool:
        return bool(self.target_param("symmetric_180", False))

    def reviewed_frame(self) -> Dict[str, Optional[str]]:
        """§5.5 frame of the target from the alias: ``generation.front_axis`` / ``up_axis`` (/ ``level_axis``, horizontal),
        per model (dict form) as ``ground_isg`` writes the reviewed icl ``frame_*`` columns. **Never defaulted** (owner
        2026‑09‑16 22:55: the painting-shaped default +x / +z hung the license plate table-flat, run 0916‑212037) —
        an alias without the keys for the spawned model is ``Reject("missing_frame")`` (re-ground the alias)."""
        c = self.choice.models["target"]
        front, up = self.target_param("front_axis"), self.target_param("up_axis")
        level = self.target_param("level_axis") if self.horizontal else None
        missing = [k for k, v in (("front_axis", front), ("up_axis", up)) if v is None]
        if missing:
            raise Reject("missing_frame", f"{self.gp.alias}: generation.{' / '.join(missing)} carries no entry for {c.category}/{c.model} "
                                          f"(the reviewed icl frame is written per model by ground_isg; the sampler does not default a frame)")
        return {"front_axis": str(front), "up_axis": str(up), "level_axis": None if level is None else str(level)}

    # ------------------------------------------------------------------ roles

    def bind_scene_roles(self):
        self.bind_floor_v2()  # the wall is bound in prepare_bindings (it needs the spawned target's extents)

    @property
    def target(self):
        return self.obj("target")

    @property
    def tname(self) -> str:
        return self.bindings.one("target").name

    @property
    def wall(self):
        return self.obj(self.wall_role)

    def support_objects(self) -> List[Any]:
        return [self.wall] if self.bindings.has(self.wall_role) else []

    # ------------------------------------------------------------------ target geometry

    def com_world(self, obj=None, pose: Optional[G.Pose] = None) -> th.Tensor:
        if pose is None:
            p, q = (obj or self.target).get_position_orientation()
            pose = (p.float(), q.float())
        return pose[0] + G.rotate(pose[1], self.com_local)

    def _com_local(self, obj) -> th.Tensor:
        try:
            total, acc = 0.0, th.zeros(3)
            rp, rq = obj.get_position_orientation()
            for link in obj.links.values():
                m = float(link.mass)
                if m <= 0:
                    continue
                lp, lq = link.get_position_orientation()
                acc += m * (lp.float() + G.rotate(lq.float(), G.t(link.center_of_mass)))
                total += m
            if total > 0:
                return G.rotate(G.qinv(rq.float()), acc / total - rp.float())
        except Exception:  # noqa: BLE001
            pass
        lo, hi = obj.aabb
        rp, rq = obj.get_position_orientation()
        return G.rotate(G.qinv(rq.float()), (lo + hi) / 2 - rp.float())

    def dims(self, hull: Optional[th.Tensor] = None) -> Tuple[float, float, float]:
        """(thickness along front, width along the wall, height along up) of the canonical hull."""
        pts = self.hull if hull is None else hull
        return tuple(float((pts @ a).max() - (pts @ a).min()) for a in (self.front, self.side, self.up))

    def patch_extents(self, hull: Optional[th.Tensor] = None) -> Tuple[float, float]:
        _, w, h = self.dims(hull)
        if not (self.horizontal and bool(self.param("swept_patch", True))):
            return w, h
        d = max(abs(a) for a in self.deviation_range())
        return swept_extent(w, h, d), swept_extent(h, w, d)

    def hang_pose(self, spot: Dict[str, Any], delta: float, hull: Optional[th.Tensor] = None, com: Optional[th.Tensor] = None) -> G.Pose:
        """§5.5 hang pose: up axis along world z, front axis along n, centre at hang height, thickness/2 + 0.01 off the
        wall, rotated in-plane by δ about n through the COM."""
        n = spot["normal"]
        q_up = hang_quat(n, self.front, self.side, self.up)
        q = G.qmul(G.quat_about_axis(n, delta), q_up)
        thickness = self.dims(hull)[0]
        centre = spot["point"] + n * (thickness / 2 + float(self.param("wall_offset_extra_m", 0.01)))
        centre = th.tensor([float(centre[0]), float(centre[1]), spot["hang_z"]])
        return centre - G.rotate(q, self.com_local if com is None else com), q

    def hull_at(self, pose: G.Pose, hull: Optional[th.Tensor] = None) -> th.Tensor:
        return G.local_points_world(self.hull if hull is None else hull, pose)

    # ------------------------------------------------------------------ contacts (§5.5)

    def contacts_for(self, spot: Dict[str, Any], delta: float, hull: Optional[th.Tensor] = None, com: Optional[th.Tensor] = None) -> Dict[str, Any]:
        """Geometry of one (spot, δ): the tilted / level hull, C, E_low / E_high, the push contact poses (pre, init,
        mid, goal, direction) and the brace candidates in order ({kind, B, a, pose, pre})."""
        n = spot["normal"]
        band = float(self.param("contact_extreme_band_m", 0.03))
        pose_t = self.hang_pose(spot, delta, hull, com)
        pose_l = self.hang_pose(spot, 0.0, hull, com)
        pts_t, pts_l = self.hull_at(pose_t, hull), self.hull_at(pose_l, hull)
        C_ = pose_t[0] + G.rotate(pose_t[1], self.com_local if com is None else com)
        thickness = self.dims(hull)[0]
        mid_plane = spot["point"] + n * (thickness / 2 + float(self.param("wall_offset_extra_m", 0.01)))
        level_axis_t = G.rotate(pose_t[1], self.side)  # ℓ_δ = the level axis rotated by δ
        E_low, E_high = extreme_points(pts_t, level_axis_t, band)
        E_low, E_high = on_plane(E_low, mid_plane, n), on_plane(E_high, mid_plane, n)
        # push: tangent about C through E_low, signed so the low end rises; c = n
        a_push = leveling_tangent(E_low, C_, n, delta)
        push_init = C.contact_eef(E_low, a_push, closing=n)
        arc = C.arc_waypoints(push_init, C_, n, -delta, n=2)  # [δ/2, goal] (the material point carried, a re-tangent)
        push_mid, push_goal = arc[0], arc[1]
        push_pre = C.pre_pose(push_init)
        # the other wrist turn (closing axis −n): the same pad contact with the gripper's broad faces swapped — the wrist
        # camera (right_realsense_link) sits on one of them and ends inside the wall for one of the two turns (first demo
        # 2026‑09‑16); step 5 picks the turn whose configuration clears the wall
        # ... and the closing axis IN the wall plane (⟂ a and n): the gripper's thin side then faces the wall — with the
        # closing axis along n the palm (far wider than the fingers along the closing axis) sits centred 2 cm off the
        # wall and penetrates it whichever way the wrist turns (first-demo loop 12:12: both turns "inside the wall")
        # ... and the approach PITCHED off the wall plane toward the room (θ = 30° / 45°: a' = cos θ·a − sin θ·n): with the
        # approach in the wall plane the forearm / wrist lie along the wall 2 cm from it and penetrate it at any wrist
        # turn (first-demo loop 12:36: every closing variant "inside the wall"); the pitched hand comes down onto the
        # contact from the room side and the wrist stands sin θ · (wrist offset) off the wall. The arc waypoints rotate
        # the pitched init pose about C (the pitch is kept), the reach-volume class stays the nearest push class.
        push_variants = []
        for pitch_deg in [float(v) for v in (self.param("contact_pitch_deg") or (30.0, 45.0, 0.0))]:
            th_ = math.radians(pitch_deg)
            a_v = C.unit(a_push * math.cos(th_) - n * math.sin(th_))
            c_in = C.unit(th.cross(a_v, n, dim=0))
            c_out = C.unit(th.cross(c_in, a_v, dim=0))  # the third axis (n-like, ⟂ a')
            for label, closing in (("in+", c_in), ("in-", -c_in), ("n+", c_out), ("n-", -c_out)):
                init_v = C.contact_eef(E_low, a_v, closing=closing)
                arc_v = C.arc_waypoints(init_v, C_, n, -delta, n=2)
                push_variants.append({"wrist": f"p{pitch_deg:.0f}/{label}", "pitch_deg": pitch_deg, "pre": C.pre_pose(init_v), "init": init_v, "mid": arc_v[0], "goal": arc_v[1]})
        # brace candidates on the level pose; E_high's side along the level axis
        side_l = G.rotate(pose_l[1], self.side)
        C_l = pose_l[0] + G.rotate(pose_l[1], self.com_local if com is None else com)
        high_sign = 1.0 if float(th.dot(E_high - C_, level_axis_t)) > 0 else -1.0
        _, w, h = self.dims(hull)
        braces = []
        for kind in brace_candidate_order(h, w, self.param("brace_kinds", BRACE_KINDS)):
            B, a = brace_point(pts_l, kind, side_l, high_sign, C_l, band)
            B = on_plane(B, mid_plane, n)
            # brace pre-contact = eef − 0.10·a (spec §5.5): the approach slides along the wall from outside the hull.
            # ``generation.brace_pre_offset_wall_m`` (default 0) adds an offset off the wall along +n. Smoke findings
            # (2026‑09‑16): the closed gripper's sphere model is flagged at the 1 cm margin at every pose within ≈ 5 cm of
            # the wall, so neither an offset of 8 cm (diagonal approach) nor an approach normal to the wall (which passes
            # through the tilted target's plane region) makes the intermediate poses pass with the wall present — the wall
            # has to be ignored along the whole brace chain (as along the push chain), in the sampler and in the program
            # check alike.
            off_wall = float(self.param("brace_pre_offset_wall_m", 0.0))
            for pitch_deg in [float(v) for v in (self.param("contact_pitch_deg") or (30.0, 45.0, 0.0))]:  # pitched off the wall (see the push variants)
                th_ = math.radians(pitch_deg)
                a_v = C.unit(a * math.cos(th_) - n * math.sin(th_))
                c_in_b = C.unit(th.cross(a_v, n, dim=0))
                c_out_b = C.unit(th.cross(c_in_b, a_v, dim=0))
                for label, closing in (("in+", c_in_b), ("in-", -c_in_b), ("n+", c_out_b), ("n-", -c_out_b)):
                    eef_w = C.fingertip_eef(B, a_v, closing=closing, clearance=float(self.param("brace_offset_m", 0.005)))
                    pre_a = C.pre_pose(eef_w)
                    pre = (pre_a[0] + off_wall * n, pre_a[1]) if off_wall else pre_a
                    braces.append({"kind": kind, "B": B, "a": a_v, "pose": eef_w, "pre": pre, "wrist": f"p{pitch_deg:.0f}/{label}", "pitch_deg": pitch_deg})
        push_arm = push_arm_of(E_low, C_, th.tensor([float(n[1]), -float(n[0]), 0.0]))
        return {"pose_tilted": pose_t, "pose_level": pose_l, "hull_tilted": pts_t, "hull_level": pts_l, "C": C_, "E_low": E_low, "E_high": E_high,
                "push": {"pre": push_pre, "init": push_init, "mid": push_mid, "goal": push_goal, "direction": a_push, "arm": push_arm,
                         "distance": float(th.linalg.norm(push_goal[0] - push_init[0])), "variants": push_variants},
                "braces": braces, "brace_arm": other_arm(push_arm), "delta": delta}

    def brace_clear(self, geo: Dict[str, Any], brace: Dict[str, Any]) -> bool:
        """§5.5: at init the brace hand must be ≥ 2 cm clear of the tilted hull (fingertip and pad-centre points)."""
        clear = float(self.param("brace_clearance_m", 0.02))
        pts = geo["hull_tilted"]
        for p in (C.fingertip(brace["pose"]), C.pad_center(brace["pose"]), brace["pose"][0]):
            if float(th.linalg.norm(pts - p, dim=1).min()) < clear:
                return False
        return True

    def contact_specs(self, geo: Dict[str, Any]) -> Tuple[P0.ContactSpec, List[Tuple[Dict[str, Any], P0.ContactSpec]]]:
        push = geo["push"]
        spec_p = P0.ContactSpec("push", push["arm"], push["init"], push["pre"], waypoints=[push["mid"], push["goal"]], after=True, classes=PUSH_CLASSES,
                                ignore_for_last=[self.target], ignore_waypoints=[self.target])
        braces = [(b, P0.ContactSpec(f"brace_{b['kind']}", geo["brace_arm"], b["pose"], b["pre"], classes=PUSH_CLASSES)) for b in geo["braces"]
                  if self.brace_clear(geo, b)]
        return spec_p, braces

    def screen(self, cands: Sequence[Stance], geo: Dict[str, Any], stats: Optional[Dict[str, Any]] = None) -> List[Stance]:
        """§5.5 Phase 2 steps 2–3 for one (spot, δ) on step-1 survivors: push contact set + ≥ 1 brace candidate in the
        reach volume; C, E_low and the first feasible B in frame (5 %); optical axis within 8° (else 20°) of C; the best
        angle class survives. ``extras["brace_kinds"]`` = the feasible brace kinds of each survivor."""
        rv = P0.reach_volume()
        bz = P0.base_z(self)
        spec_p, braces = self.contact_specs(geo)
        st_ = stats if stats is not None else {}
        st_["brace_kinds_clear"] = [b["kind"] for b, _ in braces]
        st_["brace_kinds_blocked_by_tilted_hull"] = [b["kind"] for b in geo["braces"] if b["kind"] not in st_["brace_kinds_clear"]]
        if not braces:
            return []
        reach: List[Tuple[Stance, List[Dict[str, Any]]]] = []
        n_push = 0
        kinds_ok: Dict[str, int] = {}
        for st in cands:
            if not spec_p.volume_ok(rv, st, bz):
                continue
            n_push += 1
            ok = [b for b, sp in braces if sp.volume_ok(rv, st, bz)]
            for b in ok:
                kinds_ok[b["kind"]] = kinds_ok.get(b["kind"], 0) + 1
            if ok:
                reach.append((st, ok))
        st_["step2_push_reach"] = n_push
        st_["step2_brace_reach_by_kind"] = kinds_ok
        st_["step2_reach"] = len(reach)
        if not reach:
            return []
        cams = P0.camera_poses(self, [st for st, _ in reach])
        classed = []
        for (st, ok), cam in zip(reach, cams):
            pts = th.stack([geo["C"], geo["E_low"], ok[0]["B"]])
            if not P0.in_frame(self, cam, pts):
                continue
            ang = P0.axis_angle(self, cam, geo["C"])
            k = P0.angle_class(ang)
            if k is None:
                continue
            st.focus = geo["C"].clone()
            st.extras.update({"axis_angle_deg": math.degrees(ang), "angle_class": k, "brace_kinds": [b["kind"] for b in ok]})
            classed.append((st, k))
        st_["step3_visible"] = len(classed)
        best = P0.best_class(classed)
        # the best angle class first (§5.2 step 4); the other class is kept behind it as the fall-back pool of step 5 — with
        # two survivors in the tight class the brace kinds those two reach decided everything (first-demo loop 13:44)
        best_ids = {id(st) for st in best}
        rest = [st for st, _ in classed if id(st) not in best_ids]
        for st in best:
            st.extras["class_pool"] = "best"
        for st in rest:
            st.extras["class_pool"] = "fallback"
        st_["step3_best_class"] = len(best)
        return best + rest

    # ------------------------------------------------------------------ wall spots (kept from v1)

    def _walls(self) -> Dict[str, Any]:
        return {l.prim_path: o for o in self.world.scene.objects if o.category == "walls" for l in o.links.values()}

    def _flat_patch(self, point, normal, w, h, wall) -> bool:
        nu, nv = self.param("flat_patch_rays", [5, 5])
        tol = float(self.param("flat_patch_tolerance_m", 0.01))
        tangent = th.tensor([-float(normal[1]), float(normal[0]), 0.0])
        up = th.tensor([0.0, 0.0, 1.0])
        starts, ends = [], []
        for u in th.linspace(-w / 2, w / 2, int(nu)).tolist():
            for v in th.linspace(-h / 2, h / 2, int(nv)).tolist():
                p = point + tangent * u + up * v
                starts.append(p + normal * 0.5)
                ends.append(p - normal * 0.1)
        hits = self.world.raycast(th.stack(starts), th.stack(ends))
        prims = {l.prim_path for l in wall.links.values()}
        for hh in hits:
            if not hh["hit"] or hh["rigidBody"] not in prims or abs(float(hh["distance"]) - 0.5) > tol:
                return False
        return True

    def _front_clear(self, point, normal, w, h) -> bool:
        depth = float(self.param("front_clearance_m", 0.35))
        tangent = th.tensor([-float(normal[1]), float(normal[0]), 0.0])
        c = point + normal * depth / 2
        corners = []
        for su in (-1, 1):
            for sv in (-1, 1):
                for sd in (-1, 1):
                    corners.append(c + tangent * su * (w / 2 + 0.05) + th.tensor([0.0, 0.0, sv * (h / 2 + 0.05)]) + normal * sd * depth / 2)
        corners = th.stack(corners)
        blo, bhi = corners.min(dim=0).values, corners.max(dim=0).values
        for o in self.world.scene.objects:
            if o.category in STRUCTURE or o is self.robot or o.name in self.world.spawned:
                continue
            lo, hi = o.aabb
            if bool((lo <= bhi).all() and (hi >= blo).all()):
                return False
        return True

    def _in_room(self, xy) -> bool:
        return P0.in_task_rooms(self, xy)

    def _region_spots(self, regions, w, h, hang_z: float) -> List[Dict[str, Any]]:
        step = float(self.param("region_seed_step_m", 0.25))
        walls = self._walls()
        out = []
        for r in regions:
            p0, p1 = th.tensor([float(v) for v in r["p0"]]), th.tensor([float(v) for v in r["p1"]])
            n = th.tensor([float(r["normal"][0]), float(r["normal"][1]), 0.0])
            n = n / th.linalg.norm(n)
            z = float(r.get("hang_z", hang_z))
            k = max(1, int(round(float(th.linalg.norm(p1 - p0)) / step)))
            for t in th.linspace(0.0, 1.0, k + 1).tolist():
                xy = p0 * (1 - t) + p1 * t
                start = th.tensor([float(xy[0]), float(xy[1]), z]) + n * 0.5
                end = th.tensor([float(xy[0]), float(xy[1]), z]) - n * 0.15
                hh = self.world.raycast(start.unsqueeze(0), end.unsqueeze(0))[0]
                if not hh["hit"] or hh["rigidBody"] not in walls:
                    continue
                wall = walls[hh["rigidBody"]]
                if r.get("wall") and wall.name != r["wall"]:
                    continue
                point = G.t(hh["position"]).clone()
                point[2] = z
                normal = G.t(hh["normal"]).clone()
                normal[2] = 0.0
                if float(th.linalg.norm(normal)) < 0.5:
                    continue
                normal = normal / th.linalg.norm(normal)
                if not self._flat_patch(point, normal, w, h, wall) or not self._front_clear(point, normal, w, h):
                    continue
                out.append({"wall_name": wall.name, "wall_link": hh["rigidBody"], "point": point, "normal": normal, "hang_z": z, "region": r.get("id")})
        return out

    def _ray_spots(self, w, h, hang_z: float) -> List[Dict[str, Any]]:
        floor = self.obj("floor")
        lo, hi = floor.aabb
        centre = th.tensor([float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), hang_z])
        n_dirs = int(self.param("spot_search_rays", 24, "wall_ray_directions"))
        dirs = th.stack([th.tensor([math.cos(a), math.sin(a), 0.0]) for a in th.linspace(0, 2 * math.pi, n_dirs + 1)[:-1].tolist()])
        starts = centre.unsqueeze(0).repeat(n_dirs, 1)
        hits = self.world.raycast(starts, starts + dirs * 12.0)
        walls = self._walls()
        out = []
        for hh in hits:
            if not hh["hit"] or hh["rigidBody"] not in walls:
                continue
            wall = walls[hh["rigidBody"]]
            point = G.t(hh["position"]).clone()
            point[2] = hang_z
            normal = G.t(hh["normal"]).clone()
            normal[2] = 0.0
            if float(th.linalg.norm(normal)) < 0.5:
                continue
            normal = normal / th.linalg.norm(normal)
            if not self._flat_patch(point, normal, w, h, wall) or not self._front_clear(point, normal, w, h):
                continue
            out.append({"wall_name": wall.name, "wall_link": hh["rigidBody"], "point": point, "normal": normal, "hang_z": hang_z, "region": None})
        return out

    def find_spots(self) -> List[Dict[str, Any]]:
        b = self.choice.bindings
        allowed = b.get(self.wall_role) or b.get("wall") or b.get("support")
        allowed = set(allowed) if isinstance(allowed, list) and allowed and isinstance(allowed[0], str) else None
        regions = b.get("support_regions") or b.get("wall_regions") or []
        rooms = set(self.gp.rooms(self.scene) or [])
        if rooms:
            regions = [r for r in regions if r.get("room") in (None, *rooms)]
        w, h = self.patch_extents()
        margin = float(self.param("flat_patch_margin_m", 0.10))
        hang_z = float(self.obj("floor").aabb[1][2]) + self.hang_height()
        spots = self._region_spots(regions, w + margin, h + margin, hang_z) if regions else self._ray_spots(w + margin, h + margin, hang_z)
        test_d = float(self.param("in_room_test_m", 0.8))
        spots = [s for s in spots if self._in_room(s["point"][:2] + s["normal"][:2] * test_d)]
        if allowed is not None:
            spots = [s for s in spots if s["wall_name"] in allowed]
        for i, s in enumerate(spots):
            s["id"] = i
        self.template_facts["spot_search"] = {"regions": len(regions), "patch_wh": [w, h], "n_spots": len(spots), "walls": sorted({s["wall_name"] for s in spots})}
        return spots

    # ------------------------------------------------------------------ template (§5.5)

    def prepare_bindings(self):
        """Wall spots for the spawned (scaled) target, the wall with the most spots bound as the support role."""
        target = self.target
        self.hull = self.world.local_hull(target)
        self.com_local = self._com_local(target)
        frame = self.reviewed_frame()
        try:
            self.front, self.side, self.up = frame_axes(frame["front_axis"], frame["up_axis"], frame["level_axis"])
        except (ValueError, KeyError) as e:
            raise Reject("missing_frame", f"{self.gp.alias}: the recorded frame {frame} is not a valid §5.5 frame: {e}")
        self.frame = frame
        self.kit.sync_world()
        spots = self.find_spots()
        if not spots:
            raise Reject("no_wall", f"{self.gp.alias}: no flat, free wall patch ({self.template_facts['spot_search']})")
        counts: Dict[str, int] = {}
        for s in spots:
            counts[s["wall_name"]] = counts.get(s["wall_name"], 0) + 1
        wall_name = sorted(counts, key=lambda n: (-counts[n], n))[0]
        self.spots = [s for s in spots if s["wall_name"] == wall_name]
        self.bind_scene_object(self.wall_role, self.world.obj(wall_name))
        t, w, h = self.dims()
        # the frame used (the alias' per-model record, never a default) with the derived level axis for the record
        self.template_facts.update({"frame": {**frame, "level_axis_derived": "+" + "xyz"[int(th.argmax(self.side.abs()))] if float(self.side.sum()) > 0
                                              else "-" + "xyz"[int(th.argmax(self.side.abs()))], "symmetric_180": self.symmetric_180(), "source": "customs"},
                                    "dims": {"thickness": t, "width": w, "height": h}, "scale": self.choice.models["target"].scale, "wall": wall_name,
                                    "com_local": self.com_local.tolist()})

    def sample_template(self):
        try:
            self._sample_template()
        finally:  # the facts survive a template rejection (diagnostics in template_meta)
            self.template_meta["template_facts"] = _jsonable(self.template_facts)

    def _sample_template(self):
        """Step-1 survivors per spot (cached; A6: checked in the posture the stance records), the template filter at
        ±max|δ| of the applied range (30° in the spec; 25° under A8), the scale_hint diagnostic when nothing survives."""
        lam = tuple(self.param("lateral_range_m", [-0.40, 0.40]))
        _, hi = self.deviation_range()
        filt = min(math.radians(float(self.param("template_filter_deviation_deg", 30.0))), hi)
        filt_deg = round(math.degrees(filt))
        self.template_facts["template_filter_deg"] = filt_deg
        posture, fingers = self.stance_posture()
        self.template_facts["stance_arm_posture"] = posture
        kept = []
        for s in self.spots:
            face = self.face_of(s)
            scope = {"scene": self.world.scene_model, "spot": [round(float(v), 3) for v in s["point"].tolist()], "normal": [round(float(v), 3) for v in s["normal"].tolist()],
                     "sampler": "alignment_wall"}
            self.step1[s["id"]] = P0.phase2_step1(self, face, P0.D_RANGE_DEFAULT, lam, arms_excluded=True, cache_scope=scope, posture=posture, fingers=fingers)
            n_ok = {}
            for sign in (1.0, -1.0):
                geo = self.contacts_for(s, sign * filt)
                n_ok[f"{sign * filt_deg:+.0f}"] = len(self.screen(self.step1[s["id"]], geo))
            s["n_survivors"] = n_ok
            s["weight"] = sum(n_ok.values())
            if all(v > 0 for v in n_ok.values()):
                kept.append(s)
        self.template_facts["spots"] = [{"id": s["id"], "point": s["point"].tolist(), "normal": s["normal"].tolist(), "n_step1": len(self.step1[s["id"]]),
                                         "survivors_pm_filter": s.get("n_survivors")} for s in self.spots]
        if not kept:
            hint = self.scale_hint(filt)
            self.template_meta["scale_hint"] = hint
            raise Reject("no_stance", f"{self.gp.alias}: no wall spot passes Phase 2 steps 1–3 for δ = ±{filt_deg}° at scale {self.choice.models['target'].scale} "
                                      f"(arms {posture}; {[s.get('n_survivors') for s in self.spots]}); scale_hint = {hint}")
        self.spots = kept

    def scale_hint(self, filt: float) -> Optional[float]:
        """§5.5 diagnostic: the first uniform scale s ∈ {0.9, …, 0.5} at which some spot passes steps 1–3 for δ = ±30°
        (hull and COM scaled about the root; the reach cache is robot-only). Never applied."""
        for s in [float(v) for v in self.param("scale_hint_steps", [0.9, 0.8, 0.7, 0.6, 0.5])]:
            hull, com = self.hull * s, self.com_local * s
            for spot in self.spots:
                ok = True
                for sign in (1.0, -1.0):
                    geo = self.contacts_for(spot, sign * filt, hull, com)
                    if not self.screen(self.step1[spot["id"]], geo):
                        ok = False
                        break
                if ok:
                    return s
        return None

    def face_of(self, spot: Dict[str, Any]) -> FaceFrame:
        return P0.make_face("wall", spot["normal"], th.tensor([float(spot["point"][0]), float(spot["point"][1]), float(spot["hang_z"])]))

    # ------------------------------------------------------------------ instance (§5.5)

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        tname = self.tname
        weights = np.array([float(s["weight"]) for s in self.spots])
        spot = self.spots[int(rng.choice(len(self.spots), p=weights / weights.sum()))]
        lo, hi = self.deviation_range()
        delta = float(rng.uniform(lo, hi)) * (1.0 if rng.random() < 0.5 else -1.0)
        geo = self.contacts_for(spot, delta)
        pose_t = geo["pose_tilted"]
        # the target hangs at its init pose for every check of this attempt (hinge installed after apply)
        self.target.set_position_orientation(*pose_t)
        self.target.keep_still()
        self.kit.sync_world()
        face = self.face_of(spot)
        gen_p0 = dict(self.param("prompt0") or {})
        beta_max = math.radians(float(gen_p0.get("beta_max_deg", 35.0)))
        d_min, d_max = float(gen_p0.get("d_min_m", 0.5)), float(gen_p0.get("d_max_m", 1.5))
        hull_t = geo["hull_tilted"]
        init = P0.phase1_init_pose(self, face, beta_max, d_min, d_max, lambda: (hull_t, hull_t.mean(dim=0)), rng)
        init.name = "init"
        self.sidecar.add_stance(init)
        stats: Dict[str, Any] = {}
        survivors = self.screen(self.step1[spot["id"]], geo, stats)
        if not survivors:
            raise Reject("no_manipulation_stance", f"spot {spot['id']} δ {math.degrees(delta):.1f}°: no Phase-2 survivor ({stats})")
        spec_p, braces = self.contact_specs(geo)
        arms = [geo["push"]["arm"], geo["brace_arm"]]
        pools = [[st for st in survivors if st.extras.get("class_pool", "best") == "best"], [st for st in survivors if st.extras.get("class_pool") == "fallback"]]
        errors: List[str] = []

        def extra_vis(cam, _geo=geo) -> bool:
            return P0.in_frame(self, cam, th.stack([_geo["C"], _geo["E_low"]]))

        posture, fingers = self.stance_posture()
        done = False
        for pool in pools:
            for _ in range(min(len(pool), 8)):
                i, manip = P0.draw_manipulation(self, pool, rng, arms, focus=geo["C"], extra_visibility=extra_vis, posture=posture, fingers=fingers,
                                                clearance_ignore=[self.target])
                kinds = pool[i].extras.get("brace_kinds", [])
                pool.pop(i)
                if manip is None:
                    errors.append("drawn tilt fails the re-check")
                    continue
                try:
                    self._solve_from(manip, init, geo, spec_p, [(b, sp) for b, sp in braces if b["kind"] in kinds])
                except Reject as e:
                    errors.append(f"{e.reason}: {e}")
                    self.sidecar.stances.pop("manipulation", None)
                    self.sidecar.transitions.clear()
                    self.sidecar.paths.clear()
                    self.sidecar.contacts.clear()
                    continue
                done = True
                break
            if done:
                break
        if not done:
            raise Reject("push_unreachable", f"{len(survivors)} Phase-2 survivors (δ {math.degrees(delta):+.1f}°, {stats}), none passes step 5 / step 6: " + "; ".join(errors[-4:]))
        manip = self.sidecar.stance("manipulation")
        d = Draft()
        d.object_poses[tname] = pose_t
        d.goal_poses[tname] = geo["pose_level"]
        d.robot_q = self.stance_q(init)
        d.base_pose2d = tuple(init.base_pose2d)
        d.settle_free = []
        eqs = list(self.param("orientation_equivalences", []) or [])
        tags = self.gp.orientation_equivalences("target", self.choice.models["target"].category, self.choice.models["target"].model)
        if tags:
            eqs += [list(q) for q in tags]
        if self.symmetric_180():
            eqs.append(half_turn_equivalence(self.front))
        d.pose_targets[tname] = pose_target(orientation=geo["pose_level"][1], orientation_metric="full", orientation_tolerance=self.tol("orientation", 0.08),
                                            equivalences=eqs)
        d.summary.update({"wall": spot["wall_name"], "spot": spot["id"], "deviation_deg": math.degrees(delta), "push_arm": geo["push"]["arm"], "brace_arm": geo["brace_arm"],
                          "brace_kind": self.sidecar.contacts["brace"].brace_kind, "scale": self.choice.models["target"].scale, "phase2": stats,
                          "init_stance": P0.summarize_stance(init), "manipulation_stance": P0.summarize_stance(manip), "symmetric_180": self.symmetric_180(),
                          "frame": dict(self.template_facts.get("frame") or {})})
        self.sidecar.describe(tname, self.choice.models["target"].category.replace("_", " "), deviation_deg=math.degrees(delta),
                              push_distance_cm=100 * geo["push"]["distance"])
        self._spot = spot
        return d

    def _solve_from(self, manip: Stance, init: Stance, geo: Dict[str, Any], spec_p: P0.ContactSpec, braces: List[Tuple[Dict[str, Any], P0.ContactSpec]]):
        """§5.5 step 5 (brace first, target not ignored; push with the brace locked, target ignored for the contact
        poses; combined check with the push arm at its pre-contact) and step 6; fills the sidecar."""
        target = self.target
        # §3.6: the wall is removed for the contact poses. Wall contacts (pads parallel to the wall on the mid-thickness
        # plane, 2 cm off it) keep that distance along their whole approach (10 cm along the wall), and the 1 cm check
        # margin flags the closed paddle's sphere model at every pose within ≈ 5 cm of the wall — so the IK runs with the
        # wall removed along the whole brace chain and the whole push chain (the target stays in the world except for the
        # push poses). The wall is then put BACK for a second, margin-0 check of every solved configuration (``wall_ok``):
        # the pads (2 cm off the wall, sphere half thickness ≤ 1.8 cm) pass it, a wrist / wrist camera inside the wall does
        # not (first demo 2026‑09‑16: ``right_realsense_link in contact with walls_urngpl_0`` at the side_high brace) —
        # both wrist turns of each contact are tried and the turn whose configuration clears the wall is recorded.
        wall = self.support_objects()
        push_arm, brace_arm = geo["push"]["arm"], geo["brace_arm"]
        manip.arms = [push_arm, brace_arm]
        # the oracle closes the gripper before a push / brace (oracle doc §3.6): the IK seed / locked joints carry closed
        # fingers; the idle arm of each solve sits in the stance's recorded A6 posture (``manip.q``, set by the draw)
        q0 = self.ikc.q_locked(manip, fingers=C.FINGER_CLOSED)

        def wall_ok(q: th.Tensor, ignore: Sequence[Any] = ()) -> bool:
            """The configuration with the wall present, margin 0 (``ignore`` = the target for push poses)."""
            with self.ikc.ignoring(list(ignore)):
                return bool(self.kit.collision_free(q, margin=0.0)[0])

        cam = self.camera_pose_at(manip)

        def tips_in_frame(poses: Sequence[Any]) -> bool:
            """QA ``gripper_out_of_fov`` (oracle/qa.py): from the pre-pose to the end of a brace / push the acting arm's
            fingertips must be in the head camera frame (5 %) — checked here for the recorded key poses (a pre-pose on
            the margin is a sampler defect, first demo 13:42: the side_high brace hand 0.2 m above the camera)."""
            pts = th.stack([C.fingertip(p) for p in poses])
            return P0.in_frame(self, cam, pts)

        # brace candidates in order (kind order, both wrist turns; first whose pre-contact and contact pass), then the push
        # arm (both wrist turns) with the brace arm locked at its solution; when the push chain fails with that brace, the
        # next brace candidate is tried before the caller moves to the next survivor
        why: List[str] = []
        solved = None
        for b, sp in braces:  # the target is *not* ignored for the brace (§5.5: ≥ 2 cm clear of the tilted hull)
            tag = f"{b['kind']}/{b.get('wrist', 'n+')}"
            if not tips_in_frame([b["pre"], b["pose"]]):
                why.append(f"{tag}: brace fingertips out of the head camera frame at the pre-contact / contact")
                continue
            r_pre_b = self.ikc.exact([b["pre"]], manip, brace_arm, ignore_for_last=wall, q_seed=q0, cache=False)
            if not r_pre_b.ok:
                why.append(f"{tag}: brace pre-contact")
                continue
            # the oracle's Brace approaches along the straight segment pre → contact (``line_waypoints``, 3 poses) and the
            # program check re-solves those poses (§3.9: screening at the intermediate ones, exact at the ends); the same
            # chain is solved here exactly, pose by pose, so every pose has a remembered witness solution
            chain = [tuple(p) for p in C.line_waypoints(b["pre"], b["pose"], 3)]
            with self.ikc.ignoring(wall):
                res = self.ikc.exact(chain, manip, brace_arm, q_seed=q0, cache=False)
            if not res.ok:
                why.append(f"{tag}: brace contact / approach ({res.reason})")
                continue
            if not wall_ok(res.q) or not wall_ok(r_pre_b.q):
                why.append(f"{tag}: brace arm inside the wall (wrist / camera) at the contact or pre-contact")
                continue
            arm_b = self.kit.robot.arm_control_idx[brace_arm]
            q_seed = q0.clone()
            q_seed[arm_b] = res.q[arm_b]
            arm_p = self.kit.robot.arm_control_idx[push_arm]
            for push in geo["push"].get("variants") or [geo["push"]]:
                ptag = f"{tag}/push-{push.get('wrist', 'n+')}"
                if not tips_in_frame([push["pre"], push["init"], push["mid"], push["goal"]]):
                    why.append(f"{ptag}: push fingertips out of the head camera frame along pre / arc")
                    continue
                # the push pre-contact lies 10 cm along -a from E_low (the low *side* of the target), i.e. alongside the hull
                # with ~2 cm of air: the target is removed for it as for the contact poses (the approach pre -> init runs along
                # the side and the pads only meet the target at init; documented relaxation of §3.6 for the wall push)
                r_pre = self.ikc.exact([push["pre"]], manip, push_arm, ignore_for_last=[target] + wall, q_seed=q_seed, cache=False)
                if not r_pre.ok:
                    alone = self.ikc.exact([push["pre"]], manip, push_arm, ignore_for_last=[target] + wall, q_seed=q0, cache=False)
                    why.append(f"{ptag}: push pre-contact with the brace locked" + ("" if alone.ok else " (and alone: the push arm cannot reach its pre-contact)"))
                    continue
                r_init = self.ikc.exact([push["init"]], manip, push_arm, ignore_for_last=[target] + wall, q_seed=q_seed, cache=False)
                if not r_init.ok:
                    why.append(f"{ptag}: push init contact")
                    continue
                ok_mid = self.ikc.screen([push["mid"]], manip, push_arm, ignore=[target] + wall, q_seed=q_seed)
                if not bool(ok_mid.all()):
                    why.append(f"{ptag}: push arc waypoint δ/2 (screening)")
                    continue
                r_goal = self.ikc.exact([push["goal"]], manip, push_arm, ignore_for_last=[target] + wall, q_seed=q_seed, cache=False)
                if not r_goal.ok:
                    why.append(f"{ptag}: push goal contact")
                    continue
                if not all(wall_ok(r.q, [target]) for r in (r_pre, r_init, r_goal)):
                    why.append(f"{ptag}: push arm inside the wall (wrist / camera) along the arc")
                    continue
                q_comb = q_seed.clone()
                q_comb[arm_p] = r_pre.q[arm_p]
                # brace contact + push pre-contact together: the target removed (the push hand hovers 2 cm off its side), the
                # wall present at margin 0, the arms against each other and the scene checked
                if not wall_ok(q_comb, [target]):
                    why.append(f"{ptag}: combined configuration (brace contact + push pre-contact) collides")
                    continue
                solved = (b, push)
                break
            if solved is not None:
                break
        if solved is None:
            raise Reject("push_unreachable" if any("push" in w for w in why) else "brace_unreachable",
                         f"no (brace, push) chain passes the exact IK from the manipulation stance: {'; '.join(why) or 'no brace candidate'}")
        b, push = solved
        manip.contacts = ["push", "brace"]
        manip.extras["closing_axes"] = {"brace": b.get("wrist", "n+"), "push": push.get("wrist", "n+")}
        self.sidecar.add_stance(manip)
        P0.connect(self, init, manip, float(self.param("path_max_m", P0.PATH_MAX_M)))
        tname = self.tname
        self.sidecar.add_contact(ContactRecord("push", push_arm, push["pre"], push["init"], push["mid"], push["goal"], geo["push"]["direction"], geo["push"]["distance"],
                                               object=tname))
        self.sidecar.add_contact(ContactRecord("brace", brace_arm, b["pre"], b["pose"], direction=b["a"], brace_kind=b["kind"], object=tname))
        self.sidecar.waypoint_policy = "push arc: screening at δ/2, exact at init and goal; brace: exact pre / approach / contact; wall-present margin-0 re-check of every contact configuration"

    # ------------------------------------------------------------------ pipeline hooks (§5.5 Phase 3)

    def _clear_hinge(self):
        remove_hinge(self.hinge)
        self.hinge = None

    def apply_draft_extras(self, draft: Draft):
        target, wall = self.target, self.wall
        self._clear_hinge()
        link = next((l for l in wall.links.values() if l.prim_path == self._spot["wall_link"]), None)
        anchor = self.com_world(target)
        self.hinge = create_hinge(target, anchor, self._spot["normal"], parent_link=link)
        target.keep_still()
        target.enable_gravity()
        import omnigibson as og

        og.sim.step_physics()
        target.keep_still()
        draft.extras["constraints"] = [self.hinge.to_tro(self.tname, self.bindings.one(self.wall_role).name)]

    def settle_entities(self, draft: Draft):
        return [self.target]

    def extra_checks(self, draft: Draft):
        tname = self.tname
        p, q = self.target.get_position_orientation()
        drift = G.quat_angle(q, draft.object_poses[tname][1])
        if drift > float(self.param("hinge_drift_max_rad", 0.03)):
            raise Reject("hinge_drift", f"object rotated {drift:.3f} rad on the hinge while settling")
        px = self.world.visible_pixels(exclude_structure=False)
        n_px = int(self.param("min_object_pixels", 200))
        visible = [n for n, c in px.items() if c >= n_px and n != self.robot.name]
        if len(visible) < int(self.param("min_visible_objects", 5)):
            raise Reject("too_few_visible", f"{len(visible)} objects visible: {visible}")
        draft.summary["visible_objects"] = visible
        cam = self.camera_pose_at(self.sidecar.stance("init"))
        if not self.frustum_ok(cam, self.hull_at((p.float(), q.float())), 0.05):
            raise Reject("not_in_frame", "the settled tilted target is not fully in the init image (5 %)")

    def refine_targets_from_goal(self, draft: Draft):
        pass  # orientation-only target; the level orientation is exact by construction

    def write_instance(self, draft: Draft, index: int, seed: int) -> str:
        path = super().write_instance(draft, index, seed)
        self._clear_hinge()
        return path

    def on_reject(self, err: Reject):
        self._clear_hinge()
        super().on_reject(err)

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        manip = self.sidecar.stances.get("manipulation")
        brace = self.sidecar.contacts.get("brace")
        if manip is None or brace is None:
            return []
        c = self.choice.models["target"]
        return [{"scope": {"scene": self.scene, "target": f"{c.category}/{c.model}", "wall": self._spot["wall_name"]}, "type": "contacts",
                 "push_arm": self.sidecar.contacts["push"].arm, "brace_arm": brace.arm, "brace_kind": brace.brace_kind, "deviation_deg": round(draft.summary["deviation_deg"], 1),
                 "stand_off": manip.stand_off, "lateral": manip.lateral, "trunk": [manip.translate, round(manip.tilt, 3)], "hard": False}]
