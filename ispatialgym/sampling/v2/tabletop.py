"""§4 tabletop initialization as functions on a ``SamplerV2`` (spec ``docs/instance_sampling_spec.md`` §4.2–4.7; README
"tabletop.py — signatures the reference task calls").

Phases (every rule number refers to the spec)::

    phase1_edge_and_standoff(s, top, rng, stand_off=0.45)      -> (EdgeFrame, Stance)   §4.2  any edge, s = 0.45 + overhang (>= 5 cm)
    phase1b_init_trunk(s, init, H_max)                          -> Stance               §4.3  Δ_aim / Δ_min / Δ_max -> t*, tilt 0; b, R_img
    phase1c_lateral(s, init, margins, rng)                      -> Stance               §4.4  margins table -> lateral draw, checks (1)–(3)
    phase2_manipulation_stance(s, init, target_half_extent)     -> list[Stance]         §4.5  stand-off x translate x continuous tilt, band cells,
                                                                                              focus per survivor, best angle class
    phase3_target(s, manip, rng, grasps, hull, rest, ignore)    -> (obj_pose, GraspRecord)  §4.6 step 1 (F at the focus, exact IK)

plus the shared helpers of §4.6 steps 2–6 (``place_partner``, ``place_distractors``, ``all_in_frame``,
``whole_body_check_at``, ``brace_second_arm``), the §4.7 ``placement_stance`` and the reach-band access of §4.1 / §3.10
(``BandSpec``, ``ensure_bands``: ``ReachBands.load`` on a cache hit, ``BandComputer`` in-process on a miss, written back to the
cache directory).

Camera constants come from the FK once per robot (``camera_fk``: pitch at tilt 0, x_c(t), z_c(t), hand height below the
camera) and are cached on the module; the ``[pilot]`` design values of §4.3 (22.6°, 0.22 m, 0.19 m) are the defaults of the
pure helpers and what the tests compare the FK against.

Pure helpers (no simulator, unit-tested in ``tests/test_v2_tabletop.py``): ``lateral_margins`` (the §4.4 table),
``init_trunk_delta`` (§4.3 clamp), ``CameraFK`` (conventions FK), ``cube_flip_plan`` / ``cube_tilt_pair`` /
``facing_coarse_yaw`` (§6.5 cube grasp and visible-face rules), ``distractor_spacing`` / ``exclude_lookalikes`` (§4.6 step 3,
§3.12), ``axial_init_yaw`` (§6.10). Everything that touches the simulator imports ``ik`` / ``sim_utils`` lazily so the
module (and the tests) import without OmniGibson.

Frames: base frame x ahead, lateral + = robot's left (§3.3); band cells are base-frame (x, lateral) at *any* stance of
the same support height (the band is robot-relative, §4.1: "the stand-off is not a dimension").
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.v2.stances import EdgeFrame, GraspRecord, Pose, Stance

log = logging.getLogger("ispatialgym.sampling.v2")

# ----------------------------------------------------------------------------------------------------------------
# constants (§4; the [pilot] numbers of §4.3 are the defaults of the pure helpers)
# ----------------------------------------------------------------------------------------------------------------

STAND_OFF = 0.45  # §4.2
STAND_OFFS = (0.35, 0.45, 0.55)  # §4.5
TRANSLATES = tuple(float(t) for t in RB.TRANSLATES)  # §3.3 grid (0.95 = deepest trunk)
TILT_GRID = tuple(float(k) for k in RB.TILTS)  # §3.3 grid tilts of the bands
N_TILT_SAMPLES = 7  # §4.5 "tilt continuous from 0 to the torso_joint3 limit": samples of the scan
ANGLE_CLASSES_DEG = (8.0, 20.0)  # §4.5 step 3
IN_FRAME_MARGIN = 0.05  # §3.4 "in frame (5 %)"
FOCUS_RADIUS = 0.05  # §4.6 step 1: target within 5 cm of the focus
YAW_JITTER = math.radians(22.5)  # §4.6 step 1
YAWS = tuple(RB.YAWS)  # §4.6 step 1: 8 fixed yaws (45° steps)
SUPPORT_MARGIN = 0.02  # §3.2 footprint inside the top shrunk by 2 cm
TRAY_MARGIN = 0.005  # §3.2 0.5 cm on a tray floor
MIN_GAP = 0.03  # §3.2 footprint clearance target–partner
OVERHANG_MIN = 0.05  # §4.2 AABB overhang beyond the raster edge counts when >= 5 cm
HANDS_ABOVE = 0.10  # §4.3 hands >= 10 cm above the tallest object
HAND_BELOW_CAMERA_PILOT = 0.19  # §4.3 [pilot]
CAMERA_X_PILOT = 0.22  # §4.3 [pilot]
CAMERA_PITCH_PILOT = math.radians(22.6)  # §4.3 [pilot]
MAX_TRANSLATE = 0.95  # §3.3: translate 1.0 self-collides, 0.95 is the deepest trunk everywhere
TRUNK_SWEEP_TRANSLATES = TRANSLATES  # §4.4 check (3)
GENERIC_YAWS = 2  # §4.1 generic pinch: 2 wrist yaws x 2 approach pitches
GENERIC_TILTS_DEG = (0.0, 30.0)
Z_OFFSET = 0.003  # §3.2 z = support top + bottom offset + 3 mm
PLACE_DROP = 0.003


def _reject(reason: str, message: str = "", **details):
    from ispatialgym.sampling.sim_utils import Reject

    raise Reject(reason, message, **details)


def _ik():
    from ispatialgym.sampling.v2 import ik as IK

    return IK


def _top(s, top=None):
    """The support top a sampler works on (``s.top`` as the reference task keeps it, or the one Phase 1a recorded)."""
    if top is not None:
        return top
    t = getattr(s, "top", None)
    if t is None:
        t = getattr(s, "_tabletop_top", None)
    if t is None:
        raise ValueError("no support top: call phase1_edge_and_standoff(s, top, ...) first or set s.top")
    return t


def _t(x) -> th.Tensor:
    return x if isinstance(x, th.Tensor) else th.tensor(x, dtype=th.float32)


# ----------------------------------------------------------------------------------------------------------------
# pure helpers: §4.3 trunk solve, §4.4 margins table
# ----------------------------------------------------------------------------------------------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    """``clamp(x, lo, hi) = min(max(x, lo), hi)``: when ``lo > hi`` the upper bound wins (torch / numpy semantics; §4.3
    ``Δ = clamp(Δ_aim, Δ_min, Δ_max)``: the near edge stays in frame, the hand clearance is then checked by the
    whole-body test of §4.4)."""
    return min(max(float(x), float(lo)), float(hi))


def init_trunk_delta(edge_depth: float, d_ws: float, d_near: float, x_c: float, H_max: float, hand_below: float = HAND_BELOW_CAMERA_PILOT,
                     pitch: float = CAMERA_PITCH_PILOT, phi_m: float = math.radians(46.6), hands_above: float = HANDS_ABOVE) -> Dict[str, float]:
    """§4.3: the camera height above the top ``Δ`` for the init trunk (tilt 0).

    ``edge_depth`` = distance base centre -> front edge (s + overhang), ``d_ws`` / ``d_near`` = depth from the edge of
    the workspace centre / near edge, ``x_c`` = camera ahead of the base, ``H_max`` = tallest object to be placed,
    ``hand_below`` = untucked hands below the camera, ``pitch`` = camera pitch down at tilt 0, ``phi_m`` = image-border
    angle including the visibility margin (half the vertical FOV minus the margin)::

        Δ_aim = edge_depth + d_ws − x_c                    45° elevation onto the workspace centre
        Δ_min = H_max + hands_above + hand_below           hands >= 10 cm above the tallest object
        Δ_max = (edge_depth + d_near − x_c) · tan(pitch + φ_m)   workspace near edge inside the frame
        Δ     = clamp(Δ_aim, Δ_min, Δ_max)
    """
    d_aim = float(edge_depth) + float(d_ws) - float(x_c)
    d_min = float(H_max) + float(hands_above) + float(hand_below)
    d_max = (float(edge_depth) + float(d_near) - float(x_c)) * math.tan(float(pitch) + float(phi_m))
    return {"delta": clamp(d_aim, d_min, d_max), "delta_aim": d_aim, "delta_min": d_min, "delta_max": d_max, "conflict": d_min > d_max}


def lateral_margins(margins: Dict[str, Any], lam_min: float, lam_max: float, b: float, R_img: float) -> Tuple[float, float, Dict[str, Any]]:
    """§4.4 margin table: ``(m_R, m_L, info)`` for the task row named by ``margins["kind"]`` (a row without ``kind`` but
    with ``relation`` is the IORS row the reference task passes: ``w_t, w_p, W_p, D_p_max, safety, relation, g_NT``).

    ``lam_min <= 0`` (right end of the band union), ``lam_max`` (left end), ``b`` = lateral offset of the image border at
    the source-tray depth, ``R_img`` = surface distance from the target (centre line) to the image border. Rows::

        iors (ontop, under, inside, nextto, touching): |λ_min| + w_t + D_p + 2·w_p + 0.02 / max(0, λ_max) + w_t + D_p + 2·w_p + 0.02,
                                                       D_p = min(W_p + 0.03, R_img); nextto: D_p < g_NT + 0.03 -> info["reject"]
        attachment:        |λ_min| + max(w_target, w_partner) + 0.02 / max(0, λ_max) + max(w_target, w_partner) + 0.02
        axial:             |λ_min| + w_diag + 0.02 / max(0, λ_max) + w_diag + 0.02,  w_diag = ½√(long² + short²)
        block_design:      |λ_min| + edge/2 + (edge + 0.001) + 0.02 / max(0, λ_max) + edge/2 + 0.02   (0.0125 + 0.026 + 0.02 for the 2.5 cm cube)
        disembedding:      b + lateral_margin + 0.02 on both sides (lateral_margin = 0.437 [pilot])
        articulation_small (§5.4): |λ_min| + w_a + 0.02 / max(0, λ_max) + w_a + 0.02
    """
    kind = margins.get("kind") or ("iors" if "relation" in margins else None)
    if kind is None:
        raise ValueError(f"margins row needs a 'kind' (or 'relation' for the IORS row): {sorted(margins)}")
    safety = float(margins.get("safety", 0.02))
    right = abs(min(float(lam_min), 0.0))
    left = max(0.0, float(lam_max))
    info: Dict[str, Any] = {"kind": kind, "lam_min": float(lam_min), "lam_max": float(lam_max), "b": float(b), "R_img": float(R_img), "safety": safety}
    if kind == "iors":
        w_t, w_p = float(margins["w_t"]), float(margins["w_p"])
        D_p_max = float(margins.get("D_p_max", float(margins.get("W_p", 2 * w_p)) + 0.03))
        D_p = min(D_p_max, float(R_img))
        info["D_p"] = D_p
        if margins.get("relation") == "nextto" and margins.get("g_NT") is not None:
            g_lo = float(margins["g_NT"]) + float(margins.get("nextto_gap_extra", 0.03))
            info["g_NT_lo"] = g_lo
            if D_p < g_lo:
                info["reject"] = ("pair_gap", f"D_p {D_p:.3f} < g_NT + 0.03 = {g_lo:.3f}")
        m_R = right + w_t + D_p + 2 * w_p + safety
        m_L = left + w_t + D_p + 2 * w_p + safety
    elif kind == "attachment":
        w = max(float(margins["w_target"]), float(margins["w_partner"]))
        m_R, m_L = right + w + safety, left + w + safety
    elif kind == "axial":
        if "w_diag" in margins:
            w_diag = float(margins["w_diag"])
        else:
            w_diag = 0.5 * math.hypot(float(margins["long"]), float(margins["short"]))
        info["w_diag"] = w_diag
        m_R, m_L = right + w_diag + safety, left + w_diag + safety
    elif kind == "block_design":
        edge = float(margins.get("edge", C.BLOCK_EDGE))
        pitch = edge + float(margins.get("block_gap", 0.001))
        info.update({"half_edge": edge / 2, "pitch": pitch})
        m_R = right + edge / 2 + pitch + safety  # half edge + one pitch to the robot's right of the anchor
        m_L = left + edge / 2 + safety
    elif kind == "disembedding":
        lm = float(margins.get("lateral_margin", 0.437))
        m_R = m_L = float(b) + lm + safety
    elif kind == "articulation_small":
        w_a = float(margins["w_a"])
        m_R, m_L = right + w_a + safety, left + w_a + safety
    else:
        raise ValueError(f"unknown margins row {kind!r}")
    info.update({"m_R": m_R, "m_L": m_L})
    return m_R, m_L, info


def lateral_bounds(edge_length: float, m_R: float, m_L: float) -> Optional[Tuple[float, float]]:
    """§4.4 step 2: ``[−L/2 + m_R, L/2 − m_L]`` or None when ``m_R + m_L > L`` (skip the edge)."""
    lo, hi = -0.5 * float(edge_length) + float(m_R), 0.5 * float(edge_length) - float(m_L)
    return (lo, hi) if lo <= hi else None


# ----------------------------------------------------------------------------------------------------------------
# pure helpers: camera FK tables (§3.3 "from FK once per robot")
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class CameraFK:
    """Camera offsets at tilt 0 as functions of the JoyLo translate: ``x_c(t)`` ahead of the base, ``z_c(t)`` above the
    floor (base_link at ``NOMINAL_BASE_Z``), the constant pitch down, and the untucked hand height below the camera."""

    ts: np.ndarray
    x_c: np.ndarray
    z_c: np.ndarray
    pitch: float
    hand_below: float = HAND_BELOW_CAMERA_PILOT
    fy: float = 459.0
    height_px: int = 1080

    def x_at(self, t: float) -> float:
        return float(np.interp(clamp(t, 0.0, 1.0), self.ts, self.x_c))

    def z_at(self, t: float) -> float:
        return float(np.interp(clamp(t, 0.0, 1.0), self.ts, self.z_c))

    def t_of_z(self, z: float, t_max: float = MAX_TRANSLATE) -> float:
        """``z_c⁻¹``: the translate whose camera height is ``z`` (monotone decreasing table), clamped to ``[0, t_max]``."""
        zs = self.z_c[::-1]
        ts = self.ts[::-1]
        if z >= float(zs[-1]):
            return 0.0
        if z <= float(zs[0]):
            return float(t_max)
        return clamp(float(np.interp(z, zs, ts)), 0.0, t_max)

    def phi_m(self, margin: float = IN_FRAME_MARGIN) -> float:
        """Image-border angle including the visibility margin: ``atan((H/2 − m·H) / fy)`` (§4.3 φ_m)."""
        return math.atan((0.5 * self.height_px - margin * self.height_px) / self.fy)

    def tilt_limit(self, translate: float) -> float:
        """Most negative tilt (camera down) the ``torso_joint3`` limit allows at ``translate`` (§4.5 scan range)."""
        j = C.M.TRUNK_TILT_JOINT_INDEX
        return float(C.TORSO_JOINT_LIMITS[0][j] - C.translate_to_torso_q(translate)[j])


def camera_table(n: int = 101) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(ts, x_c, z_c, pitch)`` at tilt 0 from the conventions FK (base_link at ``NOMINAL_BASE_Z`` above the floor)."""
    ts = np.linspace(0.0, 1.0, n)
    xs, zs, pitches = [], [], []
    for t in ts:
        pos, quat = C.camera_pose(C.torso_q(float(t), 0.0), base_pose=(0.0, 0.0, 0.0))
        axis = C.quat_rotate(quat, th.tensor([0.0, 0.0, -1.0]))
        xs.append(float(pos[0]))
        zs.append(float(pos[2]))
        pitches.append(math.atan2(-float(axis[2]), math.hypot(float(axis[0]), float(axis[1]))))
    return ts, np.asarray(xs), np.asarray(zs), float(np.mean(pitches))


_FK_CACHE: Dict[Any, CameraFK] = {}


def camera_fk(s=None, hand_below: Optional[float] = None) -> CameraFK:
    """The camera FK tables, once per robot (cached on the kit); without a sampler the pure conventions tables with the
    ``[pilot]`` hand height."""
    key = id(s.kit) if s is not None else None
    if key in _FK_CACHE:
        return _FK_CACHE[key]
    ts, xs, zs, pitch = camera_table()
    fk = CameraFK(ts, xs, zs, pitch)
    if s is not None:
        cm = s.world.camera_model
        fk.fy, fk.height_px = float(cm.fy), int(cm.height)
        if hand_below is None:
            try:
                q = s.kit.q_config(arms="untucked", translate=0.5, tilt=0.0)
                z_cam = float(s.kit.camera_pose(q)[0][2])
                z_hand = max(float(s.kit.eef_pose(q, arm)[0][2]) for arm in C.M.ARMS)
                hand_below = z_cam - z_hand
            except Exception as e:  # noqa: BLE001 - the pilot value stands in when the kit cannot answer
                log.warning("camera_fk: hand height from the kit failed (%r); using the [pilot] value %.2f", e, HAND_BELOW_CAMERA_PILOT)
                hand_below = HAND_BELOW_CAMERA_PILOT
        fk.hand_below = float(hand_below)
        log.info("camera FK: pitch %.1f deg, x_c(0.5) %.3f m, z_c(0..1) %.3f..%.3f m, hands %.3f m below the camera", math.degrees(fk.pitch),
                 fk.x_at(0.5), fk.z_at(0.0), fk.z_at(1.0), fk.hand_below)
    elif hand_below is not None:
        fk.hand_below = float(hand_below)
    _FK_CACHE[key] = fk
    return fk


# ----------------------------------------------------------------------------------------------------------------
# pure helpers: §6.5 cube grasp rule and visible-face rule
# ----------------------------------------------------------------------------------------------------------------

FACE_AXES = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0), "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0)}


def opposite_face(face: str) -> str:
    return ("-" if face[0] == "+" else "+") + face[1]


def face_normal_world(face: str, quat) -> th.Tensor:
    return C.quat_rotate(quat, th.tensor(FACE_AXES[face]))


def up_face(quat) -> str:
    return max(FACE_AXES, key=lambda f: float(face_normal_world(f, quat)[2]))


def cube_tilt_pair(theta_place_deg: float, flip_sign: int) -> Tuple[float, float]:
    """§6.5: ``(tilt_pick, tilt_place)`` of a flipping block — the in-hand transform is rigid, so
    ``d_pick = R_a(∓90°)·d_place``: ``tilt_pick = tilt_place − flip_sign·90`` (conventions ``cube_pick_tilt``)."""
    return C.cube_pick_tilt(theta_place_deg, flip_sign), float(theta_place_deg)


def cube_flip_plan(block_quat, goal_face: str, table_yaw: float, yaw_max_deg: float = 45.0) -> Optional[Dict[str, Any]]:
    """§6.5 smallest rotation from a block's init orientation to the table-aligned goal orientation with ``goal_face`` up:
    yaw-align (|ψ| <= ``yaw_max_deg``) to the table axes (``table_yaw`` = world yaw of the goal grid's +x, i.e. the
    robot's right), then — when the goal face is not already up — one flip of ±90° about the horizontal table axis that
    brings it up. Returns ``None`` when the init up face is the opposite of the goal face (§4.6: never allowed).

    Output: ``yaw_deg`` ψ (the yaw-align), ``flip_axis`` (unit, world; None without a flip), ``flip_sign`` (±1 / 0),
    ``face_pair`` = the block axis parallel to the flip axis after the yaw-align (the pinch axis, ``"x"`` / ``"y"`` /
    ``"z"``), ``goal_quat`` = the table-aligned orientation with ``goal_face`` up reached by ψ then the flip."""
    q0 = _t(block_quat)
    up0 = up_face(q0)
    if up0 == opposite_face(goal_face):
        return None
    # yaw-align: the block's horizontal face normals are 90° apart, as are the table axes, so exactly one ψ in
    # (−45°, 45°] aligns them (|ψ| <= yaw_max_deg by construction for a cube)
    horiz = [f for f in FACE_AXES if f not in (up0, opposite_face(up0))]
    n0 = face_normal_world(horiz[0], q0)
    theta = math.atan2(float(n0[1]), float(n0[0]))
    psi = min((C.wrap_angle(float(table_yaw) + k * math.pi / 2 - theta) for k in range(4)), key=abs)
    if abs(psi) > math.radians(yaw_max_deg) + 1e-9:
        return None
    q1 = C.qmul(C.quat_from_yaw(psi), q0)
    if goal_face == up0:
        return {"yaw_deg": math.degrees(psi), "flip_axis": None, "flip_sign": 0, "face_pair": None, "goal_quat": q1, "flips": False}
    # one flip about the horizontal table axis a = n_goal × ẑ: R_a(+90°) n_goal = +ẑ brings the (side) goal face up
    n_goal = face_normal_world(goal_face, q1)
    axis = th.cross(n_goal, th.tensor([0.0, 0.0, 1.0]), dim=0)
    axis = axis / th.linalg.norm(axis)
    q_flip = C.quat_from_axis_angle(axis, math.pi / 2)
    q_goal = C.qmul(q_flip, q1)
    if float(face_normal_world(goal_face, q_goal)[2]) < 0.99:  # numerical guard: the other sense
        q_flip = C.quat_from_axis_angle(axis, -math.pi / 2)
        q_goal = C.qmul(q_flip, q1)
        sign = -1
    else:
        sign = 1
    # pinch axis = the block axis parallel to the flip axis after the yaw-align (pads flat, spec §6.5)
    pair = max("xyz", key=lambda ax: abs(float(th.dot(C.quat_rotate(q1, th.tensor(FACE_AXES["+" + ax])), axis))))
    return {"yaw_deg": math.degrees(psi), "flip_axis": axis, "flip_sign": sign, "face_pair": pair, "goal_quat": q_goal, "flips": True}


def facing_coarse_yaw(block_quat_up: th.Tensor, goal_face: str, toward_xy) -> float:
    """§4.6 visible-face rule: the coarse yaw (multiple of 90°) about the vertical that turns ``goal_face`` (a side face
    of the block at orientation ``block_quat_up``) toward the robot (``toward_xy`` = −inward, unit). Returns the yaw
    (rad) to apply (``R_z(yaw) ∘ block_quat_up``)."""
    tw = _t([float(toward_xy[0]), float(toward_xy[1]), 0.0])
    best, best_dot = 0.0, -2.0
    for k in range(4):
        yaw = k * math.pi / 2
        n = face_normal_world(goal_face, C.qmul(C.quat_from_yaw(yaw), _t(block_quat_up)))
        d = float(th.dot(n, tw))
        if d > best_dot:
            best, best_dot = yaw, d
    return best


def face_points(block_pose: Pose, face: str, edge: float = C.BLOCK_EDGE, n: int = 3, inset: float = 0.2) -> th.Tensor:
    """3 x 3 grid of world points on ``face`` of a cube (§4.6 visible-face ray test), inset from the face border."""
    p, q = _t(block_pose[0]), _t(block_pose[1])
    normal = th.tensor(FACE_AXES[face])
    ax = "xyz"[[abs(v) for v in FACE_AXES[face]].index(1.0)]
    others = [a for a in "xyz" if a != ax]
    u = th.tensor(FACE_AXES["+" + others[0]])
    v = th.tensor(FACE_AXES["+" + others[1]])
    half = 0.5 * edge
    coords = th.linspace(-half * (1 - inset), half * (1 - inset), n)
    pts = []
    for a in coords.tolist():
        for b in coords.tolist():
            pts.append(normal * half + u * a + v * b)
    return C.transform_points((p, q), th.stack(pts))


# ----------------------------------------------------------------------------------------------------------------
# pure helpers: §4.6 step 3 distractor spacing, §3.12 look-alikes, §6.10 yaw draw
# ----------------------------------------------------------------------------------------------------------------


def distractor_spacing(pinch_width: float, extra: float = 0.01, open_width: float = C.GRIPPER_OPEN_M) -> float:
    """§4.6 step 3: ``x = (open width 0.10 − pinch width)/2 + 0.01`` (``Grasp.pad_clearance`` + margin)."""
    return 0.5 * (float(open_width) - float(pinch_width)) + float(extra)


def grasp_pinch_width(grasp) -> Optional[float]:
    """Pinch width of a library grasp (``meta.width``; the assisted-grasp results record it), None when unknown / inf."""
    w = (getattr(grasp, "meta", None) or {}).get("width")
    if w is None:
        return None
    w = float(w)
    return None if not math.isfinite(w) else w


def exclude_lookalikes(pool: Sequence[Tuple[str, str]], target: Tuple[str, str], groups: Sequence[Sequence[Tuple[str, str]]],
                       rng=None) -> List[Tuple[str, str]]:
    """§3.12 rule 1: the pool minus the target and every member of its look-alike group, then at most one model per
    look-alike group (the kept member is the first in ``pool`` order, or a random one with ``rng``); exact duplicates of
    the target never survive."""
    grp_of: Dict[Tuple[str, str], int] = {}
    for gi, g in enumerate(groups):
        for m in g:
            grp_of[tuple(m)] = gi
    target = tuple(target)
    banned = set(groups[grp_of[target]]) if target in grp_of else set()
    banned.add(target)
    cands = [tuple(m) for m in pool if tuple(m) not in banned]
    if rng is not None and cands:
        cands = [cands[int(i)] for i in rng.permutation(len(cands))]
    seen_groups: Set[int] = set()
    out: List[Tuple[str, str]] = []
    for m in cands:
        gi = grp_of.get(m)
        if gi is not None:
            if gi in seen_groups:
                continue
            seen_groups.add(gi)
        out.append(m)
    return out


def axial_init_yaw(rng, aligned_yaw: float, deviation_range_deg: Sequence[float] = (20.0, 70.0)) -> Tuple[float, float]:
    """§6.10: ``yaw = aligned yaw ± U(20°, 70°)`` with a random sign; returns ``(yaw, delta)`` (delta signed, rad)."""
    lo, hi = float(deviation_range_deg[0]), float(deviation_range_deg[1])
    delta = math.radians(float(rng.uniform(lo, hi))) * (1.0 if rng.random() < 0.5 else -1.0)
    return C.wrap_angle(float(aligned_yaw) + delta), delta


# ----------------------------------------------------------------------------------------------------------------
# object poses on a support
# ----------------------------------------------------------------------------------------------------------------


def object_pose_on(s, hull: th.Tensor, rest_quat, xy, yaw: float, z_support: float, drop: float = Z_OFFSET) -> Pose:
    """§3.2: orientation ``R_z(yaw) ∘ rest``, z = support top + bottom offset + 3 mm."""
    quat = C.qmul(C.quat_from_yaw(yaw), _t(rest_quat))
    pos = th.tensor([float(xy[0]), float(xy[1]), float(z_support) + s.world.bottom_offset(hull, quat) + float(drop)])
    return pos, quat


def hull_extents(hull: th.Tensor, rest_quat) -> th.Tensor:
    """AABB extents (3,) of a local hull at the rest orientation."""
    pts = C.transform_points((th.zeros(3), _t(rest_quat)), hull)
    return pts.max(dim=0).values - pts.min(dim=0).values


def half_extent(hull: th.Tensor, rest_quat) -> float:
    """w_x of §4.4: half the larger horizontal bbox side at the rest orientation."""
    ext = hull_extents(hull, rest_quat)
    return 0.5 * float(max(float(ext[0]), float(ext[1])))


def height_of(hull: th.Tensor, rest_quat) -> float:
    return float(hull_extents(hull, rest_quat)[2])


def hull_corners_world(hull: th.Tensor, pose: Pose) -> th.Tensor:
    pts = C.transform_points(pose, hull)
    lo, hi = pts.min(dim=0).values, pts.max(dim=0).values
    return th.tensor([[float(lo[0] if sx == 0 else hi[0]), float(lo[1] if sy == 0 else hi[1]), float(lo[2] if sz == 0 else hi[2])]
                      for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)])


def box_corners(centre_xy, half: float, z0: float, z1: float) -> th.Tensor:
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    return th.tensor([[cx + sx * half, cy + sy * half, z] for sx in (-1, 1) for sy in (-1, 1) for z in (z0, z1)])


# ----------------------------------------------------------------------------------------------------------------
# grasp library access (pipeline suffixes tolerated: the facade writes ``t`` / ``i`` / ``a`` / ``f``)
# ----------------------------------------------------------------------------------------------------------------

PIPELINE_ALIASES = {"tabletop": "tabletop", "t": "tabletop", "insert": "insert", "i": "insert", "fl": "insert", "floating": "floating", "f": "floating",
                    "": "floating", "attachment": "attachment", "a": "attachment", "handle_half": "floating", "h": "floating"}


def library_grasps(s, role: str = "target", default_pipeline: str = "tabletop", kinds: Optional[Sequence[str]] = None):
    """``(grasps, key, rest_quat)`` of the role's model for the pipeline the customs entry requires (``roles.<role>.
    require.grasps``); ``Reject("missing_grasps:<pipeline>")`` when the library has no admissible row."""
    from ispatialgym.sampling import grasp_library as GL

    choice = s.choice.models[role]
    pipeline = PIPELINE_ALIASES.get(s.gp.roles[role].grasp_pipeline() or default_pipeline, s.gp.roles[role].grasp_pipeline() or default_pipeline)
    kinds = list(kinds) if kinds else [pipeline]
    try:
        key = GL.resolve_key(choice.category, choice.model, choice.grasp_key, kinds)
        grasps = GL.load_grasps(key)
    except (FileNotFoundError, ValueError) as e:
        _reject(f"missing_grasps:{pipeline}", f"{choice.category}/{choice.model}: {e}")
    if not grasps:
        _reject(f"missing_grasps:{pipeline}", f"{choice.category}/{choice.model}: no admissible grasp row")
    try:
        rest = C.rest_quat(key)
    except FileNotFoundError:
        rest = C.IDENTITY_QUAT.clone()
    return grasps, key, rest


# ----------------------------------------------------------------------------------------------------------------
# reach bands (§4.1, §3.10 caches)
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class BandSpec:
    """What a task's reach band is computed for: the grasped object (``object`` = ``category/model``, ``grasps`` = its
    library rows, ``hull`` / ``rest_quat`` for the poses; ``generic_pinch`` for blocks / push poses), the spawn ``scale``
    (§3.10 cache key), ``placed`` objects (cache key tokens), the support height override ``z_support`` (a tray floor),
    the lattice override ``lattice_world`` (default: the free cells of the top), extra ``ignore`` objects for the grasp
    pose (a tray) and the ``arm``."""

    object: Optional[str] = None
    grasp_key: Optional[str] = None
    grasps: Optional[List[Any]] = None
    hull: Optional[th.Tensor] = None
    rest_quat: th.Tensor = field(default_factory=lambda: C.IDENTITY_QUAT.clone())
    scale: Optional[List[float]] = None
    placed: List[str] = field(default_factory=list)
    z_support: Optional[float] = None
    lattice_world: Optional[th.Tensor] = None
    ignore: List[Any] = field(default_factory=list)
    arm: str = "right"
    generic_pinch: bool = False
    height: float = 0.05
    eef_of_grasp: Optional[Callable[[Any], Pose]] = None  # T_obj->eef of a grasp entry (default ``g.pose``)
    contact: bool = False  # A1: B3's generic CONTACT set of the brace arm (closed gripper) instead of the pinch set

    @staticmethod
    def from_sampler(s, role: str = "target") -> "BandSpec":
        """The target's band from the attributes the reference task keeps (``grasps``, ``grasp_key``, ``hull[role]``,
        ``rest[role]``); a task module without library grasps sets ``s.band_spec`` explicitly."""
        mc = s.choice.models[role]
        grasps = getattr(s, "grasps", None)
        hull = (getattr(s, "hull", None) or {}).get(role) if isinstance(getattr(s, "hull", None), dict) else None
        rest = (getattr(s, "rest", None) or {}).get(role) if isinstance(getattr(s, "rest", None), dict) else None
        if hull is None:
            hull = s.world.local_hull(s.obj(role))
        if rest is None:
            rest = C.IDENTITY_QUAT.clone()
        return BandSpec(object=f"{mc.category}/{mc.model}", grasp_key=getattr(s, "grasp_key", None) or mc.grasp_key, grasps=grasps, hull=hull,
                        rest_quat=_t(rest), scale=list(mc.scale) if mc.scale else None, generic_pinch=not grasps, height=height_of(hull, rest))


def ensure_bands(s, spec: Optional[BandSpec] = None, top=None, edge: Optional[EdgeFrame] = None, force: bool = False) -> RB.ReachBands:
    """The reach band of the sampler's support and object (cached on ``s.bands``): ``ReachBands.load`` from the cache
    directories (§3.10), else computed in-process with ``BandComputer`` (the function the CLI runs) and written to
    ``reach_bands.cache_root()``."""
    cached = getattr(s, "bands", None)
    if cached is not None and not force:
        return cached
    top = _top(s, top)
    spec = spec or getattr(s, "band_spec", None) or BandSpec.from_sampler(s)
    scene, support = s.world.scene_model, top.name
    gkey = "generic" if spec.generic_pinch else spec.grasp_key
    try:
        bands = RB.ReachBands.load(scene, support, spec.object, gkey, placed=spec.placed, scale=spec.scale)
        log.info("reach band cache hit for %s / %s (%s): %s", support, spec.object, gkey, bands.path)
    except FileNotFoundError:
        log.info("reach band cache miss for %s / %s (%s): computing in-process", support, spec.object, gkey)
        bands = compute_bands(s, spec, top, edge)
    s.bands = bands
    return bands


def compute_bands(s, spec: BandSpec, top, edge: Optional[EdgeFrame] = None, lattice: float = RB.LATTICE, out_dir: Optional[str] = None,
                  trunks: Optional[Sequence[Tuple[float, float]]] = None) -> RB.ReachBands:
    """§4.1 steps 1–4 in-process (``reach_bands.BandComputer``) at the band stance (edge midpoint, s = 0.45 + overhang) of
    ``edge`` (default: the longest free edge), written as ``<cache>/<name>.npz/.json`` like the CLI."""
    world, kit = s.world, s.kit
    support = world.obj(top.name)
    bc = RB.BandComputer(world, kit, arms=(spec.arm,), lattice=lattice)
    stance = bc.stance_at_edge(top, support, RB.STAND_OFF, edge.name if edge is not None else None)
    z = float(spec.z_support) if spec.z_support is not None else float(top.z)
    lattice_world = spec.lattice_world if spec.lattice_world is not None else top.cells(free_only=True)
    ignore = [support] + list(spec.ignore)
    h = float(spec.height)
    contact_mode = bool(spec.contact and hasattr(bc, "contact_height"))
    if spec.contact and not contact_mode:
        log.info("reach_bands.BandComputer has no contact mode yet: the left band uses the generic pinch set with the left arm")
    if contact_mode:
        # B3's contact set: closed fingers, side contacts at CONTACT_HEIGHT_FRACTION·h / top presses, the classes bitmask
        bc.contact_height = h
        bc.fingers = float(getattr(C, "FINGER_CLOSED", 0.0))
        pose_fn, grasp_sets, max_off = None, None, 0.0
        z_eval = z
    elif spec.generic_pinch or not spec.grasps:
        pose_fn, grasp_sets, max_off = None, None, 0.0
        z_eval = z + max(0.6 * h, 0.03)
    else:
        hull, rest = spec.hull, spec.rest_quat

        def pose_fn(world_xy, yaw, _hull=hull, _rq=rest, _z=z):
            return object_pose_on(s, _hull, _rq, world_xy, yaw, _z)

        eef_of = spec.eef_of_grasp or (lambda g: g.pose)
        grasp_sets = [_BandGrasp(g.name, eef_of(g)) for g in spec.grasps]
        max_off = max(float(th.linalg.norm(_t(g.position))) for g in grasp_sets)
        z_eval = z
    kit.sync_world()
    gen_trunks = getattr(s, "gen", lambda k, d=None: d)("band_trunks", None)
    trunks = list(trunks) if trunks is not None else ([tuple(float(v) for v in t) for t in gen_trunks] if gen_trunks else RB.trunk_grid(z - kit.floor_z))
    t0 = time.time()
    result = bc.compute(stance, lattice_world, z_eval, pose_fn, grasp_sets, ignore, arm=spec.arm, trunks=trunks, max_grasp_offset=max_off)
    kit.sync_world()
    lat_all = np.concatenate([c[:, 1] for c in result["cells"].values() if len(c)]) if any(len(c) for c in result["cells"].values()) else np.zeros(0)
    meta: Dict[str, Any] = {"scene": world.scene_model, "rooms": list(world.rooms or []), "support": top.name, "lattice": lattice, "stand_off": RB.STAND_OFF,
                            "arms": "untucked", "grid": {"translates": list(RB.TRANSLATES), "tilts": list(RB.TILTS), "low_top_tilt": RB.LOW_TOP_TILT},
                            "seeds": 12, "pre_grasp": RB.PRE_GRASP, "yaws": list(RB.YAWS), "mode": "tabletop", "smoke": False, "scale": spec.scale,
                            "placed_specs": list(spec.placed), "object": spec.object, "grasp_key": None if spec.generic_pinch else spec.grasp_key,
                            "generic_pinch": bool(spec.generic_pinch or not spec.grasps),
                            "grasp_names": [g.name for g in (grasp_sets or [])], "rest_quat_xyzw": _t(spec.rest_quat).tolist(), "stance": stance,
                            "top": top.to_json(), "z_support": z_eval, "trunks": result["trunks"],
                            "union_extent": [float(lat_all.min()), float(lat_all.max())] if len(lat_all) else None,
                            "n_reachable_total": int(sum(len(c) for c in result["cells"].values())), "time_s": round(time.time() - t0, 1),
                            "computed_by": "sampling.v2.tabletop.compute_bands", "arm": spec.arm}
    if contact_mode:  # B3's contact-band fields (``ReachBands.find(arm="left", contact_set=..., height=...)`` matches them)
        meta.update({"contact_set": CONTACT_SET, "height_class": height_class(h), "contact_height_fraction": CONTACT_HEIGHT_FRACTION,
                     "classes": [c["name"] for c in getattr(RB, "CONTACT_CLASSES", [])], "fingers": bc.fingers})
    elif grasp_sets is None:
        meta["classes"] = list(getattr(RB, "GENERIC_CLASSES", []))
    key = RB._key({"scene": world.scene_model, "support": top.name, "z": round(z_eval, 2), "stance": stance["edge"], "s": RB.STAND_OFF, "object": spec.object,
                   "scale": spec.scale, "grasps": [(g.name, _t(g.position).tolist(), _t(g.quat).tolist()) for g in (grasp_sets or [])] or "generic",
                   "rest": _t(spec.rest_quat).tolist(), "placed": list(spec.placed), "grid": meta["grid"], "lattice": lattice, "arms": "untucked", "seeds": 12})
    meta["key"] = key
    out_dir = out_dir or RB.cache_root()
    os.makedirs(out_dir, exist_ok=True)
    oname = (spec.object or "generic").replace("/", "-")
    name = f"tabletop__{world.scene_model}__{top.name}__{oname}__{key}"
    arrays = {RB.trunk_key(tr, ti): np.asarray(c, dtype=np.float32).reshape(-1, 2) for (tr, ti), c in result["cells"].items()}
    for (tr, ti), p in result.get("pairs", {}).items():
        arrays[RB.trunk_key(tr, ti) + "__pairs"] = np.asarray(p, dtype=np.int32).reshape(-1, 3)
    for (tr, ti), m in (result.get("classes") or {}).items():  # B3's per-cell class bitmask (generic / contact sets)
        if m is not None and len(m):
            arrays[RB.trunk_key(tr, ti) + "__classes"] = np.asarray(m, dtype=np.int64).reshape(-1)
    path = os.path.join(out_dir, name + ".npz")
    np.savez_compressed(path, **arrays)
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    cells = {k: np.asarray(v, dtype=np.float32).reshape(-1, 2) for k, v in result["cells"].items() if not _trunk_collides(result["trunks"], k)}
    pairs = {k: np.asarray(v, dtype=np.int32).reshape(-1, 3) for k, v in result.get("pairs", {}).items() if k in cells}
    classes = {k: np.asarray(v, dtype=np.int64).reshape(-1) for k, v in (result.get("classes") or {}).items() if k in cells and v is not None and len(v)}
    log.info("reach band computed for %s / %s: %d reachable cells over %d trunks in %.0fs -> %s", top.name, spec.object, meta["n_reachable_total"], len(cells),
             meta["time_s"], path)
    try:
        return RB.ReachBands(meta, cells, pairs, path, classes)
    except TypeError:  # loader without the classes field
        return RB.ReachBands(meta, cells, pairs, path)


def _trunk_collides(entries: Sequence[Dict[str, Any]], key: Tuple[float, float]) -> bool:
    for e in entries:
        if abs(float(e["translate"]) - key[0]) < 1e-6 and abs(float(e["tilt"]) - key[1]) < 1e-6:
            return bool(e.get("collides"))
    return False


class _BandGrasp:
    """Minimal grasp-like entry for the band computation (``.name``, ``.pose`` = T_obj->eef, ``.position`` / ``.quat``)."""

    def __init__(self, name: str, pose: Pose):
        self.name = name
        self.position, self.quat = _t(pose[0]), _t(pose[1])

    @property
    def pose(self) -> Pose:
        return self.position, self.quat


def band_cells_world(bands: RB.ReachBands, stance: Stance, translate: Optional[float] = None, tilt: Optional[float] = None) -> th.Tensor:
    """The band cells of the grid trunk (same translate, nearest tilt) in the world at ``stance`` (robot-relative cells,
    §4.5 step 2 "band shifted with the stand-off")."""
    tr = stance.translate if translate is None else translate
    ti = stance.tilt if tilt is None else tilt
    cells = bands.band(float(tr), float(ti))
    if len(cells) == 0:
        return th.zeros(0, 2)
    return th.tensor(bands.to_world(cells, stance.base_pose2d), dtype=th.float32)


def band_union_cells(bands: RB.ReachBands) -> th.Tensor:
    """Every band cell over every trunk (base frame), deduplicated."""
    if not bands.cells:
        return th.zeros(0, 2)
    allc = np.concatenate([c for c in bands.cells.values() if len(c)]) if any(len(c) for c in bands.cells.values()) else np.zeros((0, 2))
    if len(allc) == 0:
        return th.zeros(0, 2)
    uniq = np.unique(np.round(allc, 4), axis=0)
    return th.tensor(uniq, dtype=th.float32)


def workspace(s, bands: RB.ReachBands, stance: Stance, top, erosion: float = 0.0) -> Dict[str, Any]:
    """§4.3 workspace = band union ∩ top of this edge at ``stance``: world cells, depths from the edge and laterals from
    the centre line (``d_near`` / ``d_ws`` / ``lam_min`` / ``lam_max``)."""
    base = band_union_cells(bands)
    out: Dict[str, Any] = {"cells_world": th.zeros(0, 2), "cells_base": th.zeros(0, 2), "depths": [], "laterals": []}
    if len(base) == 0:
        return out
    world_xy = th.tensor(bands.to_world(base.numpy(), stance.base_pose2d), dtype=th.float32)
    keep = [i for i in range(len(world_xy)) if top.contains(world_xy[i], erosion)]
    if not keep:
        return out
    cw = world_xy[keep]
    cb = base[keep]
    depths = [stance.local(c)[0] for c in cw]
    lats = [float(v) for v in cb[:, 1].tolist()]
    out.update({"cells_world": cw, "cells_base": cb, "depths": depths, "laterals": lats, "d_near": min(depths), "d_ws": float(np.mean(depths)),
                "d_far": max(depths), "lam_min": min(lats), "lam_max": max(lats)})
    return out


def store_feasible_set(bands: RB.ReachBands, stance: Stance, F: Sequence[Tuple[float, Any]], yaws: Sequence[float] = YAWS) -> None:
    """§4.6 step 1 / §3.10: store the feasible set F of the focus cell under (trunk, cell) in the band (``pairs`` +
    ``meta.feasible_sets``); an empty F removes the cell from the band. In-memory (and written back to the band's
    ``.npz`` / ``.json`` when the band was loaded from or saved to a cache file)."""
    if bands is None or stance.focus is None:
        return
    k = bands.nearest_trunk(float(stance.translate), float(stance.tilt))
    if k is None:
        return
    cell_base = bands.to_base(np.asarray(stance.focus[:2].tolist()), stance.base_pose2d)[0]
    i = bands.cell_index(k[0], k[1], cell_base)
    fs = bands.meta.setdefault("feasible_sets", [])
    if i is None:
        return
    if not F:
        cells = bands.cells[k]
        keep = np.array([j for j in range(len(cells)) if j != i], dtype=int)
        bands.cells[k] = cells[keep] if len(keep) else np.zeros((0, 2))
        p = bands.pairs.get(k)
        if p is not None and len(p):
            p = p[p[:, 0] != i]
            remap = {old: new for new, old in enumerate(keep.tolist())}
            p = np.array([(remap[c], y, g) for c, y, g in p.tolist() if c in remap], dtype=np.int32).reshape(-1, 3)
            bands.pairs[k] = p
        fs.append({"trunk": list(k), "cell": [float(v) for v in cell_base], "feasible": [], "removed": True})
    else:
        names = bands.meta.get("grasp_names") or []
        rows = []
        for yaw, g in F:
            yi = min(range(len(yaws)), key=lambda j: abs(C.wrap_angle(float(yaw) - yaws[j])))
            gname = getattr(g, "name", str(g))
            gi = names.index(gname) if gname in names else -1
            rows.append((i, yi, gi))
        p = bands.pairs.get(k)
        p = np.zeros((0, 3), dtype=np.int32) if p is None else p
        bands.pairs[k] = np.concatenate([p, np.array(rows, dtype=np.int32).reshape(-1, 3)]) if rows else p
        fs.append({"trunk": list(k), "cell": [float(v) for v in cell_base], "feasible": [[round(float(y), 3), getattr(g, "name", str(g))] for y, g in F]})
    if bands.path:
        try:
            arrays = {RB.trunk_key(tr, ti): np.asarray(c, dtype=np.float32).reshape(-1, 2) for (tr, ti), c in bands.cells.items()}
            for (tr, ti), pp in bands.pairs.items():
                arrays[RB.trunk_key(tr, ti) + "__pairs"] = np.asarray(pp, dtype=np.int32).reshape(-1, 3)
            np.savez_compressed(bands.path, **arrays)
            with open(bands.path[:-4] + ".json", "w") as f:
                json.dump(bands.meta, f, indent=1, default=str)
        except OSError as e:  # noqa: BLE001 - a read-only cache is not a sampling error
            log.debug("band write-back skipped (%r)", e)


# ----------------------------------------------------------------------------------------------------------------
# A1 — left-arm contact bands, per-cell approach classes and the joint two-arm placement (owner decision 2026-09-16;
# spec §4.1 band scope, §4.4 margins = union of both bands, §4.6 step 6 two-arm IK)
# ----------------------------------------------------------------------------------------------------------------

#: approach pitches of the generic push / stop set (deg from straight down toward the robot's forward), B3's
#: ``TILTS_GENERIC``: the owner's A1 decision added 45° and 60° so far cells (0.9–1.0 m) keep an IK-feasible arc. B3's
#: per-cell class bitmask (``<trunk>__classes`` in the ``.npz``, names in ``meta["classes"]``) names the generic classes
#: ``p<pitch>_y<wrist yaw>`` (pitch-major) and the left contact classes ``near|far|left|right_v|h`` / ``top_y0|y90``.
APPROACH_PITCHES_DEG: Tuple[float, ...] = tuple(float(t) for t in getattr(RB, "TILTS_GENERIC", (0.0, 30.0, 45.0, 60.0)))
#: B3's contact set of the brace arm (``reach_bands.CONTACT_SET``): closed gripper, pads on a vertical face at 0.75 of
#: the object's height (``CONTACT_HEIGHT_FRACTION``) or a top press; one band per 2 cm object height class
CONTACT_SET = getattr(RB, "CONTACT_SET", "generic_contact")
CONTACT_HEIGHT_FRACTION = float(getattr(RB, "CONTACT_HEIGHT_FRACTION", 0.75))
#: a point is "in a band cell" when it lies within this fraction of the lattice from a cell centre (half a cell = the
#: cell's own square)
CELL_TOLERANCE = 0.5
#: brace-point reach proxy used while no left contact band exists (base frame, m): the measured band envelopes of the
#: B3 run (x 0.48–0.80 m ahead, |λ| <= 0.56) with a small allowance
BRACE_PROXY_X = (0.30, 0.88)
BRACE_PROXY_LATERAL = 0.65


def height_class(height: float) -> float:
    """B3's object height class (2 cm bins, the class value = the rounded height) of the left contact bands."""
    if hasattr(RB, "height_class"):
        return float(RB.height_class(float(height)))
    return round(max(1, int(round(float(height) / 0.02))) * 0.02, 4)


def contact_height(top_z: float, height: float) -> float:
    """Height of a side contact of the brace arm on an object ``height`` tall standing on ``top_z`` (B3's contact set:
    pads at ``CONTACT_HEIGHT_FRACTION`` of the height)."""
    return float(top_z) + CONTACT_HEIGHT_FRACTION * float(height)


def left_band_compute() -> bool:
    """Whether a missing left contact band is computed in-process (15–25 min per support / height class on an L40S):
    ``ISG_LEFT_BAND_COMPUTE=1``; default off — until B3's left bands exist the brace side is decided by the exact two-arm
    IK alone and the instance records ``left_band: false``."""
    return os.environ.get("ISG_LEFT_BAND_COMPUTE", "0").strip().lower() in ("1", "true", "yes")


def left_contact_object(cls: float) -> str:
    """The ``meta.object`` token under which an in-process left contact band is cached (distinct from every real object
    so that a loader without the arm filter never mistakes it for a right-arm generic band)."""
    return f"contact-left/h{cls:.2f}"


def load_left_contact_band(s, height: float, *, top=None, edge: Optional[EdgeFrame] = None, compute: bool = True) -> Optional[RB.ReachBands]:
    """A1: the left arm's generic CONTACT band (closed gripper, brace approach) of the support for the object's height
    class, cached on ``s.bands_left[cls]``. Loader order: B3's left-arm mode
    ``ReachBands.load(scene, support, arm="left", contact_set="generic_contact", height=h)``; a loader that predates the
    keywords (``TypeError``) or has no such band (``FileNotFoundError``) falls back to the in-process band of
    ``compute_bands`` with ``BandSpec(arm="left", generic_pinch=True, contact=True)`` (B3's ``BandComputer`` contact
    mode when it has one — closed fingers, the contact classes —, else the generic top-down / pitched set with the left
    arm) under the object token ``left_contact_object(cls)`` with the meta fields B3's ``find`` matches (``arm``,
    ``contact_set``, ``height_class``). ``compute=False`` returns None on a miss (unit tests, dry runs)."""
    cls = height_class(height)
    cache = getattr(s, "bands_left", None)
    if cache is None:
        cache = {}
        try:
            s.bands_left = cache
        except AttributeError:  # a sampler without the attribute slot: no caching
            pass
    if cls in cache:
        return cache[cls]
    top = _top(s, top)
    scene, support = s.world.scene_model, top.name
    bands = None
    try:
        bands = RB.ReachBands.load(scene, support, arm="left", contact_set=CONTACT_SET, height=float(height))
        log.info("left contact band cache hit for %s (height class %.2f): %s", support, cls, bands.path)
    except TypeError:
        log.info("reach_bands loader has no left-arm mode yet: using the in-process left contact band")
        try:
            bands = RB.ReachBands.load(scene, support, left_contact_object(cls), "generic")
        except FileNotFoundError:
            bands = None
    except FileNotFoundError:
        # the B3 manifest computes the classes of its own object-height table (band metas' object_extent): a neighbouring
        # class stands in when the exact 2 cm class is missing (contact height differs by <= a few cm)
        try:
            bands = RB.ReachBands.load(scene, support, arm="left", contact_set=CONTACT_SET, height=None)
            near = bands.meta.get("height_class")
            if near is not None and abs(float(near) - cls) > 0.04 + 1e-6:
                log.info("left contact band of %s: nearest class %.2f is > 4 cm from %.2f; not used", support, float(near), cls)
                bands = None
            else:
                log.info("left contact band for %s: class %.2f stands in for %.2f (%s)", support, float(near) if near is not None else float("nan"), cls, bands.path)
        except (FileNotFoundError, TypeError):
            log.info("no B3 left contact band for %s (height class %.2f)", support, cls)
    if bands is None and compute:
        spec = BandSpec(object=left_contact_object(cls), generic_pinch=True, height=cls, arm="left", contact=True)
        bands = compute_bands(s, spec, top, edge)
    cache[cls] = bands
    return bands


def cell_classes(bands: Optional[RB.ReachBands], translate: float, tilt: float, cell_base) -> Optional[List[str]]:
    """B3's per-cell pose classes that pass at ``cell_base`` (base frame) of the trunk (``ReachBands.classes_at``:
    names in bit order of ``meta["classes"]`` from the ``<trunk>__classes`` array); None when the band records no
    classes (library-grasp bands, bands computed before A1) or the cell is not in the band."""
    if bands is None or cell_base is None:
        return None
    names = bands.meta.get("classes") if isinstance(bands.meta, dict) else None
    masks = getattr(bands, "classes", None)
    if not names or not masks:
        return None
    k = bands.nearest_trunk(float(translate), float(tilt))
    if k is None or k not in masks:
        return None
    i = bands.cell_index(k[0], k[1], np.asarray(cell_base, dtype=np.float32))
    if i is None or i >= len(masks[k]):
        return None
    m = int(masks[k][i])
    return [n for b, n in enumerate(names) if (m >> b) & 1]


def class_pitch_deg(name: str) -> Optional[float]:
    """Approach pitch of a generic class name ``p<pitch>_y<yaw>`` (None for other classes)."""
    if not name or not name.startswith("p") or "_" not in name:
        return None
    try:
        return float(int(name[1:].split("_", 1)[0]))
    except ValueError:
        return None


def approach_pitch_order(classes: Optional[Sequence[str]], pitches_deg: Sequence[float] = APPROACH_PITCHES_DEG) -> List[float]:
    """The approach pitches (deg from straight down, forward) to try for a generic push / stop pose at a cell: the
    pitches of the cell's passing classes first (shallowest first), then the remaining ones so that the 45° / 60°
    classes are tried before the placement is rejected; without a recorded mask the plain order."""
    pitches = [float(p) for p in pitches_deg]
    if not classes:
        return pitches
    passing = sorted({p for p in (class_pitch_deg(c) for c in classes) if p is not None and p in pitches})
    return passing + [p for p in pitches if p not in passing]


def contact_class(stance: Stance, approach_world, closing_world) -> Optional[str]:
    """B3's contact class name of a brace pose: ``approach_world`` = the eef approach axis (into the face / down onto
    the top), ``closing_world`` = the closing axis; base frame of ``stance`` (x ahead, lateral + = robot's left).
    Side classes ``near`` (pads on the face toward the robot, approach +x), ``far`` (−x), ``left`` (approach −lateral: the
    hand on the object's robot-left face), ``right`` (+lateral) with ``_v`` (closing vertical) / ``_h`` (horizontal);
    ``top_y0`` / ``top_y90`` for a downward approach with the closing axis lateral / ahead."""
    a = _t(approach_world)
    c = _t(closing_world)
    fx, fy = float(stance.forward[0]), float(stance.forward[1])
    lx, ly = float(stance.left[0]), float(stance.left[1])
    ax, al, az = float(a[0]) * fx + float(a[1]) * fy, float(a[0]) * lx + float(a[1]) * ly, float(a[2])
    cx, cl, cz = float(c[0]) * fx + float(c[1]) * fy, float(c[0]) * lx + float(c[1]) * ly, float(c[2])
    if az < -0.9:
        return "top_y0" if abs(cl) >= abs(cx) else "top_y90"
    if abs(az) > 0.3:  # a pitched approach belongs to no contact class
        return None
    side = ("near" if ax > 0 else "far") if abs(ax) >= abs(al) else ("left" if al < 0 else "right")
    return f"{side}_{'v' if abs(cz) > 0.7 else 'h'}"


def pitched_approach(pitch_deg: float, forward: th.Tensor) -> th.Tensor:
    """Approach direction of the generic set: straight down rotated ``pitch_deg`` toward ``forward`` (the robot's
    forward, horizontal)."""
    z = th.tensor([0.0, 0.0, 1.0])
    f = th.tensor([float(forward[0]), float(forward[1]), 0.0])
    f = f / max(float(th.linalg.norm(f)), 1e-9)
    a = math.radians(float(pitch_deg))
    return C.unit(-z * math.cos(a) + f * math.sin(a))


def in_band(bands: Optional[RB.ReachBands], stance: Stance, xy, tol: float = CELL_TOLERANCE) -> Tuple[bool, Optional[np.ndarray]]:
    """Whether the world point ``xy`` lies within ``tol``·lattice of a cell of ``bands`` at the stance's trunk (same
    translate, nearest tilt); returns ``(ok, nearest cell in base coordinates)``."""
    if bands is None:
        return False, None
    cells = bands.band(float(stance.translate), float(stance.tilt))
    if len(cells) == 0:
        return False, None
    base = bands.to_base(np.asarray([[float(xy[0]), float(xy[1])]], dtype=np.float32), stance.base_pose2d)[0]
    d = np.linalg.norm(np.asarray(cells, dtype=np.float32) - base, axis=1)
    j = int(d.argmin())
    lat = float(bands.meta.get("lattice", RB.LATTICE)) if isinstance(bands.meta, dict) else RB.LATTICE
    return bool(d[j] <= tol * lat + 1e-6), np.asarray(cells[j], dtype=np.float32)


def joint_placement(bands_right: RB.ReachBands, bands_left: Optional[RB.ReachBands], stance: Stance, work_xy, brace_xy, tol: float = CELL_TOLERANCE,
                    brace_class: Optional[str] = None) -> Dict[str, Any]:
    """A1 placement test for a candidate (target pose → work point of the right arm, brace point of the left arm):
    ``work_xy`` ∈ a right band cell AND ``brace_xy`` ∈ a left contact band cell **at the same trunk** (the stance's);
    with ``brace_class`` (B3 contact class of the intended brace pose, ``contact_class``) and a band that records the
    per-cell classes, the class must have passed at that cell. Without a left band (None) the brace side is left to
    the exact two-arm IK behind the reach proxy ``BRACE_PROXY_X`` / ``BRACE_PROXY_LATERAL`` and ``left_band`` is False
    so the caller can record it. The dict carries ``ok`` /
    ``work_ok`` / ``brace_ok`` / ``brace_class_ok``, the nearest cells (base frame), the work cell's ``classes`` and the
    brace cell's ``brace_classes`` (B3 bitmasks, None when absent)."""
    work_ok, work_cell = in_band(bands_right, stance, work_xy, tol)
    brace_classes = None
    class_ok = True
    if bands_left is None:
        # no left band yet: a reach proxy keeps the exact IK from being spent on hopeless brace points (the arms' bands
        # span x 0.48–0.80 m ahead and |λ| <= 0.56 on the pilot supports, README of the B3 run)
        bx, bl = (float(v) for v in stance.world_to_base(_t(brace_xy)[:2]).tolist())
        brace_ok, brace_cell = (BRACE_PROXY_X[0] <= bx <= BRACE_PROXY_X[1] and abs(bl) <= BRACE_PROXY_LATERAL), None
    else:
        brace_ok, brace_cell = in_band(bands_left, stance, brace_xy, tol)
        if brace_ok:
            brace_classes = cell_classes(bands_left, stance.translate, stance.tilt, brace_cell)
            if brace_class is not None and brace_classes is not None:
                class_ok = brace_class in brace_classes
    return {"ok": bool(work_ok and brace_ok and class_ok), "work_ok": bool(work_ok), "brace_ok": bool(brace_ok), "brace_class_ok": bool(class_ok),
            "work_cell": work_cell, "brace_cell": brace_cell, "left_band": bands_left is not None,
            "classes": cell_classes(bands_right, stance.translate, stance.tilt, work_cell), "brace_classes": brace_classes, "brace_class": brace_class}


def joint_two_arm_placement(s, stance: Stance, bands_right: RB.ReachBands, bands_left: Optional[RB.ReachBands], work_xy, brace_xy, work_arm: str,
                            work_poses: Sequence[Pose], brace_arm: str, brace_poses: Sequence[Pose], *, ignore_work_last: Sequence[Any] = (),
                            ignore_brace_last: Sequence[Any] = (), tol: float = CELL_TOLERANCE, brace_class: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """A1 in one call: the band test (``joint_placement``) first, then the exact two-arm IK of §4.6 step 6
    (``brace_second_arm``: work arm poses, brace arm [pre-contact, contact] with the work arm locked, combined whole-body
    check). Returns ``(ok, info)`` with ``info["reason"]`` = ``"work_off_band"`` / ``"brace_off_band"`` /
    ``"brace_class_failed"`` / the IK reason."""
    info = joint_placement(bands_right, bands_left, stance, work_xy, brace_xy, tol, brace_class=brace_class)
    if not info["work_ok"]:
        info["reason"] = "work_off_band"
        return False, info
    if not info["brace_ok"]:
        info["reason"] = "brace_off_band"
        return False, info
    if not info["brace_class_ok"]:
        info["reason"] = "brace_class_failed"
        return False, info
    res = brace_second_arm(s, stance, work_arm, work_poses, brace_arm, brace_poses, ignore_work_last=ignore_work_last, ignore_brace_last=ignore_brace_last)
    info["ik"] = res
    info["reason"] = None if res.ok else f"ik: {res.reason}"
    return bool(res.ok), info


def union_extent(*bands: Optional[RB.ReachBands]) -> Optional[Tuple[float, float]]:
    """§4.4 (A1): the lateral extent [λ_min, λ_max] of the union of the given bands (the right grasp band and the left
    contact band), None when none has cells."""
    lo, hi = math.inf, -math.inf
    for b in bands:
        if b is None:
            continue
        ext = b.union_extent() if hasattr(b, "union_extent") else None
        if ext is None:
            continue
        lo, hi = min(lo, float(ext[0])), max(hi, float(ext[1]))
    return None if lo > hi else (lo, hi)


# ----------------------------------------------------------------------------------------------------------------
# camera helpers (§3.4)
# ----------------------------------------------------------------------------------------------------------------


def image_border_lateral(s, cam_pose: Pose, stance: Stance, depth: float, z: float, margin: float = IN_FRAME_MARGIN) -> Tuple[float, float]:
    """``(b_right, b_left)``: lateral distance (m, along the edge tangent) from the robot's centre line to the point where
    the image's right / left border (with the ``margin`` fraction) meets the plane ``z`` at ``depth`` from the front edge
    (§4.3 b: "lateral offset of the image border at the source-tray depth"; §4.6 "border at tray depth")."""
    cm = s.world.camera_model
    p = stance.world(float(depth), 0.0)
    pt = th.tensor([[float(p[0]), float(p[1]), float(z)]])
    uv, dep = cm.project(cam_pose, pt)
    v = float(uv[0, 1])
    out = []
    for u in (cm.width - margin * cm.width, margin * cm.width):  # image right = robot's right (facing forward), image left = robot's left
        hit = cm.ray_plane_intersection(cam_pose, u, v, float(z))
        if hit is None:
            out.append(float("inf"))
            continue
        _, lat = stance.local(hit[:2])
        out.append(abs(lat))
    return out[0], out[1]


def r_img(s, cam_pose: Pose, stance: Stance, target_xy, z: float, side: float, margin: float = IN_FRAME_MARGIN) -> float:
    """§4.3 R_img: surface distance from the target to the image border on ``side`` (+1 = robot's left, −1 = right)."""
    depth, lat = stance.local(target_xy)
    b_r, b_l = image_border_lateral(s, cam_pose, stance, depth, z, margin)
    return max(0.0, (b_l - lat) if side > 0 else (b_r + lat))


def in_frame(s, cam_pose: Pose, points: th.Tensor, margin: float = IN_FRAME_MARGIN) -> bool:
    return bool(s.world.camera_model.all_inside(cam_pose, points, margin))


def all_in_frame(s, cam_pose: Pose, items: Sequence[Tuple[str, th.Tensor, Pose]], margin: float = IN_FRAME_MARGIN, skip: Sequence[str] = ()) -> List[str]:
    """§4.6 step 4: names of the ``(name, hull, pose)`` items whose AABB corners are not fully inside the image (``margin``);
    ``skip`` names are exempt (the disembedding destination tray)."""
    out = []
    for name, hull, pose in items:
        if name in skip:
            continue
        if not in_frame(s, cam_pose, hull_corners_world(hull, pose), margin):
            out.append(name)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Phase 1a — edge and stand-off (§4.2)
# ----------------------------------------------------------------------------------------------------------------


def edge_frames(s, top, rng=None, stand_off: float = STAND_OFF) -> List[Tuple[EdgeFrame, Stance]]:
    """Every edge of the top (random order with ``rng``) with its stance at ``stand_off`` + overhang (>= 5 cm), trunk
    (0.5, 0) — before the floor / collision checks of Phase 1a."""
    support = s.world.obj(top.name)
    lo, hi = support.aabb
    aabb_xy = th.tensor([[float(lo[0] if sx == 0 else hi[0]), float(lo[1] if sy == 0 else hi[1])] for sx in (0, 1) for sy in (0, 1)])
    edges = list(top.edges())
    order = [int(i) for i in rng.permutation(len(edges))] if rng is not None else list(range(len(edges)))
    out = []
    for i in order:
        name, mid, normal, length = edges[i]
        n = _t(normal)[:2]
        overhang = max(0.0, float((aabb_xy @ n).max()) - float(th.dot(_t(mid)[:2], n)))
        edge = EdgeFrame.from_edge(name, mid, normal, length, overhang if overhang >= OVERHANG_MIN else 0.0)
        xy = edge.base_xy(stand_off, 0.0)
        st = Stance("init", (float(xy[0]), float(xy[1]), edge.yaw), 0.5, 0.0, float(stand_off), edge, lateral=0.0)
        out.append((edge, st))
    return out


def edge_candidates(s, top, rng=None, stand_off: float = STAND_OFF) -> Iterator[Tuple[EdgeFrame, Stance]]:
    """§4.2 survivors in random order: the edge stance is on free floor and whole-body collision-free at trunk (0.5, 0)
    with untucked arms. Task modules that retry edges (axial ``edge_not_visible``) iterate this."""
    skipped: Dict[str, str] = {}
    s._tabletop_edge_log = skipped  # 1a outcomes per edge (reported with the rejections)
    for edge, st in edge_frames(s, top, rng, stand_off):
        if not s.kit.trav_free(st.base_xy):
            skipped[edge.name] = f"no floor at {[round(float(v), 2) for v in st.base_xy.tolist()]}"
            log.info("phase 1a: edge %s (L %.2f, overhang %.2f) skipped: %s", edge.name, edge.length, edge.overhang, skipped[edge.name])
            continue
        if not s.ikc.stance_collision_free(st):
            skipped[edge.name] = f"stance at {[round(float(v), 2) for v in st.base_xy.tolist()]} collides at trunk (0.5, 0)"
            log.info("phase 1a: edge %s (L %.2f, overhang %.2f) skipped: %s", edge.name, edge.length, edge.overhang, skipped[edge.name])
            continue
        log.info("phase 1a: edge %s (L %.2f, overhang %.2f) stance at %s", edge.name, edge.length, edge.overhang, [round(float(v), 2) for v in st.base_xy.tolist()])
        s._tabletop_top = top
        yield edge, st


def phase1_edge_and_standoff(s, top, rng=None, stand_off: float = STAND_OFF) -> Tuple[EdgeFrame, Stance]:
    """§4.2: any edge of the top polygon (no preference), yaw = inward normal, s = 0.45 m + the AABB overhang when >= 5 cm;
    ``Stance("init", translate 0.5, tilt 0)`` at the edge midpoint. ``Reject("no_free_edge")`` when no edge stance is on
    free floor and collision-free."""
    for edge, st in edge_candidates(s, top, rng, stand_off):
        return edge, st
    _reject("no_free_edge", f"{top.name}: no edge has a free, collision-free stance at s = {stand_off} (+ overhang): {getattr(s, '_tabletop_edge_log', {})}")


# ----------------------------------------------------------------------------------------------------------------
# Phase 1b — init trunk height (§4.3)
# ----------------------------------------------------------------------------------------------------------------


def phase1b_init_trunk(s, init: Stance, H_max: float, *, bands: Optional[RB.ReachBands] = None, top=None, margin: float = IN_FRAME_MARGIN,
                       left_bands: Optional[RB.ReachBands] = None) -> Stance:
    """§4.3: the init trunk (translate t*, tilt 0) from the camera FK: Δ = clamp(Δ_aim, Δ_min, Δ_max), t* = z_c⁻¹(h + Δ)
    clamped to [0, 0.95]; records ``b`` (image border lateral at the workspace centre depth), ``R_img`` (centre line),
    the workspace depths, the band union extent [λ_min, λ_max] and the translate range of [Δ_min, Δ_max] in
    ``stance.extras`` for Phase 1c. With ``left_bands`` (A1: the left contact band of a two-arm task) the lateral extent
    is the union of both bands' workspaces on this top (§4.4 margins use the union; the depths stay the work arm's)."""
    top = _top(s, top)
    bands = bands or ensure_bands(s, top=top, edge=init.edge)
    fk = camera_fk(s)
    ws = workspace(s, bands, init, top)
    if len(ws["cells_world"]) == 0:
        _reject("no_workspace", f"{top.name} edge {init.edge.name}: no band cell of {bands.path or 'the band'} falls on the top at s = {init.stand_off:.2f}")
    if left_bands is not None:
        ws_l = workspace(s, left_bands, init, top)
        if len(ws_l["cells_world"]):
            ws["lam_min_right"], ws["lam_max_right"] = ws["lam_min"], ws["lam_max"]
            ws["lam_min_left"], ws["lam_max_left"] = ws_l["lam_min"], ws_l["lam_max"]
            ws["lam_min"], ws["lam_max"] = min(ws["lam_min"], ws_l["lam_min"]), max(ws["lam_max"], ws_l["lam_max"])
        else:
            log.info("phase 1b: the left contact band has no cell on %s at edge %s; λ from the right band alone", top.name, init.edge.name)
    h = float(top.z) - float(s.kit.floor_z)
    phi_m = fk.phi_m(margin)
    t = float(init.translate)
    res = None
    for _ in range(3):  # x_c depends weakly on t: two fixed-point passes settle it
        res = init_trunk_delta(init.edge_depth, ws["d_ws"], ws["d_near"], fk.x_at(t), H_max, fk.hand_below, fk.pitch, phi_m)
        t = fk.t_of_z(h + res["delta"])
    st = init.with_trunk(t, 0.0, "init")
    cam = s.camera_pose_at(st)
    b_r, b_l = image_border_lateral(s, cam, st, ws["d_ws"], top.z, margin)
    b = min(b_r, b_l)
    t_lo = fk.t_of_z(h + res["delta_max"])  # deepest camera height allowed -> smallest t? (z decreases with t)
    t_hi = fk.t_of_z(h + res["delta_min"])
    st.extras.update({"H_max": float(H_max), **res, "d_ws": ws["d_ws"], "d_near": ws["d_near"], "d_far": ws["d_far"], "lam_min": ws["lam_min"],
                      "lam_max": ws["lam_max"], "x_c": fk.x_at(t), "z_c": fk.z_at(t), "b": b, "b_right": b_r, "b_left": b_l, "R_img": b,
                      "t_range": [min(t_lo, t_hi), max(t_lo, t_hi)], "camera_pitch_deg": math.degrees(fk.pitch), "hand_below": fk.hand_below,
                      "n_workspace_cells": int(len(ws["cells_world"])), "left_band": left_bands is not None,
                      **{k: ws[k] for k in ("lam_min_right", "lam_max_right", "lam_min_left", "lam_max_left") if k in ws}})
    log.info("phase 1b: edge %s h=%.2f d_ws=%.2f d_near=%.2f Δ_aim=%.2f Δ_min=%.2f Δ_max=%.2f -> Δ=%.2f t*=%.2f b=%.2f λ=[%.2f, %.2f]", init.edge.name, h,
             ws["d_ws"], ws["d_near"], res["delta_aim"], res["delta_min"], res["delta_max"], res["delta"], t, b, ws["lam_min"], ws["lam_max"])
    return st


# ----------------------------------------------------------------------------------------------------------------
# Phase 1c — lateral position (§4.4)
# ----------------------------------------------------------------------------------------------------------------


def trunk_sweep_free(s, stance: Stance, translates: Sequence[float] = TRUNK_SWEEP_TRANSLATES, margin: float = 0.01) -> bool:
    """§4.4 check (3): base + torso links + head collision-free at every grid translate, tilt 0, arms excluded. cuRobo
    checks whole-robot spheres, so the arms are hung (tucked, inside the base column) and the self-collision term is
    off — the closest available reading of "arms excluded"."""
    s.kit._ensure_synced()
    qs = th.stack([s.kit.q_at_base_pose(s.kit.q_config(arms="tucked", translate=float(tr), tilt=0.0), *stance.base_pose2d) for tr in translates])
    coll = s.kit.backend.check_collision(qs, emb_name="whole_body", sync_world=False, self_collision=False, margin=margin)
    return not bool(coll.any())


def phase1c_lateral(s, init: Stance, margins: Dict[str, Any], rng, *, top=None, bands: Optional[RB.ReachBands] = None, n_draws: int = 20,
                    in_frame_margin: float = IN_FRAME_MARGIN) -> Stance:
    """§4.4: margins of the task's row from the band union extent, ``b`` and ``R_img`` of Phase 1b; lateral position
    uniform over [−L/2 + m_R, L/2 − m_L] (``Reject("edge_too_short")`` when m_R + m_L > L); valid when (1) the base is on
    free floor, (2) the init configuration (t*, 0, untucked, open) is whole-body collision-free — else t moves within
    [Δ_min, Δ_max] (b, R_img, margins recomputed) or the lateral is redrawn —, (3) the trunk sweep of translates
    0 … 0.95 at tilt 0 is collision-free with the arms excluded. The result is the instance's **init pose**."""
    top = _top(s, top)
    if "lam_min" not in init.extras:
        init = phase1b_init_trunk(s, init, float(margins.get("H_max", init.extras.get("H_max", 0.0))), bands=bands, top=top)
    fk = camera_fk(s)
    h = float(top.z) - float(s.kit.floor_z)
    lam_min, lam_max = float(init.extras["lam_min"]), float(init.extras["lam_max"])
    L = float(init.edge.length)
    errors: Dict[str, int] = {"no_floor": 0, "collision": 0, "trunk_sweep": 0, "out_of_bounds_after_t": 0}
    t_lo, t_hi = init.extras.get("t_range", [init.translate, init.translate])

    def bounds_for(st: Stance):
        cam = s.camera_pose_at(st)
        b_r, b_l = image_border_lateral(s, cam, st, float(init.extras["d_ws"]), top.z, in_frame_margin)
        b = min(b_r, b_l)
        m_R, m_L, info = lateral_margins(margins, lam_min, lam_max, b, b)
        if info.get("reject"):
            _reject(info["reject"][0], info["reject"][1])
        return lateral_bounds(L, m_R, m_L), info, b

    bounds, info, b = bounds_for(init)
    if bounds is None:
        _reject("edge_too_short", f"edge {init.edge.name}: m_R + m_L = {info['m_R'] + info['m_L']:.2f} > L = {L:.2f} ({info})")
    for _ in range(int(n_draws)):
        lat = float(rng.uniform(bounds[0], bounds[1]))
        st = init.shifted(lateral=lat, name="init")
        if not s.kit.trav_free(st.base_xy):
            errors["no_floor"] += 1
            continue
        if not s.ikc.stance_collision_free(st):
            # move t within [Δ_min, Δ_max]: the other translates of the admissible range, nearest first
            fixed = None
            for t in sorted(np.linspace(float(t_lo), float(t_hi), 5).tolist(), key=lambda v: abs(v - st.translate)):
                if abs(t - st.translate) < 1e-6:
                    continue
                st2 = st.with_trunk(clamp(t, 0.0, MAX_TRANSLATE), 0.0, "init")
                if s.ikc.stance_collision_free(st2):
                    b2, info2, bb = bounds_for(st2)
                    if b2 is not None and b2[0] <= lat <= b2[1]:
                        st2.extras.update({"b": bb, "R_img": bb, "z_c": fk.z_at(st2.translate), "delta": fk.z_at(st2.translate) - h})
                        fixed, info, bounds = st2, info2, b2
                        break
                    errors["out_of_bounds_after_t"] += 1
            if fixed is None:
                errors["collision"] += 1
                continue
            st = fixed
        if not trunk_sweep_free(s, st):
            errors["trunk_sweep"] += 1
            continue
        st.extras.update({"margins": info, "lateral_bounds": list(bounds)})
        log.info("phase 1c: edge %s lateral %.2f in [%.2f, %.2f] (m_R %.2f, m_L %.2f), t* %.2f", init.edge.name, lat, bounds[0], bounds[1], info["m_R"],
                 info["m_L"], st.translate)
        return st
    _reject("stance_collision", f"edge {init.edge.name}: no valid lateral position in [{bounds[0]:.2f}, {bounds[1]:.2f}] after {n_draws} draws ({errors})")


# ----------------------------------------------------------------------------------------------------------------
# Phase 2 — manipulation stance (§4.5)
# ----------------------------------------------------------------------------------------------------------------


def tilt_scan(fk: CameraFK, translate: float, n: int = N_TILT_SAMPLES) -> List[float]:
    """§4.5 "tilt continuous from 0 to the torso_joint3 limit": ``n`` samples from 0 down to the limit at ``translate``."""
    lim = fk.tilt_limit(translate)
    return [float(v) for v in np.linspace(0.0, lim, int(n)).tolist()]


def default_in_frame(s, stance: Stance, cam: Pose, focus: th.Tensor, info: Any, half_extent: float, height: float, margin: float) -> bool:
    """§4.5 step 3 default: the target AABB (a box of ``half_extent`` around the focus from the top to ``height``) in frame."""
    z0 = float(focus[2]) - 0.5 * height
    return in_frame(s, cam, box_corners(focus[:2], half_extent, z0, z0 + height), margin)


def phase2_manipulation_stance(s, init: Stance, target_half_extent: float, *, target_height: Optional[float] = None,
                               in_frame_fn: Optional[Callable[..., bool]] = None, focus_candidates: Optional[Callable[..., List[Tuple[th.Tensor, Any]]]] = None,
                               stand_offs: Sequence[float] = STAND_OFFS, translates: Sequence[float] = TRANSLATES, n_tilts: int = N_TILT_SAMPLES,
                               angle_classes_deg: Sequence[float] = ANGLE_CLASSES_DEG, margin: float = IN_FRAME_MARGIN, bands: Optional[RB.ReachBands] = None,
                               top=None, erosion_extra: float = 0.02, focus_z: Optional[float] = None) -> List[Stance]:
    """§4.5: scan stand-off {0.35, 0.45, 0.55} x translate {0, …, 0.95} x tilt (0 → torso_joint3 limit, ``n_tilts`` samples)
    at the init edge and lateral; (1) whole-body collision check without objects; (2) band = the grid trunk with the
    same translate and nearest tilt, focus candidates = its cells inside the top eroded by ``target_half_extent`` +
    0.02 (shifted with the stand-off); (3) keep (stance, focus) pairs whose optical axis is within 8° of the focus (else
    20°) and whose target box is in frame (5 %; ``in_frame_fn(s, stance, cam, focus, info)`` overrides: the disembedding
    source tray, the block region); ``focus_candidates(s, stance, cam, cells_world)`` overrides the focus cells (block
    regions: ``[(focus_xy, info), ...]``). Returns the survivors of the **best angle class**, each a ``Stance
    ("manipulation")`` copy with ``focus`` (x, y, z) set and ``extras`` ``axis_angle`` / ``angle_class`` / ``focus_info``;
    the caller draws uniformly (§4.5 step 4) and checks the init → manipulation transition (step 5)."""
    top = _top(s, top)
    bands = bands or ensure_bands(s, top=top, edge=init.edge)
    fk = camera_fk(s)
    height = float(target_height) if target_height is not None else 2.0 * float(target_half_extent)
    cands: List[Stance] = []
    for so in stand_offs:
        base = init.shifted(stand_off=float(so), name="manipulation")
        if not s.kit.trav_free(base.base_xy):
            continue
        for tr in translates:
            for ti in tilt_scan(fk, float(tr), n_tilts):
                cands.append(base.with_trunk(float(tr), float(ti), "manipulation"))
    if not cands:
        return []
    free = s.ikc.stances_collision_free(cands)
    cands = [c for c, f in zip(cands, free.tolist()) if f]
    if not cands:
        log.info("phase 2: every trunk candidate collides")
        return []
    cams = s.kit.camera_poses_batch(th.stack([s.stance_q(c) for c in cands]))
    cm = s.world.camera_model
    erosion = float(target_half_extent) + float(erosion_extra)
    classes: Dict[int, List[Stance]] = {0: [], 1: []}
    n_cells = 0
    for st, cam in zip(cands, cams):
        cells = band_cells_world(bands, st)
        if len(cells) == 0:
            continue
        keep = [c for c in cells if top.contains(c, erosion)]
        if not keep:
            continue
        cw = th.stack(keep)
        foci = focus_candidates(s, st, cam, cw) if focus_candidates is not None else [(c, None) for c in cw]
        for focus_xy, info in foci:
            fz = float(focus_z) if focus_z is not None else float(top.z) + 0.5 * height
            focus = th.tensor([float(focus_xy[0]), float(focus_xy[1]), fz])
            ang = cm.angle_off_axis(cam, focus)
            if ang > math.radians(angle_classes_deg[1]):
                continue
            n_cells += 1
            ok = (in_frame_fn(s, st, cam, focus, info) if in_frame_fn is not None
                  else default_in_frame(s, st, cam, focus, info, float(target_half_extent), height, margin))
            if not ok:
                continue
            c = st.with_trunk(st.translate, st.tilt, "manipulation")
            c.focus = focus
            c.extras.update({"axis_angle": ang, "angle_class": 0 if ang <= math.radians(angle_classes_deg[0]) else 1, "focus_info": info})
            classes[c.extras["angle_class"]].append(c)
    out = classes[0] or classes[1]
    log.info("phase 2: %d trunk candidates collision-free, %d (stance, focus) pairs within 20°, %d survivors in class %s", len(cands), n_cells, len(out),
             0 if classes[0] else 1)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Phase 3 step 1 — target init pose (§4.6)
# ----------------------------------------------------------------------------------------------------------------


def phase3_target(s, manip: Stance, rng, grasps: Sequence[Any], hull: th.Tensor, rest_quat, ignore_for_grasp: Sequence[Any], *, top=None,
                  support_z: Optional[float] = None, margin: float = SUPPORT_MARGIN, focus_radius: float = FOCUS_RADIUS, yaws: Sequence[float] = YAWS,
                  arm: str = "right", object_name: Optional[str] = None, scale_key: Any = None, eef_of_grasp: Optional[Callable[[Any], Pose]] = None,
                  bands: Optional[RB.ReachBands] = None, jitter: float = YAW_JITTER, n_positions: int = 5) -> Tuple[Pose, GraspRecord]:
    """§4.6 step 1: screen the 8 fixed yaws x ``grasps`` at the focus cell (12-seed IK from ``manip``, support ignored for
    the grasp pose only) -> feasible set F (stored in the band cache under (trunk, cell); empty F removes the cell);
    draw (yaw, grasp) uniformly from F, jitter ±22.5°, place the object within 5 cm of the focus with its footprint on
    the top (``margin``; a tray floor passes ``top`` = the tray raster and ``support_z`` = its floor), derive grasp =
    object pose ∘ T_obj→eef and pre-grasp 10 cm back, confirm both with the exact IK; next element of F without
    replacement on failure; ``Reject("grasp_unreachable")`` when F is exhausted. Returns the object pose and the
    ``GraspRecord`` (``yaw``, ``grasp_key``, ``grasp_name``, both eef poses, ``eef_in_object``)."""
    IK = _ik()
    top = _top(s, top)
    z = float(support_z) if support_z is not None else float(top.z)
    focus = manip.focus
    if focus is None:
        raise ValueError("phase3_target needs a manipulation stance with a focus (Phase 2 survivor)")
    fx = focus[:2]
    eef_of = eef_of_grasp or (lambda g: g.pose)
    grasps = list(grasps)

    def pose_at(cell, yaw):
        return object_pose_on(s, hull, rest_quat, cell, yaw, z)

    F = s.ikc.feasible_sets([fx], list(yaws), grasps, pose_at, manip, arm, list(ignore_for_grasp), eef_of).get(0, [])
    store_feasible_set(bands or getattr(s, "bands", None), manip, F, yaws)
    if not F:
        _reject("grasp_unreachable", f"empty feasible set F at the focus {[round(float(v), 3) for v in fx.tolist()]} (trunk {manip.translate:.2f}/{manip.tilt:+.2f}, "
                                     f"{len(yaws)} yaws x {len(grasps)} grasps)")
    name = object_name or s.bindings.one("target").name
    fails = {"support": 0, "ik": 0}
    for i in rng.permutation(len(F)):
        yaw0, g = F[int(i)]
        yaw = float(yaw0) + float(rng.uniform(-jitter, jitter))
        for _ in range(int(n_positions)):
            r, a = float(rng.uniform(0.0, focus_radius)), float(rng.uniform(0.0, 2 * math.pi))
            xy = th.tensor([float(fx[0]) + r * math.cos(a), float(fx[1]) + r * math.sin(a)])
            pose = pose_at(xy, yaw)
            if not top.polygon_supported(s.world.footprint(hull, pose), margin=margin):
                fails["support"] += 1
                continue
            eef = IK.grasp_pose(pose, eef_of(g))
            pre = IK.pre_pose(eef)
            res = s.ikc.exact([pre, eef], manip, arm, ignore_for_last=list(ignore_for_grasp), scale_key=scale_key)
            if not res.ok:
                fails["ik"] += 1
                continue
            finger_q = list(getattr(g, "finger_q", []) or []) or None
            rec = GraspRecord(name, arm, getattr(g, "key", None), getattr(g, "name", None), yaw, pre, eef, finger_q=finger_q, eef_in_object=eef_of(g))
            rec.stance = manip.name
            return pose, rec
    _reject("grasp_unreachable", f"F ({len(F)} pairs) exhausted without an exact-IK confirmation ({fails})")


# ----------------------------------------------------------------------------------------------------------------
# §4.6 steps 2–6 — partner, distractors, frame, whole-body checks, brace
# ----------------------------------------------------------------------------------------------------------------


def place_partner(s, rng, t_pose: Pose, t_hull: th.Tensor, p_hull: th.Tensor, p_rest, stance: Stance, side: Optional[float], gap_range: Sequence[float], *,
                  top=None, cam: Optional[Pose] = None, z: Optional[float] = None, margin: float = SUPPORT_MARGIN, min_gap: float = MIN_GAP,
                  in_frame_margin: float = IN_FRAME_MARGIN, yaw: Optional[float] = None, n_tries: int = 40) -> Pose:
    """§4.6 step 2: the partner on ``side`` of the target (+1 = robot's left, −1 = right, None = random per try) along the
    edge at an edge-to-edge gap g ~ U(``gap_range``), same depth as the target, random yaw (or ``yaw``), footprint on the
    top (``margin``), >= ``min_gap`` from the target, and — with ``cam`` — fully in frame. ``Reject("partner_placement")``."""
    top = _top(s, top)
    z = float(top.z) if z is None else float(z)
    t_fp = s.world.footprint(t_hull, t_pose)
    t_lat = (t_fp - stance.base_xy) @ stance.tangent
    depth_t, _ = stance.local(t_pose[0])
    g_lo, g_hi = float(gap_range[0]), float(max(gap_range[0], gap_range[1]))
    for _ in range(int(n_tries)):
        sd = float(side) if side is not None else (1.0 if rng.random() < 0.5 else -1.0)
        g = float(rng.uniform(g_lo, g_hi))
        yw = float(rng.uniform(-math.pi, math.pi)) if yaw is None else float(yaw)
        p_quat = C.qmul(C.quat_from_yaw(yw), _t(p_rest))
        p_pts = C.transform_points((th.zeros(3), p_quat), p_hull)[:, :2]
        p_lat = p_pts @ stance.tangent
        lat = (float(t_lat.max()) + g - float(p_lat.min())) if sd > 0 else (float(t_lat.min()) - g - float(p_lat.max()))
        xy = stance.world(depth_t, lat)
        pos = th.tensor([float(xy[0]), float(xy[1]), z + s.world.bottom_offset(p_hull, p_quat) + Z_OFFSET])
        p_pose = (pos, p_quat)
        p_fp = s.world.footprint(p_hull, p_pose)
        if not top.polygon_supported(p_fp, margin=margin):
            continue
        if _polygons_overlap(t_fp, p_fp, min_gap):
            continue
        if cam is not None and not in_frame(s, cam, hull_corners_world(p_hull, p_pose), in_frame_margin):
            continue
        return p_pose
    _reject("partner_placement", f"no partner position with gap in [{g_lo:.2f}, {g_hi:.2f}] on the top and in frame after {n_tries} tries")


def polygon_gap(a: th.Tensor, b: th.Tensor) -> float:
    """Exact clearance between two convex polygons (N, 2) / (M, 2): 0 when they intersect, else the smallest
    vertex-to-edge distance (pure torch; the edge-normal SAT alone under-estimates the gap of small rotated bodies)."""
    from ispatialgym.sampling import geometry as G

    if G.polygons_overlap(a, b, margin=0.0):
        return 0.0

    def seg_dist(p: th.Tensor, poly: th.Tensor) -> float:
        n = poly.shape[0]
        best = float("inf")
        for i in range(n):
            s0, s1 = poly[i], poly[(i + 1) % n]
            e = s1 - s0
            l2 = float(th.dot(e, e))
            t = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, float(th.dot(p - s0, e)) / l2))
            best = min(best, float(th.linalg.norm(p - (s0 + t * e))))
        return best

    return min(min(seg_dist(p, b) for p in a), min(seg_dist(p, a) for p in b))


def _polygons_overlap(a: th.Tensor, b: th.Tensor, margin: float = 0.0) -> bool:
    """Whether two convex footprints are closer than ``margin`` (exact clearance, see :func:`polygon_gap`)."""
    return polygon_gap(a, b) < float(margin)


def place_distractors(s, rng, raster, z: float, target_pose: Pose, target_hull: th.Tensor, items: Sequence[Tuple[str, th.Tensor, Any]], *,
                      spacing_axis: Optional[th.Tensor] = None, spacing: float = 0.0, margin: float = TRAY_MARGIN, gap: float = 0.01,
                      cam: Optional[Pose] = None, in_frame_margin: float = IN_FRAME_MARGIN, n_tries: int = 80, budget: Optional[float] = None,
                      grow: Optional[Dict[str, float]] = None) -> Dict[str, Pose]:
    """§4.6 step 3 / §6.4: each ``(name, hull, rest_quat)`` item on the ``raster`` (``cells()`` world xy, ``polygon_supported
    (fp, margin)``, ``area()``) at a random cell and yaw, footprint supported (``margin``), >= ``gap`` from every placed
    footprint and >= ``spacing`` from the target along ``spacing_axis`` (the drawn grasp's pinch axis, world xy unit):
    the projections of the two footprints on the axis stay ``spacing`` apart. With ``budget`` the footprints grown by
    ``grow[name]`` may cover at most that fraction of the raster area (``tray_budget``). ``Reject("distractor_placement")``."""
    placed: Dict[str, Pose] = {}
    fps: Dict[str, th.Tensor] = {"__target__": s.world.footprint(target_hull, target_pose)}
    t_fp = fps["__target__"]
    cells = raster.cells()
    if len(cells) == 0:
        _reject("distractor_placement", "raster has no cells")
    used_area = 0.0
    for name, hull, rest in items:
        ok = None
        for i in rng.permutation(len(cells))[: int(n_tries)]:
            xy = cells[int(i)]
            yw = float(rng.uniform(-math.pi, math.pi))
            pose = object_pose_on(s, hull, rest, xy, yw, z)
            fp = s.world.footprint(hull, pose)
            if not raster.polygon_supported(fp, margin=margin):
                continue
            if any(_polygons_overlap(fp, other, gap) for other in fps.values()):
                continue
            if spacing_axis is not None and spacing > 0:
                ax = _t(spacing_axis)[:2]
                ax = ax / max(float(th.linalg.norm(ax)), 1e-9)
                pa, pb = t_fp @ ax, fp @ ax
                if not (float(pb.min()) - float(pa.max()) >= spacing or float(pa.min()) - float(pb.max()) >= spacing):
                    continue
            if cam is not None and not in_frame(s, cam, hull_corners_world(hull, pose), in_frame_margin):
                continue
            ok = (pose, fp)
            break
        if ok is None:
            _reject("distractor_placement", f"no free position for {name} (gap {gap}, spacing {spacing:.3f} along the pinch axis)")
        placed[name], fps[name] = ok
        if budget is not None and hasattr(raster, "area"):
            g = float((grow or {}).get(name, 0.0))
            ext = ok[1].max(dim=0).values - ok[1].min(dim=0).values
            used_area += (float(ext[0]) + 2 * g) * (float(ext[1]) + 2 * g)
            if used_area > float(budget) * float(raster.area()):
                _reject("distractor_placement", f"distractor footprints (grown by pad_clearance) cover {used_area / max(raster.area(), 1e-9):.2f} > {budget} of the floor")
    return placed


def whole_body_check_at(s, stances: Sequence[Stance], objects: Dict[str, Pose], *, margin: float = 0.01, require: bool = False, reason: str = "stance_collision") -> List[bool]:
    """§4.6 step 5: teleport the drafted ``objects`` (bddl instance -> pose) into the world, sync the collision world and
    whole-body check every stance (untucked arms, its trunk). Returns the per-stance mask; with ``require`` a failing
    stance raises ``Reject(reason)``. The objects stay placed (``apply_draft`` sets the same poses again)."""
    for name, pose in objects.items():
        obj = s.objects[name]
        obj.set_position_orientation(_t(pose[0]), _t(pose[1]))
        obj.keep_still()
    s.kit.sync_world()
    free = s.ikc.stances_collision_free(list(stances), margin=margin).tolist()
    if require:
        bad = [st.name for st, f in zip(stances, free) if not f]
        if bad:
            _reject(reason, f"stances {bad} collide with the objects placed")
    return [bool(f) for f in free]


def brace_second_arm(s, stance: Stance, work_arm: str, work_poses: Sequence[Pose], brace_arm: str, brace_poses: Sequence[Pose], *,
                     ignore_work_last: Sequence[Any] = (), ignore_brace_last: Sequence[Any] = ()):
    """§4.6 step 6: the working arm's poses first (exact IK), then the other arm's [pre-contact, contact] with the working
    arm locked at its solution, combined whole-body check (``IKChecker.exact_both_arms``). Returns the ``IKResult``."""
    return s.ikc.exact_both_arms((work_arm, list(work_poses)), (brace_arm, list(brace_poses)), stance, list(ignore_work_last), list(ignore_brace_last))


@contextmanager
def posed_in_hand(s, arm: str, obj, eef_in_object: Pose):
    """Put ``obj`` where it sits in the robot's ``arm`` per the grasp transform ``T_obj→eef`` for the duration of the
    block, so that ``kit.backend.attach_object`` (which reads the *current* object-in-eef offset) attaches it as carried;
    the object's pose is restored and the collision world re-synced afterwards. Wrap ``s.record_path`` /
    ``s.record_transition`` / place IK checks with a carried object in it."""
    p0, q0 = obj.get_position_orientation()
    p0, q0 = p0.clone(), q0.clone()
    eef = s.kit.eef_pose(s.kit.q_current(), arm)
    pos, quat = C.compose((_t(eef[0]), _t(eef[1])), C.invert((_t(eef_in_object[0]), _t(eef_in_object[1]))))
    obj.set_position_orientation(pos, quat)
    obj.keep_still()
    try:
        yield
    finally:
        obj.set_position_orientation(p0, q0)
        obj.keep_still()
        s.kit.sync_world()


def facade_v1_spec(s):
    """v1 facade only (``GroundedProblem.from_v1_spec``): the v1 ``ProblemSpec`` the entry was built from (its
    ``distractors`` pools, role options), read from the default v1 customs list; None outside the facade."""
    name = (s.gp.entry.get("_v1") or {}).get("name")
    if not name:
        return None
    from ispatialgym.sampling.spec import load_task_spec
    from ispatialgym.sampling.v2 import problem as P

    cands = [p for p in ([s.gp.customs_path] if s.gp.customs_path else []) + list(P._CUSTOMS_CANDIDATES) if p and p.endswith("task_custom_lists.json") and os.path.exists(p)]
    for path in cands:
        try:
            return load_task_spec(path, check_ready=False)[name]
        except Exception:  # noqa: BLE001 - another list, or the entry is not a v1 one
            continue
    return None


# ----------------------------------------------------------------------------------------------------------------
# §4.7 — placement stance
# ----------------------------------------------------------------------------------------------------------------


def placement_stance(s, manip: Stance, dest_xy, *, name: str = "placement", check_collision: bool = True) -> Stance:
    """§4.7: the manipulation stance shifted along the edge so that the destination centre is at the focus (same stand-off,
    translate, tilt); the base must be on free floor and — ``check_collision`` — whole-body collision-free (the caller
    repeats the check with the trays and distractors placed, ``whole_body_check_at``). ``Reject("no_placement_stance")``."""
    if manip.focus is None or manip.edge is None:
        raise ValueError("placement_stance needs a manipulation stance with a focus and an edge frame")
    dlat = float(th.dot(_t(dest_xy)[:2] - manip.focus[:2], manip.tangent))
    st = manip.shifted(lateral=manip.lateral + dlat, name=name)
    st.focus = th.tensor([float(dest_xy[0]), float(dest_xy[1]), float(manip.focus[2])])
    st.arms = list(manip.arms) if isinstance(manip.arms, list) else manip.arms
    if not s.kit.trav_free(st.base_xy):
        _reject("no_placement_stance", f"no floor at the placement stance {st.base_xy.tolist()} (lateral shift {dlat:+.2f})")
    if check_collision and not s.ikc.stance_collision_free(st):
        _reject("no_placement_stance", f"placement stance collides (lateral shift {dlat:+.2f})")
    st.extras["lateral_shift"] = dlat
    return st


__all__ = [
    "STAND_OFF", "STAND_OFFS", "TRANSLATES", "TILT_GRID", "YAWS", "FOCUS_RADIUS", "YAW_JITTER", "SUPPORT_MARGIN", "TRAY_MARGIN", "MIN_GAP",
    "clamp", "init_trunk_delta", "lateral_margins", "lateral_bounds", "CameraFK", "camera_table", "camera_fk",
    "opposite_face", "face_normal_world", "up_face", "cube_tilt_pair", "cube_flip_plan", "facing_coarse_yaw", "face_points",
    "distractor_spacing", "grasp_pinch_width", "exclude_lookalikes", "axial_init_yaw",
    "object_pose_on", "hull_extents", "half_extent", "height_of", "hull_corners_world", "box_corners", "library_grasps",
    "BandSpec", "ensure_bands", "compute_bands", "band_cells_world", "band_union_cells", "workspace", "store_feasible_set",
    "image_border_lateral", "r_img", "in_frame", "all_in_frame",
    "edge_frames", "edge_candidates", "phase1_edge_and_standoff", "phase1b_init_trunk", "trunk_sweep_free", "phase1c_lateral", "tilt_scan",
    "phase2_manipulation_stance", "phase3_target", "place_partner", "place_distractors", "whole_body_check_at", "brace_second_arm", "posed_in_hand",
    "placement_stance", "facade_v1_spec",
]
