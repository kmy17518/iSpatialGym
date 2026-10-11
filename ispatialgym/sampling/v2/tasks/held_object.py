"""Held-object tasks (spec §6.3): ``object_scaling`` and ``mental_rotation`` on ``SamplerV2`` (TODO D3 + B7).

Init: base = random free cell of the eroded map in the task room + yaw ~ U(−π, π); trunk and the idle arm at the JoyLo
reset pose; the holding arm (right unless the arm choice below picks the left) at the **fixed hold** solved once per robot
(§6.3, §10 ``[pilot]``): pad centre on the head camera's optical axis at depth ``hold.depth_m`` (0.50), trunk ``hold.trunk``
(0.5, 0), approach axis = base +y (the robot's left), closing axis = world down rolled ``hold.roll_deg`` about the approach
axis: 90° for object scaling (fingers horizontal, closing toward the robot, so a side-gripped object faces the camera with
its side; owner 2026-09-26), 15° for mental rotation. The hold is a
constant: it is cached under ``~/.cache/ispatialgym/held_hold.json`` (key = robot + hold constants) and written into
the customs entry as a ``hold`` constraint through the core's constraints write-back. There are no depth or trunk
fallbacks (v1 ``init_depth_fallbacks`` / ``hold_trunks`` are gone): a hold without an IK solution rejects the template
(``hold_pose``).

Arm (owner 2026-09-27, first-demo v2.2): the holding arm is one more dimension of the orientation rule (``orient.choose``) for
the segment hold -> reorient / present. The arms offered are ``tabletop.offered_arms`` (the default arm ``generation.arm``, and
the other arm: it has no role and the held object is free); each solves its hold with its own IK from the default posture, the
left hand's hold being the right hold's mirror image (approach base -y, roll -> -roll: the same closing axis, the right hold
turned π about it through the pad centre). Order: success, joint-limit safe (the hold configuration >= 0.05 rad from every arm
joint limit; neither safe: more clearance), easier to reach (the largest arm-joint change from the default posture); within
``orient.REACH_TOL`` the default arm. A hold whose goal search leaves no option hands the template to the other viable arm once
(``segment_fallback``). ``template_facts["hold_arm"]`` and every summary's ``arm`` record the arm, the reason and
``against_default``. The right hold keeps its cache key and entry, constraint record, B7 prior and record fields; a left hold
has its own cache key (arm + approach), ``left_arm_q`` fields and no B7 prior (the survey was rendered with the right hand).

Grasp: a library grasp of the ``floating`` pipeline (§2.2), realised as v1 §5.2 (fingers advanced 1 mm/step until both
pads touch, 4 mm press through the assisted-grasp window, gravity re-enabled, still held); a row whose attach fails is
``Reject("grasp_failed")`` and the next row is tried on the next attempt — never an AABB heuristic. The realised
grasp (finger joints, object pose in the eef frame, key / name) is what the TRO ``grasps[0]`` stores (loader
contract §3.14).

Option sets are precomputed **once per template** (they depend on the hold and the object, not on the base pose). Mental
rotation (owner 2026-10-11: start and goal orientations both drawn) = per grasp the **start set**: the hold's object
turned about its root by 13 rotation axes (face / edge / corner directions of the base frame, one per antipodal pair) ×
signed angles on a 15° grid inside ``init_angle_range_deg`` (0 = the hold itself), whole object in frame (5 %), the hand
pose reached by the screening IK from the hold with the holding arm's embodiment only (the oracle moves the arm, not the
torso) and the object attached; each instance draws a (grasp, start), places the start configuration at a free base
pose, and draws the goal = the start turned about the object root by the same axes × the 15° grid inside
``goal_angle_range_deg`` (position unchanged), screened at the three intermediate waypoints of §3.9 (SLERP about the
object root, ``conventions.orbit_waypoints``) and confirmed once with the exact IK at its base pose (principle 3), with
the held object attached, whole-body collision-free at every checked pose. Object scaling (owner
2026-09-26) = per grasp the reachable **depth window**: the object at the reference hold slid along the optical axis
over the ``depth_search_grid_m`` grid of ``depth_search_range_m``, a depth passing when the object is in frame and the
screening IK reaches the hand pose, the continuous run through the reference depth shrunk by ``depth_margin_m``; each
instance draws a start depth and a goal depth uniformly in the window with ``|d1 − d0| ≥ min_depth_change_m`` (toward
or away from the camera), solves the start configuration by IK from the default posture and confirms the straight
move (``conventions.line_waypoints``) with the exact IK at its base pose.

Visibility: ≥ ``min_visible_fraction`` (40 %) of the object visible at init and at the goal (segmentation render with
and without the robot) and the whole object in frame. The B7 prior (``runlogs/held_visibility_20260915/visibility.json``,
``scripts/held_visibility/survey.py``) skips library rows whose surveyed visible fraction is below the threshold
before they are realised; it applies only at the hold it was rendered at (its ``hold`` record, roll 15°), so at any
other hold every row is realised and measured by the renders. Mental rotation starts away from that hold (2026-10-11):
there the prior only orders the rows, every start is measured by the renders.

TRO: ``grasps[0]`` (agent, arm, object, key, name, ``object_pose_in_eef_frame``, ``finger_q``, ``ag_anchor_in_eef``) and
``robot_state.controller_targets`` (the drive targets the start settled under, which the instance loader holds), both
from the settled start (``HeldObjectV2.settle_start``); ``pose_targets[target]``
= depth metric in the camera frame (scaling; tolerance max(0.02, 0.20·|Δd|), §8) or the world orientation with the
model's ``orientation_equivalences`` (rotation; 0.15 rad; symmetric models rejected per §3.13 rule 1). Sidecar:
``stances.init`` (arms ``[<holding arm>]``, ``carried`` = target), ``grasps[0]`` (hold eef, waypoints = 3 intermediate + the
goal eef pose), ``goal`` (base, trunk, arms ``same`` + the holding arm's goal configuration ``<arm>_arm_q``), ``language``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym import REPO_ROOT
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import grasp_library as GL
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.tro_writer import pose_target
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import orient as O
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import GraspRecord, Stance

import omnigibson as og  # noqa: E402
from ispatialgym.activity.problem import AGENT_INSTANCE  # noqa: E402
from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from ispatialgym.oracle.action import ActionState  # noqa: E402

log = logging.getLogger("ispatialgym.sampling.v2")

#: the holding arm until the template's arm choice (``HeldObjectV2.arm``); its hold keeps the cache key, entry and records
#: written before the choice existed (owner 2026-09-27, first-demo v2.2)
ARM = "right"
#: the idle arm (the one not holding) and the trunk are at the JoyLo reset pose (§3.3 exception): arms zero = the kit's
#: ``tucked`` mode
IDLE_ARMS = "tucked"
#: grasp pipeline of the held-object tasks (§2.2): the iddl fixes it, the customs entry cannot change it
PIPELINE = "floating"
#: library key suffixes admitted for the floating pipeline ("" = GraspGen floating, ``__f`` floating re-run, ``__sf``
#: floating at the grasp-tested spawn scale of the thin models (§2.2), ``__h`` handle-half rows count as floating for
#: knives)
FLOATING_SUFFIXES = ("", "__f", "__sf", "__h")
HOLD_CACHE_PATH = os.environ.get("ISPATIALGYM_HELD_HOLD_CACHE", os.path.expanduser("~/.cache/ispatialgym/held_hold.json"))
VISIBILITY_PRIOR_PATH = os.environ.get("ISPATIALGYM_HELD_VISIBILITY", os.path.join(REPO_ROOT, "runlogs", "held_visibility_20260915", "visibility.json"))
#: angle grid of the precomputed rotation set (the instance draws one element; 15° keeps the set at 13 x 14 candidates)
ROTATION_ANGLE_STEP_DEG = 15.0
#: assisted-grasp window + margin: control steps the press is held while OmniGibson creates the grasp joint
GRASP_ATTACH_STEPS = 90
#: settle steps under gravity after the attach (the object must stay in hand)
GRASP_GRAVITY_STEPS = 30
#: the recorded start (object in hand) settles under its drive targets before the TRO is written, in chunks of
#: ``START_SETTLE_CHUNK`` control steps until no joint moves more than ``START_SETTLE_TOL_RAD`` over a chunk (at most
#: ``START_SETTLE_MAX_STEPS``)
START_SETTLE_CHUNK = 10
START_SETTLE_TOL_RAD = 1e-5
START_SETTLE_MAX_STEPS = 600


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (tested without a simulator)
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldConstants:
    """The §6.3 fixed hold: pad depth on the optical axis, trunk, roll of the closing axis from world down."""

    depth_m: float = 0.50
    translate: float = 0.5
    tilt: float = 0.0
    roll_deg: float = 15.0

    @staticmethod
    def from_generation(hold: Optional[Dict[str, Any]]) -> "HoldConstants":
        """``generation.hold`` = ``{"depth_m": 0.50, "trunk": [0.5, 0.0], "roll_deg": 15.0}`` (generation_defaults)."""
        if not hold:
            raise ValueError("generation.hold is missing (seed it from generation_defaults)")
        tr = hold.get("trunk", [0.5, 0.0])
        return HoldConstants(float(hold["depth_m"]), float(tr[0]), float(tr[1]), float(hold.get("roll_deg", 15.0)))

    def to_json(self) -> Dict[str, Any]:
        return {"depth_m": self.depth_m, "trunk": [self.translate, self.tilt], "roll_deg": self.roll_deg}


#: sign of each arm's hold: the approach along base ±y and the roll about it (the left hold mirrors the right one)
HOLD_SIDE = {"right": 1.0, "left": -1.0}


def hold_approach(arm: str) -> str:
    """The approach of ``arm``'s hold as the cache entry / constraint record words it."""
    return "base +y" if HOLD_SIDE[arm] > 0 else "base -y"


def hold_cache_key(robot_model: str, hold: HoldConstants, extras: Optional[Dict[str, Any]] = None) -> str:
    """Cache key of the hold solution: robot + hold constants + everything else the solution depends on (idle-arm
    posture, pad offset, camera offset)."""
    d = {"robot": robot_model, "hold": hold.to_json(), **(extras or {})}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def hold_cache_extras(kit, arm: str = ARM) -> Dict[str, Any]:
    """The non-constant inputs of the hold solution (rounded so that every process computes the same key): the idle-arm
    posture, the pad offset and the head-camera offset in the zed link; a left hold adds its arm and approach (the right
    hold's key is the one cached before the arm choice, owner 2026-09-27 first-demo v2.2)."""
    cam = [[round(float(v), 4) for v in kit.cam_in_zed[0]], [round(float(v), 4) for v in kit.cam_in_zed[1]]]
    d = {"idle_arms": IDLE_ARMS, "contact_center_z": round(float(C.CONTACT_CENTER_Z), 5), "cam_in_zed": cam, "robot_model": "r1pro"}
    if arm != "right":
        d.update({"arm": arm, "approach": hold_approach(arm)})
    return d


def hold_eef_pose(cam_pose: C.Pose, base_yaw: float, hold: HoldConstants, arm: str = ARM) -> Tuple[C.Pose, th.Tensor]:
    """The hold eef pose (world) for a head-camera pose and base yaw: pad centre on the optical axis at
    ``hold.depth_m``, approach ``a`` = base +y (robot's left), closing ``c`` = world down rolled ``roll_deg`` about
    ``a``; eef origin = pad − CONTACT_CENTER_Z·a (conventions ``pad_center``). Returns ``((pos, quat), pad)``. The left
    hand's hold is the mirror image (owner 2026-09-27, first-demo v2.2): ``a`` = base -y and the roll negated, which keeps
    the closing axis (it lies in the base x-z plane) - the right hold turned π about the closing axis through the pad
    centre, the exact mirror conjugate of the right hold up to the wrist twin."""
    side = HOLD_SIDE[arm]
    cam_pos, cam_quat = C.as_pose(cam_pose)
    axis = C.quat_rotate(cam_quat, th.tensor([0.0, 0.0, -1.0]))
    pad = cam_pos + axis * float(hold.depth_m)
    a = C.quat_rotate(C.quat_from_yaw(base_yaw), th.tensor([0.0, side, 0.0]))
    c = C.quat_rotate(C.quat_from_axis_angle(a, math.radians(side * hold.roll_deg)), th.tensor([0.0, 0.0, -1.0]))
    quat = C.eef_quat(a, c)
    origin = pad - C.CONTACT_CENTER_Z * a
    return (origin, quat), pad


def hold_json(hold: HoldConstants, arm: str = ARM) -> Dict[str, Any]:
    """The ``hold`` record of the sidecar / summary: the constants, a left hold with its arm and approach."""
    d = hold.to_json()
    if arm != "right":
        d.update({"arm": arm, "approach": hold_approach(arm)})
    return d


def hold_posture(arm: str) -> Dict[str, str]:
    """The init stance's ``posture`` record: the holding ``arm`` at the hold, the idle arm at the JoyLo reset pose."""
    return {f"{a}_arm": ("hold" if a == arm else "reset") for a in ("left", "right")}


def hold_init_arms(arm: str) -> str:
    """``template_meta["init_arms"]`` of a template whose holding arm is ``arm``."""
    return f"{RB.other_arm(arm)} {IDLE_ARMS} (JoyLo reset), {arm} hold"


def hold_record(sol: Dict[str, Any], arm: str = ARM) -> Dict[str, Any]:
    """The ``hold`` constraint (robot-level: empty scope) of ``arm``'s hold solution ``sol``; a left hold names its arm and
    carries ``left_arm_q`` (the right hold's record is the one written before the arm choice)."""
    q_key = f"{arm}_arm_q"
    return {"scope": {}, "type": "hold", "robot": sol["robot"], **({} if arm == "right" else {"arm": arm}), "depth_m": sol["depth_m"],
            "trunk": list(sol["trunk"]), "roll_deg": sol["roll_deg"], "approach": sol["approach"], "closing": sol["closing"], "eef_in_base": sol["eef_in_base"],
            q_key: sol[q_key], "idle_arms": sol["idle_arms"], "hard": False}


def same_hold(rec: Dict[str, Any], sol: Dict[str, Any], arm: str = ARM) -> bool:
    """A written ``hold`` constraint ``rec`` records ``arm``'s hold ``sol``: robot, depth, trunk, roll and the arm (a record without
    one is the right hand's)."""
    return (rec.get("robot") == sol.get("robot") and rec.get("depth_m") == sol.get("depth_m") and list(rec.get("trunk", [])) == list(sol.get("trunk", []))
            and rec.get("roll_deg") == sol.get("roll_deg") and rec.get("arm", "right") == arm)


def depth_grid(depth_range: Sequence[float], grid_m: float) -> List[float]:
    """The ``grid_m`` grid over ``depth_range`` (inclusive, ascending), snapped so 0.15 + k·0.01 reads as 0.15, 0.16, ..."""
    lo, hi = float(depth_range[0]), float(depth_range[1])
    n = int(round((hi - lo) / grid_m))
    return [round(lo + k * grid_m, 6) for k in range(n + 1) if lo + k * grid_m <= hi + 1e-9]


def depth_window(depths: Sequence[float], passed: Sequence[bool], anchor: float,
                 margin_m: float) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """One grasp's object-scaling depth window: the continuous run of passing ``depths`` (an ascending grid) that contains
    ``anchor`` (the reference hold's depth; the longest run when none does), shrunk by ``margin_m`` at both ends.
    Returns ``(window, run)``; ``window`` is None when the shrunk run is empty, both are None without a passing depth."""
    runs: List[Tuple[float, float]] = []
    start = None
    for i, ok in enumerate(passed):
        if ok and start is None:
            start = i
        if start is not None and (not ok or i == len(passed) - 1):
            runs.append((float(depths[start]), float(depths[i if ok else i - 1])))
            start = None
    if not runs:
        return None, None
    containing = [r for r in runs if r[0] - 1e-9 <= anchor <= r[1] + 1e-9]
    run = containing[0] if containing else max(runs, key=lambda r: r[1] - r[0])
    lo, hi = round(run[0] + margin_m, 6), round(run[1] - margin_m, 6)
    return ((lo, hi) if hi >= lo else None), run


def draw_start_goal(rng: np.random.Generator, window: Sequence[float], min_change_m: float, max_draws: int = 10000) -> Tuple[float, float]:
    """Object-scaling start and goal depths: both uniform in ``window``, redrawn until ``|d1 − d0| ≥ min_change_m`` - the
    object moves toward or away from the camera alike."""
    lo, hi = float(window[0]), float(window[1])
    if hi - lo <= min_change_m:
        raise ValueError(f"depth window [{lo:.3f}, {hi:.3f}] m does not admit a {min_change_m:.3f} m move")
    for _ in range(int(max_draws)):
        d0, d1 = float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi))
        if abs(d1 - d0) >= min_change_m:
            return d0, d1
    raise RuntimeError(f"no start / goal pair {min_change_m:.3f} m apart in [{lo:.3f}, {hi:.3f}] m after {max_draws} draws")


def slide_to_depth(eef: C.Pose, eef_in_object: C.Pose, cam_pos, axis: th.Tensor, depth: float) -> Tuple[C.Pose, C.Pose]:
    """The hand pose ``eef`` slid along the camera's unit optical ``axis`` so that the held object's root sits at
    ``depth`` along it (the success target's measure), orientation kept. Returns ``(eef, object root pose)`` there."""
    eef_p, eef_q = (G.t(x).float() for x in eef)
    obj = C.compose((eef_p, eef_q), C.invert(eef_in_object))
    obj_p, obj_q = G.t(obj[0]).float(), G.t(obj[1]).float()
    shift = axis.float() * (float(depth) - float(th.dot(obj_p - G.t(cam_pos).float(), axis.float())))
    return (eef_p + shift, eef_q.clone()), (obj_p + shift, obj_q.clone())


def rotation_axes() -> th.Tensor:
    """13 rotation axes in the base frame, one per antipodal pair of the 26 face / edge / corner directions of a cube
    (unit vectors): the precomputed mental-rotation set is these axes × the signed angle grid, so a rotation by −θ
    about n covers the antipodal axis."""
    dirs = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if (x, y, z) == (0, 0, 0):
                    continue
                if (x, y, z) < (-x, -y, -z):  # one representative per antipodal pair
                    continue
                dirs.append([float(x), float(y), float(z)])
    axes = th.tensor(dirs)
    return axes / th.linalg.norm(axes, dim=1, keepdim=True)


def rotation_goal_set(angle_range_deg: Sequence[float], step_deg: float = ROTATION_ANGLE_STEP_DEG,
                      axes: Optional[th.Tensor] = None) -> List[Tuple[th.Tensor, float]]:
    """Mental-rotation candidates ``(axis (3,) base frame, signed angle deg)``: every axis of :func:`rotation_axes` ×
    the ``step_deg`` grid of magnitudes inside ``angle_range_deg`` (inclusive), both signs."""
    lo, hi = float(angle_range_deg[0]), float(angle_range_deg[1])
    if not (0.0 < lo <= hi):
        raise ValueError(f"goal_angle_range_deg must satisfy 0 < lo <= hi, got {angle_range_deg}")
    n = int(math.floor((hi - lo) / step_deg + 1e-9))
    mags = [round(lo + k * step_deg, 6) for k in range(n + 1)]
    axes = rotation_axes() if axes is None else axes
    out: List[Tuple[th.Tensor, float]] = []
    for ax in axes:
        for m in mags:
            for sign in (1.0, -1.0):
                out.append((ax.clone(), sign * m))
    return out


def rotation_start_set(angle_range_deg: Sequence[float], step_deg: float = ROTATION_ANGLE_STEP_DEG,
                       axes: Optional[th.Tensor] = None) -> List[Tuple[th.Tensor, float]]:
    """Mental-rotation start candidates ``(axis (3,) base frame, signed angle deg)`` (owner 2026-10-11: the start
    orientation is drawn too): the reference hold's object turned about its root by every axis of :func:`rotation_axes`
    × the signed ``step_deg`` grid inside ``angle_range_deg``; a range from 0 includes the hold itself (0°)."""
    lo, hi = float(angle_range_deg[0]), float(angle_range_deg[1])
    if not (0.0 <= lo <= hi):
        raise ValueError(f"init_angle_range_deg must satisfy 0 <= lo <= hi, got {angle_range_deg}")
    out: List[Tuple[th.Tensor, float]] = []
    if lo == 0.0:
        out.append((th.tensor([0.0, 0.0, 1.0]), 0.0))
        lo = step_deg
    if lo <= hi:
        out += rotation_goal_set([lo, hi], step_deg, axes)
    return out


def turned_about_root(obj: C.Pose, base_yaw: float, axis_base, angle_deg: float) -> C.Pose:
    """``obj`` (world) turned about its own root by ``angle_deg`` about the base-frame axis ``axis_base``."""
    axis_w = C.quat_rotate(C.quat_from_yaw(base_yaw), th.as_tensor(axis_base, dtype=th.float32))
    dq = C.quat_from_axis_angle(axis_w, math.radians(float(angle_deg)))
    return G.t(obj[0]).clone(), C.qmul(dq, G.t(obj[1]))


def line_key_poses(eef0: C.Pose, eef1: C.Pose, n_intermediate: int) -> List[C.Pose]:
    """Straight eef motion eef0 -> eef1: ``n_intermediate`` screening poses + the goal (last), spec §3.9 / §6.3
    (``conventions.line_waypoints(p0, p1, n + 1)``)."""
    return [(C.as_tensor(p), C.as_tensor(q)) for p, q in C.line_waypoints(eef0, eef1, int(n_intermediate) + 1)]


def orbit_key_poses(eef0: C.Pose, pivot, delta_quat, n_intermediate: int) -> List[C.Pose]:
    """In-hand rotation about ``pivot`` (the object root) by ``delta_quat``: ``n_intermediate`` SLERP poses + the goal
    (``conventions.orbit_waypoints``, world frame)."""
    return [(C.as_tensor(p), C.as_tensor(q)) for p, q in C.orbit_waypoints(eef0, pivot, delta_quat, int(n_intermediate) + 1, frame="world")]


def depth_tolerance(delta_depth: float, tolerances: Dict[str, Any]) -> float:
    """§8 object scaling: ``max(depth_min_m, depth_fraction · |Δd|)`` (defaults 0.02 m / 0.20)."""
    lo = float(tolerances.get("depth_min_m", 0.02))
    frac = float(tolerances.get("depth_fraction", 0.20))
    return max(lo, frac * abs(float(delta_depth)))


def visible_fraction(visible_px: int, reference_px: int) -> float:
    """Fraction of the object's silhouette (rendered with the robot hidden) that stays visible with the robot."""
    if reference_px <= 0:
        return 0.0
    return min(1.0, float(visible_px) / float(reference_px))


def survey_arm(surveyed: Dict[str, Any]) -> str:
    """The hand a survey's ``hold`` record was rendered with: its ``arm``, else the arm of its configuration / approach (the B7
    survey predates the arm choice: ``right_arm_q``, ``base +y``)."""
    if surveyed.get("arm"):
        return str(surveyed["arm"])
    return "left" if ("left_arm_q" in surveyed or surveyed.get("approach") == hold_approach("left")) else "right"


def survey_matches_hold(surveyed: Optional[Dict[str, Any]], hold: HoldConstants, arm: str = ARM) -> bool:
    """The survey's ``hold`` record (depth, trunk, roll, hand) equals ``hold`` held by ``arm``; a survey without one matches no
    hold. A left hold never matches a right-hand survey (owner 2026-09-27, first-demo v2.2): the same row shows the camera
    another side of the object."""
    if not surveyed:
        return False
    try:
        s = HoldConstants.from_generation(surveyed)
    except (KeyError, TypeError, ValueError):
        return False
    if survey_arm(surveyed) != arm:
        return False
    return all(math.isclose(a, b, abs_tol=1e-6) for a, b in ((s.depth_m, hold.depth_m), (s.translate, hold.translate),
                                                              (s.tilt, hold.tilt), (s.roll_deg, hold.roll_deg)))


def load_visibility_prior(path: str = VISIBILITY_PRIOR_PATH, hold: Optional[HoldConstants] = None, arm: str = ARM) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """The B7 survey (``visibility.json``): ``{(grasp_key, grasp_name): {"visible_fraction", "in_frame", ...}}``; an empty
    dict when the file does not exist, or when ``hold`` is given and the survey was rendered at another hold or with the
    other hand (a visible fraction measured with the fingers at one roll says nothing about another)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    if hold is not None and not survey_matches_hold(data.get("hold"), hold, arm):
        return {}
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in data.get("records", []):
        out[(rec["grasp_key"], rec["grasp_name"])] = rec
    return out


def prior_admits(prior: Dict[Tuple[str, str], Dict[str, Any]], grasp_key: str, grasp_name: str, min_fraction: float) -> Optional[bool]:
    """``True`` / ``False`` when the survey knows the row (visible fraction >= ``min_fraction`` and in frame), ``None``
    when it does not (the row is realised and measured in the sampler)."""
    rec = prior.get((grasp_key, grasp_name))
    if rec is None or rec.get("visible_fraction") is None:
        return None
    return bool(rec.get("in_frame", True)) and float(rec["visible_fraction"]) >= float(min_fraction)


def floating_key(category: str, model: str, preferred: Optional[str]) -> str:
    """The grasp-library key of the floating pipeline (``resolve_key`` with the floating kinds only; a preferred key of
    another pipeline is not accepted — §2.2 "no fallback to another pipeline")."""
    key = GL.resolve_key(category, model, preferred, [PIPELINE])
    stem = f"{category}__{model}"
    if not key.startswith(stem) or key[len(stem):] not in FLOATING_SUFFIXES:
        raise FileNotFoundError(f"{key!r} is not a floating-pipeline key of {category}/{model}")
    return key


def start_controller_targets(robot, state, arm: str) -> Dict[str, float]:
    """``robot_state.controller_targets`` of a settled start (joint name -> drive target of ``state``): the trunk, the idle arm
    and the holding ``arm``, each group complete - the loader applies complete groups only
    (``oracle.action.apply_recorded_targets``)."""
    names = list(robot.joints.keys())
    out: Dict[str, float] = {}
    groups = [(robot.trunk_control_idx, state.trunk_target)] + [(robot.arm_control_idx[a], state.arm_target[a]) for a in (RB.other_arm(arm), arm)]
    for idx, vals in groups:
        for i, v in zip(th.as_tensor(idx).tolist(), vals.tolist()):
            out[names[int(i)]] = float(v)
    return out


@dataclass
class HoldSolution:
    """One arm's fixed hold at the template's parked base pose (``HeldObjectV2.solve_hold``): the configuration ``q`` (28-D) and
    ``q_rel`` (base virtual joints zero), the base-frame eef pose / pad centre / head camera and the record ``sol`` (the cache
    entry, ``template_facts["hold"]``)."""

    arm: str
    q: th.Tensor
    q_rel: th.Tensor
    eef_in_base: C.Pose
    pad_in_base: th.Tensor
    cam_in_base: C.Pose
    sol: Dict[str, Any]


def hold_arm_choice(ikc, default: str, offered: Sequence[str], targets: Dict[str, C.Pose], q_start: Dict[str, th.Tensor],
                    solve: Callable[[str], Tuple[Any, Optional[th.Tensor], str]]) -> Tuple[Optional[O.Scored], Dict[str, Any]]:
    """The holding arm by the orientation rule (owner 2026-09-27, first-demo v2.2; ``orient.choose``): one proposal per offered
    arm (the default arm first), its hold eef pose ``targets[arm]`` solved by ``solve(arm)`` -> ``(result, 28-D hold configuration
    or None, why)``. Success, then joint-limit safe (the hold configuration >= ``IKChecker.JOINT_LIMIT_MARGIN`` from every arm
    joint limit; neither safe: more clearance), then easier to reach (the largest arm-joint change from ``q_start[arm]``, the
    default posture the hold IK starts from); within ``orient.REACH_TOL`` the default arm. Returns ``orient.choose``'s
    ``(chosen, decision)``; ``chosen.result`` is the chosen arm's ``solve`` result."""

    def evaluate(p: O.Proposal):
        result, q, why = solve(p.arm)
        if q is None:
            return False, None, None, why
        return True, result, [q], ""

    props = O.with_arms([(a, [O.Proposal("hold", 0, a, [targets[a]])]) for a in offered])
    return O.choose(ikc, default, props, evaluate, dict(q_start), stage="hold", check_camera=False, map_twins=False)


def hold_arm_record(arm: str, default: str, offered: Sequence[str], decision: Dict[str, Any],
                    fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``template_facts["hold_arm"]``: the holding arm, why (the chooser's ``arm_choice`` reason, the segment fallback or the only
    arm offered, as ``tabletop.segment_arm_summary``), whether it is not the default arm, the arms offered, the chooser's
    ``arm_choice`` and record, and the fallback. ``arm`` / ``reason`` / ``against_default`` go into every instance summary."""
    ac = decision.get("arm_choice")
    why = "segment fallback" if fallback else ((ac or {}).get("reason") or "only arm offered")
    rec = {"arm": arm, "reason": why, "against_default": arm != default, "default": default, "offered": list(offered), "arm_choice": ac, "decision": decision}
    if fallback is not None:
        rec["segment_fallback"] = fallback
    return rec


# ----------------------------------------------------------------------------------------------------------------
# sampler
# ----------------------------------------------------------------------------------------------------------------


class HeldObjectV2(SamplerV2):
    """Shared machinery of the two held-object subtasks; the subclasses define the goal set and the pose target."""

    #: which goal numbers the language block carries
    GOAL_NUMBER_KEY = "goal"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.arm: str = ARM  # the holding arm: the template's arm choice (``choose_hold``)
        self.hold_arm: Dict[str, Any] = {}  # the choice's record (``hold_arm_record``)
        self.hold: Optional[HoldConstants] = None
        self.hold_solution: Dict[str, Any] = {}
        self.q_hold_rel: Optional[th.Tensor] = None  # 28-D, base virtual joints zero
        self.eef_in_base: Optional[C.Pose] = None
        self.pad_in_base: Optional[th.Tensor] = None
        self.cam_in_base: Optional[C.Pose] = None
        self.hull: Optional[th.Tensor] = None
        self.grasps: List[GL.Grasp] = []
        self.grasp_key: Optional[str] = None
        self.options: Dict[str, List[Dict[str, Any]]] = {}  # grasp name -> feasible goal options (base frame)
        self.grasp_stats: Dict[str, Dict[str, int]] = {}
        self.grasp_debug: Dict[str, Any] = {}
        self.prior: Dict[Tuple[str, str], Dict[str, Any]] = {}  # loaded with the hold (sample_template)
        self.equivalences: Optional[List[List[float]]] = None
        self._start_q: Optional[th.Tensor] = None  # the instance's commanded start configuration (settle_start)
        self._controller_targets: Dict[str, float] = {}

    # ------------------------------------------------------------------ generation numbers

    def g(self, key: str, default=None):
        """``generation.<key>`` of the customs entry, else the per-subtask seed of ``generation_defaults`` (the v1
        facade entry lacks the v2 keys); a missing number is a spec error, not a hidden default."""
        v = self.gen(key)
        if v is not None:
            return v
        from ispatialgym.sampling import generation_defaults as GD

        d = GD.defaults_for(self.subtask)
        if key in d:
            return d[key]
        if default is not None:
            return default
        raise Reject("spec", f"{self.gp.alias}: generation.{key} missing and no default in generation_defaults")

    def hold_constants(self) -> HoldConstants:
        return HoldConstants.from_generation(self.g("hold"))

    # ------------------------------------------------------------------ frames

    def base_pose3d(self, base2d) -> C.Pose:
        """The base frame of a base pose: ``base_link`` at ``NOMINAL_BASE_Z`` above the floor (the frame every cached
        hold quantity is expressed in; consistent with ``RobotKit.q_config``)."""
        x, y, yaw = base2d
        return th.tensor([float(x), float(y), self.kit.floor_z + C.NOMINAL_BASE_Z]), C.quat_from_yaw(yaw)

    def q_init_at(self, base2d) -> th.Tensor:
        return self.kit.q_at_base_pose(self.q_hold_rel, *base2d)

    def object_pose_at(self, base2d, grasp: GL.Grasp) -> C.Pose:
        """Object root pose (world) when ``grasp`` holds it at the hold of the base pose: eef ∘ inv(T_obj→eef)."""
        eef_w = C.compose(self.base_pose3d(base2d), self.eef_in_base)
        return C.compose(eef_w, C.invert(grasp.pose))

    def camera_at(self, base2d) -> C.Pose:
        return C.compose(self.base_pose3d(base2d), self.cam_in_base)

    def whole_in_frame(self, cam: C.Pose, obj_pose: C.Pose) -> bool:
        return self.frustum_ok(cam, self.hull_corners_world(self.hull, obj_pose), float(self.g("in_frame_margin")))

    @property
    def target_name(self) -> str:
        return self.bindings.one("target").name

    def room(self) -> Optional[str]:
        req = self.choice.bindings.get("floor")
        if isinstance(req, dict) and req.get("room"):
            return req["room"]
        rooms = self.gp.rooms(self.scene)
        return rooms[0] if rooms else None

    def in_room(self, xy) -> bool:
        room = self.room()
        if room is None:
            return True
        return self.world.scene.seg_map.get_room_instance_by_point(G.t(xy)[:2]) == room

    def scene_obstacles(self) -> List[Any]:
        """Every collision object but the robot: removed from the planner world for the robot-only facts (the hold and
        the per-template goal screening depend on the robot and the object, not on where the robot happens to be
        parked; the world is checked per instance at the sampled base pose)."""
        return [o for o in self.world.scene.objects if o is not self.robot]

    # ------------------------------------------------------------------ the fixed hold (§6.3)

    def hold_start(self) -> Tuple[HoldConstants, Tuple[float, float, float], C.Pose, th.Tensor, C.Pose]:
        """The hold constants and, at the current (parked) base pose, the base pose2d, the base pose, the default posture ``q0``
        (arms zero, trunk ``hold.trunk``: where every hold IK starts) and the head camera there."""
        hc = self.hold_constants()
        x, y, yaw = self.kit.world_base_pose2d()
        q0 = self.kit.q_at_base_pose(self.kit.q_config(arms=IDLE_ARMS, translate=hc.translate, tilt=hc.tilt), x, y, yaw)
        return hc, (x, y, yaw), self.base_pose3d((x, y, yaw)), q0, self.kit.camera_pose(q0)

    def solve_hold(self, arm: str, scene_removed: bool = False) -> HoldSolution:
        """``arm``'s hold configuration at the current (parked) base pose: cache hit verified by FK, else one exact IK from the
        default posture; a failure is ``Reject("hold_pose")`` — no depth / trunk / roll fallbacks. Solves only
        (:meth:`adopt_hold` makes a solution the template's hold); the right hold's cache key and entry are the ones written
        before the arm choice, a left hold has its own (``hold_cache_extras``, ``left_arm_q``). ``scene_removed``: the caller
        already removed the scene from the planner world (one world re-sync for every offered arm's hold)."""
        hc, (x, y, yaw), base3, q0, cam = self.hold_start()
        self.hold = hc
        eef_w, pad_w = hold_eef_pose(cam, yaw, hc, arm)
        key = hold_cache_key("r1pro", hc, hold_cache_extras(self.kit, arm))
        arm_idx = self.robot.arm_control_idx[arm]
        q_key = f"{arm}_arm_q"
        cache = _read_json(HOLD_CACHE_PATH)
        source = "solved"
        q = None
        hit = cache.get(key)
        if hit is not None:
            qc = q0.clone()
            qc[arm_idx] = th.tensor(hit[q_key], dtype=th.float32)
            eef_fk = self.kit.eef_pose(qc, arm)
            if float(th.linalg.norm(G.t(eef_fk[0]) - eef_w[0])) < 0.002 and C.quat_angle(eef_fk[1], eef_w[1]) < math.radians(1.0):
                q, source = qc, "cache"
            else:
                log.warning("held hold cache %s: FK mismatch, re-solving", key)
        with contextlib.nullcontext() if scene_removed else self.kit.ignoring(self.scene_obstacles()):  # a robot-only fact: solved and checked against the robot itself
            if q is None:
                q = self.kit.ik(eef_w, q0, arm=arm)
                if q is None:
                    raise Reject("hold_pose", f"no {arm}-arm IK solution for the fixed hold (depth {hc.depth_m}, trunk ({hc.translate}, {hc.tilt}), roll {hc.roll_deg} deg; "
                                              f"pad in base {[round(float(v), 3) for v in C.relative((pad_w, C.IDENTITY_QUAT), base3)[0]]})")
            if not bool(self.kit.collision_free(q, margin=0.01)[0]):
                raise Reject("hold_pose", "the hold configuration self-collides")
        q_rel = q.clone()
        q_rel[[self.kit.jidx[n] for n in M.BASE_VIRTUAL_JOINTS]] = 0.0
        eef_in_base = C.relative(eef_w, base3)
        pad_in_base = C.relative((pad_w, C.IDENTITY_QUAT), base3)[0]
        cam_in_base = C.relative(cam, base3)
        roll = hc.roll_deg * HOLD_SIDE[arm]
        sol = {"key": key, "source": source, "robot": "r1pro", **({} if arm == "right" else {"arm": arm}), **hc.to_json(), "approach": hold_approach(arm),
               "closing": f"world down rolled {roll} deg about the approach",
               "eef_in_base": {"position": eef_in_base[0].tolist(), "orientation": eef_in_base[1].tolist()},
               "pad_in_base": pad_in_base.tolist(), "camera_in_base": {"position": cam_in_base[0].tolist(), "orientation": cam_in_base[1].tolist()},
               q_key: q[arm_idx].tolist(), "idle_arms": IDLE_ARMS}
        if source == "solved":
            cache[key] = {k: v for k, v in sol.items() if k not in ("source",)}
            _write_json(HOLD_CACHE_PATH, cache)
        log.info("%s: fixed %s-arm hold (%s) depth %.2f trunk (%.2f, %.2f) roll %.0f deg", self.gp.alias, arm, source, hc.depth_m, hc.translate, hc.tilt, hc.roll_deg)
        return HoldSolution(arm, q, q_rel, eef_in_base, pad_in_base, cam_in_base, sol)

    def adopt_hold(self, h: HoldSolution) -> None:
        """Make ``h`` the template's hold: the holding arm (:attr:`arm`), the hold configuration, its base-frame quantities and
        the B7 prior of that hold (none for a left hold: every library row is realised and measured)."""
        self.arm = h.arm
        self.q_hold_rel, self.eef_in_base, self.pad_in_base, self.cam_in_base = h.q_rel, h.eef_in_base, h.pad_in_base, h.cam_in_base
        self.hold_solution = h.sol
        self.template_meta["init_arms"] = hold_init_arms(h.arm)
        self.prior = load_visibility_prior(VISIBILITY_PRIOR_PATH, self.hold, h.arm)
        if not self.prior:
            log.info("%s: no B7 visibility prior for this hold (%s arm, roll %.0f deg); every library row is realised and measured", self.gp.alias, h.arm,
                     self.hold.roll_deg)

    def choose_hold(self) -> None:
        """The holding arm and the template's goal set (owner 2026-09-27, first-demo v2.2): every offered arm
        (``tabletop.offered_arms``: the default arm, and the other arm - it has no role, the held object is free) solves its
        hold, :func:`hold_arm_choice` orders them, and the goal search (:meth:`precompute_options`) runs with the first. The
        goal search is the segment's later check: when it leaves no option the next viable arm's hold gets the template's
        goal search once (``segment_fallback``). ``template_facts["hold_arm"]`` records the choice."""
        default, offered = TT.default_arm(self), TT.offered_arms(self, free=True)
        hc, (_, _, yaw), _, q0, cam = self.hold_start()
        solved: Dict[str, HoldSolution] = {}
        failed: Dict[str, Reject] = {}

        def solve(arm: str):
            try:
                solved[arm] = self.solve_hold(arm, scene_removed=True)
            except Reject as e:
                failed[arm] = e
                return None, None, str(e)
            except Exception as e:  # noqa: BLE001 - the other arm is an offer: its planner failing never costs the default arm's template
                if arm == default:
                    raise
                log.warning("%s: the %s arm's hold could not be solved (%r); not offered", self.gp.alias, arm, e)
                failed[arm] = Reject("hold_pose", f"the {arm} arm's hold raised {e!r}"[:300])
                return None, None, str(failed[arm])
            return solved[arm], solved[arm].q, ""

        with self.kit.ignoring(self.scene_obstacles()):  # robot-only facts: every offered arm's hold against the robot itself
            chosen, decision = hold_arm_choice(self.ikc, default, offered, {a: hold_eef_pose(cam, yaw, hc, a)[0] for a in offered}, {a: q0 for a in offered}, solve)
        if chosen is None:
            if len(failed) == 1:
                raise next(iter(failed.values()))
            raise Reject("hold_pose", "; ".join(f"{a} arm: {e}" for a, e in failed.items()))
        self.adopt_hold(chosen.result)
        self.hold_arm = self.template_facts["hold_arm"] = hold_arm_record(self.arm, default, offered, decision)
        try:
            self.precompute_options()
            return
        except Reject as e:
            others = [a for a in O.viable_arms(decision) if a != self.arm and a in solved]
            if e.reason != "goal_infeasible" or not others:
                raise
            first = e
        fallback = {"from_arm": self.arm, "why": str(first)[:300], "goal_options": self.template_facts.get("goal_options")}
        TT._count(self, "segment_fallbacks")
        log.info("%s: segment fallback: the %s arm's hold leaves no goal option (%s); the goal search runs with the %s arm's hold", self.gp.alias, self.arm,
                 str(first)[:160], others[0])
        self.adopt_hold(solved[others[0]])
        self.hold_arm = self.template_facts["hold_arm"] = hold_arm_record(self.arm, default, offered, decision, fallback)
        try:
            self.precompute_options()
        except Reject as e:
            if e.reason != "goal_infeasible":
                raise
            fallback["failed"] = str(e)[:300]
            raise Reject("goal_infeasible", f"{fallback['from_arm']} arm: {first}; {self.arm} arm (segment fallback): {e}")
        TT._count(self, f"segment_fallback_taken.{self.arm}")

    def arm_summary(self) -> Dict[str, Any]:
        """Every instance's ``summary["arm"]``: the template's holding arm, why, and whether it is not the default arm."""
        return {"arm": self.arm, "reason": self.hold_arm.get("reason"), "against_default": bool(self.hold_arm.get("against_default", False))}

    def arm_offer_record(self) -> Dict[str, Any]:
        """``grasps[].arm_offer`` (fired / not fired; the oracle copies it into the notes of the segment's actions): ``fired`` when both
        arms' holds competed (``hold_arm_choice``), the arms offered, the holding arm and why, else why the other arm was not offered."""
        offered = list(self.hold_arm.get("offered") or [self.arm])
        out = {"fired": len(offered) > 1, "template_arms": offered, "arms": offered, "arm": self.arm, "reason": self.hold_arm.get("reason"),
               "against_default": bool(self.hold_arm.get("against_default", False)), "mirrored_band": False}
        if len(offered) < 2:
            out["not_fired"] = TT.arm_offer(self, free=True)[1] or "one arm offered"
        return out

    # ------------------------------------------------------------------ template (§6.3 precomputation)

    def sample_template(self):
        self.template_meta["init_arms"] = hold_init_arms(self.arm)
        target = self.obj("target")
        choice = self.choice.models["target"]
        self.hull = self.world.local_hull(target)
        if len(self.hull) == 0:
            raise Reject("spec", f"{target.name} has no collision geometry")
        try:
            self.grasp_key = floating_key(choice.category, choice.model, choice.grasp_key)
            self.grasps = GL.load_grasps(self.grasp_key)
        except (FileNotFoundError, ValueError) as e:
            raise Reject(f"missing_grasps:{PIPELINE}", f"{choice.category}/{choice.model}: {e}")
        self.equivalences = self.gp.orientation_equivalences("target", choice.category, choice.model)
        self.check_symmetry()
        self.choose_hold()
        self.template_facts.update({"hold": self.hold_solution, "grasp_key": self.grasp_key, "n_grasps": len(self.grasps),
                                    "orientation_equivalences": None if self.equivalences is None else len(self.equivalences),
                                    "visibility_prior_rows": len(self.prior)})
        self.template_meta["hold"] = self.hold_solution
        self.template_meta["hold_arm"] = self.hold_arm

    def check_symmetry(self):
        """§3.13 rule 1 hook (mental rotation only)."""

    def key_poses(self, eef0: C.Pose, eef1: C.Pose, obj0: C.Pose) -> List[C.Pose]:
        """The §3.9 key poses of the in-hand motion eef0 -> eef1 (3 intermediate + the goal)."""
        raise NotImplementedError

    def precompute_options(self):
        """The template's per-grasp options (base-frame quantities, valid at any base pose) that every instance draws from
        (:attr:`options`, grasp name -> options); ``Reject("goal_infeasible")`` when no grasp has one."""
        raise NotImplementedError

    # ------------------------------------------------------------------ instance (§6.3)

    def default_q_at(self, base2d) -> th.Tensor:
        """The default posture (arms zero, trunk ``hold.trunk``) at a base pose: the seed of every hold IK."""
        return self.kit.q_at_base_pose(self.kit.q_config(arms=IDLE_ARMS, translate=self.hold.translate, tilt=self.hold.tilt), *base2d)

    def sample_start_base(self, rng: np.random.Generator, q_rel: th.Tensor) -> Tuple[float, float, float]:
        """Random free cell of the eroded map inside the task room + yaw ~ U(−π, π) at which the instance's start
        configuration ``q_rel`` is whole-body collision-free (2 cm; the object is checked once it is attached)."""
        pts = self.kit.random_free_points(rng, 30, room_filter=self.in_room)
        if not pts:
            raise Reject("no_stance", "no free eroded cell in the task room")
        cands = [(float(xy[0]), float(xy[1]), float(rng.uniform(-math.pi, math.pi))) for xy in pts]
        free = self.kit.collision_free(th.stack([self.kit.q_at_base_pose(q_rel, *b) for b in cands]), margin=0.02)
        for b, f in zip(cands, free.tolist()):
            if f:
                return b
        raise Reject("no_stance", "no collision-free base pose for the start configuration")

    def grasp_order(self, rng: np.random.Generator) -> List[str]:
        """Grasps with options: clean rows before tolerated ``one_finger`` rows, higher surveyed visibility first, rows
        whose attach failed before last (per-template statistics), ties random."""
        names = list(self.options)
        names = [names[i] for i in rng.permutation(len(names))]
        flagged = {g.name for g in self.grasps if g.flags or g.verdict != "ok"}

        def vis(n):
            rec = self.prior.get((self.grasp_key, n))
            return -float(rec["visible_fraction"]) if rec and rec.get("visible_fraction") is not None else 0.0

        names.sort(key=lambda n: (n in flagged, self.grasp_stats.get(n, {}).get("failed", 0) - 2 * self.grasp_stats.get(n, {}).get("ok", 0), vis(n)))
        return names

    def describe_goal(self, info: Dict[str, Any], base2d) -> Dict[str, Any]:
        return info

    def goal_numbers(self, info: Dict[str, Any]) -> Dict[str, float]:
        return {}

    def pose_target_for(self, cam: C.Pose, goal: C.Pose, info: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    # ------------------------------------------------------------------ grasp realisation (v1 §5.2)

    def apply_draft(self, draft: Draft):
        tname = self.target_name
        target = self.obj("target")
        self.kit.apply(draft.robot_q, draft.base_pose2d)
        pose = draft.object_poses[tname]
        target.set_position_orientation(G.t(pose[0]), G.t(pose[1]))
        target.disable_gravity()
        target.wake()  # sleeping bodies keep a stale render transform after a physics-side teleport
        target.keep_still()
        gname = draft.extras["grasps"][0]["grasp_name"]
        stats = self.grasp_stats.setdefault(gname, {"ok": 0, "failed": 0})
        try:
            self.establish_grasp(target, pose)
        except Reject:
            stats["failed"] += 1
            raise
        stats["ok"] += 1
        draft.summary["grasp_realised"] = dict(self.grasp_debug)
        # the realised object pose (assisted grasping pulls the object into the pads by up to a few mm) and the closed
        # finger positions; the goal object pose follows the realised eef->object transform so that the object at the
        # goal is exactly where the hand at the goal configuration carries it
        p, q = target.get_position_orientation()
        draft.object_poses[tname] = (p.clone(), q.clone())
        draft.robot_q = self.kit.q_current()
        eef = tuple(x.float() for x in self.robot.eef_links[self.arm].get_position_orientation())
        rel = C.relative((p.float(), q.float()), eef)
        draft.extras["grasps"][0]["object_pose_in_eef_frame"] = {"position": rel[0].tolist(), "orientation": rel[1].tolist()}
        finger_q = self.robot.get_joint_positions()[self.robot.gripper_control_idx[self.arm]].tolist()
        draft.extras["grasps"][0]["finger_q"] = finger_q
        q_goal = G.t(draft.extras["goal_robot_q"])
        eef_goal = self.kit.eef_pose(q_goal, self.arm)
        goal = C.compose(eef_goal, rel)
        draft.goal_poses[tname] = goal
        cam_goal = self.kit.camera_pose(q_goal)
        draft.pose_targets[tname] = self.pose_target_for(cam_goal, goal, draft.summary["goal"])
        rec = self.sidecar.grasp_of(tname)
        if rec is not None:
            rec.finger_q = finger_q
            rec.eef_in_object = C.invert(rel)
        st = self.sidecar.stances.get("init")
        if st is not None:
            st.q = draft.robot_q
        draft.summary["goal_realised"] = {"position": goal[0].tolist(), "orientation": goal[1].tolist(),
                                          "eef_offset_deg": math.degrees(C.quat_angle(rel[1], C.invert(self.grasp_of_draft(draft).pose)[1]))}

    def grasp_of_draft(self, draft: Draft) -> GL.Grasp:
        name = draft.extras["grasps"][0]["grasp_name"]
        return next(g for g in self.grasps if g.name == name)

    def finger_joints(self):
        """Gripper joint indices of the holding arm paired with the finger link each one moves (``JointPrim.body1``)."""
        gidx = [int(i) for i in self.robot.gripper_control_idx[self.arm].tolist()]
        names = list(self.robot.joints.keys())
        finger_paths = {l.prim_path for l in self.robot.finger_links[self.arm]}
        pairs = []
        for i in gidx:
            body1 = str(self.robot.joints[names[i]].body1)
            link = next((p for p in finger_paths if p == body1 or p.endswith(body1.split("/")[-1])), None)
            pairs.append((i, link))
        if any(l is None for _, l in pairs) or len({l for _, l in pairs}) != len(pairs):
            pairs = [(i, p) for i, p in zip(gidx, sorted(finger_paths))]  # positional pairing
        return pairs

    def establish_grasp(self, target, pose: C.Pose):
        """Close the gripper on the object at the library pose without moving the object (v1 §5.2): each finger is
        advanced ``grasp_finger_advance_m_per_step`` per step until it touches, the drives then aim
        ``grasp_press_m`` past the contact positions while the assisted-grasp window elapses, the object is re-posed
        every step until the joint exists, gravity is re-enabled and the object must stay held. Any failure is
        ``Reject("grasp_failed")``."""
        state = ActionState(self.robot)
        state.hold_current()
        pairs = self.finger_joints()
        gidx = th.tensor([i for i, _ in pairs], dtype=th.long)
        lo, hi = self.robot.joint_lower_limits[gidx], self.robot.joint_upper_limits[gidx]
        target_paths = {l.prim_path for l in target.links.values()}

        def cmd_for(q):  # smooth gripper controller: [-1, 1] -> [lower, upper] joint position target (both fingers)
            return float(th.clamp(2.0 * (q.mean() - lo.mean()) / (hi.mean() - lo.mean()) - 1.0, -1.0, 1.0))

        def touching():
            _, links = self.robot._find_gripper_contacts(self.arm)
            touched = set()
            for p in target_paths:
                touched |= set(links.get(p, ()))
            return [path in touched for _, path in pairs]

        q = hi.clone()
        state.gripper_cmd[self.arm] = cmd_for(q)
        for _ in range(8):
            target.set_position_orientation(G.t(pose[0]), G.t(pose[1]))
            target.keep_still()
            self.robot.set_joint_positions(q, indices=gidx, drive=False)
            self.world.control_step(state.action())
        step = float(self.g("grasp_finger_advance_m_per_step"))
        contact = touching()
        for _ in range(int(float((hi - lo).max()) / step) + 10):
            if all(contact):
                break
            moved = False
            for k in range(len(pairs)):
                if not contact[k] and float(q[k]) > float(lo[k]) + 1e-4:
                    q[k] = max(float(lo[k]), float(q[k]) - step)
                    moved = True
            if not moved:
                raise Reject("grasp_failed", f"fingers closed without both pads touching the object (contact {contact})")
            self.robot.set_joint_positions(q, indices=gidx, drive=False)
            state.gripper_cmd[self.arm] = cmd_for(q)
            target.keep_still()
            self.world.control_step(state.action())
            contact = touching()
        if not all(contact):
            raise Reject("grasp_failed", f"pads not both in contact after closing (contact {contact})")
        q_contact = q.clone()
        press = float(self.g("grasp_press_m"))
        state.gripper_cmd[self.arm] = cmd_for(th.clamp(q_contact - press, min=lo))
        attached_ok = False
        for _ in range(GRASP_ATTACH_STEPS):
            if self.robot._ag_obj_in_hand[self.arm] is None:
                target.set_position_orientation(G.t(pose[0]), G.t(pose[1]))
                target.keep_still()
            self.world.control_step(state.action())
            if self.robot._ag_obj_in_hand[self.arm] is target:
                attached_ok = True
                break
        if not attached_ok:
            raise Reject("grasp_failed", "assisted grasping did not attach the object (both pads in contact)")

        def offset_now():
            eef = tuple(x.float() for x in self.robot.eef_links[self.arm].get_position_orientation())
            return C.relative(tuple(x.float() for x in target.get_position_orientation()), eef)

        eef0 = tuple(x.float() for x in self.robot.eef_links[self.arm].get_position_orientation())
        lib_rel = C.relative(pose, eef0)
        rel_attach = offset_now()
        self.robot.set_joint_positions(q_contact, indices=gidx, drive=False)
        state.gripper_cmd[self.arm] = cmd_for(q_contact)
        target.enable_gravity()
        self.world.control_step(state.action(), n=GRASP_GRAVITY_STEPS)
        if self.robot._ag_obj_in_hand[self.arm] is not target:
            raise Reject("grasp_failed", "object dropped after gravity was re-enabled")
        rel_grav = offset_now()
        self.grasp_debug = {
            "attach_vs_library_mm": 1000 * float(th.linalg.norm(rel_attach[0] - lib_rel[0])),
            "gravity_vs_attach_mm": 1000 * float(th.linalg.norm(rel_grav[0] - rel_attach[0])),
            "gravity_vs_library_mm": 1000 * float(th.linalg.norm(rel_grav[0] - lib_rel[0])),
            "gravity_vs_library_deg": math.degrees(C.quat_angle(rel_grav[1], lib_rel[1])),
            "finger_q_contact": q_contact.tolist(),
        }
        log.info("%s: grasp realised: %s", self.gp.alias, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.grasp_debug.items()})
        self.robot.set_joint_positions(q_contact, indices=gidx, drive=False)  # pads on the object in the saved state
        self.robot.keep_still()
        self.world.full_step(render=True)  # sync transforms / caches after the non-rendering control steps
        self.robot.keep_still()
        self.kit.sync_world()

    def release(self):
        try:
            if self.robot._ag_obj_in_hand.get(self.arm) is not None:
                self.robot.release_grasp_immediately(self.arm)
        except Exception:
            pass
        state = ActionState(self.robot)
        state.hold_current()
        state.set_gripper(self.arm, close=False)
        self.world.control_step(state.action(), n=10)

    def on_reject(self, err: Reject):
        self.release()
        super().on_reject(err)

    # ------------------------------------------------------------------ visibility (§6.3 renders)

    def self_occlusion(self, target) -> Dict[str, Any]:
        vis = self.world.object_pixels(target)
        ref = self.world.object_pixels(target, hidden=[self.robot])
        if ref == 0:
            self.world.ensure_segmented(target)
            vis = self.world.object_pixels(target)
            ref = self.world.object_pixels(target, hidden=[self.robot])
        if ref == 0:
            raise Reject("seg_missing", f"{target.name} has no instance-segmentation pixels even with the robot hidden")
        return {"visible_pixels": int(vis), "reference_pixels": int(ref), "visible_fraction": visible_fraction(vis, ref)}

    def extra_checks(self, draft: Draft):
        target = self.obj("target")
        occ = self.self_occlusion(target)
        min_vis = float(self.g("min_visible_fraction"))
        if occ["visible_fraction"] < min_vis:
            raise Reject("self_occlusion", f"init visible fraction {occ['visible_fraction']:.2f} < {min_vis} ({occ['visible_pixels']} of {occ['reference_pixels']} px)")
        cam = self.world.camera_pose()
        if not self.whole_in_frame(cam, tuple(x.float() for x in target.get_position_orientation())):
            raise Reject("out_of_view", "held object not fully inside the init image after grasping")
        draft.summary["init_visibility"] = occ

    def check_goal_feasible(self, draft: Draft):
        """Goal = the robot at the goal arm configuration with the object in hand: whole object in view, visible
        fraction within bounds and every binary goal atom (``pose_reached``) true there. Joints and object pose are
        written together before the single physics step that refreshes link poses (a step with the fingers away from
        the object would let the assisted-grasp joint yank it); ``grasped`` itself is not re-evaluated at the goal."""
        from omnigibson.object_states import PoseReached

        tname = self.target_name
        target = self.obj("target")
        state = self.world.snapshot()
        try:
            q_goal = G.t(draft.extras["goal_robot_q"]).clone()
            base_idx = [self.kit.jidx[n] for n in M.BASE_VIRTUAL_JOINTS]
            q_goal[base_idx] = self.robot.get_joint_positions()[base_idx]
            goal = draft.goal_poses[tname]
            self.robot.set_joint_positions(q_goal)
            self.robot.set_joint_velocities(th.zeros_like(q_goal))
            self.robot.keep_still()
            target.set_position_orientation(G.t(goal[0]), G.t(goal[1]))
            target.keep_still()
            og.sim.step_physics()
            self.robot.keep_still()
            target.keep_still()
            self.world.render(3)
            cam = self.world.camera_pose()
            obj_now = tuple(x.float() for x in target.get_position_orientation())
            if not self.whole_in_frame(cam, obj_now):
                raise Reject("goal_out_of_view", f"object leaves the image at the goal (obj {[round(float(v), 3) for v in obj_now[0]]})")
            occ = self.self_occlusion(target)
            min_vis = float(self.g("min_visible_fraction"))
            if occ["visible_fraction"] < min_vis:
                raise Reject("goal_self_occlusion", f"goal visible fraction {occ['visible_fraction']:.2f} < {min_vis}")
            self.configure_targets(draft)
            st = target.states[PoseReached]
            if not bool(st.get_value()):
                raise Reject("goal_infeasible", f"pose_reached false at the goal configuration (errors {st.get_error()})")
            _, status = self.compiled.check_goal(self._evaluate)
            report = self.goal_report(status)
            failed = [c["bddl"] for c in report if not c["satisfied"] and "pose_settled" not in c["bddl"]]
            if [f for f in failed if "grasped" not in f]:
                raise Reject("goal_infeasible", f"unsatisfied at the goal configuration: {failed}")
            draft.summary["goal_check"] = {"goal": report}
            draft.summary["goal_visibility"] = occ
        finally:
            self.world.restore(state)
            self.robot.keep_still()
            target.keep_still()
            self.kit.sync_world()
            self.configure_targets(draft)

    def refine_targets_from_goal(self, draft: Draft):
        pass

    def check_stances_with_objects(self, draft: Draft):
        """§4.6 step 5 for a stance that *holds* the object: the realised init configuration (hold arm, reset idle arm)
        is checked with the target attached to the planner — the core's generic check locks the arms untucked and keeps
        the object as a world obstacle, which puts the untucked holding hand into the held object. TEMP until the core
        honours ``Stance.q`` / ``Stance.carried``."""
        self.kit.sync_world()
        with attached(self.kit, self.arm, self.obj("target")):
            if not bool(self.kit.collision_free(G.t(draft.robot_q), margin=0.01)[0]):
                raise Reject("stance_collision_with_objects", "the hold configuration collides with the world at the init base pose (object attached)")

    def program_check(self, draft: Draft) -> Dict[str, Any]:
        """§3.5 with the held object attached: the program's first action (``Reorient``) already carries the target
        (``stances.init.carried``), no ``Grasp`` precedes it, so the object is attached to the planner for the whole
        check (the realised in-hand pose is what the simulator holds at this point). TEMP until the core's program
        check seeds its carried-object tracker from ``stances.init.carried``."""
        with attached(self.kit, self.arm, self.obj("target")):
            return super().program_check(draft)

    # ------------------------------------------------------------------ writing / constraints

    def write_instance(self, draft: Draft, index: int, seed: int) -> str:
        self.settle_start(draft)
        # the sidecar's init configuration is the state the TRO records: demo and evaluator start from one pose
        st = self.sidecar.stances.get("init")
        if st is not None:
            st.q = self.kit.q_current()
        path = super().write_instance(draft, index, seed)
        self.release()
        target = self.obj("target")
        self.world.park(target, list(self.world.spawned).index(target.name))
        og.sim.step_physics()
        self.kit.sync_world()
        return path

    def settle_start(self, draft: Draft) -> None:
        """The recorded start is an equilibrium under recorded drive targets. The holding arm is held at ``_start_q``
        (the hold for mental rotation, the start's IK solution for object scaling), the trunk and the idle arm at their
        current joints, the fingers where the grasp left them; the payload settles the arm (it rests a steady offset
        below its targets) until it stops moving, and the targets go into the TRO's ``robot_state.controller_targets``
        (:func:`start_controller_targets`), which the instance loader holds - a loader holding the loaded joints themselves
        lets the payload sag once more. The grasp record and the subtask's start quantities (:meth:`record_settled_start`)
        are read from the settled state."""
        state = ActionState(self.robot)
        state.set_from_q(self._start_q, arms=(self.arm,), trunk=False)
        gidx = self.robot.gripper_control_idx[self.arm]
        lo, hi = self.robot.joint_lower_limits[gidx], self.robot.joint_upper_limits[gidx]
        fingers = self.robot.get_joint_positions()[gidx]
        state.gripper_cmd[self.arm] = float(th.clamp(2.0 * (fingers.mean() - lo.mean()) / (hi.mean() - lo.mean()) - 1.0, -1.0, 1.0))
        steps, moved = 0, float("inf")
        while steps < START_SETTLE_MAX_STEPS and moved > START_SETTLE_TOL_RAD:
            q0 = self.robot.get_joint_positions().clone()
            self.world.control_step(state.action(), n=START_SETTLE_CHUNK)
            steps += START_SETTLE_CHUNK
            moved = float(th.max(th.abs(self.robot.get_joint_positions() - q0)))
        draft.summary["start_settle"] = {"steps": steps, "last_chunk_motion_rad": moved}
        self._controller_targets = start_controller_targets(self.robot, state, self.arm)
        eef = tuple(x.float() for x in self.robot.eef_links[self.arm].get_position_orientation())
        self.record_settled_start(draft, eef)
        # the grasp as the loader restores it: fingers, in-hand pose and the assisted-grasp joint's anchor (the finger
        # contact point OmniGibson chose, in the eef frame) from the settled state
        rel = C.relative(tuple(x.float() for x in self.obj("target").get_position_orientation()), eef)
        finger_q = self.robot.get_joint_positions()[gidx].tolist()
        gr = draft.extras["grasps"][0]
        gr["object_pose_in_eef_frame"] = {"position": rel[0].tolist(), "orientation": rel[1].tolist()}
        gr["finger_q"] = finger_q
        params = (getattr(self.robot, "_ag_obj_constraint_params", None) or {}).get(self.arm) or {}
        if params.get("parent_frame_pos") is not None:
            gr["ag_anchor_in_eef"] = (th.as_tensor(params["parent_frame_pos"], dtype=th.float32) * th.as_tensor(self.robot.scale, dtype=th.float32)).tolist()
        rec = self.sidecar.grasp_of(self.target_name)
        if rec is not None:
            rec.grasp = eef
            rec.pre_grasp = IK.pre_pose(eef)
            rec.finger_q = finger_q
            rec.eef_in_object = C.invert(rel)

    def record_settled_start(self, draft: Draft, eef: C.Pose) -> None:
        """The subtask's start quantities read from the settled state (``eef`` = the settled hand pose)."""

    def tro_meta_block(self, draft: Draft, instance_id: str, seed: int) -> Dict[str, Any]:
        meta = super().tro_meta_block(draft, instance_id, seed)
        if self._controller_targets:
            meta["robot_state"]["controller_targets"] = dict(self._controller_targets)
        return meta

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        """The ``hold`` constant (once per entry and holding arm) and the confirmed (grasp, goal) as a ``grasp_set`` prior."""
        recs: List[Dict[str, Any]] = []
        sol = self.hold_solution
        if not any(same_hold(r, sol, self.arm) for r in self.gp.constraints_for("hold")):
            recs.append(hold_record(sol, self.arm))
        g = draft.summary.get("grasp") or {}
        choice = self.choice.models["target"]
        recs.append({"scope": {"target": f"{choice.category}/{choice.model}"}, "type": "grasp_set", "trunk": [self.hold.translate, self.hold.tilt],
                     "hold": True, **({} if self.arm == "right" else {"arm": self.arm}), "feasible": [[0.0, g.get("name")]], "goal": draft.summary.get("goal"),
                     "hard": False})
        return recs


# ----------------------------------------------------------------------------------------------------------------
# object scaling
# ----------------------------------------------------------------------------------------------------------------


@register("object_scaling")
class ObjectScalingV2(HeldObjectV2):
    """§6.3 (owner 2026-09-26): per instance a start depth d0 and a goal depth d1 of the object root along the head
    camera's optical axis, both uniform in the grasp's reachable depth window with |d1 − d0| >= ``min_depth_change_m``,
    orientation unchanged. The start is the reference hold (``solve_hold``) slid along the axis, its configuration solved
    by IK from the default posture; target frame = camera sensor, ``position_metric = depth``, tolerance
    max(0.02, 0.20·|Δd|)."""

    def precompute_options(self):
        """Per library grasp (B7 prior first): the depth window. The object at the reference hold is slid along the
        optical axis over the ``depth_search_grid_m`` grid of ``depth_search_range_m``; a depth passes when the whole
        object is in frame (5 %) and the screening IK (the holding arm from the default posture, object attached, robot +
        object only) reaches the hand pose. The window is the continuous run of passing depths through the reference
        depth, shrunk by ``depth_margin_m`` at both ends; a grasp whose window does not admit a ``min_depth_change_m``
        move gets no options. Base-frame quantities: the windows hold at any base pose."""
        target = self.obj("target")
        ref = self.kit.world_base_pose2d()
        self.ref_base2d = ref
        base3 = self.base_pose3d(ref)
        cam = C.compose(base3, self.cam_in_base)
        axis = self.world.camera_model.optical_axis(cam)
        eef_ref = C.compose(base3, self.eef_in_base)
        q_hold = self.q_init_at(ref)
        q_default = self.default_q_at(ref)
        min_vis = float(self.g("min_visible_fraction"))
        grid = depth_grid(self.g("depth_search_range_m"), float(self.g("depth_search_grid_m")))
        margin = float(self.g("depth_margin_m"))
        min_change = float(self.g("min_depth_change_m"))
        stats = {"grasps": len(self.grasps), "prior_skipped": 0, "prior_unknown": 0, "held_out_of_view": 0, "depths": len(grid), "too_narrow": 0,
                 "windows": 0}
        self.options = {}
        windows: Dict[str, Any] = {}
        skipped = []
        for g in self.grasps:
            admits = prior_admits(self.prior, g.key, g.name, min_vis)
            if admits is False:
                stats["prior_skipped"] += 1
                skipped.append(g.name)
                continue
            if admits is None:
                stats["prior_unknown"] += 1
            obj_ref = C.compose(eef_ref, C.invert(g.pose))
            if not self.whole_in_frame(cam, obj_ref):
                stats["held_out_of_view"] += 1
                windows[g.name] = {"held_out_of_view": True}
                continue
            d_ref = float(th.dot(G.t(obj_ref[0]) - G.t(cam[0]), axis))
            poses, in_view = [], []
            for depth in grid:
                eef_d, obj_d = slide_to_depth(eef_ref, g.pose, cam[0], axis, depth)
                poses.append(eef_d)
                in_view.append(bool(self.whole_in_frame(cam, obj_d)))
            reach = [False] * len(grid)
            idx = [i for i, v in enumerate(in_view) if v]
            if idx:
                # the object at the reference hold in the simulation so the planner attaches it with the right transform
                self.kit.apply(q_hold, ref)
                target.set_position_orientation(obj_ref[0], obj_ref[1])
                target.keep_still()
                self.kit.sync_world()
                with self.kit.ignoring(self.scene_obstacles()), attached(self.kit, self.arm, target):
                    ok = self.kit.ik_batch([poses[i] for i in idx], q_default, arm=self.arm)
                for i, s in zip(idx, ok.tolist()):
                    reach[i] = bool(s)
            window, run = depth_window(grid, [v and r for v, r in zip(in_view, reach)], d_ref, margin)
            windows[g.name] = {"reference_depth": round(d_ref, 4), "run": run, "window": window,
                               "ik_failed": [dd for dd, v, r in zip(grid, in_view, reach) if v and not r],
                               "out_of_view": [dd for dd, v in zip(grid, in_view) if not v]}
            if window is None or window[1] - window[0] <= min_change:
                stats["too_narrow"] += 1
                continue
            self.options[g.name] = {"window": window, "reference_depth": d_ref}
            stats["windows"] += 1
        self.world.park(target, list(self.world.spawned).index(target.name))
        self.kit.sync_world()
        self.template_facts["goal_options"] = {**stats, "grasps_with_options": len(self.options), "prior_skipped_names": skipped,
                                               "per_grasp": {k: list(v["window"]) for k, v in self.options.items()}}
        self.template_facts["depth_windows"] = windows
        if not self.options:
            raise Reject("goal_infeasible", f"no library grasp has a depth window admitting a {min_change:.2f} m move ({stats}; {windows})")
        log.info("%s: depth windows per grasp %s (%s)", self.gp.alias, {k: v["window"] for k, v in self.options.items()}, stats)

    def start_config(self, g: GL.Grasp, d0: float) -> Optional[Tuple[th.Tensor, C.Pose, C.Pose]]:
        """The start of an instance: ``g``'s object with its root at depth ``d0`` (the reference hold slid along the optical
        axis) and the configuration holding it, solved by IK from the default posture with the scene ignored and checked
        free of self-collision (as ``solve_hold``). Returns ``(q with the base virtual joints zero, eef in the base frame,
        object root in the base frame)`` or None."""
        ref = self.ref_base2d
        base3 = self.base_pose3d(ref)
        cam = C.compose(base3, self.cam_in_base)
        eef_w, obj_w = slide_to_depth(C.compose(base3, self.eef_in_base), g.pose, cam[0], self.world.camera_model.optical_axis(cam), d0)
        with self.kit.ignoring(self.scene_obstacles()):
            q = self.kit.ik(eef_w, self.default_q_at(ref), arm=self.arm)
            if q is None or not bool(self.kit.collision_free(q, margin=0.01)[0]):
                return None
        q_rel = q.clone()
        q_rel[[self.kit.jidx[n] for n in M.BASE_VIRTUAL_JOINTS]] = 0.0
        return q_rel, C.relative(eef_w, base3), C.relative(obj_w, base3)

    def record_settled_start(self, draft: Draft, eef: C.Pose) -> None:
        """The start the summary, the sidecar and the language record (object depth, pad depth, depth change) from the
        settled state."""
        cam = self.world.camera_pose()
        axis = self.world.camera_model.optical_axis(cam)
        d0 = float(th.dot(G.t(self.obj("target").get_position_orientation()[0]).float() - G.t(cam[0]), axis))
        info = draft.summary["goal"]
        info["init_depth_m"] = d0
        info["depth_change_m"] = float(info["goal_depth_m"]) - d0
        hold = draft.summary["hold"]  # the same dict as the sidecar's init stance and goal blocks
        hold["start_depth_m"] = d0
        hold["depth_m"] = round(float(th.dot(G.t(C.pad_center(eef)) - G.t(cam[0]), axis)), 6)
        self.sidecar.language["goal_numbers"].update(self.goal_numbers(info))

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        tname = self.target_name
        target = self.obj("target")
        hc = self.hold
        n_wp = int(self.g("waypoints"))
        min_change = float(self.g("min_depth_change_m"))
        chosen = None
        errors = []
        for gname in self.grasp_order(rng)[:4]:
            g = next(x for x in self.grasps if x.name == gname)
            opt = self.options[gname]
            d0, d1 = draw_start_goal(rng, opt["window"], min_change)
            start = self.start_config(g, d0)
            if start is None:
                errors.append(f"{gname}: no start configuration at {d0:.3f} m")
                continue
            q_rel, eef0_b, obj0_b = start
            try:
                base2d = self.sample_start_base(rng, q_rel)
            except Reject as e:
                errors.append(f"{gname} at {d0:.3f} m: {e}")
                continue
            base3 = self.base_pose3d(base2d)
            q_init = self.kit.q_at_base_pose(q_rel, *base2d)
            cam = self.camera_at(base2d)
            axis = self.world.camera_model.optical_axis(cam)
            eef0, obj0 = C.compose(base3, eef0_b), C.compose(base3, obj0_b)
            if not self.whole_in_frame(cam, obj0):
                errors.append(f"{gname}: start at {d0:.3f} m out of frame")
                continue
            self.kit.apply(q_init, base2d)
            target.set_position_orientation(obj0[0], obj0[1])
            target.keep_still()
            self.kit.sync_world()
            init = Stance("init", base2d, hc.translate, hc.tilt, arms=[self.arm], carried=tname)
            init.q = q_init
            with attached(self.kit, self.arm, target):
                if not bool(self.kit.collision_free(q_init, margin=0.01)[0]):
                    errors.append(f"{gname}: start at {d0:.3f} m collides with the object attached")
                    continue
                eef1, goal = slide_to_depth(eef0, g.pose, cam[0], axis, d1)
                wps = self.key_poses(eef0, eef1, obj0)
                res = self.ikc.waypoints_ok([eef0] + wps, init, self.arm, q_seed=q_init, scale_key=self.choice.models["target"].scale)
                if not res.ok:
                    errors.append(f"{gname} {d0:.3f} -> {d1:.3f} m: {res.reason}")
                    continue
            if not self.whole_in_frame(cam, goal):
                errors.append(f"{gname}: goal at {d1:.3f} m out of frame")
                continue
            chosen = (g, obj0, eef0, goal, wps, res.q, base2d, cam, axis, init, opt)
            break
        if chosen is None:
            raise Reject("goal_infeasible", "no (grasp, start, goal) confirmed: " + "; ".join(errors[-4:]))
        g, obj0, eef0, goal, wps, q_goal, base2d, cam, axis, init, opt = chosen
        self._start_q = init.q.clone()  # the start's IK solution: the drive targets the recorded start settles under
        d0 = float(th.dot(G.t(obj0[0]) - G.t(cam[0]), axis))
        d1 = float(th.dot(G.t(goal[0]) - G.t(cam[0]), axis))
        info = {"init_depth_m": d0, "goal_depth_m": d1, "depth_change_m": d1 - d0, "direction": "away" if d1 > d0 else "toward",
                "window_m": list(opt["window"])}
        # the hold of this instance: the reference hold's pad slid with the object (depth_m = pad depth, as HoldConstants)
        hold = {**hold_json(hc, self.arm), "depth_m": round(hc.depth_m + d0 - float(opt["reference_depth"]), 6), "start_depth_m": d0,
                "reference_depth_m": hc.depth_m}
        init.extras.update({"posture": hold_posture(self.arm), "hold": hold})
        d.object_poses[tname] = obj0
        d.goal_poses[tname] = goal
        d.robot_q = init.q
        d.base_pose2d = base2d
        d.settle_free = []
        obj_in_eef = C.invert(g.pose)
        d.extras["grasps"] = [{"agent": AGENT_INSTANCE, "arm": self.arm, "object": tname, "grasp_key": g.key, "grasp_name": g.name,
                               "object_pose_in_eef_frame": {"position": obj_in_eef[0].tolist(), "orientation": obj_in_eef[1].tolist()}}]
        d.extras["goal_robot_q"] = q_goal
        d.summary.update({"grasp": {"key": g.key, "name": g.name, "flags": list(g.flags)}, "goal": info, "feasibility": {"embodiment": f"{self.arm}_arm"},
                          "hold": hold})
        d.summary["arm"] = self.arm_summary()
        d.pose_targets[tname] = self.pose_target_for(cam, goal, info)
        self.sidecar.add_stance(init)
        rec = GraspRecord(tname, self.arm, g.key, g.name, 0.0, pre_grasp=IK.pre_pose(eef0), grasp=eef0, carry_height=0.0, finger_q=list(g.finger_q),
                          eef_in_object=g.pose, stance="init", waypoints=wps)
        rec.extras["arm_offer"] = self.arm_offer_record()
        self.sidecar.add_grasp(rec)
        goal_block = self.sidecar.set_goal(base2d, hc.translate, hc.tilt, arms="same")
        goal_block.update({f"{self.arm}_arm_q": q_goal[self.robot.arm_control_idx[self.arm]].tolist(), "embodiment": f"{self.arm}_arm",
                           "eef": {"position": wps[-1][0].tolist(), "orientation": wps[-1][1].tolist()}, "hold": hold})
        self.sidecar.describe(tname, self.choice.models["target"].category.replace("_", " "), **self.goal_numbers(info))
        return d

    def key_poses(self, eef0: C.Pose, eef1: C.Pose, obj0: C.Pose) -> List[C.Pose]:
        return line_key_poses(eef0, eef1, int(self.g("waypoints")))

    def pose_target_for(self, cam: C.Pose, goal: C.Pose, info: Dict[str, Any]) -> Dict[str, Any]:
        p_cam = C.relative((G.t(goal[0]), C.IDENTITY_QUAT), cam)[0]  # goal root position in the camera frame
        tol = depth_tolerance(info["depth_change_m"], dict(self.g("tolerances")))
        return pose_target(position=p_cam.tolist(), position_metric="depth", position_tolerance=tol, frame=f"{AGENT_INSTANCE}::{self.world.cam_key}")

    def goal_numbers(self, info: Dict[str, Any]) -> Dict[str, float]:
        """Signed change (< 0 toward the camera), goal and start depths, and ``direction`` +1 away from / −1 toward the
        camera (the language block holds numbers only)."""
        change = float(info["depth_change_m"])
        return {"depth_change_cm": 100.0 * change, "goal_depth_cm": 100.0 * float(info["goal_depth_m"]),
                "start_depth_cm": 100.0 * float(info["init_depth_m"]), "direction": 1.0 if change > 0 else -1.0}


# ----------------------------------------------------------------------------------------------------------------
# mental rotation
# ----------------------------------------------------------------------------------------------------------------


@register("mental_rotation")
class MentalRotationV2(HeldObjectV2):
    """§6.3 goal: rotation about a sampled axis (13 base-frame axes × the signed 15° grid inside
    ``goal_angle_range_deg``) about the object root, position unchanged; world orientation target, 0.15 rad,
    ``orientation_equivalences`` per model (§3.13); symmetric models are rejected (``symmetric_target``).

    Start (owner 2026-10-11: the start orientation is drawn too, not always the hold's): the reference hold's object turned
    about its root by an axis × angle of ``init_angle_range_deg`` (0 = the hold itself), the configuration holding it solved
    by the screening IK from the hold once per template (:meth:`precompute_options`); each instance draws a grasp and a
    start, then a goal that turns that start, screened at the instance's base pose and confirmed by the exact IK."""

    #: goal candidates screened per drawn start (a random subset of the axis × angle set), exact checks per start, and starts
    #: tried per grasp in one attempt
    GOAL_SCREEN_MAX = 96
    GOAL_EXACT_MAX = 4
    STARTS_PER_GRASP = 3

    def check_symmetry(self):
        if self.equivalences is not None and len(self.equivalences) > 1 and bool(self.g("reject_symmetric_targets", True)):
            c = self.choice.models["target"]
            raise Reject("symmetric_target", f"{c.category}/{c.model} has a rotational symmetry ({len(self.equivalences)} equivalent orientations); §3.13 rule 1")

    def precompute_options(self):
        """Per library grasp, the start set: the reference hold's object turned by every element of
        :func:`rotation_start_set` (``init_angle_range_deg``), whole object in frame (5 %), its hand pose reached by the
        screening IK from the hold (the holding arm, object attached, robot + object only, the scene removed from the
        planner world); the IK solutions are kept as the start configurations. Base-frame quantities: the starts hold at
        any base pose."""
        target = self.obj("target")
        ref = self.kit.world_base_pose2d()
        self.ref_base2d = ref
        base3 = self.base_pose3d(ref)
        cam = C.compose(base3, self.cam_in_base)
        eef_hold = C.compose(base3, self.eef_in_base)
        q_hold = self.q_init_at(ref)
        base_idx = [self.kit.jidx[n] for n in M.BASE_VIRTUAL_JOINTS]
        starts = rotation_start_set(self.g("init_angle_range_deg"), ROTATION_ANGLE_STEP_DEG)
        stats = {"grasps": len(self.grasps), "starts": len(starts), "grasps_out_of_view": 0, "out_of_view": 0, "ik_failed": 0, "feasible": 0}
        self.options = {}
        for g in self.grasps:
            obj_hold = C.compose(eef_hold, C.invert(g.pose))
            entries = []
            for axis_b, ang in starts:
                obj0 = turned_about_root(obj_hold, ref[2], axis_b, ang)
                if not self.whole_in_frame(cam, obj0):
                    stats["out_of_view"] += 1
                    continue
                entries.append((obj0, C.compose(obj0, g.pose), {"axis_base": axis_b.tolist(), "angle_deg": float(ang)}))
            if not entries:
                stats["grasps_out_of_view"] += 1
                continue
            # the object at the hold in the simulation so the planner attaches it with the right transform
            self.kit.apply(q_hold, ref)
            target.set_position_orientation(obj_hold[0], obj_hold[1])
            target.keep_still()
            self.kit.sync_world()
            with self.kit.ignoring(self.scene_obstacles()), attached(self.kit, self.arm, target):
                ok, qs = self.kit.ik_batch_solve([eef0 for _, eef0, _ in entries], q_hold, arm=self.arm)
            opts = []
            for (obj0, eef0, info), s, q in zip(entries, ok.tolist(), qs):
                if not s:
                    stats["ik_failed"] += 1
                    continue
                q_rel = G.t(q).clone()
                q_rel[base_idx] = 0.0
                opts.append({"q_rel": q_rel, "obj_in_base": C.relative(obj0, base3), "eef_in_base": C.relative(eef0, base3), "info": info})
                stats["feasible"] += 1
            if opts:
                self.options[g.name] = opts
        self.world.park(target, list(self.world.spawned).index(target.name))
        self.kit.sync_world()
        self.template_facts["start_options"] = {**stats, "grasps_with_options": len(self.options), "per_grasp": {k: len(v) for k, v in self.options.items()}}
        self.template_meta["init_orientation"] = {"drawn": True, "init_angle_range_deg": list(self.g("init_angle_range_deg")),
                                                  "goal_angle_range_deg": list(self.g("goal_angle_range_deg"))}
        if not self.options:
            raise Reject("goal_infeasible", f"no library grasp has a start in view that the screening IK reaches ({stats})")
        log.info("%s: %d grasps with start options (%s)", self.gp.alias, len(self.options), stats)

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        tname = self.target_name
        target = self.obj("target")
        hc = self.hold
        n_wp = int(self.g("waypoints"))
        goal_set = rotation_goal_set(self.g("goal_angle_range_deg"), ROTATION_ANGLE_STEP_DEG)
        scale_key = self.choice.models["target"].scale
        chosen = None
        errors: List[str] = []
        for gname in self.grasp_order(rng)[:4]:
            g = next(x for x in self.grasps if x.name == gname)
            opts = self.options[gname]
            for si in rng.permutation(len(opts))[: self.STARTS_PER_GRASP]:
                opt = opts[int(si)]
                tag = f"{gname} start {opt['info']['angle_deg']:+.0f} deg"
                try:
                    base2d = self.sample_start_base(rng, opt["q_rel"])
                except Reject as e:
                    errors.append(f"{tag}: {e}")
                    continue
                base3 = self.base_pose3d(base2d)
                q_init = self.kit.q_at_base_pose(opt["q_rel"], *base2d)
                cam = self.camera_at(base2d)
                eef0, obj0 = C.compose(base3, opt["eef_in_base"]), C.compose(base3, opt["obj_in_base"])
                self.kit.apply(q_init, base2d)
                target.set_position_orientation(obj0[0], obj0[1])
                target.keep_still()
                self.kit.sync_world()
                init = Stance("init", base2d, hc.translate, hc.tilt, arms=[self.arm], carried=tname)
                init.q = q_init
                with attached(self.kit, self.arm, target):
                    if not bool(self.kit.collision_free(q_init, margin=0.01)[0]):
                        errors.append(f"{tag}: start collides with the object attached")
                        continue
                    cands = []
                    for k in rng.permutation(len(goal_set)):
                        axis_b, ang = goal_set[int(k)]
                        goal = turned_about_root(obj0, base2d[2], axis_b, ang)
                        if self.whole_in_frame(cam, goal):
                            cands.append((goal, {"axis_base": axis_b.tolist(), "angle_deg": float(ang)}))
                            if len(cands) >= self.GOAL_SCREEN_MAX:
                                break
                    if not cands:
                        errors.append(f"{tag}: no goal in frame")
                        continue
                    wps_all = [self.key_poses(eef0, C.compose(goal, g.pose), obj0) for goal, _ in cands]
                    if any(len(w) != n_wp + 1 for w in wps_all):
                        raise Reject("spec", f"key poses != generation.waypoints ({n_wp}) + 1")
                    ok = self.kit.ik_batch([p for w in wps_all for p in w], q_init, arm=self.arm).view(len(cands), n_wp + 1).all(dim=1)
                    passed = [i for i, s in enumerate(ok.tolist()) if s]
                    if not passed:
                        errors.append(f"{tag}: none of {len(cands)} goals passes the screening IK")
                    for ci in passed[: self.GOAL_EXACT_MAX]:
                        res = self.ikc.waypoints_ok([eef0] + wps_all[ci], init, self.arm, q_seed=q_init, scale_key=scale_key)
                        if not res.ok:
                            errors.append(f"{tag} -> {cands[ci][1]['angle_deg']:+.0f} deg: {res.reason}")
                            continue
                        chosen = (g, obj0, eef0, cands[ci][0], dict(cands[ci][1]), res.q, wps_all[ci], base2d, cam, init, opt)
                        break
                if chosen is not None:
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise Reject("goal_infeasible", "no (grasp, start, goal) confirmed: " + "; ".join(errors[-4:]))
        g, obj0, eef0, goal, info, q_goal, wps, base2d, cam, init, opt = chosen
        self._start_q = init.q.clone()  # the start's IK solution: the drive targets the recorded start settles under
        info["start"] = dict(opt["info"])
        info = self.describe_goal(info, base2d)
        hold = {**hold_json(hc, self.arm), "start_rotation": dict(info["start"])}
        init.extras.update({"posture": hold_posture(self.arm), "hold": hold})
        d.object_poses[tname] = obj0
        d.goal_poses[tname] = goal
        d.robot_q = init.q
        d.base_pose2d = base2d
        d.settle_free = []
        obj_in_eef = C.invert(g.pose)
        d.extras["grasps"] = [{"agent": AGENT_INSTANCE, "arm": self.arm, "object": tname, "grasp_key": g.key, "grasp_name": g.name,
                               "object_pose_in_eef_frame": {"position": obj_in_eef[0].tolist(), "orientation": obj_in_eef[1].tolist()}}]
        d.extras["goal_robot_q"] = q_goal
        d.summary.update({"grasp": {"key": g.key, "name": g.name, "flags": list(g.flags)}, "goal": info, "feasibility": {"embodiment": f"{self.arm}_arm"},
                          "hold": hold})
        d.summary["arm"] = self.arm_summary()
        d.pose_targets[tname] = self.pose_target_for(cam, goal, info)
        # sidecar: init stance (held, at the drawn start), the grasp with the in-hand key poses, the goal arm configuration, language
        self.sidecar.add_stance(init)
        rec = GraspRecord(tname, self.arm, g.key, g.name, 0.0, pre_grasp=IK.pre_pose(eef0), grasp=eef0, carry_height=0.0, finger_q=list(g.finger_q),
                          eef_in_object=g.pose, stance="init", waypoints=wps)
        rec.extras["arm_offer"] = self.arm_offer_record()
        self.sidecar.add_grasp(rec)
        goal_block = self.sidecar.set_goal(base2d, hc.translate, hc.tilt, arms="same")
        goal_block.update({f"{self.arm}_arm_q": q_goal[self.robot.arm_control_idx[self.arm]].tolist(), "embodiment": f"{self.arm}_arm",
                           "eef": {"position": wps[-1][0].tolist(), "orientation": wps[-1][1].tolist()}, "hold": hold})
        self.sidecar.describe(tname, self.choice.models["target"].category.replace("_", " "), **self.goal_numbers(info))
        return d

    def key_poses(self, eef0: C.Pose, eef1: C.Pose, obj0: C.Pose) -> List[C.Pose]:
        # the rotation that carries the eef from eef0 to eef1 about the object root (the object turns in place)
        dq = C.qmul(G.t(eef1[1]), C.qinv(G.t(eef0[1])))
        return orbit_key_poses(eef0, G.t(obj0[0]), dq, int(self.g("waypoints")))

    def describe_goal(self, info: Dict[str, Any], base2d) -> Dict[str, Any]:
        """The start -> goal rotation (``axis_base`` / ``angle_deg``) and the start's turn from the hold (``start``), each
        with its world axis."""
        info = dict(info)
        to_world = C.quat_from_yaw(base2d[2])
        info["axis_world"] = C.quat_rotate(to_world, th.tensor(info["axis_base"], dtype=th.float32)).tolist()
        if info.get("start"):
            info["start"] = {**info["start"], "axis_world": C.quat_rotate(to_world, th.tensor(info["start"]["axis_base"], dtype=th.float32)).tolist()}
        return info

    def pose_target_for(self, cam: C.Pose, goal: C.Pose, info: Dict[str, Any]) -> Dict[str, Any]:
        tol = float(dict(self.g("tolerances")).get("orientation", 0.15))
        return pose_target(orientation=G.t(goal[1]).tolist(), orientation_metric="full", orientation_tolerance=tol, equivalences=self.equivalences or [])

    def goal_numbers(self, info: Dict[str, Any]) -> Dict[str, float]:
        return {"angle_deg": float(info["angle_deg"])}


# ----------------------------------------------------------------------------------------------------------------
# small file helpers
# ----------------------------------------------------------------------------------------------------------------


def _read_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)
