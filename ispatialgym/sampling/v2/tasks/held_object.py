"""Held-object tasks (spec §6.3): ``object_scaling`` and ``mental_rotation`` on ``SamplerV2`` (TODO D3 + B7).

Init: base = random free cell of the eroded map in the task room + yaw ~ U(−π, π); trunk and left arm at the JoyLo
reset pose; right arm at the **fixed hold** solved once per robot (§6.3, §10 ``[pilot]``): pad centre on the head
camera's optical axis at depth ``hold.depth_m`` (0.50), trunk ``hold.trunk`` (0.5, 0), approach axis = base +y (the
robot's left), closing axis = world down rolled ``hold.roll_deg`` about the approach axis: 90° for object scaling
(fingers horizontal, closing toward the robot, so a side-gripped object faces the camera with its side; owner
2026-09-26), 15° for mental rotation. The hold is a
constant: it is cached under ``~/.cache/ispatialgym/held_hold.json`` (key = robot + hold constants) and written into
the customs entry as a ``hold`` constraint through the core's constraints write-back. There are no depth or trunk
fallbacks (v1 ``init_depth_fallbacks`` / ``hold_trunks`` are gone): a hold without an IK solution rejects the template
(``hold_pose``).

Grasp: a library grasp of the ``floating`` pipeline (§2.2), realised as v1 §5.2 (fingers advanced 1 mm/step until both
pads touch, 4 mm press through the assisted-grasp window, gravity re-enabled, still held); a row whose attach fails is
``Reject("grasp_failed")`` and the next row is tried on the next attempt — never an AABB heuristic. The realised
grasp (finger joints, object pose in the eef frame, key / name) is what the TRO ``grasps[0]`` stores (loader
contract §3.14).

Goal sets are precomputed **once per template** (they depend on the hold and the object, not on the base pose):
object scaling = the 5 cm depth grid over ``goal_depth_range_m`` with ``|d1 − d0| ≥ min_depth_change_m`` (same
orientation); mental rotation = 13 rotation axes (face / edge / corner directions of the base frame, one per antipodal
pair) × signed angles on a 15° grid inside ``goal_angle_range_deg`` (position unchanged). Each candidate is screened
with the right-arm embodiment only (the oracle moves the arm, not the torso) at the three intermediate waypoints of
§3.9 (straight line along the camera ray / SLERP about the object root, ``conventions.line_waypoints`` /
``orbit_waypoints``) with the object attached to the planner, and the whole object must stay in frame (5 %). Each
instance draws a (grasp, goal) from the feasible set and confirms it once with the exact IK at its base pose
(principle 3), with the held object attached, whole-body collision-free at every checked pose.

Visibility: ≥ ``min_visible_fraction`` (40 %) of the object visible at init and at the goal (segmentation render with
and without the robot) and the whole object in frame. The B7 prior (``runlogs/held_visibility_20260915/visibility.json``,
``scripts/held_visibility/survey.py``) skips library rows whose surveyed visible fraction is below the threshold
before they are realised; it applies only at the hold it was rendered at (its ``hold`` record, roll 15°), so at any
other hold every row is realised and measured by the renders.

TRO: ``grasps[0]`` (agent, arm, object, key, name, ``object_pose_in_eef_frame``, ``finger_q``); ``pose_targets[target]``
= depth metric in the camera frame (scaling; tolerance max(0.02, 0.20·|Δd|), §8) or the world orientation with the
model's ``orientation_equivalences`` (rotation; 0.15 rad; symmetric models rejected per §3.13 rule 1). Sidecar:
``stances.init`` (arms ``["right"]``, ``carried`` = target), ``grasps[0]`` (hold eef, waypoints = 3 intermediate + the
goal eef pose), ``goal`` (base, trunk, arms ``same`` + the goal right-arm configuration), ``language``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym import REPO_ROOT
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import grasp_library as GL
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.tro_writer import pose_target
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import GraspRecord, Stance

import omnigibson as og  # noqa: E402
from ispatialgym.activity.problem import AGENT_INSTANCE  # noqa: E402
from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from ispatialgym.oracle.action import ActionState  # noqa: E402

log = logging.getLogger("ispatialgym.sampling.v2")

ARM = "right"
#: the idle (left) arm and the trunk are at the JoyLo reset pose (§3.3 exception): arms zero = the kit's ``tucked`` mode
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


def hold_cache_key(robot_model: str, hold: HoldConstants, extras: Optional[Dict[str, Any]] = None) -> str:
    """Cache key of the hold solution: robot + hold constants + everything else the solution depends on (idle-arm
    posture, pad offset, camera offset)."""
    d = {"robot": robot_model, "hold": hold.to_json(), **(extras or {})}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def hold_cache_extras(kit) -> Dict[str, Any]:
    """The non-constant inputs of the hold solution (rounded so that every process computes the same key): the idle-arm
    posture, the pad offset and the head-camera offset in the zed link."""
    cam = [[round(float(v), 4) for v in kit.cam_in_zed[0]], [round(float(v), 4) for v in kit.cam_in_zed[1]]]
    return {"idle_arms": IDLE_ARMS, "contact_center_z": round(float(C.CONTACT_CENTER_Z), 5), "cam_in_zed": cam, "robot_model": "r1pro"}


def hold_eef_pose(cam_pose: C.Pose, base_yaw: float, hold: HoldConstants) -> Tuple[C.Pose, th.Tensor]:
    """The hold eef pose (world) for a head-camera pose and base yaw: pad centre on the optical axis at
    ``hold.depth_m``, approach ``a`` = base +y (robot's left), closing ``c`` = world down rolled ``roll_deg`` about
    ``a``; eef origin = pad − CONTACT_CENTER_Z·a (conventions ``pad_center``). Returns ``((pos, quat), pad)``."""
    cam_pos, cam_quat = C.as_pose(cam_pose)
    axis = C.quat_rotate(cam_quat, th.tensor([0.0, 0.0, -1.0]))
    pad = cam_pos + axis * float(hold.depth_m)
    a = C.quat_rotate(C.quat_from_yaw(base_yaw), th.tensor([0.0, 1.0, 0.0]))
    c = C.quat_rotate(C.quat_from_axis_angle(a, math.radians(hold.roll_deg)), th.tensor([0.0, 0.0, -1.0]))
    quat = C.eef_quat(a, c)
    origin = pad - C.CONTACT_CENTER_Z * a
    return (origin, quat), pad


def depth_goal_set(d0: float, depth_range: Sequence[float], grid_m: float, min_change_m: float) -> List[float]:
    """Object-scaling goal depths: the ``grid_m`` grid over ``depth_range`` (inclusive) with ``|d1 − d0| ≥ min_change_m``,
    ascending. Grid values are snapped to the grid step so 0.35 + k·0.05 reads as 0.35, 0.40, ... exactly."""
    lo, hi = float(depth_range[0]), float(depth_range[1])
    n = int(round((hi - lo) / grid_m))
    out = []
    for k in range(n + 1):
        d1 = round(lo + k * grid_m, 6)
        if d1 > hi + 1e-9:
            break
        if abs(d1 - d0) + 1e-9 >= min_change_m:
            out.append(d1)
    return out


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


def survey_matches_hold(surveyed: Optional[Dict[str, Any]], hold: HoldConstants) -> bool:
    """The survey's ``hold`` record (depth, trunk, roll) equals ``hold``; a survey without one matches no hold."""
    if not surveyed:
        return False
    try:
        s = HoldConstants.from_generation(surveyed)
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isclose(a, b, abs_tol=1e-6) for a, b in ((s.depth_m, hold.depth_m), (s.translate, hold.translate),
                                                              (s.tilt, hold.tilt), (s.roll_deg, hold.roll_deg)))


def load_visibility_prior(path: str = VISIBILITY_PRIOR_PATH, hold: Optional[HoldConstants] = None) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """The B7 survey (``visibility.json``): ``{(grasp_key, grasp_name): {"visible_fraction", "in_frame", ...}}``; an empty
    dict when the file does not exist, or when ``hold`` is given and the survey was rendered at another hold (a visible
    fraction measured with the fingers at one roll says nothing about another)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    if hold is not None and not survey_matches_hold(data.get("hold"), hold):
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


# ----------------------------------------------------------------------------------------------------------------
# sampler
# ----------------------------------------------------------------------------------------------------------------


class HeldObjectV2(SamplerV2):
    """Shared machinery of the two held-object subtasks; the subclasses define the goal set and the pose target."""

    #: which goal numbers the language block carries
    GOAL_NUMBER_KEY = "goal"

    def __init__(self, ctx):
        super().__init__(ctx)
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

    def solve_hold(self) -> Dict[str, Any]:
        """The hold configuration at the current (parked) base pose: cache hit verified by FK, else one exact IK; a
        failure is ``Reject("hold_pose")`` for the template — no depth / trunk / roll fallbacks."""
        hc = self.hold_constants()
        self.hold = hc
        x, y, yaw = self.kit.world_base_pose2d()
        base3 = self.base_pose3d((x, y, yaw))
        q0 = self.kit.q_at_base_pose(self.kit.q_config(arms=IDLE_ARMS, translate=hc.translate, tilt=hc.tilt), x, y, yaw)
        cam = self.kit.camera_pose(q0)
        eef_w, pad_w = hold_eef_pose(cam, yaw, hc)
        key = hold_cache_key("r1pro", hc, hold_cache_extras(self.kit))
        arm_idx = self.robot.arm_control_idx[ARM]
        cache = _read_json(HOLD_CACHE_PATH)
        source = "solved"
        q = None
        hit = cache.get(key)
        if hit is not None:
            qc = q0.clone()
            qc[arm_idx] = th.tensor(hit["right_arm_q"], dtype=th.float32)
            eef_fk = self.kit.eef_pose(qc, ARM)
            if float(th.linalg.norm(G.t(eef_fk[0]) - eef_w[0])) < 0.002 and C.quat_angle(eef_fk[1], eef_w[1]) < math.radians(1.0):
                q, source = qc, "cache"
            else:
                log.warning("held hold cache %s: FK mismatch, re-solving", key)
        with self.kit.ignoring(self.scene_obstacles()):  # a robot-only fact: solved and checked against the robot itself
            if q is None:
                q = self.kit.ik(eef_w, q0, arm=ARM)
                if q is None:
                    raise Reject("hold_pose", f"no right-arm IK solution for the fixed hold (depth {hc.depth_m}, trunk ({hc.translate}, {hc.tilt}), roll {hc.roll_deg} deg; "
                                              f"pad in base {[round(float(v), 3) for v in C.relative((pad_w, C.IDENTITY_QUAT), base3)[0]]})")
            if not bool(self.kit.collision_free(q, margin=0.01)[0]):
                raise Reject("hold_pose", "the hold configuration self-collides")
        q_rel = q.clone()
        q_rel[[self.kit.jidx[n] for n in M.BASE_VIRTUAL_JOINTS]] = 0.0
        self.q_hold_rel = q_rel
        self.eef_in_base = C.relative(eef_w, base3)
        self.pad_in_base = C.relative((pad_w, C.IDENTITY_QUAT), base3)[0]
        self.cam_in_base = C.relative(cam, base3)
        sol = {"key": key, "source": source, "robot": "r1pro", **hc.to_json(), "approach": "base +y", "closing": f"world down rolled {hc.roll_deg} deg about the approach",
               "eef_in_base": {"position": self.eef_in_base[0].tolist(), "orientation": self.eef_in_base[1].tolist()},
               "pad_in_base": self.pad_in_base.tolist(), "camera_in_base": {"position": self.cam_in_base[0].tolist(), "orientation": self.cam_in_base[1].tolist()},
               "right_arm_q": q[arm_idx].tolist(), "idle_arms": IDLE_ARMS}
        if source == "solved":
            cache[key] = {k: v for k, v in sol.items() if k not in ("source",)}
            _write_json(HOLD_CACHE_PATH, cache)
        self.hold_solution = sol
        log.info("%s: fixed hold (%s) depth %.2f trunk (%.2f, %.2f) roll %.0f deg", self.gp.alias, source, hc.depth_m, hc.translate, hc.tilt, hc.roll_deg)
        return sol

    # ------------------------------------------------------------------ template (§6.3 precomputation)

    def sample_template(self):
        self.template_meta["init_arms"] = f"left {IDLE_ARMS} (JoyLo reset), right hold"
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
        self.solve_hold()
        self.prior = load_visibility_prior(VISIBILITY_PRIOR_PATH, self.hold)
        if not self.prior:
            log.info("%s: no B7 visibility prior for this hold (roll %.0f deg); every library row is realised and measured", self.gp.alias, self.hold.roll_deg)
        self.precompute_options()
        self.template_facts.update({"hold": self.hold_solution, "grasp_key": self.grasp_key, "n_grasps": len(self.grasps),
                                    "orientation_equivalences": None if self.equivalences is None else len(self.equivalences),
                                    "visibility_prior_rows": len(self.prior)})
        self.template_meta["hold"] = self.hold_solution

    def check_symmetry(self):
        """§3.13 rule 1 hook (mental rotation only)."""

    def goal_candidates(self, cam: C.Pose, obj0: C.Pose) -> List[Tuple[C.Pose, Dict[str, Any]]]:
        """Goal object poses (world, at the reference base) with their descriptions — the per-subtask goal set."""
        raise NotImplementedError

    def key_poses(self, eef0: C.Pose, eef1: C.Pose, obj0: C.Pose) -> List[C.Pose]:
        """The §3.9 key poses of the in-hand motion eef0 -> eef1 (3 intermediate + the goal)."""
        raise NotImplementedError

    def precompute_options(self):
        """Per library grasp (B7 prior first): held object in frame at the hold, then every goal candidate whose key
        poses pass the screening IK (right arm only, object attached) with the object in frame at the goal. Expressed in
        the base frame so the options hold for any base pose; each instance confirms its draw with the exact IK."""
        target = self.obj("target")
        ref = self.kit.world_base_pose2d()
        base3 = self.base_pose3d(ref)
        cam = C.compose(base3, self.cam_in_base)
        eef0_w = C.compose(base3, self.eef_in_base)
        q_hold = self.q_init_at(ref)
        min_vis = float(self.g("min_visible_fraction"))
        stats = {"grasps": len(self.grasps), "prior_skipped": 0, "prior_unknown": 0, "held_out_of_view": 0, "candidates": 0, "goal_out_of_view": 0,
                 "ik_failed": 0, "feasible": 0}
        self.options = {}
        skipped = []
        for g in self.grasps:
            admits = prior_admits(self.prior, g.key, g.name, min_vis)
            if admits is False:
                stats["prior_skipped"] += 1
                skipped.append(g.name)
                continue
            if admits is None:
                stats["prior_unknown"] += 1
            obj0 = C.compose(eef0_w, C.invert(g.pose))
            if not self.whole_in_frame(cam, obj0):
                stats["held_out_of_view"] += 1
                continue
            entries = []
            for goal, info in self.goal_candidates(cam, obj0):
                stats["candidates"] += 1
                if not self.whole_in_frame(cam, goal):
                    stats["goal_out_of_view"] += 1
                    continue
                eef1 = C.compose(goal, g.pose)
                wps = self.key_poses(eef0_w, eef1, obj0)
                entries.append((goal, info, eef1, wps))
            if not entries:
                continue
            # the object at the hold in the simulation so the planner attaches it with the right transform
            self.kit.apply(q_hold, ref)
            target.set_position_orientation(obj0[0], obj0[1])
            target.keep_still()
            self.kit.sync_world()
            poses = [p for _, _, _, wps in entries for p in wps]
            # robot + held object only (the world is re-checked per instance at the sampled base pose)
            with self.kit.ignoring(self.scene_obstacles()), attached(self.kit, ARM, target):
                ok = self.kit.ik_batch(poses, q_hold, arm=ARM)
            n_wp = len(entries[0][3])
            ok = ok.view(len(entries), n_wp).all(dim=1)
            opts = []
            for (goal, info, eef1, wps), s in zip(entries, ok.tolist()):
                if not s:
                    stats["ik_failed"] += 1
                    continue
                opts.append({"goal_in_base": C.relative(goal, base3), "eef1_in_base": C.relative(eef1, base3),
                             "wps_in_base": [C.relative(p, base3) for p in wps], "info": info})
                stats["feasible"] += 1
            if opts:
                self.options[g.name] = opts
        self.world.park(target, list(self.world.spawned).index(target.name))
        self.kit.sync_world()
        self.template_facts["goal_options"] = {**stats, "grasps_with_options": len(self.options), "prior_skipped_names": skipped,
                                               "per_grasp": {k: len(v) for k, v in self.options.items()}}
        if not self.options:
            raise Reject("goal_infeasible", f"no library grasp has a visible hold with a feasible goal ({stats})")
        log.info("%s: %d grasps with goal options (%s)", self.gp.alias, len(self.options), stats)

    # ------------------------------------------------------------------ instance (§6.3)

    def sample_base(self, rng: np.random.Generator) -> Tuple[float, float, float]:
        """Random free cell of the eroded map inside the task room + yaw ~ U(−π, π), the hold configuration whole-body
        collision-free (2 cm; the object is checked once it is attached)."""
        pts = self.kit.random_free_points(rng, 30, room_filter=self.in_room)
        if not pts:
            raise Reject("no_stance", "no free eroded cell in the task room")
        cands = [(float(xy[0]), float(xy[1]), float(rng.uniform(-math.pi, math.pi))) for xy in pts]
        free = self.kit.collision_free(th.stack([self.q_init_at(b) for b in cands]), margin=0.02)
        for b, f in zip(cands, free.tolist()):
            if f:
                return b
        raise Reject("no_stance", "no collision-free base pose for the hold configuration")

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

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        tname = self.target_name
        target = self.obj("target")
        hc = self.hold
        base2d = self.sample_base(rng)
        base3 = self.base_pose3d(base2d)
        q_init = self.q_init_at(base2d)
        cam = self.camera_at(base2d)
        eef0 = C.compose(base3, self.eef_in_base)
        init = Stance("init", base2d, hc.translate, hc.tilt, arms=[ARM], carried=tname)
        init.q = q_init
        init.extras.update({"posture": {"left_arm": "reset", "right_arm": "hold"}, "hold": self.hold.to_json()})
        n_wp = int(self.g("waypoints"))
        chosen = None
        errors = []
        for gname in self.grasp_order(rng)[:4]:
            g = next(x for x in self.grasps if x.name == gname)
            obj0 = C.compose(eef0, C.invert(g.pose))
            self.kit.apply(q_init, base2d)
            target.set_position_orientation(obj0[0], obj0[1])
            target.keep_still()
            self.kit.sync_world()
            with attached(self.kit, ARM, target):
                if not bool(self.kit.collision_free(q_init, margin=0.01)[0]):
                    errors.append(f"{gname}: hold collides with the object attached")
                    continue
                opts = self.options[gname]
                for oi in rng.permutation(len(opts))[:4]:
                    opt = opts[int(oi)]
                    wps = [C.compose(base3, p) for p in opt["wps_in_base"]]
                    if len(wps) != n_wp + 1:
                        raise Reject("spec", f"{len(wps) - 1} intermediate waypoints precomputed, generation.waypoints = {n_wp}")
                    res = self.ikc.waypoints_ok([eef0] + wps, init, ARM, q_seed=q_init, scale_key=self.choice.models["target"].scale)
                    if not res.ok:
                        errors.append(f"{gname}/{oi}: {res.reason}")
                        continue
                    goal = C.compose(base3, opt["goal_in_base"])
                    if not self.whole_in_frame(cam, goal):
                        errors.append(f"{gname}/{oi}: goal out of frame")
                        continue
                    chosen = (g, obj0, goal, dict(opt["info"]), res.q, wps)
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise Reject("goal_infeasible", "no precomputed (grasp, goal) confirmed at this base pose: " + "; ".join(errors[-4:]))
        g, obj0, goal, info, q_goal, wps = chosen
        info = self.describe_goal(info, base2d)
        d.object_poses[tname] = obj0
        d.goal_poses[tname] = goal
        d.robot_q = q_init
        d.base_pose2d = base2d
        d.settle_free = []
        obj_in_eef = C.invert(g.pose)
        d.extras["grasps"] = [{"agent": AGENT_INSTANCE, "arm": ARM, "object": tname, "grasp_key": g.key, "grasp_name": g.name,
                               "object_pose_in_eef_frame": {"position": obj_in_eef[0].tolist(), "orientation": obj_in_eef[1].tolist()}}]
        d.extras["goal_robot_q"] = q_goal
        d.summary.update({"grasp": {"key": g.key, "name": g.name, "flags": list(g.flags)}, "goal": info, "feasibility": {"embodiment": f"{ARM}_arm"},
                          "hold": self.hold.to_json()})
        d.pose_targets[tname] = self.pose_target_for(cam, goal, info)
        # sidecar: init stance (held), the grasp with the in-hand key poses, the goal arm configuration, language
        self.sidecar.add_stance(init)
        rec = GraspRecord(tname, ARM, g.key, g.name, 0.0, pre_grasp=IK.pre_pose(eef0), grasp=eef0, carry_height=0.0, finger_q=list(g.finger_q),
                          eef_in_object=g.pose, stance="init", waypoints=wps)
        self.sidecar.add_grasp(rec)
        goal_block = self.sidecar.set_goal(base2d, hc.translate, hc.tilt, arms="same")
        goal_block.update({"right_arm_q": q_goal[self.robot.arm_control_idx[ARM]].tolist(), "embodiment": f"{ARM}_arm",
                           "eef": {"position": wps[-1][0].tolist(), "orientation": wps[-1][1].tolist()}, "hold": self.hold.to_json()})
        self.sidecar.describe(tname, self.choice.models["target"].category.replace("_", " "), **self.goal_numbers(info))
        return d

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
        eef = tuple(x.float() for x in self.robot.eef_links[ARM].get_position_orientation())
        rel = C.relative((p.float(), q.float()), eef)
        draft.extras["grasps"][0]["object_pose_in_eef_frame"] = {"position": rel[0].tolist(), "orientation": rel[1].tolist()}
        finger_q = self.robot.get_joint_positions()[self.robot.gripper_control_idx[ARM]].tolist()
        draft.extras["grasps"][0]["finger_q"] = finger_q
        q_goal = G.t(draft.extras["goal_robot_q"])
        eef_goal = self.kit.eef_pose(q_goal, ARM)
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
        """Gripper joint indices of ``ARM`` paired with the finger link each one moves (``JointPrim.body1``)."""
        gidx = [int(i) for i in self.robot.gripper_control_idx[ARM].tolist()]
        names = list(self.robot.joints.keys())
        finger_paths = {l.prim_path for l in self.robot.finger_links[ARM]}
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
            _, links = self.robot._find_gripper_contacts(ARM)
            touched = set()
            for p in target_paths:
                touched |= set(links.get(p, ()))
            return [path in touched for _, path in pairs]

        q = hi.clone()
        state.gripper_cmd[ARM] = cmd_for(q)
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
            state.gripper_cmd[ARM] = cmd_for(q)
            target.keep_still()
            self.world.control_step(state.action())
            contact = touching()
        if not all(contact):
            raise Reject("grasp_failed", f"pads not both in contact after closing (contact {contact})")
        q_contact = q.clone()
        press = float(self.g("grasp_press_m"))
        state.gripper_cmd[ARM] = cmd_for(th.clamp(q_contact - press, min=lo))
        attached_ok = False
        for _ in range(GRASP_ATTACH_STEPS):
            if self.robot._ag_obj_in_hand[ARM] is None:
                target.set_position_orientation(G.t(pose[0]), G.t(pose[1]))
                target.keep_still()
            self.world.control_step(state.action())
            if self.robot._ag_obj_in_hand[ARM] is target:
                attached_ok = True
                break
        if not attached_ok:
            raise Reject("grasp_failed", "assisted grasping did not attach the object (both pads in contact)")

        def offset_now():
            eef = tuple(x.float() for x in self.robot.eef_links[ARM].get_position_orientation())
            return C.relative(tuple(x.float() for x in target.get_position_orientation()), eef)

        eef0 = tuple(x.float() for x in self.robot.eef_links[ARM].get_position_orientation())
        lib_rel = C.relative(pose, eef0)
        rel_attach = offset_now()
        self.robot.set_joint_positions(q_contact, indices=gidx, drive=False)
        state.gripper_cmd[ARM] = cmd_for(q_contact)
        target.enable_gravity()
        self.world.control_step(state.action(), n=GRASP_GRAVITY_STEPS)
        if self.robot._ag_obj_in_hand[ARM] is not target:
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
            if self.robot._ag_obj_in_hand.get(ARM) is not None:
                self.robot.release_grasp_immediately(ARM)
        except Exception:
            pass
        state = ActionState(self.robot)
        state.hold_current()
        state.set_gripper(ARM, close=False)
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
        """§4.6 step 5 for a stance that *holds* the object: the realised init configuration (hold arm, reset left arm)
        is checked with the target attached to the planner — the core's generic check locks the arms untucked and keeps
        the object as a world obstacle, which puts the untucked right hand into the held object. TEMP until the core
        honours ``Stance.q`` / ``Stance.carried``."""
        self.kit.sync_world()
        with attached(self.kit, ARM, self.obj("target")):
            if not bool(self.kit.collision_free(G.t(draft.robot_q), margin=0.01)[0]):
                raise Reject("stance_collision_with_objects", "the hold configuration collides with the world at the init base pose (object attached)")

    def program_check(self, draft: Draft) -> Dict[str, Any]:
        """§3.5 with the held object attached: the program's first action (``Reorient``) already carries the target
        (``stances.init.carried``), no ``Grasp`` precedes it, so the object is attached to the planner for the whole
        check (the realised in-hand pose is what the simulator holds at this point). TEMP until the core's program
        check seeds its carried-object tracker from ``stances.init.carried``."""
        with attached(self.kit, ARM, self.obj("target")):
            return super().program_check(draft)

    # ------------------------------------------------------------------ writing / constraints

    def write_instance(self, draft: Draft, index: int, seed: int) -> str:
        path = super().write_instance(draft, index, seed)
        self.release()
        target = self.obj("target")
        self.world.park(target, list(self.world.spawned).index(target.name))
        og.sim.step_physics()
        self.kit.sync_world()
        return path

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        """The ``hold`` constant (once per entry) and the confirmed (grasp, goal) as a ``grasp_set`` prior."""
        recs: List[Dict[str, Any]] = []
        sol = self.hold_solution
        have = [r for r in self.gp.constraints_for("hold") if r.get("robot") == sol.get("robot") and r.get("depth_m") == sol.get("depth_m")
                and list(r.get("trunk", [])) == list(sol.get("trunk", [])) and r.get("roll_deg") == sol.get("roll_deg")]
        if not have:
            recs.append({"scope": {}, "type": "hold", "robot": sol["robot"], "depth_m": sol["depth_m"], "trunk": list(sol["trunk"]), "roll_deg": sol["roll_deg"],
                         "approach": sol["approach"], "closing": sol["closing"], "eef_in_base": sol["eef_in_base"], "right_arm_q": sol["right_arm_q"],
                         "idle_arms": sol["idle_arms"], "hard": False})
        g = draft.summary.get("grasp") or {}
        choice = self.choice.models["target"]
        recs.append({"scope": {"target": f"{choice.category}/{choice.model}"}, "type": "grasp_set", "trunk": [self.hold.translate, self.hold.tilt],
                     "hold": True, "feasible": [[0.0, g.get("name")]], "goal": draft.summary.get("goal"), "hard": False})
        return recs


# ----------------------------------------------------------------------------------------------------------------
# object scaling
# ----------------------------------------------------------------------------------------------------------------


@register("object_scaling")
class ObjectScalingV2(HeldObjectV2):
    """§6.3 goal: depth d1 on the 5 cm grid over ``goal_depth_range_m`` with |d1 − d0| >= ``min_depth_change_m``,
    orientation unchanged; target frame = camera sensor, ``position_metric = depth``, tolerance max(0.02, 0.20·|Δd|)."""

    def goal_candidates(self, cam: C.Pose, obj0: C.Pose) -> List[Tuple[C.Pose, Dict[str, Any]]]:
        axis = self.world.camera_model.optical_axis(cam)
        d0 = float(th.dot(G.t(obj0[0]) - G.t(cam[0]), axis))
        depths = depth_goal_set(d0, self.g("goal_depth_range_m"), float(self.g("goal_depth_grid_m")), float(self.g("min_depth_change_m")))
        out = []
        for d1 in depths:
            goal = (G.t(obj0[0]) + axis * (d1 - d0), G.t(obj0[1]).clone())
            out.append((goal, {"init_depth_m": d0, "goal_depth_m": d1, "depth_change_m": d1 - d0}))
        return out

    def key_poses(self, eef0: C.Pose, eef1: C.Pose, obj0: C.Pose) -> List[C.Pose]:
        return line_key_poses(eef0, eef1, int(self.g("waypoints")))

    def pose_target_for(self, cam: C.Pose, goal: C.Pose, info: Dict[str, Any]) -> Dict[str, Any]:
        p_cam = C.relative((G.t(goal[0]), C.IDENTITY_QUAT), cam)[0]  # goal root position in the camera frame
        tol = depth_tolerance(info["depth_change_m"], dict(self.g("tolerances")))
        return pose_target(position=p_cam.tolist(), position_metric="depth", position_tolerance=tol, frame=f"{AGENT_INSTANCE}::{self.world.cam_key}")

    def goal_numbers(self, info: Dict[str, Any]) -> Dict[str, float]:
        return {"depth_change_cm": 100.0 * float(info["depth_change_m"]), "goal_depth_cm": 100.0 * float(info["goal_depth_m"])}


# ----------------------------------------------------------------------------------------------------------------
# mental rotation
# ----------------------------------------------------------------------------------------------------------------


@register("mental_rotation")
class MentalRotationV2(HeldObjectV2):
    """§6.3 goal: rotation about a sampled axis (13 base-frame axes × the signed 15° grid inside
    ``goal_angle_range_deg``) about the object root, position unchanged; world orientation target, 0.15 rad,
    ``orientation_equivalences`` per model (§3.13); symmetric models are rejected (``symmetric_target``)."""

    def check_symmetry(self):
        if self.equivalences is not None and len(self.equivalences) > 1 and bool(self.g("reject_symmetric_targets", True)):
            c = self.choice.models["target"]
            raise Reject("symmetric_target", f"{c.category}/{c.model} has a rotational symmetry ({len(self.equivalences)} equivalent orientations); §3.13 rule 1")

    def goal_candidates(self, cam: C.Pose, obj0: C.Pose) -> List[Tuple[C.Pose, Dict[str, Any]]]:
        base_q = C.quat_from_yaw(self.kit.world_base_pose2d()[2])
        out = []
        for axis_base, angle_deg in rotation_goal_set(self.g("goal_angle_range_deg"), ROTATION_ANGLE_STEP_DEG):
            axis_w = C.quat_rotate(base_q, axis_base)
            dq = C.quat_from_axis_angle(axis_w, math.radians(angle_deg))
            goal = (G.t(obj0[0]).clone(), C.qmul(dq, G.t(obj0[1])))
            out.append((goal, {"axis_base": axis_base.tolist(), "angle_deg": float(angle_deg)}))
        return out

    def key_poses(self, eef0: C.Pose, eef1: C.Pose, obj0: C.Pose) -> List[C.Pose]:
        # the rotation that carries the eef from eef0 to eef1 about the object root (the object turns in place)
        dq = C.qmul(G.t(eef1[1]), C.qinv(G.t(eef0[1])))
        return orbit_key_poses(eef0, G.t(obj0[0]), dq, int(self.g("waypoints")))

    def describe_goal(self, info: Dict[str, Any], base2d) -> Dict[str, Any]:
        info = dict(info)
        info["axis_world"] = C.quat_rotate(C.quat_from_yaw(base2d[2]), th.tensor(info["axis_base"], dtype=th.float32)).tolist()
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
