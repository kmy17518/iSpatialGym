"""IK conventions of the v2 sampler (spec §3.6, §3.9, §3.10) — the only place cuRobo IK is called from v2.

* **Screening** = 12-seed batched IK per pose (``RobotKit.ik_batch``); **exact** = 32-seed IK (``RobotKit.ik``).
* **Locked joints** = the untucked init configuration at the stance's trunk (``q_locked``); the idle arm is untucked and
  part of the collision body; the working arm's solution is what the oracle tracks.
* **"Support ignored for the grasp pose only"**: ``ignore`` objects leave the cuRobo world for the *last* pose of a
  sequence (grasp / contact) and for nothing else — every other object (partner, distractors, scene) stays.
* **Waypoints** (§3.9): ``waypoints_ok`` screens the intermediate poses and confirms the endpoints exactly; the
  ``policy`` string is what the sidecar records as ``waypoint_policy``.
* **eef in frame** (§3.9): the eef body (gripper + fingertips) at the final pose projects inside the image with 5 %.
* **Bands' feasible sets F** (§4.6 / §3.10): ``feasible_sets`` screens (yaw, grasp) pairs over cells in one batch per
  trunk and returns F per cell.
* **Exact-IK cache** of v1 (``ISPATIALGYM_SAMPLING_CACHE``): every key includes the spawn scale vector of the object.

Geometry (grasp / pre / contact poses, waypoints) is ``ispatialgym.robot.conventions`` (F5, commit ``dc91cfb``); the
thin wrappers below only fix the argument shapes the task modules use (tensor pairs in, tensor pairs out) — v2 never
re-implements a conventions function (README contract 2).
"""

from __future__ import annotations

import logging
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.fastflags import fast_on, fast_value  # train-fast 2026-10-31: ISPATIALGYM_FAST_* switches (off = the former behaviour)
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.cache import cache_get, cache_key, cache_put
from ispatialgym.sampling.v2.stances import Pose, Stance

log = logging.getLogger("ispatialgym.sampling.v2")

SCREEN_SEEDS = 12
EXACT_SEEDS = 32
PRE_DIST = C.PRE_DIST
HANDLE_OFFSET = C.HANDLE_EEF_OFFSET
IN_FRAME_MARGIN = 0.05
FINGER_OPEN_DEFAULT = C.FINGER_OPEN
FINGER_CLOSED = C.FINGER_CLOSED
#: a segment between consecutive chain poses counts as a straight *approach* (tracked from the previous solution) when
#: the orientation is the same within this angle (rad) and the distance at most this (m): pre -> grasp / place /
#: contact (``PRE_DIST`` 0.10), the wall brace chain, lifts and descents recorded as pose lists
APPROACH_SAME_ORIENT_RAD = 0.05
APPROACH_MAX_M = 0.25


def is_approach(a: Pose, b: Pose) -> bool:
    """Whether ``a -> b`` is a straight same-orientation segment of at most ``APPROACH_MAX_M`` (a pre-pose to its end pose)."""
    d = float(th.linalg.norm(G.t(b[0]) - G.t(a[0])))
    return 1e-4 < d <= APPROACH_MAX_M and G.quat_angle(G.t(a[1]), G.t(b[1])) <= APPROACH_SAME_ORIENT_RAD

# ----------------------------------------------------------------------------------------------------------------
# conventions (F5) — argument-shape wrappers
# ----------------------------------------------------------------------------------------------------------------


def _pair(pose) -> Pose:
    p, q = C.as_pose(pose)
    return G.t(p), G.t(q)


def approach_axis(pose: Pose) -> th.Tensor:
    """a = eef +z in world (conventions ``approach_axis``)."""
    return C.approach_axis(G.t(pose[1]))


def pre_pose(pose: Pose, dist: float = PRE_DIST) -> Pose:
    """Pre-grasp / pre-contact / pre-place: the pose moved ``dist`` back along -a (spec §3.6; conventions ``pre_pose``)."""
    return _pair(C.pre_pose((G.t(pose[0]), G.t(pose[1])), dist))


def grasp_pose(obj_pose: Pose, eef_in_object: Pose) -> Pose:
    """eef pose = object pose ∘ T_obj→eef (conventions ``grasp_pose_from_transform``; the grasp-library entry gives
    T_obj→eef; the recorded ``grasps[].yaw`` is the object yaw, so consumers re-deriving from the settled object pose
    call ``conventions.grasp_pose(obj_pose, key, name)`` with ``yaw = 0``)."""
    return _pair(C.grasp_pose_from_transform((G.t(obj_pose[0]), G.t(obj_pose[1])), (G.t(eef_in_object[0]), G.t(eef_in_object[1]))))


def contact_eef(point, approach_into_object, closing=None) -> Pose:
    """Push / brace contact: pad centre on ``point``, approach a along ``approach_into_object`` (= -outward normal),
    pads closing along ``closing`` (default horizontal ⊥ a), eef origin = point − CONTACT_CENTER_Z·a (spec §3.6;
    conventions ``contact_eef``)."""
    return _pair(C.contact_eef(G.t(point), G.t(approach_into_object), None if closing is None else G.t(closing)))


def handle_eef(contact, approach, closing=None) -> Pose:
    """Handle grasp: approach a along ``approach`` (into the handle), pads closing along ``closing`` (default: horizontal
    ⊥ a), eef origin = contact − 0.012·a (spec §3.6; conventions ``handle_eef``)."""
    a = C.unit(G.t(approach))
    return _pair(C.handle_eef(G.t(contact), a, C._default_closing(a) if closing is None else G.t(closing)))


def line_waypoints(p0: Pose, p1: Pose, n: int = 3, endpoints: bool = False) -> List[Pose]:
    """Straight eef motion p0 -> p1 at fractions k/n (start excluded, goal last; ``endpoints=True`` prepends the start),
    orientation SLERPed (conventions ``line_waypoints``)."""
    return [_pair(p) for p in C.line_waypoints((G.t(p0[0]), G.t(p0[1])), (G.t(p1[0]), G.t(p1[1])), n, endpoints)]


# ----------------------------------------------------------------------------------------------------------------
# stance arm postures (A6)
# ----------------------------------------------------------------------------------------------------------------

POSTURES = {"untucked": "untucked", "tucked": "tucked", "reset": "tucked", "ready": "ready"}


def stance_arm_vectors(kit, stance: Stance, arms: Optional[str] = None, right_arm: Optional[th.Tensor] = None, left_arm: Optional[th.Tensor] = None,
                       fingers: Optional[float] = None) -> Tuple[th.Tensor, th.Tensor, float]:
    """``(right_arm, left_arm, fingers)`` of a stance, in this order of precedence: the explicit arguments, the stance's
    **recorded configuration** ``stance.q`` (§3.3 exceptions: the held-object hold, the camera-relocalization reset arms)
    and ``stance.fingers``, the arm joint values a written sidecar recorded (``arms_q``, a stance loaded by
    ``Stance.from_sidecar``: what the sampler's checks used, so ``validate_instances`` checks the same arms), then per
    arm: the working arms (``stance.arms`` list) untucked, the idle arm in ``stance.idle_posture`` (A6: untucked /
    ready / tucked), a posture string ``arms`` / ``stance.arms`` for both."""
    robot = kit.robot
    if stance.q is not None:
        if right_arm is None:
            right_arm = stance.q[robot.arm_control_idx["right"]].clone()
        if left_arm is None:
            left_arm = stance.q[robot.arm_control_idx["left"]].clone()
        if fingers is None and stance.fingers is None:
            w = stance.working_arms
            fingers = float(stance.q[robot.gripper_control_idx[w[0] if len(w) == 1 else "right"]][0])
    recorded = (stance.extras or {}).get("arms_q") if stance.q is None else None
    if recorded:
        if right_arm is None and recorded.get("right") is not None:
            right_arm = th.tensor(recorded["right"], dtype=th.float32)
        if left_arm is None and recorded.get("left") is not None:
            left_arm = th.tensor(recorded["left"], dtype=th.float32)
    if fingers is None:
        fingers = stance.fingers if stance.fingers is not None else FINGER_OPEN_DEFAULT
    both = POSTURES.get(arms or stance.posture, "untucked")
    working = stance.working_arms
    for arm in ("right", "left"):
        cur = right_arm if arm == "right" else left_arm
        if cur is not None:
            continue
        name = POSTURES.get(stance.idle_posture, "untucked") if (working and arm not in working and stance.idle_posture) else both
        vec = kit.arm_q(name, arm)
        if arm == "right":
            right_arm = vec
        else:
            left_arm = vec
    return right_arm, left_arm, float(fingers)


# ----------------------------------------------------------------------------------------------------------------
# IK checker
# ----------------------------------------------------------------------------------------------------------------


class _Ignoring:
    """Switch the collision meshes of ``objs`` off in the cuRobo world for the block; meshes disabled for another
    reason (attached objects, removed objects) are left alone. Falls back to ``RobotKit.ignoring`` (two world syncs)
    when the backend has no per-mesh switch."""

    def __init__(self, kit, objs: List[Any], ground: bool = True, ground_meshes: bool = True):
        self.kit = kit
        self.objs = objs
        self.ground = ground  # False: the cuRobo ground slab (``R1ProCuRobo.GROUND_SLAB_NAME``) is switched off too (floor-level pushes)
        self.ground_meshes = ground_meshes  # False (opt-in, owner 2026-09-25): the ground-like scene meshes (floors, lawn, carpets) off as well
        self.names: List[str] = []
        self._fallback = None
        self._slab_off = False

    def _slab(self, enabled: bool) -> bool:
        backend = self.kit.backend
        cub = getattr(getattr(getattr(backend, "checker", None), "data", None), "cuboids", None)
        name = getattr(backend, "GROUND_SLAB_NAME", "__ground__")
        try:
            if cub is not None and name in cub.names[0]:
                cub.set_enabled(name, enabled)
                return True
        except Exception:  # noqa: BLE001 - a backend without the slab
            return False
        return False

    def __enter__(self):
        if not self.ground:
            self._slab_off = self._slab(False)
        if not self.ground_meshes:
            self._ground_meshes(False)
        if not self.objs:
            return self
        backend = self.kit.backend
        if not (hasattr(backend, "_mesh_enable") and hasattr(backend, "_world_mesh_names")):
            self._fallback = self.kit.ignoring(self.objs)
            self._fallback.__enter__()
            return self
        self.kit._ensure_synced()
        disabled = getattr(backend, "_disabled_obstacles", set())
        active = _ACTIVE_IGNORES.setdefault(id(backend), {})
        for o in self.objs:
            for link in getattr(o, "links", {}).values():
                for cm in getattr(link, "collision_meshes", {}).values():
                    name = cm.prim_path
                    if name in backend._world_mesh_names and name not in disabled:
                        # nested contexts may ignore the same mesh: it is switched off once and back on when the last
                        # context that asked for it exits (refcount); the backend's ``_held_off`` set keeps its own ground
                        # re-enable (check_collision at margin > 0) from switching it back on meanwhile (owner 2026-09-25)
                        if active.get(name, 0) == 0:
                            backend._mesh_enable(name, False)
                        active[name] = active.get(name, 0) + 1
                        held = getattr(backend, "_held_off", None)
                        if held is None:
                            held = backend._held_off = set()
                        held.add(name)
                        self.names.append(name)
        return self

    def _ground_meshes(self, enabled: bool) -> None:
        """``ground=False`` also switches the GROUND-LIKE scene meshes off (the planner's ``_ground_mesh_names``: floors, carpets,
        the lawn, ...; owner 2026-09-25, shampoo 0925-194056: with the floor ignored and the slab off, the floor-class lift still
        collided 'with the scene' - the lawn mesh spans the house at floor level) and back on at exit; meshes another active
        ignore context holds off, or the backend disabled for its own reasons, are left to them."""
        backend = self.kit.backend
        names = getattr(backend, "_ground_mesh_names", None)
        if not names or not hasattr(backend, "_mesh_enable"):
            return
        active = _ACTIVE_IGNORES.setdefault(id(backend), {})
        disabled = getattr(backend, "_disabled_obstacles", set())
        if not enabled:
            self._ground_off = []
            held = getattr(backend, "_held_off", None)
            if held is None:
                held = backend._held_off = set()
            for name in list(names):
                if name in getattr(backend, "_world_mesh_names", set()) and name not in disabled and active.get(name, 0) == 0 and name not in held:
                    backend._mesh_enable(name, False)
                    held.add(name)
                    self._ground_off.append(name)
        else:
            held = getattr(backend, "_held_off", set())
            for name in getattr(self, "_ground_off", []):
                held.discard(name)
                if active.get(name, 0) == 0 and name not in disabled:
                    backend._mesh_enable(name, True)
            self._ground_off = []

    def __exit__(self, *exc):
        if self._slab_off:
            self._slab(True)
            self._slab_off = False
        if not self.ground_meshes:
            self._ground_meshes(True)
        if self._fallback is not None:
            self._fallback.__exit__(*exc)
            return False
        active = _ACTIVE_IGNORES.setdefault(id(self.kit.backend), {})
        held = getattr(self.kit.backend, "_held_off", None)
        for name in self.names:
            active[name] = active.get(name, 1) - 1
            if active[name] <= 0:
                active.pop(name, None)
                if held is not None:
                    held.discard(name)
                self.kit.backend._mesh_enable(name, True)
        self.names = []
        return False


#: per backend: collision meshes currently switched off by active ``_Ignoring`` contexts -> number of contexts asking
_ACTIVE_IGNORES: Dict[int, Dict[str, int]] = {}

#: the oracle's pre-shape margin (``MotionsConfig.gripper_preshape_margin``): ``execute_grasp`` opens each finger to pinch width / 2 + this
#: before the approach
ORACLE_PRESHAPE_MARGIN_M = 0.010
#: ``approach_finger_support_touch.max_penetration_m`` above this is a spec error (light contact, not a finger through the support)
FINGER_TOUCH_MAX_PENETRATION_M = 0.03


def support_policy_on() -> bool:
    """``ISPATIALGYM_SUPPORT_POLICY=1``: the grasp pose's support ignore is followed by the shared contact policy (:meth:`IKChecker.support_policy_ok`)."""
    return os.environ.get("ISPATIALGYM_SUPPORT_POLICY", "") not in ("", "0", "off", "false")


def registered_supports_only() -> bool:
    """of-v2maps 2026-11-07 (owner prompt "supports with protrusions": what counts as the support plane, the policy unchanged): the support
    policy polices only the supports whose plane the sampler REGISTERED (``IKChecker.support_plane_z``: a map hit's measured top, a tray floor,
    the floor) - an ignored object without one (a wall braced or pushed on, a door / dishwasher whose handle is grasped) is not a horizontal
    support: its AABB top as the plane put every hand pose of world_vertical / articulation_large 0.06-1.5 m "below the support"
    (v2only_before: push_unreachable / handle_unreachable on 11 / 11 aliases; i_nopol with the policy off 4 / 5 and 3 / 6 succeed).
    ``ISPATIALGYM_SUPPORT_POLICY_REGISTERED=0``: the AABB-top fallback for unregistered supports as before."""
    return os.environ.get("ISPATIALGYM_SUPPORT_POLICY_REGISTERED", "1") != "0"


def policed_supports(supports: Sequence[Any], approached: Optional[Sequence[Any]], planes: Optional[Dict[str, float]] = None,
                     registered_only: Optional[bool] = None) -> List[Any]:
    """of-v2maps 2026-11-06: the ignored objects the support policy polices - those the chain does NOT approach (the support under a grasped
    object); an object the hand is meant to touch (``approached``: the grasped target, a pushed box) is not a support of the approach.
    2026-11-07: with :func:`registered_supports_only` (default) only the supports with a registered plane (``planes``) - pass ``planes`` to apply it."""
    out = [sp for sp in supports if all(sp is not a for a in (approached or ()))]
    ro = registered_supports_only() if registered_only is None else registered_only
    if ro and planes is not None:
        out = [sp for sp in out if getattr(sp, "name", None) in planes]
    return out


def floor_support_names(ikc: Any) -> set:
    """i7-floor 2026-11-07: the names of the supports the sampler registered as the FLOOR the robot stands on (``IKChecker.support_floor_names``,
    set by ``tabletop.floor_map_bands``) - policed in the contact policy's floor mode (``contact_policy`` rule 5)."""
    v = getattr(ikc, "support_floor_names", None)
    return set(v) if v else set()


def policy_links(others: Sequence[str], supports: Sequence[Any], floors: set, ground_links: Sequence[str]) -> List[str]:
    """i7-floor 2026-11-07: the non-finger links the sphere stage of the support policy checks - all of ``others``, but the robot's
    ground-contact links (wheels, base body: ``R1ProCuRobo.ground_contact_links``) when a policed support is a registered floor (rule 5:
    they stand on it - not a grasp contact; the IK of the pose still checked them against every other obstacle)."""
    if not any(getattr(sp, "name", None) in floors for sp in supports):
        return list(others)
    gc = set(ground_links or ())
    return [l for l in others if l not in gc]


def support_plane_z(sp: Any, planes: Optional[Dict[str, float]]) -> Optional[float]:
    """of-v2maps 2026-11-06: the support policy's top plane of ``sp`` - the MEASURED top the sampler registered for it
    (``IKChecker.support_plane_z``, set by ``tabletop.map_bands``: the height the reach map was computed at), else its AABB top (the
    2026-11-02 rule: a table whose box rises above its working surface - a lip, a back panel - read as finger penetration for a flat
    object's grasp: v2scope1 inside_03 / nextto_03 / ontop_small_04 lost 499-772 grasps each to 'finger_penetration'); None without either."""
    name = getattr(sp, "name", None)
    if planes and name in planes:
        return float(planes[name])
    try:
        return float(sp.aabb[1][2])
    except Exception:  # noqa: BLE001
        return None


def finger_link_clearance(hc: Any, T_eef: th.Tensor, plane: Any, finger_q: Tuple[float, float]) -> Dict[str, float]:
    """of-v2maps 2026-11-06 (the support-policy diagnostic): the minimum signed clearance (m) of each finger link's collision hulls to
    ``plane`` at the eef pose ``T_eef`` (4x4), with the hull points of ``contact_policy.HandChecker.points`` (vertices + face samples at
    the checker's spacing, each finger shifted along its axis by its opening)."""
    import numpy as np

    from ispatialgym.grasping import hand_geometry as HG

    return {link: float(plane.sdf(W[None])[0].min()) for link, W in finger_points_world(hc, T_eef, finger_q).items()}


def finger_points_world(hc: Any, T_eef: th.Tensor, finger_q: Tuple[float, float]) -> Dict[str, th.Tensor]:
    """of-v2maps 2026-11-06: each finger link's collision-hull points (vertices + face samples at the checker's spacing, the finger shifted
    along its axis by its opening) in the world at the eef pose ``T_eef`` (4x4) - the points ``contact_policy.HandChecker`` measures."""
    import numpy as np

    from ispatialgym.grasping import hand_geometry as HG

    out: Dict[str, th.Tensor] = {}
    T = th.as_tensor(T_eef, dtype=th.float32)
    for lk in hc.geom.links:
        if lk.role != "finger":
            continue
        qf = finger_q[0] if lk.link.endswith("1") else finger_q[1]
        s = np.concatenate([HG._hull_surface(lk.link, pc.tobytes(), hc.spacing) for pc in (lk.pieces or [lk.points])])
        p = s + (float(qf) * lk.axis[None] if lk.axis is not None else 0.0)
        out[lk.link] = th.as_tensor(p, dtype=th.float32) @ T[:3, :3].T + T[:3, 3]
    return out


def pad_overlap_m(finger_pts: Dict[str, th.Tensor], obj_pts: th.Tensor, approach: th.Tensor) -> float:
    """of-v2maps 2026-11-06 (the raised-grasp fallback): how far the fingers and the object overlap along the approach axis ``approach`` -
    min(finger tip extent, object far side) - max(finger root extent, object near side), the smaller over the two fingers (a pinch needs
    both); negative = the fingers end before the object begins."""
    a = th.as_tensor(approach, dtype=th.float32)
    o = th.as_tensor(obj_pts, dtype=th.float32) @ a
    best = []
    for W in finger_pts.values():
        f = W @ a
        best.append(float(min(float(f.max()), float(o.max())) - max(float(f.min()), float(o.min()))))
    return min(best) if best else float("-inf")


def support_policy_scope() -> str:
    """of-v2maps 2026-11-06: which exact-IK chains the support policy polices - ``"grasp"`` (``ISPATIALGYM_SUPPORT_POLICY=grasp``, what
    ``tabletop.map_bands`` turns on): only chains whose arm holds nothing (the empty hand closing on an object at rest on its support - the
    pose the policy is written for); ``"all"`` (``=1`` / any other on value, the 2026-11-02 behaviour): every chain with an
    ``ignore_for_last`` - places, carries and releases included (v2maps1: 62 / 66 touching_02 attempts died at the place screen on
    'non-finger link touches the support', ontop_small_03 25 x 'palm_contact on partner_soap_dish' at the place onto the partner;
    with the policy off the same aliases wrote 4 / 7, 4 / 10, 4 / 9)."""
    return "grasp" if os.environ.get("ISPATIALGYM_SUPPORT_POLICY", "") == "grasp" else "all"


def _quat_mat(q: th.Tensor) -> th.Tensor:
    x, y, z, w = q.tolist()
    return th.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@dataclass
class FingerTouch:
    """``generation.approach_finger_support_touch`` (owner 2026-10-13 ~03:15 item 3): along a tracked grasp approach the acting arm's
    finger links may touch the grasp's support - their spheres may come within ``margin_m - max_penetration_m`` of it (light contact);
    every other link, and the finger links against every other object, keep the approach tracking's margin (0). ``preshape``: the finger
    links are checked at the opening the oracle pre-shapes to before the approach (``opening()``; None = the configuration's own)."""

    margin_m: float = 0.0
    max_penetration_m: float = 0.005
    preshape: bool = True
    supports: Callable[[], Sequence[Any]] = lambda: ()
    opening: Callable[[], Optional[float]] = lambda: None


def finger_touch_params(value: Any) -> Optional[Dict[str, Any]]:
    """``generation.approach_finger_support_touch`` -> ``{"margin_m", "max_penetration_m", "preshape"}``, None when off (absent / null /
    false); ``true`` = the defaults of :class:`FingerTouch`. ``ValueError`` for anything else (the caller's ``spec`` rejection)."""
    if value is None or value is False:
        return None
    d = FingerTouch()
    out = {"margin_m": d.margin_m, "max_penetration_m": d.max_penetration_m, "preshape": d.preshape}
    if value is True:
        return out
    if not isinstance(value, dict) or set(value) - set(out):
        raise ValueError(f"approach_finger_support_touch {value!r}: expected true or an object with {sorted(out)}")
    for k in ("margin_m", "max_penetration_m"):
        if k in value:
            v = value[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError(f"approach_finger_support_touch.{k} {v!r}: expected a number (m)")
            out[k] = float(v)
    if "preshape" in value:
        if not isinstance(value["preshape"], bool):
            raise ValueError(f"approach_finger_support_touch.preshape {value['preshape']!r}: expected true / false")
        out["preshape"] = value["preshape"]
    if out["margin_m"] < 0.0 or not 0.0 <= out["max_penetration_m"] <= FINGER_TOUCH_MAX_PENETRATION_M:
        raise ValueError(f"approach_finger_support_touch {value!r}: margin_m >= 0 and 0 <= max_penetration_m <= {FINGER_TOUCH_MAX_PENETRATION_M} expected")
    return out


@dataclass
class IKResult:
    ok: bool
    q: Optional[th.Tensor] = None
    reason: str = ""
    n_checked: int = 0

    @property
    def code(self) -> str:
        """Distinct rejection code carried by the reason (``approach_untrackable``: a straight approach whose end pose has
        no configuration in the pre pose's IK branch), else ``""`` — callers use ``res.code or "<their reason>"``."""
        if self.reason.startswith("approach_untrackable"):
            return "approach_untrackable"
        if self.reason.startswith("ik_joint_limit"):
            return "ik_joint_limit"
        if self.reason.startswith("ik_reach"):
            return "ik_reach"
        return ""



def with_policy_scope(scope: Dict[str, Any]) -> Dict[str, Any]:
    """of-v2maps 2026-11-07 (P2 §11: witnesses computed by ignoring the whole support must not satisfy the policy): with the support policy
    on, its identity (shared policy key, scope) joins the exact-IK cache key - v2 runs no longer read witnesses written without it (the
    re-policing of a cache hit stays as the guard); off: the key as before (legacy runs keep their entries)."""
    if not support_policy_on():
        return scope
    from ispatialgym.grasping import contact_policy as CP

    return {**scope, "support_policy": [CP.DEFAULT.key("right"), support_policy_scope()]}

class IKChecker:
    """§3.6 / §3.9 IK checks over a ``RobotKit``. Stateless apart from the kit and the cache namespace."""

    def __init__(self, kit, cache_scope: Optional[Dict[str, Any]] = None, default_working_arms: Sequence[str] = ("right",),
                 joint_limit_margin: Optional[float] = None, witness_residual_at_draw: Optional[float] = None):
        self.kit = kit
        self.cache_scope = dict(cache_scope or {})  # scene / rooms / placed objects: part of every cache key
        if joint_limit_margin is not None:
            # per-problem margin (``generation.joint_limit_margin_rad``; owner 2026-09-18 01:01: relaxed for touching) - the
            # instance attribute shadows the class default below; part of the cache scope so witnesses are not shared
            # across margins
            self.JOINT_LIMIT_MARGIN = float(joint_limit_margin)
            self.cache_scope["joint_limit_margin"] = round(float(joint_limit_margin), 4)
        #: (owner 2026-09-18 01:33, (a)) reach quality at the DRAW: when set (m), ``exact`` polishes the chain's last witness
        #: onto its pose and rejects the chain (``ik_reach``) when FK cannot get within this of the pose - the IK's own
        #: tolerance accepts poses ~1 cm beyond the reach boundary (joint 2 on its limit), which the program check would
        #: reject 20-60 s later and the oracle's straight legs stall on
        self.WITNESS_RESIDUAL_AT_DRAW: Optional[float] = None if witness_residual_at_draw is None else float(witness_residual_at_draw)
        if self.WITNESS_RESIDUAL_AT_DRAW is not None:
            self.cache_scope["witness_residual_at_draw"] = round(self.WITNESS_RESIDUAL_AT_DRAW, 4)
        #: the task's working arm(s) (``generation.arm``; core sets it) for stances the search checks before the module names them
        self.default_working_arms = list(default_working_arms)
        #: (arm, rounded pose) -> 28-D solution of a successful exact solve; cuRobo's remaining seeds are random, so a
        #: marginal pose can pass once and fail when re-solved (the program check re-solves every recorded pose): the
        #: sampler's own solution is used as the seed configuration then. Cleared per attempt (``core``).
        self.solutions: Dict[Tuple[str, Tuple[float, ...]], th.Tensor] = {}
        self._last_track_start: Optional[th.Tensor] = None  # pre-pose configuration of the last tracked approach
        self.last_track_stall: Optional[Dict[str, Any]] = None  # where the last ``track_chain`` stalled (``continue_chain`` callers' partial rule)
        self.finger_touch: Optional[FingerTouch] = None  # :meth:`set_finger_touch` (``generation.approach_finger_support_touch``)

    def _holding(self, arm: str) -> bool:
        """``arm`` has an object attached in the planner (``navigation.attached`` / ``R1ProCuRobo.attach_object``: a place, carry or release
        check), so its chain is not a grasp approach (:func:`support_policy_scope`)."""
        held = getattr(getattr(self.kit, "backend", None), "_attached", None)
        return isinstance(held, dict) and arm in held

    def support_policy_ok(self, q: th.Tensor, arm: str, supports: Sequence[Any], approached: Optional[Sequence[Any]] = None) -> Tuple[bool, str]:
        """The shared support-contact policy (``ispatialgym.grasping.contact_policy``) on a grasp configuration whose last pose was solved with
        ``supports`` out of the world: every link but ``arm``'s fingers clear of the world WITH the supports (the grasped / approached object
        left out, as the approach left it), and the hand's collision hulls against each support's top plane - the fingers within the 2 mm
        allowance, the palm, the wrist camera and the wrist not below it. Replaces the whole-support ignore of the last pose when
        ``ISPATIALGYM_SUPPORT_POLICY=1``. ``(ok, reason)``."""
        ev = self.support_policy_eval(q, arm, supports, approached)
        return bool(ev["ok"]), str(ev["reason"])

    def _support_policy_diag(self, q: th.Tensor, arm: str, supports: Sequence[Any], approached: Sequence[Any], target: Pose, why: str) -> Dict[str, Any]:
        """Opt-in ``ISPATIALGYM_SUPPORT_POLICY_DIAG`` (of-v2maps 2026-11-06): one ``SUPPORT_POLICY_DIAG {json}`` line per rejection - the
        penetration per finger hull, the plane z used (measured vs AABB), the approached object's scene z against the v2 rest's expected z
        (``support_policy_rest``: the plane + the rest's ``h_root``, registered by ``tabletop.map_bands``), and the IK residual of the
        configuration against the target pose."""
        import json

        from ispatialgym.oracle import r1pro_model as M

        ev = self.support_policy_eval(q, arm, supports, approached, finger_links=True)
        rec: Dict[str, Any] = {"why": why, "arm": arm, "stage": ev["stage"], "support": ev["support"], "plane_z": ev["plane_z"], "plane_src": ev["plane_src"],
                               "aabb_top": ev["aabb_top"], "finger_mm": {k[-1:]: round(1000 * v, 2) for k, v in ev["finger_by_link"].items()},
                               "clearance_mm": {k: round(1000 * v, 2) for k, v in ev["clearance"].items()},
                               "finger_q_m": ev.get("finger_q_m"), "opening_src": ev.get("opening_src"), "finger_q_config_m": ev.get("finger_q_config_m")}
        try:
            eef = self.kit.backend.fk([M.EEF_LINK[arm]], q)[M.EEF_LINK[arm]]
            rec["ik_pos_mm"] = round(1000 * float(th.linalg.norm(G.t(eef[0]).float() - G.t(target[0]).float())), 2)
            rec["ik_rot_mrad"] = round(1000 * float(G.quat_angle(G.t(eef[1]).float(), G.t(target[1]).float())), 2)
            rec["eef_z"] = round(float(G.t(eef[0])[2]), 4)
        except Exception as e:  # noqa: BLE001
            rec["ik_err"] = repr(e)[:80]
        rest = getattr(self, "support_policy_rest", None) or {}
        objs = []
        for o in approached or ():
            try:
                pos = o.get_position_orientation()[0]
                lo = o.aabb[0]
                d = {"name": getattr(o, "name", "?"), "z": round(float(pos[2]), 4), "aabb_min_z": round(float(lo[2]), 4)}
                if rest.get("h_root") is not None and ev["plane_z"] is not None:
                    d["expected_z"] = round(float(ev["plane_z"]) + float(rest["h_root"]), 4)
                    d["dz_mm"] = round(1000 * (d["z"] - d["expected_z"]), 2)
                    d["bottom_above_plane_mm"] = round(1000 * (float(lo[2]) - float(ev["plane_z"])), 2)
                objs.append(d)
            except Exception as e:  # noqa: BLE001
                objs.append({"err": repr(e)[:80]})
        rec["objects"], rec["rest"] = objs, {k: rest.get(k) for k in ("entry", "rest_id", "h_root")}
        # the CANDIDATE object pose the grasp was solved at (tabletop._phase3_draw sets it: during sampling no object stands there)
        cand = getattr(self, "_diag_object", None)
        if cand is not None and ev["plane_z"] is not None:
            try:
                pose, hull, gname = cand
                pz = float(G.t(pose[0])[2])
                lowest = float(G.local_points_world(hull, pose)[:, 2].min()) if hull is not None else None
                c = {"grasp": gname, "z": round(pz, 4), "lowest_above_plane_mm": None if lowest is None else round(1000 * (lowest - float(ev["plane_z"])), 2)}
                if rest.get("h_root") is not None:
                    c["expected_z"] = round(float(ev["plane_z"]) + float(rest["h_root"]), 4)
                    c["dz_mm"] = round(1000 * (pz - c["expected_z"]), 2)
                if rest.get("quat_xyzw") is not None:
                    up = G.rotate(G.t(pose[1]).float(), th.tensor([0.0, 0.0, 1.0]))
                    up_r = G.rotate(th.tensor([float(v) for v in rest["quat_xyzw"]]), th.tensor([0.0, 0.0, 1.0]))
                    c["tilt_vs_rest_mrad"] = round(1000 * float(th.acos(th.clamp(th.dot(up, up_r), -1.0, 1.0))), 2)
                rec["candidate"] = c
            except Exception as e:  # noqa: BLE001
                rec["candidate"] = {"err": repr(e)[:80]}
        log.info("SUPPORT_POLICY_DIAG %s", json.dumps(rec))
        return rec

    def support_policy_eval(self, q: th.Tensor, arm: str, supports: Sequence[Any], approached: Optional[Sequence[Any]] = None,
                            finger_links: bool = False) -> Dict[str, Any]:
        """:meth:`support_policy_ok` with its measurements (of-v2maps 2026-11-06): ``ok``, ``reason``, ``stage`` (``spheres`` | ``hulls`` |
        ``-``), the policed ``support`` and its ``plane_z`` / ``plane_src`` (``measured`` | ``aabb``) / ``aabb_top``, the hand hulls'
        ``clearance`` per role (m, the support the check failed on, else the last), ``finger_clearance`` (m) and with ``finger_links`` the
        clearance per finger link (``finger_by_link``). i7-floor 2026-11-07: a support registered as the floor (:func:`floor_support_names`,
        ``tabletop.floor_map_bands``) is policed in the policy's floor mode as the floor reach map was computed - the robot's ground-contact
        links out of the sphere stage (:func:`policy_links`), the hulls with the fingers' floor allowance (``HandChecker.check(floor=True)``);
        ``floor`` in the result."""
        from ispatialgym.grasping import contact_policy as CP
        from ispatialgym.oracle import r1pro_model as M

        out: Dict[str, Any] = {"ok": True, "reason": "", "stage": "-", "support": None, "plane_z": None, "plane_src": None, "aabb_top": None,
                               "clearance": {}, "finger_clearance": None, "finger_by_link": {}, "floor": False}
        others = self._non_finger_links(arm)
        if not others:
            return out
        # of-v2maps 2026-11-06: only the SUPPORTS under the approach are policed - an ignored object the hand is meant to touch (a pushed box,
        # approached by the chain) is not (v2scope1: touching_01 / 02 'touching push: no push class passes', 'finger_penetration on
        # target_box_of_sanitary_napkins'); a chain with no such support is not a grasp on a support
        all_supports = list(supports)
        supports = policed_supports(all_supports, approached, planes=dict(getattr(self, "support_plane_z", None) or {}))
        skipped = [getattr(sp, "name", "?") for sp in policed_supports(all_supports, approached, registered_only=False) if sp not in supports]
        if skipped:
            self._count("support_policy_unregistered_skipped")
            out["unregistered_skipped"] = skipped
        if not supports:
            return out
        floors = floor_support_names(self)
        others = policy_links(others, supports, floors, getattr(getattr(self.kit, "backend", None), "ground_contact_links", ()))
        drop = [o for o in (approached or ()) if all(o is not sp for sp in supports)]
        with self.ignoring(drop):
            if not bool(self.kit.links_clear(q, others, CP.DEFAULT.sphere_support_margin_m)[0]):
                out.update(ok=False, reason="non-finger link touches the support", stage="spheres")
                return out
        hc = getattr(self, "_hand_checker", None)
        if hc is None or hc.arm != arm:
            hc = self._hand_checker = CP.HandChecker(arm, CP.DEFAULT)
        eef = self.kit.backend.fk([M.EEF_LINK[arm]], q)[M.EEF_LINK[arm]]
        T = th.eye(4)
        T[:3, :3] = _quat_mat(G.t(eef[1]).float())
        T[:3, 3] = G.t(eef[0]).float()
        # of-v2maps 2026-11-06 (v2diag2): the fingers at the opening the oracle descends with - a library grasp's pre-shape (pinch width / 2
        # + the preshape margin, execute.execute_grasp; :meth:`policy_opening`, set by the grasp check), else the configuration's own
        # (fully open 0.05 m: the tilted pinches of tablespoon / carving knife / sunflower put the lower finger 4-17 mm into the table there,
        # +3.5 mm clear at the pre-shape the library validated them at)
        fq_q = float(G.t(q)[self.kit.robot.gripper_control_idx[arm]].float().mean())
        op = getattr(self, "_policy_opening", None)
        fq = float(op) if op is not None else fq_q
        out.update(finger_q_m=round(fq, 5), opening_src="preshape" if op is not None else "q", finger_q_config_m=round(fq_q, 5))
        planes = getattr(self, "support_plane_z", None)
        for sp in supports:
            top = support_plane_z(sp, planes)
            if top is None:  # a support without an AABB (a wall): the sphere rule above only
                continue
            name = getattr(sp, "name", "support")
            plane = CP.Support.plane(top, name=name)
            on_floor = name in floors
            ok, reason, cl = hc.check(T, plane, (fq, fq), floor=on_floor)
            try:
                aabb_top = float(sp.aabb[1][2])
            except Exception:  # noqa: BLE001
                aabb_top = None
            out.update(support=name, plane_z=float(top), plane_src="measured" if planes and name in planes else "aabb", aabb_top=aabb_top,
                       clearance={r: float(v[0]) for r, v in cl.items()}, finger_clearance=float(cl["finger"][0]), stage="hulls", floor=on_floor)
            if finger_links:
                out["finger_by_link"] = finger_link_clearance(hc, T, plane, (fq, fq))
            if not bool(ok[0]):
                out.update(ok=False, reason=f"{reason[0]} on {name}")
                return out
        return out

    @contextmanager
    def policy_opening(self, opening_m: Optional[float]):
        """Context manager (of-v2maps 2026-11-06): the support policy measures the fingers at ``opening_m`` per finger (the grasp's pre-shape,
        ``tabletop.policy_opening_of``) for the exact IKs inside; None = the configuration's own opening (the former behaviour)."""
        prev = getattr(self, "_policy_opening", None)
        self._policy_opening = None if opening_m is None else float(opening_m)
        try:
            yield
        finally:
            self._policy_opening = prev

    def set_finger_touch(self, touch: Optional[FingerTouch]) -> None:
        """Switch the approach finger-support touch rule (:class:`FingerTouch`) on (None: off); its parameters join the cache scope."""
        self.finger_touch = touch
        if touch is None:
            self.cache_scope.pop("finger_touch", None)
        else:
            self.cache_scope["finger_touch"] = [round(touch.margin_m, 4), round(touch.max_penetration_m, 4), bool(touch.preshape)]

    def _touch_objects(self, ignore_last: Sequence[Any]) -> List[Any]:
        """The objects of a chain's last-pose ignore set its approach fingers may touch: the grasp's supports (``FingerTouch.supports``)
        among them; [] with the rule off."""
        ft = self.finger_touch
        if ft is None or not ignore_last:
            return []
        sup = list(ft.supports() or [])
        return [o for o in ignore_last if any(o is s for s in sup)]

    def ignoring(self, objs: Sequence[Any], ground: bool = True, ground_meshes: bool = True):
        """Context manager: ``objs`` leave the cuRobo collision world for the queries inside (§3.6 "support ignored for the
        grasp pose only"). Their collision meshes are switched off by name (``R1ProCuRobo._mesh_enable``, what
        ``attach_object`` does) instead of the two full world re-syncs of ``RobotKit.ignoring`` (0.5–13 s each).
        ``ground=False`` also switches the cuRobo ground slab off (a floor-level push whose fingertips run < 4 cm above the
        floor: the padded fingertip spheres penetrate the slab while the real fingers do not; the oracle tracks the push
        legs without a collision veto)."""
        return _Ignoring(self.kit, list(objs), ground=ground, ground_meshes=ground_meshes)

    @staticmethod
    def _pose_key(pose: Pose) -> Tuple[float, ...]:
        p, q = G.t(pose[0]), G.t(pose[1])
        return tuple(round(float(v), 3) for v in p.tolist()) + tuple(round(float(v), 3) for v in q.tolist())

    SEED_POS_TOL = 0.03
    SEED_ANG_TOL = 0.15

    #: the solver's own acceptance (cuRobo ``position_tolerance`` 5 mm / ``orientation_tolerance`` 0.05 rad)
    WITNESS_POS_TOL = 0.005
    WITNESS_ANG_TOL = 0.05

    def _witness_ok(self, q: th.Tensor, pose: Pose, arm: str) -> bool:
        """``q`` reaches ``pose`` within the IK solver's acceptance (5 mm / 0.05 rad) and is collision-free in the current
        world at the solver's own margin (0: the IK's collision term has no extra clearance) — the same criterion the
        solver applies to its own solutions, evaluated deterministically."""
        from ispatialgym.oracle import r1pro_model as M

        try:
            eef = self.kit.fk([M.EEF_LINK[arm]], q)[M.EEF_LINK[arm]]
        except Exception:
            return False
        if float(th.linalg.norm(G.t(eef[0]) - G.t(pose[0]))) > self.WITNESS_POS_TOL or G.quat_angle(eef[1], G.t(pose[1])) > self.WITNESS_ANG_TOL:
            return False
        return bool(self.kit.collision_free(q, margin=0.0)[0])

    def collision_free_at(self, q: th.Tensor, pose: Pose, arm: str, ignore: Sequence[Any] = ()) -> bool:
        """``q`` (a configuration ``exact`` returned) reaches ``pose`` and is collision-free in the CURRENT world with
        ``ignore`` removed — the witness criterion re-evaluated after the world changed around the call (attachment
        2026-09-20 23:39: the attach pose is solved with the male attached and the partner ignored, then the same
        configuration is verified with the male detached and the partner present: the fingers / hand against the
        partner, which the male-partner filter never covers)."""
        with self.ignoring(list(ignore)):
            return self._witness_ok(q, pose, arm)

    def _seeded(self, q0: th.Tensor, pose: Pose, arm: str) -> th.Tensor:
        """``q0`` with the working arm set to the remembered solution of the nearest recorded pose of ``arm`` within
        (3 cm, 0.15 rad) — the same pose, or the same pose after the settle moved the object by a few mm (the other
        joints stay ``q0``'s: the base virtual joints must belong to the current robot root)."""
        p, q = G.t(pose[0]), G.t(pose[1])
        items = [(key, q_rec) for (a, key), q_rec in self.solutions.items() if a == arm]
        if not items:
            return q0
        keys = th.tensor([k for k, _ in items], dtype=th.float32)  # (M, 7): position + quaternion (vectorised search)
        d = th.linalg.norm(keys[:, :3] - p, dim=1)
        kq = keys[:, 3:] / th.linalg.norm(keys[:, 3:], dim=1, keepdim=True).clamp(min=1e-9)
        dots = th.clamp((kq * (q / th.linalg.norm(q).clamp(min=1e-9))).sum(dim=1).abs(), max=1.0)
        ang = 2.0 * th.acos(dots)
        ok = (d <= self.SEED_POS_TOL) & (ang <= self.SEED_ANG_TOL)
        if not bool(ok.any()):
            return q0
        d = th.where(ok, d, th.full_like(d, float("inf")))
        best = items[int(th.argmin(d))][1]
        out = q0.clone()
        idx = self.kit.robot.arm_control_idx[arm]
        out[idx] = best[idx]
        return out

    #: wide radius for *seeding* (not witnessing) a re-solve: the chain's other pose (PRE_DIST 0.10 away), an earlier draw
    RETRY_SEED_POS_TOL = 0.20
    RETRY_SEED_ANG_TOL = 0.8

    def _retry_seeds(self, q0: th.Tensor, pose: Pose, arm: str, chain: Sequence[th.Tensor] = ()) -> List[th.Tensor]:
        """Seeds for a re-solve after the random seeds missed ``pose``: the configurations found earlier in the same chain
        (nearest pose first) and the remembered solutions of ``arm`` within the wide radius, nearest first, up to 4."""
        p, q = G.t(pose[0]), G.t(pose[1])
        idx = self.kit.robot.arm_control_idx[arm]
        out: List[th.Tensor] = []
        for q_c in reversed(list(chain)):
            q_s = q0.clone()
            q_s[idx] = G.t(q_c)[idx]
            out.append(q_s)
        items = [(key, q_rec) for (a, key), q_rec in self.solutions.items() if a == arm]
        if items:
            keys = th.tensor([k for k, _ in items], dtype=th.float32)
            d = th.linalg.norm(keys[:, :3] - p, dim=1)
            kq = keys[:, 3:] / th.linalg.norm(keys[:, 3:], dim=1, keepdim=True).clamp(min=1e-9)
            ang = 2.0 * th.acos(th.clamp((kq * (q / th.linalg.norm(q).clamp(min=1e-9))).sum(dim=1).abs(), max=1.0))
            ok = (d <= self.RETRY_SEED_POS_TOL) & (ang <= self.RETRY_SEED_ANG_TOL)
            order = th.argsort(th.where(ok, d, th.full_like(d, float("inf"))))
            for j in order.tolist():
                if not bool(ok[j]) or len(out) >= 4:
                    break
                q_s = q0.clone()
                q_s[idx] = items[j][1][idx]
                out.append(q_s)
        return out

    # -- configurations -----------------------------------------------------------------------------------------
    POSTURES = {"untucked": "untucked", "tucked": "tucked", "reset": "tucked", "ready": "ready"}
    #: A6 (owner decision 2026-09-16): the idle arm of a manipulation stance may take any recorded collision-free posture;
    #: the alternatives tried, in order, when the untucked idle arm collides
    IDLE_POSTURES = ("untucked", "ready", "tucked")

    def arm_vectors(self, stance: Stance, arms: Optional[str] = None, right_arm: Optional[th.Tensor] = None, left_arm: Optional[th.Tensor] = None,
                    fingers: Optional[float] = None) -> Tuple[th.Tensor, th.Tensor, float]:
        """``(right_arm, left_arm, fingers)`` of a stance — see :func:`stance_arm_vectors`."""
        return stance_arm_vectors(self.kit, stance, arms, right_arm, left_arm, fingers)

    def q_locked(self, stance: Stance, arms: Optional[str] = None, right_arm: Optional[th.Tensor] = None, left_arm: Optional[th.Tensor] = None,
                 fingers: Optional[float] = None) -> th.Tensor:
        """28-D configuration at the stance (base pose + trunk) — the locked-joint source and IK seed of every check from
        it; arms and fingers per :meth:`arm_vectors`. The base virtual joints are always re-derived for the robot's
        current root."""
        r, l, f = self.arm_vectors(stance, arms, right_arm, left_arm, fingers)
        q = self.kit.q_config(arms="untucked", translate=stance.translate, tilt=stance.tilt, right_arm=r, left_arm=l, fingers=f)
        return self.kit.q_at_base_pose(q, *stance.base_pose2d)

    def stance_collision_free(self, stance: Stance, margin: float = 0.01, arms: Optional[str] = None, **arm_q) -> bool:
        return bool(self.kit.collision_free(self.q_locked(stance, arms, **arm_q), margin=margin)[0])

    def stances_collision_free(self, stances: Sequence[Stance], margin: float = 0.01, try_idle_postures: bool = True) -> th.Tensor:
        """Whole-body collision mask of the stances at their configuration. A6: a manipulation stance (``arms`` = list of
        working arms) whose untucked idle arm collides is retried with the alternative idle postures (``IDLE_POSTURES``,
        batched per posture); a survivor records the posture that passed in ``stance.idle_posture`` (its joint values go
        to the sidecar as ``arms_q``). Init stances (``arms = "untucked"``) are never re-postured."""
        if not stances:
            return th.zeros(0, dtype=th.bool)
        free = self.kit.collision_free(th.stack([self.q_locked(s) for s in stances]), margin=margin).clone()
        if not try_idle_postures:
            return free

        def idle_of(s: Stance) -> Optional[str]:
            # the stance search (tabletop Phase 2 / prompt0) checks candidates before the task module names the working
            # arm: a non-init stance without one uses the task's default working arm (``default_working_arms``)
            working = s.working_arms or (list(self.default_working_arms) if s.name not in ("init", "goal") and s.posture == "untucked" else [])
            if len(working) != 1:
                return None
            return "left" if working[0] == "right" else "right"

        retry = [i for i, s in enumerate(stances) if not bool(free[i]) and s.q is None and idle_of(s) is not None]
        for name in self.IDLE_POSTURES:
            if not retry:
                break
            todo = [i for i in retry if (stances[i].idle_posture or "untucked") != name]
            if not todo:
                continue
            qs = []
            for i in todo:
                s = stances[i]
                idle = idle_of(s)
                vec = self.kit.arm_q(self.POSTURES[name], idle)
                qs.append(self.q_locked(s, right_arm=vec if idle == "right" else None, left_arm=vec if idle == "left" else None))
            ok = self.kit.collision_free(th.stack(qs), margin=margin)
            for i, f in zip(todo, ok.tolist()):
                if f:
                    stances[i].idle_posture = name
                    if not stances[i].working_arms:  # record which arm the posture belongs to
                        stances[i].arms = list(self.default_working_arms)
                    free[i] = True
            retry = [i for i in retry if not bool(free[i])]
        return free

    # -- screening / exact --------------------------------------------------------------------------------------
    def screen(self, poses: Sequence[Pose], stance: Stance, arm: str = "right", ignore: Sequence[Any] = (), q_seed: Optional[th.Tensor] = None,
               fingers: Optional[float] = None, seeds: Sequence[th.Tensor] = (), witnesses: bool = True) -> th.Tensor:
        """12-seed batched IK per pose from the stance (locked joints = the stance's recorded configuration / posture at
        its trunk, ``fingers`` overrides the opening: closed for push / brace contacts); ``ignore`` objects are removed
        from the world for *these* poses (callers pass the support for grasp poses only).

        Before the solver is asked, a **remembered solution** of the same pose (the sampler's exact solve of that
        waypoint) is verified as a witness — FK + collision, deterministic — and counts as a pass; the 12 random seeds
        around a tucked / ready seed otherwise miss poses the 32-seed exact IK finds (the wall brace chain, 2 cm off the
        wall). Poses the batched solve still misses are retried seeded from the nearest remembered solution and from
        ``seeds`` (callers pass the endpoint solutions of a waypoint chain). Solutions found are remembered."""
        if len(poses) == 0:
            return th.zeros(0, dtype=th.bool)
        q0 = self.q_locked(stance, fingers=fingers) if q_seed is None else q_seed
        poses = list(poses)
        out = th.zeros(len(poses), dtype=th.bool)
        idx = self.kit.robot.arm_control_idx[arm]
        with self.ignoring(ignore):
            todo: List[int] = []
            for i, p in enumerate(poses):
                q_near = self._seeded(q0, p, arm) if witnesses else q0
                if not th.equal(q_near, q0) and self._witness_ok(q_near, p, arm):
                    out[i] = True
                else:
                    todo.append(i)
            if todo:
                ok, qs = self.kit.ik_batch_solve([poses[i] for i in todo], q0, arm=arm)
                for j, i in enumerate(todo):
                    if bool(ok[j]):
                        out[i] = True
                        self.solutions[(arm, self._pose_key(poses[i]))] = qs[j].clone()
            for i in [i for i in todo if not bool(out[i])]:
                # alternative seeds: the nearest remembered solution (a neighbouring waypoint), then the caller's seeds
                cands: List[th.Tensor] = []
                q_near = self._seeded(q0, poses[i], arm)
                if not th.equal(q_near, q0):
                    cands.append(q_near)
                for s in seeds:
                    if s is not None:
                        q_s = q0.clone()
                        q_s[idx] = G.t(s)[idx]
                        cands.append(q_s)
                for q_s in cands:
                    if fast_on("SINGLE_RETRY_EXACT", default=False):
                        # train-fast 2026-10-31: one pose re-solved in a padded 64-slot batch cost ~10 s (TITAN RTX, 8192-mesh store; 9 such
                        # calls = 90 s of one written instance); the 32-seed single solve from the same seed costs ~0.1 s and its solution
                        # passes the same collision checker (the nearest-to-seed one instead of the batched solver's best-of-12)
                        self._count("screen_single_retry")
                        q1s = self.kit.ik(poses[i], q_s, arm=arm)
                        if q1s is not None:
                            out[i] = True
                            self.solutions[(arm, self._pose_key(poses[i]))] = G.t(q1s).clone()
                            break
                        continue
                    ok1, q1 = self.kit.ik_batch_solve([poses[i]], q_s, arm=arm)
                    if bool(ok1[0]):
                        out[i] = True
                        self.solutions[(arm, self._pose_key(poses[i]))] = q1[0].clone()
                        break
        return out

    def screen_pairs(self, pairs: Sequence[Sequence[Pose]], stance: Stance, arm: str = "right", ignore_for_last: Sequence[Any] = (),
                     q_seed: Optional[th.Tensor] = None, fingers: Optional[float] = None) -> th.Tensor:
        """Batched screening of [pre, final] pose pairs: pre against the full world, final with ``ignore_for_last`` removed."""
        if len(pairs) == 0:
            return th.zeros(0, dtype=th.bool)
        q0 = self.q_locked(stance, fingers=fingers) if q_seed is None else q_seed
        pres = [p[0] for p in pairs]
        finals = [p[-1] for p in pairs]
        ok_pre = self.kit.ik_batch(pres, q0, arm=arm)
        idx = [i for i, s in enumerate(ok_pre.tolist()) if s]
        out = th.zeros(len(pairs), dtype=th.bool)
        if idx:
            with self.ignoring(ignore_for_last):
                ok_f = self.kit.ik_batch([finals[i] for i in idx], q0, arm=arm)
            for i, s in zip(idx, ok_f.tolist()):
                out[i] = bool(s)
        return out

    def exact(self, poses: Sequence[Pose], stance: Stance, arm: str = "right", ignore_for_last: Sequence[Any] = (),
              q_seed: Optional[th.Tensor] = None, scale_key: Any = None, cache: bool = True, fingers: Optional[float] = None,
              approached: Optional[Sequence[Any]] = None, q_first: Optional[th.Tensor] = None) -> IKResult:
        """32-seed IK of every pose in order from the stance (locked joints = the stance's recorded configuration /
        posture, ``fingers`` overrides the gripper opening); the last pose with ``ignore_for_last`` removed (the support
        under a grasp). Returns the configuration at the last pose. ``scale_key`` (the spawn scale vector of the object
        the poses belong to) is part of the cache key (§3.10).

        ``approached`` (owner decision 2026-09-16 10:15, A): the object the tracked approach segments close in on — the
        grasped target, the placed-on partner, the touched object — is removed from the world along the **whole** segment
        (side / top grasps straddle it over the last 10 cm by construction); the supports of ``ignore_for_last`` stay
        removed for the last waypoint only, the pre pose itself keeps the full world. ``None`` infers it from the end
        pose (``approached_objects``: the spawned objects whose expanded AABB holds the pad centre / fingertip).

        ``q_first`` (owner 2026-09-24): the configuration the chain's first pose is reached in - a witness recorded earlier in the
        same action (the ``pre_place`` the oracle's ``arm_to`` plans to before it tracks the descent). It is used as the first
        pose's solution when it reaches it (:meth:`reaches`) instead of a fresh 32-seed solve, so the tracked chain leaves from
        the very configuration the oracle will leave from; the backward re-tracking of ``track_approach`` may still replace it
        (``_last_track_start``) and the caller then updates the recorded witness."""
        if len(poses) == 0:
            return IKResult(True, None, "no poses", 0)
        q0 = self.q_locked(stance, fingers=fingers) if q_seed is None else q_seed
        if q_first is not None and cache:
            cache = False  # a chain pinned to a given start is never a cached verdict of the poses alone
        key = None
        if cache:
            key = cache_key("exact_ik_v3", {**self.exact_cache_scope(), "base": list(stance.base_pose2d), "trunk": [stance.translate, stance.tilt], "arm": arm,
                                            "poses": [list(G.t(p[0]).tolist()) + list(G.t(p[1]).tolist()) for p in poses],
                                            "ignore": sorted(getattr(o, "name", str(o)) for o in ignore_for_last), "scale": scale_key,
                                            "seeds": EXACT_SEEDS, "seed_arms": q0[self.kit.robot.arm_control_idx[arm]],
                                            "locked": q0[[i for a in ("left", "right") for i in self.kit.robot.arm_control_idx[a].tolist()] + self.kit.robot.gripper_control_idx["right"].tolist()]})
            hit = cache_get(key)
            fresh = hit is not None and self._fresh_draw()
            if fresh:
                self._count("ik_cache_fresh")  # train-fast 2026-11-01 (IK_FRESH_P): solve afresh although an entry exists (more branches)
            if hit is not None and not fresh:
                # a cached verdict is only a set of *candidate witnesses*: each stored configuration (arm joints on the
                # current locked configuration - the base joints belong to another robot root) is verified in the current
                # world (FK + collision, the last pose with ``ignore_for_last`` removed) before it counts; anything that
                # does not verify - a stale entry from another object layout, an entry without per-pose witnesses - is a
                # miss and gets solved. Entries written before 2026-09-16 14:45 carry the last pose only. Train-fast 2026-11-01: an
                # entry may hold several witness chains (``alts``, IK_FRESH_P); they are tried in a random order.
                chains = [hit.get("qs") or [None] * (len(poses) - 1) + [hit.get("q")]] + [c for c in (hit.get("alts") or []) if c]
                if len(chains) > 1:
                    chains = [chains[int(j)] for j in self._fresh_rng().permutation(len(chains))]
                idx = self.kit.robot.arm_control_idx[arm]
                for stored in chains:
                    verified: List[th.Tensor] = []
                    for i, (p, q_i) in enumerate(zip(poses, stored)):
                        if q_i is None or len(stored) != len(poses):
                            break
                        q_v = q0.clone()
                        q_v[idx] = th.tensor(q_i)[idx]
                        with self.ignoring(list(ignore_for_last) if i == len(poses) - 1 else []):
                            if not self._witness_ok(q_v, p, arm):
                                break
                        verified.append(q_v)
                    if len(verified) == len(poses) and bool(hit.get("ok", True)) and ignore_for_last and support_policy_on() \
                            and not (support_policy_scope() == "grasp" and self._holding(arm)):
                        # of-v2maps 2026-11-07 (the isg1020 spec's cache rule: a witness written without the policy - a legacy run's, the
                        # cache is shared - must not satisfy it): the policy re-checked on the cached witness, a rejection = this chain
                        # does not verify (the next stored chain / a fresh solve)
                        appr_c = approached if approached is not None else self.approached_objects(poses[-1])
                        if not bool(self.support_policy_eval(verified[-1], arm, ignore_for_last, appr_c)["ok"]):
                            self._count("ik_cache_policy_rejected")
                            continue
                    if len(verified) == len(poses) and bool(hit.get("ok", True)):
                        for p, q_v in zip(poses, verified):
                            self.solutions[(arm, self._pose_key(p))] = q_v.clone()
                        log.debug("exact IK cache hit verified (%d poses, arm %s)", len(poses), arm)
                        return IKResult(True, verified[-1], "cache", len(poses))
                log.debug("exact IK cache entry not verified in the current world (%d chains): solving", len(chains))
        q = None
        qs: List[Optional[th.Tensor]] = []
        n = len(poses)
        why = "exact IK failed"
        # the grasp's support the approach fingers may touch, on every approach segment of the chain (the program check tracks the
        # grasp approach as 3 segments; the support is in the last one's ignore set only)
        touch = self._touch_objects(ignore_for_last)
        tkw = {"touch": touch} if touch else {}
        for i, p in enumerate(poses):
            ign = list(ignore_for_last) if i == n - 1 else []
            if i > 0 and q is not None and is_approach(poses[i - 1], p):
                # a straight same-orientation segment from the previous pose (pre-grasp -> grasp, pre-place -> place,
                # pre-contact -> contact): the end pose is *tracked* from the previous solution so both witnesses lie in
                # one IK branch and the oracle's straight approach is executable (oracle finding: the french-fries pre_grasp
                # / grasp witnesses 3.16 rad apart -> plan_failed); an untrackable approach fails the chain
                near = list(approached) if approached is not None else self.approached_objects(p)
                q, why = self.track_approach(q, poses[i - 1], p, arm, ignore_end=ign, ignore_mid=near, q_hint=q0, **tkw)
                if q is None:
                    break
                m_lim, jn = self.joint_limit_margin(q, arm)
                if m_lim < self.JOINT_LIMIT_MARGIN:
                    # the tracked end configuration sits on a joint limit (touching: right_arm_joint2 at its limit on every
                    # witness - the oracle's approach stalled 11 mm short): re-solve the end pose with margin and track the
                    # segment backward so the pre witness follows that branch; fail the chain when no such branch exists
                    with self.ignoring(list(ign) + list(near)):
                        q_alt, why_lim = self.away_from_limits(q, p, arm, q0)
                    q_pre2 = None
                    if q_alt is not None and not th.equal(q_alt, q):
                        q_pre2, why_b = self.track_segment(q_alt, p, poses[i - 1], arm, ignore_mid=near, ignore_end=(), ignore_start=ign, **tkw)
                    if q_pre2 is None:
                        q, why = None, why_lim or f"ik_joint_limit: {jn} within {m_lim * 1000:.0f} mrad of its limit at the end of the tracked approach"
                        break
                    q = q_alt
                    self._last_track_start = q_pre2.clone()
                qs.append(q)
                if qs and len(qs) >= 2:
                    qs[-2] = self._last_track_start.clone()
                self.solutions[(arm, self._pose_key(poses[i - 1]))] = self._last_track_start.clone()
            else:
                with self.ignoring(ign):
                    # a remembered solution of this very pose is verified as a witness (FK + collision) before the solver
                    # is asked again (cuRobo's IK is randomised, a witness check is not); otherwise it seeds the solver
                    q_near = self._seeded(q0, p, arm) if q_seed is None else q0
                    # train-fast 2026-10-31 (REMEMBERED_WITH_SEED): with a caller's seed the remembered solution is consulted too - as the
                    # witness only (the solver, if asked, still starts from the caller's seed), and only when its arm joints lie within
                    # REMEMBERED_SEED_RAD (max abs difference, default 0.6 rad: the oracle's excursion bound) of the seed's - the same
                    # IK branch the caller asked for
                    q_rem = None
                    if q_seed is not None and fast_on("REMEMBERED_WITH_SEED", default=False):
                        cand = self._seeded(q0, p, arm)
                        idx_a = self.kit.robot.arm_control_idx[arm]
                        if not th.equal(cand, q0) and float((G.t(cand)[idx_a] - G.t(q0)[idx_a]).abs().max()) <= float(fast_value("REMEMBERED_SEED_RAD", 0.6)):
                            q_rem = cand
                    pinned = False
                    if i == 0 and q_first is not None and self.reaches(q_first, p, arm):
                        q = G.t(q_first).clone()  # the chain leaves from the recorded start configuration (owner 2026-09-24)
                    elif i == 0 and q_seed is not None and self._witness_ok(G.t(q_seed), p, arm):
                        # the caller's configuration IS the first pose (a held-object init: the hold posture the in-hand chain leaves
                        # from) - it is the chain's start, not a solution to pick; the joint-limit margin below is about the branch
                        # an approach would stall in and does not apply to it (object_scaling-beaker 0925-172725: 'right_arm_joint6
                        # within 3 mrad of its limit at the pose' on every precomputed (grasp, goal), the hold posture itself)
                        q, pinned = G.t(q_seed).clone(), True
                    elif q_seed is None and not th.equal(q_near, q0) and self._witness_ok(q_near, p, arm):
                        q = q_near
                    elif q_rem is not None and self._witness_ok(q_rem, p, arm):
                        self._count("remembered_with_seed")
                        q = q_rem
                    else:
                        q = self._solve_exact(p, q_near, arm)
                    if q is None:
                        # the 32 random seeds missed: retry seeded from the nearest remembered solution within the wide
                        # radius (the chain's other pose 10 cm away, an earlier draw of the same grasp) and from the
                        # solutions already found in this chain
                        for q_alt in self._retry_seeds(q0, p, arm, [x for x in qs if x is not None]):
                            q = self._solve_exact(p, q_alt, arm)
                            if q is not None:
                                break
                if q is None:
                    why = f"exact IK failed at pose {i + 1} of {n}"
                    break
                if not pinned:
                    with self.ignoring(ign):
                        q, why_lim = self.away_from_limits(q, p, arm, q0)
                    if q is None:
                        why = why_lim
                        break
            qs.append(q)
            self.solutions[(arm, self._pose_key(p))] = q.clone()
        if q is not None and self.WITNESS_RESIDUAL_AT_DRAW is not None:
            # reach quality (owner 2026-09-18 01:33): the last witness must reach its pose once polished - a solution the IK
            # accepted within its tolerance but 1 cm short lies beyond the arm's reach in this branch
            near_last = list(approached) if approached is not None else self.approached_objects(poses[-1])  # the grasped / touched object straddled by the pads
            ign_last = list(ignore_for_last) + list(near_last)
            q_pol, residual = self.polish_witness(q, poses[-1], arm, ignore=ign_last)
            if residual > self.WITNESS_RESIDUAL_AT_DRAW:
                m_lim, jn = self.joint_limit_margin(q, arm)
                q, why = None, (f"ik_reach: witness FK residual {residual * 1000:.1f} mm > {self.WITNESS_RESIDUAL_AT_DRAW * 1000:.0f} mm at pose {n} of {n} "
                                f"(the pose lies at the arm's reach boundary; {jn} {m_lim * 1000:.0f} mrad from its limit)")
            else:
                q = q_pol
                qs[-1] = q
                self.solutions[(arm, self._pose_key(poses[-1]))] = q.clone()
        if q is not None and ignore_for_last and support_policy_on() and not (support_policy_scope() == "grasp" and self._holding(arm)):
            # of-v2maps 2026-11-06: the objects the last pose closes in on (as the IK's own last-pose ignore infers them) - a push that
            # passes no ``approached`` has its pushed box exempted as the grasp's target is, the support under it stays policed
            appr = approached if approached is not None else self.approached_objects(poses[-1])
            ev_p = self.support_policy_eval(q, arm, ignore_for_last, appr)
            ok_p, why_p = bool(ev_p["ok"]), str(ev_p["reason"])
            if not ok_p:
                if os.environ.get("ISPATIALGYM_SUPPORT_POLICY_DIAG"):
                    self._support_policy_diag(q, arm, ignore_for_last, appr, poses[-1], why_p)
                # of-v2maps 2026-11-06: what the grasp-check fallback (tabletop.support_policy_fallback) needs - the rejected configuration
                # and its measurements (the deepest finger's clearance)
                self._last_policy_fail = {"q": q.clone(), "ev": ev_p, "n": len(poses), "pose": poses[-1]}
                q, why = None, f"support_policy: {why_p}"
        res = IKResult(q is not None, q, "" if q is not None else why, len(poses))
        if key is not None and res.ok:  # a failure may be a random miss of the solver: only witnesses are cached
            chain = [None if x is None else x.tolist() for x in qs]
            entry = {"ok": True, "q": q.tolist(), "qs": chain}
            if fast_on("IK_CACHE_GROW"):
                # train-fast 2026-11-01 (IK_CACHE_GROW): the entry keeps every distinct branch found (a fresh solve under IK_FRESH_P, or a
                # re-solve after the stored chains failed to verify) - the first chain as before, the others in ``alts``
                old = hit if hit is not None else cache_get(key)
                entry = self._grown_entry(old, entry, arm)
            cache_put(key, entry)
        return res

    def exact_cache_scope(self) -> Dict[str, Any]:
        """The scope of the exact-IK cache keys. Train-fast 2026-11-01 (IK_CACHE_SHARE_CROP, default on): without the scene crop - every hit
        is verified in the current world (FK + collision) before it counts, so a cropped world and the full world of the same scene and
        loaded rooms share entries (a full-world witness is collision-free among the crop's subset of objects; a crop-world witness is
        re-checked before the full world uses it). The feasible-set and start-draw caches keep the crop: their entries are not re-checked."""
        if fast_on("IK_CACHE_SHARE_CROP") and "scene_crop" in self.cache_scope:
            out = {k: v for k, v in self.cache_scope.items() if k not in ("scene_crop", "full_rooms")}
            if self.cache_scope.get("full_rooms"):
                out["rooms"] = list(self.cache_scope["full_rooms"])  # the full world's key (its scope's rooms are the rooms it loads)
        else:
            out = {k: v for k, v in self.cache_scope.items() if k != "full_rooms"}
        return with_policy_scope(out)


    # -- train-fast 2026-11-01: fresh solves and several witnesses per exact-IK cache entry ------------------------------------------
    #: share of exact-IK calls with a cache entry that solve afresh anyway (IK_FRESH_P); the new branch joins the entry
    FRESH_P = 0.2
    #: a chain is a new branch when its last pose's working-arm joints differ from every stored chain's by more than this (max abs, rad)
    BRANCH_MIN_RAD = 0.2
    #: chains kept per entry
    MAX_CHAINS = 6

    def _fresh_rng(self):
        r = getattr(self, "_fresh_rng_", None)
        if r is None:
            import numpy as np

            r = self._fresh_rng_ = np.random.default_rng()
        return r

    def _fresh_draw(self) -> bool:
        p = float(fast_value("IK_FRESH_P", self.FRESH_P))
        return p > 0 and float(self._fresh_rng().random()) < p

    def _grown_entry(self, old: Optional[Dict[str, Any]], new: Dict[str, Any], arm: str) -> Dict[str, Any]:
        """``old`` (a cache entry or None) with ``new``'s chain added as an alternative when it is another branch (:data:`BRANCH_MIN_RAD`)."""
        if not old or not old.get("qs"):
            return new
        idx = [int(i) for i in self.kit.robot.arm_control_idx[arm].tolist()]
        chains = [old["qs"]] + [c for c in (old.get("alts") or []) if c]

        def last_arm(chain):
            last = chain[-1] if chain else None
            return None if last is None else [float(last[i]) for i in idx]

        mine = last_arm(new["qs"])
        if mine is None:
            return old
        for c in chains:
            other = last_arm(c)
            if other is not None and max(abs(a - b) for a, b in zip(mine, other)) <= float(fast_value("IK_BRANCH_MIN_RAD", self.BRANCH_MIN_RAD)):
                return old  # the same branch: nothing new
        self._count("ik_cache_new_branch")
        out = dict(old)
        out["alts"] = (list(old.get("alts") or []) + [new["qs"]])[-(int(fast_value("IK_CACHE_MAX_CHAINS", self.MAX_CHAINS)) - 1):]
        return out

    # -- witness quality (primary oracle 2026-09-16 23:06) ---------------------------------------------------------------
    #: a chain witness must keep this much room (rad) to every arm joint limit: the oracle's straight approach / descent from
    #: it stalls at the limit otherwise (touching 0916-223613: right_arm_joint2 at -3.139, limit -3.142, plan_failed at grasp)
    JOINT_LIMIT_MARGIN = 0.05
    #: re-solves from random arm configurations after the seeded re-solves of ``away_from_limits`` stayed within the margin
    #: (owner 2026-09-26: bar soap's place pose - valid solutions exist on some pods and were missed on others)
    LIMIT_RETRY_RANDOM = 12
    #: the random re-solves run from this attempt of an instance on (the core sets ``attempt`` per attempt; ``None`` = outside
    #: the attempt loop, re-solves on). Owner 2026-09-26 (sandal 0926-023744 / 023746, wave 9 phase 1): a rescued solution lies
    #: in another IK branch and is remembered as a seed (``solutions``), so the later candidates of the same attempt are solved
    #: in that branch too - sandal's attempt 0, an instance on waves 6 / 7, ended without one. Attempt 0 therefore solves as
    #: before the re-solves existed; an attempt after a rejected one gets them (bar soap: its flaky pod rejects attempt 0).
    LIMIT_RETRY_RANDOM_FROM_ATTEMPT = 1
    attempt: Optional[int] = None
    #: FK(q) of a recorded witness must reach its pose within this (m) after polishing
    WITNESS_RESIDUAL_MAX = 0.002

    def joint_limit_margin(self, q: th.Tensor, arm: str) -> Tuple[float, str]:
        """``(smallest distance of the arm joints to a limit in rad, that joint's name)``."""
        from ispatialgym.oracle import r1pro_model as M

        idx = self.kit.robot.arm_control_idx[arm]
        lo = self.kit.robot.joint_lower_limits[idx].float()
        hi = self.kit.robot.joint_upper_limits[idx].float()
        qa = G.t(q)[idx]
        margins = th.minimum(qa - lo, hi - qa)
        k = int(th.argmin(margins))
        names = M.ARM_JOINTS[arm]
        return float(margins[k]), names[k] if k < len(names) else f"{arm}_arm_joint{k + 1}"

    def away_from_limits(self, q: th.Tensor, pose: Pose, arm: str, q0: th.Tensor, tries: int = 4) -> Tuple[Optional[th.Tensor], str]:
        """Keep ``q`` when its arm joints are >= ``JOINT_LIMIT_MARGIN`` from every limit; otherwise re-solve ``pose`` from other
        seeds (the remembered neighbours, then the solver's random seeds) and return the solution with the largest margin
        when it meets the bound, else ``(None, reason)``."""
        m, jn = self.joint_limit_margin(q, arm)
        if m >= self.JOINT_LIMIT_MARGIN:
            return q, ""
        self._count("limit_retry_poses")
        best, best_m = q, m
        seeds = self._retry_seeds(q0, pose, arm) + [q0] * max(0, tries - 1)
        for q_s in seeds[:tries]:
            q_alt = self._solve_max_margin(pose, q_s, arm)
            if q_alt is None:
                continue
            m_alt, _ = self.joint_limit_margin(q_alt, arm)
            if m_alt > best_m:
                best, best_m = q_alt, m_alt
            if best_m >= self.JOINT_LIMIT_MARGIN:
                self._count("limit_retry_seeded_rescued")
                return best, ""
        if self.LIMIT_RETRY_RANDOM > 0 and (self.attempt is None or self.attempt >= self.LIMIT_RETRY_RANDOM_FROM_ATTEMPT):
            # owner 2026-09-26 (bar soap 0926-011542 / 0925-235551: every solution of the place pose within 2-12 mrad of
            # right_arm_joint6's limit on some pods, a valid one for the same candidate on others): the re-solves above start
            # from the remembered seeds and q0 - near-identical starts; these start from random arm configurations (a generator
            # seeded by the pose, reproducible) and run only in this situation, when the candidate would be rejected otherwise
            idx = self.kit.robot.arm_control_idx[arm]
            lo = self.kit.robot.joint_lower_limits[idx].float()
            hi = self.kit.robot.joint_upper_limits[idx].float()
            gen = th.Generator().manual_seed(hash(tuple(round(float(v), 4) for v in list(G.t(pose[0]).tolist()) + list(G.t(pose[1]).tolist()))) % (2 ** 31))
            for k in range(self.LIMIT_RETRY_RANDOM):
                self._count("limit_retry_random")
                q_s = G.t(q0).clone().float()
                q_s[idx] = lo + (hi - lo) * th.rand(len(idx), generator=gen)
                q_alt = self._solve_max_margin(pose, q_s, arm)
                if q_alt is None:
                    continue
                m_alt, _ = self.joint_limit_margin(q_alt, arm)
                if m_alt > best_m:
                    best, best_m = q_alt, m_alt
                if best_m >= self.JOINT_LIMIT_MARGIN:
                    self._count("limit_retry_random_rescued")
                    log.info("ik: a solution %.0f mrad from every joint limit after %d random re-solves (the first solutions stood within %.0f mrad of %s's limit)",
                             best_m * 1000, k + 1, m * 1000, jn)
                    return best, ""
        m_b, jn_b = self.joint_limit_margin(best, arm)
        self._count(f"limit_reject.{jn_b}")
        return None, f"ik_joint_limit: {jn_b} within {m_b * 1000:.0f} mrad of its limit at the pose (every solution found; the approach from it would stall)"

    def _count(self, key: str, n: int = 1) -> None:
        """Add ``n`` to the kit's call counter ``key`` (``RobotKit.stats``: per instance in ``summary.json["kit"]``)."""
        st = getattr(getattr(self, "kit", None), "stats", None)
        if isinstance(st, dict):
            st[key] = st.get(key, 0) + n

    # -- return-all solves (train-fast 2026-10-31, LIMIT_RETRY_RETURN_ALL) ------------------------------------------------
    #: solutions of one solve farther than this (rad, max abs arm joint difference) from the nearest-to-seed solution belong to
    #: another IK branch and are not picked in its place (the oracle's finish / excursion bound, ``FINISH_MAX_EXCURSION``)
    RETURN_ALL_BRANCH_RAD = 0.6

    def _solve_exact(self, pose: Pose, q_seed: th.Tensor, arm: str) -> Optional[th.Tensor]:
        """One 32-seed solve of ``pose`` from ``q_seed`` for a chain witness: the solution nearest to the seed (``kit.ik``, the former
        behaviour), or with ``fast_on("LIMIT_RETRY_RETURN_ALL", default=False)`` every solution of the same solve (``kit.ik_all``) and the pick of
        :meth:`_pick_solution` within the nearest solution's branch - the one clear of the joint limits when the solve found one, so
        :meth:`away_from_limits` (4 seeded + 12 random re-solves, ~0.1 s each) usually has nothing left to do. A kit without
        ``ik_all`` (tests' stubs) solves as before."""
        ik_all = getattr(self.kit, "ik_all", None)
        if not (callable(ik_all) and fast_on("LIMIT_RETRY_RETURN_ALL", default=False)):
            return self.kit.ik(pose, q_seed, arm=arm)
        sols = ik_all(pose, q_seed, arm=arm)
        if sols is None or len(sols) == 0:
            return None
        return self._pick_solution(sols, q_seed, arm, branch=True)

    def _solve_max_margin(self, pose: Pose, q_seed: th.Tensor, arm: str) -> Optional[th.Tensor]:
        """A re-solve of :meth:`away_from_limits`: as :meth:`_solve_exact`, but every branch is eligible (the re-solves from random arm
        configurations land anywhere by design) - the solution clear of the limits nearest to the seed, else the largest margin."""
        ik_all = getattr(self.kit, "ik_all", None)
        if not (callable(ik_all) and fast_on("LIMIT_RETRY_RETURN_ALL", default=False)):
            return self.kit.ik(pose, q_seed, arm=arm)
        sols = ik_all(pose, q_seed, arm=arm)
        if sols is None or len(sols) == 0:
            return None
        return self._pick_solution(sols, q_seed, arm, branch=False)

    def _pick_solution(self, sols: th.Tensor, q_seed: th.Tensor, arm: str, branch: bool) -> th.Tensor:
        """Among the ``(K, 28)`` solutions of one solve: the candidates are all of them (``branch=False``) or those within
        ``RETURN_ALL_BRANCH_RAD`` of the solution nearest to ``q_seed`` (``branch=True``); the pick is the candidate with margin
        >= ``JOINT_LIMIT_MARGIN`` nearest to the seed - the former nearest-to-seed pick itself whenever that one is clear, so the
        branch a chain uses only changes when the former pick would have gone to ``away_from_limits`` - else the largest margin."""
        idx = self.kit.robot.arm_control_idx[arm]
        sols = G.t(sols).reshape(-1, G.t(sols).shape[-1]).float()
        qa = sols[:, idx]
        d = th.linalg.norm(qa - G.t(q_seed)[idx].float().unsqueeze(0), dim=1)
        k_near = int(th.argmin(d))
        lo = self.kit.robot.joint_lower_limits[idx].float()
        hi = self.kit.robot.joint_upper_limits[idx].float()
        margins = th.minimum(qa - lo, hi - qa).amin(dim=1)
        cand = ((qa - qa[k_near].unsqueeze(0)).abs().amax(dim=1) <= self.RETURN_ALL_BRANCH_RAD) if branch else th.ones(len(sols), dtype=th.bool)
        clear = cand & (margins >= self.JOINT_LIMIT_MARGIN)
        if bool(clear.any()):
            k = int(th.argmin(th.where(clear, d, th.full_like(d, float("inf")))))
            if k != k_near:
                self._count("return_all_repicked")
        else:
            k = int(th.argmax(th.where(cand, margins, th.full_like(margins, -float("inf")))))
        return sols[k].clone()

    def _solve_ik_path(self, emb: str, waypoints: Sequence[Dict[str, Pose]], **kw):
        """``kit.ik_path`` (timed: ``ik_path_calls`` / ``ik_path_s`` / ``ik_path_waypoints``; train-fast 2026-10-31) or, on a kit without
        it (tests' stubs), the backend's ``solve_ik_path`` with the same counters kept here."""
        fn = getattr(self.kit, "ik_path", None)
        if callable(fn):
            return fn(emb, waypoints, **kw)
        t0 = time.time()
        try:
            return self.kit.backend.solve_ik_path(emb, waypoints, **kw)
        finally:
            self._count("ik_path_calls")
            self._count("ik_path_waypoints", len(waypoints))
            self._count("ik_path_s", time.time() - t0)

    # -- gripper symmetry (twin grasps) ---------------------------------------------------------------------------
    def twin_q(self, q: Optional[th.Tensor], arm: str) -> Optional[th.Tensor]:
        """Configuration holding the twin (``conventions.twin_pose``) of the grasp ``q`` holds: ``conventions.twin_arm_q`` on
        ``arm``'s joints, every other joint kept. None when ``q`` is None or joint 5 would leave its range."""
        if q is None:
            return None
        idx = self.kit.robot.arm_control_idx[arm]
        qa = C.twin_arm_q(G.t(q)[idx])
        if qa is None:
            return None
        out = G.t(q).clone()
        out[idx] = qa.to(out.dtype)
        return out

    def polish_witness(self, q: th.Tensor, pose: Pose, arm: str, ignore: Sequence[Any] = ()) -> Tuple[th.Tensor, float]:
        """``(q', residual_m)``: a few damped-least-squares iterations drive the arm from ``q`` onto ``pose`` exactly (the IK
        accepts 5 mm, the recorded witness should reach the recorded pose); the polished configuration is kept only when
        it stays collision-free (``ignore`` removed) and within the joint-limit margin. The residual is |FK(q') - pose|."""
        from ispatialgym.oracle import r1pro_model as M

        best = G.t(q).clone()
        try:
            res = self._solve_ik_path(f"{arm}_arm", [{M.EEF_LINK[arm]: (G.t(pose[0]), G.t(pose[1]))}], q_start=best, pos_tol=1e-3, rot_tol=5e-3,
                                      max_iters=40, check_collisions=False, sync_world=False, max_joint_step=0.3)
            if res.success and res.q_traj is not None:
                cand = res.q_traj[-1].clone()
                with self.ignoring(list(ignore)):
                    free = bool(self.kit.collision_free(cand, margin=0.0)[0])
                if free and self.joint_limit_margin(cand, arm)[0] >= self.JOINT_LIMIT_MARGIN * 0.5:
                    best = cand
        except Exception as e:  # noqa: BLE001 - polishing is best effort
            log.debug("witness polish failed: %r", e)
        return best, self.fk_residual(best, pose, arm)

    def fk_residual(self, q: th.Tensor, pose: Pose, arm: str) -> float:
        """|FK(q) - pose| (m) of the eef in the current robot root frame."""
        from ispatialgym.oracle import r1pro_model as M

        eef = self.kit.fk([M.EEF_LINK[arm]], G.t(q))[M.EEF_LINK[arm]]
        return float(th.linalg.norm(G.t(eef[0]) - G.t(pose[0])))

    #: dense straight-line steps of :meth:`reach_collisions_off` from the start eef to the pose
    REACH_OFF_STEPS = 24

    def reach_collisions_off(self, pose: Pose, q_starts: Sequence[th.Tensor], arm: str) -> Dict[str, Any]:
        """Diagnostic (oracle-fix 2026-11-04 round 3): how close ``arm``'s eef gets to ``pose`` with collisions off, by damped least
        squares tracked along the straight line from each start's eef (``REACH_OFF_STEPS`` waypoints; a single far waypoint made the
        DLS return no configuration at all, and the caller measured the START: the 674-846 mm 'short' of the bookcase F_goal blame).
        The best start's ``{"q", "reached", "short_mm", "rot_rad", "limit_mrad", "limit_joint", "at_limits", "status"}``; ``reached``
        = within the witness tolerance (:meth:`reaches`)."""
        from ispatialgym.oracle import r1pro_model as M

        be = self.kit.backend
        self.kit._ensure_synced()
        best: Optional[Dict[str, Any]] = None
        for q0 in q_starts:
            q0 = G.t(q0).clone()
            eef = self.kit.fk([M.EEF_LINK[arm]], q0)[M.EEF_LINK[arm]]
            wps = C.line_waypoints((G.t(eef[0]), G.t(eef[1])), (G.t(pose[0]), G.t(pose[1])), self.REACH_OFF_STEPS)
            res = be.solve_ik_path(f"{arm}_arm", [{M.EEF_LINK[arm]: (G.t(p[0]), G.t(p[1]))} for p in wps], q_start=q0, pos_tol=1e-3, rot_tol=5e-3,
                                   max_iters=60, check_collisions=False, sync_world=False, max_joint_step=10.0)
            q = res.q_traj[-1].clone() if getattr(res, "q_traj", None) is not None else q0
            if not res.success:
                # the converged prefix stops a waypoint short: finish on the pose itself from there (a miss returns no q: keep the prefix)
                fin = be.solve_ik_path(f"{arm}_arm", [{M.EEF_LINK[arm]: (G.t(pose[0]), G.t(pose[1]))}], q_start=q, pos_tol=1e-3, rot_tol=5e-3,
                                       max_iters=200, check_collisions=False, sync_world=False, max_joint_step=10.0)
                if getattr(fin, "q_traj", None) is not None:
                    q = fin.q_traj[-1].clone()
            e = self.kit.fk([M.EEF_LINK[arm]], q)[M.EEF_LINK[arm]]
            m_lim, jn = self.joint_limit_margin(q, arm)
            diag = getattr(res, "diagnostics", None) or {}
            out = {"q": q, "reached": self.reaches(q, pose, arm), "short_mm": round(float(th.linalg.norm(G.t(e[0]) - G.t(pose[0]))) * 1000, 1),
                   "rot_rad": round(float(G.quat_angle(e[1], G.t(pose[1]))), 3), "limit_mrad": round(m_lim * 1000, 0), "limit_joint": jn,
                   "at_limits": list(diag.get("at_lower", [])) + list(diag.get("at_upper", [])), "status": str(getattr(res, "status", ""))[:120]}
            if best is None or (out["reached"], -out["short_mm"]) > (best["reached"], -best["short_mm"]):
                best = out
            if out["reached"]:
                break
        return best or {"q": None, "reached": False, "short_mm": float("nan"), "rot_rad": float("nan"), "limit_mrad": float("nan"), "limit_joint": "",
                        "at_limits": [], "status": "no start"}

    def canonical_q(self, q: th.Tensor, stance: Stance) -> th.Tensor:
        """``q`` with the virtual base joints re-expressed relative to a robot root at the **world origin** (identity pose,
        z = 0) so that FK with an identity root puts ``base_link`` at the stance's recorded ``base_pose2d`` on the measured floor
        — the configuration a sidecar consumer can FK without knowing where the sampler's articulation root stood."""
        x, y, yaw = (float(v) for v in stance.base_pose2d)
        return C.q_with_base_pose(G.t(q), x, y, yaw, root_pose2d=(0.0, 0.0, 0.0), base_z=float(self.kit.floor_z) + C.NOMINAL_BASE_Z, root_z=0.0)

    # -- straight approaches (branch consistency) -----------------------------------------------------------------
    #: dense tracking step along an approach (m), the largest joint change allowed per dense step (rad), the largest
    #: joint excursion allowed between two §3.9 waypoints (~3.3 cm apart; the oracle's finishing-motion bound), and the
    #: longest segment treated as an approach
    APPROACH_STEP_M = 0.01
    APPROACH_MAX_STEP = 0.2
    APPROACH_MAX_EXCURSION = 0.6
    APPROACH_WAYPOINT_M = 0.1 / 3

    #: an object counts as approached by an end pose when its AABB, grown by this, holds the pad centre or the fingertip
    APPROACHED_AABB_PAD = 0.03

    def approached_objects(self, end_pose: Pose) -> List[Any]:
        """Spawned objects an approach ending at ``end_pose`` closes in on: those whose AABB (hull at the PhysX pose when
        the world caches it, else ``obj.aabb``) grown by ``APPROACHED_AABB_PAD`` contains the pad centre or the fingertip
        point of the end pose. A grasp straddles the target, a place hovers 1 cm over the partner, a contact sits 5 mm
        off the touched face: all three land inside. Objects attached to the planner are already out of the world."""
        world = getattr(self.kit, "world", None)
        spawned = list(getattr(world, "spawned", {}).values()) if world is not None else []
        if not spawned:
            return []
        if hasattr(world, "sync_render_transforms"):
            world.sync_render_transforms()  # ``obj.aabb`` reads the render-side transform (stale after physics-only steps)
        pts = th.stack([C.pad_center((G.t(end_pose[0]), G.t(end_pose[1]))), C.fingertip((G.t(end_pose[0]), G.t(end_pose[1])))])
        out = []
        for o in spawned:
            try:
                lo, hi = o.aabb
            except Exception:  # noqa: BLE001
                continue
            lo, hi = G.t(lo) - self.APPROACHED_AABB_PAD, G.t(hi) + self.APPROACHED_AABB_PAD
            if bool(((pts >= lo) & (pts <= hi)).all(dim=1).any()):
                out.append(o)
        return out

    def track_segment(self, q_from: th.Tensor, pose_from: Pose, pose_to: Pose, arm: str, ignore_mid: Sequence[Any] = (),
                      ignore_end: Sequence[Any] = (), ignore_start: Sequence[Any] = (), touch: Sequence[Any] = ()) -> Tuple[Optional[th.Tensor], str]:
        """Differential-IK tracking (``backend.solve_ik_path``) of the straight segment ``pose_from -> pose_to`` from the
        configuration ``q_from`` (which reaches ``pose_from``) in ``APPROACH_STEP_M`` steps: continuous joint path, no
        step above ``APPROACH_MAX_STEP`` rad, joint excursion between consecutive §3.9 waypoints ≤ ``APPROACH_MAX_EXCURSION``,
        every configuration collision-free with ``ignore_mid`` removed — the ones within ``APPROACH_WAYPOINT_M`` of the end
        pose additionally with ``ignore_end`` removed (the last centimetres of a grasp descent run inside the support's
        sphere margin, as the end pose does: §3.6 "support ignored for the grasp pose only" at the §3.9 waypoint
        granularity), the ones within it of the start pose with ``ignore_start`` removed (backward tracking).
        ``touch`` (the grasp's support, :meth:`_touch_objects`): a configuration that collides with it in the world is re-checked by
        :meth:`_finger_touch_ok` (the finger links may touch it lightly). Returns ``(q_end, "ok")`` or ``(None, reason)``."""
        from ispatialgym.oracle import r1pro_model as M

        p0, p1 = (G.t(pose_from[0]), G.t(pose_from[1])), (G.t(pose_to[0]), G.t(pose_to[1]))
        dist = float(th.linalg.norm(p1[0] - p0[0]))
        n = max(2, int(math.ceil(dist / self.APPROACH_STEP_M)))
        wps = [tuple(w) for w in C.line_waypoints(p0, p1, n)]
        self.kit._ensure_synced()
        with self.ignoring(ignore_mid):
            res = self._solve_ik_path(f"{arm}_arm", [{M.EEF_LINK[arm]: (G.t(w[0]), G.t(w[1]))} for w in wps], q_start=G.t(q_from),
                                      check_collisions=False, sync_world=False, max_joint_step=self.APPROACH_MAX_STEP, max_iters=60)
        self.kit.stats["ik_calls"] = self.kit.stats.get("ik_calls", 0) + 1  # kept (the former accounting; the timed one is ik_path_*)
        if not res.success:
            return None, f"approach_untrackable: {res.status}"
        traj = res.q_traj  # (n + 1, 28): the start and one configuration per dense waypoint
        idx = self.kit.robot.arm_control_idx[arm]
        k = max(1, int(round(self.APPROACH_WAYPOINT_M / self.APPROACH_STEP_M)))
        coarse = list(range(0, len(traj) - 1, k)) + [len(traj) - 1]
        exc = max(float(th.max(th.abs(traj[b, idx] - traj[a, idx]))) for a, b in zip(coarse[:-1], coarse[1:]))
        if exc > self.APPROACH_MAX_EXCURSION:
            return None, f"approach_untrackable: joint excursion {exc:.2f} rad between consecutive waypoints (> {self.APPROACH_MAX_EXCURSION})"
        # configuration i (1..n) sits i / n of the way; the tails near the start / end poses take those poses' ignore sets
        m = len(traj) - 1
        is_end = [dist * (m - i) / m < self.APPROACH_WAYPOINT_M - 1e-9 for i in range(m + 1)]
        is_start = [dist * i / m < self.APPROACH_WAYPOINT_M - 1e-9 for i in range(m + 1)]
        # a short segment (the §3.9 waypoint spacing itself) lies within range of both poses: those configurations take
        # both ignore sets (the grasp's support stays removed 8 mm from the grasp whichever way the segment is tracked)
        both = [i for i in range(1, m) if is_end[i] and is_start[i]]
        near_end = [i for i in range(1, m) if is_end[i] and not is_start[i]]
        near_start = [i for i in range(1, m) if is_start[i] and not is_end[i]]
        body = [i for i in range(1, m) if not is_end[i] and not is_start[i]]
        touched: List[float] = []
        for idxs, extra, what in ((body, (), "an intermediate configuration of the approach collides"),
                                  (near_start, ignore_start, "a configuration just after the start pose collides"),
                                  (near_end, ignore_end, "a configuration just before the end pose collides"),
                                  (both, list(ignore_start) + list(ignore_end), "a configuration of the short segment collides")):
            if not idxs:
                continue
            ign = list(ignore_mid) + list(extra)
            with self.ignoring(ign):
                free = self.kit.collision_free(traj[idxs], margin=0.0)
            present = [o for o in touch if all(o is not x for x in ign)]
            if not bool(free.all()) and present:
                hit = [j for j, f in enumerate(free.tolist()) if not f]
                ok_t, pen_mm = self._finger_touch_ok(traj[[idxs[j] for j in hit]], arm, ign, present)
                free = free.clone()
                for j, ok_j in zip(hit, ok_t.tolist()):
                    free[j] = bool(ok_j)
                if bool(ok_t.any()):
                    touched.append(pen_mm)
            if not bool(free.all()):
                bad = [idxs[j] for j, f in enumerate(free.tolist()) if not f]
                return None, f"approach_untrackable: {what} ({len(bad)}/{len(idxs)}, {[round(dist * i / m, 3) for i in bad[:3]]} m from the start; {self._collision_blame(traj[bad[0]], ign)})"
        why_tail = self._policy_tail(traj, m, dist, arm, pose_from, pose_to, near_end, near_start, both, ignore_end, ignore_start)
        if why_tail:
            return None, why_tail
        ign = list(ignore_mid) + list(ignore_end)
        with self.ignoring(ign):
            end_free = bool(self.kit.collision_free(traj[-1], margin=0.0)[0])
        present = [o for o in touch if all(o is not x for x in ign)]
        if not end_free and present:
            # the program check's grasp approach is 3 segments: the first two end with the support in the world
            ok_t, pen_mm = self._finger_touch_ok(traj[-1:], arm, ign, present)
            end_free = bool(ok_t[0])
            if end_free:
                touched.append(pen_mm)
        if not end_free:
            return None, "approach_untrackable: the end configuration collides"
        if touched:
            self._count("finger_touch_allowed")
            ft = self.finger_touch
            opening = ft.opening() if ft.preshape else None
            log.info("approach finger-support touch allowed: max penetration %.0f mm (cap %.0f mm, margin %.0f mm; %s fingers vs %s; finger opening %s; %.3f m segment)",
                     max(touched), ft.max_penetration_m * 1000, ft.margin_m * 1000, arm, [getattr(o, "name", str(o)) for o in touch],
                     "configuration's own" if opening is None else f"{opening:.4f} m (pre-shaped)", dist)
        return traj[-1].clone(), "ok"

    def _policy_tail(self, traj, m: int, dist: float, arm: str, pose_from: Pose, pose_to: Pose, near_end, near_start, both,
                     ignore_end: Sequence[Any], ignore_start: Sequence[Any]) -> str:
        """of-v2maps 2026-11-07 (prompt I item 3; the isg1020 spec P2 §5: the policy applies wherever the support is ignored): the segment's
        configurations within ``APPROACH_WAYPOINT_M`` (3.3 cm) of a pose whose ignore set holds the support were collision-checked with
        the whole support removed and never policed (only the end pose was, in :meth:`exact`) - they are policed here like the end pose:
        the same switch / scope / holding rule, at the same policy opening. ``ISPATIALGYM_SUPPORT_POLICY_TAIL=0``: off. Returns the
        rejection reason, '' when the tail passes."""
        if os.environ.get("ISPATIALGYM_SUPPORT_POLICY_TAIL", "1") == "0" or not support_policy_on():
            return ""
        if support_policy_scope() == "grasp" and self._holding(arm):
            return ""
        jobs = []
        if ignore_end:
            jobs.append((sorted(set(near_end) | set(both)), list(ignore_end), pose_to))
        if ignore_start:
            jobs.append((sorted(set(near_start) | set(both)), list(ignore_start), pose_from))
        for idxs, sup, pose in jobs:
            if not idxs:
                continue
            appr = self.approached_objects(pose)
            if not policed_supports(sup, appr, planes=dict(getattr(self, "support_plane_z", None) or {})):
                continue
            for i in idxs:
                ev = self.support_policy_eval(traj[i], arm, sup, appr)
                if not bool(ev["ok"]):
                    self._count("support_policy_tail_rejected")
                    return (f"approach_untrackable: support policy on the approach tail ({ev.get('reason')}, "
                            f"{round(dist * min(i, m - i) / max(m, 1), 3)} m from the policed pose)")
            self._count("support_policy_tail_checked")
        return ""

    def _finger_touch_ok(self, qs: th.Tensor, arm: str, ignored: Sequence[Any], touch: Sequence[Any]) -> Tuple[th.Tensor, float]:
        """``(ok (K,) bool, max penetration in mm over the passing ones)`` of approach configurations that collide with ``touch`` (the grasp's
        support) while ``ignored`` is out of the world: (a) the whole body is collision-free at margin 0 with ``touch`` removed too (every
        other object, self-collision), (b) every link but ``arm``'s finger links is clear of the world at margin 0 (the palm, wrist, arm
        against the support), (c) the finger links - at the oracle's pre-shaped opening when ``FingerTouch.preshape`` - are clear of the world
        at ``margin_m - max_penetration_m`` (the light contact with the support; (a) already holds them clear of everything else)."""
        from ispatialgym.oracle import r1pro_model as M

        ft = self.finger_touch
        qs = G.t(qs)
        out = th.zeros(qs.shape[0], dtype=th.bool)
        fingers = list(M.FINGER_LINKS[arm])
        others = self._non_finger_links(arm)
        if ft is None or qs.shape[0] == 0 or not others:
            return out, 0.0
        with self.ignoring(list(ignored) + list(touch)):
            ok = self.kit.collision_free(qs, margin=0.0).clone()
        q_f = qs.clone()
        opening = ft.opening() if ft.preshape else None
        if opening is not None:
            q_f[:, self.kit.robot.gripper_control_idx[arm]] = float(opening)
        floor_m = ft.margin_m - ft.max_penetration_m
        with self.ignoring(list(ignored)):
            ok &= self.kit.links_clear(qs, others, 0.0)
            ok &= self.kit.links_clear(q_f, fingers, floor_m)
            pen_mm = 0.0
            if bool(ok.any()):
                # the deepest finger penetration over the passing configurations, to the mm (for the log)
                q_ok = q_f[ok]
                for k in range(0, int(math.ceil(ft.max_penetration_m * 1000)) + 1):
                    if bool(self.kit.links_clear(q_ok, fingers, max(floor_m, -0.001 * k)).all()):
                        pen_mm = float(k)
                        break
        return ok, pen_mm

    def _non_finger_links(self, arm: str) -> List[str]:
        """Every link with collision spheres in the whole-body model but ``arm``'s finger links ([] when the backend does not say)."""
        from ispatialgym.oracle import r1pro_model as M

        try:
            costs = self.kit.backend._collision_costs(self.kit.backend.kin_entry("whole_body"))
            links = sorted({l for l in costs.get("link_of_sphere", []) if l})
        except Exception:  # noqa: BLE001 - a backend without the sphere-to-link map: the rule cannot separate the fingers
            return []
        return [l for l in links if l not in set(M.FINGER_LINKS[arm])]

    #: the oracle's ``Motions._densify`` resolution of a tracked eef chain (``MotionsConfig.cartesian_step`` / ``cartesian_rot_step``)
    TRACK_STEP_M = 0.01
    TRACK_STEP_RAD = 0.05

    #: the oracle's ``ApproachFinish.max_excursion``: a finishing ``arm_to`` (the end of a stalled leg) may not swing any arm joint
    #: by more than this (rad) - a witness farther from the stalled configuration is in another IK branch and is refused there
    FINISH_MAX_EXCURSION = 0.6
    #: the oracle's ``ExecuteConfig.free_leg_finish_max_stall``: a free-space leg (lift / retreat / carry) whose differential IK
    #: stalls with a residual below this (m) is finished by ``arm_to``; a larger stall fails the leg
    FREE_LEG_STALL_M = 0.03
    #: the oracle's ``MotionsConfig`` differential-IK acceptance (``arm_along``): a waypoint left within the SOFT tolerances is
    #: accepted with its residual and the tracking goes on (``diff_ik_soft_pos_tol`` 1 cm / ``diff_ik_soft_rot_tol`` 0.03 rad,
    #: ``diff_ik_max_iters`` 100) - a continuation checked without them stalled where the oracle's lift goes on (block design
    #: 0925-050230: 4.1 mm at waypoint 5/9 of the lift, 8/8 attempts rejected)
    ORACLE_SOFT_POS_TOL = 0.01
    ORACLE_SOFT_ROT_TOL = 0.03
    ORACLE_DIFF_IK_MAX_ITERS = 100

    def reaches(self, q: th.Tensor, pose: Pose, arm: str) -> bool:
        """FK of ``q`` puts ``arm``'s eef at ``pose`` within the IK acceptance (5 mm / 0.05 rad); no collision check - the
        configuration is a verified end of an earlier motion, asked here only whether it *is* at the chain's first pose."""
        from ispatialgym.oracle import r1pro_model as M

        try:
            eef = self.kit.fk([M.EEF_LINK[arm]], G.t(q))[M.EEF_LINK[arm]]
        except Exception:  # noqa: BLE001
            return False
        return float(th.linalg.norm(G.t(eef[0]) - G.t(pose[0]))) <= self.WITNESS_POS_TOL and G.quat_angle(eef[1], G.t(pose[1])) <= self.WITNESS_ANG_TOL

    def continue_chain(self, q_start: th.Tensor, waypoints: Sequence[Pose], arm: str, ignore: Sequence[Any] = (), ignore_end: Sequence[Any] = (),
                       soft_stall_m: Optional[float] = None, limit_margin: float = 0.0, ignore_start: Sequence[Any] = (), ground_along: bool = True,
                       ground_meshes_along: bool = True, finish_max_remaining_m: Optional[float] = None) -> IKResult:
        """A chain that starts where the arm already **is** (owner 2026-09-24): ``q_start`` is the configuration the previous
        motion ended in (the place configuration for the retreat off it, the grasp configuration for the lift, the previous
        action's end for a ``Retreat``), and the chain is tracked from it by differential IK exactly as the oracle's
        ``arm_along`` will run it - the first pose is **not** re-solved (a 32-seed IK re-solve of the same pose lands in another
        branch at random, and the witnesses the oracle then aims its finish at lie 2-3 rad away: block design 0921-050358
        action 15, 'the sidecar's pre-pose and end-pose witnesses are in different IK branches'). A stall within
        ``soft_stall_m`` is finished by the IK of the end pose seeded at the stalled configuration and accepted only within
        :attr:`FINISH_MAX_EXCURSION` of it - the bound the oracle's finish applies. The solver runs with the oracle's own
        acceptance (soft waypoint tolerances, iteration budget; ``limit_margin`` as ``execute_retreat`` keeps the retreat off the
        joint limits); the first ``APPROACH_WAYPOINT_M`` off the start leave a contact and are checked with ``ignore_start`` (the
        start's support / the released object) removed and the ground slab off. Returns the tracked end configuration
        (``IKResult.q``); on failure ``last_track_stall`` (set by :meth:`track_chain`) tells the caller where the tracking stalled."""
        wps = list(waypoints)
        n = len(wps)
        self.last_track_stall = None
        if n == 0:
            return IKResult(True, G.t(q_start).clone(), "no poses", 0)
        if not self.reaches(q_start, wps[0], arm):
            return IKResult(False, None, "continuation: the start configuration does not reach the chain's first pose", n)
        if n == 1:
            return IKResult(True, G.t(q_start).clone(), "tracked continuation", 1)
        q_end, why = self.track_chain(q_start, wps, arm, ignore=ignore, ignore_end=ignore_end, soft_stall_m=soft_stall_m,
                                      finish_max_excursion=self.FINISH_MAX_EXCURSION, soft_pos_tol=self.ORACLE_SOFT_POS_TOL, soft_rot_tol=self.ORACLE_SOFT_ROT_TOL,
                                      max_iters=self.ORACLE_DIFF_IK_MAX_ITERS, limit_margin=float(limit_margin), ignore_start=list(ignore_start), ground_at_start=False,
                                      ground_along=ground_along, ground_meshes_along=ground_meshes_along, finish_max_remaining_m=finish_max_remaining_m)
        if q_end is None:
            return IKResult(False, None, why, n)
        return IKResult(True, q_end, "tracked continuation" + (" (stall finished)" if "finished" in why else ""), n)

    def track_chain(self, q_from: th.Tensor, waypoints: Sequence[Pose], arm: str, ignore: Sequence[Any] = (), ignore_end: Sequence[Any] = (),
                    soft_stall_m: Optional[float] = None, clear_links: Sequence[str] = (), clear_margin: float = 0.0,
                    finish_max_excursion: Optional[float] = None, soft_pos_tol: Optional[float] = None, soft_rot_tol: Optional[float] = None,
                    max_iters: int = 60, limit_margin: float = 0.0, ignore_start: Sequence[Any] = (), ground_at_start: bool = True,
                    ground_along: bool = True, ground_meshes_along: bool = True, refine_jumps: bool = False,
                    min_limit_margin: Optional[float] = None, finish_max_remaining_m: Optional[float] = None) -> Tuple[Optional[th.Tensor], str]:
        """Differential-IK tracking of a whole eef chain (a reorient: yaw-align orbit, in-hand flip) from ``q_from`` (which
        reaches ``waypoints[0]``) exactly as the oracle's ``arm_along`` tracks it - every segment densified by position
        (``TRACK_STEP_M``) AND rotation (``TRACK_STEP_RAD``), no joint step above ``APPROACH_MAX_STEP`` - the configurations
        collision-free with ``ignore`` removed (the last one with ``ignore_end`` removed too). ``track_segment`` densifies by
        position only, so a rotation in place (block design 0919-140558: the anchor's 11 deg yaw-align on a 45 deg pitched pinch,
        'joint jump 0.23 rad at waypoint 4' in the demo) was checked in 2 steps where the oracle takes 4 and drifts along a
        singular direction. Returns ``(q_end, "ok")`` or ``(None, reason)``.

        ``soft_stall_m``: the oracle finishes a free-space leg (lift, carry, descent, retreat) whose differential IK *stalls* -
        no joint jump, residual below this - by ``arm_to`` to the leg's end pose (``ApproachFinish``); with it set, such a stall
        is finished here by the exact IK of the end pose seeded from the last tracked configuration and the chain passes
        (``"ok (stall .. mm finished)"``). A joint jump or a collision fails the chain either way; a reorient (no finish in the
        oracle) is checked without it.

        ``refine_jumps``: a chain failing on a joint jump is tracked once more at half the dense spacing, as ``arm_along`` does
        (``Motions.arm_along(refine_jumps=True)``). ``min_limit_margin`` (rad): every tracked configuration after the start must
        keep this much room to every arm joint limit (``joint_limit:`` otherwise) - the oracle's differential IK freezes a joint
        that would cross its limit and swings the others in its place (mental_rotation-walker 2026-10-13: right_arm_joint6 held
        8 mrad from its lower limit for 36 of the orbit's 50 steps, joints 3 / 7 turning 1 rad each, the robot stuttered)."""
        from ispatialgym.oracle import r1pro_model as M

        wps = [(G.t(w[0]), G.t(w[1])) for w in waypoints]
        if len(wps) < 2:
            return G.t(q_from).clone(), "ok"
        dense: List[Pose] = []
        for a, b in zip(wps[:-1], wps[1:]):
            dist = float(th.linalg.norm(b[0] - a[0]))
            ang = float(C.quat_angle(a[1], b[1]))
            n = max(1, int(math.ceil(max(dist / self.TRACK_STEP_M, ang / self.TRACK_STEP_RAD))))
            dense.extend(tuple(w) for w in C.line_waypoints(a, b, n))
        self.kit._ensure_synced()
        extra_kw: Dict[str, Any] = {}
        if soft_pos_tol is not None:
            extra_kw["soft_pos_tol"] = float(soft_pos_tol)
        if soft_rot_tol is not None:
            extra_kw["soft_rot_tol"] = float(soft_rot_tol)
        if limit_margin > 0.0:
            extra_kw["limit_margin"] = float(limit_margin)

        def solve(poses):
            with self.ignoring(list(ignore)):
                return self._solve_ik_path(f"{arm}_arm", [{M.EEF_LINK[arm]: (G.t(w[0]), G.t(w[1]))} for w in poses], q_start=G.t(q_from),
                                           check_collisions=False, sync_world=False, max_joint_step=self.APPROACH_MAX_STEP, max_iters=int(max_iters), **extra_kw)

        res = solve(dense)
        if refine_jumps and not res.success and str(res.status).startswith("joint jump"):
            finer: List[Pose] = []
            prev = wps[0]
            for w in dense:
                finer.extend(tuple(x) for x in C.line_waypoints(prev, w, 2))
                prev = w
            res2 = solve(finer)
            if res2.success:
                res, dense = res2, finer
        self.kit.stats["ik_calls"] = self.kit.stats.get("ik_calls", 0) + 1
        finished = ""
        self.last_track_stall = None
        if not res.success:
            diag = dict(getattr(res, "diagnostics", None) or {})
            stall = diag.get("pos_err")
            if diag.get("stage") == "diff_ik" and stall is not None and res.q_traj is not None and len(res.q_traj) > 0:
                # where the tracking stalled (the caller's partial-retreat rule mirrors the oracle's acceptance of a short retreat)
                q_stall = res.q_traj[-1].clone()
                with self.ignoring(list(ignore)):
                    stall_free = bool(self.kit.collision_free(q_stall, margin=0.0)[0])
                self.last_track_stall = {"pos_err_m": float(stall), "fail_index": diag.get("fail_index"), "n_dense": len(dense), "q": q_stall,
                                         "remaining_m": float(th.linalg.norm(wps[-1][0] - G.t(self.kit.fk([M.EEF_LINK[arm]], q_stall)[M.EEF_LINK[arm]][0]))), "free": stall_free,
                                         "at_limit": bool(diag.get("at_lower") or diag.get("at_upper"))}
            if soft_stall_m is not None and diag.get("stage") == "diff_ik" and stall is not None and float(stall) <= float(soft_stall_m) and res.q_traj is not None and len(res.q_traj) > 0:
                # the oracle's finish: plan to the end pose from where the tracking stalled (here: its exact IK seeded there)
                q_last = res.q_traj[-1].clone()
                rem = (self.last_track_stall or {}).get("remaining_m")
                if finish_max_remaining_m is not None and rem is not None and float(rem) > float(finish_max_remaining_m):
                    # oracle-fix 2026-11-05 (generation.carry_finish_max_remaining_m; hardback hbD7: the carry at C stalled at 8/53 with 0.38 m left -
                    # accepted here by the IK finish, the oracle's planned finish next to the bookcase failed): a long remainder is no finish
                    return None, (f"approach_untrackable: chain tracking ({len(dense)} dense poses): {res.status}; the stall leaves {float(rem):.2f} m "
                                  f"(> {float(finish_max_remaining_m):.2f} m the oracle's planned finish is trusted with)")
                with self.ignoring(list(ignore) + list(ignore_end)):
                    q_fin = self.kit.ik(wps[-1], q_last, arm=arm)
                if q_fin is None:
                    return None, f"approach_untrackable: chain tracking ({len(dense)} dense poses): {res.status}; the end pose has no IK from the stall either"
                if finish_max_excursion is not None:
                    idx = self.kit.robot.arm_control_idx[arm]
                    exc = float(th.max(th.abs(G.t(q_fin)[idx] - G.t(q_last)[idx])))
                    if exc > float(finish_max_excursion):
                        # the oracle refuses such a finish ("the plan swings a joint by X rad (> 0.60 allowed for a finishing motion)")
                        return None, (f"approach_untrackable: chain tracking ({len(dense)} dense poses): {res.status}; the end pose's IK from the stall lies "
                                      f"{exc:.2f} rad away in joint space (> {float(finish_max_excursion):.2f}: another IK branch, the oracle's finish would be rejected)")
                traj = th.cat([res.q_traj, q_fin.unsqueeze(0)], dim=0)
                finished = f" (stall {float(stall) * 1000:.1f} mm at dense pose {diag.get('fail_index')} finished by IK)"
            else:
                return None, f"approach_untrackable: chain tracking ({len(dense)} dense poses): {res.status}"
        else:
            traj = res.q_traj
        if min_limit_margin is not None:
            for j in range(1, len(traj)):
                m_j, jn = self.joint_limit_margin(traj[j], arm)
                if m_j < float(min_limit_margin):
                    return None, (f"joint_limit: chain tracking ({len(dense)} dense poses): {jn} {m_j * 1000:.1f} mrad from its limit at dense pose {j} "
                                  f"(< {float(min_limit_margin) * 1000:.0f} mrad)")
        # the configurations within APPROACH_WAYPOINT_M of the START leave a contact (the grasped object still on its support, the
        # pads around the object just released): checked with ``ignore_start`` removed and the ground slab off, as ``track_segment``
        # checks the first centimetres of a leg (the chains that began 3.3 cm past their start never saw them: IORS floor class
        # 0925-115205, 'lift: 1/11 tracked chain configurations collide' - the dispenser's spheres in the slab 1 cm off the floor)
        mid = list(range(1, len(traj) - 1))
        near = [j for j in mid if j - 1 < len(dense) and float(th.linalg.norm(G.t(dense[j - 1][0]) - wps[0][0])) <= self.APPROACH_WAYPOINT_M] if (ignore_start or not ground_at_start) else []
        far = [j for j in mid if j not in set(near)]
        bad: List[int] = []
        if far:
            # ``ground_along=False``: the ground slab (top at z = 0) is off for the whole chain - a FLOOR-class lift carries an object
            # whose sphere fit reaches below its mesh bottom (shampoo dispenser 0925-171050: 'lift: 1/11 tracked chain configurations
            # collide', the one configuration just past the start band, the dispenser's spheres still in the slab 3-4 cm up); the
            # floor mesh itself is in ``ignore`` and the oracle's lift is a contact motion without a collision veto
            with self.ignoring(list(ignore), ground=ground_along, ground_meshes=ground_meshes_along):
                free = self.kit.collision_free(traj[far], margin=0.0)
            bad += [j for j, f in zip(far, free.tolist()) if not f]
        if near:
            with self.ignoring(list(ignore) + list(ignore_start), ground=ground_at_start):
                free_n = self.kit.collision_free(traj[near], margin=0.0)
            bad += [j for j, f in zip(near, free_n.tolist()) if not f]
        if bad:
            bad.sort()
            j0 = bad[0]
            try:
                d0 = float(th.linalg.norm(G.t(dense[j0 - 1][0]) - wps[0][0])) if j0 - 1 < len(dense) else float("nan")
            except Exception:  # noqa: BLE001
                d0 = float("nan")
            return None, (f"approach_untrackable: {len(bad)}/{len(mid)} tracked chain configurations collide (first: dense pose {j0} of {len(traj) - 1}, "
                          f"{d0 * 1000:.0f} mm from the start{', in the start band' if j0 in near else ''}; {self._collision_blame(traj[j0], list(ignore) + (list(ignore_start) if j0 in near else []))})")
        with self.ignoring(list(ignore) + list(ignore_end)):
            if not bool(self.kit.collision_free(traj[-1], margin=0.0)[0]):
                return None, "approach_untrackable: the tracked chain's end configuration collides"
        if clear_links and clear_margin > 0.0:
            # ``clear_links`` (the arm links, not the hand) keep ``clear_margin`` from the world along the whole chain: the oracle's QA
            # fails a grasp on any arm-table contact (block design 0919-171511: right_arm_link5 / 4 brushed the table edge during
            # the pitched approach, the end configurations 12 mm clear)
            with self.ignoring(list(ignore) + list(ignore_end)):
                clear = self.kit.links_clear(traj, clear_links, clear_margin)
            if not bool(clear.all()):
                bad = [i for i, c in enumerate(clear.tolist()) if not c]
                return None, f"arm_proximity: {len(bad)}/{len(clear)} tracked configurations put an arm link within {clear_margin * 1000:.0f} mm of the world (first at dense pose {bad[0]})"
        return traj[-1].clone(), "ok" + finished

    def _collision_blame(self, q: th.Tensor, ignored: Sequence[Any], margin: float = 0.0) -> str:
        """Diagnostic: which spawned object (or 'scene / self') a colliding configuration collides with — each spawned
        object is removed in turn on top of ``ignored``. ``margin`` must be the failing check's margin: at a smaller one
        the configuration may already be free and the first spawned object gets the blame (0921-005514: 'collides with
        target_sandal' for every drive-back pose of a 4 cm carried sweep whose first sample was free at margin 0)."""
        world = getattr(self.kit, "world", None)
        spawned = list(getattr(world, "spawned", {}).values()) if world is not None else []
        with self.ignoring(list(ignored)):
            if bool(self.kit.collision_free(q, margin=float(margin))[0]):
                return f"free at margin {float(margin):.2f} (the failing check used another margin or world)"
        for o in spawned:
            if any(o is x for x in ignored):
                continue
            with self.ignoring(list(ignored) + [o]):
                if bool(self.kit.collision_free(q, margin=float(margin))[0]):
                    return f"collides with {getattr(o, 'name', o)}" + (f" (within {float(margin):.2f} m)" if margin > 0 else "")
        try:  # world-only check: passing here means the robot (or its attached object) collides with itself
            self.kit._ensure_synced()
            with self.ignoring(list(ignored)):
                world_only = bool(self.kit.backend.check_collision(G.t(q).unsqueeze(0), emb_name="whole_body", sync_world=False, self_collision=False, margin=float(margin))[0])
            if not world_only:
                pairs = self.self_collision_pairs(q)
                named = ", ".join(f"{a} x {b} ({pen * 1000:.0f} mm)" for (a, b), pen in pairs[:3]) or "arm / attached object against the robot's own links"
                return f"self-collision ({named})"
            # the cuRobo ground slab (a cuboid whose top is the floor plane, not a scene object): re-check with it off (owner 2026-09-25,
            # shampoo 0925-171050 'lift: 1/11 tracked chain configurations collide (collides with the scene ...)': the carried dispenser's
            # sphere fit reaches below its mesh into the slab)
            with self.ignoring(list(ignored), ground=False):
                if bool(self.kit.collision_free(q, margin=float(margin))[0]):
                    return "collides with the cuRobo ground slab only (top at the floor plane; the floor mesh is ignored or clear)"
            # a non-spawned scene mesh within the margin: name the scene objects whose AABB lies nearest the base pose
            # (a heuristic - the padded proximity is not attributed per mesh by the checker)
            if world is not None and hasattr(world, "scene"):
                from ispatialgym.oracle import r1pro_model as M
                base = self.kit.fk([M.BASE_LINK], G.t(q))[M.BASE_LINK][0]
                rows = []
                for o in getattr(world.scene, "objects", []):
                    try:
                        lo, hi = o.aabb
                    except Exception:  # noqa: BLE001
                        continue
                    dx = max(float(lo[0]) - float(base[0]), 0.0, float(base[0]) - float(hi[0]))
                    dy = max(float(lo[1]) - float(base[1]), 0.0, float(base[1]) - float(hi[1]))
                    d = math.hypot(dx, dy)
                    if d < 2.5 and "floor" not in o.name and "ceiling" not in o.name:
                        rows.append((d, o.name))
                rows.sort()
                # name the scene object (owner 2026-09-25, shampoo 0925-200239: floor, lawn and slab all off, still 'collides with the
                # scene'): the nearest scene objects are removed in turn - the first whose removal frees the configuration is the body
                by_name = {o.name: o for o in getattr(world.scene, "objects", [])}
                for d, n in rows[:20]:
                    o = by_name.get(n)
                    if o is None or o is getattr(self.kit, "robot", None) or "robot" in n:
                        continue
                    with self.ignoring(list(ignored) + [o]):
                        if bool(self.kit.collision_free(q, margin=float(margin))[0]):
                            return f"collides with the scene object {n} ({d:.2f} m from the base)"
                # which of the robot's (or the attached object's) links touch the world: the sphere owners at this configuration
                link_hits = []
                try:
                    costs = self.kit.backend._collision_costs(self.kit.backend.kin_entry("whole_body"))
                    for link in sorted(set(costs.get("link_of_sphere", []))):
                        with self.ignoring(list(ignored)):
                            if not bool(self.kit.links_clear(q, [link], float(margin))[0]):
                                link_hits.append(link)
                except Exception:  # noqa: BLE001
                    pass
                hits = f"; links in the world: {', '.join(link_hits)}" if link_hits else ""
                # per world body (owner 2026-09-25): the planner disables each enabled mesh in turn and re-checks - the meshes whose
                # removal clears the penetration name the body, whatever scene object (or none) owns them
                try:
                    with self.ignoring(list(ignored)):
                        offs = self.kit.backend.world_collision_offenders(G.t(q), emb_name="whole_body", limit=4)
                    if offs:
                        hits += f"; world meshes in contact: {[str(o).rsplit('/', 2)[-2:] for o in offs]}"
                except Exception as e:  # noqa: BLE001
                    hits += f"; per-mesh search failed ({e!r})"
                try:  # the attached object's sphere fit (a coarse fit bulges far beyond the mesh)
                    for arm_, att in (getattr(self.kit.backend, "_attached", None) or {}).items():
                        sp = att.get("spheres")
                        if sp is not None:
                            hits += f"; attached({arm_}) spheres r={[round(float(r), 3) for r in G.t(sp)[:, 3].tolist()]}"
                except Exception:  # noqa: BLE001
                    pass
                if rows:
                    return "collides with the scene within %.2f m; nearest scene objects to the base: %s%s" % (float(margin), ", ".join(f"{n} ({d:.2f} m)" for d, n in rows[:4]), hits)
        except Exception:  # noqa: BLE001
            pass
        return "collides with the scene (a non-spawned object)" + (f" within {float(margin):.2f} m" if margin > 0 else "")

    def self_collision_pairs(self, q: th.Tensor) -> List[Tuple[Tuple[str, str], float]]:
        """cuRobo's overlapping self-collision link pairs at ``q`` (``backend.self_collision_pairs``, A7 relay 2026-09-16 21:07):
        ``[((link_a, link_b), penetration_m), ...]`` worst first; ``attached_object_<arm>_eef_link`` names the carried object."""
        fn = getattr(self.kit.backend, "self_collision_pairs", None)
        if fn is None:
            return []
        try:
            out = fn(G.t(q), "whole_body")
        except Exception:  # noqa: BLE001
            return []
        return [(tuple(d["links"]), float(d["penetration"])) for d in out]

    @staticmethod
    def pairs_involve_attached(pairs: Sequence[Tuple[Tuple[str, str], float]]) -> bool:
        return any("attached_object" in a or "attached_object" in b for (a, b), _ in pairs)

    def track_approach(self, q_pre: th.Tensor, pre: Pose, end: Pose, arm: str, ignore_end: Sequence[Any] = (), ignore_mid: Sequence[Any] = (),
                       q_hint: Optional[th.Tensor] = None, touch: Sequence[Any] = ()) -> Tuple[Optional[th.Tensor], str]:
        """End-pose configuration of a straight approach, branch-consistent with the pre-pose configuration ``q_pre``:
        the segment is tracked forward from ``q_pre``; when that fails the end pose is solved on its own (32 seeds around
        ``q_hint``) and the segment tracked *backward* to the pre pose — the pre configuration is then replaced by the
        end's branch (``self._last_track_start``, recorded by the caller as the pre-pose witness). ``(None, reason)`` when
        neither branch tracks (``approach_untrackable``). ``touch``: :meth:`track_segment`'s finger-support touch objects."""
        self._last_track_start = G.t(q_pre).clone()
        tkw = {"touch": list(touch)} if touch else {}
        q_end, why = self.track_segment(q_pre, pre, end, arm, ignore_mid=ignore_mid, ignore_end=ignore_end, **tkw)
        if q_end is not None:
            return q_end, "ok"
        seed = q_hint if q_hint is not None else q_pre
        with self.ignoring(list(ignore_mid) + list(ignore_end)):
            q_alt = self.kit.ik(end, self._seeded(G.t(seed), end, arm), arm=arm)
        if q_alt is None:
            return None, f"{why}; the end pose has no solution on its own either"
        q_pre2, why2 = self.track_segment(q_alt, end, pre, arm, ignore_mid=ignore_mid, ignore_end=(), ignore_start=ignore_end, **tkw)
        if q_pre2 is None:
            return None, f"{why}; from the end's branch: {why2}"
        self._last_track_start = q_pre2.clone()
        return q_alt, "ok"

    def exact_both_arms(self, work: Tuple[str, Sequence[Pose]], other: Tuple[str, Sequence[Pose]], stance: Stance, ignore_for_last: Sequence[Any] = (),
                        ignore_other_last: Sequence[Any] = (), fingers: Optional[float] = None,
                        brace_only_check: Optional[Tuple[Sequence[Any], float]] = None, q_first_other: Optional[th.Tensor] = None) -> IKResult:
        """Brace / second-arm rule (§4.6 step 6): the working arm's poses first; then the other arm with the working arm
        locked at its solution; combined whole-body check of the final configuration **with the objects the two final
        poses touch removed** (``ignore_for_last`` ∪ ``ignore_other_last``: the support and the pushed / braced object —
        §3.6 "ignored for the contact pose only"; with them in the world every pad-on-face contact read "combined
        configuration collides", which is why ``tabletop.brace_second_arm`` bypassed this method). ``fingers`` = gripper
        opening of both hands (closed for push + brace contacts). ``brace_only_check = (ignored objects, margin)``: an
        extra check of the brace arm's final configuration with the working arm back at the stance posture against
        everything but the ignored objects at ``margin`` (a stop blade against the target at its goal pose while the
        push hand, which touches the target, is out of the picture). ``q_first_other``: the other arm's chain starts from this
        arm configuration (``exact``'s ``q_first``; a wrist-flipped solution, ``orient``)."""
        arm_w, poses_w = work
        arm_o, poses_o = other
        r1 = self.exact(poses_w, stance, arm_w, ignore_for_last, cache=False, fingers=fingers)
        if not r1.ok:
            return IKResult(False, None, f"{arm_w}: {r1.reason}")
        q_seed = r1.q.clone()
        q_f = None
        if q_first_other is not None:
            q_f = q_seed.clone()
            idx_o = self.kit.robot.arm_control_idx[arm_o]
            q_f[idx_o] = G.t(q_first_other)[idx_o]
        r2 = self.exact(poses_o, stance, arm_o, ignore_other_last, q_seed=q_seed, cache=False, fingers=fingers, q_first=q_f)
        if not r2.ok:
            return IKResult(False, None, f"{arm_o} (brace): {r2.reason}")
        touched = list(ignore_for_last) + [o for o in ignore_other_last if all(o is not w for w in ignore_for_last)]
        with self.ignoring(touched):
            free = bool(self.kit.collision_free(r2.q, margin=0.01)[0])
        if not free:
            return IKResult(False, None, "combined configuration collides")
        if brace_only_check is not None:
            ignored, margin = brace_only_check
            q_b = self.q_locked(stance, fingers=fingers).clone()
            idx = self.kit.robot.arm_control_idx[arm_o]
            q_b[idx] = r2.q[idx]
            with self.ignoring(list(ignored)):
                ok_b = bool(self.kit.collision_free(q_b, margin=float(margin))[0])
            if not ok_b:
                return IKResult(False, None, f"{arm_o} (brace) intersects the target at its goal pose (margin {float(margin) * 1000:.0f} mm)")
        return IKResult(True, r2.q, "", len(poses_w) + len(poses_o))

    # -- waypoints ----------------------------------------------------------------------------------------------
    def waypoints_ok(self, waypoints: Sequence[Pose], stance: Stance, arm: str = "right", policy: str = "screen_mid_exact_ends",
                     ignore_for_last: Sequence[Any] = (), ignore_all: Sequence[Any] = (), q_seed: Optional[th.Tensor] = None,
                     scale_key: Any = None, cache: bool = True, fingers: Optional[float] = None, approached: Optional[Sequence[Any]] = None,
                     track_dense: bool = False, q_first: Optional[th.Tensor] = None) -> IKResult:
        """§3.9: screening IK at the intermediate waypoints, exact IK at the two endpoints (``policy`` names what was
        checked and is recorded as ``waypoint_policy``). ``ignore_all`` objects are removed for every waypoint (the
        carried object's own support along a lift), ``ignore_for_last`` for the final pose only. ``cache=False`` for
        re-checks that must not hit a cached verdict (program check, ``validate_instances``).

        ``track_dense`` (block design, owner 2026-09-19): a chain that is not a straight approach (a reorient) is additionally
        tracked from the first pose's solution by :meth:`track_chain` at the oracle's densification, so the recorded chain
        is one the oracle's ``arm_along`` follows without a joint jump; the returned configuration is the tracked end.

        ``q_first``: the recorded configuration of the first pose (:meth:`exact`) - the chain leaves from it."""
        wps = list(waypoints)
        if len(wps) < 2:
            return self.exact(wps, stance, arm, ignore_for_last, q_seed=q_seed, scale_key=scale_key, cache=cache, fingers=fingers, approached=approached, q_first=q_first)
        if all(is_approach(a, b) for a, b in zip(wps[:-1], wps[1:])):
            # a straight same-orientation chain (approach, lift, descent, retreat, the wall brace chain): solved by
            # ``exact`` as one tracked motion from the first pose's solution (branch-consistent witnesses at every
            # waypoint; ``approach_untrackable`` otherwise) with ``ignore_all`` removed throughout
            with self.ignoring(list(ignore_all)):
                res = self.exact(wps, stance, arm, list(ignore_for_last), q_seed=q_seed, scale_key=scale_key, cache=False, fingers=fingers, approached=approached, q_first=q_first)
            return IKResult(res.ok, res.q, policy if res.ok else res.reason, len(wps))
        # endpoints first (exact): their solutions seed the screening of the intermediate waypoints, which otherwise
        # misses poses near the ends that the 32-seed solver finds (the wall brace chain)
        r0 = self.exact([wps[0]], stance, arm, list(ignore_all), q_seed=q_seed, scale_key=scale_key, cache=cache, fingers=fingers, q_first=q_first)
        if not r0.ok:
            return IKResult(False, None, "first waypoint fails the exact IK", len(wps))
        r1 = self.exact([wps[-1]], stance, arm, list(ignore_for_last) + list(ignore_all), q_seed=q_seed, scale_key=scale_key, cache=cache, fingers=fingers)
        if not r1.ok:
            return IKResult(False, None, "last waypoint fails the exact IK", len(wps))
        mids = wps[1:-1]
        if mids:
            ok = self.screen(mids, stance, arm, ignore=list(ignore_all), q_seed=q_seed, fingers=fingers, seeds=[r0.q, r1.q])
            if not bool(ok.all()):
                bad = [i + 1 for i, s in enumerate(ok.tolist()) if not s]
                return IKResult(False, None, f"waypoints {bad} of {len(wps)} fail the screening IK", len(wps))
        if track_dense:
            q_end, why = self.track_chain(r0.q, wps, arm, ignore=list(ignore_all), ignore_end=list(ignore_for_last))
            if q_end is None:
                return IKResult(False, None, why, len(wps))
            self.solutions[(arm, self._pose_key(wps[-1]))] = q_end.clone()
            return IKResult(True, q_end, policy + "+tracked", len(wps))
        return IKResult(True, r1.q, policy, len(wps))

    # -- attached objects ---------------------------------------------------------------------------------------
    def attachable(self, obj, eef_in_object: Pose, arm: str = "right") -> bool:
        """Whether the object, attached to ``arm`` per the grasp transform ``T_obj→eef``, is free of **self**-collision
        with the robot in an untucked free-space configuration (the attached spheres vs the palm / wrist, which the
        robot model does not exempt — only the finger links are). A grasp that fails here fails every attached check
        (lift, carry, place) whatever the pose, so task modules drop it from the grasp set once per template."""
        from ispatialgym.sampling.v2.navigation import attached, attached_checks_enabled

        if not attached_checks_enabled():
            return True
        q = self.kit.q_at_base_pose(self.kit.q_config(arms="untucked", translate=0.5), *self.kit.world_base_pose2d())
        with attached(self.kit, arm, obj, eef_in_object=eef_in_object):
            self.kit._ensure_synced()
            with_self = bool(self.kit.backend.check_collision(q.unsqueeze(0), emb_name="whole_body", sync_world=False, self_collision=True, margin=0.01)[0])
            without = bool(self.kit.backend.check_collision(q.unsqueeze(0), emb_name="whole_body", sync_world=False, self_collision=False, margin=0.01)[0])
        return not (with_self and not without)

    # -- gripper visibility -------------------------------------------------------------------------------------
    def eef_in_frame(self, pose: Pose, camera_pose: Pose, margin: float = IN_FRAME_MARGIN, eef_extent: float = 0.06) -> bool:
        """§3.9 gripper in FOV: the eef body (a box of half-extent ``eef_extent`` around the eef origin, fingertips
        along +a) projects inside the image of ``camera_pose`` with ``margin`` (fraction of the image size)."""
        p, q = G.t(pose[0]), G.t(pose[1])
        pts = []
        for dx in (-eef_extent, eef_extent):
            for dy in (-eef_extent, eef_extent):
                for dz in (-0.02, 0.06):
                    pts.append(p + G.rotate(q, th.tensor([dx, dy, dz])))
        return self.kit.world.camera_model.all_inside(camera_pose, th.stack(pts), margin)

    def eef_frame_margin(self, pose: Pose, camera_pose: Pose, margin: float = IN_FRAME_MARGIN, eef_extent: float = 0.06) -> float:
        """oracle-fix 2026-11-07 (owner prompt E, frame ranking): how far :meth:`eef_in_frame`'s eef box sits inside the image border of
        ``margin`` - the smallest signed distance (px) of its 8 corners to that border (negative: outside it; ``-inf`` behind the camera).
        ``eef_in_frame`` <=> margin >= 0."""
        p, q = G.t(pose[0]), G.t(pose[1])
        pts = []
        for dx in (-eef_extent, eef_extent):
            for dy in (-eef_extent, eef_extent):
                for dz in (-0.02, 0.06):
                    pts.append(p + G.rotate(q, th.tensor([dx, dy, dz])))
        cm = self.kit.world.camera_model
        uv, depth = cm.project(camera_pose, th.stack(pts))
        if bool((depth <= 0.05).any()):
            return float("-inf")
        mu, mv = margin * cm.width, margin * cm.height
        d = th.stack([uv[:, 0] - mu, cm.width - mu - uv[:, 0], uv[:, 1] - mv, cm.height - mv - uv[:, 1]], dim=1)
        return float(d.min())

    def camera_pose(self, stance: Stance) -> Pose:
        """Head-camera pose at a stance through the shared FK (``conventions.camera_pose``, base on the measured floor, the
        kit's measured sensor offset) — the same model the oracle's FOV monitor projects with."""
        q = self.q_locked(stance)
        x, y, yaw = (float(v) for v in stance.base_pose2d)
        base = (th.tensor([x, y, float(self.kit.floor_z) + C.NOMINAL_BASE_Z]), G.quat_from_yaw(yaw))
        return C.camera_pose(q, base_pose=base, cam_in_zed=self.kit.cam_in_zed)

    # -- bands: feasible sets ------------------------------------------------------------------------------------
    def feasible_sets(self, cells: Sequence[Any], yaws: Sequence[float], grasps: Sequence[Any], pose_at: Callable[..., Pose], stance: Stance,
                      arm: str = "right", ignore_for_grasp: Sequence[Any] = (), eef_of_grasp: Optional[Callable[[Any], Pose]] = None,
                      scale_key: Any = None, cache: bool = True, pose_per_grasp: bool = False, pre_of: Optional[Callable[[Pose], Pose]] = None,
                      pre_tag: Optional[str] = None) -> Dict[int, List[Tuple[float, Any]]]:
        """§4.6 step 1 over many cells in one batch per trunk: for every cell index the feasible set F of ``(yaw, grasp)``
        pairs whose [pre-grasp, grasp] pass the screening IK from ``stance`` (grasp pose with ``ignore_for_grasp``
        removed). ``pose_at(cell, yaw) -> object pose`` (``pose_per_grasp``: ``pose_at(cell, yaw, grasp)``, an object pose that
        depends on the grasp too - the inside release at the drawn grasp's own height); ``eef_of_grasp(grasp) -> T_obj→eef``
        (default ``grasp.pose``). Cached (§3.10 band cache of F) under the stance, cells, yaws, grasp set, object poses at yaw 0
        (per grasp with ``pose_per_grasp``) and ``scale_key``; a cell with empty F is absent from the result (the caller removes it
        from the band)."""
        eef_of = eef_of_grasp or (lambda g: g.pose)
        by_name = {getattr(g, "name", str(k)): g for k, g in enumerate(grasps)}
        key = None
        if cache:
            poses0 = ([list(G.t(p[0]).tolist()) + list(G.t(p[1]).tolist()) for c in cells for p in (pose_at(c, 0.0, g) for g in grasps)] if pose_per_grasp else
                      [list(G.t(pose_at(c, 0.0)[0]).tolist()) + list(G.t(pose_at(c, 0.0)[1]).tolist()) for c in cells])
            key = cache_key("feasible_sets_v2", {**self.cache_scope, "base": list(stance.base_pose2d), "trunk": [stance.translate, stance.tilt], "arm": arm,
                                                 "cells": [list(G.t(c).tolist()) for c in cells], "yaws": list(yaws), "scale": scale_key,
                                                 "grasps": [(getattr(g, "key", None), getattr(g, "name", None), list(G.t(eef_of(g)[0]).tolist()), list(G.t(eef_of(g)[1]).tolist())) for g in grasps],
                                                 "poses0": poses0,
                                                 "ignore": sorted(getattr(o, "name", str(o)) for o in ignore_for_grasp), "seeds": SCREEN_SEEDS,
                                                 **({"pose_per_grasp": True} if pose_per_grasp else {}), **({"pre": pre_tag or "custom"} if pre_of is not None else {})})
            hit = cache_get(key)
            if hit is not None:
                self._count("feasible_sets.hit")  # oracle-fix 2026-11-05 (speed): hits / misses in the attempt's kit line
                return {int(i): [(float(yaw), by_name[gn]) for yaw, gn in lst if gn in by_name] for i, lst in hit.items()}
            self._count("feasible_sets.miss")
        pairs, index = [], []
        for i, cell in enumerate(cells):
            for yaw in yaws:
                obj_pose = None if pose_per_grasp else pose_at(cell, yaw)
                for g in grasps:
                    eef = grasp_pose(pose_at(cell, yaw, g) if pose_per_grasp else obj_pose, eef_of(g))
                    pairs.append([(pre_of or pre_pose)(eef), eef])  # pre_of: the release's own pre-place (oracle-fix 2026-11-04 round 3)
                    index.append((i, yaw, g))
        feas = self.screen_pairs(pairs, stance, arm, ignore_for_last=ignore_for_grasp)
        out: Dict[int, List[Tuple[float, Any]]] = {}
        for (i, yaw, g), s in zip(index, feas.tolist()):
            if s:
                out.setdefault(i, []).append((yaw, g))
        if key is not None:
            cache_put(key, {str(i): [(yaw, getattr(g, "name", None)) for yaw, g in lst] for i, lst in out.items()})
        return out
