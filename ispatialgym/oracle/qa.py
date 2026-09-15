"""Real-time QA of the v2 executor (``oracle_primitives_design.md`` §6.4; TODO F3).

Verdict categories (the executor's verdict / ``SkillError.reason``; this module fixes the final list):

==========================  ======================================================================================
``not_reached``             the binary goal did not hold at the end of the program (epilogue)
``tolerance``               a measured error is above its tolerance (epilogue), or an action left a task quantity
                            (camera pose, joint value) off its recorded target
``not_settled``             robot or object motion after the final no-op window (epilogue), or a placed object
                            still moving
``collision``               a robot link touched an object other than the intended contacts of the action
``ghost_hand``              assisted-grasp artefact: the held object drifted off the hand while still "in hand"
``paused``                  the robot stood still for longer than the pause budget inside an action
``gripper_out_of_fov``      an acting gripper left the head camera's frame (5 % margin) during a manipulation action
``failed_grasp``            the assisted grasp did not attach (grasp actions; push / brace exempt)
``no_motion``               a push / articulation moved nothing
``plan_failed``             trajopt (or the differential IK along recorded waypoints) found no trajectory
``execution``               a motion did not converge / an action did not reach its recorded target
``timeout``                 the per-action or per-episode step budget was exhausted
``targets_failed``          ``action.targets()`` raised (a sidecar / conventions-side problem, kept apart from
                            ``plan_failed`` so the sampler-side cause stays distinguishable)
``error``                   an unexpected exception (traceback in the run's ``meta.json``)
==========================  ======================================================================================

The per-step monitors (:class:`Monitors`) watch every ``env.step`` of an action: collision (``RigidContactAPI``
pairs of the robot — and of the carried objects outside manipulation actions — against everything but the action's
``objects``, the held objects and the floor categories; the finger links of the acting arms are exempt during
manipulation actions, their contacts are what manipulation is; a contact must persist ``collision_min_steps``
steps), ghost hand (object-in-eef pose drift beyond ``ghost_pos`` / ``ghost_rot`` while the assisted grasp says
"in hand"), prolonged pause (every joint velocity below ``pause_vel`` for ``pause_steps``), gripper in FOV
(fingertips of the acting arms projected into the head camera through the conventions' ``camera_pose`` FK and the
pinhole model of spec §3.4, only during manipulation actions, ``fov_min_steps`` consecutive frames) and the step
budgets. ``QAStats`` counts verdicts per category × action name (§6.4: a sampler-side and a planner-side cause of the
same symptom stay distinguishable).

Simulator-free: ``env`` / ``robot`` are duck-typed; the contact API is imported lazily.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.robot import conventions as C

log = logging.getLogger(__name__)

CATEGORIES: Tuple[str, ...] = (
    "not_reached",
    "tolerance",
    "not_settled",
    "collision",
    "ghost_hand",
    "paused",
    "gripper_out_of_fov",
    "failed_grasp",
    "no_motion",
    "plan_failed",
    "execution",
    "timeout",
    "targets_failed",
    "error",
)
#: categories the per-step monitors can raise
STEP_CATEGORIES: Tuple[str, ...] = ("collision", "ghost_hand", "paused", "gripper_out_of_fov", "timeout")
#: categories of the epilogue
EPILOGUE_CATEGORIES: Tuple[str, ...] = ("not_reached", "tolerance", "not_settled")

#: contacts with these object categories never count (the robot stands / drives on them)
GROUND_CATEGORIES = frozenset({"floors", "carpet", "rug", "mat", "gym_mat", "yoga_mat", "lawn", "driveway", "ceramic_tile"})


@dataclass
class Verdict:
    """One QA failure: the category, a message and where it happened."""

    category: str
    message: str = ""
    action: Optional[str] = None
    action_index: Optional[int] = None
    attempt: int = 0
    step: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown QA category {self.category!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "message": self.message,
            "action": self.action,
            "action_index": self.action_index,
            "attempt": self.attempt,
            "step": self.step,
            "details": _jsonable(self.details),
        }

    def __str__(self) -> str:
        where = f" in {self.action}[{self.action_index}] attempt {self.attempt}" if self.action else ""
        return f"[{self.category}]{where}: {self.message}"


def _jsonable(x):
    if isinstance(x, th.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def category_of(exc: BaseException) -> str:
    """QA category of an exception raised while executing an action: a ``SkillError``'s reason when it is a category
    (anything else a motion says is ``execution``), ``error`` for unexpected exceptions."""
    reason = getattr(exc, "reason", None)
    if reason is not None:
        return reason if reason in CATEGORIES else "execution"
    return "error"


# ==========================================================================================
# statistics
# ==========================================================================================


class QAStats:
    """Verdict counts per category and per (action name, category); plus the per-step monitor event totals."""

    def __init__(self):
        self.by_category: Counter = Counter()
        self.by_action: Dict[str, Counter] = defaultdict(Counter)
        self.verdicts: List[Dict[str, Any]] = []
        self.events: Counter = Counter()  # monitor events that did not (yet) become verdicts

    def record(self, verdict: Verdict) -> None:
        self.by_category[verdict.category] += 1
        self.by_action[verdict.action or "-"][verdict.category] += 1
        self.verdicts.append(verdict.to_dict())

    def event(self, kind: str, n: int = 1) -> None:
        self.events[kind] += n

    def to_dict(self) -> Dict[str, Any]:
        return {
            "by_category": {c: self.by_category[c] for c in CATEGORIES if self.by_category[c]},
            "by_action": {a: dict(c) for a, c in sorted(self.by_action.items())},
            "n_verdicts": len(self.verdicts),
            "verdicts": list(self.verdicts),
            "events": dict(self.events),
        }


# ==========================================================================================
# camera model (spec §3.4 pinhole; USD camera looks along -z, +y up)
# ==========================================================================================


@dataclass
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_sensor(cls, sensor) -> "CameraModel":
        w, h = int(sensor.image_width), int(sensor.image_height)
        fx = w * float(sensor.focal_length) / float(sensor.horizontal_aperture)
        return cls(w, h, fx, fx, 0.5 * w, 0.5 * h)

    @classmethod
    def from_hfov(cls, width: int = 1080, height: int = 1080, hfov_deg: float = 99.2) -> "CameraModel":
        fx = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(width, height, fx, fx, 0.5 * width, 0.5 * height)

    def project(self, cam_pose, point) -> Tuple[float, float, float]:
        """``(u, v, depth)`` of a world point for the camera at ``cam_pose``; depth along the optical axis (-z)."""
        pos, quat = cam_pose
        R = C.quat_to_mat(quat)
        p = R.T @ (C.as_tensor(point) - C.as_tensor(pos))
        depth = -float(p[2])
        if abs(depth) < 1e-9:
            return float("nan"), float("nan"), depth
        u = self.cx + self.fx * float(p[0]) / depth
        v = self.cy - self.fy * float(p[1]) / depth
        return u, v, depth

    def in_frame(self, cam_pose, point, margin: float = 0.05) -> bool:
        u, v, depth = self.project(cam_pose, point)
        if not depth > 0.0 or math.isnan(u):
            return False
        return margin * self.width <= u <= (1.0 - margin) * self.width and margin * self.height <= v <= (1.0 - margin) * self.height

    def intrinsics(self) -> Dict[str, float]:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy, "width": self.width, "height": self.height}


# ==========================================================================================
# monitors
# ==========================================================================================


@dataclass
class QAConfig:
    collision_min_steps: int = 2
    ghost_pos: float = 0.05
    ghost_rot: float = 0.5
    pause_steps: int = 150
    pause_vel: float = 0.01
    fov_margin: float = 0.05
    fov_min_steps: int = 5
    max_action_steps: int = 3000
    max_episode_steps: int = 20000
    ground_categories: frozenset = GROUND_CATEGORIES
    #: keep at most this many collision events in the report
    max_events: int = 200


class Monitors:
    """The per-step monitors of one episode. Call :meth:`begin_action` before an action, :meth:`check` after every
    ``env.step`` (returns the first :class:`Verdict`, or ``None``), :meth:`end_action` afterwards and
    :meth:`begin_epilogue` for the final no-op window (pause / FOV monitors off)."""

    def __init__(self, env, robot, cfg: Optional[QAConfig] = None, cam_key: Optional[str] = None, cam_in_zed=None, camera: Optional[CameraModel] = None):
        self.env = env
        self.robot = robot
        self.cfg = cfg or QAConfig()
        self.scene = getattr(env, "scene", None)
        self.cam_key = cam_key
        self.cam_in_zed = cam_in_zed
        self.camera = camera
        if self.camera is None and cam_key is not None:
            try:
                self.camera = CameraModel.from_sensor(robot.sensors[cam_key])
            except Exception:  # noqa: BLE001
                self.camera = CameraModel.from_hfov()
        self.stats = QAStats()
        self.collision_events: List[Dict[str, Any]] = []
        self.collisions_by_object: Counter = Counter()
        self._prim_cache: Dict[str, Optional[str]] = {}
        self._contact_api = None
        self._contact_failed = False
        # per-action state
        self.action = None
        self.action_index: Optional[int] = None
        self.attempt = 0
        self.manipulation = False
        self.intended: set = set()
        self.exempt_links: set = set()
        self.held: List[Any] = []
        self.arms: Tuple[str, ...] = ()
        self.action_steps = 0
        self.episode_steps = 0
        self.epilogue = False
        self._contact_streak: Counter = Counter()
        self._pause_streak = 0
        self._fov_streak: Counter = Counter()
        self._ag_ref: Dict[str, Tuple[Any, Tuple[th.Tensor, th.Tensor]]] = {}

    # ------------------------------------------------------------------ lifecycle

    def begin_action(self, action, index: int, attempt: int, intended: Iterable[str] = (), held: Sequence[Any] = (), manipulation: Optional[bool] = None) -> None:
        """``intended``: scene object names the action may touch (its ``objects`` resolved to scene names);
        ``held``: scene objects currently in hand (their contacts are the robot's own)."""
        self.action = action
        self.action_index = index
        self.attempt = attempt
        self.epilogue = False
        self.arms = tuple(getattr(action, "arms", ()) or ())
        if manipulation is None:
            from ispatialgym.oracle.execute import is_manipulation  # lazy: execute imports actions (pure) only

            manipulation = is_manipulation(action)
        self.manipulation = bool(manipulation)
        self.intended = set(intended)
        self.held = list(held)
        self.exempt_links = set()
        if self.manipulation:
            for arm in self.arms:
                self.exempt_links.update(M.FINGER_LINKS[arm])
                self.exempt_links.add(M.GRIPPER_LINK[arm])
        self.action_steps = 0
        self._contact_streak = Counter()
        self._pause_streak = 0
        self._fov_streak = Counter()

    def end_action(self) -> None:
        self.action = None
        self.action_index = None
        self.manipulation = False
        self.intended = set()
        self.exempt_links = set()
        self.held = []
        self.arms = ()

    def begin_epilogue(self, held: Sequence[Any] = ()) -> None:
        """The final no-op window: pause / FOV monitors off; objects still in hand (held-object tasks) stay the
        robot's own for the collision monitor."""
        self.epilogue = True
        self.manipulation = False
        self.action = None
        self.action_index = None
        self.held = list(held)
        self.intended = {o.name for o in self.held}
        self.exempt_links = set()
        for arm in M.ARMS:
            self.exempt_links.update(M.FINGER_LINKS[arm])
            self.exempt_links.add(M.GRIPPER_LINK[arm])
        self._pause_streak = 0
        self._fov_streak = Counter()

    def reset_ag_reference(self) -> None:
        """Forget the held-object references (after a checkpoint restore)."""
        self._ag_ref = {}

    # ------------------------------------------------------------------ per step

    def _verdict(self, category: str, message: str, **details) -> Verdict:
        v = Verdict(category, message, action=getattr(self.action, "name", None), action_index=self.action_index, attempt=self.attempt, step=self.episode_steps, details=details)
        self.stats.record(v)
        return v

    def check(self, step: Optional[int] = None) -> Optional[Verdict]:
        """Run every monitor on the current simulator state; the first failure wins."""
        self.episode_steps = self.episode_steps + 1 if step is None else int(step)
        self.action_steps += 1
        for fn in (self._check_collision, self._check_ghost_hand, self._check_pause, self._check_fov, self._check_timeout):
            v = fn()
            if v is not None:
                return v
        return None

    # -- collision -------------------------------------------------------------------------------------------------

    def _contact_pairs(self):
        if self._contact_failed:
            return set()
        try:
            if self._contact_api is None:
                from omnigibson.utils.usd_utils import RigidContactAPI

                self._contact_api = RigidContactAPI
            query = [self.robot] + ([] if self.manipulation else list(self.held))
            return self._contact_api.get_contact_pairs(self.scene.idx, query, None, current_only=False)
        except Exception as e:  # noqa: BLE001  (no contact view: monitor off, reported once)
            log.warning("collision monitor disabled: %r", e)
            self._contact_failed = True
            self.stats.event("collision_monitor_disabled")
            return set()

    def _object_of(self, body_path: str):
        if body_path in self._prim_cache:
            return self._prim_cache[body_path]
        name = None
        try:
            obj = self.scene.object_registry("prim_path", body_path.rsplit("/", 1)[0], None)
        except Exception:  # noqa: BLE001
            obj = None
        if obj is not None and getattr(obj, "category", None) not in self.cfg.ground_categories:
            name = obj.name
        self._prim_cache[body_path] = name
        return name

    def _check_collision(self) -> Optional[Verdict]:
        own_prefixes = [self.robot.prim_path + "/"] + [o.prim_path + "/" for o in self.held]
        seen: Counter = Counter()
        for own, other in self._contact_pairs():
            if any(other.startswith(p) for p in own_prefixes):
                continue
            name = self._object_of(other)
            if name is None or name in self.intended:
                continue
            own_link = own.rsplit("/", 1)[1]
            if own_link in self.exempt_links:
                continue
            seen[(own_link, name)] += 1
        for key in list(self._contact_streak):
            if key not in seen:
                del self._contact_streak[key]
        for (own_link, name), _ in seen.items():
            self._contact_streak[(own_link, name)] += 1
            self.collisions_by_object[name] += 1
            self.stats.event("contact")
            if len(self.collision_events) < self.cfg.max_events:
                self.collision_events.append({"step": self.episode_steps, "own_link": own_link, "object": name, "action": getattr(self.action, "name", None)})
            if self._contact_streak[(own_link, name)] >= self.cfg.collision_min_steps:
                return self._verdict("collision", f"{own_link} in contact with {name} for {self._contact_streak[(own_link, name)]} steps", link=own_link, object=name)
        return None

    # -- ghost hand ------------------------------------------------------------------------------------------------

    def _check_ghost_hand(self) -> Optional[Verdict]:
        ag = getattr(self.robot, "_ag_obj_in_hand", None)
        if not ag:
            return None
        for arm, obj in ag.items():
            if obj is None:
                self._ag_ref.pop(arm, None)
                continue
            ep, eq = self.robot.eef_links[arm].get_position_orientation()
            op, oq = obj.get_position_orientation()
            rel = C.relative((op.float(), oq.float()), (ep.float(), eq.float()))
            ref = self._ag_ref.get(arm)
            if ref is None or ref[0] is not obj:
                self._ag_ref[arm] = (obj, rel)
                continue
            dp = float(th.linalg.norm(rel[0] - ref[1][0]))
            da = C.quat_angle(rel[1], ref[1][1])
            if dp > self.cfg.ghost_pos or da > self.cfg.ghost_rot:
                return self._verdict("ghost_hand", f"{obj.name} drifted {dp * 100:.1f} cm / {math.degrees(da):.0f} deg off the {arm} hand while grasped", arm=arm, object=obj.name)
        return None

    # -- pause -----------------------------------------------------------------------------------------------------

    def _check_pause(self) -> Optional[Verdict]:
        if self.epilogue or self.action is None:
            return None
        vel = float(self.robot.get_joint_velocities().abs().max())
        self._pause_streak = self._pause_streak + 1 if vel < self.cfg.pause_vel else 0
        if self._pause_streak >= self.cfg.pause_steps:
            return self._verdict("paused", f"robot stationary for {self._pause_streak} steps inside {self.action.name}", steps=self._pause_streak)
        return None

    # -- gripper in FOV --------------------------------------------------------------------------------------------

    def head_camera_pose(self) -> Tuple[th.Tensor, th.Tensor]:
        """Head-camera pose by FK through the conventions (torso joints + the physics-synced base pose)."""
        q = self.robot.get_joint_positions()
        torso = q[self.robot.trunk_control_idx].float()
        bp, bq = self.robot.get_position_orientation()
        return C.camera_pose(torso, base_pose=(bp.float(), bq.float()), cam_in_zed=self.cam_in_zed)

    def _check_fov(self) -> Optional[Verdict]:
        if not self.manipulation or self.camera is None or not self.arms:
            return None
        cam = self.head_camera_pose()
        for arm in self.arms:
            ep, eq = self.robot.eef_links[arm].get_position_orientation()
            tip = C.fingertip((ep.float(), eq.float()))
            if self.camera.in_frame(cam, tip, self.cfg.fov_margin):
                self._fov_streak[arm] = 0
                continue
            self._fov_streak[arm] += 1
            self.stats.event("fov_out")
            if self._fov_streak[arm] >= self.cfg.fov_min_steps:
                u, v, d = self.camera.project(cam, tip)
                return self._verdict("gripper_out_of_fov", f"{arm} fingertips out of the head camera frame for {self._fov_streak[arm]} steps (u={u:.0f}, v={v:.0f}, depth={d:.2f})", arm=arm, u=u, v=v, depth=d)
        return None

    # -- budgets ---------------------------------------------------------------------------------------------------

    def _check_timeout(self) -> Optional[Verdict]:
        if self.action is not None and self.action_steps > self.cfg.max_action_steps:
            return self._verdict("timeout", f"{self.action.name} exceeded {self.cfg.max_action_steps} steps", steps=self.action_steps)
        if self.episode_steps > self.cfg.max_episode_steps:
            return self._verdict("timeout", f"episode exceeded {self.cfg.max_episode_steps} steps", steps=self.episode_steps)
        return None

    # ------------------------------------------------------------------ report

    def report(self) -> Dict[str, Any]:
        return {
            "stats": self.stats.to_dict(),
            "collisions": {
                "n_events": sum(self.collisions_by_object.values()),
                "by_object": dict(self.collisions_by_object),
                "events": self.collision_events[: self.cfg.max_events],
            },
            "camera": None if self.camera is None else self.camera.intrinsics(),
        }


__all__ = [
    "CATEGORIES", "STEP_CATEGORIES", "EPILOGUE_CATEGORIES", "GROUND_CATEGORIES", "Verdict", "category_of", "QAStats",
    "CameraModel", "QAConfig", "Monitors",
]
