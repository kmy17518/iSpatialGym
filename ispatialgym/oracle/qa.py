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
``protected_contact``       during a transit, a robot link or held object touched a protected (goal) object (oracle.protection)
``protected_moved``         during a transit, a protected (goal) object moved more than protection.MOVE_TOL_M
``root_moved``              the root (base link) of a free-standing articulated object moved or tipped while the program
                            worked its joint (epilogue demo gate, ``executor.RootWatch``)
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

**Gripper-in-FOV rule (b)** (owner decision 2026-09-16 17:10, oracle doc 6.4; supersedes the per-action arming of
10:15): the monitor runs in **spans per acting arm**, continuous from the grasp to the release:

* a span **opens** when the pre-grasp pose is reached (``ctx["hooks"]["fov"](True)`` after ``arm_to(pre_grasp)`` in
  Grasp) and stays open through the grasp, the lift, the carry to the pre-place, the descent and the release step —
  across action boundaries (Lift, Reorient, Articulate, BaseFollow, a Transition / Navigate made while holding);
* it **closes** after the release (``hooks["fov"](False)`` once the assisted grasp has let go in Place, or in a
  ``Retreat(release=True)``); the retreat legs and the transit to the next pre-grasp are unmonitored; every grasp →
  release pair is its own span (disembedding's grasps, block design's blocks, the flip-on-table re-grasp);
* contact spans (Brace / Push) mirror them: pre-contact reached → end of the contact leg (the brace approach, the push
  arc with its extension and back-off);
* while a span is open it is **armed unconditionally**: ``fov_min_steps`` consecutive out-of-frame steps of that arm's
  fingertips (pinhole projection of spec 3.4, ``fov_margin`` = 5 %) are the ``gripper_out_of_fov`` verdict, whether
  they left the frame or never entered it on this action (``fov_first_entry_step: null``). A recorded pose on the
  margin is a sampler defect, not a reason to soften the monitor; the motion consequence is the oracle's: the carry leg
  lift end → pre-place is a straight Cartesian leg (or through the recorded ``grasps[k].carry`` waypoint), no free
  trajopt swing;
* frames of an acting arm whose span is closed (the transit to the pre-pose, the retreat) count as ``fov_idle_out``
  events, never verdicts. Spans are part of the checkpoint (a retry restores them).

B1K's replay QA (``FieldOfViewMetric``, ``gripper_changes_outside_fov_limit=0``) only needs the gripper in view at the
grasp / release toggles; this rule is deliberately stricter.

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
    "placed_orientation",
    "placed_tilt",
    "protected_contact",
    "protected_moved",
    "root_moved",
)
#: categories the per-step monitors can raise (``protected_*``: the transit span of oracle.protection, owner 2026-09-26)
STEP_CATEGORIES: Tuple[str, ...] = ("collision", "ghost_hand", "paused", "gripper_out_of_fov", "timeout", "protected_contact", "protected_moved")
#: categories of the epilogue (``placed_orientation`` = the demo-collection gate on a placed object's orientation
#: against its recorded, non-gating TRO target — owner decision 2026-09-16 10:15; ``root_moved`` = the demo gate on the root
#: of an articulated object whose joint the program works; policy evaluation is unchanged)
EPILOGUE_CATEGORIES: Tuple[str, ...] = ("not_reached", "tolerance", "not_settled", "placed_orientation", "placed_tilt", "root_moved")

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
        self.fov_active = False
        #: gripper-in-FOV spans per arm (rule (b)): open from the pre-grasp / pre-contact to the release / end of the
        #: contact leg, across action boundaries; checkpointed with the episode
        self.fov_span: Dict[str, bool] = {arm: False for arm in M.ARMS}
        self.forbidden: set = set()
        self.watch: List[Tuple[set, str]] = []
        self.watch_hits: List[int] = []
        self.allowed: List[Tuple[set, str]] = []
        self._contact_streak: Counter = Counter()
        self._pause_streak = 0
        self._fov_streak: Counter = Counter()
        self._fov_first_entry: Dict[str, Optional[int]] = {}
        self._fov_out_steps: Counter = Counter()
        self._ag_ref: Dict[str, Tuple[Any, Tuple[th.Tensor, th.Tensor]]] = {}
        # the protected-object transit span (owner 2026-09-26, oracle.protection): open from the start of a transit (the free
        # move to a pre-grasp / pre-place / pre-contact, the carry) to its end; any NEW contact of a robot link - the acting
        # fingers included, the manipulation exemption does not apply - or of a held object with a protected object, or a
        # protected object moving more than protection.MOVE_TOL_M, is a verdict
        self.transit: Optional[Dict[str, Any]] = None
        self.protect_events: List[Dict[str, Any]] = []
        self._pairs_step: Optional[int] = None
        self._pairs: set = set()

    # ------------------------------------------------------------------ lifecycle

    def begin_transit(self, objects: Sequence[Tuple[str, Any]], label: Optional[Dict[str, Any]] = None) -> None:
        """Open the protected-object transit span over ``objects`` (``[(bddl name, scene object)]``): their positions now
        and the contacts robot / held objects already have with them (a brace on the partner, a start contact) are the
        reference - only new contacts and movement count. No objects: nothing is watched."""
        objs = [(n, o) for n, o in (objects or ()) if o is not None]
        if not objs:
            self.transit = None
            return
        start: Dict[str, th.Tensor] = {}
        for _, o in objs:
            try:
                start[o.name] = th.as_tensor(o.get_position_orientation()[0], dtype=th.float32).clone()
            except Exception:  # noqa: BLE001
                continue
        names = {o.name for _, o in objs}
        self.transit = {"objects": objs, "names": names, "start": start, "label": dict(label or {}), "exempt": set()}
        self.transit["exempt"] = {pair for pair in self._protected_pairs()}

    def end_transit(self) -> None:
        self.transit = None

    def begin_action(self, action, index: int, attempt: int, intended: Iterable[str] = (), held: Sequence[Any] = (), manipulation: Optional[bool] = None,
                     fov: Optional[bool] = None, forbidden: Iterable[str] = (), watch: Iterable[Tuple[Iterable[str], str]] = (),
                     allowed: Iterable[Tuple[Iterable[str], str]] = ()) -> None:
        """``intended``: scene object names the action may touch (its ``objects`` resolved to scene names);
        ``held``: scene objects currently in hand (their contacts are the robot's own); ``fov``: whether the
        gripper-in-FOV monitor runs during this action at all (default: every action of a program — which arms it
        watches is decided by the open spans, ``fov_span``; a replay stream passes False);
        ``forbidden``: scene objects no link — the acting fingers included — and no held object may touch, whatever
        the manipulation exemptions (the container of a ``Place(kind="drop")``: a bowl lifted by the opening fingers
        or nudged by the held object is a ``collision``, never a silent success); ``watch``: ``(link names, scene
        object)`` pairs whose contacts are only *logged* (``watch_hits`` = episode steps with a contact) — the brace
        fingers against the pushed target, read by the executor's brace retry ladder; ``allowed``: ``(link names, scene
        object)`` pairs whose contacts are neither flagged nor counted (owner 2026-09-24 item 1: the brace hand's fingers
        on the braced object while the brace holds, ``Executor.brace_allowed_pairs`` - every other link / object pair keeps
        its rule)."""
        self.action = action
        self.action_index = index
        self.attempt = attempt
        self.epilogue = False
        self.transit = None
        self.arms = tuple(getattr(action, "arms", ()) or ())
        if manipulation is None or fov is None:
            from ispatialgym.oracle.execute import fov_monitored, is_manipulation  # lazy: execute imports actions (pure) only

            if manipulation is None:
                manipulation = is_manipulation(action)
            if fov is None:
                fov = fov_monitored(action)
        self.manipulation = bool(manipulation)
        self.fov_active = bool(fov)
        self.intended = set(intended)
        self.forbidden = set(forbidden)
        self.intended -= self.forbidden
        self.watch = [(set(links), str(obj)) for links, obj in (watch or ())]
        self.watch_hits: List[int] = []
        self.allowed = [(set(links), str(obj)) for links, obj in (allowed or ())]
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
        self._fov_first_entry = {arm: None for arm in M.ARMS}
        self._fov_out_steps = Counter()

    def end_action(self) -> None:
        self.transit = None
        self.action = None
        self.action_index = None
        self.manipulation = False
        self.fov_active = False
        self.intended = set()
        self.forbidden = set()
        self.watch = []
        self.allowed = []
        self.exempt_links = set()
        self.held = []
        self.arms = ()
        self._fov_first_entry = {}
        self._fov_out_steps = Counter()

    def set_fov_enabled(self, enabled: bool = True) -> None:
        """The execute functions' hook (``ctx["hooks"]["fov"]``): open (``True``) / close (``False``) the gripper-in-FOV
        span of the current action's arms — open once the pre-grasp / pre-contact is reached, closed after the release
        / the end of the contact leg. Opening starts a fresh leg (entry step, streaks) for those arms."""
        for arm in (self.arms or ()):
            self.set_fov_span(arm, enabled)

    def set_fov_span(self, arm: str, open_: bool) -> None:
        self.fov_span[arm] = bool(open_)
        self._fov_streak[arm] = 0
        self._fov_first_entry[arm] = None
        self._fov_out_steps[arm] = 0

    def fov_spans(self) -> Dict[str, bool]:
        """The open spans (for the checkpoint)."""
        return dict(self.fov_span)

    def restore_fov_spans(self, spans: Optional[Dict[str, bool]]) -> None:
        """Put the spans back as a checkpoint had them (a retry re-runs the action from its start)."""
        for arm in M.ARMS:
            self.fov_span[arm] = bool((spans or {}).get(arm, False))
        self._fov_streak = Counter()
        self._fov_first_entry = {arm: None for arm in M.ARMS}
        self._fov_out_steps = Counter()

    @property
    def fov_enabled(self) -> bool:
        """Whether any span is open (compatibility with the per-action flag)."""
        return any(self.fov_span.values())

    def arm_fov(self, arm: Optional[str] = None) -> None:
        """Open the span of ``arm`` (every acting arm when None)."""
        for a in ([arm] if arm else list(self.arms or ())):
            self.set_fov_span(a, True)

    def begin_epilogue(self, held: Sequence[Any] = ()) -> None:
        """The final no-op window: pause / FOV monitors off; objects still in hand (held-object tasks) stay the
        robot's own for the collision monitor."""
        self.epilogue = True
        self.transit = None
        self.manipulation = False
        self.fov_active = False
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
        for fn in (self._check_protected, self._check_collision, self._check_ghost_hand, self._check_pause, self._check_fov, self._check_timeout):
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
            if self._pairs_step == self.episode_steps and self._pairs_step is not None:
                return self._pairs
            query = [self.robot] + ([] if (self.manipulation and self.transit is None) else list(self.held))
            self._pairs = self._contact_api.get_contact_pairs(self.scene.idx, query, None, current_only=False)
            self._pairs_step = self.episode_steps
            return self._pairs
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

    def _protected_pairs(self) -> set:
        """``{(own link or "held:<link>", protected object name)}`` in contact now (robot links and held objects)."""
        if self.transit is None:
            return set()
        robot_prefix = self.robot.prim_path + "/"
        held_prefixes = [o.prim_path + "/" for o in self.held]
        out = set()
        for own, other in self._contact_pairs():
            name = self._object_of(other)
            if name is None or name not in self.transit["names"]:
                continue
            if own.startswith(robot_prefix):
                out.add((own.rsplit("/", 1)[1], name))
            elif any(own.startswith(p) for p in held_prefixes):
                out.add(("held:" + own.rsplit("/", 1)[1], name))
        return out

    def _check_protected(self) -> Optional[Verdict]:
        """The transit span (``begin_transit``): a new contact with a protected object, or a protected object moved by more
        than ``protection.MOVE_TOL_M`` since the span opened, is a verdict (the executor rolls back and re-plans the
        transit via the point above its goal)."""
        if self.transit is None:
            return None
        from ispatialgym.oracle import protection as PROT

        tr = self.transit
        lab = tr.get("label") or {}
        for own_link, name in sorted(self._protected_pairs() - tr["exempt"]):
            ev = {"check": "run", "kind": "contact", "task": lab.get("task"), "action": lab.get("action"), "index": lab.get("index"), "leg": lab.get("leg"),
                  "object": name, "link": own_link, "step": self.episode_steps}
            self.protect_events.append(ev)
            log.warning("PROTECT run-check fired: task=%s action=%s[%s] leg=%s object=%s: contact of %s during the transit (step %d)", ev["task"], ev["action"], ev["index"], ev["leg"], name, own_link, self.episode_steps)
            return self._verdict("protected_contact", f"{own_link} touched the protected {name} during the {lab.get('leg') or 'transit'}", link=own_link, object=name, leg=lab.get("leg"))
        for _, o in tr["objects"]:
            p0 = tr["start"].get(o.name)
            if p0 is None:
                continue
            try:
                p = th.as_tensor(o.get_position_orientation()[0], dtype=th.float32)
            except Exception:  # noqa: BLE001
                continue
            d = float(th.linalg.norm(p - p0))
            if d > PROT.MOVE_TOL_M:
                ev = {"check": "run", "kind": "moved", "task": lab.get("task"), "action": lab.get("action"), "index": lab.get("index"), "leg": lab.get("leg"),
                      "object": o.name, "moved_m": round(d, 4), "step": self.episode_steps}
                self.protect_events.append(ev)
                log.warning("PROTECT run-check fired: task=%s action=%s[%s] leg=%s object=%s moved %.1f mm during the transit (step %d)", ev["task"], ev["action"], ev["index"], ev["leg"], o.name, d * 1000, self.episode_steps)
                return self._verdict("protected_moved", f"the protected {o.name} moved {d * 1000:.1f} mm during the {lab.get('leg') or 'transit'}", object=o.name, moved_m=d, leg=lab.get("leg"))
        return None

    def _check_collision(self) -> Optional[Verdict]:
        robot_prefix = self.robot.prim_path + "/"
        held_prefixes = [o.prim_path + "/" for o in self.held]
        own_prefixes = [robot_prefix] + held_prefixes
        seen: Counter = Counter()
        for own, other in self._contact_pairs():
            if any(other.startswith(p) for p in own_prefixes):
                continue  # robot / held object against itself
            name = self._object_of(other)
            forbidden = name is not None and name in self.forbidden
            if own.startswith(robot_prefix):
                own_link = own.rsplit("/", 1)[1]
                if self.watch and name is not None and any(own_link in links and name == obj for links, obj in self.watch):
                    if not self.watch_hits or self.watch_hits[-1] != self.episode_steps:
                        self.watch_hits.append(self.episode_steps)
            elif any(own.startswith(p) for p in held_prefixes):
                if self.manipulation and not forbidden:
                    continue  # a held object touching the world is what manipulation is (placing, pushing with it)
                own_link = "held:" + own.rsplit("/", 1)[1]
            else:
                continue  # not one of our bodies
            if not forbidden:
                if name is None or name in self.intended:
                    continue
                if own_link in self.exempt_links:
                    continue
                if self.allowed and any(own_link in links and name == obj for links, obj in self.allowed):
                    continue  # the brace hand on the object it braces (allowed for the span of the brace)
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
        constraints = getattr(self.robot, "_ag_constraints", None)
        if constraints is None:
            constraints = getattr(self.robot, "_ag_obj_constraints", None)
        releasing = getattr(self.robot, "_ag_release_counter", None) or {}
        for arm, obj in ag.items():
            # OmniGibson removes the AG constraint the moment the open command arrives but keeps _ag_obj_in_hand set
            # through its release window: the object is free (a released door swings), not a ghost in the hand
            if obj is None or (isinstance(constraints, dict) and constraints.get(arm) is None) or releasing.get(arm) is not None:
                self._ag_ref.pop(arm, None)
                continue
            ep, eq = self.robot.eef_links[arm].get_position_orientation()
            body = self._ag_body(arm, obj)
            op, oq = body.get_position_orientation()
            rel = C.relative((op.float(), oq.float()), (ep.float(), eq.float()))
            ref = self._ag_ref.get(arm)
            if ref is None or ref[0] is not obj:
                self._ag_ref[arm] = (obj, rel)
                log.info("ghost_hand reference for %s: %s", obj.name, "the object root" if body is obj else f"link {getattr(body, 'name', '?')}")
                continue
            dp = float(th.linalg.norm(rel[0] - ref[1][0]))
            da = C.quat_angle(rel[1], ref[1][1])
            if dp > self.cfg.ghost_pos or da > self.cfg.ghost_rot:
                return self._verdict("ghost_hand", f"{obj.name} drifted {dp * 100:.1f} cm / {math.degrees(da):.0f} deg off the {arm} hand while grasped", arm=arm, object=obj.name)
        return None

    def _ag_body(self, arm: str, obj: Any) -> Any:
        """The body the assisted grasp is attached to: for an articulated object (cabinet, dishwasher, door) the
        grasped *link* (handle / leaf; ``_ag_obj_constraint_params[arm]["ag_link_prim_path"]``), not the object root -
        the root stays put while the door swings, so a root-vs-hand comparison reports a ghost hand on every honest
        articulation (dishwasher 0917-010945: "drifted 5.2 cm / 8 deg" at base_follow)."""
        params = (getattr(self.robot, "_ag_obj_constraint_params", None) or {}).get(arm) or {}
        links = getattr(obj, "links", None) or {}
        name = params.get("target_link_name")  # OmniGibson 3.9 (robot.py constraint_params)
        if name and name in links:
            return links[name]
        path = params.get("ag_link_prim_path")  # older OmniGibson (manipulation_robot.py)
        if path:
            for link in links.values():
                if getattr(link, "prim_path", None) == path:
                    return link
        return obj

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
        if not self.fov_active or self.camera is None:
            return None
        watched = [arm for arm in M.ARMS if self.fov_span.get(arm)]
        idle = [arm for arm in (self.arms or ()) if not self.fov_span.get(arm)]
        if not watched and not idle:
            return None
        cam = self.head_camera_pose()
        for arm in watched + idle:
            ep, eq = self.robot.eef_links[arm].get_position_orientation()
            tip = C.fingertip((ep.float(), eq.float()))
            inside = self.camera.in_frame(cam, tip, self.cfg.fov_margin)
            if arm in idle:
                if not inside:
                    self.stats.event("fov_idle_out")  # the transit to the pre-pose / the retreat leg: not monitored
                continue
            if inside:
                self._fov_streak[arm] = 0
                if self._fov_first_entry.get(arm) is None:
                    self._fov_first_entry[arm] = self.action_steps
                    self.stats.event("fov_in_frame")
                continue
            self._fov_streak[arm] += 1
            self._fov_out_steps[arm] += 1
            self.stats.event("fov_out")
            if self._fov_streak[arm] >= self.cfg.fov_min_steps:
                u, v, d = self.camera.project(cam, tip)
                first = self._fov_first_entry.get(arm)
                what = "never in the head camera frame on this leg" if first is None else f"out of the head camera frame for {self._fov_streak[arm]} steps"
                return self._verdict("gripper_out_of_fov", f"{arm} fingertips {what} (u={u:.0f}, v={v:.0f}, depth={d:.2f})", arm=arm, u=u, v=v, depth=d,
                                     fov_first_entry_step=first, fov_out_steps=int(self._fov_out_steps[arm]))
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
