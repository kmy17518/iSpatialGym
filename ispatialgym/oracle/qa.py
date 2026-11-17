"""Real-time QA of the v2 executor (``oracle_primitives_design.md`` §6.4; TODO F3): the oracle's retry rules
(:class:`Monitors`) and the demo QA metrics of ``run_isg --qa`` (:class:`QA`).

Verdict categories (the executor's verdict / ``SkillError.reason``; this module fixes the final list):

==========================  ======================================================================================
``not_reached``             the binary goal did not hold at the end of the program (epilogue)
``tolerance``               a measured error is above its tolerance (epilogue), or an action left a task quantity
                            (camera pose, joint value) off its recorded target
``not_settled``             robot or object motion after the final no-op window (epilogue), or a placed object
                            still moving
``collision``               oracle retry rule: a robot link touched an object other than the intended contacts of the action
``held_object_drift``       oracle retry rule: the assisted-grasp object drifted off the hand while still "in hand"
``stalled``                 oracle retry rule: the robot stood still for longer than the stall budget inside an action
``gripper_out_of_fov``      oracle retry rule: an acting gripper left the head camera's frame (5 % margin) while its span
                            was open
``protected_contact``       during a transit, a robot link or held object touched a protected (goal) object (oracle.protection)
``protected_moved``         during a transit, a protected (goal) object moved more than protection.MOVE_TOL_M
``failed_grasp``            the assisted grasp did not attach (grasp actions; push / brace exempt)
``distractor_displaced``    an object the grasp may touch (``grasps[].may_touch``: disembedding's parts / distractors) is no
                            longer on its support (the source tray) after the grasp, or rose with the target in the lift
``no_motion``               a push / articulation moved nothing
``plan_failed``             trajopt (or the differential IK along recorded waypoints) found no trajectory
``execution``               a motion did not converge / an action did not reach its recorded target
``timeout``                 the per-action or per-episode step budget was exhausted
``targets_failed``          ``action.targets()`` raised (a sidecar / conventions-side problem, kept apart from
                            ``plan_failed`` so the sampler-side cause stays distinguishable)
``error``                   an unexpected exception (traceback in the run's ``meta.json``)
``robot_touching_objects``  QA hard metrics (``run_isg --qa``, below): the oracle demo fails at once; each category is the
``robot_touching_itself``   metric of that name
``pause``
``jerky_motion``
``ghost_hand``
``object_tilt``
``goal_occlusion``
``robot_wobble``
``bad_release``
``arm_stutter``
==========================  ======================================================================================

**The oracle's retry rules** (:class:`Monitors`, oracle mode only, independent of ``--qa``) watch every ``env.step`` of an
action; a verdict rolls the action back and retries it (``ExecutorConfig.retries``), the demo fails once the retries are
used up:

* collision: ``RigidContactAPI`` pairs of the robot — and of the carried objects outside manipulation actions — against
  everything but the action's ``objects``, the held objects and the floor categories; the finger links of the acting arms
  are exempt during manipulation actions, their contacts are what manipulation is; a contact must persist
  ``collision_min_steps`` steps; the container of a drop is forbidden to every link;
* held object drift: the pose of the assisted-grasp object in the hand beyond ``drift_pos`` / ``drift_rot`` from its pose
  when the grasp began, while the assisted grasp says "in hand";
* stalled: every joint velocity below ``stall_vel`` for ``stall_steps`` steps inside an action, settle steps (the
  executor's ``settle``) not counted;
* gripper out of FOV (rule (b), owner decision 2026-09-16 17:10): per acting arm a span opens when the pre-grasp /
  pre-contact pose is reached (``ctx["hooks"]["fov"](True)``) and stays open through the grasp, the lift, the carry and
  the release (or the contact leg), across action boundaries; while it is open, ``fov_min_steps`` consecutive steps of
  that arm's fingertips out of the head camera's frame (pinhole projection, ``fov_margin`` = 5 %) are the verdict,
  whether they left the frame or never entered it (``fov_first_entry_step: null``); frames of an acting arm whose span is
  closed count as ``fov_idle_out`` events. Spans are part of the checkpoint;
* the protected-object transit spans of ``oracle.protection`` and the step budgets.

In a replay / policy stream only the step budget ends the episode; the collision, held-object-drift and stalled rules run
for the statistics and the QA log. ``QAStats`` counts verdicts per category × action name (§6.4: a sampler-side and a
planner-side cause of the same symptom stay distinguishable).

**Demo QA** (:class:`QA`, ``run_isg --qa``, on by default): BEHAVIOR's JoyLo QA metrics (``Behavior-1K/joylo/gello/utils/
qa_utils.py`` ``ALL_QA_METRICS``, with their modes; the jerky-motion limits raised for the oracle), the former oracle demo
gates (all hard), the arm / hand contacts the collision rule does not count, and the oracle's own retry rules (soft: the
oracle retries them, QA logs every verdict). Every metric is logged in ``result.qa``; in an oracle rollout a hard metric
fails the demo at once (no retry; jerky motion and goal occlusion are episode measures, judged after the settle window),
in a replay / policy stream the hard metrics are logged like the soft ones. Settle steps - the settle windows, the settle
rows of a replayed demo (``Executor.stream_phases``) and the oracle's hold / settle waits - never count as a pause.

==========================  ====  ===================================================================================
``robot_touching_objects``  hard  a robot link other than the arm / gripper / finger links touches a non-floor object
                                  (BEHAVIOR collision ``robot_nonarm_nonground``, limit 0)
``robot_touching_itself``   hard  two robot links touch (BEHAVIOR collision ``robot_self``, limit 0), but the two fingers
                                  of one gripper (the closed fist of a push / brace)
``pause``                   hard  more than 50 consecutive action steps with no joint faster than 0.001 (BEHAVIOR
                                  ``prolonged_pause``), settle steps not counted
``jerky_motion``            hard  arm-joint motion over the action steps: mean |velocity| <= 1.2 (BEHAVIOR 0.15), share of
                                  steps with a joint above 0.9 x its velocity limit <= 0.005, mean |acceleration| <= 3,
                                  mean |jerk| <= 150 (BEHAVIOR 100) (BEHAVIOR ``motion``; the 70 eval demos reach 0.92 and
                                  102)
``ghost_hand``              hard  an arm more than 0.1 rad off its commanded joints when its gripper command turns to open
                                  (BEHAVIOR ``ghost_hand_appearance``, ``gh_appearance_limit_while_ungrasping`` 0)
``object_tilt``             hard  the root of a free-standing articulated object whose joint the program opens or closes
                                  (a box lid, a laptop) moves more than 10 mm or tips more than 0.04 rad (former gate
                                  ``root_moved``, its peak band)
``goal_occlusion``          hard  a target / partner at least half hidden from the head camera at the end
                                  (``ispatialgym.eval.goal_occlusion``)
``robot_wobble``            hard  a load the hands bear bends the torso more than 1 deg off its command (:class:`TorsoLoad`)
``bad_release``             hard  an assisted-grasp release that does not come from a gripper command of +1
``arm_stutter``             hard  an arm's commanded joint motion jolts repeatedly inside one action attempt (a stream: the
                                  whole stream): >= 3 velocity changes of >= 0.6 rad/s within 6 steps while moving
                                  (:class:`ArmStutter`, :class:`StutterConfig`; former gate ``arm_stutter`` / ``robot_stutter``,
                                  owner 2026-10-13), judged after the settle window
``held_object_drift``       soft  the oracle's held-object-drift rule (retried): its verdicts (a stream: its steps)
``stalled``                 soft  the oracle's stalled rule (retried): its verdicts (a stream: its steps)
``gripper_in_view``         soft  oracle: while the gripper-out-of-FOV rule runs (the program's actions): the rule's
                                  verdicts (retried), the steps an acting arm's fingertips are out of the frame while its
                                  span is open, and the gripper toggles made out of the frame while its span is open
                                  (BEHAVIOR ``gripper_in_fov``). Policy eval (``stream_fov``; a stream has no spans): a grasp
                                  out of view - a close command after which an object attaches by assisted grasp within
                                  ``grasp_attach_steps``, the fingertips out of the frame at the close - and a release out
                                  of view - an open command that detaches a held object within that window, the fingertips
                                  out of the frame at the open. Replays: not measured
``failed_grasp``            soft  oracle only: grasps whose assisted grasp did not attach (BEHAVIOR ``failed_grasp``;
                                  retried by the oracle)
``head_camera_upright``     soft  steps with the head camera link pitched more than 0.4 rad after 3 s of driving, limit 30
                                  (BEHAVIOR ``head_camera_upright_during_navigation``)
``arm_hand_contacts``       soft  arm / gripper / finger links touching an object the action is not meant to touch (in a
                                  stream: any non-floor object, task objects marked)
==========================  ====  ===================================================================================

Simulator-free: ``env`` / ``robot`` are duck-typed; the contact API is imported lazily.
"""

from __future__ import annotations

import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.robot import conventions as C

log = logging.getLogger(__name__)

HARD, SOFT = "hard", "soft"
#: the ``--qa`` metrics (module docstring): mode, whether only an oracle rollout measures it, and whether the oracle retries it
QA_METRICS: Dict[str, Dict[str, Any]] = {
    "robot_touching_objects": {"mode": HARD},
    "robot_touching_itself": {"mode": HARD},
    "pause": {"mode": HARD},
    "jerky_motion": {"mode": HARD},
    "ghost_hand": {"mode": HARD},
    "held_object_drift": {"mode": SOFT, "retried": True},
    "stalled": {"mode": SOFT, "retried": True},
    "gripper_in_view": {"mode": SOFT, "retried": True, "policy_eval": "grasps / releases out of view"},
    "failed_grasp": {"mode": SOFT, "oracle_only": True, "retried": True},
    "head_camera_upright": {"mode": SOFT},
    "arm_hand_contacts": {"mode": SOFT},
    "object_tilt": {"mode": HARD},
    "goal_occlusion": {"mode": HARD},
    "robot_wobble": {"mode": HARD},
    "bad_release": {"mode": HARD},
    "arm_stutter": {"mode": HARD},
}
HARD_METRICS: Tuple[str, ...] = tuple(k for k, v in QA_METRICS.items() if v["mode"] == HARD)
SOFT_METRICS: Tuple[str, ...] = tuple(k for k, v in QA_METRICS.items() if v["mode"] == SOFT)
#: the oracle's verdict categories the soft metrics of that name log
RULE_METRICS: Dict[str, str] = {"held_object_drift": "held_object_drift", "stalled": "stalled", "gripper_out_of_fov": "gripper_in_view",
                                "failed_grasp": "failed_grasp"}

CATEGORIES: Tuple[str, ...] = (
    "not_reached",
    "tolerance",
    "not_settled",
    "symbolic_only",
    "collision",
    "held_object_drift",
    "stalled",
    "gripper_out_of_fov",
    "failed_grasp",
    "distractor_displaced",
    "no_motion",
    "plan_failed",
    "execution",
    "timeout",
    "targets_failed",
    "error",
    "protected_contact",
    "protected_moved",
) + HARD_METRICS
#: categories the oracle's per-step retry rules can raise (``protected_*``: the transit span of oracle.protection, owner 2026-09-26)
STEP_CATEGORIES: Tuple[str, ...] = ("collision", "held_object_drift", "stalled", "gripper_out_of_fov", "timeout", "protected_contact", "protected_moved")
#: categories of the epilogue (the QA hard metrics measured over the episode included): the task's metrics give ``not_reached``
#: (the goal never held), ``not_settled`` (settle-eval on: never through a settled window) and ``symbolic_only`` (symbolic success,
#: no geometric success: the measured targets never within tolerance at the same step(s)); ``tolerance`` is the push loops'
EPILOGUE_CATEGORIES: Tuple[str, ...] = ("not_reached", "not_settled", "symbolic_only", "jerky_motion", "goal_occlusion", "arm_stutter")

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
    drift_pos: float = 0.05
    drift_rot: float = 0.5
    stall_steps: int = 150
    stall_vel: float = 0.01
    fov_margin: float = 0.05
    fov_min_steps: int = 5
    max_action_steps: int = 3000
    max_episode_steps: int = 20000
    ground_categories: frozenset = GROUND_CATEGORIES
    #: keep at most this many collision events in the report
    max_events: int = 200


class Monitors:
    """The oracle's per-step retry rules of one episode (module docstring). Call :meth:`begin_action` before an action,
    :meth:`check` after every ``env.step`` (returns the first :class:`Verdict`, or ``None``), :meth:`end_action` afterwards
    and :meth:`begin_epilogue` for the final no-op window (stalled / FOV rules off); a stream calls
    :meth:`check_timeout_only`."""

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
        self._stall_streak = 0
        self._settle = False
        self._fov_streak: Counter = Counter()
        self._fov_first_entry: Dict[str, Optional[int]] = {}
        self._fov_out_steps: Counter = Counter()
        self._ag_ref: Dict[str, Tuple[Any, Tuple[th.Tensor, th.Tensor]]] = {}
        #: a stream's held-object-drift / stalled verdicts, not returned (:meth:`check_timeout_only`), for the QA log
        self.logged: List[Verdict] = []
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
        hx = (label or {}).get("hold_exempt") or {}
        # a hand holding a press on a protected object touches it by design, whenever the contact comes and goes (execute._transit)
        self.transit["hold_exempt"] = (set(hx.get("links") or ()), str(hx.get("object"))) if hx.get("object") else None

    def end_transit(self) -> None:
        self.transit = None

    def begin_action(self, action, index: int, attempt: int, intended: Iterable[str] = (), held: Sequence[Any] = (), manipulation: Optional[bool] = None,
                     fov: Optional[bool] = None, forbidden: Iterable[str] = (), watch: Iterable[Tuple[Iterable[str], str]] = (),
                     allowed: Iterable[Tuple[Iterable[str], str]] = ()) -> None:
        """``intended``: scene object names the action may touch (its ``objects`` resolved to scene names);
        ``held``: scene objects currently in hand (their contacts are the robot's own); ``fov``: whether the
        gripper-in-FOV rule runs during this action at all (default: every action of a program — which arms it
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
        self._stall_streak = 0
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
        """Whether any span is open."""
        return any(self.fov_span.values())

    def begin_epilogue(self, held: Sequence[Any] = ()) -> None:
        """The final no-op window: stalled / FOV rules off; objects still in hand (held-object tasks) stay the robot's own
        for the collision rule."""
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
        self._stall_streak = 0
        self._fov_streak = Counter()

    def reset_ag_reference(self) -> None:
        """Forget the held-object references (after a checkpoint restore)."""
        self._ag_ref = {}

    def rule_limits(self) -> Dict[str, Dict[str, float]]:
        """The limits of the rules the soft QA metrics log."""
        c = self.cfg
        return {"held_object_drift": {"pos_m": c.drift_pos, "rot_rad": c.drift_rot}, "stalled": {"steps": c.stall_steps, "vel": c.stall_vel},
                "gripper_in_view": {"min_steps": c.fov_min_steps, "margin": c.fov_margin}}

    # ------------------------------------------------------------------ per step

    def _verdict(self, category: str, message: str, **details) -> Verdict:
        v = Verdict(category, message, action=getattr(self.action, "name", None), action_index=self.action_index, attempt=self.attempt, step=self.episode_steps, details=details)
        self.stats.record(v)
        return v

    def check(self, step: Optional[int] = None, settle: bool = False) -> Optional[Verdict]:
        """Run every rule on the current simulator state; the first failure wins. ``settle``: a settle step (a hold / settle
        wait of the motion layer), not counted by the stalled rule."""
        self.episode_steps = self.episode_steps + 1 if step is None else int(step)
        self.action_steps += 1
        self._settle = bool(settle)
        for fn in (self._check_protected, self._check_collision, self._check_held_object_drift, self._check_stall, self._check_fov, self._check_timeout):
            v = fn()
            if v is not None:
                return v
        return None

    def check_timeout_only(self, step: Optional[int] = None, settle: bool = False) -> Optional[Verdict]:
        """``check`` of a replay / policy stream (no retry rules): the step counters, the collision rule's contact statistics
        and the held-object-drift / stalled rules, their verdicts kept in :attr:`logged` for the QA log instead of returned
        (``settle``: a settle row of a replayed demo or a settle window step, not counted by the stalled rule); only the step
        budget ends the episode."""
        self.episode_steps = self.episode_steps + 1 if step is None else int(step)
        self.action_steps += 1
        self._settle = bool(settle)
        for name in ("_check_collision", "_check_held_object_drift", "_check_stall"):
            try:
                v = getattr(self, name)()
            except Exception:  # noqa: BLE001  (statistics only)
                continue
            if v is not None and v.category in ("held_object_drift", "stalled"):
                self.logged.append(v)
        return self._check_timeout()

    def pop_logged(self) -> List[Verdict]:
        """The stream verdicts :meth:`check_timeout_only` kept since the last call."""
        out, self.logged = self.logged, []
        return out

    # -- collision -------------------------------------------------------------------------------------------------

    def _contact_pairs(self, step: Optional[int] = None):
        """The robot's (and the counted held objects') ``RigidContactAPI`` pairs of ``step`` (default: the current episode
        step), queried once per step - the QA metrics read the same query."""
        if self._contact_failed:
            return set()
        key = self.episode_steps if step is None else int(step)
        try:
            if self._contact_api is None:
                from omnigibson.utils.usd_utils import RigidContactAPI

                self._contact_api = RigidContactAPI
            if self._pairs_step == key and self._pairs_step is not None:
                return self._pairs
            query = [self.robot] + ([] if (self.manipulation and self.transit is None) else list(self.held))
            self._pairs = self._contact_api.get_contact_pairs(self.scene.idx, query, None, current_only=False)
            self._pairs_step = key
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
        hx = tr.get("hold_exempt")
        braced = self._brace_protect_allowed()
        for own_link, name in sorted(self._protected_pairs() - tr["exempt"]):
            if hx is not None and own_link in hx[0] and name == hx[1]:
                continue
            if braced and any(own_link in links and name == obj for links, obj in braced):
                # owner 5r.2: the active brace's own finger / object pair (brace_allowed) during the other arm's move
                if (own_link, name) not in tr.setdefault("brace_allowed_seen", set()):
                    tr["brace_allowed_seen"].add((own_link, name))
                    self.stats.event("protect_brace_allowed")
                    log.info("PROTECT run-check: %s on the braced %s allowed during the %s (brace_allowed, %s; step %d)", own_link, name,
                             lab.get("leg") or "transit", BRACE_PROTECT_SWITCH, self.episode_steps)
                continue
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

    def _brace_protect_allowed(self) -> List[Tuple[set, str]]:
        """Owner 5r.2 (2026-11-08, an oracle QA rule; :data:`BRACE_PROTECT_SWITCH`, default off): the ``allowed`` pairs of the running action -
        ``Executor.brace_allowed_pairs``, the active braces' finger / gripper links on the object each braces, set by ``begin_action`` only while
        a brace's span contains the action - count as allowed in the protected-contact run-check, limited to the pairs of a brace arm that is not
        moving in this action (the other arm's move). Nothing else changes: no active brace = no pairs; every other pair keeps its rule; the
        movement check is unchanged. Off (``=0``): ``[]``; default on (:data:`BRACE_PROTECT_DEFAULT`)."""
        if (os.environ.get(BRACE_PROTECT_SWITCH) or BRACE_PROTECT_DEFAULT).strip() != "1":
            return []
        allowed = getattr(self, "allowed", None) or []
        if not allowed:
            return []
        moving = set()
        for arm in getattr(self, "arms", ()) or ():
            moving.update(M.FINGER_LINKS[arm])
            moving.add(M.GRIPPER_LINK[arm])
        return [(set(links) - moving, obj) for links, obj in allowed if set(links) - moving]

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

    # -- held object drift -----------------------------------------------------------------------------------------

    def _check_held_object_drift(self) -> Optional[Verdict]:
        ag = getattr(self.robot, "_ag_obj_in_hand", None)
        if not ag:
            return None
        constraints = getattr(self.robot, "_ag_constraints", None)
        if constraints is None:
            constraints = getattr(self.robot, "_ag_obj_constraints", None)
        releasing = getattr(self.robot, "_ag_release_counter", None) or {}
        for arm, obj in ag.items():
            # OmniGibson removes the AG constraint the moment the open command arrives but keeps _ag_obj_in_hand set
            # through its release window: the object is free (a released door swings), not drifting in the hand
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
                log.info("held_object_drift reference for %s: %s", obj.name, "the object root" if body is obj else f"link {getattr(body, 'name', '?')}")
                continue
            dp = float(th.linalg.norm(rel[0] - ref[1][0]))
            da = C.quat_angle(rel[1], ref[1][1])
            if dp > self.cfg.drift_pos or da > self.cfg.drift_rot:
                return self._verdict("held_object_drift", f"{obj.name} drifted {dp * 100:.1f} cm / {math.degrees(da):.0f} deg off the {arm} hand while grasped",
                                     arm=arm, object=obj.name)
        return None

    def _ag_body(self, arm: str, obj: Any) -> Any:
        """The body the assisted grasp is attached to: for an articulated object (cabinet, dishwasher, door) the
        grasped *link* (handle / leaf; ``_ag_obj_constraint_params[arm]["ag_link_prim_path"]``), not the object root -
        the root stays put while the door swings, so a root-vs-hand comparison reports a drift on every honest
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

    # -- stalled ---------------------------------------------------------------------------------------------------

    def _check_stall(self) -> Optional[Verdict]:
        if self.epilogue or self.action is None or self._settle:
            self._stall_streak = 0
            return None
        vel = float(self.robot.get_joint_velocities().abs().max())
        self._stall_streak = self._stall_streak + 1 if vel < self.cfg.stall_vel else 0
        if self._stall_streak >= self.cfg.stall_steps:
            return self._verdict("stalled", f"robot stationary for {self._stall_streak} steps inside {self.action.name}", steps=self._stall_streak)
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
                    self.stats.event("fov_idle_out")  # the transit to the pre-pose / the retreat leg: not watched
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
            "camera": None if self.camera is None else self.camera.intrinsics(),
            "collisions": {
                "n_events": sum(self.collisions_by_object.values()),
                "by_object": dict(self.collisions_by_object),
                "events": self.collision_events[: self.cfg.max_events],
            },
        }


# ==========================================================================================
# arm stutter (epilogue gate ``arm_stutter``)
# ==========================================================================================


@dataclass
class StutterConfig:
    """``arm_stutter`` (owner 2026-10-13, mental_rotation-walker: "the robot seems to stutter"). A **jolt** is a control step at
    which the commanded joint-velocity vector of one arm changes by at least ``jolt_rad_s`` (rad/s, from one step's finite
    difference to the next) while the arm moves at ``move_rad_s`` or more (its fastest joint) on both sides; ``min_jolts`` jolts
    within ``window_steps`` steps of one motion (an action attempt; a replay stream is one) are a stutter. The joints of the
    7-DOF arm only; the trunk, base and grippers are not judged.

    The walker demo (code 14a2f58b): right_arm_joint6 pinned 8 mrad from its lower limit by the differential IK of the in-hand
    orbit, the other joints swinging - 3 jolts of 0.74 / 1.04 / 0.99 rad/s at steps 2-4 and of 1.75 / 0.84 / 0.95 rad/s at steps
    37-40 (joint 3 reversing from +0.90 to -0.33 rad/s, joint 5 at 0.24 -> 1.41 -> 0.77 -> 1.43 rad/s). The other 70 counted demos
    of 2026-10-13 (s2_3tasks, s3_55tasks, iors_large) have at most 2 jolts of 0.6 rad/s in any 6 steps: a single kink where two
    legs of an action join, or a one-step speed blip of the per-waypoint step rounding."""

    jolt_rad_s: float = 0.6
    move_rad_s: float = 0.3
    window_steps: int = 6
    min_jolts: int = 3
    dt: float = 1.0 / 30.0


def find_stutters(q, segment, cfg: Optional[StutterConfig] = None) -> Dict[str, Any]:
    """The jolts and stutters of one arm's joint trace. ``q``: (T, n) joint positions per control step (the commanded
    targets); ``segment``: (T,) motion id per step - steps of one id are one motion, a negative id belongs to none. Returns
    ``{"jolts": [(index, |dv| rad/s)], "stutters": [{"first", "last", "n_jolts", "max_jolt_rad_s"}]}`` with indices into the
    trace (a jolt's index is the step at which the velocity changes); overlapping windows merge into one stutter."""
    import numpy as np

    cfg = cfg or StutterConfig()
    q = np.asarray(q, dtype=np.float64)
    seg = np.asarray(segment).reshape(-1)
    jolts: List[Tuple[int, float]] = []
    start = 0
    for t in range(1, len(seg) + 1):
        if t < len(seg) and seg[t] == seg[start]:
            continue
        if seg[start] >= 0 and t - start >= 3:
            v = np.diff(q[start:t], axis=0) / cfg.dt
            speed = np.abs(v).max(axis=1)
            for i in range(1, len(v)):
                if min(speed[i - 1], speed[i]) >= cfg.move_rad_s:
                    dv = float(np.linalg.norm(v[i] - v[i - 1]))
                    if dv >= cfg.jolt_rad_s:
                        jolts.append((start + i, dv))
        start = t
    stutters: List[Dict[str, Any]] = []
    for k, (t0, _) in enumerate(jolts):
        inside = [(t, dv) for t, dv in jolts[k:] if t - t0 < cfg.window_steps and seg[t] == seg[t0]]
        if len(inside) < cfg.min_jolts:
            continue
        last = inside[-1][0]
        peak = max(dv for _, dv in inside)
        if stutters and t0 <= stutters[-1]["last"] and seg[t0] == seg[stutters[-1]["first"]]:
            s = stutters[-1]
            s["n_jolts"] = len({t for t, _ in jolts if s["first"] <= t <= max(s["last"], last)})
            s["last"] = max(s["last"], last)
            s["max_jolt_rad_s"] = max(s["max_jolt_rad_s"], peak)
        else:
            stutters.append({"first": t0, "last": last, "n_jolts": len(inside), "max_jolt_rad_s": peak})
    for s in stutters:
        s["max_jolt_rad_s"] = round(s["max_jolt_rad_s"], 3)
    return {"jolts": [(t, round(dv, 3)) for t, dv in jolts], "stutters": stutters}


class ArmStutter:
    """QA fact and ``arm_stutter`` gate: the commanded arm joint targets of every control step (``observe``), judged per arm
    by :func:`find_stutters` at the end (``report``). ``segment`` groups the steps of one motion (an action attempt; the
    executor passes a negative id outside the program's actions); a rolled-back attempt's steps leave the record (``trim``)."""

    def __init__(self, robot, cfg: Optional[StutterConfig] = None):
        self.cfg = cfg or StutterConfig()
        self.idx_action = {arm: th.as_tensor(robot.controller_action_idx[f"arm_{arm}"]).long() for arm in M.ARMS}
        self.rows: List[Tuple[int, int, Optional[str], Dict[str, List[float]]]] = []

    def observe(self, step: int, action, segment: int, action_name: Optional[str] = None) -> None:
        a = th.as_tensor(action).reshape(-1).float()
        self.rows.append((int(step), int(segment), action_name, {arm: a[idx].tolist() for arm, idx in self.idx_action.items()}))

    def trim(self, step: int) -> None:
        self.rows = [r for r in self.rows if r[0] <= step]

    def report(self) -> Dict[str, Any]:
        cfg = self.cfg
        out: Dict[str, Any] = {"flagged": False, "stutters": [], "max_jolt_rad_s": 0.0,
                               "thresholds": {"jolt_rad_s": cfg.jolt_rad_s, "move_rad_s": cfg.move_rad_s, "window_steps": cfg.window_steps,
                                              "min_jolts": cfg.min_jolts}}
        if len(self.rows) < 3:
            return out
        steps = [r[0] for r in self.rows]
        segment = [r[1] for r in self.rows]
        names = [r[2] for r in self.rows]
        for arm in self.idx_action:
            res = find_stutters([r[3][arm] for r in self.rows], segment, cfg)
            if res["jolts"]:
                out["max_jolt_rad_s"] = max(out["max_jolt_rad_s"], max(dv for _, dv in res["jolts"]))
            for s in res["stutters"]:
                out["stutters"].append({"arm": arm, "first_step": steps[s["first"]], "last_step": steps[s["last"]], "n_jolts": s["n_jolts"],
                                        "max_jolt_rad_s": s["max_jolt_rad_s"], "action": names[s["first"]]})
        out["flagged"] = bool(out["stutters"])
        if out["stutters"]:
            out["worst"] = max(out["stutters"], key=lambda s: (s["n_jolts"], s["max_jolt_rad_s"]))
        return out


def stutter_report_hdf5(path: str, cfg: Optional[StutterConfig] = None) -> Dict[str, Any]:
    """:class:`ArmStutter`'s report of a recorded demo (``isg-demo-v2`` hdf5): the commanded arm targets of ``data/demo_0/action``
    over the action steps (``isg/action_index`` >= 0, one motion per action index and attempt), steps numbered as ``isg/step``."""
    import json

    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        g = f["data/demo_0"]
        order = json.loads(f["data"].attrs["controller_order"])
        dims = json.loads(f["data"].attrs["controller_dims"])
        act = np.asarray(g["action"][:], dtype=np.float64)
        aidx = np.asarray(g["isg/action_index"][:]).astype(int)
        att = np.asarray(g["isg/attempt"][:]).astype(int)
        steps = np.asarray(g["isg/step"][:]).astype(int)
        ann = json.loads(g.attrs.get("isg_annotations", "{}") or "{}")
    names = {int(a["index"]): a.get("name") for a in ann.get("actions", [])}
    offsets, o = {}, 0
    for k in order:
        offsets[k] = (o, o + int(dims[k]))
        o += int(dims[k])
    idx = {arm: list(range(*offsets[f"arm_{arm}"])) for arm in M.ARMS}

    class _Robot:
        controller_action_idx = {f"arm_{arm}": idx[arm] for arm in M.ARMS}

    st = ArmStutter(_Robot(), cfg)
    for t in range(len(act)):
        seg = int(aidx[t]) * 1000 + int(att[t]) if aidx[t] >= 0 else -1
        st.observe(int(steps[t]), act[t], seg, names.get(int(aidx[t])))
    return st.report()


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(description="arm_stutter report of recorded demos (data/demo_0/action of an isg-demo-v2 hdf5)")
    p.add_argument("hdf5", nargs="+")
    args = p.parse_args(argv)
    flagged = 0
    for path in args.hdf5:
        rep = stutter_report_hdf5(path)
        flagged += int(rep["flagged"])
        print(json.dumps({"hdf5": path, **rep}))
    return 1 if flagged else 0


# ==========================================================================================
# --qa: the demo QA metrics
# ==========================================================================================


class TorsoLoad:
    """``robot_wobble`` metric (owner 2026-10-09, audit of the v3.0-reconciled demos: "too much push at the end makes the robot
    wobble ... door case is unacceptable ... if unpreventable discard"): how far a load the hands bear bends the torso off its
    command - per control step the largest |trunk command - measured trunk joint| (deg), recorded from ``NOTE_DEG`` on with its
    action, so the report has the peak, when and in which action, and the peak per action. The free torso tracks within 0.01 deg
    (0.09 deg while it moves); the 28 v3.0-reconciled demos of 1008 peaked at 4.24 deg (door push-close: the head lunged 78 mm when
    the fist let go), 0.81 (brussels sprouts: fingertips on the table), 0.41 (blender open), 0.38 (door open), 0.24 (dishwasher
    close). A rolled-back attempt's steps leave the record (:meth:`trim`)."""

    NOTE_DEG = 0.05

    def __init__(self, robot):
        self.robot = robot
        self.idx_action = th.as_tensor(robot.controller_action_idx["trunk"]).long()
        self.idx_joint = th.as_tensor(robot.trunk_control_idx).long()
        self.steps: Dict[int, Tuple[float, Optional[str]]] = {}

    def observe(self, step: int, action: th.Tensor, action_name: Optional[str]) -> float:
        """Record the step; returns its deviation (deg)."""
        cmd = th.as_tensor(action).reshape(-1)[self.idx_action].float()
        q = th.as_tensor(self.robot.get_joint_positions()).reshape(-1)[self.idx_joint].float()
        dev = math.degrees(float(th.max(th.abs(cmd - q))))
        if dev >= self.NOTE_DEG:
            self.steps[int(step)] = (dev, action_name)
        else:
            self.steps.pop(int(step), None)
        return dev

    def trim(self, step: int) -> None:
        self.steps = {s: v for s, v in self.steps.items() if s <= step}

    def peak(self) -> Tuple[float, Optional[int], Optional[str]]:
        if not self.steps:
            return 0.0, None, None
        s, (d, n) = max(self.steps.items(), key=lambda kv: kv[1][0])
        return d, s, n

    def report(self) -> Dict[str, Any]:
        d, s, n = self.peak()
        per: Dict[str, float] = {}
        for _s, (dv, nm) in self.steps.items():
            per[nm or "epilogue"] = max(per.get(nm or "epilogue", 0.0), dv)
        return {"max_deg": round(d, 3), "at_step": s, "action": None if s is None else (n or "epilogue"),
                "per_action_deg": {k: round(v, 3) for k, v in sorted(per.items(), key=lambda kv: -kv[1])},
                "steps_over_note": len(self.steps), "note_deg": self.NOTE_DEG}


class RootWatch:
    """``object_tilt`` reading (owner 2026-09-27, former demo gate ``root_moved``): the root pose of every free-standing
    articulated object whose joint the program works (``Executor.operated_roots`` - a lid, a flap, a door leaf; the body is
    meant to stay put) against its pose at the first step: displacement, up-axis tilt and full rotation, the peak of each
    (with the action and step) and the value at the end."""

    def __init__(self, objects: Dict[str, Any], skipped: Optional[Dict[str, str]] = None):
        self.objects = dict(objects)
        self.skipped = dict(skipped or {})
        self.pose0: Dict[str, Tuple[th.Tensor, th.Tensor]] = {}
        self.peak: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.end: Dict[str, Dict[str, float]] = {}
        self.steps = 0

    def observe(self, action_name: Optional[str], step: int) -> Dict[str, Dict[str, float]]:
        """Read every watched root; returns this step's readings per object."""
        self.steps += 1
        z = th.tensor([0.0, 0.0, 1.0])
        out: Dict[str, Dict[str, float]] = {}
        for name, obj in self.objects.items():
            try:
                p, q = (x.float().clone() for x in obj.get_position_orientation())
            except Exception:  # noqa: BLE001
                continue
            if name not in self.pose0:
                self.pose0[name] = (p, q)
            p0, q0 = self.pose0[name]
            cur = {"displacement_m": float(th.linalg.norm(p - p0)),
                   "tilt_rad": math.acos(max(-1.0, min(1.0, float(th.dot(C.quat_rotate(q, z), C.quat_rotate(q0, z)))))),
                   "rotation_rad": float(C.quat_angle(q, q0))}
            peak = self.peak.setdefault(name, {})
            for k, v in cur.items():
                if k not in peak or v > peak[k]["value"]:
                    peak[k] = {"value": v, "action": action_name or "epilogue", "step": int(step)}
            self.end[name] = cur
            out[name] = cur
        return out

    def over_band(self, pos_m: float, tilt_rad: float) -> Dict[str, List[str]]:
        """``{object: [reasons]}``: a peak displacement above ``pos_m`` or a peak tilt above ``tilt_rad``."""
        out: Dict[str, List[str]] = {}
        for name, peak in self.peak.items():
            why = []
            if peak["displacement_m"]["value"] > pos_m:
                why.append(f"moved {peak['displacement_m']['value'] * 1000:.1f} mm in {peak['displacement_m']['action']} (> {pos_m * 1000:.0f})")
            if peak["tilt_rad"]["value"] > tilt_rad:
                why.append(f"tipped {math.degrees(peak['tilt_rad']['value']):.2f} deg in {peak['tilt_rad']['action']} (> {math.degrees(tilt_rad):.2f})")
            if why:
                out[name] = why
        return out

    def snapshot(self) -> Dict[str, Any]:
        """The readings so far (a checkpoint's copy: a rolled-back attempt leaves no trace in the demo, nor in the metric)."""
        return {"peak": {n: {k: dict(v) for k, v in p.items()} for n, p in self.peak.items()}, "end": {n: dict(e) for n, e in self.end.items()}, "steps": self.steps}

    def load(self, snap: Optional[Dict[str, Any]]) -> None:
        if snap is not None:
            self.peak = {n: {k: dict(v) for k, v in p.items()} for n, p in snap["peak"].items()}
            self.end = {n: dict(e) for n, e in snap["end"].items()}
            self.steps = int(snap["steps"])

    def report(self, pos_m: float, tilt_rad: float) -> Dict[str, Any]:
        bad = self.over_band(pos_m, tilt_rad)
        objects = {}
        for name in self.objects:
            peak, end = self.peak.get(name, {}), self.end.get(name, {})
            objects[name] = {"peak": {k: {**v, "value": round(v["value"], 5)} for k, v in peak.items()}, "end": {k: round(v, 5) for k, v in end.items()},
                             "exceeded": bad.get(name, [])}
        return {"objects": objects, "skipped": dict(self.skipped), "steps_observed": int(self.steps), "band": {"pos_m": pos_m, "tilt_rad": tilt_rad}}


#: oracle-fix 2026-11-07 (owner prompt B, the hardback's open-pitch-40 push): the elbow pair link3 / link5 of each arm, whose collision
#: hulls graze within the joint-4 limit (left: hull distance +4.3 mm at q4 -2.045, -1.2 mm at the limit -2.0944; right >= 6.9 mm; the
#: cuRobo spheres >= 16.8 mm apart, the pair not in r1pro.yaml's disabled_collision_pairs) - a collision-mesh artifact, not a fold
ELBOW_GRAZE_PAIRS: Tuple[Tuple[str, str], ...] = tuple(tuple(sorted((f"{arm}_arm_link3", f"{arm}_arm_link5"))) for arm in ("left", "right"))
SELF_EXCLUDE_SWITCH = "ISPATIALGYM_QA_SELF_EXCLUDE_ELBOW"
#: owner 5r.2 (2026-11-08, i7-contact3; close_02 300006): "1" = while a brace is active, the finger / object pairs its ``brace_allowed`` lists
#: count as allowed in the protected-contact run-check during the other arm's move (:meth:`Monitors._brace_protect_allowed`). Default "1"
#: since the lead's decision 8 (2026-11-08, owner away; it acted once, on the diagnosed pair: 300006 succeeds, close_02 N 4 4 / 4 ok vs
#: 3 / 4); "0" = the check of before
BRACE_PROTECT_SWITCH = "ISPATIALGYM_QA_BRACE_PROTECT_ALLOWED"
BRACE_PROTECT_DEFAULT = "1"


def self_excluded_pairs() -> List[Tuple[str, str]]:
    """The robot link pairs ``robot_touching_itself`` leaves out: :data:`ELBOW_GRAZE_PAIRS` with the opt-in ``ISPATIALGYM_QA_SELF_EXCLUDE_ELBOW=1``
    (oracle-fix 2026-11-07), else none (the former metric). Their contacts are still recorded (``robot_touching_itself.excluded``)."""
    return list(ELBOW_GRAZE_PAIRS) if os.environ.get(SELF_EXCLUDE_SWITCH, "") == "1" else []


class _Spans:
    """Runs of consecutive steps per key (``first_step`` / ``last_step`` / ``n`` plus the fields of the run's first step), at
    most ``cap`` runs (later runs are counted in ``dropped``)."""

    def __init__(self, cap: int = 200):
        self.cap = int(cap)
        self.runs: List[Dict[str, Any]] = []
        self.dropped = 0
        self._open: Dict[tuple, Dict[str, Any]] = {}

    def add(self, step: int, key: tuple, **fields) -> bool:
        """Count ``key`` at ``step``; True when it starts a new run."""
        run = self._open.get(key)
        if run is not None and run["last_step"] == step - 1:
            run["last_step"], run["n"] = step, run["n"] + 1
            return False
        if run is not None and run["last_step"] == step:
            return False
        if len(self.runs) >= self.cap:
            self.dropped += 1
            self._open.pop(key, None)
            return True
        run = {**fields, "first_step": int(step), "last_step": int(step), "n": 1}
        self.runs.append(run)
        self._open[key] = run
        return True

    @property
    def steps(self) -> int:
        return sum(r["n"] for r in self.runs)

    def trim(self, step: int) -> None:
        keep = []
        for r in self.runs:
            if r["first_step"] > step:
                continue
            if r["last_step"] > step:
                r["last_step"] = step
                r["n"] = step - r["first_step"] + 1
            keep.append(r)
        kept = {id(r) for r in keep}
        self.runs = keep
        self._open = {k: r for k, r in self._open.items() if id(r) in kept}

    def report(self) -> Dict[str, Any]:
        return {"steps": self.steps, "runs": [dict(r) for r in self.runs], **({"runs_dropped": self.dropped} if self.dropped else {})}


@dataclass
class QAThresholds:
    """The ``--qa`` limits: BEHAVIOR's (``joylo/gello/utils/qa_utils.py`` ``ALL_QA_METRICS`` validate_kwargs, the metric
    kwargs of ``joylo/scripts/replay_data.py``, ``og_teleop_cfg.GHOST_APPEAR_THRESHOLD``) and the former oracle demo gates'.
    ``vel_avg`` / ``jerk_avg`` are raised above BEHAVIOR's teleoperation limits (0.15 / 100): the oracle turns held objects
    faster than an operator, the 70 eval demos reach 0.92 (mental rotation, object scaling) and 102."""

    step_dt: float = 1.0 / 30.0
    pause_vel: float = 0.001
    pause_steps: int = 50
    vel_avg: float = 1.2
    vel_prop_over_09max: float = 0.005
    acc_avg: float = 3.0
    jerk_avg: float = 150.0
    ghost_rad: float = 0.1
    head_link: str = "torso_link4"
    head_tilt_rad: float = 0.4
    nav_window_s: float = 3.0
    nav_trans_m_s: float = 0.1
    nav_rot_rad_s: float = 0.05
    head_tilt_steps: int = 30
    fov_margin: float = 0.05
    object_tilt_pos_m: float = 0.010
    object_tilt_rad: float = 0.04
    wobble_deg: float = 1.0
    #: oracle-fix 2026-11-03 (articulation_open_large#02, the dishwasher: 4 of 6 replays failed robot_wobble on ONE step of 1.003-1.093 deg
    #: in the base follow, the passing one peaked at 0.914): the hard verdict fires once the torso stayed over ``wobble_deg`` for this
    #: many consecutive steps (1 = the first step over, the owner's rule as written; ``ISPATIALGYM_QA_WOBBLE_MIN_STEPS``). Every step over
    #: is recorded in the report either way
    wobble_min_steps: int = field(default_factory=lambda: int(os.environ.get("ISPATIALGYM_QA_WOBBLE_MIN_STEPS", "1")))
    max_runs: int = 200
    #: policy-eval ``gripper_in_view``: steps after a close (open) command within which an assisted-grasp attach (detach)
    #: counts as that command's grasp (release)
    grasp_attach_steps: int = 30


def _quat_xyzw(q) -> Tuple[float, float, float, float]:
    x, y, z, w = (float(v) for v in th.as_tensor(q).reshape(-1)[:4])
    return x, y, z, w


def euler_pitch(q) -> float:
    """Euler Y of an xyzw quaternion, as ``omnigibson.utils.transform_utils.quat2euler`` returns it."""
    x, y, z, w = _quat_xyzw(q)
    s = 2.0 * (w * y - z * x)
    return math.copysign(math.pi / 2.0, s) if abs(s) >= 1.0 else math.asin(s)


def euler_yaw(q) -> float:
    """Euler Z of an xyzw quaternion (``quat2euler``)."""
    x, y, z, w = _quat_xyzw(q)
    return math.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)


def _runs(steps: Sequence[int]) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    for s in steps:
        if out and out[-1]["last_step"] == s - 1:
            out[-1]["last_step"], out[-1]["n"] = s, out[-1]["n"] + 1
        else:
            out.append({"first_step": int(s), "last_step": int(s), "n": 1})
    return out


class QA:
    """``run_isg --qa``: the demo QA metrics of one episode (module docstring). :meth:`observe` after every control step;
    :meth:`begin_action` / :meth:`end_action` around an oracle action (the objects it is meant to touch, the arms whose
    gripper-in-view span the execute hook ``fov`` opens and closes: :meth:`set_fov_enabled`); :meth:`note_rule` for each
    verdict of an oracle retry rule the soft metrics log (``RULE_METRICS``); :meth:`snapshot` / :meth:`restore` with the
    executor's checkpoints (the rolled-back steps leave the per-step records, the logged rule verdicts stay: they are
    attempts); :meth:`finish` after the settle windows; :meth:`report` for ``result.qa``. In an oracle rollout
    (``oracle=True``) :meth:`observe` / :meth:`finish` return the verdict of the first hard metric (the demo fails); a stream
    gets no verdict. ``roots``: the free-standing articulated objects whose joint the program works (``object_tilt``);
    ``task_objects``: the names marked in a stream's arm / hand contacts; ``contact_pairs``: a callable returning the robot's
    ``RigidContactAPI`` pairs of the step (default: queried here); ``rules``: whether the oracle's retry rules
    (:class:`Monitors`) feed :meth:`note_rule`, ``rule_limits`` their limits (:meth:`Monitors.rule_limits`)."""

    def __init__(self, robot, *, oracle: bool, scene=None, thresholds: Optional[QAThresholds] = None, camera: Optional[CameraModel] = None,
                 cam_in_zed=None, roots: Optional[Dict[str, Any]] = None, roots_skipped: Optional[Dict[str, str]] = None,
                 task_objects: Iterable[str] = (), contact_pairs: Optional[Callable[[], set]] = None,
                 ground_categories: frozenset = GROUND_CATEGORIES, rules: bool = True, rule_limits: Optional[Dict[str, Dict[str, Any]]] = None,
                 stream_fov: bool = False):
        self.robot = robot
        self.oracle = bool(oracle)
        self.scene = scene if scene is not None else getattr(robot, "scene", None)
        self.t = thresholds or QAThresholds()
        self.camera = camera
        self.cam_in_zed = cam_in_zed
        self.task_objects = set(task_objects)
        self.ground_categories = ground_categories
        self._contact_pairs_fn = contact_pairs
        self._contact_api = None
        self._contact_failed = False
        self._prim_cache: Dict[str, Optional[str]] = {}
        self._ready = False
        cap = self.t.max_runs
        self.touch_objects = _Spans(cap)
        self.touch_self = _Spans(cap)
        self.self_excluded = set(self_excluded_pairs())  # oracle-fix 2026-11-07: opt-in, recorded in the report
        self.touch_self_excluded = _Spans(cap)
        self.arm_hand = _Spans(cap)
        self.out_of_view = _Spans(cap)
        self.wobble = _Spans(cap)
        self.toggles_out_of_view: List[Dict[str, Any]] = []
        #: policy eval (``stream_fov``, a stream only): the grasps / releases (gripper command -> assisted-grasp attach / detach)
        self.stream_fov = bool(stream_fov) and not self.oracle
        self.stream_fov_fed = False
        self.grip_events: List[Dict[str, Any]] = []
        self.n_grip_events = {"grasp": 0, "release": 0}
        self._last_toggle: Dict[str, Dict[str, Any]] = {}
        self._held_arm_prev: Optional[Dict[str, Optional[str]]] = None
        self.rules = bool(rules)
        self.rule_limits = dict(rule_limits or {})
        #: per soft rule metric: the oracle's verdicts (attempts) and, in a stream, the runs of steps a rule fired
        self.rule_events: Dict[str, List[Dict[str, Any]]] = {m: [] for m in RULE_METRICS.values()}
        self.rule_runs: Dict[str, _Spans] = {m: _Spans(cap) for m in RULE_METRICS.values()}
        self.ghost_events: List[Dict[str, Any]] = []
        self._ghost_steps: Dict[str, set] = {arm: set() for arm in M.ARMS}
        self.pause_runs: List[Dict[str, Any]] = []
        self.max_pause = 0
        self._pause_streak = 0
        self._pause_start: Optional[int] = None
        self._q_prev: Optional[th.Tensor] = None
        self._open_prev: Dict[str, bool] = {}
        self._arm_q: Dict[int, th.Tensor] = {}
        self._head: Dict[int, Tuple[float, float, float, float]] = {}
        self.fov_span: Dict[str, bool] = {arm: False for arm in M.ARMS}
        self.fov_active = False
        self.arms: Tuple[str, ...] = ()
        self.intended: set = set()
        self.action_name: Optional[str] = None
        self.root_watch = RootWatch(roots or {}, roots_skipped) if (roots or roots_skipped) else None
        self.tilt_events: List[Dict[str, Any]] = []
        self._tilted: set = set()
        try:
            self.torso: Optional[TorsoLoad] = TorsoLoad(robot)
        except Exception as e:  # noqa: BLE001  (a robot without a trunk controller: no robot_wobble)
            log.warning("QA robot_wobble unavailable (%r)", e)
            self.torso = None
        self.bad_releases: List[Dict[str, Any]] = []
        self.goal_occlusion: Optional[Dict[str, Any]] = None
        self.jerky: Optional[Dict[str, Any]] = None
        #: the executor's ``ArmStutter.report()`` of the episode (:meth:`note_arm_stutter`, before :meth:`finish`)
        self.arm_stutter: Optional[Dict[str, Any]] = None
        self.head_upright: Optional[Dict[str, Any]] = None
        self.triggered: Dict[str, Dict[str, Any]] = {}
        self.stopped_by: Optional[str] = None
        self.n_steps = 0

    # ------------------------------------------------------------------ robot structure

    def _setup(self) -> None:
        if self._ready:
            return
        self._ready = True
        r = self.robot
        self.robot_prefix = (getattr(r, "prim_path", "") or "") + "/"
        self.arm_paths: set = set()
        for attr in ("arm_links", "gripper_links", "finger_links"):
            for links in (getattr(r, attr, None) or {}).values():
                for link in links:
                    path = getattr(link, "prim_path", None)
                    if path:
                        self.arm_paths.add(path)
        self._finger_sets = [set(M.FINGER_LINKS[arm]) for arm in M.ARMS]
        cidx = getattr(r, "controller_action_idx", None) or {}
        aidx = getattr(r, "arm_control_idx", None) or {}
        self.robot_arms = [arm for arm in M.ARMS if f"arm_{arm}" in cidx and arm in aidx]
        self._arm_cmd_idx = {arm: th.as_tensor(cidx[f"arm_{arm}"]).long().reshape(-1) for arm in self.robot_arms}
        self._arm_q_idx = {arm: th.as_tensor(aidx[arm]).long().reshape(-1) for arm in self.robot_arms}
        self._grip_idx = {arm: int(th.as_tensor(cidx[f"gripper_{arm}"]).reshape(-1)[0]) for arm in M.ARMS if f"gripper_{arm}" in cidx}
        self._arm_cat = th.cat([self._arm_q_idx[a] for a in self.robot_arms]) if self.robot_arms else None
        self._arm_vmax = None
        if self._arm_cat is not None:
            try:
                self._arm_vmax = th.tensor([float(j.max_velocity) for j in r.joints.values()], dtype=th.float64)[self._arm_cat]
            except Exception:  # noqa: BLE001  (no joint limits: the share above 0.9 x max is not measured)
                self._arm_vmax = None
        self._head_link = (getattr(r, "links", None) or {}).get(self.t.head_link)

    # ------------------------------------------------------------------ action context (oracle)

    def begin_action(self, action, intended: Iterable[str] = (), fov: bool = True) -> None:
        """``fov``: whether the oracle's gripper-out-of-FOV rule runs during ``action`` (``execute.fov_monitored``); outside
        such an action (the settle windows, the hold step after a restore) ``gripper_in_view`` measures nothing."""
        self.action_name = getattr(action, "name", None)
        self.arms = tuple(getattr(action, "arms", ()) or ())
        self.intended = set(intended)
        self.fov_active = bool(fov)

    def end_action(self) -> None:
        self.action_name = None
        self.arms = ()
        self.intended = set()
        self.fov_active = False

    def set_fov_enabled(self, enabled: bool = True) -> None:
        """The execute functions' hook (``ctx["hooks"]["fov"]``): open / close the gripper-in-view span of the current action's
        arms - open once the pre-grasp / pre-contact is reached, closed after the release / the end of the contact leg."""
        for arm in self.arms or ():
            self.fov_span[arm] = bool(enabled)

    def note_rule(self, verdict: Verdict, retried: bool = False, arms: Sequence[str] = (), objects: Sequence[str] = ()) -> None:
        """A verdict of an oracle retry rule a soft metric logs (``RULE_METRICS``; others are ignored): in an oracle rollout one
        event per failed attempt, ``retried`` whether the oracle retried the action (else it failed the demo; a verdict of the
        settle window is the action ``epilogue``, never retried); in a stream the steps the rule fired, as runs."""
        metric = RULE_METRICS.get(verdict.category)
        if metric is None:
            return
        det = verdict.details or {}
        arm = det.get("arm") or (list(arms)[0] if arms else None)
        obj = det.get("object") or (list(objects)[0] if objects else None)
        if not self.oracle:
            self.rule_runs[metric].add(int(verdict.step or 0), (arm, obj), arm=arm, object=obj, message=verdict.message)
            return
        ev = {"step": verdict.step, "action": verdict.action or "epilogue", "action_index": verdict.action_index, "attempt": verdict.attempt, "arm": arm,
              "object": obj, "message": verdict.message, "retried": bool(retried)}
        ev.update({k: det[k] for k in ("steps", "fov_out_steps", "fov_first_entry_step") if k in det})
        self.rule_events[metric].append(ev)

    # ------------------------------------------------------------------ per step

    def observe(self, step: int, action, acting: bool = True, live: bool = True, held: Iterable[str] = (),
                release_violations: Sequence[Dict[str, Any]] = (), settling: bool = False,
                held_by_arm: Optional[Dict[str, Optional[str]]] = None) -> Optional[Verdict]:
        """One control step, after ``env.step``. ``acting``: an action / stream step (False in the settle windows and on the
        settle rows of a replayed demo); ``live``: a hard metric of this step may stop the oracle (False on the hold step after
        a rollback); ``held``: names of the objects in hand; ``release_violations``: this step's assisted-grasp releases that
        did not come from a gripper command of +1; ``settling``: a hold / settle wait of the oracle's motion layer (not counted
        as a pause); ``held_by_arm``: arm -> the object its assisted grasp holds after the step (policy-eval
        ``gripper_in_view``)."""
        self._setup()
        self.n_steps += 1
        step = int(step)
        a = th.as_tensor(action, dtype=th.float32).reshape(-1)
        q = th.as_tensor(self.robot.get_joint_positions(), dtype=th.float32).reshape(-1)
        opens = {arm: float(a[i]) > 0.0 for arm, i in self._grip_idx.items() if i < a.numel()}
        hard: List[Tuple[str, str, Dict[str, Any]]] = []
        self._contacts(step, set(held), hard)
        if acting:
            if settling:
                self._close_pause(step - 1)
            else:
                self._pause(step, q, hard)
            if self._arm_cat is not None:
                self._arm_q[step] = q[self._arm_cat].clone()
            self._head_sample(step)
        else:
            self._close_pause(step - 1)
        self._ghost(step, q, a, opens, hard)
        if self.oracle and self.fov_active:
            self._fov(step, q, opens)
        elif self.stream_fov and held_by_arm is not None:
            self._grip_fov(step, q, opens, held_by_arm)
        self._open_prev = opens
        self._q_prev = q.clone()
        self._tilt(step, hard)
        self._wobble(step, a, hard)
        for v in release_violations:
            ev = {**dict(v), "action": self.action_name}
            self.bad_releases.append(ev)
            hard.append(("bad_release", f"{v.get('arm')} released {v.get('object')} without a gripper command of +1 ({v.get('kind')})", ev))
        for metric, msg, det in hard:
            self.triggered.setdefault(metric, {"step": step, "message": msg})
        if self.oracle and live and hard and self.stopped_by is None:
            metric, msg, det = hard[0]
            self.stopped_by = metric
            return Verdict(metric, msg, action=self.action_name, step=step, details={"qa": HARD, **_jsonable(det)})
        return None

    # -- contacts --------------------------------------------------------------------------------------------------

    def _pairs(self) -> set:
        if self._contact_pairs_fn is not None:
            try:
                return set(self._contact_pairs_fn() or ())
            except Exception as e:  # noqa: BLE001
                log.warning("QA contact pairs unavailable (%r); querying them here", e)
                self._contact_pairs_fn = None
        if self._contact_failed:
            return set()
        try:
            if self._contact_api is None:
                from omnigibson.utils.usd_utils import RigidContactAPI

                self._contact_api = RigidContactAPI
            return set(self._contact_api.get_contact_pairs(int(getattr(self.scene, "idx", 0) or 0), [self.robot], None, current_only=False))
        except Exception as e:  # noqa: BLE001  (no contact view: the contact metrics are not measured, reported once)
            log.warning("QA contact metrics disabled: %r", e)
            self._contact_failed = True
            return set()

    def _object(self, body_path: str) -> Optional[str]:
        if body_path in self._prim_cache:
            return self._prim_cache[body_path]
        name = None
        try:
            obj = self.scene.object_registry("prim_path", body_path.rsplit("/", 1)[0], None)
        except Exception:  # noqa: BLE001
            obj = None
        if obj is not None and getattr(obj, "category", None) not in self.ground_categories:
            name = obj.name
        self._prim_cache[body_path] = name
        return name

    def _contacts(self, step: int, held: set, hard: list) -> None:
        pairs = self._pairs()
        if not pairs:
            return
        prefix = self.robot_prefix
        objects, selfs, hands, excluded = set(), set(), set(), set()
        for own, other in pairs:
            if not own.startswith(prefix):
                continue  # a held object's contact (the retry rule's query includes them)
            own_link = own.rsplit("/", 1)[1]
            if other.startswith(prefix):
                pair = tuple(sorted((own_link, other.rsplit("/", 1)[1])))
                # the two fingers of one gripper meet in the closed fist of a push / brace
                if other != own and not any(set(pair) <= fingers for fingers in self._finger_sets):
                    (excluded if pair in self.self_excluded else selfs).add(pair)
                continue
            name = self._object(other)
            if name is None:
                continue  # a floor, or a body of no scene object
            if own in self.arm_paths:
                if name not in self.intended and name not in held:
                    hands.add((own_link, name))
            else:
                objects.add((own_link, name))
        for link, name in sorted(objects):
            self.touch_objects.add(step, (link, name), link=link, object=name, action=self.action_name)
        for pair in sorted(selfs):
            self.touch_self.add(step, pair, links=list(pair), action=self.action_name)
        for pair in sorted(excluded):
            self.touch_self_excluded.add(step, pair, links=list(pair), action=self.action_name)
        for link, name in sorted(hands):
            self.arm_hand.add(step, (link, name), link=link, object=name, task_object=name in self.task_objects, action=self.action_name)
        if objects:
            hard.append(("robot_touching_objects", "robot " + ", ".join(f"{l} touches {o}" for l, o in sorted(objects)), {"contacts": sorted(objects)}))
        if selfs:
            hard.append(("robot_touching_itself", "robot " + ", ".join(f"{a} touches {b}" for a, b in sorted(selfs)), {"links": sorted(selfs)}))

    # -- pause -----------------------------------------------------------------------------------------------------

    def _pause(self, step: int, q: th.Tensor, hard: list) -> None:
        if self._q_prev is None or self._q_prev.shape != q.shape:
            return
        if bool(th.any(th.abs(q - self._q_prev) / self.t.step_dt > self.t.pause_vel)):
            self._close_pause(step - 1)
            return
        if self._pause_streak == 0:
            self._pause_start = step
        self._pause_streak += 1
        self.max_pause = max(self.max_pause, self._pause_streak)
        if self._pause_streak == self.t.pause_steps + 1:
            hard.append(("pause", f"no joint moved faster than {self.t.pause_vel} for {self._pause_streak} steps (limit {self.t.pause_steps})",
                         {"first_step": self._pause_start}))

    def _close_pause(self, last_step: int) -> None:
        if self._pause_streak > self.t.pause_steps:
            self.pause_runs.append({"first_step": self._pause_start, "last_step": int(last_step), "n": self._pause_streak})
        self._pause_streak = 0
        self._pause_start = None

    # -- ghost hand ------------------------------------------------------------------------------------------------

    def _ghost(self, step: int, q: th.Tensor, a: th.Tensor, opens: Dict[str, bool], hard: list) -> None:
        for arm in self.robot_arms:
            ci, qi = self._arm_cmd_idx[arm], self._arm_q_idx[arm]
            if a.numel() <= int(ci.max()) or q.numel() <= int(qi.max()) or ci.numel() != qi.numel():
                continue
            err = float(th.max(th.abs(q[qi] - a[ci])))
            if err <= self.t.ghost_rad:
                continue
            self._ghost_steps[arm].add(step)
            if opens.get(arm) and self._open_prev.get(arm) is False:
                ev = {"step": step, "arm": arm, "error_rad": round(err, 4), "action": self.action_name}
                self.ghost_events.append(ev)
                hard.append(("ghost_hand", f"{arm} arm {err:.3f} rad off its commanded joints when its gripper opened (> {self.t.ghost_rad})", ev))

    # -- gripper in view (oracle) ----------------------------------------------------------------------------------

    def _camera_pose(self, q: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        torso = q[th.as_tensor(self.robot.trunk_control_idx).long()].float()
        bp, bq = self.robot.get_position_orientation()
        return C.camera_pose(torso, base_pose=(bp.float(), bq.float()), cam_in_zed=self.cam_in_zed)

    def _in_view(self, cam, arm: str) -> bool:
        ep, eq = self.robot.eef_links[arm].get_position_orientation()
        return self.camera.in_frame(cam, C.fingertip((ep.float(), eq.float())), self.t.fov_margin)

    def _fov(self, step: int, q: th.Tensor, opens: Dict[str, bool]) -> None:
        if self.camera is None:
            return
        toggled = [arm for arm, o in opens.items() if self.fov_span.get(arm) and arm in self._open_prev and self._open_prev[arm] != o]
        watched = [arm for arm in M.ARMS if self.fov_span.get(arm)]
        if not watched and not toggled:
            return
        try:
            cam = self._camera_pose(q)
            for arm in watched:
                if not self._in_view(cam, arm):
                    self.out_of_view.add(step, (arm,), arm=arm, action=self.action_name)
            for arm in toggled:
                if not self._in_view(cam, arm):
                    self.toggles_out_of_view.append({"step": step, "arm": arm, "command": "open" if opens[arm] else "close", "action": self.action_name})
        except Exception as e:  # noqa: BLE001  (a robot without the conventions' camera chain: not measured, reported once)
            log.warning("QA gripper_in_view disabled: %r", e)
            self.camera = None

    def _grip_fov(self, step: int, q: th.Tensor, opens: Dict[str, bool], held_by_arm: Dict[str, Optional[str]]) -> None:
        """Policy eval: a gripper command toggle records whether the arm's fingertips are in the frame (the camera pose is only
        computed on a toggle); an assisted-grasp attach (detach) within ``grasp_attach_steps`` of the arm's latest close (open)
        command is that command's grasp (release), out of view if the fingertips were out of the frame at the command. An attach
        / detach with no such command (a release outside +1 is ``bad_release``) is not judged here."""
        held = {arm: held_by_arm.get(arm) for arm in M.ARMS}
        prev, self._held_arm_prev = self._held_arm_prev, held
        if self.camera is None:
            return
        self.stream_fov_fed = True
        try:
            cam = None
            for arm, o in opens.items():
                if arm in self._open_prev and self._open_prev[arm] != o:
                    cam = self._camera_pose(q) if cam is None else cam
                    self._last_toggle[arm] = {"step": step, "command": "open" if o else "close", "in_view": bool(self._in_view(cam, arm))}
        except Exception as e:  # noqa: BLE001  (a robot without the conventions' camera chain: not measured, reported once)
            log.warning("QA gripper_in_view (policy eval) disabled: %r", e)
            self.camera = None
            self.stream_fov_fed = False
            return
        if prev is None:
            return
        k = int(self.t.grasp_attach_steps)
        for arm in M.ARMS:
            before, now = prev.get(arm), held.get(arm)
            if before == now:
                continue
            tog = self._last_toggle.get(arm)
            for kind, command, obj in (("release", "open", before), ("grasp", "close", now)):
                if obj is None or tog is None or tog["command"] != command or step - tog["step"] > k:
                    continue
                self.n_grip_events[kind] += 1
                if not tog["in_view"] and len(self.grip_events) < self.t.max_runs:
                    self.grip_events.append({"step": step, "kind": kind, "arm": arm, "object": obj, "command_step": tog["step"],
                                             "after_steps": step - tog["step"]})

    # -- head camera upright ---------------------------------------------------------------------------------------

    def _head_sample(self, step: int) -> None:
        if self._head_link is None:
            return
        try:
            _hp, hq = self._head_link.get_position_orientation()
            bp, bq = self.robot.get_position_orientation()
        except Exception:  # noqa: BLE001
            return
        self._head[step] = (euler_pitch(hq), float(bp[0]), float(bp[1]), euler_yaw(bq))

    def _head_camera_upright(self) -> Dict[str, Any]:
        t = self.t
        out: Dict[str, Any] = {"measured": self._head_link is not None, "link": t.head_link, "limit_steps": t.head_tilt_steps,
                               "tilt_rad": t.head_tilt_rad, "navigation_window_s": t.nav_window_s}
        steps = sorted(self._head)
        tilted: List[int] = []
        window = int(t.nav_window_s / t.step_dt)
        run = 0
        for prev, s in zip(steps, steps[1:]):
            pitch, x, y, yaw = self._head[s]
            _pp, px, py, pyaw = self._head[prev]
            trans = math.hypot(x - px, y - py) / t.step_dt
            rot = abs((yaw - pyaw + math.pi) % (2.0 * math.pi) - math.pi) / t.step_dt
            run = run + 1 if (trans > t.nav_trans_m_s or rot > t.nav_rot_rad_s) else 0
            if run >= window and abs(pitch) > t.head_tilt_rad:
                tilted.append(s)
        out["steps_tilted_while_navigating"] = len(tilted)
        out["runs"] = _runs(tilted)
        out["max_abs_pitch_rad"] = round(max((abs(v[0]) for v in self._head.values()), default=0.0), 4)
        return out

    # -- jerky motion ----------------------------------------------------------------------------------------------

    def _jerky_motion(self) -> Dict[str, Any]:
        t = self.t
        limits = {"vel_avg": t.vel_avg, "vel_prop_over_09max": t.vel_prop_over_09max, "acc_avg": t.acc_avg, "jerk_avg": t.jerk_avg}
        steps = sorted(self._arm_q)
        if len(steps) < 4:
            return {"measured": False, "steps": len(steps), "limits": limits, "failed": []}
        pos = th.stack([self._arm_q[s] for s in steps]).double()
        vels = (pos[1:] - pos[:-1]) / t.step_dt
        accs = (vels[1:] - vels[:-1]) / t.step_dt
        jerks = (accs[1:] - accs[:-1]) / t.step_dt
        vels, accs, jerks = vels.abs(), accs.abs(), jerks.abs()
        values = {"vel_avg": float(vels.mean(dim=0).max()), "acc_avg": float(accs.mean(dim=0).max()), "jerk_avg": float(jerks.mean(dim=0).max()),
                  "vel_prop_over_09max": None}
        if self._arm_vmax is not None and self._arm_vmax.numel() == vels.shape[1]:
            values["vel_prop_over_09max"] = float(th.any(vels > 0.9 * self._arm_vmax.unsqueeze(0), dim=-1).sum()) / vels.shape[0]
        failed = [k for k, lim in limits.items() if values.get(k) is not None and values[k] > lim]
        return {"measured": True, "steps": len(steps), "values": {k: (None if v is None else round(v, 5)) for k, v in values.items()},
                "limits": limits, "failed": failed, "per_joint_value": "the largest over the arm joints (each joint's own mean must pass)"}

    # -- object tilt / robot wobble --------------------------------------------------------------------------------

    def _tilt(self, step: int, hard: list) -> None:
        if self.root_watch is None or not self.root_watch.objects:
            return
        t = self.t
        for name, cur in self.root_watch.observe(self.action_name, step).items():
            over = cur["displacement_m"] > t.object_tilt_pos_m or cur["tilt_rad"] > t.object_tilt_rad
            if over and name not in self._tilted:
                self._tilted.add(name)
                ev = {"step": step, "object": name, "displacement_m": round(cur["displacement_m"], 5), "tilt_rad": round(cur["tilt_rad"], 5),
                      "action": self.action_name}
                self.tilt_events.append(ev)
                hard.append(("object_tilt", f"{name} moved {cur['displacement_m'] * 1000:.1f} mm / tipped {math.degrees(cur['tilt_rad']):.2f} deg "
                                            f"(band {t.object_tilt_pos_m * 1000:.0f} mm / {math.degrees(t.object_tilt_rad):.2f} deg)", ev))
            elif not over:
                self._tilted.discard(name)

    def _wobble(self, step: int, a: th.Tensor, hard: list) -> None:
        if self.torso is None:
            return
        try:
            dev = self.torso.observe(step, a, self.action_name)
        except Exception as e:  # noqa: BLE001  (no trunk reading: the metric then has nothing to judge)
            log.warning("QA robot_wobble disabled (%r)", e)
            self.torso = None
            return
        if dev > self.t.wobble_deg:
            self.wobble.add(step, ("torso",), deg=round(dev, 3), action=self.action_name)
            run = self.wobble._open.get(("torso",))
            n = int(run["n"]) if run is not None and run.get("last_step") == step else 1
            if n == max(1, int(self.t.wobble_min_steps)):
                hard.append(("robot_wobble", f"a load bent the torso {dev:.2f} deg off its command (> {self.t.wobble_deg:.2f} deg"
                             + (f" for {n} steps" if n > 1 else "") + "): the robot visibly wobbles", {"deg": round(dev, 3), "steps": n}))

    # ------------------------------------------------------------------ checkpoints

    def snapshot(self) -> Dict[str, Any]:
        return {"fov_span": dict(self.fov_span), "pause_streak": self._pause_streak, "pause_start": self._pause_start,
                "q_prev": None if self._q_prev is None else self._q_prev.clone(), "open_prev": dict(self._open_prev),
                "root_watch": self.root_watch.snapshot() if self.root_watch is not None else None, "tilted": set(self._tilted),
                "last_toggle": {k: dict(v) for k, v in self._last_toggle.items()}, "held_arm_prev": self._held_arm_prev}

    def restore(self, snap: Dict[str, Any], step: int) -> None:
        """Back to a checkpoint taken at ``step``: the rolled-back steps leave every per-step record (the logged rule verdicts
        stay: they are attempts)."""
        step = int(step)
        self.fov_span = dict(snap["fov_span"])
        self._pause_streak, self._pause_start = snap["pause_streak"], snap["pause_start"]
        self._q_prev = snap["q_prev"]
        self._open_prev = dict(snap["open_prev"])
        self._tilted = set(snap["tilted"])
        self._last_toggle = {k: dict(v) for k, v in (snap.get("last_toggle") or {}).items()}
        self._held_arm_prev = snap.get("held_arm_prev")
        if self.root_watch is not None:
            self.root_watch.load(snap["root_watch"])
        if self.torso is not None:
            self.torso.trim(step)
        for spans in (self.touch_objects, self.touch_self, self.arm_hand, self.out_of_view, self.wobble):
            spans.trim(step)
        self._arm_q = {s: v for s, v in self._arm_q.items() if s <= step}
        self._head = {s: v for s, v in self._head.items() if s <= step}
        self._ghost_steps = {arm: {s for s in v if s <= step} for arm, v in self._ghost_steps.items()}
        for name in ("toggles_out_of_view", "ghost_events", "tilt_events", "bad_releases", "pause_runs", "grip_events"):
            setattr(self, name, [e for e in getattr(self, name) if int(e.get("step", e.get("first_step", 0)) or 0) <= step])
        self.triggered = {k: v for k, v in self.triggered.items() if (v["step"] or 0) <= step}

    # ------------------------------------------------------------------ episode end

    def note_arm_stutter(self, report: Optional[Dict[str, Any]]) -> None:
        """The episode's arm stutter fact (the executor's ``ArmStutter.report()``; empty / None: not measured), judged by :meth:`finish`."""
        self.arm_stutter = dict(report) if report else None

    def finish(self, goal_occlusion: Optional[Callable[[], Dict[str, Any]]] = None, step: Optional[int] = None) -> Optional[Verdict]:
        """After the settle windows: jerky motion and head camera upright over the recorded action steps, the arm stutter fact
        (:meth:`note_arm_stutter`), goal occlusion from ``goal_occlusion()`` (None: not measured). Returns the hard verdict of an
        oracle rollout (None otherwise)."""
        self._setup()
        self._close_pause(step if step is not None else max(self._arm_q, default=0))
        self.jerky = self._jerky_motion()
        self.head_upright = self._head_camera_upright()
        if goal_occlusion is not None:
            try:
                self.goal_occlusion = goal_occlusion()
            except Exception as e:  # noqa: BLE001
                log.warning("QA goal_occlusion not measured: %r", e)
                self.goal_occlusion = {"error": repr(e)}
        hard: List[Tuple[str, str, Dict[str, Any]]] = []
        if self.jerky.get("failed"):
            vals = self.jerky["values"]
            hard.append(("jerky_motion", "arm motion above BEHAVIOR's limits: " + ", ".join(f"{k} {vals[k]} > {self.jerky['limits'][k]}" for k in self.jerky["failed"]),
                         {"failed": list(self.jerky["failed"])}))
        if (self.arm_stutter or {}).get("flagged"):
            w, thr = self.arm_stutter["worst"], self.arm_stutter["thresholds"]
            hard.append(("arm_stutter", f"the {w['arm']} arm's commanded joint motion jolted {w['n_jolts']} times in steps {w['first_step']}-{w['last_step']} "
                                        f"({w['action'] or 'stream'}; velocity changes >= {thr['jolt_rad_s']} rad/s per step, up to {w['max_jolt_rad_s']:.2f}, "
                                        f">= {thr['min_jolts']} within {thr['window_steps']} steps): the robot visibly stutters",
                         {"arm": w["arm"], "first_step": w["first_step"], "last_step": w["last_step"]}))
        if (self.goal_occlusion or {}).get("occluded"):
            go = self.goal_occlusion
            hard.append(("goal_occlusion", f"{go.get('worst')} is {float(go.get('max_occluded') or 0.0):.0%} hidden from the head camera at the end "
                                           f"(>= {float(go.get('threshold') or 0.0):.0%})", {"worst": go.get("worst")}))
        for metric, msg, _det in hard:
            self.triggered.setdefault(metric, {"step": step, "message": msg})
        if self.oracle and hard and self.stopped_by is None:
            metric, msg, det = hard[0]
            self.stopped_by = metric
            return Verdict(metric, msg, step=step, details={"qa": HARD, **_jsonable(det)})
        return None

    # ------------------------------------------------------------------ report

    def report(self) -> Dict[str, Any]:
        self._setup()
        t = self.t
        m: Dict[str, Dict[str, Any]] = {}
        m["robot_touching_objects"] = {"limit_steps": 0, **self.touch_objects.report()}
        m["robot_touching_itself"] = {"limit_steps": 0, **self.touch_self.report(), "switch": SELF_EXCLUDE_SWITCH,
                                      "excluded_pairs": [list(p) for p in sorted(self.self_excluded)], "excluded": self.touch_self_excluded.report()}
        m["pause"] = {"limit_steps": t.pause_steps, "vel_threshold": t.pause_vel, "max_pause_steps": self.max_pause, "runs": list(self.pause_runs)}
        m["jerky_motion"] = dict(self.jerky or {"measured": False})
        m["ghost_hand"] = {"threshold_rad": t.ghost_rad, "limit_while_ungrasping": 0,
                           "per_arm": {arm: {"n_steps_total": len(self._ghost_steps.get(arm, ())),
                                             "n_steps_while_ungrasping": sum(1 for e in self.ghost_events if e["arm"] == arm)} for arm in M.ARMS},
                           "events": list(self.ghost_events)}

        def rule(metric: str, measured: bool) -> Dict[str, Any]:
            out: Dict[str, Any] = {"retried_by_oracle": True, "measured": measured}
            if metric in self.rule_limits:
                out["limits"] = dict(self.rule_limits[metric])
            if self.oracle:
                out["events"] = list(self.rule_events[metric])
            else:
                out.update(self.rule_runs[metric].report())
            return out

        m["held_object_drift"] = rule("held_object_drift", self.rules)
        m["stalled"] = rule("stalled", self.rules)
        if self.oracle:
            m["gripper_in_view"] = {**rule("gripper_in_view", self.camera is not None), "out_of_view": self.out_of_view.report(),
                                    "grasp_changes_outside_view": list(self.toggles_out_of_view)}
            m["failed_grasp"] = {"limit": 0, **rule("failed_grasp", True)}
        else:
            if self.stream_fov and self.stream_fov_fed:
                m["gripper_in_view"] = {"measured": True, "definition": "policy eval: grasps / releases with the fingertips out of the frame",
                                        "attach_window_steps": int(t.grasp_attach_steps), "margin": t.fov_margin,
                                        "grasps": self.n_grip_events["grasp"], "releases": self.n_grip_events["release"],
                                        "grasps_out_of_view": [e for e in self.grip_events if e["kind"] == "grasp"],
                                        "releases_out_of_view": [e for e in self.grip_events if e["kind"] == "release"]}
            else:
                m["gripper_in_view"] = {"measured": False, "why": "no head camera model" if self.stream_fov else "oracle / policy eval only"}
            m["failed_grasp"] = {"measured": False, "why": "oracle only"}
        m["head_camera_upright"] = dict(self.head_upright or {"measured": False})
        m["arm_hand_contacts"] = self.arm_hand.report()
        m["object_tilt"] = ({"measured": True, **self.root_watch.report(t.object_tilt_pos_m, t.object_tilt_rad), "events": list(self.tilt_events)}
                            if self.root_watch is not None else {"measured": False, "why": "no free-standing articulated object is opened or closed"})
        m["goal_occlusion"] = {"measured": self.goal_occlusion is not None, "report": self.goal_occlusion}
        m["robot_wobble"] = ({"limit_deg": t.wobble_deg, **self.torso.report(), "over_limit": self.wobble.report()} if self.torso is not None else {"measured": False})
        m["bad_release"] = {"limit": 0, "violations": list(self.bad_releases)}
        m["arm_stutter"] = dict(self.arm_stutter) if self.arm_stutter is not None else {"measured": False}
        hit = {
            "robot_touching_objects": self.touch_objects.steps > 0,
            "robot_touching_itself": self.touch_self.steps > 0,
            "pause": self.max_pause > t.pause_steps,
            "jerky_motion": bool((self.jerky or {}).get("failed")),
            "ghost_hand": bool(self.ghost_events),
            "held_object_drift": bool(self.rule_events["held_object_drift"]) or self.rule_runs["held_object_drift"].steps > 0,
            "stalled": bool(self.rule_events["stalled"]) or self.rule_runs["stalled"].steps > 0,
            "gripper_in_view": (bool(self.out_of_view.runs or self.toggles_out_of_view or self.rule_events["gripper_in_view"]) if self.oracle
                                else bool(self.grip_events)),
            "failed_grasp": self.oracle and bool(self.rule_events["failed_grasp"]),
            "head_camera_upright": (self.head_upright or {}).get("steps_tilted_while_navigating", 0) > t.head_tilt_steps,
            "arm_hand_contacts": self.arm_hand.steps > 0,
            "object_tilt": bool(self.tilt_events),
            "goal_occlusion": bool((self.goal_occlusion or {}).get("occluded")),
            "robot_wobble": bool(self.wobble.runs),
            "bad_release": bool(self.bad_releases),
            "arm_stutter": bool((self.arm_stutter or {}).get("flagged")),
        }
        for name, rec in m.items():
            rec["mode"] = QA_METRICS[name]["mode"]
            rec["triggered"] = bool(hit[name])
            if name in self.triggered:
                rec["first"] = dict(self.triggered[name])
        return {"enabled": True, "mode": "oracle" if self.oracle else "stream", "metrics": _jsonable(m),
                "hard_triggered": [k for k in HARD_METRICS if hit[k]], "soft_triggered": [k for k in SOFT_METRICS if hit[k]],
                "stopped_by": self.stopped_by, "steps_observed": self.n_steps}


def is_hard_qa(verdict: Optional[Verdict]) -> bool:
    """A verdict of a QA hard metric (it fails an oracle demo at once, without a retry)."""
    return verdict is not None and (verdict.details or {}).get("qa") == HARD


__all__ = [
    "CATEGORIES", "STEP_CATEGORIES", "EPILOGUE_CATEGORIES", "GROUND_CATEGORIES", "Verdict", "category_of", "QAStats",
    "CameraModel", "QAConfig", "Monitors", "HARD", "SOFT", "QA_METRICS", "HARD_METRICS", "SOFT_METRICS", "RULE_METRICS", "QAThresholds", "QA",
    "TorsoLoad", "RootWatch", "is_hard_qa", "euler_pitch", "euler_yaw", "StutterConfig", "find_stutters", "ArmStutter", "stutter_report_hdf5",
]


if __name__ == "__main__":
    raise SystemExit(_main())
