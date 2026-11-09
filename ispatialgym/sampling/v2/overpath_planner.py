"""i7-overpath P1 2026-11-08 (owner decision 4, ``owner_20261108g``; inside_large_01): the overhang pick's transit over the book planned by
the ORACLE'S OWN motion planner in the sampler, and its joint trajectory recorded (``grasps[0].overpath``, :mod:`ispatialgym.oracle.overpath`).

The Cartesian transit (rise / across / down, :func:`overpath.leg_plan`) planned 0 of 27 (i7op3_diag): the push configuration and the
pre-grasp configuration lie on different IK branches, so the transit must include a joint-space move - as the oracle's live planner does.
This module runs that live planner kinematically: :class:`PlanOnlyMotions` is an oracle :class:`~ispatialgym.oracle.motions.Motions`
on the SAMPLER's cuRobo backend (the same ``R1ProCuRobo`` class, its world synced from the sampler's scene) whose state is a joint vector
instead of the simulated robot - ``execute_joint_trajectory`` records the trajectory and moves that vector, nothing is simulated. The
transit is the oracle's code path for this grasp (``execute._grasp_transit``'s supports-present branch, the one the live demos take):
``Motions.protecting`` (the book protected, inflated by :data:`overpath.BOOK_MARGIN_M` instead of ``protection.INFLATION_M``) around
``Motions.transit_dip_gate`` around ``execute.reach_pre_pose`` -> ``Motions.arm_to(pose=pre-grasp, q_hint=witness)``: the checked straight
ramp, the ramp via a posture, the joint-space plans to the witness and their retries, the pose-goal fallback, the plan via a posture, the
dipping-transit gate's Cartesian leg, the protected-object gate's via-above and re-plans - with the oracle's ``MotionsConfig`` /
``ExecuteConfig`` (the train-fast ``PLAN_LADDER`` attempts as ``run_isg`` sets them). Not replicated: the supports-ignored fallbacks of
``_grasp_transit`` (the owner: the desk and the scene are obstacles) and the backend's trajopt seed count (the sampler's backend keeps
its 4 seeds per attempt; the oracle's PLAN_LADDER builds its own with 2).

The caller (``IORSLarge._overpath_plan_planner``) puts the book at its settled post-push pose in the scene first, so the planner's world
and the protected box are the post-push ones, and re-checks the recorded path as the oracle will run it."""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th

log = logging.getLogger("ispatialgym.sampling.v2")


def oracle_motions_config():
    """The ``MotionsConfig`` the oracle runs with (``run_isg``: the defaults, the train-fast ``PLAN_LADDER`` attempts per round /
    retry rounds to a witness when that switch is on)."""
    from ispatialgym.fastflags import fast_on, fast_value
    from ispatialgym.oracle.motions import MotionsConfig

    mcfg = MotionsConfig()
    if fast_on("PLAN_LADDER"):
        mcfg.plan_attempts = int(fast_value("PLAN_ATTEMPTS", 2))
        mcfg.witness_plan_retries = int(fast_value("WITNESS_PLAN_RETRIES", 1))
    return mcfg


class _KinematicBackend:
    """The sampler's cuRobo backend seen from :class:`PlanOnlyMotions`: every call forwarded, the robot state it would read from the
    simulator (``q_full``, the start / seed of a plan or an IK path given none, ``fk`` without a configuration) taken from the motions'
    joint vector instead."""

    def __init__(self, real, owner: Any):
        self._real = real
        self._owner = owner

    def __getattr__(self, name):
        return getattr(self._real, name)

    def q_full(self) -> th.Tensor:
        return self._owner._q.clone()

    def fk(self, frames, q_full=None, **kw):
        return self._real.fk(frames, q_full=self._owner._q.clone() if q_full is None else q_full, **kw)

    def solve_ik_path(self, emb, waypoints, q_start=None, **kw):
        return self._real.solve_ik_path(emb, waypoints, q_start=self._owner._q.clone() if q_start is None else q_start, **kw)

    def solve_ik(self, emb, targets, q_seed=None, **kw):
        return self._real.solve_ik(emb, targets, q_seed=self._owner._q.clone() if q_seed is None else q_seed, **kw)

    def plan_pose(self, emb, targets, q_start=None, **kw):
        if getattr(self._owner, "straight_only", False):
            return _no_plan("plan_pose")
        return self._real.plan_pose(emb, targets, q_start=self._owner._q.clone() if q_start is None else q_start, **kw)

    def plan_joint(self, emb, q_goal, q_start=None, **kw):
        if getattr(self._owner, "straight_only", False):
            return _no_plan("plan_joint")
        return self._real.plan_joint(emb, q_goal, q_start=self._owner._q.clone() if q_start is None else q_start, **kw)


def _no_plan(what: str):
    """i7-startdraw2 (``straight_only``): a trajectory-optimisation call answered as failed - only the oracle's checked straight ramps
    (direct / via a posture) and its Cartesian legs carry the transit."""
    from ispatialgym.oracle.curobo_planner import PlanResult

    return PlanResult(success=False, status=f"straight_only: {what} not run", diagnostics={"straight_only": True})


def _motions_cls():
    from ispatialgym.oracle.motions import Motions

    class PlanOnlyMotions(Motions):
        """An oracle ``Motions`` without a simulator: its state is the joint vector ``_q`` (28-D); a streamed trajectory is recorded
        (``segments``) and moves ``_q`` to its end; the planning, checks and remedies are the oracle's own methods."""

        def __init__(self, robot, backend, q0, cfg=None):  # noqa: D401 - the attributes Motions.__init__ sets that planning reads
            from ispatialgym.oracle.motions import MotionsConfig

            self.env = None
            self.robot = robot
            self._q = th.as_tensor(q0).reshape(-1).float().clone()
            self.backend = _KinematicBackend(backend, self)
            self.cfg = cfg or MotionsConfig()
            self.dt = float(getattr(backend, "control_dt", 1.0 / 30.0))
            self.stop_requested = False
            self.follow_phase = "arm"
            self.last_relaxed_links: List[str] = []
            self.last_witness_residual = None
            self.last_jam = None
            self.jams: List[Dict[str, Any]] = []
            self.keep_hold_arms = set()
            self._protect = None
            self.protect_events: List[Dict[str, Any]] = []
            self.protect_checks = 0
            self.witness_branch_accepted_events: List[Dict[str, Any]] = []
            self.witness_via_posture_events: List[Dict[str, Any]] = []
            self._dip = None
            self.straight_arm_stats: Dict[str, Any] = {"straight_arm_ok": 0, "straight_arm_fallback": 0, "waypoints": 0, "check_s": 0.0, "fallback_why": {}}
            self.tracker = None
            self.last_steps = 0
            self.settling = False
            self.gripper_intent_closed: Dict[str, bool] = {}
            self.release_events: List[Dict[str, Any]] = []
            self.transit_preshape: Dict[str, float] = {}
            self.place_shifts: Dict[str, th.Tensor] = {}
            self.segments: List[th.Tensor] = []

        # ---- the state reads of the live Motions, from the joint vector
        def q_full(self) -> th.Tensor:
            return self._q.clone()

        def eef_pose(self, arm: str):
            p, q = self._real_fk(arm)
            return th.as_tensor(p).float().reshape(3).clone(), th.as_tensor(q).float().reshape(4).clone()

        def _real_fk(self, arm: str):
            from ispatialgym.oracle import r1pro_model as M

            frame = M.EEF_LINK[arm]
            return self.backend.fk([frame], self._q.clone())[frame]

        def _deflection(self, arm: str):
            return None  # a kinematic arm is never wound up under a load

        # ---- the streamed motion: recorded, never simulated
        def execute_joint_trajectory(self, q_traj, arms, trunk: bool = False, tol=None, loaded: bool = False, converge: bool = True):
            from ispatialgym.oracle.motions import SkillError

            q_traj = th.as_tensor(q_traj).detach().float().cpu()
            if q_traj.ndim == 1:
                q_traj = q_traj.reshape(1, -1)
            if len(q_traj) > self.cfg.trajectory_max_steps:
                raise SkillError("timeout", f"trajectory of {len(q_traj)} steps exceeds the budget {self.cfg.trajectory_max_steps}")
            idx = th.cat([th.as_tensor(self.robot.arm_control_idx[a]).reshape(-1).long() for a in arms]) if arms else th.zeros(0, dtype=th.long)
            rows = []
            for q in q_traj:
                r = self._q.clone()
                r[idx] = q.reshape(-1)[idx]
                rows.append(r)
            if rows:
                self.segments.append(th.stack(rows))
                self._q = rows[-1].clone()
            return
            yield  # a generator, as the live one (nothing is emitted)

    return PlanOnlyMotions


def plan_only_motions(robot, backend, q0, cfg=None, straight_only: bool = False):
    """A :class:`PlanOnlyMotions` (an oracle ``Motions`` on ``backend``, its state ``q0``) - see the module docstring. ``straight_only``
    (i7-startdraw2): every trajopt call fails at once, so the ladder carries the transit only by its checked straight ramps."""
    mot = _motions_cls()(robot, backend, q0, cfg=cfg if cfg is not None else oracle_motions_config())
    mot.straight_only = bool(straight_only)
    return mot


def _drain(gen) -> Any:
    """Run a motion generator to its end (a plan-only motion emits nothing); its return value."""
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value


def branch_of(motions, ctx: Dict[str, Any], dip: Optional[Dict[str, Any]]) -> str:
    """Which part of the oracle's ladder produced the transit (kit ``overpath.planner.branch.<b>``)."""
    for ev in motions.protect_events:
        if str(ev.get("remedy", "")).startswith("via-above"):
            return "via_above"
    if any(str(ev.get("remedy", "")).startswith("re-plan") for ev in motions.protect_events):
        return "protect_replan"
    if isinstance(dip, dict) and dip.get("fired"):
        return "dip_cartesian"
    if motions.witness_via_posture_events:
        return "via_posture"
    S = motions.straight_arm_stats
    if S.get("straight_arm_via_ok"):
        return "straight_via"
    if S.get("straight_arm_ok"):
        return "straight"
    if (ctx.get("pre_pose_standoff") or {}).get("branch") == "planner":
        return "standoff_planner"
    return "trajopt"


def plan_transit(kit, arm: str, q0: th.Tensor, pre, q_hint: Optional[th.Tensor], supports: Sequence[Any], protected: Sequence[Tuple[str, Any]],
                 approach: Sequence[Any] = (), inflation: Optional[float] = None, straight_only: bool = False) -> Tuple[Optional[th.Tensor], Dict[str, Any]]:
    """The oracle's live transit to the pre-grasp ``pre`` (``execute._grasp_transit``, supports present) planned from ``q0`` (28-D: the
    escape's end at the grasp stance) on the sampler kit's backend: ``protected`` (``[(name, scene object)]``, the book) inflated by
    ``inflation``, ``supports`` (the desk) present as obstacles, ``approach`` the grasp's approach (the dip gate's check), ``q_hint`` the
    pre-grasp witness. Returns ``(trajectory (T, 28) from q0, note)`` - the concatenated streamed segments - or ``(None, note)`` with
    ``note["why"]``. ``note``: ``branch`` (:func:`branch_of`), ``plan_s``, ``segments``, ``protect_events``, ``dip``, ``pre_grasp_transit``.
    ``straight_only`` (i7-startdraw2, the pick transit gate): no trajectory optimisation - the transit the oracle takes WITHOUT planning
    (the checked straight ramp to the witness, or via the ready / untucked posture), or None."""
    from ispatialgym.oracle import execute as E
    from ispatialgym.oracle.motions import SkillError

    t0 = time.time()
    mot = plan_only_motions(kit.robot, kit.backend, q0, straight_only=straight_only)
    ctx: Dict[str, Any] = {}
    note: Dict[str, Any] = {}
    dip_rec: Optional[Dict[str, Any]] = None
    try:
        with mot.protecting(list(protected), leg="pre_grasp", task="sampler", action="grasp", inflation=inflation):
            gate = mot.transit_dip_gate(arm, list(approach or ())) if hasattr(mot, "transit_dip_gate") else nullcontext(None)
            with gate as dip:
                try:
                    _drain(E.reach_pre_pose(mot, None, arm, pre, list(supports), [], q_hint, ctx, E.ExecuteConfig()))
                finally:
                    dip_rec = dict(dip) if dip is not None else None
    except SkillError as e:
        note.update({"why": f"{e.reason}: {e.message}"[:600], "stage": (e.details or {}).get("stage"), "plan_s": round(time.time() - t0, 2),
                     "protect_events": _events(mot), "dip": _short_dip(dip_rec)})
        return None, note
    except Exception as e:  # noqa: BLE001 - the sampler's replica must never crash the attempt: a failed plan is a rejection
        log.warning("overpath planner: the oracle's transit planner raised %r", e)
        note.update({"why": f"error: {e!r}"[:600], "stage": "error", "plan_s": round(time.time() - t0, 2)})
        return None, note
    segs = [th.as_tensor(q0).reshape(1, -1).float().cpu()] + list(mot.segments)
    traj = th.cat(segs, dim=0)
    note.update({"branch": branch_of(mot, ctx, dip_rec), "plan_s": round(time.time() - t0, 2), "segments": [int(len(s)) for s in mot.segments],
                 "protect_events": _events(mot), "dip": _short_dip(dip_rec), "pre_pose_standoff": ctx.get("pre_pose_standoff"),
                 "via_posture": [dict(e) for e in mot.witness_via_posture_events][:2],
                 "straight": {k: v for k, v in mot.straight_arm_stats.items() if k in ("straight_arm_ok", "straight_arm_via_ok", "straight_arm_fallback")}})
    if len(traj) < 2:
        note["why"] = "the planner moved nothing"
        return None, note
    return traj, note


def _events(mot) -> List[Dict[str, Any]]:
    out = []
    for ev in mot.protect_events[:4]:
        objs = ev.get("objects") or []
        out.append({"remedy": str(ev.get("remedy", ""))[:80], "result": str(ev.get("result", ""))[:160],
                    "clearance_m": objs[0].get("clearance_m") if objs else None})
    return out


def _short_dip(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(d, dict):
        return None
    return {k: d.get(k) for k in ("fired", "dip_m", "leg") if k in d}
