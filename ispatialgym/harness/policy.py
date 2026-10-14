"""run_isg adapter: ``run_isg --mode policy --policy ispatialgym.harness.policy:HarnessPolicy`` (configuration from
``ISG_HARNESS_CONFIG``). run_isg calls ``bind(...)`` once before the stream, the policy once per control step with the
latest observation, and ``finalize(episode)`` at the end. The environment side (instance load, executor, official success,
verdict, recordings) is the same as for any served policy.

Episode ends, as in RoboProbe: the task's official success (its geometric success) ends the stream as a success; the step
budget is the environment's own limit; a planner stop (give_up, done, call budget, capability failure) is scored as a
failure whatever the scene looks like; a provider or harness failure raises :class:`InfrastructureFailure`, which run_isg
records as ``status: error`` and the summary does not score."""

from __future__ import annotations

import logging
import os
import time
import traceback
from dataclasses import asdict
from typing import Any, Dict, Optional

from ispatialgym.harness.config import HarnessConfig
from ispatialgym.harness.core.approver import SafetyAbort, TargetApprover, check_action
from ispatialgym.harness.core.goal import goal_image
from ispatialgym.harness.core.observation import ObservationEncoder
from ispatialgym.harness.core.perturb import Perturbations
from ispatialgym.harness.core.session import MotionSession
from ispatialgym.harness.core.success_check import SUCCESS_CHECK
from ispatialgym.harness.core.trace import EpisodeTrace, source_revision
from ispatialgym.harness.core.types import ARMS, PLANNER_STOPS, InfrastructureFailure, MotionOutcome, MotionRejected, Observation
from ispatialgym.harness.llm import make_client
from ispatialgym.harness.planners import EpisodeContext, make_planner

log = logging.getLogger(__name__)

EPISODE_KEYS = ("success", "reason", "success_metric", "settle_eval", "settle_steps", "symbolic_success", "symbolic_success_step",
                "geometric_success", "geometric_success_step", "error_metric", "other_mode", "n_steps", "n_action_steps")


#: run_isg's budget for an instance without a demo (``ispatialgym.eval.instances.DEFAULT_MAX_STEPS``)
DEFAULT_MAX_STEPS = 3000


def step_budget(meta: Optional[Dict[str, Any]], cfg: HarnessConfig) -> int:
    """The episode's control steps: run_isg's own budget (``meta["budget"]["max_steps"]``: ``--max-steps``, else ``max(300, 2 x``
    a dataset instance's demo length``)``), else the config's, else run_isg's default."""
    budget = (meta or {}).get("budget") or {}
    return int(budget.get("max_steps") or cfg.max_steps or DEFAULT_MAX_STEPS)


def task_info(info) -> Dict[str, Any]:
    """The task's ``ispatialgym`` block of an ``env.step`` info dict (OmniGibson nests it under ``info["done"]``)."""
    info = info or {}
    done = info.get("done") if isinstance(info.get("done"), dict) else None
    return (done or {}).get("ispatialgym") or info.get("ispatialgym") or {}


class HarnessPolicy:
    def __init__(self, cfg: Optional[HarnessConfig] = None):
        self.cfg = cfg or HarnessConfig.from_env()
        self.step = 0
        self.max_steps = step_budget(None, self.cfg)
        self.alias = self.cfg.alias
        self.bound = False
        self.stopped: Optional[Dict[str, Any]] = None
        self.last_outcome: Optional[MotionOutcome] = None
        self.motions_done = 0
        self.meta: Dict[str, Any] = {}
        self.t0 = time.time()

    # -- run_isg hooks --------------------------------------------------------------------------------------------
    def bind(self, executor, loader=None, meta=None, out_dir=None, tro_path=None, **_: Any) -> None:
        from ispatialgym.harness.core.embodiment import R1ProEmbodiment
        from ispatialgym.harness.core.motion import MotionLayer
        from ispatialgym.harness.core.planning import ArmPlanner
        from ispatialgym.harness.core.world import build_world

        cfg = self.cfg
        self.meta = meta if meta is not None else {}
        self.max_steps = step_budget(self.meta, cfg)
        self.emb = R1ProEmbodiment(executor)
        self._init_holds(loader)
        boxes = build_world(self.emb)
        world = [b.to_dict() for b in boxes]
        self.layer = MotionLayer(self.emb, ArmPlanner(self.emb.backend, boxes), cfg.limits, TargetApprover(cfg.limits))
        self.session = MotionSession(self.layer)
        trace_dir = cfg.trace_dir or (os.path.join(out_dir, "harness") if out_dir else None)
        self.trace = EpisodeTrace(trace_dir)
        self.perturb = Perturbations.from_dict(cfg.perturb)
        self.encoder = ObservationEncoder(cfg, self.perturb)
        alias = self.alias = cfg.alias or self.meta.get("variant") or self.meta.get("alias")
        goal_rgb, goal_src = goal_image(tro_path, cfg.goal_image)
        check = SUCCESS_CHECK
        mounting = self.emb.mounting()
        step_limits = {arm: [round(float(v), 5) for v in self.layer.step_limits(arm)] for arm in ARMS}
        self.planner = make_planner(cfg.planner, client=make_client(cfg.model, **cfg.model_options), cfg=cfg, encoder=self.encoder, trace=self.trace)
        self.planner.start(EpisodeContext(success_check=check, goal_rgb=goal_rgb, alias=alias, goal_source=goal_src, mounting=mounting))
        self.meta["harness"] = {"planner": cfg.planner, "model": cfg.model, "success_check": check, "goal_image": goal_src, "trace_dir": trace_dir,
                                "perturb": cfg.perturb, "max_calls": cfg.max_calls, "max_steps": self.max_steps, "code": source_revision(),
                                "planner_world": world}
        self.trace.event("start", config=cfg.to_dict(), alias=alias, success_check=check, goal_image=goal_src, mounting=mounting, max_steps=self.max_steps,
                         arm_step_limits_rad=step_limits, code=source_revision(), robot_state=self.emb.state(0).to_dict(), planner_world=world)
        self.bound = True

    def _init_holds(self, loader) -> None:
        """``Motions``' hold targets := the loader's (the instance's recorded controller targets, the held objects' finger
        commands), so a harness action holds every group it does not move exactly as the load left it."""
        st = self.emb.motions.state
        hold = getattr(loader, "hold", None)
        if hold is not None and getattr(hold, "arm_target", None):
            st.trunk_target = hold.trunk_target.clone()
            st.arm_target = {a: q.clone() for a, q in hold.arm_target.items()}
            st.gripper_cmd = dict(hold.gripper_cmd)
        else:
            st.hold_current()
        st.stop_base()
        for g in getattr(loader, "grasps", None) or []:
            self.emb.motions.gripper_intent_closed[g["arm"]] = True

    def __call__(self, obs, info):
        if not self.bound:
            raise RuntimeError("HarnessPolicy.bind() was not called: run it through run_isg --mode policy")
        if self.stopped is not None:
            return None
        self.step += 1
        if self.cfg.stop_on_success and task_info(info).get("official_success"):
            return self._stop("success")
        if self.step > self.max_steps:
            return self._stop("step_budget")
        for _ in range(self.cfg.max_calls + 50):
            if not self.session.busy and not self._next_motion(obs):
                return None
            a, outcome = self.session.step()
            if a is not None:
                try:
                    check_action(a)
                except SafetyAbort as e:
                    self._infrastructure(f"safety abort: {e}")
                return a
            self.last_outcome = outcome
            self.motions_done += 1
            req = self.session.request
            self.trace.event("motion_end", kind=req.kind if req else None, ok=outcome.ok, steps=outcome.steps, message=outcome.message, details=outcome.details)
        self._infrastructure("no action produced")

    def finalize(self, episode: Optional[Dict[str, Any]] = None, **_: Any) -> None:
        ep = episode if episode is not None else {}
        why = (self.stopped or {}).get("reason")
        forced = why in PLANNER_STOPS
        epilogue_success = ep.get("success")
        if forced and episode is not None:
            episode["success"] = False
        summary = {"stop": self.stopped, "control_steps": self.step, "motions": self.motions_done, "wall_s": round(time.time() - self.t0, 1),
                   "forced_failure": forced, "epilogue_success": epilogue_success, "episode": {k: ep.get(k) for k in EPISODE_KEYS},
                   "planner": self.planner.summary() if self.bound else {}, "model": self.cfg.model, "alias": self.alias}
        if self.bound:
            self.trace.event("finalize", **summary)
            self.trace.summary(summary)
            self.trace.close()
        self.meta.setdefault("harness", {}).update({"stop": self.stopped, "planner": summary["planner"], "forced_failure": forced,
                                                    "epilogue_success": epilogue_success,
                                                    "failure_kind": "capability" if why == "capability_failure" else None})

    # -- internals ------------------------------------------------------------------------------------------------
    def _next_motion(self, obs) -> bool:
        try:
            o = Observation(step=self.step, images=self.emb.images(obs), state=self.emb.state(self.step),
                            steps_remaining=max(0, self.max_steps - self.step))
            d = self.planner.decide(o, self.last_outcome)
            self.last_outcome = None
            while True:
                if d.kind != "motion" or d.request is None:
                    self._stop(d.reason or "stop", **d.details)
                    return False
                req = self.perturb.request(d.request)
                try:
                    plan = self.session.start(req)
                except MotionRejected as e:
                    self.trace.event("motion_rejected", kind=req.kind, params=req.params, note=req.note, reason=str(e), repairable=e.repairable, detail=e.detail)
                    d = self.planner.rejected(str(e), e.repairable)
                    continue
                self.planner.accepted(plan)
                self.trace.event("motion_start", kind=req.kind, params=req.params, note=req.note, perturbed=req is not d.request, plan=asdict(plan))
                return True
        except InfrastructureFailure as e:
            self._infrastructure(str(e))
        except Exception as e:  # noqa: BLE001 - a harness bug is not the model's failure: the episode is not scored
            self._infrastructure(f"unexpected harness failure: {type(e).__name__}: {e}", traceback=traceback.format_exc(limit=8))
        return False

    def _stop(self, why: str, /, **details: Any):
        if self.stopped is None:
            self.stopped = {**details, "reason": why, "step": self.step}
            if self.bound:
                self.trace.event("stop", **self.stopped)
        return None

    def _infrastructure(self, message: str, **details: Any):
        """Record why the episode cannot be scored, then raise (run_isg writes ``status: error``)."""
        self._stop("infrastructure", error=message, **details)
        planner = self.planner.summary() if self.bound else {}
        self.meta.setdefault("harness", {}).update({"stop": self.stopped, "planner": planner, "failure_kind": "infrastructure"})
        if self.bound:
            self.trace.summary({"stop": self.stopped, "control_steps": self.step, "motions": self.motions_done, "planner": planner,
                                "failure_kind": "infrastructure", "model": self.cfg.model, "alias": self.alias})
            self.trace.close()
        raise InfrastructureFailure(message)
