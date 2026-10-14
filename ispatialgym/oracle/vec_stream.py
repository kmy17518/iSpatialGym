"""``VecStreamExecutor``: the executor's stream (``Executor.run_stream`` / ``Executor.epilogue``) for N environments of one
vectorized ``og.Environment``, every environment with its own stream, step counters, QA monitors, recorder and verdict.

One ``env.step`` per control step for the whole batch (BEHAVIOR's ``omnigibson/eval/evaluator.py``): the ``(N, action_dim)``
action holds, per environment, the action its stream produced this step, or - once the stream has ended - the executor's
stream no-op (its **last streamed action with the base stopped**) until the batch ends. Everything that decides an episode is
per environment and reuses the single-env executor's rules unchanged:

* the stream ends at the source's ``None`` (a replay's last row - its demo's action rows and the first ``settle_steps`` of its
  settle rows, ``Slot.phases`` marking those -, a policy's horizon), at ``max_steps``, at the wall clock, at the run mode's
  geometric success (``ExecutorConfig.stop_on_success``) or at that env's monitor timeout; a stream has no epilogue; with
  ``qa=True`` every environment logs the ``--qa`` metrics of a stream (``ispatialgym.oracle.qa.QA``, ``result.qa``; goal
  occlusion is not measured here), as ``Executor.run_stream`` does;
* the verdict and the result dict of ``Executor.epilogue``: the task's metrics (``Executor.metric_verdict``: ``not_reached`` /
  ``not_settled`` / ``symbolic_only``) unless the stream ended on a monitor verdict.

A finished environment's robot keeps receiving its no-op; nothing of it is recorded any more. The per-env ``timings`` are wall
times of the shared loop (every environment advances together).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch as th

from ispatialgym.oracle.executor import Executor, ExecutorConfig, MotionSnapshot, _StreamAction, _hold_actions, _jsonable
from ispatialgym.oracle.qa import QA, QAStats, Verdict
from ispatialgym.oracle.recorder import PHASE_ORACLE, PHASE_SETTLE

ActionSource = Callable[[Any, Any], Optional[Any]]  # (obs, info) -> action | None


def isg_of(info: Any) -> Dict[str, Any]:
    info = info or {}
    return (info.get("done") or {}).get("ispatialgym") or info.get("ispatialgym") or {}


class _Rules:
    """The single-env executor's verdict rules bound to one environment (``Executor.metric_verdict`` and the D-007 branch log read
    the instance's ``cfg``, ``task``, step count ``n`` and ``isg``)."""

    metric_verdict = Executor.metric_verdict
    _reset_predicate_branches = Executor._reset_predicate_branches
    _predicate_branches = Executor._predicate_branches

    def __init__(self, cfg: ExecutorConfig, task):
        self.cfg, self.task = cfg, task
        self.isg: Dict[str, Any] = {}
        self.n = 0


@dataclass
class Slot:
    """One environment of the batch and its episode state."""

    idx: int
    robot: Any
    source: ActionSource
    cfg: ExecutorConfig = field(default_factory=ExecutorConfig)
    task: Any = None
    monitors: Any = None
    recorder: Any = None
    tracked: Dict[str, Any] = field(default_factory=dict)
    held: Callable[[], list] = lambda: []
    #: the recorded phase of every streamed action (a replay: its action rows, then its settle rows); None = all actions
    phases: Optional[List[int]] = None
    # -- state --
    stage: str = "stream"  # stream | done
    n: int = 0
    n_action_steps: int = 0
    phase_steps: Dict[int, int] = field(default_factory=lambda: {PHASE_ORACLE: 0, PHASE_SETTLE: 0})
    last_action: Optional[th.Tensor] = None
    last_obs: Any = None
    last_info: Any = None
    verdict: Optional[Verdict] = None
    stats: QAStats = field(default_factory=QAStats)
    result: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    t_stream0: float = 0.0
    rules: Any = None
    _hold_iter: Any = None
    #: ``Executor.release_check`` of this environment (facts only: the vectorized evaluator has no release verdict)
    release_check: Dict[str, Any] = field(default_factory=lambda: {"releases": [], "violations": []})
    _ag_last: Optional[Dict[str, Optional[str]]] = None
    #: the environment's ``--qa`` metrics (``VecStreamExecutor(qa=True)``)
    qa: Any = None
    wall_clock_exceeded: bool = False

    @property
    def isg(self) -> Dict[str, Any]:
        return isg_of(self.last_info)

    @property
    def done(self) -> bool:
        return self.stage == "done"


class VecStreamExecutor:
    """Streams for N environments stepped together (see the module docstring).

    ``env.step(actions, n_render_iterations)`` must return per-env lists (``obs_list, rewards, terminateds, truncateds,
    infos``); ``sim`` provides ``current_time`` for the recorders (None: not recorded)."""

    def __init__(self, env, slots: List[Slot], sim=None, n_render_iterations: int = 1, num_envs: Optional[int] = None, idle_actions: Optional[Dict[int, th.Tensor]] = None,
                 qa: bool = False):
        """``num_envs`` / ``idle_actions``: the batch may have environments without an episode (a partial last batch); their
        rows of every step's action are ``idle_actions[env_idx]`` (a hold of that robot's joints). ``qa``: log the ``--qa``
        metrics per environment (module docstring)."""
        self.qa = bool(qa)
        self.env = env
        self.slots = list(slots)
        self.sim = sim
        self.n_render_iterations = int(n_render_iterations)
        self.num_envs = int(num_envs) if num_envs is not None else (max(s.idx for s in self.slots) + 1 if self.slots else 1)
        self.idle_actions = dict(idle_actions or {})
        for s in self.slots:
            if s.monitors is not None:
                s.stats = s.monitors.stats
            s.rules = _Rules(s.cfg, s.task)
            if self.qa and s.qa is None:
                s.qa = QA(s.robot, oracle=False, scene=getattr(s.robot, "scene", None),
                          task_objects=[getattr(o, "name", n) for n, o in s.tracked.items() if o is not s.robot],
                          contact_pairs=(lambda s=s: s.monitors._contact_pairs(s.n)) if s.monitors is not None else None,
                          rules=s.monitors is not None,
                          rule_limits=s.monitors.rule_limits() if s.monitors is not None and hasattr(s.monitors, "rule_limits") else None)
        self.steps = 0  # batch steps
        self.action_dim = int(getattr(self.slots[0].robot, "action_dim", 23)) if self.slots else 23

    # ------------------------------------------------------------------ per-slot actions

    def _noop(self, s: Slot) -> th.Tensor:
        """The stream no-op of ``Executor._noop_actions``: the last streamed action with the base stopped (a stream that
        produced no action: the executor's hold of the current joints)."""
        if s.last_action is not None:
            hold = s.last_action.clone()
            idx = getattr(s.robot, "controller_action_idx", {}).get("base")
            if idx is not None:
                hold[idx] = 0.0
            return hold
        if s._hold_iter is None:
            s._hold_iter = _hold_actions(s.robot, 10**9)
        return th.as_tensor(next(s._hold_iter), dtype=th.float32).reshape(-1)

    # ------------------------------------------------------------------ run

    def run(self, obs_list: Iterable[Any], info_list: Iterable[Any]) -> List[Dict[str, Any]]:
        """Run every slot's stream; returns the per-slot result dicts (``Executor.epilogue``'s)."""
        obs_list, info_list = list(obs_list), list(info_list)
        for s in self.slots:
            s.last_obs, s.last_info = obs_list[s.idx], info_list[s.idx]
            s.t_stream0 = time.time()
            s.rules._reset_predicate_branches()
            if s.monitors is not None:
                s.monitors.begin_action(_StreamAction(), 0, 0, intended=(), held=s.held(), manipulation=False, fov=False)
        while any(not s.done for s in self.slots):
            actions = th.zeros((self.num_envs, self.action_dim), dtype=th.float32)
            for i, a in self.idle_actions.items():
                actions[i] = th.as_tensor(a, dtype=th.float32).reshape(-1)
            phases: Dict[int, int] = {}
            for s in self.slots:
                if s.stage == "stream":
                    s.wall_clock_exceeded = bool(s.cfg.max_wall_s) and time.time() - s.t_stream0 > s.cfg.max_wall_s
                    a = None if (s.wall_clock_exceeded or s.n >= s.cfg.max_steps) else s.source(s.last_obs, s.last_info)
                    if a is None:
                        self._end_stream(s, None)
                    else:
                        a = th.as_tensor(a, dtype=th.float32).reshape(-1)
                        s.last_action = a.clone()
                        actions[s.idx] = a
                        k = s.n  # the stream's k-th action
                        phases[s.idx] = int(s.phases[k]) if s.phases is not None and k < len(s.phases) else PHASE_ORACLE
                        continue
                actions[s.idx] = self._noop(s)
            if all(s.done for s in self.slots):
                break
            ag_before = {s.idx: Executor._ag_held(s) for s in self.slots if s.idx in phases}
            obs_list, rewards, terminateds, truncateds, infos = self.env.step(actions, n_render_iterations=self.n_render_iterations)
            self.steps += 1
            sim_time = None
            if self.sim is not None:
                try:
                    sim_time = float(self.sim.current_time)
                except Exception:  # noqa: BLE001
                    pass
            for s in self.slots:
                if s.idx not in phases:
                    continue  # finished: its no-op was stepped, nothing recorded
                phase = phases[s.idx]
                s.n += 1
                ag_after = Executor._ag_held(s)
                n_violations = len(s.release_check["violations"])
                Executor._check_releases(s, ag_before[s.idx], ag_after, actions[s.idx])
                s._ag_last = ag_after
                s.phase_steps[phase] = s.phase_steps.get(phase, 0) + 1
                s.last_obs, s.last_info = obs_list[s.idx], infos[s.idx]
                if s.recorder is not None:
                    try:
                        rew = float(rewards[s.idx])
                    except Exception:  # noqa: BLE001
                        rew = 0.0
                    s.recorder.record(s.n, actions[s.idx], s.last_obs, s.last_info, state=None, phase=phase, action_index=0, attempt=0, reward=rew,
                                      terminated=bool(terminateds[s.idx]), truncated=bool(truncateds[s.idx]), sim_time=sim_time)
                if s.qa is not None:
                    s.qa.observe(s.n, actions[s.idx], acting=phase == PHASE_ORACLE, held=[n for n in ag_after.values() if n],
                                 release_violations=s.release_check["violations"][n_violations:])
                verdict = None
                if s.monitors is not None:
                    verdict = s.monitors.check_timeout_only(s.n, settle=phase != PHASE_ORACLE)
                    pop = getattr(s.monitors, "pop_logged", None)
                    for v in (pop() if pop is not None else ()):
                        if s.qa is not None:
                            s.qa.note_rule(v)
                if verdict is not None:
                    self._end_stream(s, verdict)
                elif s.cfg.stop_on_success and s.isg.get("official_success"):
                    self._end_stream(s, None)
        return [s.result for s in self.slots]

    # ------------------------------------------------------------------ stream end -> verdict

    def _end_stream(self, s: Slot, verdict: Optional[Verdict]) -> None:
        s.verdict = verdict
        if s.monitors is not None:
            s.monitors.end_action()
        if s.recorder is not None:
            s.recorder.commit()
        s.timings["actions_s"] = time.time() - s.t_stream0
        s.n_action_steps = s.n
        if s.monitors is not None:
            s.monitors.begin_epilogue(held=s.held())
        self._finalize(s)

    # ------------------------------------------------------------------ verdict (Executor.epilogue's)

    def _finalize(self, s: Slot) -> None:
        t0 = time.time()
        isg = s.isg
        s.rules.isg, s.rules.n = isg, s.n
        motion = MotionSnapshot(s.robot, s.tracked).delta(MotionSnapshot(s.robot, s.tracked))  # a stream has no window (Executor.epilogue)
        binary_branch, branch_log = s.rules._predicate_branches()
        verdict = s.verdict
        if verdict is None:
            verdict = s.rules.metric_verdict(isg)
            if verdict is not None:
                s.stats.record(verdict)
        if s.qa is not None:
            s.qa.finish(step=s.n)
        s.timings["settle_s"] = time.time() - t0
        s.verdict = verdict
        evaluation = isg.get("evaluation") or {}
        s.result = {
            "success": verdict is None,
            "verdict": None if verdict is None else verdict.to_dict(),
            "reason": None if verdict is None else verdict.category,
            "success_metric": s.cfg.success_metric,
            "settle_eval": isg.get("settle_eval"),
            "settle_steps": isg.get("settle_steps"),
            "symbolic_success": bool(isg.get("symbolic_success")),
            "symbolic_success_step": isg.get("symbolic_success_step"),
            "geometric_success": bool(isg.get("geometric_success")),
            "geometric_success_step": isg.get("geometric_success_step"),
            "errors": evaluation.get("errors") or isg.get("errors") or {},
            "errors_final_step": isg.get("errors") or {},
            "error_metric": evaluation.get("error_metric"),
            "measured": evaluation.get("measured"),
            "other_mode": isg.get("other_mode"),
            "goal_first_step": isg.get("goal_first_step"),
            "binary_branch": binary_branch,
            "binary_branch_fallback": sorted(k for k, v in binary_branch.items() if v not in ("omnigibson",)),
            "binary_branch_log": branch_log,
            "support_disturbance": {},
            "wall_clock_exceeded": bool(s.wall_clock_exceeded),
            "pose_settled_config": isg.get("pose_settled_config"),
            "conditions": isg.get("conditions"),
            "group_assignments": isg.get("group_assignments"),
            "n_steps": s.n,
            "n_action_steps": s.n_action_steps,
            "n_settle_steps": s.phase_steps.get(PHASE_SETTLE, 0),
            "epilogue_steps": 0,
            "epilogue_relax": {},
            "kept_hold_arms": [],
            "jams": [],
            "witness_branch_accepted": [],
            "protection": {"protected": [], "plan_checks": 0, "fired": list(getattr(s.monitors, "protect_events", []) or [])},
            "placed_orientation": None,
            "motion_delta": motion,
            "actions": [],
            "monitors": s.monitors.report() if s.monitors is not None else {"stats": s.stats.to_dict()},
            "qa": s.qa.report() if s.qa is not None else {"enabled": False},
            "checkpoint_reports": [],
            "brace_retry": _jsonable([]),
            "goal_observation": None,
            "witness_via_posture": [],
            "release_check": _jsonable(s.release_check),
            "timings": dict(s.timings),
            "env_idx": s.idx,
            "batch_steps": self.steps,
        }
        s.stage = "done"


__all__ = ["VecStreamExecutor", "Slot", "isg_of"]
