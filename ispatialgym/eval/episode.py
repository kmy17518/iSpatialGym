"""Episode driver and result summary for ``ISpatialGymTask`` environments."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ispatialgym import single_env as SE

ISG_INFO_KEY = "ispatialgym"


def isg_info(step_info: Dict[str, Any]) -> Dict[str, Any]:
    """The ``ispatialgym`` block of an ``env.step`` info dict (the task writes it under ``info["done"]``)."""
    if ISG_INFO_KEY in step_info:
        return step_info[ISG_INFO_KEY]
    return step_info["done"][ISG_INFO_KEY]


def summarize_episode(last_info: Dict[str, Any], steps: int, terminated: bool, truncated: bool) -> Dict[str, Any]:
    """Episode result from the last ``env.step`` info dict (``ispatialgym.eval.metrics``).

    Keys:
        ``success`` -- the run mode's geometric success (the BDDL goal plus every ``(:measure ...)`` atom within tolerance at
        the same step(s); ``settle_eval``: through a settled window of ``settle_steps`` steps).
        ``symbolic_success`` / ``symbolic_success_step``, ``geometric_success`` / ``geometric_success_step`` -- the run mode's
        (``*_success_step``: the evaluated step - settle-eval off: the lowest-error candidate in the settle window after the first
        success; settle-eval on: the first settled step;
        ``*_first_step``: the first; 0-based, None when not reached); ``other_mode`` -- the same for the off mode when
        ``settle_eval`` is on, else None.
        ``errors`` / ``error_metric`` / ``measured`` -- at the run mode's evaluated geometric step, else its evaluated symbolic
        step, else the terminal step (``evaluation_step`` / ``evaluation_kind`` / ``evaluation_settled``: which, and its S);
        ``errors_final_step`` -- at the terminal step.
        ``closest_approach`` / ``closest_goal_approach`` -- the lowest-error step the episode ran (any step / a step where the goal
        held): ``step``, ``score``, ``goal``, ``measure_within``, ``error_metric``; None when no step could be scored.
    """
    isg = isg_info(last_info)
    evaluation = isg.get("evaluation") or {}
    return {
        "activity": isg["activity"],
        "activity_definition_id": isg["activity_definition_id"],
        "task_variant": isg["task_variant"],
        "success": bool(isg["official_success"]),
        "settle_eval": isg["settle_eval"],
        "settle_steps": isg["settle_steps"],
        "symbolic_success": bool(isg["symbolic_success"]),
        "symbolic_success_step": isg["symbolic_success_step"],
        "symbolic_first_step": isg.get("symbolic_first_step"),
        "geometric_success": bool(isg["geometric_success"]),
        "geometric_success_step": isg["geometric_success_step"],
        "geometric_first_step": isg.get("geometric_first_step"),
        "evaluation_step": evaluation.get("step"),
        "evaluation_kind": evaluation.get("kind"),
        "evaluation_settled": evaluation.get("settled"),
        "success_final": isg.get("success_final"),
        "closest_approach": isg.get("closest_approach"),
        "closest_goal_approach": isg.get("closest_goal_approach"),
        "other_mode": isg["other_mode"],
        "errors": evaluation.get("errors"),
        "error_metric": evaluation.get("error_metric"),
        "measured": evaluation.get("measured"),
        "errors_final_step": isg["errors"],
        "conditions": isg["conditions"],
        "group_assignments": isg["group_assignments"],
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


#: BEHAVIOR observation convention (evaluation profile ``isg-v1``): one render per control step, so the frames the
#: policy sees lag the physics state by two steps. Demos, goal images and evaluation all use this value; see
#: "Observation timing" in ``docs/task_definition_and_evaluation.md``.
N_RENDER_ITERATIONS = 1


def run_episode(
    env,
    policy: Callable[[Any, Optional[Dict[str, Any]]], Any],
    max_steps: Optional[int] = None,
    reset: bool = True,
    on_step: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    n_render_iterations: int = N_RENDER_ITERATIONS,
) -> Dict[str, Any]:
    """Roll out ``policy(obs, info) -> action`` until the task terminates / truncates (or ``max_steps``).

    Returns :func:`summarize_episode` of the final step. ``on_step(step_index, info)`` is called after every step.
    ``n_render_iterations`` is forwarded to ``env.step`` and defaults to the BEHAVIOR convention
    (:data:`N_RENDER_ITERATIONS`).
    """
    obs = SE.reset(env)[0] if reset else SE.get_obs(env)[0]
    info: Optional[Dict[str, Any]] = None
    steps = 0
    terminated = truncated = False
    while True:
        action = policy(obs, info)
        obs, _, terminated, truncated, info = SE.step(env, action, n_render_iterations=n_render_iterations)
        steps += 1
        if on_step is not None:
            on_step(steps, info)
        if terminated or truncated or (max_steps is not None and steps >= max_steps):
            break
    return summarize_episode(info, steps, terminated, truncated)
